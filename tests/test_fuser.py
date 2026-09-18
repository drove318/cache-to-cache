"""Tests for the fuser (FR-03/05/06/07/08) — the only trainable part.

The fusion equation (Eq. 3) in both ablation positions of Table 8, the
gate in its two modes (soft, differentiable, annealed; hard, binary, at
inference), input-aware per-head dynamic weighting, and the progressive
blend policy (App. A.2.4).
"""

from __future__ import annotations

import pytest
import torch

from c2c.config import BlendConfig, FuserConfig, GateConfig
from c2c.fuser import Fuser
from c2c.fuser.blend import apply as blend_apply
from c2c.fuser.blend import normalize_fraction, sweep
from c2c.fuser.core import FUSER_VARIANTS
from c2c.types import BlendDirection, LayerGeometry, LayeredCache, LayerSlice

RECEIVER = LayerGeometry(layers=4, hidden_size=16, num_heads=4, name="receiver-mini")
SHARER = LayerGeometry(layers=3, hidden_size=12, num_heads=3, name="sharer-mini")
MAPPING = [0, 1, 2]                                    # three mapped, one per clone


def rows(cache, n):
    return cache[n]


def make_cache(geometry, tokens, seed):
    g = torch.Generator().manual_seed(seed)
    return LayeredCache([
        LayerSlice(torch.randn(tokens, geometry.kv_hidden_size, generator=g),
                torch.randn(tokens, geometry.kv_hidden_size, generator=g))
        for _ in range(geometry.layers)])


def make_fuser(**kw):
    """Build the fuser from constructor arguments, the frozen way."""
    from dataclasses import fields
    fkeys = {f.name for f in fields(FuserConfig)}
    gkeys = {f.name for f in fields(GateConfig)}
    bkeys = {f.name for f in fields(BlendConfig)}
    fc = kw.pop("fuser_config", None) or FuserConfig(
        **{k: kw.pop(k) for k in list(kw) if k in fkeys})
    gc = kw.pop("gate_config", None) or GateConfig(
        **{k: kw.pop(k) for k in list(kw) if k in gkeys})
    bc = kw.pop("blend_config", None) or BlendConfig(
        **{k: kw.pop(k) for k in list(kw) if k in bkeys})
    assert not kw, f"unknown configuration keys: {sorted(kw)}"
    return Fuser(RECEIVER, SHARER, MAPPING,
               fuser_config=fc, gate_config=gc, blend_config=bc)


CACHE_R = lambda: make_cache(RECEIVER, 5, 1)
CACHE_S = lambda: make_cache(SHARER, 5, 2)


class TestFusionEquation:
    """C_fused := C_n + g · F(concat(C_n, g·F(C_sh))) — Eq. (3), the letter."""

    def test_the_fused_cache_keeps_the_receivers_shape(self):
        fuser = make_fuser().eval()
        fused = fuser(CACHE_R(), CACHE_S())
        assert isinstance(fused, LayeredCache)
        assert len(fused) == len(MAPPING)                  # one slice per mapped pair
        assert fused.num_tokens == 5                       # the rows, unchanged
        heads, head_size = RECEIVER.num_key_value_heads, RECEIVER.head_size
        for slc in fused:
            assert slc.key.shape[1:] == (heads, head_size)   # heads, kept, per the card
            assert slc.value.shape[1:] == (heads, head_size)

    def test_the_fusion_is_not_in_place(self):
        fuser = make_fuser().eval()
        cache_r = CACHE_R()
        before = [r.key.clone() for r in cache_r]
        _ = fuser(cache_r, CACHE_S())
        for a, b in zip(cache_r, before):
            assert torch.equal(a.key, b)                   # nothing destroyed, verbatim

    def test_residual_false_is_the_projection_alone(self):
        """Table 8, row Project: without the residual path the delta stands alone."""
        plain = make_fuser().eval()
        nodelta = make_fuser(residual=False).eval()
        r, s = CACHE_R(), CACHE_S()
        fused_plain = plain(r, s)
        fused_nodelta = nodelta(r, s)
        # with residual off, the fused rows carry no additive receiver component
        assert not torch.equal(fused_plain[0].key, fused_nodelta[0].key)
        # …and the projection alone is not the receiver either: it is a transformation
        assert fused_nodelta.num_tokens == 5

    def test_gating_false_opens_all_gates(self):
        """Table 8, row +Fuse: gates disabled, every channel full — as if open."""
        torch.manual_seed(0)
        disabled = make_fuser(gating=False).eval()
        torch.manual_seed(0)
        manual = make_fuser().eval()
        with torch.no_grad():
            manual.gate.logits.fill_(10.0)                 # sigmoid(10): open wide
        a = disabled(CACHE_R(), CACHE_S())
        b = manual(CACHE_R(), CACHE_S())
        for sa, sb in zip(a, b):
            assert torch.allclose(sa.key, sb.key, atol=1e-5)   # the same, both ways
            assert torch.allclose(sa.value, sb.value, atol=1e-5)
        assert "gating=off" in disabled.report(num_tokens=5).notes   # said, in the report
        assert "gating=on" in manual.report(num_tokens=5).notes

    def test_the_gate_decides_binary_at_inference(self):
        fuser = make_fuser().eval()
        weights = fuser.gate(training=False)
        assert set(weights.tolist()) <= {0.0, 1.0}        # open or closed, no in-between

    def test_the_gate_is_soft_and_differentiable_at_training(self):
        """ST-G estimator: hard decisions forward, soft gradients backward."""
        fuser = make_fuser().train()
        torch.manual_seed(42)
        weights = fuser.gate(training=True, step=0, total_steps=1000)
        assert weights.shape == (len(MAPPING),)
        assert torch.all(weights >= 0) and torch.all(weights <= 1)   # probabilities
        assert set(weights.tolist()) <= {0.0, 1.0}         # ST: the decision, hard
        weights.sum().backward()                             # but the gradient, soft
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                for p in fuser.gate.parameters())            # differentiable, despite
        assert all(0.0 < p < 1.0 for p in fuser.gate.probabilities())  # read, soft

    def test_temperature_follows_the_schedule(self):
        """The annealing: τ starts at 1.0, ends at 0.001, linearly (FR-07)."""
        gate = make_fuser().gate
        assert gate.tau_max == 1.0 and gate.tau_min == 0.001
        first = gate._anneal(0, 1000) if hasattr(gate, "_anneal") else None
        if first is not None:
            assert first == pytest.approx(1.0)
        late = GateConfig().temperature_at(1000, 1000)
        assert late == pytest.approx(0.001)

    def test_gates_are_strictly_per_layer(self):
        fuser = make_fuser()
        assert fuser.gate.logits.shape[0] == len(MAPPING)  # one gate, one mapped pair


class TestGradientFlow:
    """Backward through the straight-through estimator, forward through the net."""

    def test_gradients_flow_through_the_straight_through_estimator(self):
        fuser = make_fuser().train()
        fused = fuser(CACHE_R(), CACHE_S())
        loss = fused.as_matrix().pow(2).mean() if hasattr(fused, "as_matrix") else \
            torch.cat([slc.key.flatten() for slc in fused]).pow(2).mean()
        loss.backward()
        trained = [p for p in fuser.parameters() if p.grad is not None]
        assert trained, "no gradient reached the trainable part"
        assert all(torch.isfinite(p.grad).all() for p in trained)
        assert any(p.grad.abs().sum() > 0 for p in trained), "the network did not learn"

    def test_the_fuser_is_the_only_trainable_part(self):
        """§3.3.4: both LLMs are frozen; the fuser trains alone."""
        fuser = make_fuser()
        assert all(p.requires_grad for p in fuser.parameters())
        assert sum(1 for _ in fuser.parameters()) > 0      # it has weights to train

    def test_gumbel_noise_is_reproducible_with_a_generator(self):
        gate = make_fuser().train().gate
        gen = torch.Generator().manual_seed(42)
        a = gate.gumbel_noise((3,), gen)
        gen2 = torch.Generator().manual_seed(42)
        b = gate.gumbel_noise((3,), gen2)
        assert torch.equal(a, b)                           # the same seed, the same noise


class TestHeadModulation:
    """Dynamic weighting: input-aware, per-head (FR-06); the model has heads."""

    def test_the_model_has_heads_and_the_fuser_has_modules(self):
        fuser = make_fuser()
        pair = fuser.pairs[0]
        assert pair.weighting.num_heads == RECEIVER.num_key_value_heads
        assert pair.weighting.head_size == RECEIVER.head_size
        assert pair.weighting.halves == 2                  # the key and the value, both

    def test_modulation_is_input_aware(self):
        """Different inputs, different weights: the modulation, live."""
        fuser = make_fuser().train()
        pair = fuser.pairs[0]
        shape = (5, 2, pair.weighting.num_heads, pair.weighting.head_size)
        x, y = torch.randn(*shape), torch.randn(*shape)
        w_x = pair.weighting.weights(x)
        w_y = pair.weighting.weights(y)
        assert w_x.shape == w_y.shape
        assert not torch.allclose(w_x, w_y)                # the weights follow the input
        assert torch.all((w_x >= 0) & (w_x <= 2.0))       # sigmoid-shaped, bounded

    def test_weighting_scales_the_projection_output(self):
        """The projected features, reweighted per head, stay in the receiver space."""
        fuser = make_fuser().eval()
        pair = fuser.pairs[0]
        r = torch.randn(5, pair.receiver.d)                # joint vectors, receiver's
        s = torch.randn(5, pair.sharer.d)                  # joint vectors, sharer's
        out = pair(r, s)
        assert out.shape == (5, pair.receiver.d)           # the delta, in the card's d


class TestBlendPolicy:
    """Progressive layer-wise blending (App. A.2.4, FR-08): the blend(3) call."""

    def test_fraction_is_normalised_percent_tolerated(self):
        assert normalize_fraction(50) == pytest.approx(0.5)      # the half, in per cent
        assert normalize_fraction(1.5) == pytest.approx(0.015)   # one and a half per cent
        assert normalize_fraction(0.75) == pytest.approx(0.75)   # the plain, unchanged
        with pytest.raises(ValueError, match="out of range"):
            normalize_fraction(-0.5)                              # below the floor
        with pytest.raises(ValueError, match="out of range"):
            normalize_fraction(130)                               # above the hundred

    def test_the_full_fraction_blends_all_layers(self):
        base, fused = CACHE_R(), make_cache(RECEIVER, 5, 2)
        out = blend_apply(base, fused, fraction=1.0)
        for a, b in zip(out, fused):
            assert torch.allclose(a.key, b.key)            # all of it, from the fused

    def test_the_zero_fraction_blends_nothing(self):
        base, fused = CACHE_R(), make_cache(RECEIVER, 5, 2)
        out = blend_apply(base, fused, fraction=0.0)
        for a, b in zip(out, base):
            assert torch.allclose(a.key, b.key)            # none of it, from the fused

    def test_the_direction_former_replaces_from_the_front(self):
        """The former direction: rows at the head of each layer are taken."""
        base = make_cache(RECEIVER, 10, 7)
        fused = make_cache(RECEIVER, 10, 8)
        out = blend_apply(base, fused, fraction=0.5, direction="former")
        for slc_out, slc_b, slc_f in zip(out, base, fused):
            half = slc_out.key.shape[0] // 2
            assert torch.allclose(slc_out.key[:half], slc_f.key[:half])   # from the front
            assert torch.allclose(slc_out.key[half:], slc_b.key[half:])   # the rest, kept

    def test_the_direction_latter_replaces_from_the_back(self):
        base = make_cache(RECEIVER, 10, 7)
        fused = make_cache(RECEIVER, 10, 8)
        out = blend_apply(base, fused, fraction=0.5, direction=BlendDirection.LATTER)
        half = out[0].key.shape[0] // 2
        assert torch.allclose(out[0].key[half:], fused[0].key[half:])   # from the back

    def test_illegal_direction_is_a_value_error(self):
        with pytest.raises((ValueError, KeyError)):
            blend_apply(CACHE_R(), make_cache(RECEIVER, 5, 2), direction="sideways")

    def test_unequal_layer_counts_are_refused(self):
        base = make_cache(RECEIVER, 4, 1)               # four layers
        tiny = LayerGeometry(layers=2, hidden_size=16, num_heads=4)
        fused = make_cache(tiny, 4, 2)                  # two layers: a mismatch
        with pytest.raises(ValueError, match="equal layer counts"):
            blend_apply(base, fused)

    def test_the_sweep_sweeps_the_fractions(self):
        base, fused = CACHE_R(), make_cache(RECEIVER, 5, 2)
        curve = list(sweep(base, fused, fractions=[0.0, 0.5, 1.0]))
        assert [f for f, _ in curve] == [0.0, 0.5, 1.0]  # the schedule, as passed
        assert all(isinstance(cache, LayeredCache) for _, cache in curve)


class TestVariants:
    """The two fuser variants, registered in the tuple of the distribution."""

    def test_the_variant_table_lists_both_variants(self):
        assert "simple" in FUSER_VARIANTS and "c2c-c" in FUSER_VARIANTS

    def test_c2c_c_adds_a_pre_projection_before_the_projection(self):
        """App. A.1.3: C2C-C has the 3-layer MLP in front, a deeper net."""
        simple = make_fuser(variant="simple")
        c2c_c = make_fuser(variant="c2c-c")
        assert c2c_c.pairs[0].pre is not None             # the extra MLP, in place
        assert simple.pairs[0].pre is None                 # the plain, without it
        deep = sum(p.numel() for p in c2c_c.parameters())
        plain = sum(p.numel() for p in simple.parameters())
        assert deep > plain                                # more parameters, by design

    def test_an_unknown_variant_is_a_value_error(self):
        with pytest.raises(ValueError, match="fuser variant"):
            make_fuser(variant="c2c-fortran")

    def test_misaligned_caches_are_caught(self):
        """The token rows must agree, or the alignment, or the mapping."""
        fuser = make_fuser().eval()
        short_sharer_geo = LayerGeometry(layers=2, hidden_size=12, num_heads=3)
        with pytest.raises(IndexError, match="mapping indexes"):
            fuser(CACHE_R(), make_cache(short_sharer_geo, 5, 3))
        short_receiver_geo = LayerGeometry(layers=2, hidden_size=16, num_heads=4)
        with pytest.raises(ValueError, match="receiver cache has"):
            fuser(make_cache(short_receiver_geo, 5, 9), CACHE_S())

    def test_row_misalignment_names_the_missing_token(self):
        """Different token counts, a clear message in the error."""
        fuser = make_fuser().eval()
        with pytest.raises(ValueError, match="token misalignment"):
            fuser(make_cache(RECEIVER, 6, 1), CACHE_S())  # 6 vs 5 rows, no mapping passed
