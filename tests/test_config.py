"""Tests for the configuration tree and the environment (FR-14).

Every default is checked against the paper's published recipe
(App. A.3.5): what is not explicitly permitted in the data classes is
forbidden in the defaults. ``man c2c.config`` documents all of them.
"""

from __future__ import annotations

import pytest

from c2c.config import (AlignConfig, BlendConfig, C2CConfig, DEFAULT_SEED,
                        FuserConfig, GateConfig, MAN_C2C_CONFIG, PrivacyConfig,
                        ServeConfig, TrainRecipe, from_env, load_config)


class TestRecipeDefaults:
    """The recipe, verbatim: 500k records, 2048 max seq, batch 256, seed 42."""

    def test_the_published_recipe(self):
        r = TrainRecipe()
        assert r.num_samples == 500_000
        assert r.max_seq_length == 2048
        assert r.epochs == 1
        assert r.macro_batch_size == 256
        assert r.learning_rate == 1e-4
        assert r.warmup_ratio == pytest.approx(0.10)
        assert r.weight_decay == pytest.approx(0.01)
        assert r.max_grad_norm == pytest.approx(1.0)
        assert r.seed == DEFAULT_SEED == 42
        assert r.total_steps == 1929
        assert r.gpu_hours_budget == pytest.approx(9.0)
        assert "OpenHermes-2.5" in r.dataset

    def test_warmup_steps_follow_the_schedule(self):
        assert TrainRecipe().warmup_steps() == int(1929 * 0.10)

    def test_the_evaluation_protocol(self):
        s = ServeConfig()
        assert s.max_new_tokens == 64              # paper: max response 64
        assert s.communication_tokens_budget == 256   # paper: communication 256


class TestGateSchedule:
    """Gumbel-Sigmoid temperature, linearly annealed from 1.0 to 0.001."""

    def test_the_temperature_anneals_linearly(self):
        g = GateConfig()
        assert g.temperature_at(0, 1000) == pytest.approx(1.0)
        assert g.temperature_at(1000, 1000) == pytest.approx(0.001)
        assert g.temperature_at(500, 1000) == pytest.approx((1.0 + 0.001) / 2, abs=1e-6)

    def test_the_schedule_is_monotonic(self):
        g = GateConfig()
        curve = [g.temperature_at(s, 1000) for s in range(1001)]
        assert all(a >= b for a, b in zip(curve, curve[1:]))
        assert all(0.001 <= t <= 1.0 for t in curve)

    def test_the_defaults_are_the_published_ones(self):
        g = GateConfig()
        assert (g.tau_max, g.tau_min, g.threshold) == (1.0, 0.001, 0.5)
        assert g.straight_through is True          # ST-G estimator, as in §3.3.2


class TestBlendPolicy:
    """Fraction 0..1, percent 0..100 tolerated; the direction is former or latter."""

    def test_percent_is_tolerated(self):
        assert BlendConfig(fraction=75).fraction == pytest.approx(0.75)

    def test_out_of_range_is_clamped(self):
        assert BlendConfig(fraction=130).fraction == pytest.approx(1.0)
        assert BlendConfig(fraction=-1).fraction == pytest.approx(0.0)

    def test_illegal_direction_raises(self):
        with pytest.raises(ValueError, match="former.*latter"):
            BlendConfig(direction="sideways")


class TestFuserVariants:
    """variant: simple | c2c-c; the ablation switches default to the paper."""

    def test_the_two_variants(self):
        assert FuserConfig(variant="simple").variant == "simple"
        assert FuserConfig(variant="c2c-c").variant == "c2c-c"

    def test_illegal_variant_raises(self):
        with pytest.raises(ValueError, match="fuser variant"):
            FuserConfig(variant="c2c-fortran")

    def test_ablation_switches_are_on_by_default(self):
        f = FuserConfig()                       # Table 8: +Fuse and +Gate, full C2C
        assert f.residual is True and f.gating is True

    def test_pre_projection_layers_must_be_positive(self):
        with pytest.raises(ValueError, match="positive integer"):
            FuserConfig(pre_projection_layers=0)
        assert FuserConfig().pre_projection_layers == 3    # C2C-C: the 3-layer MLP


class TestAlignStrategy:
    def test_collision_strategies_are_the_two_documented(self):
        assert AlignConfig(token_collision="first-occurrence").token_collision \
            == "first-occurrence"
        with pytest.raises(ValueError, match="token_collision"):
            AlignConfig(token_collision="random")

    def test_layer_modes_are_the_two_documented(self):
        assert AlignConfig().layers == "terminal"           # the published default
        assert AlignConfig(layers="depth-normalized").layers == "depth-normalized"
        with pytest.raises(ValueError, match="depth-normalized"):
            AlignConfig(layers="random")


class TestModelIds:
    """The pair from the model id: c2c/<receiver>←<sharer>, ASCII separators too."""

    @pytest.mark.parametrize("model", [
        "c2c/qwen3-0.6b←qwen2.5-0.5b",
        "c2c/qwen3-0.6b->qwen2.5-0.5b",
        "c2c/qwen3-0.6b-->qwen2.5-0.5b",
        "c2c/qwen3-0.6b→qwen2.5-0.5b",
        "c2c/qwen3-0.6b--qwen2.5-0.5b",
        "c2c/qwen3-0.6b:qwen2.5-0.5b",
        "qwen3-0.6b←qwen2.5-0.5b",                  # the prefix is optional
    ])
    def test_the_separator_is_the_separator(self, model):
        pair = ServeConfig().pair_from_model(model)
        assert pair == ("qwen3-0.6b", "qwen2.5-0.5b")

    def test_a_single_model_is_no_pair(self):
        assert ServeConfig().pair_from_model("qwen3-0.6b") is None
        assert ServeConfig().pair_from_model("") is None


class TestEnvironment:
    """Every environment variable documented in the man page has an effect."""

    def test_the_environment_overrides_the_defaults(self):
        cfg = from_env(C2CConfig(), environ={
            "C2C_SEED": "7",
            "C2C_GATE_TAU_MIN": "0.5",
            "C2C_BLEND_FRACTION": "50",
            "C2C_ALIGN_LAYERS": "depth-normalized",
            "C2C_LR": "1e-3",
            "C2C_PORT": "9000",
            "C2C_PRIVACY": "on",
            "C2C_DEVICE": "cpu",
        })
        assert cfg.seed == 7                               # C2C_SEED, top level
        assert cfg.gate.tau_min == 0.5
        assert cfg.blend.fraction == pytest.approx(0.5)
        assert cfg.align.layers == "depth-normalized"
        assert cfg.train.learning_rate == 1e-3
        assert cfg.serve.port == 9000
        assert cfg.privacy.enabled is True

    def test_defaults_survive_an_empty_environment(self):
        assert from_env(C2CConfig(), environ={}) == C2CConfig()

    def test_the_json_overlay_then_the_environment(self, tmp_path):
        import json
        conf = tmp_path / "config.json"
        conf.write_text(json.dumps({"gate": {"tau_max": 0.5},
                                 "unknown_key": "is ignored"}), encoding="utf-8")
        cfg = load_config(str(conf), environ={"C2C_GATE_TAU_MAX": "0.25"})
        assert cfg.gate.tau_max == 0.25                  # environment wins, then file


class TestManPages:
    """``man c2c.config`` — the two heads, documented (FR-06)."""

    def test_man_page_documents_both_meanings_of_head(self):
        text = MAN_C2C_CONFIG
        assert "MODEL HEADS" in text                      # attention heads, FR-06
        assert "CLI HEADS" in text                        # the heads of the console
        assert "run-agent" in text                        # only to deny it exists

    def test_man_page_lists_the_environment(self):
        for var in ("C2C_SEED", "C2C_GATE_TAU_MAX", "C2C_GATE_TAU_MIN",
                  "C2C_BLEND_FRACTION", "C2C_ALIGN_LAYERS", "C2C_LR", "C2C_PORT"):
            assert var in MAN_C2C_CONFIG, f"{var} missing from the man page"

    def test_man_page_describes_the_synopsis(self):
        for section in ("NAME", "DESCRIPTION", "SYNOPSIS", "FILES", "SEE ALSO"):
            assert section in MAN_C2C_CONFIG, f"section {section} missing"
