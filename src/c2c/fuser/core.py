"""Fused-cache update — paper §3.3.1, Eqs. (3)–(4).

    C_f[n] = C_n(X) + F_n( C_n(X), C^S_{G(n)}(X) )                (3)
    y_{i+1} = P( y_i | C_f(X) ⊕ C(Y_{[0:i]}) )                     (4)

The residual in (3) is the whole point of the architecture: the fusion is
**never a destructive overwrite** of the receiver's own cache (FR-03). The
ablation floor published in Table 8 — pure projection collapses average
accuracy 20.70, fusion +24.18, gating +3.07 — is reproduced by construction
through the ``residual`` and ``gating`` switches, and asserted by the
golden suite.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor, nn

from ..config import BlendConfig, FuserConfig, GateConfig
from ..types import BlendDirection, FusionReport, LayeredCache, LayerGeometry, LayerSlice
from .blend import apply as apply_blend
from .complex import PreProjection
from .modules import DynamicWeighting, Gate, Projection

__all__ = ["FUSER_VARIANTS", "FuserPair", "Fuser"]

FUSER_VARIANTS = ("simple", "c2c-c")


def _rows(slice_: LayerSlice) -> Tensor:
    """The per-token feature vector of one layer: [n, d] = [key ‖ value].

    The paper flattens key and value into a single per-token vector; the
    joint view is taken here, and every slice of the result is returned to
    the cache on the way out.
    """
    if slice_.key.dim() == 3:  # heads view → flatten
        k = slice_.key.flatten(start_dim=1)
        v = slice_.value.flatten(start_dim=1)
    else:
        k, v = slice_.key, slice_.value
    return torch.cat((k, v), dim=-1)


def _slice(rows: Tensor, geometry: LayerGeometry) -> LayerSlice:
    """Split an [n, d] joint view back into key/value cache slices."""
    kv = geometry.kv_hidden_size
    key, value = rows.split(kv, dim=-1)
    kv_heads = geometry.kv_heads
    if kv_heads > 1 and key.dim() == 2 and key.shape[-1] % (kv_heads * geometry.head_size) == 0:
        n = key.shape[0]
        key = key.reshape(n, kv_heads, geometry.head_size)
        value = value.reshape(n, kv_heads, geometry.head_size)
    return LayerSlice(key, value)


def _gather(rows: Tensor, token_mapping: Sequence[int] | None) -> Tensor:
    """Select the aligned sharer rows for each receiver token (FR-10 result).

    ``token_mapping`` gives, for every receiver row, the index of the
    sharer row that carries the same string; identity when both models
    tokenised the prompt identically (the common case — tokenizers of the
    same family agree on >80 % of alignments, per the published figure).
    """
    if token_mapping is None:
        return rows
    idx = torch.as_tensor(list(token_mapping), dtype=torch.long, device=rows.device)
    return rows.index_select(0, idx)


class FuserPair(nn.Module):
    """One fuser clone, mapped to one (receiver layer n, sharer layer G(n)) pair.

    Pipeline (Fig. 5): concat → [pre-projection (C2C-C only)] → projection
    → dynamic weighting → feature-fusion residual add, scaled by the pair
    gate g_n. Both the key and the value half of the joint vector pass
    through the same pair module, in parallel, sharing the gate value.
    """

    def __init__(
        self, receiver: LayerGeometry, sharer: LayerGeometry, config: FuserConfig | None = None
    ):
        super().__init__()
        self.receiver = receiver
        self.sharer = sharer
        self.config = config or FuserConfig()
        d_r = receiver.d  # joint dimension of one token
        d_s = sharer.d
        self.pre = (
            PreProjection(
                d_s,
                d_r,
                layers=self.config.pre_projection_layers,
                activation=self.config.activation,
            )
            if self.config.variant == "c2c-c"
            else None
        )
        d_s_eff = d_r if self.config.variant == "c2c-c" else d_s
        if self.config.latent_size not in (None, d_r):
            msg = (
                "latent_size is honoured only when it equals the receiver "
                f"dimension {d_r}: the residual of Eq. (3) requires it; "
                f"got {self.config.latent_size}"
            )
            raise ValueError(msg)
        self.kv_heads = receiver.kv_heads
        self.head_size = receiver.head_size
        self.projection = Projection(d_r, d_s_eff, d_model=d_r, activation=self.config.activation)
        self.weighting = DynamicWeighting(self.kv_heads, self.head_size, halves=2)
        self.out_features = self.projection.projection.out_features

    def forward(self, r_rows: Tensor, s_rows: Tensor) -> Tensor:
        """Produce the fuser delta F_n for one mapped pair, shaped [n, d_r]."""
        if self.pre is not None:
            s_rows = self.pre(s_rows)
        x = torch.cat((r_rows, s_rows), dim=-1)
        delta = self.projection(x)  # [n, d_r] = [key ‖ value]
        n = delta.shape[0]
        halves_view = delta.reshape(n, 2, self.kv_heads, self.head_size)
        modulated = self.weighting(halves_view)  # input-aware head modulation
        return modulated.reshape(n, -1)


class Fuser(nn.Module):
    """The cache fuser: all pair clones plus the shared per-layer gates.

    Parameters
    ----------
    receiver, sharer:
        Layer geometries of the two models (model cards). All combinations
        of differing head counts / head sizes / layer counts are handled —
        the projection layer takes care of the dimensionality negotiation
        (FR-12), exactly as the paper's Fig. 5 prescribes.
    mapping:
        ``G(n)``: receiver layer index → sharer layer index (c2c.align).
    residual / gating:
        The Table 8 ablation switches. ``residual=False`` discards the
        receiver cache (pure projection — the floor), ``gating=False``
        disables the learnable gates (+Fuse level). Defaults reproduce the
        published C2C (both on); a build that cannot reproduce the
        ordering floor < fuse < gate in `c2c.eval` fails review.
    """

    def __init__(
        self,
        receiver: LayerGeometry,
        sharer: LayerGeometry,
        mapping: Sequence[int],
        *,
        fuser_config: FuserConfig | None = None,
        gate_config: GateConfig | None = None,
        blend_config: BlendConfig | None = None,
    ):
        super().__init__()
        self.receiver = receiver
        self.sharer = sharer
        self.mapping = list(mapping)
        self.fuser_config = fuser_config or FuserConfig()
        self.gate_config = gate_config or GateConfig()
        self.blend_config = blend_config or BlendConfig()
        if self.fuser_config.variant not in FUSER_VARIANTS:
            msg = f"unknown fuser variant {self.fuser_config.variant!r}; choose {FUSER_VARIANTS}"
            raise ValueError(msg)
        if not self.mapping:
            msg = "empty layer mapping: nothing to fuse"
            raise ValueError(msg)
        self.pairs = nn.ModuleList(
            [FuserPair(receiver, sharer, self.fuser_config) for _ in self.mapping]
        )
        self.gate = Gate(
            len(self.mapping),
            tau_max=self.gate_config.tau_max,
            tau_min=self.gate_config.tau_min,
            threshold=self.gate_config.threshold,
            straight_through=self.gate_config.straight_through,
        )

    # -- the fuse pipeline proper ───────────────────────────────────────────
    def forward(
        self,
        receiver_cache: LayeredCache,
        sharer_cache: LayeredCache,
        *,
        token_mapping: Sequence[int] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> LayeredCache:
        """Fuse both caches into one (Eq. 3). Returns a *new* LayeredCache.

        ``token_mapping``: aligned sharer row index per receiver row, as
        produced by :class:`c2c.align.TokenAligner` (FR-10). ``None`` means
        both models agree on the tokenisation (identity alignment).
        """
        if len(receiver_cache) < len(self.mapping):
            msg = (
                f"receiver cache has {len(receiver_cache)} layers, "
                f"mapping expects {len(self.mapping)}"
            )
            raise ValueError(msg)
        if len(sharer_cache) <= max(self.mapping):
            msg = (
                f"sharer cache has {len(sharer_cache)} layers, "
                f"mapping indexes layer {max(self.mapping)}"
            )
            raise IndexError(msg)
        if self.fuser_config.gating:
            weights = self.gate(training=self.training, step=step, total_steps=total_steps)
        else:
            weights = torch.ones(self.gate.n_mapped, device=self.gate.logits.device)
        fused_slices: list[LayerSlice] = []
        for n, g in enumerate(self.mapping):
            r = receiver_cache[n]
            s = sharer_cache[g]
            r_rows = _rows(r)
            s_rows = _gather(_rows(s), token_mapping)
            if r_rows.shape[0] != s_rows.shape[0]:
                msg = (
                    f"token misalignment at layer pair ({n}, {g}): "
                    f"{r_rows.shape[0]} receiver rows vs {s_rows.shape[0]} sharer rows; "
                    "pass token_mapping=... from c2c.align.TokenAligner"
                )
                raise ValueError(msg)
            delta = self.pairs[n](r_rows, s_rows)
            w = weights[n].to(delta.dtype) if weights.dim() else weights
            contribution = delta * w
            if self.fuser_config.residual:
                rows = r_rows + contribution  # Eq. (3): residual, never destructive
            else:
                rows = contribution  # Table 8 floor: projection only
            fused_slices.append(_slice(rows, self.receiver))
        fused = LayeredCache(fused_slices)
        if self.blend_config.fraction < 1.0:  # App. A.2.4 progressive blend
            fused = apply_blend(
                receiver_cache,
                fused,
                fraction=self.blend_config.fraction,
                direction=BlendDirection(self.blend_config.direction),
            )
        return fused

    # -- reports ────────────────────────────────────────────────────────────
    def report(
        self, *, num_tokens: int = 0, effective_rank: dict[str, dict[str, float]] | None = None
    ) -> FusionReport:
        """Summarise the last fusion for `c2c fuse --report` (FR-17 inputs)."""
        probs = self.gate.probabilities()
        return FusionReport(
            geometry=self.receiver,
            num_tokens=num_tokens,
            gate_values=probs,
            gate_open_ratio=self.gate.activation_ratio(),
            fused_fraction=self.blend_config.fraction,
            blend_direction=BlendDirection(self.blend_config.direction),
            effective_rank=effective_rank or {},
            notes=[
                f"variant={self.fuser_config.variant}",
                f"residual={self.fuser_config.residual}",
                f"gating={'on' if self.fuser_config.gating else 'off'}",
                f"layers mapped: {len(self.mapping)}",
            ],
        )

    def paired(self) -> Iterator[tuple[int, int]]:
        """Iterate over the mapped pairs (n, G(n)), receiver-major order."""
        return ((n, g) for n, g in enumerate(self.mapping))

    # -- Eq. (4) support ────────────────────────────────────────────────────
    @staticmethod
    def prefix_plus_suffix(fused_prefill: LayeredCache, generated: LayeredCache) -> LayeredCache:
        """⊕ in Eq. (4): ``C_f(X) ⊕ C(Y_[0:i])`` — join the prefill cache with
        the cache of everything generated so far, layer by layer."""
        return fused_prefill.concat(generated)
