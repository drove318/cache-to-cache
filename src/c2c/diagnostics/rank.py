"""Effective rank of caches — Roy & Vetterli (2007); paper Table 2, Fig. 12.

The *effective rank* measures the intrinsic dimensionality of a matrix from
its singular values: richer semantics ⇒ larger intrinsic dimensionality::

    erank(A) = exp( - Σ pᵢ · ln pᵢ ),   pᵢ = σᵢ²(A) / Σ σⱼ²(A)

(The published formula, Roy & Vetterli, *The effective rank: a measure of
effective dimensionality*, EUSIPCO 2007.)

FR-16: the receiver-side K/V caches must report their effective rank before
and after fusion; the fusion must increase it — the paper's Table 2
publishes, for the Qwen2.5-0.5B → Qwen3-0.6B pair, K 388 → 395 and
V 532 → 560. ``c2c doctor`` runs this measurement; ``c2c.eval`` asserts
the published figures when a full engine plus reference caches are
available to the harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..types import LayeredCache

__all__ = ["effective_rank", "rank_report", "RankReport"]

_TINY = torch.finfo(torch.float32).tiny


def effective_rank(matrix) -> float:
    """Compute erank(A) = exp(−Σ p log p) over the normalised squared singular values.

    Accepts a tensor-like (anything with a ``shape``; moved through
    ``torch.as_tensor``) or a plain sequence of sequences. Uses only the
    numerically stable ``torch.linalg.svdvals`` routine; the empty
    matrix has effective rank 0 (the empty set has rank 0, as it must).
    """
    x = matrix if hasattr(matrix, "square") else torch.as_tensor(matrix, dtype=torch.float32)
    if x.numel() == 0:
        return 0.0
    x = x.to(torch.float32)
    if x.dim() == 1:
        x = x.reshape(1, -1)
    elif x.dim() != 2:
        x = x.reshape(x.shape[0], -1)             # flatten trailing dims, keep rows
    # The singular values, of the principal angles, via the Gram matrix:
    # eigvalsh of AᵀA converges on exactly rank-deficient inputs, where
    # svdvals can stall (NaNs). svd remains the fallback, should the Gram
    # route refuse. (Paper §4.5, Roy & Vetterli 2007 — read the manual,
    # read the paper; the effective rank, the whole rank, nothing but rank.)
    gram = (x.transpose(-1, -2) @ x).to(torch.float64)
    try:
        s2 = torch.linalg.eigvalsh(gram).real.clamp(min=0.0)
        if bool(torch.isnan(s2).any()):
            raise RuntimeError("eigvalsh returned NaNs")
    except RuntimeError:
        s = torch.linalg.svdvals(x)
        s2 = (s * s).to(torch.float64)
    s2 = s2.flatten()
    total = float(s2.sum())
    if total <= float(_TINY):
        return 0.0
    p = s2 / total
    nz = p > _TINY
    if not bool(nz.any()):
        return 0.0
    entropy = float(-(p[nz] * torch.log(p[nz])).sum())
    return math.exp(entropy)


@dataclass(frozen=True)
class RankReport:
    """Effective-rank measurements before/after fusion, per side and kind.

    Fields hold, for each of key/value, a mapping {before: erank, after:
    erank, delta: after−before}; ``increased`` counts how many layers grew
    their rank through the fusion (the diagnostic of choice for FR-16).
    """

    key: dict = field(default_factory=dict)
    value: dict = field(default_factory=dict)
    increased: list[int] = field(default_factory=list, compare=False)

    def __str__(self):
        lines = ["effective rank (Roy & Vetterli 2007):"]
        for kind, table in (("K", self.key), ("V", self.value)):
            b, a = table.get("before", 0.0), table.get("after", 0.0)
            arrow = "↑" if a > b else ("↓" if a < b else "=")
            lines.append(f"  {kind}-cache: {b:0.1f} → {a:0.1f}  Δ {a-b:+0.1f} {arrow}")
        lines.append(f"  layers with increased rank: {len(self.increased)}")
        return "\n".join(lines)


def _mean_rank(cache: LayeredCache, part: str) -> float:
    """Mean effective rank over all layers of one cache, for one part (K or V)."""
    ranks: list[float] = []
    for row in cache:
        tensor = getattr(row, part)
        # one matrix per layer per head, flattened for the measurement
        m = tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() > 2 else tensor
        ranks.append(effective_rank(m))
    return sum(ranks) / len(ranks) if ranks else 0.0


def rank_report(before: LayeredCache, after: LayeredCache) -> RankReport:
    """Measure both caches and report the ranks of the fusion.

    ``before`` is the receiver's own C(X); ``after`` the fused C_f. Both
    must have the same number of layers (they always do, coming from the
    same receiver); a mismatch is a bug, reported as such.
    """
    if len(before) != len(after):
        msg = f"rank comparison needs equal layer counts, got {len(before)} vs {len(after)}"
        raise ValueError(msg)
    increased: list[int] = []
    for n, (b, a) in enumerate(zip(before, after, strict=True)):
        if effective_rank(a.key.reshape(-1, a.key.shape[-1])) > \
           effective_rank(b.key.reshape(-1, b.key.shape[-1])):
            increased.append(n)
    key_before, key_after = _mean_rank(before, "key"), _mean_rank(after, "key")
    val_before, val_after = _mean_rank(before, "value"), _mean_rank(after, "value")
    return RankReport(
        key={"before": key_before, "after": key_after, "delta": key_after - key_before},
        value={"before": val_before, "after": val_after, "delta": val_after - val_before},
        increased=increased,
    )
