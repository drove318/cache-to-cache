"""Fuser modules: projection, dynamic weighting, gate (paper Fig. 5).

    ┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌─────┐   ┌───┐
    │ feature      ├─→ │ projection ├─→ │ dynamic      ├─→ │ × g ├─→ │ + ├─→ fused
    │ fusion (cat) │   │  (linear)  │   │ weighting    │   │gate │   │r  │   cache
    └──────────────┘   └────────────┘   └──────────────┘   └─────┘   └───┘

The two heads, distinguished (spec FR-06): *attention heads* are the model
dimension here — the dynamic weighting module performs input-aware head
modulation, one weight per attention head, computed from the pooled cache
entries of the incoming projection. **CLI heads are documented elsewhere**
(`c2c man config`, section HEADS).

Naming follows the paper: `projection` and `feature fusion` are the two
linear layers of Fig. 5; the gate value g_n is a learnable per-layer scalar.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["Projection", "DynamicWeighting", "Gate", "get_activation"]

_ACTIVATIONS = {
    "identity": nn.Identity,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    """Map an activation by name (like ``getcoder`` maps a name to a codec)."""
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        choices = ", ".join(sorted(_ACTIVATIONS))
        msg = f"unknown activation {name!r}; choose one of: {choices}"
        raise ValueError(msg) from None


class Projection(nn.Module):
    """Concatenate receiver and sharer cache features, then project (Fig. 5).

    The concatenation is the *feature vector* of the pair:
    ``x = [receiver_cache ‖ sharer_cache]`` along the last (feature)
    dimension. The ``projection`` layer maps ``x`` into the shared hidden
    space; the ``feature_fusion`` layer refines it. Shapes::

        x:      [n_tokens, d_receiver + d_sharer]
        delta:  [n_tokens, d_model]
    """

    def __init__(self, d_receiver: int, d_sharer: int, d_model: int | None = None,
                 activation: str = "gelu"):
        super().__init__()
        d_model = d_model or d_receiver
        self.d_in = d_receiver + d_sharer
        self.projection = nn.Linear(self.d_in, d_model)
        self.feature_fusion = nn.Linear(d_model, d_model)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """Project the concatenated cache features (projection → activation → fusion)."""
        h = self.activation(self.projection(x))
        return self.feature_fusion(h)


class DynamicWeighting(nn.Module):
    """Input-aware head modulation: one multiplicative weight per attention head.

    Statistics per head (mean and max over the token axis of the projected
    cache entries) are concatenated and passed through a small modulation
    network; the resulting weights — one per attention head, sigmoid-bounded
    — reweight the projected information per token/query (spec FR-06).

    Shapes::

        projected:  [n_tokens, num_heads, head_size]
        weights:    [num_heads]
    """

    def __init__(self, num_heads: int, head_size: int, halves: int = 1,
                 hidden: int = 0):
        super().__init__()
        if num_heads <= 0 or head_size <= 0 or halves <= 0:
            msg = (f"invalid head configuration: heads={num_heads}, "
                  f"head_size={head_size}, halves={halves}")
            raise ValueError(msg)
        self.num_heads = num_heads
        self.head_size = head_size
        self.halves = halves
        stats_dim = 2 * halves * num_heads * head_size   # mean and max per entry
        hidden = hidden or max(16, stats_dim // 4)
        self.modulation = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, halves * num_heads),
        )

    def statistics(self, projected: Tensor) -> Tensor:
        """Pool the cache entries: mean and max along the token axis."""
        mean = projected.mean(dim=0).flatten()
        maximum = projected.amax(dim=0).flatten()
        return torch.cat((mean, maximum), dim=0)

    def weights(self, projected: Tensor) -> Tensor:
        """One weight per attention head (per half of the joint vector), in (0, 1)."""
        stats = self.statistics(projected)
        logits = self.modulation(stats)
        return torch.sigmoid(logits)

    def forward(self, projected: Tensor) -> Tensor:
        want = (None, self.halves, self.num_heads, self.head_size)
        got = tuple(projected.shape)
        if projected.dim() != 4 or list(got[1:]) != list(want[1:]):
            msg = (
                f"dynamic weighting expects [n, halves, num_heads, head_size] = "
                f"(?, {self.halves}, {self.num_heads}, {self.head_size}), got {got}"
            )
            raise ValueError(msg)
        w = self.weights(projected)
        return projected * w.view(1, self.halves, self.num_heads, 1)


class Gate(nn.Module):
    """Learnable per-layer gates, soft-sampled with the Gumbel-Sigmoid trick.

    One logit per mapped layer pair (FR-07). During training the gate value
    g_n is a straight-through Gumbel-Sigmoid sample — differentiable via the
    soft path — with the temperature annealed *linearly* from ``tau_max`` to
    ``tau_min`` across the training steps. At inference the gate is hard
    binary: open iff the logit is positive (sign of the logit, the argmax
    of the Bernoulli trial).
    """

    def __init__(self, n_mapped: int, *, tau_max: float = 1.0, tau_min: float = 0.001,
                 threshold: float = 0.5, straight_through: bool = True):
        super().__init__()
        if n_mapped <= 0:
            msg = f"number of mapped layers must be positive, got {n_mapped}"
            raise ValueError(msg)
        if not 0.0 < tau_min <= tau_max:
            msg = f"require 0 < tau_min <= tau_max, got {tau_min} > {tau_max}?"
            raise ValueError(msg)
        self.n_mapped = n_mapped
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.threshold = threshold
        self.straight_through = straight_through
        self.logits = nn.Parameter(torch.zeros(n_mapped))
        self.register_buffer("steps_seen", torch.zeros((), dtype=torch.long), persistent=True)

    # -- the linear temperature schedule τ(step) ────────────────────────────
    def temperature_at(self, step: int | None, total_steps: int | None) -> float:
        """τ(step) = τ_max + (τ_min − τ_max) · clamp(step/total, 0, 1)."""
        if step is None or total_steps is None or total_steps <= 0:
            return self.tau_min
        frac = min(1.0, max(0.0, step / total_steps))
        return self.tau_max + (self.tau_min - self.tau_max) * frac

    def gumbel_noise(self, shape: tuple[int, ...], generator: torch.Generator | None = None) -> Tensor:
        """Sample standard Gumbel(0, 1) noise by the inverse-transform method."""
        u = torch.rand(shape, generator=generator, device=self.logits.device)
        u = u.clamp(min=torch.finfo(torch.float32).tiny, max=1.0 - torch.finfo(torch.float32).eps)
        return -torch.log(-torch.log(u))

    def soft_weights(self, tau: float) -> Tensor:
        return torch.sigmoid(self.logits / max(tau, 1e-8))

    def hard_weights(self) -> Tensor:
        """Inference-time decision: open iff logit > 0 (the sign)."""
        return (self.logits > 0.0).to(self.logits.dtype)

    def forward(self, *, training: bool | None = None, step: int | None = None,
                total_steps: int | None = None, generator: torch.Generator | None = None) -> Tensor:
        """Return one gate value g_n per mapped layer, in [0, 1]."""
        training = self.training if training is None else training
        if not training:
            return self.hard_weights()
        tau = self.temperature_at(step, total_steps)
        noise = self.gumbel_noise(tuple(self.logits.shape), generator)
        soft = torch.sigmoid((self.logits + noise) / max(tau, 1e-8))
        hard = (soft > self.threshold).to(self.logits.dtype)
        if not self.straight_through:
            return hard.detach()
        # straight-through estimator: forward hard, backward soft.
        return hard + (soft - soft.detach())

    # -- diagnostics ────────────────────────────────────────────────────────
    def probabilities(self) -> list[float]:
        """The p_n a of the gate: sigmoid of each logit (no noise)."""
        return torch.sigmoid(self.logits).tolist()

    def activation_ratio(self, threshold: float | None = None) -> float:
        """Share of open gates at inference — the ratio `c2c doctor` reports."""
        th = self.threshold if threshold is None else threshold
        opened = int((torch.sigmoid(self.logits) > th).count_nonzero())
        return opened / self.n_mapped
