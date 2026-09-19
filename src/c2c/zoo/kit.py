"""Unified latent space for many-to-one cache communication (App. A.5.2).

Instead of one pairwise fuser per (Sharer, Receiver) pair — which costs
``O(M · N)`` trainable parameters and re-trains the world for every new
model — the kit projects **every** Sharer's joint cache vector into
**one** shared latent space ``z`` of width ``latent_dim`` (M projectors)
and then fuses ``z`` into the Receiver with a *single* stack of per-layer
fusion modules (N fusers). The learnable count is therefore
``M + N`` — ``O(M + N)``, not ``O(M · N)`` — the scalability result
published as Table 14. Many-to-one fusion (two Sharers, one Receiver) is
the Table 13 configuration, which reports 64.60 average accuracy on the
Receiver alone.

Conventions
-----------
* one *joint vector* per token: ``x = [key ‖ value]`` flattened over the
  head structure (see :func:`c2c.fuser.core._rows`);
* the latent ``z`` is one joint vector per token as well — key and value
  halves are split *after* the fusion, never before the projection;
* shapes are given as ``[n_tokens, d]`` unless stated otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..config import FuserConfig, GateConfig
from ..fuser.complex import PreProjection
from ..fuser.modules import DynamicWeighting, Gate
from ..types import LayeredCache, LayerGeometry, LayerSlice

__all__ = ["UnifiedLatentSpace"]


def _flat(tensor) -> torch.Tensor:
    """Flatten the heads of one cache side, keeping the token axis."""
    return tensor.reshape(tensor.shape[0], -1)


class _LatentFuser(nn.Module):
    """Fuse one receiver layer with the averaged latent, producing a delta.

    Fig. 5 transplanted into the latent space: concat the receiver's joint
    vector with the latent, project, refine with feature fusion, modulate
    per attention head — then split the result into its key and value
    halves for the residual update of Eq. (3).
    """

    def __init__(self, receiver: LayerGeometry, latent_dim: int, cfg: FuserConfig):
        super().__init__()
        self.kv = receiver.kv_hidden_size
        d_joint = 2 * self.kv  # the receiver's own joint vector
        self.projection = nn.Linear(d_joint + latent_dim, d_joint)
        self.feature_fusion = nn.Linear(d_joint, d_joint)
        self.act = nn.GELU()
        self.weighting = DynamicWeighting(receiver.kv_heads, receiver.head_size, halves=2)
        self.kvh = receiver.kv_heads
        self.hs = receiver.head_size

    def forward(self, r_k: torch.Tensor, r_v: torch.Tensor, z: torch.Tensor):
        """Return the ``(delta_key, delta_value)`` half-vector contributions."""
        x = torch.cat((r_k, r_v, z), dim=-1)
        h = self.act(self.projection(x))
        delta = self.feature_fusion(h)  # [n, 2·kv]
        n, d = delta.shape[0], delta.shape[-1]
        heads_view = delta.reshape(n, 2, self.kvh, self.hs)
        heads_view = self.weighting(heads_view)  # attention-head modulation
        delta = heads_view.reshape(n, d)
        delta_k, delta_v = delta.split(self.kv, dim=-1)  # the halves, as per standard
        return delta_k, delta_v


class UnifiedLatentSpace(nn.Module):
    """M projectors into one latent KV space, N per-layer fusers.

    Parameters
    ----------
    sharers:
        Layer geometries of the M Sharer models, in the order their caches
        are passed to :meth:`fuse_many`.
    receiver:
        Layer geometry of the single Receiver model.
    mapping:
        ``G(n)``: receiver layer → sharer layer (a ``c2c.align`` sequence,
        same semantics as in :class:`c2c.fuser.Fuser`).
    latent_dim:
        Width of the shared latent space; defaults to the receiver's joint
        dimension ``2·kv_hidden``.
    """

    def __init__(
        self,
        sharers: Sequence[LayerGeometry],
        receiver: LayerGeometry,
        mapping: Sequence[int],
        *,
        latent_dim: int | None = None,
        fuser_config: FuserConfig | None = None,
        gate_config: GateConfig | None = None,
    ):
        super().__init__()
        self.sharers = list(sharers)
        self.receiver = receiver
        self.mapping = list(mapping)
        if not self.sharers:
            msg = "a unified latent space needs at least one sharer"
            raise ValueError(msg)
        if not self.mapping:
            msg = "empty layer mapping: nothing to fuse"
            raise ValueError(msg)
        self.latent_dim = int(2 * receiver.kv_hidden_size if latent_dim is None else latent_dim)
        fc = fuser_config or FuserConfig()
        gc = gate_config or GateConfig()

        self.projectors = nn.ModuleList(
            [
                PreProjection(
                    s.d, self.latent_dim, layers=fc.pre_projection_layers, activation=fc.activation
                )
                for s in self.sharers
            ]
        )
        self.fusers = nn.ModuleList(
            [_LatentFuser(receiver, self.latent_dim, fc) for _ in self.mapping]
        )
        self.gate = Gate(
            len(self.mapping),
            tau_max=gc.tau_max,
            tau_min=gc.tau_min,
            threshold=gc.threshold,
            straight_through=gc.straight_through,
        )
        self.fuser_config = fc
        self._receiver_cache: LayeredCache | None = None

    # -- the parameter budget audit, as published ───────────────────────────
    def parameter_account(self) -> dict[str, int | str]:
        """Parameter counts per group — the Table 14 scalability evidence.

        ``total`` must grow linearly in the number of sharers (M) and in
        the number of mapped receiver layers (N). The golden suite checks
        the slope of this growth.
        """

        def numel(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        projector_params = sum(numel(p) for p in self.projectors)
        fuser_params = sum(numel(f) for f in self.fusers) + numel(self.gate)
        return {
            "sharers": len(self.sharers),
            "receiver_layers": len(self.mapping),
            "projector_params": projector_params,
            "fuser_params": fuser_params,
            "total": projector_params + fuser_params,
            "scaling": "O(M+N)",
        }

    # -- installation of the receiver's own cache ───────────────────────────
    def set_receiver_cache(self, cache: LayeredCache) -> None:
        """Install the receiver's own C(X); read by :meth:`fuse_many`."""
        if len(cache) != self.receiver.layers:
            msg = (
                f"receiver cache has {len(cache)} layers, geometry declares {self.receiver.layers}"
            )
            raise ValueError(msg)
        self._receiver_cache = cache

    # -- projecting into the latent space ───────────────────────────────────
    def project(self, cache: LayeredCache, *, sharer_index: int) -> list[torch.Tensor]:
        """Project one sharer's caches layer by layer into the shared space.

        Returns a list of latent tensors, one per mapped layer of *this*
        sharer's cache: ``z = P_s( C(X)_s )``, shape ``[n_tokens, latent_dim]``.
        """
        if not 0 <= sharer_index < len(self.sharers):
            msg = (
                f"sharer index {sharer_index} out of range for "
                f"{len(self.sharers)} registered sharer(s)"
            )
            raise IndexError(msg)
        projector = self.projectors[sharer_index]
        latents: list[torch.Tensor] = []
        for row in cache:
            x = torch.cat((_flat(row.key), _flat(row.value)), dim=-1)  # the joint vector
            latents.append(projector(x))
        return latents

    # -- the many-to-one fusion ─────────────────────────────────────────────
    def fuse_many(
        self,
        caches: Sequence[LayeredCache],
        *,
        token_mappings: Sequence[Sequence[int] | None] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> LayeredCache:
        """Fuse all registered sharer caches into one receiver cache.

        ``caches[i]`` belongs to ``self.sharers[i]``. For every mapped
        receiver layer ``n`` the latent of each sharer at its layer
        ``G(n)`` is gathered, averaged (equal weights — a deliberate
        baseline, as the paper publishes it),
        and injected through the layer's fusion module, scaled by the
        learnable gate g_n and applied residually — never a destructive
        overwrite (FR-03, shared provenance with the pairwise fuser).
        """
        if len(caches) != len(self.sharers):
            msg = f"expected {len(self.sharers)} caches (one per sharer), got {len(caches)}"
            raise ValueError(msg)
        if self._receiver_cache is None:
            msg = "no receiver cache installed; call set_receiver_cache(C(X)) before fuse_many()"
            raise LookupError(msg)
        for i, cache in enumerate(caches):
            if max(self.mapping) >= len(cache):
                msg = (
                    f"sharer {i} exposes {len(cache)} layers, but the mapping "
                    f"indexes layer {max(self.mapping)}"
                )
                raise IndexError(msg)
        weights = self.gate(training=self.training, step=step, total_steps=total_steps)
        latents = [self.project(c, sharer_index=i) for i, c in enumerate(caches)]

        # per receiver row, the aligned sharer rows for token selection
        aligned: list[list[int] | None] = []
        for i, cache in enumerate(caches):
            tm = None if token_mappings is None else token_mappings[i]
            if tm is None:
                aligned.append(None)
                continue
            aligned.append(list(tm))

        out: list[LayerSlice] = []
        for n in range(len(self.mapping)):
            g = self.mapping[n]
            base = self._receiver_cache[n]
            r_k, r_v = _flat(base.key), _flat(base.value)
            z_sum = None
            for i, lat in enumerate(latents):
                z = lat[g]
                rows = aligned[i]
                if rows is not None:
                    idx = torch.as_tensor(rows, dtype=torch.long, device=z.device)
                    z = z.index_select(0, idx)
                if z.shape[0] != r_k.shape[0]:
                    msg = (
                        f"sharer {i} latent has {z.shape[0]} rows, receiver has "
                        f"{r_k.shape[0]}; pass token_mappings for this pair"
                    )
                    raise ValueError(msg)
                z_sum = z if z_sum is None else z_sum + z
            if z_sum is None:
                msg = (
                    f"layer {n}: no latents to average; the zoo holds "
                    f"nothing registered for this receiver"
                )
                raise ValueError(msg)
            z_bar = z_sum / len(latents)  # the averaged latent
            delta_k, delta_v = self.fusers[n](r_k, r_v, z_bar)
            w = weights[n]
            fused_k = r_k + w * delta_k  # Eq. (3), the residual add
            fused_v = r_v + w * delta_v
            kvh = self.receiver.kv_heads
            if kvh > 1:  # the heads view back, the layout the engines install
                hs = self.receiver.head_size
                fused_k = fused_k.reshape(fused_k.shape[0], kvh, hs)
                fused_v = fused_v.reshape(fused_v.shape[0], kvh, hs)
            out.append(LayerSlice(fused_k, fused_v))
        return LayeredCache(out)
