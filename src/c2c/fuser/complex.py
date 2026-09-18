"""C2C-C — the complex fuser variant (App. A.1.3, Table 9).

An optional *pre-projection* stage: a 3-layer MLP maps the Sharer cache
into the Receiver's dimensionality **before** the concatenation of the
projection module. Selected via ``FuserConfig(variant="c2c-c")`` — a config
switch, deliberately not a second binary (FR-09): one code base, many
fusion behaviours.

The same 3-layer MLP topology also drives the cache-transformation oracle
of §3.2.2 / Fig. 3, which is why the paper's appendix reuses it; we keep
one implementation, two use sites.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .modules import get_activation

__all__ = ["PreProjection"]


class PreProjection(nn.Module):
    """Map sharer cache rows into the receiver's feature space.

    Parameters
    ----------
    in_features, out_features:
        ``d_s`` and ``d_r`` — the dimensionality of the joint vector before
        and after the mapping (the paper's flattened key/value width).
    layers:
        Number of linear layers of the MLP (default 3, App. A.1.3). The
        last layer is linear by design: the *activation* is applied between
        the layers, never after the output layer, so the mapping preserves
        the numeric range of the cache entries as far as possible.

    Notes
    -----
    ``requires_grad`` on cache entries is never set: caches are read-only
    streams of the frozen models; only this module's weights learn.
    """

    def __init__(self, in_features: int, out_features: int, *, layers: int = 3,
                 activation: str = "gelu"):
        super().__init__()
        if layers < 1:
            msg = f"an MLP with {layers} layers makes no sense; use layers >= 1"
            raise ValueError(msg)
        if in_features <= 0 or out_features <= 0:
            msg = f"non-positive feature width: {in_features} → {out_features}"
            raise ValueError(msg)
        blocks: list[nn.Module] = []
        d = in_features
        for i in range(layers):
            blocks.append(nn.Linear(d, out_features))
            if i < layers - 1:                     # activation between layers only
                blocks.append(get_activation(activation))
            d = out_features
        self.mlp = nn.Sequential(*blocks)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, rows: Tensor) -> Tensor:
        if rows.shape[-1] != self.in_features:
            msg = (
                f"pre-projection expected {self.in_features} input features, "
                f"got tensor of shape {tuple(rows.shape)}"
            )
            raise ValueError(msg)
        return self.mlp(rows)

    def __repr__(self):
        head = f"{self.in_features} → {self.out_features}"
        return f"PreProjection({head}, layers={len(self.mlp)})"
