"""Token & layer alignment between heterogeneous models (paper §3.3.3).

Fusing KV-Caches across model families and sizes requires alignment at two
levels: *tokens* (different tokenizers, same string) and *layers* (different
depths, same semantic position). See :mod:`c2c.align.tokens` and
:mod:`c2c.align.layers`.
"""

from __future__ import annotations

from .layers import (
    LayerMapping, depth_normalized_mapping, terminal_mapping,
)
from .tokens import AlignedToken, TokenAligner

__all__ = [
    "TokenAligner", "AlignedToken",
    "LayerMapping", "terminal_mapping", "depth_normalized_mapping",
]
