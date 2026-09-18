"""Failure-mode instrumentation (FR-17; paper App. A.4.6).

The paper's most honest limitation: *the contextual understanding a Sharer
provides is not always accurate and can mislead the Receiver into
generating the wrong answer*. A silent failure is an unacceptable failure;
this module makes every such event attributable and reportable — per layer,
per gate, with a confidence on the fused contribution.

Design
------
``compare(own, fused, gate_values)`` computes, per mapped layer pair:

contribution
    ‖fused − own‖ / (‖own‖ + ε) — how much the fusion actually moved
    this layer's cache, relative to the layer's own magnitude;

confidence
    the gate probability × the contribution: the share of the layer's
    change the deployment is *responsible for* (a closed gate cannot be
    blamed for noise, an open gate with a huge delta deserves a look);

suspect
    True when an open gate moved the cache beyond ``threshold`` — the
    layers to blame when a reply goes wrong, per FR-17.

Every reading is emitted as one structured log record (JSON per line,
``logging`` module, dedicated handler) — never printed to the terminal
behind the user's back, and always available to ``c2c fuse --report``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from ..types import LayeredCache

__all__ = ["LayerAttribution", "FailureProbe", "StructuredLog"]

_EPS = 1e-12


@dataclass(frozen=True)
class LayerAttribution:
    """One layer's share of the blame (or the credit)."""

    layer: int
    gate: float
    contribution: float
    confidence: float
    suspect: bool
    note: str = ""

    def __str__(self):
        blame = "SUSPECT" if self.suspect else "clean"
        return (f"layer {self.layer:>2}  g={self.gate:0.3f}  "
                f"Δ={self.contribution:0.3f}  conf={self.confidence:0.3f}  [{blame}]")


class StructuredLog(logging.Logger):
    """A logger that writes JSON-lines records to ``c2c.failure``."""

    def __init__(self, stream=None):
        super().__init__("c2c.failure")
        if not self.handlers:
            from logging import NullHandler
            self.addHandler(NullHandler())          # silence is golden
        self.setLevel(logging.INFO)
        self._stream = stream
        self.records_written: list[dict] = []      # in-memory, for tests/CLI

    def log_attribution(self, **fields) -> None:
        """Emit one structured record; returns immediately, never blocks."""
        clean = {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v)
                 for k, v in fields.items()}
        line = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
        self.records_written.append(clean)            # the mirror, always on
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()


class FailureProbe:
    """Compare own caches against fused caches and attribute the delta.

    Usage (from the CLI and the agent integrations alike)::

        probe = FailureProbe(threshold=0.5)
        blames = probe.compare(receiver_caches, fused_caches, gate_values)
        for b in blames:
            probe.log.log_attribution(**asdict(b))
    """

    def __init__(self, *, threshold: float = 0.5):
        if not 0.0 < threshold:
            msg = f"threshold must be positive, got {threshold!r}"
            raise ValueError(msg)
        self.threshold = float(threshold)
        self.log = StructuredLog()

    @staticmethod
    def _magnitude(tensor) -> float:
        t = tensor.reshape(-1) if hasattr(tensor, "reshape") else torch.as_tensor(tensor)
        return float(t.norm()) if t.numel() else 0.0

    @staticmethod
    def _difference(a, b):
        """a − b for tensor-likes; falls back to as_tensor for plain sequences."""
        if hasattr(a, "__sub__") and hasattr(b, "__sub__"):
            return a - b
        ta = torch.as_tensor(a, dtype=torch.float32)
        tb = torch.as_tensor(b, dtype=torch.float32)
        return ta - tb

    def compare(self, own: LayeredCache, fused: LayeredCache,
                gate_values: Sequence[float]) -> list[LayerAttribution]:
        """Attribute the contribution of every mapped layer pair.

        ``own`` is the receiver's C(X); ``fused`` the result C_f; both must
        have the same number of layers (they do, by construction). The gate
        values are the effective weights g_n actually applied.
        """
        if len(own) != len(fused):
            msg = f"compare needs equal layer counts, got {len(own)} vs {len(fused)}"
            raise ValueError(msg)
        if len(gate_values) != len(own):
            msg = (f"one gate value per layer required: expected {len(own)}, "
                   f"got {len(gate_values)}")
            raise ValueError(msg)
        out: list[LayerAttribution] = []
        for n, (o, f, g) in enumerate(zip(own, fused, gate_values, strict=True)):
            own_k, own_v = self._magnitude(o.key), self._magnitude(o.value)
            delta_k = self._magnitude(self._difference(f.key, o.key))
            delta_v = self._magnitude(self._difference(f.value, o.value))
            contribution = (delta_k + delta_v) / ((own_k + own_v) + _EPS)
            confidence = float(g) * contribution
            suspect = float(g) > 0.5 and contribution > self.threshold
            note = ""
            if suspect:
                note = ("open gate moved this layer beyond the threshold; the "
                        "Sharer's contextual understanding may mislead the Receiver "
                        "(paper App. A.4.6)")
            out.append(LayerAttribution(layer=n, gate=float(g),
                                       contribution=float(contribution),
                                       confidence=float(confidence),
                                       suspect=bool(suspect), note=note))
        return out

    def report(self, attributions: Sequence[LayerAttribution], *,
               emit: bool = True) -> str:
        """Format (and optionally log) a table of attributions.

        Returns the human-readable rendering; the machine-readable form
        goes to the structured log, one JSON object per line, so both the
        terminal and the agent see exactly the same event.
        """
        lines = [str(a) for a in attributions]
        if emit:
            for a in attributions:
                self.log.log_attribution(**asdict(a))
        return "\n".join(lines) if lines else "no layers compared"
