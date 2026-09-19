"""Diagnostics — effective rank, gate regimes, failure attribution, doctor.

Paper map (normative, spec §1):

* App. A.4.1 Effective rank (Table 2, Fig. 12) → :mod:`c2c.diagnostics.rank`
* App. A.4.2 Gate behaviour regimes → :mod:`c2c.diagnostics.gates`
* FR-17 failure-mode instrumentation  → :mod:`c2c.diagnostics.failure`
* ``c2c doctor`` report               → :mod:`c2c.diagnostics.doctor`
"""

from __future__ import annotations

from .failure import FailureProbe, StructuredLog
from .gates import GateReading, classify_regime
from .rank import effective_rank, rank_report

__all__ = [
    "effective_rank",
    "rank_report",
    "GateReading",
    "classify_regime",
    "FailureProbe",
    "StructuredLog",
]
