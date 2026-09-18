"""Cache probes — the oracle experiments of §3.2, reproducible offline.

These are not the experiments: they are the two *oracle* studies that
motivated C2C, implemented as reusable instruments so any harness can run
them against any pair of engines (see the manual pages for details on the
different operating modes, `c2c man probes`).

* :mod:`c2c.probes.enrich`   — cache-enrichment oracle, §3.2.1, Table 1, Fig. 4
* :mod:`c2c.probes.transform` — cache-transformation oracle, §3.2.2, Fig. 3

The published reference values are collected in :mod:`c2c.eval.golden`.
"""

from __future__ import annotations

from .enrich import EnrichmentOracle, EnrichmentResult
from .transform import TransformationOracle, TransformationResult

__all__ = ["EnrichmentOracle", "EnrichmentResult", "TransformationOracle", "TransformationResult"]
