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

#: the numerics of the package — the t-SNE oracle and its measures
_TRANSFORM_NAMES = ("TransformationOracle", "TransformationResult")


def __getattr__(name: str) -> object:
    """PEP 562: the enrichment probe (§3.2.1, numpy-side) does not pull the
    transformation oracle (§3.2.2, the t-SNE study, torch) in until asked."""
    if name in _TRANSFORM_NAMES:
        from . import transform

        return getattr(transform, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__getattr__", "__dir__"})


__all__ = ["EnrichmentOracle", "EnrichmentResult", "TransformationOracle", "TransformationResult"]
