"""Gate behaviour under different training regimes (App. A.4.2; FR-07).

The published observation: **general-purpose training favours broad gate
activation with fine-grained modulation via the weights, whereas
task-specific training favours sparse gate activation with strong reliance
on the selected layers.** This module turns that observation into a
measurable reading — one :class:`GateReading` per gate — and a
deterministic classifier.

The three regimes are named after the paper's Figure 10/11 vocabulary::

    general  → broad-open  + small weights
    mixed    → in between (report it, but do not force a label)
    task     → sparse-open + strong weights
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["GateReading", "classify_regime", "REGIMES", "BROAD_OPEN", "SPARSE_OPEN"]

REGIMES = ("general", "mixed", "task")

#: activation-ratio boundaries from the paper's discussion, rounded to a
#: practical pair of thresholds (they are engineering choices, documented):
BROAD_OPEN = 0.60      # ≥ 60 % of gates open  → general-purpose regime
SPARSE_OPEN = 0.40     # ≤ 40 % of gates open  → task-specific regime


@dataclass(frozen=True)
class GateReading:
    """One measurement of a gate's behaviour.

    ratio
        share of open gates at inference (hard decisions, p > threshold).
    mean_weight
        mean |gᵗ| over the *open* gates — the modulation strength the
        surviving contributions carry when they are injected.
    regime
        the label assigned by :func:`classify_regime`; one of :data:`REGIMES`.
    """

    ratio: float
    mean_weight: float
    regime: str

    def __str__(self):
        return (f"regime={self.regime:<7} open={self.ratio:0.2f} "
                f"mean|g|={self.mean_weight:0.3f}")


def classify_regime(ratio: float, mean_weight: float | None = None, *,
                    broad: float = BROAD_OPEN, sparse: float = SPARSE_OPEN) -> GateReading:
    """Assign one of the three regimes to a (ratio, mean|weight|) reading.

    Classification first, directions second: a gate population that opens
    broadly (*ratio* ≥ ``broad``) is *general*; one that opens sparsely
    (*ratio* ≤ ``sparse``) and relies on strong weights is *task-specific*;
    everything in between stays *mixed* and is reported as such rather than
    forced into a binary. When ``mean_weight`` is given and disagrees with
    the ratio-based reading, the disagreement is *not* silently corrected —
    it is kept in the record for the golden suite to inspect.
    """
    if not 0.0 <= ratio <= 1.0:
        msg = f"activation ratio must lie in [0, 1], got {ratio!r}"
        raise ValueError(msg)
    if mean_weight is not None and not math.isfinite(mean_weight):
        msg = f"mean weight must be finite, got {mean_weight!r}"
        raise ValueError(msg)
    if ratio >= broad:
        regime = "general"
    elif ratio <= sparse:
        regime = "task"
    else:
        regime = "mixed"
    return GateReading(ratio=float(ratio),
                      mean_weight=(float(mean_weight) if mean_weight is not None else 0.0),
                      regime=regime)
