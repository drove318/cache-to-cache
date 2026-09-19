"""Progressive blending of caches (App. A.2.4, Fig. 11; spec FR-08).

The fusion may replace only a *fraction* of the receiver's cache entries
with their fused counterparts, and it may traverse the token axis in either
direction:

* ``former``  — front-to-back: replace the former (oldest) entries first;
* ``latter``  — back-to-front: replace from the newest entry backwards.

The published observation (Fig. 11): once the fused fraction passes 50 %,
increasing it further monotonically increases accuracy, in both traversal
directions. The golden suite asserts the monotonicity with the paper's
numbers where the full stack is available; the unit suite asserts the
selection logic itself with synthetic caches.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..types import BlendDirection, LayeredCache, LayerSlice, concat_rows

__all__ = ["apply", "sweep", "normalize_fraction"]

_EPS = 1e-9


def normalize_fraction(fraction: float) -> float:
    """Normalise a fused fraction to [0, 1].

    Percentages (0–100) are tolerated and divided by 100 — the user may
    write ``--fraction 75`` or ``--fraction 0.75``; both mean the same.
    Values outside both ranges are configuration errors, loudly reported.
    """
    f = float(fraction)
    if 1.0 < f <= 100.0:            # percentage form
        f = f / 100.0
    if not 0.0 <= f <= 1.0:
        msg = f"fused fraction {fraction!r} out of range [0, 1] (or [0, 100])"
        raise ValueError(msg)
    return f


def _rows_to_replace(n_tokens: int, fraction: float) -> int:
    """How many token rows the fraction selects (round-half, ties away from zero)."""
    return int(n_tokens * fraction + 0.5 + _EPS)


def _blend_rows(base_rows, fused_rows, count: int, direction: BlendDirection):
    """Select rows from two sources by position on the token axis.

    ``former``  takes the first ``count`` rows from the fused cache;
    ``latter``  takes the last ``count`` rows from the fused cache.
    Everything else is kept from the base (receiver) cache — the blend is
    never destructive to the receiver's own information (cf. FR-03).
    """
    n = len(base_rows)
    if n != len(fused_rows):
        msg = f"blend requires equal token counts, got {n} vs {len(fused_rows)}"
        raise ValueError(msg)
    if count <= 0:
        return base_rows
    if count >= n:
        return fused_rows
    if direction is BlendDirection.FORMER:
        return concat_rows(fused_rows[:count], base_rows[count:])
    split = n - count
    return concat_rows(base_rows[:split], fused_rows[split:])


def apply(base: LayeredCache, fused: LayeredCache, *, fraction: float = 1.0,
          direction: BlendDirection | str = BlendDirection.FORMER) -> LayeredCache:
    """Blend a fused cache into its base, replacing ``fraction`` of the rows.

    Layer by layer, the selected rows are taken ``from fused``, the rest
    ``from base``; the two must agree in the number of layers and tokens
    (compared below, in the implementation of this module).
    """
    if len(base) != len(fused):
        msg = f"blend requires equal layer counts, got {len(base)} vs {len(fused)}"
        raise ValueError(msg)
    f = normalize_fraction(fraction)
    direction = direction if isinstance(direction, BlendDirection) else BlendDirection(direction)
    blended: list[LayerSlice] = []
    for b, u in zip(base, fused, strict=True):
        count = _rows_to_replace(len(b), f)
        key = _blend_rows(b.key, u.key, count, direction)
        value = _blend_rows(b.value, u.value, count, direction)
        blended.append(LayerSlice(key, value))
    return LayeredCache(blended)


def sweep(base: LayeredCache, fused: LayeredCache, *,
          fractions: list[float] | None = None,
          direction: BlendDirection | str = BlendDirection.FORMER) -> Iterator[tuple[float, LayeredCache]]:
    """Yield ``(fraction, cache)`` pairs for the Fig. 11 accuracy curve.

    A generator: the caller may stop early — plotting the whole sweep is
    the evaluation harness' job (``c2c.eval``), not ours.
    """
    if fractions is None:
        fractions = [i / 10 for i in range(11)]           # 0.0 … 1.0, step 0.1
    for fr in fractions:
        yield normalize_fraction(fr), apply(base, fused, fraction=fr, direction=direction)
