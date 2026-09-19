"""Layer alignment — mapping G(n) between the two models' depths (§3.3.3).

Two strategies, both published:

``terminal`` (the default, FR-11)
    Pair the last layers with each other first, then the penultimate, and
    so on in reverse order, until the *shallower* model's first layer is
    consumed. Receiver layers deeper than that (the model runs deeper) map
    onto that first sharer layer — closest to the input — which is the best
    available approximation of "as shallow as possible, as deep as needed".

``depth-normalized`` (App. A.1.1, Eq. (5), behind a flag)
    A linear map over normalised depth::

        G(n) = round(n · (L_S − 1) / (L_R − 1))        for L_R > 1

    NOTE (provenance): the paper's appendix text was not part of the
    extracted source of arXiv:2510.03215v2 available at build time; Eq.(5)
    is implemented as the canonical depth-normalised linear map described
    in the main text and marked as an implementation assumption. `terminal`
    alignment is taken from the paper verbatim and wins any difference.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

__all__ = ["LayerMapping", "terminal_mapping", "depth_normalized_mapping", "LAYER_MODES"]

LAYER_MODES = ("terminal", "depth-normalized")


def _terminal_table(receiver_layers: int, sharer_layers: int) -> list[int]:
    """Build G by pairing, terminal alignment (paper §3.3.3, verbatim)."""
    table: list[int] = []
    for n in range(receiver_layers):
        offset = receiver_layers - 1 - n  # distance from the last layer
        g = sharer_layers - 1 - offset  # counterclockwise on the sharer
        table.append(min(max(g, 0), sharer_layers - 1))  # keep in range
    return table


def _depth_normalized_table(receiver_layers: int, sharer_layers: int) -> list[int]:
    """Linearly map normalised depth, rounding half away from zero (Eq. 5)."""
    if receiver_layers <= 1:
        return [sharer_layers - 1] * receiver_layers
    scale = (sharer_layers - 1) / (receiver_layers - 1)
    return [
        min(max(int(round(n * scale + 0.5)), 0), sharer_layers - 1) for n in range(receiver_layers)
    ]


_TABLES = {"terminal": _terminal_table, "depth-normalized": _depth_normalized_table}


class LayerMapping(Sequence):
    """An immutable, indexable layer map ``G: receiver layer → sharer layer``.

    Behaves as a plain sequence of ints — ``G[n]`` is the sharer layer fused
    with the receiver's layer ``n`` — and knows how to enumerate the paired
    layers in both traversals::

        for pair in mapping.pairs():      # receiver-major, input → output
        for pair in reversed(mapping):    # last layer first (terminal order)
    """

    def __init__(self, receiver_layers: int, sharer_layers: int, mode: str = "terminal"):
        if mode not in _TABLES:
            msg = f"unknown alignment mode {mode!r}; choose one of {LAYER_MODES}"
            raise ValueError(msg)
        if receiver_layers <= 0 or sharer_layers <= 0:
            msg = f"models must have at least one layer, got {receiver_layers}/{sharer_layers}"
            raise ValueError(msg)
        self.receiver_layers = int(receiver_layers)
        self.sharer_layers = int(sharer_layers)
        self.mode = mode
        self._table = _TABLES[mode](self.receiver_layers, self.sharer_layers)

    # sequence protocol ───────────────────────────────────────────────────
    def __len__(self):
        return self.receiver_layers

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._table[index]
        return self._table[index]

    def __eq__(self, other):
        if isinstance(other, LayerMapping):
            return self._table == other._table
        if isinstance(other, (list, tuple)):
            return self._table == list(other)
        return NotImplemented

    def __repr__(self):
        body = ", ".join(str(g) for g in self._table)
        return (
            f"LayerMapping(mode={self.mode!r}, "
            f"receiver={self.receiver_layers}, sharer={self.sharer_layers}, "
            f"G=[{body}])"
        )

    # domain helpers ──────────────────────────────────────────────────────
    def pairs(self) -> Iterator[tuple[int, int]]:
        """Enumerate the mapped (n, G(n)) pairs, receiver-major."""
        return ((n, g) for n, g in enumerate(self._table))

    def reversed(self) -> Iterator[tuple[int, int]]:
        """The same pairs traversed from the last layer backwards."""
        return ((n, g) for n, g in reversed(list(enumerate(self._table))))

    def validate(
        self, receiver_layers: int | None = None, sharer_layers: int | None = None
    ) -> list[str]:
        """Diagnostics: report geometry mismatches as human-readable strings.

        An empty list means the mapping is consistent with both caches.
        """
        problems: list[str] = []
        if receiver_layers is not None and receiver_layers != self.receiver_layers:
            problems.append(
                f"receiver has {receiver_layers} layers, mapping covers {self.receiver_layers}"
            )
        if sharer_layers is not None:
            if min(self._table) < 0 or max(self._table) >= sharer_layers:
                problems.append(f"mapping indexes outside sharer range [0, {sharer_layers - 1}]")
            if sharer_layers != self.sharer_layers:
                problems.append(
                    f"sharer has {sharer_layers} layers, mapping was built for {self.sharer_layers}"
                )
        return problems


def terminal_mapping(receiver_layers: int, sharer_layers: int) -> LayerMapping:
    """The published default: pair the last layers first (FR-11)."""
    return LayerMapping(receiver_layers, sharer_layers, mode="terminal")


def depth_normalized_mapping(receiver_layers: int, sharer_layers: int) -> LayerMapping:
    """The alternative, behind a flag: normalise the depth axis (Eq. 5)."""
    return LayerMapping(receiver_layers, sharer_layers, mode="depth-normalized")
