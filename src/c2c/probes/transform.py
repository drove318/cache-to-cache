"""Cache-transformation oracle (paper §3.2.2, Figure 3; spec §1 map).

Experiment, as published: train a three-layer MLP to map the KV-Cache of
a source LLM (Qwen3-4B) into the representation space of a target LLM
(Qwen3-0.6B). The t-SNE visualisation reveals that the raw caches of the
two models are far apart, while the *transformed* cache falls within the
target's representation space — the observation that licenses the
concatenation of the two caches in the fuser's projection module.

The oracle here is a reusable instrument, not a one-off:
:meth:`TransformationOracle.fit` trains the projector on any pair of
captured caches, :meth:`~TransformationOracle.transform` maps a cache
into the target space, :meth:`~TransformationOracle.evaluate` measures
how deep the mapped cache lies within the target's neighbourhood (a
mean-absolute-error between aligned representations, plus a purity
score: the share of transformed tokens whose nearest target neighbour is
the intended one).

For the plotting: :meth:`TransformationOracle.export` writes a
comma-separated values file of the two-dimensional projection — exact,
deterministic, via principal component analysis (PCA by singular value
decomposition through :func:`torch.linalg.svd`). Full t-SNE is available
if the optional scikit-learn peer is installed; the fallback is
documented, never silent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import torch
from torch import nn

from ..fuser.complex import PreProjection
from ..types import LayeredCache

__all__ = ["TransformationResult", "TransformationOracle"]


@dataclass(frozen=True)
class TransformationResult:
    """How faithful (in numbers) the transformation was."""

    mae: float                       # mean absolute error to the target rows
    purity: float                    # share of rows whose nearest neighbour is intended
    distance_before: float           # mean pairwise distance, raw caches
    distance_after: float            # mean pairwise distance, transformed vs target
    steps: int = 0
    loss_curve: tuple = field(default_factory=tuple, compare=False)

    def __str__(self):
        return (f"mae {self.mae:0.4f} | purity {self.purity:0.1%} | "
               f"d(before) {self.distance_before:0.3f} → d(after) {self.distance_after:0.3f}")


def _mean_pairwise_distance(a, b) -> float:
    """Mean of the elementwise absolute differences — one pair, many pairs."""
    x = a.reshape(a.shape[0], -1)
    y = b.reshape(b.shape[0], -1)
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n == 0:
        return 0.0
    d = (x[:n] - y[:n]).abs().mean()
    return float(d)


class TransformationOracle:
    """The cache-transformation oracle of §3.2.2 / Fig. 3.

    The projector is a three-layer MLP (:class:`c2c.fuser.complex.PreProjection`)
    trained to minimise the distance between the transformed source cache
    and the target cache, in the spirit of the appendix's setup (A.3.2).
    """

    def __init__(self, *, latent_width: int | None = None, layers: int = 3,
                lr: float = 1e-3, epochs: int = 300, seed: int = 42):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.latent_width = latent_width
        self.layers = int(layers)
        self.projector: nn.Module | None = None

    # -- the fitting ----------------------------------------------------------
    def fit(self, source: LayeredCache, target: LayeredCache, *,
           epochs: int | None = None) -> TransformationResult:
        """Fit the projector from *source* caches onto *target* caches."""
        x, y = self._aligned_pair(source, target)
        if x.shape[1] != y.shape[1]:
            msg = (f"aligned representations require equal feature dimensions, "
                 f"got {x.shape[1]} and {y.shape[1]}")
            raise ValueError(msg)
        torch.manual_seed(self.seed)
        self.projector = PreProjection(int(x.shape[1]), int(y.shape[1]),
                                       layers=self.layers)
        optim = torch.optim.AdamW(self.projector.parameters(), lr=self.lr, weight_decay=0.0)
        curve: list[float] = []
        for _ in range(epochs or self.epochs):
            optim.zero_grad()
            loss = nn.functional.mse_loss(self.projector(x), y)
            loss.backward()
            optim.step()
            curve.append(float(loss.detach()))
        return self.evaluate(source, target, steps=len(curve),
                           loss_curve=tuple(curve[-8:]))

    # -- evaluating -----------------------------------------------------------
    def evaluate(self, source: LayeredCache, target: LayeredCache, *,
                steps: int = 0, loss_curve: tuple = ()) -> TransformationResult:
        """Score the current projector on the (source, target) pair."""
        if self.projector is None:
            msg = "cannot evaluate before fitting: no projector"
            raise LookupError(msg)
        x, y = self._aligned_pair(source, target)
        with torch.no_grad():
            t = self.projector(x)
            mae = float((t - y).abs().mean())
            d_before = _mean_pairwise_distance(x, y)
            d_after = _mean_pairwise_distance(t, y)
            purity = self._nearest_neighbour_purity(t, y)
        return TransformationResult(mae=mae, purity=purity,
                                 distance_before=d_before, distance_after=d_after,
                                 steps=steps, loss_curve=tuple(loss_curve))

    def transform(self, cache: LayeredCache) -> LayeredCache:
        """Map a cache into the target's representation space."""
        if self.projector is None:
            msg = "cannot transform before fitting: call .fit first"
            raise LookupError(msg)
        from ..types import LayerSlice
        rows = []
        with torch.no_grad():
            for slc in cache:
                k = self.projector(slc.key.reshape(slc.key.shape[0], -1))
                v = self.projector(slc.value.reshape(slc.value.shape[0], -1))
                rows.append(LayerSlice(k, v))
        return LayeredCache(rows)

    # -- plotting (in ASCII, see the manual for curses) ----------------------
    def export(self, source: LayeredCache, target: LayeredCache, path: str, *,
              method: str = "pca") -> str:
        """Write a two-dimensional projection of the caches as CSV.

        Columns: ``x, y, kind`` where kind ∈ {source, target, transformed}.
        ``method="pca"`` is exact and deterministic (svd); ``method="tsne"``
        uses the optional scikit-learn peer when present, and raises —
        never silently falls back — when it is not installed.
        """
        x, y = self._aligned_pair(source, target)
        with torch.no_grad():
            t = self.projector(x) if self.projector is not None else x
        data = torch.cat((x, y, t), dim=0).detach().to(torch.float64)
        if method == "tsne":
            coords = self._tsne(data)
        else:
            coords = self._pca(data)
        labels = (["source"] * x.shape[0]) + (["target"] * y.shape[0]) \
            + (["transformed"] * t.shape[0])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(("x", "y", "kind"))
            for (cx, cy), kind in zip(coords.tolist(), labels):
                writer.writerow((f"{cx:0.6f}", f"{cy:0.6f}", kind))
        return path

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _aligned_pair(source: LayeredCache, target: LayeredCache):
        """Concatenate and stack the rows of both caches into two matrices."""
        def stack_rows(cache):
            rows = [slc.key.reshape(slc.key.shape[0], -1) for slc in cache]
            return torch.cat(rows, dim=0)
        return stack_rows(source), stack_rows(target)

    @staticmethod
    def _nearest_neighbour_purity(t, y) -> float:
        """Share of transformed rows whose nearest target row is the intended one."""
        if t.shape[0] == 0 or y.shape[0] == 0:
            return 0.0
        d = torch.cdist(t, y)                      # all pairs, no stones skipped
        if d.shape[1] == 0:
            return 0.0
        nearest = d.argmin(dim=1)
        hits = int((nearest == torch.arange(t.shape[0], device=t.device)).count_nonzero())
        return hits / int(t.shape[0])

    @staticmethod
    def _pca(data) -> torch.Tensor:
        """Project onto the first two principal components, exactly."""
        centred = data - data.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        return centred @ vh[:2].transpose(0, 1)

    @staticmethod
    def _tsne(data) -> torch.Tensor:
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            msg = ("t-SNE needs the optional scikit-learn peer "
                  "(pip install scikit-learn); use method='pca' for the exact, "
                  "deterministic projection")
            raise ModuleNotFoundError(msg) from exc
        return TSNE(n_components=2, init="pca", random_state=42).fit_transform(
            data.numpy())
