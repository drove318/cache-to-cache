"""Cache-enrichment oracle (paper §3.2.1, Table 1, Figure 4; spec §1 map).

Three operating modes, one instrument:

``direct``
    Prefill the question ``X`` only; decode with ``C(X)``.
    Table 1: cache length ``|X|``, no enrichment, accuracy 58.42 %.

``few_shot``
    Prefill ``E ⊕ X`` (exemplars then question); decode with the *whole*
    cache ``C(E ⊕ X)`` — a longer cache and enriched embeddings both.
    Table 1: ``|E| + |X|``, enriched, 63.39 %.

``oracle``
    Prefill ``E ⊕ X`` but retain only the question-aligned slice, Eq. (2)::

        C*(X) = C_[|E| : |E| + |X|](E ⊕ X)

    — enrichment without extending the cache length.
    Table 1: ``|X|``, enriched, 62.34 %.

Comparing ``direct`` and ``oracle`` isolates the semantic gain: any
improvement comes from the richer question embeddings induced by the
exemplars, not from attending to additional token caches (as in
``few_shot``).

Figure 4 (selective enrichment): :meth:`EnrichmentOracle.enrich_layers`
enriches only a chosen set of layers, in ascending order of the layer's
measured benefit; enriching the top-k best-performing layers yields
slightly higher accuracy than enriching all, while enriching the worst
ones declines accuracy — the observation that motivated the learnable
gates of :class:`c2c.fuser.modules.Gate`.

Example (doctest-friendly, using the miniature engines of the test
suite)::

    >>> from c2c.integrations.reference import ReferenceEngine
    >>> oracle = EnrichmentOracle(ReferenceEngine())
    >>> [r.method for r in oracle.run(exemplars=[0, 1], question=[2, 3])]
    ['direct', 'few-shot', 'oracle']
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..types import CacheInjector, CacheProvider, LayeredCache

__all__ = ["EnrichmentResult", "EnrichmentOracle", "METHODS"]

METHODS = ("direct", "few-shot", "oracle")


@dataclass(frozen=True)
class EnrichmentResult:
    """One measurement of the enrichement oracle."""

    method: str  # one of METHOD
    cache_len: int  # |C| actually used while decoding
    enriched: bool  # were exemplars present during prefill?
    answer: str  # what the receiver said
    score: float | None = None  # user-supplied scoring, if any
    meta: dict = field(default_factory=dict, compare=False)

    def __str__(self):
        sc = "n/a" if self.score is None else f"{self.score:0.2f}"
        return (
            f"{self.method:<9} │ cache_len={self.cache_len:>3} │ "
            f"enriched={str(self.enriched):<5} │ score={sc:>5}"
        )


class EnrichmentOracle:
    """The instrument: three operating modes, one frozen pair of models.

    Parameters
    ----------
    provider, injector:
        Must both speak for the same receiver model: ``provider`` captures
        ``C(X)`` after prefill (see :class:`c2c.types.CacheProvider`),
        ``injector`` installs a cache and decodes the reply (see
        :class:`c2c.types.CacheInjector`). Usually one object implements
        both interfaces (every engine adapter of :mod:`c2c.integrations`
        does).
    score:
        A user-supplied callable ``(question, answer) -> float``; the
        oracle makes no assumptions about it — the golden suite passes the
        benchmark scorers of :mod:`c2c.eval.benchmarks`.
    """

    def __init__(
        self,
        provider: CacheProvider,
        injector: CacheInjector | None = None,
        *,
        score: Callable[[Sequence[int], str], float] | None = None,
    ):
        if not hasattr(provider, "capture"):
            msg = "enrichment oracle requires a CacheProvider (it must provide .capture)"
            raise TypeError(msg)
        self.provider = provider
        self.injector = injector if injector is not None else provider
        if not hasattr(self.injector, "install") or not hasattr(self.injector, "generate"):
            msg = "enrichment oracle requires a CacheInjector (.install, .generate)"
            raise TypeError(msg)
        self.score = score

    # -- the three operating modes ───────────────────────────────────────────
    def _decode_with(
        self, cache: LayeredCache, prompt: Sequence[int], *, max_new_tokens: int
    ) -> str:
        self.injector.install(cache, list(prompt))
        return self.injector.generate(list(prompt), max_new_tokens=max_new_tokens, temperature=0.0)

    def direct(self, question: Sequence[int], *, max_new_tokens: int = 64) -> EnrichmentResult:
        """Prefill X, decode with C(X): the unenriched reference run."""
        cache = self.provider.capture(list(question))
        answer = self._decode_with(cache, question, max_new_tokens=max_new_tokens)
        return EnrichmentResult(
            "direct", cache.num_tokens, False, answer, self._score(question, answer)
        )

    def few_shot(
        self, exemplars: Sequence[int], question: Sequence[int], *, max_new_tokens: int = 64
    ) -> EnrichmentResult:
        """Prefill E ⊕ X, decode with C(E ⊕ X): longer cache *and* enrichment."""
        prompt = list(exemplars) + list(question)
        cache = self.provider.capture(prompt)
        answer = self._decode_with(cache, prompt, max_new_tokens=max_new_tokens)
        return EnrichmentResult(
            "few-shot", cache.num_tokens, True, answer, self._score(question, answer)
        )

    def oracle(
        self, exemplars: Sequence[int], question: Sequence[int], *, max_new_tokens: int = 64
    ) -> EnrichmentResult:
        """Prefill E ⊕ X, drop the exemplar span, decode with C*(X), Eq. (2).

        The retained slice is ``C_[|E| : |E|+|X|](E ⊕ X)`` — the cache rows
        aligned with the question, counted from the beginning of the cache.
        """
        prompt = list(exemplars) + list(question)
        whole = self.provider.capture(prompt)
        start, stop = len(exemplars), len(exemplars) + len(question)
        slice_ = LayeredCache(
            [
                type(row)(row.key[start:stop], row.value[start:stop])  # the question-aligned slice
                for row in whole
            ]
        )
        answer = self._decode_with(slice_, question, max_new_tokens=max_new_tokens)
        return EnrichmentResult(
            "oracle", slice_.num_tokens, True, answer, self._score(question, answer)
        )

    def run(
        self, exemplars: Sequence[int], question: Sequence[int], *, max_new_tokens: int = 64
    ) -> list[EnrichmentResult]:
        """All three modes in one pass — the Table 1 row for one (q, E)."""
        return [
            self.direct(question, max_new_tokens=max_new_tokens),
            self.few_shot(exemplars, question, max_new_tokens=max_new_tokens),
            self.oracle(exemplars, question, max_new_tokens=max_new_tokens),
        ]

    # -- Figure 4: selective enrichment layer by layer ───────────────────────
    def enrich_layers(
        self,
        exemplars: Sequence[int],
        question: Sequence[int],
        layers: Sequence[int],
        *,
        max_new_tokens: int = 64,
    ) -> EnrichmentResult:
        """Enrich exactly the given layers of the question-aligned cache.

        Layers is an iterable of layer indices; ascending order is not
        required — the result does not depend on the traversal order.
        Layers outside the cache range raise IndexError, as usual for
        sequences. The unenriched layers keep their own ``C(X)`` rows.
        """
        whole = self.provider.capture(list(exemplars) + list(question))
        base = self.provider.capture(list(question))
        start, stop = len(exemplars), len(exemplars) + len(question)
        merged = []
        seen: set[int] = set()
        for n, row in enumerate(base):
            if n in layers:
                if n >= len(whole):
                    msg = f"layer {n} outside cache range (cache has {len(whole)} layers)"
                    raise IndexError(msg)
                src = whole[n]
                merged.append(type(row)(src.key[start:stop], src.value[start:stop]))
                seen.add(n)
            else:
                merged.append(row)
        missing = set(layers) - seen
        if missing:
            msg = f"unknown layers requested for enrichment: {sorted(missing)}"
            raise ValueError(msg)
        cache = LayeredCache(merged)
        answer = self._decode_with(cache, question, max_new_tokens=max_new_tokens)
        return EnrichmentResult(
            "oracle(selective)",
            cache.num_tokens,
            True,
            answer,
            self._score(question, answer),
            meta={"enriched_layers": sorted(layers)},
        )

    # -- helpers ────────────────────────────────────────────────────────────
    def _score(self, question: Sequence[int], answer: str) -> float | None:
        if self.score is None:
            return None
        return float(self.score(list(question), answer))
