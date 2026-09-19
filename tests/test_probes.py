"""Tests for the two oracle probes (FR-11/12): §3.2's instruments.

Cache enrichment (Table 1: Direct |X| / Few-shot |E|+|X| / Oracle |X|)
and the cache-transformation oracle (Fig. 3: the transformed cache lies
within the target's representation space). Experimental instruments first.
"""

from __future__ import annotations

import pytest
import torch

from c2c.integrations.reference import MiniatureTokenizer, ReferenceConfig, ReferenceEngine
from c2c.probes.enrich import METHODS, EnrichmentOracle, EnrichmentResult
from c2c.probes.transform import TransformationOracle
from c2c.types import LayeredCache, LayerGeometry, LayerSlice


def engine(seed=42):
    return ReferenceEngine(ReferenceConfig(seed=seed))


class TestCacheEnrichmentOracle:
    """Table 1: Direct |X|, Few-shot |E|+|X|, Oracle |X| — the three modes."""

    def setup_method(self):
        self.oracle = EnrichmentOracle(engine(), engine())
        self.e = MiniatureTokenizer(variant="uni")
        self.q = self.e.encode("what is the capital of france")
        self.exemplars = self.e.encode("paris is the capital of france")

    def test_the_three_methods(self):
        assert METHODS == ("direct", "few-shot", "oracle")        # the modes, as published
        results = self.oracle.run(self.exemplars, self.q)
        assert [r.method for r in results] == list(METHODS)        # run, all of them, once

    def test_direct_uses_the_question_alone(self):
        result = self.oracle.direct(self.q)
        assert isinstance(result, EnrichmentResult)
        assert result.method == "direct"
        assert result.cache_len == len(self.q)                      # |X|, the budget, kept
        assert result.enriched is False                             # no exemplars, no enrichment
        assert isinstance(result.answer, str)

    def test_few_shot_carries_the_exemplars_in_the_cache(self):
        result = self.oracle.few_shot(self.exemplars, self.q)
        assert result.method == "few-shot"
        assert result.cache_len == len(self.exemplars) + len(self.q)   # |E| + |X|
        assert result.enriched is True

    def test_oracle_knows_the_answer_but_shares_only_the_question(self):
        """The oracle: exemplars in the prompt, |X| in the cache."""
        result = self.oracle.oracle(self.exemplars, self.q)
        assert result.method == "oracle"
        assert result.cache_len == len(self.q)                      # |X| — the exemplars dropped
        assert result.enriched is True                              # yet the answer enriched

    def test_enrichment_raises_the_question_above_the_baseline(self):
        """Direct < Oracle on the record; all three, in one table."""
        direct, few_shot, oracle = self.oracle.run(self.exemplars, self.q)
        assert direct.cache_len < few_shot.cache_len                # |X| < |E|+|X|
        assert oracle.cache_len == direct.cache_len                  # |X| = |X|
        assert oracle.enriched and not direct.enriched               # the flag, set right

    def test_the_instrument_needs_a_provider(self):
        with pytest.raises(TypeError):                                # not an engine, not a probe
            EnrichmentOracle("not an engine")

    def test_answers_are_generated_text(self):
        for result in self.oracle.run(self.exemplars, self.q):
            assert isinstance(result.answer, str)                     # text, always
            assert "\\n" not in result.answer                          # a line, at most? any line
        assert str(self.oracle.direct(self.q)).startswith("direct")   # readable, as printed


class TestTransformationOracle:
    """Fig. 3: fit the projector; transform; export the projection."""

    def _caches(self):
        """A source and a target: learnable in thirty epochs of gradient descent."""
        g = LayerGeometry(layers=2, hidden_size=8, num_heads=2)
        src = LayeredCache([
            LayerSlice(torch.full((6, g.kv_hidden_size), 0.2),
                    torch.full((6, g.kv_hidden_size), 0.1)) for _ in range(2)])
        tgt = LayeredCache([
            LayerSlice(torch.full((6, g.kv_hidden_size), 1.8),
                    torch.full((6, g.kv_hidden_size), 0.9)) for _ in range(2)])
        return src, tgt

    def test_the_oracle_learns_to_transform(self):
        """fit, learn, transform: the distance falls, the loss descends."""
        src, tgt = self._caches()
        oracle = TransformationOracle(epochs=600, lr=0.01, seed=42)
        result = oracle.fit(src, tgt)
        assert result.loss_curve, "no learning without loss"
        assert result.loss_curve[-1] <= result.loss_curve[0] + 1e-6   # it learns, verbatim
        assert result.distance_after < result.distance_before           # closer than before
        assert 0.0 <= result.purity <= 1.0                               # a measure, bounded

    def test_transform_maps_into_the_target_space(self):
        """The transformed cache, near the target: the figure of merit."""
        src, tgt = self._caches()
        oracle = TransformationOracle(epochs=600, lr=0.01, seed=42)
        oracle.fit(src, tgt)
        moved = oracle.transform(src)
        assert isinstance(moved, LayeredCache)
        raw = (src[0].key - tgt[0].key).abs().mean()
        near = (moved[0].key - tgt[0].key).abs().mean()
        assert float(near) < float(raw)                               # the t-SNE claim, in numbers

    def test_an_unfitted_oracle_says_so(self):
        """Raise the roof before you fit: the oracle, honest."""
        src, _ = self._caches()
        with pytest.raises(LookupError, match="before fitting"):
            TransformationOracle().transform(src)                      # untrained, unpublished

    def test_the_projection_exports_a_figure(self, tmp_path):
        """Fig. 3, as data: a CSV of the projection, with the header line."""
        src, tgt = self._caches()
        oracle = TransformationOracle(epochs=8, seed=42)
        oracle.fit(src, tgt)
        path = str(tmp_path / "figure3.csv")
        assert oracle.export(src, tgt, path=path, method="pca") == path   # written, where told
        with open(path, encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        assert lines[0] == "x,y,kind"                                    # the header, as documented
        kinds = {line.split(",")[2] for line in lines[1:]}
        assert kinds == {"source", "target", "transformed"}               # the three series

    def test_tsne_without_the_optional_peer_raises(self):
        """An optional dependency, missed: a clear message, not a silent fallback."""
        src, tgt = self._caches()
        oracle = TransformationOracle(epochs=4, seed=42)
        oracle.fit(src, tgt)
        with pytest.raises(ModuleNotFoundError, match="t-SNE"):
            oracle.export(src, tgt, path="/tmp/c2c-figure-tsne.csv", method="tsne")

    def test_equal_dimensions_equal_results(self):
        """The same data, the same fit: determinism, checked twice."""
        src, tgt = self._caches()
        a = TransformationOracle(epochs=8, seed=7).fit(src, tgt)
        b = TransformationOracle(epochs=8, seed=7).fit(src, tgt)
        assert a.loss_curve == b.loss_curve                            # reproducible, by design
        assert str(a) == str(b)                                        # printed, the same
