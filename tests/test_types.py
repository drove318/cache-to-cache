"""Tests for the core data model: types, protocols, and the small helpers."""

from __future__ import annotations

import pytest
import torch

from c2c.types import (AttentionKind, BlendDirection, CacheInjector, CacheProvider,
                       FusionReport, LayeredCache, LayerGeometry, LayerSlice,
                       ModelSpec, concat_rows, select)


class TestLayerGeometry:
    def test_head_size_is_derived_from_the_model_card(self):
        g = LayerGeometry(layers=4, hidden_size=8, num_heads=2)
        assert g.head_size == 4                          # 8 = 2 × 4, as on the card
        assert g.d == 2 * g.kv_hidden_size               # key ‖ value, the joint vector

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="invalid geometry"):
            LayerGeometry(layers=0, hidden_size=8)
        with pytest.raises(ValueError, match="invalid geometry"):
            LayerGeometry(layers=4, hidden_size=-1)

    def test_invalid_head_configuration_raises(self):
        with pytest.raises(ValueError, match="invalid head configuration"):
            LayerGeometry(layers=2, hidden_size=7, num_heads=0)

    def test_non_positive_kv_heads_raise(self):
        with pytest.raises(ValueError, match="key-value heads"):
            LayerGeometry(layers=2, hidden_size=4, num_heads=2, num_key_value_heads=0)

    def test_gqa_records_the_smaller_kv_head_count(self):
        g = LayerGeometry(layers=2, hidden_size=4, num_heads=4, num_key_value_heads=1)
        assert g.kv_hidden_size == 1 * g.head_size       # one KV head shared by four
        assert g.attention is AttentionKind.MHA          # the field, as declared

    def test_describe_prints_a_model_card_line(self):
        g = LayerGeometry(layers=3, hidden_size=6, num_heads=3, name="Mini")
        line = g.describe()
        assert "Mini" in line and "3 layers" in line and "3 heads" in line


class TestLayerSlice:
    def test_length_is_the_number_of_rows(self):
        slc = LayerSlice(torch.zeros(5, 4), torch.ones(5, 4))
        assert len(slc) == 5 == slc.num_tokens

    def test_map_returns_new_slices(self):
        slc = LayerSlice(torch.ones(2, 2), torch.ones(2, 2))
        doubled = slc.map(lambda t: t * 2)
        assert doubled is not slc
        assert torch.equal(doubled.key, torch.full((2, 2), 2.0))
        assert torch.equal(doubled.value, torch.full((2, 2), 2.0))

    def test_concat_is_sequencewise(self):
        a = LayerSlice(torch.ones(2, 3), torch.ones(2, 3))
        b = LayerSlice(torch.zeros(3, 3), torch.zeros(3, 3))
        assert a.concat(b).num_tokens == 5

    def test_sub_yields_the_difference(self):
        a = LayerSlice(torch.ones(2, 2), torch.ones(2, 2))
        b = LayerSlice(torch.zeros(2, 2), torch.zeros(2, 2))
        delta = a - b
        assert torch.equal(delta.key, torch.ones(2, 2))


class TestLayeredCache:
    def _make(self, layers=3, tokens=2):
        return LayeredCache([
            LayerSlice(torch.ones(tokens, 2), torch.ones(tokens, 2))
            for _ in range(layers)])

    def test_is_a_sequence_of_slices(self):
        cache = self._make()
        assert len(cache) == 3
        assert isinstance(cache[1], LayerSlice)
        assert isinstance(cache[0:2], LayeredCache) and len(cache[0:2]) == 2
        assert list(reversed(cache))[0] is cache[2]      # the terminal traversal
        assert cache                                   # truthy, like any list

    def test_setitem_replaces_a_layer(self):
        cache = self._make(layers=2)
        cache[0] = LayerSlice(torch.zeros(2, 2), torch.zeros(2, 2))
        assert torch.equal(cache[0].key, torch.zeros(2, 2))

    def test_setitem_checks_the_types(self):
        cache = self._make(layers=1)
        with pytest.raises(TypeError, match="expected a LayerSlice"):
            cache[0] = "not a slice"

    def test_append_grows_the_cache(self):
        cache = self._make(layers=1)
        cache.append(LayerSlice(torch.ones(2, 2), torch.ones(2, 2)))
        assert len(cache) == 2

    def test_concat_requires_equal_layers(self):
        with pytest.raises(ValueError, match="layer count mismatch"):
            self._make(layers=1).concat(self._make(layers=2))

    def test_num_tokens_counts_the_rows(self):
        assert self._make(tokens=7).num_tokens == 7
        assert LayeredCache().num_tokens == 0            # the empty cache, as it must be

    def test_zeros_makes_a_zero_initialised_cache(self):
        cache = LayeredCache.zeros(layers=2, tokens=3, kv_hidden=4)
        assert len(cache) == 2 and cache.num_tokens == 3
        assert torch.equal(cache[0].key, torch.zeros(3, 4))
        assert torch.equal(cache[0].value, torch.zeros(3, 4))

    def test_map_is_not_destructive(self):
        cache = self._make()
        doubled = cache.map(lambda t: t * 2)
        assert torch.equal(cache[0].key, torch.ones(2, 2))    # the original survives
        assert torch.equal(doubled[0].key, torch.full((2, 2), 2.0))


class TestConcatRows:
    """The dispatch table: one call, four data types."""

    def test_tuples(self):
        assert concat_rows((1, 2), (3,)) == (1, 2, 3)

    def test_lists(self):
        assert concat_rows([1], [2, 3]) == [1, 2, 3]

    def test_tensors(self):
        out = concat_rows(torch.ones(2, 2), torch.zeros(1, 2))
        assert tuple(out.shape) == (3, 2)

    def test_numpy_arrays(self):
        import numpy as np
        out = concat_rows(np.ones((2, 2)), np.zeros((2, 2)))
        assert out.shape == (4, 2)

    def test_mixed_unrelated_types_raise(self):
        with pytest.raises(TypeError, match="can not concat"):
            concat_rows([1, 2], "34")

    def test_none_is_no_argument(self):
        with pytest.raises(TypeError):
            concat_rows(None, [1])


class TestSelect:
    """select: from candidates, the one with maximal key coverage."""

    def test_maximal_by_key(self):
        assert select(["a", "bbb", "cc"], key=len) == "bbb"

    def test_ties_favour_the_first(self):
        assert select(["ab", "cd"], key=len) == "ab"

    def test_empty_sequence_without_default_raises(self):
        with pytest.raises(ValueError, match="empty sequence"):
            select([])
        assert select([], default=7) == 7

    def test_predicate_filters_every_candidate(self):
        with pytest.raises(ValueError, match="predicate"):
            select([1, 2], require=lambda x: x > 10)
        assert select([1, 2], default=0, require=lambda x: x > 10) == 0


class TestProtocolsAndReports:
    def test_reference_engine_satisfies_both_protocols(self):
        from c2c.integrations.reference import ReferenceEngine
        engine = ReferenceEngine()
        assert isinstance(engine, CacheProvider)
        assert isinstance(engine, CacheInjector)

    def test_model_spec_prints_a_card(self):
        g = LayerGeometry(layers=2, hidden_size=4, num_heads=2)
        spec = ModelSpec(id="x-mini", geometry=g, family="test")
        text = str(spec)
        assert "x-mini" in text and "test" in text and "instruct" in text

    def test_fusion_report_reads_like_a_card(self):
        report = FusionReport(num_tokens=4, gate_values=[0.9, 0.1],
                           effective_rank={"key": {"before": 3.0, "after": 4.0}})
        text = str(report)
        assert "4 tokens" in text
        assert "open" in text and "closed" in text          # the gates, by name
        assert "effective rank" in text                       # the dimension, measured

    def test_blend_directions_are_two(self):
        assert str(BlendDirection.FORMER) == "former"
        assert str(BlendDirection.LATTER) == "latter"
        assert BlendDirection("latter") is BlendDirection.LATTER
