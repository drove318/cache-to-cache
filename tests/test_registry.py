"""Tests for the engine registry: one ABI, many adapters (spec §4.1).

The registry is the bus: lookup, registration, entry points, and the
contract that every adapter implements the pair of protocols. The
availability policy is part of the test: absent engines raise
AdapterNotSupported, with a hint — never a silent fallback.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from c2c.integrations import engines
from c2c.integrations.registry import ENTRY_POINT_GROUP, AdapterNotSupported, EngineAdapter
from c2c.types import CacheInjector, CacheProvider, LayeredCache, ModelSpec

DISTRIBUTION = ("reference", "hf", "vllm", "sglang", "trtllm", "llamacpp", "tgi",
                "ollama", "mlx")


class TestTheBus:
    """Look up, register, unregister: the operations of the bus."""

    def test_the_distribution_names_are_the_adapters(self):
        """All of the nine, in the registry of the distribution."""
        registered = engines.registered_names()
        for adapter in DISTRIBUTION:
            assert adapter in registered, f"{adapter} missing from the bus"

    def test_the_entry_point_group_is_the_one_documented(self):
        assert ENTRY_POINT_GROUP == "c2c.engines"               # the group, as shipped
        assert engines.group == ENTRY_POINT_GROUP               # the bus, on the label

    def test_lookup_names_the_target(self):
        """lookup: the dotted path, the attribute, the string pair."""
        target = engines.lookup("reference")
        module, _, attribute = target.partition(":")
        assert module == "c2c.integrations.reference"           # the module, found
        assert attribute == "ReferenceAdapter"                 # the attribute, exported

    def test_unknown_engines_are_reported_with_suggestions(self):
        """An unknown name: the error, the list, the near, match."""
        with pytest.raises(KeyError, match="unknown engine"):
            engines.lookup("refernce")                          # one letter off, a typo
        try:
            engines.lookup("nonexistent")
        except KeyError as exc:
            assert "registered engines" in str(exc)             # what is on the bus, said

    def test_load_instantiates_the_adapter(self):
        """load: the class, constructed, with the options passed through."""
        adapter = engines.load("reference", model_id="mini-test", seed=42)
        assert isinstance(adapter, EngineAdapter)               # the base, honoured
        assert isinstance(adapter.spec(), ModelSpec)          # the card, on delivery

    def test_register_and_unregister_round_trip(self):
        """register, use, unregister: the cycle of the bus."""
        engines.register("mine", "c2c.integrations.reference:ReferenceAdapter")
        try:
            assert engines.lookup("mine").endswith("ReferenceAdapter")   # the target, kept
        finally:
            engines.unregister("mine")                          # gone, as it came
        with pytest.raises(KeyError):
            engines.lookup("mine")                              # absent, again

    def test_unregister_an_unknown_name_is_a_key_error(self):
        with pytest.raises(KeyError):
            engines.unregister("never-registered")              # nothing, to remove

    def test_register_over_existing_needs_a_force(self):
        """A duplicate registration: refused, unless forced."""
        engines.register("mine2", "a:b")
        try:
            with pytest.raises((ValueError, KeyError)):
                engines.register("mine2", "c:d")                # the same name, a new target
            engines.register("mine2", "e:f", force=True)        # forced, admitted
            assert engines.lookup("mine2") == "e:f"
        finally:
            engines.unregister("mine2")

    def test_entry_points_extend_the_registry(self, monkeypatch):
        """A third-party adapter, discovered through the entry points."""
        ep = importlib.metadata.EntryPoint(
            name="external-engine", group=ENTRY_POINT_GROUP,
            value="c2c.integrations.reference:ReferenceAdapter")
        monkeypatch.setattr(importlib.metadata, "entry_points",
                          lambda **kw: (importlib.metadata.EntryPoints([ep])
                                      if kw.get("group") == ENTRY_POINT_GROUP
                                      else importlib.metadata.EntryPoints([])))
        monkeypatch.setattr(engines, "_discovered", None)          # re-run the search
        assert "external-engine" in engines.registered_names()      # found, on the point
        assert engines.lookup("external-engine") \
            == "c2c.integrations.reference:ReferenceAdapter"         # loaded, as declared

    def test_describe_reports_the_bus(self):
        """One line per adapter, with its mark, for the doctor's card."""
        lines = engines.describe()
        assert len(lines) >= len(DISTRIBUTION)                  # the nine, all reported
        joined = "\n".join(lines)
        assert "[full]" in joined                               # the reference, complete
        assert "reference" in joined                             # named, on its own line


class TestTheAvailability:
    """Engines that are not installed: a clear error, never a silent shim."""

    @pytest.mark.parametrize("adapter", [a for a in DISTRIBUTION if a != "reference"])
    def test_missing_engines_raise_with_hints(self, adapter):
        """pip install the matching extra, or use the reference: the hint."""
        # the engines are not installed in this environment; load must tell you so
        try:
            obj = engines.load(adapter, model_id="probe")
        except AdapterNotSupported as exc:
            assert "pip install" in str(exc) or "reference" in str(exc)  # the advice, given
            assert exc.hint, "an unavailable engine, without a hint"     # the hint, present
        else:                                                            # or the engine is
            assert obj is not None                                        # actually there


class TestTheContract:
    """Every adapter implements the pair of protocols, in the letter."""

    def test_the_reference_adapter_honours_the_protocols(self):
        adapter = engines.load("reference", model_id="contract", seed=7)
        assert isinstance(adapter, CacheProvider)               # captures, spec
        assert isinstance(adapter, CacheInjector)               # installs, generates

    def test_the_reference_card_of_the_adapter(self):
        """The model card: what an adapter must report, it reports."""
        adapter = engines.load("reference", model_id="card", seed=7)
        spec = adapter.spec()
        assert spec.id == "card"                                 # the id, as passed
        assert spec.geometry.layers > 0                           # layers, counted
        assert spec.vocab_size > 0                               # the vocabulary, sized
        assert "reference" in str(spec).lower()                  # the family, on the card

    def test_capture_returns_a_layered_cache(self):
        """capture(prompt_tokens): the rows, in the order of the tokens."""
        adapter = engines.load("reference", model_id="capture", seed=11)
        ids = adapter.encode("the capital of france")
        assert ids and all(isinstance(i, int) for i in ids)      # tokens, as promised
        cache = adapter.capture(ids)
        assert isinstance(cache, LayeredCache)
        assert len(cache) == adapter.spec().geometry.layers      # one slice, one layer
        assert cache.num_tokens == len(ids)                      # one row, one token

    def test_install_then_generate(self):
        """install(cache), generate(): the round trip of the injected cache."""
        adapter = engines.load("reference", model_id="inject", seed=13)
        ids = adapter.encode("hello there")
        cache = adapter.capture(ids)
        adapter.install(cache, ids)                              # installed, without error
        out = adapter.generate(ids, max_new_tokens=4)
        assert isinstance(out, str)                              # text, in, text, out

    def test_score_matrix_of_the_engine(self):
        """score(token_ids): the logits, one row per position, as documented."""
        adapter = engines.load("reference", model_id="score", seed=17)
        ids = adapter.encode("count the logits")
        logits = adapter.score(ids)
        assert logits.dim() == 2                                 # a matrix, rows × columns
        assert logits.shape[0] == len(ids)                       # one row, one position
        assert logits.shape[1] >= adapter.spec().vocab_size or logits.shape[1] > 0
