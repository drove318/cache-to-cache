"""Tests for the reference engine — the miniature, deterministic model.

It implements the whole path a cache can take: capture, install, score,
generate — at a thousandth the size, and a hundredth the speed. Every
e2e test in the suite leans on it.
"""

from __future__ import annotations

import pytest
import torch

from c2c.integrations.reference import (
    BOS,
    EOS,
    SPECIAL_TOKENS,
    MiniatureTokenizer,
    ReferenceAdapter,
    ReferenceConfig,
    ReferenceEngine,
)
from c2c.types import LayeredCache, ModelSpec


def _engine(seed=42, variant="uni", **cfg):
    return ReferenceEngine(ReferenceConfig(seed=seed, **cfg), tokenizer_variant=variant)


class TestTheTokenizer:
    """Words, spaces, and the pieces between them."""

    def test_the_round_trip_of_the_word(self):
        tok = MiniatureTokenizer()
        for text in ("hello world", "a b c", "  double  space  "):
            assert tok.decode(tok.encode(text)) == text  # the word, entire

    def test_the_two_granularities_differ_in_count(self):
        """uni against bi: the same text, in two measures."""
        text = "compression"
        uni_tok, bi_tok = MiniatureTokenizer(variant="uni"), MiniatureTokenizer(variant="bi")
        uni, bi = uni_tok.encode(text), bi_tok.encode(text)
        assert uni and bi
        assert len(uni) != len(bi)  # a deliberate divergence
        assert uni_tok.decode(uni) == text  # the one, entire
        assert bi_tok.decode(bi) == text  # the other, the same

    def test_specials_are_special(self):
        tok = MiniatureTokenizer()
        for piece in SPECIAL_TOKENS:  # punctuation, all of them
            ident = tok.encode(piece)[0]  # one token, whole
            assert tok.decode([ident], skip_specials=False) == piece  # the same, both ways

    def test_unknown_pieces_map_to_the_unknown(self):
        """When the vocabulary is full, the unknown token stands in."""
        tok = MiniatureTokenizer(max_pieces=0)  # no room, no pieces
        ids = tok.encode("日本語の")  # out of the lexicon
        assert ids and all(i == tok.unk_token_id for i in ids)  # all, the unknown

    def test_the_empty_string_reads_as_the_empty_list(self):
        assert MiniatureTokenizer().encode("") == []

    def test_the_growing_vocab_of_pieces(self):
        """The vocabulary grows with the texts it has seen."""
        tok = MiniatureTokenizer()
        before = tok.vocab_size
        tok.encode("sesquipedalian")  # a new word, today
        assert tok.vocab_size > before  # the lexicon, larger

    def test_bos_and_eos_are_added_by_request(self):
        """add_bos/add_eos: the sentinels, from the constants."""
        tok = MiniatureTokenizer()
        ids = tok.encode("go", add_bos=True, add_eos=True)
        plain = tok.encode("go")
        assert ids[0] == BOS and ids[-1] == EOS  # head and tail
        assert ids[1:-1] == plain  # the body, untouched


class TestTheEngine:
    """A little engine that could: capture, install, generate, score."""

    def test_the_model_card_reads_true(self):
        spec = _engine().spec()
        assert isinstance(spec, ModelSpec)  # a card, a real one
        assert spec.id == "reference-mini"  # named, as configured
        assert spec.geometry.layers == 4  # four layers, deep
        assert spec.geometry.hidden_size == 32  # thirty-two wide
        assert spec.vocab_size > 0  # a vocabulary, some

    def test_capture_returns_the_layered_cache(self):
        engine = _engine()
        ids = engine.encode("the capital of france")
        cache = engine.capture(ids)
        assert isinstance(cache, LayeredCache)
        assert len(cache) == engine.spec().geometry.layers  # one slice, one layer
        assert cache.num_tokens == len(ids)  # one row, one token
        first = cache[0]
        heads, head_size = engine.spec().geometry.num_heads, engine.spec().geometry.head_size
        assert first.key.shape == (len(ids), heads, head_size)  # heads, preserved (FR-06)
        assert first.value.shape == first.key.shape  # the pair, complete

    def test_capture_of_nothing_captures_nothing(self):
        assert len(_engine().capture([])) == 0  # the empty prompt

    def test_scoring_matrix_of_the_logits(self):
        """score: the logits, row by row, of the vocabulary."""
        engine = _engine()
        ids = engine.encode("rank and file")
        logits = engine.score(ids)
        assert logits.dim() == 2  # a matrix, not a vector
        assert logits.shape[0] == len(ids)  # one row, per position
        assert torch.isfinite(logits).all()  # the numbers, real

    def test_generation_is_deterministic(self):
        """The same prompt, twice: the same completion, verbatim."""
        engine = _engine()
        ids = engine.encode("what is the capital of france")
        a = engine.generate(ids, max_new_tokens=8)
        b = engine.generate(ids, max_new_tokens=8)
        assert a == b  # reproducibility, checked
        assert isinstance(a, str)

    def test_generation_respects_the_token_budget(self):
        """max_new_tokens: a budget, kept."""
        engine = _engine()
        ids = engine.encode("write a long essay about the weather")
        out = engine.generate(ids, max_new_tokens=3)
        produced = engine.encode(out) if out else []
        assert len(produced) <= 3 + 1  # the budget, honoured

    def test_the_empty_prompt_generates_a_response(self):
        engine = _engine()
        assert isinstance(engine.generate([], max_new_tokens=4), str)

    def test_installing_a_foreign_cache(self):
        """install: the cache, in; the generation, on."""
        engine = _engine()
        ids = engine.encode("install gentoo")
        cache = engine.capture(ids)
        engine.install(cache, ids)  # no return, no error
        out = engine.generate(ids, max_new_tokens=5)
        assert isinstance(out, str)

    def test_installing_rubbish_raises_rubbish(self):
        """install: not a cache, not installed."""
        with pytest.raises((TypeError, ValueError)):
            _engine().install(["this", "is", "not", "a", "cache"])

    def test_the_two_engines_grow_alike(self):
        """The same seed: the same weights, the same answers."""
        a, b = _engine(seed=7), _engine(seed=7)
        text = "determinism above all"
        ids, ids_b = a.encode(text), b.encode(text)  # both lexicons, seeded alike
        assert ids == ids_b  # the same pieces, both ways
        assert a.generate(ids, max_new_tokens=6) == b.generate(ids_b, max_new_tokens=6)
        assert torch.equal(a.score(ids), b.score(ids))  # bit for bit

    def test_a_different_seed_different_model(self):
        a, b = _engine(seed=1), _engine(seed=2)
        ids = a.encode("entropy")
        assert not torch.equal(a.score(ids), b.score(ids))  # the weights, apart

    def test_temperature_zero_is_greedy(self):
        """T=0: no sampling, no surprises."""
        engine = _engine()
        ids = engine.encode("the rain in spain")
        first = engine.generate(ids, max_new_tokens=6, temperature=0.0)
        again = engine.generate(ids, max_new_tokens=6, temperature=0.0)
        assert first == again  # greedy, twice

    def test_stop_strings_stop_the_string(self):
        """The stop sequence, honoured."""
        engine = _engine()
        ids = engine.encode("tell me a story about a king")
        out = engine.generate(ids, max_new_tokens=32, stop=["the"])
        assert "the" not in out.lower() or out == ""  # stopped, as told


class TestTheAdapter:
    """The registry-facing wrapper: the same engine, in a suit."""

    def test_the_adapter_delegates_every_operation(self):
        adapter = ReferenceAdapter("suited", seed=5)
        spec = adapter.spec()
        assert spec.id == "suited"  # the name, through
        ids = adapter.encode("delegate")
        cache = adapter.capture(ids)
        assert isinstance(cache, LayeredCache)  # the cache, intact
        adapter.install(cache, ids)  # the install, done
        assert isinstance(adapter.generate(ids, max_new_tokens=3), str)
        assert adapter.score(ids).dim() == 2  # the score, on

    def test_the_adapter_closes_cleanly(self):
        adapter = ReferenceAdapter("closing", seed=1)
        adapter.close()  # resources, released
        assert adapter.engine is None  # the engine, gone

    def test_the_config_of_the_engine_defaults_to_the_paper(self):
        cfg = ReferenceConfig()
        assert cfg.seed == 42 and cfg.layers == 4 and cfg.hidden_size == 32
        assert cfg.num_heads == 4 and cfg.max_seq_length == 2048  # the protocol, sized
