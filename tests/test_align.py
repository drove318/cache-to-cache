"""Tests for the two alignment levels (FR-09/10/11): tokens and layers.

§3.3.3: token alignment decodes, re-encodes, and resolves the one-to-many
case by maximal string coverage; layer alignment is terminal by default,
depth-normalised (Eq. 5) behind a flag. The published expectation — the
two collision strategies agreeing on more than 80 % of tokens — is a
test here, not a comment.
"""

from __future__ import annotations

import random

import pytest

from c2c.align.layers import LayerMapping, depth_normalized_mapping, terminal_mapping
from c2c.align.tokens import AlignedToken, TokenAligner, TokenAlignment
from c2c.integrations.reference import MiniatureTokenizer


@pytest.fixture()
def aligner():
    receiver = MiniatureTokenizer(variant="uni")
    sharer = MiniatureTokenizer(variant="bi")
    return TokenAligner(receiver, sharer, strategy="maximal-coverage")


class TestTokenAlignment:
    """Decode, re-encode, and resolve the collisions between the vocabularies."""

    def test_the_round_trip_survives_the_word_of_the_vocabulary(self):
        tok = MiniatureTokenizer()
        for text in ("the quick brown fox", "  spaces   ", "a"):
            assert tok.decode(tok.encode(text)) == text  # print it, to see it works

    def test_align_maps_the_receivers_tokens_onto_the_sharers(self):
        tok = MiniatureTokenizer()
        aligner = TokenAligner(tok, tok)  # the same vocabulary, for once
        rids = tok.encode("hello world")
        alignment = aligner.align(rids)
        assert isinstance(alignment, TokenAlignment)
        assert len(alignment) == len(rids)  # one aligned, per received
        assert all(isinstance(t, AlignedToken) for t in alignment)

    def test_one_to_many_collisions_are_resolved_by_maximal_coverage(self):
        """Where there is more than one candidate, the maximal wins."""

        class CoarseTokenizer:  # a vocabulary of whole words
            vocab = {"ab": 1, "abc": 3}
            _rev = {v: k for k, v in vocab.items()}
            all_special_ids = []  # attributes, not methods, as HF
            special_tokens = []
            unk_token = ""
            unk_token_id = 0

            def encode(self, text):
                return [self.vocab[text]] if text in self.vocab else [0]

            def encode_alternatives(self, text):
                """Every registered piece that can cover the string, longest first."""
                pieces = sorted(
                    (p for p in self.vocab if p.startswith(text)), key=len, reverse=True
                )
                alts = [[self.vocab[p]] for p in pieces]
                alts.append([0])  # the unknown, ever available
                return alts

            def decode(self, ids):
                return "".join(self._rev.get(i, "") for i in ids)

        fine = MiniatureTokenizer()  # character granularity
        rid = fine.encode("a")[0]  # the piece "a", seeded now
        aligner = TokenAligner(fine, CoarseTokenizer(), strategy="maximal-coverage")
        aligned = aligner._align_one(rid)  # the string "a": "ab", "abc" cover it
        assert aligned.text == "a"
        assert tuple(aligned.sharer_ids) == (3,)  # the maximal candidate, chosen
        assert aligned.method == "maximal-coverage"  # the strategy, recorded

    def test_unknown_tokens_fall_back_gracefully(self):
        """Where there is no candidate, the unknown token stands in."""
        receiver = MiniatureTokenizer()
        sharer = MiniatureTokenizer()
        aligner = TokenAligner(receiver, sharer)
        rid = receiver.encode("ξξ")[0]  # a Greek piece, out of the lexicon
        aligned = aligner.align([rid])[0]
        assert len(aligned.sharer_ids) >= 1  # never an empty mapping

    def test_special_tokens_are_mapped_directly(self):
        """The special characters, the specials, map one to one."""
        receiver = MiniatureTokenizer()
        sharer = MiniatureTokenizer()
        aligner = TokenAligner(receiver, sharer)
        pad_id = receiver.encode("[pad]")[0]
        aligned = aligner.align([pad_id])[0]
        assert aligned.method == "direct"  # a special: the direct map
        assert list(aligned.sharer_ids) == [sharer.encode("[pad]")[0]]

    def test_row_indices_select_the_rows_of_the_cache(self):
        """Position, not vocabulary: the fuser addresses caches by row."""
        tok = MiniatureTokenizer()
        aligner = TokenAligner(tok, tok)
        rids = tok.encode("the quick brown fox")
        rows = aligner.align(rids).row_indices()
        assert len(rows) == len(rids)
        assert all(isinstance(r, int) and r >= 0 for r in rows)

    def test_row_indices_without_a_row_map_is_a_clear_error(self):
        alignment = TokenAlignment([AlignedToken(1, "a", (1,), "direct")])
        with pytest.raises(LookupError, match="row map"):
            alignment.row_indices()  # vocabulary ids are not rows

    def test_chat_templates_are_aligned_section_by_section(self):
        """The message sections, semantically; the template, by length."""
        tok = MiniatureTokenizer()
        aligner = TokenAligner(tok, tok)
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello there"},
        ]
        sections = ("<|im start|>", "<|im sep|>assistant", "<|im end|>")
        aligned = aligner.align_chat(messages, template_sections=sections)
        assert len(aligned) == 2  # one record, one message
        for entry in aligned:
            assert isinstance(entry["aligned"], TokenAlignment)  # semantic, token by token
        padded = aligned[0]["template_padded"]
        lengths = {len(s.split()) for s in padded}
        assert lengths == {max(len(s.split()) for s in sections)}  # length-padded, equal

    @pytest.mark.parametrize("seed", [42, 7, 2026])
    def test_the_strategies_agree_over_eighty_percent(self, aligner, seed):
        """> 80 % identical alignments between the strategies (FR-10)."""
        rng = random.Random(seed)
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "cache",
            "token",
            "model",
            "layer",
            "key",
            "value",
            "attention",
        ]
        samples = [
            " ".join(rng.choice(words) for _ in range(rng.randint(1, 8))) for _ in range(300)
        ]
        tok = MiniatureTokenizer()
        rids = [rid for sample in samples for rid in tok.encode(sample)]
        agreement = aligner.strategy_agreement(rids)
        assert agreement > 0.80, f"the strategies agree on {agreement:0.1%}, not more than 80 %"

    def test_the_aligner_explains_itself(self, aligner):
        """A string, a table, the report of the alignment, printed."""
        tok = MiniatureTokenizer()
        text = aligner.explain(tok.encode("the cache"))
        assert "the" in text or "method" in text  # the columns, by name
        assert "\n" in text  # a table, not a line

    def test_illegal_strategy_is_a_value_error(self):
        tok = MiniatureTokenizer()
        with pytest.raises(ValueError, match="strategy"):
            TokenAligner(tok, tok, strategy="by-precedence")  # no such strategy


class TestLayerAlignment:
    """The terminal alignment: last layers first, in reverse order."""

    def test_terminal_alignment_pairs_the_last_layers_first(self):
        mapping = terminal_mapping(receiver_layers=5, sharer_layers=3)
        assert list(mapping.pairs())[-1] == (4, 2)  # the last, with the last
        assert list(mapping.reversed())[0] == (4, 2)  # then the penultimate, and so on
        assert list(mapping.reversed())[1] == (3, 1)

    def test_terminal_alignment_goes_back_to_the_first_layers(self):
        mapping = terminal_mapping(receiver_layers=5, sharer_layers=3)
        assert mapping[0] == 0  # the first, to the first's field
        assert mapping[-1] == 2  # the last, to the last, verbatim
        assert len(mapping) == 5  # every receiver layer, mapped

    def test_the_mapping_is_a_sequence_of_pairs(self):
        mapping = terminal_mapping(receiver_layers=2, sharer_layers=4)
        assert list(mapping) == [2, 3]  # clamped, but in range
        assert mapping == [2, 3]  # equal to the plain list

    def test_equal_length_models_align_pair_by_pair(self):
        mapping = terminal_mapping(receiver_layers=4, sharer_layers=4)
        assert list(mapping) == [0, 1, 2, 3]  # the same depth, the same layers

    def test_depth_normalized_mapping_is_the_published_function(self):
        """G(n) = round(n · (Ls − 1) / (Lr − 1)) — Eq. (5), in the numbers."""
        lr, ls = 29, 14
        mapping = depth_normalized_mapping(receiver_layers=lr, sharer_layers=ls)
        scale = (ls - 1) / (lr - 1)
        for n in range(lr):
            expected = min(max(int(n * scale + 0.5), 0), ls - 1)  # floor(x + ½), the half-away rule
            assert mapping[n] == expected, f"G({n})"
        assert mapping[-1] == ls - 1  # the last, to the last, again
        # hand-computed from Eq. (5), scale 13/28: 7→3.25→3, 14→6.5→7, 21→9.75→10
        assert [mapping[n] for n in (0, 7, 14, 21, 28)] == [0, 3, 7, 10, 13]

    def test_validate_reports_the_geometry_of_the_misfit(self):
        mapping = terminal_mapping(receiver_layers=4, sharer_layers=3)
        assert mapping.validate(receiver_layers=4, sharer_layers=3) == []
        problems = mapping.validate(receiver_layers=9, sharer_layers=2)
        assert len(problems) == 3  # both count, and range
        assert any("receiver has 9 layers" in p for p in problems)
        assert any("sharer has 2 layers" in p for p in problems)
        assert any("outside sharer range" in p for p in problems)

    def test_illegal_geometry_is_a_value_error(self):
        with pytest.raises(ValueError, match="at least one layer"):
            terminal_mapping(receiver_layers=0, sharer_layers=3)  # a mapping of nothing
        with pytest.raises(ValueError, match="at least one layer"):
            depth_normalized_mapping(receiver_layers=3, sharer_layers=0)

    def test_illegal_mode_is_a_value_error(self):
        with pytest.raises(ValueError, match="alignment mode"):
            LayerMapping(3, 3, mode="random")  # no such alignment
