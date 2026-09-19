"""Token alignment — the same string, two vocabularies (paper §3.3.3, FR-10).

Given a token sequence from the *Receiver* tokenizer, produce, for every
receiver token, the sharer token(s) encoding the very same string. Three
rules, published in the paper and implemented here:

1. one-to-many collisions — when the sharer tokenizer splits the string
   into several candidates, select by **maximal coverage** (the candidate
   whose decoded string covers the original the most, measured in string
   length); fall back to the **first occurrence** if coverage ties;
2. special tokens — map directly when both vocabularies agree on them,
   otherwise fall back to the sharer's *unknown* token;
3. chat templates — template sections are aligned by length padding
   (``<pad>``), message sections are aligned semantically, token by
   token (App. A.1.2).

The published expectation: the two selection strategies agree on more
than 80 % of alignments — :meth:`TokenAligner.strategy_agreement`
measures it, ``c2c.eval`` asserts it; deviations are treated as bugs.

Two spellings of the same protocol: tokenizers expose ``decode``; the
cache injectors of the engine adapters expose ``decode_tokens``. The
aligner accepts either, see :func:`_resolve_decode`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

__all__ = ["TokenizerLike", "AlignedToken", "TokenAligner", "TokenAlignment"]


class TokenizerLike(Protocol):
    """The tokenizer interface every aligner must handle.

    Compatible with :class:`transformers.PreTrainedTokenizer` instances
    and with the miniature implementations used by the test-suite (see
    :class:`c2c.integrations.reference.MiniatureTokenizer`).
    """

    def decode(self, token_ids: Sequence[int]) -> str:
        """Return the string encoded by exactly the given token ids."""
        ...

    def encode(self, text: str) -> list[int]:
        """Return a list of token ids encoding exactly the given string."""
        ...


def _resolve_decode(tokenizer):
    """Pick the decode callable: ``.decode`` first, ``.decode_tokens`` alias.

    Tokenizers speak ``decode``; cache injectors of the engine adapters
    speak ``decode_tokens``. The aligner accepts both spellings — one
    interface, two implementations, same meaning (of the string).
    """
    fn = getattr(tokenizer, "decode", None)
    if fn is None:
        fn = getattr(tokenizer, "decode_tokens", None)
    if fn is None:
        msg = f"{type(tokenizer).__name__} can not decode: no .decode nor .decode_tokens"
        raise TypeError(msg)
    return fn


def _resolve_encode(tokenizer):
    fn = getattr(tokenizer, "encode", None)
    if fn is None:
        msg = f"{type(tokenizer).__name__} can not encode: no .encode"
        raise TypeError(msg)
    return fn


@dataclass(frozen=True)
class AlignedToken:
    """One receiver token, mapped onto its candidate sharer token ids.

    ``method`` records how the choice was made — invaluable for
    diagnostics (``c2c fuse --report`` prints it):

    ``direct``            a special token both sides agree on;
    ``maximal-coverage``  one-to-many resolved by string length;
    ``first-occurrence``  tie (or empty coverage) broken by first occurrence;
    ``unknown``           no candidate at all, the sharer's unk token stands in.
    """

    receiver_id: int
    text: str
    sharer_ids: tuple[int, ...]
    method: str

    def __str__(self):
        ids = ", ".join(str(i) for i in self.sharer_ids)
        return f"{self.receiver_id:>5} | {self.text!r:<24} | [{ids}] | {self.method}"


class TokenAlignment(Sequence):
    """A complete alignment: an immutable, iterable sequence of
    :class:`AlignedToken`, with the conveniences of slicing."""

    def __init__(self, tokens: Sequence[AlignedToken], row_map: Sequence[int] | None = None):
        self._tokens = list(tokens)
        self._rows = list(row_map) if row_map is not None else None

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TokenAlignment(
                self._tokens[index], None if self._rows is None else self._rows[index]
            )
        return self._tokens[index]

    def __repr__(self):
        return f"TokenAlignment({len(self._tokens)} aligned tokens)"

    def row_indices(self) -> list[int]:
        """The ``token_mapping`` the fuser consumes (cache-row addressing).

        For every receiver *row* the index of the sharer cache row whose
        string covers the beginning of that row's token — position
        numbers, not vocabulary ids. Baked in by :meth:`TokenAligner.align`;
        a bare TokenAlignment built by hand must pass the map explicitly.
        """
        if self._rows is None:
            msg = (
                "no row map: construct via TokenAligner.align(...) or pass "
                "row_map=... — vocabulary ids are not row indices"
            )
            raise LookupError(msg)
        return list(self._rows)


class TokenAligner:
    """Align receiver token sequences onto a sharer vocabulary.

    The aligner is stateless apart from its configuration; two instances
    with the same arguments always produce the same alignment (determinism
    first — the golden suite relies on it).
    """

    STRATEGIES = ("maximal-coverage", "first-occurrence")

    def __init__(
        self,
        receiver: TokenizerLike,
        sharer: TokenizerLike,
        *,
        strategy: str = "maximal-coverage",
        pad_token: str = "<pad>",
        max_candidates: int = 64,
    ):
        if strategy not in self.STRATEGIES:
            msg = (
                f"unknown collision-resolution strategy {strategy!r}; "
                f"choose one of {self.STRATEGIES}"
            )
            raise ValueError(msg)
        self.receiver = receiver
        self.sharer = sharer
        self.strategy = strategy
        self.pad_token = pad_token
        self.max_candidates = max_candidates
        self._dec_r = _resolve_decode(receiver)
        self._dec_s = _resolve_decode(sharer)
        self._enc_s = _resolve_encode(sharer)
        self._specials_r = self._special_map_of(receiver)
        self._specials_s = self._special_map_of(sharer)
        self._unk = self._unknown_id_of(sharer)

    # -- helper functions (introspection of either vocabulary) ─────────────
    @staticmethod
    def _special_map_of(tokenizer) -> dict[int, str]:
        """Collect the special tokens of a tokenizer, best effort."""
        mapping: dict[int, str] = {}
        ids = getattr(tokenizer, "all_special_ids", None) or ()
        tokens = getattr(tokenizer, "special_tokens", None) or ()
        for rid, token in zip(ids, tokens):
            try:
                mapping[int(rid)] = str(token)
            except (TypeError, ValueError):
                continue
        extra_ids = getattr(tokenizer, "additional_special_ids", None) or ()
        extra = getattr(tokenizer, "additional_special_tokens", None) or ()
        for rid, token in zip(extra_ids, extra):
            try:
                mapping.setdefault(int(rid), str(token))
            except (TypeError, ValueError):
                continue
        return mapping

    @staticmethod
    def _unknown_id_of(tokenizer) -> int | None:
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is None:
            unk_token = getattr(tokenizer, "unk_token", None)
            if unk_token is not None:
                convert = getattr(tokenizer, "convert_tokens_to_ids", None)
                if convert is not None:
                    try:
                        unk = convert([unk_token])[0]
                    except (AttributeError, TypeError, IndexError):
                        unk = None
        return None if unk is None else int(unk)

    def _candidates(self, text: str) -> list[tuple[int, ...]]:
        """Encode ``text`` once, enumerate all the candidate encodings.

        The default encoding yields a single candidate; tokenizers with
        sub-word ambiguity may provide ``encode_alternatives`` to produce
        the set of alternative segmentations (the one-to-many case).
        """
        alternative = getattr(self.sharer, "encode_alternatives", None)
        if alternative is not None:
            cands = [tuple(c) for c in alternative(text)[: self.max_candidates]]
            if cands:
                return cands
        return [tuple(self._enc_s(text))][: self.max_candidates]

    def _coverage(self, ids: tuple[int, ...]) -> int:
        """The coverage of a candidate: length of its decoded string."""
        try:
            return len(self._dec_s(list(ids)))
        except Exception:  # must not crash alignment
            return 0

    # -- the alignment proper ───────────────────────────────────────────────
    def _align_one(self, rid: int) -> AlignedToken:
        text = self._dec_r([rid])
        # (2) special tokens: direct-map where the vocabularies agree …
        if rid in self._specials_r:
            token = self._specials_r[rid]
            for sid, stoken in self._specials_s.items():
                if stoken == token:
                    return AlignedToken(rid, text, (sid,), "direct")
            if self._unk is not None:  # … else the unk token stands in
                return AlignedToken(rid, text, (self._unk,), "unknown")
            # no unk in the sharer's vocabulary: encode the piece, the last
            # resort — never pass a raw receiver id into a sharer vocabulary
            return AlignedToken(rid, text, tuple(self._enc_s(text))[:1] or (0,), "unknown")
        # (1) general tokens: the one-to-many case
        if text == "":  # an empty piece stands for the space that joins words
            return AlignedToken(rid, text, tuple(self._enc_s(" ")), "first-occurrence")
        candidates = self._candidates(text)
        if not candidates:
            fallback = (
                ((self._unk,),)
                if self._unk is not None
                else (tuple(self._enc_s(text))[:1] or (0,),)
            )
            return AlignedToken(rid, text, tuple(fallback[0]), "unknown")
        if self.strategy == "first-occurrence":
            best = candidates[0]
            method = "first-occurrence"
        else:
            best = max(candidates, key=self._coverage)  # ties keep the first occurrence
            best_cov = self._coverage(best)
            ties = sum(1 for c in candidates if self._coverage(c) == best_cov)
            method = "maximal-coverage" if ties == 1 else "first-occurrence"
        return AlignedToken(rid, text, tuple(best), method)

    def align(self, receiver_ids: Sequence[int]) -> TokenAlignment:
        """Align a whole receiver token sequence (the main entry point)."""
        ids = [int(r) for r in receiver_ids]
        return TokenAlignment([self._align_one(r) for r in ids], self.select_rows(ids))

    def align_text(self, text: str) -> TokenAlignment:
        """Encode with the receiver, then align — convenience for tests/CLI."""
        return self.align(self.receiver.encode(text))

    def select_rows(self, receiver_ids: Sequence[int]) -> list[int]:
        """Map receiver row numbers onto sharer row numbers (§3.3.3).

        The fuser addresses caches by *position*: the result gives, for
        every receiver token row, the index of the sharer cache row whose
        decoded string covers the start of that row's token. Single pass
        over both tokenisations using cumulative character positions —
        O(|C_r| + |C_s|), no intermediate allocation worth mentioning.
        """
        ids = [int(r) for r in receiver_ids]
        texts = [self._dec_r([r]) for r in ids]
        whole = "".join(texts)
        sharer_ids = list(self._enc_s(whole))
        m = len(sharer_ids)
        if m == 0:
            return [0] * len(ids)
        starts_s: list[int] = []
        pos = 0
        for sid in sharer_ids:
            starts_s.append(pos)
            pos += len(self._dec_s([sid]))
        starts_s.append(pos)
        rows: list[int] = []
        j = 0
        a = 0
        for text in texts:
            while j + 1 < m and starts_s[j + 1] <= a:
                j += 1
            rows.append(j)
            a += len(text)
        return rows

    # -- chat template alignment (App. A.1.2) ───────────────────────────────
    def align_chat(
        self, messages: Sequence[dict], *, template_sections: Sequence[str] | None = None
    ):
        """Align a chat conversation between the two vocabularies.

        *Template sections* (control tokens such as ``<|im_start|>``) are
        aligned by **length padding**: the section is extended with
        repetitions of the pad token so both sides see shells of equal
        piece count; *message sections* (the user/assistant content) are
        aligned **semantically**, token by token, through :meth:`align`.

        Returns a list of ``{"role": …, "aligned": TokenAlignment}`` with
        the template spans kept verbatim in ``"template"`` and padded in
        ``"template_padded"``.
        """
        out: list[dict] = []
        for message in messages:
            content = message.get("content", "")
            tokens = self.receiver.encode(content) if content else []
            aligned = self.align(tokens)
            entry = {"role": message.get("role", "user"), "aligned": aligned}
            if template_sections:
                entry["template"] = tuple(template_sections)
                lengths = [len(s.split()) for s in template_sections]
                target = max(lengths)
                entry["template_padded"] = tuple(
                    sec if n == target else sec + " " + " ".join([self.pad_token] * (target - n))
                    for sec, n in zip(template_sections, lengths)
                )
            out.append(entry)
        return out

    # -- diagnostics of the selection process ───────────────────────────────
    def strategy_agreement(self, receiver_ids: Sequence[int]) -> float:
        """Fraction of identical alignments between the two strategies.

        The published expectation is > 0.80 (``c2c.eval`` asserts it with
        the paper's numbers; deviations are bugs).
        """
        a = self.align(receiver_ids)
        other = TokenAligner(
            self.receiver,
            self.sharer,
            strategy=(
                "first-occurrence" if self.strategy == "maximal-coverage" else "maximal-coverage"
            ),
            pad_token=self.pad_token,
            max_candidates=self.max_candidates,
        )
        b = other.align(receiver_ids)
        if not a:
            return 1.0
        same = sum(1 for x, y in zip(a, b, strict=True) if x.sharer_ids == y.sharer_ids)
        return same / len(a)

    def explain(self, receiver_ids: Sequence[int]) -> str:
        """A human-readable, aligned report of one alignment run."""
        rows = [str(t) for t in self.align(receiver_ids)]
        header = f"{'rid':>5} | {'string':<24} | candidates             | method"
        return "\n".join([header, "-" * len(header), *rows])
