"""The reference engine — a miniature model, complete and deterministic.

This module is the test bed of the whole project: a small but real
causal Transformer LM, implemented in pure PyTorch, that speaks the full
engine ABI (:class:`c2c.types.CacheProvider` **and**
:class:`c2c.types.CacheInjector`) with *lossless* cache capture and
re-installation. Everything the library promises about cache-to-cache
communication — prefill, capture, fusion, install-before-decode, greedy
and sampled generation, teacher-forced scoring — is exercised end to end
against this engine in the test-suite, without GPUs, without downloads,
and without any third-party engine.

Determinism: the engine seeds its own generator with ``config.seed``
(default 42); two instances with the same configuration and the same seed
produce bit-identical caches and replies (release gate, spec §5).

Layout of cache slices: one layer's cache is stored as two tensors,
``key`` and ``value``, each of shape ``[n_tokens, num_heads, head_size]``
— heads are rows, features are columns, and the token axis leads.

Tokenizer: a miniature, in the spirit of the real ones — a growing
vocabulary of word pieces with special tokens first. Two granularity
variants (``uni``, ``bi``) deliberately disagree in how words are split;
whitespace is kept as its own pieces, so ``decode(encode(text)) == text``
holds for every ``text`` (round-trip identity, asserted in the tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..types import (AttentionKind, LayerGeometry, LayeredCache, LayerSlice,
                     ModelSpec)
from .registry import EngineAdapter

__all__ = ["ReferenceConfig", "MiniatureTokenizer", "ReferenceEngine", "ReferenceAdapter"]

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ("[pad]", "[unk]", "bos", "eos")   # special tokens first, as usual
_BASE = len(SPECIAL_TOKENS)                            # first piece-id after the specials
_SPLIT_RE = re.compile(r"\S+|\s+")                    # words and the spaces between them


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration space of the reference engine (every default is sane)."""

    name: str = "reference-mini"
    family: str = "reference"
    layers: int = 4
    hidden_size: int = 32
    num_heads: int = 4
    max_seq_length: int = 2048
    mlp_ratio: int = 4
    vocab_limit: int = 8192
    seed: int = 42
    instruction_tuned: bool = True
    size_billions: float = 0.0000005


class MiniatureTokenizer:
    """A tiny, deterministic, whitespace-preserving tokenizer.

    Parameters
    ----------
    variant:
        ``"uni"`` maps every character of a word to its own piece;
        ``"bi"`` prefers bigrams, falling back to a single piece for the
        final odd character. The two granularities produce different token
        counts for the same string — a deliberate divergence that forces
        one-to-many collisions and lets the aligner align them (see
        :mod:`c2c.align.tokens`).
    """

    def __init__(self, *, variant: str = "uni", max_pieces: int = 8192):
        if variant not in ("uni", "bi"):
            msg = f"unknown tokenizer variant {variant!r}; choose 'uni' or 'bi'"
            raise ValueError(msg)
        self.variant = variant
        self.max_pieces = max_pieces
        self._piece_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self._id_to_piece: dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        self._next_id = _BASE

    # -- the growing vocabulary of pieces ───────────────────────────────────
    def _add_piece(self, piece: str) -> int:
        pid = self._piece_to_id.get(piece)
        if pid is not None:
            return pid
        if len(self._piece_to_id) >= self.max_pieces + _BASE:
            return UNK                                    # vocabulary full: fall back to unk
        pid = self._next_id
        self._next_id += 1
        self._piece_to_id[piece] = pid
        self._id_to_piece[pid] = piece
        return pid

    def _split_word(self, word: str) -> list[str]:
        if self.variant == "uni":
            return list(word)
        pieces: list[str] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word):
                pieces.append(word[i:i + 2])             # prefer a bigram, as usual
                i += 2
            else:
                pieces.append(word[i])                   # the final odd character goes alone
                i += 1
        return pieces

    # -- encode and decode, round trip ──────────────────────────────────────
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text`` into a list of token ids (whitespace is kept).

        The special tokens match first, whole and exact — as in any
        self-respecting byte-oriented tokenizer: ``[pad]`` is one token,
        never five characters.
        """
        ids: list[int] = [BOS] if add_bos else []
        for run in _SPLIT_RE.findall(text or ""):
            if run in SPECIAL_TOKENS:
                ids.append(self._piece_to_id[run])         # a special, whole, registered
            elif run.isspace():
                ids.append(self._add_piece(run))         # a space is a piece, too
            else:
                for piece in self._split_word(run):
                    ids.append(self._add_piece(piece))
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_specials: bool = True) -> str:
        """Decode token ids back to the string they were encoded from.

        Control tokens, by default, are dropped: ``decode`` skips the
        pad/unk/bos sentinels and stops reading at eos — the printed text
        excludes control characters. Pass ``skip_specials=False`` to keep
        them printable, as the aligner does when it must see them.
        """
        out: list[str] = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (PAD, UNK, BOS):
                if skip_specials:
                    continue
            elif tid == EOS and skip_specials:
                break
            piece = self._id_to_piece.get(tid)
            if piece is not None:
                out.append(piece)
        return "".join(out)

    # -- introspection, for the aligner and the CLI ─────────────────────────
    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._piece_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self._piece_to_id)

    @property
    def all_special_ids(self) -> list[int]:
        return list(range(len(SPECIAL_TOKENS)))

    @property
    def special_tokens(self) -> list[str]:
        return list(SPECIAL_TOKENS)

    @property
    def unk_token(self) -> str:
        return "[unk]"

    @property
    def unk_token_id(self) -> int:
        return UNK


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a read/write-able cache.

    Two modes: :meth:`prefill` processes the whole prompt and returns the
    freshly made cache slices; :meth:`decode` appends one new row to each.
    Caches are 3-dimensional, ``[num_heads, n_tokens, head_size]`` — the
    batch is one and is materialised only at the scaled-dot-product
    attention call, where the framework expects it.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            msg = f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, self.num_heads * self.head_size, bias=False)
        self.v = nn.Linear(hidden_size, self.num_heads * self.head_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """[n, hidden] → [num_heads, n, head_size]."""
        n = tensor.shape[0]
        return tensor.reshape(n, self.num_heads, self.head_size).transpose(0, 1)

    def prefill(self, x: torch.Tensor):
        """``x``: [T, hidden] → (out [T, hidden], k [h,T,hs], v [h,T,hs])."""
        q = self._heads(self.q(x))
        k = self._heads(self.k(x))
        v = self._heads(self.v(x))
        t = k.shape[1]
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        attn = nn.functional.scaled_dot_product_attention(
            q[None], k[None], v[None], attn_mask=mask[None, None], is_causal=False)
        out = attn[0].transpose(0, 1).reshape(t, self.hidden_size)
        return self.o(out), k, v

    def decode(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """``x``: [1, hidden]; caches [h, T_prev, hs] → (out, k, v) extended."""
        q = self._heads(self.q(x))                          # [h, 1, hs]
        k = torch.cat((k_cache, self._heads(self.k(x))), dim=1)
        v = torch.cat((v_cache, self._heads(self.v(x))), dim=1)
        attn = nn.functional.scaled_dot_product_attention(
            q[None], k[None], v[None], is_causal=False)     # a query of length one needs no mask
        out = attn[0].transpose(0, 1).reshape(-1, self.hidden_size)
        return self.o(out), k, v

    def score(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Teacher-forced attention over an installed prefix cache.

        ``x``: [T, hidden]; caches ``[h, T_prev, hs]``. Every query sees
        the whole prefix — it lies in the past — and, within the
        response, its own position and those before it: the causal
        triangle, shifted by the prefix. No graph is severed here; the
        autograd path runs through the installed rows to the fuser,
        which is the whole point of the exercise.
        """
        q = self._heads(self.q(x))                          # [h, T, hs]
        k = torch.cat((k_cache, self._heads(self.k(x))), dim=1)
        v = torch.cat((v_cache, self._heads(self.v(x))), dim=1)
        t = k.shape[1] - k_cache.shape[1]
        prefix = torch.ones(t, k_cache.shape[1], dtype=torch.bool, device=x.device)
        within = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        mask = torch.cat((prefix, within), dim=1)[None, None]
        attn = nn.functional.scaled_dot_product_attention(
            q[None], k[None], v[None], attn_mask=mask, is_causal=False)
        out = attn[0].transpose(0, 1).reshape(t, self.hidden_size)
        return self.o(out)


class _Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = _CausalSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.SiLU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def prefill(self, x: torch.Tensor):
        a, k, v = self.attn.prefill(self.norm1(x))
        x = x + a
        return x + self.mlp(self.norm2(x)), k, v

    def decode(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        a, k, v = self.attn.decode(self.norm1(x), k_cache, v_cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), k, v

    def score(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        a = self.attn.score(self.norm1(x), k_cache, v_cache)
        x = x + a
        return x + self.mlp(self.norm2(x))


class ReferenceEngine(nn.Module):
    """The little engine that could: capture, install, generate, score.

    A :class:`~c2c.types.CacheProvider` and :class:`~c2c.types.CacheInjector`
    in one person — for ease of testing and for use as the fallback engine
    (``--engine reference``) on machines without a real runtime attached.
    """

    def __init__(self, config: ReferenceConfig | None = None, *, tokenizer_variant: str = "uni"):
        super().__init__()
        self.config = config or ReferenceConfig()
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads:
            cfg = ReferenceConfig(**{**vars(cfg), "num_heads": 1})
        self.geometry = LayerGeometry(
            layers=cfg.layers, hidden_size=cfg.hidden_size, num_heads=cfg.num_heads,
            attention=AttentionKind.MHA, name=cfg.name,
        )
        torch.manual_seed(cfg.seed)                        # determinism first (FR-14)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.embed = nn.Embedding(cfg.vocab_limit, cfg.hidden_size)
        self.position = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [_Block(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.layers)]
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_limit, bias=True)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02, generator=self.generator)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02, generator=self.generator)
        self._installed: tuple[LayeredCache | None, list[int] | None] = (None, None)
        self.tokenizer: MiniatureTokenizer | None = None
        if tokenizer_variant != "uni":
            self.set_tokenizer(MiniatureTokenizer(variant=tokenizer_variant))
        self.eval()

    def _device(self) -> torch.device:
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    # -- CacheProvider ──────────────────────────────────────────────────────
    def spec(self) -> ModelSpec:
        return ModelSpec(
            id=self.config.name,
            geometry=self.geometry,
            family=self.config.family,
            size_billions=self.config.size_billions,
            instruction_tuned=self.config.instruction_tuned,
            vocab_size=self._active_vocab(),
        )

    def capture(self, prompt_tokens: Sequence[int]) -> LayeredCache:
        """Prefill ``prompt_tokens`` and return the per-layer cache rows."""
        if not prompt_tokens:
            return LayeredCache([])
        ids = torch.as_tensor([int(t) % self.config.vocab_limit for t in prompt_tokens],
                            dtype=torch.long, device=self._device())
        with torch.no_grad():
            x = self.embed(ids) + self.position(
                torch.arange(int(ids.shape[-1]), device=ids.device)
                .clamp(max=self.config.max_seq_length - 1))
            slices: list[LayerSlice] = []
            for block in self.blocks:
                x, k, v = block.prefill(x)
                slices.append(LayerSlice(k.transpose(0, 1), v.transpose(0, 1)))
        return LayeredCache(slices)

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache: LayeredCache, prompt_tokens: Sequence[int] | None = None) -> None:
        """Install a (possibly fused) cache, to be used by the next generation."""
        if not isinstance(cache, LayeredCache) or len(cache) != self.config.layers:
            got = len(cache) if isinstance(cache, LayeredCache) else "?"
            msg = (f"cannot install: expected a LayeredCache of {self.config.layers} "
                  f"layers, got {type(cache).__name__} with {got}")
            raise ValueError(msg)
        self._installed = (cache, list(prompt_tokens) if prompt_tokens is not None else None)

    def score(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Teacher-forced logits; row ``t`` predicts the token ``t + 1``.

        The contract is set by :mod:`c2c.train.scheme`: one pass, no side
        effects, no randomisation — the same ids always score the same
        logits. When a cache is installed (the fused C(X) of training,
        the shared cache of serving), scoring conditions on it: queries
        read the installed rows and the autograd path runs through them
        back to the fuser. No ``no_grad`` may sever that path; callers
        that want inference-only wrapping wrap it themselves.
        """
        ids = [int(t) % self.config.vocab_limit for t in token_ids]
        if not ids:
            msg = "score requires at least one token"
            raise ValueError(msg)
        tok = torch.as_tensor(ids, dtype=torch.long, device=self._device())
        installed_cache, _prompt = self._installed
        if (installed_cache is not None and len(installed_cache) == len(self.blocks)
                and installed_cache.num_tokens > 0):
            position0 = installed_cache.num_tokens           # after the prefix, come on
            pos = (torch.arange(len(ids), device=tok.device) + position0
                ).clamp(max=self.config.max_seq_length - 1)
            x = self.embed(tok) + self.position(pos)
            for i, block in enumerate(self.blocks):
                k_c = installed_cache[i].key.transpose(0, 1)     # [h, T, hs]
                v_c = installed_cache[i].value.transpose(0, 1)
                x = block.score(x, k_c, v_c)                 # read the fused rows
        else:
            pos = torch.arange(len(ids), device=tok.device)
            x = self.embed(tok) + self.position(
                pos.clamp(max=self.config.max_seq_length - 1))
            for block in self.blocks:
                x, _k, _v = block.prefill(x)                 # the pass, plain
        return self.lm_head(x)                              # [T, V]

    def generate(self, prompt_tokens: Sequence[int], *, max_new_tokens: int = 64,
                 temperature: float = 0.0, tools: Sequence[dict] | None = None,
                 stop: Sequence[str] | None = None) -> str:
        """Decode a continuation; stop at EOS or at the first stop string.

        ``tools`` is accepted for ABI compatibility and passed through by
        the serving layer to the real Receiver; this miniature engine
        yields plain text only and ignores it (a documented degradation,
        printed by ``c2c doctor``).
        """
        if max_new_tokens <= 0:
            return ""
        prompt = [int(t) % self.config.vocab_limit for t in prompt_tokens]
        installed_cache, installed_prompt = self._installed
        if installed_cache is not None and installed_prompt == prompt:
            caches = LayeredCache([LayerSlice(r.key.clone(), r.value.clone())
                                 for r in installed_cache])
        else:
            caches = self.capture(prompt)
        produced: list[int] = []
        if not caches:
            caches = self.capture([BOS])                 # condition on BOS, as usual
            position, next_id = 1, BOS
        else:
            position = len(prompt)
            next_id = prompt[-1] if prompt else BOS
        with torch.no_grad():
            for _ in range(max_new_tokens):
                tok = torch.as_tensor([next_id], dtype=torch.long, device=self._device())
                pos = torch.as_tensor([min(position, self.config.max_seq_length - 1)],
                                   dtype=torch.long, device=self._device())
                x = self.embed(tok) + self.position(pos)   # [1, hidden]
                for i, block in enumerate(self.blocks):
                    k_c = caches[i].key.transpose(0, 1)     # [h, T, hs]
                    v_c = caches[i].value.transpose(0, 1)
                    x, k_new, v_new = block.decode(x, k_c, v_c)
                    caches[i] = LayerSlice(k_new.transpose(0, 1), v_new.transpose(0, 1))
                logits = self.lm_head(x)                    # [1, V]
                logits = logits[:, :max(self._active_vocab(), _BASE)]
                if temperature and temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1,
                                                 generator=self.generator).item())
                else:
                    next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == EOS:
                    break
                produced.append(next_id)
                position += 1
        if stop:
            text = self.decode_tokens(produced)
            for s in stop:
                if s and s in text:
                    text = text[:text.find(s)]
            return text
        return self.decode_tokens(produced)

    # -- tokenizer delegation (the engine owns at most one) ─────────────────
    def set_tokenizer(self, tokenizer: MiniatureTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            self.set_tokenizer(MiniatureTokenizer())
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        if self.tokenizer is None:
            self.set_tokenizer(MiniatureTokenizer())
        return self.tokenizer.decode(list(tokens))

    def _active_vocab(self) -> int:
        """Number of pieces the head may answer with.

        An untrained head could reach entries outside the grown piece
        table, so the candidate set is truncated to the pieces actually
        in the table (plus the specials) — as every sampler does.
        """
        if self.tokenizer is not None:
            return max(self.tokenizer.vocab_size, _BASE)
        return _BASE


class ReferenceAdapter(EngineAdapter):
    """Registry-facing wrapper of :class:`ReferenceEngine`.

    Registered under the name ``reference`` in the engine group
    ``c2c.engines``; instantiated by the registry with the options given on
    the command line. The wrapper delegates every cache operation to the
    engine it holds; ``close`` releases it.
    """

    engine_name = "reference"
    DEGRADATION = None                                     # full, lossless capture

    def __init__(self, model_id: str = "reference-mini", **options):
        super().__init__(model_id, **options)
        variant = self.options.get("variant", "uni")
        seed = int(self.options.get("seed", 42))
        cfg = ReferenceConfig(
            name=str(model_id), seed=seed,
            layers=int(self.options.get("layers", 4)),
            hidden_size=int(self.options.get("hidden", 32)),
            num_heads=int(self.options.get("heads", 4)),
        )
        self.engine = ReferenceEngine(cfg, tokenizer_variant=variant)

    # -- protocol delegation, one to one (the wrapper pattern) ─────────────
    def _build_spec(self) -> ModelSpec:
        return self.engine.spec()

    def capture(self, prompt_tokens):
        return self.engine.capture(prompt_tokens)

    def install(self, cache, prompt_tokens=None):
        return self.engine.install(cache, prompt_tokens)

    def generate(self, prompt_tokens, **kwargs):
        return self.engine.generate(prompt_tokens, **kwargs)

    def score(self, token_ids):
        return self.engine.score(token_ids)

    def encode(self, text):
        return self.engine.encode(text)

    def decode_tokens(self, token_ids):
        return self.engine.decode_tokens(token_ids)

    def close(self):
        self.engine = None
