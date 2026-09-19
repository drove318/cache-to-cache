"""The relay engine: the OpenAI front, usable out of the box.

The reference engine is the house nets and needs torch; a stranger who
``pip install``s the core wheel and starts ``c2c-serve`` would otherwise
meet an empty gallery and a wall of ``model_not_found``. This engine takes
the fall: it mirrors prompts back as answers, carries its own tokenizer,
and exposes cache hooks that are honest echoes, not intelligence. Both the
banner and the gallery say the word RELAY — fusion is real, but lives
behind the ``train`` extra, and the relay never pretends otherwise.

It is also the reference implementation of the serving contract: eight
methods, no weights, every adapter written after this one can be checked
against it piece by piece.
"""

from __future__ import annotations

import numpy as np

from ..types import AttentionKind, LayeredCache, LayerGeometry, LayerSlice, ModelSpec

__all__ = ["EchoEngine", "ECHO_GEOMETRY", "available"]

#: The geometry every relay speaks with: two layers, two heads, eight wide.
ECHO_GEOMETRY = LayerGeometry(
    layers=2,
    hidden_size=8,
    num_heads=2,
    num_key_value_heads=2,
    attention=AttentionKind.MHA,
    name="c2c-echo",
)

#: ids under this floor belong to the relay's own punctuation
_WORD_FLOOR = 16
_LEX_SPAN = 224


def _piece_id(word: str) -> int:
    """32-bit FNV-1a over the codepoints, folded into the word band."""
    h = 2166136261
    for ch in word:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return _WORD_FLOOR + h % _LEX_SPAN


def available() -> bool:
    """Always. This is the fallback the installation is allowed to have."""
    return True


class EchoEngine:
    """Mirror of the CacheProvider/CacheInjector contract, pure numpy.

    ``DEGRADATION`` names exactly what is missing and which line installs it,
    the way the FR-08 doctrine demands of every falling-back adapter.
    """

    DEGRADATION = (
        "relay: mirrors prompts, fuses nothing — "
        "pip install 'c2c-cache[train]' for the reference engine"
    )

    def __init__(self, model_id: str, options: dict | None = None):
        self.model_id = str(model_id)
        self.options = dict(options or {})
        self._lex: dict[int, str] = {}  # ids seen, remembered for the way back
        self._pending: LayeredCache | None = None

    # -- the card ───────────────────────────────────────────────────────────
    def spec(self) -> ModelSpec:
        return ModelSpec(
            id=self.model_id,
            geometry=ECHO_GEOMETRY,
            family="echo",
            size_billions=None,
            instruction_tuned=False,
            vocab_size=_WORD_FLOOR + _LEX_SPAN,
            context_length=64,
        )

    @classmethod
    def report_context(cls, model_id: str, **options) -> int:
        return 64

    # -- the tokenizer the front asks of every receiver ─────────────────────
    def encode(self, text: str, **_kw) -> list[int]:
        ids: list[int] = []
        for piece in str(text).split():
            pid = _piece_id(piece.lower())
            self._lex.setdefault(pid, piece)  # the original case rides home
            ids.append(pid)
        return ids or [_piece_id("echo")]  # never answer an empty ask

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return " ".join(self._lex.get(int(t), f"<{int(t)}>") for t in token_ids)

    # -- CacheProvider ──────────────────────────────────────────────────────
    def capture(self, prompt_tokens) -> LayeredCache:
        """Deterministic rows from the token ids: the same prompt, the same cache."""
        ids = np.asarray([float(int(t)) for t in prompt_tokens], dtype=np.float64)
        width = ECHO_GEOMETRY.num_key_value_heads * ECHO_GEOMETRY.head_size
        ramp = np.arange(1, width + 1, dtype=np.float64)
        rows = (ids[:, None] * ramp[None, :] % 97.0) / 97.0
        key = rows.tolist()
        value = ((rows + 0.5) % 1.0).tolist()
        return LayeredCache([LayerSlice(key, value) for _ in range(ECHO_GEOMETRY.layers)])

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache, prompt_tokens=None) -> None:
        self._pending = cache  # remembered, honoured in kind: the relay mirrors

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ) -> str:
        words = [self._lex.get(int(t), f"<{int(t)}>") for t in list(prompt_tokens)]
        answer = " ".join(words[: max(int(max_new_tokens), 1)])
        for cut in stop or ():
            if cut and cut in answer:
                answer = answer[: answer.find(cut)]
        return answer.strip()
