"""Ollama adapter — HTTP control plane (degrades, documented).

Ollama serves ``POST /api/generate`` (and an OpenAI-compatible subtree).
Its wire protocol carries prompt *strings* and no cache, so this adapter:

* degrades to prefill-only capture and documents the degradation
  (``DEGRADATION``; surfaced by ``c2c doctor``);
* maintains a self-consistent, reversible token codec (below): the ABI
  speaks token ids everywhere, so the adapter keeps a piece table — a
  growing table of unique strings, assigned sequential integer ids, one
  entry per whitespace-delimited piece (separators kept as their own
  pieces so ``decode(encode(text)) == text`` holds for every text the
  adapter itself encodes).

For the full cache-to-cache experience with an Ollama-hosted model, put
it behind ``c2c-serve``: the front speaks OpenAI to the harness and
keeps the cache machinery on this side of the wire.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from ..types import LayeredCache, LayerGeometry, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter

__all__ = ["OllamaAdapter", "available"]

_SPLIT = re.compile(r"\S+|\s+")


def available() -> bool:
    return True  # needs only an endpoint


class OllamaAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for a running Ollama server."""

    engine_name = "Ollama"
    required_extra = None
    DEGRADATION = (
        "Ollama's wire protocol carries no cache: prefill-only capture; "
        "put the model behind c2c-serve for the full experience"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        self.base_url = str(self.options.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = float(self.options.get("timeout", 60.0))
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._degraded = False

    # -- the reversible codec ───────────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        """Split into pieces and assign each a sequential id, keeping order."""
        ids: list[int] = []
        for piece in _SPLIT.findall(text or ""):
            pid = self._piece_to_id.get(piece)
            if pid is None:
                pid = len(self._piece_to_id) + 1  # ids start at one, like in BPE
                self._piece_to_id[piece] = pid
                self._id_to_piece[pid] = piece
            ids.append(pid)
        return ids

    def decode_tokens(self, token_ids) -> str:
        """Map each id back to its piece and join them, without separators."""
        return "".join(self._id_to_piece.get(int(t), "") for t in token_ids)

    # -- HTTP plumbing ──────────────────────────────────────────────────────
    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError) as exc:
            raise AdapterNotSupported(
                f"Ollama server at {self.base_url} unreachable",
                hint="run `ollama serve`, or point --base-url at your server",
            ) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            msg = f"Ollama returned malformed JSON at {path}: {exc.msg}"
            raise ValueError(msg) from exc

    # -- protocol ───────────────────────────────────────────────────────────
    def _build_spec(self) -> ModelSpec:
        info = self._post("/api/show", {"model": self.model_id})
        arch = "ollama"
        try:
            arch = str(info["model_info"]["general"]["architecture"]) or "ollama"
        except (KeyError, TypeError):
            pass
        return ModelSpec(
            id=self.model_id,
            geometry=LayerGeometry(layers=1, hidden_size=1, num_heads=1, name=self.model_id),
            family=arch,
            instruction_tuned=True,
        )

    def capture(self, prompt_tokens):
        """No cache crosses this wire; report the documented degradation."""
        self._degraded = True
        return LayeredCache([])

    def install(self, cache, prompt_tokens=None):
        self._pending = None  # nothing to install on HTTP

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        prompt = self.decode_tokens(prompt_tokens)  # the codec restores the string
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "raw": False,
            "stream": False,
            "options": {
                "num_predict": int(max_new_tokens),
                "temperature": float(temperature or 0.0),
                **({"stop": list(stop)} if stop else {}),
            },
        }
        out = self._post("/api/generate", payload)
        return str(out.get("response", ""))

    def available_capabilities(self):
        return super().available_capabilities()
