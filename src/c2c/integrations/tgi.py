"""TGI adapter — HTTP control plane (degrades, documented).

Text-Generation-Inference speaks HTTP; its wire protocol (``POST
/generate``) carries tokens and sampling parameters but no cache. This
adapter therefore generates over the network through the standard TGI
API and falls back to prefill-only capture; the degradation is
documented. The proxy front (``c2c-serve``) remains the recommended way
to bring TGI models into a cache-to-cache deployment: register TGI as
the *sharer's* transport via an OpenAI-compatible base URL and let a
cache-aware engine do the fusing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..types import LayeredCache, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter

__all__ = ["TGIAdapter", "available"]


def available() -> bool:
    """TGI needs no client package — an endpoint URL is enough."""
    return True


class TGIAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for a running TGI server."""

    engine_name = "TGI"
    required_extra = None
    DEGRADATION = ("TGI's wire protocol carries no cache: prefill-only capture; "
                   "fusion reduced to the receiver's own cache")

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        self.base_url = str(self.options.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.timeout = float(self.options.get("timeout", 30.0))

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},   # application/json, as per RFC
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise AdapterNotSupported(f"TGI server at {self.base_url} unreachable",
                                    hint="start it, or point --base-url at it") from exc

    def _build_spec(self) -> ModelSpec:
        self._post("/info", {})  # liveness probe: unreachable raises AdapterNotSupported
        return ModelSpec(
            id=self.model_id,
            geometry=self.options.get("geometry") or _unknown_geometry(),
            family="tgi",
            instruction_tuned=bool(self.options.get("instruction_tuned", True)))

    def capture(self, prompt_tokens):
        self._degraded = True
        return LayeredCache([])                            # documented degradation

    def install(self, cache, prompt_tokens=None):
        self._pending = None

    def generate(self, prompt_tokens, *, max_new_tokens: int = 64, temperature: float = 0.0,
                 tools=None, stop=None):
        params = {"max_new_tokens": int(max_new_tokens), "decoder_input_ids":
                  list(prompt_tokens)}
        if temperature and temperature > 0.0:
            params["temperature"] = float(temperature)
        if stop:
            params["stop_sequences"] = list(stop)
        out = self._post("/generate", {"inputs": "", "parameters": params})
        return str(out.get("generated_text", out.get("token", {}).get("text", "")))

    def encode(self, text: str):
        out = self._post("/encode", {"text": text})
        return list(out.get("token_ids", out.get("ids", [])))

    def decode_tokens(self, token_ids):
        out = self._post("/decode", {"token_ids": list(token_ids)})
        return str(out.get("decoded_token", out.get("tokens", "")))

    def available_capabilities(self):
        return super().available_capabilities()


def _unknown_geometry():
    from ..types import LayerGeometry
    return LayerGeometry(layers=1, hidden_size=1, num_heads=1)
