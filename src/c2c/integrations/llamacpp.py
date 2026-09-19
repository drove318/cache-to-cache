"""llama.cpp adapter — per-engine shim (degrades, documented).

The ``llama_cpp`` Python bindings expose model loading and sampling but
keep the KV cache behind C++ handles; no public cache hook exists in the
bindings. This adapter generates through the bindings' completion API and
degrades to prefill-only capture, documenting the degradation — the same
contract every shim obeys (spec §4.1).

Requires the optional peer ``llama-cpp-python``; import is deferred.
"""

from __future__ import annotations

from ..types import LayeredCache, LayerGeometry, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter, truncated

__all__ = ["LlamaCppAdapter", "available"]


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("llama_cpp") is not None


class LlamaCppAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for llama.cpp-embedded models."""

    engine_name = "llama.cpp"
    required_extra = "llama-cpp"
    TOOLS = "ignored"  # the generate accepts them, the engine ignores them
    DEGRADATION = (
        "the bindings keep the cache behind opaque handles: "
        "prefill-only capture; fusion reduced to the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'llama-cpp' adapter needs llama-cpp-python",
                hint="pip install 'c2c-cache[llama-cpp]' or build the bindings",
            )
        from importlib import import_module

        self._lc = import_module("llama_cpp")
        self._ctx = None
        self._model = None
        self._degraded = False

    def _ensure(self):
        if self._ctx is None:
            model = self._lc.Model.from_file(
                self.model_id, vocab=self.options.get("vocab"), **self.options.get("model", {})
            )
            ctx = model.create_context(**self.options.get("context", {}))
            self._model, self._ctx = model, ctx
        return self._ctx

    def _build_spec(self) -> ModelSpec:
        ctx = self._ensure()
        n_layer = int(getattr(ctx, "n_layer", 1) or 1)
        n_head = int(getattr(ctx, "n_head", 1) or 1)
        n_embd = int(getattr(ctx, "n_embd", 1) or 1)
        return ModelSpec(
            id=self.model_id,
            geometry=LayerGeometry(
                layers=n_layer,
                hidden_size=max(n_embd, 1),
                num_heads=max(n_head, 1),
                name=(
                    self.model_id
                    if any(getattr(ctx, f, None) for f in ("n_layer", "n_head", "n_embd"))
                    else f"{self.model_id} (geometry unknown — the bindings tell none)"
                ),
            ),
            family="llama.cpp",
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def capture(self, prompt_tokens):
        self._degraded = True
        return LayeredCache([])  # opaque handles, documented

    def install(self, cache, prompt_tokens=None):
        self._pending = None  # nothing public, nothing to install

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        ctx = self._ensure()
        text = self.decode_tokens(prompt_tokens)
        sampler = ctx.get_default_sampler(
            mapping=self._lc.DEFAULT_SAMPLER_MAPPING, temperature=float(temperature or 0.0)
        )
        response = sampler.complete(
            ctx, prompt=text, max_tokens=int(max_new_tokens), stop=list(stop) if stop else None
        )
        return truncated(str(response), stop)  # the bindings may not honour stop; we do

    def encode(self, text: str):
        ctx = self._ensure()
        return ctx.tokenize(text, add_bos=True, special=self._lc.DEFAULT_SPECIAL_TOKENS)

    def decode_tokens(self, token_ids):
        ctx = self._ensure()
        return ctx.detokenize(list(token_ids), skip_special_tokens=True)
