"""SGLang adapter — RadixAttention cache hooks.

Mechanism (spec §4.1): SGLang keeps its prefix cache in a radix tree
(RadixAttention); the runtime exposes cache hooks that let this adapter
walk the tree and read the key/value rows of a matched prefix. Where the
hooks are absent (older or stripped-down builds), the adapter falls back
to prefill-only capture and documents the degradation.

Requires the ``sglang`` package; the import is deferred.
"""

from __future__ import annotations

from ..types import LayeredCache, LayerSlice, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter

__all__ = ["SGLangAdapter", "available"]


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("sglang") is not None


class SGLangAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for SGLang runtimes."""

    engine_name = "SGLang"
    required_extra = "sglang"
    DEGRADATION = (
        "no RadixAttention cache hooks in this build: prefill-only "
        "capture, fusion quality limited to the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'sglang' adapter needs the sglang package",
                hint="pip install 'c2c-cache[sglang]' or pip install sglang",
            )
        from importlib import import_module

        self._srt = import_module("sglang.srt")
        self._runtime = None
        self._hooks = self._discover_hooks()
        self._degraded = self._hooks is None
        self._pending: tuple = (None, None)  # (cache, prompt) or (None, None); never read unset

    def _discover_hooks(self):
        """Locate the radix-cache hooks of the installed runtime."""
        candidates = (
            ("sglang.srt.mem_cache", "radix_cache_hooks"),
            ("sglang.srt.managers", "cache_hooks"),
        )
        import importlib

        for module_name, attr in candidates:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            hooks = getattr(module, attr, None)
            if hooks is not None:
                return hooks
        return None

    def _ensure_runtime(self):
        if self._runtime is None:
            self._runtime = self._srt.Engine(
                model_path=self.model_id, **self.options.get("engine", {})
            )
        return self._runtime

    # -- CacheProvider ──────────────────────────────────────────────────────
    def _build_spec(self) -> ModelSpec:
        runtime = self._ensure_runtime()
        info = runtime.model_info()
        return ModelSpec(
            id=self.model_id,
            geometry=info.geometry,
            family=str(getattr(info, "family", "sglang")),
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def capture(self, prompt_tokens):
        if self._hooks is None:
            self._degraded = True
            return LayeredCache([])  # documented degradation
        rows = self._hooks.read_prefix(self._ensure_runtime(), list(prompt_tokens))
        return LayeredCache([LayerSlice(r.key, r.value) for r in rows])

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache, prompt_tokens=None):
        if isinstance(cache, LayeredCache) and len(cache):
            self._pending = (cache, list(prompt_tokens) if prompt_tokens is not None else None)
        else:
            self._pending = (None, None)

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        runtime = self._ensure_runtime()
        layers, for_prompt = self._pending
        if layers is not None and for_prompt is not None and for_prompt != list(prompt_tokens):
            layers = None  # the installation belongs to another prompt; ride fresh
        out = runtime.generate(
            prompt_token_ids=list(prompt_tokens),
            sampling_params={
                "temperature": temperature,
                "max_new_tokens": int(max_new_tokens),
                "stop": list(stop) if stop else None,
            },
            prefix_cache=layers,
        )
        return str(out)

    def encode(self, text: str):
        return self._ensure_runtime().tokenize(text)

    def decode_tokens(self, token_ids):
        return self._ensure_runtime().detokenize(list(token_ids))

    def available_capabilities(self):
        caps = super().available_capabilities()
        if self._degraded:
            caps.append(f"degraded:{self.DEGRADATION}")
        return caps
