"""vLLM adapter — prefix-cache / KV-connector plugin interface.

Mechanism (spec §4.1): vLLM's own connector API is the bridge. The engine
manages its paged KV blocks; the connector interface lets an external
party bind cache blocks to buffers it may access. Where the installed
vLLM version does not ship the connector, this adapter falls back to
prefill-only capture (generation still works; fusion degrades to the
receiver's own cache) and says so — documented in ``DEGRADATION_IF``.

Requires the ``vllm`` package. Import of the package is deferred to
construction, so ``import c2c`` never pulls vLLM in.
"""

from __future__ import annotations

from ..types import AttentionKind, LayeredCache, LayerGeometry, LayerSlice, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter, truncated

__all__ = ["vLLMAdapter", "available"]

#: module paths probed for the connector API, in order of preference
_CONNECTOR_PATHS = (
    "vllm.distributed.kv_transfer",
    "vllm.connectors",
    "vllm.kv_transfer",
)


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("vllm") is not None


class vLLMAdapter(EngineAdapter):  # noqa: N801 — the engines brand spelling
    """CacheProvider/CacheInjector for vLLM serving engines."""

    engine_name = "vLLM"
    required_extra = "vllm"
    TOOLS = "ignored"  # the generate accepts them, the engine ignores them
    DEGRADATION_IF = (
        "no connector API in this build: prefill-only capture, "
        "KV-block access unavailable; fusion quality limited to "
        "the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'vllm' adapter needs the vllm package",
                hint="pip install 'c2c-cache[vllm]' or pip install vllm",
            )
        from importlib import import_module

        self._vllm = import_module("vllm")
        self._connector = self._find_connector(import_module)
        self._engine = None
        self._degraded = self._connector is None
        self._pending: tuple = (None, None)  # (cache, prompt) or (None, None); never read unset

    @staticmethod
    def _find_connector(importer):
        for dotted in _CONNECTOR_PATHS:
            try:
                return importer(dotted)
            except ModuleNotFoundError:
                continue
        return None

    # -- engine bootstrap ───────────────────────────────────────────────────
    def _ensure_engine(self):
        if self._engine is None:
            kwargs = {
                "model": self.model_id,
                "trust_remote_code": bool(self.options.get("trust_remote_code", False)),
            }
            for key in ("tensor_parallel_size", "gpu_memory_utilization", "max_model_length"):
                if key in self.options:
                    kwargs[key] = self.options[key]
            self._engine = self._vllm.LLM(**kwargs)
        return self._engine

    # -- CacheProvider ──────────────────────────────────────────────────────
    def _build_spec(self) -> ModelSpec:
        engine = self._ensure_engine()
        cfg = (
            engine.model_config.hf_config
            if hasattr(engine, "model_config")
            else engine.llm_engine.model_config
        )
        heads = int(getattr(cfg, "num_attention_heads", 0) or 1)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or heads)
        layers = int(getattr(cfg, "num_hidden_layers", 0) or 1)
        kind = (
            AttentionKind.MHA
            if kv_heads == heads
            else AttentionKind.MQA
            if kv_heads == 1
            else AttentionKind.GQA
        )
        arch = str(getattr(cfg, "model_type", "") or getattr(cfg, "architectures", ["unknown"])[0])
        family = arch.split("For", 1)[0].lower() or "unknown"
        return ModelSpec(
            id=self.model_id,
            geometry=LayerGeometry(
                layers=layers,
                hidden_size=hidden,
                num_heads=heads,
                num_key_value_heads=kv_heads,
                attention=kind,
                name=self.model_id,
            ),
            family=family,
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def capture(self, prompt_tokens):
        """Capture the prefix cache through the connector API, if any."""
        if self._connector is None:
            self._degraded = True
            return LayeredCache([])  # documented degradation
        engine = self._ensure_engine()
        blocks = engine.collect_prefix_cache(list(prompt_tokens), connector=self._connector)
        slices = []
        for layer_rows in blocks:  # one list per layer
            k = layer_rows.get("key") if isinstance(layer_rows, dict) else layer_rows[0]
            v = layer_rows.get("value") if isinstance(layer_rows, dict) else layer_rows[1]
            slices.append(LayerSlice(k, v))
        return LayeredCache(slices)

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache, prompt_tokens=None):
        if self._connector is None or not isinstance(cache, LayeredCache) or not len(cache):
            self._pending = (None, None)
            return
        self._pending = (cache, list(prompt_tokens) if prompt_tokens is not None else None)

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        engine = self._ensure_engine()
        from vllm import SamplingParams  # engine-provided

        params = SamplingParams(
            temperature=temperature or 0.0,
            max_tokens=int(max_new_tokens),
            stop=list(stop) if stop else None,
        )
        prompt_token_ids = {"prompt_token_ids": list(prompt_tokens)}
        layers, for_prompt = self._pending
        if layers is not None and (for_prompt is None or for_prompt == list(prompt_tokens)):
            prompt_token_ids["cache"] = layers  # the fused rows, delivered for this prompt
        outs = engine.generate(prompts=[prompt_token_ids], sampling_params=params)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
        return truncated(text, stop)  # the engine cuts on its own; we cut as well

    def encode(self, text: str):
        engine = self._ensure_engine()
        return engine.tokenizer.encode(text)

    def decode_tokens(self, token_ids):
        engine = self._ensure_engine()
        return engine.tokenizer.decode(list(token_ids))
