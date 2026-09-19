"""TensorRT-LLM adapter — per-engine shim (degrades, documented).

The TensorRT-LLM runtime keeps its KV cache inside the C++ engine; its
Python bindings expose no cache hook. Per spec §4.1 this adapter falls
back to *prefill-only capture* and documents the degradation: generation
still works through the engine, but cross-model cache fusion is reduced
to the receiver's own cache unless a future engine release exposes the
blocks.
"""

from __future__ import annotations

from ..types import LayeredCache, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter

__all__ = ["TensorRTLLMAdapter", "available"]


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("tensorrt_llm") is not None


class TensorRTLLMAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for TensorRT-LLM runtimes (shimmed)."""

    engine_name = "TensorRT-LLM"
    required_extra = "trtllm"
    DEGRADATION = (
        "the runtime exposes no cache hook to Python: prefill-only "
        "capture; fusion reduced to the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'tensorrt-llm' adapter needs tensorrt_llm",
                hint="pip install 'c2c-cache[trtllm]' or follow the engine's container images",
            )
        from importlib import import_module

        self._trt = import_module("tensorrt_llm")
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = self._trt.Model.from_checkpoint(
                self.model_id, **self.options.get("model", {})
            )
        return self._model

    def _build_spec(self) -> ModelSpec:
        model = self._ensure()
        return ModelSpec(
            id=self.model_id,
            geometry=model.geometry,
            family=str(getattr(model, "model_family", "trtllm")),
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def capture(self, prompt_tokens):
        self._degraded = True
        return LayeredCache([])  # documented degradation

    def install(self, cache, prompt_tokens=None):
        self._pending = None  # no cache path into the engine

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        model = self._ensure()
        outputs = model.generate(
            [{"input_token_ids": list(prompt_tokens)}],
            {
                "end_id": model.eos_token_id,
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "stop": list(stop) if stop else None,
            },
        )
        return str(outputs[0])

    def encode(self, text: str):
        return self._ensure().tokenizer.encode(text)

    def decode_tokens(self, token_ids):
        return self._ensure().tokenizer.decode(list(token_ids))

    def available_capabilities(self):
        caps = super().available_capabilities()
        caps.append(f"degraded:{self.DEGRADATION}")
        return caps
