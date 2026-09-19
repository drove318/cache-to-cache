"""MLX adapter — per-engine shim for Apple Silicon (degrades, documented).

MLX keeps its KV cache inside the Swift/Objective-C++ runtime objects of
the ``mlx.core`` array framework; the Python bindings of MLX LM expose no
cache hook. This adapter therefore captures nothing where the cache is
concerned — it generates through the native ``mlx_lm.generate`` entry
points and falls back to prefill-only mode, documenting the degradation.

Requires the optional peers ``mlx`` and ``mlx-lm``; imports are deferred
so ``import c2c`` never touches the framework.
"""

from __future__ import annotations

from typing import Any

from ..types import LayeredCache, LayerGeometry, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter, truncated

__all__ = ["MLXAdapter", "available"]


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("mlx_lm") is not None and find_spec("mlx.core") is not None


class MLXAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for MLX LM on Apple Silicon."""

    engine_name = "MLX"
    required_extra = "mlx"
    TOOLS = "ignored"  # the generate accepts them, the engine ignores them
    DEGRADATION = (
        "the bindings expose no cache hook: prefill-only capture; "
        "fusion reduced to the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'mlx' adapter needs mlx and mlx-lm",
                hint="pip install 'c2c-cache[mlx]' on an Apple Silicon mac",
            )
        from importlib import import_module

        self._mlx = import_module("mlx.core")
        self._mlx_lm = import_module("mlx_lm")
        self._model: Any = None
        self._tokenizer: Any = None
        self._degraded = False

    def _ensure(self):
        if self._model is None:
            self._model = self._mlx_lm.utils.load(self.model_id)
            self._tokenizer = self._mlx_lm.utils.load(self.model_id, ["tokenizer"])
        return self._model

    def _build_spec(self) -> ModelSpec:
        model = self._ensure()
        cfg = getattr(model, "model_config", None) or getattr(model, "config", None)
        layers = int(getattr(cfg, "num_hidden_layers", 1) or 1) if cfg else 1
        hidden = int(getattr(cfg, "hidden_size", 1) or 1) if cfg else 1
        heads = int(getattr(cfg, "num_attention_heads", 1) or 1) if cfg else 1
        return ModelSpec(
            id=self.model_id,
            geometry=LayerGeometry(
                layers=layers,
                hidden_size=hidden,
                num_heads=heads,
                name=(
                    self.model_id
                    if cfg is not None
                    else f"{self.model_id} (geometry unknown — the model carries no config)"
                ),
            ),
            family=str(getattr(cfg, "model_type", "mlx") if cfg else "mlx"),
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def capture(self, prompt_tokens):
        self._degraded = True
        return LayeredCache([])  # documented degradation

    def install(self, cache, prompt_tokens=None):
        self._pending = None

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
        self._ensure()
        from importlib import import_module

        driver = import_module("mlx_lm.generate.driver")
        stream = driver.generate(
            prompt=self.decode_tokens(prompt_tokens),
            model=self._model,
            tokenizer=self._tokenizer,
            max_tokens=int(max_new_tokens),
            temp=float(temperature or 0.0),
        )
        return truncated("".join(stream), stop)  # the driver knows no stops; we do

    def _ensure_tokenizer(self) -> Any:
        self._ensure()  # the model carries the tokenizer
        return self._tokenizer

    def encode(self, text: str):
        return self._ensure_tokenizer().encode(text)

    def decode_tokens(self, token_ids):
        return self._ensure_tokenizer().decode(list(token_ids))
