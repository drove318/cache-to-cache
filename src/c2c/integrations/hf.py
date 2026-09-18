"""HF Transformers adapter — ``past_key_values`` via forward hooks.

Mechanism (spec §4.1): the model is loaded once; the prefill pass returns
its ``past_key_values`` cache, converted layer by layer into a
:class:`c2c.types.LayeredCache`; generation runs with the (possibly fused)
cache installed through ``past_key_values=`` on ``generate``.

Requires the ``transformers`` package (extra: ``c2c-cache[hf]``). Where
the installed version predates the cache API, the adapter falls back to
prefill-only capture and *documents the degradation* (spec §4.1) via
``DEGRADED_REASON``, surfaced by ``c2c doctor`` — never silent.

The cache layout follows the engine's own: each layer exposes ``.key``
and ``.value`` of shape ``[batch, heads, seq, head_dim]``; this adapter
drops the batch dimension (batch one, as in the paper's evaluation) and
keeps the heads view for the dynamic weighting module (FR-06).
"""

from __future__ import annotations

from ..types import (AttentionKind, LayeredCache, LayerGeometry, LayerSlice,
                     ModelSpec)
from .registry import AdapterNotSupported, EngineAdapter

__all__ = ["HFAdapter", "available"]


def available() -> bool:
    """Is the engine importable right now? (cheap, no side effects)."""
    from importlib.util import find_spec
    return find_spec("transformers") is not None


class HFAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for Hugging Face Transformers models."""

    engine_name = "HF Transformers"
    required_extra = "hf"
    DEGRADATION = None

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'hf' adapter needs the transformers package",
                hint="pip install 'c2c-cache[hf]' (or pip install transformers>=4.40)")
        from importlib import import_module
        self._tf = import_module("transformers")
        self._torch = import_module("torch")
        self._model = None
        self._tokenizer = None
        self._degraded = False
        self.DEGRADED_REASON: str | None = None

    # -- lazy loads: models are heavy; do it on first use ─────────────────
    def _ensure_model(self):
        if self._model is None:
            kwargs = {"trust_remote_code": bool(self.options.get("trust_remote_code", False))}
            device = self.options.get("device")
            if device:
                kwargs["device_map"] = device
            dtype = self.options.get("dtype")
            if dtype:
                kwargs["torch_dtype"] = getattr(self._torch, str(dtype), None) or dtype
            self._model = self._tf.AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
        return self._model

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._tf.AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=bool(self.options.get("trust_remote_code", False)))
        return self._tokenizer

    # -- CacheProvider ──────────────────────────────────────────────────────
    def _build_spec(self) -> ModelSpec:
        model = self._ensure_model()
        cfg = model.config
        heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "num_heads", 0) or 1)
        hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "hidden_dim", 0) or 0)
        kv_heads = int(getattr(cfg, "num_key_value_heads", 0)
                     or getattr(cfg, "num_kv_heads", 0) or heads)
        head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // max(heads, 1)))
        layers = int(getattr(cfg, "num_hidden_layers", 0) or 1)
        if kv_heads == heads:
            kind = AttentionKind.MHA
        elif kv_heads == 1:
            kind = AttentionKind.MQA
        else:
            kind = AttentionKind.GQA
        geometry = LayerGeometry(layers=layers, hidden_size=hidden, num_heads=heads,
                               head_size=head_dim, num_key_value_heads=kv_heads,
                               attention=kind, name=self.model_id)
        return ModelSpec(id=self.model_id, geometry=geometry,
                        family=str(getattr(cfg, "model_type", "unknown") or "unknown"),
                        instruction_tuned=bool(self.options.get("instruction_tuned", True)),
                        vocab_size=int(getattr(cfg, "vocab_size", 0) or 0))

    def capture(self, prompt_tokens):
        """Prefill and capture ``past_key_values`` as a LayeredCache."""
        model = self._ensure_model()
        ids = self._torch.as_tensor([list(prompt_tokens)], dtype=self._torch.long)
        if hasattr(model, "device"):
            ids = ids.to(model.device)
        with self._torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
        past = getattr(out, "past_key_values", None)
        if past is None:                                  # pre-cache API: degrade, document
            self._degraded = True
            self.DEGRADED_REASON = ("this transformers version does not expose "
                                   "past_key_values: prefill-only capture, fusion reduced "
                                   "to the receiver's own cache")
            return LayeredCache([])
        slices = []
        for layer in past:
            key = getattr(layer, "key", None)
            value = getattr(layer, "value", None)
            if key is None or value is None:              # legacy tuple-of-(k, v) layout
                key, value = layer[0], layer[1]
            slices.append(LayerSlice(key[0], value[0]))   # batch size is one; drop it
        return LayeredCache(slices)

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache, prompt_tokens=None):
        """Remember the cache to be installed at the next generation."""
        self._pending = (cache, list(prompt_tokens) if prompt_tokens is not None else None)

    def score(self, token_ids):
        """Teacher-forced logits, rows aligned to predict ``token_ids[t+1]``."""
        model = self._ensure_model()
        ids = self._torch.as_tensor([list(token_ids)], dtype=self._torch.long)
        if hasattr(model, "device"):
            ids = ids.to(model.device)
        with self._torch.no_grad():
            out = model(input_ids=ids)
        return out.logits[0]

    def generate(self, prompt_tokens, *, max_new_tokens: int = 64, temperature: float = 0.0,
                 tools=None, stop=None):
        model = self._ensure_model()
        ids = self._torch.as_tensor([list(prompt_tokens)], dtype=self._torch.long)
        if hasattr(model, "device"):
            ids = ids.to(model.device)
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature and temperature > 0.0),
            "eos_token_id": getattr(self._ensure_tokenizer(), "eos_token_id", None),
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(temperature)
        pending = getattr(self, "_pending", None)
        if pending is not None and isinstance(pending[0], LayeredCache) and len(pending[0]):
            kwargs["past_key_values"] = tuple(
                (sl.key[None], sl.value[None]) for sl in pending[0])
        with self._torch.no_grad():
            out = model.generate(input_ids=ids, **kwargs)
        new = out[0][len(list(prompt_tokens)):]
        text = self._ensure_tokenizer().decode(new.tolist(), skip_special_tokens=True)
        if stop:
            for s in stop:
                if s and s in text:
                    text = text[:text.find(s)]
        return text

    # -- tokenizer protocol bits, we provide ────────────────────────────────
    def encode(self, text: str):
        return self._ensure_tokenizer().encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids):
        return self._ensure_tokenizer().decode(list(token_ids), skip_special_tokens=True)

    # -- capabilities, for c2c doctor ───────────────────────────────────────
    def available_capabilities(self):
        caps = super().available_capabilities()
        if self._degraded and self.DEGRADED_REASON:
            caps.append(f"degraded:{self.DEGRADED_REASON}")
        return caps
