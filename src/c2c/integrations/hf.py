"""HF Transformers adapter — ``past_key_values`` via forward hooks.

Mechanism (spec §4.1): the model is loaded once; the prefill pass returns
its ``past_key_values`` cache, converted layer by layer into a
:class:`c2c.types.LayeredCache`; generation runs with the (possibly fused)
cache installed through ``past_key_values=`` on ``generate``.

Requires the ``transformers`` package (extra: ``c2c-cache[hf]``). Where
the installed version predates the cache API, the adapter falls back to
prefill-only capture and *documents the degradation* (spec §4.1) via
the instance's capability report, surfaced by ``c2c doctor`` — never silent.

The cache layout follows the engine's own: each layer exposes ``.key``
and ``.value`` of shape ``[batch, heads, seq, head_dim]``; this adapter
drops the batch dimension (batch one, as in the paper's evaluation) and
keeps the heads view for the dynamic weighting module (FR-06).
"""

from __future__ import annotations

from ..types import AttentionKind, LayeredCache, LayerGeometry, LayerSlice, ModelSpec
from .registry import AdapterNotSupported, EngineAdapter, truncated

__all__ = ["HFAdapter", "available"]


def available() -> bool:
    """Is the engine importable right now? (cheap, no side effects)."""
    from importlib.util import find_spec

    return find_spec("transformers") is not None


class HFAdapter(EngineAdapter):
    """CacheProvider/CacheInjector for Hugging Face Transformers models."""

    engine_name = "HF Transformers"
    required_extra = "hf"
    TOOLS = "ignored"  # the generate accepts them, the engine ignores them
    DEGRADATION = None
    #: the build's condition: an old ``transformers`` without the cache API
    DEGRADATION_IF = (
        "this transformers version does not expose past_key_values: "
        "prefill-only capture, fusion reduced to the receiver's own cache"
    )

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        if not available():
            raise AdapterNotSupported(
                "the 'hf' adapter needs the transformers package",
                hint="pip install 'c2c-cache[hf]' (or pip install transformers>=4.40)",
            )
        from importlib import import_module

        self._tf = import_module("transformers")
        self._torch = import_module("torch")
        self._model = None
        self._tokenizer = None
        self._degraded = False
        self._pending: tuple | None = None  # the cache, until the next generation

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
                self.model_id, trust_remote_code=bool(self.options.get("trust_remote_code", False))
            )
        return self._tokenizer

    # -- CacheProvider ──────────────────────────────────────────────────────
    def _build_spec(self) -> ModelSpec:
        model = self._ensure_model()
        cfg = model.config
        pick = lambda *names, default=0: next(
            (int(getattr(cfg, n, 0) or 0) for n in names if int(getattr(cfg, n, 0) or 0)), default
        )
        hidden = pick("hidden_size", "hidden_dim", "d_model")
        heads = pick("num_attention_heads", "num_heads", default=1)
        kv_heads = pick("num_key_value_heads", "num_kv_heads") or heads
        head_dim = pick("head_dim") or (hidden // max(heads, 1))
        layers = pick("num_hidden_layers", default=1)
        ctx = pick("max_position_embeddings")
        if kv_heads == heads:
            kind = AttentionKind.MHA
        elif kv_heads == 1:
            kind = AttentionKind.MQA
        else:
            kind = AttentionKind.GQA
        geometry = LayerGeometry(
            layers=layers,
            hidden_size=hidden,
            num_heads=heads,
            head_size=head_dim,
            num_key_value_heads=kv_heads,
            attention=kind,
            name=self.model_id,
        )
        return ModelSpec(
            id=self.model_id,
            geometry=geometry,
            family=str(getattr(cfg, "model_type", "unknown") or "unknown"),
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
            vocab_size=pick("vocab_size"),
            context_length=ctx,
        )

    @classmethod
    def report_context(cls, model_id: str, **options) -> int | None:
        """``AutoConfig`` only: the card's ``max_position_embeddings``, rope-scaled."""
        if not available():
            return None
        from importlib import import_module

        tf = import_module("transformers")
        try:
            cfg = tf.AutoConfig.from_pretrained(
                model_id, trust_remote_code=bool(options.get("trust_remote_code", False))
            )
        except Exception:
            return None  # cannot say; never guess
        ctx = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        if not ctx:
            return None
        factor = getattr(getattr(cfg, "rope_scaling", None), "factor", None)
        try:
            factor = float(factor) if factor else 1.0
        except (TypeError, ValueError):
            factor = 1.0
        return int(ctx * factor) if factor >= 1 else ctx

    def capture(self, prompt_tokens):
        """Prefill and capture ``past_key_values`` as a LayeredCache."""
        model = self._ensure_model()
        ids = self._torch.as_tensor([list(prompt_tokens)], dtype=self._torch.long)
        if hasattr(model, "device"):
            ids = ids.to(model.device)
        with self._torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
        past = getattr(out, "past_key_values", None)
        if past is None:  # pre-cache API: degrade, document
            self._degraded = True
            return LayeredCache([])
        slices = []
        for layer in past:
            key = getattr(layer, "key", None)
            value = getattr(layer, "value", None)
            if key is None or value is None:  # legacy tuple-of-(k, v) layout
                key, value = layer[0], layer[1]
            slices.append(LayerSlice(key.transpose(1, 2)[0], value.transpose(1, 2)[0]))
        return LayeredCache(slices)

    # -- CacheInjector ──────────────────────────────────────────────────────
    def install(self, cache, prompt_tokens=None):
        """Remember the cache to be installed at the next generation."""
        self._pending = (cache, list(prompt_tokens) if prompt_tokens is not None else None)

    def _install_cache(self, layers):
        """Build the libraries DynamicCache from installed rows, batch/heads
        ordered the way the models own updates are — [1, kv, tokens, width]."""
        geo = self.spec().geometry
        kv = max(geo.num_key_value_heads, 1)

        def batched(rows):
            t = self._torch.as_tensor(rows)
            if t.ndim == 2:  # [tokens, kv_hidden]
                per = t.shape[1] // kv
                if per * kv != t.shape[1]:
                    msg = "flattened cache row width must split evenly across the kv heads"
                    raise ValueError(msg)
                view = t.reshape(t.shape[0], kv, per)  # [tokens, kv, width]
            elif t.ndim == 3:  # [tokens, heads, head]
                if t.shape[1] != kv:
                    msg = "cache rows carry %d heads; the model stores %d per layer"
                    raise ValueError(msg % (t.shape[1], kv))
                view = t
            else:
                msg = "cache rows must be [tokens, kv_hidden] or [tokens, heads, head]"
                raise ValueError(msg)
            view = view.transpose(0, 1)  # [kv, tokens, width]
            return view.unsqueeze(0).contiguous()  # [1, kv, tokens, width]

        cache = self._tf.DynamicCache()
        for idx, sl in enumerate(layers):
            cache.update(batched(sl.key), batched(sl.value), idx)
        return cache

    def _installed(self):
        pending = getattr(self, "_pending", None)
        if pending is not None and isinstance(pending[0], LayeredCache) and len(pending[0]):
            return pending[0]
        return None

    def _installed_for(self, prompt_tokens):
        """The installed cache, if it was installed for this very prompt.

        The gate mirrors the reference engines installed_prompt comparison:
        a cache delivered for one prompt must not condition the answer to
        another — the relay rides its own, fresh."""
        pending = getattr(self, "_pending", None)
        if pending is None or not isinstance(pending[0], LayeredCache) or not len(pending[0]):
            return None
        if pending[1] is not None and pending[1] != list(prompt_tokens):
            return None
        return pending[0]

    def score(self, token_ids):
        """Teacher-forced logits, rows aligned to predict ``token_ids[t+1]``.

        When a fused cache is installed the forward runs grad-enabled: the
        frozen weights demand nothing, but gradients must still reach the
        injected rows — that path is the entire training signal."""
        model = self._ensure_model()
        ids = self._torch.as_tensor([list(token_ids)], dtype=self._torch.long)
        if hasattr(model, "device"):
            ids = ids.to(model.device)
        layers = self._installed()
        kwargs = {"past_key_values": self._install_cache(layers)} if layers is not None else {}
        if layers is None:
            with self._torch.no_grad():
                out = model(input_ids=ids, **kwargs)
        else:
            out = model(input_ids=ids, **kwargs)
        return out.logits[0]

    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        tools=None,
        stop=None,
    ):
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
        layers = self._installed_for(prompt_tokens)
        if layers is not None:
            kwargs["past_key_values"] = self._install_cache(layers)
        with self._torch.no_grad():
            out = model.generate(input_ids=ids, **kwargs)
        new = out[0][len(list(prompt_tokens)) :]
        text = self._ensure_tokenizer().decode(new.tolist(), skip_special_tokens=True)
        return truncated(text, stop)  # the house cut, shared with the other adapters

    # -- tokenizer protocol bits, we provide ────────────────────────────────
    def encode(self, text: str):
        return self._ensure_tokenizer().encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids):
        return self._ensure_tokenizer().decode(list(token_ids), skip_special_tokens=True)
