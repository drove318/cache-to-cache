"""The serving side: the model hub behind every transport.

One registry of *things that speak*: single models (``register_model``)
and pairs (``register_pair``). The proxy (``c2c-serve``), the MCP server
(``c2c-mcp``) and the CLI all resolve model ids through this hub, so a
virtual id such as ``c2c/qwen3-0.6b←qwen2.5-0.5b`` means the same thing
on every transport, on every machine, in every harness.

Resolution rules (published, see the man page ``c2c.config(5)``,
section MODEL IDS):

* the prefix ``c2c/`` selects the C2C namespace;
* the pair separator is the leftwards arrow ``←`` (U+2190); the ASCII
  alternatives ``->``, ``-->``, ``→``, ``--`` and ``:`` are accepted so
  every harness can type it;
* the first component names the **Receiver**, the second the **Sharer**
  — reading direction respected, conventions of the community preserved;
* an id naming a single registered model (no separator) answers as a
  plain relay: no fusion, just the model's own cache.

Engines are built lazily on first request through the adapter registry
(``c2c.integrations.engines``); with no engine installed the reference
engine answers, so ``c2c-serve`` is usable out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from ..config import ServeConfig
from ..types import CacheInjector, CacheProvider, ModelSpec

__all__ = ["Pair", "ResolvedTarget", "ModelHub", "default_hub"]


@dataclass
class Pair:
    """A registered collaboration between a Receiver and a Sharer."""

    receiver: str
    sharer: str
    fuser: Any | None = None                 # a c2c.fuser.Fuser, or None for relay
    aligner: Any | None = None               # a c2c.align.TokenAligner factory
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.receiver}←{self.sharer}"

    @property
    def fused(self) -> bool:
        return self.fuser is not None


@dataclass
class ResolvedTarget:
    """What a model id resolves to: the parts of the working pipeline."""

    model_id: str
    receiver: Any                     # object implementing CacheProvider+CacheInjector
    sharer: Any | None
    fuser: Any | None
    pair: Pair | None
    relay: bool                        # True ⇒ no fusion; single-model pass-through
    spec: ModelSpec | None = None


class ModelHub:
    """The registry the transports consult; the models speak to it.

    Thread-safe: every mutation of the registry takes the lock; the read
    path (resolution) reads snapshots only.
    """

    def __init__(self, config: ServeConfig | None = None):
        self.config = config or ServeConfig()
        self._lock = RLock()
        self._models: dict[str, dict] = {}           # id → entry dict
        self._pairs: dict[str, Pair] = {}            # canonical pair id → Pair
        self._factories: dict[str, Callable[[str, dict], Any]] = {}
        self.engine_name = "reference"               # built-in fallback engine
        self.curated = False                   # nothing auto, until you register

    # -- registering, the interface ──────────────────────────────────────────
    def register_model(self, model_id: str, *, provider: CacheProvider | None = None,
                       injector: CacheInjector | None = None, options: dict | None = None,
                       note: str = "") -> None:
        """Register a single model, optionally pre-configured with its parts.

        ``provider`` and ``injector`` may name one and the same object
        (most adapters are both). When both are omitted, the hub builds
        one lazily from the engine adapter named by :attr:`engine_name`.
        """
        key = self._canonical(model_id)
        self.curated = True                  # the operator has spoken: nothing auto
        with self._lock:
            self._models[key] = {
                "id": key,
                "provider": provider, "injector": injector,
                "options": dict(options or {}), "note": note,
            }

    def register_pair(self, *, receiver: str, sharer: str, fuser=None,
                     aligner=None, note: str = "") -> Pair:
        """Register a (Receiver, Sharer) collaboration under its canonical id."""
        self.curated = True                  # the operator has spoken: nothing auto
        pair = Pair(receiver=self._canonical(receiver), sharer=self._canonical(sharer),
                   fuser=fuser, aligner=aligner, note=note)
        with self._lock:
            self._pairs[pair.id] = pair
            for side in (pair.receiver, pair.sharer):     # a pair implies its two sides
                if side not in self._models:              # register both, read: one
                    self._models[side] = {"id": side, "provider": None, "injector": None,
                                        "options": {}, "note": "side of a pair"}
        return pair

    def unregister(self, model_id: str) -> bool:
        """Remove a model or a pair; True if something was removed."""
        key = self._canonical(model_id).lower()
        with self._lock:
            if self._pairs.pop(key, None) is not None:
                return True
            if self._models.pop(key, None) is not None:
                return True
        return False

    def set_engine(self, name: str) -> None:
        """Choose which adapter the hub consults when building models lazily."""
        self.engine_name = name

    def register_factory(self, name: str, factory: Callable[[str, dict], Any]) -> None:
        """Install a custom builder: ``factory(model_id, options) -> adapter``."""
        self._factories[name] = factory

    # -- resolving, the read path ────────────────────────────────────────────
    def _canonical(self, model_id: str) -> str:
        s = (model_id or "").strip()
        if s.startswith(self.config.model_prefix):
            s = s[len(self.config.model_prefix):]
        return s.lower()

    def parse_pair_id(self, model_id: str) -> tuple[str, str] | None:
        """Split a virtual id into (receiver, sharer); None for single models."""
        raw = self._canonical(model_id)
        if not raw:
            return None
        for sep in (self.config.pair_separator, "-->", "->", "→", "--", ":"):
            if sep in raw:
                left, _, right = raw.partition(sep)
                left, right = left.strip(), right.strip()
                if left and right:
                    return (left, right)
        return None

    def models(self) -> list[str]:
        """Every id, sorted, of all registered models and pairs."""
        with self._lock:
            singles = sorted(self._models)
            pairs = sorted(self._pairs)
        return [f"{self.config.model_prefix}{p}" for p in (*singles, *pairs)]

    def describe_models(self) -> list[tuple[str, str]]:
        """(id, description) pairs for the /v1/models listing."""
        out: list[tuple[str, str]] = []
        with self._lock:
            for mid in sorted(self._models):
                note = self._models[mid].get("note") or "single model (relay)"
                out.append((f"{self.config.model_prefix}{mid}", str(note)))
            for pid in sorted(self._pairs):
                pair = self._pairs[pid]
                mode = "cache-to-cache" if pair.fused else "relay"
                note = f"{mode}: {pair.receiver} receives, {pair.sharer} shares"
                if pair.note:
                    note += f" ({pair.note})"
                out.append((f"{self.config.model_prefix}{pid}", note))
        return out

    def resolve(self, model_id: str) -> ResolvedTarget | None:
        """Resolve a model id through the hub, building parts as needed.

        Returns ``None`` when the id names nothing registered and cannot
        be built — the caller (the proxy) then renders the OpenAI error
        object with type ``invalid_request_error``, code ``model_not_found``.
        """
        parsed = self.parse_pair_id(model_id)
        if parsed is None:
            single = self._canonical(model_id)
            with self._lock:
                entry = self._models.get(single)
            if entry is None:
                entry = self._make_entry(single)
            if entry is None:
                return None
            provider, injector = self._parts(single, entry)
            actor = injector if injector is not None else provider
            if actor is None:
                return None
            return ResolvedTarget(model_id=self.config.model_prefix + single,
                                receiver=actor, sharer=None, fuser=None,
                                pair=None, relay=True, spec=_spec_of(actor))
        receiver_id, sharer_id = parsed
        pair_key = f"{receiver_id}←{sharer_id}"
        with self._lock:
            pair = self._pairs.get(pair_key)
        recv_actor = self._resolve_side(receiver_id)
        shar_actor = self._resolve_side(sharer_id)
        if recv_actor is None or shar_actor is None:
            return None
        if pair is None:
            return ResolvedTarget(model_id=self.config.model_prefix + pair_key,
                                receiver=recv_actor, sharer=shar_actor, fuser=None,
                                pair=None, relay=True, spec=_spec_of(recv_actor))
        return ResolvedTarget(model_id=self.config.model_prefix + pair_key,
                            receiver=recv_actor, sharer=shar_actor, fuser=pair.fuser,
                            pair=pair, relay=pair.fuser is None, spec=_spec_of(recv_actor))

    # -- building blocks ─────────────────────────────────────────────────────
    def _resolve_side(self, model_id: str):
        with self._lock:
            entry = self._models.get(model_id)
        if entry is None:
            entry = self._make_entry(model_id)
            if entry is None:
                return None
        provider, injector = self._parts(model_id, entry)
        return injector if injector is not None else provider

    def _parts(self, model_id: str, entry: dict) -> tuple[Any, Any]:
        provider = entry.get("provider")
        injector = entry.get("injector")
        if provider is None and injector is None:
            built = self._build(model_id, entry.get("options") or {})
            if built is not None:
                provider = injector = built
                entry["provider"], entry["injector"] = provider, injector
        return provider, injector

    def _make_entry(self, model_id: str) -> dict | None:
        if self.curated:
            return None                        # curated: no auto-build, no entry
        obj = self._build(model_id, {})
        if obj is None:
            return None
        entry = {"id": model_id, "provider": obj, "injector": obj,
               "options": {}, "note": f"auto-built via {self.engine_name}"}
        with self._lock:
            self._models.setdefault(model_id, entry)
        return entry

    def _build(self, model_id: str, options: dict) -> Any | None:
        """Instantiate an adapter for *model_id* through the engine registry."""
        if not model_id:
            return None
        factory = self._factories.get(self.engine_name)
        try:
            if factory is not None:
                return factory(model_id, dict(options or {}))
            from ..integrations import engines
            return engines.load(self.engine_name, model_id=model_id, **(options or {}))
        except Exception:
            return None


def _spec_of(obj) -> ModelSpec | None:
    fn = getattr(obj, "spec", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


#: The hub every transport consults. Tests substitute fresh ones.
default_hub = ModelHub()
