"""Engine adapters: in-process peers behind the CacheProvider/CacheInjector ABI.

Paper map (normative, spec §1 & §4.1): the adapter registry plus one
adapter per inference engine —

======================  ===========================================  ===========
Engine                  Mechanism (per spec §4.1)                    Cache path
======================  ===========================================  ===========
HF Transformers          ``past_key_values`` hooks, GenerationMixin    full
vLLM                   KV-connector plugin interface                   full
SGLang                 RadixAttention cache hooks                     full
TensorRT-LLM           per-engine shim                                degraded
TGI                    HTTP control, cache read                       degraded
Ollama                 HTTP control, cache read                       degraded
MLX                    per-engine shim                                 degraded
llama.cpp              per-engine shim                                degraded
======================  ===========================================  ===========

*full* adapters capture and re-install caches without loss. *degraded*
adapters fall back to prefill-only capture where the engine exposes no
cache hook, and the degradation is documented on the adapter class itself
(``DEGRADATION`` attribute), printed by ``c2c doctor`` — never silent.

Third-party adapters: ship a package exposing the ``c2c.engines`` entry-
point group; discovery is automatic (importlib.metadata), user entries
take precedence over built-ins of the same name (local overrides), with a
visible note in ``c2c doctor``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Sequence

from ..types import ModelSpec

__all__ = ["engines", "EngineAdapter", "AdapterNotSupported", "truncated"]


def truncated(text: str, stop: Sequence[str] | None) -> str:
    """Cut *text* at the earliest of the *stop* sequences.

    Adapters whose engines honour stop lists on their own endpoint pass
    them through; every adapter cuts post-generation as well, belt and
    braces, so that a harness which passes stop always stops.
    """
    if not stop:
        return text
    cut = len(text)
    for seq in stop:
        if seq:
            at = text.find(seq)
            if 0 <= at < cut:
                cut = at
    return text[:cut]


ENTRY_POINT_GROUP = "c2c.engines"

#: the built-in distribution: one ABI, nine adapters, all listed, all
#: registered, on the bus of the engines. Adapters import their engines
#: lazily, on construction — importing this module never imports an engine,
#: and registering an adapter never loads one (availability policy, §4.1).
_BUILTIN_TARGETS: dict[str, str] = {
    "reference": "c2c.integrations.reference:ReferenceAdapter",
    "hf": "c2c.integrations.hf:HFAdapter",
    "vllm": "c2c.integrations.vllm:vLLMAdapter",
    "sglang": "c2c.integrations.sglang:SGLangAdapter",
    "trtllm": "c2c.integrations.trtllm:TensorRTLLMAdapter",
    "llamacpp": "c2c.integrations.llamacpp:LlamaCppAdapter",
    "tgi": "c2c.integrations.tgi:TGIAdapter",
    "ollama": "c2c.integrations.ollama:OllamaAdapter",
    "mlx": "c2c.integrations.mlx:MLXAdapter",
}


class AdapterNotSupported(RuntimeError):  # noqa: N818 — public name, shipped and documented
    """Raised when an adapter's engine (or a required capability) is absent.

    The message always carries the fix: which extra to install, or which
    fallback (``reference`` adapter) is available.
    """

    def __init__(self, message: str, *, hint: str | None = None):
        self.hint = hint
        full = message if not hint else f"{message} — {hint}"
        super().__init__(full)


class EngineAdapter:
    """Base class of every adapter; the contract of the engine ABI.

    Subclasses implement the two interfaces of :mod:`c2c.types`
    (``CacheProvider`` and ``CacheInjector``) against their engine's
    native API, and declare a ``DEGRADATION`` string when the engine
    cannot offer lossless capture.
    """

    #: human-readable name of the underlying engine
    engine_name: str = "unknown"
    #: the extra that provides the engine, if the capability is missing
    required_extra: str | None = None
    #: the permanent truth of the engine: what it cannot do, whatever the
    #: build; printed by c2c doctor; never empty in a shipping adapter
    DEGRADATION: str | None = None
    #: the build's condition: what an instance cannot do because the installed
    #: engine lacks a hook the adapter could use (adapters set ``_degraded``)
    DEGRADATION_IF: str | None = None

    #: what the engine's generate does with the ABI's tools argument:
    #: 'native' when the engine acts on it, 'ignored' when the parameter is
    #: accepted and dropped; None when the adapter does not say, not reported
    TOOLS: str | None = None
    #: configuration — defaults for the adapter's own options
    default_options: dict = {}

    def __init__(self, model_id: str, **options):
        self.model_id = model_id
        self.options = {**self.default_options, **options}
        self._spec: ModelSpec | None = None
        self._degraded = False  # the build's condition, set on discovering a missing hook

    # -- protocol: CacheProvider/CacheInjector delegation, all optional ────
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = self._build_spec()
        return self._spec

    def _build_spec(self) -> ModelSpec:  # override in adapters
        msg = f"{type(self).__name__} does not implement _build_spec()"
        raise NotImplementedError(msg)

    @classmethod
    def report_context(cls, model_id: str, **options) -> int | None:
        """Tokens the model's own card claims, read without loading weights.

        ``/v1/models`` uses this so a harness sizes its context from the
        served model's truth, not from a number baked into documentation.
        Adapters override where the engine exposes a cheap config read;
        ``None`` means "this engine cannot say", and the listing omits it.
        """
        return None

    def available_capabilities(self) -> list[str]:
        """What this adapter can do on this machine (for doctor, reports)."""
        caps = [
            "capture" if hasattr(self, "capture") else "capture:missing",
            "install" if hasattr(self, "install") else "install:missing",
            "generate" if hasattr(self, "generate") else "generate:missing",
            "score" if hasattr(self, "score") else "score:missing",
        ]
        why = self.DEGRADATION
        if why is None and getattr(self, "_degraded", False):
            why = self.DEGRADATION_IF
        if self.TOOLS:
            caps.append(f"tools:{self.TOOLS}")
        if why:
            caps.append(f"degraded:{why}")
        return caps

    def is_degraded(self) -> bool:
        """True when the installed build lacks a hook the adapter could use.

        A permanent ``DEGRADATION`` is the nature of the engine, no fault of
        this machine; ``DEGRADATION_IF`` under this predicate is the build's
        condition, and the actionable kind — the doctor warns of it.
        """
        return bool(self.DEGRADATION_IF) and getattr(self, "_degraded", False)

    # subclasses must override the two hooks of the capture path, if present
    def close(self) -> None:
        """Release engine resources; idempotent by design."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _EngineRegistry:
    """The registry of the adapters: register, unregister, lookup, load, and
    the reporting of ``registered_names`` and ``describe``.

    Discovery of the ``c2c.engines`` entry points is cached for the life of
    the process; :meth:`refresh` forgets the cache. Precedence: the explicit
    registrations (the built-ins, and any ``register`` — a forced one casts
    aside a discovered entry point of the same name) win over the discovered.
    """

    group = ENTRY_POINT_GROUP

    def __init__(self):
        self._registry: dict[str, str] = dict(_BUILTIN_TARGETS)
        self._discovered: dict[str, str] | None = None

    # -- registration (see the registration section below) ─────────────────
    def register(self, name: str, target: str, *, force: bool = False) -> None:
        """Register an adapter under ``name``: a ``module:attribute`` string.

        Raises ValueError if the name is already registered and *force*
        is not true (so accidental registrations do not shadow the builtins).
        """
        module_part, _, attribute_part = target.partition(":")
        if not attribute_part:
            raise ValueError(f"invalid adapter target {target!r}; expected 'module:attribute'")
        try:
            importlib.util.resolve_name(f"{module_part}.{attribute_part}", None)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError(f"invalid adapter target {target!r}: {exc}") from exc
        if not force and (name in self._registry or name in self._entry_points()):
            msg = f"engine {name!r} already registered; pass force=True"
            raise ValueError(msg)
        self._registry[name] = target
        if force and self._discovered is not None:
            self._discovered.pop(name, None)  # the deliberate shadow wins

    def unregister(self, name: str) -> None:
        """Remove an adapter from the bus: an unknown name, a key error."""
        if self._registry.pop(name, None) is None and name not in self._entry_points():
            raise KeyError(f"engine {name!r} is not registered")

    def refresh(self) -> None:
        """Forget the discovered entry points; the next lookup sees the world anew.

        Installing or removing a distribution while the process runs is
        invisible to the cache until this is called.
        """
        self._discovered = None

    # -- discovery (entry points; see the packaging section) ───────────────
    def _entry_points(self) -> dict[str, str]:
        if self._discovered is None:
            found: dict[str, str] = {}
            try:
                for ep in importlib.metadata.entry_points(group=self.group):
                    found[ep.name] = f"{ep.value}"
            except TypeError:  # older importlib needs the kw split
                eps = importlib.metadata.entry_points()
                for ep in eps.select(group=self.group):
                    found[ep.name] = str(ep.value)
            self._discovered = found
        return self._discovered

    def _builtin_names(self) -> tuple[str, ...]:
        return tuple(_BUILTIN_TARGETS)

    # -- lookup and access ─────────────────────────────────────────────────
    def registered_names(self) -> list[str]:
        names = set(self._registry) | set(self._entry_points())
        return sorted(names)

    def lookup(self, name: str) -> str:
        """Return the ``module:attribute`` string registered for *name*."""
        eps = self._entry_points()
        if name in eps:  # local (user) overrides
            return eps[name]
        try:
            return self._registry[name]
        except KeyError:
            known = ", ".join(self.registered_names()) or "none"
            msg = f"unknown engine {name!r}; registered engines: {known}"
            raise KeyError(msg) from None

    def load(self, name: str, **options):
        """Instantiate the adapter registered under *name* for *model_id*.

        ``options`` are passed through to the adapter constructor
        (configuration, see the adapter's own documentation).
        """
        target = self.lookup(name)
        module_name, _, attribute = target.partition(":")
        try:
            module = _import_module(module_name)
        except ModuleNotFoundError as exc:
            if name != "reference":
                hint = (
                    f"the '{name}' adapter needs its engine installed "
                    f"(pip install the matching extra) — or use --engine reference"
                )
            else:
                hint = (
                    "the reference adapter is the house nets; it needs the "
                    "train extra — pip install 'c2c-cache[train]'"
                )
            raise AdapterNotSupported(f"engine {name!r} unavailable: {exc}", hint=hint) from exc
        try:
            cls = getattr(module, attribute)
        except AttributeError as exc:
            raise AdapterNotSupported(
                f"engine {name!r}: {target} does not export {attribute!r}"
            ) from exc
        return cls(**options)

    # -- reporting ──────────────────────────────────────────────────────────
    def describe(self) -> list[str]:
        """One line per registered adapter, sorted by name (for c2c doctor)."""
        lines: list[str] = []
        for name in self.registered_names():
            target = self.lookup(name)
            try:
                module = _import_module(target.partition(":")[0])
                cls = getattr(module, target.partition(":")[2])
                why = getattr(cls, "DEGRADATION", None)
                mark = (
                    "degraded"
                    if why
                    else "conditional"
                    if getattr(cls, "DEGRADATION_IF", None)
                    else "full"
                )
                lines.append(f"  {name:<12} {target:<52} [{mark}]")
            except Exception as exc:  # reports must not raise
                lines.append(f"  {name:<12} {target:<52} [unloaded: {exc.__class__.__name__}]")
        return lines


def _import_module(dotted: str):
    """Import a module by its dotted name — the standard import machinery."""
    return importlib.import_module(dotted)


engines = _EngineRegistry()
