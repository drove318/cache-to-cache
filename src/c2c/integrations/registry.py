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

from ..types import CacheInjector, CacheProvider, ModelSpec

__all__ = ["engines", "EngineAdapter", "AdapterNotSupported"]

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


class AdapterNotSupported(RuntimeError):
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
    #: set when capture quality is reduced below the ABI's expectation;
    #: printed by c2c doctor; never empty in a shipping adapter
    DEGRADATION: str | None = None

    #: configuration — defaults for the adapter's own options
    default_options: dict = {}

    def __init__(self, model_id: str, **options):
        self.model_id = model_id
        self.options = {**self.default_options, **options}
        self._spec: ModelSpec | None = None

    # -- protocol: CacheProvider/CacheInjector delegation, all optional ────
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = self._build_spec()
        return self._spec

    def _build_spec(self) -> ModelSpec:          # override in adapters
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
        caps = ["capture" if hasattr(self, "capture") else "capture:missing",
               "install" if hasattr(self, "install") else "install:missing",
               "generate" if hasattr(self, "generate") else "generate:missing",
               "score" if hasattr(self, "score") else "score:missing"]
        if self.DEGRADATION:
            caps.append(f"degraded:{self.DEGRADATION}")
        return caps

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
    """The registry, a small selection of the available adapters.

    Membership tests, dictionary access (both by name and by module),
    all the usual methods of dict (see the entry-points section of the
    language documentation for details on the group name).
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

    def unregister(self, name: str) -> None:
        """Remove an adapter from the bus: an unknown name, a key error."""
        if self._registry.pop(name, None) is None and name not in self._entry_points():
            raise KeyError(f"engine {name!r} is not registered")

    # -- discovery (entry points; see the packaging section) ───────────────
    def _entry_points(self) -> dict[str, str]:
        if self._discovered is None:
            found: dict[str, str] = {}
            try:
                for ep in importlib.metadata.entry_points(group=self.group):
                    found[ep.name] = f"{ep.value}"
            except TypeError:                    # older importlib needs the kw split
                eps = importlib.metadata.entry_points()
                for ep in eps.get(self.group, []):
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
        if name in eps:                          # local (user) overrides
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
            hint = None
            if name != "reference":
                hint = (f"the '{name}' adapter needs its engine installed "
                      f"(pip install the matching extra) — or use --engine reference")
            raise AdapterNotSupported(f"engine {name!r} unavailable: {exc}", hint=hint) from exc
        try:
            cls = getattr(module, attribute)
        except AttributeError as exc:
            raise AdapterNotSupported(
                f"engine {name!r}: {target} does not export {attribute!r}") from exc
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
                mark = "degraded" if getattr(cls, "DEGRADATION", None) else "full"
                lines.append(f"  {name:<12} {target:<52} [{mark}]")
            except Exception as exc:                       # noqa: reports must not raise
                lines.append(f"  {name:<12} {target:<52} [unloaded: {exc.__class__.__name__}]")
        return lines


def _import_module(dotted: str):
    """Import a module by its dotted name — the standard import machinery."""
    return importlib.import_module(dotted)


engines = _EngineRegistry()
