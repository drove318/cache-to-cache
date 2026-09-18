"""In-process engine adapters: one ABI, many adapters (spec §4.1).

The registry (``c2c.integrations.engines``) maps names to adapter
classes; the reference adapter implements the full cache path without
any engine installed (see :mod:`c2c.integrations.reference`), while every
other adapter delegates to the native API of its engine and *documents
its degradation* where the engine exposes no cache hook.

Availability policy: importing this package never imports any engine.
Adapters import their engine lazily, on construction, and raise
:class:`~c2c.integrations.registry.AdapterNotSupported` with the exact
pip command to fix it when the engine is missing — never a bare
ImportError, never a silent fallback.
"""

from __future__ import annotations

from .registry import AdapterNotSupported, EngineAdapter, engines

__all__ = ["engines", "EngineAdapter", "AdapterNotSupported"]
