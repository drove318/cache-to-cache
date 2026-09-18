"""The out-of-process faces of C2C (spec §4.2): the any-harness guarantee.

Every transport here speaks the same core semantics through the model hub
(:mod:`c2c.serve.registry`):

``openai_proxy``  ``c2c-serve``   OpenAI-compatible HTTPS front (HL-1)
``mcp``           ``c2c-mcp``     Model Context Protocol server (HL-2)
``a2a``                         Agent-card advertisement (HL-3)
``privacy``                       EX-5 privacy mode: the wire, sealed

The CLI is control-plane only (HL-4); there is deliberately no
``c2c run-agent``: the harness stays the master, C2C is the wire.
"""

from __future__ import annotations

from .registry import ModelHub, Pair, ResolvedTarget, default_hub

__all__ = ["ModelHub", "Pair", "ResolvedTarget", "default_hub"]
