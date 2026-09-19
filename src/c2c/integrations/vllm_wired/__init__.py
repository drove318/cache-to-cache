"""The wired vLLM family (spec §4.1, M3): C2C inside the serving engine.

Why in-engine: the KV blocks are the worker's property. The wire's
companion :class:`c2c.integrations.vllm_wired.connector.C2CWiredConnector`
mounts into the engine's own ``KVConnectorBase_V1`` surface
(``register_kv_caches`` / ``wait_for_layer_load`` /
``get_num_new_matched_tokens``), so a fusion never copies a cache out
of the engine and back — one weight copy, one KV pool, no second
container. That is the only configuration that fits the box: the live
server already claims 101 of 121 GiB of unified memory.

Modules:
    resident   the resident wire: the trainable fuser, held on the
               engine's device, speaking the engine's storage dtype
    connector  the KVConnectorBase_V1 subclass (importable only inside
               a vLLM worker; host code must not import it)
    wire       wire-file loading: the safetensors the menagerie keeps,
               validated against the engine's live layer names
    sidecar    the loopback JSON protocol the front speaks to a wired
               server (tokenize, role pairs, the training channel)
    adapter    the host-side EngineAdapter: ``--engine vllm-wired``

The first landing is identity, not intelligence: a wired server with
the gate pinned closed must answer exactly as the unwired one does,
byte for byte — :class:`resident.ResidentWire` makes that the contract,
not the hope.
"""

from __future__ import annotations

__all__ = ["C2C_ENGINE_NAME"]

#: the engine name the registry and the CLI use for this family
C2C_ENGINE_NAME = "vllm-wired"
