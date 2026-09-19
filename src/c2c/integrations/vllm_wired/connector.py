"""The wired connector: C2C's hands inside the vLLM worker.

Mounted the way the engine mounts its own offload connectors —
``--kv-transfer-config '{"kv_connector_module":
"c2c.integrations.vllm_wired.connector",
"kv_connector_class": "C2CWiredConnector"}'`` — this class speaks the
engine's ``KVConnectorBase_V1`` contract and moves rows the way the
engine's own example connector does: ``slot_mapping`` from the
per-request connector metadata, gather out of the paged buffer
``(num_pages, 2, page_size, …)``, and scatter back into the same
slots inside the load window. The wire never copies a cache out of
the engine: the fusion is three moves on the engine's device.

    save_kv_layer           the sharer's prefill hands us its rows
                            (the receiver's are kept too — the residual
                            reads its own)
    wait_for_layer_load     the wire fuses that layer and scatters the
                            fused rows into the engine's own buffer
    everything else         untouched: linear-attention state, the
                            other thirty-six layers, every other request

**Fail closed.** The wire is a guest in the engine's memory: any
fusion move that raises is logged with its class and message, the
layer keeps the receiver's own rows, and the answer goes out. A wired
server that cannot fuse is a wired server that still serves.

Gate, from ``kv_connector_extra_config``: ``"c2c_gate": "closed"``
pins every request to the identity — the B0 landing, where replies
must come back byte-identical to the unwired server; ``"open"`` forces
every gate open; absent, the resident wire's learned gates decide.

The engine's symbols are imported guarded: outside a vLLM worker the
module still imports (the host's gates and tests demand that much),
and anything that tries to build a connector says so plainly. Host
tests install a stub ``vllm`` namespace first (see
tests/test_wired_connector.py) and receive the real contract.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .resident import WIRE_ENV, ResidentWire

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        SupportsHMA,
    )
except ModuleNotFoundError:
    # the engine is where this connector runs; the module still imports,
    # for the host's gates and tests — building one elsewhere says so

    def _no_engine_bases():
        """The stand-ins: presentable anywhere, answerable only in the worker."""

        class _NoEngine:
            """The stand-in base: presentable anywhere, buildable in the worker."""

            def __init__(self, *args, **kwargs):
                msg = (
                    "the wired connector runs inside a vLLM worker: "
                    "pip install 'c2c-cache[vllm]' first"
                )
                raise ModuleNotFoundError(msg)

        class _NoMarker:
            """The stand-in marker: the host has no HMA to answer for."""

        return _NoEngine, _NoMarker

    KVConnectorBase_V1, SupportsHMA = _no_engine_bases()

__all__ = ["C2CWiredConnector", "GATE_CLOSED", "GATE_OPEN", "GATE_WIRE"]

GATE_CLOSED, GATE_OPEN, GATE_WIRE = "closed", "open", "wire"


class C2CWiredConnector(KVConnectorBase_V1, SupportsHMA):
    """The C2C wire, mounted in the engine's connector slot."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        cfg = getattr(vllm_config, "kv_transfer_config", None)
        extra = getattr(cfg, "kv_connector_extra_config", None) or {}
        self.log = logging.getLogger("c2c.wired").getChild("connector")
        self.gate = str(extra.get("c2c_gate", GATE_WIRE)).lower()
        if self.gate not in (GATE_CLOSED, GATE_OPEN, GATE_WIRE):
            self.log.warning("unknown c2c_gate %r: serving closed until told better", self.gate)
            self.gate = GATE_CLOSED
        self.wire_path = str(extra.get("c2c_wire") or os.environ.get(WIRE_ENV) or "")
        self._block_size = int(
            extra.get("c2c_block_size")
            or getattr(getattr(vllm_config, "cache_config", None), "block_size", 0)
            or 0
        )
        self.staging: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.wire: ResidentWire | None = None
        self._cards: dict[str, torch.Tensor] = {}
        self._faults = 0

    # -- scheduler side: the role rides with the request ────────────────────
    def build_connector_meta(self, request: Any = None, **kwargs: Any) -> dict | None:
        """Stamp the request's C2C role onto the per-request connector meta.

        The front puts ``{"c2c": {"role": "sharer", "pair": …, "peer":
        …}}`` in the extra body; the scheduler hands it to us here, and
        the worker reads it back off ``request.c2c``.
        """
        c2c = _c2c_of(request)
        if not c2c:
            return None
        return {
            "role": str(c2c.get("role") or ""),
            "pair": str(c2c.get("pair") or ""),
            "peer": str(c2c.get("peer") or ""),
            "self_req": str(getattr(request, "request_id", "") or c2c.get("self_req") or ""),
        }

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """v1 prefills everything: no shortcut claims what the wire has not proved.

        The matched-prefix shortcut — fused rows standing in for the
        receiver's own prefill, the Table 3 latency win — returns
        ``0, False`` until the live identity proof and an A/B against
        full-prefill answers have earned it. Until then, honesty is
        the speedup's parent.
        """
        return 0, False

    # -- worker side: where the rows live ───────────────────────────────────
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Take the live paged buffers; mount the wire unless told closed."""
        try:
            super().register_kv_caches(kv_caches)
        except AttributeError:
            pass  # a base that keeps no ledger of its own
        self._cards = dict(kv_caches)
        if self.wire is None and self.gate != GATE_CLOSED:
            self._build_wire()

    def _build_wire(self) -> None:
        """Fit the resident wire against the engine's own card.

        No card, no wire: the geometry comes from the model the engine
        itself loaded, never from a guess about shapes.
        """
        if not self.wire_path:
            self.log.info("no wire mounted at launch: serving the identity")
            self.gate = GATE_CLOSED
            return
        try:
            wired = [n for n in self._cards if _looks_paged(n)]
            if not wired:
                msg = "the engine paged no attention layers it will let us see"
                raise RuntimeError(msg)
            card = _card_from_config(self, layers=len(wired))
            if card is None:
                msg = "the engine would not state its card (model_config missing?)"
                raise RuntimeError(msg)
            wire = ResidentWire(receiver_geometry=card, sharer_geometry=card, device="cuda")
            wire.note_layer_names(wired, wired)
            wire.pin(
                False if self.gate == GATE_CLOSED else (True if self.gate == GATE_OPEN else None)
            )
            wire.load(self.wire_path)
            self.wire = wire
        except Exception as exc:
            self.log.error(
                "the wire would not mount (%s: %s): serving closed", exc.__class__.__name__, exc
            )
            self.wire, self.gate = None, GATE_CLOSED

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Park the rows this prefill touched, for whoever the wire will need them.

        Both sides save: the sharer's rows feed the projection, the
        receiver's own rows feed the residual — and a request the
        engine has not paged for this layer simply does not appear.
        """
        if self.gate == GATE_CLOSED or self.wire is None or layer_name not in self._cards:
            return
        try:
            for request in _requests_of(self):
                c2c = _c2c_of(request)
                if not c2c or c2c.get("role") not in ("sharer", "receiver"):
                    continue
                req_id = str(getattr(request, "request_id", "") or "")
                if not req_id:
                    continue
                rows = _extract(kv_layer, getattr(request, "slot_mapping", None), self._block_size)
                if rows is None:
                    continue
                self.staging.setdefault(req_id, {})[layer_name] = rows
        except Exception as exc:
            self._faulty("save", layer_name, exc)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Remember which receiver request this forward is loading for."""
        self._loading = next(
            (r for r in _requests_of(self) if (_c2c_of(r) or {}).get("role") == "receiver"), None
        )

    def wait_for_layer_load(self, layer_name: str, **kwargs: Any) -> None:
        """In the engine's own load window: fuse the layer, scatter it home."""
        request = getattr(self, "_loading", None)
        if (
            request is None
            or self.gate == GATE_CLOSED
            or self.wire is None
            or layer_name not in self._cards
        ):
            return
        try:
            from ...types import LayerSlice

            c2c = _c2c_of(request) or {}
            own_id = str(c2c.get("self_req") or getattr(request, "request_id", "") or "")
            own = self.staging.get(own_id, {})
            theirs = self.staging.get(str(c2c.get("peer") or ""), {})
            if layer_name not in theirs or layer_name not in own:
                return  # the sharer never reached this layer: the receiver keeps its own
            k_r, v_r = own[layer_name]
            k_s, v_s = theirs[layer_name]
            fused = self.wire.fuse_layer(
                LayerSlice(key=k_r, value=v_r), LayerSlice(key=k_s, value=v_s), layer=layer_name
            )
            _scatter(
                self._cards[layer_name],
                getattr(request, "slot_mapping", None),
                fused,
                self._block_size,
            )
        except Exception as exc:
            self._faulty("load", layer_name, exc)

    def wait_for_save(self) -> None:
        return None  # our save is a copy, and copies do not queue

    def requires_kv_delivery(self) -> bool:
        return False  # nothing travels: the wire lives where the caches do

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        for req in finished_req_ids:
            self.staging.pop(req, None)
        return None, None

    def request_finished(self, request_id: str, **kwargs: Any) -> None:
        self.staging.pop(str(request_id), None)

    # -- manners ────────────────────────────────────────────────────────────
    def _faulty(self, side: str, layer: str, exc: Exception) -> None:
        """A fault in the wire must never become a fault in the answer."""
        self._faults += 1
        self.log.error(
            "c2c %s fault at %s (%s: %s): the receiver keeps its own rows",
            side,
            layer,
            exc.__class__.__name__,
            exc,
        )

    def stats(self) -> dict:
        return {
            "gate": self.gate,
            "wire": self.wire_path or None,
            "faults": self._faults,
            "staged": sum(len(v) for v in self.staging.values()),
            "wire_stats": self.wire.stats() if self.wire is not None else None,
        }


# ─ engine-plumbing primitives, all borrowed from the engine's own example ─


def _looks_paged(name: str) -> bool:
    n = name.lower()
    return ("kv" in n or "attn" in n or "attention" in n) and "linear" not in n


def _c2c_of(request: Any) -> dict | None:
    """The request's c2c object, off the bare attribute or the proven ferry.

    The front stamps ``{"c2c": {...}}`` inside ``kv_transfer_params``,
    which the engine's protocol folds into ``extra_args`` and ferries to
    the worker (completion protocol, lines 362-365); a stubbed test may
    still wear it straight on the request.
    """
    if request is None:
        return None
    c2c = getattr(request, "c2c", None)
    if c2c is None:
        ferry = getattr(request, "extra_args", None) or {}
        c2c = (ferry.get("kv_transfer_params") or {}).get("c2c")
    if c2c is None and isinstance(request, dict):
        c2c = request.get("c2c") or (request.get("kv_transfer_params") or {}).get("c2c")
    return c2c if isinstance(c2c, dict) and c2c.get("role") else None


def _requests_of(connector: Any) -> list:
    meta = connector._get_connector_metadata()  # noqa: SLF001 — the engine's own hook
    return list(getattr(meta, "requests", None) or []) if meta is not None else []


def _extract(layer: torch.Tensor, slot_mapping: Any, block_size: int):
    """Rows this request touched, from the paged buffer (num_pages, 2, page_size, …).

    The engine's own gather, from its example connector, kept whole:
    slot_mapping indexes the flat token axis; the second axis is K then V.
    """
    if slot_mapping is None or block_size <= 0:
        return None
    slot_mapping = slot_mapping.to(layer.device, non_blocking=True)
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    rows = layer[block_idxs, :, offsets]  # [n_slots, 2, …]
    return (rows[:, 0].contiguous(), rows[:, 1].contiguous())


def _scatter(layer: torch.Tensor, slot_mapping: Any, fused, block_size: int) -> None:
    """Write the fused rows back into the same slots, in place. The mirror image of _extract.

    The assignment form, not ``index_put_``: this torch build rejects even
    the documented tuple-of-tensors call (a binding bug on this platform),
    and ``layer[i, j, k] = rows`` is the same scatter through the door
    every tensor has always opened. The unit tests pin exactly this shape.
    """
    if slot_mapping is None or block_size <= 0:
        msg = "no slots to write: the load window never saw this request"
        raise RuntimeError(msg)
    slot_mapping = slot_mapping.to(layer.device, non_blocking=True)
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    k_axis = torch.zeros_like(block_idxs)  # the second axis is K, then V — as tensors or nothing
    v_axis = torch.ones_like(block_idxs)
    layer[block_idxs, k_axis, offsets] = fused.key.to(layer.dtype)
    layer[block_idxs, v_axis, offsets] = fused.value.to(layer.dtype)


def _card_from_config(connector: Any, *, layers: int):
    """The card the engine states for the model it loaded, or None.

    Read from the engine's own config objects, never inferred from
    tensor shapes: a wire wearing a card that was guessed is worse
    than a server that serves the identity. ``layers`` is the count
    of *paged* layers — the wire's stack, not the model's whole.
    """
    from ...types import AttentionKind, LayerGeometry

    vc = getattr(connector, "vllm_config", None) or getattr(connector, "_vllm_config", None)
    mc = getattr(vc, "model_config", None) if vc is not None else None
    text = getattr(mc, "text_config", None) or mc
    if text is None:
        return None
    try:
        return LayerGeometry(
            layers=int(layers),
            hidden_size=int(getattr(text, "hidden_size")),
            num_heads=int(getattr(text, "num_attention_heads")),
            head_size=int(getattr(text, "head_dim")),
            num_key_value_heads=int(getattr(text, "num_key_value_heads")),
            attention=AttentionKind.GQA,
        )
    except (TypeError, ValueError, AttributeError):
        return None
