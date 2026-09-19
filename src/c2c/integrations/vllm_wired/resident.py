"""The resident wire: the fuser, living where the caches live.

The fuser (:class:`c2c.fuser.core.Fuser`) is the only trainable thing in
the system — tens of megabytes beside the weights. Running it inside
the engine's worker means a fusion reads the engine's own rows and
writes fused rows back through the engine's own load window; the cache
never takes a journey the fp8 quantizer has to survive twice.

The contract is the codebase's own unit, end to end: the connector
hands ``fuse()`` one :class:`~c2c.types.LayerSlice` per wired layer —
``key`` and ``value`` each ``[n_tokens, kv_heads, head_size]``, on the
engine's device, in the model dtype (the connector, and only the
connector, knows the engine's storage dtype and brings the engine's own
dequantization with it). The wire hands back fused slices, produced by
the one true :meth:`Fuser.forward`. Nothing in this module reimplements
Eq. (3); if the equation changes, it changes once, there.

**Identity before intelligence.** With no wire resident, or the gate
pinned closed, ``fuse()`` returns the receiver's own slices — never
touched: no fuser call, no arithmetic. That is what makes the identity
landing provable: a wired server with a closed gate answers exactly as
the unwired one does, byte for byte, whatever the fp8 path does.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import torch

from ...align.layers import terminal_mapping
from ...config import TrainRecipe
from ...fuser.core import Fuser
from ...train.scheme import load_checkpoint_blob
from ...types import LayeredCache, LayerGeometry, LayerSlice

__all__ = ["ResidentWire", "WIRE_ENV"]

#: environment seam: the connector finds the wire the worker was launched with
WIRE_ENV = "C2C_WIRED_WIRE"

#: fusion staging, in tokens: the window the wire was trained in — and the
#: engine's own batch ceiling (--max-num-batched-tokens 2048). The weighting
#: pools along the token axis, so the chunk size is not a staging detail:
#: it is the shape the fusion means.
DEFAULT_CHUNK = TrainRecipe().max_seq_length


def _looks_paged(name: str) -> bool:
    """A layer name of the paged-KV kind — a fallback for a bare card.

    The engine's own ``register_kv_caches`` keys are authoritative when
    the connector supplies them. Linear-attention layers keep
    conv/recurrent state, not per-token rows, and pass through.
    """
    n = name.lower()
    return ("kv" in n or "attn" in n or "attention" in n) and "linear" not in n


class ResidentWire:
    """The fuser, resident on the engine's device. One per worker.

    Life cycle::

        wire = ResidentWire(receiver_geometry=card_r, sharer_geometry=card_s)
        wire.note_layer_names(live_keys)              # from register_kv_caches
        wire.load("/volumes/c2c/wires/flash-wire.safetensors")   # or pin(False)
        fused = wire.fuse(receiver_slices, sharer_slices)         # dict in, dict out
    """

    def __init__(
        self,
        *,
        receiver_geometry: LayerGeometry,
        sharer_geometry: LayerGeometry | None = None,
        device: str | None = None,
        chunk: int = DEFAULT_CHUNK,
        logger: logging.Logger | None = None,
    ) -> None:
        self.receiver_geometry = receiver_geometry
        self.sharer_geometry = sharer_geometry or receiver_geometry
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk = max(1, int(chunk))
        self.log = (logger or logging.getLogger("c2c.wired")).getChild("resident")
        self.fuser: Fuser | None = None
        self.pinned: bool | None = None  # None: follow the wire's own gates
        self.receiver_layers: list[str] = []
        self.sharer_layers: list[str] = []
        self.wire_path: str | None = None

    # -- the engine's handshakes ─────────────────────────────────────────────
    def note_layer_names(
        self, receiver_layers: Sequence[str], sharer_layers: Sequence[str] | None = None
    ) -> None:
        """Record which live cache layers the wire may touch.

        The connector calls this from ``register_kv_caches`` with the
        engine's own keys — those, not a guess, decide what is wired.
        """
        self.receiver_layers = [n for n in receiver_layers if _looks_paged(n)]
        pool = list(sharer_layers) if sharer_layers is not None else list(receiver_layers)
        self.sharer_layers = [n for n in pool if _looks_paged(n)]
        if not self.receiver_layers or not self.sharer_layers:
            msg = "no paged-cache layers among the names given: nothing to wire"
            raise RuntimeError(msg)

    def pin(self, gate: bool | None) -> None:
        """True: fuse with the gates forced open; False: identity; None: the wire's."""
        self.pinned = gate
        if self.fuser is not None:
            self.fuser.gate.training = False

    def load(self, path: str, *, mapping: Sequence[int] | None = None) -> dict:
        """Fit a wire file, validated against the engine that must wear it.

        A wire trained for another shape is refused here, loudly, where
        the log can see it — not at the first fusion, where the damage
        is silent. The cards the wire carries must agree with the cards
        the connector read off the engine.
        """
        if not self.receiver_layers:
            msg = "note_layer_names before load: the wire needs the engine's layers"
            raise RuntimeError(msg)
        blob = load_checkpoint_blob(path)
        cards = blob.get("geometry") or {}
        geo_r = _geometry_of(cards.get("receiver"), self.receiver_geometry, "receiver")
        geo_s = _geometry_of(cards.get("sharer"), self.sharer_geometry, "sharer")
        want = len(self.receiver_layers)
        if geo_r.layers != want:
            msg = (
                f"the wire at {path!r} was trained for {geo_r.layers} wired layers; "
                f"this engine pages {want}"
            )
            raise ValueError(msg)
        claims = (
            ("receiver", geo_r, self.receiver_geometry),
            ("sharer", geo_s, self.sharer_geometry),
        )
        for side, claim, live in claims:
            if _shape(claim) != _shape(live):
                msg = f"the wire's {side} card disagrees with the engine: {claim!r} vs {live!r}"
                raise ValueError(msg)
        layer_map: list[int] = (
            list(mapping) if mapping is not None else list(blob.get("layer_mapping") or [])
        )
        if not layer_map:
            layer_map = list(terminal_mapping(geo_r.layers, geo_s.layers))
        fuser = Fuser(geo_r, geo_s, layer_map)
        fuser.load_state_dict(blob["state_dict"])
        fuser.to(self.device)
        fuser.eval()
        self.fuser = fuser
        self.wire_path = str(path)
        self.log.info(
            "the wire is resident: %d layers, %s on %s%s",
            want,
            self._param_report(),
            self.device,
            " — gate pinned closed, identity" if self.pinned is False else "",
        )
        return self.stats()

    # -- the fusion step ────────────────────────────────────────────────────
    def fuse(
        self,
        receiver_slices: Mapping[str, LayerSlice],
        sharer_slices: Mapping[str, LayerSlice],
    ) -> dict[str, LayerSlice]:
        """Fuse one request's slices, layer for layer, through the one fuser.

        Keys the wire does not own come straight back. The chunk is the
        trained window, not a staging hint: the dynamic weighting pools
        mean and max along the token axis, so a chunk sees only its own
        statistics — and the only statistics the wire ever learned under
        are the whole of a training sequence. Prompts within the window
        fuse in one pass, the trained shape exactly; longer prompts fuse
        in trained-size blocks, the semantics the engine's own chunked
        prefill gives every other module. (A single pass over 524288
        tokens would stage ~13 GiB of rows: the memory budget keeps that
        door shut.)
        """
        if self.pinned is False or self.fuser is None:
            return dict(receiver_slices)  # identity: the receiver's own, never touched
        names = self.receiver_layers
        missing = [n for n in names if n not in receiver_slices or n not in sharer_slices]
        if missing:
            msg = f"slices missing for wired layers: {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}"
            raise KeyError(msg)
        restore = None
        if self.pinned is True:
            from dataclasses import replace

            restore = self.fuser.fuser_config
            self.fuser.fuser_config = replace(restore, gating=False)  # Table 8: weights of ones
        try:
            n_tokens = int(receiver_slices[names[0]].key.shape[0])
            if n_tokens <= self.chunk:
                fused = self._fuser_pass(receiver_slices, sharer_slices, 0, n_tokens)
            else:
                pieces: dict[str, list] = {n: [] for n in names}
                for off in range(0, n_tokens, self.chunk):
                    got = self._fuser_pass(
                        receiver_slices, sharer_slices, off, min(self.chunk, n_tokens - off)
                    )
                    for n in names:
                        pieces[n].append(got[n])
                fused = {n: _cat_slices(ps) for n, ps in pieces.items()}
            out = dict(receiver_slices)
            out.update(fused)
            return out
        finally:
            if restore is not None:
                self.fuser.fuser_config = restore

    def _fuser_pass(
        self, receiver_slices, sharer_slices, off: int, length: int
    ) -> dict[str, LayerSlice]:
        """One pass of the only call site of Eq. (3), over a token window."""
        fuser = self.fuser
        if fuser is None:
            msg = "the wire fuses only once it is resident: load before fuse"
            raise RuntimeError(msg)
        window = slice(off, off + length)
        cache_r = LayeredCache([_window(receiver_slices[n], window) for n in self.receiver_layers])
        # the sharer's side of the mapping: same order the wire was trained in
        cache_s = LayeredCache(
            [
                _window(sharer_slices[self.sharer_layers_for(n)], window)
                for n in self.receiver_layers
            ]
        )
        fused = fuser(cache_r, cache_s)
        return {n: fused[i] for i, n in enumerate(self.receiver_layers)}

    def fuse_layer(
        self, receiver_slice: LayerSlice, sharer_slice: LayerSlice, *, layer: str
    ) -> LayerSlice:
        """Fuse one layer's slice — the connector's per-layer load window.

        Same equation, same gate, one pair at a time: the engine loads
        layer by layer, and the connector may only touch what the
        engine hands it. A closed gate, or a blend that has not
        reached this layer, returns the receiver's own slice untouched.
        """
        if self.pinned is False or self.fuser is None:
            return receiver_slice
        from ...fuser.core import _rows, _slice

        n = self.receiver_layers.index(layer)
        if not self._blend_allows(n):
            return receiver_slice
        weights = self.fuser.gate(training=False)
        if self.pinned is not True:
            if not bool(weights[n].item()):
                return receiver_slice  # closed gate: the receiver's own, never touched
            weight = weights[n]
        else:
            weight = torch.ones_like(weights[n])
        r_rows = _rows(receiver_slice)
        delta = self.fuser.pairs[n](r_rows, _rows(sharer_slice))
        return _slice(
            r_rows + delta.to(r_rows.dtype) * weight.to(r_rows.dtype), self.fuser.receiver
        )

    def _blend_allows(self, n: int) -> bool:
        """Whether progressive blending lets this layer reach the fuser."""
        fuser = self.fuser
        if fuser is None:
            return False
        blend = fuser.blend_config
        if blend.fraction >= 1.0:
            return True
        total = len(self.receiver_layers)
        reach = int(blend.fraction * total + 0.9999)
        return n >= total - reach if "latter" in str(blend.direction) else n < reach

    def sharer_layers_for(self, receiver_layer: str) -> str:
        """The sharer layer mapped to this receiver layer, by trained mapping."""
        if self.fuser is None:
            return receiver_layer
        n = self.receiver_layers.index(receiver_layer)
        g = self.fuser.mapping[n]
        return self.sharer_layers[g]

    # -- the bill ────────────────────────────────────────────────────────────
    def _param_report(self) -> str:
        if self.fuser is None:
            return "no wire resident"
        params = sum(p.numel() for p in self.fuser.parameters())
        return f"{params * 4 / 2**20:0.1f} MiB"

    def stats(self) -> dict:
        params = 0 if self.fuser is None else int(sum(p.numel() for p in self.fuser.parameters()))
        return {
            "layers_total": len(self.receiver_layers),
            "layers_fusable": len(self.receiver_layers),
            "wire_params": params,
            "wire_mib": round(params * 4 / 2**20, 1),
            "device": str(self.device),
            "gate": "pinned-closed"
            if self.pinned is False
            else ("resident" if self.fuser else "none"),
            "chunk": self.chunk,
            "wire": self.wire_path,
        }


def _window(sl: LayerSlice, window: slice) -> LayerSlice:
    return LayerSlice(key=sl.key[window], value=sl.value[window])


def _cat_slices(pieces: list[LayerSlice]) -> LayerSlice:
    return LayerSlice(
        key=torch.cat([p.key for p in pieces], dim=0),
        value=torch.cat([p.value for p in pieces], dim=0),
    )


def _geometry_of(card, fallback: LayerGeometry, side: str) -> LayerGeometry:
    """The wire's claimed card for one side, or the engine's if it kept quiet."""
    if not card:
        return fallback
    if isinstance(card, LayerGeometry):
        return card
    try:
        return LayerGeometry(**dict(card))
    except Exception as exc:  # a hand-written card that will not parse
        raise ValueError(f"the wire's {side} card will not parse: {exc}") from exc


def _shape(g: LayerGeometry) -> tuple:
    """The shape facts of a card — its geometry, not its name."""
    return (g.layers, g.hidden_size, g.num_heads, g.head_size, g.num_key_value_heads, g.attention)
