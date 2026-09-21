"""The wired connector, tested where the engine is not: on stubs.

The connector imports engine symbols at module top — by design: it
runs in a worker. These tests install a minimal ``vllm`` namespace for
the duration of each test (and remove it after: the rest of the suite
must keep believing vLLM is absent when it is), then drive the
connector exactly as the engine does — ``save_kv_layer`` with a paged
buffer and per-request metadata, then ``start_load_kv`` /
``wait_for_layer_load`` around the forward's load window.

The four proofs, in the order the milestone demands them:

    identity     gate closed: nothing stages, the buffer never moves
    parity       gate resident: the written rows equal the wire's own
                 per-layer fusion, computed direct
    fail-closed  poisoned row shape: the fault is counted, nothing
                 raises, every layer keeps the receiver's own rows
    release      a finished request sheds its staging
"""

from __future__ import annotations

import abc
import importlib
import sys
import tempfile
import types as pytypes
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the wire learns in torch")

_VLLM_STUBS = (
    "vllm",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
)
_CONNECTOR = "c2c.integrations.vllm_wired.connector"


class _StubBase(abc.ABC):
    """The base a connector under test may call — as stern as the engines are.

    The real base demands its abstractmethods; a stub that flatters lets every
    connector sin pass unseen. The debts here are the real ones, verbatim.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config, self.role, self.kv_cache_config = vllm_config, role, kv_cache_config

    def register_kv_caches(self, kv_caches):
        return None

    @abc.abstractmethod
    def start_load_kv(self, forward_context, **kwargs): ...

    @abc.abstractmethod
    def update_state_after_alloc(self, request, blocks, num_external_tokens): ...


class _StubMarker(abc.ABC):
    """The HMA marker, demanding of its one abstract method, as the real one does."""

    @abc.abstractmethod
    def request_finished_all_groups(self, request, block_ids): ...

class _StubMeta(abc.ABC):
    """The metadata base: as plain as the engines make it, and no less."""


class _Req:
    def __init__(self, request_id, slot_mapping, c2c):
        self.request_id, self.slot_mapping, self.c2c = request_id, slot_mapping, c2c


class _Meta:
    def __init__(self, requests):
        self.requests = requests


def _ns(**kwargs):
    return pytypes.SimpleNamespace(**kwargs)


@pytest.fixture()
def wired():
    """The connector module, imported against a stub engine, unstubbèd after."""
    for name in (*_VLLM_STUBS, _CONNECTOR):
        sys.modules.pop(name, None)
    for name in _VLLM_STUBS:
        sys.modules[name] = pytypes.ModuleType(name)
    stub_base = sys.modules[_VLLM_STUBS[-1]]
    stub_base.KVConnectorBase_V1 = _StubBase
    stub_base.KVConnectorMetadata = _StubMeta
    stub_base.SupportsHMA = _StubMarker
    try:
        yield importlib.import_module(_CONNECTOR)
    finally:
        for name in (*_VLLM_STUBS, _CONNECTOR):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def toy():
    """Reference engines and a wire trained for them — once for the module."""
    from c2c.config import TrainRecipe
    from c2c.fuser.core import Fuser
    from c2c.integrations.reference import ReferenceConfig, ReferenceEngine
    from c2c.train.scheme import Sample, Trainer

    r = ReferenceEngine(ReferenceConfig(name="receiver-mini", seed=42))
    s = ReferenceEngine(ReferenceConfig(name="sharer-mini", seed=9))
    fuser = Fuser(r.geometry, s.geometry, list(range(r.geometry.layers)))
    trainer = Trainer(
        fuser,
        r,
        s,
        r,
        receiver_tokenizer=r,
        sharer_tokenizer=s,
        recipe=TrainRecipe(dataset="unused", max_seq_length=64, total_steps=8, epochs=1),
    )
    trainer.fit([Sample("a b c", "d e")] * 4, epochs=1)
    path = Path(tempfile.mkdtemp(prefix="c2c-wired-")) / "unit-wire.safetensors"
    trainer.save_checkpoint(str(path))
    return r, s, str(path)


def _build(wired_module, toy, gate):
    """A connector over empty paged buffers, with or without a resident wire."""
    from c2c.integrations.vllm_wired.resident import ResidentWire

    r, s, path = toy
    cfg = _ns(
        kv_transfer_config=_ns(kv_connector_extra_config={"c2c_gate": gate, "c2c_wire": path}),
        cache_config=_ns(block_size=4),
        model_config=_ns(text_config=_ns()),
    )
    conn = wired_module.C2CWiredConnector(cfg, role=None, kv_cache_config=None)
    names = [f"model.layers.{i}.self_attn.kv" for i in range(r.geometry.layers)]
    buffers = {
        n: torch.zeros(
            (8, 2, 4, r.geometry.num_key_value_heads, r.geometry.head_size), dtype=torch.float32
        )
        for n in names
    }
    conn._cards = buffers
    if gate != "closed":
        wire = ResidentWire(receiver_geometry=r.geometry, sharer_geometry=s.geometry, device="cpu")
        wire.note_layer_names(names, names)
        wire.load(path)
        wire.pin(True if gate == "open" else None)
        conn.wire = wire
    conn._meta = None
    conn._get_connector_metadata = lambda: conn._meta
    return conn, names, buffers


def _receiver_meta(conn, slots):
    conn._meta = _Meta(
        [_Req("R", slots, {"role": "receiver", "pair": "P", "peer": "S", "self_req": "R"})]
    )


def _prefill(conn, buffers, names, request_id, role, peer, slots, seeds):
    """One request's prefill: fresh rows in the buffer, then the save pass sees them."""
    for i, n in enumerate(names):
        gen = torch.Generator().manual_seed(seeds + i)
        buffers[n].copy_(torch.randn(buffers[n].shape, generator=gen, dtype=torch.float32))
    conn._meta = _Meta(
        [_Req(request_id, slots, {"role": role, "pair": "P", "peer": peer, "self_req": request_id})]
    )
    for n in names:
        conn.save_kv_layer(n, buffers[n], None)


def test_identity_the_buffer_never_moves(wired, toy):
    conn, names, buffers = _build(wired, toy, "closed")
    slots = torch.arange(24)
    _prefill(conn, buffers, names, "S", "sharer", "", slots, seeds=1)
    before = {n: buffers[n].clone() for n in names}  # after the fill, before the load
    _receiver_meta(conn, slots)
    conn.start_load_kv(None)
    for n in names:
        conn.wait_for_layer_load(n)
    assert conn.staging == {}  # a closed gate does not even stage
    assert all(torch.equal(buffers[n], before[n]) for n in names)
    assert conn._faults == 0


def test_parity_written_rows_are_the_fusers_own(wired, toy):
    conn, names, buffers = _build(wired, toy, "wire")
    slots = torch.arange(24)
    _prefill(conn, buffers, names, "S", "sharer", "", slots, seeds=11)
    _prefill(conn, buffers, names, "R", "receiver", "S", slots, seeds=37)
    _receiver_meta(conn, slots)
    conn.start_load_kv(None)
    for n in names:
        conn.wait_for_layer_load(n)
    assert conn._faults == 0
    from c2c.types import LayerSlice

    block_idxs, offsets = slots // 4, slots % 4
    for n in names:
        k_r, v_r = conn.staging["R"][n]
        k_s, v_s = conn.staging["S"][n]
        expected = conn.wire.fuse_layer(
            LayerSlice(key=k_r, value=v_r), LayerSlice(key=k_s, value=v_s), layer=n
        )
        assert torch.equal(buffers[n][block_idxs, 0, offsets], expected.key.to(buffers[n].dtype))
        assert torch.equal(buffers[n][block_idxs, 1, offsets], expected.value.to(buffers[n].dtype))


def test_fail_closed_a_poisoned_row_keeps_the_answer(wired, toy):
    conn, names, buffers = _build(wired, toy, "wire")
    conn.wire.pin(True)  # every gate forced open: no closed gate may hide the poison
    slots = torch.arange(24)
    _prefill(conn, buffers, names, "S", "sharer", "", slots, seeds=5)
    _prefill(conn, buffers, names, "R", "receiver", "S", slots, seeds=6)
    g = toy[0].geometry
    poison = torch.zeros((24, g.num_key_value_heads, g.head_size + 1))
    conn.staging["S"] = dict.fromkeys(names, (poison, poison))
    _receiver_meta(conn, slots)
    conn.start_load_kv(None)
    for n in names:
        conn.wait_for_layer_load(n)  # the poison must not reach the answer
    assert conn._faults >= len(names)  # every layer saw the fault, and kept its own
    block_idxs, offsets = slots // 4, slots % 4
    for n in names:
        k_r, v_r = conn.staging["R"][n]
        assert torch.equal(buffers[n][block_idxs, 0, offsets], k_r)
        assert torch.equal(buffers[n][block_idxs, 1, offsets], v_r)


def test_release_a_finished_request_sheds_its_staging(wired, toy):
    conn, names, buffers = _build(wired, toy, "wire")
    slots = torch.arange(24)
    _prefill(conn, buffers, names, "S", "sharer", "", slots, seeds=3)
    assert conn.staging["S"]
    conn.request_finished("S")
    assert "S" not in conn.staging


def test_c2c_of_follows_the_engines_own_ferry(wired):
    """kv_transfer_params is the road the completion protocol ferries."""
    straight = wired._c2c_of(_ns(extra_args={"kv_transfer_params": {"c2c": {"role": "sharer"}}}))
    assert straight == {"role": "sharer"}
    as_dict = wired._c2c_of({"kv_transfer_params": {"c2c": {"role": "receiver"}}})
    assert as_dict == {"role": "receiver"}
    assert wired._c2c_of(_ns(extra_args={})) is None
    assert wired._c2c_of(None) is None


def test_the_class_boots_and_the_stubs_do_not_flatter(wired):
    """The net that missed a TypeError at the engines gate, cut once and for all.

    An ABC base with unmet abstractmethods is uninstantiable, and the engine
    builds its connector at core init — so a debt inherited is a server dead.
    The assertion is the gate; the construction, the proof it opens.
    """
    assert wired.C2CWiredConnector.__abstractmethods__ == frozenset()
    cfg = _ns(
        kv_transfer_config=_ns(kv_connector_extra_config={"c2c_gate": "closed"}),
        cache_config=_ns(block_size=4),
        model_config=_ns(text_config=_ns()),
    )
    conn = wired.C2CWiredConnector(cfg, _ns(), _ns())
    assert conn.gate == "closed"
    assert conn.request_finished_all_groups(_ns(request_id="gone"), ([1], [2])) == (False, None)


def test_a_refused_ledger_degrades_to_the_identity(wired, monkeypatch):
    """The base refusing its bookkeeping is a card that disagrees: closed gate, alive worker.

    Before the hardening, any exception from the bases register_kv_caches
    propagated into the workers init and the whole engine core died at
    boot — the very failure mode the launcher must never turn into a
    downtime. Now the fault is logged, the gate falls, the receiver speaks
    from its own cache.
    """

    def _refuse(self, kv_caches):
        raise RuntimeError("the base keeps no ledger for cards like these")

    monkeypatch.setattr(_StubBase, "register_kv_caches", _refuse)
    cfg = _ns(
        kv_transfer_config=_ns(kv_connector_extra_config={"c2c_gate": "wire"}),
        cache_config=_ns(block_size=4),
        model_config=_ns(text_config=_ns()),
    )
    conn = wired.C2CWiredConnector(cfg, _ns(), _ns())
    conn.register_kv_caches({"model.layers.0.self_attn": torch.zeros(2, 2, 4, 2, 4)})
    assert conn.gate == wired.GATE_CLOSED  # closed, not dead
    assert conn.wire is None  # no mount attempted on a refused card
    conn.wait_for_layer_load("model.layers.0.self_attn")  # the path, walkable and silent
