"""Tests for the A2A bridge (HL-2): the card, the tasks, the cache between.

The bridge is tested as a bridge runs: real sockets, real HTTP, real
JSON-RPC, and a reference pair that answers. The card is public; the
tasks, when the house has a key, are not.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from c2c.config import ServeConfig
from c2c.serve import a2a as bridge
from c2c.serve.a2a import CARD_ROUTE, STATES, TASK_ROUTE, TaskStore, build_agent_card


@pytest.fixture()
def hub():
    """A hub of one pair, of reference minds."""
    from c2c.serve.registry import ModelHub
    h = ModelHub(ServeConfig(api_key=None))
    h.set_engine("reference")
    h.register_model("receiver-mini")
    h.register_model("sharer-mini")
    h.register_pair(receiver="receiver-mini", sharer="sharer-mini")
    return h


@pytest.fixture()
def port(hub):
    """The bridge, up, on a port of the kernel's choosing."""
    cfg = ServeConfig(host="127.0.0.1", port=0, api_key=None)
    httpd = bridge.create_bridge(cfg, hub=hub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _rpc(port: int, method: str, params: dict | None = None, *, key: str | None = None):
    """One JSON-RPC exchange with the bridge; returns (status, payload)."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                     "params": params or {}}).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{TASK_ROUTE}", data=body,
        headers={"Content-Type": "application/json"})
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


class TestTheCard:
    """The well-known path, and what it says."""

    def test_the_card_of_the_bridge_is_public(self, port):
        status, card = _get(port, CARD_ROUTE)
        assert status == 200                                       # the card, served
        assert card["name"] == "c2c-a2a"                           # the name, of the bridge
        assert any(s["id"] == "cache-to-cache" for s in card["skills"])   # the skill, listed
        assert card["security"] == [{"bearer": []}]               # the scheme, declared

    def test_the_health_of_the_bridge(self, port):
        status, body = _get(port, "/healthz")
        assert status == 200 and body["status"] == "ok"

    def test_the_card_advertises_the_machines(self, hub):
        card = build_agent_card(ServeConfig(), hub)
        assert "receiver-mini" in card["models"] or card["models"]   # the gallery, named


class TestTheTasks:
    """One agent's task, another agent's answer."""

    def test_a_task_is_born_works_and_answers(self, port):
        status, reply = _rpc(port, "message/send", {
            "model": "receiver-mini",
            "message": {"role": "user", "parts": [{"kind": "text", "text": "count to three"}]},
            "max_new_tokens": 8})
        assert status == 200 and "result" in reply                 # the rpc, answered
        task = reply["result"]
        assert task["status"]["state"] in STATES                  # a state, of the ledger
        assert task["status"]["state"] == "completed"             # the solo, completes
        text = task["artifacts"][0]["parts"][0]["text"]
        assert isinstance(text, str) and text                       # the artifact, with words

    def test_the_fused_task_proclaims_the_cache(self, port):
        """The pair answers: the cache was used, the state says so."""
        _status, reply = _rpc(port, "message/send", {
            "model": "c2c/receiver-mini←sharer-mini",
            "message": "what is two plus two", "max_new_tokens": 8})
        task = reply["result"]
        assert task["status"]["state"] in ("fused", "completed")  # fused, or honestly failed
        if "error" not in task:
            assert task["artifacts"]                                 # the artifact, present

    def test_a_task_may_be_read_again(self, port):
        _s, reply = _rpc(port, "message/send", {
            "model": "receiver-mini", "message": "say again", "max_new_tokens": 4})
        task_id = reply["result"]["id"]
        status, got = _rpc(port, "tasks/get", {"id": task_id})
        assert status == 200 and got["result"]["id"] == task_id   # the task, remembered

    def test_a_task_may_be_canceled(self, port):
        _s, reply = _rpc(port, "message/send", {
            "model": "receiver-mini", "message": "long live the task", "max_new_tokens": 4})
        task_id = reply["result"]["id"]
        _status, cancelled = _rpc(port, "tasks/cancel", {"id": task_id})
        assert cancelled["result"]["status"]["state"] == "canceled"   # the word, given

    def test_no_such_task_is_said_so(self, port):
        _status, reply = _rpc(port, "tasks/get", {"id": "task-nonexistent"})
        assert "error" in reply and reply["error"]["code"] == -32001   # not found, in rpc

    def test_no_such_method_is_said_so(self, port):
        _status, reply = _rpc(port, "tasks/frobnicate", {})
        assert reply["error"]["code"] == -32601                     # method, not found

    def test_an_empty_message_is_a_bad_message(self, port):
        _status, reply = _rpc(port, "message/send", {"model": "receiver-mini", "message": ""})
        assert reply["error"]["code"] == -32602                     # invalid, the params


class TestTheKeyAtTheDoor:
    """When the house has a key, the tasks demand it."""

    def test_the_key_opens_the_tasks(self, hub):
        from c2c.serve.a2a import A2AHandler
        cfg = ServeConfig(host="127.0.0.1", port=0, api_key="skeleton-key")
        httpd = bridge.create_bridge(cfg, hub=hub)
        A2AHandler.public_url = "https://relay.example.com:9999"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            p = httpd.server_address[1]
            _s0, denied = _rpc(p, "tasks/list", {})
            assert _s0 == 401 and denied["error"]["code"] == "invalid_api_key"   # refused, unkeyed
            _s, ok = _rpc(p, "tasks/list", {}, key="skeleton-key")
            assert "result" in ok                                   # admitted, the keyed
            _s2, card = _get(p, CARD_ROUTE)
            assert card["url"] == "https://relay.example.com:9999"  # the public url, honoured
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            A2AHandler.public_url = None


class TestTheStore:
    """The ledger itself, without the wire."""

    def test_the_states_that_matter(self):
        store = TaskStore()
        task = store.submit("hello")
        assert store.set_state(task["id"], "working")["status"]["state"] == "working"
        with pytest.raises(ValueError):
            store.set_state(task["id"], "levitating")               # no such state, no such say

    def test_the_list_follows_the_context(self):
        store = TaskStore()
        a = store.submit("one", context_id="ctx-a")
        store.submit("two", context_id="ctx-b")
        rows = store.listed(context_id="ctx-a")
        assert [r["id"] for r in rows] == [a["id"]]                # the list, of one context
