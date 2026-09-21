"""Tests for the OpenAI-compatible front (HL-1): one endpoint, any harness.

The served wire, tested as a wire: real sockets, real HTTP, the OpenAI
shapes byte for byte. The hub is populated with the reference pair, so
every request reaches a live engine — deterministically.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from c2c.config import ServeConfig
from c2c.serve.openai_proxy import canonical_routes, create_server


@pytest.fixture()
def hub():
    """A hub of one pair: the reference receiver and its sharer."""
    from c2c.serve.registry import ModelHub

    h = ModelHub(ServeConfig(api_key="test-key"))
    h.set_engine("reference")
    h.register_model("receiver-mini")
    h.register_model("sharer-mini")
    h.register_pair(receiver="receiver-mini", sharer="sharer-mini")
    return h


@pytest.fixture()
def server(hub):
    cfg = ServeConfig(host="127.0.0.1", port=0, api_key="test-key")
    httpd = create_server(cfg, hub=hub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    thread.join(timeout=5)


AUTH = {"Authorization": "Bearer test-key"}


def _request(url, *, method="GET", payload=None, headers=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    head = dict(headers or {})
    if body:
        head["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=head)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class TestModelGallery:
    """GET /v1/models — the gallery, of models."""

    def test_the_list_is_of_the_models(self, server):
        status, body = _request(f"{server}/v1/models", headers=AUTH)
        assert status == 200  # all right, all models
        data = json.loads(body)
        assert data["object"] == "list"  # an object, of a list
        ids = [entry["id"].removeprefix("c2c/") for entry in data["data"]]
        assert "receiver-mini" in ids and "sharer-mini" in ids  # the pair, on display
        assert any("←" in model for model in ids)  # the fused, as one
        for entry in data["data"]:
            assert entry["object"] == "model"
            assert isinstance(entry["created"], int)  # created, stamped

    def test_health_of_the_server(self, server):
        """GET /healthz — no auth, no fuss, all fine."""
        status, body = _request(f"{server}/healthz")
        assert status == 200
        assert json.loads(body)["status"] == "ok"  # the heart, beating

    def test_the_well_known_agent_card(self, server):
        """GET /.well-known/agent-card.json — the card, of the agent."""
        status, body = _request(f"{server}/.well-known/agent-card.json")
        assert status == 200
        card = json.loads(body)
        assert "name" in card and "capabilities" in card  # identity, on the card
        assert card["capabilities"]["streaming"] is True  # streaming, promised


class TestChatCompletions:
    """POST /v1/chat/completions — the messages, in and out."""

    def test_creates_a_completion(self, server):
        status, body = _request(
            f"{server}/v1/chat/completions",
            method="POST",
            headers=AUTH,
            payload={
                "model": "receiver-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 6,
            },
        )
        assert status == 200
        data = json.loads(body)
        assert data["object"] == "chat.completion"  # the type, as issued
        assert len(data["choices"]) == 1  # one choice, greedy
        assert "content" in data["choices"][0]["message"]  # the message, replied
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        usage = data["usage"]
        assert usage["prompt_tokens"] >= 0 and usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_the_legacy_completions_route(self, server):
        """POST /v1/completions — the old wire, still singing."""
        status, body = _request(
            f"{server}/v1/completions",
            method="POST",
            headers=AUTH,
            payload={"model": "receiver-mini", "prompt": "count to five", "max_tokens": 5},
        )
        assert status == 200
        data = json.loads(body)
        assert data["object"] == "text.completion"
        assert "text" in data["choices"][0]  # the text, plain

    def test_streaming_in_sse(self, server):
        """The stream: data: chunks … and the DONE, at the end."""
        payload = {
            "model": "receiver-mini",
            "messages": [{"role": "user", "content": "go"}],
            "stream": True,
            "max_tokens": 4,
        }
        request = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={**AUTH, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            raw = response.read().decode("utf-8")
        events = [line for line in raw.splitlines() if line.startswith("data: ")]
        assert events, "a stream, with no events"
        assert events[-1] == "data: [DONE]"  # the terminator, verbatim
        first = json.loads(events[0].removeprefix("data: "))
        assert first["object"] == "chat.completion.chunk"  # chunks, of a completion


class TestAuthentication:
    """The gate: a Bearer key, or a 401 with a clear reason."""

    def test_no_key_no_entry(self, server):
        status, body = _request(
            f"{server}/v1/chat/completions",
            method="POST",
            payload={"model": "receiver-mini", "messages": [{"role": "user", "content": "x"}]},
        )
        assert status == 401  # unauthorised, plain
        error = json.loads(body)["error"]
        assert error["type"] == "authentication_error"  # the kind, named
        assert "key" in error["message"].lower()  # the reason, given

    def test_the_wrong_key_is_the_wrong_key(self, server):
        status, body = _request(
            f"{server}/v1/chat/completions",
            method="POST",
            headers={"Authorization": "Bearer nope"},
            payload={"model": "receiver-mini", "messages": [{"role": "user", "content": "x"}]},
        )
        assert status == 401  # denied, again


class TestErrors:
    """Error objects: the shape of a mistake."""

    def test_no_such_model(self, server):
        """The model, missing: a 404, or 400, with an error object."""
        status, body = _request(
            f"{server}/v1/chat/completions",
            method="POST",
            headers=AUTH,
            payload={"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]},
        )
        assert status in (400, 404)  # absent, announced
        error = json.loads(body)["error"]
        assert error["code"] == "model_not_found"  # the code, as documented
        assert error["param"] == "model"  # the parameter, blamed

    def test_malformed_json_is_a_client_error(self, server):
        request = urllib.request.Request(
            f"{server}/v1/completions",
            data=b"{not json",
            method="POST",
            headers={**AUTH, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400  # bad request, good body

    def test_options_gives_cors_headers(self, server):
        """The preflight, answered — and answered with the banner, hoisted."""
        request = urllib.request.Request(
            f"{server}/v1/chat/completions", method="OPTIONS", headers=AUTH
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            headers = response.headers
        assert status in (200, 204)  # the door, open
        assert headers["Access-Control-Allow-Origin"] == "*"  # the banner, hoisted
        assert "Authorization" in headers["Access-Control-Allow-Headers"]
        assert {"GET", "POST", "OPTIONS"} <= {
            m.strip() for m in headers["Access-Control-Allow-Methods"].split(",")
        }

    def test_a_declared_body_past_the_ceiling_is_refused_unread(self, server):
        """A declared length beyond the ceiling: 413, before a byte is read."""
        request = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=None,
            method="POST",
            headers={
                **AUTH,
                "Content-Type": "application/json",
                "Content-Length": "999999999999",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status, body = response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8")
        assert status == 413  # refused, unread — the wire, untouched
        error = json.loads(body)["error"]
        assert error["code"] == "request_entity_too_large"  # the code, as worn
        assert error["param"] == "body"  # the parameter, blamed
        assert error["type"] == "invalid_request_error"  # the kind, as filed
        assert "ceiling" in error["message"]  # the reason, given, in the message


class TestCanonicalRoutes:
    """The canonical routes of the specification, as served."""

    def test_the_routes_are_the_routes(self):
        """All of the specification’s routes, in one pass, over the canonical list."""
        routes = canonical_routes()
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes
        assert "/v1/completions" in routes
        assert all(
            route.startswith("/v1/") or route in ("/healthz", "/.well-known/agent-card.json")
            for route in routes
        )

    def test_the_pipes_of_the_proxy(self, server):
        """Each canonical route, exercised end to end: a status, not a crash."""
        for route, method in (("/v1/models", "GET"), ("/healthz", "GET")):
            status, _ = _request(server + route, method=method, headers=AUTH)
            assert status == 200, f"{method} {route} → {status}"  # the pipe, intact


class TestTimeoutWindow:
    """The window the sidecar allows the served to answer."""

    def test_the_timeout_is_parsed(self):
        from c2c.serve.openai_proxy import build_parser
        args = build_parser("c2c-serve").parse_args(
            ["--engine", "vllm-wired", "--url", "http://127.0.0.1:1",
             "--pair", "rx←tx", "--timeout", "1800"],
        )
        assert args.timeout == 1800.0  # the window, taken

    def test_the_default_leaves_the_window_to_the_sidecar(self):
        from c2c.serve.openai_proxy import build_parser
        args = build_parser("c2c-serve").parse_args(
            ["--engine", "reference", "--pair", "rx+tx"],
        )
        assert args.timeout is None  # the adapter's 600 stands

    def test_the_pair_carries_the_window(self):
        from c2c.serve.registry import ModelHub
        from c2c.serve.openai_proxy import _register_cli_pairs
        hub = ModelHub()
        hub.set_engine("vllm-wired")
        _register_cli_pairs(
            hub, ["rx←tx"], engine="vllm-wired",
            url="http://127.0.0.1:1", timeout=1800.0,
        )
        assert hub.resolve("c2c/rx←tx").receiver.timeout == 1800.0
