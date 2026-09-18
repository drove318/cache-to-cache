"""Tests for the MCP server (HL-2) and the A2A bridge (HL-3).

JSON-RPC over stdio, the protocol in messages; the agent card, on the
well-known path. Every test feeds lines, every test dispatches
messages — no network, just pipes, on stdin and stdout.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from c2c.serve.a2a import CARD_ROUTE, build_agent_card
from c2c.serve.mcp import (INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST,
                           MCPServer, METHOD_NOT_FOUND, PARSE_ERROR,
                           PROTOCOL_VERSION, TOOLS)


def _server():
    """A server with in-memory pipes, and a hub it can trust — fused, live."""
    from c2c.config import ServeConfig
    from c2c.serve.registry import ModelHub
    hub = ModelHub(ServeConfig())
    hub.set_engine("reference")
    hub.register_model("receiver-mini")
    hub.register_model("sharer-mini")
    recv = hub._resolve_side("receiver-mini")
    shar = hub._resolve_side("sharer-mini")
    from c2c.fuser import Fuser
    g_r, g_s = recv.spec().geometry, shar.spec().geometry
    fuser = Fuser(g_r, g_s, list(range(min(g_r.layers, g_s.layers))))
    hub.register_pair(receiver="receiver-mini", sharer="sharer-mini", fuser=fuser)
    reader = io.StringIO()
    writer = io.StringIO()
    return MCPServer(reader=reader, writer=writer, hub=hub), reader, writer


def _ask_server(server, writer, *messages):
    """Feed the lines, read the answers, line by line."""
    for message in messages:
        server.dispatch(message)
    writer.seek(0)
    return [json.loads(line) for line in writer if line.strip()]


INITIALISE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18",
                     "clientInfo": {"name": "pytest", "version": "1"}}}


class TestTheProtocol:
    """The handshake, the capabilities, the tool catalogue."""

    def test_initialize_is_the_first_command(self):
        server, _, writer = _server()
        replies = _ask_server(server, writer, INITIALISE)
        assert len(replies) == 1                                    # one request, one answer
        reply = replies[0]
        assert reply["jsonrpc"] == "2.0"                             # the version, echoed
        assert reply["id"] == 1                                      # the id, returned
        result = reply["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION         # the protocol, agreed
        assert result["serverInfo"]["name"]                          # the server, named
        assert "tools" in result["capabilities"]                     # tools, on offer

    def test_tools_are_listed(self):
        """tools/list — the catalogue, complete."""
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listing = replies[0]["result"]["tools"]
        names = {tool["name"] for tool in listing}
        assert names == {"c2c_register_pair", "c2c_fuse", "c2c_ask"}     # the three, exactly
        for tool in listing:
            assert tool["description"]                                     # each, described
            assert "inputSchema" in tool                                    # each, schematised

    def test_the_tools_of_the_wire(self):
        """tools/call — the fuse tool, in action, on the wire."""
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "c2c_fuse",
                                     "arguments": {"model": "c2c/receiver-mini←sharer-mini",
                                                 "prompt": "what is two plus two"}}})
        result = replies[0]["result"]
        assert result["isError"] is False                          # the call, well
        payload = json.loads(result["content"][0]["text"])          # the text, parsed
        assert {"layers", "tokens"} <= set(payload)                # the report, in full
        assert payload["tokens"] >= 1                               # the rows, counted

    def test_ask_makes_a_question(self):
        """tools/call — the ask tool, and the reply."""
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "c2c_ask",
                                     "arguments": {"model": "c2c/receiver-mini←sharer-mini",
                                                 "prompt": "hello",
                                                 "max_tokens": 5}}})
        result = replies[0]["result"]
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert "answer" in payload                                  # the answer, returned
        assert "usage" in payload                                    # the tokens, counted

    def test_ping_responds_pong(self):
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 5, "method": "ping"})
        assert replies[0]["result"] == {"ok": True} or "result" in replies[0]

    def test_notifications_are_ignored_silently(self):
        """A message with no id: no answer, as the protocol says."""
        server, _, writer = _server()
        server.dispatch({"jsonrpc": "2.0", "method": "notifications/cancelled"})
        writer.seek(0)
        assert writer.read() == ""                                  # silence, golden

    def test_unknown_method_is_method_not_found(self):
        """A method, unknown: the code of the not-found, precise."""
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 6, "method": "tools/spin"})
        error = replies[0]["error"]
        assert error["code"] == METHOD_NOT_FOUND                     # −32601, the standard
        assert "tools/spin" in error["message"]                       # the name, in the note

    def test_a_broken_line_is_a_parse_error(self):
        """serve_forever, on a malformed line: the code of the parse."""
        reader = io.StringIO("{ this is not json }\n")
        writer = io.StringIO()
        server = MCPServer(reader=reader, writer=writer)
        server.serve_forever()                                       # read till EOF
        writer.seek(0)
        lines = [json.loads(line) for line in writer if line.strip()]
        assert len(lines) == 1                                        # one error, once
        assert lines[0]["error"]["code"] == PARSE_ERROR               # −32700, the standard
        assert lines[0]["id"] is None                                  # id, unknown, so null

    def test_bad_params_are_invalid_params(self):
        """tools/call, with the arguments in the wrong shape."""
        server, _, writer = _server()
        replies = _ask_server(server, writer,
                           {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                            "params": {"name": "c2c_ask", "arguments": "not-an-object"}})
        assert replies[0]["error"]["code"] == INVALID_PARAMS          # −32602, reported

    def test_errors_are_errors(self):
        """The error codes of the module, the numbers of the specification."""
        assert (PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND,
               INVALID_PARAMS, INTERNAL_ERROR) == (-32700, -32600, -32601,
                                                  -32602, -32603)   # the table, complete

    def test_tools_are_described(self):
        """The catalogue entries, complete with descriptions."""
        assert len(TOOLS) == 3
        for tool in TOOLS:
            assert tool["name"].startswith("c2c_")                    # the prefix, shared
            assert "description" in tool and "inputSchema" in tool  # documented, all


class TestAgentCard:
    """HL-3: the agent card, the well-known path."""

    def test_the_card_is_on_the_well_known_path(self):
        assert CARD_ROUTE.startswith("/") and "well-known" in CARD_ROUTE   # the path, standard
        assert CARD_ROUTE.endswith(".json")

    def test_the_card_declares_the_agent(self):
        card = build_agent_card()
        assert card["name"]                                              # a name, it has
        assert card["description"]                                        # a purpose, stated
        assert card["version"]                                             # a version, stamped
        assert card["capabilities"]["streaming"] is True                   # streaming, able
        assert isinstance(card.get("default_input_modes"), list)   # snake, as A2A does
        skills = card.get("skills", [])
        assert skills and all({"name", "description"} <= set(s) for s in skills)
        assert any("c2c" in (s.get("id", "") + s.get("name", "")).lower()
                or "cache" in s.get("id", "").lower() for s in skills)   # the skills, C2C

    def test_the_card_is_machine_readable(self):
        """Print it, to see it: the card, valid JSON."""
        card = build_agent_card()
        text = json.dumps(card)
        assert json.loads(text) == card                                  # round-trips, sound


def _http_rpc(port: int, message: dict, *, key: str | None = None):
    """One JSON-RPC message, POSTed to the server on the port."""
    body = json.dumps(message).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/", data=body,
                                    headers={"Content-Type": "application/json"})
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class TestTheHTTPTransport:
    """The same server, on a port: the tools of the trade, over HTTP."""

    @pytest.fixture()
    def port(self):
        from c2c.serve.mcp import (MCPHTTPServer, MCPRequestHandler)
        core, _reader, _writer = _server()
        saved = (MCPRequestHandler.server_core, MCPRequestHandler.api_key,
                MCPRequestHandler.verbose)
        MCPRequestHandler.server_core = core
        MCPRequestHandler.api_key = None
        MCPRequestHandler.verbose = False
        httpd = MCPHTTPServer(("127.0.0.1", 0), MCPRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_address[1]
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        (MCPRequestHandler.server_core, MCPRequestHandler.api_key,
         MCPRequestHandler.verbose) = saved

    def test_the_health_of_the_second_transport(self, port):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                    timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["status"] == "ok"                                   # the pulse, taken
        assert body["service"] == "c2c-mcp"                             # the name, on the tin
        assert {t["name"] for t in TOOLS} <= set(body["tools"])         # the tools, advertised

    def test_the_handshake_and_a_call_over_the_wire(self, port):
        status, reply = _http_rpc(port, INITIALISE)
        assert status == 200 and reply["result"]["protocolVersion"] == PROTOCOL_VERSION
        status, called = _http_rpc(port, {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "c2c_ask",
                    "arguments": {"model": "c2c/receiver-mini←sharer-mini",
                                 "prompt": "what is two plus two",
                                 "max_tokens": 6}}})
        assert status == 200 and called["id"] == 7                     # the id, returned
        result = called["result"]
        assert result["isError"] is False                              # the call, well
        payload = json.loads(result["content"][0]["text"])             # the text, parsed
        assert isinstance(payload.get("answer"), str) and payload["answer"]   # the pair, answered
        assert payload.get("used_cache") is True                       # the cache, declared

    def test_an_unknown_method_is_an_error_object(self, port):
        _status, reply = _http_rpc(port, {"jsonrpc": "2.0", "id": 3,
                                         "method": "tools/frobnicate"})
        assert reply["error"]["code"] == METHOD_NOT_FOUND               # the code, standard

    def test_the_key_at_the_http_door(self):
        """With a key on the house, the unkeyed are turned away, the keyed admitted."""
        from c2c.serve.mcp import MCPHTTPServer, MCPRequestHandler
        core, _reader, _writer = _server()
        saved = (MCPRequestHandler.server_core, MCPRequestHandler.api_key)
        MCPRequestHandler.server_core = core
        MCPRequestHandler.api_key = "skeleton-key"
        httpd = MCPHTTPServer(("127.0.0.1", 0), MCPRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            p = httpd.server_address[1]
            status, denied = _http_rpc(p, INITIALISE)
            assert status == 401 and "error" in denied                 # refused, the unkeyed
            status, ok = _http_rpc(p, INITIALISE, key="skeleton-key")
            assert status == 200 and "result" in ok                     # admitted, the keyed
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            (MCPRequestHandler.server_core, MCPRequestHandler.api_key) = saved
