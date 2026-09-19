"""``c2c-mcp`` — the Model Context Protocol server (spec HL-2).

A line-delimited JSON-RPC 2.0 server on stdio, speaking the MCP wire
that MCP-native harnesses (Claude Code, Codex CLI, Oh My Pi, any MCP
client) speak. It exposes three tools so that a harness can declare
Sharer/Receiver pairs and request fused answers inside its own agent
loop — **the harness still owns the loop** (HL-4):

``c2c_register_pair``
    Declare a collaboration: ``receiver`` and ``sharer`` model ids,
    optionally with a trained fuser checkpoint (``weights_path``) and an
    engine (``engine``). Returns the canonical pair id.

``c2c_fuse``
    Run one fusion of the pair's caches over a prompt and report the
    FusionReport (gates, ranks, tokens) — the instrument panel of the
    collaboration, no answer generated.

``c2c_ask``
    Ask the pair a question through the full pipeline: resolve, capture,
    fuse, generate. Returns the answer text.

Protocol version advertised: ``2025-06-18``. Unknown notifications are
ignored as the specification requires; unknown requests answer with the
standard JSON-RPC error objects. Run it as ``c2c-mcp`` on stdio, or
``c2c-mcp --transport http`` to serve the tools over a port.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn

from .. import __version__
from .registry import default_hub

__all__ = [
    "MCPServer",
    "main",
    "build_parser",
    "MCPRequestHandler",
    "MCPHTTPServer",
    "PROTOCOL_VERSION",
    "TOOLS",
]

PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _tool(name: str, description: str, schema: dict) -> dict:
    return {"name": name, "description": description, "inputSchema": schema}


TOOLS = [
    _tool(
        "c2c_register_pair",
        "Declare a Sharer/Receiver collaboration for cache-to-cache answers.",
        {
            "type": "object",
            "properties": {
                "receiver": {"type": "string", "description": "receiver model id"},
                "sharer": {"type": "string", "description": "sharer model id"},
                "engine": {
                    "type": "string",
                    "description": "engine adapter for auto-built models",
                    "default": "reference",
                },
                "weights_path": {
                    "type": "string",
                    "description": "optional trained fuser checkpoint (.pt)",
                },
            },
            "required": ["receiver", "sharer"],
        },
    ),
    _tool(
        "c2c_fuse",
        "Fuse the pair's caches over a prompt; report gates and ranks.",
        {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "the virtual pair id, e.g. c2c/a←b"},
                "prompt": {"type": "string", "description": "the prompt to fuse on"},
            },
            "required": ["model", "prompt"],
        },
    ),
    _tool(
        "c2c_ask",
        "Ask the pair a question through the full cache-to-cache pipeline.",
        {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "the virtual model id, e.g. c2c/a←b"},
                "prompt": {"type": "string", "description": "the question"},
                "max_tokens": {"type": "integer", "minimum": 1, "default": 64},
                "temperature": {"type": "number", "minimum": 0.0, "default": 0.0},
                "c2c": {
                    "type": "object",
                    "description": "the per-query c2c options, same shape as the served "
                    "wires c2c body: {gate: block, block_list: [sharer ids]}",
                },
            },
            "required": ["model", "prompt"],
        },
    ),
]


class MCPServer:
    """The JSON-RPC 2.0 conversation, over stdio, one line per message.

    Parameters
    ----------
    reader, writer:
        Text streams; default ``sys.stdin`` / ``sys.stdout``. The console
        scripts swap them for the tests; the diagnostics go to stderr.
    hub:
        The model hub consulted by the tools; the shared default otherwise.
    """

    def __init__(self, *, reader=None, writer=None, hub=None):
        self.reader = reader if reader is not None else sys.stdin
        self.writer = writer if writer is not None else sys.stdout
        self.hub = hub if hub is not None else default_hub
        self.running = False
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "initialize": self._on_initialize,
            "ping": self._on_ping,
            "tools/list": self._on_tools_list,
            "tools/call": self._on_tools_call,
        }

    # -- the transport ──────────────────────────────────────────────────────
    def write(self, message: dict) -> None:
        self.writer.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.writer.flush()

    def log(self, message: str) -> None:
        sys.stderr.write(f"[c2c-mcp] {message}\n")
        sys.stderr.flush()

    def serve_forever(self) -> None:
        """Read line by line; answer request by request; ignore noise."""
        self.running = True
        while self.running:
            line = self.reader.readline()
            if not line:
                break  # EOF: the client hung up
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self.write(self._error(None, PARSE_ERROR, f"parse error: {exc.msg}"))
                continue
            try:
                self.dispatch(message)
            except Exception as exc:  # a stray line is noise, not death
                self.log(f"internal error answering {line[:60]!r}: {exc.__class__.__name__}")
                self.write(self._error(None, INTERNAL_ERROR, "the server failed to answer"))

    def respond(self, message: dict) -> dict | None:
        """One JSON-RPC message in, the answer (or none) out, for the HTTP transport.

        The reply is captured thread-locally: the core is shared, the
        capture is not. No attribute of the server is ever swapped here,
        so no two requests can read one another's answers.
        """
        box: dict[str, dict] = {}
        self._route(message, lambda reply: box.__setitem__("reply", reply))
        return box.get("reply")

    def dispatch(self, message: dict) -> None:
        """The stdio way: one message in, the answer written out."""
        self._route(message, self.write)

    def _route(self, message: dict, emit) -> None:
        """The conversation itself, with the answer handed to ``emit``."""
        if not isinstance(message, dict):
            emit(
                self._error(
                    None,
                    INVALID_REQUEST,
                    "a JSON-RPC message must be an object, as the wire defines",
                )
            )
            return
        method = message.get("method")
        identifier = message.get("id")
        is_notification = identifier is None and method is not None
        if not isinstance(method, str):
            if not is_notification:
                emit(
                    self._error(
                        identifier, INVALID_REQUEST, "a JSON-RPC request must carry a method"
                    )
                )
            return
        if is_notification:  # notifications: acknowledged
            return  # ignored, as the spec says
        handler = self._handlers.get(method)
        if handler is None:
            emit(self._error(identifier, METHOD_NOT_FOUND, f"method {method!r} is not known"))
            return
        try:
            result = handler(message.get("params") or {})
        except _BadParams as exc:
            emit(self._error(identifier, INVALID_PARAMS, str(exc)))
            return
        except Exception as exc:  # report, do not crash
            self.log(f"handler {method} raised {exc.__class__.__name__}: {exc}")
            emit(
                self._error(
                    identifier,
                    INTERNAL_ERROR,
                    f"internal error: {exc.__class__.__name__} (see the server log)",
                )
            )
            return
        emit({"jsonrpc": "2.0", "id": identifier, "result": result})

    # -- the lifecycle handlers ─────────────────────────────────────────────
    def _on_initialize(self, params: dict) -> dict:
        self.log(f"client protocol {params.get('protocolVersion', '?')}")
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "c2c-mcp",
                "version": __version__,
                "instructions": (
                    "Cache-to-Cache: fuse a Sharer's KV-cache into a "
                    "Receiver's. Register a pair, ask it, inspect it."
                ),
            },
        }

    def _on_ping(self, params: dict) -> dict:
        return {}

    def _on_tools_list(self, params: dict) -> dict:
        return {"tools": TOOLS}

    def _on_tools_call(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _BadParams("arguments must be an object")
        handler = getattr(self, f"_tool_{name}", None) if isinstance(name, str) else None
        if handler is None:
            raise _BadParams(f"unknown tool {name!r}")
        try:
            payload = handler(arguments)
        except _BadParams:
            raise
        except Exception as exc:  # tool errors are results
            self.log(f"tool {name} raised {exc.__class__.__name__}: {exc}")
            return {
                "content": [
                    {
                        "type": "text",
                        # the message, delivered to the client — the tools raise
                        # remedies; the client applies them, so let the client see
                        "text": f"tool {name} failed: {exc.__class__.__name__}: {exc}",
                    }
                ],
                "isError": True,
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": payload
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False),
                }
            ],
            "isError": False,
        }

    # -- the three tools ────────────────────────────────────────────────────
    def _tool_c2c_register_pair(self, args: dict) -> dict:
        receiver = _require(args, "receiver")
        sharer = _require(args, "sharer")
        engine = args.get("engine")
        if engine:
            self.hub.set_engine(str(engine))
        weights_path = args.get("weights_path")
        fuser = None
        if weights_path:
            import os

            from ..zoo.publish import DEFAULT_ZOO_ROOT
            from .openai_proxy import _load_fuser_state

            root = os.path.realpath(
                os.path.expanduser(os.environ.get("C2C_ZOO_ROOT") or DEFAULT_ZOO_ROOT)
            )
            path = os.path.realpath(os.path.expanduser(str(weights_path)))
            if not path.startswith(root + os.sep) or not path.endswith(".pt"):
                raise _BadParams(
                    f"weights_path must name a .pt inside the zoo root {root!r}; "
                    "publish it first (c2c zoo publish), or move the zoo with C2C_ZOO_ROOT"
                )
            fuser = _load_fuser_state(path)
        pair = self.hub.register_pair(receiver=receiver, sharer=sharer, fuser=fuser)
        return {"ok": True, "id": f"c2c/{pair.id}", "fused": pair.fused}

    def _tool_c2c_fuse(self, args: dict) -> dict:
        model = _require(args, "model")
        prompt = _require(args, "prompt")
        target = self.hub.resolve(model)
        if target is None:
            raise _BadParams(f"model {model!r} is not registered and cannot be built")
        if target.relay or target.fuser is None:
            raise _BadParams(f"model {model!r} is a relay pair; register a fuser first")
        receiver, sharer, fuser = target.receiver, target.sharer, target.fuser
        if not hasattr(receiver, "capture") or not hasattr(sharer, "capture"):
            msg = (
                f"model {model!r} fuses inside its server (the wired engine keeps the cache "
                "to itself): the preview belongs to the log — ask the pair and read the answer"
            )
            raise _BadParams(msg)
        r_ids = list(receiver.encode(prompt))
        s_ids = list(sharer.encode(prompt))
        cache_r = receiver.capture(r_ids)
        cache_s = sharer.capture(s_ids)
        token_mapping = None
        if r_ids != s_ids:  # equal length, differing content, is the aligners work
            from ..align.tokens import TokenAligner

            token_mapping = TokenAligner(receiver, sharer).select_rows(r_ids)
        fused = fuser(cache_r, cache_s, token_mapping=token_mapping)
        report = fuser.report(num_tokens=len(r_ids)) if hasattr(fuser, "report") else None
        return {
            "layers": len(fused),
            "tokens": fused.num_tokens,
            "gate_values": list(report.gate_values) if report else [],
            "gate_open_ratio": report.gate_open_ratio if report else None,
            "effective_rank": report.effective_rank if report else {},
        }

    def _tool_c2c_ask(self, args: dict) -> dict:
        model = _require(args, "model")
        prompt = _require(args, "prompt")
        declared_tokens = args.get("max_tokens")
        declared_temperature = args.get("temperature")
        try:
            max_tokens = 64 if declared_tokens is None else int(declared_tokens)
            temperature = 0.0 if declared_temperature is None else float(declared_temperature)
        except (TypeError, ValueError) as exc:
            raise _BadParams(f"the numeric arguments do not parse: {exc}") from exc
        if max_tokens < 1:
            raise _BadParams(
                f"max_tokens must be at least 1 (the schema's bound), got {max_tokens}"
            )
        if temperature < 0.0:
            raise _BadParams(
                f"temperature must not be below the schema's bound (0.0), got {temperature}"
            )
        from .openai_proxy import ChatPipeline

        pipeline = ChatPipeline(hub=self.hub)
        c2c_options = args.get("c2c") if isinstance(args.get("c2c"), dict) else None
        result = pipeline.complete(
            model=model,
            prompt_text=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            tools=None,
            stop=None,
            c2c_options=c2c_options,
        )
        out = {
            "answer": result["answer"],
            "used_cache": result["used_cache"],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
            },
        }
        if result.get("gate") == "block":
            out["gate"] = "block"  # the refusal, declared
        return out

    # -- JSON-RPC error objects, per the canonical table ───────────────────
    @staticmethod
    def _error(identifier, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


class _BadParams(ValueError):  # noqa: N818 — raised and caught in this module alone
    """Raised by the tools when the arguments do not make sense."""


def _require(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _BadParams(f"missing or empty string argument {key!r}")
    return value.strip()


def build_parser(prog: str = "c2c-mcp") -> argparse.ArgumentParser:
    """The parser of the server: the transport, the port, the key."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="MCP server of the cache-to-cache house: the tools of the "
        "trade over JSON-RPC (spec HL-2).",
        epilog="the tools are the verbs; the caches are the nouns; the loop stays the client's.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        help="engine adapter for auto-built models (default 'reference')",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="the wire beneath: stdio (the default) or http",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="http only: interface to bind (default the loopback)"
    )
    parser.add_argument(
        "--port", type=int, default=8789, help="http only: port to listen on (default 8789)"
    )
    parser.add_argument(
        "--api-key", default=None, help="http only: when set, require 'Authorization: Bearer <key>'"
    )
    parser.add_argument(
        "--certfile",
        default=None,
        help="http only: PEM certificate; with --keyfile, TLS on the wire",
    )
    parser.add_argument("--keyfile", default=None, help="http only: PEM key for --certfile")
    parser.add_argument("--verbose", action="store_true", help="report every request to stderr")
    return parser


class MCPRequestHandler(BaseHTTPRequestHandler):
    """JSON-RPC over HTTP: one request, one answer, the same core as stdio."""

    server_version = f"c2c-mcp/{__version__}"
    protocol_version = "HTTP/1.1"

    server_core: MCPServer | None = None  # bound by main()
    api_key: str | None = None
    verbose = False
    timeout = 60  # a stalled body read will raise: 60s, then 408, then silence

    def log_message(self, fmt: str, *args) -> None:
        if self.verbose:
            message = fmt % args if args else fmt
            sys.stderr.write(f"[c2c-mcp] {message}\n")

    def log_error(self, fmt: str, *args) -> None:
        self.log_message("error: " + fmt, *args)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def _authorized(self) -> bool:
        if not self.api_key:
            return True  # the door, open by design
        header = self.headers.get("Authorization", "") or ""
        scheme, _, token = header.partition(" ")
        from .openai_proxy import constant_time_equals

        return scheme.lower() == "bearer" and constant_time_equals(token, self.api_key)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/healthz", "/"):
            self._json(
                {
                    "status": "ok",
                    "service": "c2c-mcp",
                    "version": __version__,
                    "protocol": PROTOCOL_VERSION,
                    "tools": [tool["name"] for tool in TOOLS],
                }
            )
            return
        self._json({"error": {"message": f"no such path: {path!r}"}}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": "Incorrect API key provided."},
                },
                HTTPStatus.UNAUTHORIZED,
            )
            return
        from .openai_proxy import MAX_BODY_BYTES

        declared = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(declared)
        except ValueError:
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": INVALID_REQUEST,
                        "message": f"Content-Length {declared!r} is not a number",
                    },
                },
                HTTPStatus.BAD_REQUEST,
            )
            return
        if length > MAX_BODY_BYTES:
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": INVALID_REQUEST,
                        "message": (
                            f"the body declares {length} bytes; the ceiling is {MAX_BODY_BYTES}"
                        ),
                    },
                },
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            message = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": PARSE_ERROR, "message": "parse error"},
                },
                HTTPStatus.BAD_REQUEST,  # the body, corrupt; the status, its neighbour's kin
            )
            return
        if not isinstance(message, dict):
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": INVALID_REQUEST, "message": "a JSON object is required"},
                }
            )
            return
        core = self.server_core
        if core is None:
            # the distinction the transport must keep: a notification earns
            # silence; an unbound core is a server's misfortune, and says so
            if "id" not in message:
                self.send_response(HTTPStatus.NO_CONTENT)  # in silence, as per the spec
                self.end_headers()
                return
            self._json(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": "the server core is not bound (main binds it)",
                    },
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        reply = core.respond(message)
        if reply is None:  # the notification, acknowledged
            self.send_response(HTTPStatus.NO_CONTENT)  # in silence, as per the spec
            self.end_headers()
            return
        self._json(reply)


class MCPHTTPServer(ThreadingMixIn, TCPServer):
    """The stdio server's twin, on a port."""

    daemon_threads = True
    allow_reuse_address = True


def main(argv: Sequence[str] | None = None, *, reader=None, writer=None, hub=None) -> int:
    """Console-script entry point: stdio by default, a port on request."""
    args = build_parser("c2c-mcp").parse_args(argv)
    if args.engine:
        default_hub.set_engine(args.engine)
    server = MCPServer(reader=reader, writer=writer, hub=hub)
    if args.transport == "stdio":
        server.serve_forever()
        return 0
    import os

    from ..utils.console import banner, error_hint
    from .openai_proxy import _preflight_tls, is_loopback

    api_key = args.api_key if args.api_key is not None else os.environ.get("C2C_API_KEY")
    if not api_key and not is_loopback(args.host):
        print(
            error_hint(
                "the HTTP transport, off the loopback, must have a key: "
                "--api-key KEY, or C2C_API_KEY in the environment",
                hint="on the loopback the pipe itself is the trust; off it, the key is",
            ),
            file=sys.stderr,
        )
        return 2
    from ..config import ServeConfig

    cfg = ServeConfig(
        host=args.host,
        port=args.port,
        certfile=args.certfile,
        keyfile=args.keyfile,
        api_key=api_key,
    )
    _preflight_tls(cfg, "c2c-mcp")  # speaks before the bind: half a TLS pair is a user error

    MCPRequestHandler.server_core = server
    MCPRequestHandler.api_key = api_key
    try:
        httpd = MCPHTTPServer((cfg.host, cfg.port), MCPRequestHandler)
        if cfg.certfile and cfg.keyfile:
            from .openai_proxy import _make_ssl_context

            httpd.socket = _make_ssl_context(cfg.certfile, cfg.keyfile).wrap_socket(
                httpd.socket, server_side=True
            )
    except OSError as exc:
        sys.stderr.write(f"c2c-mcp: cannot bind {cfg.host}:{cfg.port}: {exc}\n")
        return 1
    sys.stderr.write(banner("mcp", __version__) + "\n")
    scheme = "https" if (cfg.certfile and cfg.keyfile) else "http"
    sys.stderr.write(
        f"  listening on {scheme}://{cfg.host}:{cfg.port} "
        "(JSON-RPC over POST /); the tools of the trade\n"
    )
    sys.stderr.write(
        "  the tools are the verbs; the caches are the nouns; the loop stays the client's.\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
