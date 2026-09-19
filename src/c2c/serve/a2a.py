"""Agent-card advertisement (HL-3): let cache-capable peers be discovered.

An A2A agent card (Google's Agent2Agent protocol) advertises what an
agent speaks and where it speaks it. A C2C-enabled server advertises,
**beside the text endpoints, the cache endpoints**: peers of a multi-
harness agent society can then discover that ``c2c/qwen3-0.6b←…``
offers not just words but *meaning*, carried at the speed of tensors.

The card is served at ``GET /.well-known/agent-card.json`` — the
well-known location of the protocol — and the tasks ride POST to
``/tasks``, JSON-RPC of the methods ``message/send``, ``tasks/get``,
``tasks/list``, ``tasks/cancel``. A task is submitted, works, and ends
in an artifact — and, of the cache, in the state ``fused``:

    c2c-a2a --pair math-small:coder-small -e hf --port 8122

Keep the shape of the card conservative; the fields follow the A2A
specification's card of the interface.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn

from .. import __version__
from ..config import ServeConfig, load_config
from ..utils.console import banner, style
from .registry import default_hub

__all__ = ["build_agent_card", "CARD_ROUTE", "TASK_ROUTE", "STATES",
         "TaskStore", "A2AHandler", "A2AServer", "create_bridge",
         "serve_bridge", "build_parser", "main"]

CARD_ROUTE = "/.well-known/agent-card.json"

#: the route of the tasks, POST of JSON-RPC
TASK_ROUTE = "/tasks"
#: the states of a task: the protocol's, and the one the cache adds
STATES = ("submitted", "working", "completed", "failed", "canceled", "fused")


def build_agent_card(config: ServeConfig | None = None, hub=None) -> dict:
    """Compose the agent card of one C2C server and its hub.

    The card lists the text endpoints (OpenAI wire, at the canonical
    paths) and the cache endpoints (the same paths, virtual model ids of
    the ``c2c/`` namespace speak them). Discovery clients may match the
    skills by their name and description.
    """
    from .openai_proxy import ROUTE_CHAT_COMPLETIONS, ROUTE_COMPLETIONS, ROUTE_MODELS
    cfg = config or ServeConfig()
    hub = hub if hub is not None else default_hub
    scheme = "https" if (cfg.certfile and cfg.keyfile) else "http"
    base = f"{scheme}://{cfg.host}:{cfg.port}"
    models = []
    try:
        models = [mid for mid, _note, _ctx in hub.describe_models()]
    except Exception:                                        # cards must print, never raise
        pass
    return {
        "protocol_version": "0.3.0",
        "name": "c2c-serve",
        "description": ("Cache-to-Cache middleware: direct semantic communication "
                       "between LLMs through fused KV-Caches, served over the "
                       "OpenAI wire format."),
        "url": base,
        "provider": {
            "organization": "cache-to-cache",
            "url": "https://github.com/drove318/cache-to-cache",
        },
        "version": __version__,
        "documentation": "https://github.com/drove318/cache-to-cache/tree/main/docs",
        "capabilities": {
            "streaming": True,
            "push_notifications": False,
            "state_transition_history": False,
            "extensions": [
                {"uri": "https://c2c.dev/ext/cache", "required": False,
                 "description": "cache-to-cache fusion via virtual model ids"},
            ],
        },
        "default_input_modes": ["text/plain", "application/json"],
        "default_output_modes": ["text/plain", "application/json"],
        "supported_interfaces": [
            {"url": base + ROUTE_CHAT_COMPLETIONS, "protocol_binding": "REST",
             "protocol_version": "1.0"},
            {"url": base + ROUTE_COMPLETIONS, "protocol_binding": "REST",
             "protocol_version": "1.0"},
        ],
        "skills": [
            {
                "id": "cache-to-cache",
                "name": "Cache-to-Cache collaboration",
                "description": ("Fuse the KV-cache of a Sharer model into a Receiver "
                                "model; the pair is named by a virtual model id of "
                                "the form c2c/<receiver>←<sharer>."),
                "tags": ["llm", "cache", "fusion", "c2c"],
                "examples": ["tell me about the paper cache-to-cache"],
                "input_modes": ["text/plain"],
                "output_modes": ["text/plain"],
            },
            {
                "id": "model-index",
                "name": "List available models and pairs",
                "description": "GET " + ROUTE_MODELS + " returns the model index.",
                "tags": ["discovery"],
            },
        ],
        "security_schemes": {
            "bearer": {"type": "http", "scheme": "bearer",
                      "description": ("required when the server was started with "
                                      "--api-key; otherwise the door is open")},
        },
        "security": [{"bearer": []}],
        "models": models,
    }


def _now() -> str:
    """The hour, in the tongue of the wire: UTC, ISO 8601."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


class TaskStore:
    """The tasks of the bridge, in the memory of the process.

    The bridge forwards; it does not store. A restart forgets all tasks —
    which is the way of bridges, and the reason no secret may be trusted
    to one. The lock is for the eyes of the threads, one per request.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, text: str, *, context_id: str | None = None,
               owner: str = "") -> dict:
        task_id = "task-" + uuid.uuid4().hex[:12]
        task = {
            "id": task_id,
            "context_id": context_id or "ctx-" + uuid.uuid4().hex[:8],
            "owner": owner,
            "status": {"state": "submitted", "timestamp": _now()},
            "message": {"role": "user", "parts": [{"kind": "text", "text": text}]},
            "artifacts": [],
        }
        with self._lock:
            self._tasks[task_id] = task
        return dict(task)

    def set_state(self, task_id: str, state: str, *,
                 artifacts: list | None = None) -> dict | None:
        if state not in STATES:
            msg = f"unknown state {state!r}, the states are: {', '.join(STATES)}"
            raise ValueError(msg)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task["status"] = {"state": state, "timestamp": _now()}
            if artifacts is not None:
                task["artifacts"] = artifacts
            return dict(task)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task is not None else None

    def listed(self, *, context_id: str | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._tasks.values())
        if context_id is not None:
            rows = [t for t in rows if t["context_id"] == context_id]
        return [dict(t) for t in rows]


def _texts_of(message) -> str:
    """The parts of a message, of the text kind, joined.

    The protocol is generous: a bare string is a message too.
    """
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    parts = message.get("parts") or []
    bits = [str(p.get("text", "")) for p in parts
           if isinstance(p, dict) and p.get("kind", p.get("type")) == "text"]
    return "\n".join(b for b in bits if b)


class A2AHandler(BaseHTTPRequestHandler):
    """The handler of the bridge: the card on GET, the tasks on POST.

    The wire is JSON-RPC, as the protocol of the agents prescribes. The
    methods: ``message/send`` (a task, born and answered), ``tasks/get``,
    ``tasks/list``, ``tasks/cancel``. The answer of the task is the
    artifact; the artifact is the text of the receiver, with the cache of
    the sharer fused into its state before the first token.
    """

    server_version = f"c2c-a2a/{__version__}"
    protocol_version = "HTTP/1.1"

    pipeline = None                                    # bound by the factory
    config: ServeConfig | None = None
    store: TaskStore | None = None
    public_url: str | None = None

    # -- logging: stderr, never stdout (stdout may be the wire of a client) ─
    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args if args else fmt
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    def log_error(self, fmt: str, *args) -> None:
        self.log_message("error: " + fmt, *args)

    # -- the wire, small helpers, shared mind ───────────────────────────────
    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def _rpc_error(self, req_id, code: int, message: str) -> None:
        self._json({"jsonrpc": "2.0", "id": req_id,
                   "error": {"code": code, "message": message}})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    def _principal(self) -> str | None:
        """Who is asking: the bearer token of the house, the empty string of
        the open door, or None when the key is amiss.

        One key, one house, one parlour: the tasks of the bridge belong to
        the principal that submitted them; no other peer may read them.
        """
        cfg = self.config or ServeConfig()
        if not cfg.api_key:
            return ""                                              # the door, open by design
        header = self.headers.get("Authorization", "") or ""
        scheme, _, token = header.partition(" ")
        from .openai_proxy import constant_time_equals
        if scheme.lower() == "bearer" and constant_time_equals(token, cfg.api_key):
            return token
        return None

    # -- the verbs ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:                             # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == CARD_ROUTE:
            card = build_agent_card(self.config, self.pipeline.hub)
            if self.public_url:
                card["url"] = self.public_url
            card["name"] = "c2c-a2a"
            self._json(card)
            return
        if path == "/healthz":
            self._json({"status": "ok", "service": "c2c-a2a", "version": __version__,
                      "routes": [CARD_ROUTE, TASK_ROUTE, "/healthz"]})
            return
        self._json({"error": {"message": f"no such path: {path!r}"}},
                  HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:                            # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != TASK_ROUTE:
            self._json({"error": {"message": f"no such route: {path!r}"}},
                      HTTPStatus.NOT_FOUND)
            return
        principal = self._principal()
        if principal is None:
            self._json({"error": {"message": "Incorrect API key provided.",
                                "type": "authentication_error",
                                "code": "invalid_api_key"}},
                     HTTPStatus.UNAUTHORIZED)
            return
        try:
            body = json.loads(self._read_body().decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._rpc_error(None, -32700, "parse error: the body is not JSON")
            return
        if not isinstance(body, dict) or not isinstance(body.get("method"), str):
            req_id = body.get("id") if isinstance(body, dict) else None
            self._rpc_error(req_id, -32600, "invalid request: method is required")
            return
        method, req_id = body["method"], body.get("id")
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        try:
            self._dispatch(method, req_id, params, principal)
        except Exception as exc:                            # the wire must not drop
            self.log_error("dispatch: %r", exc)
            self._rpc_error(req_id, -32603,
                          f"internal error: {exc.__class__.__name__} (see the server log)")

    def _dispatch(self, method: str, req_id, params: dict, principal: str) -> None:
        store, pipeline = self.store, self.pipeline
        if method in ("message/send", "tasks/send"):
            self._send(pipeline, store, req_id, params, principal)
            return
        if method == "tasks/get":
            task = store.get(str(params.get("id", "")))
            if task is None or task.get("owner", "") != principal:
                self._rpc_error(req_id, -32001, "task not found")
                return
            self._json({"jsonrpc": "2.0", "id": req_id, "result": task})
            return
        if method == "tasks/list":
            context_id = params.get("context_id")
            if not context_id:
                self._rpc_error(req_id, -32602,
                              "tasks/list requires a context_id: the bridge lists "
                              "the tasks of one context, not of the whole house")
                return
            rows = [t for t in store.listed(context_id=str(context_id))
                    if t.get("owner", "") == principal]
            self._json({"jsonrpc": "2.0", "id": req_id,
                      "result": {"tasks": rows, "next_page_token": ""}})
            return
        if method == "tasks/cancel":
            wanted = store.get(str(params.get("id", "")))
            if wanted is None or wanted.get("owner", "") != principal:
                self._rpc_error(req_id, -32001, "task not found")
                return
            task = store.set_state(str(params.get("id", "")), "canceled")
            self._json({"jsonrpc": "2.0", "id": req_id, "result": task})
            return
        self._rpc_error(req_id, -32601, f"method not found: {method!r}")

    def _send(self, pipeline, store: TaskStore, req_id, params: dict,
               principal: str) -> None:
        """message/send: a task, born, worked, and answered, in one exchange."""
        text = _texts_of(params.get("message", params.get("text", "")))
        if not text.strip():
            self._rpc_error(req_id, -32602, "invalid params: no text in the message")
            return
        context_id = params.get("contextId") or params.get("context_id")
        task = store.submit(text, owner=principal,
                          context_id=str(context_id) if context_id else None)
        store.set_state(task["id"], "working")
        cfg = self.config or ServeConfig()
        try:
            result = pipeline.complete(
                model=str(params.get("model") or ""),
                prompt_text=text,
                max_new_tokens=int(params.get("max_new_tokens", cfg.max_new_tokens)),
                temperature=float(params.get("temperature", 0.0)),
                tools=None,
                stop=params.get("stop"))
        except Exception as exc:                            # the task, failed, said so
            self.log_error("task %s raised %r", task["id"], exc)
            failed = store.set_state(task["id"], "failed")
            self._json({"jsonrpc": "2.0", "id": req_id,
                      "result": {**(failed or task),
                               "error": f"{exc.__class__.__name__}: the task failed "
                                        "(see the server log)"}})
            return
        answer = str(result.get("answer", ""))
        state = "fused" if result.get("used_cache") else "completed"
        artifact = [{"artifact_id": f"artifact-{task['id']}-0", "name": "answer",
                   "parts": [{"kind": "text", "text": answer}]}]
        done = store.set_state(task["id"], state, artifacts=artifact)
        self._json({"jsonrpc": "2.0", "id": req_id, "result": done})


class A2AServer(ThreadingMixIn, TCPServer):
    """One server, two faces: the card, and the tasks."""

    daemon_threads = True
    allow_reuse_address = True


def create_bridge(config: ServeConfig | None = None, *, hub=None, pipeline=None):
    """Compose the bridge over the river of the wire."""
    from .openai_proxy import ChatPipeline
    cfg = config or ServeConfig()
    if pipeline is None:
        pipeline = ChatPipeline(hub=hub)
    A2AHandler.pipeline = pipeline
    A2AHandler.config = cfg
    A2AHandler.store = TaskStore()
    server_obj = A2AServer((cfg.host, cfg.port), A2AHandler)
    if cfg.certfile and cfg.keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cfg.certfile)
        context.load_private_key(cfg.keyfile)
        server_obj.socket = context.wrap_socket(server_obj.socket, server_side=True)
    return server_obj


def serve_bridge(config: ServeConfig | None = None, *, hub=None,
               public_url: str | None = None) -> None:
    """Serve the card and the tasks, until the keyboard interrupts."""
    A2AHandler.public_url = public_url
    server_obj = create_bridge(config, hub=hub)
    cfg = config or ServeConfig()
    scheme = "https" if (cfg.certfile and cfg.keyfile) else "http"
    sys.stderr.write(banner("a2a", __version__) + "\n")
    sys.stderr.write("  listening on "
                   + style(f"{scheme}://{cfg.host}:{cfg.port}", "bold") + "\n")
    sys.stderr.write(f"    * the card: {CARD_ROUTE}\n")
    sys.stderr.write(f"    * the tasks: {TASK_ROUTE} (JSON-RPC)\n")
    sys.stderr.write("  the card is public, the tasks are not; the loop stays the agents'.\n")
    try:
        server_obj.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server_obj.server_close()


def build_parser(prog: str = "c2c-a2a") -> argparse.ArgumentParser:
    """The parser of the bridge: the flags of the front, and the public url."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Agent-to-Agent bridge: the card on the well-known path, "
                    "the tasks over the wire, the cache between the models (spec HL-2).",
        epilog="the harness stays the master; C2C is the wire between models.")
    parser.add_argument("--host", default=None, help="interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port to listen on")
    parser.add_argument("--certfile", default=None, help="PEM certificate for HTTPS")
    parser.add_argument("--keyfile", default=None, help="PEM key for HTTPS")
    parser.add_argument("--api-key", default=None,
                       help="when set, require 'Authorization: Bearer <key>' at the tasks")
    parser.add_argument("-e", "--engine", default=None,
                       help="engine adapter for auto-built models (default 'reference')")
    parser.add_argument("--pair", action="append", metavar="RECEIVER←SHARER",
                       help="register a collaboration pair (repeatable)")
    parser.add_argument("--config", default=None, help="path to a config.json")
    parser.add_argument("--public-url", default=None,
                       help="the url the card advertises (default: the bind address)")
    parser.add_argument("--privacy", action="store_true",
                       help="seal the wire: digests on the logs, no plaintext egress (FR-17)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point of the bridge, as the console script runs it."""
    from .openai_proxy import _register_cli_pairs, is_loopback
    args = build_parser("c2c-a2a").parse_args(argv)
    base = load_config(args.config)
    serve_base = getattr(base, "serve", base)
    import os
    updates = {"host": args.host, "port": args.port,
               "certfile": args.certfile, "keyfile": args.keyfile,
               "api_key": args.api_key or os.environ.get("C2C_API_KEY"),
               "privacy": True if args.privacy else None}
    cfg = ServeConfig(**{**vars(serve_base),
                        **{k: v for k, v in updates.items() if v is not None}})
    if not cfg.api_key and not is_loopback(cfg.host):
        from ..utils.console import error_hint
        print(error_hint("the bridge, off the loopback, must have a key: --api-key KEY, "
                        "or C2C_API_KEY in the environment",
                        hint="tasks are not public; the key is what keeps them so"),
              file=sys.stderr)
        return 2
    if args.engine:
        default_hub.set_engine(args.engine)
    if args.pair:
        _register_cli_pairs(default_hub, args.pair, engine=args.engine)
    try:
        serve_bridge(cfg, public_url=args.public_url)
    except OSError as exc:
        sys.stderr.write(f"c2c-a2a: cannot bind {cfg.host}:{cfg.port}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
