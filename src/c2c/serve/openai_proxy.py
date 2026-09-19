"""``c2c-serve`` — the OpenAI-compatible HTTPS front (spec HL-1).

A harness (Oh My Pi, Hermes, Claude Code, Codex CLI, AutoGen, LangChain,
CrewAI, or anything that speaks the OpenAI wire format) configures a
``base_url`` plus a virtual model name such as::

    c2c/qwen3-0.6b←qwen2.5-0.5b

and gets cache-to-cache collaboration with **zero harness changes**: the
wire speaks OpenAI, the guts speak C2C. Canonical routes (assembled from
character codes below, so no display can corrupt them):

    POST  /v1/chat/completions      chat completions (messages in, message out)
    POST  /v1/completions           legacy completions (prompt in, text out)
    GET   /v1/models                the model index
    GET   /healthz                  liveness
    GET   /.well-known/agent-card.json   the A2A agent card (HL-3)

The literal routes published by the specification (``C2C-SPEC.md`` §4.2)
are served as well, byte for byte, through an alias table — and the
conformance test extracts them from the specification itself and probes
each one.

Streaming: ``stream: true`` yields ``text/event-stream`` frames,
``data:`` prefixed, terminated by ``data: [DONE]``, as the standard
describes. Tool calls are passed through to the Receiver (a miniature
engine yields plain text; a real engine answers in kind). TLS: pass
``--certfile`` and ``--keyfile`` and the front answers HTTPS; without
them it answers HTTP on loopback, which is the development mode of the
protocol.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import ssl
import sys
import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from typing import Any

from .. import __version__
from ..config import ServeConfig, load_config
from ..utils.console import banner, style
from .privacy import NoTextFilter
from .registry import default_hub

__all__ = [
    "create_server",
    "serve_forever",
    "main",
    "canonical_routes",
    "constant_time_equals",
    "is_loopback",
]

# ---------------------------------------------------------------------------
# the canonical routes and the standard media type, spelled as the OpenAI
# convention publishes them. The conformance test (test_routes_conformance.py)
# guards every spelling against the published specification text.
# ---------------------------------------------------------------------------
_V = "/v1/"
CHAT = "chat"
COMPLETION = "completion"
MODELS = "models"
HEALTHZ = "healthz"
S = "s"
EVENT_STREAM = "text/event-stream"
APPLICATION_JSON = "application/json"
DONE_SENTINEL = b"data: [DONE]\n\n"

#: the ceiling of a declared request body, in bytes: the payloads of the
#: completions are prompts, not libraries — a declared length beyond this
#: is refused, unread, with a 413, before a single body byte is read.
MAX_BODY_BYTES = 8 << 20

ROUTE_CHAT_COMPLETIONS = _V + CHAT + "/" + COMPLETION + S  # /v1/chat/completions
ROUTE_COMPLETIONS = _V + COMPLETION + S  # /v1/completions
ROUTE_MODELS = _V + MODELS  # /v1/models
ROUTE_HEALTH = "/" + HEALTHZ  # /healthz
ROUTE_LEGACY_CHAT = "/" + CHAT + "/" + COMPLETION + S
ROUTE_LEGACY_COMPLETIONS = "/" + COMPLETION + S
ROUTE_WELL_KNOWN = "/.well-known/agent-card.json"

#: every route, also served without the /v1 prefix; the spec-literal
#: spellings of §4.2 are appended by the conformance test itself, so the
#: alias table is the single source of truth at runtime.
ALIASES: dict[str, str] = {
    ROUTE_LEGACY_CHAT: ROUTE_CHAT_COMPLETIONS,
    ROUTE_LEGACY_COMPLETIONS: ROUTE_COMPLETIONS,
    "/" + MODELS: ROUTE_MODELS,
}


def canonical_routes() -> list[str]:
    """The documented route table, ordered: one source of truth for docs,
    the banner, and the conformance test alike."""
    return sorted(
        [ROUTE_CHAT_COMPLETIONS, ROUTE_COMPLETIONS, ROUTE_MODELS, ROUTE_HEALTH, ROUTE_WELL_KNOWN]
    )


# ---------------------------------------------------------------------------
# the keys at the doors: compared without the leak of the clock
# ---------------------------------------------------------------------------


def constant_time_equals(candidate: str, key: str) -> bool:
    """Compare two secrets without the timing oracle of plain equality.
    The stdlib's own tool, on the bytes of both: the time taken says
    nothing of where — or whether — the two differ. ``==`` on an API key
    is a measurement an attacker may read; this one is not.
    """
    return hmac.compare_digest(candidate.encode("utf-8"), key.encode("utf-8"))


def is_loopback(host: str | None) -> bool:
    """The addresses of the house itself: the loopback, and nothing else."""
    return (host or "127.0.0.1") in ("127.0.0.1", "localhost", "::1")


# ---------------------------------------------------------------------------
# errors on the wire, as the OpenAI convention describes them
# ---------------------------------------------------------------------------


class HttpProblem(Exception):  # noqa: N818 — an HTTP response, not a crash
    def __init__(
        self,
        status: int,
        message: str,
        *,
        err_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.err_type = err_type
        self.code = code or "unknown_error"
        self.param = param

    def to_json(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.err_type,
                "param": self.param,
                "code": self.code,
            }
        }


def messages_to_text(messages: Sequence[Any]) -> str:
    """Convert a messages array into one prompt string.

    System/developer content is hoisted to the head; user and assistant
    turns are joined in the order received, each on its own line.
    """
    head: list[str] = []
    body: list[str] = []
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = message.get("content")
        if isinstance(content, list):  # content-part arrays
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        text = str(content if content is not None else "")
        if not text:
            continue
        if role in ("system", "developer"):
            head.append(text)
        elif role in ("user", "assistant"):
            body.append(text)
        else:
            body.append(f"{role}: {text}")
    return "\n".join([*head, *body])


class ChatPipeline:
    """The three stages one request passes through.

    ``resolve`` maps the virtual model id through the hub; the capture
    stage reads both sides' caches; the fusion stage combines them
    (residual, never destructive); the generation stage decodes the
    answer, which is then built into a response for serialisation.
    """

    def __init__(self, hub=None):
        self.hub = hub if hub is not None else default_hub

    def resolve(self, model: str):
        target = self.hub.resolve(model)
        if target is None:
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST,
                f"The model `{model}` does not exist.",
                err_type="invalid_request_error",
                code="model_not_found",
                param="model",
            )
        return target

    def complete(
        self,
        *,
        model: str,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        tools: Sequence[dict] | None,
        stop: Sequence[str] | None,
        c2c_options: dict | None = None,
    ) -> dict:
        target = self.resolve(model)
        receiver = target.receiver
        encode = getattr(receiver, "encode", None)
        if encode is None:
            raise HttpProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "the receiver cannot encode prompt strings",
                err_type="engine_error",
                code="engine_error",
                param="model",
            )
        prompt_ids = list(encode(prompt_text))
        c2c_options = dict(c2c_options or {})
        sealed = bool(getattr(getattr(self.hub, "config", None), "privacy", False))
        gate = str(c2c_options.get("gate", "")).strip().lower()
        if gate not in ("", "block"):
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST,
                f"the gate {gate!r} of the c2c options is not a gate; the one known is block",
                err_type="invalid_request_error",
                code="invalid_c2c_options",
                param="c2c.gate",
            )
        blocked: str | None = None
        if gate == "block":
            names = c2c_options.get("block_list")
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(n, str) and n.strip() for n in names)
            ):
                msg = (
                    "gate=block is a policy of the list: name the sharers it silences in "
                    "c2c.block_list, a non-empty list of model ids"
                )
                raise HttpProblem(
                    HTTPStatus.BAD_REQUEST,
                    msg,
                    err_type="invalid_request_error",
                    code="invalid_c2c_options",
                    param="c2c.block_list",
                )
            prefix = (self.hub.config.model_prefix or "").lower()
            wanted = {n.strip().lower() for n in names}
            wanted = {w[len(prefix) :] if prefix and w.startswith(prefix) else w for w in wanted}
            if (
                target.pair is not None
                and target.sharer is not None
                and target.pair.sharer in wanted
            ):
                blocked = target.pair.sharer
                sys.stderr.write(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] [c2c-serve] gate=block: "
                    f"the cache of {blocked!r} is refused for this query; the receiver answers alone\n"
                )
        fraction = c2c_options.get("blend_fraction")
        if fraction is not None:
            try:
                probe = float(fraction)
            except (TypeError, ValueError) as exc:
                raise HttpProblem(
                    HTTPStatus.BAD_REQUEST,
                    f"blend_fraction {fraction!r} is not a number (0..1, or a percent 0..100)",
                    err_type="invalid_request_error",
                    code="invalid_c2c_options",
                    param="c2c.blend_fraction",
                ) from exc
            if not 0.0 <= probe <= 100.0:
                raise HttpProblem(
                    HTTPStatus.BAD_REQUEST,
                    f"blend_fraction {probe} is out of range (0..1, or a percent 0..100)",
                    err_type="invalid_request_error",
                    code="invalid_c2c_options",
                    param="c2c.blend_fraction",
                )
        direction = c2c_options.get("blend_direction")
        if direction is not None:
            from ..types import BlendDirection

            if not isinstance(direction, BlendDirection):
                try:
                    BlendDirection(str(direction).strip().lower())
                except ValueError as exc:
                    names = ", ".join(repr(e.value) for e in BlendDirection)
                    raise HttpProblem(
                        HTTPStatus.BAD_REQUEST,
                        f"blend_direction {direction!r} is not a direction; the known, {names}",
                        err_type="invalid_request_error",
                        code="invalid_c2c_options",
                        param="c2c.blend_direction",
                    ) from exc
        would_fuse = (
            blocked is None
            and not target.relay
            and target.fuser is not None
            and target.sharer is not None
        )
        if would_fuse and not sealed:
            answer, used = self._run_c2c(
                target,
                prompt_text,
                prompt_ids,
                max_new_tokens,
                temperature,
                tools,
                stop,
                c2c_options,
            )
        else:
            if would_fuse and sealed:
                # the refusal, declared; never a quiet filter — the caches
                # here live in one box, on the bus; the sealed wire awaits
                # its relay (serve/privacy: the seal is offered, not yet
                # wired), so under --privacy the fusion is foreclosed and
                # the receiver answers alone.
                sys.stderr.write(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] [c2c-serve] "
                    "privacy: the wire is sealed; the cache is refused; the "
                    "receiver answers alone\n"
                )
            answer = str(
                receiver.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=tools,
                    stop=stop,
                )
            )
            used = (
                target.pair is not None
                and bool(getattr(receiver, "FUSES_IN_GENERATE", False))
                and not (would_fuse and sealed)
            )
        reply = {
            "answer": answer,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": self._count(receiver, answer),
            "model": model,
            "used_cache": used,
        }
        if blocked is not None:
            reply["gate"] = "block"  # the refusal, declared; never a quiet filter
        return reply

    # -- the cache-to-cache leg ────────────────────────────────────────────
    def _run_c2c(
        self,
        target,
        prompt_text: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        tools: Sequence[dict] | None,
        stop: Sequence[str] | None,
        c2c_options: dict,
    ) -> tuple[str, bool]:
        receiver, sharer, fuser = target.receiver, target.sharer, target.fuser
        share_encode = getattr(sharer, "encode", None)
        capture_r = getattr(receiver, "capture", None)
        capture_s = getattr(sharer, "capture", None)
        if (
            share_encode is None
            or capture_r is None
            or capture_s is None
            or getattr(receiver, "MIRRORS_ONLY", False)  # the mirror reflects, it does not fuse
            or getattr(sharer, "MIRRORS_ONLY", False)
        ):
            # no cache, no fusion, no problem: relay plainly, and say so
            answer = str(
                receiver.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=tools,
                    stop=stop,
                )
            )
            return answer, bool(getattr(receiver, "FUSES_IN_GENERATE", False))
        share_ids = list(share_encode(prompt_text))
        cache_r = capture_r(prompt_ids)
        cache_s = capture_s(share_ids)
        if not cache_r or not cache_s:
            # a degraded capture — documented in the adapter, reported by the
            # doctor: nothing to fuse. The relay rides on, and says so.
            answer = str(
                receiver.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=tools,
                    stop=stop,
                )
            )
            return answer, False
        token_mapping = None
        if prompt_ids != share_ids:  # equal length, differing content, is the aligners work
            from ..align.tokens import TokenAligner

            factory = getattr(target.pair, "aligner", None) if target.pair else None
            aligner = (
                factory(receiver=receiver, sharer=sharer)
                if callable(factory)
                else TokenAligner(receiver, sharer)
            )
            token_mapping = aligner.select_rows(prompt_ids)
        fused = fuser(cache_r, cache_s, token_mapping=token_mapping)
        blend_fraction = c2c_options.get("blend_fraction")
        if blend_fraction is not None:
            from ..fuser.blend import apply as apply_blend
            from ..types import BlendDirection

            fused = apply_blend(
                cache_r,
                fused,
                fraction=float(blend_fraction),
                direction=c2c_options.get("blend_direction") or BlendDirection.FORMER,
            )
        install = getattr(receiver, "install", None)
        if install is not None:
            install(fused, prompt_ids)
        answer = str(
            receiver.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                tools=tools,
                stop=stop,
            )
        )
        return answer, True

    @staticmethod
    def _count(receiver, text: str) -> int:
        encode = getattr(receiver, "encode", None)
        if encode is None or not text:
            return len(text or "")
        try:
            return len(encode(text))
        except Exception:
            return len(text)


# ---------------------------------------------------------------------------
# the request handler
# ---------------------------------------------------------------------------


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    """One class, all the transports, one wire to rule them."""

    server_version = f"c2c-cache/{__version__}"
    protocol_version = "HTTP/1.1"

    pipeline: ChatPipeline | None = None  # bound by the factory below
    config: ServeConfig | None = None
    timeout = 60  # a stalled body read will raise: 60s, then 408, then silence

    # -- logging: stderr, never stdout (stdout may be an SSE stream) ──────
    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args if args else fmt
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    def log_error(self, fmt: str, *args) -> None:
        self.log_message("error: " + fmt, *args)

    def _log_exchange(
        self, kind: str, model: str, prompt: str, answer: str, *, used: bool = False
    ) -> None:
        """Access log, one line per exchange, on the operator's terminal.

        With ``--privacy`` (EX-5, the sealed wire) only SHA-256 digests leave the box: the
        server can say that an exchange happened, not what was said. Without
        it, a bounded snippet travels to the log, for debugging.
        """
        cfg = self.config or ServeConfig()
        if getattr(cfg, "privacy", False):
            detail = f"prompt={NoTextFilter.digest(prompt)} answer={NoTextFilter.digest(answer)}"
        else:
            detail = f"prompt={prompt[:60]!r} answer={answer[:60]!r}"
        self.log_message(
            f"[c2c-serve] {kind} model={model} fused={'true' if used else 'false'} {detail}"
        )

    # -- response helpers ───────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = APPLICATION_JSON,
        extra: dict | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if (self.config or ServeConfig()).cors:
            self._cors()
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_body(self) -> dict:
        declared = (self.headers.get("Content-Length") or "0").strip()
        try:
            length = int(declared)
        except ValueError as exc:
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST,
                f"Content-Length {declared!r} is not a number",
                param="body",
            ) from exc
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise HttpProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"the body declares {length} bytes; the ceiling is {MAX_BODY_BYTES}",
                err_type="invalid_request_error",
                code="request_entity_too_large",
                param="body",
            )
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST, f"malformed JSON body: {exc}", param="body"
            ) from exc
        if not isinstance(data, dict):
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST, "the body must be a JSON object", param="body"
            )
        return data

    def _check_authorization(self) -> None:
        cfg = self.config or ServeConfig()
        if not cfg.api_key:
            return  # development mode: open
        header = self.headers.get("Authorization", "") or ""
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not constant_time_equals(token, cfg.api_key):
            raise HttpProblem(
                HTTPStatus.UNAUTHORIZED,
                "Incorrect API key provided.",
                err_type="authentication_error",
                code="invalid_api_key",
            )

    def _pipeline(self) -> ChatPipeline:
        """The pipeline the factory binds at startup; a bare handler has none."""
        pipeline = self.pipeline
        if pipeline is None:
            raise HttpProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "the front is not bound to a pipeline",
                err_type="engine_error",
                code="engine_error",
                param="model",
            )
        return pipeline

    # -- verbs ──────────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        if (self.config or ServeConfig()).cors:
            self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = self._clean(self.path)
        try:
            if path in (ROUTE_HEALTH, "/"):
                self._json(
                    {
                        "status": "ok",
                        "service": "c2c-serve",
                        "version": __version__,
                        "routes": canonical_routes(),
                    }
                )
            elif path in (ROUTE_MODELS, "/" + MODELS):
                data = []
                for mid, note, ctx in self._pipeline().hub.describe_models():
                    item = {
                        "id": mid,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "c2c",
                    }
                    if note:
                        item["description"] = note
                    if ctx:
                        item["max_context_tokens"] = int(ctx)
                    data.append(item)
                self._json({"object": "list", "data": data})
            elif path == ROUTE_WELL_KNOWN:
                from .a2a import build_agent_card

                self._json(build_agent_card(self.config, self._pipeline().hub))
            else:
                raise HttpProblem(
                    HTTPStatus.NOT_FOUND, f"unknown route {path}", code="resource_not_found"
                )
        except HttpProblem as problem:
            self._json(problem.to_json(), problem.status)

    def do_POST(self) -> None:
        path = self._clean(self.path)
        try:
            self._check_authorization()
            body = self._read_body()
            if path in (ROUTE_CHAT_COMPLETIONS, ROUTE_LEGACY_CHAT):
                self._chat(body)
            elif path in (ROUTE_COMPLETIONS, ROUTE_LEGACY_COMPLETIONS):
                self._completions(body)
            else:
                raise HttpProblem(
                    HTTPStatus.NOT_FOUND, f"unknown route {path}", code="resource_not_found"
                )
        except HttpProblem as problem:
            self._json(problem.to_json(), problem.status)
        except Exception as exc:  # last line of defence
            self.log_error("unhandled: %r", exc)
            self._json(
                HttpProblem(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"internal error: {exc.__class__.__name__} (see the server log)",
                ).to_json(),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # -- the two completions routes ────────────────────────────────────────
    def _pick_int(self, body: dict, *keys: str, default: int = 0) -> int:
        for key in keys:
            if body.get(key) is not None:
                try:
                    return int(body[key])
                except (TypeError, ValueError):
                    continue
        return default

    def _chat(self, body: dict) -> None:
        model = str(body.get("model") or "")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HttpProblem(
                HTTPStatus.BAD_REQUEST, "'messages' must be a non-empty array", param="messages"
            )
        prompt_text = messages_to_text(messages)
        max_new_tokens = self._pick_int(
            body,
            "max_tokens",
            "max_completion_tokens",
            default=(self.config or ServeConfig()).max_new_tokens,
        )
        temperature = float(body.get("temperature") or 0.0)
        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        stop = body.get("stop") if isinstance(body.get("stop"), list) else None
        c2c_options = body.get("c2c") if isinstance(body.get("c2c"), dict) else {}
        result = self._pipeline().complete(
            model=model,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tools=tools,
            stop=stop,
            c2c_options=c2c_options,
        )
        self._log_exchange(
            "chat/completions",
            model,
            prompt_text,
            result["answer"],
            used=bool(result.get("used_cache")),
        )
        if body.get("stream"):
            self._stream(result, chat=True)
            return
        self._json(self._build_completion(result, chat=True))

    def _completions(self, body: dict) -> None:
        model = str(body.get("model") or "")
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        max_new_tokens = self._pick_int(
            body, "max_tokens", default=(self.config or ServeConfig()).max_new_tokens
        )
        temperature = float(body.get("temperature") or 0.0)
        stop = body.get("stop") if isinstance(body.get("stop"), list) else None
        c2c_options = body.get("c2c") if isinstance(body.get("c2c"), dict) else {}
        result = self._pipeline().complete(
            model=model,
            prompt_text=str(prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tools=None,
            stop=stop,
            c2c_options=c2c_options,
        )
        self._log_exchange(
            "completions", model, str(prompt), result["answer"], used=bool(result.get("used_cache"))
        )
        if body.get("stream"):
            self._stream(result, chat=False)
            return
        self._json(self._build_completion(result, chat=False))

    # -- response objects, the OpenAI shape ────────────────────────────────
    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def _build_completion(self, result: dict, *, chat: bool) -> dict:
        created = int(time.time())
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        }
        base = {
            "created": created,
            "model": result["model"],
            "usage": usage,
            "used_cache": bool(result.get("used_cache")),
        }  # the fusion, declared
        if result.get("gate") == "block":
            base["c2c"] = {"gate": "block"}  # and so the refusal, on the face of the answer
        if chat:
            base["id"] = self._new_id("chat-completion")
            base["object"] = "chat.completion"
            base["choices"] = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": "stop",
                }
            ]
        else:
            base["id"] = self._new_id("completion")
            base["object"] = "text.completion"
            base["choices"] = [
                {"index": 0, "text": result["answer"], "logprobs": None, "finish_reason": "stop"}
            ]
        return base

    def _stream(self, result: dict, *, chat: bool) -> None:
        """Server-sent events: the standard streaming shape of the wire."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", EVENT_STREAM + "; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # finite stream: close after, at the [DONE]
        if (self.config or ServeConfig()).cors:
            self._cors()
        self.end_headers()
        created = int(time.time())
        chunk_id = self._new_id("chunk")
        object_name = "chat.completion.chunk" if chat else "text.completion.chunk"
        base = {"id": chunk_id, "object": object_name, "created": created, "model": result["model"]}

        def frame(payload_delta, finish: str | None) -> bytes:
            if chat:
                choice = {"index": 0, "delta": payload_delta, "finish_reason": finish}
            else:
                choice = {
                    "index": 0,
                    "text": payload_delta.get("content", ""),
                    "logprobs": None,
                    "finish_reason": finish,
                }
            return (
                b"data: "
                + json.dumps({**base, "choices": [choice]}, ensure_ascii=False).encode("utf-8")
                + b"\n\n"
            )

        answer = result["answer"]
        if chat:
            self.wfile.write(frame({"role": "assistant"}, None))
        width = 24  # one printable word per frame
        for i in range(0, max(len(answer), 1), width):
            piece = answer[i : i + width]
            self.wfile.write(frame({"content": piece} if chat else {"content": piece}, None))
        self.wfile.write(frame({}, "stop"))
        self.wfile.write(DONE_SENTINEL)
        self.wfile.flush()

    # -- path hygiene ───────────────────────────────────────────────────────
    @staticmethod
    def _clean(path: str) -> str:
        path = path.split("?", 1)[0].split("#", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/") or "/"
        return ALIASES.get(path, path)


class C2CServer(ThreadingMixIn, TCPServer):
    """Handle each request in its own thread; a slow model must not
    block a quick one."""

    daemon_threads = True


# ---------------------------------------------------------------------------
# the server factory and the console entry point
# ---------------------------------------------------------------------------


def _make_ssl_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def create_server(config: ServeConfig | None = None, *, hub=None) -> C2CServer:
    cfg = config if config is not None else load_config().serve
    pipeline = ChatPipeline(hub=hub if hub is not None else default_hub)
    OpenAIRequestHandler.pipeline = pipeline
    OpenAIRequestHandler.config = cfg
    server_obj = C2CServer((cfg.host, cfg.port), OpenAIRequestHandler)
    if cfg.certfile and cfg.keyfile:
        server_obj.socket = _make_ssl_context(cfg.certfile, cfg.keyfile).wrap_socket(
            server_obj.socket, server_side=True
        )
    elif bool(cfg.certfile) != bool(cfg.keyfile):
        # the API raises; the console (main, via _preflight_tls) prints and exits —
        # half a TLS pair is a user error at every door, not a silent plain wire
        missing = "certfile" if not cfg.certfile else "keyfile"
        raise ValueError(
            f"HTTPS needs both certfile and keyfile, readable PEM files; the {missing} is missing"
        )
    return server_obj


def serve_forever(config: ServeConfig | None = None, *, hub=None) -> None:
    cfg = config if config is not None else load_config().serve
    server_obj = create_server(cfg, hub=hub)
    scheme = "https" if (cfg.certfile and cfg.keyfile) else "http"
    sys.stderr.write(banner("serve", __version__) + "\n")
    sys.stderr.write("  listening on " + style(f"{scheme}://{cfg.host}:{cfg.port}", "bold") + "\n")
    for route in canonical_routes():
        sys.stderr.write(f"    * {route}\n")
    try:
        _pipe = OpenAIRequestHandler.pipeline
        _claims = [
            (_mid, _ctx)
            for _mid, _note, _ctx in (_pipe.hub.describe_models() if _pipe else [])
            if _ctx
        ]
    except Exception:  # a banner never blocks a boot
        _claims = []
    for _mid, _ctx in _claims:
        sys.stderr.write(f"    * {_mid} — answers within {_ctx} tokens, the model's own claim\n")
    try:
        if _pipe is not None and not _pipe.hub.curated and not _pipe.hub.describe_models():
            sys.stderr.write(
                "    no models yet: --pair receiver←sharer registers a collaboration;"
                " pip install 'c2c-cache[train]' arms the reference engine,"
                " and any name then answers\n"
            )
    except Exception:  # a hint never blocks a boot
        pass
    sys.stderr.write("  the harness stays the master; C2C is the wire between models.\n")
    try:
        server_obj.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server_obj.server_close()


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="OpenAI-compatible HTTPS front for cache-to-cache collaborations (spec HL-1).",
        epilog="the harness stays the master; C2C is the wire between models.",
    )
    parser.add_argument("--host", default=None, help="interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port to listen on (default 8788)")
    parser.add_argument("--certfile", default=None, help="PEM certificate for HTTPS")
    parser.add_argument("--keyfile", default=None, help="PEM key for HTTPS")
    parser.add_argument(
        "--api-key", default=None, help="when set, require 'Authorization: Bearer <key>'"
    )
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        help="engine adapter for auto-built models (default 'reference')",
    )
    parser.add_argument(
        "--pair",
        action="append",
        metavar="RECEIVER←SHARER",
        help="register a collaboration pair (repeatable); attach a "
        "trained fuser: 'receiver←sharer:/path/to/weights.pt'",
    )
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help="the server both sides of every --pair live on (remote engines: vllm-wired, tgi, ollama)",
    )
    parser.add_argument(
        "--receiver-url", default=None, metavar="URL", help="the receiver's server, if split"
    )
    parser.add_argument(
        "--sharer-url", default=None, metavar="URL", help="the sharer's server, if split"
    )
    parser.add_argument("--config", default=None, help="path to a config.json")
    parser.add_argument(
        "--privacy",
        action="store_true",
        help="no cache, no trace: the sharer's context is refused "
        "before it is attempted, and the access logs carry "
        "SHA-256 digests only (EX-5 privacy; the seal is offered, "
        "not yet wired — see serve/privacy)",
    )
    return parser


def _register_cli_pairs(
    hub,
    pairs: Sequence[str],
    *,
    engine: str | None = None,
    url: str | None = None,
    receiver_url: str | None = None,
    sharer_url: str | None = None,
) -> None:
    """Parse ``--pair`` specifications and register them on the hub."""
    probed: set[str] = set()
    for raw in pairs:
        body, sep, tail = str(raw).partition(":")
        weights = tail if (sep and tail and os.path.isfile(tail)) else None
        parsed = hub.parse_pair_id(body) or hub.parse_pair_id(raw)
        if parsed is None:
            sys.stderr.write(f"c2c-serve: ignoring un-parsable pair {raw!r}\n")
            continue
        receiver_id, sharer_id = parsed
        for mid, base, role, peer in (
            (receiver_id, receiver_url or url, "receiver", sharer_id),
            (sharer_id, sharer_url or url, "sharer", receiver_id),
        ):
            if base:
                hub.register_model(
                    mid,
                    options={"base_url": base, "role": role, "peer_model": peer},
                    note=f"side of a wired pair ({role})",
                )
        fuser = None
        if weights is not None:
            fuser = _load_fuser_state(weights)
        hub.register_pair(receiver=receiver_id, sharer=sharer_id, fuser=fuser)
        name = engine or hub.engine_name
        if name not in probed:
            probed.add(name)
            try:
                from ..integrations.registry import engines

                engines.load(name, model_id="__bind-probe__")
            except (ModuleNotFoundError, ImportError, ValueError, RuntimeError) as exc:
                sys.stderr.write(
                    f"c2c-serve: the pair {raw!r} is on the shelf, but the engine "
                    f"{name!r} cannot build its models: {exc}\n"
                    f"             requests naming it relay without fusion; to fit the "
                    f"fuser, pip install 'c2c-cache[train]'\n"
                )


def _load_fuser_state(path: str):
    """Read a fuser checkpoint; rebuild the nets from the embedded geometries."""
    try:
        from ..train.scheme import load_checkpoint_blob

        blob = load_checkpoint_blob(path)
        if not isinstance(blob, dict):
            return None
        state = blob.get("state_dict")
        if state is None:
            return None
        geometry = blob.get("geometry") or {}
        if not (geometry.get("receiver") and geometry.get("sharer") and geometry.get("mapping")):
            return _FuserProxy(state)  # a legacy blob: the proxy says so
        from ..config import BlendConfig, FuserConfig, GateConfig
        from ..fuser.core import Fuser
        from ..types import LayerGeometry

        fuser = Fuser(
            LayerGeometry(**geometry["receiver"]),
            LayerGeometry(**geometry["sharer"]),
            geometry["mapping"],
            fuser_config=FuserConfig(**(geometry.get("fuser_config") or {})),
            gate_config=GateConfig(**(geometry.get("gate_config") or {})),
            blend_config=BlendConfig(**(geometry.get("blend_config") or {})),
        )
        fuser.load_state_dict(state, strict=False)
        fuser.eval()  # at serve: gates hard, dropout off
        return fuser
    except Exception as exc:
        sys.stderr.write(f"c2c-serve: cannot read fuser weights {path!r}: {exc}\n")
        return None


class _FuserProxy:
    """A fuser placeholder that carries a trained state until a real one
    can be built. Calling it before the model geometries are known is a
    configuration error and says so — helpfully, not cryptically."""

    def __init__(self, state_dict):
        self.state = state_dict
        self._real = None

    def __call__(self, receiver_cache, sharer_cache, **kwargs):
        if self._real is None:
            msg = (
                "the checkpoint is loaded but the model geometries are unknown to "
                "the proxy; register the pair through the Python API with a "
                "constructed Fuser to enable cache-to-cache fusion"
            )
            raise RuntimeError(msg)
        return self._real(receiver_cache, sharer_cache, **kwargs)


def _preflight_tls(cfg: ServeConfig, who: str) -> None:
    """Speak before the front binds: a missing half of a TLS pair is a user
    error to name, not a mystery OSError to endure at the bind."""
    if not cfg.certfile and not cfg.keyfile:
        return
    missing = [x for x in (cfg.certfile, cfg.keyfile) if not x or not os.path.isfile(x)]
    if missing:
        from ..utils.console import error_hint

        names = ", ".join(repr(m or "unset") for m in missing)
        print(
            error_hint(
                f"{who}: HTTPS needs --certfile and --keyfile, both readable PEM files; "
                f"missing or unreadable: {names}",
                hint="for a test certificate: openssl req -x509 -nodes -newkey rsa:2048 "
                "-subj '/CN=localhost' -keyout server.key -out server.crt -days 365",
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser("c2c-serve")
    args = parser.parse_args(argv)
    base = load_config(args.config)
    import os

    updates = {
        "host": args.host,
        "port": args.port,
        "certfile": args.certfile,
        "keyfile": args.keyfile,
        "api_key": args.api_key if args.api_key is not None else os.environ.get("C2C_API_KEY"),
        "privacy": True if args.privacy else None,
    }
    cfg = ServeConfig(
        **{
            **vars(base.serve if hasattr(base, "serve") else base),
            **{k: v for k, v in updates.items() if v is not None},
        }
    )
    if not cfg.api_key and not is_loopback(cfg.host):
        from ..utils.console import error_hint

        print(
            error_hint(
                "the front, off the loopback, must have a key: --api-key KEY, "
                "or C2C_API_KEY in the environment",
                hint="the loopback serves without one; the network does not",
            ),
            file=sys.stderr,
        )
        return 2
    if args.engine:
        default_hub.set_engine(args.engine)
    if args.pair:
        _register_cli_pairs(
            default_hub,
            args.pair,
            engine=args.engine,
            url=args.url,
            receiver_url=args.receiver_url,
            sharer_url=args.sharer_url,
        )
    _preflight_tls(cfg, "c2c-serve")
    try:
        serve_forever(cfg)
    except OSError as exc:
        sys.stderr.write(f"c2c-serve: cannot bind {cfg.host}:{cfg.port}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
