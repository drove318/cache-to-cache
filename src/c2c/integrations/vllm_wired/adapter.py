"""The wired adapter — the front's half of the wire.

Mechanism (spec §4.1, mirrored in ``c2c-engines(7)``): the engine is
already running, its connector already holds the wire; this adapter
never captures, never installs, never ships a row. It does two things:

* speaks the engine's own dialect for tokens (``POST /tokenize`` and
  ``POST /detokenize``, the roads the engine's serve package opens);
* stamps every completion with the pair's c2c object — the sharer's
  request first (prefill-only, the rows it computes wait in the
  connector's staging), then the receiver's, naming the peer — and
  returns the receiver's answer. The fusion itself is the connector's:
  three moves on the engine's device, inside the load window.

The ABI speaks token ids everywhere; the engine accepts them directly
(``prompt: list[int] | str | ...``), so ids ride the wire untouched and
the tokenizer stays the engine's alone.

Pairing is the hub's gift: ``c2c-serve`` registers both sides of a pair
with per-side options — ``role`` (which half this adapter is) and
``peer_model`` (what to name on the other half's request). Built without
them, the adapter speaks plain relay and says so with its claim:
``FUSES_IN_GENERATE`` is the truth the proxy reports for ``used_cache``,
and a truth it is only when the stamp is on the wire.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid

from ...types import LayerGeometry, ModelSpec
from ..registry import EngineAdapter

__all__ = ["VLLMWiredAdapter", "ServedRejected", "available"]


class ServedRejected(RuntimeError):
    """The served said 4xx to the shape of a request that came over the wire.

    The fault is the caller's, not the fuse's: a refusal that strikes down
    the body (400, 413, 422, 429 — the length, the vars, the window) rides
    on whole, the served's words kept, so a harness learns what to shorten
    rather than being handed an unknowable 500. A refusal that strikes down
    the front (a road gone, a key refused) is our own misfitting and stays
    a plain RuntimeError: 500, and the server log.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.served_status = int(status)


def available() -> bool:
    return True  # needs only a running server


class VLLMWiredAdapter(EngineAdapter):
    """The front's half: stamps the pair, rides the engine's roads."""

    engine_name = "vLLM (wired)"
    required_extra = None  # the engine lives next door, reached over http
    TOOLS = "chat"  # the completions route has no tools arm; when tools ride, the
                    # receiver's leg switches to /v1/chat/completions and the served
                    # model answers in kind (native tool_calls), rather than have the
                    # model fall back on the caller's stack and imitate the call as prose
    DEGRADATION_IF = (
        "the server was launched without a wire: the connector serves the "
        "identity and the pair answers as the receiver alone"
    )

    #: the pair's fusion is real inside generate: the stamp rides the
    #: ferry, the connector does the rest, and any failed leg raises
    FUSES_IN_GENERATE = True

    def __init__(self, model_id: str, **options):
        super().__init__(model_id, **options)
        self.base = str(self.options.get("base_url") or "http://127.0.0.1:8000").rstrip("/")
        self.timeout = float(self.options.get("timeout", 600.0))
        self.role = str(self.options.get("role") or "")
        self.peer_model = str(self.options.get("peer_model") or "")
        self.api_key = self.options.get("api_key")
        self.log = logging.getLogger("c2c.wired").getChild("adapter")

    @classmethod
    def report_context(cls, model_id: str, **options) -> int | None:
        """What the running server claims it serves: its own card, asked live.

        The weights are the server's, not ours, so the window is read over
        HTTP — the served's /v1/models entry, its max_model_len field. Any
        silence (no server, no entry, no field) is silence: None, and the
        gallery holds the number back rather than invent one.
        """
        base = str(options.get("base_url") or "http://127.0.0.1:8000").rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5.0) as response:
                card = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            return None
        entries = card.get("data") or []
        for entry in entries:
            if entry.get("id") == str(model_id):
                length = entry.get("max_model_len")
                return int(length) if length else None
        if len(entries) == 1:  # one model served: its card is the answer
            length = entries[0].get("max_model_len")
            return int(length) if length else None
        return None

    # -- the engine's roads, taken plainly ──────────────────────────────────
    def _post(self, path: str, payload: dict | None = None, *, method: str = "POST") -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = None
        if method != "GET":
            body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400] if exc.fp else ""
            msg = f"the wired server refused {path!r}: {exc.code} {exc.reason} {detail}".strip()
            if exc.code in (400, 413, 422, 429):
                raise ServedRejected(exc.code, msg) from exc  # the caller's, whole
            raise RuntimeError(msg) from exc
        except OSError as exc:
            msg = f"the wired server at {self.base} does not answer ({exc})"
            raise RuntimeError(msg) from exc

    def encode(self, text: str) -> list[int]:
        """The engine's own tokenizer, asked over the engine's own road.

        The /tokenize road will not step in on a batch: the server expects
        the prompt passed as a string, not called through the B side of the
        list, so the string rides as itself. The answer comes back as a
        tokens list, or batch; the row is taken when the server sends the
        batch of lists.
        """
        answer = self._post("/tokenize", {"model": self.model_id, "prompt": text})
        toks = answer.get("tokens") or []
        if toks and isinstance(toks[0], list):
            toks = toks[0]
        return [int(t) for t in toks]

    def decode_tokens(self, token_ids) -> str:
        answer = self._post(
            "/detokenize", {"model": self.model_id, "tokens": [int(t) for t in token_ids]}
        )
        text = answer.get("text")
        if isinstance(text, list):
            for row in text:
                if isinstance(row, str):
                    return row
            return ""
        return str(text) if text is not None else ""

    def _build_spec(self) -> ModelSpec:
        """The card the server shows; the geometry it keeps to itself.

        The listing proves the name is served (a typo dies here, with the
        flag that mends it named); geometry comes from the operator's
        ``options.geometry`` or says so plainly in the card's own name.
        """
        self._prove_served()
        geo = self.options.get("geometry")
        if isinstance(geo, dict):
            geo = LayerGeometry(**geo)  # the option, consulted: the advice is actionable
        if geo is None:
            geo = LayerGeometry(
                layers=1,
                hidden_size=1,
                num_heads=1,
                name=f"{self.model_id} (the geometry lives in the server)",
            )
        return ModelSpec(
            id=self.model_id,
            geometry=geo,
            family="vllm-wired",
            instruction_tuned=bool(self.options.get("instruction_tuned", True)),
        )

    def _prove_served(self) -> None:
        listing = self._post("/v1/models", method="GET")
        ids = {str(card.get("id")) for card in listing.get("data") or []}
        if self.model_id not in ids:
            served = ", ".join(sorted(ids)) or "(none)"
            msg = (
                f"the wired server does not serve {self.model_id!r}; it serves: {served} — "
                "check --served-model-name / --receiver / --sharer"
            )
            raise RuntimeError(msg)

    # -- the two legs, in the paper's order ─────────────────────────────────
    def generate(
        self,
        prompt_tokens,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        stop=None,
        tools=None,
        messages=None,  # the harness's array; only the chat road reads it
    ) -> str | dict:
        ids = [int(t) for t in prompt_tokens]
        stamp = None
        if self.role == "receiver" and self.peer_model:
            pair = uuid.uuid4().hex[:12]
            sharer_req = f"c2c-s-{uuid.uuid4().hex[:10]}"
            self.log.info("pair %s: the sharer %r prefills its half", pair, self.peer_model)
            self._post(
                "/v1/completions",
                {
                    "model": self.peer_model,
                    "prompt": ids,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "kv_transfer_params": {
                        "c2c": {"role": "sharer", "pair": pair, "self_req": sharer_req},
                    },
                },
            )
            stamp = {"role": "receiver", "pair": pair, "peer": sharer_req}
        if tools and messages:
            # the chat road: the tools array and the messages ride all the way
            # in to the container's chat endpoint, which applies its own tool
            # parser and brings back the structure — rather than have the model
            # fall back on prose imitation, the harness sees the calls itself
            self.log.info(
                "the receiver %r speaks over the chat road, the tools ride in",
                self.model_id,
            )
            return self._chat(messages, tools, max_new_tokens, temperature, stop, stamp)
        if stamp is None:
            # no pair, no pretense: the receiver answers from its own cache alone
            return self._complete(ids, max_new_tokens, temperature, stop)
        self.log.info("pair %s: the receiver %r speaks, the wire fuses", pair, self.model_id)
        return self._complete(ids, max_new_tokens, temperature, stop, stamp=stamp)

    def _complete(self, ids, max_new_tokens, temperature, stop, stamp=None):
        body = {
            "model": self.model_id,
            "prompt": ids,  # ids ride untouched: the engine tokenizes no string
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if stop:
            body["stop"] = list(stop)
        if stamp is not None:
            body["kv_transfer_params"] = {"c2c": stamp}
        answer = self._post("/v1/completions", body)
        for choice in answer.get("choices") or []:
            text = choice.get("text")
            if isinstance(text, str):
                return text
        msg = f"the wired server answered without text: {str(answer)[:200]}"
        raise RuntimeError(msg)

    def _chat(self, messages, tools, max_new_tokens, temperature, stop, stamp=None):
        """The chat road: the container's endpoint tokenizes the messages and
        its tool parser picks out the calls, so the harness gets the structure
        back rather than a string the model had to imitate.
        """
        body = {
            "model": self.model_id,
            "messages": list(messages),  # the array rides intact to the container
            "tools": list(tools),
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if stop:
            body["stop"] = list(stop)
        if stamp is not None:
            body["kv_transfer_params"] = {"c2c": stamp}
        answer = self._post("/v1/chat/completions", body)
        for choice in answer.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict):
                return {
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                    "finish_reason": choice.get("finish_reason") or "stop",
                }
        msg = f"the wired server answered without a message: {str(answer)[:200]}"
        raise RuntimeError(msg)

    def close(self) -> None:
        return None
