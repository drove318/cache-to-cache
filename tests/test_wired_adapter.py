"""The wired adapter, proved against a server that is not vLLM.

A loopback HTTP server records every request; the adapter rides it like
the real one. What is held to the word: the tokenizer stays the
server's (ids are its, never invented), the ids ride the prompt field
untouched, the stamp rides the ferry the completion protocol defines,
the two legs fall in the paper's order, and — the contract the whole
runbook leans on — the front reports ``used_cache`` as the truth of
what the pair actually did.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from c2c.integrations.registry import engines
from c2c.integrations.vllm_wired.adapter import VLLMWiredAdapter
from c2c.serve.openai_proxy import ChatPipeline
from c2c.serve.registry import ModelHub

_SERVED = ("receiver-model", "sharer-model")


class _Recorder(BaseHTTPRequestHandler):
    """The wired server's stand-in: answers by the gallery's word, remembers all."""

    received: list = []  # class level: the threads of one server share it

    def log_message(self, *args):
        pass  # the gallery's silence: this server does not narrate

    def do_GET(self):  # noqa: N802 — http.server's own spelling
        body = {"object": "list", "data": [{"id": name} for name in _SERVED]}
        self._say(body)

    def do_POST(self):  # noqa: N802 — http.server's own spelling
        declared = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(declared).decode("utf-8") or "{}")
        type(self).received.append((self.path, payload))
        if self.path == "/tokenize":
            # the tokenizer answers to the batch or the string alike, and
            # mirrors the shape of the request back
            prompt = payload.get("prompt") or ""
            batch = isinstance(prompt, list)
            text = prompt[0] if batch else str(prompt)
            one = [ord(ch) for ch in text]
            self._say({"tokens": [one] if batch else one})
        elif self.path == "/detokenize":
            toks = payload.get("tokens") or []
            batch = bool(toks) and isinstance(toks[0], list)
            row = toks[0] if batch else toks
            text = "".join(chr(int(t)) for t in row)
            self._say({"text": [text] if batch else text})
        elif self.path == "/v1/completions":
            self._say(
                {
                    "object": "text_completion",
                    "choices": [
                        {"index": 0, "text": f"the server spoke to {payload.get('model')}"}
                    ],
                }
            )
        elif self.path == "/v1/chat/completions":
            # the chat road: when the tools array comes in, the container
            # brings back the structure — the calls, not the prose
            if payload.get("tools"):
                self._say(
                    {
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_capital",
                                                "arguments": json.dumps({"country": "France"}),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                )
            else:
                self._say(
                    {
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "the server spoke"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
        else:
            self.send_error(404, "no such road")

    def _say(self, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    _Recorder.received = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _adapter(side: str, base: str, **options) -> VLLMWiredAdapter:
    opts = {"base_url": base, **options}
    return VLLMWiredAdapter(side, **opts)


class TestTheAdaptersOwnHands:
    def test_the_tokenizer_stays_the_servers(self, server):
        adapter = _adapter("receiver-model", server)
        ids = adapter.encode("abc")
        assert ids == [97, 98, 99]  # the servers ids, none invented
        assert adapter.decode_tokens(ids) == "abc"

    def test_a_lone_half_relays_without_pretense(self, server):
        adapter = _adapter("receiver-model", server)
        answer = adapter.generate([104, 105])
        assert answer.startswith("the server spoke")
        roads = [path for path, _ in _Recorder.received if path == "/v1/completions"]
        assert len(roads) == 1  # one leg, no pair, no stamp
        _, body = _Recorder.received[-1]
        assert "kv_transfer_params" not in body

    def test_the_pair_rides_the_ferry_in_the_papers_order(self, server):
        receiver = _adapter(
            "receiver-model",
            server,
            role="receiver",
            peer_model="sharer-model",
        )
        answer = receiver.generate([104, 105])
        legs = [body for path, body in _Recorder.received if path == "/v1/completions"]
        assert len(legs) == 2  # the sharer prefills, then the receiver speaks
        assert legs[0]["model"] == "sharer-model"
        assert legs[0]["max_tokens"] == 1  # the one token that costs nothing
        assert legs[1]["model"] == "receiver-model"
        stamp_s = legs[0]["kv_transfer_params"]["c2c"]
        stamp_r = legs[1]["kv_transfer_params"]["c2c"]
        assert stamp_s["role"] == "sharer" and stamp_r["role"] == "receiver"
        assert stamp_s["pair"] == stamp_r["pair"]  # one pair, two requests
        assert stamp_r["peer"] == stamp_s["self_req"]  # the receiver names its peer
        assert legs[1]["prompt"] == [104, 105]  # ids rode untouched, end to end
        assert answer.startswith("the server spoke")

    def test_the_tools_riding_brings_back_the_calls(self, server):
        """The tools array and the messages ride all the way in to the
        container's chat endpoint; the parser brings back the structure,
        and the pair stamping rides the chat road as ever the ids."""
        receiver = _adapter(
            "receiver-model",
            server,
            role="receiver",
            peer_model="sharer-model",
        )
        answer = receiver.generate(
            [104, 105],
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            messages=[{"role": "user", "content": "what is the capital of France?"}],
        )
        assert isinstance(answer, dict)  # the chat road: the structure, no string reply
        assert answer["finish_reason"] == "tool_calls"
        assert answer["content"] is None  # the model spoke through the calls
        call = answer["tool_calls"][0]
        assert call["function"]["name"] == "get_capital"  # the parser picked it out
        assert json.loads(call["function"]["arguments"]) == {"country": "France"}
        _, body = _Recorder.received[-1]  # the last leg was the chat road
        assert body["messages"] == [{"role": "user", "content": "what is the capital of France?"}]
        assert body["tools"]  # and the schema rode with it
        assert body["kv_transfer_params"]["c2c"]["role"] == "receiver"  # the stamp, too

    def test_a_name_the_server_does_not_serve_dies_naming_the_flag(self, server):
        adapter = _adapter("no-such-model", server)
        with pytest.raises(RuntimeError, match="served-model-name"):
            adapter.spec()


class TestTheFrontKeepsItsWord:
    """The contract the runbook leans on: the front, the hub, the stamp, the truth."""

    def test_used_cache_is_the_truth_of_the_pair(self, server):
        hub = ModelHub()
        hub.set_engine("vllm-wired")
        from c2c.serve.openai_proxy import _register_cli_pairs

        _register_cli_pairs(
            hub,
            ["receiver-model←sharer-model"],
            engine="vllm-wired",
            url=server,
        )
        result = ChatPipeline(hub=hub).complete(
            model="c2c/receiver-model←sharer-model",
            prompt_text="hi",
            max_new_tokens=8,
            temperature=0.0,
            tools=None,
            stop=None,
        )
        assert result["used_cache"] is True  # the pair fused inside the server
        assert result["answer"].startswith("the server spoke")
        legs = [body for path, body in _Recorder.received if path == "/v1/completions"]
        assert any(b.get("kv_transfer_params") for b in legs)  # and it rode the ferry

    def test_a_plain_relay_still_says_false(self, server):
        hub = ModelHub()
        hub.set_engine("vllm-wired")
        hub.register_model("receiver-model", options={"base_url": server})
        result = ChatPipeline(hub=hub).complete(
            model="c2c/receiver-model",
            prompt_text="hi",
            max_new_tokens=8,
            temperature=0.0,
            tools=None,
            stop=None,
        )
        assert result["used_cache"] is False  # no pair, no claim — the house keeps its word

    def test_the_registry_knows_the_road_by_heart(self):
        assert "vllm-wired" in engines.registered_names()
        built = engines.load("vllm-wired", model_id="receiver-model", base_url="http://x")
        assert isinstance(built, VLLMWiredAdapter)
        assert built.FUSES_IN_GENERATE is True

    def test_one_server_wearing_both_halves(self, server):
        """The boxs true topology: one name, the pair folds onto itself."""
        hub = ModelHub()
        hub.set_engine("vllm-wired")
        from c2c.serve.openai_proxy import _register_cli_pairs

        _register_cli_pairs(hub, ["receiver-model←receiver-model"], engine="vllm-wired", url=server)
        result = ChatPipeline(hub=hub).complete(
            model="c2c/receiver-model←receiver-model",
            prompt_text="ab",
            max_new_tokens=4,
            temperature=0.0,
            tools=None,
            stop=None,
        )
        assert result["used_cache"] is True  # the pair claim survives the fold
        legs = [body for path, body in _Recorder.received if path == "/v1/completions"]
        assert len(legs) == 2
        assert legs[0]["model"] == legs[1]["model"] == "receiver-model"  # one server, two legs
        assert legs[0]["kv_transfer_params"]["c2c"]["role"] == "sharer"
        assert legs[1]["kv_transfer_params"]["c2c"]["role"] == "receiver"

    def test_the_streaming_road_carries_the_calls_as_a_field(self, server):
        """The streamer must not dump the calls into the content: on the
        chat road the calls ride the final frame as a field, the content
        alone — so the harness parses the delta and stops on the finish."""
        from c2c.config import ServeConfig
        from c2c.serve.openai_proxy import _register_cli_pairs, create_server

        hub = ModelHub()
        hub.set_engine("vllm-wired")
        _register_cli_pairs(
            hub,
            ["receiver-model←sharer-model"],
            engine="vllm-wired",
            url=server,
        )
        httpd = create_server(ServeConfig(host="127.0.0.1", port=0), hub=hub)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "c2c/receiver-model←sharer-model",
                        "messages": [{"role": "user", "content": "capital of France?"}],
                        "tools": [{"type": "function", "function": {"name": "get_capital"}}],
                        "max_tokens": 8,
                        "stream": True,
                    }
                ).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in raw.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        assert frames, "a stream, with no frames"
        final = frames[-1]["choices"][0]
        assert final["finish_reason"] == "tool_calls"  # the finish, at the end
        assert final["delta"]["tool_calls"]  # the calls, as a field in the delta
        streamed = "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames)
        assert "get_capital" not in streamed  # the calls stay out of the content


class TestTheServedRefuses:
    """A 4xx from the served is the caller's to read, not a 500 mystery."""

    def test_a_refusal_on_the_shape_is_the_callers_own(self, server, monkeypatch):
        from c2c.integrations.vllm_wired.adapter import ServedRejected

        def refuse(self):
            self.send_error(400, "over the ceiling")

        monkeypatch.setattr(_Recorder, "do_POST", refuse)
        adapter = _adapter("receiver-model", server, role="receiver", peer_model="sharer-model")
        with pytest.raises(ServedRejected) as got:
            adapter._post("/v1/chat/completions", {"model": "receiver-model"})
        assert got.value.served_status == 400  # the served's status, kept

    def test_a_refusal_on_the_front_stays_ours(self, server, monkeypatch):
        from c2c.integrations.vllm_wired.adapter import ServedRejected

        def refuse(self):
            self.send_error(404, "no such road")

        monkeypatch.setattr(_Recorder, "do_POST", refuse)
        adapter = _adapter("receiver-model", server, role="receiver", peer_model="sharer-model")
        with pytest.raises(RuntimeError) as got:
            adapter._post("/tokenize", {"prompt": "ab"})
        assert not isinstance(got.value, ServedRejected)  # a misfitting on our side

    def test_the_window_is_what_the_server_says(self, server, monkeypatch):
        def card(self):
            self._say(
                {
                    "object": "list",
                    "data": [{"id": "receiver-model", "max_model_len": 524288}],
                }
            )

        monkeypatch.setattr(_Recorder, "do_GET", card)
        claimed = VLLMWiredAdapter.report_context("receiver-model", base_url=server)
        assert claimed == 524288  # read over http, never invented

    def test_silence_is_silence(self, server):
        # the stand-in's card prints no max_model_len: the gallery says nothing
        assert VLLMWiredAdapter.report_context("receiver-model", base_url=server) is None
