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
            text = (payload.get("prompt") or [""])[0]
            self._say({"tokens": [[ord(ch) for ch in text]]})
        elif self.path == "/detokenize":
            toks = (payload.get("tokens") or [[]])[0]
            self._say({"text": ["".join(chr(int(t)) for t in toks)]})
        elif self.path == "/v1/completions":
            self._say(
                {
                    "object": "text_completion",
                    "choices": [
                        {"index": 0, "text": f"the server spoke to {payload.get('model')}"}
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
