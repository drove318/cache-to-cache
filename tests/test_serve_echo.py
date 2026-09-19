"""The relay engine: the front that answers when the house nets are furled.

The core wheel installs without torch; the out-of-the-box promise lives or
dies here. Everything the relay claims — mirror semantics, the honest note,
the cache hooks, `used_cache: False` where nothing fused — is verified as a
consumer of the server would observe it, not as a poke at internals.
"""

from __future__ import annotations

import json
import threading
import urllib.request


def _boom_engine(monkeypatch):
    """Make every engine lookup fail the way a core-only install does."""
    from c2c.integrations import engines

    def boom(name, **_kw):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(engines, "load", boom)


class TestTheMirrorItself:
    def _engine(self):
        from c2c.serve.echo import EchoEngine

        return EchoEngine("probe")

    def test_encode_decode_round_trips_the_words_back(self):
        engine = self._engine()
        ids = engine.encode("the answer is four")
        assert engine.decode(ids) == "the answer is four"

    def test_encode_of_nothing_still_returns_a_token(self):
        from c2c.serve.echo import EchoEngine

        ids = EchoEngine("probe").encode("   ")
        assert isinstance(ids, list) and len(ids) == 1 and 0 <= ids[0] < 4096

    def test_generate_mirrors_within_the_budget_and_honours_stops(self):
        engine = self._engine()
        ids = engine.encode("the answer is four the planet jupiter")
        assert engine.generate(ids, max_new_tokens=4) == "the answer is four"
        assert engine.generate(ids, max_new_tokens=99, stop=["planet"]) == "the answer is four the"

    def test_capture_is_the_same_cache_twice_over(self):
        engine = self._engine()
        ids = engine.encode("hello two please")
        one = [[round(x, 6) for x in row] for row in engine.capture(ids)[0].key]
        again = [[round(x, 6) for x in row] for row in self._engine().capture(ids)[0].key]
        assert one == again  # deterministic: same prompt, same cache
        assert one and len(one) == len(ids)  # one row per token
        assert all(0.0 <= x < 1.0 for row in one for x in row)

    def test_the_card_says_relay_in_words_a_reviewer_can_quote(self):
        engine = self._engine()
        spec = engine.spec()
        assert spec.family == "echo" and engine.report_context("probe") == 64
        assert "relay" in engine.DEGRADATION and "c2c-cache[train]" in engine.DEGRADATION


class TestTheHubTakesTheFall:
    def test_a_core_only_install_still_answers(self, monkeypatch):
        """The exact stranger's journey: no torch, no pins, one request."""
        _boom_engine(monkeypatch)
        from c2c.config import ServeConfig
        from c2c.serve.echo import EchoEngine
        from c2c.serve.registry import ModelHub

        hub = ModelHub(ServeConfig())
        target = hub.resolve("anything-at-all")
        assert target is not None and isinstance(target.receiver, EchoEngine)
        notes = [note for _mid, note, _ctx in hub.describe_models()]
        assert notes and all(notes) and any("relay" in n for n in notes)

    def test_an_operator_pinned_engine_fails_honestly_instead(self, monkeypatch):
        _boom_engine(monkeypatch)
        from c2c.config import ServeConfig
        from c2c.serve.registry import ModelHub

        hub = ModelHub(ServeConfig())
        hub.set_engine("vllm")  # the operator spoke: no silent relay
        assert hub.resolve("anything-at-all") is None


class TestServedOverTheWire:
    def test_the_front_answers_out_of_the_box(self, monkeypatch):
        _boom_engine(monkeypatch)
        from c2c.config import ServeConfig
        from c2c.serve.openai_proxy import create_server
        from c2c.serve.registry import ModelHub

        hub = ModelHub(ServeConfig())
        cfg = ServeConfig(host="127.0.0.1", port=0)
        httpd = create_server(cfg, hub=hub)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps(
                {
                    "model": "out-of-the-box",
                    "messages": [{"role": "user", "content": "the answer is four"}],
                    "max_tokens": 4,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.load(response)
            text = answer["choices"][0]["message"]["content"]
            assert text == "the answer is four"  # the mirror, held to its word
            assert answer["used_cache"] is False  # and it does not claim what did not run
            with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
                gallery = json.load(response)
            assert gallery["data"] and any(
                "relay" in m.get("description", "") for m in gallery["data"]
            )
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
