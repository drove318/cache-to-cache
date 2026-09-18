"""Tests for the privacy guards (FR-17, §5 Limitations 1).

The wire, watched: seals that seal, opens that open, and texts that
never leave the building. AES-GCM frames, tamper detection, no-text
egress, and the guard, off by default.
"""

from __future__ import annotations

import pytest

from c2c.config import PrivacyConfig, ServeConfig
from c2c.serve.privacy import NoTextFilter, PrivacyGuard, WireCrypto


class TestWireCrypto:
    """seal, open, tamper: the envelope, on the wire."""

    def setup_method(self):
        self.key = WireCrypto.generate_key()
        self.crypto = WireCrypto(self.key)

    def test_a_sealed_message_round_trips(self):
        secret = b"the quick brown fox jumps over the lazy dog"
        frame = self.crypto.seal(secret)
        assert isinstance(frame, bytes)
        assert frame != secret                                    # sealed, not scrambled
        assert secret not in frame                                 # the words, hidden
        assert self.crypto.open(frame) == secret                  # opened, whole

    def test_authenticated_with_the_associated_data(self):
        """AAD: the header, bound to the body, verified on opening."""
        body = b"payload"
        frame = self.crypto.seal(body, associated_data=b"header-v1")
        assert self.crypto.open(frame, associated_data=b"header-v1") == body
        with pytest.raises((ValueError, TypeError)):               # the wrong header…
            self.crypto.open(frame, associated_data=b"header-v2")   # …breaks the seal

    def test_tampering_is_broken_opened_as_usual(self):
        """One byte off the frame, and the open, fails."""
        frame = bytearray(self.crypto.seal(b"the message, the whole message"))
        frame[len(frame) // 2] ^= 0xFF                              # a bit, astray
        with pytest.raises(ValueError):                               # the tamper, detected
            self.crypto.open(bytes(frame))

    def test_the_nonce_is_fresh_every_time(self):
        """The same message, twice, in two frames: never the same bytes."""
        assert self.crypto.seal(b"twice") != self.crypto.seal(b"twice")   # nonce, fresh
        assert self.crypto.open(self.crypto.seal(b"twice")) == b"twice"   # both, openable

    def test_a_foreign_key_opens_nothing(self):
        other = WireCrypto(WireCrypto.generate_key())
        frame = self.crypto.seal(b"private")
        with pytest.raises(ValueError):
            other.open(frame)                                       # the wrong key, no entry

    def test_a_short_key_is_no_key(self):
        """A key of the wrong length, a clear rejection."""
        with pytest.raises((ValueError, TypeError)):
            WireCrypto(b"too-short")                                 # not 256 bit, not sealed


class TestNoTextFilter:
    """no-text egress: the digests, not the words."""

    def setup_method(self):
        self.filter = NoTextFilter(keep_keys={"role", "id"})

    def test_plain_texts_are_replaced_by_digests(self):
        """A string, in: a hash, out. The words, gone."""
        text = "the capital of france is paris"
        result = self.filter.filter(text)
        assert result != text                                       # the text, filtered
        assert result.startswith("sha256:")                         # the digest, prefixed
        assert len(result) == len("sha256:") + 16                    # 16 hex, as documented

    def test_keys_are_kept_as_is(self):
        """The keys on the wire: role and id, verbatim."""
        message = {"role": "assistant", "id": "chatcm-1",
                 "content": "the words of the wind"}
        kept = self.filter.filter(message)
        assert kept["role"] == "assistant"                            # the role, kept
        assert kept["id"] == "chatcm-1"                                # the id, kept
        assert kept["content"].startswith("sha256:")                  # the content, hashed

    def test_the_structure_of_the_message_survives(self):
        """Lists and tuples and objects: the shape, kept; the text, shed."""
        nested = {"choices": [{"text": "one"}, {"text": "two"}], "n": 2}
        result = self.filter.filter(nested)
        assert result["n"] == 2                                         # numbers, numbers
        assert isinstance(result["choices"], list) and len(result["choices"]) == 2
        assert result["choices"][0]["text"] != "one"                    # strings, hashed

    def test_the_digest_is_deterministic(self):
        """The same text, twice: the same digest, once."""
        assert NoTextFilter.digest("hello") == NoTextFilter.digest("hello")
        assert NoTextFilter.digest("hello") != NoTextFilter.digest("hell0")


class TestPrivacyGuard:
    """The guard, on duty: off by default, on command."""

    def test_the_guard_sleeps_by_default(self):
        """Privacy off, the wire plain: nothing sealed, nothing filtered."""
        guard = PrivacyGuard()                                         # the knobs, at rest
        assert guard.guard() is False                                  # default, dormant
        payload = b"the prompt, the whole prompt"
        assert guard.seal(payload) == payload                            # the bytes, unchanged

    def test_the_guard_wakes_on_command(self):
        """PrivacyGuard(enabled=True): the seal, on the wire."""
        guard = PrivacyGuard(enabled=True)                               # the knob, turned on
        assert guard.guard() is True                                     # awake, at the post
        payload = b"the prompt, the whole prompt, nothing but the prompt"
        frame = guard.seal(payload)
        assert isinstance(frame, bytes) and frame != payload             # sealed, indeed
        assert payload not in frame                                       # the words, hidden

    def test_the_scrub_is_a_scrubbing(self):
        """scrub(obj, keep_keys=…): the protocol, kept; the prose, hashed."""
        guard = PrivacyGuard(enabled=True)
        event = {"event": "fuse", "prompt": "secret prompt", "tokens": 42}
        scrubbed = guard.scrub(event, keep_keys={"event"})
        assert scrubbed["event"] == "fuse"                                # the machine, kept
        assert scrubbed["tokens"] == 42                                    # the numbers, kept
        assert scrubbed["prompt"].startswith("sha256:")                   # the words, hashed


class TestTheWire:
    """Privacy, on the wire: nothing plain where nothing is sealed."""

    def test_nothing_plain_nothing_hidden(self):
        """On an off, all in the config: the switches, in the data class."""
        cfg = PrivacyConfig()
        assert cfg.enabled is False                                      # by default, off
        assert isinstance(cfg.aes_gcm, bool)                             # ciphers, declared
        assert isinstance(cfg.no_text_egress, bool)                       # egress, declared

    def test_the_served_wire_of_the_proxy(self):
        """Privacy on the front: digests on the access log, no plain words."""
        import io
        import json
        import sys
        import threading
        import urllib.request
        from c2c.serve.openai_proxy import create_server
        from c2c.serve.registry import ModelHub
        hub = ModelHub(ServeConfig(api_key="k"))
        hub.set_engine("reference")
        hub.register_model("receiver-mini")                              # one model, registered
        httpd = create_server(ServeConfig(host="127.0.0.1", port=0, api_key="k",
                                    privacy=True), hub=hub)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        buffer, real_stderr = io.StringIO(), sys.stderr
        try:
            sys.stderr = buffer
            body = json.dumps({"model": "receiver-mini",
                           "messages": [{"role": "user",
                                      "content": "the secret of the machines"}],
                           "max_tokens": 2}).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=body, method="POST",
                headers={"Content-Type": "application/json",
                       "Authorization": "Bearer k"})
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.status
        finally:
            sys.stderr = real_stderr
        httpd.shutdown()
        thread.join(timeout=5)
        logged = buffer.getvalue()
        assert status == 200                                             # the wire, live
        assert "sha256:" in logged                                       # digests, logged
        assert "secret of the machines" not in logged                    # the words, within


class TestTheSealAtThePipe:
    """The privacy flag on the served: the sharer's cache, refused before the attempt."""

    def test_a_sealed_front_answers_alone(self):
        from c2c.fuser import Fuser
        from c2c.serve.openai_proxy import ChatPipeline
        from c2c.serve.registry import ModelHub
        hub = ModelHub(ServeConfig(api_key=None, privacy=True))
        hub.set_engine("reference")
        hub.register_model("receiver-mini")
        hub.register_model("sharer-mini")
        recv = hub._resolve_side("receiver-mini")
        shar = hub._resolve_side("sharer-mini")
        g_r, g_s = recv.spec().geometry, shar.spec().geometry
        fuser = Fuser(g_r, g_s, list(range(min(g_r.layers, g_s.layers))))
        hub.register_pair(receiver="receiver-mini", sharer="sharer-mini", fuser=fuser)
        touched: list[str] = []
        real_capture = shar.capture

        def spying(ids):
            touched.append("captured")
            return real_capture(ids)

        shar.capture = spying
        pipeline = ChatPipeline(hub=hub)
        out = pipeline.complete(model="c2c/receiver-mini←sharer-mini",
                             prompt_text="what is two plus two", max_new_tokens=6,
                             temperature=0.0, tools=None, stop=None)
        assert out["used_cache"] is False                            # the seal, honoured
        assert touched == []                                          # the sharer, untouched
        assert out["answer"]                                            # the receiver, answering
