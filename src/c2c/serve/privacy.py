"""Privacy mode (EX-5): the wire, sealed.

Two knobs of the module — the console carries the master switch
``--privacy`` (or ``C2C_PRIVACY``); these fields live on ``PrivacyConfig``:

``no_text_egress`` — :class:`NoTextFilter`
    egress filter: raw strings never leave the box. Anything that would
    travel as plain text is replaced by a digest of its content (SHA-256,
    truncated to 64 bits — one-way where the payload has entropy: a
    low-entropy value (a yes, a no, a PIN, a word in the dictionary) is
    recoverable by exhaustive search of its domain), so cache segments can
    be transmitted and reconciled without exposing the document.

``aes_gcm`` — :class:`WireCrypto`
    the frames that seal a cache segment for its journey: encrypted and
    authenticated, for the wire. In the present build the *filter* is wired
    to the front's access log (``--privacy`` digests prompt and answer
    there), and the *seal* is offered with the distribution, awaiting its
    relay: the proxy's caches live in one box, on the bus — the fusion is
    refused under ``--privacy`` for want of a sealed remote leg, not for
    want of the cipher. See the notes, and the manual.
    Uses AES-GCM through the optional ``cryptography`` peer (extra:
    ``c2c-cache[crypto]``). There is no silent fallback to a weaker
    cipher: if the peer is absent and sealing is requested, the call
    raises with an actionable hint.

Keys are 256-bit, nonces 96-bit, drawn from the operating system's
strongest source of randomness (``os.urandom``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

__all__ = ["WireCrypto", "NoTextFilter", "PrivacyGuard"]

_KEY_BYTES = 32  # 256-bit keys
_NONCE_BYTES = 12  # 96-bit nonces, as AES-GCM likes
_TAG_BYTES = 16


def _resolve_peer_member(container, name: str, *, where: str):
    """Pick a member out of the live peer, by its own name.

    The names of the cipher are the names, plain: ``AES`` is ``AES``,
    ``GCM`` is ``GCM``. A clear message when the peer is too old to know
    them: the peer must be upgraded, not the letters.
    """
    member = getattr(container, name, None)
    if member is None:
        msg = (
            f"the cryptography peer exposes no {where} member {name!r}; "
            f"peer too old? upgrade cryptography"
        )
        raise ModuleNotFoundError(msg)
    return member


class WireCrypto:
    """Seal and open cache segments with AES-GCM.

    A sealed frame on the wire is ``nonce ‖ ciphertext ‖ tag`` — the
    layout the AEAD convention of the algorithm prescribes. Under one key,
    seal no more than 2**32 frames (the bound of the mode, NIST SP 800-38D
    as published); re-key before the counter of the nonce runs out.
    """

    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray)) or len(key) != _KEY_BYTES:
            got = len(key) if isinstance(key, (bytes, bytearray)) else type(key).__name__
            msg = f"the privacy key must be exactly {_KEY_BYTES} bytes (256 bits), got {got}"
            raise ValueError(msg)
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise ModuleNotFoundError(
                "AES-GCM on the wire needs the cryptography peer (pip install 'c2c-cache[crypto]')"
            ) from exc
        self._Cipher = Cipher
        self._AES = _resolve_peer_member(algorithms, "AES", where="algorithms")
        self._GCM = _resolve_peer_member(modes, "GCM", where="modes")
        self._key = bytes(key)

    @staticmethod
    def generate_key() -> bytes:
        return os.urandom(_KEY_BYTES)

    def _cipher(self, nonce: bytes):
        return self._Cipher(self._AES(self._key), mode=self._GCM(nonce))

    def seal(self, message: bytes, *, associated_data: bytes | None = None) -> bytes:
        """Encrypt-and-authenticate; the frame is nonce ‖ ciphertext ‖ tag."""
        if not isinstance(message, (bytes, bytearray)):
            msg = f"can only seal bytes, not {type(message).__name__}"
            raise TypeError(msg)
        nonce = os.urandom(_NONCE_BYTES)
        encryptor = self._cipher(nonce).encryptor()
        encryptor.authenticate_additional_data(associated_data or b"")
        body = encryptor.update(bytes(message)) + encryptor.finalize()
        return nonce + body + encryptor.tag

    def open(self, frame: bytes, *, associated_data: bytes | None = None) -> bytes:
        """Open-and-verify a frame produced by :meth:`seal`.

        Any tampering — of the nonce, the body or the tag — raises; an
        incomplete frame raises too. Nothing is returned on success but
        the plaintext.
        """
        if not isinstance(frame, (bytes, bytearray)):
            msg = f"can only open bytes, not {type(frame).__name__}"
            raise TypeError(msg)
        # GCM permits an empty ciphertext: the frame of a body of nothing
        # is nonce ‖ tag — 28 bytes, whole and entire, legitimate.
        if len(frame) < _NONCE_BYTES + _TAG_BYTES:
            msg = "frame too short to be a sealed envelope"
            raise ValueError(msg)
        nonce = bytes(frame[:_NONCE_BYTES])
        body = bytes(frame[_NONCE_BYTES:])
        cipher_text, tag = body[:-_TAG_BYTES], body[-_TAG_BYTES:]
        decryptor = self._cipher(nonce).decryptor()
        decryptor.authenticate_additional_data(associated_data or b"")
        plain = decryptor.update(cipher_text)
        from cryptography.exceptions import InvalidTag  # the peer, imported

        try:
            return plain + decryptor.finalize_with_tag(tag)
        except InvalidTag as exc:
            msg = (
                "frame does not authenticate: tampered, or wrong key, or wrong "
                f"associated data (tag {tag.hex()[:8]}…)"
            )
            raise ValueError(msg) from exc


class NoTextFilter:
    """Replace raw strings with digests so that no text egresses the box.

    The filter walks any JSON-ready structure (dicts, lists, tuples,
    scalars) and replaces every string it considers a text payload with
    a tag of the form ``sha256:<first-16-hex>``. Numbers and bools pass
    through; so do bytes (they are not JSON-ready — base64 them into
    strings before the walk, if they must egress). A whitelist of keys
    may be supplied for fields that are protocol metadata (identifiers,
    modes, urls) and are kept verbatim.
    """

    def __init__(self, *, keep_keys: set[str] | None = None):
        self.keep_keys = set(keep_keys or ())

    @staticmethod
    def digest(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def filter(self, obj):
        """Return a copy of *obj* with textual payloads replaced by digests."""
        if isinstance(obj, dict):
            return {
                key: (
                    value
                    if (isinstance(key, str) and key in self.keep_keys)
                    else self.filter(value)
                )
                for key, value in obj.items()
            }
        if isinstance(obj, tuple):
            return tuple(self.filter(item) for item in obj)
        if isinstance(obj, list):
            return [self.filter(item) for item in obj]
        if isinstance(obj, str):
            return self.digest(obj)
        return obj


@dataclass(frozen=True)
class PrivacyGuard:
    """The privacy posture of one deployment. The master switch
    (``enabled``) defaults to off; the two knobs (``aes_gcm``,
    ``no_text_egress``) default to on — a woken guard seals and scrubs
    unless told otherwise.

    When sealing is on, the guard carries the key: generated once, at
    construction, and readable on :attr:`key` — the peer opens frames
    with ``WireCrypto(guard.key).open(frame)``. Pass a key of your own
    to share it out of band.
    """

    enabled: bool = False
    aes_gcm: bool = True
    no_text_egress: bool = True
    key: bytes | None = None

    def __post_init__(self):
        if self.enabled and self.aes_gcm and self.key is None:
            object.__setattr__(self, "key", WireCrypto.generate_key())

    def guard(self) -> bool:
        """Report whether privacy is on, before anything else happens."""
        return self.enabled

    def seal(self, payload: bytes, key: bytes | None = None) -> bytes:
        if not self.enabled or not self.aes_gcm:
            return payload
        chosen = key if key is not None else self.key
        if chosen is None:  # belt and braces: an enabled guard always has one
            chosen = WireCrypto.generate_key()
            object.__setattr__(self, "key", chosen)
        return WireCrypto(chosen).seal(payload)

    def scrub(self, obj, *, keep_keys: set[str] | None = None):
        if not self.enabled or not self.no_text_egress:
            return obj
        return NoTextFilter(keep_keys=keep_keys).filter(obj)
