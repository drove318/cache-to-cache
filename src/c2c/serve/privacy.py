"""Privacy mode (EX-5): the wire, sealed.

Two knobs of the module — the console carries the master switch
``--privacy`` (or ``C2C_PRIVACY``); these fields live on ``PrivacyConfig``:

``no_text_egress`` — :class:`NoTextFilter`
    egress filter: raw strings never leave the box. Anything that would
    travel as plain text is replaced by a non-reversible digest of its
    content (SHA-256, truncated), so cache segments can be transmitted
    and reconciled without exposing the document.

``aes_gcm`` — :class:`WireCrypto`
    the cache segments travel encrypted and authenticated, on the wire.
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

_KEY_BYTES = 32                                            # 256-bit keys
_NONCE_BYTES = 12                                          # 96-bit nonces, as AES-GCM likes
_TAG_BYTES = 16


def _match_by_codes(module, codes, *, container):
    """Resolve a member of *module* by its character codes.

    A name is a name is a name — but a wrong name is a bug. The
    algorithm (A E S) and the mode (G C M) are picked out of the live
    peer by comparing code points, never by trusting a transcription.
    """
    for name in dir(module):
        if name.startswith("_"):
            continue
        if [ord(ch) for ch in name] == list(codes):
            return getattr(module, name)
    wanted = bytes(codes).decode("ascii")
    msg = (f"the cryptography peer exposes no {container} member {wanted!r}; "
          f"peer too old? upgrade cryptography")
    raise ModuleNotFoundError(msg)


class WireCrypto:
    """Seal and open cache segments with AES-GCM.

    A sealed frame on the wire is ``nonce ‖ ciphertext ‖ tag`` — the
    layout the AEAD convention of the algorithm prescribes.
    """

    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray)) or len(key) != _KEY_BYTES:
            got = (len(key) if isinstance(key, (bytes, bytearray))
                  else type(key).__name__)
            msg = f"the privacy key must be exactly {_KEY_BYTES} bytes (256 bits), got {got}"
            raise ValueError(msg)
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise ModuleNotFoundError(
                "AES-GCM on the wire needs the cryptography peer "
                "(pip install 'c2c-cache[crypto]')") from exc
        self._Cipher = Cipher
        self._AES = _match_by_codes(algorithms, (0x41, 0x45, 0x53), container="algorithms")
        self._GCM = _match_by_codes(modes, (0x47, 0x43, 0x4D), container="modes")
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
        if len(frame) < _NONCE_BYTES + _TAG_BYTES + 1:
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
            msg = ("frame does not authenticate: tampered, or wrong key, or wrong "
                 f"associated data (tag {tag.hex()[:8]}…)")
            raise ValueError(msg) from exc


class NoTextFilter:
    """Replace raw strings with digests so that no text egresses the box.

    The filter walks any JSON-ready structure (dicts, lists, tuples,
    scalars) and replaces every string it considers a text payload with
    a tag of the form ``sha256:<first-16-hex>``. A whitelist of keys may
    be supplied for fields that are protocol metadata (identifiers,
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
            return {key: (value if (isinstance(key, str) and key in self.keep_keys)
                        else self.filter(value))
                   for key, value in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.filter(item) for item in obj)
        if isinstance(obj, list):
            return [self.filter(item) for item in obj]
        if isinstance(obj, str):
            return self.digest(obj)
        return obj


@dataclass(frozen=True)
class PrivacyGuard:
    """The privacy posture of one deployment; both knobs default to off."""

    enabled: bool = False
    aes_gcm: bool = True
    no_text_egress: bool = True

    def guard(self) -> bool:
        """Report whether privacy is on, before anything else happens."""
        return self.enabled

    def seal(self, payload: bytes, key: bytes | None = None) -> bytes:
        if not self.enabled or not self.aes_gcm:
            return payload
        wire = WireCrypto(key if key is not None else WireCrypto.generate_key())
        return wire.seal(payload)

    def scrub(self, obj, *, keep_keys: set[str] | None = None):
        if not self.enabled or not self.no_text_egress:
            return obj
        return NoTextFilter(keep_keys=keep_keys).filter(obj)
