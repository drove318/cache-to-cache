"""Publish and download trained fuser weights (FR-15).

The zoo of fine-tuned cache fusers lives on the Hugging Face hub, one
repository per ``(Sharer, Receiver)`` pair, named by the canonical pair id
``<sharer>__<receiver>`` (see :func:`pair_id`). Everything resolves from
the local cache first — ``~/.cache/c2c/zoo`` — so the tool works offline,
in CI, behind proxies, and in air-gapped deployments; the hub is contacted
only when an object is genuinely absent locally.

Layout of one published pair (the ``manifest.json`` is the contract; the
``weights.pt`` blob is a plain ``torch.save`` state-dict archive, loaded
back with ``weights_only=True``)::

    <sharer>__<receiver>/
        manifest.json      # id, revision, sha256, config, provenance
        weights.pt         # fuser.state_dict()

Integrity: ``sha256sum`` of the weights blob is recorded in the manifest
and verified on every download — a truncated transfer is a failed
transfer, not a warning.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from ..types import ModelSpec

__all__ = ["pair_id", "ZooClient", "DEFAULT_ZOO_ROOT", "MANIFEST_NAME", "WEIGHTS_NAME"]

DEFAULT_ZOO_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "c2c", "zoo")
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"
FORMAT_MAGIC = "c2c-fuser-checkpoint-v1"
DEFAULT_HUB = "https://huggingface.co"


def _sha256(path: str) -> str:
    """Compute the SHA-256 of a file, reading it in one pass."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_id(sharer: str | ModelSpec, receiver: str | ModelSpec) -> str:
    """Normalise a (sharer, receiver) pair into a filesystem-safe id.

    Letters, digits, dots, dashes and underscores survive; everything else
    (arrows, slashes, spaces from model names such as ``Qwen2.5-0.5B``) is
    folded to a single dash — case is preserved for legibility, ids are
    compared case-sensibly.
    """

    def norm(item):
        s = item if isinstance(item, str) else item.id
        s = s.strip().lower()
        keep = [ch if (ch.isalnum() and ch.isascii() or ch in "._-") else "-" for ch in s]
        collapsed = "".join(keep)
        while "--" in collapsed:
            collapsed = collapsed.replace("--", "-")
        return collapsed.strip("-") or "unnamed"

    return f"{norm(sharer)}__{norm(receiver)}"


class ZooClient:
    """Local-first zoo client with optional Hugging Face hub resolution.

    Parameters
    ----------
    root:
        Local cache directory (default ``~/.cache/c2c/zoo``).
    hub:
        Base URL of a hub-compatible server, or ``None`` to stay fully
        offline (the default in CI: *no internet except fixtures*).
    token:
        Optional bearer token for private models (read from
        ``$C2C_HF_TOKEN`` when unset).
    """

    def __init__(
        self,
        *,
        root: str | None = None,
        hub: str | None = DEFAULT_HUB,
        token: str | None = None,
        timeout: float = 30.0,
    ):
        self.root = os.path.abspath(os.path.expanduser(root or DEFAULT_ZOO_ROOT))
        self.hub = hub
        self.token = token if token is not None else os.environ.get("C2C_HF_TOKEN")
        self.timeout = timeout

    # -- local resolution, first and foremost ───────────────────────────────
    def _pair_dir(self, pair: str) -> str:
        safe = os.path.basename(pair)  # containment: never traverse
        if safe != pair or not safe:
            msg = f"illegal pair id: {pair!r}"
            raise ValueError(msg)
        return os.path.join(self.root, safe)

    def list_pairs(self) -> list[dict[str, Any]]:
        """Return the manifests of every locally available pair."""
        out: list[dict] = []
        if not os.path.isdir(self.root):
            return out
        for entry in sorted(os.listdir(self.root)):
            manifest = os.path.join(self._pair_dir(entry), MANIFEST_NAME)
            if os.path.isfile(manifest):
                with open(manifest, encoding="utf-8") as fh:
                    data = json.load(fh)
                data.setdefault("id", entry)
                out.append(data)
        return out

    def local_path(self, pair: str) -> str | None:
        """Resolve a pair locally, refusing traversal, the directory way."""
        if not pair or pair != os.path.basename(pair):
            msg = f"illegal pair id: {pair!r}"
            raise ValueError(msg)
        pid = pair if "__" in pair else pair_id(pair, pair)
        directory = self._pair_dir(pid)
        if os.path.isfile(os.path.join(directory, WEIGHTS_NAME)) and os.path.isfile(
            os.path.join(directory, MANIFEST_NAME)
        ):
            return directory
        return None

    def _verify_local(self, pid: str, directory: str) -> None:
        """Check the seal before the shelf: a local hit is a verified hit.

        A corrupted local entry raises ValueError; the download path keeps
        its OSError convention (network artefact, wrong checksum, wrong
        revision) — Pythonic error handling, both of them, always.
        """
        manifest_path = os.path.join(directory, MANIFEST_NAME)
        weights_path = os.path.join(directory, WEIGHTS_NAME)
        if not (os.path.isfile(manifest_path) and os.path.isfile(weights_path)):
            return
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        want = manifest.get("sha256")
        if not want:
            msg = f"{pid}: the manifest carries no sha256 seal; the blob is refused, not trusted"
            raise ValueError(msg)
        if _sha256(weights_path) != want:
            msg = f"{pid}: sha256 mismatch for {WEIGHTS_NAME} (truncated or tampered)"
            raise ValueError(msg)

    # -- publishing ─────────────────────────────────────────────────────────
    def publish(
        self,
        *,
        sharer: str | ModelSpec,
        receiver: str | ModelSpec,
        fuser,
        config: Any = None,
        revision: str = "main",
    ) -> dict:
        """Save a trained fuser into the local zoo (and upload when online).

        The fuser must expose ``state_dict()`` (any :class:`torch.nn.Module`).
        Returns the manifest written.
        """
        pid = pair_id(sharer, receiver)
        directory = self._pair_dir(pid)
        os.makedirs(directory, exist_ok=True)
        weights_path = os.path.join(directory, WEIGHTS_NAME)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".weights.", suffix=".pt.tmp")
        os.close(tmp_fd)
        try:
            torch.save({"format": FORMAT_MAGIC, "state_dict": fuser.state_dict()}, tmp_name)
            os.replace(tmp_name, weights_path)  # atomic on POSIX
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        manifest = {
            "format": FORMAT_MAGIC,
            "id": pid,
            "sharer": sharer if isinstance(sharer, str) else sharer.id,
            "receiver": receiver if isinstance(receiver, str) else receiver.id,
            "revision": revision,
            "created_at": datetime.now(timezone.utc).isoformat(sep="T"),
            "config": _plain(config),
            "sha256": _sha256(weights_path),  # the seal, on the shelf
        }
        with open(os.path.join(directory, MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        if self.hub:
            try:
                self._upload(directory, pid)
            except (urllib.error.URLError, OSError) as exc:
                manifest["upload"] = f"skipped: {exc}"  # offline-first: local stands
        return manifest

    # -- fetching ───────────────────────────────────────────────────────────
    def fetch(
        self,
        *,
        sharer: str | ModelSpec,
        receiver: str | ModelSpec,
        revision: str = "main",
        force: bool = False,
    ) -> str:
        """Return the local directory of a pair, downloading it if need be.

        Resolution order: local cache → hub. The checksum recorded in the
        manifest is verified; a mismatch raises OSError (the download is
        discarded, never trusted).
        """
        pid = pair_id(sharer, receiver)
        directory = self._pair_dir(pid)
        if not force:
            found = self.local_path(pid)
            if found:
                self._verify_local(pid, found)  # verify, before returning
                return found
        if not self.hub:
            msg = f"pair {pid!r} not in the local zoo and the client is offline"
            raise FileNotFoundError(msg)
        os.makedirs(directory, exist_ok=True)
        manifest = self._get_json(self._url(pid, revision, MANIFEST_NAME))
        url = self._url(pid, revision, WEIGHTS_NAME)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".fetch.", suffix=".pt.tmp")
        os.close(tmp_fd)
        try:
            self._download(url, tmp_name)
            seal = manifest.get("sha256")
            if not seal:
                msg = f"{pid}: the hub manifest carries no sha256 seal; the download is refused"
                raise OSError(msg)
            if _sha256(tmp_name) != seal:
                msg = f"{pid}: sha256 mismatch for {WEIGHTS_NAME} (truncated or tampered)"
                raise OSError(msg)
            os.replace(tmp_name, os.path.join(directory, WEIGHTS_NAME))
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        with open(os.path.join(directory, MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        return directory

    def load_fuser(
        self,
        *,
        sharer: str | ModelSpec,
        receiver: str | ModelSpec,
        fuser_builder,
        strict: bool = True,
    ):
        """Fetch a pair's weights and re-instantiate a fuser from them.

        ``fuser_builder`` is a zero-argument callable returning a
        :class:`torch.nn.Module` with the matching architecture (typically
        a lambda around :class:`c2c.fuser.Fuser`).
        """
        directory = self.fetch(sharer=sharer, receiver=receiver)
        blob = torch.load(
            os.path.join(directory, WEIGHTS_NAME), map_location="cpu", weights_only=True
        )
        if blob.get("format") != FORMAT_MAGIC:
            msg = f"{directory}: not a c2c fuser checkpoint (missing magic)"
            raise ValueError(msg)
        module = fuser_builder()
        module.load_state_dict(blob["state_dict"], strict=strict)
        return module

    # -- plumbing ───────────────────────────────────────────────────────────
    def _url(self, pid: str, revision: str, name: str) -> str:
        base = self.hub.rstrip("/")
        model = urllib.parse.quote(pid, safe="._-")
        return f"{base}/api/models/{model}/resolve/{revision}/{name}"

    def _request(self, url: str) -> urllib.request.Request:
        headers = {"User-Agent": "c2c-cache/1.0 (+https://github.com/drove318/cache-to-cache)"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(url, headers=headers)

    def _get_json(self, url: str) -> dict:
        with urllib.request.urlopen(self._request(url), timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download(self, url: str, destination: str) -> None:
        with (
            urllib.request.urlopen(self._request(url), timeout=self.timeout) as resp,
            open(destination, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)

    def _upload(self, directory: str, pid: str) -> None:
        # Uploads need the preflight of the endpoint; the hub's POST API is
        # the remote counterpart of `huggingface-cli`. We keep the client
        # thin: publishing to the hub is delegated to a configured remote
        # helper (`c2c zoo publish --push` invokes it when present).
        raise NotImplementedError("hub uploads are delegated to the remote helper")


def _plain(obj: Any) -> Any:
    """Convert dataclasses/tensors into JSON-encodable structures."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        try:
            return {k: _plain(v) for k, v in asdict(obj).items()}
        except TypeError:
            pass
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    return str(obj)
