"""Manual pages of the C2C middleware — read them, then the paper.

The specification closes with an instruction to implementers: "read the
man pages, read the paper; make love, make it pass all tests." This
module locates the pages (shipped as package data under ``c2c/man/``,
plus the configuration page served from its single source of truth in
:mod:`c2c.config`) and renders them; the CLI shows them
(``c2c man <topic>``), and the Makefile installs them for the system
pager (``make man``).

Topics (sorted, as in the pager's index)::

    c2c        the command, overview, invocation, exit status
    config     the configuration tree and the two heads (FR-06)
    fuser      the cache fuser, the modules that learn
    engines    the seven whole engines and the adapters
    zoo        the zoo maintenance interface for trained fusers
"""

from __future__ import annotations

import os
from importlib import resources

__all__ = ["MAN_TOPICS", "page", "render"]

#: topic → file name, as shipped in ``c2c/man``
MAN_TOPICS: dict[str, str] = {
    "c2c": "c2c.1",
    "commands": "c2c.1",
    "config": "__constant__",  # served from c2c.config.MAN_C2C_CONFIG
    "fuser": "c2c-fuser.5",
    "engines": "c2c-engines.7",
    "zoo": "c2c-zoo.5",
    "serve": "c2c-serve.1",
    "proxy": "c2c-proxy.7",
    "mcp": "c2c-mcp.1",
    "a2a": "c2c-a2a.7",
}


def _man_dir():
    """Resolve the packaged man directory, on disk or in the wheel."""
    try:
        base = resources.files("c2c")
        candidate = base.joinpath("man")
        if candidate.is_dir():
            return candidate
    except (FileNotFoundError, NotADirectoryError, TypeError):
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(here, "man")
    return fallback if os.path.isdir(fallback) else None


def _normalise(topic: str) -> str:
    key = (topic or "").strip().lower()
    for noise in ("man ", "c2c-", "c2c.", "c2c "):
        if key.startswith(noise):
            key = key[len(noise) :]
    for noise in (".1", ".5", ".7"):
        if key.endswith(noise):
            key = key[: -len(noise)]
    key = key.strip()
    if key in ("", "intro", "overview"):
        key = "c2c"
    return key


def page(topic: str) -> str | None:
    """Return the page for *topic*, normalising common misspellings.

    The configuration page is served from its single source of truth,
    ``c2c.config.MAN_C2C_CONFIG``: constant and manual can not drift
    apart, which is the way of the union.
    """
    key = _normalise(topic)
    filename = MAN_TOPICS.get(key)
    if filename is None:
        return None
    if filename == "__constant__":
        from .. import config as _config

        return _config.MAN_C2C_CONFIG
    base = _man_dir()
    if base is None:
        return None
    try:
        return base.joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, AttributeError, OSError):
        pass
    try:
        with open(os.path.join(str(base), filename), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def render(topic: str | None) -> tuple[str, bool]:
    """Render one topic; returns ``(text, found)``."""
    text = page(topic or "c2c")
    if text is not None:
        return text, True
    index = "available topics: " + ", ".join(sorted(MAN_TOPICS))
    return f"No manual entry for {(topic or '').strip()!r}.\n{index}", False


if __name__ == "__main__":  # python -m c2c.utils.man [TOPIC]
    import sys

    body, ok = render(sys.argv[1] if len(sys.argv) > 1 else None)
    print(body)
    raise SystemExit(0 if ok else 1)
