"""Small console helpers: colour, tables, progress — stdlib only.

The CLI is the face of the library; it should be as beautiful to use as a
good man page and as polite as a well-behaved daemon: no colour when piped,
no exceptions when the user mistypes, and a helpful hint at the end of every
error message.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Sequence

__all__ = ["supports_color", "style", "Table", "Progress", "banner", "error_hint"]


def supports_color(stream=None) -> bool:
    """Figure out whether the output stream should be coloured."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or os.environ.get("C2C_NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


_RESET, _BOLD, _DIM = "0", "1", "2"
_COLORS = {
    "reset": _RESET, "bold": _BOLD, "dim": _DIM,
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36", "grey": "90",
}


def style(text: str, *styles: str, color_on: bool | None = None) -> str:
    """Style `text` with SGR sequences when the terminal allows."""
    on = supports_color() if color_on is None else color_on
    if not on or not styles:
        return text
    codes = ";".join(_COLORS[s] for s in styles if s in _COLORS)
    return f"\033[{codes}m{text}\033[0m" if codes else text


class Table:
    """A tiny, dependency-free table printer (make table, print table)."""

    def __init__(self, headers: Sequence[str], *, title: str | None = None):
        self.headers = list(headers)
        self.rows: list[list[str]] = []
        self.title = title

    def add_row(self, row: Sequence) -> None:
        self.rows.append([("" if v is None else str(v)) for v in row])

    def __str__(self):
        cols = len(self.headers)
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i in range(min(cols, len(row))):
                widths[i] = max(widths[i], len(row[i]))
        sep = "-+-".join("-" * (w + 2) for w in widths)
        lines: list[str] = []
        if self.title:
            lines.append(style(self.title, "bold"))
        lines.append(" | ".join(h.ljust(w) for h, w in zip(self.headers, widths)))
        lines.append(sep)
        for row in self.rows:
            cells = list(row) + [""] * (cols - len(row))
            lines.append(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
        return "\n".join(lines)


class Progress:
    """A progress bar for training and fusion sweeps (TTY-aware, no-op when piped)."""

    def __init__(self, total: int, *, label: str = "", width: int = 36, enabled: bool | None = None):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self._done = 0
        self._t0 = self._time()
        tty = supports_color(sys.stderr)
        self.enabled = tty if enabled is None else enabled

    @staticmethod
    def _time():
        import time
        return time.monotonic()

    def update(self, n: int = 1) -> None:
        self._done += n
        if not self.enabled:
            return
        frac = min(1.0, self._done / self.total)
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        el = self._time() - self._t0
        eta = (el / self._done) * (self.total - self._done) if self._done else 0.0
        sys.stderr.write(f"\r{self.label:<14} [{bar}] {frac:3.0%} {self._done}/{self.total} eta {eta:4.0f}s")
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def banner(name: str, version: str) -> str:
    """Render the opening banner of `c2c --version` style screens."""
    try:
        width = shutil.get_terminal_size(sys.stdout, fallback=(80, 24))[0]
    except (OSError, TypeError, ValueError):
        width = 80
    line = "─" * max(24, min(width - 2, 68))
    return (
        f"{style('┌' + line + '┐', 'cyan')}\n"
        f"  {style('C2C', 'bold', 'cyan')} {style(name, 'bold')} {style(f'v{version}', 'grey')}\n"
        f"  {style('cache-to-cache · the harness stays the master', 'grey', 'dim')}\n"
        f"{style('└' + line + '┘', 'cyan')}"
    )


def error_hint(message: str, *, hint: str | None = None) -> str:
    """Format an error message with a helpful hint underneath (PEP 20)."""
    out = [style(f"error: {message}", "red", "bold")]
    if hint:
        out.append(style(f"hint: {hint}", "grey"))
    return "\n".join(out)
