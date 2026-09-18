#!/usr/bin/env python3
"""api lint: catch dunder-style corruption in stdlib references.

Design note: the *bad* spellings are never hand-written here — they are
derived programmatically from live, self-checked *good* names (`__`→`_`,
`__`→`` around the underscores). This makes it impossible for the linter
itself to rot into the very corruption it hunts. Non-dunder stdlib names
(asdict, is_dataclass, join, loads…) are validated by tools/import_smoke.py,
which imports every module and exercises the attributes for real.

Run: python tools/api_lint.py src tests tools
"""

import re
import sys
from importlib import import_module
from pathlib import Path

# Good, live, dotted names. Verified at startup against the runtime.
_GOOD_DOTTED: list[str] = [
    "dataclasses.asdict",
    "dataclasses.is_dataclass",
    "dataclasses.FrozenInstanceError",
    "os.path.join",
    "os.path.expanduser",
    "os.path.join",
    "json.load",
    "json.dump",
    "time.monotonic",
    "shutil.get_terminal_size",
    "pathlib.Path.read_text",
    "pathlib.Path.write_text",
    "importlib.import_module",
    "str.lstrip",
    "str.rstrip",
    "str.rsplit",
    "builtins.ModuleNotFoundError",
    "io.IOBase.isatty",
    "object.__init__",
]


def _corruptions(name: str) -> list[str]:
    """Derive plausible decayed spellings from a good name (never hand-typed)."""
    out: set[str] = set()
    for cand in (name.replace("__", "_"), name.replace("__", "")):
        if cand != name and "_" in cand:
            out.add(cand)
    return sorted(out)


def _resolve(dotted: str) -> bool:
    """Import the longest module prefix (or builtins class), then walk attributes."""
    parts = dotted.split(".")
    mod = None
    idx = 0
    for i in range(len(parts), 0, -1):
        head = ".".join(parts[:i])
        try:
            mod = import_module(head)
        except ModuleNotFoundError:
            continue
        idx = i
        break
    if mod is None:
        try:
            base = import_module("builtins")
        except ModuleNotFoundError:
            return False
        if hasattr(base, parts[0]):
            mod = getattr(base, parts[0])
            idx = 1
        else:
            return False
    for attr in parts[idx:]:
        mod = getattr(mod, attr, None)
        if mod is None:
            return False
    return True


def _selfcheck() -> int:
    failures = [d for d in _GOOD_DOTTED if not _resolve(d)]
    if failures:
        for d in failures:
            print(f"api lint SELF-CHECK FAILED: reference {d!r} does not resolve",
                  file=sys.stderr)
        return 2
    return 0


def _compile_rules() -> list[tuple[re.Pattern, str]]:
    rules: list[tuple[re.Pattern, str]] = []
    for dotted in _GOOD_DOTTED:
        attr = dotted.rsplit(".", maxsplit=1)[-1]
        if not attr.startswith("__"):
            continue                      # only surrounding-underscore dunders decay this way
        for bad in _corruptions(attr):
            rules.append((re.compile(rf"\b{re.escape(bad)}\b"), dotted))
    return rules


def check_file(path: Path, rules: list[tuple[re.Pattern, str]]) -> list[str]:
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for pattern, good in rules:
            if pattern.search(line):
                problems.append(f"{path}:{lineno}: decayed name → write {good!r}")
    return problems


def main(argv: list[str]) -> int:
    code = _selfcheck()
    if code:
        return code
    rules = _compile_rules()
    roots = [Path(a) for a in (argv or ["src", "tests", "tools"])]
    files = [p for r in roots if r.exists() for p in (r.rglob("**/*.py") if r.is_dir() else [r])]
    self_names = {"api_lint.py", "dunder_lint.py", "import_smoke.py"}
    bad = 0
    for f in sorted(files):
        if f.name in self_names:
            continue
        for prob in check_file(f, rules):
            print(prob)
            bad += 1
    if bad:
        print(f"api lint: {bad} problem(s) found", file=sys.stderr)
        return 1
    print(f"api lint: {len(files)} file(s) clean ({len(rules)} derived rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
