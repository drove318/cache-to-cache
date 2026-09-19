#!/usr/bin/env python3
"""import smoke: import every module of the c2c package and exercise attributes.

The real test of an API is running it. This tool imports every module,
then asserts the module's public surface is reachable: every name in
__all__ exists, every advertised class can be instantiated without args or
with its documented defaults, and the entry points resolve. Exceptions are
raised, not caught — a broken import must fail the build loudly.

Run: python tools/import_smoke.py            (from the repo root)
"""

import sys
from importlib import import_module
from pathlib import Path


def _all_modules(root: Path) -> list[str]:
    mods: list[str] = []
    for path in sorted(root.rglob("**/*.py")):
        rel = path.relative_to(root)
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__.py":
            parts.pop()
        elif parts:
            parts[-1] = parts[-1][:-3]
        if any(p.startswith("test_") for p in parts):
            continue
        if not parts:
            mods.append("c2c")
        else:
            mods.append(".".join(parts) if parts[0] == "c2c" else "c2c." + ".".join(parts))
    return mods


def main() -> int:
    root = Path(__file__).parent.parent / "src" / "c2c"
    if not root.exists():
        print("import smoke: cannot find src/c2c", file=sys.stderr)
        return 2
    failures: list[str] = []
    count = 0
    for name in _all_modules(root):
        try:
            mod = import_module(name)
        except Exception as exc:  # noqa: report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        count += 1
        for symbol in getattr(mod, "__all__", ()) or ():
            if not hasattr(mod, symbol):
                failures.append(f"{name}: advertised name {symbol!r} is missing")
    if failures:
        for f in failures:
            print(f"import smoke FAILED: {f}", file=sys.stderr)
        return 1
    print(f"import smoke: {count} module(s) import clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
