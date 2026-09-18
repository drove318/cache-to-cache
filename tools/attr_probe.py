#!/usr/bin/env python3
"""attr probe: verify every static reference against importable reality.

Parses all package sources (AST, no execution) and, for every reference of
the shapes

    from <mod> import <name>[, <name>…]
    <Mod>.<attr>…            (attribute chains rooted at an imported module)

it checks the reference against the live module system with importlib and
getattr. References to modules that cannot be imported at all (optional
heavy engines: torch, vllm, transformers…) are skipped with a count, never
silently ignored. A reference that resolves to a missing attribute is an
error: exactly the failure mode produced by underscore drift in identifiers
like `time.monotonic` for `time.monotonic`.

Run: python tools/attr_probe.py            (from the repo root)
Exit codes: 0 clean · 1 broken references · 2 usage/environment problem.
"""

import ast
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

_IMPORTABLE_CACHE: dict[str, object | None] = {}


def _try_import(dotted: str):
    if dotted in _IMPORTABLE_CACHE:
        return _IMPORTABLE_CACHE[dotted]
    mod = None
    try:
        mod = import_module(dotted)
    except (ModuleNotFoundError, ImportError, ValueError):
        mod = None
    _IMPORTABLE_CACHE[dotted] = mod
    return mod


def _resolve_chain(mod, attrs: list[str]):
    cur = mod
    walked: list[str] = []
    for a in attrs:
        walked.append(a)
        if cur is None:
            return None, ".".join(walked)
        if not hasattr(cur, a):
            return None, ".".join(walked)
        cur = getattr(cur, a)
    return cur, None


def _module_names_of(tree: ast.Module) -> dict[str, str]:
    """Map local alias → dotted module path for `import x`, `import x.y as z`."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                target = al.asname or al.name
                aliases[target] = al.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                for al in node.names:
                    if al.name == "*":
                        continue
                    aliases[al.asname or f"{al.name}"] = f"{node.module}.{al.name}"
    return aliases


def check_file(path: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    skipped = 0
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"], 0
    aliases = _module_names_of(tree)

    # 1) `from mod import name` statements, checked name by name.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mod = _try_import(node.module)
            if mod is None:
                skipped += 1
                continue
            for al in node.names:
                if al.name == "*":
                    continue
                if not hasattr(mod, al.name):
                    errors.append(
                        f"{path}:{node.lineno}: from {node.module} import {al.name}"
                        f" → {node.module} has no attribute {al.name!r}"
                    )
    # 2) attribute chains rooted at an imported module or a builtins class.
    import builtins as _b
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue
        root = cur.id
        chain.reverse()
        mod = None
        if root in aliases:
            dotted = aliases[root]
            # split longest importable prefix
            parts = dotted.split(".")
            for i in range(len(parts), 0, -1):
                mod = _try_import(".".join(parts[:i]))
                if mod is not None:
                    chain = parts[i:] + chain
                    break
            if mod is None:
                skipped += 1
                continue
        elif hasattr(_b, root) and not getattr(_b, root, None).__class__.__name__ == "module":
            try:
                mod = getattr(_b, root)
            except AttributeError:
                continue
        else:
            continue
        _, missing = _resolve_chain(mod, chain)
        if missing:
            errors.append(f"{path}:{node.lineno}: {root}.{'.'.join(chain)} unresolved ({missing!r})")
    return errors, skipped


def main() -> int:
    if not SRC.exists():
        print(f"attr probe: cannot find {SRC}", file=sys.stderr)
        return 2
    all_errors: list[str] = []
    skipped_total = 0
    count = 0
    for f in sorted(SRC.rglob("**/*.py")):
        errs, skipped = check_file(f)
        all_errors.extend(errs)
        skipped_total += skipped
        count += 1
    for e in all_errors:
        print(f"attr probe: {e}", file=sys.stderr)
    if all_errors:
        print(f"attr probe: {len(all_errors)} broken reference(s), {count} file(s) scanned",
              file=sys.stderr)
        return 1
    print(f"attr probe: {count} file(s) clean ({skipped_total} optional-module refs skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
