#!/usr/bin/env python3
"""dunder lint: flag Python definitions whose dunder names are mis-spelled.

Pattern-matching detector (no enumerated broken strings — that is how the
first version of this linter was itself broken). Rules:

* A definition whose underscore-stripped name hits a reserved dunder core
  but is not the exact `__core__` spelling (and is not a plain private
  helper with symmetric single underscores) is reported.
* A string compared with ``==`` against a single-underscore dunder-like
  literal (e.g. `"__main__"`, `"__name__"`) is reported.

Run: python tools/dunder_lint.py src tests tools
"""

import ast
import re
import sys
from pathlib import Path

# Reserved dunder cores, spelled as plain identifiers — never as dunders.
_REAL = {
    "init", "new", "del", "repr", "str", "bytes", "hash", "bool", "len",
    "iter", "next", "enter", "exit", "call", "getitem", "setitem", "delitem",
    "contains", "sizeof", "lt", "le", "eq", "ne", "gt", "ge", "add", "sub",
    "mul", "matmul", "truediv", "floordiv", "mod", "pos", "neg", "abs",
    "round", "invert", "or", "and", "xor", "lshift", "rshift", "await",
    "aiter", "anext", "reduce", "getstate", "setstate", "getnewargs",
    "post_init", "class_getitem", "dict", "weakref", "version", "all",
    "name", "file", "builtins", "spec", "package", "debug", "wrapped",
}

# single-underscore variants only on the trailing side: `enter__`, `init__`
_TRAIL_SINGLE = re.compile(r"^[a-z0-9][a-z0-9_]*_{1,}$")


def _looks_broken(name: str) -> tuple[bool, str]:
    """Return (flagged, suggested-correct).

    A real dunder carries exactly two underscores on each side; a private
    helper exactly one on each. Asymmetric variants around a reserved core
    — `__post_init__`, `_init__`, `enter__` — are reported.
    """
    core = name.strip("_")
    if core not in _REAL:
        return False, ""
    want = f"__{core}__"
    if name == want:
        return False, ""
    lead = len(name) - len(name.lstrip("_"))
    trail = len(name) - len(name.rstrip("_"))
    if (lead, trail) in ((0, 0), (1, 1), (2, 2)):
        return False, ""      # plain name, private helper, or a real dunder alias
    return True, want


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flagged, want = _looks_broken(node.name)
            if flagged:
                problems.append(
                    f"{path}:{node.lineno}: definition {node.name!r} is a broken"
                    f" dunder (write {want!r})"
                )
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
            sides = [node.left, *node.comparators]
            for comp in sides:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    v = comp.value
                    if _TRAIL_SINGLE.match(v) or v.startswith("_") and not v.startswith("__"):
                        ok, want = _looks_broken(v)
                        if ok:
                            problems.append(
                                f"{path}:{node.lineno}: comparison against {v!r} is a"
                                f" broken dunder (write {want!r})"
                            )
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in (argv or ["src", "tests", "tools"])]
    files = [p for r in roots if r.exists() for p in (r.rglob("**/*.py") if r.is_dir() else [r])]
    bad = 0
    for f in sorted(files):
        if f.name == Path(sys.argv[0]).name:
            continue
        for prob in check_file(f):
            print(prob)
            bad += 1
    if bad:
        print(f"dunder lint: {bad} problem(s) found", file=sys.stderr)
        return 1
    print(f"dunder lint: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
