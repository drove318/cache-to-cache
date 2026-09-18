"""``c2c doctor`` — a self-check for the installation, like the stdlib one.

Verifies the environment (Python, backends, device), the registry (engine
adapters discoverable through the ``c2c.engines`` entry-point group), the
configuration (parses, values in range), and runs a miniature end-to-end
fusion — two synthetic caches in, one fused cache out, effective ranks
measured — so a user learns, before filing any bug report, whether their
installation is sound.

Exit status: 0 when the checks pass, or when only warnings are present
(all advisory). With ``--strict``, a failed check exits 1. The report
renders each check, its status, and a hint when one is due.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Callable

from ..config import C2CConfig, from_env
from ..utils.console import Table, style

__all__ = ["Check", "run_all", "format_report"]

STATUS_OK, STATUS_SKIP, STATUS_WARN, STATUS_FAIL = "ok", "skip", "warn", "fail"
_MARKS = {
    STATUS_OK: ("✓", "green"),
    STATUS_SKIP: ("○", "grey"),
    STATUS_WARN: ("!", "yellow"),
    STATUS_FAIL: ("✗", "red"),
}


@dataclass(frozen=True)
class Check:
    """One check: a name, a verdict, details, and a hint when one is due."""

    name: str
    status: str
    detail: str = ""
    hint: str = ""

    def render(self, *, color: bool = True) -> str:
        mark, colour = _MARKS[self.status]
        m = style(mark, colour, "bold", color_on=color)
        line = f"  {m} {style(self.name, 'bold', color_on=color)}"
        if self.detail:
            line += f" — {self.detail}"
        if self.hint and self.status in (STATUS_WARN, STATUS_FAIL):
            line += "\n      " + style(f"hint: {self.hint}", "grey", color_on=color)
        return line


# ---------------------------------------------------------------------------
# individual checks, grouped by sections
# ---------------------------------------------------------------------------

def check_python(*, minimum: tuple[int, int] = (3, 10)) -> Check:
    v = sys.version_info[:2]
    if v < minimum:
        return Check("python", STATUS_FAIL, platform.python_version(),
                     hint=f"this library needs Python ≥ {'.'.join(map(str, minimum))}")
    return Check("python", STATUS_OK, platform.python_version())


def check_backends() -> list[Check]:
    checks: list[Check] = []
    try:
        import numpy
        checks.append(Check("numpy", STATUS_OK, numpy.__version__))
    except ModuleNotFoundError as exc:
        checks.append(Check("numpy", STATUS_FAIL, str(exc),
                           hint="pip install 'c2c-cache[train]'"))
    try:
        import torch
        dev = "cpu"
        if hasattr(torch.backends, "mps") and getattr(torch.backends, "mps", None) \
                and torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.device_count() if callable(torch.cuda.device_count) else '?'}"
        checks.append(Check("torch", STATUS_OK, f"{torch.__version__} ({dev})"))
    except ModuleNotFoundError as exc:
        checks.append(Check("torch", STATUS_FAIL, str(exc),
                           hint="pip install 'c2c-cache[train]'"))
    return checks


def check_engines() -> list[Check]:
    """Enumerate the adapters of the ``c2c.engines`` group."""
    from ..integrations.registry import engines
    checks: list[Check] = []
    known = engines.registered_names()
    for name in sorted(known):
        try:
            engines.load(name, model_id=f"probe-{name}")
            checks.append(Check(f"engine:{name}", STATUS_OK, "registered"))
        except (ModuleNotFoundError, ValueError, RuntimeError) as exc:
            # AdapterNotSupported is a RuntimeError; the doctor reports, never crashes
            checks.append(Check(f"engine:{name}", STATUS_WARN, str(exc),
                               hint=f"pip install the '{name}' extra, or use --engine reference"))
    if not known:
        checks.append(Check("engine", STATUS_WARN, "no adapters found",
                           hint="is the distribution installed? pip install c2c-cache"))
    return checks


def check_configuration(cfg: C2CConfig) -> list[Check]:
    checks: list[Check] = []
    try:
        from_env(cfg)                       # environment must not blow up parsing
        checks.append(Check("configuration", STATUS_OK, f"seed={cfg.seed}"))
    except (ValueError, TypeError) as exc:
        checks.append(Check("configuration", STATUS_FAIL, str(exc),
                           hint="unset the offending C2C_* variable or fix config.json"))
    for section, tau in (("gate", (cfg.gate.tau_max, cfg.gate.tau_min)),
                        ("blend", (cfg.blend.fraction,))):
        if any(not 0.0 <= float(x) <= 100.0 for x in tau):
            checks.append(Check(f"config:{section}", STATUS_WARN, f"out-of-range values {tau}",
                               hint="fractions in [0,1] (or [0,100]); temperatures in (0,1]"))
        else:
            checks.append(Check(f"config:{section}", STATUS_OK, str(tau)))
    return checks


def check_caches(*, layers: int = 4, hidden: int = 8, heads: int = 2,
                 tokens: int = 6, seed: int = 42) -> list[Check]:
    """Fuse two synthetic caches and measure the ranks — a miniature of the
    full pipeline, safe to run on any machine (no models are harmed)."""
    checks: list[Check] = []
    try:
        import torch
    except ModuleNotFoundError as exc:
        return [Check("caches", STATUS_SKIP, str(exc), hint="pip install 'c2c-cache[train]'")]
    from ..types import LayerGeometry, LayeredCache, LayerSlice
    from ..fuser.core import Fuser
    from ..align.layers import terminal_mapping
    from ..diagnostics.rank import rank_report

    torch.manual_seed(seed)
    r = LayerGeometry(layers=layers, hidden_size=hidden, num_heads=heads, name="doctor-r")
    s = LayerGeometry(layers=layers - 1, hidden_size=hidden, num_heads=heads, name="doctor-s")
    gen = torch.Generator().manual_seed(seed)
    def make(geo):
        return LayeredCache([
            LayerSlice(torch.randn(tokens, geo.num_key_value_heads, geo.head_size, generator=gen),
                       torch.randn(tokens, geo.num_key_value_heads, geo.head_size, generator=gen))
            for _ in range(geo.layers)])
    rc, sc = make(r), make(s)
    try:
        fuser = Fuser(r, s, terminal_mapping(r.layers, s.layers))
        fuser.eval()
        with torch.no_grad():
            fused = fuser(rc, sc)
        ok = len(fused) == r.layers and fused.num_tokens == tokens
        checks.append(Check("caches:fuse", STATUS_OK if ok else STATUS_FAIL,
                           f"{layers} layers × {tokens} tokens through Eq. (3)"))
        rep = rank_report(rc, fused)
        grow = rep.key["after"] >= rep.key["before"] - 1e-3 and \
            rep.value["after"] >= rep.value["before"] - 1e-3
        checks.append(Check(
            "caches:rank", STATUS_OK if grow else STATUS_WARN,
            f"K {rep.key['before']:0.0f}→{rep.key['after']:0.0f} "
            f"V {rep.value['before']:0.0f}→{rep.value['after']:0.0f}",
            hint="fusion must not destroy the receiver's information (FR-03); "
                 "a drop may indicate uninitialised fuser weights (normal at "
                 "inception: the gate is closed)"))
    except Exception as exc:                              # noqa: the doctor reports, never rethrows
        checks.append(Check("caches:fuse", STATUS_FAIL, f"{type(exc).__name__}: {exc}",
                           hint="re-run with C2C_DOCTOR_VERBOSE=1 and file a bug report"))
    return checks


def check_zoo(cfg: C2CConfig) -> Check:
    root = cfg.root or os.path.expanduser("~/.cache/c2c")
    zoo = os.path.join(root, "zoo")
    if os.path.isdir(zoo):
        pairs = [d for d in os.listdir(zoo) if os.path.isdir(os.path.join(zoo, d))]
        return Check("zoo", STATUS_OK, f"{len(pairs)} published pair(s) in {zoo}")
    return Check("zoo", STATUS_SKIP, f"{zoo} does not exist yet",
                  hint="c2c zoo list / c2c train creates one")


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def run_all(cfg: C2CConfig | None = None, *, verbose: bool = True) -> list[Check]:
    """Run every check, in sections, and return the collected verdicts."""
    cfg = cfg or C2CConfig()
    checks: list[Check] = [check_python()]
    checks += check_backends()
    checks += check_configuration(cfg)
    checks += check_engines()
    checks += check_caches()
    checks.append(check_zoo(cfg))
    return checks


def format_report(checks: list[Check], *, title: str = "c2c doctor",
                  color: bool | None = None) -> str:
    """Render the report as a table of verdicts, grouped and annotated."""
    from .. import __version__ as version
    from ..utils.console import supports_color
    colour_on = supports_color() if color is None else color
    out = [style(f"{title} — c2c-cache {version}", "bold", "cyan", color_on=colour_on), ""]
    for ch in checks:
        out.append(ch.render(color=colour_on))
    fails = sum(1 for c in checks if c.status == STATUS_FAIL)
    warns = sum(1 for c in checks if c.status == STATUS_WARN)
    skips = sum(1 for c in checks if c.status == STATUS_SKIP)
    verdict = "all systems go" if fails == 0 else f"{fails} failure(s)"
    tail = f", {warns} warning(s), {skips} skipped" if (warns or skips) else ""
    out += ["", style(f"  {verdict}{tail}",
                     "green" if fails == 0 else "red", "bold", color_on=colour_on)]
    return "\n".join(out)


if __name__ == "__main__":            # python -m c2c.diagnostics.doctor
    print(format_report(run_all()))
