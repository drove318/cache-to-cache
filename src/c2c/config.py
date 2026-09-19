"""C2C configuration — dataclasses, environment, and the `man c2c.config` text.

All defaults reproduce the published recipe (paper App. A.3.5; spec FR-14).
Environment variables override defaults with the ``C2C_`` prefix; keys map
directly onto field names of the dataclasses below, uppercased (e.g.
``C2C_SEED``, ``C2C_LR``, ``C2C_GATE_TAU_MIN``).

Two heads, one word (spec FR-06): see :data:`MAN_C2C_CONFIG` — read the man
pages, or run ``c2c man config``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

__all__ = [
    "C2CConfig",
    "FuserConfig",
    "GateConfig",
    "BlendConfig",
    "AlignConfig",
    "TrainRecipe",
    "ServeConfig",
    "PrivacyConfig",
    "load_config",
    "from_env",
    "MAN_C2C_CONFIG",
    "DEFAULT_SEED",
]

DEFAULT_SEED = 42  # paper: seed 42, determinism first (FR-14, §5)


def _env_cast(value: str, typ: type) -> Any:
    """Cast an environment string to the field's declared type (small helper)."""
    origin = getattr(typ, "__origin__", typ)
    if origin in (bool,):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if origin is int:
        return int(value)
    if origin is float:
        return float(value)
    return value


@dataclass(frozen=True)
class FuserConfig:
    """Cache fuser — projection, dynamic weighting, feature fusion (Fig. 5).

    One fuser clone per mapped layer pair; the fuser is the only trainable
    component — both LLMs stay frozen (paper §3.3.4, FR-02).
    """

    variant: str = "simple"  # simple | c2c-c (C2C-C, App. A.1.3)
    activation: str = "gelu"  # between projection and feature-fusion
    latent_size: int | None = None  # override the projected dim; None → receiver d
    pre_projection_layers: int = 3  # C2C-C: 3-layer MLP pre-projection (FR-09)
    dropout: float = 0.0  # optional dropout on cache entries
    residual: bool = True  # Table 8 floor: +Fuse (Eq. 3 residual path)
    gating: bool = True  # Table 8: +Gate (False → ones, no gate)

    def __post_init__(self):
        if self.variant not in ("simple", "c2c-c"):
            msg = f"unknown fuser variant {self.variant!r}; choose 'simple' or 'c2c-c'"
            raise ValueError(msg)
        if self.pre_projection_layers < 1:
            msg = "pre-projection layers must be a positive integer"
            raise ValueError(msg)


@dataclass(frozen=True)
class GateConfig:
    """Learnable per-layer gate (FR-07).

    Gumbel-Sigmoid, temperature annealed *linearly* from ``tau_max`` to
    ``tau_min`` across training steps; differentiable while training, hard
    binary at inference.
    """

    tau_max: float = 1.0
    tau_min: float = 0.001
    threshold: float = 0.5  # hard decision at inference: open if p > θ
    straight_through: bool = True  # ST-G estimator gradients, soft sampling

    def temperature_at(self, step: int, total_steps: int) -> float:
        """The linear temperature schedule τ(step) = τ_max − (τ_max−τ_min)·step/T."""
        if total_steps <= 0:
            return self.tau_min
        frac = max(0.0, min(1.0, step / total_steps))
        return self.tau_max - (self.tau_max - self.tau_min) * frac


@dataclass(frozen=True)
class BlendConfig:
    """Progressive blending policy (FR-08, App. A.2.4, Fig. 11).

    ``fraction`` is the share of cache entries replaced by fused entries;
    ``direction`` picks traversal: ``former`` (front-to-back) or ``latter``
    (back-to-front). Above 50 % the accuracy must rise monotonically.
    """

    fraction: float = 1.0  # 0.0 … 1.0 (or 0 … 100 percent, normalized)
    direction: str = "former"  # former | latter

    def __post_init__(self):
        frac = self.fraction
        if 1 < frac <= 100:  # tolerate percentages: 75 → 0.75
            frac = frac / 100.0
        object.__setattr__(self, "fraction", min(1.0, max(0.0, frac)))
        if self.direction not in ("former", "latter"):
            msg = f"blend direction must be 'former' or 'latter', not {self.direction!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class AlignConfig:
    """Token & layer alignment (FR-10, FR-11)."""

    token_collision: str = "maximal-coverage"  # maximal-coverage | first-occurrence
    layers: str = "terminal"  # terminal | depth-normalized (Eq. 5)
    unknown_fallback: str = "keep"  # keep | replace (tokenizer behaviour)
    pad_token: str = "<pad>"  # template sections length padding

    def __post_init__(self):
        if self.token_collision not in ("maximal-coverage", "first-occurrence"):
            msg = "token_collision must be 'maximal-coverage' or 'first-occurrence'"
            raise ValueError(msg)
        if self.layers not in ("terminal", "depth-normalized"):
            msg = "layers must be 'terminal' or 'depth-normalized'"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainRecipe:
    """The published default training recipe — verbatim from App. A.3.5.

    ``estimate_steps`` (~1929) is derived: 500k samples / macro batch 256
    ≈ 1953 → the paper reports ≈1929 optimisation steps for one epoch.
    """

    dataset: str = "teknium/OpenHermes-2.5"
    num_samples: int = 500_000  # first 500k samples
    max_seq_length: int = 2048
    epochs: int = 1
    macro_batch_size: int = 256
    learning_rate: float = 1e-4  # linear scheduler
    warmup_ratio: float = 0.10  # 10 % warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # gradient clipping
    seed: int = DEFAULT_SEED
    total_steps: int = 1929  # ~1929 steps per epoch
    # convergence targets (paper Table 12 / App. A.3.5): informational
    target_train_loss_steps: int = 250
    target_eval_stable_steps: int = 1000
    gpu_hours_budget: float = 9.0  # ≤ 9 GPU-hours at 300 steps on one A100

    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_ratio)


@dataclass(frozen=True)
class ServeConfig:
    """``c2c-serve`` — OpenAI-compatible HTTPS front (HL-1)."""

    host: str = "127.0.0.1"
    port: int = 8788
    certfile: str | None = None  # HTTPS: PEM certificate
    keyfile: str | None = None  # HTTPS: PEM key
    api_key: str | None = None  # optional Bearer auth
    max_new_tokens: int = 64  # paper evaluation: max response 64
    communication_tokens_budget: int = 256  # paper: communication 256
    model_prefix: str = "c2c/"  # virtual model ids start with this prefix
    pair_separator: str = "←"  # `c2c/<receiver>←<sharer>`; ASCII ok too
    cors: bool = True
    privacy: bool = False  # --privacy: refuse the sharer, digest the logs (EX-5)

    def pair_from_model(self, model_id: str) -> tuple[str, str] | None:
        """Parse a virtual model id into (receiver, sharer).

        Accepts the canonical arrow ``←`` plus ASCII alternatives ``->``,
        ``-->``, ``:` and ``--`` so *every* harness can type it::

            c2c/qwen3-0.6b←qwen2.5-0.5b
            c2c/qwen3-0.6b->qwen2.5-0.5b
            c2c/qwen3-0.6b:qwen2.5-0.5b
        """
        if not model_id:
            return None
        raw = model_id
        if raw.startswith(self.model_prefix):
            raw = raw[len(self.model_prefix) :]
        for sep in (self.pair_separator, "+", "-->", "->", "→", "--", ":"):
            if sep in raw:
                left, right = raw.split(sep, 1)
                left, right = left.strip(), right.strip()
                if left and right:
                    return (left, right)
        return None


@dataclass(frozen=True)
class PrivacyConfig:
    """EX-5 privacy mode: cache segments on the wire without explicit text."""

    enabled: bool = False
    aes_gcm: bool = True  # AES-GCM on the wire (extra: c2c-cache[crypto])
    no_text_egress: bool = True  # `--no-text`: raw strings never leave the box


@dataclass(frozen=True)
class C2CConfig:
    """The complete configuration tree of a C2C deployment."""

    fuser: FuserConfig = field(default_factory=FuserConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    train: TrainRecipe = field(default_factory=TrainRecipe)
    serve: ServeConfig = field(default_factory=ServeConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    seed: int = DEFAULT_SEED
    device: str = "auto"  # auto | cpu | cuda | mps
    root: str | None = None  # state dir; default ~/.cache/c2c


# ---------------------------------------------------------------------------
# environment registration
# ---------------------------------------------------------------------------

_ENV_MAP: dict[str, tuple[str, type]] = {
    "C2C_SEED": ("seed", int),
    "C2C_DEVICE": ("device", str),
    "C2C_ROOT": ("root", str),
    "C2C_FUSER_VARIANT": ("fuser.variant", str),
    "C2C_FUSER_DROPOUT": ("fuser.dropout", float),
    "C2C_GATE_TAU_MAX": ("gate.tau_max", float),
    "C2C_GATE_TAU_MIN": ("gate.tau_min", float),
    "C2C_BLEND_FRACTION": ("blend.fraction", float),
    "C2C_BLEND_DIRECTION": ("blend.direction", str),
    "C2C_ALIGN_TOKEN_COLLISION": ("align.token_collision", str),
    "C2C_ALIGN_LAYERS": ("align.layers", str),
    "C2C_LR": ("train.learning_rate", float),
    "C2C_MACRO_BATCH": ("train.macro_batch_size", int),
    "C2C_MAX_SEQ": ("train.max_seq_length", int),
    "C2C_TOTAL_STEPS": ("train.total_steps", int),
    "C2C_PORT": ("serve.port", int),
    "C2C_HOST": ("serve.host", str),
    "C2C_API_KEY": ("serve.api_key", str),
    "C2C_PRIVACY": ("privacy.enabled", bool),
}


def _write(root, name: str, value) -> None:
    """Set one attribute, frozen instances and all."""
    try:
        setattr(root, name, value)
    except dataclasses.FrozenInstanceError:
        object.__setattr__(root, name, value)


def _revalidate(root) -> None:
    """Re-run the data class’s __post_init__ checks, if it has any."""
    post = getattr(type(root), "__post_init__", None)
    if post is not None:
        post(root)


def _dotted(root, path: str, value) -> None:
    """Walk the dotted path down, set the value, then re-check invariants."""
    parts = path.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    _write(root, parts[-1], value)
    _revalidate(root)


def from_env(config: C2CConfig | None = None, environ: dict[str, str] | None = None) -> C2CConfig:
    """Register environment variables into the configuration tree."""
    env = os.environ if environ is None else environ
    cfg = config or C2CConfig()
    for key, (path, typ) in _ENV_MAP.items():
        if key in env:
            try:
                _dotted(cfg, path, _env_cast(env[key], typ))
            except (TypeError, ValueError) as exc:
                sys.stderr.write(f"c2c: ignoring {key}={env[key]!r}: {exc}\n")
    return cfg


def load_config(path: str | None = None, *, environ: dict[str, str] | None = None) -> C2CConfig:
    """Read ``c2c.toml``-style JSON config, then overlay the environment.

    Plain JSON (stdlib, no external parser) at ``$C2C_ROOT/config.json`` or
    an explicit ``path``. Unknown keys are reported but do not abort —
    configuration errors should be loud, user mistakes tolerated.
    """
    import json

    warnings: list[str] = []
    cfg = C2CConfig()
    candidates = (
        [path]
        if path
        else [
            os.path.join(
                os.environ.get("C2C_ROOT", os.path.expanduser("~/.cache/c2c")), "config.json"
            ),
        ]
    )
    for cand in filter(None, candidates):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as fh:
                data = json.load(fh)
            _merge(cfg, data, warnings)
            break
    for warning in warnings:
        sys.stderr.write(f"c2c: {warning}\n")
    return from_env(cfg, environ)


def _merge(root, data: dict, warnings: list[str], prefix: str = "") -> None:
    """Walk the JSON overlay into the data-class tree, reporting unknown keys."""
    for key, value in data.items():
        attr = getattr(root, key, None)
        if attr is not None and is_dataclass(attr) and isinstance(value, dict):
            _merge(attr, value, warnings, f"{prefix}{key}.")
            continue
        names = {f.name for f in fields(root)}
        if key not in names:
            warnings.append(f"unknown config key: {prefix}{key}")
            continue
        _write(root, key, value)
        _revalidate(root)


# ---------------------------------------------------------------------------
# man c2c.config — the two heads, and every default, documented (FR-06)
# ---------------------------------------------------------------------------

MAN_C2C_CONFIG = """\\
C2C.CONFIG(5)          C2C Middleware Manual            C2C.CONFIG(5)

NAME
    c2c.config — configuration of the C2C cache-to-cache middleware


SYNOPSIS
    c2c [--config FILE] COMMAND ...
    The JSON overlay is read from $C2C_ROOT/config.json (see FILES); each
    registered C2C_* variable of the environment then overrides the file
    (see ENVIRONMENT). Precedence: defaults < file < environment.

DESCRIPTION
    The configuration tree lives in C2CConfig (fuser, gate, blend, align,
    train, serve, privacy). Every default is published here; all defaults
    reproduce the paper's recipe (arXiv:2510.03215v2, App. A.3.5).

    HEADS — beware, the word has two meanings in this project:

    1. MODEL HEADS (attention heads).  hidden_size = num_heads × head_size.
       The dynamic weighting module (FuserConfig, FR-06) performs
       input-aware head modulation: it reweights the *projected*
       information per token/query, one weight per attention head,
       computed from the pooled key/value statistics of the incoming
       cache entries. MHA/GQA/MQA are distinguished via LayerGeometry.
       (See also: the rank of the KV-cache tensors, measured by
       c2c.diagnostics.rank, documented in the module itself.)

    2. CLI HEADS (subcommand headings).  The command `c2c` groups its
       options under CLI heads such as:

           c2c train   — fit the fuser (the only trainable part)
           c2c fuse    — fuse two caches, print the FusionReport
           c2c serve   — OpenAI-compatible HTTPS front (HL-1)
           c2c eval    — golden regression against the paper's tables
           c2c doctor  — inspect the local installation
           c2c zoo     — publish/fetch fuser weights (HF hub, FR-15)

       There is deliberately NO `c2c run-agent` head: the harness stays
       the master; C2C is the wire between models (spec HL-4).

FILES
    $C2C_ROOT/config.json    JSON overlay (then environment wins)
    ~/.cache/c2c/zoo/        local zoo of trained fuser weights

ENVIRONMENT
    C2C_SEED               default seed (42)
    C2C_DEVICE             auto|cpu|cuda|mps
    C2C_FUSER_VARIANT      simple | c2c-c (the C2C-C variant, FR-09)
    C2C_FUSER_DROPOUT      dropout of the fuser paths, 0..1
    C2C_GATE_TAU_MAX       Gumbel-Sigmoid start temperature (1.0)
    C2C_GATE_TAU_MIN       Gumbel-Sigmoid stop temperature  (0.001)
    C2C_BLEND_FRACTION     fused fraction 0..1 (percent 0..100 tolerated)
    C2C_BLEND_DIRECTION    former (front-to-back) | latter (back-to-front)
    C2C_ALIGN_TOKEN_COLLISION   maximal-coverage | first-occurrence
    C2C_ALIGN_LAYERS       terminal | depth-normalized
    C2C_LR                 learning rate of the linear scheduler (1e-4)
    C2C_PORT / C2C_HOST / C2C_API_KEY        serve front
    C2C_PRIVACY            EX-5 privacy mode on/off

SEE ALSO
    c2c(1), c2c-zoo(5), c2c-fuser(5), c2c-engines(7), `c2c man config`

END C2C.CONFIG(5)
"""


if __name__ == "__main__":  # python -m c2c.config  prints the man page
    print(MAN_C2C_CONFIG)
