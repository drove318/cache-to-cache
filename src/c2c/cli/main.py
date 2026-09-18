"""The ``c2c`` console utility — the gateway to all C2C commands.

Usage (see also the manual pages shipped with the package, and
``c2c man <topic>``)::

    c2c [-h] [--version] COMMAND [OPTION] [ARGUMENT] ...

Commands (choose one, sorted by usefulness for the operator):

    doctor    — show the installation's health, check, and report
    eval      — run the golden regression suite against the paper
    fuse      — fuse two caches, print the fusion report
    man       — format the manual pages (config, commands, engines, zoo)
    serve     — start the OpenAI-compatible HTTPS front (HL-1)
    train     — fit the fuser on a dataset (the LLMs stay frozen)
    zoo       — the zoo maintenance interface for fuser checkpoints

There is deliberately no ``run-agent``: the harness stays the master,
C2C is the wire between models (spec HL-4).

Command reference: all long options follow the GNU conventions, where
they exist; short options are chosen for ease of typing. The output
follows the terminal: colour when the file descriptor is a tty and the
environment does not say otherwise (NO_COLOR, TERM=dumb), plain text
otherwise — and the diagnostics go to stderr, always.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .. import __version__
from ..config import MAN_C2C_CONFIG, load_config
from ..utils.console import Progress, Table, banner, error_hint, style

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# shared builders (of the subcommand parsers) — command line parsing rules
# ---------------------------------------------------------------------------

def _add_engine_argument(parser: argparse.ArgumentParser, default: str = "reference") -> None:
    parser.add_argument("-e", "--engine", default=default, metavar="ENGINE",
                       help=f"engine adapter to use (default: {default}); "
                            "see `c2c doctor` for the list of available engines")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receiver", required=True, metavar="ID",
                       help="receiver model id (the one that speaks)")
    parser.add_argument("--sharer", required=True, metavar="ID",
                       help="sharer model id (the one that shares)")


def _add_seed_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                       help="initialisation seed (default: 42, determinism first)")


# ---------------------------------------------------------------------------
# doctor — run a diagnostic, check the installation
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from ..diagnostics.doctor import format_report, run_all
    cfg = load_config(args.config)
    checks = run_all(cfg)
    print(format_report(checks, title="c2c doctor"))
    if args.report:
        import json as _json
        rows = [{"check": c.name, "status": c.status, "detail": c.detail,
              "hint": c.hint} for c in checks]
        print(_json.dumps({"doctor": "c2c", "checks": rows},
                      indent=2, sort_keys=True))
    failed = any(c.status == "fail" for c in checks)
    return 1 if (failed and args.strict) else 0


# ---------------------------------------------------------------------------
# fuse — make two caches, one of them telling
# ---------------------------------------------------------------------------

def cmd_fuse(args: argparse.Namespace) -> int:
    from ..integrations import engines
    from ..fuser.core import Fuser
    from ..align.layers import terminal_mapping, depth_normalized_mapping
    from ..align.tokens import TokenAligner
    from ..config import AlignConfig, BlendConfig, FuserConfig, GateConfig

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    try:
        adapter_r = engines.load(args.engine, model_id=args.receiver, seed=seed)
        adapter_s = engines.load(args.engine, model_id=args.sharer, seed=seed)
    except Exception as exc:
        print(error_hint(f"cannot build the models: {exc}",
                        hint="check `c2c doctor`; the reference engine always works: -e reference"),
              file=sys.stderr)
        return 1

    spec_r, spec_s = adapter_r.spec(), adapter_s.spec()
    mapper = (terminal_mapping if cfg.align.layers == "terminal"
             else depth_normalized_mapping)
    mapping = mapper(spec_r.geometry.layers, spec_s.geometry.layers)

    prompt_text = args.prompt if args.prompt else "hello cache, hello world"
    r_ids = adapter_r.encode(prompt_text)
    s_ids = adapter_s.encode(prompt_text)
    cache_r = adapter_r.capture(r_ids)
    cache_s = adapter_s.capture(s_ids)

    aligner = TokenAligner(adapter_r, adapter_s, strategy=cfg.align.token_collision)
    agreement = aligner.strategy_agreement(r_ids)

    fuser = Fuser(spec_r.geometry, spec_s.geometry, mapping,
                 fuser_config=FuserConfig(variant=args.variant),
                 gate_config=GateConfig(tau_max=cfg.gate.tau_max, tau_min=cfg.gate.tau_min),
                 blend_config=BlendConfig(fraction=args.fraction,
                                        direction=args.direction or cfg.blend.direction))
    fuser.eval()
    import torch
    with torch.no_grad():
        token_mapping = None if r_ids == s_ids else aligner.select_rows(r_ids)
        fused = fuser(cache_r, cache_s, token_mapping=token_mapping)

    print(banner("fuse", __version__))
    table = Table(["", "receiver", "sharer", "fused"], title="caches")
    table.add_row(("layers", len(cache_r), len(cache_s), len(fused)))
    table.add_row(("tokens", cache_r.num_tokens, cache_s.num_tokens, fused.num_tokens))
    print(table)
    if agreement < 0.80:
        print(error_hint(f"selection strategies agree on only {agreement:0.1%}",
                        hint="above 80% is the published expectation; "
                             "token_aligner may need attention (see `c2c man config`)"))
    else:
        print(f"  token alignment: strategies agree on {agreement:0.1%}  ✓")
    if args.report or args.verbose:
        print(fuser.report(num_tokens=cache_r.num_tokens))
    if args.answer:
        adapter_r.install(fused, r_ids)
        reply = adapter_r.generate(r_ids, max_new_tokens=args.max_new_tokens or 16,
                                temperature=args.temperature or 0.0)
        print(f"  reply: {style(reply or '(silence — an untrained reply)', 'cyan')}")
    return 0


# ---------------------------------------------------------------------------
# train — fit the fuser; the two LLMs stay frozen (FR-02, FR-13, FR-14)
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    from ..integrations import engines
    from ..fuser.core import Fuser
    from ..align.layers import terminal_mapping
    from ..config import AlignConfig, BlendConfig, FuserConfig, GateConfig, TrainRecipe
    from ..train.scheme import Trainer, load_jsonl_dataset, manual_seed

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    if not os.path.isfile(args.dataset):
        print(error_hint(f"dataset not found: {args.dataset}",
                        hint="a JSON-L file of {instruction, output} records; "
                             "see examples/datasets for a tiny one"),
              file=sys.stderr)
        return 1

    try:
        adapter_r = engines.load(args.engine, model_id=args.receiver, seed=seed)
        adapter_s = engines.load(args.engine, model_id=args.sharer, seed=seed)
    except Exception as exc:
        print(error_hint(f"cannot build the engines: {exc}",
                        hint="try `--engine reference` (always available)"), file=sys.stderr)
        return 1

    spec_r, spec_s = adapter_r.spec(), adapter_s.spec()
    mapping = terminal_mapping(spec_r.geometry.layers, spec_s.geometry.layers)
    fuser = Fuser(spec_r.geometry, spec_s.geometry, mapping,
                 fuser_config=FuserConfig(variant=args.variant),
                 gate_config=GateConfig(tau_max=cfg.gate.tau_max, tau_min=cfg.gate.tau_min),
                 blend_config=cfg.blend)
    recipe = TrainRecipe(
        dataset=args.dataset,
        num_samples=args.num_samples or cfg.train.num_samples,
        max_seq_length=args.max_seq_length or cfg.train.max_seq_length,
        epochs=args.epochs or cfg.train.epochs,
        macro_batch_size=cfg.train.macro_batch_size,
        learning_rate=args.lr if args.lr is not None else cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        seed=seed,
        total_steps=args.total_steps or cfg.train.total_steps,
    )
    manual_seed(seed)
    trainer = Trainer(fuser, adapter_r, adapter_s, adapter_r,
                     receiver_tokenizer=adapter_r, sharer_tokenizer=adapter_s,
                     recipe=recipe)
    dataset = list(load_jsonl_dataset(args.dataset, limit=args.num_samples))
    if not dataset:
        print(error_hint("the dataset stream is empty",
                        hint="check the JSON-L file: every line needs a context "
                             "and a response"), file=sys.stderr)
        return 1

    print(banner("train", __version__))
    bar = Progress(len(dataset) * recipe.epochs, label="fit") if sys.stderr.isatty() else None

    def on_step(step, loss, gnorm):
        if bar is not None:
            bar.update()
        if args.verbose and (step == 1 or step % 25 == 0):
            bar.close() if bar is not None else None
            print(f"  step {step:>5} loss {loss:0.4f} |grad| {gnorm:0.3f}")

    result = trainer.fit(dataset, epochs=recipe.epochs, on_step=on_step)
    if bar is not None:
        bar.close()
    print("  " + str(result))
    if args.out:
        path = trainer.save_checkpoint(args.out)
        print(f"  checkpoint saved: {path}")
    return 0


# ---------------------------------------------------------------------------
# serve — the OpenAI-compatible HTTPS front (delegates, thin layer)
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace, remaining: Sequence[str]) -> int:
    from ..serve.openai_proxy import main as serve_main
    return serve_main(remaining or None)


# ---------------------------------------------------------------------------
# eval — the golden regression against the paper (spec §5)
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> int:
    import subprocess
    targets = args.target or ["tests"]
    argv = [sys.executable, "-m", "pytest", "-ra"]
    if args.verbose:
        argv.append("-v")
    kparts: list[str] = []
    if args.table:
        from ..eval.golden import TABLE_KEYWORDS
        for n in args.table:
            keyword = TABLE_KEYWORDS.get(int(n))
            if keyword is None:
                from ..eval.golden import UNAVAILABLE
                key = f"table{int(n):02}"
                if key in UNAVAILABLE:
                    msg = f"Table {n} has no kept numbers: {UNAVAILABLE[key]}"
                    tip = "the table is mourned by name in the golden suite"
                else:
                    msg = f"no golden tests are kept for Table {n}"
                    tip = ("the kept tables are "
                           + ", ".join(str(k) for k in sorted(TABLE_KEYWORDS))
                           + "; see c2c.eval.golden")
                print(error_hint(msg, hint=tip), file=sys.stderr)
                return 2
            kparts.append(f"({keyword})")
    if args.markexpr:
        kparts.append(f"({args.markexpr})")
    if kparts:
        argv += ["-k", " and ".join(kparts)]
    argv += list(targets)
    try:
        import pytest                                        # noqa: availability check
    except ModuleNotFoundError:
        print(error_hint("pytest is not installed",
                        hint="pip install 'c2c-cache[dev]' to run the golden suite"),
              file=sys.stderr)
        return 2
    print(banner("eval — golden regression against the paper", __version__))
    return subprocess.run(argv, cwd=_find_repo_root(), check=False).returncode


def _find_repo_root() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for candidate in (here, os.path.curdir, os.path.join(here, "..", "..")):
        if os.path.isdir(os.path.join(candidate, "tests")):
            return candidate
    return os.path.curdir


# ---------------------------------------------------------------------------
# zoo — the zoo maintenance interface for trained fusers (FR-15)
# ---------------------------------------------------------------------------

def cmd_zoo(args: argparse.Namespace) -> int:
    from ..zoo.publish import ZooClient
    client = ZooClient()
    action = args.action
    if action == "list":
        rows = client.list_pairs()
        if not rows:
            print("  the zoo is empty — train a fuser, publish it, and it shall appear")
            return 0
        table = Table(["id", "sharer", "receiver", "created", "sha256"], title="zoo")
        for row in rows:
            table.add_row((row.get("id"), row.get("sharer"), row.get("receiver"),
                         str(row.get("created_at", ""))[:10],
                         str(row.get("sha256", ""))[:12]))
        print(table)
        return 0
    if action == "publish":
        if not args.checkpoint or not os.path.isfile(args.checkpoint):
            print(error_hint("publish needs --checkpoint PATH (a .pt file)",
                            hint="train first: c2c train ... --out model.pt"), file=sys.stderr)
            return 1
        import torch
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = blob.get("state_dict", blob)
        manifest = client.publish(sharer=args.sharer, receiver=args.receiver,
                                 fuser=_StateCarrier(state))
        print(f"  published {manifest['id']} to the local zoo ({client.root})")
        return 0
    if action == "get":
        if not args.sharer or not args.receiver:
            print(error_hint("get needs --sharer and --receiver", file=sys.stderr))
            return 1
        path = client.fetch(sharer=args.sharer, receiver=args.receiver)
        print(f"  fetched to {path}")
        return 0
    if action == "remove":
        removed = False
        for entry in os.listdir(client.root) if os.path.isdir(client.root) else ():
            if entry == args.name:
                import shutil
                shutil.rmtree(os.path.join(client.root, entry))
                removed = True
                break
        print(f"  {'removed' if removed else 'no such entry'}: {args.name}")
        return 0 if removed else 1
    print(error_hint(f"unknown zoo action {action!r}",
                    hint="actions: list, publish, get, remove"), file=sys.stderr)
    return 1


class _StateCarrier:
    """Adapts a raw state dict to the ``state_dict()`` protocol publish requires."""

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


# ---------------------------------------------------------------------------
# man — format the manual pages
# ---------------------------------------------------------------------------

def cmd_man(args: argparse.Namespace) -> int:
    from ..utils.man import render
    text, found = render(args.topic)
    print(text)
    return 0 if found else 1


# ---------------------------------------------------------------------------
# the parser — command line parsing, the whole shebang
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c2c",
        description="Cache-to-Cache: direct semantic communication between LLMs. "
                    "One library, every harness; the harness stays the master.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="there is deliberately no `c2c run-agent`: the harness stays the "
               "master; C2C is the wire between models.",
    )
    parser.add_argument("--version", action="version", version=f"c2c-cache {__version__}")
    sub = parser.add_subparsers(title="commands", metavar="COMMAND",
                              dest="command")

    p = sub.add_parser("doctor", help="check the installation, report the findings")
    p.add_argument("--config", default=None)
    p.add_argument("-s", "--strict", action="store_true",
                  help="exit non-zero when any check fails")
    p.add_argument("--report", action="store_true",
                  help="also print the machine-readable manifest (JSON), for the pipelines")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("fuse", help="fuse two caches; print the fusion report")
    _add_model_arguments(p)
    _add_engine_argument(p)
    _add_seed_device(p)
    p.add_argument("-p", "--prompt", default=None, help="prompt string to capture")
    p.add_argument("-f", "--fraction", type=float, default=1.0,
                  help="fused fraction 0..1 (percent 0..100 tolerated)")
    p.add_argument("-d", "--direction", choices=("former", "latter"), default=None,
                  help="blend traversal order (default from the configuration)")
    p.add_argument("--variant", choices=("simple", "c2c-c"), default="simple",
                  help="fuser variant (C2C-C: App. A.1.3)")
    p.add_argument("--report", action="store_true", help="print the full fusion report")
    p.add_argument("--answer", action="store_true", help="also generate a reply")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_fuse)

    p = sub.add_parser("train", help="fit the fuser on a dataset; the LLMs stay frozen")
    _add_model_arguments(p)
    _add_engine_argument(p)
    _add_seed_device(p)
    p.add_argument("-d", "--dataset", required=True, metavar="FILE",
                  help="JSON-Lines file of training samples")
    p.add_argument("-o", "--out", metavar="FILE", help="write the checkpoint to FILE")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--variant", choices=("simple", "c2c-c"), default="simple")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="start the OpenAI-compatible HTTPS front")
    p.add_argument("extras", nargs=argparse.REMAINDER, default=[],
                  help="see c2c-serve --help")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run the golden regression against the paper")
    p.add_argument("target", nargs="*", default=None)
    p.add_argument("--table", type=int, action="append",
                  help="run only Table N (repeatable)")
    p.add_argument("-k", "--markexpr", default=None, help="pytest expression filter")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("zoo", help="the zoo maintenance interface for trained fusers")
    p.add_argument("action", choices=("list", "publish", "get", "remove"))
    p.add_argument("--sharer", default=None)
    p.add_argument("--receiver", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--name", default=None, help="entry name for remove")
    p.set_defaults(func=cmd_zoo)

    p = sub.add_parser("man", help="format the manual pages")
    p.add_argument("topic", nargs="?", default="intro")
    p.set_defaults(func=cmd_man)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:                                   # no command given: be friendly
        parser.print_help()
        print()
        print(banner("the wire between models", __version__))
        return 0
    if args.command == "serve":
        return func(args, args.extras)
    try:
        return func(args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except BrokenPipeError:                            # the pager hung up: do not panic
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
