# Contributing to C2C

Read the man pages, read the paper, then make it pass all tests.
That is the specification's instruction to implementers, and it is the
instruction to contributors too.

## The short of it

1. **Read the two normative sources.** The paper (Fu et al., *Cache-to-
   Cache*, ICLR 2026, arXiv:2510.03215v2) is normative; where the spec
   and the paper disagree, the paper wins. Where both are silent, the
   tests decide. `c2c man c2c`, `c2c man fuser`, `c2c man config`, and the
   rest, are the living documentation; keep them true to the code.
2. **Branch from `main`,** name the branch for the change, and open the
   pull request against `main`.
3. **Make it pass all tests** before you ask for review:

   ```bash
   make test        # the unit suite, on CPU, in seconds
   make lint        # ruff, the format check, the type check, the docstyle
   make golden      # the paper's Tables 3–8, against c2c.eval.golden
   make man         # the man pages render, and the pager finds them
   ```

4. **Be civil.** See CODE_OF_CONDUCT.md. Report security per SECURITY.md,
   never in the open.

## The environment

```bash
git clone https://github.com/drove318/cache-to-cache
cd cache-to-cache
python -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"        # torch, pytest, ruff, mypy
c2c doctor                            # the checkup, printed; make it say ok
```

`c2c doctor` is the first thing to run and the last thing to trust. If it
cannot reproduce your fault, `c2c.diagnostics.failure` attributes every
fusion layer by layer — which gate opened, how far its delta travelled.

## The way of a change

- **One change, one pull request.** A new fuser variant is not a drive-by
  fix in `config.py`.
- **Tests first where behaviour changes.** The golden suite is the paper's
  numbers; a change to the fuser that moves them must say so in the
  request, and be justified against the paper, not against a re-run of the
  suite that quietly re-pinned the numbers.
- **No new dependencies in the core.** The core imports numpy and the
  standard library only; torch, transformers, vllm, sglang are extras and
  lazy. That is the "any harness, no internet except fixtures" contract,
  and it is load-bearing.
- **Public API is a promise.** `__init__.py`'s table of exports, the wire
  routes, the tool names of the MCP server, and the exit codes of the
  console are all, in their way, API. A change to any of them needs a
  deprecation route and an entry in CHANGELOG.md.
- **Style:** the code follows the style of the surrounding code — the
  house voice is the house. Ruff enforces it; `make lint` reports it.

## The plugin, the SDK (one ABI, many adapters)

An adapter is the pair of interfaces, `CacheProvider` (capture) and
`CacheInjector` (install, generate). Ship a package that exposes the
entry-point group `c2c.engines`, one entry per adapter::

    [project.entry-points."c2c.engines"]
    myengine = "my_engine.c2c:MyEngineAdapter"

The class implements the two interfaces and a `spec()` returning the
`ModelSpec` of the model it drives, plus two optional class members:
`required_extra` (the pip extra that installs the engine, or `None` for
an endpoint-only engine) and `DEGRADATION` (why the capture is
prefill-only, where the engine exposes no cache hook — printed by
`c2c doctor`, never silent). When the engine cannot be imported, raise
`AdapterNotSupported(hint=...)`: the hub answers with the relay, and
says so. The module `c2c.integrations.registry` is the implementation;
`c2c.engines(7)` is the page.

## What the CI will check

The matrix (`.github/workflows/ci.yaml`) builds, on `pull_request` and on
`push` to `main`:

- **unit** — `python -m pytest tests/` on CPU: the whole suite; the slow and
  the gpu tests skip themselves, honestly, when the hardware cannot run them;
- **lint** — `ruff check`, `ruff format --check`, and `mypy` on the sources;
- **import** — `python tools/import_smoke.py`: every module imports, with
  and without torch present, on a machine with no network;
- **golden** — rides inside the test job (`tests/test_golden.py`): the
  paper's Tables 3–8, within tolerance, against the shipped fixtures;
  skipped, not failed, when the hardware cannot run the models (the suite
  is honest about what it did not check). Targeted runs: `c2c eval
  --table N --verbose`.

A red CI does not merge. A missing fixture does not silently pass.

## Signing your work

Every commit carries the sign-off, per the Developer Certificate of
Origin:

```
Signed-off-by: Your Name <you@example.com>
```

`git commit --signoff` adds it. It certifies you wrote the change, or
have the right to pass it on, under the Apache License 2.0.

## Questions, ideas, and the shape of a good one

Open an issue. The best ones say: what you ran, what it printed, what you
expected, and which paragraph of the paper you are reading. Paste the
`c2c doctor --report` output; it is the one screen that says the whole
state of the machine.

*Make love, make it pass all tests.*
