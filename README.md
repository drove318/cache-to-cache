# C2C — Cache-to-Cache

**Direct semantic communication between large language models.**

C2C is a middleware library, not a harness. It never owns an agent loop,
a tool dispatcher, or a session. It is the *wire between models*: it fuses
one model's KV-cache into another's — through a trainable fuser, never
through generated text — and bolts into whatever stack you already run.

Normative basis: Fu et al., *Cache-to-Cache: Direct Semantic Communication
Between Large Language Models*, ICLR 2026, arXiv:2510.03215v2. Where this
README and the paper disagree, the paper wins.

```text
        sharer LLM ──► cache C(X) ─┐
                                     ├─► fuser (Eq. 3, trainable) ─► fused cache ─► installed ─► receiver answers
        receiver LLM ─ cache ∅ ────┘         frozen weights everywhere
```

---

## Install

```bash
pip install c2c-cache                 # core: types, config, CLI, servers, diagnostics
pip install "c2c-cache[train]"       # + torch: the fuser, the trainer, the alignment nets
pip install "c2c-cache[gpu]"         # + CUDA builds of the engines you wire
```

Python ≥ 3.10, `numpy≥1.24` always; `torch` only where neurons learn.
Everything that does not learn — capture, wire, serve, probe, diagnose —
runs without torch, so C2C imports cleanly in places where torch cannot go.

## Quickstart

One console, seven commands. Try the whole paradigm in sixty seconds:

```bash
pip install "c2c-cache[train]"
c2c doctor                                   # the checkup, printed
c2c fuse --receiver receiver-mini --sharer sharer-mini \
       -e reference --prompt "what is two plus two" --answer --report
c2c train -d fixtures/datasets/tiny.jsonl \
       --receiver receiver-mini --sharer sharer-mini -e reference --report
c2c eval --table 4 --verbose                  # the paper's Tables 3–8, as far as CI can see
```

The answer of the fusion prints, and the report of the fuser tells: per-layer
gates, injection order, blend fractions — the numbers of the paper, on your
terminal.

## Use, anywhere; wire between models

The same wire, every harness. Point a client, or a harness, at the served
front — an OpenAI-compatible endpoint — and the models speak cache-to-cache:

```bash
c2c-serve --pair receiver-mini:sharer-mini -e hf --port 8121 --api-key "$C2C_API_KEY"
```

```python
from openai import client                       # any client, any harness, any model
client = client(base_url="http://127.0.0.1:8121/v1", api_key=os.environ["C2C_API_KEY"])
client.chat.completions.create(model="c2c/receiver-mini+sharer-mini",
                               messages=[{"role": "user", "content": "two plus two?"}])
```

Hermes, OpenClaw, Claude Code, Codex, Pi, OpenCode — every harness installs,
every harness communicates. See [`docs/harnesses/`](docs/harnesses/), one
recipe per harness; for the agnostic, the generic route is
[`docs/harnesses/generic.md`](docs/harnesses/generic.md). The MCP server
(`c2c-mcp`) and the A2A bridge (`c2c-a2a`) are documented on their own
man pages: `man c2c-mcp`, `man c2c-a2a`.

## The fuser, in one screen

The fuser is the neural heart of C2C — three modules, one residual
(Eq. 3): `fused = receiver + gate · f(concat(receiver, projected_sharer))`.

| module              | paper                     | code                                    |
|-------------------|--------------------------|-----------------------------------------|
| projection         | §3.3.2, Eq. 2           | `c2c.fuser.modules.complex`             |
| dynamic weighting  | §3.3.2, Eq. 4           | `c2c.fuser.weighting`                   |
| gating             | §3.3.2, Eq. 5; Table 8 | `c2c.fuser.gating`                      |

Alignment of the nets — token and layer — lives in `c2c.align`; the
training scheme (both LLMs frozen, only the fuser learns) in `c2c.train`;
the oracles of §3.1–3.2 (cache enrichment, cache transformation) in
`c2c.probes`. Configuration is `c2c.config` — see `man c2c.config` before
you set anything, and note there that the configuration knows two heads:
the attention heads of the geometry, and the steering heads of the console.

## Where to look

| you want                      | read                                   |
|------------------------------|----------------------------------------|
| the console, in full         | `man c2c`, `man c2c-commands`          |
| the configuration            | `man c2c.config`                        |
| the fuser, the nets          | `man c2c-fuser`                         |
| the engines, the adapters    | `man c2c-engines`                       |
| the menagerie (zoo)          | `man c2c-zoo`                           |
| the served, the relay        | `man c2c-serve`, `man c2c-proxy`       |
| the tools of the trade (MCP) | `man c2c-mcp`                           |
| the agents, toward each other (A2A) | `man c2c-a2a`                    |
| the paper, per the tables    | `c2c eval --table N --verbose`          |
| something is wrong            | `c2c doctor --report`, `c2c.failures`   |

## Reproducibility

The golden regression suite replays the paper's Tables 3–8 against the
shipped implementation: `c2c.eval.golden` carries the numbers as published;
the fixtures under `fixtures/` (two toy models, five samples, four prompts)
make the suite run offline — no internet except fixtures. `c2c doctor`
diagnoses what your environment can, in one screen; what it cannot, it
reports honestly.

## Contributing, security, conduct

Read the man pages before you contribute:
[CONTRIBUTING.md](CONTRIBUTING.md). Report a vulnerability, never in the
open, per [SECURITY.md](SECURITY.md). Be kind; this project upholds a
[code of conduct](CODE_OF_CONDUCT.md). The change log of the changes:
[CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE). *Make love, make it pass
all tests.*
