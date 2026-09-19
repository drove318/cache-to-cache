# C2C — Cache-to-Cache

**Direct semantic communication between large language models.**

C2C is a middleware library, not a harness. It never owns an agent loop,
a tool dispatcher, or a session. It is the *wire between models*: it fuses
one model's KV-cache into another's — through a trainable fuser, never
through generated text — and bolts into whatever stack you already run.

As published: C2C lifts the average accuracy of a Receiver **6.4 to 14.2 points**
over the individual models, beats text-to-text communication by **3.1 to 5.4
points**, and answers about **2.5×** faster (Fu et al., Abstract; Tables 3–8).

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

Two facts decide the recipe. Modern Debian and Ubuntu mark the system Python
*externally managed* (PEP 668) — a bare `pip install` there is refused, with
or without `sudo`. And `c2c-cache` is not on the package index yet. So: one
virtual environment, and the repository itself as the index — every command
below is tested from a virgin venv:

```bash
git clone https://github.com/drove318/cache-to-cache
cd cache-to-cache
python3 -m venv .venv && . .venv/bin/activate
pip install ".[train,dev]"          # train: torch for the fuser — dev: pytest for `c2c eval`
```

Just the tools, no clone? Inside the venv, resolve straight from GitHub —
`doctor` and `fuse` work from this install; `train` and `eval` need the
fixtures in a checkout:

```bash
pip install "c2c-cache[train] @ git+https://github.com/drove318/cache-to-cache.git"
```

`[gpu]`, instead of `[train]`, on machines where an engine adapter wants its
own CUDA builds. Python ≥ 3.10, `numpy≥1.24` always; `torch` only where
neurons learn. Everything that does not learn — capture, wire, serve, probe,
diagnose — runs without torch, so C2C imports cleanly in places where torch
cannot go.

## Quickstart

One console, one paste, on a machine that has never seen C2C. Everything the
four commands need — the clone, the venv, the package — travels inside the
paste; the second time you are here, only `. cache-to-cache/.venv/bin/activate`
is needed:

```bash
git clone https://github.com/drove318/cache-to-cache
cd cache-to-cache
python3 -m venv .venv && . .venv/bin/activate
pip install ".[train,dev]"          # the first paste fetches torch — about a minute
c2c doctor                                   # the checkup, printed
c2c fuse --receiver receiver-mini --sharer sharer-mini \
       -e reference --prompt "what is two plus two" --answer --report
c2c train -d fixtures/datasets/tiny.jsonl \
       --receiver receiver-mini --sharer sharer-mini -e reference --epochs 1 \
       -o checkpoints/toy-fuser.safetensors
c2c eval --table 4 --verbose                 # the paper's Tables 3–8, as far as CI can see
```

The answer of the fusion prints, and the report of the fuser tells: per-layer
gates, injection order, blend fractions — the shape of your fuser, on your
terminal. The numbers of the paper ride with `c2c eval` (just above).

The miniature pair is plumbing: deterministic reference nets, no weights, no
downloads — four commands that prove the whole wire on any machine. Real
pairs ride the same two commands with `-e hf`, on any Hugging-Face-format
folders the machine can reach (`pip install "c2c-cache[hf]"` first):

```bash
c2c train -d data/your-instructions.jsonl \
       --receiver Qwen/Qwen3-0.6B --sharer Qwen/Qwen2.5-Math-1.5B \
       -e hf --epochs 1 -o checkpoints/math-to-small.safetensors
c2c-serve -e hf --pair Qwen/Qwen3-0.6B+Qwen/Qwen2.5-Math-1.5B:checkpoints/math-to-small.safetensors
```

Point any OpenAI client at `http://127.0.0.1:8788/v1`, name the pair in
`model=` (the gallery at `/v1/models` prints the exact strings), and the
answer arrives from the receivers mouth, informed by the sharers cache —
a row of the paper's Table 7, the heterogeneous pair, on your hardware.

## Use, anywhere; wire between models

The same wire, every harness. Start the front with the fuser you just
trained (`c2c-serve --pair receiver-mini+sharer-mini:checkpoints/toy-fuser.safetensors`),
point a client at the served front — an OpenAI-compatible endpoint — and the
models speak cache-to-cache; ask the gallery (`/v1/models`) for the exact
name of the pair, and of every single model the front will build on demand:

```python
import os
from openai import OpenAI                       # any client, any harness, any model
client = OpenAI(base_url="http://127.0.0.1:8788/v1",
                api_key=os.environ.get("C2C_API_KEY", "not-needed-on-loopback"))
answer = client.chat.completions.create(
    model="c2c/receiver-mini+sharer-mini",
    messages=[{"role": "user", "content": "two plus two?"}])
print(answer.choices[0].message.content)
```

Oh My Pi (omp), Hermès, Claude Code, Codex, AutoGen, LangChain, CrewAI — every harness installs,
every harness communicates. See [`docs/harnesses/`](docs/harnesses/), one
recipe per harness; for the agnostic, the generic route is
[`docs/harnesses/generic.md`](docs/harnesses/generic.md). The MCP server
(`c2c-mcp`) and the A2A bridge (`c2c-a2a`) are documented on their own
man pages: `c2c man mcp`, `c2c man a2a`.

## The fuser, in one screen

The fuser is the neural heart of C2C — three modules, one residual
(Eq. 3): `fused = receiver + gate · f(concat(receiver, projected_sharer))`.

| module              | paper                     | code                                    |
|-------------------|--------------------------|-----------------------------------------|
| projection         | §3.3.2, Fig. 5            | `c2c.fuser.modules`                   |
| dynamic weighting  | §3.3.2, Fig. 5            | `c2c.fuser.modules`                   |
| gating             | §3.3.2, Fig. 5; Table 8  | `c2c.fuser.modules`                   |
| complex (C2C-C)    | App. A.1.3; Table 9      | `c2c.fuser.complex`                   |

Alignment of the nets — token and layer — lives in `c2c.align`; the
training scheme (both LLMs frozen, only the fuser learns) in `c2c.train`;
the oracles of §3.2 (cache enrichment, cache transformation) in
`c2c.probes`. Configuration is `c2c.config` — see `c2c man config` before
you set anything, and note there that the configuration knows two heads:
the attention heads of the geometry, and the steering heads of the console.

## Where to look

| you want                      | read                                   |
|------------------------------|----------------------------------------|
| the console, in full         | `c2c man c2c`, `c2c man commands`            |
| the configuration            | `c2c man config`                          |
| the fuser, the nets          | `c2c man fuser`                           |
| the engines, the adapters    | `c2c man engines`                         |
| the menagerie (zoo)          | `c2c man zoo`                             |
| the served, the relay        | `c2c man serve`, `c2c man proxy`          |
| the tools of the trade (MCP) | `c2c man mcp`                             |
| the agents, toward each other (A2A) | `c2c man a2a`                   |
| the paper, per the tables    | `c2c eval --table N --verbose`          |
| something is wrong            | `c2c doctor --report`, `c2c.diagnostics.failure` |

Every page reads anywhere with `c2c man <topic>`; `make man` lays the eight
files where the system pager finds them.

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
