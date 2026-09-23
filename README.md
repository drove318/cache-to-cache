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

## The featured setup: Qwen3.8 on a wired vLLM, driven by omp

The road below was walked end to end on a DGX Spark (GB10, unified memory):
one container, the served weights never leave the box, the harness none the
better. It is the setup this repository is tested against — copy, paste,
and try. The miniature pair above is the crossbar: a deterministic toy the
CI can reach, proof without a GPU; the real models ride the bus and the
answer can be bypassed when the container takes the role of the B test.

Four steps, in order. The box reaches the end of the parse in step 4 and
stop, or the caller can stack:

**1. The plain server.** Bring up vLLM in docker serving
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` as `qwen3.8-flash-next`, 524288
context, on loopback port 8888. On this box the operator brings up the
container through `/c2c/boot.sh`; `road-b/launch_wired.sh --help` will
step in and the engine can be bypassed, so the caller can examine the
route map and fall back on the shell's escape.

**2. The identity proof.** While the plain server still stands, keep its
word and take the six greedy faces; then relaunch with the connector
riding, gate closed, and verify the answers byte for byte:

```bash
road-b/identity_proof.sh baseline                          # the plain server's word
GATE=closed road-b/launch_wired.sh                         # the connector mounts in
WIRED_BOOT=plain road-b/identity_proof.sh verify           # B0 HOLDS, face after face
```

The launch replays the container's birth certificate from `docker inspect`
— image, argv, env, the engine-patching binds — installs the wheel from
this checkout, and adds only `--kv-transfer-config`; the plain container
is parked, never deleted, and a launch that fails to reach health rolls
itself back. The road, with its two traps and their cures, is
[`road-b/`](road-b/README.md).

**3. The front.** Let the served speak to the OpenAI side, so the harness
need not call:

```bash
./c2c-venv/bin/c2c-serve --engine vllm-wired \
    --url http://127.0.0.1:8888 \
    --pair 'qwen3.8-flash-next←qwen3.8-flash-next' \
    --timeout 1800
```

The front answers on :8788; the gallery at `/v1/models` prints the exact
pair names, and the access log carries the fused truth of every exchange.
With `tools` in the request the containers parser picks out the calls and
the finish_reason stops on `tool_calls` — the harness parses the delta
and the caller never falls. The `--timeout 1800` window gives the long
haul room: the wired pair rides each leg of a 500k prompt to minutes, and
the 600-second default would drop the call before the answer arrives.

**4. The harness.** Point Oh My Pi at the front and stop — the harness
stays the master:

```bash
omp --model 'c2c/qwen3.8-flash-next←qwen3.8-flash-next'
```

The provider block, in `~/.omp/agent/models.yml`, registers the front as
one model among any other:

```yaml
    c2c:
        api: openai-completions
        auth: none
        baseUrl: http://127.0.0.1:8788/v1
        models:
            - id: c2c/qwen3.8-flash-next←qwen3.8-flash-next
              name: Qwen3.8 wired pair
              # The pair rides the served: these are its numbers, not the
              # harness's defaults. Left out, the harness bills prompt plus
              # output over the ceiling, the served 400s, and every leg
              # comes back a 500.
              contextWindow: 524288
              maxTokens: 32768
```

To try another model or harness, the substitute need not call: name the
model the container serves in `--pair` and the harness's `model=`, point
`--url` at another server, leave the `--port` to the front's default of
8788, or raise `--timeout` when the context runs long. Until a wire is
fitted, the gate stays closed and the answers stand, word for word — see
`road-b/train_handoff.md` for the two roads that can fit one.

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

And when the models live in a wired vLLM server, the same front rides the
engines own connector: `c2c-serve --engine vllm-wired --url http://127.0.0.1:8888
--pair 'name←name'` — token ids ride untouched, the fusion happens inside the
worker, and nothing but a role crosses the wire. The full road, rehearsal
scripts and byte-exact identity proof included, is
[`road-b/`](road-b/README.md) — and the wire itself, the one thing still to
be fitted, has its map in [`road-b/train_handoff.md`](road-b/train_handoff.md).

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
