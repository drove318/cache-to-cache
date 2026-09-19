# road-b — the wired road, walked

The box already runs the model: vLLM serving
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` as `qwen3.8-flash-next`, 524288
context, fp8 caches, on port 8888. road-b mounts the C2C wire inside
that same server — no second model, no cache ever crossing the wire —
and proves, byte by byte, that nothing changed until you ask it to.

## The commands, in order

```bash
# 1 — while the plain server still stands: keep its word
road-b/identity_proof.sh baseline

# 2 — relaunch with the connector riding, gate closed (the default)
GATE=closed road-b/launch_wired.sh

# 3 — the answer must be identical, face after face. B0.
road-b/identity_proof.sh verify

# 4 — the front, so any harness rides the wire without changing
c2c-serve --engine vllm-wired \
    --url http://127.0.0.1:8888 \
    --pair 'qwen3.8-flash-next←qwen3.8-flash-next'
```

Step 2 rebuilds the container from its own birth certificate (image,
argv, env, binds — including the read-only engine patches the box
mounts under `Qwen3.8-Flash-Next-Single-DGX-Spark/files/`), installs
the wheel from this checkout, and adds the one flag that mounts the
connector in the worker. The plain container is parked as
`vllm-fn-tp1-plain`, never deleted; a wired launch that fails to reach
health rolls itself back.

Then, when a wire exists (road-b/train_handoff.md — the ledger and the
two roads):

```bash
GATE=open C2C_WIRE=/path/to/w.pt road-b/launch_wired.sh
road-b/identity_proof.sh verify     # now expect the answers to MOVE, and be better
```

## What each gate means

| gate | the connector, in the worker | the answers |
|---|---|---|
| `closed` | rides every layer, touches none | must be identical — B0 |
| `open` | resident wire fuses inside the load window | must be the models own voice, informed |
| no wire given | serves the identity and says so | identical, loudly honest |

Anything that cannot be proved safe fails closed: a poisoned row, a
missing wire, a card that disagrees — the receiver answers from its own
cache, the fault is logged, nothing pretends.

## The two traps this box has died of before

- **One pool, two servers.** The memory is unified; the research-loop
  watchdog keeps a competing vLLM on :8000 that wakes itself. The
  launch script refuses to start while :8000 answers, and the freeze
  is in its message. Do not work around that refusal.
- **The binds are the engine.** Four read-only mounts patch vLLM
  itself (PLE offload, quantization, qsa ops). Anything that relaunches
  the image by hand and forgets them boots a different model. The
  script replays them from the inspect, always.

`c2c doctor` reports the whole installation, wired adapters included.
