# The training handoff — how to fit the wire this box cannot fit

Status: serving is shipped and proved; **training the wire is the operators
road**, and this page is its map. Nothing here is a stub pretending to be
a command: `c2c train -e vllm-wired` refuses, in these same words, and
that refusal is the honest interface.

## The ledger (why not here)

1. **The engines forward is a leaf.** The wired server reads K/V through
   fp8 paged attention kernels. They answer — `prompt_logprobs` and the
   servers own spec-eval prove it — but no gradient flows back through a
   kernel that was never built to. The loss has a value and no direction.
2. **The pool is one.** This GB10 has 121 GiB of unified memory, no
   separate VRAM (the sibling servers script root-caused this twice, and
   the second time the kernel OOM killer took the desktop with the
   server). The live server holds its share at `--gpu-memory-utilization
   0.72`; a second, full-precision copy of the model for scoring would
   not fit beside it, and forcing the attempt is a death, not an error.
3. **The wire must wear the servers geometry.** The connector verifies
   every wire against the card the engine states for the model it loaded
   — head counts, row widths, layer names — and turns away a wire that
   does not match ("the wire's receiver card disagrees with the engine").
   A wire trained at someones textbook geometry is, literally, the wrong
   tensor here. That check is the guarantee; it is also the reason an
   off-the-shelf checkpoint cannot just be dropped in.

The fuser itself is small — a few tens of millions of parameters, a
couple hundred MB of optimizer state. It is not what will not fit. The
model, twice, is.

## The two roads that do work

### Road one — the offline twin (the practical one)

Train on the models full-precision weights, on another day on this box
(while the server is down) or on another card entirely:

1. **Learn the servers card.** Launch wired (`GATE=closed
   road-b/launch_wired.sh`) and read the connectors own lines:
   `docker logs vllm-fn-tp1 | grep -i 'wire\|card\|paged'`. Whatever the
   engine states there is what the wire must wear. Pin it:
   `--receiver-geometry`/`--sharer-geometry` in the config, or the
   adapters `options.geometry`.
2. **Train the HF path at that card** with the checkpoints original
   (unquantized) weights as the two models:
   `c2c train -e hf --config <card + recipe> --dataset <the paper's
   48k instruction samples>` — the paper's own scheme, both models
   frozen, the fuser the only trainable thing (Eq. 3, the three
   modules, Gumbel-Sigmoid annealed gates).
3. **Dress it for the quantized server.** The rows the wired server
   fuses are fp8-equal — the twin trained the fuser on the
   full-precision rows of the same layers. The paper trains and serves
   the same precision; here they differ, so verify before trusting:
   A/B the open-gate answers against the identity baselines (six faces,
   `identity_proof.sh verify` with `GATE=open`) and expect the blend to
   read as quality, not mush. If it is mush, the twin drifted from the
   servers truth and the rows must be sampled from the server itself.
4. **Mount it:** `GATE=open C2C_WIRE=/path/to/w.pt road-b/launch_wired.sh`.
   A wire that fails the card check, the connector refuses to mount —
   it serves the identity and logs why. Failing closed is the contract.

### Road two — the differentiable oracle (the research one)

Inside the worker the model is already resident; what is missing is a
forward that gradients can flow through — an eager, non-paged attention
over the very rows the connector stages (they are already parked, per
layer, in `staging`). A spike:

- a training-mode layer copy (the engines own weights, plain torch
  matmuls, no fp8 kernels), scoring the target tokens under the fused
  rows, with the fuser on the workers device beside them;
- the CE gradient then flows: rows → fuser → gate, all in-process;
- the serving path never touches it; an env-gated driver mounts it
  only when the operator asks.

This is real research, not a flag: kernel parity, memory, and the same
quantized-rows-vs-eager-scores gap as road ones step 3.

## What is already yours

Identity proved (B0), the connector riding the workers slot, the gate
policy (closed/open/wire), fail-closed mounting, the front speaking to
the wired server, and the rollout with rollback: all shipped, all
green, all under `road-b/`. The wire is the one thing between the
parity path and the fusion path, and this page is how to fit it.
