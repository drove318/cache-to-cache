# The change log of the changes

All notable changes to C2C — Cache-to-Cache — are documented here.
The format follows Keep a Changelog; the numbers, Semantic Versioning.

## [unreleased] — the wired road, walked twice: the tokens, the tools, the streams

### Added
- The tenth engine adapter, `vllm-wired`: the front speaks to a running
  wired vLLM server. Token ids ride the prompt field untouched, the
  tokenizer stays the servers own (/tokenize, /detokenize), and the pair
  rides the ferry the engines completion protocol defines
  (kv_transfer_params.c2c) — no new endpoints, no fork.
- The wired connector family: `C2CWiredConnector` in the workers
  connector slot, the resident wire running Eq. (3) on the engines device
  with fp8 requantisation, and fail-closed at every step. The factorys
  own gates — the SupportsHMA marker, the three-argument constructor —
  passed on the containers own interpreter.
- `used_cache` tells the truth of the pair: an in-engine fuse claims it,
  a plain relay never does, and a sealed privacy request forecloses it.
- The remote engines finally reachable from the CLI: `--url`,
  `--receiver-url`, `--sharer-url` carry the server address to every
  HTTP adapter, wired included.
- `road-b/`, the wired road walked: a launch that replays the containers
  birth certificate from `docker inspect` (argv, env, the engine-patching
  binds, host net, gpus) and adds only `--kv-transfer-config`; a
  `DRY_RUN=1` rehearsal that prints the exact line and touches nothing;
  health before word, automatic rollback on any failure; a six-face
  greedy identity proof (B0) that admits only byte-exact answers; and
  `train_handoff.md` — the ledger why the wire cannot be fitted on this
  box, and the two roads that can.
- `c2c train --engine vllm-wired` refuses, and refuses honestly: the
  fp8 kernels are a value without a gradient, the unified pool holds no
  second model, and the wire must wear the servers geometry.
- `c2c-serve --timeout SECS` sets the window the sidecar allows the
  served to answer; the 600-second default covers ordinary exchanges,
  but the wired pair rides each leg of a 500k prompt to minutes, so the
  long haul names the window up (1800 seconds on this box).

### Fixed
- The wired connector carried two abstract debts it had never paid:
  `request_finished_all_groups` (the HMA marker) and `update_state_after_alloc`
  (the base). Python will not build a class in debt to its contract — the
  factorys `connector_cls(...)` raised TypeError and the engine core died at
  init before a single token was scored. Both are paid at the letters of the
  base (the wire holds copies, never the engines blocks: `False, None`); the
  test stubs now demand the same abstractmethods the real bases do, so a
  debt shows in the suite, not at two in the morning. The class is proved
  instantiable on the containers own interpreter, under the real bases,
  before any launch replays it.
- A base that refuses the cards no longer kills the worker: registration
  failure logs the fault, closes the gate, and the receiver answers from its
  own cache — fail closed, never fatal.
- The tokenizer's roads take the shapes the server will step in on: the
  `/tokenize` prompt as a string, the `/detokenize` ids flat — the batch
  form is refused on the B side of the line, and the front answers through
  on both roads.
- The tools arm is no longer dropped in good faith: when the `tools` array
  rides, the wired adapter's receiver's leg switches to the container's
  `/v1/chat/completions`, where the served model's own parser picks out the
  calls — the harness sees `finish_reason=tool_calls` and the structure,
  in place of the model's imitation of the call as prose. The pair
  stamping rides the chat road as well as the ids; the non-chat engines
  can bypass the branch, so the caller need not stop for the type change.
- The streamer brings the calls back as a field: on the chat road the
  `tool_calls` ride the final frame's delta, the content framed alone —
  the dumps stay out of the wire, and the harness parses the delta and
  stops on the finish.
- The face-2 budget of the identity proof leaves headroom for an open
  think block, so the bare number reaches the bottom when the model
  opens its reasoning: B0 HOLDS on all six faces.

## [1.0.0] — the first public release (roadmap, per the specification's M6)

### Added
- The cache-to-cache paradigm itself: Fu et al., ICLR 2026, arXiv:2510.03215v2,
  reproduced in full, with the paper as the normative basis.
- The fuser: projection, dynamic weighting, and learnable gating — the three
  modules of §3.3.2 — as residual fusion, Eq. (3), `fused = receiver + gate · f(concat(...))`.
- Alignment of tokens and layers: maximal string coverage, first occurrence,
  terminal layer alignment, depth-normalised mapping (Eq. 5).
- The training scheme: both LLMs frozen, the fuser alone learns; Gumbel-Sigmoid
  annealing, temperature 1.0 to 0.001 (FR-07).
- The oracles of §3.2: cache enrichment (Table 1) and cache transformation
  (t-SNE, Fig. 3), exposed in `c2c.probes`.
- Nine engine adapters with a shim fallback per specification (HF Transformers,
  vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX, TGI, Ollama, reference).
- The OpenAI-compatible front: `c2c-serve`, with HTTPS via certificate, and the
  API key, optional. Routes: /v1/models, /v1/chat/completions, /v1/completions,
  /healthz, and the agent card on its well-known path.
- The MCP server (`c2c-mcp`) with the three tools of the interface —
  `c2c_register_pair`, `c2c_fuse`, `c2c_ask` — on stdio or over a port; the A2A
  bridge (`c2c-a2a`) with the card on the well-known path and the tasks as JSON-RPC.
- The zoo: publish, get, list, remove — O(1) in the count of sharers, per
  the contract of FR-15 / EX-1; and the blending of the many caches, in one
  call, linear in the number of caches, `c2c.zoo.UnifiedLatentSpace`.
- The diagnostics: `c2c doctor` (with `--report`, the machine-readable
  manifest), the gate attributions of every fusion (`c2c.diagnostics.failure`),
  the effective-rank estimator (Roy & Vetterli), and the gates of quality.
- The golden regression suite: `c2c eval --table N`, the numbers of the paper,
  Tables 3 through 8, with the fixtures offline — no internet except fixtures.
- The policy of the list, `gate=block`, per query on every served front: name the
  sharers to silence in `c2c.block_list`; the named cache is not consulted, the
  refusal is logged, and the answer carries `gate: "block"` — documented, never
  silently filtered (SECURITY.md, the paper's first limitation, spec §8).
- The control plane: `c2c` with seven commands — doctor, fuse, train, serve,
  eval, zoo, man — and deliberately no eighth, for there is no agent loop here.
- The man pages: `c2c man c2c`, `c2c man config`, `c2c man fuser`, `c2c man engines`,
  `c2c man zoo`, `c2c man serve`, `c2c man proxy`, `c2c man mcp`, `c2c man a2a` —
  read anywhere, no pager install needed; `make man` lays the eight files for `man`.
- Progressive blending of the caches, at inference, in the serving path: the
  fraction, from 0 to 100, `-f`, with the direction, former or latter, `-d`.

## [0.1.0] — 2026-09-17, the scaffold of the whole

### Added
- The package skeleton: types, config, protocols, utils; the fuser, the aligner,
  the trainer, the probes, the diagnostics, the zoo, the served, the CLI.
- pyproject.toml: the name on the tin, `c2c-cache`; the console scripts, four;
  the license, Apache License 2.0.
- The reference engine: two small models, deterministically built from a seed —
  enough to train a fuser, enough to serve a completion, enough to test the
  whole of it, offline, on any harness, on any CPU.
- The first tests: unit, golden, conformance, and the smoke tests of the served.

[1.0.0]: https://github.com/drove318/cache-to-cache/releases/tag/v1.0.0
[0.1.0]: https://github.com/drove318/cache-to-cache/releases/tag/v0.1.0
