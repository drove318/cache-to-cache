# The change log of the changes

All notable changes to C2C — Cache-to-Cache — are documented here.
The format follows Keep a Changelog; the numbers, Semantic Versioning.

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
- The oracles of §3.1–3.2: cache enrichment (Table 1) and cache transformation
  (t-SNE, Table 2), exposed as `c2c probe`.
- Nine engine adapters with a shim fallback per specification (HF Transformers,
  vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX, TGX, Ollama, reference).
- The OpenAI-compatible front: `c2c-serve`, with HTTPS via certificate, and the
  API key, optional. Routes: /v1/models, /v1/chat/completions, /v1/completions,
  /healthz, and the agent card on its well-known path.
- The MCP server (`c2c-mcp`) with the three tools of the interface —
  `c2c_register_pair`, `c2c_fuse`, `c2c_ask` — on stdio or over a port; the A2A
  bridge (`c2c-a2a`) with the card on the well-known path and the tasks as JSON-RPC.
- The zoo: publish, get, list, remove — O(1) in the count of sharers, per
  the contract of FR-13 / EX-1; and the blending of the many caches, in one
  call, linear in the number of caches, `c2c.zoo.fuse_many_to_one`.
- The diagnostics: `c2c doctor` (with `--report`, the machine-readable
  manifest), the gate attributions of every fusion (`c2c.diagnostics.failure`),
  the effective-rank estimator (Roy & Vetterli), and the gates of quality.
- The golden regression suite: `c2c eval --table N`, the numbers of the paper,
  Tables 3 through 8, with the fixtures offline — no internet except fixtures.
- The control plane: `c2c` with seven commands — doctor, fuse, train, serve,
  eval, zoo, man — and deliberately no eighth, for there is no agent loop here.
- The man pages: `man c2c`, `man c2c.config`, `man c2c-fuser`, `man c2c-engines`,
  `man c2c-zoo`, `man c2c-serve`, `man c2c-proxy`, `man c2c-mcp`, `man c2c-a2a`.
- Progressive blending of the caches, at inference, in the serving path: the
  fraction, from 0 to 100, `-f`, with the direction, former or latter, `-d`.

## [0.1.0] — 2026-09-17, the scaffold of the whole

### Added
- The package skeleton: types, config, protocols, utils; the fuser, the aligner,
  the trainer, the probes, the diagnostics, the zoo, the served, the CLI.
- pyproject.toml: the name on the tin, `c2c-cache`; the console scripts, eight;
  the license, Apache License 2.0.
- The reference engine: two small models, deterministically built from a seed —
  enough to train a fuser, enough to serve a completion, enough to test the
  whole of it, offline, on any harness, on any CPU.
- The first tests: unit, golden, conformance, and the smoke tests of the served.

[1.0.0]: https://github.com/paul-j-reuer-account/cache-to-cache/releases/tag/v1.0.0
[0.1.0]: https://github.com/paul-j-reuer-account/cache-to-cache/releases/tag/v0.1.0
