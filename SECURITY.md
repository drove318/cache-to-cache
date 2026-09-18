# Security Policy

C2C is a wire between models. A wire carries what it is given — and a wire
between machines is a place where trust must be measured twice. This document
says what C2C protects, how to report a fault in it, and what the attacker
may reach.

## Reporting a vulnerability

Do **not** open a public issue for a security fault. Write to the maintainer,
encrypted, at `paul.j.reuer@gmail.com`; the fingerprint is published on the
releases page. One plain text per message; a reply within three working days,
a fix within ninety days, coordinated disclosure on publication.

If the fault is in a dependency — torch, transformers, vllm — report it
upstream as well; C2C pins minimum versions and will adopt the patched.

## The attack surface, honestly told

1. **The served front** (`c2c-serve`). It answers on a port. On the
   loopback (the default) it may stand open, as the pipe there is the
   trust; off the loopback it will **not start without a key** — the same
   rule as the bridge and the MCP server over HTTP. With `--certfile` and
   `--keyfile` it speaks HTTPS; a certificate from an authority, not a
   self-signed one, for production. The API key, when set, is checked with
   a constant-time comparison (`hmac.compare_digest`) — no oracle there,
   no early-exit leak on the length of the key. Internal faults go to the
   server log; the client sees the class of the fault, not its details.

2. **The caches on the wire.** A KV-cache is a tensor, and a tensor is not
   code — but a tensor is data, and data is how this library rolls. A cache
   received over the wire is checked against the geometry it declares: the
   shapes, the count of layers, the width of the hidden state (FR-15, the
   wire contract). A cache that lies about its shape is refused, not fused.
   The fuser's projections are affine, and affine maps do not execute
   instructions; still, deserialising weights from an untrusted source is
   `torch.load(..., weights_only=True)` and nothing more — with one
   allowlist, the `c2c.types` enum members a checkpoint legitimately
   carries (`torch.serialization.safe_globals`), and no global else.
   With `--privacy` the sharer's cache is refused before it is attempted:
   the receiver answers alone, and the access log carries digests only.

3. **The zoo.** `c2c zoo publish` writes to a shared root, default
   `~/.cache/c2c/zoo`. The manifest is JSON; the sha256 of the weights,
   sealed at publish time, is re-hashed at every read — local shelf or hub
   download alike. The seals **fail closed**: a manifest that carries no
   seal is a manifest refused, and a blob whose hash does not match its
   seal is quarantined, not loaded. The MCP `c2c_register_pair` tool takes
   `weights_path` from within the zoo root only (`C2C_ZOO_ROOT` moves it),
   and only `.pt` files: an agent's weights come from the zoo, or from
   nowhere. Do not publish private data to the zoo: the root is not a
   secret store. For secrets there is the environment, and the access and
   secrecy bits of it: `C2C_API_KEY`, `C2C_ZOO_ROOT`, `C2C_TRUSTED_PUBLISHERS`.

4. **The agent bridge** (`c2c-a2a`). The card is public, by the protocol's
   design; the tasks are not. Each task is bound to the **principal** that
   submitted it — the bearer token of the house — and no other peer may
   read it, cancel it, or count it; `tasks/list` speaks of one
   `context_id`, of one principal, never of the whole store. One key is
   one house, one parlour: run one bridge per house, or serve the house
   with care. The bridge will not bind an address off the loopback without
   a key. It forwards to the front; beyond the living tasks it stores
   nothing. Bearer tokens are checked in constant time; the audience of
   the card is the `--public-url`, and that URL is not a directive from
   the peer.

5. **The MCP server** (`c2c-mcp`). JSON-RPC over stdio, and stdio is a pipe
   with no authentication of its own: whoever can write to the pipe is the
   client. Run it only beside clients you trust. The HTTP transport
   (`--transport http`) serves the loopback without a key as a pipe of the
   same trust; off the loopback a key is required to start, the JSON-RPC
   core answers each request on its own thread with no shared mutable
   state, tool faults return the fault's class to the caller and its
   details to the server log, and the disk-touching tool argument
   (`weights_path`) reaches no further than the zoo root. TLS rides it with
   `--certfile`/`--keyfile` — a bearer key on the plain wire belongs to the
   loopback, not to a network.

## The paper's limitations, as security concerns

The paper §5 names it: a weak sharer can degrade a strong receiver. In
serving terms, that is an availability and integrity concern, not merely a
quality one. C2C's answer is defence in depth:

- the gates (`c2c.diagnostics.gates`) — the effective rank of the fused
  cache is measured, and a degenerate fusion is reported, not hidden;
- the gate values (`c2c.fuser.gating`) — the learned decision to inject or
  not to inject, per layer, per token;
- the fusion mode (`c2c.config`) — `--privacy` on the command line: no
  cache, no trace, the sharer's context dropped, the receiver answering
  alone. It is a flag, and it is off by default; set it, and the fusion is
  refused before it is attempted.
- the policy of the list — `gate=block`: a named sharer, blocked. The
  weak, silenced; the strong, heard.

## Supply chain

- torch and friends are optional extras; the core does not import them at
  the top of any module. A build with no network builds the core only.
- The CI matrix runs without internet, except fixtures (see `c2c eval`).
- Every release is tagged, and the tag is signed; the artefacts on PyPI
  carry the same sha256 as the git archive, and `c2c doctor --report` will
  tell you which one you have.
