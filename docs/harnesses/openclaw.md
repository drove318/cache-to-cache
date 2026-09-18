# C2C with OpenClaw

OpenClaw reaches its models over an OpenAI-shaped base URL. C2C answers that
shape. No claw, no loop: OpenClaw keeps its agent; C2C keeps the cache.

## 1. Serve the pair, and tell the claw where the door is

```bash
c2c-serve --pair reason-small:act-small -e sglang \
          --host 127.0.0.1 --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. Point OpenClaw at the door

In OpenClaw's configuration (`config.toml`, its `[model]` stanza):

```toml
[model]
provider  = "openai-compatible"
base_url  = "http://127.0.0.1:8121/v1"     # https, when --certfile is given
api_key   = "${C2C_API_KEY}"
model     = "c2c/reason-small+act-small"    # the pair reads as one model
```

## 3. Proof of the fuse

```bash
c2c doctor --report          # the state of the house, in one screen
c2c eval --table 4           # what the fusion is worth, at this geometry
```

OpenClaw will say its tools as before; the completion now travels a shorter
path — the sharer's context reaches the receiver as cache, not as tokens.

## Fault notes

- *stream stalls*: SSE is on by default for chat; if OpenClaw buffers, the
  fault is the client's, not the relay's — see `man c2c-proxy`.
- *pair unknown*: register it in the server's `[pair]` sections, or pass
  `--pair` at the start.

See also `man c2c-serve`, `docs/harnesses/generic.md`, `SECURITY.md`.
