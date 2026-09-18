# C2C with OpenCode

OpenCode is an open terminal agent; like its kin, it points at a model
provider and expects the OpenAI wire. C2C is that provider, and the pair of
models reads as one model.

## 1. Serve the pair

```bash
pip install "c2c-cache[train]"
c2c-serve --pair coder-1b:reviewer-3b -e mlx \
          --host 127.0.0.1 --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. Configure OpenCode

In OpenCode's configuration (`opencode.json`), the provider block:

```json
{
  "$schema": "https://opencode.dev/schema.json",
  "model": "c2c/coder-1b+reviewer-3b",
  "provider": {
    "c2c": {
      "npm": "@ai/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8121/v1",
        "apiKey": "{env:C2C_API_KEY}"
      }
    }
  }
}
```

## 3. The proof, in the log

```text
opencode> refactor src/parser to drop the global state
c2c-serve: /v1/chat/completions  fused=true  receiver=coder-1b  sharer=reviewer-3b  layers=12/12
```

`fused=true` is the whole of the claim: the reviewer's cache reached the
coder's decode without a single token between them.

## Fault notes

- *ECONNREFUSED*: the front is down, or the port is taken — `--port`.
- *401*: the key differs at the two ends of the wire.

See also `man c2c-serve`, `man c2c-engines` (the MLX adapter, and its
Apple-silicon way), `docs/harnesses/generic.md`.
