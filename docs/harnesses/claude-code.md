# C2C with Claude Code

Claude Code speaks to models through an OpenAI-compatible endpoint when
configured with a custom base URL. C2C is that endpoint; the pair of models
becomes one model name.

## 1. The front, up

```bash
c2c-serve --pair code-small:review-small -e vllm \
          --host 127.0.0.1 --port 8121 \
          --certfile ~/certs/cert.pem --keyfile ~/certs/key.pem \
          --api-key "$C2C_API_KEY"
```

## 2. The environment, set

Claude Code reads the OpenAI conventions from the environment:

```bash
export OPENAI_BASE_URL="https://127.0.0.1:8121/v1"
export OPENAI_API_KEY="$C2C_API_KEY"
claude code --model "c2c/code-small+review-small" "read src/, fix the failing tests"
```

(If your build names the variables differently — `ANTHROPIC_...` for the
native route — set the OpenAI-compatible ones too; the relay listens on the
OpenAI wire.)

## 3. The works, verified

Ask for a completion through the pair, and confirm in the server log:

```text
c2c-serve: POST /v1/chat/completions  model=c2c/code-small+review-small  fused=true  layers=12  gates=0.61..0.88
```

The `fused=true` and the gate range say: the cache came, saw, and conquered
the text channel. No text between the models; the tools stay Claude's.

## Fault notes

- *401*: the key, mismatched between shell and server.
- *404, model not found*: the pair is not on the shelf; list it with
  `curl .../v1/models`, register it with `--pair`.
- *tls handshake*: a self-signed certificate must be in the trust store;
  or run plain HTTP on loopback only.

See also `man c2c-serve`, `man c2c-proxy`, `docs/harnesses/generic.md`.
