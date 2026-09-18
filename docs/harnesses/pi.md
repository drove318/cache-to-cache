# C2C with Pi

Pi keeps its agent loop, its tools, its session. C2C inserts nowhere in
that ledger; C2C only serves the model endpoint — and through that endpoint,
two models answer as one.

## 1. The front, on the port

```bash
c2c-serve --pair answer-small:draft-small -e llamacpp \
          --host 127.0.0.1 --port 8121 \
          --api-key "$C2C_API_KEY"
```

(llama.cpp through the C2C adapter; the GGML builds, if present, load
themselves — see `man c2c-engines`.)

## 2. The settings, in Pi

Pi reads an OpenAI-compatible provider from its settings file:

```json
{
  "providers": {
    "c2c": {
      "baseUrl": "http://127.0.0.1:8121/v1",
      "apiKey": "${C2C_API_KEY}",
      "model": "c2c/answer-small+draft-small"
    }
  },
  "activeProvider": "c2c"
}
```

## 3. The test of it

```bash
pi "what is the capital of australia?"
# canberra — and in the server log: fused=true, gates reported per layer
```

To compare the fused answer against the plain one, run the same question
with the solo receiver (`"model": "answer-small"`) — the difference is the
cache arriving where the text did not.

## Fault notes

- *time out, connect*: the front is not up; `c2c doctor --report`.
- *model not found*: `--pair` at the start names the pair; `/v1/models`
  lists what the front will serve.

See also `man c2c-serve`, `docs/harnesses/generic.md`.
