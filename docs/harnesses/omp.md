# C2C with Oh My Pi (omp)

omp is the harness this library was written beside: C2C mounts into omp as
a model provider, and never touches the agent loop, the tools, or the
session. The wire between models is the only thing C2C is.

## 1. The front, up

```bash
c2c-serve --pair math-small:coder-small -e hf --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. The provider, registered

In omp's configuration, add the C2C provider beside the native ones —
`base_url` + key, the OpenAI wire, no fork, no patch:

```json
{
  "providers": {
    "c2c": {
      "kind": "openai-compatible",
      "base_url": "http://127.0.0.1:8121/v1",
      "api_key": "${C2C_API_KEY}"
    }
  },
  "default_provider": "c2c"
}
```

## 3. The models, by name

Address the fused pair as one model wherever omp takes a model name:

```text
/model c2c/math-small+coder-small
```

The completion arrives from the fused state; the tools, the loop, the
history — omp's, unchanged.

## Fault notes

- *provider unreachable*: the front is down; `c2c doctor --report` says why.
- *model not found*: the pair is unregistered; `--pair` at the front's start.

See also `man c2c-serve`, `man c2c-proxy`, `docs/harnesses/generic.md`.
