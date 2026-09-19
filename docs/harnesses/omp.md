# C2C with Oh My Pi (omp)

omp is the harness this library was written beside: C2C mounts into omp as
a model provider, and never touches the agent loop, the tools, or the
session. The wire between models is the only thing C2C is.

## 1. The front, up

From the clone (see the README's Quickstart — one paste gets the venv and
the install): train a wire for the built-in toy pair, then serve the pair
with that wire attached. The toy models answer gibberish — they are
random-init by design; the point is the `fused=true` in the front's own log
line, which says the cache actually travelled:

```bash
c2c train -d fixtures/datasets/tiny.jsonl \
     --receiver receiver-mini --sharer sharer-mini -e reference \
     --epochs 3 -o demo-fuser.safetensors
c2c-serve --pair receiver-mini+sharer-mini:demo-fuser.safetensors \
     -e reference --host 127.0.0.1 --port 8788 --api-key c2c-demo-key
```

Real models ride the same two commands. With `-e hf` (`pip install
transformers`; the adapter builds both models itself), the two halves of the
pair are Hugging Face ids or local model folders:

```bash
c2c-serve --pair Qwen/Qwen3-0.6B+Qwen/Qwen2.5-Math-1.5B:fuser.safetensors \
     -e hf --host 127.0.0.1 --port 8788 --api-key c2c-demo-key
```

## 2. The provider, registered

omp reads custom endpoints from `~/.omp/agent/models.yml` — the OpenAI wire,
no fork, no patch (`apiKey` names an environment variable or is the literal
key; `authHeader` sends it as `Authorization: Bearer`):

```yaml
providers:
  c2c:
    baseUrl: http://127.0.0.1:8788/v1
    api: openai-completions
    apiKey: c2c-demo-key
    authHeader: true
    models:
      - id: receiver-mini+sharer-mini
        name: C2C demo pair (toy models, fused wire)
        contextWindow: 2048
        maxTokens: 256
```

Restart omp to load it; `omp models find c2c` confirms it registered.

## 3. The models, by name

Address the fused pair as one model wherever omp takes a model name:

```text
/model c2c/receiver-mini+sharer-mini
```

The completion arrives from the fused state; the tools, the loop, the
history — omp's, unchanged.

## Fault notes

- *provider unreachable*: the front is down; `c2c doctor --report` says why.
- *model not found*: the pair is unregistered; `--pair` at the front's start.
- *`fused=false` in the front's log*: no trained fuser is attached to that
  pair — the front answers from the receiver alone and says so. Train one,
  or pass `pair:weights.safetensors` at `--pair`.
