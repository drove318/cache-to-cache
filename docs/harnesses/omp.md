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
     -e reference --host 127.0.0.1 --port 8788
```

Real models ride the same two commands. With `-e hf` (`pip install
transformers`; the adapter builds both models itself), the two halves of the
pair are Hugging Face ids or local model folders:

```bash
c2c-serve --pair Qwen/Qwen3-0.6B+Qwen/Qwen2.5-Math-1.5B:fuser.safetensors \
     -e hf --host 127.0.0.1 --port 8788 --api-key "$(openssl rand -hex 32)"
```

## 2. The provider, registered

Paste this into your terminal — it writes `c2c:` into `~/.omp/agent/models.yml`
alongside your existing providers (creating the file if you have none, keeping a
`.bak`, refusing a second paste), leaves every other field at omp's defaults,
then asks omp itself to confirm:

```bash
python3 - <<'EOF'
import os, shutil
p = os.path.expanduser("~/.omp/agent/models.yml")
block = """    c2c:
        api: openai-completions
        auth: none
        baseUrl: http://127.0.0.1:8788/v1
        models:
            - id: receiver-mini+sharer-mini
              name: C2C demo pair (toy models, fused wire)

"""
os.makedirs(os.path.dirname(p), exist_ok=True)
src = open(p).read() if os.path.isfile(p) else ""
if "\n    c2c:" in "\n" + src:
    print("c2c already registered; nothing changed")
else:
    if src:
        shutil.copy(p, p + ".bak")
    lines = src.splitlines(True)
    if lines and lines[0].rstrip() == "providers:":
        lines.insert(1, block)
    else:
        lines.insert(0, "providers:\n" + block)
    open(p, "w").writelines(lines)
    print("c2c merged" + (", backup at " + p + ".bak" if src else ""))
EOF
omp models find c2c
```

The table listing `receiver-mini+sharer-mini` is the proof. Restart omp —
config loads at startup — and the pair appears in the model picker like any
other entry. Off-localhost fronts: TLS (`--certfile/--keyfile`) plus
`--api-key` at the start, and swap the entry's `auth: none` for
`apiKey: <that key>` + `authHeader: true`.

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
