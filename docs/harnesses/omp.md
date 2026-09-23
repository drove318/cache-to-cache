# C2C with Oh My Pi (omp)

omp is the harness this library was written beside: C2C mounts into omp as
a model provider, and never touches the agent loop, the tools, or the
session. The wire between models is the only thing C2C is.

## 1. The front, up

The featured road, as walked: a wired vLLM container on :8888 carrying
`qwen3.8-flash-next`, the front speaking OpenAI to it, and omp to the
front. The full walk — the plain server first, the identity proof while
it stands, the launch that mounts the connector and parks the plain —
is [`road-b/`](../../road-b/README.md); the two commands that finish
are:

```bash
GATE=closed road-b/launch_wired.sh
./c2c-venv/bin/c2c-serve --engine vllm-wired \
    --url http://127.0.0.1:8888 \
    --pair 'qwen3.8-flash-next←qwen3.8-flash-next' \
    --host 127.0.0.1 --port 8788
```

The front answers on :8788 with the pair, the gallery, and the access
log; `fused=false` at the gate, which holds until the wire is fitted
(`road-b/train_handoff.md`, the two roads that can). To substitute
another model, name it in `--pair` and at the harness; to reach a
different server, give `--url`; the caller need not `--stop` — the
front's default port of 8788 can be bypassed for `--port` switches.

No wired server in sight? The miniature pair is the crossbar, and the
CI will not budge:

```bash
c2c train -d fixtures/datasets/tiny.jsonl \
     --receiver receiver-mini --sharer sharer-mini -e reference \
     --epochs 3 -o demo-fuser.safetensors
c2c-serve --pair receiver-mini+sharer-mini:demo-fuser.safetensors \
     -e reference --host 127.0.0.1 --port 8788
```

The toy models answer gibberish — random-init by design; the point is
the `fused=true` in the front's own log line, which says the cache
travelled. Real HF models ride the same two commands with `-e hf`
(`pip install transformers`; the adapter builds both halves itself):

```bash
c2c-serve --pair Qwen/Qwen3-0.6B+Qwen/Qwen2.5-Math-1.5B:fuser.safetensors \
     -e hf --host 127.0.0.1 --port 8788 --api-key "$(openssl rand -hex 32)"
```

## 2. The provider, registered

Paste this into your terminal — it writes `c2c:` into `~/.omp/agent/models.yml`
alongside your existing providers (creating the file if you have none, keeping a
`.bak`, refusing a second paste), takes each model's window from the live
front where the served server can state it, leaves every other field at
omp's defaults,
then asks omp itself to confirm:

```bash
python3 - <<'EOF'
import json, os, shutil, urllib.error, urllib.request
base = os.environ.get("C2C_BASE", "http://127.0.0.1:8788/v1")
try:                                          # the numbers, from the live front
    with urllib.request.urlopen(base + "/models", timeout=5) as response:
        gallery = json.load(response)["data"]
except OSError:
    print("the front does not answer at", base)
    print("start it first — the command of section 1, or simply:")
    print("  c2c-serve --pair receiver-mini+sharer-mini")
    raise SystemExit(2)
if not gallery:
    print("the front serves no models: register a pair when you start it, e.g.")
    print("  c2c-serve --pair receiver-mini+sharer-mini          # toy pair, no downloads")
    print("  c2c-serve --pair Qwen/Qwen3-0.6B+Qwen/Qwen2.5-Math-1.5B -e hf   # real pair")
    raise SystemExit(2)
rows = sorted({((lambda r: r[4:] if r.startswith("c2c/") else r)(str(m.get("id", ""))),
                str(m.get("description") or ""), m.get("max_context_tokens"))
               for m in gallery if m.get("id")})
block = "    c2c:\n        api: openai-completions\n        auth: none\n"
block += f"        baseUrl: {base}\n        models:\n"
for mid, desc, ctx in rows:                  # the ids and notes the front prints itself
    block += (f"            - id: {json.dumps(mid)}\n"
              f"              name: {json.dumps(desc or 'C2C ' + mid)}\n")
    if ctx:                                  # the window, from the served's own card:
        block += f"              contextWindow: {int(ctx)}\n"  # never from a default
block += "\n"
p = os.path.expanduser("~/.omp/agent/models.yml")
os.makedirs(os.path.dirname(p), exist_ok=True)
src = open(p).read() if os.path.isfile(p) else ""
if "\n    c2c:" in "\n" + src:
    print("c2c already registered; nothing changed (paste again after removing its block)")
else:
    if src:
        shutil.copy(p, p + ".bak")
    lines = src.splitlines(True)
    if lines and lines[0].rstrip() == "providers:":
        lines.insert(1, block)
    else:
        lines.insert(0, "providers:\n" + block)
    open(p, "w").writelines(lines)
    print(f"c2c merged, {len(rows)} model(s) taken from the live front: "
          + ", ".join(mid for mid, _desc, _ctx in rows)
          + (", backup at " + p + ".bak" if src else ""))
EOF
omp models find c2c
```

The table, listing the ids the paste just read off the front, is the proof.
Restart omp — config loads at startup — and the pair appears in the model
picker like any other entry. Off-localhost fronts: TLS (`--certfile/--keyfile`)
plus `--api-key` at the start, and set `C2C_BASE` before the paste so the
entry carries the right URL, then swap the entry's `auth: none` for
`apiKey: <that key>` + `authHeader: true`.

## 3. The models, by name

Address one fused pair as one model, wherever omp takes a model name — the
ids below are the demo pair's; use what `omp models find c2c` lists for you:

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
