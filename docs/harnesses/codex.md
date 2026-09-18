# C2C with Codex

Codex addresses its model provider by configuration; C2C is a provider that
answers on the OpenAI wire. The agent stays Codex's; the cache becomes the
wire's.

## 1. Start the front with the pair you mean to fuse

```bash
c2c-serve --pair gpt-family-small:coder-tiny -e hf \
          --host 127.0.0.1 --port 8121 \
          --api-key "$C2C_API_KEY" --privacy off
```

`--privacy off` is the default and says: the sharer's context may be fused.
With `--privacy on` the sharer's cache is dropped and the receiver answers
alone — the paper's limitation, honoured (F-17).

## 2. Configure Codex

`~/.codex/config.toml`:

```toml
[model_providers.c2c]
base_url = "http://127.0.0.1:8121/v1"
api_key  = "${C2C_API_KEY}"
name     = "openai-compatible"

[model]
provider = "c2c"
model    = "c2c/gpt-family-small+coder-tiny"
```

## 3. Run, and read the proof

```bash
codex "explain the theorem of pythagoras, briefly"
```

In the server log the line says `fused=true` with the per-layer gate values;
in the answer, the gain the paper measured — 6.4 % to 14.2 % over the
stronger model alone, Tables 3–8 (`c2c eval` replays them against the
fixtures).

## Fault notes

- *provider unreachable*: the front is down; `c2c doctor` says why.
- *invalid model name*: the name must read `c2c/RECEIVER+SHARER`, or plain
  `RECEIVER` for the solo run.

See also `man c2c-serve`, `docs/harnesses/generic.md`.
