# C2C with Hermès

Hermès drives its models through an OpenAI-compatible endpoint. C2C serves
that exact shape; point Hermès at the wire and the models speak cache-to-cache.

## 1. Serve the pair

```bash
python3 -m venv .venv && . .venv/bin/activate    # the system pip is PEP 668-locked
pip install "c2c-cache[train] @ git+https://github.com/drove318/cache-to-cache.git"
c2c-serve --pair math-small:coder-small -e hf \
          --host 127.0.0.1 --port 8121 \
          --certfile certs/cert.pem --keyfile certs/key.pem \
          --api-key "$C2C_API_KEY"
```

## 2. Configure Hermès

In Hermès' model settings (its `providers` block), add one provider:

```yaml
providers:
  c2c:
    base_url: https://127.0.0.1:8121/v1     # no trailing slash
    api_key: ${C2C_API_KEY}
    models:
      - c2c/math-small+coder-small            # the fused pair, as one model
```

Nothing else changes: the loop, the tools, the session — Hermès' own, as
always. C2C is the wire, not the loom.

## 3. Verify the wire

```bash
curl -sk https://127.0.0.1:8121/healthz                 # {"status": "ok"}
curl -sk https://127.0.0.1:8121/v1/models \
     -H "authorization: bearer $C2C_API_KEY"             # the pair, on the shelf
```

Then ask Hermès something arithmetic, and watch the answer come from the
fused state — no text between the models, only cache.

## Fault notes

- *401, the key amiss*: set `C2C_API_KEY` the same at both ends.
- *certificate, untrusted*: a self-signed cert needs the CA bundle on the
  client side, or serve plain HTTP on a loopback you own.
- *model not found*: the pair is not registered — see `c2c-serve --pair`,
  and the `[pair]` sections of the config file (`man c2c.config`).

See also `man c2c-serve`, `man c2c-proxy`, `docs/harnesses/generic.md`.
