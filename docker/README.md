# the completions image: c2c-serve, baked

Build from the root of the checkout:

```bash
docker build -f docker/Dockerfile.completions -t c2c/completions:latest .
```

Run the front, on a pair, with the key from the environment:

```bash
docker run --rm --publish 8121:8121 \
       --env C2C_API_KEY="$(openssl rand -hex 16)" \
       --volume "$PWD/models:/models:ro" \
       c2c/completions:latest \
       --pair math-small:coder-small -e hf --host 0.0.0.0 --port 8121
```

The image listens on all interfaces **only because you said so** with
`--host 0.0.0.0`; the default bind of the binary is the loopback. Between
the container and the world, put a TLS terminator: pass `--certfile` and
`--keyfile`, or terminate at a reverse proxy that speaks HTTPS.

The weights are a volume, not a layer: models do not belong in images.
`/models` is where the engine adapters look, see `man c2c-engines`.
