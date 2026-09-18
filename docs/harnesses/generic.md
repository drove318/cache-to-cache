# C2C, with any harness at all

The title says it. C2C speaks the OpenAI wire protocol; any harness that
can point at an OpenAI-compatible endpoint — Langchain, LlamaIndex, an
editor plugin, a shell, a `curl` in a cron job — is a harness C2C can
serve. No harness owns the agent loop; none ever shall.

## The three words of the contract

1. **Base URL**: `http://HOST:PORT/v1` (or `https://`, when the front is
   given `--certfile` and `--keyfile`). Trailing slash, as the client
   expects; the relay is liberal about it.
2. **Key**: `authorization: bearer $C2C_API_KEY`, when the front was
   started with a key. Without a key, bind to `127.0.0.1` — the default —
   and to nothing else.
3. **Model name**: `c2c/RECEIVER+SHARER` names the fused pair;
   `c2c/RECEIVER` names the solo run. The routes: `/v1/models`,
   `/v1/chat/completions`, `/v1/completions`, `/healthz`.

## The generic snippet

```bash
# the front, up
c2c-serve --pair math-small:coder-small -e hf --port 8121 \
          --api-key "$C2C_API_KEY" &

# the client, any client, the OpenAI shape
python - <<'EOF'
from openai import OpenAI
import os
client = OpenAI(base_url="http://127.0.0.1:8121/v1",
                api_key=os.environ["C2C_API_KEY"])
answer = client.chat.completions.create(
    model="c2c/math-small+coder-small",
    messages=[{"role": "user", "content": "prove: two plus two equals four"}])
print(answer.choices[0].message.content)
EOF
```

The Python here is the OpenAI SDK's own; swap it for `curl`, for the
TypeScript SDK, for the Rust crate — the wire does not know the difference,
and does not care.

## The MCP route, for tool-using agents

If the harness speaks the Model Context Protocol instead of (or beside) the
completion API, mount the MCP server:

```bash
c2c-mcp                              # stdio: for a client on the pipe
c2c-mcp --transport http --port 8789 # or over a port, JSON-RPC on POST /
```

Tools: `c2c_register_pair` (declare the collaboration), `c2c_fuse` (the
report of one fusion: gates, ranks, tokens), `c2c_ask` (the full pipeline:
resolve, capture, fuse, generate). The agent calls the tools; the tools
call the fusion. The loop remains the agent's — see `man c2c-mcp`.

## The A2A route, agent unto agent

For agent-to-agent meshes, the bridge publishes the card at the well-known
path (`GET /.well-known/agent-card.json`) and carries tasks as JSON-RPC
(`POST /tasks`: `message/send`, `tasks/get`, `tasks/list`, `tasks/cancel`).
The card is public, the tasks are not — `man c2c-a2a`:

```bash
c2c-a2a --pair math-small:coder-small -e hf --port 8122 \
        --public-url https://relay.example.com:8122
```

## Fault notes, generic

- *401*: the key, mismatched; compare `C2C_API_KEY` at both ends.
- *404 on the model*: the pair is not registered; `--pair` at the start.
- *connection refused*: the front is down; `c2c doctor --report`, last.

Everything else: `SECURITY.md` for the trust model, `man c2c-proxy(7)` for
the wire, `man c2c(1)` for the console.
