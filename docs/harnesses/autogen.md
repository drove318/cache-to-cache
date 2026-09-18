# C2C with AutoGen

AutoGen speaks OpenAI-compatible endpoints natively; C2C is that endpoint.
The conversation, the agents, the group chat — AutoGen's. C2C only changes
how the models behind them talk to one another: cache-to-cache, not
text-to-text.

## 1. The front, up

```bash
c2c-serve --pair writer-small:critic-small -e sglang --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. The client, pointed

AutoGen's OpenAI-compatible client is configured by the environment and the
model name — the three words of the contract, nothing more:

```python
import os
from autogen import AgentChat
from autogen.oai import OpenAIWrapper

os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8121/v1"
os.environ["OPENAI_API_KEY"]  = os.environ["C2C_API_KEY"]

OpenAIWrapper.api_base = os.environ["OPENAI_API_BASE"]
OpenAIWrapper.api_key  = os.environ["OPENAI_API_KEY"]

writer = AgentChat(name="writer",
                   model="c2c/writer-small+critic-small")   # the pair, as one
```

Wherever your AutoGen build takes an OpenAI client, the same three fields
carry: base URL, key, model. The version of the SDK is neither C2C's
concern nor yours: the wire is the wire.

## 3. The proof

```bash
curl http://127.0.0.1:8121/healthz                    # {"status": "ok"}
curl http://127.0.0.1:8121/v1/models \
     -H "authorization: bearer $C2C_API_KEY"          # the pair, on the shelf
```

The group chat answers; the server log reports the exchanges:

```text
[c2c-serve] chat/completions model=c2c/writer-small+critic-small fused=true ...
```

## Fault notes

- *401*: the key, mismatched between the two ends.
- *stream stalls*: SSE is default for chat; the fault then is the client's,
  not the relay's — `man c2c-proxy`.

See also `man c2c-serve`, `docs/harnesses/generic.md`.
