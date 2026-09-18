# C2C with LangChain

LangChain's OpenAI-compatible chat client is the door; C2C walks through
it. Chains, agents, tools, memory — LangChain's side of the wire. The
models' side of the wire becomes cache-to-cache.

## 1. The front, up

```bash
c2c-serve --pair reason-small:knowledge-small -e hf --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. The client, pointed

```python
import os
from langchain_openai import ChatOpenAI     # langchain-community package

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8121/v1",
    api_key=os.environ["C2C_API_KEY"],
    model="c2c/reason-small+knowledge-small",   # the pair, as one model
)

answer = llm.invoke("prove: two plus two equals four")
```

The chain drops in wherever a `ChatOpenAI` was: the three fields — base
URL, key, model — are the whole of the contract. Older layouts import
`ChatOpenAI` from `langchain.chat_models.openai`; the fields are the same.

## 3. The proof

```bash
curl http://127.0.0.1:8121/healthz                 # {"status": "ok"}
curl http://127.0.0.1:8121/v1/models \
     -H "authorization: bearer $C2C_API_KEY"        # the gallery, of pairs
```

The answer returns through the chain; the server log says `fused=true`,
and the mean is that the sharer's cache reached the receiver's decode
without one token of text between them.

## Fault notes

- *401*: the key, mismatched; compare `C2C_API_KEY` at both ends.
- *model not found*: the pair is unregistered; `--pair` at the front's start.

See also `man c2c-serve`, `man c2c-proxy`, `docs/harnesses/generic.md`.
