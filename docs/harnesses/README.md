# The harnesses, one and all

C2C is middleware: it bolts onto the harness you already run, and changes
the way the models speak to one another — from text to cache. This shelf
holds the recipes, one leaf per harness, in the house voice of the project.

| leaf                | harness            | the way in                                        |
|-------------------|--------------------|---------------------------------------------------|
| `hermes.md`       | Hermès             | `providers:` block, the `c2c` provider            |
| `omp.md`          | Oh My Pi (omp)     | the provider registry, `kind: openai-compatible` |
| `claude-code.md`  | Claude Code        | environment, `OPENAI_BASE_URL` and kin            |
| `codex.md`        | Codex              | `config.toml`, `model_providers.c2c`              |
| `autogen.md`      | AutoGen            | the OpenAI-compatible client, the three fields    |
| `langchain.md`    | LangChain          | `ChatOpenAI`, base URL, key, model                |
| `crewai.md`       | CrewAI             | the `LLM`, model, base URL, key                    |
| `generic.md`      | any of them        | the three words: base URL, key, model name        |

The pattern is always the same, because the contract is always the same:

1. `c2c-serve --pair RECEIVER+SHARER -e ENGINE` — the front, on a port, speaks the
   OpenAI wire. On the loopback it may stand open; off it, the key is
   required (SECURITY.md, the honest surface).
2. The harness points at the front (base URL `.../v1`, bearer key).
3. The harness addresses one model: `c2c/RECEIVER+SHARER`. The fusion is
   the middleware's affair; the agent loop is nobody's to change.

Where a harness speaks MCP instead, mount `c2c-mcp` (`man c2c-mcp`); where
agents speak to agents, the bridge is `c2c-a2a` (`man c2c-a2a`).

The trust model, the exposure surface, and the privacy switch are treated
honestly in `SECURITY.md`. The paper is the normative basis; where the
paper and a harness disagree, the paper wins, and the harness should be
told so in an issue.

*Read the man pages, read the paper; make love, make it pass all tests.*
