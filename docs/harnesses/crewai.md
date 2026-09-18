# C2C with CrewAI

CrewAI's agents run on an OpenAI-compatible `LLM`; C2C is that LLM — a
pair of models speaking as one, through fused caches. The crew, the
tasks, the process — CrewAI's. The wire between the models — C2C's.

## 1. The front, up

```bash
c2c-serve --pair planner-small:researcher-small -e hf --port 8121 \
          --api-key "$C2C_API_KEY"
```

## 2. The crew, pointed

```python
import os
from crewai import Agent, Crew, Task
from crewai.llms import LLM

llm = LLM(model="c2c/planner-small+researcher-small",     # the pair, as one
          base_url="http://127.0.0.1:8121/v1",
          api_key=os.environ["C2C_API_KEY"])

planner   = Agent(role="planner",   goal="plan the work",  backstory="…", llm=llm)
researcher = Agent(role="researcher", goal="find the facts", backstory="…", llm=llm)

crew = Crew(agents=[planner, researcher],
            tasks=[Task(description="…", expected_output="…", agent=planner)])
crew.kickoff()
```

Wherever your CrewAI build takes an `LLM`, the same three fields carry:
`model`, `base_url`, `api_key`. The crew does not know the difference
between one model and a collaboration; the log does.

## 3. The proof

```bash
curl http://127.0.0.1:8121/healthz                  # {"status": "ok"}
curl http://127.0.0.1:8121/v1/models \
     -H "authorization: bearer $C2C_API_KEY"          # the crew, on the shelf
```

Each exchange logs in the front's access line, with the word
`fused=true` where the cache crossed.

## Fault notes

- *401*: the key, mismatched; compare `C2C_API_KEY` at both ends.
- *model not found*: the pair is unregistered; `--pair` at the front's start.

See also `man c2c-serve`, `docs/harnesses/generic.md`.
