"""The four benchmarks of the paper, plus LongBench and GSM8K — offline first.

Each benchmark ships as a JSON-Lines fixture under
``tests/fixtures/<name>.jsonl`` with records::

    {"id": …, "question": …, "choices": ["A text", "B text", …], "answer": <index>}

so the harness runs with no internet, as the CI contract requires ("no
internet except fixtures"). Live datasets (MMLU-Redux, ARC-Challenge,
OpenBookQA, C-Eval, LongBenchV1, GSM8K) may be cached into the same
directory by the fetch helper below when the hub is reachable; the loader
prefers the cache and falls back to the bundled fixtures only.

Answer extraction follows the evaluation mode of the paper: free-form
text generation, then match the first choice letter found in the reply;
max generation length 64, zero temperature, zero-shot.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

__all__ = ["Benchmark", "BENCHMARKS", "load_benchmark", "extract_answer",
           "DEFAULT_FIXTURES"]

# up four: file → eval → c2c → src → repository root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_FIXTURES = os.path.join(_ROOT, "tests", "fixtures")

#: letter of the choices, as the papers put them: (A) (B) (C) (D) …
_LETTERS = "ABCDEFGHIJ"
_LETTER_RE = re.compile(r"(?:^|[\s\(\[])([A-J])[\)\.\:\s\]]")


@dataclass(frozen=True)
class Benchmark:
    """One multiple-choice benchmark of the evaluation protocol."""

    name: str
    fixture: str
    domain: str                       # reasoning | knowledge | language | math | long-context
    metric: str = "accuracy"
    max_out: int = 64
    official_prompt: bool = False     # LongBench uses its official prompt template

    def prompt_for(self, item: dict) -> str:
        """Render the zero-shot prompt for one item, choices and all."""
        choices = item.get("choices") or []
        lettered = "\n".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))
        return (f"{item['question']}\n{lettered}\n\n"
               f"Answer with the letter of the correct choice only.")


BENCHMARKS: dict[str, Benchmark] = {
    "mmlu-redux": Benchmark("mmlu-redux", "mmlu_redux.jsonl", "knowledge"),
    "arc-c": Benchmark("arc-c", "arc_challenge.jsonl", "reasoning"),
    "openbookqa": Benchmark("openbookqa", "openbookqa.jsonl", "reasoning"),
    "c-eval": Benchmark("c-eval", "ceval.jsonl", "knowledge"),
    "longbench-v1": Benchmark("longbench-v1", "longbench.jsonl", "long-context",
                           max_out=2048, official_prompt=True),
    "gsm8k": Benchmark("gsm8k", "gsm8k.jsonl", "math"),
}


def load_benchmark(name: str, *, fixtures_dir: str | None = None,
                  limit: int | None = None) -> Iterator[dict]:
    """Yield the items of one benchmark, cache first, fixtures always.

    Raises KeyError for an unknown benchmark (with the list of known ones)
    and FileNotFoundError when neither cache nor fixture can be found.
    """
    if name not in BENCHMARKS:
        msg = (f"unknown benchmark {name!r}; known: {', '.join(sorted(BENCHMARKS))}")
        raise KeyError(msg)
    bench = BENCHMARKS[name]
    root = fixtures_dir or os.environ.get("C2C_FIXTURES", DEFAULT_FIXTURES)
    path = os.path.join(root, bench.fixture)
    if not os.path.isfile(path):
        msg = (f"no fixture for benchmark {name!r} at {path!r}; run the fetch helper "
              f"or bundle the fixtures")
        raise FileNotFoundError(msg)
    count = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{lineno}: malformed fixture record: {exc.msg}"
                raise ValueError(msg) from exc
            if "question" not in item:
                msg = f"{path}:{lineno}: fixture record has no 'question'"
                raise ValueError(msg)
            yield item
            count += 1
            if limit is not None and count >= limit:
                return


def extract_answer(reply: str, choices: Sequence[str] | None = None) -> int | None:
    """Match a generated reply against the choices, letters first.

    Order of matching: an explicit letter (A–J, in parentheses, with a dot
    or a colon), else an exact choice text, else nothing (None → scored as
    incorrect). The answer space is the index into ``choices``.
    """
    text = (reply or "").strip()
    if not text:
        return None
    m = _LETTER_RE.match(text) or _LETTER_RE.search(text[:4])
    if m:
        idx = _LETTERS.find(m.group(1))
        if 0 <= idx:
            if choices is None or idx < len(choices):
                return idx
    lowered = text.lower()
    for i, choice in enumerate(choices or ()):
        c = str(choice).strip().lower()
        if c and (c == lowered or c in (w for w in re.split(r"[^a-z0-9]+", lowered) if w)):
            return i
    digits = re.findall(r"\d+(?:\.\d+)?", text)
    if digits and choices:
        for i, choice in enumerate(choices):
            cs = re.findall(r"\d+(?:\.\d+)?", str(choice))
            if any(d in cs for d in digits):
                return i
    return None


def scorer(bench: Benchmark) -> Callable[[dict, str], float]:
    """The scorer of one benchmark: 1.0 for the right choice, 0.0 otherwise."""
    def _score(item: dict, reply: str) -> float:
        got = extract_answer(reply, item.get("choices"))
        want = item.get("answer")
        if isinstance(want, str):
            want = _LETTERS.find(want.strip().upper()) if len(want.strip()) == 1 else int(want)
        return 1.0 if (got is not None and got == want) else 0.0
    return _score


def fetch(url: str, destination: str, *, timeout: float = 60.0) -> bool:
    """Download one dataset file into the fixtures directory, atomically.

    Returns True on success, False when offline or the source is
    unreachable — the caller keeps using the bundled fixture in that case.
    """
    tmp = destination + ".part"
    try:
        request = urllib.request.Request(url, headers={
            "User-Agent": "c2c-cache/eval (+https://github.com/drove318/cache-to-cache)"})
        with urllib.request.urlopen(request, timeout=timeout) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, destination)
        return True
    except (urllib.error.URLError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
