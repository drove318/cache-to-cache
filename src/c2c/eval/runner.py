"""The runner: drive any pair of engines through the evaluation protocol.

``run(...)`` executes the paper's protocol — zero-shot, greedy, T=0, max
response 64, communication 256 — over a benchmark fixture, for both the
single models and the collaboration, and reports the table as the paper
would print it. ``compare(...)`` matches a measured table against the
golden values and yields ``(check, expected, got, delta, pass)`` rows,
the way the unittest test runner does it.
"""

from __future__ import annotations

import time
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field

from .benchmarks import BENCHMARKS, load_benchmark, scorer
from .golden import TOLERANCE

__all__ = ["EvalResult", "run", "compare"]


@dataclass
class EvalResult:
    """Scores and timings of one evaluation run, as published in Table 4/7."""

    mode: str  # receiver-only | sharer-only | t2t | c2c | routing
    benchmark: str
    accuracy: float = 0.0  # percent, as in the paper's tables
    total_time_s: float = 0.0
    items: int = 0
    per_item: list[float] = field(default_factory=list)

    def __str__(self):
        return (
            f"{self.mode:<13} {self.benchmark:<12} acc {self.accuracy:0.2f} "
            f"({self.items} items, {self.total_time_s:0.1f} s)"
        )


def run(
    bench_name: str,
    ask: Callable[..., str],
    *,
    mode: str = "eval",
    max_new_tokens: int | None = None,
    limit: int | None = None,
    fixtures_dir: str | None = None,
) -> EvalResult:
    """Evaluate one ``ask`` callable on one benchmark, in one mode.

    ``ask(prompt) -> reply`` is the only thing the runner needs of the
    engine: prompt in, text out. The runner owns protocol (prompt template,
    scoring, timing), the engine owns inference. An ``ask`` that accepts
    ``max_new_tokens`` is handed the benchmark's cap; one that does not is
    asked plainly, the way a text-to-text baseline would be.
    """
    import inspect

    try:
        accepts_cap = "max_new_tokens" in inspect.signature(ask).parameters
    except (ValueError, TypeError):  # a callable with an uninspectable signature
        accepts_cap = False
    bench = BENCHMARKS.get(bench_name)
    if bench is None:
        msg = f"unknown benchmark {bench_name!r}; known: {', '.join(sorted(BENCHMARKS))}"
        raise KeyError(msg)
    score = scorer(bench)
    cap = bench.max_out if max_new_tokens is None else int(max_new_tokens)
    result = EvalResult(mode=mode, benchmark=bench.name)
    for item in load_benchmark(bench.name, fixtures_dir=fixtures_dir, limit=limit):
        item.setdefault("prompt", bench.prompt_for(item))
        prompt = item["prompt"]
        t0 = time.perf_counter()
        if accepts_cap:  # the cap rides, when the engine offers it
            reply = ask(prompt, max_new_tokens=cap)
        else:
            reply = ask(prompt)
        elapsed = time.perf_counter() - t0
        got = score(item, str(reply))
        result.per_item.append(got)
        result.total_time_s += elapsed
        result.items += 1
    if result.items:
        result.accuracy = 100.0 * sum(result.per_item) / result.items
    return result


def compare(measured: dict[str, float], golden: dict[str, float], *, tolerance: float = TOLERANCE):
    """Yield ``Row(check, expected, got, delta, passed)`` per benchmark key.

    ``check`` is the comparison performed — here the absolute difference of
    the accuracy, which must not exceed the published tolerance (±0.5 %
    for the main results, spec §4.2).
    """
    Row = namedtuple("Row", "check expected got delta passed", rename=True)
    for key, expected in sorted(golden.items()):
        got = measured.get(key)
        if got is None:
            yield Row(check=key, expected=expected, got=None, delta=None, passed=False)
            continue
        delta = abs(float(got) - float(expected))
        yield Row(
            check=key, expected=expected, got=float(got), delta=delta, passed=delta <= tolerance
        )
