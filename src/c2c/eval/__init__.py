"""The evaluation harness — golden regression against the paper (spec §5).

``c2c.eval`` reproduces the paper's Tables and Figures as pytest cases
with the published numbers as golden values. Where the PDF text
extraction of arXiv:2510.03215v2 delivered a number, it is transcribed
here verbatim (cross-checked twice, once against the abstract, once
against the section text); where the extraction delivered nothing (some
appendix tables are set in images the extractor could not read), the
entry is *declared unavailable* and the corresponding test skips with
that exact reason — never silently, never faked.

Modules:

    c2c.eval.golden      the numbers as published (the golden values)
    c2c.eval.benchmarks  the four benchmarks, offline-first loaders
    c2c.eval.runner      run(...) — drives any engine through the suite

Evaluation protocol (paper §4.1): zero-shot, greedy, temperature 0, max
response 64 tokens, communication budget 256 tokens; LongBench with the
official prompts, 2048 tokens out; average accuracy as the performance
metric, average inference time (single A100, batch size one) as the
efficiency metric. Determinism: seed 42, fixed tokenizer revisions.
"""

from __future__ import annotations

from .benchmarks import BENCHMARKS, load_benchmark
from .golden import FIGURES, TABLES, UNAVAILABLE
from .runner import compare, run

__all__ = ["TABLES", "FIGURES", "UNAVAILABLE", "BENCHMARKS", "load_benchmark", "compare", "run"]
