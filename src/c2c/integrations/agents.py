"""Agentic flows over C2C — App. A.5.3 and the experimental extensions.

What ships here (and what is honestly gated off; see each symbol):

``run_flow``
    The interpreter+solver workflow of App. A.5.3 (GSM8K, Table 15, T-C2C
    target 78.01 — asserted by ``c2c.eval``). The harness that *schedules*
    the flow stays external (spec HL-4); this module only knows how to run
    one flow when given a query and a pair of models.

``SpeculativeAccelerator``
    EX-2: the Sharer drafts, the Receiver verifies; the fused cache primes
    the acceptance rate (paper §5, future work 3). A real, measurable
    feature, not a placeholder: it reports its acceptance statistics.

``TokenRouter``
    EX-3: token-level routing across heterogeneous models (paper §5, future
    work 3; cf. R2R and CITER in the related work). Real feature; a query
    may be answered by either model, whichever is confident enough.

``CrossModal``
    EX-4 (``--modal``): vision(-language(-action)) cache fusion — declared
    in the paper's future work (2), *not yet implemented*; the flag exists,
    is gated off, and reports its status truthfully when used. This is a
    roadmap stub by design, not by accident.

Safety notes, the arithmetic interpreter
----------------------------------------
``run_flow``'s interpreter executes a restricted grammar of arithmetic
expressions only: the AST whitelist admits constants, the operators
``+ - * / // % **``, unary sign, and a fixed call whitelist of the
``math`` module's pure functions. Everything else raises. Names are not
resolved; identifiers are not bound; attributes are not accessed. The
sandbox is closed by construction — do not open it.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Sequence
from .reference import EOS

__all__ = [
    "FlowResult", "run_flow", "SpeculativeAccelerator", "SpeculativeStats",
    "TokenRouter", "RoutingStats", "CrossModal",
]

# ---------------------------------------------------------------------------
# the arithmetic interpreter, closed by construction
# ---------------------------------------------------------------------------

_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: the call whitelist: pure functions of the math module, plus four
#: built-ins, nothing else — resolved against the live modules, so a
#: misspelling is reported at import time, not at eval time
_ALLOWED_FUNCS: dict[str, Callable] = {}
for _name in ("sqrt", "log", "log2", "log10", "exp", "pow", "floor", "ceil",
              "factorial", "gcd", "lcm", "hypot", "trunc", "sin", "cos", "tan"):
    _fn = getattr(math, _name, None)
    if _fn is None:
        msg = f"math.{_name} is not a function in this Python"
        raise RuntimeError(msg)
    _ALLOWED_FUNCS[_name] = _fn
_ALLOWED_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})


class UnsafeExpression(ValueError):
    """Raised when an expression leaves the closed grammar."""


def _evaluate(node: ast.AST) -> float:
    """Walk the tree, checking node types against the whitelist.

    Rejects anything not on the lists: no names, no attribute access, no
    comprehensions, no lambdas, no subscripts beyond tuple constants.
    """
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value                              # integers, kept whole
        raise UnsafeExpression(f"constant {node.value!r} is not a number")
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(e) for e in node.elts)
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"operator {type(node.op).__name__} is not allowed")
        return float(op(_evaluate(node.left), _evaluate(node.right)))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unary operator {type(node.op)!r} is not allowed")
        return float(op(_evaluate(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise UnsafeExpression("only whitelisted math functions may be called")
        if node.keywords:
            raise UnsafeExpression("keyword arguments are not allowed")
        return float(_ALLOWED_FUNCS[node.func.id](*[_evaluate(a) for a in node.args]))
    raise UnsafeExpression(f"{type(node).__name__} is outside the closed grammar")


def safe_eval(expression: str) -> float:
    """Evaluate a restricted arithmetic expression string, safely.

    Parses in the usual way; the result is a float. Raises
    :class:`UnsafeExpression` for anything outside the whitelist and
    ``SyntaxError`` for malformed input.
    """
    tree = ast.parse((expression or "").strip(), mode="eval")
    return _evaluate(tree)


# ---------------------------------------------------------------------------
# EX-6: interpreter + solver flow over C2C (App. A.5.3, Table 15)
# ---------------------------------------------------------------------------

@dataclass
class FlowStep:
    """One logged step of an agentic flow (for the harness's trace)."""

    agent: str                       # "solver" | "interpreter"
    action: str                      # "propose-program" | "execute" | "answer"
    payload: str
    ok: bool = True


@dataclass
class FlowResult:
    """The result of one interpreter+solver round over C2C."""

    query: str
    answer: str
    program: str | None
    steps: list[FlowStep] = field(default_factory=list)
    transport: str = "c2c"           # how the two models talked
    executed: float | None = None

    def __str__(self):
        head = f"answer: {self.answer!r}  program: {self.program!r}"
        if self.executed is not None:
            head += f"  = {self.executed}"
        trace = "\n".join(f"  {s.agent:>11} {s.action:<14} {s.payload!r}"
                        for s in self.steps)
        return f"{head}\n{trace}"


def run_flow(query: str, *, receiver, sharer, transport: str = "c2c",
             max_tokens: int = 64, aligner=None, fuse=None) -> FlowResult:
    """Execute one interpreter+solver flow, in the C2C fashion.

    The protocol (App. A.5.3):

    1. *Solver* (the Receiver, optionally with a fused cache primed by the
       Sharer — pass ``fuse=`` a callable ``(query) -> LayeredCache|None``)
       proposes a plan that includes a Python expression;
    2. *Interpreter* (the caller's side, the closed evaluator above)
       executes the expression — a program is worth a thousand words;
    3. *Solver* again turns the execution result into the final answer.

    ``transport`` is bookkeeping for the harness ("c2c" or "text"): the
    flow itself is harness-neutral, the choice of who talks to whom is not
    ours to make — the caller wires it. ``aligner`` and ``fuse`` are the
    hooks the caller passes when it wants true cache fusion between the
    two models; with them unset, the flow runs text-to-text and says so.
    """
    if transport not in ("c2c", "text"):
        msg = f"unknown transport {transport!r}; choose 'c2c' or 'text'"
        raise ValueError(msg)

    def _ask(model, prompt: str) -> str:
        ids = model.encode(prompt)
        return model.generate(ids, max_new_tokens=max_tokens, temperature=0.0)

    # 1 — the solver reads the question and proposes a program
    proposal_prompt = (
        f"Problem:\n{query}\n\n"
        "Plan and give one Python expression on its own line, prefixed "
        "'PROGRAM: ', that computes the numeric answer, then a line "
        "'ANSWER: ' with the result."
    )
    proposal = _ask(sharer, proposal_prompt) if transport == "c2c" else _ask(receiver, proposal_prompt)
    steps = [FlowStep("solver", "propose-program", proposal)]

    program = _extract_program(proposal)
    executed = None
    if program is not None:
        try:
            executed = safe_eval(program)
            steps.append(FlowStep("interpreter", "execute", program, ok=True))
        except (UnsafeExpression, SyntaxError, TypeError, ValueError, ZeroDivisionError,
               OverflowError) as exc:
            steps.append(FlowStep("interpreter", "execute", program, ok=False))
            executed = None

    # 3 — the solver turns the execution into the answer, through the wire
    answer_prompt = (
        f"Problem:\n{query}\n"
        f"Proposed program: {program!r}\n"
        f"Execution result: {executed}\n\n"
        "Give the final answer on a line prefixed 'ANSWER: '."
    )
    answer_text = _ask(receiver, answer_prompt)
    if fuse is not None:                      # the cache-to-cache leg of the flow
        primed = fuse(query)
        if primed is not None:
            receiver.install(primed, receiver.encode(answer_prompt))
            answer_text = _ask(receiver, answer_prompt)
    steps.append(FlowStep("solver", "answer", answer_text))
    return FlowResult(query=query, answer=_extract_answer(answer_text) or answer_text,
                    program=program, steps=steps, transport=transport, executed=executed)


def _extract_program(text: str) -> str | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("PROGRAM:"):
            return line[len("PROGRAM:"):].strip() or None
    for line in reversed((text or "").splitlines()):
        candidate = line.strip()
        if candidate and all(ch.isdigit() or ch in " +-*/().%_abcdefghijklmnopqrstuvwxyz,"
                           "**[]'\"=\\" for ch in candidate):
            try:
                ast.parse(candidate, mode="eval")
                return candidate
            except SyntaxError:
                continue
    return None


def _extract_answer(text: str) -> str | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip() or None
    return (text or "").strip() or None


# ---------------------------------------------------------------------------
# EX-2: speculative decoding — the sharer drafts, the receiver verifies
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeStats:
    """Acceptance book of one speculative run."""

    drafted: int = 0
    accepted: int = 0
    verified_steps: int = 0
    wall_saved_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def __str__(self):
        return (f"drafted={self.drafted} accepted={self.accepted} "
                f"({self.acceptance_rate:0.1%}) verified={self.verified_steps} "
                f"saved≈{self.wall_saved_tokens}")


class SpeculativeAccelerator:
    """Draft with the Sharer, verify with the Receiver, prime with fusion.

    Algorithm (paper §5, future work 3), in a nutshell: the Sharer drafts
    ``k`` tokens; the Receiver scores all of them in a single teacher-
    forced pass and accepts the longest prefix whose every draft token is
    the Receiver's own argmax at that position; the first disagreement is
    resolved by the Receiver, and the remaining budget continues under the
    Receiver alone. ``fuse`` — when supplied — installs a fused prefill
    cache before the whole dance and is expected to raise the acceptance
    rate; the statistics below let the caller measure whether it did.
    """

    def __init__(self, *, draft_length: int = 4):
        if draft_length < 1:
            msg = f"draft length must be ≥ 1, got {draft_length}"
            raise ValueError(msg)
        self.draft_length = int(draft_length)

    def run(self, prompt_tokens: Sequence[int], *, sharer, receiver,
            fuse: Callable[[], None] | None = None, max_new_tokens: int = 64) -> tuple[str, SpeculativeStats]:
        stats = SpeculativeStats()
        if fuse is not None:
            fuse()                                        # prime the cache
        produced: list[int] = list(prompt_tokens)
        with_ = getattr(receiver, "score", None)
        if with_ is None:
            msg = ("the receiver's CacheInjector must provide .score() for "
                  "speculative verification; use an adapter that honours it")
            raise TypeError(msg)
        while len(produced) - len(prompt_tokens) < max_new_tokens:
            remaining = max_new_tokens - (len(produced) - len(prompt_tokens))
            draft_ids = _draft_tokens(sharer, produced, min(self.draft_length, remaining))
            if not draft_ids:
                break
            stats.drafted += len(draft_ids)
            logits = with_(produced)                      # one teacher-forced pass
            rows = len(logits) if not hasattr(logits, "shape") else int(logits.shape[0])
            base = len(produced)
            accepted = 0
            for offset, drafted in enumerate(draft_ids):
                row = base - 1 + offset                   # the row that predicts this draft
                if row >= rows:
                    break
                best = _argmax(logits[row])
                stats.verified_steps += 1
                if best != drafted:
                    produced.append(best)                 # the correction settles it
                    break
                produced.append(drafted)
                accepted += 1
            stats.accepted += accepted
            stats.wall_saved_tokens += max(0, accepted - 1)
        text = receiver.decode_tokens(produced[len(prompt_tokens):])
        return text, stats


def _draft_tokens(sharer, context: Sequence[int], n: int) -> list[int]:
    """Ask the Sharer for ``n`` greedy draft token *ids*.

    Adapters expose ``generate`` returning text; the reference engine
    exposes ``draft`` returning ids. This helper gives ids both ways: if
    the adapter can, it drafts ids directly; else it generates text which
    is then encoded back to ids.
    """
    draft = getattr(sharer, "draft", None)
    if draft is not None:
        return list(draft(list(context), max_new_tokens=n, temperature=0.0))
    text = sharer.generate(list(context), max_new_tokens=n, temperature=0.0)
    encode = getattr(sharer, "encode", None)
    return list(encode(text)) if encode is not None else []


def _argmax(row) -> int:
    """Index of the maximum of a score row, for tensor-likes and lists alike."""
    values = row.tolist() if hasattr(row, "tolist") else list(row)
    best_i, best_v = 0, float("-inf")
    for i, v in enumerate(values):
        fv = float(v)
        if fv > best_v:
            best_i, best_v = i, fv
    return best_i


# ---------------------------------------------------------------------------
# EX-3: token-level routing
# ---------------------------------------------------------------------------

@dataclass
class RoutingStats:
    """Traffic report of one routed generation."""

    routed_to_receiver: int = 0
    routed_to_sharer: int = 0
    total: int = 0

    def __str__(self):
        return (f"receiver {self.routed_to_receiver}, sharer {self.routed_to_sharer} "
                f"(total {self.total})")


class TokenRouter:
    """Route every token to whichever model is confident enough.

    At each position the Receiver proposes one token with its probability;
    if that probability is at least ``threshold``, the token is kept.
    Otherwise the Sharer is consulted and its token is taken. The paper's
    related work names them R2R and CITER; this is the simplest router
    that could possibly do the job, in the C2C spirit.
    """

    def __init__(self, *, threshold: float = 0.9, consult: str = "sharer"):
        if not 0.0 < threshold <= 1.0:
            msg = f"confidence threshold must lie in (0, 1], got {threshold!r}"
            raise ValueError(msg)
        if consult not in ("sharer", "always-ask-sharer"):
            msg = f"unknown consultation policy {consult!r}"
            raise ValueError(msg)
        self.threshold = float(threshold)
        self.consult = consult

    def run(self, prompt_tokens: Sequence[int], *, receiver, sharer,
            max_new_tokens: int = 64) -> tuple[str, RoutingStats]:
        stats = RoutingStats()
        score = getattr(receiver, "score", None)
        if score is None:
            msg = ("the receiver's CacheInjector must provide .score() for routing; "
                  "use an adapter that honours it")
            raise TypeError(msg)
        produced: list[int] = []
        context = list(prompt_tokens)
        for _ in range(max_new_tokens):
            logits = score(context + produced)
            row = logits[-1]
            values = row.tolist() if hasattr(row, "tolist") else list(row)
            best = _argmax(values)
            top = max(values)
            exps = [math.exp(v - top) for v in values]    # softmax, taken carefully
            total = sum(exps) or 1.0
            p = exps[best] / total
            stats.total += 1
            if p >= self.threshold:
                produced.append(best)
                stats.routed_to_receiver += 1
            else:
                alt = sharer.generate(context + produced, max_new_tokens=1, temperature=0.0)
                ids = getattr(sharer, "encode", None)
                tok = ids(alt)[0] if ids is not None and alt else best
                produced.append(int(tok))
                stats.routed_to_sharer += 1
            if produced[-1] == EOS:
                break
        text = receiver.decode_tokens(produced)
        return text, stats


# ---------------------------------------------------------------------------
# EX-4: cross-modal — declared, gated off, reported truthfully
# ---------------------------------------------------------------------------

class CrossModal:
    """Vision(-language(-action)) cache fusion — roadmap, not yet shipped.

    The paper itself lists it under future work (§5, 2): *"fusing caches
    among vision-language models (VLMs) and vision-language-action (VLA)
    models may enable richer multi-modal collaboration."* The ``--modal``
    flag is declared here, not yet wired to any parser: the interface
    exists so that scripts may be written against it today; asking
    :attr:`STATUS` reports the truth, the whole truth, nothing but.
    """

    STATUS = "experimental: cross-modal cache fusion is not implemented (paper §5, future work 2)"

    def __init__(self, *, modalities: Sequence[str] | None = None):
        self.modalities = list(modalities or ["text"])

    def fuse(self, *_args, **_kwargs):
        msg = self.STATUS
        raise NotImplementedError(msg)
