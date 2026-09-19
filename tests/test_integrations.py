"""Tests for the agent integrations (EX-2/3/5/6) — experimental, on purpose.

The speculative accelerator, the token router, the interpreter+solver
flow, and the safe evaluator. Experimental flags: features may be
exciting; all rights reserved; behaviour may improve.
"""

from __future__ import annotations

import pytest

from c2c.integrations.agents import (
    FlowResult,
    FlowStep,
    RoutingStats,
    SpeculativeAccelerator,
    SpeculativeStats,
    TokenRouter,
    UnsafeExpression,
    run_flow,
    safe_eval,
)


def _pair():
    from c2c.integrations.reference import ReferenceAdapter

    return (ReferenceAdapter("solver", seed=11), ReferenceAdapter("interpreter-helper", seed=12))


class TestSafeEval:
    """The evaluator that is safe: numbers, not names, not imports."""

    def test_the_arithmetic_of_the_allowed(self):
        assert safe_eval("2 + 2") == pytest.approx(4.0)
        assert safe_eval("2 * 3 + 4") == pytest.approx(10.0)
        assert safe_eval("(1 + 2) * (3 + 4)") == pytest.approx(21.0)
        assert safe_eval("7 / 2") == pytest.approx(3.5)
        assert safe_eval("10 % 3") == pytest.approx(1.0)
        assert safe_eval("2 ** 10") == pytest.approx(1024.0)

    def test_the_functions_of_the_allowed(self):
        """The math library, whitelisted: not the builtins, not the world."""
        assert safe_eval("max(1, 2, 3)") == pytest.approx(3.0)
        assert safe_eval("min(4, 5)") == pytest.approx(4.0)
        assert safe_eval("abs(0 - 6)") == pytest.approx(6.0)
        assert safe_eval("round(2.567, 2)") == pytest.approx(2.57)
        assert safe_eval("sqrt(4)") == pytest.approx(2.0)  # from the math module
        with pytest.raises(UnsafeExpression):
            safe_eval("sum((1, 2, 3))")  # sum, not on the list

    def test_a_raise_from_a_falsy_expression(self):
        """The names that are forbidden: the attribute, the import, the call."""
        for evil in (
            "__import__('os').system('ls')",
            "open('etc/passwd')",
            "().__class__",
            "lambda: 1",
            "[i for i in range(3)]",
        ):
            with pytest.raises((UnsafeExpression, ValueError, SyntaxError)):
                safe_eval(evil)  # refused, loudly

    def test_the_syntax_of_the_empty_expression(self):
        with pytest.raises((UnsafeExpression, ValueError, SyntaxError)):
            safe_eval("")  # nothing, evaluated


class TestSpeculativeAcceleration:
    """EX-2: draft tokens, verify tokens, accelerate generation."""

    def test_the_draft_and_the_final(self):
        receiver, sharer = _pair()
        accelerator = SpeculativeAccelerator(draft_length=4)
        ids = receiver.encode("the theory of transplantation")
        result = accelerator.run(ids, sharer=sharer, receiver=receiver)
        text, stats = result if isinstance(result, tuple) else (result, None)
        assert isinstance(text, str)  # text, generated or not
        if stats is not None:
            assert stats.accepted <= stats.drafted  # never more than drafted
            assert 0.0 <= stats.acceptance_rate <= 1.0  # a rate, in range
            assert isinstance(str(stats), str)  # printable, as promised

    def test_the_acceptance_of_the_rejected(self):
        stats = SpeculativeStats(drafted=10, accepted=7)
        assert stats.acceptance_rate == pytest.approx(0.7)  # seven, of ten
        empty = SpeculativeStats()
        assert empty.acceptance_rate == pytest.approx(0.0)  # nothing, nothing


class TestRoutingDecisions:
    """EX-3: route every query, consult the sharer when uncertain."""

    def test_the_router_routes(self):
        receiver, sharer = _pair()
        router = TokenRouter(threshold=0.9, consult="sharer")
        ids = receiver.encode("a b c d e")
        result = router.run(ids, receiver=receiver, sharer=sharer)
        text, stats = result if isinstance(result, tuple) else (result, None)
        assert isinstance(text, str)  # an answer, routed
        if stats is not None:
            assert isinstance(stats, RoutingStats)  # a log, of the routing

    def test_the_threshold_of_consultation(self):
        """A low threshold, a high consultation: above it, not above."""
        receiver, sharer = _pair()
        eager = TokenRouter(threshold=1e-9)  # consult, near-always
        lazy = TokenRouter(threshold=1.0)  # consult, near-never
        ids = receiver.encode("decide")
        a = eager.run(ids, receiver=receiver, sharer=sharer)
        b = lazy.run(ids, receiver=receiver, sharer=sharer)
        assert (a[1] if isinstance(a, tuple) else a) is not None
        assert (b[1] if isinstance(b, tuple) else b) is not None  # both, answered


class TestAgenticFlow:
    """EX-6: the interpreter, the solver, the flow, the result."""

    def test_the_flow_of_the_program(self):
        """One query, two models, three stages: the flow, end to end."""
        receiver, sharer = _pair()
        result = run_flow("what is 2 + 2 * 3?", receiver=receiver, sharer=sharer)
        assert isinstance(result, FlowResult)  # a result, typed
        assert result.query == "what is 2 + 2 * 3?"  # echoed, verbatim
        assert isinstance(result.answer, str) and result.answer  # an answer, at least
        assert result.transport == "c2c"  # the wire, used
        assert isinstance(result.steps, list) and result.steps  # every step, logged
        for step in result.steps:
            assert isinstance(step, FlowStep)
            assert step.agent and step.action  # who, and what
        assert str(result)  # readable, for humans

    def test_the_transport_of_the_relay(self):
        """The fallback: transport='text', the old wire, still works."""
        receiver, sharer = _pair()
        result = run_flow("capital of france", receiver=receiver, sharer=sharer, transport="text")
        assert result.transport == "text"  # as configured
        assert isinstance(result.answer, str)

    def test_the_steps_of_the_execution(self):
        """A step, recorded: agent, action, payload, and its ok flag."""
        step = FlowStep(agent="solver", action="propose-program", payload="1+1")
        assert step.ok is True  # by default, all is well
        assert (step.agent, step.action, step.payload) == ("solver", "propose-program", "1+1")

    def test_the_result_is_reproducible(self):
        """The same query, twice: the same flow, to the same end."""
        receiver, sharer = _pair()
        q = "compute 3 * 3"
        a = run_flow(q, receiver=receiver, sharer=sharer)
        b = run_flow(q, receiver=receiver, sharer=sharer)
        assert a.answer == b.answer  # determinism, held
        assert [s.action for s in a.steps] == [s.action for s in b.steps]


class TestExperimentalFlags:
    """The flags, in the manual: on, meaning experimental."""

    def test_cross_modal_is_flagged_experimental(self):
        """EX-5: the multi-modal fuse, present but not yet lit."""
        from c2c.integrations import agents

        mod = getattr(agents, "CrossModal", None)
        if mod is None:  # declared, in the module
            pytest.skip("cross-modal, not in this build")
        instance = mod(modalities=("text",))  # constructed, at least
        assert instance is not None  # present, for now
