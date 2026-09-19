"""Tests for diagnostics, doctor, and failure modes (FR-16/FR-17).

The effective rank (§4.5, Roy & Vetterli 2007), the rank report of the
fusion, the gate regimes, the failure attribution of §5.1, and the
doctor's checkup. Measurements, not measurements of measurements.
"""

from __future__ import annotations

import io
import json
import math

import pytest
import torch

from c2c.config import C2CConfig
from c2c.diagnostics.failure import FailureProbe, LayerAttribution, StructuredLog
from c2c.diagnostics.gates import BROAD_OPEN, REGIMES, SPARSE_OPEN, GateReading, classify_regime
from c2c.diagnostics.rank import RankReport, effective_rank, rank_report
from c2c.types import LayeredCache, LayerGeometry, LayerSlice

GEOMETRY = LayerGeometry(layers=3, hidden_size=8, num_heads=2)


def cache(geometry, seed, *, degenerate=False):
    g = torch.Generator().manual_seed(seed)
    slices = []
    for _ in range(geometry.layers):
        if degenerate:  # a rank-1 cache, all alike
            base = torch.randn(geometry.kv_hidden_size, 1, generator=g)
            row = base @ torch.randn(1, 1, generator=g)
            key = row.expand(geometry.kv_hidden_size, geometry.kv_hidden_size).contiguous()
        else:
            key = torch.randn(geometry.kv_hidden_size, geometry.kv_hidden_size, generator=g)
            key = key + torch.eye(geometry.kv_hidden_size)  # well-conditioned, on purpose
        value = key.clone()
        slices.append(LayerSlice(key, value))
    return LayeredCache(slices)


class TestEffectiveRank:
    """erank(A) = exp(−Σ pᵢ ln pᵢ), p from the squared singular values."""

    def test_the_identity_has_full_rank(self):
        assert effective_rank(torch.eye(5)) == pytest.approx(5.0, abs=1e-4)

    def test_a_rank_one_matrix_has_effective_rank_one(self):
        a = torch.ones(8, 1) @ torch.ones(1, 8)  # one, the outer, the all
        assert effective_rank(a) == pytest.approx(1.0, abs=1e-4)

    def test_the_empty_matrix_has_rank_zero(self):
        assert effective_rank(torch.empty(0, 0)) == 0.0  # the empty set, as it must be

    def test_a_vector_is_read_as_one_row(self):
        """1-D input: one row, hence one singular value, hence rank one."""
        assert effective_rank(torch.tensor([3.0, 4.0])) == pytest.approx(1.0, abs=1e-6)
        # and read, from the table of the singular values, as published:
        sigma = torch.tensor([3.0, 4.0])  # a diagonal matrix, by hand
        s2 = sigma * sigma
        p = s2 / s2.sum()
        expected = math.exp(-float((p * torch.log(p)).sum()))
        assert effective_rank(torch.diag(sigma)) == pytest.approx(expected, rel=1e-4)

    def test_numbers_read_aloud_from_the_singular_values(self):
        """diagnostic, read from the standard table of the formula."""
        sigma = torch.tensor([2.0, 1.0, 1.0, 0.0])  # four, of them singular
        s2 = sigma * sigma
        p = s2 / s2.sum()
        expected = math.exp(-float((p[p > 0] * torch.log(p[p > 0])).sum()))
        assert effective_rank(torch.diag(sigma)) == pytest.approx(expected, rel=1e-4)

    def test_buffers_are_accepted_from_any_sequence(self):
        """Plain Python lists, of the same numbers, give the same reading."""
        plain = [[2.0, 0.0], [0.0, 1.0]]
        assert effective_rank(plain) == pytest.approx(effective_rank(torch.tensor(plain)), rel=1e-6)

    def test_rich_semantics_rise_high(self):
        """A random well-conditioned cache ranks above a degenerate one."""
        rich = effective_rank(cache(GEOMETRY, 1, degenerate=False)[0].key)
        flat = effective_rank(cache(GEOMETRY, 2, degenerate=True)[0].key)
        assert rich > flat  # Table 2, the paper's claim


class TestRankReport:
    """Rank before fusion, rank after fusion: Table 2, in the format."""

    def test_the_comparison_is_made(self):
        before = cache(GEOMETRY, 5, degenerate=True)
        after = cache(GEOMETRY, 5, degenerate=False)
        report = rank_report(before, after)
        assert isinstance(report, RankReport)
        assert report.key, "the key report, empty"
        assert all(k in report.key for k in ("before", "after"))
        assert report.key["after"] > report.key["before"]  # fusion must increase
        assert report.key["delta"] == pytest.approx(report.key["after"] - report.key["before"])
        assert set(report.value) >= {"before", "after", "delta"}

    def test_the_increase_is_reported(self):
        before = cache(GEOMETRY, 5, degenerate=True)
        after = cache(GEOMETRY, 5, degenerate=False)
        report = rank_report(before, after)
        assert report.increased  # at least one layer, improved
        assert 0 in report.increased  # the first, among them

    def test_the_report_reads_as_printable(self):
        before = cache(GEOMETRY, 5, degenerate=True)
        after = cache(GEOMETRY, 5, degenerate=False)
        text = str(rank_report(before, after))
        assert "K" in text or "key" in text.lower()  # the keys, mentioned
        assert "V" in text or "value" in text.lower()  # the values, likewise


class TestGateRegimes:
    """The three regimes, one reading: general · mixed · task."""

    def test_the_regimes_are_three(self):
        assert REGIMES == ("general", "mixed", "task")

    def test_a_broadly_open_gate_is_general(self):
        reading = classify_regime(1.0)
        assert reading.regime == "general"
        assert isinstance(reading, GateReading)

    def test_a_sparely_open_gate_is_task_specific(self):
        assert classify_regime(0.0).regime == "task"
        assert classify_regime(SPARSE_OPEN).regime == "task"  # the boundary, inclusive

    def test_the_middle_regime_is_mixed(self):
        mid = (BROAD_OPEN + SPARSE_OPEN) / 2
        assert classify_regime(mid).regime == "mixed"

    def test_the_boundaries_are_the_boundaries(self):
        assert classify_regime(0.60).regime == "general"  # at least, 60 % open
        assert classify_regime(0.61).regime == "general"
        assert classify_regime(0.40).regime == "task"  # at most, 40 % open
        assert classify_regime(0.39).regime == "task"

    def test_a_ratio_out_of_range_is_a_value_error(self):
        with pytest.raises(ValueError):
            classify_regime(-0.1)  # below the floor
        with pytest.raises(ValueError):
            classify_regime(1.1)  # above the ceiling

    def test_a_non_finite_weight_is_a_value_error(self):
        with pytest.raises(ValueError, match="finite"):
            classify_regime(0.5, float("nan"))  # NaN, not a number

    def test_readings_are_str(self):
        assert isinstance(str(GateReading(ratio=0.5, mean_weight=0.3, regime="mixed")), str)


class TestFailureProbe:
    """§5.1, the fault; the sharer, named; the layer, attributed."""

    def setup_method(self):
        self.probe = FailureProbe()
        self.good = cache(GEOMETRY, 9)

    def test_a_good_cache_is_a_good_cache_is_a_blameless_cache(self):
        attributions = self.probe.compare(self.good, self.good, [1.0] * GEOMETRY.layers)
        assert all(isinstance(a, LayerAttribution) for a in attributions)
        assert all(not a.suspect for a in attributions)  # no faults, no reports

    def test_a_weak_sharer_is_named_and_flagged(self):
        """The weak-Sharer noise of §5.1: attributed to the layer, blamed."""
        weak = cache(GEOMETRY, 9, degenerate=True)
        mixed = LayeredCache([self.good[0], weak[1], self.good[2]])
        attributions = self.probe.compare(self.good, mixed, [1.0] * GEOMETRY.layers)
        suspects = [a.layer for a in attributions if a.suspect]
        assert 1 in suspects, "the weak layer, unattributed"  # the finger, pointed

    def test_attributions_are_counted_per_layer(self):
        attributions = self.probe.compare(self.good, cache(GEOMETRY, 10), [1.0] * GEOMETRY.layers)
        assert [a.layer for a in attributions] == list(range(GEOMETRY.layers))

    def test_the_report_is_a_string(self):
        attributions = self.probe.compare(self.good, self.good, [0.5] * GEOMETRY.layers)
        text = self.probe.report(attributions)
        assert isinstance(text, str) and text  # printed, readable
        assert "layer" in text.lower()  # the column, header'd

    def test_the_threshold_of_blame_is_configurable(self):
        strict = FailureProbe(threshold=0.000001)  # a hair, triggers all
        weak = cache(GEOMETRY, 9, degenerate=True)
        assert any(a.suspect for a in strict.compare(self.good, weak, [1.0] * GEOMETRY.layers))


class TestStructuredLog:
    """JSON lines, one record per event, as the manual instructs."""

    def test_records_are_written_as_json(self):
        stream = io.StringIO()
        log = StructuredLog(stream=stream)
        log.log_attribution(event="train", step=1, loss=0.25, gate=0.75)
        stream.seek(0)
        lines = [line for line in stream.read().splitlines() if line]
        assert len(lines) == 1  # one line, one record
        record = json.loads(lines[0])  # parseable, in full
        assert record["event"] == "train" and record["step"] == 1
        assert record["loss"] == pytest.approx(0.25)  # the float, faithful

    def test_the_in_memory_mirror_keeps_the_records(self):
        log = StructuredLog()  # no stream, still logging
        log.log_attribution(event="fuse", fraction=0.5)
        assert log.records_written[-1]["event"] == "fuse"  # the mirror, exact

    def test_non_finite_floats_do_not_break_the_json(self):
        stream = io.StringIO()
        log = StructuredLog(stream=stream)
        log.log_attribution(inf=float("inf"), nan=float("nan"), ok=1.0)
        stream.seek(0)
        record = json.loads(stream.readline())  # still valid JSON
        assert record["ok"] == 1.0  # the numbers, sane


class TestDoctor:
    """c2c doctor — the report card of the whole installation."""

    def test_the_checkup_runs_clean(self):
        from c2c.diagnostics.doctor import Check, run_all

        checks = run_all(C2CConfig(), verbose=False)
        assert checks, "the doctor said nothing at all"
        assert all(isinstance(c, Check) for c in checks)  # checks, all of them

    def test_the_card_formats_the_findings(self):
        from c2c.diagnostics.doctor import format_report, run_all

        checks = run_all(C2CConfig(), verbose=False)
        text = format_report(checks)
        assert "doctor" in text.lower()  # the title, on the card
        assert "python" in text.lower() or "ok" in text.lower()  # the findings, listed

    def test_a_missing_peer_is_reported_not_hidden(self):
        """The availability checks: every adapter, its engine's absence."""
        from c2c.diagnostics.doctor import run_all

        checks = run_all(C2CConfig(), verbose=False)
        names = [c.name for c in checks]
        assert any("engine" in name or "peer" in name for name in names)  # the peers, named

    def test_the_status_codes_are_the_status_codes(self):
        from c2c.diagnostics.doctor import STATUS_FAIL, STATUS_OK, STATUS_SKIP, STATUS_WARN

        for token in (STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_SKIP):
            assert isinstance(token, str) and token  # printable, in the card


class TestTheDoctorInRelayMode:
    """A machine without torch is a supported machine: report it, do not die on it.

    Regression: the diagnostics once bound the backend at module scope, so
    ``c2c doctor`` — the very tool that tells you a backend is missing —
    died with a traceback before it could say so. Both the package import
    and the checkup must survive a world without the nets.
    """

    def test_the_checkup_survives_a_world_without_torch(self, monkeypatch):
        import sys

        class DeclinesTorch:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] == "torch":
                    raise ModuleNotFoundError("No module named 'torch'")
                return None

        monkeypatch.setattr(sys, "meta_path", [DeclinesTorch(), *sys.meta_path])
        for name in [
            n
            for n in list(sys.modules)
            if n == "torch"
            or n.startswith("torch.")
            or n == "c2c.diagnostics"
            or n.startswith("c2c.diagnostics.")
        ]:
            monkeypatch.delitem(sys.modules, name, raising=False)

        from c2c.diagnostics.doctor import STATUS_OK, STATUS_SKIP, STATUS_WARN, run_all

        checks = run_all(C2CConfig(), verbose=False)
        by_name = {c.name: c for c in checks}
        assert by_name["torch"].status == STATUS_WARN  # reported, not raised
        assert by_name["torch"].hint  # and, with a way out
        assert by_name["caches"].status == STATUS_SKIP  # the miniature, furled
        assert by_name["numpy"].status == STATUS_OK  # the base dep, sound
