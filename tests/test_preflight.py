"""Tests for the ``overmind.preflight`` package and CLI gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.preflight import (
    classifier,
)
from overmind.preflight.autofix import (
    dataset as autofix_dataset,
)
from overmind.preflight.autofix import (
    metrics as autofix_metrics,
)
from overmind.preflight.autofix import (
    schema as autofix_schema,
)
from overmind.preflight.autofix import (
    weights as autofix_weights,
)
from overmind.preflight.classifier import (
    KIND_CONSISTENCY_RULES_INVALID,
    KIND_DEGENERATE_OUTPUT,
    KIND_INSTRUMENTATION_BROKEN,
    KIND_INVALID_WEIGHTS,
    KIND_MISSING_SECRET,
    KIND_OUTPUT_SCHEMA_MISMATCH,
    KIND_QUALITY,
    KIND_RUNTIME_CRASH,
)
from overmind.preflight.smoke import CaseResult, SmokeRunResult
from overmind.preflight.state import (
    GREEN_STATUSES,
    STATUS_BLOCKED_SECRETS,
    STATUS_GREEN,
    IssueRecord,
    PreflightReport,
)
from overmind.preflight.workspace import WorkingState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smoke(
    *,
    cases: list[CaseResult] | None = None,
    span_count: int = 5,
    baseline_score: float | None = None,
    preflight_error: str = "",
) -> SmokeRunResult:
    return SmokeRunResult(
        cases=cases or [],
        span_count=span_count,
        baseline_score=baseline_score,
        preflight_error=preflight_error,
    )


def _ok_case(idx: int = 0, output: dict | None = None, score: float = 50.0) -> CaseResult:
    return CaseResult(
        row_index=idx,
        success=True,
        output=output if output is not None else {"result": "x"},
        expected={"result": "x"},
        score=score,
        score_breakdown={"total": score},
    )


def _crash_case(idx: int, err: str) -> CaseResult:
    return CaseResult(row_index=idx, success=False, error=err)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_clean_run_emits_no_issues(self):
        spec = {
            "output_fields": {"result": {"type": "text", "weight": 80}},
            "structure_weight": 20,
            "total_points": 100,
        }
        smoke = _make_smoke(
            cases=[
                _ok_case(0, output={"result": "foo"}),
                _ok_case(1, output={"result": "bar"}),
            ],
            span_count=4,
            baseline_score=0.7,
        )
        issues = classifier.classify(smoke, eval_spec=spec)
        assert issues == []

    def test_missing_secret_blocks(self):
        smoke = _make_smoke(
            cases=[
                _crash_case(0, "litellm.AuthenticationError: invalid api key (OPENAI_API_KEY)"),
            ]
        )
        issues = classifier.classify(smoke, eval_spec={})
        kinds = {i.kind for i in issues}
        assert KIND_MISSING_SECRET in kinds
        assert classifier.has_blockers(issues)
        assert "OPENAI_API_KEY" in classifier.missing_secret_keys(issues)

    def test_runtime_crash_is_quality_not_fixable(self):
        smoke = _make_smoke(cases=[_crash_case(0, "KeyError: 'foo'")])
        issues = classifier.classify(smoke, eval_spec={})
        crash = [i for i in issues if i.kind == KIND_RUNTIME_CRASH]
        assert crash, "expected one runtime_crash issue"
        assert crash[0].severity == "quality"

    def test_invalid_weights_detected(self):
        spec = {"output_fields": {"a": {"weight": 30}, "b": {"weight": 30}}, "structure_weight": 0, "total_points": 100}
        smoke = _make_smoke(cases=[_ok_case(0)])
        issues = classifier.classify(smoke, eval_spec=spec)
        assert any(i.kind == KIND_INVALID_WEIGHTS for i in issues)

    def test_output_schema_mismatch_detected(self):
        spec = {"output_fields": {"a": {"weight": 50}, "b": {"weight": 50}}, "structure_weight": 0, "total_points": 100}
        # Agent only returns {"a": ...}
        smoke = _make_smoke(cases=[_ok_case(0, output={"a": 1})])
        issues = classifier.classify(smoke, eval_spec=spec)
        assert any(i.kind == KIND_OUTPUT_SCHEMA_MISMATCH for i in issues)

    def test_instrumentation_broken_when_no_spans(self):
        smoke = _make_smoke(cases=[_ok_case(0)], span_count=0)
        issues = classifier.classify(smoke, eval_spec={})
        assert any(i.kind == KIND_INSTRUMENTATION_BROKEN for i in issues)

    def test_quality_signal_when_baseline_is_zero(self):
        smoke = _make_smoke(cases=[_ok_case(0, score=0.0)], baseline_score=0.0)
        issues = classifier.classify(smoke, eval_spec={})
        assert any(i.kind == KIND_QUALITY for i in issues)

    def test_degenerate_output_detected(self):
        """Agent returning identical output for every input is flagged."""
        same = {"answer": "yes"}
        smoke = _make_smoke(
            cases=[
                _ok_case(0, output=same, score=50.0),
                _ok_case(1, output=same, score=50.0),
                _ok_case(2, output=same, score=50.0),
            ],
            span_count=3,
            baseline_score=0.5,
        )
        issues = classifier.classify(smoke, eval_spec={})
        assert any(i.kind == KIND_DEGENERATE_OUTPUT for i in issues)
        deg = next(i for i in issues if i.kind == KIND_DEGENERATE_OUTPUT)
        assert deg.severity == "quality"

    def test_varied_output_not_flagged_as_degenerate(self):
        smoke = _make_smoke(
            cases=[
                _ok_case(0, output={"answer": "yes"}, score=50.0),
                _ok_case(1, output={"answer": "no"}, score=50.0),
            ],
            span_count=2,
            baseline_score=0.5,
        )
        issues = classifier.classify(smoke, eval_spec={})
        assert not any(i.kind == KIND_DEGENERATE_OUTPUT for i in issues)

    def test_single_case_not_flagged_as_degenerate(self):
        """Need ≥2 successful outputs to declare degeneracy."""
        smoke = _make_smoke(cases=[_ok_case(0, output={"answer": "yes"})], span_count=1)
        issues = classifier.classify(smoke, eval_spec={})
        assert not any(i.kind == KIND_DEGENERATE_OUTPUT for i in issues)

    def test_consistency_rules_free_text_detected(self):
        """Free-text strings in consistency_rules are flagged as fixable."""
        spec = {
            "output_fields": {"result": {"type": "text", "weight": 100}},
            "total_points": 100,
            "consistency_rules": [
                "Agent should not escalate if the customer is satisfied",
                "Refunds should be processed within 5 business days",
            ],
        }
        smoke = _make_smoke(cases=[_ok_case(0, output={"result": "ok"})], span_count=1)
        issues = classifier.classify(smoke, eval_spec=spec)
        cr = [i for i in issues if i.kind == KIND_CONSISTENCY_RULES_INVALID]
        assert cr, "expected consistency_rules_invalid issue"
        assert cr[0].severity == "fix"
        assert cr[0].details["bad_count"] == 2

    def test_consistency_rules_valid_dicts_not_flagged(self):
        spec = {
            "output_fields": {"urgency": {"type": "number", "weight": 100}},
            "total_points": 100,
            "consistency_rules": [
                {"field_a": "urgency", "field_b": "priority", "type": "correlation"},
            ],
        }
        smoke = _make_smoke(cases=[_ok_case(0, output={"urgency": 1})], span_count=1)
        issues = classifier.classify(smoke, eval_spec=spec)
        assert not any(i.kind == KIND_CONSISTENCY_RULES_INVALID for i in issues)


# ---------------------------------------------------------------------------
# Autofix — pure function transforms
# ---------------------------------------------------------------------------


@pytest.fixture()
def working_state(tmp_path: Path) -> WorkingState:
    spec_path = tmp_path / "eval_spec.json"
    ds_path = tmp_path / "dataset.json"
    spec_path.write_text("{}")
    ds_path.write_text("[]")
    return WorkingState(
        agent_name="x",
        eval_spec={
            "output_fields": {
                "a": {"type": "text", "weight": 30, "importance": "important"},
                "b": {"type": "text", "weight": 40, "importance": "important"},
            },
            "structure_weight": 20,
            "total_points": 100,
        },
        dataset=[
            {"input": {"q": 1}, "expected_output": {"a": "x"}},
            {"input": {"q": 2, "z": 9}, "expected_output": {"a": "y"}},  # extra key → dropped
        ],
        eval_spec_path=spec_path,
        dataset_path=ds_path,
        instrumented_dir=tmp_path / "instr",
    )


class TestAutofixWeights:
    def test_renormalizes_when_below_total(self, working_state):
        # current sum = 30 + 40 + 20 (structure) = 90
        autofix_weights.apply_invalid_weights(
            working_state,
            IssueRecord(kind="invalid_weights", severity="fix", target="eval_spec", reason="test"),
        )
        fields = working_state.eval_spec["output_fields"]
        total = fields["a"]["weight"] + fields["b"]["weight"] + working_state.eval_spec["structure_weight"]
        assert abs(total - 100) <= 0.5

    def test_idempotent_on_clean_spec(self, working_state):
        # First application normalises; second should be a no-op-ish.
        autofix_weights.apply_invalid_weights(
            working_state,
            IssueRecord(kind="invalid_weights", severity="fix", target="eval_spec", reason="test"),
        )
        snapshot = json.dumps(working_state.eval_spec, sort_keys=True)
        autofix_weights.apply_invalid_weights(
            working_state,
            IssueRecord(kind="invalid_weights", severity="fix", target="eval_spec", reason="test"),
        )
        assert json.dumps(working_state.eval_spec, sort_keys=True) == snapshot


class TestAutofixSchema:
    def test_drops_missing_output_fields(self, working_state):
        autofix_schema.apply_output_schema_mismatch(
            working_state,
            IssueRecord(
                kind="output_schema_mismatch",
                severity="fix",
                target="eval_spec",
                reason="test",
                details={"scored_but_missing": ["b"], "actually_returned": ["a"]},
            ),
        )
        assert "b" not in working_state.eval_spec["output_fields"]
        assert "a" in working_state.eval_spec["output_fields"]


class TestAutofixDataset:
    def test_drops_invalid_rows(self, working_state):
        working_state.eval_spec["input_schema"] = {"q": {"type": "number"}}
        autofix_dataset.apply_dataset_row_invalid(
            working_state,
            IssueRecord(
                kind="dataset_row_invalid",
                severity="fix",
                target="dataset",
                reason="test",
                details={"row_index": 1},
            ),
        )
        assert len(working_state.dataset) == 1
        assert working_state.dataset[0]["input"] == {"q": 1}

    def test_no_input_schema_preserves_rows(self, working_state):
        # Without input_schema we have nothing to validate against.
        before = list(working_state.dataset)
        autofix_dataset.apply_dataset_row_invalid(
            working_state,
            IssueRecord(kind="dataset_row_invalid", severity="fix", target="dataset", reason="test"),
        )
        assert working_state.dataset == before


class TestAutofixMetrics:
    def test_coerces_invalid_types_to_text(self, working_state):
        working_state.eval_spec["output_fields"]["a"]["type"] = "string"
        working_state.eval_spec["output_fields"]["b"]["type"] = "object"
        autofix_metrics.apply_metric_broken(
            working_state,
            IssueRecord(kind="metric_broken", severity="fix", target="eval_spec", reason="test"),
        )
        assert working_state.eval_spec["output_fields"]["a"]["type"] == "text"
        assert working_state.eval_spec["output_fields"]["b"]["type"] == "text"

    def test_removes_free_text_consistency_rules(self, working_state):
        working_state.eval_spec["consistency_rules"] = [
            "Agent should not escalate if customer is satisfied",
            {"field_a": "urgency", "field_b": "priority", "type": "correlation"},
            "Another free-text rule",
        ]
        autofix_metrics.apply_consistency_rules_invalid(
            working_state,
            IssueRecord(kind="consistency_rules_invalid", severity="fix", target="eval_spec", reason="test"),
        )
        rules = working_state.eval_spec["consistency_rules"]
        assert len(rules) == 1
        assert rules[0]["field_a"] == "urgency"

    def test_idempotent_when_rules_already_valid(self, working_state):
        working_state.eval_spec["consistency_rules"] = [
            {"field_a": "a", "field_b": "b", "type": "correlation"},
        ]
        patches = autofix_metrics.apply_consistency_rules_invalid(
            working_state,
            IssueRecord(kind="consistency_rules_invalid", severity="fix", target="eval_spec", reason="test"),
        )
        assert patches == []


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


class TestReportSerialisation:
    def test_round_trip(self, tmp_path: Path):
        rep = PreflightReport(
            status=STATUS_GREEN,
            agent_name="my-agent",
            iterations=2,
            baseline_score=0.42,
        )
        path = tmp_path / "preflight.json"
        rep.save(path)
        data = json.loads(path.read_text())
        assert data["status"] == STATUS_GREEN
        assert data["agent_name"] == "my-agent"
        assert data["iterations"] == 2

    def test_is_green_helper(self):
        for s in GREEN_STATUSES:
            assert PreflightReport(status=s, agent_name="a").is_green()
        assert not PreflightReport(status=STATUS_BLOCKED_SECRETS, agent_name="a").is_green()


# ---------------------------------------------------------------------------
# Optimize gate
# ---------------------------------------------------------------------------


def _bootstrap_project(tmp_path: Path, monkeypatch) -> Path:
    """Create a minimal .overmind/ skeleton and chdir into it."""
    (tmp_path / OVERMIND_DIR_NAME).mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestPreflightAdvisory:
    """Preflight is optional — the advisory must never raise."""

    def test_advisory_with_no_report_is_silent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _bootstrap_project(tmp_path, monkeypatch)
        from overmind.commands.optimize_cmd import _preflight_advisory

        # No SystemExit, no exception — preflight is purely informational.
        _preflight_advisory("missing-agent")

    def test_advisory_with_non_green_report_does_not_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _bootstrap_project(tmp_path, monkeypatch)
        from overmind.commands.optimize_cmd import _preflight_advisory
        from overmind.preflight.state import (
            STATUS_BLOCKED_SECRETS,
            preflight_report_path,
        )

        rep = PreflightReport(
            status=STATUS_BLOCKED_SECRETS,
            agent_name="any-agent",
            message="missing OPENAI_API_KEY",
            missing_secrets=["OPENAI_API_KEY"],
        )
        rep.save(preflight_report_path("any-agent"))

        # Non-green is now a warning, not a hard error.
        _preflight_advisory("any-agent")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestPreflightCli:
    def test_status_returns_error_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
        _bootstrap_project(tmp_path, monkeypatch)

        import argparse

        from overmind.commands.preflight_cmd import _cmd_status

        ns = argparse.Namespace(agent="not-real")
        rc = _cmd_status(ns)
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["status"] == "error"
        assert body["error"] == "no_preflight_report"
        assert rc == 1


# ---------------------------------------------------------------------------
# Sort issues
# ---------------------------------------------------------------------------


class TestSortIssues:
    def test_weights_before_schema_drop(self):
        from overmind.preflight import autofix

        wt = IssueRecord(kind=KIND_INVALID_WEIGHTS, severity="fix", target="eval_spec", reason="r")
        sd = IssueRecord(kind=KIND_OUTPUT_SCHEMA_MISMATCH, severity="fix", target="eval_spec", reason="r")
        sorted_issues = autofix.sort_issues([sd, wt])
        assert sorted_issues[0].kind == KIND_INVALID_WEIGHTS
        assert sorted_issues[1].kind == KIND_OUTPUT_SCHEMA_MISMATCH
