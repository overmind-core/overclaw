"""Tests for the ``overmind.preflight`` package and CLI gate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.preflight import (
    classifier,
    hashes,
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
    KIND_ENTRYPOINT_REPAIR,
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
        smoke = _make_smoke(cases=[_ok_case(0), _ok_case(1)], span_count=4, baseline_score=0.7)
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


# ---------------------------------------------------------------------------
# Hashes / report serialisation
# ---------------------------------------------------------------------------


class TestHashes:
    def test_match_when_unchanged(self):
        a = {"entrypoint": "x", "eval_spec": "y"}
        ok, diff = hashes.hashes_match(a, a)
        assert ok and not diff

    def test_diff_keys_when_changed(self):
        ok, diff = hashes.hashes_match({"a": "1", "b": "2"}, {"a": "1", "b": "3"})
        assert not ok
        assert diff == ["b"]


class TestReportSerialisation:
    def test_round_trip(self, tmp_path: Path):
        rep = PreflightReport(
            status=STATUS_GREEN,
            agent_name="my-agent",
            iterations=2,
            baseline_score=0.42,
            hashes={"entrypoint": "abc"},
        )
        path = tmp_path / "preflight.json"
        rep.save(path)
        data = json.loads(path.read_text())
        assert data["status"] == STATUS_GREEN
        assert data["hashes"]["entrypoint"] == "abc"

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


class TestOptimizeGate:
    def test_blocks_when_no_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _bootstrap_project(tmp_path, monkeypatch)
        from overmind.commands.optimize_cmd import _enforce_preflight_gate

        with pytest.raises(SystemExit) as exc:
            _enforce_preflight_gate("missing-agent")
        assert exc.value.code == 2

    def test_skip_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _bootstrap_project(tmp_path, monkeypatch)
        monkeypatch.setenv("OVERMIND_SKIP_PREFLIGHT", "1")
        from overmind.commands.optimize_cmd import _enforce_preflight_gate

        # No report exists, but the bypass should let it through.
        _enforce_preflight_gate("any-agent")  # should not raise


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestPreflightCli:
    def test_status_returns_error_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
        _bootstrap_project(tmp_path, monkeypatch)

        args = type("Args", (), {})()
        args.func = None
        args.step = "status"
        # Build an args object via a real argparse run for one subcommand.
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        # The real wiring uses a parent dispatcher; construct the same shape.
        outer = sub.add_parser("preflight")
        outer.add_subparsers(dest="step")
        # Minimal: directly invoke the status handler.
        from overmind.commands.preflight_cmd import _cmd_status

        ns = argparse.Namespace(agent="not-real")
        rc = _cmd_status(ns)
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["status"] == "error"
        assert body["error"] == "no_preflight_report"
        assert rc == 1


# ---------------------------------------------------------------------------
# Entrypoint repair (LLM-driven)
# ---------------------------------------------------------------------------


class TestEntrypointClassification:
    def test_crash_in_harness_becomes_entrypoint_repair(self, tmp_path):
        harness = tmp_path / "agent_entry.py"
        harness.write_text("def run(query): pass\n")
        err = (
            "Traceback (most recent call last):\n"
            f'  File "{harness}", line 1, in run\n'
            '    raise KeyError("query")\n'
            "KeyError: 'query'"
        )
        smoke = _make_smoke(cases=[_crash_case(0, err)])
        issues = classifier.classify(smoke, eval_spec={}, entrypoint_path=str(harness))
        kinds = {i.kind for i in issues}
        assert KIND_ENTRYPOINT_REPAIR in kinds
        assert KIND_RUNTIME_CRASH not in kinds

    def test_crash_outside_harness_stays_runtime_crash(self, tmp_path):
        harness = tmp_path / "agent_entry.py"
        harness.write_text("x = 1\n")
        err = "Traceback ... File '/some/other/file.py', line 5\nKeyError: 'x'"
        smoke = _make_smoke(cases=[_crash_case(0, err)])
        issues = classifier.classify(smoke, eval_spec={}, entrypoint_path=str(harness))
        kinds = {i.kind for i in issues}
        assert KIND_RUNTIME_CRASH in kinds
        assert KIND_ENTRYPOINT_REPAIR not in kinds

    def test_output_schema_mismatch_also_emits_entrypoint_repair(self, tmp_path):
        harness = tmp_path / "h.py"
        harness.write_text("def run(): return {}\n")
        spec = {
            "output_fields": {"a": {"weight": 50}, "b": {"weight": 50}},
            "structure_weight": 0,
            "total_points": 100,
        }
        smoke = _make_smoke(cases=[_ok_case(0, output={"a": 1})])
        issues = classifier.classify(smoke, eval_spec=spec, entrypoint_path=str(harness))
        kinds = [i.kind for i in issues]
        # Both signals emitted; runner's sort_issues runs entrypoint
        # repair first and falls back to schema drop if it no-ops.
        assert KIND_ENTRYPOINT_REPAIR in kinds
        assert KIND_OUTPUT_SCHEMA_MISMATCH in kinds


class TestEntrypointHandler:
    def _state_with_harness(self, tmp_path):
        from overmind.preflight.workspace import WorkingState

        harness = tmp_path / "agent_entry.py"
        harness.write_text("def run(query):\n    return {'answer': query}\n")
        inst = tmp_path / "inst"
        inst.mkdir()
        (inst / "agent_entry.py").write_text(harness.read_text())
        return WorkingState(
            agent_name="x",
            eval_spec={
                "input_schema": {"query": {"type": "text"}},
                "output_fields": {"answer": {"weight": 100}},
            },
            dataset=[{"input": {"query": "hello"}, "expected_output": {"answer": "hi"}}],
            eval_spec_path=tmp_path / "eval_spec.json",
            dataset_path=tmp_path / "dataset.json",
            instrumented_dir=inst,
            entrypoint_path=harness,
        )

    def test_skips_when_no_credentials(self, tmp_path, monkeypatch):
        from overmind.preflight.autofix import entrypoint as ep_autofix

        # Ensure no provider keys are set.
        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("ANALYZER_MODEL", raising=False)

        state_obj = self._state_with_harness(tmp_path)
        issue = IssueRecord(
            kind=KIND_ENTRYPOINT_REPAIR,
            severity="fix",
            target="entrypoint",
            reason="harness crashed",
            details={"row_index": 0, "raw": "boom"},
        )
        patches = ep_autofix.apply_entrypoint_repair(state_obj, issue)
        assert patches == []
        # Budget is consumed only when we actually try, so it must
        # still be 0 here.
        assert state_obj.entrypoint_repair_attempts == 0

    def test_skips_when_budget_exhausted(self, tmp_path, monkeypatch):
        from overmind.preflight.autofix import entrypoint as ep_autofix

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        state_obj = self._state_with_harness(tmp_path)
        state_obj.max_entrypoint_repairs = 1
        state_obj.entrypoint_repair_attempts = 1

        issue = IssueRecord(
            kind=KIND_ENTRYPOINT_REPAIR,
            severity="fix",
            target="entrypoint",
            reason="r",
            details={},
        )
        patches = ep_autofix.apply_entrypoint_repair(state_obj, issue)
        assert patches == []

    def test_reverts_when_no_change(self, tmp_path, monkeypatch):
        from overmind.preflight.autofix import entrypoint as ep_autofix

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        state_obj = self._state_with_harness(tmp_path)
        original = state_obj.entrypoint_path.read_bytes()

        # Stub coding agent to do nothing (no-op).
        with patch("overmind.coding_agent.agent.run") as mock_run:
            mock_run.return_value = None
            issue = IssueRecord(
                kind=KIND_ENTRYPOINT_REPAIR,
                severity="fix",
                target="entrypoint",
                reason="r",
                details={},
            )
            patches = ep_autofix.apply_entrypoint_repair(state_obj, issue)
        assert patches == []
        # File must be unchanged.
        assert state_obj.entrypoint_path.read_bytes() == original
        assert state_obj.entrypoint_repair_attempts == 1

    def test_records_patch_and_syncs_to_instrumented_copy(self, tmp_path, monkeypatch):
        from overmind.preflight.autofix import entrypoint as ep_autofix

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        state_obj = self._state_with_harness(tmp_path)

        new_source = "def run(query):\n    return {'answer': 'fixed'}\n"

        def fake_run(*, instruction, model, cwd, worktree, extra_instructions, max_steps):
            # Simulate coding agent rewriting the harness file.
            state_obj.entrypoint_path.write_text(new_source)

        with patch("overmind.coding_agent.agent.run", side_effect=fake_run):
            issue = IssueRecord(
                kind=KIND_ENTRYPOINT_REPAIR,
                severity="fix",
                target="entrypoint",
                reason="harness wrong",
                details={},
            )
            patches = ep_autofix.apply_entrypoint_repair(state_obj, issue)

        assert len(patches) == 1
        patch_rec = patches[0]
        assert patch_rec.file == str(state_obj.entrypoint_path)
        assert patch_rec.before_hash and patch_rec.after_hash
        assert patch_rec.before_hash != patch_rec.after_hash
        # The instrumented copy was synced.
        assert (state_obj.instrumented_dir / "agent_entry.py").read_text() == new_source
        # Re-instrumentation is queued for the runner.
        assert "agent_entry.py" in state_obj.reinstrument_requests

    def test_reverts_when_collateral_files_touched(self, tmp_path, monkeypatch):
        from overmind.preflight.autofix import entrypoint as ep_autofix

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        state_obj = self._state_with_harness(tmp_path)
        sibling = state_obj.entrypoint_path.parent / "evil.py"
        sibling.write_text("# pre-existing\n")
        original = state_obj.entrypoint_path.read_bytes()

        def fake_run(**_kwargs):
            # Coding agent edits a forbidden sibling — handler must revert.
            sibling.write_text("# tampered\n")
            state_obj.entrypoint_path.write_text("def run(): return {'answer': 'x'}\n")

        with patch("overmind.coding_agent.agent.run", side_effect=fake_run):
            issue = IssueRecord(
                kind=KIND_ENTRYPOINT_REPAIR,
                severity="fix",
                target="entrypoint",
                reason="r",
                details={},
            )
            patches = ep_autofix.apply_entrypoint_repair(state_obj, issue)

        assert patches == []
        # Harness reverted to original bytes.
        assert state_obj.entrypoint_path.read_bytes() == original


class TestSortIssues:
    def test_entrypoint_repair_runs_before_schema_drop(self):
        from overmind.preflight import autofix

        ep = IssueRecord(kind=KIND_ENTRYPOINT_REPAIR, severity="fix", target="entrypoint", reason="r")
        sd = IssueRecord(kind=KIND_OUTPUT_SCHEMA_MISMATCH, severity="fix", target="eval_spec", reason="r")
        sorted_issues = autofix.sort_issues([sd, ep])
        assert sorted_issues[0].kind == KIND_ENTRYPOINT_REPAIR
        assert sorted_issues[1].kind == KIND_OUTPUT_SCHEMA_MISMATCH
