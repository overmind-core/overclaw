"""Tests for overmind.optimize.execution_backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from overmind.optimize.execution_backend import (
    BackendOutput,
    BackendPlan,
    ShadowBackend,
    SubprocessBackend,
    build_default_plan,
    should_try_next,
)
from overmind.optimize.failure_classifier import (
    FailureDiagnosis,
    FailureMode,
)
from overmind.optimize.provenance import Confidence, SourceTag, TraceSource
from overmind.optimize.runner import RunOutput


def _mk_runner(ok: bool = True, stderr: str = "", err: str = "") -> MagicMock:
    runner = MagicMock()
    runner.ensure_environment = MagicMock()
    runner.cleanup = MagicMock()
    ro = RunOutput(
        success=ok,
        data={"result": "ok"} if ok else None,
        error=err,
        stdout="",
        stderr=stderr,
        returncode=0 if ok else 1,
    )
    runner.run = MagicMock(return_value=ro)
    return runner


# ---------------------------------------------------------------------------
# SubprocessBackend
# ---------------------------------------------------------------------------


class TestSubprocessBackend:
    def test_success_produces_real_confidence(self):
        runner = _mk_runner(ok=True)
        backend = SubprocessBackend(runner)
        backend.prepare()
        out = backend.run({"x": 1})
        assert out.success is True
        assert out.backend == "subprocess"
        assert out.confidence.score == pytest.approx(1.0)
        assert out.confidence.summary == {"real_subprocess": 1}

    def test_pathful_cassette_is_preserved(self, tmp_path: Path):
        """The caller's cassette must be threaded through unchanged.

        Historically ``Cassette`` carried a ``__len__`` that returned 0
        for an empty cassette, so a careless ``cassette or open_cassette(None)``
        check would silently drop the caller's path on the first run.
        That class is now a plain frozen dataclass; this test pins the
        threading so a future refactor cannot silently re-swap.
        """
        from overmind.optimize.cassette import open_cassette

        cass_path = tmp_path / "c.jsonl"
        fresh = open_cassette(cass_path)
        runner = _mk_runner(ok=True)
        backend = SubprocessBackend(runner, cassette=fresh)
        assert backend._cassette.path == cass_path

    def test_failure_classifies_diagnosis(self):
        runner = _mk_runner(
            ok=False,
            stderr="ModuleNotFoundError: No module named 'foo'",
            err="Exit 1",
        )
        backend = SubprocessBackend(runner)
        backend.prepare()
        out = backend.run({"x": 1})
        assert out.success is False
        assert out.diagnosis is not None
        assert out.diagnosis.mode == FailureMode.MISSING_DEPENDENCY

    def test_prepare_calls_ensure_environment_once(self):
        runner = _mk_runner(ok=True)
        backend = SubprocessBackend(runner)
        backend.prepare()
        backend.prepare()
        assert runner.ensure_environment.call_count == 1


# ---------------------------------------------------------------------------
# ShadowBackend
# ---------------------------------------------------------------------------


class TestShadowBackend:
    def test_success_reads_empty_sidecar_as_tagless(self, tmp_path: Path):
        runner = _mk_runner(ok=True)
        from overmind.optimize.cassette import open_cassette

        cass = open_cassette(tmp_path / "c.jsonl")
        backend = ShadowBackend(runner, cassette=cass, provenance_dir=tmp_path)
        backend.prepare()
        out = backend.run({"x": 1})
        assert out.backend == "shadow"
        assert out.success is True
        # No provenance sidecar was written (runner is a mock), so tags are empty.
        assert out.provenance == []
        # The confidence reason reflects that the shadow run had no intercepts.
        assert "shadow" in out.confidence.reason

    def test_success_with_provenance(self, tmp_path: Path):
        runner = _mk_runner(ok=True)
        from overmind.optimize.cassette import open_cassette

        cass = open_cassette(tmp_path / "c.jsonl")
        prov_dir = tmp_path / "prov"
        backend = ShadowBackend(runner, cassette=cass, provenance_dir=prov_dir)
        backend.prepare()

        # Write a sidecar *before* the run so the backend picks it up.
        # The backend names files prov-00001.jsonl, prov-00002.jsonl, …
        (prov_dir).mkdir(parents=True, exist_ok=True)
        sidecar = prov_dir / "prov-00001.jsonl"
        import json

        sidecar.write_text(
            json.dumps({"name": "llm:gpt-4o", "source": "llm_real", "reason": "r"})
            + "\n"
            + json.dumps({"name": "browser", "source": "simulated", "reason": "b"})
            + "\n",
            encoding="utf-8",
        )

        out = backend.run({"x": 1})
        sources = [t.source for t in out.provenance]
        assert TraceSource.LLM_REAL in sources
        assert TraceSource.SIMULATED in sources

    def test_failure_returns_diagnosis(self, tmp_path: Path):
        runner = _mk_runner(
            ok=False,
            stderr="playwright not installed",
            err="Browser launch failed",
        )
        from overmind.optimize.cassette import open_cassette

        cass = open_cassette(tmp_path / "c.jsonl")
        backend = ShadowBackend(runner, cassette=cass, provenance_dir=tmp_path)
        backend.prepare()
        out = backend.run({"x": 1})
        assert out.success is False
        assert out.diagnosis is not None
        assert out.diagnosis.mode == FailureMode.BROWSER_RUNTIME_ERROR


# ---------------------------------------------------------------------------
# build_default_plan / should_try_next
# ---------------------------------------------------------------------------


class TestBackendPlan:
    def test_default_plan_has_two_backends(self, tmp_path: Path):
        runner = _mk_runner(ok=True)
        plan = build_default_plan(
            runner=runner,
            cassette_path=tmp_path / "c.jsonl",
            provenance_dir=tmp_path / "prov",
        )
        names = [b.name for b in plan]
        assert names == ["subprocess", "shadow"]

    def test_disable_shadow_fallback(self, tmp_path: Path):
        runner = _mk_runner(ok=True)
        plan = build_default_plan(
            runner=runner,
            cassette_path=tmp_path / "c.jsonl",
            provenance_dir=tmp_path / "prov",
            enable_shadow_fallback=False,
        )
        assert len(plan) == 1
        assert plan.backends[0].name == "subprocess"

    def test_iter_over_plan(self):
        fake = [object(), object()]
        plan = BackendPlan(backends=fake)  # type: ignore[arg-type]
        assert list(plan) == fake


class TestShouldTryNext:
    def test_none_diag_no_fallback(self):
        assert should_try_next(None) is False

    def test_none_mode_no_fallback(self):
        diag = FailureDiagnosis(
            mode=FailureMode.NONE,
            summary="",
            remediation="",
            retryable=False,
        )
        assert should_try_next(diag) is False

    def test_import_error_no_fallback(self):
        diag = FailureDiagnosis(
            mode=FailureMode.IMPORT_ERROR,
            summary="",
            remediation="",
            retryable=False,
        )
        assert should_try_next(diag) is False

    def test_missing_api_key_recoverable_via_cassette(self):
        # Shadow + cassette replay may succeed even without a live API key.
        diag = FailureDiagnosis(
            mode=FailureMode.API_KEY_MISSING,
            summary="",
            remediation="",
            retryable=True,
        )
        assert should_try_next(diag) is True

    def test_browser_error_triggers_fallback(self):
        diag = FailureDiagnosis(
            mode=FailureMode.BROWSER_RUNTIME_ERROR,
            summary="",
            remediation="",
            retryable=True,
        )
        assert should_try_next(diag) is True

    def test_timeout_triggers_fallback(self):
        diag = FailureDiagnosis(
            mode=FailureMode.TIMEOUT,
            summary="",
            remediation="",
            retryable=True,
        )
        assert should_try_next(diag) is True


# ---------------------------------------------------------------------------
# BackendOutput shape
# ---------------------------------------------------------------------------


class TestBackendOutput:
    def test_properties_forward_to_run_output(self):
        ro = RunOutput(success=True, data={"ok": 1}, error="", stdout="", stderr="", returncode=0)
        bo = BackendOutput(
            run_output=ro,
            backend="subprocess",
            provenance=[SourceTag(source=TraceSource.LLM_REAL)],
            confidence=Confidence(score=0.9),
        )
        assert bo.success is True
        assert bo.data == {"ok": 1}
        assert bo.error == ""
