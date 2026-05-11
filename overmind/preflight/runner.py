"""Preflight: run agent end-to-end, classify failures, patch what we can, report.

Two passes at most:
  Pass 1 — smoke run → classify → apply deterministic fixes
  Pass 2 — re-smoke (only when fixes were applied) → classify → report

No convergence loop, no snapshots, no LLM calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from overmind.core.paths import (
    agent_setup_spec_dir,
    load_agent_dotenv,
    load_overmind_dotenv,
)
from overmind.preflight import autofix
from overmind.preflight.classifier import (
    KIND_DEGENERATE_OUTPUT,
    KIND_QUALITY,
    KIND_RUNTIME_CRASH,
    classify,
    has_blockers,
    missing_secret_keys,
)
from overmind.preflight.smoke import run_smoke, smoke_to_jsonable
from overmind.preflight.state import (
    STATUS_BLOCKED_NO_CONVERGENCE,
    STATUS_BLOCKED_SECRETS,
    STATUS_GREEN,
    STATUS_GREEN_QUALITY,
    PreflightReport,
    preflight_dir,
    preflight_log_path,
    preflight_report_path,
)
from overmind.preflight.workspace import WorkingState
from overmind.utils.instrument import instrument_directory

logger = logging.getLogger("overmind.preflight.runner")

_DEFAULT_MAX_ROWS = 2
_DEFAULT_TIMEOUT = 120

_QUALITY_KINDS = frozenset({KIND_RUNTIME_CRASH, KIND_DEGENERATE_OUTPUT, KIND_QUALITY})


def run_preflight(
    agent_name: str,
    *,
    max_rows: int = _DEFAULT_MAX_ROWS,
    timeout: int = _DEFAULT_TIMEOUT,
    secrets_provided: dict[str, str] | None = None,
) -> PreflightReport:
    """Run preflight for *agent_name*: smoke-test end-to-end, auto-fix plumbing, report."""
    load_overmind_dotenv()
    load_agent_dotenv(agent_name)

    pf_dir = preflight_dir(agent_name)
    pf_dir.mkdir(parents=True, exist_ok=True)
    log_path = preflight_log_path(agent_name)
    log_handle = log_path.open("a", encoding="utf-8")
    _log(log_handle, "preflight_start", {"agent": agent_name, "max_rows": max_rows})

    if secrets_provided:
        from overmind.preflight.secrets_scan import set_secret

        for key, value in secrets_provided.items():
            outcome = set_secret(agent_name, key, value, validate=False)
            _log(log_handle, "secret_provided", {"key": key, "ok": outcome.get("status") == "ok"})

    spec_dir = agent_setup_spec_dir(agent_name)
    spec_path = spec_dir / "eval_spec.json"
    dataset_path = spec_dir / "dataset.json"

    if not spec_path.is_file() or not dataset_path.is_file():
        report = _make_report(
            agent_name,
            status=STATUS_BLOCKED_NO_CONVERGENCE,
            message="eval_spec.json or dataset.json missing — run /overmind-generate-spec-and-dataset first.",
            log_path=log_path,
        )
        _log(log_handle, "missing_artifacts", {"spec": str(spec_path), "dataset": str(dataset_path)})
        log_handle.close()
        report.save(preflight_report_path(agent_name))
        return report

    state = WorkingState.load(agent_name)

    # Instrument upfront (idempotent).
    if state.instrumented_dir.is_dir():
        modified = instrument_directory(str(state.instrumented_dir))
        if modified:
            _log(log_handle, "instrumented", {"files": modified})

    trace_path = pf_dir / "trace.jsonl"

    # --- Pass 1 ---
    smoke1 = run_smoke(
        agent_name,
        eval_spec_path=str(spec_path),
        dataset_path=str(dataset_path),
        max_rows=max_rows,
        timeout=timeout,
        trace_file=trace_path,
    )
    _log(log_handle, "smoke_pass_1", smoke_to_jsonable(smoke1))

    issues1 = classify(smoke1, eval_spec=state.eval_spec)

    # Missing credentials → hard block; surface to user.
    if has_blockers(issues1):
        keys = missing_secret_keys(issues1)
        report = _make_report(
            agent_name,
            status=STATUS_BLOCKED_SECRETS,
            message=f"Missing credentials: {', '.join(keys)}." if keys else "Provider authentication failed.",
            log_path=log_path,
            issues_remaining=[i.to_dict() for i in issues1],
            missing_secrets=keys,
            smoke=smoke1,
        )
        _log(log_handle, "blocked_secrets", {"keys": keys})
        log_handle.close()
        report.save(preflight_report_path(agent_name))
        return report

    # Apply all deterministic fixes.
    fixable = autofix.sort_issues([i for i in issues1 if i.severity == "fix"])
    patches_applied: list[dict] = []

    for issue in fixable:
        patches = autofix.fix(state, issue)
        for p in patches:
            p.iteration = 1
            patches_applied.append(p.to_dict())
            _log(log_handle, "patch", p.to_dict())

    if patches_applied:
        state.persist()
        _log(log_handle, "fixes_persisted", {"count": len(patches_applied)})
        if state.reinstrument_requests and state.instrumented_dir.is_dir():
            instrument_directory(str(state.instrumented_dir))
            state.reinstrument_requests.clear()

    # --- Pass 2: re-validate only when we made changes ---
    if patches_applied:
        smoke2 = run_smoke(
            agent_name,
            eval_spec_path=str(spec_path),
            dataset_path=str(dataset_path),
            max_rows=max_rows,
            timeout=timeout,
            trace_file=trace_path,
        )
        _log(log_handle, "smoke_pass_2", smoke_to_jsonable(smoke2))
        final_smoke = smoke2
        issues_final = classify(smoke2, eval_spec=state.eval_spec)
    else:
        final_smoke = smoke1
        issues_final = issues1

    # Determine final status.
    remaining_fixable = [i for i in issues_final if i.severity == "fix"]
    non_fixable = [i for i in issues_final if i.severity != "fix"]

    if remaining_fixable:
        final_status = STATUS_BLOCKED_NO_CONVERGENCE
        final_message = (
            "Some plumbing issues could not be auto-fixed. "
            "See preflight.log and address the issues_remaining before running optimize."
        )
    elif any(i.kind in _QUALITY_KINDS for i in non_fixable):
        final_status = STATUS_GREEN_QUALITY
        final_message = (
            "Pipeline runs end-to-end. Some agent quality issues noted "
            "(crashes, low score, or degenerate output) — leave those to overmind optimize."
        )
    elif final_smoke.failed() > 0 or final_smoke.preflight_error:
        final_status = STATUS_GREEN_QUALITY
        final_message = "Pipeline runs but some cases failed inside the agent — overmind optimize will fix those."
    else:
        final_status = STATUS_GREEN
        final_message = "Pipeline is healthy and ready for overmind optimize."

    report = _make_report(
        agent_name,
        status=final_status,
        message=final_message,
        log_path=log_path,
        patches_applied=patches_applied,
        issues_remaining=[i.to_dict() for i in non_fixable + remaining_fixable],
        smoke=final_smoke,
    )
    _log(log_handle, "preflight_end", {"status": final_status, "patches": len(patches_applied)})
    log_handle.close()
    report.save(preflight_report_path(agent_name))
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    agent_name: str,
    *,
    status: str,
    message: str,
    log_path: Path,
    patches_applied: list[dict] | None = None,
    issues_remaining: list[dict] | None = None,
    missing_secrets: list[str] | None = None,
    smoke=None,
) -> PreflightReport:
    from overmind.core.paths import agent_env_path

    return PreflightReport(
        status=status,
        agent_name=agent_name,
        iterations=1 if patches_applied else 0,
        baseline_score=smoke.baseline_score if smoke else None,
        span_count=smoke.span_count if smoke else 0,
        cases_run=len(smoke.cases) if smoke else 0,
        cases_succeeded=smoke.succeeded() if smoke else 0,
        cases_failed=smoke.failed() if smoke else 0,
        patches_applied=patches_applied or [],
        issues_remaining=issues_remaining or [],
        missing_secrets=missing_secrets or [],
        secrets_env_path=str(agent_env_path(agent_name)),
        snapshots_dir="",
        log_path=str(log_path),
        message=message,
    )


def _log(handle, event: str, payload: dict) -> None:
    line = json.dumps({"ts": time.time(), "event": event, **payload}, default=str)
    handle.write(line + "\n")
    handle.flush()
    logger.debug("preflight %s %s", event, payload)
