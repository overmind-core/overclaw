"""Main convergence loop: instrument → smoke → classify → patch → repeat.

Public entry point: :func:`run_preflight`.

Loop invariants:

- Every patch is snapshotted into
  ``.overmind/agents/<name>/preflight/snapshots/iter_<N>/`` *before*
  being applied, so the user can roll back any autonomous mutation.
- Secrets are the only blocker the loop cannot resolve on its own.
  When the classifier emits ``missing_secret``, the loop short-circuits
  with ``status="blocked_secrets"`` and a structured ``missing_secrets``
  list — the skill picks that up and asks the user via ``AskQuestion``.
- The loop is idempotent against a green pipeline (no patches, no
  iterations beyond the first).
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
    KIND_QUALITY,
    KIND_RUNTIME_CRASH,
    classify,
    has_blockers,
    missing_secret_keys,
)
from overmind.preflight.hashes import compute_hashes
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
    preflight_snapshots_dir,
)
from overmind.preflight.workspace import WorkingState
from overmind.utils.instrument import instrument_directory

logger = logging.getLogger("overmind.preflight.runner")


_DEFAULT_MAX_ITERS = 5
_DEFAULT_MAX_ROWS = 2
_DEFAULT_TIMEOUT = 120


def run_preflight(
    agent_name: str,
    *,
    max_iters: int = _DEFAULT_MAX_ITERS,
    max_rows: int = _DEFAULT_MAX_ROWS,
    timeout: int = _DEFAULT_TIMEOUT,
    secrets_provided: dict[str, str] | None = None,
) -> PreflightReport:
    """Run the full preflight convergence loop for *agent_name*.

    *secrets_provided* (optional) — if the host skill already collected
    a credential answer from the user this turn, the runner persists it
    via :func:`overmind.preflight.secrets_scan.set_secret` before the
    first iteration so a single invocation can both accept and verify
    new keys.
    """
    load_overmind_dotenv()
    load_agent_dotenv(agent_name)

    pf_dir = preflight_dir(agent_name)
    pf_dir.mkdir(parents=True, exist_ok=True)
    log_path = preflight_log_path(agent_name)
    snapshots_root = preflight_snapshots_dir(agent_name)
    snapshots_root.mkdir(parents=True, exist_ok=True)

    log_handle = log_path.open("a", encoding="utf-8")
    _log(log_handle, "preflight_start", {"agent": agent_name, "max_iters": max_iters, "max_rows": max_rows})

    if secrets_provided:
        from overmind.preflight.secrets_scan import set_secret

        for key, value in secrets_provided.items():
            outcome = set_secret(agent_name, key, value, validate=False)
            _log(log_handle, "secret_provided", {"key": key, "ok": outcome.get("status") == "ok"})

    spec_dir = agent_setup_spec_dir(agent_name)
    spec_path = spec_dir / "eval_spec.json"
    dataset_path = spec_dir / "dataset.json"
    if not spec_path.is_file() or not dataset_path.is_file():
        report = _early_exit_report(
            agent_name,
            status=STATUS_BLOCKED_NO_CONVERGENCE,
            message=("eval_spec.json or dataset.json missing — run /overmind-generate-spec-and-dataset first."),
            log_path=log_path,
            snapshots_root=snapshots_root,
        )
        _log(log_handle, "missing_artifacts", {"spec": str(spec_path), "dataset": str(dataset_path)})
        log_handle.close()
        report.save(preflight_report_path(agent_name))
        return report

    state = WorkingState.load(agent_name)

    # Instrument once up front so the very first smoke run sees `@observe()`
    # decorators on every agent function.  Idempotent (`is_instrumented`
    # short-circuits) so re-running preflight is cheap.
    if state.instrumented_dir.is_dir():
        modified = instrument_directory(str(state.instrumented_dir))
        if modified:
            _log(log_handle, "initial_instrumentation", {"files_modified": modified})

    trace_path = pf_dir / "trace.jsonl"
    patches_applied: list[dict] = []
    issues_remaining: list[dict] = []
    last_smoke = None
    final_status = STATUS_BLOCKED_NO_CONVERGENCE
    final_message = ""

    iteration = 0
    while iteration < max_iters:
        iteration += 1
        _log(log_handle, "iteration_begin", {"iter": iteration})

        # Snapshot before any patch so rollback is always possible.
        snap_dir = snapshots_root / f"iter_{iteration:02d}"
        state.snapshot_into(snap_dir)

        smoke = run_smoke(
            agent_name,
            eval_spec_path=str(spec_path),
            dataset_path=str(dataset_path),
            max_rows=max_rows,
            timeout=timeout,
            trace_file=trace_path,
        )
        last_smoke = smoke
        _log(log_handle, "smoke_run", smoke_to_jsonable(smoke))

        issues = classify(
            smoke,
            eval_spec=state.eval_spec,
            entrypoint_path=str(state.entrypoint_path) if state.entrypoint_path else None,
        )

        if has_blockers(issues):
            keys = missing_secret_keys(issues)
            final_status = STATUS_BLOCKED_SECRETS
            final_message = (
                f"Missing credentials: {', '.join(keys)}."
                if keys
                else "Provider authentication failed; supply the appropriate API key."
            )
            issues_remaining = [i.to_dict() for i in issues]
            _log(log_handle, "blocked_secrets", {"keys": keys})
            break

        # Anything that's not "fix" severity is left for optimize / quality
        # tracking (KIND_RUNTIME_CRASH, KIND_QUALITY).
        fixable = [i for i in issues if i.severity == "fix"]
        non_fixable = [i for i in issues if i.severity != "fix"]

        if not fixable:
            # No more deterministic plumbing fixes to apply — pipeline is
            # as healthy as preflight can make it.  Decide green vs
            # green_with_quality_notes based on the runtime signal.
            if any(i.kind == KIND_QUALITY for i in non_fixable) or _baseline_is_low(smoke):
                final_status = STATUS_GREEN_QUALITY
                final_message = "Pipeline runs end-to-end. Baseline score is low — leave the rest to overmind optimize."
            elif any(i.kind == KIND_RUNTIME_CRASH for i in non_fixable):
                final_status = STATUS_GREEN_QUALITY
                final_message = (
                    "Pipeline runs but some cases crash inside the agent. "
                    "Those are agent-quality bugs for overmind optimize to fix."
                )
            else:
                final_status = STATUS_GREEN
                final_message = "Pipeline is healthy and ready for overmind optimize."
            issues_remaining = [i.to_dict() for i in non_fixable]
            _log(log_handle, "converged", {"status": final_status})
            break

        # Apply every fixable issue this iteration, then loop.
        # Sort so high-leverage handlers (e.g. entrypoint repair) run
        # before fallback ones (e.g. spec drop).
        fixable = autofix.sort_issues(fixable)
        applied_this_iter = 0
        for issue in fixable:
            patches = autofix.fix(state, issue)
            if not patches:
                continue
            for patch in patches:
                patch.iteration = iteration
                file_path = Path(patch.file)
                patch.before_hash = state.file_hash(file_path) if file_path.is_file() else ""
            spec_changed, ds_changed = state.persist()
            for patch in patches:
                file_path = Path(patch.file)
                patch.after_hash = state.file_hash(file_path) if file_path.is_file() else ""
                patches_applied.append(patch.to_dict())
                _log(log_handle, "patch", patch.to_dict())
                applied_this_iter += 1
            _log(log_handle, "persisted", {"eval_spec": spec_changed, "dataset": ds_changed})

        # Some handlers (entrypoint repair, deps add) request a fresh
        # instrumentation pass so the next smoke run sees the updated
        # source.  instrument_directory is idempotent.
        if state.reinstrument_requests and state.instrumented_dir.is_dir():
            modified = instrument_directory(str(state.instrumented_dir))
            _log(
                log_handle,
                "reinstrumented",
                {
                    "requested": sorted(state.reinstrument_requests),
                    "files_modified": modified,
                },
            )
            state.reinstrument_requests.clear()

        if applied_this_iter == 0:
            # Classifier emitted fixable issues but no handler produced
            # a real change — break to avoid an infinite no-op loop.
            final_status = STATUS_BLOCKED_NO_CONVERGENCE
            final_message = (
                "Issues were detected but no autonomous patch could be applied. See preflight.log for details."
            )
            issues_remaining = [i.to_dict() for i in fixable + non_fixable]
            _log(log_handle, "stalled", {"issues": [i.to_dict() for i in issues]})
            break

    if iteration >= max_iters and final_status == STATUS_BLOCKED_NO_CONVERGENCE and last_smoke is not None:
        # Hit the budget — record what's left.
        residual = classify(last_smoke, eval_spec=state.eval_spec)
        issues_remaining = [i.to_dict() for i in residual]
        final_message = (
            f"Did not converge within {max_iters} iterations. "
            "Inspect preflight.log and rerun after addressing the residual issues."
        )

    hashes = compute_hashes(agent_name)

    report = PreflightReport(
        status=final_status,
        agent_name=agent_name,
        iterations=iteration,
        baseline_score=last_smoke.baseline_score if last_smoke else None,
        span_count=last_smoke.span_count if last_smoke else 0,
        cases_run=len(last_smoke.cases) if last_smoke else 0,
        cases_succeeded=last_smoke.succeeded() if last_smoke else 0,
        cases_failed=last_smoke.failed() if last_smoke else 0,
        hashes=hashes,
        patches_applied=patches_applied,
        issues_remaining=issues_remaining,
        missing_secrets=(
            missing_secret_keys(classify(last_smoke, eval_spec=state.eval_spec))
            if last_smoke and final_status == STATUS_BLOCKED_SECRETS
            else []
        ),
        secrets_env_path=str(_agent_env_path(agent_name)),
        snapshots_dir=str(snapshots_root),
        log_path=str(log_path),
        message=final_message,
    )
    report.save(preflight_report_path(agent_name))
    _log(log_handle, "preflight_end", {"status": final_status, "iterations": iteration})
    log_handle.close()
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_env_path(agent_name: str) -> Path:
    from overmind.core.paths import agent_env_path

    return agent_env_path(agent_name)


def _baseline_is_low(smoke) -> bool:
    """Treat <0.05 baseline as a quality note rather than a perfect green."""
    if smoke.baseline_score is None:
        return False
    return smoke.baseline_score < 0.05


def _log(handle, event: str, payload: dict) -> None:
    line = json.dumps({"ts": time.time(), "event": event, **payload}, default=str)
    handle.write(line + "\n")
    handle.flush()
    logger.debug("preflight %s %s", event, payload)


def _early_exit_report(
    agent_name: str,
    *,
    status: str,
    message: str,
    log_path: Path,
    snapshots_root: Path,
) -> PreflightReport:
    return PreflightReport(
        status=status,
        agent_name=agent_name,
        iterations=0,
        baseline_score=None,
        span_count=0,
        cases_run=0,
        cases_succeeded=0,
        cases_failed=0,
        hashes={},
        patches_applied=[],
        issues_remaining=[],
        missing_secrets=[],
        secrets_env_path=str(_agent_env_path(agent_name)),
        snapshots_dir=str(snapshots_root),
        log_path=str(log_path),
        message=message,
    )
