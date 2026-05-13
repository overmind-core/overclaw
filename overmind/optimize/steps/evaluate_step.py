"""``overmind optimize-step evaluate`` — score one candidate worktree."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import force_flush_traces, observe_safe

logger = logging.getLogger("overmind.optimize.steps.evaluate")


@observe_safe(span_name="overmind.optimize.evaluate", type=SpanType.WORKFLOW)
def run_evaluate(
    state: SkillRunState,
    *,
    iteration: int,
    candidate_id: str,
    candidate_dir: str,
) -> dict[str, Any]:
    set_tag(attrs.OPTIMIZE_ITERATION, int(iteration))
    set_tag(attrs.OPTIMIZE_CANDIDATE_METHOD, candidate_id)
    set_tag(attrs.OPTIMIZE_PHASE, "evaluate_candidate")
    set_tag(attrs.OPTIMIZE_STEP, "evaluate")
    if state.job_id:
        set_tag(attrs.JOB_ID, state.job_id)
    cfg = state.to_config()
    optimizer = Optimizer(cfg)

    worktree = Path(candidate_dir)
    if not worktree.is_dir():
        return {
            "status": "error",
            "error": "missing_worktree",
            "message": f"Candidate worktree not found at {worktree}.",
        }

    # Resolve the entry file inside the worktree. Prefer plan.json's
    # candidate-specific entry; fall back to the original agent_path basename.
    plan_path = worktree / "plan.json"
    entry_file: str
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text())
            entry_file = plan.get("entry_file") or Path(cfg.agent_path).name
        except Exception:
            entry_file = Path(cfg.agent_path).name
    else:
        entry_file = Path(cfg.agent_path).name

    entry_path = worktree / entry_file
    if not entry_path.is_file():
        return {
            "status": "error",
            "error": "missing_entry_file",
            "message": f"No {entry_file} inside {worktree}.",
        }

    run_name = f"iter_{iteration:03d}_{candidate_id}"
    try:
        result = optimizer.evaluate_worktree(str(entry_path), run_name)
    except Exception as exc:
        logger.exception(f"evaluate {run_name} crashed")
        return {
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
            "candidate_id": candidate_id,
            "iteration": iteration,
        }

    score_path = worktree / "score.json"
    score_path.write_text(json.dumps(result, indent=2, default=str))

    avg_total = float(result["avg_total"])
    set_tag(attrs.OPTIMIZE_CANDIDATE_SCORE, avg_total)
    set_tag(attrs.OPTIMIZE_CANDIDATE_METHOD, str(candidate_id))
    force_flush_traces(timeout_millis=1500)

    return {
        "status": "ok",
        "step": "evaluate",
        "iteration": iteration,
        "candidate_id": candidate_id,
        "candidate_dir": str(worktree),
        "entry_path": str(entry_path),
        "score_path": str(score_path),
        "avg_total": avg_total,
    }
