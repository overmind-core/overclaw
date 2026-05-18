"""``overmind optimize-step evaluate`` — score one candidate worktree."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.core.registry import project_root, project_root_from_agent_file
from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import force_flush_traces, observe_safe
from overmind.utils.atomic_io import atomic_write_json

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

    # Resolve the entry file inside the worktree. Order of preference:
    #   1. ``plan.json["entry_file"]`` — the canonical relative path the
    #      diagnose step wrote when it materialised the worktree
    #      (multi-file bundles need this; the file lives at
    #      ``<worktree>/<package>/.../entry.py``, not at the worktree
    #      root).
    #   2. The agent_path resolved relative to the optimizer's project
    #      root — covers older worktrees written before plan.json
    #      persisted ``entry_file``.
    #   3. The bare basename of ``cfg.agent_path`` — last-ditch fallback
    #      for single-file bundles where the file is at the worktree
    #      root.
    plan_path = worktree / "plan.json"
    plan_entry_file: str | None = None
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text())
            plan_entry_file = plan.get("entry_file") or None
        except Exception:
            plan_entry_file = None

    entry_candidates: list[str] = []
    if plan_entry_file:
        entry_candidates.append(plan_entry_file)

    # Project-root-relative agent path is the canonical relative layout
    # the diagnose step uses when materialising multi-file worktrees.
    # Compute it from ``agent_path`` directly so this fallback works
    # even though :class:`Optimizer` does not build its bundle in
    # ``__init__`` (only ``run_baseline`` / ``run_diagnose`` do).
    try:
        agent_path_abs = Path(cfg.agent_path).resolve()
        root = project_root_from_agent_file(str(agent_path_abs)) or project_root()
        if root is not None:
            try:
                root_resolved = Path(root).resolve()
                rel = str(agent_path_abs.relative_to(root_resolved))
                if rel and rel not in entry_candidates:
                    entry_candidates.append(rel)
            except ValueError:
                pass
    except Exception:
        pass

    base_name = Path(cfg.agent_path).name
    if base_name not in entry_candidates:
        entry_candidates.append(base_name)

    entry_path: Path | None = None
    entry_file: str = entry_candidates[0]
    for rel in entry_candidates:
        candidate_path = worktree / rel
        if candidate_path.is_file():
            entry_path = candidate_path
            entry_file = rel
            break

    if entry_path is None:
        return {
            "status": "error",
            "error": "missing_entry_file",
            "message": (
                f"No entry file found inside {worktree}. Tried "
                f"{entry_candidates!r}. If you upgraded from a "
                "single-file bundle, re-run `overmind optimize-step "
                "diagnose` so plan.json captures the new relative path."
            ),
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
    atomic_write_json(score_path, result)

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
