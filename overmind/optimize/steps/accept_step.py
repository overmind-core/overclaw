"""``overmind optimize-step accept`` — apply gates, promote winner.

This MVP implementation applies the simplest meaningful gate: pick the
candidate with the highest ``avg_total`` and promote it iff its score is
strictly greater than the current best. Cross-run regression suite,
holdout enforcement, complexity penalty, and failure-cluster updates
will be ported in a follow-up.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState

logger = logging.getLogger("overmind.optimize.steps.accept")


def _prune_git_worktrees(candidate_dirs: list[str]) -> None:
    """Remove git worktree registrations for all candidate directories.

    Called after accept/reject so stale worktree entries don't accumulate.
    Falls back silently if git is unavailable or the paths aren't git worktrees.
    """
    for candidate_dir in candidate_dirs:
        p = Path(candidate_dir)
        if not p.exists():
            continue
        try:
            git_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if git_result.returncode != 0:
                continue
            git_root = git_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(p)],
                cwd=git_root,
                capture_output=True,
                timeout=15,
            )
        except Exception as exc:
            logger.debug("Skipping worktree prune for %s: %s", candidate_dir, exc)


def _load_candidate_results(path: str) -> list[dict]:
    """Read the candidate-results JSON.

    Accepts either a list of ``{candidate_id, score_path, candidate_dir, ...}``
    entries or a dict ``{candidates: [...]}``.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    if not isinstance(raw, list):
        raise TypeError(f"candidate_results must be a list, got {type(raw).__name__}")
    return raw


def run_accept(
    state: SkillRunState,
    *,
    iteration: int,
    candidate_results_path: str,
) -> dict[str, Any]:
    candidates = _load_candidate_results(candidate_results_path)
    if not candidates:
        state.bump_stall()
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "score": state.best_score,
            "status": "skip",
            "description": "No candidates to evaluate",
        })
        state.save()
        return {
            "status": "ok",
            "step": "accept",
            "iteration": iteration,
            "decision": "no_candidates",
            "best_score": state.best_score,
            "stall_count": state.stall_count,
        }

    # Resolve scores from each score.json
    scored: list[dict] = []
    for cand in candidates:
        score_path = cand.get("score_path")
        if not score_path:
            continue
        try:
            score_data = json.loads(Path(score_path).read_text())
            scored.append({
                "candidate_id": cand["candidate_id"],
                "candidate_dir": cand["candidate_dir"],
                "entry_path": cand.get("entry_path"),
                "avg_total": float(score_data["avg_total"]),
                "evaluation": score_data["evaluation"],
                "case_results": score_data["case_results"],
            })
        except Exception as exc:
            logger.warning("Failed to load score for %s: %s", cand.get("candidate_id"), exc)

    if not scored:
        state.bump_stall()
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "score": state.best_score,
            "status": "crash",
            "description": "All candidates failed to evaluate",
        })
        state.save()
        return {
            "status": "ok",
            "step": "accept",
            "iteration": iteration,
            "decision": "all_crashed",
            "best_score": state.best_score,
            "stall_count": state.stall_count,
        }

    scored.sort(key=lambda c: -c["avg_total"])
    winner = scored[0]
    delta = winner["avg_total"] - state.best_score

    cfg = state.to_config()

    if winner["avg_total"] > state.best_score:
        # Promote: write working_path, update state, reset stall
        optimizer = Optimizer(cfg)
        # Hydrate the optimizer enough that commit_winner can compute paths
        optimizer.best_score = state.best_score
        optimizer.best_code = (
            Path(state.working_path or cfg.agent_path).read_text()
            if Path(state.working_path or cfg.agent_path).is_file()
            else ""
        )
        optimizer._baseline_train_score = state.baseline_score or state.best_score

        optimizer.commit_winner(
            winner_entry_path=winner["entry_path"],
            winner_eval=winner["evaluation"],
            winner_case_results=winner["case_results"],
        )
        # commit_winner wrote both agent_working.* and _latest_items.json
        new_working_path = str(optimizer.output_dir / f"agent_working{Path(cfg.agent_path).suffix or '.py'}")

        state.update_best(
            score=winner["avg_total"],
            iteration=iteration,
            code_path=new_working_path,
            files_dir=state.best_files_dir,
        )
        state.working_path = new_working_path
        state.successful_changes.append({
            "iteration": iteration,
            "candidate_id": winner["candidate_id"],
            "delta": delta,
            "from": state.results[-1].get("score") if state.results else None,
            "to": winner["avg_total"],
        })
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "candidate_id": winner["candidate_id"],
            "score": winner["avg_total"],
            "status": "accept",
            "description": f"Improved by {delta:+.2f}",
        })
        decision = "accept"
    else:
        state.bump_stall()
        state.failed_attempts.append({
            "iteration": iteration,
            "best_candidate_id": winner["candidate_id"],
            "best_score": winner["avg_total"],
            "current_best": state.best_score,
            "delta": delta,
        })
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "candidate_id": winner["candidate_id"],
            "score": winner["avg_total"],
            "status": "reject",
            "description": f"No improvement ({delta:+.2f})",
        })
        decision = "reject"

    state.save()

    early_stop = cfg.early_stopping_patience > 0 and state.stall_count >= cfg.early_stopping_patience
    if early_stop:
        state.early_stopping_triggered = True
        state.save()

    _prune_git_worktrees([c["candidate_dir"] for c in candidates if c.get("candidate_dir")])

    return {
        "status": "ok",
        "step": "accept",
        "iteration": iteration,
        "decision": decision,
        "winner": {
            "candidate_id": winner["candidate_id"],
            "avg_total": winner["avg_total"],
            "delta": delta,
        },
        "all_scores": [{"candidate_id": c["candidate_id"], "avg_total": c["avg_total"]} for c in scored],
        "best_score": state.best_score,
        "stall_count": state.stall_count,
        "early_stop": early_stop,
    }
