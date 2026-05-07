"""``overmind optimize-step baseline`` — run the baseline phase."""

from __future__ import annotations

import logging
from typing import Any

from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState

logger = logging.getLogger("overmind.optimize.steps.baseline")


def run_baseline(state: SkillRunState) -> dict[str, Any]:
    """Reconstruct an Optimizer from *state* and run :meth:`Optimizer.run_baseline_phase`."""
    cfg = state.to_config()
    optimizer = Optimizer(cfg)

    result = optimizer.run_baseline_phase()

    state.phase = "baseline_complete"
    state.dataset_size = int(result["dataset_size"])
    state.train_size = int(result["train_size"])
    state.holdout_size = int(result["holdout_size"])
    state.baseline_score = float(result["baseline_score"])
    state.best_score = float(result["best_score"])
    state.best_iteration = 0
    state.best_code_path = result["best_code_path"]
    state.best_files_dir = result.get("best_files_dir", "")
    state.best_case_scores = list(result.get("best_case_scores", []))
    state.working_path = result["best_code_path"]
    state.working_dir = result.get("best_files_dir", "")
    state.output_dir = result["output_dir"]
    state.iteration = 0
    state.stall_count = 0
    state.record_iteration({
        "iteration": "baseline",
        "score": state.baseline_score,
        "status": "keep",
        "description": "Initial baseline",
    })
    state.save()

    return {
        "status": "ok",
        "step": "baseline",
        "state_path": state.state_path,
        "baseline_score": state.baseline_score,
        "best_score": state.best_score,
        "dataset_size": state.dataset_size,
        "train_size": state.train_size,
        "holdout_size": state.holdout_size,
        "working_path": state.working_path,
        "working_dir": state.working_dir,
        "output_dir": state.output_dir,
    }
