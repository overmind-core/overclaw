"""``overmind optimize-step report`` — render report.md from final state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState

logger = logging.getLogger("overmind.optimize.steps.report")


def run_report(state: SkillRunState) -> dict[str, Any]:
    cfg = state.to_config()
    optimizer = Optimizer(cfg)

    # Translate SkillRunState row format into the flat-string row format
    # ``Optimizer._log_result`` produces, so ``_write_report_md`` /
    # ``_generate_report`` see what they expect.
    translated_rows: list[dict[str, str]] = []
    dim_keys = [k for _, k in optimizer.evaluator.get_dimension_labels()]
    for row in state.results:
        flat: dict[str, str] = {
            "iteration": str(row.get("iteration", "?")),
            "avg_score": f"{float(row.get('score', 0)):.1f}",
        }
        for k in dim_keys:
            flat[k] = "0.0"
        flat["status"] = str(row.get("status", ""))
        flat["description"] = str(row.get("description", ""))[:200]
        translated_rows.append(flat)
    optimizer.results = translated_rows
    optimizer.best_score = float(state.best_score)
    optimizer._baseline_train_score = float(state.baseline_score or state.best_score)
    optimizer.successful_changes = list(state.successful_changes)
    optimizer.failed_attempts = list(state.failed_attempts)
    if state.best_code_path and Path(state.best_code_path).is_file():
        optimizer.best_code = Path(state.best_code_path).read_text()

    try:
        report_path = optimizer.render_report_only()
    except Exception as exc:
        logger.exception("report rendering failed")
        return {
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        }

    state.phase = "complete"
    state.save()

    return {
        "status": "ok",
        "step": "report",
        "report_path": report_path,
        "best_score": state.best_score,
        "baseline_score": state.baseline_score,
        "iterations_completed": state.iteration,
        "early_stopping_triggered": state.early_stopping_triggered,
    }
