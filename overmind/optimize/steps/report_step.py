"""``overmind optimize-step report`` — render report.md from final state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import force_flush_traces, observe_safe

logger = logging.getLogger("overmind.optimize.steps.report")


@observe_safe(span_name="overmind.optimize.report", type=SpanType.WORKFLOW)
def run_report(state: SkillRunState) -> dict[str, Any]:
    set_tag(attrs.OPTIMIZE_PHASE, "report")
    set_tag(attrs.OPTIMIZE_STEP, "report")
    if state.job_id:
        set_tag(attrs.JOB_ID, state.job_id)
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

    # ---- Final-state OTel tags: drive ``Job`` columns + flip status ----
    baseline = float(state.baseline_score or state.best_score or 0.0)
    set_tag(attrs.OPTIMIZE_FINAL_BEST_SCORE, float(state.best_score))
    set_tag(attrs.OPTIMIZE_REPORT_BEST_SCORE, float(state.best_score))
    set_tag(attrs.OPTIMIZE_BASELINE_SCORE, baseline)
    set_tag(attrs.OPTIMIZE_REPORT_IMPROVEMENT, float(state.best_score - baseline))
    set_tag(attrs.OPTIMIZE_TOTAL_ACCEPTED, len(state.successful_changes or []))
    set_tag(attrs.OPTIMIZE_TOTAL_REJECTED, len(state.failed_attempts or []))
    set_tag(attrs.OPTIMIZE_STALL_COUNT, int(state.stall_count))

    # Persist the rendered ``report.md`` + final best agent code onto
    # the active span so OTLP populates ``Job.report_markdown`` and
    # ``Job.best_agent_code``.
    try:
        report_text = Path(report_path).read_text(encoding="utf-8")
        set_tag(attrs.OPTIMIZE_REPORT_MARKDOWN, report_text)
    except Exception:
        logger.debug("report: failed to stamp report.md", exc_info=True)
    if state.best_code_path and Path(state.best_code_path).is_file():
        try:
            set_tag(
                attrs.OPTIMIZE_BEST_AGENT_CODE,
                Path(state.best_code_path).read_text(encoding="utf-8"),
            )
        except Exception:
            logger.debug("report: failed to stamp best_agent_code", exc_info=True)

    # Terminal lifecycle marker — flips ``Job.status`` to ``completed``.
    set_tag(attrs.OPTIMIZE_RUN_STATUS, "completed")
    force_flush_traces(timeout_millis=3000)

    return {
        "status": "ok",
        "step": "report",
        "report_path": report_path,
        "best_score": state.best_score,
        "baseline_score": state.baseline_score,
        "iterations_completed": state.iteration,
        "early_stopping_triggered": state.early_stopping_triggered,
    }
