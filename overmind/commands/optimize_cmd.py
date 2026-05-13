"""
Overmind optimize — Agent Optimizer

Usage:
    overmind optimize <agent-name>
    overmind optimize <agent-name> --fast
"""

import logging

from overmind import SpanType, attrs, set_tag
from overmind.core.paths import load_agent_dotenv
from overmind.core.registry import get_agent_id
from overmind.optimize.config import collect_config
from overmind.optimize.optimizer import Optimizer
from overmind.storage import configure_storage
from overmind.tracing import force_flush_traces, observe_safe

logger = logging.getLogger("overmind.commands.optimize")


@observe_safe(span_name="overmind.optimize", type=SpanType.WORKFLOW)
def main(
    agent_name: str,
    fast: bool = False,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> None:
    logger.info(f"optimize: start agent={agent_name} fast={fast}")

    # Load agent-specific .env before anything else so the agent's credentials
    # are available throughout the entire optimize run (config collection,
    # agent execution, and evaluation).
    load_agent_dotenv(agent_name)

    config = collect_config(
        agent_name=agent_name,
        fast=fast,
        scope_globs=scope_globs,
        max_files=max_files,
        max_chars=max_chars,
    )
    logger.info(
        f"optimize: collected config agent_path={config.agent_path} "
        f"iterations={config.iterations} parallel={getattr(config, 'parallel', False)}"
    )

    # CLI-level flags
    set_tag(attrs.COMMAND, "optimize")
    set_tag(attrs.OPTIMIZE_AGENT_NAME, agent_name)
    set_tag(attrs.AGENT_NAME, agent_name)
    set_tag(attrs.OPTIMIZE_FAST, fast)

    # Refresh agent_id from registry in case setup just created/updated it
    config.agent_id = get_agent_id(agent_name)

    logger.info(f"optimize: storage agent_id={config.agent_id}")
    configure_storage(
        agent_path=config.agent_path,
        agent_id=config.agent_id,
        agent_name=agent_name,
    )

    # Config-level tags — everything the user chose or defaulted to
    set_tag(attrs.OPTIMIZE_AGENT_PATH, config.agent_path)
    set_tag(attrs.OPTIMIZE_ENTRYPOINT_FN, config.entrypoint_fn)
    set_tag(attrs.OPTIMIZE_STORAGE_BACKEND, "api")
    set_tag(attrs.OPTIMIZE_ANALYZER_MODEL, config.analyzer_model or "")
    set_tag(attrs.OPTIMIZE_LLM_JUDGE_MODEL, config.llm_judge_model or "disabled")
    set_tag(attrs.OPTIMIZE_ITERATIONS, config.iterations)
    set_tag(attrs.OPTIMIZE_CANDIDATES_PER_ITERATION, config.candidates_per_iteration)
    set_tag(attrs.OPTIMIZE_PARALLEL, config.parallel)
    set_tag(attrs.OPTIMIZE_MAX_WORKERS, config.max_workers)
    set_tag(attrs.OPTIMIZE_RUNS_PER_EVAL, config.runs_per_eval)
    set_tag(attrs.OPTIMIZE_REGRESSION_THRESHOLD, config.regression_threshold)
    set_tag(attrs.OPTIMIZE_HOLDOUT_RATIO, config.holdout_ratio)
    set_tag(attrs.OPTIMIZE_HOLDOUT_ENFORCEMENT, config.holdout_enforcement)
    set_tag(attrs.OPTIMIZE_EARLY_STOPPING_PATIENCE, config.early_stopping_patience)
    set_tag(attrs.OPTIMIZE_CROSS_RUN_PERSISTENCE, config.cross_run_persistence)
    set_tag(attrs.OPTIMIZE_FAILURE_CLUSTERING, config.failure_clustering)
    set_tag(attrs.OPTIMIZE_ADAPTIVE_FOCUS, config.adaptive_focus)
    set_tag(attrs.OPTIMIZE_MODEL_BACKTESTING, config.model_backtesting)
    if config.backtest_models:
        set_tag(attrs.OPTIMIZE_BACKTEST_MODELS, ",".join(config.backtest_models))
    set_tag(attrs.OPTIMIZE_EVAL_SPEC_PATH, config.eval_spec_path or "")
    set_tag(attrs.OPTIMIZE_DATA_PATH, config.data_path or "")

    optimizer = Optimizer(config)
    try:
        optimizer.run()
    except KeyboardInterrupt:
        logger.warning(f"optimize: interrupted by user (KeyboardInterrupt) agent={agent_name}")
        _finalize_failed_job(optimizer, reason="Interrupted by user (KeyboardInterrupt)")
        raise
    except BaseException as exc:
        logger.exception(f"optimize: run failed for agent={agent_name}")
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        _finalize_failed_job(optimizer, reason=reason)
        raise
    logger.info(f"optimize: run complete agent={agent_name}")


def _finalize_failed_job(_: Optimizer, *, reason: str) -> None:
    """Flush in-flight OTel spans after the optimize loop aborts.

    Called when the optimize loop is interrupted (Ctrl-C) or aborts
    with an exception.  Stamps ``overmind.optimize.run_status = failed``
    + ``overmind.error.message`` on the active span so OTLP flips
    ``Job.status`` to ``failed`` and sweeps any iteration still in
    ``RUNNING`` to ``DISCARD``, then blocks long enough for the
    BatchSpanProcessor to drain pending exports.
    """
    try:
        set_tag(attrs.OPTIMIZE_RUN_STATUS, "failed")
        if reason:
            set_tag(attrs.ERROR_MESSAGE, reason[:1000])
    except Exception:
        logger.debug("optimize: failed to stamp terminal status", exc_info=True)
    try:
        force_flush_traces(timeout_millis=10_000)
    except Exception:
        logger.exception("optimize: force_flush_traces raised; continuing teardown")
