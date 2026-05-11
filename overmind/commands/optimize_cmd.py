"""
Overmind optimize — Agent Optimizer

Usage:
    overmind optimize <agent-name>
    overmind optimize <agent-name> --fast
"""

import logging

from rich.console import Console

from overmind import SpanType, attrs, set_tag
from overmind.client import flush_pending_api_updates
from overmind.core.paths import load_agent_dotenv
from overmind.core.registry import get_agent_id
from overmind.optimize.config import collect_config
from overmind.optimize.optimizer import Optimizer
from overmind.preflight import load_report, preflight_report_path
from overmind.preflight.state import GREEN_STATUSES
from overmind.storage import configure_storage
from overmind.utils.tracing import force_flush_traces, traced

logger = logging.getLogger("overmind.commands.optimize")


def _preflight_advisory(agent_name: str) -> None:
    """Print a non-blocking hint about preflight status.

    Preflight is *recommended* before optimize but not required: if the
    user wants to head straight into optimize without smoke-testing the
    pipeline first, that's their call.  We only warn when a report
    exists and is non-green, since that's strictly more information
    than we'd otherwise have.
    """
    console = Console(stderr=True)
    report = load_report(agent_name)
    if report is None:
        logger.info(
            "optimize: no preflight report for %s — running without smoke check. "
            "Run `overmind preflight run %s` (or /overmind-preflight) for "
            "an upfront validation pass.",
            agent_name,
            agent_name,
        )
        return

    if report.status in GREEN_STATUSES:
        logger.info(
            "optimize: preflight signed off status=%s baseline=%.4f iterations=%d",
            report.status,
            report.baseline_score or 0.0,
            report.iterations,
        )
        return

    report_path = preflight_report_path(agent_name)
    console.print(
        f"\n  [bold yellow]Notice:[/bold yellow] last preflight status is "
        f"[bold]{report.status}[/bold] — proceeding with optimize anyway.\n"
        f"  {report.message}\n"
        f"  Inspect the report:  [cyan]{report_path}[/cyan]\n"
        f"  Re-run preflight:    [bold]overmind preflight run {agent_name}[/bold]\n"
    )
    if report.missing_secrets:
        console.print(
            f"  Missing credentials reported by preflight: [bold]{', '.join(report.missing_secrets)}[/bold]\n"
        )


@traced(span_name="overmind_optimize", type=SpanType.WORKFLOW)
def main(
    agent_name: str,
    fast: bool = False,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> None:
    logger.info("optimize: start agent=%s fast=%s", agent_name, fast)

    # Load agent-specific .env before anything else so the agent's credentials
    # are available throughout the entire optimize run (config collection,
    # agent execution, and evaluation).
    load_agent_dotenv(agent_name)

    # Soft advisory only — preflight is recommended but optional.
    _preflight_advisory(agent_name)

    config = collect_config(
        agent_name=agent_name,
        fast=fast,
        scope_globs=scope_globs,
        max_files=max_files,
        max_chars=max_chars,
    )
    logger.info(
        "optimize: collected config agent_path=%s iterations=%d parallel=%s",
        config.agent_path,
        config.iterations,
        getattr(config, "parallel", False),
    )

    # CLI-level flags
    set_tag(attrs.COMMAND, "optimize")
    set_tag(attrs.OPTIMIZE_AGENT_NAME, agent_name)
    set_tag(attrs.AGENT_NAME, agent_name)
    set_tag(attrs.OPTIMIZE_FAST, fast)

    # Refresh agent_id from registry in case setup just created/updated it
    config.agent_id = get_agent_id(agent_name)

    logger.info("optimize: storage agent_id=%s", config.agent_id)
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
        logger.warning("optimize: interrupted by user (KeyboardInterrupt) agent=%s", agent_name)
        _finalize_failed_job(optimizer, reason="Interrupted by user (KeyboardInterrupt)")
        raise
    except BaseException as exc:
        logger.exception("optimize: run failed for agent=%s", agent_name)
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        _finalize_failed_job(optimizer, reason=reason)
        raise
    logger.info("optimize: run complete agent=%s", agent_name)


def _finalize_failed_job(optimizer: Optimizer, *, reason: str) -> None:
    """Mark the optimize Job as FAILED on the API and flush partial progress.

    Called when the optimize loop is interrupted (Ctrl-C) or aborts with an
    exception. Whatever iterations / experiments have run up until that
    point are already streamed to the backend via :class:`ApiReporter` and
    OTLP spans; this final hook just (a) flips the Job status to ``failed``
    so the UI stops showing it as ``running`` and (b) blocks long enough for
    in-flight HTTP / OTLP traffic to drain so partial state is durable.
    """
    reporter = getattr(optimizer, "_reporter", None)
    if reporter is not None:
        try:
            reporter.on_failed(reason=reason)
        except Exception:
            logger.exception("optimize: reporter.on_failed raised; continuing teardown")
    try:
        flush_pending_api_updates(timeout=10.0)
    except Exception:
        logger.exception("optimize: flush_pending_api_updates raised; continuing teardown")
    try:
        force_flush_traces(timeout_millis=10_000)
    except Exception:
        logger.exception("optimize: force_flush_traces raised; continuing teardown")
