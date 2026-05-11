"""
Overmind optimize — Agent Optimizer

Usage:
    overmind optimize <agent-name>
    overmind optimize <agent-name> --fast
"""

import logging
import os

from rich.console import Console

from overmind import SpanType, attrs, set_tag
from overmind.client import flush_pending_api_updates
from overmind.core.paths import load_agent_dotenv
from overmind.core.registry import get_agent_id
from overmind.optimize.config import collect_config
from overmind.optimize.optimizer import Optimizer
from overmind.preflight import load_report, preflight_report_path
from overmind.preflight.hashes import compute_hashes, hashes_match
from overmind.preflight.state import GREEN_STATUSES
from overmind.storage import configure_storage
from overmind.utils.tracing import force_flush_traces, traced

logger = logging.getLogger("overmind.commands.optimize")


def _enforce_preflight_gate(agent_name: str) -> None:
    """Refuse to optimize unless preflight is green and hashes match.

    Bypass with the ``OVERMIND_SKIP_PREFLIGHT=1`` env var or
    ``--skip-preflight`` CLI flag (when running interactively this is a
    foot-gun — the optimize loop will fail later for the same plumbing
    reasons preflight would have caught and fixed).
    """
    if os.environ.get("OVERMIND_SKIP_PREFLIGHT") == "1":
        logger.warning(
            "optimize: OVERMIND_SKIP_PREFLIGHT=1 — bypassing the preflight gate; "
            "infrastructure failures during optimize are now your responsibility.",
        )
        return

    console = Console(stderr=True)
    report_path = preflight_report_path(agent_name)
    report = load_report(agent_name)
    if report is None:
        console.print(
            "\n  [bold red]Error:[/bold red] preflight has not been run for this agent.\n"
            f"  Expected report at: [cyan]{report_path}[/cyan]\n\n"
            "  Run the validation gate first:\n"
            f"    [bold]overmind preflight run {agent_name}[/bold]\n"
            f"  …or invoke the [bold]/overmind-preflight[/bold] skill.\n\n"
            "  To skip this check (not recommended):\n"
            "    [dim]OVERMIND_SKIP_PREFLIGHT=1 overmind optimize <name>[/dim]\n"
        )
        raise SystemExit(2)

    if report.status not in GREEN_STATUSES:
        console.print(
            f"\n  [bold red]Error:[/bold red] preflight status is "
            f"[bold]{report.status}[/bold] — pipeline is not ready for optimize.\n"
            f"  {report.message}\n\n"
            f"  Inspect the report:  [cyan]{report_path}[/cyan]\n"
            f"  Re-run the gate:    [bold]overmind preflight run {agent_name}[/bold]\n"
        )
        if report.missing_secrets:
            console.print(
                "  Missing credentials: "
                f"[bold]{', '.join(report.missing_secrets)}[/bold]\n"
                f"  Save each via:    [bold]echo -n <value> | overmind preflight set-secret "
                f"{agent_name} --key <KEY>[/bold]\n"
            )
        raise SystemExit(2)

    fresh = compute_hashes(agent_name)
    ok, diff = hashes_match(report.hashes or {}, fresh)
    if not ok:
        console.print(
            "\n  [bold yellow]Warning:[/bold yellow] preflight is "
            f"[bold]stale[/bold] — these artifacts changed since it ran: "
            f"[bold]{', '.join(diff)}[/bold]\n"
            f"  Re-run:    [bold]overmind preflight run {agent_name}[/bold]\n\n"
            "  To proceed anyway (not recommended):\n"
            "    [dim]OVERMIND_SKIP_PREFLIGHT=1 overmind optimize <name>[/dim]\n"
        )
        raise SystemExit(2)

    logger.info(
        "optimize: preflight gate ok status=%s baseline=%.4f iterations=%d",
        report.status,
        report.baseline_score or 0.0,
        report.iterations,
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

    # Hard gate: refuse to optimize unless preflight has signed off and the
    # artifacts haven't drifted.  This is the contract preflight provides.
    _enforce_preflight_gate(agent_name)

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
