"""
Overmind optimize — Agent Optimizer

Usage:
    overmind optimize <agent-name>
    overmind optimize <agent-name> --fast
"""

import logging

from overmind import SpanType, attrs, set_tag
from overmind.client import ApiReporter
from overmind.core.registry import get_agent_id
from overmind.optimize.config import collect_config
from overmind.optimize.optimizer import Optimizer
from overmind.storage import configure_storage
from overmind.tracing import force_flush_traces, observe_safe

logger = logging.getLogger("overmind.commands.optimize")


def _build_reporter(config, job_id: str) -> ApiReporter | None:
    """Build an :class:`ApiReporter` bound to a locally-minted ``job_id``.

    The Job row itself is created by the backend's OTLP ingest pipeline
    when it sees the workflow root span tagged with
    ``attrs.JOB_ID = job_id``. The reporter only carries the same UUID so
    its terminal :meth:`on_complete` / :meth:`on_failed` PATCHes target
    the right row. Returns ``None`` when the REST client cannot be
    constructed (no ``OVERMIND_API_KEY``) or when ``agent_id`` is
    unknown — in those cases the run continues silently with OTLP-only
    telemetry, exactly as before.

    Avoids the historical REST :func:`_create_job` path, whose POSTs the
    backend silently orphans (the resulting Job has no project FK and is
    invisible to project-scoped reads), and which the OTLP ingest cannot
    retroactively bind span-derived ``baseline_score`` / iteration rows
    to.
    """
    agent_id = getattr(config, "agent_id", None)
    if not agent_id:
        logger.info("optimize: no agent_id available — skipping Job row creation")
        return None
    try:
        reporter = ApiReporter.attach_to_job(agent_id=agent_id, job_id=job_id)
    except Exception:
        logger.debug(
            "optimize: ApiReporter.attach_to_job raised; continuing without reporter",
            exc_info=True,
        )
        return None
    if reporter is None:
        logger.info("optimize: ApiReporter.attach_to_job returned None — reporter unavailable")
        return reporter
    logger.info(
        f"optimize: OTLP-driven Job id={job_id} for agent_id={agent_id} "
        "(row will be created by backend OTLP ingest from workflow span)"
    )
    return reporter


@observe_safe(span_name="overmind.optimize", type=SpanType.WORKFLOW)
def main(
    agent_name: str,
    fast: bool = False,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
    local: bool = False,
) -> None:
    logger.info(f"optimize: start agent={agent_name} fast={fast} local={local}")

    if not local:
        from overmind.client import get_project_id, is_configured
        from overmind.workflow.runner import run_server_workflow

        if is_configured() and get_project_id():
            config: dict = {"agent_name": agent_name}
            if scope_globs:
                config["scope_globs"] = list(scope_globs)
            if max_files is not None:
                config["max_files"] = max_files
            if max_chars is not None:
                config["max_chars"] = max_chars
            if fast:
                config["max_iterations"] = 3
                config["dataset_size"] = 5
            run_server_workflow(
                agent_name,
                "optimize_full",
                config=config,
                fast=fast,
            )
            return

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

    # Mint the Job UUID locally and provision the backend ``Job`` row via
    # a short-lived OTLP workflow span. Mirrors ``overmind optimize-step
    # init``'s proven pattern: the long-running ``@observe_safe`` workflow
    # span around ``main`` stays open for the full run (so
    # ``BatchSpanProcessor`` never exports it mid-flight), but the **inner
    # closed span** below flushes within ~2 s and carries the canonical
    # job-creation attributes (``SPAN_TYPE=WORKFLOW`` + ``COMMAND=optimize``
    # + ``JOB_ID`` + config snapshot). The OTLP ingest creates the ``Job``
    # row from this span — with project context inherited from the OTel
    # resource set by :func:`overmind.init` — before the first baseline /
    # iteration span lands. Every subsequent span carrying the same
    # ``attrs.JOB_ID`` then auto-attaches: ``baseline_score``,
    # ``current_iteration``, dimension scores, and ``JobIteration`` rows
    # populate in the UI live, without any extra REST writes. The
    # :class:`ApiReporter` we build shares the same UUID so its terminal
    # :meth:`on_complete` / :meth:`on_failed` PATCHes
    # (``best_agent_code`` / ``report_markdown``) land on the right row.
    import uuid as _uuid

    from opentelemetry import trace as _otel_trace

    job_id = str(_uuid.uuid4())

    try:
        from overmind import __version__ as _sdk_version
    except Exception:
        _sdk_version = "unknown"
    _tracer = _otel_trace.get_tracer("overmind", _sdk_version)
    with _tracer.start_as_current_span("overmind.optimize.init") as _root_span:
        _root_span.set_attribute(attrs.SPAN_TYPE, SpanType.WORKFLOW.value)
        _root_span.set_attribute(attrs.COMMAND, "optimize")
        _root_span.set_attribute(attrs.JOB_ID, job_id)
        _root_span.set_attribute(attrs.AGENT_NAME, agent_name)
        _root_span.set_attribute(attrs.OPTIMIZE_AGENT_NAME, agent_name)
        _root_span.set_attribute(attrs.OPTIMIZE_AGENT_PATH, config.agent_path or "")
        _root_span.set_attribute(attrs.OPTIMIZE_ENTRYPOINT_FN, config.entrypoint_fn or "")
        _root_span.set_attribute(attrs.OPTIMIZE_ITERATIONS, int(config.iterations))
        _root_span.set_attribute(attrs.OPTIMIZE_CANDIDATES_PER_ITERATION, int(config.candidates_per_iteration))
        _root_span.set_attribute(attrs.OPTIMIZE_ANALYZER_MODEL, config.analyzer_model or "")
        _root_span.set_attribute(attrs.OPTIMIZE_LLM_JUDGE_MODEL, config.llm_judge_model or "disabled")
        _root_span.set_attribute(attrs.OPTIMIZE_PARALLEL, bool(config.parallel))
        _root_span.set_attribute(attrs.OPTIMIZE_MAX_WORKERS, int(config.max_workers))
        _root_span.set_attribute(attrs.OPTIMIZE_EARLY_STOPPING_PATIENCE, int(config.early_stopping_patience))
        _root_span.set_attribute(attrs.OPTIMIZE_EVAL_SPEC_PATH, config.eval_spec_path or "")
        _root_span.set_attribute(attrs.OPTIMIZE_DATA_PATH, config.data_path or "")
        _root_span.set_attribute(attrs.OPTIMIZE_RUN_STATUS, "running")
        _root_span.set_attribute(attrs.STATUS, "running")

    # Stamp the same JOB_ID on the outer ``@observe_safe`` workflow span
    # so every child span (baseline / iteration / candidate eval) inherits
    # it via OTel context. The OTLP ingest then routes their attributes
    # (``OPTIMIZE_BASELINE_SCORE``, ``OPTIMIZE_ITERATION_SCORE``, …) onto
    # the Job row provisioned above.
    set_tag(attrs.JOB_ID, job_id)

    # Best-effort flush so the init span reaches the backend within seconds
    # rather than waiting for the next BatchSpanProcessor schedule tick.
    try:
        force_flush_traces(timeout_millis=3_000)
    except Exception:
        logger.debug("optimize: force_flush_traces after init span raised", exc_info=True)

    reporter = _build_reporter(config, job_id)

    optimizer = Optimizer(config)
    try:
        optimizer.run()
    except KeyboardInterrupt:
        logger.warning(f"optimize: interrupted by user (KeyboardInterrupt) agent={agent_name}")
        _finalize_failed_job(optimizer, reporter, reason="Interrupted by user (KeyboardInterrupt)")
        raise
    except BaseException as exc:
        logger.exception(f"optimize: run failed for agent={agent_name}")
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        _finalize_failed_job(optimizer, reporter, reason=reason)
        raise

    _finalize_completed_job(optimizer, reporter)
    logger.info(f"optimize: run complete agent={agent_name}")


def _read_report_markdown(optimizer: Optimizer) -> str | None:
    """Return the rendered ``report.md`` content, or ``None`` if unavailable."""
    try:
        report_path = optimizer.output_dir / "report.md"
        if report_path.exists():
            return report_path.read_text(encoding="utf-8")
    except Exception:
        logger.debug("optimize: failed to read report.md", exc_info=True)
    return None


def _finalize_completed_job(optimizer: Optimizer, reporter: ApiReporter | None) -> None:
    """Push terminal Job state to the backend after a successful run."""
    if reporter is None:
        return
    try:
        baseline_score = float(getattr(optimizer, "_baseline_train_score", 0.0) or 0.0)
        best_score = float(getattr(optimizer, "best_score", baseline_score) or baseline_score)
        report_md = _read_report_markdown(optimizer)
        best_code = getattr(optimizer, "best_code", None) or None
        reporter.on_complete(
            best_score=best_score,
            baseline_score=baseline_score,
            report_markdown=report_md,
            best_agent_code=best_code,
        )
    except Exception:
        logger.debug("optimize: reporter.on_complete failed; continuing", exc_info=True)


def _finalize_failed_job(
    optimizer: Optimizer,
    reporter: ApiReporter | None,
    *,
    reason: str,
) -> None:
    """Flush in-flight OTel spans after the optimize loop aborts and
    explicitly mark the Job row as failed via the REST API.

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
    if reporter is not None:
        try:
            reporter.on_failed(reason or "")
        except Exception:
            logger.debug("optimize: reporter.on_failed failed; continuing", exc_info=True)
    try:
        force_flush_traces(timeout_millis=10_000)
    except Exception:
        logger.exception("optimize: force_flush_traces raised; continuing teardown")
