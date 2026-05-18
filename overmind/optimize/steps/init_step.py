"""``overmind optimize-step init`` — collect skill-driven config & seed state.

Unlike :func:`overmind.optimize.config.collect_config`, this step takes
all settings as JSON on stdin (or a JSON file via ``--config-json``)
because the host coding agent is the one prompting the user — Rich
prompts wouldn't render usefully here. The skill renders questions via
``AskQuestion``, then hands the resulting dict to this step.

The step:
  1. Validates the agent name and resolves its file path.
  2. Loads ``setup_spec/eval_spec.json`` and ``setup_spec/dataset.json``.
  3. Applies ``apply_eval_spec_scope`` so scope fields default from the spec.
  4. Builds a :class:`Config` from the merged dict + defaults.
  5. Writes ``experiments/skill_state.json`` (the cross-step state file).
  6. Emits a JSON envelope describing what was set up.

Re-running ``init`` overwrites the state file (the skill should ask
the user before doing so).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.core.paths import (
    agent_experiments_dir,
    agent_setup_spec_dir,
)
from overmind.core.registry import get_agent_id, resolve_agent
from overmind.optimize.config import Config, apply_eval_spec_scope
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import force_flush_traces, get_tracer, observe_safe

logger = logging.getLogger("overmind.optimize.steps.init")


SKILL_STATE_FILENAME = "skill_state.json"


def _detect_language(agent_path: str) -> str:
    ext = Path(agent_path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".mts": "typescript",
    }.get(ext, "python")


def _coerce_into_config_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys and coerce value types to what ``Config`` expects."""
    known = set(Config.__dataclass_fields__)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in known:
            continue
        out[k] = v
    return out


def _resolve_agent_id(
    agent_name: str,
    agent_path: str,
    fn_name: str,
) -> str | None:
    """Resolve the backend agent UUID, self-healing the local registry.

    Path B's terminal ``ApiReporter.on_complete`` / ``on_failed``
    PATCHes are addressed by ``agent_id`` + ``job_id``. The local
    ``.overmind/agents.toml`` registry only has an ``id`` field when
    something previously wrote it back from the backend; for
    historical agents registered before that flow existed the column
    is empty, which forced ``optimize-step`` to no-op the terminal
    PATCH and leave the Job stuck in ``running`` forever.

    Resolution order:

    1. ``.overmind/agents.toml`` (already-persisted backend UUID).
    2. The currently-configured storage backend's ``get_agent_id()``
       (already-fetched-this-process UUID).
    3. A best-effort ``storage.save_spec(spec)`` round-trip to ask
       the backend (the same upsert the skill's sync block does).
       On success we persist the returned UUID back to
       ``agents.toml`` via :func:`save_agent` so subsequent invocations
       resolve at step 1 with zero network cost.

    Returns ``None`` only when storage cannot be configured (e.g. no
    ``OVERMIND_API_KEY``) — in that case the run continues with
    OTLP-only telemetry, identical to Path A's behaviour.
    """
    cached = get_agent_id(agent_name)
    if cached:
        return cached

    try:
        from overmind.storage import (
            StorageNotConfiguredError,
            configure_storage,
            get_storage,
        )

        configure_storage(agent_path=agent_path, agent_name=agent_name)
        try:
            storage = get_storage()
        except StorageNotConfiguredError:
            logger.debug("init: storage not configured; agent_id will be None")
            return None
        resolved = storage.get_agent_id()
        if not resolved:
            spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
            if spec_path.is_file():
                try:
                    spec = json.loads(spec_path.read_text())
                    storage.save_spec(spec)
                    resolved = storage.get_agent_id()
                except Exception:
                    logger.debug(
                        "init: save_spec round-trip failed; agent_id stays None",
                        exc_info=True,
                    )
        if resolved:
            try:
                from overmind.core.registry import (
                    _read_registry_entries,
                    save_agent,
                )

                # Reuse the entrypoint string already on disk so we
                # only patch the missing ``id`` column.  ``save_agent``
                # rewrites the whole row, so without this lookup we'd
                # silently rewrite a working dotted-module entrypoint
                # (``scripts.overmind_entrypoint:run``) into a bare
                # file-stem form (``overmind_entrypoint:run``) that
                # subsequent ``resolve_agent`` calls might not find.
                existing_entry = next(
                    (e for e in _read_registry_entries() if e.get("name") == agent_name),
                    None,
                )
                entrypoint_spec = (existing_entry or {}).get("entrypoint")
                if not entrypoint_spec:
                    entrypoint_spec = (
                        f"{Path(agent_path).stem}:{fn_name}" if Path(agent_path).suffix == ".py" else agent_path
                    )
                save_agent(agent_name, entrypoint_spec, id=resolved)
                logger.info(f"init: persisted agent_id={resolved} into agents.toml for agent={agent_name}")
            except Exception:
                logger.debug(
                    "init: save_agent persistence failed; agent_id resolved in-process only",
                    exc_info=True,
                )
        return resolved or None
    except Exception:
        logger.debug(
            "init: agent_id resolution raised; agent_id will be None",
            exc_info=True,
        )
        return None


@observe_safe(span_name="overmind.optimize.init", type=SpanType.FUNCTION)
def run_init(
    *,
    agent_name: str,
    user_settings: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the SkillRunState for *agent_name* from *user_settings*.

    Args:
        agent_name: Registered agent slug.
        user_settings: Dict of ``Config`` field overrides collected by the
            skill via ``AskQuestion``. Unknown keys are ignored.
        overwrite: If False and a state file already exists, refuse with
            an error envelope. If True, overwrite (caller has confirmed).

    Returns:
        A JSON-serializable result envelope. On success::

            {"status": "ok", "state_path": "...", "summary": {...}}
    """
    try:
        agent_path, fn_name = resolve_agent(agent_name)
    except SystemExit:
        return {
            "status": "error",
            "error": "agent_not_registered",
            "message": (
                f"Agent {agent_name!r} is not registered in .overmind/agents.toml. "
                "Run the /register-agent skill (or `overmind agent register`) first."
            ),
        }

    spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
    data_path = agent_setup_spec_dir(agent_name) / "dataset.json"

    if not spec_path.is_file():
        return {
            "status": "error",
            "error": "missing_eval_spec",
            "message": (
                f"No evaluation spec found at {spec_path}. "
                f"Run `overmind setup {agent_name}` (or invoke the "
                "/generate-policy-and-eval skill) first."
            ),
        }
    if not data_path.is_file():
        return {
            "status": "error",
            "error": "missing_dataset",
            "message": (
                f"No dataset found at {data_path}. "
                f"Run `overmind setup {agent_name}` (or invoke the "
                "/generate-dataset skill) first."
            ),
        }

    experiments_dir = agent_experiments_dir(agent_name)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    state_file = experiments_dir / SKILL_STATE_FILENAME

    if state_file.exists() and not overwrite:
        return {
            "status": "error",
            "error": "state_already_exists",
            "message": (
                f"Skill state already exists at {state_file}. Re-run with "
                "--overwrite (or have the skill confirm with the user) to "
                "start a fresh optimization run."
            ),
            "state_path": str(state_file),
        }

    cfg_kwargs = _coerce_into_config_kwargs(user_settings)
    cfg_kwargs.setdefault("agent_name", agent_name)
    cfg_kwargs.setdefault("agent_path", agent_path)
    cfg_kwargs.setdefault("entrypoint_fn", fn_name)
    cfg_kwargs.setdefault("agent_id", _resolve_agent_id(agent_name, agent_path, fn_name))
    cfg_kwargs.setdefault("language", _detect_language(agent_path))
    cfg_kwargs.setdefault("eval_spec_path", str(spec_path))
    cfg_kwargs.setdefault("data_path", str(data_path))

    cfg = Config(**cfg_kwargs)

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    apply_eval_spec_scope(cfg, spec)

    state = SkillRunState.from_config(
        agent_name=agent_name,
        config=cfg,
        state_path=state_file,
    )
    state.output_dir = str(experiments_dir)
    state.job_id = str(uuid.uuid4())

    # Emit the optimize workflow root span and persist its W3C
    # traceparent so every subsequent ``overmind optimize-step`` CLI
    # invocation attaches its spans as children of this root — single
    # trace_id, single Job in the UI.  Stamps the full Config snapshot
    # (iterations, candidates-per-iteration, models, dataset, eval spec
    # path, …) so the OTLP ingest can populate the Job header on the
    # very first span it sees, even if the user Ctrl-C's before the
    # baseline finishes.
    tracer = get_tracer()
    with tracer.start_as_current_span("overmind.optimize") as root_span:
        root_span.set_attribute(attrs.SPAN_TYPE, SpanType.WORKFLOW.value)
        root_span.set_attribute(attrs.COMMAND, "optimize")
        root_span.set_attribute(attrs.JOB_ID, state.job_id)
        root_span.set_attribute(attrs.AGENT_NAME, agent_name)
        root_span.set_attribute(attrs.OPTIMIZE_AGENT_NAME, agent_name)
        root_span.set_attribute(attrs.OPTIMIZE_AGENT_PATH, cfg.agent_path or "")
        root_span.set_attribute(attrs.OPTIMIZE_ENTRYPOINT_FN, cfg.entrypoint_fn or "")
        root_span.set_attribute(attrs.OPTIMIZE_ITERATIONS, int(cfg.iterations))
        root_span.set_attribute(attrs.OPTIMIZE_CANDIDATES_PER_ITERATION, int(cfg.candidates_per_iteration))
        root_span.set_attribute(attrs.OPTIMIZE_ANALYZER_MODEL, cfg.analyzer_model or "")
        root_span.set_attribute(attrs.OPTIMIZE_LLM_JUDGE_MODEL, cfg.llm_judge_model or "disabled")
        root_span.set_attribute(attrs.OPTIMIZE_PARALLEL, bool(cfg.parallel))
        root_span.set_attribute(attrs.OPTIMIZE_MAX_WORKERS, int(cfg.max_workers))
        root_span.set_attribute(attrs.OPTIMIZE_EARLY_STOPPING_PATIENCE, int(cfg.early_stopping_patience))
        root_span.set_attribute(attrs.OPTIMIZE_EVAL_SPEC_PATH, cfg.eval_spec_path or "")
        root_span.set_attribute(attrs.OPTIMIZE_DATA_PATH, cfg.data_path or "")
        root_span.set_attribute(attrs.OPTIMIZE_RUN_STATUS, "running")
        root_span.set_attribute(attrs.STATUS, "running")

        # Capture the trace context BEFORE the span ends so subsequent
        # CLI invocations can stitch onto it.  W3C format:
        # ``00-<trace_id_hex32>-<span_id_hex16>-<flags>``.
        ctx = root_span.get_span_context()
        flags = "01" if ctx.trace_flags else "00"
        state.traceparent = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{flags}"

    # Also stamp the same job id on the active init span (parent of the
    # workflow root via @observe_safe) so OTLP-side coalescing works as
    # a fallback when TRACEPARENT propagation is unavailable (e.g. the
    # user invokes steps from different shells).
    set_tag(attrs.JOB_ID, state.job_id)
    set_tag(attrs.OPTIMIZE_STEP, "init")

    state.save()

    # Make sure the root span (and our init span) ship to the backend
    # before this CLI process exits — without this the Job row stays
    # invisible until the next step lands.
    force_flush_traces(timeout_millis=3000)

    logger.info(
        f"optimize-step init: agent={agent_name} state={state_file} "
        f"job_id={state.job_id} "
        f"iterations={cfg.iterations} candidates={cfg.candidates_per_iteration}"
    )

    return {
        "status": "ok",
        "state_path": str(state_file),
        "job_id": state.job_id,
        "traceparent": state.traceparent,
        "agent_path": cfg.agent_path,
        "entrypoint_fn": cfg.entrypoint_fn,
        "eval_spec_path": cfg.eval_spec_path,
        "data_path": cfg.data_path,
        "summary": {
            "agent_name": agent_name,
            "iterations": cfg.iterations,
            "candidates_per_iteration": cfg.candidates_per_iteration,
            "parallel": cfg.parallel,
            "max_workers": cfg.max_workers,
            "runs_per_eval": cfg.runs_per_eval,
            "regression_threshold": cfg.regression_threshold,
            "holdout_ratio": cfg.holdout_ratio,
            "holdout_enforcement": cfg.holdout_enforcement,
            "early_stopping_patience": cfg.early_stopping_patience,
            "diagnosis_case_fraction": cfg.diagnosis_case_fraction,
            "smoke_test_cases": cfg.smoke_test_cases,
            "cross_run_persistence": cfg.cross_run_persistence,
            "failure_clustering": cfg.failure_clustering,
            "adaptive_focus": cfg.adaptive_focus,
            "model_backtesting": cfg.model_backtesting,
            "backtest_models": list(cfg.backtest_models),
            "analyzer_model": cfg.analyzer_model,
            "codegen_model": cfg.codegen_model or cfg.analyzer_model,
            "codegen_max_steps": cfg.codegen_max_steps,
            "llm_judge_model": cfg.llm_judge_model,
            "optimizable_scope": list(cfg.optimizable_scope),
            "read_only_scope": list(cfg.read_only_scope),
            "max_resolved_files": cfg.max_resolved_files,
            "max_total_chars": cfg.max_total_chars,
        },
        "config": asdict(cfg),
    }
