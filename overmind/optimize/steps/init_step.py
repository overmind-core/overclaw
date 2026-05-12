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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from overmind.core.paths import (
    agent_experiments_dir,
    agent_setup_spec_dir,
    load_agent_dotenv,
)
from overmind.core.registry import get_agent_id, resolve_agent
from overmind.optimize.config import Config, apply_eval_spec_scope
from overmind.optimize.steps.state import SkillRunState
from overmind.preflight import load_report, preflight_report_path
from overmind.preflight.state import GREEN_STATUSES

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
    load_agent_dotenv(agent_name)

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
                "/overmind-generate-spec-and-dataset skill) first."
            ),
        }
    if not data_path.is_file():
        return {
            "status": "error",
            "error": "missing_dataset",
            "message": (
                f"No dataset found at {data_path}. "
                f"Run `overmind setup {agent_name}` (or invoke the "
                "/overmind-generate-spec-and-dataset skill) first."
            ),
        }

    # Preflight is recommended but optional — surface its status in
    # the response envelope so the caller can warn the user, but never
    # refuse to initialize optimization based on it.
    preflight_summary: dict[str, Any] | None = None
    report = load_report(agent_name)
    if report is not None:
        preflight_summary = {
            "status": report.status,
            "is_green": report.status in GREEN_STATUSES,
            "message": report.message,
            "missing_secrets": list(report.missing_secrets),
            "report_path": str(preflight_report_path(agent_name)),
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
    cfg_kwargs.setdefault("agent_id", get_agent_id(agent_name))
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
    state.save()

    logger.info(
        "optimize-step init: agent=%s state=%s iterations=%d candidates=%d",
        agent_name,
        state_file,
        cfg.iterations,
        cfg.candidates_per_iteration,
    )

    return {
        "status": "ok",
        "state_path": str(state_file),
        "agent_path": cfg.agent_path,
        "entrypoint_fn": cfg.entrypoint_fn,
        "eval_spec_path": cfg.eval_spec_path,
        "data_path": cfg.data_path,
        "preflight": preflight_summary,
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
            "context_scope": list(cfg.context_scope),
            "exclude_scope": list(cfg.exclude_scope),
            "max_resolved_files": cfg.max_resolved_files,
            "max_total_chars": cfg.max_total_chars,
        },
        "config": asdict(cfg),
    }
