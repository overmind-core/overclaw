"""
OverClaw optimize — Agent Optimizer

Usage:
    overclaw optimize <agent-name>
    overclaw optimize <agent-name> --fast
"""

from overclaw.client import get_client, get_project_id
from overclaw.core.paths import load_agent_dotenv
from overclaw.core.registry import get_agent_id
from overclaw.optimize.config import collect_config
from overclaw.optimize.optimizer import Optimizer
from overclaw.storage import configure_storage
from overmind_sdk import observe, SpanType, set_tag


@observe(span_name="overclaw_optimize", type=SpanType.WORKFLOW)
def main(agent_name: str, fast: bool = False) -> None:
    # Load agent-specific .env before anything else so the agent's credentials
    # are available throughout the entire optimize run (config collection,
    # agent execution, and evaluation).
    load_agent_dotenv(agent_name)

    # CLI-level flags
    set_tag("overclaw.command", "optimize")
    set_tag("overclaw.optimize.agent_name", agent_name)
    set_tag("overclaw.optimize.fast", str(fast))

    config = collect_config(agent_name=agent_name, fast=fast)

    # Refresh agent_id from registry in case setup just created/updated it
    config.agent_id = get_agent_id(agent_name)

    use_api_backend = bool(config.agent_id and get_client() and get_project_id())
    configure_storage(
        agent_path=config.agent_path,
        agent_id=config.agent_id,
        backend="api" if use_api_backend else "fs",
    )

    # Config-level tags — everything the user chose or defaulted to
    set_tag("overclaw.optimize.agent_path", config.agent_path)
    set_tag("overclaw.optimize.entrypoint_fn", config.entrypoint_fn)
    set_tag("overclaw.optimize.storage_backend", "api" if use_api_backend else "fs")
    set_tag("overclaw.optimize.analyzer_model", config.analyzer_model or "")
    set_tag("overclaw.optimize.llm_judge_model", config.llm_judge_model or "disabled")
    set_tag("overclaw.optimize.iterations", str(config.iterations))
    set_tag(
        "overclaw.optimize.candidates_per_iteration",
        str(config.candidates_per_iteration),
    )
    set_tag("overclaw.optimize.parallel", str(config.parallel))
    set_tag("overclaw.optimize.max_workers", str(config.max_workers))
    set_tag("overclaw.optimize.runs_per_eval", str(config.runs_per_eval))
    set_tag("overclaw.optimize.regression_threshold", str(config.regression_threshold))
    set_tag("overclaw.optimize.holdout_ratio", str(config.holdout_ratio))
    set_tag("overclaw.optimize.holdout_enforcement", str(config.holdout_enforcement))
    set_tag(
        "overclaw.optimize.early_stopping_patience", str(config.early_stopping_patience)
    )
    set_tag(
        "overclaw.optimize.cross_run_persistence", str(config.cross_run_persistence)
    )
    set_tag("overclaw.optimize.failure_clustering", str(config.failure_clustering))
    set_tag("overclaw.optimize.adaptive_focus", str(config.adaptive_focus))
    set_tag("overclaw.optimize.model_backtesting", str(config.model_backtesting))
    if config.backtest_models:
        set_tag("overclaw.optimize.backtest_models", ",".join(config.backtest_models))
    set_tag("overclaw.optimize.eval_spec_path", config.eval_spec_path or "")
    set_tag("overclaw.optimize.data_path", config.data_path or "")

    optimizer = Optimizer(config)
    optimizer.run()
