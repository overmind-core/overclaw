"""
OverClaw optimize — Agent Optimizer

Usage:
    overclaw optimize <agent-name>
    overclaw optimize <agent-name> --fast
"""

from rich.console import Console

from overclaw.client import get_client, get_project_id
from overclaw.commands.setup_cmd import _sync_setup_artifacts
from overclaw.core.paths import load_agent_dotenv
from overclaw.core.registry import get_agent_id
from overclaw.optimize.config import collect_config
from overclaw.optimize.optimizer import Optimizer
from overclaw.storage import configure_storage


def main(agent_name: str, fast: bool = False) -> None:
    # Load agent-specific .env before anything else so the agent's credentials
    # are available throughout the entire optimize run (config collection,
    # agent execution, and evaluation).
    load_agent_dotenv(agent_name)

    config = collect_config(agent_name=agent_name, fast=fast)

    # If API is configured, make sure setup artifacts are synced first and
    # refresh agent_id from registry (sync may create/update it).
    if get_client() and get_project_id():
        _sync_setup_artifacts(agent_name, config.agent_path, Console())
        config.agent_id = get_agent_id(agent_name)

    use_api_backend = bool(config.agent_id and get_client() and get_project_id())
    configure_storage(
        agent_path=config.agent_path,
        agent_id=config.agent_id,
        backend="api" if use_api_backend else "fs",
    )
    optimizer = Optimizer(config)
    optimizer.run()
