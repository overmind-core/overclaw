"""Filesystem layout for OverClaw state under the project root.

The project root is the directory that contains the OverClaw state directory
(see :func:`~overclaw.core.registry.project_root` and
:data:`~overclaw.core.constants.OVERCLAW_DIR_NAME`). Agent code stays where you
put it (e.g. ``agents/...``). The registry of agent names and entrypoints is
``<state>/agents.toml``. Per-agent data lives under ``<state>/agents/<name>/``.
Environment variables are stored in ``<state>/.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.registry import project_root


def _safe_agent_segment(agent_name: str) -> str:
    if not agent_name or agent_name in (".", ".."):
        raise ValueError("agent name must be non-empty and not '.' or '..'")
    if os.sep in agent_name or (os.altsep and os.altsep in agent_name):
        raise ValueError(f"agent name must not contain path separators: {agent_name!r}")
    return agent_name


def overclaw_dir() -> Path:
    """OverClaw state directory at the project root."""
    return project_root() / OVERCLAW_DIR_NAME


def overclaw_env_path() -> Path:
    """API keys and model defaults (``.env`` inside the state directory)."""
    return overclaw_dir() / ".env"


def agents_registry_path() -> Path:
    """Registered agent names and entrypoints (``agents.toml``)."""
    return overclaw_dir() / "agents.toml"


def agent_overclaw_dir(agent_name: str) -> Path:
    """Per-agent state: ``<state>/agents/<name>/``."""
    return overclaw_dir() / "agents" / _safe_agent_segment(agent_name)


def agent_setup_spec_dir(agent_name: str) -> Path:
    return agent_overclaw_dir(agent_name) / "setup_spec"


def agent_experiments_dir(agent_name: str) -> Path:
    return agent_overclaw_dir(agent_name) / "experiments"


def agent_env_path(agent_name: str) -> Path:
    """Per-agent ``.env`` at ``<state>/agents/<name>/.env``."""
    return agent_overclaw_dir(agent_name) / ".env"


def load_overclaw_dotenv() -> None:
    """Load state-directory ``.env`` into the process environment (no-op if missing)."""
    path = overclaw_env_path()
    if path.is_file():
        load_dotenv(path)


def load_agent_dotenv(agent_name: str) -> None:
    """Load per-agent ``.env`` into the process environment, overriding any existing values.

    No-op if the file does not exist.  Agent-specific vars take precedence over
    the global ``.overclaw/.env`` so credentials saved during ``overclaw setup``
    are always used when the agent is run or optimised.
    """
    path = agent_env_path(agent_name)
    if path.is_file():
        load_dotenv(path, override=True)
