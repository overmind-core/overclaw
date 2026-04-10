"""Shared provider credential utilities used by init, setup, and optimize commands."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from rich.console import Console

from overclaw.utils.display import rel
from overclaw.utils.io import read_api_key_masked
from overclaw.utils.models import get_provider_display_name

# Maps LiteLLM provider prefix → env vars required to authenticate with that provider.
PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "bedrock": ["AWS_BEARER_TOKEN_BEDROCK"],
}


def update_agent_env(path: Path, agent_name: str, updates: dict[str, str]) -> None:
    """Merge *updates* into the agent's ``.env``, preserving all existing keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.exists():
        existing = {k: (v or "") for k, v in (dotenv_values(path) or {}).items()}
    existing.update(updates)
    lines = [f"# OverClaw agent env — {agent_name}", ""]
    for key, val in existing.items():
        lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_provider_api_keys(
    model: str, env_path: Path, agent_name: str, console: Console
) -> None:
    """Prompt for any provider credentials missing from both the global environment
    and the agent's ``.env``, then persist and reload them.

    Called after interactive model selection so the user is never left with a
    chosen provider whose API key hasn't been configured yet.
    """
    from overclaw.core.paths import load_agent_dotenv

    provider = model.split("/")[0] if "/" in model else ""
    key_names = PROVIDER_ENV_KEYS.get(provider, [])
    if not key_names:
        return

    existing_agent: dict[str, str] = (
        {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()}
        if env_path.exists()
        else {}
    )

    missing = [
        k
        for k in key_names
        if not os.getenv(k, "").strip() and not existing_agent.get(k, "").strip()
    ]
    if not missing:
        return

    provider_label = get_provider_display_name(provider)
    console.print(
        f"\n  [yellow]Missing credentials for {provider_label}.[/yellow] "
        f"[dim]Enter them below — they will be saved to "
        f"[cyan]{rel(env_path)}[/cyan].[/dim]"
    )

    updates: dict[str, str] = {}
    for key_name in missing:
        console.print(f"  [dim]Required: [bold]{key_name}[/bold][/dim]")
        val = read_api_key_masked(key_name)
        if val.strip():
            updates[key_name] = val.strip()

    if updates:
        update_agent_env(env_path, agent_name, updates)
        for key_name in updates:
            console.print(
                f"  [bold green]✓[/bold green] Saved [bold]{key_name}[/bold]"
                f"  [dim]→ {rel(env_path)}[/dim]"
            )
        load_agent_dotenv(agent_name)
