"""Shared provider credential utilities used by init, setup, and optimize commands.

All credentials live in **one** place — ``<project>/.overmind/.env`` — to avoid
the ``override=True`` footgun that earlier per-agent ``.env`` files introduced
(a placeholder in ``<state>/agents/<name>/.env`` would silently win over the
real value in the project ``.env``).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from rich.console import Console

from overmind.core.paths import overmind_env_path
from overmind.utils.display import rel
from overmind.utils.io import read_api_key_masked
from overmind.utils.models import get_provider_display_name

# Maps LiteLLM provider prefix → env vars required to authenticate with that provider.
PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "bedrock": ["AWS_BEARER_TOKEN_BEDROCK"],
}


def update_overmind_env(updates: dict[str, str], *, path: Path | None = None) -> Path:
    """Merge *updates* into the project ``.overmind/.env`` and write it atomically.

    Returns the path that was written so callers can surface it in the UI.
    Existing keys not present in *updates* are preserved.  Pass *path*
    explicitly only in tests that need to write to an isolated location.
    """
    env_path = path or overmind_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if env_path.exists():
        existing = {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()}
    existing.update(updates)
    lines = ["# Overmind — managed by `overmind init` / `overmind setup`", ""]
    for key, val in existing.items():
        lines.append(f"{key}={val}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def ensure_provider_api_keys(model: str, console: Console) -> None:
    """Prompt for any provider credentials missing from both the process
    environment and the project ``.overmind/.env``, then persist them.

    Called after interactive model selection so the user is never left with a
    chosen provider whose API key hasn't been configured yet.
    """
    provider = model.split("/")[0] if "/" in model else ""
    key_names = PROVIDER_ENV_KEYS.get(provider, [])
    if not key_names:
        return

    env_path = overmind_env_path()
    existing: dict[str, str] = (
        {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()} if env_path.exists() else {}
    )

    missing = [k for k in key_names if not os.getenv(k, "").strip() and not existing.get(k, "").strip()]
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
        update_overmind_env(updates)
        for key_name in updates:
            console.print(f"  [bold green]✓[/bold green] Saved [bold]{key_name}[/bold]  [dim]→ {rel(env_path)}[/dim]")
        load_dotenv(env_path, override=False)
