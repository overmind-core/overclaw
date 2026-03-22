"""overclaw sync — upload local setup artifacts and traces to Overmind."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from overclaw.client import _create_trace, flush_pending_api_updates, get_client, get_project_id
from overclaw.commands.setup_cmd import _sync_setup_artifacts
from overclaw.core.paths import agent_experiments_dir, load_overclaw_dotenv
from overclaw.core.registry import get_agent_id, load_registry


def _sync_traces_for_agent(agent_name: str, agent_id: str, console: Console) -> int:
    """Upload locally-stored traces for an agent to the API. Returns count uploaded."""
    client = get_client()
    if not client or not agent_id:
        return 0

    traces_root = agent_experiments_dir(agent_name) / "traces"
    if not traces_root.exists():
        return 0

    synced = 0
    for trace_file in sorted(traces_root.glob("*/*.json")):
        run_name = trace_file.parent.name
        try:
            trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        trace_data["trace_group"] = trace_data.get("trace_group") or run_name
        _create_trace(client, agent_id, None, trace_data)
        synced += 1

    if synced:
        flush_pending_api_updates(timeout=20.0)

    return synced


def main(agent_name: str | None = None) -> None:
    """Sync local setup artifacts and traces to Overmind for one or all agents."""
    load_overclaw_dotenv()
    console = Console()

    if not get_client() or not get_project_id():
        console.print(
            "[red]Overmind API is not configured.[/red] "
            "Set OVERMIND_API_URL, OVERMIND_API_TOKEN, and OVERMIND_PROJECT_ID in .overclaw/.env."
        )
        raise SystemExit(1)

    registry = load_registry()
    if not registry:
        console.print("[yellow]No registered agents found.[/yellow]")
        return

    names = [agent_name] if agent_name else sorted(registry.keys())
    synced_agents = 0
    skipped_agents = 0

    for name in names:
        row = registry.get(name)
        if not row:
            console.print(f"[yellow]Skipping {name}: not registered.[/yellow]")
            skipped_agents += 1
            continue

        agent_path = row.get("file_path", "")
        if not agent_path or not Path(agent_path).exists():
            console.print(
                f"[yellow]Skipping {name}: registered file missing ({agent_path or 'unknown'}).[/yellow]"
            )
            skipped_agents += 1
            continue

        _sync_setup_artifacts(name, agent_path, console)

        agent_id = get_agent_id(name)
        if agent_id:
            traces_synced = _sync_traces_for_agent(name, agent_id, console)
            if traces_synced:
                console.print(f"  [dim]Synced {traces_synced} trace(s) to Overmind.[/dim]")

        synced_agents += 1

    console.print(
        f"\n[bold green]Sync complete.[/bold green] "
        f"Processed: {synced_agents} agent(s), skipped: {skipped_agents}."
    )
