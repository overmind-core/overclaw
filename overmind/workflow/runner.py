"""Run a server-orchestrated workflow from the CLI."""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any

import overmind
from overmind.client import get_client, get_project_id, is_configured
from overmind.core.registry import load_registry
from overmind.workflow.daemon_proc import DaemonProcess

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _require_client():
    if not is_configured():
        raise SystemExit(
            "Server workflow requires OVERMIND_API_URL, OVERMIND_API_KEY, and OVERMIND_PROJECT_ID.\n"
            "Run `overmind init` or pass --local to use the legacy on-machine pipeline."
        )
    client = get_client()
    if client is None:
        raise SystemExit("Could not create Overmind API client.")
    project_id = get_project_id()
    if not project_id:
        raise SystemExit("OVERMIND_PROJECT_ID is required for server workflows.")
    return client, project_id


def _entrypoint_fn(agent_name: str) -> str:
    registry = load_registry()
    entry = registry.get(agent_name, {})
    return entry.get("fn_name") or "run"


def run_server_workflow(
    agent_name: str,
    workflow_name: str,
    *,
    config: dict[str, Any] | None = None,
    fast: bool = False,
    manage_daemon: bool = True,
    console=None,
) -> Any:
    """Start a workflow on the server, run a daemon, poll until done."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner

    console = console or Console()
    client, project_id = _require_client()

    agent = client.resolve_agent(agent_name)
    agent_id = str(agent.id) if agent else None

    wf_config = dict(config or {})
    wf_config.setdefault("agent_name", agent_name)
    wf_config.setdefault("entrypoint_fn", _entrypoint_fn(agent_name))
    wf_config.setdefault("require_criteria_approval", not fast)
    if fast:
        wf_config.setdefault("dataset_size", 5)
        wf_config.setdefault("max_iterations", 3)

    session = client.create_cli_session(
        project_id,
        hostname=socket.gethostname(),
        cli_version=overmind.__version__,
        agent_name=agent_name,
    )
    session_id = str(session.id)

    daemon: DaemonProcess | None = None
    if manage_daemon:
        daemon = DaemonProcess()
        try:
            daemon.start(session_id=session_id, agent_name=agent_name)
        except RuntimeError as exc:
            console.print(f"[red]Failed to start daemon:[/red] {exc}")
            raise SystemExit(1) from exc

    console.print(
        Panel(
            f"[bold]Workflow[/bold] [cyan]{workflow_name}[/cyan]\n"
            f"[dim]Agent: {agent_name} · Session: {session_id[:8]}…[/dim]",
            border_style="cyan",
        )
    )

    try:
        run = client.start_workflow_run(
            project_id,
            workflow_name=workflow_name,
            agent_id=agent_id,
            client_session_id=session_id,
            config=wf_config,
        )
        run_id = str(run.id)
        logger.info("Started workflow %s run %s", workflow_name, run_id)

        with Live(console=console, refresh_per_second=4) as live:
            while True:
                run = client.workflow_runs_retrieve(id=run_id)
                status = getattr(run.status, "value", str(run.status))
                block = run.current_block or "—"
                live.update(Spinner("dots", text=f"{status} · block={block}"))

                if status in TERMINAL_STATUSES:
                    break

                if status == "waiting_user":
                    live.stop()
                    _handle_user_approval(client, run_id, run, console, fast=fast)
                    live.start()
                    continue

                time.sleep(2)

        run = client.workflow_runs_retrieve(id=run_id)
        final_status = getattr(run.status, "value", str(run.status))
        if final_status == "completed":
            console.print(f"\n[green]Workflow completed.[/green] Run ID: {run_id}")
        elif final_status == "failed":
            console.print(f"\n[red]Workflow failed:[/red] {run.error or 'unknown error'}")
            raise SystemExit(1)
        else:
            console.print(f"\n[yellow]Workflow ended with status {final_status}[/yellow]")
        return run
    finally:
        if daemon is not None:
            daemon.stop()


def _handle_user_approval(client, run_id: str, run: Any, console, *, fast: bool) -> None:
    from rich.prompt import Confirm

    ctx = run.context if isinstance(run.context, dict) else {}
    user_prompt = ctx.get("user_prompt") or {}
    prompt_type = user_prompt.get("type", "approval")

    if fast:
        client.submit_workflow_user_response(run_id, approved=True)
        return

    if prompt_type == "criteria_approval":
        eval_spec = user_prompt.get("eval_spec", {})
        fields = eval_spec.get("output_fields") or eval_spec.get("output_fields", {})
        console.print("\n[bold]Proposed evaluation criteria[/bold]")
        if isinstance(fields, dict):
            for name, spec in fields.items():
                weight = spec.get("weight", "?") if isinstance(spec, dict) else "?"
                console.print(f"  • [cyan]{name}[/cyan] (weight {weight})")
        else:
            console.print(json.dumps(eval_spec, indent=2)[:2000])

    approved = Confirm.ask("Approve and continue?", default=True)
    client.submit_workflow_user_response(
        run_id,
        approved=approved,
        feedback={"approved": approved},
    )
    if not approved:
        console.print("[yellow]Workflow cancelled by user.[/yellow]")
        raise SystemExit(1)
