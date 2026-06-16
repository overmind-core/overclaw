"""Run a server-orchestrated optimize run from the CLI."""

from __future__ import annotations

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
            "Optimize requires OVERMIND_API_URL, OVERMIND_API_KEY, and OVERMIND_PROJECT_ID.\n"
            "Run `overmind init` first."
        )
    client = get_client()
    if client is None:
        raise SystemExit("Could not create Overmind API client.")
    project_id = get_project_id()
    if not project_id:
        raise SystemExit("OVERMIND_PROJECT_ID is required.")
    return client, project_id


def _entrypoint_fn(agent_name: str) -> str:
    registry = load_registry()
    entry = registry.get(agent_name, {})
    return entry.get("fn_name") or "run"


def run_optimize(
    agent_name: str,
    *,
    config: dict[str, Any] | None = None,
    fast: bool = False,
    manage_daemon: bool = True,
    console=None,
) -> Any:
    """Start an optimize run on the server, run a daemon, poll until done."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner

    console = console or Console()
    client, _project_id = _require_client()

    agent = client.resolve_agent(agent_name)
    if agent is None:
        raise SystemExit(f"Agent {agent_name!r} not found on the platform.")

    wf_config = dict(config or {})
    wf_config.setdefault("agent_name", agent_name)
    wf_config.setdefault("entrypoint_fn", _entrypoint_fn(agent_name))
    wf_config.setdefault("require_criteria_approval", not fast)
    if fast:
        wf_config.setdefault("dataset_size", 5)
        wf_config.setdefault("max_iterations", 3)

    session = client.create_cli_session(
        _project_id,
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
            f"[bold]Optimize[/bold] [cyan]{agent_name}[/cyan]\n"
            f"[dim]Session: {session_id[:8]}…[/dim]",
            border_style="cyan",
        )
    )

    try:
        run = client.start_optimize_run(
            str(agent.id),
            client_session_id=session_id,
            config=wf_config,
        )
        run_id = str(run.get("id", ""))
        logger.info("Started optimize run %s", run_id)

        with Live(console=console, refresh_per_second=4) as live:
            while True:
                run = client.optimize_runs_retrieve(run_id)
                status = str(run.get("status", ""))
                phase = (run.get("result") or {}).get("phase", "—")
                live.update(Spinner("dots", text=f"{status} · phase={phase}"))

                if status in TERMINAL_STATUSES:
                    break

                if status == "waiting_user":
                    live.stop()
                    _handle_user_approval(client, run_id, run, console, fast=fast)
                    live.start()
                    continue

                time.sleep(2)

        final_status = str(run.get("status", ""))
        if final_status == "completed":
            console.print(f"\n[green]Optimize completed.[/green] Run ID: {run_id}")
        elif final_status == "failed":
            console.print(f"\n[red]Optimize failed:[/red] {run.get('error') or 'unknown error'}")
            raise SystemExit(1)
        else:
            console.print(f"\n[yellow]Optimize ended with status {final_status}[/yellow]")
        return run
    finally:
        if daemon is not None:
            daemon.stop()


def _handle_user_approval(client, run_id: str, run: dict, console, *, fast: bool) -> None:
    from rich.prompt import Confirm

    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    user_prompt = result.get("user_prompt") or {}
    prompt_type = user_prompt.get("type", "approval")

    if fast:
        client.submit_optimize_user_response(run_id, approved=True)
        return

    if prompt_type == "dataset_incompatible":
        console.print("\n[bold red]Dataset incompatible with the entrypoint[/bold red]")
        console.print(user_prompt.get("message", ""))
        proceed = Confirm.ask("Proceed with this dataset anyway?", default=False)
        client.submit_optimize_user_response(run_id, approved=proceed, feedback={"approved": proceed})
        if not proceed:
            raise SystemExit(1)
        return

    if prompt_type == "criteria_approval":
        eval_spec = user_prompt.get("eval_spec", {})
        fields = eval_spec.get("output_fields") or []
        console.print("\n[bold]Proposed eval criteria[/bold]")
        for field in fields:
            console.print(f"  • {field.get('name')} (weight {field.get('weight')})")

    approved = Confirm.ask("Approve and continue?", default=True)
    client.submit_optimize_user_response(run_id, approved=approved)


# Backward-compatible alias used by workflow_cmd
def run_server_workflow(
    agent_name: str,
    workflow_name: str,
    **kwargs: Any,
) -> Any:
    if workflow_name not in ("optimize_loop", "optimize_setup", "optimize"):
        raise SystemExit(f"Workflow {workflow_name!r} is no longer supported; use optimize.")
    return run_optimize(agent_name, **kwargs)
