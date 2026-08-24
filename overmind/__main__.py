"""Overmind CLI — optimise LLM agents and manage agent skills.

Commands:
    init                      Configure an IDE with Overmind MCP and skills.
    optimise                  Run the optimisation loop on a registered agent.
    skills                    Manage Overmind agent skills.

Use --help with any command or subcommand for details.
"""
# Allow ``python -m overmind …`` when a different ``overmind`` script shadows PATH.

import json
import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from overmind.optimizer import (
    API_KEY,
    API_URL,
    HEARTBEAT_INTERVAL,
    IDLE_INTERVAL,
    WORK_DIR,
    configure_logging,
    run_optimizer,
)
from overmind.skills import get_destination_dir, skills_app, sync_skills

app = typer.Typer(
    name="overmind",
    help="Overmind CLI optimise LLM agents and manage agent skills.",
    epilog="Overmind https://overmindlab.ai is a tool for optimising LLM agents and managing agent skills.",
    pretty_exceptions_enable=True,
    no_args_is_help=True,
)

console = Console()

current_dir = os.path.dirname(os.path.abspath(__file__))


OPTIMISE_HELP = "Register this machine as an optimiser and run the optimisation loop."


def optimise(
    api_key: Annotated[
        str,
        typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key", show_default=False),
    ] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = API_URL,
    cwd: Annotated[
        str,
        typer.Option(envvar="OVERMIND_CWD", help="Repo root to run commands in (defaults to the current dir)"),
    ] = "",
    poll_interval: Annotated[
        float, typer.Option(envvar="OPTIMIZER_POLL_INTERVAL", help="Idle poll seconds")
    ] = IDLE_INTERVAL,
    heartbeat_interval: Annotated[
        float,
        typer.Option(envvar="OPTIMIZER_HEARTBEAT_INTERVAL", help="Idle 'still alive' log seconds"),
    ] = HEARTBEAT_INTERVAL,
    log_level: Annotated[str, typer.Option(envvar="OPTIMIZER_LOG_LEVEL", help="DEBUG/INFO/WARNING/ERROR")] = "INFO",
):
    """
    Register with the Overmind backend and run the optimisation loop: poll for
    queued commands (from the server-side experiment FSM), run them against
    the repo in ``cwd``, and report results back.
    """
    configure_logging(log_level)
    run_optimizer(
        api_url=api_url,
        api_key=api_key or API_KEY,
        cwd=cwd or WORK_DIR,
        poll_interval=poll_interval,
        heartbeat_interval=heartbeat_interval,
    )


app.command("optimise", help=OPTIMISE_HELP)(optimise)
# ``optimize`` stays registered (hidden) so scripts written against the old
# American spelling keep working.
app.command("optimize", hidden=True, help=f"Deprecated alias for `optimise`. {OPTIMISE_HELP}")(optimise)

app.add_typer(skills_app, name="skills")


MCP_URLS = {
    "staging": "https://staging.overmindlab.ai/api/mcp/",
    "development": "http://localhost:8000/api/mcp/",
    "local": "http://localhost:8000/api/mcp/",
    "dev": "http://localhost:8000/api/mcp/",
}


def _write_codex_mcp(path: Path, url: str, api_key: str | None = None) -> None:
    auth = (
        'env_http_headers = { "X-Api-Key" = "OVERMIND_API_KEY" }'
        if api_key is None
        else f'http_headers = {{ "X-Api-Key" = {json.dumps(api_key)} }}'
    )
    block = "\n".join([
        "[mcp_servers.overmind]",
        f"url = {json.dumps(url)}",
        auth,
    ])
    existing = path.read_text() if path.exists() else ""
    existing = re.sub(
        r"(?ms)^\[\[?mcp_servers\.overmind(?:\.[^\]]+)?\]\]?\s*\n.*?(?=^\s*\[|\Z)",
        "",
        existing,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{existing.rstrip()}\n\n{block}\n" if existing.strip() else f"{block}\n")


def overmind_init(
    env: Annotated[str, typer.Option(help="production, staging or dev")] = "production",
    api_key: Annotated[str, typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key")] = API_KEY,
    ide: Annotated[str, typer.Option(..., help="cursor, claude, claude_code, opencode or codex")] = "cursor",
):
    """
    Add (or update) the overmind MCP server config and install the overmind
    skill, leaving any other configured servers untouched.
    """
    url = MCP_URLS.get(env, "https://api.overmindlab.ai/api/mcp/")
    dest = get_destination_dir(ide)

    if ide == "codex":
        if env in {"local", "development", "dev"}:
            raise typer.BadParameter("Codex setup supports production or staging", param_hint="--env")
        if not os.environ.get("OVERMIND_API_KEY") and not api_key:
            raise typer.BadParameter("set OVERMIND_API_KEY before running init", param_hint="OVERMIND_API_KEY")
        mcp_path = Path.cwd() / ".codex" / "config.toml"
        _write_codex_mcp(mcp_path, url, None if os.environ.get("OVERMIND_API_KEY") else api_key)
        console.print(f"overmind MCP server written to {mcp_path}")
        sync_skills(["overmind"], ide=ide)
        console.print(f"overmind skill installed to {dest}/skills/overmind")
        return

    if ide == "opencode":
        # opencode reads MCP servers from project-root opencode.json, not <dest>/mcp.json
        mcp_path = Path.cwd() / "opencode.json"
        config = (
            json.loads(mcp_path.read_text()) if mcp_path.exists() else {"$schema": "https://opencode.ai/config.json"}
        )
        config.setdefault("mcp", {})["overmind"] = {
            "type": "remote",
            "url": url,
            "enabled": True,
            "headers": {"X-Api-Key": api_key},
        }
    else:
        mcp_path = Path.cwd() / dest / "mcp.json"
        config = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
        config.setdefault("mcpServers", {})["overmind"] = {
            "url": url,
            "headers": {"X-Api-Key": api_key},
        }

    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(json.dumps(config, indent=2) + "\n")
    console.print(f"overmind MCP server written to {mcp_path.name if ide == 'opencode' else f'{dest}/mcp.json'}")

    sync_skills(["overmind"], ide=ide)
    console.print(f"overmind skill installed to {dest}/skills/overmind")

    # claude mcp add --transport http corridor https://app.corridor.dev/api/mcp --header "Authorization: Bearer ..."


app.command("init")(overmind_init)


if __name__ == "__main__":
    app()
