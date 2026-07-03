"""Overmind CLI — optimize LLM agents and manage agent skills.

Commands:
    optimize                  Run the optimization loop on a registered agent.
    skills                    Manage Overmind agent skills.

Use --help with any command or subcommand for details.
"""
# Allow ``python -m overmind …`` when a different ``overmind`` script shadows PATH.

import os
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
from overmind.skills import skills_app

app = typer.Typer(
    name="overmind",
    help="Overmind CLI optimize LLM agents and manage agent skills.",
    epilog="Overmind https://overmindlab.ai is a tool for optimizing LLM agents and managing agent skills.",
    pretty_exceptions_enable=True,
    no_args_is_help=True,
)

console = Console()

current_dir = os.path.dirname(os.path.abspath(__file__))


@app.command(help="Register this machine as an optimizer and run the optimization loop.")
def optimize(
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
    Register with the Overmind backend and run the optimization loop: poll for
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


app.add_typer(skills_app, name="skills")

if __name__ == "__main__":
    app()
