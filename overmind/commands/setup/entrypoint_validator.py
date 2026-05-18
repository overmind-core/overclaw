"""Verify (and, if needed, regenerate) the agent's entry-point function."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from overmind.core.constants import overmind_rel
from overmind.core.registry import save_agent
from overmind.optimize.evaluator import has_entrypoint
from overmind.utils.display import confirm_option, make_spinner_progress, rel, select_option

__all__ = ["validate_agent_entrypoint"]


def validate_agent_entrypoint(
    agent_path: str,
    fn_name: str,
    agent_name: str,
    console: Console,
    *,
    fast: bool = False,
) -> tuple[str, str]:
    """Verify the agent file defines the registered entry function.

    Returns ``(agent_path, fn_name)`` — unchanged when valid, or updated to
    point at a generated wrapper when the user opts in.
    """
    from overmind.entrypoint_wrapper import (
        generate_entrypoint_wrapper,
        wrapper_entrypoint,
    )
    from overmind.optimize.runner import AgentRunner

    code = Path(agent_path).read_text()

    p = Path(agent_path).resolve()
    try:
        runner = AgentRunner(agent_dir=p.parent, entry_file=p.name, entrypoint_fn=fn_name)
        found = runner.validate_entrypoint(code)
    except ValueError:
        found = has_entrypoint(code, fn_name)

    if found:
        return agent_path, fn_name

    if fast:
        console.print(
            f"\n  [bold red]Error:[/bold red] Function [bold]{fn_name}()[/bold] not found "
            f"in [cyan]{agent_path}[/cyan].\n"
            f"  Generate a wrapper first:\n"
            f"    [bold]overmind agent register {agent_name} <module:function>[/bold]\n"
        )
        raise SystemExit(1)

    console.print(
        f"\n  [bold yellow]\u26a0[/bold yellow]  Function [bold]{fn_name}()[/bold] not found "
        f"in [cyan]{rel(agent_path)}[/cyan].\n"
    )
    console.print(
        "  Overmind needs a function that takes input and returns output, e.g.:\n"
        "    [dim]def run(input_data: dict) -> dict[/dim]\n"
    )

    choice = select_option(
        [
            "Generate an entrypoint wrapper (Overmind reads your code and creates one)",
            "I'll fix it myself (exit setup)",
        ],
        title="How would you like to proceed?",
        default_index=0,
        console=console,
    )

    if choice != 0:
        console.print(f"\n  Fix the entrypoint and re-run:\n    [bold]overmind setup {agent_name}[/bold]\n")
        raise SystemExit(1)

    agent_dir = p.parent
    console.print()
    with make_spinner_progress(console) as progress:
        progress.add_task("  Analyzing agent code and generating wrapper\u2026")
        wp = generate_entrypoint_wrapper(agent_dir, agent_name)

    if wp == "refused":
        console.print(
            "\n  [bold yellow]⚠[/bold yellow]  This agent's code is too complex for an "
            "auto-generated wrapper.\n\n"
            "  The wrapper needs to be a trivial bridge (import + call), but this\n"
            "  agent would require re-implementing agent-specific logic.\n\n"
            "  Add a [bold]def run(input_data: dict) -> dict[/bold] function directly\n"
            "  in your agent code, then re-register:\n"
            f"    [bold]overmind agent register {agent_name} <your_module:run>[/bold]\n"
        )
        raise SystemExit(1)

    if wp is None or not wp.is_file():
        console.print(
            "\n  [bold red]\u2717[/bold red]  Wrapper generation failed.\n"
            "  This can happen if no LLM model is configured.\n"
            f"  Set [bold]ANALYZER_MODEL[/bold] in [bold]{overmind_rel('.env')}[/bold] "
            "or write the wrapper manually.\n"
        )
        raise SystemExit(1)

    wrapper_code = wp.read_text(encoding="utf-8")

    from rich.syntax import Syntax

    console.print()
    console.print(
        Panel(
            f"[bold green]Generated entrypoint wrapper[/bold green]\n\n"
            f"  File:     [cyan]{rel(wp)}[/cyan]\n"
            f"  Function: [bold]run(input_data: dict) -> dict[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )

    if confirm_option("Review the generated code?", default=True, console=console):
        console.print()
        console.print(
            Syntax(
                wrapper_code,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        )

    console.print()
    if not confirm_option("Continue setup with this wrapper?", default=True, console=console):
        console.print(f"\n  [dim]Edit [cyan]{rel(wp)}[/cyan] and re-run setup.[/dim]\n")
        raise SystemExit(0)

    new_agent_path = str(wp)
    new_fn_name = "run"
    new_ep = wrapper_entrypoint(agent_name)
    save_agent(agent_name, new_ep)

    console.print(f"  [dim]Updated registry \u2192 {new_ep}[/dim]\n")
    return new_agent_path, new_fn_name
