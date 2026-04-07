"""OverClaw agent management commands.

overclaw agent register <name> <module:function>
overclaw agent list
overclaw agent remove <name>
overclaw agent update <name> <module:function>
overclaw agent show <name>
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from overclaw.utils.display import BRAND
from overclaw.core.paths import agent_experiments_dir, agent_setup_spec_dir
from overclaw.core.constants import overclaw_rel
from overclaw.core.registry import (
    load_registry,
    remove_agent,
    save_agent,
    validate_entrypoint,
)


def _other_agents_with_entrypoint(
    registry: dict[str, dict[str, str]],
    entrypoint: str,
    *,
    exclude_name: str | None = None,
) -> list[str]:
    """Return registered agent names (other than *exclude_name*) using this entrypoint."""
    ep = entrypoint.strip()
    return sorted(
        n
        for n, data in registry.items()
        if (exclude_name is None or n != exclude_name)
        and data.get("entrypoint", "").strip() == ep
    )


def _confirm_duplicate_entrypoint(
    console: Console,
    entrypoint: str,
    existing_names: list[str],
    *,
    for_update: bool = False,
) -> None:
    """Print a warning and exit unless the user confirms."""
    listed = ", ".join(f"[bold]{n}[/bold]" for n in existing_names)
    console.print(
        f"\n  [yellow]Warning:[/yellow] This entrypoint is already registered for: {listed}\n"
        f"  [dim]{entrypoint}[/dim]\n\n"
        f"  Each agent name has its own tree under [bold]{overclaw_rel('agents')}/[/bold] "
        "(setup_spec, experiments). The same entrypoint can be registered twice with "
        "different names; eval data stays separate per name.\n"
    )
    prompt = (
        "  Point this agent at that shared entrypoint anyway?"
        if for_update
        else "  Register this entrypoint for another agent name anyway?"
    )
    if not Confirm.ask(prompt, default=False):
        console.print("  [dim]Aborted.[/dim]\n")
        raise SystemExit(0)


def cmd_register(name: str, entrypoint: str) -> None:
    console = Console()
    registry = load_registry()

    if name in registry:
        current_ep = registry[name]["entrypoint"].strip()
        if current_ep == entrypoint.strip():
            raise SystemExit(0)
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is already registered.\n"
            f"  Current entrypoint: [dim]{registry[name]['entrypoint']}[/dim]\n\n"
            f"  To use a different entrypoint:\n"
            f"    [bold]overclaw agent update {name} <module:function>[/bold]\n"
        )
        raise SystemExit(1)

    dupes = _other_agents_with_entrypoint(registry, entrypoint)
    if dupes:
        _confirm_duplicate_entrypoint(console, entrypoint, dupes, for_update=False)

    file_path, fn = validate_entrypoint(entrypoint)
    save_agent(name, entrypoint)

    console.print(
        f"\n  [bold green]✓[/bold green]  "
        f"Agent '[bold]{name}[/bold]' registered.\n"
        f"  [dim]Entrypoint:[/dim] {entrypoint}\n"
        f"  [dim]File:[/dim]      {file_path}\n"
        f"  [dim]Function:[/dim]  {fn}\n\n"
        f"  Next step: [bold {BRAND}]overclaw setup {name}[/bold {BRAND}]\n"
    )


def cmd_list() -> None:
    console = Console()
    registry = load_registry()

    if not registry:
        console.print(
            "\n  [dim]No agents registered yet.[/dim]\n\n"
            "  Register one:\n"
            "    [bold]overclaw agent register <name> <module:function>[/bold]\n"
        )
        return

    table = Table(border_style="cyan", show_header=True, show_lines=False)
    table.add_column("NAME", style=f"bold {BRAND}")
    table.add_column("ENTRYPOINT")
    table.add_column("FILE", justify="center")

    for name, data in registry.items():
        file_ok = (
            "[green]✓[/green]" if Path(data["file_path"]).exists() else "[red]✗[/red]"
        )
        table.add_row(name, data["entrypoint"], file_ok)

    console.print()
    console.print(table)
    console.print()


def cmd_remove(name: str) -> None:
    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            "  To see all registered agents:\n"
            "    [bold]overclaw agent list[/bold]\n"
        )
        raise SystemExit(1)

    console.print(
        f"\n  Agent '[bold]{name}[/bold]'  [dim]{registry[name]['entrypoint']}[/dim]"
    )
    if not Confirm.ask(f"  Remove '{name}' from the registry?", default=True):
        console.print("  [dim]Aborted.[/dim]\n")
        raise SystemExit(0)

    remove_agent(name)
    console.print(
        f"\n  [bold green]✓[/bold green]  Agent '[bold]{name}[/bold]' removed.\n"
    )


def cmd_update(name: str, entrypoint: str) -> None:
    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            f"  Use register instead:\n"
            f"    [bold]overclaw agent register {name} {entrypoint}[/bold]\n"
        )
        raise SystemExit(1)

    old_ep_raw = registry[name]["entrypoint"]
    if old_ep_raw.strip() == entrypoint.strip():
        raise SystemExit(0)

    dupes = _other_agents_with_entrypoint(registry, entrypoint, exclude_name=name)
    if dupes:
        _confirm_duplicate_entrypoint(console, entrypoint, dupes, for_update=True)

    validate_entrypoint(entrypoint)
    save_agent(name, entrypoint)

    console.print(
        f"\n  [dim]Old entrypoint:[/dim] {old_ep_raw}\n"
        f"  [dim]New entrypoint:[/dim] {entrypoint}\n"
    )


def cmd_show(name: str) -> None:
    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            "  To see all registered agents:\n"
            "    [bold]overclaw agent list[/bold]\n"
        )
        raise SystemExit(1)

    data = registry[name]
    file_path = Path(data["file_path"])
    file_exists = file_path.exists()

    spec_path = agent_setup_spec_dir(name) / "eval_spec.json"
    spec_exists = spec_path.exists()

    experiments_dir = agent_experiments_dir(name)
    exp_files = (
        [f for f in experiments_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        if experiments_dir.exists()
        else []
    )

    file_status = "[green]✓ exists[/green]" if file_exists else "[red]✗ not found[/red]"
    spec_status = (
        "[green]✓ ready[/green]" if spec_exists else "[yellow]not run yet[/yellow]"
    )
    exp_status = (
        f"[green]✓ {len(exp_files)} file(s)[/green]"
        if exp_files
        else "[yellow]not run yet[/yellow]"
    )

    lines = (
        f"[bold]Name:[/bold]        {name}\n"
        f"[bold]Entrypoint:[/bold]  {data['entrypoint']}\n"
        f"[bold]File:[/bold]        {data['file_path']}  {file_status}\n"
        f"[bold]Setup spec:[/bold]  {spec_path}  {spec_status}\n"
        f"[bold]Experiments:[/bold] {exp_status}"
    )

    console.print()
    console.print(
        Panel(
            lines,
            title=f"[bold {BRAND}]{name}[/bold {BRAND}]",
            border_style=BRAND,
            padding=(1, 2),
        )
    )
    console.print()
