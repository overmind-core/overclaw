"""Detect external imports and walk the user through adding a manifest.

Extracted from ``overmind/commands/setup_cmd.py``: keeps the dependency
probing logic — and its three-way interactive prompt — in a focused module
so the orchestrator can stay thin.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from overmind.utils.display import confirm_option, rel, select_option

__all__ = ["check_agent_dependencies"]


def check_agent_dependencies(
    agent_path: str,
    agent_name: str,
    console: Console,
    *,
    fast: bool = False,
    instrumented_dir: Path | None = None,
) -> None:
    """Detect external imports without a dependency manifest and guide the user.

    When *instrumented_dir* is provided the dependency check and any
    generated manifest are placed inside the instrumented copy so the
    original agent source is never modified.  The manifest is also written
    to the instrumented **root** (the top of the copied tree) so the
    runner's ``ensure_environment`` finds it when provisioning the sandbox.

    In interactive mode: offers to generate a ``requirements.txt`` /
    ``package.json`` or lets the user handle it themselves.  In ``--fast``
    mode: fails with a clear message.
    """
    from overmind.optimize.runner import (
        Language,
        detect_external_imports,
        generate_package_json,
        generate_requirements_txt,
        has_dep_manifest,
        imports_to_package_names,
    )

    p = Path(agent_path).resolve()
    agent_dir = p.parent
    entry_file = p.name

    check_dir = instrumented_dir if instrumented_dir is not None else agent_dir

    try:
        language = Language.from_path(entry_file)
    except ValueError:
        return

    if has_dep_manifest(check_dir, language):
        console.print(f"  [bold green]\u2713[/bold green] Found dependency manifest in [dim]{rel(check_dir)}[/dim]")
        return

    inst_entry = check_dir / entry_file if instrumented_dir is not None else p
    if inst_entry.is_file():
        ext_imports = detect_external_imports(check_dir, entry_file, language)
    else:
        ext_imports = detect_external_imports(agent_dir, entry_file, language)
    if not ext_imports:
        return

    packages = imports_to_package_names(ext_imports, language)
    is_python = language == Language.PYTHON
    manifest_name = "requirements.txt" if is_python else "package.json"

    console.print()
    console.print(
        Panel(
            f"[bold yellow]No dependency file found[/bold yellow]\n\n"
            f"Your agent imports [bold]{len(ext_imports)}[/bold] external package(s):\n"
            f"  [cyan]{', '.join(ext_imports[:12])}"
            f"{'…' if len(ext_imports) > 12 else ''}[/cyan]\n\n"
            f"But there is no [bold]{manifest_name}[/bold] in the project.\n\n"
            f"Overmind needs a dependency file to create an isolated\n"
            f"environment so your agent runs reliably.",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    if fast:
        console.print(f"  [red]Create a [bold]{manifest_name}[/bold] in your project and re-run setup.[/red]\n")
        raise SystemExit(1)

    choice = select_option(
        [
            f"Generate {manifest_name} (auto-detected — you review before continuing)",
            f"I'll create {manifest_name} myself (exit setup, re-run when ready)",
            "Skip isolation — use the current environment (not recommended)",
        ],
        title="How would you like to proceed?",
        default_index=0,
        console=console,
    )

    if choice == 0:
        dest = check_dir / ("requirements.txt" if is_python else "package.json")
        if is_python:
            content = generate_requirements_txt(packages)
        else:
            content = generate_package_json(packages, agent_name)

        dest.write_text(content)

        console.print()
        console.print(
            Panel(
                f"[bold green]Generated {manifest_name}[/bold green]\n\n"
                + "\n".join(f"  {pkg}" for pkg in sorted(set(packages)))
                + f"\n\n[dim]Saved to: {rel(dest)}[/dim]\n\n"
                + "[yellow]Versions are unpinned. Review and pin versions\n"
                "for reproducibility before production use.[/yellow]",
                border_style="green",
                padding=(1, 2),
            )
        )

        if not confirm_option("Continue with setup?", default=True, console=console):
            console.print(f"\n  [dim]Edit [cyan]{rel(dest)}[/cyan] and re-run setup when ready.[/dim]\n")
            raise SystemExit(0)

    elif choice == 1:
        console.print(
            f"\n  Create [bold]{manifest_name}[/bold] in your project, then re-run:\n"
            f"    [bold]overmind setup {agent_name}[/bold]\n"
        )
        raise SystemExit(0)

    else:
        console.print(
            "\n  [yellow]Skipping dependency isolation.[/yellow]\n"
            "  [dim]The agent will run using packages from the current environment.\n"
            "  If imports fail during optimization, create a dependency file and retry.[/dim]\n"
        )
