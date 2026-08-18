"""Overmind CLI — optimise LLM agents and manage agent skills.

Commands:
    optimise                  Run the optimisation loop on a registered agent.
    skills                    Manage Overmind agent skills.

Use --help with any command or subcommand for details.
"""

import logging
import os
import shutil
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from typer import echo

from overmind.skills_db import Skill, skills

console = Console()

# Package dir (`.../site-packages/overmind` or `.../repo/overmind`). Skills live at
# the repo-root `skills/` for agent installers; the wheel force-includes that
# tree at `overmind/skills/` so sync still works from an installed package.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _skill_src(slug: str) -> str | None:
    """First candidate dir containing the slug: packaged wheel tree, then repo root.

    A dev checkout has unrelated prompt files at ``overmind/skills/``, so the
    packaged dir existing is not enough — it must contain the skill itself.
    """
    candidates = (
        os.path.join(_PACKAGE_DIR, "skills", slug),
        os.path.join(os.path.dirname(_PACKAGE_DIR), "skills", slug),
    )
    return next((p for p in candidates if os.path.isdir(p)), None)


skills_app = typer.Typer(help="Manage Overmind agent skills.")


@skills_app.command("list", help="List all installed or available skills.")
def list_skills(verbose: bool = False):
    if not verbose:
        for skill in skills:
            echo(skill.name)
        return

    table = Table(title="Overmind Skills")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description", style="dim")
    table.add_column("Version", style="bold red")
    table.add_column("Provider", style="bold green")
    for skill in skills:
        table.add_row(skill.name, skill.description, skill.version, skill.provider)
    console.print(table)


@skills_app.command("sync", help="Sync one or more skills to the latest version.")
def sync_skills(
    names: Annotated[list[str], typer.Argument(..., help="Skill name(s) to update")],
    ide: Annotated[str, typer.Option(..., help="IDE to use")] = "cursor",  # ide: cursor, claude code etc
):
    for name in names:
        skill = next((s for s in skills if s.name == name or s.slug == name), None)
        if skill:
            sync_skill(skill, ide)
        else:
            print(f"Skill {name} not found")


def sync_skill(skill: Skill, ide: str):
    """Copy the whole skill directory (SKILL.md + references/) into the destination."""
    src = _skill_src(skill.slug)
    if src is None:
        raise FileNotFoundError(f"Skill directory not found for: {skill.slug}")
    dest = os.path.join(get_destination_dir(ide), "skills", skill.slug)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    logging.info(f"Copied {src} to {dest}")


def get_destination_dir(ide: str):
    if ide == "cursor":
        return ".cursor"

    if ide == "claude_code" or ide == "claude" or ide == "claude-code":
        return ".claude"

    if ide == "opencode":
        return ".opencode"

    raise typer.BadParameter(f"use cursor, claude_code or opencode, got: {ide}", param_hint="--ide")
