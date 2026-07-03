"""Overmind CLI — optimize LLM agents and manage agent skills.

Commands:
    optimize                  Run the optimization loop on a registered agent.
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

current_dir = os.path.dirname(os.path.abspath(__file__))


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
        skill = next((s for s in skills if s.name == name), None)
        if skill:
            sync_skill(skill, ide)
        else:
            print(f"Skill {name} not found")


def sync_skill(skill: Skill, ide: str):
    """Copy a skill file into the destination directory for Cursor IDE."""
    dest_dir = get_destination_dir(ide)
    dest_path = os.path.join(dest_dir, os.path.basename(skill.slug))

    if os.path.exists(dest_path):
        logging.info(f"{dest_path} already exists, skipping copy.")
        return
    shutil.copy2(skill.slug, dest_path)
    logging.info(f"Copied {skill.slug} to {dest_path}")


def get_destination_dir(ide: str):
    if ide == "cursor":
        return ".cursor/skills"

    if ide == "claude":
        return ".claude/skills"

    raise ValueError(f"Invalid IDE: {ide}")
