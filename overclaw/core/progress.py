"""Shared progress and display utilities for the OverClaw CLI.

``make_spinner_progress(console)``
    Returns a ``rich.progress.Progress`` instance pre-configured with
    a brand-orange spinner and text column.  Use it everywhere a single
    blocking LLM call needs a visual indicator::

        with make_spinner_progress(console, transient=True) as progress:
            progress.add_task("  Analyzing agent code…")
            result = some_blocking_llm_call(...)

``rel(path)``
    Convert an absolute path to a relative path from the current working
    directory for display purposes.  Falls back to the original path string
    if the path is not under the CWD.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from overclaw.core.branding import BRAND

__all__ = ["BRAND", "make_spinner_progress", "rel"]


def rel(path: str | Path) -> str:
    """Return *path* relative to CWD for display; falls back to absolute."""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def make_spinner_progress(console: Console, *, transient: bool = False) -> Progress:
    """Return a ``Progress`` with brand-orange spinner and text.

    Pass ``transient=True`` to erase the spinner line when the context exits,
    leaving a clean terminal for the next ``console.print`` call.
    """
    return Progress(
        SpinnerColumn(style=BRAND),
        TextColumn(f"[bold {BRAND}]{{task.description}}"),
        console=console,
        transient=transient,
    )
