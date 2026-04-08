"""CLI branding and display helpers for OverClaw.

Brand constants
---------------
BRAND   The primary brand colour (#ED670F — Overmind orange).
        Import this everywhere a hard-coded colour string would otherwise appear.

The SVG favicon is a 192×192 pixel-art image built from 6×6 px tiles,
giving a 32×32 colour grid.  We convert it to terminal art using the UTF-8
upper-half-block character (▀) to pair rows, halving the line count.

Logo / prompts
--------------
render_logo(console, *, small=False)
    Print the logo centred.  ``small=True`` uses half resolution (16 cols × 8
    lines) for per-question use; the default is full resolution (32 × 16).

overmind_prompt(console, prompt, **kwargs) -> str
    Show the small logo then call Rich's Prompt.ask.

select_option(options, *, title, default_index, console) -> int
    Present a list of options that the user navigates with arrow keys.
    Returns the selected index.

Progress / paths
----------------
make_spinner_progress(console, …)
    Returns a ``rich.progress.Progress`` with brand-orange spinner.

rel(path)
    Path relative to CWD for display.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.text import Text
from simple_term_menu import TerminalMenu

# ---------------------------------------------------------------------------
# Brand colour
# ---------------------------------------------------------------------------

BRAND = "#ED670F"  # Overmind orange

# ---------------------------------------------------------------------------
# Logo rendering
# ---------------------------------------------------------------------------

_SVG_PATH = Path(__file__).resolve().parent.parent / "static" / "overmind_favicon.svg"
_SVG_GRID_SIZE = 32
_SVG_TILE = 6
_SVG_NS = "http://www.w3.org/2000/svg"
_logo_grid_cache: list[list[str | None]] | None = None


def _load_logo_grid() -> list[list[str | None]]:
    """Return a 32×32 grid of hex colour strings (None = transparent)."""
    global _logo_grid_cache
    if _logo_grid_cache is not None:
        return _logo_grid_cache

    if not _SVG_PATH.exists():
        _logo_grid_cache = []
        return _logo_grid_cache

    try:
        root_el = ET.parse(_SVG_PATH).getroot()
        grid: list[list[str | None]] = [
            [None] * _SVG_GRID_SIZE for _ in range(_SVG_GRID_SIZE)
        ]
        for rect in root_el.iter(f"{{{_SVG_NS}}}rect"):
            fill = rect.get("fill", "").strip()
            if not fill.startswith("#"):
                continue
            try:
                gx = int(rect.get("x") or 0) // _SVG_TILE
                gy = int(rect.get("y") or 0) // _SVG_TILE
            except (ValueError, TypeError):
                continue
            if 0 <= gx < _SVG_GRID_SIZE and 0 <= gy < _SVG_GRID_SIZE:
                grid[gy][gx] = fill
        _logo_grid_cache = grid
    except Exception:
        _logo_grid_cache = []

    return _logo_grid_cache


def render_logo(console: Console, *, small: bool = False) -> None:
    """Print the Overmind favicon as colour block art, centred.

    ``small=True`` renders at half scale (16 cols × 8 lines) for per-question
    use; the default renders at full scale (32 cols × 16 lines) for headers.
    """
    grid = _load_logo_grid()
    if not grid:
        return

    col_step = 2 if small else 1
    row_step = 4 if small else 2

    for row in range(0, _SVG_GRID_SIZE, row_step):
        top = grid[row]
        mid_row = row + (row_step // 2)
        mid = grid[mid_row] if mid_row < _SVG_GRID_SIZE else [None] * _SVG_GRID_SIZE
        line = Text()
        for col in range(0, _SVG_GRID_SIZE, col_step):
            tc, bc = top[col], mid[col]
            if tc is None and bc is None:
                line.append(" ")
            elif tc is None:
                line.append("▄", style=bc)
            elif bc is None:
                line.append("▀", style=tc)
            else:
                line.append("▀", style=f"{tc} on {bc}")
        console.print(line, justify="center")


def overmind_prompt(console: Console, prompt: str, **kwargs) -> str:
    """Print the small Overmind logo above a free-text question."""
    console.print()
    render_logo(console, small=True)
    return Prompt.ask(prompt.lstrip(), **kwargs)


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


def select_option(
    options: list[str],
    *,
    title: str = "",
    default_index: int = 0,
    console: Console | None = None,
) -> int:
    """Present *options* as an arrow-key navigable menu and return the chosen index.

    Falls back to a numbered ``Prompt.ask`` when the terminal doesn't support
    the interactive menu (e.g. non-TTY / CI).
    """
    if console and title:
        console.print(f"\n  [dim]{title}[/dim]")

    menu = TerminalMenu(
        options,
        cursor_index=default_index,
        menu_cursor="  ▸ ",
        menu_cursor_style=("fg_yellow", "bold"),
        menu_highlight_style=("fg_yellow", "bold"),
    )
    idx = menu.show()

    if idx is None:
        raise SystemExit(0)
    if console:
        console.print(f"  [bold]{options[idx]}[/bold]")
        console.print()
    return idx


def confirm_option(
    prompt: str,
    *,
    default: bool = True,
    console: Console | None = None,
) -> bool:
    """Yes/No confirmation via arrow-key menu. Returns ``True`` for Yes."""
    if console:
        console.print(f"\n  [dim]{prompt}[/dim]")

    choices = ["Yes", "No"]
    menu = TerminalMenu(
        choices,
        cursor_index=0 if default else 1,
        menu_cursor="  ▸ ",
        menu_cursor_style=("fg_yellow", "bold"),
        menu_highlight_style=("fg_yellow", "bold"),
    )
    idx = menu.show()

    if idx is None:
        raise SystemExit(0)
    if console:
        console.print(f"  [bold]{choices[idx]}[/bold]")
        console.print()
    return idx == 0


__all__ = [
    "BRAND",
    "confirm_option",
    "make_spinner_progress",
    "overmind_prompt",
    "rel",
    "render_logo",
    "select_option",
]
