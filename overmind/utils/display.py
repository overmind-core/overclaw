"""CLI branding and display helpers for Overmind.

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

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
from simple_term_menu import TerminalMenu

from overmind.core.logging import log_prompt

logger = logging.getLogger("overmind.display")

# ---------------------------------------------------------------------------
# Non-interactive detection
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})


def is_non_interactive() -> bool:
    """Return True when the CLI should skip interactive widgets entirely.

    Triggered **only** by explicit opt-in so that piping stdout (e.g.
    ``overmind agent register foo bar | tee log.txt``) does NOT silently
    suppress the menu — ``simple_term_menu`` reads/writes ``/dev/tty``
    directly and is happy in that case.

    Recognised signals (any one is enough):
    * ``OVERMIND_NONINTERACTIVE=1`` — preferred, set by ``--non-interactive``.
    * ``CI=1`` (or ``true``/``yes``…) — common CI convention.

    Sandboxes that don't expose ``/dev/tty`` at all are handled separately
    by the ``OSError`` fallback inside :func:`select_option` and
    :func:`confirm_option`; you don't need to opt in for that path.
    """
    flag = os.environ.get("OVERMIND_NONINTERACTIVE", "").strip().lower()
    if flag in _TRUTHY:
        return True
    if os.environ.get("CI", "").strip().lower() in _TRUTHY:
        return True
    return False


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
        grid: list[list[str | None]] = [[None] * _SVG_GRID_SIZE for _ in range(_SVG_GRID_SIZE)]
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
    # Note: the underlying rich.prompt.Prompt is monkey-patched in
    # overmind.core.logging to log every ask, so we don't double-log here.
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


def _criteria_detail(ftype: str, fc: dict) -> str:
    """Return the human-readable scoring detail for a single eval field."""
    if ftype == "enum":
        return "partial credit" if fc.get("partial_credit", True) else "exact match only"
    if ftype == "number":
        return f"tolerance \u00b1{fc.get('tolerance', 10)}"
    if ftype == "text":
        return "check non-empty" if fc.get("eval_mode", "non_empty") == "non_empty" else "skip"
    return "exact match"


def render_criteria_table(
    console: Console,
    criteria: dict,
    output_schema: dict,
    *,
    title: str = "Proposed Evaluation Criteria",
    colorize_importance: bool = False,
    show_lines: bool = False,
    padding: tuple[int, int] | None = None,
    min_widths: bool = False,
) -> None:
    """Render the standard "field / importance / scoring detail" Rich table.

    Centralises the Rich table that previously had three near-identical copies
    in :mod:`overmind.setup.agent_analyzer`, :mod:`overmind.setup.questionnaire`,
    and :mod:`overmind.commands.setup_cmd`.

    Parameters
    ----------
    console:
        The Rich console to print to.  A blank line is printed first so the
        table is visually separated from preceding output.
    criteria:
        ``analysis["proposed_criteria"]`` (or refined equivalent).
    output_schema:
        ``analysis["output_schema"]`` — used to look up the field type for
        each row so the scoring detail can be rendered consistently.
    title:
        Override the default ``"Proposed Evaluation Criteria"`` title.
    colorize_importance:
        When ``True``, importance text is colourised (red/yellow/dim).
    show_lines:
        Forwarded to :class:`rich.table.Table`.
    padding:
        Forwarded to :class:`rich.table.Table` when supplied.
    min_widths:
        When ``True``, apply the wider column widths used by the analyzer's
        proposal table.
    """
    fields_criteria = criteria.get("fields", {})
    if not fields_criteria:
        return

    table_kwargs: dict[str, object] = {"title": title, "border_style": "green"}
    if show_lines:
        table_kwargs["show_lines"] = True
    if padding is not None:
        table_kwargs["padding"] = padding
    table = Table(**table_kwargs)

    if min_widths:
        table.add_column("Field", style="bold", min_width=12)
        table.add_column("Importance", min_width=10)
        table.add_column("Scoring Detail", ratio=1)
    else:
        table.add_column("Field", style="bold")
        table.add_column("Importance")
        table.add_column("Scoring Detail")

    for field_name, fc in fields_criteria.items():
        importance = fc.get("importance", "important")
        ftype = output_schema.get(field_name, {}).get("type", "text")
        detail = _criteria_detail(ftype, fc)

        if colorize_importance:
            imp_style = "red" if importance == "critical" else "yellow" if importance == "important" else "dim"
            importance_cell = f"[{imp_style}]{importance}[/{imp_style}]"
        else:
            importance_cell = importance

        table.add_row(field_name, importance_cell, detail)

    sw = criteria.get("structure_weight", 20)
    table.add_row(
        "[dim]structure[/dim]",
        "[dim]\u2014[/dim]",
        f"[dim]{sw} pts for completeness[/dim]",
    )
    console.print()
    console.print(table)


def _clamp_default_index(default_index: int, n: int) -> int:
    """Clamp *default_index* into ``[0, n-1]`` (assuming ``n >= 1``)."""
    if n <= 0:
        return 0
    if default_index < 0:
        return 0
    if default_index >= n:
        return n - 1
    return default_index


def _select_option_prompt_fallback(
    options: list[str],
    *,
    title: str,
    default_index: int,
    console: Console | None,
) -> int:
    """Numbered ``Prompt.ask`` fallback used when the arrow-key menu can't run.

    Shows each option as ``[N] label`` and asks for the 1-based index, with
    *default_index* (0-based) pre-selected.  Returns the chosen 0-based index.

    If stdin is closed (e.g. coding agent subprocess) the prompt raises
    ``EOFError`` and we silently fall back to *default_index* — better to
    keep going with a sane default than to crash mid-registration.
    """
    if console:
        for i, label in enumerate(options, start=1):
            marker = " [bold yellow](default)[/bold yellow]" if (i - 1) == default_index else ""
            console.print(f"    [{i}] {label}{marker}")

    while True:
        try:
            choice = IntPrompt.ask(
                "  Enter choice number",
                default=default_index + 1,
                show_default=True,
                console=console,
            )
        except EOFError:
            logger.info(
                "select_option text-prompt fallback: stdin EOF, using default_index=%d",
                default_index,
            )
            if console:
                console.print(
                    f"  [dim]No input available — using default: [bold]{options[default_index]}[/bold].[/dim]"
                )
            return default_index
        if 1 <= choice <= len(options):
            return choice - 1
        if console:
            console.print(f"  [red]Please enter a number between 1 and {len(options)}.[/red]")


def select_option(
    options: list[str],
    *,
    title: str = "",
    default_index: int = 0,
    console: Console | None = None,
) -> int:
    """Present *options* as an arrow-key navigable menu and return the chosen index.

    Behavior by environment:

    * **Interactive TTY** — renders the arrow-key menu via ``simple_term_menu``.
    * **Non-interactive** (``OVERMIND_NONINTERACTIVE=1``, ``CI=1``, or no TTY)
      — silently returns *default_index* and logs the selection.
    * **TTY without ``/dev/tty``** (sandboxed / restricted shells) — falls
      back to a numbered ``Prompt.ask`` so the command keeps working.
    """
    if not options:
        raise ValueError("select_option requires a non-empty options list")

    default_index = _clamp_default_index(default_index, len(options))

    if console and title:
        console.print(f"\n  [dim]{title}[/dim]")

    logger.debug(f"select_option presented title={title!r} options={options!r} default_index={default_index}")

    if is_non_interactive():
        idx = default_index
        log_prompt(
            title or "(select)",
            f"[{idx}] {options[idx]} (non-interactive default)",
            kind="select",
            default=options[default_index],
            logger=logger,
        )
        if console:
            console.print(f"  [bold]{options[idx]}[/bold]  [dim](non-interactive default)[/dim]")
            console.print()
        return idx

    try:
        menu = TerminalMenu(
            options,
            cursor_index=default_index,
            menu_cursor="  ▸ ",
            menu_cursor_style=("fg_yellow", "bold"),
            menu_highlight_style=("fg_yellow", "bold"),
        )
        idx = menu.show()
    except OSError as exc:
        # ``simple_term_menu`` opens ``/dev/tty`` directly — both during
        # ``TerminalMenu.__init__`` and ``menu.show()`` — and raises
        # ``OSError: [Errno 6] Device not configured`` in shells that
        # don't expose a controlling terminal (sandboxed coding-agent
        # subprocesses, Docker without ``-t``, some CI runners).  Fall
        # back to a textual prompt so the user can still complete the
        # command; the prompt itself silently uses *default_index* when
        # stdin is also closed.
        logger.info(f"select_option falling back to text prompt (TerminalMenu OSError: {exc})")
        if console:
            console.print("  [dim]Arrow-key menu unavailable in this shell — using a text prompt instead.[/dim]")
        idx = _select_option_prompt_fallback(options, title=title, default_index=default_index, console=console)

    if idx is None:
        logger.info(f"select_option cancelled title={title!r}")
        raise SystemExit(0)
    log_prompt(
        title or "(select)",
        f"[{idx}] {options[idx]}",
        kind="select",
        default=options[default_index] if 0 <= default_index < len(options) else None,
        logger=logger,
    )
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
    """Yes/No confirmation via arrow-key menu. Returns ``True`` for Yes.

    Behavior by environment:

    * **Interactive TTY** — arrow-key Yes/No menu.
    * **Non-interactive** (``OVERMIND_NONINTERACTIVE=1``, ``CI=1``, or no
      TTY) — returns *default* silently (with a one-line note) and logs it.
    * **TTY without ``/dev/tty``** — falls back to ``rich.prompt.Confirm``.
    """
    if console:
        console.print(f"\n  [dim]{prompt}[/dim]")

    logger.debug(f"confirm_option presented prompt={prompt!r} default={default}")

    if is_non_interactive():
        log_prompt(
            prompt,
            f"{'Yes' if default else 'No'} (non-interactive default)",
            kind="confirm",
            default="Yes" if default else "No",
            logger=logger,
        )
        if console:
            console.print(f"  [bold]{'Yes' if default else 'No'}[/bold]  [dim](non-interactive default)[/dim]")
            console.print()
        return default

    choices = ["Yes", "No"]
    try:
        menu = TerminalMenu(
            choices,
            cursor_index=0 if default else 1,
            menu_cursor="  ▸ ",
            menu_cursor_style=("fg_yellow", "bold"),
            menu_highlight_style=("fg_yellow", "bold"),
        )
        idx = menu.show()
    except OSError as exc:
        # See ``select_option`` for the same /dev/tty fallback rationale.
        # When stdin is also unavailable the ``Confirm.ask`` raises
        # ``EOFError`` — we surface that as the supplied *default* so the
        # caller can keep moving instead of crashing.
        logger.info(f"confirm_option falling back to text prompt (TerminalMenu OSError: {exc})")
        if console:
            console.print("  [dim]Arrow-key menu unavailable in this shell — using a text prompt instead.[/dim]")
        try:
            answer = Confirm.ask("  Confirm", default=default, console=console)
        except EOFError:
            logger.info(
                "confirm_option text-prompt fallback: stdin EOF, using default=%s",
                default,
            )
            if console:
                console.print(
                    f"  [dim]No input available — using default: [bold]{'Yes' if default else 'No'}[/bold].[/dim]"
                )
            answer = default
        idx = 0 if answer else 1

    if idx is None:
        logger.info(f"confirm_option cancelled prompt={prompt!r}")
        raise SystemExit(0)
    log_prompt(
        prompt,
        "Yes" if idx == 0 else "No",
        kind="confirm",
        default="Yes" if default else "No",
        logger=logger,
    )
    if console:
        console.print(f"  [bold]{choices[idx]}[/bold]")
        console.print()
    return idx == 0


__all__ = [
    "BRAND",
    "confirm_option",
    "is_non_interactive",
    "make_spinner_progress",
    "overmind_prompt",
    "rel",
    "render_logo",
    "select_option",
]
