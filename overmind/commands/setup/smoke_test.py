"""Subprocess smoke-test helpers used at the start and end of setup.

These functions exercise the agent's entry function through
:class:`AgentRunner` (so dependency isolation, venv provisioning, and
``.env`` loading all match the optimizer's behaviour) and surface the
result through the same Rich panels the rest of setup uses.

Public helpers
--------------
:func:`resolve_seed_json_files`
    Normalise a ``--data`` argument to a list of JSON files.
:func:`smoke_test_agent`
    Run the agent once and capture success / error output.
:func:`run_beginning_smoke_test`
    Pre-setup smoke test using seed data (hard-fails on failure).
:func:`run_end_smoke_test`
    Post-setup smoke test using the generated dataset (warns only).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from overmind.core.paths import agent_instrumented_dir, agent_setup_spec_dir
from overmind.core.registry import project_root_from_agent_file
from overmind.optimize.data import (
    check_consistent_fields,
    load_data,
    normalize_data_fields,
)
from overmind.utils.display import make_spinner_progress, rel

logger = logging.getLogger("overmind.commands.setup.smoke_test")

__all__ = [
    "resolve_seed_json_files",
    "run_beginning_smoke_test",
    "run_end_smoke_test",
    "smoke_test_agent",
]


def resolve_seed_json_files(data_arg: str | None, *, console: Console) -> list[Path]:
    """Resolve ``--data`` to a list of JSON seed files.

    Accepts a single ``*.json`` file or a directory of ``*.json`` files.
    Hard-fails (SystemExit 1) when the path doesn't exist or doesn't
    match these shapes; returns an empty list when *data_arg* is blank.
    """
    if not (data_arg or "").strip():
        return []
    raw = data_arg.strip()
    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError as exc:
        console.print(
            f"\n  [red]Error:[/red] Could not resolve [bold]--data[/bold] path [cyan]{raw}[/cyan] [dim]({exc})[/dim]"
        )
        raise SystemExit(1) from exc
    if not p.exists():
        console.print(f"\n  [red]Error:[/red] [bold]--data[/bold] path does not exist: [cyan]{raw}[/cyan]")
        raise SystemExit(1)
    if p.is_file():
        if p.suffix.lower() != ".json":
            console.print(
                f"\n  [red]Error:[/red] [bold]--data[/bold] must be a [bold].json[/bold] file or a "
                f"directory of JSON files; got [cyan]{p.name}[/cyan]"
            )
            raise SystemExit(1)
        return [p]
    if p.is_dir():
        found = sorted(p.glob("*.json"))
        if not found:
            console.print(
                f"  [yellow]Warning:[/yellow] No [bold].json[/bold] files in [cyan]{rel(p)}[/cyan] "
                "— continuing without seed files from this path."
            )
        return found
    console.print(f"\n  [red]Error:[/red] [bold]--data[/bold] must be a file or directory: [cyan]{raw}[/cyan]")
    raise SystemExit(1)


def smoke_test_agent(
    agent_path: str,
    fn_name: str,
    input_case: dict,
    env_dir: str | Path | None = None,
    agent_dir: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Run the agent via subprocess and call ``fn_name(input_case)`` once.

    Returns ``(True, None)`` on success or ``(False, error_message)`` on any
    exception.  Uses :class:`AgentRunner` so dependency isolation, venv
    provisioning, and ``.env`` loading all match the optimizer.

    *agent_dir* overrides the working directory for the subprocess.  When
    running the instrumented copy, pass the instrumented root so local
    imports resolve correctly.  *env_dir* should point to the **original**
    project root so dependency manifests, ``.venv``, and ``.env`` are found.
    """
    from overmind.optimize.runner import AgentRunner, RunnerConfig

    try:
        p = Path(agent_path).resolve()
        if agent_dir is not None:
            resolved_agent_dir = Path(agent_dir).resolve()
        else:
            pr = project_root_from_agent_file(agent_path)
            resolved_agent_dir = pr if pr is not None else p.parent
        entry_file = str(p.relative_to(resolved_agent_dir))
        logger.debug(
            f"smoke_test: agent_path={agent_path} fn={fn_name} entry={entry_file} "
            f"agent_dir={resolved_agent_dir} env_dir={env_dir}"
        )

        runner = AgentRunner(
            agent_dir=resolved_agent_dir,
            entry_file=entry_file,
            entrypoint_fn=fn_name,
            config=RunnerConfig(timeout=300),
            env_dir=Path(env_dir) if env_dir else None,
        )
        runner.ensure_environment()
        result = runner.run(input_case)
        runner.cleanup()
        if result.success:
            logger.debug(f"smoke_test: agent={agent_path} succeeded")
            return True, None
        parts = [result.error] if result.error else []
        if result.stderr and result.stderr.strip() not in (result.error or ""):
            parts.append(result.stderr[-2000:])
        logger.warning(
            f"smoke_test: agent={agent_path} failed rc={result.returncode} err={(result.error or '')[:300]}"
        )
        return False, "\n".join(parts) or "Unknown error"

    except Exception as exc:
        logger.exception(f"smoke_test: exception for agent={agent_path}")
        return False, str(exc)


def run_beginning_smoke_test(
    agent_path: str,
    agent_name: str,
    fn_name: str,
    console: Console,
    *,
    fast: bool = False,
    data_path: str | None = None,
    instrumented_entry: str | None = None,
) -> None:
    """Smoke-test the agent with the first seed case when ``--data`` supplies JSON.

    When *instrumented_entry* is provided the smoke test runs against the
    instrumented copy (with the original project root as ``env_dir`` so
    dependency manifests and venvs are found).  Hard-fails (SystemExit 1)
    when seed data exists but the agent crashes.  Skips when ``--data`` is
    omitted — use ``--data`` for an early smoke check.
    """
    existing_json = resolve_seed_json_files(data_path, console=console)

    if not existing_json:
        console.print(
            "  [dim]Skipping pre-setup smoke test with seed data "
            "(pass [bold]--data[/bold] with a JSON file or directory of JSON files).[/dim]"
        )
        return

    console.print(
        f"  [dim]Using seed data from [cyan]{rel(existing_json[0])}[/cyan] for smoke test…[/dim]"
    )

    try:
        cases = load_data(str(existing_json[0]))
    except Exception:
        console.print(
            f"  [dim]Could not read [cyan]{existing_json[0].name}[/cyan] — skipping smoke test.[/dim]"
        )
        return

    if not cases:
        console.print(
            f"  [dim][cyan]{existing_json[0].name}[/cyan] is empty — skipping pre-setup smoke test.[/dim]"
        )
        return

    consistent, common_fields, bad_indices = check_consistent_fields(cases)
    if not consistent:
        console.print(
            f"\n  [bold red]Error:[/bold red] Not all data points in "
            f"[cyan]{existing_json[0].name}[/cyan] have the same fields.\n"
            f"  First case fields: {sorted(common_fields)}\n"
            f"  Mismatched at indices: {bad_indices[:10]}"
            + ("  …" if len(bad_indices) > 10 else "")
            + "\n  Please ensure every entry in your seed file has identical top-level keys.\n"
        )
        raise SystemExit(1)

    cases = normalize_data_fields(cases, console, require_output=False, agent_name=agent_name)

    run_path = instrumented_entry or agent_path
    env_dir: str | Path | None = None
    inst_root: str | Path | None = None
    if instrumented_entry:
        pr = project_root_from_agent_file(agent_path)
        env_dir = pr if pr is not None else Path(agent_path).resolve().parent
        inst_root = agent_instrumented_dir(agent_name)

    first_input = cases[0].get("input", cases[0])
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task(
            f"  Smoke-testing agent using {existing_json[0].name} ({len(cases)} case(s))…"
        )
        success, error = smoke_test_agent(
            run_path,
            fn_name,
            first_input,
            env_dir=env_dir,
            agent_dir=inst_root,
        )

    if success:
        console.print("  [bold green]✓[/bold green]  [dim]Agent smoke test passed.[/dim]\n")
    else:
        console.print(
            f"\n  [bold red]✗  Agent smoke test failed[/bold red]\n"
            f"  [dim]{error}[/dim]\n\n"
            "  Fix the error above before running setup.\n"
        )
        raise SystemExit(1)


def run_end_smoke_test(
    agent_name: str,
    agent_path: str,
    fn_name: str,
    console: Console,
    instrumented_entry: str | None = None,
) -> None:
    """Validate the agent runs against the first generated dataset case.

    When *instrumented_entry* is provided the smoke test executes against
    the instrumented copy (matching what the optimizer will run) with the
    original project root as ``env_dir``.

    Issues a warning panel on failure but does NOT abort — the spec is
    already saved and the user should be informed rather than left with a
    silent problem.
    """
    dataset_path = agent_setup_spec_dir(agent_name) / "dataset.json"
    if not dataset_path.exists():
        return

    try:
        cases = load_data(str(dataset_path))
    except Exception:
        return

    if not cases:
        return

    run_path = instrumented_entry or agent_path
    env_dir: str | Path | None = None
    inst_root: str | Path | None = None
    if instrumented_entry:
        pr = project_root_from_agent_file(agent_path)
        env_dir = pr if pr is not None else Path(agent_path).resolve().parent
        inst_root = agent_instrumented_dir(agent_name)

    first_input = cases[0].get("input", cases[0])
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task("  Post-setup smoke test against first dataset case…")
        success, error = smoke_test_agent(
            run_path,
            fn_name,
            first_input,
            env_dir=env_dir,
            agent_dir=inst_root,
        )

    if success:
        console.print(
            "  [bold green]✓[/bold green]  Agent smoke test passed — ready for optimization.\n"
        )
    else:
        console.print(
            Panel(
                "[bold yellow]⚠  Smoke test warning[/bold yellow]\n\n"
                "The agent raised an error on a sample dataset case:\n"
                f"[dim]{error}[/dim]\n\n"
                "The setup spec has been saved. Review the error above before running:\n"
                f"  [bold]overmind optimize {agent_name}[/bold]\n\n"
                "Validate the agent endpoint against the setup dataset (important during "
                "optimization) with:\n"
                f"  [bold]overmind agent validate {agent_name} --data "
                f".overmind/agents/{agent_name}/setup_spec/dataset.json[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
