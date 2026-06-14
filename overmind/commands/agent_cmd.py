"""Overmind agent management commands.

overmind agent register <name> <module:function>
overmind agent list
overmind agent remove <name>
overmind agent update <name> <module:function>
overmind agent show <name>
overmind agent validate <name> --data <path>
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from overmind import SpanType, attrs, set_tag
from overmind.commands.agent_env import instrument_agent_files
from overmind.core.constants import overmind_rel
from overmind.core.paths import (
    agent_experiments_dir,
    agent_setup_spec_dir,
    load_overmind_dotenv,
)
from overmind.core.registry import (
    EntrypointNotFoundError,
    EntrypointSignatureError,
    load_registry,
    remove_agent,
    resolve_entrypoint,
    resolve_entrypoint_file,
    resolve_module_to_file,
    save_agent,
)
from overmind.tracing import observe_safe
from overmind.utils.display import BRAND, confirm_option, rel, select_option


def _sync_agent_id_to_registry(
    name: str,
    entrypoint: str,
    agent_path: str,
    *,
    description: str | None = None,
) -> str | None:
    """Best-effort: upsert a minimal Agent record in the Overmind backend
    and persist the returned UUID into ``.overmind/agents.toml`` so later
    commands (``optimize``, ``setup``, ``optimize-step``) can attribute
    runs to the right Agent record without re-querying the API.

    Returns the UUID string on success, ``None`` on any failure (missing
    API key, project resolution failure, network error). All failures are
    swallowed — the local registry write already succeeded, so a missing
    backend record only affects UI visibility, not local registration.
    """
    try:
        from overmind.storage import StorageNotConfiguredError, configure_storage, get_storage
    except Exception:  # pragma: no cover - defensive import guard
        return None

    try:
        configure_storage(agent_path=agent_path, agent_name=name)
    except Exception:
        return None
    try:
        storage = get_storage()
    except StorageNotConfiguredError:
        return None
    except Exception:
        return None

    minimal_spec: dict = {"agent_description": (description or name)[:512]}
    try:
        storage.save_spec(minimal_spec)
    except Exception:
        return None

    agent_id = storage.get_agent_id() or None
    if not agent_id:
        return None

    try:
        save_agent(name, entrypoint, id=agent_id)
    except Exception:
        return agent_id
    return agent_id


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
        if (exclude_name is None or n != exclude_name) and data.get("entrypoint", "").strip() == ep
    )


def _print_post_register_next_step(console: Console, name: str) -> None:
    """Suggest validating the agent with seed data after registration."""
    console.print(
        f"  Next step: [bold {BRAND}]overmind agent validate {name} "
        f"--data <path/to/seed.json>[/bold {BRAND}] "
        f"[dim](if you have seed data)[/dim]\n"
        f"  [dim]Then connect this machine to the server with [/dim]"
        f"[bold {BRAND}]overmind daemon[/bold {BRAND}][dim] and optimize from the Overmind UI.[/dim]\n"
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
        f"  Each agent name has its own tree under [bold]{overmind_rel('agents')}/[/bold] "
        "(setup_spec, experiments). The same entrypoint can be registered twice with "
        "different names; eval data stays separate per name.\n"
    )
    prompt = (
        "Point this agent at that shared entrypoint anyway?"
        if for_update
        else "Register this entrypoint for another agent name anyway?"
    )
    if not confirm_option(prompt, default=False, console=console):
        console.print("  [dim]Aborted.[/dim]\n")
        raise SystemExit(0)


def _ensure_model_for_wrapper(console: Console) -> None:
    """Make sure an LLM model is available for wrapper generation.

    Assumes overmind dotenv is already loaded.  If no model is configured,
    prompts the user to select one for this session.
    """
    import os

    from overmind.utils.model_picker import prompt_for_catalog_litellm_model
    from overmind.utils.models import DEFAULT_ANALYZER_MODEL

    for env_var in ("ANALYZER_MODEL", "ENV_SETUP_MODEL"):
        if os.environ.get(env_var, "").strip():
            return

    try:
        import litellm  # noqa: F401

        return
    except ImportError:
        pass

    console.print("\n  [dim]Wrapper generation requires an LLM model. Select one to continue.[/dim]")
    model = prompt_for_catalog_litellm_model(
        console,
        select_prompt="  Select model for wrapper generation (number)",
        default_model=DEFAULT_ANALYZER_MODEL,
        no_catalog_prompt="  Enter model identifier",
    )
    os.environ["ANALYZER_MODEL"] = model


def _offer_wrapper_generation(
    name: str,
    exc: EntrypointNotFoundError | EntrypointSignatureError,
    agent_path: str,
    console: Console,
) -> tuple[str, Path, str] | None:
    """Prompt the user to auto-generate an entrypoint wrapper.

    The wrapper is generated inside ``.overmind/agents/<name>/`` — the
    user's original code is never modified.

    Returns ``(new_entrypoint, file_path, fn_name)`` on success, or
    ``None`` if the user declines or generation fails.
    """
    from overmind.entrypoint_wrapper import (
        generate_entrypoint_wrapper,
        wrapper_entrypoint,
    )
    from overmind.utils.display import make_spinner_progress

    agent_dir = Path(agent_path).resolve().parent

    if isinstance(exc, EntrypointSignatureError):
        console.print(
            f"\n  [bold yellow]\u26a0[/bold yellow]  [bold]{exc.fn_name}()[/bold] in "
            f"[cyan]{rel(exc.file_path)}[/cyan] {exc.reason}.\n"
        )
    else:
        console.print(
            f"\n  [bold yellow]\u26a0[/bold yellow]  [bold]{exc.fn_name}()[/bold] is not a callable "
            f"function in [cyan]{rel(exc.file_path)}[/cyan].\n"
        )
    console.print(
        "  Overmind needs a function that takes input and returns output, e.g.:\n"
        "    [dim]def run(input_data: dict) -> dict[/dim]\n"
    )

    choice = select_option(
        [
            "Generate an entrypoint wrapper (Overmind reads your code and creates one)",
            "I'll write the wrapper myself (exit, re-run when ready)",
            "Point me to a different module:function (exit)",
        ],
        title="How would you like to proceed?",
        default_index=0,
        console=console,
    )

    if choice != 0:
        if choice == 1:
            console.print(
                f"\n  Create a wrapper with [bold]def run(input_data: dict) -> dict[/bold] "
                f"and re-register:\n"
                f"    [bold]overmind agent register {name} <module:run>[/bold]\n"
            )
        else:
            console.print(
                f"\n  Re-register with the correct entrypoint:\n"
                f"    [bold]overmind agent register {name} <module:function>[/bold]\n"
            )
        return None

    _ensure_model_for_wrapper(console)

    console.print()
    with make_spinner_progress(console) as progress:
        progress.add_task("  Analyzing agent code and generating wrapper\u2026")
        wp = generate_entrypoint_wrapper(agent_dir, name)

    if wp == "refused":
        console.print(
            "\n  [bold yellow]⚠[/bold yellow]  This agent's code is too complex for an "
            "auto-generated wrapper.\n\n"
            "  The wrapper needs to be a trivial bridge (import + call), but this\n"
            "  agent would require re-implementing agent-specific logic.\n\n"
            "  Add a [bold]def run(input_data: dict) -> dict[/bold] function directly\n"
            "  in your agent code, then re-register:\n"
            f"    [bold]overmind agent register {name} <your_module:run>[/bold]\n"
        )
        return None

    if wp is None or not wp.is_file():
        console.print(
            "\n  [bold red]✗[/bold red]  Wrapper generation failed.\n"
            "  This can happen if no LLM model is configured.\n"
            f"  Set [bold]ANALYZER_MODEL[/bold] in [bold]{overmind_rel('.env')}[/bold] "
            "or write the wrapper manually.\n"
        )
        return None

    wrapper_code = wp.read_text(encoding="utf-8")

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
    if not confirm_option("Register with this entrypoint?", default=True, console=console):
        console.print(f"\n  [dim]Edit [cyan]{rel(wp)}[/cyan] and re-run register.[/dim]\n")
        return None

    new_ep = wrapper_entrypoint(name)
    return new_ep, wp, "run"


def _auto_generate_wrapper(
    name: str,
    agent_path: str,
    console: Console,
) -> tuple[str, Path, str] | None:
    """Generate an entrypoint wrapper when the user provided only a filename.

    Unlike :func:`_offer_wrapper_generation`, this path is taken *before* any
    entrypoint validation — the user has already confirmed they want auto-gen,
    so we skip the "function not found" preamble and go straight to generation.

    Returns ``(new_entrypoint, file_path, fn_name)`` on success, or ``None``
    if the user declines or generation fails.
    """
    from overmind.entrypoint_wrapper import (
        generate_entrypoint_wrapper,
        wrapper_entrypoint,
    )
    from overmind.utils.display import make_spinner_progress

    agent_dir = Path(agent_path).resolve().parent

    _ensure_model_for_wrapper(console)

    console.print()
    with make_spinner_progress(console) as progress:
        progress.add_task("  Analyzing agent code and generating wrapper…")
        wp = generate_entrypoint_wrapper(agent_dir, name)

    if wp == "refused":
        console.print(
            "\n  [bold yellow]⚠[/bold yellow]  This agent's code is too complex for an "
            "auto-generated wrapper.\n\n"
            "  The wrapper needs to be a trivial bridge (import + call), but this\n"
            "  agent would require re-implementing agent-specific logic.\n\n"
            "  Add a [bold]def run(input_data: dict) -> dict[/bold] function directly\n"
            "  in your agent code, then re-register:\n"
            f"    [bold]overmind agent register {name} <your_module:run>[/bold]\n"
        )
        return None

    if wp is None or not wp.is_file():
        console.print(
            "\n  [bold red]✗[/bold red]  Wrapper generation failed.\n"
            "  This can happen if no LLM model is configured.\n"
            f"  Set [bold]ANALYZER_MODEL[/bold] in [bold]{overmind_rel('.env')}[/bold] "
            "or write the wrapper manually.\n"
        )
        return None

    wrapper_code = wp.read_text(encoding="utf-8")

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
    if not confirm_option("Register with this entrypoint?", default=True, console=console):
        console.print(f"\n  [dim]Edit [cyan]{rel(wp)}[/cyan] and re-run register.[/dim]\n")
        return None

    new_ep = wrapper_entrypoint(name)
    return new_ep, wp, "run"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@observe_safe(span_name="overmind.agent.register", type=SpanType.WORKFLOW)
def cmd_register(name: str, entrypoint: str) -> None:
    set_tag(attrs.AGENT_NAME, name)
    set_tag(attrs.AGENT_ENTRYPOINT, entrypoint)
    console = Console()
    load_overmind_dotenv()
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
            f"    [bold]overmind agent update {name} <module:function>[/bold]\n"
        )
        raise SystemExit(1)

    # ---- Handle filename-only input (no entrypoint function specified) ----
    if ":" not in entrypoint:
        file_path = resolve_module_to_file(entrypoint)
        if file_path is None:
            console.print(
                f"\n  [bold red]Error:[/bold red] "
                f"Cannot find file for '[bold]{entrypoint}[/bold]'.\n\n"
                f"  If this is a module path, specify a function too:\n"
                f"    [bold]overmind agent register {name} {entrypoint}:run[/bold]\n"
            )
            raise SystemExit(1)

        console.print(
            f"\n  [bold yellow]No entrypoint function specified.[/bold yellow]\n"
            f"  [dim]File:[/dim] [cyan]{rel(file_path)}[/cyan]\n\n"
            f"  Since no entrypoint was specified, an overmind entrypoint wrapper\n"
            f"  will be generated automatically for this agent.\n"
        )
        if not confirm_option("Generate entrypoint automatically?", default=True, console=console):
            console.print(
                f"\n  [dim]Re-register with an explicit entrypoint:\n"
                f"    [bold]overmind agent register {name} {entrypoint}:run[/bold][/dim]\n"
            )
            raise SystemExit(0)

        agent_path = str(file_path)

        console.print()
        console.print(Rule(style="dim"))
        instrument_agent_files(agent_path, name, console)

        result = _auto_generate_wrapper(name, agent_path, console)
        if result is None:
            raise SystemExit(1)
        entrypoint, file_path, fn = result

        save_agent(name, entrypoint)
        agent_id = _sync_agent_id_to_registry(name, entrypoint, agent_path)
        console.print(
            f"\n  [bold green]\u2713[/bold green]  "
            f"Agent '[bold]{name}[/bold]' registered.\n"
            f"  [dim]Entrypoint:[/dim] {entrypoint}\n"
            f"  [dim]File:[/dim]      {file_path}\n"
            f"  [dim]Function:[/dim]  {fn}\n" + (f"  [dim]Agent ID:[/dim]  {agent_id}\n" if agent_id else "")
        )
        if agent_id:
            set_tag(attrs.AGENT_ID, agent_id)
        _print_post_register_next_step(console, name)
        return

    dupes = _other_agents_with_entrypoint(registry, entrypoint)
    if dupes:
        _confirm_duplicate_entrypoint(console, entrypoint, dupes, for_update=False)

    # ---- Resolve module path to file (quick, no function check yet) ----
    try:
        file_path, fn = resolve_entrypoint_file(entrypoint)
        agent_path = str(file_path)
    except ValueError as exc:
        console.print(f"\n  [bold red]Error:[/bold red] {exc}\n")
        raise SystemExit(1) from exc

    # ---- 1. Copy agent source into .overmind/ (instrumentation) ----
    console.print()
    console.print(Rule(style="dim"))
    instrument_agent_files(agent_path, name, console)

    # ---- 2. Validate entrypoint function (may trigger wrapper generation) ----
    try:
        resolve_entrypoint(entrypoint)
    except (EntrypointNotFoundError, EntrypointSignatureError) as exc:
        result = _offer_wrapper_generation(name, exc, agent_path, console)
        if result is None:
            raise SystemExit(1) from exc
        entrypoint = result[0]
        file_path = result[1]
        fn = result[2]

    # ---- 3. Save to registry ----
    save_agent(name, entrypoint)
    agent_id = _sync_agent_id_to_registry(name, entrypoint, agent_path)

    # ``file_path`` is usually a Path; stringify defensively so the
    # span attribute carries a plain string regardless of where the
    # caller resolved it from.
    set_tag(attrs.AGENT_FILE_PATH, str(file_path))
    set_tag(attrs.AGENT_FUNCTION_NAME, fn)
    if agent_id:
        set_tag(attrs.AGENT_ID, agent_id)

    console.print(
        f"\n  [bold green]\u2713[/bold green]  "
        f"Agent '[bold]{name}[/bold]' registered.\n"
        f"  [dim]Entrypoint:[/dim] {entrypoint}\n"
        f"  [dim]File:[/dim]      {file_path}\n"
        f"  [dim]Function:[/dim]  {fn}\n" + (f"  [dim]Agent ID:[/dim]  {agent_id}\n" if agent_id else "")
    )
    _print_post_register_next_step(console, name)


@observe_safe(span_name="overmind.agent.list", type=SpanType.WORKFLOW)
def cmd_list() -> None:
    console = Console()
    registry = load_registry()
    set_tag(attrs.AGENT_REGISTERED_COUNT, len(registry))

    if not registry:
        console.print(
            "\n  [dim]No agents registered yet.[/dim]\n\n"
            "  Register one:\n"
            "    [bold]overmind agent register <name> <module:function>[/bold]\n"
        )
        return

    table = Table(border_style="cyan", show_header=True, show_lines=False)
    table.add_column("NAME", style=f"bold {BRAND}")
    table.add_column("ENTRYPOINT")
    table.add_column("FILE", justify="center")

    for name, data in registry.items():
        file_ok = "[green]\u2713[/green]" if Path(data["file_path"]).exists() else "[red]\u2717[/red]"
        table.add_row(name, data["entrypoint"], file_ok)

    console.print()
    console.print(table)
    console.print()


@observe_safe(span_name="overmind.agent.pull", type=SpanType.WORKFLOW)
def cmd_pull() -> None:
    """Write ``.overmind/agents.toml`` from the agents extracted on the server."""
    from overmind.client import get_client, sync_registry_from_server

    console = Console()
    load_overmind_dotenv()

    client = get_client()
    if client is None:
        console.print(
            "\n  [bold red]Error:[/bold red] Overmind API is not configured.\n\n"
            "  Set your project API key first:\n"
            f"    [dim]{overmind_rel('.env')}[/dim] → [bold]OVERMIND_API_KEY=ovr_...[/bold]\n"
            "  (and [bold]OVERMIND_API_URL[/bold] for a self-hosted backend).\n"
        )
        raise SystemExit(1)

    try:
        entries = sync_registry_from_server(client)
    except Exception as exc:
        console.print(f"\n  [bold red]Error:[/bold red] Could not fetch agents: {exc}\n")
        raise SystemExit(1) from exc

    set_tag(attrs.AGENT_REGISTERED_COUNT, len(entries))

    if not entries:
        console.print(
            "\n  [dim]No agents with a runnable entrypoint were found on the server.[/dim]\n\n"
            "  Connect and analyze a repository in the Overmind UI first, then\n"
            "  run [bold]overmind agent pull[/bold] again.\n"
        )
        return

    table = Table(border_style="cyan", show_header=True, show_lines=False)
    table.add_column("NAME", style=f"bold {BRAND}")
    table.add_column("ENTRYPOINT")
    table.add_column("FILE", justify="center")

    for entry in sorted(entries, key=lambda e: e["name"].lower()):
        try:
            file_path, _ = resolve_entrypoint_file(entry["entrypoint"])
            file_ok = "[green]\u2713[/green]" if file_path.exists() else "[red]\u2717[/red]"
        except Exception:
            file_ok = "[red]\u2717[/red]"
        table.add_row(entry["name"], entry["entrypoint"], file_ok)

    console.print(
        f"\n  [bold green]\u2713[/bold green]  Wrote {len(entries)} agent(s) to "
        f"[dim]{overmind_rel('agents.toml')}[/dim]"
    )
    console.print(table)
    console.print(
        "\n  [dim]A [red]\u2717[/red] FILE means the entrypoint isn't present in this checkout.[/dim]\n"
        f"  [dim]Connect this machine to the server:[/dim] [bold {BRAND}]overmind daemon[/bold {BRAND}]\n"
    )


@observe_safe(span_name="overmind.agent.remove", type=SpanType.WORKFLOW)
def cmd_remove(name: str) -> None:
    set_tag(attrs.AGENT_NAME, name)
    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            "  To see all registered agents:\n"
            "    [bold]overmind agent list[/bold]\n"
        )
        raise SystemExit(1)

    set_tag(attrs.AGENT_ENTRYPOINT, registry[name]["entrypoint"])

    console.print(f"\n  Agent '[bold]{name}[/bold]'  [dim]{registry[name]['entrypoint']}[/dim]")
    if not confirm_option(f"Remove '{name}' from the registry?", default=True, console=console):
        console.print("  [dim]Aborted.[/dim]\n")
        raise SystemExit(0)

    remove_agent(name)

    from overmind.core.paths import agent_instrumented_dir

    inst_dir = agent_instrumented_dir(name)
    if inst_dir.exists():
        import shutil

        shutil.rmtree(inst_dir)
        console.print(f"  [dim]Removed instrumented copy at {rel(inst_dir)}[/dim]")

    set_tag(attrs.AGENT_REMOVED, "true")
    console.print(f"\n  [bold green]\u2713[/bold green]  Agent '[bold]{name}[/bold]' removed.\n")


@observe_safe(span_name="overmind.agent.update", type=SpanType.WORKFLOW)
def cmd_update(name: str, entrypoint: str) -> None:
    set_tag(attrs.COMMAND, "agent.update")
    set_tag(attrs.AGENT_NEW_ENTRYPOINT, entrypoint)
    console = Console()
    load_overmind_dotenv()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            f"  Use register instead:\n"
            f"    [bold]overmind agent register {name} {entrypoint}[/bold]\n"
        )
        raise SystemExit(1)

    old_ep_raw = registry[name]["entrypoint"]
    set_tag(attrs.AGENT_OLD_ENTRYPOINT, old_ep_raw)

    if old_ep_raw.strip() == entrypoint.strip():
        raise SystemExit(0)

    dupes = _other_agents_with_entrypoint(registry, entrypoint, exclude_name=name)
    if dupes:
        _confirm_duplicate_entrypoint(console, entrypoint, dupes, for_update=True)

    try:
        file_path, _ = resolve_entrypoint_file(entrypoint)
        agent_path = str(file_path)
    except ValueError as exc:
        console.print(f"\n  [bold red]Error:[/bold red] {exc}\n")
        raise SystemExit(1) from exc

    # 1. Re-instrument
    console.print()
    console.print(Rule(style="dim"))
    instrument_agent_files(agent_path, name, console)

    # 2. Validate entrypoint function (may trigger wrapper generation)
    try:
        resolve_entrypoint(entrypoint)
    except (EntrypointNotFoundError, EntrypointSignatureError) as exc:
        result = _offer_wrapper_generation(name, exc, agent_path, console)
        if result is None:
            raise SystemExit(1) from exc
        entrypoint = result[0]

    # 3. Save
    save_agent(name, entrypoint)

    console.print(f"\n  [dim]Old entrypoint:[/dim] {old_ep_raw}\n  [dim]New entrypoint:[/dim] {entrypoint}\n")


@observe_safe(span_name="overmind.agent.show", type=SpanType.WORKFLOW)
def cmd_show(name: str) -> None:
    set_tag(attrs.AGENT_NAME, name)
    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            "  To see all registered agents:\n"
            "    [bold]overmind agent list[/bold]\n"
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

    file_status = "[green]\u2713 exists[/green]" if file_exists else "[red]\u2717 not found[/red]"

    set_tag(attrs.AGENT_ENTRYPOINT, data["entrypoint"])
    set_tag(attrs.AGENT_FILE_PATH, data["file_path"])
    set_tag(attrs.AGENT_FILE_EXISTS, bool(file_exists))
    set_tag(attrs.AGENT_SETUP_SPEC_READY, bool(spec_exists))
    set_tag(attrs.AGENT_EXPERIMENT_FILE_COUNT, len(exp_files))

    spec_status = "[green]\u2713 ready[/green]" if spec_exists else "[yellow]not run yet[/yellow]"
    exp_status = f"[green]\u2713 {len(exp_files)} file(s)[/green]" if exp_files else "[yellow]not run yet[/yellow]"

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


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@observe_safe(span_name="overmind.agent.validate", type=SpanType.WORKFLOW)
def cmd_validate(name: str, data: str) -> None:
    """Run the agent's entrypoint against test data to verify it works."""
    import json

    from overmind.core.paths import agent_instrumented_dir
    from overmind.core.registry import project_root_from_agent_file, resolve_agent
    from overmind.optimize.data import check_consistent_fields, load_data, normalize_data_fields
    from overmind.optimize.runner import AgentRunner, RunnerConfig
    from overmind.utils.display import make_spinner_progress

    console = Console()
    load_overmind_dotenv()
    set_tag(attrs.AGENT_NAME, name)

    registry = load_registry()
    if name not in registry:
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is not registered.\n\n"
            "  To see all registered agents:\n"
            "    [bold]overmind agent list[/bold]\n"
        )
        raise SystemExit(1)

    agent_path, fn_name = resolve_agent(name)

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"\n  [bold red]Error:[/bold red] Data path not found: [cyan]{data}[/cyan]\n")
        raise SystemExit(1)

    json_files: list[Path] = []
    if data_path.is_dir():
        # Accept JSON, JSONL, and NDJSON datasets — exporters frequently
        # emit one JSON object per line for streaming reasons; rejecting
        # them here would force users to convert files just to validate.
        discovered: set[Path] = set()
        for pattern in ("*.json", "*.jsonl", "*.ndjson"):
            discovered.update(data_path.glob(pattern))
        json_files = sorted(discovered)
        if not json_files:
            console.print(
                f"\n  [bold red]Error:[/bold red] No .json/.jsonl/.ndjson files found in [cyan]{data}[/cyan]\n"
            )
            raise SystemExit(1)
    else:
        json_files = [data_path]

    cases: list[dict] = []
    for jf in json_files:
        try:
            cases.extend(load_data(str(jf)))
        except Exception as exc:
            console.print(f"\n  [bold red]Error:[/bold red] Could not load [cyan]{jf}[/cyan]: {exc}\n")
            raise SystemExit(1)

    if not cases:
        console.print(f"\n  [bold yellow]Warning:[/bold yellow] No test cases found in [cyan]{data}[/cyan]\n")
        raise SystemExit(1)

    # ── Field-consistency check ───────────────────────────────────────────
    consistent, common_fields, bad_indices = check_consistent_fields(cases)
    if not consistent:
        console.print(
            f"\n  [bold red]Error:[/bold red] Not all data points have the same fields.\n"
            f"  First case fields: {sorted(common_fields)}\n"
            f"  Mismatched at indices: {bad_indices[:10]}"
            + ("  …" if len(bad_indices) > 10 else "")
            + "\n  Please ensure every entry in your data file has identical top-level keys.\n"
        )
        raise SystemExit(1)

    # ── Field mapping (prompt if data lacks standard 'input' key) ────────
    cases = normalize_data_fields(cases, console, require_output=False, agent_name=name)

    first_case = cases[0]
    test_input = first_case.get("input", first_case)
    label = json.dumps(test_input, default=str)
    if len(label) > 120:
        label = label[:117] + "..."

    console.print(
        f"\n  Agent:      [bold]{name}[/bold]\n"
        f"  Entrypoint: [dim]{registry[name]['entrypoint']}[/dim]\n"
        f"  Data:       [cyan]{data}[/cyan]  ({len(cases)} case(s), running first)\n"
        f"  Input:      [dim]{label}[/dim]\n"
    )

    p = Path(agent_path).resolve()
    inst_root = agent_instrumented_dir(name)
    if inst_root.exists() and str(p).startswith(str(inst_root)):
        resolved_agent_dir = inst_root
        env_dir_path: Path | None = project_root_from_agent_file(agent_path) or p.parent
    else:
        pr = project_root_from_agent_file(agent_path)
        resolved_agent_dir = pr if pr is not None else p.parent
        env_dir_path = None

    entry_file = str(p.relative_to(resolved_agent_dir))
    runner = AgentRunner(
        agent_dir=resolved_agent_dir,
        entry_file=entry_file,
        entrypoint_fn=fn_name,
        config=RunnerConfig(timeout=300),
        env_dir=env_dir_path,
    )

    console.print(Rule(style="dim"))
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task("  Setting up agent environment…")
        runner.ensure_environment()

    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task("  Running agent…")
        try:
            result = runner.run(test_input)
            error = result.error if not result.success else ""
        except Exception as exc:
            result = None
            error = str(exc)

    runner.cleanup()

    console.print()
    console.print(Rule(style="dim"))
    if error:
        console.print(f"\n  [bold red]✗[/bold red]  Validation failed.\n      [dim red]{error}[/dim red]\n")
        raise SystemExit(1)
    else:
        output_str = json.dumps(result.data, indent=2, default=str)
        if len(output_str) > 500:
            output_str = output_str[:497] + "..."
        console.print(f"\n  [bold green]✓[/bold green]  Validation passed.\n  [dim]Output:[/dim]\n  {output_str}\n")
