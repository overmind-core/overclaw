"""
Overmind setup — Agent Setup

Analyzes your agent's code, tools, and orchestration to propose evaluation
criteria. Lets you accept them or iteratively refine through conversation.

The setup flow:
  Phase 1: Agent Analysis  — examine code, tools, schemas
  Phase 2: Policy          — define domain rules and constraints
  Phase 3: Dataset         — generate/analyze test data (after policy, before criteria)
  Phase 4: Eval Criteria   — propose and refine scoring rules

Usage:
    overmind setup <agent-name>
    overmind setup <agent-name> --data path/to/seed.json
    overmind setup <agent-name> --data path/to/json_dir/
    overmind setup <agent-name> --fast
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.rule import Rule

from overmind import SpanType, attrs, set_tag
from overmind.client import (
    ProjectResolutionError,
    flush_pending_api_updates,
    get_client,
    get_project_id,
    resolve_project_id,
    upsert_agent,
)
from overmind.core.constants import overmind_rel
from overmind.core.paths import (
    agent_env_path,
    agent_instrumented_dir,
    agent_setup_spec_dir,
    load_agent_dotenv,
    load_overmind_dotenv,
)
from overmind.core.registry import (
    get_agent_id,
    load_registry,
    project_root_from_agent_file,
    resolve_agent,
    save_agent,
)
from overmind.optimize.data import (
    check_consistent_fields,
    generate_diverse_synthetic_data,
    generate_synthetic_data,
    load_data,
    normalize_data_fields,
)
from overmind.optimize.data_analyzer import analyze_seed_coverage, validate_seed_data
from overmind.optimize.evaluator import has_entrypoint
from overmind.setup.agent_analyzer import analyze_agent
from overmind.setup.policy_generator import (
    display_policy,
    elicit_policy,
    generate_policy_from_code,
    improve_existing_policy,
    refine_policy,
    save_policy,
)
from overmind.setup.questionnaire import run_questionnaire
from overmind.setup.spec_generator import generate_spec_from_proposal, save_spec
from overmind.storage import configure_storage, get_storage
from overmind.tracing import observe_safe
from overmind.utils.display import (
    BRAND,
    confirm_option,
    make_spinner_progress,
    rel,
    render_logo,
    select_option,
)
from overmind.utils.model_picker import prompt_for_catalog_litellm_model
from overmind.utils.models import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_DATAGEN_MODEL,
    normalize_to_litellm_model_id,
)
from overmind.utils.policy import default_policy_path, format_for_synthetic_data
from overmind.utils.provider_keys import (
    PROVIDER_ENV_KEYS as _PROVIDER_ENV_KEYS,
)
from overmind.utils.provider_keys import (
    ensure_provider_api_keys as _ensure_provider_api_keys,
)
from overmind.utils.provider_keys import (
    update_agent_env as _update_agent_env,
)

logger = logging.getLogger("overmind.commands.setup")


def _check_agent_dependencies(
    agent_path: str,
    agent_name: str,
    console: Console,
    *,
    fast: bool = False,
    instrumented_dir: Path | None = None,
) -> None:
    """Detect external imports without a dependency manifest and guide the user.

    When *instrumented_dir* is provided the dependency check and any generated
    manifest are placed inside the instrumented copy so the original agent
    source is never modified.  The manifest is also written to the
    instrumented **root** (the top of the copied tree) so the runner's
    ``ensure_environment`` finds it when provisioning the sandbox.

    In interactive mode: offers to generate a requirements.txt / package.json
    or lets the user handle it themselves.  In fast mode: fails with a clear
    message.
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


def _validate_agent_entrypoint(
    agent_path: str,
    fn_name: str,
    agent_name: str,
    console: Console,
    *,
    fast: bool = False,
) -> tuple[str, str]:
    """Verify the agent file defines the registered entry function.

    Returns ``(agent_path, fn_name)`` — unchanged when valid, or
    updated to point at a generated wrapper when the user opts in.
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

    # --- Entrypoint not found — offer wrapper generation ---
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


def _clear_existing_eval_spec(agent_name: str, console: Console, *, fast: bool = False) -> None:
    with suppress(Exception):
        storage = get_storage()
        if fast:
            storage.clear_setup_spec()
            console.print("  [dim]Cleared setup artifacts in Overmind (fast mode).[/dim]")
            return
        if confirm_option(
            "Delete existing setup artifacts in Overmind and start fresh?",
            default=True,
            console=console,
        ):
            storage.clear_setup_spec()
            console.print("  [dim]Cleared Overmind setup artifacts.[/dim]")
        else:
            console.print("  [dim]Keeping existing Overmind setup artifacts.[/dim]")
        return

    spec_dir = agent_setup_spec_dir(agent_name)
    if not spec_dir.exists():
        return

    existing = [f for f in spec_dir.iterdir() if f.name != ".gitkeep"]
    if not existing:
        return

    console.print(f"\n  [yellow]Found {len(existing)} existing file(s) in setup_spec/[/yellow]")

    if fast:
        shutil.rmtree(spec_dir)
        spec_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared (fast mode).[/dim]")
        return

    if confirm_option(
        "Delete existing setup spec files and start fresh?",
        default=True,
        console=console,
    ):
        shutil.rmtree(spec_dir)
        spec_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared.[/dim]")
    else:
        console.print("  [dim]Keeping existing files. New spec will overwrite setup_spec/eval_spec.json.[/dim]")


def _instrument_agent_files(agent_path: str, agent_name: str, console: Console) -> tuple[str, Path]:
    """Copy agent source into .overmind/ — delegates to shared module."""
    from overmind.commands.agent_env import instrument_agent_files

    return instrument_agent_files(agent_path, agent_name, console)


def _save_and_finish(
    spec: dict,
    agent_name: str,
    console: Console,
    policy_md: str | None = None,
    *,
    policy_file_already_saved: bool = False,
):
    spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    save_spec(spec, str(spec_path))
    if policy_md and not policy_file_already_saved:
        save_policy(policy_md, default_policy_path(agent_name))

    storage = None
    with suppress(Exception):
        storage = get_storage()
    if storage is not None:
        storage.save_spec(spec)
        if policy_md:
            storage.save_policy(policy_md, spec.get("policy"))

    n_fields = len(spec.get("output_fields", {}))
    has_tools = bool(spec.get("tool_config", {}).get("expected_tools"))
    has_consistency = bool(spec.get("consistency_rules"))
    has_judge = spec.get("llm_judge_weight", 0) > 0
    has_policy = bool(spec.get("policy"))

    set_tag(attrs.SETUP_EVAL_SPEC_FIELD_COUNT, n_fields)
    set_tag(attrs.SETUP_EVAL_SPEC_HAS_TOOLS, has_tools)
    set_tag(attrs.SETUP_EVAL_SPEC_HAS_JUDGE, has_judge)
    set_tag(attrs.SETUP_EVAL_SPEC_HAS_POLICY, has_policy)
    set_tag(attrs.SETUP_EVAL_SPEC_STRUCTURE_WEIGHT, spec.get("structure_weight", 0))
    if has_tools:
        set_tag(attrs.SETUP_EVAL_SPEC_TOOL_COUNT, len(spec["tool_config"]["expected_tools"]))
    if has_consistency:
        set_tag(attrs.SETUP_EVAL_SPEC_CONSISTENCY_RULE_COUNT, len(spec["consistency_rules"]))

    # Emit the full eval spec so the OTLP ingest pipeline can populate
    # input_schema, output_fields, tool_config etc. on the Agent record.
    try:
        set_tag(attrs.SETUP_EVAL_SPEC, json.dumps(spec, separators=(",", ":"), default=str))
    except Exception:  # noqa: S110
        pass

    features: list[str] = [f"{n_fields} output field(s)"]
    if has_tools:
        n_tools = len(spec["tool_config"]["expected_tools"])
        features.append(f"{n_tools} tool(s) monitored")
    if has_consistency:
        features.append(f"{len(spec['consistency_rules'])} consistency rule(s)")
    if has_judge:
        features.append("LLM-as-Judge enabled")
    if has_policy:
        n_rules = len(spec["policy"].get("domain_rules", spec["policy"].get("decision_rules", [])))
        features.append(f"policy ({n_rules} rule(s))")

    console.print(f"\n  [bold green]\u2713[/bold green] Spec saved  [dim]→ {rel(spec_path)}[/dim]")
    if policy_md and not policy_file_already_saved:
        pol_path = default_policy_path(agent_name)
        console.print(f"  [bold green]\u2713[/bold green] Policy saved  [dim]→ {rel(pol_path)}[/dim]")
    if storage is not None:
        console.print("  [dim]Queued sync to Overmind backend.[/dim]")
    console.print(f"  [dim]Spec covers: {', '.join(features)}[/dim]")
    next_cmd = agent_name
    console.print(f"\n  Next step: [bold {BRAND}]overmind optimize {next_cmd}[/bold {BRAND}]\n")


def _smoke_test_agent(
    agent_path: str,
    fn_name: str,
    input_case: dict,
    env_dir: str | Path | None = None,
    agent_dir: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Run the agent via subprocess and call fn_name(input_case) once.

    Returns (True, None) on success or (False, error_message) on any exception.
    Uses the AgentRunner for full dependency isolation.

    *agent_dir* overrides the working directory for the subprocess.  When
    running the instrumented copy, pass the instrumented root so local
    imports resolve correctly.  When ``None``, the project root is detected
    via ``project_root_from_agent_file`` (falls back to the entry file's
    parent directory).

    *env_dir* should point to the **original** project root so dependency
    manifests, ``.venv``, and ``.env`` are found.
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
        logger.warning(f"smoke_test: agent={agent_path} failed rc={result.returncode} err={(result.error or '')[:300]}")
        return False, "\n".join(parts) or "Unknown error"

    except Exception as exc:
        logger.exception(f"smoke_test: exception for agent={agent_path}")
        return False, str(exc)


def _resolve_seed_json_files(data_arg: str | None, *, console: Console) -> list[Path]:
    """Resolve ``--data`` to a list of JSON seed files (single file or ``*.json`` in a directory)."""
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


def _prompt_seed_data_flag_early(agent_name: str, *, console: Console) -> None:
    """Explain seed data, ask about ``--data``; if yes, print the command and exit setup."""
    console.print(
        "  [dim]It looks like no seed data was provided ([bold]--data[/bold] was not set). "
        "Seed JSON (real or representative inputs) lets Overmind smoke-test your agent early, "
        "shape the evaluation dataset around your domain, and validate or augment cases "
        "against your policy. Without it, setup relies on synthetic data only (or you can skip "
        "dataset steps later), which may not match your true payloads or edge cases.[/dim]"
    )
    console.print()
    if not confirm_option(
        "Do you want to provide seed data using --data?",
        default=False,
        console=console,
    ):
        return
    quoted = shlex.quote(agent_name)
    console.print()
    console.print(
        "  [dim]Re-run setup with [bold]--data[/bold] pointing at a JSON file or a directory "
        "of [bold]*.json[/bold] files:[/dim]"
    )
    console.print(f"    [bold cyan]overmind setup {quoted} --data path/to/cases.json[/bold cyan]")
    console.print(f"    [bold cyan]overmind setup {quoted} --data path/to/dataset_folder/[/bold cyan]")
    console.print()
    console.print("  [dim]Exiting setup — run the command above when your seed files are ready.[/dim]\n")
    raise SystemExit(0)


def _run_beginning_smoke_test(
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
    existing_json = _resolve_seed_json_files(data_path, console=console)

    if not existing_json:
        console.print(
            "  [dim]Skipping pre-setup smoke test with seed data "
            "(pass [bold]--data[/bold] with a JSON file or directory of JSON files).[/dim]"
        )
        return

    console.print(f"  [dim]Using seed data from [cyan]{rel(existing_json[0])}[/cyan] for smoke test…[/dim]")

    try:
        cases = load_data(str(existing_json[0]))
    except Exception:
        console.print(f"  [dim]Could not read [cyan]{existing_json[0].name}[/cyan] — skipping smoke test.[/dim]")
        return

    if not cases:
        console.print(f"  [dim][cyan]{existing_json[0].name}[/cyan] is empty — skipping pre-setup smoke test.[/dim]")
        return

    # ── Field-consistency check ───────────────────────────────────────────
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

    # ── Field mapping (prompt once, then persisted for validate / optimize) ─
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
        progress.add_task(f"  Smoke-testing agent using {existing_json[0].name} ({len(cases)} case(s))…")
        success, error = _smoke_test_agent(
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


def _run_end_smoke_test(
    agent_name: str,
    agent_path: str,
    fn_name: str,
    console: Console,
    instrumented_entry: str | None = None,
) -> None:
    """Validate the agent runs against the first generated dataset case.

    When *instrumented_entry* is provided the smoke test executes against the
    instrumented copy (matching what the optimizer will run) with the
    original project root as ``env_dir``.

    Issues a warning panel on failure but does NOT abort — the spec is already
    saved and the user should be informed rather than left with a silent problem.
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
        success, error = _smoke_test_agent(
            run_path,
            fn_name,
            first_input,
            env_dir=env_dir,
            agent_dir=inst_root,
        )

    if success:
        console.print("  [bold green]✓[/bold green]  Agent smoke test passed — ready for optimization.\n")
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


def _data_dir(agent_path: str) -> Path:
    """Historical default sibling ``data/`` directory (no longer used unless you pass ``--data``)."""
    return Path(agent_path).resolve().parent / "data"


def _build_eval_spec_stub(
    analysis: dict,
    policy_data: dict | None = None,
    entrypoint_fn: str = "",
) -> dict:
    """Build a minimal eval-spec-like dict from analysis for schema validation.

    At setup time the real eval spec doesn't exist yet, but the data
    generation functions need ``input_schema`` and ``output_fields`` for
    validation.  The analysis dict has ``output_schema`` which uses the
    same per-field shape (type, values, range, description).

    ``entrypoint_fn`` is carried through so that the data-generation
    prompts can reference the function name when explaining the input
    schema contract (the runner dispatches via ``**kwargs``).
    """
    output_schema = analysis.get("output_schema", {})
    output_fields: dict = {}
    for field, info in output_schema.items():
        entry = dict(info)
        entry.setdefault("weight", 10)
        entry.setdefault("importance", "important")
        output_fields[field] = entry

    stub: dict = {
        "agent_description": analysis.get("description", ""),
        "input_schema": analysis.get("input_schema", {}),
        "output_fields": output_fields,
    }
    if entrypoint_fn:
        stub["entrypoint_fn"] = entrypoint_fn
    if policy_data:
        stub["policy"] = policy_data
    return stub


def _resolve_datagen_model(console: Console, *, fast: bool = False) -> str:
    """Resolve the synthetic-data generation model."""
    raw = os.getenv("SYNTHETIC_DATAGEN_MODEL", "").strip()
    if raw:
        resolved = normalize_to_litellm_model_id(raw) or raw
        if fast:
            return resolved
        if confirm_option(
            f"Use {resolved} from {overmind_rel('.env')} for data generation?",
            default=True,
            console=console,
        ):
            return resolved

    if fast:
        console.print("\n[red]Fast mode requires SYNTHETIC_DATAGEN_MODEL in the environment.[/red]")
        raise SystemExit(1)

    if not raw:
        console.print(
            "\n  [dim]Setup uses an LLM to work with your test data: it reviews coverage "
            "against your policy and eval sketch, and can generate additional synthetic "
            "cases that look like real inputs for your agent. That requires a model with "
            "API access (same idea as codegen or chat — the model proposes structured "
            "examples, not random JSON).[/dim]"
        )
        console.print(
            f"\n  [dim]No default yet: [cyan]SYNTHETIC_DATAGEN_MODEL[/cyan] is not set in "
            f"{overmind_rel('.env')}. Pick a provider and model below; we’ll remember it "
            "for the next setup or optimize run.[/dim]"
        )
    else:
        console.print(
            "\n  [dim]Choose an LLM for synthetic test-data work (coverage analysis and "
            "any generated cases are drafted to match your agent and policy).[/dim]"
        )
    return prompt_for_catalog_litellm_model(
        console,
        select_prompt="  Select model for data generation (number)",
        env_default=None,
        default_model=DEFAULT_DATAGEN_MODEL,
        no_catalog_prompt="  Enter model for data generation (provider/model)",
    )


_OUTPUT_FORMAT_OPTIONS = [
    "JSON object (structured key/value response)",
    "Plain text / string (free-form prose, summary, etc.)",
    "Markdown (formatted text with headings, lists, etc.)",
    "List of items (JSON array or bullet points)",
    "Other (describe below)",
]


def _prompt_expected_output_format(console: Console) -> str:
    """Ask the user what output format their agent produces.

    Returns a short description string suitable for injection into the
    synthetic-data generation prompt.
    """
    console.print()
    console.print(
        "  [dim]Without seed data, Overmind needs to know the shape of your agent's "
        "output so the synthetic dataset has the right structure.[/dim]"
    )

    idx = select_option(
        _OUTPUT_FORMAT_OPTIONS,
        title="What format does your agent return?",
        default_index=0,
        console=console,
    )
    choice = _OUTPUT_FORMAT_OPTIONS[idx]

    if idx == len(_OUTPUT_FORMAT_OPTIONS) - 1:
        custom = Prompt.ask(
            "  Describe the expected output format",
            console=console,
        ).strip()
        if custom:
            return custom
        return "unstructured text"

    return choice.split("(")[0].strip()


def _save_dataset(
    cases: list[dict],
    agent_name: str,
    console: Console,
    *,
    source: str = "synthetic",
    generator_model: str = "",
    policy_md: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Persist the final dataset.

    Always writes ``setup_spec/dataset.json`` locally. When an API
    backend is configured, also POSTs a new versioned :class:`Dataset`
    to Overmind via :meth:`OvermindClient.upsert_dataset` — datasets
    are persisted explicitly through the REST API, not derived from
    span attributes by the OTLP ingest path.
    """
    agent = get_client().get_agent(agent_name)
    data_path = agent_setup_spec_dir(agent_name) / "dataset.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(cases, f, indent=2)

    storage = None
    with suppress(Exception):
        storage = get_storage()

    dataset_meta: dict | None = None
    if storage is not None:
        # If the caller didn't supply policy_md but a local policies.md exists,
        # hash it so the dataset record is linked to the policy it was drafted
        # against.  Avoids pushing every dataset call site to re-load the file.
        resolved_policy_md = policy_md
        if resolved_policy_md is None:
            policy_path = Path(default_policy_path(agent_name))
            if policy_path.exists():
                with suppress(Exception):
                    resolved_policy_md = policy_path.read_text(encoding="utf-8")
        policy_hash = ""
        if resolved_policy_md:
            policy_hash = hashlib.sha256(resolved_policy_md.encode("utf-8")).hexdigest()[:64]
        storage._agent_id = str(agent.id)

        dataset_meta = storage.save_dataset(
            cases,
            source=source,
            generator_model=generator_model,
            policy_hash=policy_hash,
            metadata=metadata or {},
        )

    dataset_id: str | None = dataset_meta.get("id") if dataset_meta else None

    console.print(
        f"\n  [bold {BRAND}]✓[/bold {BRAND}]  Saved [bold]{len(cases)}[/bold] cases  [dim]→ {rel(data_path)}[/dim]"
    )
    if storage is not None:
        if dataset_id:
            version_str = f"v{dataset_meta['version']}  " if dataset_meta and dataset_meta.get("version") else ""
            console.print(f"  [dim]Dataset {version_str}uploaded to Overmind (id: {dataset_id}).[/dim]")
        else:
            console.print("  [dim]Dataset upload to Overmind skipped or failed — local copy kept.[/dim]")
    return str(data_path)


def _ensure_remote_agent_id(
    agent_name: str,
    agent_path: str,
    console: Console,
    spec: dict | None = None,
) -> str | None:
    """Ensure a remote Overmind agent exists; return its id when available."""
    existing_id = get_agent_id(agent_name)
    client = get_client()
    project_id: str | None
    try:
        project_id = resolve_project_id(client) if client else None
    except ProjectResolutionError as exc:
        console.print(f"  [yellow]Warning:[/yellow] {exc}")
        project_id = None
    if existing_id:
        # Verify stored id belongs to the currently configured project.
        # This avoids silently writing to another project's similarly-slugged agent.
        if client and project_id:
            with suppress(Exception):
                existing = client.agents_retrieve(id=UUID(existing_id))
                existing_project = str(getattr(existing, "project", "") or "")
                if existing_project == str(project_id):
                    return existing_id
                console.print(
                    "  [yellow]Stored agent id belongs to a different project; "
                    "creating a project-local agent id.[/yellow]"
                )
        else:
            return existing_id

    if not client or not project_id:
        return None

    console.print("  [dim]No remote id found. Creating agent in Overmind...[/dim]")
    try:
        minimal_spec = {
            "agent_description": f"{agent_name} agent",
            "agent_path": agent_path,
            "input_schema": {},
            "output_fields": {},
            "structure_weight": 20,
            "total_points": 100,
        }
        create_spec = spec if isinstance(spec, dict) and spec else minimal_spec
        result = upsert_agent(
            client,
            project_id=project_id,
            agent_path=agent_path,
            spec=create_spec,
            agent_name=agent_name,
        )
        new_id = str(result.id)
        entrypoint = (load_registry().get(agent_name, {}) or {}).get("entrypoint")
        if entrypoint:
            save_agent(agent_name, entrypoint, id=new_id)
        console.print("  [dim]Remote agent created and id stored in agents.toml.[/dim]")
        return new_id
    except Exception as exc:
        # Retry once with a minimal payload in case local artifacts contain
        # fields rejected by the backend's current schema/version.
        if spec:
            with suppress(Exception):
                result = upsert_agent(
                    client,
                    project_id=project_id,
                    agent_path=agent_path,
                    spec=minimal_spec,
                    agent_name=agent_name,
                )
                new_id = str(result.id)
                entrypoint = (load_registry().get(agent_name, {}) or {}).get("entrypoint")
                if entrypoint:
                    save_agent(agent_name, entrypoint, id=new_id)
                console.print("  [dim]Remote agent created and id stored in agents.toml.[/dim]")
                return new_id
        console.print(f"  [yellow]Warning:[/yellow] Could not create agent in Overmind. [dim]({exc})[/dim]")
        return None


def _sync_setup_artifacts(agent_name: str, agent_path: str, console: Console) -> None:
    """Upload local setup artifacts to Overmind backend if configured.

    Gating rules:
      * No client (no ``OVERMIND_API_KEY``) → silently skip.
      * Client + env-set project id → proceed (legacy path).
      * Client + project-scoped ``ovr_…`` key → proceed; the project id is
        resolved lazily inside :func:`_ensure_remote_agent_id`.
      * Client + user JWT *and* no env project id → skip (we can't pick a
        project unambiguously and the caller never opted-in via env).
    """
    client = get_client()
    if not client:
        return
    if not get_project_id():
        # Allow project-scoped keys to proceed (they self-resolve); for
        # user-scoped JWTs without an explicit project id, bail out early
        # so we don't make a projects_list() call that's only going to fail
        # ambiguously.
        token = os.getenv("OVERMIND_API_KEY", "").strip()
        if not token.startswith("ovr_"):
            return

    spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
    dataset_path = agent_setup_spec_dir(agent_name) / "dataset.json"
    policy_path = Path(default_policy_path(agent_name))

    spec: dict | None = None
    if spec_path.exists():
        with suppress(Exception):
            loaded = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                spec = loaded

    agent_id = _ensure_remote_agent_id(agent_name, agent_path, console, spec=spec)
    if not agent_id:
        return

    configure_storage(agent_path=agent_path, agent_id=agent_id, agent_name=agent_name)
    try:
        storage = get_storage()
    except Exception:
        return

    synced: list[str] = []

    if spec_path.exists():
        with suppress(Exception):
            loaded = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                spec = loaded
                storage.save_spec(spec)
                synced.append("spec")

    if dataset_path.exists():
        with suppress(Exception):
            loaded = json.loads(dataset_path.read_text(encoding="utf-8"))
            cases = loaded.get("test_cases", []) if isinstance(loaded, dict) else loaded
            if isinstance(cases, list):
                # Uploading a pre-existing local dataset.json — provenance is
                # unknown, so record it as ``seed`` (user-provided data).
                storage.save_dataset(
                    cases,
                    source="seed",
                    metadata={"synced_from": str(dataset_path)},
                )
                synced.append("dataset")

    if policy_path.exists():
        with suppress(Exception):
            policy_md = policy_path.read_text(encoding="utf-8")
            policy_data = spec.get("policy") if isinstance(spec, dict) else None
            storage.save_policy(
                policy_md,
                policy_data if isinstance(policy_data, dict) else None,
            )
            synced.append("policy")

    if synced:
        flush_pending_api_updates(timeout=20.0)
        console.print(f"  [dim]Synced setup artifacts to Overmind ({', '.join(synced)}).[/dim]")


@observe_safe(span_name="overmind.setup.data_phase", type=SpanType.FUNCTION)
def _run_data_phase(
    analysis: dict,
    policy_data: dict | None,
    agent_path: str,
    agent_name: str,
    model: str,
    console: Console,
    *,
    fast: bool = False,
    data_path: str | None = None,
    entrypoint_fn: str = "",
) -> None:
    """Phase 3: Generate or analyze+augment the test dataset.

    This runs after policy is finalized and before eval criteria generation.
    """
    set_tag(attrs.SETUP_FAST, fast)
    set_tag(attrs.SETUP_MODEL, model)
    agent_code = analysis.get("_agent_code_section") or Path(agent_path).read_text()
    description = analysis.get("description", "")
    policy_context = format_for_synthetic_data(policy_data) if policy_data else None
    eval_stub = _build_eval_spec_stub(analysis, policy_data, entrypoint_fn=entrypoint_fn)

    seed_files: list[Path] = []
    if (data_path or "").strip():
        seed_files = _resolve_seed_json_files(data_path.strip(), console=console)
    has_seed_data = bool(seed_files)

    # ── Fast mode ──────────────────────────────────────────────────────────
    if fast:
        datagen_model = _resolve_datagen_model(console, fast=True)
        _pin_model_to_agent_env(
            datagen_model,
            "SYNTHETIC_DATAGEN_MODEL",
            agent_env_path(agent_name),
            agent_name,
        )
        if has_seed_data:
            seed_cases = load_data(str(seed_files[0]))
            seed_cases = normalize_data_fields(seed_cases, console, require_output=False, agent_name=agent_name)
            console.print(
                f"  [dim]Seed data found ({len(seed_cases)} cases) — copying to setup_spec/dataset.json[/dim]"
            )

            # Validate seed data against the input schema before saving.
            from overmind.optimize.data import validate_case_against_spec

            invalid_count = 0
            for case in seed_cases:
                errs = validate_case_against_spec(case, eval_stub)
                if errs:
                    invalid_count += 1
                    if invalid_count <= 3:
                        console.print(f"  [yellow]⚠ Validation issue:[/yellow] [dim]{'; '.join(errs)}[/dim]")
            if invalid_count:
                console.print(
                    f"  [yellow]{invalid_count}/{len(seed_cases)} cases have schema issues "
                    f"(input keys may not match entrypoint params).[/yellow]"
                )

            _save_dataset(
                seed_cases,
                agent_name,
                console,
                source="seed",
                metadata={"seed_file": str(seed_files[0])},
            )

        else:
            with make_spinner_progress(console, transient=True) as progress:
                progress.add_task(f"  Generating synthetic dataset ({datagen_model})…")
                cases = generate_synthetic_data(
                    description,
                    model=datagen_model,
                    num_samples=15,
                    agent_code=agent_code,
                    policy_context=policy_context,
                    expected_output_hint="",
                )
            _save_dataset(
                cases,
                agent_name,
                console,
                source="synthetic",
                generator_model=datagen_model,
                metadata={"num_samples": 15},
            )
        return

    # ── Interactive mode ───────────────────────────────────────────────────
    if has_seed_data:
        seed_path = seed_files[0]
        seed_data = load_data(str(seed_path))
        seed_data = normalize_data_fields(seed_data, console, require_output=False, agent_name=agent_name)
        console.print(
            f"  [bold {BRAND}]Seed data found[/bold {BRAND}]  [dim]{seed_path.name}  ·  {len(seed_data)} cases[/dim]"
        )
        console.print()

        if not confirm_option("Use this seed data?", default=True, console=console):
            # User rejected seed data — offer to generate from scratch
            if not confirm_option(
                "Generate a synthetic dataset from scratch?",
                default=True,
                console=console,
            ):
                console.print("  [dim]Skipping dataset generation.[/dim]")
                return
            datagen_model = _resolve_datagen_model(console)
            _pin_model_to_agent_env(
                datagen_model,
                "SYNTHETIC_DATAGEN_MODEL",
                agent_env_path(agent_name),
                agent_name,
            )
            _ensure_provider_api_keys(datagen_model, agent_env_path(agent_name), agent_name, console)
            _handle_no_data_path(
                analysis=analysis,
                policy_context=policy_context,
                agent_path=agent_path,
                agent_name=agent_name,
                agent_code=agent_code,
                description=description,
                eval_stub=eval_stub,
                datagen_model=datagen_model,
                console=console,
            )
            return

        # User wants to use the seed data
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        console.print(
            Panel(
                "[dim]Recommended: run a quick quality pass before this becomes "
                f"[cyan]{rel(agent_setup_spec_dir(agent_name) / 'dataset.json')}[/cyan]. "
                "Stronger test data means [bold]optimize[/bold] reflects real weaknesses instead "
                "of noise from bad fixtures or blind spots.[/dim]\n\n"
                "[dim]• [bold]Validate[/bold] — check each case against your agent’s expected "
                "inputs/outputs and the eval stub so malformed or inconsistent rows are caught "
                "early.[/dim]\n"
                "[dim]• [bold]Analyze coverage[/bold] — compare your seed set to your policy "
                "and proposed criteria to highlight missing scenarios (personas, edge cases, "
                "tool paths).[/dim]\n"
                "[dim]• [bold]Augment[/bold] — optionally generate extra synthetic cases aimed "
                "at those gaps so you get breadth without hand-editing many new examples.[/dim]\n\n"
                "[dim]If you skip this, we copy your seed file unchanged — fastest, but you "
                "won’t get validation errors, gap analysis, or suggested additions.[/dim]",
                title=f"[bold {BRAND}]Seed data quality[/bold {BRAND}]",
                border_style=BRAND,
                padding=(1, 2),
            )
        )
        console.print()
        if not confirm_option(
            "Run validation, coverage analysis, and optional augmentation?",
            default=True,
            console=console,
        ):
            console.print(f"  [dim]Using seed data as-is ({len(seed_data)} cases).[/dim]")
            _save_dataset(
                seed_data,
                agent_name,
                console,
                source="seed",
                metadata={"seed_file": str(seed_path)},
            )
            return

        datagen_model = _resolve_datagen_model(console)
        _pin_model_to_agent_env(
            datagen_model,
            "SYNTHETIC_DATAGEN_MODEL",
            agent_env_path(agent_name),
            agent_name,
        )
        _ensure_provider_api_keys(datagen_model, agent_env_path(agent_name), agent_name, console)
        _handle_seed_data_path(
            seed_path,
            seed_data=seed_data,
            analysis=analysis,
            policy_data=policy_data,
            policy_context=policy_context,
            agent_path=agent_path,
            agent_name=agent_name,
            agent_code=agent_code,
            description=description,
            eval_stub=eval_stub,
            datagen_model=datagen_model,
            console=console,
        )

    else:
        # No seed files from --data / empty directory / user skipped seed path
        console.print("  [dim]No seed data in use — you can generate a synthetic dataset or skip.[/dim]")
        console.print()
        if not confirm_option("Generate a synthetic dataset from scratch?", default=True, console=console):
            console.print("  [dim]Skipping dataset generation.[/dim]")
            return

        expected_output_hint = _prompt_expected_output_format(console)

        datagen_model = _resolve_datagen_model(console)
        _pin_model_to_agent_env(
            datagen_model,
            "SYNTHETIC_DATAGEN_MODEL",
            agent_env_path(agent_name),
            agent_name,
        )
        _ensure_provider_api_keys(datagen_model, agent_env_path(agent_name), agent_name, console)
        _handle_no_data_path(
            analysis=analysis,
            policy_context=policy_context,
            agent_path=agent_path,
            agent_name=agent_name,
            agent_code=agent_code,
            description=description,
            eval_stub=eval_stub,
            datagen_model=datagen_model,
            console=console,
            expected_output_hint=expected_output_hint,
        )


def _handle_seed_data_path(
    seed_path: Path,
    *,
    seed_data: list[dict],
    analysis: dict,
    policy_data: dict | None,
    policy_context: str | None,
    agent_path: str,
    agent_name: str,
    agent_code: str,
    description: str,
    eval_stub: dict,
    datagen_model: str,
    console: Console,
) -> None:
    """Validate seed rows, analyze coverage vs policy/eval stub, optionally augment (caller chose not to skip)."""
    validate_seed_data(seed_data, eval_stub, console=console)

    coverage = analyze_seed_coverage(
        cases=seed_data,
        eval_spec=eval_stub,
        policy_context=policy_context,
        agent_description=description,
        model=datagen_model,
        console=console,
    )

    gaps = coverage.get("coverage_gaps", [])
    suggested = coverage.get("suggested_additional_cases", 0)
    quality_score = coverage.get("overall_quality_score", 0)

    if not gaps and quality_score >= 8:
        console.print(
            f"\n  [bold {BRAND}]✓[/bold {BRAND}]  [dim]Seed data has excellent coverage — no augmentation needed.[/dim]"
        )
        _save_dataset(
            seed_data,
            agent_name,
            console,
            source="seed",
            metadata={
                "seed_file": str(seed_path),
                "coverage_quality_score": quality_score,
            },
        )
        return

    num_to_generate = max(suggested, len(gaps) * 2, 5)
    console.print()
    console.print(
        f"  [dim]Recommended augmentation:[/dim]  [bold]{num_to_generate}[/bold] additional cases"
        + (f"  [dim]({len(gaps)} gap(s) identified)[/dim]" if gaps else "")
    )
    num_to_generate = IntPrompt.ask("  Additional cases to generate", default=num_to_generate)

    if num_to_generate <= 0:
        console.print(f"\n  [dim]Skipping augmentation — keeping {len(seed_data)} seed case(s) as-is.[/dim]")
        _save_dataset(seed_data, agent_name, console)
        return

    console.print()
    new_cases = generate_diverse_synthetic_data(
        agent_description=description,
        model=datagen_model,
        num_samples=num_to_generate,
        num_personas=min(3, max(1, len(gaps))),
        agent_code=agent_code,
        policy_context=policy_context,
        eval_spec=eval_stub,
        existing_cases=seed_data,
        coverage_gaps=gaps,
        console=console,
    )

    combined = seed_data + new_cases
    console.print(
        f"\n  [dim]Merged:[/dim]"
        f"  {len(seed_data)} seed  +  {len(new_cases)} generated"
        f"  [bold]= {len(combined)} total[/bold]"
    )
    _save_dataset(
        combined,
        agent_name,
        console,
        source="augmented",
        generator_model=datagen_model,
        metadata={
            "seed_file": str(seed_path),
            "seed_count": len(seed_data),
            "generated_count": len(new_cases),
            "coverage_gaps": len(gaps),
            "coverage_quality_score": quality_score,
        },
    )


def _handle_no_data_path(
    *,
    analysis: dict,
    policy_context: str | None,
    agent_path: str,
    agent_name: str,
    agent_code: str,
    description: str,
    eval_stub: dict,
    datagen_model: str,
    console: Console,
    expected_output_hint: str = "",
) -> None:
    """Path A: no seed data — full persona-driven generation."""
    console.print(
        "  [dim]No existing dataset found. Overmind will generate diverse test cases"
        " using your agent policy and code as context.[/dim]"
    )
    console.print()

    num_samples = IntPrompt.ask("  Test cases to generate", default=20)
    console.print(
        "  [dim]The number of user personas determines how many distinct user types, roles, or scenarios the generated test cases will represent—for example, SME, GC, end user, or distinct legal/commercial stances.[/dim]"
    )
    num_personas = IntPrompt.ask(
        "  How many user personas? (More = broader coverage)",
        default=5,
        show_default=True,
        console=console,
    )

    console.print()
    cases = generate_diverse_synthetic_data(
        agent_description=description,
        model=datagen_model,
        num_samples=num_samples,
        num_personas=num_personas,
        agent_code=agent_code,
        policy_context=policy_context,
        eval_spec=eval_stub,
        console=console,
        expected_output_hint=expected_output_hint,
    )

    _save_dataset(
        cases,
        agent_name,
        console,
        source="synthetic",
        generator_model=datagen_model,
        metadata={"num_samples": num_samples, "num_personas": num_personas},
    )


def _display_proposed_criteria(analysis: dict, console: Console) -> None:
    """Show the proposed evaluation criteria table so the user can review it."""
    from rich.table import Table

    criteria = analysis.get("proposed_criteria", {})
    fields_criteria = criteria.get("fields", {})
    output_schema = analysis.get("output_schema", {})

    if not fields_criteria:
        console.print("  [dim]No proposed criteria available.[/dim]")
        return

    table = Table(title="Proposed Evaluation Criteria", border_style="green")
    table.add_column("Field", style="bold")
    table.add_column("Importance")
    table.add_column("Scoring Detail")

    for field_name, fc in fields_criteria.items():
        importance = fc.get("importance", "important")
        ftype = output_schema.get(field_name, {}).get("type", "text")

        if ftype == "enum":
            detail = "partial credit" if fc.get("partial_credit", True) else "exact match only"
        elif ftype == "number":
            detail = f"tolerance \u00b1{fc.get('tolerance', 10)}"
        elif ftype == "text":
            mode = fc.get("eval_mode", "non_empty")
            detail = "check non-empty" if mode == "non_empty" else "skip"
        else:
            detail = "exact match"

        table.add_row(field_name, importance, detail)

    sw = criteria.get("structure_weight", 20)
    table.add_row(
        "[dim]structure[/dim]",
        "[dim]\u2014[/dim]",
        f"[dim]{sw} pts for completeness[/dim]",
    )
    console.print(table)


def _write_agent_env(path: Path, agent_name: str, env_vars: dict[str, str]) -> None:
    """Write agent-specific env vars — delegates to shared module."""
    from overmind.commands.agent_env import write_agent_env

    write_agent_env(path, agent_name, env_vars)


def _pin_model_to_agent_env(model: str, env_key: str, env_path: Path, agent_name: str) -> None:
    """Save *model* under *env_key* in the agent's ``.env`` and copy any
    provider credentials from the global environment if not already present.

    This makes the agent env self-contained: Overmind will always load it
    instead of the global ``.overmind/.env`` when setting up or optimizing the
    agent.
    """
    updates: dict[str, str] = {env_key: model}

    provider = model.split("/")[0] if "/" in model else ""
    env_key_names = _PROVIDER_ENV_KEYS.get(provider, [])
    if env_key_names:
        existing = {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()} if env_path.exists() else {}
        for key_name in env_key_names:
            if not existing.get(key_name, "").strip():
                global_val = os.getenv(key_name, "").strip()
                if global_val:
                    updates[key_name] = global_val

    _update_agent_env(env_path, agent_name, updates)


def _collect_agent_provider_config(agent_name: str, console: Console) -> None:
    """Ask which LLM provider the agent uses — delegates to shared module."""
    from overmind.commands.agent_env import collect_agent_provider_config

    collect_agent_provider_config(agent_name, console)


@observe_safe(span_name="overmind.setup", type=SpanType.WORKFLOW)
def main(
    agent_name: str,
    fast: bool = False,
    policy: str | None = None,
    data: str | None = None,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> None:
    logger.info(f"setup: start agent={agent_name} fast={fast} policy={policy} data={data}")
    load_overmind_dotenv()

    # CLI-level flags — set as soon as the span is open so the
    # backend can identify the run before any phase work begins.
    set_tag(attrs.SETUP_FAST, fast)
    set_tag(attrs.SETUP_HAS_POLICY, policy)
    set_tag(attrs.SETUP_HAS_SEED_DATA, data)
    if policy:
        set_tag(attrs.SETUP_POLICY_PATH, policy)
    if data:
        set_tag(attrs.SETUP_DATA_PATH, data)

    console = Console()
    console.print()
    render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]Overmind \u2014 Agent Setup[/bold cyan]\n"
            "[dim]Analyze your agent, define policies, and build "
            "evaluation criteria[/dim]",
            border_style=BRAND,
        )
    )

    if fast:
        console.print(
            "  [dim]Fast mode: no prompts; clearing prior setup_spec if present; "
            "requires ANALYZER_MODEL and SYNTHETIC_DATAGEN_MODEL.[/dim]\n"
        )

    agent_path, fn_name = resolve_agent(agent_name)

    data_opt = (data or "").strip() or None
    if not fast and not data_opt:
        console.print()
        _prompt_seed_data_flag_early(agent_name, console=console)

    # Agent env vars (API keys) — may have been configured during register.
    # The shared function skips if already configured; always load into env.
    console.print()
    console.print(Rule(style="dim"))
    if not fast:
        _collect_agent_provider_config(agent_name, console)
    load_agent_dotenv(agent_name)

    agent_id = _ensure_remote_agent_id(agent_name, agent_path, console)
    configure_storage(
        agent_path=agent_path,
        agent_id=agent_id,
        agent_name=agent_name,
    )
    set_tag(attrs.SETUP_STORAGE_BACKEND, "api")
    set_tag(attrs.SETUP_AGENT_PATH, agent_path)
    set_tag(attrs.SETUP_ENTRYPOINT_FN, fn_name)

    _sigint_flushed = {"done": False}

    def _handle_sigint(_signum, _frame):
        if _sigint_flushed["done"]:
            raise SystemExit(130)
        _sigint_flushed["done"] = True
        console.print("\n  [yellow]Interrupted. Flushing pending Overmind updates...[/yellow]")
        flush_pending_api_updates(timeout=8.0)
        console.print("  [dim]Pending updates flushed. Exiting.[/dim]\n")
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)
    agent_path, fn_name = _validate_agent_entrypoint(agent_path, fn_name, agent_name, console, fast=fast)

    if policy and not Path(policy).exists():
        console.print(f"\n  [red]Error:[/red] Policy file {policy} does not exist.")
        raise SystemExit(1)

    _clear_existing_eval_spec(agent_name, console, fast=fast)

    # The instrumented copy is created at register time.  Re-use it as-is so
    # we don't wipe the generated _overmind_entrypoint wrapper (if any).  Only
    # copy if the instrumented dir is missing (e.g. manual cleanup).
    instrumented_root = agent_instrumented_dir(agent_name)
    if instrumented_root.exists():
        p = Path(agent_path).resolve()
        if str(p).startswith(str(instrumented_root)):
            instrumented_entry = str(p)
        else:
            pr = project_root_from_agent_file(agent_path)
            copy_root = pr if pr is not None else p.parent
            instrumented_entry = str(instrumented_root / p.relative_to(copy_root))
    else:
        instrumented_entry, instrumented_root = _instrument_agent_files(agent_path, agent_name, console)
    _check_agent_dependencies(agent_path, agent_name, console, fast=fast, instrumented_dir=instrumented_root)

    console.print()
    console.print(Rule(style="dim"))
    _run_beginning_smoke_test(
        agent_path,
        agent_name,
        fn_name,
        console,
        fast=fast,
        data_path=data_opt,
        instrumented_entry=instrumented_entry,
    )

    if fast:
        raw_model = os.getenv("ANALYZER_MODEL", "").strip()
        if not raw_model:
            console.print("\n[red]Fast mode requires ANALYZER_MODEL in the environment.[/red]")
            console.print(
                f"[dim]Set it in {overmind_rel('.env')} or your shell (see .env.example). "
                "Run without --fast to pick a model interactively.[/dim]\n"
            )
            raise SystemExit(1)
        model = normalize_to_litellm_model_id(raw_model) or raw_model
        set_tag(attrs.SETUP_ANALYZER_MODEL, model)
        _pin_model_to_agent_env(model, "ANALYZER_MODEL", agent_env_path(agent_name), agent_name)

        if not os.getenv("SYNTHETIC_DATAGEN_MODEL", "").strip():
            console.print("\n[red]Fast mode requires SYNTHETIC_DATAGEN_MODEL in the environment.[/red]")
            console.print(
                "[dim]Used by Overmind optimize when generating synthetic test data. "
                f"Set it in {overmind_rel('.env')} (see .env.example) or run without --fast.[/dim]\n"
            )
            raise SystemExit(1)

    if not fast:
        console.print()
        console.print(Rule(style="dim"))
        raw_model = os.getenv("ANALYZER_MODEL", "").strip()
        if raw_model:
            model = normalize_to_litellm_model_id(raw_model) or raw_model
            display = model.split("/", 1)[-1] if "/" in model else model
            console.print(f"\n  [dim]ANALYZER_MODEL is already set to[/dim] [cyan]{display}[/cyan]")
            if not confirm_option(
                f"Use {display} as the analyzer model?",
                default=True,
                console=console,
            ):
                model = prompt_for_catalog_litellm_model(
                    console,
                    select_prompt="  Select model for agent analysis (number)",
                    env_default=model,
                    default_model=DEFAULT_ANALYZER_MODEL,
                    no_catalog_prompt="  Enter model for analysis (provider/model)",
                )
        else:
            console.print(
                f"\n  [dim]No ANALYZER_MODEL set in {overmind_rel('.env')} — select a model for agent analysis.[/dim]"
            )
            model = prompt_for_catalog_litellm_model(
                console,
                select_prompt="  Select model for agent analysis (number)",
                env_default=None,
                default_model=DEFAULT_ANALYZER_MODEL,
                no_catalog_prompt="  Enter model for analysis (provider/model)",
            )
        _pin_model_to_agent_env(model, "ANALYZER_MODEL", agent_env_path(agent_name), agent_name)
        _ensure_provider_api_keys(model, agent_env_path(agent_name), agent_name, console)
        load_agent_dotenv(agent_name)

    # ---- Phase 1: Agent Analysis ----
    logger.info(f"PHASE BEGIN setup.phase1.agent_analysis agent={agent_name} model={model}")
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Phase 1 \u00b7 Agent Analysis[/bold]\n"
            "[dim]Examining code structure, tool definitions, "
            "parameter constraints, and data dependencies[/dim]",
            border_style=BRAND,
        )
    )

    analysis = analyze_agent(
        agent_path,
        model,
        console,
        entrypoint_fn=fn_name,
        max_resolved_files=max_files if max_files is not None else 48,
        max_total_chars=max_chars if max_chars is not None else 80_000,
        scope_hint_globs=list(scope_globs) if scope_globs else None,
    )
    logger.info(
        f"PHASE END   setup.phase1.agent_analysis fields={list(analysis.get('output_schema', {}).keys())} "
        f"criteria_fields={list(analysis.get('proposed_criteria', {}).get('fields', {}).keys())}"
    )

    set_tag(attrs.SETUP_PHASE, "agent_analysis")
    set_tag(attrs.SETUP_FAST, fast)
    set_tag(attrs.SETUP_AGENT_PATH, agent_path)
    set_tag(attrs.SETUP_ANALYZER_MODEL, model)
    set_tag(attrs.SETUP_ENTRYPOINT_FN, fn_name)

    # ---- Phase 2: Policy Definition ----
    logger.info(f"PHASE BEGIN setup.phase2.policy agent={agent_name} fast={fast} policy_arg={policy}")
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Phase 2 \u00b7 Agent Policy[/bold]\n"
            "[dim]Define the decision rules, constraints, and expectations "
            "that govern your agent's behaviour[/dim]",
            border_style=BRAND,
        )
    )

    set_tag(attrs.SETUP_PHASE, "policy")
    policy_md: str | None = None
    policy_data: dict | None = None

    if fast:
        if policy:
            policy_md, policy_data, _changes = improve_existing_policy(analysis, policy, model, console)
        else:
            policy_md, policy_data = generate_policy_from_code(analysis, model, console)
            console.print(
                f"  [dim]Auto-generated policy from code. Edit "
                f"[cyan]{rel(default_policy_path(agent_name))}[/cyan] "
                f"to improve optimization quality.[/dim]"
            )

        logger.info(f"PHASE END   setup.phase2.policy agent={agent_name} fast=True")

        # ---- Phase 3 (fast): Dataset ----
        logger.info(f"PHASE BEGIN setup.phase3.dataset agent={agent_name} fast=True data={data_opt}")
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        console.print(
            Panel(
                "[bold]Phase 3 \u00b7 Dataset[/bold]\n[dim]Preparing test data for optimization[/dim]",
                border_style=BRAND,
            )
        )
        set_tag(attrs.SETUP_PHASE, "dataset")
        _run_data_phase(
            analysis,
            policy_data,
            agent_path,
            agent_name,
            model,
            console,
            fast=True,
            data_path=data_opt,
            entrypoint_fn=fn_name,
        )
        logger.info(f"PHASE END   setup.phase3.dataset agent={agent_name} fast=True")

        set_tag(attrs.SETUP_PHASE, "complete")
        spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
        _save_and_finish(spec, agent_name, console, policy_md=policy_md)

        _run_end_smoke_test(
            agent_name,
            agent_path,
            fn_name,
            console,
            instrumented_entry=instrumented_entry,
        )
        _sync_setup_artifacts(agent_name, agent_path, console)
        logger.info(f"setup: fast-mode complete agent={agent_name}")

        return

    pol_path = default_policy_path(agent_name)

    if policy:
        # ---- Path A: User provided a policy document ----
        console.print(f"  [dim]Analyzing your policy from [cyan]{policy}[/cyan] against agent code…[/dim]\n")
        improved_md, improved_data, change_summary = improve_existing_policy(analysis, policy, model, console)

        if change_summary:
            console.print()
            console.print(Rule(style="dim"))
            console.print()
            console.print(
                Panel(
                    "[bold]Suggested Improvements[/bold]\n\n" + change_summary,
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

        display_policy(improved_md, improved_data, console)

        console.print()
        pol_choice = select_option(
            [
                "Use the improved policy (with suggested changes)",
                "Keep my original policy (no changes)",
            ],
            title="Which policy would you like to use?",
            default_index=0,
            console=console,
        )

        if pol_choice == 0:
            policy_md, policy_data = improved_md, improved_data
        else:
            from overmind.setup.policy_generator import generate_policy_from_document

            policy_md, policy_data = generate_policy_from_document(analysis, policy, model, console)
    else:
        # ---- Path B: No policy document provided ----
        # Ask whether the user wants to define policies interactively or let
        # the system infer them from code.
        console.print("  [dim]No policy document provided.[/dim]\n")
        pol_input_choice = select_option(
            [
                "Define policies interactively (recommended — describe your domain rules)",
                "Auto-generate from agent code (faster, less accurate)",
            ],
            title="How would you like to define the agent policy?",
            default_index=1,
            console=console,
        )

        if pol_input_choice == 0:
            policy_md, policy_data = elicit_policy(analysis, model, console)
        else:
            policy_md, policy_data = generate_policy_from_code(analysis, model, console)
            display_policy(policy_md, policy_data, console)

    # ---- Policy review / refinement loop ----
    policy_round = 0
    while True:
        console.print()
        if confirm_option("Are you satisfied with this policy?", default=True, console=console):
            save_policy(policy_md, pol_path)
            set_tag(attrs.SETUP_AGENT_POLICY_MARKDOWN, policy_md)
            set_tag(attrs.SETUP_AGENT_POLICY_DATA, policy_data)
            console.print(f"\n  [dim]You can always edit the policy later at [cyan]{rel(pol_path)}[/cyan][/dim]")
            break

        policy_round += 1
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        console.print(
            Panel(
                f"[bold]Policy Refinement Round {policy_round}[/bold]",
                border_style=BRAND,
            )
        )
        policy_md, policy_data = refine_policy(policy_md, policy_data, analysis, model, console)

    logger.info(f"PHASE END   setup.phase2.policy agent={agent_name} fast=False")

    # ---- Phase 3: Dataset ----
    logger.info(f"PHASE BEGIN setup.phase3.dataset agent={agent_name} data={data_opt}")
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Phase 3 \u00b7 Dataset[/bold]\n[dim]Generate or analyze test data for optimization[/dim]",
            border_style=BRAND,
        )
    )
    set_tag(attrs.SETUP_PHASE, "dataset")
    _run_data_phase(
        analysis,
        policy_data,
        agent_path,
        agent_name,
        model,
        console,
        data_path=data_opt,
        entrypoint_fn=fn_name,
    )
    logger.info(f"PHASE END   setup.phase3.dataset agent={agent_name}")

    # ---- Phase 4: Evaluation Criteria ----
    logger.info(f"PHASE BEGIN setup.phase4.eval_criteria agent={agent_name}")
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Phase 4 \u00b7 Evaluation Criteria[/bold]\n"
            "[dim]Proposed scoring rules for your agent's output fields[/dim]",
            border_style=BRAND,
        )
    )

    set_tag(attrs.SETUP_PHASE, "eval_criteria")

    _display_proposed_criteria(analysis, console)

    iteration = 0
    while True:
        console.print()
        if confirm_option(
            "Are you satisfied with the evaluation criteria?",
            default=True,
            console=console,
        ):
            logger.info(
                f"PHASE END   setup.phase4.eval_criteria agent={agent_name} iterations={iteration} accepted=True"
            )
            spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
            _save_and_finish(
                spec,
                agent_name,
                console,
                policy_md=policy_md,
                policy_file_already_saved=True,
            )

            _run_end_smoke_test(
                agent_name,
                agent_path,
                fn_name,
                console,
                instrumented_entry=instrumented_entry,
            )
            _sync_setup_artifacts(agent_name, agent_path, console)
            logger.info(f"setup: complete agent={agent_name}")

            return

        console.print()
        choice = select_option(
            [
                "Refine criteria through conversation",
                "Save now and edit the spec manually",
            ],
            title="Refinement Options:",
            default_index=0,
            console=console,
        )

        if choice == 1:
            logger.info(
                f"PHASE END   setup.phase4.eval_criteria agent={agent_name} iterations={iteration} "
                "accepted=False save_manual=True"
            )
            spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
            _save_and_finish(
                spec,
                agent_name,
                console,
                policy_md=policy_md,
                policy_file_already_saved=True,
            )
            spec_out = agent_setup_spec_dir(agent_name) / "eval_spec.json"
            console.print(
                f"  [dim]Edit [cyan]{spec_out}[/cyan] to fine-tune the criteria, then run the optimizer.[/dim]\n"
            )

            _run_end_smoke_test(
                agent_name,
                agent_path,
                fn_name,
                console,
                instrumented_entry=instrumented_entry,
            )
            _sync_setup_artifacts(agent_name, agent_path, console)
            logger.info(f"setup: complete-save-manual agent={agent_name}")

            return

        iteration += 1
        logger.info(f"setup.phase4.eval_criteria refinement round={iteration} agent={agent_name}")
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        console.print(
            Panel(
                f"[bold]Refinement Round {iteration}[/bold]",
                border_style=BRAND,
            )
        )
        analysis = run_questionnaire(analysis, model, console)
