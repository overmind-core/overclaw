"""
OverClaw setup — Agent Setup

Analyzes your agent's code, tools, and orchestration to propose evaluation
criteria. Lets you accept them or iteratively refine through conversation.

The setup flow:
  Phase 1: Agent Analysis  — examine code, tools, schemas
  Phase 2: Policy          — define domain rules and constraints
  Phase 3: Dataset         — generate/analyze test data (after policy, before criteria)
  Phase 4: Eval Criteria   — propose and refine scoring rules

Usage:
    overclaw setup <agent-name>
    overclaw setup <agent-name> --data path/to/seed.json
    overclaw setup <agent-name> --data path/to/json_dir/
    overclaw setup <agent-name> --fast
"""

import json
import os
import shlex
import signal
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.rule import Rule

from overclaw.utils.display import (
    BRAND,
    confirm_option,
    make_spinner_progress,
    rel,
    render_logo,
    select_option,
)
from overclaw.core.constants import overclaw_rel
from overclaw.utils.io import read_api_key_masked
from overclaw.utils.model_picker import prompt_for_catalog_litellm_model
from overclaw.utils.models import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_DATAGEN_MODEL,
    normalize_to_litellm_model_id,
)
from overclaw.utils.provider_keys import (
    PROVIDER_ENV_KEYS as _PROVIDER_ENV_KEYS,
    ensure_provider_api_keys as _ensure_provider_api_keys,
    update_agent_env as _update_agent_env,
)
from overclaw.utils.policy import default_policy_path, format_for_synthetic_data
from overclaw.optimize.data import (
    generate_diverse_synthetic_data,
    generate_synthetic_data,
    load_data,
)
from overclaw.optimize.data_analyzer import analyze_seed_coverage, validate_seed_data
from overclaw.optimize.evaluator import has_entrypoint
from overclaw.core.paths import (
    agent_env_path,
    agent_setup_spec_dir,
    load_agent_dotenv,
    load_overclaw_dotenv,
)
from overclaw.core.registry import (
    get_agent_id,
    load_registry,
    resolve_agent,
    save_agent,
)
from overclaw.client import (
    _run_async,
    flush_pending_api_updates,
    get_client,
    get_project_id,
    upsert_agent,
)
from overclaw.setup.agent_analyzer import analyze_agent
from overclaw.setup.policy_generator import (
    display_policy,
    elicit_policy,
    generate_policy_from_code,
    improve_existing_policy,
    refine_policy,
    save_policy,
)
from overclaw.setup.questionnaire import run_questionnaire
from overclaw.setup.spec_generator import generate_spec_from_proposal, save_spec
from overclaw.storage import configure_storage, get_storage
from overclaw.storage.api import ApiBackend


def _validate_agent_entrypoint(agent_path: str, fn_name: str, console: Console) -> None:
    """Exit with a clear message if the agent file lacks the registered entry function."""
    code = Path(agent_path).read_text()
    if not has_entrypoint(code, fn_name):
        console.print(
            f"\n  [bold red]Error:[/bold red] Function [bold]{fn_name}()[/bold] not found "
            f"in [cyan]{agent_path}[/cyan].\n"
        )
        console.print(
            f"  OverClaw calls [bold]agent.{fn_name}(case_input)[/bold] for every test case.\n"
            f"  Make sure your agent file defines:\n\n"
            f"  [dim]def {fn_name}(input: dict) -> dict:\n"
            f"      # your agent logic here\n"
            f"      return {{...}}[/dim]\n\n"
            f"  Or update the registered entrypoint:\n"
            f"    [bold]overclaw agent update <name> <module:{fn_name}>[/bold]\n"
        )
        raise SystemExit(1)


def _clear_existing_eval_spec(
    agent_name: str, console: Console, *, fast: bool = False
) -> None:
    with suppress(ValueError):
        storage = get_storage()
        if isinstance(storage, ApiBackend):
            if fast:
                storage.clear_setup_spec()
                console.print(
                    "  [dim]Cleared setup artifacts in Overmind (fast mode).[/dim]"
                )
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

    console.print(
        f"\n  [yellow]Found {len(existing)} existing file(s) in setup_spec/[/yellow]"
    )

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
        console.print(
            "  [dim]Keeping existing files. New spec will overwrite setup_spec/eval_spec.json.[/dim]"
        )


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
    with suppress(ValueError):
        storage = get_storage()
    if isinstance(storage, ApiBackend):
        storage.save_spec(spec)
        if policy_md:
            storage.save_policy(policy_md, spec.get("policy"))

    n_fields = len(spec.get("output_fields", {}))
    has_tools = bool(spec.get("tool_config", {}).get("expected_tools"))
    has_consistency = bool(spec.get("consistency_rules"))
    has_judge = spec.get("llm_judge_weight", 0) > 0
    has_policy = bool(spec.get("policy"))

    features: list[str] = [f"{n_fields} output field(s)"]
    if has_tools:
        n_tools = len(spec["tool_config"]["expected_tools"])
        features.append(f"{n_tools} tool(s) monitored")
    if has_consistency:
        features.append(f"{len(spec['consistency_rules'])} consistency rule(s)")
    if has_judge:
        features.append("LLM-as-Judge enabled")
    if has_policy:
        n_rules = len(
            spec["policy"].get("domain_rules", spec["policy"].get("decision_rules", []))
        )
        features.append(f"policy ({n_rules} rule(s))")

    console.print(
        f"\n  [bold green]\u2713[/bold green] Spec saved  [dim]→ {rel(spec_path)}[/dim]"
    )
    if policy_md and not policy_file_already_saved:
        pol_path = default_policy_path(agent_name)
        console.print(
            f"  [bold green]\u2713[/bold green] Policy saved  "
            f"[dim]→ {rel(pol_path)}[/dim]"
        )
    if isinstance(storage, ApiBackend):
        console.print("  [dim]Queued sync to Overmind backend.[/dim]")
    console.print(f"  [dim]Spec covers: {', '.join(features)}[/dim]")
    next_cmd = agent_name
    console.print(
        f"\n  Next step: [bold {BRAND}]overclaw optimize {next_cmd}[/bold {BRAND}]\n"
    )


def _smoke_test_agent(
    agent_path: str, fn_name: str, input_case: dict
) -> tuple[bool, str | None]:
    """Import the agent module and call fn_name(input_case) once.

    Returns (True, None) on success or (False, error_message) on any exception.
    Uses a throwaway module name so repeated imports don't collide.
    """
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            "_overclaw_smoke_agent", agent_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        getattr(module, fn_name)(input_case)
        return True, None
    except Exception as exc:
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
            f"\n  [red]Error:[/red] Could not resolve [bold]--data[/bold] path "
            f"[cyan]{raw}[/cyan] [dim]({exc})[/dim]"
        )
        raise SystemExit(1) from exc
    if not p.exists():
        console.print(
            f"\n  [red]Error:[/red] [bold]--data[/bold] path does not exist: [cyan]{raw}[/cyan]"
        )
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
    console.print(
        f"\n  [red]Error:[/red] [bold]--data[/bold] must be a file or directory: [cyan]{raw}[/cyan]"
    )
    raise SystemExit(1)


def _prompt_seed_data_flag_early(agent_name: str, *, console: Console) -> None:
    """Explain seed data, ask about ``--data``; if yes, print the command and exit setup."""
    console.print(
        "  [dim]It looks like no seed data was provided ([bold]--data[/bold] was not set). "
        "Seed JSON (real or representative inputs) lets OverClaw smoke-test your agent early, "
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
    console.print(
        f"    [bold cyan]overclaw setup {quoted} --data path/to/cases.json[/bold cyan]"
    )
    console.print(
        f"    [bold cyan]overclaw setup {quoted} --data path/to/dataset_folder/[/bold cyan]"
    )
    console.print()
    console.print(
        "  [dim]Exiting setup — run the command above when your seed files are ready.[/dim]\n"
    )
    raise SystemExit(0)


def _run_beginning_smoke_test(
    agent_path: str,
    fn_name: str,
    console: Console,
    *,
    fast: bool = False,
    data_path: str | None = None,
) -> None:
    """Smoke-test the agent with the first seed case when ``--data`` supplies JSON.

    Hard-fails (SystemExit 1) when seed data exists but the agent crashes.
    Skips when ``--data`` is omitted — use ``--data`` for an early smoke check.
    """
    existing_json = _resolve_seed_json_files(data_path, console=console)

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
            f"  [dim]Could not read [cyan]{existing_json[0].name}[/cyan] — "
            "skipping smoke test.[/dim]"
        )
        return

    if not cases:
        console.print(
            f"  [dim][cyan]{existing_json[0].name}[/cyan] is empty — "
            "skipping pre-setup smoke test.[/dim]"
        )
        return

    first_input = cases[0].get("input", cases[0])
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task(
            f"  Smoke-testing agent using {existing_json[0].name} ({len(cases)} case(s))…"
        )
        success, error = _smoke_test_agent(agent_path, fn_name, first_input)

    if success:
        console.print(
            "  [bold green]✓[/bold green]  [dim]Agent smoke test passed.[/dim]\n"
        )
    else:
        console.print(
            f"\n  [bold red]✗  Agent smoke test failed[/bold red]\n"
            f"  [dim]{error}[/dim]\n\n"
            "  Fix the error above before running setup.\n"
        )
        raise SystemExit(1)


def _run_end_smoke_test(
    agent_name: str, agent_path: str, fn_name: str, console: Console
) -> None:
    """Validate the agent runs against the first generated dataset case.

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

    first_input = cases[0].get("input", cases[0])
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task("  Post-setup smoke test against first dataset case…")
        success, error = _smoke_test_agent(agent_path, fn_name, first_input)

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
                f"  [bold]overclaw optimize {agent_name}[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )


def _data_dir(agent_path: str) -> Path:
    """Historical default sibling ``data/`` directory (no longer used unless you pass ``--data``)."""
    return Path(agent_path).resolve().parent / "data"


def _build_eval_spec_stub(analysis: dict, policy_data: dict | None = None) -> dict:
    """Build a minimal eval-spec-like dict from analysis for schema validation.

    At setup time the real eval spec doesn't exist yet, but the data
    generation functions need ``input_schema`` and ``output_fields`` for
    validation.  The analysis dict has ``output_schema`` which uses the
    same per-field shape (type, values, range, description).
    f"""
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
            f"Use {resolved} from {overclaw_rel('.env')} for data generation?",
            default=True,
            console=console,
        ):
            return resolved

    if fast:
        console.print(
            "\n[red]Fast mode requires SYNTHETIC_DATAGEN_MODEL in the environment.[/red]"
        )
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
            f"{overclaw_rel('.env')}. Pick a provider and model below; we’ll remember it "
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


def _save_dataset(cases: list[dict], agent_name: str, console: Console) -> str:
    """Write the final dataset to setup_spec/dataset.json. Returns the path."""
    data_path = agent_setup_spec_dir(agent_name) / "dataset.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(cases, f, indent=2)

    storage = None
    with suppress(ValueError):
        storage = get_storage()
    if isinstance(storage, ApiBackend):
        storage.save_dataset(cases)

    console.print(
        f"\n  [bold {BRAND}]✓[/bold {BRAND}]"
        f"  Saved [bold]{len(cases)}[/bold] cases"
        f"  [dim]→ {rel(data_path)}[/dim]"
    )
    if isinstance(storage, ApiBackend):
        console.print("  [dim]Queued dataset sync to Overmind backend.[/dim]")
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
    project_id = get_project_id()
    if existing_id:
        # Verify stored id belongs to the currently configured project.
        # This avoids silently writing to another project's similarly-slugged agent.
        if client and project_id:
            with suppress(Exception):
                existing = _run_async(client.agents_retrieve(id=UUID(existing_id)))
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
                )
                new_id = str(result.id)
                entrypoint = (load_registry().get(agent_name, {}) or {}).get(
                    "entrypoint"
                )
                if entrypoint:
                    save_agent(agent_name, entrypoint, id=new_id)
                console.print(
                    "  [dim]Remote agent created and id stored in agents.toml.[/dim]"
                )
                return new_id
        console.print(
            "  [yellow]Warning:[/yellow] Could not create agent in Overmind. "
            f"[dim]({exc})[/dim]"
        )
        return None


def _sync_setup_artifacts(agent_name: str, agent_path: str, console: Console) -> None:
    """Upload local setup artifacts to Overmind backend if configured."""
    if not get_client() or not get_project_id():
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

    configure_storage(agent_path=agent_path, agent_id=agent_id, backend="api")
    storage = get_storage()
    if not isinstance(storage, ApiBackend):
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
                storage.save_dataset(cases)
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
        console.print(
            f"  [dim]Synced setup artifacts to Overmind ({', '.join(synced)}).[/dim]"
        )


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
) -> None:
    """Phase 3: Generate or analyze+augment the test dataset.

    This runs after policy is finalized and before eval criteria generation.
    """
    agent_code = analysis.get("_agent_code_section") or Path(agent_path).read_text()
    description = analysis.get("description", "")
    policy_context = format_for_synthetic_data(policy_data) if policy_data else None
    eval_stub = _build_eval_spec_stub(analysis, policy_data)

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
            console.print(
                f"  [dim]Seed data found ({len(seed_cases)} cases)"
                " — copying to setup_spec/dataset.json[/dim]"
            )
            _save_dataset(seed_cases, agent_name, console)
        else:
            with make_spinner_progress(console, transient=True) as progress:
                progress.add_task(f"  Generating synthetic dataset ({datagen_model})…")
                cases = generate_synthetic_data(
                    description,
                    model=datagen_model,
                    num_samples=15,
                    agent_code=agent_code,
                    policy_context=policy_context,
                )
            _save_dataset(cases, agent_name, console)
        return

    # ── Interactive mode ───────────────────────────────────────────────────
    if has_seed_data:
        seed_path = seed_files[0]
        seed_data = load_data(str(seed_path))
        console.print(
            f"  [bold {BRAND}]Seed data found[/bold {BRAND}]"
            f"  [dim]{seed_path.name}  ·  {len(seed_data)} cases[/dim]"
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
            _ensure_provider_api_keys(
                datagen_model, agent_env_path(agent_name), agent_name, console
            )
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
            console.print(
                f"  [dim]Using seed data as-is ({len(seed_data)} cases).[/dim]"
            )
            _save_dataset(seed_data, agent_name, console)
            return

        datagen_model = _resolve_datagen_model(console)
        _pin_model_to_agent_env(
            datagen_model,
            "SYNTHETIC_DATAGEN_MODEL",
            agent_env_path(agent_name),
            agent_name,
        )
        _ensure_provider_api_keys(
            datagen_model, agent_env_path(agent_name), agent_name, console
        )
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
        console.print(
            "  [dim]No seed data in use — you can generate a synthetic dataset or skip.[/dim]"
        )
        console.print()
        if not confirm_option(
            "Generate a synthetic dataset from scratch?", default=True, console=console
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
        _ensure_provider_api_keys(
            datagen_model, agent_env_path(agent_name), agent_name, console
        )
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
            f"\n  [bold {BRAND}]✓[/bold {BRAND}]"
            "  [dim]Seed data has excellent coverage — no augmentation needed.[/dim]"
        )
        _save_dataset(seed_data, agent_name, console)
        return

    num_to_generate = max(suggested, len(gaps) * 2, 5)
    console.print()
    console.print(
        f"  [dim]Recommended augmentation:[/dim]  [bold]{num_to_generate}[/bold] additional cases"
        + (f"  [dim]({len(gaps)} gap(s) identified)[/dim]" if gaps else "")
    )
    num_to_generate = IntPrompt.ask(
        "  Additional cases to generate", default=num_to_generate
    )

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
    _save_dataset(combined, agent_name, console)


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
) -> None:
    """Path A: no seed data — full persona-driven generation."""
    console.print(
        "  [dim]No existing dataset found. OverClaw will generate diverse test cases"
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
    )

    _save_dataset(cases, agent_name, console)


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
            detail = (
                "partial credit"
                if fc.get("partial_credit", True)
                else "exact match only"
            )
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
    """Write agent-specific env vars to ``.overclaw/agents/<name>/.env``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# OverClaw agent env — {agent_name}", ""]
    for key, val in env_vars.items():
        lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _describe_configured_agent_llm_provider(existing: dict[str, str]) -> str | None:
    """Summarize LLM provider from env keys and ``ANALYZER_MODEL`` (no secret values)."""
    oai_key = (existing.get("OPENAI_API_KEY") or "").strip()
    ant_key = (existing.get("ANTHROPIC_API_KEY") or "").strip()
    base_url = (existing.get("OPENAI_BASE_URL") or "").strip()
    analyzer = (existing.get("ANALYZER_MODEL") or "").strip()

    label: str | None = None

    if base_url and oai_key:
        label = "Other (OpenAI-compatible SDK, custom base URL)"
    elif analyzer and "/" in analyzer:
        prefix, _, _rest = analyzer.partition("/")
        pl = prefix.lower()
        if pl == "anthropic":
            label = "Anthropic"
        elif pl == "openai":
            label = "OpenAI"
        else:
            label = f"Provider {prefix}"
    elif oai_key and ant_key:
        label = "OpenAI and Anthropic"
    elif ant_key:
        label = "Anthropic"
    elif oai_key:
        label = "OpenAI"

    if label is None:
        if analyzer:
            return f"Analyzer model: {analyzer}"
        return None

    if analyzer:
        return f"{label} · analyzer: {analyzer}"
    return label


def _pin_model_to_agent_env(
    model: str, env_key: str, env_path: Path, agent_name: str
) -> None:
    """Save *model* under *env_key* in the agent's ``.env`` and copy any
    provider credentials from the global environment if not already present.

    This makes the agent env self-contained: OverClaw will always load it
    instead of the global ``.overclaw/.env`` when setting up or optimizing the
    agent.
    """
    updates: dict[str, str] = {env_key: model}

    provider = model.split("/")[0] if "/" in model else ""
    env_key_names = _PROVIDER_ENV_KEYS.get(provider, [])
    if env_key_names:
        existing = (
            {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()}
            if env_path.exists()
            else {}
        )
        for key_name in env_key_names:
            if not existing.get(key_name, "").strip():
                global_val = os.getenv(key_name, "").strip()
                if global_val:
                    updates[key_name] = global_val

    _update_agent_env(env_path, agent_name, updates)


def _collect_agent_provider_config(agent_name: str, console: Console) -> None:
    """Ask which LLM provider the agent uses and save credentials to its per-agent .env."""
    env_path = agent_env_path(agent_name)

    # If already configured, offer to skip
    if env_path.exists() and env_path.stat().st_size > 0:
        existing = {
            k: v
            for k, v in (dotenv_values(env_path) or {}).items()
            if (v or "").strip()
        }
        if existing:
            console.print(
                f"\n  [dim]Agent env already configured at [cyan]{rel(env_path)}[/cyan] "
                f"({len(existing)} variable(s) set).[/dim]"
            )
            provider_hint = _describe_configured_agent_llm_provider(existing)
            if provider_hint:
                console.print(
                    f"  [dim]Looks like this agent is set up for:[/dim] "
                    f"{escape(provider_hint)}"
                )
            if not confirm_option(
                "Reconfigure agent model provider?", default=False, console=console
            ):
                return

    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Agent model provider[/bold]", style=BRAND))
    console.print(
        "  [dim]Which provider does your agent use to call its LLM?\n"
        "  Credentials are saved to [cyan]"
        + overclaw_rel(f"agents/{agent_name}/.env")
        + "[/cyan] and loaded automatically when the agent runs.[/dim]"
    )

    pick = select_option(
        ["OpenAI", "Anthropic", "Other"],
        title="Select provider:",
        default_index=0,
        console=console,
    )

    if pick == 0:  # OpenAI
        existing_key = os.getenv("OPENAI_API_KEY", "").strip()
        if existing_key:
            console.print(
                f"\n  [dim]OPENAI_API_KEY is already set in "
                f"{overclaw_rel('.env')} — using it for this agent.[/dim]"
            )
            key = existing_key
        else:
            console.print("\n  [dim]Enter your OpenAI API key for this agent.[/dim]")
            key = read_api_key_masked("OPENAI_API_KEY")
        _write_agent_env(env_path, agent_name, {"OPENAI_API_KEY": key})
        console.print(
            f"  [bold green]✓[/bold green] Saved [bold]OPENAI_API_KEY[/bold]"
            f" → [dim]{rel(env_path)}[/dim]"
        )

    elif pick == 1:  # Anthropic
        existing_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if existing_key:
            console.print(
                f"\n  [dim]ANTHROPIC_API_KEY is already set in "
                f"{overclaw_rel('.env')} — using it for this agent.[/dim]"
            )
            key = existing_key
        else:
            console.print("\n  [dim]Enter your Anthropic API key for this agent.[/dim]")
            key = read_api_key_masked("ANTHROPIC_API_KEY")
        _write_agent_env(env_path, agent_name, {"ANTHROPIC_API_KEY": key})
        console.print(
            f"  [bold green]✓[/bold green] Saved [bold]ANTHROPIC_API_KEY[/bold]"
            f" → [dim]{rel(env_path)}[/dim]"
        )

    else:  # Other provider
        if confirm_option(
            "Is your provider compatible with the OpenAI SDK?",
            default=True,
            console=console,
        ):
            console.print(
                "\n  [dim]Enter the base URL for your OpenAI-compatible endpoint "
                "(e.g. https://api.example.com/v1).[/dim]"
            )
            base_url = Prompt.ask("  OPENAI_BASE_URL").strip()
            console.print("  [dim]Enter the API key for your provider.[/dim]")
            key = read_api_key_masked("OPENAI_API_KEY")
            _write_agent_env(
                env_path,
                agent_name,
                {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": key},
            )
            console.print(
                f"  [bold green]✓[/bold green] Saved [bold]OPENAI_BASE_URL[/bold] and "
                f"[bold]OPENAI_API_KEY[/bold] → [dim]{rel(env_path)}[/dim]"
            )
        else:
            # Not OpenAI-compatible — create an empty file and wait for the user to fill it
            _write_agent_env(env_path, agent_name, {})
            console.print(
                f"\n  [yellow]Created[/yellow] [bold]{env_path}[/bold]\n\n"
                "  [dim]Open that file and add any environment variables your agent\n"
                "  needs to call its LLM — API keys, base URLs, custom tokens, etc.\n\n"
                "  Example:\n"
                "    MY_PROVIDER_API_KEY=sk-...\n"
                "    MY_PROVIDER_BASE_URL=https://api.example.com/v1[/dim]"
            )
            confirm_option(
                "Confirm once you've added your env variables to the file"
                " (select Yes to continue — safe to skip if no env vars are needed)",
                default=True,
                console=console,
            )
            # load_agent_dotenv() is called immediately after this function returns in
            # main(), so any variables the user just saved will be loaded before the
            # smoke test runs.


def main(
    agent_name: str,
    fast: bool = False,
    policy: str | None = None,
    data: str | None = None,
) -> None:
    load_overclaw_dotenv()

    console = Console()
    console.print()
    render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]OverClaw \u2014 Agent Setup[/bold cyan]\n"
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

    # Ask which provider the agent uses and save its credentials to the agent .env.
    # Skip interactive prompt in fast mode; always load whatever is already on disk.
    console.print()
    console.print(Rule(style="dim"))
    if not fast:
        _collect_agent_provider_config(agent_name, console)
    load_agent_dotenv(agent_name)

    agent_id = _ensure_remote_agent_id(agent_name, agent_path, console)
    use_api_backend = bool(agent_id and get_client() and get_project_id())
    configure_storage(
        agent_path=agent_path,
        agent_id=agent_id,
        backend="api" if use_api_backend else "fs",
    )

    _sigint_flushed = {"done": False}

    def _handle_sigint(_signum, _frame):
        if _sigint_flushed["done"]:
            raise SystemExit(130)
        _sigint_flushed["done"] = True
        console.print(
            "\n  [yellow]Interrupted. Flushing pending Overmind updates...[/yellow]"
        )
        with suppress(Exception):
            _sync_setup_artifacts(agent_name, agent_path, console)
        flush_pending_api_updates(timeout=8.0)
        console.print("  [dim]Pending updates flushed. Exiting.[/dim]\n")
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)
    _validate_agent_entrypoint(agent_path, fn_name, console)

    if policy and not Path(policy).exists():
        console.print(f"\n  [red]Error:[/red] Policy file {policy} does not exist.")
        raise SystemExit(1)

    console.print()
    console.print(Rule(style="dim"))
    _run_beginning_smoke_test(
        agent_path, fn_name, console, fast=fast, data_path=data_opt
    )

    if fast:
        raw_model = os.getenv("ANALYZER_MODEL", "").strip()
        if not raw_model:
            console.print(
                "\n[red]Fast mode requires ANALYZER_MODEL in the environment.[/red]"
            )
            console.print(
                f"[dim]Set it in {overclaw_rel('.env')} or your shell (see .env.example). "
                "Run without --fast to pick a model interactively.[/dim]\n"
            )
            raise SystemExit(1)
        model = normalize_to_litellm_model_id(raw_model) or raw_model
        _pin_model_to_agent_env(
            model, "ANALYZER_MODEL", agent_env_path(agent_name), agent_name
        )

        if not os.getenv("SYNTHETIC_DATAGEN_MODEL", "").strip():
            console.print(
                "\n[red]Fast mode requires SYNTHETIC_DATAGEN_MODEL in the environment.[/red]"
            )
            console.print(
                "[dim]Used by OverClaw optimize when generating synthetic test data. "
                f"Set it in {overclaw_rel('.env')} (see .env.example) or run without --fast.[/dim]\n"
            )
            raise SystemExit(1)

    _clear_existing_eval_spec(agent_name, console, fast=fast)

    if not fast:
        console.print()
        console.print(Rule(style="dim"))
        raw_model = os.getenv("ANALYZER_MODEL", "").strip()
        if raw_model:
            model = normalize_to_litellm_model_id(raw_model) or raw_model
            display = model.split("/", 1)[-1] if "/" in model else model
            console.print(
                f"\n  [dim]ANALYZER_MODEL is already set to[/dim] [cyan]{display}[/cyan]"
            )
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
                f"\n  [dim]No ANALYZER_MODEL set in {overclaw_rel('.env')} — "
                "select a model for agent analysis.[/dim]"
            )
            model = prompt_for_catalog_litellm_model(
                console,
                select_prompt="  Select model for agent analysis (number)",
                env_default=None,
                default_model=DEFAULT_ANALYZER_MODEL,
                no_catalog_prompt="  Enter model for analysis (provider/model)",
            )
        _pin_model_to_agent_env(
            model, "ANALYZER_MODEL", agent_env_path(agent_name), agent_name
        )
        _ensure_provider_api_keys(
            model, agent_env_path(agent_name), agent_name, console
        )
        load_agent_dotenv(agent_name)

    # ---- Phase 1: Agent Analysis ----
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
    analysis = analyze_agent(agent_path, model, console, entrypoint_fn=fn_name)

    # ---- Phase 2: Policy Definition ----
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

    policy_md: str | None = None
    policy_data: dict | None = None

    if fast:
        if policy:
            policy_md, policy_data, _changes = improve_existing_policy(
                analysis, policy, model, console
            )
        else:
            policy_md, policy_data = generate_policy_from_code(analysis, model, console)
            console.print(
                f"  [dim]Auto-generated policy from code. Edit "
                f"[cyan]{rel(default_policy_path(agent_name))}[/cyan] "
                f"to improve optimization quality.[/dim]"
            )

        # ---- Phase 3 (fast): Dataset ----
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        console.print(
            Panel(
                "[bold]Phase 3 \u00b7 Dataset[/bold]\n"
                "[dim]Preparing test data for optimization[/dim]",
                border_style=BRAND,
            )
        )
        _run_data_phase(
            analysis,
            policy_data,
            agent_path,
            agent_name,
            model,
            console,
            fast=True,
            data_path=data_opt,
        )

        spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
        _save_and_finish(spec, agent_name, console, policy_md=policy_md)
        _run_end_smoke_test(agent_name, agent_path, fn_name, console)
        _sync_setup_artifacts(agent_name, agent_path, console)
        return

    pol_path = default_policy_path(agent_name)

    if policy:
        # ---- Path A: User provided a policy document ----
        console.print(
            f"  [dim]Analyzing your policy from [cyan]{policy}[/cyan] "
            f"against agent code…[/dim]\n"
        )
        improved_md, improved_data, change_summary = improve_existing_policy(
            analysis, policy, model, console
        )

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
            from overclaw.setup.policy_generator import generate_policy_from_document

            policy_md, policy_data = generate_policy_from_document(
                analysis, policy, model, console
            )
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
        if confirm_option(
            "Are you satisfied with this policy?", default=True, console=console
        ):
            save_policy(policy_md, pol_path)
            console.print(
                f"\n  [dim]You can always edit the policy later at "
                f"[cyan]{rel(pol_path)}[/cyan][/dim]"
            )
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
        policy_md, policy_data = refine_policy(
            policy_md, policy_data, analysis, model, console
        )

    # ---- Phase 3: Dataset ----
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Phase 3 \u00b7 Dataset[/bold]\n"
            "[dim]Generate or analyze test data for optimization[/dim]",
            border_style=BRAND,
        )
    )
    _run_data_phase(
        analysis,
        policy_data,
        agent_path,
        agent_name,
        model,
        console,
        data_path=data_opt,
    )

    # ---- Phase 4: Evaluation Criteria ----
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

    # Re-display the proposed criteria so the user sees what they're approving.
    # The criteria were first shown during Phase 1 (agent analysis) but policy
    # context may change the user's perspective.
    _display_proposed_criteria(analysis, console)

    iteration = 0
    while True:
        console.print()
        if confirm_option(
            "Are you satisfied with the evaluation criteria?",
            default=True,
            console=console,
        ):
            spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
            _save_and_finish(
                spec,
                agent_name,
                console,
                policy_md=policy_md,
                policy_file_already_saved=True,
            )
            _run_end_smoke_test(agent_name, agent_path, fn_name, console)
            _sync_setup_artifacts(agent_name, agent_path, console)
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
                f"  [dim]Edit [cyan]{spec_out}[/cyan] to fine-tune "
                f"the criteria, then run the optimizer.[/dim]\n"
            )
            _run_end_smoke_test(agent_name, agent_path, fn_name, console)
            _sync_setup_artifacts(agent_name, agent_path, console)
            return

        iteration += 1
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
