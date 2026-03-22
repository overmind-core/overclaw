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
    overclaw setup <agent-name> --fast
"""

import json
import os
import signal
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule

from overclaw.core.branding import BRAND, render_logo
from overclaw.core.constants import overclaw_rel
from overclaw.core.model_picker import prompt_for_catalog_litellm_model
from overclaw.core.progress import make_spinner_progress, rel
from overclaw.core.models import normalize_to_litellm_model_id
from overclaw.core.policy import default_policy_path, format_for_synthetic_data
from overclaw.optimize.data import (
    generate_diverse_synthetic_data,
    generate_synthetic_data,
    load_data,
)
from overclaw.optimize.data_analyzer import analyze_seed_coverage, validate_seed_data
from overclaw.optimize.evaluator import has_entrypoint
from overclaw.core.paths import (
    agent_setup_spec_dir,
    load_overclaw_dotenv,
)
from overclaw.core.registry import get_agent_id, load_registry, resolve_agent, save_agent
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
                console.print("  [dim]Cleared setup artifacts in Overmind (fast mode).[/dim]")
                return
            if Confirm.ask(
                "Delete existing setup artifacts in Overmind and start fresh?",
                default=True,
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

    if Confirm.ask("Delete existing setup spec files and start fresh?", default=True):
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
):
    spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    save_spec(spec, str(spec_path))
    if policy_md:
        from overclaw.setup.policy_generator import save_policy

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
    if policy_md:
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


def _data_dir(agent_path: str) -> Path:
    """Directory where the user's own seed data lives (read-only for setup)."""
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
        if Confirm.ask(
            f"  Use [cyan]{resolved}[/cyan] from {overclaw_rel('.env')} for data generation?",
            default=True,
        ):
            return resolved

    if fast:
        console.print(
            "\n[red]Fast mode requires SYNTHETIC_DATAGEN_MODEL in the environment.[/red]"
        )
        raise SystemExit(1)

    if not raw:
        console.print(
            f"\n  [dim]SYNTHETIC_DATAGEN_MODEL not set in {overclaw_rel('.env')}[/dim]"
        )
    return prompt_for_catalog_litellm_model(
        console,
        select_prompt="   Select model for data generation (number)",
        env_default=None,
        no_catalog_prompt="   Enter model for data generation (provider/model)",
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
                entrypoint = (load_registry().get(agent_name, {}) or {}).get("entrypoint")
                if entrypoint:
                    save_agent(agent_name, entrypoint, id=new_id)
                console.print("  [dim]Remote agent created and id stored in agents.toml.[/dim]")
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
) -> None:
    """Phase 3: Generate or analyze+augment the test dataset.

    This runs after policy is finalized and before eval criteria generation.
    f"""
    agent_code = analysis.get("_agent_code_section") or Path(agent_path).read_text()
    description = analysis.get("description", "")
    policy_context = format_for_synthetic_data(policy_data) if policy_data else None
    eval_stub = _build_eval_spec_stub(analysis, policy_data)

    data_dir = _data_dir(agent_path)
    existing_json = sorted(data_dir.glob("*.json")) if data_dir.is_dir() else []
    has_seed_data = bool(existing_json)

    # ── Fast mode ──────────────────────────────────────────────────────────
    if fast:
        datagen_model = _resolve_datagen_model(console, fast=True)
        if has_seed_data:
            seed_cases = load_data(str(existing_json[0]))
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
        seed_path = existing_json[0]
        seed_data = load_data(str(seed_path))
        console.print(
            f"  [bold {BRAND}]Seed data found[/bold {BRAND}]"
            f"  [dim]{seed_path.name}  ·  {len(seed_data)} cases[/dim]"
        )
        console.print()

        if not Confirm.ask("  Use this seed data?", default=True):
            # User rejected seed data — offer to generate from scratch
            console.print()
            if not Confirm.ask(
                "  Generate a synthetic dataset from scratch?", default=True
            ):
                console.print("  [dim]Skipping dataset generation.[/dim]")
                return
            datagen_model = _resolve_datagen_model(console)
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
        if not Confirm.ask("  Validate, analyze and augment?", default=True):
            console.print(
                f"  [dim]Using seed data as-is ({len(seed_data)} cases).[/dim]"
            )
            _save_dataset(seed_data, agent_name, console)
            return

        datagen_model = _resolve_datagen_model(console)
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
        # No seed data at all
        console.print("  [dim]No seed data found.[/dim]")
        console.print()
        if not Confirm.ask(
            "  Generate a synthetic dataset from scratch?", default=True
        ):
            console.print("  [dim]Skipping dataset generation.[/dim]")
            return
        datagen_model = _resolve_datagen_model(console)
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
    """Validate, analyze and augment existing seed data (routing already decided by caller)."""
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
        "  How many usersity? (More = broader coverage)",
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


def main(
    agent_name: str,
    fast: bool = False,
    policy: str | None = None,
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
        raw_model = os.getenv("ANALYZER_MODEL", "").strip()
        if raw_model:
            model = normalize_to_litellm_model_id(raw_model) or raw_model
        else:
            model = prompt_for_catalog_litellm_model(
                console,
                select_prompt="   Select model for agent analysis (number)",
                env_default=None,
                no_catalog_prompt="   Enter model for analysis (provider/model)",
            )

    # ---- Phase 1: Agent Analysis ----
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
        console.print(
            Panel(
                "[bold]Phase 3 \u00b7 Dataset[/bold]\n"
                "[dim]Preparing test data for optimization[/dim]",
                border_style=BRAND,
            )
        )
        _run_data_phase(
            analysis, policy_data, agent_path, agent_name, model, console, fast=True
        )

        spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
        _save_and_finish(spec, agent_name, console, policy_md=policy_md)
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
            console.print(
                Panel(
                    "[bold]Suggested Improvements[/bold]\n\n" + change_summary,
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

        display_policy(improved_md, improved_data, console)

        console.print()
        console.print("  [bold]Which policy would you like to use?[/bold]\n")
        console.print(
            f"    [bold {BRAND}][1][/bold {BRAND}] Use the improved policy "
            "[dim](with suggested changes)[/dim]"
        )
        console.print(
            f"    [bold {BRAND}][2][/bold {BRAND}] Keep my original policy [dim](no changes)[/dim]"
        )
        console.print()
        pol_choice = Prompt.ask("  Choice", choices=["1", "2"], default="1")

        if pol_choice == "1":
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
        console.print("  [bold]How would you like to define the agent policy?[/bold]\n")
        console.print(
            f"    [bold {BRAND}][1][/bold {BRAND}] Define policies interactively "
            "[dim](recommended — describe your domain rules)[/dim]"
        )
        console.print(
            f"    [bold {BRAND}][2][/bold {BRAND}] Auto-generate from agent code "
            "[dim](faster, less accurate)[/dim]"
        )
        console.print()
        pol_input_choice = Prompt.ask("  Choice", choices=["1", "2"], default="2")

        if pol_input_choice == "1":
            policy_md, policy_data = elicit_policy(analysis, model, console)
        else:
            console.print("  [dim]Inferring policy from agent code…[/dim]\n")
            policy_md, policy_data = generate_policy_from_code(analysis, model, console)
            display_policy(policy_md, policy_data, console)

    # ---- Policy review / refinement loop ----
    policy_round = 0
    while True:
        console.print()
        if Confirm.ask("Are you satisfied with this policy?", default=True):
            console.print(
                f"\n  [dim]You can always edit the policy later at "
                f"[cyan]{rel(pol_path)}[/cyan][/dim]"
            )
            break

        policy_round += 1
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
    console.print(
        Panel(
            "[bold]Phase 3 \u00b7 Dataset[/bold]\n"
            "[dim]Generate or analyze test data for optimization[/dim]",
            border_style=BRAND,
        )
    )
    _run_data_phase(analysis, policy_data, agent_path, agent_name, model, console)

    # ---- Phase 4: Evaluation Criteria ----
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
        if Confirm.ask("Are you satisfied with the evaluation criteria?", default=True):
            spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
            _save_and_finish(spec, agent_name, console, policy_md=policy_md)
            _sync_setup_artifacts(agent_name, agent_path, console)
            return

        console.print()
        console.print(Rule("[bold]Refinement Options[/bold]", style=BRAND))
        console.print(
            f"    [bold {BRAND}][1][/bold {BRAND}] Refine criteria through conversation"
        )
        console.print(
            f"    [bold {BRAND}][2][/bold {BRAND}] Save now and edit the spec manually"
        )
        console.print()
        choice = Prompt.ask("  Choice", choices=["1", "2"], default="1")

        if choice == "2":
            spec = generate_spec_from_proposal(analysis, policy_data=policy_data)
            _save_and_finish(spec, agent_name, console, policy_md=policy_md)
            spec_out = agent_setup_spec_dir(agent_name) / "eval_spec.json"
            console.print(
                f"  [dim]Edit [cyan]{spec_out}[/cyan] to fine-tune "
                f"the criteria, then run the optimizer.[/dim]\n"
            )
            _sync_setup_artifacts(agent_name, agent_path, console)
            return

        iteration += 1
        console.print()
        console.print(
            Panel(
                f"[bold]Refinement Round {iteration}[/bold]",
                border_style=BRAND,
            )
        )
        analysis = run_questionnaire(analysis, model, console)
