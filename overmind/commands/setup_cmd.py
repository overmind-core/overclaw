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

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.rule import Rule

from overmind import SpanType, attrs, set_tag
from overmind.client import (
    flush_pending_api_updates,
    get_client,
)
from overmind.commands.setup import (
    check_agent_dependencies as _check_agent_dependencies,
)
from overmind.commands.setup import (
    ensure_remote_agent_id as _ensure_remote_agent_id,
)
from overmind.commands.setup import (
    run_beginning_smoke_test as _run_beginning_smoke_test,
)
from overmind.commands.setup import (
    run_end_smoke_test as _run_end_smoke_test,
)
from overmind.commands.setup import (
    smoke_test_agent as _smoke_test_agent,  # noqa: F401 — re-exported for tests
)
from overmind.commands.setup import (
    sync_setup_artifacts as _sync_setup_artifacts,
)
from overmind.commands.setup import (
    validate_agent_entrypoint as _validate_agent_entrypoint,
)
from overmind.commands.setup.smoke_test import (
    resolve_seed_json_files as _resolve_seed_json_files,
)
from overmind.core.constants import overmind_rel
from overmind.core.paths import (
    agent_instrumented_dir,
    agent_setup_spec_dir,
    load_overmind_dotenv,
)
from overmind.core.registry import (
    project_root_from_agent_file,
    resolve_agent,
)
from overmind.optimize.data import (
    generate_diverse_synthetic_data,
    generate_synthetic_data,
    load_data,
    normalize_data_fields,
)
from overmind.optimize.data_analyzer import analyze_seed_coverage, validate_seed_data
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
    render_criteria_table,
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
    ensure_provider_api_keys as _ensure_provider_api_keys,
)
from overmind.utils.provider_keys import (
    update_overmind_env as _update_overmind_env,
)

logger = logging.getLogger("overmind.commands.setup")


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
        _pin_model_to_overmind_env(datagen_model, "SYNTHETIC_DATAGEN_MODEL")
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
            _pin_model_to_overmind_env(datagen_model, "SYNTHETIC_DATAGEN_MODEL")
            _ensure_provider_api_keys(datagen_model, console)
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
        _pin_model_to_overmind_env(datagen_model, "SYNTHETIC_DATAGEN_MODEL")
        _ensure_provider_api_keys(datagen_model, console)
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
        _pin_model_to_overmind_env(datagen_model, "SYNTHETIC_DATAGEN_MODEL")
        _ensure_provider_api_keys(datagen_model, console)
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
    criteria = analysis.get("proposed_criteria", {})
    if not criteria.get("fields"):
        console.print("  [dim]No proposed criteria available.[/dim]")
        return
    render_criteria_table(console, criteria, analysis.get("output_schema", {}))


def _pin_model_to_overmind_env(model: str, env_key: str) -> None:
    """Save *model* under *env_key* in the project ``.overmind/.env``.

    Provider credentials are **not** copied from the live environment — they
    already live in the same file (the user added them via ``overmind init``),
    or in the shell environment, which subsequent ``load_dotenv()`` calls
    respect without override.
    """
    _update_overmind_env({env_key: model})


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
        _pin_model_to_overmind_env(model, "ANALYZER_MODEL")

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
        _pin_model_to_overmind_env(model, "ANALYZER_MODEL")
        _ensure_provider_api_keys(model, console)
        load_overmind_dotenv()

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
