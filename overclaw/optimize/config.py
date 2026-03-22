"""Interactive configuration collection for the optimization run."""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table

from overclaw.core.branding import BRAND, render_logo
from overclaw.core.constants import overclaw_rel
from overclaw.core.progress import rel
from overclaw.core.model_picker import prompt_for_catalog_litellm_model
from overclaw.core.models import (
    get_default_models_for_provider,
    get_models_for_provider,
    get_providers,
    normalize_to_litellm_model_id,
)
from overclaw.core.paths import (
    agent_experiments_dir,
    agent_setup_spec_dir,
    load_overclaw_dotenv,
)
from overclaw.core.registry import get_agent_id, resolve_agent


def _agent_eval_spec_path(agent_name: str) -> Path:
    """Eval spec under per-agent setup_spec (see :func:`agent_setup_spec_dir`)."""
    return agent_setup_spec_dir(agent_name) / "eval_spec.json"


def _agent_dataset_path(agent_name: str) -> Path:
    """Dataset under per-agent setup_spec (see :func:`agent_setup_spec_dir`)."""
    return agent_setup_spec_dir(agent_name) / "dataset.json"


def _clear_existing_experiments(
    agent_name: str, console: Console, *, fast: bool = False
) -> None:
    """If experiments/ already has files, ask the user before wiping them (unless fast)."""
    exp_dir = agent_experiments_dir(agent_name)
    if not exp_dir.exists():
        return

    existing = list(exp_dir.rglob("*"))
    files = [f for f in existing if f.is_file() and f.name != ".gitkeep"]
    if not files:
        return

    console.print(
        f"\n  [yellow]Found {len(files)} existing file(s) in experiments/[/yellow]"
    )

    if fast:
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared (fast mode).[/dim]")
        return

    if Confirm.ask("Delete existing experiment results and start fresh?", default=True):
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared.[/dim]")
    else:
        console.print(
            "  [dim]Keeping existing files. New results will overwrite them.[/dim]"
        )


@dataclass
class Config:
    agent_name: str
    agent_path: str
    entrypoint_fn: str
    agent_id: str | None = None
    eval_spec_path: str = ""
    data_path: str | None = None
    model_backtesting: bool = False
    backtest_models: list[str] = field(default_factory=list)
    iterations: int = 5
    analyzer_model: str = ""
    candidates_per_iteration: int = 3
    parallel: bool = True
    max_workers: int = 5
    runs_per_eval: int = 1
    llm_judge_model: str | None = None
    regression_threshold: float = 0.35
    holdout_ratio: float = 0.2
    early_stopping_patience: int = 3
    smoke_test_cases: int = 2
    diagnosis_case_fraction: float = 0.7
    holdout_enforcement: bool = True
    overfit_gap_threshold: float = 10.0
    holdout_weight: float = 0.3
    catastrophic_holdout_threshold: float = 0.5
    max_code_growth_ratio: float = 2.5
    reeval_margin: float = 3.0
    # Multi-file optimization scope: relative paths of files the LLM may
    # modify.  When empty, only the entry file is optimizable.
    optimizable_scope: list[str] = field(default_factory=list)


def _select_backtest_models(console: Console) -> list[str]:
    chosen: list[str] = []

    for provider in get_providers():
        models = get_models_for_provider(provider)
        defaults = get_default_models_for_provider(provider)
        default_indices = [str(i + 1) for i, m in enumerate(models) if m in defaults]

        console.print(f"\n   [bold]{provider.title()}[/bold]")
        for i, name in enumerate(models, 1):
            tag = " [dim](default)[/dim]" if name in defaults else ""
            console.print(f"     [{i}] {name}{tag}")

        raw = (
            Prompt.ask(
                "   Select models (comma-separated numbers, 'all', or 'none')",
                default=",".join(default_indices),
            )
            .strip()
            .lower()
        )

        if raw == "none":
            continue
        if raw == "all":
            chosen.extend(models)
            continue

        for token in raw.split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(models):
                    chosen.append(models[idx])

    return chosen


def _analyzer_default_from_env() -> str | None:
    raw = os.getenv("ANALYZER_MODEL", "").strip()
    if not raw:
        return None
    return normalize_to_litellm_model_id(raw) or raw


def _require_analyzer_model_env_fast(console: Console) -> str:
    """Fast mode must not guess a model; require ANALYZER_MODEL explicitly."""
    raw = os.getenv("ANALYZER_MODEL", "").strip()
    if not raw:
        console.print(
            "\n[red]Fast mode requires ANALYZER_MODEL in the environment.[/red]"
        )
        console.print(
            f"[dim]Set it in {overclaw_rel('.env')} or your shell (see .env.example). "
            "Interactive mode can pick a model without this variable.[/dim]\n"
        )
        raise SystemExit(1)
    return normalize_to_litellm_model_id(raw) or raw


def _collect_config_fast(agent_name: str, console: Console) -> Config:
    """Build Config with the same defaults as accepting every interactive prompt.

    Requires ANALYZER_MODEL. Data is always loaded from disk (prepared
    during ``overclaw setup``).
    """
    agent_path, fn_name = resolve_agent(agent_name)
    cfg = Config(
        agent_name=agent_name,
        agent_path=agent_path,
        entrypoint_fn=fn_name,
        agent_id=get_agent_id(agent_name),
    )

    console.print("\n  [dim]Fast mode: defaults only (no judge, no backtesting).[/dim]")
    console.print(f"  [dim]Agent: {rel(cfg.agent_path)}[/dim]")

    cfg.analyzer_model = _require_analyzer_model_env_fast(console)

    spec_path = _agent_eval_spec_path(cfg.agent_name)
    if not spec_path.exists():
        console.print(
            f"\n[red]No evaluation spec found at [bold]{rel(spec_path)}[/bold].[/red]"
        )
        console.print(
            "Run OverClaw setup first: "
            f"[bold]overclaw setup --fast {agent_name}[/bold] "
            "to analyze your agent and define evaluation criteria.\n"
        )
        raise SystemExit(1)

    _clear_existing_experiments(cfg.agent_name, console, fast=True)

    cfg.eval_spec_path = str(spec_path)

    data_path = _agent_dataset_path(cfg.agent_name)
    if not data_path.exists():
        console.print(
            f"\n[red]No dataset found at [bold]{rel(data_path)}[/bold].[/red]"
        )
        console.print(
            "Run OverClaw setup first: "
            f"[bold]overclaw setup --fast {agent_name}[/bold] "
            "to generate the evaluation dataset.\n"
        )
        raise SystemExit(1)
    cfg.data_path = str(data_path)

    console.print(f"  [dim]Spec:     {rel(spec_path)}[/dim]")
    console.print(f"  [dim]Dataset:  {rel(data_path)}[/dim]")
    console.print(f"  [dim]Model:    {cfg.analyzer_model}[/dim]")

    return cfg


def collect_config(agent_name: str, *, fast: bool = False) -> Config:
    """Collect optimization settings (interactive, or defaults when fast=True)."""
    load_overclaw_dotenv()
    console = Console()
    if fast:
        return _collect_config_fast(agent_name, console)

    agent_path, fn_name = resolve_agent(agent_name)
    cfg = Config(
        agent_name=agent_name,
        agent_path=agent_path,
        entrypoint_fn=fn_name,
        agent_id=get_agent_id(agent_name),
    )

    console.print()
    render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]OverClaw \u2014 Agent Optimizer[/bold cyan]\n"
            "[dim]Automatically improve your AI agent through structured experimentation[/dim]",
            border_style=BRAND,
        )
    )

    console.print(f"\n  [dim]Agent: {rel(cfg.agent_path)}[/dim]")

    # ---- Check for existing experiments ----
    _clear_existing_experiments(cfg.agent_name, console)

    # ---- Eval spec (under per-agent setup_spec) ----
    spec_path = _agent_eval_spec_path(cfg.agent_name)
    if not spec_path.exists():
        console.print(
            f"\n[red]No evaluation spec found at [bold]{rel(spec_path)}[/bold].[/red]"
        )
        console.print(
            "Run OverClaw setup first: "
            f"[bold]overclaw setup {agent_name}[/bold] "
            "to analyze your agent and define evaluation criteria.\n"
        )
        raise SystemExit(1)

    cfg.eval_spec_path = str(spec_path)

    with open(spec_path) as f:
        spec = json.load(f)

    console.print(f"  [dim]Spec:  {rel(spec_path)}[/dim]")

    # Show what the spec contains
    field_count = len(spec.get("output_fields", {}))
    has_tools = bool(spec.get("tool_config", {}).get("expected_tools"))
    has_consistency = bool(spec.get("consistency_rules"))
    features = []
    if has_tools:
        features.append("tool usage scoring")
    if has_consistency:
        features.append("cross-field consistency checks")
    if features:
        console.print(f"  [dim]Spec features: {', '.join(features)}[/dim]")
    console.print(f"  [dim]Scoring {field_count} output fields[/dim]")

    # ---- Data path (auto-resolved from setup_spec/) ----
    data_path = _agent_dataset_path(cfg.agent_name)
    if not data_path.exists():
        console.print(
            f"\n[red]No dataset found at [bold]{rel(data_path)}[/bold].[/red]"
        )
        console.print(
            "Run OverClaw setup first: "
            f"[bold]overclaw setup {agent_name}[/bold] "
            "to generate the evaluation dataset.\n"
        )
        raise SystemExit(1)
    cfg.data_path = str(data_path)
    console.print(f"\n  [dim]Dataset:  {rel(data_path)}[/dim]")

    # ---- Analyzer model ----
    console.print()
    console.print(Rule("[bold]Analyzer Model[/bold]", style=BRAND))
    console.print(
        "   [dim]The analyzer model diagnoses failures and generates improvements.[/dim]"
    )

    env_analyzer = os.getenv("ANALYZER_MODEL", "").strip()
    if env_analyzer:
        normalized = normalize_to_litellm_model_id(env_analyzer)
        display = normalized or env_analyzer
        if Confirm.ask(
            f"Use [cyan]{display}[/cyan] from {overclaw_rel('.env')} as analyzer model?",
            default=True,
        ):
            cfg.analyzer_model = normalized or env_analyzer
        else:
            cfg.analyzer_model = prompt_for_catalog_litellm_model(
                console,
                select_prompt="   Select analyzer model (number)",
                env_default=_analyzer_default_from_env(),
                no_catalog_prompt="   Enter analyzer model",
            )
    else:
        console.print(
            f"   [yellow]No ANALYZER_MODEL found in {overclaw_rel('.env')}[/yellow]"
        )
        cfg.analyzer_model = prompt_for_catalog_litellm_model(
            console,
            select_prompt="   Select analyzer model (number)",
            env_default=_analyzer_default_from_env(),
            no_catalog_prompt="   Enter analyzer model",
        )

    # ---- LLM-as-Judge ----
    console.print()
    console.print(Rule("[bold]Evaluation Settings[/bold]", style=BRAND))
    console.print(
        "   [dim]LLM-as-Judge adds semantic quality scoring alongside mechanical matching.[/dim]"
    )
    use_judge = Confirm.ask(
        "Enable LLM-as-Judge scoring? (adds ~10% eval cost)", default=False
    )
    if use_judge:
        console.print(
            "   [dim]Using the analyzer model for judging. "
            f"You can also set LLM_JUDGE_MODEL in {overclaw_rel('.env')}.[/dim]"
        )
        judge_env = os.getenv("LLM_JUDGE_MODEL", "").strip()
        if judge_env:
            cfg.llm_judge_model = normalize_to_litellm_model_id(judge_env) or judge_env
        else:
            cfg.llm_judge_model = cfg.analyzer_model

    # ---- Optimization settings ----
    console.print()
    console.print(Rule("[bold]Optimization Settings[/bold]", style=BRAND))
    console.print(
        "   [dim]Each iteration: improve from the current best agent, evaluate "
        "candidates on the same dataset and criteria, promote the best accepted "
        "change, then repeat—until this many rounds or early stopping.[/dim]"
    )
    cfg.iterations = IntPrompt.ask("   Optimization iterations", default=5)
    console.print(
        "   [dim]Same eval goal for every variant; parallel passes bias edits toward "
        "tools, core logic, input handling, then system prompt (broader if N is larger). "
        "If N≥3, the last uses a second diagnosis for diversity. Higher N costs more "
        "but improves best-of-N odds.[/dim]"
    )
    cfg.candidates_per_iteration = IntPrompt.ask(
        "   Candidates per iteration (best-of-N)", default=3
    )

    cfg.parallel = Confirm.ask("Run agent in parallel?", default=True)
    if cfg.parallel:
        cfg.max_workers = IntPrompt.ask("   Max parallel workers", default=5)

    # ---- Advanced settings ----
    console.print()
    if Confirm.ask("Configure advanced settings?", default=False):
        console.print()
        console.print(Rule("[bold]Advanced[/bold]", style="dim"))
        cfg.runs_per_eval = IntPrompt.ask(
            "   Runs per evaluation (for stability, 1=fast, 2-3=robust)", default=1
        )
        console.print(
            "   [dim]Regression threshold: max fraction of cases that can regress.[/dim]"
        )
        threshold_str = Prompt.ask("   Regression threshold (0.0-1.0)", default="0.2")
        try:
            cfg.regression_threshold = float(threshold_str)
        except ValueError:
            cfg.regression_threshold = 0.2

        console.print(
            "   [dim]Holdout ratio: fraction of data withheld from the optimizer "
            "to detect overfitting.[/dim]"
        )
        holdout_str = Prompt.ask(
            "   Holdout ratio (0.0-0.4, 0=disabled)", default="0.2"
        )
        try:
            cfg.holdout_ratio = max(0.0, min(0.4, float(holdout_str)))
        except ValueError:
            cfg.holdout_ratio = 0.2

        cfg.holdout_enforcement = Confirm.ask(
            "Enforce holdout (revert if holdout degrades)?", default=True
        )

        console.print(
            "   [dim]Early stopping patience: stop after N consecutive "
            "iterations without improvement.[/dim]"
        )
        patience_str = Prompt.ask(
            "   Early stopping patience (0=disabled)", default="3"
        )
        try:
            cfg.early_stopping_patience = max(0, int(patience_str))
        except ValueError:
            cfg.early_stopping_patience = 3

        console.print(
            "   [dim]Diagnosis case fraction: fraction of training cases shown "
            "to the analyzer (lower = less overfitting risk).[/dim]"
        )
        fraction_str = Prompt.ask("   Diagnosis case fraction (0.5-1.0)", default="0.7")
        try:
            cfg.diagnosis_case_fraction = max(0.5, min(1.0, float(fraction_str)))
        except ValueError:
            cfg.diagnosis_case_fraction = 0.7

    # ---- Summary ----
    console.print()
    table = Table(title="Configuration Summary", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Agent", f"{cfg.agent_name}  [dim]({cfg.agent_path})[/dim]")
    table.add_row("Eval spec", cfg.eval_spec_path)
    table.add_row("Data file", cfg.data_path or "\u2014")
    table.add_row("Analyzer model", cfg.analyzer_model)
    table.add_row(
        "LLM-as-Judge",
        cfg.llm_judge_model or "[dim]disabled[/dim]",
    )
    table.add_row("Iterations", str(cfg.iterations))
    table.add_row("Candidates/iteration", str(cfg.candidates_per_iteration))
    table.add_row(
        "Parallel execution",
        f"Yes ({cfg.max_workers} workers)" if cfg.parallel else "No",
    )
    if cfg.runs_per_eval > 1:
        table.add_row("Runs per eval", str(cfg.runs_per_eval))
    table.add_row("Regression threshold", f"{cfg.regression_threshold:.0%}")
    if cfg.holdout_ratio > 0:
        table.add_row("Holdout ratio", f"{cfg.holdout_ratio:.0%}")
    if cfg.holdout_enforcement:
        table.add_row(
            "Holdout enforcement",
            f"Blended ({1 - cfg.holdout_weight:.0%} train, "
            f"{cfg.holdout_weight:.0%} holdout)",
        )
    if cfg.early_stopping_patience > 0:
        table.add_row("Early stopping", f"After {cfg.early_stopping_patience} stalls")
    if cfg.diagnosis_case_fraction < 1.0:
        table.add_row(
            "Diagnosis visibility", f"{cfg.diagnosis_case_fraction:.0%} of cases"
        )
    console.print(table)
    console.print()

    if not Confirm.ask("Proceed with these settings?", default=True):
        raise SystemExit("Aborted by user.")

    return cfg
