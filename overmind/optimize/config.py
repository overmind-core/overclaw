"""Interactive configuration collection for the optimization run."""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table

from overmind.core.constants import overmind_rel
from overmind.core.paths import (
    agent_env_path,
    agent_experiments_dir,
    agent_setup_spec_dir,
    load_overmind_dotenv,
)
from overmind.core.registry import get_agent_id, resolve_agent
from overmind.utils.display import BRAND, confirm_option, rel, render_logo
from overmind.utils.model_picker import prompt_for_catalog_litellm_model
from overmind.utils.models import (
    DEFAULT_ANALYZER_MODEL,
    get_default_models_for_provider,
    get_models_for_provider,
    get_providers,
    normalize_to_litellm_model_id,
)
from overmind.utils.provider_keys import ensure_provider_api_keys


def _detect_language(agent_path: str) -> str:
    """Auto-detect language from the agent file extension."""
    ext = Path(agent_path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".mts": "typescript",
    }.get(ext, "python")


def _agent_eval_spec_path(agent_name: str) -> Path:
    """Eval spec under per-agent setup_spec (see :func:`agent_setup_spec_dir`)."""
    return agent_setup_spec_dir(agent_name) / "eval_spec.json"


def _agent_dataset_path(agent_name: str) -> Path:
    """Dataset under per-agent setup_spec (see :func:`agent_setup_spec_dir`)."""
    return agent_setup_spec_dir(agent_name) / "dataset.json"


def _clear_existing_experiments(agent_name: str, console: Console, *, fast: bool = False) -> None:
    """If experiments/ already has files, ask the user before wiping them (unless fast)."""
    exp_dir = agent_experiments_dir(agent_name)
    if not exp_dir.exists():
        return

    existing = list(exp_dir.rglob("*"))
    files = [f for f in existing if f.is_file() and f.name != ".gitkeep"]
    if not files:
        return

    console.print(f"\n  [yellow]Found {len(files)} existing file(s) in experiments/[/yellow]")

    if fast:
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared (fast mode).[/dim]")
        return

    if confirm_option(
        "Delete existing experiment results and start fresh?",
        default=True,
        console=console,
    ):
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        console.print("  [dim]Cleared.[/dim]")
    else:
        console.print("  [dim]Keeping existing files. New results will overwrite them.[/dim]")


@dataclass
class Config:
    agent_name: str
    agent_path: str
    entrypoint_fn: str
    agent_id: str | None = None
    eval_spec_path: str = ""
    data_path: str | None = None
    # Auto-detected from agent_path extension; override for ambiguous cases.
    language: str = ""
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
    # Coding agent settings: model and step budget for the agentic codegen loop.
    # When codegen_model is empty, falls back to analyzer_model.
    codegen_model: str = ""
    codegen_max_steps: int = 50
    # Cross-run persistence: carry failure clusters, regression suite, and
    # change history across ``overmind optimize`` invocations.
    cross_run_persistence: bool = True
    # Failure clustering: group failed cases by structural signature and
    # track resolution status across iterations.
    failure_clustering: bool = True
    # Regression gate threshold: max fraction of cross-run regression cases
    # that may fail before a candidate is rejected (0.0 = strict, 1.0 = off).
    regression_gate_threshold: float = 0.2
    # Automated focus targeting: dynamically weight codegen focus areas
    # based on failure analysis instead of static round-robin.
    adaptive_focus: bool = True
    # Whether to include LLM judge scoring during regression suite checks.
    # When False (default), regression checks are faster but may miss
    # semantic quality regressions on judge-heavy specs.
    judge_in_regression: bool = False
    # Read-only context files (globs relative to project root), merged into the
    # bundle but never edited by the coding agent.
    context_scope: list[str] = field(default_factory=list)
    # Read-only files (globs relative to project root) that MUST be present in
    # the bundle / worktree but MUST NOT be modified by candidates. Enforced
    # at accept time via a content diff against the bundle's baseline. Unlike
    # ``context_scope`` (advisory; relies on the analyzer prompt for steering),
    # ``read_only_scope`` is enforced — a candidate whose worktree mutates any
    # listed file is rejected before scoring.
    read_only_scope: list[str] = field(default_factory=list)
    # Extra path globs to exclude from BFS (merged with .overmindignore).
    exclude_scope: list[str] = field(default_factory=list)
    # Extra sys.path-style search directories under the project root that the
    # import resolver should treat as package roots. Supports hyphenated
    # layouts (``python-backend/``) and explicit ``src/`` layouts whose
    # package directories aren't direct children of the project root.
    # When empty, :func:`overmind.utils.code.discover_search_paths`
    # auto-discovers from ``pyproject.toml`` and an existing ``src/`` dir.
    bundle_search_paths: list[str] = field(default_factory=list)
    # Cap how many files import-resolution collects; also bounds prompt size.
    max_resolved_files: int = 24
    max_total_chars: int = 60_000


class SpecValidationError(ValueError):
    """Raised when ``eval_spec.json`` is structurally invalid.

    The message always includes the JSON path of the offending field
    (``consistency_rules[2].field_a`` etc.) so authors can fix their
    spec without trial-and-error. Inherits from :class:`ValueError` so
    callers that already catch ``ValueError`` keep working.
    """


_VALID_RULE_TYPES = frozenset({"correlation", "ordering"})
_VALID_RULE_OPERATORS = frozenset({"<=", "<", ">=", ">"})


def _is_path_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(p, str) for p in value)


def _validate_consistency_rules(rules: object) -> None:
    """Pin the shape that :class:`SpecEvaluator` actually consumes.

    The evaluator iterates and calls ``rule.get(...)`` on each entry, so
    a list of natural-language strings (a tempting LLM output) crashes
    with ``AttributeError: 'str' object has no attribute 'get'`` mid-run.
    We catch that and every adjacent shape error here, with a JSON path
    pointing at the offender so users fix once instead of debugging a
    cryptic stack trace.
    """
    if rules is None:
        return
    if not isinstance(rules, list):
        raise SpecValidationError(
            f"consistency_rules: expected a list, got {type(rules).__name__}"
        )
    for i, rule in enumerate(rules):
        prefix = f"consistency_rules[{i}]"
        if not isinstance(rule, dict):
            raise SpecValidationError(
                f"{prefix}: each rule must be an object with field_a/field_b "
                f"keys; got {type(rule).__name__} "
                f"(natural-language rules should live in policies.md, not "
                f"consistency_rules — those describe machine-checked "
                f"cross-field invariants)"
            )
        for key in ("field_a", "field_b"):
            if key not in rule:
                raise SpecValidationError(f"{prefix}.{key}: required string is missing")
            if not isinstance(rule[key], str) or not rule[key]:
                raise SpecValidationError(
                    f"{prefix}.{key}: must be a non-empty string referring to "
                    f"an output_fields entry"
                )
        if "type" in rule and rule["type"] not in _VALID_RULE_TYPES:
            raise SpecValidationError(
                f"{prefix}.type: must be one of "
                f"{sorted(_VALID_RULE_TYPES)}, got {rule['type']!r}"
            )
        if "operator" in rule and rule["operator"] not in _VALID_RULE_OPERATORS:
            raise SpecValidationError(
                f"{prefix}.operator: must be one of "
                f"{sorted(_VALID_RULE_OPERATORS)}, got {rule['operator']!r}"
            )
        if "penalty" in rule and not isinstance(rule["penalty"], (int, float)):
            raise SpecValidationError(
                f"{prefix}.penalty: must be numeric, got "
                f"{type(rule['penalty']).__name__}"
            )


def _validate_scope(scope: object) -> None:
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise SpecValidationError(
            f"scope: expected an object, got {type(scope).__name__}"
        )
    for key in (
        "optimizable_paths",
        "context_paths",
        "read_only_paths",
        "exclude_paths",
        "search_paths",
    ):
        if key in scope and not _is_path_list(scope[key]):
            raise SpecValidationError(
                f"scope.{key}: must be a list of strings"
            )


def _validate_output_fields(fields: object) -> None:
    """``output_fields`` is the only structurally required block — the
    evaluator does ``self.spec["output_fields"]`` (no .get) and would
    raise KeyError otherwise. Catch the obvious shape problems early."""
    if fields is None:
        # SpecEvaluator will raise KeyError; that's the legacy behaviour and
        # not something we want to mask. The validator is here to catch
        # *shape* errors when the key exists.
        return
    if not isinstance(fields, dict):
        raise SpecValidationError(
            f"output_fields: expected an object, got {type(fields).__name__}"
        )
    for name, cfg in fields.items():
        if not isinstance(cfg, dict):
            raise SpecValidationError(
                f"output_fields.{name}: must be an object describing the "
                f"field (type/description/weight/...)"
            )
        if "weight" in cfg and not isinstance(cfg["weight"], (int, float)):
            raise SpecValidationError(
                f"output_fields.{name}.weight: must be numeric"
            )
        if "values" in cfg and not isinstance(cfg["values"], list):
            raise SpecValidationError(
                f"output_fields.{name}.values: must be a list"
            )


def validate_eval_spec(spec: object) -> None:
    """Validate the structural shape of a loaded ``eval_spec.json``.

    Called from :func:`apply_eval_spec_scope` so every code path that
    loads a spec (fast mode, interactive mode, downstream tooling) gets
    the same gate. Raises :class:`SpecValidationError` with a JSON path
    pointing at the offending field on the first problem found —
    fail-fast keeps the error message focused; users can fix one thing
    at a time without re-reading the whole list.
    """
    if not isinstance(spec, dict):
        raise SpecValidationError(
            f"eval_spec: expected an object at the top level, got "
            f"{type(spec).__name__}"
        )
    _validate_output_fields(spec.get("output_fields"))
    _validate_consistency_rules(spec.get("consistency_rules"))
    _validate_scope(spec.get("scope"))


def _expand_scope_patterns(patterns: list[str], root: Path) -> set[str]:
    """Expand a list of literal paths / globs into concrete relative files.

    Mirrors the same logic :meth:`AgentBundle.from_entry_point` uses
    internally, so the overlap check sees the same set the bundler
    will materialize. Patterns that match no files survive as the
    literal pattern string — that way overlaps between two
    typo'd-but-equal patterns are still caught (string overlap is the
    weakest guarantee; file overlap is the strongest).
    """
    expanded: set[str] = set()
    for pattern in patterns:
        if not pattern:
            continue
        abs_p = root / pattern
        if abs_p.is_file():
            expanded.add(pattern)
            continue
        matched = [p for p in root.glob(pattern) if p.is_file()]
        if not matched:
            expanded.add(pattern)
            continue
        for m in matched:
            try:
                expanded.add(str(m.relative_to(root)))
            except ValueError:
                pass
    return expanded


def _project_root_for(cfg: Config) -> Path | None:
    """Best-effort project root for *cfg* without raising.

    Used by file-level overlap detection. We don't want spec validation
    to fail just because the agent path isn't fully resolvable yet
    (e.g. in unit tests with synthetic configs).
    """
    if not cfg.agent_path:
        return None
    try:
        from overmind.core.registry import project_root_from_agent_file

        agent_path = Path(cfg.agent_path).resolve()
        if not agent_path.exists():
            return None
        root_str = project_root_from_agent_file(str(agent_path))
        return Path(root_str).resolve() if root_str else None
    except Exception:
        return None


def apply_eval_spec_scope(cfg: Config, spec: dict) -> None:
    """Fill scope-related fields from ``eval_spec.json`` when not already set.

    Raises
    ------
    SpecValidationError
        If the spec is structurally invalid (e.g. ``consistency_rules``
        contains plain strings instead of structured rule dicts). The
        single chokepoint here means every loader benefits from the
        same gate.
    ValueError
        If ``optimizable_paths`` and ``read_only_paths`` overlap, either
        as literal patterns OR — when a project root is resolvable —
        after expanding both pattern lists against the filesystem.
        Overlap is almost certainly a configuration mistake (it
        requests both "edit this" and "do not edit this" for the same
        path) and silently choosing one would mask the bug. Fail fast
        at init instead.
    """
    validate_eval_spec(spec)

    scope = spec.get("scope") or {}
    if not cfg.optimizable_scope:
        paths = scope.get("optimizable_paths")
        if paths:
            cfg.optimizable_scope = list(paths)
    if not cfg.context_scope:
        ctx = scope.get("context_paths")
        if ctx:
            cfg.context_scope = list(ctx)
    if not cfg.read_only_scope:
        ro = scope.get("read_only_paths")
        if ro:
            cfg.read_only_scope = list(ro)
    if not cfg.exclude_scope:
        excl = scope.get("exclude_paths")
        if excl:
            cfg.exclude_scope = list(excl)
    if not cfg.bundle_search_paths:
        sp = scope.get("search_paths")
        if sp:
            cfg.bundle_search_paths = list(sp)

    # Tier 1 — literal pattern overlap. Catches the obvious case even
    # when we can't resolve the filesystem.
    literal_overlap = set(cfg.optimizable_scope) & set(cfg.read_only_scope)
    if literal_overlap:
        raise ValueError(
            "eval_spec scope error: the following paths appear in both "
            "optimizable_paths and read_only_paths — pick one: "
            + ", ".join(sorted(literal_overlap))
        )

    # Tier 2 — file-level overlap after glob expansion. Catches the
    # subtler case where the patterns differ as strings but resolve to
    # overlapping files (e.g. ``**/*.py`` vs ``entry.py``). Skipped
    # when no project root is resolvable (synthetic configs in tests).
    root = _project_root_for(cfg)
    if (
        root is not None
        and cfg.optimizable_scope
        and cfg.read_only_scope
    ):
        opt_files = _expand_scope_patterns(cfg.optimizable_scope, root)
        ro_files = _expand_scope_patterns(cfg.read_only_scope, root)
        file_overlap = opt_files & ro_files
        if file_overlap:
            raise ValueError(
                "eval_spec scope error: optimizable_paths and "
                "read_only_paths resolve to overlapping files after glob "
                "expansion — pick one (the patterns differ as strings, "
                "but match the same files): "
                + ", ".join(sorted(file_overlap))
            )


def _select_backtest_models(console: Console) -> list[str]:
    chosen: list[str] = []

    for provider in get_providers():
        models = get_models_for_provider(provider)
        defaults = get_default_models_for_provider(provider)
        default_indices = [str(i + 1) for i, m in enumerate(models) if m in defaults]

        console.print(f"\n  [bold]{provider.title()}[/bold]")
        for i, name in enumerate(models, 1):
            tag = " [dim](default)[/dim]" if name in defaults else ""
            console.print(f"    [{i}] {name}{tag}")

        raw = (
            Prompt
            .ask(
                "  Select models (comma-separated numbers, 'all', or 'none')",
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
        console.print("\n[red]Fast mode requires ANALYZER_MODEL in the environment.[/red]")
        console.print(
            f"[dim]Set it in {overmind_rel('.env')} or your shell (see .env.example). "
            "Interactive mode can pick a model without this variable.[/dim]\n"
        )
        raise SystemExit(1)
    return normalize_to_litellm_model_id(raw) or raw


def _collect_config_fast(
    agent_name: str,
    console: Console,
    *,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> Config:
    """Build Config with the same defaults as accepting every interactive prompt.

    Requires ANALYZER_MODEL. Data is always loaded from disk (prepared
    during ``overmind setup``).
    """
    agent_path, fn_name = resolve_agent(agent_name)
    cfg = Config(
        agent_name=agent_name,
        agent_path=agent_path,
        entrypoint_fn=fn_name,
        agent_id=get_agent_id(agent_name),
        language=_detect_language(agent_path),
    )

    console.print("\n  [dim]Fast mode: defaults only (no judge, no backtesting).[/dim]")
    console.print(f"  [dim]Agent: {rel(cfg.agent_path)}[/dim]")

    cfg.analyzer_model = _require_analyzer_model_env_fast(console)

    spec_path = _agent_eval_spec_path(cfg.agent_name)
    if not spec_path.exists():
        console.print(f"\n[red]No evaluation spec found at [bold]{rel(spec_path)}[/bold].[/red]")
        console.print(
            "Run Overmind setup first: "
            f"[bold]overmind setup --fast {agent_name}[/bold] "
            "to analyze your agent and define evaluation criteria.\n"
        )
        raise SystemExit(1)

    _clear_existing_experiments(cfg.agent_name, console, fast=True)

    cfg.eval_spec_path = str(spec_path)

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    apply_eval_spec_scope(cfg, spec)

    data_path = _agent_dataset_path(cfg.agent_name)
    if not data_path.exists():
        console.print(f"\n[red]No dataset found at [bold]{rel(data_path)}[/bold].[/red]")
        console.print(
            "Run Overmind setup first: "
            f"[bold]overmind setup --fast {agent_name}[/bold] "
            "to generate the evaluation dataset.\n"
        )
        raise SystemExit(1)
    cfg.data_path = str(data_path)

    console.print(f"  [dim]Spec:     {rel(spec_path)}[/dim]")
    console.print(f"  [dim]Dataset:  {rel(data_path)}[/dim]")
    console.print(f"  [dim]Model:    {cfg.analyzer_model}[/dim]")

    if scope_globs:
        cfg.optimizable_scope = list(scope_globs)
    if max_files is not None:
        cfg.max_resolved_files = max_files
    if max_chars is not None:
        cfg.max_total_chars = max_chars

    return cfg


def collect_config(
    agent_name: str,
    *,
    fast: bool = False,
    scope_globs: list[str] | None = None,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> Config:
    """Collect optimization settings (interactive, or defaults when fast=True)."""
    load_overmind_dotenv()
    console = Console()
    if fast:
        return _collect_config_fast(
            agent_name,
            console,
            scope_globs=scope_globs,
            max_files=max_files,
            max_chars=max_chars,
        )

    agent_path, fn_name = resolve_agent(agent_name)
    cfg = Config(
        agent_name=agent_name,
        agent_path=agent_path,
        entrypoint_fn=fn_name,
        agent_id=get_agent_id(agent_name),
        language=_detect_language(agent_path),
    )

    console.print()
    render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]Overmind \u2014 Agent Optimizer[/bold cyan]\n"
            "[dim]Automatically improve your AI agent through structured experimentation[/dim]",
            border_style=BRAND,
        )
    )

    console.print(f"\n  [dim]Agent: {rel(cfg.agent_path)}[/dim]")

    console.print()
    console.print(Rule(style="dim"))

    # ---- Check for existing experiments ----
    _clear_existing_experiments(cfg.agent_name, console)

    # ---- Eval spec (under per-agent setup_spec) ----
    spec_path = _agent_eval_spec_path(cfg.agent_name)
    if not spec_path.exists():
        console.print(f"\n[red]No evaluation spec found at [bold]{rel(spec_path)}[/bold].[/red]")
        console.print(
            "Run Overmind setup first: "
            f"[bold]overmind setup {agent_name}[/bold] "
            "to analyze your agent and define evaluation criteria.\n"
        )
        raise SystemExit(1)

    cfg.eval_spec_path = str(spec_path)

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    apply_eval_spec_scope(cfg, spec)

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
        console.print(f"\n[red]No dataset found at [bold]{rel(data_path)}[/bold].[/red]")
        console.print(
            f"Run Overmind setup first: [bold]overmind setup {agent_name}[/bold] to generate the evaluation dataset.\n"
        )
        raise SystemExit(1)
    cfg.data_path = str(data_path)
    console.print(f"\n  [dim]Dataset:  {rel(data_path)}[/dim]")

    # ---- Analyzer model ----
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Analyzer Model[/bold]", style=BRAND))
    console.print("  [dim]The analyzer model diagnoses failures and generates improvements.[/dim]")

    env_analyzer = os.getenv("ANALYZER_MODEL", "").strip()
    if env_analyzer:
        normalized = normalize_to_litellm_model_id(env_analyzer)
        display = normalized or env_analyzer
        if confirm_option(
            f"Use {display} from {overmind_rel('.env')} as analyzer model?",
            default=True,
            console=console,
        ):
            cfg.analyzer_model = normalized or env_analyzer
        else:
            cfg.analyzer_model = prompt_for_catalog_litellm_model(
                console,
                select_prompt="  Select analyzer model (number)",
                env_default=_analyzer_default_from_env(),
                default_model=DEFAULT_ANALYZER_MODEL,
                no_catalog_prompt="  Enter analyzer model",
            )
            ensure_provider_api_keys(
                cfg.analyzer_model,
                agent_env_path(cfg.agent_name),
                cfg.agent_name,
                console,
            )
    else:
        console.print(f"  [yellow]No ANALYZER_MODEL found in {overmind_rel('.env')}[/yellow]")
        cfg.analyzer_model = prompt_for_catalog_litellm_model(
            console,
            select_prompt="  Select analyzer model (number)",
            env_default=_analyzer_default_from_env(),
            default_model=DEFAULT_ANALYZER_MODEL,
            no_catalog_prompt="  Enter analyzer model",
        )
        ensure_provider_api_keys(cfg.analyzer_model, agent_env_path(cfg.agent_name), cfg.agent_name, console)

    # ---- LLM-as-Judge ----
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Evaluation Settings[/bold]", style=BRAND))
    console.print("  [dim]LLM-as-Judge adds semantic quality scoring alongside mechanical matching.[/dim]")
    use_judge = confirm_option(
        "Enable LLM-as-Judge scoring? (adds ~10% eval cost)",
        default=False,
        console=console,
    )
    if use_judge:
        console.print(
            "  [dim]Using the analyzer model for judging. "
            f"You can also set LLM_JUDGE_MODEL in {overmind_rel('.env')}.[/dim]"
        )
        judge_env = os.getenv("LLM_JUDGE_MODEL", "").strip()
        if judge_env:
            cfg.llm_judge_model = normalize_to_litellm_model_id(judge_env) or judge_env
        else:
            cfg.llm_judge_model = cfg.analyzer_model

    # ---- Optimization settings ----
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Optimization Settings[/bold]", style=BRAND))
    console.print(
        "  [dim]Each iteration: improve from the current best agent, evaluate "
        "candidates on the same dataset and criteria, promote the best accepted "
        "change, then repeat—until this many rounds or early stopping.[/dim]"
    )
    cfg.iterations = IntPrompt.ask("  Optimization iterations", default=5)
    console.print(
        "  [dim]Same eval goal for every variant; parallel passes bias edits toward "
        "tools, core logic, input handling, then system prompt (broader if N is larger). "
        "If N≥3, the last uses a second diagnosis for diversity. Higher N costs more "
        "but improves best-of-N odds.[/dim]"
    )
    cfg.candidates_per_iteration = IntPrompt.ask("  Candidates per iteration (best-of-N)", default=3)

    cfg.parallel = confirm_option("Run agent in parallel?", default=True, console=console)
    if cfg.parallel:
        cfg.max_workers = IntPrompt.ask("  Max parallel workers", default=5)

    # ---- Advanced settings ----
    console.print()
    if confirm_option("Configure advanced settings?", default=False, console=console):
        console.print()
        console.print(Rule("[bold]Advanced[/bold]", style="dim"))
        cfg.runs_per_eval = IntPrompt.ask("  Runs per evaluation (for stability, 1=fast, 2-3=robust)", default=1)
        console.print("  [dim]Regression threshold: max fraction of cases that can regress.[/dim]")
        threshold_str = Prompt.ask("  Regression threshold (0.0-1.0)", default="0.2")
        try:
            cfg.regression_threshold = float(threshold_str)
        except ValueError:
            cfg.regression_threshold = 0.2

        console.print("  [dim]Holdout ratio: fraction of data withheld from the optimizer to detect overfitting.[/dim]")
        holdout_str = Prompt.ask("  Holdout ratio (0.0-0.4, 0=disabled)", default="0.2")
        try:
            cfg.holdout_ratio = max(0.0, min(0.4, float(holdout_str)))
        except ValueError:
            cfg.holdout_ratio = 0.2

        cfg.holdout_enforcement = confirm_option(
            "Enforce holdout (revert if holdout degrades)?",
            default=True,
            console=console,
        )

        console.print("  [dim]Early stopping patience: stop after N consecutive iterations without improvement.[/dim]")
        patience_str = Prompt.ask("  Early stopping patience (0=disabled)", default="3")
        try:
            cfg.early_stopping_patience = max(0, int(patience_str))
        except ValueError:
            cfg.early_stopping_patience = 3

        console.print(
            "  [dim]Diagnosis case fraction: fraction of training cases shown "
            "to the analyzer (lower = less overfitting risk).[/dim]"
        )
        fraction_str = Prompt.ask("  Diagnosis case fraction (0.5-1.0)", default="0.7")
        try:
            cfg.diagnosis_case_fraction = max(0.5, min(1.0, float(fraction_str)))
        except ValueError:
            cfg.diagnosis_case_fraction = 0.7

        cfg.cross_run_persistence = confirm_option(
            "Enable cross-run persistence? (carry knowledge across optimize runs)",
            default=True,
            console=console,
        )

        cfg.failure_clustering = confirm_option(
            "Enable failure clustering? (group failures by root cause)",
            default=True,
            console=console,
        )

        cfg.adaptive_focus = confirm_option(
            "Enable adaptive focus targeting? (auto-weight codegen focus areas)",
            default=True,
            console=console,
        )

    # ---- Summary ----
    console.print()
    console.print(Rule(style="dim"))
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
            f"Blended ({1 - cfg.holdout_weight:.0%} train, {cfg.holdout_weight:.0%} holdout)",
        )
    if cfg.early_stopping_patience > 0:
        table.add_row("Early stopping", f"After {cfg.early_stopping_patience} stalls")
    if cfg.diagnosis_case_fraction < 1.0:
        table.add_row("Diagnosis visibility", f"{cfg.diagnosis_case_fraction:.0%} of cases")
    features: list[str] = []
    if cfg.cross_run_persistence:
        features.append("cross-run persistence")
    if cfg.failure_clustering:
        features.append("failure clustering")
    if cfg.adaptive_focus:
        features.append("adaptive focus")
    if features:
        table.add_row("Smart features", ", ".join(features))
    console.print(table)
    console.print()

    if not confirm_option("Proceed with these settings?", default=True, console=console):
        raise SystemExit("Aborted by user.")

    if scope_globs:
        cfg.optimizable_scope = list(scope_globs)
    if max_files is not None:
        cfg.max_resolved_files = max_files
    if max_chars is not None:
        cfg.max_total_chars = max_chars

    return cfg
