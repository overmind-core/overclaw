"""
Core optimization loop.

Loads the target agent, runs it against a dataset, collects traces,
sends everything to the analyzer, applies improvements, and iterates.

Features:
- Full tool trace capture and propagation to the analyzer
- Regression-aware acceptance (case-level delta checking)
- Multi-run evaluation for statistical stability
- Agentic UX with rich progress reporting
"""

from __future__ import annotations

import contextvars
import difflib
import json
import logging
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from overmind import SpanType, attrs, set_tag
from overmind import start_span as otel_span
from overmind.core.paths import (
    agent_experiments_dir,
    agent_instrumented_dir,
    agent_run_state_path,
)
from overmind.core.registry import project_root_from_agent_file
from overmind.optimize.analyzer import (
    compute_focus_weights,
    format_component_weights,
    generate_candidates,
)
from overmind.optimize.config import Config
from overmind.optimize.data import load_data
from overmind.optimize.evaluator import (
    SpecEvaluator,
    load_evaluator,
)
from overmind.optimize.execution_backend import (
    BackendOutput,
    BackendPlan,
    build_default_plan,
    should_try_next,
)
from overmind.optimize.failure_registry import (
    FailureRegistry,
    format_clusters_for_diagnosis,
)
from overmind.optimize.pipeline.scoring import (
    compute_complexity_penalty as _scoring_compute_complexity_penalty,
)
from overmind.optimize.pipeline.scoring import (
    count_conditional_branches as _scoring_count_conditional_branches,
)
from overmind.optimize.pipeline.scoring import (
    count_function_defs as _scoring_count_function_defs,
)
from overmind.optimize.pipeline.scoring import (
    detect_data_leakage as _scoring_detect_data_leakage,
)
from overmind.optimize.pipeline.scoring import (
    prompt_size as _scoring_prompt_size,
)
from overmind.optimize.run_state import RunState, RunSummary
from overmind.optimize.runner import AgentRunner, Language, RunnerConfig
from overmind.optimize.trace_reader import (
    ParsedTrace,
    attach_shadow_provenance,
    parse_trace_file,
)
from overmind.tracing import force_flush_traces, observe_safe
from overmind.utils.atomic_io import atomic_write_json
from overmind.utils.code import AgentBundle
from overmind.utils.display import BRAND, confirm_option, make_spinner_progress, rel
from overmind.utils.instrument import deinstrument_source
from overmind.utils.policy import (
    format_for_codegen,
    format_for_diagnosis,
    format_for_judge,
    load_policy_data,
)


def _is_subpath(child: Path, parent: Path) -> bool:
    """Return True if *child* is at or below *parent* (both must be resolved)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


_OTLP_EVAL_SPEC_TAG_MAX = 100_000


def _eval_spec_json_for_otlp_tag(spec: dict) -> str:
    """Serialize eval_spec for ``overmind.optimize.eval_spec`` — shrink if over OTLP limits."""

    def dumps(d: dict) -> str:
        return json.dumps(d, separators=(",", ":"), default=str)

    s = dumps(spec)
    if len(s) <= _OTLP_EVAL_SPEC_TAG_MAX:
        return s

    lean_keys = (
        "input_schema",
        "output_fields",
        "output_schema",
        "structure_weight",
        "total_points",
        "tool_config",
        "tool_usage_weight",
        "consistency_rules",
        "optimizable_elements",
        "fixed_elements",
        "entrypoint_fn",
        "agent_description",
        "description",
        "model",
        "analyzer_model",
    )
    lean = {k: spec[k] for k in lean_keys if k in spec}
    s = dumps(lean)
    if len(s) <= _OTLP_EVAL_SPEC_TAG_MAX:
        return s

    for drop in ("consistency_rules", "optimizable_elements", "fixed_elements"):
        if drop in lean:
            lean[drop] = []
            s = dumps(lean)
            if len(s) <= _OTLP_EVAL_SPEC_TAG_MAX:
                return s

    minimal = {
        k: lean[k]
        for k in (
            "input_schema",
            "output_fields",
            "output_schema",
            "structure_weight",
            "total_points",
            "tool_config",
            "tool_usage_weight",
            "entrypoint_fn",
            "agent_description",
            "description",
            "model",
            "analyzer_model",
        )
        if k in lean
    }
    return dumps(minimal)


class Optimizer:
    """Runs the full optimization pipeline for an agent."""

    @observe_safe("optimizer.init")
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        set_tag(attrs.AGENT_NAME, config.agent_name)
        set_tag(attrs.OPTIMIZE_AGENT_PATH, config.agent_path)
        set_tag(attrs.OPTIMIZE_ITERATIONS, config.iterations)
        set_tag(attrs.OPTIMIZE_ANALYZER_MODEL, config.analyzer_model)
        set_tag(attrs.OPTIMIZE_CANDIDATES_PER_ITERATION, config.candidates_per_iteration)
        set_tag(attrs.OPTIMIZE_PARALLEL, config.parallel)

        # Load policy from eval spec for injection into pipeline stages
        with open(config.eval_spec_path) as _f:
            _spec = json.load(_f)
        set_tag(attrs.OPTIMIZE_EVAL_SPEC, _eval_spec_json_for_otlp_tag(_spec))
        self._policy_data = load_policy_data(_spec)
        self._policy_diagnosis = format_for_diagnosis(self._policy_data or {})
        self._policy_codegen = format_for_codegen(self._policy_data or {})
        self._policy_judge = format_for_judge(self._policy_data or {})

        set_tag(attrs.SETUP_POLICY_PATH, getattr(config, "llm_judge_model", "disabled"))

        self.evaluator: SpecEvaluator = load_evaluator(
            config.eval_spec_path,
            llm_judge_model=getattr(config, "llm_judge_model", None),
            policy_judge_rubric=self._policy_judge,
        )
        self.results: list[dict] = []
        self.best_score: float = 0.0
        self.best_code: str = ""
        self.best_case_scores: list[float] = []
        self.failed_attempts: list[dict] = []
        self.successful_changes: list[dict] = []
        self.output_dir = agent_experiments_dir(config.agent_name)
        self.traces_dir = self.output_dir / "traces"
        self.analysis_dir = self.output_dir / "analysis"
        self.backtest_results: dict[str, dict] = {}
        self.stall_count: int = 0
        self._baseline_code: str = ""
        self._baseline_train_score: float = 0.0
        self.accepted_snapshots: list[dict] = []

        # Multi-file state
        self._bundle: AgentBundle | None = None
        self._best_files: dict[str, str] = {}
        self._baseline_files: dict[str, str] = {}

        # Resolve agent copy created by ``overmind setup``.
        # Setup copies the project tree (plain, no decorators); the optimizer
        # instruments all .py files with @observe() so overmind-sdk traces
        # are captured for both the baseline and candidate runs.
        self._instrumented_agent_path = self._resolve_instrumented_path()
        self._instrument_agent_copy()

        # --- Process-isolated agent runner ---
        self._runner = self._build_runner(self._instrumented_agent_path, config.entrypoint_fn)
        self._logger = logging.getLogger("overmind.optimize.optimizer")

        # --- Cross-run state & failure clustering ---
        use_persistence = getattr(config, "cross_run_persistence", True)
        if use_persistence:
            self._run_state = RunState.load(
                agent_run_state_path(config.agent_name),
                config.agent_name,
            )
            self.failed_attempts = self._run_state.seed_failed_attempts()
            self.successful_changes = self._run_state.seed_successful_changes()
        else:
            self._run_state = RunState(
                agent_run_state_path(config.agent_name),
                config.agent_name,
            )

        use_clustering = getattr(config, "failure_clustering", True)
        if use_clustering and use_persistence:
            self._failure_registry = self._run_state.failure_registry
        else:
            self._failure_registry = FailureRegistry()

        self._run_id = self._run_state.begin_run()
        self._session_failed: list[dict] = []
        self._session_successful: list[dict] = []

    def _resolve_instrumented_path(self) -> str:
        """Return the path to the agent copy if it exists.

        ``overmind setup`` copies the **project root** tree to
        ``.overmind/agents/<name>/instrumented/`` as a plain copy.
        ``_instrument_agent_copy`` then adds ``@observe()`` decorators
        to all ``.py`` files so traces are captured.  The entry file
        lives at its original relative path inside that tree
        (e.g. ``instrumented/evals/harness.py``).

        Falls back to the original ``config.agent_path`` when no copy is
        present.
        """
        inst_dir = agent_instrumented_dir(self.config.agent_name)
        original = Path(self.config.agent_path).resolve()

        pr = project_root_from_agent_file(self.config.agent_path)
        if pr is not None:
            entry_relpath = original.relative_to(pr)
        else:
            entry_relpath = Path(original.name)

        candidate = inst_dir / entry_relpath
        if candidate.is_file():
            return str(candidate)

        return self.config.agent_path

    @observe_safe("optimizer.instrument_agent_copy")
    def _instrument_agent_copy(self) -> None:
        """Add ``@observe()`` instrumentation to all ``.py`` files in the agent copy.

        Called once at optimizer init so that both the baseline and all
        subsequent candidate runs produce overmind-sdk trace spans.
        Only modifies the copy under ``.overmind/``; original files are
        never touched.  No-op when the copy doesn't exist (fallback to
        original agent path).
        """
        from overmind.utils.instrument import instrument_directory

        inst_dir = agent_instrumented_dir(self.config.agent_name)
        if not inst_dir.is_dir():
            return
        resolved = Path(self._instrumented_agent_path).resolve()
        if not _is_subpath(resolved, inst_dir.resolve()):
            return
        instrument_directory(inst_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @observe_safe("optimizer.run", SpanType.WORKFLOW)
    def run(self):
        self._logger.info(
            f"Optimizer.run starting agent={getattr(self.config, 'agent_name', '?')} "
            f"iterations={self.config.iterations} parallel={getattr(self.config, 'parallel', False)} "
            f"max_workers={getattr(self.config, 'max_workers', None)}"
        )
        set_tag(attrs.AGENT_NAME, self.config.agent_name)
        set_tag(attrs.OPTIMIZE_ITERATIONS, self.config.iterations)
        set_tag(attrs.OPTIMIZE_PARALLEL, self.config.parallel)
        set_tag(attrs.OPTIMIZE_MAX_WORKERS, self.config.max_workers)
        set_tag(attrs.OPTIMIZE_ANALYZER_MODEL, self.config.analyzer_model)
        set_tag(attrs.OPTIMIZE_CANDIDATES_PER_ITERATION, self.config.candidates_per_iteration)

        self._setup_output_dirs()
        dataset = self._load_dataset()
        self._logger.info(f"Loaded dataset with {len(dataset)} cases")

        # Split into train (optimizer sees) and holdout (final generalization check)
        holdout_ratio = getattr(self.config, "holdout_ratio", 0.2)
        train_set, holdout_set = Optimizer._split_dataset(dataset, holdout_ratio)
        set_tag(attrs.OPTIMIZE_DATASET_TOTAL, len(dataset))
        set_tag(attrs.OPTIMIZE_DATASET_TRAIN, len(train_set))
        set_tag(attrs.OPTIMIZE_DATASET_HOLDOUT, len(holdout_set))
        self._logger.info(f"Dataset split: train={len(train_set)} holdout={len(holdout_set)} ratio={holdout_ratio:.2f}")

        self.console.print()
        self.console.print(Rule(f"[bold {BRAND}]Overmind Agent Optimizer[/bold {BRAND}]", style=BRAND))
        agent_label = (
            f"{self.config.agent_name}  [dim]({self.config.agent_path})[/dim]"
            if getattr(self.config, "agent_name", "")
            else self.config.agent_path
        )
        info_lines = f"  [dim]Agent:[/dim]  {agent_label}\n  [dim]Cases:[/dim]  {len(dataset)} total"
        if holdout_set:
            info_lines += f" ({len(train_set)} train, {len(holdout_set)} holdout)"
        info_lines += f"\n  [dim]Model:[/dim]  {self.config.analyzer_model}"
        if self._policy_data:
            n_rules = len(self._policy_data.get("domain_rules", self._policy_data.get("decision_rules", [])))
            n_constraints = len(
                self._policy_data.get("output_constraints", self._policy_data.get("hard_constraints", []))
            )
            info_lines += f"\n  [dim]Policy:[/dim] {n_rules} rule(s), {n_constraints} constraint(s)"
        if self._run_state.has_prior_runs:
            n_runs = len(self._run_state.run_history)
            n_reg = len(self._run_state.regression_cases)
            n_clusters = len(self._failure_registry.clusters)
            best_prior = self._run_state.best_prior_score
            info_lines += (
                f"\n  [dim]History:[/dim] {n_runs} prior run(s), "
                f"best {best_prior:.1f}, "
                f"{n_reg} regression case(s), "
                f"{n_clusters} failure cluster(s)"
            )
        self.console.print(info_lines)

        # ---- Phase 1: Baseline ----
        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()
        self.console.print(
            Panel(
                "[bold]Phase 1 · Establishing Baseline[/bold]\n"
                "[dim]Running your agent on training cases to measure starting performance[/dim]",
                border_style=BRAND,
            )
        )
        baseline_code = Path(self.config.agent_path).read_text()
        self._baseline_code = baseline_code

        # Build the agent bundle for multi-file context
        self._bundle = self._build_bundle()
        _opt_logger = logging.getLogger(__name__)
        if self._bundle:
            n_files = len(self._bundle.original_files)
            total_chars = sum(len(s) for s in self._bundle.original_files.values())
            n_opt = self._bundle.optimizable_file_count()
            n_ro = max(0, n_files - n_opt)
            cap = getattr(self.config, "max_total_chars", 60_000)
            _opt_logger.info(f"bundle: files={n_files} chars={total_chars}/{cap} optimizable={n_opt} read_only={n_ro}")
            if self._bundle.is_multi_file():
                self.console.print(
                    f"  [dim]Bundle:[/dim] {n_files} file(s) resolved, {n_opt} optimizable, "
                    f"{n_ro} read-only, {total_chars}/{cap} chars"
                )
            else:
                self.console.print(f"  [dim]Bundle:[/dim] {n_files} file(s), {total_chars}/{cap} chars")

        # Provision agent environment (install deps into venv / node_modules)
        with make_spinner_progress(self.console, transient=True) as _prov:
            _prov.add_task(f"  Provisioning {self._runner.language.value} environment…")
            self._ensure_runner_env()
        self.console.print(f"  [dim]Runtime:[/dim] {self._runner.language.value} (subprocess isolation)")

        baseline_eval, _, baseline_items = self._run_agent_on_dataset(
            self._instrumented_agent_path, train_set, "baseline"
        )
        self.best_score = baseline_eval["avg_total"]
        self._baseline_train_score = self.best_score
        self.best_code = baseline_code
        self.best_case_scores = [item["score"]["total"] for item in baseline_items]
        set_tag(attrs.OPTIMIZE_BASELINE_SCORE, float(self.best_score))

        # Emit a short-lived "milestone" span that flushes immediately with
        # the baseline score so the platform UI can surface it without
        # waiting for the (long-running) root ``overmind_optimize`` span to
        # end.  The OTel BatchSpanProcessor only exports finished spans, so
        # tags written on the root span aren't visible until the whole run
        # completes — a child span like this gives us per-phase real-time
        # progress for free.
        with otel_span(
            "optimizer.baseline_complete",
            attributes={
                attrs.OPTIMIZE_PHASE: "baseline_complete",
                attrs.OPTIMIZE_BASELINE_SCORE: float(self.best_score),
                attrs.OPTIMIZE_DATASET_TRAIN: len(train_set),
                attrs.OPTIMIZE_DATASET_HOLDOUT: len(holdout_set),
                attrs.PROGRESS_PHASE: "baseline_complete",
                attrs.PROGRESS_CURRENT: 0,
                attrs.PROGRESS_TOTAL: self.config.iterations,
            },
        ):
            pass

        self._log_result("baseline", baseline_eval, "keep", "Initial baseline")
        self._print_eval(baseline_eval, "Baseline (train)", prev_evaluation=None)

        # Ingest baseline failures into the failure registry
        if getattr(self.config, "failure_clustering", True):
            baseline_case_results = self._build_case_results(baseline_items, train_set)
            touched = self._failure_registry.ingest_iteration(
                0,
                baseline_case_results,
                self.evaluator.spec,
            )
            if touched:
                open_n = self._failure_registry.get_open_count()
                self.console.print(f"  [dim]Failure clusters: {len(touched)} identified, {open_n} open[/dim]")

        # Baseline diagnostics
        self._print_baseline_diagnostics(baseline_eval, baseline_items)

        # Track multi-file state
        if self._bundle:
            self._baseline_files = dict(self._bundle.original_files)
            self._best_files = dict(self._bundle.original_files)

        # Working copy
        _ext = Path(self.config.agent_path).suffix or ".py"
        working_path = self.output_dir / f"agent_working{_ext}"
        working_path.write_text(baseline_code)
        working_dir: Path | None = None
        if self._bundle and self._bundle.is_multi_file():
            working_dir = self.output_dir / "agent_working"
            self._write_file_set(working_dir, self._best_files)

        # ---- Phase 2: Optimization loop ----
        n_candidates = getattr(self.config, "candidates_per_iteration", 3)
        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Phase 2 · Optimization Loop[/bold]\n"
                f"[dim]{self.config.iterations} iterations \u00d7 "
                f"{n_candidates} candidates · Diagnosing failures, "
                f"generating fixes, testing improvements[/dim]",
                border_style=BRAND,
            )
        )

        latest_case_results = self._build_case_results(baseline_items, train_set)
        latest_eval = baseline_eval

        for i in range(1, self.config.iterations + 1):
            with otel_span(
                "optimizer.iteration",
                attributes={
                    attrs.OPTIMIZE_ITERATION: i,
                    attrs.OPTIMIZE_TOTAL_ITERATIONS: self.config.iterations,
                    attrs.OPTIMIZE_BEST_SCORE_BEFORE: float(self.best_score),
                    attrs.OPTIMIZE_STALL_COUNT: int(self.stall_count),
                    # Generic progress tags so the platform can render a
                    # uniform progress bar without optimizer-specific keys.
                    attrs.PROGRESS_PHASE: "iteration",
                    attrs.PROGRESS_CURRENT: i,
                    attrs.PROGRESS_TOTAL: self.config.iterations,
                },
            ):
                self._logger.info(
                    f"STAGE BEGIN optimizer.iteration iter={i}/{self.config.iterations} "
                    f"best_score={self.best_score:.4f} stall_count={self.stall_count}"
                )
                self.console.print()
                self.console.print(
                    Rule(
                        f"[bold cyan]Iteration {i}/{self.config.iterations}[/bold cyan]",
                        style="cyan",
                    )
                )

                current_code = working_path.read_text()

                # Temperature annealing
                t_start, t_end = 0.8, 0.4
                temperature = t_start - (t_start - t_end) * (i - 1) / max(self.config.iterations - 1, 1)

                # Stall detection: increase exploration
                if self.stall_count >= 2:
                    temperature = min(temperature + 0.2, 1.0)
                    self.console.print("  [yellow]Detected stall — increasing exploration[/yellow]")
                set_tag(attrs.OPTIMIZE_TEMPERATURE, float(temperature))

                # --- Compute focus weights & cluster context ---
                _cluster_ctx = ""
                _component_ctx = ""
                _focus_weights: dict[str, float] | None = None

                if getattr(self.config, "failure_clustering", True):
                    priority_clusters = self._failure_registry.get_priority_clusters()
                    if priority_clusters:
                        _cluster_ctx = format_clusters_for_diagnosis(priority_clusters)

                if getattr(self.config, "adaptive_focus", True):
                    _focus_weights = compute_focus_weights(
                        latest_case_results,
                        latest_eval,
                        self.evaluator.spec,
                        self._failure_registry,
                        self.successful_changes,
                        self.failed_attempts,
                        is_multi_file=(self._bundle is not None and self._bundle.is_multi_file()),
                    )
                    _component_ctx = format_component_weights(_focus_weights)

                    top_focus = max(_focus_weights, key=_focus_weights.get)  # type: ignore[arg-type]
                    top_pct = _focus_weights[top_focus] * 100
                    self.console.print(f"  [dim]Focus targeting:[/dim] {top_focus} ({top_pct:.0f}%)")

                # --- Step 1: Diagnosis & candidate generation ---
                self.console.print(
                    f"  [dim]Step 1:[/dim] Analyzing failures and generating "
                    f"{n_candidates} candidates (temp={temperature:.2f})"
                )
                with make_spinner_progress(self.console) as progress:
                    task = progress.add_task("  Diagnosing and generating improvements…")

                    try:
                        # Build agent_files for the coding agent: use the
                        # current best multi-file state, or fall back to the
                        # single entry file.
                        _agent_files = self._current_agent_files(current_code)

                        candidates = generate_candidates(
                            current_code,
                            case_results=latest_case_results,
                            evaluation_results=latest_eval,
                            model=self.config.analyzer_model,
                            eval_spec=self.evaluator.spec,
                            failed_attempts=self.failed_attempts,
                            successful_changes=self.successful_changes,
                            allow_model_change=bool(self.config.model_backtesting and self.config.backtest_models),
                            num_candidates=n_candidates,
                            temperature=temperature,
                            diagnosis_case_fraction=getattr(self.config, "diagnosis_case_fraction", 0.7),
                            iteration_seed=i * 7919,
                            policy_context=self._policy_diagnosis,
                            policy_constraints=self._policy_codegen,
                            entrypoint_fn=self.config.entrypoint_fn,
                            bundle=self._bundle,
                            agent_files=_agent_files,
                            codegen_model=getattr(self.config, "codegen_model", ""),
                            codegen_max_steps=getattr(self.config, "codegen_max_steps", 50),
                            cluster_context=_cluster_ctx,
                            component_weights_context=_component_ctx,
                            focus_weights=_focus_weights,
                        )
                    except Exception as exc:
                        self._logger.exception(f"Iteration {i} analyzer error")
                        progress.update(task, description=f"  [red]Analyzer error: {exc}")
                        self._log_result(
                            f"iter_{i:03d}",
                            latest_eval,
                            "error",
                            f"Analyzer failed: {exc}",
                        )
                        self.stall_count += 1
                        continue

                    progress.update(task, completed=True)
                    self._logger.info(f"Iteration {i} generated {len(candidates)} candidate(s)")
                    set_tag(attrs.OPTIMIZE_N_CANDIDATES_GENERATED, len(candidates))

                # Show diagnosis if available (full text; wrap inside panel)
                for cand in candidates:
                    diag = cand.get("diagnosis")
                    if diag and diag.get("root_cause"):
                        self.console.print(
                            Panel(
                                Text(diag["root_cause"].strip(), overflow="fold"),
                                title="[dim]Diagnosis[/dim]",
                                border_style="dim",
                                expand=False,
                            )
                        )
                        tool_issues = diag.get("tool_issues", [])
                        if tool_issues:
                            ti_table = Table(
                                show_header=True,
                                header_style="bold yellow",
                                border_style="dim",
                                title="[dim]Tool issues[/dim]",
                            )
                            ti_table.add_column("Issue", overflow="fold")
                            for ti in tool_issues:
                                issue_txt = ti.get("issue", "") or "—"
                                ti_table.add_row(Text(str(issue_txt), overflow="fold"))
                            self.console.print(ti_table)
                        break

                # --- Step 2: Validate candidates ---
                self.console.print("  [dim]Step 2:[/dim] Validating candidates")
                valid = []
                for idx, cand in enumerate(candidates):
                    code = cand.get("updated_code")
                    bundle_updates = cand.get("bundle_updates")
                    method = cand.get("method", "unknown")

                    # Resolve bundle updates into a unified code string
                    if not code and bundle_updates and self._bundle:
                        resolved = self._resolve_bundle_candidate(bundle_updates)
                        if resolved is not None:
                            code = resolved["entry_code"]
                            cand["updated_code"] = code
                            cand["_resolved_files"] = resolved["files"]
                        else:
                            self.console.print(
                                f"    Candidate {idx + 1} ({method}): [yellow]bundle splice validation failed[/yellow]"
                            )
                            continue

                    if not code:
                        debug = cand.get("_debug", {})
                        if isinstance(debug, list):
                            debug = debug[0] if debug else {}
                        reason = "no code"
                        if debug.get("error"):
                            reason = f"error: {debug['error'][:60]}"
                        elif debug.get("finish_reason") == "length":
                            reason = "response truncated"
                        elif debug.get("response_len", 0) > 0:
                            reason = "parsing failed"
                        self.console.print(f"    Candidate {idx + 1} ({method}): [yellow]{reason}[/yellow]")
                        continue
                    if not self._validate_code(code):
                        ext = Path(self.config.agent_path).suffix or ".txt"
                        failed_path = self.output_dir / f"failed_iter_{i:03d}_c{idx}{ext}"
                        failed_path.write_text(code)
                        self.console.print(
                            f"    Candidate {idx + 1} ({method}): [yellow]syntax/interface validation failed[/yellow]"
                        )
                        continue
                    valid.append((idx, cand))

                if valid:
                    summary_keys: list[tuple[str, ...]] = []
                    for _vidx, vcand in valid:
                        sugs = vcand.get("suggestions") or []
                        summary_keys.append(tuple(str(s) for s in sugs))
                    distinct_summaries = set(summary_keys)
                    all_same_summary = len(distinct_summaries) == 1
                    shared_text = summary_keys[0] if all_same_summary else ()

                    if all_same_summary and shared_text:
                        self.console.print(
                            Panel(
                                Text("; ".join(shared_text), overflow="fold"),
                                title="[dim]Planned changes (shared for all variants)[/dim]",
                                border_style="dim",
                                expand=False,
                            )
                        )

                    cand_table = Table(
                        show_header=True,
                        header_style="bold",
                        border_style="dim",
                        title="[dim]Validated candidates[/dim]",
                        show_lines=not all_same_summary,
                    )
                    cand_table.add_column("#", justify="right", style="cyan", width=4)
                    cand_table.add_column(
                        "Codegen focus",
                        style="magenta",
                        overflow="fold",
                        max_width=44,
                    )
                    if all_same_summary:
                        for vidx, vcand in valid:
                            cand_table.add_row(str(vidx + 1), vcand.get("method", "unknown"))
                    else:
                        cand_table.add_column("Change summary", overflow="fold")
                        for vidx, vcand in valid:
                            method = vcand.get("method", "unknown")
                            sugs = vcand.get("suggestions") or []
                            if sugs:
                                summary_cell = Text("; ".join(str(s) for s in sugs), overflow="fold")
                            else:
                                summary_cell = Text("—", style="dim")
                            cand_table.add_row(str(vidx + 1), method, summary_cell)
                    self.console.print(cand_table)

                set_tag(attrs.OPTIMIZE_N_CANDIDATES_VALID, len(valid))
                if not valid:
                    self.console.print("  [yellow]No valid candidates this iteration.[/yellow]")
                    self._log_result(f"iter_{i:03d}", latest_eval, "skip", "No valid candidates")
                    self.stall_count += 1
                    continue

                # --- Step 2.5: Smoke test (quick catastrophic-failure filter) ---
                smoke_n = getattr(self.config, "smoke_test_cases", 2)
                if smoke_n > 0 and len(train_set) > smoke_n and self.best_score > 0:
                    smoke_set = random.Random(i * 6271).sample(train_set, smoke_n)
                    smoke_threshold = self.best_score * 0.4
                    surviving: list[tuple[int, dict]] = []

                    for idx, cand in valid:
                        tmp_path = self._write_candidate_to_disk(cand)
                        try:
                            s_eval, _, _ = self._run_agent_on_dataset(
                                str(tmp_path),
                                smoke_set,
                                f"smoke_{i:03d}_c{idx}",
                            )
                        except Exception:
                            s_eval = None
                        finally:
                            self._cleanup_candidate(tmp_path, cand)

                        if s_eval is None:
                            self.console.print(f"    Candidate {idx + 1}: [red]crashed in smoke test[/red]")
                        elif s_eval["avg_total"] >= smoke_threshold:
                            surviving.append((idx, cand))
                        else:
                            self.console.print(
                                f"    Candidate {idx + 1}: "
                                f"[yellow]failed smoke test "
                                f"({s_eval['avg_total']:.1f} < "
                                f"{smoke_threshold:.1f})[/yellow]"
                            )
                    if surviving:
                        valid = surviving
                    elif valid:
                        self.console.print(
                            "  [yellow]All candidates failed smoke test, proceeding with full eval anyway.[/yellow]"
                        )

                # --- Step 3: Evaluate candidates ---
                self.console.print(f"  [dim]Step 3:[/dim] Evaluating {len(valid)} candidate(s) against test cases")
                best_cand = None
                best_cand_eval = None
                best_cand_score = -1.0
                best_cand_items = None
                best_cand_case_scores: list[float] = []

                for orig_idx, cand in valid:
                    with otel_span(
                        "optimizer.evaluate_candidate",
                        attributes={
                            attrs.OPTIMIZE_ITERATION: i,
                            attrs.OPTIMIZE_CANDIDATE_INDEX: int(orig_idx),
                            attrs.OPTIMIZE_CANDIDATE_METHOD: str(cand.get("method", "")),
                        },
                    ):
                        tmp_path = self._write_candidate_to_disk(cand)
                        try:
                            runs_per = getattr(self.config, "runs_per_eval", 1)
                            self._logger.debug(
                                f"Iter {i} candidate {orig_idx}: evaluating (runs_per={runs_per}, path={tmp_path})"
                            )
                            set_tag(attrs.OPTIMIZE_RUNS_PER_EVAL, int(runs_per))
                            if runs_per > 1:
                                c_eval, c_items = self._run_multi_eval(
                                    str(tmp_path),
                                    train_set,
                                    f"iter_{i:03d}_c{orig_idx}",
                                    runs_per,
                                )
                            else:
                                c_eval, _, c_items = self._run_agent_on_dataset(
                                    str(tmp_path),
                                    train_set,
                                    f"iter_{i:03d}_c{orig_idx}",
                                )
                        except Exception:
                            self._logger.exception(f"Iter {i} candidate {orig_idx} crashed during evaluation")
                            c_eval = None
                            c_items = None
                        finally:
                            self._cleanup_candidate(tmp_path, cand)

                        if c_eval is None:
                            self.console.print(f"    Candidate {orig_idx + 1}: [red]crashed[/red]")
                            set_tag(attrs.OPTIMIZE_ITERATION_DECISION, "crash")
                            continue

                        c_score = c_eval["avg_total"]
                        self.console.print(f"    Candidate {orig_idx + 1}: [cyan]{c_score:.1f}[/cyan] / 100")
                        set_tag(attrs.OPTIMIZE_CANDIDATE_SCORE, float(c_score))

                        complexity_penalty = self._compute_complexity_penalty(
                            cand["updated_code"],
                            train_set=train_set,
                            raw_score=c_score,
                        )
                        adjusted_score = c_score - complexity_penalty
                        set_tag(
                            attrs.OPTIMIZE_COMPLEXITY_PENALTY,
                            float(complexity_penalty),
                        )
                        set_tag(
                            attrs.OPTIMIZE_CANDIDATE_ADJUSTED_SCORE,
                            float(adjusted_score),
                        )
                        if complexity_penalty > 0:
                            self.console.print(
                                f"      [dim]Complexity penalty: -{complexity_penalty:.1f} → {adjusted_score:.1f}[/dim]"
                            )

                        if adjusted_score > best_cand_score:
                            best_cand = cand
                            best_cand_eval = c_eval
                            best_cand_score = adjusted_score
                            best_cand_items = c_items
                            best_cand_case_scores = [item["score"]["total"] for item in c_items]

                if best_cand is None or best_cand_eval is None:
                    self.console.print("  [yellow]All candidates crashed. Reverting.[/yellow]")
                    working_path.write_text(self.best_code)
                    self._log_result(
                        f"iter_{i:03d}",
                        {"avg_total": 0},
                        "crash",
                        "All candidates crashed",
                    )
                    self.stall_count += 1
                    continue

                # --- Step 3.5: Confirmation re-eval for close calls ---
                reeval_margin = getattr(self.config, "reeval_margin", 3.0)
                runs_per = getattr(self.config, "runs_per_eval", 1)
                is_close_call = (
                    runs_per <= 1
                    and best_cand_score <= self.best_score
                    and best_cand_score > self.best_score - reeval_margin
                    and best_cand_eval is not None
                )
                if is_close_call and best_cand is not None:
                    self.console.print(
                        f"  [dim]Close call ({best_cand_score:.1f} vs "
                        f"{self.best_score:.1f}) — confirming with 3 runs[/dim]"
                    )
                    tmp_path = self._write_candidate_to_disk(best_cand)
                    try:
                        confirm_eval, confirm_items = self._run_multi_eval(
                            str(tmp_path),
                            train_set,
                            f"iter_{i:03d}_confirm",
                            3,
                        )
                        confirmed_score = confirm_eval["avg_total"]
                        confirm_penalty = self._compute_complexity_penalty(
                            best_cand["updated_code"],
                            train_set=train_set,
                            raw_score=confirmed_score,
                        )
                        confirmed_adjusted = confirmed_score - confirm_penalty
                        self.console.print(
                            f"  [dim]Confirmed score: {confirmed_score:.1f} (adjusted: {confirmed_adjusted:.1f})[/dim]"
                        )
                        best_cand_eval = confirm_eval
                        best_cand_score = confirmed_adjusted
                        best_cand_items = confirm_items
                        best_cand_case_scores = [item["score"]["total"] for item in confirm_items]
                    except Exception:  # noqa: S110
                        pass
                    finally:
                        self._cleanup_candidate(tmp_path, best_cand)

                # --- Step 4: Regression-aware acceptance ---
                desc = "; ".join(best_cand.get("suggestions", [])[:2])
                prev_eval = dict(latest_eval)

                accept, reason = self._check_acceptance(
                    best_cand_score,
                    best_cand_case_scores,
                    best_cand_items,
                    train_set,
                    candidate_eval=best_cand_eval,
                )

                # Cross-run regression gate: check that previously-fixed failures
                # stay fixed (only when the within-run gate passed).
                if accept and self._run_state.regression_cases:
                    reg_fail = self._check_regression_suite(best_cand, train_set)
                    reg_threshold = getattr(self.config, "regression_gate_threshold", 0.2)
                    n_reg = len(self._run_state.regression_cases)
                    if reg_fail > n_reg * reg_threshold:
                        accept = False
                        reason = (
                            f"Regression gate: {reg_fail}/{n_reg} previously-fixed "
                            f"cases regressed (threshold: {reg_threshold:.0%})"
                        )
                    elif reg_fail > 0:
                        reason = f"{reason}; regression gate: {reg_fail}/{n_reg} minor regressions (within threshold)"

                # --- Step 4.5: Periodic holdout probe (overfitting early detection) ---
                holdout_probe_interval = getattr(self.config, "holdout_probe_interval", 3)
                if accept and holdout_set and i % holdout_probe_interval == 0 and best_cand is not None:
                    self.console.print(f"  [dim]Holdout probe (iteration {i})…[/dim]")
                    probe_path = self._write_candidate_to_disk(best_cand)
                    try:
                        probe_eval, _, _ = self._run_agent_on_dataset(
                            str(probe_path),
                            holdout_set,
                            f"holdout_probe_{i:03d}",
                        )
                        probe_score = probe_eval["avg_total"]
                        train_gap = best_cand_score - probe_score
                        overfit_threshold = getattr(self.config, "holdout_probe_gap_threshold", 15.0)
                        if train_gap > overfit_threshold:
                            accept = False
                            reason = (
                                f"Holdout probe: train={best_cand_score:.1f} vs "
                                f"holdout={probe_score:.1f} "
                                f"(gap={train_gap:.1f} > {overfit_threshold:.1f}) "
                                f"— likely overfitting"
                            )
                            self.console.print(
                                f"    [yellow]Holdout gap {train_gap:.1f} exceeds threshold — rejecting[/yellow]"
                            )
                        else:
                            self.console.print(
                                f"    [dim]Holdout probe OK: train={best_cand_score:.1f}, "
                                f"holdout={probe_score:.1f} "
                                f"(gap={train_gap:.1f})[/dim]"
                            )
                    except Exception:  # noqa: S110
                        pass
                    finally:
                        self._cleanup_candidate(probe_path, best_cand)

                # Per-dimension scores for this iteration — stamped on
                # the OTel iteration span so the OTLP ingest can project
                # them onto the corresponding ``JobIteration`` row.
                dim_scores = {
                    key: float(best_cand_eval.get(key, 0)) for _, key in self.evaluator.get_dimension_labels()
                }

                set_tag(attrs.OPTIMIZE_ACCEPTED, bool(accept))
                set_tag(
                    attrs.OPTIMIZE_ITERATION_DECISION,
                    "keep" if accept else "discard",
                )
                set_tag(attrs.OPTIMIZE_ITERATION_REASON, str(reason or ""))
                set_tag(attrs.OPTIMIZE_ITERATION_SCORE, float(best_cand_score))
                set_tag(
                    attrs.OPTIMIZE_ITERATION_IMPROVEMENT,
                    float(best_cand_score - self.best_score),
                )
                if dim_scores:
                    set_tag(attrs.OPTIMIZE_ITERATION_DIMENSION_SCORES, dim_scores)
                # Stamp the full best-candidate code and its human-readable
                # suggestions so OTLP can project them onto the matching
                # ``JobIteration`` row (``agent_code`` + ``description``).
                cand_code = best_cand.get("updated_code") or ""
                if cand_code:
                    set_tag(attrs.OPTIMIZE_ITERATION_AGENT_CODE, cand_code)
                cand_suggestions = best_cand.get("suggestions") or []
                if cand_suggestions:
                    set_tag(
                        attrs.OPTIMIZE_ITERATION_SUGGESTIONS,
                        [str(s) for s in cand_suggestions],
                    )

                if accept:
                    improvement = best_cand_score - self.best_score
                    self.console.print(
                        f"\n  [bold green]\u2713 Accepted: {self.best_score:.1f} \u2192 "
                        f"{best_cand_score:.1f} (+{improvement:.1f})[/bold green]"
                    )
                    if reason:
                        self.console.print(f"    [dim]{reason}[/dim]")

                    resolved_files = best_cand.get("_resolved_files")
                    prev_files_snapshot = dict(self._best_files) if self._best_files else None

                    if resolved_files and prev_files_snapshot:
                        changed_files = [fp for fp, src in resolved_files.items() if prev_files_snapshot.get(fp) != src]
                        if changed_files:
                            files_text = "  ".join(f"[cyan]{fp}[/cyan]" for fp in sorted(changed_files))
                            self.console.print(f"    [dim]Updated:[/dim]  {files_text}")

                    self._animate_code_update(
                        self.best_code,
                        best_cand["updated_code"],
                        resolved_files=resolved_files,
                        prev_files=prev_files_snapshot,
                    )
                    dim_deltas = self._compute_dimension_deltas(latest_eval, best_cand_eval)
                    change_record = {
                        "suggestions": best_cand.get("suggestions", []),
                        "improvement": (f"+{improvement:.1f} ({self.best_score:.1f} \u2192 {best_cand_score:.1f})"),
                        "score_before": self.best_score,
                        "score_after": best_cand_score,
                        "dimension_deltas": dim_deltas,
                        "method": best_cand.get("method", ""),
                    }
                    self.successful_changes.append(change_record)
                    self._session_successful.append(change_record)
                    self.best_score = best_cand_score
                    self.best_code = best_cand["updated_code"]
                    self.best_case_scores = best_cand_case_scores
                    working_path.write_text(self.best_code)

                    # Update multi-file state
                    if best_cand.get("_resolved_files"):
                        self._best_files.update(best_cand["_resolved_files"])
                        if working_dir:
                            self._write_file_set(working_dir, self._best_files)
                        self._rebuild_bundle()

                    self.accepted_snapshots.append({
                        "code": self.best_code,
                        "files": (dict(self._best_files) if self._best_files else None),
                        "train_score": best_cand_score,
                        "iteration": i,
                    })
                    latest_eval = best_cand_eval
                    latest_case_results = self._build_case_results(best_cand_items, train_set)

                    # Update failure registry: ingest new results and check resolutions
                    if getattr(self.config, "failure_clustering", True):
                        self._failure_registry.ingest_iteration(
                            i,
                            latest_case_results,
                            self.evaluator.spec,
                        )
                        newly_resolved = self._failure_registry.update_resolution_status(
                            i,
                            latest_case_results,
                            self.evaluator.spec,
                            change_summary=desc,
                        )
                        if newly_resolved:
                            self.console.print(
                                f"    [green]\u2713 Resolved {len(newly_resolved)} failure cluster(s)[/green]"
                            )
                            self._promote_resolved_to_regression(
                                newly_resolved,
                                latest_case_results,
                                train_set,
                                i,
                            )

                    self._log_result(f"iter_{i:03d}", best_cand_eval, "keep", desc)
                    self.stall_count = 0
                else:
                    self.console.print(
                        f"\n  [red]\u2717 Rejected: {best_cand_score:.1f} vs best {self.best_score:.1f}[/red]"
                    )
                    if reason:
                        self.console.print(f"    [dim]{reason}[/dim]")
                    working_path.write_text(self.best_code)
                    self._log_result(f"iter_{i:03d}", best_cand_eval, "discard", desc)
                    dim_deltas = self._compute_dimension_deltas(latest_eval, best_cand_eval)
                    fail_record = {
                        "suggestions": best_cand.get("suggestions", []),
                        "score": best_cand_score,
                        "reason": reason or f"No improvement ({best_cand_score:.1f})",
                        "dimension_deltas": dim_deltas,
                        "method": best_cand.get("method", ""),
                    }
                    self.failed_attempts.append(fail_record)
                    self._session_failed.append(fail_record)
                    self.stall_count += 1

                self._print_eval(
                    best_cand_eval,
                    f"Iteration {i} (best candidate)",
                    prev_evaluation=prev_eval,
                )

                self._logger.info(
                    f"STAGE END   optimizer.iteration iter={i}/{self.config.iterations} "
                    f"best_score={self.best_score:.4f} stall_count={self.stall_count} "
                    f"best_cand_score={(float(best_cand_eval.get('avg_total', 0)) if best_cand_eval else 0.0):.4f}"
                )

                # Early stopping
                patience = getattr(self.config, "early_stopping_patience", 3)
                if patience > 0 and self.stall_count >= patience:
                    self._logger.info(f"Early stopping triggered stall_count={self.stall_count} patience={patience}")
                    self.console.print(
                        f"\n  [yellow]Early stopping: {self.stall_count} consecutive "
                        f"iterations without improvement "
                        f"(patience={patience}).[/yellow]"
                    )
                    break

        # Save best agent
        _ext = Path(self.config.agent_path).suffix or ".py"
        best_path = self.output_dir / f"best_agent{_ext}"
        best_path.write_text(self.best_code)

        # Save multi-file output when applicable
        if self._best_files and self._bundle and self._bundle.is_multi_file():
            best_dir = self.output_dir / "best_agent"
            self._write_file_set(best_dir, self._best_files)
            self.console.print(f"  [dim]Multi-file output: {best_dir}/ ({len(self._best_files)} files)[/dim]")

        # ---- Holdout evaluation (blended-score generalization check) ----
        if holdout_set:
            self.console.print()
            self.console.print(Rule(style="dim"))
            self.console.print()
            self.console.print(
                Panel(
                    "[bold]Holdout Evaluation · Generalization Check[/bold]\n"
                    "[dim]Testing the optimized agent on unseen cases "
                    "using blended train/holdout scoring[/dim]",
                    border_style="yellow",
                )
            )
            holdout_eval, _, _ = self._run_agent_on_dataset(str(best_path), holdout_set, "holdout")
            holdout_score = holdout_eval["avg_total"]

            baseline_holdout_eval, _, _ = self._run_agent_on_dataset(
                self._instrumented_agent_path, holdout_set, "holdout_baseline"
            )
            baseline_holdout_score = baseline_holdout_eval["avg_total"]
            train_improvement = self.best_score - self._baseline_train_score
            holdout_improvement = holdout_score - baseline_holdout_score

            holdout_w = getattr(self.config, "holdout_weight", 0.3)
            blended_improvement = (1 - holdout_w) * train_improvement + holdout_w * holdout_improvement

            self.console.print(
                f"  [bold]Train improvement:[/bold]   "
                f"+{train_improvement:.1f} ({self._baseline_train_score:.1f} "
                f"\u2192 {self.best_score:.1f})"
            )
            self.console.print(
                f"  [bold]Holdout improvement:[/bold] "
                f"+{holdout_improvement:.1f} ({baseline_holdout_score:.1f} "
                f"\u2192 {holdout_score:.1f})"
            )
            self.console.print(
                f"  [bold]Blended improvement:[/bold] "
                f"+{blended_improvement:.1f} "
                f"(weight: {1 - holdout_w:.0%} train, {holdout_w:.0%} holdout)"
            )

            overfit_gap = train_improvement - holdout_improvement
            holdout_enforcement = getattr(self.config, "holdout_enforcement", True)
            catastrophic_threshold = getattr(self.config, "catastrophic_holdout_threshold", 0.5)

            is_catastrophic = (
                baseline_holdout_score > 0 and holdout_score < baseline_holdout_score * catastrophic_threshold
            )

            needs_rollback = holdout_enforcement and (is_catastrophic or blended_improvement < 0)

            reverted = False
            rollback_target = None
            if needs_rollback:
                if is_catastrophic:
                    self.console.print(
                        "\n  [bold red]Catastrophic holdout degradation — "
                        "rolling back.[/bold red]\n"
                        f"  Holdout dropped to {holdout_score:.1f}, below "
                        f"{catastrophic_threshold:.0%} of baseline "
                        f"({baseline_holdout_score:.1f})"
                    )
                else:
                    self.console.print(
                        "\n  [bold red]Blended improvement is negative — "
                        "rolling back.[/bold red]\n"
                        f"  Train gained +{train_improvement:.1f} but holdout "
                        f"lost {holdout_improvement:.1f}, "
                        f"blended: {blended_improvement:+.1f}"
                    )

                rollback_target = self._rollback_to_best_snapshot(
                    best_path,
                    holdout_set,
                    baseline_holdout_score,
                    holdout_w,
                    catastrophic_threshold,
                )

                if rollback_target:
                    self.console.print(
                        f"\n  [green]Selected iteration "
                        f"{rollback_target['iteration']} snapshot "
                        f"(train: {rollback_target['train_score']:.1f}, "
                        f"holdout: {rollback_target['holdout_score']:.1f}, "
                        f"blended: {rollback_target['blended_improvement']:+.1f})"
                        f"[/green]"
                    )
                    self.best_code = rollback_target["code"]
                    self.best_score = rollback_target["train_score"]
                    best_path.write_text(self.best_code)
                    if rollback_target.get("files"):
                        self._best_files = dict(rollback_target["files"])
                        self._rebuild_bundle()
                    holdout_eval = rollback_target["holdout_eval"]
                    holdout_score = rollback_target["holdout_score"]
                    holdout_improvement = holdout_score - baseline_holdout_score
                    train_improvement = self.best_score - self._baseline_train_score
                    blended_improvement = (1 - holdout_w) * train_improvement + holdout_w * holdout_improvement
                    overfit_gap = train_improvement - holdout_improvement
                    reverted = True
                else:
                    self.console.print(
                        "\n  [bold red]No intermediate snapshot has positive "
                        "blended improvement "
                        "\u2014 reverting to original baseline.[/bold red]"
                    )
                    best_path.write_text(self._baseline_code)
                    self.best_code = self._baseline_code
                    self.best_score = self._baseline_train_score
                    if self._baseline_files:
                        self._best_files = dict(self._baseline_files)
                        self._rebuild_bundle()
                    reverted = True
            elif overfit_gap > 5.0:
                self.console.print(
                    f"\n  [bold yellow]Warning: Overfitting gap detected."
                    f"[/bold yellow] "
                    f"Train gained +{train_improvement:.1f} but holdout "
                    f"only +{holdout_improvement:.1f} "
                    f"(gap: {overfit_gap:.1f}). "
                    f"Blended improvement is still positive "
                    f"({blended_improvement:+.1f}), so keeping the result."
                )
            else:
                self.console.print("\n  [green]Holdout performance confirms generalization.[/green]")

            self._print_eval(
                holdout_eval,
                "Holdout",
                prev_evaluation=baseline_holdout_eval,
            )

            self._holdout_results = {
                "train_improvement": self.best_score - self._baseline_train_score,
                "holdout_improvement": holdout_improvement,
                "blended_improvement": blended_improvement,
                "holdout_score": holdout_score,
                "baseline_holdout_score": baseline_holdout_score,
                "overfit_gap": overfit_gap,
                "holdout_weight": holdout_w,
                "reverted": reverted,
                "rollback_iteration": (rollback_target["iteration"] if rollback_target else None),
            }
            set_tag(attrs.OPTIMIZE_HOLDOUT_SCORE, float(holdout_score))
            set_tag(
                attrs.OPTIMIZE_HOLDOUT_BASELINE_SCORE,
                float(baseline_holdout_score),
            )
            set_tag(attrs.OPTIMIZE_HOLDOUT_IMPROVEMENT, float(holdout_improvement))
            set_tag(attrs.OPTIMIZE_BLENDED_IMPROVEMENT, float(blended_improvement))
            set_tag(attrs.OPTIMIZE_HOLDOUT_REVERTED, bool(reverted))
            set_tag(attrs.OPTIMIZE_OVERFIT_GAP, float(overfit_gap))

        # ---- Phase 3: Model backtesting (optional) ----
        if self.config.model_backtesting and self.config.backtest_models:
            backtest_data = holdout_set if holdout_set else train_set
            self.console.print()
            self.console.print(Rule(style="dim"))
            self.console.print()
            self.console.print(
                Panel(
                    "[bold]Phase 3 · Model Backtesting[/bold]\n"
                    f"[dim]Testing optimized agent across different models "
                    f"on {'holdout' if holdout_set else 'training'} data "
                    f"({len(backtest_data)} cases)[/dim]",
                    border_style="magenta",
                )
            )
            self._run_backtesting(backtest_data)

        # ---- Phase 4: Report ----
        set_tag(attrs.OPTIMIZE_FINAL_BEST_SCORE, float(self.best_score))
        # Mirror of ``OPTIMIZE_FINAL_BEST_SCORE`` under the legacy key the
        # OTLP ingest checks first when computing ``Job.best_score``.
        set_tag(attrs.OPTIMIZE_REPORT_BEST_SCORE, float(self.best_score))
        # Headline improvement = best - baseline.  Drives ``Job.improvement``
        # (OTLP reads ``overmind.optimize.report_improvement``).
        baseline = float(getattr(self, "_baseline_train_score", 0.0) or 0.0)
        set_tag(
            attrs.OPTIMIZE_REPORT_IMPROVEMENT,
            float(self.best_score - baseline),
        )
        # Run-level summary counters that the OTLP ingest folds into
        # ``Job.result["summary"]`` (alongside the per-iteration
        # ``stall_count`` already stamped on each iteration span).
        set_tag(
            attrs.OPTIMIZE_TOTAL_ACCEPTED,
            len(getattr(self, "_session_successful", [])),
        )
        set_tag(
            attrs.OPTIMIZE_TOTAL_REJECTED,
            len(getattr(self, "_session_failed", [])),
        )
        set_tag(attrs.OPTIMIZE_STALL_COUNT, int(self.stall_count))

        self._generate_report()
        # Stamp the final artefacts on the run span so OTLP can persist
        # them onto ``Job.report_markdown`` / ``Job.best_agent_code``.
        try:
            report_path = self.output_dir / "report.md"
            if report_path.exists():
                set_tag(
                    attrs.OPTIMIZE_REPORT_MARKDOWN,
                    report_path.read_text(encoding="utf-8"),
                )
        except Exception:
            self._logger.exception("optimize: failed to stamp report.md on span")
        if self.best_code:
            set_tag(attrs.OPTIMIZE_BEST_AGENT_CODE, self.best_code)
        # Terminal lifecycle marker — flips ``Job.status`` to ``completed``
        # on the OTLP side (works for both legacy CLI and skill flows,
        # without OTLP having to guess from span topology).
        set_tag(attrs.OPTIMIZE_RUN_STATUS, "completed")
        # Force-flush any pending OTel spans so the backend sees the final
        # ``overmind.optimize.final_best_score`` / report tags before the
        # process exits.  ``otlp.py`` projects those onto the Job row.
        force_flush_traces(timeout_millis=5000)

        # ---- Persist cross-run state ----
        if getattr(self.config, "cross_run_persistence", True):
            self._run_state.accumulate_failed(self._session_failed)
            self._run_state.accumulate_successful(self._session_successful)
            iters_done = len(self._session_successful) + len(self._session_failed)
            self._run_state.end_run(
                RunSummary(
                    run_id=self._run_id,
                    started_at=0,
                    finished_at=time.time(),
                    baseline_score=self._baseline_train_score,
                    final_score=self.best_score,
                    iterations_completed=iters_done,
                    accepted_changes=len(self._session_successful),
                    rejected_changes=len(self._session_failed),
                ),
            )
            self._run_state.save()
            n_reg = len(self._run_state.regression_cases)
            n_clusters = len(self._failure_registry.clusters)
            self.console.print(
                f"\n  [dim]Cross-run state saved: {n_clusters} cluster(s), {n_reg} regression case(s)[/dim]"
            )

        # ---- Offer to commit optimized code back to the original agent ----
        try:
            self._prompt_commit_to_original_sources()
        except Exception:
            self._logger.exception("Failed to prompt/commit optimized sources back to originals")

    # ------------------------------------------------------------------
    # Skill-driven phase methods
    #
    # These public methods carve ``Optimizer.run()`` into pieces a host
    # coding agent can drive via ``overmind optimize-step`` (see
    # ``overmind/optimize/steps/`` and ``.cursor/skills/overmind-optimize-agent``).
    # They reuse all of the existing private helpers, so behaviour is
    # identical to running the full ``run()`` end-to-end.
    # ------------------------------------------------------------------

    @observe_safe("optimizer.run_baseline_phase", SpanType.WORKFLOW)
    def run_baseline_phase(self) -> dict:
        """Execute Phase 1 of :meth:`run` (baseline) and return a state dict.

        Mirrors the baseline portion of :meth:`run` (dataset load, split,
        bundle build, env provisioning, baseline eval, failure-cluster
        ingest, working-copy materialisation) but stops before the
        iteration loop. Populates the same ``self.*`` fields the loop
        body expects, so a follow-up call to one of the per-iteration
        methods sees a consistent state.

        Returns a dict with everything the skill needs to persist::

            {
                "baseline_score": float,
                "best_score": float,
                "best_code_path": str,        # working_path
                "best_files_dir": str,        # multi-file root, "" if single
                "best_case_scores": list[float],
                "dataset_size": int,
                "train_size": int,
                "holdout_size": int,
                "output_dir": str,
            }
        """
        self._setup_output_dirs()
        dataset = self._load_dataset()
        holdout_ratio = getattr(self.config, "holdout_ratio", 0.2)
        train_set, holdout_set = Optimizer._split_dataset(dataset, holdout_ratio)

        baseline_code = Path(self.config.agent_path).read_text()
        self._baseline_code = baseline_code
        self._bundle = self._build_bundle()

        self._ensure_runner_env()

        baseline_eval, _, baseline_items = self._run_agent_on_dataset(
            self._instrumented_agent_path, train_set, "baseline"
        )
        self.best_score = baseline_eval["avg_total"]
        self._baseline_train_score = self.best_score
        self.best_code = baseline_code
        self.best_case_scores = [item["score"]["total"] for item in baseline_items]

        if getattr(self.config, "failure_clustering", True):
            baseline_case_results = self._build_case_results(baseline_items, train_set)
            self._failure_registry.ingest_iteration(0, baseline_case_results, self.evaluator.spec)

        if self._bundle:
            self._baseline_files = dict(self._bundle.original_files)
            self._best_files = dict(self._bundle.original_files)

        _ext = Path(self.config.agent_path).suffix or ".py"
        working_path = self.output_dir / f"agent_working{_ext}"
        working_path.write_text(baseline_code)
        working_dir: Path | None = None
        if self._bundle and self._bundle.is_multi_file():
            working_dir = self.output_dir / "agent_working"
            self._write_file_set(working_dir, self._best_files)

        # Persist baseline eval items so later iterations can recover them
        # without re-running the agent.
        items_path = self.output_dir / "_baseline_items.json"
        atomic_write_json(
            items_path,
            {
                "avg_total": baseline_eval["avg_total"],
                "evaluation": baseline_eval,
                "case_results": Optimizer._build_case_results(baseline_items, train_set),
            },
            indent=None,
        )

        # Cache train/holdout to disk so subsequent step CLIs work on the
        # same split without re-shuffling.
        split_path = self.output_dir / "_split.json"
        atomic_write_json(
            split_path,
            {"train": train_set, "holdout": holdout_set},
            indent=None,
        )

        return {
            "baseline_score": float(self.best_score),
            "best_score": float(self.best_score),
            "best_code_path": str(working_path),
            "best_files_dir": str(working_dir) if working_dir else "",
            "best_case_scores": [float(s) for s in self.best_case_scores],
            "dataset_size": len(dataset),
            "train_size": len(train_set),
            "holdout_size": len(holdout_set),
            "output_dir": str(self.output_dir),
            "baseline_items_path": str(items_path),
            "split_path": str(split_path),
        }

    def load_train_holdout(self) -> tuple[list[dict], list[dict]]:
        """Load the cached train/holdout split written by :meth:`run_baseline_phase`."""
        split_path = self.output_dir / "_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Cached split not found at {split_path}. Run `overmind optimize-step baseline` first."
            )
        data = json.loads(split_path.read_text())
        return data["train"], data["holdout"]

    def load_latest_eval(self) -> tuple[dict, list[dict]]:
        """Return ``(evaluation_dict, case_results_list)`` from disk.

        Prefers ``_latest_items.json`` (written after each accepted
        iteration) and falls back to ``_baseline_items.json``.
        """
        for name in ("_latest_items.json", "_baseline_items.json"):
            p = self.output_dir / name
            if p.is_file():
                data = json.loads(p.read_text())
                return data["evaluation"], data["case_results"]
        raise FileNotFoundError(
            f"No baseline or latest items file under {self.output_dir}. Run `overmind optimize-step baseline` first."
        )

    @observe_safe("optimizer.run_diagnose_phase", SpanType.WORKFLOW)
    def run_diagnose_phase(
        self,
        iteration: int,
        current_code: str,
        latest_eval: dict,
        latest_case_results: list[dict],
    ) -> list[dict]:
        """Generate N candidate change plans **without** running codegen.

        Returns a list of plan dicts, one per candidate, each shaped
        like::

            {
                "candidate_id": "c0",
                "method": "<focus area>",
                "diagnosis": {...},
                "suggestions": [...],
                "edit_instructions": "<plain-text prompt for the host coder>",
            }

        The skill uses these to spawn parallel sub-coding-agents.
        """
        n_candidates = getattr(self.config, "candidates_per_iteration", 3)

        # Temperature annealing matches the in-process loop.
        t_start, t_end = 0.8, 0.4
        denom = max(self.config.iterations - 1, 1)
        temperature = t_start - (t_start - t_end) * (iteration - 1) / denom
        if self.stall_count >= 2:
            temperature = min(temperature + 0.2, 1.0)

        cluster_ctx = ""
        component_ctx = ""
        focus_weights: dict[str, float] | None = None

        if getattr(self.config, "failure_clustering", True):
            priority_clusters = self._failure_registry.get_priority_clusters()
            if priority_clusters:
                cluster_ctx = format_clusters_for_diagnosis(priority_clusters)

        if getattr(self.config, "adaptive_focus", True):
            focus_weights = compute_focus_weights(
                latest_case_results,
                latest_eval,
                self.evaluator.spec,
                self._failure_registry,
                self.successful_changes,
                self.failed_attempts,
                is_multi_file=(self._bundle is not None and self._bundle.is_multi_file()),
            )
            component_ctx = format_component_weights(focus_weights)

        agent_files = self._current_agent_files(current_code)

        plans = generate_candidates(
            current_code,
            case_results=latest_case_results,
            evaluation_results=latest_eval,
            model=self.config.analyzer_model,
            eval_spec=self.evaluator.spec,
            failed_attempts=self.failed_attempts,
            successful_changes=self.successful_changes,
            allow_model_change=bool(self.config.model_backtesting and self.config.backtest_models),
            num_candidates=n_candidates,
            temperature=temperature,
            diagnosis_case_fraction=getattr(self.config, "diagnosis_case_fraction", 0.7),
            iteration_seed=iteration * 7919,
            policy_context=self._policy_diagnosis,
            policy_constraints=self._policy_codegen,
            entrypoint_fn=self.config.entrypoint_fn,
            bundle=self._bundle,
            agent_files=agent_files,
            codegen_model=getattr(self.config, "codegen_model", ""),
            codegen_max_steps=getattr(self.config, "codegen_max_steps", 50),
            cluster_context=cluster_ctx,
            component_weights_context=component_ctx,
            focus_weights=focus_weights,
            return_plans_only=True,
        )

        out: list[dict] = []
        for idx, plan in enumerate(plans):
            out.append({
                "candidate_id": f"c{idx}",
                "method": plan.get("method", "unknown"),
                "diagnosis": plan.get("diagnosis", {}),
                "suggestions": plan.get("suggestions", []),
                "edit_instructions": plan.get("edit_instructions", ""),
                "focus_area": plan.get("focus_area", ""),
                "policy_context": self._policy_codegen,
            })
        return out

    @observe_safe("optimizer.evaluate_worktree", SpanType.WORKFLOW)
    def evaluate_worktree(
        self,
        worktree_entry_path: str,
        run_name: str,
        dataset_subset: list[dict] | None = None,
    ) -> dict:
        """Run ``self.config.entrypoint_fn`` from *worktree_entry_path* against the train set.

        Returns a serialisable dict::

            {"avg_total": float, "evaluation": dict, "case_results": list[dict]}
        """
        train, _ = self.load_train_holdout()
        ds = dataset_subset if dataset_subset is not None else train

        set_tag(attrs.OPTIMIZE_WORKTREE_RUN_NAME, run_name)
        set_tag(attrs.OPTIMIZE_WORKTREE_ENTRY_PATH, str(worktree_entry_path))
        set_tag(attrs.OPTIMIZE_WORKTREE_CASES_TOTAL, len(ds))

        started = time.monotonic()
        c_eval, _, c_items = self._run_agent_on_dataset(worktree_entry_path, ds, run_name)
        duration = time.monotonic() - started

        avg_total = float(c_eval.get("avg_total", 0.0))
        dim_scores = {
            key: float(c_eval.get(key, 0.0)) for _, key in self.evaluator.get_dimension_labels() if key in c_eval
        }
        per_case_totals = [float(item["score"].get("total", 0.0)) for item in c_items]
        pass_threshold = float(getattr(self.config, "case_pass_threshold", 70.0))
        pass_rate = (
            sum(1 for s in per_case_totals if s >= pass_threshold) / len(per_case_totals) if per_case_totals else 0.0
        )

        set_tag(attrs.OPTIMIZE_WORKTREE_AVG_SCORE, avg_total)
        set_tag(attrs.OPTIMIZE_WORKTREE_DIMENSION_SCORES, dim_scores)
        set_tag(attrs.OPTIMIZE_WORKTREE_PASS_RATE, float(pass_rate))
        set_tag(attrs.OPTIMIZE_WORKTREE_DURATION_SECONDS, float(duration))

        return {
            "avg_total": avg_total,
            "evaluation": c_eval,
            "case_results": Optimizer._build_case_results(c_items, ds),
        }

    @observe_safe("optimizer.commit_winner", SpanType.WORKFLOW)
    def commit_winner(
        self,
        winner_entry_path: str,
        winner_eval: dict,
        winner_case_results: list[dict],
    ) -> None:
        """Promote *winner_entry_path* as the new best agent.

        Updates ``self.best_*`` and writes the iteration's
        ``_latest_items.json`` so the next ``diagnose`` call sees fresh
        case results. Does **not** modify the user's source tree.
        """
        new_code = Path(winner_entry_path).read_text()
        self.best_code = new_code
        self.best_score = float(winner_eval["avg_total"])
        self.best_case_scores = [float(c["score"]["total"]) for c in winner_case_results]

        _ext = Path(self.config.agent_path).suffix or ".py"
        working_path = self.output_dir / f"agent_working{_ext}"
        working_path.write_text(new_code)

        atomic_write_json(
            self.output_dir / "_latest_items.json",
            {
                "avg_total": winner_eval["avg_total"],
                "evaluation": winner_eval,
                "case_results": winner_case_results,
            },
            indent=None,
        )

    @observe_safe("optimizer.render_report_only", SpanType.WORKFLOW)
    def render_report_only(self) -> str:
        """Render ``report.md`` from ``self.results``. Returns the report path."""
        self._setup_output_dirs()
        baseline_score = self._baseline_train_score or self.best_score
        self._write_report_md(baseline_score)
        return str(self.output_dir / "report.md")

    # ------------------------------------------------------------------
    # Commit optimized sources back to original agent files
    # ------------------------------------------------------------------

    @observe_safe("optimizer.collect_commit_targets")
    def _collect_commit_targets(self) -> list[tuple[Path, str, str]]:
        """Return ``[(abs_path, original, optimized), ...]`` for files to commit.

        Each ``optimized`` string has all overmind-sdk instrumentation
        (imports, ``overmind_init()``, ``@observe()`` decorators) stripped
        so what the user sees and commits matches the style of their
        original source.  Files whose cleaned optimized content matches
        the on-disk original are included with identical strings so the
        caller can decide whether to show them or not.
        """
        targets: list[tuple[Path, str, str]] = []

        use_bundle = self._bundle is not None and self._best_files and self._bundle.is_multi_file()

        if use_bundle:
            root = Path(self._bundle.project_root).resolve()
            for rel_path, new_src in self._best_files.items():
                abs_path = (root / rel_path).resolve()
                try:
                    original = abs_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                cleaned = deinstrument_source(new_src) if rel_path.endswith(".py") else new_src
                targets.append((abs_path, original, cleaned))
        else:
            abs_path = Path(self.config.agent_path).resolve()
            try:
                original = abs_path.read_text(encoding="utf-8")
            except Exception:
                return targets
            new_src = self.best_code or original
            cleaned = deinstrument_source(new_src) if abs_path.suffix == ".py" else new_src
            targets.append((abs_path, original, cleaned))

        return targets

    @observe_safe("optimizer.prompt_commit")
    def _prompt_commit_to_original_sources(self) -> None:
        """After optimization, show diffs and offer to write them to originals.

        Skips silently when the cleaned optimized source is identical to
        what's already on disk for every file, or when there was no
        meaningful improvement over the baseline.
        """
        if self.best_score <= self._baseline_train_score:
            return

        targets = self._collect_commit_targets()
        changed: list[tuple[Path, str, str]] = [t for t in targets if t[1] != t[2]]
        if not changed:
            return

        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()
        self.console.print(
            Panel(
                "[bold]Commit Optimized Code to Original Agent[/bold]\n"
                "[dim]Review the diff for each file. If you accept, the "
                "cleaned-up optimized source will replace your original "
                "agent files on disk (overmind-sdk instrumentation is "
                "stripped first).[/dim]",
                border_style=BRAND,
            )
        )

        cwd = Path.cwd()
        for abs_path, original, cleaned in changed:
            try:
                shown_path = abs_path.relative_to(cwd)
            except ValueError:
                shown_path = abs_path
            diff_text = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    cleaned.splitlines(keepends=True),
                    fromfile=f"a/{shown_path}",
                    tofile=f"b/{shown_path}",
                    n=3,
                )
            )
            if not diff_text:
                continue
            self.console.print()
            self.console.print(f"  [bold]{shown_path}[/bold]")
            self.console.print(
                Syntax(
                    diff_text,
                    "diff",
                    theme="ansi_dark",
                    background_color="default",
                    word_wrap=False,
                )
            )

        self.console.print()
        accepted = confirm_option(
            "Do you want to commit these changes back to the original agent?",
            default=True,
            console=self.console,
        )
        if not accepted:
            self.console.print(
                "  [dim]Skipped. Your original agent files are unchanged.[/dim]\n"
                f"  [dim]Optimized sources remain available under "
                f"[bold]{self.output_dir}[/bold].[/dim]\n"
            )
            return

        written = 0
        for abs_path, _, cleaned in changed:
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(cleaned, encoding="utf-8")
                written += 1
            except Exception as exc:
                self.console.print(f"  [red]Failed to write {abs_path}: {exc}[/red]")

        self.console.print(
            f"\n  [bold green]\u2713[/bold green] "
            f"Committed {written} file(s) to the original agent.\n"
            f"  [dim]Re-register or re-instrument to refresh the "
            f"[bold]{agent_instrumented_dir(self.config.agent_name)}[/bold] "
            f"copy before the next optimize run.[/dim]\n"
        )

    # ------------------------------------------------------------------
    # Complexity penalty (prompt bloat + code growth + override detection)
    # ------------------------------------------------------------------

    @observe_safe("optimizer.compute_complexity_penalty")
    def _compute_complexity_penalty(
        self,
        candidate_code: str,
        train_set: list[dict] | None = None,
        raw_score: float | None = None,
    ) -> float:
        """Penalize candidates with excessive prompt, code, or logic growth.

        Thin wrapper around :func:`overmind.optimize.pipeline.scoring.compute_complexity_penalty`
        that supplies state from ``self`` (baseline code, best code/score,
        config thresholds, eval-spec vocabulary).
        """
        return _scoring_compute_complexity_penalty(
            candidate_code,
            baseline_code=self._baseline_code,
            best_code=self.best_code,
            best_score=self.best_score,
            train_set=train_set,
            raw_score=raw_score,
            max_code_growth_ratio=getattr(self.config, "max_code_growth_ratio", 2.5),
            known_domain_values=self._domain_vocabulary(),
        )

    def _domain_vocabulary(self) -> set[str]:
        """Collect enum values from the eval spec used for leakage exclusions."""
        vocab: set[str] = set()
        for field_cfg in self.evaluator.spec.get("output_fields", {}).values():
            for v in field_cfg.get("values", []):
                vocab.add(str(v).strip().lower())
        return vocab

    @observe_safe("optimizer.detect_data_leakage")
    def _detect_data_leakage(self, candidate_code: str, train_set: list[dict]) -> int:
        """Count expected-output literals leaked from training data into the candidate.

        Thin wrapper around :func:`overmind.optimize.pipeline.scoring.detect_data_leakage`.
        """
        return _scoring_detect_data_leakage(
            candidate_code,
            self._baseline_code or "",
            train_set,
            known_domain_values=self._domain_vocabulary(),
        )

    # Legacy attribute aliases so any in-tree callers (and tests) still resolve.
    _get_prompt_size = staticmethod(_scoring_prompt_size)
    _count_conditional_branches = staticmethod(_scoring_count_conditional_branches)
    _count_function_defs = staticmethod(_scoring_count_function_defs)

    # ------------------------------------------------------------------
    # Dataset splitting
    # ------------------------------------------------------------------

    @observe_safe("optimizer.split_dataset")
    @staticmethod
    def _split_dataset(dataset: list[dict], holdout_ratio: float) -> tuple[list[dict], list[dict]]:
        """Split dataset into train and holdout sets.

        Shuffles with a fixed seed for reproducibility.
        """
        if holdout_ratio <= 0 or len(dataset) < 5:
            return dataset, []

        n_holdout = max(1, int(len(dataset) * holdout_ratio))
        indices = list(range(len(dataset)))
        random.Random(42).shuffle(indices)
        holdout_idx = set(indices[:n_holdout])

        train = [d for i, d in enumerate(dataset) if i not in holdout_idx]
        holdout = [d for i, d in enumerate(dataset) if i in holdout_idx]
        return train, holdout

    # ------------------------------------------------------------------
    # Holdout snapshot rollback
    # ------------------------------------------------------------------

    @observe_safe("optimizer.rollback_to_best_snapshot")
    def _rollback_to_best_snapshot(
        self,
        best_path: Path,
        holdout_set: list[dict],
        baseline_holdout_score: float,
        holdout_weight: float,
        catastrophic_threshold: float,
    ) -> dict | None:
        """Find the snapshot that maximizes blended (train + holdout) improvement.

        Evaluates up to ``MAX_SNAPSHOTS_TO_TEST`` most-recent accepted snapshots
        (excluding the last one, which already triggered rollback) on the holdout
        set.  Each snapshot is scored with:

            blended = (1 - holdout_weight) * train_imp + holdout_weight * holdout_imp

        The snapshot with the highest *positive* blended improvement is returned.
        Snapshots with catastrophic holdout degradation (below
        ``catastrophic_threshold`` of baseline) are excluded regardless of
        blended score.

        Returns the best snapshot info dict, or ``None`` if no snapshot achieves
        a positive blended improvement (caller should revert to baseline).
        """
        MAX_SNAPSHOTS_TO_TEST = 4

        candidates = self.accepted_snapshots[:-1] if self.accepted_snapshots else []
        candidates = list(reversed(candidates[-MAX_SNAPSHOTS_TO_TEST:]))

        if not candidates:
            return None

        self.console.print(
            f"  [dim]Evaluating {len(candidates)} earlier snapshot(s) "
            f"against holdout (picking best blended score)\u2026[/dim]"
        )

        scored: list[tuple[float, dict]] = []

        for snap in candidates:
            self.console.print(f"    [dim]Iteration {snap['iteration']} (train: {snap['train_score']:.1f})\u2026[/dim]")
            best_path.write_text(snap["code"])
            try:
                snap_eval, _, _ = self._run_agent_on_dataset(
                    str(best_path),
                    holdout_set,
                    f"holdout_snap_{snap['iteration']}",
                )
            except Exception:
                continue

            snap_holdout_score = snap_eval["avg_total"]
            snap_holdout_imp = snap_holdout_score - baseline_holdout_score
            snap_train_imp = snap["train_score"] - self._baseline_train_score

            snap_is_catastrophic = (
                baseline_holdout_score > 0 and snap_holdout_score < baseline_holdout_score * catastrophic_threshold
            )
            if snap_is_catastrophic:
                self.console.print(
                    f"      [red]\u2717 Catastrophic holdout drop "
                    f"({snap_holdout_score:.1f} < "
                    f"{baseline_holdout_score * catastrophic_threshold:.1f})"
                    f"[/red]"
                )
                continue

            blended = (1 - holdout_weight) * snap_train_imp + holdout_weight * snap_holdout_imp
            self.console.print(
                f"      holdout: {snap_holdout_score:.1f} ({snap_holdout_imp:+.1f}), blended: {blended:+.1f}"
            )

            if blended > 0:
                scored.append((
                    blended,
                    {
                        "code": snap["code"],
                        "files": snap.get("files"),
                        "train_score": snap["train_score"],
                        "holdout_score": snap_holdout_score,
                        "holdout_eval": snap_eval,
                        "iteration": snap["iteration"],
                        "blended_improvement": blended,
                    },
                ))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Dimension delta computation
    # ------------------------------------------------------------------

    @observe_safe("optimizer.compute_dimension_deltas")
    def _compute_dimension_deltas(self, old_eval: dict, new_eval: dict) -> dict[str, float]:
        """Per-dimension score deltas (only includes changes > 0.5)."""
        deltas: dict[str, float] = {}
        for _, key in self.evaluator.get_dimension_labels():
            old_val = old_eval.get(key, 0)
            new_val = new_eval.get(key, 0)
            delta = round(new_val - old_val, 1)
            if abs(delta) > 0.5:
                deltas[key] = delta
        return deltas

    # ------------------------------------------------------------------
    # Regression-aware acceptance
    # ------------------------------------------------------------------

    @observe_safe("optimizer.check_acceptance")
    def _check_acceptance(
        self,
        candidate_score: float,
        candidate_case_scores: list[float],
        candidate_items: list[dict],
        dataset: list[dict],
        *,
        candidate_eval: dict | None = None,
    ) -> tuple[bool, str]:
        """Check if a candidate should be accepted.

        Uses a four-tier acceptance strategy:
        0. Noise-floor gate — when multi-run stdev data is available, reject
           improvements smaller than ``noise_factor * stdev``.
        1. Net-positive override — if the average score improved meaningfully
           and fewer than half the cases had major regressions, accept.
        2. Magnitude override — if improvements outweigh regressions by 1.2x,
           accept even if the regression ratio exceeds the threshold.
        3. Standard threshold — accept if the fraction of regressed cases is
           within the configured limit.

        Per-case regression sensitivity is set to 3.0 points (on a 100-point
        scale) so that small LLM-variance fluctuations are not counted as
        regressions.

        Returns (accept, reason).
        """
        threshold = getattr(self.config, "regression_threshold", 0.35)

        if candidate_score <= self.best_score:
            return False, (f"No improvement ({candidate_score:.1f} vs best {self.best_score:.1f})")

        net_improvement = candidate_score - self.best_score

        # Tier 0: Noise-floor gate — require improvement to exceed the
        # observed run-to-run variance when multi-run eval is used.
        if candidate_eval and "_stdev" in candidate_eval:
            stdev = candidate_eval["_stdev"]
            noise_factor = 1.0
            noise_floor = noise_factor * stdev
            if noise_floor > 0 and net_improvement < noise_floor:
                return False, (
                    f"Improvement {net_improvement:.1f} is within noise floor "
                    f"(stdev={stdev:.1f}, required ≥ {noise_floor:.1f})"
                )

        if not self.best_case_scores or not candidate_case_scores:
            return True, ""

        n = min(len(self.best_case_scores), len(candidate_case_scores))
        regressions = 0
        regression_magnitude = 0.0
        improvements = 0
        improvement_magnitude = 0.0

        for j in range(n):
            delta = candidate_case_scores[j] - self.best_case_scores[j]
            if delta < -3.0:
                regressions += 1
                regression_magnitude += abs(delta)
            elif delta > 3.0:
                improvements += 1
                improvement_magnitude += delta

        regression_ratio = regressions / max(n, 1)

        # Tier 1: Net-positive override — average improved meaningfully and
        # fewer than half the cases had major (>3pt) regressions.
        if net_improvement >= 0.5 and regression_ratio <= 0.5:
            return True, (
                f"Net positive ({net_improvement:+.1f} avg, "
                f"{improvements} improved, {regressions} regressed out of {n})"
            )

        # Tier 2 & 3: standard threshold with magnitude override
        if regression_ratio > threshold:
            if improvement_magnitude > regression_magnitude * 1.2:
                return True, (
                    f"Accepted despite {regressions}/{n} regressions "
                    f"(improvement magnitude {improvement_magnitude:.1f} "
                    f"outweighs regression {regression_magnitude:.1f})"
                )
            return False, (f"Too many regressions: {regressions}/{n} cases regressed (threshold: {threshold:.0%})")

        return True, (f"{improvements} improved, {regressions} regressed out of {n} cases")

    # ------------------------------------------------------------------
    # Cross-run regression gate
    # ------------------------------------------------------------------

    @observe_safe("optimizer.check_regression_suite")
    def _check_regression_suite(
        self,
        candidate: dict,
        train_set: list[dict],
    ) -> int:
        """Evaluate a candidate against the cross-run regression suite.

        Returns the number of regression cases that fail (score below
        their stored min_score).  The caller decides whether this exceeds
        the configured threshold.
        """
        if not self._run_state.regression_cases:
            return 0

        tmp_path = self._write_candidate_to_disk(candidate)
        try:
            runner = self._build_runner(str(tmp_path), self.config.entrypoint_fn)
            runner.ensure_environment()
        except Exception:
            self._cleanup_candidate(tmp_path, candidate)
            return len(self._run_state.regression_cases)

        # Traces from the agent subprocess flow through OTel to the remote
        # backend (the SDK reads ``TRACEPARENT`` injected by ``runner.run``
        # and reports its own spans there).  We no longer collect a local
        # trace file here, so the regression-gate evaluator scores without
        # tool-trace context.
        outputs: list[dict | None] = []
        failed_indices: set[int] = set()
        for rc_idx, rc in enumerate(self._run_state.regression_cases):
            run_output = runner.run(rc.case_input)
            if run_output.success:
                outputs.append(run_output.data)
            else:
                outputs.append(None)
                failed_indices.add(rc_idx)

        failures = 0
        for rc_idx, rc in enumerate(self._run_state.regression_cases):
            if rc_idx in failed_indices:
                failures += 1
                continue

            skip_judge = not getattr(self.config, "judge_in_regression", False)
            score = self.evaluator.evaluate_output(
                outputs[rc_idx],
                rc.expected_output,
                input_data=rc.case_input,
                tool_trace=[],
                _skip_judge=skip_judge,
            )
            if score["total"] < rc.min_score:
                failures += 1

        self._cleanup_candidate(tmp_path, candidate)
        runner.cleanup()
        return failures

    @observe_safe("optimizer.promote_resolved_to_regression")
    def _promote_resolved_to_regression(
        self,
        resolved_clusters: list,
        case_results: list[dict],
        train_set: list[dict],
        iteration: int,
    ) -> None:
        """Promote resolved failure cluster exemplars to the regression suite."""
        for cluster in resolved_clusters:
            for case_idx in cluster.exemplar_case_indices:
                if case_idx >= len(case_results) or case_idx >= len(train_set):
                    continue
                case_data = train_set[case_idx]
                case_result = case_results[case_idx]
                score = case_result.get("score", {}).get("total", 60.0)
                self._run_state.add_regression_case(
                    case_input=case_data.get("input", {}),
                    expected_output=case_data.get("expected_output", case_data.get("expected", {})),
                    min_score=max(score * 0.8, 50.0),
                    run_id=self._run_id,
                    iteration=iteration,
                    cluster_id=cluster.cluster_id,
                )

    # ------------------------------------------------------------------
    # Multi-run evaluation
    # ------------------------------------------------------------------

    @observe_safe("optimizer.run_multi_eval")
    def _run_multi_eval(
        self,
        agent_path: str,
        dataset: list[dict],
        run_name: str,
        num_runs: int,
    ) -> tuple[dict, list[dict]]:
        """Run the agent multiple times and return the median run for stability.

        Uses the median-scoring run's eval and items so that per-case scores
        and aggregate score are consistent with each other.
        """
        runs: list[tuple[float, dict, list[dict]]] = []

        for r in range(num_runs):
            r_eval, _, r_items = self._run_agent_on_dataset(agent_path, dataset, f"{run_name}_r{r}")
            runs.append((r_eval["avg_total"], r_eval, r_items))

        all_scores = [t for t, _, _ in runs]

        if num_runs > 1:
            mean = statistics.mean(all_scores)
            stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            self.console.print(f"      [dim]Multi-run: mean={mean:.1f}, stdev={stdev:.1f}, runs={all_scores}[/dim]")

        runs.sort(key=lambda x: x[0])
        median_idx = len(runs) // 2
        _, median_eval, median_items = runs[median_idx]

        if num_runs > 1:
            median_eval["_stdev"] = stdev
            median_eval["_all_runs"] = all_scores

        return median_eval, median_items

    # ------------------------------------------------------------------
    # Per-case result builder (with full tool traces)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_case_results(eval_items: list[dict], dataset: list[dict]) -> list[dict]:
        results: list[dict] = []
        for item, case in zip(eval_items, dataset):
            results.append({
                "input": case.get("input", {}),
                "expected": item["expected"],
                "output": item["output"],
                "score": item["score"],
                "tool_calls": item.get("tool_calls", []),
                "tool_trace": item.get("tool_trace", []),
            })
        return results

    # ------------------------------------------------------------------
    # Baseline diagnostics
    # ------------------------------------------------------------------

    @observe_safe("optimizer.print_baseline_diagnostics")
    def _print_baseline_diagnostics(self, evaluation: dict, items: list[dict]):
        """Print smart diagnostics about the baseline run."""
        self.console.print()
        max_scores = self.evaluator.get_max_scores()

        # Find saturated dimensions
        saturated = []
        weak = []
        for display, key in self.evaluator.get_dimension_labels():
            val = evaluation.get(key, 0)
            mx = max_scores.get(key, 0)
            if mx > 0:
                pct = val / mx
                if pct >= 0.95:
                    saturated.append(display)
                elif pct < 0.5:
                    weak.append((display, val, mx))

        if saturated:
            self.console.print(f"  [dim]Saturated dimensions (already near-perfect): {', '.join(saturated)}[/dim]")
        if weak:
            self.console.print("  [yellow]Weak dimensions (biggest improvement room):[/yellow]")
            for name, val, mx in weak:
                self.console.print(f"    {name}: {val:.1f}/{mx:.0f} ({val / mx * 100:.0f}%)")

        # Tool usage summary
        all_tools_used: dict[str, int] = {}
        cases_with_no_tools = 0
        for item in items:
            trace = item.get("tool_trace", [])
            if not trace:
                cases_with_no_tools += 1
            for tc in trace:
                name = tc.get("name", "")
                all_tools_used[name] = all_tools_used.get(name, 0) + 1

        if all_tools_used:
            self.console.print("  [dim]Tool usage across baseline:[/dim]")
            for name, count in sorted(all_tools_used.items(), key=lambda x: -x[1]):
                self.console.print(f"    {name}: {count}/{len(items)} cases")
            if cases_with_no_tools:
                self.console.print(f"    [yellow]{cases_with_no_tools} cases used no tools[/yellow]")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    @observe_safe("optimizer.load_dataset")
    def _load_dataset(self) -> list[dict]:
        """Load the dataset from disk.

        Data is prepared during ``overmind setup`` (generated synthetically
        or analyzed/augmented from seed data).  The optimizer only loads it.
        """
        from overmind.optimize.data import normalize_data_fields

        self.console.print(f"  [dim]Loading data from {self.config.data_path}…[/dim]")
        cases = load_data(self.config.data_path)
        cases = normalize_data_fields(
            cases,
            self.console,
            require_output=True,
            agent_name=getattr(self.config, "agent_name", None) or None,
        )
        return cases

    # ------------------------------------------------------------------
    # Multi-file bundle helpers
    # ------------------------------------------------------------------

    def _current_agent_files(self, current_code: str) -> dict[str, str]:
        """Return the current file set for the coding agent.

        For multi-file agents, uses ``_best_files``.  For single-file,
        derives a ``{relative_path: source}`` dict from the entry file.
        """
        if self._best_files:
            # Ensure the entry file has the latest code read from the
            # working path (which is the source of truth each iteration).
            result = dict(self._best_files)
            if self._bundle:
                result[self._bundle.entry_file] = current_code
            return result

        # Single-file fallback: derive relative path from the agent path
        from overmind.core.registry import project_root_from_agent_file

        pr = project_root_from_agent_file(self.config.agent_path)
        if pr:
            try:
                rel = str(Path(self.config.agent_path).resolve().relative_to(pr))
            except ValueError:
                rel = Path(self.config.agent_path).name
        else:
            rel = Path(self.config.agent_path).name
        return {rel: current_code}

    @observe_safe("optimizer.build_bundle")
    def _build_bundle(self) -> AgentBundle | None:
        """Build an ``AgentBundle`` from the current config."""
        from overmind.optimize.bundle_factory import build_agent_bundle

        return build_agent_bundle(self.config)

    @observe_safe("optimizer.rebuild_bundle")
    def _rebuild_bundle(self) -> None:
        """Rebuild the bundle from current ``_best_files`` state.

        Called after accepting a multi-file candidate so subsequent
        iterations see the updated file contents and pieces.
        """
        if not self._bundle or not self._best_files:
            return

        from overmind.utils.code import extract_pieces

        self._bundle.original_files = dict(self._best_files)
        new_pieces = []
        opt_files = set(self._bundle.optimizable_files)

        ordered_paths = [self._bundle.entry_file] + [p for p in self._best_files if p != self._bundle.entry_file]

        for rel_path in ordered_paths:
            if rel_path not in self._best_files:
                continue
            source = self._best_files[rel_path]
            is_opt = rel_path in opt_files
            pieces = extract_pieces(rel_path, source, optimizable=is_opt)
            new_pieces.extend(pieces)

        self._bundle.pieces = new_pieces
        self._bundle._assign_ids()

    @observe_safe("optimizer.resolve_bundle_candidate")
    def _resolve_bundle_candidate(self, bundle_updates: dict) -> dict | None:
        """Resolve bundle updates into modified files.

        Only the whole-file ``file_updates`` format is supported — the
        legacy piece-ID splice path was deleted in the client cleanup.

        Returns ``{"entry_code": str, "files": {rel_path: source}}`` or
        ``None`` if validation fails or the candidate did not produce
        any file updates.
        """
        if not self._bundle:
            return None

        file_updates = bundle_updates.get("file_updates")
        if not file_updates:
            return None

        modified = self._bundle.apply_file_updates(file_updates)
        if modified is None:
            return None

        full_files = dict(self._best_files) if self._best_files else {}
        full_files.update(modified)

        entry_code = full_files.get(self._bundle.entry_file)
        if not entry_code:
            return None

        if not self._runner.validate_entrypoint(entry_code):
            return None

        return {"entry_code": entry_code, "files": modified}

    @observe_safe("optimizer.write_candidate_to_disk")
    def _write_candidate_to_disk(self, cand: dict) -> Path:
        """Write a candidate to disk for evaluation, handling both modes.

        For multi-file candidates with ``_resolved_files``, creates a
        temporary directory tree.  For single-file, creates a temp ``.py``.
        Each Python file is auto-instrumented with ``@observe()`` so
        overmind-sdk traces are captured during evaluation.
        Returns the path to the entry file.
        """
        from overmind.utils.instrument import instrument_source

        resolved = cand.get("_resolved_files")
        if resolved and self._bundle:
            tmp_dir = Path(tempfile.mkdtemp(prefix="overmind_", dir=str(self.output_dir)))
            all_files = self._bundle.get_full_file_set(resolved)
            for rel_path, source in all_files.items():
                dest = tmp_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if rel_path.endswith(".py"):
                    source = instrument_source(source)
                dest.write_text(source)
            return tmp_dir / self._bundle.entry_file

        ext = Path(self.config.agent_path).suffix or ".py"
        fd, tmp_str = tempfile.mkstemp(suffix=ext, dir=str(self.output_dir))
        os.close(fd)
        tmp = Path(tmp_str)
        code = cand["updated_code"]
        if ext == ".py":
            code = instrument_source(code)
        tmp.write_text(code)
        return tmp

    @observe_safe("optimizer.cleanup_candidate")
    def _cleanup_candidate(self, tmp_path: Path, cand: dict) -> None:
        """Clean up temporary files/dirs created by ``_write_candidate_to_disk``."""
        resolved = cand.get("_resolved_files")
        if resolved and self._bundle:
            # tmp_path is entry_file inside a temp dir — remove the dir
            tmp_dir = tmp_path
            for _ in range(10):
                if tmp_dir.parent == self.output_dir or tmp_dir == tmp_dir.parent:
                    break
                tmp_dir = tmp_dir.parent
            if tmp_dir != self.output_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _write_file_set(directory: Path, files: dict[str, str]) -> None:
        """Write a set of files to *directory*, preserving relative paths."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for rel_path, source in files.items():
            dest = directory / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source)

    # ------------------------------------------------------------------
    # Agent loading & execution
    # ------------------------------------------------------------------

    @observe_safe("optimizer.build_runner")
    def _build_runner(
        self,
        agent_path: str,
        entrypoint_fn: str,
        extra_env: dict[str, str] | None = None,
    ) -> AgentRunner:
        """Create an AgentRunner for the given agent file.

        ``env_dir`` always points to the *original* agent project root
        so that dependency manifests, ``.venv``, and ``.env`` files are
        found even when the code being evaluated lives in the
        instrumented or experiments folder.

        ``agent_dir`` is resolved to the project root so that local
        cross-package imports work correctly.  For paths inside the
        instrumented tree, the instrumented directory is used as
        ``agent_dir`` (it mirrors the original project layout).

        For paths inside an experiments worktree
        (``.../experiments/<worktree>/...``), the worktree itself is
        used as ``agent_dir``.  Each candidate / baseline worktree is a
        self-contained snapshot of the agent project, so using the
        worktree as the import root ensures ``from agent import …``-
        style sibling imports inside a harness resolve to the
        worktree's local files instead of silently shadowing them with
        the project-root baseline.
        """
        p = Path(agent_path).resolve()

        inst_dir = agent_instrumented_dir(self.config.agent_name).resolve()
        exp_dir = agent_experiments_dir(self.config.agent_name).resolve()
        if _is_subpath(p, inst_dir):
            agent_dir = inst_dir
            entry_file = str(p.relative_to(inst_dir))
        elif _is_subpath(p, exp_dir):
            rel_parts = p.relative_to(exp_dir).parts
            if len(rel_parts) >= 2:
                worktree = exp_dir / rel_parts[0]
                agent_dir = worktree
                entry_file = str(p.relative_to(worktree))
            else:
                agent_dir = p.parent
                entry_file = p.name
        else:
            pr = project_root_from_agent_file(agent_path)
            if pr is not None:
                agent_dir = pr
                entry_file = str(p.relative_to(pr))
            else:
                agent_dir = p.parent
                entry_file = p.name

        original_pr = project_root_from_agent_file(self.config.agent_path)
        original_agent_dir = original_pr if original_pr is not None else Path(self.config.agent_path).resolve().parent
        cfg = RunnerConfig(extra_env=extra_env or {})
        return AgentRunner(
            agent_dir=agent_dir,
            entry_file=entry_file,
            entrypoint_fn=entrypoint_fn,
            config=cfg,
            env_dir=original_agent_dir,
        )

    @observe_safe("optimizer.ensure_runner_env")
    def _ensure_runner_env(self) -> None:
        """Provision the runner's environment (deps install). Idempotent."""
        from overmind.optimize.runner import MissingDependenciesError

        try:
            self._runner.ensure_environment()
        except MissingDependenciesError as exc:
            self.console.print(
                f"\n  [bold red]Missing dependency file[/bold red]\n\n"
                f"  {exc}\n\n"
                f"  Run [bold]overmind setup {self.config.agent_name}[/bold] to configure\n"
                f"  dependencies interactively, or create a dependency file manually.\n"
            )
            raise SystemExit(1)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            self.console.print(f"  [bold red]Failed to provision agent environment:[/bold red]\n  [dim]{stderr}[/dim]")
            raise

    @observe_safe("optimizer.run_agent_on_dataset")
    def _run_agent_on_dataset(
        self,
        agent_path: str,
        dataset: list[dict],
        run_name: str,
    ) -> tuple[dict, list[ParsedTrace], list[dict]]:
        runner = self._build_runner(agent_path, self.config.entrypoint_fn)
        runner.ensure_environment()

        # Per-run trace directory. Each case's subprocess writes its OTel
        # spans to ``trace_dir / case_<idx>.jsonl`` via the local
        # ``JsonlFileSpanExporter`` installed by the wrapper bootstrap when
        # ``OVERMIND_TRACE_FILE`` is set. The parent reads those files back
        # in :meth:`_build_eval_results` so tool / LLM traces are available
        # to :class:`SpecEvaluator` (Tool Usage, llm token counts, etc.)
        # without any backend roundtrip. Set to ``None`` to disable local
        # capture (shadow/cassette modes do this themselves).
        trace_dir: Path | None = self.traces_dir / run_name
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir  # legacy alias used by some logging paths

        cassette_path = self._cassette_path_for(run_name)
        shadow_prov_dir = self._shadow_prov_dir_for(run_name)
        plan = build_default_plan(
            runner=runner,
            cassette_path=cassette_path,
            provenance_dir=shadow_prov_dir,
            enable_shadow_fallback=self._shadow_fallback_enabled(),
        )

        set_tag(attrs.RUN_AGENT_RUN_NAME, run_name)
        set_tag(attrs.RUN_AGENT_AGENT_PATH, str(agent_path))
        set_tag(attrs.RUN_AGENT_CASES_TOTAL, len(dataset))
        set_tag(attrs.RUN_AGENT_PARALLEL, bool(self.config.parallel))
        set_tag(attrs.RUN_AGENT_MAX_WORKERS, int(self.config.max_workers or 0))
        set_tag(attrs.RUN_AGENT_BACKENDS, [b.name for b in plan])

        started = time.monotonic()
        if self.config.parallel:
            self._logger.debug(
                f"Running agent in parallel: run={run_name} cases={len(dataset)} "
                f"workers={self.config.max_workers} trace={trace_path} backends={len(plan)}"
            )
            batch_eval, traces, eval_items = self._run_parallel_subprocess(runner, dataset, run_name, trace_path, plan)
        else:
            self._logger.debug(
                f"Running agent sequentially: run={run_name} cases={len(dataset)} "
                f"trace={trace_path} backends={len(plan)}"
            )
            batch_eval, traces, eval_items = self._run_sequential_subprocess(
                runner, dataset, run_name, trace_path, plan
            )
        duration = time.monotonic() - started

        # Aggregated batch eval — gives the OTLP ingest and the UI a
        # single span where the run's headline numbers live.
        avg_total = float(batch_eval.get("avg_total", 0.0))
        dim_scores = {
            key: float(batch_eval.get(key, 0.0))
            for _, key in self.evaluator.get_dimension_labels()
            if key in batch_eval
        }
        per_case_totals = [float(item["score"].get("total", 0.0)) for item in eval_items]
        pass_threshold = float(getattr(self.config, "case_pass_threshold", 70.0))
        pass_rate = (
            sum(1 for s in per_case_totals if s >= pass_threshold) / len(per_case_totals) if per_case_totals else 0.0
        )

        set_tag(attrs.RUN_AGENT_AVG_SCORE, avg_total)
        if dim_scores:
            set_tag(attrs.RUN_AGENT_DIMENSION_SCORES, dim_scores)
        set_tag(attrs.RUN_AGENT_PASS_RATE, float(pass_rate))
        set_tag(attrs.RUN_AGENT_DURATION_SECONDS, float(duration))
        if per_case_totals:
            # Cap the inline per-case list to keep the span attribute
            # small — full per-case results live on the individual
            # ``optimizer.run_case_with_plan`` spans.
            set_tag(attrs.RUN_AGENT_PER_CASE_SCORES, per_case_totals[:200])

        return batch_eval, traces, eval_items

    def _cassette_path_for(self, run_name: str) -> Path:
        """Cassette file used to record/replay external calls for *run_name*."""
        base = self.traces_dir if hasattr(self, "traces_dir") else Path(".")
        cass_dir = Path(base).parent / "cassettes"
        cass_dir.mkdir(parents=True, exist_ok=True)
        # One cassette per agent (not per run_name) so the cassette
        # accumulates knowledge across iterations.
        return cass_dir / f"{self.config.agent_name or 'agent'}.jsonl"

    def _shadow_prov_dir_for(self, run_name: str) -> Path:
        """Per-run directory holding shadow-execution sidecar JSONLs."""
        base = self.traces_dir if hasattr(self, "traces_dir") else Path(".")
        d = Path(base).parent / "shadow" / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _shadow_fallback_enabled(self) -> bool:
        """Global switch for the subprocess → shadow fallback.

        Defaults to ``True`` so the optimizer transparently recovers from
        runtime failures.  Users can disable via ``config.enable_shadow``
        or the ``OVERMIND_DISABLE_SHADOW`` environment variable.
        """
        if os.environ.get("OVERMIND_DISABLE_SHADOW") == "1":
            return False
        return bool(getattr(self.config, "enable_shadow", True))

    @observe_safe("optimizer.run_sequential_subprocess")
    def _run_sequential_subprocess(
        self,
        runner: AgentRunner,
        dataset: list[dict],
        run_name: str,
        trace_path: Path | None = None,
        plan: BackendPlan | None = None,
    ) -> tuple[dict, list[ParsedTrace], list[dict]]:
        outputs: list[dict | None] = []
        cases_data: list[dict] = []
        provenance_by_idx: dict[int, list[dict]] = {}
        success_count = 0
        fail_count = 0
        backends_used: dict[str, int] = {}

        with Progress(
            SpinnerColumn(style=BRAND),
            TextColumn(f"[bold {BRAND}]{{task.description}}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("  Running agent…", total=len(dataset))

            for idx, case in enumerate(dataset):
                self._logger.debug(f"[{run_name}] Sequential case {idx + 1}/{len(dataset)}")
                backend_output = self._run_case_with_plan(plan, case["input"], trace_path, idx, run_name)
                run_output = backend_output.run_output
                backends_used[backend_output.backend] = backends_used.get(backend_output.backend, 0) + 1
                if run_output.success:
                    outputs.append(run_output.data)
                    success_count += 1
                else:
                    self._logger.warning(
                        f"[{run_name}] Sequential case {idx} failed backend={backend_output.backend} "
                        f"rc={run_output.returncode} err={(run_output.error or '')[:300]}"
                    )
                    outputs.append({"error": run_output.error})
                    fail_count += 1
                if backend_output.provenance:
                    provenance_by_idx[idx] = [t.to_dict() for t in backend_output.provenance]
                cases_data.append(case)
                progress.advance(task)

        set_tag(attrs.RUN_AGENT_RUN_NAME, run_name)
        set_tag(attrs.RUN_AGENT_CASES_TOTAL, len(dataset))
        set_tag(attrs.RUN_AGENT_CASES_SUCCEEDED, success_count)
        set_tag(attrs.RUN_AGENT_CASES_FAILED, fail_count)
        set_tag(attrs.RUN_AGENT_CASES_WITH_PROVENANCE, len(provenance_by_idx))
        set_tag(attrs.RUN_AGENT_BACKEND_USED, backends_used)
        return self._build_eval_results(outputs, cases_data, run_name, trace_path, provenance_by_idx)

    @observe_safe("optimizer.run_parallel_subprocess")
    def _run_parallel_subprocess(
        self,
        runner: AgentRunner,
        dataset: list[dict],
        run_name: str,
        trace_path: Path | None = None,
        plan: BackendPlan | None = None,
    ) -> tuple[dict, list[ParsedTrace], list[dict]]:
        results_by_idx: dict[int, dict | None] = {}
        provenance_by_idx: dict[int, list[dict]] = {}
        backends_used: dict[str, int] = {}
        success_count = 0
        fail_count = 0
        counters_lock = threading.Lock()

        def _run_one(case: dict, idx: int) -> tuple[int, dict | None, list[dict], str, bool]:
            self._logger.debug(f"[{run_name}] Dispatching case {idx} on worker thread")
            backend_output = self._run_case_with_plan(plan, case["input"], trace_path, idx, run_name)
            run_output = backend_output.run_output
            prov = [t.to_dict() for t in backend_output.provenance]
            if run_output.success:
                self._logger.debug(
                    f"[{run_name}] Case {idx} succeeded backend={backend_output.backend} "
                    f"(stderr_bytes={len(run_output.stderr or '')}, prov={len(prov)})"
                )
                return idx, run_output.data, prov, backend_output.backend, True
            self._logger.warning(
                f"[{run_name}] Case {idx} failed backend={backend_output.backend} "
                f"rc={run_output.returncode} err={(run_output.error or '')[:300]}"
            )
            return idx, {"error": run_output.error}, prov, backend_output.backend, False

        with Progress(
            SpinnerColumn(style=BRAND),
            TextColumn(f"[bold {BRAND}]{{task.description}}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("  Running agent…", total=len(dataset))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                # Propagate the active OTel context to every worker thread
                # so per-case spans (e.g. optimizer.run_case_with_plan,
                # overmind_llm_completion) nest under the parent iteration
                # / workflow span instead of becoming orphan roots.
                parent_ctx = contextvars.copy_context()
                futures = {
                    pool.submit(parent_ctx.copy().run, _run_one, case, idx): idx for idx, case in enumerate(dataset)
                }
                last_trace_flush = time.monotonic()
                for future in as_completed(futures):
                    idx_result, output, prov, backend_name, ok = future.result()
                    results_by_idx[idx_result] = output
                    if prov:
                        provenance_by_idx[idx_result] = prov
                    with counters_lock:
                        backends_used[backend_name] = backends_used.get(backend_name, 0) + 1
                        if ok:
                            success_count += 1
                        else:
                            fail_count += 1
                    progress.advance(task)
                    now = time.monotonic()
                    if now - last_trace_flush >= 1.0:
                        force_flush_traces(timeout_millis=300)
                        last_trace_flush = now
                force_flush_traces(timeout_millis=1000)

        outputs = [results_by_idx[i] for i in range(len(dataset))]
        set_tag(attrs.RUN_AGENT_RUN_NAME, run_name)
        set_tag(attrs.RUN_AGENT_CASES_TOTAL, len(dataset))
        set_tag(attrs.RUN_AGENT_CASES_SUCCEEDED, success_count)
        set_tag(attrs.RUN_AGENT_CASES_FAILED, fail_count)
        set_tag(attrs.RUN_AGENT_CASES_WITH_PROVENANCE, len(provenance_by_idx))
        set_tag(attrs.RUN_AGENT_BACKEND_USED, backends_used)
        return self._build_eval_results(outputs, dataset, run_name, trace_path, provenance_by_idx)

    @observe_safe("optimizer.run_case_with_plan")
    def _run_case_with_plan(
        self,
        plan: BackendPlan | None,
        input_data,
        trace_path: Path | None,
        idx: int,
        run_name: str,
    ) -> BackendOutput:
        """Execute a single case using the backend plan with fallback.

        Tries each backend in order; bails out on the first success or on a
        non-retryable failure (missing API key, import error, …).  The
        returned :class:`BackendOutput` carries provenance tags and a
        :class:`Confidence` that the evaluator/optimizer can inspect.

        *trace_path* is the run-level trace directory (``traces_dir/<run>/``);
        we derive a per-case file ``case_<idx>.jsonl`` under it so concurrent
        worker subprocesses never share an append target. Pass ``None`` to
        disable local trace capture entirely.
        """
        # Stamp per-case context up front so the span carries it even if
        # the run raises before we get to the "after" tags below.
        set_tag(attrs.RUN_CASE_RUN_NAME, run_name)
        set_tag(attrs.RUN_CASE_INDEX, int(idx))
        if isinstance(input_data, dict):
            set_tag(attrs.RUN_CASE_INPUT_KEYS, sorted(str(k) for k in input_data)[:32])
        try:
            input_chars = len(json.dumps(input_data, default=str))
        except Exception:
            input_chars = len(str(input_data))
        set_tag(attrs.RUN_CASE_INPUT_CHARS, int(input_chars))

        case_trace_file: Path | None = None
        if trace_path is not None:
            case_trace_file = trace_path / f"case_{idx:04d}.jsonl"

        if plan is None or len(plan) == 0:
            raise RuntimeError("BackendPlan missing — cannot run case")

        backends_tried: list[str] = []
        started = time.monotonic()
        last_output: BackendOutput | None = None
        for i, backend in enumerate(plan):
            backend.prepare()
            backends_tried.append(backend.name)
            out = backend.run(input_data, trace_file=case_trace_file)
            last_output = out
            if out.success:
                if i > 0:
                    self._logger.info(
                        f"[{run_name}] Case {idx} recovered via backend={backend.name} after subprocess failure."
                    )
                self._stamp_run_case_outcome(out, backends_tried=backends_tried, started=started)
                return out
            if not should_try_next(out.diagnosis):
                self._stamp_run_case_outcome(out, backends_tried=backends_tried, started=started)
                return out
        assert last_output is not None
        self._stamp_run_case_outcome(last_output, backends_tried=backends_tried, started=started)
        return last_output

    def _stamp_run_case_outcome(
        self,
        out: BackendOutput,
        *,
        backends_tried: list[str],
        started: float,
    ) -> None:
        """Stamp per-case result tags on the active ``run_case_with_plan`` span."""
        set_tag(attrs.RUN_CASE_BACKEND_ATTEMPTS, len(backends_tried))
        set_tag(attrs.RUN_CASE_BACKENDS_TRIED, list(backends_tried))
        set_tag(attrs.RUN_CASE_BACKEND_USED, str(out.backend))
        set_tag(attrs.RUN_CASE_SUCCESS, bool(out.success))
        set_tag(attrs.RUN_CASE_RETURNCODE, int(getattr(out.run_output, "returncode", 0) or 0))
        if out.error:
            set_tag(attrs.RUN_CASE_ERROR, str(out.error)[:2000])
        try:
            output_chars = len(json.dumps(out.data, default=str)) if out.data is not None else 0
        except Exception:
            output_chars = len(str(out.data)) if out.data is not None else 0
        set_tag(attrs.RUN_CASE_OUTPUT_CHARS, int(output_chars))
        set_tag(attrs.RUN_CASE_DURATION_SECONDS, float(time.monotonic() - started))
        set_tag(attrs.RUN_CASE_PROVENANCE_COUNT, len(out.provenance or []))
        confidence = getattr(out, "confidence", None)
        if confidence is not None:
            try:
                set_tag(attrs.RUN_CASE_CONFIDENCE, confidence.to_dict())
            except Exception:
                self._logger.debug("run_case: failed to serialise confidence", exc_info=True)

    def _load_case_traces(self, trace_dir: Path | None, n_cases: int) -> list[ParsedTrace]:
        """Read back per-case ``ParsedTrace``s from the run's trace directory.

        Each ``trace_dir / case_<idx>.jsonl`` was written by the agent
        subprocess (via :class:`JsonlFileSpanExporter` installed in the
        wrapper bootstrap when ``OVERMIND_TRACE_FILE`` was set). Missing
        or unreadable files yield an empty :class:`ParsedTrace` so the
        caller always gets a list of length *n_cases*.
        """
        if trace_dir is None:
            return [ParsedTrace() for _ in range(n_cases)]

        results: list[ParsedTrace] = []
        for idx in range(n_cases):
            case_file = Path(trace_dir) / f"case_{idx:04d}.jsonl"
            if case_file.exists():
                try:
                    results.append(parse_trace_file(case_file))
                    continue
                except Exception as exc:
                    self._logger.warning(f"_load_case_traces: failed to parse {case_file}: {exc}")
            results.append(ParsedTrace())
        return results

    @observe_safe("optimizer.build_eval_results")
    def _build_eval_results(
        self,
        outputs: list[dict | None],
        dataset: list[dict],
        run_name: str,
        trace_path: Path | None,
        provenance_by_idx: dict[int, list[dict]] | None = None,
    ) -> tuple[dict, list[ParsedTrace], list[dict]]:
        """Build per-case eval items from agent outputs and shadow provenance.

        When *trace_path* is a directory, each case's per-process trace
        file (``trace_path / case_<idx>.jsonl``) is parsed back into a
        :class:`ParsedTrace` so :class:`SpecEvaluator` sees the actual
        ``tool_trace`` for Tool Usage scoring and the LLM token counts.
        Missing or empty files yield an empty :class:`ParsedTrace` and
        the evaluator falls back to its "dimension unscorable" path.

        When *provenance_by_idx* is provided (populated by the shadow
        backend), each :class:`ParsedTrace` is decorated with per-call
        source tags and the per-case score gains ``_confidence`` /
        ``_source_summary`` metadata so the optimizer can reason about how
        trustworthy the signal is.
        """
        provenance_by_idx = provenance_by_idx or {}

        traces: list[ParsedTrace] = []
        eval_items: list[dict] = []

        per_line_traces: list[ParsedTrace] = self._load_case_traces(trace_path, len(dataset))

        # Attach shadow provenance tags in bulk — keeps ParsedTrace and
        # tool_trace rows in sync with what actually ran.
        sidecar_tags = [provenance_by_idx.get(i, []) for i in range(len(dataset))]
        attach_shadow_provenance(per_line_traces, sidecar_tags)

        for idx, (output, case) in enumerate(zip(outputs, dataset)):
            parsed_trace = per_line_traces[idx]
            traces.append(parsed_trace)
            case_prov = provenance_by_idx.get(idx, [])

            expected = case.get("expected_output", case.get("expected", {}))
            tool_trace = parsed_trace.tool_trace
            tool_calls = [t["name"] for t in tool_trace]

            score = self.evaluator.evaluate_output(
                output,
                expected,
                input_data=case.get("input"),
                tool_trace=tool_trace,
                _skip_judge=True,
                source_tags=case_prov,
            )

            eval_items.append({
                "input": case.get("input"),
                "output": output,
                "expected": expected,
                "score": score,
                "tool_calls": tool_calls,
                "tool_trace": tool_trace,
                "source_tags": case_prov,
            })

        batch_eval = self.evaluator.evaluate_batch(eval_items)

        set_tag(attrs.RUN_AGENT_RUN_NAME, run_name)
        set_tag(attrs.RUN_AGENT_CASES_TOTAL, len(dataset))
        set_tag(attrs.RUN_AGENT_CASES_WITH_PROVENANCE, len(provenance_by_idx))
        return batch_eval, traces, eval_items

    # ------------------------------------------------------------------
    # Code update animation
    # ------------------------------------------------------------------

    def _applying_changes_panel_title(self, label: str | None = None) -> Text:
        """Title for the diff panel: file whose content is being shown."""
        if label is None:
            if self._bundle and self._bundle.is_multi_file():
                label = self._bundle.entry_file
            else:
                label = rel(Path(self.config.agent_path))
        title = Text()
        title.append("Applying changes")
        title.append(" · ")
        title.append(label, style="cyan")
        return title

    def _animate_single_file_diff(self, old_code: str, new_code: str, label: str | None = None) -> None:
        """Render an animated diff panel for a single file."""
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        opcodes = difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes()

        diff_lines: list[tuple[str, str]] = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for ln in old_lines[i1:i2]:
                    diff_lines.append(("equal", ln.rstrip("\n")))
            elif tag == "replace":
                for ln in old_lines[i1:i2]:
                    diff_lines.append(("remove", ln.rstrip("\n")))
                for ln in new_lines[j1:j2]:
                    diff_lines.append(("add", ln.rstrip("\n")))
            elif tag == "delete":
                for ln in old_lines[i1:i2]:
                    diff_lines.append(("remove", ln.rstrip("\n")))
            elif tag == "insert":
                for ln in new_lines[j1:j2]:
                    diff_lines.append(("add", ln.rstrip("\n")))

        context = 3
        visible: list[tuple[str, str]] = []
        for vi, (kind, line) in enumerate(diff_lines):
            if kind != "equal":
                visible.append((kind, line))
                continue
            near_change = False
            for offset in range(-context, context + 1):
                neighbor = vi + offset
                if 0 <= neighbor < len(diff_lines) and diff_lines[neighbor][0] != "equal":
                    near_change = True
                    break
            if near_change:
                visible.append((kind, line))

        if not visible:
            return

        rendered = Text()
        delay = max(0.03, min(0.12, 6.0 / len(visible)))
        panel_title = self._applying_changes_panel_title(label)

        with Live(
            Panel(rendered, title=panel_title, border_style=BRAND),
            console=self.console,
            refresh_per_second=30,
        ) as live:
            for kind, line in visible:
                if kind == "remove":
                    rendered.append(f"- {line}\n", style="bold red")
                elif kind == "add":
                    rendered.append(f"+ {line}\n", style="bold green")
                else:
                    rendered.append(f"  {line}\n", style="dim")
                live.update(Panel(rendered, title=panel_title, border_style=BRAND))
                time.sleep(delay)

        self.console.print()

    def _animate_code_update(
        self,
        old_code: str,
        new_code: str,
        resolved_files: dict[str, str] | None = None,
        prev_files: dict[str, str] | None = None,
    ) -> None:
        """Animate the diff for an accepted candidate.

        For multi-file candidates, shows a diff panel per changed file.
        For single-file candidates, shows the entry-point diff as before.
        """
        if resolved_files and prev_files:
            for file_path, new_source in sorted(resolved_files.items()):
                old_source = prev_files.get(file_path, "")
                if old_source != new_source:
                    self._animate_single_file_diff(old_source, new_source, label=file_path)
        else:
            self._animate_single_file_diff(old_code, new_code)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @observe_safe("optimizer.validate_code")
    def _validate_code(self, code: str) -> bool:
        from overmind.optimize.runner import (
            _validate_js_entrypoint,
            _validate_python_entrypoint,
            _validate_python_syntax,
        )

        fn_name = self.config.entrypoint_fn
        lang = self._runner.language

        if lang == Language.PYTHON:
            if not _validate_python_syntax(code):
                return False
            if not _validate_python_entrypoint(code, fn_name):
                return False
            # Skip module-level import validation — agent code often has
            # heavy side effects at import time (SDK init, dotenv, MCP
            # connections) that fail outside the real agent environment.
            # AST-level syntax + entrypoint checks are sufficient here;
            # the actual subprocess runner will catch real import errors.
        else:
            if not _validate_js_entrypoint(code, fn_name):
                return False

        return True

    # ------------------------------------------------------------------
    # Model backtesting
    # ------------------------------------------------------------------

    @observe_safe("optimizer.run_backtesting")
    def _run_backtesting(self, dataset: list[dict]):
        for model_id in self.config.backtest_models:
            with otel_span(
                "optimizer.backtest_model",
                attributes={attrs.OPTIMIZE_BACKTEST_MODEL: model_id},
            ):
                self.console.print(f"\n  Testing with [bold]{model_id}[/bold]…")
                modified_code = re.sub(
                    r'MODEL\s*=\s*"[^"]*"',
                    f'MODEL = "{model_id}"',
                    self.best_code,
                )
                ext = Path(self.config.agent_path).suffix or ".py"
                tmp_path = self.output_dir / f"agent_backtest{ext}"
                tmp_path.write_text(modified_code)

                try:
                    bt_eval, _, _ = self._run_agent_on_dataset(
                        str(tmp_path),
                        dataset,
                        f"backtest_{model_id.replace('/', '_')}",
                    )
                    self.backtest_results[model_id] = bt_eval
                    self.console.print(f"    Score: [cyan]{bt_eval['avg_total']:.1f}[/cyan] / 100")
                    set_tag(
                        attrs.OPTIMIZE_BACKTEST_SCORE,
                        float(bt_eval.get("avg_total", 0)),
                    )
                except Exception as exc:
                    self.console.print(f"    [red]Failed: {exc}[/red]")
                    self.backtest_results[model_id] = {
                        "avg_total": 0,
                        "error": str(exc),
                    }

        if self.backtest_results:
            dim_labels = self.evaluator.get_dimension_labels()

            table = Table(title="Model Backtesting Results", border_style="magenta")
            table.add_column("Model", style="bold")
            table.add_column("Avg Score", justify="right")
            for display, _ in dim_labels:
                table.add_column(display, justify="right")

            for mid, res in sorted(
                self.backtest_results.items(),
                key=lambda x: x[1].get("avg_total", 0),
                reverse=True,
            ):
                row = [mid, f"{res.get('avg_total', 0):.1f}"]
                for _, key in dim_labels:
                    row.append(f"{res.get(key, 0):.1f}")
                table.add_row(*row)

            self.console.print()
            self.console.print(table)

    # ------------------------------------------------------------------
    # Logging & reporting
    # ------------------------------------------------------------------

    @observe_safe("optimizer.setup_output_dirs")
    def _setup_output_dirs(self):
        for d in (self.output_dir, self.traces_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        results_tsv = self.output_dir / "results.tsv"
        if not results_tsv.exists():
            dim_cols = "\t".join(key for _, key in self.evaluator.get_dimension_labels())
            header = f"iteration\tavg_score\t{dim_cols}\tstatus\tdescription\n"
            results_tsv.write_text(header)

    def _log_result(self, iteration: str, evaluation: dict, status: str, desc: str):
        row: dict[str, str] = {
            "iteration": iteration,
            "avg_score": f"{evaluation.get('avg_total', 0):.1f}",
        }
        for _, key in self.evaluator.get_dimension_labels():
            row[key] = f"{evaluation.get(key, 0):.1f}"
        row["status"] = status
        row["description"] = desc.replace("\t", " ")

        self.results.append(row)
        line = "\t".join(row.values()) + "\n"
        with open(self.output_dir / "results.tsv", "a") as f:
            f.write(line)

    def _print_eval(
        self,
        evaluation: dict,
        label: str,
        prev_evaluation: dict | None = None,
    ):
        score = evaluation.get("avg_total", 0)
        color = "green" if score >= 70 else "yellow" if score >= 40 else "red"

        if prev_evaluation:
            prev_score = prev_evaluation.get("avg_total", 0)
            delta = score - prev_score
            d_color = "green" if delta > 0 else "red" if delta < 0 else "dim"
            sign = "+" if delta > 0 else ""
            self.console.print(
                f"  [bold]{label}[/bold] — avg score: "
                f"[{color}]{prev_score:.1f} \u2192 {score:.1f}[/{color}] "
                f"[{d_color}]({sign}{delta:.1f})[/{d_color}] / 100"
            )
        else:
            self.console.print(f"  [bold]{label}[/bold] — avg score: [{color}]{score:.1f}[/{color}] / 100")

        max_scores = self.evaluator.get_max_scores()
        for display, key in self.evaluator.get_dimension_labels():
            val = evaluation.get(key, 0)
            max_val = max_scores.get(key, 0)
            if max_val == 0 and val == 0:
                continue
            if prev_evaluation:
                prev_val = prev_evaluation.get(key, 0)
                delta = val - prev_val
                d_color = "green" if delta > 0 else "red" if delta < 0 else "dim"
                sign = "+" if delta > 0 else ""
                self.console.print(
                    f"    {display:>18}: {val:.1f} / {max_val:.0f}  [{d_color}]({sign}{delta:.1f})[/{d_color}]"
                )
            else:
                self.console.print(f"    {display:>18}: {val:.1f} / {max_val:.0f}")

    @observe_safe("optimizer.generate_report")
    def _generate_report(self):
        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()
        self.console.print(
            Panel(
                "[bold]Optimization Complete[/bold]",
                border_style="green",
            )
        )

        table = Table(title="Optimization History", border_style="cyan")
        table.add_column("Iteration", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        table.add_column("Description")

        for row in self.results:
            status = row["status"]
            style = "green" if status == "keep" else "red" if status in ("discard", "crash") else "yellow"
            table.add_row(
                row["iteration"],
                row["avg_score"],
                f"[{style}]{status}[/{style}]",
                row["description"][:60],
            )

        self.console.print(table)

        baseline = self.results[0] if self.results else {}
        baseline_score = float(baseline.get("avg_score", 0))
        improvement = self.best_score - baseline_score

        self.console.print()
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Baseline score:", f"{baseline_score:.1f}")
        summary.add_row("Best score:", f"[bold green]{self.best_score:.1f}[/bold green]")
        if improvement > 0:
            summary.add_row("Improvement:", f"[bold]+{improvement:.1f} points[/bold]")
        holdout = getattr(self, "_holdout_results", None)
        if holdout:
            ho_gap = holdout["overfit_gap"]
            gap_style = "green" if ho_gap <= 5 else "yellow" if ho_gap <= 10 else "red"
            blended = holdout.get("blended_improvement", 0)
            bl_style = "green" if blended > 0 else "red"
            ho_w = holdout.get("holdout_weight", 0.3)
            summary.add_row(
                "Holdout score:",
                f"{holdout['holdout_score']:.1f} ({holdout['holdout_improvement']:+.1f})",
            )
            summary.add_row(
                "Blended improvement:",
                f"[{bl_style}]{blended:+.1f}[/{bl_style}] ({1 - ho_w:.0%} train, {ho_w:.0%} holdout)",
            )
            summary.add_row(
                "Overfit gap:",
                f"[{gap_style}]{ho_gap:.1f} pts[/{gap_style}]",
            )
            if holdout.get("reverted"):
                if holdout.get("rollback_iteration"):
                    summary.add_row(
                        "Action:",
                        f"[bold yellow]Selected iteration "
                        f"{holdout['rollback_iteration']} snapshot "
                        f"(best blended score)[/bold yellow]",
                    )
                else:
                    summary.add_row(
                        "Action:",
                        "[bold red]Reverted to baseline (no snapshot with positive blended improvement)[/bold red]",
                    )
        rel_out = self.output_dir
        try:
            rel_out = self.output_dir.relative_to(Path.cwd())
        except ValueError:
            pass
        if self._bundle and self._bundle.is_multi_file():
            summary.add_row(
                "Best agent:",
                f"[cyan]{rel_out / 'best_agent/'}[/cyan] (multi-file)",
            )
        else:
            _ext = Path(self.config.agent_path).suffix or ".py"
            summary.add_row("Best agent:", f"[cyan]{rel_out / f'best_agent{_ext}'}[/cyan]")
        summary.add_row("Results log:", f"[cyan]{rel_out / 'results.tsv'}[/cyan]")
        summary.add_row("Traces:", f"[cyan]{rel_out / 'traces/'}[/cyan]")
        self.console.print(Panel(summary, border_style="green", title="Summary"))

        self._write_report_md(baseline_score)

    @observe_safe("optimizer.write_report_md")
    def _write_report_md(self, baseline_score: float):
        dim_labels = self.evaluator.get_dimension_labels()

        policy_line = ""
        if self._policy_data:
            n_rules = len(self._policy_data.get("domain_rules", self._policy_data.get("decision_rules", [])))
            n_constraints = len(
                self._policy_data.get("output_constraints", self._policy_data.get("hard_constraints", []))
            )
            policy_line = f"**Policy:** {n_rules} domain rule(s), {n_constraints} constraint(s)\n"

        lines = [
            "# Overmind Optimization Report\n",
            f"**Agent:** `{self.config.agent_path}`\n",
            f"**Iterations:** {self.config.iterations}\n",
            f"**Candidates per iteration:** {getattr(self.config, 'candidates_per_iteration', 1)}\n",
            f"**Analyzer model:** `{self.config.analyzer_model}`\n",
        ]
        if policy_line:
            lines.append(policy_line)
        lines += [
            "",
            "## Results\n",
            "| Baseline | Best | Improvement |",
            "|----------|------|-------------|",
            f"| {baseline_score:.1f} | {self.best_score:.1f} | +{self.best_score - baseline_score:.1f} |",
            "",
            "## Iteration Log\n",
            "| Iteration | Score | Status | Description |",
            "|-----------|-------|--------|-------------|",
        ]
        for row in self.results:
            lines.append(f"| {row['iteration']} | {row['avg_score']} | {row['status']} | {row['description'][:80]} |")

        if self.backtest_results:
            bt_header_cols = " | ".join(d for d, _ in dim_labels)
            bt_sep_cols = " | ".join("---" for _ in dim_labels)
            lines.extend([
                "",
                "## Model Backtesting\n",
                f"| Model | Score | {bt_header_cols} |",
                f"|-------|-------| {bt_sep_cols} |",
            ])
            for mid, res in sorted(
                self.backtest_results.items(),
                key=lambda x: x[1].get("avg_total", 0),
                reverse=True,
            ):
                dim_vals = " | ".join(f"{res.get(k, 0):.1f}" for _, k in dim_labels)
                lines.append(f"| {mid} | {res.get('avg_total', 0):.1f} | {dim_vals} |")

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n")
