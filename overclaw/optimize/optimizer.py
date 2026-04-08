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

import difflib
import importlib.util
import random
import re
import shutil
import statistics
import tempfile
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

from overclaw.utils.code import AgentBundle
from overclaw.core.paths import agent_experiments_dir, agent_run_state_path
from overclaw.utils.display import BRAND, make_spinner_progress, rel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from overclaw.client import ApiReporter, flush_pending_api_updates
from overclaw.optimize.analyzer import (
    compute_focus_weights,
    format_component_weights,
    generate_candidates,
)
from overclaw.optimize.config import Config
from overclaw.optimize.data import load_data
from overclaw.optimize.evaluator import (
    SpecEvaluator,
    load_evaluator,
)
from overclaw.optimize.failure_registry import (
    FailureRegistry,
    format_clusters_for_diagnosis,
)
from overclaw.optimize.run_state import RunState, RunSummary
from overclaw.utils.policy import (
    format_for_codegen,
    format_for_diagnosis,
    format_for_judge,
    load_policy_data,
)
from overclaw.core.tracer import Tracer, set_current_tracer
from overclaw.storage import get_storage


class Optimizer:
    """Runs the full optimization pipeline for an agent."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        try:
            self._storage = get_storage()
        except ValueError:
            self._storage = None
        self._reporter: ApiReporter | None = None

        # Load policy from eval spec for injection into pipeline stages
        import json as _json

        with open(config.eval_spec_path) as _f:
            _spec = _json.load(_f)
        self._policy_data = load_policy_data(_spec)
        self._policy_diagnosis = format_for_diagnosis(self._policy_data or {})
        self._policy_codegen = format_for_codegen(self._policy_data or {})
        self._policy_judge = format_for_judge(self._policy_data or {})

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        self._setup_output_dirs()
        dataset = self._load_dataset()

        # Split into train (optimizer sees) and holdout (final generalization check)
        holdout_ratio = getattr(self.config, "holdout_ratio", 0.2)
        train_set, holdout_set = self._split_dataset(dataset, holdout_ratio)

        self.console.print()
        self.console.print(
            Rule(f"[bold {BRAND}]OverClaw Agent Optimizer[/bold {BRAND}]", style=BRAND)
        )
        agent_label = (
            f"{self.config.agent_name}  [dim]({self.config.agent_path})[/dim]"
            if getattr(self.config, "agent_name", "")
            else self.config.agent_path
        )
        info_lines = (
            f"  [dim]Agent:[/dim]  {agent_label}\n"
            f"  [dim]Cases:[/dim]  {len(dataset)} total"
        )
        if holdout_set:
            info_lines += f" ({len(train_set)} train, {len(holdout_set)} holdout)"
        info_lines += f"\n  [dim]Model:[/dim]  {self.config.analyzer_model}"
        if self._policy_data:
            n_rules = len(
                self._policy_data.get(
                    "domain_rules", self._policy_data.get("decision_rules", [])
                )
            )
            n_constraints = len(
                self._policy_data.get(
                    "output_constraints", self._policy_data.get("hard_constraints", [])
                )
            )
            info_lines += (
                f"\n  [dim]Policy:[/dim] {n_rules} rule(s), "
                f"{n_constraints} constraint(s)"
            )
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
        if self._bundle and self._bundle.is_multi_file():
            n_files = len(self._bundle.original_files)
            n_opt = self._bundle.optimizable_file_count()
            self.console.print(
                f"  [dim]Bundle:[/dim] {n_files} file(s) resolved, {n_opt} optimizable"
            )

        baseline_eval, baseline_traces, baseline_items = self._run_agent_on_dataset(
            self.config.agent_path, train_set, "baseline"
        )
        self.best_score = baseline_eval["avg_total"]
        self._baseline_train_score = self.best_score
        self.best_code = baseline_code
        self.best_case_scores = [item["score"]["total"] for item in baseline_items]

        # Start real-time API reporting once we know baseline score.
        if self.config.agent_id:
            self._reporter = ApiReporter.create(
                agent_id=self.config.agent_id,
                analyzer_model=self.config.analyzer_model,
                num_iterations=self.config.iterations,
                candidates_per_iteration=getattr(
                    self.config,
                    "candidates_per_iteration",
                    1,
                ),
            )
            if self._reporter and self._storage:
                self._storage.set_job_id(self._reporter.job_id)
                self._reporter.on_baseline(self.best_score)

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
                self.console.print(
                    f"  [dim]Failure clusters: {len(touched)} identified, "
                    f"{open_n} open[/dim]"
                )

        # Baseline diagnostics
        self._print_baseline_diagnostics(baseline_eval, baseline_items)

        # Track multi-file state
        if self._bundle:
            self._baseline_files = dict(self._bundle.original_files)
            self._best_files = dict(self._bundle.original_files)

        # Working copy
        working_path = self.output_dir / "agent_working.py"
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
            temperature = t_start - (t_start - t_end) * (i - 1) / max(
                self.config.iterations - 1, 1
            )

            # Stall detection: increase exploration
            if self.stall_count >= 2:
                temperature = min(temperature + 0.2, 1.0)
                self.console.print(
                    "  [yellow]Detected stall — increasing exploration[/yellow]"
                )

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
                )
                _component_ctx = format_component_weights(_focus_weights)

                top_focus = max(_focus_weights, key=_focus_weights.get)  # type: ignore[arg-type]
                top_pct = _focus_weights[top_focus] * 100
                self.console.print(
                    f"  [dim]Focus targeting:[/dim] {top_focus} ({top_pct:.0f}%)"
                )

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
                        allow_model_change=bool(
                            self.config.model_backtesting
                            and self.config.backtest_models
                        ),
                        num_candidates=n_candidates,
                        temperature=temperature,
                        diagnosis_case_fraction=getattr(
                            self.config, "diagnosis_case_fraction", 0.7
                        ),
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
                            f"    Candidate {idx + 1} ({method}): "
                            f"[yellow]bundle splice validation failed[/yellow]"
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
                    self.console.print(
                        f"    Candidate {idx + 1} ({method}): [yellow]{reason}[/yellow]"
                    )
                    continue
                if not self._validate_code(code):
                    ext = Path(self.config.agent_path).suffix or ".txt"
                    failed_path = self.output_dir / f"failed_iter_{i:03d}_c{idx}{ext}"
                    failed_path.write_text(code)
                    self.console.print(
                        f"    Candidate {idx + 1} ({method}): "
                        f"[yellow]syntax/interface validation failed[/yellow]"
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
                        cand_table.add_row(
                            str(vidx + 1), vcand.get("method", "unknown")
                        )
                else:
                    cand_table.add_column("Change summary", overflow="fold")
                    for vidx, vcand in valid:
                        method = vcand.get("method", "unknown")
                        sugs = vcand.get("suggestions") or []
                        if sugs:
                            summary_cell = Text(
                                "; ".join(str(s) for s in sugs), overflow="fold"
                            )
                        else:
                            summary_cell = Text("—", style="dim")
                        cand_table.add_row(str(vidx + 1), method, summary_cell)
                self.console.print(cand_table)

            if not valid:
                self.console.print(
                    "  [yellow]No valid candidates this iteration.[/yellow]"
                )
                self._log_result(
                    f"iter_{i:03d}", latest_eval, "skip", "No valid candidates"
                )
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
                        self.console.print(
                            f"    Candidate {idx + 1}: [red]crashed in smoke test[/red]"
                        )
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
                        "  [yellow]All candidates failed smoke test, "
                        "proceeding with full eval anyway.[/yellow]"
                    )

            # --- Step 3: Evaluate candidates ---
            self.console.print(
                f"  [dim]Step 3:[/dim] Evaluating {len(valid)} candidate(s) "
                f"against test cases"
            )
            best_cand = None
            best_cand_eval = None
            best_cand_score = -1.0
            best_cand_items = None
            best_cand_case_scores: list[float] = []

            for orig_idx, cand in valid:
                tmp_path = self._write_candidate_to_disk(cand)
                try:
                    runs_per = getattr(self.config, "runs_per_eval", 1)
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
                    c_eval = None
                    c_items = None
                finally:
                    self._cleanup_candidate(tmp_path, cand)

                if c_eval is None:
                    self.console.print(
                        f"    Candidate {orig_idx + 1}: [red]crashed[/red]"
                    )
                    continue

                c_score = c_eval["avg_total"]
                self.console.print(
                    f"    Candidate {orig_idx + 1}: [cyan]{c_score:.1f}[/cyan] / 100"
                )

                complexity_penalty = self._compute_complexity_penalty(
                    cand["updated_code"],
                    train_set=train_set,
                    raw_score=c_score,
                )
                adjusted_score = c_score - complexity_penalty
                if complexity_penalty > 0:
                    self.console.print(
                        f"      [dim]Complexity penalty: "
                        f"-{complexity_penalty:.1f} → {adjusted_score:.1f}[/dim]"
                    )

                if adjusted_score > best_cand_score:
                    best_cand = cand
                    best_cand_eval = c_eval
                    best_cand_score = adjusted_score
                    best_cand_items = c_items
                    best_cand_case_scores = [item["score"]["total"] for item in c_items]

            if best_cand is None or best_cand_eval is None:
                self.console.print(
                    "  [yellow]All candidates crashed. Reverting.[/yellow]"
                )
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
                        f"  [dim]Confirmed score: {confirmed_score:.1f} "
                        f"(adjusted: {confirmed_adjusted:.1f})[/dim]"
                    )
                    best_cand_eval = confirm_eval
                    best_cand_score = confirmed_adjusted
                    best_cand_items = confirm_items
                    best_cand_case_scores = [
                        item["score"]["total"] for item in confirm_items
                    ]
                except Exception:
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
                    reason = (
                        f"{reason}; regression gate: {reg_fail}/{n_reg} minor "
                        f"regressions (within threshold)"
                    )

            if accept:
                improvement = best_cand_score - self.best_score
                self.console.print(
                    f"\n  [bold green]\u2713 Accepted: {self.best_score:.1f} \u2192 "
                    f"{best_cand_score:.1f} (+{improvement:.1f})[/bold green]"
                )
                if reason:
                    self.console.print(f"    [dim]{reason}[/dim]")
                self._animate_code_update(self.best_code, best_cand["updated_code"])
                dim_deltas = self._compute_dimension_deltas(latest_eval, best_cand_eval)
                change_record = {
                    "suggestions": best_cand.get("suggestions", []),
                    "improvement": (
                        f"+{improvement:.1f} "
                        f"({self.best_score:.1f} \u2192 {best_cand_score:.1f})"
                    ),
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

                self.accepted_snapshots.append(
                    {
                        "code": self.best_code,
                        "files": (dict(self._best_files) if self._best_files else None),
                        "train_score": best_cand_score,
                        "iteration": i,
                    }
                )
                latest_eval = best_cand_eval
                latest_case_results = self._build_case_results(
                    best_cand_items, train_set
                )

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
                            f"    [green]\u2713 Resolved {len(newly_resolved)} "
                            f"failure cluster(s)[/green]"
                        )
                        self._promote_resolved_to_regression(
                            newly_resolved,
                            latest_case_results,
                            train_set,
                            i,
                        )

                self._log_result(f"iter_{i:03d}", best_cand_eval, "keep", desc)
                if self._reporter:
                    dim_scores = {
                        key: float(best_cand_eval.get(key, 0))
                        for _, key in self.evaluator.get_dimension_labels()
                    }
                    self._reporter.on_iteration(
                        order=i,
                        avg_score=float(best_cand_eval.get("avg_total", 0)),
                        decision="keep",
                        agent_code=self.best_code,
                        description=desc,
                        dimension_scores=dim_scores,
                    )
                self.stall_count = 0
            else:
                self.console.print(
                    f"\n  [red]\u2717 Rejected: {best_cand_score:.1f} "
                    f"vs best {self.best_score:.1f}[/red]"
                )
                if reason:
                    self.console.print(f"    [dim]{reason}[/dim]")
                working_path.write_text(self.best_code)
                self._log_result(f"iter_{i:03d}", best_cand_eval, "discard", desc)
                if self._reporter:
                    dim_scores = {
                        key: float(best_cand_eval.get(key, 0))
                        for _, key in self.evaluator.get_dimension_labels()
                    }
                    self._reporter.on_iteration(
                        order=i,
                        avg_score=float(best_cand_eval.get("avg_total", 0)),
                        decision="discard",
                        agent_code=best_cand.get("updated_code"),
                        description=desc,
                        dimension_scores=dim_scores,
                    )
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

            # Early stopping
            patience = getattr(self.config, "early_stopping_patience", 3)
            if patience > 0 and self.stall_count >= patience:
                self.console.print(
                    f"\n  [yellow]Early stopping: {self.stall_count} consecutive "
                    f"iterations without improvement "
                    f"(patience={patience}).[/yellow]"
                )
                break

        # Save best agent
        best_path = self.output_dir / "best_agent.py"
        best_path.write_text(self.best_code)

        # Save multi-file output when applicable
        if self._best_files and self._bundle and self._bundle.is_multi_file():
            best_dir = self.output_dir / "best_agent"
            self._write_file_set(best_dir, self._best_files)
            self.console.print(
                f"  [dim]Multi-file output: {best_dir}/ "
                f"({len(self._best_files)} files)[/dim]"
            )

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
            holdout_eval, _, holdout_items = self._run_agent_on_dataset(
                str(best_path), holdout_set, "holdout"
            )
            holdout_score = holdout_eval["avg_total"]

            baseline_holdout_eval, _, _ = self._run_agent_on_dataset(
                self.config.agent_path, holdout_set, "holdout_baseline"
            )
            baseline_holdout_score = baseline_holdout_eval["avg_total"]
            train_improvement = self.best_score - self._baseline_train_score
            holdout_improvement = holdout_score - baseline_holdout_score

            holdout_w = getattr(self.config, "holdout_weight", 0.3)
            blended_improvement = (
                1 - holdout_w
            ) * train_improvement + holdout_w * holdout_improvement

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
            catastrophic_threshold = getattr(
                self.config, "catastrophic_holdout_threshold", 0.5
            )

            is_catastrophic = (
                baseline_holdout_score > 0
                and holdout_score < baseline_holdout_score * catastrophic_threshold
            )

            needs_rollback = holdout_enforcement and (
                is_catastrophic or blended_improvement < 0
            )

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
                    blended_improvement = (
                        1 - holdout_w
                    ) * train_improvement + holdout_w * holdout_improvement
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
                self.console.print(
                    "\n  [green]Holdout performance confirms generalization.[/green]"
                )

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
                "rollback_iteration": (
                    rollback_target["iteration"] if rollback_target else None
                ),
            }
            if self._reporter:
                self._reporter.on_holdout(self._holdout_results)

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
        self._generate_report()
        if self._reporter:
            report_md = (self.output_dir / "report.md").read_text(encoding="utf-8")
            self._reporter.on_complete(
                best_score=self.best_score,
                baseline_score=self._baseline_train_score,
                report_markdown=report_md,
                best_agent_code=self.best_code,
                backtest_results=self.backtest_results or None,
            )
            flush_pending_api_updates(timeout=20.0)

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
                f"\n  [dim]Cross-run state saved: {n_clusters} cluster(s), "
                f"{n_reg} regression case(s)[/dim]"
            )

    # ------------------------------------------------------------------
    # Complexity penalty (prompt bloat + code growth + override detection)
    # ------------------------------------------------------------------

    def _compute_complexity_penalty(
        self,
        candidate_code: str,
        train_set: list[dict] | None = None,
        raw_score: float | None = None,
    ) -> float:
        """Penalize candidates with excessive prompt, code, or logic growth.

        Four dimensions (all use quadratic ramps so small overshoots get
        tiny penalties while large overshoots are still meaningful):

        1. SYSTEM_PROMPT bloat (vs original baseline)
        2. Total code size growth (vs original baseline, size-adaptive threshold)
        3. New conditional branches (vs original baseline)
        4. Hardcoded expected-output literals (data leakage from training set)

        When *raw_score* is provided the total penalty is capped at 60% of
        the raw improvement over the current best, ensuring genuine
        improvements always yield at least partial net gain.
        """
        penalty = 0.0

        # 1. Prompt bloat (vs original baseline, threshold 2.0x)
        baseline_prompt = self._get_prompt_size(self._baseline_code or self.best_code)
        cand_prompt = self._get_prompt_size(candidate_code)
        if baseline_prompt > 0:
            prompt_ratio = cand_prompt / baseline_prompt
            if prompt_ratio > 2.0:
                overshoot = prompt_ratio - 2.0
                penalty += min(3.0, overshoot**2 * 2.0)

        # 2. Total code growth (vs original baseline, size-adaptive)
        if self._baseline_code:
            baseline_lines = len(self._baseline_code.splitlines())
            candidate_lines = len(candidate_code.splitlines())
            if baseline_lines > 0:
                max_ratio = getattr(self.config, "max_code_growth_ratio", 2.5)
                if baseline_lines < 150:
                    max_ratio += 1.0
                elif baseline_lines < 300:
                    max_ratio += 0.5
                code_ratio = candidate_lines / baseline_lines
                if code_ratio > max_ratio:
                    overshoot = code_ratio - max_ratio
                    penalty += min(5.0, overshoot**2 * 1.5)

        # 3. New conditional branches (vs original baseline)
        if self._baseline_code:
            baseline_branches = self._count_conditional_branches(self._baseline_code)
            candidate_branches = self._count_conditional_branches(candidate_code)
            new_branches = candidate_branches - baseline_branches
            if new_branches > 20:
                overshoot = new_branches - 20
                penalty += min(3.0, overshoot**2 * 0.01)

        # 4. Hardcoded expected-output literals from training data
        if train_set and self._baseline_code:
            leakage = self._detect_data_leakage(candidate_code, train_set)
            if leakage > 0:
                penalty += min(5.0, leakage * 1.5)

        # Cap penalty at 60% of the raw improvement so genuine gains always
        # produce at least partial net progress.
        if raw_score is not None and penalty > 0:
            raw_improvement = raw_score - self.best_score
            if raw_improvement > 0:
                max_allowed = raw_improvement * 0.6
                penalty = min(penalty, max_allowed)

        return penalty

    def _detect_data_leakage(self, candidate_code: str, train_set: list[dict]) -> int:
        """Count expected-output literals that appear in new code but not the baseline.

        Excludes known enum values from the eval spec (these are domain
        vocabulary the agent *should* reference, not leakage) and raises the
        minimum string length to 6 to avoid false positives on short generic
        words like "warm", "high", "cold".
        """
        new_lines = set(candidate_code.splitlines()) - set(
            self._baseline_code.splitlines()
        )
        new_code_text = "\n".join(new_lines)
        if not new_code_text.strip():
            return 0

        IGNORE_VALUES = {
            "",
            "true",
            "false",
            "none",
            "null",
            "yes",
            "no",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "10",
            "100",
        }

        known_domain_values: set[str] = set()
        for field_cfg in self.evaluator.spec.get("output_fields", {}).values():
            for v in field_cfg.get("values", []):
                known_domain_values.add(str(v).strip().lower())

        leaked = 0
        seen_values: set[str] = set()
        for case in train_set:
            expected = case.get("expected_output", case.get("expected", {}))
            if not isinstance(expected, dict):
                continue
            for val in expected.values():
                s = str(val).strip()
                if len(s) < 6 or s.lower() in IGNORE_VALUES:
                    continue
                if s.lower() in known_domain_values:
                    continue
                if s in seen_values:
                    continue
                if s in new_code_text:
                    leaked += 1
                    seen_values.add(s)
        return leaked

    @staticmethod
    def _get_prompt_size(code: str) -> int:
        m = re.search(
            r'SYSTEM_PROMPT\s*=\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')',
            code,
            re.DOTALL,
        )
        return len(m.group(1)) if m else 0

    @staticmethod
    def _count_conditional_branches(code: str) -> int:
        """Count if/elif branches as a proxy for post-processing complexity."""
        return sum(
            1
            for line in code.splitlines()
            if line.strip().startswith(("if ", "elif ", "if(", "elif("))
        )

    # ------------------------------------------------------------------
    # Dataset splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_dataset(
        dataset: list[dict], holdout_ratio: float
    ) -> tuple[list[dict], list[dict]]:
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
            self.console.print(
                f"    [dim]Iteration {snap['iteration']} "
                f"(train: {snap['train_score']:.1f})\u2026[/dim]"
            )
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
                baseline_holdout_score > 0
                and snap_holdout_score < baseline_holdout_score * catastrophic_threshold
            )
            if snap_is_catastrophic:
                self.console.print(
                    f"      [red]\u2717 Catastrophic holdout drop "
                    f"({snap_holdout_score:.1f} < "
                    f"{baseline_holdout_score * catastrophic_threshold:.1f})"
                    f"[/red]"
                )
                continue

            blended = (
                1 - holdout_weight
            ) * snap_train_imp + holdout_weight * snap_holdout_imp
            self.console.print(
                f"      holdout: {snap_holdout_score:.1f} "
                f"({snap_holdout_imp:+.1f}), "
                f"blended: {blended:+.1f}"
            )

            if blended > 0:
                scored.append(
                    (
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
                    )
                )

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Dimension delta computation
    # ------------------------------------------------------------------

    def _compute_dimension_deltas(
        self, old_eval: dict, new_eval: dict
    ) -> dict[str, float]:
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

    def _check_acceptance(
        self,
        candidate_score: float,
        candidate_case_scores: list[float],
        candidate_items: list[dict],
        dataset: list[dict],
    ) -> tuple[bool, str]:
        """Check if a candidate should be accepted.

        Uses a three-tier acceptance strategy:
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
            return False, (
                f"No improvement ({candidate_score:.1f} vs best {self.best_score:.1f})"
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
        net_improvement = candidate_score - self.best_score

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
            return False, (
                f"Too many regressions: {regressions}/{n} cases regressed "
                f"(threshold: {threshold:.0%})"
            )

        return True, (
            f"{improvements} improved, {regressions} regressed out of {n} cases"
        )

    # ------------------------------------------------------------------
    # Cross-run regression gate
    # ------------------------------------------------------------------

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
            agent = self._load_agent_module(str(tmp_path))
        except Exception:
            self._cleanup_candidate(tmp_path, candidate)
            return len(self._run_state.regression_cases)

        failures = 0
        for rc in self._run_state.regression_cases:
            tracer = Tracer(trace_id="regression_check")
            set_current_tracer(tracer)
            try:
                output = getattr(agent, self.config.entrypoint_fn)(rc.case_input)
            except Exception:
                failures += 1
                continue
            finally:
                tracer.finish()
                set_current_tracer(None)

            tool_trace = [
                {
                    "name": span.name,
                    "args": span.metadata.get("args", {}),
                    "result": span.metadata.get("result", {}),
                    "error": span.error,
                    "latency_ms": span.latency_ms,
                }
                for span in tracer.trace.spans
                if hasattr(span, "span_type") and span.span_type == "tool_call"
            ]
            score = self.evaluator.evaluate_output(
                output,
                rc.expected_output,
                input_data=rc.case_input,
                tool_trace=tool_trace,
                _skip_judge=True,
            )
            if score["total"] < rc.min_score:
                failures += 1

        self._cleanup_candidate(tmp_path, candidate)
        return failures

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
                    expected_output=case_data.get(
                        "expected_output", case_data.get("expected", {})
                    ),
                    min_score=max(score * 0.8, 50.0),
                    run_id=self._run_id,
                    iteration=iteration,
                    cluster_id=cluster.cluster_id,
                )

    # ------------------------------------------------------------------
    # Multi-run evaluation
    # ------------------------------------------------------------------

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
            r_eval, _, r_items = self._run_agent_on_dataset(
                agent_path, dataset, f"{run_name}_r{r}"
            )
            runs.append((r_eval["avg_total"], r_eval, r_items))

        all_scores = [t for t, _, _ in runs]

        if num_runs > 1:
            mean = statistics.mean(all_scores)
            stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            self.console.print(
                f"      [dim]Multi-run: mean={mean:.1f}, "
                f"stdev={stdev:.1f}, runs={all_scores}[/dim]"
            )

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
            results.append(
                {
                    "input": case.get("input", {}),
                    "expected": item["expected"],
                    "output": item["output"],
                    "score": item["score"],
                    "tool_calls": item.get("tool_calls", []),
                    "tool_trace": item.get("tool_trace", []),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Baseline diagnostics
    # ------------------------------------------------------------------

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
            self.console.print(
                f"  [dim]Saturated dimensions (already near-perfect): "
                f"{', '.join(saturated)}[/dim]"
            )
        if weak:
            self.console.print(
                "  [yellow]Weak dimensions (biggest improvement room):[/yellow]"
            )
            for name, val, mx in weak:
                self.console.print(
                    f"    {name}: {val:.1f}/{mx:.0f} ({val / mx * 100:.0f}%)"
                )

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
                self.console.print(
                    f"    [yellow]{cases_with_no_tools} cases used no tools[/yellow]"
                )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> list[dict]:
        """Load the dataset from disk.

        Data is prepared during ``overclaw setup`` (generated synthetically
        or analyzed/augmented from seed data).  The optimizer only loads it.
        """
        self.console.print(f"  [dim]Loading data from {self.config.data_path}…[/dim]")
        return load_data(self.config.data_path)

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
        from overclaw.core.registry import project_root_from_agent_file

        pr = project_root_from_agent_file(self.config.agent_path)
        if pr:
            try:
                rel = str(Path(self.config.agent_path).resolve().relative_to(pr))
            except ValueError:
                rel = Path(self.config.agent_path).name
        else:
            rel = Path(self.config.agent_path).name
        return {rel: current_code}

    def _build_bundle(self) -> AgentBundle | None:
        """Build an ``AgentBundle`` from the current config."""
        from overclaw.core.registry import project_root, project_root_from_agent_file

        pr = project_root_from_agent_file(self.config.agent_path)
        if pr is None:
            try:
                pr = project_root()
            except SystemExit:
                return None
        project_root_str = str(pr)

        opt_scope = getattr(self.config, "optimizable_scope", []) or []
        try:
            if opt_scope:
                return AgentBundle.from_entry_point(
                    self.config.agent_path,
                    project_root_str,
                    self.config.entrypoint_fn,
                    optimizable_paths=opt_scope,
                )
            return AgentBundle.from_entry_point(
                self.config.agent_path,
                project_root_str,
                self.config.entrypoint_fn,
            )
        except Exception:
            return None

    def _rebuild_bundle(self) -> None:
        """Rebuild the bundle from current ``_best_files`` state.

        Called after accepting a multi-file candidate so subsequent
        iterations see the updated file contents and pieces.
        """
        if not self._bundle or not self._best_files:
            return

        from overclaw.utils.code import extract_pieces

        self._bundle.original_files = dict(self._best_files)
        new_pieces = []
        opt_files = set(self._bundle.optimizable_files)

        ordered_paths = [self._bundle.entry_file] + [
            p for p in self._best_files if p != self._bundle.entry_file
        ]

        for rel_path in ordered_paths:
            if rel_path not in self._best_files:
                continue
            source = self._best_files[rel_path]
            is_opt = rel_path in opt_files
            pieces = extract_pieces(rel_path, source, optimizable=is_opt)
            new_pieces.extend(pieces)

        self._bundle.pieces = new_pieces
        self._bundle._assign_ids()

    def _resolve_bundle_candidate(self, bundle_updates: dict) -> dict | None:
        """Resolve bundle updates into modified files.

        Supports two formats:
        - ``file_updates``: whole-file replacements (preferred)
        - ``piece_updates``: legacy line-range splice (fallback)

        Returns ``{"entry_code": str, "files": {rel_path: source}}`` or
        ``None`` if validation fails.
        """
        if not self._bundle:
            return None

        from overclaw.utils.code import has_entrypoint_ast

        file_updates = bundle_updates.get("file_updates")
        if file_updates:
            modified = self._bundle.apply_file_updates(file_updates)
        else:
            piece_updates = bundle_updates.get("piece_updates", {})
            new_pieces = bundle_updates.get("new_pieces", [])
            modified = self._bundle.apply_updates(piece_updates, new_pieces)

        if modified is None:
            return None

        full_files = dict(self._best_files) if self._best_files else {}
        full_files.update(modified)

        entry_code = full_files.get(self._bundle.entry_file)
        if not entry_code:
            return None

        fn = self.config.entrypoint_fn
        if not has_entrypoint_ast(entry_code, fn):
            return None

        return {"entry_code": entry_code, "files": modified}

    def _write_candidate_to_disk(self, cand: dict) -> Path:
        """Write a candidate to disk for evaluation, handling both modes.

        For multi-file candidates with ``_resolved_files``, creates a
        temporary directory tree.  For single-file, creates a temp ``.py``.
        Returns the path to the entry file.
        """
        resolved = cand.get("_resolved_files")
        if resolved and self._bundle:
            tmp_dir = Path(
                tempfile.mkdtemp(prefix="overclaw_", dir=str(self.output_dir))
            )
            all_files = self._bundle.get_full_file_set(resolved)
            for rel_path, source in all_files.items():
                dest = tmp_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source)
            return tmp_dir / self._bundle.entry_file

        tmp = Path(tempfile.mktemp(suffix=".py", dir=str(self.output_dir)))
        tmp.write_text(cand["updated_code"])
        return tmp

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

    @staticmethod
    def _load_agent_module(path: str):
        spec = importlib.util.spec_from_file_location("_agent_mod", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _run_agent_on_dataset(
        self,
        agent_path: str,
        dataset: list[dict],
        run_name: str,
    ) -> tuple[dict, list[Tracer], list[dict]]:
        agent = self._load_agent_module(agent_path)

        if self.config.parallel:
            return self._run_parallel(agent, agent_path, dataset, run_name)
        return self._run_sequential(agent, dataset, run_name)

    def _run_sequential(
        self, agent, dataset: list[dict], run_name: str
    ) -> tuple[dict, list[Tracer], list[dict]]:
        tracers: list[Tracer] = []
        eval_items: list[dict] = []

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
                tracer, score_item = self._run_single_case(agent, case, run_name, idx)
                tracers.append(tracer)
                eval_items.append(score_item)
                progress.advance(task)

        batch_eval = self.evaluator.evaluate_batch(eval_items)
        return batch_eval, tracers, eval_items

    def _run_parallel(
        self, agent, agent_path: str, dataset: list[dict], run_name: str
    ) -> tuple[dict, list[Tracer], list[dict]]:
        results_by_idx: dict[int, tuple[Tracer, dict]] = {}

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
                futures = {
                    pool.submit(self._run_single_case, agent, case, run_name, idx): idx
                    for idx, case in enumerate(dataset)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    tracer, score_item = future.result()
                    results_by_idx[idx] = (tracer, score_item)
                    progress.advance(task)

        tracers = []
        eval_items = []
        for idx in range(len(dataset)):
            tracer, score_item = results_by_idx[idx]
            tracers.append(tracer)
            eval_items.append(score_item)

        batch_eval = self.evaluator.evaluate_batch(eval_items)
        return batch_eval, tracers, eval_items

    def _run_single_case(
        self, agent, case: dict, run_name: str, idx: int
    ) -> tuple[Tracer, dict]:
        """Execute the agent on one test case and return (tracer, eval_item)."""
        tracer = Tracer(trace_id=f"{run_name}_{idx:03d}")
        set_current_tracer(tracer)
        tracer.set_input(case["input"])

        try:
            output = getattr(agent, self.config.entrypoint_fn)(case["input"])
        except Exception as exc:
            output = {"error": str(exc)}

        tracer.set_output(output)
        tracer.finish()
        set_current_tracer(None)

        trace_path = self.traces_dir / run_name / f"{idx:03d}.json"
        tracer.trace.save(str(trace_path))
        if self._reporter:
            trace_payload = tracer.trace.to_dict()
            trace_payload["trace_group"] = run_name
            self._reporter.on_trace(trace_payload)

        expected = case.get("expected_output", case.get("expected", {}))

        # Extract full tool trace data (args + results, not just names)
        tool_trace = [
            {
                "name": span.name,
                "args": span.metadata.get("args", {}),
                "result": span.metadata.get("result", {}),
                "error": span.error,
                "latency_ms": span.latency_ms,
            }
            for span in tracer.trace.spans
            if hasattr(span, "span_type") and span.span_type == "tool_call"
        ]

        tool_calls = [t["name"] for t in tool_trace]

        score = self.evaluator.evaluate_output(
            output,
            expected,
            input_data=case.get("input"),
            tool_trace=tool_trace,
        )
        tracer.trace.score = score["total"]

        return tracer, {
            "output": output,
            "expected": expected,
            "score": score,
            "tool_calls": tool_calls,
            "tool_trace": tool_trace,
        }

    # ------------------------------------------------------------------
    # Code update animation
    # ------------------------------------------------------------------

    def _applying_changes_panel_title(self) -> Text:
        """Title for the diff panel: file whose content is being shown."""
        if self._bundle and self._bundle.is_multi_file():
            label = self._bundle.entry_file
        else:
            label = rel(Path(self.config.agent_path))
        title = Text()
        title.append("Applying changes")
        title.append(" · ")
        title.append(label, style="cyan")
        return title

    def _animate_code_update(self, old_code: str, new_code: str) -> None:
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
                if (
                    0 <= neighbor < len(diff_lines)
                    and diff_lines[neighbor][0] != "equal"
                ):
                    near_change = True
                    break
            if near_change:
                visible.append((kind, line))

        if not visible:
            return

        rendered = Text()
        delay = max(0.03, min(0.12, 6.0 / len(visible)))
        panel_title = self._applying_changes_panel_title()

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_code(self, code: str) -> bool:
        from overclaw.utils.code import has_entrypoint_ast

        agent_path = Path(self.config.agent_path)
        ext = agent_path.suffix.lower()

        if ext == ".py":
            try:
                compile(code, "<agent>", "exec")
            except SyntaxError:
                return False

        fn_name = self.config.entrypoint_fn
        if not has_entrypoint_ast(code, fn_name):
            return False

        if ext == ".py":
            import tempfile

            tmp = Path(tempfile.mktemp(suffix=".py"))
            try:
                tmp.write_text(code)
                mod = self._load_agent_module(str(tmp))
                if not callable(getattr(mod, fn_name, None)):
                    return False
            except Exception:
                return False
            finally:
                tmp.unlink(missing_ok=True)

        return True

    # ------------------------------------------------------------------
    # Model backtesting
    # ------------------------------------------------------------------

    def _run_backtesting(self, dataset: list[dict]):
        for model_id in self.config.backtest_models:
            self.console.print(f"\n  Testing with [bold]{model_id}[/bold]…")
            modified_code = re.sub(
                r'MODEL\s*=\s*"[^"]*"',
                f'MODEL = "{model_id}"',
                self.best_code,
            )
            tmp_path = self.output_dir / "agent_backtest.py"
            tmp_path.write_text(modified_code)

            try:
                bt_eval, _, _ = self._run_agent_on_dataset(
                    str(tmp_path),
                    dataset,
                    f"backtest_{model_id.replace('/', '_')}",
                )
                self.backtest_results[model_id] = bt_eval
                self.console.print(
                    f"    Score: [cyan]{bt_eval['avg_total']:.1f}[/cyan] / 100"
                )
            except Exception as exc:
                self.console.print(f"    [red]Failed: {exc}[/red]")
                self.backtest_results[model_id] = {"avg_total": 0, "error": str(exc)}

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

    def _setup_output_dirs(self):
        for d in (self.output_dir, self.traces_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        results_tsv = self.output_dir / "results.tsv"
        if not results_tsv.exists():
            dim_cols = "\t".join(
                key for _, key in self.evaluator.get_dimension_labels()
            )
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
            self.console.print(
                f"  [bold]{label}[/bold] — avg score: "
                f"[{color}]{score:.1f}[/{color}] / 100"
            )

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
                    f"    {display:>18}: {val:.1f} / {max_val:.0f}"
                    f"  [{d_color}]({sign}{delta:.1f})[/{d_color}]"
                )
            else:
                self.console.print(f"    {display:>18}: {val:.1f} / {max_val:.0f}")

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
            style = (
                "green"
                if status == "keep"
                else "red"
                if status in ("discard", "crash")
                else "yellow"
            )
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
        summary.add_row(
            "Best score:", f"[bold green]{self.best_score:.1f}[/bold green]"
        )
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
                f"{holdout['holdout_score']:.1f} "
                f"({holdout['holdout_improvement']:+.1f})",
            )
            summary.add_row(
                "Blended improvement:",
                f"[{bl_style}]{blended:+.1f}[/{bl_style}] "
                f"({1 - ho_w:.0%} train, {ho_w:.0%} holdout)",
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
                        "[bold red]Reverted to baseline "
                        "(no snapshot with positive blended improvement)"
                        "[/bold red]",
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
            summary.add_row("Best agent:", f"[cyan]{rel_out / 'best_agent.py'}[/cyan]")
        summary.add_row("Results log:", f"[cyan]{rel_out / 'results.tsv'}[/cyan]")
        summary.add_row("Traces:", f"[cyan]{rel_out / 'traces/'}[/cyan]")
        self.console.print(Panel(summary, border_style="green", title="Summary"))

        self._write_report_md(baseline_score)

    def _write_report_md(self, baseline_score: float):
        dim_labels = self.evaluator.get_dimension_labels()

        policy_line = ""
        if self._policy_data:
            n_rules = len(
                self._policy_data.get(
                    "domain_rules", self._policy_data.get("decision_rules", [])
                )
            )
            n_constraints = len(
                self._policy_data.get(
                    "output_constraints", self._policy_data.get("hard_constraints", [])
                )
            )
            policy_line = (
                f"**Policy:** {n_rules} domain rule(s), {n_constraints} constraint(s)\n"
            )

        lines = [
            "# OverClaw Optimization Report\n",
            f"**Agent:** `{self.config.agent_path}`\n",
            f"**Iterations:** {self.config.iterations}\n",
            f"**Candidates per iteration:** "
            f"{getattr(self.config, 'candidates_per_iteration', 1)}\n",
            f"**Analyzer model:** `{self.config.analyzer_model}`\n",
        ]
        if policy_line:
            lines.append(policy_line)
        lines += [
            "",
            "## Results\n",
            "| Baseline | Best | Improvement |",
            "|----------|------|-------------|",
            f"| {baseline_score:.1f} | {self.best_score:.1f} "
            f"| +{self.best_score - baseline_score:.1f} |",
            "",
            "## Iteration Log\n",
            "| Iteration | Score | Status | Description |",
            "|-----------|-------|--------|-------------|",
        ]
        for row in self.results:
            lines.append(
                f"| {row['iteration']} | {row['avg_score']} | {row['status']} "
                f"| {row['description'][:80]} |"
            )

        if self.backtest_results:
            bt_header_cols = " | ".join(d for d, _ in dim_labels)
            bt_sep_cols = " | ".join("---" for _ in dim_labels)
            lines.extend(
                [
                    "",
                    "## Model Backtesting\n",
                    f"| Model | Score | {bt_header_cols} |",
                    f"|-------|-------| {bt_sep_cols} |",
                ]
            )
            for mid, res in sorted(
                self.backtest_results.items(),
                key=lambda x: x[1].get("avg_total", 0),
                reverse=True,
            ):
                dim_vals = " | ".join(f"{res.get(k, 0):.1f}" for _, k in dim_labels)
                lines.append(f"| {mid} | {res.get('avg_total', 0):.1f} | {dim_vals} |")

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n")
