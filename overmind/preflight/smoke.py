"""Run the agent against a tiny dataset slice and capture structured outcomes.

This is the only place inside :mod:`overmind.preflight` that actually
executes the agent.  Everything else (classifier, autofix, runner) operates
on the result struct emitted here.

Reuses the production :class:`overmind.optimize.runner.AgentRunner` and
:class:`overmind.optimize.evaluator.SpecEvaluator` so the preflight
behaviour is identical to what optimize will see — same subprocess
isolation, same shadow-runtime support, same scoring semantics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from overmind.core.paths import (
    agent_instrumented_dir,
    load_agent_dotenv,
    load_overmind_dotenv,
)
from overmind.core.registry import project_root_from_agent_file, resolve_agent
from overmind.optimize.data import load_data
from overmind.optimize.evaluator import load_evaluator
from overmind.optimize.runner import AgentRunner, RunnerConfig

logger = logging.getLogger("overmind.preflight.smoke")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Outcome of one row of the smoke dataset."""

    row_index: int
    success: bool
    output: Any = None
    expected: Any = None
    error: str = ""  # subprocess / runtime error
    score: float | None = None  # None when the scorer raised
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    scorer_error: str = ""  # populated when SpecEvaluator raised


@dataclass
class SmokeRunResult:
    """Aggregated result of one smoke pass."""

    cases: list[CaseResult] = field(default_factory=list)
    baseline_score: float | None = None
    span_count: int = 0
    preflight_error: str = ""  # set when env provisioning blew up before running anything

    def successful_outputs(self) -> list[Any]:
        return [c.output for c in self.cases if c.success and c.output is not None]

    def succeeded(self) -> int:
        return sum(1 for c in self.cases if c.success)

    def failed(self) -> int:
        return sum(1 for c in self.cases if not c.success)


# ---------------------------------------------------------------------------
# Dataset slicing
# ---------------------------------------------------------------------------


def select_smoke_cases(dataset: list[dict], max_rows: int) -> list[dict]:
    """Pick the first *max_rows* rows that have a usable ``input`` field.

    Stable / deterministic so re-runs operate on the same slice.
    """
    out: list[dict] = []
    for case in dataset:
        if not isinstance(case, dict):
            continue
        if "input" not in case and "input_data" not in case:
            continue
        out.append(case)
        if len(out) >= max_rows:
            break
    return out


# ---------------------------------------------------------------------------
# Runner construction
# ---------------------------------------------------------------------------


def _resolve_agent_dir(agent_name: str, agent_path: str) -> tuple[Path, str, Path | None]:
    """Return ``(agent_dir, entry_relpath, env_dir)`` for the runner.

    Mirrors ``Optimizer._build_runner`` semantics so preflight runs the
    instrumented copy when present, and falls back to the original tree
    otherwise.
    """
    p = Path(agent_path).resolve()
    inst_dir = agent_instrumented_dir(agent_name)
    if inst_dir.is_dir():
        # Always run from the instrumented copy when it exists.  The
        # entry file's relative location is preserved by
        # `instrument_agent_files`, so just rebase the path.
        proj_root = project_root_from_agent_file(agent_path)
        if proj_root is not None and p.is_relative_to(proj_root):
            entry_relpath = p.relative_to(proj_root)
        else:
            entry_relpath = Path(p.name)
        candidate = inst_dir / entry_relpath
        if candidate.is_file():
            env_dir = proj_root if proj_root is not None else p.parent
            return inst_dir, str(entry_relpath), env_dir

    proj_root = project_root_from_agent_file(agent_path)
    agent_dir = proj_root if proj_root is not None else p.parent
    entry_relpath = p.relative_to(agent_dir) if p.is_relative_to(agent_dir) else Path(p.name)
    return agent_dir, str(entry_relpath), None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_smoke(
    agent_name: str,
    *,
    eval_spec_path: str,
    dataset_path: str,
    max_rows: int = 2,
    timeout: int = 120,
    trace_file: Path | None = None,
) -> SmokeRunResult:
    """Run the agent against the first *max_rows* dataset rows and score them.

    All exceptions inside the runner / evaluator are converted into
    fields on the returned :class:`SmokeRunResult` — the function never
    raises.  This is the deterministic substrate the classifier consumes.
    """
    load_overmind_dotenv()
    load_agent_dotenv(agent_name)

    result = SmokeRunResult()

    try:
        agent_path, fn_name = resolve_agent(agent_name)
    except SystemExit as exc:
        result.preflight_error = f"agent_not_registered: {exc}"
        return result

    try:
        dataset = load_data(dataset_path)
    except Exception as exc:
        result.preflight_error = f"dataset_load_failed: {type(exc).__name__}: {exc}"
        return result

    cases_in = select_smoke_cases(dataset, max_rows)
    if not cases_in:
        result.preflight_error = "dataset_empty: no rows with an 'input' field"
        return result

    agent_dir, entry_file, env_dir = _resolve_agent_dir(agent_name, agent_path)
    runner = AgentRunner(
        agent_dir=agent_dir,
        entry_file=entry_file,
        entrypoint_fn=fn_name,
        config=RunnerConfig(timeout=timeout),
        env_dir=env_dir,
    )

    try:
        runner.ensure_environment()
    except Exception as exc:
        result.preflight_error = f"environment_provisioning_failed: {type(exc).__name__}: {exc}"
        return result

    # Build the evaluator up front so a totally broken spec surfaces
    # before we waste time invoking the agent.  The classifier turns
    # that into an `invalid_weights` / `metric_broken` issue.
    try:
        evaluator = load_evaluator(eval_spec_path)
    except Exception as exc:
        result.preflight_error = f"eval_spec_load_failed: {type(exc).__name__}: {exc}"
        return result

    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        if trace_file.exists():
            trace_file.unlink()

    case_scores: list[float] = []

    for idx, case in enumerate(cases_in):
        inp = case.get("input", case.get("input_data"))
        expected = case.get("expected_output", case.get("output"))

        try:
            run_output = runner.run(inp, trace_file=trace_file)
        except Exception as exc:
            result.cases.append(
                CaseResult(
                    row_index=idx,
                    success=False,
                    expected=expected,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if not run_output.success:
            result.cases.append(
                CaseResult(
                    row_index=idx,
                    success=False,
                    expected=expected,
                    error=run_output.error or run_output.stderr or "agent returned no output",
                )
            )
            continue

        score_dict: dict[str, Any] = {}
        score_total: float | None = None
        scorer_err = ""
        try:
            score_dict = evaluator.evaluate_output(
                run_output.data,
                expected,
                input_data=inp if isinstance(inp, dict) else None,
            )
            score_total = float(score_dict.get("total", 0.0))
        except Exception as exc:
            scorer_err = f"{type(exc).__name__}: {exc}"

        result.cases.append(
            CaseResult(
                row_index=idx,
                success=True,
                output=run_output.data,
                expected=expected,
                score=score_total,
                score_breakdown=score_dict,
                scorer_error=scorer_err,
            )
        )
        if score_total is not None:
            case_scores.append(score_total)

    runner.cleanup()

    if case_scores:
        result.baseline_score = sum(case_scores) / len(case_scores) / 100.0

    # Span count is best-effort — used only as a wiring health signal.
    if trace_file is not None and trace_file.is_file():
        try:
            with trace_file.open() as fh:
                result.span_count = sum(1 for line in fh if line.strip())
        except OSError:
            pass

    return result


def smoke_to_jsonable(result: SmokeRunResult) -> dict[str, Any]:
    """Render a :class:`SmokeRunResult` as plain JSON-friendly dicts."""
    return {
        "preflight_error": result.preflight_error,
        "baseline_score": result.baseline_score,
        "span_count": result.span_count,
        "succeeded": result.succeeded(),
        "failed": result.failed(),
        "cases": [
            {
                "row_index": c.row_index,
                "success": c.success,
                "score": c.score,
                "error": c.error[:400] if c.error else "",
                "scorer_error": c.scorer_error[:400] if c.scorer_error else "",
                "output_preview": json.dumps(c.output, default=str)[:400] if c.output is not None else "",
            }
            for c in result.cases
        ],
    }
