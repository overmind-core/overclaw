"""``overmind optimize-step diagnose`` — produce N candidate plans + worktrees.

For each plan, this step:
  1. Creates a working directory ``experiments/iter_NNN_cI/`` containing a
     copy of the current best agent files.
  2. Writes ``PROMPT.md`` (the codegen instructions a host coding agent
     can hand to a sub-coding-agent).
  3. Writes ``plan.json`` (the structured plan: focus area, diagnosis,
     suggestions).

Returns a JSON envelope listing the worktrees so the skill can fan out
sub-agents in parallel.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import force_flush_traces, observe_safe

logger = logging.getLogger("overmind.optimize.steps.diagnose")


_PROMPT_HEADER = """\
# Code edit task — candidate {candidate_id} (iteration {iteration})

You are an expert coding agent improving an AI agent codebase.
Your job is to apply targeted edits to the source files in the working directory
so the next evaluation scores higher than the current best ({best_score:.1f}/100).

## Working directory

All source files live in this directory. The entry point is `{entry_file}`
and the function to keep callable is `{entrypoint_fn}(...)`.

Read `plan.json` in this directory for the full structured diagnosis
(focus area, root cause, suggested changes).

## Coding rules

- **Read before editing**: start by reading `{entry_file}` and any supporting
  files the diagnosis references. Understand the architecture first.
- When modifying a function, check its callers and callees for needed updates.
- Use grep/glob to locate code when you are unsure which file contains it.
- Preserve existing code style, imports, and conventions.
- Keep the entrypoint signature compatible (do not rename or remove `{entrypoint_fn}`).
- Prefer find-and-replace (edit) tools over full-file rewrites for existing files.
  Provide enough surrounding context in the old_string to ensure a unique match.
- After a non-trivial edit, re-read the modified file to confirm correctness.
- Verify syntax before finishing:
  `python -c "import ast, pathlib; ast.parse(pathlib.Path('{entry_file}').read_text())"`
- Ensure cross-file consistency: imports, function signatures, data flow.
- You MAY create new helper functions in existing or new files if the diagnosis
  calls for structural improvements.
- Do NOT add comments narrating your changes.
- Stop as soon as the requested change is applied. Do not add tests,
  rewrite unrelated files, or add unnecessary instrumentation.
- Do NOT create copies of files — edit files in place.

## Anti-overfitting rules (critical — violations cause automatic rejection)

The evaluation suite includes unseen holdout cases. Hardcoded or pattern-matched
fixes score well on training cases but fail on holdout and are rejected.

- Do NOT hardcode responses, field values, or answers for specific inputs seen
  in test results or the diagnosis output.
- Do NOT add `if`/`elif`/`match` branches that pattern-match on specific field
  values or example data to return a hardcoded result.
- Do NOT add post-processing formulas (e.g. `result = a * b`) that overwrite
  the LLM's output — the LLM often gets these right through judgment; a
  mechanical formula destroys those correct outputs.
- Do NOT add lookup tables, dictionaries, or maps keyed by example input values.
- Prefer general-purpose improvements: better prompt wording, smarter output
  parsing, cleaner logic flow, or new helper functions.
- Prefer structural improvements over input-specific rules.

## Focus for this candidate

{focus_area_section}

## Diagnosis (shared across candidates this iteration)

{diagnosis_block}

## Detailed edit instructions

{edit_instructions}

## When you are done

Print a final message that includes the literal token `OPTIMIZE_DONE` so the
parent skill knows you are finished. Do not commit or push anything.
"""


def _write_worktree(
    *,
    iteration: int,
    candidate_id: str,
    plan: dict,
    files: dict[str, str],
    entry_file: str,
    entrypoint_fn: str,
    best_score: float,
    output_dir: Path,
) -> Path:
    worktree = output_dir / f"iter_{iteration:03d}_{candidate_id}"

    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.mkdir(parents=True)

    for rel_path, source in files.items():
        target = worktree / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)

    diag = plan.get("diagnosis", {}) or {}
    diagnosis_block = diag.get("root_cause", "") or "(no diagnosis available)"
    if diag.get("changes"):
        diagnosis_block += "\n\n### Suggested changes\n"
        for c in diag["changes"]:
            diagnosis_block += f"- {c.get('action', '')}\n"

    focus = plan.get("focus_area") or "general"
    focus_area_section = f"**Focus area:** `{focus}` — this candidate should bias edits toward this area."

    prompt = _PROMPT_HEADER.format(
        candidate_id=candidate_id,
        iteration=iteration,
        best_score=best_score,
        entry_file=entry_file,
        entrypoint_fn=entrypoint_fn,
        focus_area_section=focus_area_section,
        diagnosis_block=diagnosis_block,
        edit_instructions=plan.get("edit_instructions", "(no detailed instructions)"),
    )
    (worktree / "PROMPT.md").write_text(prompt)

    (worktree / "plan.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "iteration": iteration,
                "method": plan.get("method"),
                "focus_area": focus,
                "diagnosis": diag,
                "suggestions": plan.get("suggestions", []),
            },
            indent=2,
            default=str,
        )
    )

    return worktree


@observe_safe(span_name="overmind.optimize.diagnose", type=SpanType.FUNCTION)
def run_diagnose(state: SkillRunState, *, iteration: int) -> dict[str, Any]:
    set_tag(attrs.OPTIMIZE_ITERATION, iteration)
    set_tag(attrs.OPTIMIZE_PHASE, "diagnose")
    set_tag(attrs.OPTIMIZE_STEP, "diagnose")
    if state.job_id:
        set_tag(attrs.JOB_ID, state.job_id)
    cfg = state.to_config()
    optimizer = Optimizer(cfg)

    # Re-seed cross-step mutable state so diagnosis sees prior history.
    optimizer.failed_attempts = list(state.failed_attempts)
    optimizer.successful_changes = list(state.successful_changes)
    optimizer.stall_count = int(state.stall_count)

    # Rebuild the bundle once so diagnosis prompts reference the right pieces.
    optimizer._bundle = optimizer._build_bundle()
    if optimizer._bundle:
        optimizer._best_files = dict(optimizer._bundle.original_files)

    # Read current best code (working_path is updated each accepted iteration).
    working_path = Path(state.working_path or state.best_code_path or cfg.agent_path)
    if not working_path.is_file():
        return {
            "status": "error",
            "error": "missing_working_copy",
            "message": f"No working agent found at {working_path}.",
        }
    current_code = working_path.read_text()

    latest_eval, latest_case_results = optimizer.load_latest_eval()

    from overmind.optimize import analyzer as _analyzer

    _analyzer.reset_last_diagnosis_error()
    plans = optimizer.run_diagnose_phase(
        iteration=iteration,
        current_code=current_code,
        latest_eval=latest_eval,
        latest_case_results=latest_case_results,
    )
    diagnose_error = _analyzer.get_last_diagnosis_error()

    # Build per-candidate worktrees so the host coding agent can edit them
    # in parallel without stepping on each other.
    output_dir = Path(state.output_dir)
    bundle_files = (
        optimizer._best_files
        if optimizer._bundle and optimizer._best_files
        else {Path(cfg.agent_path).name: current_code}
    )
    entry_file = optimizer._bundle.entry_file if optimizer._bundle else Path(cfg.agent_path).name

    worktrees = []
    for plan in plans:
        wt = _write_worktree(
            iteration=iteration,
            candidate_id=plan["candidate_id"],
            plan=plan,
            files=bundle_files,
            entry_file=entry_file,
            entrypoint_fn=cfg.entrypoint_fn,
            best_score=state.best_score,
            output_dir=output_dir,
        )
        worktrees.append({
            "candidate_id": plan["candidate_id"],
            "worktree": str(wt),
            "prompt_path": str(wt / "PROMPT.md"),
            "plan_path": str(wt / "plan.json"),
            "entry_file": entry_file,
            "entry_path": str(wt / entry_file),
            "method": plan.get("method"),
            "focus_area": plan.get("focus_area"),
        })

    state.phase = f"diagnose_complete_iter_{iteration}"
    state.iteration = iteration
    state.save()

    set_tag(attrs.OPTIMIZE_N_CANDIDATES_GENERATED, len(worktrees))
    set_tag(
        attrs.CANDIDATES_METHODS,
        [str(c.get("method") or "unknown") for c in worktrees],
    )
    force_flush_traces(timeout_millis=1500)

    requested = int(getattr(cfg, "candidates_per_iteration", len(worktrees)) or len(worktrees))
    all_failed = bool(worktrees) and all((c.get("method") or "").startswith("failed") for c in worktrees)
    degraded = all_failed or (requested > 0 and len(worktrees) < requested)

    envelope: dict[str, Any] = {
        "status": "warn" if degraded else "ok",
        "step": "diagnose",
        "state_path": state.state_path,
        "iteration": iteration,
        "n_candidates": len(worktrees),
        "candidates": worktrees,
    }
    if degraded:
        hint = (
            "Diagnosis LLM call failed; falling back to a single empty "
            "candidate. Check that the analyzer model's provider key "
            "(e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY) is present in "
            ".overmind/agents/<name>/.env or .overmind/.env."
        )
        envelope["diagnose_warning"] = {
            "requested_candidates": requested,
            "returned_candidates": len(worktrees),
            "all_failed": all_failed,
            "last_error": diagnose_error,
            "hint": hint,
        }
    return envelope
