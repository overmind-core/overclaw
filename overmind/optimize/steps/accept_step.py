"""``overmind optimize-step accept`` — apply gates, promote winner.

This MVP implementation applies the simplest meaningful gate: pick the
candidate with the highest ``avg_total`` and promote it iff its score is
strictly greater than the current best. Cross-run regression suite,
holdout enforcement, complexity penalty, and failure-cluster updates
will be ported in a follow-up.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from overmind import SpanType, attrs, set_tag
from overmind.optimize.optimizer import Optimizer
from overmind.optimize.steps.state import SkillRunState
from overmind.tracing import observe_safe

logger = logging.getLogger("overmind.optimize.steps.accept")


def _prune_git_worktrees(candidate_dirs: list[str]) -> None:
    """Remove git worktree registrations for all candidate directories.

    Called after accept/reject so stale worktree entries don't accumulate.
    Falls back silently if git is unavailable or the paths aren't git worktrees.
    """
    for candidate_dir in candidate_dirs:
        p = Path(candidate_dir)
        if not p.exists():
            continue
        try:
            git_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if git_result.returncode != 0:
                continue
            git_root = git_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(p)],
                cwd=git_root,
                capture_output=True,
                timeout=15,
            )
        except Exception as exc:
            logger.debug(f"Skipping worktree prune for {candidate_dir}: {exc}")


def _load_candidate_results(path: str) -> list[dict]:
    """Read the candidate-results JSON.

    Accepts either a list of ``{candidate_id, score_path, candidate_dir, ...}``
    entries or a dict ``{candidates: [...]}``.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    if not isinstance(raw, list):
        raise TypeError(f"candidate_results must be a list, got {type(raw).__name__}")
    return raw


def _read_only_cache_key(read_only_scope: list[str]) -> str:
    """Stable key for the cached read-only baseline.

    JSON-serialising the sorted scope list keeps the key resilient to
    insertion order while still detecting genuine config changes (e.g.
    a user adds a new fixture to ``read_only_paths`` mid-run).
    """
    return json.dumps(sorted(read_only_scope))


def _load_or_build_read_only_baseline(state: SkillRunState, cfg) -> tuple[dict[str, str], set[str]]:
    """Return ``(baseline_files, read_only_paths)`` for the accept gate.

    Reads from :attr:`SkillRunState.read_only_baseline` when the cache
    key matches the current ``read_only_scope``. Otherwise builds the
    bundle once, stuffs the captured baseline content into state, and
    saves. Subsequent iterations skip the BFS entirely.

    Bundle-build failures degrade to "no baseline" rather than crash —
    the accept step still runs (we just can't enforce read-only on
    this iteration). That matches the prior behaviour of the inline
    rebuild and keeps the optimizer running on transient I/O hiccups.
    """
    current_key = _read_only_cache_key(list(cfg.read_only_scope))
    if state.read_only_baseline and state.read_only_baseline_key == current_key:
        return (
            dict(state.read_only_baseline),
            set(state.read_only_baseline.keys()),
        )

    try:
        optimizer = Optimizer(cfg)
        optimizer._bundle = optimizer._build_bundle()
        if optimizer._bundle is None:
            return {}, set()
        ro_paths = set(optimizer._bundle.read_only_files)
        baseline = {rel: src for rel, src in optimizer._bundle.original_files.items() if rel in ro_paths}
    except Exception as exc:
        logger.warning(f"accept: bundle rebuild failed; skipping read-only check: {exc}")
        return {}, set()

    # Persist for next iteration — invalidated automatically when the
    # scope list changes (cache_key check above).
    state.read_only_baseline = baseline
    state.read_only_baseline_key = current_key
    state.save()
    return baseline, ro_paths


def _candidate_violates_read_only(
    candidate_dir: str,
    read_only_paths: set[str],
    baseline_files: dict[str, str],
) -> list[str]:
    """Return the read-only files a candidate worktree modified vs. baseline.

    An empty list means the candidate is clean. A non-empty list contains
    every read-only file that was modified, deleted, or unreadable. The
    comparison is strict byte-equality against the bundle's captured
    baseline content; this catches even whitespace-only edits, which is
    intentional — read-only means read-only.

    Files that are in ``read_only_paths`` but absent from
    ``baseline_files`` are skipped (we can't know what the baseline looked
    like) rather than reported as violations, so a misconfigured spec
    can't false-positive-reject every candidate.
    """
    if not read_only_paths or not baseline_files:
        return []
    violated: list[str] = []
    wt = Path(candidate_dir)
    for rel in sorted(read_only_paths):
        baseline_src = baseline_files.get(rel)
        if baseline_src is None:
            continue
        wt_file = wt / rel
        if not wt_file.is_file():
            violated.append(rel)
            continue
        try:
            wt_src = wt_file.read_text(encoding="utf-8")
        except Exception:
            violated.append(rel)
            continue
        if wt_src != baseline_src:
            violated.append(rel)
    return violated


@observe_safe(span_name="overmind.optimize.accept", type=SpanType.WORKFLOW)
def run_accept(
    state: SkillRunState,
    *,
    iteration: int,
    candidate_results_path: str,
) -> dict[str, Any]:
    set_tag(attrs.OPTIMIZE_ITERATION, iteration)
    set_tag(attrs.OPTIMIZE_PHASE, "accept")
    set_tag(attrs.OPTIMIZE_STEP, "accept")
    if state.job_id:
        set_tag(attrs.JOB_ID, state.job_id)
    candidates = _load_candidate_results(candidate_results_path)
    if not candidates:
        state.bump_stall()
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "score": state.best_score,
            "status": "skip",
            "description": "No candidates to evaluate",
        })
        state.save()
        return {
            "status": "ok",
            "step": "accept",
            "iteration": iteration,
            "decision": "no_candidates",
            "best_score": state.best_score,
            "stall_count": state.stall_count,
        }

    # Resolve scores from each score.json
    scored: list[dict] = []
    for cand in candidates:
        score_path = cand.get("score_path")
        if not score_path:
            continue
        try:
            score_data = json.loads(Path(score_path).read_text())
            scored.append({
                "candidate_id": cand["candidate_id"],
                "candidate_dir": cand["candidate_dir"],
                "entry_path": cand.get("entry_path"),
                "avg_total": float(score_data["avg_total"]),
                "evaluation": score_data["evaluation"],
                "case_results": score_data["case_results"],
            })
        except Exception as exc:
            logger.warning(f"Failed to load score for {cand.get('candidate_id')}: {exc}")

    if not scored:
        state.bump_stall()
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "score": state.best_score,
            "status": "crash",
            "description": "All candidates failed to evaluate",
        })
        state.save()
        return {
            "status": "ok",
            "step": "accept",
            "iteration": iteration,
            "decision": "all_crashed",
            "best_score": state.best_score,
            "stall_count": state.stall_count,
        }

    cfg = state.to_config()

    # ---- Read-only enforcement -----------------------------------------
    # Before picking a winner by score, drop any candidate whose worktree
    # mutated a file declared as ``read_only_paths`` in the eval spec.
    # This is the only gate that holds the line for harness files like the
    # registered entrypoint: PROMPT.md guidance + analyzer steering are
    # advisory; this is enforcement.
    #
    # Cache the baseline content on ``SkillRunState`` so we
    # rebuild the bundle at most once per run, not once per iteration.
    # The cache key is the serialized read_only_scope list; if the user
    # edits their spec mid-run (rare), the key changes and we rebuild.
    read_only_violations: list[dict] = []
    if cfg.read_only_scope:
        baseline_files, read_only_paths = _load_or_build_read_only_baseline(state, cfg)

        if read_only_paths:
            clean: list[dict] = []
            for cand in scored:
                violated = _candidate_violates_read_only(cand["candidate_dir"], read_only_paths, baseline_files)
                if violated:
                    logger.warning(f"Rejecting candidate {cand['candidate_id']}: modified read_only files: {violated}")
                    read_only_violations.append({
                        "candidate_id": cand["candidate_id"],
                        "candidate_dir": cand["candidate_dir"],
                        "avg_total": cand["avg_total"],
                        "files": violated,
                    })
                else:
                    clean.append(cand)
            scored = clean

    if not scored:
        # Every candidate violated the read-only invariant. Treat this as
        # a stall (similar to all_crashed) but surface a distinct decision
        # so the host skill can hint the user that the diagnosis prompt
        # needs tightening, not that the agents crashed.
        state.bump_stall()
        state.failed_attempts.append({
            "iteration": iteration,
            "reason": "read_only_violation",
            "violations": read_only_violations,
        })
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "score": state.best_score,
            "status": "reject",
            "description": (
                "All candidates modified read_only files: "
                + ", ".join(f"{v['candidate_id']}->{v['files']}" for v in read_only_violations)
            ),
        })
        state.save()

        early_stop = cfg.early_stopping_patience > 0 and state.stall_count >= cfg.early_stopping_patience
        if early_stop:
            state.early_stopping_triggered = True
            state.save()

        _prune_git_worktrees([c["candidate_dir"] for c in candidates if c.get("candidate_dir")])

        return {
            "status": "ok",
            "step": "accept",
            "iteration": iteration,
            "decision": "read_only_violation",
            "violations": read_only_violations,
            "best_score": state.best_score,
            "stall_count": state.stall_count,
            "early_stop": early_stop,
        }

    scored.sort(key=lambda c: -c["avg_total"])
    winner = scored[0]
    delta = winner["avg_total"] - state.best_score

    if winner["avg_total"] > state.best_score:
        # Promote: write working_path, update state, reset stall
        optimizer = Optimizer(cfg)
        # Hydrate the optimizer enough that commit_winner can compute paths
        optimizer.best_score = state.best_score
        optimizer.best_code = (
            Path(state.working_path or cfg.agent_path).read_text()
            if Path(state.working_path or cfg.agent_path).is_file()
            else ""
        )
        optimizer._baseline_train_score = state.baseline_score or state.best_score

        optimizer.commit_winner(
            winner_entry_path=winner["entry_path"],
            winner_eval=winner["evaluation"],
            winner_case_results=winner["case_results"],
        )
        # commit_winner wrote both agent_working.* and _latest_items.json
        new_working_path = str(optimizer.output_dir / f"agent_working{Path(cfg.agent_path).suffix or '.py'}")

        state.update_best(
            score=winner["avg_total"],
            iteration=iteration,
            code_path=new_working_path,
            files_dir=state.best_files_dir,
        )
        state.working_path = new_working_path
        state.successful_changes.append({
            "iteration": iteration,
            "candidate_id": winner["candidate_id"],
            "delta": delta,
            "from": state.results[-1].get("score") if state.results else None,
            "to": winner["avg_total"],
        })
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "candidate_id": winner["candidate_id"],
            "score": winner["avg_total"],
            "status": "accept",
            "description": f"Improved by {delta:+.2f}",
        })
        decision = "accept"
    else:
        state.bump_stall()
        state.failed_attempts.append({
            "iteration": iteration,
            "best_candidate_id": winner["candidate_id"],
            "best_score": winner["avg_total"],
            "current_best": state.best_score,
            "delta": delta,
        })
        state.record_iteration({
            "iteration": f"iter_{iteration:03d}",
            "candidate_id": winner["candidate_id"],
            "score": winner["avg_total"],
            "status": "reject",
            "description": f"No improvement ({delta:+.2f})",
        })
        decision = "reject"

    state.save()

    early_stop = cfg.early_stopping_patience > 0 and state.stall_count >= cfg.early_stopping_patience
    if early_stop:
        state.early_stopping_triggered = True
        state.save()

    _prune_git_worktrees([c["candidate_dir"] for c in candidates if c.get("candidate_dir")])

    # ---- Stamp iteration outcome on the active OTel span so the OTLP
    # ingest can build the matching ``JobIteration`` row.  Mirrors the
    # tags the legacy CLI loop sets on its ``optimizer.iteration`` span
    # (decision / score / dimension_scores / suggestions / agent_code /
    # per-candidate analytics).
    iter_decision = "keep" if decision == "accept" else "discard"
    set_tag(attrs.OPTIMIZE_ITERATION_DECISION, iter_decision)
    set_tag(attrs.OPTIMIZE_ITERATION_SCORE, float(winner["avg_total"]))
    set_tag(attrs.OPTIMIZE_ITERATION_IMPROVEMENT, float(delta))
    set_tag(
        attrs.OPTIMIZE_ITERATION_REASON,
        f"{'Improved' if decision == 'accept' else 'No improvement'} ({delta:+.2f})",
    )
    set_tag(attrs.OPTIMIZE_STALL_COUNT, state.stall_count)
    set_tag(attrs.OPTIMIZE_ACCEPTED, bool(decision == "accept"))

    # Winner code → ``JobIteration.agent_code``.
    winner_entry = winner.get("entry_path")
    if winner_entry and Path(winner_entry).is_file():
        try:
            set_tag(attrs.OPTIMIZE_ITERATION_AGENT_CODE, Path(winner_entry).read_text())
        except Exception:
            logger.debug("accept: failed to read winner entry file", exc_info=True)

    # Per-dimension scores from the winning candidate's evaluation
    # payload.  ``evaluate_worktree`` writes the full evaluator output
    # into ``score.json`` → ``winner["evaluation"]``.
    eval_payload = winner.get("evaluation") or {}
    dim_scores = {k: float(v) for k, v in eval_payload.items() if k != "avg_total" and isinstance(v, (int, float))}
    if dim_scores:
        set_tag(attrs.OPTIMIZE_ITERATION_DIMENSION_SCORES, dim_scores)

    # Suggestions: re-read plan.json for the winning candidate (the
    # diagnose step persisted them so we can surface them now).
    suggestions: list[str] = []
    plan_path = Path(winner.get("candidate_dir", "")) / "plan.json"
    if plan_path.is_file():
        try:
            plan_data = json.loads(plan_path.read_text())
            suggestions = [str(s) for s in (plan_data.get("suggestions") or [])]
        except Exception:
            logger.debug("accept: failed to read winner plan.json", exc_info=True)
    if suggestions:
        set_tag(attrs.OPTIMIZE_ITERATION_SUGGESTIONS, suggestions)

    # Emit one short-lived ``optimizer.evaluate_candidate``-shaped span
    # per scored candidate so OTLP's ``_build_candidate_entry`` can
    # construct the ``c0 / c1 / c2`` candidate badges on the
    # JobIteration row.  Lives entirely inside the accept span (we open
    # and close each one synchronously) so they all carry the same
    # ``overmind.job.id`` via context propagation.
    from overmind import start_span as _otel_span  # local import: avoid heavy import on cold step

    for cand in scored:
        with _otel_span(
            "optimizer.evaluate_candidate",
            attributes={
                attrs.OPTIMIZE_ITERATION: int(iteration),
                attrs.OPTIMIZE_CANDIDATE_METHOD: str(cand["candidate_id"]),
                attrs.OPTIMIZE_CANDIDATE_SCORE: float(cand["avg_total"]),
                attrs.OPTIMIZE_CANDIDATE_ADJUSTED_SCORE: float(cand["avg_total"]),
            },
        ):
            pass

    if early_stop:
        # Early stopping ends the run on the accept step itself —
        # signal terminal state so OTLP flips ``Job.status`` to
        # ``completed``.
        set_tag(attrs.OPTIMIZE_RUN_STATUS, "completed")
        set_tag(attrs.OPTIMIZE_FINAL_BEST_SCORE, float(state.best_score))
        set_tag(
            attrs.OPTIMIZE_REPORT_IMPROVEMENT,
            float(state.best_score - (state.baseline_score or state.best_score)),
        )

    envelope: dict[str, Any] = {
        "status": "ok",
        "step": "accept",
        "iteration": iteration,
        "decision": decision,
        "winner": {
            "candidate_id": winner["candidate_id"],
            "avg_total": winner["avg_total"],
            "delta": delta,
        },
        "all_scores": [{"candidate_id": c["candidate_id"], "avg_total": c["avg_total"]} for c in scored],
        "best_score": state.best_score,
        "stall_count": state.stall_count,
        "early_stop": early_stop,
    }
    if read_only_violations:
        # Surface skipped-over candidates so the host coding agent can
        # tighten its diagnosis prompt or detect a runaway sub-agent.
        envelope["read_only_violations"] = read_only_violations
    return envelope
