"""``overmind optimize-step`` — JSON-in/JSON-out pieces of the optimize loop.

Designed to be invoked by a host coding agent (Cursor / Codex / Claude
Code) running ``.cursor/skills/overmind-optimize-agent/SKILL.md``. The skill owns
loop control, parallel candidate fan-out, and early stopping; this CLI
owns the heavy lifting (config persistence, baseline eval, diagnosis,
candidate eval, acceptance gates, report rendering).

All subcommands print a single JSON envelope to stdout::

    {"status": "ok"|"error"|"not_implemented", ...}

so the skill can ``json.loads`` the output and branch deterministically.

Backend lifecycle parity with ``overmind optimize`` (Path A):

* Per-iteration + per-candidate state is already pushed by OTLP span
  attributes from each subcommand (``OPTIMIZE_ITERATION``,
  ``OPTIMIZE_CANDIDATE_*``, ``OPTIMIZE_ITERATION_DECISION``,
  ``OPTIMIZE_STALL_COUNT``, etc) so the UI updates live exactly the
  same way it does for Path A.
* The two **terminal** writes Path A handles via ``ApiReporter``
  (``on_complete`` / ``on_failed`` REST PATCHes) are now mirrored here:
  - ``report`` calls ``reporter.on_complete(...)`` after rendering so
    the Job row flips to ``completed`` and the rendered ``report.md``
    + best agent code are persisted server-side.
  - The CLI dispatcher's outer ``except`` catches ``BaseException``,
    rehydrates the reporter from the persisted ``skill_state.json``,
    stamps ``OPTIMIZE_RUN_STATUS=failed`` + ``ERROR_MESSAGE``, calls
    ``reporter.on_failed(reason)``, and force-flushes traces with a
    long timeout so the failure lands in the UI even when the host
    process crashes hard.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("overmind.commands.optimize_step")


# ----------------------------------------------------------------------
# Backend-lifecycle reporter (parity with overmind optimize Path A)
# ----------------------------------------------------------------------
def _build_reporter_from_state_path(state_path: str | None):
    """Rehydrate an :class:`ApiReporter` from a persisted ``skill_state.json``.

    Returns ``None`` when the state file is missing / unreadable, when
    the file does not carry both an ``agent_id`` and a ``job_id``, or
    when REST is unavailable (no ``OVERMIND_API_KEY``). Callers must
    no-op in those cases — the OTLP path will still update the UI
    (subject to span flushing) but the belt-and-braces terminal PATCH
    is skipped.
    """
    if not isinstance(state_path, str) or not state_path:
        return None
    try:
        data = json.loads(Path(state_path).read_text())
    except Exception:
        logger.debug(
            "optimize-step: could not read state for reporter rehydrate",
            exc_info=True,
        )
        return None
    job_id = data.get("job_id") or ""
    cfg = data.get("config") or {}
    agent_id = cfg.get("agent_id") or data.get("agent_id") or ""
    if not job_id or not agent_id:
        logger.debug("optimize-step: state missing job_id/agent_id; no terminal PATCH")
        return None
    try:
        from overmind.client import ApiReporter

        return ApiReporter.attach_to_job(agent_id=str(agent_id), job_id=str(job_id))
    except Exception:
        logger.debug(
            "optimize-step: ApiReporter.attach_to_job raised; no terminal PATCH will be sent",
            exc_info=True,
        )
        return None


def _finalize_failed_step(state_path: str | None, reason: str) -> None:
    """Mirror ``overmind optimize``'s ``_finalize_failed_job`` for Path B.

    Stamps ``OPTIMIZE_RUN_STATUS=failed`` + ``ERROR_MESSAGE`` on the
    currently-active span, then fires a terminal REST PATCH against
    the Job row (via :class:`ApiReporter`) marking it ``failed`` with
    the rendered reason. Flushes OTel with a generous timeout so the
    failure tail lands in the UI before the process exits.

    All failure modes are swallowed — the dispatcher is already on the
    error path and must not raise from the finalizer.
    """
    try:
        from overmind import attrs, set_tag

        set_tag(attrs.OPTIMIZE_RUN_STATUS, "failed")
        if reason:
            set_tag(attrs.ERROR_MESSAGE, str(reason)[:1000])
    except Exception:
        logger.debug(
            "optimize-step: failed to stamp terminal failed status",
            exc_info=True,
        )
    reporter = _build_reporter_from_state_path(state_path)
    if reporter is not None:
        try:
            reporter.on_failed(reason or "")
        except Exception:
            logger.debug(
                "optimize-step: reporter.on_failed failed; continuing",
                exc_info=True,
            )
    try:
        from overmind.tracing import force_flush_traces

        force_flush_traces(timeout_millis=10_000)
    except Exception:
        logger.debug(
            "optimize-step: force_flush_traces raised; continuing teardown",
            exc_info=True,
        )


# ----------------------------------------------------------------------
# Pre-init bootstrap
# ----------------------------------------------------------------------

_OVERMIND_LOCAL_API_KEY_PLACEHOLDER = "skill-local-no-export"


def resolve_agent_name_from_state(state_path: str | None) -> str | None:
    """Read the agent name persisted by ``optimize-step init`` (best effort).

    ``optimize-step`` subcommands receive the agent name only on ``init``;
    every other invocation reads it from the skill state file.  Returns
    ``None`` when *state_path* is missing or unreadable so callers can
    fall through to other lookup paths.
    """
    if not isinstance(state_path, str) or not state_path:
        return None
    try:
        data = json.loads(Path(state_path).read_text())
    except Exception:
        return None
    agent = data.get("agent_name") or (data.get("config") or {}).get("agent_name")
    if isinstance(agent, str) and agent:
        return agent
    return None


def bootstrap_optimize_step(args: argparse.Namespace) -> None:
    """Apply the pre-``overmind.init()`` environment patches the skill loop needs.

    Three side-effects, all idempotent and safe to call from the CLI
    dispatcher before ``overmind.init()`` runs:

    1. **API-key placeholder.** ``optimize-step`` is invoked non-
       interactively, so an interactive prompt for ``OVERMIND_API_KEY``
       would deadlock the loop.  When the env var is unset we install a
       placeholder so the SDK can mint trace IDs / set tags locally.

    2. **Traceparent rehydration.** Every subprocess that isn't ``init``
       reads the W3C ``traceparent`` persisted by ``init`` from the skill
       state file and exports it into ``TRACEPARENT`` so the SDK's
       :func:`_attach_remote_parent_if_present` can stitch this run into
       the workflow trace.  Pre-set ``TRACEPARENT`` / ``OTEL_TRACEPARENT``
       always win.

    3. **Agent name binding.** Resolved from ``args.agent`` (``init``) or
       the persisted skill state file (every other subcommand) and stored
       at ``args.resolved_agent_name`` so downstream logging / tracing has
       a stable handle regardless of which subcommand was invoked.
    """
    if not os.getenv("OVERMIND_API_KEY"):
        os.environ["OVERMIND_API_KEY"] = _OVERMIND_LOCAL_API_KEY_PLACEHOLDER

    step = getattr(args, "step", None)
    if step != "init" and not os.getenv("TRACEPARENT") and not os.getenv("OTEL_TRACEPARENT"):
        state_path = getattr(args, "state", None)
        if isinstance(state_path, str) and state_path:
            try:
                state = json.loads(Path(state_path).read_text())
                tp = state.get("traceparent")
                if isinstance(tp, str) and tp:
                    os.environ["TRACEPARENT"] = tp
            except Exception:
                # Bad / missing state file falls through to legacy
                # ``overmind.job.id``-based coalescing in OTLP ingest.
                logger.debug(
                    "optimize-step: could not read traceparent from %s",
                    state_path,
                    exc_info=True,
                )

    agent = getattr(args, "agent", None)
    if not (isinstance(agent, str) and agent):
        agent = resolve_agent_name_from_state(getattr(args, "state", None))
    args.resolved_agent_name = agent if isinstance(agent, str) and agent else None


# ----------------------------------------------------------------------
# stdin / arg helpers
# ----------------------------------------------------------------------
def _load_json_arg(value: str | None, *, allow_stdin: bool = True) -> dict[str, Any]:
    """Resolve a JSON-blob argument from either ``-`` (stdin), a path, or inline JSON."""
    if value is None:
        return {}
    if value == "-" and allow_stdin:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    p = Path(value)
    if p.is_file():
        return json.loads(p.read_text())
    return json.loads(value)


def _emit(envelope: dict[str, Any]) -> int:
    print(json.dumps(envelope, indent=2, default=str))
    if envelope.get("status") == "error":
        return 1
    return 0


# ----------------------------------------------------------------------
# Subcommand implementations
# ----------------------------------------------------------------------
def _cmd_init(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.init_step import run_init

    settings = _load_json_arg(args.settings)
    result = run_init(
        agent_name=args.agent,
        user_settings=settings,
        overwrite=bool(args.overwrite),
    )
    return _emit(result)


def _load_state(state_arg: str):
    from overmind.optimize.steps.state import SkillRunState

    return SkillRunState.load(state_arg)


def _cmd_baseline(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.baseline_step import run_baseline

    state = _load_state(args.state)
    return _emit(run_baseline(state))


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.diagnose_step import run_diagnose

    state = _load_state(args.state)
    return _emit(run_diagnose(state, iteration=args.iteration))


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.evaluate_step import run_evaluate

    state = _load_state(args.state)
    return _emit(
        run_evaluate(
            state,
            candidate_dir=args.candidate_dir,
            candidate_id=args.candidate_id,
            iteration=args.iteration,
        )
    )


def _cmd_accept(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.accept_step import run_accept

    state = _load_state(args.state)
    return _emit(
        run_accept(
            state,
            iteration=args.iteration,
            candidate_results_path=args.candidate_results,
        )
    )


def _cmd_report(args: argparse.Namespace) -> int:
    from overmind.optimize.steps.report_step import run_report

    state = _load_state(args.state)
    return _emit(run_report(state))


def _cmd_status(args: argparse.Namespace) -> int:
    """Print the current SkillRunState as JSON. Useful for the skill loop."""
    from dataclasses import asdict

    state = _load_state(args.state)
    return _emit({
        "status": "ok",
        "state": asdict(state),
        "early_stop": (
            state.config.get("early_stopping_patience", 0) > 0
            and state.stall_count >= state.config.get("early_stopping_patience", 0)
        ),
    })


# ----------------------------------------------------------------------
# Argparse wiring
# ----------------------------------------------------------------------
def build_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Attach the ``optimize-step`` subcommand tree to a parent argparse parser."""
    p = subparsers.add_parser(
        "optimize-step",
        help="Skill-driven primitives for `overmind optimize` (advanced; usually invoked by a SKILL.md, not by hand)",
        description=(
            "JSON-in/JSON-out building blocks for the host-coding-agent-driven "
            "optimization loop (see .cursor/skills/overmind-optimize-agent/SKILL.md). "
            "Each subcommand emits a single JSON envelope on stdout."
        ),
    )
    sub = p.add_subparsers(dest="step", required=True, metavar="STEP")

    # init
    p_init = sub.add_parser(
        "init",
        help="Build the cross-step state from a settings JSON dict (skill collects via AskQuestion).",
    )
    p_init.add_argument("agent", help="Registered agent name")
    p_init.add_argument(
        "--settings",
        default="-",
        help="Path to settings JSON, inline JSON string, or '-' for stdin (default).",
    )
    p_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing skill_state.json without prompting.",
    )
    p_init.set_defaults(func=_cmd_init)

    # baseline
    p_base = sub.add_parser("baseline", help="Run baseline eval on the training set.")
    p_base.add_argument("--state", required=True, help="Path to skill_state.json")
    p_base.set_defaults(func=_cmd_baseline)

    # diagnose
    p_diag = sub.add_parser(
        "diagnose",
        help="Diagnose failures and emit N candidate change plans (skill spawns sub-coding-agents).",
    )
    p_diag.add_argument("--state", required=True)
    p_diag.add_argument("--iteration", type=int, required=True)
    p_diag.set_defaults(func=_cmd_diagnose)

    # evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate one candidate worktree against the training set; writes score.json.",
    )
    p_eval.add_argument("--state", required=True)
    p_eval.add_argument("--iteration", type=int, required=True)
    p_eval.add_argument("--candidate-id", required=True, help="e.g. c0, c1, c2")
    p_eval.add_argument(
        "--candidate-dir",
        required=True,
        help="Path to the worktree the host coding agent edited.",
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    # accept
    p_acc = sub.add_parser(
        "accept",
        help="Apply acceptance gates (regression, holdout, complexity); promote winner; bump stall_count.",
    )
    p_acc.add_argument("--state", required=True)
    p_acc.add_argument("--iteration", type=int, required=True)
    p_acc.add_argument(
        "--candidate-results",
        required=True,
        help="Path to JSON listing per-candidate score.json paths.",
    )
    p_acc.set_defaults(func=_cmd_accept)

    # report
    p_rep = sub.add_parser("report", help="Render report.md / results.tsv from final state.")
    p_rep.add_argument("--state", required=True)
    p_rep.set_defaults(func=_cmd_report)

    # status
    p_stat = sub.add_parser(
        "status",
        help="Print the current SkillRunState as JSON (used by the skill loop to read stall_count, best_score, etc).",
    )
    p_stat.add_argument("--state", required=True)
    p_stat.set_defaults(func=_cmd_status)


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by ``overmind/cli.py``'s top-level dispatcher.

    Catches ``BaseException`` so that ``KeyboardInterrupt`` and other
    abnormal exits still get a terminal Job PATCH (Job stays
    ``running`` forever otherwise). ``KeyboardInterrupt`` is re-raised
    so the user still sees the interrupt; the finalizer runs before
    the re-raise so the UI reflects the failure.
    """
    func = getattr(args, "func", None)
    if func is None:
        return _emit({
            "status": "error",
            "error": "no_step",
            "message": "No optimize-step subcommand specified.",
        })
    step = getattr(args, "step", "?")
    state_path = getattr(args, "state", None)
    try:
        return func(args)
    except KeyboardInterrupt:
        logger.warning(f"optimize-step {step}: interrupted by user (KeyboardInterrupt)")
        _finalize_failed_step(state_path, reason="Interrupted by user (KeyboardInterrupt)")
        raise
    except BaseException as exc:
        logger.exception(f"optimize-step {step} failed")
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        _finalize_failed_step(state_path, reason=reason)
        return _emit({
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        })
