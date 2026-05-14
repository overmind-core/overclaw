"""``overmind optimize-step`` — JSON-in/JSON-out pieces of the optimize loop.

Designed to be invoked by a host coding agent (Cursor / Codex / Claude
Code) running ``.cursor/skills/overmind-optimize-agent/SKILL.md``. The skill owns
loop control, parallel candidate fan-out, and early stopping; this CLI
owns the heavy lifting (config persistence, baseline eval, diagnosis,
candidate eval, acceptance gates, report rendering).

All subcommands print a single JSON envelope to stdout::

    {"status": "ok"|"error"|"not_implemented", ...}

so the skill can ``json.loads`` the output and branch deterministically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("overmind.commands.optimize_step")


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
    """Entry point invoked by ``overmind/cli.py``'s top-level dispatcher."""
    func = getattr(args, "func", None)
    if func is None:
        return _emit({
            "status": "error",
            "error": "no_step",
            "message": "No optimize-step subcommand specified.",
        })
    try:
        return func(args)
    except Exception as exc:
        logger.exception(f"optimize-step {getattr(args, 'step', '?')} failed")
        return _emit({
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        })
