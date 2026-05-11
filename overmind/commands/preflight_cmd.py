"""``overmind preflight`` — JSON-in/JSON-out validation gate before optimize.

Five subcommands, all emitting a single JSON envelope on stdout so the
host coding agent can ``json.loads`` and branch deterministically:

- ``scan <agent>``          — static credential discovery (read-only).
- ``set-secret <agent>``    — persist one credential into the per-agent
                              ``.env`` (value is read from stdin so it
                              never appears in shell history).
- ``run <agent>``           — full convergence loop (instrument, smoke,
                              classify, fix, repeat).  Exits non-zero
                              when status is not green.
- ``status <agent>``        — print the persisted ``preflight.json``.
- ``reset <agent>``         — delete preflight state so optimize is
                              forced to wait for a fresh run.

The ``optimize`` and ``optimize-step init`` commands consult the same
persisted report via :func:`overmind.preflight.is_preflight_green` so
the gate is enforced consistently across every entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from typing import Any

from overmind.preflight import (
    load_report,
    preflight_report_path,
    run_preflight,
    scan_secrets,
    set_secret,
)
from overmind.preflight.state import (
    GREEN_STATUSES,
    preflight_dir,
)

logger = logging.getLogger("overmind.commands.preflight")


def _emit(envelope: dict[str, Any], *, ok_statuses: set[str] | None = None) -> int:
    print(json.dumps(envelope, indent=2, default=str))
    status = envelope.get("status")
    if status == "error":
        return 1
    if ok_statuses is not None and status not in ok_statuses:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_scan(args: argparse.Namespace) -> int:
    payload = scan_secrets(args.agent)
    payload["status"] = "ok"
    return _emit(payload)


def _cmd_set_secret(args: argparse.Namespace) -> int:
    if args.value is not None:
        value = args.value
    else:
        value = sys.stdin.read().strip()
    if not value:
        return _emit({"status": "error", "error": "empty_value", "key": args.key})
    outcome = set_secret(args.agent, args.key, value, validate=not args.no_validate)
    return _emit(outcome)


def _cmd_run(args: argparse.Namespace) -> int:
    secrets_provided: dict[str, str] = {}
    if args.with_secrets_stdin:
        raw = sys.stdin.read().strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return _emit({
                    "status": "error",
                    "error": "invalid_secrets_json",
                    "message": str(exc),
                })
            if isinstance(parsed, dict):
                secrets_provided = {
                    k: str(v) for k, v in parsed.items() if isinstance(k, str) and v is not None and str(v).strip()
                }

    report = run_preflight(
        args.agent,
        max_iters=args.max_iters,
        max_rows=args.max_rows,
        timeout=args.timeout,
        secrets_provided=secrets_provided or None,
    )
    return _emit(report.to_dict(), ok_statuses=set(GREEN_STATUSES))


def _cmd_status(args: argparse.Namespace) -> int:
    report = load_report(args.agent)
    if report is None:
        return _emit({
            "status": "error",
            "error": "no_preflight_report",
            "message": (
                f"No preflight report found at {preflight_report_path(args.agent)}. "
                "Run `overmind preflight run <agent>` (or the /overmind-preflight skill)."
            ),
        })
    payload = report.to_dict()
    payload["report_path"] = str(preflight_report_path(args.agent))
    return _emit(payload)


def _cmd_reset(args: argparse.Namespace) -> int:
    pf = preflight_dir(args.agent)
    if pf.exists():
        shutil.rmtree(pf)
    return _emit({
        "status": "ok",
        "agent": args.agent,
        "removed": str(pf),
    })


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "preflight",
        help="Validate the agent + spec + dataset pipeline before `overmind optimize`",
        description=(
            "Runs the agent against a 2-row dataset slice, classifies any "
            "failure into deterministic kinds, and autonomously fixes every "
            "plumbing issue (eval-spec / dataset / schema / instrumentation). "
            "Writes a structured report at "
            ".overmind/agents/<name>/preflight/preflight.json."
        ),
    )
    sub = p.add_subparsers(dest="step", required=True, metavar="STEP")

    # scan
    p_scan = sub.add_parser(
        "scan",
        help="Static scan for env vars / provider keys the agent needs (read-only).",
    )
    p_scan.add_argument("agent", help="Registered agent name")
    p_scan.set_defaults(func=_cmd_scan)

    # set-secret
    p_set = sub.add_parser(
        "set-secret",
        help="Persist a single credential into the per-agent .env (value via stdin).",
    )
    p_set.add_argument("agent", help="Registered agent name")
    p_set.add_argument("--key", required=True, help="Env var name (e.g. OPENAI_API_KEY)")
    p_set.add_argument(
        "--value",
        default=None,
        help="Inline value (use stdin instead for secrets you don't want in shell history)",
    )
    p_set.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the cheap one-token completion check.",
    )
    p_set.set_defaults(func=_cmd_set_secret)

    # run
    p_run = sub.add_parser(
        "run",
        help="Run the full preflight convergence loop.",
    )
    p_run.add_argument("agent", help="Registered agent name")
    p_run.add_argument("--max-iters", type=int, default=5, help="Max convergence iterations (default 5)")
    p_run.add_argument("--max-rows", type=int, default=2, help="Dataset rows to smoke per iteration (default 2)")
    p_run.add_argument("--timeout", type=int, default=120, help="Per-case subprocess timeout, seconds (default 120)")
    p_run.add_argument(
        "--with-secrets-stdin",
        action="store_true",
        help="Read a JSON object {KEY: VALUE} of credentials from stdin and persist them before running.",
    )
    p_run.set_defaults(func=_cmd_run)

    # status
    p_stat = sub.add_parser("status", help="Print the persisted preflight.json")
    p_stat.add_argument("agent", help="Registered agent name")
    p_stat.set_defaults(func=_cmd_status)

    # reset
    p_reset = sub.add_parser(
        "reset",
        help="Delete preflight state (forces optimize to wait for a fresh run).",
    )
    p_reset.add_argument("agent", help="Registered agent name")
    p_reset.set_defaults(func=_cmd_reset)


def main(args: argparse.Namespace) -> int:
    func = getattr(args, "func", None)
    if func is None:
        return _emit({
            "status": "error",
            "error": "no_step",
            "message": "No preflight subcommand specified.",
        })
    try:
        return func(args)
    except Exception as exc:
        logger.exception("preflight %s failed", getattr(args, "step", "?"))
        return _emit({
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        })
