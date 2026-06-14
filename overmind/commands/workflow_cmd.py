"""overmind workflow — start and monitor server-orchestrated workflows."""

from __future__ import annotations

import argparse
import json
import logging

from overmind.core.logging import setup_logging
from overmind.workflow.runner import run_server_workflow

logger = logging.getLogger(__name__)


def build_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "workflow",
        help="Start a server-orchestrated workflow",
    )
    p.add_argument("agent_name", help="Registered agent name")
    p.add_argument(
        "--workflow",
        default="optimize_loop",
        choices=["optimize_setup", "optimize_loop"],
        help="Workflow to run",
    )
    p.add_argument(
        "--entrypoint-fn",
        default="",
        help="Entrypoint function name (default: from registry)",
    )
    p.add_argument(
        "--no-daemon",
        action="store_true",
        help="Do not spawn daemon subprocess (assume one is already running)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip interactive approval prompts",
    )
    p.add_argument(
        "--smoke-input",
        default="",
        help="JSON smoke test input (optional)",
    )


def main(args: argparse.Namespace) -> None:
    setup_logging()
    config: dict = {}
    if args.entrypoint_fn:
        config["entrypoint_fn"] = args.entrypoint_fn
    if args.smoke_input:
        config["smoke_input"] = json.loads(args.smoke_input)

    run_server_workflow(
        args.agent_name,
        args.workflow,
        config=config,
        fast=args.fast,
        manage_daemon=not args.no_daemon,
    )
