"""overmind daemon — long-lived CLI daemon for server workflow orchestration."""

from __future__ import annotations

import argparse
import logging

from overmind.core.logging import setup_logging
from overmind.daemon.main import run_daemon

logger = logging.getLogger(__name__)


def build_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "daemon",
        help="Run the CLI daemon (polling command runner)",
    )
    p.add_argument(
        "--agent",
        dest="agent_name",
        help="Default agent name for commands",
    )
    p.add_argument(
        "--session-id",
        help="Reuse an existing client session ID",
    )


def main(args: argparse.Namespace) -> None:
    setup_logging()
    run_daemon(
        agent_name=getattr(args, "agent_name", None),
        session_id=getattr(args, "session_id", None),
    )
