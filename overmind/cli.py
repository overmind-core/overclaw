"""
Overmind — CLI entry point

The optimization "brain" (analysis, eval-criteria, diagnosis, candidate
generation) lives on the Overmind server. This CLI registers agents and runs
the thin client daemon that executes server-orchestrated workflows locally.

Commands:
    overmind init                                      Configure API keys and model defaults
    overmind agent register <name> <module:function>   Register an agent
    overmind agent list                                List all registered agents
    overmind agent remove <name>                       Remove a registered agent
    overmind agent update <name> <module:function>     Update a registered agent's entrypoint
    overmind agent show <name>                         Show agent registration and pipeline status
    overmind agent validate <name> --data PATH         Run the agent against test data
    overmind daemon [--agent NAME]                     Run CLI daemon (server workflow executor)
    overmind workflow <name> [--workflow NAME]         Start server-orchestrated workflow
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from opentelemetry import context
from opentelemetry import trace as _otel_trace
from opentelemetry.trace import Status, StatusCode

import overmind
from overmind import attrs
from overmind.commands.agent_cmd import (
    cmd_list,
    cmd_pull,
    cmd_register,
    cmd_remove,
    cmd_show,
    cmd_update,
    cmd_validate,
)
from overmind.commands.daemon_cmd import build_subparser as _build_daemon_parser
from overmind.commands.daemon_cmd import main as _daemon
from overmind.commands.init_cmd import main as _init
from overmind.commands.workflow_cmd import build_subparser as _build_workflow_parser
from overmind.commands.workflow_cmd import main as _workflow
from overmind.core.constants import OVERMIND_DIR_NAME, overmind_rel
from overmind.core.logging import setup_logging
from overmind.core.paths import load_overmind_dotenv
from overmind.core.registry import require_overmind_initialized
from overmind.tracing import force_flush_traces

_FMT = argparse.RawDescriptionHelpFormatter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overmind",
        formatter_class=_FMT,
        description="Overmind — autonomous agent optimization through structured experimentation.",
        epilog=(
            "Typical workflow:\n"
            "  1. overmind init                                  # set API keys + models\n"
            "  2. overmind agent register <name> <module:fn>     # register your agent\n"
            "  3. overmind daemon                                # connect this machine to the server\n"
            "  4. overmind workflow <name>                       # run server-orchestrated optimization\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── init ────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "init",
        formatter_class=_FMT,
        help=f"Configure API keys and model defaults in {overmind_rel('.env')}",
        description="Configure API keys and default models for Overmind.",
        epilog=(
            f"Writes or updates {overmind_rel('.env')} under the project root with:\n"
            "  - OPENAI_API_KEY / ANTHROPIC_API_KEY\n"
            "  - ANALYZER_MODEL        (used by agent validation + env provisioning)\n"
            "\n"
            "Run once per project before registering agents.\n"
            "Safe to re-run — existing values are shown and can be kept.\n"
            "\n"
            "Example:\n"
            "  overmind init\n"
        ),
    )

    # ── agent ────────────────────────────────────────────────────────────────
    agent_p = subparsers.add_parser(
        "agent",
        formatter_class=_FMT,
        help="Manage registered agents (register / list / remove / update / show)",
        description=(
            "Manage the Overmind registry (register, list, remove, update, show).\n"
            "\n"
            "Each entry maps a short agent name to a Python module:function\n"
            "entrypoint. Registering an agent lets you run setup and optimize\n"
            "by name instead of by file path."
        ),
        epilog=(
            "Examples:\n"
            "  overmind agent register lead-qualification agents.agent1.sample_agent:run\n"
            "  overmind agent list\n"
            "  overmind agent show lead-qualification\n"
            "  overmind agent update lead-qualification agents.agent2.new_agent:run\n"
            "  overmind agent remove lead-qualification\n"
        ),
    )
    agent_subs = agent_p.add_subparsers(dest="agent_command", metavar="SUBCOMMAND")
    agent_subs.required = True

    reg_p = agent_subs.add_parser(
        "register",
        formatter_class=_FMT,
        help="Register a new agent",
        description=(
            "Register an agent by giving it a name and a Python entrypoint.\n"
            "\n"
            "The entrypoint is a dotted module path and a function name\n"
            "separated by a colon:  module.path:function_name\n"
            "\n"
            "The module path is resolved relative to the project root\n"
            f"(project root: directory with `{OVERMIND_DIR_NAME}/`; run `overmind init` first).\n"
            "\n"
            "Overmind validates that the file exists and the function is\n"
            "defined before saving the entry."
        ),
        epilog=(
            "Examples:\n"
            "  overmind agent register lead-qualification agents.agent1.sample_agent:run\n"
            "  overmind agent register support-bot agents.support.bot:handle\n"
            "\n"
            "After registering, run:\n"
            "  overmind setup <name>\n"
        ),
    )
    reg_p.add_argument("name", metavar="NAME", help="Short agent name (e.g. lead-qualification)")
    reg_p.add_argument(
        "entrypoint",
        metavar="MODULE:FUNCTION",
        help="Python entrypoint (e.g. agents.agent1.sample_agent:run)",
    )
    reg_p.add_argument(
        "--non-interactive",
        "-y",
        action="store_true",
        help=(
            "skip every interactive prompt (provider menu, confirmations, env "
            f"variable values). Uses an existing {overmind_rel('agents', '<name>', '.env')} "
            "as-is; create it manually first when running in sandboxed shells or CI."
        ),
    )

    agent_subs.add_parser(
        "list",
        formatter_class=_FMT,
        help="List all registered agents",
        description="List all agents registered in the Overmind registry.",
        epilog=(
            "Columns:\n"
            "  NAME        — the agent name used with setup and optimize\n"
            "  ENTRYPOINT  — the registered module:function\n"
            "  FILE        — ✓ if the agent file exists on disk, ✗ if not\n"
            "\n"
            "Example:\n"
            "  overmind agent list\n"
        ),
    )

    agent_subs.add_parser(
        "pull",
        formatter_class=_FMT,
        help="Write .overmind/agents.toml from the agents extracted on the server",
        description=(
            "Fetch every agent the Overmind backend has extracted for this\n"
            "project and write them all into the local registry\n"
            f"({overmind_rel('agents.toml')}) in one pass.\n"
            "\n"
            "Use this after Overmind analyzes a connected repository: instead of\n"
            "registering each agent by hand, pull materialises the whole registry\n"
            "(name, entrypoint, and server id) so `overmind daemon` can run them.\n"
            "Local-only entries are preserved."
        ),
        epilog=(
            "Requires OVERMIND_API_KEY (and OVERMIND_API_URL for self-hosted).\n"
            "\n"
            "Example:\n"
            "  overmind agent pull\n"
        ),
    )

    rem_p = agent_subs.add_parser(
        "remove",
        formatter_class=_FMT,
        help="Remove a registered agent",
        description=(
            "Remove an agent from the Overmind registry.\n"
            "\n"
            "This only removes the registry entry — it does not delete the\n"
            "agent source file or per-agent setup and experiment data on disk."
        ),
        epilog=("Example:\n  overmind agent remove lead-qualification\n"),
    )
    rem_p.add_argument("name", metavar="NAME", help="Agent name to remove")

    upd_p = agent_subs.add_parser(
        "update",
        formatter_class=_FMT,
        help="Update a registered agent's entrypoint",
        description=(
            "Update the module:function entrypoint for an existing agent.\n"
            "\n"
            "Use this when you move or rename the agent file without wanting\n"
            "to remove and re-register it from scratch.\n"
            "\n"
            "The new entrypoint is validated (file exists, function defined)\n"
            "before the registry is updated."
        ),
        epilog=("Example:\n  overmind agent update lead-qualification agents.agent2.new_agent:run\n"),
    )
    upd_p.add_argument("name", metavar="NAME", help="Agent name to update")
    upd_p.add_argument(
        "entrypoint",
        metavar="MODULE:FUNCTION",
        help="New Python entrypoint (e.g. agents.agent2.new_agent:run)",
    )
    upd_p.add_argument(
        "--non-interactive",
        "-y",
        action="store_true",
        help=(
            "skip every interactive prompt (provider menu, confirmations, env "
            f"variable values). Uses an existing {overmind_rel('agents', '<name>', '.env')} "
            "as-is; create it manually first when running in sandboxed shells or CI."
        ),
    )

    show_p = agent_subs.add_parser(
        "show",
        formatter_class=_FMT,
        help="Show agent registration and pipeline status",
        description=("Show the registration details and current pipeline status for\na single agent."),
        epilog=(
            "Status fields:\n"
            "  File         — whether the registered file exists on disk\n"
            "  Setup spec   — whether overmind setup has been run\n"
            f"                 ({overmind_rel('agents', '<name>', 'setup_spec', 'eval_spec.json')})\n"
            "  Experiments  — whether overmind optimize has produced output\n"
            f"                 (files under {overmind_rel('agents', '<name>', 'experiments')}/)\n"
            "\n"
            "Example:\n"
            "  overmind agent show lead-qualification\n"
        ),
    )
    show_p.add_argument("name", metavar="NAME", help="Agent name to inspect")

    val_p = agent_subs.add_parser(
        "validate",
        formatter_class=_FMT,
        help="Validate an agent's entrypoint by running it against test data",
        description=(
            "Run the agent against one or more JSON test cases to verify\n"
            "that the registered entrypoint works end-to-end.\n"
            "\n"
            'Each case should be a JSON object with an "input" key (or be\n'
            "the input dict itself).  The agent is invoked via the same\n"
            "subprocess runner used by setup and optimize."
        ),
        epilog=(
            "Examples:\n"
            "  overmind agent validate gsec --data tests/case.json\n"
            "  overmind agent validate gsec --data tests/cases/\n"
        ),
    )
    val_p.add_argument("name", metavar="NAME", help="Agent name to validate")
    val_p.add_argument(
        "--data",
        metavar="PATH",
        required=True,
        help="Path to a JSON file or directory of JSON files with test cases",
    )

    # ── daemon / workflow (client-server FSM) ───────────────────────────────
    _build_daemon_parser(subparsers)
    _build_workflow_parser(subparsers)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Honour ``--non-interactive`` / ``-y`` on subcommands that opt in
    # (currently ``agent register`` and ``agent update``).  Setting the
    # env var here means every helper that imports
    # ``overmind.utils.display.is_non_interactive`` automatically picks
    # the non-interactive code paths without having to thread a flag
    # through every call site.
    if getattr(args, "non_interactive", False):
        os.environ["OVERMIND_NONINTERACTIVE"] = "1"

    if args.command != "init":
        require_overmind_initialized()
        load_overmind_dotenv()

    load_dotenv(".env")
    load_dotenv(".overmind/.env", override=True)

    # Wire up logging as early as possible so every module that gets
    # imported next (commands, optimizer, coding agent, …) can emit debug
    # traces from its module-level loggers.  ``overmind init`` configures
    # its logger after it creates ``.overmind/`` so the log lands there.
    if args.command != "init":
        log_path = setup_logging()

        logging.getLogger("overmind.cli").info(
            "CLI invoked command=%s argv=%s log_file=%s", args.command, sys.argv[1:], log_path
        )

    try:
        if args.command != "init":
            overmind.init(service_name="overmind.cli", providers=None)

        if args.command == "init":
            _init()

        elif args.command == "agent":
            if args.agent_command == "register":
                context.attach(context.set_value(attrs.AGENT_NAME, args.name))
                cmd_register(args.name, args.entrypoint)
            elif args.agent_command == "list":
                cmd_list()
            elif args.agent_command == "pull":
                cmd_pull()
            elif args.agent_command == "remove":
                context.attach(context.set_value(attrs.AGENT_NAME, args.name))
                cmd_remove(args.name)
            elif args.agent_command == "update":
                context.attach(context.set_value(attrs.AGENT_NAME, args.name))
                cmd_update(args.name, args.entrypoint)
            elif args.agent_command == "show":
                context.attach(context.set_value(attrs.AGENT_NAME, args.name))
                cmd_show(args.name)
            elif args.agent_command == "validate":
                context.attach(context.set_value(attrs.AGENT_NAME, args.name))
                cmd_validate(args.name, args.data)

        elif args.command == "daemon":
            _daemon(args)

        elif args.command == "workflow":
            context.attach(context.set_value(attrs.AGENT_NAME, args.agent_name))
            _workflow(args)

    except KeyboardInterrupt:
        span = _otel_trace.get_current_span()
        if span.is_recording():
            span.record_exception(KeyboardInterrupt())
            span.set_status(Status(StatusCode.ERROR, "Interrupted by user (KeyboardInterrupt)"))
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(130) from None
    finally:
        # CLI exits the process right after this; give the BatchSpanProcessor
        # a generous window so the workflow span and any in-flight LLM /
        # tool child spans land on the backend before teardown.
        force_flush_traces(timeout_millis=10_000)


if __name__ == "__main__":
    main()
