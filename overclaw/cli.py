"""
OverClaw — CLI entry point

Commands:
    overclaw init                                      Configure API keys and model defaults
    overclaw agent register <name> <module:function>   Register an agent
    overclaw agent list                                List all registered agents
    overclaw agent remove <name>                       Remove a registered agent
    overclaw agent update <name> <module:function>     Update a registered agent's entrypoint
    overclaw agent show <name>                         Show agent registration and pipeline status
    overclaw setup <name> [--fast]                     Analyze agent and define eval criteria
    overclaw optimize <name> [--fast]                  Run the optimization loop
    overclaw sync [name]                               Sync local setup artifacts to Overmind
    overclaw sync-optimize [name]                      Sync local optimize artifacts to Overmind
"""

from __future__ import annotations

import argparse
import sys

from overclaw.core.constants import OVERCLAW_DIR_NAME, overclaw_rel

_FMT = argparse.RawDescriptionHelpFormatter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overclaw",
        formatter_class=_FMT,
        description="OverClaw — autonomous agent optimization through structured experimentation.",
        epilog=(
            "Typical workflow:\n"
            "  1. overclaw init                                  # set API keys + models\n"
            "  2. overclaw agent register <name> <module:fn>     # register your agent\n"
            "  3. overclaw setup <name>                          # build eval criteria\n"
            "  4. overclaw optimize <name>                       # run the optimizer\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── init ────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "init",
        formatter_class=_FMT,
        help=f"Configure API keys and model defaults in {overclaw_rel('.env')}",
        description="Configure API keys and default models for OverClaw.",
        epilog=(
            f"Writes or updates {overclaw_rel('.env')} under the project root with:\n"
            "  - OPENAI_API_KEY / ANTHROPIC_API_KEY\n"
            "  - ANALYZER_MODEL        (used by setup and optimize)\n"
            "  - SYNTHETIC_DATAGEN_MODEL  (used by setup for test-data generation)\n"
            "\n"
            "Run once per project before using setup or optimize.\n"
            "Safe to re-run — existing values are shown and can be kept.\n"
            "\n"
            "Example:\n"
            "  overclaw init\n"
        ),
    )

    # ── agent ────────────────────────────────────────────────────────────────
    agent_p = subparsers.add_parser(
        "agent",
        formatter_class=_FMT,
        help="Manage registered agents (register / list / remove / update / show)",
        description=(
            "Manage the OverClaw registry (register, list, remove, update, show).\n"
            "\n"
            "Each entry maps a short agent name to a Python module:function\n"
            "entrypoint. Registering an agent lets you run setup and optimize\n"
            "by name instead of by file path."
        ),
        epilog=(
            "Examples:\n"
            "  overclaw agent register lead-qualification agents.agent1.sample_agent:run\n"
            "  overclaw agent list\n"
            "  overclaw agent show lead-qualification\n"
            "  overclaw agent update lead-qualification agents.agent2.new_agent:run\n"
            "  overclaw agent remove lead-qualification\n"
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
            f"(project root: directory with `{OVERCLAW_DIR_NAME}/`; run `overclaw init` first).\n"
            "\n"
            "OverClaw validates that the file exists and the function is\n"
            "defined before saving the entry."
        ),
        epilog=(
            "Examples:\n"
            "  overclaw agent register lead-qualification agents.agent1.sample_agent:run\n"
            "  overclaw agent register support-bot agents.support.bot:handle\n"
            "\n"
            "After registering, run:\n"
            "  overclaw setup <name>\n"
        ),
    )
    reg_p.add_argument(
        "name", metavar="NAME", help="Short agent name (e.g. lead-qualification)"
    )
    reg_p.add_argument(
        "entrypoint",
        metavar="MODULE:FUNCTION",
        help="Python entrypoint (e.g. agents.agent1.sample_agent:run)",
    )

    agent_subs.add_parser(
        "list",
        formatter_class=_FMT,
        help="List all registered agents",
        description="List all agents registered in the OverClaw registry.",
        epilog=(
            "Columns:\n"
            "  NAME        — the agent name used with setup and optimize\n"
            "  ENTRYPOINT  — the registered module:function\n"
            "  FILE        — ✓ if the agent file exists on disk, ✗ if not\n"
            "\n"
            "Example:\n"
            "  overclaw agent list\n"
        ),
    )

    rem_p = agent_subs.add_parser(
        "remove",
        formatter_class=_FMT,
        help="Remove a registered agent",
        description=(
            "Remove an agent from the OverClaw registry.\n"
            "\n"
            "This only removes the registry entry — it does not delete the\n"
            "agent source file or per-agent setup and experiment data on disk."
        ),
        epilog=("Example:\n  overclaw agent remove lead-qualification\n"),
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
        epilog=(
            "Example:\n"
            "  overclaw agent update lead-qualification agents.agent2.new_agent:run\n"
        ),
    )
    upd_p.add_argument("name", metavar="NAME", help="Agent name to update")
    upd_p.add_argument(
        "entrypoint",
        metavar="MODULE:FUNCTION",
        help="New Python entrypoint (e.g. agents.agent2.new_agent:run)",
    )

    show_p = agent_subs.add_parser(
        "show",
        formatter_class=_FMT,
        help="Show agent registration and pipeline status",
        description=(
            "Show the registration details and current pipeline status for\n"
            "a single agent."
        ),
        epilog=(
            "Status fields:\n"
            "  File         — whether the registered file exists on disk\n"
            "  Setup spec   — whether overclaw setup has been run\n"
            f"                 ({overclaw_rel('agents', '<name>', 'setup_spec', 'eval_spec.json')})\n"
            "  Experiments  — whether overclaw optimize has produced output\n"
            f"                 (files under {overclaw_rel('agents', '<name>', 'experiments')}/)\n"
            "\n"
            "Example:\n"
            "  overclaw agent show lead-qualification\n"
        ),
    )
    show_p.add_argument("name", metavar="NAME", help="Agent name to inspect")

    # ── setup ────────────────────────────────────────────────────────────────
    setup_p = subparsers.add_parser(
        "setup",
        formatter_class=_FMT,
        help="Analyze your agent, define policies, and build evaluation criteria",
        description=(
            "Analyze your agent and build an evaluation spec through a\n"
            "4-phase interactive flow.\n"
            "\n"
            "  Phase 1 · Agent Analysis   — examines code structure, tools, schemas\n"
            "  Phase 2 · Agent Policy     — defines domain rules and constraints\n"
            "  Phase 3 · Dataset          — generates or validates test data\n"
            "  Phase 4 · Eval Criteria    — proposes and refines scoring rules\n"
            "\n"
            f"Output is saved to {overclaw_rel('agents', '<name>', 'setup_spec', 'eval_spec.json')}.\n"
            "Run this before overclaw optimize."
        ),
        epilog=(
            "Flags:\n"
            "  --fast      Skips all interactive prompts. Requires ANALYZER_MODEL\n"
            f"              and SYNTHETIC_DATAGEN_MODEL set in {overclaw_rel('.env')}.\n"
            "  --policy    Path to an existing policy document (.md or .txt).\n"
            "              OverClaw will analyze it against your agent code and\n"
            "              suggest improvements before using it.\n"
            "\n"
            "Examples:\n"
            "  overclaw setup lead-qualification\n"
            "  overclaw setup lead-qualification --fast\n"
            "  overclaw setup lead-qualification --policy docs/domain-rules.md\n"
        ),
    )
    setup_p.add_argument(
        "agent",
        metavar="AGENT_NAME",
        help="registered agent name (see: overclaw agent list)",
    )
    setup_p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "skip all prompts; requires ANALYZER_MODEL and SYNTHETIC_DATAGEN_MODEL in "
            f"{overclaw_rel('.env')}"
        ),
    )
    setup_p.add_argument(
        "--policy",
        default=None,
        metavar="POLICY_PATH",
        help="path to an existing policy/domain document (.md, .txt)",
    )

    # ── optimize ─────────────────────────────────────────────────────────────
    opt_p = subparsers.add_parser(
        "optimize",
        formatter_class=_FMT,
        help="Run the optimization loop against your agent",
        description=(
            "Run the iterative optimization loop against your agent.\n"
            "\n"
            "Requires overclaw setup to have been run first.\n"
            "\n"
            "Each iteration:\n"
            "  1. Runs the agent against the evaluation dataset\n"
            "  2. Diagnoses failures using the analyzer model\n"
            "  3. Generates and tests candidate improvements\n"
            "  4. Accepts changes that improve the score\n"
            "\n"
            f"Results and traces are saved to {overclaw_rel('agents', '<name>', 'experiments')}/.\n"
            "The best-performing version is written to experiments/best_agent.py there."
        ),
        epilog=(
            "Flags:\n"
            "  --fast    Skips all interactive prompts. Requires ANALYZER_MODEL\n"
            f"            in {overclaw_rel('.env')}. Uses defaults for all optimizer settings\n"
            "            (no LLM judge, no backtesting, 5 iterations).\n"
            "\n"
            "Examples:\n"
            "  overclaw optimize lead-qualification\n"
            "  overclaw optimize lead-qualification --fast\n"
        ),
    )
    opt_p.add_argument(
        "agent",
        metavar="AGENT_NAME",
        help="registered agent name (see: overclaw agent list)",
    )
    opt_p.add_argument(
        "--fast",
        action="store_true",
        help=f"skip all prompts; requires ANALYZER_MODEL in {overclaw_rel('.env')}",
    )

    # ── sync ─────────────────────────────────────────────────────────────────
    sync_p = subparsers.add_parser(
        "sync",
        formatter_class=_FMT,
        help="Sync local setup artifacts to Overmind",
        description=(
            "Upload local setup artifacts (eval spec, dataset, policy) from "
            f"{overclaw_rel('agents', '<name>', 'setup_spec')} to Overmind.\n"
            "\n"
            "Useful when artifacts were generated before Overmind API keys were configured."
        ),
        epilog=(
            "Examples:\n"
            "  overclaw sync                  # sync all registered agents\n"
            "  overclaw sync lead-qualification  # sync one agent\n"
        ),
    )
    sync_p.add_argument(
        "agent",
        nargs="?",
        metavar="AGENT_NAME",
        help="optional registered agent name (defaults to all registered agents)",
    )

    # ── sync-optimize ────────────────────────────────────────────────────────
    sync_opt_p = subparsers.add_parser(
        "sync-optimize",
        formatter_class=_FMT,
        help="Sync local optimize artifacts to Overmind",
        description=(
            "Upload local optimize artifacts from "
            f"{overclaw_rel('agents', '<name>', 'experiments')} to Overmind.\n"
            "\n"
            "Useful when optimization ran before Overmind API keys were configured."
        ),
        epilog=(
            "Examples:\n"
            "  overclaw sync-optimize                  # sync all registered agents\n"
            "  overclaw sync-optimize lead-qualification  # sync one agent\n"
        ),
    )
    sync_opt_p.add_argument(
        "agent",
        nargs="?",
        metavar="AGENT_NAME",
        help="optional registered agent name (defaults to all registered agents)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "init":
        from overclaw.core.registry import require_overclaw_initialized

        require_overclaw_initialized()

    try:
        if args.command == "init":
            from overclaw.commands.init_cmd import main as _init

            _init()

        elif args.command == "agent":
            from overclaw.commands.agent_cmd import (
                cmd_list,
                cmd_register,
                cmd_remove,
                cmd_show,
                cmd_update,
            )

            if args.agent_command == "register":
                cmd_register(args.name, args.entrypoint)
            elif args.agent_command == "list":
                cmd_list()
            elif args.agent_command == "remove":
                cmd_remove(args.name)
            elif args.agent_command == "update":
                cmd_update(args.name, args.entrypoint)
            elif args.agent_command == "show":
                cmd_show(args.name)

        elif args.command == "setup":
            from overclaw.commands.setup_cmd import main as _setup

            _setup(agent_name=args.agent, fast=args.fast, policy=args.policy)

        elif args.command == "optimize":
            from overclaw.commands.optimize_cmd import main as _optimize

            _optimize(agent_name=args.agent, fast=args.fast)

        elif args.command == "sync":
            from overclaw.commands.sync_cmd import main as _sync

            _sync(agent_name=args.agent)

        elif args.command == "sync-optimize":
            from overclaw.commands.sync_optimize_cmd import main as _sync_optimize

            _sync_optimize(agent_name=args.agent)

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
