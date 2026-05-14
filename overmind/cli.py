"""
Overmind — CLI entry point

Commands:
    overmind init                                      Configure API keys and model defaults
    overmind agent register <name> <module:function>   Register an agent
    overmind agent list                                List all registered agents
    overmind agent remove <name>                       Remove a registered agent
    overmind agent update <name> <module:function>     Update a registered agent's entrypoint
    overmind agent show <name>                         Show agent registration and pipeline status
    overmind setup <name> [--data PATH] [--fast]      Analyze agent and define eval criteria
    overmind optimize <name> [--fast]                  Run the optimization loop
    overmind doctor <name>                             Diagnose bundle scope and eval spec (read-only)
    overmind sync [name]                               Sync local setup artifacts to Overmind
    overmind sync-optimize [name]                      Sync local optimize artifacts to Overmind
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from unittest.mock import MagicMock

from dotenv import load_dotenv
from opentelemetry import context
from opentelemetry import trace as _otel_trace
from opentelemetry.trace import Status, StatusCode

import overmind
from overmind import attrs
from overmind.commands.agent_cmd import (
    cmd_list,
    cmd_register,
    cmd_remove,
    cmd_show,
    cmd_update,
    cmd_validate,
)
from overmind.commands.init_cmd import main as _init
from overmind.commands.optimize_cmd import main as _optimize
from overmind.commands.optimize_step_cmd import (
    build_subparser as _build_optimize_step_parser,
)
from overmind.commands.optimize_step_cmd import main as _optimize_step
from overmind.commands.setup_cmd import main as _setup
from overmind.core.constants import OVERMIND_DIR_NAME, overmind_rel
from overmind.core.logging import setup_logging
from overmind.core.paths import load_agent_dotenv, load_overmind_dotenv
from overmind.core.registry import require_overmind_initialized
from overmind.tracing import force_flush_traces

_FMT = argparse.RawDescriptionHelpFormatter


def _bundle_cli_kwargs(args: object) -> dict:
    """Return scope / bundle cap kwargs for setup & optimize (test-safe).

    When the CLI is exercised via ``MagicMock`` parse results (unit tests),
    unspecified attributes auto-resolve to nested mocks — coerce those to
    ``None`` so downstream commands receive real optional values.
    """
    scope_globs = getattr(args, "scope_globs", None)
    max_files = getattr(args, "max_files", None)
    max_chars = getattr(args, "max_chars", None)
    if isinstance(scope_globs, MagicMock):
        scope_globs = None
    if isinstance(max_files, MagicMock):
        max_files = None
    if isinstance(max_chars, MagicMock):
        max_chars = None
    return {
        "scope_globs": scope_globs,
        "max_files": max_files,
        "max_chars": max_chars,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overmind",
        formatter_class=_FMT,
        description="Overmind — autonomous agent optimization through structured experimentation.",
        epilog=(
            "Typical workflow:\n"
            "  1. overmind init                                  # set API keys + models\n"
            "  2. overmind agent register <name> <module:fn>     # register your agent\n"
            "  3. overmind setup <name>                          # build eval criteria\n"
            "  4. overmind optimize <name>                       # run the optimizer\n"
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
            "  - ANALYZER_MODEL        (used by setup and optimize)\n"
            "  - SYNTHETIC_DATAGEN_MODEL  (used by setup for test-data generation)\n"
            "\n"
            "Run once per project before using setup or optimize.\n"
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
            f"Output is saved to {overmind_rel('agents', '<name>', 'setup_spec', 'eval_spec.json')}.\n"
            "Run this before overmind optimize."
        ),
        epilog=(
            "Flags:\n"
            "  --fast      Skips all interactive prompts. Requires ANALYZER_MODEL\n"
            f"              and SYNTHETIC_DATAGEN_MODEL set in {overmind_rel('.env')}.\n"
            "  --data      JSON seed dataset file or a directory of *.json files.\n"
            "              Omit this flag to choose in the wizard (or run without seed).\n"
            "  --policy    Path to an existing policy document (.md or .txt).\n"
            "              Overmind will analyze it against your agent code and\n"
            "              suggest improvements before using it.\n"
            "\n"
            "Examples:\n"
            "  overmind setup lead-qualification\n"
            "  overmind setup lead-qualification --data ./data/cases.json\n"
            "  overmind setup lead-qualification --data ./seed_data/\n"
            "  overmind setup lead-qualification --fast\n"
            "  overmind setup lead-qualification --policy docs/domain-rules.md\n"
        ),
    )
    setup_p.add_argument(
        "agent",
        metavar="AGENT_NAME",
        help="registered agent name (see: overmind agent list)",
    )
    setup_p.add_argument(
        "--fast",
        action="store_true",
        help=(f"skip all prompts; requires ANALYZER_MODEL and SYNTHETIC_DATAGEN_MODEL in {overmind_rel('.env')}"),
    )
    setup_p.add_argument(
        "--policy",
        default=None,
        metavar="POLICY_PATH",
        help="path to an existing policy/domain document (.md, .txt)",
    )
    setup_p.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help=(
            "path to a JSON seed dataset file or a directory of *.json files "
            "(optional; wizard can remind you of this flag)"
        ),
    )
    setup_p.add_argument(
        "--scope",
        dest="scope_globs",
        action="append",
        default=None,
        metavar="GLOB",
        help=(
            "optimizable path glob relative to project root (repeatable); "
            "passed as hints to the analyzer for `scope.optimizable_paths`"
        ),
    )
    setup_p.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="max files to follow from the entrypoint during Phase 1 analysis (default: 48)",
    )
    setup_p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        metavar="N",
        help="max total characters for dependency context during Phase 1 (default: 80000)",
    )

    # ── optimize ─────────────────────────────────────────────────────────────
    opt_p = subparsers.add_parser(
        "optimize",
        formatter_class=_FMT,
        help="Run the optimization loop against your agent",
        description=(
            "Run the iterative optimization loop against your agent.\n"
            "\n"
            "Requires overmind setup to have been run first.\n"
            "\n"
            "Each iteration:\n"
            "  1. Runs the agent against the evaluation dataset\n"
            "  2. Diagnoses failures using the analyzer model\n"
            "  3. Generates and tests candidate improvements\n"
            "  4. Accepts changes that improve the score\n"
            "\n"
            f"Results and traces are saved to {overmind_rel('agents', '<name>', 'experiments')}/.\n"
            "The best-performing version is written to experiments/best_agent.py there."
        ),
        epilog=(
            "Flags:\n"
            "  --fast    Skips all interactive prompts. Requires ANALYZER_MODEL\n"
            f"            in {overmind_rel('.env')}. Uses defaults for all optimizer settings\n"
            "            (no LLM judge, no backtesting, 5 iterations).\n"
            "\n"
            "Examples:\n"
            "  overmind optimize lead-qualification\n"
            "  overmind optimize lead-qualification --fast\n"
        ),
    )
    opt_p.add_argument(
        "agent",
        metavar="AGENT_NAME",
        help="registered agent name (see: overmind agent list)",
    )
    opt_p.add_argument(
        "--fast",
        action="store_true",
        help=f"skip all prompts; requires ANALYZER_MODEL in {overmind_rel('.env')}",
    )
    opt_p.add_argument(
        "--scope",
        dest="scope_globs",
        action="append",
        default=None,
        metavar="GLOB",
        help=(
            "override optimizable path globs from eval_spec (repeatable); "
            "replaces `scope.optimizable_paths` for this run"
        ),
    )
    opt_p.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="override max import-closure file count for the bundle (default: from Config)",
    )
    opt_p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        metavar="N",
        help="override max total characters in the bundle (default: from Config)",
    )

    # ── optimize-step (skill-driven) ────────────────────────────────────────
    _build_optimize_step_parser(subparsers)

    # ── doctor ───────────────────────────────────────────────────────────────
    doctor_p = subparsers.add_parser(
        "doctor",
        formatter_class=_FMT,
        help="Diagnose bundle scope and eval spec for a registered agent (read-only)",
        description=(
            "Prints how Overmind would resolve the multi-file bundle: file count, "
            "character budget, suggested scope from eval_spec.json, and wrapper status. "
            "Does not call any LLM and does not modify files."
        ),
        epilog=("Example:\n  overmind doctor my-agent\n"),
    )
    doctor_p.add_argument(
        "agent",
        metavar="AGENT_NAME",
        help="registered agent name (see: overmind agent list)",
    )

    return parser


def _resolve_agent_name_for_env(args: argparse.Namespace) -> str | None:
    """Best-effort agent-name lookup so we can load the per-agent ``.env``.

    Agent-scoped subcommands carry the name in different places:

    - ``agent``/``setup``/``optimize``/``doctor``/``sync``: ``args.name`` or
      ``args.agent`` directly.
    - ``optimize-step init``: ``args.agent``.
    - Other ``optimize-step`` subcommands: persisted in
      ``skill_state.json`` (read via ``--state``).

    Returns ``None`` when no agent context is in scope (e.g. ``init``).
    """
    name = getattr(args, "name", None) or getattr(args, "agent", None)
    if isinstance(name, str) and name:
        return name
    state = getattr(args, "state", None)
    if isinstance(state, str) and state:
        try:
            import json
            from pathlib import Path

            data = json.loads(Path(state).read_text())
            agent = data.get("agent_name") or (data.get("config") or {}).get("agent_name")
            if isinstance(agent, str) and agent:
                return agent
        except Exception:
            return None
    return None


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
    # Per-agent ``.env`` (``.overmind/agents/<name>/.env``) holds the
    # provider keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, …) that the
    # analyzer / codegen LLM calls need. Without this, ``optimize-step``
    # subcommands silently lose API keys and the analyzer fails with a
    # litellm AuthenticationError that gets swallowed deep in the loop.
    _agent_name = _resolve_agent_name_for_env(args)
    if _agent_name:
        load_agent_dotenv(_agent_name)
    # ``optimize-step`` is invoked non-interactively by a host coding agent
    # via SKILL.md.  Avoid prompting for OVERMIND_API_KEY when none is set —
    # initialise with a placeholder so spans/tags work locally even without
    # a real OTLP endpoint.
    if args.command == "optimize-step" and not os.getenv("OVERMIND_API_KEY"):
        os.environ["OVERMIND_API_KEY"] = "skill-local-no-export"

    # Distributed tracing across optimize-step CLI invocations: read the
    # W3C traceparent persisted by ``optimize-step init`` and export it
    # into the environment BEFORE ``overmind.init()`` runs.  The SDK's
    # :func:`_attach_remote_parent_if_present` picks it up so every span
    # this step emits becomes a child of the workflow root span emitted
    # at init time — single trace_id, single Job row in the UI, regardless
    # of how many separate shells the host coding agent spawns to drive
    # the loop.  ``init`` skips this (it owns root span creation), and
    # any pre-set ``TRACEPARENT`` from an enclosing tracer wins over the
    # state-file value.
    if (
        args.command == "optimize-step"
        and getattr(args, "step", None) != "init"
        and not os.getenv("TRACEPARENT")
        and not os.getenv("OTEL_TRACEPARENT")
    ):
        state_path = getattr(args, "state", None)
        if isinstance(state_path, str) and state_path:
            try:
                import json as _json
                from pathlib import Path as _Path

                _state = _json.loads(_Path(state_path).read_text())
                _tp = _state.get("traceparent")
                if isinstance(_tp, str) and _tp:
                    os.environ["TRACEPARENT"] = _tp
            except Exception:
                # Bad / missing state file falls through to legacy
                # ``overmind.job.id``-based coalescing in OTLP ingest.
                logging.getLogger("overmind.cli").debug(
                    "optimize-step: could not read traceparent from %s",
                    state_path,
                    exc_info=True,
                )

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

        elif args.command == "setup":
            _kw = _bundle_cli_kwargs(args)
            context.attach(context.set_value(attrs.AGENT_NAME, args.agent))
            _setup(
                agent_name=args.agent,
                fast=args.fast,
                policy=args.policy,
                data=args.data,
                **_kw,
            )

        elif args.command == "optimize":
            _kw = _bundle_cli_kwargs(args)
            context.attach(context.set_value(attrs.AGENT_NAME, args.agent))
            _optimize(agent_name=args.agent, fast=args.fast, **_kw)

        elif args.command == "optimize-step":
            agent_name = getattr(args, "agent", None)
            if agent_name:
                context.attach(context.set_value(attrs.AGENT_NAME, agent_name))
            raise SystemExit(_optimize_step(args))

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
