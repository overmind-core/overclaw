"""Sub-package housing the focused building blocks for ``overmind setup``.

``overmind/commands/setup_cmd.py`` is the user-facing entry point — it owns
the orchestration / Rich UI / branching logic.  The helpers in this package
are the self-contained pieces it composes (dependency probing, entrypoint
validation, subprocess smoke tests, and remote-API synchronisation).

Public surface
--------------
:func:`check_agent_dependencies`
    Detect external imports without a manifest and guide the user (or fail
    fast in ``--fast`` mode) until the agent has a ``requirements.txt`` or
    ``package.json``.
:func:`validate_agent_entrypoint`
    Make sure the registered entry function exists; offer to generate a
    wrapper when it doesn't.
:func:`smoke_test_agent`
    Run the agent once through :class:`AgentRunner` to confirm it actually
    starts and returns a result.
:func:`run_beginning_smoke_test` / :func:`run_end_smoke_test`
    Pre- and post-setup smoke tests that wrap :func:`smoke_test_agent` with
    Rich progress + error rendering.
:func:`ensure_remote_agent_id` / :func:`sync_setup_artifacts`
    Create / look up the remote agent record and push setup artifacts
    (spec, dataset, policy) to the Overmind backend.
"""

from overmind.commands.setup.dependencies import check_agent_dependencies
from overmind.commands.setup.entrypoint_validator import validate_agent_entrypoint
from overmind.commands.setup.remote_sync import (
    ensure_remote_agent_id,
    sync_setup_artifacts,
)
from overmind.commands.setup.smoke_test import (
    run_beginning_smoke_test,
    run_end_smoke_test,
    smoke_test_agent,
)

__all__ = [
    "check_agent_dependencies",
    "ensure_remote_agent_id",
    "run_beginning_smoke_test",
    "run_end_smoke_test",
    "smoke_test_agent",
    "sync_setup_artifacts",
    "validate_agent_entrypoint",
]
