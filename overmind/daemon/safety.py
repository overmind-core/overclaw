"""Command safety: the daemon only ever shells out to a tiny git allowlist.

Patch apply/reset are the only mutating operations the server can ask the
daemon to run against the working tree, and they go through one chokepoint
(:func:`run_git`) that refuses anything outside :data:`ALLOWED_GIT_SUBCOMMANDS`.
Agent execution itself is isolated subprocess work handled by ``AgentRunner`` —
never a server-supplied shell string.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Read-only inspection + the branch/patch ops the mirrored-branch orchestrator
# needs (the client recreates the server's branches and commits the same diffs).
# Deliberately no `push`, `clone`, `remote`, or `config` — the daemon never writes
# to the remote and never reconfigures the repo; PRs are opened server-side.
ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {
        "apply",
        "reset",
        "checkout",
        "stash",
        "status",
        "rev-parse",
        "diff",
        "add",
        "commit",
        "branch",
        "fetch",
    }
)


class UnsafeCommandError(PermissionError):
    """Raised when a git subcommand is not on the allowlist."""


def assert_git_allowed(args: list[str]) -> None:
    if not args or args[0] not in ALLOWED_GIT_SUBCOMMANDS:
        raise UnsafeCommandError(
            f"git subcommand {args[0] if args else '(none)'!r} is not permitted; "
            f"allowed: {sorted(ALLOWED_GIT_SUBCOMMANDS)}"
        )


def run_git(
    repo_root: Path,
    args: list[str],
    *,
    stdin: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> str:
    """Run an allowlisted git command in *repo_root*; raise on failure.

    ``env`` is merged over the process environment (used to stamp a commit
    identity via ``GIT_*_NAME``/``EMAIL`` without a non-allowlisted ``git -c``).
    """
    assert_git_allowed(args)
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **env} if env else None,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git command failed").strip()[:2000])
    return proc.stdout
