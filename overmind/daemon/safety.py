"""Command allowlist for the CLI daemon.

The platform is the brain: it dispatches commands for the daemon to run on the
machine where the agent's code lives. To keep that channel from becoming a
remote-code-execution vector, the daemon may only ever run a small, fixed set of
commands — a handful of non-destructive git working-tree operations plus the
agent run. Every subprocess the daemon spawns on the server's behalf is checked
here *before* it is launched.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

# The exact command contract shared with the platform, kept as display strings
# for parity with the server-side documentation.
ALLOWED_COMMANDS: tuple[str, ...] = (
    "git apply",
    "git reset",
    "git checkout",
    "git stash",
    "overmind agent run",
)

# Git subcommands the daemon may invoke (the git half of ``ALLOWED_COMMANDS``).
ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({"apply", "reset", "checkout", "stash"})


class UnsafeCommandError(RuntimeError):
    """Raised when the daemon is asked to run a command outside the allowlist."""


def assert_safe(argv: Sequence[str]) -> None:
    """Raise :class:`UnsafeCommandError` unless ``argv`` is on the allowlist."""
    if not argv:
        raise UnsafeCommandError("refusing to run an empty command")

    # Match on the basename so an absolute path (``/usr/bin/git``) is allowed.
    program = os.path.basename(str(argv[0]))

    if program == "git":
        if len(argv) < 2:
            raise UnsafeCommandError("refusing to run bare 'git' with no subcommand")
        subcommand = str(argv[1])
        if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
            raise UnsafeCommandError(f"git subcommand not allowed: {subcommand}")
        return

    if program == "overmind":
        if list(argv[1:3]) == ["agent", "run"]:
            return
        attempted = " ".join(str(a) for a in argv[1:3]) or "<none>"
        raise UnsafeCommandError(f"overmind command not allowed: {attempted}")

    raise UnsafeCommandError(f"command not allowed: {program}")


def run(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``argv`` after asserting it is on the allowlist."""
    assert_safe(argv)
    return subprocess.run(
        list(argv),
        cwd=cwd,
        timeout=timeout,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def run_git(
    *args: str,
    cwd: str | Path | None = None,
    **kwargs: object,
) -> subprocess.CompletedProcess:
    """Run an allowlisted ``git`` command, e.g. ``run_git("apply", patch_path)``."""
    return run(["git", *args], cwd=cwd, **kwargs)
