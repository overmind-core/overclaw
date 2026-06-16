"""Tests for the daemon command safety layer (overmind.daemon.safety).

The client must only ever run the commands documented in the platform contract:
``git apply``, ``git reset``, ``git checkout``, ``git stash`` and the agent run.
"""

from __future__ import annotations

import subprocess

import pytest

from overmind.daemon import safety


class TestAllowlist:
    def test_allowed_commands_match_contract(self):
        assert safety.ALLOWED_COMMANDS == (
            "git apply",
            "git reset",
            "git checkout",
            "git stash",
            "overmind agent run",
        )

    def test_allowed_git_subcommands(self):
        assert safety.ALLOWED_GIT_SUBCOMMANDS == {"apply", "reset", "checkout", "stash"}


class TestAssertSafe:
    @pytest.mark.parametrize("subcommand", ["apply", "reset", "checkout", "stash"])
    def test_permitted_git_subcommands_pass(self, subcommand):
        safety.assert_safe(["git", subcommand, "."])

    def test_overmind_agent_run_passes(self):
        safety.assert_safe(["overmind", "agent", "run", "--agent-id", "abc"])

    def test_git_with_absolute_path_basename_matches(self):
        safety.assert_safe(["/usr/bin/git", "checkout", "--", "."])

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["git"],
            ["git", "push"],
            ["git", "clean", "-fdx"],
            ["git", "rm", "-rf", "."],
            ["git", "commit", "-m", "x"],
            ["rm", "-rf", "/"],
            ["overmind", "init"],
            ["overmind", "optimize"],
            ["bash", "-c", "echo hi"],
            ["python", "-c", "print(1)"],
        ],
    )
    def test_disallowed_commands_raise(self, argv):
        with pytest.raises(safety.UnsafeCommandError):
            safety.assert_safe(argv)


class TestRun:
    def test_run_rejects_unsafe_before_spawn(self, tmp_path):
        with pytest.raises(safety.UnsafeCommandError):
            safety.run(["git", "push", "origin", "main"], cwd=tmp_path)

    def test_run_git_executes_allowed_command(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        proc = safety.run_git("stash", "list", cwd=tmp_path)
        assert isinstance(proc, subprocess.CompletedProcess)
        assert proc.returncode == 0
