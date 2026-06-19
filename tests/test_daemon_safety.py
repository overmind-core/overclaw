"""Tests for the daemon's git command allowlist — the only shell surface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from overmind.daemon import safety


class TestAllowlist:
    @pytest.mark.parametrize("sub", sorted(safety.ALLOWED_GIT_SUBCOMMANDS))
    def test_allowed_subcommands_pass(self, sub: str):
        safety.assert_git_allowed([sub])  # must not raise

    @pytest.mark.parametrize("sub", ["push", "clone", "config", "remote", "rm", "merge", "rebase"])
    def test_disallowed_subcommands_raise(self, sub: str):
        with pytest.raises(safety.UnsafeCommandError):
            safety.assert_git_allowed([sub])

    @pytest.mark.parametrize("sub", ["commit", "add", "branch", "fetch"])
    def test_mirrored_branch_subcommands_are_allowed(self, sub: str):
        # The mirrored-branch model needs to recreate branches + commit diffs locally.
        safety.assert_git_allowed([sub])  # must not raise

    def test_empty_args_raise(self):
        with pytest.raises(safety.UnsafeCommandError):
            safety.assert_git_allowed([])


class TestRunGit:
    def test_runs_allowlisted_command(self, tmp_path: Path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        out = safety.run_git(tmp_path, ["rev-parse", "--is-inside-work-tree"])
        assert out.strip() == "true"

    def test_blocks_disallowed_command(self, tmp_path: Path):
        with pytest.raises(safety.UnsafeCommandError):
            safety.run_git(tmp_path, ["push", "origin", "main"])

    def test_raises_on_git_failure(self, tmp_path: Path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        with pytest.raises(RuntimeError):
            safety.run_git(tmp_path, ["apply", "--whitespace=nowarn", "-"], stdin="not a diff\n")
