"""No-network sanity checks for the optimizer client's command-running path.

Covers ``run_command`` (traceparent shape, success/failure round trip) plus the
per-command patch selection and apply/reset flow in ``poll_once`` — the fix for
the bug where every candidate in an iteration ran against candidate 0's diff.
"""

from __future__ import annotations

import subprocess

from overmind.optimizer import _select_patch, poll_once, run_command


def test_traceparent_shape():
    from overmind.optimizer import _new_traceparent

    header, trace_id = _new_traceparent()
    assert header == f"00-{trace_id}-{header.split('-')[2]}-01", header
    assert len(trace_id) == 32 and int(trace_id, 16) >= 0, trace_id


def test_run_command_success_round_trip():
    ok, result, error = run_command({"command": "echo hi", "timeout": 10})
    assert ok and result["output"].strip() == "hi" and len(result["trace_id"]) == 32, (ok, result)
    assert error == "", error


def test_run_command_failure_reports_exit_code_and_stderr():
    ok, result, error = run_command({"command": "echo boom >&2; exit 3", "timeout": 10})
    assert not ok and result["exit_code"] == 3 and "boom" in error, (ok, result, error)


def test_select_patch_prefers_code_path_even_when_empty():
    # code_path present (even "") beats iteration_patch — "" means baseline.
    assert _select_patch({"code_path": "", "iteration_patch": "legacy-diff"}) == ""
    assert _select_patch({"code_path": "diff-a", "iteration_patch": "diff-b"}) == "diff-a"


def test_select_patch_falls_back_to_iteration_patch_when_code_path_absent():
    assert _select_patch({"iteration_patch": "legacy-diff"}) == "legacy-diff"
    assert _select_patch({}) == ""


class _FakeAPI:
    """Stub for the two ``poll_once`` calls out to the real backend."""

    def __init__(self, commands: list[dict]):
        self._commands = commands
        self.results: dict[str, tuple[bool, dict, str]] = {}

    def poll(self, session_id: str) -> list[dict]:
        commands, self._commands = self._commands, []
        return commands

    def submit_result(self, command_id: str, *, success: bool, result: dict, error: str) -> None:
        self.results[command_id] = (success, result, error)


def _init_repo(tmp_path) -> str:
    cwd = str(tmp_path)
    run = lambda *args: subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    (tmp_path / "f.txt").write_text("base\n")
    run("add", "-A")
    run("commit", "-q", "-m", "init")
    return cwd


def _diff_for(value: str) -> str:
    return f"diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-base\n+{value}\n"


def test_poll_once_applies_each_commands_own_patch_not_the_first(tmp_path):
    """Acceptance check from the spec: N candidates with distinct diffs each
    run against their own diff, and the baseline candidate runs unpatched —
    not three runs of candidate 0."""
    cwd = _init_repo(tmp_path)
    commands = [
        {"id": "cand-0", "command": "cat f.txt", "code_path": "", "timeout": 10},
        {"id": "cand-1", "command": "cat f.txt", "code_path": _diff_for("candidate-1"), "timeout": 10},
        {"id": "cand-2", "command": "cat f.txt", "code_path": _diff_for("candidate-2"), "timeout": 10},
    ]
    api = _FakeAPI(commands)

    ran = poll_once(api, "session", cwd)

    assert ran == 3
    assert api.results["cand-0"][0] and api.results["cand-0"][1]["output"].strip() == "base"
    assert api.results["cand-1"][0] and api.results["cand-1"][1]["output"].strip() == "candidate-1"
    assert api.results["cand-2"][0] and api.results["cand-2"][1]["output"].strip() == "candidate-2"
    # Tree is left clean at the last-run command's patch — not stuck mid-batch.
    assert (tmp_path / "f.txt").read_text().strip() == "candidate-2"


def test_poll_once_reports_failure_when_patch_does_not_apply(tmp_path):
    cwd = _init_repo(tmp_path)
    bad_diff = "diff --git a/nope.txt b/nope.txt\n--- a/nope.txt\n+++ b/nope.txt\n@@ -1 +1 @@\n-x\n+y\n"
    commands = [{"id": "cand-bad", "command": "cat f.txt", "code_path": bad_diff, "timeout": 10}]
    api = _FakeAPI(commands)

    ran = poll_once(api, "session", cwd)

    assert ran == 1
    success, _result, error = api.results["cand-bad"]
    assert not success and "git apply failed" in error


def test_poll_once_falls_back_to_iteration_patch_for_old_backends(tmp_path):
    cwd = _init_repo(tmp_path)
    commands = [{"id": "cand-legacy", "command": "cat f.txt", "iteration_patch": _diff_for("legacy"), "timeout": 10}]
    api = _FakeAPI(commands)

    poll_once(api, "session", cwd)

    assert api.results["cand-legacy"][0]
    assert api.results["cand-legacy"][1]["output"].strip() == "legacy"
