"""Tests for the daemon command handlers (bundle / run / apply / reset)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from overmind.daemon import handlers, safety


def _git(args: list[str], cwd: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env)


def _init_repo(repo: Path) -> None:
    _git(["init", "-q", "-b", "main"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "init"], repo)


def _make_diff(repo: Path, target: Path, addition: str) -> str:
    """Produce a git unified diff that appends *addition*, then restore the tree."""
    target.write_text(target.read_text() + addition)
    diff = subprocess.run(
        ["git", "diff"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    _git(["checkout", "--", "."], repo)
    return diff


def _current_branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_dispatch_unknown_kind_reports_error(tmp_project: Path):
    ctx = handlers.HandlerContext.create()
    ok, result, error = handlers.dispatch({"kind": "frobnicate", "payload": {}}, ctx)
    assert ok is False
    assert result == {}
    assert "unknown" in error


def test_upload_bundle_returns_agent_files(tmp_project: Path):
    ctx = handlers.HandlerContext.create()
    ok, result, error = handlers.dispatch(
        {
            "kind": "upload_bundle",
            "payload": {
                "agent_path": "agents/agent1/sample_agent.py",
                "entrypoint_fn": "run",
            },
        },
        ctx,
    )
    assert ok is True, error
    files = result["bundle"]["files"]
    assert any(f["path"].endswith("sample_agent.py") for f in files)
    assert result["bundle"]["entry_file"].endswith("sample_agent.py")


def test_run_command_invokes_runner(tmp_project: Path, monkeypatch):
    # Avoid real venv provisioning: stand in a fake runner that echoes input.
    class FakeOut:
        success = True
        data = {"echo": "hi"}
        stdout = "ran"
        stderr = ""
        error = ""

    class FakeRunner:
        def ensure_environment(self):
            return None

        def run(self, input_data):
            assert input_data == {"q": "hi"}
            return FakeOut()

    ctx = handlers.HandlerContext.create()
    monkeypatch.setattr(ctx, "runner_for", lambda *a, **k: FakeRunner())
    ok, result, error = handlers.dispatch(
        {
            "kind": "run_command",
            "payload": {
                "agent_path": "agents/agent1/sample_agent.py",
                "entrypoint_fn": "run",
                "input": {"q": "hi"},
            },
        },
        ctx,
    )
    assert ok is True, error
    assert result["output"] == {"echo": "hi"}
    # When telemetry is live the run carries a 32-hex trace id for correlation; an
    # offline daemon simply omits it. Either way it must never be malformed.
    if "trace_id" in result:
        assert len(result["trace_id"]) == 32
        int(result["trace_id"], 16)


def test_run_command_returns_trace_id_for_correlation(tmp_project: Path, monkeypatch):
    """When tracing is live, the run command carries its span's 32-hex trace id back.

    The server stores this on the command result; the baseline batch's ids become
    each later replay's ``original_trace_id`` so the two traces can be compared.
    """
    from contextlib import contextmanager

    class _FakeSpanContext:
        is_valid = True
        trace_id = 0x0123456789ABCDEF0123456789ABCDEF

    class _FakeSpan:
        def get_span_context(self):
            return _FakeSpanContext()

    @contextmanager
    def _fake_trace(_payload):
        yield _FakeSpan()

    monkeypatch.setattr(handlers, "_run_trace", _fake_trace)

    class FakeOut:
        success = True
        data = {"echo": "hi"}
        stdout = "ran"
        stderr = ""
        error = ""

    class FakeRunner:
        def ensure_environment(self):
            return None

        def run(self, input_data):
            return FakeOut()

    ctx = handlers.HandlerContext.create()
    monkeypatch.setattr(ctx, "runner_for", lambda *a, **k: FakeRunner())
    ok, result, error = handlers.dispatch(
        {"kind": "run_command", "payload": {"agent_path": "agents/agent1/sample_agent.py", "input": {}}},
        ctx,
    )
    assert ok is True, error
    assert result["trace_id"] == "0123456789abcdef0123456789abcdef"


def test_run_command_failure_is_reported(tmp_project: Path, monkeypatch):
    class FakeOut:
        success = False
        data = None
        stdout = ""
        stderr = "boom traceback"
        error = "boom"

    class FakeRunner:
        def ensure_environment(self):
            return None

        def run(self, input_data):
            return FakeOut()

    ctx = handlers.HandlerContext.create()
    monkeypatch.setattr(ctx, "runner_for", lambda *a, **k: FakeRunner())
    ok, _result, error = handlers.dispatch(
        {"kind": "run_command", "payload": {"agent_path": "agents/agent1/sample_agent.py", "input": {}}},
        ctx,
    )
    assert ok is False
    assert "boom" in error


def test_apply_patch_then_reset_round_trips(tmp_project: Path):
    repo = tmp_project
    target = repo / "agents" / "agent1" / "sample_agent.py"

    _git(["init", "-q"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "init"], repo)

    # Produce a real unified diff via git, then restore the clean tree.
    original = target.read_text()
    target.write_text(original + "\n# sentinel-marker\n")
    diff = subprocess.run(
        ["git", "diff"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    _git(["checkout", "--", "."], repo)
    assert "sentinel-marker" not in target.read_text()

    ctx = handlers.HandlerContext.create()

    ok, result, error = handlers.dispatch({"kind": "apply_patch", "payload": {"diff": diff}}, ctx)
    assert ok is True, error
    assert result["applied"] is True
    assert "sentinel-marker" in target.read_text()

    ok, result, error = handlers.dispatch({"kind": "reset", "payload": {"diff": diff}}, ctx)
    assert ok is True, error
    assert result["reset"] is True
    assert "sentinel-marker" not in target.read_text()


def test_empty_diff_is_a_noop(tmp_project: Path):
    ctx = handlers.HandlerContext.create()
    ok, result, _error = handlers.dispatch({"kind": "apply_patch", "payload": {"diff": "  "}}, ctx)
    assert ok is True
    assert result["applied"] is False


# --- mirrored-branch model -------------------------------------------------


def test_apply_patch_creates_candidate_branch_and_commits(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    target = repo / "agents" / "agent1" / "sample_agent.py"
    diff = _make_diff(repo, target, "# sentinel-marker\n")
    ctx = handlers.HandlerContext.create()

    ok, result, error = handlers.dispatch(
        {
            "kind": "apply_patch",
            "payload": {
                "base_branch": "main",
                "branch": "overmind/optimize/abc1234/i0c0",
                "diff": diff,
                "reset_to_base": True,
            },
        },
        ctx,
    )

    assert ok is True, error
    assert result["applied"] is True
    assert result["branch"] == "overmind/optimize/abc1234/i0c0"
    assert _current_branch(repo) == "overmind/optimize/abc1234/i0c0"
    assert "sentinel-marker" in target.read_text()
    # The diff was committed, so the tree is clean (never left dirty).
    assert safety.run_git(repo, ["status", "--porcelain"]).strip() == ""


def test_apply_patch_accumulates_winner_on_working_branch(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    safety.run_git(repo, ["checkout", "-B", "overmind/optimize/abc1234/work", "main"])
    target = repo / "agents" / "agent1" / "sample_agent.py"
    diff = _make_diff(repo, target, "# winner-marker\n")
    ctx = handlers.HandlerContext.create()

    ok, _result, error = handlers.dispatch(
        {
            "kind": "apply_patch",
            "payload": {
                "base_branch": "overmind/optimize/abc1234/work",
                "branch": "overmind/optimize/abc1234/work",
                "diff": diff,
                "reset_to_base": False,  # commit in place
            },
        },
        ctx,
    )

    assert ok is True, error
    assert _current_branch(repo) == "overmind/optimize/abc1234/work"
    assert "winner-marker" in target.read_text()


def test_apply_patch_assert_clean_rejects_dirty_repo(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    (repo / "scratch.txt").write_text("uncommitted work")  # dirty (untracked)
    ctx = handlers.HandlerContext.create()

    ok, _result, error = handlers.dispatch(
        {
            "kind": "apply_patch",
            "payload": {
                "base_branch": "main",
                "branch": "overmind/optimize/abc1234/work",
                "diff": "",
                "reset_to_base": True,
                "assert_clean": True,
            },
        },
        ctx,
    )

    assert ok is False
    assert "uncommitted changes" in error


def test_apply_patch_parks_on_base_sha_mismatch(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    ctx = handlers.HandlerContext.create()

    ok, _result, error = handlers.dispatch(
        {
            "kind": "apply_patch",
            "payload": {
                "base_branch": "main",
                "branch": "overmind/optimize/abc1234/work",
                "diff": "",
                "reset_to_base": True,
                "base_sha": "0" * 40,  # not the local HEAD
            },
        },
        ctx,
    )

    assert ok is False
    assert "not at the commit" in error


def test_reset_checks_out_base_and_drops_run_branches(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    safety.run_git(repo, ["checkout", "-B", "overmind/optimize/abc1234/work", "main"])
    safety.run_git(repo, ["checkout", "-B", "overmind/optimize/abc1234/i0c0", "main"])
    ctx = handlers.HandlerContext.create()

    ok, _result, error = handlers.dispatch(
        {
            "kind": "reset",
            "payload": {"base_branch": "main", "cleanup_prefix": "overmind/optimize/abc1234"},
        },
        ctx,
    )

    assert ok is True, error
    assert _current_branch(repo) == "main"
    branches = safety.run_git(repo, ["branch", "--format=%(refname:short)"])
    assert "overmind/optimize/abc1234/work" not in branches
    assert "overmind/optimize/abc1234/i0c0" not in branches


def test_reset_drops_single_candidate_branch(tmp_project: Path):
    repo = tmp_project
    _init_repo(repo)
    safety.run_git(repo, ["checkout", "-B", "overmind/optimize/abc1234/work", "main"])
    safety.run_git(repo, ["checkout", "-B", "overmind/optimize/abc1234/i0c0", "main"])
    ctx = handlers.HandlerContext.create()

    ok, result, error = handlers.dispatch(
        {
            "kind": "reset",
            "payload": {
                "base_branch": "overmind/optimize/abc1234/work",
                "branch": "overmind/optimize/abc1234/i0c0",
            },
        },
        ctx,
    )

    assert ok is True, error
    assert _current_branch(repo) == "overmind/optimize/abc1234/work"
    assert result["deleted"] == ["overmind/optimize/abc1234/i0c0"]
