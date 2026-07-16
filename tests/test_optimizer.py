"""No-network sanity checks for the optimizer client's command-running path.

Moved from the old ``selftest()`` in :mod:`overmind.optimizer` — asserts the
traceparent shape and that a trivial command round-trips into the result dict
(success + captured output + trace id), and that a failing command reports
failure with stderr.  Exercises ``_setup_candidate_branch`` with a real temp
git repo.
"""

from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from overmind.optimizer import (
    OptimizerAPI,
    _new_traceparent,
    _runtime_metadata,
    _start_heartbeat_thread,
    run_command,
)


def test_runtime_metadata_reports_python_and_uv():
    meta = _runtime_metadata()
    assert meta["python.executable"]
    assert meta["python.version_info"].count(".") == 2
    assert isinstance(meta["uv.exists"], bool)
    if meta["uv.exists"]:
        assert meta["uv.path"]
        assert meta["uv.version"]
    else:
        assert meta["uv.path"] is None
        assert meta["uv.version"] is None


def test_traceparent_shape():
    header, trace_id = _new_traceparent()
    assert header == f"00-{trace_id}-{header.split('-')[2]}-01", header
    assert len(trace_id) == 32 and int(trace_id, 16) >= 0, trace_id


def test_run_command_success_round_trip():
    # No iteration_id → branch check is skipped; exercises the pure run path.
    ok, result, error = run_command({"command": "echo hi", "timeout": 10})
    assert ok and result["output"].strip() == "hi" and len(result["trace_id"]) == 32, (ok, result)
    assert error == "", error


def test_run_command_failure_reports_exit_code_and_stderr():
    ok, result, error = run_command({"command": "echo boom >&2; exit 3", "timeout": 10})
    assert not ok and result["exit_code"] == 3 and "boom" in error, (ok, result, error)


# ── OptimizerAPI.poll sends lease flag ───────────────────────────────────────


def _make_api(responses: list):
    """Return an OptimizerAPI whose requests.Session is mocked."""
    api = OptimizerAPI.__new__(OptimizerAPI)
    api._lock = threading.Lock()
    mock_session = MagicMock()
    # Each call to session.post returns the next response in the list.
    mock_session.post.side_effect = [
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: r)
        for r in responses
    ]
    api.session = mock_session
    api.base_url = "http://test"
    return api


def test_poll_lease_true_sends_lease_true():
    api = _make_api([{"commands": [{"id": "abc"}]}])
    result = api.poll("sess-1", lease=True)
    assert result == [{"id": "abc"}]
    _, kwargs = api.session.post.call_args
    assert kwargs["json"]["lease"] is True


def test_poll_lease_false_sends_lease_false():
    api = _make_api([{"commands": []}])
    result = api.poll("sess-1", lease=False)
    assert result == []
    _, kwargs = api.session.post.call_args
    assert kwargs["json"]["lease"] is False


def test_poll_default_lease_is_true():
    api = _make_api([{"commands": []}])
    api.poll("sess-1")
    _, kwargs = api.session.post.call_args
    assert kwargs["json"]["lease"] is True


# ── heartbeat thread calls poll(lease=False) while main loop is busy ─────────


def test_heartbeat_thread_pings_while_main_is_blocked():
    """Heartbeat thread fires at least once while main thread simulates a blocked run."""
    pings: list[bool] = []

    def fake_poll(session_id, *, lease=True):
        pings.append(lease)
        return []

    api = OptimizerAPI.__new__(OptimizerAPI)
    api._lock = threading.Lock()
    api.base_url = "http://test"
    api.session = MagicMock()
    api.poll = fake_poll  # type: ignore[method-assign]

    _start_heartbeat_thread(api, "sess-hb", heartbeat_ping_interval=0.05)
    time.sleep(0.2)  # let the heartbeat thread fire a few times

    # The thread should have fired heartbeat pings (lease=False).
    assert pings.count(False) >= 1
    # The main loop has not called poll at all, so no lease=True pings.
    assert pings.count(True) == 0


# ── _setup_candidate_branch: per-candidate patches from a shared base ─────────


def _init_git_repo(path):
    """Create a minimal git repo with one file committed on a 'main' branch."""
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    (path / "agent.py").write_text("PROMPT = 'v0'\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def _make_patch(old: str, new: str, filename: str = "agent.py") -> str:
    """Build a minimal unified diff string replacing ``old`` with ``new`` in ``filename``."""
    import difflib

    return "".join(
        difflib.unified_diff(
            [old + "\n"],
            [new + "\n"],
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


def test_setup_candidate_branch_distinct_patches_produce_distinct_contents(tmp_path):
    """Two candidates with different patches each land on their own branch with distinct file contents."""
    from overmind.optimizer import _setup_candidate_branch

    _init_git_repo(tmp_path)
    base_ref = "main"

    patch_a = _make_patch("PROMPT = 'v0'", "PROMPT = 'candidate-a'")
    patch_b = _make_patch("PROMPT = 'v0'", "PROMPT = 'candidate-b'")

    _setup_candidate_branch(base_ref, "cand-a", patch_a, tmp_path)
    content_a = (tmp_path / "agent.py").read_text()

    _setup_candidate_branch(base_ref, "cand-b", patch_b, tmp_path)
    content_b = (tmp_path / "agent.py").read_text()

    assert "candidate-a" in content_a, content_a
    assert "candidate-b" in content_b, content_b
    assert content_a != content_b

    # Both branches exist and each points to its own commit (not main's commit).
    main_sha = subprocess.run(["git", "rev-parse", "main"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    sha_a = subprocess.run(["git", "rev-parse", "cand-a"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    sha_b = subprocess.run(["git", "rev-parse", "cand-b"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert sha_a != main_sha
    assert sha_b != main_sha
    assert sha_a != sha_b


def test_setup_candidate_branch_cumulative_patch_applies_cleanly(tmp_path):
    """A cumulative patch (base improvement + extra change) applies correctly from main."""
    from overmind.optimizer import _setup_candidate_branch

    _init_git_repo(tmp_path)
    base_ref = "main"

    # Simulate a cumulative diff: best-so-far change + new candidate change, all from main.
    cumulative_patch = _make_patch("PROMPT = 'v0'", "PROMPT = 'v0 + best + extra'")

    _setup_candidate_branch(base_ref, "cand-cumulative", cumulative_patch, tmp_path)
    content = (tmp_path / "agent.py").read_text()

    assert "v0 + best + extra" in content, content


def test_setup_candidate_branch_idempotent_on_restart(tmp_path):
    """Calling setup again for the same candidate (restart scenario) resets to a clean state."""
    from overmind.optimizer import _setup_candidate_branch

    _init_git_repo(tmp_path)
    base_ref = "main"
    patch = _make_patch("PROMPT = 'v0'", "PROMPT = 'restarted'")

    _setup_candidate_branch(base_ref, "cand-r", patch, tmp_path)
    # Call again — should force-reset without error.
    _setup_candidate_branch(base_ref, "cand-r", patch, tmp_path)
    content = (tmp_path / "agent.py").read_text()
    assert "restarted" in content, content
