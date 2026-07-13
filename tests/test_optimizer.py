"""No-network sanity checks for the optimizer client's command-running path.

Moved from the old ``selftest()`` in :mod:`overmind.optimizer` — asserts the
traceparent shape and that a trivial command round-trips into the result dict
(success + captured output + trace id), and that a failing command reports
failure with stderr.  Does not exercise ``_setup_iteration_branch`` (needs a
real git repo).
"""

from __future__ import annotations

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
