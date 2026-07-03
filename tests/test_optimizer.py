"""No-network sanity checks for the optimizer client's command-running path.

Moved from the old ``selftest()`` in :mod:`overmind.optimizer` — asserts the
traceparent shape and that a trivial command round-trips into the result dict
(success + captured output + trace id), and that a failing command reports
failure with stderr.  Does not exercise ``_setup_iteration_branch`` (needs a
real git repo).
"""

from __future__ import annotations

from overmind.optimizer import _new_traceparent, run_command


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
