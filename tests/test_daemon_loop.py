"""Tests for the daemon poll/execute/report cycle (poll_once)."""

from __future__ import annotations

from overmind.daemon import handlers, main


class FakeAPI:
    """Records submitted results; ``poll`` returns a fixed batch once."""

    def __init__(self, commands, *, fail_submit_for=None):
        self._commands = commands
        self._fail_submit_for = fail_submit_for or set()
        self.submitted: list[tuple[str, bool, str]] = []

    def poll(self, session_id, *, agent_name="", cli_version=""):
        return {"commands": self._commands}

    def submit_result(self, command_id, *, success, result, error):
        if command_id in self._fail_submit_for:
            raise RuntimeError("network down")
        self.submitted.append((command_id, success, error))


def test_poll_once_dispatches_and_reports(monkeypatch):
    cmds = [{"id": "c1", "kind": "x", "payload": {}}, {"id": "c2", "kind": "y", "payload": {}}]
    api = FakeAPI(cmds)
    seen = []

    def fake_dispatch(command, ctx):
        seen.append(command["id"])
        return True, {"ok": command["id"]}, ""

    monkeypatch.setattr(handlers, "dispatch", fake_dispatch)

    handled = main.poll_once(api, "sess-1", ctx=object(), agent_name="a")

    assert handled == 2
    assert seen == ["c1", "c2"]
    assert [s[0] for s in api.submitted] == ["c1", "c2"]
    assert all(s[1] for s in api.submitted)


def test_poll_once_reports_failures(monkeypatch):
    api = FakeAPI([{"id": "c1", "kind": "x", "payload": {}}])
    monkeypatch.setattr(handlers, "dispatch", lambda c, ctx: (False, {}, "kaboom"))

    main.poll_once(api, "sess-1", ctx=object())

    assert api.submitted == [("c1", False, "kaboom")]


def test_poll_once_survives_submit_errors(monkeypatch):
    # A failed result submission must not crash the loop (reconcile will time it out).
    api = FakeAPI([{"id": "c1", "kind": "x", "payload": {}}], fail_submit_for={"c1"})
    monkeypatch.setattr(handlers, "dispatch", lambda c, ctx: (True, {}, ""))

    handled = main.poll_once(api, "sess-1", ctx=object())

    assert handled == 1
    assert api.submitted == []
