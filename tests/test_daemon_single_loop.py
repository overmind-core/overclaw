"""Tests for the daemon single-loop command claim/execute/report cycle."""

from __future__ import annotations

from unittest.mock import MagicMock

from overmind.daemon.handlers import CommandHandler
from overmind.daemon.main import _daemon_project_root, single_loop


def test_single_loop_fetches_executes_and_reports():
    client = MagicMock()
    handlers = MagicMock(spec=CommandHandler)
    handlers.dispatch.return_value = (True, {"ok": True}, "")

    client.fetch_next_command.return_value = {
        "id": "cmd-1",
        "kind": "reset",
        "payload": {"optimize_run_id": "run-1"},
    }

    assert single_loop(client, "session-1", handlers) is True
    client.fetch_next_command.assert_called_once_with("session-1")
    handlers.dispatch.assert_called_once()
    client.submit_command_result.assert_called_once_with(
        "session-1",
        "cmd-1",
        success=True,
        result={"ok": True},
        error="",
    )


def test_single_loop_returns_false_when_idle():
    client = MagicMock()
    client.fetch_next_command.return_value = None
    handlers = MagicMock(spec=CommandHandler)

    assert single_loop(client, "session-1", handlers) is False
    handlers.dispatch.assert_not_called()
    client.submit_command_result.assert_not_called()


def test_daemon_root_does_not_escape_repo_to_stray_overmind(tmp_path, monkeypatch):
    """A stray `.overmind/` above the repo must not hijack the daemon's root.

    Reproduces the `KeyError: '<entry path>'` from upload_bundle: the un-inited
    repo's daemon was rooting bundles at `$HOME` (the only `.overmind/` ancestor).
    """
    home = tmp_path
    (home / ".overmind").mkdir()  # stray, e.g. ~/.overmind
    repo = home / "work" / "platform"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()  # un-inited repo: has .git, no .overmind

    monkeypatch.chdir(repo)
    assert _daemon_project_root() == repo.resolve()


def test_daemon_root_prefers_inited_project_within_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".overmind").mkdir()
    subdir = repo / "services" / "agent"
    subdir.mkdir(parents=True)

    monkeypatch.chdir(subdir)
    assert _daemon_project_root() == repo.resolve()
