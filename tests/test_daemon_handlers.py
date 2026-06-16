"""Tests for the daemon command handlers (apply_patch / reset / run_agent).

These exercise the real git working-tree manipulation through the safety layer
using a throwaway repository created under ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from overmind import attrs as oc_attrs
from overmind.daemon.handlers import CommandHandler


@contextmanager
def _capture_spans():
    """Capture span names opened via ``start_span`` and tags set via ``set_tag``."""
    spans: list[str] = []
    tags: dict[str, object] = {}

    @contextmanager
    def fake_start_span(name, **_kwargs):
        spans.append(name)
        yield None

    with (
        patch("overmind.daemon.handlers.start_span", fake_start_span),
        patch("overmind.daemon.handlers.set_tag", lambda k, v: tags.__setitem__(k, v)),
    ):
        yield spans, tags


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "agent.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)


def _diff_for(path: Path, edit) -> str:
    """Apply ``edit`` to the tree, capture a clean diff, then restore the tree.

    Uses the staged diff so newly-added files are captured too, then unstages,
    discards tracked edits, and removes untracked files so the repo is pristine
    before the handler is exercised.
    """
    edit(path)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    diff = subprocess.run(["git", "diff", "--cached"], cwd=path, capture_output=True, text=True).stdout
    subprocess.run(["git", "reset", "-q"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "--", "."], cwd=path, check=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=path, check=True)
    return diff


class TestUploadBundle:
    def test_upload_bundle_from_agent_path(self, tmp_path):
        agent_file = tmp_path / "my_agent.py"
        agent_file.write_text("def run():\n    return {'answer': 1}\n")

        handler = CommandHandler(root=str(tmp_path))
        ok, result, err = handler.dispatch({
            "kind": "upload_bundle",
            "payload": {"agent_path": "my_agent.py", "entrypoint_fn": "run"},
        })
        assert ok, err
        bundle = result["bundle"]
        assert bundle["entry_path"] == "my_agent.py"
        assert bundle["entrypoint_fn"] == "run"
        assert "my_agent.py" in bundle["files"]
        assert "def run" in bundle["files"]["my_agent.py"]

    def test_unknown_kind_is_rejected(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        ok, result, err = handler.dispatch({"kind": "rm_rf", "payload": {}})
        assert ok is False
        assert result == {}
        assert "Unknown command kind" in err

    def test_apply_patch_empty_diff_fails(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        ok, _result, err = handler.dispatch({"kind": "apply_patch", "payload": {"diff": ""}})
        assert ok is False
        assert err == "Empty diff"

    def test_run_agent_missing_payload_reports_error(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        ok, _result, err = handler.dispatch({"kind": "run_agent", "payload": {}})
        assert ok is False
        assert "agent_path" in err


class TestApplyAndReset:
    def test_apply_then_reset_modified_file(self, tmp_path):
        _init_repo(tmp_path)

        def edit(p: Path) -> None:
            (p / "agent.py").write_text("VALUE = 2\n")

        diff = _diff_for(tmp_path, edit)
        handler = CommandHandler(root=str(tmp_path))

        ok, result, err = handler.dispatch({
            "kind": "apply_patch",
            "workflow_run_id": "run-1",
            "payload": {"diff": diff, "candidate_id": "cand-1"},
        })
        assert ok, err
        assert result == {"applied": True, "candidate_id": "cand-1"}
        assert (tmp_path / "agent.py").read_text() == "VALUE = 2\n"

        ok, result, err = handler.dispatch({
            "kind": "reset",
            "payload": {"workflow_run_id": "run-1"},
        })
        assert ok, err
        assert result == {"reset": True}
        assert (tmp_path / "agent.py").read_text() == "VALUE = 1\n"

    def test_reset_removes_patch_added_file(self, tmp_path):
        _init_repo(tmp_path)

        def edit(p: Path) -> None:
            (p / "new_module.py").write_text("X = 42\n")

        diff = _diff_for(tmp_path, edit)
        handler = CommandHandler(root=str(tmp_path))

        ok, _result, err = handler.dispatch({
            "kind": "apply_patch",
            "workflow_run_id": "run-2",
            "payload": {"diff": diff},
        })
        assert ok, err
        assert (tmp_path / "new_module.py").exists()

        ok, _result, err = handler.dispatch({
            "kind": "reset",
            "payload": {"workflow_run_id": "run-2"},
        })
        assert ok, err
        assert not (tmp_path / "new_module.py").exists()

    def test_apply_patch_invalid_diff_fails(self, tmp_path):
        _init_repo(tmp_path)
        handler = CommandHandler(root=str(tmp_path))
        ok, _result, err = handler.dispatch({
            "kind": "apply_patch",
            "workflow_run_id": "run-3",
            "payload": {"diff": "not a real diff\n"},
        })
        assert ok is False
        assert err


class TestCommandOtel:
    """Every dispatched command opens one Overmind span tagged with run correlation."""

    def test_command_opens_span_with_correlation(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        with _capture_spans() as (spans, tags):
            ok, _result, _err = handler.dispatch({
                "id": "cmd-1",
                "kind": "reset",
                "payload": {"optimize_run_id": "run-1"},
            })
        assert ok is True
        assert spans == ["overmind.command.reset"]
        assert tags[oc_attrs.COMMAND] == "reset"
        assert tags[oc_attrs.JOB_ID] == "cmd-1"
        assert tags[oc_attrs.WORKFLOW_RUN_ID] == "run-1"
        assert tags[oc_attrs.STATUS] == "success"

    def test_run_agent_span_tags_run_kind(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        with (
            _capture_spans() as (spans, tags),
            patch(
                "overmind.daemon.handlers.run_agent_from_platform",
                return_value={"results": [], "count": 0},
            ),
        ):
            handler.dispatch({
                "id": "cmd-2",
                "kind": "run_agent",
                "payload": {"subset": "baseline", "iteration": 0, "agent_id": "a-1"},
            })
        assert spans == ["overmind.command.run_agent"]
        assert tags[oc_attrs.RUN_KIND] == "baseline"
        assert tags[oc_attrs.AGENT_ID] == "a-1"
        assert tags[oc_attrs.STATUS] == "success"

    def test_failed_command_tags_error_status(self, tmp_path):
        handler = CommandHandler(root=str(tmp_path))
        with _capture_spans() as (spans, tags):
            ok, _result, _err = handler.dispatch({
                "id": "cmd-3",
                "kind": "apply_patch",
                "payload": {"diff": ""},
            })
        assert ok is False
        assert spans == ["overmind.command.apply_patch"]
        assert tags[oc_attrs.STATUS] == "failed"
        assert tags[oc_attrs.ERROR_MESSAGE] == "Empty diff"
