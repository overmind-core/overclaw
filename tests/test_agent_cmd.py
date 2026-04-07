"""Tests for overclaw.commands.agent_cmd — agent management commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.commands.agent_cmd import (
    _confirm_duplicate_entrypoint,
    _other_agents_with_entrypoint,
    cmd_list,
    cmd_register,
    cmd_remove,
    cmd_show,
    cmd_update,
)


# ---------------------------------------------------------------------------
# _other_agents_with_entrypoint
# ---------------------------------------------------------------------------


class TestOtherAgentsWithEntrypoint:
    def test_finds_duplicates(self):
        registry = {
            "a": {"entrypoint": "m:f"},
            "b": {"entrypoint": "m:f"},
            "c": {"entrypoint": "m:g"},
        }
        result = _other_agents_with_entrypoint(registry, "m:f")
        assert result == ["a", "b"]

    def test_excludes_name(self):
        registry = {
            "a": {"entrypoint": "m:f"},
            "b": {"entrypoint": "m:f"},
        }
        result = _other_agents_with_entrypoint(registry, "m:f", exclude_name="a")
        assert result == ["b"]

    def test_no_duplicates(self):
        registry = {"a": {"entrypoint": "m:f"}}
        assert _other_agents_with_entrypoint(registry, "m:g") == []

    def test_whitespace_handled(self):
        registry = {"a": {"entrypoint": "  m:f  "}}
        assert _other_agents_with_entrypoint(registry, "m:f") == ["a"]

    def test_empty_registry(self):
        assert _other_agents_with_entrypoint({}, "m:f") == []


# ---------------------------------------------------------------------------
# _confirm_duplicate_entrypoint
# ---------------------------------------------------------------------------


class TestConfirmDuplicateEntrypoint:
    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=True)
    def test_user_confirms(self, _mock_confirm):
        console = MagicMock()
        _confirm_duplicate_entrypoint(console, "m:f", ["existing"])

    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=False)
    def test_user_aborts(self, _mock_confirm):
        console = MagicMock()
        with pytest.raises(SystemExit) as exc_info:
            _confirm_duplicate_entrypoint(console, "m:f", ["existing"])
        assert exc_info.value.code == 0

    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=True)
    def test_for_update_prompt_text(self, _mock_confirm):
        console = MagicMock()
        _confirm_duplicate_entrypoint(console, "m:f", ["existing"], for_update=True)


# ---------------------------------------------------------------------------
# cmd_register
# ---------------------------------------------------------------------------


class TestCmdRegister:
    def test_register_new_agent(self, tmp_project):
        # Use a different entrypoint than the already-registered one to avoid
        # the duplicate-entrypoint interactive prompt.
        cmd_register("new-agent", "agents.agent1.sample_agent:helper")
        from overclaw.core.registry import load_registry

        registry = load_registry()
        assert "new-agent" in registry

    def test_register_already_exists_different_entrypoint(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_register("my-agent", "agents.agent1.sample_agent:helper")
        assert exc_info.value.code == 1

    def test_register_same_entrypoint_idempotent(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_register("my-agent", "agents.agent1.sample_agent:run")
        assert exc_info.value.code == 0

    def test_register_invalid_entrypoint(self, tmp_project):
        with pytest.raises(SystemExit):
            cmd_register("test", "nonexistent.module:func")

    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=True)
    def test_register_duplicate_entrypoint_confirmed(self, _mock_confirm, tmp_project):
        cmd_register("second-agent", "agents.agent1.sample_agent:run")
        from overclaw.core.registry import load_registry

        assert "second-agent" in load_registry()


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    def test_list_with_agents(self, tmp_project, capsys):
        cmd_list()

    def test_list_empty_registry(self, tmp_project_empty, capsys):
        cmd_list()


# ---------------------------------------------------------------------------
# cmd_remove
# ---------------------------------------------------------------------------


class TestCmdRemove:
    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=True)
    def test_remove_existing(self, _mock_confirm, tmp_project):
        cmd_remove("my-agent")
        from overclaw.core.registry import load_registry

        assert "my-agent" not in load_registry()

    @patch("overclaw.commands.agent_cmd.confirm_option", return_value=False)
    def test_remove_aborted(self, _mock_confirm, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_remove("my-agent")
        assert exc_info.value.code == 0

    def test_remove_nonexistent(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_remove("nonexistent")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# cmd_update
# ---------------------------------------------------------------------------


class TestCmdUpdate:
    def test_update_existing(self, tmp_project):
        cmd_update("my-agent", "agents.agent1.sample_agent:helper")
        from overclaw.core.registry import load_registry

        registry = load_registry()
        assert registry["my-agent"]["entrypoint"] == "agents.agent1.sample_agent:helper"

    def test_update_nonexistent(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_update("nonexistent", "agents.agent1.sample_agent:run")
        assert exc_info.value.code == 1

    def test_update_invalid_entrypoint(self, tmp_project):
        with pytest.raises(SystemExit):
            cmd_update("my-agent", "gone.module:func")

    def test_update_same_entrypoint_no_op(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_update("my-agent", "agents.agent1.sample_agent:run")
        assert exc_info.value.code == 0
        from overclaw.core.registry import load_registry

        assert (
            load_registry()["my-agent"]["entrypoint"]
            == "agents.agent1.sample_agent:run"
        )


# ---------------------------------------------------------------------------
# cmd_show
# ---------------------------------------------------------------------------


class TestCmdShow:
    def test_show_existing(self, tmp_project, capsys):
        cmd_show("my-agent")

    def test_show_nonexistent(self, tmp_project):
        with pytest.raises(SystemExit) as exc_info:
            cmd_show("nonexistent")
        assert exc_info.value.code == 1

    def test_show_with_setup_spec(self, tmp_project):
        spec_dir = (
            tmp_project / OVERCLAW_DIR_NAME / "agents" / "my-agent" / "setup_spec"
        )
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "eval_spec.json").write_text("{}")
        cmd_show("my-agent")

    def test_show_with_experiments(self, tmp_project):
        exp_dir = (
            tmp_project / OVERCLAW_DIR_NAME / "agents" / "my-agent" / "experiments"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "result.json").write_text("{}")
        cmd_show("my-agent")
