"""Tests for overclaw.cli — parser construction, command routing, help output."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overclaw.cli import _build_parser, main


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_returns_argument_parser(self):
        parser = _build_parser()
        assert parser.prog == "overclaw"

    def test_top_level_help_contains_workflow(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        out = capsys.readouterr().out
        assert "overclaw" in out.lower()

    def test_no_command_raises(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    # ── init ──────────────────────────────────────────────────────────────

    def test_init_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_init_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["init", "--help"])
        out = capsys.readouterr().out
        assert "api" in out.lower() or "keys" in out.lower() or ".env" in out.lower()

    # ── agent register ────────────────────────────────────────────────────

    def test_agent_register_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["agent", "register", "foo", "mod.sub:run"])
        assert args.command == "agent"
        assert args.agent_command == "register"
        assert args.name == "foo"
        assert args.entrypoint == "mod.sub:run"

    def test_agent_register_missing_args(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "register"])

    def test_agent_register_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "register", "--help"])
        out = capsys.readouterr().out
        assert "entrypoint" in out.lower() or "module" in out.lower()

    # ── agent list ────────────────────────────────────────────────────────

    def test_agent_list_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["agent", "list"])
        assert args.agent_command == "list"

    def test_agent_list_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "list", "--help"])
        out = capsys.readouterr().out
        assert "list" in out.lower()

    # ── agent remove ──────────────────────────────────────────────────────

    def test_agent_remove_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["agent", "remove", "bar"])
        assert args.agent_command == "remove"
        assert args.name == "bar"

    def test_agent_remove_missing_name(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "remove"])

    # ── agent update ──────────────────────────────────────────────────────

    def test_agent_update_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["agent", "update", "baz", "m:f"])
        assert args.agent_command == "update"
        assert args.name == "baz"
        assert args.entrypoint == "m:f"

    def test_agent_update_missing_entrypoint(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "update", "baz"])

    # ── agent show ────────────────────────────────────────────────────────

    def test_agent_show_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["agent", "show", "qux"])
        assert args.agent_command == "show"
        assert args.name == "qux"

    def test_agent_show_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent", "show", "--help"])
        out = capsys.readouterr().out
        assert "show" in out.lower()

    # ── agent with no subcommand ──────────────────────────────────────────

    def test_agent_no_subcommand(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent"])

    # ── setup ─────────────────────────────────────────────────────────────

    def test_setup_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["setup", "my-agent"])
        assert args.command == "setup"
        assert args.agent == "my-agent"
        assert args.fast is False
        assert args.policy is None

    def test_setup_with_fast(self):
        parser = _build_parser()
        args = parser.parse_args(["setup", "my-agent", "--fast"])
        assert args.fast is True

    def test_setup_with_policy(self):
        parser = _build_parser()
        args = parser.parse_args(["setup", "my-agent", "--policy", "/path/to/doc.md"])
        assert args.policy == "/path/to/doc.md"

    def test_setup_missing_agent(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["setup"])

    def test_setup_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["setup", "--help"])
        out = capsys.readouterr().out
        assert "setup" in out.lower()
        assert "--fast" in out

    # ── optimize ──────────────────────────────────────────────────────────

    def test_optimize_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["optimize", "my-agent"])
        assert args.command == "optimize"
        assert args.agent == "my-agent"
        assert args.fast is False

    def test_optimize_with_fast(self):
        parser = _build_parser()
        args = parser.parse_args(["optimize", "my-agent", "--fast"])
        assert args.fast is True

    def test_optimize_missing_agent(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["optimize"])

    def test_optimize_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["optimize", "--help"])
        out = capsys.readouterr().out
        assert "optimize" in out.lower()

    # ── sync ──────────────────────────────────────────────────────────────

    def test_sync_parsed_all(self):
        parser = _build_parser()
        args = parser.parse_args(["sync"])
        assert args.command == "sync"
        assert args.agent is None

    def test_sync_parsed_single_agent(self):
        parser = _build_parser()
        args = parser.parse_args(["sync", "my-agent"])
        assert args.command == "sync"
        assert args.agent == "my-agent"

    def test_sync_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["sync", "--help"])
        out = capsys.readouterr().out
        assert "sync" in out.lower()

    def test_sync_optimize_parsed_all(self):
        parser = _build_parser()
        args = parser.parse_args(["sync-optimize"])
        assert args.command == "sync-optimize"
        assert args.agent is None

    def test_sync_optimize_parsed_single_agent(self):
        parser = _build_parser()
        args = parser.parse_args(["sync-optimize", "my-agent"])
        assert args.command == "sync-optimize"
        assert args.agent == "my-agent"

    def test_sync_optimize_help(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["sync-optimize", "--help"])
        out = capsys.readouterr().out
        assert "sync" in out.lower()


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    @pytest.fixture(autouse=True)
    def _stub_require_overclaw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "overclaw.core.registry.require_overclaw_initialized",
            lambda: None,
        )

    def test_init_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "init"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.init_cmd.main") as mock_init:
                main()
                mock_init.assert_called_once()

    def test_agent_register_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "register"
            mock_args.name = "test"
            mock_args.entrypoint = "m:f"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.agent_cmd.cmd_register") as mock_fn:
                main()
                mock_fn.assert_called_once_with("test", "m:f")

    def test_agent_list_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "list"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.agent_cmd.cmd_list") as mock_fn:
                main()
                mock_fn.assert_called_once()

    def test_agent_remove_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "remove"
            mock_args.name = "test"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.agent_cmd.cmd_remove") as mock_fn:
                main()
                mock_fn.assert_called_once_with("test")

    def test_agent_update_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "update"
            mock_args.name = "test"
            mock_args.entrypoint = "m:f"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.agent_cmd.cmd_update") as mock_fn:
                main()
                mock_fn.assert_called_once_with("test", "m:f")

    def test_agent_show_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "show"
            mock_args.name = "test"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.agent_cmd.cmd_show") as mock_fn:
                main()
                mock_fn.assert_called_once_with("test")

    def test_setup_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "setup"
            mock_args.agent = "my-agent"
            mock_args.fast = True
            mock_args.policy = None
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.setup_cmd.main") as mock_fn:
                main()
                mock_fn.assert_called_once_with(
                    agent_name="my-agent", fast=True, policy=None
                )

    def test_optimize_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "optimize"
            mock_args.agent = "my-agent"
            mock_args.fast = False
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.optimize_cmd.main") as mock_fn:
                main()
                mock_fn.assert_called_once_with(agent_name="my-agent", fast=False)

    def test_sync_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "sync"
            mock_args.agent = "my-agent"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.sync_cmd.main") as mock_fn:
                main()
                mock_fn.assert_called_once_with(agent_name="my-agent")

    def test_sync_optimize_dispatches(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "sync-optimize"
            mock_args.agent = "my-agent"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.sync_optimize_cmd.main") as mock_fn:
                main()
                mock_fn.assert_called_once_with(agent_name="my-agent")

    def test_keyboard_interrupt_exits_130(self):
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "init"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch(
                "overclaw.commands.init_cmd.main", side_effect=KeyboardInterrupt
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 130


# ---------------------------------------------------------------------------
# .overclaw gate (CLI)
# ---------------------------------------------------------------------------


class TestMainRequiresOverclawDir:
    def test_agent_list_exits_when_overclaw_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "agent"
            mock_args.agent_command = "list"
            mock_parser.return_value.parse_args.return_value = mock_args
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_init_runs_when_overclaw_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("overclaw.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "init"
            mock_parser.return_value.parse_args.return_value = mock_args
            with patch("overclaw.commands.init_cmd.main") as mock_init:
                main()
                mock_init.assert_called_once()
