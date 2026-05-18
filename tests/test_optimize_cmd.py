"""Tests for overmind.commands.optimize_cmd — optimize command entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from overmind.commands.optimize_cmd import main


class TestOptimizeMain:
    @patch("overmind.commands.optimize_cmd.configure_storage")
    @patch("overmind.commands.optimize_cmd.get_agent_id")
    @patch("overmind.commands.optimize_cmd.Optimizer")
    @patch("overmind.commands.optimize_cmd.collect_config")
    def test_calls_optimizer(
        self,
        mock_config,
        mock_optimizer,
        mock_get_agent_id,
        mock_configure_storage,
    ):
        mock_cfg = MagicMock()
        mock_config.return_value = mock_cfg
        mock_get_agent_id.return_value = "agent-id-123"
        main(agent_name="test", fast=True)
        mock_config.assert_called_once_with(
            agent_name="test",
            fast=True,
            scope_globs=None,
            max_files=None,
            max_chars=None,
        )
        mock_optimizer.assert_called_once_with(mock_cfg)
        mock_optimizer.return_value.run.assert_called_once()
