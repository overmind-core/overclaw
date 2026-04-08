"""Tests for overclaw.commands.optimize_cmd — optimize command entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from overclaw.commands.optimize_cmd import main


class TestOptimizeMain:
    @patch("overclaw.commands.optimize_cmd.Optimizer")
    @patch("overclaw.commands.optimize_cmd.collect_config")
    @patch("overclaw.commands.optimize_cmd.load_agent_dotenv")
    def test_calls_optimizer(self, mock_load_dotenv, mock_config, mock_optimizer):
        mock_cfg = MagicMock()
        mock_config.return_value = mock_cfg
        main(agent_name="test", fast=True)
        mock_load_dotenv.assert_called_once_with("test")
        mock_config.assert_called_once_with(agent_name="test", fast=True)
        mock_optimizer.assert_called_once_with(mock_cfg)
        mock_optimizer.return_value.run.assert_called_once()
