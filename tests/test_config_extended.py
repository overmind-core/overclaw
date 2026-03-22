"""Extended tests for overclaw.optimize.config — interactive config collection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.paths import agent_setup_spec_dir
from overclaw.optimize.config import (
    _analyzer_default_from_env,
    _collect_config_fast,
    _select_backtest_models,
)


class TestAnalyzerDefaultFromEnv:
    def test_with_known_model(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        result = _analyzer_default_from_env()
        assert "gpt-5.4" in result

    def test_empty(self, monkeypatch):
        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        assert _analyzer_default_from_env() is None

    def test_unknown_model(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "custom-model")
        result = _analyzer_default_from_env()
        assert result == "custom-model"


class TestDatasetPathResolution:
    def test_returns_dataset_json(self, overclaw_tmp_project):
        result = str(agent_setup_spec_dir("a1") / "dataset.json")
        assert result.endswith("dataset.json")
        assert OVERCLAW_DIR_NAME in result


class TestSelectBacktestModels:
    @patch("overclaw.optimize.config.Prompt")
    def test_select_none(self, mock_prompt):
        mock_prompt.ask.return_value = "none"
        console = MagicMock()
        result = _select_backtest_models(console)
        assert result == []

    @patch("overclaw.optimize.config.Prompt")
    def test_select_all(self, mock_prompt):
        mock_prompt.ask.return_value = "all"
        console = MagicMock()
        result = _select_backtest_models(console)
        assert len(result) > 0

    @patch("overclaw.optimize.config.Prompt")
    def test_select_specific(self, mock_prompt):
        mock_prompt.ask.return_value = "1"
        console = MagicMock()
        result = _select_backtest_models(console)
        assert len(result) >= 1


class TestCollectConfigFast:
    def test_missing_spec_exits(self, tmp_project, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        with pytest.raises(SystemExit):
            _collect_config_fast("my-agent", MagicMock())

    def test_success(self, tmp_project, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        spec_dir = (
            tmp_project / OVERCLAW_DIR_NAME / "agents" / "my-agent" / "setup_spec"
        )
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "eval_spec.json").write_text(
            '{"output_fields": {}, "structure_weight": 20, "total_points": 100}'
        )
        (spec_dir / "dataset.json").write_text('[{"input": {}, "expected_output": {}}]')
        cfg = _collect_config_fast("my-agent", MagicMock())
        assert cfg.agent_name == "my-agent"
        assert "gpt-5.4" in cfg.analyzer_model

    def test_missing_dataset_exits(self, tmp_project, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        spec_dir = (
            tmp_project / OVERCLAW_DIR_NAME / "agents" / "my-agent" / "setup_spec"
        )
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "eval_spec.json").write_text(
            '{"output_fields": {}, "structure_weight": 20, "total_points": 100}'
        )
        with pytest.raises(SystemExit):
            _collect_config_fast("my-agent", MagicMock())
