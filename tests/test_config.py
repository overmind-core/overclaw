"""Tests for overclaw.optimize.config — Config dataclass and helpers."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.paths import agent_experiments_dir, agent_setup_spec_dir
from overclaw.optimize.config import (
    Config,
    _clear_existing_experiments,
    _require_analyzer_model_env_fast,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = Config(agent_name="test", agent_path="/path", entrypoint_fn="run")
        assert cfg.iterations == 5
        assert cfg.candidates_per_iteration == 3
        assert cfg.parallel is True
        assert cfg.max_workers == 5
        assert cfg.regression_threshold == 0.35
        assert cfg.holdout_ratio == 0.2
        assert cfg.early_stopping_patience == 3
        assert cfg.holdout_enforcement is True
        assert cfg.overfit_gap_threshold == 10.0
        assert cfg.holdout_weight == 0.3
        assert cfg.catastrophic_holdout_threshold == 0.5
        assert cfg.max_code_growth_ratio == 2.5

    def test_custom_values(self):
        cfg = Config(
            agent_name="x",
            agent_path="/p",
            entrypoint_fn="run",
            iterations=10,
            parallel=False,
        )
        assert cfg.iterations == 10
        assert cfg.parallel is False

    def test_all_fields_have_defaults_except_required(self):
        required = {"agent_name", "agent_path", "entrypoint_fn"}
        for f in fields(Config):
            if f.name in required:
                continue
            assert (
                f.default is not f.default_factory
                if hasattr(f, "default_factory")
                else True
            )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_experiments_dir(self, overclaw_tmp_project: Path):
        result = agent_experiments_dir("agent1")
        assert result.name == "experiments"
        assert result.parent.name == "agent1"
        assert OVERCLAW_DIR_NAME in str(result)

    def test_eval_spec_file_path(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("agent1") / "eval_spec.json"
        assert str(result).endswith("eval_spec.json")
        assert "setup_spec" in str(result)
        assert OVERCLAW_DIR_NAME in str(result)

    def test_dataset_file_path(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("agent1") / "dataset.json"
        assert str(result).endswith("dataset.json")
        assert "setup_spec" in str(result)


# ---------------------------------------------------------------------------
# _clear_existing_experiments
# ---------------------------------------------------------------------------


class TestClearExistingExperiments:
    def test_no_experiments_dir(self, overclaw_tmp_project: Path):
        console = MagicMock()
        _clear_existing_experiments("x", console)

    def test_empty_experiments_dir(self, overclaw_tmp_project: Path):
        exp_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        console = MagicMock()
        _clear_existing_experiments("x", console)

    def test_fast_mode_clears(self, overclaw_tmp_project: Path):
        exp_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console, fast=True)
        assert exp_dir.exists()
        assert not list(exp_dir.iterdir())  # cleaned

    @patch("overclaw.optimize.config.Confirm")
    def test_interactive_user_confirms(self, mock_confirm, overclaw_tmp_project: Path):
        mock_confirm.ask.return_value = True
        exp_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console)
        assert not list(exp_dir.iterdir())

    @patch("overclaw.optimize.config.Confirm")
    def test_interactive_user_declines(self, mock_confirm, overclaw_tmp_project: Path):
        mock_confirm.ask.return_value = False
        exp_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console)
        assert (exp_dir / "result.json").exists()  # kept


# ---------------------------------------------------------------------------
# _require_analyzer_model_env_fast
# ---------------------------------------------------------------------------


class TestRequireAnalyzerModelEnvFast:
    def test_returns_model_when_set(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        console = MagicMock()
        result = _require_analyzer_model_env_fast(console)
        assert "gpt-5.4" in result

    def test_exits_when_not_set(self, monkeypatch):
        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        console = MagicMock()
        with pytest.raises(SystemExit) as exc_info:
            _require_analyzer_model_env_fast(console)
        assert exc_info.value.code == 1

    def test_exits_when_empty(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "  ")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _require_analyzer_model_env_fast(console)
