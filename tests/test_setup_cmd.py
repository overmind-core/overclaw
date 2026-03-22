"""Tests for overclaw.commands.setup_cmd — setup command helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.paths import agent_setup_spec_dir
from overclaw.commands.setup_cmd import (
    _build_eval_spec_stub,
    _clear_existing_eval_spec,
    _data_dir,
    _display_proposed_criteria,
    _resolve_datagen_model,
    _save_and_finish,
    _save_dataset,
    _validate_agent_entrypoint,
)


class TestValidateAgentEntrypoint:
    def test_valid(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x):\n    pass\n")
        console = MagicMock()
        _validate_agent_entrypoint(str(agent), "run", console)

    def test_missing_function(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def other(x):\n    pass\n")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _validate_agent_entrypoint(str(agent), "run", console)


class TestPathHelpers:
    def test_agent_setup_spec_dir(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1")
        assert str(result).endswith("setup_spec")
        assert OVERCLAW_DIR_NAME in str(result)

    def test_eval_spec_under_setup_spec(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1") / "eval_spec.json"
        assert str(result).endswith("eval_spec.json")

    def test_dataset_under_setup_spec(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1") / "dataset.json"
        assert str(result).endswith("dataset.json")

    def test_data_dir(self):
        result = _data_dir("/project/agents/a1/agent.py")
        assert result.name == "data"


class TestClearExistingEvalSpec:
    def test_no_dir(self, overclaw_tmp_project: Path):
        console = MagicMock()
        _clear_existing_eval_spec("nope", console)

    def test_empty_dir(self, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        console = MagicMock()
        _clear_existing_eval_spec("x", console)

    def test_fast_clears(self, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console, fast=True)
        assert not (spec_dir / "spec.json").exists()

    @patch("overclaw.commands.setup_cmd.Confirm")
    def test_interactive_confirm(self, mock_confirm, overclaw_tmp_project: Path):
        mock_confirm.ask.return_value = True
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console)

    @patch("overclaw.commands.setup_cmd.Confirm")
    def test_interactive_decline(self, mock_confirm, overclaw_tmp_project: Path):
        mock_confirm.ask.return_value = False
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console)
        assert (spec_dir / "spec.json").exists()


class TestBuildEvalSpecStub:
    def test_basic(self):
        analysis = {
            "output_schema": {
                "status": {"type": "enum", "values": ["a", "b"]},
                "score": {"type": "number", "range": [0, 100]},
            },
            "input_schema": {"name": {"type": "string"}},
            "description": "Test",
        }
        stub = _build_eval_spec_stub(analysis)
        assert "status" in stub["output_fields"]
        assert stub["output_fields"]["status"]["weight"] == 10

    def test_with_policy(self):
        analysis = {"output_schema": {}}
        policy = {"purpose": "test"}
        stub = _build_eval_spec_stub(analysis, policy)
        assert stub["policy"]["purpose"] == "test"


class TestSaveAndFinish:
    def test_saves_spec(self, overclaw_tmp_project: Path):
        spec = {"output_fields": {"x": {"weight": 10}}, "total_points": 100}
        console = MagicMock()
        _save_and_finish(spec, "myagent", console)
        spec_path = agent_setup_spec_dir("myagent") / "eval_spec.json"
        assert spec_path.exists()

    def test_saves_policy(self, overclaw_tmp_project: Path):
        spec = {"output_fields": {}, "policy": {"domain_rules": ["r1"]}}
        console = MagicMock()
        _save_and_finish(spec, "myagent", console, policy_md="# Policy")
        from overclaw.core.policy import default_policy_path

        assert Path(default_policy_path("myagent")).exists()


class TestSaveDataset:
    def test_saves(self, overclaw_tmp_project: Path):
        console = MagicMock()
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        path = _save_dataset(cases, "dsagent", console)
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text())
        assert len(loaded) == 1


class TestResolveDategenModel:
    def test_fast_with_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        console = MagicMock()
        result = _resolve_datagen_model(console, fast=True)
        assert "gpt-5.4" in result

    def test_fast_without_env(self, monkeypatch):
        monkeypatch.delenv("SYNTHETIC_DATAGEN_MODEL", raising=False)
        console = MagicMock()
        with pytest.raises(SystemExit):
            _resolve_datagen_model(console, fast=True)

    @patch("overclaw.commands.setup_cmd.Confirm")
    def test_interactive_with_env(self, mock_confirm, monkeypatch):
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        mock_confirm.ask.return_value = True
        console = MagicMock()
        result = _resolve_datagen_model(console, fast=False)
        assert "gpt-5.4" in result


class TestDisplayProposedCriteria:
    def test_with_criteria(self):
        analysis = {
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "status": {"importance": "critical", "partial_credit": True},
                    "score": {"importance": "important", "tolerance": 10},
                    "reason": {"importance": "minor", "eval_mode": "non_empty"},
                },
            },
            "output_schema": {
                "status": {"type": "enum"},
                "score": {"type": "number"},
                "reason": {"type": "text"},
            },
        }
        console = MagicMock()
        _display_proposed_criteria(analysis, console)

    def test_no_criteria(self):
        console = MagicMock()
        _display_proposed_criteria({}, console)
