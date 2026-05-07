"""Tests for overmind.commands.setup_cmd.main — the full setup wizard."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import overmind
from overmind.core.constants import OVERMIND_DIR_NAME


class TestSetupCmdMainFast:
    @patch("overmind.commands.setup_cmd.generate_synthetic_data")
    @patch("overmind.commands.setup_cmd.generate_policy_from_code")
    @patch("overmind.commands.setup_cmd.analyze_agent")
    @patch("overmind.commands.setup_cmd.resolve_agent")
    @patch("overmind.commands.setup_cmd.load_overmind_dotenv")
    def test_fast_mode_no_seed(
        self,
        mock_load_env,
        mock_resolve,
        mock_analyze,
        mock_policy,
        mock_gen,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        agent_dir = tmp_path / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        mock_resolve.return_value = (str(agent_file), "run")
        mock_analyze.return_value = {
            "description": "Test agent",
            "output_schema": {"status": {"type": "enum", "values": ["a", "b"]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"status": {"importance": "critical"}},
            },
        }
        mock_policy.return_value = (
            "# Policy",
            {"purpose": "test", "domain_rules": ["r1"]},
        )
        mock_gen.return_value = [
            {"input": {"x": 1}, "expected_output": {"status": "a"}}
        ]

        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        monkeypatch.chdir(tmp_path)
        overmind.init()

        from overmind.commands.setup_cmd import main

        main(agent_name="test", fast=True)

        spec_path = (
            tmp_path
            / OVERMIND_DIR_NAME
            / "agents"
            / "test"
            / "setup_spec"
            / "eval_spec.json"
        )
        assert spec_path.exists()

    @patch("overmind.commands.setup_cmd.generate_synthetic_data")
    @patch("overmind.commands.setup_cmd.improve_existing_policy")
    @patch("overmind.commands.setup_cmd.analyze_agent")
    @patch("overmind.commands.setup_cmd.resolve_agent")
    @patch("overmind.commands.setup_cmd.load_overmind_dotenv")
    def test_fast_mode_with_policy(
        self,
        mock_load_env,
        mock_resolve,
        mock_analyze,
        mock_improve,
        mock_gen,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        agent_dir = tmp_path / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        policy_file = tmp_path / "policy.md"
        policy_file.write_text("# My Policy\nRule 1\n")

        mock_resolve.return_value = (str(agent_file), "run")
        mock_analyze.return_value = {
            "description": "Test",
            "output_schema": {},
            "proposed_criteria": {"structure_weight": 20, "fields": {}},
        }
        mock_improve.return_value = (
            "# Improved",
            {"purpose": "improved", "domain_rules": []},
            "Added things",
        )
        mock_gen.return_value = [{"input": {}, "expected_output": {}}]

        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        monkeypatch.chdir(tmp_path)

        from overmind.commands.setup_cmd import main

        main(agent_name="test", fast=True, policy=str(policy_file))

    @patch("overmind.commands.setup_cmd.resolve_agent")
    @patch("overmind.commands.setup_cmd.load_overmind_dotenv")
    def test_fast_no_analyzer_model(
        self, mock_load_env, mock_resolve, tmp_path, monkeypatch
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        agent_file = tmp_path / "agent.py"
        agent_file.write_text("def run(x): return {}\n")
        mock_resolve.return_value = (str(agent_file), "run")

        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)
        overmind.init()

        from overmind.commands.setup_cmd import main

        with pytest.raises(SystemExit):
            main(agent_name="test", fast=True)

    @patch("overmind.commands.setup_cmd.resolve_agent")
    @patch("overmind.commands.setup_cmd.load_overmind_dotenv")
    def test_fast_no_datagen_model(
        self, mock_load_env, mock_resolve, tmp_path, monkeypatch
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        agent_file = tmp_path / "agent.py"
        agent_file.write_text("def run(x): return {}\n")
        mock_resolve.return_value = (str(agent_file), "run")

        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        monkeypatch.delenv("SYNTHETIC_DATAGEN_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)
        overmind.init()

        from overmind.commands.setup_cmd import main

        with pytest.raises(SystemExit):
            main(agent_name="test", fast=True)


class TestSetupCmdMainInteractive:
    @patch("overmind.commands.setup_cmd._collect_agent_provider_config")
    @patch("overmind.commands.setup_cmd.run_questionnaire")
    @patch("overmind.commands.setup_cmd.confirm_option")
    @patch("overmind.commands.setup_cmd.select_option")
    @patch("overmind.commands.setup_cmd._run_data_phase")
    @patch("overmind.commands.setup_cmd.generate_policy_from_code")
    @patch("overmind.commands.setup_cmd.analyze_agent")
    @patch("overmind.commands.setup_cmd.prompt_for_catalog_litellm_model")
    @patch("overmind.commands.setup_cmd.resolve_agent")
    @patch("overmind.commands.setup_cmd.load_overmind_dotenv")
    @patch(
        "overmind.utils.provider_keys.read_api_key_masked",
        return_value="sk-ant-api03-test-placeholder",
    )
    def test_interactive_auto_policy(
        self,
        _mock_provider_read_api_key,
        mock_load_env,
        mock_resolve,
        mock_picker,
        mock_analyze,
        mock_policy,
        mock_data_phase,
        mock_select,
        mock_confirm,
        mock_questionnaire,
        _mock_skip_provider,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        agent_dir = tmp_path / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        mock_resolve.return_value = (str(agent_file), "run")
        mock_picker.return_value = "openai/gpt-5.4"
        mock_analyze.return_value = {
            "description": "Test",
            "output_schema": {},
            "proposed_criteria": {"structure_weight": 20, "fields": {}},
        }
        mock_policy.return_value = ("# Policy", {"purpose": "test", "domain_rules": []})

        # select_option calls: policy mode pick (index 1 = auto-generate from code).
        mock_select.side_effect = [1]

        # First confirm is seed-data (--data); decline so setup continues. Later: policy & criteria.
        def _confirm_side_effect(prompt: str, **kwargs):
            pl = prompt.lower()
            if "seed data" in pl and "--data" in prompt:
                return False
            if "reconfigure" in pl:
                return False
            return True

        mock_confirm.side_effect = _confirm_side_effect

        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)

        from overmind.commands.setup_cmd import main

        main(agent_name="test", fast=False)

        spec_path = (
            tmp_path
            / OVERMIND_DIR_NAME
            / "agents"
            / "test"
            / "setup_spec"
            / "eval_spec.json"
        )
        assert spec_path.exists()
