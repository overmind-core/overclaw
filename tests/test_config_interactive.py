"""Tests for overclaw.optimize.config — interactive collect_config flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME


class TestCollectConfigInteractive:
    @patch("overclaw.optimize.config.Confirm")
    @patch("overclaw.optimize.config.IntPrompt")
    @patch("overclaw.optimize.config.prompt_for_catalog_litellm_model")
    @patch("overclaw.optimize.config.resolve_agent")
    @patch("overclaw.optimize.config.load_overclaw_dotenv")
    def test_basic_flow(
        self,
        mock_load_env,
        mock_resolve,
        mock_picker,
        mock_int,
        mock_confirm,
        overclaw_tmp_project,
        monkeypatch,
    ):
        agent_dir = overclaw_tmp_project / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        spec_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "test" / "setup_spec"
        )
        spec_dir.mkdir(parents=True)
        (spec_dir / "eval_spec.json").write_text(
            '{"output_fields": {}, "structure_weight": 20}'
        )
        (spec_dir / "dataset.json").write_text('[{"input": {}, "expected_output": {}}]')

        mock_resolve.return_value = (str(agent_file), "run")
        mock_picker.return_value = "openai/gpt-5.4"
        mock_int.ask.side_effect = [5, 3, 5]
        mock_confirm.ask.side_effect = [
            True,  # use env analyzer model
            False,  # don't use judge
            True,  # parallel
            False,  # no advanced
            True,  # proceed
        ]
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")

        from overclaw.optimize.config import collect_config

        cfg = collect_config(agent_name="test", fast=False)
        assert cfg.agent_name == "test"
        assert cfg.iterations == 5

    @patch("overclaw.optimize.config.Confirm")
    @patch("overclaw.optimize.config.IntPrompt")
    @patch("overclaw.optimize.config.Prompt")
    @patch("overclaw.optimize.config.prompt_for_catalog_litellm_model")
    @patch("overclaw.optimize.config.resolve_agent")
    @patch("overclaw.optimize.config.load_overclaw_dotenv")
    def test_with_advanced_settings(
        self,
        mock_load_env,
        mock_resolve,
        mock_picker,
        mock_prompt,
        mock_int,
        mock_confirm,
        overclaw_tmp_project,
        monkeypatch,
    ):
        agent_dir = overclaw_tmp_project / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        spec_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "test" / "setup_spec"
        )
        spec_dir.mkdir(parents=True)
        (spec_dir / "eval_spec.json").write_text(
            '{"output_fields": {}, "structure_weight": 20}'
        )
        (spec_dir / "dataset.json").write_text('[{"input": {}, "expected_output": {}}]')

        mock_resolve.return_value = (str(agent_file), "run")
        mock_picker.return_value = "openai/gpt-5.4"
        mock_int.ask.side_effect = [3, 2, 3, 2]
        mock_prompt.ask.side_effect = ["0.3", "0.2", "3", "0.7"]
        mock_confirm.ask.side_effect = [
            True,  # use env analyzer
            False,  # no judge
            True,  # parallel
            True,  # advanced settings
            True,  # holdout enforcement
            True,  # proceed
        ]
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")

        from overclaw.optimize.config import collect_config

        cfg = collect_config(agent_name="test", fast=False)
        assert cfg.regression_threshold == 0.3
        assert cfg.holdout_ratio == 0.2

    @patch("overclaw.optimize.config.Confirm")
    @patch("overclaw.optimize.config.IntPrompt")
    @patch("overclaw.optimize.config.prompt_for_catalog_litellm_model")
    @patch("overclaw.optimize.config.resolve_agent")
    @patch("overclaw.optimize.config.load_overclaw_dotenv")
    def test_no_spec_exits(
        self,
        mock_load_env,
        mock_resolve,
        mock_picker,
        mock_int,
        mock_confirm,
        overclaw_tmp_project,
    ):
        agent_file = overclaw_tmp_project / "agent.py"
        agent_file.write_text("def run(x): return {}\n")
        mock_resolve.return_value = (str(agent_file), "run")

        from overclaw.optimize.config import collect_config

        with pytest.raises(SystemExit):
            collect_config(agent_name="test", fast=False)

    @patch("overclaw.optimize.config.Confirm")
    @patch("overclaw.optimize.config.IntPrompt")
    @patch("overclaw.optimize.config.prompt_for_catalog_litellm_model")
    @patch("overclaw.optimize.config.resolve_agent")
    @patch("overclaw.optimize.config.load_overclaw_dotenv")
    def test_user_aborts(
        self,
        mock_load_env,
        mock_resolve,
        mock_picker,
        mock_int,
        mock_confirm,
        overclaw_tmp_project,
        monkeypatch,
    ):
        agent_dir = overclaw_tmp_project / "agents" / "test"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "sample.py"
        agent_file.write_text("def run(x): return {}\n")

        spec_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "test" / "setup_spec"
        )
        spec_dir.mkdir(parents=True)
        (spec_dir / "eval_spec.json").write_text(
            '{"output_fields": {}, "structure_weight": 20}'
        )
        (spec_dir / "dataset.json").write_text('[{"input": {}, "expected_output": {}}]')

        mock_resolve.return_value = (str(agent_file), "run")
        mock_picker.return_value = "openai/gpt-5.4"
        mock_int.ask.side_effect = [5, 3, 5]
        mock_confirm.ask.side_effect = [
            True,  # use env analyzer
            False,  # no judge
            True,  # parallel
            False,  # no advanced
            False,  # DO NOT proceed
        ]
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")

        from overclaw.optimize.config import collect_config

        with pytest.raises(SystemExit):
            collect_config(agent_name="test", fast=False)
