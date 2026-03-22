"""Tests for overclaw.core.model_picker — interactive model selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from overclaw.core.model_picker import prompt_for_catalog_litellm_model


class TestPromptForCatalogLitellmModel:
    @patch("overclaw.core.model_picker.Prompt")
    def test_picks_model(self, mock_prompt):
        mock_prompt.ask.return_value = "1"
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console, select_prompt="Pick", env_default=None
        )
        assert "/" in result

    @patch("overclaw.core.model_picker.Prompt")
    def test_with_env_default(self, mock_prompt):
        mock_prompt.ask.return_value = "1"
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console,
            select_prompt="Pick",
            env_default="openai/gpt-5.4",
        )
        assert result

    @patch("overclaw.core.model_picker.get_litellm_model_ids", return_value=[])
    @patch("overclaw.core.model_picker.Prompt")
    def test_empty_catalog_fallback(self, mock_prompt, mock_ids):
        mock_prompt.ask.return_value = "custom/model"
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console, select_prompt="Pick", no_catalog_prompt="Enter model"
        )
        assert result == "custom/model"
