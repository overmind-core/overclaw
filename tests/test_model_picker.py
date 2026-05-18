"""Tests for overmind.utils.model_picker — interactive model selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from overmind.utils.model_picker import prompt_for_catalog_litellm_model


class TestPromptForCatalogLitellmModel:
    @patch("overmind.utils.model_picker.list_models_for_provider", return_value=None)
    @patch("overmind.utils.model_picker.select_option", return_value=0)
    def test_picks_model(self, _mock_select, _mock_live):
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console, select_prompt="Pick", env_default=None
        )
        assert "/" in result

    @patch("overmind.utils.model_picker.list_models_for_provider", return_value=None)
    @patch("overmind.utils.model_picker.select_option", return_value=0)
    def test_with_env_default(self, _mock_select, _mock_live):
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console,
            select_prompt="Pick",
            env_default="openai/gpt-5.4",
        )
        assert result

    @patch("overmind.utils.model_picker.get_litellm_model_ids", return_value=[])
    @patch("overmind.utils.model_picker.Prompt")
    def test_empty_catalog_fallback(self, mock_prompt, _mock_ids):
        mock_prompt.ask.return_value = "custom/model"
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console, select_prompt="Pick", no_catalog_prompt="Enter model"
        )
        assert result == "custom/model"

    @patch(
        "overmind.utils.model_picker.list_models_for_provider",
        return_value=["gpt-5.4-2026-03-05", "gpt-5.4", "gpt-5"],
    )
    @patch("overmind.utils.model_picker.select_option", return_value=0)
    def test_uses_live_listing_when_discovery_succeeds(self, _mock_select, mock_live):
        """When discovery returns a non-empty list, the picker should show it."""
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console,
            select_prompt="Pick",
            env_default=None,
            env={"OPENAI_API_KEY": "sk-test"},
        )
        mock_live.assert_called_once()
        # First provider in get_providers() is openai, and we returned a live
        # list whose first entry is gpt-5.4-2026-03-05 (select_option returns 0).
        assert result == "openai/gpt-5.4-2026-03-05"

    @patch("overmind.utils.model_picker.list_models_for_provider", return_value=[])
    @patch("overmind.utils.model_picker.get_models_for_provider", return_value=["gpt-5"])
    @patch("overmind.utils.model_picker.select_option", return_value=0)
    def test_falls_back_to_static_catalog_when_discovery_returns_empty(
        self, _mock_select, _mock_static, _mock_live
    ):
        console = MagicMock()
        result = prompt_for_catalog_litellm_model(
            console,
            select_prompt="Pick",
            env_default=None,
        )
        assert result == "openai/gpt-5"
