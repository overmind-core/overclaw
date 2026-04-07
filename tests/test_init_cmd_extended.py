"""Extended tests for overclaw.commands.init_cmd — interactive wizard flows."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from overclaw.commands.init_cmd import (
    _collect_anthropic,
    _collect_openai,
    _prompt_optional_api_key,
)
from overclaw.utils.io import read_api_key_masked


class TestPromptOptionalApiKey:
    @patch("overclaw.commands.init_cmd.read_api_key_masked", return_value="sk-test")
    def test_saves_key(self, mock_read):
        console = MagicMock()
        env: dict[str, str] = {}
        _prompt_optional_api_key(console, label="Test", env_key="TEST_KEY", env=env)
        assert env["TEST_KEY"] == "sk-test"

    @patch("overclaw.commands.init_cmd.read_api_key_masked", return_value="")
    def test_empty_key_clears(self, mock_read):
        console = MagicMock()
        env: dict[str, str] = {"TEST_KEY": "old"}
        _prompt_optional_api_key(console, label="Test", env_key="TEST_KEY", env=env)
        assert env["TEST_KEY"] == ""


class TestCollectOpenai:
    def test_skips_when_configured(self):
        console = MagicMock()
        env = {"OPENAI_API_KEY": "sk-real-key"}
        _collect_openai(console, env)

    @patch("overclaw.commands.init_cmd._prompt_optional_api_key")
    def test_prompts_when_not_configured(self, mock_prompt):
        console = MagicMock()
        env = {"OPENAI_API_KEY": ""}
        _collect_openai(console, env)
        mock_prompt.assert_called_once()


class TestCollectAnthropic:
    def test_skips_when_configured(self):
        console = MagicMock()
        env = {"ANTHROPIC_API_KEY": "sk-ant-real"}
        _collect_anthropic(console, env)

    @patch("overclaw.commands.init_cmd._prompt_optional_api_key")
    def test_prompts_when_not_configured(self, mock_prompt):
        console = MagicMock()
        env = {"ANTHROPIC_API_KEY": ""}
        _collect_anthropic(console, env)
        mock_prompt.assert_called_once()


class TestReadApiKeyMasked:
    @patch("overclaw.utils.io.sys")
    @patch("overclaw.utils.io.getpass")
    def test_non_tty_falls_back(self, mock_getpass, mock_sys):
        mock_sys.stdin.isatty.return_value = False
        mock_getpass.getpass.return_value = "sk-test"
        result = read_api_key_masked("Test")
        assert result == "sk-test"
