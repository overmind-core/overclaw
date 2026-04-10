"""Tests for overclaw.commands.init_cmd — environment setup wizard helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from overclaw.commands.init_cmd import (
    _collect_missing_key_for_model,
    _key_configured,
    _model_provider,
    _primary_env_from_os,
    _write_env,
)


# ---------------------------------------------------------------------------
# _key_configured
# ---------------------------------------------------------------------------


class TestKeyConfigured:
    def test_empty(self):
        assert _key_configured("") is False

    def test_whitespace(self):
        assert _key_configured("   ") is False

    def test_placeholder_your_key_here(self):
        assert _key_configured("your_key_here") is False

    def test_placeholder_changeme(self):
        assert _key_configured("changeme") is False

    def test_placeholder_xxx(self):
        assert _key_configured("xxxx") is False

    def test_placeholder_your_key_here_variant(self):
        assert _key_configured("YOUR-KEY-HERE") is False

    def test_valid_key(self):
        assert _key_configured("sk-abc123def456") is True

    def test_real_looking_key(self):
        assert _key_configured("sk-ant-api03-abc123") is True


# ---------------------------------------------------------------------------
# _model_provider
# ---------------------------------------------------------------------------


class TestModelProvider:
    def test_full_litellm_id(self):
        assert _model_provider("openai/gpt-5.4") == "openai"

    def test_bare_known_model(self):
        assert _model_provider("gpt-5.4") == "openai"

    def test_anthropic(self):
        assert _model_provider("anthropic/claude-sonnet-4-6") == "anthropic"

    def test_unknown_model(self):
        result = _model_provider("custom-model")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _primary_env_from_os
# ---------------------------------------------------------------------------


class TestPrimaryEnvFromOs:
    def test_reads_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        monkeypatch.delenv("SYNTHETIC_DATAGEN_MODEL", raising=False)
        env = _primary_env_from_os()
        assert env["OPENAI_API_KEY"] == "sk-test"
        assert env["ANTHROPIC_API_KEY"] == ""
        assert "ANALYZER_MODEL" in env
        assert "SYNTHETIC_DATAGEN_MODEL" in env


# ---------------------------------------------------------------------------
# _collect_missing_key_for_model
# ---------------------------------------------------------------------------


class TestCollectMissingKeyForModel:
    @patch("overclaw.commands.init_cmd.read_api_key_masked", return_value="")
    def test_prompts_when_key_missing(self, _mock_read, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        console = MagicMock()
        _collect_missing_key_for_model(
            console, "openai/gpt-5.4", {"OPENAI_API_KEY": ""}
        )
        console.print.assert_called()

    def test_no_prompt_when_key_present(self):
        console = MagicMock()
        _collect_missing_key_for_model(
            console, "openai/gpt-5.4", {"OPENAI_API_KEY": "sk-real-key"}
        )
        console.print.assert_not_called()

    def test_unknown_provider_no_prompt(self):
        console = MagicMock()
        _collect_missing_key_for_model(console, "custom/model", {})
        console.print.assert_not_called()


# ---------------------------------------------------------------------------
# _write_env
# ---------------------------------------------------------------------------


class TestWriteEnv:
    def test_writes_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("")
        env = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "",
            "ANALYZER_MODEL": "gpt-5.4",
            "SYNTHETIC_DATAGEN_MODEL": "",
        }
        _write_env(env_path, env)
        content = env_path.read_text()
        assert "OPENAI_API_KEY=sk-test" in content
        assert "ANALYZER_MODEL=gpt-5.4" in content
        assert "SYNTHETIC_DATAGEN_MODEL" not in content  # empty skipped

    def test_preserves_extra_keys(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("CUSTOM_KEY=custom_value\n")
        env = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "",
            "ANALYZER_MODEL": "gpt-5.4",
        }
        _write_env(env_path, env)
        content = env_path.read_text()
        assert "CUSTOM_KEY=custom_value" in content

    def test_includes_header(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("")
        _write_env(
            env_path,
            {
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "ANALYZER_MODEL": "m",
            },
        )
        content = env_path.read_text()
        assert "overclaw init" in content.lower() or "OverClaw" in content
