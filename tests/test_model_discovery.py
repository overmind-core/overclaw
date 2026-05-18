"""Tests for overmind.utils.model_discovery — live provider model listings.

The helpers are best-effort: every transport failure or unexpected payload
must return ``None`` (or an empty list) so the caller can fall back to the
static catalog without crashing.  Network is mocked at the ``_http_get_json``
seam so no real HTTP traffic leaves the test process.
"""

from __future__ import annotations

from unittest.mock import patch

from overmind.utils import model_discovery
from overmind.utils.model_discovery import (
    list_anthropic_models,
    list_models_for_provider,
    list_openai_models,
    list_openrouter_models,
)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class TestListOpenAIModels:
    def test_returns_none_for_empty_key(self):
        assert list_openai_models("") is None
        assert list_openai_models("   ") is None

    @patch("overmind.utils.model_discovery._http_get_json", return_value=None)
    def test_returns_none_on_transport_failure(self, _mock_http):
        assert list_openai_models("sk-test") is None

    @patch(
        "overmind.utils.model_discovery._http_get_json",
        return_value={"data": "not-a-list"},
    )
    def test_returns_none_on_bad_payload_shape(self, _mock_http):
        assert list_openai_models("sk-test") is None

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_filters_to_chat_models(self, mock_http):
        mock_http.return_value = {
            "data": [
                {"id": "gpt-5.4"},
                {"id": "gpt-5.4-2026-03-05"},
                {"id": "gpt-5"},
                {"id": "o3"},
                {"id": "o3-mini"},
                # Should be filtered out:
                {"id": "text-embedding-3-small"},
                {"id": "whisper-1"},
                {"id": "tts-1"},
                {"id": "dall-e-3"},
                {"id": "gpt-4o-audio-preview"},
                {"id": "gpt-4o-transcribe"},
                # Bad entries:
                {},
                {"id": ""},
                "not-a-dict",
            ]
        }
        ids = list_openai_models("sk-test")
        assert ids is not None
        assert "gpt-5.4" in ids
        assert "gpt-5.4-2026-03-05" in ids
        assert "o3" in ids
        assert "o3-mini" in ids
        # filtered:
        assert "text-embedding-3-small" not in ids
        assert "whisper-1" not in ids
        assert "tts-1" not in ids
        assert "dall-e-3" not in ids
        assert "gpt-4o-audio-preview" not in ids
        assert "gpt-4o-transcribe" not in ids

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_sorts_descending_so_newest_floats_top(self, mock_http):
        mock_http.return_value = {
            "data": [{"id": "gpt-5"}, {"id": "gpt-5.4"}, {"id": "gpt-5.2"}]
        }
        ids = list_openai_models("sk-test")
        assert ids == ["gpt-5.4", "gpt-5.2", "gpt-5"]

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_honours_custom_base_url(self, mock_http):
        mock_http.return_value = {"data": [{"id": "gpt-5"}]}
        list_openai_models("sk-test", base_url="https://gateway.example.com/v3")
        called_url = mock_http.call_args[0][0]
        assert called_url == "https://gateway.example.com/v3/v1/models"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class TestListAnthropicModels:
    def test_returns_none_for_empty_key(self):
        assert list_anthropic_models("") is None

    @patch("overmind.utils.model_discovery._http_get_json", return_value=None)
    def test_returns_none_on_transport_failure(self, _mock_http):
        assert list_anthropic_models("sk-ant-test") is None

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_returns_all_ids(self, mock_http):
        mock_http.return_value = {
            "data": [
                {"id": "claude-sonnet-4-6-20260205", "type": "model"},
                {"id": "claude-opus-4-6-20260205", "type": "model"},
                {"id": "claude-haiku-4-5-20251001", "type": "model"},
                # Garbage entries — must not crash:
                {"id": None},
                {},
                "stray-string",
            ]
        }
        ids = list_anthropic_models("sk-ant-test")
        assert ids == [
            "claude-sonnet-4-6-20260205",
            "claude-opus-4-6-20260205",
            "claude-haiku-4-5-20251001",
        ]

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_sends_required_headers(self, mock_http):
        mock_http.return_value = {"data": []}
        list_anthropic_models("sk-ant-secret")
        called_kwargs = mock_http.call_args.kwargs
        headers = called_kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-secret"
        assert headers["anthropic-version"]


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
class TestListOpenRouterModels:
    @patch("overmind.utils.model_discovery._http_get_json", return_value=None)
    def test_returns_none_on_transport_failure(self, _mock_http):
        assert list_openrouter_models() is None

    @patch("overmind.utils.model_discovery._http_get_json")
    def test_returns_sorted_ids(self, mock_http):
        mock_http.return_value = {
            "data": [
                {"id": "x-ai/grok-4.20"},
                {"id": "anthropic/claude-opus-4.7-fast"},
                {"id": "google/gemma-4-31b-it"},
            ]
        }
        ids = list_openrouter_models()
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
class TestListModelsForProvider:
    def test_returns_none_for_provider_without_key(self):
        # No key in env, no key in os.environ
        assert list_models_for_provider("openai", env={}) is None
        assert list_models_for_provider("anthropic", env={}) is None

    @patch("overmind.utils.model_discovery.list_openai_models", return_value=["gpt-5"])
    def test_uses_env_snapshot_key_over_process_env(self, mock_openai):
        list_models_for_provider("openai", env={"OPENAI_API_KEY": "from-init"})
        # The first positional argument is the api key.
        assert mock_openai.call_args.args[0] == "from-init"

    @patch("overmind.utils.model_discovery.list_openai_models", return_value=["gpt-5"])
    def test_forwards_base_url(self, mock_openai):
        list_models_for_provider(
            "openai",
            env={"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://example.com/v3"},
        )
        assert mock_openai.call_args.kwargs == {"base_url": "https://example.com/v3"}

    @patch(
        "overmind.utils.model_discovery.list_anthropic_models",
        return_value=["claude-sonnet-4-6-20260205"],
    )
    def test_anthropic_dispatch(self, _mock_anthropic):
        result = list_models_for_provider("anthropic", env={"ANTHROPIC_API_KEY": "k"})
        assert result == ["claude-sonnet-4-6-20260205"]

    @patch(
        "overmind.utils.model_discovery.list_openrouter_models",
        return_value=["anthropic/claude-opus-4.7-fast"],
    )
    def test_openrouter_dispatch_requires_no_key(self, _mock_openrouter):
        result = list_models_for_provider("openrouter", env={})
        assert result == ["anthropic/claude-opus-4.7-fast"]

    def test_bedrock_returns_none_for_now(self):
        # Bedrock requires boto3 + AWS signing — discovery deliberately
        # deferred to the custom-input UX, so the dispatch must return None.
        assert list_models_for_provider("bedrock", env={}) is None

    def test_unknown_provider_returns_none(self):
        assert list_models_for_provider("definitely-not-a-provider", env={}) is None


# ---------------------------------------------------------------------------
# _http_get_json safety
# ---------------------------------------------------------------------------
class TestHttpGetJsonSafety:
    """The HTTP helper must swallow every conceivable failure mode.

    Discovery is best-effort: if it raises, the picker would crash and
    interactive init would die.  Verify each branch returns ``None``.
    """

    def test_url_error_returns_none(self):
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
            assert model_discovery._http_get_json("https://example.invalid") is None

    def test_http_error_returns_none(self):
        import urllib.error

        err = urllib.error.HTTPError(
            "https://example.invalid", 401, "Unauthorized", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            assert model_discovery._http_get_json("https://example.invalid") is None

    def test_timeout_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
            assert model_discovery._http_get_json("https://example.invalid") is None

    def test_non_json_body_returns_none(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"not json"

        with patch("urllib.request.urlopen", return_value=_Resp()):
            assert model_discovery._http_get_json("https://example.invalid") is None
