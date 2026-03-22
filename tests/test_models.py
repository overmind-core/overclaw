"""Tests for overclaw.core.models — model catalog and helpers."""

from __future__ import annotations

from overclaw.core.models import (
    SUPPORTED_LLM_MODELS,
    get_default_models_for_provider,
    get_litellm_model_ids,
    get_models_for_provider,
    get_providers,
    model_name_for_env_storage,
    normalize_to_litellm_model_id,
)


class TestSupportedModels:
    def test_catalog_not_empty(self):
        assert len(SUPPORTED_LLM_MODELS) > 0

    def test_each_entry_has_provider_and_name(self):
        for m in SUPPORTED_LLM_MODELS:
            assert "provider" in m
            assert "model_name" in m
            assert m["provider"]
            assert m["model_name"]


class TestGetProviders:
    def test_returns_providers(self):
        providers = get_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_no_duplicates(self):
        providers = get_providers()
        assert len(providers) == len(set(providers))

    def test_preserves_order(self):
        providers = get_providers()
        first_provider = SUPPORTED_LLM_MODELS[0]["provider"]
        assert providers[0] == first_provider


class TestGetModelsForProvider:
    def test_openai_models(self):
        models = get_models_for_provider("openai")
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_anthropic_models(self):
        models = get_models_for_provider("anthropic")
        assert len(models) > 0

    def test_unknown_provider(self):
        assert get_models_for_provider("unknown-provider") == []


class TestGetDefaultModelsForProvider:
    def test_openai_defaults(self):
        defaults = get_default_models_for_provider("openai")
        assert len(defaults) > 0
        for d in defaults:
            assert d in get_models_for_provider("openai")

    def test_anthropic_defaults(self):
        defaults = get_default_models_for_provider("anthropic")
        assert len(defaults) > 0

    def test_unknown_provider(self):
        assert get_default_models_for_provider("unknown") == []


class TestGetLitellmModelIds:
    def test_format(self):
        ids = get_litellm_model_ids()
        assert len(ids) == len(SUPPORTED_LLM_MODELS)
        for model_id in ids:
            assert "/" in model_id

    def test_each_has_provider_prefix(self):
        for model_id in get_litellm_model_ids():
            provider, name = model_id.split("/", 1)
            assert provider
            assert name


class TestNormalizeToLitellmModelId:
    def test_bare_model_name(self):
        result = normalize_to_litellm_model_id("gpt-5.4")
        assert result == "openai/gpt-5.4"

    def test_full_litellm_id(self):
        result = normalize_to_litellm_model_id("openai/gpt-5.4")
        assert result == "openai/gpt-5.4"

    def test_unknown_model(self):
        assert normalize_to_litellm_model_id("unknown-model-xyz") is None

    def test_empty_string(self):
        assert normalize_to_litellm_model_id("") is None

    def test_whitespace_stripped(self):
        result = normalize_to_litellm_model_id("  gpt-5.4  ")
        assert result == "openai/gpt-5.4"

    def test_anthropic_model(self):
        result = normalize_to_litellm_model_id("claude-sonnet-4-6")
        assert result == "anthropic/claude-sonnet-4-6"


class TestModelNameForEnvStorage:
    def test_known_model_returns_bare_name(self):
        result = model_name_for_env_storage("openai/gpt-5.4")
        assert result == "gpt-5.4"

    def test_bare_known_model(self):
        result = model_name_for_env_storage("gpt-5.4")
        assert result == "gpt-5.4"

    def test_unknown_model_returned_as_is(self):
        result = model_name_for_env_storage("custom-provider/custom-model")
        assert result == "custom-provider/custom-model"

    def test_empty_string(self):
        assert model_name_for_env_storage("") == ""

    def test_whitespace(self):
        assert model_name_for_env_storage("  ") == ""
