"""Model configurations and backtesting catalog."""

# Providers that have a fixed model catalog the user selects from.
SUPPORTED_LLM_MODELS = [
    # ── OpenAI ──────────────────────────────────────────────────────────
    {"provider": "openai", "model_name": "gpt-5.4", "provider_display_name": "OpenAI"},
    {
        "provider": "openai",
        "model_name": "gpt-5.4-pro",
        "provider_display_name": "OpenAI",
    },
    {
        "provider": "openai",
        "model_name": "gpt-5.4-mini",
        "provider_display_name": "OpenAI",
    },
    {
        "provider": "openai",
        "model_name": "gpt-5.4-nano",
        "provider_display_name": "OpenAI",
    },
    {"provider": "openai", "model_name": "gpt-5.2", "provider_display_name": "OpenAI"},
    {
        "provider": "openai",
        "model_name": "gpt-5.2-pro",
        "provider_display_name": "OpenAI",
    },
    {"provider": "openai", "model_name": "gpt-5", "provider_display_name": "OpenAI"},
    {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "provider_display_name": "OpenAI",
    },
    {
        "provider": "openai",
        "model_name": "gpt-5-nano",
        "provider_display_name": "OpenAI",
    },
    # ── Anthropic ────────────────────────────────────────────────────────
    {
        "provider": "anthropic",
        "model_name": "claude-opus-4-6",
        "provider_display_name": "Anthropic",
    },
    {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-6",
        "provider_display_name": "Anthropic",
    },
    {
        "provider": "anthropic",
        "model_name": "claude-opus-4-5",
        "provider_display_name": "Anthropic",
    },
    {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-5",
        "provider_display_name": "Anthropic",
    },
    {
        "provider": "anthropic",
        "model_name": "claude-haiku-4-5",
        "provider_display_name": "Anthropic",
    },
]

# Providers with no fixed catalog — the user types the model name directly.
# LiteLLM model-id prefix → human-readable display name.
CUSTOM_MODEL_PROVIDERS: dict[str, str] = {
    "bedrock": "AWS Bedrock",
    "openrouter": "OpenRouter",
}

DEFAULT_BACKTEST_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5"],
}

# Pre-selected defaults for interactive model prompts.
DEFAULT_ANALYZER_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_DATAGEN_MODEL = "anthropic/claude-sonnet-4-6"


def get_providers() -> list[str]:
    """Return deduplicated provider list: catalog providers first, then custom-input providers."""
    seen: set[str] = set()
    providers: list[str] = []
    for m in SUPPORTED_LLM_MODELS:
        p = m["provider"]
        if p not in seen:
            seen.add(p)
            providers.append(p)
    for p in CUSTOM_MODEL_PROVIDERS:
        if p not in seen:
            seen.add(p)
            providers.append(p)
    return providers


def get_provider_display_name(provider: str) -> str:
    """Return the human-readable display name for *provider*."""
    if provider in CUSTOM_MODEL_PROVIDERS:
        return CUSTOM_MODEL_PROVIDERS[provider]
    for m in SUPPORTED_LLM_MODELS:
        if m["provider"] == provider:
            return m.get("provider_display_name") or provider.title()
    return provider.title()


def is_custom_model_provider(provider: str) -> bool:
    """Return True for providers where the user must supply the model name (no fixed catalog)."""
    return provider in CUSTOM_MODEL_PROVIDERS


def get_models_for_provider(provider: str) -> list[str]:
    return [m["model_name"] for m in SUPPORTED_LLM_MODELS if m["provider"] == provider]


def get_default_models_for_provider(provider: str) -> list[str]:
    return DEFAULT_BACKTEST_MODELS.get(provider, [])


def get_litellm_model_ids() -> list[str]:
    """Return supported models as LiteLLM identifiers ``provider/model``."""
    return [f"{m['provider']}/{m['model_name']}" for m in SUPPORTED_LLM_MODELS]


def normalize_to_litellm_model_id(model: str) -> str | None:
    """Map a bare model name or ``provider/model`` to a catalog id if recognized."""
    model = model.strip()
    if not model:
        return None
    for m in SUPPORTED_LLM_MODELS:
        full = f"{m['provider']}/{m['model_name']}"
        if model == full or model == m["model_name"]:
            return full
    return None


def model_name_for_env_storage(model: str) -> str:
    """If *model* is a catalog entry (bare or ``provider/model``), return bare ``model_name``.

    Unknown ids (custom providers) are returned unchanged.
    """
    model = model.strip()
    if not model:
        return model
    full = normalize_to_litellm_model_id(model)
    if full:
        return full.split("/", 1)[1]
    return model
