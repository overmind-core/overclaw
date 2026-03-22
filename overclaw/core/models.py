"""Model configurations and backtesting catalog."""

SUPPORTED_LLM_MODELS = [
    # ── OpenAI ──────────────────────────────────────────────────────────
    {"provider": "openai", "model_name": "gpt-5.4"},
    {"provider": "openai", "model_name": "gpt-5.4-pro"},
    {"provider": "openai", "model_name": "gpt-5.4-mini"},
    {"provider": "openai", "model_name": "gpt-5.4-nano"},
    {"provider": "openai", "model_name": "gpt-5.2"},
    {"provider": "openai", "model_name": "gpt-5.2-pro"},
    {"provider": "openai", "model_name": "gpt-5"},
    {"provider": "openai", "model_name": "gpt-5-mini"},
    {"provider": "openai", "model_name": "gpt-5-nano"},
    # ── Anthropic ────────────────────────────────────────────────────────
    {"provider": "anthropic", "model_name": "claude-opus-4-6"},
    {"provider": "anthropic", "model_name": "claude-sonnet-4-6"},
    {"provider": "anthropic", "model_name": "claude-opus-4-5"},
    {"provider": "anthropic", "model_name": "claude-sonnet-4-5"},
    {"provider": "anthropic", "model_name": "claude-haiku-4-5"},
]

DEFAULT_BACKTEST_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5"],
}


def get_providers() -> list[str]:
    """Return deduplicated provider list preserving order."""
    seen: set[str] = set()
    providers: list[str] = []
    for m in SUPPORTED_LLM_MODELS:
        p = m["provider"]
        if p not in seen:
            seen.add(p)
            providers.append(p)
    return providers


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
