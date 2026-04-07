"""Interactive selection of a LiteLLM model from the supported catalog."""

from rich.console import Console
from rich.prompt import Prompt

from overclaw.utils.display import select_option
from overclaw.utils.models import (
    get_litellm_model_ids,
    get_models_for_provider,
    get_provider_display_name,
    get_providers,
)


def prompt_for_catalog_litellm_model(
    console: Console,
    *,
    select_prompt: str = "",
    env_default: str | None = None,
    default_model: str | None = None,
    no_catalog_prompt: str = "   Enter model (provider/model)",
) -> str:
    """First ask which provider, then which model; return the chosen ``provider/model`` id.

    *env_default*    — current value from the environment; shown as ``(from .env)`` and
                       used as the pre-selected choice when present.
    *default_model*  — caller-supplied fallback (e.g. ``DEFAULT_ANALYZER_MODEL``) used
                       as the pre-selected choice when *env_default* is absent.
    """
    models = get_litellm_model_ids()
    if not models:
        return Prompt.ask(no_catalog_prompt)

    providers = get_providers()

    effective_default = env_default or (
        default_model if default_model and default_model in models else None
    )

    # ── Step 1: pick provider ────────────────────────────────────────────────
    default_provider_idx = 0
    if effective_default:
        eff_provider = effective_default.split("/")[0]
        if eff_provider in providers:
            default_provider_idx = providers.index(eff_provider)

    provider_labels = []
    for prov in providers:
        label = get_provider_display_name(prov)
        if env_default and env_default.split("/")[0] == prov:
            label += "  (from .env)"
        provider_labels.append(label)

    provider_idx = select_option(
        provider_labels,
        title="Select provider:",
        default_index=default_provider_idx,
        console=console,
    )
    chosen_provider = providers[provider_idx]

    # ── Step 2: pick model within provider ───────────────────────────────────
    provider_models = get_models_for_provider(chosen_provider)

    default_model_idx = 0
    if effective_default:
        eff_prov, _, eff_model = effective_default.partition("/")
        if eff_prov == chosen_provider and eff_model in provider_models:
            default_model_idx = provider_models.index(eff_model)

    model_labels = []
    for model_name in provider_models:
        label = model_name
        if env_default and env_default == f"{chosen_provider}/{model_name}":
            label += "  (from .env)"
        model_labels.append(label)

    model_idx = select_option(
        model_labels,
        title=f"Select {get_provider_display_name(chosen_provider)} model:",
        default_index=default_model_idx,
        console=console,
    )
    chosen_model = provider_models[model_idx]

    return f"{chosen_provider}/{chosen_model}"
