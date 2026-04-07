"""Interactive selection of a LiteLLM model from the supported catalog."""

from rich.console import Console
from rich.prompt import Prompt

from overclaw.utils.display import BRAND
from overclaw.utils.models import (
    get_litellm_model_ids,
    get_models_for_provider,
    get_provider_display_name,
    get_providers,
)


def prompt_for_catalog_litellm_model(
    console: Console,
    *,
    select_prompt: str,
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

    # env value takes priority; caller default is the fallback
    effective_default = env_default or (
        default_model if default_model and default_model in models else None
    )

    # ── Step 1: pick provider ────────────────────────────────────────────────
    console.print("\n   [dim]Available providers:[/dim]")
    provider_keys = [str(i) for i in range(1, len(providers) + 1)]

    default_provider_key = "1"
    if effective_default:
        eff_provider = effective_default.split("/")[0]
        if eff_provider in providers:
            default_provider_key = str(providers.index(eff_provider) + 1)

    for i, prov in enumerate(providers, 1):
        tag = (
            " [dim](from .env)[/dim]"
            if env_default and env_default.split("/")[0] == prov
            else ""
        )
        console.print(
            f"     [bold {BRAND}][{i}][/bold {BRAND}] {get_provider_display_name(prov)}{tag}"
        )

    provider_pick = Prompt.ask(
        "   Select provider (number)",
        choices=provider_keys,
        default=default_provider_key,
    )
    chosen_provider = providers[int(provider_pick) - 1]

    # ── Step 2: pick model within provider ───────────────────────────────────
    provider_models = get_models_for_provider(chosen_provider)
    model_keys = [str(i) for i in range(1, len(provider_models) + 1)]

    default_model_key = "1"
    if effective_default:
        eff_prov, _, eff_model = effective_default.partition("/")
        if eff_prov == chosen_provider and eff_model in provider_models:
            default_model_key = str(provider_models.index(eff_model) + 1)

    console.print(
        f"\n   [dim]Available {get_provider_display_name(chosen_provider)} models:[/dim]"
    )
    for i, model_name in enumerate(provider_models, 1):
        tag = (
            " [dim](from .env)[/dim]"
            if env_default and env_default == f"{chosen_provider}/{model_name}"
            else ""
        )
        console.print(f"     [bold {BRAND}][{i}][/bold {BRAND}] {model_name}{tag}")

    model_pick = Prompt.ask(
        select_prompt,
        choices=model_keys,
        default=default_model_key,
    )
    chosen_model = provider_models[int(model_pick) - 1]

    return f"{chosen_provider}/{chosen_model}"
