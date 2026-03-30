"""Interactive selection of a LiteLLM model from the supported catalog."""

from rich.console import Console
from rich.prompt import Prompt

from overclaw.core.branding import BRAND
from overclaw.core.models import get_providers, get_models_for_provider, get_litellm_model_ids


def prompt_for_catalog_litellm_model(
    console: Console,
    *,
    select_prompt: str,
    env_default: str | None = None,
    no_catalog_prompt: str = "   Enter model (provider/model)",
) -> str:
    """First ask which provider, then which model; return the chosen ``provider/model`` id.

    *env_default* highlights a row as ``(from .env)`` when it appears in the catalog.
    """
    models = get_litellm_model_ids()
    if not models:
        return Prompt.ask(no_catalog_prompt)

    providers = get_providers()

    # ── Step 1: pick provider ────────────────────────────────────────────────
    console.print("\n   [dim]Available providers:[/dim]")
    provider_keys = [str(i) for i in range(1, len(providers) + 1)]

    default_provider_key = "1"
    if env_default:
        env_provider = env_default.split("/")[0]
        if env_provider in providers:
            default_provider_key = str(providers.index(env_provider) + 1)

    for i, prov in enumerate(providers, 1):
        tag = " [dim](from .env)[/dim]" if env_default and env_default.split("/")[0] == prov else ""
        console.print(f"     [bold {BRAND}][{i}][/bold {BRAND}] {prov.title()}{tag}")

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
    if env_default:
        env_prov, _, env_model = env_default.partition("/")
        if env_prov == chosen_provider and env_model in provider_models:
            default_model_key = str(provider_models.index(env_model) + 1)

    console.print(f"\n   [dim]Available {chosen_provider.title()} models:[/dim]")
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
