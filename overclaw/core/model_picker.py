"""Interactive selection of a LiteLLM model from the supported catalog."""

from rich.console import Console
from rich.prompt import Prompt

from overclaw.core.branding import BRAND
from overclaw.core.models import get_litellm_model_ids


def prompt_for_catalog_litellm_model(
    console: Console,
    *,
    select_prompt: str,
    env_default: str | None = None,
    no_catalog_prompt: str = "   Enter model (provider/model)",
) -> str:
    """List supported models by provider; return the chosen ``provider/model`` id.

    Rows show the short model name (e.g. ``claude-sonnet-4-6``) under each provider;
    the returned value remains the full LiteLLM id.

    *env_default* highlights a row as ``(from .env)`` and becomes the numeric default
    when it appears in the catalog.
    """
    models = get_litellm_model_ids()
    if not models:
        return Prompt.ask(no_catalog_prompt)

    console.print("\n   [dim]Available models:[/dim]")
    choice_keys = [str(i) for i in range(1, len(models) + 1)]
    default_key = "1"
    if env_default and env_default in models:
        default_key = str(models.index(env_default) + 1)

    current_provider: str | None = None
    for i, mid in enumerate(models, 1):
        prov, _, model_name = mid.partition("/")
        display = model_name if model_name else mid
        if prov != current_provider:
            current_provider = prov
            console.print(f"\n   [bold]{prov.title()}[/bold]")
        tag = " [dim](from .env)[/dim]" if env_default and mid == env_default else ""
        console.print(f"     [bold {BRAND}][{i}][/bold {BRAND}] {display}{tag}")

    pick = Prompt.ask(
        select_prompt,
        choices=choice_keys,
        default=default_key,
    )
    return models[int(pick) - 1]
