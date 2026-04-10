"""
Interactive environment setup for OverClaw.

Loads ``.overclaw/.env`` with python-dotenv, reads current values from
``os.environ``, skips questions when API keys are already set, then writes
merged variables back (preserving other keys from the file via
``dotenv_values``).

Usage:
    overclaw init
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from overclaw.utils.display import BRAND, confirm_option, render_logo as _render_logo
from overclaw.core.constants import OVERCLAW_DIR_NAME, overclaw_rel
from overclaw.utils.io import read_api_key_masked
from overclaw.core.registry import init_project_root
from overclaw.utils.model_picker import prompt_for_catalog_litellm_model
from overclaw.utils.models import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_DATAGEN_MODEL,
    model_name_for_env_storage,
    normalize_to_litellm_model_id,
)
from overmind_sdk import observe, SpanType, set_tag


# Keys we may set or clear; other keys from the state-dir .env are preserved on write.
PRIMARY_ENV_KEYS = [
    "OVERMIND_API_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANALYZER_MODEL",
    "SYNTHETIC_DATAGEN_MODEL",
]

PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

KEYS_TO_COLLECT = ("OVERMIND_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _primary_env_from_os() -> dict[str, str]:
    """Snapshot wizard-managed keys from the environment (after ``load_dotenv``)."""
    return {k: (os.getenv(k, "") or "") for k in PRIMARY_ENV_KEYS}


def _key_configured(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    if re.fullmatch(r"your[-_]?key[-_]?here|changeme|xxx+", v, re.I):
        return False
    return True


def _model_provider(litellm_model: str) -> str:
    raw = litellm_model.strip()
    canonical = normalize_to_litellm_model_id(raw) or raw
    if "/" in canonical:
        return canonical.split("/", 1)[0].lower().strip()
    return canonical.lower().strip()


def _warn_missing_key_for_model(
    console: Console, model: str, env: dict[str, str]
) -> None:
    prov = _model_provider(model)
    ek = PROVIDER_ENV_KEY.get(prov)
    if not ek:
        return
    if _key_configured(env.get(ek, "")):
        return
    console.print(
        f"\n  [yellow]Warning:[/yellow] model [cyan]{model}[/cyan] typically needs "
        f"[bold]{ek}[/bold], which is empty. Set it before running the pipeline.\n"
    )


def _prompt_optional_api_key(
    console: Console,
    *,
    label: str,
    env_key: str,
    env: dict[str, str],
) -> None:
    """Ask for an API key; empty input means this provider is not used (key cleared)."""
    console.print(
        f"  [dim]Enter your {label} API key, or press Enter to skip if you are not "
        f"using {label}.[/dim]"
    )
    key = read_api_key_masked(f"{label} API key")
    if key:
        env[env_key] = key
        console.print("  [dim]Saved as[/dim] [green]*******[/green]")
    else:
        env[env_key] = ""
        console.print(
            f"  [dim]Left empty (not using this provider). You can run init again or "
            f"edit {overclaw_rel('.env')}.[/dim]"
        )


def _collect_openai(console: Console, env: dict[str, str]) -> None:
    if _key_configured(env.get("OPENAI_API_KEY", "")):
        console.print(
            "  [dim]OPENAI_API_KEY is already set — skipping OpenAI setup.[/dim]"
        )
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]OpenAI[/bold]", style=BRAND))
    _prompt_optional_api_key(console, label="OpenAI", env_key="OPENAI_API_KEY", env=env)


def _collect_anthropic(console: Console, env: dict[str, str]) -> None:
    if _key_configured(env.get("ANTHROPIC_API_KEY", "")):
        console.print(
            "  [dim]ANTHROPIC_API_KEY is already set — skipping Anthropic setup.[/dim]"
        )
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Anthropic[/bold]", style=BRAND))
    _prompt_optional_api_key(
        console, label="Anthropic", env_key="ANTHROPIC_API_KEY", env=env
    )


def _collect_overmind_backend(console: Console, env: dict[str, str]) -> None:
    """Configure Overmind API token used by storage backend auto-selection."""
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Overmind backend (recommended)[/bold]", style=BRAND))
    console.print(
        "  [dim]OverClaw can store setup/optimization artifacts in Overmind for tracking "
        "and visualization. If not configured, it stores artifacts on local disk.[/dim]"
    )

    existing = env.get("OVERMIND_API_TOKEN", "")
    if _key_configured(existing):
        console.print(
            "  [dim]OVERMIND_API_TOKEN is already set — Overmind backend will be preferred "
            "when OVERMIND_API_URL is also configured.[/dim]"
        )
        if confirm_option(
            "Replace existing Overmind API token?", default=False, console=console
        ):
            token = read_api_key_masked("Overmind API key")
            env["OVERMIND_API_TOKEN"] = token
            if token:
                console.print("  [dim]Saved as[/dim] [green]*******[/green]")
            else:
                console.print(
                    "  [dim]Cleared token. Storage will fall back to local disk unless set later.[/dim]"
                )
        return

    console.print(
        "  [dim]Enter your Overmind API key to prefer remote storage "
        "(or press Enter to keep local disk storage).[/dim]"
    )
    console.print("  [dim]Expected token prefix: [bold]ovr_[/bold][/dim]")
    token = read_api_key_masked("Overmind API key")
    if token:
        env["OVERMIND_API_TOKEN"] = token
        console.print("  [dim]Saved as[/dim] [green]*******[/green]")
        if not _key_configured(os.getenv("OVERMIND_API_URL", "")):
            console.print(
                "  [yellow]Note:[/yellow] OVERMIND_API_URL is empty. Set it in "
                f"{overclaw_rel('.env')} to enable Overmind backend."
            )
    else:
        env["OVERMIND_API_TOKEN"] = ""
        console.print(
            "  [dim]No token set. OverClaw will use local disk storage.[/dim]"
        )


def _collect_analyzer_model(console: Console, env: dict[str, str]) -> str:
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Analyzer model[/bold]", style=BRAND))
    console.print(
        "  [dim]Used to diagnose failures and propose code changes during optimization.[/dim]"
    )
    raw = env.get("ANALYZER_MODEL", "").strip()
    if raw:
        normalized = normalize_to_litellm_model_id(raw) or raw
        display = model_name_for_env_storage(normalized)
        if confirm_option(
            f"Use {display} from the environment as the analyzer model?",
            default=True,
            console=console,
        ):
            return normalized or raw
    else:
        console.print("  [yellow]No ANALYZER_MODEL in the environment yet.[/yellow]")

    chosen = prompt_for_catalog_litellm_model(
        console,
        select_prompt="  Select analyzer model (number)",
        env_default=normalize_to_litellm_model_id(raw) if raw else None,
        default_model=DEFAULT_ANALYZER_MODEL,
        no_catalog_prompt="  Enter analyzer model (provider/model)",
    )
    return chosen


def _collect_synthetic_datagen_model(
    console: Console, env: dict[str, str]
) -> str | None:
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Synthetic data generation[/bold]", style=BRAND))
    console.print(
        "  [dim]When enabled, the optimizer can generate synthetic test cases from "
        "your agent spec (see optimize / config data source).[/dim]"
    )
    if not confirm_option(
        "Configure a model for synthetic data generation in your pipeline?",
        default=False,
        console=console,
    ):
        return None

    raw = env.get("SYNTHETIC_DATAGEN_MODEL", "").strip()
    if raw:
        normalized = normalize_to_litellm_model_id(raw) or raw
        display = model_name_for_env_storage(normalized)
        if confirm_option(
            f"Use {display} from the environment for synthetic data generation?",
            default=True,
            console=console,
        ):
            return normalized or raw
    else:
        console.print(
            "  [yellow]No SYNTHETIC_DATAGEN_MODEL in the environment — pick a model.[/yellow]"
        )

    return prompt_for_catalog_litellm_model(
        console,
        select_prompt="  Select synthetic data generation model (number)",
        env_default=normalize_to_litellm_model_id(raw) if raw else None,
        default_model=DEFAULT_DATAGEN_MODEL,
        no_catalog_prompt="  Enter model for synthetic data (provider/model)",
    )


def _write_env(path: Path, env: dict[str, str]) -> None:
    """Merge wizard state into the existing env file (via ``dotenv_values``), then write."""
    file_vals = dotenv_values(path) or {}
    merged: dict[str, str] = {
        k: (v if v is not None else "") for k, v in file_vals.items()
    }
    for k in KEYS_TO_COLLECT:
        if k in env:
            merged[k] = env[k]
    if not (merged.get("OVERMIND_API_TOKEN") or "").strip():
        merged.pop("OVERMIND_API_TOKEN", None)
    merged["ANALYZER_MODEL"] = env["ANALYZER_MODEL"]
    if env.get("SYNTHETIC_DATAGEN_MODEL", "").strip():
        merged["SYNTHETIC_DATAGEN_MODEL"] = env["SYNTHETIC_DATAGEN_MODEL"]
    else:
        merged.pop("SYNTHETIC_DATAGEN_MODEL", None)

    lines: list[str] = ["# OverClaw — generated or updated by overclaw init", ""]
    seen: set[str] = set()

    for key in PRIMARY_ENV_KEYS:
        if key not in merged:
            continue
        val = merged[key]
        if key == "SYNTHETIC_DATAGEN_MODEL" and not val.strip():
            continue
        lines.append(f"{key}={val}")
        seen.add(key)

    rest = sorted(k for k in merged if k not in seen)
    if rest:
        lines.append("")
        for key in rest:
            lines.append(f"{key}={merged[key]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@observe(span_name="overclaw_init", type=SpanType.FUNCTION)
def main() -> None:
    set_tag("overclaw.command", "init")
    oc_dir = init_project_root() / OVERCLAW_DIR_NAME
    oc_dir.mkdir(parents=True, exist_ok=True)
    env_path = oc_dir / ".env"
    console = Console()
    console.print()
    _render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]OverClaw — environment setup[/bold cyan]\n"
            f"[dim]Configure API keys and model defaults in {overclaw_rel('.env')}[/dim]",
            border_style=BRAND,
        )
    )

    load_dotenv(env_path)
    env = _primary_env_from_os()
    console.print(
        f"\n  [dim]Loaded environment from [bold]{env_path}[/bold] "
        "(python-dotenv) and current process env[/dim]"
    )

    _collect_openai(console, env)
    _collect_anthropic(console, env)
    _collect_overmind_backend(console, env)

    analyzer = _collect_analyzer_model(console, env)
    env["ANALYZER_MODEL"] = model_name_for_env_storage(
        normalize_to_litellm_model_id(analyzer) or analyzer
    )

    synthetic = _collect_synthetic_datagen_model(console, env)
    if synthetic:
        env["SYNTHETIC_DATAGEN_MODEL"] = model_name_for_env_storage(
            normalize_to_litellm_model_id(synthetic) or synthetic
        )
    else:
        env.pop("SYNTHETIC_DATAGEN_MODEL", None)

    _warn_missing_key_for_model(console, env["ANALYZER_MODEL"], env)
    if synthetic:
        _warn_missing_key_for_model(console, env["SYNTHETIC_DATAGEN_MODEL"], env)

    console.print()
    console.print(Rule(style="dim"))

    for k in KEYS_TO_COLLECT:
        env.setdefault(k, "")

    _write_env(env_path, env)

    # Capture what was configured so it shows up in traces
    set_tag("overclaw.init.env_path", str(env_path))
    set_tag(
        "overclaw.init.has_openai_key",
        str(bool((env.get("OPENAI_API_KEY") or "").strip())),
    )
    set_tag(
        "overclaw.init.has_anthropic_key",
        str(bool((env.get("ANTHROPIC_API_KEY") or "").strip())),
    )
    set_tag(
        "overclaw.init.has_overmind_token",
        str(bool((env.get("OVERMIND_API_TOKEN") or "").strip())),
    )
    set_tag("overclaw.init.analyzer_model", env.get("ANALYZER_MODEL") or "")
    set_tag(
        "overclaw.init.has_synthetic_datagen_model",
        str(bool((env.get("SYNTHETIC_DATAGEN_MODEL") or "").strip())),
    )

    console.print(f"\n  [green]Wrote[/green] {env_path}")
    console.print(
        "  [dim]Run setup / optimize as usual; keys are read on startup.[/dim]\n"
    )
