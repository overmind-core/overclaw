"""
Interactive environment setup for Overmind.

Loads ``.overmind/.env`` with python-dotenv, reads current values from
``os.environ``, skips questions when API keys are already set, then writes
merged variables back (preserving other keys from the file via
``dotenv_values``).

Usage:
    overmind init
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from overmind import SpanType, attrs, set_tag
from overmind.core.constants import OVERMIND_DIR_NAME, overmind_rel
from overmind.core.registry import init_project_root
from overmind.utils.display import BRAND, confirm_option, select_option
from overmind.utils.display import render_logo as _render_logo
from overmind.utils.io import read_api_key_masked
from overmind.utils.model_picker import prompt_for_catalog_litellm_model
from overmind.utils.models import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_DATAGEN_MODEL,
    model_name_for_env_storage,
    normalize_to_litellm_model_id,
)
from overmind.utils.tracing import traced

# Keys we may set or clear; other keys from the state-dir .env are preserved on write.
# OVERMIND_API_KEY is intentionally excluded — it must be set as a system/shell
# environment variable and is never written to the project .env file.
PRIMARY_ENV_KEYS = [
    "OVERMIND_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANALYZER_MODEL",
    "SYNTHETIC_DATAGEN_MODEL",
]

# Maps LiteLLM provider prefix → the single env var needed to authenticate.
PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
}

KEYS_TO_COLLECT = (
    "OVERMIND_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
)


def _primary_env_from_os() -> dict[str, str]:
    """Snapshot wizard-managed keys from the environment (after ``load_dotenv``)."""
    return {k: (os.getenv(k, "") or "") for k in PRIMARY_ENV_KEYS}


def _key_configured(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    return not re.fullmatch(r"your[-_]?key[-_]?here|changeme|xxx+", v, re.IGNORECASE)


def _model_provider(litellm_model: str) -> str:
    raw = litellm_model.strip()
    canonical = normalize_to_litellm_model_id(raw) or raw
    if "/" in canonical:
        return canonical.split("/", 1)[0].lower().strip()
    return canonical.lower().strip()


def _collect_missing_key_for_model(console: Console, model: str, env: dict[str, str]) -> None:
    """If the provider credential for *model* is absent from both *env* and the
    live process environment, prompt the user to enter it now and save it into
    *env* so it gets written to ``.env``.
    """
    prov = _model_provider(model)
    ek = PROVIDER_ENV_KEY.get(prov)
    if not ek:
        return
    # Check env dict AND os.environ — the key may already be set globally but
    # not yet reflected in the snapshot (e.g. OPENROUTER_API_KEY set in shell).
    if _key_configured(env.get(ek, "")) or _key_configured(os.getenv(ek, "")):
        return

    from overmind.utils.models import get_provider_display_name

    provider_label = get_provider_display_name(prov)
    console.print(
        f"\n  [yellow]Missing credential:[/yellow] [cyan]{provider_label}[/cyan] "
        f"requires [bold]{ek}[/bold] but it is not set."
    )
    val = read_api_key_masked(ek)
    if val.strip():
        env[ek] = val.strip()
        console.print(f"  [bold green]✓[/bold green] Saved [bold]{ek}[/bold] [dim]→ will be written to .env[/dim]")
    else:
        console.print(
            f"  [yellow]Skipped.[/yellow] [dim]Set [bold]{ek}[/bold] in "
            f"{overmind_rel('.env')} before running the pipeline.[/dim]"
        )


def _prompt_optional_api_key(
    console: Console,
    *,
    label: str,
    env_key: str,
    env: dict[str, str],
) -> None:
    """Ask for an API key; empty input means this provider is not used (key cleared)."""
    console.print(f"  [dim]Enter your {label} API key, or press Enter to skip if you are not using {label}.[/dim]")
    key = read_api_key_masked(f"{label} API key")
    if key:
        env[env_key] = key
        console.print("  [dim]Saved as[/dim] [green]*******[/green]")
    else:
        env[env_key] = ""
        console.print(
            f"  [dim]Left empty (not using this provider). You can run init again or edit {overmind_rel('.env')}.[/dim]"
        )


def _collect_openai(console: Console, env: dict[str, str]) -> None:
    if _key_configured(env.get("OPENAI_API_KEY", "")):
        console.print("  [dim]OPENAI_API_KEY is already set — skipping OpenAI setup.[/dim]")
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]OpenAI[/bold]", style=BRAND))
    _prompt_optional_api_key(console, label="OpenAI", env_key="OPENAI_API_KEY", env=env)


def _collect_anthropic(console: Console, env: dict[str, str]) -> None:
    if _key_configured(env.get("ANTHROPIC_API_KEY", "")):
        console.print("  [dim]ANTHROPIC_API_KEY is already set — skipping Anthropic setup.[/dim]")
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Anthropic[/bold]", style=BRAND))
    _prompt_optional_api_key(console, label="Anthropic", env_key="ANTHROPIC_API_KEY", env=env)


def _collect_overmind_api_key(console: Console, env: dict[str, str]) -> None:
    if _key_configured(env.get("OVERMIND_API_KEY", "")):
        console.print("  [dim]OVERMIND_API_KEY is already set — skipping Overmind API key setup.[/dim]")
        return
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Overmind API key[/bold]", style=BRAND))
    _prompt_optional_api_key(console, label="Overmind API key", env_key="OVERMIND_API_KEY", env=env)


def _collect_analyzer_model(console: Console, env: dict[str, str]) -> str:
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]Analyzer model[/bold]", style=BRAND))
    console.print("  [dim]Used to diagnose failures and propose code changes during optimization.[/dim]")
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
    _collect_missing_key_for_model(console, chosen, env)
    return chosen


def _collect_synthetic_datagen_model(console: Console, env: dict[str, str]) -> str | None:
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
        console.print("  [yellow]No SYNTHETIC_DATAGEN_MODEL in the environment — pick a model.[/yellow]")

    chosen = prompt_for_catalog_litellm_model(
        console,
        select_prompt="  Select synthetic data generation model (number)",
        env_default=normalize_to_litellm_model_id(raw) if raw else None,
        default_model=DEFAULT_DATAGEN_MODEL,
        no_catalog_prompt="  Enter model for synthetic data (provider/model)",
    )
    _collect_missing_key_for_model(console, chosen, env)
    return chosen


_SKILLS_SOURCE_DIR = Path(__file__).parent.parent / "skills"

_PLATFORM_SKILLS_DIRS: dict[str, str] = {
    "cursor": ".cursor/skills",
    "claude": ".claude/skills",
}


def _install_skills(console: Console, project_root: Path) -> None:
    """Copy bundled Overmind skills into the user's IDE skills directory."""
    console.print()
    console.print(Rule(style="dim"))
    console.print(Rule("[bold]IDE skills[/bold]", style=BRAND))
    console.print(
        "  [dim]Overmind ships skills for Cursor and Claude Code that let your AI agent run Overmind commands.[/dim]"
    )

    platform_options = ["Cursor (default)", "Claude Code"]
    idx = select_option(
        platform_options,
        title="Which AI coding platform are you using?",
        default_index=0,
        console=console,
    )
    platform = "cursor" if idx == 0 else "claude"

    skills_dir = project_root / _PLATFORM_SKILLS_DIRS[platform]
    skills_dir.mkdir(parents=True, exist_ok=True)

    if not _SKILLS_SOURCE_DIR.is_dir():
        console.print(f"  [yellow]Skills source directory not found at {_SKILLS_SOURCE_DIR}; skipping.[/yellow]")
        return

    installed: list[str] = []
    for skill_src in sorted(_SKILLS_SOURCE_DIR.iterdir()):
        if not skill_src.is_dir():
            continue
        skill_dst = skills_dir / skill_src.name
        if skill_dst.exists():
            shutil.rmtree(skill_dst)
        shutil.copytree(skill_src, skill_dst)
        installed.append(skill_src.name)

    if installed:
        for name in installed:
            console.print(f"  [bold green]✓[/bold green] [bold]{name}[/bold] → [dim]{skills_dir / name}[/dim]")
        console.print(f"\n  [dim]Installed {len(installed)} skill(s) for [bold]{platform_options[idx]}[/bold].[/dim]")
    else:
        console.print("  [yellow]No skills found to install.[/yellow]")


def _write_env(path: Path, env: dict[str, str]) -> None:
    """Merge wizard state into the existing env file (via ``dotenv_values``), then write."""
    file_vals = dotenv_values(path) or {}
    merged: dict[str, str] = {k: (v if v is not None else "") for k, v in file_vals.items()}
    for k in KEYS_TO_COLLECT:
        if k in env:
            merged[k] = env[k]
    merged["ANALYZER_MODEL"] = env["ANALYZER_MODEL"]
    if env.get("SYNTHETIC_DATAGEN_MODEL", "").strip():
        merged["SYNTHETIC_DATAGEN_MODEL"] = env["SYNTHETIC_DATAGEN_MODEL"]
    else:
        merged.pop("SYNTHETIC_DATAGEN_MODEL", None)

    lines: list[str] = ["# Overmind — generated or updated by overmind init", ""]
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


@traced(span_name="overmind_init", type=SpanType.WORKFLOW)
def main() -> None:
    set_tag(attrs.COMMAND, "init")
    oc_dir = init_project_root() / OVERMIND_DIR_NAME
    oc_dir.mkdir(parents=True, exist_ok=True)

    import logging

    from overmind.core.logging import setup_logging

    log_path = setup_logging()
    logging.getLogger("overmind.init").info(f"Running overmind init in {oc_dir} (log_file={log_path})")

    env_path = oc_dir / ".env"
    console = Console()
    console.print()
    _render_logo(console)
    console.print()
    console.print(
        Panel.fit(
            f"[bold {BRAND}]Overmind[/bold {BRAND}] [bold cyan]Overmind — environment setup[/bold cyan]\n"
            f"[dim]Configure API keys and model defaults in {overmind_rel('.env')}[/dim]",
            border_style=BRAND,
        )
    )

    _install_skills(console, oc_dir.parent)

    load_dotenv(env_path)
    env = _primary_env_from_os()
    console.print(
        f"\n  [dim]Loaded environment from [bold]{env_path}[/bold] (python-dotenv) and current process env[/dim]"
    )

    _collect_openai(console, env)
    _collect_anthropic(console, env)
    _collect_overmind_api_key(console, env)

    analyzer = _collect_analyzer_model(console, env)
    env["ANALYZER_MODEL"] = model_name_for_env_storage(normalize_to_litellm_model_id(analyzer) or analyzer)

    synthetic = _collect_synthetic_datagen_model(console, env)
    if synthetic:
        env["SYNTHETIC_DATAGEN_MODEL"] = model_name_for_env_storage(
            normalize_to_litellm_model_id(synthetic) or synthetic
        )
    else:
        env.pop("SYNTHETIC_DATAGEN_MODEL", None)

    console.print()
    console.print(Rule(style="dim"))

    for k in KEYS_TO_COLLECT:
        env.setdefault(k, "")

    _write_env(env_path, env)

    logging.getLogger("overmind.init").info(f"Wrote env file {env_path} (keys={sorted(env.keys())})")

    # Capture what was configured so it shows up in traces
    set_tag(attrs.INIT_ENV_PATH, str(env_path))
    set_tag(
        attrs.INIT_HAS_OPENAI_KEY,
        str(bool((env.get("OPENAI_API_KEY") or "").strip())),
    )
    set_tag(
        attrs.INIT_HAS_ANTHROPIC_KEY,
        str(bool((env.get("ANTHROPIC_API_KEY") or "").strip())),
    )
    set_tag(
        attrs.INIT_HAS_OVERMIND_TOKEN,
        str(bool((os.getenv("OVERMIND_API_KEY") or "").strip())),
    )
    set_tag(attrs.INIT_ANALYZER_MODEL, env.get("ANALYZER_MODEL") or "")
    set_tag(
        attrs.INIT_HAS_SYNTHETIC_DATAGEN_MODEL,
        str(bool((env.get("SYNTHETIC_DATAGEN_MODEL") or "").strip())),
    )

    console.print(f"\n  [green]Wrote[/green] {env_path}")
    console.print("  [dim]Run setup / optimize as usual; keys are read on startup.[/dim]\n")
