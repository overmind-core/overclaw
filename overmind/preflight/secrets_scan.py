"""Static + runtime detection of credentials the agent needs.

Two surfaces:

- :func:`scan_secrets`  — read-only, JSON-serialisable.  Returns the list
  of env-var names that are required by the agent code (or the analyzer
  / provider strings discovered in the entrypoint and adjacent modules)
  but are missing from both the global environment and the per-agent
  ``.env``.  The skill reads this list and asks the user once.

- :func:`set_secret`    — append a single credential into the per-agent
  ``.env`` and reload it.  Idempotent, never logs the value, hardens the
  file to ``0600`` after writing.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from dotenv import dotenv_values

from overmind.core.paths import (
    agent_env_path,
    agent_instrumented_dir,
    load_agent_dotenv,
    load_overmind_dotenv,
    overmind_env_path,
)
from overmind.core.registry import resolve_agent
from overmind.utils.env_scan import discover_env_var_defaults
from overmind.utils.provider_keys import PROVIDER_ENV_KEYS, update_agent_env

# Always considered "supplied by the OS" — never asked from the user.
_SYSTEM_ENV_VARS: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "PWD",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONIOENCODING",
})

# Keys that are part of overmind plumbing (already created by `init`).
_OVERMIND_INTERNAL_VARS: frozenset[str] = frozenset({
    "OVERMIND_API_KEY",
    "OVERMIND_TRACE_FILE",
    "OVERMIND_DISABLE_SHADOW",
    "ANALYZER_MODEL",
    "SYNTHETIC_DATAGEN_MODEL",
    "ENV_SETUP_MODEL",
    "LLM_JUDGE_MODEL",
})


# Heuristic provider detection from a litellm-style model string.
_PROVIDER_FROM_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "bedrock": "bedrock",
    "gemini": "gemini",
    "google": "gemini",
    "vertex_ai": "gemini",
    "mistral": "mistral",
    "cohere": "cohere",
    "groq": "groq",
}


_PROVIDER_KEYS_EXTRA: dict[str, list[str]] = {
    "gemini": ["GEMINI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
}


def _all_provider_keys() -> dict[str, list[str]]:
    out = dict(PROVIDER_ENV_KEYS)
    for k, v in _PROVIDER_KEYS_EXTRA.items():
        out[k] = v
    return out


_MODEL_STRING_RE = re.compile(
    r"""['"]                # opening quote
        (?P<full>
          (?P<prov>openai|anthropic|openrouter|bedrock|gemini|google|vertex_ai|mistral|cohere|groq)
          /[A-Za-z0-9_./:-]+
        )
        ['"]
    """,
    re.VERBOSE,
)


def _read_sources(agent_name: str) -> dict[str, str]:
    """Return ``{relpath: source}`` for every ``.py`` under the instrumented copy.

    The instrumented copy is the source of truth for what the agent will
    actually import at run time, so it's also the right surface for
    secret discovery.  Falls back to just the entrypoint file when the
    copy doesn't exist yet.
    """
    sources: dict[str, str] = {}
    inst_dir = agent_instrumented_dir(agent_name)
    if inst_dir.is_dir():
        for path in inst_dir.rglob("*.py"):
            try:
                sources[str(path)] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return sources

    try:
        agent_path, _ = resolve_agent(agent_name)
    except SystemExit:
        return sources
    p = Path(agent_path)
    if p.is_file():
        try:
            sources[str(p)] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return sources


def _detected_providers(sources: dict[str, str]) -> set[str]:
    """Find provider prefixes in literal model strings (``"openai/gpt-4o"``)."""
    found: set[str] = set()
    for text in sources.values():
        for m in _MODEL_STRING_RE.finditer(text):
            prefix = m.group("prov")
            mapped = _PROVIDER_FROM_PREFIX.get(prefix)
            if mapped:
                found.add(mapped)
    return found


def _is_supplied(name: str, env_overrides: dict[str, str]) -> bool:
    """True iff *name* has a non-empty, non-placeholder value somewhere."""
    val = env_overrides.get(name) or os.environ.get(name, "")
    val = (val or "").strip()
    if not val:
        return False
    # Treat common placeholder forms as missing so the skill keeps prompting
    # until the user actually pastes a real key.
    placeholders = ("<set-me>", "<your-", "REPLACE_ME", "changeme", "TODO")
    return not any(val.startswith(p) for p in placeholders)


def scan_secrets(agent_name: str) -> dict[str, object]:
    """Return a JSON-friendly description of credentials the agent needs.

    Output schema::

        {
            "agent_name": "...",
            "env_path":   ".overmind/agents/<name>/.env",
            "discovered_env_vars": {"FOO": "default-or-null"},
            "providers_detected": ["openai", "anthropic"],
            "required_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", ...],
            "missing": ["OPENAI_API_KEY"],
            "supplied": ["ANTHROPIC_API_KEY"],
        }

    The skill picks up ``missing`` to drive a single ``AskQuestion``
    batch, then calls :func:`set_secret` per answer.
    """
    load_overmind_dotenv()
    load_agent_dotenv(agent_name)

    sources = _read_sources(agent_name)
    discovered = discover_env_var_defaults(sources)

    # Drop noise — system vars, overmind internals, and the entries that
    # only ever appear inside the wrapper bootstrap.
    filtered: dict[str, str | None] = {}
    for name, default in discovered.items():
        if name in _SYSTEM_ENV_VARS or name in _OVERMIND_INTERNAL_VARS:
            continue
        filtered[name] = default

    providers = _detected_providers(sources)
    required_from_providers: list[str] = []
    for prov in sorted(providers):
        for key in _all_provider_keys().get(prov, []):
            if key not in required_from_providers:
                required_from_providers.append(key)

    # Combine everything into a single required list, preserving order.
    required: list[str] = []
    for key in required_from_providers:
        if key not in required:
            required.append(key)
    for key in sorted(filtered):
        if key not in required:
            required.append(key)

    env_path = agent_env_path(agent_name)
    env_overrides: dict[str, str] = {}
    if env_path.is_file():
        env_overrides = {k: (v or "") for k, v in (dotenv_values(env_path) or {}).items()}
    overmind_env = overmind_env_path()
    if overmind_env.is_file():
        for k, v in (dotenv_values(overmind_env) or {}).items():
            env_overrides.setdefault(k, v or "")

    missing = [k for k in required if not _is_supplied(k, env_overrides)]
    supplied = [k for k in required if _is_supplied(k, env_overrides)]

    return {
        "agent_name": agent_name,
        "env_path": str(env_path),
        "discovered_env_vars": filtered,
        "providers_detected": sorted(providers),
        "required_keys": required,
        "missing": missing,
        "supplied": supplied,
    }


def set_secret(agent_name: str, key: str, value: str, *, validate: bool = True) -> dict[str, object]:
    """Persist *value* under *key* in the per-agent ``.env`` and reload it.

    Returns a small JSON envelope describing the outcome (no values are
    ever included, only the key name and the resolved file path).  When
    *validate* is True and the key looks like a known provider key, a
    cheap sanity check is attempted (e.g. ``litellm.completion`` with
    ``max_tokens=1``).  Failures are reported via ``"validated": false``
    and a diagnostic in ``"validate_error"`` — the value is still saved.
    """
    key = key.strip()
    if not key:
        return {"status": "error", "error": "empty_key"}

    val = value.strip()
    if not val:
        return {"status": "error", "error": "empty_value", "key": key}

    env_path = agent_env_path(agent_name)
    update_agent_env(env_path, agent_name, {key: val})
    try:
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    load_agent_dotenv(agent_name)

    out: dict[str, object] = {
        "status": "ok",
        "key": key,
        "env_path": str(env_path),
        "validated": False,
    }
    if validate and _looks_like_provider_key(key):
        ok, err = _try_validate_provider_key(key, val)
        out["validated"] = ok
        if not ok and err:
            out["validate_error"] = err
    return out


def _looks_like_provider_key(name: str) -> bool:
    for keys in _all_provider_keys().values():
        if name in keys:
            return True
    return False


def _try_validate_provider_key(key: str, value: str) -> tuple[bool, str]:
    """Attempt a one-token completion to check the key is live.

    Returns ``(ok, error_message)``.  Never raises.  Skipped silently
    when ``litellm`` cannot be imported or no obvious test model exists
    for the provider.
    """
    test_models = {
        "OPENAI_API_KEY": "openai/gpt-4o-mini",
        "ANTHROPIC_API_KEY": "anthropic/claude-3-5-haiku-20241022",
        "OPENROUTER_API_KEY": "openrouter/openai/gpt-4o-mini",
        "GEMINI_API_KEY": "gemini/gemini-1.5-flash",
        "MISTRAL_API_KEY": "mistral/mistral-tiny",
        "GROQ_API_KEY": "groq/llama-3.1-8b-instant",
        "COHERE_API_KEY": "cohere/command-r",
    }
    model = test_models.get(key)
    if not model:
        return True, ""
    try:
        import litellm  # type: ignore
    except ImportError:
        return True, ""

    os.environ[key] = value
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
