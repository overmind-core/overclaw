"""Live model discovery for LLM providers.

The static catalog in :mod:`overmind.utils.models` is convenient but drifts
out of date the moment a provider ships or deprecates a model.  When a user
runs ``overmind init`` and has just supplied a valid API key, we can do
better than guessing: hit the provider's ``/v1/models`` endpoint and offer
the real, current list.

The functions in this module are all best-effort: a network failure, an
invalid key, or an unfamiliar response shape returns ``None`` (or an empty
list) so the caller can transparently fall back to the static catalog
without breaking the interactive flow.

All listings are returned as **bare model ids** (e.g. ``gpt-5.4``,
``claude-sonnet-4-6-20260205``) without the ``provider/`` prefix; callers
add that prefix when persisting the LiteLLM model id.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 8.0


# ---------------------------------------------------------------------------
# Low-level HTTP helper (no third-party deps — runs before .venv may exist)
# ---------------------------------------------------------------------------
def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict | list | None:
    """GET *url* and return parsed JSON, or ``None`` on any failure."""
    try:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("model discovery GET %s failed: %s", url, exc)
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("model discovery GET %s returned non-JSON: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
# Chat-capable model prefixes.  The /v1/models endpoint also returns
# embeddings, TTS, image, transcription, and moderation models — filter
# them out so the picker only shows things a chat-completion analyzer can
# actually call.
_OPENAI_CHAT_PREFIXES: tuple[str, ...] = ("gpt-", "o1", "o3", "o4", "chatgpt-")
# Drop these substrings even when they match a chat prefix (e.g. ``gpt-4o-audio-preview``):
_OPENAI_NON_CHAT_SUBSTRINGS: tuple[str, ...] = (
    "audio",
    "realtime",
    "transcribe",
    "tts",
    "image",
    "search",
    "moderation",
    "embedding",
)


def list_openai_models(api_key: str, *, base_url: str | None = None) -> list[str] | None:
    """Return the chat-capable model ids the OpenAI endpoint advertises.

    ``base_url`` lets callers point at OpenAI-compatible gateways (Azure
    OpenAI, vLLM, OpenRouter via OpenAI shim, etc.).  Returns ``None`` on
    transport failure or an unrecognised payload shape so the caller can
    fall back to the static catalog.
    """
    if not api_key.strip():
        return None
    base = (base_url or "https://api.openai.com").rstrip("/")
    payload = _http_get_json(
        f"{base}/v1/models",
        headers={"Authorization": f"Bearer {api_key.strip()}"},
    )
    if not isinstance(payload, dict):
        return None
    items = payload.get("data")
    if not isinstance(items, list):
        return None
    ids: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        if not any(mid.startswith(p) for p in _OPENAI_CHAT_PREFIXES):
            continue
        if any(sub in mid for sub in _OPENAI_NON_CHAT_SUBSTRINGS):
            continue
        ids.add(mid)
    # Sort so the newest-looking ids float to the top.  We deliberately
    # use reverse-lexicographic ordering: OpenAI's dated suffixes
    # (``-2026-03-05``) sort newest-first this way, and the bare alias
    # (``gpt-5.4``) sorts above its dated variants.
    return sorted(ids, reverse=True)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
_ANTHROPIC_API_VERSION = "2023-06-01"


def list_anthropic_models(api_key: str) -> list[str] | None:
    """Return the model ids the Anthropic API advertises for the given key."""
    if not api_key.strip():
        return None
    payload = _http_get_json(
        "https://api.anthropic.com/v1/models?limit=1000",
        headers={
            "x-api-key": api_key.strip(),
            "anthropic-version": _ANTHROPIC_API_VERSION,
        },
    )
    if not isinstance(payload, dict):
        return None
    items = payload.get("data")
    if not isinstance(items, list):
        return None
    ids: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id")
        if isinstance(mid, str) and mid:
            ids.add(mid)
    return sorted(ids, reverse=True)


# ---------------------------------------------------------------------------
# OpenRouter (public catalogue — no auth required)
# ---------------------------------------------------------------------------
def list_openrouter_models() -> list[str] | None:
    """Return all OpenRouter model paths (no auth required)."""
    payload = _http_get_json("https://openrouter.ai/api/v1/models")
    if not isinstance(payload, dict):
        return None
    items = payload.get("data")
    if not isinstance(items, list):
        return None
    ids: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id")
        if isinstance(mid, str) and mid:
            ids.add(mid)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def list_models_for_provider(
    provider: str,
    *,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Best-effort live lookup. Returns ``None`` if discovery is unavailable.

    *env* is preferred over ``os.environ`` so callers can pass an in-progress
    init snapshot (the user may have just typed a key that hasn't been
    exported yet).  When *env* is omitted the process environment is used.
    """
    env = env if env is not None else {}

    def _key(name: str) -> str:
        return (env.get(name) or os.getenv(name) or "").strip()

    if provider == "openai":
        key = _key("OPENAI_API_KEY")
        if not key:
            return None
        return list_openai_models(key, base_url=(env.get("OPENAI_BASE_URL") or "").strip() or None)
    if provider == "anthropic":
        key = _key("ANTHROPIC_API_KEY")
        if not key:
            return None
        return list_anthropic_models(key)
    if provider == "openrouter":
        return list_openrouter_models()
    # Bedrock requires the AWS SDK and signed requests; defer to the
    # custom-input UX so we don't drag boto3 into the init dependency
    # surface for a single picker call.
    return None
