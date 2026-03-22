"""Provider-specific adjustments for LiteLLM completion calls."""

from __future__ import annotations

from typing import Any

import litellm


def completion_kwargs_for_model(model: str, **kwargs: object) -> dict:
    """Build kwargs for ``litellm.completion``, applying all provider-specific rules.

    Rules applied:
    - OpenAI newer chat models reject ``temperature``; it is removed.
    - Anthropic models receive ``cache_control`` for prompt caching.

    If the provider cannot be resolved (unknown model id), kwargs are returned unchanged.
    """
    out: dict = dict(kwargs)
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        return out
    if provider == "openai":
        out.pop("temperature", None)
    if provider == "anthropic":
        out["cache_control"] = {"type": "ephemeral"}
    return out


def llm_completion(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs: object,
) -> Any:
    """Drop-in wrapper around ``litellm.completion`` with all provider rules applied.

    Use this instead of calling ``litellm.completion`` directly so that every
    call site automatically benefits from provider-specific adjustments
    (temperature stripping for OpenAI, prompt caching for Anthropic, etc.).
    """
    return litellm.completion(
        model=model,
        messages=messages,
        tools=tools or None,
        **completion_kwargs_for_model(model, **kwargs),
    )
