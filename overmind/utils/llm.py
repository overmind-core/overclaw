"""Provider-specific adjustments for LiteLLM completion calls."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from overmind import SpanType, attrs, set_tag
from overmind.utils.tracing import start_child_span

logger = logging.getLogger("overmind.llm")


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


def _provider_for(model: str) -> str:
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        return str(provider)
    except Exception:
        return "unknown"


def _summarize_messages(messages: list[dict]) -> tuple[int, int, str]:
    """Return ``(num_messages, total_chars, roles)`` for compact logging."""
    total_chars = 0
    roles: list[str] = []
    for msg in messages or []:
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        roles.append(role)
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or ""
                    if isinstance(text, str):
                        total_chars += len(text)
    return len(messages or []), total_chars, ",".join(roles)


def _response_preview(response: Any, limit: int = 160) -> str:
    """Best-effort single-line preview of ``response.choices[0].message``."""
    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None) or {}
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if not isinstance(content, str):
            return ""
        flat = " ".join(content.split())
        if len(flat) <= limit:
            return flat
        return flat[: limit - 1] + "…"
    except Exception:
        return ""


def _usage_tuple(response: Any) -> tuple[int, int, int]:
    """Return ``(prompt_tokens, completion_tokens, total_tokens)`` or zeros."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    try:
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            int(getattr(usage, "total_tokens", 0) or 0),
        )
    except Exception:
        return 0, 0, 0


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

    Every call is logged with model, provider, message shape, latency, token
    usage, and a truncated response preview so multi-thread pipelines remain
    debuggable from the log file alone.
    """
    provider = _provider_for(model)
    num_msgs, total_chars, roles = _summarize_messages(messages)
    num_tools = len(tools or [])
    kwarg_keys = ",".join(sorted(k for k in kwargs if k != "api_key"))

    logger.debug(
        f"llm_completion BEGIN model={model} provider={provider} messages={num_msgs} "
        f"chars={total_chars} roles={roles} tools={num_tools} kwargs=[{kwarg_keys}]"
    )
    # Wrap each LLM call in its own child span so it flushes to the backend
    # as soon as the call returns — long-running parent spans don't stall
    # progress visibility in the trace UI.
    with start_child_span("overmind_llm_completion", span_type=SpanType.LLM):
        set_tag(attrs.LLM_MODEL, model)
        set_tag("type", "llm_call")
        set_tag(attrs.LLM_PROVIDER, provider)
        set_tag(attrs.LLM_REQUEST_MESSAGE_COUNT, str(num_msgs))
        set_tag(attrs.LLM_REQUEST_MESSAGE_CHARS, str(total_chars))
        set_tag(attrs.LLM_REQUEST_TOOL_COUNT, str(num_tools))
        if kwarg_keys:
            set_tag(attrs.LLM_REQUEST_KWARGS, kwarg_keys)

        t0 = time.monotonic()
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools or None,
                **completion_kwargs_for_model(model, **kwargs),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            set_tag(attrs.LLM_ELAPSED_SECONDS, f"{elapsed:.3f}")
            set_tag(attrs.LLM_ERROR, type(exc).__name__)
            logger.exception(
                f"llm_completion FAIL  model={model} provider={provider} "
                f"elapsed={elapsed:.2f}s error={type(exc).__name__}"
            )
            raise

        elapsed = time.monotonic() - t0
        pt, ct, tt = _usage_tuple(response)
        preview = _response_preview(response)
        set_tag(attrs.LLM_ELAPSED_SECONDS, f"{elapsed:.3f}")
        set_tag(attrs.LLM_USAGE_PROMPT_TOKENS, str(pt))
        set_tag(attrs.LLM_USAGE_COMPLETION_TOKENS, str(ct))
        set_tag(attrs.LLM_USAGE_TOTAL_TOKENS, str(tt))
        logger.info(
            f"llm_completion OK    model={model} provider={provider} elapsed={elapsed:.2f}s "
            f"tokens_in={pt} tokens_out={ct} total={tt} preview={preview!r}"
        )
        return response
