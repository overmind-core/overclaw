"""Provider-specific adjustments for LiteLLM completion calls."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from overmind import SpanType, attrs, set_tag
from overmind.genai_usage import compute_cost
from overmind.tracing import start_child_span

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


def _response_chars(response: Any) -> int | None:
    """Total characters of the response message content, or ``None``."""
    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None) or {}
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str):
            return len(content)
    except Exception:
        return None
    return None


def _finish_reason(response: Any) -> str | None:
    try:
        return getattr(response.choices[0], "finish_reason", None) or None
    except Exception:
        return None


def _usage_tokens(response: Any) -> tuple[int | None, int | None, int | None]:
    """Return ``(prompt, completion, total)`` tokens, each ``None`` when absent.

    Never coerces a missing value to ``0`` — the server distinguishes "no
    token data" from "zero tokens", and zero-filling would poison the
    per-agent rollup.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def _pick(*names: str) -> int | None:
        for name in names:
            raw = getattr(usage, name, None)
            if raw is None and isinstance(usage, dict):
                raw = usage.get(name)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    prompt = _pick("prompt_tokens", "input_tokens")
    completion = _pick("completion_tokens", "output_tokens")
    total = _pick("total_tokens")
    if total is None and (prompt or completion):
        total = (prompt or 0) + (completion or 0)
    return prompt, completion, total


def _provider_cost(response: Any, model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    """Best-effort USD cost for a completion.

    Priority: provider-reported cost (litellm ``_hidden_params`` /
    OpenRouter ``usage.cost``) → ``litellm.completion_cost`` → pricing table
    via :func:`overmind.genai_usage.compute_cost`.  ``None`` when unknown.
    """
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        reported = hidden.get("response_cost")
        if isinstance(reported, (int, float)) and reported > 0:
            return round(float(reported), 8)

    usage = getattr(response, "usage", None)
    usage_cost = getattr(usage, "cost", None)
    if usage_cost is None and isinstance(usage, dict):
        usage_cost = usage.get("cost")
    if isinstance(usage_cost, (int, float)) and usage_cost > 0:
        return round(float(usage_cost), 8)

    try:
        computed = litellm.completion_cost(completion_response=response)
        if isinstance(computed, (int, float)) and computed > 0:
            return round(float(computed), 8)
    except Exception:
        logger.debug("litellm.completion_cost failed for model=%s", model, exc_info=True)

    return compute_cost(model, prompt_tokens, completion_tokens)


def _stamp_request(model: str, provider: str, num_msgs: int, total_chars: int, num_tools: int, kwargs: dict) -> None:
    """Stamp request-side genai.* attributes on the current LLM span."""
    set_tag(attrs.LLM_MODEL, model)
    set_tag(attrs.OTEL_LLM_REQUEST_MODEL, model)
    set_tag(attrs.LLM_PROVIDER, provider)
    set_tag(attrs.OTEL_LLM_SYSTEM, provider)
    set_tag(attrs.LLM_REQUEST_MESSAGE_COUNT, num_msgs)
    set_tag(attrs.LLM_REQUEST_MESSAGE_CHARS, total_chars)
    set_tag(attrs.LLM_REQUEST_TOOL_COUNT, num_tools)
    kwarg_keys = ",".join(sorted(k for k in kwargs if k != "api_key"))
    if kwarg_keys:
        set_tag(attrs.LLM_REQUEST_KWARGS, kwarg_keys)
    if (temperature := kwargs.get("temperature")) is not None:
        set_tag(attrs.LLM_REQUEST_TEMPERATURE, temperature)
    if (max_tokens := kwargs.get("max_tokens")) is not None:
        set_tag(attrs.LLM_REQUEST_MAX_TOKENS, max_tokens)
    if (top_p := kwargs.get("top_p")) is not None:
        set_tag(attrs.LLM_REQUEST_TOP_P, top_p)


def _stamp_response(response: Any, model: str, elapsed: float) -> tuple[int | None, int | None, int | None]:
    """Stamp usage / cost / response-shape genai.* attributes. Returns token tuple."""
    prompt, completion, total = _usage_tokens(response)
    set_tag(attrs.LLM_ELAPSED_SECONDS, round(elapsed, 3))
    if prompt is not None:
        set_tag(attrs.LLM_PROMPT_TOKENS, prompt)
        set_tag(attrs.OTEL_LLM_USAGE_PROMPT_TOKENS, prompt)
    if completion is not None:
        set_tag(attrs.LLM_COMPLETION_TOKENS, completion)
        set_tag(attrs.OTEL_LLM_USAGE_COMPLETION_TOKENS, completion)
    if total is not None:
        set_tag(attrs.LLM_TOTAL_TOKENS, total)
        set_tag(attrs.OTEL_LLM_USAGE_TOTAL_TOKENS, total)

    cost = _provider_cost(response, model, prompt, completion)
    if cost is not None:
        set_tag(attrs.LLM_COST, cost)

    response_model = getattr(response, "model", None)
    if isinstance(response_model, str) and response_model:
        set_tag(attrs.LLM_RESPONSE_MODEL, response_model)
    chars = _response_chars(response)
    if chars is not None:
        set_tag(attrs.LLM_RESPONSE_MESSAGE_CHARS, chars)
    if finish := _finish_reason(response):
        set_tag(attrs.LLM_RESPONSE_FINISH_REASON, finish)

    return prompt, completion, total


def _streaming_completion(
    model: str,
    provider: str,
    messages: list[dict],
    tools: list[dict] | None,
    kwargs: dict,
    num_msgs: int,
    total_chars: int,
    num_tools: int,
) -> Any:
    """Trace a streamed completion, recording time-to-first-token.

    The child span is held open across generator consumption so the
    ``genai.time_to_first_token_seconds`` and reconstructed usage/cost land
    when the stream is exhausted (not when this function returns).
    """
    cm = start_child_span("overmind_llm_completion", span_type=SpanType.LLM)
    span = cm.__enter__()
    _stamp_request(model, provider, num_msgs, total_chars, num_tools, kwargs)
    set_tag(attrs.LLM_STREAMING, True)

    t0 = time.monotonic()
    try:
        stream = litellm.completion(
            model=model,
            messages=messages,
            tools=tools or None,
            **completion_kwargs_for_model(model, **kwargs),
        )
    except Exception as exc:
        set_tag(attrs.LLM_ELAPSED_SECONDS, round(time.monotonic() - t0, 3))
        set_tag(attrs.LLM_ERROR, type(exc).__name__)
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise

    def _traced_stream():
        chunks: list[Any] = []
        first = True
        try:
            for chunk in stream:
                if first:
                    span.set_attribute(attrs.LLM_TTFT_SECONDS, round(time.monotonic() - t0, 3))
                    first = False
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            set_tag(attrs.LLM_ERROR, type(exc).__name__)
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            elapsed = time.monotonic() - t0
            try:
                rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
            except Exception:
                rebuilt = None
                set_tag(attrs.LLM_ELAPSED_SECONDS, round(elapsed, 3))
            if rebuilt is not None:
                _stamp_response(rebuilt, model, elapsed)
            cm.__exit__(None, None, None)

    return _traced_stream()


def llm_completion(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs: object,
) -> Any:
    """Drop-in wrapper around ``litellm.completion`` with all provider rules applied.

    Use this instead of calling ``litellm.completion`` directly so that every
    call site automatically benefits from provider-specific adjustments
    (temperature stripping for OpenAI, prompt caching for Anthropic, etc.)
    and emits the canonical Overmind ``genai.*`` tracing attributes (model,
    provider, request params, token usage, cost, latency, response shape).

    Streaming (``stream=True``) is supported: the returned generator is
    wrapped so time-to-first-token and reconstructed usage/cost are recorded
    when the stream completes.
    """
    provider = _provider_for(model)
    num_msgs, total_chars, roles = _summarize_messages(messages)
    num_tools = len(tools or [])

    logger.debug(
        f"llm_completion BEGIN model={model} provider={provider} messages={num_msgs} "
        f"chars={total_chars} roles={roles} tools={num_tools} stream={bool(kwargs.get('stream'))}"
    )

    if kwargs.get("stream"):
        return _streaming_completion(model, provider, messages, tools, dict(kwargs), num_msgs, total_chars, num_tools)

    # Wrap each LLM call in its own child span so it flushes to the backend
    # as soon as the call returns — long-running parent spans don't stall
    # progress visibility in the trace UI.  The span type is stamped by
    # :func:`start_child_span` via ``SpanType.LLM``.
    with start_child_span("overmind_llm_completion", span_type=SpanType.LLM):
        _stamp_request(model, provider, num_msgs, total_chars, num_tools, dict(kwargs))

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
            set_tag(attrs.LLM_ELAPSED_SECONDS, round(elapsed, 3))
            set_tag(attrs.LLM_ERROR, type(exc).__name__)
            logger.exception(
                f"llm_completion FAIL  model={model} provider={provider} "
                f"elapsed={elapsed:.2f}s error={type(exc).__name__}"
            )
            raise

        elapsed = time.monotonic() - t0
        prompt, completion, total = _stamp_response(response, model, elapsed)
        preview = _response_preview(response)
        logger.info(
            f"llm_completion OK    model={model} provider={provider} elapsed={elapsed:.2f}s "
            f"tokens_in={prompt} tokens_out={completion} total={total} preview={preview!r}"
        )
        return response
