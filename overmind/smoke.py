"""Smoke-mode stubs: patch provider SDK clients one layer below the
Traceloop instrumentors so ``overmind.init()`` can be exercised end-to-end
(instrumentor spans still fire) without making real network calls.

Activated by setting ``OVERMIND_SMOKE=1`` before calling :func:`overmind.tracing.init`.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_patched: dict[str, Any] = {}


def _canned_text() -> str:
    return os.environ.get("OVERMIND_SMOKE_RESPONSE") or "[overmind-smoke] canned response"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _patch_openai() -> None:
    if not _installed("openai"):
        return
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    def _build(model: str) -> ChatCompletion:
        return ChatCompletion.model_construct(
            id="chatcmpl-overmind-smoke",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice.model_construct(
                    index=0,
                    finish_reason="stop",
                    logprobs=None,
                    message=ChatCompletionMessage.model_construct(role="assistant", content=_canned_text()),
                )
            ],
            usage=CompletionUsage.model_construct(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    # ponytail: stream=True returns a single fake chunk, not a real SSE
    # stream; good enough for a smoke test, not for streaming behaviour.
    def _fake_stream(model: str):
        yield _build(model)

    def _sync_create(self, *args, **kwargs):
        model = kwargs.get("model", "overmind-smoke-model")
        if kwargs.get("stream"):
            return _fake_stream(model)
        return _build(model)

    async def _async_create(self, *args, **kwargs):
        return _sync_create(self, *args, **kwargs)

    _patched["openai"] = (Completions.create, AsyncCompletions.create)
    Completions.create = _sync_create
    AsyncCompletions.create = _async_create


def _patch_anthropic() -> None:
    if not _installed("anthropic"):
        return
    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import Message, TextBlock, Usage

    def _build(model: str) -> Message:
        return Message.model_construct(
            id="msg-overmind-smoke",
            type="message",
            role="assistant",
            model=model,
            content=[TextBlock.model_construct(type="text", text=_canned_text())],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage.model_construct(input_tokens=10, output_tokens=20),
        )

    def _fake_stream(model: str):
        yield _build(model)

    def _sync_create(self, *args, **kwargs):
        model = kwargs.get("model", "overmind-smoke-model")
        if kwargs.get("stream"):
            return _fake_stream(model)
        return _build(model)

    async def _async_create(self, *args, **kwargs):
        return _sync_create(self, *args, **kwargs)

    _patched["anthropic"] = (Messages.create, AsyncMessages.create)
    Messages.create = _sync_create
    AsyncMessages.create = _async_create


def _patch_google_genai() -> None:
    if not _installed("google.genai"):
        return
    from google.genai.models import Models
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    def _build(model: str) -> GenerateContentResponse:
        return GenerateContentResponse.model_construct(
            candidates=[
                Candidate.model_construct(
                    content=Content.model_construct(
                        role="model", parts=[Part.model_construct(text=_canned_text())]
                    ),
                    finish_reason="STOP",
                )
            ],
            model_version=model,
        )

    def _generate_content(self, *args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "overmind-smoke-model")
        return _build(model)

    _patched["google.genai"] = (Models.generate_content,)
    Models.generate_content = _generate_content


def activate_smoke_mode() -> None:
    """Patch installed provider clients with canned responses. Idempotent."""
    if _patched:
        logger.debug("Smoke mode already active")
        return
    # A provider lib may be absent or an unsupported major version (e.g. legacy
    # openai<1 has no openai.resources); smoke mode still covers the rest.
    for patch in (_patch_openai, _patch_anthropic, _patch_google_genai):
        try:
            patch()
        except (ImportError, AttributeError) as exc:
            logger.debug("smoke patch skipped: %s", exc)


def deactivate_smoke_mode() -> None:
    """Restore any provider methods patched by :func:`activate_smoke_mode`."""
    if "openai" in _patched:
        from openai.resources.chat.completions import AsyncCompletions, Completions

        Completions.create, AsyncCompletions.create = _patched.pop("openai")
    if "anthropic" in _patched:
        from anthropic.resources.messages import AsyncMessages, Messages

        Messages.create, AsyncMessages.create = _patched.pop("anthropic")
    if "google.genai" in _patched:
        from google.genai.models import Models

        (Models.generate_content,) = _patched.pop("google.genai")
