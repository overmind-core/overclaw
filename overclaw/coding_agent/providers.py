"""LLM provider backed by LiteLLM — unified interface with tool-calling support.

Uses the project's existing litellm dependency so any model litellm supports
(OpenAI, Anthropic, Google, etc.) works without extra SDK installs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from overclaw.utils.llm import completion_kwargs_for_model

logger = logging.getLogger("overclaw.coding_agent.providers")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class LiteLLMProvider:
    """Provider that delegates to ``litellm.completion`` with tool support."""

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def model_id(self) -> str:
        return self.model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        base_kwargs: dict[str, Any] = {}
        if temperature is not None:
            base_kwargs["temperature"] = temperature

        kwargs = completion_kwargs_for_model(self.model, **base_kwargs)
        kwargs["model"] = self.model
        kwargs["messages"] = messages

        if tools:
            kwargs["tools"] = tools

        try:
            resp = litellm.completion(**kwargs)
        except Exception:
            logger.exception("LiteLLM completion failed")
            raise

        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage: dict[str, int] = {}
        if resp.usage:
            usage = {
                "input": resp.usage.prompt_tokens or 0,
                "output": resp.usage.completion_tokens or 0,
            }

        return ChatResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
