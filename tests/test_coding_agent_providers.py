"""Regression guard: LiteLLMProvider must route through llm_completion().

Bug history: ``coding_agent/providers.py`` previously called
``litellm.completion`` directly, which meant every coding-agent step was
invisible to overmind tracing — every other LLM call site in the codebase
goes through :func:`overmind.utils.llm.llm_completion` (which opens a
``gen_ai.*`` child span).  This test pins the wiring so a future refactor
cannot silently drop tracing again.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from overmind.coding_agent.providers import LiteLLMProvider


def _fake_response(text: str = "ok") -> MagicMock:
    """Build a minimal duck-typed litellm response object."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 12
    usage.completion_tokens = 4
    usage.total_tokens = 16

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_chat_routes_through_llm_completion() -> None:
    """The coding agent must reach litellm via the traced wrapper."""
    with patch(
        "overmind.coding_agent.providers.llm_completion",
        return_value=_fake_response("hello"),
    ) as mocked:
        provider = LiteLLMProvider("openai/gpt-5-mini")
        result = provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
        )

    assert result.text == "hello"
    assert mocked.call_count == 1
    kwargs = mocked.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-5-mini"
    assert kwargs["temperature"] == 0.3
    assert kwargs["tools"] is None


def test_chat_passes_tools_through() -> None:
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    with patch(
        "overmind.coding_agent.providers.llm_completion",
        return_value=_fake_response(),
    ) as mocked:
        provider = LiteLLMProvider("openai/gpt-5-mini")
        provider.chat(messages=[{"role": "user", "content": "x"}], tools=tools)

    assert mocked.call_args.kwargs["tools"] == tools


def test_chat_omits_temperature_when_not_provided() -> None:
    with patch(
        "overmind.coding_agent.providers.llm_completion",
        return_value=_fake_response(),
    ) as mocked:
        provider = LiteLLMProvider("openai/gpt-5-mini")
        provider.chat(messages=[{"role": "user", "content": "x"}])

    assert "temperature" not in mocked.call_args.kwargs


def test_tool_call_arguments_are_parsed() -> None:
    """Tool-call decoding must still work after the routing change."""
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "read_file"
    tool_call.function.arguments = '{"path": "foo.py"}'

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tool_call]
    choice.finish_reason = "tool_calls"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None

    with patch("overmind.coding_agent.providers.llm_completion", return_value=resp):
        provider = LiteLLMProvider("openai/gpt-5-mini")
        result = provider.chat(messages=[{"role": "user", "content": "x"}])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "foo.py"}


def test_malformed_tool_arguments_fall_back_to_raw() -> None:
    """Malformed JSON in tool arguments must not crash the chat call."""
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "read_file"
    tool_call.function.arguments = "{not valid json"

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tool_call]
    choice.finish_reason = "tool_calls"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None

    with patch("overmind.coding_agent.providers.llm_completion", return_value=resp):
        provider = LiteLLMProvider("openai/gpt-5-mini")
        result = provider.chat(messages=[{"role": "user", "content": "x"}])

    assert result.tool_calls[0].arguments == {"raw": "{not valid json"}
