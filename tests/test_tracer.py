"""Tests for overclaw.core.tracer — tracing infrastructure."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.tracer import (
    Span,
    Trace,
    Tracer,
    call_llm,
    call_tool,
    get_current_tracer,
    set_current_tracer,
)


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_finish_sets_end_time(self):
        span = Span(span_type="test", name="s1", start_time=time.time())
        assert span.end_time == 0.0
        span.finish()
        assert span.end_time > 0.0
        assert span.latency_ms > 0.0

    def test_defaults(self):
        span = Span(span_type="llm", name="gpt", start_time=1.0)
        assert span.metadata == {}
        assert span.error is None


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_to_dict(self):
        trace = Trace(trace_id="t1")
        d = trace.to_dict()
        assert d["trace_id"] == "t1"
        assert isinstance(d["spans"], list)

    def test_save(self, tmp_path):
        trace = Trace(trace_id="t2", input_data={"x": 1})
        path = str(tmp_path / "traces" / "t2.json")
        trace.save(path)
        loaded = json.loads(open(path).read())
        assert loaded["trace_id"] == "t2"
        assert loaded["input_data"] == {"x": 1}


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class TestTracer:
    def test_add_span(self):
        tracer = Tracer(trace_id="t3")
        span = Span(span_type="test", name="s", start_time=time.time())
        tracer.add_span(span)
        assert len(tracer.trace.spans) == 1

    def test_set_input_output(self):
        tracer = Tracer(trace_id="t4")
        tracer.set_input({"a": 1})
        tracer.set_output({"b": 2})
        assert tracer.trace.input_data == {"a": 1}
        assert tracer.trace.output_data == {"b": 2}

    def test_finish(self):
        tracer = Tracer(trace_id="t5")
        assert tracer.trace.end_time == 0.0
        tracer.finish()
        assert tracer.trace.end_time > 0.0
        assert tracer.trace.total_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Thread-local tracer
# ---------------------------------------------------------------------------


class TestThreadLocalTracer:
    def test_default_is_none(self):
        set_current_tracer(None)
        assert get_current_tracer() is None

    def test_set_and_get(self):
        tracer = Tracer(trace_id="t6")
        set_current_tracer(tracer)
        assert get_current_tracer() is tracer
        set_current_tracer(None)


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


class TestCallLlm:
    def _mock_response(self, content="hello", tokens=10, cost=0.001):
        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = tokens
        mock_resp.usage.completion_tokens = tokens
        mock_resp.usage.total_tokens = tokens * 2
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        mock_resp.choices[0].message.tool_calls = None
        return mock_resp

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_success_no_tracer(self, mock_lp, mock_tracer):
        set_current_tracer(None)
        mock_lp.completion.return_value = self._mock_response()
        mock_tracer.completion_cost.return_value = 0.001

        result = call_llm("test-model", [{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "hello"

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_success_with_tracer(self, mock_lp, mock_tracer):
        tracer = Tracer(trace_id="t7")
        set_current_tracer(tracer)
        mock_lp.completion.return_value = self._mock_response()
        mock_tracer.completion_cost.return_value = 0.005

        call_llm("model", [{"role": "user", "content": "test"}])

        assert len(tracer.trace.spans) == 1
        span = tracer.trace.spans[0]
        assert span.span_type == "llm_call"
        assert span.error is None
        assert tracer.trace.total_tokens == 20
        set_current_tracer(None)

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_error_propagated(self, mock_lp, mock_tracer):
        set_current_tracer(None)
        mock_lp.completion.side_effect = RuntimeError("API down")

        with pytest.raises(RuntimeError, match="API down"):
            call_llm("model", [{"role": "user", "content": "hi"}])

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_error_span_recorded(self, mock_lp, mock_tracer):
        tracer = Tracer(trace_id="t8")
        set_current_tracer(tracer)
        mock_lp.completion.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            call_llm("model", [{"role": "user", "content": "hi"}])

        assert len(tracer.trace.spans) == 1
        assert tracer.trace.spans[0].error == "fail"
        set_current_tracer(None)

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_with_tools(self, mock_lp, mock_tracer):
        set_current_tracer(None)
        mock_resp = self._mock_response()
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = '{"q": "test"}'
        mock_resp.choices[0].message.tool_calls = [tc]
        mock_lp.completion.return_value = mock_resp
        mock_tracer.completion_cost.return_value = 0.0

        result = call_llm(
            "model",
            [{"role": "user", "content": "hi"}],
            tools=[{"function": {"name": "search"}}],
        )
        assert result is not None

    @patch("overclaw.core.tracer.litellm")
    @patch("overclaw.utils.llm.litellm")
    def test_cost_exception_handled(self, mock_lp, mock_tracer):
        set_current_tracer(None)
        mock_lp.completion.return_value = self._mock_response()
        mock_tracer.completion_cost.side_effect = Exception("cost error")

        result = call_llm("model", [{"role": "user", "content": "hi"}])
        assert result is not None


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


class TestCallTool:
    def test_success(self):
        set_current_tracer(None)
        fn = MagicMock(return_value={"result": 42})
        result = call_tool("my_tool", {"x": 1}, fn)
        assert result == {"result": 42}
        fn.assert_called_once_with(x=1)

    def test_with_tracer(self):
        tracer = Tracer(trace_id="t9")
        set_current_tracer(tracer)
        fn = MagicMock(return_value="ok")
        call_tool("tool", {"a": "b"}, fn)

        assert len(tracer.trace.spans) == 1
        span = tracer.trace.spans[0]
        assert span.span_type == "tool_call"
        assert span.name == "tool"
        assert span.error is None
        set_current_tracer(None)

    def test_error_propagated(self):
        set_current_tracer(None)
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            call_tool("tool", {}, fn)

    def test_error_span_recorded(self):
        tracer = Tracer(trace_id="t10")
        set_current_tracer(tracer)
        fn = MagicMock(side_effect=TypeError("oops"))

        with pytest.raises(TypeError):
            call_tool("tool", {"k": "v"}, fn)

        assert tracer.trace.spans[0].error == "oops"
        set_current_tracer(None)
