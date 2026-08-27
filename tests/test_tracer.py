"""
Unit tests for the observe decorator.
"""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode

from overmind.tracing import SpanType, observe, observe_safe


@pytest.fixture(autouse=True)
def reset_sdk_state():
    """Force-mark the SDK initialised so observe() doesn't call through.

    Tests here patch ``get_tracer`` and do not need a live SDK; the flag is
    restored so later tests on the same pytest-xdist worker keep the no-op
    tracer installed in ``tests/conftest.py``.
    """
    from overmind import tracing

    saved_initialized = tracing._initialized
    saved_tracer = tracing._tracer
    tracing._initialized = True
    yield
    tracing._initialized = saved_initialized
    tracing._tracer = saved_tracer


@pytest.fixture
def mock_tracer():
    """Create a mock tracer with span context manager."""
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    return mock_tracer, mock_span


def test_observe_sync_basic(mock_tracer):
    """Test basic synchronous function tracing."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def add_numbers(a: int, b: int):
            return a + b

        result = add_numbers(5, 3)

        assert result == 8
        (span_name,) = mock_tracer_obj.start_as_current_span.call_args.args
        assert span_name.endswith("add_numbers")
        assert mock_span.set_attribute.call_count >= 2
        mock_span.set_status.assert_called_once()
        assert mock_span.set_status.call_args[0][0].status_code == StatusCode.OK


def test_observe_with_custom_span_name(mock_tracer):
    """Test decorator with custom span name."""

    mock_tracer_obj, _mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe(span_name="custom_operation")
        def my_function(x: int):
            return x * 2

        result = my_function(10)

        assert result == 20
        mock_tracer_obj.start_as_current_span.assert_called_once()
        assert mock_tracer_obj.start_as_current_span.call_args.args == ("custom_operation",)


def test_observe_captures_inputs(mock_tracer):
    """Test that function inputs are captured."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def process_data(name: str, age: int, metadata: dict):
            return {"processed": True}

        process_data("Alice", 30, {"city": "NYC"})

        # Check that inputs were set
        input_calls = [c for c in mock_span.set_attribute.call_args_list if "inputs" in str(c)]
        assert len(input_calls) > 0


def test_observe_captures_outputs(mock_tracer):
    """Test that function outputs are captured."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def get_result():
            return {"status": "success", "value": 42}

        result = get_result()

        assert result == {"status": "success", "value": 42}
        # Check that outputs were set
        output_calls = [c for c in mock_span.set_attribute.call_args_list if "outputs" in str(c)]
        assert len(output_calls) > 0


def test_observe_handles_exceptions(mock_tracer):
    """Test that exceptions are properly recorded."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Check that exception was recorded
        mock_span.record_exception.assert_called_once()
        # Check that error status was set
        status_calls = [c for c in mock_span.set_status.call_args_list]
        assert len(status_calls) > 0
        # Verify error status
        error_call = status_calls[-1]
        assert error_call[0][0].status_code == StatusCode.ERROR


def test_observe_async(mock_tracer):
    """Test async function tracing."""
    import asyncio

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe(span_name="async_operation")
        async def async_add(a: int, b: int):
            await asyncio.sleep(0.01)
            return a + b

        result = asyncio.run(async_add(10, 20))

        assert result == 30
        mock_tracer_obj.start_as_current_span.assert_called_once()
        assert mock_tracer_obj.start_as_current_span.call_args.args == ("async_operation",)
        mock_span.set_status.assert_called_once()
        assert mock_span.set_status.call_args[0][0].status_code == StatusCode.OK


def test_observe_async_with_exception(mock_tracer):
    """Test async function exception handling."""
    import asyncio

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        async def async_fail():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError, match="Async error"):
            asyncio.run(async_fail())

        mock_span.record_exception.assert_called_once()
        status_calls = [c for c in mock_span.set_status.call_args_list]
        assert status_calls[-1][0][0].status_code == StatusCode.ERROR


def test_observe_async_cancellation_marks_cancelled(mock_tracer):
    """asyncio.CancelledError ends the span cancelled, not failed, and propagates."""
    import asyncio

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        async def cancelled_op():
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cancelled_op())

        attributes = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attributes["overmind.status"] == "cancelled"
        assert "overmind.error.type" not in attributes
        mock_span.record_exception.assert_called_once()


def test_observe_with_kwargs(mock_tracer):
    """Test function with keyword arguments."""

    mock_tracer_obj, _mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def greet(name: str, greeting: str = "Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")

        assert result == "Hi, World!"
        mock_tracer_obj.start_as_current_span.assert_called_once()


def test_observe_preserves_function_metadata(mock_tracer):
    """Test that function metadata is preserved."""

    mock_tracer_obj, _mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def documented_function(param: int) -> int:
            """This is a test function."""
            return param * 2

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function."


def test_observe_with_complex_types(mock_tracer):
    """Test tracing with complex data types."""

    mock_tracer_obj, _mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def process_complex(data: dict, items: list):
            return {"processed": len(items), "data_keys": list(data.keys())}

        result = process_complex({"key1": "value1", "key2": 123}, [1, 2, 3, 4, 5])

        assert result == {"processed": 5, "data_keys": ["key1", "key2"]}
        mock_tracer_obj.start_as_current_span.assert_called_once()


def test_observe_with_no_args(mock_tracer):
    """Test function with no arguments."""

    mock_tracer_obj, _mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def get_constant():
            return 42

        result = get_constant()

        assert result == 42
        mock_tracer_obj.start_as_current_span.assert_called_once()


def test_observe_with_positional_only_args(mock_tracer):
    """Test function with positional arguments."""

    mock_tracer_obj, _ = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe()
        def multiply(x, y):
            return x * y

        result = multiply(6, 7)

        assert result == 42
        mock_tracer_obj.start_as_current_span.assert_called_once()


def test_observe_skips_self_in_class_method(mock_tracer):
    """Test that self is not captured for instance methods."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        class TestClass:
            def __init__(self):
                self.value = 10

            @observe()
            def instance_method(self, x: int, y: int):
                return self.value + x + y

        obj = TestClass()
        result = obj.instance_method(5, 3)

        assert result == 18
        # Check that inputs were captured but self was skipped
        input_calls = [c for c in mock_span.set_attribute.call_args_list if "inputs" in str(c)]
        assert len(input_calls) > 0
        # Verify self is not in the captured inputs
        captured_inputs = str(input_calls[0])
        assert "self" not in captured_inputs or '"self"' not in captured_inputs


def test_observe_skips_cls_in_classmethod(mock_tracer):
    """Test that cls is not captured for class methods."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        class TestClass:
            class_value = 20

            @classmethod
            @observe()
            def class_method(cls, x: int, y: int):
                return cls.class_value + x + y

        result = TestClass.class_method(5, 3)

        assert result == 28
        # Check that inputs were captured but cls was skipped
        input_calls = [c for c in mock_span.set_attribute.call_args_list if "inputs" in str(c)]
        assert len(input_calls) > 0
        # Verify cls is not in the captured inputs
        captured_inputs = str(input_calls[0])
        assert "cls" not in captured_inputs or '"cls"' not in captured_inputs


def test_observe_async_class_method(mock_tracer):
    """Test async class method with self skipped."""
    import asyncio

    mock_tracer_obj, _ = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        class AsyncTestClass:
            def __init__(self):
                self.value = 15

            @observe()
            async def async_method(self, x: int):
                await asyncio.sleep(0.01)
                return self.value * x

        obj = AsyncTestClass()
        result = asyncio.run(obj.async_method(2))

        assert result == 30
        mock_tracer_obj.start_as_current_span.assert_called_once()


def test_observe_safe_does_not_capture_io(mock_tracer):
    """observe_safe never captures inputs/outputs."""

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe_safe()
        def secure_func(api_key: str, payload: dict):
            return {"ok": True}

        result = secure_func("sk-secret", {"a": 1})

        assert result == {"ok": True}
        mock_tracer_obj.start_as_current_span.assert_called_once()
        assert mock_tracer_obj.start_as_current_span.call_args.args[0].endswith("secure_func")
        assert mock_span.set_status.call_args[0][0].status_code == StatusCode.OK
        io_calls = [c for c in mock_span.set_attribute.call_args_list if "inputs" in str(c) or "outputs" in str(c)]
        assert io_calls == []


def test_observe_safe_async(mock_tracer):
    """observe_safe supports async functions and spans the real execution."""
    import asyncio

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe_safe(span_name="secure_async")
        async def async_fetch(token: str):
            await asyncio.sleep(0.01)
            return "data"

        assert asyncio.run(async_fetch("t0ken")) == "data"
        assert mock_tracer_obj.start_as_current_span.call_args.args == ("secure_async",)
        assert mock_span.set_status.call_args[0][0].status_code == StatusCode.OK


def test_observe_safe_async_exception_is_recorded(mock_tracer):
    """Async exceptions mark the span failed (regression: the old sync-only
    wrapper finalised the span before the coroutine ran)."""
    import asyncio

    mock_tracer_obj, mock_span = mock_tracer

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe_safe()
        async def async_fail(token: str):
            await asyncio.sleep(0.01)
            raise RuntimeError("secure failure")

        with pytest.raises(RuntimeError, match="secure failure"):
            asyncio.run(async_fail("t0ken"))

        mock_span.record_exception.assert_called_once()
        status_calls = list(mock_span.set_status.call_args_list)
        assert status_calls[-1][0][0].status_code == StatusCode.ERROR


def test_observe_safe_tool_metadata(mock_tracer):
    """observe_safe stamps tool metadata for SpanType.TOOL spans."""
    from overmind.attrs import TOOL_ARG_KEYS, TOOL_NAME

    mock_tracer_obj, mock_span = mock_tracer
    # Tool spans are environment evidence; a mock span has no real context to remember.
    mock_span.get_span_context.return_value.is_valid = False

    with patch("overmind.tracing.get_tracer", return_value=mock_tracer_obj):

        @observe_safe(type=SpanType.TOOL)
        def my_tool(secret: str, query: str):
            return "ok"

        my_tool("hunter2", "find things")

        set_attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert set_attrs.get(TOOL_NAME) == "my_tool"
        assert set_attrs.get(TOOL_ARG_KEYS) == ["secret", "query"]


def test_observe_safe_exported_from_package():
    """observe_safe is importable from the package root."""
    from overmind import observe_safe

    assert callable(observe_safe)
