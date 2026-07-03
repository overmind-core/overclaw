"""Tests for the lifecycle / state helpers in :mod:`overmind.tracing`.

Helpers covered
---------------
* :func:`overmind.tracing.set_workflow_name`
* :func:`overmind.tracing.set_agent_name`
* :func:`overmind.tracing.set_conversation_id`
* :func:`overmind.tracing.capture_exception`
* :func:`overmind.tracing.force_flush_traces`
* :func:`overmind.tracing.set_tag`
* :func:`overmind.tracing.set_user`
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overmind.tracing import (
    capture_exception,
    force_flush_traces,
    set_agent_name,
    set_conversation_id,
    set_tag,
    set_user,
    set_workflow_name,
)


@pytest.fixture
def recording_span():
    """Patch ``trace.get_current_span`` to a fresh recording-mock span."""
    span = MagicMock()
    span.is_recording.return_value = True
    with patch("overmind.tracing.trace.get_current_span", return_value=span):
        yield span


@pytest.fixture
def ended_span():
    span = MagicMock()
    span.is_recording.return_value = False
    with patch("overmind.tracing.trace.get_current_span", return_value=span):
        yield span


class TestCaptureException:
    def test_records_exception_and_sets_error_status(self, recording_span):
        exc = ValueError("boom")
        capture_exception(exc)
        recording_span.record_exception.assert_called_once_with(exc)
        recording_span.set_status.assert_called_once()

    def test_silent_on_ended_span(self, ended_span):
        capture_exception(ValueError("boom"))
        ended_span.record_exception.assert_not_called()
        ended_span.set_status.assert_not_called()


class TestSetTagGuards:
    def test_ignored_when_span_ended(self, ended_span):
        set_tag("foo", "bar")
        ended_span.set_attribute.assert_not_called()


class TestContextHelpers:
    """Workflow / agent / conversation helpers attach to the OTel context."""

    @patch("overmind.tracing.attach")
    @patch("overmind.tracing.set_value", side_effect=lambda key, value: (key, value))
    def test_set_workflow_name_attaches(self, mock_set_value, mock_attach):
        set_workflow_name("checkout-flow")
        mock_set_value.assert_called_once()
        assert "checkout-flow" in mock_set_value.call_args.args
        mock_attach.assert_called_once()

    @patch("overmind.tracing.attach")
    @patch("overmind.tracing.set_value", side_effect=lambda key, value: (key, value))
    def test_set_agent_name_attaches(self, mock_set_value, mock_attach):
        set_agent_name("my-agent")
        assert "my-agent" in mock_set_value.call_args.args
        mock_attach.assert_called_once()

    @patch("overmind.tracing.attach")
    @patch("overmind.tracing.set_value", side_effect=lambda key, value: (key, value))
    def test_set_conversation_id_attaches(self, mock_set_value, mock_attach):
        set_conversation_id("conv-123")
        assert "conv-123" in mock_set_value.call_args.args
        mock_attach.assert_called_once()


class TestSetUser:
    def test_writes_user_attributes(self, recording_span):
        set_user("user-1", email="u@example.com", username="u")
        keys = {c.args[0] for c in recording_span.set_attribute.call_args_list}
        assert {"user.id", "user.email", "user.username"} <= keys


class TestForceFlushTraces:
    def test_no_op_when_provider_lacks_force_flush(self):
        class _Stub:
            pass

        with patch("overmind.tracing.trace.get_tracer_provider", return_value=_Stub()):
            force_flush_traces(timeout_millis=500)

    def test_calls_force_flush_when_provider_supports_it(self):
        provider = MagicMock()
        with patch("overmind.tracing.trace.get_tracer_provider", return_value=provider):
            force_flush_traces(timeout_millis=750)
        provider.force_flush.assert_called_once_with(timeout_millis=750)


class TestTracingAll:
    """Regression guard for the public ``overmind.tracing.__all__`` surface."""

    def test_all_contains_core_lifecycle_helpers(self):
        import overmind.tracing as tr

        expected = {
            "capture_exception",
            "enable_tracing",
            "force_flush_traces",
            "init",
            "observe",
            "set_agent_name",
            "set_conversation_id",
            "set_tag",
            "set_user",
            "set_workflow_name",
            "start_span",
        }
        missing = expected - set(tr.__all__)
        assert missing == set(), f"Missing from tracing.__all__: {missing}"

    def test_all_names_resolve(self):
        import overmind.tracing as tr

        for name in tr.__all__:
            assert hasattr(tr, name), f"__all__ lists '{name}' but module has no attribute"
