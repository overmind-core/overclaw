import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_sdk_state():
    """Reset SDK state before each test, restoring it afterwards.

    Each test in this module needs a clean slate so it can call init() and
    observe the result.  We save the pre-test state and put it back on
    teardown so that tests in *other* modules (which rely on the no-op
    tracer installed by conftest.py) are not broken when they run after
    these tests.
    """
    import overmind.tracing as sdk

    saved_initialized = sdk._initialized
    saved_tracer = sdk._tracer
    sdk._initialized = False
    sdk._tracer = None
    yield
    sdk._initialized = saved_initialized
    sdk._tracer = saved_tracer


@pytest.fixture
def mock_opentelemetry():
    """Mock all OpenTelemetry dependencies."""
    mock_fastapi_class = MagicMock()
    mock_openai_class = MagicMock()

    with (
        patch("overmind.tracing.TracerProvider") as mock_provider,
        patch("overmind.tracing.OTLPSpanExporter") as mock_exporter,
        patch("overmind.tracing.BatchSpanProcessor") as mock_processor,
        patch("overmind.tracing.trace") as mock_trace,
    ):
        # Set up tracer mock
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        yield {
            "provider": mock_provider,
            "exporter": mock_exporter,
            "processor": mock_processor,
            "trace": mock_trace,
            "tracer": mock_tracer,
        }


def test_sdk_init_configures_tracing(mock_opentelemetry):
    """Test that calling init configures OpenTelemetry correctly."""
    from overmind import tracing

    with (
        patch.object(tracing, "FastAPIInstrumentor", create=True) as mock_fastapi,
        patch.object(tracing, "OpenAIInstrumentor", create=True) as mock_openai,
    ):
        # Mock the dynamic imports
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry.instrumentation.fastapi": MagicMock(FastAPIInstrumentor=mock_fastapi),
                "opentelemetry.instrumentation.openai": MagicMock(OpenAIInstrumentor=mock_openai),
            },
        ):
            tracing.init(
                overmind_api_key="test_key",
                overmind_base_url="http://localhost:4318",
                service_name="test-service",
                environment="testing",
            )

    # Verify Exporter configuration
    mock_opentelemetry["exporter"].assert_called_with(
        endpoint="http://localhost:4318/api/v1/traces", headers={"X-Api-Key": "test_key"}
    )

    # Verify tracer provider was set
    mock_opentelemetry["trace"].set_tracer_provider.assert_called_once()


def test_sdk_init_only_once():
    """Test that init only runs once."""
    from overmind import tracing

    with (
        patch("overmind.tracing.TracerProvider"),
        patch("overmind.tracing.OTLPSpanExporter"),
        patch("overmind.tracing.BatchSpanProcessor"),
        patch("overmind.tracing.trace") as mock_trace,
    ):
        tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")
        tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")

        # Should only be called once
        assert mock_trace.set_tracer_provider.call_count == 1


def test_sdk_init_handles_missing_deps():
    """Test that init doesn't crash if optional instrumentation libraries are missing."""
    from overmind import tracing

    with (
        patch("overmind.tracing.TracerProvider"),
        patch("overmind.tracing.OTLPSpanExporter"),
        patch("overmind.tracing.BatchSpanProcessor"),
        patch("overmind.tracing.trace"),
    ):
        # Should not raise exception even if instrumentors fail to import
        tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")


def test_get_tracer_before_init():
    """Test that get_tracer raises if SDK not initialised."""
    from overmind import tracing

    with pytest.raises(RuntimeError, match="not initialised"):
        tracing.get_tracer()


def test_get_tracer_after_init(mock_opentelemetry):
    """Test that get_tracer returns tracer after init."""
    from overmind import tracing

    tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")

    tracer = tracing.get_tracer()
    assert tracer is not None


def test_set_user(mock_opentelemetry):
    """Test that set_user adds user attributes to current span."""
    from overmind import tracing

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_opentelemetry["trace"].get_current_span.return_value = mock_span

    tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")
    tracing.set_user(user_id="user123", email="test@example.com")

    mock_span.set_attribute.assert_any_call("user.id", "user123")
    mock_span.set_attribute.assert_any_call("user.email", "test@example.com")


def test_set_tag(mock_opentelemetry):
    """Test that set_tag adds custom attributes to current span."""
    from overmind import tracing

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_opentelemetry["trace"].get_current_span.return_value = mock_span

    tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")
    tracing.set_tag("tenant.id", "tenant123")

    mock_span.set_attribute.assert_called_with("tenant.id", "tenant123")


def test_capture_exception(mock_opentelemetry):
    """Test that capture_exception records exception on current span."""
    from overmind import tracing

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_opentelemetry["trace"].get_current_span.return_value = mock_span

    tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")

    test_exception = ValueError("test error")
    tracing.capture_exception(test_exception)

    mock_span.record_exception.assert_called_once_with(test_exception)


def test_service_name_from_env():
    """Test that service name can be set via environment variable."""
    from overmind import tracing

    with (
        patch("overmind.tracing.TracerProvider") as mock_provider,
        patch("overmind.tracing.OTLPSpanExporter"),
        patch("overmind.tracing.BatchSpanProcessor"),
        patch("overmind.tracing.trace"),
        patch("overmind.tracing.Resource") as mock_resource,
        patch.dict(os.environ, {"OVERMIND_SERVICE_NAME": "env-service"}),
    ):
        tracing.init(overmind_api_key="test_key", overmind_base_url="http://localhost:4318")

        # Check that Resource.create was called with the env service name
        call_args = mock_resource.create.call_args[0][0]
        assert call_args["service.name"] == "env-service"
