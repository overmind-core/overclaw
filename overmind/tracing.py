"""Overmind SDK tracing primitives.

This module is the public surface of the Overmind tracer:

* :func:`init` wires an OTLP exporter to the user's Overmind project,
  enables any vendor auto-instrumentations they ask for, and attaches
  a remote parent span when the process is spawned by the optimizer
  (so subprocess traces stitch into one tree).
* :func:`observe` / :func:`start_span` — decorator + context manager
  for hand-rolled spans inside an agent.  Both stamp the canonical
  ``overmind.span.type`` and ``overmind.status`` attributes so the
  Overmind backend can render them without parsing OTel internals.
* :func:`set_tag`, :func:`set_user`, :func:`capture_exception` —
  Sentry-style helpers that operate on the current span.

Attribute keys are defined in :mod:`overmind.attrs`; never hardcode
``overmind.*`` strings here.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from pathlib import PurePath
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from rich.console import Console

from overmind import attrs
from overmind.utils.io import read_api_key_masked

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

_strict_mode = os.environ.get("OVERMIND_STRICT_MODE", "false").lower() == "true"

# Global state to track initialization
_initialized = False
_tracer: trace.Tracer | None = None
_providers: set[str] = set()
console = Console()

# ---------------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.overmindlab.ai"


def get_api_settings(
    overmind_api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    overmind_api_key = overmind_api_key or os.getenv("OVERMIND_API_KEY")
    base_url = base_url or os.getenv("OVERMIND_API_URL") or DEFAULT_BASE_URL

    # Avoid prompting for key if running as library or during tests
    # Detect (roughly) if running under pytest or other test envs
    _in_test = "PYTEST_CURRENT_TEST" in os.environ or any("pytest" in arg for arg in sys.argv)
    # Also don't prompt if running as a non-interactive script
    _interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if not overmind_api_key:
        if _in_test or not _interactive:
            # If testing, never read or prompt for the key, just fail immediately
            raise RuntimeError("Missing OVERMIND_API_KEY. Set the environment variable to use Overmind services.")

        console.print(
            "\n[bold red]Missing OVERMIND_API_KEY.[/bold red]"
            "\n[dim]To access Overmind services, you need an API key.[/dim]"
            "\n[green]Visit[/green] [underline]https://console.overmindlab.ai/projects[/underline] [green]to create your API key.[/green]"
        )
        console.print("\nPlease paste your API key here: [bold]ovr_Xxx[/bold]")
        overmind_api_key = read_api_key_masked("OVERMIND_API_KEY")

        if not overmind_api_key:
            console.print("\n[bold red]No API key provided. Unable to continue. Exiting.[/bold red]\n")
            sys.exit(1)
        os.environ["OVERMIND_API_KEY"] = overmind_api_key
        console.print("\n[bold green]API key set successfully for this session.[/bold green]\n")

    return overmind_api_key, base_url.rstrip("/")


def serialize_dataclass(obj: Any) -> Any:
    """
    Recursively serialize dataclass and nested dataclasses into
    JSON-serializable objects (dicts, lists, primitives).

    Falls back to str(obj) if not serializable.
    Handles nested dataclasses, lists/tuples/sets of dataclasses, and dicts.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = serialize(value)
        return result
    elif isinstance(obj, (list, tuple, set)):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        # Only serialize keys if they're strings or basic types
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback: try to get __dict__, else use str
        if hasattr(obj, "__dict__"):
            return serialize(vars(obj))
        return str(obj)


def _default_serializer(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return serialize_dataclass(obj)
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            logger.exception("Error serializing object")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


def serialize(obj) -> str:
    try:
        raw = json.dumps(obj, default=_default_serializer, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        raw = repr(obj)

    return raw


# ---------------------------------------------------------------------------
# Provider auto-instrumentation
# ---------------------------------------------------------------------------


def enable_agno():
    name, module = "agno", "agno"
    global _providers
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if importlib.util.find_spec(module) is None:
        if _strict_mode:
            raise ImportError(f"{module} is not installed. Please install it with `pip install {module}`.")
        logger.warning(f"{module} is not installed. Please install it with `pip install {module}`.")
        return

    from opentelemetry.instrumentation.agno import AgnoInstrumentor

    AgnoInstrumentor().instrument()
    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def enable_openai():
    name, module = "openai", "openai"
    global _providers
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if importlib.util.find_spec(module) is None:
        if _strict_mode:
            raise ImportError(f"{module} is not installed. Please install it with `pip install {module}`.")
        logger.warning(f"{module} is not installed. Please install it with `pip install {module}`.")
        return

    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def enable_anthropic():
    name, module = "anthropic", "anthropic"
    global _providers
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if importlib.util.find_spec(module) is None:
        if _strict_mode:
            raise ImportError(f"{module} is not installed. Please install it with `pip install {module}`.")
        logger.warning(f"{module} is not installed. Please install it with `pip install {module}`.")
        return

    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    AnthropicInstrumentor().instrument()

    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def enable_google_genai():
    name, module = "google", "google.genai"

    global _providers
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if importlib.util.find_spec(module) is None:
        module = module.replace(".", "-")
        if _strict_mode:
            raise ImportError(f"{module} is not installed. Please install it with `pip install {module}`.")
        logger.warning(f"{module} is not installed. Please install it with `pip install {module}`.")
        return

    from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

    GoogleGenerativeAiInstrumentor().instrument()

    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def enable_tracing(providers: list[str] | None = None):
    if providers == []:
        # if no providers are provided, enable all supported providers
        providers = ["openai", "anthropic", "google", "agno"]
    logger.info(f"Enabling tracing for providers: {providers}")

    if providers is None:
        return
    if "agno" in providers:
        enable_agno()
    if "openai" in providers:
        enable_openai()
    if "anthropic" in providers:
        enable_anthropic()
    if "google" in providers:
        enable_google_genai()


# Context-propagation keys.  We use the canonical ``overmind.*``
# attribute string as the context key for resources that show up on
# every span (currently just the agent name, which lives in
# :data:`overmind.attrs.AGENT_NAME`) so the on-start processor can
# stamp the same key without re-mapping.  Workflow + conversation
# labels don't have a 1:1 span-attribute counterpart, so they use
# their own dotted symbolic names.
_CTX_KEY_WORKFLOW_NAME = "overmind.workflow.name"
_CTX_KEY_CONVERSATION_ID = "overmind.conversation.id"


def _span_processor_on_start(span: trace.Span, parent_context: trace.Context | None = None):
    """Stamp every new span with the ambient agent / workflow context.

    These keys are written into the OTel context by :func:`set_agent_name` /
    :func:`set_workflow_name` (or, for the CLI, by ``cli.main`` directly)
    so every child span automatically carries the resource identifiers
    the Overmind backend needs to attach the span to an :class:`Agent`
    row — no per-span :func:`set_tag` calls required.
    """
    if value := get_value(_CTX_KEY_WORKFLOW_NAME):
        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, str(value))
    if agent_name := get_value(attrs.AGENT_NAME):
        span.set_attribute(attrs.AGENT_NAME, str(agent_name))
    if conversation_id := get_value(_CTX_KEY_CONVERSATION_ID):
        span.set_attribute("conversation.id", str(conversation_id))


# ---------------------------------------------------------------------------
# Remote parent context propagation (subprocess / distributed tracing)
# ---------------------------------------------------------------------------


def _attach_remote_parent_if_present() -> None:
    """Attach a remote parent span from the ``TRACEPARENT`` environment variable.

    The overmind optimizer injects ``TRACEPARENT`` (W3C Trace Context format)
    into every agent subprocess before spawning it.  Calling this function
    immediately after the TracerProvider is registered makes every OTel span
    started in the subprocess a child of the optimizer's current span, so all
    per-case evaluation runs appear under a single unified parent trace.

    Safe to call when ``TRACEPARENT`` is absent — no-op in that case.
    """
    raw = os.environ.get("TRACEPARENT") or os.environ.get("OTEL_TRACEPARENT")
    if not raw:
        return
    try:
        from opentelemetry import context as _ctx
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        propagator = TraceContextTextMapPropagator()
        remote_ctx = propagator.extract(carrier={"traceparent": raw.strip()})
        _ctx.attach(remote_ctx)
        logger.debug(f"Attached remote parent context from TRACEPARENT: {raw}")
    except Exception as exc:
        logger.debug(f"Could not attach remote parent context: {exc}")


# ---------------------------------------------------------------------------
# SDK initialization
# ---------------------------------------------------------------------------


def init(
    overmind_api_key: str | None = None,
    *,
    service_name: str | None = None,
    environment: str | None = None,
    providers: list[str] | None = None,
    overmind_base_url: str | None = None,
):
    """
    Initialize the Overmind SDK for automatic monitoring.

    Example:
        import overmind
        overmind.init(service_name="my-backend", environment="production", providers=["openai", "anthropic", "google", "agno"])

    Args:
        overmind_api_key: Your Overmind API key. If not provided, uses OVERMIND_API_KEY env var.
        service_name: Name of your service (appears in traces). Defaults to OVERMIND_SERVICE_NAME
                      env var or "unknown-service".
        environment: Environment name (e.g., "production", "staging"). Defaults to
                     OVERMIND_ENVIRONMENT env var or "development".
        providers: List of providers to trace. Supported values: "openai", "anthropic", "google", "agno".
        overmind_base_url: Base URL for traces. If not provided, uses OVERMIND_API_URL env var.
    """
    global _initialized, _tracer

    if _initialized:
        # user can call init again with different providers, so we should not skip
        # there is no such thing as remove initialization
        logger.debug(f"Overmind SDK already initialized, reinitializing with providers: {providers}")
        enable_tracing(providers)
        return

    environment = (
        environment or os.environ.get("OVERMIND_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "development"
    )

    overmind_api_key, overmind_base_url = get_api_settings(overmind_api_key, overmind_base_url)

    endpoint = f"{overmind_base_url}/api/v1/traces"

    # Configure OpenTelemetry Provider with rich resource attributes
    from overmind import __version__ as _SDK_VERSION

    resource = Resource.create({
        "service.name": service_name or os.environ.get("OVERMIND_SERVICE_NAME") or "overmind-telemetry",
        "service.version": os.environ.get("SERVICE_VERSION", "unknown"),
        "deployment.environment": environment,
        "overmind.sdk.name": "overmind-python",
        "overmind.sdk.version": _SDK_VERSION,
    })

    provider = TracerProvider(resource=resource)

    # Configure OTLP Exporter
    headers = {"X-Api-Key": overmind_api_key}

    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

    # Tighten the batch flush cadence so closed child spans show up in the
    # backend within ~2s instead of the OTel default 5s.  Long-running
    # workflow spans rely on this to stream progress while still open.
    schedule_delay_millis = int(os.environ.get("OVERMIND_SPAN_FLUSH_INTERVAL_MS", "2000"))
    max_export_batch_size = int(os.environ.get("OVERMIND_SPAN_MAX_EXPORT_BATCH_SIZE", "256"))
    span_processor = BatchSpanProcessor(
        otlp_exporter,
        schedule_delay_millis=schedule_delay_millis,
        max_export_batch_size=max_export_batch_size,
    )
    provider.add_span_processor(span_processor)
    span_processor.on_start = _span_processor_on_start

    # Set global Trace Provider
    trace.set_tracer_provider(provider)

    # Store tracer for custom spans
    _tracer = trace.get_tracer("overmind", _SDK_VERSION)
    enable_tracing(providers)

    # Distributed tracing: if the process was spawned by the overmind
    # optimizer (or any other orchestrator) with a W3C TRACEPARENT env var,
    # attach it as the ambient OTel context so every span created in this
    # process becomes a child of the parent optimizer span — forming a single
    # unified trace across subprocess boundaries.
    _attach_remote_parent_if_present()

    _initialized = True
    logger.info(f"Overmind SDK initialized: service={service_name}, environment={environment}")


def get_tracer() -> trace.Tracer:
    """
    Get the Overmind tracer for creating custom spans.

    Example:
        tracer = overmind.get_tracer()
        with tracer.start_as_current_span("my-operation") as span:
            span.set_attribute("user.id", user_id)
            # ... your code ...

    Returns:
        OpenTelemetry Tracer instance.

    Raises:
        RuntimeError: If SDK not initialized.
    """
    if not _initialized or _tracer is None:
        raise RuntimeError("Overmind SDK not initialized. Call overmind.init() first.")
    return _tracer


# ---------------------------------------------------------------------------
# Span attribute helpers
# ---------------------------------------------------------------------------


def set_user(user_id: str, email: str | None = None, username: str | None = None) -> None:
    """
    Associate current trace with a user (like Sentry's set_user).

    Call this in your request handler to tag traces with user info.

    Example:
        @app.middleware("http")
        async def add_user_context(request: Request, call_next):
            if request.state.user:
                overmind.set_user(user_id=request.state.user.id)
            return await call_next(request)

    Args:
        user_id: Unique user identifier.
        email: Optional user email.
        username: Optional username.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("user.id", user_id)
        if email:
            span.set_attribute("user.email", email)
        if username:
            span.set_attribute("user.username", username)


def _coerce_attribute_value(value: Any) -> Any:
    """Project *value* onto an OTel-compatible attribute value.

    OTel attribute values must be primitives or sequences of strings.
    Anything richer (dicts, mixed-type lists, dataclasses, pydantic
    models …) is serialised to a JSON string so the receiving end can
    round-trip it with ``json.loads`` instead of getting a Python
    ``repr`` blob.  ``None`` becomes the empty string for consistency
    with the OTel SDK's own rejection of ``None`` values.
    """
    if value is None:
        return ""
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return list(value)
        try:
            return json.dumps(value, default=_default_serializer, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=_default_serializer, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _safe_set_attribute(otel_span, key: str, value: Any) -> None:
    """Set *key* / *value* on *otel_span* via :func:`_coerce_attribute_value`."""
    otel_span.set_attribute(key, _coerce_attribute_value(value))


def set_tag(key: str, value) -> None:
    """Add a custom tag to the current span.

    Accepts ``str`` / ``int`` / ``float`` / ``bool`` / ``list[str]``
    natively; richer values (dict, mixed-type list, dataclass, pydantic
    model, …) are JSON-encoded so the OTLP ingest can round-trip them
    with ``json.loads``.

    Example::

        overmind.set_tag("feature.flag", "new-checkout-flow")
        overmind.set_tag("iteration", 3)
        overmind.set_tag("score", 85.2)
        overmind.set_tag("overmind.setup.scope", {"optimizable_paths": [...]})
    """
    span = trace.get_current_span()
    if not span.is_recording():
        logger.debug("set_tag(%s=…) ignored: current span has ended %s", key, span)
        return
    _safe_set_attribute(span, key, value)


def capture_exception(exception: Exception) -> None:
    """
    Record an exception on the current span.

    Example:
        try:
            risky_operation()
        except Exception as e:
            overmind.capture_exception(e)
            raise

    Args:
        exception: The exception to record.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def set_workflow_name(workflow_name: str) -> None:
    """Attach a Traceloop-compatible workflow label to every subsequent span.

    Stored in the OTel context so child spans pick it up automatically;
    the on-start processor copies it onto each span as
    ``SpanAttributes.TRACELOOP_WORKFLOW_NAME``.
    """
    attach(set_value(_CTX_KEY_WORKFLOW_NAME, workflow_name))


def set_agent_name(agent_name: str) -> None:
    """Bind the current OTel context to *agent_name*.

    Once attached, every span created downstream is stamped with
    ``overmind.agent.name`` by the on-start processor, which is what
    lets the Overmind backend route the trace to the right
    :class:`Agent` record without per-call tagging.
    """
    attach(set_value(attrs.AGENT_NAME, agent_name))


def set_conversation_id(conversation_id: str) -> None:
    """Tag downstream spans with a stable ``conversation.id``.

    Useful for chat agents where multiple traces belong to the same
    user-visible session; the backend can group them in the UI.
    """
    attach(set_value(_CTX_KEY_CONVERSATION_ID, conversation_id))


# ---------------------------------------------------------------------------
# Span types and decorators
# ---------------------------------------------------------------------------


class SpanType(str, Enum):
    FUNCTION = "function"
    ENTRY_POINT = "entry_point"
    WORKFLOW = "workflow"
    TOOL = "tool_call"
    LLM = "llm_call"


# Type names whose instances should never be serialised into span
# attributes (UI helpers, internal OTel handles, …).  Matching by name
# avoids importing rich / opentelemetry just for an isinstance check.
_SKIP_INPUT_TYPES = frozenset({
    "Console",
    "Progress",
    "Live",
    "Table",
    "Panel",
    "TracerProvider",
    "Tracer",
    "Span",
})


def _should_skip_value(value: Any) -> bool:
    return type(value).__name__ in _SKIP_INPUT_TYPES


def _prepare_for_otel(value: Any) -> Any:
    """Normalise *value* into something :func:`serialize` can handle.

    Primitives pass through untouched; UI helpers collapse to a tag
    like ``"<Console>"``; pydantic models use ``model_dump``; paths
    stringify; everything else falls back to ``repr``-style coercion.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if _should_skip_value(value):
        return f"<{type(value).__name__}>"
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return str(value)
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, PurePath):
        return str(value)
    return str(value)


def _stamp_span_metadata(otel_span, span_type: SpanType) -> None:
    """Stamp the canonical Overmind metadata onto a freshly-opened span.

    Currently just :data:`overmind.attrs.SPAN_TYPE`, but centralised so
    future additions (sdk version, command, …) land in one place.
    OTel already records the span name on the proto, so we don't
    duplicate it as an attribute.
    """
    otel_span.set_attribute(attrs.SPAN_TYPE, span_type.value)


def _finalize_span(
    otel_span,
    exc: BaseException | None,
    start_monotonic: float,
) -> None:
    """Stamp lifecycle status + duration as the span closes.

    Called from both the decorator and the context-manager paths so
    the OTLP ingest sees the same shape regardless of which surface
    the caller used.  *exc* is ``None`` on success; otherwise the
    exception being propagated.
    """
    duration = max(0.0, time.monotonic() - start_monotonic)
    otel_span.set_attribute(attrs.DURATION_SECONDS, duration)

    if exc is None:
        otel_span.set_attribute(attrs.STATUS, "success")
        otel_span.set_status(Status(StatusCode.OK))
        return

    if isinstance(exc, KeyboardInterrupt):
        otel_span.set_attribute(attrs.STATUS, "cancelled")
        otel_span.record_exception(exc)
        otel_span.set_status(Status(StatusCode.ERROR, "Interrupted by user (KeyboardInterrupt)"))
        return

    otel_span.set_attribute(attrs.STATUS, "failed")
    otel_span.set_attribute(attrs.ERROR_TYPE, type(exc).__name__)
    otel_span.set_attribute(attrs.ERROR_MESSAGE, str(exc)[:1024])
    otel_span.record_exception(exc)
    otel_span.set_status(Status(StatusCode.ERROR, str(exc)))


def _capture_inputs(
    otel_span,
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> None:
    """Serialise *args* / *kwargs* and attach them as ``inputs``.

    Skips the leading ``self`` / ``cls`` of bound methods and any
    argument whose type name is in :data:`_SKIP_INPUT_TYPES`.  Errors
    in serialisation never propagate — capture is best-effort
    instrumentation, not part of the wrapped call.
    """
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        is_method = bool(param_names) and param_names[0] in ("self", "cls")
        start_idx = 1 if is_method else 0

        inputs: dict[str, Any] = {}
        for i, arg in enumerate(args[start_idx:], start=start_idx):
            if _should_skip_value(arg):
                continue
            param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
            inputs[param_name] = _prepare_for_otel(arg)
        for key, value in kwargs.items():
            if _should_skip_value(value):
                continue
            inputs[key] = _prepare_for_otel(value)

        otel_span.set_attribute("inputs", serialize(inputs))
    except Exception:
        logger.debug("observe(): input capture failed for %s", func.__name__, exc_info=True)


def _capture_output(otel_span, result: Any) -> None:
    """Serialise *result* and attach it as ``outputs`` (best-effort)."""
    try:
        otel_span.set_attribute("outputs", serialize(_prepare_for_otel(result)))
    except Exception:
        logger.debug("observe(): output capture failed", exc_info=True)


def observe(
    span_name: str | None = None,
    type: SpanType = SpanType.FUNCTION,
) -> Callable[[Callable], Callable]:
    """Decorator that traces a function with OpenTelemetry.

    Wraps the call in a span named *span_name* (defaults to the
    function name), captures positional / keyword arguments as
    ``inputs`` and the return value as ``outputs``, and stamps the
    canonical :data:`overmind.attrs.SPAN_TYPE`,
    :data:`overmind.attrs.STATUS`, and
    :data:`overmind.attrs.DURATION_SECONDS` attributes on exit.

    Supports both sync and async functions — the wrapper picks the
    right runtime via :func:`inspect.iscoroutinefunction`.
    """

    def decorator(func: Callable) -> Callable:
        name = span_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(name) as otel_span:
                    _stamp_span_metadata(otel_span, type)
                    _capture_inputs(otel_span, func, args, kwargs)
                    start = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        _finalize_span(otel_span, exc, start)
                        raise
                    _capture_output(otel_span, result)
                    _finalize_span(otel_span, None, start)
                    return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as otel_span:
                _stamp_span_metadata(otel_span, type)
                _capture_inputs(otel_span, func, args, kwargs)
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    _finalize_span(otel_span, exc, start)
                    raise
                _capture_output(otel_span, result)
                _finalize_span(otel_span, None, start)
                return result

        return sync_wrapper

    return decorator


@contextmanager
def start_span(
    name: str,
    span_type: SpanType = SpanType.FUNCTION,
    attributes: dict[str, Any] | None = None,
):
    """Context manager that opens a child span under the current trace.

    The companion to :func:`observe` for loops / conditional blocks
    where a decorator isn't practical.  Stamps the same canonical
    metadata (``overmind.span.type`` / ``overmind.status`` /
    ``overmind.duration.seconds``) on the span as ``observe`` does on
    decorated functions.

    Example::

        for i in range(iterations):
            with start_span("iteration", attributes={"iteration": i}):
                # ... iteration work ...
                set_tag("decision", "keep")
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as otel_span:
        _stamp_span_metadata(otel_span, span_type)
        if attributes:
            for key, value in attributes.items():
                _safe_set_attribute(otel_span, key, value)
        start = time.monotonic()
        try:
            yield otel_span
        except BaseException as exc:
            _finalize_span(otel_span, exc, start)
            raise
        else:
            _finalize_span(otel_span, None, start)


# ---------------------------------------------------------------------------
# Span creation
# ---------------------------------------------------------------------------


@contextmanager
def start_child_span(
    name: str,
    *,
    span_type: SpanType = SpanType.FUNCTION,
    attributes: Mapping[str, Any] | None = None,
):
    """Open a span as an explicit child of the current OTel span.

    ``overmind.start_span`` already creates spans in the active context,
    but we re-attach the current span explicitly so the parent/child
    tree stays stable across nested wrappers and mixed instrumentation
    stacks (Traceloop, OTel auto-instrumentations, our own decorators).

    *attributes* are applied on the new span at start time so they are
    visible to any downstream span processor (the BatchSpanProcessor
    only flushes finished spans, but on-start tags are still useful for
    in-process processors).
    """
    current = trace.get_current_span()
    token = None
    try:
        if current is not None and current.get_span_context().is_valid:
            token = attach(trace.set_span_in_context(current))
        with start_span(name, span_type=span_type, attributes=dict(attributes or {})) as span:
            yield span
    finally:
        if token is not None:
            detach(token)


def conversation(conversation_id: str):
    """Decorator that sets a conversation ID in the current context."""

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                set_conversation_id(conversation_id)
                return await fn(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                set_conversation_id(conversation_id)
                return fn(*args, **kwargs)

            return sync_wrapper

    return decorator


def function(name: str | None = None):
    """Decorator that traces a function span."""
    return observe(span_name=name, type=SpanType.FUNCTION)


def entry_point(name: str | None = None):
    """Decorator that traces an entry point span."""
    return observe(span_name=name, type=SpanType.ENTRY_POINT)


def workflow(name: str | None = None):
    """Decorator that traces a workflow span."""
    return observe(span_name=name, type=SpanType.WORKFLOW)


def tool(name: str | None = None):
    """Decorator that traces a tool span."""
    return observe(span_name=name, type=SpanType.TOOL)


"""Tracing helpers for Overmind.

The overmind SDK ships ``@observe`` (in :mod:`overmind.tracing`) which
serialises a function's positional arguments and return value into
``inputs`` / ``outputs`` span attributes.  That's useful for general
observability but unsafe inside Overmind itself, where most traced
functions touch the user's agent source, prompts, datasets, or
credentials.

This module provides the leaner primitives the rest of the CLI builds
on:

* :func:`observe_safe` — drop-in for ``@observe`` that opens a child
  span **without** capturing inputs or outputs.  Use :func:`set_tag`
  inside the function for the specific scalar / categorical metadata
  you want to surface.
* :func:`start_child_span` — explicit context-manager variant for
  loops, conditional blocks, or sub-stages within a workflow.
* :func:`set_progress` / :func:`set_status` /
  :func:`set_iteration_analytics` — helpers that stamp the right
  ``overmind.*`` attributes on the current span using the canonical
  keys from :mod:`overmind.attrs`.  Keeping the keys centralised here
  is what lets the OTLP ingest pipeline parse them reliably.
* :func:`force_flush_traces` — best-effort flush so terminal events
  reach the backend before the CLI exits.
"""


def observe_safe(
    span_name: str | None = None,
    type: SpanType = SpanType.FUNCTION,
) -> Callable[[F], F]:
    """Decorator that opens a span without capturing arguments / return.

    Unlike :func:`overmind.observe`, the wrapped function's inputs and
    outputs are **not** serialised as span attributes — appropriate for
    Overmind-internal code paths that handle prompts, source, datasets,
    or credentials.  Use :func:`set_tag` inside the function for the
    specific scalar / categorical metadata you want to surface.
    """

    def decorator(func: F) -> F:
        name = span_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with start_child_span(name, span_type=type):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Exporter control
# ---------------------------------------------------------------------------


def force_flush_traces(timeout_millis: int = 1000) -> None:
    """Best-effort exporter flush for near real-time trace visibility.

    Called on terminal events (run finish, ``KeyboardInterrupt``, fatal
    exception) so the platform sees the final ``overmind.status`` tag
    before the CLI exits.  No-op when the active provider doesn't
    expose ``force_flush`` (e.g. the OTel API default ``ProxyTracerProvider``).
    """
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_millis)


def set_progress(
    phase: str,
    *,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Mark the current span with a human-readable progress milestone.

    *phase* is a short label (``"baseline_complete"``,
    ``"iteration"``, ``"holdout"`` …).  *current* and *total*, when
    provided, drive a uniform progress bar in the UI.
    """
    set_tag(attrs.PROGRESS_PHASE, phase)
    if current is not None:
        set_tag(attrs.PROGRESS_CURRENT, current)
    if total is not None:
        set_tag(attrs.PROGRESS_TOTAL, total)


def set_status(
    status: str,
    *,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Stamp the current span with an explicit lifecycle status.

    *status* must be one of ``"running"`` / ``"success"`` /
    ``"failed"`` / ``"cancelled"``.  When *status* is ``"failed"`` the
    caller should also pass *error_type* and a scrubbed
    *error_message* so the UI can surface a useful summary without
    cracking open the underlying exception payload.
    """
    set_tag(attrs.STATUS, status)
    if error_type:
        set_tag(attrs.ERROR_TYPE, error_type)
    if error_message:
        set_tag(attrs.ERROR_MESSAGE, error_message)


def set_iteration_analytics(
    *,
    iteration: int,
    decision: str,
    score: float | None = None,
    improvement: float | None = None,
    reason: str | None = None,
    dimension_scores: Mapping[str, float] | None = None,
) -> None:
    """Stamp the current span with optimizer iteration analytics.

    All keys are emitted under the ``overmind.optimize.*`` namespace
    so the OTLP ingest path can fold them into the corresponding
    :class:`JobIteration` row regardless of which child span finalises
    the iteration.
    """
    set_tag(attrs.OPTIMIZE_ITERATION, iteration)
    set_tag(attrs.OPTIMIZE_ITERATION_DECISION, decision)
    if score is not None:
        set_tag(attrs.OPTIMIZE_ITERATION_SCORE, score)
    if improvement is not None:
        set_tag(attrs.OPTIMIZE_ITERATION_IMPROVEMENT, improvement)
    if reason:
        set_tag(attrs.OPTIMIZE_ITERATION_REASON, reason)
    if dimension_scores:
        set_tag(
            attrs.OPTIMIZE_ITERATION_DIMENSION_SCORES,
            {k: float(v) for k, v in dimension_scores.items()},
        )


__all__ = [
    "set_iteration_analytics",
    "set_progress",
    "set_status",
]


# __all__ = ["force_flush_traces"]
