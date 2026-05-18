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
from overmind.core.constants import DEFAULT_BASE_URL
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

# ``DEFAULT_BASE_URL`` is re-exported above from ``overmind.core.constants`` so
# external imports (``from overmind.tracing import DEFAULT_BASE_URL``) continue
# to resolve. Edit the canonical value in ``overmind/core/constants.py``.


def get_api_settings(
    overmind_api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    overmind_api_key = overmind_api_key or os.getenv("OVERMIND_API_KEY")
    base_url = base_url or DEFAULT_BASE_URL

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


# Recursion guard for :func:`_normalize_for_json`.  Pathological inputs
# (``MagicMock``, cyclic ``__dict__`` references, Pydantic v1 models whose
# ``model_dump`` returns more models, …) used to cause unbounded recursion
# and 27 GB RSS spikes before we hit ``RecursionError``.  10 levels is
# more than any real LLM input we capture in tracing.
_MAX_NORMALIZE_DEPTH = 10


def _normalize_for_json(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert *obj* into a value :func:`json.dumps` can handle.

    Produces a tree of plain Python primitives (``str``/``int``/``float``/
    ``bool``/``None``), ``list`` and ``dict``.  Unknown objects fall back to
    a stringified placeholder so serialisation never crashes the caller.

    Handles, in order:

    * recursion-depth guard — past 10 levels we stringify the rest of the tree
    * primitives (passthrough)
    * dataclass instances → dict of normalised fields
    * pydantic-style ``model_dump()`` providers (only if it returns a real dict)
    * UI types listed in :data:`_SKIP_INPUT_TYPES` → ``"<TypeName>"`` tag
    * ``Mapping`` / ``Sequence`` (incl. ``set`` / ``tuple``) → list/dict of normalised members
    * ``bytes`` → hex string
    * ``PurePath`` → string
    * everything else → ``str(obj)``
    """
    if _depth > _MAX_NORMALIZE_DEPTH:
        return f"<truncated:{type(obj).__name__}>"
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _normalize_for_json(getattr(obj, f.name), _depth=_depth + 1)
            for f in dataclasses.fields(obj)
        }
    if _should_skip_value(obj):
        return f"<{type(obj).__name__}>"
    if isinstance(obj, dict):
        return {str(k): _normalize_for_json(v, _depth=_depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_normalize_for_json(item, _depth=_depth + 1) for item in obj]
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, PurePath):
        return str(obj)
    # Pydantic-style ``model_dump`` is the last resort because some test
    # doubles (``MagicMock``) expose a callable ``model_dump`` that returns
    # another mock — recurse there without a guard and you melt the heap.
    dumper = getattr(obj, "model_dump", None)
    if callable(dumper):
        try:
            dumped = dumper()
        except Exception:
            return str(obj)
        if isinstance(dumped, dict):
            return _normalize_for_json(dumped, _depth=_depth + 1)
        return str(obj)
    if hasattr(obj, "__dict__"):
        try:
            items = vars(obj).items()
        except TypeError:
            return str(obj)
        return {
            k: _normalize_for_json(v, _depth=_depth + 1)
            for k, v in items
            if not k.startswith("_")
        }
    return str(obj)


def _json_dumps(obj: Any) -> str:
    """JSON-serialise *obj* via :func:`_normalize_for_json`, never raises."""
    try:
        return json.dumps(_normalize_for_json(obj), ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        return repr(obj)


# Legacy aliases retained for any in-process callers that may still hold a
# reference; new code should use :func:`_normalize_for_json` /
# :func:`_json_dumps` directly.
serialize_dataclass = _normalize_for_json
serialize = _json_dumps


def _default_serializer(obj):
    """JSON ``default=`` hook used when an external caller hands us
    a ``json.dumps`` call directly (currently nothing internal does).

    Retained as a thin shim over :func:`_normalize_for_json` so external
    code that imports it doesn't break.
    """
    normalized = _normalize_for_json(obj)
    if isinstance(normalized, (dict, list, str, int, float, bool, type(None))):
        return normalized
    return str(normalized)


# ---------------------------------------------------------------------------
# Provider auto-instrumentation
# ---------------------------------------------------------------------------


def _enable_provider(name: str, module: str, instrumentor_factory) -> None:
    """Instrument *module* with the OTel instrumentor returned by *instrumentor_factory*.

    Idempotent: a second call for the same *name* is a no-op.  When the
    upstream module isn't installed we either raise (strict mode) or log a
    warning so the rest of the SDK keeps working.

    Parameters
    ----------
    name:
        Stable identifier stored in :data:`_providers` for the idempotency check.
    module:
        Module name to probe with :func:`importlib.util.find_spec`.
    instrumentor_factory:
        Zero-arg callable that returns an OTel instrumentor instance.  We import
        the instrumentor lazily *inside* this callable so providers that aren't
        being enabled don't pay the import cost.
    """
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if importlib.util.find_spec(module) is None:
        install_name = module.replace(".", "-")
        msg = f"{install_name} is not installed. Please install it with `pip install {install_name}`."
        if _strict_mode:
            raise ImportError(msg)
        logger.warning(msg)
        return

    instrumentor_factory().instrument()
    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def _agno_factory():
    from opentelemetry.instrumentation.agno import AgnoInstrumentor

    return AgnoInstrumentor()


def _openai_factory():
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    return OpenAIInstrumentor()


def _anthropic_factory():
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    return AnthropicInstrumentor()


def _google_genai_factory():
    from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor

    return GoogleGenerativeAiInstrumentor()


def enable_agno() -> None:
    _enable_provider("agno", "agno", _agno_factory)


def enable_openai() -> None:
    _enable_provider("openai", "openai", _openai_factory)


def enable_anthropic() -> None:
    _enable_provider("anthropic", "anthropic", _anthropic_factory)


def enable_google_genai() -> None:
    _enable_provider("google", "google.genai", _google_genai_factory)


_PROVIDER_ENABLERS = {
    "agno": enable_agno,
    "openai": enable_openai,
    "anthropic": enable_anthropic,
    "google": enable_google_genai,
}


def enable_tracing(providers: list[str] | None = None) -> None:
    if providers == []:
        # If no providers are provided, enable all supported providers.
        providers = list(_PROVIDER_ENABLERS.keys())
    logger.info(f"Enabling tracing for providers: {providers}")

    if providers is None:
        return
    for name in providers:
        enabler = _PROVIDER_ENABLERS.get(name)
        if enabler is None:
            logger.warning(f"Unknown tracing provider: {name!r}")
            continue
        enabler()


# Context-propagation keys.  We use the canonical ``overmind.*``
# attribute string as the context key for resources that show up on
# every span (currently just the agent name, which lives in
# :data:`overmind.attrs.AGENT_NAME`) so the on-start processor can
# stamp the same key without re-mapping.  Workflow + conversation
# labels don't have a 1:1 span-attribute counterpart, so they use
# their own dotted symbolic names.
_CTX_KEY_WORKFLOW_NAME = attrs.WORKFLOW_NAME
_CTX_KEY_CONVERSATION_ID = attrs.CONVERSATION_ID


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
        overmind_base_url: Base URL for traces. If not provided, uses the Overmind Cloud endpoint.
    """
    global _initialized, _tracer

    if _initialized:
        # user can call init again with different providers, so we should not skip
        # there is no such thing as remove initialization
        logger.debug(f"Overmind SDK already initialized, reinitializing with providers: {providers}")
        enable_tracing(providers)
        return

    # When running inside the optimize-step subprocess, the runner wrapper
    # configures a local JSONL TracerProvider via ``OVERMIND_TRACE_FILE`` and
    # deliberately strips ``OVERMIND_API_KEY`` from the env so spans land in a
    # file instead of the cloud backend. Any ``overmind.init()`` calls that
    # were instrumented into the agent entrypoint should reuse that already-
    # configured provider rather than crashing on the missing API key or
    # silently replacing the wrapper's exporter.
    if os.environ.get("OVERMIND_TRACE_FILE") and not (overmind_api_key or os.environ.get("OVERMIND_API_KEY")):
        from overmind import __version__ as _SDK_VERSION
        logger.debug(
            "Overmind SDK init() skipped: OVERMIND_TRACE_FILE is set and no "
            "OVERMIND_API_KEY available; reusing the local file-exporter "
            "TracerProvider configured by the optimize runner wrapper.",
        )
        _tracer = trace.get_tracer("overmind", _SDK_VERSION)
        enable_tracing(providers)
        _attach_remote_parent_if_present()
        _initialized = True
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


def _coerce_to_otel_attribute(value: Any) -> Any:
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
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return _json_dumps(value)


# Legacy alias retained for any in-process callers that may still hold a
# reference; new code should use :func:`_coerce_to_otel_attribute` directly.
_coerce_attribute_value = _coerce_to_otel_attribute


def _safe_set_attribute(otel_span, key: str, value: Any) -> None:
    """Set *key* / *value* on *otel_span* via :func:`_coerce_to_otel_attribute`."""
    otel_span.set_attribute(key, _coerce_to_otel_attribute(value))


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


# Legacy alias retained for callers that haven't migrated yet.  The
# behaviour is now identical to :func:`_normalize_for_json` since both
# share the same skip-list, model-dump, dataclass, and path-coercion logic.
_prepare_for_otel = _normalize_for_json


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
            inputs[param_name] = _normalize_for_json(arg)
        for key, value in kwargs.items():
            if _should_skip_value(value):
                continue
            inputs[key] = _normalize_for_json(value)

        otel_span.set_attribute("inputs", _json_dumps(inputs))
    except Exception:
        logger.debug("observe(): input capture failed for %s", func.__name__, exc_info=True)


def _capture_output(otel_span, result: Any) -> None:
    """Serialise *result* and attach it as ``outputs`` (best-effort)."""
    try:
        otel_span.set_attribute("outputs", _json_dumps(result))
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


# ---------------------------------------------------------------------------
# Safe tracing helpers
#
# The overmind SDK ships @observe which serialises a function's positional
# arguments and return value into ``inputs`` / ``outputs`` span attributes.
# That's useful for general observability but unsafe inside Overmind itself,
# where most traced functions touch the user's agent source, prompts,
# datasets, or credentials.
#
# observe_safe      — drop-in for @observe that opens a child span WITHOUT
#                     capturing inputs or outputs.  Use set_tag inside the
#                     function for specific scalar / categorical metadata.
# start_child_span  — explicit context-manager variant for loops, conditional
#                     blocks, or sub-stages within a workflow.
# set_progress / set_status / set_iteration_analytics
#                   — stamp the right overmind.* attributes on the current
#                     span using canonical keys from overmind.attrs so the
#                     OTLP ingest pipeline can parse them reliably.
# force_flush_traces — best-effort flush so terminal events reach the
#                     backend before the CLI exits.
# ---------------------------------------------------------------------------


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
    "DEFAULT_BASE_URL",
    "SpanType",
    "capture_exception",
    "conversation",
    "enable_agno",
    "enable_anthropic",
    "enable_google_genai",
    "enable_openai",
    "enable_tracing",
    "entry_point",
    "force_flush_traces",
    "function",
    "get_api_settings",
    "get_tracer",
    "init",
    "observe",
    "observe_safe",
    "serialize",
    "serialize_dataclass",
    "set_agent_name",
    "set_conversation_id",
    "set_iteration_analytics",
    "set_progress",
    "set_status",
    "set_tag",
    "set_user",
    "set_workflow_name",
    "start_child_span",
    "start_span",
    "tool",
    "workflow",
]
