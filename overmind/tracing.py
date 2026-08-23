"""Overmind SDK tracing: init(), span decorators/context managers, and
Sentry-style helpers. Attribute keys live in :mod:`overmind.attrs`."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from enum import Enum
from functools import wraps
from pathlib import Path, PurePath
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode

from overmind import attrs
from overmind.genai_usage import canonical_usage_updates

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

_strict_mode = os.environ.get("OVERMIND_STRICT_MODE", "false").lower() == "true"

_initialized = False
_tracer: trace.Tracer | None = None
_providers: set[str] = set()

DEFAULT_BASE_URL = os.getenv("OVERMIND_API_URL", "https://api.overmindlab.ai")


def get_api_settings(
    overmind_api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    overmind_api_key = overmind_api_key or os.getenv("OVERMIND_API_KEY")
    if not overmind_api_key:
        raise RuntimeError(
            "Missing OVERMIND_API_KEY. Set the environment variable to use Overmind services. "
            "Create a key at https://console.overmindlab.ai/projects"
        )
    return overmind_api_key, (base_url or DEFAULT_BASE_URL).rstrip("/")


# Guards against cyclic / mock-heavy inputs that recurse forever.
_MAX_NORMALIZE_DEPTH = 10


def _normalize_for_json(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert *obj* into JSON-serialisable primitives; never raises."""
    if _depth > _MAX_NORMALIZE_DEPTH:
        return f"<truncated:{type(obj).__name__}>"
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _normalize_for_json(getattr(obj, f.name), _depth=_depth + 1) for f in dataclasses.fields(obj)}
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
    # model_dump last: MagicMock exposes a callable one that returns more mocks.
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
        return {k: _normalize_for_json(v, _depth=_depth + 1) for k, v in items if not k.startswith("_")}
    return str(obj)


def _json_dumps(obj: Any) -> str:
    """JSON-serialise *obj* via :func:`_normalize_for_json`, never raises."""
    try:
        return json.dumps(_normalize_for_json(obj), ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        return repr(obj)


# Legacy aliases for external callers.
serialize_dataclass = _normalize_for_json
serialize = _json_dumps


def _default_serializer(obj):
    """Legacy ``json.dumps(default=...)`` hook for external callers."""
    normalized = _normalize_for_json(obj)
    if isinstance(normalized, (dict, list, str, int, float, bool, type(None))):
        return normalized
    return str(normalized)


# ---------------------------------------------------------------------------
# Provider auto-instrumentation
# ---------------------------------------------------------------------------


def _enable_provider(name: str, module: str, instrumentor_factory) -> None:
    """Instrument *module* if installed. Idempotent; missing module raises
    only in strict mode. Factories import lazily to avoid upfront cost."""
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
    if providers == []:  # empty list means "all"
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


# OTel context keys (canonical attribute strings double as keys).
_CTX_KEY_WORKFLOW_NAME = attrs.WORKFLOW_NAME
_CTX_KEY_CONVERSATION_ID = attrs.CONVERSATION_ID
# Context-only key (never a span attribute): a one-shot _PendingTurn cell set
# on entry into a handoff capability scope.
_CTX_KEY_PENDING_TURN = "overmind.capability.pending_turn"


class _PendingTurn:
    """One-shot cell: the first span started inside a handoff capability scope
    consumes it and becomes the new scoring unit's boundary (``unit_kind="turn"``).
    Shared by reference across context copies, so exactly one span wins."""

    __slots__ = ("consumed",)

    def __init__(self) -> None:
        self.consumed = False


def _span_processor_on_start(span: trace.Span, parent_context: trace.Context | None = None):
    """Stamp every new span with the ambient agent / workflow context
    written by the ``set_*`` helpers."""
    if value := get_value(_CTX_KEY_WORKFLOW_NAME):
        span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, str(value))
    if agent_name := get_value(attrs.AGENT_NAME):
        span.set_attribute(attrs.AGENT_NAME, str(agent_name))
    if agent_id := get_value(attrs.AGENT_ID):
        span.set_attribute(attrs.AGENT_ID, str(agent_id))
    if project_id := get_value(attrs.PROJECT_ID):
        span.set_attribute(attrs.PROJECT_ID, str(project_id))
    if conversation_id := get_value(_CTX_KEY_CONVERSATION_ID):
        span.set_attribute("conversation.id", str(conversation_id))
    pending = get_value(_CTX_KEY_PENDING_TURN)
    if isinstance(pending, _PendingTurn) and not pending.consumed:
        pending.consumed = True
        span.set_attribute(attrs.UNIT_KIND, "turn")


class _GenAiUsageSpanProcessor(SpanProcessor):
    """Mirror OTel ``gen_ai.*`` usage onto canonical ``genai.*`` keys at span
    end, so auto-instrumented spans carry the tokens/cost keys the server reads.

    ponytail: mutates ``span._attributes`` (ReadableSpan has no set_attribute).
    Since opentelemetry-sdk 1.43 the ended span's ``BoundedAttributes`` rejects
    item assignment, so we write into its backing ``._dict`` when present and
    fall back to direct assignment for older SDKs. Upgrade path if this seals
    too: a SpanExporter wrapper.
    """

    def on_start(self, span: trace.Span, parent_context: trace.Context | None = None) -> None:
        return

    def on_end(self, span) -> None:
        try:
            self.patch_on_end(span)
        except Exception:
            logger.debug("genai enrichment could not set attributes", exc_info=True)

    def patch_on_end(self, span) -> None:
        updates = canonical_usage_updates(span.attributes or {})
        if not updates:
            return
        target = getattr(span, "_attributes", None)
        if target is None:
            return
        # BoundedAttributes (1.43+) is read-only once the span ends; its plain
        # ``_dict`` backing store still accepts writes and is what ``attributes``
        # proxies. Older SDKs accept assignment on the object itself.
        sink = getattr(target, "_dict", target)
        for key, value in updates.items():
            try:
                sink[key] = value
            except Exception:
                logger.debug("genai enrichment could not set %s", key, exc_info=True)


# ---------------------------------------------------------------------------
# Remote parent context propagation (subprocess / distributed tracing)
# ---------------------------------------------------------------------------


def _attach_remote_parent_if_present() -> None:
    """Attach the W3C ``TRACEPARENT`` env var (injected by the optimiser into
    agent subprocesses) as the ambient parent context. No-op when absent."""
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
# Git commit sha auto-detection (resource attribute ``vcs.ref.head.revision``)
# ---------------------------------------------------------------------------


_GIT_SHA_ENV_VARS = (
    "OVERMIND_GIT_SHA",  # explicit override, checked first
    "GIT_SHA",
    "GIT_COMMIT",
    "GITHUB_SHA",
    "RENDER_GIT_COMMIT",
    "VERCEL_GIT_COMMIT_SHA",
    "HEROKU_SLUG_COMMIT",
    "CI_COMMIT_SHA",
)


def _detect_git_sha(start: Path | None = None) -> str | None:
    """Best-effort commit sha of the running code: env vars first, then
    ``.git/HEAD`` walking up from *start* (default cwd). Never raises,
    never shells out to git."""
    for var in _GIT_SHA_ENV_VARS:
        if sha := os.environ.get(var, "").strip():
            return sha
    try:
        start = start or Path.cwd()
        for directory in (start, *start.parents):
            # ponytail: ``.git`` as a *file* (worktree / submodule) is not
            # resolved; upgrade path is following its ``gitdir:`` pointer.
            head = directory / ".git" / "HEAD"
            if not head.is_file():
                continue
            content = head.read_text(encoding="utf-8").strip()
            if not content.startswith("ref:"):
                return content or None  # detached HEAD holds the sha itself
            ref_name = content[4:].strip()
            ref_file = directory / ".git" / ref_name
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip() or None
            packed = directory / ".git" / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    sha, _, name = line.partition(" ")
                    if name == ref_name:
                        return sha
            return None
    except Exception:
        logger.debug("git sha detection failed", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# SDK initialisation
# ---------------------------------------------------------------------------


def _seed_identity_context(
    agent_id: str | None,
    agent_name: str | None,
    project_id: str | None,
) -> None:
    """Attach identity values to the OTel context; the on-start processor
    copies them onto every span (including auto-instrumented ones)."""
    if agent_id:
        set_agent_id(agent_id)
    if agent_name:
        set_agent_name(agent_name)
    if project_id:
        set_project_id(project_id)


def init(
    overmind_api_key: str | None = None,
    *,
    service_name: str | None = None,
    environment: str | None = None,
    providers: list[str] | None = None,
    overmind_base_url: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
    project_id: str | None = None,
):
    """Initialise the Overmind SDK for automatic monitoring.

    Args:
        overmind_api_key: API key; defaults to OVERMIND_API_KEY env var.
        service_name: Service name in traces; defaults to OVERMIND_SERVICE_NAME.
        environment: e.g. "production"; defaults to OVERMIND_ENVIRONMENT or "development".
        providers: Providers to trace: "openai", "anthropic", "google", "agno".
        overmind_base_url: Trace endpoint base URL; defaults to Overmind Cloud.
        agent_id: Agent UUID (preferred over agent_name); defaults to OVERMIND_AGENT_ID.
        agent_name: Human-readable agent name; defaults to OVERMIND_AGENT_NAME.
        project_id: Project UUID, only needed for session auth; defaults to OVERMIND_PROJECT_ID.
    """
    global _initialized, _tracer

    agent_id = agent_id or os.environ.get("OVERMIND_AGENT_ID")
    agent_name = agent_name or os.environ.get("OVERMIND_AGENT_NAME")
    project_id = project_id or os.environ.get("OVERMIND_PROJECT_ID")

    if _initialized:
        # Re-init only refreshes identity + providers; exporters stay as-is.
        logger.debug(f"Overmind SDK already initialised, reinitialising with providers: {providers}")
        _seed_identity_context(agent_id, agent_name, project_id)
        enable_tracing(providers)
        return

    # Optimise-step subprocess: the runner wrapper set up a file-exporter
    # provider (OVERMIND_TRACE_FILE) and stripped the API key — reuse it
    # instead of crashing or replacing the exporter.
    if os.environ.get("OVERMIND_TRACE_FILE") and not (overmind_api_key or os.environ.get("OVERMIND_API_KEY")):
        from overmind import __version__ as sdk_version

        logger.debug(
            "Overmind SDK init() skipped: OVERMIND_TRACE_FILE is set and no "
            "OVERMIND_API_KEY available; reusing the local file-exporter "
            "TracerProvider configured by the optimise runner wrapper.",
        )
        _tracer = trace.get_tracer("overmind", sdk_version)
        _seed_identity_context(agent_id, agent_name, project_id)
        enable_tracing(providers)
        _attach_remote_parent_if_present()
        _initialized = True
        return

    environment = (
        environment or os.environ.get("OVERMIND_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "development"
    )

    overmind_api_key, overmind_base_url = get_api_settings(overmind_api_key, overmind_base_url)

    endpoint = f"{overmind_base_url}/api/v1/traces"

    from overmind import __version__ as sdk_version

    resource_attributes = {
        "service.name": service_name or os.environ.get("OVERMIND_SERVICE_NAME") or "overmind-telemetry",
        "service.version": os.environ.get("SERVICE_VERSION", sdk_version),
        "deployment.environment": environment,
        attrs.SDK_NAME: "overmind-python",
        attrs.SDK_VERSION: sdk_version,
    }
    # Identity on the resource lets the server resolve Agent/Project directly.
    if agent_id:
        resource_attributes[attrs.AGENT_ID] = agent_id
    if agent_name:
        resource_attributes[attrs.AGENT_NAME] = agent_name
    if project_id:
        resource_attributes[attrs.PROJECT_ID] = project_id
    # Commit sha binds every trace to the exact code the process runs.
    if git_sha := _detect_git_sha():
        resource_attributes[attrs.VCS_REF_HEAD_REVISION] = git_sha

    resource = Resource.create(resource_attributes)

    provider = TracerProvider(resource=resource)
    # Must run before the exporting processor so its on-end mutation is exported.
    provider.add_span_processor(_GenAiUsageSpanProcessor())

    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, headers={"X-Api-Key": overmind_api_key})

    # Flush every ~2s (OTel default 5s) so progress streams while spans are open.
    schedule_delay_millis = int(os.environ.get("OVERMIND_SPAN_FLUSH_INTERVAL_MS", "2000"))
    max_export_batch_size = int(os.environ.get("OVERMIND_SPAN_MAX_EXPORT_BATCH_SIZE", "256"))
    span_processor = BatchSpanProcessor(
        otlp_exporter,
        schedule_delay_millis=schedule_delay_millis,
        max_export_batch_size=max_export_batch_size,
    )
    provider.add_span_processor(span_processor)
    span_processor.on_start = _span_processor_on_start

    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer("overmind", sdk_version)
    _seed_identity_context(agent_id, agent_name, project_id)
    enable_tracing(providers)
    _attach_remote_parent_if_present()

    _initialized = True
    logger.info(f"Overmind SDK initialised: service={service_name}, environment={environment}")


def get_tracer() -> trace.Tracer:
    """Return the Overmind tracer; raises RuntimeError if not initialised."""
    if not _initialized or _tracer is None:
        raise RuntimeError("Overmind SDK not initialised. Call overmind.init() first.")
    return _tracer


# ---------------------------------------------------------------------------
# Span attribute helpers
# ---------------------------------------------------------------------------


def set_user(user_id: str, email: str | None = None, username: str | None = None) -> None:
    """Associate the current trace with a user (like Sentry's ``set_user``)."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("user.id", user_id)
        if email:
            span.set_attribute("user.email", email)
        if username:
            span.set_attribute("user.username", username)


def _coerce_to_otel_attribute(value: Any) -> Any:
    """Coerce *value* to an OTel-legal attribute; rich values become JSON."""
    if value is None:
        return ""
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return _json_dumps(value)


# Legacy alias.
_coerce_attribute_value = _coerce_to_otel_attribute


def _safe_set_attribute(otel_span, key: str, value: Any) -> None:
    """Set *key* / *value* on *otel_span* via :func:`_coerce_to_otel_attribute`."""
    otel_span.set_attribute(key, _coerce_to_otel_attribute(value))


def set_tag(key: str, value) -> None:
    """Add a custom tag to the current span; rich values are JSON-encoded."""
    span = trace.get_current_span()
    if not span.is_recording():
        logger.debug("set_tag(%s=…) ignored: current span has ended %s", key, span)
        return
    _safe_set_attribute(span, key, value)


def capture_exception(exception: Exception) -> None:
    """Record an exception on the current span and mark it as errored."""
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def set_workflow_name(workflow_name: str) -> None:
    """Attach a Traceloop-compatible workflow label to every subsequent span."""
    attach(set_value(attrs.WORKFLOW_NAME, workflow_name))


def set_agent_name(agent_name: str) -> None:
    """Stamp ``overmind.agent.name`` on every span created downstream."""
    attach(set_value(attrs.AGENT_NAME, agent_name))


def set_agent_id(agent_id: str) -> None:
    """Stamp ``overmind.agent.id`` (server UUID) on every downstream span."""
    attach(set_value(attrs.AGENT_ID, agent_id))


def set_project_id(project_id: str) -> None:
    """Stamp ``overmind.project.id`` on every downstream span."""
    attach(set_value(attrs.PROJECT_ID, project_id))


def set_conversation_id(conversation_id: str) -> None:
    """Tag downstream spans with a stable ``conversation.id`` for session grouping."""
    attach(set_value(attrs.CONVERSATION_ID, conversation_id))


# ---------------------------------------------------------------------------
# Capability scoping
# ---------------------------------------------------------------------------


def _is_handoff(name: str | None, id: str | None) -> bool:
    """Entering a capability that differs from the active one, mid-trace.

    Identity is compared on the finest shared grain: ids when both sides have
    one, else names. Mixed grains (id-only scope under name-only identity)
    are never treated as a handoff — a boundary is only declared when the
    identities are provably different."""
    if not trace.get_current_span().get_span_context().is_valid:
        return False
    active_id = get_value(attrs.AGENT_ID)
    active_name = get_value(attrs.AGENT_NAME)
    if id and active_id:
        return str(id) != str(active_id)
    if name and active_name:
        return str(name) != str(active_name)
    return False


class _CapabilityScope:
    """Context manager (sync or async) and decorator produced by
    :func:`capability`. Entering attaches the capability identity to the OTel
    context (async-safe via contextvars) so the on-start processor stamps it
    on every span created inside; exiting restores the outer identity."""

    def __init__(self, name: str | None, id: str | None) -> None:
        if not name and not id:
            raise ValueError("capability() requires a name and/or id")
        self._name = str(name) if name else None
        self._id = str(id) if id else None
        self._token = None

    def __enter__(self) -> _CapabilityScope:
        ctx = set_value(_CTX_KEY_PENDING_TURN, _PendingTurn() if _is_handoff(self._name, self._id) else None)
        # Both keys are always written: a name-only scope must not inherit the
        # outer scope's id (the server resolves id before name).
        ctx = set_value(attrs.AGENT_NAME, self._name, ctx)
        ctx = set_value(attrs.AGENT_ID, self._id, ctx)
        self._token = attach(ctx)
        return self

    def __exit__(self, *exc) -> bool:
        if self._token is not None:
            detach(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> _CapabilityScope:
        return self.__enter__()

    async def __aexit__(self, *exc) -> bool:
        return self.__exit__(*exc)

    def __call__(self, func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with _CapabilityScope(self._name, self._id):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _CapabilityScope(self._name, self._id):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]


def capability(name: str | None = None, *, id: str | None = None) -> _CapabilityScope:
    """Declare that all work inside belongs to the named capability.

    Usable as a context manager (``with`` / ``async with``) or decorator.
    Every span created inside carries ``overmind.agent.name`` / ``.id``; on
    exit the outer identity is restored. Entering a *different* capability
    mid-trace is a handoff: the first span of the new scope is stamped
    ``overmind.unit_kind = "turn"`` so the platform opens a new scoring unit.
    The identity must be one the project declared — nothing is auto-created."""
    return _CapabilityScope(name, id)


# ---------------------------------------------------------------------------
# Span types and decorators
# ---------------------------------------------------------------------------


class SpanType(str, Enum):
    FUNCTION = "function"
    ENTRY_POINT = "entry_point"
    WORKFLOW = "workflow"
    TOOL = "tool_call"
    LLM = "llm_call"
    RETRIEVAL = "retrieval"


_PROVENANCE_VALUES = frozenset({"user", "agent", "environment", "harness"})
_UNIT_KINDS = frozenset({"turn", "run"})

# Span types whose payloads have an unambiguous provenance class: tool results
# and retrieved documents are environment observations, model completions are
# agent-authored.  Everything else needs an explicit ``provenance=``.
_SPAN_TYPE_PROVENANCE = {
    SpanType.TOOL: "environment",
    SpanType.RETRIEVAL: "environment",
    SpanType.LLM: "agent",
}


def _validate_provenance(value: str | None) -> None:
    if value is not None and value not in _PROVENANCE_VALUES:
        raise ValueError(f"provenance must be one of {sorted(_PROVENANCE_VALUES)}, got {value!r}")


def _stamp_evidence(otel_span, span_type: SpanType, provenance: str | None) -> None:
    """Stamp the evaluation evidence contract: the provenance class of the
    span's payloads, and ``unit_kind = "run"`` on entry points (run roots)."""
    if provenance := provenance or _SPAN_TYPE_PROVENANCE.get(span_type):
        otel_span.set_attribute(attrs.PROVENANCE, provenance)
    if span_type is SpanType.ENTRY_POINT:
        # A capability-handoff boundary already carries ``turn`` from the
        # on-start processor; don't downgrade it to the run-root fallback.
        existing = getattr(otel_span, "attributes", None)
        if not existing or existing.get(attrs.UNIT_KIND) != "turn":
            otel_span.set_attribute(attrs.UNIT_KIND, "run")


# Type names never serialised into span attributes; matched by name to
# avoid importing rich / opentelemetry for an isinstance check.
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


# Legacy alias.
_prepare_for_otel = _normalize_for_json


def _finalize_span(
    otel_span,
    exc: BaseException | None,
    start_monotonic: float,
) -> None:
    """Stamp lifecycle status + duration as the span closes (*exc* is None on success)."""
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
    """Attach *args* / *kwargs* as ``inputs`` (best-effort; skips self/cls
    and :data:`_SKIP_INPUT_TYPES`)."""
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


def _stamp_code_identity(otel_span, func: Callable) -> None:
    """Stamp ``code.namespace`` / ``code.function.name`` from the *unwrapped*
    function so the server can bind the span to a code-symbol anchor
    (``module.qualname``). Best-effort, never raises."""
    try:
        target = inspect.unwrap(func)
        if module := getattr(target, "__module__", None):
            otel_span.set_attribute(attrs.CODE_NAMESPACE, module)
        if qualname := getattr(target, "__qualname__", None):
            otel_span.set_attribute(attrs.CODE_FUNCTION_NAME, qualname)
    except Exception:
        logger.debug("observe(): code identity capture failed", exc_info=True)


def _stamp_tool_metadata(otel_span, tool_name: str, func: Callable, args: tuple, kwargs: dict) -> None:
    """Stamp ``tool.name`` / ``tool.arg_keys`` (keys only) on a tool span."""
    try:
        otel_span.set_attribute(attrs.TOOL_NAME, tool_name)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        is_method = bool(param_names) and param_names[0] in ("self", "cls")
        start_idx = 1 if is_method else 0
        arg_keys = [param_names[i] if i < len(param_names) else f"arg_{i}" for i in range(start_idx, len(args))]
        arg_keys += list(kwargs.keys())
        if arg_keys:
            otel_span.set_attribute(attrs.TOOL_ARG_KEYS, arg_keys)
    except Exception:
        logger.debug("tool(): metadata capture failed for %s", tool_name, exc_info=True)


def observe(
    span_name: str | None = None,
    type: SpanType = SpanType.FUNCTION,
    agent_id: str | None = None,
    project_id: str | None = None,
    provenance: str | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator (sync or async) that traces a function: captures ``inputs`` /
    ``outputs`` and stamps canonical span type / status / duration.

    ``provenance`` overrides the evidence provenance class
    (``user`` / ``agent`` / ``environment`` / ``harness``); tool, retrieval
    and LLM spans get their natural class automatically.

    ``agent_id`` routes through :func:`capability`: the decorated span *and*
    its children carry the identity, and calling into a different capability
    mid-trace marks a handoff boundary."""
    _validate_provenance(provenance)

    def _capability_scope():
        return capability(id=agent_id) if agent_id else nullcontext()

    def decorator(func: Callable) -> Callable:
        name = span_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with _capability_scope(), tracer.start_as_current_span(name) as otel_span:
                    otel_span.set_attribute(attrs.SPAN_TYPE, type.value)
                    _stamp_evidence(otel_span, type, provenance)
                    _stamp_code_identity(otel_span, func)
                    if project_id:
                        otel_span.set_attribute(attrs.PROJECT_ID, project_id)
                    if agent_id:
                        otel_span.set_attribute(attrs.AGENT_ID, agent_id)
                    if type == SpanType.TOOL:
                        _stamp_tool_metadata(otel_span, name, func, args, kwargs)
                    _capture_inputs(otel_span, func, args, kwargs)
                    start = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        if type == SpanType.TOOL:
                            otel_span.set_attribute(attrs.TOOL_ERROR, exc.__class__.__name__)
                        _finalize_span(otel_span, exc, start)
                        raise
                    _capture_output(otel_span, result)
                    _finalize_span(otel_span, None, start)
                    return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with _capability_scope(), tracer.start_as_current_span(name) as otel_span:
                otel_span.set_attribute(attrs.SPAN_TYPE, type.value)
                _stamp_evidence(otel_span, type, provenance)
                _stamp_code_identity(otel_span, func)
                if project_id:
                    otel_span.set_attribute(attrs.PROJECT_ID, project_id)
                if agent_id:
                    otel_span.set_attribute(attrs.AGENT_ID, agent_id)
                if type == SpanType.TOOL:
                    _stamp_tool_metadata(otel_span, name, func, args, kwargs)
                _capture_inputs(otel_span, func, args, kwargs)
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    if type == SpanType.TOOL:
                        otel_span.set_attribute(attrs.TOOL_ERROR, exc.__class__.__name__)
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
    *,
    provenance: str | None = None,
):
    """Context-manager companion to :func:`observe`; stamps the same
    canonical span metadata."""
    _validate_provenance(provenance)
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as otel_span:
        otel_span.set_attribute(attrs.SPAN_TYPE, span_type.value)
        _stamp_evidence(otel_span, span_type, provenance)
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
    provenance: str | None = None,
):
    """Open a span as an explicit child of the current OTel span; re-attaching
    the parent keeps the tree stable across mixed instrumentation stacks."""
    current = trace.get_current_span()
    token = None
    try:
        if current is not None and current.get_span_context().is_valid:
            token = attach(trace.set_span_in_context(current))
        with start_span(name, span_type=span_type, attributes=dict(attributes or {}), provenance=provenance) as span:
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


def function(name: str | None = None, **kwargs):
    """Decorator that traces a function span."""
    return observe(span_name=name, type=SpanType.FUNCTION, **kwargs)


def entry_point(name: str | None = None, **kwargs):
    """Decorator that traces an entry point span."""
    return observe(span_name=name, type=SpanType.ENTRY_POINT, **kwargs)


def workflow(name: str | None = None, **kwargs):
    """Decorator that traces a workflow span."""
    return observe(span_name=name, type=SpanType.WORKFLOW, **kwargs)


def tool(name: str | None = None, **kwargs):
    """Decorator that traces a tool span (adds ``tool.name`` / ``tool.arg_keys``)."""
    return observe(span_name=name, type=SpanType.TOOL, **kwargs)


def retrieval(name: str | None = None, **kwargs):
    """Decorator that traces a retrieval / RAG step span."""
    return observe(span_name=name, type=SpanType.RETRIEVAL, **kwargs)


def mark_unit(kind: str) -> None:
    """Stamp ``overmind.unit_kind`` on the current span: ``"turn"`` begins a
    user-visible unit of work, ``"run"`` is the root of a full agent run.
    ``@entry_point`` spans are marked ``"run"`` automatically; use this for
    turn boundaries and for harness spans the SDK doesn't wrap."""
    if kind not in _UNIT_KINDS:
        raise ValueError(f"mark_unit() kind must be one of {sorted(_UNIT_KINDS)}, got {kind!r}")
    set_tag(attrs.UNIT_KIND, kind)


def _grounding_span_id(handle: Any) -> str:
    """Accept a span_id hex string or any OTel span handle (e.g. the span
    yielded by :func:`start_span`)."""
    if isinstance(handle, str):
        return handle
    return format(handle.get_span_context().span_id, "016x")


def deliver(
    payload: Any,
    *,
    grounded_by: list[Any] | None = None,
    name: str = "deliver",
    provenance: str = "agent",
) -> None:
    """Capture the terminal deliverable of a run on its own child span:
    the payload is serialised into ``outputs`` and the span carries
    ``overmind.delivery = true``.  ``grounded_by`` names the evidence spans
    the deliverable rests on (span_id hex strings or span handles)."""
    _validate_provenance(provenance)
    with start_child_span(name, provenance=provenance) as otel_span:
        otel_span.set_attribute(attrs.DELIVERY, True)
        if grounded_by:
            otel_span.set_attribute(attrs.GROUNDED_BY, json.dumps([_grounding_span_id(h) for h in grounded_by]))
        _capture_output(otel_span, payload)


# ---------------------------------------------------------------------------
# Safe tracing helpers (no input/output capture — for code that touches
# prompts, datasets, or credentials)
# ---------------------------------------------------------------------------


def observe_safe(
    span_name: str | None = None,
    type: SpanType = SpanType.FUNCTION,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> Callable[[F], F]:
    """Like :func:`observe` but never captures arguments or return values;
    use :func:`set_tag` inside the function for specific metadata."""

    def decorator(func: F) -> F:
        name = span_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with start_child_span(name, span_type=type) as otel_span:
                if project_id:
                    otel_span.set_attribute(attrs.PROJECT_ID, project_id)
                if agent_id:
                    otel_span.set_attribute(attrs.AGENT_ID, agent_id)
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Exporter control
# ---------------------------------------------------------------------------


def force_flush_traces(timeout_millis: int = 1000) -> None:
    """Best-effort exporter flush before exit; no-op if the provider
    lacks ``force_flush``."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_millis)


def set_progress(
    phase: str,
    *,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Mark the current span with a progress milestone; *current* / *total*
    drive the UI progress bar."""
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
    """Stamp an explicit lifecycle status: "running" / "success" / "failed"
    / "cancelled". Pass error details when failed."""
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
    """Stamp optimiser iteration analytics (``overmind.optimize.*``) on the
    current span."""
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
    "capability",
    "capture_exception",
    "conversation",
    "deliver",
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
    "mark_unit",
    "observe",
    "observe_safe",
    "retrieval",
    "serialize",
    "serialize_dataclass",
    "set_agent_id",
    "set_agent_name",
    "set_conversation_id",
    "set_iteration_analytics",
    "set_progress",
    "set_project_id",
    "set_status",
    "set_tag",
    "set_user",
    "set_workflow_name",
    "start_child_span",
    "start_span",
    "tool",
    "workflow",
]
