"""Overmind SDK tracing: init(), span decorators/context managers, and
Sentry-style helpers. Attribute keys live in :mod:`overmind.attrs`.

Every helper degrades gracefully: without ``init()`` (or without an API key)
decorators call straight through, spans are non-recording, and nothing raises.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
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
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
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
_init_lock = threading.Lock()
# Libraries call init() at their entry point, so a keyless user would see the
# "tracing disabled" line on every run — log it once per process, then DEBUG.
_keyless_logged = False

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


# ---------------------------------------------------------------------------
# Serialisation + scrubbing
# ---------------------------------------------------------------------------

# Guards against cyclic / mock-heavy inputs that recurse forever.
_MAX_NORMALIZE_DEPTH = 10

_DATA_URL_RE = re.compile(r"^data:[\w.+-]+/[\w.+-]+;base64,")
_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{512,}$")
# Substring match on lowered dict keys, so provider-prefixed names
# (``openai_api_key``, ``access_token``, ``sensitive_data``) are caught too.
_SECRET_KEY_MARKERS = (
    "password",
    "secret",
    "token",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "sensitive",
)
# Extra exact-match keys, extended via ``init(redact_keys=...)``.
_extra_redact_keys: frozenset[str] = frozenset()

# Bytes payloads longer than this (screenshots, audio) become a placeholder
# instead of a hex dump.
_MAX_BYTES_HEX = 256


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _extra_redact_keys or any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def _scrub_text(value: str) -> str:
    """Redact base64 blobs / data URLs; all other text passes through in full."""
    if _DATA_URL_RE.match(value) or (len(value) >= 512 and _BASE64ISH_RE.match(value)):
        return f"<base64 {len(value)} chars>"
    return value


def _normalize_for_json(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert *obj* into JSON-serialisable primitives, scrubbing
    secrets and binary blobs on the way; never raises."""
    if _depth > _MAX_NORMALIZE_DEPTH:
        return f"<truncated:{type(obj).__name__}>"
    if isinstance(obj, str):
        return _scrub_text(obj)
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _normalize_for_json(getattr(obj, f.name), _depth=_depth + 1) for f in dataclasses.fields(obj)}
    if _should_skip_value(obj):
        return f"<{type(obj).__name__}>"
    if isinstance(obj, dict):
        return {
            str(k): ("<redacted>" if _is_secret_key(str(k)) else _normalize_for_json(v, _depth=_depth + 1))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_normalize_for_json(item, _depth=_depth + 1) for item in obj]
    if isinstance(obj, bytes):
        return obj.hex() if len(obj) <= _MAX_BYTES_HEX else f"<bytes {len(obj)}>"
    if isinstance(obj, PurePath):
        return str(obj)
    # model_dump last: MagicMock exposes a callable one that returns more mocks.
    dumper = getattr(obj, "model_dump", None)
    if callable(dumper):
        try:
            dumped = dumper(exclude_none=True, mode="json")
        except TypeError:
            try:
                dumped = dumper()
            except Exception:
                return str(obj)
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
            str(k): ("<redacted>" if _is_secret_key(str(k)) else _normalize_for_json(v, _depth=_depth + 1))
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


# Public alias: platform telemetry (overbae brain-chat/workshop) serialises
# manual span inputs/outputs through the same scrubbing path.
serialize = _json_dumps


# ---------------------------------------------------------------------------
# Message normalisation (LLM evidence: role + full text, vision payloads drop)
# ---------------------------------------------------------------------------

_MESSAGE_ROLE_MAP = {"human": "user", "ai": "assistant"}


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            elif getattr(part, "text", None):
                parts.append(str(part.text))
        return "\n".join(parts)
    return str(content)


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalise a chat-message list (dicts, OpenAI/Anthropic/LangChain
    objects) into ``{"role", "content"}`` entries with full text. Image and
    base64 parts are dropped; tool calls carried as ``tool_calls`` /
    ``tool_call_id`` when present."""
    out: list[dict[str, Any]] = []
    for message in messages or []:
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "user")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            tool_call_id = message.get("tool_call_id")
        else:
            role = str(getattr(message, "role", None) or getattr(message, "type", None) or "user")
            content = getattr(message, "text", None) or getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
            tool_call_id = getattr(message, "tool_call_id", None)
        entry: dict[str, Any] = {"role": _MESSAGE_ROLE_MAP.get(role, role), "content": _message_text(content)}
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} if isinstance(tc, dict) else str(tc)
                for tc in tool_calls
            ]
        if tool_call_id:
            entry["tool_call_id"] = str(tool_call_id)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Provider auto-instrumentation
# ---------------------------------------------------------------------------

# provider name -> (importable module gate, instrumentation module, class)
_PROVIDER_MODULES: dict[str, tuple[str, str, str]] = {
    "agno": ("agno", "opentelemetry.instrumentation.agno", "AgnoInstrumentor"),
    "openai": ("openai", "opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
    "anthropic": ("anthropic", "opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
    "google": ("google.genai", "opentelemetry.instrumentation.google_generativeai", "GoogleGenerativeAiInstrumentor"),
    # Covers LangChain AND LangGraph (the OpenInference instrumentor hooks the
    # shared callback system). Instrumentor ships as the ``overmind[langchain]``
    # extra, not a core dependency.
    "langchain": ("langchain_core", "openinference.instrumentation.langchain", "LangChainInstrumentor"),
}

# Providers whose instrumentor is an optional extra rather than a core
# dependency; value is the extra name for the install hint.
_PROVIDER_EXTRAS: dict[str, str] = {"langchain": "langchain"}


def _module_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # parent namespace package absent
        return False


def _enable_provider(name: str, module: str, instrumentation_module: str, class_name: str) -> None:
    """Instrument *module* if installed. Idempotent; a missing target library
    or missing extra-shipped instrumentor raises only in strict mode.
    Instrumentors import lazily to avoid upfront cost."""
    if name in _providers:
        logger.debug(f"{name} already enabled")
        return

    if not _module_installed(module):
        install_name = module.replace(".", "-")
        msg = f"{install_name} is not installed. Please install it with `pip install {install_name}`."
        if _strict_mode:
            raise ImportError(msg)
        logger.warning(msg)
        return

    if not _module_installed(instrumentation_module):
        extra = _PROVIDER_EXTRAS.get(name, name)
        msg = f"The {name} instrumentor is not installed. Please install it with `pip install 'overmind[{extra}]'`."
        if _strict_mode:
            raise ImportError(msg)
        logger.warning(msg)
        return

    instrumentor_cls = getattr(importlib.import_module(instrumentation_module), class_name)
    instrumentor_cls().instrument()
    _providers.add(name)
    logger.info(f"{name} instrumentation enabled")


def _detect_providers() -> list[str]:
    """Provider names whose target library AND instrumentor are both installed."""
    return [
        name
        for name, (module, instrumentation_module, _) in _PROVIDER_MODULES.items()
        if _module_installed(module) and _module_installed(instrumentation_module)
    ]


def enable_tracing(providers: list[str] | str | None = None) -> None:
    """Instrument the named providers ("openai" / "anthropic" / "google" /
    "agno" / "langchain"); an empty list means all, ``"auto"`` detects and
    enables every provider whose target library and instrumentor are both
    installed. For fan-out setups that skip ``init()`` and export through
    their own TracerProvider."""
    if providers is None:
        return
    if isinstance(providers, str):
        if providers != "auto":
            raise ValueError(f'providers must be a list of provider names or "auto", got {providers!r}')
        providers = _detect_providers()
        logger.info('providers="auto" resolved to: %s', ", ".join(providers) or "none")
    elif providers == []:  # empty list means "all"
        providers = list(_PROVIDER_MODULES)
    logger.info(f"Enabling tracing for providers: {providers}")
    for name in providers:
        spec = _PROVIDER_MODULES.get(name)
        if spec is None:
            logger.warning(f"Unknown tracing provider: {name!r}")
            continue
        _enable_provider(name, *spec)


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


# ---------------------------------------------------------------------------
# Span stamping — single owner
#
# Every "what does this span get stamped with" decision lives here. Creation
# sites (observe / start_span / the turn registry) declare a span's shape as
# creation-time attributes via _declared_attributes(); the on-start processor
# then resolves the final stamps against the ambient context. A task scope
# entered inside an existing span labels it only through _task_scope_may_label.
# ---------------------------------------------------------------------------


def _boundary_kind(attributes: Mapping[str, Any]) -> str | None:
    """``"run"`` / ``"turn"`` / None from a span's own attributes; an entry
    point carrying no unit kind counts as a run boundary."""
    kind = attributes.get(attrs.UNIT_KIND)
    if kind in _UNIT_KINDS:
        return str(kind)
    if attributes.get(attrs.SPAN_TYPE) == SpanType.ENTRY_POINT.value:
        return "run"
    return None


def _has_local_parent(span) -> bool:
    """True when the span nests under an in-process span. A remote parent
    (TRACEPARENT into a subprocess) does not count: that process's entry
    point is still its own run boundary."""
    parent = getattr(span, "parent", None)
    return parent is not None and parent.is_valid and not parent.is_remote


def _declared_attributes(
    span_type: SpanType,
    provenance: str | None,
    unit: str | None,
    behaviour_key: str | None = None,
) -> dict[str, str]:
    """Creation-time declaration of a span's role: span type, provenance
    (explicit, else the type's natural class), unit kind (explicit, else
    ``"run"`` on entry points), and — for turn-registry spans — the behaviour
    key the span owns. Passed as start attributes so the on-start resolver
    sees the whole declaration."""
    declared: dict[str, str] = {attrs.SPAN_TYPE: span_type.value}
    if effective := provenance or _SPAN_TYPE_PROVENANCE.get(span_type):
        declared[attrs.PROVENANCE] = effective
    if kind := unit or ("run" if span_type is SpanType.ENTRY_POINT else None):
        declared[attrs.UNIT_KIND] = kind
    if behaviour_key:
        declared[attrs.BEHAVIOUR_KEY] = behaviour_key
    return declared


def _span_processor_on_start(span: trace.Span, parent_context: trace.Context | None = None):
    """Resolve every newborn span's final stamps — the one place unit and key
    decisions are made. States, from the span's declared attributes plus the
    ambient context:

    - every span: identity / workflow / conversation context values;
    - handoff boundary (first span in a pending-turn capability scope):
      ``unit_kind="turn"``, overriding even a declared run — a sub-run
      entered through a handoff is that handoff's unit;
    - declared run with a local parent: demoted to ``"turn"`` — one run
      boundary per trace, the root;
    - declared run at the root: stays ``"run"``, never carries a behaviour key;
    - turn unit / interior span: ambient behaviour key, unless the span
      declared its own key (a turn-registry span is never re-keyed).
    """
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

    declared = getattr(span, "attributes", None) or {}
    kind = _boundary_kind(declared)
    pending = get_value(_CTX_KEY_PENDING_TURN)
    if isinstance(pending, _PendingTurn) and not pending.consumed:
        pending.consumed = True
        kind = "turn"
    elif kind == "run" and _has_local_parent(span):
        kind = "turn"
    if kind is not None and declared.get(attrs.UNIT_KIND) != kind:
        span.set_attribute(attrs.UNIT_KIND, kind)

    if (behaviour_key := get_value(attrs.BEHAVIOUR_KEY)) and kind != "run" and attrs.BEHAVIOUR_KEY not in declared:
        span.set_attribute(attrs.BEHAVIOUR_KEY, str(behaviour_key))


def _task_scope_may_label(span) -> bool:
    """A task scope may label the span it was entered inside only when it can
    prove ownership: the span is recording, is not a unit boundary of any
    kind, and does not already carry a behaviour key set by another owner."""
    if not span.is_recording():
        return False
    attributes = getattr(span, "attributes", None) or {}
    return _boundary_kind(attributes) is None and attrs.BEHAVIOUR_KEY not in attributes


# ``init(export_orphan_spans=True)`` disables suppression; read dynamically so
# a re-init can flip it without rebuilding the provider.
_export_orphan_spans = False
_orphan_suppressed_logged = False


class _OrphanSpanSampler(Sampler):
    """Suppress orphan fragments: a declared ``function`` span that starts a
    NEW local trace (no parent, no boundary declaration) is sampled out, and
    its children fall with it. Everything deliberate still exports — boundary
    declarations (``@entry_point``, ``start_span(unit=...)``, ``run()``),
    other declared span types (a bare ``@tool`` / ``@workflow`` root is a
    choice), foreign spans with no declaration (auto-instrumented roots), and
    anything continuing a remote parent (``TRACEPARENT``)."""

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ) -> SamplingResult:
        parent = trace.get_current_span(parent_context).get_span_context()
        if parent.is_valid:
            keep = parent.is_remote or parent.trace_flags.sampled
            return SamplingResult(Decision.RECORD_AND_SAMPLE if keep else Decision.DROP, attributes)
        declared = attributes or {}
        is_orphan = _boundary_kind(declared) is None and declared.get(attrs.SPAN_TYPE) == SpanType.FUNCTION.value
        if _export_orphan_spans or not is_orphan:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
        global _orphan_suppressed_logged
        if not _orphan_suppressed_logged:
            _orphan_suppressed_logged = True
            logger.warning(
                "span %r started a new trace outside any run boundary and was not exported. "
                "Wrap the call in overmind.run(...) or @overmind.entry_point, or pass "
                "init(export_orphan_spans=True) to export orphan spans.",
                name,
            )
        return SamplingResult(Decision.DROP, attributes)

    def get_description(self) -> str:
        return "OvermindOrphanSpanSampler"


def _span_attribute_sink(span) -> dict[str, Any] | None:
    """Writable attribute map on an ended span, or ``None`` if unavailable."""
    target = getattr(span, "_attributes", None)
    if target is None:
        return None
    # BoundedAttributes (1.43+) is read-only once the span ends; its plain
    # ``_dict`` backing store still accepts writes and is what ``attributes``
    # proxies. Older SDKs accept assignment on the object itself.
    return getattr(target, "_dict", target)


def _is_legal_otel_attribute(value: Any) -> bool:
    if isinstance(value, (bool, str, int, float)):
        return True
    if isinstance(value, (list, tuple)) and all(isinstance(item, (bool, str, int, float)) for item in value):
        return True
    return False


def _encode_otlp_attribute(key: str, value: Any) -> None:
    """Raise if the OTLP protobuf encoder cannot represent *value*."""
    from opentelemetry.exporter.otlp.proto.common._internal import _encode_key_value

    _encode_key_value(key, value)


def _sanitize_span_attributes(span) -> None:
    """Force ``inputs`` / ``outputs`` (and any illegal value) to OTLP-safe
    primitives before the batch exporter runs.

    The OTLP protobuf encoder hits ``DecodeError`` on some structured
    ``AnyValue`` payloads (dicts, nested lists, protobuf messages). JSON
    strings encode cleanly.
    """
    sink = _span_attribute_sink(span)
    if not sink:
        return
    for key, value in list(sink.items()):
        safe = value
        if key in {"inputs", "outputs"} or not _is_legal_otel_attribute(value):
            safe = value if isinstance(value, str) else _coerce_to_otel_attribute(value)
            if key in {"inputs", "outputs"} and not isinstance(safe, str):
                safe = _json_dumps(value)
        try:
            _encode_otlp_attribute(str(key), safe)
        except Exception:
            safe = _json_dumps(value) if not isinstance(safe, str) else safe.encode("utf-8", "replace").decode("utf-8")
            try:
                _encode_otlp_attribute(str(key), safe)
            except Exception:
                safe = f"<unencodable {type(value).__name__}>"
        if safe is not value:
            try:
                sink[key] = safe
            except Exception:
                logger.debug("could not sanitize attribute %s", key, exc_info=True)


class _GenAiUsageSpanProcessor(SpanProcessor):
    """Mirror OTel ``gen_ai.*`` usage onto canonical ``genai.*`` keys at span
    end, so auto-instrumented spans carry the tokens/cost keys the server reads.

    Also sanitizes attributes the OTLP exporter cannot encode (notably structured
    ``inputs`` / ``outputs``). Must run before the exporting processor.

    ponytail: mutates ``span._attributes`` (ReadableSpan has no set_attribute).
    """

    def on_start(self, span: trace.Span, parent_context: trace.Context | None = None) -> None:
        return

    def on_end(self, span) -> None:
        try:
            self.patch_on_end(span)
        except Exception:
            logger.debug("genai enrichment could not set attributes", exc_info=True)

    def patch_on_end(self, span) -> None:
        _sanitize_span_attributes(span)
        updates = canonical_usage_updates(span.attributes or {})
        if not updates:
            return
        sink = _span_attribute_sink(span)
        if sink is None:
            return
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
        attach(set_value(attrs.AGENT_ID, str(agent_id)))
    if agent_name:
        attach(set_value(attrs.AGENT_NAME, str(agent_name)))
    if project_id:
        attach(set_value(attrs.PROJECT_ID, str(project_id)))


def _enable_debug_logging() -> None:
    """Make the ``overmind`` logger tree visible even when the host app never
    configured logging: DEBUG level plus one stderr handler."""
    overmind_logger = logging.getLogger("overmind")
    overmind_logger.setLevel(logging.DEBUG)
    if not any(isinstance(handler, logging.StreamHandler) for handler in overmind_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[overmind] %(levelname)s %(name)s: %(message)s"))
        overmind_logger.addHandler(handler)


def _log_debug_summary(
    endpoint: str,
    agent_id: str | None,
    agent_name: str | None,
    project_id: str | None,
    flush_interval_ms: int | None = None,
    max_batch_size: int | None = None,
) -> None:
    export = (
        f"batch (flush_interval_ms={flush_interval_ms}, max_batch_size={max_batch_size})"
        if flush_interval_ms is not None
        else "pre-configured provider"
    )
    logger.info(
        "Overmind debug: endpoint=%s | agent_id=%s agent_name=%s project_id=%s | providers=%s | export=%s | export_orphan_spans=%s",
        endpoint,
        agent_id,
        agent_name,
        project_id,
        ", ".join(sorted(_providers)) or "none",
        export,
        _export_orphan_spans,
    )


def init(
    overmind_api_key: str | None = None,
    *,
    service_name: str | None = None,
    environment: str | None = None,
    providers: list[str] | str | None = None,
    overmind_base_url: str | None = None,
    agent_id: str | None = None,
    agent_name: str | None = None,
    project_id: str | None = None,
    redact_keys: Iterable[str] | None = None,
    export_orphan_spans: bool = False,
    debug: bool = False,
) -> bool:
    """Initialise the Overmind SDK for automatic monitoring.

    Returns True when tracing is active. Without an API key (and outside
    strict mode) it logs, leaves every helper a no-op, and returns False —
    safe to call unconditionally in apps where Overmind is optional.
    Idempotent and thread-safe; re-init refreshes identity, providers, and
    the orphan-export policy only.

    Args:
        overmind_api_key: API key; defaults to OVERMIND_API_KEY env var.
        service_name: Service name in traces; defaults to OVERMIND_SERVICE_NAME.
        environment: e.g. "production"; defaults to OVERMIND_ENVIRONMENT or "development".
        providers: Providers to auto-instrument: "openai", "anthropic", "google",
            "agno", "langchain" (LangChain + LangGraph; needs the
            ``overmind[langchain]`` extra). ``"auto"`` detects and enables
            every provider whose library and instrumentor are both installed.
        overmind_base_url: Trace endpoint base URL; defaults to Overmind Cloud.
        agent_id: Agent UUID (preferred over agent_name); defaults to OVERMIND_AGENT_ID.
        agent_name: Human-readable agent name; defaults to OVERMIND_AGENT_NAME.
        project_id: Project UUID, only needed for session auth; defaults to OVERMIND_PROJECT_ID.
        redact_keys: Extra dict keys (exact, case-insensitive) redacted from
            captured inputs/outputs, on top of the built-in secret patterns.
        export_orphan_spans: Export ``function`` spans that start a new trace
            outside any run boundary. Off by default — the platform quarantines
            such single-fragment traces as noise (see
            ``docs/tracing-attributes.md`` §7).
        debug: Log a one-line setup summary (endpoint, identity, enabled
            instrumentors, export mode) and raise the ``overmind`` logger to
            DEBUG with a stderr handler.
    """
    global _initialized, _tracer, _extra_redact_keys, _keyless_logged, _export_orphan_spans

    if debug:
        _enable_debug_logging()

    agent_id = agent_id or os.environ.get("OVERMIND_AGENT_ID")
    agent_name = agent_name or os.environ.get("OVERMIND_AGENT_NAME")
    project_id = project_id or os.environ.get("OVERMIND_PROJECT_ID")
    _export_orphan_spans = bool(export_orphan_spans)

    if redact_keys:
        _extra_redact_keys = _extra_redact_keys | {str(k).lower() for k in redact_keys}

    with _init_lock:
        if _initialized:
            # Re-init only refreshes identity + providers; exporters stay as-is.
            logger.debug(f"Overmind SDK already initialised, reinitialising with providers: {providers}")
            _seed_identity_context(agent_id, agent_name, project_id)
            enable_tracing(providers)
            if debug:
                _log_debug_summary("(unchanged — already initialised)", agent_id, agent_name, project_id)
            return True

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
            if debug:
                _log_debug_summary(f"file:{os.environ['OVERMIND_TRACE_FILE']}", agent_id, agent_name, project_id)
            return True

        overmind_api_key = overmind_api_key or os.getenv("OVERMIND_API_KEY")
        if not overmind_api_key:
            msg = (
                "Overmind tracing disabled: no OVERMIND_API_KEY. "
                "Create a key at https://console.overmindlab.ai/projects"
            )
            if _strict_mode:
                raise RuntimeError(msg)
            logger.log(logging.DEBUG if _keyless_logged else logging.INFO, msg)
            _keyless_logged = True
            return False
        overmind_base_url = (overmind_base_url or DEFAULT_BASE_URL).rstrip("/")

        environment = (
            environment or os.environ.get("OVERMIND_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "development"
        )

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

        provider = TracerProvider(resource=resource, sampler=_OrphanSpanSampler())
        # Must run before the exporting processor so its on-end mutation is exported.
        provider.add_span_processor(_GenAiUsageSpanProcessor())
        provider.add_span_processor(_TurnLifecycleSpanProcessor())

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
        span_processor.on_start = _span_processor_on_start  # type: ignore[method-assign]

        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer("overmind", sdk_version)
        _seed_identity_context(agent_id, agent_name, project_id)
        enable_tracing(providers)
        _attach_remote_parent_if_present()

        _initialized = True
        logger.info(f"Overmind SDK initialised: service={service_name}, environment={environment}")
        if debug:
            _log_debug_summary(
                endpoint,
                agent_id,
                agent_name,
                project_id,
                flush_interval_ms=schedule_delay_millis,
                max_batch_size=max_export_batch_size,
            )
        return True


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


def set_conversation_id(conversation_id: str) -> None:
    """Tag downstream spans with a stable ``conversation.id`` for session grouping."""
    attach(set_value(attrs.CONVERSATION_ID, conversation_id))


# ---------------------------------------------------------------------------
# Capability scoping
# ---------------------------------------------------------------------------


# Matches the platform's Capability.slug convention: lowercase,
# non-alphanumeric runs collapse to single hyphens.
_IDENTITY_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _identity_slug(value: str) -> str:
    return _IDENTITY_SLUG_RE.sub("-", value.lower()).strip("-")


def _is_handoff(name: str | None, id: str | None) -> bool:
    """Entering a capability that differs from the active one, mid-trace.

    Identity is compared on the finest shared grain: ids when both sides
    have one (a name never shadows an id), else names on the slug grain —
    the server resolves a slug and its display spelling to the same
    capability, so they are not different identities. Mixed grains (id-only
    scope under name-only identity) are never treated as a handoff — a
    boundary is only declared when the identities are provably different."""
    if not trace.get_current_span().get_span_context().is_valid:
        return False
    active_id = get_value(attrs.AGENT_ID)
    active_name = get_value(attrs.AGENT_NAME)
    if id and active_id:
        return str(id) != str(active_id)
    if name and active_name:
        return _identity_slug(str(name)) != _identity_slug(str(active_name))
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
        self._token: Any = None

    def __enter__(self) -> _CapabilityScope:
        ctx = set_value(_CTX_KEY_PENDING_TURN, _PendingTurn() if _is_handoff(self._name, self._id) else None)
        # Both keys are always written: a name-only scope must not inherit the
        # outer scope's id (the server resolves id before name).
        ctx = set_value(attrs.AGENT_NAME, self._name, ctx)
        ctx = set_value(attrs.AGENT_ID, self._id, ctx)
        # Behaviour keys are capability-scoped: a key declared under the outer
        # capability means nothing here, so the ambient key resets with the
        # identity (detach restores it once this scope closes).
        ctx = set_value(attrs.BEHAVIOUR_KEY, None, ctx)
        self._token = attach(ctx)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            detach(self._token)
            self._token = None

    async def __aenter__(self) -> _CapabilityScope:
        return self.__enter__()

    async def __aexit__(self, *exc) -> None:
        self.__exit__(*exc)

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
    """Declare that all work inside belongs to one capability.

    ``id`` — the capability's UUID from the Console — is the identifier the
    server resolves first and is stable through renames; pin it whenever you
    have it. The positional ``name`` accepts the capability's slug (stable
    through renames, recommended) or its display name (a mutable label the
    server resolves through its alias table); it is safe to send alongside
    ``id`` but never load-bearing when an id is present.

    Usable as a context manager (``with`` / ``async with``) or decorator.
    Every span created inside carries ``overmind.agent.id`` / ``.name``; on
    exit the outer identity is restored. Entering a *different* capability
    mid-trace is a handoff: the first span of the new scope is stamped
    ``overmind.unit_kind = "turn"`` so the platform opens a new scoring unit.
    The identity must be one the project declared — nothing is auto-created."""
    return _CapabilityScope(name, id)


class _TurnRegistry:
    """Open turn-unit spans keyed by (trace_id, behaviour key).

    A behaviour's activity is non-contiguous in loop-shaped agents (debate
    rounds interleave with other phases), so its turn span outlives each task
    scope; it ends when the trace's run-boundary span ends (or at flush as a
    backstop), with the last scope-exit time so durations stay truthful."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: dict[tuple[int, str], trace.Span] = {}
        self._last_activity_ns: dict[tuple[int, str], int] = {}

    def get_or_start(self, key: str) -> trace.Span:
        ambient = trace.get_current_span().get_span_context()
        with self._lock:
            if ambient.is_valid and (span := self._spans.get((ambient.trace_id, key))) is not None:
                return span
            span = get_tracer().start_span(
                key, attributes=_declared_attributes(SpanType.FUNCTION, None, "turn", behaviour_key=key)
            )
            self._spans[(span.get_span_context().trace_id, key)] = span
            return span

    def touch(self, span: trace.Span, key: str) -> None:
        with self._lock:
            self._last_activity_ns[(span.get_span_context().trace_id, key)] = time.time_ns()

    def end_for_trace(self, trace_id: int) -> None:
        self._end(lambda entry: entry[0] == trace_id)

    def end_all(self) -> None:
        self._end(lambda entry: True)

    def _end(self, match: Callable[[tuple[int, str]], bool]) -> None:
        with self._lock:
            entries = [entry for entry in self._spans if match(entry)]
            ended = [(self._spans.pop(entry), self._last_activity_ns.pop(entry, None)) for entry in entries]
        # end() re-enters the processor chain (export, _span_processor_on_end),
        # so it must run outside the lock.
        for span, end_ns in ended:
            span.end(end_time=end_ns)


_turn_registry = _TurnRegistry()


def _span_processor_on_end(span) -> None:
    """End the trace's open turn spans when its run-boundary span ends."""
    if _boundary_kind(getattr(span, "attributes", None) or {}) != "run":
        return
    ctx = span.get_span_context()
    if ctx.is_valid:
        _turn_registry.end_for_trace(ctx.trace_id)


class _TurnLifecycleSpanProcessor(SpanProcessor):
    """Carrier for :func:`_span_processor_on_end` — unlike ``on_start``, the
    batch processor's ``on_end`` does the exporting and cannot be overridden."""

    def on_start(self, span: trace.Span, parent_context: trace.Context | None = None) -> None:
        return

    def on_end(self, span) -> None:
        _span_processor_on_end(span)


class _TaskScope:
    """Context manager and decorator produced by :func:`task`. Stamps
    ``overmind.behaviour.key`` on every span created inside, and on the span
    it was entered inside when :func:`_task_scope_may_label` proves ownership
    (never a unit boundary, never over an existing key); exiting restores the
    outer key. With ``unit="turn"`` the scope instead makes the behaviour's
    turn span (lazily created, re-used across re-entries) the current span."""

    def __init__(self, key: str, unit: str | None = None) -> None:
        key = (key or "").strip()
        if not key:
            raise ValueError("task() requires a key")
        if unit is not None and unit != "turn":
            raise ValueError(
                f'task() unit must be "turn", got {unit!r} — run boundaries are '
                'declared by entry points / start_span(unit="run")'
            )
        self._key = key
        self._unit = unit
        self._token: Any = None
        self._turn_span: trace.Span | None = None

    def __enter__(self) -> _TaskScope:
        ctx = set_value(attrs.BEHAVIOUR_KEY, self._key)
        if self._unit == "turn" and _initialized:
            self._turn_span = _turn_registry.get_or_start(self._key)
            ctx = trace.set_span_in_context(self._turn_span, ctx)
        self._token = attach(ctx)
        if self._turn_span is None:
            span = trace.get_current_span()
            if _task_scope_may_label(span):
                span.set_attribute(attrs.BEHAVIOUR_KEY, self._key)
        return self

    def __exit__(self, *exc) -> None:
        if self._turn_span is not None:
            _turn_registry.touch(self._turn_span, self._key)
            self._turn_span = None
        if self._token is not None:
            detach(self._token)
            self._token = None

    async def __aenter__(self) -> _TaskScope:
        return self.__enter__()

    async def __aexit__(self, *exc) -> None:
        self.__exit__(*exc)

    def __call__(self, func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with _TaskScope(self._key, self._unit):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _TaskScope(self._key, self._unit):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]


def task(key: str, *, unit: str | None = None) -> _TaskScope:
    """Declare the Behaviour.slug this work belongs to.

    Usable as a context manager (``with`` / ``async with``) or decorator.
    Optional — the server binds structurally when this is absent. No-op when
    nothing is recording (SDK not initialised). Restores the outer key on exit.

    ``unit="turn"`` additionally makes the scope a scoring unit: it lazily
    opens one turn span per (trace, key) that spans created inside nest under.
    Re-entering the same key re-uses the still-open span, so a phase's
    non-contiguous activity lands in one unit; the span ends when the trace's
    run-boundary span ends, at the phase's last scope-exit time."""
    return _TaskScope(key, unit)


# ---------------------------------------------------------------------------
# Span types and evidence contract
# ---------------------------------------------------------------------------


class SpanType(str, Enum):
    FUNCTION = "function"
    ENTRY_POINT = "entry_point"
    WORKFLOW = "workflow"
    TOOL = "tool_call"
    LLM = "llm_call"
    RETRIEVAL = "retrieval"


def _coerce_span_type(value: SpanType | str) -> SpanType:
    """Accept SpanType members, wire values ("tool_call"), or friendly names
    ("tool", "llm", "entry_point")."""
    if isinstance(value, SpanType):
        return value
    try:
        return SpanType(value)
    except ValueError:
        member = getattr(SpanType, str(value).upper(), None)
        if member is None:
            raise ValueError(f"unknown span type {value!r}") from None
        return member


_PROVENANCE_VALUES = frozenset({"user", "agent", "environment", "harness"})
_UNIT_KINDS = frozenset({"turn", "run"})
_CAPTURE_MODES = frozenset({"auto", "none", "messages"})

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


def _validate_unit(value: str | None) -> None:
    if value is not None and value not in _UNIT_KINDS:
        raise ValueError(f"unit must be one of {sorted(_UNIT_KINDS)}, got {value!r}")


# ---------------------------------------------------------------------------
# Evidence registry — environment-provenance spans collected per trace so
# deliver() can ground the terminal deliverable without app bookkeeping.
# Keyed by trace id (not a ContextVar): middleware and subagent hooks often
# run in isolated tasks/loops whose context never propagates back.
# ---------------------------------------------------------------------------

_MAX_TRACKED_TRACES = 256
_evidence_lock = threading.Lock()
_trace_evidence: dict[int, list[str]] = {}


def _current_trace_id() -> int | None:
    ctx = trace.get_current_span().get_span_context()
    return ctx.trace_id if ctx.is_valid else None


def _remember_evidence(otel_span) -> None:
    ctx = otel_span.get_span_context()
    if not ctx.is_valid:
        return
    with _evidence_lock:
        bucket = _trace_evidence.get(ctx.trace_id)
        if bucket is None:
            while len(_trace_evidence) >= _MAX_TRACKED_TRACES:
                _trace_evidence.pop(next(iter(_trace_evidence)))
            bucket = _trace_evidence[ctx.trace_id] = []
        bucket.append(format(ctx.span_id, "016x"))


def _pop_evidence(trace_id: int | None) -> list[str]:
    if trace_id is None:
        return []
    with _evidence_lock:
        return _trace_evidence.pop(trace_id, [])


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

    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        otel_span.set_attribute(attrs.STATUS, "cancelled")
        otel_span.record_exception(exc)
        otel_span.set_status(Status(StatusCode.ERROR, f"Cancelled ({type(exc).__name__})"))
        return

    otel_span.set_attribute(attrs.STATUS, "failed")
    otel_span.set_attribute(attrs.ERROR_TYPE, type(exc).__name__)
    otel_span.set_attribute(attrs.ERROR_MESSAGE, str(exc)[:1024])
    otel_span.record_exception(exc)
    otel_span.set_status(Status(StatusCode.ERROR, str(exc)))


def _bound_arguments(func: Callable, args: tuple, kwargs: dict, ignore: frozenset[str]) -> dict[str, Any]:
    """Name the positional args, merge kwargs, and drop self/cls, ignored
    names, and never-serialised runtime types."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    is_method = bool(param_names) and param_names[0] in ("self", "cls")
    start_idx = 1 if is_method else 0

    bound: dict[str, Any] = {}
    for i, arg in enumerate(args[start_idx:], start=start_idx):
        bound[param_names[i] if i < len(param_names) else f"arg_{i}"] = arg
    bound.update(kwargs)
    return {k: v for k, v in bound.items() if k not in ignore and not _should_skip_value(v)}


def _capture_inputs(
    otel_span,
    bound: dict[str, Any],
    capture: str,
    format_input: Callable[[dict[str, Any]], Any] | None,
) -> None:
    """Attach the call's inputs per the capture mode (best-effort)."""
    try:
        if format_input is not None:
            payload: Any = format_input(bound)
        elif capture == "messages":
            payload = {"messages": normalize_messages(bound.get("messages") or bound.get("input_messages"))}
        else:
            payload = bound
        otel_span.set_attribute("inputs", _json_dumps(payload))
    except Exception:
        logger.debug("observe(): input capture failed", exc_info=True)


def _capture_output(
    otel_span,
    result: Any,
    capture: str = "auto",
    format_output: Callable[[Any, dict[str, Any]], Any] | None = None,
    bound: dict[str, Any] | None = None,
) -> None:
    """Serialise the return value per the capture mode (best-effort)."""
    try:
        if format_output is not None:
            payload: Any = format_output(result, bound or {})
        elif capture == "messages" and isinstance(result, list):
            payload = {"messages": normalize_messages(result)}
        else:
            payload = result
        otel_span.set_attribute("outputs", _json_dumps(payload))
    except Exception:
        logger.debug("observe(): output capture failed", exc_info=True)


def code_identity_attributes(func: Callable) -> dict[str, str]:
    """``code.namespace`` / ``code.function.name`` from the *unwrapped*
    function so the server can bind the span to a code-symbol anchor
    (``module.qualname``). Best-effort, never raises."""
    out: dict[str, str] = {}
    try:
        target = inspect.unwrap(func)
        if module := getattr(target, "__module__", None):
            out[attrs.CODE_NAMESPACE] = module
        if qualname := getattr(target, "__qualname__", None):
            out[attrs.CODE_FUNCTION_NAME] = qualname
    except Exception:
        logger.debug("observe(): code identity capture failed", exc_info=True)
    return out


def _stamp_tool_metadata(otel_span, tool_name: str, bound: dict[str, Any]) -> None:
    """Stamp ``tool.name`` / ``tool.arg_keys`` (keys only) on a tool span."""
    try:
        otel_span.set_attribute(attrs.TOOL_NAME, tool_name)
        if bound:
            otel_span.set_attribute(attrs.TOOL_ARG_KEYS, list(bound.keys()))
    except Exception:
        logger.debug("tool(): metadata capture failed for %s", tool_name, exc_info=True)


class _Outcome:
    """Return-value cell the observe wrappers fill before the span closes."""

    __slots__ = ("result",)

    def __init__(self) -> None:
        self.result: Any = None


@contextmanager
def _traced_call(
    func: Callable,
    args: tuple,
    kwargs: dict,
    *,
    name: str,
    tool_name: str,
    span_type: SpanType,
    declared: Mapping[str, str],
    capability_name: str | None,
    capability_id: str | None,
    capture: str,
    ignore: frozenset[str],
    format_input: Callable | None,
    format_output: Callable | None,
):
    """Shared body of the observe wrappers: capability scope, declared span
    attributes, input/output capture, lifecycle finalisation, cancellation
    flush."""
    scope = _CapabilityScope(capability_name, capability_id) if capability_name or capability_id else nullcontext()
    try:
        with scope, get_tracer().start_as_current_span(name, attributes=declared) as otel_span:
            bound = {}
            try:
                bound = _bound_arguments(func, args, kwargs, ignore)
            except Exception:
                logger.debug("observe(): argument binding failed for %s", name, exc_info=True)
            if span_type is SpanType.TOOL:
                _stamp_tool_metadata(otel_span, tool_name, bound)
            if capture != "none":
                _capture_inputs(otel_span, bound, capture, format_input)
            outcome = _Outcome()
            start = time.monotonic()
            try:
                yield outcome
            except BaseException as exc:
                if span_type is SpanType.TOOL:
                    otel_span.set_attribute(attrs.TOOL_ERROR, type(exc).__name__)
                _finalize_span(otel_span, exc, start)
                raise
            if capture != "none":
                _capture_output(otel_span, outcome.result, capture, format_output, bound)
            if declared.get(attrs.PROVENANCE) == "environment":
                _remember_evidence(otel_span)
            _finalize_span(otel_span, None, start)
    except (KeyboardInterrupt, asyncio.CancelledError):
        # The span already ended (the with-block unwound); flushing the run
        # root keeps a cancellation from leaving the trace rootless until
        # the batch timeout.
        if span_type is SpanType.ENTRY_POINT:
            force_flush_traces(timeout_millis=5000)
        raise


def observe(
    span_name: str | Callable[..., str] | None = None,
    type: SpanType | str = SpanType.FUNCTION,
    *,
    provenance: str | None = None,
    unit: str | None = None,
    capability: str | None = None,
    capability_id: str | None = None,
    capture: str = "auto",
    ignore: Iterable[str] = (),
    format_input: Callable[[dict[str, Any]], Any] | None = None,
    format_output: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> Callable[[F], F]:
    """Decorator (sync or async) that traces a function with the full
    evidence contract. No-op call-through when the SDK is not initialised.

    Args:
        span_name: Span name; defaults to the function's ``__qualname__``.
            A callable receives each call's arguments and names the span
            per invocation — for polymorphic dispatchers, where the span
            (and a tool span's ``tool.name``) must follow the dispatched
            action, e.g. ``name=lambda self, action, **p: action.name``.
        type: Span type — a :class:`SpanType` or a name like ``"tool"`` /
            ``"llm"`` / ``"entry_point"``.
        provenance: Evidence provenance class (``user`` / ``agent`` /
            ``environment`` / ``harness``); tool, retrieval and LLM spans get
            their natural class automatically.
        unit: ``"run"`` / ``"turn"`` scoring-unit marker; entry points are
            marked ``"run"`` automatically.
        capability: Capability slug or display name to scope this span *and
            its children* to (see :func:`capability`); a differing identity
            mid-trace marks a handoff boundary.
        capability_id: Capability UUID — the recommended identifier, stable
            through renames; the server resolves it before any name. Send it
            with or without ``capability``.
        capture: ``"auto"`` (scrubbed args/result), ``"none"`` (no payloads),
            or ``"messages"`` (normalise the ``messages`` argument and a
            list result into role/content chat evidence).
        ignore: Argument names never captured (heavy runtime objects,
            sessions, model handles).
        format_input: Optional ``fn(bound_args) -> payload`` overriding input
            capture.
        format_output: Optional ``fn(result, bound_args) -> payload``
            overriding output capture.
    """
    span_type = _coerce_span_type(type)
    _validate_provenance(provenance)
    _validate_unit(unit)
    if capture not in _CAPTURE_MODES:
        raise ValueError(f"capture must be one of {sorted(_CAPTURE_MODES)}, got {capture!r}")
    ignore_set = frozenset(ignore)

    def decorator(func: F) -> F:
        static_name = span_name if isinstance(span_name, str) else None
        call_kwargs = dict(
            name=static_name or func.__qualname__,
            tool_name=static_name or func.__name__,
            span_type=span_type,
            declared=_declared_attributes(span_type, provenance, unit) | code_identity_attributes(func),
            capability_name=capability,
            capability_id=capability_id,
            capture=capture,
            ignore=ignore_set,
            format_input=format_input,
            format_output=format_output,
        )

        def resolved_kwargs(args: tuple, kwargs: dict) -> dict:
            if not callable(span_name):
                return call_kwargs
            try:
                name = str(span_name(*args, **kwargs))
            except Exception:
                logger.debug("observe(): span name callable failed for %s", func.__qualname__, exc_info=True)
                return call_kwargs
            return {**call_kwargs, "name": name, "tool_name": name}

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not _initialized:
                    return await func(*args, **kwargs)
                with _traced_call(func, args, kwargs, **resolved_kwargs(args, kwargs)) as outcome:
                    outcome.result = await func(*args, **kwargs)
                    return outcome.result

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _initialized:
                return func(*args, **kwargs)
            with _traced_call(func, args, kwargs, **resolved_kwargs(args, kwargs)) as outcome:
                outcome.result = func(*args, **kwargs)
                return outcome.result

        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def start_span(
    name: str,
    span_type: SpanType | str = SpanType.FUNCTION,
    attributes: Mapping[str, Any] | None = None,
    *,
    provenance: str | None = None,
    unit: str | None = None,
):
    """Context-manager companion to :func:`observe`; stamps the same canonical
    span metadata. Yields a non-recording span when the SDK is uninitialised,
    so instrumentation never crashes an app without an API key."""
    span_type = _coerce_span_type(span_type)
    _validate_provenance(provenance)
    _validate_unit(unit)
    if not _initialized:
        yield trace.INVALID_SPAN
        return
    declared = _declared_attributes(span_type, provenance, unit)
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=declared) as otel_span:
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
            if declared.get(attrs.PROVENANCE) == "environment":
                _remember_evidence(otel_span)
            _finalize_span(otel_span, None, start)


@contextmanager
def _start_child_span(
    name: str,
    *,
    span_type: SpanType = SpanType.FUNCTION,
    provenance: str | None = None,
):
    """Open a span as an explicit child of the current OTel span; re-attaching
    the parent keeps the tree stable across mixed instrumentation stacks."""
    current = trace.get_current_span()
    token = None
    try:
        if current is not None and current.get_span_context().is_valid:
            token = attach(trace.set_span_in_context(current))
        with start_span(name, span_type=span_type, provenance=provenance) as span:
            yield span
    finally:
        if token is not None:
            detach(token)


def entry_point(name: str | None = None, **kwargs) -> Callable[[F], F]:
    """Decorator that traces an entry point span (marked ``unit_kind = "run"``)."""
    return observe(span_name=name, type=SpanType.ENTRY_POINT, **kwargs)


def workflow(name: str | None = None, **kwargs) -> Callable[[F], F]:
    """Decorator that traces a workflow span."""
    return observe(span_name=name, type=SpanType.WORKFLOW, **kwargs)


def tool(name: str | Callable[..., str] | None = None, **kwargs) -> Callable[[F], F]:
    """Decorator that traces a tool span (adds ``tool.name`` / ``tool.arg_keys``).

    ``name`` may be a callable receiving each call's arguments, so a
    polymorphic dispatcher emits per-action tool spans from one decoration:
    ``@tool(name=lambda self, action, **p: action)``. The span name and
    ``tool.name`` both follow the resolved value."""
    return observe(span_name=name, type=SpanType.TOOL, **kwargs)


def retrieval(name: str | None = None, **kwargs) -> Callable[[F], F]:
    """Decorator that traces a retrieval / RAG step span."""
    return observe(span_name=name, type=SpanType.RETRIEVAL, **kwargs)


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
    ``overmind.delivery = true``.

    ``grounded_by`` names the evidence spans the deliverable rests on
    (span_id hex strings or span handles). When omitted, the environment-
    provenance spans the SDK collected for the current trace are used —
    call inside the run so the trace is still active."""
    if not _initialized:
        return
    _validate_provenance(provenance)
    if grounded_by is None:
        grounded_by = _pop_evidence(_current_trace_id())
    with _start_child_span(name, provenance=provenance) as otel_span:
        otel_span.set_attribute(attrs.DELIVERY, True)
        if grounded_by:
            otel_span.set_attribute(attrs.GROUNDED_BY, json.dumps([_grounding_span_id(h) for h in grounded_by]))
        _capture_output(otel_span, payload)


def observe_safe(
    span_name: str | Callable[..., str] | None = None,
    type: SpanType | str = SpanType.FUNCTION,
    **kwargs: Any,
) -> Callable[[F], F]:
    """:func:`observe` with ``capture="none"``: never records arguments or
    return values. Manual escape hatch for code that handles credentials;
    prefer masking values before they reach traced functions."""
    return observe(span_name, type, capture="none", **kwargs)


def force_flush_traces(timeout_millis: int = 1000) -> None:
    """Best-effort exporter flush before exit; no-op if the provider
    lacks ``force_flush``. Ends any still-open turn spans first so a run
    that never closed its boundary span still exports its units."""
    _turn_registry.end_all()
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_millis)


__all__ = [
    "SpanType",
    "capability",
    "capture_exception",
    "deliver",
    "enable_tracing",
    "entry_point",
    "force_flush_traces",
    "get_api_settings",
    "get_tracer",
    "init",
    "normalize_messages",
    "observe",
    "retrieval",
    "serialize",
    "set_conversation_id",
    "set_tag",
    "set_user",
    "set_workflow_name",
    "start_span",
    "task",
    "tool",
    "workflow",
]
