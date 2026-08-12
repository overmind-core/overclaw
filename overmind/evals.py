"""Runtime eval declarations — the ``overmind.eval.*`` envelope.

Each public function emits a span event on the current span; the Overmind
platform parses these server-side, so event names and payload shapes are a
pinned wire contract (v1) — see ``docs/tracing-attributes.md`` §6.  All
functions no-op (with a debug log) when there is no recording span.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from opentelemetry import trace

from overmind import attrs
from overmind.tracing import _coerce_to_otel_attribute, _json_dumps, _normalize_for_json

logger = logging.getLogger(__name__)

_EXPECT_KINDS = frozenset({"contains", "regex", "schema", "constraint", "checkpoints"})
_EXPECT_SCOPES = frozenset({"span", "trace", "conversation"})


def _emit(event_name: str, payload: dict[str, Any]) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        logger.debug("%s ignored: no recording span", event_name)
        return
    span.add_event(
        event_name,
        {attrs.EVAL_SCHEMA_VERSION: 1, attrs.EVAL_PAYLOAD: _json_dumps(payload)},
    )


def expect(
    kind: str,
    spec: Any,
    *,
    id: str | None = None,
    scope: str = "trace",
    gate: bool = False,
) -> None:
    """Declare a runtime expectation for server-side evaluation of this run.

    Args:
        kind: One of ``contains`` / ``regex`` / ``schema`` / ``constraint`` /
            ``checkpoints``.
        spec: What to check — a string (substring, regex, natural-language
            constraint) or an object (e.g. a JSON schema, or for
            ``checkpoints`` the ordered list of checkpoint names).
        id: Stable identifier; derived as a short hash of kind+spec when omitted.
        scope: What the expectation applies to: ``span`` / ``trace`` / ``conversation``.
        gate: When true, failing this expectation caps the score (hard fail).
    """
    if kind not in _EXPECT_KINDS:
        raise ValueError(f"expect() kind must be one of {sorted(_EXPECT_KINDS)}, got {kind!r}")
    if scope not in _EXPECT_SCOPES:
        raise ValueError(f"expect() scope must be one of {sorted(_EXPECT_SCOPES)}, got {scope!r}")
    spec = _normalize_for_json(spec)
    if id is None:
        # Stable across runs so the platform can dedupe/aggregate per expectation.
        canonical = json.dumps(spec, sort_keys=True, ensure_ascii=False)
        id = hashlib.sha256(f"{kind}:{canonical}".encode()).hexdigest()[:12]
    _emit(
        attrs.EVAL_EXPECTATION_EVENT,
        {"id": id, "kind": kind, "spec": spec, "scope": scope, "gate": bool(gate)},
    )


def eval_context(**facts: Any) -> None:
    """Attach runtime facts for the judge; values coerced like :func:`set_tag`."""
    _emit(
        attrs.EVAL_CONTEXT_EVENT,
        {"facts": {key: _coerce_to_otel_attribute(value) for key, value in facts.items()}},
    )


def checkpoint(name: str) -> None:
    """Mark a named trajectory milestone / turn boundary."""
    _emit(attrs.EVAL_CHECKPOINT_EVENT, {"name": name})


def end_conversation() -> None:
    """Signal the conversation is complete; triggers conversation-scope scoring."""
    _emit(attrs.EVAL_CONVERSATION_END_EVENT, {})


__all__ = ["checkpoint", "end_conversation", "eval_context", "expect"]
