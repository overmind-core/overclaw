"""Run-lifecycle scope: the bracket every integration otherwise hand-rolls.

``overmind.run(...)`` composes existing primitives — capability scope,
entry-point run span, intent, conversation id, tags, exception status, flush
on exit — and yields a handle whose ``deliver()`` captures the terminal
deliverable. All stamping decisions stay with the resolver in
:mod:`overmind.tracing`.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any

from overmind.evals import intent as _intent
from overmind.tracing import (
    SpanType,
    code_identity_attributes,
    deliver,
    force_flush_traces,
    set_conversation_id,
    start_span,
)
from overmind.tracing import capability as _capability_scope

logger = logging.getLogger(__name__)


class RunHandle:
    """Yielded by :func:`run`; ``span`` is the run-boundary span."""

    __slots__ = ("span",)

    def __init__(self, span) -> None:
        self.span = span

    def deliver(self, payload: Any, **kwargs) -> None:
        """Capture the run's terminal deliverable (see :func:`overmind.deliver`).

        Call it inside the unit that produced the deliverable (e.g. within its
        ``task(key, unit="turn")`` scope) so the delivery span nests there."""
        deliver(payload, **kwargs)


def _resolve(value: Any, args: tuple, kwargs: dict) -> Any:
    """Resolve a decorator-form parameter: a callable receives the wrapped
    call's arguments (``intent=lambda self, *a, **k: self.task``); anything
    else passes through. A failing callable degrades to None, never raises."""
    if not callable(value):
        return value
    try:
        return value(*args, **kwargs)
    except Exception:
        logger.debug("run(): callable parameter failed", exc_info=True)
        return None


@contextmanager
def _run_scope(
    name: str,
    capability: str | None,
    capability_id: str | None,
    intent: str | None,
    conversation_id: str | None,
    tags: Mapping[str, Any] | None,
    identity: Mapping[str, str] | None = None,
):
    capability = capability or os.environ.get("OVERMIND_AGENT_NAME")
    capability_id = capability_id or os.environ.get("OVERMIND_AGENT_ID")
    scope = _capability_scope(capability, id=capability_id) if capability or capability_id else nullcontext()
    attributes = dict(identity or {})
    if tags:
        attributes.update(tags)
    try:
        with scope:
            if conversation_id:
                set_conversation_id(str(conversation_id))
            with start_span(name, span_type=SpanType.ENTRY_POINT, unit="run", attributes=attributes or None) as span:
                if intent:
                    _intent(str(intent))
                yield RunHandle(span)
    finally:
        # After the boundary span ends, so still-open turn spans close under
        # it (turn lifecycle rides the run span's end) and export in the flush.
        force_flush_traces()


class _RunScope:
    """Context manager and decorator produced by :func:`run`."""

    def __init__(
        self,
        name: str | None,
        capability: str | Callable[..., str | None] | None,
        capability_id: str | Callable[..., str | None] | None,
        intent: str | Callable[..., str | None] | None,
        conversation_id: str | Callable[..., str | None] | None,
        tags: Mapping[str, Any] | Callable[..., Mapping[str, Any] | None] | None,
    ) -> None:
        self._name = name
        self._capability = capability
        self._capability_id = capability_id
        self._intent = intent
        self._conversation_id = conversation_id
        self._tags = tags
        self._cm: Any = None

    def __enter__(self) -> RunHandle:
        # Context-manager form has no wrapped call, so callables resolve
        # with no arguments (zero-arg callables work; others degrade to None).
        self._cm = _run_scope(
            self._name or "run",
            _resolve(self._capability, (), {}),
            _resolve(self._capability_id, (), {}),
            _resolve(self._intent, (), {}),
            _resolve(self._conversation_id, (), {}),
            _resolve(self._tags, (), {}),
        )
        return self._cm.__enter__()

    def __exit__(self, *exc) -> bool:
        cm, self._cm = self._cm, None
        return cm.__exit__(*exc)

    def _decorated_scope(self, func: Callable, args: tuple, kwargs: dict):
        return _run_scope(
            self._name or func.__qualname__,
            _resolve(self._capability, args, kwargs),
            _resolve(self._capability_id, args, kwargs),
            _resolve(self._intent, args, kwargs),
            _resolve(self._conversation_id, args, kwargs),
            _resolve(self._tags, args, kwargs),
            identity=code_identity_attributes(func),
        )

    def __call__(self, func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self._decorated_scope(func, args, kwargs):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self._decorated_scope(func, args, kwargs):
                return func(*args, **kwargs)

        return sync_wrapper


def run(
    name: str | None = None,
    *,
    capability: str | Callable[..., str | None] | None = None,
    capability_id: str | Callable[..., str | None] | None = None,
    intent: str | Callable[..., str | None] | None = None,
    conversation_id: str | Callable[..., str | None] | None = None,
    tags: Mapping[str, Any] | Callable[..., Mapping[str, Any] | None] | None = None,
) -> _RunScope:
    """One scope for a whole agent run — context manager or decorator.

    Enters the capability identity (explicit args, else the
    ``OVERMIND_AGENT_NAME`` / ``OVERMIND_AGENT_ID`` env vars, else no scope —
    ``init()``'s ambient identity already covers single-capability agents),
    opens the entry-point run span (``unit="run"``, so turn units opened
    inside close with it), declares the intent, and flushes the exporter once
    the boundary span has ended. Exceptions mark the run span failed and
    re-raise. No-op-safe when the SDK is uninitialised.

    As a context manager (``with overmind.run(...) as handle``) the span is
    named *name* (default ``"run"``) and the handle delivers the terminal
    payload.

    As a decorator (``@overmind.run(...)``, sync or async) every parameter
    except *name* also accepts a callable receiving the wrapped call's
    arguments — ``intent=lambda self, *a, **k: self.task`` — resolved per
    invocation. The span is named *name* (default the function's
    ``__qualname__``) and carries the function's ``code.namespace`` /
    ``code.function.name``, so one decoration satisfies an entry-point
    scan-contract anchor. The return value is not auto-delivered — call
    ``overmind.deliver()`` inside the unit that produced it.
    """
    if callable(name):
        raise TypeError("run() must be called: use @overmind.run(...) with parentheses")
    return _RunScope(name, capability, capability_id, intent, conversation_id, tags)


__all__ = ["RunHandle", "run"]
