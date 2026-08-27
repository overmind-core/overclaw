"""
Overmind Python Client

Overmind: autonomous agent optimisation through structured experimentation.
Overmind: automatic observability for LLM applications.

"""

__version__ = "0.1.60"

from opentelemetry.overmind.prompt import PromptString

from .client import Client, ModelDeleted, OvermindInferenceError
from .evals import checkpoint, end_conversation, eval_context, expect, intent
from .lifecycle import RunHandle, run
from .tracing import (
    SpanType,
    capability,
    capture_exception,
    deliver,
    entry_point,
    force_flush_traces,
    init,
    normalize_messages,
    observe,
    observe_safe,
    retrieval,
    set_conversation_id,
    set_tag,
    set_user,
    set_workflow_name,
    start_span,
    task,
    tool,
    workflow,
)

__all__ = [
    "Client",
    "ModelDeleted",
    "OvermindInferenceError",
    "PromptString",
    "RunHandle",
    "SpanType",
    "capability",
    "capture_exception",
    "checkpoint",
    "deliver",
    "end_conversation",
    "entry_point",
    "eval_context",
    "expect",
    "force_flush_traces",
    "init",
    "intent",
    "normalize_messages",
    "observe",
    "observe_safe",
    "retrieval",
    "run",
    "set_conversation_id",
    "set_tag",
    "set_user",
    "set_workflow_name",
    "start_span",
    "task",
    "tool",
    "workflow",
]
