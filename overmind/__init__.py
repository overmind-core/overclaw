"""
Overmind Python Client

Overmind: autonomous agent optimisation through structured experimentation.
Overmind: automatic observability for LLM applications.

"""

__version__ = "0.1.54"

from opentelemetry.overmind.prompt import PromptString

from .client import Client, ModelDeleted, OvermindInferenceError
from .evals import checkpoint, end_conversation, eval_context, expect, intent
from .tracing import (
    SpanType,
    capture_exception,
    conversation,
    deliver,
    entry_point,
    force_flush_traces,
    function,
    get_tracer,
    init,
    mark_unit,
    observe,
    retrieval,
    set_agent_id,
    set_agent_name,
    set_conversation_id,
    set_project_id,
    set_tag,
    set_user,
    set_workflow_name,
    start_span,
    tool,
    workflow,
)

__all__ = [
    "Client",
    "ModelDeleted",
    "OvermindInferenceError",
    "PromptString",
    "SpanType",
    "capture_exception",
    "checkpoint",
    "conversation",
    "deliver",
    "end_conversation",
    "entry_point",
    "eval_context",
    "expect",
    "force_flush_traces",
    "function",
    "get_tracer",
    "init",
    "intent",
    "mark_unit",
    "observe",
    "retrieval",
    "set_agent_id",
    "set_agent_name",
    "set_conversation_id",
    "set_project_id",
    "set_tag",
    "set_user",
    "set_workflow_name",
    "start_span",
    "tool",
    "workflow",
]
