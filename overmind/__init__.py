"""
Overmind Python Client

Overmind: autonomous agent optimisation through structured experimentation.
Overmind: automatic observability for LLM applications.

"""

__version__ = "0.1.58"

from opentelemetry.overmind.prompt import PromptString

from .client import Client, ModelDeleted, OvermindInferenceError
from .tracing import (
    SpanType,
    capture_exception,
    conversation,
    entry_point,
    force_flush_traces,
    function,
    get_tracer,
    init,
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
    "conversation",
    "entry_point",
    "force_flush_traces",
    "function",
    "get_tracer",
    "init",
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
