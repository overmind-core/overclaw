"""
Overmind Python Client

Overmind: autonomous agent optimization through structured experimentation.
Overmind: automatic observability for LLM applications.

"""

__version__ = "0.1.48"

# from opentelemetry.overmind.prompt import PromptString

from .tracing import (
    SpanType,
    capture_exception,
    entry_point,
    function,
    get_tracer,
    init,
    observe,
    set_tag,
    set_user,
    start_span,
    tool,
    workflow,
)

__all__ = [
    # "PromptString",
    "SpanType",
    "capture_exception",
    "entry_point",
    "function",
    "get_tracer",
    "init",
    "observe",
    "set_tag",
    "set_user",
    "start_span",
    "tool",
    "workflow",
]
