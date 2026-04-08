"""Base tool definition and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    output: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Protocol that all tools must implement."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


@dataclass
class ToolContext:
    """Context passed to every tool execution."""

    session_id: str
    worktree: str
    cwd: str
    file_tracker: Any  # FileTracker instance

    def record_read(self, path: str) -> None:
        self.file_tracker.record_read(self.session_id, path)

    def assert_read(self, path: str) -> None:
        self.file_tracker.assert_fresh(self.session_id, path)


def schema_to_openai(tool: Tool) -> dict[str, Any]:
    """Convert a Tool into an OpenAI function-calling tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
