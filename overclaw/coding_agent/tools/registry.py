"""Tool registry — resolves available tools for a given model."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult, schema_to_openai


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools_for_model(self, model_id: str) -> list[Tool]:
        """Return tools appropriate for the given model.

        GPT-5.x models get apply_patch instead of edit/write.
        All other models get edit/write instead of apply_patch.
        """
        use_patch = "gpt-" in model_id and "gpt-4" not in model_id
        result: list[Tool] = []
        for tool in self._tools.values():
            if tool.name == "apply_patch" and not use_patch:
                continue
            if tool.name in ("edit", "write") and use_patch:
                continue
            result.append(tool)
        return result

    def openai_schemas(self, model_id: str) -> list[dict[str, Any]]:
        return [schema_to_openai(t) for t in self.tools_for_model(model_id)]

    def execute(
        self, name: str, params: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(output=f"Unknown tool: {name}")
        try:
            return tool.execute(params, ctx)
        except Exception as exc:
            return ToolResult(output=f"Tool error: {exc}")
