"""Glob tool — find files by pattern using ripgrep."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from .base import ToolContext, ToolResult

MAX_FILES = 100

DESCRIPTION = (
    "Find files matching a glob pattern.\n\n"
    "Usage:\n"
    "- Patterns not starting with '**/' are auto-prepended with '**/' for recursive search.\n"
    "- Returns up to 100 matching file paths.\n"
    "- Results sorted by modification time."
)

PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files (e.g. '*.py', '**/test_*.py')",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (defaults to cwd)",
        },
    },
    "required": ["pattern"],
}


class GlobTool:
    name = "glob"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = params.get("pattern", "")
        search_path = params.get("path", ctx.cwd)
        if not os.path.isabs(search_path):
            search_path = os.path.join(ctx.cwd, search_path)

        if not pattern.startswith("**/"):
            pattern = "**/" + pattern

        cmd = ["rg", "--files", "--glob", pattern, search_path]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=ctx.cwd,
            )
        except FileNotFoundError:
            return ToolResult(output="Error: ripgrep (rg) is not installed.")
        except subprocess.TimeoutExpired:
            return ToolResult(output="Glob search timed out.")

        files = [line for line in result.stdout.strip().split("\n") if line]

        def mtime(p: str) -> float:
            try:
                return os.path.getmtime(p)
            except OSError:
                return 0

        files.sort(key=mtime, reverse=True)

        truncated = len(files) > MAX_FILES
        files = files[:MAX_FILES]

        if not files:
            return ToolResult(output=f"No files found matching: {pattern}")

        output = "\n".join(files)
        if truncated:
            output += f"\n\n(Showing first {MAX_FILES} files)"
        return ToolResult(
            output=output,
            title=f"glob: {pattern}",
            metadata={"truncated": truncated},
        )
