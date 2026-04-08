"""Grep tool — search file contents using ripgrep."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from .base import ToolContext, ToolResult

MAX_MATCHES = 100
MAX_LINE_LEN = 2000

DESCRIPTION = (
    "Search file contents using regex pattern.\n\n"
    "Usage:\n"
    "- Uses ripgrep (rg) for fast searching.\n"
    "- Supports full regex syntax.\n"
    "- Use the glob parameter to filter files (e.g. '*.py').\n"
    "- Returns up to 100 matches."
)

PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search in (defaults to cwd)",
        },
        "glob": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g. '*.py')",
        },
    },
    "required": ["pattern"],
}


class GrepTool:
    name = "grep"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = params.get("pattern", "")
        search_path = params.get("path", ctx.cwd)
        if not os.path.isabs(search_path):
            search_path = os.path.join(ctx.cwd, search_path)
        glob_filter = params.get("glob")

        cmd = ["rg", "-nH", "--max-count", str(MAX_MATCHES), pattern, search_path]
        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=ctx.cwd,
            )
        except FileNotFoundError:
            return ToolResult(
                output="Error: ripgrep (rg) is not installed. Install it first."
            )
        except subprocess.TimeoutExpired:
            return ToolResult(output="Search timed out after 30 seconds.")

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        truncated = len(lines) > MAX_MATCHES
        lines = lines[:MAX_MATCHES]

        capped: list[str] = []
        for line in lines:
            if len(line) > MAX_LINE_LEN:
                line = line[:MAX_LINE_LEN] + "..."
            capped.append(line)

        if not capped:
            return ToolResult(output=f"No matches found for pattern: {pattern}")

        output = "\n".join(capped)
        if truncated:
            output += f"\n\n(Results capped at {MAX_MATCHES} matches)"
        return ToolResult(
            output=output,
            title=f"grep: {pattern}",
            metadata={"truncated": truncated},
        )
