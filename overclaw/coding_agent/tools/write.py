"""Write tool — full file overwrite with read-before-write enforcement."""

from __future__ import annotations

import difflib
import os
from typing import Any

from .base import ToolContext, ToolResult

DESCRIPTION = (
    "Writes a file to the local filesystem.\n\n"
    "Usage:\n"
    "- This tool will overwrite the existing file.\n"
    "- For existing files, you MUST read the file first.\n"
    "- Prefer editing existing files over creating new ones."
)

PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {
            "type": "string",
            "description": "Absolute path to the file to write",
        },
        "content": {
            "type": "string",
            "description": "The complete file content to write",
        },
    },
    "required": ["filePath", "content"],
}


class WriteTool:
    name = "write"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = params.get("filePath", "")
        fp = raw if os.path.isabs(raw) else os.path.join(ctx.cwd, raw)
        content = params.get("content", "")

        exists = os.path.isfile(fp)
        old = ""
        if exists:
            ctx.assert_read(fp)
            with open(fp) as f:
                old = f.read()

        parent = os.path.dirname(fp)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(fp, "w") as f:
            f.write(content)

        ctx.record_read(fp)

        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                content.splitlines(),
                fromfile=fp,
                tofile=fp,
                lineterm="",
            )
        )

        return ToolResult(
            output="File written successfully.",
            title=os.path.relpath(fp, ctx.worktree),
            metadata={"diff": diff},
        )
