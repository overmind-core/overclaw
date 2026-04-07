"""Bash tool — execute shell commands with timeout."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from .base import ToolContext, ToolResult
from ..truncate import truncate

DEFAULT_TIMEOUT = 120

DESCRIPTION = (
    "Execute a shell command.\n\n"
    "Usage:\n"
    "- Run commands in the project directory.\n"
    "- Use the 'timeout' parameter to set a max duration in seconds.\n"
    "- Commands that take longer than the timeout are killed.\n"
    "- Prefer dedicated tools (read, edit, grep, glob) over shell equivalents."
)

PARAMS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 120)",
        },
        "description": {
            "type": "string",
            "description": "Short description of what this command does (5-10 words)",
        },
    },
    "required": ["command", "description"],
}


class BashTool:
    name = "bash"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = params.get("command", "")
        timeout = params.get("timeout", DEFAULT_TIMEOUT)
        desc = params.get("description", "")

        timed_out = False
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=ctx.cwd,
                env={**os.environ, "TERM": "dumb"},
            )
            output = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            output = (exc.stdout or b"").decode(errors="replace") + (
                exc.stderr or b""
            ).decode(errors="replace")
            exit_code = -1
            output += (
                f"\n\nbash tool terminated command after exceeding timeout {timeout}s"
            )

        output, was_truncated = truncate(output)
        return ToolResult(
            output=output,
            title=desc,
            metadata={
                "exit": exit_code,
                "truncated": was_truncated,
                "timed_out": timed_out,
            },
        )
