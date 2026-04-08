"""System prompt construction for the coding agent used during optimization.

Tailored for the overclaw optimize workflow: the agent receives a diagnosis
of what to change and applies targeted code edits to an agent codebase.
"""

from __future__ import annotations

import platform
from datetime import datetime

BASE_PROMPT = """\
You are an expert coding agent that applies targeted improvements to AI agent code.

You are given a diagnosis describing specific changes to make. Your job is to
read the relevant source files, understand the current implementation, and apply
the recommended changes precisely.

# Rules
- Be surgical: only change what the diagnosis asks for.
- Read files before editing — the edit tool enforces this.
- Use grep/glob to locate code when you are unsure which file contains it.
- Preserve existing code style, imports, and conventions.
- Do NOT add comments explaining your changes.
- Do NOT rename the entry function or change its signature unless explicitly told to.
- After editing, re-read the file to verify correctness if the change was complex.
- Prefer the edit tool (find-and-replace) over write (full overwrite) for existing files.

# Tool usage
- Call multiple tools in parallel when the calls are independent.
- Prefer dedicated tools (read, edit, grep, glob) over shell equivalents.
- When using edit, provide enough surrounding context in oldString to ensure a unique match."""


def build_system_prompt(
    cwd: str,
    worktree: str,
    model_id: str = "",
) -> str:
    """Build the full system prompt for the coding agent."""
    parts = [BASE_PROMPT]

    env_block = "\n".join(
        [
            "",
            f"Model: {model_id}" if model_id else "",
            "<env>",
            f"  Working directory: {cwd}",
            f"  Platform: {platform.system().lower()}",
            f"  Date: {datetime.now().strftime('%A %b %d, %Y')}",
            "</env>",
        ]
    )
    parts.append(env_block)

    return "\n".join(parts)
