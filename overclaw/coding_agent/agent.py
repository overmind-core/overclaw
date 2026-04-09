"""Agent loop — the core orchestrator that drives the LLM / tool cycle.

Flow:
  1. Build system prompt + user instruction
  2. Send to LLM with available tools
  3. If LLM returns tool calls -> execute each, feed results back
  4. If LLM returns text with no tool calls -> done
  5. Repeat until done or max_steps reached
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .providers import LiteLLMProvider
from .system_prompt import build_system_prompt
from .tools.base import ToolContext
from .tools.registry import ToolRegistry
from .file_tracker import FileTracker
from .truncate import truncate

logger = logging.getLogger("overclaw.coding_agent")

MAX_STEPS = 50
DOOM_THRESHOLD = 3


@dataclass
class StepRecord:
    """One step in the agent loop."""

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Final result of the agent run."""

    text: str
    steps: list[StepRecord]
    total_usage: dict[str, int]


def run(
    instruction: str,
    model: str,
    cwd: str,
    worktree: str | None = None,
    extra_instructions: list[str] | None = None,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Run the coding agent loop to completion.

    Args:
        instruction: The task description (typically a codegen instruction
            derived from a diagnosis).
        model: LiteLLM model identifier (e.g. "anthropic/claude-sonnet-4-20250514").
        cwd: Working directory — the root of the agent code to modify.
        worktree: Project root (defaults to *cwd*).
        extra_instructions: Additional system-prompt text fragments.
        max_steps: Maximum LLM <-> tool round-trips.

    Returns:
        AgentResult with final text, step history, and usage stats.
    """
    worktree = worktree or cwd
    provider = LiteLLMProvider(model=model)

    tracker = FileTracker()
    registry = _build_registry()
    ctx = ToolContext(
        session_id="codegen",
        worktree=worktree,
        cwd=cwd,
        file_tracker=tracker,
    )

    system = build_system_prompt(
        cwd=cwd,
        worktree=worktree,
        model_id=model,
    )
    if extra_instructions:
        system += "\n" + "\n".join(extra_instructions)

    schemas = registry.openai_schemas(model)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]

    steps: list[StepRecord] = []
    total_usage: dict[str, int] = {"input": 0, "output": 0}
    recent_calls: list[tuple[str, str]] = []

    for step_num in range(max_steps):
        logger.debug("Coding agent step %d/%d", step_num + 1, max_steps)

        resp = provider.chat(messages=messages, tools=schemas or None)
        for k in ("input", "output"):
            total_usage[k] = total_usage.get(k, 0) + resp.usage.get(k, 0)

        record = StepRecord(role="assistant", content=resp.text)

        if not resp.tool_calls:
            steps.append(record)
            logger.debug("Coding agent finished (no tool calls)")
            return AgentResult(text=resp.text, steps=steps, total_usage=total_usage)

        # Doom-loop detection
        for tc in resp.tool_calls:
            sig = (tc.name, json.dumps(tc.arguments, sort_keys=True))
            recent_calls.append(sig)

        if len(recent_calls) >= DOOM_THRESHOLD:
            tail = recent_calls[-DOOM_THRESHOLD:]
            if all(t == tail[0] for t in tail):
                logger.warning(
                    "Doom loop: %s called %d times with same args",
                    tail[0][0],
                    DOOM_THRESHOLD,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You have called {tail[0][0]} with the same arguments "
                            f"{DOOM_THRESHOLD} times. This appears to be a loop. "
                            "Try a different approach."
                        ),
                    }
                )
                recent_calls.clear()
                continue

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": resp.text or None,
        }
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in resp.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in resp.tool_calls:
            logger.debug("  Tool: %s", tc.name)
            result = registry.execute(tc.name, tc.arguments, ctx)
            output, _was_truncated = truncate(result.output)
            record.tool_calls.append({"name": tc.name, "args": tc.arguments})
            record.tool_results.append({"name": tc.name, "output": output[:500]})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

        steps.append(record)

    logger.warning("Coding agent hit max_steps (%d)", max_steps)
    return AgentResult(
        text="Agent reached maximum steps without completing.",
        steps=steps,
        total_usage=total_usage,
    )


def _build_registry() -> ToolRegistry:
    from .tools.read import ReadTool
    from .tools.edit import EditTool
    from .tools.write import WriteTool
    from .tools.grep import GrepTool
    from .tools.glob_tool import GlobTool
    from .tools.bash import BashTool
    from .tools.apply_patch import ApplyPatchTool

    reg = ToolRegistry()
    reg.register(ReadTool())
    reg.register(EditTool())
    reg.register(WriteTool())
    reg.register(GrepTool())
    reg.register(GlobTool())
    reg.register(BashTool())
    reg.register(ApplyPatchTool())
    return reg
