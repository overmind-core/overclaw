"""Coding agent module — agentic code editing for the optimization loop.

Public API:
    apply_code_changes(working_dir, instruction, model, ...) -> CodingAgentResult
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("overclaw.coding_agent")


@dataclass
class CodingAgentResult:
    """Result of running the coding agent against a working directory."""

    file_updates: dict[str, str]
    text: str
    steps_taken: int
    usage: dict[str, int] = field(default_factory=dict)


def apply_code_changes(
    agent_files: dict[str, str],
    instruction: str,
    model: str,
    *,
    entry_file: str | None = None,
    extra_instructions: list[str] | None = None,
    max_steps: int = 50,
) -> CodingAgentResult:
    """Run the coding agent to apply changes to a set of agent files.

    Creates an isolated temporary directory, writes the agent files there,
    runs the agentic tool loop, then diffs the results against the originals.

    Args:
        agent_files: ``{relative_path: source_code}`` — the current agent files.
        instruction: Task description (typically derived from a diagnosis).
        model: LiteLLM model identifier.
        entry_file: Relative path to the entry file (for context; defaults to
            first key in *agent_files*).
        extra_instructions: Additional system-prompt fragments.
        max_steps: Maximum LLM <-> tool round-trips.

    Returns:
        CodingAgentResult with the dict of modified files (relative paths
        mapped to new source code), agent text output, step count, and usage.
    """
    from .agent import run as _run_agent

    tmp_dir = tempfile.mkdtemp(prefix="overclaw_codegen_")
    try:
        # Write agent files to temp directory
        for rel_path, source in agent_files.items():
            dest = Path(tmp_dir) / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source, encoding="utf-8")

        # Run the coding agent
        result = _run_agent(
            instruction=instruction,
            model=model,
            cwd=tmp_dir,
            worktree=tmp_dir,
            extra_instructions=extra_instructions,
            max_steps=max_steps,
        )

        # Diff: read back all files and compare against originals
        file_updates: dict[str, str] = {}

        for rel_path, original_source in agent_files.items():
            file_path = Path(tmp_dir) / rel_path
            if not file_path.exists():
                continue
            new_source = file_path.read_text(encoding="utf-8")
            if new_source.rstrip() != original_source.rstrip():
                file_updates[rel_path] = new_source

        # Check for new files the agent may have created
        for root, _dirs, files in os.walk(tmp_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                full = Path(root) / fname
                rel = str(full.relative_to(tmp_dir))
                if rel not in agent_files:
                    file_updates[rel] = full.read_text(encoding="utf-8")

        return CodingAgentResult(
            file_updates=file_updates,
            text=result.text,
            steps_taken=len(result.steps),
            usage=result.total_usage,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
