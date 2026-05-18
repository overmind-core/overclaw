"""Agent source instrumentation helpers shared by register and setup.

Historically this module also collected provider credentials and
``os.environ`` defaults into a per-agent ``.env`` file at
``<state>/agents/<name>/.env``.  That file was loaded with
``override=True``, so any stale placeholder in it silently won over the real
value in the project ``.overmind/.env``.  Per-agent ``.env`` is now gone —
credentials live exclusively in the project file written by
``overmind init`` — and only the source-instrumentation helper remains here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from overmind.core.paths import agent_instrumented_dir
from overmind.core.registry import project_root_from_agent_file
from overmind.utils.display import rel

# Directories we never copy into the instrumented agent tree — virtualenvs,
# node_modules, VCS metadata, and the Overmind state directory itself.
_SKIP_DIRS = {
    ".venv",
    "venv",
    "node_modules",
    ".overmind_runners",
    "__pycache__",
    ".git",
    ".overmind",
}


def instrument_agent_files(agent_path: str, agent_name: str, console: Console) -> tuple[str, Path]:
    """Copy the agent's source tree to ``.overmind/agents/<name>/instrumented/``.

    The original files are never modified.  This is a **plain copy** — no
    ``@observe()`` decorators or overmind imports are added here.
    Instrumentation (imports + decorators) is applied later by the
    optimizer when it actually needs tracing.

    The copy boundary is the **project root** (the directory containing
    ``.overmind/``), not just the entry file's parent.  This ensures that
    local imports across sibling packages are available in the copy.

    Returns ``(instrumented_entry_path, instrumented_root_dir)``.
    """
    p = Path(agent_path).resolve()
    dest_dir = agent_instrumented_dir(agent_name)
    if not p.exists():
        return agent_path, dest_dir

    pr = project_root_from_agent_file(agent_path)
    copy_root = pr if pr is not None else p.parent
    entry_relpath = p.relative_to(copy_root)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for src_file in copy_root.rglob("*"):
        if any(part in _SKIP_DIRS for part in src_file.parts):
            continue
        if src_file.is_dir():
            continue
        rel_path = src_file.relative_to(copy_root)
        dst_file = dest_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        file_count += 1

    instrumented_entry = str(dest_dir / entry_relpath)
    console.print(
        f"  [bold green]\u2713[/bold green] Copied agent source ({file_count} file(s)) to [dim]{rel(dest_dir)}[/dim]"
    )
    return instrumented_entry, dest_dir
