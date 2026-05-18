"""Ignore-file predicate used when bundling agent source for the analyzer.

The setup and optimize flows both walk the agent's project tree to build a
code "bundle" — a compact representation of the agent's import closure that
the analyzer model reviews.  We want to exclude the usual noise (virtual
envs, caches, build artefacts, instrumented copies) to keep the bundle
small and focused on real agent logic.

This module exposes :func:`build_ignore_predicate`, which reads the
project's ``.gitignore`` and ``.overmindignore`` (if present) plus a
baked-in skip list and returns a callable ``(rel_path: str) -> bool`` that
downstream walkers use to decide whether to include a file.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

logger = logging.getLogger(__name__)

# A callable that decides whether a *relative* path (relative to the
# project root) should be skipped when walking the agent's source tree.
# Used by the bundle factory, agent-env command, and the agent analyzer
# so they all share a single, self-documenting type signature.
IgnorePredicate: TypeAlias = Callable[[str], bool]

# Directories we always skip.  Conservative — purely build / cache paths.
_ALWAYS_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".overmind",
    ".overmind_runners",
    ".overmind_test",
    "dist",
    "build",
    ".eggs",
    ".ipynb_checkpoints",
}

# File-name patterns we always skip.
_ALWAYS_SKIP_GLOBS = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.class",
    "*.DS_Store",
    "*.log",
    "*.lock",
    "*.pkl",
    "*.sqlite",
    "*.db",
)


def _parse_ignore_file(path: Path) -> list[str]:
    """Return non-comment, non-blank lines from an ignore file."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        pat = pat.rstrip("/")
        if not pat:
            continue
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if fnmatch.fnmatch(rel_path, pat.lstrip("/")):
            return True
        if fnmatch.fnmatch(rel_path, f"*/{pat}"):
            return True
        if fnmatch.fnmatch(rel_path, f"{pat}/*"):
            return True
    return False


def build_ignore_predicate(root: Path) -> IgnorePredicate:
    """Return a predicate that decides whether a *relative* path is ignored.

    The predicate is called with a path string relative to *root*.  It
    returns ``True`` if the path should be skipped.
    """
    root = Path(root).resolve()
    gitignore = _parse_ignore_file(root / ".gitignore")
    overmind_ignore = _parse_ignore_file(root / ".overmindignore")
    extra = gitignore + overmind_ignore

    def predicate(rel_path: str) -> bool:
        norm = rel_path.replace("\\", "/").lstrip("./")
        parts = [p for p in norm.split("/") if p]

        if any(p in _ALWAYS_SKIP_DIRS for p in parts):
            return True
        if _matches_any(norm, _ALWAYS_SKIP_GLOBS):
            return True
        if _matches_any(parts[-1], _ALWAYS_SKIP_GLOBS) if parts else False:
            return True
        if extra and _matches_any(norm, extra):
            return True
        return False

    return predicate
