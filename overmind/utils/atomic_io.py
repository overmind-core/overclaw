"""Atomic filesystem writes.

When a long-running step crashes mid-write — SIGKILL, OOM, kernel panic —
a non-atomic ``Path.write_text`` leaves a half-written file on disk.  The
next iteration then either crashes on JSON-decode or, worse, silently
loads partial state and ships wrong answers.

:func:`atomic_write_text` and :func:`atomic_write_json` write to a
sibling ``.tmp`` file first and then rename it over the target.  On POSIX
filesystems ``Path.replace`` is atomic, so readers see either the old
contents or the new — never a partial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_text", "atomic_write_json"]


def atomic_write_text(path: str | Path, data: str, *, encoding: str = "utf-8") -> Path:
    """Write *data* to *path* atomically.  Creates parent dirs as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(data, encoding=encoding)
    tmp.replace(target)
    return target


def atomic_write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int | None = 2,
    default: Any = str,
    sort_keys: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Serialise *obj* to JSON and write to *path* atomically.

    Defaults to ``indent=2`` and ``default=str`` because almost every call
    site in the codebase wants pretty-printed JSON for state files that
    humans occasionally inspect; pass ``indent=None`` for compact output.
    """
    payload = json.dumps(obj, indent=indent, default=default, sort_keys=sort_keys)
    return atomic_write_text(path, payload, encoding=encoding)
