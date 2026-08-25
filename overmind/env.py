"""Load project ``.env`` for the CLI without overriding explicit shell exports."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(cwd: str | Path | None = None) -> None:
    """Populate ``os.environ`` from dotenv files; existing vars always win."""
    env_file = os.environ.get("OVERMIND_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)

    load_dotenv(Path.cwd() / ".env", override=False)

    for root in _extra_roots(cwd):
        load_dotenv(root / ".env", override=False)


def _extra_roots(cwd: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    env_cwd = os.environ.get("OVERMIND_CWD")
    if env_cwd:
        roots.append(Path(env_cwd))
    if cwd:
        roots.append(Path(cwd))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in roots:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out
