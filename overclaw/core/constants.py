"""Shared filesystem constants for OverClaw state under the project root."""

from __future__ import annotations

# Directory created by ``overclaw init``; its presence marks the project root.
OVERCLAW_DIR_NAME = ".overclaw"


def overclaw_rel(*segments: str) -> str:
    """Build a POSIX-style path under the state dir for user-facing messages."""
    return "/".join((OVERCLAW_DIR_NAME, *segments))
