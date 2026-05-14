"""Shared constants for Overmind: filesystem layout and backend defaults."""

from __future__ import annotations

# Directory created by ``overmind init``; its presence marks the project root.
OVERMIND_DIR_NAME = ".overmind"

# Default Overmind Cloud backend URL.
#
# Used by both the tracing exporter (``overmind.tracing``) and the
# control-plane client (``overmind.client``).  ``OVERMIND_API_KEY`` is the
# only credential needed; the backend URL is always this constant.
DEFAULT_BASE_URL = "https://api.overmindlab.ai"


def overmind_rel(*segments: str) -> str:
    """Build a POSIX-style path under the state dir for user-facing messages."""
    return "/".join((OVERMIND_DIR_NAME, *segments))
