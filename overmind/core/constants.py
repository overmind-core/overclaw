"""Shared constants for Overmind: filesystem layout and backend defaults."""

from __future__ import annotations

# Directory created by ``overmind init``; its presence marks the project root.
OVERMIND_DIR_NAME = ".overmind"

# Default Overmind Cloud backend URL.
#
# Used as a fallback when ``OVERMIND_API_URL`` is not set in the environment,
# so that both the tracing exporter (``overmind.tracing``) and the control-plane
# client (``overmind.client``) behave consistently for cloud users who only
# configure ``OVERMIND_API_KEY``. Self-hosted deployments override this by
# setting ``OVERMIND_API_URL`` explicitly.
DEFAULT_BASE_URL = "https://api.overmindlab.ai"


def overmind_rel(*segments: str) -> str:
    """Build a POSIX-style path under the state dir for user-facing messages."""
    return "/".join((OVERMIND_DIR_NAME, *segments))
