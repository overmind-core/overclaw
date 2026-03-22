"""Storage layer for OverClaw.

Usage
-----
Call :func:`get_storage` from anywhere::

    from overclaw.storage import get_storage

    storage = get_storage()

    # Write
    storage.save_spec(spec_dict)
    storage.save_dataset(cases)
    storage.save_policy(policy_md)
    storage.save_trace(trace_data, run_name="baseline", idx=0)

    # Read
    spec   = storage.load_spec()
    cases  = storage.load_dataset()
    policy = storage.load_policy()

    # Delete
    storage.delete_spec()
    storage.delete_dataset()
    storage.delete_traces(run_name="baseline")   # or None to delete all
    storage.clear_setup_spec()                   # spec + dataset + policy
    storage.clear_experiments()                  # traces + report + artifacts

    # Identity helpers
    agent_id = storage.get_agent_id()   # None for FsBackend
    storage.set_job_id("job-uuid")      # bind to an optimization job

Backend selection
-----------------
``get_storage()`` automatically picks the best backend:

* **``FsBackend``** — always available; stores artifacts relative to the
  agent file's parent directory (``setup_spec/`` and ``experiments/``).
* **``ApiBackend``** — used when ``OVERMIND_API_URL`` and
  ``OVERMIND_API_TOKEN`` are both set in the environment.

Factory behavior
----------------
``get_storage()`` is intentionally simple: each call constructs and returns a
fresh backend instance.  No thread-local or global cache is used.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from overclaw.client import is_configured
from overclaw.core.paths import load_overclaw_dotenv
from overclaw.storage.api import ApiBackend
from overclaw.storage.base import StorageBackend
from overclaw.storage.fs import FsBackend


@dataclass
class _BoundStorageConfig:
    agent_path: str | None = None
    agent_id: str | None = None
    job_id: str | None = None
    backend: Literal["fs", "api"] | None = None
    client: object | None = None


_BOUND_STORAGE: ContextVar[_BoundStorageConfig | None] = ContextVar(
    "_BOUND_STORAGE",
    default=None,
)


def configure_storage(
    *,
    agent_path: str,
    agent_id: str | None = None,
    job_id: str | None = None,
    backend: Literal["fs", "api"] | None = None,
    client: object | None = None,
) -> None:
    """Bind storage defaults for the current execution context.

    Once configured, callers can use ``get_storage()`` with no arguments.
    """
    _BOUND_STORAGE.set(
        _BoundStorageConfig(
            agent_path=agent_path,
            agent_id=agent_id,
            job_id=job_id,
            backend=backend,
            client=client,
        )
    )


def clear_storage_binding() -> None:
    """Clear previously bound storage defaults in the current context."""
    _BOUND_STORAGE.set(None)


def _backend_kind(backend: str | None) -> str:
    """Resolve the backend hint to a backend kind."""
    if backend in ("fs", "api"):
        return backend

    return "api" if is_configured() else "fs"


def _build(
    agent_path: str,
    *,
    agent_id: str | None,
    job_id: str | None,
    backend: str | None,
    client: object | None,
) -> StorageBackend:
    """Construct a fresh backend.  Never raises — falls back to FsBackend."""
    try:
        kind = _backend_kind(backend)

        if backend == "fs" or kind == "fs":
            return FsBackend(agent_path=agent_path)

        return ApiBackend(
            agent_id=agent_id or "",
            agent_path=agent_path,
            job_id=job_id,
            client=client,
        )
    except Exception:
        return FsBackend(agent_path=agent_path)


def _resolve_storage_identity() -> tuple[str, str | None]:
    """Resolve identity with precedence: configured binding > env."""
    load_overclaw_dotenv()
    bound = _BOUND_STORAGE.get()

    env_agent_path = os.getenv("OVERCLAW_AGENT_PATH", "").strip() or None
    env_agent_id = os.getenv("OVERMIND_AGENT_ID", "").strip() or None

    # Ensure project id is loaded from .env into process env for downstream
    # API helpers that read it lazily (e.g. overclaw.client.get_project_id()).
    os.getenv("OVERMIND_PROJECT_ID", "").strip()

    final_agent_path = ((bound.agent_path if bound else None) or env_agent_path or "").strip()
    final_agent_id = (
        (bound.agent_id if bound else None) or env_agent_id or ""
    ).strip() or None

    if not final_agent_path:
        raise ValueError(
            "get_storage() requires configure_storage(agent_path=...) or OVERCLAW_AGENT_PATH in .overclaw/.env (or process env)"
        )

    return final_agent_path, final_agent_id


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_storage() -> StorageBackend:
    """Return a :class:`StorageBackend` for an agent — always succeeds.

    If anything goes wrong during construction the function silently falls
    back to :class:`FsBackend` so callers are never left without a backend.

    Notes
    -----
    Resolution precedence:
    1) values bound via :func:`configure_storage`
    2) env fallback loaded from ``.overclaw/.env``

    Relevant env vars:
    - ``OVERCLAW_AGENT_PATH`` (fallback for agent path)
    - ``OVERMIND_AGENT_ID`` (fallback for agent id)
    - ``OVERMIND_PROJECT_ID`` (used by API helpers)
    - ``OVERMIND_API_URL`` + ``OVERMIND_API_TOKEN`` (select API backend when set)
    """
    bound = _BOUND_STORAGE.get()
    agent_path, agent_id = _resolve_storage_identity()
    resolved_job_id = bound.job_id if bound else None
    resolved_backend = bound.backend if bound else None
    resolved_client = bound.client if bound else None

    instance = _build(
        agent_path,
        agent_id=agent_id,
        job_id=resolved_job_id,
        backend=resolved_backend,
        client=resolved_client,
    )

    if resolved_job_id:
        instance.set_job_id(resolved_job_id)

    return instance


def get_storage_class() -> type[StorageBackend]:
    """Return the backend class selected by current environment config."""
    bound = _BOUND_STORAGE.get()
    if bound and bound.backend == "api":
        return ApiBackend
    if bound and bound.backend == "fs":
        return FsBackend
    return ApiBackend if is_configured() else FsBackend


__all__ = [
    "ApiBackend",
    "FsBackend",
    "StorageBackend",
    "clear_storage_binding",
    "configure_storage",
    "get_storage",
    "get_storage_class",
]
