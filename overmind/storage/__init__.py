"""Storage layer for Overmind.

Overmind stores all setup artifacts — eval specs, datasets, and policies —
in the Overmind backend.  Local filesystem persistence is no longer
supported; every save / load goes through the generated
``overmind.openapi_client`` SDK.

Usage
-----
::

    from overmind.storage import configure_storage, get_storage

    configure_storage(agent_path="agents/agent1/sample_agent.py")
    storage = get_storage()

    storage.save_spec(spec_dict)
    storage.save_dataset(cases)
    storage.save_policy(policy_md)

    spec   = storage.load_spec()
    cases  = storage.load_dataset()
    policy = storage.load_policy()

    storage.delete_spec()
    storage.delete_dataset()
    storage.clear_setup_spec()           # spec + dataset + policy

Configuration
-------------
``get_storage()`` requires one environment variable:

* ``OVERMIND_API_KEY`` — bearer token (required).

If ``OVERMIND_API_KEY`` is missing, :func:`get_storage` raises
:class:`StorageNotConfiguredError`.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass

from overmind.client import is_configured
from overmind.core.paths import load_overmind_dotenv
from overmind.storage.api import ApiBackend
from overmind.storage.base import StorageBackend


class StorageNotConfiguredError(RuntimeError):
    """Raised when ``OVERMIND_API_KEY`` is not set."""


@dataclass
class _BoundStorageConfig:
    agent_path: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    job_id: str | None = None
    client: object | None = None


_BOUND_STORAGE: ContextVar[_BoundStorageConfig | None] = ContextVar(
    "_BOUND_STORAGE",
    default=None,
)


def configure_storage(
    *,
    agent_path: str,
    agent_id: str | None = None,
    agent_name: str | None = None,
    job_id: str | None = None,
    client: object | None = None,
) -> None:
    """Bind storage defaults for the current execution context.

    After this call, ``get_storage()`` (no args) returns a backend bound to
    the supplied identity.
    """
    _BOUND_STORAGE.set(
        _BoundStorageConfig(
            agent_path=agent_path,
            agent_id=agent_id,
            agent_name=agent_name,
            job_id=job_id,
            client=client,
        )
    )


def _resolve_storage_identity() -> tuple[str, str | None]:
    """Resolve identity with precedence: configured binding > env."""
    load_overmind_dotenv()
    bound = _BOUND_STORAGE.get()

    env_agent_path = os.getenv("OVERMIND_AGENT_PATH", "").strip() or None
    env_agent_id = os.getenv("OVERMIND_AGENT_ID", "").strip() or None

    final_agent_path = ((bound.agent_path if bound else None) or env_agent_path or "").strip()
    final_agent_id = ((bound.agent_id if bound else None) or env_agent_id or "").strip() or None

    if not final_agent_path:
        raise ValueError(
            "get_storage() requires configure_storage(agent_path=...) "
            "or OVERMIND_AGENT_PATH in .overmind/.env (or process env)"
        )

    return final_agent_path, final_agent_id


def get_storage() -> StorageBackend:
    """Return a configured :class:`StorageBackend`.

    Raises
    ------
    StorageNotConfiguredError
        If ``OVERMIND_API_KEY`` is not set.
    ValueError
        If no agent path can be resolved (neither bound nor in env).
    """
    load_overmind_dotenv()
    if not is_configured():
        raise StorageNotConfiguredError(
            "Overmind API is not configured. Set OVERMIND_API_KEY "
            "in .overmind/.env or the process environment before calling "
            "get_storage()."
        )

    bound = _BOUND_STORAGE.get()
    agent_path, agent_id = _resolve_storage_identity()
    agent_name = bound.agent_name if bound else None
    job_id = bound.job_id if bound else None
    client = bound.client if bound else None

    instance = ApiBackend(
        agent_id=agent_id or "",
        agent_path=agent_path,
        agent_name=agent_name,
        job_id=job_id,
        client=client,
    )
    if job_id:
        instance.set_job_id(job_id)
    return instance


__all__ = [
    "ApiBackend",
    "StorageBackend",
    "StorageNotConfiguredError",
    "configure_storage",
    "get_storage",
]
