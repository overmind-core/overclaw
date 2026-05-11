"""Content hashing used for the preflight ↔ optimize freshness gate.

The optimize CLI compares the hashes recorded in ``preflight.json``
against fresh hashes of the same artifacts.  If anything diverges,
optimize refuses to start with a clear "preflight is stale" message.

Only the **set of env-var names** is hashed for ``.env`` files — never
their values — so secrets stay out of the report.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dotenv import dotenv_values

from overmind.core.paths import (
    agent_env_path,
    agent_instrumented_dir,
    agent_setup_spec_dir,
)
from overmind.core.registry import resolve_agent


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return _sha256_bytes(path.read_bytes())


def _hash_env_keys(path: Path) -> str:
    """Hash the *names* (not values) of env vars in *path*.

    This lets the gate notice when a user adds or removes a credential
    without leaking the credential itself into ``preflight.json``.
    """
    if not path.is_file():
        return ""
    keys = sorted((dotenv_values(path) or {}).keys())
    return _sha256_bytes("\n".join(keys).encode())


def _hash_entrypoint(agent_name: str) -> str:
    """Hash the entrypoint file via the registered agent path.

    Falls back to the empty string when the agent is not registered yet
    (preflight will catch that separately and refuse with a clear error).
    """
    try:
        agent_path, _ = resolve_agent(agent_name)
    except SystemExit:
        return ""
    return _hash_file(Path(agent_path))


def _hash_instrumented_tree(agent_name: str) -> str:
    """Roll up a hash over every file in the instrumented copy.

    Sorted by relative path so the digest is stable across runs.
    """
    root = agent_instrumented_dir(agent_name)
    if not root.is_dir():
        return ""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def compute_hashes(agent_name: str) -> dict[str, str]:
    """Return the full hash map persisted into :class:`PreflightReport`.

    Keys covered:

    - ``entrypoint`` — the registered entrypoint file as it sits on disk.
    - ``instrumented`` — the entire ``.overmind/agents/<name>/instrumented/``
      tree, so any in-place edit triggers a stale gate.
    - ``eval_spec`` / ``dataset`` — the canonical setup_spec artifacts.
    - ``env_keys`` — sorted *names* of env vars in the per-agent ``.env``
      (values are intentionally never hashed).
    """
    spec_dir = agent_setup_spec_dir(agent_name)
    return {
        "entrypoint": _hash_entrypoint(agent_name),
        "instrumented": _hash_instrumented_tree(agent_name),
        "eval_spec": _hash_file(spec_dir / "eval_spec.json"),
        "dataset": _hash_file(spec_dir / "dataset.json"),
        "env_keys": _hash_env_keys(agent_env_path(agent_name)),
    }


def hashes_match(stored: dict[str, str], fresh: dict[str, str]) -> tuple[bool, list[str]]:
    """Return ``(all_match, diff_keys)``."""
    diff = [k for k in fresh if stored.get(k, "") != fresh.get(k, "")]
    return (not diff, diff)
