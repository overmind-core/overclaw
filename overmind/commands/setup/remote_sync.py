"""Push local setup artifacts (spec, dataset, policy) to the Overmind backend."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from rich.console import Console

from overmind.client import flush_pending_api_updates, get_client, upsert_agent
from overmind.core.paths import agent_setup_spec_dir
from overmind.core.registry import get_agent_id, load_registry, save_agent
from overmind.storage import configure_storage, get_storage
from overmind.utils.policy import default_policy_path

__all__ = ["ensure_remote_agent_id", "sync_setup_artifacts"]


def ensure_remote_agent_id(
    agent_name: str,
    agent_path: str,
    console: Console,
    spec: dict | None = None,
) -> str | None:
    """Ensure a remote Overmind agent exists; return its id when available.

    Returns ``None`` when no client is configured (e.g. ``OVERMIND_API_KEY``
    is unset) or when the backend rejects both the full and minimal spec.
    """
    existing_id = get_agent_id(agent_name)
    if existing_id:
        return existing_id

    client = get_client()
    if not client:
        return None

    console.print("  [dim]No remote id found. Creating agent in Overmind...[/dim]")
    minimal_spec = {
        "agent_description": f"{agent_name} agent",
        "agent_path": agent_path,
        "input_schema": {},
        "output_fields": {},
        "structure_weight": 20,
        "total_points": 100,
    }
    create_spec = spec if isinstance(spec, dict) and spec else minimal_spec
    try:
        result = upsert_agent(
            client,
            agent_path=agent_path,
            spec=create_spec,
            agent_name=agent_name,
        )
        new_id = str(result.id)
        entrypoint = (load_registry().get(agent_name, {}) or {}).get("entrypoint")
        if entrypoint:
            save_agent(agent_name, entrypoint, id=new_id)
        console.print("  [dim]Remote agent created and id stored in agents.toml.[/dim]")
        return new_id
    except Exception as exc:
        if spec:
            with suppress(Exception):
                result = upsert_agent(
                    client,
                    agent_path=agent_path,
                    spec=minimal_spec,
                    agent_name=agent_name,
                )
                new_id = str(result.id)
                entrypoint = (load_registry().get(agent_name, {}) or {}).get("entrypoint")
                if entrypoint:
                    save_agent(agent_name, entrypoint, id=new_id)
                console.print("  [dim]Remote agent created and id stored in agents.toml.[/dim]")
                return new_id
        console.print(f"  [yellow]Warning:[/yellow] Could not create agent in Overmind. [dim]({exc})[/dim]")
        return None


def sync_setup_artifacts(agent_name: str, agent_path: str, console: Console) -> None:
    """Upload local setup artifacts to the Overmind backend when configured.

    Silently skips when ``OVERMIND_API_KEY`` is not set.  The project is
    inferred server-side from the API key.
    """
    client = get_client()
    if not client:
        return

    spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
    dataset_path = agent_setup_spec_dir(agent_name) / "dataset.json"
    policy_path = Path(default_policy_path(agent_name))

    spec: dict | None = None
    if spec_path.exists():
        with suppress(Exception):
            loaded = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                spec = loaded

    agent_id = ensure_remote_agent_id(agent_name, agent_path, console, spec=spec)
    if not agent_id:
        return

    configure_storage(agent_path=agent_path, agent_id=agent_id, agent_name=agent_name)
    try:
        storage = get_storage()
    except Exception:
        return

    synced: list[str] = []

    if spec_path.exists():
        with suppress(Exception):
            loaded = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                spec = loaded
                storage.save_spec(spec)
                synced.append("spec")

    if dataset_path.exists():
        with suppress(Exception):
            loaded = json.loads(dataset_path.read_text(encoding="utf-8"))
            cases = loaded.get("test_cases", []) if isinstance(loaded, dict) else loaded
            if isinstance(cases, list):
                storage.save_dataset(
                    cases,
                    source="seed",
                    metadata={"synced_from": str(dataset_path)},
                )
                synced.append("dataset")

    if policy_path.exists():
        with suppress(Exception):
            policy_md = policy_path.read_text(encoding="utf-8")
            policy_data = spec.get("policy") if isinstance(spec, dict) else None
            storage.save_policy(
                policy_md,
                policy_data if isinstance(policy_data, dict) else None,
            )
            synced.append("policy")

    if synced:
        flush_pending_api_updates(timeout=20.0)
        console.print(f"  [dim]Synced setup artifacts to Overmind ({', '.join(synced)}).[/dim]")
