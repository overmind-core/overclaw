"""Mutable in-memory view of the artifacts preflight is allowed to touch.

Autofix handlers mutate ``state.eval_spec`` / ``state.dataset`` in
memory and return :class:`PatchRecord` entries.  The runner then writes
the updated dicts back to disk via :meth:`persist`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from overmind.core.paths import (
    agent_instrumented_dir,
    agent_setup_spec_dir,
)
from overmind.core.registry import resolve_agent


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class WorkingState:
    """In-memory bundle of everything the autofix handlers may mutate."""

    agent_name: str
    eval_spec: dict[str, Any]
    dataset: list[dict]
    eval_spec_path: Path
    dataset_path: Path
    instrumented_dir: Path
    # Relative paths (relative to ``instrumented_dir``) that handlers want
    # re-instrumented before the next smoke pass.
    reinstrument_requests: set[str] = field(default_factory=set)
    # Dependency package names that handlers added to requirements.txt.
    deps_to_add: set[str] = field(default_factory=set)
    # Absolute path to the registered Overmind entrypoint file.
    entrypoint_path: Path | None = None

    @classmethod
    def load(cls, agent_name: str) -> WorkingState:
        spec_path = agent_setup_spec_dir(agent_name) / "eval_spec.json"
        ds_path = agent_setup_spec_dir(agent_name) / "dataset.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"eval_spec.json not found: {spec_path}")
        if not ds_path.is_file():
            raise FileNotFoundError(f"dataset.json not found: {ds_path}")
        eval_spec = json.loads(spec_path.read_text())
        raw_ds = json.loads(ds_path.read_text())
        if isinstance(raw_ds, dict) and "test_cases" in raw_ds:
            dataset = list(raw_ds["test_cases"])
        elif isinstance(raw_ds, list):
            dataset = list(raw_ds)
        else:
            dataset = []
        try:
            file_path, _fn_name = resolve_agent(agent_name)
            entrypoint_path: Path | None = Path(file_path) if file_path else None
        except SystemExit:
            entrypoint_path = None
        except Exception:
            entrypoint_path = None
        return cls(
            agent_name=agent_name,
            eval_spec=eval_spec,
            dataset=dataset,
            eval_spec_path=spec_path,
            dataset_path=ds_path,
            instrumented_dir=agent_instrumented_dir(agent_name),
            entrypoint_path=entrypoint_path,
        )

    def file_hash(self, path: Path) -> str:
        if not path.is_file():
            return ""
        return _sha256(path.read_bytes())

    def persist(self) -> tuple[bool, bool]:
        """Write any in-memory changes back to disk.

        Returns ``(eval_spec_changed, dataset_changed)``.
        """
        new_spec = json.dumps(self.eval_spec, indent=2, sort_keys=False) + "\n"
        new_ds = json.dumps(self.dataset, indent=2, default=str) + "\n"

        spec_changed = False
        ds_changed = False

        old_spec = self.eval_spec_path.read_text() if self.eval_spec_path.is_file() else ""
        if old_spec != new_spec:
            self.eval_spec_path.write_text(new_spec)
            spec_changed = True

        old_ds = self.dataset_path.read_text() if self.dataset_path.is_file() else ""
        if old_ds != new_ds:
            self.dataset_path.write_text(new_ds)
            ds_changed = True

        return spec_changed, ds_changed
