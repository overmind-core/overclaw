"""Filesystem storage backend for OverClaw.

All artifacts are stored relative to the agent's parent directory:

    <agent_dir>/
    ├── setup_spec/
    │   ├── eval_spec.json   ← eval spec
    │   ├── dataset.json     ← test-case dataset
    │   └── policies.md      ← policy Markdown
    └── experiments/
        ├── traces/
        │   └── <run_name>/
        │       └── <idx:03d>.json
        ├── results.tsv
        ├── report.md
        ├── best_agent.py
        └── ...              ← other artifacts (working copy, failed candidates)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from overclaw.storage.base import StorageBackend


# ---------------------------------------------------------------------------
# Path helpers (module-level so they can be imported by other modules)
# ---------------------------------------------------------------------------


def spec_path(agent_path: str) -> Path:
    """Return the canonical path to the eval spec for *agent_path*."""
    return Path(agent_path).resolve().parent / "setup_spec" / "eval_spec.json"


def dataset_path(agent_path: str) -> Path:
    """Return the canonical path to the dataset for *agent_path*."""
    return Path(agent_path).resolve().parent / "setup_spec" / "dataset.json"


def policy_path(agent_path: str) -> Path:
    """Return the canonical path to the policy Markdown for *agent_path*."""
    return Path(agent_path).resolve().parent / "setup_spec" / "policies.md"


def experiments_dir(agent_path: str) -> Path:
    """Return the experiments output directory for *agent_path*."""
    return Path(agent_path).resolve().parent / "experiments"


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class FsBackend(StorageBackend):
    """Stores all OverClaw artifacts on the local filesystem.

    Parameters
    ----------
    agent_path:
        Absolute or relative path to the agent Python file.  All artifact
        paths are derived from the parent directory of this file.
    """

    def __init__(self, agent_path: str) -> None:
        self._agent_path = agent_path
        self._exp_dir = experiments_dir(agent_path)

    # ------------------------------------------------------------------
    # Convenience accessors (useful for optimizer code that needs raw paths)
    # ------------------------------------------------------------------

    @property
    def agent_path(self) -> str:
        return self._agent_path

    def get_spec_path(self) -> Path:
        """Return the absolute ``Path`` of the eval spec file."""
        return spec_path(self._agent_path)

    def get_dataset_path(self) -> Path:
        """Return the absolute ``Path`` of the dataset file."""
        return dataset_path(self._agent_path)

    def get_policy_path(self) -> Path:
        """Return the absolute ``Path`` of the policy Markdown file."""
        return policy_path(self._agent_path)

    def get_artifact_path(self, name: str) -> Path:
        """Return the absolute ``Path`` for a named experiment artifact."""
        return self._exp_dir / name

    def get_experiments_dir(self) -> Path:
        """Return the absolute ``Path`` of the experiments directory."""
        return self._exp_dir

    # ------------------------------------------------------------------
    # Eval spec
    # ------------------------------------------------------------------

    def save_spec(self, spec: dict) -> None:
        path = spec_path(self._agent_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

    def load_spec(self) -> dict | None:
        path = spec_path(self._agent_path)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def delete_spec(self) -> None:
        spec_path(self._agent_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def save_dataset(self, cases: list[dict]) -> None:
        path = dataset_path(self._agent_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)

    def load_dataset(self) -> list[dict] | None:
        path = dataset_path(self._agent_path)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("test_cases", [])
        return data

    def delete_dataset(self) -> None:
        dataset_path(self._agent_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def save_policy(self, policy_md: str, policy_data: dict | None = None) -> None:
        path = policy_path(self._agent_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(policy_md, encoding="utf-8")

    def load_policy(self) -> str | None:
        path = policy_path(self._agent_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def delete_policy(self) -> None:
        policy_path(self._agent_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def save_trace(self, trace_data: dict, run_name: str, idx: int) -> None:
        trace_path = self._exp_dir / "traces" / run_name / f"{idx:03d}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, default=str)

    def delete_traces(self, run_name: str | None = None) -> None:
        if run_name:
            target = self._exp_dir / "traces" / run_name
            if target.exists():
                shutil.rmtree(target)
        else:
            target = self._exp_dir / "traces"
            if target.exists():
                shutil.rmtree(target)

    # ------------------------------------------------------------------
    # Generic artifacts
    # ------------------------------------------------------------------

    def save_artifact(self, content: str, name: str) -> None:
        path = self._exp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def load_artifact(self, name: str) -> str | None:
        path = self._exp_dir / name
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def delete_artifact(self, name: str) -> None:
        (self._exp_dir / name).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Results log
    # ------------------------------------------------------------------

    def init_results_log(self, dim_keys: list[str]) -> None:
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        path = self._exp_dir / "results.tsv"
        if not path.exists():
            dim_cols = "\t".join(dim_keys)
            path.write_text(
                f"iteration\tavg_score\t{dim_cols}\tstatus\tdescription\n",
                encoding="utf-8",
            )

    def append_result_row(self, row: dict, dim_keys: list[str]) -> None:
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        line = "\t".join(str(v) for v in row.values()) + "\n"
        with open(self._exp_dir / "results.tsv", "a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def save_report(self, report_md: str, best_code: str | None = None) -> None:
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        (self._exp_dir / "report.md").write_text(report_md, encoding="utf-8")
        if best_code is not None:
            (self._exp_dir / "best_agent.py").write_text(best_code, encoding="utf-8")

    def load_report(self) -> str | None:
        path = self._exp_dir / "report.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Bulk cleanup
    # ------------------------------------------------------------------

    def clear_setup_spec(self) -> None:
        spec_dir = spec_path(self._agent_path).parent
        if spec_dir.exists():
            shutil.rmtree(spec_dir)

    def clear_experiments(self) -> None:
        if self._exp_dir.exists():
            shutil.rmtree(self._exp_dir)
        self._exp_dir.mkdir(parents=True, exist_ok=True)
