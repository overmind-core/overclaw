"""Dataclasses + on-disk layout for preflight runs.

The convergence loop in :mod:`overmind.preflight.runner` produces a
:class:`PreflightReport` that gets serialised to
``.overmind/agents/<name>/preflight/preflight.json``.  The optimize CLI
reads that file before starting and refuses to run if it is missing,
non-green, or its content hashes don't match the current artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from overmind.core.paths import agent_overmind_dir

# ---------------------------------------------------------------------------
# Status taxonomy
# ---------------------------------------------------------------------------

# Pipeline runs and scorer returns a finite number; safe to optimize.
STATUS_GREEN = "green"
# Pipeline healthy but baseline score is 0 / very low — that's optimize's job.
STATUS_GREEN_QUALITY = "green_with_quality_notes"
# At least one missing credential blocks execution; user must supply it.
STATUS_BLOCKED_SECRETS = "blocked_secrets"
# Autonomous patches couldn't reach a runnable state inside the iter budget.
STATUS_BLOCKED_NO_CONVERGENCE = "blocked_no_convergence"

GREEN_STATUSES = frozenset({STATUS_GREEN, STATUS_GREEN_QUALITY})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class IssueRecord:
    """A single failure surfaced by the smoke run.

    *kind* maps to the autofix taxonomy (see classifier.py).  *target*
    identifies the artifact the fix would touch
    (``"eval_spec" | "dataset" | "instrumented" | "env"``).
    """

    kind: str
    severity: str  # "block" | "fix" | "quality"
    target: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchRecord:
    """A single autonomous mutation applied during the loop.

    ``before_hash`` / ``after_hash`` are SHA-256 hexdigests of the
    affected file's bytes, so reviewers can see exactly which files
    moved and by how much without having to diff snapshots.
    """

    iteration: int
    kind: str
    file: str
    before_hash: str
    after_hash: str
    reason: str
    diff_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    """JSON envelope describing the outcome of a preflight run."""

    status: str
    agent_name: str
    iterations: int = 0
    baseline_score: float | None = None
    span_count: int = 0
    cases_run: int = 0
    cases_succeeded: int = 0
    cases_failed: int = 0
    patches_applied: list[dict[str, Any]] = field(default_factory=list)
    issues_remaining: list[dict[str, Any]] = field(default_factory=list)
    missing_secrets: list[str] = field(default_factory=list)
    secrets_env_path: str = ""
    snapshots_dir: str = ""
    log_path: str = ""
    timestamp: float = field(default_factory=time.time)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_green(self) -> bool:
        return self.status in GREEN_STATUSES

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------


def preflight_dir(agent_name: str) -> Path:
    """Per-agent preflight workspace: ``.overmind/agents/<name>/preflight/``."""
    return agent_overmind_dir(agent_name) / "preflight"


def preflight_report_path(agent_name: str) -> Path:
    """Where :class:`PreflightReport` is persisted."""
    return preflight_dir(agent_name) / "preflight.json"


def preflight_log_path(agent_name: str) -> Path:
    return preflight_dir(agent_name) / "preflight.log"


def preflight_snapshots_dir(agent_name: str) -> Path:
    return preflight_dir(agent_name) / "snapshots"


# ---------------------------------------------------------------------------
# Read helpers (used by optimize gate + skill `status` command)
# ---------------------------------------------------------------------------


def load_report(agent_name: str) -> PreflightReport | None:
    """Return the persisted preflight report, or ``None`` if absent / invalid."""
    path = preflight_report_path(agent_name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    known = set(PreflightReport.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    try:
        return PreflightReport(**filtered)
    except TypeError:
        return None


def is_preflight_green(agent_name: str) -> bool:
    """Convenience used by the optimize gate."""
    rep = load_report(agent_name)
    return bool(rep and rep.is_green())
