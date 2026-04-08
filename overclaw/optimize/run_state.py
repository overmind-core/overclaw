"""Cross-run persistent state for an agent's optimization history.

Stores accumulated knowledge — failure clusters, regression cases,
change history — so that successive ``overclaw optimize`` invocations
build on prior runs rather than starting from scratch.

Stored at ``<overclaw_dir>/agents/<name>/run_state.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from overclaw.optimize.failure_registry import FailureRegistry

_log = logging.getLogger("overclaw.optimize.run_state")


@dataclass
class RunSummary:
    """Compact record of one optimization run."""

    run_id: int
    started_at: float
    finished_at: float
    baseline_score: float
    final_score: float
    iterations_completed: int
    accepted_changes: int
    rejected_changes: int

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "baseline_score": self.baseline_score,
            "final_score": self.final_score,
            "iterations_completed": self.iterations_completed,
            "accepted_changes": self.accepted_changes,
            "rejected_changes": self.rejected_changes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunSummary:
        return cls(
            run_id=d["run_id"],
            started_at=d.get("started_at", 0),
            finished_at=d.get("finished_at", 0),
            baseline_score=d.get("baseline_score", 0),
            final_score=d.get("final_score", 0),
            iterations_completed=d.get("iterations_completed", 0),
            accepted_changes=d.get("accepted_changes", 0),
            rejected_changes=d.get("rejected_changes", 0),
        )


@dataclass
class RegressionCase:
    """A case that was failing, got fixed, and must stay fixed."""

    case_input: dict
    expected_output: dict
    min_score: float
    fixed_in_run: int
    fixed_in_iteration: int
    cluster_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_input": self.case_input,
            "expected_output": self.expected_output,
            "min_score": self.min_score,
            "fixed_in_run": self.fixed_in_run,
            "fixed_in_iteration": self.fixed_in_iteration,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegressionCase:
        return cls(
            case_input=d.get("case_input", {}),
            expected_output=d.get("expected_output", {}),
            min_score=d.get("min_score", 60.0),
            fixed_in_run=d.get("fixed_in_run", 0),
            fixed_in_iteration=d.get("fixed_in_iteration", 0),
            cluster_id=d.get("cluster_id"),
        )


# Cap persisted history to avoid unbounded growth
_MAX_FAILED_ATTEMPTS = 50
_MAX_SUCCESSFUL_CHANGES = 50
_MAX_REGRESSION_CASES = 100


class RunState:
    """Accumulated cross-run knowledge for one agent."""

    def __init__(self, path: Path, agent_name: str) -> None:
        self.path = path
        self.agent_name = agent_name

        self.run_history: list[RunSummary] = []
        self.failure_registry = FailureRegistry()
        self.cumulative_failed_attempts: list[dict] = []
        self.cumulative_successful_changes: list[dict] = []
        self.regression_cases: list[RegressionCase] = []
        self.component_failure_weights: dict[str, float] = {}
        self._current_run_id: int = 0

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "agent_name": self.agent_name,
            "run_history": [r.to_dict() for r in self.run_history],
            "failure_registry": self.failure_registry.to_dict(),
            "cumulative_failed_attempts": self.cumulative_failed_attempts[
                -_MAX_FAILED_ATTEMPTS:
            ],
            "cumulative_successful_changes": self.cumulative_successful_changes[
                -_MAX_SUCCESSFUL_CHANGES:
            ],
            "regression_cases": [
                rc.to_dict() for rc in self.regression_cases[-_MAX_REGRESSION_CASES:]
            ],
            "component_failure_weights": self.component_failure_weights,
        }

    def _load_from_dict(self, d: dict) -> None:
        self.run_history = [RunSummary.from_dict(r) for r in d.get("run_history", [])]
        reg_data = d.get("failure_registry")
        if reg_data:
            self.failure_registry = FailureRegistry.from_dict(reg_data)
        self.cumulative_failed_attempts = d.get("cumulative_failed_attempts", [])
        self.cumulative_successful_changes = d.get("cumulative_successful_changes", [])
        self.regression_cases = [
            RegressionCase.from_dict(rc) for rc in d.get("regression_cases", [])
        ]
        self.component_failure_weights = d.get("component_failure_weights", {})

    # -- Load / Save --

    @classmethod
    def load(cls, path: Path, agent_name: str) -> RunState:
        state = cls(path, agent_name)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state._load_from_dict(data)
                _log.info(
                    "Loaded run state: %d prior run(s), %d regression case(s), "
                    "%d failure cluster(s)",
                    len(state.run_history),
                    len(state.regression_cases),
                    len(state.failure_registry.clusters),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                _log.warning("Corrupt run_state.json, starting fresh: %s", exc)
        return state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(self.path)
        _log.info("Saved run state to %s", self.path)

    # -- Run lifecycle --

    def begin_run(self) -> int:
        self._current_run_id = len(self.run_history) + 1
        return self._current_run_id

    def end_run(self, summary: RunSummary) -> None:
        self.run_history.append(summary)
        self.component_failure_weights = (
            self.failure_registry.compute_component_weights()
        )

    # -- Seed session state from accumulated history --

    def seed_failed_attempts(self, recent_n: int = 15) -> list[dict]:
        """Return the most recent cross-run failed attempts for seeding."""
        return list(self.cumulative_failed_attempts[-recent_n:])

    def seed_successful_changes(self, recent_n: int = 15) -> list[dict]:
        """Return the most recent cross-run successful changes for seeding."""
        return list(self.cumulative_successful_changes[-recent_n:])

    # -- Accumulate from current run --

    def accumulate_failed(self, attempts: list[dict]) -> None:
        self.cumulative_failed_attempts.extend(attempts)
        self.cumulative_failed_attempts = self.cumulative_failed_attempts[
            -_MAX_FAILED_ATTEMPTS:
        ]

    def accumulate_successful(self, changes: list[dict]) -> None:
        self.cumulative_successful_changes.extend(changes)
        self.cumulative_successful_changes = self.cumulative_successful_changes[
            -_MAX_SUCCESSFUL_CHANGES:
        ]

    # -- Regression suite --

    def add_regression_case(
        self,
        case_input: dict,
        expected_output: dict,
        min_score: float,
        run_id: int,
        iteration: int,
        cluster_id: str | None = None,
    ) -> None:
        input_key = json.dumps(case_input, sort_keys=True, default=str)
        for existing in self.regression_cases:
            if (
                json.dumps(existing.case_input, sort_keys=True, default=str)
                == input_key
            ):
                existing.min_score = max(existing.min_score, min_score)
                return

        self.regression_cases.append(
            RegressionCase(
                case_input=case_input,
                expected_output=expected_output,
                min_score=min_score,
                fixed_in_run=run_id,
                fixed_in_iteration=iteration,
                cluster_id=cluster_id,
            )
        )
        if len(self.regression_cases) > _MAX_REGRESSION_CASES:
            self.regression_cases = self.regression_cases[-_MAX_REGRESSION_CASES:]

    @property
    def has_prior_runs(self) -> bool:
        return len(self.run_history) > 0

    @property
    def total_prior_iterations(self) -> int:
        return sum(r.iterations_completed for r in self.run_history)

    @property
    def best_prior_score(self) -> float:
        if not self.run_history:
            return 0.0
        return max(r.final_score for r in self.run_history)
