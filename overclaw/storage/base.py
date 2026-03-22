"""Abstract storage backend for OverClaw artifacts.

Every on-disk operation the CLI performs — saving eval specs, datasets, policy
files, traces, optimizer artifacts, and reports — goes through a ``StorageBackend``.
Two concrete implementations ship out of the box:

* ``FsBackend``  — stores everything on the local filesystem (default).
* ``ApiBackend`` — stores everything in the Overmind API (requires credentials).

Create one via the ``get_storage()`` factory in ``overclaw.storage``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Common interface for reading, writing, and deleting OverClaw artifacts.

    A backend is scoped to a single **agent** (identified by its file path)
    and, optionally, a single **optimization job**.  All path / identifier
    resolution happens inside the backend so callers only deal with logical
    names.

    Artifacts are grouped into five categories:

    1. **Eval spec** — JSON dict describing the agent's evaluation criteria.
    2. **Dataset** — list of test-case dicts.
    3. **Policy** — Markdown string (+ optional structured dict) encoding
       domain rules and output constraints.
    4. **Traces** — individual ``Trace.to_dict()`` payloads produced during
       an optimization run.
    5. **Experiment artifacts** — miscellaneous files produced by the
       optimizer: working agent copy, best agent, results log, final report,
       failed candidate snapshots, etc.
    """

    # ------------------------------------------------------------------
    # Eval spec
    # ------------------------------------------------------------------

    @abstractmethod
    def save_spec(self, spec: dict) -> None:
        """Persist the evaluation spec."""

    @abstractmethod
    def load_spec(self) -> dict | None:
        """Return the evaluation spec, or ``None`` if not found."""

    @abstractmethod
    def delete_spec(self) -> None:
        """Delete the evaluation spec (silently ignores missing items)."""

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    @abstractmethod
    def save_dataset(self, cases: list[dict]) -> None:
        """Persist the test-case dataset."""

    @abstractmethod
    def load_dataset(self) -> list[dict] | None:
        """Return the dataset, or ``None`` if not found."""

    @abstractmethod
    def delete_dataset(self) -> None:
        """Delete the dataset (silently ignores missing items)."""

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    @abstractmethod
    def save_policy(self, policy_md: str, policy_data: dict | None = None) -> None:
        """Persist the policy.

        Parameters
        ----------
        policy_md:
            Full Markdown text of the policy document.
        policy_data:
            Optional structured representation (used by the optimizer
            pipeline).  Backends that only support text may ignore this.
        """

    @abstractmethod
    def load_policy(self) -> str | None:
        """Return the policy Markdown, or ``None`` if not found."""

    @abstractmethod
    def delete_policy(self) -> None:
        """Delete the policy (silently ignores missing items)."""

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    @abstractmethod
    def save_trace(self, trace_data: dict, run_name: str, idx: int) -> None:
        """Persist one trace record produced during an optimization run.

        Parameters
        ----------
        trace_data:
            Output of ``Trace.to_dict()``.
        run_name:
            Logical name for the evaluation run (e.g. ``"baseline"`` or
            ``"iter_001_c0"``).  Used to group traces.
        idx:
            Zero-based index of the test case within the run.
        """

    @abstractmethod
    def delete_traces(self, run_name: str | None = None) -> None:
        """Delete traces.

        If *run_name* is given, only delete traces for that run; otherwise
        delete all traces for this agent/job.
        """

    # ------------------------------------------------------------------
    # Experiment artifacts
    # ------------------------------------------------------------------

    @abstractmethod
    def save_artifact(self, content: str, name: str) -> None:
        """Write a named text artifact.

        The *name* is a relative path within the experiment output area,
        e.g. ``"best_agent.py"``, ``"report.md"``,
        ``"failed_iter_001_c0.py"``.  Sub-directories in *name* are
        created automatically by backends that support them.
        """

    @abstractmethod
    def load_artifact(self, name: str) -> str | None:
        """Read a named artifact, or ``None`` if not found."""

    @abstractmethod
    def delete_artifact(self, name: str) -> None:
        """Delete a named artifact (silently ignores missing items)."""

    # ------------------------------------------------------------------
    # Results log
    # ------------------------------------------------------------------

    @abstractmethod
    def init_results_log(self, dim_keys: list[str]) -> None:
        """Create (or reset) the results log with the appropriate header.

        Parameters
        ----------
        dim_keys:
            Ordered list of dimension score column names to include
            after the ``avg_score`` column.
        """

    @abstractmethod
    def append_result_row(self, row: dict, dim_keys: list[str]) -> None:
        """Append one iteration row to the results log.

        Parameters
        ----------
        row:
            Ordered dict with keys: ``iteration``, ``avg_score``,
            one entry per *dim_keys*, ``status``, ``description``.
        dim_keys:
            Same list passed to ``init_results_log``.
        """

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @abstractmethod
    def save_report(self, report_md: str, best_code: str | None = None) -> None:
        """Persist the final optimization report.

        Parameters
        ----------
        report_md:
            Markdown text of the optimization report.
        best_code:
            Source code of the best agent found (optional).  When provided,
            backends should persist it alongside the report.
        """

    @abstractmethod
    def load_report(self) -> str | None:
        """Return the optimization report Markdown, or ``None`` if not found."""

    # ------------------------------------------------------------------
    # Job / agent identity helpers
    # ------------------------------------------------------------------

    def get_agent_id(self) -> str | None:
        """Return the remote agent UUID, or ``None`` for local-only backends."""
        return None

    def set_job_id(self, job_id: str) -> None:  # noqa: B027
        """Associate this backend with a specific optimization job.

        Called by the optimizer once the job record has been created so that
        trace and report writes can be linked to the job.  The default
        implementation is a no-op for backends that don't support jobs.
        """

    # ------------------------------------------------------------------
    # Bulk cleanup
    # ------------------------------------------------------------------

    def clear_setup_spec(self) -> None:
        """Delete the eval spec, dataset, and policy in one call.

        Default implementation calls the individual ``delete_*`` methods.
        Backends may override for a more efficient atomic operation.
        """
        self.delete_spec()
        self.delete_dataset()
        self.delete_policy()

    def clear_experiments(self) -> None:  # noqa: B027
        """Delete all experiment artifacts (traces, log, report, working files).

        Default implementation is a no-op; backends that can enumerate
        artifacts should override this method.
        """
