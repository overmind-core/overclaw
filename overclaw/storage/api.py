"""API storage backend for OverClaw.

Maps every ``StorageBackend`` operation to a call against the Overmind REST
API (via the generated ``openapi_client``).  The mapping is:

  Artifact          API resource
  ─────────         ────────────
  eval spec         Agent fields (input_schema, output_fields, …)
  dataset           Agent.eval_dataset.cases
  policy            Prompt (label="policy", replace: delete old + one create)
  trace             Trace + nested Span records
  report / code     Job.report_markdown / Job.best_agent_code
  artifact (other)  no-op — FS-only artifacts are ignored silently

Writes are always fire-and-forget (non-blocking); reads block with a
configurable timeout.  All failures are swallowed so that an unreachable
API never breaks the optimizer.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from uuid import UUID

from overclaw.client import (
    _fire,
    _create_trace,
    _patch_job,
    _run_async,
    _submit_async,
    create_policy_prompt,
    get_client,
    get_project_id,
    upsert_agent,
)
from overclaw.openapi_client.models.patched_agent_request import PatchedAgentRequest
from overclaw.storage.base import StorageBackend


class ApiBackend(StorageBackend):
    """Stores OverClaw artifacts in the Overmind API.

    Parameters
    ----------
    agent_id:
        Overmind agent UUID string.  May be updated in-place after
        ``save_spec`` upserts the agent record.
    agent_path:
        Local path to the agent file — used by slug derivation and by
        ``save_policy`` to attach agent source code.
    job_id:
        Overmind job UUID string.  Required for report / iteration
        persistence.  Can be set later via :attr:`job_id`.
    client:
        Pre-built ``OverClawClient``.  When ``None`` (default), the client
        is created lazily from ``OVERMIND_API_URL`` / ``OVERMIND_API_TOKEN``
        environment variables.
    """

    def __init__(
        self,
        agent_id: str,
        agent_path: str,
        *,
        job_id: str | None = None,
        client: Any = None,
    ) -> None:
        self._agent_id = agent_id
        self._agent_path = agent_path
        self._job_id = job_id
        self._client = client
        self._dim_keys: list[str] = []

    # ------------------------------------------------------------------
    # Identity helpers (StorageBackend interface + extras)
    # ------------------------------------------------------------------

    def get_agent_id(self) -> str | None:
        """Return the Overmind agent UUID, or ``None`` if not yet resolved."""
        return self._agent_id or None

    def set_job_id(self, job_id: str) -> None:
        """Bind this backend to a specific optimization job."""
        self._job_id = job_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str) -> None:
        self._agent_id = value

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @job_id.setter
    def job_id(self, value: str | None) -> None:
        self._job_id = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client_(self):
        """Return the API client, building it lazily if needed."""
        if self._client is not None:
            return self._client

        return get_client()

    def _run(self, coro, timeout: float = 30.0) -> Any:
        """Block until *coro* completes and return its result."""
        return _run_async(coro, timeout=timeout)

    def _fire(self, coro) -> None:
        """Submit *coro* to the background event loop and return immediately."""
        _submit_async(coro)

    def _patch_job(self, **fields: Any) -> None:
        """Patch the current job record (fire-and-forget)."""
        if not self._job_id:
            return
        client = self._client_()
        if not client:
            return
        with contextlib.suppress(Exception):
            _patch_job(client, self._job_id, **fields)

    def _project_id(self) -> str | None:
        return get_project_id()

    # ------------------------------------------------------------------
    # Eval spec
    # ------------------------------------------------------------------

    def save_spec(self, spec: dict) -> None:
        """Upsert the agent record with spec fields.

        Updates :attr:`agent_id` in-place if the API creates a new record.
        """
        client = self._client_()
        if not client:
            return
        project_id = self._project_id()
        if not project_id:
            return
        # First-write bootstrap must be synchronous so we can capture and keep
        # the remote agent id for subsequent non-blocking writes.
        if not self._agent_id:
            try:
                result = upsert_agent(
                    client,
                    project_id=project_id,
                    agent_path=self._agent_path,
                    spec=spec,
                )
                self._agent_id = str(result.id)
                return
            except Exception:
                return

        # Existing agent: patch in background to avoid user-facing latency.
        _fire(
            upsert_agent,
            client,
            project_id=project_id,
            agent_path=self._agent_path,
            spec=spec,
        )

    def load_spec(self) -> dict | None:
        """Fetch spec fields from the agent record."""
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            agent = self._run(client.agents_retrieve(id=UUID(self._agent_id)))
            spec: dict = {
                "agent_description": agent.description or "",
                "agent_path": agent.agent_path or self._agent_path,
                "input_schema": agent.input_schema or {},
                "output_fields": agent.output_fields or {},
                "structure_weight": (
                    agent.structure_weight if agent.structure_weight is not None else 20
                ),
                "total_points": (
                    agent.total_points if agent.total_points is not None else 100
                ),
            }
            for key in (
                "tool_config",
                "consistency_rules",
                "optimizable_elements",
                "fixed_elements",
            ):
                val = getattr(agent, key, None)
                if val:
                    spec[key] = val
            if agent.tool_usage_weight is not None:
                spec["tool_usage_weight"] = agent.tool_usage_weight
            # Embed policy stored alongside the dataset blob
            if agent.eval_dataset and isinstance(agent.eval_dataset, dict):
                policy_data = agent.eval_dataset.get("policy")
                if policy_data:
                    spec["policy"] = policy_data
            return spec
        except Exception:
            return None

    def delete_spec(self) -> None:
        """Delete the agent record from the API."""
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        try:
            self._run(client.agents_destroy(id=UUID(self._agent_id)))
            self._agent_id = ""
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def save_dataset(self, cases: list[dict]) -> None:
        """Upsert the agent record with the dataset cases."""
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        project_id = self._project_id()
        if not project_id:
            return
        spec = self.load_spec() or {}
        _fire(
            upsert_agent,
            client,
            project_id=project_id,
            agent_path=self._agent_path,
            spec=spec,
            dataset=cases,
        )

    def load_dataset(self) -> list[dict] | None:
        """Fetch dataset cases from the agent record's eval_dataset blob."""
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            agent = self._run(client.agents_retrieve(id=UUID(self._agent_id)))
            if not agent.eval_dataset:
                return None
            blob = agent.eval_dataset
            if isinstance(blob, dict):
                return blob.get("cases") or []
            if isinstance(blob, list):
                return blob
            return None
        except Exception:
            return None

    def delete_dataset(self) -> None:
        """Clear dataset cases from the agent record (sets eval_dataset to None)."""
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        try:
            patch = PatchedAgentRequest(eval_dataset=None)
            self._run(
                client.agents_partial_update(
                    id=UUID(self._agent_id),
                    patched_agent_request=patch,
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def save_policy(self, policy_md: str, policy_data: dict | None = None) -> None:
        """Upsert the Prompt with ``label="policy"`` (update in place when present)."""
        if not self._agent_id:
            return

        client = self._client_()
        if not client:
            return
        try:
            agent_code: str | None = None
            with contextlib.suppress(Exception):
                agent_code = Path(self._agent_path).read_text(encoding="utf-8")
            create_policy_prompt(
                client,
                agent_id=self._agent_id,
                policy_md=policy_md,
                agent_code=agent_code,
            )
        except Exception:
            pass

    def load_policy(self) -> str | None:
        """Fetch the most-recent policy prompt's Markdown from the API."""
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            page = self._run(client.prompts_list(agent=UUID(self._agent_id)))
            # Return the first prompt with label="policy"
            for prompt in page.results or []:
                if getattr(prompt, "label", None) == "policy":
                    return getattr(prompt, "system_prompt", None)
            return None
        except Exception:
            return None

    def delete_policy(self) -> None:
        """Delete all policy prompts for this agent."""
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        try:
            page = self._run(client.prompts_list(agent=UUID(self._agent_id)))
            for prompt in page.results or []:
                if getattr(prompt, "label", None) == "policy":
                    with contextlib.suppress(Exception):
                        self._run(client.prompts_destroy(id=prompt.id))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def save_trace(self, trace_data: dict, run_name: str, idx: int) -> None:
        """Create a Trace record (with nested Span records) — fire-and-forget."""
        client = self._client_()
        if not client or not self._agent_id:
            return

        # Ensure trace_group is set to run_name for later filtering
        payload = dict(trace_data)
        payload["trace_group"] = payload.get("trace_group") or run_name
        _create_trace(client, self._agent_id, self._job_id, payload)

    def delete_traces(self, run_name: str | None = None) -> None:
        """Delete traces from the API.

        When *run_name* is given only traces whose ``trace_group`` matches
        are deleted; otherwise all traces for this agent are removed.
        """
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        try:
            kwargs: dict[str, Any] = {"agent": UUID(self._agent_id)}
            if run_name:
                kwargs["trace_group"] = run_name

            page_num = 1
            while True:
                page = self._run(client.traces_list(**kwargs, page=page_num))
                for trace in page.results or []:
                    with contextlib.suppress(Exception):
                        self._run(client.traces_destroy(id=trace.id))
                if not page.next:
                    break
                page_num += 1
                if page_num > 50:
                    break
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generic artifacts
    # ------------------------------------------------------------------

    def save_artifact(self, content: str, name: str) -> None:
        """Persist well-known artifacts to the job record; all others are no-ops.

        Supported names:
        * ``"best_agent.py"``  → ``Job.best_agent_code``
        * ``"report.md"``      → ``Job.report_markdown``
        """
        if name == "best_agent.py":
            self._patch_job(best_agent_code=content)
        elif name == "report.md":
            self._patch_job(report_markdown=content)
        # All other artifact names are silently ignored — they are local
        # working files with no API equivalent.

    def load_artifact(self, name: str) -> str | None:
        """Retrieve a well-known artifact from the job record."""
        if not self._job_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            job = self._run(client.jobs_retrieve(id=UUID(self._job_id)))
            if name == "best_agent.py":
                return getattr(job, "best_agent_code", None)
            if name == "report.md":
                return getattr(job, "report_markdown", None)
            return None
        except Exception:
            return None

    def delete_artifact(self, name: str) -> None:
        """No-op for the API backend (no generic artifact storage)."""

    # ------------------------------------------------------------------
    # Results log  (in-memory accumulation; not pushed to API)
    # ------------------------------------------------------------------

    def init_results_log(self, dim_keys: list[str]) -> None:
        """Reset the in-memory results buffer."""
        self._dim_keys = list(dim_keys)
        self._results: list[dict] = []

    def append_result_row(self, row: dict, dim_keys: list[str]) -> None:
        """Accumulate rows in memory.

        The API backend does not push TSV rows to the backend; per-iteration
        progress is already streamed via ``ApiReporter``.  This method exists
        so callers can use the same interface regardless of backend.
        """
        if not hasattr(self, "_results"):
            self._results = []
        self._results.append(dict(row))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def save_report(self, report_md: str, best_code: str | None = None) -> None:
        """Patch the job record with the report and best agent code."""
        if not self._job_id:
            return
        fields: dict[str, Any] = {"report_markdown": report_md}
        if best_code is not None:
            fields["best_agent_code"] = best_code
        self._patch_job(**fields)

    def load_report(self) -> str | None:
        """Fetch the report Markdown from the job record."""
        return self.load_artifact("report.md")

    # ------------------------------------------------------------------
    # Bulk cleanup  (no-op — nothing to sweep in the API)
    # ------------------------------------------------------------------

    def clear_setup_spec(self) -> None:
        """Delete the agent record entirely from the API."""
        self.delete_spec()

    def clear_experiments(self) -> None:
        """Delete all traces for this agent and clear the job report fields."""
        self.delete_traces()
        if self._job_id:
            self._patch_job(report_markdown=None, best_agent_code=None)
