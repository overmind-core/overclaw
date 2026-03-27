"""OverClaw API client.

Configure via environment variables:
    OVERMIND_API_URL      Base URL of the Overmind backend (e.g. http://localhost:8000)
    OVERMIND_API_TOKEN    Bearer token  (ovr_core_... or any JWT)
    OVERMIND_PROJECT_ID   UUID of the project to associate agents with

Usage::

    from overclaw.client import get_client, upsert_agent, fetch_agent_spec_and_dataset
    from overclaw.client import ApiReporter
    from overclaw.client import read_project_toml, write_project_toml
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from uuid import UUID

import tomlkit

from overclaw.openapi_client import ApiClient, Configuration
from overclaw.openapi_client.api.agents_api import AgentsApi
from overclaw.openapi_client.api.auth_api import AuthApi
from overclaw.openapi_client.api.job_iterations_api import JobIterationsApi
from overclaw.openapi_client.api.jobs_api import JobsApi
from overclaw.openapi_client.api.organisations_api import OrganisationsApi
from overclaw.openapi_client.api.projects_api import ProjectsApi
from overclaw.openapi_client.api.prompts_api import PromptsApi
from overclaw.openapi_client.api.spans_api import SpansApi
from overclaw.openapi_client.api.suggestions_api import SuggestionsApi
from overclaw.openapi_client.api.traces_api import TracesApi
from overclaw.openapi_client.models.agent_request import AgentRequest
from overclaw.openapi_client.models.job_iteration_request import JobIterationRequest
from overclaw.openapi_client.models.job_iteration_status_enum import (
    JobIterationStatusEnum,
)
from overclaw.openapi_client.models.job_request import JobRequest
from overclaw.openapi_client.models.patched_agent_request import PatchedAgentRequest
from overclaw.openapi_client.models.patched_job_request import PatchedJobRequest
from overclaw.openapi_client.models.prompt_request import PromptRequest
from overclaw.openapi_client.models.span_request import SpanRequest
from overclaw.openapi_client.models.span_type_enum import SpanTypeEnum
from overclaw.openapi_client.models.status1b7_enum import Status1b7Enum
from overclaw.openapi_client.models.trace_create_request import TraceCreateRequest

# Module-level thread pool for fire-and-forget API pushes.
# Daemon threads so they don't block process exit.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="overclaw-api")
_pending_futures: set[Future] = set()
_pending_lock = threading.Lock()

# Serialize policy sync per agent so concurrent upserts cannot each create a row.
_policy_upsert_locks: dict[str, threading.Lock] = {}
_policy_upsert_locks_guard = threading.Lock()


def _policy_upsert_lock(agent_id: str) -> threading.Lock:
    with _policy_upsert_locks_guard:
        if agent_id not in _policy_upsert_locks:
            _policy_upsert_locks[agent_id] = threading.Lock()
        return _policy_upsert_locks[agent_id]


def _track_future(fut: Future) -> Future:
    """Track a background future so pending writes can be flushed on demand."""
    with _pending_lock:
        _pending_futures.add(fut)

    def _done(_f: Future) -> None:
        with _pending_lock:
            _pending_futures.discard(_f)

    fut.add_done_callback(_done)
    return fut


# ---------------------------------------------------------------------------
# Shared background async event loop
# ---------------------------------------------------------------------------
# A single event loop runs forever in a daemon thread.  All async API calls
# are submitted here so they never block the main thread or user code.

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_loop_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop
    if _bg_loop is None or not _bg_loop.is_running():
        with _bg_loop_lock:
            if _bg_loop is None or not _bg_loop.is_running():
                _bg_loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=_bg_loop.run_forever,
                    daemon=True,
                    name="overclaw-async",
                )
                t.start()
    return _bg_loop


def _submit_async(coro) -> Future:
    """Fire-and-forget: submit a coroutine to the background loop; never blocks."""
    fut = asyncio.run_coroutine_threadsafe(coro, _get_bg_loop())
    return _track_future(fut)


def _run_async(coro, timeout: float = 30.0) -> Any:
    """Submit a coroutine to the background loop and wait for its result."""
    return asyncio.run_coroutine_threadsafe(coro, _get_bg_loop()).result(
        timeout=timeout
    )


def flush_pending_api_updates(timeout: float = 8.0) -> None:
    """Best-effort wait for currently pending background API writes.

    Intended for graceful shutdown paths (e.g. Ctrl+C) so queued writes are
    pushed before process exit.
    """
    with _pending_lock:
        pending = list(_pending_futures)
    if not pending:
        return
    done, not_done = wait(pending, timeout=timeout)
    # Best-effort completion; cancel leftovers to avoid dangling references.
    for fut in not_done:
        with _pending_lock:
            _pending_futures.discard(fut)
        with contextlib.suppress(Exception):
            fut.cancel()


class OverClawClient(
    AgentsApi,
    AuthApi,
    JobIterationsApi,
    JobsApi,
    OrganisationsApi,
    ProjectsApi,
    PromptsApi,
    SpansApi,
    SuggestionsApi,
    TracesApi,
): ...


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_client() -> OverClawClient | None:
    """Return a configured client if OVERMIND_API_URL and OVERMIND_API_TOKEN are set.

    The token is sent as ``Authorization: Bearer <token>``.
    Returns ``None`` when either variable is missing.
    """
    base_url = os.getenv("OVERMIND_API_URL", "").strip().rstrip("/")
    token = os.getenv("OVERMIND_API_TOKEN", "").strip()
    if not base_url or not token:
        return None
    cfg = Configuration(host=base_url, api_key=token)
    cfg.access_token = token
    cfg.proxy_headers = {
        "X-Api-Key": token,
    }
    return OverClawClient(api_client=ApiClient(configuration=cfg))


def is_configured() -> bool:
    """Return True if both OVERMIND_API_URL and OVERMIND_API_TOKEN are set."""
    return bool(
        os.getenv("OVERMIND_API_URL", "").strip()
        and os.getenv("OVERMIND_API_TOKEN", "").strip()
    )


def get_project_id() -> str | None:
    """Return OVERMIND_PROJECT_ID from env, or None."""
    return os.getenv("OVERMIND_PROJECT_ID", "").strip() or None


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------


def _make_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:50] or "agent"


def agent_slug_from_path(agent_path: str) -> str:
    """Derive a stable, URL-safe slug from the agent file path.

    e.g. ``agents/agent1/sample_agent.py``  →  ``agent1-sample-agent``
    """
    p = Path(agent_path).resolve()
    stem = p.stem  # "sample_agent"
    parent = p.parent.name  # "agent1"
    name = f"{parent}-{stem}" if parent else stem
    return _make_slug(name)


# ---------------------------------------------------------------------------
# Per-agent project.toml helpers
# ---------------------------------------------------------------------------


def _project_toml_path(agent_path: str) -> Path:
    """Return the path to the per-agent project.toml file.

    Lives next to the agent file, e.g.
    ``agents/agent1/sample_agent.py``  →  ``agents/agent1/project.toml``
    """
    return Path(agent_path).resolve().parent / "project.toml"


def write_project_toml(agent_path: str, data: dict) -> None:
    """Write *data* to the per-agent project.toml, merging with any existing content.

    Only keys present in *data* are updated; all other keys are preserved.
    """
    p = _project_toml_path(agent_path)
    try:
        existing: tomlkit.TOMLDocument
        if p.exists():
            existing = tomlkit.loads(p.read_text(encoding="utf-8"))
        else:
            existing = tomlkit.document()
            existing.add(tomlkit.comment("OverClaw — auto-generated per-agent config"))
            existing.add(tomlkit.nl())

        def _deep_set(doc: Any, keys: list[str], value: Any) -> None:
            for key in keys[:-1]:
                if key not in doc:
                    doc.add(key, tomlkit.table())
                doc = doc[key]
            doc[keys[-1]] = value

        def _flatten_and_set(doc: Any, d: dict, prefix: list[str]) -> None:
            for k, v in d.items():
                if isinstance(v, dict):
                    _flatten_and_set(doc, v, prefix + [k])
                else:
                    _deep_set(doc, prefix + [k], v)

        _flatten_and_set(existing, data, [])
        p.write_text(tomlkit.dumps(existing), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Background fire-and-forget
# ---------------------------------------------------------------------------


def _fire(fn, *args, **kwargs) -> None:
    """Submit *fn* to the background thread pool — returns immediately."""

    def _run():
        try:
            fn(*args, **kwargs)
        except Exception:
            pass  # Swallow silently; API failures must never break the optimizer

    fut = _executor.submit(_run)
    _track_future(fut)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def upsert_agent(
    client: OverClawClient,
    project_id: str,
    agent_path: str,
    spec: dict,
    dataset: list[dict] | None = None,
    policy_data: dict | None = None,
) -> Any:
    """Create or update the Agent record for *agent_path* in the API.

    Stores all eval-spec fields plus dataset cases + policy JSON in
    ``eval_dataset``.  Returns the Agent object.
    """
    slug = agent_slug_from_path(agent_path)
    description = (spec.get("agent_description") or "")[:512] or None
    name = (description or slug)[:255]

    eval_dataset: dict[str, Any] = {}
    if dataset is not None:
        eval_dataset["cases"] = dataset
    if policy_data:
        eval_dataset["policy"] = policy_data

    existing = None
    try:
        page = _run_async(client.agents_list(project=UUID(project_id)))
        for ag in page.results or []:
            if ag.slug == slug or ag.agent_path == agent_path:
                existing = ag
                break
    except Exception:
        # Fallback for backends that may not support `project` filter.
        try:
            page = _run_async(client.agents_list())
            for ag in page.results or []:
                ag_project = str(getattr(ag, "project", "") or "")
                if ag_project != str(project_id):
                    continue
                if ag.slug == slug or ag.agent_path == agent_path:
                    existing = ag
                    break
        except Exception:
            pass

    shared: dict[str, Any] = dict(
        description=description,
        agent_path=(agent_path or "")[:512] or None,
        input_schema=spec.get("input_schema"),
        output_fields=spec.get("output_fields"),
        structure_weight=spec.get("structure_weight"),
        total_points=spec.get("total_points"),
        tool_config=spec.get("tool_config"),
        tool_usage_weight=spec.get("tool_usage_weight"),
        consistency_rules=spec.get("consistency_rules"),
        optimizable_elements=spec.get("optimizable_elements"),
        fixed_elements=spec.get("fixed_elements"),
        eval_dataset=eval_dataset or None,
    )

    if existing:
        patch = PatchedAgentRequest(**shared)
        result = _run_async(
            client.agents_partial_update(id=existing.id, patched_agent_request=patch)
        )
    else:
        req = AgentRequest(name=name, slug=slug, project=UUID(project_id), **shared)
        result = _run_async(client.agents_create(agent_request=req))

    # Persist agent_id to the per-agent project.toml immediately

    return result


async def _list_all_prompts_for_agent(
    client: OverClawClient, agent_id: str
) -> list[Any]:
    """Walk every page of ``/api/prompts/?agent=…`` (DRF pagination)."""
    aid = UUID(agent_id)
    collected: list[Any] = []
    page_num = 1
    while page_num <= 200:
        page = await client.prompts_list(agent=aid, page=page_num)
        collected.extend(page.results or [])
        if not page.next:
            break
        page_num += 1
    return collected


async def _replace_policy_prompt_async(
    client: OverClawClient,
    agent_id: str,
    policy_md: str,
    agent_code: str | None,
) -> None:
    """Exactly one ``label=policy`` row: remove any existing, then create fresh.

    Avoids duplicate ``v1`` rows from races or older clients; does not bump
    version on an existing row — there is always a single replacement record.
    """
    aid = UUID(agent_id)
    for p in await _list_all_prompts_for_agent(client, agent_id):
        if getattr(p, "label", None) != "policy":
            continue
        with contextlib.suppress(Exception):
            await client.prompts_destroy(id=p.id)

    req = PromptRequest(
        agent=aid,
        label="policy",
        system_prompt=policy_md,
        full_agent_code=agent_code,
    )
    await client.prompts_create(prompt_request=req)


def create_policy_prompt(
    client: OverClawClient,
    agent_id: str,
    policy_md: str,
    agent_code: str | None = None,
) -> None:
    """Sync agent policy: replace every ``label=policy`` prompt with one new row.

    Runs synchronously (waits for the API) under a per-agent lock so concurrent
    calls cannot each create a policy row. Safe to call from ``overclaw sync``.
    """
    with _policy_upsert_lock(str(agent_id)):
        _run_async(
            _replace_policy_prompt_async(client, agent_id, policy_md, agent_code),
            timeout=60.0,
        )


def fetch_agent_spec_and_dataset(
    client: OverClawClient,
    agent_path: str,
    project_id: str | None = None,
) -> tuple[dict, list[dict], str] | None:
    """Fetch the eval spec, dataset, and agent UUID for *agent_path* from the API.

    Returns ``(spec_dict, dataset_cases, agent_id)`` on success, or ``None``
    if the agent is not found or the request fails.
    """
    slug = agent_slug_from_path(agent_path)
    try:
        kwargs: dict[str, Any] = {}
        if project_id:
            kwargs["project"] = project_id
        page = _run_async(client.agents_list(**kwargs))
        agent_stub = None
        for ag in page.results or []:
            if ag.slug == slug or ag.agent_path == agent_path:
                agent_stub = ag
                break
        if not agent_stub:
            return None
        # agents_list returns AgentList (lightweight); fetch full detail
        agent = _run_async(client.agents_retrieve(id=agent_stub.id))
    except Exception:
        return None

    spec: dict[str, Any] = {
        "agent_description": agent.description or "",
        "agent_path": agent.agent_path or agent_path,
        "input_schema": agent.input_schema or {},
        "output_fields": agent.output_fields or {},
        "structure_weight": agent.structure_weight
        if agent.structure_weight is not None
        else 20,
        "total_points": agent.total_points if agent.total_points is not None else 100,
    }
    if agent.tool_config:
        spec["tool_config"] = agent.tool_config
    if agent.tool_usage_weight is not None:
        spec["tool_usage_weight"] = agent.tool_usage_weight
    if agent.consistency_rules:
        spec["consistency_rules"] = agent.consistency_rules
    if agent.optimizable_elements:
        spec["optimizable_elements"] = agent.optimizable_elements
    if agent.fixed_elements:
        spec["fixed_elements"] = agent.fixed_elements

    dataset: list[dict] = []
    if agent.eval_dataset:
        blob = agent.eval_dataset
        if isinstance(blob, dict):
            dataset = blob.get("cases", [])
            policy_data = blob.get("policy")
            if policy_data:
                spec["policy"] = policy_data
        elif isinstance(blob, list):
            dataset = blob

    return spec, dataset, str(agent.id)


# ---------------------------------------------------------------------------
# Job / iteration helpers (used by ApiReporter)
# ---------------------------------------------------------------------------


def _create_job(
    client: OverClawClient,
    agent_id: str,
    analyzer_model: str,
    num_iterations: int,
    candidates_per_iteration: int,
) -> str | None:
    """Create a Job record and return its UUID string, or None on failure."""
    try:
        req = JobRequest(
            agent=UUID(agent_id),
            status=Status1b7Enum.RUNNING,
            analyzer_model=analyzer_model[:128],
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
            data_source="dataset",
        )
        job = _run_async(client.jobs_create(job_request=req))
        return str(job.id)
    except Exception:
        return None


def _patch_job(client: OverClawClient, job_id: str, **fields: Any) -> None:
    try:
        patch = PatchedJobRequest(**fields)
        _submit_async(
            client.jobs_partial_update(id=UUID(job_id), patched_job_request=patch)
        )
    except Exception:
        pass


def _create_iteration(
    client: OverClawClient,
    job_id: str,
    order: int,
    name: str,
    avg_score: float,
    status: JobIterationStatusEnum,
    description: str,
    agent_code: str | None,
    dimension_scores: dict | None,
) -> str | None:
    try:
        req = JobIterationRequest(
            job=UUID(job_id),
            iteration_name=name[:64],
            order=order,
            avg_score=avg_score,
            status=status,
            description=(description or "")[:500],
            agent_code=agent_code or "",
            dimension_scores=dimension_scores or {},
        )
        iteration = _run_async(client.job_iterations_create(job_iteration_request=req))
        return str(iteration.id)
    except Exception:
        return None


def fetch_traces_as_dataset(
    client: OverClawClient,
    agent_id: str,
    sample_size: int = 20,
    *,
    seed: int = 42,
) -> list[dict]:
    """Fetch traces for *agent_id* from the API and convert to dataset cases.

    Returns a list of ``{"input": ..., "expected_output": ...}`` dicts
    randomly sampled from available traces.  Returns an empty list when
    no traces exist or the request fails.
    """

    try:
        # Paginate through all trace IDs
        all_traces: list = []
        page_num = 1
        while True:
            page = _run_async(client.traces_list(agent=UUID(agent_id), page=page_num))
            results = page.results or []
            all_traces.extend(results)
            if not page.next:
                break
            page_num += 1
            if page_num > 20:  # safety cap
                break
    except Exception:
        return []

    if not all_traces:
        return []

    # Sample from available traces
    rng = random.Random(seed)
    sampled = rng.sample(all_traces, min(sample_size, len(all_traces)))

    # Fetch full detail for each sampled trace to get input_data / output_data
    cases: list[dict] = []
    for trace_stub in sampled:
        try:
            trace = _run_async(client.traces_retrieve(id=trace_stub.id))
            input_data = trace.input_data
            output_data = trace.output_data
            if not input_data and not output_data:
                continue
            cases.append(
                {
                    "input": input_data or {},
                    "expected_output": output_data or {},
                }
            )
        except Exception:
            continue

    return cases


def count_agent_traces(
    client: OverClawClient,
    agent_id: str,
) -> int:
    """Return the total number of traces for *agent_id*, or 0 on failure."""
    try:
        page = _run_async(client.traces_list(agent=UUID(agent_id), page=1))
        return page.count or len(page.results or [])
    except Exception:
        return 0


def _create_trace(
    client: OverClawClient,
    agent_id: str,
    job_id: str | None,
    trace_data: dict,
    *,
    iteration_id: str | None = None,
) -> None:
    """Create a Trace record (with nested spans) from an overclaw Trace dict.

    ``trace_data`` is the output of ``Trace.to_dict()``.
    """
    try:
        spans: list[SpanRequest] = []
        for idx, s in enumerate(trace_data.get("spans", [])):
            st = s.get("span_type", "llm_call")
            span_type = (
                SpanTypeEnum.LLM_CALL if st == "llm_call" else SpanTypeEnum.TOOL_CALL
            )
            spans.append(
                SpanRequest(
                    span_type=span_type,
                    name=(s.get("name", "unknown"))[:255],
                    order=idx,
                    start_time=s.get("start_time", 0),
                    end_time=s.get("end_time", 0),
                    latency_ms=s.get("latency_ms", 0),
                    metadata=s.get("metadata"),
                    error=s.get("error"),
                )
            )

        req = TraceCreateRequest(
            agent=UUID(agent_id),
            job=UUID(job_id) if job_id else None,
            iteration=UUID(iteration_id) if iteration_id else None,
            trace_id=(trace_data.get("trace_id", "unknown"))[:255],
            trace_group=(trace_data.get("trace_group", ""))[:128] or None,
            input_data=trace_data.get("input_data"),
            output_data=trace_data.get("output_data"),
            total_latency_ms=trace_data.get("total_latency_ms", 0),
            total_tokens=int(trace_data.get("total_tokens", 0)),
            total_cost=trace_data.get("total_cost", 0),
            score=trace_data.get("score", 0),
            error=trace_data.get("error"),
            start_time=trace_data.get("start_time", 0),
            end_time=trace_data.get("end_time", 0),
        )
        # Create the trace and get its id, then create spans for that trace.
        fut = _submit_async(client.traces_create(trace_create_request=req))
        try:
            trace = fut.result(timeout=10)
            trace_id = str(trace.id)
        except Exception:
            return

        for span in spans:
            span.trace = trace_id
            _submit_async(client.spans_create(span_request=span))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ApiReporter — streams optimize progress to the backend in real time
# ---------------------------------------------------------------------------


class ApiReporter:
    """Streams optimize progress events to the Overmind API as they happen.

    All API calls are submitted to a background thread pool so they never
    block or slow down the optimization loop.  Failures are silently swallowed.

    Usage::

        reporter = ApiReporter.create(agent_id, config)
        reporter.on_baseline(score)
        reporter.on_iteration(i, eval_result, "keep", code, "improved prompt", dims)
        reporter.on_complete(best_score, improvement, report_md, best_code)
    """

    def __init__(
        self,
        client: OverClawClient,
        agent_id: str,
        job_id: str,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._job_id = job_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        agent_id: str,
        analyzer_model: str,
        num_iterations: int,
        candidates_per_iteration: int,
    ) -> ApiReporter | None:
        """Build a reporter if the API is configured.  Returns None otherwise."""
        client = get_client()
        if not client or not agent_id:
            return None

        job_id = _create_job(
            client,
            agent_id=agent_id,
            analyzer_model=analyzer_model,
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
        )
        if not job_id:
            return None

        return cls(client, agent_id, job_id)

    # ------------------------------------------------------------------
    # Event hooks (all fire-and-forget)
    # ------------------------------------------------------------------

    def on_baseline(self, score: float) -> None:
        """Called once the baseline has been evaluated."""
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            baseline_score=score,
            status=Status1b7Enum.RUNNING,
        )

    def on_iteration(
        self,
        order: int,
        avg_score: float,
        decision: str,
        agent_code: str | None = None,
        description: str = "",
        dimension_scores: dict | None = None,
    ) -> None:
        """Called after each iteration is accepted or rejected."""
        status = (
            JobIterationStatusEnum.KEEP
            if decision == "keep"
            else JobIterationStatusEnum.DISCARD
        )
        _fire(
            _create_iteration,
            self._client,
            self._job_id,
            order,
            f"iter_{order:03d}",
            avg_score,
            status,
            description,
            agent_code,
            dimension_scores,
        )

    def on_complete(
        self,
        best_score: float,
        baseline_score: float,
        report_markdown: str | None = None,
        best_agent_code: str | None = None,
        backtest_results: dict | None = None,
    ) -> None:
        """Called when the full optimization run is done."""
        improvement = best_score - baseline_score
        fields: dict[str, Any] = {
            "status": Status1b7Enum.COMPLETED,
            "best_score": best_score,
            "improvement": improvement,
        }
        if report_markdown is not None:
            fields["report_markdown"] = report_markdown
        if best_agent_code is not None:
            fields["best_agent_code"] = best_agent_code
        if backtest_results is not None:
            fields["backtest_results"] = backtest_results
        _fire(_patch_job, self._client, self._job_id, **fields)

    def on_trace(self, trace_data: dict) -> None:
        """Called after each test case to upload the trace to the API.

        ``trace_data`` is the output of ``Trace.to_dict()``.
        """
        _fire(
            _create_trace,
            self._client,
            self._agent_id,
            self._job_id,
            trace_data,
        )

    def on_holdout(self, holdout_results: dict) -> None:
        """Called after holdout evaluation to store holdout metrics on the Job.

        Merges holdout data into ``backtest_results`` since both are JSONB.
        """
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            backtest_results={"holdout": holdout_results},
        )

    @property
    def job_id(self) -> str:
        return self._job_id

    def on_failed(self, reason: str = "") -> None:
        """Called if the optimization run aborts with an error."""
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            status=Status1b7Enum.FAILED,
            report_markdown=f"Run failed: {reason}",
        )
