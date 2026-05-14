"""Overmind API client.

A thin, synchronous-feeling facade over the generated
``overmind.openapi_client`` SDK.  Configure via the standard environment
variables:

* ``OVERMIND_API_URL``     Base URL of the Overmind backend (e.g. ``http://localhost:8000``)
* ``OVERMIND_API_KEY``     Either a project-scoped key (``ovr_…``) or a user JWT.
* ``OVERMIND_PROJECT_ID``  Optional UUID of the project to associate agents with.
                           When the key is project-scoped and the user has access
                           to exactly one project, this is auto-discovered via
                           :func:`resolve_project_id`.

Why this module exists
----------------------
The generated SDK is fully asynchronous, but the Overmind CLI is a
synchronous-first surface.  Wrapping every call in ``asyncio.run(...)``
at the call site is noisy and accidentally creates / tears down a new
event loop on every invocation.  Instead, we keep a single background
asyncio loop running on a daemon thread and route every coroutine into
it — ``OvermindClient`` then exposes ergonomic, blocking helpers
(``get_agent``, ``resolve_agent``, ``upsert_dataset`` …) that hide that
machinery from the caller.

Example::

    from overmind.client import get_client

    client = get_client()
    agent = client.resolve_agent("support-triage")
    dataset = client.upsert_dataset(agent.id, datapoints=[...], source="seed")
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from uuid import UUID

import tomlkit

from overmind.core.constants import DEFAULT_BASE_URL
from overmind.openapi_client import ApiClient, Configuration
from overmind.openapi_client.api.agents_api import AgentsApi
from overmind.openapi_client.api.auth_api import AuthApi
from overmind.openapi_client.api.datasets_api import DatasetsApi
from overmind.openapi_client.api.job_iterations_api import JobIterationsApi
from overmind.openapi_client.api.jobs_api import JobsApi
from overmind.openapi_client.api.projects_api import ProjectsApi
from overmind.openapi_client.api.traces_api import TracesApi
from overmind.openapi_client.models.agent_request import AgentRequest
from overmind.openapi_client.models.datapoint_request import DatapointRequest
from overmind.openapi_client.models.dataset_request import DatasetRequest
from overmind.openapi_client.models.job_iteration_request import JobIterationRequest
from overmind.openapi_client.models.job_iteration_status_enum import (
    JobIterationStatusEnum,
)
from overmind.openapi_client.models.job_request import JobRequest
from overmind.openapi_client.models.job_status_enum import JobStatusEnum
from overmind.openapi_client.models.patched_agent_request import PatchedAgentRequest
from overmind.openapi_client.models.patched_job_request import PatchedJobRequest
from overmind.openapi_client.models.source_enum import SourceEnum

logger = logging.getLogger("overmind.client")

# ---------------------------------------------------------------------------
# Background execution: thread pool + asyncio loop
# ---------------------------------------------------------------------------
# Fire-and-forget API writes are dispatched here so the optimizer / setup
# loops never block on network I/O.  Daemon threads so they don't block exit.

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="overmind-api")
_pending_futures: set[Future] = set()
_pending_lock = threading.Lock()

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_loop_lock = threading.Lock()


def _track_future(fut: Future) -> Future:
    """Track a background future so pending writes can be flushed on demand."""
    with _pending_lock:
        _pending_futures.add(fut)

    def _done(_f: Future) -> None:
        with _pending_lock:
            _pending_futures.discard(_f)

    fut.add_done_callback(_done)
    return fut


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop
    if _bg_loop is None or not _bg_loop.is_running():
        with _bg_loop_lock:
            if _bg_loop is None or not _bg_loop.is_running():
                _bg_loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=_bg_loop.run_forever,
                    daemon=True,
                    name="overmind-async",
                )
                t.start()
                logger.debug(f"Started background asyncio loop thread={t.name} loop={_bg_loop!r}")
    return _bg_loop


def _submit_async(coro) -> Future:
    """Fire-and-forget: submit a coroutine to the background loop; never blocks."""
    fut = asyncio.run_coroutine_threadsafe(coro, _get_bg_loop())
    return _track_future(fut)


def _run_async(coro, timeout: float = 30.0) -> Any:
    """Submit a coroutine to the background loop and wait for its result.

    Strictly rejects non-coroutine arguments. The generated OpenAPI SDK is
    fully **synchronous** — calling ``_run_async(client.agents_partial_update(...))``
    silently fires the HTTP request, then raises ``TypeError`` from inside
    ``asyncio.run_coroutine_threadsafe``, where every existing call site has
    a bare ``except Exception:`` that swallows it and reports failure to the
    caller. The result is "remote write succeeded, local state thinks it
    failed" — the worst possible debug experience. Failing loudly with a
    typed error here means future "let me wrap this in ``_run_async``"
    attempts blow up in CI rather than in production.
    """
    if not asyncio.iscoroutine(coro):
        raise TypeError(
            f"_run_async expected a coroutine, got {type(coro).__name__}. "
            "The generated SDK is synchronous — call the method directly "
            "instead of wrapping it in _run_async()."
        )
    return asyncio.run_coroutine_threadsafe(coro, _get_bg_loop()).result(timeout=timeout)


def _fire(fn, *args, **kwargs) -> None:
    """Submit *fn* to the background thread pool — returns immediately."""
    fn_label = getattr(fn, "__name__", repr(fn))

    def _run() -> None:
        logger.debug(f"_fire: running {fn_label} on background thread")
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception(f"_fire: background {fn_label} failed")

    fut = _executor.submit(_run)
    _track_future(fut)


def flush_pending_api_updates(timeout: float = 8.0) -> None:
    """Best-effort wait for currently pending background API writes."""
    with _pending_lock:
        pending = list(_pending_futures)
    if not pending:
        logger.debug("flush_pending_api_updates: nothing to flush")
        return
    logger.info(f"Flushing {len(pending)} pending API update(s) (timeout={timeout:.1f}s)")
    done, not_done = wait(pending, timeout=timeout)
    logger.info(f"Flush complete: done={len(done)} not_done={len(not_done)}")
    for fut in not_done:
        with _pending_lock:
            _pending_futures.discard(fut)
        with contextlib.suppress(Exception):
            fut.cancel()


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


class OvermindClient(
    AgentsApi,
    AuthApi,
    DatasetsApi,
    JobIterationsApi,
    JobsApi,
    ProjectsApi,
    TracesApi,
):
    """Convenience facade over the generated OpenAPI client.

    The base classes (``AgentsApi``, ``ProjectsApi``, …) expose every
    endpoint as a **synchronous** method — they return decoded model
    objects (``Agent``, ``PaginatedProjectList``, …) directly. Call them
    in the obvious way::

        agent = client.agents_retrieve(id=agent_id)
        page  = client.projects_list(page_size=2)

    Do **not** wrap SDK calls in :func:`_run_async`. The generated client
    is sync; :func:`_run_async` is reserved for the few coroutines this
    module still defines internally (background trace flushing etc.).
    Wrapping a sync call in :func:`_run_async` will raise ``TypeError``
    by construction — see that function's docstring for the why.

    This subclass adds higher-level helpers (``resolve_agent``,
    ``upsert_dataset``, ``patch_job``…) for the operations the CLI calls
    most often, so call sites stay readable.
    """

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def get_agent(self, agent_slug: str) -> Any:
        """Return the agent identified by *agent_slug*.

        Raises :class:`ValueError` when no agent with that slug exists
        in the project bound to this client's API token.
        """
        page = self.agents_list(slug=agent_slug)
        results = page.results or []
        if not results:
            raise ValueError(f"Agent {agent_slug!r} not found")
        return results[0]

    def resolve_agent(self, agent_slug: str) -> Any | None:
        """Return the agent identified by *agent_slug*, or ``None``.

        The non-raising counterpart of :meth:`get_agent` — convenient
        when the caller wants to "create if missing" without juggling
        exceptions.
        """
        try:
            return self.get_agent(agent_slug)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def upsert_dataset(
        self,
        agent_id: str | UUID,
        datapoints: list[dict],
        *,
        source: str = "synthetic",
        generator_model: str = "",
        policy_hash: str = "",
        metadata: dict | None = None,
        name: str = "",
        make_active: bool = True,
    ) -> dict | None:
        """Create a new versioned :class:`Dataset` for *agent_id*.

        Returns the JSON-serialised dataset record on success, or
        ``None`` when the API call fails (the error is logged).
        """
        return create_dataset(
            self,
            agent_id=str(agent_id),
            datapoints=datapoints,
            source=source,
            generator_model=generator_model,
            policy_hash=policy_hash,
            metadata=metadata,
            name=name,
            make_active=make_active,
        )


def get_client() -> OvermindClient | None:
    """Return a configured client if ``OVERMIND_API_KEY`` is set.

    ``OVERMIND_API_URL`` is optional: when unset, it falls back to
    :data:`overmind.core.constants.DEFAULT_BASE_URL` (the Overmind Cloud
    endpoint), mirroring the behaviour of :mod:`overmind.tracing`.
    Self-hosted deployments override the default by setting
    ``OVERMIND_API_URL`` explicitly.

    **Auth routing:** the SDK previously stamped the token into both
    ``Authorization: Bearer …`` *and* ``X-Api-Key``.  The backend prefers
    the Bearer header and resolves it via the user-JWT path, which grants
    the request access to **every** project the user belongs to — making
    ``OVERMIND_PROJECT_ID`` feel mandatory even for project-scoped keys.
    We now route ``ovr_…`` keys through ``ApiKeyAuth`` only so the backend
    can derive the project directly from the key, and only fall back to
    ``Authorization: Bearer`` for plain JWTs / dev tokens.
    """
    base_url = (os.getenv("OVERMIND_API_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/")
    token = os.getenv("OVERMIND_API_KEY", "").strip()
    if not token:
        logger.debug("get_client: API not configured (OVERMIND_API_KEY not set)")
        return None

    cfg = Configuration(host=base_url)
    if token.startswith("ovr_"):
        # Project-scoped API key: send ONLY as X-Api-Key so the backend
        # routes through the key-auth middleware (project derived from
        # the key) and never falls back to the user-JWT path.
        cfg.api_key = {"ApiKeyAuth": token}
        logger.debug(f"get_client: using ApiKeyAuth (ovr_* key) for host={base_url}")
    else:
        # Plain JWT / cookie / dev token: legacy Bearer path.
        cfg.access_token = token
        logger.debug(f"get_client: using BearerAuth (user JWT) for host={base_url}")
    return OvermindClient(api_client=ApiClient(configuration=cfg))


def is_configured() -> bool:
    """Return True if ``OVERMIND_API_KEY`` is set.

    ``OVERMIND_API_URL`` is optional — :func:`get_client` falls back to the
    Overmind Cloud default when it is unset — so the control-plane client is
    considered configured as soon as a bearer token is present.
    """
    return bool(os.getenv("OVERMIND_API_KEY", "").strip())


def get_project_id() -> str | None:
    """Return ``OVERMIND_PROJECT_ID`` from env, or None.

    Most callers should prefer :func:`resolve_project_id` which falls back
    to a single-project lookup via the Overmind API when the env var isn't
    set — that's the path that makes ``OVERMIND_PROJECT_ID`` optional for
    project-scoped ``ovr_…`` keys.
    """
    return os.getenv("OVERMIND_PROJECT_ID", "").strip() or None


class ProjectResolutionError(RuntimeError):
    """Raised when the project id can't be determined from env or the API.

    Carries a list of accessible project ids/slugs (when the failure was
    "multiple projects, ambiguous") so callers / skills can surface a
    helpful error without having to re-query the API.
    """

    def __init__(self, message: str, *, candidates: list[tuple[str, str]] | None = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


# Process-wide cache so repeated calls don't pay the projects_list cost.
_resolved_project_id: str | None = None


def resolve_project_id(client: OvermindClient | None = None) -> str:
    """Return the project UUID this client should write to.

    Resolution order:

    1. ``OVERMIND_PROJECT_ID`` env var (preferred — always wins).
    2. ``client.projects_list()`` — when the bound credentials see exactly
       one project, that id is used and cached in :envvar:`OVERMIND_PROJECT_ID`
       for the rest of the process.
    3. Otherwise raises :class:`ProjectResolutionError` with the candidate
       projects, so the caller can render a clear "pick one" message.

    Use this anywhere the previous code did ``get_project_id()`` *and* needed
    the value to be present.  Pure-display helpers that just want to know
    "did the user set the env var?" should keep calling :func:`get_project_id`.
    """
    global _resolved_project_id

    env_pid = get_project_id()
    if env_pid:
        return env_pid

    if _resolved_project_id:
        return _resolved_project_id

    if client is None:
        client = get_client()
    if client is None:
        raise ProjectResolutionError(
            "Cannot resolve project: OVERMIND_API_KEY is not set."
        )

    try:
        page = client.projects_list(page_size=2)
    except Exception as exc:
        raise ProjectResolutionError(
            "Cannot resolve project: projects_list() call failed. "
            "Set OVERMIND_PROJECT_ID explicitly to bypass auto-discovery."
        ) from exc

    results = list(page.results or [])
    if not results:
        raise ProjectResolutionError(
            "The configured OVERMIND_API_KEY has no accessible projects. "
            "Generate a project-scoped key from the Overmind console and try again."
        )

    if len(results) == 1:
        pid = str(results[0].id)
        _resolved_project_id = pid
        os.environ.setdefault("OVERMIND_PROJECT_ID", pid)
        logger.info(f"resolve_project_id: auto-resolved single project id={pid}")
        return pid

    candidates = [(str(p.id), getattr(p, "slug", "") or getattr(p, "name", "")) for p in results]
    listed = ", ".join(f"{slug or '?'}={pid}" for pid, slug in candidates)
    raise ProjectResolutionError(
        "OVERMIND_API_KEY can access multiple projects — set OVERMIND_PROJECT_ID "
        f"explicitly to pick one. Candidates: {listed}",
        candidates=candidates,
    )


def _reset_project_id_cache() -> None:
    """Test helper: clear the in-process project-id cache.

    Re-exported under :func:`overmind.client._reset_project_id_cache` so
    unit tests can flip the env var between cases without leaking state.
    """
    global _resolved_project_id
    _resolved_project_id = None


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------


def _make_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:50] or "agent"


def agent_slug_from_path(agent_path: str) -> str:
    """Derive a stable, URL-safe slug from the agent file path."""
    p = Path(agent_path).resolve()
    stem = p.stem
    parent = p.parent.name
    name = f"{parent}-{stem}" if parent else stem
    return _make_slug(name)


# ---------------------------------------------------------------------------
# Per-agent project.toml helpers
# ---------------------------------------------------------------------------


def _project_toml_path(agent_path: str) -> Path:
    """Return the path to the per-agent project.toml file."""
    return Path(agent_path).resolve().parent / "project.toml"


def write_project_toml(agent_path: str, data: dict) -> None:
    """Write *data* to the per-agent project.toml, merging with any existing content."""
    p = _project_toml_path(agent_path)
    try:
        existing: tomlkit.TOMLDocument
        if p.exists():
            existing = tomlkit.loads(p.read_text(encoding="utf-8"))
        else:
            existing = tomlkit.document()
            existing.add(tomlkit.comment("Overmind — auto-generated per-agent config"))
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
    except Exception:  # noqa: S110
        pass


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def _agent_shared_fields(spec: dict, agent_path: str) -> dict[str, Any]:
    """Build the common Agent / PatchedAgent payload fields from a spec dict."""
    description = (spec.get("agent_description") or "")[:512] or None
    return dict(  # noqa: C408
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
    )


def upsert_agent(
    client: OvermindClient,
    project_id: str,
    agent_path: str,
    spec: dict,
    *,
    agent_name: str | None = None,
) -> Any:
    """Create or update the Agent record for *agent_path* in the API.

    Stores eval-spec fields on the Agent.  Datasets are persisted separately
    via :func:`create_dataset`; policy is stored on
    ``Agent.policy_markdown`` / ``Agent.policy_data`` via
    ``agents_partial_update``.

    When *agent_name* is supplied it is used as the slug base so the REST-API
    record stays in sync with the slug that the OTLP ingest path derives from
    ``overmind.agent.name`` — preventing duplicate agents with different slugs
    (e.g. "support-triage" vs "support-triage-agent").

    Returns the Agent object (typed model from the OpenAPI client).
    """
    if agent_name:
        slug = _make_slug(agent_name)
    else:
        slug = agent_slug_from_path(agent_path)
    description = (spec.get("agent_description") or "")[:512] or None
    name = (description or agent_name or slug)[:255]

    logger.info(f"upsert_agent: slug={slug} project_id={project_id}")

    existing = None
    try:
        page = client.agents_list(project=UUID(project_id))
        for ag in page.results or []:
            if ag.slug == slug or ag.agent_path == agent_path:
                existing = ag
                break
    except Exception:
        logger.debug(
            "upsert_agent: filtered list failed, falling back to unfiltered",
            exc_info=True,
        )
        try:
            page = client.agents_list()
            for ag in page.results or []:
                ag_project = str(getattr(ag, "project", "") or "")
                if ag_project != str(project_id):
                    continue
                if ag.slug == slug or ag.agent_path == agent_path:
                    existing = ag
                    break
        except Exception:
            logger.warning("upsert_agent: unfiltered list also failed", exc_info=True)

    shared = _agent_shared_fields(spec, agent_path)

    if existing:
        patch = PatchedAgentRequest(**shared)
        result = client.agents_partial_update(id=existing.id, patched_agent_request=patch)
        logger.info(f"upsert_agent: updated existing agent id={existing.id} slug={slug}")
    else:
        req = AgentRequest(name=name, slug=slug, project=UUID(project_id), **shared)
        result = client.agents_create(agent_request=req)
        logger.info(f"upsert_agent: created new agent id={getattr(result, 'id', '?')} slug={slug}")
    return result


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _source_enum(source: str) -> SourceEnum:
    """Map a string to :class:`SourceEnum`, defaulting to synthetic."""
    key = (source or "").strip().lower()
    mapping: dict[str, SourceEnum] = {
        "seed": SourceEnum.SEED,
        "synthetic": SourceEnum.SYNTHETIC,
        "augmented": SourceEnum.AUGMENTED,
        "production": SourceEnum.PRODUCTION,
    }
    return mapping.get(key, SourceEnum.SYNTHETIC)


def _datapoint_request(dp: Any, index: int) -> DatapointRequest:
    """Coerce a loosely-typed case dict into ``DatapointRequest``."""
    if not isinstance(dp, dict):
        return DatapointRequest(order=index, input={"value": dp})
    return DatapointRequest(
        order=dp.get("order", index),
        input=dp.get("input", {}) or {},
        expected_output=dp.get("expected_output"),
        persona=(dp.get("persona", "") or "")[:128],
        tags=dp.get("tags", []) or [],
    )


def create_dataset(
    client: OvermindClient,
    agent_id: str,
    datapoints: list[dict],
    *,
    source: str = "synthetic",
    generator_model: str = "",
    policy_hash: str = "",
    metadata: dict | None = None,
    name: str = "",
    make_active: bool = True,
) -> dict | None:
    """``POST /api/datasets/`` via :meth:`DatasetsApi.datasets_create`."""
    req = DatasetRequest(
        agent=UUID(agent_id),
        name=name[:255],
        source=_source_enum(source),
        generator_model=(generator_model or "")[:128],
        policy_hash=(policy_hash or "")[:64],
        metadata=metadata or {},
        make_active=make_active,
        datapoints=[_datapoint_request(dp, i) for i, dp in enumerate(datapoints)],
    )
    try:
        created = client.datasets_create(dataset_request=req)
        return created.model_dump(mode="json") if created is not None else None
    except Exception:
        logger.exception(f"create_dataset failed agent_id={agent_id}")
        return None


def fetch_dataset_datapoints(client: OvermindClient, dataset_id: str) -> list[dict]:
    """Return every datapoint for *dataset_id* via :meth:`DatasetsApi.datasets_datapoints_list`."""
    collected: list[dict] = []
    page_num = 1
    while page_num <= 200:
        try:
            page = client.datasets_datapoints_list(id=UUID(dataset_id), page=page_num)
        except Exception:
            logger.debug(
                f"fetch_dataset_datapoints: page={page_num} failed dataset_id={dataset_id}",
                exc_info=True,
            )
            break
        results = page.results or []
        for dp in results:
            with contextlib.suppress(Exception):
                collected.append(dp.model_dump(mode="json"))
        if not page.next:
            break
        page_num += 1
    return collected


def get_active_dataset_id(client: OvermindClient, agent_id: str) -> str | None:
    """Return the agent's ``active_dataset`` UUID, or ``None``."""
    try:
        agent = client.agents_retrieve(id=UUID(agent_id))
    except Exception:
        return None
    active = getattr(agent, "active_dataset", None)
    return str(active) if active else None


def delete_dataset(client: OvermindClient, dataset_id: str) -> bool:
    """``DELETE /api/datasets/{id}/`` via :meth:`DatasetsApi.datasets_destroy`."""
    try:
        client.datasets_destroy(id=UUID(dataset_id))
        return True
    except Exception:
        logger.exception(f"delete_dataset failed dataset_id={dataset_id}")
        return False


# ---------------------------------------------------------------------------
# Job / iteration helpers (used by ApiReporter)
# ---------------------------------------------------------------------------


def _create_job(
    client: OvermindClient,
    agent_id: str,
    analyzer_model: str,
    num_iterations: int,
    candidates_per_iteration: int,
) -> str | None:
    """Create a Job record and return its UUID string, or None on failure."""
    try:
        req = JobRequest(
            agent=UUID(agent_id),
            status=JobStatusEnum.RUNNING,
            analyzer_model=analyzer_model[:128],
            num_iterations=num_iterations,
            candidates_per_iteration=candidates_per_iteration,
            data_source="dataset",
        )
        job = client.jobs_create(job_request=req)
        logger.info(f"_create_job: created job id={job.id} agent_id={agent_id} iterations={num_iterations}")
        return str(job.id)
    except Exception:
        logger.exception(f"_create_job: failed to create job for agent_id={agent_id}")
        return None


def _patch_job(client: OvermindClient, job_id: str, **fields: Any) -> None:
    try:
        patch = PatchedJobRequest(**fields)
        client.jobs_partial_update(id=UUID(job_id), patched_job_request=patch)
        logger.debug(f"_patch_job: job_id={job_id} fields={list(fields)}")
    except Exception:
        logger.exception(f"_patch_job: failed job_id={job_id}")


def _create_iteration(
    client: OvermindClient,
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
        iteration = client.job_iterations_create(job_iteration_request=req)
        logger.info(f"_create_iteration: job_id={job_id} order={order} status={status} avg_score={avg_score:.4f}")
        return str(iteration.id)
    except Exception:
        logger.exception(f"_create_iteration: failed job_id={job_id} order={order}")
        return None


# ---------------------------------------------------------------------------
# ApiReporter — streams optimize progress to the backend in real time
# ---------------------------------------------------------------------------


class ApiReporter:
    """Streams optimize lifecycle events to the Overmind backend.

    Telemetry (per-span attributes, tool calls, LLM usage, iteration
    analytics) is emitted by the OpenTelemetry pipeline and parsed by
    ``overbae.api.otlp`` — this reporter only owns the small set of
    Job / JobIteration writes the OTLP path cannot model cleanly:

    * Creating the parent :class:`Job` row up front so subsequent
      iteration spans have something to attach to.
    * Updating ``Job`` status, ``current_iteration`` and rolling logs
      while the run is in flight.
    * Stamping the final ``best_agent_code`` / ``report_markdown`` on
      the Job when the run completes.

    All writes go through :func:`_fire` / :func:`_submit_async` so the
    optimizer loop never blocks on network I/O.
    """

    def __init__(
        self,
        client: OvermindClient,
        agent_id: str,
        job_id: str,
    ) -> None:
        self._client = client or get_client()
        self._agent_id = agent_id
        self._job_id = job_id
        self._logs: list[dict] = []

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

    @property
    def job_id(self) -> str:
        return self._job_id

    def on_log(self, message: str, level: str = "info") -> None:
        """Append a log entry and push the current log buffer to the backend."""
        import time

        self._logs.append({"ts": time.time(), "level": level, "msg": message})
        _fire(_patch_job, self._client, self._job_id, logs=list(self._logs))

    def on_progress(self, current_iteration: int, best_score: float | None = None) -> None:
        """Update the job's current iteration and optionally its best score."""
        fields: dict[str, Any] = {"current_iteration": current_iteration}
        if best_score is not None:
            fields["best_score"] = best_score
        _fire(_patch_job, self._client, self._job_id, **fields)

    def on_baseline(self, score: float) -> None:
        """Called once the baseline has been evaluated."""
        import time

        self._logs.append({"ts": time.time(), "level": "info", "msg": f"Baseline evaluated: score {score:.2f}"})
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            baseline_score=score,
            best_score=score,
            status=JobStatusEnum.RUNNING,
            logs=list(self._logs),
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
        import time

        status = JobIterationStatusEnum.KEEP if decision == "keep" else JobIterationStatusEnum.DISCARD
        _fire(
            _create_iteration,
            self._client,
            self._job_id,
            order,
            f"Experiment {order}",
            avg_score,
            status,
            description,
            agent_code,
            dimension_scores,
        )
        # Update job's current iteration and best score in real time
        patch_fields: dict[str, Any] = {"current_iteration": order}
        if decision == "keep":
            patch_fields["best_score"] = avg_score
        decision_label = "accepted" if decision == "keep" else "discarded"
        self._logs.append({
            "ts": time.time(),
            "level": "info",
            "msg": f"Experiment {order}: {decision_label} (score {avg_score:.2f})"
            + (f" — {description}" if description else ""),
        })
        patch_fields["logs"] = list(self._logs)
        _fire(_patch_job, self._client, self._job_id, **patch_fields)

    def on_complete(
        self,
        best_score: float,
        baseline_score: float,
        report_markdown: str | None = None,
        best_agent_code: str | None = None,
        backtest_results: dict | None = None,
    ) -> None:
        """Called when the full optimization run is done."""
        import time

        improvement = best_score - baseline_score
        self._logs.append({
            "ts": time.time(),
            "level": "info",
            "msg": (f"Optimization complete — best score {best_score:.2f} (improvement {improvement:+.2f})"),
        })
        fields: dict[str, Any] = {
            "status": JobStatusEnum.COMPLETED,
            "best_score": best_score,
            "improvement": improvement,
            "logs": list(self._logs),
        }
        if report_markdown is not None:
            fields["report_markdown"] = report_markdown
        if best_agent_code is not None:
            fields["best_agent_code"] = best_agent_code
        if backtest_results is not None:
            fields["backtest_results"] = backtest_results
        _fire(_patch_job, self._client, self._job_id, **fields)

    def on_holdout(self, holdout_results: dict) -> None:
        """Called after holdout evaluation to store holdout metrics on the Job."""
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            backtest_results={"holdout": holdout_results},
        )

    def on_failed(self, reason: str = "") -> None:
        """Called if the optimization run aborts with an error."""
        import time

        self._logs.append({"ts": time.time(), "level": "error", "msg": f"Run failed: {reason}"})
        _fire(
            _patch_job,
            self._client,
            self._job_id,
            status=JobStatusEnum.FAILED,
            report_markdown=f"Run failed: {reason}",
            logs=list(self._logs),
        )
