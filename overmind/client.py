"""Overmind API client.

A thin, synchronous-feeling facade over the generated
``overmind.openapi_client`` SDK.  Configure via the standard environment
variable:

* ``OVERMIND_API_KEY``  A project-scoped key (``ovr_…``) is recommended; a user
                        JWT is also accepted for legacy flows.  The backend URL
                        defaults to the Overmind Cloud endpoint and the project
                        is inferred server-side from the key — there is no
                        client-side project disambiguation.

Concurrency model
-----------------
The generated OpenAPI SDK is **synchronous** — every API method returns
a decoded model object directly.  Long-running CLI flows still need to
avoid blocking on network I/O, so fire-and-forget API writes are sent
through :func:`_fire`, which submits the call to a small background
thread pool.  :func:`flush_pending_api_updates` lets the caller wait for
in-flight writes to drain at shutdown.

Do **not** wrap SDK calls in asyncio primitives — they are not
coroutines.  Past attempts to do so silently swallowed ``TypeError``
after the HTTP call had already succeeded.

Example::

    from overmind.client import get_client

    client = get_client()
    agent = client.resolve_agent("support-triage")
    dataset = client.upsert_dataset(agent.id, datapoints=[...], source="seed")
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from uuid import UUID

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
from overmind.openapi_client.models.job_status_enum import JobStatusEnum
from overmind.openapi_client.models.patched_agent_request import PatchedAgentRequest
from overmind.openapi_client.models.patched_job_request import PatchedJobRequest
from overmind.openapi_client.models.source_enum import SourceEnum

logger = logging.getLogger("overmind.client")

# ---------------------------------------------------------------------------
# Background execution: small thread pool for fire-and-forget API writes.
# ---------------------------------------------------------------------------
# Daemon threads so background writes never block process exit.  The
# optimizer / setup loops dispatch every PATCH/POST through :func:`_fire`
# and call :func:`flush_pending_api_updates` at shutdown to drain in-flight
# requests.

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="overmind-api")
_pending_futures: set[Future] = set()
_pending_lock = threading.Lock()


def _track_future(fut: Future) -> Future:
    """Track a background future so pending writes can be flushed on demand."""
    with _pending_lock:
        _pending_futures.add(fut)

    def _done(_f: Future) -> None:
        with _pending_lock:
            _pending_futures.discard(_f)

    fut.add_done_callback(_done)
    return fut


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

    The SDK is sync; do not wrap calls in any asyncio primitive.  Fire
    background writes through :func:`_fire` instead.

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

    Always connects to :data:`overmind.core.constants.DEFAULT_BASE_URL`
    (the Overmind Cloud endpoint).
    """
    base_url = DEFAULT_BASE_URL.rstrip("/")
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
    """Return True if ``OVERMIND_API_KEY`` is set."""
    return bool(os.getenv("OVERMIND_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Project resolution (implementation detail of upsert_agent)
# ---------------------------------------------------------------------------
# The generated ``AgentRequest`` schema requires a ``project: UUID`` field, so
# even though the backend infers the project from the API key for *every other*
# endpoint, the SDK side still has to attach a UUID to agent-creation payloads.
# We do a one-shot ``projects_list(page_size=1)`` lookup and cache the result.
#
# This is *not* a public API: callers should never have to think about project
# ids.  If the backend ever drops the schema-level ``project`` requirement, the
# helper and its single call site can disappear.

_cached_project_uuid: str | None = None


def _resolve_project_uuid(client: OvermindClient) -> str | None:
    """Best-effort lookup of the project UUID accessible to this API key.

    Returns ``None`` when the call fails or no project is visible — callers
    are expected to treat that as a soft failure (skip the remote write).
    Cached in-process so we pay the network cost at most once.
    """
    global _cached_project_uuid
    if _cached_project_uuid:
        return _cached_project_uuid
    try:
        page = client.projects_list(page_size=1)
    except Exception:
        logger.debug("_resolve_project_uuid: projects_list failed", exc_info=True)
        return None
    results = list(page.results or [])
    if not results:
        return None
    _cached_project_uuid = str(results[0].id)
    return _cached_project_uuid


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

    The project is inferred server-side from the API key; the SDK only needs
    to attach a UUID to the create payload (see :func:`_resolve_project_uuid`).

    Returns the Agent object (typed model from the OpenAPI client).
    """
    if agent_name:
        slug = _make_slug(agent_name)
    else:
        slug = agent_slug_from_path(agent_path)
    description = (spec.get("agent_description") or "")[:512] or None
    name = (description or agent_name or slug)[:255]

    logger.info(f"upsert_agent: slug={slug}")

    existing = None
    try:
        page = client.agents_list()
        for ag in page.results or []:
            if ag.slug == slug or ag.agent_path == agent_path:
                existing = ag
                break
    except Exception:
        logger.warning("upsert_agent: agents_list failed", exc_info=True)

    shared = _agent_shared_fields(spec, agent_path)

    if existing:
        patch = PatchedAgentRequest(**shared)
        result = client.agents_partial_update(id=existing.id, patched_agent_request=patch)
        logger.info(f"upsert_agent: updated existing agent id={existing.id} slug={slug}")
        return result

    project_uuid = _resolve_project_uuid(client)
    if not project_uuid:
        raise RuntimeError(
            "upsert_agent: could not resolve a project UUID for the configured "
            "OVERMIND_API_KEY. Generate a project-scoped key from the Overmind "
            "console and try again."
        )
    req = AgentRequest(name=name, slug=slug, project=UUID(project_uuid), **shared)
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
# Job patch helper (used by the surviving ApiReporter terminal stamps).
# ---------------------------------------------------------------------------


def _patch_job(client: OvermindClient, job_id: str, **fields: Any) -> None:
    try:
        patch = PatchedJobRequest(**fields)
        client.jobs_partial_update(id=UUID(job_id), patched_job_request=patch)
        logger.debug(f"_patch_job: job_id={job_id} fields={list(fields)}")
    except Exception:
        logger.exception(f"_patch_job: failed job_id={job_id}")


# ---------------------------------------------------------------------------
# ApiReporter — terminal-state stamps for an OTLP-ingested Job.
# ---------------------------------------------------------------------------


class ApiReporter:
    """Stamps the final ``best_agent_code`` / ``report_markdown`` on a Job.

    All in-flight telemetry (per-span attributes, tool calls, LLM usage,
    iteration analytics) is emitted by the OpenTelemetry pipeline and
    parsed by ``overbae.api.otlp`` — the OTLP root span creates the
    ``Job`` row from a locally-minted UUID, and every subsequent span
    carrying the same :data:`overmind.attrs.JOB_ID` is bound to it.

    This reporter only owns the **two terminal writes** the OTLP path
    cannot model cleanly: a final PATCH stamping the Job as completed
    (with the report markdown + best agent code) or failed.  Both writes
    go through :func:`_fire` so the optimizer loop never blocks on the
    network at shutdown.
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
    def attach_to_job(
        cls,
        agent_id: str,
        job_id: str,
    ) -> ApiReporter | None:
        """Build a reporter for a Job whose UUID was minted locally.

        The caller tags the workflow root span with
        ``attrs.JOB_ID = job_id`` — the backend's OTLP ingest then
        creates the ``Job`` row from the span (project inherited from
        the OTel resource) and binds every subsequent span carrying the
        same ``JOB_ID`` to it.  ``baseline_score``, ``current_iteration``,
        and ``JobIteration`` rows populate from span attributes — not
        from REST writes.

        Returns ``None`` when the API is not configured.
        """
        client = get_client()
        if not client or not agent_id or not job_id:
            return None
        return cls(client, agent_id, job_id)

    @property
    def job_id(self) -> str:
        return self._job_id

    def on_complete(
        self,
        best_score: float,
        baseline_score: float,
        report_markdown: str | None = None,
        best_agent_code: str | None = None,
        backtest_results: dict | None = None,
    ) -> None:
        """Stamp the Job as completed."""
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

    def on_failed(self, reason: str = "") -> None:
        """Stamp the Job as failed."""
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
