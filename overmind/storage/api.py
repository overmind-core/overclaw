"""API storage backend for Overmind.

Maps every :class:`StorageBackend` operation to a call against the Overmind
REST API via the generated ``overmind.openapi_client`` SDK.  No hand-rolled
URLs — every endpoint used here comes from the OpenAPI schema.

Mapping
-------
==========  ==================================================================
Artifact    API surface (all from ``overmind.openapi_client``)
==========  ==================================================================
eval spec   ``AgentsApi.agents_create`` / ``agents_partial_update`` /
            ``agents_eval_spec_retrieve`` / ``agents_destroy``
dataset     ``DatasetsApi.datasets_create`` / ``datasets_destroy`` /
            ``datasets_datapoints_list``;
            ``AgentsApi.agents_retrieve`` to resolve ``active_dataset``
policy      ``AgentsApi.agents_partial_update`` —
            ``policy_markdown`` + ``policy_data`` fields on the Agent record
==========  ==================================================================
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any
from uuid import UUID

from overmind.client import (
    ProjectResolutionError,
    _fire,
    create_dataset,
    fetch_dataset_datapoints,
    get_active_dataset_id,
    get_client,
    resolve_project_id,
    upsert_agent,
)
from overmind.client import (
    delete_dataset as _delete_dataset_via_api,
)
from overmind.openapi_client.models.patched_agent_request import PatchedAgentRequest
from overmind.storage.base import StorageBackend

logger = logging.getLogger("overmind.storage.api")


class ApiBackend(StorageBackend):
    """Stores Overmind setup artifacts in the Overmind API.

    Parameters
    ----------
    agent_id:
        Overmind agent UUID string.  Updated in-place after ``save_spec``
        creates a new record.
    agent_path:
        Local path to the agent file — used for slug derivation in
        ``upsert_agent``.
    job_id:
        Overmind job UUID string (optional; can be set later via
        :meth:`set_job_id`).
    client:
        Pre-built :class:`OvermindClient`.  When ``None`` (default) the
        client is built lazily from ``OVERMIND_API_KEY``.
    """

    def __init__(
        self,
        agent_id: str,
        agent_path: str,
        *,
        agent_name: str | None = None,
        job_id: str | None = None,
        client: Any = None,
    ) -> None:
        self._agent_id = agent_id
        self._agent_path = agent_path
        self._agent_name = agent_name
        self._job_id = job_id
        self._client = client

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_agent_id(self) -> str | None:
        return self._agent_id or None

    def set_job_id(self, job_id: str) -> None:
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
        if self._client is not None:
            return self._client
        return get_client()

    def _project_id(self) -> str | None:
        """Return the active project id, auto-resolving from the API key when possible.

        Returns ``None`` (rather than raising) if resolution fails, so the
        higher-level ``save_*`` methods preserve their "best-effort, never
        crash the user's setup flow" contract.  The original error is logged.
        """
        client = self._client_()
        if client is None:
            return None
        try:
            return resolve_project_id(client)
        except ProjectResolutionError as exc:
            logger.warning(f"_project_id: could not resolve project ({exc})")
            return None

    def _patch_agent(self, **fields: Any) -> bool:
        """PATCH this agent record via ``agents_partial_update``.

        ``client.agents_partial_update`` is a **sync** SDK method — it
        returns an :class:`Agent` model directly. The previous code wrapped
        it in :func:`_run_async`, which silently swallowed a ``TypeError``
        and returned ``False`` even on successful PATCHes. Call it directly.
        """
        if not self._agent_id:
            return False
        client = self._client_()
        if not client:
            return False
        try:
            patch = PatchedAgentRequest(**fields)
            client.agents_partial_update(id=UUID(self._agent_id), patched_agent_request=patch)
            return True
        except Exception:
            logger.exception(f"agents_partial_update failed agent_id={self._agent_id}")
            return False

    # ------------------------------------------------------------------
    # Eval spec
    # ------------------------------------------------------------------

    def save_spec(self, spec: dict) -> None:
        """Upsert the agent record with eval-spec fields.

        First write is synchronous so the freshly minted agent UUID is captured
        in :attr:`agent_id`; later writes for an existing agent run in the
        background.
        """
        client = self._client_()
        if not client:
            return
        project_id = self._project_id()
        if not project_id:
            return
        if not self._agent_id:
            try:
                result = upsert_agent(
                    client,
                    project_id=project_id,
                    agent_path=self._agent_path,
                    spec=spec,
                    agent_name=self._agent_name,
                )
                self._agent_id = str(result.id)
            except Exception:
                logger.exception("save_spec: initial upsert_agent failed")
            return

        _submit_async_upsert(
            client,
            project_id=project_id,
            agent_path=self._agent_path,
            spec=spec,
            agent_name=self._agent_name,
        )

    def load_spec(self) -> dict | None:
        """Fetch eval-spec fields via ``agents_eval_spec_retrieve``.

        Both ``agents_eval_spec_retrieve`` and ``agents_retrieve`` are
        synchronous SDK methods. They were previously wrapped in
        :func:`_run_async`, which raised ``TypeError`` after the HTTP call
        had already succeeded, so this function always returned ``None``
        even though the data was available. Call directly.
        """
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            response = client.agents_eval_spec_retrieve(id=UUID(self._agent_id))
        except Exception:
            return None
        spec: dict[str, Any] = response.to_dict()
        # Augment with policy/agent metadata that's stored on the Agent record
        # but not exposed by the eval-spec endpoint.
        with contextlib.suppress(Exception):
            agent = client.agents_retrieve(id=UUID(self._agent_id))
            if agent.policy_data:
                spec["policy"] = agent.policy_data
        return spec

    def delete_spec(self) -> None:
        """Destroy the agent record via ``agents_destroy``.

        ``agents_destroy`` is synchronous. Local state (``_agent_id``) is
        cleared **after** the remote DELETE succeeds so a transient API
        failure doesn't leave the storage backend pointing at a stale id.
        """
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        try:
            client.agents_destroy(id=UUID(self._agent_id))
        except Exception:
            logger.exception("delete_spec: agents_destroy failed")
            return
        self._agent_id = ""

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        datapoints: list[dict],
        *,
        source: str = "synthetic",
        generator_model: str = "",
        policy_hash: str = "",
        metadata: dict | None = None,
        make_active: bool = True,
    ) -> dict | None:
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            created = create_dataset(
                client,
                agent_id=self._agent_id,
                datapoints=datapoints,
                source=source,
                generator_model=generator_model,
                policy_hash=policy_hash,
                metadata=metadata,
                make_active=make_active,
            )
        except Exception:
            return None
        if not created or not isinstance(created, dict):
            return None
        return {
            "id": created.get("id"),
            "version": created.get("version"),
            "source": created.get("source") or source,
            "num_datapoints": created.get("num_datapoints") or len(datapoints),
            "agent_id": self._agent_id,
            "generator_model": created.get("generator_model") or generator_model,
        }

    def load_dataset(self) -> list[dict] | None:
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        dataset_id = get_active_dataset_id(client, self._agent_id)
        if not dataset_id:
            return None
        try:
            return fetch_dataset_datapoints(client, dataset_id)
        except Exception:
            return None

    def delete_dataset(self) -> None:
        if not self._agent_id:
            return
        client = self._client_()
        if not client:
            return
        dataset_id = get_active_dataset_id(client, self._agent_id)
        if not dataset_id:
            return
        with contextlib.suppress(Exception):
            _delete_dataset_via_api(client, dataset_id)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def save_policy(self, policy_md: str, policy_data: dict | None = None) -> None:
        """Patch ``Agent.policy_markdown`` (and ``policy_data`` when provided)."""
        fields: dict[str, Any] = {"policy_markdown": policy_md}
        if policy_data is not None:
            fields["policy_data"] = policy_data
        self._patch_agent(**fields)

    def load_policy(self) -> str | None:
        if not self._agent_id:
            return None
        client = self._client_()
        if not client:
            return None
        try:
            agent = client.agents_retrieve(id=UUID(self._agent_id))
        except Exception:
            return None
        return getattr(agent, "policy_markdown", None) or None

    def delete_policy(self) -> None:
        """Clear ``policy_markdown`` and ``policy_data`` on the agent record."""
        self._patch_agent(policy_markdown=None, policy_data=None)


def _submit_async_upsert(
    client: Any,
    *,
    project_id: str,
    agent_path: str,
    spec: dict,
    agent_name: str | None = None,
) -> None:
    """Run ``upsert_agent`` on a background thread (fire-and-forget)."""
    _fire(
        upsert_agent,
        client,
        project_id=project_id,
        agent_path=agent_path,
        spec=spec,
        agent_name=agent_name,
    )
