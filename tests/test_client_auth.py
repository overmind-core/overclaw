"""Tests for overmind.client auth routing and agent upsert.

Covers:

* :func:`get_client` — project-scoped ``ovr_…`` keys flow through
  ``ApiKeyAuth`` *only*; legacy JWTs/dev tokens stay on the ``Bearer`` path.
* :func:`upsert_agent` — works against a key with a single accessible
  project, with no ``OVERMIND_PROJECT_ID`` env-var required.  The project
  is inferred server-side from the API key; the SDK only needs to attach
  a UUID to the create payload.
* Historical safety: the project-id resolution surface (public function,
  env-var override, in-process cache, asyncio bridge) is fully gone and
  cannot be re-introduced silently.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from overmind import client as client_mod
from overmind.client import get_client, upsert_agent

# ---------------------------------------------------------------------------
# get_client — auth routing
# ---------------------------------------------------------------------------


class TestGetClientAuthRouting:
    """``ovr_…`` keys must use ApiKeyAuth only; non-``ovr_…`` tokens use Bearer."""

    def test_returns_none_when_api_key_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
        assert get_client() is None

    def test_ovr_key_uses_api_key_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_i97_test_abc123")

        client = get_client()
        assert client is not None

        cfg = client.api_client.configuration
        # ApiKeyAuth dict registered, BearerAuth NOT activated.
        assert cfg.api_key == {"ApiKeyAuth": "ovr_i97_test_abc123"}
        assert not cfg.access_token, "Bearer path must stay off for ovr_* keys"

        # And the SDK's auth_settings should emit X-Api-Key, not Authorization.
        settings = cfg.auth_settings()
        assert "ApiKeyAuth" in settings
        assert settings["ApiKeyAuth"]["in"] == "header"
        assert settings["ApiKeyAuth"]["key"] == "X-Api-Key"
        assert settings["ApiKeyAuth"]["value"] == "ovr_i97_test_abc123"
        assert "BearerAuth" not in settings

    def test_non_ovr_token_uses_bearer(self, monkeypatch: pytest.MonkeyPatch):
        # JWT-shaped token must keep the legacy Bearer path so existing
        # user-token flows / Clerk JWTs don't break.
        monkeypatch.setenv("OVERMIND_API_KEY", "eyJhbGci.payload.sig")

        client = get_client()
        assert client is not None

        cfg = client.api_client.configuration
        assert cfg.access_token == "eyJhbGci.payload.sig"
        # api_key must NOT contain the JWT under ApiKeyAuth — otherwise the
        # backend would still see the double-stamp footgun.
        assert not isinstance(cfg.api_key, dict) or "ApiKeyAuth" not in cfg.api_key

        settings = cfg.auth_settings()
        assert (
            "BearerAuth" in settings
            or "jwtAuth" in settings
            or any(s.get("type") == "bearer" for s in settings.values())
        )

    def test_uses_default_base_url(self, monkeypatch: pytest.MonkeyPatch):
        from overmind.core.constants import DEFAULT_BASE_URL

        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_test")

        client = get_client()
        assert client is not None
        assert client.api_client.configuration.host == DEFAULT_BASE_URL.rstrip("/")

    def test_works_without_overmind_project_id(self, monkeypatch: pytest.MonkeyPatch):
        """``OVERMIND_PROJECT_ID`` was removed; setting it must be a no-op."""
        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_test")
        monkeypatch.setenv("OVERMIND_PROJECT_ID", "should-be-ignored")

        client = get_client()
        assert client is not None
        # No project-id state is read from env or attached to the client.
        assert not hasattr(client_mod, "OVERMIND_PROJECT_ID")


# ---------------------------------------------------------------------------
# upsert_agent — implicit project resolution
# ---------------------------------------------------------------------------


class _FakeProject:
    def __init__(self, pid: str, slug: str = "") -> None:
        self.id = pid
        self.slug = slug
        self.name = slug or pid


class _FakeAgent:
    def __init__(self, agent_id: str, slug: str, agent_path: str) -> None:
        self.id = agent_id
        self.slug = slug
        self.agent_path = agent_path


def _page(*items: object) -> SimpleNamespace:
    return SimpleNamespace(results=list(items))


@pytest.fixture(autouse=True)
def _clear_project_cache():
    """Each test starts with a clean module-level project-uuid cache."""
    client_mod._cached_project_uuid = None
    yield
    client_mod._cached_project_uuid = None


class TestUpsertAgentProjectResolution:
    """``upsert_agent`` must work without any explicit project plumbing."""

    def test_create_resolves_project_from_api_key(self):
        client = MagicMock()
        client.agents_list.return_value = _page()
        client.projects_list.return_value = _page(_FakeProject("11111111-1111-1111-1111-111111111111", "demo"))
        client.agents_create.return_value = SimpleNamespace(id="agent-id")

        result = upsert_agent(
            client,
            agent_path="/tmp/agent.py",
            spec={"agent_description": "demo"},
            agent_name="demo-agent",
        )

        assert result.id == "agent-id"
        # projects_list is consulted exactly once for the create payload's UUID.
        client.projects_list.assert_called_once_with(page_size=1)

    def test_update_skips_project_lookup(self):
        client = MagicMock()
        client.agents_list.return_value = _page(
            _FakeAgent("existing-id", "demo-agent", "/tmp/agent.py"),
        )
        client.agents_partial_update.return_value = SimpleNamespace(id="existing-id")

        upsert_agent(
            client,
            agent_path="/tmp/agent.py",
            spec={"agent_description": "demo"},
            agent_name="demo-agent",
        )

        # Updates never need the project UUID — the existing agent is already scoped.
        client.projects_list.assert_not_called()
        client.agents_partial_update.assert_called_once()

    def test_create_raises_when_no_project_accessible(self):
        client = MagicMock()
        client.agents_list.return_value = _page()
        client.projects_list.return_value = _page()  # zero projects

        with pytest.raises(RuntimeError, match="project UUID"):
            upsert_agent(
                client,
                agent_path="/tmp/agent.py",
                spec={"agent_description": "demo"},
                agent_name="demo-agent",
            )

    def test_project_uuid_is_cached(self):
        client = MagicMock()
        client.agents_list.return_value = _page()
        client.projects_list.return_value = _page(
            _FakeProject("22222222-2222-2222-2222-222222222222", "solo"),
        )
        client.agents_create.return_value = SimpleNamespace(id="agent-id-1")

        upsert_agent(client, agent_path="/tmp/a.py", spec={}, agent_name="a")

        # Reset agents_list/agents_create to simulate a second, distinct create.
        client.agents_list.return_value = _page()
        client.agents_create.return_value = SimpleNamespace(id="agent-id-2")
        upsert_agent(client, agent_path="/tmp/b.py", spec={}, agent_name="b")

        assert client.projects_list.call_count == 1


# ---------------------------------------------------------------------------
# Historical safety: deleted surface must stay deleted
# ---------------------------------------------------------------------------


class TestDeletedProjectIdSurface:
    """The public project-id resolution surface was removed in the cleanup.

    If anyone ever re-introduces ``resolve_project_id``, the
    ``_resolved_project_id`` cache, ``_reset_project_id_cache``,
    ``ProjectResolutionError``, or the asyncio bridge functions, these
    tests fail fast so the old bugs (silent failures, multi-project
    ambiguity, TypeError-after-success) cannot come back.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "resolve_project_id",
            "_resolved_project_id",
            "_reset_project_id_cache",
            "ProjectResolutionError",
            "_run_async",
            "_submit_async",
        ],
    )
    def test_symbol_is_gone(self, name: str) -> None:
        assert not hasattr(client_mod, name), (
            f"{name} re-introduced — see Phase 3 of the cleanup plan for context."
        )
