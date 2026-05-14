"""Tests for overmind.client auth routing & project-id resolution.

Covers:

* :func:`get_client` — project-scoped ``ovr_…`` keys flow through
  ``ApiKeyAuth`` *only*; legacy JWTs/dev tokens stay on the ``Bearer`` path.
* :func:`resolve_project_id` — env wins, falls back to a single-project
  ``projects_list()`` lookup, and raises :class:`ProjectResolutionError`
  on ambiguity / empty results / outright failure.

These tests guard the fix for the "OVERMIND_PROJECT_ID feels mandatory
even with a project-scoped key" regression: the old client double-stamped
both auth schemes, the backend preferred Bearer, and the user-JWT path
exposed every project the user could see.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from overmind import client as client_mod
from overmind.client import (
    ProjectResolutionError,
    _reset_project_id_cache,
    get_client,
    get_project_id,
    resolve_project_id,
)


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
        monkeypatch.setenv("OVERMIND_API_URL", "https://api.example.com")

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
        monkeypatch.setenv("OVERMIND_API_URL", "https://api.example.com")

        client = get_client()
        assert client is not None

        cfg = client.api_client.configuration
        assert cfg.access_token == "eyJhbGci.payload.sig"
        # api_key must NOT contain the JWT under ApiKeyAuth — otherwise the
        # backend would still see the double-stamp footgun.
        assert not isinstance(cfg.api_key, dict) or "ApiKeyAuth" not in cfg.api_key

        settings = cfg.auth_settings()
        assert "BearerAuth" in settings or "jwtAuth" in settings or any(
            s.get("type") == "bearer" for s in settings.values()
        )

    def test_falls_back_to_default_base_url(self, monkeypatch: pytest.MonkeyPatch):
        from overmind.core.constants import DEFAULT_BASE_URL

        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_test")
        monkeypatch.delenv("OVERMIND_API_URL", raising=False)

        client = get_client()
        assert client is not None
        assert client.api_client.configuration.host == DEFAULT_BASE_URL.rstrip("/")


# ---------------------------------------------------------------------------
# resolve_project_id
# ---------------------------------------------------------------------------


class _FakeProject:
    def __init__(self, pid: str, slug: str = "") -> None:
        self.id = pid
        self.slug = slug
        self.name = slug or pid


def _page(*projects: _FakeProject) -> SimpleNamespace:
    return SimpleNamespace(results=list(projects))


@pytest.fixture(autouse=True)
def _clear_project_cache():
    """Each test starts with a clean module-level cache."""
    _reset_project_id_cache()
    yield
    _reset_project_id_cache()


class TestResolveProjectId:
    """The generated SDK is synchronous; ``resolve_project_id`` must call
    ``client.projects_list(...)`` directly, never through ``_run_async``.

    These tests stub the SDK method itself rather than the ``_run_async``
    bridge — the old pattern silently bypassed the real production path
    and masked a TypeError that fired after the HTTP call had already
    succeeded. The ``test_does_not_wrap_in_run_async`` case below is the
    standing regression guard.
    """

    def test_env_var_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_PROJECT_ID", "11111111-1111-1111-1111-111111111111")
        client = MagicMock()
        client.projects_list.side_effect = AssertionError("API should not be touched")
        assert resolve_project_id(client) == "11111111-1111-1111-1111-111111111111"
        client.projects_list.assert_not_called()

    def test_single_project_auto_discovered(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_test")

        client = MagicMock()
        client.projects_list.return_value = _page(_FakeProject("proj-uuid-1", "demo"))

        pid = resolve_project_id(client)

        assert pid == "proj-uuid-1"
        client.projects_list.assert_called_once_with(page_size=2)
        # Cache the resolution into the env var for downstream callers.
        assert get_project_id() == "proj-uuid-1"

    def test_multi_project_raises_with_candidates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)

        client = MagicMock()
        client.projects_list.return_value = _page(
            _FakeProject("a-uuid", "alpha"),
            _FakeProject("b-uuid", "beta"),
        )

        with pytest.raises(ProjectResolutionError) as exc_info:
            resolve_project_id(client)

        candidates = exc_info.value.candidates
        assert ("a-uuid", "alpha") in candidates
        assert ("b-uuid", "beta") in candidates
        assert "OVERMIND_PROJECT_ID" in str(exc_info.value)

    def test_no_projects_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        client = MagicMock()
        client.projects_list.return_value = _page()
        with pytest.raises(ProjectResolutionError, match="no accessible projects"):
            resolve_project_id(client)

    def test_api_error_wrapped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        client = MagicMock()
        client.projects_list.side_effect = RuntimeError("boom")
        with pytest.raises(ProjectResolutionError, match="projects_list"):
            resolve_project_id(client)

    def test_no_client_no_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
        with pytest.raises(ProjectResolutionError, match="OVERMIND_API_KEY"):
            resolve_project_id(None)

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        """Repeated calls must not re-hit the API once a project is locked in."""
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_cache_test")

        client = MagicMock()
        client.projects_list.return_value = _page(_FakeProject("only-one", "solo"))

        assert resolve_project_id(client) == "only-one"
        assert resolve_project_id(client) == "only-one"

        assert client.projects_list.call_count == 1, "Second call must use the cache"

    def test_does_not_wrap_in_run_async(self, monkeypatch: pytest.MonkeyPatch):
        """Regression guard: ``resolve_project_id`` must call ``projects_list``
        directly. The generated SDK is sync — wrapping it in ``_run_async``
        previously raised ``TypeError`` *after* the HTTP call already succeeded,
        and the caller re-wrapped that as
        ``ProjectResolutionError("projects_list call failed")``.

        We stub ``_run_async`` itself with a sentinel that fails loudly so
        any reintroduced wrapper would trip this test before shipping.
        """
        monkeypatch.delenv("OVERMIND_PROJECT_ID", raising=False)
        monkeypatch.setenv("OVERMIND_API_KEY", "ovr_test")

        client = MagicMock()
        client.projects_list.return_value = _page(_FakeProject("uuid", "solo"))

        sentinel = MagicMock(side_effect=AssertionError(
            "resolve_project_id should call projects_list directly — the "
            "generated SDK is synchronous. Do not wrap it in _run_async."
        ))
        with patch.object(client_mod, "_run_async", sentinel):
            assert resolve_project_id(client) == "uuid"
        sentinel.assert_not_called()
