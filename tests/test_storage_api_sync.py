"""Tests for ``overmind.storage.api.ApiBackend`` sync-call behaviour.

The generated OpenAPI SDK is synchronous — ``agents_partial_update``,
``agents_eval_spec_retrieve``, ``agents_retrieve``, and ``agents_destroy``
all return decoded model objects directly. Earlier code wrapped them in
``_run_async``, which silently swallowed a ``TypeError`` *after* the HTTP
call had already succeeded. The result was "remote write landed, local
state thinks it failed" — the worst debug surface.

These tests pin the post-fix contract:

* Every public ``ApiBackend`` method that touches an SDK call invokes the
  SDK method *directly* (no ``_run_async`` bridge).
* ``delete_spec`` clears ``_agent_id`` only on success — a transient API
  failure must not leave the backend pointing at a stale id while the
  remote record still exists.
* A standing regression guard asserts ``_run_async`` is never called from
  any of these paths, so reintroducing the wrapper would fail this suite
  before it could ship to production.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from overmind import client as client_mod
from overmind.storage.api import ApiBackend

AGENT_UUID = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def backend() -> tuple[ApiBackend, MagicMock]:
    """Return ``(backend, client_mock)`` wired up so SDK calls are observable."""
    client = MagicMock()
    be = ApiBackend(
        agent_id=AGENT_UUID,
        agent_path="entry.py",
        agent_name="test-agent",
        client=client,
    )
    return be, client


# ---------------------------------------------------------------------------
# _patch_agent — used by save_policy / delete_policy / partial-update save_spec
# ---------------------------------------------------------------------------


class TestPatchAgentSync:
    def test_save_policy_calls_partial_update_directly(self, backend):
        be, client = backend
        be.save_policy("# new policy", policy_data={"flags": ["x"]})

        client.agents_partial_update.assert_called_once()
        kwargs = client.agents_partial_update.call_args.kwargs
        assert kwargs["id"] == UUID(AGENT_UUID)
        patch_req = kwargs["patched_agent_request"]
        # Attribute access works because PatchedAgentRequest is a Pydantic
        # BaseModel; this survives any to_dict() implementation drift.
        assert patch_req.policy_markdown == "# new policy"
        assert patch_req.policy_data == {"flags": ["x"]}

    def test_delete_policy_clears_via_partial_update(self, backend):
        be, client = backend
        be.delete_policy()

        client.agents_partial_update.assert_called_once()
        kwargs = client.agents_partial_update.call_args.kwargs
        patch_req = kwargs["patched_agent_request"]
        # The fix sends explicit None values to clear the fields.
        assert patch_req.policy_markdown is None
        assert patch_req.policy_data is None

    def test_patch_agent_returns_silently_on_no_agent_id(self):
        client = MagicMock()
        be = ApiBackend(agent_id="", agent_path="entry.py", client=client)
        be.save_policy("# policy")
        client.agents_partial_update.assert_not_called()


# ---------------------------------------------------------------------------
# load_spec — agents_eval_spec_retrieve + agents_retrieve augmentation
# ---------------------------------------------------------------------------


class TestLoadSpecSync:
    def test_returns_spec_dict_from_sync_call(self, backend):
        be, client = backend
        client.agents_eval_spec_retrieve.return_value = SimpleNamespace(
            to_dict=lambda: {"agent_description": "x", "input_schema": {}}
        )
        client.agents_retrieve.return_value = SimpleNamespace(policy_data=None)

        spec = be.load_spec()

        assert spec == {"agent_description": "x", "input_schema": {}}
        client.agents_eval_spec_retrieve.assert_called_once_with(id=UUID(AGENT_UUID))

    def test_augments_with_policy_data_when_present(self, backend):
        be, client = backend
        client.agents_eval_spec_retrieve.return_value = SimpleNamespace(
            to_dict=lambda: {"agent_description": "x"}
        )
        client.agents_retrieve.return_value = SimpleNamespace(
            policy_data={"version": 3, "rules": ["a", "b"]}
        )

        spec = be.load_spec()

        assert spec["policy"] == {"version": 3, "rules": ["a", "b"]}
        client.agents_retrieve.assert_called_once_with(id=UUID(AGENT_UUID))

    def test_returns_none_on_eval_spec_failure(self, backend):
        be, client = backend
        client.agents_eval_spec_retrieve.side_effect = RuntimeError("404")
        assert be.load_spec() is None


# ---------------------------------------------------------------------------
# delete_spec — state-clear ordering matters
# ---------------------------------------------------------------------------


class TestDeleteSpecSync:
    def test_clears_agent_id_only_on_success(self, backend):
        be, client = backend
        assert be.agent_id == AGENT_UUID

        be.delete_spec()

        client.agents_destroy.assert_called_once_with(id=UUID(AGENT_UUID))
        assert be.agent_id == "", "Successful destroy must clear local state"

    def test_keeps_agent_id_on_failure(self, backend):
        be, client = backend
        client.agents_destroy.side_effect = RuntimeError("backend exploded")

        be.delete_spec()  # logs, doesn't raise

        client.agents_destroy.assert_called_once_with(id=UUID(AGENT_UUID))
        assert be.agent_id == AGENT_UUID, (
            "Failed destroy must NOT clear local state — otherwise the next "
            "save_spec would orphan a record on the backend"
        )


# ---------------------------------------------------------------------------
# load_policy
# ---------------------------------------------------------------------------


class TestLoadPolicySync:
    def test_returns_markdown_from_sync_call(self, backend):
        be, client = backend
        client.agents_retrieve.return_value = SimpleNamespace(
            policy_markdown="# loaded policy"
        )
        assert be.load_policy() == "# loaded policy"
        client.agents_retrieve.assert_called_once_with(id=UUID(AGENT_UUID))

    def test_returns_none_when_missing(self, backend):
        be, client = backend
        client.agents_retrieve.return_value = SimpleNamespace(policy_markdown=None)
        assert be.load_policy() is None

    def test_returns_none_on_api_error(self, backend):
        be, client = backend
        client.agents_retrieve.side_effect = RuntimeError("500")
        assert be.load_policy() is None


# ---------------------------------------------------------------------------
# Cross-cutting regression guard — the asyncio bridge stays deleted.
# ---------------------------------------------------------------------------


class TestNoAsyncBridge:
    """Standing guard: the asyncio bridge functions stay deleted.

    Bug 6 was a hand-rolled ``_run_async`` that wrapped sync SDK calls;
    the wrapper raised ``TypeError`` *after* the HTTP request succeeded
    and every caller's bare ``except Exception:`` reported failure to
    the user. The wrapper was deleted in the client.py cleanup, so this
    test only needs to assert it stays gone.
    """

    def test_run_async_does_not_exist(self):
        assert not hasattr(client_mod, "_run_async"), (
            "_run_async re-introduced — the generated SDK is sync. "
            "Wrap fire-and-forget writes in client._fire instead."
        )

    def test_submit_async_does_not_exist(self):
        assert not hasattr(client_mod, "_submit_async"), (
            "_submit_async re-introduced — the generated SDK is sync. "
            "Wrap fire-and-forget writes in client._fire instead."
        )

    def test_get_bg_loop_does_not_exist(self):
        assert not hasattr(client_mod, "_get_bg_loop"), (
            "_get_bg_loop re-introduced — the asyncio bridge is gone."
        )

    def test_apibackend_still_works_after_cleanup(self, backend):
        """Sanity check that ApiBackend operations still work after the cleanup."""
        be, _client = backend
        # These calls used to be the regression-test surface for the async bridge.
        # Now they just verify ApiBackend's public methods don't crash.
        be.delete_spec()
        be.save_policy("# md")
        be.delete_policy()
