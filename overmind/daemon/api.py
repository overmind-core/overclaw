"""HTTP transport for the optimization daemon.

A small ``requests`` wrapper over the backend's ``cli/*`` and ``optimize-runs``
endpoints. These payloads are plain JSON, so — per the project's "use what's
already here, minimum code" rule — we talk to them directly with the
already-vendored ``requests`` dependency instead of regenerating the heavyweight
OpenAPI SDK just for a handful of endpoints.
``ponytail``: if these endpoints grow rich/typed request bodies, regenerate the
SDK (``make generate_python_client``) and swap this module for SDK calls.
"""

from __future__ import annotations

import os

import requests


def resolve_base_url() -> str:
    return os.getenv("OVERMIND_API_URL", "http://api.overmind-dev.orb.local:8000").rstrip("/")


def resolve_token() -> str:
    return os.getenv("OVERMIND_API_KEY", "").strip()


class DaemonAPI:
    """Synchronous client for the daemon's control-plane calls."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 600.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self._auth_headers(token))

    @staticmethod
    def _auth_headers(token: str) -> dict[str, str]:
        # ovr_* project keys route through the key-auth middleware via X-Api-Key;
        # anything else is treated as a user JWT on the Bearer path.
        if token.startswith("ovr_"):
            return {"X-Api-Key": token}
        return {"Authorization": f"Bearer {token}"}

    def _post(self, path: str, body: dict) -> dict:
        resp = self._session.post(f"{self._base}{path}", json=body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get(self, path: str) -> dict:
        resp = self._session.get(f"{self._base}{path}", timeout=self._timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── sessions ────────────────────────────────────────────────────────
    def register_session(self, *, agent_name: str = "", cli_version: str = "", hostname: str = "", metadata: dict | None = None) -> dict:
        return self._post(
            "/api/cli/sessions/",
            {
                "agent_name": agent_name,
                "cli_version": cli_version,
                "hostname": hostname,
                "metadata": metadata or {},
            },
        )

    def poll(self, session_id: str, *, agent_name: str = "", cli_version: str = "") -> dict:
        return self._post(
            f"/api/cli/sessions/{session_id}/poll/",
            {"agent_name": agent_name, "cli_version": cli_version},
        )

    # ── commands ────────────────────────────────────────────────────────
    def submit_result(self, command_id: str, *, success: bool, result: dict | None = None, error: str = "") -> dict:
        return self._post(
            f"/api/cli/commands/{command_id}/result/",
            {"success": success, "result": result or {}, "error": error},
        )

    # ── runs ────────────────────────────────────────────────────────────
    def start_run(self, *, agent_id: str, client_session: str | None = None, **config) -> dict:
        body = {"agent": agent_id, **{k: v for k, v in config.items() if v is not None}}
        if client_session:
            body["client_session"] = client_session
        return self._post("/api/optimize-runs/", body)

    def get_run(self, run_id: str) -> dict:
        return self._get(f"/api/optimize-runs/{run_id}/")

    def cancel_run(self, run_id: str) -> dict:
        return self._post(f"/api/optimize-runs/{run_id}/cancel/", {})

    def respond_run(self, run_id: str, *, approved: bool, payload: dict | None = None) -> dict:
        return self._post(
            f"/api/optimize-runs/{run_id}/user-response/",
            {"approved": approved, "payload": payload or {}},
        )

    def close(self) -> None:
        self._session.close()
