"""JSON-RPC client for the Overmind platform MCP endpoint."""

from __future__ import annotations

import json
from typing import Any

import requests

from overmind.platform.types import ToolCallResult, ToolDetail, ToolSummary, infer_tool_domain
from overmind.tracing import get_api_settings

DEFAULT_TIMEOUT = 120


class PlatformError(Exception):
    """Raised when the platform MCP endpoint returns an error."""


class PlatformClient:
    """Proxy for ``POST /api/mcp/`` — list, describe, and call platform tools."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        key, url = get_api_settings(api_key, base_url)
        self._mcp_url = f"{url}/api/mcp/"
        self._timeout = timeout
        self._request_id = 0
        self._session = session or requests.Session()
        if session is None:
            self._session.headers.update({
                "X-Api-Key": key,
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            })

    def list_tools(self) -> list[ToolSummary]:
        tools = self._rpc("tools/list").get("tools", [])
        return [
            ToolSummary(
                name=tool["name"],
                description=(tool.get("description") or "").strip(),
                domain=infer_tool_domain(tool["name"]),
            )
            for tool in tools
        ]

    def describe_tool(self, name: str) -> ToolDetail:
        for tool in self._rpc("tools/list").get("tools", []):
            if tool.get("name") == name:
                return ToolDetail(
                    name=name,
                    description=(tool.get("description") or "").strip(),
                    input_schema=tool.get("inputSchema") or {},
                    domain=infer_tool_domain(name),
                )
        raise PlatformError(f"Tool not found: {name}")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        return ToolCallResult(
            content=result.get("content") or [],
            is_error=bool(result.get("isError")),
            structured_content=result.get("structuredContent"),
        )

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        try:
            resp = self._session.post(self._mcp_url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            raise PlatformError(f"Request failed: {exc}") from exc
        data = _parse_mcp_response(resp)
        if "error" in data:
            err = data["error"]
            msg = err.get("message") or str(err)
            raise PlatformError(msg)
        return data.get("result") or {}


def _parse_mcp_response(resp: requests.Response) -> dict[str, Any]:
    if not resp.ok:
        try:
            detail = resp.json()
            msg = detail.get("error", {}).get("message") or detail.get("detail") or resp.text[:400]
        except Exception:
            msg = resp.text[:400]
        raise PlatformError(f"HTTP {resp.status_code}: {msg}")

    content_type = resp.headers.get("Content-Type", "")
    text = resp.text
    if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
        return _parse_sse_payload(text)
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise PlatformError(f"Invalid JSON response: {text[:200]}") from exc


def _parse_sse_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "result" in data or "error" in data:
            return data
    raise PlatformError("No JSON-RPC payload in SSE response")

