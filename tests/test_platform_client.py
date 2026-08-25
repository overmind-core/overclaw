"""Tests for overmind.platform.client — JSON-RPC MCP proxy."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

from overmind.platform.client import PlatformClient, PlatformError, _parse_mcp_response, _parse_sse_payload
from overmind.platform.types import infer_tool_domain


def _mock_response(
    *,
    ok: bool = True,
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
    content_type: str = "application/json",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text or (json.dumps(json_data) if json_data is not None else "")
    resp.headers = {"Content-Type": content_type}
    resp.json.return_value = json_data or {}
    return resp


def _client_with_session(session: MagicMock) -> PlatformClient:
    return PlatformClient(api_key="ovr_test", base_url="https://api.example.com", session=session)


def test_infer_tool_domain_prefixes():
    assert infer_tool_domain("list_eval_runs") == "evals"
    assert infer_tool_domain("workshop_clean") == "workshop"
    assert infer_tool_domain("graph_walk") == "graph"
    assert infer_tool_domain("something_random") == "other"


def test_list_tools_json_rpc_payload():
    session = MagicMock()
    session.headers = {}
    rpc_result = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "list_eval_runs", "description": "List eval runs", "inputSchema": {"type": "object"}},
                {"name": "foo_tool", "description": "Other", "inputSchema": {}},
            ]
        },
    }
    session.post.return_value = _mock_response(json_data=rpc_result)

    tools = _client_with_session(session).list_tools()

    session.post.assert_called_once()
    call = session.post.call_args
    assert call.args[0] == "https://api.example.com/api/mcp/"
    payload = call.kwargs["json"]
    assert payload["method"] == "tools/list"
    assert payload["jsonrpc"] == "2.0"
    assert len(tools) == 2
    assert tools[0].name == "list_eval_runs"
    assert tools[0].domain == "evals"
    assert tools[1].domain == "other"


def test_describe_tool_returns_schema():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = _mock_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "create_eval_run",
                        "description": "Start an eval run",
                        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    }
                ]
            },
        }
    )

    detail = _client_with_session(session).describe_tool("create_eval_run")

    assert detail.name == "create_eval_run"
    assert detail.input_schema["properties"]["name"]["type"] == "string"
    assert detail.domain == "evals"


def test_describe_tool_missing_raises():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = _mock_response(
        json_data={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    )

    with pytest.raises(PlatformError, match="Tool not found"):
        _client_with_session(session).describe_tool("missing_tool")


def test_call_tool_json_rpc_payload():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = _mock_response(
        json_data={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "isError": False,
            },
        }
    )

    result = _client_with_session(session).call_tool("list_capabilities", {})

    call = session.post.call_args
    payload = call.kwargs["json"]
    assert payload["method"] == "tools/call"
    assert payload["params"] == {"name": "list_capabilities", "arguments": {}}
    assert result.text() == '{"ok": true}'
    assert not result.is_error


def test_rpc_error_raises_platform_error():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = _mock_response(
        json_data={"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "bad request"}}
    )

    with pytest.raises(PlatformError, match="bad request"):
        _client_with_session(session).list_tools()


def test_http_error_raises_platform_error():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = _mock_response(ok=False, status_code=401, text="unauthorized")

    with pytest.raises(PlatformError, match="HTTP 401"):
        _client_with_session(session).list_tools()


def test_parse_sse_payload():
    text = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n"
    data = _parse_sse_payload(text)
    assert data["result"] == {"tools": []}


def test_parse_mcp_response_sse_content_type():
    resp = _mock_response(
        text="data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n",
        content_type="text/event-stream",
    )
    data = _parse_mcp_response(resp)
    assert data["result"] == {"tools": []}
