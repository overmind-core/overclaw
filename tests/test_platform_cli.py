"""Tests for overmind platform CLI commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from overmind.__main__ import app
from overmind.platform.client import PlatformError
from overmind.platform.types import ToolCallResult, ToolDetail, ToolSummary

runner = CliRunner()


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.list_tools.return_value = [
        ToolSummary(name="list_eval_runs", description="List eval runs", domain="evals"),
        ToolSummary(name="foo_tool", description="Other tool", domain="other"),
    ]
    client.describe_tool.return_value = ToolDetail(
        name="create_eval_run",
        description="Start eval",
        input_schema={"type": "object"},
        domain="evals",
    )
    client.call_tool.return_value = ToolCallResult(
        content=[{"type": "text", "text": '{"status":"ok"}'}],
        is_error=False,
    )
    return client


@patch("overmind.commands.platform.PlatformClient")
def test_platform_list_table(mock_cls):
    mock_cls.return_value = _mock_client()
    result = runner.invoke(app, ["platform", "list"], env={"OVERMIND_API_KEY": "k"}, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "list_eval_runs" in result.output
    assert "evals" in result.output


@patch("overmind.commands.platform.PlatformClient")
def test_platform_list_domain_filter(mock_cls):
    mock_cls.return_value = _mock_client()
    result = runner.invoke(
        app,
        ["platform", "list", "--domain", "evals"],
        env={"OVERMIND_API_KEY": "k"},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "list_eval_runs" in result.output
    assert "foo_tool" not in result.output


@patch("overmind.commands.platform.PlatformClient")
def test_platform_list_json(mock_cls):
    mock_cls.return_value = _mock_client()
    result = runner.invoke(
        app,
        ["platform", "list", "--json"],
        env={"OVERMIND_API_KEY": "k"},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["name"] == "list_eval_runs"


def test_platform_list_bad_domain():
    result = runner.invoke(
        app,
        ["platform", "list", "--domain", "nope"],
        env={"OVERMIND_API_KEY": "k"},
    )
    assert result.exit_code != 0


@patch("overmind.commands.platform.PlatformClient")
def test_platform_describe_json(mock_cls):
    mock_cls.return_value = _mock_client()
    result = runner.invoke(
        app,
        ["platform", "describe", "create_eval_run", "--json"],
        env={"OVERMIND_API_KEY": "k"},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["name"] == "create_eval_run"
    assert data["inputSchema"]["type"] == "object"


@patch("overmind.commands.platform.PlatformClient")
def test_platform_call_with_args(mock_cls):
    mock_cls.return_value = _mock_client()
    result = runner.invoke(
        app,
        ["platform", "call", "list_capabilities", "--args", "{}", "--json"],
        env={"OVERMIND_API_KEY": "k"},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    mock_cls.return_value.call_tool.assert_called_once_with("list_capabilities", {})
    data = json.loads(result.output)
    assert data["isError"] is False


@patch("overmind.commands.platform.PlatformClient")
def test_platform_call_args_file(mock_cls, tmp_path):
    args_path = tmp_path / "args.json"
    args_path.write_text('{"kind":"eval_run","id":"abc"}')
    mock_cls.return_value = _mock_client()
    result = runner.invoke(
        app,
        ["platform", "call", "job_status", "--args-file", str(args_path), "--json"],
        env={"OVERMIND_API_KEY": "k"},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    mock_cls.return_value.call_tool.assert_called_once_with(
        "job_status",
        {"kind": "eval_run", "id": "abc"},
    )


@patch("overmind.commands.platform.PlatformClient")
def test_platform_call_platform_error(mock_cls):
    mock_cls.return_value.call_tool.side_effect = PlatformError("tool failed")
    result = runner.invoke(
        app,
        ["platform", "call", "foo", "--args", "{}"],
        env={"OVERMIND_API_KEY": "k"},
    )
    assert result.exit_code != 0
    assert "tool failed" in result.output
