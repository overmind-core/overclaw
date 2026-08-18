import json
from pathlib import Path

from typer.testing import CliRunner

from overmind.__main__ import app

runner = CliRunner()


def _init(tmp_path: Path, ide: str) -> None:
    result = runner.invoke(
        app,
        ["init", "--ide", ide, "--env", "local", "--api-key", "k"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


def test_init_writes_cursor_mcp_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init(tmp_path, "cursor")
    cfg = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
    assert cfg["mcpServers"]["overmind"] == {
        "url": "http://localhost:8000/api/mcp/",
        "headers": {"X-Api-Key": "k"},
    }
    assert (tmp_path / ".cursor" / "skills" / "overmind" / "SKILL.md").is_file()


def test_init_claude_alias_and_opencode_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init(tmp_path, "claude")
    assert (tmp_path / ".claude" / "mcp.json").is_file()

    _init(tmp_path, "opencode")
    cfg = json.loads((tmp_path / "opencode.json").read_text())
    assert cfg["mcp"]["overmind"] == {
        "type": "remote",
        "url": "http://localhost:8000/api/mcp/",
        "enabled": True,
        "headers": {"X-Api-Key": "k"},
    }
    assert (tmp_path / ".opencode" / "skills" / "overmind" / "SKILL.md").is_file()
