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


def test_init_claude_writes_project_root_mcp_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)

    for alias in ("claude", "claude_code", "claude-code"):
        _init(tmp_path, alias)

        # Claude Code reads .mcp.json at the project root, never .claude/mcp.json.
        cfg = json.loads((tmp_path / ".mcp.json").read_text())
        assert cfg["mcpServers"]["overmind"] == {
            "type": "http",
            "url": "http://localhost:8000/api/mcp/",
            "headers": {"X-Api-Key": "k"},
        }
        assert not (tmp_path / ".claude" / "mcp.json").exists()
        assert (tmp_path / ".claude" / "skills" / "overmind" / "SKILL.md").is_file()


def test_init_claude_preserves_other_servers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "uvx", "args": ["other"]}}}))

    _init(tmp_path, "claude")

    cfg = json.loads((tmp_path / ".mcp.json").read_text())
    assert cfg["mcpServers"]["other"] == {"command": "uvx", "args": ["other"]}
    assert "overmind" in cfg["mcpServers"]


def test_init_claude_keeps_env_key_out_of_committed_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OVERMIND_API_KEY", "secret")
    result = runner.invoke(app, ["init", "--ide", "claude", "--env", "local"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    cfg = json.loads((tmp_path / ".mcp.json").read_text())
    assert cfg["mcpServers"]["overmind"]["headers"] == {"X-Api-Key": "${OVERMIND_API_KEY}"}
    assert "secret" not in (tmp_path / ".mcp.json").read_text()


def test_init_opencode_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init(tmp_path, "opencode")
    cfg = json.loads((tmp_path / "opencode.json").read_text())
    assert cfg["mcp"]["overmind"] == {
        "type": "remote",
        "url": "http://localhost:8000/api/mcp/",
        "enabled": True,
        "headers": {"X-Api-Key": "k"},
    }
    assert (tmp_path / ".opencode" / "skills" / "overmind" / "SKILL.md").is_file()


def test_init_codex_writes_project_config_and_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OVERMIND_API_KEY", "k")
    result = runner.invoke(app, ["init", "--ide", "codex", "--env", "production"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    config = (tmp_path / ".codex" / "config.toml").read_text()
    assert '[mcp_servers.overmind]\nurl = "https://api.overmindlab.ai/api/mcp/"' in config
    assert 'env_http_headers = { "X-Api-Key" = "OVERMIND_API_KEY" }' in config
    assert (tmp_path / ".agents" / "skills" / "overmind" / "SKILL.md").is_file()


def test_init_codex_replaces_own_section_and_preserves_other_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OVERMIND_API_KEY", "k")
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        'model = "gpt-5"\n\n'
        '[mcp_servers.overmind]\nurl = "https://old.example"\n\n'
        '[mcp_servers.other]\nurl = "https://other.example"\n'
    )

    result = runner.invoke(app, ["init", "--ide", "codex"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    config = config_path.read_text()
    assert 'model = "gpt-5"' in config
    assert 'url = "https://other.example"' in config
    assert config.count("[mcp_servers.overmind]") == 1
    assert 'url = "https://old.example"' not in config
    assert 'env_http_headers = { "X-Api-Key" = "OVERMIND_API_KEY" }' in config
