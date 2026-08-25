"""Tests for overmind.env — dotenv loading for the CLI."""

from __future__ import annotations

import os

from overmind.env import load_project_env


def test_load_project_env_from_cwd_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OVERMIND_API_KEY=from_dotenv\n")
    load_project_env()
    assert os.environ["OVERMIND_API_KEY"] == "from_dotenv"


def test_load_project_env_does_not_override_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OVERMIND_API_KEY=from_dotenv\n")
    monkeypatch.setenv("OVERMIND_API_KEY", "from_shell")
    load_project_env()
    assert os.environ["OVERMIND_API_KEY"] == "from_shell"


def test_load_project_env_from_cwd_argument(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("OVERMIND_API_KEY=from_repo\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
    load_project_env(repo)
    assert os.environ["OVERMIND_API_KEY"] == "from_repo"


def test_load_project_env_from_overmind_cwd_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("OVERMIND_API_KEY=from_env_cwd\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
    monkeypatch.setenv("OVERMIND_CWD", str(repo))
    load_project_env()
    assert os.environ["OVERMIND_API_KEY"] == "from_env_cwd"


def test_load_project_env_from_overmind_env_file(tmp_path, monkeypatch):
    custom = tmp_path / "custom.env"
    custom.write_text("OVERMIND_API_KEY=from_custom\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
    monkeypatch.setenv("OVERMIND_ENV_FILE", str(custom))
    load_project_env()
    assert os.environ["OVERMIND_API_KEY"] == "from_custom"
