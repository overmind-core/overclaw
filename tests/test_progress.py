"""Tests for overclaw.utils.display — display utilities."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from overclaw.utils.display import make_spinner_progress, rel


class TestRel:
    def test_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        full_path = tmp_path / "sub" / "file.txt"
        result = rel(str(full_path))
        assert result == str(Path("sub") / "file.txt")

    def test_outside_cwd_returns_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = rel("/some/other/path")
        assert result == "/some/other/path"

    def test_path_object(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = rel(tmp_path / "file.txt")
        assert result == "file.txt"


class TestMakeSpinnerProgress:
    def test_returns_progress(self):
        console = Console()
        progress = make_spinner_progress(console)
        assert progress is not None

    def test_transient_mode(self):
        console = Console()
        progress = make_spinner_progress(console, transient=True)
        assert progress is not None
