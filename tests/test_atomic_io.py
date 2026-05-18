"""Tests for :mod:`overmind.utils.atomic_io`.

The whole point of the module is that an interrupted write must never leave
a half-written file at the target path.  The tests below exercise that
contract by injecting a failure between the ``tmp`` write and the rename.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from overmind.utils.atomic_io import atomic_write_json, atomic_write_text


class TestAtomicWriteText:
    def test_creates_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "deeply" / "nested" / "file.txt"
        atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_replaces_existing_file(self, tmp_path: Path):
        target = tmp_path / "f.txt"
        target.write_text("old")
        atomic_write_text(target, "new")
        assert target.read_text() == "new"

    def test_no_tmp_left_behind_on_success(self, tmp_path: Path):
        target = tmp_path / "f.txt"
        atomic_write_text(target, "x")
        siblings = list(tmp_path.iterdir())
        assert len(siblings) == 1, f"Expected only the target, found: {siblings}"
        assert siblings[0] == target

    def test_interrupt_before_rename_leaves_original_intact(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("ORIGINAL")

        # Simulate a crash between writing the temp file and the rename.
        with patch("pathlib.Path.replace", side_effect=OSError("simulated crash")), pytest.raises(OSError):
            atomic_write_text(target, "PARTIAL")

        # Original target survives untouched; the partial lives in a sibling .tmp.
        assert target.read_text() == "ORIGINAL"
        tmp_files = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert tmp_files, "tmp file should exist so callers can recover or inspect"


class TestAtomicWriteJson:
    def test_round_trip(self, tmp_path: Path):
        target = tmp_path / "data.json"
        payload = {"a": 1, "b": [2, 3]}
        atomic_write_json(target, payload)
        assert json.loads(target.read_text()) == payload

    def test_indent_defaults_to_2(self, tmp_path: Path):
        target = tmp_path / "data.json"
        atomic_write_json(target, {"a": 1})
        assert "  " in target.read_text()

    def test_indent_none_compact(self, tmp_path: Path):
        target = tmp_path / "data.json"
        atomic_write_json(target, {"a": 1}, indent=None)
        assert target.read_text() == '{"a": 1}'

    def test_default_str_handles_non_serialisable(self, tmp_path: Path):
        target = tmp_path / "data.json"
        atomic_write_json(target, {"path": Path("/tmp/x")})
        # default=str converts the Path object — JSON still loads cleanly.
        loaded = json.loads(target.read_text())
        assert loaded["path"] == "/tmp/x"
