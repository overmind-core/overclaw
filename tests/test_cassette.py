"""Tests for the surviving public surface of ``overmind.optimize.cassette``.

The module previously exposed a full in-Python record/replay API
(``Cassette.record``, ``Cassette.replay``, ``NullCassette``, ``__len__``,
etc.) that production code never called — every real cassette read/write
happens inside the subprocess via :mod:`overmind.optimize.shadow_runtime`.
That dead API was deleted; the remaining surface is:

* :func:`make_key` — stable content-addressable key, also used by the
  subprocess shadow runtime so its output is identical to keys computed
  out-of-process for diagnostics.
* :class:`CassetteEntry` — on-disk row schema; the shadow runtime emits
  rows in this shape.
* :class:`Cassette` / :func:`open_cassette` — path-holder threaded
  through ``ExecutionBackend`` instances.
"""

from __future__ import annotations

from pathlib import Path

from overmind.optimize.cassette import (
    Cassette,
    CassetteEntry,
    make_key,
    open_cassette,
)


class TestMakeKey:
    def test_same_inputs_same_key(self):
        k1 = make_key("llm", "gpt-4o", {"messages": [{"role": "user", "content": "hi"}]})
        k2 = make_key("llm", "gpt-4o", {"messages": [{"role": "user", "content": "hi"}]})
        assert k1 == k2

    def test_different_kind_different_key(self):
        assert make_key("llm", "gpt-4o", {}) != make_key("tool", "gpt-4o", {})

    def test_different_identifier_different_key(self):
        assert make_key("llm", "gpt-4o", {}) != make_key("llm", "claude", {})

    def test_dict_ordering_is_canonical(self):
        assert make_key("llm", "x", {"b": 2, "a": 1}) == make_key("llm", "x", {"a": 1, "b": 2})

    def test_nested_dict_stability(self):
        k1 = make_key("llm", "x", {"m": [{"role": "u", "content": "hi"}]})
        k2 = make_key("llm", "x", {"m": [{"content": "hi", "role": "u"}]})
        assert k1 == k2


class TestCassetteEntry:
    def test_default_version(self):
        entry = CassetteEntry(
            kind="llm",
            identifier="m",
            key="abc",
            payload={"x": 1},
            result={"y": 2},
        )
        assert entry.version == 1
        assert entry.metadata == {}

    def test_asdict_roundtrip(self):
        from dataclasses import asdict

        entry = CassetteEntry(
            kind="llm",
            identifier="m",
            key="abc",
            payload={"x": 1},
            result={"y": 2},
            metadata={"latency": 0.5},
        )
        d = asdict(entry)
        assert CassetteEntry(**d) == entry


class TestCassettePathHolder:
    def test_open_cassette_with_path(self, tmp_path: Path):
        cass = open_cassette(tmp_path / "c.jsonl")
        assert isinstance(cass, Cassette)
        assert cass.path == tmp_path / "c.jsonl"

    def test_open_cassette_with_none(self):
        cass = open_cassette(None)
        assert isinstance(cass, Cassette)
        assert cass.path is None

    def test_open_cassette_with_empty_string(self):
        cass = open_cassette("")
        assert isinstance(cass, Cassette)
        assert cass.path is None

    def test_string_path_is_coerced(self, tmp_path: Path):
        cass = open_cassette(str(tmp_path / "c.jsonl"))
        assert cass.path == tmp_path / "c.jsonl"

    def test_pathless_cassette_is_truthy(self):
        """No ``__len__`` means an empty cassette is truthy by default.

        This guards against re-introducing the historical bug where a
        truthiness check on ``Cassette`` would silently swap in a no-op
        instance because the (now-deleted) ``__len__`` returned 0.
        """
        cass = open_cassette(None)
        assert bool(cass) is True
