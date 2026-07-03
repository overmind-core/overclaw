"""Tests for :func:`overmind.tracing._normalize_for_json` and
:func:`overmind.tracing._coerce_to_otel_attribute`.

These tests exercise:

* dataclasses, including nested dataclasses inside dicts / lists
* pydantic-style ``model_dump`` providers
* ``set`` / ``frozenset`` → list
* ``PurePath`` and ``bytes`` coercion
* skip-list types (``Console``, ``Span``, …) get a tag, not a JSON dump
* OTel attribute coercion preserves primitives and ``list[str]`` shapes
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import PurePath

from overmind.tracing import _coerce_to_otel_attribute, _json_dumps, _normalize_for_json


@dataclasses.dataclass
class _Point:
    x: int
    y: int


@dataclasses.dataclass
class _Container:
    name: str
    point: _Point


class _PydanticLike:
    def model_dump(self) -> dict:
        return {"kind": "pydantic", "ok": True}


class _Console:
    """Stand-in for any of the rich UI types in :data:`_SKIP_INPUT_TYPES`."""

    def __init__(self) -> None:
        self.x = 1


# Force the lookup by *type name* used inside the skip-list check.
_Console.__name__ = "Console"


class TestNormalizeForJson:
    def test_passthrough_for_primitives(self):
        for v in ("s", 1, 1.5, True, False, None):
            assert _normalize_for_json(v) == v

    def test_dataclass_flattens_to_dict(self):
        result = _normalize_for_json(_Point(x=3, y=4))
        assert result == {"x": 3, "y": 4}

    def test_nested_dataclass_recurses(self):
        result = _normalize_for_json(_Container(name="origin", point=_Point(0, 0)))
        assert result == {"name": "origin", "point": {"x": 0, "y": 0}}

    def test_pydantic_model_dump_used(self):
        assert _normalize_for_json(_PydanticLike()) == {"kind": "pydantic", "ok": True}

    def test_pydantic_model_dump_failure_falls_back_to_str(self):
        class _BadDump:
            def model_dump(self):
                raise RuntimeError("boom")

        out = _normalize_for_json(_BadDump())
        assert isinstance(out, str)

    def test_set_normalises_to_list(self):
        result = _normalize_for_json({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_frozenset_normalises_to_list(self):
        assert sorted(_normalize_for_json(frozenset({"a", "b"}))) == ["a", "b"]

    def test_tuple_normalises_to_list(self):
        assert _normalize_for_json((1, "x", True)) == [1, "x", True]

    def test_bytes_hex_encoded(self):
        assert _normalize_for_json(b"\x00\xff") == "00ff"

    def test_path_stringified(self):
        assert _normalize_for_json(PurePath("/tmp/x")) == "/tmp/x"

    def test_skip_type_returns_tag(self):
        out = _normalize_for_json(_Console())
        assert out == "<Console>"

    def test_dict_keys_stringified(self):
        assert _normalize_for_json({1: "a"}) == {"1": "a"}

    def test_unknown_object_stringified(self):
        class _Opaque:
            pass

        out = _normalize_for_json(_Opaque())
        # ``_Opaque`` has no __dict__ entries, so we fall back to repr.
        assert isinstance(out, (str, dict))

    def test_result_is_json_serialisable(self):
        """Whatever we return must round-trip through :func:`json.dumps`."""
        normalised = _normalize_for_json({
            "point": _Point(1, 2),
            "tags": {"alpha", "beta"},
            "path": PurePath("/tmp/x"),
            "raw": b"\x01",
        })
        json.dumps(normalised)


class TestJsonDumps:
    def test_round_trip_via_json_loads(self):
        raw = _json_dumps({"point": _Point(1, 2), "tags": {"alpha"}})
        loaded = json.loads(raw)
        assert loaded["point"] == {"x": 1, "y": 2}
        assert loaded["tags"] == ["alpha"]

    def test_never_raises(self):
        class _CycleSafe:
            def __repr__(self) -> str:
                return "<safe>"

        # Even objects with no useful representation should not raise.
        assert isinstance(_json_dumps(_CycleSafe()), str)


class TestCoerceToOtelAttribute:
    def test_none_becomes_empty_string(self):
        assert _coerce_to_otel_attribute(None) == ""

    def test_primitives_pass_through(self):
        for v in ("s", 1, 1.5, True, False):
            assert _coerce_to_otel_attribute(v) == v

    def test_list_of_strings_preserved(self):
        assert _coerce_to_otel_attribute(["a", "b"]) == ["a", "b"]

    def test_mixed_list_becomes_json_string(self):
        out = _coerce_to_otel_attribute([1, "a"])
        assert isinstance(out, str)
        assert json.loads(out) == [1, "a"]

    def test_dict_becomes_json_string(self):
        out = _coerce_to_otel_attribute({"k": 1})
        assert isinstance(out, str)
        assert json.loads(out) == {"k": 1}

    def test_dataclass_becomes_json_string(self):
        out = _coerce_to_otel_attribute(_Point(1, 2))
        assert json.loads(out) == {"x": 1, "y": 2}
