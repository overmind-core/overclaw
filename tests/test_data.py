"""Tests for overclaw.optimize.data — data loading, validation, dedup, generation."""

from __future__ import annotations

import json

import pytest

from overclaw.optimize.data import (
    _apply_dedup,
    _canonicalize,
    _default_personas,
    _format_input_schema,
    _format_output_schema,
    _get_key_fields,
    _is_near_duplicate,
    _safe_parse_json,
    _stratified_sample,
    load_data,
    validate_case_against_spec,
)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_load_array(self, tmp_path):
        data = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_data(str(path))
        assert result == data

    def test_load_object_with_test_cases(self, tmp_path):
        data = {"test_cases": [{"input": {"x": 1}}]}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_data(str(path))
        assert result == [{"input": {"x": 1}}]

    def test_unrecognized_format_raises(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"key": "value"}))
        with pytest.raises(ValueError, match="Unrecognized"):
            load_data(str(path))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/file.json")

    def test_empty_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("[]")
        assert load_data(str(path)) == []


# ---------------------------------------------------------------------------
# validate_case_against_spec
# ---------------------------------------------------------------------------


SAMPLE_SPEC = {
    "input_schema": {
        "company_name": {"type": "string"},
        "budget": {"type": "number"},
    },
    "output_fields": {
        "qualification": {
            "type": "enum",
            "values": ["hot", "warm", "cold"],
        },
        "score": {
            "type": "number",
            "range": [0, 100],
        },
        "reasoning": {
            "type": "text",
            "eval_mode": "non_empty",
        },
    },
}


class TestValidateCaseAgainstSpec:
    def test_valid_case(self):
        case = {
            "input": {"company_name": "Acme", "budget": 50000},
            "expected_output": {
                "qualification": "hot",
                "score": 85,
                "reasoning": "Good match",
            },
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert errors == []

    def test_missing_input(self):
        errors = validate_case_against_spec({"expected_output": {}}, SAMPLE_SPEC)
        assert any("input" in e.lower() for e in errors)

    def test_non_dict_input(self):
        errors = validate_case_against_spec(
            {"input": "string", "expected_output": {}}, SAMPLE_SPEC
        )
        assert any("input" in e.lower() for e in errors)

    def test_missing_expected_output(self):
        errors = validate_case_against_spec({"input": {}}, SAMPLE_SPEC)
        assert any("expected_output" in e.lower() for e in errors)

    def test_missing_input_field(self):
        case = {
            "input": {"company_name": "Acme"},
            "expected_output": {"qualification": "hot", "score": 50, "reasoning": "ok"},
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("budget" in e for e in errors)

    def test_invalid_enum_value(self):
        case = {
            "input": {"company_name": "Acme", "budget": 1000},
            "expected_output": {
                "qualification": "invalid",
                "score": 50,
                "reasoning": "ok",
            },
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("qualification" in e for e in errors)

    def test_number_wrong_type(self):
        case = {
            "input": {"company_name": "Acme", "budget": 1000},
            "expected_output": {
                "qualification": "hot",
                "score": "not a number",
                "reasoning": "ok",
            },
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("score" in e and "number" in e for e in errors)

    def test_number_out_of_range(self):
        case = {
            "input": {"company_name": "Acme", "budget": 1000},
            "expected_output": {
                "qualification": "hot",
                "score": 150,
                "reasoning": "ok",
            },
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("range" in e for e in errors)

    def test_empty_non_empty_text(self):
        case = {
            "input": {"company_name": "Acme", "budget": 1000},
            "expected_output": {"qualification": "hot", "score": 50, "reasoning": ""},
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("non-empty" in e for e in errors)

    def test_missing_output_field(self):
        case = {
            "input": {"company_name": "Acme", "budget": 1000},
            "expected_output": {"qualification": "hot"},
        }
        errors = validate_case_against_spec(case, SAMPLE_SPEC)
        assert any("score" in e for e in errors)

    def test_completely_empty(self):
        errors = validate_case_against_spec({}, SAMPLE_SPEC)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# _safe_parse_json
# ---------------------------------------------------------------------------


class TestSafeParseJson:
    def test_plain_json(self):
        assert _safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_array(self):
        result = _safe_parse_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_fenced_json(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert _safe_parse_json(text) == {"key": "value"}

    def test_fenced_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert _safe_parse_json(text) == {"key": "value"}

    def test_embedded_json_object(self):
        text = 'Here is the result: {"key": 42} and more text'
        assert _safe_parse_json(text) == {"key": 42}

    def test_embedded_json_array(self):
        text = "Result: [1, 2] trailing"
        assert _safe_parse_json(text) == [1, 2]

    def test_single_quotes_repaired(self):
        result = _safe_parse_json("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_trailing_comma_repaired(self):
        result = _safe_parse_json('{"key": "value",}')
        assert result == {"key": "value"}

    def test_completely_invalid(self):
        assert _safe_parse_json("not json at all") is None

    def test_empty_string(self):
        assert _safe_parse_json("") is None


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_consistent(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _canonicalize(d1) == _canonicalize(d2)

    def test_lowercase(self):
        result = _canonicalize({"Key": "Value"})
        assert result == result.lower()


class TestGetKeyFields:
    def test_extracts_strings(self):
        result = _get_key_fields({"name": "John Doe", "id": 42, "x": "ab"})
        assert "john doe" in result
        assert len(result) == 1  # "ab" is too short (len 2)

    def test_empty_dict(self):
        assert _get_key_fields({}) == ()


class TestIsNearDuplicate:
    def test_duplicate(self):
        existing = {("john doe",)}
        assert _is_near_duplicate({"name": "John Doe"}, existing) is True

    def test_not_duplicate(self):
        existing = {("jane doe",)}
        assert _is_near_duplicate({"name": "John Doe"}, existing) is False

    def test_empty_keys(self):
        assert _is_near_duplicate({"id": 1}, set()) is False


# ---------------------------------------------------------------------------
# _apply_dedup
# ---------------------------------------------------------------------------


class TestApplyDedup:
    def test_adds_valid_cases(self):
        spec = {
            "input_schema": {},
            "output_fields": {},
        }
        cases = [
            {"input": {"name": "case one"}, "expected_output": {}},
            {"input": {"name": "case two"}, "expected_output": {}},
        ]
        out: list[dict] = []
        seen_c: set[str] = set()
        seen_k: set[tuple[str, ...]] = set()
        added, dups = _apply_dedup(cases, spec, seen_c, seen_k, out)
        assert added == 2
        assert dups == 0
        assert len(out) == 2

    def test_drops_exact_duplicates(self):
        cases = [
            {"input": {"name": "same"}, "expected_output": {}},
            {"input": {"name": "same"}, "expected_output": {}},
        ]
        out: list[dict] = []
        seen_c: set[str] = set()
        seen_k: set[tuple[str, ...]] = set()
        added, dups = _apply_dedup(cases, {}, seen_c, seen_k, out)
        assert added == 1
        assert dups == 1

    def test_drops_non_dict_cases(self):
        cases = ["not a dict", 42]
        out: list[dict] = []
        added, dups = _apply_dedup(cases, {}, set(), set(), out)
        assert added == 0

    def test_strips_meta_key(self):
        cases = [
            {"input": {"x": "test value"}, "expected_output": {}, "_meta": "remove me"}
        ]
        out: list[dict] = []
        added, _ = _apply_dedup(cases, {}, set(), set(), out)
        assert added == 1
        assert "_meta" not in out[0]


# ---------------------------------------------------------------------------
# _stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_no_downsample_when_under_n(self):
        cases = [{"i": i} for i in range(5)]
        result = _stratified_sample(cases, 10)
        assert len(result) == 5

    def test_downsamples_to_n(self):
        cases = [{"i": i} for i in range(20)]
        result = _stratified_sample(cases, 5)
        assert len(result) == 5

    def test_deterministic(self):
        # _stratified_sample mutates the input list (shuffle in place), so
        # we need fresh copies for each call to test determinism.
        cases_a = [{"i": i} for i in range(20)]
        cases_b = [{"i": i} for i in range(20)]
        a = _stratified_sample(cases_a, 5)
        b = _stratified_sample(cases_b, 5)
        assert a == b


# ---------------------------------------------------------------------------
# _default_personas
# ---------------------------------------------------------------------------


class TestDefaultPersonas:
    def test_returns_requested_count(self):
        personas = _default_personas(3)
        assert len(personas) == 3

    def test_max_five(self):
        personas = _default_personas(10)
        assert len(personas) == 5

    def test_each_has_name(self):
        for p in _default_personas(5):
            assert "name" in p
            assert p["name"]


# ---------------------------------------------------------------------------
# Schema formatters
# ---------------------------------------------------------------------------


class TestFormatSchemas:
    def test_format_input_schema(self):
        spec = {
            "input_schema": {
                "name": {"type": "string", "description": "Company name"},
            }
        }
        result = _format_input_schema(spec)
        assert "name" in result
        assert "string" in result

    def test_format_input_schema_empty(self):
        assert _format_input_schema({}) == ""

    def test_format_output_schema_enum(self):
        spec = {
            "output_fields": {
                "status": {
                    "type": "enum",
                    "description": "Status",
                    "values": ["a", "b"],
                },
            }
        }
        result = _format_output_schema(spec)
        assert "enum" in result
        assert "allowed values" in result

    def test_format_output_schema_number_range(self):
        spec = {
            "output_fields": {
                "score": {
                    "type": "number",
                    "description": "Score",
                    "range": [0, 100],
                },
            }
        }
        result = _format_output_schema(spec)
        assert "range" in result

    def test_format_output_schema_empty(self):
        assert _format_output_schema({}) == ""
