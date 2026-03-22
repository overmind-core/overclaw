"""Tests for overclaw.optimize.data_analyzer — seed data validation and coverage."""

from __future__ import annotations

from unittest.mock import MagicMock


from overclaw.optimize.data_analyzer import (
    _display_analysis,
    _fallback_analysis,
    validate_seed_data,
)


# ---------------------------------------------------------------------------
# validate_seed_data
# ---------------------------------------------------------------------------


class TestValidateSeedData:
    def test_all_valid(self):
        spec = {
            "input_schema": {"x": {"type": "string"}},
            "output_fields": {"y": {"type": "text", "eval_mode": "non_empty"}},
        }
        cases = [
            {"input": {"x": "hello"}, "expected_output": {"y": "result"}},
            {"input": {"x": "world"}, "expected_output": {"y": "output"}},
        ]
        result = validate_seed_data(cases, spec)
        assert result["valid_count"] == 2
        assert result["invalid_count"] == 0

    def test_some_invalid(self):
        spec = {
            "input_schema": {},
            "output_fields": {
                "status": {"type": "enum", "values": ["a", "b"]},
            },
        }
        cases = [
            {"input": {}, "expected_output": {"status": "a"}},
            {"input": {}, "expected_output": {"status": "invalid"}},
        ]
        result = validate_seed_data(cases, spec)
        assert result["valid_count"] == 1
        assert result["invalid_count"] == 1
        assert len(result["issues"]) == 1

    def test_empty_cases(self):
        result = validate_seed_data([], {"input_schema": {}, "output_fields": {}})
        assert result["total_cases"] == 0
        assert result["valid_count"] == 0


# ---------------------------------------------------------------------------
# _fallback_analysis
# ---------------------------------------------------------------------------


class TestFallbackAnalysis:
    def test_detects_missing_enum_values(self):
        spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["a", "b", "c"]},
            },
        }
        cases = [
            {"expected_output": {"status": "a"}},
            {"expected_output": {"status": "b"}},
        ]
        result = _fallback_analysis(cases, spec)
        assert result["overall_quality_score"] == 5
        gaps = result["coverage_gaps"]
        assert len(gaps) == 1
        assert "c" in gaps[0]["description"]

    def test_no_gaps_when_all_covered(self):
        spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["a", "b"]},
            },
        }
        cases = [
            {"expected_output": {"status": "a"}},
            {"expected_output": {"status": "b"}},
        ]
        result = _fallback_analysis(cases, spec)
        assert result["coverage_gaps"] == []

    def test_empty_cases(self):
        spec = {"output_fields": {"status": {"type": "enum", "values": ["a"]}}}
        result = _fallback_analysis([], spec)
        assert len(result["coverage_gaps"]) == 1

    def test_no_enum_fields(self):
        spec = {"output_fields": {"score": {"type": "number"}}}
        result = _fallback_analysis([{"expected_output": {"score": 5}}], spec)
        assert result["coverage_gaps"] == []


# ---------------------------------------------------------------------------
# _display_analysis
# ---------------------------------------------------------------------------


class TestDisplayAnalysis:
    def test_displays_without_error(self):
        console = MagicMock()
        analysis = {
            "overall_quality_score": 7,
            "case_count": 10,
            "difficulty_distribution": {"easy": 3, "medium": 5, "hard": 2},
            "coverage_gaps": [
                {
                    "severity": "high",
                    "area": "enum:status",
                    "description": "Missing value C",
                },
            ],
            "uncovered_policy_rules": ["Rule X"],
            "quality_issues": [{"case_index": 3, "issue": "Bad data"}],
            "augmentation_recommendation": "Add 5 more cases",
        }
        _display_analysis(analysis, console)
        assert console.print.called

    def test_displays_no_gaps(self):
        console = MagicMock()
        analysis = {
            "overall_quality_score": 9,
            "case_count": 20,
            "coverage_gaps": [],
        }
        _display_analysis(analysis, console)

    def test_displays_low_score(self):
        console = MagicMock()
        analysis = {"overall_quality_score": 2, "case_count": 3}
        _display_analysis(analysis, console)

    def test_displays_string_score(self):
        console = MagicMock()
        analysis = {"overall_quality_score": "unknown", "case_count": 0}
        _display_analysis(analysis, console)
