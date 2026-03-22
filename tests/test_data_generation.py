"""Tests for overclaw.optimize.data — diverse generation pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overclaw.optimize.data import (
    DatasetGenerationError,
    _apply_dedup,
    _canonicalize,
    _get_key_fields,
    _is_near_duplicate,
    _print_coverage_report,
    _retry_dropped_slots,
    _stratified_sample,
    generate_diverse_synthetic_data,
    validate_case_against_spec,
)


class TestCanonicalizeAndDedup:
    def test_canonicalize_deterministic(self):
        inp = {"b": 2, "a": 1}
        c1 = _canonicalize(inp)
        c2 = _canonicalize({"a": 1, "b": 2})
        assert c1 == c2

    def test_get_key_fields(self):
        inp = {"name": "Alice", "company": "Acme", "score": 50}
        kf = _get_key_fields(inp)
        assert isinstance(kf, tuple)

    def test_is_near_duplicate_empty(self):
        assert not _is_near_duplicate({"name": "Alice"}, set())

    def test_is_near_duplicate_match(self):
        inp = {"name": "Alice", "company": "Acme", "extra": "x"}
        kf = _get_key_fields(inp)
        seen = {kf}
        result = _is_near_duplicate(inp, seen)
        assert result is True

    def test_apply_dedup_filters_duplicates(self):
        out: list[dict] = []
        seen_c: set[str] = set()
        seen_kf: set[tuple] = set()
        cases = [
            {"input": {"x": 1}, "expected_output": {"y": 2}},
            {"input": {"x": 1}, "expected_output": {"y": 2}},
        ]
        added, dups = _apply_dedup(cases, {}, seen_c, seen_kf, out)
        assert added == 1
        assert dups == 1

    def test_apply_dedup_validates_spec(self):
        out: list[dict] = []
        seen_c: set[str] = set()
        seen_kf: set[tuple] = set()
        spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["hot", "cold"]},
            }
        }
        cases = [
            {"input": {"x": 1}, "expected_output": {"status": "hot"}},
            {"input": {"x": 2}, "expected_output": {"status": "invalid_value"}},
        ]
        added, dups = _apply_dedup(cases, spec, seen_c, seen_kf, out)
        assert added >= 1


class TestValidateCaseAgainstSpec:
    def test_valid_case(self):
        spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["hot", "cold"]},
                "score": {"type": "number", "range": [0, 100]},
            }
        }
        case = {"input": {"x": 1}, "expected_output": {"status": "hot", "score": 50}}
        errors = validate_case_against_spec(case, spec)
        assert errors == []

    def test_missing_input(self):
        case = {"expected_output": {"x": 1}}
        errors = validate_case_against_spec(case, {})
        assert any("input" in e for e in errors)

    def test_enum_value_violation(self):
        spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["hot", "cold"]},
            }
        }
        case = {"input": {}, "expected_output": {"status": "invalid"}}
        errors = validate_case_against_spec(case, spec)
        assert len(errors) >= 1


class TestStratifiedSample:
    def test_returns_all_when_small(self):
        cases = [{"input": {"x": i}} for i in range(5)]
        result = _stratified_sample(cases, 10)
        assert len(result) == 5

    def test_samples_down(self):
        cases = [{"input": {"x": i}} for i in range(50)]
        result = _stratified_sample(list(cases), 10)
        assert len(result) == 10


class TestRetryDroppedSlots:
    @patch("overclaw.optimize.data._generate_batch")
    def test_retries_and_succeeds(self, mock_batch):
        mock_batch.return_value = [{"input": {"x": 99}, "expected_output": {"y": 1}}]
        out: list[dict] = []
        added = _retry_dropped_slots(
            retry_slots=[0],
            personas=[{"name": "Test"}],
            agent_description="desc",
            agent_code=None,
            eval_spec={},
            policy_context=None,
            model="model",
            existing_snapshot=[],
            coverage_gaps=None,
            seen_canonical=set(),
            seen_key_fields=set(),
            out=out,
        )
        assert added >= 1


class TestPrintCoverageReport:
    def test_with_enum_and_number(self):
        console = MagicMock()
        eval_spec = {
            "output_fields": {
                "status": {"type": "enum", "values": ["hot", "cold"]},
                "score": {"type": "number", "range": [0, 100]},
            }
        }
        cases = [
            {"expected_output": {"status": "hot", "score": 50}},
            {"expected_output": {"status": "cold", "score": 90}},
        ]
        _print_coverage_report(cases, eval_spec, console)

    def test_empty_cases(self):
        console = MagicMock()
        _print_coverage_report([], {}, console)


class TestGenerateDiverseSyntheticData:
    @patch("overclaw.optimize.data._per_persona_parallel_shards_round")
    @patch("overclaw.optimize.data._generate_personas")
    def test_success(self, mock_personas, mock_round):
        mock_personas.return_value = [
            {"name": "Tester", "skill_level": "expert", "intent": "standard"}
        ]
        mock_round.return_value = [
            [{"input": {"x": i}, "expected_output": {"y": i}} for i in range(5)]
        ]

        console = MagicMock()
        result = generate_diverse_synthetic_data(
            "desc", "model", num_samples=3, num_personas=1, console=console
        )
        assert len(result) == 3

    @patch("overclaw.optimize.data._per_persona_parallel_shards_round")
    @patch("overclaw.optimize.data._generate_personas")
    def test_empty_raises(self, mock_personas, mock_round):
        mock_personas.return_value = [{"name": "T"}]
        mock_round.return_value = [[]]

        console = MagicMock()
        with pytest.raises(DatasetGenerationError):
            generate_diverse_synthetic_data(
                "desc", "model", num_samples=5, num_personas=1, console=console
            )
