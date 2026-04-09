"""Tests for overclaw.optimize.evaluator — spec-driven scoring."""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from overclaw.optimize.evaluator import (
    SpecEvaluator,
    _JUDGE_FALLBACK_SCORE,
    has_entrypoint,
    has_run_entrypoint,
    load_evaluator,
)


# ---------------------------------------------------------------------------
# has_entrypoint / has_run_entrypoint
# ---------------------------------------------------------------------------


class TestHasEntrypoint:
    def test_def_run_paren(self):
        assert has_entrypoint("def run(input_data):", "run") is True

    def test_def_run_space_paren(self):
        assert has_entrypoint("def run (input_data):", "run") is True

    def test_missing_function(self):
        assert has_entrypoint("def other(x):", "run") is False

    def test_custom_name(self):
        assert has_entrypoint("def handle(request):", "handle") is True

    def test_substring_not_matched(self):
        assert has_entrypoint("def runner(x):", "run") is False


class TestHasRunEntrypoint:
    def test_delegates(self):
        assert has_run_entrypoint("def run(x):") is True
        assert has_run_entrypoint("def other(x):") is False


# ---------------------------------------------------------------------------
# load_evaluator
# ---------------------------------------------------------------------------


class TestLoadEvaluator:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_evaluator("/nonexistent/spec.json")

    def test_loads_valid_spec(self, sample_eval_spec):
        evaluator = load_evaluator(sample_eval_spec)
        assert isinstance(evaluator, SpecEvaluator)


# ---------------------------------------------------------------------------
# SpecEvaluator — construction and validation
# ---------------------------------------------------------------------------


class TestSpecEvaluatorConstruction:
    def test_loads_fields(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        assert "qualification" in ev.fields
        assert "score" in ev.fields

    def test_weight_mismatch_warns(self, tmp_path):
        spec = {
            "output_fields": {
                "f": {"type": "text", "weight": 50, "eval_mode": "non_empty"}
            },
            "structure_weight": 20,
            "total_points": 200,  # deliberate mismatch
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SpecEvaluator(str(path))
            assert len(w) >= 1
            assert "mismatch" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# SpecEvaluator._score_enum
# ---------------------------------------------------------------------------


class TestScoreEnum:
    def test_exact_match(self):
        config = {"weight": 30, "values": ["hot", "warm", "cold"]}
        assert SpecEvaluator._score_enum("hot", "hot", config) == 30.0

    def test_case_insensitive_match(self):
        config = {"weight": 30, "values": ["hot", "warm", "cold"]}
        assert SpecEvaluator._score_enum("HOT", "hot", config) == 30.0

    def test_mismatch_no_partial(self):
        config = {"weight": 30, "values": ["hot", "warm", "cold"]}
        assert SpecEvaluator._score_enum("warm", "hot", config) == 0.0

    def test_partial_credit(self):
        config = {
            "weight": 30,
            "values": ["hot", "warm", "cold"],
            "partial_credit": True,
            "partial_score": 6,
        }
        assert SpecEvaluator._score_enum("warm", "hot", config) == 6

    def test_invalid_value_no_partial(self):
        config = {
            "weight": 30,
            "values": ["hot", "warm", "cold"],
            "partial_credit": True,
            "partial_score": 6,
        }
        assert SpecEvaluator._score_enum("invalid", "hot", config) == 0.0

    def test_none_actual(self):
        config = {"weight": 30, "values": ["hot"]}
        assert SpecEvaluator._score_enum(None, "hot", config) == 0.0

    def test_none_expected(self):
        config = {"weight": 30, "values": ["hot"]}
        assert SpecEvaluator._score_enum("hot", None, config) == 0.0


# ---------------------------------------------------------------------------
# SpecEvaluator._score_number
# ---------------------------------------------------------------------------


class TestScoreNumber:
    def test_exact_match(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(50, 50, config) == 20.0

    def test_within_tolerance(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(55, 50, config) == 20.0

    def test_within_double_tolerance(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(65, 50, config) == 10.0

    def test_beyond_double_tolerance(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(80, 50, config) == 0.0

    def test_tolerance_bands(self):
        config = {
            "weight": 20,
            "tolerance_bands": [
                {"within": 5, "score_pct": 1.0},
                {"within": 10, "score_pct": 0.8},
                {"within": 15, "score_pct": 0.5},
            ],
        }
        assert SpecEvaluator._score_number(50, 50, config) == 20.0
        assert SpecEvaluator._score_number(54, 50, config) == 20.0
        assert SpecEvaluator._score_number(57, 50, config) == 16.0
        assert SpecEvaluator._score_number(62, 50, config) == 10.0
        assert SpecEvaluator._score_number(70, 50, config) == 0.0

    def test_non_numeric_actual(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number("abc", 50, config) == 0.0

    def test_none_values(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(None, None, config) == 20.0

    def test_negative_numbers(self):
        config = {"weight": 20, "tolerance": 10}
        assert SpecEvaluator._score_number(-5, -5, config) == 20.0


# ---------------------------------------------------------------------------
# SpecEvaluator._score_text — all modes
# ---------------------------------------------------------------------------


class TestScoreText:
    def test_non_empty_mode_with_content(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert ev._score_text("Some text", None, config) == 15.0

    def test_non_empty_mode_empty_string(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert ev._score_text("", None, config) == 0.0

    def test_non_empty_mode_whitespace_only(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert ev._score_text("   ", None, config) == 0.0

    def test_non_empty_mode_none(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert ev._score_text(None, None, config) == 0.0

    def test_skip_mode(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "skip"}
        assert ev._score_text("text", None, config) == 0.0

    def test_unknown_mode(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "unknown_mode"}
        assert ev._score_text("text", None, config) == 0.0


class TestTextSimilarity:
    def test_identical_text(self):
        score = SpecEvaluator._text_similarity(
            "Large budget enterprise company", "Large budget enterprise company"
        )
        assert score == pytest.approx(1.0)

    def test_high_overlap(self):
        score = SpecEvaluator._text_similarity(
            "Large budget enterprise firm",
            "Large budget enterprise company",
        )
        assert score > 0.6

    def test_no_overlap(self):
        score = SpecEvaluator._text_similarity(
            "xyz abc 123", "Large budget enterprise company"
        )
        assert score < 0.2

    def test_empty_actual(self):
        assert SpecEvaluator._text_similarity("", "some expected text") == 0.0

    def test_empty_expected(self):
        assert SpecEvaluator._text_similarity("some text", "") == 1.0

    def test_both_empty(self):
        assert SpecEvaluator._text_similarity("", "") == 1.0

    def test_none_values(self):
        assert SpecEvaluator._text_similarity(None, "text") == 0.0
        assert SpecEvaluator._text_similarity("text", None) == 1.0

    def test_very_short_actual_penalized(self):
        score = SpecEvaluator._text_similarity(
            "a",
            "This is a really long expected text with many details and specifics",
        )
        assert score < 0.3

    def test_similarity_mode_integration(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "similarity"}
        score = ev._score_text(
            "Large budget enterprise lead",
            "Large budget enterprise company",
            config,
        )
        assert 0 < score <= 15.0


class TestTextKeywordCoverage:
    def test_full_coverage(self):
        score = SpecEvaluator._text_keyword_coverage(
            "The company has a large budget and is enterprise",
            "large budget enterprise",
        )
        assert score == pytest.approx(1.0)

    def test_partial_coverage(self):
        score = SpecEvaluator._text_keyword_coverage(
            "The company has a large budget",
            "large budget enterprise premium",
        )
        assert 0.3 < score < 0.8

    def test_no_coverage(self):
        score = SpecEvaluator._text_keyword_coverage(
            "xyz abc 123",
            "large budget enterprise",
        )
        assert score == 0.0

    def test_empty_actual(self):
        assert SpecEvaluator._text_keyword_coverage("", "keywords here") == 0.0

    def test_empty_expected(self):
        assert SpecEvaluator._text_keyword_coverage("some text", "") == 1.0

    def test_stopwords_only_expected(self):
        score = SpecEvaluator._text_keyword_coverage("text", "the a an is")
        assert score == 1.0

    def test_keyword_mode_integration(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        config = {"weight": 15, "eval_mode": "keyword_coverage"}
        score = ev._score_text(
            "Large budget enterprise lead",
            "large budget enterprise",
            config,
        )
        assert score == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# SpecEvaluator.evaluate_output
# ---------------------------------------------------------------------------


class TestEvaluateOutput:
    def test_perfect_score(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        output = {
            "qualification": "hot",
            "score": 85,
            "reasoning": "Good match",
            "is_enterprise": True,
        }
        expected = {
            "qualification": "hot",
            "score": 85,
            "reasoning": "Good match",
            "is_enterprise": True,
        }
        scores = ev.evaluate_output(output, expected)
        assert scores["total"] > 0
        assert scores["qualification"] == 30.0
        assert scores["is_enterprise"] == 15.0
        assert scores["reasoning"] == 15.0

    def test_all_wrong(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        output = {
            "qualification": "cold",
            "score": 0,
            "reasoning": "",
            "is_enterprise": False,
        }
        expected = {
            "qualification": "hot",
            "score": 85,
            "reasoning": "x",
            "is_enterprise": True,
        }
        scores = ev.evaluate_output(output, expected)
        assert scores["qualification"] < 30.0
        assert scores["reasoning"] == 0.0
        assert scores["is_enterprise"] == 0.0

    def test_missing_fields(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        scores = ev.evaluate_output({}, {"qualification": "hot"})
        assert scores["structure"] == 0.0

    def test_structure_weighted_by_importance(self, sample_eval_spec):
        """Structure score should weight critical fields more than minor ones."""
        ev = SpecEvaluator(sample_eval_spec)
        output_with_critical = {"qualification": "hot"}
        output_with_minor = {"is_enterprise": True}
        score_critical = ev.evaluate_output(output_with_critical, {})["structure"]
        score_minor = ev.evaluate_output(output_with_minor, {})["structure"]
        assert score_critical > score_minor

    def test_structure_zero_is_valid(self, tmp_path):
        """Numeric 0 should count as present (not empty)."""
        spec = {
            "output_fields": {
                "count": {
                    "type": "number",
                    "weight": 50,
                    "tolerance": 5,
                    "range": [0, 100],
                }
            },
            "structure_weight": 20,
            "total_points": 70,
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        ev = SpecEvaluator(str(path))
        scores = ev.evaluate_output({"count": 0}, {"count": 0})
        assert scores["structure"] == 20.0

    def test_total_non_negative(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        scores = ev.evaluate_output({}, {})
        assert scores["total"] >= 0.0

    def test_type_correctness_penalty_present(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        scores = ev.evaluate_output(
            {
                "qualification": "hot",
                "score": 85,
                "reasoning": "ok",
                "is_enterprise": True,
            },
            {
                "qualification": "hot",
                "score": 85,
                "reasoning": "ok",
                "is_enterprise": True,
            },
        )
        assert "type_correctness_penalty" in scores
        assert scores["type_correctness_penalty"] == 0.0

    def test_type_error_penalized(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        scores = ev.evaluate_output(
            {
                "qualification": "hot",
                "score": "not_a_number",
                "reasoning": "ok",
                "is_enterprise": True,
            },
            {
                "qualification": "hot",
                "score": 85,
                "reasoning": "ok",
                "is_enterprise": True,
            },
        )
        assert scores["type_correctness_penalty"] < 0.0


# ---------------------------------------------------------------------------
# SpecEvaluator.evaluate_batch
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    def test_empty_batch(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        result = ev.evaluate_batch([])
        assert result["avg_total"] == 0.0
        assert result["count"] == 0

    def test_batch_averages(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        results = [
            {
                "output": {
                    "qualification": "hot",
                    "score": 85,
                    "reasoning": "x",
                    "is_enterprise": True,
                },
                "expected": {
                    "qualification": "hot",
                    "score": 85,
                    "reasoning": "x",
                    "is_enterprise": True,
                },
            },
            {
                "output": {
                    "qualification": "cold",
                    "score": 0,
                    "reasoning": "",
                    "is_enterprise": False,
                },
                "expected": {
                    "qualification": "hot",
                    "score": 85,
                    "reasoning": "y",
                    "is_enterprise": True,
                },
            },
        ]
        batch_result = ev.evaluate_batch(results)
        assert batch_result["count"] == 2
        assert "avg_total" in batch_result
        assert "individual_scores" in batch_result

    def test_pre_scored_items(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        results = [
            {"output": {}, "expected": {}, "score": {"total": 50.0, "structure": 10.0}},
        ]
        batch_result = ev.evaluate_batch(results)
        assert batch_result["avg_total"] == 50.0

    def test_batch_queues_pre_scored_items_for_judge(self, tmp_path):
        """Pre-scored items missing llm_judge should be queued for batch judging."""
        spec = {
            "output_fields": {
                "result": {"type": "text", "weight": 50, "eval_mode": "non_empty"}
            },
            "structure_weight": 20,
            "total_points": 100,
            "llm_judge_weight": 30,
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        ev = SpecEvaluator(str(path), llm_judge_model="test-model")

        pre_scored = {
            "output": {"result": "test"},
            "expected": {"result": "test"},
            "input": {"q": "test"},
            "score": {"total": 70.0, "structure": 20.0, "result": 50.0},
        }
        with patch.object(ev, "_score_with_llm_judge", return_value=0.8) as mock_judge:
            result = ev.evaluate_batch([pre_scored])
            mock_judge.assert_called_once()
        assert result["individual_scores"][0]["llm_judge"] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# SpecEvaluator.get_dimension_labels / get_max_scores
# ---------------------------------------------------------------------------


class TestDimensionLabelsAndMaxScores:
    def test_labels_include_structure(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        labels = ev.get_dimension_labels()
        label_names = [lbl[0] for lbl in labels]
        assert "Structure" in label_names

    def test_labels_include_type_correctness(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        labels = ev.get_dimension_labels()
        label_names = [lbl[0] for lbl in labels]
        assert "Type Correctness" in label_names

    def test_max_scores(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        maxes = ev.get_max_scores()
        assert maxes["avg_structure"] == 20.0
        assert maxes["avg_qualification"] == 30.0
        assert maxes["avg_type_correctness_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Type correctness
# ---------------------------------------------------------------------------


class TestTypeCorrectness:
    def test_all_correct_types(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_type_correctness(
            {
                "qualification": "hot",
                "score": 85,
                "reasoning": "text here",
                "is_enterprise": True,
            }
        )
        assert penalty == 0.0

    def test_wrong_number_type(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_type_correctness(
            {
                "qualification": "hot",
                "score": "eighty-five",
                "reasoning": "ok",
                "is_enterprise": True,
            }
        )
        assert penalty == -2.0

    def test_multiple_type_errors(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_type_correctness(
            {
                "qualification": "invalid_value",
                "score": "not_number",
                "reasoning": 123,
                "is_enterprise": "maybe",
            }
        )
        assert penalty <= -6.0

    def test_cap_at_minus_10(self, tmp_path):
        spec = {
            "output_fields": {
                f"f{i}": {"type": "number", "weight": 5, "tolerance": 1}
                for i in range(10)
            },
            "structure_weight": 10,
            "total_points": 60,
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        ev = SpecEvaluator(str(path))
        penalty = ev._check_type_correctness(
            {f"f{i}": "not_a_number" for i in range(10)}
        )
        assert penalty == -10.0

    def test_none_values_not_penalized(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_type_correctness({})
        assert penalty == 0.0

    def test_bool_accepts_01(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_type_correctness(
            {
                "is_enterprise": 1,
            }
        )
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# Tool scoring
# ---------------------------------------------------------------------------


class TestToolScoring:
    def test_no_trace(self, sample_eval_spec_with_tools):
        ev = SpecEvaluator(sample_eval_spec_with_tools)
        assert ev._score_tool_usage(None) == 0.0
        assert ev._score_tool_usage([]) == 0.0

    def test_no_config_returns_1(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        assert ev._score_tool_usage([{"name": "any"}]) == 1.0

    def test_completeness(self, sample_eval_spec_with_tools):
        ev = SpecEvaluator(sample_eval_spec_with_tools)
        trace = [
            {
                "name": "search",
                "args": {"query_type": "web"},
                "result": {"results": "data"},
            },
            {"name": "analyze", "args": {"data": "data"}, "result": {}},
        ]
        score = ev._score_tool_usage(trace)
        assert score > 0.0

    def test_tool_arguments_scoring(self):
        constraints = {"search": {"query_type": ["web", "local"]}}
        trace = [
            {"name": "search", "args": {"query_type": "web"}},
            {"name": "search", "args": {"query_type": "invalid"}},
        ]
        score = SpecEvaluator._score_tool_arguments(trace, constraints)
        assert score == 0.5

    def test_tool_chaining_scoring(self):
        deps = [
            {
                "from_tool": "search",
                "from_field": "data",
                "to_tool": "analyze",
                "to_param": "input",
            },
        ]
        trace = [
            {"name": "search", "args": {}, "result": {"data": "value"}},
            {"name": "analyze", "args": {"input": "value"}, "result": {}},
        ]
        score = SpecEvaluator._score_tool_chaining(trace, deps)
        assert score == 1.0

    def test_chaining_mismatch(self):
        deps = [
            {
                "from_tool": "search",
                "from_field": "data",
                "to_tool": "analyze",
                "to_param": "input",
            },
        ]
        trace = [
            {"name": "search", "args": {}, "result": {"data": "value"}},
            {"name": "analyze", "args": {"input": "wrong"}, "result": {}},
        ]
        score = SpecEvaluator._score_tool_chaining(trace, deps)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


class TestCrossFieldConsistency:
    def test_no_rules_no_penalty(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        penalty = ev._check_cross_field_consistency(
            {"qualification": "hot", "score": 85},
            {"qualification": "hot", "score": 85},
        )
        assert penalty <= 0.0

    def test_inferred_rules(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        rules = ev._infer_consistency_rules({"qualification": "hot", "score": 85})
        assert len(rules) > 0
        assert rules[0]["type"] == "correlation"

    def test_ordering_rule_violation(self, tmp_path):
        spec = {
            "output_fields": {
                "min_val": {
                    "type": "number",
                    "weight": 20,
                    "tolerance": 5,
                    "range": [0, 100],
                },
                "max_val": {
                    "type": "number",
                    "weight": 20,
                    "tolerance": 5,
                    "range": [0, 100],
                },
            },
            "structure_weight": 20,
            "total_points": 60,
            "consistency_rules": [
                {
                    "field_a": "min_val",
                    "field_b": "max_val",
                    "type": "ordering",
                    "operator": "<=",
                    "penalty": 5.0,
                }
            ],
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        ev = SpecEvaluator(str(path))
        penalty = ev._check_cross_field_consistency({"min_val": 80, "max_val": 20}, {})
        assert penalty == -5.0

    def test_ordering_rule_satisfied(self, tmp_path):
        spec = {
            "output_fields": {
                "min_val": {
                    "type": "number",
                    "weight": 20,
                    "tolerance": 5,
                    "range": [0, 100],
                },
                "max_val": {
                    "type": "number",
                    "weight": 20,
                    "tolerance": 5,
                    "range": [0, 100],
                },
            },
            "structure_weight": 20,
            "total_points": 60,
            "consistency_rules": [
                {
                    "field_a": "min_val",
                    "field_b": "max_val",
                    "type": "ordering",
                    "operator": "<=",
                    "penalty": 5.0,
                }
            ],
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        ev = SpecEvaluator(str(path))
        penalty = ev._check_cross_field_consistency({"min_val": 20, "max_val": 80}, {})
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# _is_contradictory
# ---------------------------------------------------------------------------


class TestIsContradictory:
    def test_high_number_worst_enum(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        result = ev._is_contradictory("score", 90, "qualification", "cold")
        assert result is True

    def test_consistent_values(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        result = ev._is_contradictory("score", 90, "qualification", "hot")
        assert result is False

    def test_non_numeric_value(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        result = ev._is_contradictory("score", "abc", "qualification", "hot")
        assert result is False

    def test_unknown_enum_value(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        result = ev._is_contradictory("score", 90, "qualification", "unknown")
        assert result is False


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


class TestLlmJudge:
    @patch("overclaw.utils.llm.litellm")
    def test_judge_success(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="test-model")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(
            {
                "semantic_correctness": 8,
                "internal_consistency": 7,
                "reasoning_quality": 9,
            }
        )
        mock_litellm.completion.return_value = mock_resp

        score = ev._score_with_llm_judge(
            {"input": "test"}, {"expected": "x"}, {"output": "y"}
        )
        assert 0.0 <= score <= 1.0

    @patch("overclaw.utils.llm.litellm")
    def test_judge_with_policy(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(
            sample_eval_spec,
            llm_judge_model="test-model",
            policy_judge_rubric="test rubric",
        )
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(
            {
                "semantic_correctness": 8,
                "internal_consistency": 7,
                "reasoning_quality": 9,
                "policy_compliance": 8,
            }
        )
        mock_litellm.completion.return_value = mock_resp

        score = ev._score_with_llm_judge({"input": "test"}, {}, {})
        assert 0.0 <= score <= 1.0

    @patch("overclaw.utils.llm.litellm")
    def test_judge_retries_on_parse_failure(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="model")
        fail_resp = MagicMock()
        fail_resp.choices = [MagicMock()]
        fail_resp.choices[0].message.content = "not json"

        success_resp = MagicMock()
        success_resp.choices = [MagicMock()]
        success_resp.choices[0].message.content = json.dumps(
            {
                "semantic_correctness": 8,
                "internal_consistency": 7,
                "reasoning_quality": 9,
            }
        )
        mock_litellm.completion.side_effect = [fail_resp, success_resp]

        score = ev._score_with_llm_judge({"input": "test"}, {}, {})
        assert score != _JUDGE_FALLBACK_SCORE
        assert 0.0 <= score <= 1.0

    @patch("overclaw.utils.llm.litellm")
    def test_judge_parse_failure_returns_fallback(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="model")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "not json"
        mock_litellm.completion.return_value = mock_resp

        score = ev._score_with_llm_judge({}, {}, {})
        assert score == _JUDGE_FALLBACK_SCORE

    @patch("overclaw.utils.llm.litellm")
    def test_judge_exception_retries_then_fallback(
        self, mock_litellm, sample_eval_spec
    ):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="model")
        mock_litellm.completion.side_effect = RuntimeError("boom")

        score = ev._score_with_llm_judge({}, {}, {})
        assert score == _JUDGE_FALLBACK_SCORE
        assert mock_litellm.completion.call_count == 3


# ---------------------------------------------------------------------------
# spec_generator
# ---------------------------------------------------------------------------


class TestSpecGenerator:
    def test_auto_judge_weight_with_text_fields(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "rating": {"type": "enum", "values": ["good", "bad"]},
                "reason": {"type": "text", "description": "Explanation"},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "rating": {"importance": "critical"},
                    "reason": {"importance": "important"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec.get("llm_judge_weight", 0) == 10
        assert spec["total_points"] == 100

    def test_auto_judge_weight_with_policy(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "rating": {"type": "enum", "values": ["good", "bad"]},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "rating": {"importance": "critical"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis, policy_data={"rules": "test"})
        assert spec.get("llm_judge_weight", 0) == 10

    def test_no_judge_weight_without_text_or_policy(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "value": {"type": "number", "range": [0, 100]},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "value": {"importance": "important"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec.get("llm_judge_weight", 0) == 0

    def test_text_default_eval_mode_similarity(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "reason": {"type": "text", "description": "Explanation"},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "reason": {"importance": "important"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec["output_fields"]["reason"]["eval_mode"] == "similarity"

    def test_auto_consistency_rules_generated(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "quality": {"type": "enum", "values": ["high", "medium", "low"]},
                "score_val": {"type": "number", "range": [0, 100]},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "quality": {"importance": "important"},
                    "score_val": {"importance": "important"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        rules = spec.get("consistency_rules", [])
        assert len(rules) > 0
        assert any(r["type"] == "correlation" for r in rules)

    def test_ordering_rules_for_min_max(self):
        from overclaw.setup.spec_generator import generate_spec_from_proposal

        analysis = {
            "output_schema": {
                "min_price": {"type": "number", "range": [0, 1000000]},
                "max_price": {"type": "number", "range": [0, 1000000]},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "min_price": {"importance": "important"},
                    "max_price": {"importance": "important"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        rules = spec.get("consistency_rules", [])
        ordering_rules = [r for r in rules if r["type"] == "ordering"]
        assert len(ordering_rules) >= 1
        assert ordering_rules[0]["operator"] == "<="
