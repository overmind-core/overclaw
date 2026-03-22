"""Tests for overclaw.optimize.evaluator — spec-driven scoring."""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from overclaw.optimize.evaluator import (
    SpecEvaluator,
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
# SpecEvaluator._score_text
# ---------------------------------------------------------------------------


class TestScoreText:
    def test_non_empty_mode_with_content(self):
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert SpecEvaluator._score_text("Some text", config) == 15.0

    def test_non_empty_mode_empty_string(self):
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert SpecEvaluator._score_text("", config) == 0.0

    def test_non_empty_mode_whitespace_only(self):
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert SpecEvaluator._score_text("   ", config) == 0.0

    def test_non_empty_mode_none(self):
        config = {"weight": 15, "eval_mode": "non_empty"}
        assert SpecEvaluator._score_text(None, config) == 0.0

    def test_skip_mode(self):
        config = {"weight": 15, "eval_mode": "skip"}
        assert SpecEvaluator._score_text("text", config) == 0.0

    def test_unknown_mode(self):
        config = {"weight": 15, "eval_mode": "unknown_mode"}
        assert SpecEvaluator._score_text("text", config) == 0.0


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

    def test_structure_partial(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        output = {"qualification": "hot", "score": 50}
        expected = {"qualification": "hot", "score": 50}
        scores = ev.evaluate_output(output, expected)
        assert 0 < scores["structure"] < 20.0

    def test_total_non_negative(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        scores = ev.evaluate_output({}, {})
        assert scores["total"] >= 0.0


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


# ---------------------------------------------------------------------------
# SpecEvaluator.get_dimension_labels / get_max_scores
# ---------------------------------------------------------------------------


class TestDimensionLabelsAndMaxScores:
    def test_labels_include_structure(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        labels = ev.get_dimension_labels()
        label_names = [lbl[0] for lbl in labels]
        assert "Structure" in label_names

    def test_max_scores(self, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec)
        maxes = ev.get_max_scores()
        assert maxes["avg_structure"] == 20.0
        assert maxes["avg_qualification"] == 30.0


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
    @patch("overclaw.core.litellm_params.litellm")
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

    @patch("overclaw.core.litellm_params.litellm")
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

    @patch("overclaw.core.litellm_params.litellm")
    def test_judge_parse_failure_returns_fallback(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="model")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "not json"
        mock_litellm.completion.return_value = mock_resp

        score = ev._score_with_llm_judge({}, {}, {})
        assert score == 0.5

    @patch("overclaw.core.litellm_params.litellm")
    def test_judge_exception_returns_fallback(self, mock_litellm, sample_eval_spec):
        ev = SpecEvaluator(sample_eval_spec, llm_judge_model="model")
        mock_litellm.completion.side_effect = RuntimeError("boom")

        score = ev._score_with_llm_judge({}, {}, {})
        assert score == 0.5
