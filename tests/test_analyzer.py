"""Tests for overclaw.optimize.analyzer — code analysis and generation helpers."""

from __future__ import annotations


from overclaw.optimize.analyzer import (
    _build_fingerprints,
    _detect_agent_model,
    _extract_code_and_analysis,
    _find_weakest_dimension,
    _format_dimension_deltas,
    _format_failed_attempts,
    _format_fixed_elements,
    _format_optimizable_elements,
    _format_per_case_results,
    _format_score_breakdown,
    _format_scoring_mechanics,
    _format_successful_changes,
    _format_tool_usage_analysis,
    _matches_fingerprint,
    _measure_system_prompt,
)


# ---------------------------------------------------------------------------
# _measure_system_prompt
# ---------------------------------------------------------------------------


class TestMeasureSystemPrompt:
    def test_with_triple_double_quotes(self):
        code = 'SYSTEM_PROMPT = """Hello\\nWorld\\nEnd"""'
        chars, lines = _measure_system_prompt(code)
        assert chars > 0
        assert lines >= 1

    def test_with_triple_single_quotes(self):
        code = "SYSTEM_PROMPT = '''Line1\\nLine2'''"
        chars, lines = _measure_system_prompt(code)
        assert chars > 0

    def test_no_system_prompt(self):
        code = "def run(x): return x"
        chars, lines = _measure_system_prompt(code)
        assert chars == 0
        assert lines == 0


# ---------------------------------------------------------------------------
# _format_scoring_mechanics
# ---------------------------------------------------------------------------


class TestFormatScoringMechanics:
    def test_none_spec(self):
        result = _format_scoring_mechanics(None)
        assert "no evaluation spec" in result.lower()

    def test_with_fields(self):
        spec = {
            "structure_weight": 20,
            "output_fields": {
                "status": {"type": "enum", "weight": 30, "values": ["a", "b"]},
                "score": {"type": "number", "weight": 20, "tolerance": 10},
                "text": {"type": "text", "weight": 15, "eval_mode": "non_empty"},
                "flag": {"type": "boolean", "weight": 15},
            },
        }
        result = _format_scoring_mechanics(spec)
        assert "Structure" in result
        assert "Status" in result
        assert "enum" in result
        assert "number" in result
        assert "text" in result.lower()
        assert "boolean" in result.lower()

    def test_with_partial_credit(self):
        spec = {
            "structure_weight": 20,
            "output_fields": {
                "x": {
                    "type": "enum",
                    "weight": 30,
                    "values": ["a"],
                    "partial_credit": True,
                    "partial_score": 6,
                },
            },
        }
        result = _format_scoring_mechanics(spec)
        assert "6" in result

    def test_with_tolerance_bands(self):
        spec = {
            "structure_weight": 20,
            "output_fields": {
                "x": {
                    "type": "number",
                    "weight": 20,
                    "tolerance_bands": [{"within": 5, "score_pct": 1.0}],
                },
            },
        }
        result = _format_scoring_mechanics(spec)
        assert "±5" in result or "5" in result

    def test_with_tool_and_judge_weights(self):
        spec = {
            "structure_weight": 20,
            "output_fields": {},
            "tool_usage_weight": 10,
            "llm_judge_weight": 15,
        }
        result = _format_scoring_mechanics(spec)
        assert "Tool" in result
        assert "Judge" in result


# ---------------------------------------------------------------------------
# _format_per_case_results
# ---------------------------------------------------------------------------


class TestFormatPerCaseResults:
    def test_empty(self):
        result = _format_per_case_results([], None)
        assert "no results" in result.lower()

    def test_with_cases(self):
        cases = [
            {
                "input": {"name": "test"},
                "output": {"status": "hot"},
                "score": {"total": 75, "status": 30, "structure": 20},
            },
        ]
        spec = {
            "output_fields": {
                "status": {"type": "enum", "weight": 30, "values": ["hot", "cold"]}
            },
            "structure_weight": 20,
        }
        result = _format_per_case_results(cases, spec)
        assert "Case 1" in result

    def test_case_fraction(self):
        cases = [
            {"input": {"i": i}, "output": {}, "score": {"total": i * 10}}
            for i in range(10)
        ]
        result_full = _format_per_case_results(cases, None, case_fraction=1.0)
        result_partial = _format_per_case_results(cases, None, case_fraction=0.5)
        assert len(result_partial) <= len(result_full)


# ---------------------------------------------------------------------------
# _format_tool_usage_analysis
# ---------------------------------------------------------------------------


class TestFormatToolUsageAnalysis:
    def test_empty(self):
        result = _format_tool_usage_analysis([])
        assert "no tool data" in result.lower()

    def test_with_traces(self):
        cases = [
            {
                "tool_trace": [
                    {"name": "search", "args": {"q": "test"}, "result": {}},
                    {"name": "analyze", "args": {}, "result": {}},
                ],
            },
        ]
        result = _format_tool_usage_analysis(cases)
        assert "search" in result
        assert "analyze" in result

    def test_with_errors(self):
        cases = [
            {"tool_trace": [{"name": "bad_tool", "args": {}, "error": "timeout"}]},
        ]
        result = _format_tool_usage_analysis(cases)
        assert "timeout" in result


# ---------------------------------------------------------------------------
# _format_score_breakdown
# ---------------------------------------------------------------------------


class TestFormatScoreBreakdown:
    def test_empty(self):
        result = _format_score_breakdown({}, None)
        assert "no breakdown" in result.lower()

    def test_with_averages(self):
        evaluation = {"avg_structure": 18.0, "avg_status": 25.0, "avg_total": 43.0}
        spec = {"structure_weight": 20, "output_fields": {"status": {"weight": 30}}}
        result = _format_score_breakdown(evaluation, spec)
        assert "Structure" in result
        assert "Status" in result


# ---------------------------------------------------------------------------
# _find_weakest_dimension
# ---------------------------------------------------------------------------


class TestFindWeakestDimension:
    def test_no_spec(self):
        name, score, max_val = _find_weakest_dimension({}, None)
        assert name == "unknown"

    def test_finds_weakest(self):
        spec = {
            "structure_weight": 20,
            "output_fields": {
                "a": {"weight": 30},
                "b": {"weight": 50},
            },
        }
        evaluation = {
            "avg_structure": 20.0,
            "avg_a": 30.0,
            "avg_b": 10.0,  # worst gap
        }
        name, score, max_val = _find_weakest_dimension(evaluation, spec)
        assert name == "B"
        assert max_val == 50.0


# ---------------------------------------------------------------------------
# _detect_agent_model
# ---------------------------------------------------------------------------


class TestDetectAgentModel:
    def test_detects_model_name(self):
        code = 'MODEL = "gpt-5.4-mini"'
        name, capability = _detect_agent_model(code)
        assert name == "gpt-5.4-mini"
        assert capability == "lightweight"

    def test_detects_pro_model(self):
        code = 'model = "gpt-5.4-pro"'
        name, capability = _detect_agent_model(code)
        assert capability == "very capable"

    def test_no_model(self):
        code = "def run(x): return x"
        name, capability = _detect_agent_model(code)
        assert name == "unknown"
        assert capability == "capable"


# ---------------------------------------------------------------------------
# _extract_code_and_analysis
# ---------------------------------------------------------------------------


class TestExtractCodeAndAnalysis:
    def test_json_and_code_blocks(self):
        text = (
            '```json\n{"analysis": "root cause", "suggestions": ["fix A"]}\n```\n'
            "```python\ndef run(input_data):\n    return {}\n```"
        )
        analysis, suggestions, code = _extract_code_and_analysis(
            text, "def run(input_data):\n    pass"
        )
        assert analysis == "root cause"
        assert "fix A" in suggestions
        assert code is not None
        assert "def run" in code

    def test_no_code_block(self):
        text = '{"analysis": "issue"}'
        _, _, code = _extract_code_and_analysis(text, "")
        assert code is None or code == ""  # no code fences

    def test_embedded_json(self):
        text = 'Result: {"root_cause": "bad logic", "changes": [{"action": "fix X"}]}'
        analysis, suggestions, _ = _extract_code_and_analysis(text, "")
        assert "bad logic" in analysis or analysis == ""


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


class TestBuildFingerprints:
    def test_with_function(self):
        code = "def run(input_data):\n    return {}"
        fps = _build_fingerprints(code)
        assert len(fps) > 0

    def test_empty_code(self):
        assert _build_fingerprints("") == []

    def test_no_function(self):
        code = "x = 1\ny = 2"
        fps = _build_fingerprints(code)
        assert "run" in fps  # fallback


class TestMatchesFingerprint:
    def test_matches(self):
        assert _matches_fingerprint("def run(input_data): pass", ["def run"]) is True

    def test_no_match(self):
        assert _matches_fingerprint("def other(): pass", ["def run"]) is False

    def test_empty_fingerprints(self):
        assert _matches_fingerprint("x" * 200, []) is True

    def test_short_text_no_fps(self):
        assert _matches_fingerprint("short", []) is False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatFixedElements:
    def test_none_spec(self):
        result = _format_fixed_elements(None)
        assert "Tool implementation" in result

    def test_with_elements(self):
        spec = {"fixed_elements": ["element A", "element B"]}
        result = _format_fixed_elements(spec)
        assert "element A" in result
        assert "element B" in result

    def test_empty_elements(self):
        result = _format_fixed_elements({"fixed_elements": []})
        assert "Tool implementation" in result


class TestFormatOptimizableElements:
    def test_none_spec(self):
        result = _format_optimizable_elements(None)
        assert "Prompts" in result

    def test_with_elements(self):
        spec = {"optimizable_elements": ["prompt A"]}
        result = _format_optimizable_elements(spec)
        assert "prompt A" in result


class TestFormatDimensionDeltas:
    def test_empty(self):
        assert _format_dimension_deltas({}) == ""

    def test_gains_and_losses(self):
        deltas = {"structure": 5.0, "score": -3.0}
        result = _format_dimension_deltas(deltas)
        assert "Gains" in result
        assert "Losses" in result


class TestFormatFailedAttempts:
    def test_none(self):
        assert _format_failed_attempts(None) == "(none yet)"

    def test_with_attempts(self):
        attempts = [
            {"reason": "regression", "score": 45.0, "suggestions": ["try X"]},
        ]
        result = _format_failed_attempts(attempts)
        assert "regression" in result
        assert "try X" in result


class TestFormatSuccessfulChanges:
    def test_none(self):
        assert _format_successful_changes(None) == "(none yet)"

    def test_with_changes(self):
        changes = [
            {"improvement": "+5 pts", "suggestions": ["changed prompt"]},
        ]
        result = _format_successful_changes(changes)
        assert "+5 pts" in result
