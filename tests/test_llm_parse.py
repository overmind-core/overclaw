"""Tests for the unified LLM-JSON-parse helper.

These cases were drawn from real LLM-response shapes the codebase
previously handled with six different inline parsers.  Locking each
case down in tests means future refactors of
``overmind/utils/llm_parse.py`` cannot silently regress one of the
historical call sites.
"""

from __future__ import annotations

import json

import pytest

from overmind.utils.llm_parse import (
    LLMParseError,
    parse_json_object,
    repair_json_string,
)


class TestHappyPath:
    def test_plain_object(self) -> None:
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_plain_array(self) -> None:
        assert parse_json_object("[1, 2, 3]") == [1, 2, 3]

    def test_with_surrounding_whitespace(self) -> None:
        assert parse_json_object('   {"a": 1}   ') == {"a": 1}


class TestFencedBlocks:
    def test_json_fenced(self) -> None:
        text = 'Here you go:\n```json\n{"a": 1}\n```\nThanks!'
        assert parse_json_object(text) == {"a": 1}

    def test_unlabelled_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert parse_json_object(text) == {"a": 1}

    def test_fenced_with_trailing_comma(self) -> None:
        text = '```json\n{"a": 1, "b": 2,}\n```'
        assert parse_json_object(text) == {"a": 1, "b": 2}


class TestProseSurroundingJSON:
    def test_prose_before_object(self) -> None:
        text = 'Sure, here is the result: {"a": 1}'
        assert parse_json_object(text) == {"a": 1}

    def test_prose_before_and_after(self) -> None:
        text = 'Output:\n{"a": 1}\nThat\'s all.'
        assert parse_json_object(text) == {"a": 1}

    def test_only_array_surrounded_by_prose(self) -> None:
        text = "Here is the list: [1, 2, 3] please use it."
        assert parse_json_object(text) == [1, 2, 3]


class TestRepairCases:
    def test_trailing_comma_in_object(self) -> None:
        assert parse_json_object('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self) -> None:
        assert parse_json_object("[1, 2, 3,]") == [1, 2, 3]

    def test_unescaped_newline_in_string(self) -> None:
        text = '{"msg": "line1\nline2"}'
        parsed = parse_json_object(text)
        assert parsed == {"msg": "line1\nline2"}

    def test_unescaped_tab_in_string(self) -> None:
        text = '{"msg": "a\tb"}'
        parsed = parse_json_object(text)
        assert parsed == {"msg": "a\tb"}

    def test_single_quoted_keys_and_values(self) -> None:
        # Fallback path: replace single quotes with double quotes.
        text = "{'a': 'hello'}"
        assert parse_json_object(text) == {"a": "hello"}


class TestNestedBraces:
    def test_nested_object(self) -> None:
        text = '{"a": {"b": {"c": 1}}}'
        assert parse_json_object(text) == {"a": {"b": {"c": 1}}}

    def test_nested_object_with_prose(self) -> None:
        text = 'Result: {"a": {"b": {"c": 1}}} done.'
        assert parse_json_object(text) == {"a": {"b": {"c": 1}}}


class TestFailureModes:
    def test_empty_string_raises_by_default(self) -> None:
        with pytest.raises(LLMParseError):
            parse_json_object("")

    def test_none_input_raises_by_default(self) -> None:
        with pytest.raises(LLMParseError):
            parse_json_object(None)  # type: ignore[arg-type]

    def test_no_braces_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_json_object("just prose, no JSON anywhere")

    def test_empty_string_default_returns_default(self) -> None:
        assert parse_json_object("", on_fail="default", default={}) == {}

    def test_no_braces_default_returns_default(self) -> None:
        sentinel = {"original": True}
        assert parse_json_object("nope", on_fail="default", default=sentinel) is sentinel

    def test_failure_preserves_content_on_exception(self) -> None:
        content = "totally invalid output"
        with pytest.raises(LLMParseError) as exc_info:
            parse_json_object(content)
        assert exc_info.value.content == content


class TestRepairHelper:
    def test_repair_handles_newlines(self) -> None:
        assert repair_json_string('{"msg": "a\nb"}') == {"msg": "a\nb"}

    def test_repair_returns_none_on_unsalvageable(self) -> None:
        assert repair_json_string("not json at all") is None


class TestRealLLMShapes:
    """Snapshots from observed LLM outputs in the codebase."""

    def test_setup_agent_analyzer_shape(self) -> None:
        # setup/agent_analyzer.py expected dicts with output_schema + proposed_criteria.
        text = """Here's the analysis:
```json
{
  "output_schema": {"category": {"type": "enum"}},
  "proposed_criteria": {"fields": {"category": {"importance": "critical"}}}
}
```"""
        result = parse_json_object(text)
        assert "output_schema" in result
        assert "proposed_criteria" in result

    def test_setup_questionnaire_shape(self) -> None:
        # Refined criteria with trailing commas and prose.
        text = """The refined criteria are:
{"fields": {"category": {"importance": "critical", "partial_credit": true,},}}"""
        result = parse_json_object(text)
        assert result["fields"]["category"]["partial_credit"] is True

    def test_evaluator_judge_score_shape(self) -> None:
        # optimize/evaluator.py expected dicts like {"score": 0.85, "reason": "..."}.
        text = 'The score is: {"score": 0.85, "reason": "matches expected"}'
        assert parse_json_object(text) == {"score": 0.85, "reason": "matches expected"}

    def test_data_synthetic_cases_shape(self) -> None:
        # optimize/data.py expected lists of cases.
        text = """```json
[
  {"input": "case1", "expected": "A"},
  {"input": "case2", "expected": "B"}
]
```"""
        assert parse_json_object(text) == [
            {"input": "case1", "expected": "A"},
            {"input": "case2", "expected": "B"},
        ]


class TestRoundTrip:
    def test_dump_then_parse_is_identity(self) -> None:
        original = {"a": [1, 2, {"b": "c"}], "d": None, "e": True}
        text = json.dumps(original)
        assert parse_json_object(text) == original
