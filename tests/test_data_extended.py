"""Extended tests for overclaw.optimize.data — LLM-dependent generation paths."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from overclaw.optimize.data import (
    _generate_batch,
    _generate_personas,
    _llm_call,
    generate_synthetic_data,
)


class TestLlmCall:
    @patch("overclaw.utils.llm.litellm")
    def test_success(self, mock_litellm):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "response"
        mock_litellm.completion.return_value = mock_resp

        result = _llm_call("model", "prompt", max_retries=1)
        assert result == "response"

    @patch("overclaw.utils.llm.litellm")
    @patch("overclaw.optimize.data.time.sleep")
    def test_rate_limit_retry(self, mock_sleep, mock_litellm):
        import litellm as real_litellm

        mock_litellm.completion.side_effect = [
            real_litellm.RateLimitError("rate limited", "provider", "model"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))]),
        ]

        result = _llm_call("model", "prompt", max_retries=2)
        assert result == "ok"

    @patch("overclaw.utils.llm.litellm")
    def test_all_retries_fail(self, mock_litellm):
        import litellm as real_litellm

        mock_litellm.completion.side_effect = real_litellm.Timeout(
            "timeout", "provider", "model"
        )
        result = _llm_call("model", "prompt", max_retries=2)
        assert result is None


class TestGeneratePersonas:
    @patch("overclaw.optimize.data._llm_call")
    def test_success(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "personas": [
                    {"name": "Tester", "skill_level": "expert", "intent": "standard"},
                ]
            }
        )
        result = _generate_personas("desc", None, {}, None, "model", 1)
        assert len(result) == 1
        assert result[0]["name"] == "Tester"

    @patch("overclaw.optimize.data._llm_call")
    def test_fallback_on_failure(self, mock_call):
        mock_call.return_value = None
        result = _generate_personas("desc", None, {}, None, "model", 3)
        assert len(result) == 3

    @patch("overclaw.optimize.data._llm_call")
    def test_fallback_on_bad_json(self, mock_call):
        mock_call.return_value = "not json"
        result = _generate_personas("desc", None, {}, None, "model", 2)
        assert len(result) == 2


class TestGenerateBatch:
    @patch("overclaw.optimize.data._llm_call")
    def test_success(self, mock_call):
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        mock_call.return_value = json.dumps({"cases": cases})
        result = _generate_batch(
            {"name": "Tester"}, "desc", None, {}, None, "model", 1, []
        )
        assert len(result) == 1

    @patch("overclaw.optimize.data._llm_call")
    def test_returns_empty_on_failure(self, mock_call):
        mock_call.return_value = None
        result = _generate_batch(
            {"name": "Tester"}, "desc", None, {}, None, "model", 1, []
        )
        assert result == []

    @patch("overclaw.optimize.data._llm_call")
    def test_parses_bare_array(self, mock_call):
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        mock_call.return_value = json.dumps(cases)
        result = _generate_batch(
            {"name": "Tester"}, "desc", None, {}, None, "model", 1, []
        )
        assert len(result) == 1


class TestGenerateSyntheticData:
    @patch("overclaw.utils.llm.litellm")
    def test_success(self, mock_litellm):
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(cases)
        mock_litellm.completion.return_value = mock_resp

        result = generate_synthetic_data("desc", "model", num_samples=1)
        assert len(result) == 1

    @patch("overclaw.utils.llm.litellm")
    def test_parse_failure_raises(self, mock_litellm):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "no json array"
        mock_litellm.completion.return_value = mock_resp

        with pytest.raises(ValueError, match="Failed to parse"):
            generate_synthetic_data("desc", "model")

    @patch("overclaw.utils.llm.litellm")
    def test_with_agent_code_and_policy(self, mock_litellm):
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(cases)
        mock_litellm.completion.return_value = mock_resp

        result = generate_synthetic_data(
            "desc",
            "model",
            agent_code="def run(x): pass",
            policy_context="some rules",
        )
        assert len(result) >= 1
