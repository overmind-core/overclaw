"""Extended tests for overclaw.optimize.analyzer — LLM-dependent code paths."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from overclaw.optimize.analyzer import (
    _run_codegen,
    _run_diagnosis,
    analyze_and_improve,
    generate_candidates,
)


class TestRunDiagnosis:
    @patch("overclaw.core.litellm_params.litellm")
    def test_success(self, mock_litellm):
        diagnosis = {"root_cause": "bad prompt", "changes": [{"action": "fix prompt"}]}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = f"```json\n{json.dumps(diagnosis)}\n```"
        mock_litellm.completion.return_value = mock_resp

        result = _run_diagnosis(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="model",
            eval_spec=None,
            failed_attempts=None,
            successful_changes=None,
            allow_model_change=False,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert result["root_cause"] == "bad prompt"

    @patch("overclaw.core.litellm_params.litellm")
    def test_parse_failure_returns_none(self, mock_litellm):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "no json"
        mock_litellm.completion.return_value = mock_resp

        result = _run_diagnosis(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="model",
            eval_spec=None,
            failed_attempts=None,
            successful_changes=None,
            allow_model_change=False,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert result is None

    @patch("overclaw.core.litellm_params.litellm")
    def test_exception_returns_none(self, mock_litellm):
        mock_litellm.completion.side_effect = RuntimeError("API error")

        result = _run_diagnosis(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="model",
            eval_spec=None,
            failed_attempts=None,
            successful_changes=None,
            allow_model_change=False,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert result is None

    @patch("overclaw.core.litellm_params.litellm")
    def test_with_focus_area(self, mock_litellm):
        diagnosis = {"root_cause": "tool issue", "changes": []}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(diagnosis)
        mock_litellm.completion.return_value = mock_resp

        result = _run_diagnosis(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="model",
            eval_spec=None,
            failed_attempts=None,
            successful_changes=None,
            allow_model_change=False,
            temperature=0.7,
            focus_area="tool_description",
            entrypoint_fn="run",
        )
        assert result is not None


class TestRunCodegen:
    @patch("overclaw.core.litellm_params.litellm")
    def test_success(self, mock_litellm):
        code = "def run(input_data):\n    return {'result': 'improved'}\n"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = f"```python\n{code}\n```"
        mock_litellm.completion.return_value = mock_resp

        result = _run_codegen(
            agent_code="def run(input_data): pass",
            diagnosis={"root_cause": "test", "changes": []},
            model="model",
            eval_spec=None,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert result is not None
        assert "def run" in result

    @patch("overclaw.core.litellm_params.litellm")
    def test_exception_returns_none(self, mock_litellm):
        mock_litellm.completion.side_effect = RuntimeError("API fail")
        result = _run_codegen(
            agent_code="def run(x): pass",
            diagnosis={},
            model="model",
            eval_spec=None,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert result is None


class TestGenerateCandidates:
    @patch("overclaw.optimize.analyzer._run_codegen")
    @patch("overclaw.optimize.analyzer._run_diagnosis")
    def test_single_candidate(self, mock_diag, mock_codegen):
        mock_diag.return_value = {"root_cause": "issue", "changes": [{"action": "fix"}]}
        mock_codegen.return_value = "def run(x): return {}"

        result = generate_candidates(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="model",
            num_candidates=1,
            entrypoint_fn="run",
        )
        assert len(result) >= 1
        assert result[0]["updated_code"] is not None

    @patch("overclaw.optimize.analyzer._run_codegen")
    @patch("overclaw.optimize.analyzer._run_diagnosis")
    def test_all_fail(self, mock_diag, mock_codegen):
        mock_diag.return_value = None
        mock_codegen.return_value = None

        with patch("overclaw.core.litellm_params.litellm") as mock_litellm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "no code"
            mock_resp.choices[0].finish_reason = "stop"
            mock_litellm.completion.return_value = mock_resp

            result = generate_candidates(
                agent_code="def run(x): pass",
                case_results=[],
                evaluation_results={"avg_total": 50},
                model="model",
                num_candidates=1,
                entrypoint_fn="run",
            )
            assert result[0]["method"] == "failed"


class TestAnalyzeAndImprove:
    @patch("overclaw.optimize.analyzer.generate_candidates")
    def test_delegates(self, mock_gen):
        mock_gen.return_value = [{"analysis": "test", "updated_code": "code"}]
        result = analyze_and_improve(
            agent_code="def run(x): pass",
            traces=[],
            evaluation_results={"avg_total": 50},
            model="model",
            entrypoint_fn="run",
        )
        assert result["analysis"] == "test"
