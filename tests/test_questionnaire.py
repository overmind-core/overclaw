"""Tests for overclaw.setup.questionnaire — criteria refinement."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from overclaw.setup.questionnaire import _display_refined, run_questionnaire


class TestRunQuestionnaire:
    @patch("overclaw.setup.questionnaire.Prompt")
    @patch("overclaw.setup.questionnaire.overmind_prompt")
    @patch("overclaw.utils.llm.litellm")
    def test_success(self, mock_litellm, mock_prompt, mock_rich_prompt):
        mock_prompt.side_effect = ["change importance", "accuracy", "wrong enum"]
        mock_rich_prompt.ask.return_value = ""

        refined = {
            "structure_weight": 20,
            "fields": {"status": {"importance": "critical"}},
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(refined)
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        analysis = {
            "output_schema": {"status": {"type": "enum"}},
            "proposed_criteria": {"structure_weight": 20, "fields": {}},
        }
        result = run_questionnaire(analysis, "model", console)
        assert (
            result["proposed_criteria"]["fields"]["status"]["importance"] == "critical"
        )

    @patch("overclaw.setup.questionnaire.Prompt")
    @patch("overclaw.setup.questionnaire.overmind_prompt")
    @patch("overclaw.utils.llm.litellm")
    def test_parse_failure_keeps_original(
        self, mock_litellm, mock_prompt, mock_rich_prompt
    ):
        mock_prompt.side_effect = ["feedback", "good output", "mistakes"]
        mock_rich_prompt.ask.return_value = ""

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "no json here"
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        original = {"structure_weight": 20, "fields": {"x": {"importance": "minor"}}}
        analysis = {"output_schema": {}, "proposed_criteria": original}
        result = run_questionnaire(analysis, "model", console)
        assert result["proposed_criteria"] == original


class TestDisplayRefined:
    def test_with_fields(self):
        console = MagicMock()
        criteria = {
            "structure_weight": 20,
            "fields": {
                "status": {"importance": "critical"},
                "score": {"importance": "important", "tolerance": 5},
                "text": {"importance": "minor", "eval_mode": "non_empty"},
                "flag": {"importance": "important"},
            },
        }
        analysis = {
            "output_schema": {
                "status": {"type": "enum"},
                "score": {"type": "number"},
                "text": {"type": "text"},
                "flag": {"type": "boolean"},
            }
        }
        _display_refined(criteria, analysis, console)

    def test_empty_fields(self):
        console = MagicMock()
        _display_refined({"fields": {}}, {}, console)
        _display_refined({}, {}, console)
