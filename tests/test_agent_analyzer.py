"""Tests for overclaw.setup.agent_analyzer — LLM-based code analysis."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.setup.agent_analyzer import _display_analysis, analyze_agent


def _ensure_overclaw_root(base: Path) -> None:
    """analyze_agent requires a project root containing the OverClaw state dir."""
    (base / OVERCLAW_DIR_NAME).mkdir(parents=True, exist_ok=True)


class TestAnalyzeAgent:
    @patch("overclaw.utils.llm.litellm")
    def test_success(self, mock_litellm, tmp_path):
        _ensure_overclaw_root(tmp_path)
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x):\n    return {'status': 'ok'}\n")

        analysis = {
            "description": "Test agent",
            "output_schema": {"status": {"type": "enum", "values": ["ok", "fail"]}},
            "proposed_criteria": {"structure_weight": 20, "fields": {}},
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(analysis)
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        result = analyze_agent(str(agent), "model", console, entrypoint_fn="run")
        assert result["description"] == "Test agent"
        assert result["_agent_path"] == str(agent)
        assert "_agent_code" in result

    @patch("overclaw.utils.llm.litellm")
    def test_parse_failure_exits(self, mock_litellm, tmp_path):
        _ensure_overclaw_root(tmp_path)
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): pass\n")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "not json at all"
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        with pytest.raises(SystemExit):
            analyze_agent(str(agent), "model", console, entrypoint_fn="run")

    @patch("overclaw.utils.llm.litellm")
    def test_json_embedded_in_text(self, mock_litellm, tmp_path):
        _ensure_overclaw_root(tmp_path)
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {}\n")

        analysis = {"description": "Embedded", "output_schema": {}}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[
            0
        ].message.content = f"Some preamble {json.dumps(analysis)} trailing"
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        result = analyze_agent(str(agent), "model", console, entrypoint_fn="run")
        assert result["description"] == "Embedded"


class TestDisplayAnalysis:
    def test_basic(self):
        console = MagicMock()
        analysis = {
            "description": "Test",
            "output_schema": {"status": {"type": "enum", "values": ["a", "b"]}},
        }
        _display_analysis(analysis, console)

    def test_with_criteria(self):
        console = MagicMock()
        analysis = {
            "description": "Test",
            "output_schema": {"score": {"type": "number", "range": [0, 100]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"score": {"importance": "critical", "tolerance": 5}},
            },
        }
        _display_analysis(analysis, console)

    def test_with_tools(self):
        console = MagicMock()
        analysis = {
            "description": "Test",
            "output_schema": {},
            "tool_analysis": {
                "tools": {
                    "search": {
                        "description_quality": "good",
                        "issues": [],
                        "param_constraints": {"q": ["web"]},
                    }
                },
                "dependencies": [
                    {
                        "from_tool": "a",
                        "from_field": "x",
                        "to_tool": "b",
                        "to_param": "y",
                        "description": "chain",
                    }
                ],
                "orchestration_issues": ["Missing retry logic"],
            },
            "consistency_rules": [
                {"field_a": "x", "field_b": "y", "description": "corr", "penalty": 3}
            ],
        }
        _display_analysis(analysis, console)
