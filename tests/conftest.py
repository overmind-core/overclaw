"""Shared fixtures for the Overmind test suite."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider

import overmind.tracing as _overmind_tracing

from overmind.core.constants import OVERMIND_DIR_NAME


# ---------------------------------------------------------------------------
# OTel / overmind SDK bootstrap for tests
# ---------------------------------------------------------------------------
#
# ``overmind.start_span`` / ``overmind.observe`` / ``overmind.set_tag`` all
# go through ``overmind.tracing.get_tracer()`` which raises until
# ``overmind.init()`` is called.  We don't want test runs to actually
# export anything, so we install a no-op ``TracerProvider`` (no exporter)
# and flip the SDK's internal ``_initialized`` flag manually.  This lets
# every traced/observed function in Overmind run as-is during tests
# without a real Overmind API key or HTTP exporter.

_provider = TracerProvider()
_otel_trace.set_tracer_provider(_provider)
_overmind_tracing._tracer = _provider.get_tracer("overmind", "test")
_overmind_tracing._initialized = True
os.environ["OVERMIND_API_KEY"] = "test"

@pytest.fixture()
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a minimal project with Overmind state dir and a sample agent."""
    overmind_dir = tmp_path / OVERMIND_DIR_NAME
    overmind_dir.mkdir(parents=True)
    (overmind_dir / "agents.toml").write_text(
        textwrap.dedent("""\
        # Overmind agent registry

        agents = [
            { name = "my-agent", entrypoint = "agents.agent1.sample_agent:run" },
        ]
        """),
        encoding="utf-8",
    )

    agent_dir = tmp_path / "agents" / "agent1"
    agent_dir.mkdir(parents=True)
    agent_file = agent_dir / "sample_agent.py"
    agent_file.write_text(
        textwrap.dedent("""\
        def run(input_data: dict) -> dict:
            return {"result": "ok"}

        def helper(input_data: dict) -> dict:
            return {"result": "helper"}
        """),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def overmind_tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal project + chdir so ``project_root()`` resolves here."""
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def tmp_project_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Project with state dir but no ``agents.toml`` (empty registry)."""
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def sample_eval_spec(tmp_path: Path) -> str:
    """Write a realistic eval spec JSON and return its path."""
    spec = {
        "agent_description": "Test agent",
        "agent_path": "agents/agent1/sample_agent.py",
        "input_schema": {
            "company_name": {"type": "string", "description": "Company name"},
            "budget": {"type": "number", "description": "Budget"},
        },
        "output_fields": {
            "qualification": {
                "type": "enum",
                "description": "Lead qualification",
                "weight": 30,
                "importance": "critical",
                "values": ["hot", "warm", "cold"],
                "partial_credit": True,
                "partial_score": 6,
            },
            "score": {
                "type": "number",
                "description": "Score",
                "weight": 20,
                "importance": "important",
                "range": [0, 100],
                "tolerance": 10,
                "tolerance_bands": [
                    {"within": 5, "score_pct": 1.0},
                    {"within": 10, "score_pct": 0.8},
                    {"within": 15, "score_pct": 0.5},
                    {"within": 25, "score_pct": 0.25},
                ],
            },
            "reasoning": {
                "type": "text",
                "description": "Reasoning",
                "weight": 15,
                "importance": "important",
                "eval_mode": "non_empty",
            },
            "is_enterprise": {
                "type": "boolean",
                "description": "Enterprise flag",
                "weight": 15,
                "importance": "important",
            },
        },
        "structure_weight": 20,
        "total_points": 100,
        "consistency_rules": [],
        "optimizable_elements": ["system_prompt", "format_input"],
        "fixed_elements": ["tool implementations"],
    }
    spec_path = tmp_path / "eval_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return str(spec_path)


@pytest.fixture()
def sample_eval_spec_with_tools(tmp_path: Path) -> str:
    """Eval spec with tool_config for tool scoring tests."""
    spec = {
        "agent_description": "Tool agent",
        "output_fields": {
            "result": {
                "type": "text",
                "description": "Result",
                "weight": 50,
                "eval_mode": "non_empty",
            },
        },
        "structure_weight": 20,
        "total_points": 100,
        "tool_config": {
            "expected_tools": ["search", "analyze"],
            "param_constraints": {
                "search": {"query_type": ["web", "local", "database"]},
            },
            "dependencies": [
                {
                    "from_tool": "search",
                    "from_field": "results",
                    "to_tool": "analyze",
                    "to_param": "data",
                }
            ],
        },
        "tool_usage_weight": 30,
    }
    spec_path = tmp_path / "eval_spec_tools.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return str(spec_path)


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> str:
    """Write a sample dataset JSON and return its path."""
    cases = [
        {
            "input": {"company_name": "Acme Corp", "budget": 50000},
            "expected_output": {
                "qualification": "hot",
                "score": 85,
                "reasoning": "Large budget enterprise",
                "is_enterprise": True,
            },
        },
        {
            "input": {"company_name": "Tiny LLC", "budget": 500},
            "expected_output": {
                "qualification": "cold",
                "score": 20,
                "reasoning": "Small budget startup",
                "is_enterprise": False,
            },
        },
    ]
    data_path = tmp_path / "dataset.json"
    data_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return str(data_path)

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = Mock()
    response.status_code = 200
    response.content = b'{"status": "success"}'
    response.json.return_value = {"status": "success"}
    return response
