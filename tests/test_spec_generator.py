"""Tests for overclaw.setup.spec_generator — eval spec construction."""

from __future__ import annotations

import json
from pathlib import Path


from overclaw.setup.spec_generator import (
    IMPORTANCE_MULTIPLIERS,
    _build_spec,
    generate_spec_from_proposal,
    save_spec,
)


# ---------------------------------------------------------------------------
# IMPORTANCE_MULTIPLIERS
# ---------------------------------------------------------------------------


class TestImportanceMultipliers:
    def test_critical_highest(self):
        assert IMPORTANCE_MULTIPLIERS["critical"] > IMPORTANCE_MULTIPLIERS["important"]

    def test_important_higher_than_minor(self):
        assert IMPORTANCE_MULTIPLIERS["important"] > IMPORTANCE_MULTIPLIERS["minor"]


# ---------------------------------------------------------------------------
# generate_spec_from_proposal
# ---------------------------------------------------------------------------


class TestGenerateSpecFromProposal:
    def test_basic_spec(self):
        analysis = {
            "description": "Test agent",
            "output_schema": {
                "status": {"type": "enum", "values": ["a", "b"]},
                "score": {"type": "number", "range": [0, 100]},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "status": {"importance": "critical"},
                    "score": {"importance": "minor", "tolerance": 5},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec["structure_weight"] == 20
        assert spec["total_points"] == 100
        assert "status" in spec["output_fields"]
        assert "score" in spec["output_fields"]
        assert spec["output_fields"]["status"]["type"] == "enum"

    def test_weights_sum_to_available(self):
        analysis = {
            "output_schema": {
                "a": {"type": "text"},
                "b": {"type": "text"},
                "c": {"type": "text"},
            },
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "a": {"importance": "critical"},
                    "b": {"importance": "important"},
                    "c": {"importance": "minor"},
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        field_weights = sum(f["weight"] for f in spec["output_fields"].values())
        # 100 - structure - reserved llm_judge slot for text fields
        assert spec["llm_judge_weight"] == 10
        assert field_weights == 70  # 100 - 20 structure - 10 llm_judge

    def test_with_policy_data(self):
        analysis = {
            "output_schema": {"x": {"type": "text"}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"x": {"importance": "important"}},
            },
        }
        policy = {"purpose": "test", "domain_rules": ["rule 1"]}
        spec = generate_spec_from_proposal(analysis, policy_data=policy)
        assert spec["policy"] == policy

    def test_with_tool_analysis(self):
        analysis = {
            "output_schema": {"x": {"type": "text"}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"x": {"importance": "important"}},
            },
            "tool_analysis": {
                "tools": {"search": {}},
                "expected_tools": ["search"],
                "dependencies": [],
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert "tool_config" in spec
        assert spec["tool_usage_weight"] == 10

    def test_enum_partial_credit(self):
        analysis = {
            "output_schema": {"status": {"type": "enum", "values": ["a", "b"]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "status": {"importance": "critical", "partial_credit": True}
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec["output_fields"]["status"]["partial_credit"] is True

    def test_number_tolerance_bands(self):
        analysis = {
            "output_schema": {"score": {"type": "number", "range": [0, 100]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"score": {"importance": "important", "tolerance": 10}},
            },
        }
        spec = generate_spec_from_proposal(analysis)
        bands = spec["output_fields"]["score"]["tolerance_bands"]
        assert len(bands) == 4
        assert bands[0]["score_pct"] == 1.0

    def test_text_eval_mode(self):
        analysis = {
            "output_schema": {"reason": {"type": "text"}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "reason": {"importance": "important", "eval_mode": "non_empty"}
                },
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert spec["output_fields"]["reason"]["eval_mode"] == "non_empty"


# ---------------------------------------------------------------------------
# _build_spec
# ---------------------------------------------------------------------------


class TestBuildSpec:
    def test_empty_schema(self):
        spec = _build_spec({}, {}, {}, {}, 20)
        assert spec["structure_weight"] == 20
        assert spec["total_points"] == 100
        assert spec["output_fields"] == {}

    def test_consistency_rules_included(self):
        analysis = {
            "consistency_rules": [
                {"field_a": "x", "field_b": "y", "type": "correlation"}
            ]
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "consistency_rules" in spec

    def test_no_tools_no_tool_config(self):
        spec = _build_spec({}, {"x": {"type": "text"}}, {"x": "important"}, {}, 20)
        assert "tool_config" not in spec


# ---------------------------------------------------------------------------
# save_spec
# ---------------------------------------------------------------------------


class TestSaveSpec:
    def test_creates_directory(self, tmp_path):
        path = str(tmp_path / "deep" / "dir" / "spec.json")
        save_spec({"key": "value"}, path)
        loaded = json.loads(Path(path).read_text())
        assert loaded["key"] == "value"

    def test_overwrites(self, tmp_path):
        path = str(tmp_path / "spec.json")
        save_spec({"v": 1}, path)
        save_spec({"v": 2}, path)
        loaded = json.loads(Path(path).read_text())
        assert loaded["v"] == 2
