"""Tests for overclaw.utils.policy — policy loading and formatting."""

from __future__ import annotations

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.utils.policy import (
    _get_constraints,
    _get_edge_cases,
    _get_quality_expectations,
    _get_rules,
    _is_two_layer,
    default_policy_path,
    format_for_codegen,
    format_for_diagnosis,
    format_for_judge,
    format_for_synthetic_data,
    load_policy_data,
    load_policy_markdown,
)


# ---------------------------------------------------------------------------
# Helper: two-layer vs legacy policies
# ---------------------------------------------------------------------------


TWO_LAYER_POLICY = {
    "purpose": "Qualify sales leads",
    "domain_rules": ["Rule 1", "Rule 2"],
    "domain_edge_cases": [
        {"scenario": "Edge 1", "correct_handling": "Handle 1"},
        "Edge 2 string",
    ],
    "output_constraints": ["Constraint A", "Constraint B"],
    "terminology": {"MQL": "Marketing Qualified Lead"},
    "tool_requirements": ["Must use search tool"],
    "decision_mapping": ["High budget -> hot lead"],
    "quality_expectations": ["Response must be detailed"],
}

LEGACY_POLICY = {
    "decision_rules": ["Legacy rule 1"],
    "hard_constraints": ["Legacy constraint"],
    "edge_cases": ["Legacy edge case"],
    "quality_expectations": ["Legacy quality"],
    "priority_order": ["safety", "accuracy"],
}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestIsTwoLayer:
    def test_two_layer(self):
        assert _is_two_layer(TWO_LAYER_POLICY) is True

    def test_legacy(self):
        assert _is_two_layer(LEGACY_POLICY) is False

    def test_empty(self):
        assert _is_two_layer({}) is False


# ---------------------------------------------------------------------------
# Rule getters
# ---------------------------------------------------------------------------


class TestGetRules:
    def test_two_layer(self):
        assert _get_rules(TWO_LAYER_POLICY) == ["Rule 1", "Rule 2"]

    def test_legacy(self):
        assert _get_rules(LEGACY_POLICY) == ["Legacy rule 1"]

    def test_empty(self):
        assert _get_rules({}) == []


class TestGetConstraints:
    def test_two_layer(self):
        assert _get_constraints(TWO_LAYER_POLICY) == ["Constraint A", "Constraint B"]

    def test_legacy(self):
        assert _get_constraints(LEGACY_POLICY) == ["Legacy constraint"]


class TestGetEdgeCases:
    def test_two_layer(self):
        cases = _get_edge_cases(TWO_LAYER_POLICY)
        assert len(cases) == 2
        assert isinstance(cases[0], dict)
        assert isinstance(cases[1], str)

    def test_legacy(self):
        assert _get_edge_cases(LEGACY_POLICY) == ["Legacy edge case"]


class TestGetQualityExpectations:
    def test_two_layer(self):
        assert _get_quality_expectations(TWO_LAYER_POLICY) == [
            "Response must be detailed"
        ]

    def test_legacy(self):
        assert _get_quality_expectations(LEGACY_POLICY) == ["Legacy quality"]


# ---------------------------------------------------------------------------
# format_for_diagnosis
# ---------------------------------------------------------------------------


class TestFormatForDiagnosis:
    def test_empty_policy(self):
        assert format_for_diagnosis({}) == ""

    def test_none_policy(self):
        assert format_for_diagnosis(None) == ""

    def test_includes_purpose(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "Qualify sales leads" in result

    def test_includes_rules(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "Rule 1" in result
        assert "Domain Rules" in result

    def test_includes_constraints(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "Constraint A" in result

    def test_includes_decision_mapping(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "hot lead" in result

    def test_includes_terminology(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "MQL" in result

    def test_includes_tool_requirements(self):
        result = format_for_diagnosis(TWO_LAYER_POLICY)
        assert "search tool" in result

    def test_legacy_priority_order(self):
        result = format_for_diagnosis(LEGACY_POLICY)
        assert "safety" in result
        assert "accuracy" in result


# ---------------------------------------------------------------------------
# format_for_codegen
# ---------------------------------------------------------------------------


class TestFormatForCodegen:
    def test_empty_policy(self):
        assert format_for_codegen({}) == ""

    def test_none_policy(self):
        assert format_for_codegen(None) == ""

    def test_includes_constraints(self):
        result = format_for_codegen(TWO_LAYER_POLICY)
        assert "Constraint A" in result

    def test_includes_tool_requirements(self):
        result = format_for_codegen(TWO_LAYER_POLICY)
        assert "search tool" in result

    def test_includes_decision_mapping(self):
        result = format_for_codegen(TWO_LAYER_POLICY)
        assert "hot lead" in result

    def test_no_rules_or_edge_cases(self):
        result = format_for_codegen(TWO_LAYER_POLICY)
        assert "Domain Rules" not in result


# ---------------------------------------------------------------------------
# format_for_synthetic_data
# ---------------------------------------------------------------------------


class TestFormatForSyntheticData:
    def test_empty_policy(self):
        assert format_for_synthetic_data({}) == ""

    def test_includes_rules(self):
        result = format_for_synthetic_data(TWO_LAYER_POLICY)
        assert "Rule 1" in result

    def test_includes_edge_cases_dict(self):
        result = format_for_synthetic_data(TWO_LAYER_POLICY)
        assert "Edge 1" in result
        assert "Handle 1" in result

    def test_includes_edge_cases_string(self):
        result = format_for_synthetic_data(TWO_LAYER_POLICY)
        assert "Edge 2 string" in result

    def test_includes_terminology(self):
        result = format_for_synthetic_data(TWO_LAYER_POLICY)
        assert "MQL" in result

    def test_includes_constraints(self):
        result = format_for_synthetic_data(TWO_LAYER_POLICY)
        assert "Constraint A" in result


# ---------------------------------------------------------------------------
# format_for_judge
# ---------------------------------------------------------------------------


class TestFormatForJudge:
    def test_empty_policy(self):
        assert format_for_judge({}) == ""

    def test_includes_rules(self):
        result = format_for_judge(TWO_LAYER_POLICY)
        assert "Rule 1" in result

    def test_includes_constraints(self):
        result = format_for_judge(TWO_LAYER_POLICY)
        assert "Constraint A" in result

    def test_includes_decision_mapping(self):
        result = format_for_judge(TWO_LAYER_POLICY)
        assert "hot lead" in result

    def test_includes_quality_expectations(self):
        result = format_for_judge(TWO_LAYER_POLICY)
        assert "detailed" in result


# ---------------------------------------------------------------------------
# load helpers
# ---------------------------------------------------------------------------


class TestDefaultPolicyPath:
    def test_returns_setup_spec_path(self, overclaw_tmp_project):
        result = default_policy_path("agent1")
        assert "setup_spec" in result
        assert OVERCLAW_DIR_NAME in result
        assert "agent1" in result
        assert result.endswith("policies.md")


class TestLoadPolicyData:
    def test_with_policy(self):
        spec = {"policy": {"purpose": "test"}}
        assert load_policy_data(spec) == {"purpose": "test"}

    def test_without_policy(self):
        assert load_policy_data({}) is None

    def test_none_policy(self):
        assert load_policy_data({"policy": None}) is None


class TestLoadPolicyMarkdown:
    def test_file_exists(self, overclaw_tmp_project):
        setup_dir = (
            overclaw_tmp_project / OVERCLAW_DIR_NAME / "agents" / "p1" / "setup_spec"
        )
        setup_dir.mkdir(parents=True)
        policy_file = setup_dir / "policies.md"
        policy_file.write_text("# Policy\nRule 1")
        result = load_policy_markdown("p1")
        assert "Rule 1" in result

    def test_file_missing(self, overclaw_tmp_project):
        result = load_policy_markdown("missing-agent")
        assert result is None
