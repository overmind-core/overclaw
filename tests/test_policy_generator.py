"""Tests for overclaw.setup.policy_generator — policy generation and parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from overclaw.setup.policy_generator import (
    _default_policy_data,
    _extract_markdown_and_json,
    _migrate_legacy_policy,
    display_policy,
    generate_policy_from_code,
    generate_policy_from_document,
    improve_existing_policy,
    save_policy,
)


class TestExtractMarkdownAndJson:
    def test_markdown_and_json_blocks(self):
        text = (
            "```markdown\n# Agent Policy\nRule 1\n```\n"
            '```json\n{"domain_rules": ["rule1"]}\n```'
        )
        md, data = _extract_markdown_and_json(text)
        assert "Agent Policy" in md
        assert data["domain_rules"] == ["rule1"]

    def test_md_abbreviation(self):
        text = '```md\n# Policy\nContent\n```\n```json\n{"purpose": "test"}\n```'
        md, data = _extract_markdown_and_json(text)
        assert "Policy" in md
        assert data["purpose"] == "test"

    def test_fallback_heading(self):
        text = '# Agent Policy\nSome rules here\n```json\n{"domain_rules": []}\n```'
        md, data = _extract_markdown_and_json(text)
        assert "Agent Policy" in md

    def test_no_json_fallback_to_embedded(self):
        text = 'Some text {"purpose": "test", "domain_rules": ["r1"]} more text'
        _, data = _extract_markdown_and_json(text)
        assert data.get("purpose") == "test"

    def test_no_json_at_all(self):
        text = "Just plain text with no JSON"
        md, data = _extract_markdown_and_json(text)
        assert data == {}

    def test_legacy_format_migrated(self):
        text = '```json\n{"decision_rules": ["old rule"], "hard_constraints": ["old constraint"]}\n```'
        _, data = _extract_markdown_and_json(text)
        assert "domain_rules" in data


class TestMigrateLegacyPolicy:
    def test_already_new_format(self):
        data = {"domain_rules": ["r1"]}
        assert _migrate_legacy_policy(data) is data

    def test_empty(self):
        assert _migrate_legacy_policy({}) == {}

    def test_legacy_to_new(self):
        data = {
            "purpose": "test",
            "decision_rules": ["rule1"],
            "hard_constraints": ["c1"],
            "edge_cases": [{"scenario": "e1", "expected": "handle"}],
            "quality_expectations": ["q1"],
        }
        result = _migrate_legacy_policy(data)
        assert result["domain_rules"] == ["rule1"]
        assert result["output_constraints"] == ["c1"]
        assert result["domain_edge_cases"][0]["correct_handling"] == "handle"

    def test_edge_case_already_has_correct_handling(self):
        data = {
            "decision_rules": [],
            "edge_cases": [{"scenario": "e1", "correct_handling": "keep"}],
        }
        result = _migrate_legacy_policy(data)
        assert result["domain_edge_cases"][0]["correct_handling"] == "keep"


class TestDefaultPolicyData:
    def test_has_all_keys(self):
        data = _default_policy_data()
        assert "purpose" in data
        assert "domain_rules" in data
        assert "domain_edge_cases" in data
        assert "terminology" in data
        assert "output_constraints" in data
        assert "tool_requirements" in data
        assert "decision_mapping" in data
        assert "quality_expectations" in data


class TestSavePolicy:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "deep" / "policies.md")
        save_policy("# Policy", path)
        assert Path(path).exists()
        assert Path(path).read_text().startswith("# Policy")


class TestDisplayPolicy:
    def test_displays(self):
        console = MagicMock()
        display_policy(
            "# Policy",
            {
                "domain_rules": ["r1"],
                "output_constraints": ["c1"],
                "domain_edge_cases": [],
                "terminology": {"t": "d"},
            },
            console,
        )
        assert console.print.called


class TestGeneratePolicyFromCode:
    @patch("overclaw.core.litellm_params.litellm")
    def test_success(self, mock_litellm):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "```markdown\n# Policy\nContent\n```\n"
            '```json\n{"purpose": "test", "domain_rules": ["r1"]}\n```'
        )
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        analysis = {"description": "Test agent", "_agent_code": "def run(x): pass"}
        md, data = generate_policy_from_code(analysis, "model", console)
        assert "Policy" in md
        assert data["domain_rules"] == ["r1"]

    @patch("overclaw.core.litellm_params.litellm")
    def test_fallback_on_parse_failure(self, mock_litellm):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "no json or markdown here"
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        md, data = generate_policy_from_code({"description": "Test"}, "model", console)
        assert "Auto-generated" in md
        assert data == _default_policy_data()


class TestGeneratePolicyFromDocument:
    @patch("overclaw.core.litellm_params.litellm")
    def test_success(self, mock_litellm, tmp_path):
        doc = tmp_path / "policy.md"
        doc.write_text("# My Rules\nRule 1\n")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "```markdown\n# Structured Policy\n```\n"
            '```json\n{"purpose": "test", "domain_rules": ["r1"]}\n```'
        )
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        md, data = generate_policy_from_document(
            {"description": "Test"}, str(doc), "model", console
        )
        assert data["domain_rules"] == ["r1"]


class TestImproveExistingPolicy:
    @patch("overclaw.core.litellm_params.litellm")
    def test_success(self, mock_litellm, tmp_path):
        doc = tmp_path / "policy.md"
        doc.write_text("# Old Policy\nOld rules\n")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "```changes\n- Added new rule\n```\n"
            "```markdown\n# Improved Policy\n```\n"
            '```json\n{"purpose": "improved", "domain_rules": ["new"]}\n```'
        )
        mock_litellm.completion.return_value = mock_resp

        console = MagicMock()
        md, data, changes = improve_existing_policy(
            {"description": "Test", "_agent_code": "pass"}, str(doc), "model", console
        )
        assert "Added new rule" in changes
        assert data["purpose"] == "improved"
