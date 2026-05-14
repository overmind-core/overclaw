"""Tests for overmind.optimize.config — Config dataclass and helpers."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.core.paths import agent_experiments_dir, agent_setup_spec_dir
from overmind.optimize.config import (
    Config,
    SpecValidationError,
    _clear_existing_experiments,
    _require_analyzer_model_env_fast,
    apply_eval_spec_scope,
    validate_eval_spec,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = Config(agent_name="test", agent_path="/path", entrypoint_fn="run")
        assert cfg.iterations == 5
        assert cfg.candidates_per_iteration == 3
        assert cfg.parallel is True
        assert cfg.max_workers == 5
        assert cfg.regression_threshold == 0.35
        assert cfg.holdout_ratio == 0.2
        assert cfg.early_stopping_patience == 3
        assert cfg.holdout_enforcement is True
        assert cfg.overfit_gap_threshold == 10.0
        assert cfg.holdout_weight == 0.3
        assert cfg.catastrophic_holdout_threshold == 0.5
        assert cfg.max_code_growth_ratio == 2.5

    def test_custom_values(self):
        cfg = Config(
            agent_name="x",
            agent_path="/p",
            entrypoint_fn="run",
            iterations=10,
            parallel=False,
        )
        assert cfg.iterations == 10
        assert cfg.parallel is False

    def test_all_fields_have_defaults_except_required(self):
        required = {"agent_name", "agent_path", "entrypoint_fn"}
        for f in fields(Config):
            if f.name in required:
                continue
            assert (
                f.default is not f.default_factory
                if hasattr(f, "default_factory")
                else True
            )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_experiments_dir(self, overmind_tmp_project: Path):
        result = agent_experiments_dir("agent1")
        assert result.name == "experiments"
        assert result.parent.name == "agent1"
        assert OVERMIND_DIR_NAME in str(result)

    def test_eval_spec_file_path(self, overmind_tmp_project: Path):
        result = agent_setup_spec_dir("agent1") / "eval_spec.json"
        assert str(result).endswith("eval_spec.json")
        assert "setup_spec" in str(result)
        assert OVERMIND_DIR_NAME in str(result)

    def test_dataset_file_path(self, overmind_tmp_project: Path):
        result = agent_setup_spec_dir("agent1") / "dataset.json"
        assert str(result).endswith("dataset.json")
        assert "setup_spec" in str(result)


# ---------------------------------------------------------------------------
# _clear_existing_experiments
# ---------------------------------------------------------------------------


class TestClearExistingExperiments:
    def test_no_experiments_dir(self, overmind_tmp_project: Path):
        console = MagicMock()
        _clear_existing_experiments("x", console)

    def test_empty_experiments_dir(self, overmind_tmp_project: Path):
        exp_dir = (
            overmind_tmp_project / OVERMIND_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        console = MagicMock()
        _clear_existing_experiments("x", console)

    def test_fast_mode_clears(self, overmind_tmp_project: Path):
        exp_dir = (
            overmind_tmp_project / OVERMIND_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console, fast=True)
        assert exp_dir.exists()
        assert not list(exp_dir.iterdir())  # cleaned

    @patch("overmind.optimize.config.confirm_option", return_value=True)
    def test_interactive_user_confirms(self, _mock_confirm, overmind_tmp_project: Path):
        exp_dir = (
            overmind_tmp_project / OVERMIND_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console)
        assert not list(exp_dir.iterdir())

    @patch("overmind.optimize.config.confirm_option", return_value=False)
    def test_interactive_user_declines(self, _mock_confirm, overmind_tmp_project: Path):
        exp_dir = (
            overmind_tmp_project / OVERMIND_DIR_NAME / "agents" / "x" / "experiments"
        )
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.json").write_text("{}")
        console = MagicMock()
        _clear_existing_experiments("x", console)
        assert (exp_dir / "result.json").exists()  # kept


# ---------------------------------------------------------------------------
# _require_analyzer_model_env_fast
# ---------------------------------------------------------------------------


class TestRequireAnalyzerModelEnvFast:
    def test_returns_model_when_set(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "gpt-5.4")
        console = MagicMock()
        result = _require_analyzer_model_env_fast(console)
        assert "gpt-5.4" in result

    def test_exits_when_not_set(self, monkeypatch):
        monkeypatch.delenv("ANALYZER_MODEL", raising=False)
        console = MagicMock()
        with pytest.raises(SystemExit) as exc_info:
            _require_analyzer_model_env_fast(console)
        assert exc_info.value.code == 1

    def test_exits_when_empty(self, monkeypatch):
        monkeypatch.setenv("ANALYZER_MODEL", "  ")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _require_analyzer_model_env_fast(console)


# ---------------------------------------------------------------------------
# apply_eval_spec_scope
# ---------------------------------------------------------------------------


def _bare_cfg(**overrides) -> Config:
    cfg = Config(agent_name="x", agent_path="/p", entrypoint_fn="run")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestApplyEvalSpecScope:
    def test_picks_up_optimizable_paths(self):
        cfg = _bare_cfg()
        apply_eval_spec_scope(cfg, {"scope": {"optimizable_paths": ["a.py", "b.py"]}})
        assert cfg.optimizable_scope == ["a.py", "b.py"]

    def test_picks_up_context_paths(self):
        cfg = _bare_cfg()
        apply_eval_spec_scope(cfg, {"scope": {"context_paths": ["docs/README.md"]}})
        assert cfg.context_scope == ["docs/README.md"]

    def test_picks_up_read_only_paths(self):
        """``read_only_paths`` is the new machine-readable knob for declaring
        harness / fixture files that must not be edited by candidates."""
        cfg = _bare_cfg()
        apply_eval_spec_scope(cfg, {"scope": {"read_only_paths": ["entrypoint.py"]}})
        assert cfg.read_only_scope == ["entrypoint.py"]

    def test_picks_up_exclude_paths(self):
        cfg = _bare_cfg()
        apply_eval_spec_scope(cfg, {"scope": {"exclude_paths": [".venv/**"]}})
        assert cfg.exclude_scope == [".venv/**"]

    def test_picks_up_search_paths(self):
        """``search_paths`` is the new sys.path-style knob for hyphenated
        and src-layout repos. The Config field must mirror it."""
        cfg = _bare_cfg()
        apply_eval_spec_scope(
            cfg,
            {"scope": {"search_paths": ["python-backend", "src"]}},
        )
        assert cfg.bundle_search_paths == ["python-backend", "src"]

    def test_existing_scope_not_overwritten(self):
        """If a Config field is already set (e.g. via interactive prompts),
        the spec must not stomp on it."""
        cfg = _bare_cfg(
            optimizable_scope=["already.py"],
            context_scope=["already_ctx.py"],
            read_only_scope=["already_ro.py"],
            exclude_scope=["already_excl/**"],
        )
        apply_eval_spec_scope(
            cfg,
            {
                "scope": {
                    "optimizable_paths": ["from_spec.py"],
                    "context_paths": ["from_spec_ctx.py"],
                    "read_only_paths": ["from_spec_ro.py"],
                    "exclude_paths": ["from_spec_excl/**"],
                }
            },
        )
        assert cfg.optimizable_scope == ["already.py"]
        assert cfg.context_scope == ["already_ctx.py"]
        assert cfg.read_only_scope == ["already_ro.py"]
        assert cfg.exclude_scope == ["already_excl/**"]

    def test_empty_spec_leaves_defaults(self):
        cfg = _bare_cfg()
        apply_eval_spec_scope(cfg, {})
        assert cfg.optimizable_scope == []
        assert cfg.read_only_scope == []

    def test_overlap_between_optimizable_and_read_only_is_rejected(self):
        """Listing a path in both ``optimizable_paths`` and ``read_only_paths``
        is a configuration mistake — fail fast at init rather than silently
        resolving one way."""
        cfg = _bare_cfg()
        with pytest.raises(ValueError, match="overmind_entrypoint.py"):
            apply_eval_spec_scope(
                cfg,
                {
                    "scope": {
                        "optimizable_paths": ["agent.py", "overmind_entrypoint.py"],
                        "read_only_paths": ["overmind_entrypoint.py"],
                    }
                },
            )

    def test_no_overlap_passes(self):
        cfg = _bare_cfg()
        apply_eval_spec_scope(
            cfg,
            {
                "scope": {
                    "optimizable_paths": ["agent.py"],
                    "read_only_paths": ["entrypoint.py"],
                }
            },
        )
        assert cfg.optimizable_scope == ["agent.py"]
        assert cfg.read_only_scope == ["entrypoint.py"]


# ---------------------------------------------------------------------------
# eval_spec schema validation
# ---------------------------------------------------------------------------
#
# The evaluator iterates ``consistency_rules`` with ``rule.get(...)``.
# A list of natural-language strings (a tempting LLM output for that key)
# blew up the baseline run with ``AttributeError: 'str' object has no
# attribute 'get'``. These tests pin the new fail-fast validation so the
# error surfaces at spec-load time with a JSON path to the offender,
# instead of mid-eval as a cryptic AttributeError.


class TestValidateEvalSpec:
    def test_well_formed_spec_passes(self):
        validate_eval_spec({
            "output_fields": {"answer": {"type": "text", "weight": 50}},
            "consistency_rules": [
                {
                    "field_a": "tool_calls",
                    "field_b": "tool_calls",
                    "type": "ordering",
                    "operator": "<=",
                    "penalty": 0,
                }
            ],
            "scope": {"optimizable_paths": ["agent.py"]},
        })

    def test_top_level_must_be_object(self):
        with pytest.raises(SpecValidationError, match="eval_spec"):
            validate_eval_spec("not an object")  # type: ignore[arg-type]

    def test_string_rule_rejected_with_index(self):
        """The exact airline-run regression: LLM-style natural-language
        strings in consistency_rules must be rejected, and the message
        must point at the offending index."""
        with pytest.raises(SpecValidationError, match=r"consistency_rules\[0\]"):
            validate_eval_spec({
                "output_fields": {"answer": {"type": "text"}},
                "consistency_rules": [
                    "Tool calls should match the question type",
                ],
            })

    def test_missing_field_a_rejected(self):
        with pytest.raises(SpecValidationError, match=r"consistency_rules\[1\].field_a"):
            validate_eval_spec({
                "output_fields": {},
                "consistency_rules": [
                    {"field_a": "a", "field_b": "b"},
                    {"field_b": "b"},
                ],
            })

    def test_unknown_rule_type_rejected(self):
        with pytest.raises(SpecValidationError, match="type"):
            validate_eval_spec({
                "output_fields": {},
                "consistency_rules": [
                    {"field_a": "a", "field_b": "b", "type": "magic"},
                ],
            })

    def test_unknown_operator_rejected(self):
        with pytest.raises(SpecValidationError, match="operator"):
            validate_eval_spec({
                "output_fields": {},
                "consistency_rules": [
                    {
                        "field_a": "a",
                        "field_b": "b",
                        "type": "ordering",
                        "operator": "≈",
                    },
                ],
            })

    def test_non_numeric_penalty_rejected(self):
        with pytest.raises(SpecValidationError, match="penalty"):
            validate_eval_spec({
                "output_fields": {},
                "consistency_rules": [
                    {
                        "field_a": "a",
                        "field_b": "b",
                        "penalty": "high",
                    },
                ],
            })

    def test_consistency_rules_not_a_list_rejected(self):
        with pytest.raises(SpecValidationError, match="consistency_rules"):
            validate_eval_spec({
                "output_fields": {},
                "consistency_rules": {"a": 1},
            })

    def test_scope_paths_must_be_strings(self):
        with pytest.raises(SpecValidationError, match=r"scope.optimizable_paths"):
            validate_eval_spec({
                "output_fields": {},
                "scope": {"optimizable_paths": ["a.py", 123]},
            })

    def test_scope_must_be_object(self):
        with pytest.raises(SpecValidationError, match="scope"):
            validate_eval_spec({
                "output_fields": {},
                "scope": "everything",
            })

    def test_output_fields_must_be_object(self):
        with pytest.raises(SpecValidationError, match="output_fields"):
            validate_eval_spec({"output_fields": ["not", "an", "object"]})

    def test_output_field_weight_must_be_numeric(self):
        with pytest.raises(SpecValidationError, match=r"output_fields.answer.weight"):
            validate_eval_spec({
                "output_fields": {"answer": {"weight": "lots"}},
            })

    def test_missing_optional_blocks_ok(self):
        """Specs without consistency_rules or scope are valid — the
        validator only fires when keys are present."""
        validate_eval_spec({"output_fields": {"a": {"type": "text"}}})

    def test_apply_eval_spec_scope_propagates_validation_error(self):
        """The hook must reject bad specs before scope merging runs."""
        cfg = _bare_cfg()
        with pytest.raises(SpecValidationError, match=r"consistency_rules\[0\]"):
            apply_eval_spec_scope(cfg, {"consistency_rules": ["plain string"]})


# ---------------------------------------------------------------------------
# File-level glob overlap detection
# ---------------------------------------------------------------------------
#
# The literal-pattern overlap check catches obvious mistakes but
# misses the case where two different-looking globs expand to the
# same files (e.g. ``**/*.py`` and ``entry.py``). This tier-2 check
# resolves both lists against the filesystem and raises on
# intersection.


class TestScopeFileLevelOverlap:
    def _make_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "agent.py").write_text("def run(x): return x\n")
        (tmp_path / "entry.py").write_text("def run(x): return x\n")
        (tmp_path / "helper.py").write_text("\n")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_glob_overlap_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``**/*.py`` covers everything, so listing ``entry.py`` as
        read-only is a contradiction. The file-level check catches it
        even though the string-level check would not."""
        project = self._make_project(tmp_path, monkeypatch)
        cfg = Config(
            agent_name="x",
            agent_path=str(project / "agent.py"),
            entrypoint_fn="run",
        )
        with pytest.raises(ValueError, match="resolve to overlapping files"):
            apply_eval_spec_scope(
                cfg,
                {
                    "scope": {
                        "optimizable_paths": ["**/*.py"],
                        "read_only_paths": ["entry.py"],
                    }
                },
            )

    def test_glob_overlap_message_lists_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project = self._make_project(tmp_path, monkeypatch)
        cfg = Config(
            agent_name="x",
            agent_path=str(project / "agent.py"),
            entrypoint_fn="run",
        )
        with pytest.raises(ValueError, match="entry.py"):
            apply_eval_spec_scope(
                cfg,
                {
                    "scope": {
                        "optimizable_paths": ["**/*.py"],
                        "read_only_paths": ["entry.py"],
                    }
                },
            )

    def test_disjoint_globs_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Patterns that match disjoint sets must not trigger the
        overlap check, even if they're both globs."""
        project = self._make_project(tmp_path, monkeypatch)
        (project / "fixtures").mkdir()
        (project / "fixtures" / "data.py").write_text("")
        cfg = Config(
            agent_name="x",
            agent_path=str(project / "agent.py"),
            entrypoint_fn="run",
        )
        apply_eval_spec_scope(
            cfg,
            {
                "scope": {
                    "optimizable_paths": ["agent.py", "helper.py"],
                    "read_only_paths": ["fixtures/*.py"],
                }
            },
        )
        assert cfg.optimizable_scope == ["agent.py", "helper.py"]
        assert cfg.read_only_scope == ["fixtures/*.py"]

    def test_missing_project_root_falls_back_to_literal_check(self):
        """When the agent path doesn't resolve to a real project root
        (synthetic tests), file-level expansion is skipped — the
        literal-equality check still runs."""
        cfg = _bare_cfg()  # agent_path="/p" — doesn't exist
        # Literal overlap still raises.
        with pytest.raises(ValueError, match="overmind_entrypoint.py"):
            apply_eval_spec_scope(
                cfg,
                {
                    "scope": {
                        "optimizable_paths": [
                            "agent.py",
                            "overmind_entrypoint.py",
                        ],
                        "read_only_paths": ["overmind_entrypoint.py"],
                    }
                },
            )
        # Non-literal disjoint globs pass (no filesystem to expand).
        cfg2 = _bare_cfg()
        apply_eval_spec_scope(
            cfg2,
            {
                "scope": {
                    "optimizable_paths": ["**/*.py"],
                    "read_only_paths": ["entry.py"],
                }
            },
        )
