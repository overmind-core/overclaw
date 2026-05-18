"""Tests for overmind.setup.spec_generator — eval spec construction."""

from __future__ import annotations

import json
from pathlib import Path


from overmind.setup.spec_generator import (
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


class TestScopeReadOnlyDefault:
    """The spec generator must protect the registered Overmind entrypoint
    from being edited by candidates, even when the analyzer leaves
    ``scope.read_only_paths`` empty. The accept step only enforces
    what's declared in the spec; without an auto-default here, harness files
    silently become editable."""

    def test_empty_scope_gets_entrypoint_read_only(self):
        analysis = {"_entry_rel": "overmind_entrypoint.py"}
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert spec["scope"]["read_only_paths"] == ["overmind_entrypoint.py"]

    def test_entry_appended_when_other_read_only_paths_exist(self):
        """LLM declared fixtures but forgot the entrypoint — we must
        still protect it."""
        analysis = {
            "_entry_rel": "overmind_entrypoint.py",
            "scope": {
                "optimizable_paths": ["agent/**/*.py"],
                "read_only_paths": ["tests/fixtures/*.json"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        ro = spec["scope"]["read_only_paths"]
        assert "overmind_entrypoint.py" in ro
        assert "tests/fixtures/*.json" in ro

    def test_entry_in_optimizable_is_not_added_to_read_only(self):
        """Single-file agent: the entry IS the agent. Auto-add must not
        push it into read_only and contradict optimizable."""
        analysis = {
            "_entry_rel": "agent.py",
            "scope": {"optimizable_paths": ["agent.py"]},
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "read_only_paths" not in spec["scope"]

    def test_entry_already_in_read_only_not_duplicated(self):
        analysis = {
            "_entry_rel": "overmind_entrypoint.py",
            "scope": {
                "optimizable_paths": ["agent/**/*.py"],
                "read_only_paths": ["overmind_entrypoint.py"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        ro = spec["scope"]["read_only_paths"]
        assert ro.count("overmind_entrypoint.py") == 1

    def test_no_entry_rel_no_scope_change(self):
        """When the analyzer step didn't stash ``_entry_rel`` (older
        callers, fallback paths), the generator must not crash; it
        simply skips the auto-default."""
        analysis = {"scope": {"optimizable_paths": ["agent.py"]}}
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert spec["scope"] == {"optimizable_paths": ["agent.py"]}


# ---------------------------------------------------------------------------
# Deterministic ``search_paths`` injection from entry sys.path mutations
# ---------------------------------------------------------------------------
#
# The analyzer prompt asks the LLM to emit ``search_paths`` whenever the
# entry mutates ``sys.path``. The LLM forgets. ``_build_spec`` re-runs
# the AST partial evaluator and injects any missing paths so the spec
# is always self-consistent with the entry's actual runtime behaviour,
# regardless of how good the LLM was.


def _make_syspath_project(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tmp project with ``.overmind/`` marker, an entry that
    mutates ``sys.path``, and the actual ``py-backend/`` dir on disk so
    the AST evaluator validates it as a real directory."""
    from overmind.core.constants import OVERMIND_DIR_NAME

    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        '_BACKEND = Path(__file__).resolve().parent / "py-backend"\n'
        'if str(_BACKEND) not in sys.path:\n'
        '    sys.path.insert(0, str(_BACKEND))\n'
    )
    (tmp_path / "py-backend").mkdir()
    return tmp_path, entry


class TestSearchPathsInjection:
    def test_injects_when_analyzer_omits(self, tmp_path):
        """Spec without ``search_paths`` gets the entry-derived path
        injected automatically — this is the failure mode that occurs
        when the analyzer LLM forgets the declarative rule."""
        root, entry = _make_syspath_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {"optimizable_paths": ["entry.py"]},
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert spec["scope"].get("search_paths") == ["py-backend"]

    def test_noop_when_analyzer_already_declared(self, tmp_path):
        """Idempotent: a correct spec from the LLM passes through
        unchanged, no duplicate entries."""
        root, entry = _make_syspath_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["entry.py"],
                "search_paths": ["py-backend"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert spec["scope"]["search_paths"] == ["py-backend"]

    def test_appends_to_divergent_declaration(self, tmp_path):
        """If the LLM declared a *different* search path, the
        post-process appends the detected one rather than replacing.
        More search paths are safe; missing ones aren't."""
        root, entry = _make_syspath_project(tmp_path)
        # Create a second directory the LLM "thinks" is the package root
        # so it survives the relative-to-root validation.
        (tmp_path / "libs").mkdir()
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["entry.py"],
                "search_paths": ["libs"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "libs" in spec["scope"]["search_paths"]
        assert "py-backend" in spec["scope"]["search_paths"]

    def test_no_syspath_mutation_no_injection(self, tmp_path):
        """An entry without ``sys.path`` mutations leaves the scope
        untouched. ``search_paths`` is not added as an empty key."""
        from overmind.core.constants import OVERMIND_DIR_NAME

        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        entry = tmp_path / "entry.py"
        entry.write_text("def run(x): return x\n")

        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {"optimizable_paths": ["entry.py"]},
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "search_paths" not in spec["scope"]


# ---------------------------------------------------------------------------
# Two-scope output shape + legacy collapse
# ---------------------------------------------------------------------------
#
# The spec generator always emits two scope lists
# (``optimizable_paths`` + ``read_only_paths``). The LLM occasionally
# still proposes the older ``context_paths`` / ``exclude_paths`` shape,
# so ``_build_spec`` collapses those into the two-scope form:
# ``context_paths`` merges into ``read_only_paths``; ``exclude_paths``
# is dropped on the floor (project-level drops belong in
# ``.overmindignore`` or are already covered by Overmind's hard-coded
# skip list).
#
# It also auto-adds the entry file to ``read_only_paths`` so the
# accept step diff-checks prevent candidates from editing the
# registered harness even when the analyzer forgets to declare it.


def _trivial_entry_project(tmp_path: Path) -> tuple[Path, Path]:
    from overmind.core.constants import OVERMIND_DIR_NAME

    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    entry = tmp_path / "entry.py"
    entry.write_text("def run(x): return x\n")
    return tmp_path, entry


class TestTwoScopeOutputShape:
    def test_legacy_context_paths_collapse_into_read_only(self, tmp_path):
        """``context_paths`` is no longer emitted. Anything the LLM
        puts there is merged into ``read_only_paths`` (strictly safer:
        enforced at accept time)."""
        root, entry = _trivial_entry_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["agent_logic.py"],
                "context_paths": ["README.md", "policies.md"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "context_paths" not in spec["scope"]
        assert "README.md" in spec["scope"]["read_only_paths"]
        assert "policies.md" in spec["scope"]["read_only_paths"]

    def test_legacy_exclude_paths_are_dropped(self, tmp_path):
        """``exclude_paths`` is no longer emitted. Project-level drops
        belong in ``.overmindignore``; Overmind's hard-coded skip list
        handles env-level cases like ``__pycache__``."""
        root, entry = _trivial_entry_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["agent_logic.py"],
                "exclude_paths": ["tests/**", "**/__pycache__/**"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "exclude_paths" not in spec["scope"]

    def test_entry_auto_added_to_read_only(self, tmp_path):
        """The entry is the registered harness; the accept step
        diff-check should prevent candidate edits even when the
        analyzer forgets to declare it. ``_build_spec`` auto-adds."""
        root, entry = _trivial_entry_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["agent_logic.py"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "entry.py" in spec["scope"]["read_only_paths"]

    def test_entry_in_optimizable_is_left_editable(self, tmp_path):
        """Single-file agent edge case: the entry IS the agent under
        test. Don't add it to read_only — the candidate must be free
        to edit it."""
        root, entry = _trivial_entry_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["entry.py"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert "entry.py" in spec["scope"]["optimizable_paths"]
        assert "entry.py" not in spec["scope"].get("read_only_paths", [])

    def test_clean_spec_unchanged(self, tmp_path):
        """When the LLM emits the new two-scope shape correctly, the
        post-process is a no-op for these fields."""
        root, entry = _trivial_entry_project(tmp_path)
        analysis = {
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "scope": {
                "optimizable_paths": ["agent/**/*.py"],
                "read_only_paths": ["entry.py"],
            },
        }
        spec = _build_spec(analysis, {}, {}, {}, 20)
        assert spec["scope"]["read_only_paths"] == ["entry.py"]
        assert spec["scope"]["optimizable_paths"] == ["agent/**/*.py"]
        assert "context_paths" not in spec["scope"]
        assert "exclude_paths" not in spec["scope"]


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
