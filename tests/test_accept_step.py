"""Tests for overmind.optimize.steps.accept_step — read_only enforcement.

The accept step is the single chokepoint that promotes a candidate as
the new best agent. Anything we want to *enforce* about candidates
(rather than merely *prompt* about) must live here. These tests pin the
read-only invariant:

1. A candidate worktree that mutated a file declared in
   ``read_only_paths`` is rejected even if its evaluation score is the
   highest.
2. When *every* candidate violates the invariant, the run stalls with a
   distinct ``decision == 'read_only_violation'`` envelope rather than
   masquerading as a crash.
3. The byte-equality diff catches deletes, edits, and unreadable files.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from pathlib import Path

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.optimize.config import Config
from overmind.optimize.steps.accept_step import (
    _candidate_violates_read_only,
    _load_or_build_read_only_baseline,
    _read_only_cache_key,
    run_accept,
)
from overmind.optimize.steps.state import SkillRunState

# ---------------------------------------------------------------------------
# _candidate_violates_read_only — unit tests
# ---------------------------------------------------------------------------


class TestCandidateViolatesReadOnly:
    def test_clean_candidate_returns_empty(self, tmp_path: Path):
        baseline = {"entry.py": "def run(x):\n    return x\n"}
        cand_dir = tmp_path / "c1"
        cand_dir.mkdir()
        (cand_dir / "entry.py").write_text(baseline["entry.py"])
        assert (
            _candidate_violates_read_only(str(cand_dir), {"entry.py"}, baseline) == []
        )

    def test_modified_file_is_flagged(self, tmp_path: Path):
        baseline = {"entry.py": "def run(x):\n    return x\n"}
        cand_dir = tmp_path / "c1"
        cand_dir.mkdir()
        (cand_dir / "entry.py").write_text("def run(x):\n    return 99\n")
        assert _candidate_violates_read_only(str(cand_dir), {"entry.py"}, baseline) == [
            "entry.py"
        ]

    def test_whitespace_only_change_is_flagged(self, tmp_path: Path):
        """Strict byte-equality: trailing whitespace is still a mutation."""
        baseline = {"entry.py": "def run(x):\n    return x\n"}
        cand_dir = tmp_path / "c1"
        cand_dir.mkdir()
        (cand_dir / "entry.py").write_text("def run(x):\n    return x\n\n")
        assert _candidate_violates_read_only(str(cand_dir), {"entry.py"}, baseline) == [
            "entry.py"
        ]

    def test_missing_file_is_flagged(self, tmp_path: Path):
        baseline = {"entry.py": "def run(x):\n    return x\n"}
        cand_dir = tmp_path / "c1"
        cand_dir.mkdir()
        assert _candidate_violates_read_only(str(cand_dir), {"entry.py"}, baseline) == [
            "entry.py"
        ]

    def test_unknown_path_skipped(self, tmp_path: Path):
        """Paths in read_only_paths that aren't in baseline_files are
        silently skipped so a misconfigured spec can't false-positive."""
        baseline = {"entry.py": "x"}
        cand_dir = tmp_path / "c1"
        cand_dir.mkdir()
        (cand_dir / "entry.py").write_text("x")
        assert (
            _candidate_violates_read_only(
                str(cand_dir), {"entry.py", "ghost.py"}, baseline
            )
            == []
        )

    def test_empty_read_only_set_short_circuits(self, tmp_path: Path):
        assert _candidate_violates_read_only(str(tmp_path), set(), {"a": "b"}) == []

    def test_empty_baseline_short_circuits(self, tmp_path: Path):
        assert _candidate_violates_read_only(str(tmp_path), {"a"}, {}) == []


# ---------------------------------------------------------------------------
# run_accept — integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _accept_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)


def _make_run_accept_setup(
    tmp_path: Path, with_read_only: bool = True
) -> tuple[SkillRunState, Path]:
    """Build a minimal project + state for ``run_accept`` to consume.

    Returns ``(state, project_root)``.
    """
    agent_dir = tmp_path / "agents" / "test"
    agent_dir.mkdir(parents=True)
    entry_file = agent_dir / "entrypoint.py"
    entry_file.write_text(
        textwrap.dedent("""\
        from logic import answer

        def run(input_data: dict) -> dict:
            return {"qualification": answer()}
        """),
        encoding="utf-8",
    )
    (agent_dir / "logic.py").write_text(
        textwrap.dedent("""\
        def answer():
            return "hot"
        """),
        encoding="utf-8",
    )

    spec = {
        "agent_description": "Test agent",
        "output_fields": {
            "qualification": {
                "type": "enum",
                "weight": 80,
                "values": ["hot", "warm", "cold"],
                "importance": "critical",
            },
        },
        "structure_weight": 20,
        "total_points": 100,
    }
    if with_read_only:
        spec["scope"] = {
            "optimizable_paths": ["agents/test/logic.py"],
            "read_only_paths": ["agents/test/entrypoint.py"],
        }
    else:
        spec["scope"] = {"optimizable_paths": ["agents/test/logic.py"]}

    spec_dir = agent_dir / "setup_spec"
    spec_dir.mkdir()
    spec_path = spec_dir / "eval_spec.json"
    spec_path.write_text(json.dumps(spec))

    data_path = spec_dir / "dataset.json"
    data_path.write_text(
        json.dumps(
            [{"input": {"company": "Acme"}, "expected_output": {"qualification": "hot"}}]
        )
    )

    cfg = Config(
        agent_name="test-agent",
        agent_path=str(entry_file),
        entrypoint_fn="run",
        eval_spec_path=str(spec_path),
        data_path=str(data_path),
        analyzer_model="test-model",
        iterations=1,
        candidates_per_iteration=2,
        parallel=False,
        holdout_ratio=0.0,
        early_stopping_patience=0,
        optimizable_scope=["agents/test/logic.py"],
        read_only_scope=["agents/test/entrypoint.py"] if with_read_only else [],
    )

    state_path = (
        tmp_path / OVERMIND_DIR_NAME / "agents" / "test-agent" / "experiments"
    )
    state_path.mkdir(parents=True)
    state = SkillRunState.from_config(
        agent_name="test-agent",
        config=cfg,
        state_path=str(state_path / "skill_state.json"),
    )
    state.best_score = 50.0
    state.config = asdict(cfg)
    state.save()

    return state, tmp_path


def _write_candidate(
    project_root: Path,
    cand_id: str,
    score: float,
    *,
    entry_contents: str | None,
    logic_contents: str = 'def answer():\n    return "warm"\n',
) -> dict:
    """Materialize a candidate worktree + score.json. Returns the candidate
    record that ``run_accept`` expects in ``candidate_results``."""
    cand_dir = project_root / "experiments" / cand_id
    cand_dir.mkdir(parents=True)
    agent_subdir = cand_dir / "agents" / "test"
    agent_subdir.mkdir(parents=True)
    if entry_contents is not None:
        (agent_subdir / "entrypoint.py").write_text(entry_contents)
    (agent_subdir / "logic.py").write_text(logic_contents)

    score_path = cand_dir / "score.json"
    score_path.write_text(
        json.dumps(
            {
                "avg_total": score,
                "evaluation": {"avg_total": score},
                "case_results": [
                    {
                        "input": {},
                        "expected": {},
                        "output": {},
                        "score": {"total": score},
                    }
                ],
            }
        )
    )

    entry_path = agent_subdir / "entrypoint.py"
    return {
        "candidate_id": cand_id,
        "candidate_dir": str(cand_dir),
        "entry_path": str(entry_path),
        "score_path": str(score_path),
    }


def _baseline_entry_text(project_root: Path) -> str:
    return (project_root / "agents" / "test" / "entrypoint.py").read_text()


class TestRunAcceptReadOnlyEnforcement:
    def test_high_scoring_candidate_rejected_if_modifies_read_only(
        self, tmp_path: Path
    ):
        state, root = _make_run_accept_setup(tmp_path, with_read_only=True)
        baseline_entry = _baseline_entry_text(root)

        # c1: highest score but mutated the entrypoint → must be rejected.
        c1 = _write_candidate(
            root,
            "c1",
            score=95.0,
            entry_contents=baseline_entry + "\n# subtly different\n",
        )
        # c2: lower score, clean → wins.
        c2 = _write_candidate(
            root, "c2", score=70.0, entry_contents=baseline_entry
        )

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps([c1, c2]))

        envelope = run_accept(
            state, iteration=1, candidate_results_path=str(results_path)
        )

        assert envelope["decision"] == "accept"
        assert envelope["winner"]["candidate_id"] == "c2"
        assert envelope["winner"]["avg_total"] == 70.0
        assert "read_only_violations" in envelope
        assert envelope["read_only_violations"][0]["candidate_id"] == "c1"
        assert "agents/test/entrypoint.py" in envelope["read_only_violations"][0][
            "files"
        ]

    def test_all_dirty_returns_read_only_violation_decision(self, tmp_path: Path):
        state, root = _make_run_accept_setup(tmp_path, with_read_only=True)
        baseline_entry = _baseline_entry_text(root)
        c1 = _write_candidate(
            root, "c1", score=95.0, entry_contents=baseline_entry + "# x\n"
        )
        c2 = _write_candidate(
            root, "c2", score=80.0, entry_contents=baseline_entry + "# y\n"
        )

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps([c1, c2]))

        envelope = run_accept(
            state, iteration=1, candidate_results_path=str(results_path)
        )

        assert envelope["decision"] == "read_only_violation"
        assert envelope["best_score"] == 50.0  # unchanged
        assert envelope["stall_count"] == 1
        assert {v["candidate_id"] for v in envelope["violations"]} == {"c1", "c2"}

    def test_clean_candidates_still_promote_when_above_best(self, tmp_path: Path):
        state, root = _make_run_accept_setup(tmp_path, with_read_only=True)
        baseline_entry = _baseline_entry_text(root)
        c1 = _write_candidate(root, "c1", score=90.0, entry_contents=baseline_entry)

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps([c1]))

        envelope = run_accept(
            state, iteration=1, candidate_results_path=str(results_path)
        )

        assert envelope["decision"] == "accept"
        assert envelope["winner"]["candidate_id"] == "c1"
        assert "read_only_violations" not in envelope

    def test_disabled_enforcement_skips_diff(self, tmp_path: Path):
        """When ``read_only_scope`` is empty, candidates can edit anything
        (existing behavior preserved)."""
        state, root = _make_run_accept_setup(tmp_path, with_read_only=False)
        baseline_entry = _baseline_entry_text(root)
        c1 = _write_candidate(
            root, "c1", score=95.0, entry_contents=baseline_entry + "# free\n"
        )

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps([c1]))

        envelope = run_accept(
            state, iteration=1, candidate_results_path=str(results_path)
        )

        assert envelope["decision"] == "accept"
        assert envelope["winner"]["candidate_id"] == "c1"
        assert "read_only_violations" not in envelope


# ---------------------------------------------------------------------------
# Read-only baseline caching
# ---------------------------------------------------------------------------
#
# Before the patch ``run_accept`` rebuilt the bundle every iteration
# just to recover the read-only baseline. The cache lives on
# ``SkillRunState`` and is keyed by the sorted ``read_only_scope`` so
# legitimate config changes invalidate it. These tests pin the cache
# contract.


class TestReadOnlyBaselineCache:
    def test_cache_key_is_order_independent(self):
        assert _read_only_cache_key(["b", "a"]) == _read_only_cache_key(
            ["a", "b"]
        )

    def test_cache_key_changes_when_scope_changes(self):
        assert _read_only_cache_key(["a"]) != _read_only_cache_key(["a", "b"])

    def test_first_call_populates_cache_and_saves(self, tmp_path: Path):
        state, _root = _make_run_accept_setup(tmp_path, with_read_only=True)
        assert state.read_only_baseline == {}
        assert state.read_only_baseline_key == ""

        cfg = state.to_config()
        baseline, ro_paths = _load_or_build_read_only_baseline(state, cfg)

        assert "agents/test/entrypoint.py" in baseline
        assert ro_paths == {"agents/test/entrypoint.py"}
        # Persisted onto state and saved.
        assert state.read_only_baseline == baseline
        assert state.read_only_baseline_key == _read_only_cache_key(
            cfg.read_only_scope
        )
        # Reload from disk to confirm save() flushed.
        reloaded = SkillRunState.load(state.state_path)
        assert reloaded.read_only_baseline == baseline

    def test_second_call_reuses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The second call must not rebuild the bundle — we monkeypatch
        ``Optimizer`` to crash so a re-entry would be visible."""
        state, _root = _make_run_accept_setup(tmp_path, with_read_only=True)
        cfg = state.to_config()

        # First call — builds and caches.
        baseline_first, ro_paths_first = _load_or_build_read_only_baseline(
            state, cfg
        )

        # Sabotage Optimizer so any rebuild attempt would fail loudly.
        def _explode(*args, **kwargs):
            raise AssertionError("bundle should not be rebuilt")

        monkeypatch.setattr(
            "overmind.optimize.steps.accept_step.Optimizer", _explode
        )

        # Second call — must hit the cache.
        baseline_second, ro_paths_second = _load_or_build_read_only_baseline(
            state, cfg
        )
        assert baseline_second == baseline_first
        assert ro_paths_second == ro_paths_first

    def test_cache_invalidated_when_scope_changes(self, tmp_path: Path):
        """Editing the spec mid-run (adding a new fixture to
        read_only_paths) must invalidate the cache and trigger a
        rebuild — otherwise the new file isn't enforced."""
        state, _root = _make_run_accept_setup(tmp_path, with_read_only=True)
        cfg = state.to_config()
        _load_or_build_read_only_baseline(state, cfg)
        first_key = state.read_only_baseline_key

        # Simulate a spec edit: add another file to read_only_scope.
        cfg.read_only_scope = list(cfg.read_only_scope) + ["agents/test/logic.py"]
        _load_or_build_read_only_baseline(state, cfg)
        assert state.read_only_baseline_key != first_key

    def test_state_serialisation_roundtrip(self, tmp_path: Path):
        """Cache survives a save/load roundtrip — that's what makes
        cross-step persistence work for cluster/CI runners that don't
        keep the SkillRunState in memory."""
        state, _root = _make_run_accept_setup(tmp_path, with_read_only=True)
        cfg = state.to_config()
        _load_or_build_read_only_baseline(state, cfg)

        reloaded = SkillRunState.load(state.state_path)
        assert reloaded.read_only_baseline == state.read_only_baseline
        assert reloaded.read_only_baseline_key == state.read_only_baseline_key
