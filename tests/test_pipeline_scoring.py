"""Drift-guard tests for :mod:`overmind.optimize.pipeline.scoring`.

These tests pin the behaviour of the pure scoring helpers that
``Optimizer._compute_complexity_penalty`` (and friends) delegate to.  They
exist for two reasons:

1. Catch silent regressions when the scoring math is touched.
2. Verify ``Optimizer`` exposes the same numbers as the pure helpers — if a
   refactor accidentally re-implements the math on the optimizer instance
   the two will drift apart and these tests will fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from overmind.optimize.optimizer import Optimizer
from overmind.optimize.pipeline import scoring


class TestPromptSize:
    def test_extracts_triple_quoted_prompt(self):
        code = 'SYSTEM_PROMPT = """hello"""'
        assert scoring.prompt_size(code) == len("hello")

    def test_returns_zero_when_no_prompt(self):
        assert scoring.prompt_size("def run(x): pass") == 0


class TestCounters:
    def test_counts_conditional_branches(self):
        code = "if x:\n    pass\nelif y:\n    pass\nif(z):\n    pass\n"
        assert scoring.count_conditional_branches(code) == 3

    def test_counts_function_defs(self):
        code = "def a():\n    pass\nasync def b():\n    pass\n"
        assert scoring.count_function_defs(code) == 2


class TestDetectDataLeakage:
    def test_no_leakage_when_candidate_identical(self):
        baseline = "def run(x): return {}"
        train = [{"expected_output": {"v": "longvalue123"}}]
        assert scoring.detect_data_leakage(baseline, baseline, train) == 0

    def test_detects_added_literal(self):
        baseline = "def run(x): return {}"
        candidate = baseline + "\nLEAK = 'longvalue123'"
        train = [{"expected_output": {"v": "longvalue123"}}]
        assert scoring.detect_data_leakage(candidate, baseline, train) >= 1

    def test_skips_known_domain_values(self):
        baseline = "def run(x): return {}"
        candidate = baseline + "\nCAT = 'longvalue123'"
        train = [{"expected_output": {"v": "longvalue123"}}]
        assert (
            scoring.detect_data_leakage(
                candidate,
                baseline,
                train,
                known_domain_values={"longvalue123"},
            )
            == 0
        )

    def test_skips_short_literals(self):
        baseline = "def run(x): return {}"
        candidate = baseline + "\nLEAK = 'warm'"
        train = [{"expected_output": {"v": "warm"}}]
        assert scoring.detect_data_leakage(candidate, baseline, train) == 0


class TestComputeComplexityPenalty:
    def test_no_penalty_when_candidate_unchanged(self):
        baseline = "def run(x):\n    return {}\n"
        assert (
            scoring.compute_complexity_penalty(
                baseline, baseline_code=baseline, best_code=baseline, best_score=0.0
            )
            == 0.0
        )

    def test_penalises_prompt_bloat(self):
        baseline = 'SYSTEM_PROMPT = """short"""\n'
        candidate = 'SYSTEM_PROMPT = """' + ("verylongprompt" * 50) + '"""\n'
        penalty = scoring.compute_complexity_penalty(
            candidate, baseline_code=baseline, best_code=baseline, best_score=0.0
        )
        assert penalty > 0

    def test_penalises_code_growth(self):
        baseline = "def run(x):\n    return {}\n"
        candidate = "def run(x):\n" + "    x = 1\n" * 200
        penalty = scoring.compute_complexity_penalty(
            candidate, baseline_code=baseline, best_code=baseline, best_score=0.0
        )
        assert penalty > 0

    def test_penalty_capped_by_raw_improvement(self):
        baseline = "def run(x):\n    return {}\n"
        candidate = "def run(x):\n" + "    x = 1\n" * 500
        capped = scoring.compute_complexity_penalty(
            candidate,
            baseline_code=baseline,
            best_code=baseline,
            best_score=0.0,
            raw_score=1.0,
        )
        assert capped <= 0.6


class TestOptimizerDelegation:
    """Optimizer.* methods must produce identical results to the pure helpers."""

    def _stub_optimizer(self, baseline: str) -> Optimizer:
        opt = Optimizer.__new__(Optimizer)
        opt._baseline_code = baseline
        opt.best_code = baseline
        opt.best_score = 0.0
        opt.config = MagicMock(max_code_growth_ratio=2.5)
        opt.evaluator = MagicMock()
        opt.evaluator.spec = {"output_fields": {}}
        return opt

    def test_complexity_penalty_matches_pure_helper(self):
        baseline = "def run(x):\n    return {}\n"
        candidate = "def run(x):\n" + "    x = 1\n" * 100
        opt = self._stub_optimizer(baseline)
        from_method = opt._compute_complexity_penalty(candidate, raw_score=None)
        from_pure = scoring.compute_complexity_penalty(
            candidate,
            baseline_code=baseline,
            best_code=baseline,
            best_score=0.0,
            train_set=None,
            raw_score=None,
            max_code_growth_ratio=2.5,
            known_domain_values=set(),
        )
        assert from_method == from_pure

    def test_data_leakage_matches_pure_helper(self):
        baseline = "def run(x):\n    return {}\n"
        candidate = baseline + "\nLEAK = 'longvalue123'"
        train = [{"expected_output": {"v": "longvalue123"}}]
        opt = self._stub_optimizer(baseline)
        from_method = opt._detect_data_leakage(candidate, train)
        from_pure = scoring.detect_data_leakage(candidate, baseline, train)
        assert from_method == from_pure

    @pytest.mark.parametrize(
        "code",
        [
            "",
            "x = 1",
            'SYSTEM_PROMPT = """hi"""',
            "def a():\n    pass\nif x:\n    pass\n",
        ],
    )
    def test_static_helpers_match_pure_helpers(self, code: str):
        assert Optimizer._get_prompt_size(code) == scoring.prompt_size(code)
        assert Optimizer._count_conditional_branches(code) == scoring.count_conditional_branches(code)
        assert Optimizer._count_function_defs(code) == scoring.count_function_defs(code)


class TestPipelinePackageSurface:
    def test_exposes_public_helpers(self):
        from overmind.optimize import pipeline

        for name in (
            "compute_complexity_penalty",
            "count_conditional_branches",
            "count_function_defs",
            "detect_data_leakage",
            "prompt_size",
        ):
            assert hasattr(pipeline, name), f"pipeline.{name} should be exported"
