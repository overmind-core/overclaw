"""Tests for Optimizer.run() — the full optimization pipeline."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.optimize.config import Config
from overclaw.optimize.optimizer import Optimizer


@pytest.fixture(autouse=True)
def _optimizer_run_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / OVERCLAW_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)


def _make_config(tmp_path: Path) -> Config:
    agent_dir = tmp_path / "agents" / "test"
    agent_dir.mkdir(parents=True)
    agent_file = agent_dir / "sample_agent.py"
    agent_file.write_text(
        textwrap.dedent("""\
        def run(input_data: dict) -> dict:
            return {"qualification": "hot", "score": 80, "reasoning": "test", "is_enterprise": True}
    """)
    )

    spec = {
        "output_fields": {
            "qualification": {
                "type": "enum",
                "weight": 30,
                "values": ["hot", "warm", "cold"],
                "importance": "critical",
                "partial_credit": True,
                "partial_score": 6,
            },
            "score": {
                "type": "number",
                "weight": 20,
                "range": [0, 100],
                "tolerance": 10,
                "tolerance_bands": [
                    {"within": 5, "score_pct": 1.0},
                    {"within": 10, "score_pct": 0.8},
                ],
            },
            "reasoning": {"type": "text", "weight": 15, "eval_mode": "non_empty"},
            "is_enterprise": {"type": "boolean", "weight": 15},
        },
        "structure_weight": 20,
        "total_points": 100,
    }
    spec_dir = agent_dir / "setup_spec"
    spec_dir.mkdir()
    spec_path = spec_dir / "eval_spec.json"
    spec_path.write_text(json.dumps(spec))

    dataset = [
        {
            "input": {"company": "Acme"},
            "expected_output": {
                "qualification": "hot",
                "score": 80,
                "reasoning": "big",
                "is_enterprise": True,
            },
        },
        {
            "input": {"company": "Small"},
            "expected_output": {
                "qualification": "cold",
                "score": 20,
                "reasoning": "small",
                "is_enterprise": False,
            },
        },
        {
            "input": {"company": "Med"},
            "expected_output": {
                "qualification": "warm",
                "score": 50,
                "reasoning": "mid",
                "is_enterprise": False,
            },
        },
        {
            "input": {"company": "Big"},
            "expected_output": {
                "qualification": "hot",
                "score": 90,
                "reasoning": "huge",
                "is_enterprise": True,
            },
        },
        {
            "input": {"company": "Tiny"},
            "expected_output": {
                "qualification": "cold",
                "score": 10,
                "reasoning": "tiny",
                "is_enterprise": False,
            },
        },
    ]
    data_path = spec_dir / "dataset.json"
    data_path.write_text(json.dumps(dataset))

    return Config(
        agent_name="test-agent",
        agent_path=str(agent_file),
        entrypoint_fn="run",
        eval_spec_path=str(spec_path),
        data_path=str(data_path),
        analyzer_model="test-model",
        iterations=1,
        candidates_per_iteration=1,
        parallel=False,
        holdout_ratio=0.0,
        early_stopping_patience=0,
        smoke_test_cases=0,
    )


class TestOptimizerRun:
    @patch("overclaw.optimize.optimizer.time.sleep")
    @patch("overclaw.optimize.optimizer.generate_candidates")
    def test_full_run_with_improvement(self, mock_gen, mock_sleep, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)

        improved_code = textwrap.dedent("""\
            def run(input_data: dict) -> dict:
                score = 80
                return {"qualification": "hot", "score": score, "reasoning": "improved", "is_enterprise": True}
        """)
        mock_gen.return_value = [
            {
                "updated_code": improved_code,
                "method": "codegen",
                "suggestions": ["Improved scoring"],
                "diagnosis": {"root_cause": "test issue"},
            }
        ]

        opt.run()
        assert opt.best_score > 0
        assert (opt.output_dir / "best_agent.py").exists()
        assert (opt.output_dir / "report.md").exists()

    @patch("overclaw.optimize.optimizer.generate_candidates")
    def test_run_with_no_valid_candidates(self, mock_gen, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)

        mock_gen.return_value = [{"updated_code": None, "method": "failed"}]

        opt.run()
        assert (opt.output_dir / "best_agent.py").exists()

    @patch("overclaw.optimize.optimizer.generate_candidates")
    def test_run_with_analyzer_error(self, mock_gen, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)

        mock_gen.side_effect = RuntimeError("API error")

        opt.run()
        assert opt.stall_count >= 1

    @patch("overclaw.optimize.optimizer.time.sleep")
    @patch("overclaw.optimize.optimizer.generate_candidates")
    def test_run_with_holdout(self, mock_gen, mock_sleep, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.holdout_ratio = 0.2
        opt = Optimizer(cfg)

        mock_gen.return_value = [
            {
                "updated_code": "def run(x):\n    return {'qualification': 'hot', 'score': 85, 'reasoning': 'ok', 'is_enterprise': True}\n",
                "method": "codegen",
                "suggestions": ["Minor tweak"],
            }
        ]

        opt.run()
        assert hasattr(opt, "_holdout_results")

    @patch("overclaw.optimize.optimizer.generate_candidates")
    def test_run_with_syntax_error_candidate(self, mock_gen, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)

        mock_gen.return_value = [
            {
                "updated_code": "def run(x:\n    return {}",
                "method": "codegen",
                "suggestions": ["Bad code"],
            }
        ]

        opt.run()
        assert (opt.output_dir / "best_agent.py").exists()


class TestRunMultiEval:
    def test_multi_eval(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()

        result_eval, result_items = opt._run_multi_eval(
            cfg.agent_path,
            [
                {
                    "input": {"company": "Test"},
                    "expected_output": {
                        "qualification": "hot",
                        "score": 80,
                        "reasoning": "ok",
                        "is_enterprise": True,
                    },
                }
            ],
            "multi_test",
            2,
        )
        assert "avg_total" in result_eval
        assert "_stdev" in result_eval


class TestRunParallel:
    def test_parallel_execution(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.parallel = True
        cfg.max_workers = 2
        opt = Optimizer(cfg)
        opt._setup_output_dirs()

        agent = Optimizer._load_agent_module(cfg.agent_path)
        dataset = [
            {
                "input": {"company": "A"},
                "expected_output": {
                    "qualification": "hot",
                    "score": 80,
                    "reasoning": "ok",
                    "is_enterprise": True,
                },
            },
            {
                "input": {"company": "B"},
                "expected_output": {
                    "qualification": "cold",
                    "score": 20,
                    "reasoning": "no",
                    "is_enterprise": False,
                },
            },
        ]
        eval_result, tracers, items = opt._run_parallel(
            agent, cfg.agent_path, dataset, "par_test"
        )
        assert len(items) == 2
        assert "avg_total" in eval_result
