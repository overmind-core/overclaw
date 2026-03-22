"""Tests for overclaw.optimize.optimizer — the core optimization loop."""

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
def _optimizer_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / OVERCLAW_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)


def _make_config(
    tmp_path: Path, spec: dict | None = None, dataset: list | None = None
) -> Config:
    agent_dir = tmp_path / "agents" / "test"
    agent_dir.mkdir(parents=True)
    agent_file = agent_dir / "sample_agent.py"
    agent_file.write_text(
        textwrap.dedent("""\
        SYSTEM_PROMPT = \"\"\"You are a test agent.\"\"\"
        MODEL = "gpt-5.4-mini"
        def run(input_data: dict) -> dict:
            return {"qualification": "hot", "score": 80, "reasoning": "test", "is_enterprise": True}
    """)
    )

    spec = spec or {
        "agent_description": "Test agent",
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
            "reasoning": {
                "type": "text",
                "weight": 15,
                "eval_mode": "non_empty",
                "importance": "important",
            },
            "is_enterprise": {
                "type": "boolean",
                "weight": 15,
                "importance": "important",
            },
        },
        "structure_weight": 20,
        "total_points": 100,
    }
    spec_dir = agent_dir / "setup_spec"
    spec_dir.mkdir()
    spec_path = spec_dir / "eval_spec.json"
    spec_path.write_text(json.dumps(spec))

    dataset = dataset or [
        {
            "input": {"company": "Acme"},
            "expected_output": {
                "qualification": "hot",
                "score": 85,
                "reasoning": "big",
                "is_enterprise": True,
            },
        },
        {
            "input": {"company": "Tiny"},
            "expected_output": {
                "qualification": "cold",
                "score": 20,
                "reasoning": "small",
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
    )


class TestOptimizerInit:
    def test_init_creates_evaluator(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        assert opt.evaluator is not None
        assert opt.best_score == 0.0
        assert opt.results == []

    def test_init_loads_policy(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            spec={
                "output_fields": {
                    "x": {"type": "text", "weight": 80, "eval_mode": "non_empty"}
                },
                "structure_weight": 20,
                "total_points": 100,
                "policy": {"purpose": "test", "domain_rules": ["rule1"]},
            },
        )
        opt = Optimizer(cfg)
        assert opt._policy_data is not None
        assert "rule1" in opt._policy_diagnosis


class TestSplitDataset:
    def test_no_holdout(self):
        data = [{"i": i} for i in range(10)]
        train, holdout = Optimizer._split_dataset(data, 0.0)
        assert len(train) == 10
        assert holdout == []

    def test_small_dataset_no_split(self):
        data = [{"i": i} for i in range(3)]
        train, holdout = Optimizer._split_dataset(data, 0.2)
        assert len(train) == 3
        assert holdout == []

    def test_standard_split(self):
        data = [{"i": i} for i in range(20)]
        train, holdout = Optimizer._split_dataset(data, 0.2)
        assert len(holdout) == 4
        assert len(train) == 16

    def test_deterministic(self):
        data = [{"i": i} for i in range(20)]
        t1, h1 = Optimizer._split_dataset(data, 0.2)
        t2, h2 = Optimizer._split_dataset(data, 0.2)
        assert t1 == t2
        assert h1 == h2


class TestBuildCaseResults:
    def test_builds_correct_structure(self):
        eval_items = [
            {
                "expected": {"q": "hot"},
                "output": {"q": "warm"},
                "score": {"total": 50},
                "tool_calls": [],
                "tool_trace": [],
            },
        ]
        dataset = [{"input": {"name": "test"}}]
        results = Optimizer._build_case_results(eval_items, dataset)
        assert len(results) == 1
        assert results[0]["input"] == {"name": "test"}
        assert results[0]["score"]["total"] == 50


class TestValidateCode:
    def test_valid_python(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        code = "def run(input_data):\n    return {}\n"
        assert opt._validate_code(code) is True

    def test_syntax_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        code = "def run(input_data:\n    return {}"
        assert opt._validate_code(code) is False

    def test_missing_entrypoint(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        code = "def other(x):\n    return {}\n"
        assert opt._validate_code(code) is False

    def test_import_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        code = "import nonexistent_module_xyz\ndef run(x):\n    return {}\n"
        assert opt._validate_code(code) is False


class TestGetPromptSize:
    def test_with_prompt(self):
        code = 'SYSTEM_PROMPT = """Hello world"""\ndef run(x): pass'
        assert Optimizer._get_prompt_size(code) == len("Hello world")

    def test_no_prompt(self):
        assert Optimizer._get_prompt_size("def run(x): pass") == 0


class TestCountConditionalBranches:
    def test_counts_if_elif(self):
        code = "if x:\n  pass\nelif y:\n  pass\nif(z):\n  pass\n"
        assert Optimizer._count_conditional_branches(code) == 3

    def test_no_branches(self):
        assert Optimizer._count_conditional_branches("x = 1\ny = 2\n") == 0


class TestComputeComplexityPenalty:
    def test_no_penalty_for_small_code(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "def run(x):\n    return {}\n"
        opt.best_code = opt._baseline_code
        penalty = opt._compute_complexity_penalty("def run(x):\n    return {}\n")
        assert penalty == 0.0

    def test_penalty_for_prompt_bloat(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_code = 'SYSTEM_PROMPT = """short"""\ndef run(x): pass'
        opt._baseline_code = opt.best_code
        bloated = 'SYSTEM_PROMPT = """' + "x" * 500 + '"""\ndef run(x): pass'
        penalty = opt._compute_complexity_penalty(bloated)
        assert penalty > 0

    def test_penalty_for_code_growth(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "def run(x): pass\n"
        opt.best_code = opt._baseline_code
        huge = "\n".join(f"# line {i}" for i in range(100)) + "\ndef run(x): pass\n"
        penalty = opt._compute_complexity_penalty(huge)
        assert penalty > 0

    def test_penalty_for_many_branches(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "def run(x): pass\n"
        opt.best_code = opt._baseline_code
        branchy = (
            "\n".join(f"if cond_{i}:\n  pass" for i in range(25))
            + "\ndef run(x): pass\n"
        )
        penalty = opt._compute_complexity_penalty(branchy)
        assert penalty > 0


class TestDetectDataLeakage:
    def test_no_leakage(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "def run(x): pass"
        candidate = "def run(x):\n    return {'result': 'ok'}"
        train = [{"expected_output": {"result": "secret_value"}}]
        assert opt._detect_data_leakage(candidate, train) == 0

    def test_detects_leakage(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "def run(x): pass"
        candidate = "def run(x):\n    if True: return 'secret_value_here'"
        train = [{"expected_output": {"result": "secret_value_here"}}]
        assert opt._detect_data_leakage(candidate, train) >= 1

    def test_ignores_short_values(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._baseline_code = "x=1"
        candidate = "x=1\nif True: y='ab'"
        train = [{"expected_output": {"x": "ab"}}]
        assert opt._detect_data_leakage(candidate, train) == 0


class TestCheckAcceptance:
    def test_no_improvement_rejected(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_score = 60.0
        opt.best_case_scores = [60.0, 60.0]
        accept, reason = opt._check_acceptance(55.0, [55.0, 55.0], [], [])
        assert accept is False

    def test_improvement_accepted(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_score = 60.0
        opt.best_case_scores = [60.0, 60.0]
        accept, reason = opt._check_acceptance(70.0, [70.0, 70.0], [], [])
        assert accept is True

    def test_too_many_regressions_rejected(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_score = 50.0
        opt.best_case_scores = [80.0, 80.0, 80.0, 80.0, 80.0]
        accept, reason = opt._check_acceptance(
            55.0, [20.0, 20.0, 20.0, 90.0, 90.0], [], []
        )
        assert accept is False
        assert "regress" in reason.lower()

    def test_regressions_accepted_when_improvements_outweigh(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_score = 50.0
        opt.best_case_scores = [50.0, 50.0, 50.0]
        accept, _ = opt._check_acceptance(60.0, [10.0, 90.0, 90.0], [], [])
        assert accept is True

    def test_empty_scores(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt.best_score = 50.0
        opt.best_case_scores = []
        accept, _ = opt._check_acceptance(60.0, [], [], [])
        assert accept is True


class TestComputeDimensionDeltas:
    def test_computes_deltas(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        old = {"avg_structure": 15.0, "avg_qualification": 20.0}
        new = {"avg_structure": 18.0, "avg_qualification": 20.0}
        deltas = opt._compute_dimension_deltas(old, new)
        assert "avg_structure" in deltas
        assert deltas["avg_structure"] == 3.0


class TestSetupOutputDirs:
    def test_creates_dirs(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        assert opt.output_dir.exists()
        assert opt.traces_dir.exists()
        assert opt.analysis_dir.exists()
        assert (opt.output_dir / "results.tsv").exists()


class TestLogResult:
    def test_appends_to_results(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        opt._log_result("baseline", {"avg_total": 50.0}, "keep", "test")
        assert len(opt.results) == 1
        content = (opt.output_dir / "results.tsv").read_text()
        assert "baseline" in content


class TestPrintEval:
    def test_without_prev(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._print_eval({"avg_total": 75.0}, "Test")

    def test_with_prev(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._print_eval(
            {"avg_total": 75.0},
            "Test",
            prev_evaluation={"avg_total": 60.0},
        )


class TestPrintBaselineDiagnostics:
    def test_runs_without_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        evaluation = {
            "avg_structure": 20.0,
            "avg_qualification": 30.0,
            "avg_score": 20.0,
        }
        items = [
            {"tool_trace": [{"name": "search", "args": {}}]},
            {"tool_trace": []},
        ]
        opt._print_baseline_diagnostics(evaluation, items)


class TestLoadAgentModule:
    def test_loads_module(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {'ok': True}\n")
        mod = Optimizer._load_agent_module(str(agent))
        assert mod.run({"x": 1}) == {"ok": True}


class TestAnimateCodeUpdate:
    def test_no_changes(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._animate_code_update("same", "same")

    @patch("overclaw.optimize.optimizer.time.sleep")
    def test_with_changes(self, mock_sleep, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._animate_code_update("line1\nline2\n", "line1\nline3\n")


class TestRunSingleCase:
    def test_runs_agent(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        agent = Optimizer._load_agent_module(cfg.agent_path)
        case = {
            "input": {"company": "Test"},
            "expected_output": {
                "qualification": "hot",
                "score": 80,
                "reasoning": "ok",
                "is_enterprise": True,
            },
        }
        tracer, item = opt._run_single_case(agent, case, "test", 0)
        assert "output" in item
        assert "score" in item
        assert item["score"]["total"] >= 0

    def test_handles_agent_exception(self, tmp_path):
        agent_dir = tmp_path / "agents" / "bad"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "agent.py"
        agent_file.write_text("def run(x):\n    raise ValueError('boom')\n")
        cfg = _make_config(tmp_path)
        cfg.agent_path = str(agent_file)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        agent = Optimizer._load_agent_module(str(agent_file))
        case = {"input": {}, "expected_output": {}}
        tracer, item = opt._run_single_case(agent, case, "test", 0)
        assert "error" in item["output"]


class TestGenerateReport:
    def test_generates_report(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        opt.best_code = "def run(x): pass"
        opt.best_score = 75.0
        opt.results = [
            {
                "iteration": "baseline",
                "avg_score": "50.0",
                "status": "keep",
                "description": "Initial",
            }
        ]
        opt._generate_report()
        report = opt.output_dir / "report.md"
        assert report.exists()
        content = report.read_text()
        assert "OverClaw" in content

    def test_report_with_backtest(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        opt.best_code = "def run(x): pass"
        opt.best_score = 75.0
        opt.results = [
            {
                "iteration": "baseline",
                "avg_score": "50.0",
                "status": "keep",
                "description": "Initial",
            }
        ]
        opt.backtest_results = {"openai/gpt-5.4": {"avg_total": 70.0}}
        opt._generate_report()
        content = (opt.output_dir / "report.md").read_text()
        assert "Backtesting" in content


class TestRunBacktesting:
    @patch("overclaw.optimize.optimizer.Optimizer._run_agent_on_dataset")
    def test_backtesting(self, mock_run, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.model_backtesting = True
        cfg.backtest_models = ["openai/gpt-5.4-mini"]
        opt = Optimizer(cfg)
        opt._setup_output_dirs()
        opt.best_code = 'MODEL = "gpt-5.4"\ndef run(x): pass\n'
        mock_run.return_value = ({"avg_total": 65.0}, [], [])
        opt._run_backtesting([{"input": {}, "expected_output": {}}])
        assert "openai/gpt-5.4-mini" in opt.backtest_results
