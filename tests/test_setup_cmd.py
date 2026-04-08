"""Tests for overclaw.commands.setup_cmd — setup command helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.paths import agent_setup_spec_dir
from overclaw.commands.setup_cmd import (
    _build_eval_spec_stub,
    _clear_existing_eval_spec,
    _data_dir,
    _display_proposed_criteria,
    _prompt_seed_data_flag_early,
    _resolve_datagen_model,
    _resolve_seed_json_files,
    _run_beginning_smoke_test,
    _run_end_smoke_test,
    _save_and_finish,
    _save_dataset,
    _smoke_test_agent,
    _validate_agent_entrypoint,
)


class TestValidateAgentEntrypoint:
    def test_valid(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x):\n    pass\n")
        console = MagicMock()
        _validate_agent_entrypoint(str(agent), "run", console)

    def test_missing_function(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def other(x):\n    pass\n")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _validate_agent_entrypoint(str(agent), "run", console)


class TestPathHelpers:
    def test_agent_setup_spec_dir(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1")
        assert str(result).endswith("setup_spec")
        assert OVERCLAW_DIR_NAME in str(result)

    def test_eval_spec_under_setup_spec(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1") / "eval_spec.json"
        assert str(result).endswith("eval_spec.json")

    def test_dataset_under_setup_spec(self, overclaw_tmp_project: Path):
        result = agent_setup_spec_dir("a1") / "dataset.json"
        assert str(result).endswith("dataset.json")

    def test_data_dir(self):
        result = _data_dir("/project/agents/a1/agent.py")
        assert result.name == "data"


class TestClearExistingEvalSpec:
    def test_no_dir(self, overclaw_tmp_project: Path):
        console = MagicMock()
        _clear_existing_eval_spec("nope", console)

    def test_empty_dir(self, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        console = MagicMock()
        _clear_existing_eval_spec("x", console)

    def test_fast_clears(self, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console, fast=True)
        assert not (spec_dir / "spec.json").exists()

    @patch("overclaw.commands.setup_cmd.confirm_option", return_value=True)
    def test_interactive_confirm(self, _mock_confirm, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console)

    @patch("overclaw.commands.setup_cmd.confirm_option", return_value=False)
    def test_interactive_decline(self, _mock_confirm, overclaw_tmp_project: Path):
        spec_dir = agent_setup_spec_dir("x")
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.json").write_text("{}")
        console = MagicMock()
        _clear_existing_eval_spec("x", console)
        assert (spec_dir / "spec.json").exists()


class TestBuildEvalSpecStub:
    def test_basic(self):
        analysis = {
            "output_schema": {
                "status": {"type": "enum", "values": ["a", "b"]},
                "score": {"type": "number", "range": [0, 100]},
            },
            "input_schema": {"name": {"type": "string"}},
            "description": "Test",
        }
        stub = _build_eval_spec_stub(analysis)
        assert "status" in stub["output_fields"]
        assert stub["output_fields"]["status"]["weight"] == 10

    def test_with_policy(self):
        analysis = {"output_schema": {}}
        policy = {"purpose": "test"}
        stub = _build_eval_spec_stub(analysis, policy)
        assert stub["policy"]["purpose"] == "test"


class TestSaveAndFinish:
    def test_saves_spec(self, overclaw_tmp_project: Path):
        spec = {"output_fields": {"x": {"weight": 10}}, "total_points": 100}
        console = MagicMock()
        _save_and_finish(spec, "myagent", console)
        spec_path = agent_setup_spec_dir("myagent") / "eval_spec.json"
        assert spec_path.exists()

    def test_saves_policy(self, overclaw_tmp_project: Path):
        spec = {"output_fields": {}, "policy": {"domain_rules": ["r1"]}}
        console = MagicMock()
        _save_and_finish(spec, "myagent", console, policy_md="# Policy")
        from overclaw.utils.policy import default_policy_path

        assert Path(default_policy_path("myagent")).exists()


class TestSaveDataset:
    def test_saves(self, overclaw_tmp_project: Path):
        console = MagicMock()
        cases = [{"input": {"x": 1}, "expected_output": {"y": 2}}]
        path = _save_dataset(cases, "dsagent", console)
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text())
        assert len(loaded) == 1


class TestResolveDategenModel:
    def test_fast_with_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        console = MagicMock()
        result = _resolve_datagen_model(console, fast=True)
        assert "gpt-5.4" in result

    def test_fast_without_env(self, monkeypatch):
        monkeypatch.delenv("SYNTHETIC_DATAGEN_MODEL", raising=False)
        console = MagicMock()
        with pytest.raises(SystemExit):
            _resolve_datagen_model(console, fast=True)

    @patch("overclaw.commands.setup_cmd.confirm_option", return_value=True)
    def test_interactive_with_env(self, _mock_confirm, monkeypatch):
        monkeypatch.setenv("SYNTHETIC_DATAGEN_MODEL", "gpt-5.4")
        console = MagicMock()
        result = _resolve_datagen_model(console, fast=False)
        assert "gpt-5.4" in result


class TestDisplayProposedCriteria:
    def test_with_criteria(self):
        analysis = {
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {
                    "status": {"importance": "critical", "partial_credit": True},
                    "score": {"importance": "important", "tolerance": 10},
                    "reason": {"importance": "minor", "eval_mode": "non_empty"},
                },
            },
            "output_schema": {
                "status": {"type": "enum"},
                "score": {"type": "number"},
                "reason": {"type": "text"},
            },
        }
        console = MagicMock()
        _display_proposed_criteria(analysis, console)

    def test_no_criteria(self):
        console = MagicMock()
        _display_proposed_criteria({}, console)


# ---------------------------------------------------------------------------
# Smoke-test helpers
# ---------------------------------------------------------------------------


class TestSmokeTestAgent:
    def test_success(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x):\n    return {'ok': True}\n")
        ok, err = _smoke_test_agent(str(agent), "run", {"val": 1})
        assert ok is True
        assert err is None

    def test_agent_raises_returns_false(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x):\n    raise ValueError('boom')\n")
        ok, err = _smoke_test_agent(str(agent), "run", {})
        assert ok is False
        assert "boom" in err

    def test_import_error_returns_false(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("this is not valid python !!!\n")
        ok, err = _smoke_test_agent(str(agent), "run", {})
        assert ok is False
        assert err is not None

    def test_missing_function_returns_false(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text("def other(x):\n    return {}\n")
        ok, err = _smoke_test_agent(str(agent), "run", {})
        assert ok is False
        assert err is not None

    def test_input_passed_through(self, tmp_path):
        agent = tmp_path / "agent.py"
        agent.write_text(
            "def run(x):\n"
            "    if x.get('key') != 'value':\n"
            "        raise AssertionError('wrong input')\n"
            "    return {}\n"
        )
        ok, err = _smoke_test_agent(str(agent), "run", {"key": "value"})
        assert ok is True

        ok2, err2 = _smoke_test_agent(str(agent), "run", {"key": "wrong"})
        assert ok2 is False
        assert "wrong input" in err2


class TestPromptSeedDataFlagEarly:
    @patch("overclaw.commands.setup_cmd.confirm_option", return_value=True)
    def test_yes_prints_command_and_exits_zero(self, _mock_confirm):
        console = MagicMock()
        with pytest.raises(SystemExit) as exc:
            _prompt_seed_data_flag_early("my-agent", console=console)
        assert exc.value.code == 0

    @patch("overclaw.commands.setup_cmd.confirm_option", return_value=False)
    def test_no_does_not_exit(self, _mock_confirm):
        console = MagicMock()
        _prompt_seed_data_flag_early("my-agent", console=console)


class TestResolveSeedJsonFiles:
    def test_none_returns_empty(self):
        console = MagicMock()
        assert _resolve_seed_json_files(None, console=console) == []
        assert _resolve_seed_json_files("", console=console) == []
        assert _resolve_seed_json_files("   ", console=console) == []

    def test_single_json_file(self, tmp_path):
        f = tmp_path / "a.json"
        f.write_text("[]")
        console = MagicMock()
        assert _resolve_seed_json_files(str(f), console=console) == [f.resolve()]

    def test_directory_glob(self, tmp_path):
        (tmp_path / "z.json").write_text("[]")
        (tmp_path / "a.json").write_text("[]")
        console = MagicMock()
        got = _resolve_seed_json_files(str(tmp_path), console=console)
        assert [p.name for p in got] == ["a.json", "z.json"]

    def test_missing_path_exits(self, tmp_path):
        console = MagicMock()
        with pytest.raises(SystemExit):
            _resolve_seed_json_files(str(tmp_path / "nope.json"), console=console)

    def test_non_json_file_exits(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _resolve_seed_json_files(str(f), console=console)


class TestRunBeginningSmokTest:
    def test_no_data_flag_skips(self, tmp_path):
        """Without --data the pre-setup smoke test does not load seed JSON."""
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {}\n")
        console = MagicMock()
        _run_beginning_smoke_test(str(agent), "run", console)
        first_call_args = console.print.call_args_list[0][0][0]
        assert "Skipping" in first_call_args
        assert "--data" in first_call_args

    def test_empty_seed_file_skips(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cases.json").write_text("[]")
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {}\n")
        console = MagicMock()
        _run_beginning_smoke_test(str(agent), "run", console, data_path=str(data_dir))
        # Should mention the file name in the skip message
        all_output = " ".join(str(c) for c in console.print.call_args_list)
        assert "cases.json" in all_output

    def test_success_with_seed(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cases.json").write_text(
            json.dumps([{"input": {"x": 1}, "expected_output": {"y": 2}}])
        )
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {'y': x.get('x')}\n")
        console = MagicMock()
        _run_beginning_smoke_test(str(agent), "run", console, data_path=str(data_dir))
        # Should not raise

    def test_failure_exits(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cases.json").write_text(
            json.dumps([{"input": {"x": 1}, "expected_output": {}}])
        )
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): raise RuntimeError('agent broken')\n")
        console = MagicMock()
        with pytest.raises(SystemExit) as exc_info:
            _run_beginning_smoke_test(
                str(agent), "run", console, data_path=str(data_dir)
            )
        assert exc_info.value.code == 1

    def test_unreadable_seed_file_skips(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cases.json").write_text("not valid json {{{")
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {}\n")
        console = MagicMock()
        # Should not raise — bad JSON is treated as unreadable, silently skipped
        _run_beginning_smoke_test(str(agent), "run", console, data_path=str(data_dir))

    def test_seed_case_without_input_key(self, tmp_path):
        """Cases stored as flat dicts (no 'input' wrapper) should still work."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cases.json").write_text(
            json.dumps([{"company": "Acme", "budget": 1000}])
        )
        agent = tmp_path / "agent.py"
        agent.write_text("def run(x): return {'ok': True}\n")
        console = MagicMock()
        _run_beginning_smoke_test(str(agent), "run", console, data_path=str(data_dir))


class TestRunEndSmokeTest:
    def test_no_dataset_skips(self, overclaw_tmp_project):
        agent = overclaw_tmp_project / "agent.py"
        agent.write_text("def run(x): return {}\n")
        console = MagicMock()
        # dataset.json doesn't exist — should return silently
        _run_end_smoke_test("myagent", str(agent), "run", console)
        console.print.assert_not_called()

    def test_success(self, overclaw_tmp_project):
        from overclaw.core.paths import agent_setup_spec_dir

        spec_dir = agent_setup_spec_dir("myagent")
        spec_dir.mkdir(parents=True)
        dataset_path = spec_dir / "dataset.json"
        dataset_path.write_text(
            json.dumps([{"input": {"x": 1}, "expected_output": {"y": 2}}])
        )
        agent = overclaw_tmp_project / "agent.py"
        agent.write_text("def run(x): return {'y': x.get('x')}\n")
        console = MagicMock()
        _run_end_smoke_test("myagent", str(agent), "run", console)
        # Should print success, not raise
        console.print.assert_called()

    def test_failure_warns_does_not_exit(self, overclaw_tmp_project):
        from overclaw.core.paths import agent_setup_spec_dir

        spec_dir = agent_setup_spec_dir("myagent")
        spec_dir.mkdir(parents=True)
        dataset_path = spec_dir / "dataset.json"
        dataset_path.write_text(
            json.dumps([{"input": {"x": 1}, "expected_output": {}}])
        )
        agent = overclaw_tmp_project / "agent.py"
        agent.write_text("def run(x): raise ValueError('oops')\n")
        console = MagicMock()
        # Must NOT raise SystemExit — just warn
        _run_end_smoke_test("myagent", str(agent), "run", console)
        console.print.assert_called()

    def test_empty_dataset_skips(self, overclaw_tmp_project):
        from overclaw.core.paths import agent_setup_spec_dir

        spec_dir = agent_setup_spec_dir("myagent")
        spec_dir.mkdir(parents=True)
        (spec_dir / "dataset.json").write_text("[]")
        agent = overclaw_tmp_project / "agent.py"
        agent.write_text("def run(x): return {}\n")
        console = MagicMock()
        _run_end_smoke_test("myagent", str(agent), "run", console)
        console.print.assert_not_called()
