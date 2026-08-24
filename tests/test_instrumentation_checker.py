import json
import subprocess

from typer.testing import CliRunner

from overmind.__main__ import app
from overmind.instrumentation_checker import check_plan


def _plan(**placement):
    return {"placements": [{"file": "agent.py", "qualname": "Agent.run", **placement}]}


def test_check_fixed_sync_async_and_existing_decorators(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n"
        "def other(fn): return fn\n\n"
        "class Agent:\n"
        "    @classmethod\n"
        "    @overmind.task('sync', capture_io=False)\n"
        "    @other\n"
        "    def run(cls): return cls\n\n"
        "    @staticmethod\n"
        "    @overmind.task('async')\n"
        "    async def other_run(): return None\n"
    )
    result = check_plan(_plan(key="sync"), tmp_path)
    assert result["ok"]
    assert not result["errors"]


def test_check_reports_duplicates_missing_nested_and_stale_targets(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n\n"
        "@overmind.task('old')\n"
        "def run():\n"
        "    @overmind.task('nested')\n"
        "    def child(): pass\n"
        "\n"
        "class Agent:\n"
        "    @overmind.task('a')\n"
        "    @overmind.task('b')\n"
        "    def method(self): pass\n"
    )
    result = check_plan(
        {
            "placements": [
                {"file": "agent.py", "qualname": "Agent.method", "key": "a"},
                {"file": "agent.py", "qualname": "Agent.missing", "key": "x"},
            ]
        },
        tmp_path,
    )
    codes = {error["code"] for error in result["errors"]}
    assert {"task.duplicate", "target.missing", "task.nested"} <= codes


def test_check_dynamic_key_from_and_allowed_keys(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.task_key)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    result = check_plan(
        {
            "placements": [
                {
                    "file": "agent.py",
                    "qualname": "dispatch",
                    "mode": "dynamic",
                    "allowed_keys": ["a", "b"],
                }
            ]
        },
        tmp_path,
    )
    assert result["ok"]


def test_cli_json_is_machine_readable_and_returns_failure(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(key="missing")))
    result = CliRunner().invoke(
        app, ["instrumentation", "check", "--plan-file", str(plan), "--root", str(tmp_path), "--format", "json"]
    )
    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["ok"] is False


def test_revision_mismatch_is_reported_when_git_revision_is_available(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text("import overmind\n@overmind.task('a')\ndef run(): pass\n")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="actual\n", stderr="")
    )
    result = check_plan(
        {"revision": "expected", "placements": [{"file": "agent.py", "qualname": "run", "key": "a"}]}, tmp_path
    )
    assert not result["ok"]
    assert any(error["code"] == "revision.mismatch" for error in result["errors"])
