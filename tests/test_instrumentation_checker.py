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


def test_check_plan_normalises_server_shaped_placement(tmp_path, monkeypatch):
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.task_key)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="deadbeef\n", stderr=""),
    )
    plan = {
        "analyzed_sha": "deadbeef",
        "placements": [
            {
                "target": {
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "module": "dispatch",
                    "import_line": "import overmind",
                },
                "placement_mode": "dynamic_key",
                "required_task_decorator": "@overmind.task(key_from=lambda request: request.task_key)",
                "allowed_keys": ["a", "b"],
                "analyzed_sha": "deadbeef",
            }
        ],
    }
    result = check_plan(plan, tmp_path)
    assert result["ok"], result["errors"]
    assert result["revision"]["status"] == "pass"


def _whole_repo_placement(*, capability, capability_id, plan_id, analyzed_sha, **extra):
    return {
        "placement_id": f"{capability_id}-placement",
        "analyzed_sha": analyzed_sha,
        "capability": capability,
        "capability_id": capability_id,
        "plan_id": plan_id,
        "why": f"entry point for {capability}",
        "tier": 1,
        "smoke_hint": f"call the {capability} entry with synthetic args",
        **extra,
    }


def test_check_plan_accepts_whole_repo_payload_across_capabilities(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text(
        "import overmind\n\n"
        "class Agent:\n"
        "    @overmind.task('sync-key')\n"
        "    def run(self):\n"
        "        return self\n"
    )
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.task_key)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="deadbeef\n", stderr=""),
    )
    plan = {
        "placements": [
            _whole_repo_placement(
                capability="support-agent",
                capability_id="11111111-1111-1111-1111-111111111111",
                plan_id="22222222-2222-2222-2222-222222222222",
                analyzed_sha="deadbeef",
                key="sync-key",
                placement_mode="fixed",
                target={"file": "agent.py", "qualname": "Agent.run", "module": "agent", "import_line": "import overmind"},
                required_task_decorator="@overmind.task('sync-key')",
                constraints={},
            ),
            _whole_repo_placement(
                capability="billing-agent",
                capability_id="33333333-3333-3333-3333-333333333333",
                plan_id="44444444-4444-4444-4444-444444444444",
                analyzed_sha="deadbeef",
                placement_mode="dynamic_key",
                allowed_keys=["a", "b"],
                target={
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "module": "dispatch",
                    "import_line": "import overmind",
                },
                required_task_decorator="@overmind.task(key_from=lambda request: request.task_key)",
                constraints={},
            ),
        ],
        "plans": [
            {
                "capability": "support-agent",
                "capability_id": "11111111-1111-1111-1111-111111111111",
                "plan_id": "22222222-2222-2222-2222-222222222222",
                "placement_count": 1,
            },
            {
                "capability": "billing-agent",
                "capability_id": "33333333-3333-3333-3333-333333333333",
                "plan_id": "44444444-4444-4444-4444-444444444444",
                "placement_count": 1,
            },
        ],
        "ambiguous": [],
        "dropped": [],
        "minted": {"behaviours": 2, "versions": 2},
    }
    result = check_plan(plan, tmp_path)
    assert result["ok"], result["errors"]
    assert result["revision"]["status"] == "pass"


def test_check_plan_reports_per_placement_analyzed_sha_mismatch(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text(
        "import overmind\n@overmind.task('sync-key')\ndef run(): pass\n"
    )
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.task_key)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="deadbeef\n", stderr=""),
    )
    plan = {
        "placements": [
            _whole_repo_placement(
                capability="support-agent",
                capability_id="11111111-1111-1111-1111-111111111111",
                plan_id="22222222-2222-2222-2222-222222222222",
                analyzed_sha="deadbeef",
                key="sync-key",
                placement_mode="fixed",
                target={"file": "agent.py", "qualname": "run", "module": "agent", "import_line": "import overmind"},
                required_task_decorator="@overmind.task('sync-key')",
                constraints={},
            ),
            _whole_repo_placement(
                capability="billing-agent",
                capability_id="33333333-3333-3333-3333-333333333333",
                plan_id="44444444-4444-4444-4444-444444444444",
                analyzed_sha="stale-sha",
                placement_mode="dynamic_key",
                allowed_keys=["a", "b"],
                target={
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "module": "dispatch",
                    "import_line": "import overmind",
                },
                required_task_decorator="@overmind.task(key_from=lambda request: request.task_key)",
                constraints={},
            ),
        ],
    }
    result = check_plan(plan, tmp_path)
    assert not result["ok"]
    assert result["revision"]["status"] == "fail"
    mismatches = [error for error in result["errors"] if error["code"] == "revision.mismatch"]
    assert mismatches
    assert all(error.get("file") == "dispatch.py" and error.get("qualname") == "dispatch" for error in mismatches)
