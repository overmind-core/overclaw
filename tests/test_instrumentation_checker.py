"""Tests for the static plan checker and the ``overmind instrumentation`` CLI."""

from __future__ import annotations

import json
import subprocess

from typer.testing import CliRunner

from overmind.__main__ import app
from overmind.instrumentation_checker import check_plan

DISPATCH_SOURCE = (
    "import overmind\n\ndef dispatch(request):\n    with overmind.task(request.task_key):\n        return request\n"
)


def _plan(**placement):
    return {"placements": [{"file": "agent.py", "qualname": "Agent.run", **placement}]}


def _fixed_git_revision(monkeypatch, sha):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n", stderr=""),
    )


def test_check_fixed_sync_async_and_existing_decorators(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n"
        "def other(fn): return fn\n\n"
        "class Agent:\n"
        "    @classmethod\n"
        "    @overmind.task('sync', unit='turn')\n"
        "    @other\n"
        "    def run(cls): return cls\n\n"
        "    @staticmethod\n"
        "    @overmind.task('async')\n"
        "    async def other_run(): return None\n"
    )
    result = check_plan(_plan(key="sync"), tmp_path)
    assert result["ok"], result["errors"]


def test_check_fixed_accepts_a_context_manager_boundary(tmp_path):
    """``task()`` is a context manager too, and that form carries ``unit=``."""
    (tmp_path / "agent.py").write_text(
        "import overmind\n\n"
        "def answer(question):\n"
        "    with overmind.task('answer-question', unit='turn'):\n"
        "        return question\n"
    )
    result = check_plan(
        {"placements": [{"file": "agent.py", "qualname": "answer", "key": "answer-question"}]}, tmp_path
    )
    assert result["ok"], result["errors"]


def test_check_fixed_rejects_two_boundaries_in_one_target(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n\n"
        "@overmind.task('a')\n"
        "def answer(question):\n"
        "    with overmind.task('a'):\n"
        "        return question\n"
    )
    result = check_plan({"placements": [{"file": "agent.py", "qualname": "answer", "key": "a"}]}, tmp_path)
    assert not result["ok"]
    assert any(error["code"] == "task.duplicate" for error in result["errors"])


def test_check_fixed_reports_a_key_mismatch(tmp_path):
    (tmp_path / "agent.py").write_text("import overmind\n\n@overmind.task('other')\ndef answer(): pass\n")
    result = check_plan({"placements": [{"file": "agent.py", "qualname": "answer", "key": "a"}]}, tmp_path)
    assert any(error["code"] == "task.key_mismatch" for error in result["errors"])


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


def test_check_requires_import_overmind(tmp_path):
    (tmp_path / "agent.py").write_text("from overmind import task\n\n@task('a')\ndef run(): pass\n")
    result = check_plan({"placements": [{"file": "agent.py", "qualname": "run", "key": "a"}]}, tmp_path)
    assert not result["ok"]
    assert any(error["code"] == "import.missing" for error in result["errors"])


def test_check_dynamic_accepts_a_computed_key_context_manager(tmp_path):
    """``task()`` takes a literal key, so dynamic dispatch computes it and
    enters the scope as a context manager."""
    (tmp_path / "dispatch.py").write_text(DISPATCH_SOURCE)
    result = check_plan(
        {
            "placements": [
                {
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "placement_mode": "dynamic_key",
                    "allowed_keys": ["a", "b"],
                }
            ]
        },
        tmp_path,
    )
    assert result["ok"], result["errors"]


def test_check_dynamic_rejects_a_literal_key(tmp_path):
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\ndef dispatch(request):\n    with overmind.task('fixed'):\n        return request\n"
    )
    result = check_plan(
        {
            "placements": [
                {
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "placement_mode": "dynamic_key",
                    "allowed_keys": ["a"],
                }
            ]
        },
        tmp_path,
    )
    assert not result["ok"]
    assert any(error["code"] == "dynamic.shape" for error in result["errors"])


def test_check_dynamic_requires_non_empty_unique_allowed_keys(tmp_path):
    (tmp_path / "dispatch.py").write_text(DISPATCH_SOURCE)
    result = check_plan(
        {
            "placements": [
                {"file": "dispatch.py", "qualname": "dispatch", "placement_mode": "dynamic_key", "allowed_keys": []}
            ]
        },
        tmp_path,
    )
    assert any(error["code"] == "dynamic.allowed_keys" for error in result["errors"])


def test_check_dynamic_matches_a_key_from_decorator(tmp_path):
    """``key_from=`` placements still validate statically, for plans minted
    against an SDK build that offers the selector form."""
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.task_key)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    result = check_plan(
        {
            "placements": [
                {
                    "target": {"file": "dispatch.py", "qualname": "dispatch"},
                    "placement_mode": "dynamic_key",
                    "required_task_decorator": "@overmind.task(key_from=lambda request: request.task_key)",
                    "allowed_keys": ["a", "b"],
                }
            ]
        },
        tmp_path,
    )
    assert result["ok"], result["errors"]


def test_check_dynamic_reports_a_key_from_mismatch(tmp_path):
    (tmp_path / "dispatch.py").write_text(
        "import overmind\n\n"
        "@overmind.task(key_from=lambda request: request.route)\n"
        "def dispatch(request):\n"
        "    return request\n"
    )
    result = check_plan(
        {
            "placements": [
                {
                    "target": {"file": "dispatch.py", "qualname": "dispatch"},
                    "placement_mode": "dynamic_key",
                    "required_task_decorator": "@overmind.task(key_from=lambda request: request.task_key)",
                    "allowed_keys": ["a", "b"],
                }
            ]
        },
        tmp_path,
    )
    assert not result["ok"]
    assert any(error["code"] == "dynamic.key_from_mismatch" for error in result["errors"])


def test_revision_mismatch_is_reported_when_git_revision_is_available(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text("import overmind\n@overmind.task('a')\ndef run(): pass\n")
    _fixed_git_revision(monkeypatch, "actual")
    result = check_plan(
        {"revision": "expected", "placements": [{"file": "agent.py", "qualname": "run", "key": "a"}]}, tmp_path
    )
    assert not result["ok"]
    assert any(error["code"] == "revision.mismatch" for error in result["errors"])


def _whole_repo_placement(*, capability, capability_id, plan_id, analyzed_sha, **extra):
    return {
        "placement_id": f"{capability_id}-placement",
        "analyzed_sha": analyzed_sha,
        "capability": capability,
        "capability_id": capability_id,
        "plan_id": plan_id,
        "required_identity": {"capability_id": capability_id, "capability_name": capability, "how": "capability"},
        "why": f"entry point for {capability}",
        "tier": 1,
        "smoke_hint": f"call the {capability} entry with synthetic args",
        **extra,
    }


def _whole_repo_plan(*, dispatch_sha="deadbeef"):
    return {
        "placements": [
            _whole_repo_placement(
                capability="support-agent",
                capability_id="11111111-1111-1111-1111-111111111111",
                plan_id="22222222-2222-2222-2222-222222222222",
                analyzed_sha="deadbeef",
                key="sync-key",
                placement_mode="fixed",
                allowed_keys=None,
                target={
                    "file": "agent.py",
                    "qualname": "Agent.run",
                    "module": "agent",
                    "import_line": "import overmind",
                },
                required_task_decorator="@overmind.task('sync-key')",
            ),
            _whole_repo_placement(
                capability="billing-agent",
                capability_id="33333333-3333-3333-3333-333333333333",
                plan_id="44444444-4444-4444-4444-444444444444",
                analyzed_sha=dispatch_sha,
                placement_mode="dynamic_key",
                allowed_keys=["a", "b"],
                target={
                    "file": "dispatch.py",
                    "qualname": "dispatch",
                    "module": "dispatch",
                    "import_line": "import overmind",
                },
                required_task_decorator="with overmind.task(request.task_key):",
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


def _write_whole_repo_sources(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import overmind\n\nclass Agent:\n    @overmind.task('sync-key')\n    def run(self):\n        return self\n"
    )
    (tmp_path / "dispatch.py").write_text(DISPATCH_SOURCE)


def test_check_plan_accepts_whole_repo_payload_across_capabilities(tmp_path, monkeypatch):
    _write_whole_repo_sources(tmp_path)
    _fixed_git_revision(monkeypatch, "deadbeef")

    result = check_plan(_whole_repo_plan(), tmp_path)

    assert result["ok"], result["errors"]
    assert result["revision"]["status"] == "pass"


def test_check_plan_reports_per_placement_analyzed_sha_mismatch(tmp_path, monkeypatch):
    _write_whole_repo_sources(tmp_path)
    _fixed_git_revision(monkeypatch, "deadbeef")

    result = check_plan(_whole_repo_plan(dispatch_sha="stale-sha"), tmp_path)

    assert not result["ok"]
    assert result["revision"]["status"] == "fail"
    mismatches = [error for error in result["errors"] if error["code"] == "revision.mismatch"]
    assert mismatches
    assert all(error["file"] == "dispatch.py" and error["qualname"] == "dispatch" for error in mismatches)


def test_cli_check_json_is_machine_readable_and_returns_failure(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(key="missing")))

    result = CliRunner().invoke(
        app, ["instrumentation", "check", "--plan-file", str(plan), "--root", str(tmp_path), "--format", "json"]
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_cli_check_rejects_an_unknown_format(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(key="missing")))
    result = CliRunner().invoke(app, ["instrumentation", "check", "--plan-file", str(plan), "--format", "yaml"])
    assert result.exit_code == 2


def test_cli_scan_writes_the_scan_payload(tmp_path):
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    out = tmp_path / "candidates.json"

    result = CliRunner().invoke(app, ["instrumentation", "scan", "--root", str(tmp_path), "--out", str(out)])

    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["files"][0]["path"] == "app.py"


def test_cli_smoke_runs_scripts_and_echoes_hint_only_placements(tmp_path):
    script = tmp_path / "smoke_entry.py"
    script.write_text(
        "import json, os, pathlib\n"
        "pathlib.Path(os.environ['OVERMIND_TRACE_FILE']).write_text(\n"
        "    json.dumps({'smoke': os.environ['OVERMIND_SMOKE']}) + '\\n'\n"
        ")\n"
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({
            "placements": [
                {"smoke_script": "smoke_entry.py"},
                {"target": {"file": "other.py", "qualname": "run"}, "smoke_hint": "call run() with a fake ticket"},
            ]
        })
    )
    spans = tmp_path / "spans.jsonl"

    result = CliRunner().invoke(
        app,
        ["instrumentation", "smoke", "--plan-file", str(plan), "--out", str(spans), "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert json.loads(spans.read_text()) == {"smoke": "1"}
    assert "TODO other.py run: call run() with a fake ticket" in result.stdout


def test_cli_smoke_fails_when_a_script_fails(tmp_path):
    (tmp_path / "boom.py").write_text("raise SystemExit(3)\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"placements": [{"smoke_script": "boom.py"}]}))

    result = CliRunner().invoke(
        app,
        [
            "instrumentation",
            "smoke",
            "--plan-file",
            str(plan),
            "--out",
            str(tmp_path / "spans.jsonl"),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1


def test_cli_verify_gates_on_bound_units(tmp_path, monkeypatch):
    from overmind import __main__ as cli

    spans = tmp_path / "spans.jsonl"
    spans.write_text('{"span_id": "a"}\n\n{"span_id": "b"}\n')
    sent: dict = {}

    def fake_call(api_url, api_key, tool, arguments, timeout):
        sent.update(tool=tool, arguments=arguments)
        return {"tasks": [{"behaviour_key": "k", "binding_source": "declared"}], "capabilities": [], "errors": []}

    monkeypatch.setattr(cli, "_mcp_call", fake_call)
    result = CliRunner().invoke(app, ["instrumentation", "verify", "--spans-file", str(spans)])

    assert result.exit_code == 0
    assert sent["tool"] == "verify_instrumentation_spans"
    assert sent["arguments"]["spans"] == [{"span_id": "a"}, {"span_id": "b"}]

    monkeypatch.setattr(
        cli,
        "_mcp_call",
        lambda *args, **kwargs: {"tasks": [{"binding_source": "anchor_join"}], "errors": []},
    )
    assert CliRunner().invoke(app, ["instrumentation", "verify", "--spans-file", str(spans)]).exit_code == 0

    monkeypatch.setattr(
        cli,
        "_mcp_call",
        lambda *args, **kwargs: {"tasks": [{"binding_source": "unbound"}], "errors": []},
    )
    assert CliRunner().invoke(app, ["instrumentation", "verify", "--spans-file", str(spans)]).exit_code == 1

    monkeypatch.setattr(cli, "_mcp_call", lambda *args, **kwargs: {"tasks": [], "errors": []})
    assert CliRunner().invoke(app, ["instrumentation", "verify", "--spans-file", str(spans)]).exit_code == 1


def test_cli_plan_writes_the_plan_and_reports_dropped(tmp_path, monkeypatch):
    from overmind import __main__ as cli

    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    sent: dict = {}

    def fake_call(api_url, api_key, tool, arguments, timeout):
        sent.update(tool=tool, arguments=arguments)
        return {
            "placements": [{"key": "k"}],
            "plans": [{"capability": "support", "placement_count": 1}],
            "ambiguous": [],
            "dropped": [{"key": "x", "reason": "no entry point"}],
            "minted": {"behaviours": 1, "versions": 1},
        }

    monkeypatch.setattr(cli, "_mcp_call", fake_call)
    plan = tmp_path / "plan.json"
    candidates = tmp_path / "candidates.json"

    result = CliRunner().invoke(
        app,
        [
            "instrumentation",
            "plan",
            "--root",
            str(tmp_path),
            "--out",
            str(plan),
            "--candidates-out",
            str(candidates),
            "--capability",
            "support",
        ],
    )

    assert result.exit_code == 0
    assert sent["tool"] == "plan_instrumentation"
    assert sent["arguments"]["capability_name_or_slug"] == "support"
    assert sent["arguments"]["candidates"] == json.loads(candidates.read_text())
    assert json.loads(plan.read_text())["placements"] == [{"key": "k"}]
    assert "no entry point" in result.stdout


def test_cli_plan_exits_on_planner_errors(tmp_path, monkeypatch):
    from overmind import __main__ as cli

    monkeypatch.setattr(cli, "_mcp_call", lambda *args, **kwargs: {"errors": ["no capability declared"]})
    result = CliRunner().invoke(
        app,
        [
            "instrumentation",
            "plan",
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "plan.json"),
            "--candidates-out",
            str(tmp_path / "candidates.json"),
        ],
    )
    assert result.exit_code == 1
    assert not (tmp_path / "plan.json").exists()


def test_cli_version_prints_the_sdk_version():
    from overmind import __version__

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help_lists_every_instrumentation_command():
    result = CliRunner().invoke(app, ["instrumentation", "--help"])
    assert result.exit_code == 0
    for command in ("scan", "plan", "check", "smoke", "verify"):
        assert command in result.stdout


def test_cli_help_documents_every_option_of_each_command():
    expected = {
        "scan": {"--root", "--out"},
        "plan": {"--root", "--out", "--candidates-out", "--capability", "--api-url", "--api-key"},
        "check": {"--plan-file", "--root", "--format"},
        "smoke": {"--plan-file", "--out", "--root"},
        "verify": {"--spans-file", "--capability", "--api-url", "--api-key"},
    }
    for command, options in expected.items():
        result = CliRunner().invoke(app, ["instrumentation", command, "--help"])
        assert result.exit_code == 0
        # Rich wraps long help panels, so compare against unwrapped output.
        rendered = result.stdout.replace("\n", " ")
        for option in options:
            assert option in rendered, (command, option)
