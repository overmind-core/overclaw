"""Tests for overclaw.core.registry — agents.toml registry CRUD."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from overclaw.utils.code import AgentBundle
from overclaw.core.constants import OVERCLAW_DIR_NAME
from overclaw.core.registry import (
    _entries_to_toml_array,
    _module_to_file,
    _raw_agents_to_entries,
    _str_val,
    init_project_root,
    load_registry,
    parse_entrypoint,
    project_root,
    project_root_from_agent_file,
    remove_agent,
    require_overclaw_initialized,
    resolve_agent,
    save_agent,
    validate_entrypoint,
)


# ---------------------------------------------------------------------------
# parse_entrypoint
# ---------------------------------------------------------------------------


class TestParseEntrypoint:
    def test_valid_entrypoint(self):
        module, fn = parse_entrypoint("agents.agent1.sample_agent:run")
        assert module == "agents.agent1.sample_agent"
        assert fn == "run"

    def test_colon_at_end(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_entrypoint("some.module:")

    def test_colon_at_start(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_entrypoint(":function")

    def test_no_colon(self):
        with pytest.raises(ValueError, match="Invalid entrypoint"):
            parse_entrypoint("no_colon_here")

    def test_multiple_colons_takes_last(self):
        module, fn = parse_entrypoint("a.b:c:d")
        assert module == "a.b:c"
        assert fn == "d"

    def test_whitespace_stripped(self):
        module, fn = parse_entrypoint("  mod.sub : func  ")
        assert module == "mod.sub"
        assert fn == "func"

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_entrypoint("")

    def test_only_colon(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_entrypoint(":")


# ---------------------------------------------------------------------------
# _str_val
# ---------------------------------------------------------------------------


class TestStrVal:
    def test_none(self):
        assert _str_val(None) == ""

    def test_string(self):
        assert _str_val("  hello  ") == "hello"

    def test_int(self):
        assert _str_val(42) == "42"


# ---------------------------------------------------------------------------
# _raw_agents_to_entries
# ---------------------------------------------------------------------------


class TestRawAgentsToEntries:
    def test_none(self):
        assert _raw_agents_to_entries(None) == []

    def test_empty_list(self):
        assert _raw_agents_to_entries([]) == []

    def test_array_format(self):
        raw = [
            {"name": "a", "entrypoint": "m:f"},
            {"name": "b", "entrypoint": "m2:g"},
        ]
        entries = _raw_agents_to_entries(raw)
        assert len(entries) == 2
        assert entries[0]["name"] == "a"

    def test_array_skips_incomplete(self):
        raw = [{"name": "a"}, {"entrypoint": "m:f"}, {"name": "b", "entrypoint": "x:y"}]
        entries = _raw_agents_to_entries(raw)
        assert len(entries) == 1
        assert entries[0]["name"] == "b"

    def test_legacy_dict_format(self):
        raw = {
            "agent1": {"entrypoint": "m1:f1"},
            "agent2": {"entrypoint": "m2:f2"},
        }
        entries = _raw_agents_to_entries(raw)
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert names == {"agent1", "agent2"}

    def test_legacy_dict_with_non_dict_value(self):
        raw = {"agent1": "not-a-dict"}
        entries = _raw_agents_to_entries(raw)
        assert entries == []

    def test_unexpected_type(self):
        assert _raw_agents_to_entries(42) == []
        assert _raw_agents_to_entries("string") == []


# ---------------------------------------------------------------------------
# _entries_to_toml_array
# ---------------------------------------------------------------------------


class TestEntriesToTomlArray:
    def test_sorted_by_name(self):
        entries = [
            {"name": "zoo", "entrypoint": "z:z"},
            {"name": "alpha", "entrypoint": "a:a"},
        ]
        arr = _entries_to_toml_array(entries)
        assert arr[0]["name"] == "alpha"
        assert arr[1]["name"] == "zoo"

    def test_empty_list(self):
        arr = _entries_to_toml_array([])
        assert len(arr) == 0


# ---------------------------------------------------------------------------
# _module_to_file
# ---------------------------------------------------------------------------


class TestModuleToFile:
    def test_converts_dots_to_path(self, tmp_path):
        result = _module_to_file("agents.agent1.sample_agent", tmp_path)
        expected = tmp_path / "agents" / "agent1" / "sample_agent.py"
        assert result == expected

    def test_single_module(self, tmp_path):
        result = _module_to_file("mymodule", tmp_path)
        assert result == tmp_path / "mymodule.py"


# ---------------------------------------------------------------------------
# project_root / init_project_root
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_resolves_overclaw_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / OVERCLAW_DIR_NAME).mkdir()
        monkeypatch.chdir(tmp_path)
        assert project_root() == tmp_path.resolve()

    def test_resolves_overclaw_in_parent(self, tmp_path, monkeypatch):
        (tmp_path / OVERCLAW_DIR_NAME).mkdir()
        child = tmp_path / "sub" / "dir"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        assert project_root() == tmp_path.resolve()

    def test_exits_when_no_overclaw(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            project_root()
        assert exc_info.value.code == 1

    def test_init_project_root_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert init_project_root() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# project_root_from_agent_file
# ---------------------------------------------------------------------------


class TestProjectRootFromAgentFile:
    def test_finds_overclaw_above_deep_entry(self, tmp_path):
        (tmp_path / OVERCLAW_DIR_NAME).mkdir()
        runner = tmp_path / "agents" / "agent3" / "original_agent" / "runner.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("# entry\n", encoding="utf-8")
        assert project_root_from_agent_file(runner) == tmp_path.resolve()

    def test_returns_none_without_overclaw_ancestor(self, tmp_path):
        runner = tmp_path / "nested" / "runner.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("#\n", encoding="utf-8")
        assert project_root_from_agent_file(runner) is None

    def test_bundler_sees_multiple_agent_files_for_nested_entry(self, tmp_path):
        (tmp_path / OVERCLAW_DIR_NAME).mkdir()
        pkg = tmp_path / "agents" / "demo" / "original_agent"
        pkg.mkdir(parents=True)
        (pkg / "config.py").write_text('MODEL = "x"\n', encoding="utf-8")
        (pkg / "runner.py").write_text(
            textwrap.dedent("""\
            from agents.demo.original_agent.config import MODEL

            def run(input_data: dict) -> dict:
                return {"m": MODEL}
            """),
            encoding="utf-8",
        )
        root = project_root_from_agent_file(pkg / "runner.py")
        assert root == tmp_path.resolve()
        bundle = AgentBundle.from_entry_point(str(pkg / "runner.py"), str(root), "run")
        assert bundle.is_multi_file()
        agent_paths = {p for p in bundle.original_files if p.startswith("agents/")}
        assert "agents/demo/original_agent/config.py" in agent_paths
        assert "agents/demo/original_agent/runner.py" in agent_paths


# ---------------------------------------------------------------------------
# require_overclaw_initialized
# ---------------------------------------------------------------------------


class TestRequireOverclawInitialized:
    def test_no_op_when_overclaw_present(self, tmp_path, monkeypatch):
        (tmp_path / OVERCLAW_DIR_NAME).mkdir()
        monkeypatch.chdir(tmp_path)
        require_overclaw_initialized()

    def test_exits_when_no_overclaw(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            require_overclaw_initialized()
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# validate_entrypoint
# ---------------------------------------------------------------------------


class TestValidateEntrypoint:
    def test_valid(self, tmp_project):
        file_path, fn = validate_entrypoint("agents.agent1.sample_agent:run")
        assert fn == "run"
        assert file_path.exists()

    def test_invalid_format(self, tmp_project):
        with pytest.raises(SystemExit):
            validate_entrypoint("no_colon")

    def test_missing_file(self, tmp_project):
        with pytest.raises(SystemExit):
            validate_entrypoint("nonexistent.module:run")

    def test_missing_function(self, tmp_project):
        with pytest.raises(SystemExit):
            validate_entrypoint("agents.agent1.sample_agent:nonexistent_func")

    def test_function_with_space_before_paren(self, tmp_project):
        agent_file = tmp_project / "agents" / "agent1" / "sample_agent.py"
        agent_file.write_text("def spaced_run (input_data):\n    pass\n")
        file_path, fn = validate_entrypoint("agents.agent1.sample_agent:spaced_run")
        assert fn == "spaced_run"


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    def test_loads_agents(self, tmp_project):
        registry = load_registry()
        assert "my-agent" in registry
        assert registry["my-agent"]["entrypoint"] == "agents.agent1.sample_agent:run"
        assert registry["my-agent"]["fn_name"] == "run"

    def test_empty_registry(self, tmp_project_empty):
        registry = load_registry()
        assert registry == {}

    def test_malformed_entrypoint_still_loads(self, tmp_path, monkeypatch):
        oc = tmp_path / OVERCLAW_DIR_NAME
        oc.mkdir()
        (oc / "agents.toml").write_text(
            textwrap.dedent("""\
            agents = [{ name = "bad", entrypoint = "no-colon" }]
            """),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        registry = load_registry()
        assert "bad" in registry
        assert registry["bad"]["file_path"] == ""
        assert registry["bad"]["fn_name"] == ""


# ---------------------------------------------------------------------------
# save_agent
# ---------------------------------------------------------------------------


class TestSaveAgent:
    def test_adds_new_agent(self, tmp_project):
        save_agent("new-agent", "agents.agent1.sample_agent:helper")
        registry = load_registry()
        assert "new-agent" in registry
        assert "my-agent" in registry  # existing preserved

    def test_overwrites_existing(self, tmp_project):
        save_agent("my-agent", "agents.agent1.sample_agent:helper")
        registry = load_registry()
        assert registry["my-agent"]["entrypoint"] == "agents.agent1.sample_agent:helper"

    def test_creates_agents_toml_if_missing(self, tmp_project_empty):
        agent_dir = Path.cwd() / "agents" / "test"
        agent_dir.mkdir(parents=True)
        (agent_dir / "mod.py").write_text("def f(): pass\n")
        save_agent("test-agent", "agents.test.mod:f")
        registry = load_registry()
        assert "test-agent" in registry
        assert (Path.cwd() / OVERCLAW_DIR_NAME / "agents.toml").is_file()


# ---------------------------------------------------------------------------
# remove_agent
# ---------------------------------------------------------------------------


class TestRemoveAgent:
    def test_removes_existing(self, tmp_project):
        remove_agent("my-agent")
        registry = load_registry()
        assert "my-agent" not in registry

    def test_raises_keyerror_for_missing(self, tmp_project):
        with pytest.raises(KeyError):
            remove_agent("nonexistent")


# ---------------------------------------------------------------------------
# resolve_agent
# ---------------------------------------------------------------------------


class TestResolveAgent:
    def test_resolves_known_agent(self, tmp_project):
        file_path, fn_name = resolve_agent("my-agent")
        assert Path(file_path).exists()
        assert fn_name == "run"

    def test_unknown_agent_exits(self, tmp_project):
        with pytest.raises(SystemExit):
            resolve_agent("not-registered")

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        oc = tmp_path / OVERCLAW_DIR_NAME
        oc.mkdir()
        (oc / "agents.toml").write_text(
            'agents = [{ name = "ghost", entrypoint = "gone.module:run" }]\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            resolve_agent("ghost")
