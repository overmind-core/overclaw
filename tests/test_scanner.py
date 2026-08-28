"""Tests for the pure-AST instrumentation candidate scan."""

from __future__ import annotations

from overmind.scanner import scan


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text)


def test_scan_classifies_kinds_and_frameworks(tmp_path):
    _write(
        tmp_path,
        "api.py",
        "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/items')\ndef list_items():\n    return []\n",
    )
    _write(
        tmp_path,
        "cli.py",
        "import click\n\n@click.command()\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
    )
    _write(tmp_path, "agent.py", "class Agent:\n    pass\n\nclass MyAgent(Agent):\n    def run(self):\n        pass\n")
    _write(
        tmp_path,
        "llm.py",
        "import openai\n\ndef ask(client):\n    return client.chat.completions.create(model='gpt', messages=[])\n",
    )
    _write(
        tmp_path,
        "tools.py",
        "def search(query):\n    return query\n\ntools = [search]\n\n@tool\ndef other_tool(x):\n    return x\n",
    )

    result = scan(str(tmp_path))

    assert result["schema_version"] == 1
    assert "repo_sha" in result
    assert set(result["frameworks_detected"]) >= {"fastapi", "click", "openai"}

    by_path = {file["path"]: file for file in result["files"]}
    assert by_path["api.py"]["symbols"][0]["kind"] == "route"
    assert by_path["api.py"]["symbols"][0]["qualname"] == "api.list_items"
    assert {s["qualname"]: s["kind"] for s in by_path["cli.py"]["symbols"]}["cli.main"] == "entry"
    assert {s["qualname"]: s["kind"] for s in by_path["agent.py"]["symbols"]}["agent.MyAgent"] == "agent_class"
    assert {s["qualname"]: s["kind"] for s in by_path["llm.py"]["symbols"]}["llm.ask"] == "llm_call"

    tool_symbols = {s["qualname"]: s["kind"] for s in by_path["tools.py"]["symbols"]}
    assert tool_symbols["tools.search"] == "tool"
    assert tool_symbols["tools.other_tool"] == "tool"

    # Files with no symbols are omitted, and ordering is deterministic.
    assert list(by_path) == sorted(by_path)
    for file in result["files"]:
        linenos = [symbol["lineno"] for symbol in file["symbols"]]
        assert linenos == sorted(linenos)


def test_scan_symbol_shape(tmp_path):
    _write(tmp_path, "m.py", '@tool\ndef helper(a, b=1, *args, **kwargs):\n    """Does a thing."""\n    pass\n')
    symbol = scan(str(tmp_path))["files"][0]["symbols"][0]
    assert symbol == {
        "qualname": "m.helper",
        "kind": "tool",
        "signature": "(a, b, *args, **kwargs)",
        "docstring": "Does a thing.",
        "decorators": ["tool"],
        "lineno": 2,
        "source_line": "def helper(a, b=1, *args, **kwargs):",
    }


def test_scan_omits_files_with_no_symbols(tmp_path):
    _write(tmp_path, "plain.py", "x = 1\n")
    assert scan(str(tmp_path))["files"] == []


def test_scan_skips_tests_docs_and_venvs(tmp_path):
    for directory in ("tests", "docs", ".venv"):
        (tmp_path / directory).mkdir()
        _write(tmp_path, f"{directory}/thing.py", "def main():\n    pass\n")
    _write(tmp_path, "test_top_level.py", "def main():\n    pass\n")
    _write(tmp_path, "app.py", "def main():\n    pass\n")

    result = scan(str(tmp_path))

    assert [file["path"] for file in result["files"]] == ["app.py"]
    assert result["skipped"]["directories"] == 3
    assert result["skipped"]["files"] == 4


def test_scan_llm_call_via_litellm_import(tmp_path):
    _write(
        tmp_path, "llm.py", "import litellm\n\ndef ask():\n    return litellm.completion(model='gpt', messages=[])\n"
    )
    symbols = {s["qualname"]: s["kind"] for s in scan(str(tmp_path))["files"][0]["symbols"]}
    assert symbols["llm.ask"] == "llm_call"


def test_scan_llm_call_via_langchain_import(tmp_path):
    _write(
        tmp_path,
        "llm.py",
        "from langchain_openai import ChatOpenAI\n\ndef ask():\n    return ChatOpenAI(model='gpt-4').invoke('hi')\n",
    )
    symbols = {s["qualname"]: s["kind"] for s in scan(str(tmp_path))["files"][0]["symbols"]}
    assert symbols["llm.ask"] == "llm_call"


def test_scan_llm_call_needs_an_imported_client(tmp_path):
    """A bare ``.invoke()`` on an unknown object is not an LLM call."""
    _write(tmp_path, "plain.py", "def ask(thing):\n    return thing.invoke('hi')\n")
    assert scan(str(tmp_path))["files"] == []


def test_scan_entry_excludes_nested_main(tmp_path):
    _write(tmp_path, "m.py", "class Runner:\n    def main(self):\n        pass\n")
    assert scan(str(tmp_path))["files"] == []


def test_scan_qualifies_same_named_top_level_symbols(tmp_path):
    _write(tmp_path, "one.py", "def main():\n    pass\n")
    _write(tmp_path, "two.py", "def main():\n    pass\n")

    result = scan(str(tmp_path))

    assert [file["symbols"][0]["qualname"] for file in result["files"]] == [
        "one.main",
        "two.main",
    ]
