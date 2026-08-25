from overmind.scanner import scan


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text)


def test_scan_classifies_kinds_and_frameworks(tmp_path):
    _write(
        tmp_path,
        "api.py",
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "@app.get('/items')\n"
        "def list_items():\n"
        "    return []\n",
    )
    _write(
        tmp_path,
        "cli.py",
        "import click\n\n"
        "@click.command()\n"
        "def main():\n"
        "    pass\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
    )
    _write(
        tmp_path,
        "agent.py",
        "class Agent:\n    pass\n\n"
        "class MyAgent(Agent):\n"
        "    def run(self):\n"
        "        pass\n",
    )
    _write(
        tmp_path,
        "llm.py",
        "import openai\n\n"
        "def ask(client):\n"
        "    return client.chat.completions.create(model='gpt', messages=[])\n",
    )
    _write(
        tmp_path,
        "tools.py",
        "def search(query):\n    return query\n\n"
        "tools = [search]\n\n"
        "@tool\n"
        "def other_tool(x):\n"
        "    return x\n",
    )

    result = scan(str(tmp_path))

    assert result["schema_version"] == 1
    assert "repo_sha" in result
    assert set(result["frameworks_detected"]) >= {"fastapi", "click", "openai"}

    by_path = {f["path"]: f for f in result["files"]}
    assert by_path["api.py"]["symbols"][0]["kind"] == "route"
    assert by_path["api.py"]["symbols"][0]["qualname"] == "list_items"

    cli_kinds = {s["qualname"]: s["kind"] for s in by_path["cli.py"]["symbols"]}
    assert cli_kinds["main"] == "entry"

    agent_symbols = {s["qualname"]: s["kind"] for s in by_path["agent.py"]["symbols"]}
    assert agent_symbols["MyAgent"] == "agent_class"

    llm_symbols = {s["qualname"]: s["kind"] for s in by_path["llm.py"]["symbols"]}
    assert llm_symbols["ask"] == "llm_call"

    tool_symbols = {s["qualname"]: s["kind"] for s in by_path["tools.py"]["symbols"]}
    assert tool_symbols["search"] == "tool"
    assert tool_symbols["other_tool"] == "tool"

    # files with no symbols are omitted, and ordering is deterministic.
    assert list(by_path) == sorted(by_path)
    for file in result["files"]:
        linenos = [s["lineno"] for s in file["symbols"]]
        assert linenos == sorted(linenos)


def test_scan_symbol_shape(tmp_path):
    _write(
        tmp_path,
        "m.py",
        "@tool\n"
        "def helper(a, b=1, *args, **kwargs):\n"
        "    \"\"\"Does a thing.\"\"\"\n"
        "    pass\n",
    )
    result = scan(str(tmp_path))
    symbol = result["files"][0]["symbols"][0]
    assert symbol["qualname"] == "helper"
    assert symbol["kind"] == "tool"
    assert symbol["signature"] == "(a, b, *args, **kwargs)"
    assert symbol["docstring"] == "Does a thing."
    assert symbol["decorators"] == ["tool"]
    assert symbol["lineno"] == 2


def test_scan_omits_files_with_no_symbols(tmp_path):
    _write(tmp_path, "plain.py", "x = 1\n")
    result = scan(str(tmp_path))
    assert result["files"] == []
