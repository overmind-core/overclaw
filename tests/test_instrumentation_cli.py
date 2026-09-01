from __future__ import annotations

from overmind.__main__ import _run_smoke, _write_smoke_scaffolds


def test_scaffold_imports_the_planned_function(tmp_path):
    placements = [
        {
            "key": "generate-report",
            "smoke_hint": "call the report entry",
            "target": {
                "module": "backend.server.app",
                "qualname": "backend.server.app.generate_report",
            },
        }
    ]

    assert _write_smoke_scaffolds(placements, tmp_path) == ["smoke_generate_report.py"]

    scaffold = (tmp_path / "smoke_generate_report.py").read_text()
    assert "from backend.server.app import generate_report" in scaffold
    assert "generate_report(...)" in scaffold
    assert "instance =" not in scaffold


def test_smoke_subprocess_uses_the_file_exporter(tmp_path, monkeypatch):
    script = tmp_path / "smoke.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['OVERMIND_TRACE_FILE']).write_text(\n"
        "    f\"{os.getenv('OVERMIND_API_KEY')}|{os.getenv('OVERMIND_API_URL')}|\"\n"
        "    f\"{os.getenv('OVERMIND_SMOKE')}\"\n"
        ")\n"
    )
    monkeypatch.setenv("OVERMIND_API_KEY", "key")
    monkeypatch.setenv("OVERMIND_API_URL", "http://localhost:8000")
    out = tmp_path / "spans.jsonl"

    assert _run_smoke([{"smoke_script": script.name}], out, tmp_path) == (1, 0, [])
    assert out.read_text() == "None|None|1"
