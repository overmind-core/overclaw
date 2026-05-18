"""End-to-end integration test for the bundler + spec generator pipeline
against a non-standard layout (hyphenated package root + dynamic import).

This is the codebase-agnostic replacement for the airline-repo probe used
during the previous debugging cycle. If this test passes, the same flow
holds for any consumer repo whose layout depends on a ``sys.path``
mutation in the entry — hyphenated dirs, sibling-package monorepos,
projects without ``src/`` or ``pyproject.toml``.

The fixture is constructed inside ``tmp_path`` rather than checked in
under ``tests/fixtures/`` so we get hermetic per-test state and avoid
polluting the repo root with synthetic Python packages.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.setup.agent_analyzer import _display_analysis
from overmind.setup.spec_generator import generate_spec_from_proposal
from overmind.utils.code import AgentBundle


@pytest.fixture()
def hyphen_layout(tmp_path: Path) -> tuple[Path, Path]:
    """A small project that exercises every fix in this batch:

    * ``entry.py`` lives at the repo root and mutates ``sys.path`` to
      register ``py-backend/`` — a hyphenated directory whose name is
      not a valid Python identifier, so it cannot appear as a top-level
      package.
    * The package itself (``pkg``) is two levels deep and re-exports
      from a sub-package, so the BFS has to follow chains and pick up
      intermediate ``__init__.py`` files.
    * The entry also imports ``sibling`` directly (a top-level module
      under ``py-backend/`` that's *not* part of ``pkg``).
    * A guarded ``importlib.import_module`` is in scope to confirm we
      don't regress on the ``__overmind_imports__`` static-hint path.
    """
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    entry = tmp_path / "entry.py"
    entry.write_text(
        textwrap.dedent("""\
        import importlib
        import sys
        from pathlib import Path

        _PROJECT_ROOT = Path(__file__).resolve().parent
        _BACKEND = _PROJECT_ROOT / "py-backend"
        if str(_BACKEND) not in sys.path:
            sys.path.insert(0, str(_BACKEND))

        # Static hint for the dynamic loader below — exercises the
        # ``__overmind_imports__`` path so its interaction with
        # ``sys.path`` detection is covered.
        __overmind_imports__ = ["pkg.sub.deep"]

        from pkg.mod import compute
        import sibling

        def run(payload):
            mod = importlib.import_module(payload["mod"])
            return {"value": compute(payload), "deep": mod.value(), "sib": sibling.tag()}
        """)
    )

    backend = tmp_path / "py-backend"
    pkg = backend / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("def compute(x): return x\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "deep.py").write_text("def value(): return 42\n")
    (backend / "sibling.py").write_text("def tag(): return 'sib'\n")

    return tmp_path, entry


class TestHyphenLayoutEndToEnd:
    def test_bundle_is_multi_file_without_explicit_search_paths(
        self, hyphen_layout
    ):
        """The bundler picks up the entry's ``sys.path.insert`` via
        ``_detect_entry_search_paths`` and produces a multi-file bundle
        covering the package, sub-package, and sibling module — no
        ``search_paths`` argument required from the caller."""
        root, entry = hyphen_layout
        bundle = AgentBundle.from_entry_point(
            entry_path=str(entry),
            project_root=str(root),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert bundle.is_multi_file()
        assert "entry.py" in files
        assert "py-backend/pkg/mod.py" in files
        assert "py-backend/pkg/__init__.py" in files
        assert "py-backend/pkg/sub/deep.py" in files
        assert "py-backend/pkg/sub/__init__.py" in files
        assert "py-backend/sibling.py" in files

    def test_spec_injects_search_paths_when_llm_omits(self, hyphen_layout):
        """The LLM-emitted analysis lacks ``search_paths`` (a very
        common failure mode); ``_build_spec`` runs the AST evaluator
        and injects ``py-backend`` deterministically so the spec on
        disk matches the entry's actual runtime behaviour."""
        root, entry = hyphen_layout
        analysis = {
            "description": "synthetic",
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "_entrypoint_fn": "run",
            "output_schema": {"value": {"type": "number", "range": [0, 100]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"value": {"importance": "important"}},
            },
            "scope": {
                "optimizable_paths": ["py-backend/pkg/**/*.py"],
                # No "search_paths" — simulating the LLM forgetting.
                # No "exclude_paths" — D is exercised in the next test.
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert "py-backend" in spec["scope"].get("search_paths", [])

    def test_spec_collapses_legacy_exclude_paths(self, hyphen_layout):
        """The LLM put the entry in ``exclude_paths`` and added some
        infra paths. The post-process drops ``exclude_paths`` entirely
        (project-level drops belong in ``.overmindignore``; Overmind's
        hard-coded skip list handles env-level) and auto-adds the
        entry to ``read_only_paths`` so the accept step protects it."""
        root, entry = hyphen_layout
        analysis = {
            "description": "synthetic",
            "_agent_path": str(entry),
            "_entry_rel": "entry.py",
            "_entrypoint_fn": "run",
            "output_schema": {"value": {"type": "number", "range": [0, 100]}},
            "proposed_criteria": {
                "structure_weight": 20,
                "fields": {"value": {"importance": "important"}},
            },
            "scope": {
                "optimizable_paths": ["py-backend/pkg/**/*.py"],
                "exclude_paths": ["entry.py", "tests/**"],
            },
        }
        spec = generate_spec_from_proposal(analysis)
        assert "exclude_paths" not in spec["scope"]
        assert "entry.py" in spec["scope"]["read_only_paths"]

    def test_display_renders_new_scope_rows(self, hyphen_layout):
        """``_display_analysis`` prints ``read_only_paths`` and
        ``search_paths`` so users actually see what the analyzer (and
        the post-process) emitted. Capture by mocking the console and
        scanning the printed strings — the rich Table objects render
        labels we can assert against."""
        console = MagicMock()
        analysis = {
            "description": "synthetic",
            "output_schema": {},
            "scope": {
                "optimizable_paths": ["py-backend/pkg/**/*.py"],
                "read_only_paths": ["entry.py", "README.md"],
                "search_paths": ["py-backend"],
            },
        }
        _display_analysis(analysis, console)
        # Flatten every printed positional arg into a single string and
        # confirm the new field labels are present somewhere in the
        # output. The exact rich-rendered layout is not part of the
        # contract — only that the new keys are surfaced at all.
        printed = " ".join(
            str(arg)
            for call in console.print.call_args_list
            for arg in call.args
        )
        assert "Read-only" in printed
        assert "Search paths" in printed
