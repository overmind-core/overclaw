"""Tests for ``detect_external_imports`` — the dep-manifest gate.

This is the entry point for ``AgentRunner.ensure_environment``'s check
that the agent has a ``requirements.txt`` (or equivalent) covering every
PyPI package it actually imports. The pre-fix version only parsed the
entry file, so:

- Transitive external imports (entry imports a helper, helper imports
  ``litellm``) were invisible. Agents crashed mid-run with
  ``ModuleNotFoundError``.
- Nested local packages (``python_backend/airline/`` inside the
  project) were falsely flagged as external because the depth-1
  ``project_root.iterdir()`` heuristic didn't see them. Registration
  raised ``MissingDependenciesError`` for projects whose layout the
  bundler itself fully supports.

These tests pin the post-fix behavior: closure-aware, unified with the
bundle resolver.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.optimize.runner import (
    Language,
    detect_external_imports,
    find_dep_manifest_dir,
    has_dep_manifest,
)


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestDetectExternalImportsPython:
    def test_transitive_external_is_detected(self, project: Path):
        """Entry imports a local helper, helper imports an external
        package — the external must be flagged."""
        (project / "entry.py").write_text(
            textwrap.dedent("""\
            from helper import compute

            def run(input_data):
                return {"result": compute(input_data)}
            """)
        )
        (project / "helper.py").write_text(
            textwrap.dedent("""\
            import litellm

            def compute(x):
                return litellm.completion(x)
            """)
        )
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert "litellm" in result

    def test_nested_local_package_is_not_external(self, project: Path):
        """The airline-shaped bug. A package nested under a subdir is
        local, not external — detect must not raise the alarm."""
        (project / "entry.py").write_text(
            textwrap.dedent("""\
            from python_backend.airline.agents import triage

            def run(input_data):
                return {"result": triage(input_data)}
            """)
        )
        pkg = project / "python_backend"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        airline = pkg / "airline"
        airline.mkdir()
        (airline / "__init__.py").write_text("")
        (airline / "agents.py").write_text(
            textwrap.dedent("""\
            def triage(x):
                return x
            """)
        )
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert "python_backend" not in result
        assert "airline" not in result

    def test_mixed_externals_and_locals_returns_only_externals(self, project: Path):
        """Real agents import a mix. Verify exactly the externals come
        back, deduplicated, with locals and stdlib filtered out."""
        (project / "entry.py").write_text(
            textwrap.dedent("""\
            import os
            import json
            import litellm
            from openai import OpenAI
            from helper import compute

            def run(input_data):
                return {"result": compute(litellm.completion(input_data))}
            """)
        )
        (project / "helper.py").write_text(
            textwrap.dedent("""\
            import litellm  # duplicate, must dedupe

            def compute(x):
                return x
            """)
        )
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert set(result) == {"litellm", "openai"}
        # Stdlib is filtered.
        assert "os" not in result
        assert "json" not in result
        # Local helper is filtered.
        assert "helper" not in result

    def test_overmind_sdk_is_not_external(self, project: Path):
        """The ``overmind`` SDK is always available at runtime even
        when not in ``requirements.txt`` — it's installed by the
        runner itself."""
        (project / "entry.py").write_text(
            textwrap.dedent("""\
            import overmind
            from overmind import observe

            def run(input_data):
                return {}
            """)
        )
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert "overmind" not in result

    def test_missing_entry_file_returns_empty(self, project: Path):
        result = detect_external_imports(project, "does_not_exist.py", Language.PYTHON)
        assert result == []

    def test_entry_only_no_local_helpers(self, project: Path):
        """Single-file agent with only stdlib + one external. The
        closure walk degenerates to just the entry, but the result
        must still be correct."""
        (project / "entry.py").write_text(
            textwrap.dedent("""\
            import os
            import requests

            def run(input_data):
                return {"status": requests.get("https://example.com").status_code}
            """)
        )
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert result == ["requests"]

    def test_deeply_transitive_external(self, project: Path):
        """Three-hop import chain: entry -> a -> b -> external. The
        old single-file scanner would only see ``a`` (local) and miss
        ``c_external`` entirely; the closure walk catches it."""
        (project / "entry.py").write_text("from a import f\n\ndef run(x): return f(x)\n")
        (project / "a.py").write_text("from b import g\ndef f(x): return g(x)\n")
        (project / "b.py").write_text("import c_external\ndef g(x): return c_external.do(x)\n")
        result = detect_external_imports(project, "entry.py", Language.PYTHON)
        assert "c_external" in result

    def test_relative_imports_do_not_leak_as_externals(self, project: Path):
        """A relative import (``from . import x``) inside a package
        must not surface as an external — the resolver handles these
        and they're always local by definition."""
        pkg = project / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "entry.py").write_text(
            textwrap.dedent("""\
            from . import helper
            from .helper import compute

            def run(input_data):
                return {"result": compute(input_data)}
            """)
        )
        (pkg / "helper.py").write_text(
            textwrap.dedent("""\
            def compute(x):
                return x
            """)
        )
        result = detect_external_imports(pkg, "entry.py", Language.PYTHON)
        assert result == []


# ---------------------------------------------------------------------------
# Manifest walk-up
# ---------------------------------------------------------------------------
#
# ``has_dep_manifest`` is the gate that triggers
# ``MissingDependenciesError`` when the agent has external imports but
# no ``requirements.txt`` / ``pyproject.toml``. Pre-Patch-4c it only
# looked at the agent's own directory, producing a false-positive miss
# for monorepos and src-layouts that keep manifests at the repo root.
# The walk now ascends to the Overmind project root so the gate
# matches the install-from-here behaviour of ``_provision_python``.


class TestHasDepManifestWalkUp:
    def test_manifest_at_agent_dir(self, project: Path):
        (project / "requirements.txt").write_text("requests\n")
        assert has_dep_manifest(project, Language.PYTHON) is True
        assert find_dep_manifest_dir(project, Language.PYTHON) == project

    def test_manifest_at_project_root_monorepo(self, project: Path):
        """The defining monorepo case: agent lives in
        ``services/triage/`` while ``requirements.txt`` sits at the
        repo root. Pre-fix this raised MissingDependenciesError; now
        the walk finds the manifest one level up."""
        (project / "requirements.txt").write_text("requests\n")
        agent_dir = project / "services" / "triage"
        agent_dir.mkdir(parents=True)
        assert has_dep_manifest(agent_dir, Language.PYTHON) is True
        assert find_dep_manifest_dir(agent_dir, Language.PYTHON) == project

    def test_manifest_at_intermediate_dir(self, project: Path):
        """If both the project root AND an intermediate ancestor have
        a manifest, the closest ancestor wins (it's the most specific
        scope and what the user would expect to install from)."""
        (project / "requirements.txt").write_text("pkg-a\n")
        intermediate = project / "services"
        intermediate.mkdir()
        (intermediate / "pyproject.toml").write_text(
            "[project]\nname = 'svc'\n"
        )
        agent_dir = intermediate / "triage"
        agent_dir.mkdir()
        assert find_dep_manifest_dir(agent_dir, Language.PYTHON) == intermediate

    def test_no_manifest_anywhere(self, project: Path):
        agent_dir = project / "services" / "triage"
        agent_dir.mkdir(parents=True)
        assert has_dep_manifest(agent_dir, Language.PYTHON) is False
        assert find_dep_manifest_dir(agent_dir, Language.PYTHON) is None

    def test_walk_stops_at_project_root(self, tmp_path: Path):
        """A manifest one level ABOVE the Overmind project root must
        be ignored — picking it up would couple unrelated projects
        (e.g. a parent monorepo whose deps don't match)."""
        outer = tmp_path / "outer_monorepo"
        outer.mkdir()
        (outer / "requirements.txt").write_text("alien\n")

        inner_project = outer / "overmind_project"
        inner_project.mkdir()
        (inner_project / OVERMIND_DIR_NAME).mkdir()

        agent_dir = inner_project / "services"
        agent_dir.mkdir()
        # Walk hits the project root and stops; the alien manifest above
        # it is never consulted.
        assert find_dep_manifest_dir(agent_dir, Language.PYTHON) is None
        assert has_dep_manifest(agent_dir, Language.PYTHON) is False

    def test_pyproject_toml_detected(self, project: Path):
        (project / "pyproject.toml").write_text("[project]\nname='x'\n")
        agent_dir = project / "deep" / "nested"
        agent_dir.mkdir(parents=True)
        assert find_dep_manifest_dir(agent_dir, Language.PYTHON) == project

    def test_javascript_walkup(self, project: Path):
        """The walk-up applies to JS just as much as Python — monorepos
        with a root-level ``package.json`` are common."""
        (project / "package.json").write_text('{"name":"root"}')
        agent_dir = project / "apps" / "web"
        agent_dir.mkdir(parents=True)
        assert has_dep_manifest(agent_dir, Language.JAVASCRIPT) is True
        assert find_dep_manifest_dir(agent_dir, Language.JAVASCRIPT) == project
