"""Tests for overmind.optimize.bundle_factory — read_only_scope plumbing.

These tests guard against the class of bug encountered when an agent
declared its entrypoint inside ``exclude_paths`` (collapsing the bundle)
and had no machine-readable way to say "this file is in the bundle but
candidates may not edit it." ``read_only_paths`` is the proper knob, and
these tests pin its end-to-end behavior across ``AgentBundle.from_entry_point``
and ``build_agent_bundle``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.optimize.bundle_factory import build_agent_bundle
from overmind.optimize.config import Config
from overmind.utils.code import (
    AgentBundle,
    BundleConfigError,
    _detect_entry_search_paths,
    resolve_local_files,
)


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal Overmind project with an entry file and a helper module."""
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    (tmp_path / "entrypoint.py").write_text(
        textwrap.dedent("""\
        from helper import compute

        def run(input_data: dict) -> dict:
            return {"result": compute(input_data)}
        """),
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        textwrap.dedent("""\
        def compute(x):
            return x
        """),
        encoding="utf-8",
    )
    (tmp_path / "agent_logic.py").write_text(
        textwrap.dedent("""\
        PROMPT = "Hello"
        """),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_config(project: Path, **overrides) -> Config:
    cfg = Config(
        agent_name="bundle-test",
        agent_path=str(project / "entrypoint.py"),
        entrypoint_fn="run",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestReadOnlyScopeBundling:
    def test_read_only_file_present_in_original_files(self, project: Path):
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            read_only_scope=["helper.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "helper.py" in bundle.original_files

    def test_read_only_file_excluded_from_optimizable(self, project: Path):
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            read_only_scope=["helper.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "helper.py" not in bundle.optimizable_files
        assert "helper.py" in bundle.read_only_files

    def test_entry_file_in_read_only_scope_is_not_optimizable(self, project: Path):
        """The entry file is auto-added to opt_set by ``from_entry_point``;
        read_only_scope must override that so candidates cannot edit the
        registered harness."""
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            read_only_scope=["entrypoint.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "entrypoint.py" in bundle.original_files
        assert "entrypoint.py" in bundle.read_only_files
        assert "entrypoint.py" not in bundle.optimizable_files

    def test_optimizable_only_when_no_read_only_scope(self, project: Path):
        cfg = _make_config(project, optimizable_scope=["agent_logic.py"])
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert bundle.read_only_files == set()
        assert "entrypoint.py" in bundle.optimizable_files

    def test_read_only_glob_expansion(self, project: Path):
        """Globs in read_only_scope must be expanded relative to project root."""
        (project / "fixtures").mkdir()
        (project / "fixtures" / "data1.py").write_text("VALUE = 1\n")
        (project / "fixtures" / "data2.py").write_text("VALUE = 2\n")
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            read_only_scope=["fixtures/*.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "fixtures/data1.py" in bundle.read_only_files
        assert "fixtures/data2.py" in bundle.read_only_files

    def test_baseline_content_captured_for_diff(self, project: Path):
        """The bundle must capture the file's content at bundle time so the
        accept step can diff a candidate worktree against it."""
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            read_only_scope=["entrypoint.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        captured = bundle.original_files["entrypoint.py"]
        on_disk = (project / "entrypoint.py").read_text()
        assert captured == on_disk


class TestFromEntryPointDirect:
    """Lower-level checks against ``AgentBundle.from_entry_point`` so the
    contract is pinned independent of the factory."""

    def test_read_only_paths_param_excludes_from_opt_set(self, project: Path):
        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entrypoint.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=["agent_logic.py", "entrypoint.py"],
            read_only_paths=["entrypoint.py"],
        )
        assert "entrypoint.py" in bundle.read_only_files
        assert "entrypoint.py" not in bundle.optimizable_files
        assert "agent_logic.py" in bundle.optimizable_files

    def test_missing_read_only_file_is_silently_ignored(self, project: Path):
        """Patterns that don't match any file (typos, deleted files) shouldn't
        crash bundle construction — they just don't contribute anything."""
        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entrypoint.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=["agent_logic.py"],
            read_only_paths=["does_not_exist.py"],
        )
        assert "does_not_exist.py" not in bundle.original_files
        assert bundle.read_only_files == set()

    def test_read_only_paths_default_is_empty(self, project: Path):
        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entrypoint.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=["agent_logic.py"],
        )
        assert bundle.read_only_files == set()


# ---------------------------------------------------------------------------
# BFS dependency mapping — nested package resolution
# ---------------------------------------------------------------------------
#
# These tests pin the post-fix behavior of ``_is_local_module`` /
# ``resolve_local_files``. Before the fix, the BFS only recognised
# packages that lived as DIRECT children of ``project_root``. Anything
# nested under a subdir (``src/``, ``python_backend/``, monorepo-style
# layouts) was misclassified as external and silently dropped from the
# bundle, leaving candidate worktrees missing transitive code.


@pytest.fixture()
def nested_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Project with a deeply nested local package and a root-level entry."""
    (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
    (tmp_path / "entry.py").write_text(
        textwrap.dedent("""\
        from python_backend.airline.agents import triage
        from python_backend.airline.tools import lookup

        def run(input_data):
            return {"result": triage(lookup(input_data))}
        """),
        encoding="utf-8",
    )
    pkg = tmp_path / "python_backend"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    inner = pkg / "airline"
    inner.mkdir()
    (inner / "__init__.py").write_text("")
    (inner / "agents.py").write_text(
        textwrap.dedent("""\
        from python_backend.airline.context import Context

        def triage(x):
            return Context(x).resolve()
        """),
        encoding="utf-8",
    )
    (inner / "tools.py").write_text(
        textwrap.dedent("""\
        def lookup(x):
            return x
        """),
        encoding="utf-8",
    )
    (inner / "context.py").write_text(
        textwrap.dedent("""\
        class Context:
            def __init__(self, x):
                self.x = x
            def resolve(self):
                return self.x
        """),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestNestedPackageResolution:
    def test_nested_package_files_are_bundled(self, nested_project: Path):
        """The exact bug from the airline run: a top-level entry imports a
        package nested under a subdirectory. All transitively-imported
        files must end up in the bundle."""
        bundle = AgentBundle.from_entry_point(
            entry_path=str(nested_project / "entry.py"),
            project_root=str(nested_project),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "entry.py" in files
        assert "python_backend/airline/agents.py" in files
        assert "python_backend/airline/tools.py" in files
        assert "python_backend/airline/context.py" in files

    def test_namespace_package_without_init_is_followed(self, tmp_path: Path):
        """PEP 420 namespace packages have no ``__init__.py``. The
        resolver must still find leaf modules under them by walking the
        dotted path directly to the ``.py`` file — otherwise modern
        Python projects (which often skip empty ``__init__.py`` files)
        would silently lose code from the bundle."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            from mypkg.submod import helper

            def run(input_data):
                return {"result": helper(input_data)}
            """)
        )
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        # Deliberately no __init__.py — this is a namespace package.
        (pkg / "submod.py").write_text(
            textwrap.dedent("""\
            def helper(x):
                return x
            """)
        )
        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        assert "mypkg/submod.py" in bundle.original_files

    def test_stdlib_imports_not_bundled(self, nested_project: Path):
        """Adding a stdlib import to the entry must not cause stdlib files
        to be sucked into the bundle. The stdlib short-circuit in
        ``_is_local_module`` is what protects against this."""
        (nested_project / "entry.py").write_text(
            textwrap.dedent("""\
            import os
            import sys
            import json
            from python_backend.airline.agents import triage

            def run(input_data):
                return {"result": triage(os.environ.get("X", ""))}
            """)
        )
        bundle = AgentBundle.from_entry_point(
            entry_path=str(nested_project / "entry.py"),
            project_root=str(nested_project),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "os.py" not in files
        assert "sys.py" not in files
        assert "json.py" not in files
        assert "json/__init__.py" not in files
        assert "python_backend/airline/agents.py" in files

    def test_flat_layout_regression(self, project: Path):
        """The original flat layout (helper.py next to entry.py) must keep
        working after the heuristic change."""
        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entrypoint.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        assert "entrypoint.py" in bundle.original_files
        assert "helper.py" in bundle.original_files

    def test_external_unresolvable_module_is_dropped(self, nested_project: Path):
        """A genuinely external import (no matching file under
        project_root) must NOT be added to the bundle. This is the
        symmetric guarantee to the nested-resolution win — we don't
        want to accidentally bundle the world."""
        (nested_project / "entry.py").write_text(
            textwrap.dedent("""\
            import litellm  # not on disk; must be treated as external
            from python_backend.airline.agents import triage

            def run(input_data):
                return {"result": triage(input_data)}
            """)
        )
        bundle = AgentBundle.from_entry_point(
            entry_path=str(nested_project / "entry.py"),
            project_root=str(nested_project),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "litellm.py" not in files
        assert "litellm/__init__.py" not in files
        assert "python_backend/airline/agents.py" in files


# ---------------------------------------------------------------------------
# context_scope wiring
# ---------------------------------------------------------------------------
#
# ``scope.context_paths`` in eval_spec.json is supposed to materialise
# read-only context files into the bundle (and into candidate
# worktrees). Earlier versions stopped at ``Config`` —
# ``build_agent_bundle`` never forwarded ``cfg.context_scope`` to
# ``AgentBundle.from_entry_point``'s ``prefetched_files`` slot. These
# tests pin the end-to-end behaviour so the field stays live.


class TestContextScopeWiring:
    def test_single_file_in_context_scope_is_bundled(self, project: Path):
        (project / "policies.md").write_text("# Policies\nbe nice\n")
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["policies.md"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "policies.md" in bundle.original_files
        assert bundle.original_files["policies.md"].startswith("# Policies")

    def test_context_scope_files_are_not_optimizable(self, project: Path):
        """``context_paths`` is advisory: include the file, but candidates
        shouldn't see it as editable."""
        (project / "policies.md").write_text("rules\n")
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["policies.md"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "policies.md" not in bundle.optimizable_files

    def test_glob_pattern_expanded(self, project: Path):
        (project / "docs").mkdir()
        (project / "docs" / "a.md").write_text("A\n")
        (project / "docs" / "b.md").write_text("B\n")
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["docs/*.md"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "docs/a.md" in bundle.original_files
        assert "docs/b.md" in bundle.original_files

    def test_context_scope_combines_with_optimizable(self, project: Path):
        """The two scopes are independent: context files appear alongside
        the optimizable closure without displacing anything."""
        (project / "policies.md").write_text("rules\n")
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["policies.md"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "entrypoint.py" in bundle.original_files  # entry survives
        assert "agent_logic.py" in bundle.original_files  # optimizable survives
        assert "policies.md" in bundle.original_files  # context added

    def test_empty_context_scope_default(self, project: Path):
        cfg = _make_config(project, optimizable_scope=["agent_logic.py"])
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        # Nothing exotic: just the BFS closure + entry + optimizable.
        assert {"entrypoint.py", "helper.py", "agent_logic.py"} <= set(
            bundle.original_files
        )

    def test_missing_pattern_does_not_crash(self, project: Path):
        """A typo'd or stale glob shouldn't break bundle construction."""
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["does/not/exist/*.md"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        # Nothing added, nothing exploded.
        assert all(
            not p.startswith("does/not/exist") for p in bundle.original_files
        )

    def test_context_scope_does_not_override_bfs_resolved_file(
        self, project: Path
    ):
        """If a file is already in the bundle via BFS, listing it in
        context_scope is a no-op — the BFS source wins. This guards
        against accidental clobbering when authors over-declare scope."""
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            context_scope=["helper.py"],  # already in BFS closure
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "helper.py" in bundle.original_files
        # Optimizability is decided by the BFS path, not context_scope.
        # helper.py is reached from the entry and not in read_only_scope,
        # so it remains in optimizable_files when optimizable_paths covers
        # it; here only agent_logic.py is optimizable.
        assert "helper.py" not in bundle.optimizable_files


# ---------------------------------------------------------------------------
# Intermediate __init__.py bundling
# ---------------------------------------------------------------------------
#
# When BFS resolved ``pkg.subpkg.module`` it landed on ``module.py``
# and stopped. Real-world packages put meaningful code in
# ``__init__.py`` files (re-exports, version constants, side-effecting
# setup like dotenv loading). Pre-Patch-4b those files never entered
# the bundle, so candidate worktrees ran against a different package
# than the user's baseline. These tests pin the corrected behaviour
# without regressing PEP 420 namespace package support.


class TestInitFilesBundled:
    def test_intermediate_init_is_bundled(self, tmp_path: Path):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            from pkg.subpkg.module import helper

            def run(input_data):
                return {"result": helper(input_data)}
            """)
        )
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("VERSION = '1.0'\n")
        sub = pkg / "subpkg"
        sub.mkdir()
        (sub / "__init__.py").write_text("from .module import helper\n")
        (sub / "module.py").write_text("def helper(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "pkg/subpkg/module.py" in files
        assert "pkg/__init__.py" in files
        assert "pkg/subpkg/__init__.py" in files
        assert bundle.original_files["pkg/__init__.py"] == "VERSION = '1.0'\n"

    def test_namespace_package_no_regression(self, tmp_path: Path):
        """PEP 420 namespace packages have no ``__init__.py``. The
        helper only returns files that exist, so namespace layouts
        continue to work — no init files appear in the bundle, but
        the leaf module does."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            "from nspkg.module import helper\n"
            "def run(x): return helper(x)\n"
        )
        ns = tmp_path / "nspkg"
        ns.mkdir()
        # Deliberately no __init__.py.
        (ns / "module.py").write_text("def helper(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "nspkg/module.py" in files
        assert "nspkg/__init__.py" not in files  # didn't exist; not invented

    def test_root_level_init_not_pulled_in(self, tmp_path: Path):
        """A stray ``__init__.py`` directly at the project root must
        not be bundled. It isn't on any package's import path and
        pulling it in surprises users whose repo root just happens to
        have one (e.g. editable installs)."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "__init__.py").write_text("# repo-root init, unrelated\n")
        (tmp_path / "entry.py").write_text(
            "from pkg.module import helper\n"
            "def run(x): return helper(x)\n"
        )
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("PKG = 'real'\n")
        (pkg / "module.py").write_text("def helper(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "pkg/__init__.py" in files
        assert "__init__.py" not in files  # the root-level one

    def test_partial_init_chain(self, tmp_path: Path):
        """If only the inner package has an ``__init__.py`` (e.g. a
        namespace parent + regular child), only the existing one is
        bundled — no fabrication."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            "from outer.inner.mod import helper\n"
            "def run(x): return helper(x)\n"
        )
        outer = tmp_path / "outer"
        outer.mkdir()
        # No outer/__init__.py — namespace.
        inner = outer / "inner"
        inner.mkdir()
        (inner / "__init__.py").write_text("REAL = True\n")
        (inner / "mod.py").write_text("def helper(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "outer/inner/mod.py" in files
        assert "outer/inner/__init__.py" in files
        assert "outer/__init__.py" not in files


# ---------------------------------------------------------------------------
# bundle_search_paths plumbing
# ---------------------------------------------------------------------------
#
# The resolver only looks at the importing file's directory and the
# project root by default. Real-world layouts that aren't importable
# as Python packages by name — hyphenated dirs (``python-backend/``),
# explicit ``src/`` layouts, or dirs declared via
# ``[tool.setuptools.package-dir]`` — were therefore invisible. These
# tests pin the new sys.path-style ``search_paths`` extension and the
# auto-discovery hooks.


class TestSearchPaths:
    def test_hyphenated_dir_search_path_makes_package_resolvable(
        self, tmp_path: Path
    ):
        """The defining airline-style bug: ``python-backend/`` can't be
        imported by name (hyphen), but adding it as a search path makes
        ``airline.*`` resolvable from a top-level entry."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            "from airline.agents import triage\n"
            "def run(x): return triage(x)\n"
        )
        backend = tmp_path / "python-backend"
        backend.mkdir()
        airline = backend / "airline"
        airline.mkdir()
        (airline / "__init__.py").write_text("")
        (airline / "agents.py").write_text("def triage(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
            search_paths=["python-backend"],
        )
        files = set(bundle.original_files)
        assert "python-backend/airline/agents.py" in files
        assert "python-backend/airline/__init__.py" in files

    def test_src_layout_autodiscovered(self, tmp_path: Path):
        """``src/`` is a standard convention; auto-discovery picks it up
        without requiring users to declare it in the spec."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            "from mypkg.api import handle\n"
            "def run(x): return handle(x)\n"
        )
        src = tmp_path / "src"
        src.mkdir()
        pkg = src / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "api.py").write_text("def handle(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
            # Note: no explicit search_paths — autodiscovery handles src/.
        )
        files = set(bundle.original_files)
        assert "src/mypkg/api.py" in files

    def test_pyproject_package_dir_autodiscovered(self, tmp_path: Path):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text(
            "[tool.setuptools]\n"
            "package-dir = { '' = 'libs' }\n"
        )
        (tmp_path / "entry.py").write_text(
            "from pkg.api import handle\n"
            "def run(x): return handle(x)\n"
        )
        libs = tmp_path / "libs"
        libs.mkdir()
        pkg = libs / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "api.py").write_text("def handle(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "libs/pkg/api.py" in files

    def test_explicit_search_paths_override_autodiscovery(
        self, tmp_path: Path
    ):
        """Passing ``search_paths=[]`` is a way to opt out of autodiscovery;
        passing a non-empty list takes effect without merging in
        autodiscovered ones."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        # Construct a layout where the BFS would find the helper via the
        # explicit path only (not via autodiscovery), so we can prove
        # the explicit list is what's in effect.
        (tmp_path / "entry.py").write_text(
            "from helper import compute\n"
            "def run(x): return compute(x)\n"
        )
        wantdir = tmp_path / "want"
        wantdir.mkdir()
        (wantdir / "helper.py").write_text("def compute(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
            search_paths=["want"],
        )
        assert "want/helper.py" in bundle.original_files

    def test_search_path_outside_project_root_is_rejected(
        self, tmp_path: Path
    ):
        """A search path that escapes the project root must not be
        honored — bundling files outside the repo would couple the
        candidate worktree to absolute filesystem state."""
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / "leaky.py").write_text("def x(): return 1\n")

        project = tmp_path / "project"
        project.mkdir()
        (project / OVERMIND_DIR_NAME).mkdir()
        (project / "entry.py").write_text(
            "from leaky import x\ndef run(_): return x()\n"
        )

        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entry.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=None,
            search_paths=["../outer"],
        )
        # leaky.py is outside the project root; the resolver refuses to
        # follow the escape and the import simply isn't classified as
        # local. The bundle contains only the entry.
        assert "leaky.py" not in bundle.original_files
        assert "../outer/leaky.py" not in bundle.original_files

    def test_search_paths_via_eval_spec_scope(self, project: Path):
        """End-to-end: ``scope.search_paths`` from eval_spec is plumbed
        through Config and into the bundle resolver."""
        backend = project / "python-backend"
        backend.mkdir()
        airline = backend / "airline"
        airline.mkdir()
        (airline / "__init__.py").write_text("")
        (airline / "agents.py").write_text("def triage(x): return x\n")
        (project / "entrypoint.py").write_text(
            "from airline.agents import triage\n"
            "def run(x): return triage(x)\n"
        )

        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            bundle_search_paths=["python-backend"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is not None
        assert "python-backend/airline/agents.py" in bundle.original_files


# ---------------------------------------------------------------------------
# Refuse to ignore the entry file
# ---------------------------------------------------------------------------
#
# Listing the entry file in ``exclude_paths`` used to collapse the
# bundle to single-file mode silently. The fix raises
# ``BundleConfigError`` with guidance to use ``read_only_paths``
# instead. ``build_agent_bundle`` catches it and returns ``None`` so
# the optimizer logs a clear error rather than crashing.


class TestEntryIgnoredRejected:
    def test_resolve_local_files_raises_when_entry_ignored(self, project: Path):
        with pytest.raises(BundleConfigError, match="entrypoint.py"):
            resolve_local_files(
                str(project / "entrypoint.py"),
                str(project),
                should_ignore_rel=lambda rel: rel == "entrypoint.py",
            )

    def test_error_message_points_at_read_only_paths(self, project: Path):
        """The message must steer the user toward the right fix
        (read_only_paths) rather than just saying "can't ignore"."""
        with pytest.raises(BundleConfigError, match="read_only_paths"):
            resolve_local_files(
                str(project / "entrypoint.py"),
                str(project),
                should_ignore_rel=lambda rel: rel == "entrypoint.py",
            )

    def test_build_agent_bundle_returns_none_on_entry_ignore(
        self, project: Path, caplog
    ):
        cfg = _make_config(
            project,
            optimizable_scope=["agent_logic.py"],
            exclude_scope=["entrypoint.py"],
        )
        bundle = build_agent_bundle(cfg)
        assert bundle is None

    def test_non_entry_ignored_path_still_works(self, project: Path):
        """Ignoring a non-entry file (e.g. helper.py) is fine — only
        the entry is special."""
        # Should not raise:
        bundle = AgentBundle.from_entry_point(
            entry_path=str(project / "entrypoint.py"),
            project_root=str(project),
            entrypoint_fn="run",
            optimizable_paths=None,
            should_ignore_rel=lambda rel: rel == "helper.py",
        )
        assert "entrypoint.py" in bundle.original_files
        assert "helper.py" not in bundle.original_files


# ---------------------------------------------------------------------------
# __overmind_imports__ static hint for dynamic imports
# ---------------------------------------------------------------------------
#
# Dynamic imports (``importlib.import_module``, plugin loaders, lazy
# proxies) are invisible to the static BFS. The lightweight static
# substitute is a module-level ``__overmind_imports__`` list. Authors
# of dynamic-import shims declare it; the BFS treats the names as
# ordinary imports. These tests pin the parsing and end-to-end
# bundling behaviour.


class TestStaticImportHint:
    def test_overmind_imports_list_is_followed(self, tmp_path: Path):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            import importlib

            __overmind_imports__ = ["airline.agents", "airline.tools"]

            def run(input_data):
                mod = importlib.import_module(input_data["mod"])
                return {"result": mod.execute(input_data)}
            """)
        )
        airline = tmp_path / "airline"
        airline.mkdir()
        (airline / "__init__.py").write_text("")
        (airline / "agents.py").write_text(
            "def execute(x): return x\n"
        )
        (airline / "tools.py").write_text(
            "def execute(x): return x\n"
        )

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        files = set(bundle.original_files)
        assert "airline/agents.py" in files
        assert "airline/tools.py" in files

    def test_overmind_imports_tuple_also_accepted(self, tmp_path: Path):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            'import importlib\n'
            '__overmind_imports__ = ("helpers.compute",)\n'
            'def run(x): return x\n'
        )
        helpers = tmp_path / "helpers"
        helpers.mkdir()
        (helpers / "__init__.py").write_text("")
        (helpers / "compute.py").write_text("def f(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        assert "helpers/compute.py" in bundle.original_files

    def test_non_string_entries_are_ignored(self, tmp_path: Path):
        """Computed entries (variables, function calls) can't be safely
        evaluated during static analysis — they must be silently
        skipped, not crash the BFS."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            import os
            _name = "airline.agents"
            __overmind_imports__ = [_name, "real.module", os.environ.get("X")]
            def run(x): return x
            """)
        )
        real = tmp_path / "real"
        real.mkdir()
        (real / "__init__.py").write_text("")
        (real / "module.py").write_text("def f(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        # Only the string-literal entry is followed.
        files = set(bundle.original_files)
        assert "real/module.py" in files

    def test_missing_module_in_hint_is_silently_skipped(
        self, tmp_path: Path
    ):
        """A typo'd or removed module in __overmind_imports__ shouldn't
        crash bundling — same forgiveness as a missing scope glob."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            '__overmind_imports__ = ["does.not.exist"]\n'
            'def run(x): return x\n'
        )
        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
        )
        # Only the entry made it in; nothing exploded.
        assert "entry.py" in bundle.original_files


# ---------------------------------------------------------------------------
# Entry-derived sys.path search paths
# ---------------------------------------------------------------------------
#
# When the entry file mutates ``sys.path`` at module top, the runtime
# imports resolve against the inserted directories. The bundle BFS
# needs the same view or every cross-tree import is misclassified as
# external (and the analyzer ends up with a single-file bundle). These
# tests pin both the AST partial evaluator (``_detect_entry_search_paths``)
# and its integration into ``resolve_local_files`` / ``from_entry_point``.


class TestDetectEntrySearchPaths:
    """Unit tests for the AST partial evaluator. Each test crafts an
    entry file with one specific shape of sys.path mutation and asserts
    the resolver discovers the intended directory.
    """

    def _make_root(self, tmp_path: Path, *subdirs: str) -> Path:
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True, exist_ok=True)
        for sub in subdirs:
            (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        return tmp_path

    def test_string_literal_insert(self, tmp_path: Path):
        root = self._make_root(tmp_path, "python-backend")
        (root / "entry.py").write_text(
            'import sys\n'
            'sys.path.insert(0, "python-backend")\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == [(root / "python-backend").resolve()]

    def test_path_dunder_file_parent_div(self, tmp_path: Path):
        root = self._make_root(tmp_path, "python-backend")
        (root / "entry.py").write_text(
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, Path(__file__).parent / "python-backend")\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == [(root / "python-backend").resolve()]

    def test_airline_style_with_constant_and_resolve(self, tmp_path: Path):
        """Module-level constant assigned with chained .resolve().parent
        is captured and reused by the sys.path call — this is the exact
        shape the airline-customer-service entrypoint uses."""
        root = self._make_root(tmp_path, "python-backend")
        (root / "entry.py").write_text(
            'import sys\n'
            'from pathlib import Path\n'
            '_PROJECT_ROOT = Path(__file__).resolve().parent\n'
            '_BACKEND_DIR = _PROJECT_ROOT / "python-backend"\n'
            'if str(_BACKEND_DIR) not in sys.path:\n'
            '    sys.path.insert(0, str(_BACKEND_DIR))\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == [(root / "python-backend").resolve()]

    def test_os_path_join_with_dirname(self, tmp_path: Path):
        root = self._make_root(tmp_path, "backend")
        (root / "entry.py").write_text(
            'import os, sys\n'
            'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == [(root / "backend").resolve()]

    def test_multiple_inserts_preserve_order(self, tmp_path: Path):
        root = self._make_root(tmp_path, "first", "second")
        (root / "entry.py").write_text(
            'import sys\n'
            'sys.path.insert(0, "first")\n'
            'sys.path.append("second")\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == [
            (root / "first").resolve(),
            (root / "second").resolve(),
        ]

    def test_path_outside_project_root_is_dropped(self, tmp_path: Path):
        """A user who points sys.path at /etc or a sibling repo would
        otherwise leak files into the bundle. The detector refuses any
        path that doesn't strictly live under project_root."""
        outer = tmp_path / "outer"
        outer.mkdir()
        project = tmp_path / "project"
        (project / OVERMIND_DIR_NAME).mkdir(parents=True)
        (project / "entry.py").write_text(
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).parent.parent / "outer"))\n'
        )
        detected = _detect_entry_search_paths(project / "entry.py", project)
        assert detected == []

    def test_nonexistent_directory_is_dropped(self, tmp_path: Path):
        root = self._make_root(tmp_path)  # No python-backend created.
        (root / "entry.py").write_text(
            'import sys\n'
            'sys.path.insert(0, "python-backend")\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        assert detected == []

    def test_unsupported_expression_silently_skipped(self, tmp_path: Path):
        """f-strings, env lookups, function calls into user code — any
        runtime-dependent expression yields None and is dropped.
        Nothing else in the file should crash bundling."""
        root = self._make_root(tmp_path, "static-dir")
        (root / "entry.py").write_text(
            'import os, sys\n'
            'sys.path.insert(0, os.environ["MY_DIR"])\n'
            'sys.path.insert(0, f"prefix-{1+1}")\n'
            'sys.path.insert(0, "static-dir")\n'
        )
        detected = _detect_entry_search_paths(root / "entry.py", root)
        # Only the literal "static-dir" survives; the other two yield None.
        assert detected == [(root / "static-dir").resolve()]

    def test_no_syspath_mutations_returns_empty(self, tmp_path: Path):
        root = self._make_root(tmp_path)
        (root / "entry.py").write_text(
            'def run(x):\n    return x\n'
        )
        assert _detect_entry_search_paths(root / "entry.py", root) == []


class TestEntryDerivedSearchPathsIntegration:
    """End-to-end: a multi-file project whose layout is only knowable
    via the entry's sys.path mutation. Without ``_detect_entry_search_paths``
    wiring the bundle would be single-file; with it, BFS reaches the
    real package.
    """

    def test_bundle_picks_up_entry_syspath_without_explicit_declaration(
        self, tmp_path: Path
    ):
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            import sys
            from pathlib import Path

            _BACKEND = Path(__file__).resolve().parent / "py-backend"
            if str(_BACKEND) not in sys.path:
                sys.path.insert(0, str(_BACKEND))

            from pkg.mod import compute

            def run(x):
                return {"result": compute(x)}
            """)
        )
        backend = tmp_path / "py-backend"
        (backend / "pkg").mkdir(parents=True)
        (backend / "pkg" / "__init__.py").write_text("")
        (backend / "pkg" / "mod.py").write_text("def compute(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
            # NO search_paths argument — auto-discovery must find it.
        )
        files = set(bundle.original_files)
        assert "entry.py" in files
        assert "py-backend/pkg/mod.py" in files
        assert "py-backend/pkg/__init__.py" in files

    def test_explicit_empty_search_paths_disables_entry_detection(
        self, tmp_path: Path
    ):
        """``search_paths=[]`` is the documented escape hatch; it must
        also opt out of entry-derived detection so users keep full
        control when they ask for it."""
        (tmp_path / OVERMIND_DIR_NAME).mkdir(parents=True)
        (tmp_path / "entry.py").write_text(
            textwrap.dedent("""\
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent / "py-backend"))
            from pkg.mod import compute

            def run(x):
                return compute(x)
            """)
        )
        backend = tmp_path / "py-backend"
        (backend / "pkg").mkdir(parents=True)
        (backend / "pkg" / "__init__.py").write_text("")
        (backend / "pkg" / "mod.py").write_text("def compute(x): return x\n")

        bundle = AgentBundle.from_entry_point(
            entry_path=str(tmp_path / "entry.py"),
            project_root=str(tmp_path),
            entrypoint_fn="run",
            optimizable_paths=None,
            search_paths=[],  # explicit opt-out
        )
        # Entry only; the explicit empty list overrides auto-discovery.
        assert "py-backend/pkg/mod.py" not in bundle.original_files
