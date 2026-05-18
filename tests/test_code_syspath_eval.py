"""Tests for :mod:`overmind.code.syspath_eval`.

These complement ``tests/test_bundle_factory.py`` — which exercises
``detect_entry_search_paths`` end-to-end via real entry files — by
pinning the contract of the pure :func:`eval_path_expr` helper directly,
so future refactors of the AST grammar produce immediate, focused test
failures.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from overmind.code.syspath_eval import detect_entry_search_paths, eval_path_expr


def _expr_node(source: str) -> ast.AST:
    """Parse a single expression and return its AST node."""
    return ast.parse(source, mode="eval").body


@pytest.fixture
def fake_entry(tmp_path: Path) -> Path:
    f = tmp_path / "pkg" / "entry.py"
    f.parent.mkdir(parents=True)
    f.write_text("")
    return f


class TestEvalPathExprConstants:
    def test_string_literal_is_resolved_relative_to_file(self, fake_entry: Path):
        node = _expr_node("'lib'")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parent / "lib"

    def test_absolute_string_literal_kept_as_is(self, fake_entry: Path):
        node = _expr_node("'/abs/lib'")
        assert eval_path_expr(node, file=fake_entry, constants={}) == Path("/abs/lib")

    def test_non_string_literal_returns_none(self, fake_entry: Path):
        assert eval_path_expr(_expr_node("42"), file=fake_entry, constants={}) is None

    def test_file_name_returns_entry_path(self, fake_entry: Path):
        node = _expr_node("__file__")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry

    def test_unknown_name_returns_none(self, fake_entry: Path):
        assert eval_path_expr(_expr_node("unknown"), file=fake_entry, constants={}) is None

    def test_known_constant_is_returned(self, fake_entry: Path):
        target = Path("/some/lib")
        assert (
            eval_path_expr(_expr_node("ROOT"), file=fake_entry, constants={"ROOT": target}) == target
        )


class TestEvalPathExprComposition:
    def test_path_division_with_string(self, fake_entry: Path):
        node = _expr_node("__file__ / 'sub'")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry / "sub"

    def test_path_division_with_non_string_returns_none(self, fake_entry: Path):
        node = _expr_node("__file__ / 42")
        assert eval_path_expr(node, file=fake_entry, constants={}) is None

    def test_parent_attribute(self, fake_entry: Path):
        node = _expr_node("__file__.parent")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parent

    def test_parents_subscript(self, fake_entry: Path):
        node = _expr_node("__file__.parents[1]")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parents[1]

    def test_parents_out_of_range_returns_none(self, fake_entry: Path):
        node = _expr_node("__file__.parents[99]")
        assert eval_path_expr(node, file=fake_entry, constants={}) is None

    def test_path_wrapper_unwraps(self, fake_entry: Path):
        node = _expr_node("Path(__file__)")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry

    def test_str_wrapper_unwraps(self, fake_entry: Path):
        node = _expr_node("str(__file__)")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry

    def test_resolve_call_collapses_to_base(self, fake_entry: Path):
        node = _expr_node("Path(__file__).resolve()")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry


class TestEvalPathExprOsPath:
    def test_join_with_strings(self, fake_entry: Path):
        node = _expr_node("os.path.join(str(Path(__file__).parent), 'a', 'b')")
        assert (
            eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parent / "a" / "b"
        )

    def test_join_with_non_string_arg_returns_none(self, fake_entry: Path):
        node = _expr_node("os.path.join(str(Path(__file__).parent), variable)")
        assert eval_path_expr(node, file=fake_entry, constants={}) is None

    def test_dirname_returns_parent(self, fake_entry: Path):
        node = _expr_node("os.path.dirname(str(Path(__file__)))")
        assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parent

    def test_abspath_realpath_collapse(self, fake_entry: Path):
        for fn in ("abspath", "realpath"):
            node = _expr_node(f"os.path.{fn}(str(Path(__file__).parent))")
            assert eval_path_expr(node, file=fake_entry, constants={}) == fake_entry.parent


class TestDetectEntrySearchPaths:
    def test_returns_empty_when_no_mutations(self, tmp_path: Path):
        entry = tmp_path / "entry.py"
        entry.write_text("import os\nprint(1)\n")
        assert detect_entry_search_paths(entry, tmp_path) == []

    def test_returns_empty_on_syntax_error(self, tmp_path: Path):
        entry = tmp_path / "entry.py"
        entry.write_text("def broken(:\n")
        assert detect_entry_search_paths(entry, tmp_path) == []

    def test_returns_empty_when_path_outside_root(self, tmp_path: Path):
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, '/outside-root/lib')
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == []

    def test_inserts_relative_to_entry(self, tmp_path: Path):
        (tmp_path / "lib").mkdir()
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, str(Path(__file__).parent / 'lib'))
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == [(tmp_path / "lib").resolve()]

    def test_dedups_repeated_inserts(self, tmp_path: Path):
        (tmp_path / "lib").mkdir()
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, str(Path(__file__).parent / 'lib'))
                sys.path.append(str(Path(__file__).parent / 'lib'))
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == [(tmp_path / "lib").resolve()]

    def test_handles_extend_with_list_literal(self, tmp_path: Path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.extend([
                    str(Path(__file__).parent / 'a'),
                    str(Path(__file__).parent / 'b'),
                ])
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == [
            (tmp_path / "a").resolve(),
            (tmp_path / "b").resolve(),
        ]

    def test_walks_into_if_bodies(self, tmp_path: Path):
        (tmp_path / "lib").mkdir()
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                if True:
                    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == [(tmp_path / "lib").resolve()]

    def test_uses_module_constant_bindings(self, tmp_path: Path):
        (tmp_path / "lib").mkdir()
        entry = tmp_path / "entry.py"
        entry.write_text(
            textwrap.dedent(
                """
                import sys
                ROOT = Path(__file__).parent
                sys.path.insert(0, str(ROOT / 'lib'))
                """
            )
        )
        assert detect_entry_search_paths(entry, tmp_path) == [(tmp_path / "lib").resolve()]
