"""Sub-package for static analysis helpers backing the agent bundler.

``overmind/utils/code.py`` is the main entry point (``CodePiece``,
``AgentBundle``, ``resolve_local_files``, …).  This sub-package houses
focused helpers that the bundler composes — extracted so they can be
tested in isolation without pulling the whole BFS machinery into scope.

Submodules
----------
:mod:`overmind.code.syspath_eval`
    AST-based partial evaluator for ``sys.path`` mutations at module top.
"""

from overmind.code.syspath_eval import (
    detect_entry_search_paths,
    eval_path_expr,
)

__all__ = ["detect_entry_search_paths", "eval_path_expr"]
