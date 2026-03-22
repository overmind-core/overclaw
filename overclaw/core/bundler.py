"""
Agent code bundler for multi-file optimization.

Resolves all project-local code reachable from an agent's entry file,
stores complete file sources, and provides whole-file replacement logic
to apply targeted updates back to original files.

The bundler bridges the gap between multi-file agent codebases and
the single-prompt optimization loop.  It produces a compact virtual
representation of only the code the agent actually uses, tagged with
origin information, and maps LLM-generated updates back into the
original file tree.

Usage::

    bundle = AgentBundle.from_entry_point(
        entry_path="agents/my_agent/agent.py",
        project_root="/path/to/project",
        entrypoint_fn="run",
    )

    # Render for LLM prompt
    prompt_text = bundle.to_prompt_text()

    # After LLM produces updated files, apply them
    modified_files = bundle.apply_file_updates(file_updates)
"""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Code piece representation (internal analytics — not used in prompts)
# ---------------------------------------------------------------------------


@dataclass
class CodePiece:
    """A single extractable unit of code with its origin metadata."""

    piece_id: str
    file_path: str
    symbol_name: str
    symbol_type: str  # "imports" | "constant" | "function" | "class"
    source: str
    optimizable: bool
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    base_indent: int = 0


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _node_name(node: ast.AST) -> str | None:
    """Return the top-level name bound by *node*, or ``None``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _node_start_line(node: ast.AST) -> int:
    """First line of *node*, accounting for decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _detect_base_indent(source_lines: list[str], start: int) -> int:
    """Detect the indentation level of the first non-empty line."""
    for line in source_lines[start:]:
        stripped = line.lstrip()
        if stripped:
            return len(line) - len(stripped)
    return 0


def _source_segment(lines: list[str], start: int, end: int) -> str:
    """Extract source from *lines* (0-indexed start, 0-indexed exclusive end)."""
    return "".join(lines[start:end])


def _names_referenced_in(node: ast.AST) -> set[str]:
    """Collect all ``Name`` identifiers referenced inside *node*."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            root = child
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
    return names


def has_entrypoint_ast(source: str, fn_name: str) -> bool:
    """Check via AST whether *source* defines a top-level function *fn_name*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == fn_name:
                return True
    return False


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

_STDLIB_TOP: frozenset[str] = getattr(
    sys, "stdlib_module_names", frozenset()
) | frozenset(sys.builtin_module_names)


def _is_local_module(module_name: str, project_root: Path) -> bool:
    """Return True if *module_name* likely resolves to a project-local file."""
    top = module_name.split(".")[0]
    if top in _STDLIB_TOP:
        return False
    candidate = project_root / top
    return candidate.exists() or (candidate.with_suffix(".py")).exists()


def _resolve_module_to_file(
    module_name: str,
    from_file: Path,
    project_root: Path,
) -> Path | None:
    """Try to resolve a dotted module to a ``.py`` file under *project_root*."""
    parts = module_name.split(".")

    bases = [project_root]
    pkg_dir = from_file.parent
    if pkg_dir != project_root:
        bases.insert(0, pkg_dir)

    for base in bases:
        candidate = base / "/".join(parts)
        py_path = candidate.with_suffix(".py")
        if py_path.exists():
            try:
                py_path.relative_to(project_root)
            except ValueError:
                continue
            return py_path
        init_path = candidate / "__init__.py"
        if init_path.exists():
            try:
                init_path.relative_to(project_root)
            except ValueError:
                continue
            return init_path

    return None


def _collect_import_targets(source: str) -> list[str]:
    """Return dotted module names from all import statements in *source*.

    Handles both absolute imports and relative imports (``from . import x``).
    For relative imports without an explicit module (level-only), the
    importing file's package is used as the base.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    targets: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                targets.append(node.module)
            elif node.level and node.level > 0 and node.names:
                for alias in node.names:
                    targets.append(alias.name)
    return targets


def _resolve_relative_import(
    node: ast.ImportFrom,
    from_file: Path,
    project_root: Path,
) -> list[Path]:
    """Resolve a relative ImportFrom node to concrete file paths."""
    results: list[Path] = []
    pkg_dir = from_file.parent

    for _ in range(max(0, (node.level or 0) - 1)):
        pkg_dir = pkg_dir.parent

    if node.module:
        parts = node.module.split(".")
        candidate = pkg_dir / "/".join(parts)
        py_path = candidate.with_suffix(".py")
        if py_path.exists():
            try:
                py_path.relative_to(project_root)
                results.append(py_path)
            except ValueError:
                pass
        init_path = candidate / "__init__.py"
        if init_path.exists():
            try:
                init_path.relative_to(project_root)
                results.append(init_path)
            except ValueError:
                pass
    else:
        for alias in node.names:
            candidate = pkg_dir / alias.name
            py_path = candidate.with_suffix(".py")
            if py_path.exists():
                try:
                    py_path.relative_to(project_root)
                    results.append(py_path)
                except ValueError:
                    pass
            init_path = candidate / "__init__.py"
            if init_path.exists():
                try:
                    init_path.relative_to(project_root)
                    results.append(init_path)
                except ValueError:
                    pass

    return results


def resolve_local_files(
    entry_path: str,
    project_root: str,
    *,
    max_depth: int = 6,
) -> dict[str, str]:
    """Recursively resolve all project-local files reachable from *entry_path*.

    Returns ``{relative_path: source_code}`` with the entry file first.
    """
    root = Path(project_root).resolve()
    entry = Path(entry_path).resolve()
    result: dict[str, str] = {}
    visited: set[Path] = set()

    def _walk(file_path: Path, depth: int) -> None:
        if depth > max_depth or file_path in visited:
            return
        if not file_path.exists() or not file_path.is_file():
            return
        try:
            file_path.relative_to(root)
        except ValueError:
            return

        visited.add(file_path)
        source = file_path.read_text(encoding="utf-8")
        rel = str(file_path.relative_to(root))
        result[rel] = source

        # Resolve absolute imports
        for mod_name in _collect_import_targets(source):
            if not _is_local_module(mod_name, root):
                continue
            resolved = _resolve_module_to_file(mod_name, file_path, root)
            if resolved:
                _walk(resolved, depth + 1)

        # Resolve relative imports via AST
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                for resolved in _resolve_relative_import(node, file_path, root):
                    _walk(resolved, depth + 1)

    _walk(entry, 0)
    return result


# ---------------------------------------------------------------------------
# Piece extraction (internal — used for analytics, not for prompts)
# ---------------------------------------------------------------------------


def _extract_import_block(source: str, tree: ast.Module) -> tuple[str, int, int] | None:
    """Extract the contiguous import block at the top of a module.

    Returns ``(source_text, start_line_1indexed, end_line_1indexed)`` or None.
    """
    lines = source.splitlines(keepends=True)
    import_nodes = [
        n
        for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    if not import_nodes:
        return None

    start = import_nodes[0].lineno
    end = import_nodes[-1].end_lineno or import_nodes[-1].lineno
    return _source_segment(lines, start - 1, end), start, end


def extract_pieces(
    rel_path: str,
    source: str,
    *,
    optimizable: bool = True,
    used_names: set[str] | None = None,
) -> list[CodePiece]:
    """Extract top-level code pieces from *source*.

    If *used_names* is provided, only pieces whose symbol name is in the set
    (or that are referenced by those pieces) are included.  When ``None``,
    all top-level symbols are extracted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name="__full_file__",
                symbol_type="constant",
                source=source,
                optimizable=optimizable,
                line_start=1,
                line_end=source.count("\n") + 1,
            )
        ]

    lines = source.splitlines(keepends=True)
    pieces: list[CodePiece] = []

    # --- Imports (always included) ---
    imp = _extract_import_block(source, tree)
    if imp:
        imp_src, imp_start, imp_end = imp
        pieces.append(
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name="__imports__",
                symbol_type="imports",
                source=imp_src.rstrip("\n"),
                optimizable=optimizable,
                line_start=imp_start,
                line_end=imp_end,
            )
        )

    # --- Build a map of all top-level definitions ---
    top_level_nodes: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        name = _node_name(node)
        if name:
            top_level_nodes.append((name, node))

    # --- Resolve which names to include ---
    if used_names is not None:
        included = set(used_names)
        node_map = {name: node for name, node in top_level_nodes}
        changed = True
        while changed:
            changed = False
            for name in list(included):
                if name not in node_map:
                    continue
                refs = _names_referenced_in(node_map[name])
                for ref in refs:
                    if ref in node_map and ref not in included:
                        included.add(ref)
                        changed = True
    else:
        included = {name for name, _ in top_level_nodes}

    # --- Extract each included symbol ---
    for name, node in top_level_nodes:
        if name not in included:
            continue

        start_line = _node_start_line(node)
        end_line = node.end_lineno or start_line
        seg = _source_segment(lines, start_line - 1, end_line)
        indent = _detect_base_indent(lines, start_line - 1)

        sym_type = "constant"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym_type = "function"
        elif isinstance(node, ast.ClassDef):
            sym_type = "class"

        pieces.append(
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name=name,
                symbol_type=sym_type,
                source=seg.rstrip("\n"),
                optimizable=optimizable,
                line_start=start_line,
                line_end=end_line,
                base_indent=indent,
            )
        )

    return pieces


# ---------------------------------------------------------------------------
# Agent bundle
# ---------------------------------------------------------------------------


@dataclass
class AgentBundle:
    """Virtual representation of a multi-file agent for the optimization prompt.

    Holds original file sources tagged with optimizability.  The LLM sees
    complete files (not fragments) and returns complete updated files.
    """

    entry_file: str  # relative path to entry point
    entry_function: str
    pieces: list[CodePiece] = field(default_factory=list)
    original_files: dict[str, str] = field(default_factory=dict)
    project_root: str = ""
    optimizable_files: set[str] = field(default_factory=set)

    # --- Construction ---------------------------------------------------

    @classmethod
    def from_entry_point(
        cls,
        entry_path: str,
        project_root: str,
        entrypoint_fn: str,
        *,
        optimizable_paths: Sequence[str] | None = None,
        max_total_chars: int = 150_000,
    ) -> AgentBundle:
        """Build a bundle by resolving all local dependencies from *entry_path*.

        Parameters
        ----------
        entry_path:
            Absolute path to the agent's entry file.
        project_root:
            Absolute path to the project root.
        entrypoint_fn:
            Name of the entry function the optimizer invokes.
        optimizable_paths:
            Relative paths (under *project_root*) of files the LLM may modify.
            Defaults to ``[entry_file_relative_path]``.
        max_total_chars:
            Token budget expressed as characters.  Read-only files beyond
            this budget are demoted to signature-only representation.
        """
        root = Path(project_root).resolve()
        entry = Path(entry_path).resolve()
        entry_rel = str(entry.relative_to(root))

        local_files = resolve_local_files(entry_path, project_root)

        if optimizable_paths is None:
            opt_set = set(local_files.keys())
        else:
            opt_set = set(optimizable_paths)
            opt_set.add(entry_rel)

        bundle = cls(
            entry_file=entry_rel,
            entry_function=entrypoint_fn,
            original_files=dict(local_files),
            project_root=project_root,
            optimizable_files=set(opt_set),
        )

        # Extract pieces for internal analytics (symbol tracking)
        ordered_paths = [entry_rel] + [p for p in local_files if p != entry_rel]

        total_chars = 0
        for rel_path in ordered_paths:
            source = local_files[rel_path]
            is_opt = rel_path in opt_set

            pieces = extract_pieces(rel_path, source, optimizable=is_opt)

            for p in pieces:
                total_chars += len(p.source)

            if total_chars > max_total_chars and not is_opt:
                sig_pieces = _signatures_only(pieces)
                bundle.pieces.extend(sig_pieces)
            else:
                bundle.pieces.extend(pieces)

        bundle._assign_ids()

        return bundle

    @classmethod
    def from_single_file(
        cls,
        entry_path: str,
        project_root: str,
        entrypoint_fn: str,
    ) -> AgentBundle:
        """Backward-compatible constructor for single-file agents.

        Extracts pieces from the entry file only.
        """
        root = Path(project_root).resolve()
        entry = Path(entry_path).resolve()
        entry_rel = str(entry.relative_to(root))
        source = entry.read_text(encoding="utf-8")

        pieces = extract_pieces(entry_rel, source, optimizable=True)

        bundle = cls(
            entry_file=entry_rel,
            entry_function=entrypoint_fn,
            original_files={entry_rel: source},
            project_root=project_root,
            optimizable_files={entry_rel},
            pieces=pieces,
        )
        bundle._assign_ids()
        return bundle

    # --- ID assignment --------------------------------------------------

    def _assign_ids(self) -> None:
        """Assign positional IDs ``P0``, ``P1``, … to all pieces."""
        for idx, piece in enumerate(self.pieces):
            piece.piece_id = f"P{idx}"

    # --- Prompt rendering -----------------------------------------------

    def to_prompt_text(self) -> str:
        """Render the bundle as whole-file sections for the LLM prompt.

        Each file is shown in full (or signature-only for compressed
        read-only deps), clearly delimited with optimizability tags.
        """
        sections: list[str] = []

        ordered_paths = [self.entry_file] + [
            p for p in self.original_files if p != self.entry_file
        ]

        for rel_path in ordered_paths:
            source = self.original_files.get(rel_path)
            if source is None:
                continue

            is_opt = rel_path in self.optimizable_files
            tag = "OPTIMIZABLE" if is_opt else "READ-ONLY"

            file_pieces = self.pieces_for_file(rel_path)
            has_signature_only = any(
                p.symbol_name == "__signature__" or p.source.rstrip().endswith("...")
                for p in file_pieces
                if p.symbol_type in ("function", "class")
            )

            if has_signature_only and not is_opt:
                sig_text = "\n\n".join(p.source for p in file_pieces)
                sections.append(
                    f"\n# ===== FILE: {rel_path} [{tag}] =====\n"
                    f"```python\n{sig_text}\n```"
                )
            else:
                sections.append(
                    f"\n# ===== FILE: {rel_path} [{tag}] =====\n"
                    f"```python\n{source}\n```"
                )

        return "\n".join(sections)

    def get_entry_code(self) -> str:
        """Return the full original source of the entry file."""
        return self.original_files.get(self.entry_file, "")

    def get_all_optimizable_code(self) -> str:
        """Concatenated source of all optimizable files (for metrics)."""
        return "\n\n".join(
            source
            for rel_path, source in self.original_files.items()
            if rel_path in self.optimizable_files
        )

    def get_optimizable_piece_ids(self) -> list[str]:
        """Return piece IDs of all optimizable pieces."""
        return [p.piece_id for p in self.pieces if p.optimizable]

    # --- Piece lookup ---------------------------------------------------

    def piece_by_id(self, piece_id: str) -> CodePiece | None:
        """Look up a piece by its positional ID."""
        for p in self.pieces:
            if p.piece_id == piece_id:
                return p
        return None

    def pieces_for_file(self, rel_path: str) -> list[CodePiece]:
        """Return all pieces belonging to *rel_path*."""
        return [p for p in self.pieces if p.file_path == rel_path]

    # --- Whole-file update application ----------------------------------

    def apply_file_updates(
        self,
        file_updates: dict[str, str],
    ) -> dict[str, str] | None:
        """Apply whole-file updates to the bundle.

        Parameters
        ----------
        file_updates:
            Mapping of ``{relative_path: complete_new_source}``.

        Returns
        -------
        dict or None
            ``{relative_path: validated_source}`` for files that actually
            changed, or ``None`` if any file has a syntax error.
        """
        modified: dict[str, str] = {}

        for rel_path, new_source in file_updates.items():
            # Only allow updates to optimizable files
            if rel_path not in self.optimizable_files:
                continue

            # Validate syntax
            if rel_path.endswith(".py"):
                try:
                    ast.parse(new_source)
                except SyntaxError:
                    return None

            # Only include if actually different from original
            original = self.original_files.get(rel_path, "")
            if new_source.rstrip() != original.rstrip():
                modified[rel_path] = new_source

        return modified

    def get_full_file_set(
        self,
        updates: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return the complete file set, with optional updates merged in.

        Useful for creating temp directories for validation/execution.
        """
        result = dict(self.original_files)
        if updates:
            result.update(updates)
        return result

    # --- Legacy splice-based update application -------------------------

    def apply_updates(
        self,
        updates: dict[str, str],
        new_pieces: list[tuple[str, str]] | None = None,
    ) -> dict[str, str] | None:
        """Apply piece-level updates to the original files.

        Legacy method kept for backward compatibility.  Prefer
        ``apply_file_updates`` for the whole-file approach.

        Parameters
        ----------
        updates:
            Mapping of ``{piece_id: new_source_code}``.
        new_pieces:
            Optional list of ``(target_file_rel_path, source_code)`` for
            newly created symbols.

        Returns
        -------
        dict or None
            ``{relative_path: updated_source}`` for files that changed, or
            ``None`` if a splice produced invalid Python.
        """
        modified: dict[str, str] = {}
        pieces_by_file: dict[str, list[tuple[CodePiece, str]]] = {}

        for pid, new_code in updates.items():
            piece = self.piece_by_id(pid)
            if piece is None:
                continue
            if not piece.optimizable:
                continue
            pieces_by_file.setdefault(piece.file_path, []).append((piece, new_code))

        for rel_path, piece_updates in pieces_by_file.items():
            source = self.original_files.get(rel_path)
            if source is None:
                continue

            piece_updates.sort(key=lambda x: x[0].line_start, reverse=True)
            for piece, new_code in piece_updates:
                source = splice_piece(source, piece, new_code)

            modified[rel_path] = source

        if new_pieces:
            for target_file, new_code in new_pieces:
                if target_file in modified:
                    modified[target_file] = append_piece(
                        modified[target_file], new_code
                    )
                elif target_file in self.original_files:
                    modified[target_file] = append_piece(
                        self.original_files[target_file], new_code
                    )

        for rel_path, source in modified.items():
            if rel_path.endswith(".py"):
                try:
                    ast.parse(source)
                except SyntaxError:
                    return None

        return modified

    # --- Backward compatibility -----------------------------------------

    def to_single_file_code(self) -> str:
        """If this is a single-file bundle, return entry file source.

        Falls back gracefully for use in places that still expect a
        single code string.
        """
        return self.original_files.get(self.entry_file, "")

    def is_multi_file(self) -> bool:
        """Return True if the bundle spans more than one file."""
        return len(self.original_files) > 1

    def optimizable_file_count(self) -> int:
        """Count distinct files that have optimizable pieces."""
        return len(self.optimizable_files & set(self.original_files.keys()))


# ---------------------------------------------------------------------------
# Splice helpers (kept for legacy apply_updates path)
# ---------------------------------------------------------------------------


def _normalize_indent(code: str, target_indent: int) -> str:
    """Re-indent *code* so its base indentation matches *target_indent* spaces."""
    code_lines = code.splitlines(keepends=True)
    if not code_lines:
        return code

    current_indent = 0
    for line in code_lines:
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            current_indent = len(line) - len(stripped)
            break

    if current_indent == target_indent:
        return code

    delta = target_indent - current_indent
    result_lines: list[str] = []
    for line in code_lines:
        if not line.strip():
            result_lines.append(line)
            continue
        if delta > 0:
            result_lines.append(" " * delta + line)
        else:
            remove = min(abs(delta), len(line) - len(line.lstrip()))
            result_lines.append(line[remove:])

    return "".join(result_lines)


def splice_piece(
    source: str,
    piece: CodePiece,
    new_code: str,
) -> str:
    """Replace *piece* in *source* with *new_code*, preserving surrounding code."""
    lines = source.splitlines(keepends=True)
    normalized = _normalize_indent(new_code, piece.base_indent)
    if not normalized.endswith("\n"):
        normalized += "\n"

    before = lines[: piece.line_start - 1]
    after = lines[piece.line_end :]

    return "".join(before) + normalized + "".join(after)


def append_piece(source: str, new_code: str) -> str:
    """Append *new_code* to the end of *source* with proper spacing."""
    if not source.endswith("\n"):
        source += "\n"
    if not source.endswith("\n\n"):
        source += "\n"
    return source + new_code.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signatures_only(pieces: list[CodePiece]) -> list[CodePiece]:
    """Demote pieces to signature-only versions for context compression."""
    result: list[CodePiece] = []
    for p in pieces:
        if p.symbol_type == "imports":
            result.append(p)
            continue
        if p.symbol_type in ("function", "class"):
            sig = _extract_signature(p.source)
            if sig:
                result.append(
                    CodePiece(
                        piece_id=p.piece_id,
                        file_path=p.file_path,
                        symbol_name=p.symbol_name,
                        symbol_type=p.symbol_type,
                        source=sig,
                        optimizable=p.optimizable,
                        line_start=p.line_start,
                        line_end=p.line_end,
                        base_indent=p.base_indent,
                    )
                )
                continue
        result.append(p)
    return result


def _extract_signature(source: str) -> str | None:
    """Extract the function/class signature + docstring from *source*."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines = source.splitlines()
            sig_end = node.body[0].lineno - 1 if node.body else node.lineno
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            ):
                sig_end = node.body[0].end_lineno or sig_end
            sig_lines = lines[:sig_end]
            sig_lines.append("    ...")
            return "\n".join(sig_lines)

        if isinstance(node, ast.ClassDef):
            lines = source.splitlines()
            result_lines: list[str] = []
            in_class = False
            for i, line in enumerate(lines):
                if not in_class:
                    result_lines.append(line)
                    if line.strip().startswith("class "):
                        in_class = True
                elif in_class:
                    stripped = line.strip()
                    if stripped.startswith("def ") or stripped.startswith("async def "):
                        result_lines.append(line)
                        result_lines.append("        ...")
                    elif stripped.startswith('"""') or stripped.startswith("'''"):
                        result_lines.append(line)
                    elif stripped and not stripped.startswith("#"):
                        if "=" in stripped:
                            result_lines.append(line)
            return "\n".join(result_lines)

    return None
