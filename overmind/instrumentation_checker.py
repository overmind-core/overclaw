"""Static validation for MCP instrumentation placement plans."""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _Placement:
    file: str
    qualname: str
    mode: str
    key: str | None
    key_from: str | None
    allowed_keys: tuple[str, ...] | None


@dataclass(frozen=True)
class _Boundary:
    node: ast.AST
    owner: tuple[str, ...] | None
    decorator: bool
    call: ast.Call | None
    line: int


@dataclass
class _Source:
    path: Path
    tree: ast.AST
    functions: list[tuple[tuple[str, ...], ast.FunctionDef | ast.AsyncFunctionDef]]
    boundaries: list[_Boundary]
    has_overmind_import: bool
    has_wrong_overmind_import: bool


def _issue(code: str, status: str, message: str, file: str = "", qualname: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "status": status, "message": message}
    if file:
        result["file"] = file
    if qualname:
        result["qualname"] = qualname
    return result


def _result(checks: list[dict[str, Any]], revision: dict[str, Any] | None = None) -> dict[str, Any]:
    failures = [check for check in checks if check["status"] == "fail"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "ok": not failures,
        "checks": checks,
        "errors": failures,
        "summary": {
            "passed": sum(check["status"] == "pass" for check in checks),
            "failed": len(failures),
            "skipped": sum(check["status"] == "skip" for check in checks),
        },
    }
    if revision is not None:
        result["revision"] = revision
    return result


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _expression_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = _first(value, "expression", "path", "source", "name")
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    return ast.dump(expression, annotate_fields=True, include_attributes=False)


def _decorator_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().removeprefix("@").strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not _is_task_ref(expression.func):
        return None
    return _literal_task_key(expression)


def _decorator_key_from(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().removeprefix("@").strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not _is_task_ref(expression.func):
        return None
    key_from_expr = _task_key_from(expression)
    if key_from_expr is None:
        return None
    return ast.dump(key_from_expr, annotate_fields=True, include_attributes=False)


def _placements(plan: Any) -> tuple[list[Any], str | None]:
    if isinstance(plan, list):
        return plan, None
    if not isinstance(plan, Mapping):
        return [], "the plan must be a JSON object with placements"

    value = _first(plan, "placements", "targets", "tasks")
    if value is None and isinstance(plan.get("plan"), (Mapping, list)):
        return _placements(plan["plan"])
    if value is None:
        return [], "the plan is missing placements"
    if isinstance(value, list):
        return value, None
    if isinstance(value, Mapping):
        grouped: list[Any] = []
        for group in ("fixed", "dynamic", "primary"):
            entries = value.get(group)
            if isinstance(entries, list):
                grouped.extend(entries)
        if grouped:
            return grouped, None
        return [value], None
    return [], "placements must be a JSON array"


def _normalise_placement(raw: Any) -> tuple[_Placement | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "each placement must be a JSON object"
    target = raw.get("target") if isinstance(raw.get("target"), Mapping) else raw
    task = raw.get("task") if isinstance(raw.get("task"), Mapping) else raw
    file = _first(target, "file", "path", "target_file", "source_file")
    qualname = _first(target, "qualname", "target_qualname", "entry_qualname", "entrypoint_qualname")
    if not isinstance(file, str) or not file.strip():
        return None, "placement file is required"
    if not isinstance(qualname, str) or not qualname.strip():
        return None, "placement qualname is required"

    key = _first(task, "key", "task_key", "behaviour_key", "behavior_key", "expected_key")
    if key is None:
        key = _decorator_key(_first(raw, "task_decorator", "decorator"))
    key_from_value = _first(task, "key_from", "dynamic_key_from")
    # A fixed placement still carries ``allowed_keys: null`` or ``[]`` from the
    # server; only a non-empty list marks a shared (dynamic) entry.
    dynamic = key_from_value is not None or bool(task.get("allowed_keys"))
    mode = str(_first(task, "mode", "kind", "placement", "placement_mode") or "").lower()
    dynamic = dynamic or "dynamic" in mode or mode in {"context", "context_manager"}
    if dynamic:
        key_from = _expression_text(key_from_value)
        if key_from is None:
            key_from = _decorator_key_from(_first(raw, "required_task_decorator", "task_decorator", "decorator"))
        allowed = task.get("allowed_keys")
        if allowed is None and isinstance(task.get("allowed"), Mapping):
            allowed = _first(task["allowed"], "keys", "values")
        if isinstance(allowed, list) and all(isinstance(item, str) for item in allowed):
            allowed_keys: tuple[str, ...] | None = tuple(allowed)
        else:
            allowed_keys = None
        return _Placement(file.strip(), qualname.strip(), "dynamic", None, key_from, allowed_keys), None

    if not isinstance(key, str) or not key.strip():
        return None, "fixed placement requires a non-empty task key"
    return _Placement(file.strip(), qualname.strip(), "fixed", key.strip(), None, None), None


def _is_task_ref(node: ast.AST) -> bool:
    # entry_point declares the run-boundary contract the same way task declares
    # a turn's: both carry the behaviour key a fixed placement pins.
    return (
        isinstance(node, ast.Attribute)
        and node.attr in ("task", "entry_point")
        and isinstance(node.value, ast.Name)
        and node.value.id == "overmind"
    )


def _literal_task_key(call: ast.Call) -> str | None:
    if len(call.args) > 1:
        return None
    key_keyword = next((keyword for keyword in call.keywords if keyword.arg == "key"), None)
    if call.args and key_keyword is not None:
        return None
    if any(keyword.arg == "key_from" for keyword in call.keywords):
        return None
    value = call.args[0] if call.args else key_keyword.value if key_keyword else None
    return (
        value.value.strip()
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.strip()
        else None
    )


def _task_key_from(call: ast.Call) -> ast.AST | None:
    if call.args:
        return None
    for keyword in call.keywords:
        if keyword.arg == "key_from":
            return keyword.value
    return None


class _AstIndex(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.function_stack: list[tuple[str, ...]] = []
        self.functions: list[tuple[tuple[str, ...], ast.FunctionDef | ast.AsyncFunctionDef]] = []
        self.boundaries: list[_Boundary] = []

    @property
    def owner(self) -> tuple[str, ...] | None:
        return self.function_stack[-1] if self.function_stack else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        path = tuple(self.scope)
        self.functions.append((path, node))
        for decorator in node.decorator_list:
            if _is_task_ref(decorator):
                self.boundaries.append(_Boundary(decorator, path, True, None, decorator.lineno))
            elif isinstance(decorator, ast.Call) and _is_task_ref(decorator.func):
                self.boundaries.append(_Boundary(decorator, path, True, decorator, decorator.lineno))
        self.function_stack.append(path)
        self.generic_visit(node)
        self.function_stack.pop()
        self.scope.pop()

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            expression = item.context_expr
            if isinstance(expression, ast.Call) and _is_task_ref(expression.func):
                self.boundaries.append(_Boundary(expression, self.owner, False, expression, expression.lineno))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)


def _source(path: Path) -> _Source | dict[str, Any]:
    try:
        text = path.read_text()
    except OSError as exc:
        return _issue("file.missing", "fail", f"target file cannot be read: {exc}")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return _issue("source.syntax_error", "fail", f"target file is not valid Python: {exc.msg}")

    index = _AstIndex()
    index.visit(tree)
    has_import = False
    has_wrong_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            has_import = has_import or any(
                alias.name == "overmind" and alias.asname in (None, "overmind") for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "overmind":
            has_wrong_import = True
    return _Source(path, tree, index.functions, index.boundaries, has_import, has_wrong_import)


def _resolve_file(root: Path, value: str) -> Path:
    relative = value.split("#", 1)[0].strip()
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _matches(path: tuple[str, ...], target: str) -> bool:
    dotted = ".".join(path)
    return target == dotted or target.endswith(f".{dotted}")


def _boundary_nested(first: _Boundary, second: _Boundary) -> bool:
    if first.owner and second.owner and first.owner != second.owner:
        return (len(first.owner) < len(second.owner) and second.owner[: len(first.owner)] == first.owner) or (
            len(second.owner) < len(first.owner) and first.owner[: len(second.owner)] == second.owner
        )
    if first.owner != second.owner or first.decorator == second.decorator:
        return False
    first_end = getattr(first.node, "end_lineno", first.line)
    second_end = getattr(second.node, "end_lineno", second.line)
    return first.line <= second.line <= first_end or second.line <= first.line <= second_end


def _revision_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = _first(value, "sha", "revision", "value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else None


def check_plan(plan: Any, root: str | Path = ".") -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    placements_raw, plan_error = _placements(plan)
    if plan_error:
        return _result([_issue("plan.invalid", "fail", plan_error)])
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return _result([_issue("root.missing", "fail", f"root directory does not exist: {root_path}")])

    plan_revisions: list[str] = []
    if isinstance(plan, Mapping):
        for field in ("revision", "source_revision", "git_sha", "analyzed_sha"):
            if (revision := _revision_value(plan.get(field))) is not None:
                plan_revisions.append(revision)
        version = plan.get("version")
        if isinstance(version, Mapping) and (revision := _revision_value(version.get("analyzed_sha"))) is not None:
            plan_revisions.append(revision)
    placements: list[_Placement] = []
    placement_revisions: list[tuple[_Placement, list[str]]] = []
    for raw in placements_raw:
        placement, error = _normalise_placement(raw)
        if error:
            checks.append(_issue("plan.placement_invalid", "fail", error))
            continue
        assert placement is not None
        placements.append(placement)
        revisions: list[str] = []
        if isinstance(raw, Mapping):
            for field in ("revision", "analyzed_sha"):
                if (revision := _revision_value(raw.get(field))) is not None:
                    revisions.append(revision)
            version = raw.get("version")
            if isinstance(version, Mapping) and (revision := _revision_value(version.get("analyzed_sha"))) is not None:
                revisions.append(revision)
        placement_revisions.append((placement, revisions))

    expected_revisions = plan_revisions + [rev for _, revs in placement_revisions for rev in revs]
    revision: dict[str, Any] = {"expected": sorted(set(expected_revisions)), "actual": None, "status": "skip"}
    if expected_revisions:
        actual = _git_revision(root_path)
        revision["actual"] = actual
        if actual is None:
            checks.append(
                _issue(
                    "revision.unavailable", "skip", "plan revision is present but the local git revision is unavailable"
                )
            )
        else:
            mismatched = any(actual != expected for expected in plan_revisions)
            if mismatched:
                checks.append(_issue("revision.mismatch", "fail", "local revision does not match the plan"))
            for placement, revs in placement_revisions:
                if any(actual != expected for expected in revs):
                    mismatched = True
                    checks.append(
                        _issue(
                            "revision.mismatch",
                            "fail",
                            f"local revision does not match the plan for this placement "
                            f"(expected {sorted(set(revs))!r}, actual {actual!r})",
                            placement.file,
                            placement.qualname,
                        )
                    )
            if mismatched:
                revision["status"] = "fail"
            else:
                revision["status"] = "pass"
                checks.append(_issue("revision.match", "pass", "local revision matches the plan"))

    cache: dict[Path, _Source | dict[str, Any]] = {}
    import_checked: set[Path] = set()
    for placement in placements:
        path = _resolve_file(root_path, placement.file)
        loaded = cache.get(path)
        if loaded is None:
            loaded = _source(path)
            cache[path] = loaded
        if isinstance(loaded, dict):
            checks.append({**loaded, "file": placement.file, "qualname": placement.qualname})
            continue
        if path not in import_checked:
            import_checked.add(path)
            if loaded.has_overmind_import:
                checks.append(_issue("import.correct", "pass", "source imports overmind", placement.file))
            else:
                message = "source must use `import overmind` for `@overmind.task(...)`"
                if loaded.has_wrong_overmind_import:
                    message += "; `from overmind import ...` is not the planned import"
                checks.append(_issue("import.missing", "fail", message, placement.file))

        matches = [
            (path_parts, node) for path_parts, node in loaded.functions if _matches(path_parts, placement.qualname)
        ]
        if not matches:
            checks.append(
                _issue(
                    "target.missing",
                    "fail",
                    "planned qualname was not found in the target file",
                    placement.file,
                    placement.qualname,
                )
            )
            continue
        if len(matches) > 1:
            checks.append(
                _issue(
                    "target.ambiguous",
                    "fail",
                    "planned qualname matches more than one definition",
                    placement.file,
                    placement.qualname,
                )
            )
            continue
        target_path, _ = matches[0]
        checks.append(_issue("target.found", "pass", "planned qualname found", placement.file, placement.qualname))
        decorators = [
            boundary for boundary in loaded.boundaries if boundary.decorator and boundary.owner == target_path
        ]
        contexts = [
            boundary for boundary in loaded.boundaries if not boundary.decorator and boundary.owner == target_path
        ]
        if placement.mode == "fixed":
            # ``task()`` is a decorator OR a context manager; either satisfies a
            # fixed placement, but only one boundary may own the target.
            fixed_boundaries = decorators + contexts
            if len(fixed_boundaries) == 0:
                checks.append(
                    _issue(
                        "task.missing",
                        "fail",
                        "target has no overmind.task or entry_point boundary",
                        placement.file,
                        placement.qualname,
                    )
                )
            elif len(fixed_boundaries) > 1:
                checks.append(
                    _issue(
                        "task.duplicate",
                        "fail",
                        "target has more than one overmind.task boundary",
                        placement.file,
                        placement.qualname,
                    )
                )
            else:
                boundary = fixed_boundaries[0]
                if boundary.call is None:
                    checks.append(
                        _issue(
                            "task.shape",
                            "fail",
                            "overmind.task must be called with a key: @overmind.task(key)",
                            placement.file,
                            placement.qualname,
                        )
                    )
                else:
                    key = _literal_task_key(boundary.call)
                    if key != placement.key:
                        checks.append(
                            _issue(
                                "task.key_mismatch",
                                "fail",
                                f"task key is {key!r}, expected {placement.key!r}",
                                placement.file,
                                placement.qualname,
                            )
                        )
                    else:
                        checks.append(
                            _issue(
                                "task.fixed",
                                "pass",
                                "exact fixed overmind.task(key) placement found",
                                placement.file,
                                placement.qualname,
                            )
                        )
        else:
            if placement.key_from is None:
                checks.append(
                    _issue(
                        "dynamic.key_from",
                        "skip",
                        "plan does not constrain the selector expression",
                        placement.file,
                        placement.qualname,
                    )
                )
            if (
                placement.allowed_keys is None
                or not placement.allowed_keys
                or any(not key.strip() for key in placement.allowed_keys)
                or len(set(placement.allowed_keys)) != len(placement.allowed_keys)
            ):
                checks.append(
                    _issue(
                        "dynamic.allowed_keys",
                        "fail",
                        "allowed_keys must be a non-empty list of unique non-empty strings",
                        placement.file,
                        placement.qualname,
                    )
                )
            dynamic_boundaries = decorators + contexts
            if len(dynamic_boundaries) != 1:
                checks.append(
                    _issue(
                        "dynamic.boundary",
                        "fail",
                        "dynamic target must contain exactly one dynamic overmind.task boundary",
                        placement.file,
                        placement.qualname,
                    )
                )
            else:
                boundary = dynamic_boundaries[0]
                call = boundary.call
                expression = _task_key_from(call) if call is not None else None
                if expression is not None:
                    actual = ast.dump(expression, annotate_fields=True, include_attributes=False)
                    if placement.key_from is not None and actual != placement.key_from:
                        checks.append(
                            _issue(
                                "dynamic.key_from_mismatch",
                                "fail",
                                "dynamic task key expression does not match key_from",
                                placement.file,
                                placement.qualname,
                            )
                        )
                    else:
                        checks.append(
                            _issue(
                                "dynamic.placement",
                                "pass",
                                "dynamic key_from task decorator found",
                                placement.file,
                                placement.qualname,
                            )
                        )
                elif call is not None and not boundary.decorator and _literal_task_key(call) is None:
                    checks.append(
                        _issue(
                            "dynamic.placement",
                            "pass",
                            "dynamic overmind.task(<expression>) boundary found",
                            placement.file,
                            placement.qualname,
                        )
                    )
                else:
                    checks.append(
                        _issue(
                            "dynamic.shape",
                            "fail",
                            "dynamic task boundary must compute its key: `with overmind.task(<expression>)` "
                            "or `@overmind.task(key_from=...)`",
                            placement.file,
                            placement.qualname,
                        )
                    )

    for index, first in enumerate(
        boundaries := [
            boundary for loaded in cache.values() if isinstance(loaded, _Source) for boundary in loaded.boundaries
        ]
    ):
        for second in boundaries[index + 1 :]:
            if _boundary_nested(first, second):
                checks.append(
                    _issue(
                        "task.nested",
                        "fail",
                        "task boundaries may not be nested",
                        qualname=".".join(second.owner or first.owner or ()),
                    )
                )
                break
    return _result(checks, revision)


def check_plan_file(plan_file: str | Path, root: str | Path = ".") -> dict[str, Any]:
    path = Path(plan_file)
    try:
        plan = json.loads(path.read_text())
    except OSError as exc:
        return _result([_issue("plan.missing", "fail", f"plan file cannot be read: {exc}")])
    except json.JSONDecodeError as exc:
        return _result([_issue("plan.invalid_json", "fail", f"plan file is not valid JSON: {exc.msg}")])
    return check_plan(plan, root)
