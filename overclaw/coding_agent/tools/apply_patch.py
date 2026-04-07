"""Apply-patch tool — line-based patching for GPT models.

Uses a structured patch format with *** Begin Patch / *** End Patch markers.
Hunks are line-based with context seeking and 4-pass fuzzy matching:
  1. Exact match
  2. Right-strip (trailing whitespace)
  3. Full trim (both ends)
  4. Unicode-normalized trim
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ToolContext, ToolResult

DESCRIPTION = (
    "Apply a patch to one or more files using the structured patch format.\n\n"
    "Format:\n"
    "  *** Begin Patch\n"
    "  *** Add File: <path>\n"
    "  +line1\n"
    "  +line2\n"
    "  *** Update File: <path>\n"
    "  @@ optional_context_line\n"
    "   context line\n"
    "  -removed line\n"
    "  +added line\n"
    "  *** Delete File: <path>\n"
    "  *** End Patch"
)

PARAMS = {
    "type": "object",
    "properties": {
        "patchText": {
            "type": "string",
            "description": "The full patch text describing all changes",
        },
    },
    "required": ["patchText"],
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    context: str | None = None
    is_eof: bool = False


@dataclass
class AddHunk:
    path: str
    contents: str


@dataclass
class DeleteHunk:
    path: str


@dataclass
class UpdateHunk:
    path: str
    move_path: str | None
    chunks: list[Chunk]


Hunk = AddHunk | DeleteHunk | UpdateHunk


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _strip_heredoc(text: str) -> str:
    m = re.match(r"^(?:cat\s+)?<<['\"]?(\w+)['\"]?\s*\n([\s\S]*?)\n\1\s*$", text)
    return m.group(2) if m else text


def parse_patch(text: str) -> list[Hunk]:
    cleaned = _strip_heredoc(text.strip())
    lines = cleaned.split("\n")

    begin = next(
        (i for i, line in enumerate(lines) if line.strip() == "*** Begin Patch"), None
    )
    end = next(
        (i for i, line in enumerate(lines) if line.strip() == "*** End Patch"), None
    )
    if begin is None or end is None or begin >= end:
        raise ValueError("Invalid patch format: missing Begin/End markers")

    hunks: list[Hunk] = []
    i = begin + 1

    while i < end:
        line = lines[i]

        if line.startswith("*** Add File:"):
            path = line[len("*** Add File:") :].strip()
            i += 1
            content_lines: list[str] = []
            while i < end and not lines[i].startswith("***"):
                if lines[i].startswith("+"):
                    content_lines.append(lines[i][1:])
                i += 1
            hunks.append(AddHunk(path=path, contents="\n".join(content_lines)))

        elif line.startswith("*** Delete File:"):
            path = line[len("*** Delete File:") :].strip()
            hunks.append(DeleteHunk(path=path))
            i += 1

        elif line.startswith("*** Update File:"):
            path = line[len("*** Update File:") :].strip()
            move_path = None
            i += 1
            if i < end and lines[i].startswith("*** Move to:"):
                move_path = lines[i][len("*** Move to:") :].strip()
                i += 1
            chunks: list[Chunk] = []
            while i < end and not lines[i].startswith("***"):
                if lines[i].startswith("@@"):
                    ctx = lines[i][2:].strip() or None
                    i += 1
                    old: list[str] = []
                    new: list[str] = []
                    is_eof = False
                    while (
                        i < end
                        and not lines[i].startswith("@@")
                        and not lines[i].startswith("***")
                    ):
                        cl = lines[i]
                        if cl == "*** End of File":
                            is_eof = True
                            i += 1
                            break
                        if cl.startswith(" "):
                            old.append(cl[1:])
                            new.append(cl[1:])
                        elif cl.startswith("-"):
                            old.append(cl[1:])
                        elif cl.startswith("+"):
                            new.append(cl[1:])
                        i += 1
                    chunks.append(
                        Chunk(old_lines=old, new_lines=new, context=ctx, is_eof=is_eof)
                    )
                else:
                    i += 1
            hunks.append(UpdateHunk(path=path, move_path=move_path, chunks=chunks))
        else:
            i += 1

    return hunks


# ---------------------------------------------------------------------------
# Line matching (4-pass fuzzy seek)
# ---------------------------------------------------------------------------


def _normalize_unicode(s: str) -> str:
    return (
        s.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201a", "'")
        .replace("\u201b", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u201f", '"')
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2015", "-")
        .replace("\u2026", "...")
        .replace("\u00a0", " ")
    )


def _try_match(
    lines: list[str],
    pattern: list[str],
    start: int,
    compare: Any,
    eof: bool,
) -> int:
    if eof:
        from_end = len(lines) - len(pattern)
        if from_end >= start:
            if all(
                compare(lines[from_end + j], pattern[j]) for j in range(len(pattern))
            ):
                return from_end
    for i in range(start, len(lines) - len(pattern) + 1):
        if all(compare(lines[i + j], pattern[j]) for j in range(len(pattern))):
            return i
    return -1


def _seek(lines: list[str], pattern: list[str], start: int, eof: bool = False) -> int:
    if not pattern:
        return -1
    r = _try_match(lines, pattern, start, lambda a, b: a == b, eof)
    if r != -1:
        return r
    r = _try_match(lines, pattern, start, lambda a, b: a.rstrip() == b.rstrip(), eof)
    if r != -1:
        return r
    r = _try_match(lines, pattern, start, lambda a, b: a.strip() == b.strip(), eof)
    if r != -1:
        return r
    return _try_match(
        lines,
        pattern,
        start,
        lambda a, b: _normalize_unicode(a.strip()) == _normalize_unicode(b.strip()),
        eof,
    )


# ---------------------------------------------------------------------------
# Apply logic
# ---------------------------------------------------------------------------


def _apply_update(filepath: str, chunks: list[Chunk]) -> str:
    with open(filepath) as f:
        original = f.read()

    orig_lines = original.split("\n")
    if orig_lines and orig_lines[-1] == "":
        orig_lines.pop()

    replacements: list[tuple[int, int, list[str]]] = []
    idx = 0

    for chunk in chunks:
        if chunk.context:
            ci = _seek(orig_lines, [chunk.context], idx)
            if ci == -1:
                raise ValueError(
                    f"Failed to find context '{chunk.context}' in {filepath}"
                )
            idx = ci + 1

        if not chunk.old_lines:
            ins = (
                len(orig_lines) - 1
                if orig_lines and orig_lines[-1] == ""
                else len(orig_lines)
            )
            replacements.append((ins, 0, chunk.new_lines))
            continue

        pattern = chunk.old_lines
        new_slice = chunk.new_lines
        found = _seek(orig_lines, pattern, idx, chunk.is_eof)

        if found == -1 and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = _seek(orig_lines, pattern, idx, chunk.is_eof)

        if found == -1:
            raise ValueError(
                f"Failed to find expected lines in {filepath}:\n"
                + "\n".join(chunk.old_lines)
            )

        replacements.append((found, len(pattern), new_slice))
        idx = found + len(pattern)

    replacements.sort(key=lambda r: r[0])
    result = list(orig_lines)
    for start, count, new in reversed(replacements):
        result[start : start + count] = new

    if not result or result[-1] != "":
        result.append("")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class ApplyPatchTool:
    name = "apply_patch"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        text = params.get("patchText", "")
        hunks = parse_patch(text)
        if not hunks:
            raise ValueError("No files were modified.")

        results: list[str] = []

        for hunk in hunks:
            if isinstance(hunk, AddHunk):
                fp = (
                    hunk.path
                    if os.path.isabs(hunk.path)
                    else os.path.join(ctx.cwd, hunk.path)
                )
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as f:
                    f.write(hunk.contents)
                ctx.record_read(fp)
                results.append(f"Added: {hunk.path}")

            elif isinstance(hunk, DeleteHunk):
                fp = (
                    hunk.path
                    if os.path.isabs(hunk.path)
                    else os.path.join(ctx.cwd, hunk.path)
                )
                os.remove(fp)
                results.append(f"Deleted: {hunk.path}")

            elif isinstance(hunk, UpdateHunk):
                fp = (
                    hunk.path
                    if os.path.isabs(hunk.path)
                    else os.path.join(ctx.cwd, hunk.path)
                )
                ctx.assert_read(fp)
                new_content = _apply_update(fp, hunk.chunks)
                target = fp
                if hunk.move_path:
                    target = (
                        hunk.move_path
                        if os.path.isabs(hunk.move_path)
                        else os.path.join(ctx.cwd, hunk.move_path)
                    )
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w") as f:
                    f.write(new_content)
                if hunk.move_path:
                    os.remove(fp)
                    results.append(f"Moved: {hunk.path} -> {hunk.move_path}")
                else:
                    results.append(f"Updated: {hunk.path}")
                ctx.record_read(target)

        return ToolResult(output="\n".join(results), title="apply_patch")
