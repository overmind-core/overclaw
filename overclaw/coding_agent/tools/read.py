"""Read tool — read files and directories with line numbers."""

from __future__ import annotations

import os
from typing import Any

from .base import ToolContext, ToolResult

DEFAULT_LIMIT = 2000
MAX_LINE_LEN = 2000
MAX_BYTES = 50 * 1024

DESCRIPTION = (
    "Read a file or directory from the local filesystem.\n\n"
    "Usage:\n"
    "- filePath should be an absolute path.\n"
    "- Returns up to 2000 lines by default.\n"
    "- offset is 1-indexed line number to start from.\n"
    "- Contents are prefixed with line numbers: '<line>: <content>'.\n"
    "- Use grep to find specific content in large files.\n"
    "- Use glob to discover files by pattern."
)

PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {
            "type": "string",
            "description": "Absolute path to the file or directory to read",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read (default 2000)",
        },
    },
    "required": ["filePath"],
}

_BINARY_EXTS = {
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".class",
    ".jar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".bin",
    ".dat",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".wasm",
    ".pyc",
    ".pyo",
}


def _is_binary(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in _BINARY_EXTS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        if not chunk:
            return False
        if b"\x00" in chunk:
            return True
        non_print = sum(1 for b in chunk if b < 9 or (13 < b < 32))
        return non_print / len(chunk) > 0.3
    except OSError:
        return False


class ReadTool:
    name = "read"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = params.get("filePath", "")
        fp = raw if os.path.isabs(raw) else os.path.join(ctx.cwd, raw)
        offset = params.get("offset", 1)
        limit = params.get("limit", DEFAULT_LIMIT)

        if offset < 1:
            raise ValueError("offset must be >= 1")

        if not os.path.exists(fp):
            parent = os.path.dirname(fp)
            base = os.path.basename(fp).lower()
            suggestions = []
            try:
                for entry in os.listdir(parent):
                    if base in entry.lower() or entry.lower() in base:
                        suggestions.append(os.path.join(parent, entry))
                        if len(suggestions) >= 3:
                            break
            except OSError:
                pass
            hint = (
                "\n\nDid you mean one of these?\n" + "\n".join(suggestions)
                if suggestions
                else ""
            )
            raise FileNotFoundError(f"File not found: {fp}{hint}")

        title = os.path.relpath(fp, ctx.worktree)

        if os.path.isdir(fp):
            entries = sorted(os.listdir(fp))
            for i, e in enumerate(entries):
                if os.path.isdir(os.path.join(fp, e)):
                    entries[i] = e + "/"
            start = offset - 1
            sliced = entries[start : start + limit]
            truncated = start + len(sliced) < len(entries)
            lines = [f"<path>{fp}</path>", "<type>directory</type>", "<entries>"]
            lines.append("\n".join(sliced))
            if truncated:
                lines.append(
                    f"\n(Showing {len(sliced)} of {len(entries)} entries. "
                    f"Use offset={offset + len(sliced)} to see more.)"
                )
            else:
                lines.append(f"\n({len(entries)} entries)")
            lines.append("</entries>")
            return ToolResult(
                output="\n".join(lines),
                title=title,
                metadata={"truncated": truncated},
            )

        if _is_binary(fp):
            raise ValueError(f"Cannot read binary file: {fp}")

        raw_lines: list[str] = []
        total = 0
        truncated_by_bytes = False
        has_more = False
        byte_count = 0
        start = offset - 1

        with open(fp, errors="replace") as f:
            for line_text in f:
                total += 1
                if total <= start:
                    continue
                if len(raw_lines) >= limit:
                    has_more = True
                    continue
                line_text = line_text.rstrip("\n").rstrip("\r")
                if len(line_text) > MAX_LINE_LEN:
                    line_text = (
                        line_text[:MAX_LINE_LEN]
                        + f"... (truncated to {MAX_LINE_LEN} chars)"
                    )
                size = len(line_text.encode()) + 1
                if byte_count + size > MAX_BYTES:
                    truncated_by_bytes = True
                    has_more = True
                    break
                raw_lines.append(line_text)
                byte_count += size

        numbered = [f"{i + offset}: {line}" for i, line in enumerate(raw_lines)]
        last_read = offset + len(raw_lines) - 1
        next_offset = last_read + 1
        truncated = has_more or truncated_by_bytes

        out = [f"<path>{fp}</path>", "<type>file</type>", "<content>"]
        out.append("\n".join(numbered))

        if truncated_by_bytes:
            out.append(
                f"\n\n(Output capped at {MAX_BYTES // 1024} KB. "
                f"Lines {offset}-{last_read}. Use offset={next_offset} to continue.)"
            )
        elif has_more:
            out.append(
                f"\n\n(Showing lines {offset}-{last_read} of {total}. "
                f"Use offset={next_offset} to continue.)"
            )
        else:
            out.append(f"\n\n(End of file - total {total} lines)")
        out.append("\n</content>")

        ctx.record_read(fp)

        return ToolResult(
            output="\n".join(out),
            title=title,
            metadata={"truncated": truncated, "preview": "\n".join(raw_lines[:20])},
        )
