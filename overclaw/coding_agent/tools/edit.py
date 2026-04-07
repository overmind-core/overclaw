"""Edit tool — string find-and-replace with a 9-strategy fuzzy matching chain.

The LLM provides an oldString and newString. The tool reads the entire file,
locates the oldString using a chain of progressively relaxed matchers, and
splices in the newString. The match MUST be unique (unless replaceAll=True).

Replacer chain (tried in order, first unique match wins):
  1. SimpleReplacer        — exact substring
  2. LineTrimmedReplacer   — trim each line before comparing
  3. BlockAnchorReplacer   — match first/last lines, Levenshtein on middle
  4. WhitespaceNormalizedReplacer — collapse whitespace
  5. IndentationFlexibleReplacer  — strip common indentation
  6. EscapeNormalizedReplacer     — unescape \\n \\t etc.
  7. TrimmedBoundaryReplacer      — trim leading/trailing blank lines
  8. ContextAwareReplacer         — anchor lines + 50% middle similarity
  9. MultiOccurrenceReplacer      — all exact matches (for replaceAll)
"""

from __future__ import annotations

import os
import re
import difflib
from typing import Any
from collections.abc import Generator

from .base import ToolContext, ToolResult


DESCRIPTION = (
    "Performs exact string replacements in files.\n\n"
    "Usage:\n"
    "- You must read the file before editing. This tool will error otherwise.\n"
    "- The edit will FAIL if oldString is not found or found multiple times.\n"
    "- Use replaceAll to replace every occurrence of oldString.\n"
    "- Preserve the exact indentation from the file when specifying oldString."
)

PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {
            "type": "string",
            "description": "The absolute path to the file to modify",
        },
        "oldString": {
            "type": "string",
            "description": "The text to replace",
        },
        "newString": {
            "type": "string",
            "description": "The replacement text (must differ from oldString)",
        },
        "replaceAll": {
            "type": "boolean",
            "description": "Replace all occurrences of oldString (default false)",
        },
    },
    "required": ["filePath", "oldString", "newString"],
}

# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        m[i][0] = i
    for j in range(len(b) + 1):
        m[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            m[i][j] = min(m[i - 1][j] + 1, m[i][j - 1] + 1, m[i - 1][j - 1] + cost)
    return m[len(a)][len(b)]


# ---------------------------------------------------------------------------
# Replacer generators — each yields candidate strings to search for
# ---------------------------------------------------------------------------

Replacer = Generator[str]

SINGLE_THRESHOLD = 0.0
MULTI_THRESHOLD = 0.3


def _simple(content: str, find: str) -> Replacer:
    yield find


def _line_trimmed(content: str, find: str) -> Replacer:
    original = content.split("\n")
    search = find.split("\n")
    if search and search[-1] == "":
        search.pop()
    for i in range(len(original) - len(search) + 1):
        match = True
        for j in range(len(search)):
            if original[i + j].strip() != search[j].strip():
                match = False
                break
        if match:
            start = sum(len(original[k]) + 1 for k in range(i))
            end = start
            for k in range(len(search)):
                end += len(original[i + k])
                if k < len(search) - 1:
                    end += 1
            yield content[start:end]


def _block_anchor(content: str, find: str) -> Replacer:
    original = content.split("\n")
    search = find.split("\n")
    if len(search) < 3:
        return
    if search and search[-1] == "":
        search.pop()

    first = search[0].strip()
    last = search[-1].strip()
    candidates: list[tuple[int, int]] = []

    for i in range(len(original)):
        if original[i].strip() != first:
            continue
        for j in range(i + 2, len(original)):
            if original[j].strip() == last:
                candidates.append((i, j))
                break

    if not candidates:
        return

    def _extract(start: int, end: int) -> str:
        s = sum(len(original[k]) + 1 for k in range(start))
        e = s
        for k in range(start, end + 1):
            e += len(original[k])
            if k < end:
                e += 1
        return content[s:e]

    def _similarity(start: int, end: int) -> float:
        actual = end - start + 1
        to_check = min(len(search) - 2, actual - 2)
        if to_check <= 0:
            return 1.0
        total = 0.0
        for j in range(1, min(len(search) - 1, actual - 1) + 1):
            a = original[start + j].strip()
            b = search[j].strip()
            mx = max(len(a), len(b))
            if mx == 0:
                continue
            total += (1 - _levenshtein(a, b) / mx) / to_check
        return total

    if len(candidates) == 1:
        s, e = candidates[0]
        if _similarity(s, e) >= SINGLE_THRESHOLD:
            yield _extract(s, e)
        return

    best, best_sim = None, -1.0
    for s, e in candidates:
        actual = e - s + 1
        to_check = min(len(search) - 2, actual - 2)
        if to_check <= 0:
            sim = 1.0
        else:
            sim = 0.0
            for j in range(1, min(len(search) - 1, actual - 1) + 1):
                a = original[s + j].strip()
                b = search[j].strip()
                mx = max(len(a), len(b))
                if mx == 0:
                    continue
                sim += 1 - _levenshtein(a, b) / mx
            sim /= to_check
        if sim > best_sim:
            best_sim = sim
            best = (s, e)

    if best_sim >= MULTI_THRESHOLD and best:
        yield _extract(*best)


def _whitespace_normalized(content: str, find: str) -> Replacer:
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    nf = norm(find)
    lines = content.split("\n")

    for line in lines:
        if norm(line) == nf:
            yield line
        elif nf in norm(line):
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                m = re.search(pattern, line)
                if m:
                    yield m.group(0)

    flines = find.split("\n")
    if len(flines) > 1:
        for i in range(len(lines) - len(flines) + 1):
            block = lines[i : i + len(flines)]
            if norm("\n".join(block)) == nf:
                yield "\n".join(block)


def _indentation_flexible(content: str, find: str) -> Replacer:
    def strip_indent(text: str) -> str:
        lines = text.split("\n")
        nonempty = [ln for ln in lines if ln.strip()]
        if not nonempty:
            return text
        mn = min(len(ln) - len(ln.lstrip()) for ln in nonempty)
        return "\n".join(ln[mn:] if ln.strip() else ln for ln in lines)

    nf = strip_indent(find)
    clines = content.split("\n")
    flines = find.split("\n")

    for i in range(len(clines) - len(flines) + 1):
        block = "\n".join(clines[i : i + len(flines)])
        if strip_indent(block) == nf:
            yield block


def _escape_normalized(content: str, find: str) -> Replacer:
    _map = {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "'": "'",
        '"': '"',
        "`": "`",
        "\\": "\\",
        "\n": "\n",
        "$": "$",
    }

    def unescape(s: str) -> str:
        def repl(m: re.Match) -> str:
            return _map.get(m.group(1), m.group(0))

        return re.sub(r"\\([ntr'\"` \\\n$])", repl, s)

    uf = unescape(find)
    if uf in content:
        yield uf

    lines = content.split("\n")
    flines = uf.split("\n")
    for i in range(len(lines) - len(flines) + 1):
        block = "\n".join(lines[i : i + len(flines)])
        if unescape(block) == uf:
            yield block


def _trimmed_boundary(content: str, find: str) -> Replacer:
    trimmed = find.strip()
    if trimmed == find:
        return
    if trimmed in content:
        yield trimmed
    lines = content.split("\n")
    flines = find.split("\n")
    for i in range(len(lines) - len(flines) + 1):
        block = "\n".join(lines[i : i + len(flines)])
        if block.strip() == trimmed:
            yield block


def _context_aware(content: str, find: str) -> Replacer:
    flines = find.split("\n")
    if len(flines) < 3:
        return
    if flines and flines[-1] == "":
        flines.pop()
    clines = content.split("\n")
    first = flines[0].strip()
    last = flines[-1].strip()

    for i in range(len(clines)):
        if clines[i].strip() != first:
            continue
        for j in range(i + 2, len(clines)):
            if clines[j].strip() == last:
                block = clines[i : j + 1]
                if len(block) == len(flines):
                    matching = total = 0
                    for k in range(1, len(block) - 1):
                        bl = block[k].strip()
                        fl = flines[k].strip()
                        if bl or fl:
                            total += 1
                            if bl == fl:
                                matching += 1
                    if total == 0 or matching / total >= 0.5:
                        yield "\n".join(block)
                break


def _multi_occurrence(content: str, find: str) -> Replacer:
    start = 0
    while True:
        idx = content.find(find, start)
        if idx == -1:
            break
        yield find
        start = idx + len(find)


# ---------------------------------------------------------------------------
# Core replace function
# ---------------------------------------------------------------------------

_REPLACERS = [
    _simple,
    _line_trimmed,
    _block_anchor,
    _whitespace_normalized,
    _indentation_flexible,
    _escape_normalized,
    _trimmed_boundary,
    _context_aware,
    _multi_occurrence,
]


def replace(content: str, old: str, new: str, replace_all: bool = False) -> str:
    """Locate old in content using the replacer chain and splice in new."""
    if old == new:
        raise ValueError("No changes to apply: oldString and newString are identical.")

    not_found = True
    for replacer in _REPLACERS:
        for search in replacer(content, old):
            idx = content.find(search)
            if idx == -1:
                continue
            not_found = False
            if replace_all:
                return content.replace(search, new)
            last = content.rfind(search)
            if idx != last:
                continue
            return content[:idx] + new + content[idx + len(search) :]

    if not_found:
        raise ValueError(
            "Could not find oldString in the file. "
            "It must match exactly, including whitespace, indentation, and line endings."
        )
    raise ValueError(
        "Found multiple matches for oldString. "
        "Provide more surrounding context to make the match unique."
    )


# ---------------------------------------------------------------------------
# Line ending helpers
# ---------------------------------------------------------------------------


def _detect_ending(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n")


def _convert(text: str, ending: str) -> str:
    if ending == "\n":
        return text
    return text.replace("\n", "\r\n")


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class EditTool:
    name = "edit"
    description = DESCRIPTION
    parameters = PARAMS

    def execute(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = params.get("filePath", "")
        fp = raw if os.path.isabs(raw) else os.path.join(ctx.cwd, raw)
        old_str = params.get("oldString", "")
        new_str = params.get("newString", "")
        replace_all = params.get("replaceAll", False)

        if old_str == new_str:
            raise ValueError("oldString and newString are identical.")

        lock = ctx.file_tracker.lock_for(fp)
        with lock:
            if old_str == "":
                content_new = new_str
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as f:
                    f.write(content_new)
                ctx.record_read(fp)
                return ToolResult(
                    output="File created/overwritten successfully.",
                    title=os.path.relpath(fp, ctx.worktree),
                )

            if not os.path.isfile(fp):
                raise FileNotFoundError(f"File {fp} not found")

            ctx.assert_read(fp)

            with open(fp) as f:
                content_old = f.read()

            ending = _detect_ending(content_old)
            old_norm = _convert(_normalize(old_str), ending)
            new_norm = _convert(_normalize(new_str), ending)

            content_new = replace(content_old, old_norm, new_norm, replace_all)

            with open(fp, "w") as f:
                f.write(content_new)

            ctx.record_read(fp)

        diff = "\n".join(
            difflib.unified_diff(
                content_old.splitlines(),
                content_new.splitlines(),
                fromfile=fp,
                tofile=fp,
                lineterm="",
            )
        )
        return ToolResult(
            output="Edit applied successfully.",
            title=os.path.relpath(fp, ctx.worktree),
            metadata={"diff": diff},
        )
