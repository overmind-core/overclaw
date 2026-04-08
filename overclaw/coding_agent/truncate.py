"""Output truncation — caps large tool outputs to stay within context limits."""

from __future__ import annotations

MAX_LINES = 1500
MAX_BYTES = 48_000


def truncate(
    text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES
) -> tuple[str, bool]:
    """Truncate text to fit within limits.

    Returns (output, was_truncated).
    """
    lines = text.split("\n")
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return text, False

    kept: list[str] = []
    total = 0
    for line in lines:
        if len(kept) >= max_lines:
            break
        size = len(line.encode()) + 1
        if total + size > max_bytes:
            break
        kept.append(line)
        total += size

    result = "\n".join(kept)
    result += f"\n\n(Output truncated. Showed {len(kept)} of {len(lines)} lines.)"
    return result, True
