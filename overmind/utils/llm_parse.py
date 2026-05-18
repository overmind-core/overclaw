"""Single source of truth for parsing JSON out of LLM completion responses.

LLM outputs are notoriously inconsistent: a model that promised "return JSON
only" will happily return fenced ```json blocks, prose before the JSON, prose
after, unescaped newlines inside string literals, and trailing commas.  Every
call site in the codebase used to roll its own ``content.find("{")`` /
``rfind("}")`` snippet, with subtly different failure handling.

This module provides one function, :func:`parse_json_object`, that:

* Strips markdown fences.
* Tries ``json.loads`` on the full text, then on a brace-bounded slice
  (objects and arrays), then on a repaired version of those.
* Repairs common LLM mistakes (unescaped newlines/tabs inside strings,
  trailing commas, single-quoted string delimiters).
* Lets the caller choose between raising :class:`LLMParseError` and
  returning a sentinel/default when parsing fails.

Replaces the duplicated parsers previously in:

* ``overmind/setup/agent_analyzer.py``
* ``overmind/setup/questionnaire.py``
* ``overmind/setup/policy_generator.py``
* ``overmind/optimize/data.py`` (``_safe_parse_json`` + ``_repair_json_string``)
* ``overmind/optimize/runner.py`` (``_try_parse_json``)
* ``overmind/optimize/evaluator.py`` (3 inline copies)
* ``overmind/optimize/analyzer.py`` (3 inline copies)
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, overload

__all__ = ["LLMParseError", "parse_json_object", "repair_json_string"]


class LLMParseError(ValueError):
    """Raised by :func:`parse_json_object` when ``on_fail='raise'``.

    The original ``content`` is preserved on the exception so callers can
    show a truncated preview in their error message without re-fetching it.
    """

    def __init__(self, message: str, *, content: str = "") -> None:
        super().__init__(message)
        self.content = content


_FENCED_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def repair_json_string(text: str) -> Any:
    """Attempt to repair common LLM JSON formatting mistakes.

    Returns the parsed object on success, ``None`` if even the repaired
    text fails to parse.  Public so call sites that need a second-chance
    repair after their own pre-processing can reach for it directly.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
        i += 1
    repaired = "".join(result)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _try_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _bracket_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    end = text.rfind(close_ch)
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


@overload
def parse_json_object(content: str, *, on_fail: Literal["raise"] = "raise") -> Any: ...


@overload
def parse_json_object(content: str, *, on_fail: Literal["default"], default: Any = None) -> Any: ...


def parse_json_object(
    content: str,
    *,
    on_fail: Literal["raise", "default"] = "raise",
    default: Any = None,
) -> Any:
    """Best-effort extraction of a JSON value from an LLM completion.

    Parameters
    ----------
    content:
        Raw text from ``response.choices[0].message.content``.  May be
        ``None`` or empty, in which case the call always fails (raise or
        default per ``on_fail``).
    on_fail:
        ``"raise"`` — raise :class:`LLMParseError` if no JSON can be
        extracted.  ``"default"`` — return the ``default`` value instead.
    default:
        Returned only when ``on_fail='default'`` and parsing fails.

    Returns
    -------
    The parsed JSON value (typically ``dict`` or ``list``) on success.
    On failure, either raises or returns ``default``.
    """
    if not content or not isinstance(content, str):
        if on_fail == "raise":
            raise LLMParseError("LLM response was empty or non-string", content=content or "")
        return default

    text = content.strip()

    parsed = _try_loads(text)
    if parsed is not None:
        return parsed

    fenced = _FENCED_BLOCK.search(text)
    if fenced:
        inner = fenced.group(1).strip()
        parsed = _try_loads(inner)
        if parsed is not None:
            return parsed
        repaired = repair_json_string(inner)
        if repaired is not None:
            return repaired

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        candidate = _bracket_slice(text, open_ch, close_ch)
        if candidate is None:
            continue
        parsed = _try_loads(candidate)
        if parsed is not None:
            return parsed
        repaired = repair_json_string(candidate)
        if repaired is not None:
            return repaired

    quoted = text.replace("'", '"')
    quoted = re.sub(r",\s*([}\]])", r"\1", quoted)
    parsed = _try_loads(quoted)
    if parsed is not None:
        return parsed

    if on_fail == "raise":
        raise LLMParseError("No parseable JSON found in LLM response", content=content)
    return default
