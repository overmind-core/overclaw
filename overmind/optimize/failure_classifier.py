"""Classify subprocess failures into actionable :class:`FailureMode` values.

When an agent run fails, the optimizer needs to decide whether the failure is
*recoverable* (try a different backend, re-run with a guard) or *fatal* (stop
optimizing and report to the user).  Historically the optimizer treated every
failure as an opaque error string and kept iterating, wasting budget on what
were really environment problems.

This module parses stderr, exit codes, and timeout flags into a
:class:`FailureMode` enum with a recommended *next backend* and a human
remediation hint.  It is intentionally dependency-free so it can be unit
tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    """Category of subprocess failure.

    Each mode encodes what *kind* of problem we're dealing with and what the
    runner should try next.
    """

    NONE = "none"
    """No failure detected; run succeeded."""

    IMPORT_ERROR = "import_error"
    """User's agent file could not be imported (missing dep, syntax error,
    relative import path, etc.).  Fixable at the environment or wrapper
    level."""

    MODULE_LEVEL_CRASH = "module_level_crash"
    """Top-level code in the agent module raised (e.g. argparse.parse_args
    reading our wrapper's sys.argv, load_dotenv failing)."""

    MISSING_DEPENDENCY = "missing_dependency"
    """Specific module not found during import — classic ModuleNotFoundError
    with a package name the user hasn't installed."""

    API_KEY_MISSING = "api_key_missing"
    """LLM or API call failed for lack of credentials."""

    BROWSER_RUNTIME_ERROR = "browser_runtime_error"
    """Browser/Playwright specific failure: binary missing, display missing,
    element not found, timeout on navigation."""

    NETWORK_ERROR = "network_error"
    """Generic network failure (DNS, connection refused, SSL)."""

    TIMEOUT = "timeout"
    """Subprocess exceeded the configured timeout."""

    SCHEMA_MISMATCH = "schema_mismatch"
    """Agent was invoked with the wrong argument shape (kwargs vs dict vs
    positional)."""

    CRASH = "crash"
    """Generic unclassified crash (uncaught exception, non-zero exit without
    a recognisable signature)."""

    UNKNOWN = "unknown"
    """We couldn't read stderr or there was no signal at all."""


@dataclass(frozen=True)
class FailureDiagnosis:
    """Outcome of classifying a failed run."""

    mode: FailureMode
    summary: str
    remediation: str
    retryable: bool
    """Whether a different backend or retry might succeed."""
    excerpt: str = ""
    """Short trimmed excerpt of the stderr/error message useful for logs."""


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------
# Each entry: (FailureMode, compiled pattern, summary template, remediation,
# retryable)  Patterns are applied in order; first match wins.

_PATTERNS: list[tuple[FailureMode, re.Pattern[str], str, str, bool]] = [
    (
        FailureMode.API_KEY_MISSING,
        re.compile(
            r"(OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|"
            r"missing[_ ]api[_ ]key|AuthenticationError|401.*Unauthorized|"
            r"invalid[_ ]api[_ ]key|Incorrect API key)",
            re.IGNORECASE,
        ),
        "Agent failed because an LLM provider API key was missing or invalid.",
        "Add the required API key to .overmind/.env (run `overmind init`) or to the shell environment. "
        "Overmind will also try shadow execution with cassette replay — if "
        "the LLM call was recorded on a prior successful run, optimization "
        "can still proceed without a live key.",
        True,
    ),
    (
        FailureMode.BROWSER_RUNTIME_ERROR,
        re.compile(
            r"(playwright[^\n]*(?:not installed|Executable doesn't exist)|"
            r"browserType\.launch|chromium[^\n]*not found|"
            r"browser[_ ]use[^\n]*error|webdriver|selenium\.common\.exceptions|"
            r"no such element|DISPLAY environment variable)",
            re.IGNORECASE,
        ),
        "Browser runtime failed — the headless browser could not be started or an automation step errored.",
        "Install the browser (e.g. ``playwright install chromium``) or run "
        "the agent in a container with the browser preinstalled.  Overmind "
        "will fall back to shadow execution with simulated browser responses.",
        True,
    ),
    (
        FailureMode.MISSING_DEPENDENCY,
        re.compile(
            r"(ModuleNotFoundError: No module named ['\"]([\w_.-]+)['\"]|"
            r"ImportError: cannot import name ['\"]([\w_.-]+)['\"])",
        ),
        "A Python dependency is missing from the agent's environment.",
        "Add the missing package to requirements.txt / pyproject.toml and let Overmind reprovision the environment.",
        True,
    ),
    (
        FailureMode.MODULE_LEVEL_CRASH,
        re.compile(
            r"(argparse[^\n]*(?:error|SystemExit)|"
            r"SystemExit: \d+|"
            r"argument --[\w-]+ is required|"
            r"unrecognized arguments|"
            r"the following arguments are required)",
            re.IGNORECASE,
        ),
        "The agent's top-level code executed on import and crashed (typically "
        "argparse reading the wrapper's sys.argv, or a side-effectful init).",
        "Overmind will retry with an argument-parser guard; if that still "
        "fails, move the CLI parsing behind an ``if __name__ == '__main__':`` "
        "block.",
        True,
    ),
    (
        FailureMode.SCHEMA_MISMATCH,
        re.compile(
            r"(TypeError:[^\n]*(?:positional argument|unexpected keyword argument|"
            r"missing \d+ required positional)|"
            r"takes \d+ positional arguments but \d+ were given)",
        ),
        "The agent's function signature doesn't match the dataset's input "
        "shape (Overmind picked the wrong calling convention).",
        "Overmind will retry using a schema-driven dispatcher.  If the "
        "problem persists, ensure your entrypoint function accepts a single "
        "``dict`` parameter or that the dataset keys match your function's "
        "parameter names exactly.",
        True,
    ),
    (
        FailureMode.NETWORK_ERROR,
        re.compile(
            r"(ConnectionError|ConnectionRefused|Temporary failure in name "
            r"resolution|SSL[^\n]*(?:verify|handshake)|"
            r"urllib3[^\n]*MaxRetryError|requests\.exceptions\.ConnectionError)",
            re.IGNORECASE,
        ),
        "The agent could not reach a network endpoint (DNS / connection / SSL failure).",
        "Check the agent's internet access.  Overmind will fall back to "
        "shadow execution with cassette replay for network calls.",
        True,
    ),
    (
        FailureMode.IMPORT_ERROR,
        re.compile(
            r"(ImportError|SyntaxError|IndentationError|"
            r"attempted relative import with no known parent package)",
        ),
        "The agent module failed to import (syntax error or broken import statement).",
        "Fix the import / syntax error surfaced in stderr.  Shadow execution cannot help here — the code must parse.",
        False,
    ),
]


_EXCERPT_MAX_LEN = 600


def _excerpt(text: str, max_len: int = _EXCERPT_MAX_LEN) -> str:
    """Return the most informative tail of *text* for logs."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return "…" + text[-max_len:]


def classify_failure(
    *,
    stderr: str | None,
    returncode: int = 0,
    timed_out: bool = False,
    error: str | None = None,
) -> FailureDiagnosis:
    """Classify a failure from its ``stderr`` / ``returncode`` / ``timed_out``.

    Returns :attr:`FailureMode.NONE` when no failure is present.
    """
    if returncode == 0 and not timed_out and not error:
        return FailureDiagnosis(
            mode=FailureMode.NONE,
            summary="Run succeeded.",
            remediation="",
            retryable=False,
        )

    if timed_out:
        return FailureDiagnosis(
            mode=FailureMode.TIMEOUT,
            summary="Agent exceeded its execution timeout.",
            remediation=(
                "Raise the per-agent timeout for slow frameworks (browser, "
                "multi-step autonomous) or enable shadow execution so "
                "Overmind can proceed without live external calls."
            ),
            retryable=True,
            excerpt=_excerpt(stderr or ""),
        )

    # Combine error + stderr for pattern matching — some failures only surface
    # in the short ``error`` string set by the runner.
    haystack = "\n".join(filter(None, [error or "", stderr or ""]))
    if not haystack.strip():
        return FailureDiagnosis(
            mode=FailureMode.UNKNOWN,
            summary="Agent failed with no diagnostic output.",
            remediation=(
                "Re-run with verbose logging or attach a fresh stderr stream. "
                "Overmind will fall back to shadow execution."
            ),
            retryable=True,
        )

    for mode, pattern, summary, remediation, retryable in _PATTERNS:
        if pattern.search(haystack):
            return FailureDiagnosis(
                mode=mode,
                summary=summary,
                remediation=remediation,
                retryable=retryable,
                excerpt=_excerpt(haystack),
            )

    return FailureDiagnosis(
        mode=FailureMode.CRASH,
        summary="Agent crashed with an unrecognised error.",
        remediation=(
            "Overmind will retry with a hardened wrapper; if that also "
            "fails, inspect stderr and consider running the agent manually."
        ),
        retryable=True,
        excerpt=_excerpt(haystack),
    )


def is_recoverable_via_shadow(diagnosis: FailureDiagnosis) -> bool:
    """Decide whether a failure can be worked around with shadow execution.

    Shadow execution re-runs the agent with external calls intercepted and
    replayed / simulated.  It helps with network / browser / timeout / auth
    failures (when a cassette hit bypasses the real external call), but
    cannot fix syntax errors, missing local modules, or a broken import
    graph — those all fail the same way inside shadow mode.
    """
    return diagnosis.mode in {
        FailureMode.TIMEOUT,
        FailureMode.BROWSER_RUNTIME_ERROR,
        FailureMode.NETWORK_ERROR,
        FailureMode.MODULE_LEVEL_CRASH,
        FailureMode.SCHEMA_MISMATCH,
        FailureMode.API_KEY_MISSING,
        FailureMode.CRASH,
        FailureMode.UNKNOWN,
    }
