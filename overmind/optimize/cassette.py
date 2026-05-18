"""Record/replay cassette path holder + content-addressable key helper.

A cassette is an on-disk, append-only JSONL file of external interactions
(LLM completions, intercepted tool calls, browser/network calls) keyed by
a stable hash of ``(kind, identifier, payload)``.

The actual record/replay logic lives in
:mod:`overmind.optimize.shadow_runtime`, which runs **inside the
subprocess** and reads/writes the JSONL file directly with pure stdlib
so it has no Overmind import dependencies.  The orchestrating layer
(``overmind.optimize``) only needs a tiny path-holder to thread the
cassette file path through ``ExecutionBackend`` instances, plus the
shared :func:`make_key` helper so out-of-subprocess code can compute
stable keys for diagnostics.

Historical note: this module previously exposed a full in-Python record /
replay API (``Cassette.record``, ``Cassette.replay``, ``NullCassette``,
``__len__``, etc.).  Production never called any of those methods —
every real read/write happens inside the subprocess — so the API was
deleted, leaving only the small surface that production actually uses.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CASSETTE_FORMAT_VERSION = 1


@dataclass
class CassetteEntry:
    """A single recorded call.

    Public for backward-compatibility with anyone introspecting cassette
    files; the subprocess shadow runtime writes the same on-disk schema.

    Attributes:
        kind:        "llm" | "tool" | "http" | "subprocess" | ...
        identifier:  A stable, caller-chosen string identifying *which* call
                     site this is (e.g. model name for LLM, tool name for
                     tool calls, URL for HTTP).
        key:         Stable hash of (kind, identifier, payload) — used for
                     lookup.
        payload:     The inputs to the call (messages, args, request body,
                     ...).  Stored verbatim so a human can inspect the
                     cassette.
        result:      The recorded result.  Must be JSON-serialisable.
        metadata:    Free-form dict for diagnostic info (latency, tokens,
                     cost, …).  Optional.
    """

    kind: str
    identifier: str
    key: str
    payload: Any
    result: Any
    metadata: dict = field(default_factory=dict)
    version: int = _CASSETTE_FORMAT_VERSION


def _canonical_json(obj: Any) -> str:
    """Return a canonical JSON encoding of *obj* for hashing."""
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


def make_key(kind: str, identifier: str, payload: Any) -> str:
    """Compute the cassette lookup key for a call.

    The key is a hex SHA-256 of ``kind|identifier|canonical(payload)``.
    Stable across processes — used by the subprocess shadow runtime when
    deciding whether a request is a replay hit.
    """
    blob = "|".join([kind, identifier, _canonical_json(payload)])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Cassette:
    """Path-holder for an on-disk cassette file (or no-op when ``path is None``).

    The orchestrating code passes this object through
    :class:`overmind.optimize.execution_backend.SubprocessBackend` /
    :class:`ShadowBackend`, which read :attr:`path` and forward it to
    the subprocess via the ``OVERMIND_CASSETTE_FILE`` env var.  All
    actual record / replay happens inside the subprocess (see
    :mod:`overmind.optimize.shadow_runtime`).
    """

    path: Path | None = None


def open_cassette(path: str | os.PathLike | None) -> Cassette:
    """Return a :class:`Cassette` bound to *path*, or path-less when ``None``."""
    if not path:
        return Cassette(path=None)
    return Cassette(path=Path(path))
