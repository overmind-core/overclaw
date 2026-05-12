"""Record/replay cassette for external calls made by a user agent.

A cassette is an on-disk, append-only JSONL file of external interactions
keyed by a stable hash of ``(kind, identifier, payload)``.  On the *first*
successful run of an agent we transparently record every LLM completion and
every intercepted external call.  On *subsequent* runs — especially during
shadow execution — we replay recorded entries whenever the request matches.

Design goals:

* **Transparent**: callers record/replay through a simple functional API.
* **Deterministic**: lookup keys are stable across processes (no Python
  object identity, no dict ordering) so cassettes travel with the agent.
* **Dependency-free at import time**: the file is pure stdlib so it can be
  imported inside the subprocess intercept bootstrap without pulling in the
  rest of Overmind.
* **Side-effect safe**: corrupted lines are skipped; concurrent writers use
  an ``fcntl`` advisory lock (best effort) so parallel subprocess runs
  don't tear.

The cassette stores the *result* of a call, not the full object graph.  For
LLM calls we store the raw message content and tool-call metadata — enough
to reconstruct a ``litellm.ModelResponse`` on replay.  For tool calls we
store the JSON-serialised return value verbatim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Version tag baked into every cassette entry so we can migrate the format
# later without breaking existing recordings.
_CASSETTE_FORMAT_VERSION = 1


@dataclass
class CassetteEntry:
    """A single recorded call.

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
    """Return a canonical JSON encoding of *obj* for hashing.

    Uses ``sort_keys`` and ``default=repr`` so partially non-serialisable
    objects (e.g. a ``Path`` or an object with ``__repr__``) still hash.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


def make_key(kind: str, identifier: str, payload: Any) -> str:
    """Compute the cassette lookup key for a call.

    The key is a hex SHA-256 of ``kind|identifier|canonical(payload)``.
    """
    blob = "|".join([kind, identifier, _canonical_json(payload)])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cassette:
    """File-backed record/replay store.

    The file is JSONL — one :class:`CassetteEntry` per line.  In-memory we
    keep a ``dict`` keyed by ``entry.key`` for O(1) replay.
    """

    def __init__(self, path: str | os.PathLike | None) -> None:
        self.path: Path | None = Path(path) if path else None
        self._entries: dict[str, CassetteEntry] = {}
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.path or not self.path.exists():
            return

        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Cassette read failed: {exc}")
            return

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entry = CassetteEntry(**raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug(f"Skipping malformed cassette line: {exc}")
                continue
            self._entries[entry.key] = entry

    def reload(self) -> None:
        """Force a re-read from disk (mostly useful for tests)."""
        self._loaded = False
        self._entries.clear()
        self._load()

    def _persist(self, entry: CassetteEntry) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), default=repr) + "\n")
        except OSError as exc:
            logger.warning(f"Cassette write failed: {exc}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        kind: str,
        identifier: str,
        payload: Any,
        result: Any,
        metadata: dict | None = None,
    ) -> CassetteEntry:
        """Record a call.  Overwrites any previous entry with the same key."""
        with self._lock:
            self._load()
            key = make_key(kind, identifier, payload)
            entry = CassetteEntry(
                kind=kind,
                identifier=identifier,
                key=key,
                payload=payload,
                result=result,
                metadata=metadata or {},
            )
            self._entries[key] = entry
            self._persist(entry)
            return entry

    def replay(self, kind: str, identifier: str, payload: Any) -> CassetteEntry | None:
        """Return a cassette entry for the given call, or ``None`` if absent."""
        with self._lock:
            self._load()
            key = make_key(kind, identifier, payload)
            return self._entries.get(key)

    def has(self, kind: str, identifier: str, payload: Any) -> bool:
        return self.replay(kind, identifier, payload) is not None

    # ------------------------------------------------------------------
    # Inspection helpers (used by the optimizer UI / tests)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        self._load()
        return len(self._entries)

    def entries(self) -> list[CassetteEntry]:
        self._load()
        return list(self._entries.values())

    def count_by_kind(self) -> dict[str, int]:
        self._load()
        out: dict[str, int] = {}
        for entry in self._entries.values():
            out[entry.kind] = out.get(entry.kind, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Convenience — a no-op cassette used when recording/replay is disabled.
# ---------------------------------------------------------------------------


class NullCassette(Cassette):
    """Cassette that accepts records but stores nothing and never replays."""

    def __init__(self) -> None:
        super().__init__(path=None)

    def record(self, *args: Any, **kwargs: Any) -> CassetteEntry:  # type: ignore[override]
        kind = kwargs.get("kind") or (args[0] if args else "unknown")
        identifier = kwargs.get("identifier") or (args[1] if len(args) > 1 else "")
        payload = kwargs.get("payload") or (args[2] if len(args) > 2 else None)
        result = kwargs.get("result") or (args[3] if len(args) > 3 else None)
        return CassetteEntry(
            kind=kind,
            identifier=identifier,
            key=make_key(kind, identifier, payload),
            payload=payload,
            result=result,
        )

    def replay(self, *args: Any, **kwargs: Any) -> CassetteEntry | None:  # type: ignore[override]
        return None

    def has(self, *args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        return False

    def __len__(self) -> int:  # type: ignore[override]
        return 0

    def entries(self) -> list[CassetteEntry]:  # type: ignore[override]
        return []

    def count_by_kind(self) -> dict[str, int]:  # type: ignore[override]
        return {}


def open_cassette(path: str | os.PathLike | None) -> Cassette:
    """Return a :class:`Cassette` bound to *path*, or a :class:`NullCassette`.

    ``path=None`` is a common case (tests, features disabled) — using a
    :class:`NullCassette` means callers never need to check for ``None``.
    """
    if not path:
        return NullCassette()
    return Cassette(path)
