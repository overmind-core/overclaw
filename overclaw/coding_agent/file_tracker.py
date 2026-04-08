"""File staleness tracker — prevents editing files that changed since last read.

Tracks per-session file read timestamps and asserts freshness before edits,
preventing stale overwrites.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class _Stamp:
    read_at: datetime
    mtime: float | None
    size: int | None


def _stat(path: str) -> _Stamp:
    try:
        s = os.stat(path)
        return _Stamp(
            read_at=datetime.now(timezone.utc),
            mtime=s.st_mtime,
            size=s.st_size,
        )
    except OSError:
        return _Stamp(read_at=datetime.now(timezone.utc), mtime=None, size=None)


class FileTracker:
    """Tracks which files have been read and their modification state."""

    def __init__(self) -> None:
        self._reads: dict[str, dict[str, _Stamp]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def lock_for(self, path: str) -> threading.Lock:
        with self._global_lock:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    def record_read(self, session: str, path: str) -> None:
        if session not in self._reads:
            self._reads[session] = {}
        self._reads[session][path] = _stat(path)

    def was_read(self, session: str, path: str) -> bool:
        return path in self._reads.get(session, {})

    def assert_fresh(self, session: str, path: str) -> None:
        """Raise if the file was never read or was modified since last read."""
        stamps = self._reads.get(session, {})
        prev = stamps.get(path)
        if prev is None:
            raise FileNotReadError(path)
        cur = _stat(path)
        if cur.mtime != prev.mtime or cur.size != prev.size:
            raise FileStaleError(path, prev.read_at, cur.mtime)


class FileNotReadError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"You must read file {path} before overwriting it. Use the read tool first."
        )


class FileStaleError(Exception):
    def __init__(self, path: str, read_at: datetime, mtime: float | None) -> None:
        modified = (
            datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            if mtime
            else "unknown"
        )
        super().__init__(
            f"File {path} has been modified since it was last read.\n"
            f"Last modification: {modified}\n"
            f"Last read: {read_at.isoformat()}\n\n"
            f"Please read the file again before modifying it."
        )
