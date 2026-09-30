"""Small on-disk cache (JSON entries + pickled blobs).

The cache is only an optimisation: a missing, corrupt or unwritable
cache never breaks a command, it is just ignored.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any


def make_key(*parts: str) -> str:
    """Build a cache key from the values that determine a result.

    Args:
        *parts: Strings identifying the computation.

    Returns:
        A SHA-256 hexadecimal digest.
    """
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def index_version(index_dir: Path) -> str:
    """Identify the current state of an index.

    Any ``index`` or ``update`` that changes the corpus changes this
    value, which invalidates every cache entry that depends on it.

    Args:
        index_dir: Index directory.

    Returns:
        A short version string.
    """
    manifest = index_dir / "manifest.json"
    if manifest.is_file():
        return hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    stat = (index_dir / "chunks.json").stat()
    return f"{stat.st_size}-{int(stat.st_mtime)}"


def _warn(message: str) -> None:
    """Print a cache warning without failing."""
    print(f"[WARN] cache: {message}", file=sys.stderr)


class DiskCache:
    """Key/value cache stored as files in a directory."""

    def __init__(self, directory: str, max_entries: int = 5000) -> None:
        """Create the cache handle (the directory is created lazily).

        Args:
            directory: Cache directory.
            max_entries: Maximum number of JSON entries kept by ``prune``.
        """
        self.directory = Path(directory)
        self.max_entries = max_entries

    def _write(self, path: Path, data: bytes) -> None:
        """Atomically write bytes (temp file + rename)."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except OSError as exc:
            _warn(f"cannot write {path.name}: {exc}")

    def get(self, key: str) -> Any:
        """Read a JSON entry.

        Args:
            key: Entry key.

        Returns:
            The stored value, or None if missing or unreadable.
        """
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            _warn(f"ignoring corrupt entry {path.name}")
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a JSON-serialisable value.

        Args:
            key: Entry key.
            value: Value to store.
        """
        self._write(
            self.directory / f"{key}.json",
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
        )

    def load_pickle(self, name: str) -> Any:
        """Read a pickled blob.

        Args:
            name: Blob name.

        Returns:
            The object, or None if missing or unreadable.
        """
        path = self.directory / f"{name}.pkl"
        if not path.is_file():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            _warn(f"ignoring corrupt blob {path.name}")
            path.unlink(missing_ok=True)
            return None

    def save_pickle(self, name: str, obj: Any) -> None:
        """Store a picklable object.

        Args:
            name: Blob name.
            obj: Object to store.
        """
        self._write(
            self.directory / f"{name}.pkl",
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def purge(self, prefix: str, keep: str) -> None:
        """Delete stale blobs ``<prefix>*.pkl`` except ``keep``.

        Args:
            prefix: Blob name prefix.
            keep: Blob name to keep.
        """
        if not self.directory.is_dir():
            return
        for path in self.directory.glob(f"{prefix}*.pkl"):
            if path.stem != keep:
                path.unlink(missing_ok=True)

    def prune(self) -> None:
        """Drop the oldest JSON entries beyond ``max_entries``."""
        if not self.directory.is_dir():
            return
        files = sorted(
            self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime
        )
        for path in files[: max(0, len(files) - self.max_entries)]:
            path.unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every cache file (never other files).

        Returns:
            Number of deleted files.
        """
        if not self.directory.is_dir():
            return 0
        count = 0
        for pattern in ("*.json", "*.pkl", "*.tmp"):
            for path in self.directory.glob(pattern):
                path.unlink(missing_ok=True)
                count += 1
        return count
