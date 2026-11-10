"""Bind every content identity to exactly one existing target below a namespace root.

Responsibilities:
  - Record which target owns a digest below a namespace root, relative to that root
  - Reject a fresh binding whose digest already owns another existing target
  - Replace entries whose target no longer exists, under a per-digest lock
  - Repair the missing binding of an existing, already admitted target

Design principles:
  - One JSON entry per digest, written atomically before its target is published
  - The caller's digest decides identity
  - Names, labels and paths never decide identity, so a relabelled copy is still a duplicate
  - Entries fail closed: a malformed entry or an escaping target is an error, never a free slot
  - The conflict exception belongs to the caller, so each domain reports duplicates in its own terms

This module does NOT:
  - Compute digests, create or delete targets, or decide which differences make content new
  - Validate the target's content; only its existence below the namespace root
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from . import common_locking as locking
from . import common_paths as paths
from . import common_serialization as serialization

_ENTRY_KEYS = frozenset({"digest", "target", "kind", "created_at"})

ConflictFactory = Callable[[Path, Path], Exception]


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One admitted binding of a digest to a target below the namespace root."""

    digest: str
    target: Path
    kind: str
    created_at: str


class IdentityRegistry:
    """Digest-to-target bindings of one namespace such as the runs or the Datasets of a storage root."""

    def __init__(self, root: Path | str, kinds: Collection[str]) -> None:
        """Scope the registry to a namespace root and the entry kinds it admits."""
        self.root = Path(root).expanduser().resolve(strict=False)
        self.kinds = frozenset(kinds)

    def _read(self, digest: str) -> RegistryEntry | None:
        """Load and admit one entry; malformed entries fail closed."""
        path = paths.resolve_identity_registry_entry(self.root, digest)
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            msg = f"Identity registry entry is not a regular file: {path}"
            raise RuntimeError(msg)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            msg = f"Identity registry entry is not valid JSON: {path}"
            raise RuntimeError(msg) from error
        relative = payload.get("target") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _ENTRY_KEYS
            or payload["digest"] != digest
            or payload["kind"] not in self.kinds
            or not isinstance(payload["created_at"], str)
            or not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            msg = f"Identity registry entry is invalid: {path}"
            raise RuntimeError(msg)
        return RegistryEntry(digest, self.root / PurePosixPath(relative), str(payload["kind"]), str(payload["created_at"]))

    def _contained(self, target: Path | str) -> Path:
        """Resolve a target and require it below the namespace root."""
        requested = Path(target).expanduser().resolve(strict=False)
        if not requested.is_relative_to(self.root):
            msg = f"Target {requested} lies outside its namespace root {self.root}."
            raise ValueError(msg)
        return requested

    def lookup(self, digest: str) -> RegistryEntry | None:
        """Return the current owner of a digest, or ``None`` if unowned or stale."""
        entry = self._read(digest)
        if entry is None or not entry.target.exists():
            return None
        return entry

    def ensure_available(self, digest: str, target: Path | str, *, conflict: ConflictFactory) -> None:
        """Fail fast, before any allocation, when another existing target owns the digest.

        Raises
        ------
        Exception
            ``conflict(existing, requested)`` if the digest is bound to an existing target other than ``target``.

        """
        requested = self._contained(target)
        entry = self.lookup(digest)
        if entry is not None and entry.target.resolve(strict=False) != requested:
            raise conflict(entry.target, requested)

    def _write(self, digest: str, requested: Path, kind: str) -> Path:
        """Atomically bind ``digest`` to ``requested`` and return the entry path."""
        entry_path = paths.resolve_identity_registry_entry(self.root, digest)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        serialization.atomic_write_json(
            entry_path,
            {
                "digest": digest,
                "target": requested.relative_to(self.root).as_posix(),
                "kind": kind,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return entry_path

    def _admitted_kind(self, kind: str) -> None:
        """Reject a kind this registry does not admit."""
        if kind not in self.kinds:
            msg = f"Identity kind {kind!r} is not one of {sorted(self.kinds)}."
            raise ValueError(msg)

    @contextmanager
    def reserve(self, digest: str, target: Path | str, *, kind: str, conflict: ConflictFactory) -> Iterator[None]:
        """Bind ``target`` before the caller publishes it, under the digest lock.

        The binding is written before the caller's block runs, so a crash after the
        publication can never leave an unbound target; a crash before it leaves an
        entry whose target does not exist, which counts as stale. If the block
        raises, the new binding is removed again. An entry naming another existing
        target rejects the start before the block runs.

        Raises
        ------
        ValueError
            If ``target`` lies outside the namespace root or ``kind`` is not admitted.
        Exception
            ``conflict(existing, requested)`` if another existing target already owns the digest.

        """
        requested = self._contained(target)
        self._admitted_kind(kind)
        with locking.exclusive_file_lock(paths.resolve_identity_registry_lock(self.root, digest), blocking=True):
            entry = self._read(digest)
            if entry is not None and entry.target.exists() and entry.target.resolve(strict=False) != requested:
                raise conflict(entry.target, requested)
            written = None if entry is not None and entry.target.resolve(strict=False) == requested else self._write(digest, requested, kind)
            try:
                yield
            except BaseException:
                if written is not None:
                    written.unlink(missing_ok=True)
                raise

    def ensure_bound(self, digest: str, target: Path | str, *, kind: str, conflict: ConflictFactory) -> None:
        """Bind an existing, already admitted target whose binding is missing or stale.

        Raises
        ------
        ValueError
            If ``target`` lies outside the namespace root, does not exist, or ``kind`` is not admitted.
        Exception
            ``conflict(existing, requested)`` if another existing target owns the digest.

        """
        requested = self._contained(target)
        self._admitted_kind(kind)
        if not requested.exists():
            msg = f"Only an existing target can be bound: {requested}"
            raise ValueError(msg)
        with locking.exclusive_file_lock(paths.resolve_identity_registry_lock(self.root, digest), blocking=True):
            entry = self._read(digest)
            if entry is not None and entry.target.exists():
                if entry.target.resolve(strict=False) != requested:
                    raise conflict(entry.target, requested)
                return
            self._write(digest, requested, kind)
