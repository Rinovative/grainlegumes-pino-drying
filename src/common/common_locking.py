"""Coordinate exclusive filesystem writers with process-scoped advisory locks.

Responsibilities:
  - Acquire shared reader locks and re-entrant exclusive writer locks on persistent lock files
  - Support blocking serialization and fail-fast writer leases
  - Release locks automatically on exceptions and process exit
  - Close inherited lock descriptors in forked child processes

Design principles:
  - Lock files live outside any target directory a caller may delete
  - Kernel-owned locks, rather than file existence, define lease ownership
  - Re-entrancy is limited to the owning process thread and exact lock path

This module does NOT:
  - Delete persistent lock files or infer ownership from their existence
  - Coordinate threads through an in-memory mutex in place of a kernel lock
  - Make a multi-file operation transactional beyond the caller-held context
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager


class FileLockUnavailableError(RuntimeError):
    """
    Report fail-fast contention at the public file-lock boundary.

    ``exclusive_file_lock(..., blocking=False)`` raises this project-specific
    error only when another thread or process owns the advisory lock. Filesystem
    creation, descriptor, and kernel failures remain ``OSError`` instances.
    """


class _ThreadLockState(threading.local):
    """Per-thread re-entrant lock ownership."""

    def __init__(self) -> None:
        self.locks: set[Path] = set()


_THREAD_LOCK_STATE = _ThreadLockState()
_OPEN_LOCK_DESCRIPTORS: set[int] = set()
_OPEN_LOCK_DESCRIPTORS_GUARD = threading.Lock()


def _before_fork() -> None:
    """Freeze descriptor registration before the process descriptor table forks."""
    _OPEN_LOCK_DESCRIPTORS_GUARD.acquire()


def _after_fork_parent() -> None:
    """Resume descriptor registration in the parent after a fork."""
    _OPEN_LOCK_DESCRIPTORS_GUARD.release()


def _after_fork_child() -> None:
    """
    Remove inherited parent lock ownership from the forked child process.

    Every registered descriptor is closed best effort, global and thread-local
    ownership registries are cleared, and the inherited registration guard is
    released so the child can acquire independent locks safely.
    """
    descriptors = tuple(_OPEN_LOCK_DESCRIPTORS)
    _OPEN_LOCK_DESCRIPTORS.clear()
    for descriptor in descriptors:
        with suppress(OSError):
            os.close(descriptor)
    _THREAD_LOCK_STATE.locks.clear()
    _OPEN_LOCK_DESCRIPTORS_GUARD.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)


def _open_lock_file(path: Path) -> int:
    """
    Open a non-inheritable regular lock file without following a final symlink.

    The descriptor is opened read/write with owner-only permissions when the
    file is created. A non-regular target is rejected and the descriptor is
    closed before the originating ``OSError`` reaches the lock boundary.
    """
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        msg = f"Lock path is not a regular file: {path}"
        raise OSError(msg)
    return descriptor


def file_lock_is_active(path: Path | str) -> bool:
    """Return whether an existing regular lock file is currently held, without creating it."""
    lock_path = Path(path).expanduser().absolute()
    try:
        marker_mode = lock_path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(marker_mode):
        msg = f"Lock path is not a regular file: {lock_path}"
        raise OSError(msg)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            msg = f"Lock path is not a regular file: {lock_path}"
            raise OSError(msg)
        try:
            # Every writer lease is exclusive; a shared probe also works on a
            # readable marker that cannot be opened for writing.
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return True
            raise
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


@contextmanager
def _hold_file_lock(
    path: Path | str,
    *,
    operation: int,
    blocking: bool,
    contention_message: str,
) -> Iterator[Path]:
    """Hold one advisory lock of the given ``flock`` kind; re-entry in the owning thread yields at once."""
    lock_path = Path(path).expanduser().resolve(strict=False)
    if lock_path in _THREAD_LOCK_STATE.locks:
        yield lock_path
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _OPEN_LOCK_DESCRIPTORS_GUARD:
        descriptor = _open_lock_file(lock_path)
        _OPEN_LOCK_DESCRIPTORS.add(descriptor)
    try:
        fcntl.flock(descriptor, operation if blocking else operation | fcntl.LOCK_NB)
    except OSError as error:
        with _OPEN_LOCK_DESCRIPTORS_GUARD:
            _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
            os.close(descriptor)
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            msg = f"{contention_message}: {lock_path}"
            raise FileLockUnavailableError(msg) from error
        raise

    _THREAD_LOCK_STATE.locks.add(lock_path)
    try:
        yield lock_path
    finally:
        _THREAD_LOCK_STATE.locks.remove(lock_path)
        with _OPEN_LOCK_DESCRIPTORS_GUARD:
            _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def shared_file_lock(
    path: Path | str,
    *,
    blocking: bool,
) -> AbstractContextManager[Path]:
    """Hold a shared advisory reader lock, optionally failing on an active writer."""
    return _hold_file_lock(
        path,
        operation=fcntl.LOCK_SH,
        blocking=blocking,
        contention_message="Shared file lock is blocked by an active writer",
    )


def exclusive_file_lock(
    path: Path | str,
    *,
    blocking: bool,
) -> AbstractContextManager[Path]:
    """
    Hold an exclusive advisory lock for the duration of one context.

    Parameters
    ----------
    path : Path | str
        Persistent lock-file path. Its parent directory is created.
    blocking : bool
        Wait for the current owner when true. Fail immediately when false.

    Yields
    ------
    pathlib.Path
        Canonical lock-file path.

    Raises
    ------
    FileLockUnavailableError
        If a nonblocking lock is held by another process or thread.
    OSError
        If the lock directory or regular lock file cannot be created, opened,
        locked, unlocked, or closed.

    Notes
    -----
    Re-entry is recognized only for the same canonical path in the owning
    thread. The persistent lock file is not deleted on release. Kernel lock
    ownership, not file existence, is authoritative. The descriptor and lock
    are released when the context exits, including during exception unwinding.

    """
    return _hold_file_lock(
        path,
        operation=fcntl.LOCK_EX,
        blocking=blocking,
        contention_message="Exclusive file lock is already held",
    )
