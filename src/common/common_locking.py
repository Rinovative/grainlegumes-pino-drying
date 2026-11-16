"""Coordinate exclusive filesystem writers with advisory locks that also exclude across the nodes of a shared filesystem.

Responsibilities:
  - Acquire shared reader locks and re-entrant exclusive writer locks on persistent lock files
  - Support blocking serialization and fail-fast writer leases
  - Release locks automatically on exceptions and process exit
  - Close inherited lock descriptors in forked child processes

Design principles:
  - Lock files live outside any target directory a caller may delete
  - Kernel-owned locks, rather than file existence, define lease ownership
  - Re-entrancy is limited to the owning process thread and exact lock path
  - Locks are whole-file open-file-description (OFD) locks, so they exclude across NFS clients and the file server

This module does NOT:
  - Delete persistent lock files or infer ownership from their existence
  - Coordinate threads through an in-memory mutex in place of a kernel lock
  - Make a multi-file operation transactional beyond the caller-held context
  - Guess the ``struct flock`` layout of a platform other than Linux x86_64; there it fails closed

Notes:
  Why OFD locks:
    - On ICE the storage is local ZFS on the login node and NFS 4.2 on the compute nodes. An NFS client forwards
      ``flock`` to the server as a POSIX lock, which does not conflict with a ``flock`` held on the server itself
      (measured: a compute node acquired an exclusive ``flock`` while the login node held one).
    - OFD locks are POSIX-family locks, so they conflict with the POSIX locks the server holds for its clients
      (measured: a login-node OFD lock blocked OFD and ``lockf`` on a compute node).
    - Like ``flock`` they belong to the open file description: another descriptor, thread, or process conflicts, and
      closing an unrelated descriptor keeps the lock.

"""

from __future__ import annotations

import errno
import fcntl
import os
import platform
import stat
import struct
import sys
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

# struct flock on Linux x86_64: l_type, l_whence, padding, l_start, l_len, l_pid, padding (32 bytes).
_FLOCK_LAYOUT: Final = "hh4xqqi4x"
_OFD_LOCKS_SUPPORTED: Final = sys.platform == "linux" and platform.machine() == "x86_64" and hasattr(fcntl, "F_OFD_SETLK")


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


def _require_ofd_locks() -> None:
    """Fail closed on a platform whose OFD lock commands or ``struct flock`` layout this module does not know."""
    if not _OFD_LOCKS_SUPPORTED:
        msg = f"File locks require Linux x86_64 open-file-description locks, not {sys.platform} {platform.machine()}."
        raise RuntimeError(msg)


def _whole_file_lock(kind: int) -> bytes:
    """Return the ``struct flock`` of one whole-file request of ``kind``; an OFD request must carry ``l_pid`` 0."""
    return struct.pack(_FLOCK_LAYOUT, kind, os.SEEK_SET, 0, 0, 0)


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
    """Return whether an existing regular lock file is currently held by a writer, without creating it."""
    _require_ofd_locks()
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
        # Every writer lease is exclusive; a shared probe reports only a writer and also works on a read-only marker.
        probe = fcntl.fcntl(descriptor, fcntl.F_OFD_GETLK, _whole_file_lock(fcntl.F_RDLCK))
        return struct.unpack(_FLOCK_LAYOUT, probe)[0] != fcntl.F_UNLCK
    finally:
        os.close(descriptor)


@contextmanager
def _hold_file_lock(
    path: Path | str,
    *,
    kind: int,
    blocking: bool,
    contention_message: str,
) -> Iterator[Path]:
    """Hold one whole-file OFD lock of ``kind`` (``F_RDLCK`` or ``F_WRLCK``); re-entry in the owning thread yields at once."""
    _require_ofd_locks()
    lock_path = Path(path).expanduser().resolve(strict=False)
    if lock_path in _THREAD_LOCK_STATE.locks:
        yield lock_path
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _OPEN_LOCK_DESCRIPTORS_GUARD:
        descriptor = _open_lock_file(lock_path)
        _OPEN_LOCK_DESCRIPTORS.add(descriptor)
    try:
        fcntl.fcntl(descriptor, fcntl.F_OFD_SETLKW if blocking else fcntl.F_OFD_SETLK, _whole_file_lock(kind))
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
                fcntl.fcntl(descriptor, fcntl.F_OFD_SETLK, _whole_file_lock(fcntl.F_UNLCK))
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
        kind=fcntl.F_RDLCK,
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
        kind=fcntl.F_WRLCK,
        blocking=blocking,
        contention_message="Exclusive file lock is already held",
    )
