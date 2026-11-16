"""Protect the exclusion of the shared file locks against the POSIX locks a file server holds for other nodes.

Responsibilities:
  - Check that a held exclusive or shared lock blocks another process's POSIX write lock until it is released
  - Check that another process's POSIX write lock makes both fail-fast leases report contention and the probe report a writer

Design principles:
  - Another process's ``lockf`` lock stands in for the POSIX lock the NFS server holds on behalf of a compute node
  - Child processes run only the standard library, so each holds or tries exactly one kernel lock

This module does NOT:
  - Reach a second node; the cross-node acceptance runs as a Slurm job against a lock held on the login node
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from src import common

if TYPE_CHECKING:
    from pathlib import Path

_BLOCKED_EXIT = 3
_TRY_LOCKF = (
    "import fcntl, os, sys\n"
    "descriptor = os.open(sys.argv[1], os.O_RDWR)\n"
    "try:\n"
    "    fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
    "except OSError:\n"
    f"    sys.exit({_BLOCKED_EXIT})\n"
)
_HOLD_LOCKF = (
    "import fcntl, os, sys\n"
    "descriptor = os.open(sys.argv[1], os.O_RDWR)\n"
    "fcntl.lockf(descriptor, fcntl.LOCK_EX)\n"
    "print('held', flush=True)\n"
    "sys.stdin.read()\n"
)


def _another_process_takes_lockf(path: Path) -> bool:
    """Return whether another process acquires a non-blocking POSIX write lock on ``path``."""
    result = subprocess.run([sys.executable, "-I", "-c", _TRY_LOCKF, str(path)], check=False, timeout=60)  # noqa: S603 -- test-owned probe
    assert result.returncode in {0, _BLOCKED_EXIT}
    return result.returncode == 0


@pytest.mark.parametrize("kind", ["exclusive", "shared"])
def test_a_held_lock_blocks_the_posix_lock_the_file_server_takes_for_another_node(tmp_path: Path, kind: str) -> None:
    """While this process holds the lock, another process's lockf fails, as the server's lock for a compute node must; release frees it."""
    path = tmp_path / "locks" / "campaign.lock"
    hold = common.locking.exclusive_file_lock if kind == "exclusive" else common.locking.shared_file_lock
    with hold(path, blocking=True):
        assert not _another_process_takes_lockf(path)
    assert _another_process_takes_lockf(path)


def test_a_posix_lock_of_another_process_is_contention_for_every_lease_and_the_probe(tmp_path: Path) -> None:
    """A writer lock held for another node is seen here: both fail-fast leases raise and the probe reports a writer until it ends."""
    path = tmp_path / "campaign.lock"
    path.touch()
    holder = subprocess.Popen(  # noqa: S603 -- test-owned lock holder
        [sys.executable, "-I", "-c", _HOLD_LOCKF, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "held\n"
        assert common.locking.file_lock_is_active(path)
        for lease in (common.locking.exclusive_file_lock, common.locking.shared_file_lock):
            with pytest.raises(common.locking.FileLockUnavailableError), lease(path, blocking=False):
                pytest.fail("a lease must not be granted while another process holds the writer lock")
    finally:
        holder.communicate("", timeout=60)
    assert holder.returncode == 0
    assert not common.locking.file_lock_is_active(path)
    with common.locking.exclusive_file_lock(path, blocking=False):
        assert common.locking.file_lock_is_active(path)
