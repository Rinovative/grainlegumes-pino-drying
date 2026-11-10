"""Protect the orphan-snapshot prune of the source fingerprint script.

Responsibilities:
  - Check that report mode lists an orphan and writes nothing, and that --apply deletes it under an existing lock file
  - Check each keeping condition: storage references, a same-named tree, scheduler evidence, recent changes, defective records
  - Check the fail-closed stops: symlinks, a missing or held lock, the shared budget, a change between scan and deletion
  - Check the command line's sorted JSON keys, its scheduler queries and its exit codes

Design principles:
  - Every case builds its own storage and runtime folders under tmp_path and injects scheduler evidence and clock
  - The injected clock lies two days after the files were written, so only explicitly refreshed mtimes count as recent

This module does NOT:
  - Run Slurm, touch the real storage or runtime folders, or capture snapshots through Git
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import runpy
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/source_fingerprint.py"
_DAY = 86400.0
_COMMIT = "c" * 40


def _quiet() -> tuple[bool | None, bool | None]:
    return True, True


@dataclass
class _Store:
    """A storage and a trees root in tmp_path plus the script's live globals, which tests may patch."""

    script: dict[str, Any]
    storage: Path
    trees: Path
    now: float = field(default_factory=lambda: time.time() + 2 * _DAY)

    @property
    def snapshots(self) -> Path:
        return self.storage / "source_snapshots"

    def prune(self, *, apply: bool, scheduler: Callable[[], tuple[bool | None, bool | None]] = _quiet) -> dict[str, Any]:
        return self.script["prune_source_snapshots"](self.storage, self.trees, scheduler=scheduler, now=self.now, apply=apply)


@pytest.fixture
def store(tmp_path: Path) -> _Store:
    module = runpy.run_path(str(_SCRIPT), run_name="test_source_snapshot_prune")
    (tmp_path / "storage/source_snapshots").mkdir(parents=True)
    return _Store(module["prune_source_snapshots"].__globals__, tmp_path / "storage", tmp_path / "runtime/sources")


def _snapshot(storage: Path, digit: str, *, commit: str = _COMMIT) -> str:
    """Create a valid snapshot whose fingerprint repeats ``digit`` and return its folder name."""
    digest = digit * 64
    name = f"{commit[:12]}-{digest[:12]}"
    folder = storage / "source_snapshots" / name
    (folder / "untracked").mkdir(parents=True)
    (folder / "untracked/new.py").write_text("VALUE = 1\n", encoding="utf-8")
    (folder / "tracked.diff").write_bytes(b"")
    (folder / "snapshot.json").write_text(json.dumps({"git_commit": commit, "source_fingerprint_sha256": digest}), encoding="utf-8")
    return name


def _lock(trees: Path, name: str) -> Path:
    lock = trees / ".locks" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    return lock


def _size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def _outcome(**changes: Any) -> dict[str, Any]:
    return {"orphans": [], "defective": {}, "blocked": None, "refused": {}, "skipped": None, "bytes": 0, **changes}


def _on_first_lock(store: _Store, monkeypatch: pytest.MonkeyPatch, action: Callable[[], object]) -> None:
    """Run ``action`` once, as soon as the prune holds its first tree lock: after the scan, before the recheck."""
    real_lock = store.script["_tree_lock"]
    done: list[bool] = []

    @contextlib.contextmanager
    def lock_then_act(*arguments: Any, **options: Any) -> Iterator[None]:
        with real_lock(*arguments, **options):
            if not done:
                done.append(True)
                action()
            yield

    monkeypatch.setitem(store.script, "_tree_lock", lock_then_act)


def test_report_lists_an_orphan_and_writes_nothing(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    assert store.prune(apply=False) == _outcome(orphans=[name], bytes=_size(store.snapshots / name))
    assert (store.snapshots / name / "snapshot.json").is_file()
    assert not store.trees.exists()  # not even the lock files that --apply would create


def test_apply_deletes_an_orphan_whose_lock_file_exists(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    lock = _lock(store.trees, name)
    size = _size(store.snapshots / name)
    assert store.prune(apply=True) == _outcome(orphans=[name], bytes=size)
    assert not (store.snapshots / name).exists()
    assert lock.is_file()


def test_storage_references_keep_snapshots_but_the_commit_alone_does_not(store: _Store) -> None:
    by_name = _snapshot(store.storage, "1", commit="a" * 40)
    by_digest = _snapshot(store.storage, "e", commit="b" * 40)
    commit_only = _snapshot(store.storage, "3", commit="d" * 40)
    (store.storage / "01_generation/run").mkdir(parents=True)
    (store.storage / "01_generation/run/epochs.jsonl").write_text(json.dumps({"tree": by_name}) + "\n", encoding="utf-8")
    (store.storage / "02_datasets/.state").mkdir(parents=True)
    (store.storage / "02_datasets/.state/binding.yaml").write_text(f"source_worktree_sha256: {'E' * 64}\n", encoding="utf-8")
    (store.storage / "03_runs").mkdir()
    (store.storage / "03_runs/session.json").write_text(json.dumps({"source_commit": "d" * 40}), encoding="utf-8")
    assert store.prune(apply=False)["orphans"] == [commit_only]
    assert {by_name, by_digest, commit_only} <= {path.name for path in store.snapshots.iterdir()}


def test_a_same_named_tree_keeps_the_snapshot(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    (store.trees / name).mkdir()
    assert store.prune(apply=True) == _outcome()
    assert (store.snapshots / name / "snapshot.json").is_file()


@pytest.mark.parametrize(
    ("evidence", "stop"),
    [((False, True), "skipped"), ((True, False), "skipped"), ((None, True), "blocked"), ((True, None), "blocked")],
)
def test_scheduler_evidence_must_show_an_empty_queue_and_no_recent_job(store: _Store, evidence: tuple[bool | None, bool | None], stop: str) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    result = store.prune(apply=True, scheduler=lambda: evidence)
    assert result == _outcome(**{stop: result[stop]})
    assert isinstance(result[stop], str)
    assert (store.snapshots / name / "snapshot.json").is_file()


@pytest.mark.parametrize("recent", ["snapshot", "trees"])
def test_a_recent_change_keeps_the_snapshot(store: _Store, recent: str) -> None:
    name = _snapshot(store.storage, "1")
    lock = _lock(store.trees, name)
    target = store.snapshots / name / "untracked" if recent == "snapshot" else lock
    os.utime(target, (store.now - 60, store.now - 60))
    result = store.prune(apply=True)
    assert result == (_outcome() if recent == "snapshot" else _outcome(skipped=f"{store.trees} changed within the last 24 h"))
    assert (store.snapshots / name / "snapshot.json").is_file()


def test_defective_snapshots_are_reported_and_kept(store: _Store) -> None:
    missing = "0" * 12 + "-" + "0" * 12
    (store.snapshots / missing).mkdir()
    renamed = "f" * 12 + "-" + "f" * 12
    (store.snapshots / _snapshot(store.storage, "4")).rename(store.snapshots / renamed)
    interrupted = f".prune-{missing}-0123abcd"  # left behind by a removal that crashed after its private rename
    (store.snapshots / interrupted).mkdir()
    result = store.prune(apply=True)
    assert result == _outcome(defective=result["defective"])
    assert sorted(result["defective"]) == [interrupted, missing, renamed]
    assert "folder name" in result["defective"][renamed]
    assert result["defective"][interrupted] == "interrupted removal"
    assert (store.snapshots / missing).is_dir()
    assert (store.snapshots / interrupted).is_dir()
    assert (store.snapshots / renamed / "snapshot.json").is_file()


@pytest.mark.parametrize("link", ["storage", "locks"])
def test_a_symlink_blocks_all_deletion(store: _Store, tmp_path: Path, link: str) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    if link == "storage":
        (store.storage / "records").mkdir()
        (store.storage / "records/elsewhere").symlink_to(tmp_path)
    else:
        (store.trees / ".locks").rename(tmp_path / "real_locks")
        (store.trees / ".locks").symlink_to(tmp_path / "real_locks")
    result = store.prune(apply=True)
    assert result == _outcome(blocked=result["blocked"])
    assert isinstance(result["blocked"], str)
    assert (store.snapshots / name / "snapshot.json").is_file()


def test_a_missing_lock_file_is_created_and_nothing_is_deleted(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    assert store.prune(apply=True) == _outcome(skipped="lock files created; deletion in a later quiet run")
    assert (store.trees / ".locks" / f"{name}.lock").is_file()
    assert (store.snapshots / name / "snapshot.json").is_file()


def test_a_held_tree_lock_refuses_the_snapshot(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    with store.script["_tree_lock"](store.trees, name, exclusive=False):  # a launcher reuses the snapshot's tree
        result = store.prune(apply=True)
    assert result == _outcome(refused={name: "tree lock held by a launcher or a prune"})
    assert (store.snapshots / name / "snapshot.json").is_file()


@pytest.mark.parametrize("limit", ["_SCAN_MAX_ENTRIES", "_SCAN_MAX_TOTAL_BYTES", "_SCAN_SECONDS"])
def test_a_budget_exhausted_in_the_first_recheck_blocks_all_further_deletion(store: _Store, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    names = [_snapshot(store.storage, digit) for digit in "12"]
    for name in names:
        _lock(store.trees, name)
    real_monotonic = time.monotonic

    def exhaust() -> None:
        if limit == "_SCAN_SECONDS":  # the deadline is fixed when the run starts, so the clock moves past it instead
            monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 3600)
        else:
            monkeypatch.setitem(store.script, limit, 0)

    _on_first_lock(store, monkeypatch, exhaust)
    result = store.prune(apply=True)
    assert result == _outcome(blocked=result["blocked"])
    assert "over budget" in result["blocked"]
    assert all((store.snapshots / name / "snapshot.json").is_file() for name in names)


def test_a_snapshot_changed_between_scan_and_deletion_is_refused(store: _Store, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    edit = (store.snapshots / name / "untracked/new.py").write_text
    _on_first_lock(store, monkeypatch, lambda: edit("VALUE = 2  # edited after the scan\n", encoding="utf-8"))
    assert store.prune(apply=True) == _outcome(refused={name: "snapshot changed since the scan"})
    assert (store.snapshots / name / "snapshot.json").is_file()


@pytest.mark.parametrize("change", ["queue", "trees"])
def test_a_routine_stop_in_the_recheck_only_skips(store: _Store, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    """A queue that turns busy or a tree root touched after the scan is a routine skip (exit 0), not a refusal."""
    name = _snapshot(store.storage, "1")
    lock = _lock(store.trees, name)
    answers = iter([(True, True), (change != "queue", True)])
    _on_first_lock(store, monkeypatch, lambda: os.utime(lock, (store.now, store.now)) if change == "trees" else None)
    result = store.prune(apply=True, scheduler=lambda: next(answers))
    skipped = "Slurm queue not empty" if change == "queue" else f"{store.trees} changed within the last 24 h"
    assert result == _outcome(skipped=skipped)
    assert (store.snapshots / name / "snapshot.json").is_file()


@pytest.mark.parametrize("error", [errno.EIO, errno.EAGAIN])  # EAGAIN raises BlockingIOError, which only the lock itself may refuse
def test_an_os_error_in_the_recheck_blocks_the_remaining_deletions(store: _Store, monkeypatch: pytest.MonkeyPatch, error: int) -> None:
    names = [_snapshot(store.storage, digit) for digit in "12"]
    for name in names:
        _lock(store.trees, name)

    real_state = store.script["_trees_state"]
    failures = [OSError(error, "injected I/O error")]

    def unreadable_once(*arguments: Any) -> tuple[set[str], int]:
        if failures:
            raise failures.pop()
        return real_state(*arguments)

    _on_first_lock(store, monkeypatch, lambda: monkeypatch.setitem(store.script, "_trees_state", unreadable_once))
    assert store.prune(apply=True) == _outcome(blocked=f"[Errno {error}] injected I/O error")
    assert all((store.snapshots / name / "snapshot.json").is_file() for name in names)


def test_eagain_from_opening_the_lock_file_blocks_instead_of_counting_as_lock_held(store: _Store, monkeypatch: pytest.MonkeyPatch) -> None:
    names = [_snapshot(store.storage, digit) for digit in "12"]
    for name in names:
        _lock(store.trees, name)
    real_open = os.open
    failures = [BlockingIOError(errno.EAGAIN, "injected")]

    def open_lock_failing_once(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if str(path).endswith(".lock") and failures:
            raise failures.pop()
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_lock_failing_once)
    assert store.prune(apply=True) == _outcome(blocked=f"[Errno {errno.EAGAIN}] injected")
    assert all((store.snapshots / name / "snapshot.json").is_file() for name in names)


def test_an_unreadable_snapshot_folder_blocks_instead_of_counting_as_defective(store: _Store) -> None:
    name = _snapshot(store.storage, "1")
    folder = store.snapshots / name / "untracked"
    folder.chmod(0)
    try:
        result = store.prune(apply=False)
    finally:
        folder.chmod(0o755)
    assert result == _outcome(blocked=result["blocked"])
    assert "Permission denied" in result["blocked"]


def test_a_symlinked_lock_file_blocks_the_prune(store: _Store, tmp_path: Path) -> None:
    name = _snapshot(store.storage, "1")
    lock = _lock(store.trees, name)
    lock.unlink()
    (tmp_path / "elsewhere.lock").touch()
    lock.symlink_to(tmp_path / "elsewhere.lock")
    result = store.prune(apply=True)
    assert result == _outcome(blocked=result["blocked"])
    assert (store.snapshots / name / "snapshot.json").is_file()


def test_a_snapshot_swapped_for_a_symlink_after_its_recheck_removes_nothing_elsewhere(
    store: _Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "snapshot.json").write_text("{}\n", encoding="utf-8")
    real_plan = store.script["_entry_plan"]
    calls: list[Path] = []

    def plan_then_swap(root: Path, budget: Any) -> dict[str, Any]:
        plan = real_plan(root, budget)
        calls.append(root)
        if calls.count(store.snapshots / name) == 2:  # the recheck under the lock has just matched the scanned plan
            root.rename(tmp_path / "moved")
            root.symlink_to(victim, target_is_directory=True)
        return plan

    monkeypatch.setitem(store.script, "_entry_plan", plan_then_swap)
    result = store.prune(apply=True)
    assert (victim / "snapshot.json").is_file()
    assert result["orphans"] == []
    assert (tmp_path / "moved/snapshot.json").is_file()


@pytest.mark.parametrize("lock", ["held", "missing"])
def test_a_locks_folder_symlinked_after_the_scan_is_never_used(store: _Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lock: str) -> None:
    """Neither the lock nor the lock-file bootstrap may reach another folder through a symlinked .locks."""
    name = _snapshot(store.storage, "1")
    store.trees.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real_referenced = store.script["_referenced"]

    def scan_then_link(*arguments: Any) -> set[str]:
        found = real_referenced(*arguments)
        (store.trees / ".locks").symlink_to(elsewhere, target_is_directory=True)
        return found

    monkeypatch.setitem(store.script, "_referenced", scan_then_link)
    with contextlib.ExitStack() as stack:
        if lock == "held":  # a launcher holds the lock file that the symlink leads to
            handle = stack.enter_context((elsewhere / f"{name}.lock").open("a+"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        result = store.prune(apply=True)
    assert (store.trees / ".locks").is_symlink()  # the swap after the scan happened
    assert result == _outcome(blocked=result["blocked"])
    assert isinstance(result["blocked"], str)
    assert sorted(path.name for path in elsewhere.iterdir()) == ([f"{name}.lock"] if lock == "held" else [])
    assert (store.snapshots / name / "snapshot.json").is_file()


def test_a_folder_swapped_in_after_the_record_unlink_survives(store: _Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    name = _snapshot(store.storage, "1")
    _lock(store.trees, name)
    real_unlink = os.unlink

    def unlink_then_swap(path: Any, *, dir_fd: int | None = None) -> None:
        real_unlink(path, dir_fd=dir_fd)
        if path == "snapshot.json" and dir_fd is not None and not (tmp_path / "moved").exists():
            (store.snapshots / name).rename(tmp_path / "moved")
            (store.snapshots / name / "keep").mkdir(parents=True)  # another folder now holds the snapshot's name

    monkeypatch.setattr(os, "unlink", unlink_then_swap)
    result = store.prune(apply=True)
    assert (store.snapshots / name / "keep").is_dir()
    assert result == _outcome(refused={name: result["refused"].get(name)})
    assert isinstance(result["refused"][name], str)


def test_modes_never_follow_an_entry_swapped_for_a_symlink(store: _Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A directory replaced by a symlink while modes are applied must not change anything outside the tree."""
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub/file.txt").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("y", encoding="utf-8")
    (outside / "file.txt").chmod(0o444)
    swapped: list[bool] = []

    def swap_once() -> None:
        if not swapped:
            swapped.append(True)
            (tree / "sub").rename(tmp_path / "moved")
            (tree / "sub").symlink_to(outside, target_is_directory=True)

    real_fchmod, real_chmod = os.fchmod, Path.chmod
    monkeypatch.setattr(os, "fchmod", lambda descriptor, mode: (swap_once(), real_fchmod(descriptor, mode)))
    monkeypatch.setattr(Path, "chmod", lambda path, mode, **options: (swap_once(), real_chmod(path, mode, **options)))
    store.script["_set_tree_write"](tree, writable=True)
    assert swapped
    assert stat.S_IMODE((outside / "file.txt").stat().st_mode) == 0o444


def test_command_line_prints_sorted_json_queries_slurm_and_exits_two_on_a_refusal(store: _Store, tmp_path: Path) -> None:
    name = _snapshot(store.storage, "1")
    stamp = time.time() - 2 * _DAY  # the command line uses the real clock
    for path in [store.snapshots / name, *(store.snapshots / name).rglob("*")]:
        os.utime(path, (stamp, stamp), follow_symlinks=False)
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "slurm.log"
    for tool in ("squeue", "sacct"):
        (binary / tool).write_text(f'#!/bin/sh\necho "{tool} $*" >> "{log}"\n', encoding="utf-8")
        (binary / tool).chmod(0o755)
    environment = {**os.environ, "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}"}

    def run(*extra: str) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(_SCRIPT), "--prune-snapshots", str(store.storage), str(store.trees), *extra]
        return subprocess.run(command, env=environment, capture_output=True, text=True, check=False)  # noqa: S603 -- test-owned argument vector

    report = run()
    assert (report.returncode, json.loads(report.stdout)) == (0, _outcome(orphans=[name], bytes=_size(store.snapshots / name)))
    assert list(json.loads(report.stdout)) == sorted(_outcome())
    queries = log.read_text(encoding="utf-8").splitlines()
    assert [line.split(" -u ")[0] for line in queries] == ["squeue -h -t all", "sacct -X -n -P"]
    assert queries[1].endswith("-S now-24hours -o JobID")
    defective = "0" * 12 + "-" + "0" * 12
    (store.snapshots / defective).mkdir()
    applied = run("--apply")
    assert applied.returncode == 2
    assert set(json.loads(applied.stdout)["defective"]) == {defective}
    assert (store.trees / ".locks" / f"{name}.lock").is_file()
    assert run("--apply", "--force").returncode != 0
