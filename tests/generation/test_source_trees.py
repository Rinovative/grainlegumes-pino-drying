"""Protect frozen source trees: materialization, verification, leases and pruning.

Responsibilities:
  - Check that a materialized tree is verified, read-only and byte-exact to its snapshot
  - Check that concurrent publication of one snapshot accepts identical content and fails closed on any other
  - Check that a changed, symlinked or missing tree is refused instead of repaired
  - Check lease validation and the deletion rules of pruning with explicit clocks

Design principles:
  - Tiny controlled Git responses retain real filesystem, patch and hash checks
  - Tree verification must work with neither a Git directory nor a Git executable
  - Pruning deletes only on positive scheduler evidence and never by sleeping

This module does NOT:
  - Run Slurm, initialize Git repositories or execute code from a materialized tree
  - Cover snapshot capture and restoration beyond what tree materialization needs
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import os
import runpy
import stat
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from src.generation.contracts import generation_contracts_source as source_contracts

if TYPE_CHECKING:
    from collections.abc import Iterator

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/source_fingerprint.py"
_NOW = 10_000_000.0
_DAY = 86400.0
_OLD_TREE = "a" * 12 + "-" + "1" * 12
_LEASED_TREE = "a" * 12 + "-" + "2" * 12
_RECENT_TREE = "a" * 12 + "-" + "3" * 12
_STALE_LEASE_TREE = "a" * 12 + "-" + "4" * 12


@pytest.fixture
def tree_fixture(tmp_path: Path) -> Iterator[tuple[dict[str, Any], Path, Path, Path]]:
    module = runpy.run_path(str(_SCRIPT), run_name="test_source_trees")
    repository = tmp_path / "original"
    repository.mkdir(mode=0o750)
    (repository / "code.py").write_text("value = 2\n")
    (repository / "code.py").chmod(0o640)
    (repository / "notes").mkdir(mode=0o2750)
    (repository / "notes/readme.txt").write_text("retained input\n")
    (repository / "notes/readme.txt").chmod(0o700)
    (repository / "notes/link").symlink_to("readme.txt")
    diff = b"diff --git a/code.py b/code.py\n--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    def git(_root: Path, *arguments: str, **kwargs: Any) -> SimpleNamespace:
        operation = arguments[0]
        if operation == "diff":
            return SimpleNamespace(stdout=diff)
        if operation == "rev-parse":
            return SimpleNamespace(stdout="a" * 40 + "\n")
        if operation == "ls-files":
            if "--ignored" in arguments:
                return SimpleNamespace(stdout=b"")
            value = b"notes/link\0notes/readme.txt\0" if "--others" in arguments else b"code.py\0"
            return SimpleNamespace(stdout=value)
        if operation == "ls-tree":
            return SimpleNamespace(stdout=b"code.py\0")
        if operation == "archive":
            with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
                member = tarfile.TarInfo("code.py")
                content = b"value = 1\n"
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
            return SimpleNamespace(stdout=b"")
        message = f"Unexpected read-only Git operation: {arguments}"
        raise AssertionError(message)

    module["source_fingerprint"].__globals__["_git"] = git
    trees = tmp_path / "runtime/sources"
    yield module, repository, tmp_path / "storage/source_snapshots", trees
    if trees.is_dir():  # frozen directories would otherwise resist temporary-directory cleanup
        for entry in trees.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                module["_set_tree_write"](entry, writable=True)


def _directory_modes(tree: Path) -> list[int]:
    directories = [tree, *(path for path in tree.rglob("*") if path.is_dir() and not path.is_symlink())]
    return [stat.S_IMODE(path.stat().st_mode) for path in directories]


def _plain_tree(trees: Path, tree_id: str, *, age: float) -> Path:
    """Create a minimal tree directory with a frozen subdirectory whose mtime is ``age`` seconds old."""
    tree = trees / tree_id
    (tree / "sub").mkdir(parents=True)
    (tree / "sub/file.txt").write_text("x")
    (tree / "sub").chmod(0o555)
    tree.chmod(0o555)
    os.utime(tree, (_NOW - age, _NOW - age))
    return tree


def _lease(trees: Path, tree_id: str, job_id: str, *, age: float) -> None:
    lease = trees / ".leases" / tree_id / job_id
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.touch()
    os.utime(lease, (_NOW - age, _NOW - age))


def test_materialize_creates_verified_read_only_tree_with_recorded_modes(tree_fixture: tuple) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    tree = Path(receipt["tree_path"])
    snapshot = Path(receipt["snapshot_path"])
    assert tree == trees / receipt["tree_id"]
    assert receipt["tree_id"] == snapshot.parent.name == "a" * 12 + "-" + receipt["source_fingerprint_sha256"][:12]
    assert receipt["git_commit"] == "a" * 40
    assert receipt["snapshot_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert snapshot.is_relative_to(snapshots)
    assert (tree / "code.py").read_text() == "value = 2\n"
    assert stat.S_IMODE((tree / "code.py").stat().st_mode) == 0o640 & ~0o222  # recorded modes without any write bit
    assert stat.S_IMODE((tree / "notes/readme.txt").stat().st_mode) == 0o700 & ~0o222
    assert (tree / "notes/link").readlink() == Path("readme.txt")
    assert not (tree / ".git").exists()
    assert all(mode & 0o222 == 0 for mode in _directory_modes(tree))
    assert stat.S_IMODE((tree / "notes").stat().st_mode) == stat.S_IMODE((repository / "notes").stat().st_mode) & ~0o222
    assert [path.name for path in trees.iterdir() if not path.name.startswith(".")] == [receipt["tree_id"]]
    verified = module["verify_source_tree"](tree, snapshots)
    assert verified["tree_id"] == receipt["tree_id"]
    assert verified["git_commit"] == receipt["git_commit"]
    assert verified["source_fingerprint_sha256"] == receipt["source_fingerprint_sha256"]
    assert verified["full_modes_verified"] is True
    assert module["_SNAPSHOT_SCHEMA_VERSION"] == source_contracts.SOURCE_SNAPSHOT_SCHEMA_VERSION  # the host script cannot import src


def test_second_materialize_reuses_the_existing_tree(tree_fixture: tuple) -> None:
    module, repository, snapshots, trees = tree_fixture
    first = module["materialize_source_tree"](repository, snapshots, trees)
    tree = Path(first["tree_path"])
    inode = tree.stat().st_ino
    os.utime(tree, (_NOW - 10 * _DAY, _NOW - 10 * _DAY))
    second = module["materialize_source_tree"](repository, snapshots, trees)
    assert second == first
    assert tree.stat().st_ino == inode
    assert tree.stat().st_mtime > _NOW  # reuse counts as recent use, so a concurrent prune keeps the tree
    assert [path.name for path in trees.iterdir() if not path.name.startswith(".")] == [first["tree_id"]]


def test_tampered_tree_is_refused_and_not_repaired(tree_fixture: tuple) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    tree = Path(receipt["tree_path"])
    with pytest.raises(PermissionError):  # tree files are frozen too
        (tree / "code.py").write_text("value = 3\n")
    (tree / "code.py").chmod(0o640)
    (tree / "code.py").write_text("value = 3\n")
    with pytest.raises(ValueError, match="inventory"):
        module["materialize_source_tree"](repository, snapshots, trees)
    with pytest.raises(ValueError, match="inventory"):
        module["verify_source_tree"](tree, snapshots)
    assert (tree / "code.py").read_text() == "value = 3\n"
    assert [path.name for path in trees.iterdir() if not path.name.startswith(".")] == [receipt["tree_id"]]


def test_unexpected_member_symlinked_and_unsnapshotted_trees_are_refused(tree_fixture: tuple, tmp_path: Path) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    tree = Path(receipt["tree_path"])
    module["_set_tree_write"](tree, writable=True)
    (tree / "extra.py").write_text("injected\n")
    with pytest.raises(ValueError, match="membership"):
        module["verify_source_tree"](tree, snapshots)
    link = tmp_path / receipt["tree_id"]
    link.symlink_to(tree, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module["verify_source_tree"](link, snapshots)
    with pytest.raises(FileNotFoundError, match="Snapshot"):
        module["verify_source_tree"](tree, tmp_path / "other_snapshots")
    with pytest.raises(ValueError, match="<commit12>"):
        module["verify_source_tree"](tmp_path, snapshots)


def test_verify_source_tree_needs_neither_git_directory_nor_git_call(tree_fixture: tuple) -> None:
    module, repository, snapshots, trees = tree_fixture
    tree = Path(module["materialize_source_tree"](repository, snapshots, trees)["tree_path"])

    def forbidden(_root: Path, *arguments: str, **_kwargs: Any) -> None:
        message = f"Tree verification must not call Git: {arguments}"
        raise AssertionError(message)

    module["source_fingerprint"].__globals__["_git"] = forbidden
    assert not (tree / ".git").exists()
    assert module["verify_source_tree"](tree, snapshots)["inventory_entries_verified"] > 0


@pytest.mark.parametrize("job_id", ["", "12x", "12_", "_3", "1_2_3", "../1", "-5", "12 "])
def test_lease_rejects_malformed_job_ids(tree_fixture: tuple, job_id: str) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    with pytest.raises(ValueError, match="job ID"):
        module["lease_source_tree"](trees, receipt["tree_id"], job_id)
    assert not (trees / ".leases").exists()


def test_lease_takes_its_shared_tree_lock_on_a_readable_descriptor(tree_fixture: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the compute nodes' NFS a shared flock is a POSIX read lock, which fails with EBADF on a write-only descriptor."""
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    real_flock = fcntl.flock

    def nfs_flock(descriptor: int, operation: int) -> None:
        if operation & fcntl.LOCK_SH and fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_WRONLY:
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", nfs_flock)
    assert module["lease_source_tree"](trees, receipt["tree_id"], "4242").is_file()


def test_lease_validates_the_tree_and_is_idempotent(tree_fixture: tuple) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    tree_id = receipt["tree_id"]
    for bad in ("../x", "short", tree_id.upper()):
        with pytest.raises(ValueError, match="tree ID"):
            module["lease_source_tree"](trees, bad, "1")
    with pytest.raises(FileNotFoundError, match="missing"):
        module["lease_source_tree"](trees, "b" * 12 + "-" + "c" * 12, "1")
    first = module["lease_source_tree"](trees, tree_id, "123")
    again = module["lease_source_tree"](trees, tree_id, "123")
    array = module["lease_source_tree"](trees, tree_id, "124_7")
    assert first == again == trees / ".leases" / tree_id / "123"
    assert first.is_file()
    assert first.stat().st_size == 0
    assert sorted(path.name for path in first.parent.iterdir()) == ["123", "124_7"]
    assert array.name == "124_7"


def test_prune_deletes_only_old_unused_trees_and_their_leases(tree_fixture: tuple) -> None:
    module, _repository, _snapshots, trees = tree_fixture
    _plain_tree(trees, _OLD_TREE, age=3 * _DAY)
    _lease(trees, _OLD_TREE, "11", age=3 * _DAY)
    _plain_tree(trees, _LEASED_TREE, age=3 * _DAY)
    _lease(trees, _LEASED_TREE, "22", age=3 * _DAY)
    _plain_tree(trees, _RECENT_TREE, age=60)
    _plain_tree(trees, _STALE_LEASE_TREE, age=3 * _DAY)
    _lease(trees, _STALE_LEASE_TREE, "44", age=60)
    result = module["prune_source_trees"](trees, {"22", "33"}, now=_NOW)
    assert result == {"removed": [_OLD_TREE], "kept": [_LEASED_TREE, _RECENT_TREE, _STALE_LEASE_TREE]}
    assert not (trees / _OLD_TREE).exists()
    assert not (trees / ".leases" / _OLD_TREE).exists()
    assert (trees / _LEASED_TREE / "sub/file.txt").is_file()
    assert (trees / ".leases" / _LEASED_TREE / "22").is_file()
    assert (trees / _RECENT_TREE).is_dir()
    assert (trees / _STALE_LEASE_TREE).is_dir()
    aged = module["prune_source_trees"](trees, set(), now=_NOW + 2 * _DAY)
    assert aged["removed"] == [_LEASED_TREE, _RECENT_TREE, _STALE_LEASE_TREE]
    assert aged["kept"] == []


def test_prune_rechecks_under_the_tree_lock_and_keeps_a_tree_reused_meanwhile(tree_fixture: tuple) -> None:
    """A launcher holding the shared lock while it refreshes an old tree wins against a prune that started before."""
    module, _repository, _snapshots, trees = tree_fixture
    tree = _plain_tree(trees, _OLD_TREE, age=3 * _DAY)
    started = threading.Event()
    result: dict[str, Any] = {}

    def prune() -> None:
        started.set()
        result.update(module["prune_source_trees"](trees, set(), now=_NOW))

    with module["_tree_lock"](trees, _OLD_TREE, exclusive=False):
        worker = threading.Thread(target=prune)
        worker.start()
        started.wait(timeout=5)
        worker.join(timeout=0.5)
        assert worker.is_alive()  # prune waits for the shared lock before deciding
        os.utime(tree, (_NOW, _NOW))  # the launcher reuses the tree
    worker.join(timeout=5)
    assert result == {"removed": [], "kept": [_OLD_TREE]}
    assert tree.is_dir()


@pytest.mark.parametrize("identical", [True, False])
def test_snapshot_published_concurrently_is_accepted_only_when_identical(
    tree_fixture: tuple, monkeypatch: pytest.MonkeyPatch, identical: bool
) -> None:
    """A process that loses the rename to another one reuses an identical snapshot and fails closed on any other content."""
    module, repository, snapshots, _trees = tree_fixture
    capture = module["capture_source_snapshot"]
    original_rename = Path.rename
    racing: list[dict[str, str]] = []

    def rename(self: Path, target: Path) -> Path:
        if self.name.startswith(".source-snapshot-") and not racing:
            racing.append({})
            if identical:
                racing[0] = capture(repository, snapshots)  # the other process publishes the same snapshot between check and rename
            else:
                Path(target).mkdir(parents=True)
                (Path(target) / "snapshot.json").write_text("{}\n", encoding="utf-8")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    if identical:
        receipt = capture(repository, snapshots)
        assert receipt == racing[0]
        assert [entry.name for entry in snapshots.iterdir()] == [Path(receipt["snapshot_path"]).parent.name]
    else:
        with pytest.raises(FileExistsError, match="conflicts with current source bytes"):
            capture(repository, snapshots)
        assert [entry.name for entry in snapshots.iterdir() if entry.name.startswith(".source-snapshot-")] == []
    assert len(racing) == 1


def test_snapshot_named_after_other_content_is_refused(tree_fixture: tuple) -> None:
    """A snapshot directory whose name does not match its own commit and fingerprint can neither verify nor restore a tree."""
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)
    renamed = snapshots / _OLD_TREE
    Path(receipt["snapshot_path"]).parent.rename(renamed)
    with pytest.raises(ValueError, match="not named after its own commit and fingerprint"):
        module["materialize_snapshot_tree"](repository, renamed / "snapshot.json", trees)
    module["_set_tree_write"](Path(receipt["tree_path"]), writable=True)
    Path(receipt["tree_path"]).rename(trees / _OLD_TREE)
    with pytest.raises(ValueError, match="not named after its own commit and fingerprint"):
        module["verify_source_tree"](trees / _OLD_TREE, snapshots)


def test_prune_without_scheduler_answer_deletes_nothing(tree_fixture: tuple) -> None:
    module, _repository, _snapshots, trees = tree_fixture
    _plain_tree(trees, _OLD_TREE, age=30 * _DAY)
    staging = trees / f".staging-{_OLD_TREE}-deadbeef"
    staging.mkdir()
    os.utime(staging, (_NOW - 30 * _DAY, _NOW - 30 * _DAY))
    result = module["prune_source_trees"](trees, None, now=_NOW)
    assert result == {"removed": [], "kept": [_OLD_TREE]}
    assert (trees / _OLD_TREE / "sub/file.txt").is_file()
    assert staging.is_dir()


def test_prune_removes_only_stale_staging_directories(tree_fixture: tuple) -> None:
    module, _repository, _snapshots, trees = tree_fixture
    stale = trees / f".staging-{_OLD_TREE}-aaaaaaaa"
    fresh = trees / f".staging-{_OLD_TREE}-bbbbbbbb"
    for staging, age in ((stale, 2 * _DAY), (fresh, 60)):
        (staging / "inner").mkdir(parents=True)
        (staging / "inner/file.txt").write_text("x")
        (staging / "inner").chmod(0o555)
        os.utime(staging, (_NOW - age, _NOW - age))
    (trees / "unrelated").mkdir()
    result = module["prune_source_trees"](trees, set(), now=_NOW)
    assert result == {"removed": [stale.name], "kept": []}
    assert not stale.exists()
    assert fresh.is_dir()
    assert (trees / "unrelated").is_dir()


@pytest.mark.parametrize(
    ("squeue_output", "expected"),
    [("101\n102_3\n\n", {"101", "102_3", "102"}), ("", set())],
)
def test_active_job_ids_follow_squeue_and_array_syntax(
    tree_fixture: tuple,
    monkeypatch: pytest.MonkeyPatch,
    squeue_output: str,
    expected: set[str],
) -> None:
    module = tree_fixture[0]
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=squeue_output)

    monkeypatch.setattr(subprocess, "run", run)
    assert module["_active_slurm_job_ids"]() == expected
    assert calls[0][:3] == ["squeue", "-h", "-u"]
    assert calls[0][-2:] == ["-o", "%i"]


@pytest.mark.parametrize("failure", [FileNotFoundError("squeue"), subprocess.CalledProcessError(1, "squeue")])
def test_active_job_ids_are_unknown_when_squeue_fails(tree_fixture: tuple, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    module = tree_fixture[0]

    def run(_command: list[str], **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(subprocess, "run", run)
    assert module["_active_slurm_job_ids"]() is None


def test_command_line_verifies_leases_and_rejects_unknown_forms(
    tree_fixture: tuple,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, repository, snapshots, trees = tree_fixture
    receipt = module["materialize_source_tree"](repository, snapshots, trees)

    def run(*arguments: str) -> str:
        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), *arguments])
        runpy.run_path(str(_SCRIPT), run_name="__main__")
        return capsys.readouterr().out

    assert f'"tree_id": "{receipt["tree_id"]}"' in run("--verify-tree", receipt["tree_path"], str(snapshots))
    assert run("--lease", str(trees), receipt["tree_id"], "7").strip() == str(trees / ".leases" / receipt["tree_id"] / "7")
    with pytest.raises(SystemExit, match="--materialize"):
        run("--bogus", "first", "second")
