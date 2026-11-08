"""Fingerprint, preserve and freeze a Git worktree for scheduler admission.

Responsibilities:
  - Hash tracked changes against HEAD and untracked source contents
  - Capture immutable source snapshots of clean or dirty trees and restore them for reproduction
  - Materialize and verify read-only source trees restored from snapshots
  - Record job leases on source trees and prune trees that no queued or running job uses
  - Expose fingerprint, restore, materialize, materialize-snapshot, verify-tree, lease, and prune command-line operations

Design principles:
  - Preserve the established source fingerprint and bind snapshots to Git HEAD
  - Verify complete source bytes, paths, and permissions before restoration
  - Publish a snapshot or source tree by one atomic rename; identical content published first by a concurrent process is accepted
  - Fail closed on any other content under a published name; never repair or overwrite a snapshot or tree
  - Verify a tree against its snapshot without Git, so frozen trees stay checkable where only the tree exists
  - Delete trees only on positive scheduler evidence; an unanswered queue query keeps every tree

This module does NOT:
  - Establish scientific run identity or authorize scheduler work
  - Modify a source checkout, Git index, or Git history
  - Run scientific workflows during restoration
  - Submit jobs or decide which source tree a job executes from

Notes:
  Source tree store (trees_root, normally runtime/sources):
    <commit12>-<fingerprint12>/ is one read-only tree named like its snapshot,
    whose manifest must carry that same commit and fingerprint;
    .leases/<tree_id>/<lease_id> are empty lease files for a Slurm job ID;
    .locks/<tree_id>.lock serializes reuse and leasing (shared) against
    pruning (exclusive, rechecked under the lock);
    .staging-<tree_id>-<suffix>/ is an unpublished restore. Files and
    directories lose all write bits, and verification compares their modes
    with write bits masked.

"""

from __future__ import annotations

import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import posixpath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator

_SNAPSHOT_SCHEMA_VERSION: Final = 1
_FINGERPRINT_ARGUMENT_COUNT = 2
_RESTORE_ARGUMENT_COUNT = 5
_MATERIALIZE_ARGUMENT_COUNT = 5
_VERIFY_TREE_ARGUMENT_COUNT = 4
_LEASE_ARGUMENT_COUNT = 5
_PRUNE_ARGUMENT_COUNT = 3
_CHUNK_SIZE = 1024 * 1024
_GIT_OBJECT_ID_LENGTH = 40
_SHA256_LENGTH = 64
_CACHE_DIRECTORIES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_SNAPSHOT_NAME_PART_LENGTH = 12
_TREE_ID_PATTERN = re.compile(r"[0-9a-f]{12}-[0-9a-f]{12}")
_JOB_ID_PATTERN = re.compile(r"[0-9]+(_[0-9]+)?")
_LEASES_DIRNAME = ".leases"
_LOCKS_DIRNAME = ".locks"
_STAGING_PREFIX = ".staging-"
_ALL_WRITE_BITS = 0o222
_PRUNE_MINIMUM_AGE_SECONDS = 86400.0
_SQUEUE_TIMEOUT_SECONDS = 60


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _conflict(message: str) -> NoReturn:
    raise FileExistsError(message)


def _git(root: Path, *arguments: str, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, **kwargs)  # noqa: S603, S607 -- fixed executable and argument vector


def source_fingerprint(root: Path) -> str:
    """Return the SHA256 of tracked changes and untracked source contents."""
    tracked = _git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--", stdout=subprocess.PIPE).stdout
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", stdout=subprocess.PIPE).stdout
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(len(tracked).to_bytes(8, "big"))
    digest.update(tracked)
    for raw in sorted(part for part in untracked.split(b"\0") if part):
        path = root / os.fsdecode(raw)
        digest.update(b"untracked\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        if path.is_symlink():
            value = os.fsencode(path.readlink())
            digest.update(b"symlink\0")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        elif path.is_file():
            digest.update(b"file\0")
            digest.update((path.stat().st_mode & stat.S_IXUSR).to_bytes(1, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                    digest.update(chunk)
        else:
            _invalid(f"Cannot fingerprint source path: {path}")
    return digest.hexdigest()


def _inventory(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {".": {"path": ".", "kind": "directory", "mode": stat.S_IMODE(root.stat().st_mode)}}
    for relative in paths:
        _assert_no_symlink_parents(root, relative)
        path = root / relative
        if not path.exists() and not path.is_symlink():
            continue
        for parent in relative.parents:
            if parent == Path():
                break
            parent_path = root / parent
            entries[parent.as_posix()] = {"path": parent.as_posix(), "kind": "directory", "mode": stat.S_IMODE(parent_path.stat().st_mode)}
        name = relative.as_posix()
        if path.is_symlink():
            target = _readlink(path)
            _validate_link_target(name, target)
            _validate_resolved_link(root, path)
            entries[name] = {"path": name, "kind": "symlink", "target": target}
        elif path.is_dir():
            entries[name] = {"path": name, "kind": "directory", "mode": stat.S_IMODE(path.stat().st_mode)}
        elif path.is_file():
            entries[name] = {"path": name, "kind": "file", "mode": stat.S_IMODE(path.stat().st_mode), "sha256": _sha256_file(path)}
        else:
            _invalid(f"Unsupported source entry: {path}")
    records = list(entries.values())
    return sorted(records, key=lambda item: item["path"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readlink(path: Path) -> str:
    """Return a symlink target decoded with the filesystem codec."""
    return os.fsdecode(os.fsencode(path.readlink()))


def _validate_resolved_link(root: Path, link: Path) -> None:
    try:
        link.resolve(strict=False).relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        _invalid(f"Symlink resolves outside the source tree: {link}")


def capture_source_snapshot(root: Path, destination_root: Path) -> dict[str, str]:
    """
    Persist the exact source, clean or with uncommitted changes, admitted for one run.

    The snapshot is named after its commit and fingerprint and published by one
    atomic rename. An existing snapshot of that name, including one a concurrent
    process published first, is accepted only when its manifest is byte-identical
    and still matches the source; any other content under the name fails closed.
    """
    root = root.resolve()
    before = source_fingerprint(root)
    head = _git(root, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    diff = _git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--", stdout=subprocess.PIPE).stdout
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", stdout=subprocess.PIPE).stdout
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".source-snapshot-", dir=destination_root))
    try:
        (temporary / "tracked.diff").write_bytes(diff)
        records = []
        for raw in sorted(part for part in untracked.split(b"\0") if part):
            relative = _safe_relative(os.fsdecode(raw))
            _assert_no_symlink_parents(root, relative)
            source = root / relative
            target = temporary / "untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                link = _readlink(source)
                target.symlink_to(link)
                records.append({"path": relative.as_posix(), "kind": "symlink", "target": link})
            elif source.is_file():
                shutil.copyfile(source, target, follow_symlinks=False)
                mode = stat.S_IMODE(source.stat().st_mode)
                target.chmod(mode)
                records.append({"path": relative.as_posix(), "kind": "file", "mode": mode, "sha256": _sha256_file(target)})
            else:
                _invalid(f"Cannot snapshot untracked source path: {source}")
        tracked_paths = _git(root, "ls-files", "-z", stdout=subprocess.PIPE).stdout
        ignored_paths = _ignored_paths(root)
        source_names = {part for part in tracked_paths.split(b"\0") + untracked.split(b"\0") if part}
        source_paths = [_safe_relative(os.fsdecode(raw)) for raw in sorted(source_names)]
        inventory = _inventory(root, source_paths)
        current_head = _git(root, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
        if (
            source_fingerprint(root) != before
            or current_head != head
            or _inventory(root, source_paths) != inventory
            or _ignored_paths(root) != ignored_paths
        ):
            message = "Generation source changed while its dirty snapshot was captured."
            raise RuntimeError(message)
        manifest = {
            "schema_kind": "generation_source_snapshot",
            "schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "git_commit": head,
            "source_fingerprint_sha256": before,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked": records,
            "full_modes_recoverable": True,
            "inventory": inventory,
            "ignored_paths": ignored_paths,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        (temporary / "snapshot.json").write_bytes(manifest_bytes)
        final = destination_root / f"{head[:_SNAPSHOT_NAME_PART_LENGTH]}-{before[:_SNAPSHOT_NAME_PART_LENGTH]}"
        if not final.exists():
            try:
                temporary.rename(final)
            except OSError:
                if not final.exists():
                    raise
                # Another process published its snapshot between the check and the rename; it is judged like an earlier one.
        if temporary.exists():
            if final.is_symlink() or not final.is_dir() or not _matching_snapshot(final, manifest_bytes):
                _conflict("Existing Generation source snapshot conflicts with current source bytes.")
            verify_source_snapshot(final / "snapshot.json", root)
        return {"fingerprint": before, "snapshot_path": str(final / "snapshot.json"), "snapshot_sha256": _sha256_file(final / "snapshot.json")}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _matching_snapshot(directory: Path, manifest_bytes: bytes) -> bool:
    path = directory / "snapshot.json"
    return not path.is_symlink() and path.is_file() and path.read_bytes() == manifest_bytes


def _ignored_paths(root: Path) -> list[str]:
    output = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z", stdout=subprocess.PIPE).stdout
    paths = [os.fsdecode(raw) for raw in output.split(b"\0") if raw]
    for value in paths:
        _safe_relative(value.rstrip("/"))
    return sorted(set(paths))


def _validate_exclusions(ignored_paths: list[str], inventory: list[dict[str, Any]]) -> None:
    for exclusion in ignored_paths:
        subtree = exclusion.endswith("/")
        relative = _safe_relative(exclusion.rstrip("/"))
        name = relative.as_posix()
        prefix = f"{name}/"
        for item in inventory:
            item_path = item.get("path")
            if item_path == name or (subtree and isinstance(item_path, str) and item_path.startswith(prefix)):
                _invalid(f"Ignored-path exclusion hides inventoried source: {exclusion}")


def _is_excluded(name: str, ignored_paths: list[str]) -> bool:
    for exclusion in ignored_paths:
        relative = exclusion.rstrip("/")
        if name == relative or (exclusion.endswith("/") and name.startswith(f"{relative}/")):
            return True
    return False


def _source_membership(root: Path, ignored_paths: list[str]) -> set[str]:
    members: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        parent = Path(current)
        retained_directories = []
        for name in directories:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if name == ".git" or name in _CACHE_DIRECTORIES or _is_excluded(relative, ignored_paths):
                continue
            if path.is_symlink():
                members.add(relative)
            else:
                retained_directories.append(name)
        directories[:] = retained_directories
        for name in files:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if name == ".git" or _is_excluded(relative, ignored_paths):
                continue
            if path.is_symlink() or path.is_file():
                members.add(relative)
    return members


def _safe_relative(value: str) -> Path:
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts) or "\\" in value:
        _invalid(f"Unsafe source path: {value!r}")
    return Path(*parsed.parts)


def _validate_link_target(relative_path: str, target: str) -> None:
    if not target or target.startswith("/") or "\\" in target:
        _invalid(f"Unsafe symlink target at {relative_path}: {target!r}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative_path), target))
    if resolved == ".." or resolved.startswith("../"):
        _invalid(f"Symlink escapes restored source at {relative_path}: {target!r}")


def _assert_no_symlink_parents(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            _invalid(f"Cannot write through source symlink parent: {current}")


def _clean_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        environment.pop(name, None)
    return environment


def _validate_patch(diff: bytes) -> None:
    for line in diff.splitlines():
        if line.startswith((b"--- a/", b"+++ b/")):
            value = line[6:].split(b"\t", 1)[0].decode("utf-8", errors="surrogateescape")
            if value != "/dev/null":
                _safe_relative(value)
        elif line.startswith((b"rename from ", b"rename to ", b"copy from ", b"copy to ")):
            value = line.split(b" ", 2)[2].decode("utf-8", errors="surrogateescape")
            _safe_relative(value)


def _snapshot_fingerprint(diff: bytes, snapshot_dir: Path, entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(len(diff).to_bytes(8, "big"))
    digest.update(diff)
    for item in sorted(entries, key=lambda entry: os.fsencode(entry["path"])):
        raw = os.fsencode(item["path"])
        digest.update(b"untracked\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        saved = snapshot_dir / "untracked" / _safe_relative(item["path"])
        if item["kind"] == "symlink":
            value = os.fsencode(_readlink(saved))
            digest.update(b"symlink\0")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        else:
            digest.update(b"file\0")
            executable = item.get("mode", 0) & stat.S_IXUSR
            digest.update(executable.to_bytes(1, "big"))
            with saved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            relative = _safe_relative(member.name.rstrip("/"))
            if member.isdev() or member.isfifo() or member.issparse():
                _invalid(f"Unsupported Git archive entry: {member.name}")
            target = destination / relative
            _assert_no_symlink_parents(destination, relative)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                _validate_link_target(relative.as_posix(), member.linkname)
                target.symlink_to(member.linkname)
            elif member.isfile():
                source = bundle.extractfile(member)
                if source is None:
                    _invalid(f"Unreadable Git archive entry: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            else:
                _invalid(f"Unsupported Git archive entry: {member.name}")


def _validate_tree_links(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        parent = Path(current)
        for name in directories + files:
            path = parent / name
            if path.is_symlink():
                _validate_link_target(path.relative_to(root).as_posix(), _readlink(path))
                _validate_resolved_link(root, path)


def _load_snapshot(snapshot_path: Path) -> tuple[Path, str, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        _invalid("Snapshot manifest must be a regular non-symlink file.")
    snapshot_path = snapshot_path.resolve(strict=True)
    manifest = json.loads(snapshot_path.read_bytes())
    version = manifest.get("schema_version")
    # Strict like common.serialization.is_schema_version, which this stdlib-only script cannot import: JSON true or 1.0 is no version 1.
    if manifest.get("schema_kind") != "generation_source_snapshot" or type(version) is not int or version != _SNAPSHOT_SCHEMA_VERSION:
        _invalid("Unsupported Generation source snapshot schema.")
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or len(commit) != _GIT_OBJECT_ID_LENGTH or any(character not in "0123456789abcdef" for character in commit):
        _invalid("Snapshot Git commit must be a full 40-character object ID.")
    snapshot_dir = snapshot_path.parent
    diff_path = snapshot_dir / "tracked.diff"
    if diff_path.is_symlink() or not diff_path.is_file():
        _invalid("Snapshot tracked diff is missing or unsafe.")
    diff = diff_path.read_bytes()
    _validate_sha256(manifest.get("tracked_diff_sha256"), "tracked diff")
    _validate_sha256(manifest.get("source_fingerprint_sha256"), "source fingerprint")
    if hashlib.sha256(diff).hexdigest() != manifest.get("tracked_diff_sha256"):
        _invalid("Snapshot tracked diff digest does not match its manifest.")
    _validate_patch(diff)
    untracked = manifest.get("untracked")
    if not isinstance(untracked, list):
        _invalid("Snapshot untracked inventory is invalid.")
    for item in untracked:
        relative = _safe_relative(item["path"])
        saved = snapshot_dir / "untracked" / relative
        if not _is_below(saved, snapshot_dir) or (saved.is_symlink() and item.get("kind") != "symlink"):
            _invalid(f"Snapshot payload escapes or is unsafe: {relative}")
        if item.get("kind") == "file":
            if saved.is_symlink() or not saved.is_file() or _sha256_file(saved) != item.get("sha256"):
                _invalid(f"Snapshot untracked file failed verification: {relative}")
        elif item.get("kind") == "symlink":
            if not saved.is_symlink() or _readlink(saved) != item.get("target"):
                _invalid(f"Snapshot untracked symlink failed verification: {relative}")
            _validate_link_target(relative.as_posix(), item["target"])
        else:
            _invalid(f"Unsupported untracked entry: {relative}")
    if _snapshot_fingerprint(diff, snapshot_dir, untracked) != manifest.get("source_fingerprint_sha256"):
        _invalid("Snapshot source fingerprint does not match its retained diff and untracked bytes.")
    inventory = manifest.get("inventory")
    if manifest.get("full_modes_recoverable") is not True or not isinstance(inventory, list):
        _invalid("Snapshot full-mode inventory is missing.")
    ignored_paths = manifest.get("ignored_paths")
    if not isinstance(ignored_paths, list) or any(not isinstance(value, str) for value in ignored_paths):
        _invalid("Snapshot ignored-path exclusions are missing or invalid.")
    _validate_exclusions(ignored_paths, inventory)
    return snapshot_dir, commit, diff, untracked, inventory


def _apply_tracked_diff(destination: Path, diff: bytes) -> None:
    if not diff:
        return
    patch_paths = _patch_paths(diff)
    for path in patch_paths:
        _assert_no_symlink_parents(destination, path)
        if any(other != path and path in other.parents for other in patch_paths):
            _invalid("Snapshot patch paths overlap and could traverse a newly created symlink.")
    environment = _clean_git_environment()
    environment["GIT_CEILING_DIRECTORIES"] = str(destination.parent)
    subprocess.run(["git", "apply", "--binary", "--check"], input=diff, cwd=destination, env=environment, check=True)  # noqa: S607 -- fixed executable and argument vector
    subprocess.run(["git", "apply", "--binary"], input=diff, cwd=destination, env=environment, check=True)  # noqa: S607 -- fixed executable and argument vector


def _restore_untracked(snapshot_dir: Path, destination: Path, entries: list[dict[str, Any]]) -> None:
    for item in entries:
        relative = _safe_relative(item["path"])
        target = destination / relative
        _assert_no_symlink_parents(destination, relative)
        if target.exists() or target.is_symlink():
            _invalid(f"Untracked snapshot would overwrite restored tracked content: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        saved = snapshot_dir / "untracked" / relative
        if item["kind"] == "file":
            shutil.copyfile(saved, target)
            target.chmod(item.get("mode", 0o644) & 0o7777)
        else:
            target.symlink_to(item["target"])


def _apply_inventory(destination: Path, inventory: list[dict[str, Any]]) -> None:
    for item in sorted(inventory, key=lambda entry: entry["path"].count("/"), reverse=True):
        relative = item["path"]
        target = destination if relative == "." else destination / _safe_relative(relative)
        if item["kind"] == "directory":
            if not target.is_dir() or target.is_symlink():
                _invalid(f"Restored directory missing: {relative}")
            target.chmod(item["mode"] & 0o7777)
        elif item["kind"] == "file":
            if target.is_symlink() or not target.is_file() or _sha256_file(target) != item["sha256"]:
                _invalid(f"Restored file content mismatch: {relative}")
            target.chmod(item["mode"] & 0o7777)
        elif item["kind"] == "symlink":
            if not target.is_symlink() or _readlink(target) != item["target"]:
                _invalid(f"Restored symlink mismatch: {relative}")
        else:
            _invalid(f"Invalid source inventory kind: {item['kind']}")
    inventory_paths = [_safe_relative(item["path"]) for item in inventory if item["path"] != "."]
    if _inventory(destination, inventory_paths) != inventory:
        _invalid("Restored source inventory does not match the snapshot.")


def _without_write_bits(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the inventory with write bits cleared from file and directory modes."""
    return [{**item, "mode": item["mode"] & ~_ALL_WRITE_BITS} if item["kind"] in {"file", "directory"} else item for item in inventory]


def verify_source_snapshot(snapshot_path: Path, source_root: Path, *, frozen: bool = False) -> dict[str, Any]:
    """
    Verify selected source paths against a retained snapshot.

    Parameters
    ----------
    snapshot_path : Path
        Snapshot manifest whose retained evidence is authoritative.
    source_root : Path
        Reconstructed or existing source tree to inspect without modifying it.
    frozen : bool, optional
        Compare file and directory modes with write bits masked on both sides,
        as needed for a read-only source tree; every other permission bit is
        still compared exactly.

    Returns
    -------
    dict[str, Any]
        Verification status and the selected inventory scope.

    Raises
    ------
    ValueError
        If snapshot evidence or selected source bytes, links, or modes differ.
    FileNotFoundError
        If required snapshot files are absent.

    """
    snapshot_dir, _, diff, untracked, inventory = _load_snapshot(snapshot_path)
    manifest = json.loads((snapshot_dir / "snapshot.json").read_bytes())
    selected_paths = [_safe_relative(item["path"]) for item in inventory if item["path"] != "."]
    actual_inventory = _inventory(source_root, selected_paths)
    if frozen:
        actual_inventory, inventory = _without_write_bits(actual_inventory), _without_write_bits(inventory)
    if actual_inventory != inventory:
        _invalid("Source tree does not match the snapshot's selected path inventory.")
    expected_members = {item["path"] for item in inventory if item["kind"] in {"file", "symlink"}}
    actual_members = _source_membership(source_root, manifest["ignored_paths"])
    if actual_members != expected_members:
        unexpected = sorted(actual_members - expected_members)
        missing = sorted(expected_members - actual_members)
        _invalid(f"Source membership differs from snapshot; unexpected={unexpected}, missing={missing}.")
    return {
        "source_root": str(source_root),
        "source_fingerprint_sha256": _snapshot_fingerprint(diff, snapshot_dir, untracked),
        "full_modes_verified": True,
        "inventory_entries_verified": len(inventory),
        "inventory_scope": "Recorded tracked and untracked source paths; captured ignored paths, Git metadata, and named Python caches are excluded.",
        "captured_ignored_path_count": len(manifest["ignored_paths"]),
    }


def _restore_tree(
    repository: Path,
    snapshot_dir: Path,
    destination: Path,
    commit: str,
    diff: bytes,
    untracked: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> None:
    archive_fd, archive_name = tempfile.mkstemp(prefix="source-archive-")
    os.close(archive_fd)
    archive = Path(archive_name)
    try:
        with archive.open("wb") as stream:
            _git(repository, "archive", "--format=tar", commit, stdout=stream, env=_clean_git_environment())
        _extract_archive(archive, destination)
        _validate_tree_links(destination)
        tracked_paths = _git(repository, "ls-tree", "-r", "-z", "--name-only", commit, stdout=subprocess.PIPE, env=_clean_git_environment()).stdout
        for raw in (part for part in tracked_paths.split(b"\0") if part):
            relative = _safe_relative(os.fsdecode(raw))
            if not (destination / relative).exists() and not (destination / relative).is_symlink():
                _invalid(f"Git archive omitted tracked source path: {relative}")
        _apply_tracked_diff(destination, diff)
        _restore_untracked(snapshot_dir, destination, untracked)
        _apply_inventory(destination, inventory)
    finally:
        archive.unlink(missing_ok=True)


def restore_source_snapshot(repository: Path, snapshot_path: Path, destination: Path) -> dict[str, Any]:
    """Restore a validated snapshot into a new empty isolated directory."""
    repository = repository.resolve()
    snapshot_dir, commit, diff, untracked, inventory = _load_snapshot(snapshot_path)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        _conflict(f"Restore destination must not exist: {destination}")
    if _is_below(destination, repository) or _is_below(destination, snapshot_dir):
        _invalid("Restore destination must be outside the repository and source snapshot.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        _restore_tree(repository, snapshot_dir, destination, commit, diff, untracked, inventory)
        verification = verify_source_snapshot(snapshot_path, destination)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"destination": str(destination), **verification, "full_modes_recoverable": True}


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve())
    except ValueError:
        return False
    else:
        return True


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        _invalid(f"Snapshot {label} SHA256 is invalid.")


def _patch_paths(diff: bytes) -> list[Path]:
    paths = []
    for line in diff.splitlines():
        if line.startswith(b"+++ b/"):
            value = line[6:].split(b"\t", 1)[0].decode("utf-8", errors="surrogateescape")
            if value != "/dev/null":
                paths.append(_safe_relative(value))
    return paths


def _validate_tree_id(tree_id: str) -> None:
    if _TREE_ID_PATTERN.fullmatch(tree_id) is None:
        _invalid(f"Source tree ID must be <commit12>-<fingerprint12> in lowercase hexadecimal: {tree_id!r}")


def _set_tree_write(root: Path, *, writable: bool) -> None:
    """Grant owner write to, or strip every write bit from, root and each real directory and file below it."""
    entries = [root]
    for current, names, files in os.walk(root, followlinks=False):
        entries.extend(path for path in (Path(current) / name for name in (*names, *files)) if not path.is_symlink())
    for entry in entries:
        mode = stat.S_IMODE(entry.stat().st_mode)
        entry.chmod(mode | stat.S_IWUSR if writable else mode & ~_ALL_WRITE_BITS)


@contextlib.contextmanager
def _tree_lock(trees_root: Path, tree_id: str, *, exclusive: bool) -> Iterator[None]:
    """Hold the per-tree lock: shared while a tree is reused or leased, exclusive while pruning decides and deletes."""
    directory = trees_root / _LOCKS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    # Readable on purpose: on the NFS mounts of the compute nodes flock becomes a POSIX lock, and a shared POSIX lock on a
    # write-only descriptor fails with EBADF (probe 873129, 2026-10-28).
    with (directory / f"{tree_id}.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _discard_directory(path: Path) -> None:
    """Delete a directory this process owns, including one frozen read-only."""
    if path.is_dir() and not path.is_symlink():
        _set_tree_write(path, writable=True)
        shutil.rmtree(path)


def verify_source_tree(tree: Path, snapshot_root: Path) -> dict[str, Any]:
    """
    Verify a source tree against the snapshot named like it, without Git.

    Parameters
    ----------
    tree : Path
        Source tree directory named ``<commit12>-<fingerprint12>``.
    snapshot_root : Path
        Root of the durable snapshots; ``<tree name>/snapshot.json`` is the evidence.

    Returns
    -------
    dict[str, Any]
        The snapshot verification result plus the ``tree_id`` and the snapshot's ``git_commit``.

    Raises
    ------
    ValueError
        If the tree is a symlink or not a directory, its name is not a snapshot
        name, or its bytes, links, or modes differ from the snapshot.
    FileNotFoundError
        If the matching snapshot manifest is absent.

    """
    _validate_tree_id(tree.name)
    if tree.is_symlink() or not tree.is_dir():
        _invalid(f"Source tree must be a real directory, not a symlink or file: {tree}")
    snapshot = snapshot_root / tree.name / "snapshot.json"
    if not snapshot.is_file():
        message = f"Snapshot for source tree {tree.name} is missing: {snapshot}"
        raise FileNotFoundError(message)
    commit = _authenticated_snapshot_commit(snapshot)
    verification = verify_source_snapshot(snapshot, tree, frozen=True)
    return {**verification, "tree_id": tree.name, "git_commit": commit}


def _authenticated_snapshot_commit(snapshot: Path) -> str:
    """Return the snapshot's commit after checking that its directory is named after its own commit and fingerprint."""
    manifest = json.loads(snapshot.read_bytes())
    commit = manifest.get("git_commit")
    fingerprint = manifest.get("source_fingerprint_sha256")
    if not isinstance(commit, str) or not isinstance(fingerprint, str):
        _invalid(f"Snapshot manifest lacks its commit or fingerprint: {snapshot}")
    expected = f"{commit[:_SNAPSHOT_NAME_PART_LENGTH]}-{fingerprint[:_SNAPSHOT_NAME_PART_LENGTH]}"
    if snapshot.parent.name != expected:
        _invalid(f"Snapshot directory {snapshot.parent.name} is not named after its own commit and fingerprint ({expected}).")
    return commit


def materialize_source_tree(repository: Path, snapshot_root: Path, trees_root: Path) -> dict[str, str]:
    """
    Return the verified read-only source tree of the repository's current code state.

    The state is captured as a snapshot (clean or dirty) and restored once into
    ``trees_root/<commit12>-<fingerprint12>``. An existing tree is verified and
    reused, never repaired or overwritten.

    Parameters
    ----------
    repository : Path
        Git checkout whose current state is frozen.
    snapshot_root : Path
        Durable root receiving the snapshot.
    trees_root : Path
        Root receiving the tree; it must lie outside the repository.

    Returns
    -------
    dict[str, str]
        ``tree_id``, ``tree_path``, ``snapshot_path``, ``snapshot_sha256``,
        ``git_commit`` and ``source_fingerprint_sha256``.

    Raises
    ------
    ValueError
        If an existing tree is unsafe or differs from its snapshot.
    FileExistsError
        If an existing snapshot of the same name conflicts with the current source.
    RuntimeError
        If the source changes while it is captured.

    """
    receipt = capture_source_snapshot(repository, snapshot_root)
    return _publish_tree(repository, Path(receipt["snapshot_path"]), trees_root)


def materialize_snapshot_tree(repository: Path, snapshot_path: Path, trees_root: Path) -> dict[str, str]:
    """
    Return the verified read-only source tree of an existing snapshot.

    Unlike :func:`materialize_source_tree` this never captures the repository's
    current state: it restores the recorded snapshot, for example to continue a
    unit bound to an older code state whose tree was pruned. The repository only
    supplies the snapshot's base commit.

    Parameters
    ----------
    repository : Path
        Git checkout that still contains the snapshot's base commit.
    snapshot_path : Path
        ``snapshot.json`` of the durable snapshot ``<snapshot_root>/<tree_id>/``.
    trees_root : Path
        Root receiving the tree; it must lie outside the repository.

    Returns
    -------
    dict[str, str]
        The same receipt as :func:`materialize_source_tree`.

    Raises
    ------
    ValueError
        If the snapshot is not named like a tree, or an existing tree differs from it.
    FileNotFoundError
        If the snapshot is absent.

    """
    _validate_tree_id(snapshot_path.parent.name)
    if snapshot_path.name != "snapshot.json" or snapshot_path.is_symlink() or not snapshot_path.is_file():
        message = f"Source snapshot manifest is missing or unsafe: {snapshot_path}"
        raise FileNotFoundError(message)
    return _publish_tree(repository, snapshot_path, trees_root)


def _publish_tree(repository: Path, snapshot_path: Path, trees_root: Path) -> dict[str, str]:
    """Reuse the verified tree of one snapshot or restore and publish it atomically."""
    snapshot_root = snapshot_path.parent.parent
    tree_id = snapshot_path.parent.name
    _authenticated_snapshot_commit(snapshot_path)
    trees_root.mkdir(parents=True, exist_ok=True)
    tree = trees_root / tree_id
    reused = False
    with _tree_lock(trees_root, tree_id, exclusive=False):
        if tree.exists() or tree.is_symlink():
            if not tree.is_symlink():
                os.utime(tree)  # mark the reused tree as recent under the lock, so a waiting prune rechecks and keeps it
            verify_source_tree(tree, snapshot_root)
            reused = True
    if not reused:
        staging = trees_root / f"{_STAGING_PREFIX}{tree_id}-{secrets.token_hex(4)}"
        try:
            restore_source_snapshot(repository, snapshot_path, staging)
            _set_tree_write(staging, writable=False)
        except BaseException:
            _discard_directory(staging)
            raise
        with _tree_lock(trees_root, tree_id, exclusive=False):
            try:
                staging.rename(tree)
            except OSError:
                _discard_directory(staging)
                if tree.is_symlink() or not tree.is_dir():
                    raise
                verify_source_tree(tree, snapshot_root)  # another process published the same tree first
    manifest = json.loads(snapshot_path.read_bytes())
    return {
        "tree_id": tree_id,
        "tree_path": str(tree),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _sha256_file(snapshot_path),
        "git_commit": manifest["git_commit"],
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
    }


def lease_source_tree(trees_root: Path, tree_id: str, job_id: str) -> Path:
    """
    Record that a Slurm job uses a source tree, so pruning keeps it.

    Parameters
    ----------
    trees_root : Path
        Root of the source trees.
    tree_id : str
        Tree name ``<commit12>-<fingerprint12>``.
    job_id : str
        Slurm job ID or array task ID ``<array>_<task>``.

    Returns
    -------
    Path
        The empty lease file; repeated calls with the same arguments are no-ops.

    Raises
    ------
    ValueError
        If the tree ID or job ID is malformed.
    FileNotFoundError
        If the tree does not exist as a real directory.

    """
    _validate_tree_id(tree_id)
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        _invalid(f"Lease ID must be a Slurm job ID or <array>_<task>: {job_id!r}")
    tree = trees_root / tree_id
    lease = trees_root / _LEASES_DIRNAME / tree_id / job_id
    with _tree_lock(trees_root, tree_id, exclusive=False):
        if tree.is_symlink() or not tree.is_dir():
            message = f"Cannot lease a missing source tree: {tree}"
            raise FileNotFoundError(message)
        lease.parent.mkdir(parents=True, exist_ok=True)
        lease.touch()
    return lease


def prune_source_trees(
    trees_root: Path,
    active_job_ids: set[str] | None,
    *,
    now: float,
    minimum_age_seconds: float = _PRUNE_MINIMUM_AGE_SECONDS,
) -> dict[str, list[str]]:
    """
    Delete source trees that no known job uses and that nobody touched recently.

    Parameters
    ----------
    trees_root : Path
        Root of the source trees.
    active_job_ids : set[str] | None
        Job IDs the scheduler still knows; ``None`` means the queue could not be
        read, which deletes nothing.
    now : float
        Current time in seconds since the epoch.
    minimum_age_seconds : float, optional
        A tree or lease younger than this, and an unpublished restore younger
        than this, is kept.

    Returns
    -------
    dict[str, list[str]]
        Names of removed and kept trees; removed also lists stale staging directories.

    """
    removed: list[str] = []
    kept: list[str] = []
    if not trees_root.is_dir():
        return {"removed": removed, "kept": kept}
    cutoff = now - minimum_age_seconds
    for entry in sorted(trees_root.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name.startswith(_STAGING_PREFIX):
            if active_job_ids is not None and entry.stat().st_mtime < cutoff:
                _discard_directory(entry)
                removed.append(entry.name)
            continue
        if _TREE_ID_PATTERN.fullmatch(entry.name) is None:
            continue
        if active_job_ids is None:
            kept.append(entry.name)
            continue
        # Decide and delete under the exclusive tree lock, so a launcher that reuses or leases the tree meanwhile is seen.
        with _tree_lock(trees_root, entry.name, exclusive=True):
            lease_directory = trees_root / _LEASES_DIRNAME / entry.name
            leases = sorted(lease_directory.iterdir()) if lease_directory.is_dir() else []
            in_use = any(lease.name in active_job_ids for lease in leases)
            newest = max([entry.stat().st_mtime, *(lease.stat().st_mtime for lease in leases)])
            if in_use or newest > cutoff:
                kept.append(entry.name)
                continue
            _discard_directory(entry)
            shutil.rmtree(lease_directory, ignore_errors=True)
            removed.append(entry.name)
    return {"removed": removed, "kept": kept}


def _active_slurm_job_ids() -> set[str] | None:
    """Return the current user's queued or running job IDs, or None when the queue cannot be read."""
    try:
        result = subprocess.run(  # noqa: S603 -- fixed executable and argument vector
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],  # noqa: S607 -- scheduler client resolved from PATH
            check=True,
            capture_output=True,
            text=True,
            timeout=_SQUEUE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, KeyError):
        return None
    identifiers: set[str] = set()
    for line in result.stdout.splitlines():
        job_id = line.strip()
        if job_id:
            identifiers.add(job_id)
            identifiers.add(job_id.split("_", 1)[0])
    return identifiers


if __name__ == "__main__":
    if len(sys.argv) == _FINGERPRINT_ARGUMENT_COUNT:
        print(source_fingerprint(Path(sys.argv[1])))
    elif len(sys.argv) == _RESTORE_ARGUMENT_COUNT and sys.argv[2] == "--restore":
        print(json.dumps(restore_source_snapshot(Path(sys.argv[1]), Path(sys.argv[3]), Path(sys.argv[4])), sort_keys=True))
    elif len(sys.argv) == _MATERIALIZE_ARGUMENT_COUNT and sys.argv[2] == "--materialize":
        print(json.dumps(materialize_source_tree(Path(sys.argv[1]), Path(sys.argv[3]), Path(sys.argv[4])), sort_keys=True))
    elif len(sys.argv) == _MATERIALIZE_ARGUMENT_COUNT and sys.argv[2] == "--materialize-snapshot":
        print(json.dumps(materialize_snapshot_tree(Path(sys.argv[1]), Path(sys.argv[3]), Path(sys.argv[4])), sort_keys=True))
    elif len(sys.argv) == _VERIFY_TREE_ARGUMENT_COUNT and sys.argv[1] == "--verify-tree":
        print(json.dumps(verify_source_tree(Path(sys.argv[2]), Path(sys.argv[3])), sort_keys=True))
    elif len(sys.argv) == _LEASE_ARGUMENT_COUNT and sys.argv[1] == "--lease":
        print(lease_source_tree(Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
    elif len(sys.argv) == _PRUNE_ARGUMENT_COUNT and sys.argv[1] == "--prune":
        print(json.dumps(prune_source_trees(Path(sys.argv[2]), _active_slurm_job_ids(), now=time.time()), sort_keys=True))
    else:
        usage = (
            f"Usage: {sys.argv[0]} REPOSITORY [--restore SNAPSHOT_JSON DESTINATION"
            " | --materialize SNAPSHOT_ROOT TREES_ROOT | --materialize-snapshot SNAPSHOT_JSON TREES_ROOT]\n"
            f"       {sys.argv[0]} --verify-tree TREE SNAPSHOT_ROOT | --lease TREES_ROOT TREE_ID LEASE_ID | --prune TREES_ROOT"
        )
        raise SystemExit(usage)
