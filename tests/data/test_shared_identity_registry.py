"""Protect the shared identity registry that keeps one content digest to one existing target.

Responsibilities:
  - Check reserve, lookup, stale replacement, containment and concurrent reservation
  - Check that malformed entries fail closed and that the caller's conflict error is raised unchanged

Design principles:
  - The registry is exercised with plain directories and digests, independent of runs and Datasets
  - Identity is the digest alone; target names never decide it

This module does NOT:
  - Compute run or Dataset digests; the run and Dataset suites cover their own admission
"""

from __future__ import annotations

import json
import shutil
import threading
from typing import TYPE_CHECKING, Any

import pytest

from src import common

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_DIGEST = "a" * 64
_KINDS = ("run", "trial")


class _DuplicateError(FileExistsError):
    """Test-owned duplicate error that keeps both targets."""

    def __init__(self, existing: Path, requested: Path) -> None:
        self.existing = existing
        self.requested = requested
        super().__init__(f"{existing} already owns the digest; refused {requested}")


def _registry(root: Path) -> common.identity_registry.IdentityRegistry:
    return common.identity_registry.IdentityRegistry(root, _KINDS)


def _bind(registry: common.identity_registry.IdentityRegistry, target: Path, *, kind: str = "run", digest: str = _DIGEST) -> None:
    """Reserve ``digest`` for ``target`` and allocate it, as a fresh start does."""
    with registry.reserve(digest, target, kind=kind, conflict=_DuplicateError):
        target.mkdir(parents=True)


def test_reserve_binds_the_target_relative_to_the_root_and_lookup_reports_the_owner(tmp_path: Path) -> None:
    """A reserved digest is found again until its target disappears; the entry stores a root-relative path."""
    registry = _registry(tmp_path)
    target = tmp_path / "task" / "name"
    assert registry.lookup(_DIGEST) is None

    _bind(registry, target, kind="trial")

    entry = registry.lookup(_DIGEST)
    assert entry is not None
    assert (entry.digest, entry.target, entry.kind) == (_DIGEST, target, "trial")
    stored = json.loads(common.paths.resolve_identity_registry_entry(tmp_path, _DIGEST).read_text(encoding="utf-8"))
    assert stored["target"] == "task/name"
    assert common.paths.resolve_identity_registry_lock(tmp_path, _DIGEST).parent.is_dir()


def test_reserve_rejects_another_existing_target_and_replaces_a_stale_entry(tmp_path: Path) -> None:
    """One digest owns one target; an entry whose target is gone is replaced, never reused silently."""
    registry = _registry(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _bind(registry, first)

    with pytest.raises(_DuplicateError) as rejected:
        _bind(registry, second)
    registry.ensure_available(_DIGEST, first, conflict=_DuplicateError)
    with pytest.raises(_DuplicateError):
        registry.ensure_available(_DIGEST, second, conflict=_DuplicateError)

    assert (rejected.value.existing, rejected.value.requested) == (first, second)
    assert not second.exists()
    shutil.rmtree(first)  # The entry is stale now: its target no longer exists.
    assert registry.lookup(_DIGEST) is None
    registry.ensure_available(_DIGEST, second, conflict=_DuplicateError)

    _bind(registry, second)

    entry = registry.lookup(_DIGEST)
    assert entry is not None
    assert entry.target == second


def test_reserve_binds_nothing_when_allocation_fails(tmp_path: Path) -> None:
    """A failed allocation leaves no entry that could block a retry."""
    registry = _registry(tmp_path)

    def fail_allocation() -> None:
        with registry.reserve(_DIGEST, tmp_path / "run", kind="run", conflict=_DuplicateError):
            message = "allocation failed"
            raise OSError(message)

    with pytest.raises(OSError, match="allocation failed"):
        fail_allocation()

    assert registry.lookup(_DIGEST) is None
    assert not common.paths.resolve_identity_registry_entry(tmp_path, _DIGEST).exists()


def test_reserve_binds_before_the_target_is_published(tmp_path: Path) -> None:
    """The entry exists while the caller publishes, so a crash after publication never leaves an unbound target."""
    registry = _registry(tmp_path)
    target = tmp_path / "task" / "name"
    entry = common.paths.resolve_identity_registry_entry(tmp_path, _DIGEST)

    with registry.reserve(_DIGEST, target, kind="run", conflict=_DuplicateError):
        assert json.loads(entry.read_text(encoding="utf-8"))["target"] == "task/name"
        assert registry.lookup(_DIGEST) is None  # Not yet published: the binding counts as stale until the target exists.
        target.mkdir(parents=True)

    owner = registry.lookup(_DIGEST)
    assert owner is not None
    assert owner.target == target


def test_ensure_bound_repairs_a_missing_binding_and_rejects_another_owner(tmp_path: Path) -> None:
    """An existing, admitted target gets its binding back; a digest owned by another target is refused."""
    registry = _registry(tmp_path)
    first = tmp_path / "task" / "first"
    first.mkdir(parents=True)

    registry.ensure_bound(_DIGEST, first, kind="run", conflict=_DuplicateError)
    owner = registry.lookup(_DIGEST)
    assert owner is not None
    assert owner.target == first

    second = tmp_path / "task" / "second"
    second.mkdir()
    with pytest.raises(_DuplicateError):
        registry.ensure_bound(_DIGEST, second, kind="run", conflict=_DuplicateError)
    with pytest.raises(ValueError, match="existing target"):
        registry.ensure_bound("b" * 64, tmp_path / "task" / "absent", kind="run", conflict=_DuplicateError)


def test_targets_outside_the_root_and_unknown_kinds_are_refused(tmp_path: Path) -> None:
    """The namespace root contains every target; a kind outside the registry's set never reaches an entry."""
    registry = _registry(tmp_path / "namespace")

    elsewhere = tmp_path / "elsewhere"
    with pytest.raises(ValueError, match="outside its namespace root"), registry.reserve(_DIGEST, elsewhere, kind="run", conflict=_DuplicateError):
        pytest.fail("a target outside the root must never be admitted")
    with pytest.raises(ValueError, match="outside its namespace root"):
        registry.ensure_available(_DIGEST, elsewhere, conflict=_DuplicateError)
    inside = tmp_path / "namespace" / "x"
    with pytest.raises(ValueError, match="not one of"), registry.reserve(_DIGEST, inside, kind="dataset", conflict=_DuplicateError):
        pytest.fail("an unknown kind must never be admitted")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entry: entry.update(digest="b" * 64),
        lambda entry: entry.update(target="../escape"),
        lambda entry: entry.update(target="/absolute"),
        lambda entry: entry.update(kind="dataset"),
        lambda entry: entry.update(schema_version=2),
        lambda entry: entry.update(extra="field"),
    ],
    ids=["digest", "parent-target", "absolute-target", "kind", "schema", "extra-key"],
)
def test_malformed_entries_fail_closed(tmp_path: Path, mutate: Callable[[dict[str, Any]], object]) -> None:
    """A corrupt entry is an error, never a free slot for a duplicate."""
    registry = _registry(tmp_path)
    _bind(registry, tmp_path / "run")
    entry_path = common.paths.resolve_identity_registry_entry(tmp_path, _DIGEST)
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    mutate(entry)
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Identity registry entry is invalid"):
        registry.lookup(_DIGEST)
    entry_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        registry.lookup(_DIGEST)


def test_concurrent_reservations_of_one_digest_admit_exactly_one_target(tmp_path: Path) -> None:
    """Two simultaneous fresh starts of the same identity cannot both allocate a target."""
    registry = _registry(tmp_path)
    targets = [tmp_path / "left", tmp_path / "right"]
    barrier = threading.Barrier(len(targets))
    outcomes: dict[str, str] = {}

    def start(target: Path) -> None:
        barrier.wait(timeout=5)
        try:
            _bind(registry, target)
        except _DuplicateError:
            outcomes[target.name] = "rejected"
        else:
            outcomes[target.name] = "admitted"

    threads = [threading.Thread(target=start, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(outcomes.values()) == ["admitted", "rejected"]
    assert sum(target.exists() for target in targets) == 1
