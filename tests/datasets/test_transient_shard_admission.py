"""Protect portable PT admission without relaxing immutable file integrity.

Responsibilities:
  - Exercise relocated or copied files, content verification and local cache safety
  - Keep changed size, changed bytes and unsafe paths fail closed
  - Cover canonical-index and shard receipt admission using tiny metadata fixtures
  - Reject shards whose bytes match refreshed receipt evidence but whose case payload contradicts the Dataset index

Design principles:
  - Unchanged stat evidence needs no hash; any other stat change except size is admitted only by
    the published digest
  - Verification cache entries bind the path, size, observed stat identity, and expected digest
  - Semantic case binding is checked after, and independently of, byte-integrity admission

This module does NOT:
  - fit scalers or train models, and builds Sources only through the shared compact transient package fixture
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest
import torch

from src import common
from src.datasets.packages import dataset_packages_transient_shards as owner

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def clear_content_cache() -> Iterator[None]:
    """Keep process-local verification evidence independent between cases."""
    owner._verified_portable_file_content.cache_clear()
    yield
    owner._verified_portable_file_content.cache_clear()


@pytest.fixture
def published_file(tmp_path: Path) -> tuple[Path, int, dict[str, int], str]:
    """Return real tiny bytes and publication evidence from another device."""
    path = tmp_path / "published.bin"
    path.write_bytes(b"good")
    size, observed = owner._file_stat_identity(path)
    return path, size, {**observed, "device": observed["device"] + 1}, common.serialization.file_sha256(path)


def _matches(published: tuple[Path, int, dict[str, int], str]) -> bool:
    path, size, stat, digest = published
    return owner._published_file_matches(path, expected_size=size, expected_stat=stat, expected_sha256=digest)


def test_unchanged_local_evidence_does_not_hash(published_file: tuple[Path, int, dict[str, int], str], monkeypatch: pytest.MonkeyPatch) -> None:
    path, size, _, digest = published_file
    _, observed = owner._file_stat_identity(path)
    monkeypatch.setattr(common.serialization, "file_sha256", lambda _path: pytest.fail("unchanged evidence rehashed"))
    assert _matches((path, size, observed, digest))


@pytest.mark.parametrize("field", ["device", "inode", "mtime_ns", "ctime_ns"])
def test_stat_change_hashes_once(published_file: tuple[Path, int, dict[str, int], str], monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    path, size, _, digest = published_file
    _, observed = owner._file_stat_identity(path)
    changed = (path, size, {**observed, field: observed[field] + 1}, digest)
    calls: list[Path] = []
    real_hash = common.serialization.file_sha256

    def observed_hash(path: Path) -> str:
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr(common.serialization, "file_sha256", observed_hash)
    assert _matches(changed)
    assert _matches(changed)
    assert calls == [path]


def test_stat_change_with_corrupt_content_is_rejected(published_file: tuple[Path, int, dict[str, int], str]) -> None:
    path, size, _, digest = published_file
    path.write_bytes(b"evil")
    _, changed = owner._file_stat_identity(path)
    # Even matching portable stat evidence cannot replace the published digest.
    assert not _matches((path, size, {**changed, "device": changed["device"] + 1}, digest))


def test_size_change_never_hashes(published_file: tuple[Path, int, dict[str, int], str], monkeypatch: pytest.MonkeyPatch) -> None:
    path, size, expected, digest = published_file
    monkeypatch.setattr(common.serialization, "file_sha256", lambda _path: pytest.fail("size mismatch was bypassed"))
    assert not _matches((path, size + 1, expected, digest))


@pytest.mark.parametrize("size", [None, True])
def test_malformed_size_is_rejected(published_file: tuple[Path, int, dict[str, int], str], size: int | None) -> None:
    path, _, expected, digest = published_file
    assert not owner._published_file_matches(path, expected_size=size, expected_stat=expected, expected_sha256=digest)


def test_cache_binds_expected_hash(published_file: tuple[Path, int, dict[str, int], str]) -> None:
    path, size, expected, _ = published_file
    assert _matches(published_file)
    assert not _matches((path, size, expected, "0" * 64))


def test_cache_does_not_admit_later_write(published_file: tuple[Path, int, dict[str, int], str]) -> None:
    path, _, _, _ = published_file
    assert _matches(published_file)
    path.write_bytes(b"evil")
    observed = path.stat()
    os.utime(path, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000))
    assert not _matches(published_file)


def test_cache_binds_path(published_file: tuple[Path, int, dict[str, int], str], monkeypatch: pytest.MonkeyPatch) -> None:
    path, size, expected, digest = published_file
    other = path.with_name("other.bin")
    other.write_bytes(b"evil")
    _, observed = owner._file_stat_identity(path)
    # Equal local stat tokens in distinct namespaces must not alias cache entries.
    monkeypatch.setattr(owner, "_file_stat_identity", lambda _path: (size, observed))
    assert _matches(published_file)
    assert not _matches((other, size, expected, digest))


def test_write_during_hash_fails_closed(published_file: tuple[Path, int, dict[str, int], str], monkeypatch: pytest.MonkeyPatch) -> None:
    real_hash = common.serialization.file_sha256

    def changing_hash(path: Path) -> str:
        result = real_hash(path)
        path.write_bytes(b"replacement")
        return result

    monkeypatch.setattr(common.serialization, "file_sha256", changing_hash)
    assert not _matches(published_file)


def test_symlink_is_not_admitted(published_file: tuple[Path, int, dict[str, int], str]) -> None:
    path, size, expected, digest = published_file
    link = path.with_name("alias.bin")
    link.symlink_to(path.name)
    assert not _matches((link, size, expected, digest))


@pytest.fixture
def receipt_fixture(tmp_path: Path) -> dict[str, Any]:
    """Publish tiny metadata evidence; tensor semantics are tested on real PT."""
    directory = tmp_path / "transient_pt_shards"
    directory.mkdir()
    index = {"index_digest": "5" * 64, "contract_digest": "6" * 64, "cases": [{"package_case_id": "case"}], "samples": [{"case_index": 0}]}
    canonical = tmp_path / "dataset.json"
    common.serialization.atomic_write_json(canonical, index)
    manifest = {
        "dataset_id": "fixture",
        "dataset_digest": "1" * 64,
        "payload_filename": canonical.name,
        "payload_sha256": common.serialization.file_sha256(canonical),
    }
    shard = directory / "shard_00000.pt"
    torch.save({"fixture": torch.tensor([1])}, shard)
    canonical_size, canonical_stat = owner._file_stat_identity(canonical)
    shard_size, shard_stat = owner._file_stat_identity(shard)
    publication = owner._publication_identity_from_dataset(manifest, index, "4" * 64)
    record = {
        "filename": shard.name,
        "shard_index": 0,
        "case_ids": ["case"],
        "case_count": 1,
        "transition_count": 1,
        "tensor_bytes": 8,
        "size_bytes": shard_size,
        "stat_identity": shard_stat,
        "sha256": common.serialization.file_sha256(shard),
        "oversized_single_case": False,
        "oversized_reason": None,
    }
    receipt = {
        "schema_kind": owner.TRANSIENT_PT_RECEIPT_SCHEMA_KIND,
        "schema_version": owner.TRANSIENT_PT_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "created_at": "fixture",
        "dataset_id": manifest["dataset_id"],
        "dataset_digest": manifest["dataset_digest"],
        "index_digest": index["index_digest"],
        "transient_contract_digest": index["contract_digest"],
        "package_manifest_sha256": "4" * 64,
        "canonical_payload_sha256": manifest["payload_sha256"],
        "canonical_payload_size_bytes": canonical_size,
        "canonical_payload_stat_identity": canonical_stat,
        "publication_identity": publication,
        "target_shard_bytes": 1_000_000,
        "derived_payload_id": owner._derived_payload_id(
            manifest=manifest, index=index, publication_identity=publication, target_shard_bytes=1_000_000, case_groups=[["case"]]
        ),
        "shard_count": 1,
        "case_count": 1,
        "transition_count": 1,
        "total_size_bytes": shard_size,
        "case_locator": {"case": {"shard_index": 0, "case_position": 0, "case_index": 0}},
        "shards": [record],
        "shard_inventory_digest": common.serialization.canonical_json_sha256([record]),
    }
    return {
        "directory": directory,
        "canonical": canonical,
        "shard": shard,
        "manifest": manifest,
        "index": index,
        "receipt": receipt,
        "publication": publication,
    }


def _admit(fixture: dict[str, Any]) -> dict[str, Any]:
    size, stat = owner._file_stat_identity(fixture["canonical"])
    return owner._validate_receipt_directory(
        fixture["directory"],
        fixture["receipt"],
        manifest=fixture["manifest"],
        index=fixture["index"],
        manifest_sha256="4" * 64,
        canonical_payload_size_bytes=size,
        canonical_payload_stat_identity=stat,
        canonical_payload_path=fixture["canonical"],
        publication_identity=fixture["publication"],
        validation_depth="evidence",
    )


@pytest.mark.parametrize("field", ["device", "inode", "ctime_ns"])
@pytest.mark.parametrize("member", ["canonical", "shard"])
def test_receipt_stat_change_requires_content(receipt_fixture: dict[str, Any], member: str, field: str) -> None:
    receipt = receipt_fixture["receipt"]
    stat = receipt["canonical_payload_stat_identity"] if member == "canonical" else receipt["shards"][0]["stat_identity"]
    stat[field] += 1
    receipt["shard_inventory_digest"] = common.serialization.canonical_json_sha256(receipt["shards"])
    original = common.serialization.canonical_json_sha256(receipt)
    assert _admit(receipt_fixture) == receipt
    assert common.serialization.canonical_json_sha256(receipt) == original
    owner._verified_portable_file_content.cache_clear()
    path = receipt_fixture[member]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    _, current = owner._file_stat_identity(path)
    # Retain the immutable content digest while the stat evidence differs again.
    stat.update(current)
    stat[field] += 1
    receipt["shard_inventory_digest"] = common.serialization.canonical_json_sha256(receipt["shards"])
    with pytest.raises(owner.TransientShardContractError, match=r"identity is invalid|changed after immutable"):
        _admit(receipt_fixture)


def _republish_shard(package: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite shard 0 with a mutated payload and refresh its size, stat, digest and inventory evidence like the producer."""
    dataset_id, root = package["train_dataset"], package["root"]
    receipt_path = owner.transient_shard_directory(dataset_id, storage_root=root) / owner.TRANSIENT_PT_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = receipt["shards"][0]
    shard = receipt_path.parent / record["filename"]
    payload = torch.load(shard, map_location="cpu", weights_only=True)
    mutate(payload)
    common.serialization.atomic_torch_save(payload, shard)
    size_bytes, stat_identity = owner._file_stat_identity(shard)
    record.update(size_bytes=size_bytes, stat_identity=stat_identity, sha256=common.serialization.file_sha256(shard))
    receipt["total_size_bytes"] = sum(item["size_bytes"] for item in receipt["shards"])
    receipt["shard_inventory_digest"] = common.serialization.canonical_json_sha256(receipt["shards"])
    common.serialization.atomic_write_json(receipt_path, receipt)


def _reorder_dynamic_fields(payload: dict[str, Any]) -> None:
    order = payload["cases"][0]["field_order"]
    order["dynamic"] = list(reversed(order["dynamic"]))


def _swap_case_values(key: str) -> Callable[[dict[str, Any]], None]:
    """Bind the payload value of one case to the neighbouring case's evidence."""

    def swap(payload: dict[str, Any]) -> None:
        first, second = payload["cases"][:2]
        first[key], second[key] = second[key], first[key]

    return swap


@pytest.mark.parametrize(
    "corrupt",
    [_reorder_dynamic_fields, _swap_case_values("samples"), _swap_case_values("case_record")],
    ids=["field_order", "sample_binding", "case_record_binding"],
)
def test_hash_consistent_shard_with_wrong_case_semantics_is_rejected(
    transient_package: dict[str, Any], corrupt: Callable[[dict[str, Any]], None]
) -> None:
    """Refuse a shard whose bytes match its refreshed evidence but whose case payload contradicts the Dataset index.

    An unchanged rewrite is admitted first, so the failure after corruption comes from the semantic case checks
    and not from stale size, stat or digest evidence.
    """
    dataset_id, root = transient_package["train_dataset"], transient_package["root"]

    def reopen() -> None:
        owner.load_transient_shard_receipt(dataset_id, storage_root=root, validation_depth="full")
        owner.load_transient_shard_payload(dataset_id, 0, storage_root=root)

    _republish_shard(transient_package, lambda _payload: None)
    reopen()

    _republish_shard(transient_package, corrupt)
    with pytest.raises(owner.TransientShardContractError):
        owner.load_transient_shard_receipt(dataset_id, storage_root=root, validation_depth="full")
    with pytest.raises(owner.TransientShardContractError):
        owner.load_transient_shard_payload(dataset_id, 0, storage_root=root)
