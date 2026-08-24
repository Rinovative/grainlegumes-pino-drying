"""
dataset_packages_references.py

Publish and resolve immutable task-local Dataset-reference records.

Responsibilities:
  - Validate concise logical Dataset references and self-contained record schemas
  - Bind references to admitted immutable package and manifest identities
  - Publish no-overwrite reference records under per-reference advisory locks
  - Resolve and inspect bounded reference evidence without copying package payloads

Design principles:
  - Logical names are task-local while exact package identities remain authoritative
  - Reference records are immutable metadata with complete identity evidence
  - Publication and resolution fail closed on unsafe, missing, or mismatched state

This module does NOT:
  - Build Dataset packages, select revisions, or rewrite existing references
  - Copy package payloads, create symlinks, or maintain mutable latest aliases
"""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from src import common, domain
from src.datasets.contracts import dataset_contracts_views as views

from . import dataset_packages_manifest as package_manifest

DATASET_REFERENCE_SCHEMA_KIND: Final = "vp2_dataset_reference"
DATASET_REFERENCE_SCHEMA_VERSION: Final = 1
_DATASET_REFERENCE_NAME_PATTERN: Final = re.compile(r"[A-Za-z0-9._+-]{1,96}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_REFERENCE_RECORD_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "task",
        "name",
        "revision",
        "dataset_id",
        "dataset_digest",
        "manifest_sha256",
        "payload_sha256",
        "dataset_view",
        "evaluation_regime",
        "materials",
        "source_package",
        "creation",
    }
)
_SOURCE_PACKAGE_KEYS: Final = frozenset(
    {
        "dataset_name",
        "campaign_id",
        "campaign_digest",
        "source_case_count",
        "source_batch_ids",
        "source_simulation_profiles",
        "source_git_commits",
        "channel_contract_digest",
    }
)
_CREATION_KEYS: Final = frozenset({"created_at", "publisher"})


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """One validated task-local logical Dataset reference."""

    name: str
    revision: int

    def __post_init__(self) -> None:
        """Validate the frozen contract at direct construction."""
        _validate_reference_name(self.name)
        _validate_revision(self.revision)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetRef:
        """Create one Dataset reference from its exact authored mapping."""
        if not isinstance(value, Mapping) or set(value) != {"name", "revision"}:
            message = "Dataset reference must contain exactly name and revision."
            raise ValueError(message)
        return cls(name=_validate_reference_name(value["name"]), revision=_validate_revision(value["revision"]))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible authored representation."""
        return {"name": self.name, "revision": self.revision}

    @property
    def display_name(self) -> str:
        """Return the concise human label with revision zero omitted."""
        return self.name if self.revision == 0 else f"{self.name}_d{self.revision}"


def _validate_reference_name(value: object) -> str:
    """Require one concise safe logical Dataset-reference name."""
    if not isinstance(value, str) or _DATASET_REFERENCE_NAME_PATTERN.fullmatch(value) is None:
        message = "Dataset reference name must contain 1 to 96 ASCII alphanumeric, '.', '_', '+', or '-' characters."
        raise ValueError(message)
    common.paths.validate_logical_name(value, label="dataset reference name")
    return value


def _validate_task(value: object) -> str:
    """Require one registered safe task identifier."""
    task = common.paths.validate_logical_name(value, label="task")
    domain.tasks.registry.get_task(task)
    return task


def _validate_revision(value: object) -> int:
    """Require one explicit non-boolean non-negative revision integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"Dataset reference revision must be a non-negative integer, got {value!r}."
        raise ValueError(message)
    return value


def _require_sha256(value: object, *, label: str) -> str:
    """Require one lowercase SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        message = f"Dataset reference {label} must be a lowercase SHA-256 digest."
        raise ValueError(message)
    return value


def _require_string_sequence(value: object, *, label: str) -> list[str]:
    """Require one non-empty unique sequence of non-empty strings."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        message = f"Dataset reference {label} must be a non-empty string list."
        raise ValueError(message)
    if len(set(value)) != len(value):
        message = f"Dataset reference {label} must not contain duplicates."
        raise ValueError(message)
    return list(value)


def _validate_source_package(value: object) -> dict[str, Any]:
    """Validate the bounded source-package provenance carried by a record."""
    if not isinstance(value, dict) or set(value) != _SOURCE_PACKAGE_KEYS:
        message = "Dataset reference source_package keys do not match the current schema."
        raise ValueError(message)
    if not isinstance(value["dataset_name"], str) or not value["dataset_name"]:
        raise ValueError("Dataset reference source_package.dataset_name must be a non-empty string.")
    if not isinstance(value["campaign_id"], str) or not value["campaign_id"]:
        raise ValueError("Dataset reference source_package.campaign_id must be a non-empty string.")
    _require_sha256(value["campaign_digest"], label="source_package.campaign_digest")
    if isinstance(value["source_case_count"], bool) or not isinstance(value["source_case_count"], int) or value["source_case_count"] < 1:
        raise ValueError("Dataset reference source_package.source_case_count must be a positive integer.")
    _require_string_sequence(value["source_batch_ids"], label="source_package.source_batch_ids")
    _require_string_sequence(value["source_simulation_profiles"], label="source_package.source_simulation_profiles")
    _require_string_sequence(value["source_git_commits"], label="source_package.source_git_commits")
    _require_sha256(value["channel_contract_digest"], label="source_package.channel_contract_digest")
    return dict(value)


def _validate_creation(value: object) -> dict[str, str]:
    """Validate immutable reference creation metadata without reading external state."""
    if not isinstance(value, dict) or set(value) != _CREATION_KEYS:
        raise ValueError("Dataset reference creation keys do not match the current schema.")
    if value["publisher"] != "dataset_reference" or not isinstance(value["created_at"], str):
        raise ValueError("Dataset reference creation metadata is invalid.")
    try:
        parsed = datetime.fromisoformat(value["created_at"])
    except ValueError as error:
        raise ValueError("Dataset reference creation.created_at must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError("Dataset reference creation.created_at must be timezone-aware.")
    return {"created_at": value["created_at"], "publisher": value["publisher"]}


def validate_dataset_reference_record(value: Any) -> dict[str, Any]:
    """Validate one self-contained Dataset-reference record without external reads."""
    if not isinstance(value, dict) or set(value) != _REFERENCE_RECORD_KEYS:
        raise ValueError("Dataset reference record keys do not match the current schema.")
    task = _validate_task(value["task"])
    name = _validate_reference_name(value["name"])
    revision = _validate_revision(value["revision"])
    dataset_id = common.paths.validate_logical_name(value["dataset_id"], label="dataset_id")
    if (
        value["schema_kind"] != DATASET_REFERENCE_SCHEMA_KIND
        or value["schema_version"] != DATASET_REFERENCE_SCHEMA_VERSION
        or not isinstance(value["dataset_view"], str)
        or not value["dataset_view"]
        or not isinstance(value["evaluation_regime"], str)
        or not value["evaluation_regime"]
    ):
        raise ValueError("Dataset reference record identity fields are invalid.")
    materials = _require_string_sequence(value["materials"], label="materials")
    return {
        "schema_kind": DATASET_REFERENCE_SCHEMA_KIND,
        "schema_version": DATASET_REFERENCE_SCHEMA_VERSION,
        "task": task,
        "name": name,
        "revision": revision,
        "dataset_id": dataset_id,
        "dataset_digest": _require_sha256(value["dataset_digest"], label="dataset_digest"),
        "manifest_sha256": _require_sha256(value["manifest_sha256"], label="manifest_sha256"),
        "payload_sha256": _require_sha256(value["payload_sha256"], label="payload_sha256"),
        "dataset_view": value["dataset_view"],
        "evaluation_regime": value["evaluation_regime"],
        "materials": materials,
        "source_package": _validate_source_package(value["source_package"]),
        "creation": _validate_creation(value["creation"]),
    }


def _manifest_path(dataset_id: str, *, storage_root: Path | str | None) -> Path:
    """Return the canonical package manifest path for one exact dataset ID."""
    return common.paths.get_dataset_metadata_root(storage_root=storage_root) / dataset_id / "dataset_manifest.json"


def _record_from_manifest(task: str, reference: DatasetRef, manifest: Mapping[str, Any], *, storage_root: Path | str | None) -> dict[str, Any]:
    """Derive one complete immutable reference record from an admitted manifest."""
    view = views.get_view(str(manifest["dataset_view"]))
    view_task = view.id if view.registered_task_id is None else view.registered_task_id
    if manifest["registered_task_id"] != view.registered_task_id or view_task != task:
        message = (
            f"Dataset package {manifest['dataset_id']!r} is incompatible with task {task!r}: "
            f"manifest task is {manifest['registered_task_id']!r} and view task is {view_task!r}."
        )
        raise ValueError(message)
    manifest_path = _manifest_path(str(manifest["dataset_id"]), storage_root=storage_root)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"Dataset package manifest is missing or unsafe: {manifest_path}.")
    return validate_dataset_reference_record(
        {
            "schema_kind": DATASET_REFERENCE_SCHEMA_KIND,
            "schema_version": DATASET_REFERENCE_SCHEMA_VERSION,
            "task": task,
            "name": reference.name,
            "revision": reference.revision,
            "dataset_id": manifest["dataset_id"],
            "dataset_digest": manifest["dataset_digest"],
            "manifest_sha256": common.serialization.file_sha256(manifest_path),
            "payload_sha256": manifest["payload_sha256"],
            "dataset_view": manifest["dataset_view"],
            "evaluation_regime": manifest["evaluation_regime"],
            "materials": list(manifest["materials"]),
            "source_package": {
                "dataset_name": manifest["dataset_name"],
                "campaign_id": manifest["campaign_id"],
                "campaign_digest": manifest["campaign_digest"],
                "source_case_count": manifest["source_case_count"],
                "source_batch_ids": sorted(manifest["source_batch_ids"]),
                "source_simulation_profiles": sorted(manifest["source_simulation_profiles"]),
                "source_git_commits": sorted(manifest["source_git_commits"]),
                "channel_contract_digest": manifest["channel_contract_digest"],
            },
            "creation": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "publisher": "dataset_reference",
            },
        }
    )


def _read_record(path: Path) -> dict[str, Any]:
    """Read one safe reference record and validate its pure schema."""
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Dataset reference record is missing or unsafe: {path}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Dataset reference record is unreadable: {path}.") from error
    return validate_dataset_reference_record(raw)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one JSON file without replacing an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = f"{json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)}\n".encode()
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_conflict_message(
    task: str,
    reference: DatasetRef,
    existing: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> str:
    """Return one actionable immutable-binding conflict diagnostic."""
    return (
        "Dataset reference conflict: "
        f"task={task!r}, name={reference.name!r}, revision={reference.revision}; "
        f"existing dataset_id={existing['dataset_id']!r}, requested dataset_id={requested['dataset_id']!r}. "
        "Action: inspect the existing binding and choose an explicit new revision."
    )


def publish_dataset_reference(
    task: str,
    name: str,
    revision: int,
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fully admit a package then atomically publish or reuse one exact reference binding."""
    task = _validate_task(task)
    reference = DatasetRef(name=name, revision=revision)
    dataset_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    manifest = package_manifest.load_package_manifest(dataset_id, storage_root=storage_root)
    requested = _record_from_manifest(task, reference, manifest, storage_root=storage_root)
    path = common.paths.resolve_dataset_reference_path(task, reference.name, reference.revision, storage_root=storage_root)
    lock_path = common.paths.resolve_dataset_reference_lock_path(task, reference.name, reference.revision, storage_root=storage_root)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if path.exists() or path.is_symlink():
            existing = _read_record(path)
            if (
                existing["dataset_id"] == requested["dataset_id"]
                and existing["manifest_sha256"] == requested["manifest_sha256"]
                and existing["payload_sha256"] == requested["payload_sha256"]
            ):
                return {**existing, "status": "reused"}
            raise FileExistsError(_reference_conflict_message(task, reference, existing, requested))
        try:
            _atomic_create_json(path, requested)
        except FileExistsError as error:
            existing = _read_record(path)
            if (
                existing["dataset_id"] == requested["dataset_id"]
                and existing["manifest_sha256"] == requested["manifest_sha256"]
                and existing["payload_sha256"] == requested["payload_sha256"]
            ):
                return {**existing, "status": "reused"}
            raise FileExistsError(_reference_conflict_message(task, reference, existing, requested)) from error
    return {**requested, "status": "published"}


def _available_reference_revisions(name_root: Path) -> tuple[int, ...]:
    """Return deterministic available revisions for one safe logical-name directory."""
    if not name_root.exists():
        return ()
    if not name_root.is_dir() or name_root.is_symlink():
        raise ValueError(f"Dataset reference name root is unsafe: {name_root}.")
    revisions: list[int] = []
    for candidate in name_root.iterdir():
        match = re.fullmatch(r"r([0-9]+)\.json", candidate.name)
        if match is None:
            continue
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"Dataset reference namespace contains an unsafe revision entry: {candidate}.")
        revisions.append(int(match.group(1)))
    return tuple(sorted(revisions))


def resolve_dataset_reference(
    task: str,
    name: str,
    revision: int,
    *,
    storage_root: Path | str | None = None,
    validate_payload_hash: bool = False,
) -> dict[str, Any]:
    """Resolve one reference after validating its current immutable manifest evidence."""
    task = _validate_task(task)
    reference = DatasetRef(name=name, revision=revision)
    if not isinstance(validate_payload_hash, bool):
        raise TypeError("validate_payload_hash must be a bool.")
    path = common.paths.resolve_dataset_reference_path(task, reference.name, reference.revision, storage_root=storage_root)
    if not path.exists() and not path.is_symlink():
        available = _available_reference_revisions(path.parent)
        raise FileNotFoundError(
            f"Dataset reference is missing: task={task!r}, name={reference.name!r}, revision={reference.revision}. "
            f"Available revisions: {list(available)}."
        )
    record = _read_record(path)
    if record["task"] != task or record["name"] != reference.name or record["revision"] != reference.revision:
        raise ValueError(f"Dataset reference record identity does not match its path: {path}.")
    manifest = (
        package_manifest.load_package_manifest(record["dataset_id"], storage_root=storage_root)
        if validate_payload_hash
        else package_manifest.load_package_manifest_evidence(record["dataset_id"], storage_root=storage_root)
    )
    manifest_path = _manifest_path(record["dataset_id"], storage_root=storage_root)
    current_manifest_sha256 = common.serialization.file_sha256(manifest_path)
    expected = {
        "dataset_id": manifest["dataset_id"],
        "dataset_digest": manifest["dataset_digest"],
        "manifest_sha256": current_manifest_sha256,
        "payload_sha256": manifest["payload_sha256"],
        "dataset_view": manifest["dataset_view"],
        "evaluation_regime": manifest["evaluation_regime"],
        "materials": list(manifest["materials"]),
    }
    view = views.get_view(str(manifest["dataset_view"]))
    view_task = view.id if view.registered_task_id is None else view.registered_task_id
    if manifest["registered_task_id"] != view.registered_task_id or view_task != task or any(record[key] != value for key, value in expected.items()):
        raise ValueError(f"Dataset reference {task}/{reference.name}/r{reference.revision} no longer matches immutable package evidence.")
    counts = {
        "sample_count": int(manifest["sample_count"]),
        "source_case_count": int(manifest["source_case_count"]),
        "transition_count": int(manifest["transition_count"]),
    }
    return {**record, **counts, "status": "resolved", "manifest_status": "valid", "display_name": reference.display_name}


def resolve_dataset_reference_record(
    task: str,
    name: str,
    revision: int,
    *,
    storage_root: Path | str | None = None,
    validate_payload_hash: bool = False,
) -> dict[str, Any]:
    """Resolve one reference and return only its self-contained immutable record."""
    resolved = resolve_dataset_reference(
        task,
        name,
        revision,
        storage_root=storage_root,
        validate_payload_hash=validate_payload_hash,
    )
    return validate_dataset_reference_record({key: resolved[key] for key in _REFERENCE_RECORD_KEYS})


def list_dataset_references(task: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], ...]:
    """List valid task-local reference records in deterministic logical order."""
    task = _validate_task(task)
    root = common.paths.get_dataset_references_root(storage_root=storage_root) / task
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Dataset reference task root is unsafe: {root}.")
    records: list[dict[str, Any]] = []
    for name_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not name_dir.is_dir() or name_dir.is_symlink():
            raise ValueError(f"Dataset reference namespace contains an unsafe entry: {name_dir}.")
        name = _validate_reference_name(name_dir.name)
        for path in sorted(name_dir.glob("r*.json"), key=lambda candidate: candidate.name):
            record = _read_record(path)
            if record["task"] != task or record["name"] != name or path.name != f"r{record['revision']}.json":
                raise ValueError(f"Dataset reference record identity does not match its path: {path}.")
            records.append(record)
    return tuple(sorted(records, key=lambda item: (item["name"], item["revision"])))


def inspect_dataset_reference(task: str, name: str, revision: int, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Resolve one reference and return its concise inspection summary."""
    resolved = resolve_dataset_reference(task, name, revision, storage_root=storage_root)
    return {
        key: resolved[key]
        for key in (
            "task",
            "name",
            "revision",
            "display_name",
            "dataset_id",
            "dataset_digest",
            "dataset_view",
            "evaluation_regime",
            "status",
            "manifest_status",
            "sample_count",
            "source_case_count",
            "transition_count",
        )
    }
