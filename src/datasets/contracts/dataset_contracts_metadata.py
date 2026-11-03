"""Read immutable Dataset selection manifests for steady-flow consumers.

Responsibilities:
  - Bind current steady manifests to admitted payload and ordered Source membership
  - Resolve Dataset metadata directories through canonical path owners
  - Return compact manifest-backed identity and count previews without tensor reads

Design principles:
  - Manifest identities remain distinct from tensor-content fingerprints
  - Verified payload identity must agree with task, source provenance and sample order

This module does NOT:
  - Deserialize training tensors or compute their content fingerprint
  - Publish Dataset packages or assign experiment roles
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common
from src.datasets.packages import dataset_packages_manifest as package_manifest

if TYPE_CHECKING:
    from src.datasets.contracts.dataset_contracts_identity import DatasetIdentity
    from src.domain.tasks.domain_task_spec import TaskSpec

_STEADY_TASK = "steady_flow"


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """One admitted current steady Dataset manifest and its immutable payload."""

    directory: Path
    manifest: dict[str, Any]
    dataset_path: Path


@dataclass(frozen=True, slots=True)
class DatasetMetadataSummary:
    """Small manifest-backed preview without deserializing training tensors."""

    dataset_id: str
    dataset_path: Path
    metadata_directory: Path
    dataset_exists: bool
    task_id: str
    data_contract_digest: str
    dataset_digest: str
    sample_count: int
    artifact_size_bytes: int


def _load_manifest(directory: Path, *, dataset_id: str, dataset_path: Path) -> dict[str, Any]:
    """Require the single current manifest and verify exact published payload bytes."""
    if not dataset_path.is_file() or dataset_path.is_symlink():
        message = f"Dataset payload is missing or unsafe: {dataset_path}"
        raise FileNotFoundError(message)
    if not directory.is_dir() or directory.is_symlink():
        message = f"Dataset metadata directory is missing or unsafe: {directory}"
        raise FileNotFoundError(message)
    manifest_path = directory / common.paths.DATASET_MANIFEST_FILENAME
    if dataset_path.parent.resolve() != directory.resolve() or not manifest_path.is_file() or manifest_path.is_symlink():
        message = "Dataset folder must own its current selection manifest and payload."
        raise ValueError(message)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset selection manifest is unreadable: {manifest_path}"
        raise ValueError(message) from error
    manifest = package_manifest.validate_manifest_content(raw, task=_STEADY_TASK, dataset_id=dataset_id, payload_path=dataset_path)
    if manifest["dataset_view"] != _STEADY_TASK:
        message = "Steady Dataset metadata requires the current steady_flow selection manifest."
        raise ValueError(message)
    return manifest


def validate_dataset_metadata_directory(
    directory: Path | str,
    *,
    dataset_identity: DatasetIdentity,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Bind a verified steady payload identity to its current selection manifest."""
    logical_id = common.paths.validate_logical_name(dataset_identity.dataset_id, label="dataset_id")
    path = Path(dataset_path) if dataset_path is not None else common.paths.resolve_dataset_path(_STEADY_TASK, logical_id)
    manifest = _load_manifest(Path(directory), dataset_id=logical_id, dataset_path=path)
    selection = dataset_identity.source_selection_identity
    if (
        dataset_identity.task != "steady_flow"
        or manifest["registered_task_id"] != dataset_identity.task
        or manifest["sample_count"] != dataset_identity.sample_count
        or dataset_identity.source_provenance != {key: manifest[key] for key in package_manifest.PACKAGE_PROVENANCE_KEYS}
        or not isinstance(selection, dict)
        or selection.get("schema_kind") != "dataset_source_selection"
        or selection.get("schema_version") != 1
        or selection.get("source_cases") != manifest["source_case_identities"]
        or tuple(item["package_case_id"] for item in manifest["source_case_identities"]) != dataset_identity.sample_ids
    ):
        message = "Steady Dataset payload identity disagrees with its current selection manifest."
        raise ValueError(message)
    return DatasetMetadata(Path(directory), manifest, path)


def load_dataset_metadata(
    dataset_id: str,
    *,
    dataset_identity: DatasetIdentity,
    metadata_root: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Resolve and validate one current steady Dataset manifest against its payload."""
    directory = common.paths.resolve_dataset_metadata_dir(_STEADY_TASK, dataset_id, metadata_root=metadata_root)
    if dataset_path is None:
        dataset_path = common.paths.resolve_dataset_path(_STEADY_TASK, dataset_id, dataset_root=metadata_root)
    return validate_dataset_metadata_directory(directory, dataset_identity=dataset_identity, dataset_path=dataset_path)


def load_dataset_metadata_summary(
    dataset_id: str,
    *,
    task: TaskSpec,
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
) -> DatasetMetadataSummary:
    """Read a current manifest digest and count without loading training tensors."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    root = Path(dataset_root).expanduser() if dataset_root is not None else common.paths.get_datasets_root()
    if metadata_root is not None and Path(metadata_root).expanduser().resolve() != root.resolve():
        message = "Dataset manifest and payload require the same Dataset root."
        raise ValueError(message)
    if task.id != _STEADY_TASK:
        message = f"Dataset selection for {logical_id!r} does not match TaskSpec {task.id!r}."
        raise ValueError(message)
    directory = common.paths.resolve_dataset_dir(task.id, logical_id, dataset_root=root)
    dataset_path = common.paths.resolve_dataset_path(task.id, logical_id, dataset_root=root)
    manifest = _load_manifest(directory, dataset_id=logical_id, dataset_path=dataset_path)
    if manifest["registered_task_id"] != task.id:
        message = f"Dataset selection for {logical_id!r} does not match TaskSpec {task.id!r}."
        raise ValueError(message)
    return DatasetMetadataSummary(
        dataset_id=logical_id,
        dataset_path=dataset_path,
        metadata_directory=directory,
        dataset_exists=True,
        task_id=task.id,
        data_contract_digest=task.data_contract_digest,
        dataset_digest=str(manifest["dataset_digest"]),
        sample_count=int(manifest["sample_count"]),
        artifact_size_bytes=dataset_path.stat().st_size,
    )
