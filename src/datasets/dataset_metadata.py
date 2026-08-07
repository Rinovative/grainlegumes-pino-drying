"""
===============================================================================
dataset_metadata.py
===============================================================================
Validate and summarize current model-training metadata packages.

Responsibilities:
  - Validate one current terminal simulation-manifest snapshot
  - Bind profile, template, learning-view, and tensor identity to a dataset
  - Verify exact snapshot and optional final-dataset artifact integrity
  - Provide metadata-only summaries for planning and notebook previews

Design principles:
  - Metadata admits only the current Python simulation-batch schema
  - Source profile and template provenance remain explicit and fail-closed
  - Metadata-only summaries do not deserialize training tensors

This module does NOT:
  - Admit historical generated-batch, sampling, dataset, or timing schemas
  - Construct final datasets, splits, normalizers, checkpoints, or artifacts
  - Define a transient-drying tensor contract
===============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common, domain
from src.datasets.dataset_identity import (
    TRAINING_DATASET_SCHEMA_VERSION,
    DatasetIdentity,
    build_generated_batch_identity,
    validate_dataset_data_contract_digest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

METADATA_FILENAME = "dataset_metadata.json"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
METADATA_SCHEMA_KIND = "training_dataset_metadata"
METADATA_SCHEMA_VERSION = 1
BUILDER_MODULE = "src.datasets.dataset_build"
PUBLICATION_METHOD = "atomic_directory_rename"
SOURCE_MANIFEST_SCHEMA_KIND = "simulation_batch_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_SPATIAL_DIMENSIONS = 2
_MANIFEST_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "simulation_profile",
        "available_learning_views",
        "airflow_source",
        "batch_id",
        "batch_identity",
        "template",
        "export_contract_sha256",
        "intended_case_indices",
        "cases",
    }
)
_METADATA_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_id",
        "scientific_identity",
        "artifacts",
        "operational_provenance",
    }
)
_SCIENTIFIC_KEYS = frozenset(
    {
        "dataset_schema_version",
        "dataset_fingerprint",
        "task_id",
        "data_contract_digest",
        "source_batch_id",
        "source_simulation_profile",
        "source_template_sha256",
        "airflow_source",
        "generated_batch_identity_sha256",
        "sample_count",
        "spatial_shape",
        "tensors",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Validated metadata package bound to one final training dataset."""

    directory: Path
    metadata: dict[str, Any]
    source_manifest: dict[str, Any]
    timing: dict[str, Any] | None = None

    @property
    def source_manifest_sha256(self) -> str:
        """Return the exact validated source-manifest snapshot digest."""
        snapshots = self.metadata["artifacts"]["snapshots"]
        return str(snapshots[SOURCE_MANIFEST_FILENAME]["sha256"])

    @property
    def timing_summary(self) -> dict[str, Any]:
        """Return the current operational timing-availability summary."""
        return dict(self.metadata["operational_provenance"]["timing"])


@dataclass(frozen=True, slots=True)
class DatasetMetadataSummary:
    """Describe one validated metadata package without loading tensor content."""

    dataset_id: str
    dataset_path: Path
    metadata_directory: Path
    dataset_exists: bool
    task_id: str
    data_contract_digest: str
    fingerprint: str
    sample_ids: tuple[str, ...]
    sample_count: int
    spatial_shape: tuple[int, ...]
    generated_batch_identity_sha256: str
    artifact_size_bytes: int


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required regular JSON object."""
    if not path.is_file() or path.is_symlink():
        msg = f"Missing or unsafe {label}: {path}"
        raise FileNotFoundError(msg)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Could not load {label}: {path}"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"{label} must contain a JSON object: {path}"
        raise TypeError(msg)
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], *, label: str) -> None:
    """Require exactly one declared mapping surface."""
    missing = sorted(set(expected).difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        msg = f"{label} keys do not match: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)


def _require_sha256(value: Any, *, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        msg = f"{label} must be a lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    """Return one positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{label} must be a positive integer."
        raise ValueError(msg)
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    """Return one non-negative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{label} must be a non-negative integer."
        raise ValueError(msg)
    return value


def _require_spatial_shape(value: Any, *, label: str) -> tuple[int, int]:
    """Return the task-owned two-dimensional spatial shape."""
    if not isinstance(value, list) or len(value) != _SPATIAL_DIMENSIONS:
        msg = f"{label} must contain exactly {_SPATIAL_DIMENSIONS} dimensions."
        raise ValueError(msg)
    return tuple(_require_positive_int(item, label=f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _manifest_sample_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return ordered canonical identifiers from current case indices."""
    indices = manifest.get("intended_case_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in indices)
        or indices != sorted(set(indices))
    ):
        msg = "Source manifest intended_case_indices are malformed."
        raise ValueError(msg)
    return tuple(f"case_{value:04d}" for value in indices)


def _validate_source_manifest_snapshot(manifest: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any]]:
    """Validate one terminal current Python simulation-manifest snapshot."""
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="Source manifest snapshot")
    if (
        manifest["schema_kind"] != SOURCE_MANIFEST_SCHEMA_KIND
        or manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "complete"
    ):
        msg = "Source manifest snapshot is not one supported terminal simulation batch."
        raise ValueError(msg)
    if not isinstance(manifest["batch_id"], str) or not manifest["batch_id"]:
        msg = "Source manifest snapshot batch_id must be a non-empty string."
        raise ValueError(msg)
    sample_ids = _manifest_sample_ids(manifest)
    generated_identity = build_generated_batch_identity(manifest)
    if tuple(generated_identity["intended_case_ids"]) != sample_ids:
        msg = "Source manifest identity membership is inconsistent."
        raise ValueError(msg)
    return manifest, sample_ids, generated_identity


def _validate_file_artifact(value: Any, *, label: str) -> dict[str, Any]:
    """Validate one file identity declaration."""
    if not isinstance(value, dict):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(value, {"filename", "sha256", "size_bytes"}, label=label)
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        msg = f"{label}.filename must be one basename."
        raise ValueError(msg)
    _require_sha256(value["sha256"], label=f"{label}.sha256")
    _require_positive_int(value["size_bytes"], label=f"{label}.size_bytes")
    return value


def _validate_snapshot_entry(value: Any, *, label: str) -> dict[str, Any]:
    """Validate the required manifest-snapshot declaration."""
    if not isinstance(value, dict):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(value, {"sha256", "size_bytes", "required", "role"}, label=label)
    _require_sha256(value["sha256"], label=f"{label}.sha256")
    _require_positive_int(value["size_bytes"], label=f"{label}.size_bytes")
    if value["required"] is not True or value["role"] != "validated_generation_manifest":
        msg = f"{label} has invalid ownership metadata."
        raise ValueError(msg)
    return value


def _validate_tensor_contract(value: Any, *, identity: DatasetIdentity, task: domain.tasks.spec.TaskSpec) -> None:
    """Bind persisted tensor shape and dtype declarations to TaskSpec."""
    if not isinstance(value, dict) or set(value) != {"inputs", "outputs"}:
        msg = "Dataset metadata tensors must contain exactly inputs and outputs."
        raise ValueError(msg)
    expected = {
        "inputs": {"dtype": "float32", "shape": [identity.sample_count, task.in_channels, *identity.spatial_shape]},
        "outputs": {"dtype": "float32", "shape": [identity.sample_count, task.out_channels, *identity.spatial_shape]},
    }
    if value != expected:
        msg = "Dataset metadata tensor declarations do not match the dataset identity and TaskSpec."
        raise ValueError(msg)


def _validate_timing_summary(value: Any, *, sample_count: int) -> dict[str, Any]:
    """Validate the current explicit absence of an aggregated timing snapshot."""
    expected = {
        "status": "unavailable",
        "intended_case_count": sample_count,
        "measured_case_count": 0,
    }
    if value != expected:
        msg = "Dataset metadata timing summary does not match current unavailable-timing semantics."
        raise ValueError(msg)
    return expected


def validate_dataset_metadata_directory(
    directory: Path | str,
    *,
    dataset_identity: DatasetIdentity,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Validate one complete current metadata package and its optional dataset."""
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        msg = f"Dataset metadata directory is missing or unsafe: {root}"
        raise FileNotFoundError(msg)
    actual_files = {entry.name for entry in root.iterdir() if entry.is_file() and not entry.is_symlink()}
    if actual_files != {METADATA_FILENAME, SOURCE_MANIFEST_FILENAME} or any(entry.is_dir() or entry.is_symlink() for entry in root.iterdir()):
        msg = f"Dataset metadata package membership is invalid: {root}"
        raise ValueError(msg)
    metadata = _load_json(root / METADATA_FILENAME, label="dataset metadata")
    manifest, sample_ids, generated_identity = _validate_source_manifest_snapshot(
        _load_json(root / SOURCE_MANIFEST_FILENAME, label="source manifest snapshot")
    )
    _require_exact_keys(metadata, _METADATA_KEYS, label="Dataset metadata")
    if metadata["schema_kind"] != METADATA_SCHEMA_KIND or metadata["schema_version"] != METADATA_SCHEMA_VERSION:
        msg = "Unsupported dataset metadata schema."
        raise ValueError(msg)
    if metadata["dataset_id"] != dataset_identity.dataset_id:
        msg = "Dataset metadata dataset_id does not match the final dataset identity."
        raise ValueError(msg)
    scientific = metadata["scientific_identity"]
    if not isinstance(scientific, dict):
        msg = "Dataset metadata scientific_identity must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(scientific, _SCIENTIFIC_KEYS, label="Dataset metadata scientific_identity")
    task = domain.tasks.registry.get_task(dataset_identity.task)
    expected_scientific = {
        "dataset_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_identity.fingerprint,
        "task_id": task.id,
        "data_contract_digest": dataset_identity.data_contract_digest,
        "source_batch_id": manifest["batch_id"],
        "source_simulation_profile": manifest["simulation_profile"],
        "source_template_sha256": manifest["template"]["sha256"],
        "airflow_source": manifest["airflow_source"],
        "generated_batch_identity_sha256": generated_identity["batch_manifest_identity_sha256"],
        "sample_count": dataset_identity.sample_count,
        "spatial_shape": list(dataset_identity.spatial_shape),
        "tensors": scientific["tensors"],
    }
    if {key: scientific[key] for key in expected_scientific if key != "tensors"} != {
        key: expected_scientific[key] for key in expected_scientific if key != "tensors"
    }:
        msg = "Dataset metadata scientific identity disagrees with its dataset or source manifest."
        raise ValueError(msg)
    _validate_tensor_contract(scientific["tensors"], identity=dataset_identity, task=task)
    if sample_ids != dataset_identity.sample_ids:
        msg = "Source manifest membership does not match final dataset sample_ids."
        raise ValueError(msg)
    if dataset_identity.generated_batch_identity_sha256 != generated_identity["batch_manifest_identity_sha256"] or (
        dataset_identity.generated_batch_identity is not None and dataset_identity.generated_batch_identity != generated_identity
    ):
        msg = "Source manifest identity does not match the final dataset payload."
        raise ValueError(msg)
    artifacts = metadata["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"dataset", "snapshots"}:
        msg = "Dataset metadata artifacts must contain exactly dataset and snapshots."
        raise ValueError(msg)
    dataset_artifact = _validate_file_artifact(artifacts["dataset"], label="Dataset metadata dataset artifact")
    snapshots = artifacts["snapshots"]
    if not isinstance(snapshots, dict) or set(snapshots) != {SOURCE_MANIFEST_FILENAME}:
        msg = "Dataset metadata snapshots must contain exactly the current source manifest."
        raise ValueError(msg)
    snapshot = _validate_snapshot_entry(
        snapshots[SOURCE_MANIFEST_FILENAME],
        label="Dataset metadata source manifest snapshot",
    )
    manifest_path = root / SOURCE_MANIFEST_FILENAME
    if snapshot["size_bytes"] != manifest_path.stat().st_size or snapshot["sha256"] != common.serialization.file_sha256(manifest_path):
        msg = "Source manifest snapshot does not match its declared identity."
        raise ValueError(msg)
    operational = metadata["operational_provenance"]
    if not isinstance(operational, dict):
        msg = "Dataset metadata operational_provenance must be a mapping."
        raise TypeError(msg)
    _require_exact_keys(
        operational,
        {"builder_module", "publication_method", "source_manifest_sha256", "timing"},
        label="Dataset metadata operational_provenance",
    )
    if (
        operational["builder_module"] != BUILDER_MODULE
        or operational["publication_method"] != PUBLICATION_METHOD
        or operational["source_manifest_sha256"] != snapshot["sha256"]
    ):
        msg = "Dataset metadata operational provenance is inconsistent."
        raise ValueError(msg)
    _validate_timing_summary(operational["timing"], sample_count=dataset_identity.sample_count)
    if dataset_path is not None:
        resolved_dataset = Path(dataset_path)
        if not resolved_dataset.is_file() or resolved_dataset.is_symlink():
            msg = f"Final dataset artifact is missing or unsafe: {resolved_dataset}"
            raise FileNotFoundError(msg)
        if (
            resolved_dataset.name != dataset_artifact["filename"]
            or resolved_dataset.stat().st_size != dataset_artifact["size_bytes"]
            or common.serialization.file_sha256(resolved_dataset) != dataset_artifact["sha256"]
        ):
            msg = "Final dataset artifact does not match its metadata identity."
            raise ValueError(msg)
    return DatasetMetadata(root, metadata, manifest)


def load_dataset_metadata(
    dataset_id: str,
    *,
    dataset_identity: DatasetIdentity,
    metadata_root: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> DatasetMetadata:
    """Resolve and validate one current model-training metadata package."""
    directory = common.paths.resolve_dataset_metadata_dir(dataset_id, metadata_root=metadata_root)
    return validate_dataset_metadata_directory(
        directory,
        dataset_identity=dataset_identity,
        dataset_path=dataset_path,
    )


def load_dataset_metadata_summary(
    dataset_id: str,
    *,
    task: domain.tasks.spec.TaskSpec,
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
) -> DatasetMetadataSummary:
    """Validate and summarize one metadata package without loading tensors."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    directory = common.paths.resolve_dataset_metadata_dir(logical_id, metadata_root=metadata_root)
    dataset_path = common.paths.resolve_dataset_path(logical_id, dataset_root=dataset_root)
    metadata = _load_json(directory / METADATA_FILENAME, label="dataset metadata")
    _manifest, sample_ids, generated_identity = _validate_source_manifest_snapshot(
        _load_json(directory / SOURCE_MANIFEST_FILENAME, label="source manifest snapshot")
    )
    scientific = metadata.get("scientific_identity")
    if not isinstance(scientific, dict):
        msg = "Dataset metadata scientific_identity must be a mapping."
        raise TypeError(msg)
    if scientific.get("task_id") != task.id:
        msg = f"Dataset metadata for {logical_id!r} does not match TaskSpec {task.id!r}."
        raise ValueError(msg)
    data_contract_digest = validate_dataset_data_contract_digest(
        scientific.get("data_contract_digest"),
        task=task,
        label="Dataset metadata data_contract_digest",
    )
    sample_count = _require_positive_int(scientific.get("sample_count"), label="Dataset metadata sample_count")
    spatial_shape = _require_spatial_shape(scientific.get("spatial_shape"), label="Dataset metadata spatial_shape")
    identity = DatasetIdentity(
        dataset_id=logical_id,
        task=task.id,
        data_contract_digest=data_contract_digest,
        fingerprint=_require_sha256(scientific.get("dataset_fingerprint"), label="Dataset metadata dataset_fingerprint"),
        sample_ids=sample_ids,
        sample_count=sample_count,
        spatial_shape=spatial_shape,
        generated_batch_identity_sha256=str(generated_identity["batch_manifest_identity_sha256"]),
        generated_batch_identity=generated_identity,
    )
    package = validate_dataset_metadata_directory(directory, dataset_identity=identity)
    artifact = package.metadata["artifacts"]["dataset"]
    if dataset_path.exists() and (not dataset_path.is_file() or dataset_path.is_symlink()):
        msg = f"Final dataset artifact is not a regular file: {dataset_path}"
        raise FileNotFoundError(msg)
    exists = dataset_path.is_file() and not dataset_path.is_symlink()
    if exists and (dataset_path.name != artifact["filename"] or dataset_path.stat().st_size != artifact["size_bytes"]):
        msg = "Configured training dataset name or size does not match its metadata package."
        raise ValueError(msg)
    return DatasetMetadataSummary(
        dataset_id=identity.dataset_id,
        dataset_path=dataset_path,
        metadata_directory=directory,
        dataset_exists=exists,
        task_id=identity.task,
        data_contract_digest=identity.data_contract_digest,
        fingerprint=identity.fingerprint,
        sample_ids=identity.sample_ids,
        sample_count=identity.sample_count,
        spatial_shape=identity.spatial_shape,
        generated_batch_identity_sha256=str(generated_identity["batch_manifest_identity_sha256"]),
        artifact_size_bytes=int(artifact["size_bytes"]),
    )
