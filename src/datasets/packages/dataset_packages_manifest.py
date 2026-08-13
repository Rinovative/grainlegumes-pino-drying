"""
===============================================================================
dataset_packages_manifest.py
===============================================================================
Admit immutable Dataset package manifests and their exact payloads.
Responsibilities:
  - Define the current package-manifest schema and portable provenance keys
  - Validate manifest identity, payload hash, and steady tensor binding
  - Resolve package metadata and payload paths through canonical path owners
Design principles:
  - Admission is independent of package planning, publication, and runtime loading
  - Package identity is recomputed through the Dataset identity owner
  - Unsafe, missing, malformed, or mismatched artifacts fail closed
This module does NOT:
  - Build packages, select source cases, construct Datasets, or create DataLoaders
  - Support alternate schemas, aliases, fallback paths, or partial admission
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from src import common
from src.datasets.contracts import dataset_contracts_identity as identity
from src.generation.cases import generation_cases_case as case_contract
from src.generation.cases import generation_cases_config as config_contract

from . import dataset_packages_trajectory as trajectory

DATASET_PACKAGE_SCHEMA_KIND: Final = "vp2_dataset_package_manifest"
DATASET_PACKAGE_SCHEMA_VERSION: Final = 1
PACKAGE_PROVENANCE_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "dataset_view",
        "registered_task_id",
        "evaluation_regime",
        "materials",
        "channel_contract",
        "channel_contract_digest",
        "source_simulation_profiles",
        "source_batches",
        "source_batch_ids",
        "source_template_digests",
        "source_git_commits",
        "source_case_identities",
        "included_source_cases",
        "excluded_source_cases",
        "source_selection_decisions",
        "matched_case_input_ids",
        "airflow_provenance",
        "steady_flow_conditioning",
        "material_file_identities",
        "operation_config_digests",
        "campaign_name",
        "campaign_id",
        "campaign_digest",
        "campaign_purpose",
        "material_roles",
        "evaluation_regimes",
        "material_memberships",
        "source_role",
        "training_eligible",
        "duplicate_case_input_policy",
        "case_membership",
        "split_membership",
        "membership_counts",
        "available_ood_groups",
        "ood_group_indexes",
        "ood_parameter_indexes",
        "task_relevant_ood_parameters",
        "material_counts",
        "source_profile_counts",
        "candidate_source_case_count",
        "builder_identity",
        "schema_identity",
    }
)
PACKAGE_MANIFEST_KEYS: Final = PACKAGE_PROVENANCE_KEYS | {
    "dataset_id",
    "dataset_digest",
    "payload_filename",
    "sample_count",
    "source_case_count",
    "transition_count",
    "payload_sha256",
}


def _validate_schema_identity(value: Any, *, dataset_view: Any) -> None:
    """Require the exact active package, case, HDF5, and view schema identity."""
    if dataset_view == "transient_drying":
        payload_schema_version = trajectory.TRANSIENT_INDEX_SCHEMA_VERSION
    elif dataset_view == "steady_flow":
        payload_schema_version = identity.TRAINING_DATASET_SCHEMA_VERSION
    else:
        message = f"Dataset package declares unsupported view {dataset_view!r}."
        raise ValueError(message)
    expected = {
        "package": DATASET_PACKAGE_SCHEMA_VERSION,
        "case_hdf5": config_contract.CANONICAL_HDF5_SCHEMA_VERSION,
        "generation_case": case_contract.CASE_CONTRACT_DIGEST,
        "transient_index": payload_schema_version,
    }
    if value != expected:
        message = "Dataset package does not carry the current exact generation case-contract identity."
        raise ValueError(message)


def validate_manifest_content(
    manifest: Any,
    *,
    dataset_id: str,
    payload_path: Path,
) -> dict[str, Any]:
    """Validate one current manifest against its portable identity and payload."""
    if not isinstance(manifest, dict) or set(manifest) != PACKAGE_MANIFEST_KEYS:
        message = f"Dataset package manifest keys do not match the current schema for {dataset_id!r}."
        raise ValueError(message)
    _validate_schema_identity(
        manifest["schema_identity"],
        dataset_view=manifest["dataset_view"],
    )
    provenance = {key: manifest[key] for key in PACKAGE_PROVENANCE_KEYS}
    expected_id, expected_digest = identity.package_identity_from_provenance(provenance)
    if (
        manifest["schema_kind"] != DATASET_PACKAGE_SCHEMA_KIND
        or manifest["schema_version"] != DATASET_PACKAGE_SCHEMA_VERSION
        or manifest["dataset_id"] != dataset_id
        or manifest["dataset_id"] != expected_id
        or manifest["dataset_digest"] != expected_digest
        or manifest["payload_filename"] != payload_path.name
        or manifest["payload_sha256"] != common.serialization.file_sha256(payload_path)
        or manifest["sample_count"] < 1
        or manifest["source_case_count"] < 1
    ):
        message = f"Dataset package manifest does not bind the exact payload identity for {dataset_id!r}."
        raise ValueError(message)
    return dict(manifest)


def load_package_manifest(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one package manifest and its exact payload hash."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    metadata_path = common.paths.get_dataset_metadata_root(storage_root=storage_root) / logical_id / "dataset_manifest.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        message = f"Dataset package manifest is missing or unsafe: {metadata_path}."
        raise FileNotFoundError(message)
    try:
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset package manifest is unreadable: {metadata_path}."
        raise ValueError(message) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("payload_filename"), str):
        message = f"Dataset package manifest is malformed: {metadata_path}."
        raise TypeError(message)
    payload_path = common.paths.get_dataset_payload_root(storage_root=storage_root) / logical_id / manifest["payload_filename"]
    if not payload_path.is_file() or payload_path.is_symlink():
        message = f"Dataset package payload is missing or unsafe: {payload_path}."
        raise FileNotFoundError(message)
    return validate_manifest_content(manifest, dataset_id=logical_id, payload_path=payload_path)


def load_steady_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Bind a steady package manifest to its validated tensor payload identity."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    payload_path = Path(dataset_path)
    manifest_path = Path(metadata_root) / logical_id / "dataset_manifest.json"
    if not payload_path.is_file() or payload_path.is_symlink():
        message = f"Dataset package payload is missing or unsafe: {payload_path}."
        raise FileNotFoundError(message)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        message = f"Dataset package manifest is missing or unsafe: {manifest_path}."
        raise FileNotFoundError(message)
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset package manifest is unreadable: {manifest_path}."
        raise ValueError(message) from error
    manifest = validate_manifest_content(raw_manifest, dataset_id=logical_id, payload_path=payload_path)
    provenance = {key: manifest[key] for key in PACKAGE_PROVENANCE_KEYS}
    if (
        manifest["dataset_view"] != "steady_flow"
        or manifest["registered_task_id"] != dataset_identity.task
        or manifest["sample_count"] != dataset_identity.sample_count
        or dataset_identity.dataset_id != logical_id
        or dataset_identity.source_provenance != provenance
    ):
        message = f"Steady package manifest does not bind its validated tensor identity: {manifest_path}."
        raise ValueError(message)
    return manifest
