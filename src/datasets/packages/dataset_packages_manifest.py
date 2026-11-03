"""Validate immutable Dataset selection and fixed-membership manifests.

Responsibilities:
  - Admit the exact current manifest schema and ordered Source evidence
  - Bind each contributing Source's replacement policy and candidate outcomes to the selected cases
  - Bind Dataset semantic identity and optional payload-byte hashes
  - Load full or metadata-only package evidence from the task-qualified Dataset directory
  - Verify steady payload bindings
  - Project declared OOD Dataset membership for downstream evaluation planning

Design principles:
  - Dataset-view contracts own channel and task interpretation
  - The task directory and the fixed payload name must agree with the manifest's own view
  - Evidence-only reads explicitly omit rehashing previously admitted payloads
  - Unsafe paths and unsupported historical manifest schemas fail closed
  - Declared assignment projection does not admit prediction artifacts

This module does NOT:
  - Build package payloads, publish manifests or migrate historical schemas
  - Choose Dataset roles or materialize runtime Dataset samples
  - Establish saved-output existence or evaluation-artifact admissibility
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from src import common
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_views as views

DATASET_PACKAGE_SCHEMA_KIND: Final = "dataset_selection_manifest"
DATASET_PACKAGE_SCHEMA_VERSION: Final = 3
_STEADY_TASK: Final = "steady_flow"
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
PACKAGE_PROVENANCE_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "dataset_view",
        "registered_task_id",
        "source_task",
        "materials",
        "channel_contract",
        "channel_contract_digest",
        "selection",
        "case_partition",
        "plan_invocation",
        "grid",
        "source_case_identities",
        "source_outcomes",
        "builder_identity",
        "producer_evidence",
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


def _positive_int(value: Any, *, label: str, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        msg = f"Dataset manifest {label} must be a {'non-negative' if zero else 'positive'} integer."
        raise ValueError(msg)
    return value


_OUTCOME_KEYS: Final = frozenset({"source_name", "campaign_digest", "replacement_policy", "materials", "candidates"})


def _validate_source_outcomes(manifest: Mapping[str, Any]) -> None:
    """Bind each contributing Source's recorded outcomes to the selected cases and to their own counts."""
    outcomes = manifest["source_outcomes"]
    members = {
        (entry["source"]["source_name"], entry["source"]["material_family"], entry["source"]["case_id"]): entry
        for entry in manifest["source_case_identities"]
    }
    source_names = list(dict.fromkeys(name for name, _material, _case in members))
    if (
        not isinstance(outcomes, list)
        or any(not isinstance(item, dict) or set(item) != _OUTCOME_KEYS for item in outcomes)
        or [item["source_name"] for item in outcomes] != source_names
    ):
        msg = "Dataset source_outcomes must describe exactly the contributing Sources, in selection order."
        raise ValueError(msg)
    successful = {
        (item["source_name"], row["material_family"], row["case_id"])
        for item in outcomes
        for row in item["candidates"]
        if row["outcome"] == "successful"
    }
    if not set(members) <= successful:
        msg = "Dataset source_outcomes do not record every selected case as a successful candidate."
        raise ValueError(msg)
    for item in outcomes:
        for material, counts in item["materials"].items():
            rows = [row for row in item["candidates"] if row["material_family"] == material]
            primary = [row for row in rows if row["origin"] == "primary"]
            replacement = [row for row in rows if row["origin"] == "replacement"]
            if counts != {
                "original_candidates": len(primary),
                "original_failed": sum(row["outcome"] == "failed" for row in primary),
                "replacement_candidates": len(replacement),
                "replacement_failed": sum(row["outcome"] == "failed" for row in replacement),
                "successful": sum(row["outcome"] == "successful" for row in rows),
                "pending": sum(row["outcome"] == "pending" for row in rows),
            }:
                msg = f"Dataset source_outcomes counts of {material!r} disagree with their candidate table."
                raise ValueError(msg)


def validate_manifest_content(
    manifest: Any,
    *,
    task: str,
    dataset_id: str,
    payload_path: Path,
    validate_payload_hash: bool = True,
) -> dict[str, Any]:
    """Admit exact current schema and bind ordered source evidence to payload bytes.

    ``task`` is the task directory the manifest was read from; it must be the manifest's own view.
    """
    provenance_keys = PACKAGE_PROVENANCE_KEYS
    if not isinstance(manifest, dict) or set(manifest) != PACKAGE_MANIFEST_KEYS:
        msg = f"Dataset manifest keys do not match current contract for {dataset_id!r}."
        raise ValueError(msg)
    if manifest["schema_kind"] != DATASET_PACKAGE_SCHEMA_KIND or manifest["schema_version"] != DATASET_PACKAGE_SCHEMA_VERSION:
        msg = "Historical Dataset manifest contract is unsupported."
        raise ValueError(msg)
    invocation = manifest["plan_invocation"]
    if invocation is not None and (
        not isinstance(invocation, list) or not invocation or any(not isinstance(item, str) or not item for item in invocation)
    ):
        msg = "Dataset plan_invocation must retain an exact non-empty argument vector or explicit null."
        raise ValueError(msg)
    view = views.get_view(manifest["dataset_view"])
    if manifest["registered_task_id"] != view.registered_task_id or manifest["source_task"] != view.id:
        msg = "Dataset view and source task identity disagree."
        raise ValueError(msg)
    if manifest["dataset_view"] != task:
        msg = f"Dataset {dataset_id!r} belongs to task {manifest['dataset_view']!r} but was read as task {task!r}."
        raise ValueError(msg)
    if manifest["channel_contract_digest"] != view.contract_digest:
        msg = "Dataset channel contract digest disagrees with its view."
        raise ValueError(msg)
    from . import dataset_packages_reproduction as reproduction  # noqa: PLC0415 -- schema-bound producer evidence

    reproduction.validate_producer_evidence(manifest["producer_evidence"])
    if manifest["builder_identity"].get("implementation_sha256") != reproduction.conversion_implementation_sha256(manifest["dataset_view"]):
        message = "Dataset conversion/interpreter implementation changed; rebuild from its exact Source selection before consumption."
        raise ValueError(message)
    _positive_int(manifest["sample_count"], label="sample_count")
    _positive_int(manifest["source_case_count"], label="source_case_count")
    _positive_int(manifest["transition_count"], label="transition_count", zero=True)
    if not isinstance(manifest["source_case_identities"], list) or manifest["source_case_count"] != len(manifest["source_case_identities"]):
        msg = "Dataset source count disagrees with ordered identities."
        raise ValueError(msg)
    expected_materials = list(dict.fromkeys(entry["source"]["material_family"] for entry in manifest["source_case_identities"]))
    if manifest["materials"] != expected_materials:
        msg = "Dataset materials disagree with ordered source membership."
        raise ValueError(msg)
    _validate_source_outcomes(manifest)
    if manifest["dataset_view"] == "steady_flow" and (manifest["sample_count"] != manifest["source_case_count"] or manifest["transition_count"] != 0):
        msg = "Steady Dataset case and sample counts disagree."
        raise ValueError(msg)
    if manifest["dataset_view"] == "transient_drying" and manifest["transition_count"] != manifest["sample_count"]:
        msg = "Transient Dataset transition and sample counts disagree."
        raise ValueError(msg)
    if not isinstance(manifest["materials"], list) or not manifest["materials"] or len(manifest["materials"]) != len(set(manifest["materials"])):
        msg = "Dataset materials must be a non-empty unique list."
        raise ValueError(msg)
    provenance = {key: manifest[key] for key in provenance_keys}
    expected_id, expected_digest = identity.package_identity_from_provenance(provenance)
    if (
        manifest["dataset_id"] != dataset_id
        or dataset_id != expected_id
        or manifest["dataset_digest"] != expected_digest
        or manifest["payload_filename"] != common.paths.get_dataset_payload_filename(task)
        or payload_path.name != manifest["payload_filename"]
        or not isinstance(manifest["payload_sha256"], str)
        or _SHA256_PATTERN.fullmatch(manifest["payload_sha256"]) is None
        or not isinstance(validate_payload_hash, bool)
    ):
        msg = f"Dataset manifest does not bind exact identity for {dataset_id!r}."
        raise ValueError(msg)
    if validate_payload_hash and common.serialization.file_sha256(payload_path) != manifest["payload_sha256"]:
        msg = f"Dataset payload bytes disagree with manifest for {dataset_id!r}."
        raise ValueError(msg)
    return dict(manifest)


def _load(task: str, dataset_id: str, *, storage_root: Path | str | None, validate_payload_hash: bool) -> dict[str, Any]:
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    datasets_root = common.paths.get_datasets_root(storage_root=storage_root)
    path = common.paths.resolve_dataset_manifest_path(task, logical_id, dataset_root=datasets_root)
    if not path.is_file() or path.is_symlink():
        msg = f"Dataset manifest is missing or unsafe: {path}."
        raise FileNotFoundError(msg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Dataset manifest is unreadable: {path}."
        raise ValueError(msg) from error
    payload = common.paths.resolve_dataset_path(task, logical_id, dataset_root=datasets_root)
    if not payload.is_file() or payload.is_symlink():
        msg = f"Dataset payload is missing or unsafe: {payload}."
        raise FileNotFoundError(msg)
    return validate_manifest_content(raw, task=task, dataset_id=logical_id, payload_path=payload, validate_payload_hash=validate_payload_hash)


def load_package_manifest(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read current Dataset manifest and verify physical payload bytes."""
    return _load(task, dataset_id, storage_root=storage_root, validate_payload_hash=True)


def load_package_manifest_evidence(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read current Dataset manifest without rehashing an already admitted payload."""
    return _load(task, dataset_id, storage_root=storage_root, validate_payload_hash=False)


def declared_evaluation_assignments(
    task: str,
    train_dataset_id: str,
    additional_dataset_ids: Sequence[str] = (),
    *,
    storage_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Project declared OOD Dataset membership, without asserting artifact availability or admissibility.

    The result describes only material-holdout membership declared in fixed Dataset
    partitions. It does not establish that saved predictions or any other outputs
    exist, nor that such artifacts are admissible for evaluation.
    """
    if isinstance(additional_dataset_ids, (str, bytes)) or not isinstance(additional_dataset_ids, Sequence):
        msg = "Additional Dataset IDs must be provided as an ordered sequence."
        raise TypeError(msg)
    dataset_ids = [train_dataset_id, *additional_dataset_ids]
    if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in dataset_ids):
        msg = "Dataset IDs must be non-empty strings."
        raise ValueError(msg)
    if len(dataset_ids) != len(set(dataset_ids)):
        msg = "Training and additional Dataset IDs must be unique."
        raise ValueError(msg)

    primary_manifest = load_package_manifest_evidence(task, train_dataset_id, storage_root=storage_root)
    primary_partition = primary_manifest["case_partition"]
    if primary_partition is None:
        msg = f"Training Dataset {train_dataset_id!r} must declare a fixed case partition."
        raise ValueError(msg)

    declared_ids: list[str] = []
    for dataset_id in additional_dataset_ids:
        manifest = load_package_manifest_evidence(task, dataset_id, storage_root=storage_root)
        fixed_partition = manifest["case_partition"]
        if fixed_partition is None or not fixed_partition["ood"]:
            msg = f"Additional Dataset {dataset_id!r} must declare a non-empty fixed OOD partition."
            raise ValueError(msg)
        declared_ids.append(dataset_id)
    if primary_partition["ood"]:
        declared_ids.append(train_dataset_id)
    return [
        {"dataset_id": dataset_id, "role": "ood", "relation": "Dataset-declared material holdout", "assumptions": []} for dataset_id in declared_ids
    ]


def load_steady_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Bind an admitted steady payload to its exact package manifest."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    path = common.paths.resolve_dataset_manifest_path(_STEADY_TASK, logical_id, dataset_root=metadata_root)
    payload = Path(dataset_path)
    if not path.is_file() or path.is_symlink() or not payload.is_file() or payload.is_symlink():
        msg = "Steady Dataset manifest or payload is missing or unsafe."
        raise FileNotFoundError(msg)
    manifest = validate_manifest_content(json.loads(path.read_text(encoding="utf-8")), task=_STEADY_TASK, dataset_id=logical_id, payload_path=payload)
    provenance_keys = PACKAGE_PROVENANCE_KEYS
    provenance = {key: manifest[key] for key in provenance_keys}
    if (
        manifest["dataset_view"] != "steady_flow"
        or manifest["registered_task_id"] != dataset_identity.task
        or manifest["sample_count"] != dataset_identity.sample_count
        or dataset_identity.dataset_id != logical_id
        or dataset_identity.source_provenance != provenance
    ):
        msg = "Steady Dataset payload identity disagrees with its manifest."
        raise ValueError(msg)
    return manifest
