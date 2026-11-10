"""Validate immutable Dataset selection and fixed-membership manifests.

Responsibilities:
  - Admit the exact current manifest schema and ordered Source evidence
  - Bind each contributing Source's replacement policy and candidate outcomes to the selected cases
  - Bind Dataset semantic identity and optional payload-byte hashes
  - Read full or metadata-only package evidence through one reader of the task-qualified Dataset directory
  - Bind the identity a loaded payload states about itself (steady tensor payload, transient index) to its manifest
  - Project declared OOD Dataset membership for downstream evaluation planning

Design principles:
  - Dataset-view contracts own channel and task interpretation
  - One manifest validator and one payload-identity binding serve both Dataset views; only the steady payload embeds provenance
  - The task directory and the fixed payload name must agree with the manifest's own view
  - Evidence-only reads explicitly omit rehashing previously admitted payloads
  - Unsafe paths and manifests that differ from the current schema fail closed
  - A recorded conversion digest is evidence: it must be well formed but is never compared with the running conversion code
  - Declared assignment projection does not admit prediction artifacts

This module does NOT:
  - Build package payloads or publish manifests
  - Choose Dataset roles or materialize runtime Dataset samples
  - Establish saved-output existence or evaluation-artifact admissibility
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.contracts.dataset_contracts_identity import DATASET_PACKAGE_SCHEMA_KIND

if TYPE_CHECKING:
    from pathlib import Path

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
PACKAGE_PROVENANCE_KEYS: Final = frozenset(
    {
        "schema_kind",
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
    "derived_payload_sha256",
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
    selection = manifest["selection"]
    # Case references carry only the design name, so a revision label is bound here through each case's recorded Source.
    if isinstance(selection, Mapping) and selection.get("kind") == "design_order" and selection.get("source_names") != source_names:
        msg = "Design-order Dataset selection must name exactly the Source its cases come from."
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
    if manifest["schema_kind"] != DATASET_PACKAGE_SCHEMA_KIND:
        msg = "Dataset manifest schema_kind does not match the current contract."
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
    from . import dataset_packages_selection as selection  # noqa: PLC0415 -- selection imports the Generation catalog

    selection.validate_selection_request(reproduction.manifest_selection_request(manifest))
    derived = manifest["derived_payload_sha256"]
    if (derived is None) == (task == "transient_drying") or (derived is not None and _SHA256_PATTERN.fullmatch(str(derived)) is None):
        msg = f"Dataset manifest for {dataset_id!r} must bind the transient PT shard receipt digest exactly for the transient view."
        raise ValueError(msg)
    if validate_payload_hash:
        from . import dataset_packages_transient_shards as shards  # noqa: PLC0415 -- the shard module imports this validator

        # A missing payload or receipt raises FileNotFoundError naming its path; only differing bytes are refused here.
        receipt = payload_path.parent / shards.TRANSIENT_PT_SHARD_DIRECTORY / shards.TRANSIENT_PT_RECEIPT_FILENAME
        if common.serialization.file_sha256(payload_path) != manifest["payload_sha256"] or (
            derived is not None and common.serialization.file_sha256(receipt) != derived
        ):
            msg = f"Dataset payload bytes disagree with manifest for {dataset_id!r}."
            raise ValueError(msg)
    return dict(manifest)


def read_dataset_manifest(
    directory: Path,
    *,
    task: str,
    dataset_id: str,
    payload_path: Path | None = None,
    validate_payload_hash: bool = True,
) -> dict[str, Any]:
    """Admit the manifest of one Dataset folder against the payload that the same folder owns.

    ``payload_path``, when given, is the payload a caller already loaded; it must be the folder's own payload file.
    """
    path = directory / common.paths.DATASET_MANIFEST_FILENAME
    payload = directory / common.paths.get_dataset_payload_filename(task)
    if payload_path is not None and (payload_path.parent.resolve() != directory.resolve() or payload_path.name != payload.name):
        msg = "Dataset folder must own its current selection manifest and payload."
        raise ValueError(msg)
    if not directory.is_dir() or directory.is_symlink() or not path.is_file() or path.is_symlink():
        msg = f"Dataset manifest is missing or unsafe: {path}."
        raise FileNotFoundError(msg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Dataset manifest is unreadable: {path}."
        raise ValueError(msg) from error
    if not payload.is_file() or payload.is_symlink():
        msg = f"Dataset payload is missing or unsafe: {payload}."
        raise FileNotFoundError(msg)
    return validate_manifest_content(raw, task=task, dataset_id=dataset_id, payload_path=payload, validate_payload_hash=validate_payload_hash)


def _validate_payload_identity(
    manifest: Mapping[str, Any],
    *,
    dataset_id: str,
    task: str,
    contract_digest: str,
    sample_count: int,
    case_ids: Sequence[str],
) -> None:
    """Require the identity a loaded payload states about itself to equal its admitted manifest, for every Dataset view.

    The steady tensor payload and the transient trajectory index both state their Dataset ID, task, channel contract digest,
    sample count and ordered package case IDs.
    """
    if (
        manifest["dataset_id"] != dataset_id
        or manifest["registered_task_id"] != task
        or manifest["channel_contract_digest"] != contract_digest
        or manifest["sample_count"] != sample_count
        or [entry["package_case_id"] for entry in manifest["source_case_identities"]] != list(case_ids)
    ):
        msg = f"Dataset {manifest['dataset_id']!r} payload identity disagrees with its manifest."
        raise ValueError(msg)


def validate_steady_payload_identity(manifest: Mapping[str, Any], dataset_identity: identity.DatasetIdentity) -> None:
    """Bind a verified steady tensor payload, which also embeds its selection provenance, to its admitted manifest."""
    _validate_payload_identity(
        manifest,
        dataset_id=dataset_identity.dataset_id,
        task=dataset_identity.task,
        contract_digest=dataset_identity.data_contract_digest,
        sample_count=dataset_identity.sample_count,
        case_ids=dataset_identity.sample_ids,
    )
    selection = dataset_identity.source_selection_identity
    if (
        dataset_identity.source_provenance != {key: manifest[key] for key in PACKAGE_PROVENANCE_KEYS}
        or not isinstance(selection, dict)
        or selection.get("schema_kind") != identity.SOURCE_SELECTION_SCHEMA_KIND
        or selection.get("source_cases") != manifest["source_case_identities"]
    ):
        msg = "Steady Dataset payload provenance disagrees with its manifest."
        raise ValueError(msg)


def validate_transient_payload_identity(manifest: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    """Bind an admitted transient trajectory index, and through it the PT shards built from it, to its manifest."""
    _validate_payload_identity(
        manifest,
        dataset_id=index["dataset_id"],
        task=index["dataset_view"],
        contract_digest=index["contract_digest"],
        sample_count=index["sample_count"],
        case_ids=[case["package_case_id"] for case in index["cases"]],
    )


def _load(task: str, dataset_id: str, *, storage_root: Path | str | None, validate_payload_hash: bool) -> dict[str, Any]:
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    directory = common.paths.resolve_dataset_dir(task, logical_id, dataset_root=common.paths.get_datasets_root(storage_root=storage_root))
    return read_dataset_manifest(directory, task=task, dataset_id=logical_id, validate_payload_hash=validate_payload_hash)


def load_package_manifest(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read current Dataset manifest and verify physical payload bytes."""
    return _load(task, dataset_id, storage_root=storage_root, validate_payload_hash=True)


def load_package_manifest_evidence(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read current Dataset manifest without rehashing an already admitted payload."""
    return _load(task, dataset_id, storage_root=storage_root, validate_payload_hash=False)


def declared_evaluation_assignments(
    task: str,
    train_dataset_id: str,
    material_ood_dataset_ids: Sequence[str] = (),
    *,
    storage_root: Path | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project declared OOD Dataset membership per OOD Evaluation role, without asserting artifact availability.

    ``material_ood`` lists every configured material-OOD Dataset in order, then the training Dataset when it declares
    material-OOD cases; ``parameter_ood`` lists the training Dataset when it declares parameter-OOD cases. The result
    describes only membership declared in fixed Dataset partitions. It does not establish that saved predictions or
    any other outputs exist, nor that such artifacts are admissible for evaluation.
    """
    if isinstance(material_ood_dataset_ids, (str, bytes)) or not isinstance(material_ood_dataset_ids, Sequence):
        msg = "Material-OOD Dataset IDs must be provided as an ordered sequence."
        raise TypeError(msg)
    dataset_ids = [train_dataset_id, *material_ood_dataset_ids]
    if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in dataset_ids):
        msg = "Dataset IDs must be non-empty strings."
        raise ValueError(msg)
    if len(dataset_ids) != len(set(dataset_ids)):
        msg = "Training and material-OOD Dataset IDs must be unique."
        raise ValueError(msg)

    primary_manifest = load_package_manifest_evidence(task, train_dataset_id, storage_root=storage_root)
    entries = primary_manifest["source_case_identities"]
    materials = {entry["package_case_id"]: entry["source"]["material_family"] for entry in entries}
    primary_partition = partition_contract.validate_training_partition(
        primary_manifest["case_partition"],
        dataset_id=train_dataset_id,
        case_ids=tuple(materials),
        material_by_case=materials,
        design_kind_by_case={entry["package_case_id"]: entry["design"]["design_kind"] for entry in entries},
    )

    material: list[str] = []
    for dataset_id in material_ood_dataset_ids:
        fixed_partition = load_package_manifest_evidence(task, dataset_id, storage_root=storage_root)["case_partition"]
        if fixed_partition is None or not partition_contract.evaluation_cases(fixed_partition, "material_ood"):
            msg = f"Material-OOD Dataset {dataset_id!r} must declare non-empty fixed OOD cases."
            raise ValueError(msg)
        material.append(dataset_id)
    declared = {
        "material_ood": [*material, *([train_dataset_id] if partition_contract.evaluation_cases(primary_partition, "material_ood") else [])],
        "parameter_ood": [train_dataset_id] if partition_contract.evaluation_cases(primary_partition, "parameter_ood") else [],
    }
    relation = {"material_ood": "Dataset-declared material holdout", "parameter_ood": "Dataset-declared parameter holdout"}
    return {
        role: [{"dataset_id": dataset_id, "role": role, "relation": relation[role], "assumptions": []} for dataset_id in dataset_ids]
        for role, dataset_ids in declared.items()
    }
