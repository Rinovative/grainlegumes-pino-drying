"""
generation_workflow.py

Own post-simulation publication evidence, dataset gates, and source cleanup.
Responsibilities:
  - Validate exact GPU transfer inventories and every declared dataset package
  - Persist the staged all-workflow receipt used for idempotent continuation
  - Authorize and execute run-scoped CPU source cleanup after every local gate
  - Report generation, package, staging, run, and cleanup storage state
Design principles:
  - GPU generation sources and immutable learning views remain separate layers
  - Cleanup authorization binds immutable identities, hashes, paths, and bytes
  - Existing valid publications are validated and reused instead of recreated
This module does NOT:
  - Implement SSH, rsync, Slurm submission, COMSOL execution, or scientific logic
  - Delete GPU generation sources, dataset packages, repositories, or templates
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from src import common

from . import generation_campaign as campaign_runtime
from .cases import generation_cases_config as config_service
from .publication import generation_publication_campaign_evidence as campaign_evidence
from .runtime import generation_runtime_batch as batch_runtime
from .runtime import generation_runtime_workspace as workspace_service
from .validation import generation_validation_pilot as pilot_service

if TYPE_CHECKING:
    from collections.abc import Mapping

DATASET_RECEIPT_FILENAME: Final = "dataset_packages_complete.json"
INCOMPLETE_DATASET_RECEIPT_FILENAME: Final = "dataset_packages_incomplete.json"
PARTIAL_COMPLETION_RECEIPT_FILENAME: Final = "partial_completion.json"
ALL_WORKFLOW_RECEIPT_FILENAME: Final = "all_workflow.json"
CPU_CLEANUP_RECEIPT_FILENAME: Final = "cpu_source_cleanup.json"
DATASET_RECEIPT_SCHEMA_KIND: Final = "generation_dataset_packages_complete"
DATASET_EXTENSION_DIRECTORY_NAME: Final = "dataset_package_extensions"
DATASET_EXTENSION_SCHEMA_KIND: Final = "generation_dataset_package_extension"
PACKAGE_STATE_SCHEMA_KIND: Final = "generation_campaign_package_state"
ALL_WORKFLOW_SCHEMA_KIND: Final = "generation_all_workflow"
CPU_CLEANUP_SCHEMA_KIND: Final = "generation_cpu_source_cleanup"
CPU_CLEANUP_TRANSACTION_SCHEMA_KIND: Final = "generation_cpu_source_cleanup_transaction"
WORKFLOW_SCHEMA_VERSION: Final = 1
_SHA256_LENGTH: Final = 64
_SOURCE_CLEANUP_READY_CAMPAIGN_STATES: Final = frozenset({"successful", "transfer_complete"})
_CLEANUP_RECEIPT_IDENTITY_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "selected_batch_ids",
        "slurm_job_ids",
        "scheduler_job_name",
        "campaign_terminal_sha256",
        "authorization_sha256",
        "source_host",
        "source_storage_root",
        "destination_storage_root",
        "transfer_receipt_sha256",
        "dataset_receipt_sha256",
        "workflow_gate_sha256",
        "source_inventory_sha256",
        "destination_inventory_sha256",
        "source_directories",
        "source_file_count",
        "source_bytes_reclaimed",
    }
)


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required non-symlink JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is unreadable: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain one JSON object: {path}"
        raise TypeError(message)
    return value


def _tree_size(path: Path) -> int:
    """Return exact regular-file bytes below one symlink-free path."""
    if not path.exists():
        return 0
    if path.is_symlink():
        message = f"Storage size target is a symbolic link: {path}"
        raise ValueError(message)
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        message = f"Storage size target is not a regular file or directory: {path}"
        raise ValueError(message)
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            message = f"Storage tree contains a symbolic link: {item}"
            raise ValueError(message)
        if item.is_file():
            total += item.stat().st_size
    return total


def _transfer_receipt_path(run_id: str, *, storage_root: Path) -> Path:
    """Return the GPU transfer receipt path."""
    return campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / "transfer_complete.json"


def _dataset_receipt_path(run_id: str, *, storage_root: Path) -> Path:
    """Return the complete dataset-gate receipt path."""
    return campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / DATASET_RECEIPT_FILENAME


def _dataset_receipt_lock_path(run_id: str, *, storage_root: Path) -> Path:
    """Return the GPU-local dataset-finalization lock path."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    return common.paths.get_generation_state_root(storage_root=storage_root) / "dataset-package-locks" / f"{safe_id}.lock"


def _dataset_extension_lock_path(
    run_id: str,
    package_plan: Mapping[str, Any],
    *,
    storage_root: Path,
) -> Path:
    """Return the lock dedicated to one package-extension request."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    digest = common.serialization.canonical_json_sha256(dict(package_plan))
    return common.paths.get_generation_state_root(storage_root=storage_root) / "dataset-package-extension-locks" / f"{safe_id}-{digest}.lock"


def _dataset_extension_directory(run_id: str, *, storage_root: Path) -> Path:
    """Return the immutable per-package extension receipt directory."""
    return (
        campaign_evidence.campaign_run_directory(
            run_id,
            storage_root=storage_root,
        )
        / DATASET_EXTENSION_DIRECTORY_NAME
    )


def _dataset_extension_path(
    run_id: str,
    package_plan: Mapping[str, Any],
    *,
    storage_root: Path,
) -> Path:
    """Return one content-addressed package-extension receipt path."""
    digest = common.serialization.canonical_json_sha256(dict(package_plan))
    return (
        _dataset_extension_directory(
            run_id,
            storage_root=storage_root,
        )
        / f"{digest}.json"
    )


def _all_receipt_path(run_id: str, *, storage_root: Path) -> Path:
    """Return the all-workflow receipt path."""
    return campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / ALL_WORKFLOW_RECEIPT_FILENAME


def _cleanup_receipt_path(run_id: str, *, storage_root: Path) -> Path:
    """Return the compact CPU source-cleanup receipt path."""
    return campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / CPU_CLEANUP_RECEIPT_FILENAME


def _package_runtime_evidence(
    dataset_id: str,
    *,
    storage_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a validated manifest, inspection, and bounded loader smoke."""
    from src.datasets import packages as package_service  # noqa: PLC0415

    manifest = package_service.load_package_manifest(dataset_id, storage_root=storage_root)
    inspection = package_service.inspect_dataset_package(dataset_id, storage_root=storage_root)
    membership = "train" if manifest["evaluation_regime"] == "id" and manifest["training_eligible"] is True else None
    smoke = {
        f"workers_{num_workers}": package_service.smoke_dataset_package(
            dataset_id,
            storage_root=storage_root,
            membership=membership,
            num_workers=num_workers,
            hdf5_cache_size=1,
        )
        for num_workers in (0, 2)
    }
    return manifest, inspection, smoke


def _package_record(
    result: Mapping[str, Any],
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Return exact package hashes plus inspection and loader evidence."""
    dataset_id = common.paths.validate_logical_name(result.get("dataset_id"), label="dataset_id")
    manifest, inspection, smoke = _package_runtime_evidence(dataset_id, storage_root=storage_root)
    manifest_path = common.paths.get_dataset_metadata_root(storage_root=storage_root) / dataset_id / "dataset_manifest.json"
    payload_path = common.paths.get_dataset_packages_root(storage_root=storage_root) / dataset_id / str(manifest["payload_filename"])
    return {
        "dataset_name": manifest["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": manifest["dataset_view"],
        "evaluation_regime": manifest["evaluation_regime"],
        "build_status": result.get("status"),
        "manifest_relative_path": manifest_path.relative_to(storage_root).as_posix(),
        "manifest_sha256": common.serialization.file_sha256(manifest_path),
        "payload_relative_path": payload_path.relative_to(storage_root).as_posix(),
        "payload_sha256": common.serialization.file_sha256(payload_path),
        "source_case_count": manifest["source_case_count"],
        "sample_count": manifest["sample_count"],
        "transition_count": manifest["transition_count"],
        "inspection": inspection,
        "loader_smoke": smoke,
    }


def _validate_package_record(
    record: Any,
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Validate one dataset receipt record against the current immutable package."""
    if not isinstance(record, dict):
        message = "Dataset receipt package records must be JSON objects."
        raise TypeError(message)
    required = {
        "dataset_name",
        "dataset_id",
        "dataset_view",
        "evaluation_regime",
        "build_status",
        "manifest_relative_path",
        "manifest_sha256",
        "payload_relative_path",
        "payload_sha256",
        "source_case_count",
        "sample_count",
        "transition_count",
        "inspection",
        "loader_smoke",
    }
    if set(record) != required:
        message = f"Dataset receipt package keys are invalid for {record.get('dataset_id')!r}."
        raise ValueError(message)
    dataset_id = common.paths.validate_logical_name(record["dataset_id"], label="dataset_id")
    manifest, inspection, smoke = _package_runtime_evidence(dataset_id, storage_root=storage_root)
    manifest_path = common.paths.get_dataset_metadata_root(storage_root=storage_root) / dataset_id / "dataset_manifest.json"
    payload_path = common.paths.get_dataset_packages_root(storage_root=storage_root) / dataset_id / str(manifest["payload_filename"])
    expected = {
        "dataset_name": manifest["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": manifest["dataset_view"],
        "evaluation_regime": manifest["evaluation_regime"],
        "manifest_relative_path": manifest_path.relative_to(storage_root).as_posix(),
        "manifest_sha256": common.serialization.file_sha256(manifest_path),
        "payload_relative_path": payload_path.relative_to(storage_root).as_posix(),
        "payload_sha256": common.serialization.file_sha256(payload_path),
        "source_case_count": manifest["source_case_count"],
        "sample_count": manifest["sample_count"],
        "transition_count": manifest["transition_count"],
        "inspection": inspection,
        "loader_smoke": smoke,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        message = f"Dataset receipt no longer binds package {dataset_id!r}."
        raise ValueError(message)
    if record["build_status"] not in {"complete", "reused"}:
        message = f"Dataset receipt has an invalid build status for {dataset_id!r}."
        raise ValueError(message)
    return dict(record)


def _package_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return one normalized package request or record identity."""
    return (
        str(value["dataset_name"]),
        str(value["dataset_view"]),
        str(value["evaluation_regime"]),
    )


def _package_binding(record: Mapping[str, Any]) -> dict[str, str]:
    """Return immutable payload and manifest evidence for one package."""
    return {
        "dataset_id": str(record["dataset_id"]),
        "manifest_sha256": str(record["manifest_sha256"]),
        "payload_sha256": str(record["payload_sha256"]),
    }


def _campaign_source_artifact_identity(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Return a digest over every admitted batch and canonical HDF5 artifact."""
    batches: list[dict[str, Any]] = []
    case_count = 0
    for batch in campaign.batches:
        terminal = batch_runtime.admit_terminal_batch(
            batch.batch_storage_name,
            storage_root=storage_root,
            validation_depth="routine",
        )
        if (
            terminal.batch_id != batch.batch_id
            or terminal.batch_identity != batch.batch_identity
            or terminal.simulation_profile != campaign.profile.id
        ):
            message = f"Terminal batch {batch.batch_name!r} disagrees with the package-extension simulation plan."
            raise RuntimeError(message)
        cases: list[dict[str, Any]] = []
        for case in terminal.cases:
            artifact = case.artifact("processed", "case.h5")
            cases.append(
                {
                    **case.record_payload(),
                    "case_hdf5_size_bytes": artifact.size_bytes,
                    "case_hdf5_sha256": artifact.sha256,
                }
            )
        case_count += len(cases)
        batches.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "manifest_sha256": terminal.manifest_sha256,
                "scientific_config_digest": terminal.scientific_config_digest,
                "simulation_profile": terminal.simulation_profile,
                "template": {
                    "relative_path": terminal.template_relative_path,
                    "sha256": terminal.template_sha256,
                },
                "export_contract_sha256": terminal.export_contract_sha256,
                "available_learning_views": list(terminal.available_learning_views),
                "cases": cases,
            }
        )
    payload = {
        "campaign_digest": campaign.campaign_digest,
        "campaign_id": campaign.campaign_id,
        "simulation_profile": campaign.profile.id,
        "membership": campaign.membership,
        "batches": batches,
    }
    return {
        "artifact_set_sha256": common.serialization.canonical_json_sha256(payload),
        "batch_count": len(batches),
        "case_count": case_count,
        "batch_manifests": [
            {
                "batch_id": batch["batch_id"],
                "manifest_sha256": batch["manifest_sha256"],
            }
            for batch in batches
        ],
    }


def _id_companion_binding(
    plan: Mapping[str, Any],
    *,
    current_plans: tuple[dict[str, Any], ...],
    records: Mapping[tuple[str, str, str], dict[str, Any]],
) -> dict[str, str] | None:
    """Return the published ID companion bound to one non-ID extension."""
    if plan["evaluation_regime"] == "id":
        return None
    candidates = tuple(
        candidate for candidate in current_plans if candidate["dataset_view"] == plan["dataset_view"] and candidate["evaluation_regime"] == "id"
    )
    if len(candidates) != 1:
        message = f"Package extension {plan['dataset_name']!r} requires exactly one declared ID leakage companion for the same Dataset view."
        raise ValueError(message)
    companion = records.get(_package_key(candidates[0]))
    if companion is None:
        message = f"Package extension {plan['dataset_name']!r} requires its ID companion to be valid before publication."
        raise RuntimeError(message)
    return _package_binding(companion)


def _validate_package_extension(
    run_id: str,
    plan: Mapping[str, Any],
    *,
    storage_root: Path,
    base_receipt: Mapping[str, Any],
    workflow_receipt: Mapping[str, Any],
    source_artifact_set: Mapping[str, Any],
    id_companion: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Validate one immutable additive package receipt and its exact source."""
    path = _dataset_extension_path(
        run_id,
        plan,
        storage_root=storage_root,
    )
    receipt = _load_json(path, label="Dataset package extension receipt")
    package = receipt.get("package")
    required = {
        "schema_kind",
        "schema_version",
        "status",
        "completed_at",
        "campaign_run_id",
        "campaign_id",
        "campaign_digest",
        "source_git_commit",
        "selected_batch_ids",
        "transfer_receipt_sha256",
        "base_dataset_receipt_sha256",
        "base_all_workflow_receipt_sha256",
        "package_plan",
        "package_plan_digest",
        "source_artifact_set",
        "id_companion",
        "cpu_source_cleanup_reopened",
        "package",
    }
    plan_digest = common.serialization.canonical_json_sha256(dict(plan))
    validated_package = _validate_package_record(
        package,
        storage_root=storage_root,
    )
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != DATASET_EXTENSION_SCHEMA_KIND
        or receipt.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or not isinstance(receipt.get("completed_at"), str)
        or not receipt["completed_at"]
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != base_receipt["campaign_id"]
        or receipt.get("campaign_digest") != base_receipt["campaign_digest"]
        or receipt.get("source_git_commit") != base_receipt["git_commit"]
        or receipt.get("selected_batch_ids") != base_receipt["selected_batch_ids"]
        or receipt.get("transfer_receipt_sha256") != base_receipt["transfer_receipt_sha256"]
        or receipt.get("base_dataset_receipt_sha256") != common.serialization.file_sha256(_dataset_receipt_path(run_id, storage_root=storage_root))
        or receipt.get("base_all_workflow_receipt_sha256") != common.serialization.file_sha256(_all_receipt_path(run_id, storage_root=storage_root))
        or receipt.get("package_plan") != dict(plan)
        or receipt.get("package_plan_digest") != plan_digest
        or receipt.get("source_artifact_set") != dict(source_artifact_set)
        or receipt.get("id_companion") != (None if id_companion is None else dict(id_companion))
        or receipt.get("cpu_source_cleanup_reopened") is not False
        or _package_key(validated_package) != _package_key(plan)
        or workflow_receipt.get("workflow_result") != "success"
    ):
        message = f"Dataset package extension receipt is invalid: {path}"
        raise ValueError(message)
    return validated_package


def validate_campaign_package_state(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the historical base plus every currently requested extension."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    base = validate_dataset_packages_receipt(
        run_id,
        storage_root=storage,
    )
    launch_campaign = campaign_evidence.campaign_for_run(
        run_id,
        storage_root=storage,
    )
    current_campaign = campaign_evidence.current_campaign_for_run(
        run_id,
        storage_root=storage,
    )
    current_plans = tuple(dict(plan) for plan in current_campaign.dataset_packages)
    current_by_key = {_package_key(plan): plan for plan in current_plans}
    launch_by_key = {_package_key(plan): dict(plan) for plan in launch_campaign.dataset_packages}
    if any(current_by_key.get(key) != plan for key, plan in launch_by_key.items()):
        message = "Current campaign removed or changed a launch-time Dataset package."
        raise RuntimeError(message)
    records = {_package_key(record): dict(record) for record in base["packages"]}
    if set(records) != set(launch_by_key):
        message = "Historical Dataset package receipt disagrees with the launch snapshot."
        raise RuntimeError(message)
    extension_paths: list[str] = []
    missing_plans = tuple(plan for plan in current_plans if _package_key(plan) not in records)
    if missing_plans:
        workflow = validate_completed_workflow(
            run_id,
            storage_root=storage,
        )
        source_artifact_set = _campaign_source_artifact_identity(
            current_campaign,
            storage_root=storage,
        )
        ordered_missing = tuple(
            sorted(
                missing_plans,
                key=lambda plan: plan["evaluation_regime"] != "id",
            )
        )
        for plan in ordered_missing:
            companion = _id_companion_binding(
                plan,
                current_plans=current_plans,
                records=records,
            )
            record = _validate_package_extension(
                run_id,
                plan,
                storage_root=storage,
                base_receipt=base,
                workflow_receipt=workflow,
                source_artifact_set=source_artifact_set,
                id_companion=companion,
            )
            records[_package_key(plan)] = record
            extension_paths.append(
                _dataset_extension_path(
                    run_id,
                    plan,
                    storage_root=storage,
                )
                .relative_to(storage)
                .as_posix()
            )
    ordered_records = [records[_package_key(plan)] for plan in current_plans]
    if len({record["dataset_id"] for record in ordered_records}) != len(ordered_records):
        message = "Current campaign package state resolves duplicate Dataset IDs."
        raise RuntimeError(message)
    return {
        "schema_kind": PACKAGE_STATE_SCHEMA_KIND,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "complete",
        "campaign_run_id": run_id,
        "campaign_digest": current_campaign.campaign_digest,
        "package_request_digest": current_campaign.package_request_digest,
        "declared_package_count": len(current_plans),
        "base_dataset_receipt_sha256": common.serialization.file_sha256(_dataset_receipt_path(run_id, storage_root=storage)),
        "packages": ordered_records,
        "extension_receipts": extension_paths,
    }


def _validate_dataset_receipt_payload(
    run_id: str,
    receipt: Mapping[str, Any],
    *,
    storage: Path,
) -> None:
    """Validate one dataset receipt payload against current durable evidence."""
    transfer = campaign_runtime.validate_transferred_campaign(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    receipt_path = _dataset_receipt_path(run_id, storage_root=storage)
    required = {
        "schema_kind",
        "schema_version",
        "status",
        "completed_at",
        "campaign_run_id",
        "campaign_id",
        "campaign_digest",
        "git_commit",
        "selected_batch_ids",
        "declared_package_count",
        "transfer_receipt_sha256",
        "pilot_pre_cleanup_receipt_sha256",
        "packages",
    }
    packages = receipt.get("packages")
    pilot_pre_cleanup_sha256 = None
    if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
        pilot_service.validate_pilot_pre_cleanup(run_id, storage_root=storage)
        pilot_pre_cleanup_sha256 = common.serialization.file_sha256(
            pilot_service.pilot_check_directory(run_id, storage_root=storage) / pilot_service.PILOT_PRE_CLEANUP_FILENAME
        )
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != DATASET_RECEIPT_SCHEMA_KIND
        or receipt.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != campaign.campaign_id
        or receipt.get("campaign_digest") != campaign.campaign_digest
        or receipt.get("git_commit") != transfer["git_commit"]
        or receipt.get("selected_batch_ids") != [batch["batch_id"] for batch in terminal["batches"]]
        or receipt.get("declared_package_count") != len(campaign.dataset_packages)
        or receipt.get("transfer_receipt_sha256") != common.serialization.file_sha256(_transfer_receipt_path(run_id, storage_root=storage))
        or receipt.get("pilot_pre_cleanup_receipt_sha256") != pilot_pre_cleanup_sha256
        or not isinstance(packages, list)
        or len(packages) != len(campaign.dataset_packages)
    ):
        message = f"Dataset package completion receipt is invalid: {receipt_path}"
        raise ValueError(message)
    validated = [_validate_package_record(record, storage_root=storage) for record in packages]
    declared = {_package_key(plan) for plan in campaign.dataset_packages}
    observed = {_package_key(record) for record in validated}
    if observed != declared or len({record["dataset_id"] for record in validated}) != len(validated):
        message = f"Dataset receipt does not cover each declared package exactly once: {receipt_path}"
        raise ValueError(message)


def _repair_dataset_receipt_transfer_binding(
    run_id: str,
    *,
    storage: Path,
) -> dict[str, Any]:
    """Repair only a stale transfer-receipt hash after exact revalidation."""
    receipt_path = _dataset_receipt_path(run_id, storage_root=storage)
    receipt = _load_json(receipt_path, label="dataset package completion receipt")
    candidate = dict(receipt)
    candidate["transfer_receipt_sha256"] = common.serialization.file_sha256(_transfer_receipt_path(run_id, storage_root=storage))
    _validate_dataset_receipt_payload(run_id, candidate, storage=storage)
    common.serialization.atomic_write_json(receipt_path, candidate)
    return validate_dataset_packages_receipt(run_id, storage_root=storage)


def validate_dataset_packages_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate every package, inspection, smoke, and campaign-bound receipt."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    receipt = _load_json(
        _dataset_receipt_path(run_id, storage_root=storage),
        label="dataset package completion receipt",
    )
    _validate_dataset_receipt_payload(run_id, receipt, storage=storage)
    return receipt


def build_campaign_datasets(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build the launch package set or only missing additive extensions."""
    from src.datasets import packages as package_service  # noqa: PLC0415

    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    campaign_runtime.validate_transferred_campaign(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(
        run_id,
        storage_root=storage,
    )
    launch_campaign = campaign_evidence.campaign_for_run(
        run_id,
        storage_root=storage,
    )
    if terminal["dataset_packages"] != list(launch_campaign.dataset_packages):
        message = "Terminal campaign Dataset declarations differ from the launch snapshot."
        raise RuntimeError(message)
    receipt_path = _dataset_receipt_path(run_id, storage_root=storage)
    if not receipt_path.exists():
        lock_path = _dataset_receipt_lock_path(run_id, storage_root=storage)
        with common.locking.exclusive_file_lock(lock_path, blocking=False):
            if not receipt_path.exists():
                pilot_pre_cleanup_sha256 = None
                if launch_campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
                    pilot_service.validate_pilot_pre_cleanup(
                        run_id,
                        storage_root=storage,
                        require_live_evidence=True,
                    )
                    pilot_pre_cleanup_sha256 = common.serialization.file_sha256(
                        pilot_service.pilot_check_directory(
                            run_id,
                            storage_root=storage,
                        )
                        / pilot_service.PILOT_PRE_CLEANUP_FILENAME
                    )
                results = package_service.build_campaign_packages(
                    launch_campaign,
                    storage_root=storage,
                )
                package_records = [_package_record(result, storage_root=storage) for result in results]
                receipt = {
                    "schema_kind": DATASET_RECEIPT_SCHEMA_KIND,
                    "schema_version": WORKFLOW_SCHEMA_VERSION,
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "campaign_run_id": run_id,
                    "campaign_id": launch_campaign.campaign_id,
                    "campaign_digest": launch_campaign.campaign_digest,
                    "git_commit": terminal["git_commit"],
                    "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
                    "declared_package_count": len(launch_campaign.dataset_packages),
                    "transfer_receipt_sha256": common.serialization.file_sha256(_transfer_receipt_path(run_id, storage_root=storage)),
                    "pilot_pre_cleanup_receipt_sha256": (pilot_pre_cleanup_sha256),
                    "packages": package_records,
                }
                common.serialization.atomic_write_json(receipt_path, receipt)
        return validate_dataset_packages_receipt(
            run_id,
            storage_root=storage,
        )

    lock_path = _dataset_receipt_lock_path(run_id, storage_root=storage)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        try:
            base = validate_dataset_packages_receipt(
                run_id,
                storage_root=storage,
            )
        except ValueError:
            base = _repair_dataset_receipt_transfer_binding(
                run_id,
                storage=storage,
            )
    if not _all_receipt_path(run_id, storage_root=storage).is_file():
        return base

    current_campaign = campaign_evidence.current_campaign_for_run(
        run_id,
        storage_root=storage,
    )
    current_plans = tuple(dict(plan) for plan in current_campaign.dataset_packages)
    records = {_package_key(record): dict(record) for record in base["packages"]}
    missing = tuple(plan for plan in current_plans if _package_key(plan) not in records)
    if not missing:
        return validate_campaign_package_state(
            run_id,
            storage_root=storage,
        )

    workflow = validate_completed_workflow(
        run_id,
        storage_root=storage,
    )
    source_artifact_set = _campaign_source_artifact_identity(
        current_campaign,
        storage_root=storage,
    )
    ordered_missing = tuple(sorted(missing, key=lambda plan: plan["evaluation_regime"] != "id"))
    for plan in ordered_missing:
        companion = _id_companion_binding(
            plan,
            current_plans=current_plans,
            records=records,
        )
        extension_path = _dataset_extension_path(
            run_id,
            plan,
            storage_root=storage,
        )
        extension_lock = _dataset_extension_lock_path(
            run_id,
            plan,
            storage_root=storage,
        )
        with common.locking.exclusive_file_lock(extension_lock, blocking=False):
            if not extension_path.exists():
                result = package_service.build_dataset_package(
                    current_campaign,
                    str(plan["dataset_view"]),
                    str(plan["evaluation_regime"]),
                    storage_root=storage,
                )
                package_record = _package_record(
                    result,
                    storage_root=storage,
                )
                extension = {
                    "schema_kind": DATASET_EXTENSION_SCHEMA_KIND,
                    "schema_version": WORKFLOW_SCHEMA_VERSION,
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "campaign_run_id": run_id,
                    "campaign_id": base["campaign_id"],
                    "campaign_digest": base["campaign_digest"],
                    "source_git_commit": base["git_commit"],
                    "selected_batch_ids": base["selected_batch_ids"],
                    "transfer_receipt_sha256": base["transfer_receipt_sha256"],
                    "base_dataset_receipt_sha256": (common.serialization.file_sha256(receipt_path)),
                    "base_all_workflow_receipt_sha256": (common.serialization.file_sha256(_all_receipt_path(run_id, storage_root=storage))),
                    "package_plan": plan,
                    "package_plan_digest": (common.serialization.canonical_json_sha256(plan)),
                    "source_artifact_set": source_artifact_set,
                    "id_companion": companion,
                    "cpu_source_cleanup_reopened": False,
                    "package": package_record,
                }
                extension_path.parent.mkdir(parents=True, exist_ok=True)
                common.serialization.atomic_write_json(
                    extension_path,
                    extension,
                )
            record = _validate_package_extension(
                run_id,
                plan,
                storage_root=storage,
                base_receipt=base,
                workflow_receipt=workflow,
                source_artifact_set=source_artifact_set,
                id_companion=companion,
            )
        records[_package_key(plan)] = record
    return validate_campaign_package_state(
        run_id,
        storage_root=storage,
    )


def find_compatible_completed_campaign_source(
    campaign_path: Path | str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Find one exact transferred base eligible for package-only continuation."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    expected = config_service.load_campaign_config(
        campaign_path,
        require_executable=True,
    )
    root = (
        common.paths.get_generation_meta_root(
            storage_root=storage,
        )
        / "campaigns"
    )
    missing = {
        "schema_kind": "generation_compatible_campaign_source",
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "missing",
        "campaign_digest": expected.campaign_digest,
        "package_request_digest": expected.package_request_digest,
        "campaign_run_id": None,
    }
    if not root.exists():
        return missing
    if not root.is_dir() or root.is_symlink():
        message = f"Generation campaign metadata root is unsafe: {root}"
        raise ValueError(message)
    expected_path = expected.source_path.resolve()
    expected_batches = [batch.batch_id for batch in expected.batches]
    candidates: list[dict[str, Any]] = []
    matching_invalid: list[dict[str, str]] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            message = f"Generation campaign metadata contains an unsafe entry: {directory}"
            raise ValueError(message)
        run_id = common.paths.validate_logical_name(
            directory.name,
            label="campaign_run_id",
        )
        try:
            manifest = campaign_evidence.load_campaign_run(
                run_id,
                storage_root=storage,
            )
        except (FileNotFoundError, TypeError, ValueError):
            continue
        try:
            configured_path = campaign_evidence.resolve_campaign_config_path(manifest["campaign_config"])
        except (FileNotFoundError, TypeError, ValueError):
            continue
        if (
            configured_path != expected_path
            or manifest.get("campaign_id") != expected.campaign_id
            or manifest.get("campaign_digest") != expected.campaign_digest
            or manifest.get("selected_batch_names") != [batch.batch_name for batch in expected.batches]
            or manifest.get("state") != "complete"
            or not (directory / ALL_WORKFLOW_RECEIPT_FILENAME).is_file()
        ):
            continue
        try:
            terminal = campaign_runtime.validate_terminal_campaign(
                run_id,
                storage_root=storage,
            )
            transfer = campaign_runtime.validate_transferred_campaign(
                run_id,
                storage_root=storage,
            )
            workflow = validate_completed_workflow(
                run_id,
                storage_root=storage,
            )
            launch = campaign_evidence.campaign_for_run(
                run_id,
                storage_root=storage,
            )
            current = campaign_evidence.current_campaign_for_run(
                run_id,
                storage_root=storage,
            )
            artifact_set = _campaign_source_artifact_identity(
                current,
                storage_root=storage,
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            matching_invalid.append({"campaign_run_id": run_id, "error": str(error)})
            continue
        if [batch["batch_id"] for batch in terminal["batches"]] != expected_batches:
            matching_invalid.append(
                {
                    "campaign_run_id": run_id,
                    "error": "completed batch inventory differs from the requested simulation plan",
                }
            )
            continue
        launch_keys = {_package_key(plan) for plan in launch.dataset_packages}
        extension_paths = [
            _dataset_extension_path(
                run_id,
                plan,
                storage_root=storage,
            )
            for plan in current.dataset_packages
            if _package_key(plan) not in launch_keys
        ]
        unsafe_paths = [path for path in extension_paths if path.is_symlink() or (path.exists() and not path.is_file())]
        if unsafe_paths:
            matching_invalid.append(
                {
                    "campaign_run_id": run_id,
                    "error": f"unsafe Dataset extension receipt paths: {unsafe_paths}",
                }
            )
            continue
        if any(not path.exists() for path in extension_paths):
            package_status = "extension_required"
        else:
            try:
                validate_campaign_package_state(
                    run_id,
                    storage_root=storage,
                )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                matching_invalid.append({"campaign_run_id": run_id, "error": str(error)})
                continue
            package_status = "complete"
        candidates.append(
            {
                "campaign_run_id": run_id,
                "source_git_commit": terminal["git_commit"],
                "artifact_set_sha256": artifact_set["artifact_set_sha256"],
                "transfer_inventory_sha256": transfer["transfer_inventory_sha256"],
                "base_all_workflow_sha256": common.serialization.file_sha256(_all_receipt_path(run_id, storage_root=storage)),
                "cpu_source_state": workflow["cpu_cleanup_complete"]["status"],
                "package_state": package_status,
            }
        )
    if not candidates:
        if matching_invalid:
            message = f"Scientifically matching completed campaign evidence is not safe for package-only reuse: {matching_invalid}"
            raise RuntimeError(message)
        return missing
    if len(candidates) != 1:
        candidate_ids = [candidate["campaign_run_id"] for candidate in candidates]
        message = f"More than one compatible completed campaign source exists; explicit source selection is required: {candidate_ids}."
        raise RuntimeError(message)
    selected = candidates[0]
    return {
        "schema_kind": "generation_compatible_campaign_source",
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "compatible_complete",
        "campaign_digest": expected.campaign_digest,
        "package_request_digest": expected.package_request_digest,
        **selected,
    }


def _cleanup_directory_records(
    transfer: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return transferred files and per-directory bytes eligible for CPU deletion."""
    directories = [
        str(directory)
        for batch in plan["batches"]
        for directory in (
            batch["meta_directory"],
            batch["raw_directory"],
            batch["processed_directory"],
        )
    ]
    files = transfer.get("files")
    if not isinstance(files, list):
        message = "Transfer receipt has no exact file inventory."
        raise TypeError(message)
    eligible_files = [
        dict(record)
        for record in files
        if isinstance(record, dict) and any(str(record.get("relative_path", "")).startswith(f"{directory}/") for directory in directories)
    ]
    records: list[dict[str, Any]] = []
    for directory in directories:
        owned = [record for record in eligible_files if str(record["relative_path"]).startswith(f"{directory}/")]
        records.append(
            {
                "relative_path": directory,
                "file_count": len(owned),
                "size_bytes": sum(int(record["size_bytes"]) for record in owned),
            }
        )
    if len(eligible_files) != sum(int(record["file_count"]) for record in records):
        message = "CPU cleanup directory inventory overlaps or omits transferred files."
        raise RuntimeError(message)
    return eligible_files, records


def _transferred_file_bindings(transfer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return explicit equal source and destination hashes after publication validation."""
    return [
        {
            "relative_path": record["relative_path"],
            "size_bytes": record["size_bytes"],
            "source_sha256": record["sha256"],
            "destination_sha256": record["sha256"],
        }
        for record in transfer["files"]
    ]


def _workflow_gate_payload(
    *,
    run_id: str,
    terminal: Mapping[str, Any],
    transfer: Mapping[str, Any],
    datasets_receipt: Mapping[str, Any],
    storage: Path,
) -> dict[str, Any]:
    """Return the stable pre-cleanup identity of every completed local gate."""
    return {
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "cpu_source_host": transfer["source_host"],
        "cpu_source_root": transfer["source_storage_root"],
        "gpu_storage_root": str(storage),
        "gpu_generation_destination": str(common.paths.get_generation_root(storage_root=storage)),
        "transferred_file_count": transfer["transferred_file_count"],
        "transferred_bytes": transfer["transferred_bytes"],
        "transfer_inventory_sha256": transfer["transfer_inventory_sha256"],
        "transfer_receipt_sha256": common.serialization.file_sha256(_transfer_receipt_path(run_id, storage_root=storage)),
        "dataset_receipt_sha256": common.serialization.file_sha256(_dataset_receipt_path(run_id, storage_root=storage)),
        "dataset_ids": [record["dataset_id"] for record in datasets_receipt["packages"]],
        "dataset_package_hashes": [
            {
                "dataset_id": record["dataset_id"],
                "manifest_sha256": record["manifest_sha256"],
                "payload_sha256": record["payload_sha256"],
            }
            for record in datasets_receipt["packages"]
        ],
    }


def prepare_all_workflow_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    cleanup_requested: bool = True,
) -> dict[str, Any]:
    """Persist every completed gate before optional verified CPU source cleanup."""
    if not isinstance(cleanup_requested, bool):
        message = "cleanup_requested must be boolean."
        raise TypeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_transferred_campaign(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    datasets_receipt = validate_dataset_packages_receipt(run_id, storage_root=storage)
    plan = campaign_runtime.campaign_transfer_plan(run_id, storage_root=storage)
    eligible_files, source_directories = _cleanup_directory_records(transfer, plan=plan)
    gate_payload = _workflow_gate_payload(
        run_id=run_id,
        terminal=terminal,
        transfer=transfer,
        datasets_receipt=datasets_receipt,
        storage=storage,
    )
    workflow_gate_sha256 = common.serialization.canonical_json_sha256(gate_payload)
    package_records = datasets_receipt["packages"]
    receipt_path = _all_receipt_path(run_id, storage_root=storage)
    existing = _load_json(receipt_path, label="all-workflow receipt") if receipt_path.exists() else None
    if existing is not None and existing.get("workflow_result") == "success":
        try:
            return validate_completed_workflow(run_id, storage_root=storage)
        except ValueError:
            safe_rebuild_identity = {
                "schema_kind": ALL_WORKFLOW_SCHEMA_KIND,
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "campaign_run_id": run_id,
                "campaign_id": terminal["campaign_id"],
                "git_commit": terminal["git_commit"],
                "cpu_source_host": transfer["source_host"],
                "cpu_source_root": transfer["source_storage_root"],
                "transferred_file_count": transfer["transferred_file_count"],
                "transferred_bytes": transfer["transferred_bytes"],
                "transferred_files": _transferred_file_bindings(transfer),
                "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
                "gpu_storage_root": str(storage),
                "gpu_generation_destination": str(common.paths.get_generation_root(storage_root=storage)),
                "dataset_ids": [record["dataset_id"] for record in package_records],
                "dataset_package_hashes": [
                    {
                        "dataset_id": record["dataset_id"],
                        "manifest_sha256": record["manifest_sha256"],
                        "payload_sha256": record["payload_sha256"],
                    }
                    for record in package_records
                ],
                "package_inspection_results": [record["inspection"] for record in package_records],
                "loader_smoke_results": [record["loader_smoke"] for record in package_records],
                "cleanup_requested": False,
                "cpu_source_directories": source_directories,
                "cpu_source_file_count": len(eligible_files),
                "cpu_bytes_reclaimable": sum(int(record["size_bytes"]) for record in eligible_files),
                "cpu_bytes_reclaimed": 0,
                "cpu_cleanup_receipt": None,
                "generation_complete": {
                    "status": "complete",
                    "evidence": {
                        "campaign_terminal_sha256": transfer["campaign_terminal_sha256"],
                        "selected_batch_ids": gate_payload["selected_batch_ids"],
                    },
                },
                "gpu_publication_complete": {
                    "status": "complete",
                    "evidence": {
                        "destination": gate_payload["gpu_generation_destination"],
                        "inventory_sha256": transfer["transfer_inventory_sha256"],
                    },
                },
                "loader_smokes_complete": {
                    "status": "complete",
                    "evidence": {
                        "count": len(package_records),
                        "results": [record["loader_smoke"] for record in package_records],
                    },
                },
                "cpu_cleanup_complete": {
                    "status": "skipped_by_request",
                    "evidence": None,
                },
            }
            transfer_stage = existing.get("transfer_complete")
            dataset_stage = existing.get("dataset_packages_complete")
            stable_transfer_stage = (
                {
                    **transfer_stage,
                    "evidence": {
                        **transfer_stage["evidence"],
                        "transfer_receipt_sha256": gate_payload["transfer_receipt_sha256"],
                    },
                }
                if isinstance(transfer_stage, dict) and isinstance(transfer_stage.get("evidence"), dict)
                else None
            )
            stable_dataset_stage = (
                {
                    **dataset_stage,
                    "evidence": {
                        **dataset_stage["evidence"],
                        "dataset_receipt_sha256": gate_payload["dataset_receipt_sha256"],
                    },
                }
                if isinstance(dataset_stage, dict) and isinstance(dataset_stage.get("evidence"), dict)
                else None
            )
            expected_transfer_stage = {
                "status": "complete",
                "evidence": {
                    "transfer_receipt_sha256": gate_payload["transfer_receipt_sha256"],
                    "file_count": transfer["transferred_file_count"],
                    "size_bytes": transfer["transferred_bytes"],
                    "inventory_sha256": transfer["transfer_inventory_sha256"],
                },
            }
            expected_dataset_stage = {
                "status": "complete",
                "evidence": {
                    "dataset_receipt_sha256": gate_payload["dataset_receipt_sha256"],
                    "dataset_ids": gate_payload["dataset_ids"],
                },
            }
            variable_rebuild_keys = {
                "workflow_gate_sha256",
                "transfer_complete",
                "dataset_packages_complete",
                "recorded_at",
                "ready_at",
                "completed_at",
                "workflow_result",
            }
            timestamps_are_valid = all(
                isinstance(existing.get(key), str) and bool(existing[key]) for key in ("recorded_at", "ready_at", "completed_at")
            )
            if (
                set(existing) != {*safe_rebuild_identity, *variable_rebuild_keys}
                or any(existing.get(key) != value for key, value in safe_rebuild_identity.items())
                or stable_transfer_stage != expected_transfer_stage
                or stable_dataset_stage != expected_dataset_stage
                or not timestamps_are_valid
            ):
                raise
    recorded_at = existing.get("recorded_at") if existing is not None else _utc_now()
    cpu_stage_status = "pending" if cleanup_requested else "skipped_by_request"
    workflow_result = "ready_for_cpu_cleanup" if cleanup_requested else "success"
    receipt = {
        "schema_kind": ALL_WORKFLOW_SCHEMA_KIND,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "cpu_source_host": transfer["source_host"],
        "cpu_source_root": transfer["source_storage_root"],
        "gpu_storage_root": str(storage),
        "gpu_generation_destination": str(common.paths.get_generation_root(storage_root=storage)),
        "transferred_file_count": transfer["transferred_file_count"],
        "transferred_bytes": transfer["transferred_bytes"],
        "transferred_files": _transferred_file_bindings(transfer),
        "dataset_ids": [record["dataset_id"] for record in package_records],
        "dataset_package_hashes": [
            {
                "dataset_id": record["dataset_id"],
                "manifest_sha256": record["manifest_sha256"],
                "payload_sha256": record["payload_sha256"],
            }
            for record in package_records
        ],
        "package_inspection_results": [record["inspection"] for record in package_records],
        "loader_smoke_results": [record["loader_smoke"] for record in package_records],
        "cleanup_requested": cleanup_requested,
        "cpu_source_directories": source_directories,
        "cpu_source_file_count": len(eligible_files),
        "cpu_bytes_reclaimable": sum(int(record["size_bytes"]) for record in eligible_files),
        "cpu_bytes_reclaimed": 0,
        "cpu_cleanup_receipt": None,
        "workflow_gate_sha256": workflow_gate_sha256,
        "generation_complete": {
            "status": "complete",
            "evidence": {
                "campaign_terminal_sha256": transfer["campaign_terminal_sha256"],
                "selected_batch_ids": gate_payload["selected_batch_ids"],
            },
        },
        "transfer_complete": {
            "status": "complete",
            "evidence": {
                "transfer_receipt_sha256": gate_payload["transfer_receipt_sha256"],
                "file_count": transfer["transferred_file_count"],
                "size_bytes": transfer["transferred_bytes"],
                "inventory_sha256": transfer["transfer_inventory_sha256"],
            },
        },
        "gpu_publication_complete": {
            "status": "complete",
            "evidence": {
                "destination": gate_payload["gpu_generation_destination"],
                "inventory_sha256": transfer["transfer_inventory_sha256"],
            },
        },
        "dataset_packages_complete": {
            "status": "complete",
            "evidence": {
                "dataset_receipt_sha256": gate_payload["dataset_receipt_sha256"],
                "dataset_ids": gate_payload["dataset_ids"],
            },
        },
        "loader_smokes_complete": {
            "status": "complete",
            "evidence": {
                "count": len(package_records),
                "results": [record["loader_smoke"] for record in package_records],
            },
        },
        "cpu_cleanup_complete": {
            "status": cpu_stage_status,
            "evidence": None,
        },
        "recorded_at": recorded_at,
        "ready_at": _utc_now(),
        "completed_at": _utc_now() if workflow_result == "success" else None,
        "workflow_result": workflow_result,
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    if workflow_result == "success":
        return validate_completed_workflow(run_id, storage_root=storage)
    return validate_all_workflow_receipt(run_id, storage_root=storage)


def validate_all_workflow_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate all local stage evidence in a pending or successful receipt."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_transferred_campaign(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    datasets_receipt = validate_dataset_packages_receipt(run_id, storage_root=storage)
    receipt_path = _all_receipt_path(run_id, storage_root=storage)
    receipt = _load_json(receipt_path, label="all-workflow receipt")
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "selected_batch_ids",
        "cpu_source_host",
        "cpu_source_root",
        "gpu_storage_root",
        "gpu_generation_destination",
        "transferred_file_count",
        "transferred_bytes",
        "transferred_files",
        "dataset_ids",
        "dataset_package_hashes",
        "package_inspection_results",
        "loader_smoke_results",
        "cleanup_requested",
        "cpu_source_directories",
        "cpu_source_file_count",
        "cpu_bytes_reclaimable",
        "cpu_bytes_reclaimed",
        "cpu_cleanup_receipt",
        "workflow_gate_sha256",
        "generation_complete",
        "transfer_complete",
        "gpu_publication_complete",
        "dataset_packages_complete",
        "loader_smokes_complete",
        "cpu_cleanup_complete",
        "recorded_at",
        "ready_at",
        "completed_at",
        "workflow_result",
    }
    gate_payload = _workflow_gate_payload(
        run_id=run_id,
        terminal=terminal,
        transfer=transfer,
        datasets_receipt=datasets_receipt,
        storage=storage,
    )
    expected_gate_sha256 = common.serialization.canonical_json_sha256(gate_payload)
    package_records = datasets_receipt["packages"]
    eligible_files, source_directories = _cleanup_directory_records(
        transfer,
        plan=campaign_runtime.campaign_transfer_plan(run_id, storage_root=storage),
    )
    expected_complete_stages = {
        "generation_complete": {
            "status": "complete",
            "evidence": {
                "campaign_terminal_sha256": transfer["campaign_terminal_sha256"],
                "selected_batch_ids": gate_payload["selected_batch_ids"],
            },
        },
        "transfer_complete": {
            "status": "complete",
            "evidence": {
                "transfer_receipt_sha256": gate_payload["transfer_receipt_sha256"],
                "file_count": transfer["transferred_file_count"],
                "size_bytes": transfer["transferred_bytes"],
                "inventory_sha256": transfer["transfer_inventory_sha256"],
            },
        },
        "gpu_publication_complete": {
            "status": "complete",
            "evidence": {
                "destination": gate_payload["gpu_generation_destination"],
                "inventory_sha256": transfer["transfer_inventory_sha256"],
            },
        },
        "dataset_packages_complete": {
            "status": "complete",
            "evidence": {
                "dataset_receipt_sha256": gate_payload["dataset_receipt_sha256"],
                "dataset_ids": gate_payload["dataset_ids"],
            },
        },
        "loader_smokes_complete": {
            "status": "complete",
            "evidence": {
                "count": len(package_records),
                "results": [record["loader_smoke"] for record in package_records],
            },
        },
    }
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != ALL_WORKFLOW_SCHEMA_KIND
        or receipt.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != terminal["campaign_id"]
        or receipt.get("git_commit") != terminal["git_commit"]
        or receipt.get("selected_batch_ids") != gate_payload["selected_batch_ids"]
        or receipt.get("cpu_source_host") != transfer["source_host"]
        or receipt.get("cpu_source_root") != transfer["source_storage_root"]
        or receipt.get("gpu_storage_root") != str(storage)
        or receipt.get("gpu_generation_destination") != gate_payload["gpu_generation_destination"]
        or receipt.get("transferred_file_count") != transfer["transferred_file_count"]
        or receipt.get("transferred_bytes") != transfer["transferred_bytes"]
        or receipt.get("transferred_files") != _transferred_file_bindings(transfer)
        or receipt.get("dataset_ids") != gate_payload["dataset_ids"]
        or receipt.get("dataset_package_hashes") != gate_payload["dataset_package_hashes"]
        or receipt.get("package_inspection_results") != [record["inspection"] for record in package_records]
        or receipt.get("loader_smoke_results") != [record["loader_smoke"] for record in package_records]
        or receipt.get("cpu_source_directories") != source_directories
        or receipt.get("cpu_source_file_count") != len(eligible_files)
        or receipt.get("cpu_bytes_reclaimable") != sum(int(record["size_bytes"]) for record in eligible_files)
        or receipt.get("workflow_gate_sha256") != expected_gate_sha256
        or any(receipt.get(stage) != expected for stage, expected in expected_complete_stages.items())
        or not isinstance(receipt.get("recorded_at"), str)
        or not receipt["recorded_at"]
        or not isinstance(receipt.get("ready_at"), str)
        or not receipt["ready_at"]
        or receipt.get("workflow_result") not in {"ready_for_cpu_cleanup", "success"}
    ):
        message = f"All-workflow receipt is invalid: {receipt_path}"
        raise ValueError(message)
    cleanup_stage = receipt.get("cpu_cleanup_complete")
    if not isinstance(cleanup_stage, dict) or cleanup_stage.get("status") not in {
        "pending",
        "skipped_by_request",
        "complete",
    }:
        message = f"All-workflow CPU cleanup stage is invalid: {receipt_path}"
        raise ValueError(message)
    cleanup_evidence = cleanup_stage.get("evidence")
    if receipt["workflow_result"] == "ready_for_cpu_cleanup":
        invalid_cleanup_state = (
            receipt["cleanup_requested"] is not True
            or cleanup_stage["status"] != "pending"
            or cleanup_evidence is not None
            or receipt["cpu_cleanup_receipt"] is not None
            or receipt["cpu_bytes_reclaimed"] != 0
            or receipt["completed_at"] is not None
        )
    elif cleanup_stage["status"] == "skipped_by_request":
        invalid_cleanup_state = (
            receipt["cleanup_requested"] is not False
            or cleanup_evidence is not None
            or receipt["cpu_cleanup_receipt"] is not None
            or receipt["cpu_bytes_reclaimed"] != 0
            or not isinstance(receipt["completed_at"], str)
            or not receipt["completed_at"]
        )
    else:
        invalid_cleanup_state = (
            cleanup_stage["status"] != "complete"
            or receipt["cleanup_requested"] is not True
            or not isinstance(cleanup_evidence, dict)
            or set(cleanup_evidence or {}) != {"authorization_sha256", "receipt_sha256", "reclaimed_bytes"}
            or receipt["cpu_cleanup_receipt"] != cleanup_evidence
            or receipt["cpu_bytes_reclaimed"] != receipt["cpu_bytes_reclaimable"]
            or (cleanup_evidence or {}).get("reclaimed_bytes") != receipt["cpu_bytes_reclaimed"]
            or any(
                not isinstance((cleanup_evidence or {}).get(field), str)
                or len((cleanup_evidence or {})[field]) != _SHA256_LENGTH
                or any(character not in "0123456789abcdef" for character in (cleanup_evidence or {})[field])
                for field in ("authorization_sha256", "receipt_sha256")
            )
            or not isinstance(receipt["completed_at"], str)
            or not receipt["completed_at"]
        )
    if invalid_cleanup_state:
        message = f"All-workflow CPU cleanup state is inconsistent: {receipt_path}"
        raise ValueError(message)
    return receipt


def validate_completed_workflow(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Require a fully successful all-workflow receipt."""
    receipt = validate_all_workflow_receipt(run_id, storage_root=storage_root)
    if receipt["workflow_result"] != "success":
        message = f"All workflow is not terminally successful for {run_id!r}."
        raise RuntimeError(message)
    return receipt


def _pilot_cleanup_evidence(
    workflow: Mapping[str, Any],
) -> pilot_service.CleanupWorkflowEvidence:
    """Return the terminal cleanup binding from a validated workflow receipt."""
    cleanup = workflow["cpu_cleanup_complete"]
    status = cleanup["status"]
    receipt = workflow["cpu_cleanup_receipt"]
    if status == "complete":
        if not isinstance(receipt, dict):
            message = "Completed workflow cleanup lacks its validated receipt evidence."
            raise RuntimeError(message)
        receipt_sha256 = cast("str", receipt["receipt_sha256"])
        reclaimed_bytes = cast("int", receipt["reclaimed_bytes"])
    elif status == "skipped_by_request":
        receipt_sha256 = None
        reclaimed_bytes = 0
    else:
        message = "Pilot cleanup requires a terminal all-workflow receipt."
        raise RuntimeError(message)
    return pilot_service.CleanupWorkflowEvidence(
        campaign_run_id=cast("str", workflow["campaign_run_id"]),
        status=cast("str", status),
        receipt_sha256=receipt_sha256,
        reclaimed_bytes=reclaimed_bytes,
    )


def record_pilot_cleanup_result(
    run_id: str,
    *,
    storage_root: Path | str | None,
    cpu_source_removed: bool,
    cpu_bytes_reclaimed: int,
    cpu_cleanup_receipt_sha256: str | None,
    transfer_staging_removed: bool,
    staging_bytes_reclaimed: int,
    staging_cleanup_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Bind validated workflow cleanup evidence into the pilot receipt."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    workflow = validate_all_workflow_receipt(run_id, storage_root=storage)
    return pilot_service.finalize_cleanup_receipt(
        run_id,
        storage_root=storage,
        workflow_evidence=_pilot_cleanup_evidence(workflow),
        cpu_source_removed=cpu_source_removed,
        cpu_bytes_reclaimed=cpu_bytes_reclaimed,
        cpu_cleanup_receipt_sha256=cpu_cleanup_receipt_sha256,
        transfer_staging_removed=transfer_staging_removed,
        staging_bytes_reclaimed=staging_bytes_reclaimed,
        staging_cleanup_receipt_sha256=staging_cleanup_receipt_sha256,
    )


def validate_completed_pilot_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate pilot cleanup against the terminal all-workflow receipt."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    workflow = validate_all_workflow_receipt(run_id, storage_root=storage)
    evidence = _pilot_cleanup_evidence(workflow)
    receipt = pilot_service.validate_pilot_receipt(
        run_id,
        storage_root=storage,
        require_cleanup_complete=True,
    )
    if evidence.campaign_run_id != run_id:
        message = f"Pilot cleanup workflow evidence belongs to another run: {run_id}"
        raise ValueError(message)
    cleanup_requested = bool(receipt["cleanup"]["cleanup_requested"])
    if cleanup_requested:
        if (
            evidence.status != "complete"
            or evidence.receipt_sha256 != receipt["cleanup"]["cpu_source"]["receipt_sha256"]
            or evidence.reclaimed_bytes != receipt["cleanup"]["cpu_source"]["bytes_reclaimed"]
        ):
            message = f"Pilot CPU cleanup is not bound to the all-workflow receipt: {run_id}"
            raise ValueError(message)
    elif evidence.status != "skipped_by_request":
        message = f"Pilot CPU-source retention is not bound to the all-workflow receipt: {run_id}"
        raise ValueError(message)
    return receipt


def _cleanup_authorization_payload(
    run_id: str,
    *,
    storage: Path,
    workflow: Mapping[str, Any],
    transfer: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact cross-host CPU cleanup authorization identity."""
    eligible_files, source_directories = _cleanup_directory_records(
        transfer,
        plan=campaign_runtime.campaign_transfer_plan(run_id, storage_root=storage),
    )
    return {
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "source_host": transfer["source_host"],
        "source_storage_root": transfer["source_storage_root"],
        "destination_storage_root": str(storage),
        "transfer_receipt_sha256": common.serialization.file_sha256(_transfer_receipt_path(run_id, storage_root=storage)),
        "dataset_receipt_sha256": common.serialization.file_sha256(_dataset_receipt_path(run_id, storage_root=storage)),
        "workflow_gate_sha256": workflow["workflow_gate_sha256"],
        "source_inventory_sha256": common.serialization.canonical_json_sha256(eligible_files),
        "source_file_count": len(eligible_files),
        "source_bytes": sum(int(record["size_bytes"]) for record in eligible_files),
        "source_directories": source_directories,
    }


def cpu_cleanup_authorization(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Authorize deletion only after transfer, package, inspection, smoke, and receipt gates."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    workflow = validate_all_workflow_receipt(run_id, storage_root=storage)
    cleanup_status = workflow["cpu_cleanup_complete"]["status"]
    if cleanup_status not in {"pending", "skipped_by_request", "complete"}:
        message = f"Workflow {run_id!r} is not eligible for CPU cleanup."
        raise RuntimeError(message)
    transfer = campaign_runtime.validate_transferred_campaign(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    payload = _cleanup_authorization_payload(
        run_id,
        storage=storage,
        workflow=workflow,
        transfer=transfer,
        terminal=terminal,
    )
    return {
        **payload,
        "authorization_sha256": common.serialization.canonical_json_sha256(payload),
    }


def _remote_authorization_payload(
    run_id: str,
    *,
    storage: Path,
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_host: str,
    destination_storage_root: str,
    transfer_receipt_sha256: str,
    dataset_receipt_sha256: str,
    workflow_gate_sha256: str,
    source_inventory_sha256: str,
    source_file_count: int,
    source_bytes: int,
) -> dict[str, Any]:
    """Return the CPU reconstruction of a GPU-issued cleanup authorization."""
    source_directories = [
        {
            "relative_path": directory,
            "file_count": 0,
            "size_bytes": 0,
        }
        for batch in plan["batches"]
        for directory in (
            batch["meta_directory"],
            batch["raw_directory"],
            batch["processed_directory"],
        )
    ]
    return {
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "source_host": source_host,
        "source_storage_root": str(storage),
        "destination_storage_root": destination_storage_root,
        "transfer_receipt_sha256": transfer_receipt_sha256,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "workflow_gate_sha256": workflow_gate_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "source_file_count": source_file_count,
        "source_bytes": source_bytes,
        "source_directories": source_directories,
    }


def _other_campaign_source_references(
    run_id: str,
    *,
    storage: Path,
    source_directories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return other campaign runs that reference any cleanup directory."""
    owned = {(storage / str(record["relative_path"])).resolve() for record in source_directories}
    campaigns_root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    if not campaigns_root.is_dir():
        return []
    references: list[dict[str, Any]] = []
    for directory in sorted(campaigns_root.iterdir()):
        if directory.name == run_id:
            continue
        if not directory.is_dir() or directory.is_symlink():
            message = f"Campaign metadata contains an unsafe run entry: {directory}"
            raise ValueError(message)
        other_run_id = common.paths.validate_logical_name(directory.name, label="campaign_run_id")
        manifest = campaign_evidence.load_campaign_run(other_run_id, storage_root=storage)
        shared: set[str] = set()
        for batch in manifest["batches"]:
            for key in ("meta_directory", "raw_directory", "processed_directory"):
                candidate = Path(str(batch[key])).resolve()
                if not candidate.is_relative_to(storage):
                    message = f"Campaign {other_run_id!r} references source outside storage: {candidate}"
                    raise ValueError(message)
                if candidate in owned:
                    shared.add(candidate.relative_to(storage).as_posix())
        if shared:
            references.append(
                {
                    "campaign_run_id": other_run_id,
                    "state": manifest["state"],
                    "shared_directories": sorted(shared),
                }
            )
    return references


def _source_cleanup_inventory(
    run_id: str,
    *,
    storage: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return current CPU files and exact eligible source-directory metrics."""
    full = campaign_runtime.campaign_transfer_inventory(run_id, storage_root=storage)
    return _cleanup_directory_records(full, plan=plan)


def _cleanup_authorization_from_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the cleanup authorization fields bound by one receipt identity."""
    return {
        "campaign_run_id": identity.get("campaign_run_id"),
        "campaign_id": identity.get("campaign_id"),
        "git_commit": identity.get("git_commit"),
        "selected_batch_ids": identity.get("selected_batch_ids"),
        "source_host": identity.get("source_host"),
        "source_storage_root": identity.get("source_storage_root"),
        "destination_storage_root": identity.get("destination_storage_root"),
        "transfer_receipt_sha256": identity.get("transfer_receipt_sha256"),
        "dataset_receipt_sha256": identity.get("dataset_receipt_sha256"),
        "workflow_gate_sha256": identity.get("workflow_gate_sha256"),
        "source_inventory_sha256": identity.get("source_inventory_sha256"),
        "source_file_count": identity.get("source_file_count"),
        "source_bytes": identity.get("source_bytes_reclaimed"),
        "source_directories": identity.get("source_directories"),
    }


def _cleanup_source_directories_are_valid(
    value: Any,
    *,
    storage: Path,
) -> bool:
    """Return whether exact cleanup-directory metrics are safe and nonoverlapping."""
    if not isinstance(value, list) or not value:
        return False
    relative_paths: list[Path] = []
    generation_root = common.paths.get_generation_root(storage_root=storage).resolve()
    for record in value:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "file_count",
            "size_bytes",
        }:
            return False
        raw_path = record.get("relative_path")
        if not isinstance(raw_path, str) or any(character in raw_path for character in "\r\n\t"):
            return False
        relative = Path(raw_path)
        resolved = (storage / relative).resolve(strict=False)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != raw_path
            or resolved == generation_root
            or not resolved.is_relative_to(generation_root)
            or isinstance(record.get("file_count"), bool)
            or not isinstance(record.get("file_count"), int)
            or int(record["file_count"]) < 0
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or int(record["size_bytes"]) < 0
        ):
            return False
        relative_paths.append(relative)
    if len(relative_paths) != len(set(relative_paths)):
        return False
    return not any(
        first.is_relative_to(second) or second.is_relative_to(first)
        for index, first in enumerate(relative_paths)
        for second in relative_paths[index + 1 :]
    )


def _cleanup_receipt_identity_is_valid(
    identity: Any,
    *,
    run_id: str,
    storage: Path,
    authorization_sha256: str,
) -> bool:
    """Return whether one cleanup identity is complete, hash-bound, and safe."""
    if not isinstance(identity, dict) or set(identity) != _CLEANUP_RECEIPT_IDENTITY_KEYS:
        return False
    source_directories = identity.get("source_directories")
    directory_records = cast("list[dict[str, Any]]", source_directories)
    selected_batch_ids = identity.get("selected_batch_ids")
    slurm_job_ids = identity.get("slurm_job_ids")
    digest_fields = (
        "campaign_terminal_sha256",
        "authorization_sha256",
        "transfer_receipt_sha256",
        "dataset_receipt_sha256",
        "workflow_gate_sha256",
        "source_inventory_sha256",
        "destination_inventory_sha256",
    )
    if (
        identity.get("schema_kind") != CPU_CLEANUP_SCHEMA_KIND
        or identity.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or identity.get("status") != "complete"
        or identity.get("campaign_run_id") != run_id
        or not isinstance(identity.get("campaign_id"), str)
        or not identity["campaign_id"]
        or not isinstance(identity.get("git_commit"), str)
        or not identity["git_commit"]
        or identity.get("authorization_sha256") != authorization_sha256
        or identity.get("source_storage_root") != str(storage)
        or not isinstance(identity.get("source_host"), str)
        or not identity["source_host"]
        or any(character in identity["source_host"] for character in "\r\n\t")
        or not isinstance(identity.get("destination_storage_root"), str)
        or any(character in identity["destination_storage_root"] for character in "\r\n\t")
    ):
        return False
    destination = Path(identity["destination_storage_root"])
    if not destination.is_absolute() or destination == Path("/") or ".." in destination.parts:
        return False
    if (
        not isinstance(selected_batch_ids, list)
        or not selected_batch_ids
        or not all(isinstance(batch_id, str) and batch_id for batch_id in selected_batch_ids)
        or len(selected_batch_ids) != len(set(selected_batch_ids))
        or not isinstance(slurm_job_ids, list)
        or not slurm_job_ids
        or not all(isinstance(job_id, str) and job_id.isdigit() for job_id in slurm_job_ids)
        or len(slurm_job_ids) != len(set(slurm_job_ids))
        or not isinstance(identity.get("scheduler_job_name"), str)
        or not identity["scheduler_job_name"]
        or any(
            not isinstance(identity.get(field), str)
            or len(identity[field]) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in identity[field])
            for field in digest_fields
        )
        or not _cleanup_source_directories_are_valid(source_directories, storage=storage)
        or identity.get("destination_inventory_sha256") != identity.get("source_inventory_sha256")
        or isinstance(identity.get("source_file_count"), bool)
        or not isinstance(identity.get("source_file_count"), int)
        or identity["source_file_count"] < 0
        or isinstance(identity.get("source_bytes_reclaimed"), bool)
        or not isinstance(identity.get("source_bytes_reclaimed"), int)
        or identity["source_bytes_reclaimed"] < 0
        or sum(int(record["file_count"]) for record in directory_records) != identity.get("source_file_count")
        or sum(int(record["size_bytes"]) for record in directory_records) != identity.get("source_bytes_reclaimed")
        or common.serialization.canonical_json_sha256(_cleanup_authorization_from_identity(identity)) != authorization_sha256
    ):
        return False
    terminal_path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / "campaign_terminal.json"
    return (
        terminal_path.is_file()
        and not terminal_path.is_symlink()
        and common.serialization.file_sha256(terminal_path) == identity.get("campaign_terminal_sha256")
    )


def _validate_remote_cleanup_receipt(
    run_id: str,
    *,
    storage: Path,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Validate retained CPU cleanup provenance after bulk source removal."""
    path = _cleanup_receipt_path(run_id, storage_root=storage)
    receipt = _load_json(path, label="CPU source cleanup receipt")
    identity = {key: receipt.get(key) for key in _CLEANUP_RECEIPT_IDENTITY_KEYS}
    source_directories = identity.get("source_directories")
    invalid_source_state = not isinstance(source_directories, list) or any(
        (storage / str(record["relative_path"])).exists() or (storage / str(record["relative_path"])).is_symlink()
        for record in source_directories or []
        if isinstance(record, dict) and "relative_path" in record
    )
    if (
        set(receipt) != _CLEANUP_RECEIPT_IDENTITY_KEYS | {"completed_at"}
        or not _cleanup_receipt_identity_is_valid(
            identity,
            run_id=run_id,
            storage=storage,
            authorization_sha256=authorization_sha256,
        )
        or not isinstance(receipt.get("completed_at"), str)
        or not receipt["completed_at"]
        or invalid_source_state
    ):
        message = f"CPU source cleanup receipt is invalid: {path}"
        raise ValueError(message)
    return {
        **receipt,
        "receipt_path": str(path),
        "receipt_sha256": common.serialization.file_sha256(path),
    }


def _cleanup_transaction_path(run_id: str, *, storage: Path) -> Path:
    """Return the run-scoped CPU cleanup transaction directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    return common.paths.get_generation_state_root(storage_root=storage) / "source-cleanup" / safe_id


def _cleanup_transaction_moves(
    *,
    storage: Path,
    transaction: Path,
    source_directories: list[dict[str, Any]],
    require_sources: bool,
) -> list[tuple[Path, Path]]:
    """Resolve safe canonical-to-transaction directory moves."""
    payload_root = transaction / "payload"
    resolved_payload = payload_root.resolve(strict=False)
    generation_root = common.paths.get_generation_root(storage_root=storage).resolve()
    moves: list[tuple[Path, Path]] = []
    for record in source_directories:
        relative = Path(str(record["relative_path"]))
        source = storage / relative
        target = payload_root / relative
        resolved_source = source.resolve(strict=False)
        resolved_target = target.resolve(strict=False)
        if (
            resolved_source == generation_root
            or not resolved_source.is_relative_to(generation_root)
            or not resolved_target.is_relative_to(resolved_payload)
            or source.is_symlink()
            or target.is_symlink()
        ):
            message = f"CPU cleanup transaction path is protected or unsafe: {source}"
            raise ValueError(message)
        if require_sources and (not source.is_dir() or source.stat().st_dev != transaction.stat().st_dev):
            message = f"CPU cleanup source is missing, unsafe, or cross-device: {source}"
            raise ValueError(message)
        moves.append((source, target))
    return moves


def _cleanup_request_matches_identity(
    identity: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> bool:
    """Return whether retry arguments exactly match the durable transaction."""
    expected = {
        "source_host": request["source_host"],
        "destination_storage_root": request["destination_storage_root"],
        "transfer_receipt_sha256": request["transfer_receipt_sha256"],
        "dataset_receipt_sha256": request["dataset_receipt_sha256"],
        "workflow_gate_sha256": request["workflow_gate_sha256"],
        "source_inventory_sha256": request["source_inventory_sha256"],
        "source_file_count": request["source_file_count"],
        "source_bytes_reclaimed": request["source_bytes"],
        "authorization_sha256": request["authorization_sha256"],
    }
    return all(identity.get(key) == value for key, value in expected.items())


def _load_cleanup_transaction(
    run_id: str,
    *,
    storage: Path,
    transaction: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and authenticate one interrupted CPU cleanup transaction."""
    if not transaction.is_dir() or transaction.is_symlink():
        message = f"CPU cleanup transaction is missing or unsafe: {transaction}"
        raise ValueError(message)
    marker_path = transaction / "transaction.json"
    marker = _load_json(marker_path, label="CPU cleanup transaction marker")
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "authorization_sha256",
        "status",
        "receipt_identity",
        "created_at",
        "completed_at",
    }
    identity = marker.get("receipt_identity")
    status = marker.get("status")
    completed_at = marker.get("completed_at")
    if (
        set(marker) != required
        or marker.get("schema_kind") != CPU_CLEANUP_TRANSACTION_SCHEMA_KIND
        or marker.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or marker.get("campaign_run_id") != run_id
        or marker.get("authorization_sha256") != request["authorization_sha256"]
        or status not in {"planned", "detached", "disposed"}
        or not isinstance(marker.get("created_at"), str)
        or not marker["created_at"]
        or (status == "disposed" and (not isinstance(completed_at, str) or not completed_at))
        or (status != "disposed" and completed_at is not None)
        or not _cleanup_receipt_identity_is_valid(
            identity,
            run_id=run_id,
            storage=storage,
            authorization_sha256=str(request["authorization_sha256"]),
        )
        or not _cleanup_request_matches_identity(
            cast("Mapping[str, Any]", identity),
            request=request,
        )
    ):
        message = f"CPU cleanup transaction marker is invalid: {marker_path}"
        raise ValueError(message)
    return marker


def _recover_cpu_cleanup_transaction(
    run_id: str,
    *,
    storage: Path,
    transaction: Path,
    receipt_path: Path,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Rollback a planned detach or finish an authorized detached disposal."""
    marker_path = transaction / "transaction.json"
    if not marker_path.exists():
        entries = list(transaction.iterdir()) if transaction.is_dir() and not transaction.is_symlink() else []
        recoverable_entries = all(
            entry.is_file() and not entry.is_symlink() and entry.name.startswith(".transaction.json.") and entry.name.endswith(".tmp")
            for entry in entries
        )
        if transaction.is_dir() and not transaction.is_symlink() and recoverable_entries:
            shutil.rmtree(transaction)
            return None
        message = f"CPU cleanup transaction has no recoverable marker: {transaction}"
        raise RuntimeError(message)
    marker = _load_cleanup_transaction(
        run_id,
        storage=storage,
        transaction=transaction,
        request=request,
    )
    identity = marker["receipt_identity"]
    source_directories = identity["source_directories"]
    moves = _cleanup_transaction_moves(
        storage=storage,
        transaction=transaction,
        source_directories=source_directories,
        require_sources=False,
    )
    if marker["status"] == "planned":
        if receipt_path.exists():
            message = "A planned cleanup transaction conflicts with a completed cleanup receipt."
            raise RuntimeError(message)
        for source, target in reversed(moves):
            source_present = source.exists() or source.is_symlink()
            target_present = target.exists() or target.is_symlink()
            if source_present and target_present:
                message = f"Cleanup rollback has both source and detached payload: {source}"
                raise RuntimeError(message)
            if target_present:
                if not target.is_dir() or target.is_symlink():
                    message = f"Cleanup rollback payload is unsafe: {target}"
                    raise ValueError(message)
                source.parent.mkdir(parents=True, exist_ok=True)
                target.replace(source)
            elif not source_present or not source.is_dir() or source.is_symlink():
                message = f"Cleanup rollback cannot reconstruct source directory: {source}"
                raise RuntimeError(message)
        shutil.rmtree(transaction)
        return None

    for source, _target in moves:
        if source.exists() or source.is_symlink():
            message = f"Detached cleanup transaction unexpectedly has a live source: {source}"
            raise RuntimeError(message)
    payload_root = transaction / "payload"
    if payload_root.exists() or payload_root.is_symlink():
        if not payload_root.is_dir() or payload_root.is_symlink():
            message = f"Detached cleanup payload is unsafe: {payload_root}"
            raise ValueError(message)
        shutil.rmtree(payload_root)
    if marker["status"] != "disposed":
        marker["status"] = "disposed"
        marker["completed_at"] = _utc_now()
        common.serialization.atomic_write_json(marker_path, marker)
    receipt = {**identity, "completed_at": marker["completed_at"]}
    if receipt_path.exists():
        existing = _load_json(receipt_path, label="CPU source cleanup receipt")
        if existing != receipt:
            message = f"CPU cleanup receipt conflicts with its transaction: {receipt_path}"
            raise RuntimeError(message)
    else:
        common.serialization.atomic_write_json(receipt_path, receipt)
    validated = _validate_remote_cleanup_receipt(
        run_id,
        storage=storage,
        authorization_sha256=str(request["authorization_sha256"]),
    )
    shutil.rmtree(transaction)
    return validated


def _cleanup_cpu_campaign_source_locked(
    run_id: str,
    *,
    storage: Path,
    request: Mapping[str, Any],
    confirm: bool,
) -> dict[str, Any]:
    """Execute one serialized CPU cleanup attempt or recovery."""
    authorization_sha256 = str(request["authorization_sha256"])
    receipt_path = _cleanup_receipt_path(run_id, storage_root=storage)
    transaction = _cleanup_transaction_path(run_id, storage=storage)
    if transaction.exists() or transaction.is_symlink():
        if not confirm:
            if receipt_path.exists():
                return _validate_remote_cleanup_receipt(
                    run_id,
                    storage=storage,
                    authorization_sha256=authorization_sha256,
                )
            message = "An interrupted CPU cleanup transaction requires an explicit confirmed retry."
            raise RuntimeError(message)
        recovered = _recover_cpu_cleanup_transaction(
            run_id,
            storage=storage,
            transaction=transaction,
            receipt_path=receipt_path,
            request=request,
        )
        if recovered is not None:
            return recovered
    if receipt_path.exists():
        return _validate_remote_cleanup_receipt(
            run_id,
            storage=storage,
            authorization_sha256=authorization_sha256,
        )

    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    plan = campaign_runtime.campaign_transfer_plan(run_id, storage_root=storage)
    eligible_files, source_directories = _source_cleanup_inventory(
        run_id,
        storage=storage,
        plan=plan,
    )
    payload = _remote_authorization_payload(
        run_id,
        storage=storage,
        terminal=terminal,
        plan=plan,
        source_host=str(request["source_host"]),
        destination_storage_root=str(request["destination_storage_root"]),
        transfer_receipt_sha256=str(request["transfer_receipt_sha256"]),
        dataset_receipt_sha256=str(request["dataset_receipt_sha256"]),
        workflow_gate_sha256=str(request["workflow_gate_sha256"]),
        source_inventory_sha256=str(request["source_inventory_sha256"]),
        source_file_count=int(request["source_file_count"]),
        source_bytes=int(request["source_bytes"]),
    )
    actual_directory_by_path = {record["relative_path"]: record for record in source_directories}
    for record in payload["source_directories"]:
        actual = actual_directory_by_path[str(record["relative_path"])]
        record["file_count"] = actual["file_count"]
        record["size_bytes"] = actual["size_bytes"]
    source_inventory_sha256 = str(request["source_inventory_sha256"])
    source_file_count = int(request["source_file_count"])
    source_bytes = int(request["source_bytes"])
    if (
        common.serialization.canonical_json_sha256(payload) != authorization_sha256
        or common.serialization.canonical_json_sha256(eligible_files) != source_inventory_sha256
        or len(eligible_files) != source_file_count
        or sum(int(record["size_bytes"]) for record in eligible_files) != source_bytes
    ):
        message = "CPU source no longer matches the GPU cleanup authorization."
        raise ValueError(message)
    shared_references = _other_campaign_source_references(
        run_id,
        storage=storage,
        source_directories=source_directories,
    )
    if shared_references:
        message = f"CPU cleanup source is referenced by other campaign runs: {shared_references}"
        raise RuntimeError(message)
    status = campaign_runtime.campaign_status(run_id, storage_root=storage, query_scheduler=True)
    scheduler_errors = [str(record["error"]) for record in (status["squeue"], status["sacct"]) if record.get("error") is not None]
    if scheduler_errors:
        message = f"Cannot prove campaign jobs are inactive and terminal: {scheduler_errors}"
        raise RuntimeError(message)
    if status["squeue"]["output"]:
        message = "CPU cleanup rejects an active campaign."
        raise RuntimeError(message)
    if status["campaign_state"] not in _SOURCE_CLEANUP_READY_CAMPAIGN_STATES:
        message = f"CPU cleanup requires a successful terminal source publication, got {status['campaign_state']!r}."
        raise RuntimeError(message)
    result = {
        "campaign_run_id": run_id,
        "status": "eligible",
        "mode": "delete" if confirm else "dry-run",
        "source_directories": source_directories,
        "source_file_count": source_file_count,
        "reclaimable_bytes": source_bytes,
        "authorization_sha256": authorization_sha256,
        "destination_storage_root": request["destination_storage_root"],
    }
    if not confirm:
        return result

    terminal_path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / "campaign_terminal.json"
    receipt_identity = {
        "schema_kind": CPU_CLEANUP_SCHEMA_KIND,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "complete",
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "slurm_job_ids": terminal["slurm_job_ids"],
        "scheduler_job_name": terminal["scheduler_job_name"],
        "campaign_terminal_sha256": common.serialization.file_sha256(terminal_path),
        "authorization_sha256": authorization_sha256,
        "source_host": request["source_host"],
        "source_storage_root": str(storage),
        "destination_storage_root": request["destination_storage_root"],
        "transfer_receipt_sha256": request["transfer_receipt_sha256"],
        "dataset_receipt_sha256": request["dataset_receipt_sha256"],
        "workflow_gate_sha256": request["workflow_gate_sha256"],
        "source_inventory_sha256": source_inventory_sha256,
        "destination_inventory_sha256": source_inventory_sha256,
        "source_directories": source_directories,
        "source_file_count": source_file_count,
        "source_bytes_reclaimed": source_bytes,
    }
    if not _cleanup_receipt_identity_is_valid(
        receipt_identity,
        run_id=run_id,
        storage=storage,
        authorization_sha256=authorization_sha256,
    ):
        message = "CPU cleanup receipt identity is invalid before source detachment."
        raise ValueError(message)
    marker_path = transaction / "transaction.json"
    marker = {
        "schema_kind": CPU_CLEANUP_TRANSACTION_SCHEMA_KIND,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "authorization_sha256": authorization_sha256,
        "status": "planned",
        "receipt_identity": receipt_identity,
        "created_at": _utc_now(),
        "completed_at": None,
    }
    common.serialization.atomic_write_json(marker_path, marker)
    payload_root = transaction / "payload"
    payload_root.mkdir()
    try:
        planned_moves = _cleanup_transaction_moves(
            storage=storage,
            transaction=transaction,
            source_directories=source_directories,
            require_sources=True,
        )
    except Exception:
        shutil.rmtree(transaction)
        raise
    moves: list[tuple[Path, Path]] = []
    try:
        for source, target in planned_moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            moves.append((source, target))
    except Exception:
        rollback_errors: list[str] = []
        for source, target in reversed(moves):
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                target.replace(source)
            except OSError as error:  # noqa: PERF203 -- attempt every rollback
                rollback_errors.append(f"{target}: {error}")
        if not rollback_errors:
            shutil.rmtree(transaction)
        else:
            message = f"CPU cleanup rollback is incomplete: {rollback_errors}"
            raise RuntimeError(message) from None
        raise
    marker["status"] = "detached"
    common.serialization.atomic_write_json(marker_path, marker)
    shutil.rmtree(payload_root)
    marker["status"] = "disposed"
    marker["completed_at"] = _utc_now()
    common.serialization.atomic_write_json(marker_path, marker)
    receipt = {**receipt_identity, "completed_at": marker["completed_at"]}
    common.serialization.atomic_write_json(receipt_path, receipt)
    validated = _validate_remote_cleanup_receipt(
        run_id,
        storage=storage,
        authorization_sha256=authorization_sha256,
    )
    shutil.rmtree(transaction)
    return validated


def cleanup_cpu_campaign_source(
    run_id: str,
    *,
    storage_root: Path | str,
    source_host: str,
    destination_storage_root: str,
    transfer_receipt_sha256: str,
    dataset_receipt_sha256: str,
    workflow_gate_sha256: str,
    source_inventory_sha256: str,
    source_file_count: int,
    source_bytes: int,
    authorization_sha256: str,
    confirm: bool = False,
) -> dict[str, Any]:
    """Dry-run or transactionally remove one fully authorized CPU campaign source."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    if not isinstance(destination_storage_root, str):
        message = "CPU cleanup destination storage root must be text."
        raise TypeError(message)
    destination = Path(destination_storage_root)
    digests = (
        transfer_receipt_sha256,
        dataset_receipt_sha256,
        workflow_gate_sha256,
        source_inventory_sha256,
        authorization_sha256,
    )
    if (
        not isinstance(source_host, str)
        or not source_host
        or any(character in source_host for character in "\r\n\t")
        or not destination.is_absolute()
        or destination == Path("/")
        or ".." in destination.parts
        or any(character in destination_storage_root for character in "\r\n\t")
        or not all(isinstance(value, str) for value in digests)
        or any(len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value) for value in digests)
        or isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count < 0
        or isinstance(source_bytes, bool)
        or not isinstance(source_bytes, int)
        or source_bytes < 0
    ):
        message = "CPU cleanup authorization arguments are malformed."
        raise ValueError(message)
    request = {
        "source_host": source_host,
        "destination_storage_root": destination_storage_root,
        "transfer_receipt_sha256": transfer_receipt_sha256,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "workflow_gate_sha256": workflow_gate_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "source_file_count": source_file_count,
        "source_bytes": source_bytes,
        "authorization_sha256": authorization_sha256,
    }
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    lock_path = common.paths.get_generation_state_root(storage_root=storage) / "source-cleanup-locks" / f"{safe_id}.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        return _cleanup_cpu_campaign_source_locked(
            run_id,
            storage=storage,
            request=request,
            confirm=confirm,
        )


def record_cpu_cleanup_complete(
    run_id: str,
    *,
    storage_root: Path | str | None,
    authorization_sha256: str,
    cleanup_receipt_sha256: str,
    reclaimed_bytes: int,
) -> dict[str, Any]:
    """Finalize the local workflow receipt from one authorized remote cleanup."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    workflow = validate_all_workflow_receipt(run_id, storage_root=storage)
    if workflow["cpu_cleanup_complete"]["status"] == "complete":
        if (
            workflow["cpu_cleanup_receipt"].get("authorization_sha256") != authorization_sha256
            or workflow["cpu_cleanup_receipt"].get("receipt_sha256") != cleanup_receipt_sha256
            or workflow["cpu_bytes_reclaimed"] != reclaimed_bytes
        ):
            message = f"Existing local CPU cleanup evidence conflicts for {run_id!r}."
            raise FileExistsError(message)
        return workflow
    authorization = cpu_cleanup_authorization(run_id, storage_root=storage)
    if (
        authorization["authorization_sha256"] != authorization_sha256
        or len(cleanup_receipt_sha256) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in cleanup_receipt_sha256)
        or reclaimed_bytes != authorization["source_bytes"]
    ):
        message = "Remote CPU cleanup result does not match the local authorization."
        raise ValueError(message)
    workflow["cpu_cleanup_complete"] = {
        "status": "complete",
        "evidence": {
            "authorization_sha256": authorization_sha256,
            "receipt_sha256": cleanup_receipt_sha256,
            "reclaimed_bytes": reclaimed_bytes,
        },
    }
    workflow["cleanup_requested"] = True
    workflow["cpu_cleanup_receipt"] = dict(workflow["cpu_cleanup_complete"]["evidence"])
    workflow["cpu_bytes_reclaimed"] = reclaimed_bytes
    workflow["completed_at"] = _utc_now()
    workflow["workflow_result"] = "success"
    common.serialization.atomic_write_json(
        _all_receipt_path(run_id, storage_root=storage),
        workflow,
    )
    return validate_completed_workflow(run_id, storage_root=storage)


def record_workflow_failure(
    run_id: str,
    *,
    storage_root: Path | str,
    stage: str,
    continuation_command: str,
    cpu_bytes_retained: int,
) -> Path:
    """Persist and re-admit one append-only compact workflow failure record."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    if not stage or not continuation_command or cpu_bytes_retained < 0:
        message = "Workflow failure evidence is incomplete."
        raise ValueError(message)
    campaign_evidence.load_campaign_run(safe_id, storage_root=storage)
    root = (
        campaign_evidence.campaign_run_directory(
            safe_id,
            storage_root=storage,
        )
        / "workflow_failures"
    )
    lock_path = common.paths.get_generation_state_root(storage_root=storage) / "workflow-failure-locks" / f"{safe_id}.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            message = f"Workflow failure evidence directory is unsafe: {root}"
            raise ValueError(message)
        indices = [
            int(candidate.stem.removeprefix("failure-"))
            for candidate in root.glob("failure-*.json")
            if candidate.is_file() and not candidate.is_symlink() and candidate.stem.removeprefix("failure-").isdigit()
        ]
        path = root / f"failure-{max(indices, default=0) + 1:04d}.json"
        receipt = {
            "schema_kind": "generation_all_workflow_failure",
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "campaign_run_id": safe_id,
            "stage": stage,
            "continuation_command": continuation_command,
            "cpu_bytes_retained": cpu_bytes_retained,
            "recorded_at": _utc_now(),
        }
        common.serialization.atomic_write_json(path, receipt)
        if _load_json(path, label="workflow failure receipt") != receipt:
            message = f"Workflow failure receipt could not be re-admitted: {path}"
            raise RuntimeError(message)
    return path.resolve()


def _manifest_source_directories(
    run_id: str,
    *,
    storage: Path,
) -> tuple[Path, ...]:
    """Return run-owned CPU source directories even before terminal publication."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    directories: list[Path] = []
    roots = {
        "meta_directory": common.paths.get_generation_meta_root(storage_root=storage).resolve(),
        "raw_directory": common.paths.get_generation_raw_root(storage_root=storage).resolve(),
        "processed_directory": common.paths.get_generation_processed_root(storage_root=storage).resolve(),
    }
    for batch in manifest["batches"]:
        for key, root in roots.items():
            value = Path(str(batch[key])).resolve()
            if value == root or not value.is_relative_to(root):
                message = f"Campaign source directory escapes its owned {key} root: {value}"
                raise ValueError(message)
            directories.append(value)
    if len(directories) != len(set(directories)):
        message = f"Campaign {run_id!r} contains duplicate source directories."
        raise ValueError(message)
    return tuple(directories)


def campaign_source_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = False,
) -> dict[str, Any]:
    """Report one host's campaign source bytes, state, and cleanup eligibility."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    cleanup_path = _cleanup_receipt_path(run_id, storage_root=storage)
    campaign_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    if cleanup_path.exists():
        raw_receipt = _load_json(cleanup_path, label="CPU source cleanup receipt")
        authorization_sha256 = raw_receipt.get("authorization_sha256")
        if not isinstance(authorization_sha256, str):
            message = f"CPU source cleanup receipt has no authorization identity: {cleanup_path}"
            raise ValueError(message)
        receipt = _validate_remote_cleanup_receipt(
            run_id,
            storage=storage,
            authorization_sha256=authorization_sha256,
        )
        campaign_metadata_bytes = _tree_size(campaign_directory)
        return {
            "campaign_run_id": run_id,
            "campaign_state": "successful",
            "source_state": "cleaned",
            "transfer_status": "transferred_and_cleaned",
            "cleanup_eligibility": "already_complete",
            "active_slurm": False,
            "source_directories": receipt["source_directories"],
            "reclaimable_bytes": 0,
            "campaign_metadata_bytes": campaign_metadata_bytes,
            "size_bytes": campaign_metadata_bytes,
            "cleanup_receipt": str(cleanup_path),
        }
    directories = _manifest_source_directories(run_id, storage=storage)
    directory_records: list[dict[str, Any]] = [
        {
            "path": str(path),
            "exists": path.is_dir() and not path.is_symlink(),
            "size_bytes": _tree_size(path),
        }
        for path in directories
    ]
    state = "unknown"
    active = None
    scheduler_error = None
    try:
        status = campaign_runtime.campaign_status(
            run_id,
            storage_root=storage,
            query_scheduler=query_scheduler,
        )
        state = str(status["campaign_state"])
        active = bool(status["squeue"]["output"]) if query_scheduler else None
        scheduler_errors = [
            str(record["error"]) for record in (status["squeue"], status["sacct"]) if query_scheduler and record.get("error") is not None
        ]
        scheduler_error = "; ".join(scheduler_errors) or None
    except (OSError, RuntimeError, ValueError) as error:
        scheduler_error = str(error)
    terminal = (campaign_directory / "campaign_terminal.json").is_file()
    cleanup_eligibility = (
        "requires_gpu_authorization"
        if terminal and active is False and scheduler_error is None and state in _SOURCE_CLEANUP_READY_CAMPAIGN_STATES
        else "ineligible"
    )
    transfer_status = "gpu_receipt_present" if (campaign_directory / "transfer_complete.json").is_file() else "not_recorded_on_this_host"
    if terminal and transfer_status == "gpu_receipt_present":
        source_state = "retained"
    elif terminal:
        source_state = "awaiting_collection"
    else:
        source_state = "active"
    source_bytes = sum(int(record["size_bytes"]) for record in directory_records)
    campaign_metadata_bytes = _tree_size(campaign_directory)
    return {
        "campaign_run_id": run_id,
        "campaign_state": state,
        "source_state": source_state,
        "transfer_status": transfer_status,
        "cleanup_eligibility": cleanup_eligibility,
        "active_slurm": active,
        "scheduler_error": scheduler_error,
        "source_directories": directory_records,
        "reclaimable_bytes": source_bytes,
        "campaign_metadata_bytes": campaign_metadata_bytes,
        "size_bytes": source_bytes + campaign_metadata_bytes,
        "cleanup_receipt": None,
    }


def _safe_campaign_source_status(
    run_id: str,
    *,
    storage: Path,
    query_scheduler: bool,
) -> dict[str, Any]:
    """Return one run status or its explicit validation error."""
    try:
        return campaign_source_status(
            run_id,
            storage_root=storage,
            query_scheduler=query_scheduler,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        return {
            "campaign_run_id": run_id,
            "campaign_state": "invalid",
            "source_state": "invalid",
            "error": str(error),
            "reclaimable_bytes": 0,
            "size_bytes": 0,
        }


def storage_status(
    *,
    storage_root: Path | str,
    role: str,
    run_id: str | None = None,
    query_scheduler: bool = False,
) -> dict[str, Any]:
    """Report generation, dataset, staging, package, run, and cleanup storage state."""
    if role not in {"gpu", "cpu"}:
        message = "Storage status role must be 'gpu' or 'cpu'."
        raise ValueError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    generation_root = common.paths.get_generation_root(storage_root=storage)
    datasets_root = common.paths.get_datasets_root(storage_root=storage)
    experiments_root = common.paths.get_experiments_root(storage_root=storage)
    campaigns_root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    run_ids = (
        [common.paths.validate_logical_name(run_id, label="campaign_run_id")]
        if run_id is not None
        else (
            [
                common.paths.validate_logical_name(path.name, label="campaign_run_id")
                for path in sorted(campaigns_root.iterdir())
                if path.is_dir() and not path.is_symlink()
            ]
            if campaigns_root.is_dir()
            else []
        )
    )
    runs = [
        _safe_campaign_source_status(
            current_run_id,
            storage=storage,
            query_scheduler=query_scheduler,
        )
        for current_run_id in run_ids
    ]
    staging = list(
        workspace_service.transfer_staging_candidates(
            storage_root=storage,
            run_id=run_id,
        )
    )
    packages: list[dict[str, Any]] = []
    package_errors: list[dict[str, str]] = []
    metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage)
    if metadata_root.is_dir():
        from src.datasets import packages as package_service  # noqa: PLC0415

        for directory in sorted(metadata_root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                package_errors.append({"dataset_id": directory.name, "error": "unsafe metadata entry"})
                continue
            try:
                dataset_id = common.paths.validate_logical_name(directory.name, label="dataset_id")
                manifest = package_service.load_package_manifest(dataset_id, storage_root=storage)
                package_service.inspect_dataset_package(dataset_id, storage_root=storage)
                packages.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_view": manifest["dataset_view"],
                        "evaluation_regime": manifest["evaluation_regime"],
                        "size_bytes": _tree_size(directory) + _tree_size(common.paths.get_dataset_packages_root(storage_root=storage) / dataset_id),
                    }
                )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                package_errors.append({"dataset_id": directory.name, "error": str(error)})
    package_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for package in packages:
        key = (str(package["dataset_view"]), str(package["evaluation_regime"]))
        group = package_groups.setdefault(
            key,
            {
                "dataset_view": key[0],
                "evaluation_regime": key[1],
                "package_count": 0,
                "size_bytes": 0,
            },
        )
        group["package_count"] += 1
        group["size_bytes"] += int(package["size_bytes"])
    return {
        "role": role,
        "storage_root": str(storage),
        "roots": {
            "generation": str(generation_root),
            "datasets": str(datasets_root),
            "experiments": str(experiments_root),
            "transfer_staging": str(storage / ".incoming"),
        },
        "generation_total_bytes": _tree_size(generation_root),
        "datasets_total_bytes": _tree_size(datasets_root),
        "experiments_total_bytes": _tree_size(experiments_root),
        "runs": runs,
        "packages": packages,
        "packages_by_view_regime": [package_groups[key] for key in sorted(package_groups)],
        "missing_source_errors": package_errors,
        "transfer_staging": staging,
        "transfer_staging_bytes": sum(int(record["size_bytes"]) for record in staging),
        "protected_cleanup_targets": [
            str(generation_root),
            str(datasets_root),
        ],
    }


def record_incomplete_campaign_datasets(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Record that Dataset membership is incomplete without minting Dataset IDs."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_partially_transferred_campaign(
        run_id,
        storage_root=storage,
    )
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    receipt_path = run_directory / INCOMPLETE_DATASET_RECEIPT_FILENAME
    receipt = {
        "schema_kind": "generation_dataset_packages_incomplete",
        "schema_version": 1,
        "status": "incomplete",
        "campaign_run_id": run_id,
        "campaign_id": transfer["campaign_id"],
        "git_commit": transfer["git_commit"],
        "partial_transfer_receipt_sha256": common.serialization.file_sha256(run_directory / "transfer_partial.json"),
        "successful_cases": transfer["successful_cases"],
        "failed_cases": transfer["failed_cases"],
        "declared_package_count": len(
            campaign_evidence.campaign_from_manifest(campaign_evidence.load_campaign_run(run_id, storage_root=storage)).dataset_packages
        ),
        "packages": [],
        "dataset_ids": [],
        "reason": "required campaign membership is incomplete",
        "recorded_at": _utc_now(),
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    return validate_incomplete_campaign_datasets(run_id, storage_root=storage)


def validate_incomplete_campaign_datasets(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate that incomplete membership has not been published as complete."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_partially_transferred_campaign(
        run_id,
        storage_root=storage,
    )
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    receipt_path = run_directory / INCOMPLETE_DATASET_RECEIPT_FILENAME
    receipt = _load_json(receipt_path, label="incomplete Dataset receipt")
    required = {
        "schema_kind",
        "schema_version",
        "status",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "partial_transfer_receipt_sha256",
        "successful_cases",
        "failed_cases",
        "declared_package_count",
        "packages",
        "dataset_ids",
        "reason",
        "recorded_at",
    }
    campaign = campaign_evidence.campaign_from_manifest(campaign_evidence.load_campaign_run(run_id, storage_root=storage))
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != "generation_dataset_packages_incomplete"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "incomplete"
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != transfer["campaign_id"]
        or receipt.get("git_commit") != transfer["git_commit"]
        or receipt.get("partial_transfer_receipt_sha256") != common.serialization.file_sha256(run_directory / "transfer_partial.json")
        or receipt.get("successful_cases") != transfer["successful_cases"]
        or receipt.get("failed_cases") != transfer["failed_cases"]
        or receipt.get("declared_package_count") != len(campaign.dataset_packages)
        or receipt.get("packages") != []
        or receipt.get("dataset_ids") != []
        or not isinstance(receipt.get("recorded_at"), str)
        or not receipt["recorded_at"]
    ):
        message = f"Incomplete Dataset receipt is invalid: {receipt_path}"
        raise ValueError(message)
    return receipt


def prepare_partial_completion_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Persist validated partial completion with retained-source resume metadata."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_partially_transferred_campaign(
        run_id,
        storage_root=storage,
    )
    datasets = validate_incomplete_campaign_datasets(
        run_id,
        storage_root=storage,
    )
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    receipt = {
        "schema_kind": "generation_partial_completion",
        "schema_version": 1,
        "workflow_result": "partial",
        "campaign_run_id": run_id,
        "campaign_id": transfer["campaign_id"],
        "git_commit": transfer["git_commit"],
        "campaign_state": "completed_with_failures",
        "partial_transfer_receipt_sha256": common.serialization.file_sha256(run_directory / "transfer_partial.json"),
        "incomplete_dataset_receipt_sha256": common.serialization.file_sha256(run_directory / INCOMPLETE_DATASET_RECEIPT_FILENAME),
        "successful_cases": transfer["successful_cases"],
        "failed_cases": transfer["failed_cases"],
        "dataset_status": datasets["status"],
        "dataset_ids": [],
        "cpu_source_host": transfer["source_host"],
        "cpu_source_root": transfer["source_storage_root"],
        "cpu_source_retained": True,
        "cleanup_requested": False,
        "resume_command": f"resume {run_id}",
        "completed_at": _utc_now(),
    }
    common.serialization.atomic_write_json(
        run_directory / PARTIAL_COMPLETION_RECEIPT_FILENAME,
        receipt,
    )
    return validate_partial_completion_receipt(run_id, storage_root=storage)


def validate_partial_completion_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate partial publication, incomplete packages, and retained-source intent."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    transfer = campaign_runtime.validate_partially_transferred_campaign(
        run_id,
        storage_root=storage,
    )
    datasets = validate_incomplete_campaign_datasets(
        run_id,
        storage_root=storage,
    )
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    receipt_path = run_directory / PARTIAL_COMPLETION_RECEIPT_FILENAME
    receipt = _load_json(receipt_path, label="partial completion receipt")
    required = {
        "schema_kind",
        "schema_version",
        "workflow_result",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "campaign_state",
        "partial_transfer_receipt_sha256",
        "incomplete_dataset_receipt_sha256",
        "successful_cases",
        "failed_cases",
        "dataset_status",
        "dataset_ids",
        "cpu_source_host",
        "cpu_source_root",
        "cpu_source_retained",
        "cleanup_requested",
        "resume_command",
        "completed_at",
    }
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != "generation_partial_completion"
        or receipt.get("schema_version") != 1
        or receipt.get("workflow_result") != "partial"
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != transfer["campaign_id"]
        or receipt.get("git_commit") != transfer["git_commit"]
        or receipt.get("campaign_state") != "completed_with_failures"
        or receipt.get("partial_transfer_receipt_sha256") != common.serialization.file_sha256(run_directory / "transfer_partial.json")
        or receipt.get("incomplete_dataset_receipt_sha256") != common.serialization.file_sha256(run_directory / INCOMPLETE_DATASET_RECEIPT_FILENAME)
        or receipt.get("successful_cases") != transfer["successful_cases"]
        or receipt.get("failed_cases") != transfer["failed_cases"]
        or receipt.get("dataset_status") != datasets["status"]
        or receipt.get("dataset_ids") != []
        or receipt.get("cpu_source_host") != transfer["source_host"]
        or receipt.get("cpu_source_root") != transfer["source_storage_root"]
        or receipt.get("cpu_source_retained") is not True
        or receipt.get("cleanup_requested") is not False
        or receipt.get("resume_command") != f"resume {run_id}"
        or not isinstance(receipt.get("completed_at"), str)
        or not receipt["completed_at"]
    ):
        message = f"Partial completion receipt is invalid: {receipt_path}"
        raise ValueError(message)
    return receipt
