"""
===============================================================================
generation_validation_pilot.py
===============================================================================
Own technical pilot terminal evidence, storage accounting, and summaries.
Responsibilities:
  - Terminally classify every pilot case as validated success or retained failure
  - Record exact CPU, staging, and permanent GPU storage inventories
  - Persist one canonical pilot receipt and derived CSV/Markdown views
Design principles:
  - Pilot cases never enter evaluation or learning dataset membership
  - Cleanup gates bind retained evidence before any source deletion
  - Failed and successful materials use the same generic evidence path
This module does NOT:
  - Execute SSH, Slurm, COMSOL, rsync, or destructive source cleanup
  - Retune scientific values, invent tolerances, or build production datasets
===============================================================================
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

from . import generation_validation_pilot_analysis as analysis_service

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

PILOT_TERMINAL_SCHEMA_KIND: Final = "generation_pilot_campaign_terminal"
PILOT_RECEIPT_SCHEMA_KIND: Final = "generation_pilot_check"
PILOT_SOURCE_INVENTORY_FILENAME: Final = "pilot_source_inventory.json"
PILOT_STAGING_INVENTORY_FILENAME: Final = "pilot_staging_inventory.json"
PILOT_RECEIPT_FILENAME: Final = "pilot_check.json"
PILOT_PRE_CLEANUP_FILENAME: Final = "pilot_check_pre_cleanup.json"
PILOT_STAGING_CLEANUP_FILENAME: Final = "pilot_staging_cleanup.json"
PILOT_SUMMARY_CSV: Final = "summary.csv"
PILOT_SUMMARY_MARKDOWN: Final = "summary.md"
PILOT_SCHEMA_VERSION: Final = 1
_MIN_REGULAR_STATE_COUNT: Final = 2
PILOT_RESULT_CLASSES: Final = (
    "PASS",
    "PASS_WITH_WARNINGS",
    "TOO_FAST",
    "NOT_DRY_WITHIN_HORIZON",
    "INPUT_FAILED",
    "SOLVER_FAILED",
    "EXPORT_FAILED",
    "CONVERSION_FAILED",
    "INVALID_RESULT",
    "PHYSICAL_CONTRACT_VIOLATION",
)
_FAILURE_STAGE_RESULT_CLASS: Final = {
    "input": "INPUT_FAILED",
    "solver": "SOLVER_FAILED",
    "export": "EXPORT_FAILED",
    "conversion": "CONVERSION_FAILED",
    "invalid_result": "INVALID_RESULT",
}
_RUNTIME_FAILURE_RESULT_CLASSES: Final = frozenset({"INPUT_FAILED", "SOLVER_FAILED", "EXPORT_FAILED", "CONVERSION_FAILED"})
_REQUIRED_CASE_RESULT_FIELDS: Final = frozenset(
    {
        "material",
        "material_role",
        "case_kind",
        "case_index",
        "case_id",
        "solver_status",
        "result_class",
        "target_reached",
        "drying_time_h",
        "drying_time_days",
        "last_valid_time_h",
        "stop_reason",
        "failed_stage",
        "warning_count",
        "final_X_wb_bulk",
        "final_f_wet_dm",
    }
)
_MAX_FIXED_POINT_ITERATIONS: Final = 20
_SHA256_LENGTH: Final = 64


@dataclass(frozen=True, slots=True)
class CleanupWorkflowEvidence:
    """Validated terminal workflow evidence required by pilot cleanup."""

    campaign_run_id: str
    status: str
    receipt_sha256: str | None
    reclaimed_bytes: int

    def __post_init__(self) -> None:
        """Validate one terminal cleanup or retention binding."""
        if not self.campaign_run_id:
            message = "Cleanup workflow evidence requires a campaign-run identity."
            raise ValueError(message)
        if self.status == "complete":
            digest = self.receipt_sha256
            if (
                not isinstance(digest, str)
                or len(digest) != _SHA256_LENGTH
                or any(character not in "0123456789abcdef" for character in digest)
                or isinstance(self.reclaimed_bytes, bool)
                or not isinstance(self.reclaimed_bytes, int)
                or self.reclaimed_bytes < 0
            ):
                message = "Completed cleanup workflow evidence is malformed."
                raise ValueError(message)
        elif self.status == "skipped_by_request":
            if self.receipt_sha256 is not None or self.reclaimed_bytes != 0:
                message = "Retained-source workflow evidence cannot report cleanup results."
                raise ValueError(message)
        else:
            message = f"Cleanup workflow evidence has nonterminal status {self.status!r}."
            raise ValueError(message)


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def pilot_check_directory(run_id: str, *, storage_root: Path | str | None) -> Path:
    """Return the canonical pilot-check metadata directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="pilot_check_id")
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    return common.paths.get_generation_meta_root(storage_root=storage) / "pilot_checks" / safe_id


def pilot_receipt_path(run_id: str, *, storage_root: Path | str | None) -> Path:
    """Return the canonical machine-readable pilot receipt path."""
    return pilot_check_directory(run_id, storage_root=storage_root) / PILOT_RECEIPT_FILENAME


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


def _validate_case_result(record: Any) -> None:
    """Validate one canonical pilot case result and its uncensored-time rules."""
    if not isinstance(record, dict) or not _REQUIRED_CASE_RESULT_FIELDS.issubset(record):
        message = "Pilot case result is missing required canonical fields."
        raise ValueError(message)
    if (
        not isinstance(record["material"], str)
        or not record["material"]
        or record["material_role"] not in config_service.MATERIAL_ROLES
        or record["case_kind"] not in config_service.PILOT_CASE_KINDS
        or record["result_class"] not in PILOT_RESULT_CLASSES
        or record["solver_status"] not in {"not_started", "failed", "success"}
        or not isinstance(record["case_index"], int)
        or isinstance(record["case_index"], bool)
        or record["case_index"] < 1
        or not isinstance(record["case_id"], str)
        or not record["case_id"]
        or not isinstance(record["stop_reason"], str)
        or not record["stop_reason"]
        or not isinstance(record["warning_count"], int)
        or isinstance(record["warning_count"], bool)
        or record["warning_count"] < 0
        or not isinstance(record.get("warnings"), list)
        or record["warning_count"] != len(record["warnings"])
        or (record["target_reached"] is not None and not isinstance(record["target_reached"], bool))
    ):
        message = f"Pilot case result has invalid canonical values: {record.get('case_id')!r}"
        raise ValueError(message)
    numeric_fields = (
        "drying_time_h",
        "drying_time_days",
        "last_valid_time_h",
        "final_X_wb_bulk",
        "final_f_wet_dm",
    )
    if any(
        value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)))
        for value in (record[name] for name in numeric_fields)
    ):
        message = f"Pilot case result has non-finite canonical values: {record['case_id']}"
        raise ValueError(message)
    drying_time_h = record["drying_time_h"]
    drying_time_days = record["drying_time_days"]
    if (
        (drying_time_h is None) != (drying_time_days is None)
        or (record["target_reached"] is not True and drying_time_h is not None)
        or (drying_time_h is not None and not math.isclose(float(drying_time_days), float(drying_time_h) / 24.0, rel_tol=0.0, abs_tol=1.0e-12))
    ):
        message = f"Pilot drying time is fabricated or inconsistent: {record['case_id']}"
        raise ValueError(message)
    failed_stage = record["failed_stage"]
    if failed_stage is None:
        if record["result_class"] in _FAILURE_STAGE_RESULT_CLASS.values():
            message = f"Pilot runtime failure lacks a stage: {record['case_id']}"
            raise ValueError(message)
    elif failed_stage not in _FAILURE_STAGE_RESULT_CLASS or _FAILURE_STAGE_RESULT_CLASS[failed_stage] != record["result_class"]:
        message = f"Pilot failure stage and result class disagree: {record['case_id']}"
        raise ValueError(message)


def _validate_case_results(
    records: Any,
    *,
    expected_materials: tuple[str, ...] | None = None,
) -> None:
    """Validate each canonical result and optional campaign material coverage."""
    if not isinstance(records, list) or not records:
        message = "Pilot receipt must contain case results."
        raise ValueError(message)
    for record in records:
        _validate_case_result(record)
    if expected_materials is not None and (
        not expected_materials
        or len(expected_materials) != len(set(expected_materials))
        or {record["material"] for record in records} != set(expected_materials)
    ):
        message = "Pilot case results do not cover the resolved campaign material inventory."
        raise ValueError(message)


def _validate_storage_projection(value: Any) -> None:
    """Validate a projection bound to one resolved production campaign."""
    required = {
        "target_campaign_id",
        "target_campaign_digest",
        "simulation_profile",
        "target_case_count",
        "regular_state_count",
        "regular_time_start_h",
        "time_horizon_h",
        "status",
        "mean_based_bytes",
        "median_based_bytes",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        message = "Pilot storage projection lacks its resolved production contract."
        raise ValueError(message)
    positive_counts = (value["target_case_count"], value["regular_state_count"])
    numeric_times = (value["regular_time_start_h"], value["time_horizon_h"])
    digest = value["target_campaign_digest"]
    if (
        not isinstance(value["target_campaign_id"], str)
        or not value["target_campaign_id"]
        or not isinstance(digest, str)
        or len(digest) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
        or value["simulation_profile"] != profiles.TRANSIENT_DRYING_PROFILE
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in positive_counts)
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in numeric_times)
        or float(value["time_horizon_h"]) <= float(value["regular_time_start_h"])
        or value["status"] not in {"available", "unavailable"}
    ):
        message = "Pilot storage projection has an invalid resolved production contract."
        raise ValueError(message)
    estimates = (value["mean_based_bytes"], value["median_based_bytes"])
    if value["status"] == "available":
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in estimates):
            message = "Available pilot storage projections require non-negative byte estimates."
            raise ValueError(message)
    elif any(item is not None for item in estimates):
        message = "Unavailable pilot storage projections cannot report byte estimates."
        raise ValueError(message)


def _regular_files(roots: Iterable[Path], *, excluded: set[Path] | None = None) -> list[Path]:
    """Return unique regular files below safe roots without following symlinks."""
    excluded_resolved = set() if excluded is None else {path.resolve() for path in excluded}
    files: dict[Path, Path] = {}
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir() or resolved_root.is_symlink():
            message = f"Pilot inventory root is missing or unsafe: {resolved_root}"
            raise FileNotFoundError(message)
        for path in sorted(resolved_root.rglob("*")):
            if path.is_symlink():
                message = f"Pilot inventory contains a symbolic link: {path}"
                raise ValueError(message)
            if path.is_file() and path.resolve() not in excluded_resolved:
                files[path.resolve()] = path.resolve()
    return sorted(files.values())


def _file_records(files: Iterable[Path], *, storage_root: Path) -> list[dict[str, Any]]:
    """Return stable relative path, size, and hash records."""
    records = []
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(storage_root):
            message = f"Pilot inventory file escapes storage: {resolved}"
            raise ValueError(message)
        records.append(
            {
                "relative_path": resolved.relative_to(storage_root).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": common.serialization.file_sha256(resolved),
            }
        )
    records.sort(key=lambda record: str(record["relative_path"]))
    return records


def _validate_owned_directory(path: Path, *, owner: Path, label: str) -> None:
    """Reject traversal, symlink, and non-directory states below one owned root."""
    if not owner.is_dir() or owner.is_symlink():
        message = f"{label} owner is missing or unsafe: {owner}"
        raise ValueError(message)
    try:
        relative = path.relative_to(owner)
    except ValueError as error:
        message = f"{label} escapes its owned root: {path}"
        raise ValueError(message) from error
    current = owner
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            message = f"{label} contains a symbolic-link component: {current}"
            raise ValueError(message)
    if not path.resolve().is_relative_to(owner.resolve()):
        message = f"{label} resolves outside its owned root: {path}"
        raise ValueError(message)
    if path.exists() and not path.is_dir():
        message = f"{label} is not a directory: {path}"
        raise ValueError(message)
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_symlink():
                message = f"{label} contains a symbolic link: {item}"
                raise ValueError(message)


def _copy_failure_evidence(
    *,
    source: Path,
    source_artifacts: Path,
    destination: Path,
    destination_owner: Path,
) -> dict[str, Any]:
    """Publish one validated private failure record and compact artifacts."""
    payload = _load_json(source, label="case failure evidence")
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    _validate_owned_directory(
        destination,
        owner=destination_owner,
        label="published pilot failure evidence directory",
    )
    destination.mkdir(parents=True, exist_ok=True)
    _validate_owned_directory(
        destination,
        owner=destination_owner,
        label="published pilot failure evidence directory",
    )
    failure_path = destination / "failure.json"
    if failure_path.exists():
        if not failure_path.is_file() or failure_path.is_symlink() or failure_path.read_text(encoding="utf-8") != serialized:
            message = f"Published pilot failure evidence conflicts: {failure_path}"
            raise FileExistsError(message)
    else:
        common.serialization.atomic_write_text(failure_path, serialized)
    declared = payload["retained_artifacts"]
    artifact_records: list[dict[str, Any]] = []
    for relative_value, identity in sorted(declared.items()):
        relative = Path(relative_value)
        source_path = (source_artifacts / relative).resolve()
        target = (destination / "artifacts" / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not source_path.is_relative_to(source_artifacts.resolve())
            or not target.is_relative_to(destination.resolve())
            or not source_path.is_file()
            or source_path.is_symlink()
            or source_path.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(source_path) != identity["sha256"]
        ):
            message = f"Private pilot failure artifact is invalid: {source_path}"
            raise ValueError(message)
        target.parent.mkdir(parents=True, exist_ok=True)
        _validate_owned_directory(
            destination,
            owner=destination_owner,
            label="published pilot failure evidence directory",
        )
        if target.exists():
            if (
                not target.is_file()
                or target.is_symlink()
                or target.stat().st_size != identity["size_bytes"]
                or common.serialization.file_sha256(target) != identity["sha256"]
            ):
                message = f"Published pilot failure artifact conflicts: {target}"
                raise FileExistsError(message)
        else:
            shutil.copy2(source_path, target)
        artifact_records.append(
            {
                "relative_path": target.relative_to(destination).as_posix(),
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )
    _validate_owned_directory(
        destination,
        owner=destination_owner,
        label="published pilot failure evidence directory",
    )
    actual_artifacts = {item.relative_to(destination).as_posix() for item in destination.rglob("*") if item.is_file() and item != failure_path}
    if actual_artifacts != {record["relative_path"] for record in artifact_records}:
        message = f"Published pilot failure artifact membership is invalid: {destination}"
        raise ValueError(message)
    return {
        "relative_path": failure_path.name,
        "sha256": common.serialization.file_sha256(failure_path),
        "size_bytes": failure_path.stat().st_size,
        "retained_artifacts": artifact_records,
        "retained_evidence_size_bytes": failure_path.stat().st_size + sum(int(record["size_bytes"]) for record in artifact_records),
        "error": payload["error"],
        "missing_or_invalid_artifacts": payload["missing_or_invalid_artifacts"],
        "scratch_cleanup": payload["scratch_cleanup"],
    }


def finalize_pilot_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish terminal pilot evidence after every case succeeds or fails durably."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    if campaign.campaign_purpose != config_service.PILOT_CAMPAIGN_PURPOSE:
        message = f"Campaign {run_id!r} is not a pilot-check campaign."
        raise ValueError(message)
    if not manifest["slurm_job_ids"]:
        message = f"Pilot campaign scheduler identity has not been recovered: {run_id}"
        raise RuntimeError(message)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    terminal_path = run_directory / "campaign_terminal.json"
    if terminal_path.exists():
        return terminal_path
    case_records: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    for batch in campaign.batches:
        raw_root = common.paths.resolve_generated_batch_dir(
            batch.batch_id,
            stage="raw",
            storage_root=storage,
        )
        processed_root = common.paths.resolve_generated_batch_dir(
            batch.batch_id,
            stage="processed",
            storage_root=storage,
        )
        meta_root = runtime_service.batch_meta_directory(batch, storage_root=storage)
        for directory in (raw_root, processed_root, meta_root):
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                message = f"Pilot batch evidence directory is unsafe: {directory}"
                raise ValueError(message)
        completed = 0
        failed = 0
        batch_cases: list[dict[str, Any]] = []
        for case_index in batch.case_indices:
            case_id = batch.case_id(case_index)
            assignment = batch.case_assignment(case_index)
            if runtime_service.completed_case_is_valid(
                batch,
                case_index,
                storage_root=storage,
            ):
                provenance = runtime_service.validate_completed_case(
                    batch,
                    case_index,
                    storage_root=storage,
                )
                processed = runtime_service.processed_case_directory(
                    batch,
                    case_index,
                    storage_root=storage,
                )
                raw = runtime_service.raw_case_directory(
                    batch,
                    case_index,
                    storage_root=storage,
                )
                record = {
                    "batch_name": batch.batch_name,
                    "batch_id": batch.batch_id,
                    "case_id": case_id,
                    "case_index": case_index,
                    "material": batch.material_family,
                    "material_role": batch.material_role,
                    "case_kind": assignment["pilot_case_kind"],
                    "terminal_state": "success",
                    "simulation_case_id": provenance["simulation_case_id"],
                    "processed_directory": processed.relative_to(storage).as_posix(),
                    "raw_directory": raw.relative_to(storage).as_posix(),
                    "failure_evidence": None,
                }
                completed += 1
            elif runtime_service.case_failure_is_recorded(
                batch,
                case_index,
                storage_root=storage,
            ):
                source = runtime_service.case_failure_path(
                    batch,
                    case_index,
                    storage_root=storage,
                )
                payload = _load_json(source, label="case failure evidence")
                cleanup = payload["scratch_cleanup"]
                if cleanup["status"] not in {"complete", "not_created"}:
                    message = f"Pilot failure scratch is not safely terminal for {batch.batch_name}/{case_id}."
                    raise RuntimeError(message)
                destination = meta_root / "pilot_failure_evidence" / case_id
                evidence = _copy_failure_evidence(
                    source=source,
                    source_artifacts=runtime_service.case_failure_artifacts_directory(
                        batch,
                        case_index,
                        storage_root=storage,
                    ),
                    destination=destination,
                    destination_owner=meta_root,
                )
                evidence["relative_path"] = (destination / "failure.json").relative_to(storage).as_posix()
                evidence["evidence_directory"] = destination.relative_to(storage).as_posix()
                record = {
                    "batch_name": batch.batch_name,
                    "batch_id": batch.batch_id,
                    "case_id": case_id,
                    "case_index": case_index,
                    "material": batch.material_family,
                    "material_role": batch.material_role,
                    "case_kind": assignment["pilot_case_kind"],
                    "terminal_state": "failure",
                    "simulation_case_id": None,
                    "processed_directory": None,
                    "raw_directory": None,
                    "failure_evidence": evidence,
                }
                failed += 1
            else:
                message = f"Pilot case has no terminal success or failure evidence: {batch.batch_name}/{case_id}."
                raise RuntimeError(message)
            batch_cases.append(record)
            case_records.append(record)

        batch_terminal = {
            "schema_kind": "generation_pilot_batch_terminal",
            "schema_version": PILOT_SCHEMA_VERSION,
            "batch_name": batch.batch_name,
            "batch_id": batch.batch_id,
            "batch_identity": batch.batch_identity,
            "material": batch.material_family,
            "material_role": batch.material_role,
            "planned": len(batch.case_indices),
            "completed": completed,
            "failed": failed,
            "cases": batch_cases,
        }
        common.serialization.atomic_write_json(
            meta_root / "pilot_batch_terminal.json",
            batch_terminal,
        )
        batch_records.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_count": len(batch.case_indices),
                "completed": completed,
                "failed": failed,
                "terminal_sha256": common.serialization.file_sha256(meta_root / "pilot_batch_terminal.json"),
            }
        )

    terminal = {
        "schema_kind": PILOT_TERMINAL_SCHEMA_KIND,
        "schema_version": PILOT_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_config": manifest["campaign_config"],
        "campaign_purpose": campaign.campaign_purpose,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "git_commit": manifest["git_commit"],
        "slurm_job_ids": manifest["slurm_job_ids"],
        "scheduler_job_name": manifest["scheduler_job_name"],
        "scheduler_log_directory": manifest["scheduler_log_directory"],
        "cases_per_material": len(campaign.batches[0].case_indices),
        "materials": list(campaign.material_inventory),
        "batches": batch_records,
        "cases": case_records,
        "dataset_packages": [],
        "terminal_counts": {
            "planned": len(case_records),
            "successful": sum(record["terminal_state"] == "success" for record in case_records),
            "failed": sum(record["terminal_state"] == "failure" for record in case_records),
        },
    }
    serialized = json.dumps(terminal, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    common.serialization.atomic_write_text(terminal_path, serialized)
    return terminal_path


def validate_pilot_terminal(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Revalidate one terminal pilot and every success/failure evidence path."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / "campaign_terminal.json"
    if not path.exists():
        path = finalize_pilot_campaign(run_id, storage_root=storage)
    terminal = _load_json(path, label="pilot terminal campaign")
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    materials = terminal.get("materials")
    if (
        terminal.get("schema_kind") != PILOT_TERMINAL_SCHEMA_KIND
        or terminal.get("schema_version") != PILOT_SCHEMA_VERSION
        or terminal.get("campaign_run_id") != run_id
        or terminal.get("campaign_id") != campaign.campaign_id
        or terminal.get("campaign_digest") != campaign.campaign_digest
        or terminal.get("campaign_purpose") != config_service.PILOT_CAMPAIGN_PURPOSE
        or materials != list(campaign.material_inventory)
        or terminal.get("dataset_packages") != []
    ):
        message = f"Pilot terminal campaign is malformed: {path}"
        raise ValueError(message)
    for record in terminal["cases"]:
        if (
            record.get("material") not in terminal["materials"]
            or record.get("material_role") not in config_service.MATERIAL_ROLES
            or record.get("case_kind") not in config_service.PILOT_CASE_KINDS
            or not isinstance(record.get("case_index"), int)
            or isinstance(record.get("case_index"), bool)
        ):
            message = f"Pilot terminal case provenance is malformed: {path}"
            raise ValueError(message)
        if record["terminal_state"] == "success":
            directory = storage / record["processed_directory"]
            storage_service.validate_case_hdf5(
                directory / "case.h5",
                expected_profile="transient_drying",
            )
        else:
            evidence = record["failure_evidence"]
            evidence_path = storage / evidence["relative_path"]
            artifacts = evidence["retained_artifacts"]
            if (
                common.serialization.file_sha256(evidence_path) != evidence["sha256"]
                or evidence_path.stat().st_size != evidence["size_bytes"]
                or evidence["retained_evidence_size_bytes"] != evidence["size_bytes"] + sum(int(item["size_bytes"]) for item in artifacts)
            ):
                message = f"Pilot failure evidence changed: {evidence_path}"
                raise ValueError(message)
            evidence_directory = storage / evidence["evidence_directory"]
            for artifact in artifacts:
                artifact_path = evidence_directory / artifact["relative_path"]
                if (
                    not artifact_path.is_file()
                    or artifact_path.is_symlink()
                    or artifact_path.stat().st_size != artifact["size_bytes"]
                    or common.serialization.file_sha256(artifact_path) != artifact["sha256"]
                ):
                    message = f"Pilot failure artifact changed: {artifact_path}"
                    raise ValueError(message)
    return terminal


def pilot_transfer_plan(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return public directories for successful cases and published failure evidence."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    terminal = validate_pilot_terminal(run_id, storage_root=storage)
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)

    def relative(directory: Path) -> str:
        resolved = directory.resolve()
        if not resolved.is_dir() or resolved.is_symlink() or not resolved.is_relative_to(storage):
            message = f"Pilot transfer directory is missing or unsafe: {resolved}"
            raise FileNotFoundError(message)
        return resolved.relative_to(storage).as_posix()

    return {
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "git_commit": manifest["git_commit"],
        "campaign_config": manifest["campaign_config"],
        "campaign_directory": relative(campaign_evidence.campaign_run_directory(run_id, storage_root=storage)),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_count": len(batch.case_indices),
                "meta_directory": relative(runtime_service.batch_meta_directory(batch, storage_root=storage)),
                "raw_directory": relative(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_id,
                        stage="raw",
                        storage_root=storage,
                    )
                ),
                "processed_directory": relative(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_id,
                        stage="processed",
                        storage_root=storage,
                    )
                ),
            }
            for batch in campaign.batches
        ],
        "terminal_counts": terminal["terminal_counts"],
    }


def _plan_roots(plan: Mapping[str, Any], *, storage: Path) -> list[Path]:
    """Return unique plan directories without path traversal."""
    relative_values = [plan["campaign_directory"]]
    relative_values.extend(
        directory
        for batch in plan["batches"]
        for directory in (
            batch["meta_directory"],
            batch["raw_directory"],
            batch["processed_directory"],
        )
    )
    roots: dict[Path, Path] = {}
    for value in relative_values:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            message = f"Pilot transfer plan contains an unsafe path: {value!r}"
            raise ValueError(message)
        resolved = (storage / relative).resolve()
        if not resolved.is_relative_to(storage):
            message = f"Pilot transfer plan escapes storage: {value!r}"
            raise ValueError(message)
        roots[resolved] = resolved
    return sorted(roots)


def _source_case_inventory(
    terminal: Mapping[str, Any],
    *,
    storage: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return exact bytes and file counts for each successful/failure case."""
    case_bytes: dict[str, int] = {}
    case_counts: dict[str, int] = {}
    for record in terminal["cases"]:
        key = f"{record['material']}:{record['case_id']}"
        if record["terminal_state"] == "success":
            roots = [storage / record["processed_directory"], storage / record["raw_directory"]]
        else:
            roots = [(storage / record["failure_evidence"]["relative_path"]).parent]
        files = _regular_files(roots)
        case_bytes[key] = sum(path.stat().st_size for path in files)
        case_counts[key] = len(files)
    return case_bytes, case_counts


def _write_fixed_point_source_inventory(
    path: Path,
    payload: dict[str, Any],
    *,
    base_bytes: int,
    base_count: int,
    base_other_bytes: int,
) -> dict[str, Any]:
    """Include the inventory receipt itself in exact byte and file totals."""
    file_size = 0
    for _ in range(_MAX_FIXED_POINT_ITERATIONS):
        payload["cpu_source_bytes_before_cleanup"] = base_bytes + file_size
        payload["cpu_source_file_count_before_cleanup"] = base_count + 1
        payload["cpu_other_bytes"] = base_other_bytes + file_size
        common.serialization.atomic_write_json(path, payload)
        updated = path.stat().st_size
        if updated == file_size:
            return payload
        file_size = updated
    message = f"Pilot source inventory size did not reach a fixed point: {path}"
    raise RuntimeError(message)


def record_cpu_source_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Record exact CPU source bytes and files before transfer or cleanup."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    terminal = validate_pilot_terminal(run_id, storage_root=storage)
    plan = pilot_transfer_plan(run_id, storage_root=storage)
    path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / PILOT_SOURCE_INVENTORY_FILENAME
    roots = _plan_roots(plan, storage=storage)
    files = _regular_files(roots, excluded={path})
    records = _file_records(files, storage_root=storage)
    case_bytes, case_counts = _source_case_inventory(terminal, storage=storage)
    log_bytes = sum(path.stat().st_size for path in files if path.suffix in {".log", ".out", ".err"})
    export_bytes = sum(path.stat().st_size for path in files if path.suffix == ".csv" or "exports" in path.parts)
    base_bytes = sum(int(record["size_bytes"]) for record in records)
    base_other = base_bytes - log_bytes - export_bytes
    payload = {
        "schema_kind": "generation_pilot_cpu_source_inventory",
        "schema_version": PILOT_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "recorded_at": _utc_now(),
        "inventory_scope": "public_pilot_transfer_plan_plus_this_receipt",
        "inventory_records_exclude_self_hash": True,
        "inventory_sha256": common.serialization.canonical_json_sha256(records),
        "files": records,
        "cpu_source_bytes_before_cleanup": 0,
        "cpu_source_file_count_before_cleanup": 0,
        "cpu_case_bytes_before_cleanup": case_bytes,
        "cpu_case_file_counts": case_counts,
        "cpu_logs_bytes": log_bytes,
        "cpu_exports_bytes": export_bytes,
        "cpu_other_bytes": 0,
        "cleanup_eligible_publication_directories": [
            directory
            for batch in plan["batches"]
            for directory in (
                batch["meta_directory"],
                batch["raw_directory"],
                batch["processed_directory"],
            )
        ],
    }
    result = _write_fixed_point_source_inventory(
        path,
        payload,
        base_bytes=base_bytes,
        base_count=len(records),
        base_other_bytes=base_other,
    )
    actual = _regular_files(roots)
    if (
        sum(item.stat().st_size for item in actual) != result["cpu_source_bytes_before_cleanup"]
        or len(actual) != result["cpu_source_file_count_before_cleanup"]
    ):
        message = "Pilot CPU source inventory does not match the exact post-receipt tree."
        raise RuntimeError(message)
    return result


def record_transfer_staging_inventory(
    run_id: str,
    *,
    staging_root: Path | str,
) -> dict[str, Any]:
    """Record exact transfer-staging bytes before publication and later cleanup."""
    staging = workspace_service.validate_transfer_staging(staging_root, run_id=run_id)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=staging)
    path = run_directory / PILOT_STAGING_INVENTORY_FILENAME
    files = _regular_files([staging], excluded={path})
    base_bytes = sum(item.stat().st_size for item in files)
    file_size = 0
    payload = {
        "schema_kind": "generation_pilot_transfer_staging_inventory",
        "schema_version": PILOT_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "recorded_at": _utc_now(),
        "transfer_staging_path": str(staging),
        "transfer_staging_bytes_before_cleanup": 0,
        "transfer_staging_file_count": 0,
    }
    for _ in range(_MAX_FIXED_POINT_ITERATIONS):
        payload["transfer_staging_bytes_before_cleanup"] = base_bytes + file_size
        payload["transfer_staging_file_count"] = len(files) + 1
        common.serialization.atomic_write_json(path, payload)
        updated = path.stat().st_size
        if updated == file_size:
            break
        file_size = updated
    else:
        message = f"Pilot staging inventory size did not reach a fixed point: {path}"
        raise RuntimeError(message)
    actual = _regular_files([staging])
    if (
        sum(item.stat().st_size for item in actual) != payload["transfer_staging_bytes_before_cleanup"]
        or len(actual) != payload["transfer_staging_file_count"]
    ):
        message = "Pilot staging inventory does not match the exact staging tree."
        raise RuntimeError(message)
    return payload


def validate_cpu_source_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the transferred exact pre-cleanup CPU source inventory."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / PILOT_SOURCE_INVENTORY_FILENAME
    payload = _load_json(path, label="transferred pilot CPU source inventory")
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "recorded_at",
        "inventory_scope",
        "inventory_records_exclude_self_hash",
        "inventory_sha256",
        "files",
        "cpu_source_bytes_before_cleanup",
        "cpu_source_file_count_before_cleanup",
        "cpu_case_bytes_before_cleanup",
        "cpu_case_file_counts",
        "cpu_logs_bytes",
        "cpu_exports_bytes",
        "cpu_other_bytes",
        "cleanup_eligible_publication_directories",
    }
    files = payload.get("files")
    if (
        set(payload) != required
        or payload.get("schema_kind") != "generation_pilot_cpu_source_inventory"
        or payload.get("schema_version") != PILOT_SCHEMA_VERSION
        or payload.get("campaign_run_id") != run_id
        or payload.get("inventory_scope") != "public_pilot_transfer_plan_plus_this_receipt"
        or payload.get("inventory_records_exclude_self_hash") is not True
        or not isinstance(files, list)
        or common.serialization.canonical_json_sha256(files) != payload.get("inventory_sha256")
    ):
        message = f"Pilot CPU source inventory is malformed: {path}"
        raise ValueError(message)
    if any(
        not isinstance(record, dict)
        or set(record) != {"relative_path", "size_bytes", "sha256"}
        or not isinstance(record["relative_path"], str)
        or not isinstance(record["size_bytes"], int)
        or isinstance(record["size_bytes"], bool)
        or record["size_bytes"] < 0
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != _SHA256_LENGTH
        for record in files
    ):
        message = f"Pilot CPU source inventory file records are malformed: {path}"
        raise ValueError(message)
    recorded_bytes = sum(int(record["size_bytes"]) for record in files) + path.stat().st_size
    recorded_count = len(files) + 1
    classified_bytes = int(payload["cpu_logs_bytes"]) + int(payload["cpu_exports_bytes"]) + int(payload["cpu_other_bytes"])
    if (
        payload.get("cpu_source_bytes_before_cleanup") != recorded_bytes
        or payload.get("cpu_source_file_count_before_cleanup") != recorded_count
        or classified_bytes != recorded_bytes
        or not isinstance(payload.get("cpu_case_bytes_before_cleanup"), dict)
        or not isinstance(payload.get("cpu_case_file_counts"), dict)
        or not isinstance(payload.get("cleanup_eligible_publication_directories"), list)
    ):
        message = f"Pilot CPU source inventory totals are inconsistent: {path}"
        raise ValueError(message)
    return payload


def validate_transfer_staging_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    require_staging_present: bool = False,
) -> dict[str, Any]:
    """Validate recorded staging size and, while retained, its exact live tree."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage) / PILOT_STAGING_INVENTORY_FILENAME
    payload = _load_json(path, label="pilot transfer staging inventory")
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "recorded_at",
        "transfer_staging_path",
        "transfer_staging_bytes_before_cleanup",
        "transfer_staging_file_count",
    }
    staging_value = payload.get("transfer_staging_path")
    if (
        set(payload) != required
        or payload.get("schema_kind") != "generation_pilot_transfer_staging_inventory"
        or payload.get("schema_version") != PILOT_SCHEMA_VERSION
        or payload.get("campaign_run_id") != run_id
        or not isinstance(staging_value, str)
        or not Path(staging_value).is_absolute()
        or not isinstance(payload.get("transfer_staging_bytes_before_cleanup"), int)
        or isinstance(payload.get("transfer_staging_bytes_before_cleanup"), bool)
        or payload["transfer_staging_bytes_before_cleanup"] < 0
        or not isinstance(payload.get("transfer_staging_file_count"), int)
        or isinstance(payload.get("transfer_staging_file_count"), bool)
        or payload["transfer_staging_file_count"] < 1
    ):
        message = f"Pilot transfer staging inventory is malformed: {path}"
        raise ValueError(message)
    staging = Path(staging_value)
    if staging.exists():
        validated = workspace_service.validate_transfer_staging(staging, run_id=run_id)
        files = _regular_files([validated])
        if (
            sum(item.stat().st_size for item in files) != payload["transfer_staging_bytes_before_cleanup"]
            or len(files) != payload["transfer_staging_file_count"]
        ):
            message = f"Pilot transfer staging changed after inventory: {staging}"
            raise ValueError(message)
    elif require_staging_present:
        message = f"Pilot transfer staging is missing before cleanup authorization: {staging}"
        raise FileNotFoundError(message)
    return payload


def _failure_case_result(record: Mapping[str, Any], *, storage: Path) -> dict[str, Any]:
    """Return one canonical failed-case result from retained evidence."""
    evidence = record["failure_evidence"]
    evidence_path = storage / evidence["relative_path"]
    payload = _load_json(evidence_path, label="published pilot failure evidence")
    failed_stage = str(payload["failure_stage"])
    try:
        result_class = _FAILURE_STAGE_RESULT_CLASS[failed_stage]
    except KeyError as error:
        message = f"Published pilot failure has an unsupported stage: {failed_stage!r}"
        raise ValueError(message) from error
    solver_status = "not_started" if failed_stage == "input" else "failed" if failed_stage == "solver" else "success"
    return {
        "case_id": record["case_id"],
        "case_index": record["case_index"],
        "simulation_case_id": None,
        "material": record["material"],
        "material_role": record["material_role"],
        "case_kind": record["case_kind"],
        "solver_status": solver_status,
        "result_class": result_class,
        "target_reached": None,
        "drying_time_h": None,
        "drying_time_days": None,
        "last_valid_time_h": None,
        "stop_reason": "runtime_failure",
        "failed_stage": failed_stage,
        "warning_count": 0,
        "final_X_wb_bulk": None,
        "final_f_wet_dm": None,
        "hard_contract": {
            "solver_completed": solver_status == "success",
            "failure_type": payload["error"]["type"],
            "failure_message": payload["error"]["message"],
            "missing_or_invalid_artifacts": payload["missing_or_invalid_artifacts"],
        },
        "duration": {
            "result": result_class,
            "target_reached": None,
            "drying_time_h": None,
            "drying_time_days": None,
            "last_valid_time_h": None,
            "stop_reason": "runtime_failure",
        },
        "physical_bound": {"status": "unavailable_due_to_runtime_failure"},
        "conservation_diagnostic": {"status": "unavailable_due_to_runtime_failure"},
        "trend_diagnostic": {"status": "unavailable_due_to_runtime_failure"},
        "extrema_diagnostic": {"status": "unavailable_due_to_runtime_failure"},
        "schedule_input_sanity": {"status": "unavailable_due_to_runtime_failure"},
        "applicability_domain_diagnostic": {"status": "unavailable_due_to_runtime_failure"},
        "numerical_runtime": {"status": "failed"},
        "storage": {
            "canonical_hdf5_bytes": None,
            "retained_failure_evidence_bytes": evidence["retained_evidence_size_bytes"],
        },
        "warnings": [],
        "retained_evidence_path": str(storage / evidence["evidence_directory"]),
    }


def _invalid_success_case_result(
    record: Mapping[str, Any],
    error: Exception,
    *,
    storage: Path,
) -> dict[str, Any]:
    """Preserve a solver-complete case whose canonical analysis is invalid."""
    retained = storage / record["processed_directory"]
    return {
        "case_id": record["case_id"],
        "case_index": record["case_index"],
        "simulation_case_id": record["simulation_case_id"],
        "material": record["material"],
        "material_role": record["material_role"],
        "case_kind": record["case_kind"],
        "solver_status": "success",
        "result_class": "INVALID_RESULT",
        "target_reached": None,
        "drying_time_h": None,
        "drying_time_days": None,
        "last_valid_time_h": None,
        "stop_reason": "analysis_failure",
        "failed_stage": "invalid_result",
        "warning_count": 0,
        "final_X_wb_bulk": None,
        "final_f_wet_dm": None,
        "hard_contract": {
            "solver_completed": True,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "missing_or_invalid_artifacts": [],
        },
        "duration": {
            "result": "INVALID_RESULT",
            "target_reached": None,
            "drying_time_h": None,
            "drying_time_days": None,
            "last_valid_time_h": None,
            "stop_reason": "analysis_failure",
        },
        "physical_bound": {"status": "unavailable_due_to_invalid_result"},
        "conservation_diagnostic": {"status": "unavailable_due_to_invalid_result"},
        "trend_diagnostic": {"status": "unavailable_due_to_invalid_result"},
        "extrema_diagnostic": {"status": "unavailable_due_to_invalid_result"},
        "schedule_input_sanity": {"status": "unavailable_due_to_invalid_result"},
        "applicability_domain_diagnostic": {"status": "unavailable_due_to_invalid_result"},
        "numerical_runtime": {"status": "analysis_failed"},
        "storage": {"canonical_hdf5_bytes": None},
        "warnings": [],
        "retained_evidence_path": str(retained),
    }


def _per_material(
    cases: list[dict[str, Any]],
    materials: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate one directly comparable row per configured pilot material."""
    rows: list[dict[str, Any]] = []
    for material in materials:
        selected = [record for record in cases if record["material"] == material]
        nominal = next(record for record in selected if record["case_kind"] == "nominal_reference")
        successful = [
            record for record in selected if record["solver_status"] == "success" and record["storage"].get("canonical_hdf5_bytes") is not None
        ]
        durations = [float(record["drying_time_h"]) for record in successful if record["drying_time_h"] is not None]
        hdf5_sizes = [int(record["storage"]["canonical_hdf5_bytes"]) for record in successful]
        native = [float(record["conservation_diagnostic"]["comsol_mt_mass_balance"]["max_abs"]) for record in successful]
        independent = [
            float(record["conservation_diagnostic"]["independent_water_balances"]["total_water"]["max_abs_residual_kg"])
            for record in successful
            if record["conservation_diagnostic"]["independent_water_balances"]["status"] == "available"
        ]
        rows.append(
            {
                "material": material,
                "material_role": selected[0]["material_role"],
                "nominal_result_class": nominal["result_class"],
                "nominal_duration_result": nominal["duration"]["result"],
                "nominal_stop_consistent": nominal["duration"].get("stop_consistent"),
                "nominal_drying_duration_h": nominal["drying_time_h"],
                "successful_duration_median_h": statistics.median(durations) if durations else None,
                "successful_duration_min_h": min(durations) if durations else None,
                "successful_duration_max_h": max(durations) if durations else None,
                "target_reached_count": sum(record["target_reached"] is True for record in successful),
                "runtime_failure_count": sum(record["result_class"] in _RUNTIME_FAILURE_RESULT_CLASSES for record in selected),
                "physical_contract_violation_count": sum(record["result_class"] == "PHYSICAL_CONTRACT_VIOLATION" for record in selected),
                "worst_mt_mass_balance_max_abs": max(native) if native else None,
                "worst_independent_total_water_residual_kg": max(independent) if independent else None,
                "median_hdf5_size_bytes": statistics.median(hdf5_sizes) if hdf5_sizes else None,
            }
        )
    return rows


def _problems(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only affected cases with exact values and retained evidence paths."""
    problems: list[dict[str, Any]] = []
    for record in cases:
        common_fields = {
            "material": record["material"],
            "material_role": record["material_role"],
            "case_id": record["case_id"],
            "case_kind": record["case_kind"],
            "result_class": record["result_class"],
            "retained_gpu_evidence_path": record["retained_evidence_path"],
        }
        if record["result_class"] in _RUNTIME_FAILURE_RESULT_CLASSES:
            problems.append(
                {
                    **common_fields,
                    "problem_category": record["failed_stage"],
                    "explanation": record["hard_contract"]["failure_message"],
                    "actual_value": record["hard_contract"]["failure_type"],
                    "reference_value": "successful canonical case lifecycle",
                }
            )
            continue
        if record["result_class"] == "INVALID_RESULT":
            problems.append(
                {
                    **common_fields,
                    "problem_category": "invalid_result",
                    "explanation": record["hard_contract"]["failure_message"],
                    "actual_value": record["hard_contract"]["failure_type"],
                    "reference_value": "valid canonical pilot analysis",
                }
            )
            continue
        duration = record["duration"]
        if (
            duration["result"]
            in {
                "TOO_FAST",
                "NOT_DRY_WITHIN_HORIZON",
                "RIGHT_CENSORED",
                "INVALID_RESULT",
            }
            or "natural_duration_outside_nominal_window" in record["warnings"]
        ):
            problems.append(
                {
                    **common_fields,
                    "problem_category": "duration",
                    "explanation": duration["result"],
                    "actual_value": duration["drying_time_h"],
                    "reference_value": duration["adequacy_window_h"],
                }
            )
        if not duration["stop_consistent"]:
            problems.append(
                {
                    **common_fields,
                    "problem_category": "hard_contract",
                    "explanation": "target stop disagrees with configured wet-fraction threshold",
                    "actual_value": duration["final_f_wet_dm"],
                    "reference_value": duration["configured_threshold"],
                }
            )
        schedule = record["schedule_input_sanity"]
        if schedule["status"] != "pass":
            problems.append(
                {
                    **common_fields,
                    "problem_category": "physical_bound",
                    "explanation": "heater-only schedule feasibility contract was violated",
                    "actual_value": schedule["checks"],
                    "reference_value": {
                        "T_in_bc": ">=T_amb",
                        "omega_in_bc": ">0",
                        "phi_source_air": "0<phi<=1",
                        "phi_in_bc": {
                            "minimum": schedule["configured_phi_operational_min"],
                            "maximum": schedule["configured_phi_operational_max"],
                        },
                    },
                }
            )
        problems.extend(
            {
                **common_fields,
                "problem_category": "physical_bound",
                "explanation": f"{violation['quantity']} violates {violation['rule']}",
                "actual_value": {
                    "min": violation["observed_min"],
                    "max": violation["observed_max"],
                    "count": violation["violation_count"],
                },
                "reference_value": violation["rule"],
            }
            for violation in record["physical_bound"]["violations"]
        )
        for name, trend in record["trend_diagnostic"].items():
            if trend["positive_step_count"]:
                problems.append(
                    {
                        **common_fields,
                        "problem_category": "trend_diagnostic",
                        "explanation": f"positive steps observed in {name}",
                        "actual_value": trend,
                        "reference_value": "no-rewetting model; no automatic numerical tolerance",
                    }
                )
        problems.extend(
            {
                **common_fields,
                "problem_category": "applicability_domain_diagnostic",
                "explanation": f"{applicability['record']} uses {applicability['overlap']}",
                "actual_value": applicability["evidence"],
                "reference_value": applicability["applicability"],
            }
            for applicability in record["applicability_domain_diagnostic"]["records"]
            if applicability["overlap"]
            in {
                "material_transfer",
                "product_form_transfer",
                "regime_transfer",
                "engineering_extension",
            }
        )
    return problems


def _summary_csv(receipt: Mapping[str, Any]) -> str:
    """Render the compact per-case CSV solely from canonical receipt data."""
    columns: tuple[str, ...] = (
        "material",
        "material_role",
        "case_kind",
        "case_index",
        "case_id",
        "solver_status",
        "result_class",
        "target_reached",
        "drying_time_h",
        "drying_time_days",
        "last_valid_time_h",
        "stop_reason",
        "failed_stage",
        "warning_count",
        "final_X_wb_bulk",
        "final_f_wet_dm",
        "mt_mass_balance_max_abs",
        "canonical_hdf5_bytes",
    )
    stream = io.StringIO(newline="")
    writer: csv.DictWriter[str] = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in receipt["cases"]:
        native = record["conservation_diagnostic"].get("comsol_mt_mass_balance", {})
        row = {name: record[name] for name in columns if name in _REQUIRED_CASE_RESULT_FIELDS}
        row["mt_mass_balance_max_abs"] = native.get("max_abs")
        row["canonical_hdf5_bytes"] = record["storage"].get("canonical_hdf5_bytes")
        writer.writerow(row)
    return stream.getvalue()


def _summary_markdown(receipt: Mapping[str, Any]) -> str:
    """Render a concise Markdown view solely from canonical receipt data."""
    lines = [
        "# VP2 transient pilot check",
        "",
        f"Pilot: `{receipt['pilot_check_id']}`",
        "",
        "| Material | Role | Kind | Index | Case | Solver | Result class | Dry h | Target | Stage | Warnings | MB max abs | HDF5 bytes |",
        "|---|---|---|---:|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for record in receipt["cases"]:
        native = record["conservation_diagnostic"].get("comsol_mt_mass_balance", {})
        lines.append(
            (
                "| {material} | {role} | {kind} | {index} | {case} | {solver} | "
                "{result} | {dry} | {target} | {stage} | {warnings} | {balance} | {size} |"
            ).format(
                material=record["material"],
                role=record["material_role"],
                kind=record["case_kind"],
                index=record["case_index"],
                case=record["case_id"],
                solver=record["solver_status"],
                result=record["result_class"],
                dry=record["drying_time_h"],
                target=record["target_reached"],
                stage=record["failed_stage"],
                warnings=record["warning_count"],
                balance=native.get("max_abs"),
                size=record["storage"].get("canonical_hdf5_bytes"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-material summary",
            "",
            (
                "| Material | Role | Nominal class | Duration result | Nominal h | Median h | Min h | Max h | Targets | "
                "Runtime failures | Physical violations | Worst native MB | Worst total residual kg | "
                "Median HDF5 bytes |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            "| {material} | {material_role} | {nominal_result_class} | {nominal_duration_result} | {nominal_drying_duration_h} | "
            "{successful_duration_median_h} | {successful_duration_min_h} | "
            "{successful_duration_max_h} | {target_reached_count} | "
            "{runtime_failure_count} | {physical_contract_violation_count} | "
            "{worst_mt_mass_balance_max_abs} | {worst_independent_total_water_residual_kg} | "
            "{median_hdf5_size_bytes} |"
        ).format(**row)
        for row in receipt["per_material"]
    )
    lines.extend(
        [
            "",
            "## Problems",
            "",
        ]
    )
    if receipt["problems"]:
        lines.extend(
            (
                f"- {problem['material']} {problem['case_id']} ({problem['case_kind']}): "
                f"{problem['problem_category']} — {problem['explanation']}; "
                f"actual={problem['actual_value']}; reference={problem['reference_value']}; "
                f"evidence={problem['retained_gpu_evidence_path']}"
            )
            for problem in receipt["problems"]
        )
    else:
        lines.append("No affected cases.")
    lines.extend(
        [
            "",
            "## Storage and cleanup",
            "",
            f"- CPU source before cleanup: {receipt['pre_cleanup_cpu_inventory']['cpu_source_bytes_before_cleanup']} bytes",
            f"- CPU exports before cleanup: {receipt['pre_cleanup_cpu_inventory']['cpu_exports_bytes']} bytes",
            f"- CPU logs before cleanup: {receipt['pre_cleanup_cpu_inventory']['cpu_logs_bytes']} bytes",
            f"- Transfer staging before cleanup: {receipt['transfer_staging_inventory']['transfer_staging_bytes_before_cleanup']} bytes",
            f"- Current permanent GPU pilot: {receipt['post_transfer_gpu_inventory']['current_pilot_gpu_permanent_bytes']} bytes",
            f"- CPU cleanup: {receipt['cleanup']['cpu_source']['status']} ({receipt['cleanup']['cpu_source']['bytes_reclaimed']} bytes reclaimed)",
            (
                f"- Staging cleanup: {receipt['cleanup']['transfer_staging']['status']} "
                f"({receipt['cleanup']['transfer_staging']['bytes_reclaimed']} bytes reclaimed)"
            ),
            "",
            f"## Projection for {receipt['production_storage_projection']['target_case_count']} transient cases",
            "",
            f"- Target campaign: {receipt['production_storage_projection']['target_campaign_id']}",
            f"- Configured regular states per full-horizon case: {receipt['production_storage_projection']['regular_state_count']}",
            f"- Configured time horizon: {receipt['production_storage_projection']['time_horizon_h']} h",
            f"- Basis: {receipt['production_storage_projection'].get('basis')}",
            f"- Observed mean-based: {receipt['production_storage_projection'].get('mean_based_bytes')}",
            f"- Observed median-based: {receipt['production_storage_projection'].get('median_based_bytes')}",
            f"- Configured-horizon projection: {receipt['production_storage_projection'].get('full_horizon_projection')}",
            "",
            "All conservation residuals are reported without an invented acceptance tolerance.",
            "",
        ]
    )
    return "\n".join(lines)


def _gpu_inventory(
    run_id: str,
    *,
    storage: Path,
) -> dict[str, Any]:
    """Measure permanent GPU generation and pilot metadata separately."""
    plan = pilot_transfer_plan(run_id, storage_root=storage)
    generation_files = _regular_files(_plan_roots(plan, storage=storage))
    pilot_directory = pilot_check_directory(run_id, storage_root=storage)
    pilot_files = _regular_files([pilot_directory]) if pilot_directory.is_dir() else []
    generation_bytes = sum(path.stat().st_size for path in generation_files)
    hdf5_bytes = sum(path.stat().st_size for path in generation_files if path.name == "case.h5")
    meta_roots = {
        (storage / plan["campaign_directory"]).resolve(),
        *{(storage / batch["meta_directory"]).resolve() for batch in plan["batches"]},
    }
    meta_files = _regular_files(meta_roots)
    meta_bytes = sum(path.stat().st_size for path in meta_files)
    log_bytes = sum(path.stat().st_size for path in generation_files if path.suffix in {".log", ".out", ".err"})
    failure_bytes = sum(path.stat().st_size for path in generation_files if "pilot_failure_evidence" in path.parts)
    pilot_meta_bytes = sum(path.stat().st_size for path in pilot_files)
    return {
        "gpu_generation_bytes": generation_bytes,
        "gpu_generation_hdf5_bytes": hdf5_bytes,
        "gpu_generation_meta_bytes": meta_bytes,
        "gpu_pilot_logs_bytes": log_bytes,
        "gpu_dataset_incremental_bytes_if_any": 0,
        "retained_failure_evidence_bytes": failure_bytes,
        "pilot_receipt_and_summary_bytes": pilot_meta_bytes,
        "current_pilot_gpu_permanent_bytes": generation_bytes + pilot_meta_bytes,
    }


def _write_receipt_and_views(
    run_id: str,
    receipt: dict[str, Any],
    *,
    storage: Path,
    write_pre_cleanup_snapshot: bool,
) -> dict[str, Any]:
    """Write canonical JSON then converge exact GPU inventory and derived views."""
    directory = pilot_check_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / PILOT_RECEIPT_FILENAME
    csv_path = directory / PILOT_SUMMARY_CSV
    markdown_path = directory / PILOT_SUMMARY_MARKDOWN
    snapshot_path = directory / PILOT_PRE_CLEANUP_FILENAME
    for _ in range(_MAX_FIXED_POINT_ITERATIONS):
        common.serialization.atomic_write_json(receipt_path, receipt)
        common.serialization.atomic_write_text(csv_path, _summary_csv(receipt))
        common.serialization.atomic_write_text(markdown_path, _summary_markdown(receipt))
        if write_pre_cleanup_snapshot:
            common.serialization.atomic_write_json(snapshot_path, receipt)
        measured = _gpu_inventory(run_id, storage=storage)
        if measured == receipt.get("post_transfer_gpu_inventory"):
            break
        receipt["post_transfer_gpu_inventory"] = measured
    else:
        message = f"Pilot GPU inventory did not reach a fixed point: {directory}"
        raise RuntimeError(message)
    common.serialization.atomic_write_json(receipt_path, receipt)
    common.serialization.atomic_write_text(csv_path, _summary_csv(receipt))
    common.serialization.atomic_write_text(markdown_path, _summary_markdown(receipt))
    if write_pre_cleanup_snapshot:
        common.serialization.atomic_write_json(snapshot_path, receipt)
        measured = _gpu_inventory(run_id, storage=storage)
        if measured != receipt["post_transfer_gpu_inventory"]:
            receipt["post_transfer_gpu_inventory"] = measured
            return _write_receipt_and_views(
                run_id,
                receipt,
                storage=storage,
                write_pre_cleanup_snapshot=True,
            )
    return receipt


def _production_projection_contract(production_campaign: Path | str) -> dict[str, Any]:
    """Resolve the production count and time basis for pilot projections."""
    campaign = config_service.load_campaign_config(
        production_campaign,
        require_executable=False,
    )
    if (
        campaign.campaign_purpose != "family_generalization"
        or campaign.profile.id != profiles.TRANSIENT_DRYING_PROFILE
        or not campaign.batches
        or campaign.total_case_count < 1
    ):
        message = "Pilot projections require a transient family-generalization campaign."
        raise ValueError(message)
    time_contracts: list[tuple[float, ...]] = []
    for batch in campaign.batches:
        time = batch.scientific_values.get("time")
        raw_times = time.get("regular_times") if isinstance(time, dict) else None
        if not isinstance(raw_times, list) or len(raw_times) < _MIN_REGULAR_STATE_COUNT:
            message = f"Production batch {batch.batch_name!r} lacks resolved regular times."
            raise ValueError(message)
        regular_times = tuple(float(value) for value in raw_times)
        if not all(math.isfinite(value) for value in regular_times) or any(current <= previous for previous, current in pairwise(regular_times)):
            message = f"Production batch {batch.batch_name!r} has invalid resolved regular times."
            raise ValueError(message)
        time_contracts.append(regular_times)
    regular_times = time_contracts[0]
    if any(candidate != regular_times for candidate in time_contracts[1:]):
        message = "Production batches disagree on the resolved regular-time contract."
        raise ValueError(message)
    return {
        "target_campaign_id": campaign.campaign_id,
        "target_campaign_digest": campaign.campaign_digest,
        "simulation_profile": campaign.profile.id,
        "target_case_count": campaign.total_case_count,
        "regular_state_count": len(regular_times),
        "regular_time_start_h": regular_times[0],
        "time_horizon_h": regular_times[-1],
    }


def prepare_pilot_receipt(
    run_id: str,
    *,
    production_campaign: Path | str,
    storage_root: Path | str | None = None,
    cleanup_requested: bool = True,
) -> dict[str, Any]:
    """Analyze evidence using an explicit resolved production projection target."""
    if not isinstance(cleanup_requested, bool):
        message = "cleanup_requested must be boolean."
        raise TypeError(message)
    projection_contract = _production_projection_contract(production_campaign)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    terminal = validate_pilot_terminal(run_id, storage_root=storage)
    campaign_evidence.validate_transfer_receipt(
        run_id,
        terminal=terminal,
        plan=pilot_transfer_plan(run_id, storage_root=storage),
        storage_root=storage,
    )
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    snapshot_path = pilot_check_directory(run_id, storage_root=storage) / PILOT_PRE_CLEANUP_FILENAME
    if snapshot_path.exists():
        snapshot = validate_pilot_pre_cleanup(run_id, storage_root=storage)
        if snapshot.get("cleanup", {}).get("cleanup_requested") is not cleanup_requested:
            message = "Existing pilot cleanup choice conflicts with the current --keep-cpu-source selection."
            raise ValueError(message)
        projection = snapshot.get("production_storage_projection")
        if not isinstance(projection, dict) or any(projection.get(key) != value for key, value in projection_contract.items()):
            message = "Existing pilot projection conflicts with the resolved production campaign."
            raise ValueError(message)
        return snapshot
    source_inventory = validate_cpu_source_inventory(run_id, storage_root=storage)
    staging_inventory = validate_transfer_staging_inventory(
        run_id,
        storage_root=storage,
        require_staging_present=True,
    )
    cases: list[dict[str, Any]] = []
    for record in terminal["cases"]:
        if record["terminal_state"] == "success":
            try:
                result = analysis_service.analyze_successful_case(
                    storage / record["processed_directory"],
                    case_kind=record["case_kind"],
                )
            except Exception as error:  # noqa: BLE001 -- invalid canonical evidence is a required result class
                result = _invalid_success_case_result(record, error, storage=storage)
            cases.append(result)
        else:
            cases.append(_failure_case_result(record, storage=storage))
    _validate_case_results(cases, expected_materials=campaign.material_inventory)
    successful = [record for record in cases if record["solver_status"] == "success" and record["storage"].get("canonical_hdf5_bytes") is not None]
    per_material = _per_material(cases, campaign.material_inventory)
    duration_ready = all(
        row["nominal_duration_result"] == "PASS"
        and row["nominal_result_class"] in {"PASS", "PASS_WITH_WARNINGS"}
        and row["nominal_stop_consistent"] is True
        for row in per_material
    )
    retained_paths = [record["retained_evidence_path"] for record in cases]
    if not all(Path(path).exists() for path in retained_paths):
        message = "Pilot cleanup is not authorized because retained GPU evidence is missing."
        raise FileNotFoundError(message)
    receipt = {
        "schema_kind": PILOT_RECEIPT_SCHEMA_KIND,
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot_check_id": run_id,
        "campaign_run_id": run_id,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "git_commit": terminal["git_commit"],
        "config_digests": {
            "campaign_digest": campaign.campaign_digest,
            "batch_scientific_config_digests": {batch.batch_name: batch.scientific_config_digest for batch in campaign.batches},
        },
        "template_digest": campaign.profile.template_sha256,
        "execution_resources": copy.deepcopy(campaign.execution_values),
        "materials": list(campaign.material_inventory),
        "cases_per_material": terminal["cases_per_material"],
        "case_counts": terminal["terminal_counts"],
        "cases": cases,
        "per_material": per_material,
        "problems": _problems(cases),
        "pre_cleanup_cpu_inventory": source_inventory,
        "transfer_staging_inventory": staging_inventory,
        "post_transfer_gpu_inventory": {},
        "production_storage_projection": {
            **analysis_service.production_storage_projection(
                successful,
                target_case_count=projection_contract["target_case_count"],
                regular_state_count=projection_contract["regular_state_count"],
            ),
            **projection_contract,
        },
        "retained_evidence_paths": retained_paths,
        "cleanup": {
            "authorized": True,
            "authorization_reason": "all terminal evidence transferred, hash-validated, analyzed, and retained paths resolved",
            "cleanup_requested": cleanup_requested,
            "cpu_source": {
                "status": "pending" if cleanup_requested else "retained_by_request",
                "removed": False,
                "bytes_reclaimed": 0,
                "receipt_sha256": None,
            },
            "transfer_staging": {
                "status": "pending",
                "removed": False,
                "bytes_reclaimed": 0,
            },
        },
        "readiness": {
            "duration_reference_validation": "complete" if duration_ready else "failed",
            "duration_ready": duration_ready,
            "runtime_failure_count": sum(record["result_class"] in _RUNTIME_FAILURE_RESULT_CLASSES for record in cases),
            "invalid_result_count": sum(record["result_class"] == "INVALID_RESULT" for record in cases),
            "physical_contract_violation_count": sum(record["result_class"] == "PHYSICAL_CONTRACT_VIOLATION" for record in cases),
            "result_class_counts": {
                result_class: sum(record["result_class"] == result_class for record in cases) for result_class in PILOT_RESULT_CLASSES
            },
        },
        "scientific_interpretation": {
            "automatic_parameter_retuning": False,
            "mass_balance_acceptance_tolerance": None,
            "storage_budget_guard": None,
            "experimental_validation_claimed": False,
        },
        "transfer_receipt_sha256": common.serialization.file_sha256(run_directory / "transfer_complete.json"),
        "recorded_at": _utc_now(),
        "completed_at": None,
    }
    _write_receipt_and_views(
        run_id,
        receipt,
        storage=storage,
        write_pre_cleanup_snapshot=True,
    )
    return validate_pilot_pre_cleanup(
        run_id,
        storage_root=storage,
        require_live_evidence=True,
    )


def validate_pilot_pre_cleanup(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    require_live_evidence: bool = False,
) -> dict[str, Any]:
    """Validate the immutable pilot evidence snapshot required for cleanup."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    directory = pilot_check_directory(run_id, storage_root=storage)
    snapshot = _load_json(
        directory / PILOT_PRE_CLEANUP_FILENAME,
        label="pilot pre-cleanup snapshot",
    )
    _validate_case_results(snapshot.get("cases"))
    _validate_storage_projection(snapshot.get("production_storage_projection"))
    retained = snapshot.get("retained_evidence_paths", [])
    retained_paths = [Path(value).resolve() for value in retained if isinstance(value, str)]
    if (
        snapshot.get("schema_kind") != PILOT_RECEIPT_SCHEMA_KIND
        or snapshot.get("schema_version") != PILOT_SCHEMA_VERSION
        or snapshot.get("pilot_check_id") != run_id
        or snapshot.get("cleanup", {}).get("authorized") is not True
        or len(retained_paths) != len(retained)
        or not retained_paths
        or any(not path.exists() or path.is_symlink() or not path.is_relative_to(storage) for path in retained_paths)
    ):
        message = f"Pilot pre-cleanup evidence is invalid: {directory}"
        raise ValueError(message)
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    _validate_case_results(
        snapshot.get("cases"),
        expected_materials=campaign.material_inventory,
    )
    if (
        snapshot.get("campaign_id") != campaign.campaign_id
        or snapshot.get("campaign_digest") != campaign.campaign_digest
        or snapshot.get("materials") != list(campaign.material_inventory)
    ):
        message = f"Pilot pre-cleanup campaign evidence is invalid: {directory}"
        raise ValueError(message)
    validate_cpu_source_inventory(run_id, storage_root=storage)
    validate_transfer_staging_inventory(
        run_id,
        storage_root=storage,
        require_staging_present=require_live_evidence,
    )
    if require_live_evidence:
        terminal = validate_pilot_terminal(run_id, storage_root=storage)
        campaign_evidence.validate_transfer_receipt(
            run_id,
            terminal=terminal,
            plan=pilot_transfer_plan(run_id, storage_root=storage),
            storage_root=storage,
        )
        canonical = _load_json(directory / PILOT_RECEIPT_FILENAME, label="canonical pilot receipt")
        if (
            canonical != snapshot
            or (directory / PILOT_SUMMARY_CSV).read_text(encoding="utf-8") != _summary_csv(snapshot)
            or (directory / PILOT_SUMMARY_MARKDOWN).read_text(encoding="utf-8") != _summary_markdown(snapshot)
        ):
            message = f"Pilot receipt and derived views were not durable before cleanup: {directory}"
            raise ValueError(message)
    return snapshot


def cleanup_recorded_transfer_staging(
    run_id: str,
    *,
    storage_root: Path | str | None,
    confirm: bool,
) -> dict[str, Any]:
    """Transactionally remove only the exact inventoried pilot staging tree."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    validate_pilot_pre_cleanup(run_id, storage_root=storage)
    inventory = validate_transfer_staging_inventory(run_id, storage_root=storage)
    staging = Path(inventory["transfer_staging_path"])
    expected_bytes = int(inventory["transfer_staging_bytes_before_cleanup"])
    expected_files = int(inventory["transfer_staging_file_count"])
    directory = pilot_check_directory(run_id, storage_root=storage)
    receipt_path = directory / PILOT_STAGING_CLEANUP_FILENAME
    identity = {
        "schema_kind": "generation_pilot_staging_cleanup",
        "schema_version": PILOT_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "staging_path": str(staging),
        "expected_bytes": expected_bytes,
        "expected_file_count": expected_files,
        "pre_cleanup_snapshot_sha256": common.serialization.file_sha256(directory / PILOT_PRE_CLEANUP_FILENAME),
    }
    existing = _load_json(receipt_path, label="pilot staging cleanup receipt") if receipt_path.exists() else None
    if existing is not None and any(existing.get(key) != value for key, value in identity.items()):
        message = f"Pilot staging cleanup receipt identity conflicts: {receipt_path}"
        raise ValueError(message)
    if existing is not None and existing.get("status") == "complete":
        if (
            staging.exists()
            or existing.get("removed") is not True
            or existing.get("reclaimed_bytes") != expected_bytes
            or not isinstance(existing.get("completed_at"), str)
        ):
            message = f"Pilot staging cleanup completion is invalid: {receipt_path}"
            raise ValueError(message)
        return existing
    if not confirm:
        return {
            **identity,
            "status": "cleanup_not_authorized",
            "removed": False,
            "reclaimed_bytes": 0,
        }
    if existing is None:
        validate_pilot_pre_cleanup(
            run_id,
            storage_root=storage,
            require_live_evidence=True,
        )
        existing = {
            **identity,
            "status": "pending",
            "removed": False,
            "reclaimed_bytes": 0,
            "started_at": _utc_now(),
            "completed_at": None,
        }
        common.serialization.atomic_write_json(receipt_path, existing)
    elif existing.get("status") != "pending":
        message = f"Pilot staging cleanup receipt has an unsupported state: {receipt_path}"
        raise ValueError(message)
    if staging.exists():
        validate_transfer_staging_inventory(
            run_id,
            storage_root=storage,
            require_staging_present=True,
        )
        reclaimed = workspace_service.cleanup_transfer_staging(
            staging,
            storage_root=storage,
            run_id=run_id,
        )
    else:
        reclaimed = expected_bytes
    if staging.exists() or reclaimed != expected_bytes:
        message = "Pilot transfer staging cleanup did not reclaim the exact inventoried tree."
        raise RuntimeError(message)
    receipt = {
        **identity,
        "status": "complete",
        "removed": True,
        "reclaimed_bytes": reclaimed,
        "started_at": existing["started_at"],
        "completed_at": _utc_now(),
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    return receipt


def finalize_cleanup_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None,
    workflow_evidence: CleanupWorkflowEvidence,
    cpu_source_removed: bool,
    cpu_bytes_reclaimed: int,
    cpu_cleanup_receipt_sha256: str | None,
    transfer_staging_removed: bool,
    staging_bytes_reclaimed: int,
    staging_cleanup_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Finalize canonical cleanup state after source and staging verification."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    validate_pilot_pre_cleanup(run_id, storage_root=storage)
    receipt = _load_json(
        pilot_receipt_path(run_id, storage_root=storage),
        label="canonical pilot receipt",
    )
    for value, label in (
        (cpu_bytes_reclaimed, "cpu_bytes_reclaimed"),
        (staging_bytes_reclaimed, "staging_bytes_reclaimed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = f"{label} must be a non-negative integer."
            raise ValueError(message)
    expected_staging = int(receipt["transfer_staging_inventory"]["transfer_staging_bytes_before_cleanup"])
    staging_cleanup_path = pilot_check_directory(run_id, storage_root=storage) / PILOT_STAGING_CLEANUP_FILENAME
    staging_cleanup = _load_json(staging_cleanup_path, label="pilot staging cleanup receipt") if staging_cleanup_path.exists() else None
    if transfer_staging_removed and (
        staging_bytes_reclaimed != expected_staging
        or staging_cleanup is None
        or staging_cleanup.get("status") != "complete"
        or staging_cleanup.get("removed") is not True
        or staging_cleanup.get("reclaimed_bytes") != expected_staging
        or staging_cleanup_receipt_sha256 != common.serialization.file_sha256(staging_cleanup_path)
    ):
        message = "Verified staging cleanup differs from the recorded pre-cleanup inventory."
        raise ValueError(message)
    cleanup_requested = bool(receipt["cleanup"]["cleanup_requested"])
    if cleanup_requested and not cpu_source_removed:
        message = "Requested pilot CPU source cleanup was not verified."
        raise RuntimeError(message)
    if not cleanup_requested and cpu_source_removed:
        message = "--keep-cpu-source pilot unexpectedly removed CPU source."
        raise RuntimeError(message)
    if not isinstance(workflow_evidence, CleanupWorkflowEvidence):
        message = "Pilot cleanup requires validated workflow evidence."
        raise TypeError(message)
    if workflow_evidence.campaign_run_id != run_id:
        message = "Pilot cleanup workflow evidence belongs to a different campaign run."
        raise ValueError(message)
    if cleanup_requested:
        if (
            workflow_evidence.status != "complete"
            or workflow_evidence.receipt_sha256 != cpu_cleanup_receipt_sha256
            or workflow_evidence.reclaimed_bytes != cpu_bytes_reclaimed
        ):
            message = "Pilot CPU cleanup result differs from the validated all-workflow receipt."
            raise ValueError(message)
    elif workflow_evidence.status != "skipped_by_request" or cpu_cleanup_receipt_sha256 is not None or cpu_bytes_reclaimed != 0:
        message = "Retained pilot CPU source differs from the validated all-workflow receipt."
        raise ValueError(message)
    receipt["cleanup"]["cpu_source"] = {
        "status": "complete" if cpu_source_removed else "retained_by_request",
        "removed": cpu_source_removed,
        "bytes_reclaimed": cpu_bytes_reclaimed,
        "receipt_sha256": cpu_cleanup_receipt_sha256,
    }
    receipt["cleanup"]["transfer_staging"] = {
        "status": "complete" if transfer_staging_removed else "cleanup_not_authorized",
        "removed": transfer_staging_removed,
        "bytes_reclaimed": staging_bytes_reclaimed,
        "receipt_sha256": staging_cleanup_receipt_sha256,
    }
    receipt["completed_at"] = _utc_now() if transfer_staging_removed else None
    return _write_receipt_and_views(
        run_id,
        receipt,
        storage=storage,
        write_pre_cleanup_snapshot=False,
    )


def validate_pilot_receipt(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    require_cleanup_complete: bool = False,
) -> dict[str, Any]:
    """Validate canonical pilot identity, views, and optional cleanup completion."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    validate_pilot_pre_cleanup(run_id, storage_root=storage)
    path = pilot_receipt_path(run_id, storage_root=storage)
    receipt = _load_json(path, label="canonical pilot receipt")
    campaign = campaign_evidence.campaign_for_run(run_id, storage_root=storage)
    _validate_case_results(
        receipt.get("cases"),
        expected_materials=campaign.material_inventory,
    )
    _validate_storage_projection(receipt.get("production_storage_projection"))
    directory = path.parent
    if (
        receipt.get("schema_kind") != PILOT_RECEIPT_SCHEMA_KIND
        or receipt.get("schema_version") != PILOT_SCHEMA_VERSION
        or receipt.get("pilot_check_id") != run_id
        or receipt.get("campaign_purpose") != config_service.PILOT_CAMPAIGN_PURPOSE
        or receipt.get("materials") != list(campaign.material_inventory)
        or receipt.get("scientific_interpretation", {}).get("mass_balance_acceptance_tolerance") is not None
        or receipt.get("scientific_interpretation", {}).get("storage_budget_guard") is not None
        or not all(Path(item).exists() for item in receipt.get("retained_evidence_paths", []))
        or (directory / PILOT_SUMMARY_CSV).read_text(encoding="utf-8") != _summary_csv(receipt)
        or (directory / PILOT_SUMMARY_MARKDOWN).read_text(encoding="utf-8") != _summary_markdown(receipt)
    ):
        message = f"Canonical pilot receipt or derived views are invalid: {path}"
        raise ValueError(message)
    if require_cleanup_complete and (
        receipt["cleanup"]["transfer_staging"]["status"] != "complete"
        or (receipt["cleanup"]["cleanup_requested"] and receipt["cleanup"]["cpu_source"]["status"] != "complete")
    ):
        message = f"Pilot cleanup is not terminally complete: {run_id}"
        raise RuntimeError(message)
    if require_cleanup_complete:
        staging_cleanup_path = directory / PILOT_STAGING_CLEANUP_FILENAME
        staging_sha256 = receipt["cleanup"]["transfer_staging"].get("receipt_sha256")
        if (
            not staging_cleanup_path.is_file()
            or staging_cleanup_path.is_symlink()
            or common.serialization.file_sha256(staging_cleanup_path) != staging_sha256
            or _load_json(staging_cleanup_path, label="pilot staging cleanup receipt").get("status") != "complete"
        ):
            message = f"Pilot staging cleanup evidence is invalid: {run_id}"
            raise ValueError(message)
    return receipt


def terminal_summary(receipt: Mapping[str, Any]) -> str:
    """Render the compact terminal overview, problem list, storage, and projection."""
    lines = [
        "Material       Kind              Case       Solver       Result                       Dry[h] Target Stage          MB|max|      HDF5",
    ]
    for record in receipt["cases"]:
        native = record["conservation_diagnostic"].get("comsol_mt_mass_balance", {})
        lines.append(
            ("{material:<14} {kind:<17} {case:<10} {solver:<12} {result:<28} {dry!s:<6} {target!s:<6} {stage!s:<14} {balance!s:<12} {size!s}").format(
                material=record["material"],
                kind=record["case_kind"],
                case=record["case_id"],
                solver=record["solver_status"],
                result=record["result_class"],
                dry=record["drying_time_h"],
                target=record["target_reached"],
                stage=record["failed_stage"],
                balance=native.get("max_abs"),
                size=record["storage"].get("canonical_hdf5_bytes"),
            )
        )
    lines.extend(["", "Per-material summary"])
    lines.extend(
        f"{row['material']:<14} nominal={row['nominal_result_class']:<28} "
        f"nominal_h={row['nominal_drying_duration_h']} "
        f"duration_med/min/max={row['successful_duration_median_h']}/"
        f"{row['successful_duration_min_h']}/{row['successful_duration_max_h']} "
        f"targets={row['target_reached_count']} failures={row['runtime_failure_count']} "
        f"physical={row['physical_contract_violation_count']} "
        f"worst_native_mb={row['worst_mt_mass_balance_max_abs']} "
        f"worst_total_residual={row['worst_independent_total_water_residual_kg']} "
        f"median_h5={row['median_hdf5_size_bytes']}"
        for row in receipt["per_material"]
    )
    lines.extend(["", "Problems"])
    if receipt["problems"]:
        lines.extend(
            f"{item['material']} {item['case_id']} {item['case_kind']} "
            f"{item['problem_category']}: {item['explanation']}; "
            f"actual={item['actual_value']}; reference={item['reference_value']}; "
            f"evidence={item['retained_gpu_evidence_path']}"
            for item in receipt["problems"]
        )
    else:
        lines.append("none")
    cpu = receipt["pre_cleanup_cpu_inventory"]
    staging = receipt["transfer_staging_inventory"]
    gpu = receipt["post_transfer_gpu_inventory"]
    cleanup = receipt["cleanup"]
    projection = receipt["production_storage_projection"]
    lines.extend(
        [
            "",
            "Storage measured before cleanup",
            f"CPU source: {cpu['cpu_source_bytes_before_cleanup']} bytes",
            f"CPU exports: {cpu['cpu_exports_bytes']} bytes",
            f"CPU logs: {cpu['cpu_logs_bytes']} bytes",
            f"Transfer staging: {staging['transfer_staging_bytes_before_cleanup']} bytes",
            "",
            "Permanent GPU pilot",
            f"Canonical HDF5: {gpu['gpu_generation_hdf5_bytes']} bytes",
            f"Metadata / summaries: {gpu['gpu_generation_meta_bytes'] + gpu['pilot_receipt_and_summary_bytes']} bytes",
            f"Retained failure evidence: {gpu['retained_failure_evidence_bytes']} bytes",
            f"Total permanent GPU: {gpu['current_pilot_gpu_permanent_bytes']} bytes",
            "",
            "Cleanup",
            f"CPU source removed: {cleanup['cpu_source']['removed']}",
            f"Staging removed: {cleanup['transfer_staging']['removed']}",
            f"CPU bytes reclaimed: {cleanup['cpu_source']['bytes_reclaimed']}",
            f"Staging bytes reclaimed: {cleanup['transfer_staging']['bytes_reclaimed']}",
            "",
            f"Projection for {projection['target_case_count']} transient cases",
            f"Target campaign: {projection['target_campaign_id']}",
            f"Configured regular states per full-horizon case: {projection['regular_state_count']}",
            f"Configured time horizon: {projection['time_horizon_h']} h",
            f"Observed mean-based: {projection.get('mean_based_bytes')}",
            f"Observed median-based: {projection.get('median_based_bytes')}",
            f"Configured-horizon projection: {projection.get('full_horizon_projection')}",
        ]
    )
    return "\n".join(lines)
