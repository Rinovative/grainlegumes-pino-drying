"""
generation_publication_campaign_evidence.py

Own cross-purpose campaign-run identity and immutable transfer evidence.
Responsibilities:
  - Resolve canonical campaign-run paths and validate persisted run manifests
  - Reconstruct the exact resolved campaign bound to a launched run
  - Build and validate symlink-free transfer inventories and receipts
Design principles:
  - Campaign state is independent of purpose-specific terminal evidence
  - Transfer validation consumes explicit terminal and directory-plan evidence
  - Persisted identities fail closed against repository configuration changes
This module does NOT:
  - Submit scheduler jobs or classify pilot case outcomes
  - Publish transferred files or build dataset packages
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_paths as path_contract
from src.generation.contracts import generation_contracts_source as source_service

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_FORBIDDEN_NONORDINARY_SCHEDULER_OPTIONS: Final = (
    "--array",
    "--exclusive",
    "--nodelist",
    "--reservation",
)
_RUN_MANIFEST_SCHEMA_VERSION: Final = 1
TECHNICAL_SMOKE_EVIDENCE_FILENAME: Final = "technical_smoke_evidence.json"
RUNTIME_PROGRESS_DIRECTORY_NAME: Final = "progress"
_RUN_MANIFEST_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "campaign_name",
        "campaign_id",
        "campaign_digest",
        "campaign_config",
        "simulation_profile",
        "selected_batch_names",
        "git_commit",
        "execution_config_digest",
        "slurm_job_ids",
        "scheduler_job_name",
        "scheduler_log_directory",
        "submission_config",
        "submissions",
        "submission_intent",
        "admission_reservations",
        "remote_storage_root",
        "campaign_meta_directory",
        "batches",
        "dataset_packages",
        "state",
    }
)
_RUN_MANIFEST_OPTIONAL_KEYS: Final = frozenset({"synthetic_completion"})
POST_TRANSFER_OPERATIONAL_PATHS: Final = frozenset(
    {
        "all_workflow.json",
        "cpu_source_cleanup.json",
        "dataset_package_extensions",
        "dataset_packages_complete.json",
        "dataset_packages_complete.lock",
        "campaign_partial.json",
        "dataset_packages_incomplete.json",
        "partial_completion.json",
        "transfer_partial.json",
        "transfer_complete.json",
        TECHNICAL_SMOKE_EVIDENCE_FILENAME,
        RUNTIME_PROGRESS_DIRECTORY_NAME,
        "workflow_failures",
    }
)


def campaign_run_directory(
    run_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one persistent campaign-run metadata directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    return common.paths.get_generation_meta_root(storage_root=storage_root) / "campaigns" / safe_id


def campaign_run_manifest_path(
    run_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the persistent campaign-run manifest path."""
    return campaign_run_directory(run_id, storage_root=storage_root) / "campaign_run.json"


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load {label}: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain one JSON object: {path}"
        raise TypeError(message)
    return value


def resolve_campaign_config_path(value: Any) -> Path:
    """Resolve one safe repository-relative campaign configuration path."""
    if not isinstance(value, str) or not value:
        message = "Campaign-run campaign_config must be non-empty repository-relative text."
        raise TypeError(message)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:3] != ("configs", "generation", "campaigns"):
        message = f"Campaign-run campaign_config is not a canonical generation campaign path: {value!r}."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    resolved = (repository / relative).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as error:
        message = f"Campaign-run campaign_config escapes the repository: {value!r}."
        raise ValueError(message) from error
    if not resolved.is_file() or resolved.is_symlink():
        message = f"Campaign-run campaign_config is missing or unsafe: {resolved}."
        raise FileNotFoundError(message)
    return resolved


def _validate_campaign_scheduler_job_ids(
    value: object,
    *,
    run_id: str,
) -> list[str]:
    """Validate the ordered scheduler identities persisted by one campaign."""
    if (
        not isinstance(value, list)
        or len(value) != len(set(value))
        or not all(isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id) is not None for job_id in value)
    ):
        message = f"Campaign-run submission state is malformed: {run_id}."
        raise ValueError(message)
    return value


def _validate_campaign_run_header(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> list[str]:
    """Validate top-level campaign, source, and scheduler identity."""
    if (
        not set(manifest) >= _RUN_MANIFEST_KEYS
        or bool(set(manifest).difference(_RUN_MANIFEST_KEYS | _RUN_MANIFEST_OPTIONAL_KEYS))
        or manifest.get("schema_kind") != "generation_campaign_run"
        or manifest.get("schema_version") != _RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("campaign_run_id") != run_id
    ):
        message = f"Unsupported or malformed campaign-run manifest: {run_id}."
        raise ValueError(message)
    if "synthetic_completion" in manifest:
        from src.generation import generation_campaign_completion as completion_service  # noqa: PLC0415 -- break the completion/evidence import cycle

        completion_service.validate_synthetic_manifest_extension(manifest["synthetic_completion"])
    source_service.validate_git_commit(manifest.get("git_commit"))
    for key in ("campaign_digest", "execution_config_digest"):
        value = manifest.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            message = f"Campaign-run {key} is malformed: {run_id}."
            raise ValueError(message)
    batch_names = manifest.get("selected_batch_names")
    if (
        not isinstance(batch_names, list)
        or not batch_names
        or not all(isinstance(name, str) for name in batch_names)
        or len(batch_names) != len(set(batch_names))
    ):
        message = f"Campaign-run batch selection is malformed: {run_id}."
        raise ValueError(message)
    states = {
        "ready",
        "submitting",
        "active",
        "submission_failed",
        "submission_unknown",
        "scheduler_unknown",
        "failure_threshold_reached",
        "license_blocked",
        "completed_with_failures",
        "complete",
        "cancel_requested",
        "force_cancel_requested",
    }
    if manifest.get("state") not in states:
        message = f"Campaign-run submission state is malformed: {run_id}."
        raise ValueError(message)
    job_ids = _validate_campaign_scheduler_job_ids(
        manifest.get("slurm_job_ids"),
        run_id=run_id,
    )
    common.paths.validate_logical_name(
        manifest.get("scheduler_job_name"),
        label="scheduler_job_name",
    )
    log_directory = manifest.get("scheduler_log_directory")
    if not isinstance(log_directory, str) or not Path(log_directory).is_absolute():
        message = f"Campaign-run scheduler log directory is malformed: {run_id}."
        raise ValueError(message)
    return job_ids


def _validate_submission_configuration(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> None:
    """Validate the persisted feeder and one-case allocation contract."""
    submission = manifest.get("submission_config")
    if not isinstance(submission, dict):
        message = f"Campaign-run submission configuration is malformed: {run_id}."
        raise TypeError(message)
    if set(submission) != {
        "max_admission_cases",
        "poll_interval_seconds",
        "max_running_cases",
        "cores_per_case",
        "maximum_failed_cases",
        "temporary_license_retry",
        "partition",
        "wall_time",
        "scheduler_options",
    }:
        message = f"Campaign-run submission configuration is malformed: {run_id}."
        raise ValueError(message)
    for key in ("max_admission_cases", "poll_interval_seconds", "cores_per_case"):
        value = submission[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            message = f"Campaign-run submission configuration {key!r} is malformed: {run_id}."
            raise ValueError(message)
    maximum_failed_cases = submission["maximum_failed_cases"]
    if isinstance(maximum_failed_cases, bool) or not isinstance(maximum_failed_cases, int) or maximum_failed_cases < 0:
        message = f"Campaign-run submission configuration 'maximum_failed_cases' is malformed: {run_id}."
        raise ValueError(message)
    retry = submission["temporary_license_retry"]
    in_allocation = retry.get("in_allocation_retry") if isinstance(retry, dict) else None
    if (
        not isinstance(retry, dict)
        or set(retry)
        != {
            "enabled",
            "initial_delay_seconds",
            "maximum_delay_seconds",
            "maximum_wait_seconds",
            "in_allocation_retry",
        }
        or not isinstance(retry.get("enabled"), bool)
        or any(
            isinstance(retry.get(key), bool) or not isinstance(retry.get(key), (int, float)) or not 0.0 < float(retry[key]) < float("inf")
            for key in (
                "initial_delay_seconds",
                "maximum_delay_seconds",
            )
        )
        or (
            retry.get("maximum_wait_seconds") is not None
            and (
                isinstance(retry["maximum_wait_seconds"], bool)
                or not isinstance(retry["maximum_wait_seconds"], (int, float))
                or not 0.0 < float(retry["maximum_wait_seconds"]) < float("inf")
            )
        )
        or retry["maximum_delay_seconds"] < retry["initial_delay_seconds"]
        or (retry["maximum_wait_seconds"] is not None and retry["maximum_wait_seconds"] < retry["initial_delay_seconds"])
        or not isinstance(in_allocation, dict)
        or set(in_allocation) != {"enabled", "maximum_window_seconds", "pause_after_capacity_failure_seconds"}
        or not isinstance(in_allocation.get("enabled"), bool)
        or any(
            isinstance(in_allocation.get(key), bool)
            or not isinstance(in_allocation.get(key), (int, float))
            or not 0.0 < float(in_allocation[key]) < float("inf")
            for key in ("maximum_window_seconds", "pause_after_capacity_failure_seconds")
        )
    ):
        message = f"Campaign-run temporary-license retry configuration is malformed: {run_id}."
        raise ValueError(message)
    max_running = submission["max_running_cases"]
    if max_running is not None and (isinstance(max_running, bool) or not isinstance(max_running, int) or max_running < 1):
        message = f"Campaign-run max_running_cases is malformed: {run_id}."
        raise ValueError(message)
    options = submission["scheduler_options"]
    if not isinstance(options, list) or not all(isinstance(option, str) and option.startswith("--") for option in options):
        message = f"Campaign-run scheduler options are malformed: {run_id}."
        raise ValueError(message)


def _validate_submission_record(
    record: object,
    *,
    index: int,
    run_id: str,
) -> tuple[str | None, bool]:
    """Validate one atomic submission attempt and return its state summary."""
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "submission_index",
            "mode",
            "recorded_at",
            "case",
            "job_name",
            "command",
            "job_id",
            "status",
            "error",
        }
        or record.get("submission_index") != index
        or record.get("mode") not in {"initial", "resume", "license_retry"}
        or not isinstance(record.get("recorded_at"), str)
        or not isinstance(record.get("case"), dict)
        or set(record["case"]) != {"batch_name", "batch_id", "case_index", "case_id"}
        or isinstance(record["case"].get("case_index"), bool)
        or not isinstance(record["case"].get("case_index"), int)
        or not isinstance(record.get("command"), list)
        or not record["command"]
        or not all(isinstance(argument, str) for argument in record["command"])
        or record.get("status") not in {"submitting", "submitted", "submission_failed"}
    ):
        message = f"Campaign-run submission record {index} is malformed: {run_id}."
        raise ValueError(message)
    for key in ("batch_name", "batch_id", "case_id"):
        common.paths.validate_logical_name(
            record["case"].get(key),
            label=f"submission case {key}",
        )
    common.paths.validate_logical_name(
        record.get("job_name"),
        label="submission job_name",
    )
    if any(
        argument == forbidden or argument.startswith(f"{forbidden}=")
        for argument in record["command"]
        for forbidden in _FORBIDDEN_NONORDINARY_SCHEDULER_OPTIONS
    ):
        message = f"Campaign-run submission {index} uses forbidden packing, exclusivity, or reservation: {run_id}."
        raise ValueError(message)
    job_id = record.get("job_id")
    if record["status"] == "submitted":
        if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None or record["error"] is not None:
            message = f"Campaign-run submitted record {index} is malformed: {run_id}."
            raise ValueError(message)
        return job_id, False
    if job_id is not None:
        message = f"Campaign-run unresolved record {index} has a job ID: {run_id}."
        raise ValueError(message)
    if record["status"] == "submitting":
        return None, True
    if not isinstance(record["error"], str) or not record["error"]:
        message = f"Campaign-run failed submission {index} has no error: {run_id}."
        raise ValueError(message)
    return None, False


def _validate_admission_reservations(manifest: Mapping[str, Any], *, run_id: str) -> None:
    """Validate durable admitted cases waiting outside scheduler allocation."""
    reservations = manifest.get("admission_reservations")
    if not isinstance(reservations, list):
        message = f"Campaign-run admission reservations are malformed: {run_id}."
        raise TypeError(message)
    identities: set[tuple[str, int]] = set()
    for reservation in reservations:
        if not isinstance(reservation, dict) or set(reservation) != {"batch_name", "batch_id", "case_index", "case_id"}:
            message = f"Campaign-run admission reservation is malformed: {run_id}."
            raise ValueError(message)
        case_index = reservation.get("case_index")
        if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 1:
            message = f"Campaign-run admission reservation case index is malformed: {run_id}."
            raise ValueError(message)
        for key in ("batch_name", "batch_id", "case_id"):
            common.paths.validate_logical_name(
                reservation.get(key),
                label=f"admission reservation {key}",
            )
        identity = str(reservation["batch_id"]), case_index
        if identity in identities:
            message = f"Campaign-run admission reservation is duplicated: {run_id}."
            raise ValueError(message)
        identities.add(identity)


def _validate_submission_records(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[list[str], list[int]]:
    """Validate every submission record and summarize persisted intent."""
    submissions = manifest.get("submissions")
    if not isinstance(submissions, list):
        message = f"Campaign-run submissions are malformed: {run_id}."
        raise TypeError(message)
    persisted_ids: list[str] = []
    unresolved: list[int] = []
    for index, record in enumerate(submissions, start=1):
        job_id, is_unresolved = _validate_submission_record(
            record,
            index=index,
            run_id=run_id,
        )
        if job_id is not None:
            persisted_ids.append(job_id)
        if is_unresolved:
            unresolved.append(index)
    return persisted_ids, unresolved


def load_campaign_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one persisted dynamic-feeder campaign run."""
    manifest = load_json_object(
        campaign_run_manifest_path(run_id, storage_root=storage_root),
        label="campaign-run manifest",
    )
    manifest.setdefault("admission_reservations", [])
    job_ids = _validate_campaign_run_header(manifest, run_id=run_id)
    _validate_submission_configuration(manifest, run_id=run_id)
    _validate_admission_reservations(manifest, run_id=run_id)
    persisted_ids, unresolved = _validate_submission_records(
        manifest,
        run_id=run_id,
    )
    if persisted_ids != job_ids:
        message = f"Campaign-run job IDs disagree with submission records: {run_id}."
        raise ValueError(message)
    intent = manifest.get("submission_intent")
    if (intent is None and unresolved) or (intent is not None and unresolved != [intent]):
        message = f"Campaign-run durable submission intent is inconsistent: {run_id}."
        raise ValueError(message)
    resolve_campaign_config_path(manifest.get("campaign_config"))
    return manifest


def load_campaign_scheduler_job_ids(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> tuple[str, ...]:
    """Load only exact persisted scheduler ownership for one campaign run."""
    path = campaign_run_manifest_path(run_id, storage_root=storage_root)
    if not path.is_file() or path.is_symlink():
        message = f"Campaign-run manifest is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    manifest = load_json_object(path, label="campaign-run manifest")
    if (
        manifest.get("schema_kind") != "generation_campaign_run"
        or manifest.get("schema_version") != _RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("campaign_run_id") != run_id
    ):
        message = f"Unsupported or malformed campaign-run manifest: {run_id}."
        raise ValueError(message)
    job_ids = _validate_campaign_scheduler_job_ids(
        manifest.get("slurm_job_ids"),
        run_id=run_id,
    )
    persisted_ids, unresolved = _validate_submission_records(
        manifest,
        run_id=run_id,
    )
    if persisted_ids != job_ids:
        message = f"Campaign-run job IDs disagree with submission records: {run_id}."
        raise ValueError(message)
    intent = manifest.get("submission_intent")
    if (intent is None and unresolved) or (intent is not None and unresolved != [intent]):
        message = f"Campaign-run durable submission intent is inconsistent: {run_id}."
        raise ValueError(message)
    return tuple(job_ids)


def _validate_admission_reservations_for_campaign(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
) -> None:
    """Bind operational admission reservations to exact unsent plan members."""
    planned = {
        (batch.batch_id, case_index): (batch.batch_name, batch.case_id(case_index)) for batch in campaign.batches for case_index in batch.case_indices
    }
    submissions = manifest.get("submissions", [])
    reservations = manifest.get("admission_reservations", [])
    submitted = {(str(record["case"]["batch_id"]), int(record["case"]["case_index"])) for record in submissions}
    for reservation in reservations:
        key = str(reservation["batch_id"]), int(reservation["case_index"])
        expected = planned.get(key)
        if expected != (reservation["batch_name"], reservation["case_id"]):
            message = "Campaign admission reservation is not an exact resolved plan member."
            raise ValueError(message)
        if key in submitted:
            message = "Campaign admission reservation conflicts with durable submission history."
            raise ValueError(message)


def current_campaign_from_manifest(
    manifest: Mapping[str, Any],
    *,
    require_executable: bool = True,
) -> config_service.CampaignConfig:
    """Resolve current package requests over one unchanged simulation plan."""
    if "synthetic_completion" in manifest:
        from src.generation import generation_campaign_completion as completion_service  # noqa: PLC0415 -- break the completion/evidence import cycle

        campaign = completion_service.campaign_from_synthetic_manifest(
            manifest,
            require_executable=require_executable,
        )
        _validate_admission_reservations_for_campaign(manifest, campaign)
        return campaign
    config_path = resolve_campaign_config_path(manifest["campaign_config"])
    campaign = config_service.load_campaign_config(
        config_path,
        require_executable=require_executable,
    )
    campaign = campaign.select_batches(tuple(manifest["selected_batch_names"]))
    if (
        campaign.campaign_id != manifest["campaign_id"]
        or campaign.campaign_digest != manifest["campaign_digest"]
        or common.serialization.canonical_json_sha256(campaign.execution_values) != manifest["execution_config_digest"]
        or [batch.batch_name for batch in campaign.batches] != manifest["selected_batch_names"]
    ):
        message = "Campaign simulation configuration or execution-view identity changed after launch."
        raise RuntimeError(message)
    _validate_admission_reservations_for_campaign(manifest, campaign)
    return campaign


def campaign_from_manifest(
    manifest: Mapping[str, Any],
    *,
    require_executable: bool = True,
) -> config_service.CampaignConfig:
    """Resolve the launch-time package snapshot for run continuation or inspection."""
    campaign = current_campaign_from_manifest(
        manifest,
        require_executable=require_executable,
    )
    snapshot = manifest.get("dataset_packages")
    if not isinstance(snapshot, list) or not all(isinstance(package, dict) for package in snapshot):
        message = "Campaign-run Dataset package snapshot is malformed."
        raise TypeError(message)
    current_by_name = {str(package["dataset_name"]): package for package in campaign.dataset_packages}
    snapshot_names = [str(package.get("dataset_name")) for package in snapshot]
    if len(snapshot_names) != len(set(snapshot_names)) or any(
        current_by_name.get(name) is None
        or config_service.dataset_package_scientific_plan(current_by_name[name]) != config_service.dataset_package_scientific_plan(package)
        for name, package in zip(snapshot_names, snapshot, strict=True)
    ):
        message = "Campaign launch-time Dataset package declarations were removed or changed."
        raise RuntimeError(message)
    packages = tuple(copy.deepcopy(snapshot))
    return replace(
        campaign,
        evaluation_regimes=tuple(dict.fromkeys(str(package["evaluation_regime"]) for package in packages)),
        dataset_packages=packages,
        package_request_digest=common.serialization.canonical_json_sha256(list(packages)),
    )


def current_campaign_for_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> config_service.CampaignConfig:
    """Return current additive package requests for a persisted simulation run."""
    return current_campaign_from_manifest(
        load_campaign_run(run_id, storage_root=storage_root),
    )


def campaign_for_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> config_service.CampaignConfig:
    """Return the exact repository campaign and batch selection bound to a run."""
    return campaign_from_manifest(
        load_campaign_run(run_id, storage_root=storage_root),
    )


def directory_identity(
    directory: Path,
    *,
    ignored_relative_paths: frozenset[str] = frozenset(),
) -> str:
    """Return a symlink-free exact tree identity for transfer comparison."""
    if not directory.is_dir() or directory.is_symlink():
        message = f"Transfer directory is missing or unsafe: {directory}"
        raise FileNotFoundError(message)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            message = f"Transfer directory contains a symbolic link: {path}"
            raise ValueError(message)
        relative = path.relative_to(directory).as_posix()
        if not path.is_file() or _relative_path_is_ignored(relative, ignored_relative_paths):
            continue
        records[relative] = {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return common.serialization.canonical_json_sha256(records)


def _relative_path_is_ignored(
    relative_path: str,
    ignored_relative_paths: frozenset[str],
) -> bool:
    """Return whether one file is below an ignored operational path."""
    return any(relative_path == ignored or relative_path.startswith(f"{ignored}/") for ignored in ignored_relative_paths)


def _planned_transfer_files(
    plan: Mapping[str, Any],
    *,
    storage_root: Path,
) -> list[tuple[str, Path, int]]:
    """Enumerate exact planned files once without reading their payload bytes."""
    relative_directories = [
        directory
        for batch in plan["batches"]
        for directory in (
            batch["meta_directory"],
            batch["raw_directory"],
            batch["processed_directory"],
            *batch["attempt_directories"],
        )
    ]
    relative_directories.append(plan["campaign_directory"])
    files: list[tuple[str, Path, int]] = []
    seen: set[str] = set()
    for relative_value in relative_directories:
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            message = f"Transfer plan contains an unsafe directory: {relative_value!r}"
            raise ValueError(message)
        directory = (storage_root / relative).resolve()
        if not directory.is_relative_to(storage_root) or not directory.is_dir() or directory.is_symlink():
            message = f"Transfer source is missing or unsafe: {directory}"
            raise FileNotFoundError(message)
        ignored = POST_TRANSFER_OPERATIONAL_PATHS if relative_value == plan["campaign_directory"] else frozenset()
        for candidate in sorted(directory.rglob("*")):
            if candidate.is_symlink():
                message = f"Transfer source contains a symbolic link: {candidate}"
                raise ValueError(message)
            child = candidate.relative_to(directory).as_posix()
            if not candidate.is_file() or _relative_path_is_ignored(child, ignored):
                continue
            relative_path = candidate.relative_to(storage_root).as_posix()
            if relative_path in seen:
                message = f"Transfer inventory contains a duplicate path: {relative_path!r}"
                raise RuntimeError(message)
            seen.add(relative_path)
            files.append(
                (
                    relative_path,
                    candidate,
                    candidate.stat().st_size,
                )
            )
    return sorted(files, key=lambda item: item[0])


def _emit_inventory_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    files_validated: int,
    files_total: int,
    bytes_validated: int,
    bytes_total: int,
) -> None:
    """Emit bounded factual progress without another inventory traversal."""
    if callback is None:
        return
    interval = max(1, (files_total + 19) // 20)
    if files_validated not in {0, files_total} and files_validated % interval != 0:
        return
    callback(
        {
            "operation": "transfer_content_validation",
            "files_validated": files_validated,
            "files_total": files_total,
            "bytes_validated": bytes_validated,
            "bytes_total": bytes_total,
            "eta": "unavailable",
        }
    )


def transfer_inventory_from_plan(
    plan: Mapping[str, Any],
    *,
    storage_root: Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Hash each exact campaign transfer file once with bounded progress."""
    planned = _planned_transfer_files(plan, storage_root=storage_root)
    files_total = len(planned)
    bytes_total = sum(size_bytes for _, _, size_bytes in planned)
    bytes_validated = 0
    records: list[dict[str, Any]] = []
    _emit_inventory_progress(
        progress,
        files_validated=0,
        files_total=files_total,
        bytes_validated=0,
        bytes_total=bytes_total,
    )
    for index, (relative_path, candidate, size_bytes) in enumerate(
        planned,
        start=1,
    ):
        records.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": common.serialization.file_sha256(candidate),
            }
        )
        bytes_validated += size_bytes
        _emit_inventory_progress(
            progress,
            files_validated=index,
            files_total=files_total,
            bytes_validated=bytes_validated,
            bytes_total=bytes_total,
        )
    return {
        "file_count": files_total,
        "size_bytes": bytes_total,
        "files": records,
        "inventory_sha256": common.serialization.canonical_json_sha256(records),
    }


def admit_transfer_inventory(
    inventory: Mapping[str, Any],
    *,
    storage_root: Path,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Admit a persisted transfer inventory with metadata-only file checks.

    This path proves receipt shape, safe regular-file membership, persisted
    sizes, and the aggregate identity.  It intentionally does not hash payload
    bytes; callers requiring fresh content validation must reconstruct the
    inventory through :func:`transfer_inventory_from_plan`.
    """
    required = {"file_count", "size_bytes", "files", "inventory_sha256"}
    if set(inventory) != required:
        message = "Transfer inventory has an unsupported schema."
        raise ValueError(message)
    records = inventory.get("files")
    if not isinstance(records, list):
        message = "Transfer inventory files must be a list."
        raise TypeError(message)
    planned = (
        None
        if plan is None
        else {
            relative_path: size_bytes
            for relative_path, _path, size_bytes in _planned_transfer_files(
                plan,
                storage_root=storage_root,
            )
        }
    )
    seen: set[str] = set()
    previous = ""
    for record in records:
        if not isinstance(record, dict) or set(record) != {"relative_path", "size_bytes", "sha256"}:
            message = "Transfer inventory file record is malformed."
            raise ValueError(message)
        relative_value = record["relative_path"]
        size_bytes = record["size_bytes"]
        digest = record["sha256"]
        relative = Path(relative_value) if isinstance(relative_value, str) else None
        if (
            relative is None
            or not relative_value
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_value in seen
            or relative_value <= previous
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            message = "Transfer inventory file record is malformed."
            raise ValueError(message)
        if planned is None:
            candidate = storage_root
            for part in relative.parts:
                candidate = candidate / part
                if candidate.is_symlink():
                    message = f"Transferred file is missing, unsafe, or has a changed size: {relative_value!r}."
                    raise FileNotFoundError(message)
            resolved = candidate.resolve()
            if not resolved.is_relative_to(storage_root) or not resolved.is_file() or resolved.stat().st_size != size_bytes:
                message = f"Transferred file is missing, unsafe, or has a changed size: {relative_value!r}."
                raise FileNotFoundError(message)
        elif planned.get(relative_value) != size_bytes:
            message = f"Transferred file is missing, unsafe, extra, or has a changed size: {relative_value!r}."
            raise FileNotFoundError(message)
        seen.add(relative_value)
        previous = relative_value
    if planned is not None and set(planned) != seen:
        message = "Transferred file membership differs from its inventory."
        raise FileNotFoundError(message)
    if (
        inventory.get("file_count") != len(records)
        or inventory.get("size_bytes") != sum(int(record["size_bytes"]) for record in records)
        or inventory.get("inventory_sha256") != common.serialization.canonical_json_sha256(records)
    ):
        message = "Transfer inventory aggregate evidence is malformed."
        raise ValueError(message)
    return dict(inventory)


def _validate_transfer_receipt_against_inventory(
    run_id: str,
    *,
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    storage_root: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one receipt against either fresh or admitted inventory evidence."""
    admitted_inventory = admit_transfer_inventory(
        inventory,
        storage_root=storage_root,
        plan=plan,
    )
    run_directory = campaign_run_directory(run_id, storage_root=storage_root)
    terminal_path = run_directory / "campaign_terminal.json"
    receipt_path = run_directory / "transfer_complete.json"
    receipt = load_json_object(receipt_path, label="transfer completion receipt")
    required = {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "source_host",
        "source_storage_root",
        "destination_storage_root",
        "campaign_terminal_sha256",
        "transferred_file_count",
        "transferred_bytes",
        "transfer_inventory_sha256",
        "files",
        "directories",
        "terminal_validation",
        "source_removed",
    }
    expected_directories = {
        plan["campaign_directory"],
        *(
            directory
            for batch in plan["batches"]
            for directory in (
                batch["meta_directory"],
                batch["raw_directory"],
                batch["processed_directory"],
                *batch["attempt_directories"],
            )
        ),
    }
    directory_records = receipt.get("directories")
    observed_directories = (
        {record.get("directory") for record in directory_records if isinstance(record, dict)} if isinstance(directory_records, list) else set()
    )
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != "generation_campaign_transfer"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "transfer_complete"
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != terminal["campaign_id"]
        or receipt.get("git_commit") != terminal["git_commit"]
        or receipt.get("destination_storage_root") != str(storage_root)
        or receipt.get("campaign_terminal_sha256") != common.serialization.file_sha256(terminal_path)
        or receipt.get("transferred_file_count") != admitted_inventory["file_count"]
        or receipt.get("transferred_bytes") != admitted_inventory["size_bytes"]
        or receipt.get("transfer_inventory_sha256") != admitted_inventory["inventory_sha256"]
        or receipt.get("files") != admitted_inventory["files"]
        or observed_directories != expected_directories
        or receipt.get("terminal_validation") != {"status": "pass", "batch_count": len(terminal["batches"])}
        or receipt.get("source_removed") is not False
    ):
        message = f"Transfer completion receipt or GPU publication is invalid: {receipt_path}"
        raise ValueError(message)
    source_host = receipt.get("source_host")
    source_root = receipt.get("source_storage_root")
    if (
        not isinstance(source_host, str)
        or not source_host
        or not isinstance(source_root, str)
        or not Path(source_root).is_absolute()
        or Path(source_root) == Path("/")
    ):
        message = f"Transfer source identity is invalid: {receipt_path}"
        raise ValueError(message)
    return receipt


def validate_transfer_receipt(
    run_id: str,
    *,
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fully rehash immutable GPU publication against terminal evidence."""
    destination = path_contract.resolve_storage_root(storage_root, create=False)
    inventory = transfer_inventory_from_plan(plan, storage_root=destination)
    return _validate_transfer_receipt_against_inventory(
        run_id,
        terminal=terminal,
        plan=plan,
        storage_root=destination,
        inventory=inventory,
    )


def admit_transfer_receipt(
    run_id: str,
    *,
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Admit immutable transfer evidence without rehashing campaign payloads."""
    destination = path_contract.resolve_storage_root(storage_root, create=False)
    receipt_path = campaign_run_directory(run_id, storage_root=destination) / "transfer_complete.json"
    receipt = load_json_object(receipt_path, label="transfer completion receipt")
    inventory = {
        "file_count": receipt.get("transferred_file_count"),
        "size_bytes": receipt.get("transferred_bytes"),
        "files": receipt.get("files"),
        "inventory_sha256": receipt.get("transfer_inventory_sha256"),
    }
    return _validate_transfer_receipt_against_inventory(
        run_id,
        terminal=terminal,
        plan=plan,
        storage_root=destination,
        inventory=inventory,
    )
