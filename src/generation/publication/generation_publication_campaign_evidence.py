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
    from collections.abc import Mapping

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
        "remote_storage_root",
        "campaign_meta_directory",
        "batches",
        "dataset_packages",
        "state",
    }
)
POST_TRANSFER_OPERATIONAL_PATHS: Final = frozenset(
    {
        "all_workflow.json",
        "cpu_source_cleanup.json",
        "dataset_package_extensions",
        "dataset_packages_complete.json",
        "dataset_packages_complete.lock",
        "transfer_complete.json",
        TECHNICAL_SMOKE_EVIDENCE_FILENAME,
        RUNTIME_PROGRESS_DIRECTORY_NAME,
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


def _validate_campaign_run_header(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> list[str]:
    """Validate top-level campaign, source, and scheduler identity."""
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or manifest.get("schema_kind") != "generation_campaign_run"
        or manifest.get("schema_version") != _RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("campaign_run_id") != run_id
    ):
        message = f"Unsupported or malformed campaign-run manifest: {run_id}."
        raise ValueError(message)
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
    job_ids = manifest.get("slurm_job_ids")
    if (
        manifest.get("state") not in states
        or not isinstance(job_ids, list)
        or len(job_ids) != len(set(job_ids))
        or not all(isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id) is not None for job_id in job_ids)
    ):
        message = f"Campaign-run submission state is malformed: {run_id}."
        raise ValueError(message)
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
        "pending_buffer",
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
    for key in ("pending_buffer", "poll_interval_seconds", "cores_per_case"):
        value = submission[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            message = f"Campaign-run submission configuration {key!r} is malformed: {run_id}."
            raise ValueError(message)
    maximum_failed_cases = submission["maximum_failed_cases"]
    if isinstance(maximum_failed_cases, bool) or not isinstance(maximum_failed_cases, int) or maximum_failed_cases < 0:
        message = f"Campaign-run submission configuration 'maximum_failed_cases' is malformed: {run_id}."
        raise ValueError(message)
    retry = submission["temporary_license_retry"]
    if (
        not isinstance(retry, dict)
        or set(retry)
        != {
            "enabled",
            "initial_delay_seconds",
            "maximum_delay_seconds",
            "maximum_wait_seconds",
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
    job_ids = _validate_campaign_run_header(manifest, run_id=run_id)
    _validate_submission_configuration(manifest, run_id=run_id)
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


def current_campaign_from_manifest(
    manifest: Mapping[str, Any],
) -> config_service.CampaignConfig:
    """Resolve the current package requests over one unchanged simulation plan."""
    config_path = resolve_campaign_config_path(manifest["campaign_config"])
    campaign = config_service.load_campaign_config(config_path)
    campaign = campaign.select_batches(tuple(manifest["selected_batch_names"]))
    if (
        campaign.campaign_id != manifest["campaign_id"]
        or campaign.campaign_digest != manifest["campaign_digest"]
        or common.serialization.canonical_json_sha256(campaign.execution_values) != manifest["execution_config_digest"]
        or [batch.batch_name for batch in campaign.batches] != manifest["selected_batch_names"]
    ):
        message = "Campaign simulation configuration or execution-view identity changed after launch."
        raise RuntimeError(message)
    return campaign


def campaign_from_manifest(
    manifest: Mapping[str, Any],
) -> config_service.CampaignConfig:
    """Resolve the launch-time package snapshot for ordinary run continuation."""
    campaign = current_campaign_from_manifest(manifest)
    snapshot = manifest.get("dataset_packages")
    if not isinstance(snapshot, list) or not all(isinstance(package, dict) for package in snapshot):
        message = "Campaign-run Dataset package snapshot is malformed."
        raise TypeError(message)
    current_by_name = {str(package["dataset_name"]): package for package in campaign.dataset_packages}
    snapshot_names = [str(package.get("dataset_name")) for package in snapshot]
    if len(snapshot_names) != len(set(snapshot_names)) or any(
        current_by_name.get(name) != package for name, package in zip(snapshot_names, snapshot, strict=True)
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


def transfer_inventory_from_plan(
    plan: Mapping[str, Any],
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Return the exact symlink-free campaign transfer inventory."""
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
    records: list[dict[str, Any]] = []
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
        ignored_relative_paths = POST_TRANSFER_OPERATIONAL_PATHS if relative_value == plan["campaign_directory"] else frozenset()
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                message = f"Transfer source contains a symbolic link: {path}"
                raise ValueError(message)
            relative_to_directory = path.relative_to(directory).as_posix()
            if not path.is_file() or _relative_path_is_ignored(relative_to_directory, ignored_relative_paths):
                continue
            relative_path = path.relative_to(storage_root).as_posix()
            if relative_path in seen:
                message = f"Transfer inventory contains a duplicate path: {relative_path!r}"
                raise RuntimeError(message)
            seen.add(relative_path)
            records.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": path.stat().st_size,
                    "sha256": common.serialization.file_sha256(path),
                }
            )
    records.sort(key=lambda record: str(record["relative_path"]))
    return {
        "file_count": len(records),
        "size_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
        "inventory_sha256": common.serialization.canonical_json_sha256(records),
    }


def validate_transfer_receipt(
    run_id: str,
    *,
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate immutable GPU publication against explicit terminal evidence."""
    destination = path_contract.resolve_storage_root(storage_root, create=False)
    inventory = transfer_inventory_from_plan(plan, storage_root=destination)
    run_directory = campaign_run_directory(run_id, storage_root=destination)
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
        or receipt.get("destination_storage_root") != str(destination)
        or receipt.get("campaign_terminal_sha256") != common.serialization.file_sha256(terminal_path)
        or receipt.get("transferred_file_count") != inventory["file_count"]
        or receipt.get("transferred_bytes") != inventory["size_bytes"]
        or receipt.get("transfer_inventory_sha256") != inventory["inventory_sha256"]
        or receipt.get("files") != inventory["files"]
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
