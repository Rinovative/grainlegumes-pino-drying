"""
===============================================================================
generation_campaign_evidence.py
===============================================================================
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
===============================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common

from . import generation_config as config_service
from . import generation_profiles as profiles
from . import generation_source as source_service
from . import generation_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Mapping

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
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
        "slurm_job_ids",
        "scheduler_job_name",
        "scheduler_log_directory",
        "submission_command",
        "submission_history",
        "resource_plan",
        "wall_time",
        "remote_storage_root",
        "campaign_meta_directory",
        "batches",
        "dataset_packages",
        "state",
    }
)
TRANSFER_OPERATIONAL_RECEIPTS: Final = frozenset(
    {
        "all_workflow.json",
        "cpu_source_cleanup.json",
        "dataset_packages_complete.json",
        "transfer_complete.json",
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


def load_campaign_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one persisted campaign run."""
    manifest = load_json_object(
        campaign_run_manifest_path(run_id, storage_root=storage_root),
        label="campaign-run manifest",
    )
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or manifest.get("schema_kind") != "generation_campaign_run"
        or manifest.get("schema_version") != 1
        or manifest.get("campaign_run_id") != run_id
    ):
        message = f"Unsupported or malformed campaign-run manifest: {run_id}."
        raise ValueError(message)
    source_service.validate_git_commit(manifest.get("git_commit"))
    batch_names = manifest.get("selected_batch_names")
    if (
        not isinstance(batch_names, list)
        or not batch_names
        or not all(isinstance(name, str) for name in batch_names)
        or len(batch_names) != len(set(batch_names))
    ):
        message = f"Campaign-run batch selection is malformed: {run_id}."
        raise ValueError(message)
    state = manifest.get("state")
    job_ids = manifest.get("slurm_job_ids")
    if (
        state not in {"submitting", "submitted", "resubmitting", "cancel_requested"}
        or not isinstance(job_ids, list)
        or len(job_ids) != len(set(job_ids))
        or not all(isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id) is not None for job_id in job_ids)
    ):
        message = f"Campaign-run submission state is malformed: {run_id}."
        raise ValueError(message)
    if (state == "submitting" and job_ids) or (state in {"submitted", "resubmitting", "cancel_requested"} and not job_ids):
        message = f"Campaign-run scheduler identity disagrees with state {state!r}: {run_id}."
        raise ValueError(message)
    common.paths.validate_logical_name(
        manifest.get("scheduler_job_name"),
        label="scheduler_job_name",
    )
    log_directory = manifest.get("scheduler_log_directory")
    if not isinstance(log_directory, str) or not Path(log_directory).is_absolute():
        message = f"Campaign-run scheduler log directory is malformed: {run_id}."
        raise ValueError(message)
    command = manifest.get("submission_command")
    history = manifest.get("submission_history")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) for argument in command)
        or not isinstance(history, list)
        or not history
    ):
        message = f"Campaign-run submission command is malformed: {run_id}."
        raise ValueError(message)
    for attempt_index, attempt in enumerate(history, start=1):
        if (
            not isinstance(attempt, dict)
            or set(attempt) != {"attempt", "kind", "recorded_at", "command", "job_id"}
            or attempt.get("attempt") != attempt_index
            or attempt.get("kind") not in {"initial", "resume"}
            or not isinstance(attempt.get("recorded_at"), str)
            or not isinstance(attempt.get("command"), list)
            or not attempt["command"]
            or not all(isinstance(argument, str) for argument in attempt["command"])
            or (
                attempt.get("job_id") is not None and (not isinstance(attempt["job_id"], str) or _JOB_ID_PATTERN.fullmatch(attempt["job_id"]) is None)
            )
        ):
            message = f"Campaign-run submission history is malformed: {run_id}."
            raise ValueError(message)
    resolve_campaign_config_path(manifest.get("campaign_config"))
    return manifest


def campaign_from_manifest(
    manifest: Mapping[str, Any],
) -> config_service.CampaignConfig:
    """Resolve the exact canonical, smoke, or pilot campaign execution view."""
    config_path = resolve_campaign_config_path(manifest["campaign_config"])
    if manifest["campaign_name"] == (f"{profiles.TRANSIENT_DRYING_PROFILE}_{config_service.PILOT_CAMPAIGN_PURPOSE}"):
        batch_records = manifest.get("batches")
        counts = (
            {record.get("case_count") for record in batch_records}
            if isinstance(batch_records, list) and all(isinstance(record, dict) for record in batch_records)
            else set()
        )
        if len(counts) != 1:
            message = "Pilot campaign manifest has no uniform cases-per-material identity."
            raise RuntimeError(message)
        cases_per_material = next(iter(counts))
        if isinstance(cases_per_material, bool) or not isinstance(
            cases_per_material,
            int,
        ):
            message = "Pilot campaign manifest cases-per-material identity is malformed."
            raise RuntimeError(message)
        campaign = config_service.load_campaign_config(
            config_path,
            pilot_cases_per_material=cases_per_material,
        )
    else:
        campaign = config_service.load_campaign_config(config_path)
        campaign = campaign.select_batches(tuple(manifest["selected_batch_names"]))
    if (
        campaign.campaign_id != manifest["campaign_id"]
        or campaign.campaign_digest != manifest["campaign_digest"]
        or [batch.batch_name for batch in campaign.batches] != manifest["selected_batch_names"]
    ):
        message = "Campaign configuration or execution-view identity changed after launch."
        raise RuntimeError(message)
    return campaign


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
    ignored_names: frozenset[str] = frozenset(),
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
        if not path.is_file() or path.name in ignored_names:
            continue
        relative = path.relative_to(directory).as_posix()
        records[relative] = {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return common.serialization.canonical_json_sha256(records)


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
        ignored_names = TRANSFER_OPERATIONAL_RECEIPTS if relative_value == plan["campaign_directory"] else frozenset()
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                message = f"Transfer source contains a symbolic link: {path}"
                raise ValueError(message)
            if not path.is_file() or path.name in ignored_names:
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
    destination = workspace_service.resolve_storage_root(storage_root, create=False)
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
