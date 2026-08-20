"""
generation_campaign.py

Persist, feed, inspect, and terminally validate one campaign run.
Responsibilities:
  - Bind campaign execution to one clean exact Git commit and execution config
  - Reconcile exact per-case Slurm jobs within one logical admission pool
  - Persist scheduler identity before and after each ordinary job submission
Design principles:
  - One Slurm job owns one exact campaign case with no arrays or node packing
  - Running jobs are unlimited unless the execution config declares a cap
  - Durable case evidence and scheduler accounting make resume duplicate-safe
This module does NOT:
  - Generate scientific inputs, implement SSH/rsync, or build dataset packages
  - Poll indefinitely, submit a whole campaign queue, or delete remote sources
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from src import common

from .cases import generation_cases_config as config_service
from .cases import generation_cases_input as input_service
from .contracts import generation_contracts_source as source_service
from .publication import generation_publication_attempt as attempt_service
from .publication import generation_publication_campaign_evidence as campaign_evidence
from .publication import generation_publication_storage as storage_service
from .runtime import generation_runtime_batch as batch_runtime
from .runtime import generation_runtime_cluster as cluster_service
from .runtime import generation_runtime_license as license_service
from .runtime import generation_runtime_progress as progress_service
from .runtime import generation_runtime_workspace as workspace_service
from .validation import generation_validation_pilot as pilot_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_CASE_ID_PATTERN: Final = re.compile(r"case_([0-9]{4,})")
_TERMINAL_FAILURE_STATES: Final = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
_ACTIVE_PENDING_STATE: Final = "PENDING"
_JOB_NAME_PATTERN: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_SLURM_JOB_NAME_LENGTH: Final = 48
_TRANSFER_PUBLICATION_JOURNAL: Final = ".generation-transfer-publication.json"
_PARTIAL_TRANSFER_PUBLICATION_JOURNAL: Final = ".generation-partial-transfer-publication.json"
_PARTIAL_CAMPAIGN_FILENAME: Final = "campaign_partial.json"
_PARTIAL_TRANSFER_FILENAME: Final = "transfer_partial.json"
_MATERIAL_JOB_CODES: Final = {
    "lentil": "lentil",
    "chickpea": "chickpea",
    "kidney_bean": "kidney",
    "field_pea": "fieldpea",
    "rapeseed": "rapeseed",
    "sunflower_seed": "sunflower",
}
_REGIME_JOB_CODES: Final = {
    "id": "id",
    "parameter_ood": "param",
    "near_family_ood": "near",
    "far_family_ood": "far",
    "extreme_family_ood": "stress",
    "none": "none",
}
_TERMINAL_TIMESTAMP_OWNER: Final = "generation_campaign.terminal_timestamp_normalization.v1"
_CASE_RECONCILIATION_OWNER: Final = "generation_campaign.case_reconciliation.v1"
_MAX_RECONCILIATION_REASON_CHARACTERS: Final = 512


@dataclass(frozen=True, slots=True)
class TerminalTimestampProjection:
    """Preserve one raw terminal timestamp and an optional safe UTC projection."""

    original_value: str | None
    source_timezone: str | None
    normalized_utc_value: str | None
    normalization_owner: str
    normalization_reason: str

    @property
    def completed_at(self) -> str | None:
        """Return the usable presentation timestamp, if one is proven."""
        return self.normalized_utc_value

    def evidence(self) -> dict[str, str | None]:
        """Return compact operational normalization evidence."""
        return {
            "original_value": self.original_value,
            "source_timezone": self.source_timezone,
            "normalized_utc_value": self.normalized_utc_value,
            "normalization_owner": self.normalization_owner,
            "normalization_reason": self.normalization_reason,
        }


@dataclass(frozen=True, slots=True)
class AdmittedCaseAttempt:
    """Bind one attempt to its current or historical replay role."""

    attempt: attempt_service.AttemptEvidence
    historical: bool


class CaseLocalReconciliationError(RuntimeError):
    """Describe one deterministic case-local defect safe to isolate."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        scientific_success_valid: bool,
        source_path: Path | None,
        source_sha256: str | None,
    ) -> None:
        """Initialize one bounded case-local reconciliation failure."""
        super().__init__(message)
        self.category = category
        self.scientific_success_valid = scientific_success_valid
        self.source_path = source_path
        self.source_sha256 = source_sha256


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _repository_commit() -> str:
    """Return the clean repository HEAD commit."""
    repository = common.paths.get_project_root()
    status = subprocess.run(  # noqa: S603 -- fixed Git argument vector
        ["git", "-C", str(repository), "status", "--porcelain"],  # noqa: S607 -- site PATH owns Git
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        message = "Campaign launch requires a clean CPU repository checkout."
        raise RuntimeError(message)
    result = subprocess.run(  # noqa: S603 -- fixed Git argument vector
        ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
        check=True,
        capture_output=True,
        text=True,
    )
    return source_service.validate_git_commit(result.stdout.strip())


def _execution_config_digest(campaign: config_service.CampaignConfig) -> str:
    """Return the exact resolved execution configuration identity."""
    return common.serialization.canonical_json_sha256(campaign.execution_values)


def campaign_run_id(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
) -> str:
    """Return the digest-bound immutable campaign-run identifier."""
    payload = {
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_ids": [batch.batch_id for batch in campaign.batches],
        "git_commit": source_service.validate_git_commit(git_commit),
        "execution_config_digest": _execution_config_digest(campaign),
    }
    digest = common.serialization.canonical_json_sha256(payload)
    return f"{campaign.campaign_name}__{digest[:16]}"


def _repository_relative_campaign_config(campaign: config_service.CampaignConfig) -> str:
    """Return the campaign source as one safe repository-relative path."""
    repository = common.paths.get_project_root().resolve()
    source = campaign.source_path.resolve()
    try:
        relative = source.relative_to(repository)
    except ValueError as error:
        message = "Campaign configuration must remain inside the exact repository checkout."
        raise ValueError(message) from error
    value = relative.as_posix()
    if campaign_evidence.resolve_campaign_config_path(value) != source:
        message = "Campaign configuration did not resolve to its persisted repository-relative path."
        raise RuntimeError(message)
    return value


def _submission_config(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Return the compact feeder and one-case allocation contract."""
    submission = campaign.execution_values["submission"]
    cluster = campaign.execution_values["cluster"]
    return {
        "max_admission_cases": int(submission["max_admission_cases"]),
        "poll_interval_seconds": int(submission["poll_interval_seconds"]),
        "max_running_cases": submission["max_running_cases"],
        "cores_per_case": int(cluster["cores_per_case"]),
        "maximum_failed_cases": int(campaign.execution_values["runtime"]["maximum_failed_cases"]),
        "temporary_license_retry": dict(campaign.execution_values["runtime"]["temporary_license_retry"]),
        "partition": cluster["partition"],
        "wall_time": cluster["wall_time"],
        "scheduler_options": list(cluster["scheduler_options"]),
    }


def _task_payload(task: cluster_service.CampaignTask) -> dict[str, Any]:
    """Return one JSON-ready exact campaign task identity."""
    return {
        "batch_name": task.batch_name,
        "batch_id": task.batch_id,
        "case_index": task.case_index,
        "case_id": task.case_id,
    }


def _task_from_payload(
    campaign: config_service.CampaignConfig,
    payload: Mapping[str, Any],
) -> cluster_service.CampaignTask:
    """Re-resolve one persisted task against exact current campaign membership."""
    case_index = payload.get("case_index")
    if isinstance(case_index, bool) or not isinstance(case_index, int):
        message = "Persisted campaign task has no integer case index."
        raise TypeError(message)
    task = cluster_service.require_campaign_task(
        campaign,
        batch_name=str(payload.get("batch_name")),
        case_index=case_index,
    )
    persisted = {key: payload.get(key) for key in ("batch_name", "batch_id", "case_index", "case_id")}
    if _task_payload(task) != persisted:
        message = "Persisted campaign task no longer matches campaign membership."
        raise ValueError(message)
    return task


def _profile_job_code(campaign: config_service.CampaignConfig) -> str:
    """Return the compact readable profile component for Slurm names."""
    return "td" if campaign.profile.id == "transient_drying" else "sf"


def _regime_job_code(
    campaign: config_service.CampaignConfig,
    batch: config_service.GenerationConfig,
) -> str:
    """Return the operator-facing regime component for one campaign batch."""
    if campaign.campaign_purpose == "technical_runtime_smoke":
        return "smoke"
    if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
        return "pilot"
    if batch.sampling_regime == "parameter_ood":
        return "param"
    try:
        return _REGIME_JOB_CODES[batch.evaluation_regime]
    except KeyError as error:
        message = f"Campaign batch has no readable Slurm regime code: {batch.evaluation_regime!r}."
        raise ValueError(message) from error


def _scheduler_job_name(
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    attempt_index: int,
    run_id: str,
) -> str:
    """Return one readable, unique, bounded Slurm name for a case attempt."""
    if attempt_index < 1:
        message = "Campaign Slurm attempt_index must be positive."
        raise ValueError(message)
    batch = campaign.batch(task.batch_name)
    material = _MATERIAL_JOB_CODES.get(batch.material_family)
    if material is None:
        material = re.sub(r"[^a-z0-9]+", "", batch.material_family.lower())
    suffix = run_id.rsplit("__", maxsplit=1)[-1][:4]
    value = f"{_profile_job_code(campaign)}-{material}-{_regime_job_code(campaign, batch)}-c{task.case_index:04d}-a{attempt_index:02d}-{suffix}"
    if len(value) > _MAX_SLURM_JOB_NAME_LENGTH or _JOB_NAME_PATTERN.fullmatch(value) is None:
        message = f"Campaign Slurm job name is unsafe or exceeds 48 characters: {value!r}."
        raise ValueError(message)
    return value


def _state_batch_root_for_plan(
    batch: config_service.GenerationConfig,
    *,
    storage_root: Path,
) -> Path:
    """Return a flat batch state path without creating it."""
    return common.paths.resolve_generation_state_batch_directory(
        batch.batch_storage_name,
        storage_root=storage_root,
    )


def _new_campaign_manifest(
    campaign: config_service.CampaignConfig,
    *,
    run_id: str,
    requested_commit: str,
    run_directory: Path,
    scheduler_log_directory: Path,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return durable feeder state before the first one-case submission."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    scheduler_prefix = f"{_profile_job_code(campaign)}-campaign-{run_id.rsplit('__', maxsplit=1)[-1][:4]}"
    return {
        "schema_kind": "generation_campaign_run",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_config": _repository_relative_campaign_config(campaign),
        "simulation_profile": campaign.profile.id,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "git_commit": requested_commit,
        "execution_config_digest": _execution_config_digest(campaign),
        "slurm_job_ids": [],
        "scheduler_job_name": scheduler_prefix,
        "scheduler_log_directory": str(scheduler_log_directory),
        "submission_config": _submission_config(campaign),
        "submissions": [],
        "submission_intent": None,
        "admission_reservations": [],
        "remote_storage_root": str(storage),
        "campaign_meta_directory": str(run_directory),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "case_count": len(batch.case_indices),
                "meta_directory": str(batch_runtime.batch_meta_directory(batch, storage_root=storage)),
                "raw_directory": str(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_storage_name,
                        stage="raw",
                        storage_root=storage,
                    )
                ),
                "processed_directory": str(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_storage_name,
                        stage="processed",
                        storage_root=storage,
                    )
                ),
            }
            for batch in campaign.batches
        ],
        "dataset_packages": list(campaign.dataset_packages),
        "state": "ready",
    }


def plan_campaign(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve one exact dynamic-feeder plan without mutation or submission."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    if not storage.is_dir():
        message = f"Campaign plan requires the prepared storage root: {storage}"
        raise FileNotFoundError(message)
    run_id = campaign_run_id(campaign, git_commit=requested_commit)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    log_directory = run_directory / "scheduler"
    tasks = cluster_service.campaign_tasks(campaign)
    first_command = cluster_service.build_campaign_case_slurm_submission_command(
        campaign,
        tasks[0],
        run_id=run_id,
        scheduler_log_directory=log_directory,
        scheduler_job_name=_scheduler_job_name(
            campaign,
            tasks[0],
            attempt_index=1,
            run_id=run_id,
        ),
        attempt_index=1,
    )
    return {
        "schema_kind": "generation_campaign_plan",
        "schema_version": 1,
        "state": "planned",
        "filesystem_mutated": False,
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "campaign_config": _repository_relative_campaign_config(campaign),
        "git_commit": requested_commit,
        "paths": {
            "repository": str(common.paths.get_project_root().resolve()),
            "storage_root": str(storage),
            "run_root": str(run_directory),
            "log_root": str(log_directory),
            "failures": [str(_state_batch_root_for_plan(batch, storage_root=storage) / "failures") for batch in campaign.batches],
            "publications": [
                {
                    "batch_id": batch.batch_id,
                    "raw": str(common.paths.resolve_generated_batch_dir(batch.batch_storage_name, stage="raw", storage_root=storage)),
                    "processed": str(common.paths.resolve_generated_batch_dir(batch.batch_storage_name, stage="processed", storage_root=storage)),
                }
                for batch in campaign.batches
            ],
        },
        "templates": {
            profile_id: {
                "path": str(template.absolute_path),
                "sha256": template.sha256,
            }
            for profile_id, template in config_service.discover_profile_template_identities().items()
        },
        "execution_config": campaign.execution_values,
        "submission_config": _submission_config(campaign),
        "planned_case_jobs": len(tasks),
        "first_submission_command": first_command,
        "submission_model": "one ordinary non-exclusive Slurm job per case within one shared logical admission pool",
    }


def _scheduler_output(command: list[str]) -> tuple[str, str | None]:
    """Return scheduler output or one explicit unavailable reason."""
    try:
        result = subprocess.run(  # noqa: S603 -- callers provide fixed scheduler argv
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return "", str(error)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        return result.stdout.strip(), detail
    return result.stdout.strip(), None


def _parse_submitted_job_id(output: str) -> str:
    """Return one exact root Slurm job ID from parsable sbatch output."""
    token = output.strip().split(";", maxsplit=1)[0]
    if _JOB_ID_PATTERN.fullmatch(token) is None:
        message = f"Slurm returned an invalid parsable job identifier: {output!r}."
        raise RuntimeError(message)
    return token


def _submit_case(command: Sequence[str], *, git_commit: str, run_id: str) -> str:
    """Submit one typed case job and return its numeric Slurm identity."""
    environment = os.environ.copy()
    environment["GENERATION_GIT_COMMIT"] = git_commit
    environment["GENERATION_CAMPAIGN_RUN_ID"] = run_id
    result = subprocess.run(  # noqa: S603 -- cluster service owns the Slurm argv
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return _parse_submitted_job_id(result.stdout)


def _parse_scheduler_rows(output: str, *, field_count: int) -> dict[str, list[str]]:
    """Return exact root-job scheduler rows keyed by numeric job ID."""
    rows: dict[str, list[str]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < field_count or _JOB_ID_PATTERN.fullmatch(fields[0]) is None:
            continue
        rows[fields[0]] = fields
    return rows


def _scheduler_evidence(job_ids: Sequence[str]) -> dict[str, Any]:
    """Query exact campaign jobs from both live queue and durable accounting."""
    if not job_ids:
        return {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
            "active": {},
            "accounted": {},
        }
    selection = ",".join(job_ids)
    squeue_command = [
        "squeue",
        "--noheader",
        f"--jobs={selection}",
        "--format=%i|%T|%R|%N|%V|%S|%M",
    ]
    sacct_command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        selection,
        "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition",
    ]
    squeue_output, squeue_error = _scheduler_output(squeue_command)
    sacct_output, sacct_error = _scheduler_output(sacct_command)
    return {
        "squeue": {"command": squeue_command, "output": squeue_output, "error": squeue_error},
        "sacct": {"command": sacct_command, "output": sacct_output, "error": sacct_error},
        "active": _parse_scheduler_rows(squeue_output, field_count=2),
        "accounted": _parse_scheduler_rows(sacct_output, field_count=3),
    }


def _require_scheduler_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed unless both live and accounting queries succeeded."""
    for owner in ("squeue", "sacct"):
        error = evidence[owner]["error"]
        if error is not None:
            message = f"Cannot reconcile campaign jobs because {owner} failed: {error}"
            raise RuntimeError(message)


def _recover_submission_intent(
    manifest: dict[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Recover one accepted job after interruption between sbatch and persistence."""
    intent = manifest["submission_intent"]
    if intent is None:
        return manifest
    submission_index = int(intent)
    submissions = manifest["submissions"]
    record = submissions[submission_index - 1]
    if record["submission_index"] != submission_index or record["status"] != "submitting" or record["job_id"] is not None:
        message = "Campaign submission intent does not identify one unresolved record."
        raise RuntimeError(message)
    job_name = str(record["job_name"])
    commands = (
        ["squeue", "--noheader", f"--name={job_name}", "--format=%i|%T|%R"],
        ["sacct", "--noheader", "--parsable2", f"--name={job_name}", "--format=JobIDRaw,State,ExitCode"],
    )
    candidates: set[str] = set()
    errors: list[str] = []
    for command in commands:
        output, error = _scheduler_output(command)
        if error is not None:
            errors.append(error)
            continue
        candidates.update(_parse_scheduler_rows(output, field_count=2))
    if errors:
        message = f"Cannot recover unresolved campaign submission {job_name!r}: {errors}"
        raise RuntimeError(message)
    if not candidates:
        return manifest
    if len(candidates) != 1:
        message = f"Scheduler recovery for {job_name!r} is ambiguous across {sorted(candidates)}."
        raise RuntimeError(message)
    job_id = next(iter(candidates))
    record["job_id"] = job_id
    record["status"] = "submitted"
    manifest["slurm_job_ids"].append(job_id)
    manifest["submission_intent"] = None
    manifest["state"] = "active"
    common.serialization.atomic_write_json(
        campaign_evidence.campaign_run_manifest_path(
            str(manifest["campaign_run_id"]),
            storage_root=storage_root,
        ),
        manifest,
    )
    return campaign_evidence.load_campaign_run(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )


def _task_submissions(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
) -> tuple[Mapping[str, Any], ...]:
    """Return submission records for one exact case in attempt order."""
    expected = _task_payload(task)
    return tuple(record for record in manifest["submissions"] if record["case"] == expected)


def _require_unambiguous_active_task_ownership(
    task: cluster_service.CampaignTask,
    active_records: Sequence[Mapping[str, Any]],
) -> None:
    """Reject more than one active Slurm submission for one exact case."""
    if len(active_records) <= 1:
        return
    job_ids = sorted(str(record["job_id"]) for record in active_records)
    message = f"Conflicting active Slurm submission ownership for {task.batch_id}/{task.case_id}: {job_ids}"
    raise RuntimeError(message)


def _scheduler_state(value: str) -> str:
    """Normalize one Slurm state token without its optional suffix."""
    return value.split("+", maxsplit=1)[0].split(maxsplit=1)[0]


def _scheduler_field(row: object, index: int) -> str | None:
    """Return one available scheduler field from a parsed row."""
    if not isinstance(row, list) or index >= len(row):
        return None
    value = row[index].strip()
    if not value or value.upper() in {"N/A", "NONE", "NOT_SET", "UNKNOWN", "(NULL)"}:
        return None
    return value


def _task_scheduler_view(
    latest_submission: Mapping[str, Any] | None,
    scheduler: Mapping[str, Any],
) -> dict[str, str | None]:
    """Return named scheduler evidence for one task's latest attempt."""
    if latest_submission is None:
        return {
            "latest_job_id": None,
            "latest_job_name": None,
            "scheduler_state": None,
            "node": None,
            "submit_time": None,
            "start_time": None,
            "elapsed": None,
            "end_time": None,
            "queue_age": None,
        }
    raw_job_id = latest_submission.get("job_id")
    job_id = raw_job_id if isinstance(raw_job_id, str) else None
    job_name_value = latest_submission.get("job_name")
    job_name = job_name_value if isinstance(job_name_value, str) else None
    active = scheduler["active"].get(job_id) if job_id is not None else None
    accounted = scheduler["accounted"].get(job_id) if job_id is not None else None
    if active is not None:
        raw_scheduler_state = _scheduler_field(active, 1)
        scheduler_state = _scheduler_state(raw_scheduler_state) if raw_scheduler_state is not None else None
        is_pending = scheduler_state == _ACTIVE_PENDING_STATE
        return {
            "latest_job_id": job_id,
            "latest_job_name": job_name,
            "scheduler_state": scheduler_state,
            "node": None if is_pending else _scheduler_field(active, 3),
            "submit_time": _scheduler_field(active, 4),
            "start_time": None if is_pending else _scheduler_field(active, 5),
            "elapsed": None if is_pending else _scheduler_field(active, 6),
            "end_time": None,
            "queue_age": _scheduler_field(active, 6) if is_pending else None,
        }
    return {
        "latest_job_id": job_id,
        "latest_job_name": job_name,
        "scheduler_state": (_scheduler_state(accounted_state) if (accounted_state := _scheduler_field(accounted, 1)) is not None else None),
        "node": _scheduler_field(accounted, 7),
        "submit_time": _scheduler_field(accounted, 3),
        "start_time": _scheduler_field(accounted, 4),
        "elapsed": _scheduler_field(accounted, 6),
        "end_time": _scheduler_field(accounted, 5),
        "queue_age": None,
    }


def _task_runtime_progress_view(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    *,
    latest_job_id: str | None,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return best-effort progress for one task's latest exact job."""
    if latest_job_id is None:
        return {
            "availability": "unavailable",
            "reason": "no_job",
            "age_seconds": None,
            "stale": None,
        }
    return progress_service.load_runtime_progress(
        str(manifest["campaign_run_id"]),
        latest_job_id,
        _task_payload(task),
        storage_root=storage_root,
        manifest=manifest,
    )


def _admitted_case_attempt(
    manifest: Mapping[str, Any],
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> AdmittedCaseAttempt | None:
    """Return the newest attempt after exact campaign and case identity checks."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    current_run_id = str(manifest["campaign_run_id"])
    attempt = attempt_service.latest_case_attempt(
        batch,
        task.case_index,
        current_run_id,
        storage_root=storage,
    )
    historical = False
    if attempt is None:
        candidate = attempt_service.latest_case_attempt_across_campaign_runs(
            batch,
            task.case_index,
            storage_root=storage,
        )
        if (
            candidate is None
            or candidate.payload["campaign_run_id"] == current_run_id
            or candidate.payload["failure_stage"] not in {"conversion", "publication"}
        ):
            return None
        attempt = candidate
        historical = True
    input_reference = input_service.admit_persisted_input_case(
        batch,
        task.case_index,
        str(attempt.payload["input_generation_id"]),
        storage_root=storage,
    )
    canonical_raw_case = input_reference.case_directory.resolve().relative_to(storage).as_posix()
    expected = {
        "campaign_purpose": batch.scientific_values["campaign_purpose"],
        "batch_storage_name": batch.batch_storage_name,
        "batch_id": batch.batch_id,
        "batch_identity": batch.batch_identity,
        "input_generation_id": input_reference.source_id,
        "case_id": task.case_id,
        "case_index": task.case_index,
        "case_input_id": input_reference.case_input_id,
        "simulation_case_id": input_reference.simulation_case_id,
        "canonical_raw_case": canonical_raw_case,
        "scientific_config_digest": batch.scientific_config_digest,
        "export_contract_sha256": common.serialization.canonical_json_sha256(batch.scientific_values["output_contract"]),
    }
    attempt_run_id = common.paths.validate_logical_name(
        attempt.directory.parent.name,
        label="attempt campaign_run_id",
    )
    current_run_identity_valid = attempt.payload.get("campaign_run_id") == attempt_run_id and (
        historical or (attempt_run_id == current_run_id and attempt.payload.get("solver_git_commit") == manifest["git_commit"])
    )
    if (
        not current_run_identity_valid
        or any(attempt.payload.get(key) != value for key, value in expected.items())
        or attempt.payload.get("template")
        != {
            "relative_path": batch.template_relative_path,
            "sha256": batch.template_sha256,
        }
    ):
        message = f"Attempt evidence disagrees with its persisted campaign case: {attempt.receipt_path}"
        raise RuntimeError(message)
    return AdmittedCaseAttempt(
        attempt=attempt,
        historical=historical,
    )


def _successful_status_path(
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the already-validated processed status used for presentation."""
    return (
        batch_runtime.processed_case_directory(
            batch,
            task.case_index,
            storage_root=storage_root,
        )
        / "status.json"
    )


def _empty_successful_status_summary() -> dict[str, Any]:
    """Return presentation defaults that do not own scientific success."""
    return {
        "quality_flag_count": 0,
        "simulated_end_time": None,
        "simulated_end_time_unit": None,
        "final_bulk_moisture_wb": None,
        "target_moisture_wb": None,
    }


def _optional_presentation_float(value: object) -> float | None:
    """Return one display-only numeric value without raising on huge integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        return None


def _successful_status_reconciliation_error(
    task: cluster_service.CampaignTask,
    source_path: Path,
    message: str,
) -> CaseLocalReconciliationError:
    """Bind one presentation-only defect to the exact source bytes."""
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        detail = f"Could not bind successful-case presentation source bytes: {source_path}"
        raise RuntimeError(detail) from error
    return CaseLocalReconciliationError(
        f"Successful-case presentation metadata is unusable for {task.case_id}: {message}",
        category="successful_case_presentation_metadata",
        scientific_success_valid=True,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _successful_status_summary(
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return compact fields without letting presentation metadata own success."""
    status_path = _successful_status_path(
        batch,
        task,
        storage_root=storage_root,
    )
    if not status_path.exists() and batch.profile.id == "transient_drying":
        message = f"Successful transient case lacks required processed status evidence: {status_path}"
        raise RuntimeError(message)
    if not status_path.exists():
        return _empty_successful_status_summary()
    if status_path.is_symlink() or not status_path.is_file():
        message = f"Processed case status path is unsafe: {status_path}"
        raise RuntimeError(message)
    try:
        status = campaign_evidence.load_json_object(
            status_path,
            label="processed case status",
        )
    except (TypeError, ValueError) as error:
        raise _successful_status_reconciliation_error(
            task,
            status_path,
            str(error),
        ) from error
    count = status.get("quality_flag_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _successful_status_reconciliation_error(
            task,
            status_path,
            "malformed quality-flag count",
        )
    units = status.get("units")
    unit_values = units if isinstance(units, dict) else {}
    simulated_end_value = _optional_presentation_float(status.get("t_stop_exact"))
    simulated_end_unit = unit_values.get("t_stop_exact")
    target_entry = batch.scientific_values["material"]["parameter_registry"].get("X_target_wb") if batch.profile.id == "transient_drying" else None
    target_moisture_wb = _optional_presentation_float(target_entry.get("nominal")) if isinstance(target_entry, dict) else None
    simulated_end_available = simulated_end_value is not None and isinstance(simulated_end_unit, str) and bool(simulated_end_unit)
    final_bulk_moisture_wb = None
    if batch.profile.id == "transient_drying":
        canonical_case_path = status_path.parent / "case.h5"
        try:
            final_bulk_moisture_wb = storage_service.read_transient_final_bulk_moisture(
                canonical_case_path,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            if canonical_case_path.exists():
                raise _successful_status_reconciliation_error(
                    task,
                    canonical_case_path,
                    str(error),
                ) from error
            message = f"Successful transient case lacks required canonical final bulk-moisture evidence: {canonical_case_path}"
            raise RuntimeError(message) from error
    return {
        "quality_flag_count": count,
        "simulated_end_time": (simulated_end_value if simulated_end_available else None),
        "simulated_end_time_unit": (simulated_end_unit if simulated_end_available else None),
        "final_bulk_moisture_wb": final_bulk_moisture_wb,
        "target_moisture_wb": target_moisture_wb,
    }


def _scheduler_terminal_case_state(scheduler_state: str) -> tuple[str, str]:
    """Map one terminal scheduler outcome to a resumable case classification."""
    if scheduler_state == "CANCELLED":
        return "cancelled", "scheduler_cancelled"
    if scheduler_state in {"DEADLINE", "TIMEOUT"}:
        return "timed_out", f"scheduler_{scheduler_state.lower()}"
    if scheduler_state in {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED"}:
        return "interrupted", f"scheduler_{scheduler_state.lower()}"
    return (
        "failed",
        (f"scheduler_{scheduler_state.lower()}" if scheduler_state in _TERMINAL_FAILURE_STATES else "completed_without_valid_case_evidence"),
    )


def _postprocessing_replay_view(
    batch: config_service.GenerationConfig,
    failure_stage: str | None,
    attempt: attempt_service.AttemptEvidence | None,
) -> dict[str, Any]:
    """Project replay eligibility without changing attempt or workspace state."""
    default = {
        "postprocessing_replay_available": False,
        "postprocessing_state": "not_applicable",
        "replay_eligible": False,
        "replay_running": False,
        "replay_blocked": False,
        "replay_block_reason": None,
        "replay_attempt_count": 0,
        "replay_evidence_path": None,
    }
    if attempt is None or failure_stage not in {"conversion", "publication"}:
        return default
    status = batch_runtime.replay_case_postprocessing_status(batch, attempt)
    eligible = status["eligible"] is True
    blocked = status["blocked"] is True
    evidence = status["evidence_path"]
    return {
        "postprocessing_replay_available": attempt.replay_available,
        "postprocessing_state": ("replay_blocked" if blocked else "replay_eligible" if eligible else "replay_unavailable"),
        "replay_eligible": eligible,
        "replay_running": False,
        "replay_blocked": blocked,
        "replay_block_reason": str(status["reason"]) if blocked else None,
        "replay_attempt_count": int(status["attempt_count"]),
        "replay_evidence_path": evidence if isinstance(evidence, str) else None,
    }


def _license_retry_is_active(
    state: str,
    latest_submission: Mapping[str, Any] | None,
    runtime_progress: Mapping[str, Any],
) -> bool:
    """Return whether the current pre-solver job is a license retry."""
    return bool(
        latest_submission is not None
        and latest_submission.get("mode") == "license_retry"
        and state in {"pending", "active", "scheduler_unknown"}
        and not _runtime_proves_license_acquired({"runtime_progress": runtime_progress})
    )


def _terminal_timestamp_projection(
    value: object,
    *,
    source_timezone: tzinfo | None,
) -> TerminalTimestampProjection:
    """Normalize one timestamp only when its timezone provenance is authoritative."""
    original = value if isinstance(value, str) and value else None
    if original is None:
        return TerminalTimestampProjection(
            original_value=None,
            source_timezone=None,
            normalized_utc_value=None,
            normalization_owner=_TERMINAL_TIMESTAMP_OWNER,
            normalization_reason="terminal_timestamp_unavailable",
        )
    try:
        parsed = datetime.fromisoformat(original.replace("Z", "+00:00"))
    except ValueError:
        return TerminalTimestampProjection(
            original_value=original,
            source_timezone=None,
            normalized_utc_value=None,
            normalization_owner=_TERMINAL_TIMESTAMP_OWNER,
            normalization_reason="terminal_timestamp_ambiguous_for_presentation",
        )
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return TerminalTimestampProjection(
            original_value=original,
            source_timezone="embedded_offset",
            normalized_utc_value=parsed.astimezone(timezone.utc).isoformat(),
            normalization_owner=_TERMINAL_TIMESTAMP_OWNER,
            normalization_reason="terminal_timestamp_timezone_aware",
        )
    if source_timezone is None:
        return TerminalTimestampProjection(
            original_value=original,
            source_timezone=None,
            normalized_utc_value=None,
            normalization_owner=_TERMINAL_TIMESTAMP_OWNER,
            normalization_reason="terminal_timestamp_ambiguous_for_presentation",
        )
    localized = parsed.replace(tzinfo=source_timezone)
    if localized.utcoffset() is None:
        message = "Authoritative terminal source timezone has no UTC offset."
        raise ValueError(message)
    return TerminalTimestampProjection(
        original_value=original,
        source_timezone=str(source_timezone),
        normalized_utc_value=localized.astimezone(timezone.utc).isoformat(),
        normalization_owner=_TERMINAL_TIMESTAMP_OWNER,
        normalization_reason="terminal_timestamp_normalized",
    )


def _successful_completion_at(
    state: str,
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> TerminalTimestampProjection:
    """Project the admitted successful-processing timestamp for presentation."""
    if state != "successful":
        return _terminal_timestamp_projection(None, source_timezone=None)
    processing_path = (
        batch_runtime.processed_case_directory(
            batch,
            task.case_index,
            storage_root=storage_root,
        )
        / "processing_provenance.json"
    )
    if not processing_path.exists():
        return _terminal_timestamp_projection(None, source_timezone=None)
    if processing_path.is_symlink() or not processing_path.is_file():
        message = f"Successful-case processing provenance path is unsafe: {processing_path}"
        raise RuntimeError(message)
    processing = campaign_evidence.load_json_object(
        processing_path,
        label="successful-case processing provenance",
    )
    return _terminal_timestamp_projection(
        processing.get("recorded_at"),
        source_timezone=None,
    )


_TERMINAL_TIMESTAMP_SCHEMA_KIND: Final = "generation_terminal_timestamp_projection"
_TERMINAL_TIMESTAMP_SCHEMA_VERSION: Final = 1
_TERMINAL_TIMESTAMP_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "slurm_job_id",
        "source_field",
        "source_sha256",
        "original_value",
        "source_timezone",
        "normalized_utc_value",
        "normalization_owner",
        "normalization_reason",
        "recorded_at",
    }
)


def _terminal_timestamp_evidence_path(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    job_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one exact operational Slurm-End projection receipt path."""
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Terminal timestamp evidence requires one numeric Slurm job ID."
        raise ValueError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    safe_batch = common.paths.validate_logical_name(task.batch_id, label="batch_id")
    safe_case = common.paths.validate_logical_name(task.case_id, label="case_id")
    return run_directory / "case_reconciliation" / safe_batch / safe_case / f"{job_id}.terminal_timestamp.json"


def _validate_terminal_timestamp_evidence(
    payload: object,
    *,
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    job_id: str,
    path: Path,
) -> dict[str, Any]:
    """Validate one exact schema-version-1 external timestamp projection."""
    if (
        not isinstance(payload, dict)
        or set(payload) != _TERMINAL_TIMESTAMP_KEYS
        or payload.get("schema_kind") != _TERMINAL_TIMESTAMP_SCHEMA_KIND
        or payload.get("schema_version") != _TERMINAL_TIMESTAMP_SCHEMA_VERSION
        or payload.get("campaign_run_id") != manifest["campaign_run_id"]
        or payload.get("batch_id") != task.batch_id
        or payload.get("case_id") != task.case_id
        or payload.get("slurm_job_id") != job_id
        or payload.get("source_field") != "sacct.End"
        or not isinstance(payload.get("source_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["source_sha256"]) is None
        or not isinstance(payload.get("original_value"), str)
        or not payload["original_value"]
        or payload.get("normalization_owner") != _TERMINAL_TIMESTAMP_OWNER
        or payload.get("normalization_reason")
        not in {
            "terminal_timestamp_timezone_aware",
            "terminal_timestamp_normalized",
            "terminal_timestamp_ambiguous_for_presentation",
        }
        or (payload.get("source_timezone") is not None and not isinstance(payload["source_timezone"], str))
        or (payload.get("normalized_utc_value") is not None and not isinstance(payload["normalized_utc_value"], str))
    ):
        message = f"Terminal timestamp projection evidence is malformed: {path}"
        raise ValueError(message)
    expected_digest = common.serialization.canonical_json_sha256(
        {
            "slurm_job_id": job_id,
            "source_field": "sacct.End",
            "original_value": payload["original_value"],
        }
    )
    if payload["source_sha256"] != expected_digest:
        message = f"Terminal timestamp source digest is malformed: {path}"
        raise ValueError(message)
    normalized = payload["normalized_utc_value"]
    if normalized is not None:
        _parse_utc_timestamp(
            normalized,
            label="Normalized terminal timestamp",
        )
    reason = payload["normalization_reason"]
    if reason == "terminal_timestamp_ambiguous_for_presentation" and (
        payload["source_timezone"] is not None or payload["normalized_utc_value"] is not None
    ):
        message = f"Ambiguous terminal timestamp evidence fabricated a timezone: {path}"
        raise ValueError(message)
    if reason == "terminal_timestamp_timezone_aware" and (payload["source_timezone"] != "embedded_offset" or payload["normalized_utc_value"] is None):
        message = f"Timezone-aware terminal timestamp evidence is incomplete: {path}"
        raise ValueError(message)
    if reason == "terminal_timestamp_normalized" and (
        payload["source_timezone"] is None or payload["source_timezone"] == "embedded_offset" or payload["normalized_utc_value"] is None
    ):
        message = f"Normalized terminal timestamp evidence lacks provenance: {path}"
        raise ValueError(message)
    _parse_utc_timestamp(
        str(payload["recorded_at"]),
        label="Terminal timestamp evidence record",
    )
    return payload


def _load_terminal_timestamp_evidence(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    job_id: str,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load one existing external timestamp projection without scheduler I/O."""
    path = _terminal_timestamp_evidence_path(
        manifest,
        task,
        job_id,
        storage_root=storage_root,
    )
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        message = f"Terminal timestamp projection evidence is unsafe: {path}"
        raise ValueError(message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load terminal timestamp projection evidence: {path}"
        raise ValueError(message) from error
    return _validate_terminal_timestamp_evidence(
        payload,
        manifest=manifest,
        task=task,
        job_id=job_id,
        path=path,
    )


def _record_terminal_timestamp_evidence(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    job_id: str,
    projection: TerminalTimestampProjection,
    *,
    storage_root: Path | str | None,
) -> Path | None:
    """Persist raw Slurm End and its non-scientific UTC projection."""
    if projection.original_value is None:
        return None
    path = _terminal_timestamp_evidence_path(
        manifest,
        task,
        job_id,
        storage_root=storage_root,
    )
    payload = {
        "schema_kind": _TERMINAL_TIMESTAMP_SCHEMA_KIND,
        "schema_version": _TERMINAL_TIMESTAMP_SCHEMA_VERSION,
        "campaign_run_id": manifest["campaign_run_id"],
        "batch_id": task.batch_id,
        "case_id": task.case_id,
        "slurm_job_id": job_id,
        "source_field": "sacct.End",
        "source_sha256": common.serialization.canonical_json_sha256(
            {
                "slurm_job_id": job_id,
                "source_field": "sacct.End",
                "original_value": projection.original_value,
            }
        ),
        **projection.evidence(),
        "recorded_at": _utc_now(),
    }
    _validate_terminal_timestamp_evidence(
        payload,
        manifest=manifest,
        task=task,
        job_id=job_id,
        path=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        message = f"Terminal timestamp projection directory is unsafe: {path.parent}"
        raise ValueError(message)
    if path.exists():
        existing = _load_terminal_timestamp_evidence(
            manifest,
            task,
            job_id,
            storage_root=storage_root,
        )
        comparable = dict(payload)
        comparable["recorded_at"] = None if existing is None else existing["recorded_at"]
        if existing != comparable:
            message = f"Terminal timestamp projection evidence conflicts: {path}"
            raise FileExistsError(message)
        return path
    common.serialization.atomic_write_json(path, payload)
    _validate_terminal_timestamp_evidence(
        payload,
        manifest=manifest,
        task=task,
        job_id=job_id,
        path=path,
    )
    return path


_CASE_RECONCILIATION_SCHEMA_KIND: Final = "generation_case_reconciliation"
_CASE_RECONCILIATION_SCHEMA_VERSION: Final = 1
_CASE_RECONCILIATION_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "case_index",
        "failure_category",
        "scientific_success_valid",
        "admission_continues",
        "source_path",
        "source_sha256",
        "reconciliation_owner",
        "reconciliation_dependency_sha256",
        "resulting_state",
        "reason",
        "recorded_at",
    }
)


def _case_reconciliation_dependency_sha256() -> str:
    """Return the dependency-scoped identity for case-local classification."""
    return common.serialization.canonical_json_sha256(
        {
            "owner": _CASE_RECONCILIATION_OWNER,
            "schema_version": _CASE_RECONCILIATION_SCHEMA_VERSION,
            "categories": ["successful_case_presentation_metadata"],
            "states": ["successful", "case_reconciliation_failed"],
        }
    )


def _case_reconciliation_source(
    source_path: Path,
    *,
    storage_root: Path | str | None,
) -> tuple[str, str]:
    """Return one safe storage-relative source path and its exact byte digest."""
    if source_path.is_symlink() or not source_path.is_file():
        message = f"Case-local reconciliation source is unsafe: {source_path}"
        raise RuntimeError(message)
    storage = workspace_service.resolve_storage_root(
        storage_root,
        create=False,
    ).resolve()
    resolved = source_path.resolve()
    try:
        relative = resolved.relative_to(storage).as_posix()
    except ValueError as error:
        message = f"Case-local reconciliation source escaped storage: {source_path}"
        raise RuntimeError(message) from error
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        message = f"Could not bind case-local reconciliation source: {source_path}"
        raise RuntimeError(message) from error
    return relative, source_sha256


def _case_reconciliation_evidence_path(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    source_sha256: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return a direct digest-keyed case-local reconciliation receipt path."""
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        message = "Case-local reconciliation requires one SHA-256 source identity."
        raise ValueError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    safe_batch = common.paths.validate_logical_name(task.batch_id, label="batch_id")
    safe_case = common.paths.validate_logical_name(task.case_id, label="case_id")
    dependency = _case_reconciliation_dependency_sha256()
    return run_directory / "case_reconciliation_failures" / safe_batch / safe_case / f"{source_sha256}.{dependency}.json"


def _case_reconciliation_payload(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    error: CaseLocalReconciliationError,
    *,
    source_path: str,
    source_sha256: str,
    recorded_at: str,
) -> dict[str, Any]:
    """Build one bounded schema-version-1 case-local classification receipt."""
    reason = " ".join(str(error).split())
    if len(reason) > _MAX_RECONCILIATION_REASON_CHARACTERS:
        reason = f"{reason[: _MAX_RECONCILIATION_REASON_CHARACTERS - 3].rstrip()}..."
    resulting_state = "successful" if error.scientific_success_valid else "case_reconciliation_failed"
    return {
        "schema_kind": _CASE_RECONCILIATION_SCHEMA_KIND,
        "schema_version": _CASE_RECONCILIATION_SCHEMA_VERSION,
        "campaign_run_id": manifest["campaign_run_id"],
        "batch_id": task.batch_id,
        "case_id": task.case_id,
        "case_index": task.case_index,
        "failure_category": error.category,
        "scientific_success_valid": error.scientific_success_valid,
        "admission_continues": True,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "reconciliation_owner": _CASE_RECONCILIATION_OWNER,
        "reconciliation_dependency_sha256": (_case_reconciliation_dependency_sha256()),
        "resulting_state": resulting_state,
        "reason": reason,
        "recorded_at": recorded_at,
    }


def _validate_case_reconciliation_evidence(
    payload: object,
    *,
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    source_path: str,
    source_sha256: str,
    path: Path,
) -> dict[str, Any]:
    """Validate one exact case-local classification without hiding corruption."""
    if (
        not isinstance(payload, dict)
        or set(payload) != _CASE_RECONCILIATION_KEYS
        or payload.get("schema_kind") != _CASE_RECONCILIATION_SCHEMA_KIND
        or payload.get("schema_version") != _CASE_RECONCILIATION_SCHEMA_VERSION
        or payload.get("campaign_run_id") != manifest["campaign_run_id"]
        or payload.get("batch_id") != task.batch_id
        or payload.get("case_id") != task.case_id
        or payload.get("case_index") != task.case_index
        or payload.get("failure_category") != "successful_case_presentation_metadata"
        or not isinstance(payload.get("scientific_success_valid"), bool)
        or payload.get("admission_continues") is not True
        or payload.get("source_path") != source_path
        or payload.get("source_sha256") != source_sha256
        or payload.get("reconciliation_owner") != _CASE_RECONCILIATION_OWNER
        or payload.get("reconciliation_dependency_sha256") != _case_reconciliation_dependency_sha256()
        or payload.get("resulting_state") not in {"successful", "case_reconciliation_failed"}
        or not isinstance(payload.get("reason"), str)
        or not payload["reason"]
        or len(payload["reason"]) > _MAX_RECONCILIATION_REASON_CHARACTERS
        or not isinstance(payload.get("recorded_at"), str)
    ):
        message = f"Case-local reconciliation evidence is malformed: {path}"
        raise ValueError(message)
    expected_state = "successful" if payload["scientific_success_valid"] else "case_reconciliation_failed"
    if payload["resulting_state"] != expected_state:
        message = f"Case-local reconciliation state is contradictory: {path}"
        raise ValueError(message)
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        message = f"Case-local reconciliation source reference is unsafe: {path}"
        raise ValueError(message)
    _parse_utc_timestamp(
        payload["recorded_at"],
        label="Case-local reconciliation record",
    )
    return payload


def _load_case_reconciliation_evidence(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    *,
    source_path: str,
    source_sha256: str,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load one direct digest-keyed case-local receipt, if already recorded."""
    path = _case_reconciliation_evidence_path(
        manifest,
        task,
        source_sha256,
        storage_root=storage_root,
    )
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        message = f"Case-local reconciliation evidence is unsafe: {path}"
        raise ValueError(message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load case-local reconciliation evidence: {path}"
        raise ValueError(message) from error
    return _validate_case_reconciliation_evidence(
        payload,
        manifest=manifest,
        task=task,
        source_path=source_path,
        source_sha256=source_sha256,
        path=path,
    )


def _record_case_reconciliation_evidence(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    error: CaseLocalReconciliationError,
    *,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], Path]:
    """Persist one deterministic case-local receipt bound to unchanged bytes."""
    if error.source_path is None or error.source_sha256 is None:
        message = "Case-local reconciliation error lacks exact source evidence."
        raise RuntimeError(message)
    source_path, source_sha256 = _case_reconciliation_source(
        error.source_path,
        storage_root=storage_root,
    )
    if source_sha256 != error.source_sha256:
        message = f"Case-local reconciliation source changed while it was classified: {error.source_path}"
        raise RuntimeError(message)
    path = _case_reconciliation_evidence_path(
        manifest,
        task,
        source_sha256,
        storage_root=storage_root,
    )
    payload = _case_reconciliation_payload(
        manifest,
        task,
        error,
        source_path=source_path,
        source_sha256=source_sha256,
        recorded_at=_utc_now(),
    )
    _validate_case_reconciliation_evidence(
        payload,
        manifest=manifest,
        task=task,
        source_path=source_path,
        source_sha256=source_sha256,
        path=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        message = f"Case-local reconciliation directory is unsafe: {path.parent}"
        raise ValueError(message)
    if path.exists():
        existing = _load_case_reconciliation_evidence(
            manifest,
            task,
            source_path=source_path,
            source_sha256=source_sha256,
            storage_root=storage_root,
        )
        if existing is None:
            message = f"Case-local reconciliation evidence disappeared while loading: {path}"
            raise RuntimeError(message)
        comparable = dict(payload)
        comparable["recorded_at"] = existing["recorded_at"]
        if existing != comparable:
            message = f"Case-local reconciliation evidence conflicts: {path}"
            raise FileExistsError(message)
        return existing, path
    common.serialization.atomic_write_json(path, payload)
    return payload, path


def _matching_case_reconciliation_evidence(
    manifest: Mapping[str, Any],
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], Path] | None:
    """Load the exact unchanged presentation defect without scanning storage."""
    run_directory = campaign_evidence.campaign_run_directory(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    receipt_directory = (
        run_directory
        / "case_reconciliation_failures"
        / common.paths.validate_logical_name(task.batch_id, label="batch_id")
        / common.paths.validate_logical_name(task.case_id, label="case_id")
    )
    if not receipt_directory.exists():
        return None
    if receipt_directory.is_symlink() or not receipt_directory.is_dir():
        message = f"Case-local reconciliation directory is unsafe: {receipt_directory}"
        raise ValueError(message)
    source = _successful_status_path(
        batch,
        task,
        storage_root=storage_root,
    )
    if not source.exists():
        return None
    source_path, source_sha256 = _case_reconciliation_source(
        source,
        storage_root=storage_root,
    )
    payload = _load_case_reconciliation_evidence(
        manifest,
        task,
        source_path=source_path,
        source_sha256=source_sha256,
        storage_root=storage_root,
    )
    if payload is None:
        return None
    return (
        payload,
        _case_reconciliation_evidence_path(
            manifest,
            task,
            source_sha256,
            storage_root=storage_root,
        ),
    )


def _unsubmitted_task_state(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    submissions: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Return controller lifecycle for a case without active terminal evidence."""
    if submissions:
        return "never_started", str(submissions[-1]["error"] or "submission_failed")
    if (task.batch_id, task.case_index) in _admission_reservation_keys(manifest):
        return "admission_waiting", "admission_reserved"
    return "never_started", "not_submitted"


def _task_state(  # noqa: C901, PLR0912, PLR0915 -- centralized case evidence reconciliation
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
    cached_case_reconciliation: tuple[Mapping[str, Any], Path | None] | None = None,
) -> dict[str, Any]:
    """Reconcile one case from processed, attempt, and exact scheduler evidence."""
    batch = campaign.batch(task.batch_name)
    submissions = _task_submissions(manifest, task)
    latest_submission = submissions[-1] if submissions else None
    active_records = [record for record in submissions if record["job_id"] in scheduler["active"]]
    _require_unambiguous_active_task_ownership(task, active_records)
    unknown_records = [
        record
        for record in submissions
        if record["status"] == "submitted" and record["job_id"] not in scheduler["active"] and record["job_id"] not in scheduler["accounted"]
    ]
    retry_attempt: Mapping[str, Any] | None = None
    allocation_window: Mapping[str, Any] | None = None
    attempt: attempt_service.AttemptEvidence | None = None
    admitted_attempt: AdmittedCaseAttempt | None = None
    case_reconciliation = None if cached_case_reconciliation is None else cached_case_reconciliation[0]
    case_reconciliation_evidence_path = (
        None if cached_case_reconciliation is None or cached_case_reconciliation[1] is None else str(cached_case_reconciliation[1])
    )
    pipeline = {
        "solver_state": "not_started",
        "exports_state": "not_started",
        "conversion_state": "not_started",
        "diagnostics_state": "not_started",
        "publication_state": "not_started",
    }
    failure_stage: str | None = None
    replay = _postprocessing_replay_view(batch, failure_stage, attempt)
    attempt_index: int | None = None
    quality_flag_count = 0
    successful_status: Mapping[str, Any] = {
        "simulated_end_time": None,
        "simulated_end_time_unit": None,
        "final_bulk_moisture_wb": None,
        "target_moisture_wb": None,
    }
    canonical_raw_case: str | None = None
    license_retry_eligible = False
    license_wait_exhausted = False
    license_first_blocked_at: str | None = None
    license_next_retry_at: str | None = None

    if batch_runtime.completed_case_is_valid(
        batch,
        task.case_index,
        storage_root=storage_root,
    ):
        state = "successful"
        reason = "validated_case_evidence"
        pipeline = {
            "solver_state": "succeeded",
            "exports_state": "succeeded",
            "conversion_state": "succeeded",
            "diagnostics_state": "complete",
            "publication_state": "succeeded",
        }
        if (
            case_reconciliation is not None
            and case_reconciliation.get("scientific_success_valid") is True
            and case_reconciliation.get("resulting_state") == "successful"
        ):
            successful_status = _empty_successful_status_summary()
        else:
            successful_status = _successful_status_summary(
                batch,
                task,
                storage_root=storage_root,
            )
        quality_flag_count = int(successful_status["quality_flag_count"])
    else:
        latest_accounted = next(
            (scheduler["accounted"][record["job_id"]] for record in reversed(submissions) if record["job_id"] in scheduler["accounted"]),
            None,
        )
        if active_records:
            active_state = _scheduler_state(scheduler["active"][active_records[-1]["job_id"]][1])
            state = "pending" if active_state == _ACTIVE_PENDING_STATE else "active"
            reason = active_state
        elif unknown_records:
            state = "scheduler_unknown"
            reason = str(unknown_records[-1]["job_id"])
        else:
            if latest_submission is not None:
                latest_job_id = str(latest_submission["job_id"])
                retry_attempt = license_service.latest_wait_for_job(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    job_id=latest_job_id,
                    storage_root=storage_root,
                )
                allocation_window = license_service.load_in_allocation_license_window(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    job_id=latest_job_id,
                    storage_root=storage_root,
                )
            if retry_attempt is not None:
                first_blocked_at = retry_attempt["first_blocked_at"]
                if first_blocked_at is not None and not isinstance(first_blocked_at, str):
                    message = f"Temporary-license first blocked timestamp is malformed for {task.case_id}."
                    raise TypeError(message)
                license_first_blocked_at = first_blocked_at
                next_retry_at = retry_attempt["next_retry_at"]
                if next_retry_at is not None and not isinstance(next_retry_at, str):
                    message = f"Temporary-license retry timestamp is malformed for {task.case_id}."
                    raise TypeError(message)
                license_next_retry_at = next_retry_at
            admitted_attempt = _admitted_case_attempt(
                manifest,
                batch,
                task,
                storage_root=storage_root,
            )
            if admitted_attempt is not None and admitted_attempt.historical and submissions:
                admitted_attempt = None
            attempt = None if admitted_attempt is None else admitted_attempt.attempt
            if attempt is not None:
                state = str(attempt.payload["case_state"])
                reason = str(attempt.payload["reason"])
                failure_stage = str(attempt.payload["failure_stage"])
                pipeline = {
                    key: str(attempt.payload[key])
                    for key in (
                        "solver_state",
                        "exports_state",
                        "conversion_state",
                        "diagnostics_state",
                        "publication_state",
                    )
                }
                replay = _postprocessing_replay_view(batch, failure_stage, attempt)
                attempt_index = int(attempt.payload["attempt_index"])
                flags = attempt.payload.get("quality_flags")
                quality_flag_count = len(flags) if isinstance(flags, list) else 0
                raw_reference = attempt.payload.get("canonical_raw_case")
                canonical_raw_case = raw_reference if isinstance(raw_reference, str) else None
                if state == "license_blocked":
                    cleanup = attempt_service.attempt_cleanup_evidence(attempt)
                    if cleanup is not None and cleanup["status"] == "failed":
                        state = "failed"
                        reason = "license_attempt_scratch_cleanup_failed"
                        pipeline["solver_state"] = "failed"
                    else:
                        if retry_attempt is None:
                            message = f"License-blocked attempt lacks its retry receipt: {attempt.receipt_path}"
                            raise ValueError(message)
                        if compact_license_attempt_payloads:
                            attempt_service.compact_license_only_attempt_payload(
                                batch,
                                task.case_index,
                                str(manifest["campaign_run_id"]),
                                retry_attempt,
                                storage_root=storage_root,
                            )
                        license_wait_exhausted = not bool(retry_attempt["retry_budget_remaining"])
                        license_retry_eligible = not license_wait_exhausted and license_service.wait_record_is_eligible(retry_attempt)
            elif retry_attempt is not None:
                completed_allocation = latest_accounted is not None and _scheduler_state(latest_accounted[1]) == "COMPLETED"
                if completed_allocation and (allocation_window is None or allocation_window.get("outcome") != "window_exhausted"):
                    state = "failed"
                    reason = "completed_without_valid_license_window_receipt"
                    failure_stage = "solver"
                    pipeline["solver_state"] = "failed"
                else:
                    state = "license_blocked"
                    reason = str(allocation_window["reason"]) if allocation_window is not None else license_service.TEMPORARY_LICENSE_CAPACITY
                    license_wait_exhausted = not bool(retry_attempt["retry_budget_remaining"])
                    license_retry_eligible = not license_wait_exhausted and license_service.wait_record_is_eligible(retry_attempt)
            elif batch_runtime.case_failure_is_recorded(
                batch,
                task.case_index,
                storage_root=storage_root,
                execution_run_id=str(manifest["campaign_run_id"]),
                git_commit=str(manifest["git_commit"]),
            ):
                state = "failed"
                reason = "case_failure_evidence"
                failure_stage = "solver"
                pipeline["solver_state"] = "failed"
            elif latest_accounted is not None:
                state, reason = _scheduler_terminal_case_state(_scheduler_state(latest_accounted[1]))
                failure_stage = "solver"
                pipeline["solver_state"] = state
            else:
                state, reason = _unsubmitted_task_state(
                    manifest,
                    task,
                    submissions,
                )

    scheduler_view = _task_scheduler_view(latest_submission, scheduler)
    runtime_progress = _task_runtime_progress_view(
        manifest,
        task,
        latest_job_id=scheduler_view["latest_job_id"],
        storage_root=storage_root,
    )
    license_retry_active = _license_retry_is_active(
        state,
        latest_submission,
        runtime_progress,
    )
    completion_timestamp = _successful_completion_at(
        state,
        batch,
        task,
        storage_root=storage_root,
    )
    scheduler_terminal_timestamp = _terminal_timestamp_projection(
        scheduler_view.get("end_time") if state == "successful" else None,
        source_timezone=None,
    )
    terminal_timestamp_evidence_path: str | None = None
    terminal_job_id = scheduler_view["latest_job_id"]
    if state == "successful" and isinstance(terminal_job_id, str) and _JOB_ID_PATTERN.fullmatch(terminal_job_id) is not None:
        if compact_license_attempt_payloads and scheduler_terminal_timestamp.original_value is not None:
            recorded_path = _record_terminal_timestamp_evidence(
                manifest,
                task,
                terminal_job_id,
                scheduler_terminal_timestamp,
                storage_root=storage_root,
            )
            terminal_timestamp_evidence_path = None if recorded_path is None else str(recorded_path)
        existing_timestamp = _load_terminal_timestamp_evidence(
            manifest,
            task,
            terminal_job_id,
            storage_root=storage_root,
        )
        if existing_timestamp is not None:
            if (
                scheduler_terminal_timestamp.original_value is not None
                and existing_timestamp["original_value"] != scheduler_terminal_timestamp.original_value
            ):
                message = f"Terminal timestamp source changed for one completed Slurm job: {terminal_job_id}"
                raise RuntimeError(message)
            scheduler_terminal_timestamp = TerminalTimestampProjection(
                original_value=str(existing_timestamp["original_value"]),
                source_timezone=existing_timestamp["source_timezone"],
                normalized_utc_value=existing_timestamp["normalized_utc_value"],
                normalization_owner=str(existing_timestamp["normalization_owner"]),
                normalization_reason=str(existing_timestamp["normalization_reason"]),
            )
            terminal_timestamp_evidence_path = str(
                _terminal_timestamp_evidence_path(
                    manifest,
                    task,
                    terminal_job_id,
                    storage_root=storage_root,
                )
            )
    status_artifact_recoveries = (
        ()
        if scheduler_view["latest_job_id"] is None
        else license_service.load_in_allocation_status_artifact_recoveries(
            batch,
            task.case_index,
            campaign_run_id=str(manifest["campaign_run_id"]),
            job_id=str(scheduler_view["latest_job_id"]),
            storage_root=storage_root,
        )
    )
    failed_timestamp = _terminal_timestamp_projection(
        (
            attempt.payload.get("recorded_at")
            if attempt is not None
            and state
            in {
                "failed",
                "timed_out",
                "exports_failed",
                "conversion_failed",
                "publication_failed",
            }
            else scheduler_view.get("end_time")
        ),
        source_timezone=None,
    )
    failed_at = (
        failed_timestamp.completed_at
        if state
        in {
            "failed",
            "timed_out",
            "exports_failed",
            "conversion_failed",
            "publication_failed",
        }
        else None
    )
    return {
        **_task_payload(task),
        "material": batch.material_family,
        "requested_cores": int(campaign.execution_values["cluster"]["cores_per_case"]),
        "state": state,
        "reason": reason,
        "submission_count": len(submissions),
        "attempt_index": attempt_index,
        "attempt_campaign_run_id": None if attempt is None else str(attempt.payload["campaign_run_id"]),
        "failure_stage": failure_stage,
        **pipeline,
        "quality_flag_count": quality_flag_count,
        "completed_at": completion_timestamp.completed_at,
        "terminal_timestamp": scheduler_terminal_timestamp.evidence(),
        "terminal_timestamp_evidence_path": (terminal_timestamp_evidence_path),
        "failed_at": failed_at,
        **successful_status,
        **replay,
        "canonical_raw_case": canonical_raw_case,
        **scheduler_view,
        "runtime_progress": runtime_progress,
        "temporary_license_retry": retry_attempt,
        "in_allocation_license_window": allocation_window,
        "status_artifact_recoveries": list(status_artifact_recoveries),
        "status_artifact_recovery_count": sum(record["cleanup_state"] == "complete" for record in status_artifact_recoveries),
        "license_retry_active": license_retry_active,
        "license_retry_eligible": license_retry_eligible,
        "license_wait_exhausted": license_wait_exhausted,
        "license_first_blocked_at": license_first_blocked_at,
        "license_next_retry_at": license_next_retry_at,
        "evidence_path": None if attempt is None else str(attempt.receipt_path),
        "case_reconciliation": case_reconciliation,
        "case_reconciliation_evidence_path": (case_reconciliation_evidence_path),
        "automatic_continuation_allowed": bool(
            replay["replay_eligible"]
            or license_retry_eligible
            or state
            in {
                "admission_waiting",
                "never_started",
                "cancelled",
                "interrupted",
            }
        ),
    }


def _finalize_completed_batches(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None,
) -> None:
    """Idempotently publish batch manifests once every member validates."""
    for batch in campaign.batches:
        if all(batch_runtime.completed_case_is_valid(batch, case_index, storage_root=storage_root) for case_index in batch.case_indices):
            batch_runtime.finalize_batch(batch, storage_root=storage_root)


def _case_reconciliation_failed_view(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_path: Path | None,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return one isolated lifecycle view while unrelated cases continue."""
    batch = campaign.batch(task.batch_name)
    submissions = _task_submissions(manifest, task)
    latest_submission = submissions[-1] if submissions else None
    active_records = [record for record in submissions if record["job_id"] in scheduler["active"]]
    unknown_records = [
        record
        for record in submissions
        if record["status"] == "submitted" and record["job_id"] not in scheduler["active"] and record["job_id"] not in scheduler["accounted"]
    ]
    _require_unambiguous_active_task_ownership(task, active_records)
    if active_records:
        active_state = _scheduler_state(scheduler["active"][active_records[-1]["job_id"]][1])
        state = "pending" if active_state == _ACTIVE_PENDING_STATE else "active"
    elif unknown_records:
        state = "scheduler_unknown"
    else:
        state = str(evidence["resulting_state"])
    scheduler_view = _task_scheduler_view(latest_submission, scheduler)
    runtime_progress = _task_runtime_progress_view(
        manifest,
        task,
        latest_job_id=scheduler_view["latest_job_id"],
        storage_root=storage_root,
    )
    failure_stage = "reconciliation" if state == "case_reconciliation_failed" else None
    pipeline = {
        "solver_state": "unknown" if failure_stage is not None else "not_started",
        "exports_state": "unknown" if failure_stage is not None else "not_started",
        "conversion_state": "unknown" if failure_stage is not None else "not_started",
        "diagnostics_state": "unknown" if failure_stage is not None else "not_started",
        "publication_state": "unknown" if failure_stage is not None else "not_started",
    }
    timestamp = _terminal_timestamp_projection(None, source_timezone=None)
    return {
        **_task_payload(task),
        "material": batch.material_family,
        "requested_cores": int(campaign.execution_values["cluster"]["cores_per_case"]),
        "state": state,
        "reason": str(evidence["failure_category"]),
        "submission_count": len(submissions),
        "attempt_index": None,
        "attempt_campaign_run_id": None,
        "failure_stage": failure_stage,
        **pipeline,
        "quality_flag_count": 0,
        "completed_at": None,
        "terminal_timestamp": timestamp.evidence(),
        "terminal_timestamp_evidence_path": None,
        "failed_at": (str(evidence["recorded_at"]) if state == "case_reconciliation_failed" else None),
        **_empty_successful_status_summary(),
        **_postprocessing_replay_view(batch, failure_stage, None),
        "canonical_raw_case": None,
        **scheduler_view,
        "runtime_progress": runtime_progress,
        "temporary_license_retry": None,
        "in_allocation_license_window": None,
        "status_artifact_recoveries": [],
        "status_artifact_recovery_count": 0,
        "license_retry_active": False,
        "license_retry_eligible": False,
        "license_wait_exhausted": False,
        "license_first_blocked_at": None,
        "license_next_retry_at": None,
        "evidence_path": None if evidence_path is None else str(evidence_path),
        "case_reconciliation": dict(evidence),
        "case_reconciliation_evidence_path": (None if evidence_path is None else str(evidence_path)),
        "automatic_continuation_allowed": state in {"active", "pending", "scheduler_unknown"},
    }


def _reconciled(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Reconcile every case while isolating only typed case-local defects."""
    task_views: list[dict[str, Any]] = []
    for task in cluster_service.campaign_tasks(campaign):
        batch = campaign.batch(task.batch_name)
        cached = _matching_case_reconciliation_evidence(
            manifest,
            batch,
            task,
            storage_root=storage_root,
        )
        if cached is not None and cached[0]["resulting_state"] == "case_reconciliation_failed":
            view = _case_reconciliation_failed_view(
                manifest,
                campaign,
                task,
                scheduler,
                cached[0],
                cached[1],
                storage_root=storage_root,
            )
            task_views.append(view)
            continue
        try:
            view = _task_state(
                manifest,
                campaign,
                task,
                scheduler,
                storage_root=storage_root,
                compact_license_attempt_payloads=(compact_license_attempt_payloads),
                cached_case_reconciliation=cached,
            )
        except CaseLocalReconciliationError as error:
            if error.source_path is None or error.source_sha256 is None:
                message = f"Typed case-local reconciliation error lacks exact source evidence for {task.case_id}."
                raise RuntimeError(message) from error
            source_path, source_sha256 = _case_reconciliation_source(
                error.source_path,
                storage_root=storage_root,
            )
            if source_sha256 != error.source_sha256:
                message = f"Case-local evidence changed during reconciliation for {task.case_id}."
                raise RuntimeError(message) from error
            if compact_license_attempt_payloads:
                evidence, evidence_path = _record_case_reconciliation_evidence(
                    manifest,
                    task,
                    error,
                    storage_root=storage_root,
                )
            else:
                evidence = _case_reconciliation_payload(
                    manifest,
                    task,
                    error,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    recorded_at=_utc_now(),
                )
                evidence_path = None
            if error.scientific_success_valid:
                view = _task_state(
                    manifest,
                    campaign,
                    task,
                    scheduler,
                    storage_root=storage_root,
                    compact_license_attempt_payloads=(compact_license_attempt_payloads),
                    cached_case_reconciliation=(evidence, evidence_path),
                )
            else:
                view = _case_reconciliation_failed_view(
                    manifest,
                    campaign,
                    task,
                    scheduler,
                    evidence,
                    evidence_path,
                    storage_root=storage_root,
                )
        task_views.append(view)
    reservation_keys = _admission_reservation_keys(manifest)
    waiting_keys = {_task_identity_key(view) for view in task_views if view["state"] == "admission_waiting"}
    if waiting_keys != reservation_keys:
        message = "Campaign admission reservations conflict with reconciled case evidence."
        raise RuntimeError(message)
    persisted_job_ids = set(manifest["slurm_job_ids"])
    states = [_scheduler_state(fields[1]) for job_id, fields in scheduler["active"].items() if job_id in persisted_job_ids]
    pending_jobs = sum(state == _ACTIVE_PENDING_STATE for state in states)
    running_jobs = len(states) - pending_jobs
    return task_views, pending_jobs, running_jobs


def _next_case_attempt_index(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
) -> int:
    """Return one plus the count of real persisted jobs for this case."""
    return (
        sum(
            record["status"] == "submitted" and isinstance(record["job_id"], str) and bool(record["job_id"])
            for record in _task_submissions(manifest, task)
        )
        + 1
    )


def _submit_one(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    mode: str,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist one intent, submit one case, and atomically persist its job ID."""
    if mode not in {"initial", "resume", "license_retry"}:
        message = f"Unsupported campaign submission mode: {mode!r}."
        raise ValueError(message)
    index = len(manifest["submissions"]) + 1
    attempt_index = _next_case_attempt_index(manifest, task)
    job_name = _scheduler_job_name(
        campaign,
        task,
        attempt_index=attempt_index,
        run_id=str(manifest["campaign_run_id"]),
    )
    command = cluster_service.build_campaign_case_slurm_submission_command(
        campaign,
        task,
        run_id=str(manifest["campaign_run_id"]),
        scheduler_log_directory=Path(manifest["scheduler_log_directory"]),
        scheduler_job_name=job_name,
        attempt_index=attempt_index,
    )
    record = {
        "submission_index": index,
        "mode": mode,
        "recorded_at": _utc_now(),
        "case": _task_payload(task),
        "job_name": job_name,
        "command": command,
        "job_id": None,
        "status": "submitting",
        "error": None,
    }
    manifest["submissions"].append(record)
    manifest["submission_intent"] = index
    manifest["state"] = "submitting"
    path = campaign_evidence.campaign_run_manifest_path(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    common.serialization.atomic_write_json(path, manifest)
    try:
        job_id = _submit_case(
            command,
            git_commit=str(manifest["git_commit"]),
            run_id=str(manifest["campaign_run_id"]),
        )
    except subprocess.CalledProcessError as error:
        record["status"] = "submission_failed"
        record["error"] = error.stderr.strip() or str(error)
        manifest["submission_intent"] = None
        manifest["state"] = "submission_failed"
        common.serialization.atomic_write_json(path, manifest)
        raise
    record["job_id"] = job_id
    record["status"] = "submitted"
    manifest["slurm_job_ids"].append(job_id)
    manifest["submission_intent"] = None
    manifest["state"] = "active"
    common.serialization.atomic_write_json(path, manifest)
    return campaign_evidence.load_campaign_run(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )


def _recover_submission_gate(
    manifest: dict[str, Any],
    *,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], bool]:
    """Recover one durable intent or mark its scheduler identity unknown."""
    if manifest["submission_intent"] is None:
        return manifest, False
    recovered = _recover_submission_intent(
        manifest,
        storage_root=storage_root,
    )
    if recovered["submission_intent"] is None:
        return recovered, False
    recovered["state"] = "submission_unknown"
    common.serialization.atomic_write_json(
        campaign_evidence.campaign_run_manifest_path(
            str(recovered["campaign_run_id"]),
            storage_root=storage_root,
        ),
        recovered,
    )
    return recovered, True


def feed_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile exact jobs and fill the configured safe submission capacity."""
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage_root,
    )
    lock_path = run_directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        manifest = campaign_evidence.load_campaign_run(
            run_id,
            storage_root=storage_root,
        )
        if manifest["state"] in {"cancel_requested", "force_cancel_requested"}:
            return manifest
        current_commit = _repository_commit()
        if current_commit != manifest["git_commit"]:
            message = f"CPU checkout commit {current_commit} does not match run commit {manifest['git_commit']}."
            raise RuntimeError(message)
        campaign = campaign_evidence.campaign_from_manifest(manifest)
        manifest, submission_unknown = _recover_submission_gate(
            manifest,
            storage_root=storage_root,
        )
        if submission_unknown:
            return manifest
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        task_views, pending_jobs, running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
            compact_license_attempt_payloads=True,
        )
        successful = [view for view in task_views if view["state"] == "successful"]
        if len(successful) == len(task_views):
            _finalize_completed_batches(campaign, storage_root=storage_root)
            manifest["state"] = "complete"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest
        if any(view["state"] == "scheduler_unknown" for view in task_views):
            manifest["state"] = "scheduler_unknown"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest

        return _fill_submission_capacity(
            manifest,
            campaign,
            task_views,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            scheduler=scheduler,
            storage_root=storage_root,
        )


_LICENSE_ACQUIRED_RUNTIME_PHASES = frozenset(
    {
        "stationary_airflow",
        "transient_drying",
        "collecting_exports",
        "canonicalizing",
        "validating",
        "publishing",
    }
)
_LICENSE_RESOLVED_RUNTIME_PHASES = frozenset({"completed", "failed"})


def _runtime_proves_license_acquired(view: Mapping[str, Any]) -> bool:
    """Return whether existing progress proves license checkout and solver work."""
    runtime = view.get("runtime_progress")
    if not isinstance(runtime, dict) or runtime.get("availability") != "available":
        return False
    phase = runtime.get("phase")
    if phase in {"stationary_airflow", "transient_drying"}:
        return runtime.get("parser_state") == "available"
    return phase in _LICENSE_ACQUIRED_RUNTIME_PHASES


def _task_identity_key(view: Mapping[str, Any]) -> tuple[str, int]:
    """Return one logical campaign-case identity from a reconciled view."""
    return str(view["batch_id"]), int(view["case_index"])


def _view_consumes_admission(view: Mapping[str, Any]) -> bool:
    """Return whether one logical case currently occupies admission."""
    state = view.get("state")
    if state == "active":
        return not _runtime_proves_license_acquired(view)
    return state in {
        "admission_waiting",
        "pending",
        "scheduler_unknown",
        "license_blocked",
        "cancelled",
        "interrupted",
    }


def _submission_intent(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, int], str] | None:
    """Return the logical case and mode bound to one durable intent."""
    intent = manifest.get("submission_intent")
    if intent is None:
        return None
    if isinstance(intent, bool) or not isinstance(intent, int) or intent < 1:
        message = "Campaign submission intent is malformed."
        raise ValueError(message)
    submissions = manifest.get("submissions")
    if not isinstance(submissions, list) or intent > len(submissions):
        message = "Campaign submission intent has no durable submission record."
        raise ValueError(message)
    record = submissions[intent - 1]
    if not isinstance(record, dict) or record.get("submission_index") != intent:
        message = "Campaign submission intent does not match its durable record."
        raise ValueError(message)
    case = record.get("case")
    if not isinstance(case, dict):
        message = "Campaign submission intent lacks one logical case."
        raise TypeError(message)
    case_index = case.get("case_index")
    if isinstance(case_index, bool) or not isinstance(case_index, int):
        message = "Campaign submission intent has an invalid case index."
        raise TypeError(message)
    mode = record.get("mode")
    if mode not in {"initial", "resume", "license_retry"}:
        message = "Campaign submission intent has an invalid mode."
        raise ValueError(message)
    return (str(case.get("batch_id")), case_index), str(mode)


def _admission_reservation_keys(manifest: Mapping[str, Any]) -> frozenset[tuple[str, int]]:
    """Return logical cases covered by persisted admission reservations."""
    reservations = manifest.get("admission_reservations", [])
    if not isinstance(reservations, list):
        message = "Campaign admission reservations are malformed."
        raise TypeError(message)
    return frozenset((str(record["batch_id"]), int(record["case_index"])) for record in reservations)


def _logical_admission_case_keys(
    manifest: Mapping[str, Any],
    task_views: Sequence[Mapping[str, Any]],
) -> frozenset[tuple[str, int]]:
    """Reconstruct logical admission membership from durable case state."""
    keys = {_task_identity_key(view) for view in task_views if _view_consumes_admission(view)}
    keys.update(_admission_reservation_keys(manifest))
    intent = _submission_intent(manifest)
    if intent is not None:
        keys.add(intent[0])
    return frozenset(keys)


def _admission_summary(
    manifest: Mapping[str, Any],
    task_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return disjoint logical admission occupancy components."""
    categories: dict[str, set[tuple[str, int]]] = {
        "pending": set(),
        "starting": set(),
        "license_waiting": set(),
        "acquiring_license": set(),
    }
    for view in task_views:
        if not _view_consumes_admission(view):
            continue
        key = _task_identity_key(view)
        runtime = view.get("runtime_progress")
        acquiring = view.get("license_retry_active") is True or (
            view.get("state") == "active" and isinstance(runtime, dict) and runtime.get("phase") in {"starting_solver", "acquiring_comsol_license"}
        )
        if acquiring:
            category = "acquiring_license"
        elif view["state"] == "license_blocked":
            category = "license_waiting"
        elif view["state"] == "pending":
            category = "pending"
        else:
            category = "starting"
        categories[category].add(key)
    for key in _admission_reservation_keys(manifest):
        for values in categories.values():
            values.discard(key)
        categories["starting"].add(key)
    intent = _submission_intent(manifest)
    if intent is not None:
        key, mode = intent
        for values in categories.values():
            values.discard(key)
        categories["acquiring_license" if mode == "license_retry" else "starting"].add(key)
    components = {name: len(values) for name, values in categories.items()}
    admission_count = sum(components.values())
    expected = len(_logical_admission_case_keys(manifest, task_views))
    if admission_count != expected:
        message = "Campaign admission components do not match logical membership."
        raise RuntimeError(message)
    maximum = int(manifest["submission_config"]["max_admission_cases"])
    if admission_count > maximum:
        message = f"Campaign logical admission exceeds max_admission_cases: {admission_count} > {maximum}."
        raise RuntimeError(message)
    return {
        "count": admission_count,
        "maximum": maximum,
        "components": components,
    }


def _failure_population_counts(
    task_views: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Return precise disjoint failure and operational populations."""
    return {
        "solver_failed": sum(
            view["state"] == "failed" and view.get("failure_stage") == "solver" and view.get("temporary_license_retry") is None for view in task_views
        ),
        "technical_runtime_timed_out": sum(view["state"] == "timed_out" and view.get("temporary_license_retry") is None for view in task_views),
        "conversion_failed": sum(view["state"] == "conversion_failed" for view in task_views),
        "publication_failed": sum(view["state"] == "publication_failed" for view in task_views),
        "case_reconciliation_failed": sum(view["state"] == "case_reconciliation_failed" for view in task_views),
        "license_blocked": sum(view["state"] == "license_blocked" for view in task_views),
        "replay_blocked": sum(view.get("replay_blocked") is True for view in task_views),
    }


def _solver_failure_threshold_exceeded(
    task_views: Sequence[Mapping[str, Any]],
    *,
    maximum_failed_cases: int,
) -> bool:
    """Return whether solver failures and timeouts exceed the configured budget."""
    counts = _failure_population_counts(task_views)
    return counts["solver_failed"] + counts["technical_runtime_timed_out"] > maximum_failed_cases


def _normal_admission_status(
    manifest: Mapping[str, Any],
    task_views: Sequence[Mapping[str, Any]],
    *,
    running_jobs: int,
) -> tuple[bool, str | None]:
    """Return whether eligible fresh work is prevented from admission."""
    restart_available = any(view["state"] in {"cancelled", "interrupted"} for view in task_views)
    fresh_available = any(view["state"] == "never_started" for view in task_views)
    if not restart_available and not fresh_available:
        return False, None
    submission_config = manifest["submission_config"]
    if manifest.get("submission_intent") is not None:
        return True, "submission_intent_unresolved"
    max_running = submission_config["max_running_cases"]
    if max_running is not None and running_jobs >= int(max_running):
        return True, "max_running_cases_reached"
    if fresh_available and len(_logical_admission_case_keys(manifest, task_views)) >= int(submission_config["max_admission_cases"]):
        return True, "max_admission_cases_reached"
    if (
        fresh_available
        and not restart_available
        and _solver_failure_threshold_exceeded(
            task_views,
            maximum_failed_cases=int(submission_config["maximum_failed_cases"]),
        )
    ):
        return True, "solver_failure_threshold_exceeded"
    return False, None


def _parse_utc_timestamp(value: str, *, label: str) -> datetime:
    """Return one timezone-aware UTC timestamp from durable evidence."""
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        message = f"{label} is not an ISO timestamp: {value!r}."
        raise ValueError(message) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        message = f"{label} must be timezone-aware: {value!r}."
        raise ValueError(message)
    return timestamp.astimezone(timezone.utc)


def _retry_candidate_sort_key(
    view: Mapping[str, Any],
    *,
    plan_order: Mapping[tuple[str, int], int],
) -> tuple[datetime, datetime, int, str, int]:
    """Return the stable due-first campaign order for eligible license retries."""
    next_retry_at = view.get("license_next_retry_at")
    first_blocked_at = view.get("license_first_blocked_at")
    if not isinstance(next_retry_at, str) or not isinstance(first_blocked_at, str):
        message = "Eligible temporary-license retry lacks durable due and first-blocked timestamps."
        raise TypeError(message)
    return (
        _parse_utc_timestamp(next_retry_at, label="Temporary-license next retry timestamp"),
        _parse_utc_timestamp(first_blocked_at, label="Temporary-license first blocked timestamp"),
        plan_order[_task_identity_key(view)],
        str(view["batch_id"]),
        int(view["case_index"]),
    )


def _reserve_initial_launch(
    manifest: dict[str, Any],
    task: cluster_service.CampaignTask,
    admission_keys: set[tuple[str, int]],
    reservation_keys: frozenset[tuple[str, int]],
    *,
    maximum: int,
) -> None:
    """Durably admit one fresh case before its Slurm submission."""
    key = task.batch_id, task.case_index
    if key in admission_keys:
        if key not in reservation_keys:
            message = "Fresh case is admitted without a controller-side launch reservation."
            raise RuntimeError(message)
        return
    if len(admission_keys) >= maximum:
        message = "Fresh case admission would exceed the logical admission limit."
        raise RuntimeError(message)
    reservations = manifest.setdefault("admission_reservations", [])
    if not isinstance(reservations, list):
        message = "Campaign admission reservations are malformed."
        raise TypeError(message)
    reservations.append(_task_payload(task))
    admission_keys.add(key)


def _release_initial_launch_reservation(
    manifest: dict[str, Any],
    *,
    key: tuple[str, int],
) -> None:
    """Release one reservation as its durable submission intent takes over."""
    manifest["admission_reservations"] = [
        record for record in manifest.get("admission_reservations", []) if (str(record["batch_id"]), int(record["case_index"])) != key
    ]


def _fill_submission_capacity(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    task_views: Sequence[Mapping[str, Any]],
    *,
    pending_jobs: int,
    running_jobs: int,
    scheduler: Mapping[str, Any],
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Submit owned retries and admit fresh cases within one logical pool."""
    del scheduler
    run_id = str(manifest["campaign_run_id"])
    max_admission_cases = int(manifest["submission_config"]["max_admission_cases"])
    max_running = manifest["submission_config"]["max_running_cases"]
    maximum_failed_cases = int(manifest["submission_config"]["maximum_failed_cases"])
    admission_keys = set(_logical_admission_case_keys(manifest, task_views))
    selected_tasks: set[tuple[str, int]] = set()
    plan_order = {(task.batch_id, task.case_index): index for index, task in enumerate(cluster_service.campaign_tasks(campaign))}

    while True:
        if max_running is not None and running_jobs >= int(max_running):
            manifest["state"] = "active"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest

        eligible_blocked = sorted(
            (
                view
                for view in task_views
                if view["state"] == "license_blocked" and view["license_retry_eligible"] is True and _task_identity_key(view) not in selected_tasks
            ),
            key=lambda view: _retry_candidate_sort_key(
                view,
                plan_order=plan_order,
            ),
        )
        next_retry = eligible_blocked[0] if eligible_blocked else None
        next_restart = next(
            (view for view in task_views if view["state"] in {"cancelled", "interrupted"} and _task_identity_key(view) not in selected_tasks),
            None,
        )
        reservation_keys = _admission_reservation_keys(manifest)
        next_reserved = next(
            (
                view
                for view in task_views
                if view["state"] in {"admission_waiting", "never_started"}
                and _task_identity_key(view) in reservation_keys
                and _task_identity_key(view) not in selected_tasks
            ),
            None,
        )
        next_unsent = next(
            (
                view
                for view in task_views
                if view["state"] == "never_started"
                and _task_identity_key(view) not in reservation_keys
                and _task_identity_key(view) not in selected_tasks
            ),
            None,
        )
        circuit_breaker_tripped = _solver_failure_threshold_exceeded(
            task_views,
            maximum_failed_cases=maximum_failed_cases,
        )
        next_view = next_retry or next_restart or next_reserved
        if next_view is None and not circuit_breaker_tripped and len(admission_keys) < max_admission_cases:
            next_view = next_unsent
        if next_view is None:
            if pending_jobs or running_jobs:
                manifest["state"] = "active"
            elif any(view["state"] == "license_blocked" for view in task_views):
                manifest["state"] = "license_blocked"
            elif admission_keys:
                manifest["state"] = "active"
            elif circuit_breaker_tripped and next_unsent is not None:
                manifest["state"] = "failure_threshold_reached"
            else:
                manifest["state"] = "completed_with_failures"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest

        task = _task_from_payload(campaign, next_view)
        key = (task.batch_id, task.case_index)
        selected_tasks.add(key)
        if next_view["state"] == "license_blocked":
            mode = "license_retry"
        elif next_view["state"] in {"cancelled", "interrupted"}:
            mode = "resume"
        else:
            mode = "initial"
        if mode == "initial":
            _reserve_initial_launch(
                manifest,
                task,
                admission_keys,
                reservation_keys,
                maximum=max_admission_cases,
            )
        elif key not in admission_keys:
            message = "Retry or resume case no longer owns logical admission."
            raise RuntimeError(message)

        current_commit = _repository_commit()
        if current_commit != manifest["git_commit"]:
            message = f"Solver admission requires the exact original campaign source commit {manifest['git_commit']}, got {current_commit}."
            raise RuntimeError(message)
        if mode == "initial":
            _release_initial_launch_reservation(manifest, key=key)
        manifest = _submit_one(
            manifest,
            campaign,
            task,
            mode=mode,
            storage_root=storage_root,
        )


def submit_campaign(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create durable campaign state and fill its safe submission capacity."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    input_service.prepare_campaign_inputs(
        campaign,
        git_commit=requested_commit,
        storage_root=storage_root,
    )
    run_id = campaign_run_id(campaign, git_commit=requested_commit)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root)
    run_directory.mkdir(parents=True, exist_ok=True)
    scheduler_log_directory = run_directory / "scheduler"
    scheduler_log_directory.mkdir(exist_ok=True)
    path = run_directory / "campaign_run.json"
    lock_path = run_directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        intent = _new_campaign_manifest(
            campaign,
            run_id=run_id,
            requested_commit=requested_commit,
            run_directory=run_directory,
            scheduler_log_directory=scheduler_log_directory,
            storage_root=storage_root,
        )
        if path.exists():
            existing = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
            immutable_keys = set(intent).difference(
                {
                    "dataset_packages",
                    "slurm_job_ids",
                    "submissions",
                    "submission_intent",
                    "admission_reservations",
                    "state",
                }
            )
            if any(existing[key] != intent[key] for key in immutable_keys):
                message = f"Existing campaign-run manifest conflicts with {run_id!r}."
                raise FileExistsError(message)
        else:
            common.serialization.atomic_write_json(path, intent)
    return feed_campaign(run_id, storage_root=storage_root)


def _write_campaign_manifest(
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist and re-admit one campaign manifest state transition."""
    run_id = str(manifest["campaign_run_id"])
    common.serialization.atomic_write_json(
        campaign_evidence.campaign_run_manifest_path(
            run_id,
            storage_root=storage_root,
        ),
        dict(manifest),
    )
    return campaign_evidence.load_campaign_run(
        run_id,
        storage_root=storage_root,
    )


def resume_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fill scheduler capacity first, then process independent local replays."""
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage_root,
    )
    lock_path = run_directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        manifest = campaign_evidence.load_campaign_run(
            run_id,
            storage_root=storage_root,
        )
        manifest, submission_unknown = _recover_submission_gate(
            manifest,
            storage_root=storage_root,
        )
        if submission_unknown:
            return manifest
        campaign = campaign_evidence.campaign_from_manifest(manifest)
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        task_views, pending_jobs, running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
            compact_license_attempt_payloads=True,
        )
        if all(view["state"] == "successful" for view in task_views):
            _finalize_completed_batches(
                campaign,
                storage_root=storage_root,
            )
            manifest["state"] = "complete"
            return _write_campaign_manifest(
                manifest,
                storage_root=storage_root,
            )

        submission_count_before = len(manifest.get("submissions", ()))
        manifest = _fill_submission_capacity(
            manifest,
            campaign,
            task_views,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            scheduler=scheduler,
            storage_root=storage_root,
        )
        admission_active = pending_jobs > 0 or running_jobs > 0 or len(manifest.get("submissions", ())) > submission_count_before

        replayed = False
        replay_failed = False
        replay_views = [
            view
            for view in task_views
            if view["state"] in {"conversion_failed", "publication_failed"}
            and view.get("replay_eligible", view.get("postprocessing_replay_available")) is True
        ]
        for replay_view in replay_views:
            batch = campaign.batch(str(replay_view["batch_name"]))
            try:
                outcome = batch_runtime.replay_case_postprocessing(
                    batch,
                    int(replay_view["case_index"]),
                    source_campaign_run_id=str(replay_view["attempt_campaign_run_id"]),
                    storage_root=storage_root,
                )
            except (
                batch_runtime.CaseCleanupError,
                batch_runtime.CaseLocalReplayError,
            ):
                replay_failed = True
                continue
            if outcome.status in {"replayed", "skipped"}:
                replayed = True
            else:
                replay_failed = True

        if replayed:
            _finalize_completed_batches(
                campaign,
                storage_root=storage_root,
            )
        all_successful = all(
            batch_runtime.completed_case_is_valid(
                batch_config,
                case_index,
                storage_root=storage_root,
            )
            for batch_config in campaign.batches
            for case_index in batch_config.case_indices
        )
        if all_successful:
            manifest["state"] = "complete"
        elif admission_active:
            manifest["state"] = "active"
        elif replay_failed and manifest["state"] not in {
            "license_blocked",
            "failure_threshold_reached",
        }:
            manifest["state"] = "completed_with_failures"
        return _write_campaign_manifest(
            manifest,
            storage_root=storage_root,
        )


def cancel_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Persist and then request graceful or force cancellation of exact jobs."""
    if not isinstance(force, bool):
        message = "force must be boolean."
        raise TypeError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage_root,
    )
    lock_path = run_directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        manifest = campaign_evidence.load_campaign_run(
            run_id,
            storage_root=storage_root,
        )
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        active_ids = [job_id for job_id in manifest["slurm_job_ids"] if job_id in scheduler["active"]]
        pending_ids = [job_id for job_id in active_ids if _scheduler_state(scheduler["active"][job_id][1]) == _ACTIVE_PENDING_STATE]
        running_ids = [job_id for job_id in active_ids if job_id not in pending_ids]
        commands: list[list[str]] = []
        if force and active_ids:
            commands.append(["scancel", "--signal=KILL", "--full", *active_ids])
        elif not force:
            if pending_ids:
                commands.append(["scancel", *pending_ids])
            if running_ids:
                commands.append(["scancel", "--signal=TERM", "--batch", *running_ids])

        receipt_path = run_directory / "cancellations.json"
        existing: list[dict[str, Any]] = []
        if receipt_path.exists():
            raw = campaign_evidence.load_json_object(
                receipt_path,
                label="campaign cancellation receipt",
            )
            if (
                raw.get("schema_kind") != "generation_campaign_cancellations"
                or raw.get("schema_version") != 1
                or raw.get("campaign_run_id") != run_id
                or not isinstance(raw.get("attempts"), list)
            ):
                message = f"Campaign cancellation receipt is malformed: {receipt_path}"
                raise ValueError(message)
            existing = raw["attempts"]
        attempt = {
            "recorded_at": _utc_now(),
            "mode": "force" if force else "graceful",
            "pending_job_ids": pending_ids,
            "running_job_ids": running_ids,
            "commands": [],
        }
        receipt = {
            "schema_kind": "generation_campaign_cancellations",
            "schema_version": 1,
            "campaign_run_id": run_id,
            "attempts": [*existing, attempt],
        }
        common.serialization.atomic_write_json(receipt_path, receipt)
        manifest["state"] = "force_cancel_requested" if force else "cancel_requested"
        _write_campaign_manifest(manifest, storage_root=storage_root)

        command_results: list[dict[str, Any]] = []
        failed_commands: list[str] = []
        for command in commands:
            result = subprocess.run(  # noqa: S603 -- persisted numeric Slurm IDs only
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            command_results.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
            )
            if result.returncode != 0:
                failed_commands.append(result.stderr.strip() or f"{command[0]} exit {result.returncode}")
        attempt["commands"] = command_results
        common.serialization.atomic_write_json(receipt_path, receipt)
        if failed_commands:
            message = f"Slurm cancellation failed after its request was persisted: {failed_commands}"
            raise RuntimeError(message)
        return receipt


def campaign_accounting(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact squeue and sacct commands and their current output."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    evidence = _scheduler_evidence(manifest["slurm_job_ids"])
    return {
        "campaign_run_id": run_id,
        "squeue": evidence["squeue"],
        "sacct": evidence["sacct"],
    }


def record_worker_interruption(
    run_id: str,
    *,
    storage_root: Path | str | None,
    signal_name: str,
    exit_code: int,
) -> Path:
    """Persist one best-effort per-case Slurm interruption receipt."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Worker interruption evidence requires SLURM_JOB_ID."
        raise ValueError(message)
    directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / "interruptions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.json"
    payload = {
        "schema_kind": "generation_worker_interruption",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "recorded_at": _utc_now(),
        "signal": signal_name,
        "exit_code": exit_code,
        "slurm_job_id": job_id,
        "hostname": os.uname().nodename,
    }
    common.serialization.atomic_write_json(path, payload)
    return path


def _batch_status(
    batch: config_service.GenerationConfig,
    task_views: Sequence[Mapping[str, Any]],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return persistent per-batch case counts and terminal evidence."""
    selected = [view for view in task_views if view["batch_name"] == batch.batch_name]
    state_root = batch_runtime._state_batch_root(  # noqa: SLF001
        batch,
        storage_root=storage_root,
    )
    quarantine = state_root / "quarantine"
    terminal_path = (
        batch_runtime.batch_meta_directory(
            batch,
            storage_root=storage_root,
        )
        / "batch_manifest.json"
    )
    failure_counts = _failure_population_counts(selected)
    return {
        "batch_name": batch.batch_name,
        "batch_id": batch.batch_id,
        "planned": len(selected),
        "successful": sum(view["state"] == "successful" for view in selected),
        "active": sum(view["state"] == "active" for view in selected),
        "pending": sum(view["state"] == "pending" for view in selected),
        "admission_waiting": sum(view["state"] == "admission_waiting" for view in selected),
        "never_started": sum(view["state"] == "never_started" for view in selected),
        "solver_failed": failure_counts["solver_failed"],
        "timed_out": failure_counts["technical_runtime_timed_out"],
        "exports_failed": sum(view["state"] == "exports_failed" for view in selected),
        "conversion_failed": failure_counts["conversion_failed"],
        "publication_failed": failure_counts["publication_failed"],
        "case_reconciliation_failed": failure_counts["case_reconciliation_failed"],
        "replay_blocked": failure_counts["replay_blocked"],
        "cancelled": sum(view["state"] == "cancelled" for view in selected),
        "interrupted": sum(view["state"] == "interrupted" for view in selected),
        "quality_flagged": sum(int(view["quality_flag_count"]) > 0 for view in selected),
        "license_blocked": sum(view["state"] == "license_blocked" for view in selected),
        "license_retry_eligible": sum(view["state"] == "license_blocked" and view["license_retry_eligible"] is True for view in selected),
        "scheduler_unknown": sum(view["state"] == "scheduler_unknown" for view in selected),
        "quarantined": (len(tuple(quarantine.iterdir())) if quarantine.is_dir() else 0),
        "terminal_manifest": str(terminal_path),
        "terminal_manifest_available": terminal_path.is_file(),
    }


def _public_task_view(view: Mapping[str, Any]) -> dict[str, Any]:
    """Return one work unit in the shared public state vocabulary."""
    classified_state = str(view["state"])
    if classified_state == "active":
        state = "running"
    elif classified_state == "pending":
        state = "scheduler_pending"
    elif classified_state in {
        "successful",
        "license_blocked",
        "admission_waiting",
        "never_started",
    }:
        state = classified_state
    else:
        state = "failed"
    return {**view, "state": state, "classified_state": classified_state}


def _pilot_license_observability(
    campaign: config_service.CampaignConfig,
    task_views: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Project pilot solver concurrency and license waits from existing evidence."""
    intervals: list[tuple[datetime, datetime]] = []
    successful_starts = 0
    blocked_submissions = 0
    accumulated_wait = 0.0
    accumulated_probe = 0.0
    features: set[str] = set()
    error_codes: set[str] = set()
    by_identity = {(str(view["batch_id"]), int(view["case_index"])): view for view in task_views}
    for batch in campaign.batches:
        for case_index in batch.case_indices:
            view = by_identity[(batch.batch_id, case_index)]
            if view["state"] == "successful":
                execution_path = (
                    batch_runtime.processed_case_directory(
                        batch,
                        case_index,
                        storage_root=storage_root,
                    )
                    / "execution_provenance.json"
                )
                execution = json.loads(execution_path.read_text(encoding="utf-8"))
                result = execution.get("result") if isinstance(execution, dict) else None
                if not isinstance(result, dict):
                    message = f"Pilot execution provenance is malformed: {execution_path}"
                    raise ValueError(message)
                started = result.get("started_at")
                ended = result.get("ended_at")
                if isinstance(started, str) and isinstance(ended, str):
                    start = datetime.fromisoformat(started).astimezone(timezone.utc)
                    end = datetime.fromisoformat(ended).astimezone(timezone.utc)
                    if end < start:
                        message = f"Pilot solver interval is negative: {execution_path}"
                        raise ValueError(message)
                    intervals.append((start, end))
                    successful_starts += 1
            wait = license_service.load_temporary_license_wait(
                batch,
                case_index,
                campaign_run_id=run_id,
                storage_root=storage_root,
            )
            if wait is None:
                continue
            retry_count = int(wait["retry_count"])
            blocked_submissions += retry_count
            accumulated_wait += float(wait["cumulative_wait_seconds"])
            features.add(str(wait["feature"]))
            if wait["error_code"] is not None:
                error_codes.add(str(wait["error_code"]))
            attempt_root = common.paths.resolve_generation_attempt_case_directory(
                batch.batch_storage_name,
                batch.case_id(case_index),
                run_id,
                storage_root=storage_root,
            )
            for attempt_index in range(1, retry_count + 1):
                receipt_path = attempt_root / f"attempt_{attempt_index:04d}" / "attempt.json"
                if not receipt_path.is_file() or receipt_path.is_symlink():
                    continue
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                elapsed = receipt.get("elapsed_seconds") if isinstance(receipt, dict) else None
                if (
                    isinstance(receipt, dict)
                    and receipt.get("campaign_run_id") == run_id
                    and receipt.get("batch_id") == batch.batch_id
                    and receipt.get("case_id") == batch.case_id(case_index)
                    and receipt.get("case_state") == "license_blocked"
                    and receipt.get("attempt_index") == attempt_index
                    and isinstance(receipt.get("job_id"), str)
                    and _JOB_ID_PATTERN.fullmatch(str(receipt["job_id"])) is not None
                    and not isinstance(elapsed, bool)
                    and isinstance(elapsed, (int, float))
                    and float(elapsed) >= 0.0
                ):
                    accumulated_probe += float(elapsed)
    events = sorted(
        [(start, 1) for start, _end in intervals] + [(end, -1) for _start, end in intervals],
        key=lambda item: (item[0], item[1]),
    )
    concurrent = 0
    peak = 0
    for _timestamp, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
    return {
        "observed_peak_solver_concurrency": peak,
        "successful_solver_start_count": successful_starts,
        "license_blocked_submission_count": blocked_submissions,
        "accumulated_license_wait_seconds": accumulated_wait,
        "accumulated_license_probe_seconds": accumulated_probe,
        "detected_license_features": sorted(features),
        "detected_license_error_codes": sorted(error_codes),
        "observed_license_concurrency_lower_bound": peak,
    }


def campaign_status(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Reconstruct exact feeder, scheduler, batch, and case state."""
    manifest = campaign_evidence.load_campaign_run(
        run_id,
        storage_root=storage_root,
    )
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    scheduler = (
        _scheduler_evidence(manifest["slurm_job_ids"])
        if query_scheduler
        else {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
            "active": {},
            "accounted": {},
        }
    )
    if query_scheduler:
        _require_scheduler_evidence(scheduler)
    task_views, pending_jobs, running_jobs = _reconciled(
        manifest,
        campaign,
        scheduler,
        storage_root=storage_root,
    )
    batches = [
        _batch_status(
            batch,
            task_views,
            storage_root=storage_root,
        )
        for batch in campaign.batches
    ]
    failure_counts = _failure_population_counts(task_views)
    count_states = {
        "planned": len(task_views),
        "successful": sum(view["state"] == "successful" for view in task_views),
        "active": sum(view["state"] == "active" for view in task_views),
        "pending": sum(view["state"] == "pending" for view in task_views),
        "admission_waiting": sum(view["state"] == "admission_waiting" for view in task_views),
        "never_started": sum(view["state"] == "never_started" for view in task_views),
        "solver_failed": failure_counts["solver_failed"],
        "timed_out": failure_counts["technical_runtime_timed_out"],
        "exports_failed": sum(view["state"] == "exports_failed" for view in task_views),
        "conversion_failed": failure_counts["conversion_failed"],
        "publication_failed": failure_counts["publication_failed"],
        "case_reconciliation_failed": failure_counts["case_reconciliation_failed"],
        "cancelled": sum(view["state"] == "cancelled" for view in task_views),
        "interrupted": sum(view["state"] == "interrupted" for view in task_views),
        "quality_flagged": sum(int(view["quality_flag_count"]) > 0 for view in task_views),
        "license_blocked": failure_counts["license_blocked"],
        "replay_blocked": failure_counts["replay_blocked"],
    }
    unknown_cases = sum(view["state"] == "scheduler_unknown" for view in task_views)
    license_retry_eligible_cases = sum(view["state"] == "license_blocked" and view["license_retry_eligible"] is True for view in task_views)
    replayable_cases = sum(view.get("replay_eligible", view.get("postprocessing_replay_available")) is True for view in task_views)
    unresolved_failure_states = {
        "failed",
        "timed_out",
        "exports_failed",
        "conversion_failed",
        "publication_failed",
        "case_reconciliation_failed",
    }
    unresolved_cases = sum(view["state"] in unresolved_failure_states for view in task_views)
    unresolved_solver_failures = failure_counts["solver_failed"] + failure_counts["technical_runtime_timed_out"]
    maximum_failed_cases = int(manifest["submission_config"]["maximum_failed_cases"])
    circuit_breaker_tripped = _solver_failure_threshold_exceeded(
        task_views,
        maximum_failed_cases=maximum_failed_cases,
    )
    admission_blocked, admission_block_reason = _normal_admission_status(
        manifest,
        task_views,
        running_jobs=running_jobs,
    )
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage_root,
    )
    transfer_receipt = (run_directory / "transfer_complete.json").is_file()
    publication_complete = all(batch["terminal_manifest_available"] for batch in batches)
    has_scheduler_activity = bool(pending_jobs or running_jobs or count_states["active"] or count_states["pending"])
    all_successful = count_states["successful"] == count_states["planned"]
    cancellation_requested = manifest["state"] in {
        "cancel_requested",
        "force_cancel_requested",
    }

    if has_scheduler_activity:
        state = "running"
        next_command = f"status {run_id}"
    elif manifest["submission_intent"] is not None or unknown_cases:
        state = "submission_pending_or_unknown"
        next_command = f"status {run_id}"
    elif count_states["license_blocked"]:
        state = "license_blocked"
        next_command = f"status {run_id}"
    elif all_successful and publication_complete:
        state = "transfer_complete" if transfer_receipt else "successful"
        next_command = f"status {run_id}" if transfer_receipt else f"collect {run_id}"
    elif cancellation_requested:
        state = "cancelled"
        next_command = f"resume {run_id}"
    elif (
        count_states["never_started"] or count_states["cancelled"] or count_states["interrupted"] or replayable_cases or license_retry_eligible_cases
    ):
        state = "feeding"
        next_command = f"resume {run_id}"
    elif unresolved_cases:
        state = "completed_with_failures"
        next_command = f"resume {run_id}"
    else:
        state = "feeding"
        next_command = f"feed-campaign {run_id}"

    failed_cases = unresolved_cases
    cases_per_material = len(campaign.batches[0].case_indices) if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE else None
    public_task_views = [_public_task_view(view) for view in task_views]
    admission = _admission_summary(manifest, task_views)
    license_observability = (
        _pilot_license_observability(
            campaign,
            task_views,
            run_id=run_id,
            storage_root=storage_root,
        )
        if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE
        else None
    )
    work_unit_counts = {
        state: sum(view["state"] == state for view in public_task_views)
        for state in (
            "successful",
            "running",
            "scheduler_pending",
            "license_blocked",
            "admission_waiting",
            "never_started",
            "failed",
        )
    }
    work_unit_counts["total"] = len(public_task_views)
    return {
        "campaign_run_id": run_id,
        "campaign_state": state,
        "campaign_purpose": campaign.campaign_purpose,
        "cases_per_material": cases_per_material,
        "git_commit": manifest["git_commit"],
        "execution_config_digest": manifest.get("execution_config_digest"),
        "slurm_job_ids": manifest["slurm_job_ids"],
        "scheduler_job_name": manifest["scheduler_job_name"],
        "scheduler_log_directory": manifest["scheduler_log_directory"],
        "submission_config": manifest["submission_config"],
        "counts": count_states,
        "planned_cases": count_states["planned"],
        "completed_cases": count_states["successful"],
        "failed_cases": failed_cases,
        "pending_jobs": pending_jobs,
        "running_jobs": running_jobs,
        "unsent_cases": count_states["never_started"],
        "unknown_cases": unknown_cases,
        "license_blocked_cases": count_states["license_blocked"],
        "status_artifact_recovery_count": sum(int(view.get("status_artifact_recovery_count", 0)) for view in task_views),
        "admission": admission,
        "license_retry_eligible_cases": license_retry_eligible_cases,
        "postprocessing_replay_available_cases": replayable_cases,
        "replay_blocked_cases": failure_counts["replay_blocked"],
        "failure_counts": failure_counts,
        "unresolved_solver_failed_cases": unresolved_solver_failures,
        "admission_blocked": admission_blocked,
        "admission_block_reason": admission_block_reason,
        "maximum_failed_cases": maximum_failed_cases,
        "failure_circuit_breaker_tripped": circuit_breaker_tripped,
        "cancellation_requested": cancellation_requested,
        "squeue": manifest_scheduler_view(scheduler["squeue"]),
        "sacct": manifest_scheduler_view(scheduler["sacct"]),
        "batches": batches,
        "cases": public_task_views,
        "work_unit_counts": work_unit_counts,
        "license_observability": license_observability,
        "remote_storage_root": manifest["remote_storage_root"],
        "campaign_meta_directory": manifest["campaign_meta_directory"],
        "suggested_next_command": next_command,
    }


def manifest_scheduler_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return scheduler output without duplicating parsed internal maps."""
    return {
        "command": list(value["command"]),
        "output": value["output"],
        "error": value["error"],
    }


def run_campaign_case_job(
    run_id: str,
    batch_name: str,
    case_index: int,
    *,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> batch_runtime.CaseRunOutcome:
    """Execute the exact campaign case bound to the current Slurm allocation."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    current_commit = _repository_commit()
    if current_commit != manifest["git_commit"] or source_service.required_git_commit() != manifest["git_commit"]:
        message = "Campaign case worker checkout does not match its persisted run commit."
        raise RuntimeError(message)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    task = cluster_service.require_campaign_task(
        campaign,
        batch_name=batch_name,
        case_index=case_index,
    )
    cores = int(manifest["submission_config"]["cores_per_case"])
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    job_id = os.environ.get("SLURM_JOB_ID")
    if allocated is None or not allocated.isdigit() or int(allocated) != cores:
        message = f"Campaign case allocation must equal {cores} cores, got {allocated!r}."
        raise RuntimeError(message)
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Campaign case execution requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    matches = [record for record in manifest["submissions"] if record["job_id"] == job_id and record["case"] == _task_payload(task)]
    if len(matches) != 1:
        message = "Current Slurm job is not bound to this exact campaign case."
        raise RuntimeError(message)
    return cluster_service.run_campaign_case(
        campaign,
        task,
        cores_per_case=cores,
        scheduler_kind="slurm",
        storage_root=storage_root,
        work_root=work_root,
    )


def finalize_campaign_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish immutable terminal campaign evidence after all cases validate."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
        pilot_service.validate_pilot_terminal(
            run_id,
            storage_root=storage_root,
        )
        return campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / "campaign_terminal.json"
    if not manifest["slurm_job_ids"]:
        campaign_status(run_id, storage_root=storage_root)
        manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
        if not manifest["slurm_job_ids"]:
            message = f"Campaign scheduler identity has not been recovered for {run_id!r}."
            raise RuntimeError(message)
    batch_manifests: list[dict[str, Any]] = []
    for batch in campaign.batches:
        batch_manifest = batch_runtime.validate_terminal_batch(
            batch,
            storage_root=storage_root,
            validation_depth="routine",
        )
        path = batch_runtime.batch_meta_directory(batch, storage_root=storage_root) / "batch_manifest.json"
        if batch_manifest["git_commit"] != manifest["git_commit"]:
            message = f"Batch {batch.batch_id!r} was produced by a different Git commit."
            raise RuntimeError(message)
        batch_manifests.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "manifest_sha256": common.serialization.file_sha256(path),
                "case_count": len(batch_manifest["cases"]),
            }
        )
    terminal = {
        "schema_kind": "generation_campaign_terminal",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_config": manifest["campaign_config"],
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "git_commit": manifest["git_commit"],
        "slurm_job_ids": manifest["slurm_job_ids"],
        "scheduler_job_name": manifest["scheduler_job_name"],
        "scheduler_log_directory": manifest["scheduler_log_directory"],
        "batches": batch_manifests,
        "dataset_packages": list(campaign.dataset_packages),
    }
    path = campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root) / "campaign_terminal.json"
    serialized = json.dumps(terminal, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            message = f"Existing campaign terminal evidence conflicts for {run_id!r}."
            raise FileExistsError(message)
    else:
        common.serialization.atomic_write_text(path, serialized)
    return path


def validate_terminal_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate terminal campaign evidence and every referenced batch."""
    terminal_path = finalize_campaign_run(run_id, storage_root=storage_root)
    return campaign_evidence.load_json_object(terminal_path, label="terminal campaign manifest")


def campaign_transfer_plan(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return terminally validated storage-relative directories for collection."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
        return pilot_service.pilot_transfer_plan(
            run_id,
            storage_root=storage_root,
        )
    validate_terminal_campaign(run_id, storage_root=storage_root)
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()

    def relative_directory(directory: Path) -> str:
        resolved = directory.resolve()
        if not resolved.is_dir() or resolved.is_symlink():
            message = f"Transfer source is missing or unsafe: {resolved}."
            raise FileNotFoundError(message)
        try:
            relative = resolved.relative_to(storage)
        except ValueError as error:
            message = f"Transfer source escapes the storage root: {resolved}."
            raise ValueError(message) from error
        return relative.as_posix()

    def attempt_directories(
        batch: config_service.GenerationConfig,
    ) -> list[str]:
        directories: list[str] = []
        for case_index in batch.case_indices:
            candidate = common.paths.resolve_generation_attempt_case_directory(
                batch.batch_storage_name,
                batch.case_id(case_index),
                run_id,
                storage_root=storage,
            )
            if candidate.is_symlink():
                message = f"Attempt transfer source is unsafe: {candidate}."
                raise ValueError(message)
            if not candidate.exists():
                continue
            directories.append(relative_directory(candidate))
        return directories

    return {
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "git_commit": manifest["git_commit"],
        "campaign_config": manifest["campaign_config"],
        "campaign_directory": relative_directory(campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root)),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_count": len(batch.case_indices),
                "meta_directory": relative_directory(batch_runtime.batch_meta_directory(batch, storage_root=storage_root)),
                "raw_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_storage_name,
                        stage="raw",
                        storage_root=storage_root,
                    )
                ),
                "processed_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_storage_name,
                        stage="processed",
                        storage_root=storage_root,
                    )
                ),
                "attempt_directories": attempt_directories(batch),
            }
            for batch in campaign.batches
        ],
    }


def _validate_partial_campaign_evidence(
    run_id: str,
    evidence: Mapping[str, Any],
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Bind partial status evidence to the exact launch and configured cases."""
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "campaign_id",
        "git_commit",
        "campaign_state",
        "successful_cases",
        "failed_cases",
        "resume_command",
        "recorded_at",
        "transfer_plan",
    }
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    plan = evidence.get("transfer_plan")
    successful = evidence.get("successful_cases")
    failed = evidence.get("failed_cases")
    case_keys = {
        "batch_name",
        "batch_id",
        "case_id",
        "case_index",
        "state",
        "classified_state",
    }
    expected_cases = {
        (batch.batch_name, batch.batch_id, batch.case_id(case_index), case_index) for batch in campaign.batches for case_index in batch.case_indices
    }
    records = [*successful, *failed] if isinstance(successful, list) and isinstance(failed, list) else []
    observed_cases = {
        (
            record.get("batch_name"),
            record.get("batch_id"),
            record.get("case_id"),
            record.get("case_index"),
        )
        for record in records
        if isinstance(record, dict)
    }
    expected_batches = [
        {
            "batch_name": batch.batch_name,
            "batch_id": batch.batch_id,
            "case_count": len(batch.case_indices),
        }
        for batch in campaign.batches
    ]
    plan_batches = plan.get("batches") if isinstance(plan, dict) else None
    if (
        set(evidence) != required
        or evidence.get("schema_kind") != "generation_campaign_partial"
        or evidence.get("schema_version") != 1
        or evidence.get("campaign_run_id") != run_id
        or evidence.get("campaign_id") != campaign.campaign_id
        or evidence.get("git_commit") != manifest["git_commit"]
        or evidence.get("campaign_state") != "completed_with_failures"
        or evidence.get("resume_command") != f"resume {run_id}"
        or not isinstance(evidence.get("recorded_at"), str)
        or not evidence["recorded_at"]
        or not isinstance(plan, dict)
        or set(plan)
        != {
            "campaign_run_id",
            "campaign_name",
            "git_commit",
            "campaign_config",
            "campaign_directory",
            "batches",
        }
        or plan.get("campaign_run_id") != run_id
        or plan.get("campaign_name") != campaign.campaign_name
        or plan.get("git_commit") != manifest["git_commit"]
        or plan.get("campaign_config") != manifest["campaign_config"]
        or not isinstance(plan.get("campaign_directory"), str)
        or not isinstance(plan_batches, list)
        or len(plan_batches) != len(expected_batches)
        or any(
            not isinstance(batch, dict)
            or set(batch)
            != {
                "batch_name",
                "batch_id",
                "case_count",
                "meta_directory",
                "raw_directory",
                "processed_directory",
                "attempt_directories",
            }
            for batch in plan_batches
        )
        or [
            {
                "batch_name": batch.get("batch_name"),
                "batch_id": batch.get("batch_id"),
                "case_count": batch.get("case_count"),
            }
            for batch in plan_batches
            if isinstance(batch, dict)
        ]
        != expected_batches
        or not isinstance(successful, list)
        or not successful
        or not isinstance(failed, list)
        or not failed
        or any(not isinstance(record, dict) or set(record) != case_keys for record in records)
        or observed_cases != expected_cases
        or len(observed_cases) != len(records)
        or any(record["state"] != "successful" or record["classified_state"] != "successful" for record in successful)
        or any(record["state"] != "failed" or record["classified_state"] == "successful" for record in failed)
    ):
        message = "Partial campaign evidence conflicts with the launch campaign."
        raise ValueError(message)
    batches_by_name = {batch.batch_name: batch for batch in campaign.batches}
    for record in successful:
        batch = batches_by_name[str(record["batch_name"])]
        if not batch_runtime.completed_case_is_valid(
            batch,
            int(record["case_index"]),
            storage_root=storage_root,
        ):
            message = f"Successful case publication is invalid: {record['case_id']!r}."
            raise RuntimeError(message)
    return dict(evidence)


def partial_campaign_transfer_plan(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return a validated transfer plan for one resumable partial campaign."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    evidence_path = (
        campaign_evidence.campaign_run_directory(
            run_id,
            storage_root=storage,
        )
        / _PARTIAL_CAMPAIGN_FILENAME
    )
    if evidence_path.is_file() and not refresh:
        evidence = campaign_evidence.load_json_object(
            evidence_path,
            label="partial campaign evidence",
        )
        validated = _validate_partial_campaign_evidence(
            run_id,
            evidence,
            storage_root=storage,
        )
        return cast("dict[str, Any]", validated["transfer_plan"])

    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    status = campaign_status(run_id, storage_root=storage, query_scheduler=True)
    successful = [case for case in status["cases"] if case["state"] == "successful"]
    failed = [case for case in status["cases"] if case["state"] == "failed"]
    if status["campaign_state"] != "completed_with_failures" or not successful or not failed:
        message = "Partial publication requires successful cases and genuine terminal case failures."
        raise RuntimeError(message)
    batches_by_name = {batch.batch_name: batch for batch in campaign.batches}
    for case in successful:
        batch = batches_by_name[str(case["batch_name"])]
        if not batch_runtime.completed_case_is_valid(
            batch,
            int(case["case_index"]),
            storage_root=storage,
        ):
            message = f"Successful case publication is invalid: {case['case_id']!r}."
            raise RuntimeError(message)

    def relative_directory(directory: Path) -> str:
        resolved = directory.resolve()
        if not resolved.is_dir() or resolved.is_symlink():
            message = f"Partial transfer source is missing or unsafe: {resolved}."
            raise FileNotFoundError(message)
        try:
            return resolved.relative_to(storage).as_posix()
        except ValueError as error:
            message = f"Partial transfer source escapes the storage root: {resolved}."
            raise ValueError(message) from error

    def attempt_directories(batch: config_service.GenerationConfig) -> list[str]:
        directories: list[str] = []
        for case_index in batch.case_indices:
            candidate = common.paths.resolve_generation_attempt_case_directory(
                batch.batch_storage_name,
                batch.case_id(case_index),
                run_id,
                storage_root=storage,
            )
            if candidate.is_symlink():
                message = f"Attempt transfer source is unsafe: {candidate}."
                raise ValueError(message)
            if candidate.exists():
                directories.append(relative_directory(candidate))
        return directories

    plan = {
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "git_commit": manifest["git_commit"],
        "campaign_config": manifest["campaign_config"],
        "campaign_directory": relative_directory(evidence_path.parent),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_count": len(batch.case_indices),
                "meta_directory": relative_directory(batch_runtime.batch_meta_directory(batch, storage_root=storage)),
                "raw_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(batch.batch_storage_name, stage="raw", storage_root=storage)
                ),
                "processed_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(batch.batch_storage_name, stage="processed", storage_root=storage)
                ),
                "attempt_directories": attempt_directories(batch),
            }
            for batch in campaign.batches
        ],
    }
    case_fields = (
        "batch_name",
        "batch_id",
        "case_id",
        "case_index",
        "state",
        "classified_state",
    )
    evidence = {
        "schema_kind": "generation_campaign_partial",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_id": campaign.campaign_id,
        "git_commit": manifest["git_commit"],
        "campaign_state": "completed_with_failures",
        "successful_cases": [{field: case[field] for field in case_fields} for case in successful],
        "failed_cases": [{field: case[field] for field in case_fields} for case in failed],
        "resume_command": f"resume {run_id}",
        "recorded_at": _utc_now(),
        "transfer_plan": plan,
    }
    common.serialization.atomic_write_json(evidence_path, evidence)
    validated = _validate_partial_campaign_evidence(
        run_id,
        evidence,
        storage_root=storage,
    )
    return cast("dict[str, Any]", validated["transfer_plan"])


def campaign_transfer_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact terminal campaign files, byte count, and content hashes."""
    root = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    plan = campaign_transfer_plan(run_id, storage_root=root)
    return campaign_evidence.transfer_inventory_from_plan(plan, storage_root=root)


def campaign_transfer_authority(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return the canonical terminal identity and inventory for transfer repair."""
    terminal = validate_terminal_campaign(run_id, storage_root=storage_root)
    inventory = campaign_transfer_inventory(run_id, storage_root=storage_root)
    return {
        "schema_kind": "generation_campaign_transfer_authority",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "file_count": inventory["file_count"],
        "size_bytes": inventory["size_bytes"],
        "inventory_sha256": inventory["inventory_sha256"],
    }


def _transfer_ignored_paths(plan: dict[str, Any], relative_value: str) -> frozenset[str]:
    """Return post-transfer operational paths ignored for one directory."""
    if relative_value == plan["campaign_directory"]:
        return campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS
    return frozenset()


def _campaign_transfer_directory_records(
    plan: dict[str, Any],
    *,
    staging: Path,
) -> list[dict[str, str]]:
    """Return journal-ready directory identities from complete incoming bytes."""
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
    records: list[dict[str, str]] = []
    for relative_value in relative_directories:
        relative = Path(relative_value)
        source = (staging / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(staging):
            message = f"Transfer plan contains an unsafe directory: {relative_value!r}"
            raise ValueError(message)
        records.append(
            {
                "relative_path": relative_value,
                "identity": campaign_evidence.directory_identity(
                    source,
                    ignored_relative_paths=_transfer_ignored_paths(plan, relative_value),
                ),
            }
        )
    return records


def _load_or_create_campaign_transfer_journal(
    run_id: str,
    *,
    staging: Path,
    destination: Path,
    source_host: str,
    source_storage_root: str,
    partial: bool = False,
) -> dict[str, Any]:
    """Load or immutably establish interruption-recovery transfer evidence."""
    journal_path = staging / (_PARTIAL_TRANSFER_PUBLICATION_JOURNAL if partial else _TRANSFER_PUBLICATION_JOURNAL)
    expected_identity = {
        "schema_kind": "generation_transfer_publication",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "source_host": source_host,
        "source_storage_root": source_storage_root,
        "destination_storage_root": str(destination),
    }
    expected_keys = {
        *expected_identity,
        "terminal",
        "plan",
        "source_inventory",
        "directories",
    }
    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Could not read transfer publication journal: {journal_path}"
            raise ValueError(message) from error
        if (
            not isinstance(journal, dict)
            or set(journal) != expected_keys
            or any(journal.get(key) != value for key, value in expected_identity.items())
        ):
            message = f"Transfer publication journal conflicts: {journal_path}"
            raise RuntimeError(message)
        return journal
    if partial:
        evidence_path = (
            campaign_evidence.campaign_run_directory(
                run_id,
                storage_root=staging,
            )
            / _PARTIAL_CAMPAIGN_FILENAME
        )
        terminal = _validate_partial_campaign_evidence(
            run_id,
            campaign_evidence.load_json_object(
                evidence_path,
                label="partial campaign evidence",
            ),
            storage_root=staging,
        )
        plan = partial_campaign_transfer_plan(run_id, storage_root=staging)
    else:
        terminal = validate_terminal_campaign(run_id, storage_root=staging)
        plan = campaign_transfer_plan(run_id, storage_root=staging)
    source_inventory = campaign_evidence.transfer_inventory_from_plan(
        plan,
        storage_root=staging,
    )
    journal = {
        **expected_identity,
        "terminal": (
            terminal
            if partial
            else {
                "campaign_id": terminal["campaign_id"],
                "git_commit": terminal["git_commit"],
            }
        ),
        "plan": plan,
        "source_inventory": source_inventory,
        "directories": _campaign_transfer_directory_records(plan, staging=staging),
    }
    common.serialization.atomic_write_json(journal_path, journal)
    return journal


def _campaign_directory_files(
    directory: Path,
    *,
    ignored_relative_paths: frozenset[str],
) -> dict[str, Path]:
    """Return safe regular files participating in one directory identity."""
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            message = f"Transfer directory contains a symbolic link: {path}"
            raise ValueError(message)
        relative = path.relative_to(directory).as_posix()
        ignored = any(relative == ignored_path or relative.startswith(f"{ignored_path}/") for ignored_path in ignored_relative_paths)
        if path.is_file() and not ignored:
            files[relative] = path
    return files


def _publish_missing_campaign_files(
    source: Path,
    target: Path,
    *,
    ignored_relative_paths: frozenset[str],
    expected_identity: str,
) -> None:
    """Atomically add exact missing source files to an unchanged host subset."""
    source_files = _campaign_directory_files(
        source,
        ignored_relative_paths=ignored_relative_paths,
    )
    target_files = _campaign_directory_files(
        target,
        ignored_relative_paths=ignored_relative_paths,
    )
    for relative, target_path in target_files.items():
        source_path = source_files.get(relative)
        if source_path is None or common.serialization.file_sha256(target_path) != common.serialization.file_sha256(source_path):
            message = f"Existing transfer destination conflicts with incoming identity: {target}"
            raise FileExistsError(message)
    for relative in sorted(set(source_files).difference(target_files)):
        source_path = source_files[relative]
        target_path = (target / relative).resolve()
        if not target_path.is_relative_to(target):
            message = f"Missing transfer file escapes its destination: {relative!r}"
            raise ValueError(message)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.parent.is_symlink():
            message = f"Missing transfer file parent is unsafe: {target_path.parent}"
            raise ValueError(message)
        source_path.replace(target_path)
    if (
        campaign_evidence.directory_identity(
            target,
            ignored_relative_paths=ignored_relative_paths,
        )
        != expected_identity
    ):
        message = f"Repaired transfer destination still differs from incoming identity: {target}"
        raise RuntimeError(message)


def _publish_incoming_campaign_directories(
    journal: dict[str, Any],
    *,
    staging: Path,
    destination: Path,
) -> list[dict[str, str]]:
    """Publish journaled incoming directories by verified atomic rename."""
    plan = journal.get("plan")
    records = journal.get("directories")
    if not isinstance(plan, dict) or not isinstance(records, list):
        message = "Transfer publication journal plan or directories are malformed."
        raise TypeError(message)
    outcomes: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"relative_path", "identity"}:
            message = "Transfer publication journal directory is malformed."
            raise TypeError(message)
        relative_value = record["relative_path"]
        expected_identity = record["identity"]
        if not isinstance(relative_value, str) or not isinstance(expected_identity, str):
            message = "Transfer publication journal directory identity is malformed."
            raise TypeError(message)
        relative = Path(relative_value)
        source = (staging / relative).resolve()
        target = (destination / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(staging) or not target.is_relative_to(destination):
            message = f"Transfer directory escapes a storage root: {relative_value!r}"
            raise ValueError(message)
        ignored = _transfer_ignored_paths(plan, relative_value)
        if target.exists():
            target_identity = campaign_evidence.directory_identity(
                target,
                ignored_relative_paths=ignored,
            )
            if source.exists() and campaign_evidence.directory_identity(source, ignored_relative_paths=ignored) != expected_identity:
                message = f"Incoming transfer source conflicts with its journal: {source}"
                raise RuntimeError(message)
            if target_identity == expected_identity:
                status = "reused"
            else:
                if not source.is_dir() or source.is_symlink():
                    message = f"Incomplete transfer destination has no safe incoming source: {target}"
                    raise FileNotFoundError(message)
                _publish_missing_campaign_files(
                    source,
                    target,
                    ignored_relative_paths=ignored,
                    expected_identity=expected_identity,
                )
                status = "repaired"
        else:
            if not source.is_dir() or source.is_symlink():
                message = f"Incoming transfer source is missing or unsafe: {source}"
                raise FileNotFoundError(message)
            if campaign_evidence.directory_identity(source, ignored_relative_paths=ignored) != expected_identity:
                message = f"Incoming transfer source conflicts with its journal: {source}"
                raise RuntimeError(message)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            if campaign_evidence.directory_identity(target, ignored_relative_paths=ignored) != expected_identity:
                message = f"Transferred directory identity changed during atomic publication: {target}"
                raise RuntimeError(message)
            status = "published"
        outcomes.append(
            {
                "directory": relative_value,
                "status": status,
                "identity": expected_identity,
            }
        )
    return outcomes


def publish_transferred_campaign(
    run_id: str,
    *,
    staging_root: Path | str,
    destination_root: Path | str,
    source_host: str,
    source_storage_root: str,
    partial: bool = False,
) -> dict[str, Any]:
    """Validate incoming bytes and atomically rename them into final locations."""
    if not source_host or any(character in source_host for character in "\r\n\t"):
        message = "Transfer source_host must be non-empty text without control characters."
        raise ValueError(message)
    source_storage = Path(source_storage_root)
    if (
        not source_storage.is_absolute()
        or source_storage == Path("/")
        or ".." in source_storage.parts
        or any(character in source_storage_root for character in "\r\n\t")
    ):
        message = "Transfer source_storage_root must be one absolute non-root path without traversal or controls."
        raise ValueError(message)
    staging = workspace_service.validate_transfer_staging(staging_root, run_id=run_id)
    destination = workspace_service.resolve_storage_root(destination_root, create=True)
    incoming_root = (destination / ".incoming").resolve()
    if not staging.is_relative_to(incoming_root) or staging.stat().st_dev != destination.stat().st_dev:
        message = "Transfer staging must be below destination .incoming on the destination filesystem."
        raise ValueError(message)
    journal = _load_or_create_campaign_transfer_journal(
        run_id,
        staging=staging,
        destination=destination,
        source_host=source_host,
        source_storage_root=source_storage_root,
        partial=partial,
    )
    plan = journal["plan"]
    source_inventory = journal["source_inventory"]
    terminal = journal["terminal"]
    if not isinstance(plan, dict) or not isinstance(source_inventory, dict) or not isinstance(terminal, dict):
        message = "Transfer publication journal terminal evidence is malformed."
        raise TypeError(message)
    if partial:
        if terminal.get("schema_kind") != "generation_campaign_partial":
            message = "Transfer publication journal partial evidence is malformed."
            raise ValueError(message)
        existing_receipt_path = (
            campaign_evidence.campaign_run_directory(
                run_id,
                storage_root=destination,
            )
            / _PARTIAL_TRANSFER_FILENAME
        )
        if existing_receipt_path.exists():
            existing_partial = campaign_evidence.load_json_object(
                existing_receipt_path,
                label="partial transfer receipt",
            )
            expected_identity = {
                "campaign_run_id": run_id,
                "campaign_id": terminal["campaign_id"],
                "git_commit": terminal["git_commit"],
                "source_host": source_host,
                "source_storage_root": source_storage_root,
                "destination_storage_root": str(destination),
            }
            if any(existing_partial.get(key) != value for key, value in expected_identity.items()):
                message = f"Existing partial transfer identity conflicts: {existing_receipt_path}"
                raise FileExistsError(message)
    outcomes = _publish_incoming_campaign_directories(
        journal,
        staging=staging,
        destination=destination,
    )
    if partial:
        evidence_path = (
            campaign_evidence.campaign_run_directory(
                run_id,
                storage_root=destination,
            )
            / _PARTIAL_CAMPAIGN_FILENAME
        )
        if terminal.get("schema_kind") != "generation_campaign_partial":
            message = "Transfer publication journal partial evidence is malformed."
            raise ValueError(message)
        common.serialization.atomic_write_json(evidence_path, terminal)
        evidence = campaign_evidence.load_json_object(
            evidence_path,
            label="partial campaign evidence",
        )
        destination_inventory = campaign_evidence.transfer_inventory_from_plan(
            plan,
            storage_root=destination,
        )
        if destination_inventory != source_inventory:
            message = "Published partial campaign inventory differs from the incoming transfer source."
            raise RuntimeError(message)
        receipt_path = evidence_path.parent / _PARTIAL_TRANSFER_FILENAME
        receipt = {
            "schema_kind": "generation_campaign_partial_transfer",
            "schema_version": 1,
            "status": "partial",
            "recorded_at": _utc_now(),
            "campaign_run_id": run_id,
            "campaign_id": evidence["campaign_id"],
            "git_commit": evidence["git_commit"],
            "source_host": source_host,
            "source_storage_root": source_storage_root,
            "destination_storage_root": str(destination),
            "campaign_partial_sha256": common.serialization.file_sha256(evidence_path),
            "transferred_file_count": source_inventory["file_count"],
            "transferred_bytes": source_inventory["size_bytes"],
            "transfer_inventory_sha256": source_inventory["inventory_sha256"],
            "files": source_inventory["files"],
            "directories": outcomes,
            "successful_cases": evidence["successful_cases"],
            "failed_cases": evidence["failed_cases"],
            "source_removed": False,
        }
        common.serialization.atomic_write_json(receipt_path, receipt)
        return validate_partially_transferred_campaign(
            run_id,
            storage_root=destination,
        )

    validated = validate_terminal_campaign(run_id, storage_root=destination)
    destination_inventory = campaign_evidence.transfer_inventory_from_plan(plan, storage_root=destination)
    if destination_inventory != source_inventory:
        message = "Published campaign inventory differs from the incoming transfer source."
        raise RuntimeError(message)
    terminal_path = campaign_evidence.campaign_run_directory(run_id, storage_root=destination) / "campaign_terminal.json"
    receipt_path = campaign_evidence.campaign_run_directory(run_id, storage_root=destination) / "transfer_complete.json"
    identity = {
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "source_host": source_host,
        "source_storage_root": source_storage_root,
        "destination_storage_root": str(destination),
        "campaign_terminal_sha256": common.serialization.file_sha256(terminal_path),
        "transferred_file_count": source_inventory["file_count"],
        "transferred_bytes": source_inventory["size_bytes"],
        "transfer_inventory_sha256": source_inventory["inventory_sha256"],
        "files": source_inventory["files"],
    }
    if receipt_path.exists():
        try:
            existing = validate_transferred_campaign(run_id, storage_root=destination)
        except (FileNotFoundError, TypeError, ValueError):
            existing = None
        if existing is not None:
            if any(existing.get(key) != value for key, value in identity.items()):
                message = f"Existing transfer completion receipt conflicts: {receipt_path}"
                raise FileExistsError(message)
            return existing
    receipt = {
        "schema_kind": "generation_campaign_transfer",
        "schema_version": 1,
        "status": "transfer_complete",
        "recorded_at": _utc_now(),
        **identity,
        "directories": outcomes,
        "terminal_validation": {
            "status": "pass",
            "batch_count": len(validated["batches"]),
        },
        "source_removed": False,
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    return validate_transferred_campaign(run_id, storage_root=destination)


def repair_transferred_campaign(
    run_id: str,
    *,
    source_host: str,
    source_storage_root: str,
    authority: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Reconstruct transfer evidence only for an exact canonical host copy."""
    if not source_host or any(character in source_host for character in "\r\n\t"):
        message = "Transfer source_host must be non-empty text without control characters."
        raise ValueError(message)
    source_storage = Path(source_storage_root)
    if (
        not source_storage.is_absolute()
        or source_storage == Path("/")
        or ".." in source_storage.parts
        or any(character in source_storage_root for character in "\r\n\t")
    ):
        message = "Transfer source_storage_root must be one absolute non-root path without traversal or controls."
        raise ValueError(message)
    destination = workspace_service.resolve_storage_root(storage_root, create=False)
    terminal = validate_terminal_campaign(run_id, storage_root=destination)
    plan = campaign_transfer_plan(run_id, storage_root=destination)
    inventory = campaign_evidence.transfer_inventory_from_plan(
        plan,
        storage_root=destination,
    )
    expected_authority = {
        "schema_kind": "generation_campaign_transfer_authority",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "file_count": inventory["file_count"],
        "size_bytes": inventory["size_bytes"],
        "inventory_sha256": inventory["inventory_sha256"],
    }
    if dict(authority) != expected_authority:
        message = "Host campaign publication differs from the canonical CPU transfer authority."
        raise RuntimeError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=destination,
    )
    receipt_path = run_directory / "transfer_complete.json"
    if receipt_path.exists():
        try:
            existing = validate_transferred_campaign(run_id, storage_root=destination)
        except (FileNotFoundError, TypeError, ValueError):
            existing = None
        if existing is not None:
            if existing["source_host"] != source_host or existing["source_storage_root"] != source_storage_root:
                message = f"Existing transfer completion receipt conflicts: {receipt_path}"
                raise FileExistsError(message)
            return existing
    directory_values = [
        directory
        for batch in plan["batches"]
        for directory in (
            batch["meta_directory"],
            batch["raw_directory"],
            batch["processed_directory"],
            *batch["attempt_directories"],
        )
    ]
    directory_values.append(plan["campaign_directory"])
    directories = [
        {
            "directory": relative,
            "status": "reused",
            "identity": campaign_evidence.directory_identity(
                destination / relative,
                ignored_relative_paths=_transfer_ignored_paths(plan, relative),
            ),
        }
        for relative in directory_values
    ]
    terminal_path = run_directory / "campaign_terminal.json"
    receipt = {
        "schema_kind": "generation_campaign_transfer",
        "schema_version": 1,
        "status": "transfer_complete",
        "recorded_at": _utc_now(),
        "campaign_run_id": run_id,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "source_host": source_host,
        "source_storage_root": source_storage_root,
        "destination_storage_root": str(destination),
        "campaign_terminal_sha256": common.serialization.file_sha256(terminal_path),
        "transferred_file_count": inventory["file_count"],
        "transferred_bytes": inventory["size_bytes"],
        "transfer_inventory_sha256": inventory["inventory_sha256"],
        "files": inventory["files"],
        "directories": directories,
        "terminal_validation": {
            "status": "pass",
            "batch_count": len(terminal["batches"]),
        },
        "source_removed": False,
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    return validate_transferred_campaign(run_id, storage_root=destination)


def validate_transferred_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the immutable GPU publication and exact transfer receipt."""
    destination = workspace_service.resolve_storage_root(storage_root, create=False)
    terminal = validate_terminal_campaign(run_id, storage_root=destination)
    plan = campaign_transfer_plan(run_id, storage_root=destination)
    return campaign_evidence.validate_transfer_receipt(
        run_id,
        terminal=terminal,
        plan=plan,
        storage_root=destination,
    )


def validate_partially_transferred_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate a resumable partial GPU publication without promoting it to complete."""
    destination = workspace_service.resolve_storage_root(storage_root, create=False)
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=destination,
    )
    evidence_path = run_directory / _PARTIAL_CAMPAIGN_FILENAME
    evidence = campaign_evidence.load_json_object(
        evidence_path,
        label="partial campaign evidence",
    )
    plan = partial_campaign_transfer_plan(run_id, storage_root=destination)
    inventory = campaign_evidence.transfer_inventory_from_plan(
        plan,
        storage_root=destination,
    )
    receipt_path = run_directory / _PARTIAL_TRANSFER_FILENAME
    receipt = campaign_evidence.load_json_object(
        receipt_path,
        label="partial transfer receipt",
    )
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
        "campaign_partial_sha256",
        "transferred_file_count",
        "transferred_bytes",
        "transfer_inventory_sha256",
        "files",
        "directories",
        "successful_cases",
        "failed_cases",
        "source_removed",
    }
    successful = evidence.get("successful_cases")
    failed = evidence.get("failed_cases")
    if (
        set(receipt) != required
        or receipt.get("schema_kind") != "generation_campaign_partial_transfer"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "partial"
        or receipt.get("campaign_run_id") != run_id
        or receipt.get("campaign_id") != evidence.get("campaign_id")
        or receipt.get("git_commit") != evidence.get("git_commit")
        or receipt.get("destination_storage_root") != str(destination)
        or receipt.get("campaign_partial_sha256") != common.serialization.file_sha256(evidence_path)
        or receipt.get("transferred_file_count") != inventory["file_count"]
        or receipt.get("transferred_bytes") != inventory["size_bytes"]
        or receipt.get("transfer_inventory_sha256") != inventory["inventory_sha256"]
        or receipt.get("files") != inventory["files"]
        or receipt.get("successful_cases") != successful
        or receipt.get("failed_cases") != failed
        or not isinstance(successful, list)
        or not successful
        or not isinstance(failed, list)
        or not failed
        or receipt.get("source_removed") is not False
    ):
        message = f"Partial transfer receipt or GPU publication is invalid: {receipt_path}"
        raise ValueError(message)
    campaign = campaign_evidence.campaign_from_manifest(campaign_evidence.load_campaign_run(run_id, storage_root=destination))
    batches = {batch.batch_name: batch for batch in campaign.batches}
    for case in successful:
        if not isinstance(case, dict) or case.get("classified_state") != "successful":
            message = f"Partial successful-case evidence is invalid: {receipt_path}"
            raise ValueError(message)
        batch = batches[str(case["batch_name"])]
        if not batch_runtime.completed_case_is_valid(
            batch,
            int(case["case_index"]),
            storage_root=destination,
        ):
            message = f"Partial successful case is not publishable: {case['case_id']!r}."
            raise RuntimeError(message)
    if any(not isinstance(case, dict) or case.get("classified_state") == "successful" for case in failed):
        message = f"Partial failed-case evidence is invalid: {receipt_path}"
        raise ValueError(message)
    return receipt
