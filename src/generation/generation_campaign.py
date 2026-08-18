"""
generation_campaign.py

Persist, feed, inspect, and terminally validate one campaign run.
Responsibilities:
  - Bind campaign execution to one clean exact Git commit and execution config
  - Reconcile exact per-case Slurm jobs and restore a small pending buffer
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

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common

from .cases import generation_cases_config as config_service
from .cases import generation_cases_input as input_service
from .contracts import generation_contracts_source as source_service
from .publication import generation_publication_attempt as attempt_service
from .publication import generation_publication_campaign_evidence as campaign_evidence
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
        "pending_buffer": int(submission["pending_buffer"]),
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


def _scheduler_job_name(prefix: str, submission_index: int) -> str:
    """Return one unique bounded job name for an exact submission intent."""
    return f"{prefix}-{submission_index:04d}"


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
    scheduler_prefix = f"vp2-{run_id.rsplit('__', maxsplit=1)[-1]}"
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
    prefix = f"vp2-{run_id.rsplit('__', maxsplit=1)[-1]}"
    first_command = cluster_service.build_campaign_case_slurm_submission_command(
        campaign,
        tasks[0],
        run_id=run_id,
        scheduler_log_directory=log_directory,
        scheduler_job_name=_scheduler_job_name(prefix, 1),
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
        "submission_model": "one ordinary non-exclusive Slurm job per case, restored one job at a time to the pending buffer",
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
        }
    return {
        "latest_job_id": job_id,
        "latest_job_name": job_name,
        "scheduler_state": (_scheduler_state(accounted_state) if (accounted_state := _scheduler_field(accounted, 1)) is not None else None),
        "node": _scheduler_field(accounted, 7),
        "submit_time": _scheduler_field(accounted, 3),
        "start_time": _scheduler_field(accounted, 4),
        "elapsed": _scheduler_field(accounted, 6),
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
) -> attempt_service.AttemptEvidence | None:
    """Return the newest attempt after exact campaign and case identity checks."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    attempt = attempt_service.latest_case_attempt(
        batch,
        task.case_index,
        str(manifest["campaign_run_id"]),
        storage_root=storage,
    )
    if attempt is None:
        return None
    input_reference = input_service.admit_persisted_input_case(
        batch,
        task.case_index,
        str(attempt.payload["input_generation_id"]),
        storage_root=storage,
    )
    canonical_raw_case = input_reference.case_directory.resolve().relative_to(storage).as_posix()
    expected = {
        "campaign_run_id": manifest["campaign_run_id"],
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
        "solver_git_commit": manifest["git_commit"],
        "scientific_config_digest": batch.scientific_config_digest,
        "export_contract_sha256": common.serialization.canonical_json_sha256(batch.scientific_values["output_contract"]),
    }
    if any(attempt.payload.get(key) != value for key, value in expected.items()) or attempt.payload.get("template") != {
        "relative_path": batch.template_relative_path,
        "sha256": batch.template_sha256,
    }:
        message = f"Attempt evidence disagrees with its persisted campaign case: {attempt.receipt_path}"
        raise RuntimeError(message)
    return attempt


def _successful_quality_flag_count(
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> int:
    """Return the admitted processed quality-flag count for one successful case."""
    status_path = (
        batch_runtime.processed_case_directory(
            batch,
            task.case_index,
            storage_root=storage_root,
        )
        / "status.json"
    )
    if not status_path.is_file():
        return 0
    status = campaign_evidence.load_json_object(
        status_path,
        label="processed case status",
    )
    count = status.get("quality_flag_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        message = f"Processed case quality-flag count is malformed: {status_path}"
        raise ValueError(message)
    return count


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


def _task_state(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Reconcile one case from processed, attempt, and exact scheduler evidence."""
    batch = campaign.batch(task.batch_name)
    submissions = _task_submissions(manifest, task)
    latest_submission = submissions[-1] if submissions else None
    retry_attempt: Mapping[str, Any] | None = None
    attempt: attempt_service.AttemptEvidence | None = None
    pipeline = {
        "solver_state": "not_started",
        "exports_state": "not_started",
        "conversion_state": "not_started",
        "diagnostics_state": "not_started",
        "publication_state": "not_started",
    }
    failure_stage: str | None = None
    replay_available = False
    attempt_index: int | None = None
    quality_flag_count = 0
    canonical_raw_case: str | None = None

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
        quality_flag_count = _successful_quality_flag_count(
            batch,
            task,
            storage_root=storage_root,
        )
    else:
        active_records = [record for record in submissions if record["job_id"] in scheduler["active"]]
        unknown_records = [
            record
            for record in submissions
            if record["status"] == "submitted" and record["job_id"] not in scheduler["active"] and record["job_id"] not in scheduler["accounted"]
        ]
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
                retry_attempt = license_service.latest_attempt_for_job(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    job_id=str(latest_submission["job_id"]),
                    storage_root=storage_root,
                )
            attempt = _admitted_case_attempt(
                manifest,
                batch,
                task,
                storage_root=storage_root,
            )
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
                replay_available = attempt.replay_available
                attempt_index = int(attempt.payload["attempt_index"])
                flags = attempt.payload.get("quality_flags")
                quality_flag_count = len(flags) if isinstance(flags, list) else 0
                raw_reference = attempt.payload.get("canonical_raw_case")
                canonical_raw_case = raw_reference if isinstance(raw_reference, str) else None
            elif retry_attempt is not None:
                if not retry_attempt["retry_budget_remaining"]:
                    state = "failed"
                    reason = license_service.EXHAUSTED_REASON
                    failure_stage = "solver"
                    pipeline["solver_state"] = "failed"
                elif license_service.retry_attempt_is_eligible(retry_attempt):
                    state = "retry_eligible"
                    reason = license_service.TEMPORARY_LICENSE_CAPACITY
                else:
                    state = "retry_waiting"
                    reason = license_service.TEMPORARY_LICENSE_CAPACITY
            elif batch_runtime.case_failure_is_recorded(
                batch,
                task.case_index,
                storage_root=storage_root,
                execution_run_id=str(manifest["campaign_run_id"]),
                git_commit=str(manifest["git_commit"]),
            ):
                state = "failed"
                reason = "legacy_case_failure_evidence"
                failure_stage = "solver"
                pipeline["solver_state"] = "failed"
            elif latest_accounted is not None:
                state, reason = _scheduler_terminal_case_state(_scheduler_state(latest_accounted[1]))
                failure_stage = "solver"
                pipeline["solver_state"] = state
            elif submissions:
                state = "never_started"
                reason = str(submissions[-1]["error"] or "submission_failed")
            else:
                state = "never_started"
                reason = "not_submitted"

    scheduler_view = _task_scheduler_view(latest_submission, scheduler)
    runtime_progress = _task_runtime_progress_view(
        manifest,
        task,
        latest_job_id=scheduler_view["latest_job_id"],
        storage_root=storage_root,
    )
    return {
        **_task_payload(task),
        "state": state,
        "reason": reason,
        "submission_count": len(submissions),
        "attempt_index": attempt_index,
        "failure_stage": failure_stage,
        **pipeline,
        "quality_flag_count": quality_flag_count,
        "postprocessing_replay_available": replay_available,
        "canonical_raw_case": canonical_raw_case,
        **scheduler_view,
        "runtime_progress": runtime_progress,
        "temporary_license_retry": retry_attempt,
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


def _reconciled(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return task views plus exact pending/running counts for this campaign."""
    task_views = [
        _task_state(
            manifest,
            campaign,
            task,
            scheduler,
            storage_root=storage_root,
        )
        for task in cluster_service.campaign_tasks(campaign)
    ]
    persisted_job_ids = set(manifest["slurm_job_ids"])
    states = [_scheduler_state(fields[1]) for job_id, fields in scheduler["active"].items() if job_id in persisted_job_ids]
    pending_jobs = sum(state == _ACTIVE_PENDING_STATE for state in states)
    running_jobs = len(states) - pending_jobs
    return task_views, pending_jobs, running_jobs


def _submit_one(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    mode: str,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist one intent, submit one case, and atomically persist its job ID."""
    if mode not in {"initial", "resume", "license_retry", "explicit_retry"}:
        message = f"Unsupported campaign submission mode: {mode!r}."
        raise ValueError(message)
    index = len(manifest["submissions"]) + 1
    job_name = _scheduler_job_name(str(manifest["scheduler_job_name"]), index)
    command = cluster_service.build_campaign_case_slurm_submission_command(
        campaign,
        task,
        run_id=str(manifest["campaign_run_id"]),
        scheduler_log_directory=Path(manifest["scheduler_log_directory"]),
        scheduler_job_name=job_name,
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


def feed_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile exact jobs and submit at most one normally feedable case."""
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
        if manifest["submission_intent"] is not None:
            manifest = _recover_submission_intent(
                manifest,
                storage_root=storage_root,
            )
            if manifest["submission_intent"] is not None:
                manifest["state"] = "submission_unknown"
                common.serialization.atomic_write_json(
                    campaign_evidence.campaign_run_manifest_path(
                        run_id,
                        storage_root=storage_root,
                    ),
                    manifest,
                )
                return manifest
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        task_views, pending_jobs, running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
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

        pending_buffer = int(manifest["submission_config"]["pending_buffer"])
        max_running = manifest["submission_config"]["max_running_cases"]
        if pending_jobs >= pending_buffer or (max_running is not None and running_jobs >= int(max_running)):
            manifest["state"] = "active"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest

        next_retry = next(
            (view for view in task_views if view["state"] == "retry_eligible"),
            None,
        )
        next_unsent = next(
            (view for view in task_views if view["state"] == "never_started"),
            None,
        )
        unresolved_solver_failures = sum(view["state"] in {"failed", "timed_out"} for view in task_views)
        maximum_failed_cases = int(manifest["submission_config"]["maximum_failed_cases"])
        circuit_breaker_tripped = unresolved_solver_failures > maximum_failed_cases
        next_view = next_retry
        if next_view is None and not circuit_breaker_tripped:
            next_view = next_unsent
        if next_view is None:
            if pending_jobs or running_jobs:
                manifest["state"] = "active"
            elif any(view["state"] == "retry_waiting" for view in task_views):
                manifest["state"] = "waiting_retry"
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
        return _submit_one(
            manifest,
            campaign,
            task,
            mode=("license_retry" if next_view["state"] == "retry_eligible" else "initial"),
            storage_root=storage_root,
        )


def submit_campaign(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create durable campaign state and submit only the first eligible case job."""
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
            immutable_keys = set(intent).difference({"slurm_job_ids", "submissions", "submission_intent", "state"})
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
    """Advance at most one case according to the explicit resume matrix."""
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
        campaign = campaign_evidence.campaign_from_manifest(manifest)
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        task_views, pending_jobs, running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
        )
        if pending_jobs or running_jobs:
            manifest["state"] = "active"
            return _write_campaign_manifest(
                manifest,
                storage_root=storage_root,
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

        replay_view = next(
            (
                view
                for view in task_views
                if view["state"] in {"conversion_failed", "publication_failed"} and view["postprocessing_replay_available"] is True
            ),
            None,
        )
        if replay_view is not None:
            batch = campaign.batch(str(replay_view["batch_name"]))
            outcome = batch_runtime.replay_case_postprocessing(
                batch,
                int(replay_view["case_index"]),
                storage_root=storage_root,
            )
            if outcome.status not in {"replayed", "skipped"}:
                manifest["state"] = "completed_with_failures"
                return _write_campaign_manifest(
                    manifest,
                    storage_root=storage_root,
                )
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
            manifest["state"] = "complete" if all_successful else "active"
            return _write_campaign_manifest(
                manifest,
                storage_root=storage_root,
            )

        restart_view = next(
            (view for view in task_views if view["state"] in {"cancelled", "interrupted"}),
            None,
        )
        unresolved_solver_failures = sum(view["state"] in {"failed", "timed_out"} for view in task_views)
        maximum_failed_cases = int(manifest["submission_config"]["maximum_failed_cases"])
        if restart_view is None and unresolved_solver_failures <= maximum_failed_cases:
            restart_view = next(
                (view for view in task_views if view["state"] == "never_started"),
                None,
            )
        if restart_view is None:
            manifest["state"] = "completed_with_failures"
            return _write_campaign_manifest(
                manifest,
                storage_root=storage_root,
            )

        current_commit = _repository_commit()
        if current_commit != manifest["git_commit"]:
            message = f"Solver resume requires the exact original campaign source commit {manifest['git_commit']}, got {current_commit}."
            raise RuntimeError(message)
        manifest["state"] = "active"
        task = _task_from_payload(campaign, restart_view)
        return _submit_one(
            manifest,
            campaign,
            task,
            mode=("resume" if restart_view["state"] in {"cancelled", "interrupted"} else "initial"),
            storage_root=storage_root,
        )


def retry_campaign_case(
    run_id: str,
    batch_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Explicitly submit one selected unresolved scientific case from time zero."""
    match = _CASE_ID_PATTERN.fullmatch(case_id)
    if match is None:
        message = f"retry-case requires a canonical case ID, got {case_id!r}."
        raise ValueError(message)
    case_index = int(match.group(1))
    if case_index < 1:
        message = f"retry-case requires a positive case index, got {case_id!r}."
        raise ValueError(message)
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
        current_commit = _repository_commit()
        if current_commit != manifest["git_commit"]:
            message = f"Explicit solver retry requires the exact original campaign source commit {manifest['git_commit']}, got {current_commit}."
            raise RuntimeError(message)
        campaign = campaign_evidence.campaign_from_manifest(manifest)
        task = cluster_service.require_campaign_task(
            campaign,
            batch_name=batch_name,
            case_index=case_index,
        )
        if task.case_id != case_id:
            message = f"retry-case identity mismatch for {batch_name}/{case_id}."
            raise ValueError(message)
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        task_views, _pending_jobs, _running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
        )
        matches = [view for view in task_views if view["batch_name"] == batch_name and view["case_id"] == case_id]
        if len(matches) != 1:
            message = f"Campaign has no unique retry target {batch_name}/{case_id}."
            raise RuntimeError(message)
        view = matches[0]
        if view["state"] == "successful":
            message = f"Successful case cannot be retried: {batch_name}/{case_id}."
            raise ValueError(message)
        if view["state"] in {
            "active",
            "pending",
            "scheduler_unknown",
            "retry_waiting",
            "retry_eligible",
        }:
            message = f"Case is still active or operationally retrying: {batch_name}/{case_id}."
            raise RuntimeError(message)
        if view["state"] in {"cancelled", "interrupted", "never_started"}:
            message = f"Case {batch_name}/{case_id} belongs to normal campaign resume."
            raise ValueError(message)
        if view["state"] in {"conversion_failed", "publication_failed"} and view["postprocessing_replay_available"] is True:
            message = f"Case {batch_name}/{case_id} has valid postprocessing replay evidence; use resume."
            raise ValueError(message)
        eligible = {
            "failed",
            "timed_out",
            "exports_failed",
            "conversion_failed",
            "publication_failed",
        }
        if view["state"] not in eligible:
            message = f"Case {batch_name}/{case_id} is not eligible for explicit retry from state {view['state']!r}."
            raise ValueError(message)
        manifest["state"] = "active"
        return _submit_one(
            manifest,
            campaign,
            task,
            mode="explicit_retry",
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
    return {
        "batch_name": batch.batch_name,
        "batch_id": batch.batch_id,
        "planned": len(selected),
        "successful": sum(view["state"] == "successful" for view in selected),
        "active": sum(view["state"] == "active" for view in selected),
        "pending": sum(view["state"] == "pending" for view in selected),
        "never_started": sum(view["state"] == "never_started" for view in selected),
        "solver_failed": sum(view["state"] == "failed" for view in selected),
        "timed_out": sum(view["state"] == "timed_out" for view in selected),
        "exports_failed": sum(view["state"] == "exports_failed" for view in selected),
        "conversion_failed": sum(view["state"] == "conversion_failed" for view in selected),
        "publication_failed": sum(view["state"] == "publication_failed" for view in selected),
        "cancelled": sum(view["state"] == "cancelled" for view in selected),
        "interrupted": sum(view["state"] == "interrupted" for view in selected),
        "quality_flagged": sum(int(view["quality_flag_count"]) > 0 for view in selected),
        "waiting_retry": sum(view["state"] == "retry_waiting" for view in selected),
        "retry_eligible": sum(view["state"] == "retry_eligible" for view in selected),
        "scheduler_unknown": sum(view["state"] == "scheduler_unknown" for view in selected),
        "quarantined": (len(tuple(quarantine.iterdir())) if quarantine.is_dir() else 0),
        "terminal_manifest": str(terminal_path),
        "terminal_manifest_available": terminal_path.is_file(),
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
    count_states = {
        "planned": len(task_views),
        "successful": sum(view["state"] == "successful" for view in task_views),
        "active": sum(view["state"] == "active" for view in task_views),
        "pending": sum(view["state"] == "pending" for view in task_views),
        "never_started": sum(view["state"] == "never_started" for view in task_views),
        "solver_failed": sum(view["state"] == "failed" for view in task_views),
        "timed_out": sum(view["state"] == "timed_out" for view in task_views),
        "exports_failed": sum(view["state"] == "exports_failed" for view in task_views),
        "conversion_failed": sum(view["state"] == "conversion_failed" for view in task_views),
        "publication_failed": sum(view["state"] == "publication_failed" for view in task_views),
        "cancelled": sum(view["state"] == "cancelled" for view in task_views),
        "interrupted": sum(view["state"] == "interrupted" for view in task_views),
        "quality_flagged": sum(int(view["quality_flag_count"]) > 0 for view in task_views),
        "waiting_retry": sum(view["state"] == "retry_waiting" for view in task_views),
    }
    unknown_cases = sum(view["state"] == "scheduler_unknown" for view in task_views)
    retry_eligible_cases = sum(view["state"] == "retry_eligible" for view in task_views)
    replayable_cases = sum(view["postprocessing_replay_available"] is True for view in task_views)
    unresolved_failure_states = {
        "failed",
        "timed_out",
        "exports_failed",
        "conversion_failed",
        "publication_failed",
    }
    unresolved_cases = sum(view["state"] in unresolved_failure_states for view in task_views)
    unresolved_solver_failures = count_states["solver_failed"] + count_states["timed_out"]
    maximum_failed_cases = int(manifest["submission_config"]["maximum_failed_cases"])
    circuit_breaker_tripped = unresolved_solver_failures > maximum_failed_cases
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
    elif count_states["waiting_retry"]:
        state = "waiting_retry"
        next_command = f"status {run_id}"
    elif all_successful and publication_complete:
        state = "transfer_complete" if transfer_receipt else "successful"
        next_command = f"status {run_id}" if transfer_receipt else f"collect {run_id}"
    elif cancellation_requested:
        state = "cancelled"
        next_command = f"resume {run_id}"
    elif count_states["never_started"] or count_states["cancelled"] or count_states["interrupted"] or replayable_cases or retry_eligible_cases:
        state = "feeding"
        next_command = f"resume {run_id}"
    elif unresolved_cases:
        state = "completed_with_failures"
        next_command = f"resume {run_id}"
    else:
        state = "feeding"
        next_command = f"feed-campaign {run_id}"

    failed_cases = unresolved_cases
    return {
        "campaign_run_id": run_id,
        "campaign_state": state,
        "campaign_purpose": campaign.campaign_purpose,
        "cases_per_material": (len(campaign.batches[0].case_indices) if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE else None),
        "git_commit": manifest["git_commit"],
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
        "retry_waiting_cases": count_states["waiting_retry"],
        "retry_eligible_cases": retry_eligible_cases,
        "postprocessing_replay_available_cases": replayable_cases,
        "unresolved_solver_failed_cases": unresolved_solver_failures,
        "maximum_failed_cases": maximum_failed_cases,
        "failure_circuit_breaker_tripped": circuit_breaker_tripped,
        "cancellation_requested": cancellation_requested,
        "squeue": manifest_scheduler_view(scheduler["squeue"]),
        "sacct": manifest_scheduler_view(scheduler["sacct"]),
        "batches": batches,
        "cases": task_views,
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


def campaign_transfer_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact terminal campaign files, byte count, and content hashes."""
    root = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    plan = campaign_transfer_plan(run_id, storage_root=root)
    return campaign_evidence.transfer_inventory_from_plan(plan, storage_root=root)


def publish_transferred_campaign(
    run_id: str,
    *,
    staging_root: Path | str,
    destination_root: Path | str,
    source_host: str,
    source_storage_root: str,
) -> dict[str, Any]:
    """Validate staged bytes, atomically publish directories, and mark transfer."""
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
    staging = workspace_service.validate_transfer_staging(
        staging_root,
        run_id=run_id,
    )
    destination = workspace_service.resolve_storage_root(
        destination_root,
        create=True,
    )
    if staging == destination:
        message = "Transfer staging and destination storage roots must differ."
        raise ValueError(message)
    terminal = validate_terminal_campaign(run_id, storage_root=staging)
    plan = campaign_transfer_plan(run_id, storage_root=staging)
    source_inventory = campaign_evidence.transfer_inventory_from_plan(plan, storage_root=staging)
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
    publication_root = common.paths.get_generation_state_root(storage_root=destination) / "transfer-publication"
    outcomes: list[dict[str, str]] = []
    for directory_index, relative_value in enumerate(relative_directories):
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            message = f"Transfer plan contains an unsafe directory: {relative_value!r}"
            raise ValueError(message)
        source = (staging / relative).resolve()
        target = (destination / relative).resolve()
        if not source.is_relative_to(staging) or not target.is_relative_to(destination):
            message = f"Transfer directory escapes a storage root: {relative_value!r}"
            raise ValueError(message)
        ignored_relative_paths = campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS if relative_value == plan["campaign_directory"] else frozenset()
        source_identity = campaign_evidence.directory_identity(source, ignored_relative_paths=ignored_relative_paths)
        if target.exists():
            target_identity = campaign_evidence.directory_identity(target, ignored_relative_paths=ignored_relative_paths)
            if target_identity != source_identity:
                message = f"Existing transfer destination conflicts with staged identity: {target}"
                raise FileExistsError(message)
            outcomes.append(
                {
                    "directory": relative_value,
                    "status": "reused",
                    "identity": source_identity,
                }
            )
            continue
        publication_case_id = f"transfer-{directory_index:04d}"
        publication_stage = workspace_service.create_publication_staging(
            storage_root=destination,
            publication_root=publication_root,
            run_id=run_id,
            case_id=publication_case_id,
        )
        payload = publication_stage / "payload"
        shutil.copytree(source, payload)
        if campaign_evidence.directory_identity(payload, ignored_relative_paths=ignored_relative_paths) != source_identity:
            message = f"Transfer copy changed staged identity: {relative_value!r}"
            raise RuntimeError(message)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.replace(target)
        if campaign_evidence.directory_identity(target, ignored_relative_paths=ignored_relative_paths) != source_identity:
            message = f"Transferred directory identity changed during publication: {target}"
            raise RuntimeError(message)
        workspace_service.cleanup_publication_staging(
            publication_stage,
            storage_root=destination,
            publication_root=publication_root,
            run_id=run_id,
            case_id=publication_case_id,
            allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        outcomes.append(
            {
                "directory": relative_value,
                "status": "published",
                "identity": source_identity,
            }
        )
    validated = validate_terminal_campaign(run_id, storage_root=destination)
    destination_inventory = campaign_evidence.transfer_inventory_from_plan(plan, storage_root=destination)
    if destination_inventory != source_inventory:
        message = "Published GPU campaign inventory differs from the staged transfer source."
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
        existing = validate_transferred_campaign(run_id, storage_root=destination)
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
