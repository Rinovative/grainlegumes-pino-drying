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

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
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
_JOB_NAME_PATTERN: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_SLURM_JOB_NAME_LENGTH: Final = 48
_TRANSFER_PUBLICATION_JOURNAL: Final = ".generation-transfer-publication.json"
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
) -> attempt_service.AttemptEvidence | None:
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
        if candidate is None or candidate.payload["campaign_run_id"] == current_run_id:
            return None
        if candidate.payload["failure_stage"] not in {"conversion", "publication"}:
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
    return attempt


def _successful_status_summary(
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return compact terminal fields from the already-consumed case status."""
    status_path = (
        batch_runtime.processed_case_directory(
            batch,
            task.case_index,
            storage_root=storage_root,
        )
        / "status.json"
    )
    if not status_path.is_file():
        return {
            "quality_flag_count": 0,
            "simulated_end_time": None,
            "simulated_end_time_unit": None,
            "final_moisture_name": None,
            "final_moisture_value": None,
            "final_moisture_unit": None,
        }
    status = campaign_evidence.load_json_object(
        status_path,
        label="processed case status",
    )
    count = status.get("quality_flag_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        message = f"Processed case quality-flag count is malformed: {status_path}"
        raise ValueError(message)
    units = status.get("units")
    unit_values = units if isinstance(units, dict) else {}
    simulated_end = status.get("t_stop_exact")
    simulated_end_unit = unit_values.get("t_stop_exact")
    final_moisture = status.get("f_wet_dm_final")
    final_moisture_unit = unit_values.get("f_wet_dm_final")
    simulated_end_value = float(simulated_end) if not isinstance(simulated_end, bool) and isinstance(simulated_end, (int, float)) else None
    final_moisture_value = float(final_moisture) if not isinstance(final_moisture, bool) and isinstance(final_moisture, (int, float)) else None
    simulated_end_available = simulated_end_value is not None and isinstance(simulated_end_unit, str) and bool(simulated_end_unit)
    final_moisture_available = final_moisture_value is not None and isinstance(final_moisture_unit, str) and bool(final_moisture_unit)
    return {
        "quality_flag_count": count,
        "simulated_end_time": simulated_end_value if simulated_end_available else None,
        "simulated_end_time_unit": simulated_end_unit if simulated_end_available else None,
        "final_moisture_name": "f_wet_dm_final" if final_moisture_available else None,
        "final_moisture_value": final_moisture_value if final_moisture_available else None,
        "final_moisture_unit": final_moisture_unit if final_moisture_available else None,
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


def _successful_completion_at(
    state: str,
    scheduler_view: Mapping[str, Any],
    *,
    case_id: str,
) -> str | None:
    """Return the authoritative Slurm terminal time for one validated success."""
    end_time = scheduler_view.get("end_time")
    if state != "successful" or not isinstance(end_time, str):
        return None
    return _parse_utc_timestamp(
        end_time,
        label=f"Successful case terminal timestamp for {case_id}",
    ).isoformat()


def _unsubmitted_task_state(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    submissions: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Return controller lifecycle for a case without active terminal evidence."""
    if submissions:
        return "never_started", str(submissions[-1]["error"] or "submission_failed")
    if (task.batch_id, task.case_index) in _admission_reservation_keys(manifest):
        return "admission_waiting", "license_launch_pacing"
    return "never_started", "not_submitted"


def _task_state(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
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
    replay = _postprocessing_replay_view(batch, failure_stage, attempt)
    attempt_index: int | None = None
    quality_flag_count = 0
    successful_status: Mapping[str, Any] = {
        "simulated_end_time": None,
        "simulated_end_time_unit": None,
        "final_moisture_name": None,
        "final_moisture_value": None,
        "final_moisture_unit": None,
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
        successful_status = _successful_status_summary(
            batch,
            task,
            storage_root=storage_root,
        )
        quality_flag_count = int(successful_status["quality_flag_count"])
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
                retry_attempt = license_service.latest_wait_for_job(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    job_id=str(latest_submission["job_id"]),
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
                state = "license_blocked"
                reason = license_service.TEMPORARY_LICENSE_CAPACITY
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
    completed_at = _successful_completion_at(
        state,
        scheduler_view,
        case_id=task.case_id,
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
        "completed_at": completed_at,
        **successful_status,
        **replay,
        "canonical_raw_case": canonical_raw_case,
        **scheduler_view,
        "runtime_progress": runtime_progress,
        "temporary_license_retry": retry_attempt,
        "license_retry_active": license_retry_active,
        "license_retry_eligible": license_retry_eligible,
        "license_wait_exhausted": license_wait_exhausted,
        "license_first_blocked_at": license_first_blocked_at,
        "license_next_retry_at": license_next_retry_at,
        "evidence_path": None if attempt is None else str(attempt.receipt_path),
        "automatic_continuation_allowed": bool(
            replay["replay_eligible"] or license_retry_eligible or state in {"admission_waiting", "never_started", "cancelled", "interrupted"}
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


def _reconciled(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return task views plus exact pending/running counts for this campaign."""
    task_views = [
        _task_state(
            manifest,
            campaign,
            task,
            scheduler,
            storage_root=storage_root,
            compact_license_attempt_payloads=compact_license_attempt_payloads,
        )
        for task in cluster_service.campaign_tasks(campaign)
    ]
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
    """Return logical cases admitted while waiting for launch pacing."""
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
        "retrying": set(),
    }
    for view in task_views:
        if not _view_consumes_admission(view):
            continue
        key = _task_identity_key(view)
        if view.get("license_retry_active") is True:
            category = "retrying"
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
        categories["retrying" if mode == "license_retry" else "starting"].add(key)
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


def _campaign_retry_launch_spacing_seconds(manifest: Mapping[str, Any]) -> float | None:
    """Return derived retry-launch spacing or no gate for one admission case."""
    submission = manifest["submission_config"]
    maximum = submission["max_admission_cases"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        message = "Campaign max_admission_cases must be a positive integer."
        raise ValueError(message)
    retry = submission["temporary_license_retry"]
    if retry["enabled"] is not True or maximum == 1:
        return None
    initial_delay = float(retry["initial_delay_seconds"])
    if initial_delay <= 0.0:
        message = "Temporary-license initial retry delay must be positive."
        raise ValueError(message)
    return initial_delay / maximum


def _license_pressure_active(task_views: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether this campaign currently has pre-solver license pressure."""
    return any(view["state"] == "license_blocked" or view.get("license_retry_active") is True for view in task_views)


def _last_pre_solver_launch_at(manifest: Mapping[str, Any]) -> datetime | None:
    """Return the latest durable submitted or submitting pre-solver launch."""
    timestamps: list[datetime] = []
    for record in manifest["submissions"]:
        if record["status"] not in {"submitted", "submitting"}:
            continue
        timestamps.append(
            _parse_utc_timestamp(
                str(record["recorded_at"]),
                label="Campaign pre-solver launch timestamp",
            )
        )
    return max(timestamps, default=None)


def _campaign_retry_launch_pacing(
    manifest: Mapping[str, Any],
    task_views: Sequence[Mapping[str, Any]],
    *,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Return the durable campaign-level gate for pre-solver license launches."""
    spacing = _campaign_retry_launch_spacing_seconds(manifest)
    active = spacing is not None and _license_pressure_active(task_views)
    last_launch = _last_pre_solver_launch_at(manifest) if active else None
    current = _parse_utc_timestamp(_utc_now(), label="Campaign controller clock") if at is None else at
    if current.tzinfo is None or current.utcoffset() is None:
        message = "Campaign controller clock must be timezone-aware."
        raise ValueError(message)
    current = current.astimezone(timezone.utc)
    next_launch = None if last_launch is None else last_launch + timedelta(seconds=spacing or 0.0)
    return {
        "active": active,
        "spacing_seconds": spacing,
        "last_launch_at": None if last_launch is None else last_launch.isoformat(),
        "next_launch_at": None if next_launch is None else next_launch.isoformat(),
        "launch_allowed": not active or next_launch is None or current >= next_launch,
    }


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
    """Durably admit one fresh case before its paced Slurm launch."""
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

        pacing = _campaign_retry_launch_pacing(manifest, task_views)
        if pacing["launch_allowed"] is not True:
            manifest["state"] = "active" if pending_jobs or running_jobs else "license_blocked"
            common.serialization.atomic_write_json(
                campaign_evidence.campaign_run_manifest_path(
                    run_id,
                    storage_root=storage_root,
                ),
                manifest,
            )
            return manifest

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
            except Exception:  # noqa: BLE001 -- durable replay evidence owns diagnostics
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
    elif classified_state in {"successful", "license_blocked", "admission_waiting", "never_started"}:
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
    license_retry_launch_pacing = _campaign_retry_launch_pacing(manifest, task_views)
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
        "admission": admission,
        "license_retry_eligible_cases": license_retry_eligible_cases,
        "license_retry_launch_pacing": license_retry_launch_pacing,
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
) -> dict[str, Any]:
    """Load or immutably establish interruption-recovery transfer evidence."""
    journal_path = staging / _TRANSFER_PUBLICATION_JOURNAL
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
    terminal = validate_terminal_campaign(run_id, storage_root=staging)
    plan = campaign_transfer_plan(run_id, storage_root=staging)
    source_inventory = campaign_evidence.transfer_inventory_from_plan(
        plan,
        storage_root=staging,
    )
    journal = {
        **expected_identity,
        "terminal": {
            "campaign_id": terminal["campaign_id"],
            "git_commit": terminal["git_commit"],
        },
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
    )
    outcomes = _publish_incoming_campaign_directories(
        journal,
        staging=staging,
        destination=destination,
    )
    plan = journal["plan"]
    source_inventory = journal["source_inventory"]
    terminal = journal["terminal"]
    if not isinstance(plan, dict) or not isinstance(source_inventory, dict) or not isinstance(terminal, dict):
        message = "Transfer publication journal terminal evidence is malformed."
        raise TypeError(message)
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
