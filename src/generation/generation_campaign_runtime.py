"""
===============================================================================
generation_campaign_runtime.py
===============================================================================
Persist, submit, inspect, and terminally validate one campaign run.
Responsibilities:
  - Bind campaign execution to one clean exact Git commit and resource plan
  - Submit the single shared Slurm worker pool and persist scheduler identities
  - Reconstruct scheduler/case status after the launch process exits
Design principles:
  - Campaign-run identity binds science, code, and execution without altering DOE
  - Terminal evidence is immutable and requires every batch manifest
  - Scheduler commands use argument vectors and non-interactive process boundaries
This module does NOT:
  - Generate scientific inputs, implement SSH/rsync, or build dataset packages
  - Poll indefinitely, fabricate scheduler identities, or delete remote sources
===============================================================================
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common

from . import generation_cluster as cluster_service
from . import generation_config as config_service
from . import generation_profiles as profiles
from . import generation_runtime as batch_runtime
from . import generation_source as source_service
from . import generation_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Mapping

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_SCHEDULER_REQUIRED_FIELDS: Final = 2
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
_SCHEDULER_LOG_PATTERN: Final = re.compile(r"slurm-([0-9]+)_[0-9]+[.](?:out|err)")
_TRANSFER_OPERATIONAL_RECEIPTS: Final = frozenset(
    {
        "all_workflow.json",
        "cpu_source_cleanup.json",
        "dataset_packages_complete.json",
        "transfer_complete.json",
    }
)


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


def campaign_run_id(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    resource_plan: cluster_service.ResourcePlan,
) -> str:
    """Return the digest-bound immutable campaign-run identifier."""
    payload = {
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_ids": [batch.batch_id for batch in campaign.batches],
        "git_commit": source_service.validate_git_commit(git_commit),
        "resource_plan": asdict(resource_plan),
        "wall_time": campaign.execution_values["cluster"]["wall_time"],
    }
    digest = common.serialization.canonical_json_sha256(payload)
    return f"{campaign.campaign_name}__{digest[:16]}"


def _run_directory(run_id: str, *, storage_root: Path | str | None) -> Path:
    """Return one persistent campaign-run metadata directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    return common.paths.get_generation_meta_root(storage_root=storage_root) / "campaigns" / safe_id


def _run_manifest_path(run_id: str, *, storage_root: Path | str | None) -> Path:
    """Return the persistent campaign-run manifest path."""
    return _run_directory(run_id, storage_root=storage_root) / "campaign_run.json"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
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


def _campaign_config_path(value: Any) -> Path:
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
    if _campaign_config_path(value) != source:
        message = "Campaign configuration did not resolve to its persisted repository-relative path."
        raise RuntimeError(message)
    return value


def load_campaign_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load one campaign run after its launch process has exited."""
    manifest = _load_json(
        _run_manifest_path(run_id, storage_root=storage_root),
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
    common.paths.validate_logical_name(manifest.get("scheduler_job_name"), label="scheduler_job_name")
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
    _campaign_config_path(manifest.get("campaign_config"))
    return manifest


def _campaign_from_manifest(manifest: Mapping[str, Any]) -> config_service.CampaignConfig:
    """Resolve the exact predeclared batch selection persisted by a run."""
    campaign = config_service.load_campaign_config(_campaign_config_path(manifest["campaign_config"]))
    if campaign.campaign_id != manifest["campaign_id"]:
        message = "Campaign configuration identity changed after launch."
        raise RuntimeError(message)
    return campaign.select_batches(tuple(manifest["selected_batch_names"]))


def campaign_for_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> config_service.CampaignConfig:
    """Return the exact repository campaign and batch selection bound to a run."""
    return _campaign_from_manifest(load_campaign_run(run_id, storage_root=storage_root))


def _new_campaign_manifest(
    campaign: config_service.CampaignConfig,
    *,
    run_id: str,
    requested_commit: str,
    resource_plan: cluster_service.ResourcePlan,
    command: list[str],
    run_directory: Path,
    scheduler_job_name: str,
    scheduler_log_directory: Path,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return the durable pre-submission intent for one exact campaign run."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
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
        "slurm_job_ids": [],
        "scheduler_job_name": scheduler_job_name,
        "scheduler_log_directory": str(scheduler_log_directory),
        "submission_command": command,
        "submission_history": [
            {
                "attempt": 1,
                "kind": "initial",
                "recorded_at": _utc_now(),
                "command": command,
                "job_id": None,
            }
        ],
        "resource_plan": asdict(resource_plan),
        "wall_time": campaign.execution_values["cluster"]["wall_time"],
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
                        batch.batch_id,
                        stage="raw",
                        storage_root=storage,
                    )
                ),
                "processed_directory": str(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_id,
                        stage="processed",
                        storage_root=storage,
                    )
                ),
            }
            for batch in campaign.batches
        ],
        "dataset_packages": list(campaign.dataset_packages),
        "state": "submitting",
    }


def plan_campaign(
    campaign: config_service.CampaignConfig,
    *,
    resource_plan: cluster_service.ResourcePlan,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve one exact campaign plan without creating files or submitting."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    if not storage.is_dir():
        message = f"Campaign plan requires the prepared storage root: {storage}"
        raise FileNotFoundError(message)
    run_id = campaign_run_id(
        campaign,
        git_commit=requested_commit,
        resource_plan=resource_plan,
    )
    run_directory = _run_directory(run_id, storage_root=storage)
    log_directory = run_directory / "scheduler"
    scheduler_job_name = f"vp2-{run_id.rsplit('__', maxsplit=1)[-1]}"
    command = cluster_service.build_campaign_slurm_submission_command(
        campaign,
        plan=resource_plan,
        scheduler_log_directory=log_directory,
        scheduler_job_name=scheduler_job_name,
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
            "failures": [
                str(
                    _state_batch_root_for_plan(
                        batch,
                        storage_root=storage,
                    )
                    / "failures"
                )
                for batch in campaign.batches
            ],
            "publications": [
                {
                    "batch_id": batch.batch_id,
                    "raw": str(
                        common.paths.resolve_generated_batch_dir(
                            batch.batch_id,
                            stage="raw",
                            storage_root=storage,
                        )
                    ),
                    "processed": str(
                        common.paths.resolve_generated_batch_dir(
                            batch.batch_id,
                            stage="processed",
                            storage_root=storage,
                        )
                    ),
                }
                for batch in campaign.batches
            ],
        },
        "templates": {
            profile_id: {
                "path": str(profiles.get_profile(profile_id).template_path),
                "sha256": profiles.get_profile(profile_id).template_sha256,
            }
            for profile_id in profiles.available_profiles()
        },
        "execution_config": campaign.execution_values,
        "resource_plan": asdict(resource_plan),
        "submission_command": command,
    }


def _state_batch_root_for_plan(
    batch: config_service.GenerationConfig,
    *,
    storage_root: Path,
) -> Path:
    """Return a batch state path without creating it."""
    return common.paths.get_generation_state_root(storage_root=storage_root) / batch.profile.id / batch.batch_id


def submit_campaign(
    campaign: config_service.CampaignConfig,
    *,
    resource_plan: cluster_service.ResourcePlan,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Persist intent, submit one shared worker pool, and record its job identity."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    run_id = campaign_run_id(
        campaign,
        git_commit=requested_commit,
        resource_plan=resource_plan,
    )
    run_directory = _run_directory(run_id, storage_root=storage_root)
    run_directory.mkdir(parents=True, exist_ok=True)
    scheduler_log_directory = run_directory / "scheduler"
    scheduler_log_directory.mkdir(exist_ok=True)
    scheduler_job_name = f"vp2-{run_id.rsplit('__', maxsplit=1)[-1]}"
    command = cluster_service.build_campaign_slurm_submission_command(
        campaign,
        plan=resource_plan,
        scheduler_log_directory=scheduler_log_directory,
        scheduler_job_name=scheduler_job_name,
    )
    intent = _new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit=requested_commit,
        resource_plan=resource_plan,
        command=command,
        run_directory=run_directory,
        scheduler_job_name=scheduler_job_name,
        scheduler_log_directory=scheduler_log_directory,
        storage_root=storage_root,
    )
    path = run_directory / "campaign_run.json"
    lock_path = run_directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if path.exists():
            existing = load_campaign_run(run_id, storage_root=storage_root)
            normalized = {
                **existing,
                "slurm_job_ids": [],
                "submission_history": intent["submission_history"],
                "state": "submitting",
            }
            if normalized != intent:
                message = f"Existing campaign-run manifest conflicts with {run_id!r}."
                raise FileExistsError(message)
            if existing["state"] == "submitted":
                return existing
            message = (
                f"Campaign run {run_id!r} has a durable submission intent but no recovered job ID. "
                "Use campaign-status before considering a new submission."
            )
            raise RuntimeError(message)
        common.serialization.atomic_write_json(path, intent)
        environment = os.environ.copy()
        environment["GENERATION_GIT_COMMIT"] = requested_commit
        environment["GENERATION_CAMPAIGN_RUN_ID"] = run_id
        result = subprocess.run(  # noqa: S603 -- typed cluster service builds the Slurm argv
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        job_token = result.stdout.strip().split(";", maxsplit=1)[0]
        if _JOB_ID_PATTERN.fullmatch(job_token) is None:
            message = f"Slurm returned an invalid parsable job identifier: {result.stdout!r}."
            raise RuntimeError(message)
        history = [dict(attempt) for attempt in intent["submission_history"]]
        history[-1]["job_id"] = job_token
        manifest = {
            **intent,
            "slurm_job_ids": [job_token],
            "submission_history": history,
            "state": "submitted",
        }
        common.serialization.atomic_write_json(path, manifest)
    return manifest


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


def resume_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Submit a fresh worker pool for only non-validated campaign cases."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    current_commit = _repository_commit()
    if current_commit != manifest["git_commit"]:
        message = f"CPU checkout commit {current_commit} does not match run commit {manifest['git_commit']}."
        raise RuntimeError(message)
    active_output, active_error = _scheduler_output(
        [
            "squeue",
            "--noheader",
            "--jobs",
            ",".join(manifest["slurm_job_ids"]),
            "--format=%i|%T|%R",
        ]
    )
    if active_error is not None:
        message = f"Cannot prove previous Slurm attempts are inactive: {active_error}"
        raise RuntimeError(message)
    if active_output:
        message = "Cannot resume while a previous campaign Slurm attempt is active."
        raise RuntimeError(message)
    if manifest["state"] == "resubmitting":
        message = "A resume intent is unresolved; run campaign-status before resubmitting."
        raise RuntimeError(message)
    campaign = _campaign_from_manifest(manifest)
    remaining = sum(
        not batch_runtime.completed_case_is_valid(
            batch,
            case_index,
            storage_root=storage_root,
        )
        for batch in campaign.batches
        for case_index in batch.case_indices
    )
    if remaining == 0:
        message = f"Campaign {run_id!r} has no incomplete cases to resume."
        raise RuntimeError(message)
    original = manifest["resource_plan"]
    plan = cluster_service.build_resource_plan(
        max_nodes=original["max_nodes"],
        cases_per_node=original["cases_per_node"],
        cores_per_case=original["cores_per_case"],
        max_parallel_cases=original["max_parallel_cases"],
        cores_per_node=original["cores_per_node"],
        remaining_cases=remaining,
    )
    command = cluster_service.build_campaign_slurm_submission_command(
        campaign,
        plan=plan,
        scheduler_log_directory=Path(manifest["scheduler_log_directory"]),
        scheduler_job_name=manifest["scheduler_job_name"],
    )
    path = _run_manifest_path(run_id, storage_root=storage_root)
    lock_path = _run_directory(run_id, storage_root=storage_root) / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        manifest = load_campaign_run(run_id, storage_root=storage_root)
        history = [dict(attempt) for attempt in manifest["submission_history"]]
        history.append(
            {
                "attempt": len(history) + 1,
                "kind": "resume",
                "recorded_at": _utc_now(),
                "command": command,
                "job_id": None,
            }
        )
        intent = {
            **manifest,
            "submission_command": command,
            "submission_history": history,
            "state": "resubmitting",
        }
        common.serialization.atomic_write_json(path, intent)
        environment = os.environ.copy()
        environment["GENERATION_GIT_COMMIT"] = manifest["git_commit"]
        environment["GENERATION_CAMPAIGN_RUN_ID"] = run_id
        result = subprocess.run(  # noqa: S603 -- validated Slurm argv from the cluster service
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        job_id = _parse_submitted_job_id(result.stdout)
        history[-1]["job_id"] = job_id
        updated = {
            **intent,
            "slurm_job_ids": [*manifest["slurm_job_ids"], job_id],
            "submission_history": history,
            "state": "submitted",
        }
        common.serialization.atomic_write_json(path, updated)
    return load_campaign_run(run_id, storage_root=storage_root)


def cancel_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Request cancellation of every persisted Slurm attempt and record it."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    job_ids = list(manifest["slurm_job_ids"])
    if not job_ids:
        message = f"Campaign {run_id!r} has no persisted Slurm job IDs to cancel."
        raise RuntimeError(message)
    command = ["scancel", *job_ids]
    result = subprocess.run(  # noqa: S603 -- persisted numeric Slurm job IDs only
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    receipt_path = (
        _run_directory(
            run_id,
            storage_root=storage_root,
        )
        / "cancellations.json"
    )
    existing: list[dict[str, Any]] = []
    if receipt_path.exists():
        raw = _load_json(receipt_path, label="campaign cancellation receipt")
        if raw.get("schema_kind") != "generation_campaign_cancellations":
            message = f"Campaign cancellation receipt is malformed: {receipt_path}"
            raise ValueError(message)
        attempts = raw.get("attempts")
        if not isinstance(attempts, list):
            message = f"Campaign cancellation attempts are malformed: {receipt_path}"
            raise ValueError(message)
        existing = attempts
    attempt = {
        "recorded_at": _utc_now(),
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    receipt = {
        "schema_kind": "generation_campaign_cancellations",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "attempts": [*existing, attempt],
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    updated = {**manifest, "state": "cancel_requested"}
    common.serialization.atomic_write_json(
        _run_manifest_path(run_id, storage_root=storage_root),
        updated,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        message = f"Slurm cancellation failed after its receipt was persisted: {detail}"
        raise RuntimeError(message)
    return receipt


def campaign_accounting(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact squeue and sacct commands and their current output."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    selection = ",".join(manifest["slurm_job_ids"])
    squeue_command = [
        "squeue",
        "--noheader",
        "--jobs",
        selection,
        "--format=%i|%T|%R",
    ]
    sacct_command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        selection,
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
    ]
    squeue_output, squeue_error = _scheduler_output(squeue_command)
    sacct_output, sacct_error = _scheduler_output(sacct_command)
    return {
        "campaign_run_id": run_id,
        "squeue": {
            "command": squeue_command,
            "output": squeue_output,
            "error": squeue_error,
        },
        "sacct": {
            "command": sacct_command,
            "output": sacct_output,
            "error": sacct_error,
        },
    }


def record_worker_interruption(
    run_id: str,
    *,
    storage_root: Path | str | None,
    signal_name: str,
    exit_code: int,
) -> Path:
    """Persist one best-effort Slurm worker interruption receipt."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    job_id = os.environ.get("SLURM_JOB_ID")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not job_id or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Worker interruption evidence requires SLURM_JOB_ID."
        raise ValueError(message)
    directory = _run_directory(run_id, storage_root=storage_root) / "interruptions"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "none" if array_id is None else array_id
    path = directory / f"{job_id}_{suffix}.json"
    payload = {
        "schema_kind": "generation_worker_interruption",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "recorded_at": _utc_now(),
        "signal": signal_name,
        "exit_code": exit_code,
        "slurm_job_id": job_id,
        "slurm_array_task_id": array_id,
        "hostname": os.uname().nodename,
    }
    common.serialization.atomic_write_json(path, payload)
    return path


def _scheduler_job_ids(
    *outputs: str,
    scheduler_log_directory: str,
) -> tuple[str, ...]:
    """Recover root Slurm job IDs from scheduler output and durable log names."""
    recovered: set[str] = set()
    for output in outputs:
        for line in output.splitlines():
            token = line.strip().split("|", maxsplit=1)[0]
            match = _JOB_ID_PATTERN.match(token)
            if match is not None:
                recovered.add(match.group())
    log_directory = Path(scheduler_log_directory)
    if log_directory.is_dir() and not log_directory.is_symlink():
        for log_path in log_directory.iterdir():
            match = _SCHEDULER_LOG_PATTERN.fullmatch(log_path.name)
            if log_path.is_file() and not log_path.is_symlink() and match is not None:
                recovered.add(match.group(1))
    return tuple(sorted(recovered, key=int))


def _persist_recovered_job_ids(
    run_id: str,
    manifest: Mapping[str, Any],
    job_ids: tuple[str, ...],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Atomically complete an interrupted initial or resume receipt."""
    existing = list(manifest["slurm_job_ids"])
    new_ids = [job_id for job_id in job_ids if job_id not in existing]
    if not new_ids:
        return dict(manifest)
    history = [dict(attempt) for attempt in manifest["submission_history"]]
    pending = [attempt for attempt in history if attempt["job_id"] is None]
    if len(pending) != 1 or len(new_ids) != 1:
        message = f"Scheduler recovery for {run_id!r} is ambiguous across new job IDs {new_ids}."
        raise RuntimeError(message)
    pending[0]["job_id"] = new_ids[0]
    updated = {
        **manifest,
        "slurm_job_ids": [*existing, new_ids[0]],
        "submission_history": history,
        "state": "submitted",
    }
    common.serialization.atomic_write_json(
        _run_manifest_path(run_id, storage_root=storage_root),
        updated,
    )
    return load_campaign_run(run_id, storage_root=storage_root)


def _case_is_active(
    batch: config_service.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None,
) -> bool:
    """Return whether another process currently owns the case lock."""
    lock_path = batch_runtime.case_lock_path(
        batch,
        case_index,
        storage_root=storage_root,
    )
    manager = common.locking.exclusive_file_lock(lock_path, blocking=False)
    try:
        manager.__enter__()
    except common.locking.FileLockUnavailableError:
        return True
    manager.__exit__(None, None, None)
    return False


def _batch_status(
    batch: config_service.GenerationConfig,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return persistent per-batch case counts and terminal evidence."""
    completed = 0
    active = 0
    failed = 0
    for case_index in batch.case_indices:
        if batch_runtime.completed_case_is_valid(
            batch,
            case_index,
            storage_root=storage_root,
        ):
            completed += 1
        elif _case_is_active(batch, case_index, storage_root=storage_root):
            active += 1
        elif batch_runtime.case_failure_is_recorded(
            batch,
            case_index,
            storage_root=storage_root,
        ):
            failed += 1
    state_root = batch_runtime._state_batch_root(batch, storage_root=storage_root)  # noqa: SLF001
    quarantine = state_root / "quarantine"
    quarantined = len(tuple(quarantine.iterdir())) if quarantine.is_dir() else 0
    terminal_path = batch_runtime.batch_meta_directory(batch, storage_root=storage_root) / "batch_manifest.json"
    total = len(batch.case_indices)
    return {
        "batch_name": batch.batch_name,
        "batch_id": batch.batch_id,
        "planned": total,
        "completed": completed,
        "active": active,
        "failed": failed,
        "quarantined": quarantined,
        "pending": max(total - completed - active - failed, 0),
        "terminal_manifest": str(terminal_path),
        "terminal_manifest_available": terminal_path.is_file(),
    }


def campaign_status(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Reconstruct campaign, scheduler, batch, and case state."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    campaign = _campaign_from_manifest(manifest)
    batches = [_batch_status(batch, storage_root=storage_root) for batch in campaign.batches]
    job_ids = list(manifest["slurm_job_ids"])
    squeue_output = ""
    squeue_error = None
    sacct_output = ""
    sacct_error = None
    if query_scheduler:
        if job_ids and manifest["state"] != "resubmitting":
            scheduler_selection = ",".join(job_ids)
            squeue_command = ["squeue", "--noheader", "--jobs", scheduler_selection, "--format=%i|%T|%R"]
            sacct_command = ["sacct", "--noheader", "--parsable2", "--jobs", scheduler_selection, "--format=JobIDRaw,State,ExitCode"]
        else:
            scheduler_name = str(manifest["scheduler_job_name"])
            squeue_command = ["squeue", "--noheader", f"--name={scheduler_name}", "--format=%i|%T|%R"]
            sacct_command = ["sacct", "--noheader", "--parsable2", f"--name={scheduler_name}", "--format=JobIDRaw,State,ExitCode"]
        squeue_output, squeue_error = _scheduler_output(squeue_command)
        sacct_output, sacct_error = _scheduler_output(sacct_command)
    if not job_ids or manifest["state"] == "resubmitting":
        recovered = _scheduler_job_ids(
            squeue_output,
            sacct_output,
            scheduler_log_directory=str(manifest["scheduler_log_directory"]),
        )
        if recovered:
            manifest = _persist_recovered_job_ids(
                run_id,
                manifest,
                recovered,
                storage_root=storage_root,
            )
            job_ids = list(manifest["slurm_job_ids"])
    states = {
        token.split("|")[1].split("+", maxsplit=1)[0].split()[0]
        for token in sacct_output.splitlines()
        if len(token.split("|")) >= _SCHEDULER_REQUIRED_FIELDS and token.split("|")[1].strip()
    }
    squeue_states = {
        token.split("|")[1].split("+", maxsplit=1)[0].split()[0]
        for token in squeue_output.splitlines()
        if len(token.split("|")) >= _SCHEDULER_REQUIRED_FIELDS and token.split("|")[1].strip()
    }
    complete = all(batch["terminal_manifest_available"] for batch in batches)
    completed_cases = sum(batch["completed"] for batch in batches)
    failed_cases = sum(batch["failed"] for batch in batches)
    cancellation_receipt = (_run_directory(run_id, storage_root=storage_root) / "cancellations.json").is_file()
    transfer_receipt = (_run_directory(run_id, storage_root=storage_root) / "transfer_complete.json").is_file()
    failure_states = states.intersection(_TERMINAL_FAILURE_STATES.difference({"CANCELLED"}))
    if transfer_receipt:
        state = "transfer_complete"
        next_command = f"status {run_id}"
    elif complete:
        state = "publication_complete"
        next_command = f"transfer {run_id}"
    elif cancellation_receipt and not squeue_output and ("CANCELLED" in states or manifest["state"] == "cancel_requested"):
        state = "cancelled"
        next_command = f"resume {run_id}"
    elif failure_states or (failed_cases and not squeue_output):
        state = "partially_failed" if completed_cases else "failed"
        next_command = f"resume {run_id}"
    elif "RUNNING" in squeue_states:
        state = "running"
        next_command = f"status {run_id}"
    elif squeue_output:
        state = "submitted"
        next_command = f"status {run_id}"
    elif states and states.issubset({"COMPLETED"}):
        state = "completed"
        next_command = f"validate {run_id}"
    elif manifest["state"] in {"submitting", "resubmitting"}:
        state = "submission_pending_or_unknown"
        next_command = f"status {run_id}"
    else:
        state = "submitted"
        next_command = f"accounting {run_id}"
    return {
        "campaign_run_id": run_id,
        "campaign_state": state,
        "git_commit": manifest["git_commit"],
        "slurm_job_ids": job_ids,
        "scheduler_job_name": manifest["scheduler_job_name"],
        "scheduler_log_directory": manifest["scheduler_log_directory"],
        "squeue": {"output": squeue_output, "error": squeue_error},
        "sacct": {"output": sacct_output, "error": sacct_error},
        "batches": batches,
        "remote_storage_root": manifest["remote_storage_root"],
        "campaign_meta_directory": manifest["campaign_meta_directory"],
        "suggested_next_command": next_command,
    }


def finalize_campaign_run(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish immutable terminal campaign evidence after all batches validate."""
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    if not manifest["slurm_job_ids"]:
        campaign_status(run_id, storage_root=storage_root)
        manifest = load_campaign_run(run_id, storage_root=storage_root)
        if not manifest["slurm_job_ids"]:
            message = f"Campaign scheduler identity has not been recovered for {run_id!r}."
            raise RuntimeError(message)
    campaign = _campaign_from_manifest(manifest)
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
    path = _run_directory(run_id, storage_root=storage_root) / "campaign_terminal.json"
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
    return _load_json(terminal_path, label="terminal campaign manifest")


def campaign_transfer_plan(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return terminally validated storage-relative directories for collection."""
    validate_terminal_campaign(run_id, storage_root=storage_root)
    manifest = load_campaign_run(run_id, storage_root=storage_root)
    campaign = _campaign_from_manifest(manifest)
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

    return {
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "git_commit": manifest["git_commit"],
        "campaign_config": manifest["campaign_config"],
        "campaign_directory": relative_directory(_run_directory(run_id, storage_root=storage_root)),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_count": len(batch.case_indices),
                "meta_directory": relative_directory(batch_runtime.batch_meta_directory(batch, storage_root=storage_root)),
                "raw_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_id,
                        stage="raw",
                        storage_root=storage_root,
                    )
                ),
                "processed_directory": relative_directory(
                    common.paths.resolve_generated_batch_dir(
                        batch.batch_id,
                        stage="processed",
                        storage_root=storage_root,
                    )
                ),
            }
            for batch in campaign.batches
        ],
    }


def _directory_identity(
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


def _transfer_inventory_from_plan(
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
        ignored_names = _TRANSFER_OPERATIONAL_RECEIPTS if relative_value == plan["campaign_directory"] else frozenset()
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


def campaign_transfer_inventory(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact terminal campaign files, byte count, and content hashes."""
    root = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    plan = campaign_transfer_plan(run_id, storage_root=root)
    return _transfer_inventory_from_plan(plan, storage_root=root)


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
    source_inventory = _transfer_inventory_from_plan(plan, storage_root=staging)
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
        ignored = _TRANSFER_OPERATIONAL_RECEIPTS if relative_value == plan["campaign_directory"] else frozenset()
        source_identity = _directory_identity(source, ignored_names=ignored)
        if target.exists():
            target_identity = _directory_identity(target, ignored_names=ignored)
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
        if _directory_identity(payload, ignored_names=ignored) != source_identity:
            message = f"Transfer copy changed staged identity: {relative_value!r}"
            raise RuntimeError(message)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.replace(target)
        if _directory_identity(target, ignored_names=ignored) != source_identity:
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
    destination_inventory = _transfer_inventory_from_plan(plan, storage_root=destination)
    if destination_inventory != source_inventory:
        message = "Published GPU campaign inventory differs from the staged transfer source."
        raise RuntimeError(message)
    terminal_path = _run_directory(run_id, storage_root=destination) / "campaign_terminal.json"
    receipt_path = _run_directory(run_id, storage_root=destination) / "transfer_complete.json"
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
    inventory = _transfer_inventory_from_plan(plan, storage_root=destination)
    terminal_path = _run_directory(run_id, storage_root=destination) / "campaign_terminal.json"
    receipt_path = _run_directory(run_id, storage_root=destination) / "transfer_complete.json"
    receipt = _load_json(receipt_path, label="transfer completion receipt")
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
