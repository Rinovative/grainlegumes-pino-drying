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
  - Poll indefinitely, cancel jobs, or invent absent scheduler identities
===============================================================================
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common

from . import generation_cluster as cluster_service
from . import generation_config as config_service
from . import generation_runtime as batch_runtime
from . import generation_source as source_service

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
        state not in {"submitting", "submitted"}
        or not isinstance(job_ids, list)
        or len(job_ids) > 1
        or not all(isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id) is not None for job_id in job_ids)
    ):
        message = f"Campaign-run submission state is malformed: {run_id}."
        raise ValueError(message)
    if (state == "submitting" and job_ids) or (state == "submitted" and not job_ids):
        message = f"Campaign-run scheduler identity disagrees with state {state!r}: {run_id}."
        raise ValueError(message)
    common.paths.validate_logical_name(manifest.get("scheduler_job_name"), label="scheduler_job_name")
    log_directory = manifest.get("scheduler_log_directory")
    if not isinstance(log_directory, str) or not Path(log_directory).is_absolute():
        message = f"Campaign-run scheduler log directory is malformed: {run_id}."
        raise ValueError(message)
    command = manifest.get("submission_command")
    if not isinstance(command, list) or not command or not all(isinstance(argument, str) for argument in command):
        message = f"Campaign-run submission command is malformed: {run_id}."
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
            normalized = {**existing, "slurm_job_ids": [], "state": "submitting"}
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
        manifest = {**intent, "slurm_job_ids": [job_token], "state": "submitted"}
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
    """Atomically complete an interrupted submission receipt."""
    if not job_ids:
        return dict(manifest)
    if len(job_ids) != 1:
        message = f"Scheduler recovery for {run_id!r} is ambiguous across job IDs {list(job_ids)}."
        raise RuntimeError(message)
    updated = {**manifest, "slurm_job_ids": list(job_ids), "state": "submitted"}
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
        if job_ids:
            scheduler_selection = ",".join(job_ids)
            squeue_command = ["squeue", "--noheader", "--jobs", scheduler_selection, "--format=%i|%T|%R"]
            sacct_command = ["sacct", "--noheader", "--parsable2", "--jobs", scheduler_selection, "--format=JobIDRaw,State,ExitCode"]
        else:
            scheduler_name = str(manifest["scheduler_job_name"])
            squeue_command = ["squeue", "--noheader", f"--name={scheduler_name}", "--format=%i|%T|%R"]
            sacct_command = ["sacct", "--noheader", "--parsable2", f"--name={scheduler_name}", "--format=JobIDRaw,State,ExitCode"]
        squeue_output, squeue_error = _scheduler_output(squeue_command)
        sacct_output, sacct_error = _scheduler_output(sacct_command)
    if not job_ids:
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
        token.split("|")[1].split("+", maxsplit=1)[0] for token in sacct_output.splitlines() if len(token.split("|")) >= _SCHEDULER_REQUIRED_FIELDS
    }
    complete = all(batch["terminal_manifest_available"] for batch in batches)
    if complete:
        state = "complete"
        next_command = f"collect {run_id}"
    elif states.intersection(_TERMINAL_FAILURE_STATES):
        state = "failed"
        next_command = f"status {run_id}"
    elif squeue_output:
        state = "active"
        next_command = f"status {run_id}"
    elif manifest["state"] == "submitting":
        state = "submission_pending_or_unknown"
        next_command = f"status {run_id}"
    else:
        state = "pending_or_scheduler_history"
        next_command = f"status {run_id}"
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
