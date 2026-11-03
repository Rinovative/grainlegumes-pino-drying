"""Persist, feed, inspect, and terminally validate one campaign run.

Responsibilities:
  - Bind campaign execution to one clean exact Git commit and execution config
  - Submit Slurm node jobs in the node-job slots and dispatch one ticket per case step within one logical admission pool
  - Persist node-job and dispatch identity as intents before and as records after each scheduler or ticket action
  - Derive one scheduler row per dispatch from step accounting, the allocation state, and the worker receipts
  - Replay a persisted cancellation on every feed until the node jobs have ended
  - Record the success decision of a validly published case whose step ended without writing it
  - Record the failed decision of a case whose solve the runtime stopped itself (stalled, timed out, solver error) once its step ended
  - Record each attempt's typed runtime reason; write the derived ``cases.csv`` view when the campaign settles complete
  - Validate terminal case evidence and publish exact campaign inventories

Design principles:
  - A node job owns an allocation and a dispatch owns one case step in it; the feeder is the only writer of run.json, tickets, and control files
  - Tickets are published once and recovered by exact replay, so a crash never duplicates or loses a dispatch
  - Durable case evidence and scheduler accounting make resume duplicate-safe

This module does NOT:
  - Construct scientific cases, execute COMSOL, or build dataset packages
  - Poll indefinitely, submit a whole campaign queue, or delete remote sources
  - Run case steps itself: the generic node worker consumes the tickets

Notes:
  Entry-point flow:
    - `plan_campaign` returns a non-mutating execution plan.
    - `submit_campaign` creates or continues persisted execution and invokes the sole `feed_campaign` reconciliation and admission path.
    - `resume_campaign` re-enters that feeder; `run_campaign_case_job` is the per-step case worker boundary.
    - `campaign_status` reconstructs current campaign, scheduler, batch, and case state for inspection; each case carries
      the physical start and horizon of its batch's persisted resolved configuration, its exact ``parent_failed_case_id``
      and its typed failure reason.
  Feed cycle order:
    - Plan replacements for failed slots within the replacement capacity; when none remains and every candidate is
      decided, end ``blocked`` after the scheduler read, draining every open node job as a completion does.
    - Resolve an interrupted node-job submission, read the scheduler, then observe ended node jobs and finish pending leases.
    - Finish interrupted drains and interrupted dispatches before any new work.
    - Reconcile every case, decide each validly published case whose step ended undecided, and check completion by candidate decisions.
    - Decide each runtime-stopped solve whose latest dispatch has terminal step accounting and no active dispatch; a feed
      that records one returns at once, so the next feed plans its replacement before any idle node job is drained.
    - On completion seal the selected pool, write ``cases.csv`` atomically, and drain every node job.
    - Dispatch cases into node jobs with free ticket capacity, and submit a node job on a free seat when none has.
    - Drain a node job with a stop ticket once nothing more is dispatchable or the campaign is settled.
  Dispatch evidence:
    - A dispatch is identified by its dispatch key ``<allocation>-<submission_index>``; its step row decides its state, not the allocation.
    - Without a step row it waits while its node job is alive and is persisted as ``not_started`` once the node job ended, or its
      finished receipt is, at least the grace period ago; such a case restarts without consuming an attempt.
    - Only the latest started dispatch of a case owns the case state; attempt evidence of an older dispatch is never admitted.

"""

from __future__ import annotations

import copy
import getpass
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Final

from src import common

from .cases import generation_cases_admission as admission_service
from .cases import generation_cases_config as config_service
from .cases import generation_cases_input as input_service
from .contracts import generation_contracts_source as source_service
from .publication import generation_publication_attempt as attempt_service
from .publication import generation_publication_campaign_evidence as campaign_evidence
from .publication import generation_publication_storage as storage_service
from .runtime import generation_runtime_batch as batch_runtime
from .runtime import generation_runtime_cluster as cluster_service
from .runtime import generation_runtime_dispatch_identity as dispatch_identity
from .runtime import generation_runtime_license as license_service
from .runtime import generation_runtime_node as node_service
from .runtime import generation_runtime_progress as progress_service
from .runtime import generation_runtime_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
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
_TERMINAL_ACCOUNTING_STATES: Final = frozenset({"COMPLETED", *_TERMINAL_FAILURE_STATES})
_ACTIVE_PENDING_STATE: Final = "PENDING"
_ATTEMPT_INDEX_ENVIRONMENT: Final = "GENERATION_ATTEMPT_INDEX"
_CANCELLATION_SCHEMA_VERSION: Final = 2
_CANCEL_STATES: Final = frozenset({"cancel_requested", "force_cancel_requested"})
_DISPATCH_BINDING_WAIT_SECONDS: Final = 120.0  # NFS visibility of the dispatch record to a step that starts right after its ticket
_DISPATCH_BINDING_POLL_SECONDS: Final = 2.0
_NODE_INTENT_GRACE_SECONDS: Final = 120.0  # an unresolved node-job submission is proven absent only this long after its intent
_NODE_LEASE_ATTEMPTS: Final = 3
_NODE_LEASE_RETRY_SECONDS: Final = 2.0
_NOT_STARTED_GRACE_SECONDS: Final = (
    120.0  # a missing step row proves a never started step only this long after the node job ended or the step finished
)
_DISPATCH_TICKETS_PER_CASE_SLOT: Final = 2  # unresolved dispatches per node job: two per concurrent case slot keep the next steps queued
_ALIVE_NODE_JOB_STATES: Final = frozenset({"PENDING", "RUNNING", "CONFIGURING"})
_SETTLED_CAMPAIGN_STATES: Final = frozenset({"complete", "completed_with_failures", "staged", "failure_threshold_reached", "blocked"})
_MAX_CONTROLLER_HANDLE_BYTES: Final = 255
_RECONCILIATION_PROGRESS_DELAY_SECONDS: Final = 2.0
_RECONCILIATION_PROGRESS_HEARTBEAT_SECONDS: Final = 30.0
_MATERIAL_JOB_CODES: Final = {
    "lentil": "lentil",
    "chickpea": "chickpea",
    "kidney_bean": "kidney",
    "field_pea": "fieldpea",
    "rapeseed": "rapeseed",
    "sunflower_seed": "sunflower",
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
    """Bind one attempt to its exact campaign run; ``current`` marks that the case's latest started dispatch wrote it."""

    attempt: attempt_service.AttemptEvidence
    current: bool


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
    return datetime.now(UTC).isoformat()


def _repository_commit() -> str:
    """Return the clean repository HEAD commit, or the verified commit of the admitted source tree."""
    if common.paths.source_tree_id() is not None:
        return source_service.verify_admitted_source_tree().git_commit
    repository = common.paths.get_project_root()
    status = subprocess.run(  # noqa: S603 -- fixed Git argument vector
        ["git", "-C", str(repository), "status", "--porcelain"],  # noqa: S607 -- site PATH owns Git
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        if os.environ.get("GENERATION_SOURCE_SNAPSHOT") is None or os.environ.get("GENERATION_SOURCE_SHA256") is None:
            message = "Campaign launch requires a clean shared repository or an admitted exact source snapshot."
            raise RuntimeError(message)
        source_service.validate_admitted_source_before_publication()
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
    """Return the stable source campaign controller handle."""
    source_service.validate_git_commit(git_commit)
    handle = f"source~{campaign.profile.id}~{campaign.source_name}"
    if len(handle.encode("ascii")) > _MAX_CONTROLLER_HANDLE_BYTES:
        msg = "Source campaign controller handle exceeds a filesystem component limit."
        raise ValueError(msg)
    return handle


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
        "node_jobs": int(cluster["node_jobs"]),
        "cores_per_case": int(cluster["cores_per_case"]),
        "concurrent_cases_per_node": int(cluster["concurrent_cases_per_node"]),
        "memory_per_case": cluster["memory_per_case"],
        "node_job_memory": cluster["node_job_memory"],
        "node_job_wall_time": cluster["node_job_wall_time"],
        "maximum_failed_cases": int(campaign.execution_values["runtime"]["maximum_failed_cases"]),
        "license_wait": copy.deepcopy(campaign.execution_values["runtime"]["license_wait"]),
        "partition": cluster["partition"],
        "scheduler_options": list(cluster["scheduler_options"]),
    }


def _node_ticket_capacity(submission_config: Mapping[str, Any]) -> int:
    """Return the unresolved dispatches one node job may hold: two per concurrent case slot."""
    return _DISPATCH_TICKETS_PER_CASE_SLOT * int(submission_config["concurrent_cases_per_node"])


def _maximum_dispatch_capacity(submission_config: Mapping[str, Any]) -> int:
    """Return the unresolved dispatches all node jobs of the campaign may hold together."""
    return int(submission_config["node_jobs"]) * _node_ticket_capacity(submission_config)


_EXECUTION_RESOURCE_PATHS = (
    ("site", "cpu_host"),
    ("site", "partition"),
    ("site", "cores_per_node"),
    ("site", "python_module"),
    ("site", "comsol_module"),
    ("site", "python_executable"),
    ("site", "comsol_executable"),
    ("runtime", "timeout_seconds"),
    ("runtime", "graceful_stop_reserve_seconds"),
    ("runtime", "executable"),
    ("runtime", "module_initialization"),
    ("cluster", "cores_per_case"),
    ("cluster", "node_jobs"),
    ("cluster", "memory_per_case"),
    ("cluster", "node_memory_overhead"),
    ("cluster", "node_memory_limit"),
    ("cluster", "node_job_wall_time"),
    ("cluster", "concurrent_cases_per_node"),
    ("cluster", "node_job_memory"),
    ("cluster", "scheduler_options"),
    ("cluster", "cores_per_node"),
    ("cluster", "partition"),
    ("submission", "poll_interval_seconds"),
    ("submission", "max_admission_cases"),
)


def _assert_execution_resource_compatibility(original: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    """Allow only resource changes; retain original failure and data policies."""
    old = copy.deepcopy(dict(original))
    new = copy.deepcopy(dict(requested))
    for section, key in _EXECUTION_RESOURCE_PATHS:
        if section not in old or section not in new or key not in old[section] or key not in new[section]:
            message = f"Execution resource path {section}.{key} is absent."
            raise ValueError(message)
        new[section][key] = old[section][key]
    if new != old:
        msg = "Campaign continuation changes frozen execution policy outside the resource allowlist."
        raise ValueError(msg)


def _execution_binding(manifest: Mapping[str, Any], *, submission_index: int | None = None) -> dict[str, Any]:
    """Return the exact source and resources owning one submission epoch."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 evidence boundary

    epochs = lifecycle_service.execution_epochs(manifest)
    if epochs:
        if submission_index is None:
            return epochs[-1]
        matching = [record for record in epochs if record["first_submission_index"] <= submission_index]
        if matching:
            return matching[-1]
    return {
        "git_commit": manifest["git_commit"],
        "source_sha256": None,
        "source_snapshot": None,
        "source_snapshot_sha256": None,
        "execution_config_digest": manifest["execution_config_digest"],
        "submission_config": manifest["submission_config"],
    }


def _require_explicit_source_transition(run_id: str, *, source_changed: bool) -> None:
    """Refuse to continue a bound campaign with other code unless its launcher was asked to."""
    if source_changed and os.environ.get(source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE) != source_service.EXPLICIT_SOURCE_TRANSITION:
        message = (
            f"Campaign {run_id!r} is bound to its admitted source tree; its launcher continues it there. "
            "Pass --continue-with-current-code to continue it deliberately with the current code."
        )
        raise RuntimeError(message)


def bound_source_snapshot(run_id: str, *, storage_root: Path | str | None = None) -> Path | None:
    """
    Return the snapshot of the code state an existing campaign executes in.

    The latest execution epoch owns the binding; a campaign without epochs uses
    the snapshot admitted at its creation. ``None`` means the campaign does not
    exist yet, so a launch freezes the current code.

    Raises
    ------
    RuntimeError
        If the campaign exists but records no source snapshot.

    """
    if not campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root).is_file():
        return None
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    binding = _execution_binding(manifest)
    value = binding["source_snapshot"]
    if value is None:
        origin = manifest.get("source_snapshot")
        value = origin.get("path") if isinstance(origin, Mapping) else None
    if not isinstance(value, str) or not value:
        message = f"Campaign {run_id!r} records no source snapshot, so its code state cannot be frozen again."
        raise RuntimeError(message)
    return Path(value)


def _active_submission_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the current feeder capacity and worker allocation contract."""
    return _execution_binding(manifest)["submission_config"]


def _validate_binding_source(binding: Mapping[str, Any]) -> None:
    """Ensure this process matches the exact admitted source snapshot."""
    for key, variable in (
        ("source_sha256", "GENERATION_SOURCE_SHA256"),
        ("source_snapshot", "GENERATION_SOURCE_SNAPSHOT"),
        ("source_snapshot_sha256", "GENERATION_SOURCE_SNAPSHOT_SHA256"),
    ):
        expected = binding[key]
        if expected is not None and expected != os.environ.get(variable):
            message = f"Campaign execution source differs from its admitted {variable}."
            raise RuntimeError(message)


def _requested_execution_binding(campaign: config_service.CampaignConfig, commit: str) -> dict[str, Any]:
    """Capture the admitted source and complete resolved execution values."""
    return {
        "git_commit": commit,
        "source_sha256": os.environ.get("GENERATION_SOURCE_SHA256"),
        "source_snapshot": os.environ.get("GENERATION_SOURCE_SNAPSHOT"),
        "source_snapshot_sha256": os.environ.get("GENERATION_SOURCE_SNAPSHOT_SHA256"),
        "execution_values": copy.deepcopy(campaign.execution_values),
        "execution_config_digest": _execution_config_digest(campaign),
        "submission_config": _submission_config(campaign),
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
    *,
    task_index: Mapping[tuple[str, int], cluster_service.CampaignTask] | None = None,
) -> cluster_service.CampaignTask:
    """Re-resolve one persisted task against exact current campaign membership."""
    case_index = payload.get("case_index")
    if isinstance(case_index, bool) or not isinstance(case_index, int):
        message = "Persisted campaign task has no integer case index."
        raise TypeError(message)
    if task_index is None:
        task = cluster_service.require_campaign_task(
            campaign,
            batch_name=str(payload.get("batch_name")),
            case_index=case_index,
        )
    else:
        persisted_task = task_index.get((str(payload.get("batch_id")), case_index))
        if persisted_task is None:
            message = "Persisted campaign task is not configured campaign membership."
            raise ValueError(message)
        task = persisted_task
    persisted = {key: payload.get(key) for key in ("batch_name", "batch_id", "case_index", "case_id")}
    if _task_payload(task) != persisted:
        message = "Persisted campaign task no longer matches campaign membership."
        raise ValueError(message)
    return task


def _profile_job_code(campaign: config_service.CampaignConfig) -> str:
    """Return the compact readable profile component for Slurm names."""
    return "td" if campaign.profile.id == "transient_drying" else "sf"


def _step_name(
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    submission_index: int,
) -> str:
    """Return the readable, campaign-unique Slurm step name of one dispatch."""
    if isinstance(submission_index, bool) or not isinstance(submission_index, int) or submission_index < 1:
        message = "Campaign dispatch submission_index must be positive."
        raise ValueError(message)
    family = campaign.batch(task.batch_name).material_family
    material = _MATERIAL_JOB_CODES.get(family) or re.sub(r"[^a-z0-9]+", "", family.lower())
    value = f"{_profile_job_code(campaign)}-{material}-c{task.case_index:04d}-s{submission_index}"
    if len(value) > campaign_evidence.MAX_STEP_NAME_LENGTH or campaign_evidence.STEP_NAME_PATTERN.fullmatch(value) is None:
        message = f"Campaign Slurm step name is unsafe or exceeds {campaign_evidence.MAX_STEP_NAME_LENGTH} characters: {value!r}."
        raise ValueError(message)
    return value


def _batch_state_directory(
    batch: config_service.GenerationConfig,
    *,
    storage_root: Path | str | None,
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
    input_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return durable feeder state before the first node-job submission."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    manifest = {
        "schema_kind": "generation_campaign_run",
        "schema_version": campaign_evidence.RUN_MANIFEST_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "source_name": campaign.source_name,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_config": _repository_relative_campaign_config(campaign),
        "simulation_profile": campaign.profile.id,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "git_commit": requested_commit,
        "execution_config_digest": _execution_config_digest(campaign),
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
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
                "input_generation_id": str(input_sources[batch.batch_name]["input_generation_id"]),
                "input_source_git_commit": str(input_sources[batch.batch_name]["source_git_commit"]),
                "input_case_indices": list(input_sources[batch.batch_name]["case_indices"]),
                "meta_directory": str(batch_runtime.batch_meta_directory(batch, storage_root=storage)),
                "raw_directory": str(common.paths.resolve_generation_cases_directory(batch.batch_storage_name, storage_root=storage)),
                "processed_directory": str(common.paths.resolve_generation_cases_directory(batch.batch_storage_name, storage_root=storage)),
            }
            for batch in campaign.batches
        ],
        "state": "ready",
    }
    snapshot_path = os.environ.get("GENERATION_SOURCE_SNAPSHOT")
    snapshot_sha256 = os.environ.get("GENERATION_SOURCE_SNAPSHOT_SHA256")
    if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION and snapshot_path is not None:
        if snapshot_sha256 is None:
            message = "Canonical campaign source snapshot lacks its exact manifest hash."
            raise ValueError(message)
        manifest["source_snapshot"] = {
            "path": snapshot_path,
            "sha256": snapshot_sha256,
            "source_fingerprint_sha256": os.environ["GENERATION_SOURCE_SHA256"],
        }
    return manifest


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
        message = f"Shared repository commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    if not storage.is_dir():
        message = f"Campaign plan requires the prepared storage root: {storage}"
        raise FileNotFoundError(message)
    run_id = campaign_run_id(campaign, git_commit=requested_commit)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    log_directory = common.paths.get_runtime_root().resolve() / "logs" / "generation"
    tasks = cluster_service.campaign_tasks(campaign)
    submission = _submission_config(campaign)
    time_limit, required_seconds = cluster_service.case_step_limits(float(campaign.execution_values["runtime"]["timeout_seconds"]))
    return {
        "schema_kind": "generation_campaign_plan",
        "schema_version": 2,
        "state": "planned",
        "filesystem_mutated": False,
        "campaign_run_id": run_id,
        "campaign_name": campaign.campaign_name,
        "source_name": campaign.source_name,
        "campaign_id": campaign.campaign_id,
        "campaign_config": _repository_relative_campaign_config(campaign),
        "git_commit": requested_commit,
        "paths": {
            "repository": str(common.paths.get_project_root().resolve()),
            "storage_root": str(storage),
            "run_root": str(run_directory),
            "log_root": str(log_directory),
            "failures": [str(_batch_state_directory(batch, storage_root=storage) / "failures") for batch in campaign.batches],
            "publications": [
                {
                    "batch_id": batch.batch_id,
                    "raw": str(common.paths.resolve_generation_cases_directory(batch.batch_storage_name, storage_root=storage)),
                    "processed": str(common.paths.resolve_generation_cases_directory(batch.batch_storage_name, storage_root=storage)),
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
        "submission_config": submission,
        "planned_cases": len(tasks),
        "node_job": {
            "seats": list(range(1, submission["node_jobs"] + 1)),
            "cores_per_node": int(campaign.execution_values["site"]["cores_per_node"]),
            "node_job_memory": submission["node_job_memory"],
            "node_job_wall_time": submission["node_job_wall_time"],
            "partition": submission["partition"],
            "concurrent_cases_per_node": submission["concurrent_cases_per_node"],
            "unresolved_dispatches_per_node_job": _node_ticket_capacity(submission),
        },
        "case_step": {
            "cores": submission["cores_per_case"],
            "memory": submission["memory_per_case"],
            "time_limit": time_limit,
            "required_node_seconds": required_seconds,
        },
        "submission_model": "Slurm node jobs in the node-job slots; one dispatched ticket per case step within one shared logical admission pool",
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


def _current_user_squeue_command(format_fields: str) -> list[str]:
    """Query only the current user's live queue without selecting expired IDs."""
    scheduler_user = getpass.getuser()
    if not scheduler_user or any(character in scheduler_user for character in "\x00\r\n"):
        message = "Could not determine the current scheduler user safely."
        raise RuntimeError(message)
    return ["squeue", "--noheader", f"--user={scheduler_user}", f"--format={format_fields}"]


def _owned_current_user_queue_rows(
    output: str,
    *,
    owned_job_ids: Collection[str],
    exact_field_count: int | None = None,
) -> dict[str, list[str]]:
    """Admit well-formed live rows, retaining only exact persisted root jobs."""
    owned = set(owned_job_ids)
    rows: dict[str, list[str]] = {}
    minimum_fields = 2
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) < minimum_fields or not fields[0] or not fields[1]:
            message = f"squeue returned malformed current-user liveness evidence: {raw_line!r}."
            raise ValueError(message)
        job_id = fields[0]
        if job_id not in owned:
            continue
        if (exact_field_count is not None and len(fields) != exact_field_count) or job_id in rows:
            message = f"squeue returned ambiguous owned scheduler liveness evidence: {raw_line!r}."
            raise ValueError(message)
        rows[job_id] = fields
    return rows


def query_active_scheduler_job_ids(
    job_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return exact persisted jobs still visible for the current Slurm user."""
    persisted = tuple(job_ids)
    if len(persisted) != len(set(persisted)) or any(not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None for job_id in persisted):
        message = "Scheduler liveness query received malformed persisted job IDs."
        raise ValueError(message)
    if not persisted:
        return ()
    command = _current_user_squeue_command("%i|%T")
    output, error = _scheduler_output(command)
    if error is not None:
        message = f"squeue failed for the current scheduler user: {error}"
        raise RuntimeError(message)
    active = _owned_current_user_queue_rows(output, owned_job_ids=persisted, exact_field_count=2)
    return tuple(job_id for job_id in persisted if job_id in active)


def _parse_scheduler_rows(output: str, *, field_count: int) -> dict[str, list[str]]:
    """Return exact root-job scheduler rows keyed by numeric job ID."""
    rows: dict[str, list[str]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < field_count or _JOB_ID_PATTERN.fullmatch(fields[0]) is None:
            continue
        rows[fields[0]] = fields
    return rows


def _empty_scheduler_command() -> dict[str, Any]:
    """Return the evidence of a scheduler query that was not needed."""
    return {"command": [], "output": "", "error": None}


def _scheduler_evidence(job_ids: Sequence[str]) -> dict[str, Any]:
    """Query the campaign's node-job allocations and their job steps from the live queue and durable accounting.

    ``allocations`` holds the node-job rows by allocation id and ``steps`` the job-step accounting rows by
    ``(allocation id, step name)``. ``active`` and ``accounted`` stay empty here: they are the per-dispatch rows
    keyed by dispatch key that ``_dispatch_evidence`` derives from these rows and the dispatch records.
    """
    if not job_ids:
        return {
            "squeue": _empty_scheduler_command(),
            "sacct": _empty_scheduler_command(),
            "sacct_steps": _empty_scheduler_command(),
            "allocations": {"active": {}, "accounted": {}},
            "steps": {},
            "active": {},
            "accounted": {},
            "not_started": [],
        }
    selection = ",".join(job_ids)
    squeue_command = _current_user_squeue_command("%i|%T|%R|%N|%V|%S|%M")
    sacct_command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        selection,
        "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition",
    ]
    steps_command = ["sacct", "--noheader", "--parsable2", "--jobs", selection, f"--format={cluster_service.STEP_ACCOUNTING_FORMAT}"]
    squeue_output, squeue_error = _scheduler_output(squeue_command)
    sacct_output, sacct_error = _scheduler_output(sacct_command)
    steps_output, steps_error = _scheduler_output(steps_command)
    steps: dict[tuple[str, str], dict[str, str]] = {}
    if steps_error is None:
        try:
            steps = cluster_service.parse_step_accounting(steps_output)
        except ValueError as error:
            steps_error = str(error)
    return {
        "squeue": {"command": squeue_command, "output": squeue_output, "error": squeue_error},
        "sacct": {"command": sacct_command, "output": sacct_output, "error": sacct_error},
        "sacct_steps": {"command": steps_command, "output": steps_output, "error": steps_error},
        "allocations": {
            "active": {} if squeue_error is not None else _owned_current_user_queue_rows(squeue_output, owned_job_ids=job_ids, exact_field_count=7),
            "accounted": _parse_scheduler_rows(sacct_output, field_count=3),
        },
        "steps": steps,
        "active": {},
        "accounted": {},
        "not_started": [],
    }


def _require_scheduler_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed unless the live queue, allocation accounting, and step accounting queries all succeeded."""
    for owner in ("squeue", "sacct", "sacct_steps"):
        error = evidence[owner]["error"]
        if error is not None:
            message = f"Cannot reconcile campaign jobs because {owner} failed: {error}"
            raise RuntimeError(message)


def _require_slurm_scheduler(campaign: config_service.CampaignConfig) -> None:
    """Require the Slurm scheduler for node-job submission."""
    if campaign.execution_values["cluster"]["scheduler_kind"] != "slurm":
        message = "Campaign node-job submission requires scheduler_kind='slurm'."
        raise ValueError(message)


def _persist_manifest(manifest: Mapping[str, Any], *, storage_root: Path | str | None) -> None:
    """Persist the feeder state atomically; the single feeder writer holds the submission lock."""
    common.serialization.atomic_write_json(
        campaign_evidence.campaign_run_manifest_path(str(manifest["campaign_run_id"]), storage_root=storage_root),
        manifest,
    )


def _node_job_directory(manifest: Mapping[str, Any], record: Mapping[str, Any], *, storage_root: Path | str | None) -> Path:
    """Return the absolute node directory of one node-job record."""
    directory = campaign_evidence.campaign_node_job_directory(str(manifest["campaign_run_id"]), str(record["token"]), storage_root=storage_root)
    return directory if directory.is_absolute() else directory.resolve()


def _node_job_liveness(record: Mapping[str, Any], scheduler: Mapping[str, Any], directory: Path) -> str:
    """Classify one submitted node job as ``alive`` (queued or running), ``terminal`` (ended), or ``unknown`` (not yet judged).

    A worker exit record, a persisted terminal observation, or a terminal accounting state ends the node job; a
    missing queue row alone does not, because accounting can lag the queue. The persisted observation keeps the end
    monotonic after accounting retention has dropped the allocation.
    """
    if record.get("terminal_observed_at") is not None or node_service.read_node_record(directory, "exit") is not None:
        return "terminal"
    job_id = str(record["job_id"])
    if job_id in scheduler["allocations"]["active"]:
        return "alive"
    accounted_state = _scheduler_field(scheduler["allocations"]["accounted"].get(job_id), 1)
    if accounted_state is not None and _scheduler_state(accounted_state) in _TERMINAL_ACCOUNTING_STATES:
        return "terminal"
    return "unknown"


def _open_node_jobs(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> list[dict[str, Any]]:
    """Return the submitted node jobs that are not known to have ended, in record order."""
    return [
        record
        for record in manifest["node_jobs"]
        if record["status"] == "submitted"
        and _node_job_liveness(record, scheduler, _node_job_directory(manifest, record, storage_root=storage_root)) != "terminal"
    ]


def _observe_terminal_node_jobs(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> None:
    """Record once when the feeder first sees each submitted node job ended."""
    changed = False
    for record in manifest["node_jobs"]:
        if record["status"] != "submitted" or record["terminal_observed_at"] is not None:
            continue
        if _node_job_liveness(record, scheduler, _node_job_directory(manifest, record, storage_root=storage_root)) == "terminal":
            record["terminal_observed_at"] = _utc_now()
            changed = True
    if changed:
        _persist_manifest(manifest, storage_root=storage_root)


def _stop_unleased_node_job(manifest: dict[str, Any], record: dict[str, Any], *, storage_root: Path | str | None) -> str | None:
    """Stop a node job whose source tree lease failed: cancel marker, then scancel; return a scancel failure text, if any."""
    node_service.publish_node_cancel(_node_job_directory(manifest, record, storage_root=storage_root))
    record["cancel_published"] = True
    _persist_manifest(manifest, storage_root=storage_root)
    result = subprocess.run(  # noqa: S603 -- persisted numeric Slurm job id only
        ["scancel", str(record["job_id"])],  # noqa: S607 -- site PATH owns Slurm
        check=False,
        capture_output=True,
        text=True,
    )
    return None if result.returncode == 0 else (result.stderr.strip() or f"scancel exit {result.returncode}")


def _lease_node_job(manifest: dict[str, Any], record: dict[str, Any], *, storage_root: Path | str | None) -> None:
    """Lease the executing tree for one submitted node job, with a persisted result.

    A lease that fails every attempt leaves a node job that no tree pruning protection covers: it is cancelled
    (marker, then scancel), kept tracked until it ends, and the feed stops with an explicit error.
    """
    job_id = str(record["job_id"])
    for attempt in range(1, _NODE_LEASE_ATTEMPTS + 1):
        if cluster_service.lease_executing_tree(job_id):
            record["lease"] = "ok"
            _persist_manifest(manifest, storage_root=storage_root)
            return
        if attempt < _NODE_LEASE_ATTEMPTS:
            sleep(_NODE_LEASE_RETRY_SECONDS)
    record["lease"] = "failed"
    record["error"] = f"source tree lease failed {_NODE_LEASE_ATTEMPTS} times"
    _persist_manifest(manifest, storage_root=storage_root)
    scancel_error = _stop_unleased_node_job(manifest, record, storage_root=storage_root)
    suffix = f" and scancel failed: {scancel_error}" if scancel_error is not None else ""
    message = f"Node job {job_id} was cancelled because its source tree could not be leased{suffix}."
    raise RuntimeError(message)


def _lease_pending_node_jobs(manifest: dict[str, Any], *, storage_root: Path | str | None) -> None:
    """Finish pending leases and replay the stop of every failed-lease node job until its end was observed.

    A crash after persisting ``lease: failed``, or a failed scancel, must not leave an unleased allocation open: the
    marker and the scancel are repeated on every feed (both are idempotent) until ``terminal_observed_at`` is set.
    """
    for record in manifest["node_jobs"]:
        if record["status"] != "submitted":
            continue
        if record["lease"] == "pending":
            _lease_node_job(manifest, record, storage_root=storage_root)
        elif record["lease"] == "failed" and record["terminal_observed_at"] is None:
            _stop_unleased_node_job(manifest, record, storage_root=storage_root)


def _set_ledger_state(manifest: dict[str, Any], state: str) -> None:
    """Set the ledger state of a submission or recovery step; a persisted cancellation always wins over it."""
    if manifest["state"] not in _CANCEL_STATES:
        manifest["state"] = state


def _recover_node_job_intent(manifest: dict[str, Any], *, storage_root: Path | str | None) -> bool:
    """Resolve an interrupted node-job submission; return whether the intent is clear and the feed may go on.

    The queue row of the seat whose comment is the submission token, else the worker's ``node.json``, names the job.
    Only a failed query, or less than the grace period without any trace, leaves the intent unresolved; a submission
    with no queue row, no worker record, and a full grace period behind it is proven absent.
    """
    token = manifest["node_job_intent"]
    if token is None:
        return True
    record = next(item for item in manifest["node_jobs"] if item["token"] == token)
    try:
        rows = cluster_service.query_seat_jobs([int(record["seat"])])
    except RuntimeError as error:
        print(f"Node-job submission {token} stays unresolved: {error}", file=sys.stderr)
        return False
    matches = [job_id for job_id, (_seat, _state, comment) in rows.items() if comment == token]
    if len(matches) > 1:
        message = f"Node-job recovery for token {token} is ambiguous across jobs {sorted(matches)}."
        raise RuntimeError(message)
    job_id = matches[0] if matches else None
    if job_id is None:
        worker_record = node_service.read_node_record(_node_job_directory(manifest, record, storage_root=storage_root), "node")
        job_id = None if worker_record is None else str(worker_record["job_id"])
    if job_id is None:
        recorded = _parse_utc_timestamp(str(record["recorded_at"]), label="Node-job recorded_at")
        if (datetime.now(UTC) - recorded).total_seconds() < _NODE_INTENT_GRACE_SECONDS:
            return False
        record["status"] = "submission_failed"
        record["error"] = "no queued job and no worker record carry this submission token"
    else:
        if _JOB_ID_PATTERN.fullmatch(job_id) is None or job_id in manifest["slurm_job_ids"]:
            message = f"Node-job recovery for token {token} found an unusable job id: {job_id!r}."
            raise RuntimeError(message)
        record["status"] = "submitted"
        record["job_id"] = job_id
        manifest["slurm_job_ids"].append(job_id)
    manifest["node_job_intent"] = None
    _set_ledger_state(manifest, "active")
    _persist_manifest(manifest, storage_root=storage_root)
    return True


def _fail_node_job_submission(manifest: dict[str, Any], record: dict[str, Any], error: str, *, storage_root: Path | str | None) -> None:
    """Persist a node-job submission that certainly created no job."""
    record["status"] = "submission_failed"
    record["error"] = error or "node-job submission failed"
    manifest["node_job_intent"] = None
    _set_ledger_state(manifest, "submission_failed")
    _persist_manifest(manifest, storage_root=storage_root)


def _submit_node_job(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    *,
    seat: int,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist the node-job intent, submit the node job on one seat, record its job id, and lease its tree.

    Order: intent, node directory, sbatch, submitted record with the job id, tree lease with a persisted result.
    """
    _require_slurm_scheduler(campaign)
    cluster_service.executing_tree_id()
    binding = _execution_binding(manifest)
    _validate_binding_source(binding)
    config = _active_submission_config(manifest)
    site = campaign.execution_values["site"]
    cores_per_node = int(site["cores_per_node"])
    exclusions = cluster_service.standard_node_exclusions(str(config["partition"]), cores_per_node, match="at_least")
    token = secrets.token_hex(8)
    record: dict[str, Any] = {
        "token": token,
        "seat": seat,
        "node_directory": campaign_evidence.node_job_relative_directory(token),
        "job_id": None,
        "status": "submitting",
        "lease": "pending",
        "draining": False,
        "stop_ticket_seq": None,
        "next_ticket_seq": 1,
        "cancel_published": False,
        "recorded_at": _utc_now(),
        "terminal_observed_at": None,
        "error": None,
    }
    manifest["node_jobs"].append(record)
    manifest["node_job_intent"] = token
    _set_ledger_state(manifest, "submitting")
    source_service.validate_admitted_source_before_publication()
    _persist_manifest(manifest, storage_root=storage_root)
    directory = _node_job_directory(manifest, record, storage_root=storage_root)
    try:
        for subdirectory in (node_service.TICKETS_DIRECTORY, node_service.RECEIPTS_DIRECTORY, node_service.CONTROL_DIRECTORY):
            (directory / subdirectory).mkdir(parents=True, exist_ok=True)
        command = cluster_service.build_node_job_submission_command(
            seat=seat,
            submission_token=token,
            node_directory=directory,
            storage_root=Path(manifest["remote_storage_root"]),
            cores_per_node=cores_per_node,
            node_job_memory=str(config["node_job_memory"]),
            node_job_wall_time=str(config["node_job_wall_time"]),
            partition=str(config["partition"]),
            exclusions=exclusions,
            comsol_module=str(site["comsol_module"]),
            python_executable=str(site["python_executable"]),
            comsol_executable=str(site["comsol_executable"]),
            scheduler_options=list(config["scheduler_options"]),
        )
    except (OSError, ValueError) as error:
        _fail_node_job_submission(manifest, record, str(error), storage_root=storage_root)
        raise
    try:
        job_id = cluster_service.submit_node_job(
            command,
            git_commit=str(binding["git_commit"]),
            environment={dispatch_identity.CAMPAIGN_RUN_ID_ENVIRONMENT: str(manifest["campaign_run_id"])},
        )
    except subprocess.CalledProcessError as error:
        _fail_node_job_submission(manifest, record, error.stderr.strip() or str(error), storage_root=storage_root)
        raise
    record["status"] = "submitted"
    record["job_id"] = job_id
    manifest["slurm_job_ids"].append(job_id)
    manifest["node_job_intent"] = None
    _set_ledger_state(manifest, "active")
    _persist_manifest(manifest, storage_root=storage_root)
    _lease_node_job(manifest, record, storage_root=storage_root)
    return record


def _free_seat(config: Mapping[str, Any]) -> int | None:
    """Return the lowest node-job slot of the campaign that no queued or running own node job holds, or None."""
    held = {seat for seat, _state, _comment in cluster_service.query_seat_jobs(range(1, int(config["node_jobs"]) + 1)).values()}
    return next((seat for seat in range(1, int(config["node_jobs"]) + 1) if seat not in held), None)


def _case_ticket(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    record: Mapping[str, Any],
) -> node_service.NodeTicket:
    """Return the step ticket that runs one dispatched case; the persisted dispatch record determines it completely."""
    index = int(record["submission_index"])
    config = _execution_binding(manifest, submission_index=index)["submission_config"]
    time_limit, required_seconds = cluster_service.case_step_limits(float(campaign.execution_values["runtime"]["timeout_seconds"]))
    tree = common.paths.get_project_root().resolve()
    case = record["case"]
    cores = int(config["cores_per_case"])
    return node_service.NodeTicket(
        seq=int(record["ticket_seq"]),
        kind=node_service.STEP_KIND,
        group=0,
        cores=cores,
        memory=str(config["memory_per_case"]),
        time_limit=time_limit,
        required_seconds=required_seconds,
        label=str(record["step_name"]),
        argv=(
            str(tree / "scripts" / node_service.LAUNCHER_NAME),
            str(tree),
            "campaign-case",
            str(manifest["campaign_run_id"]),
            str(case["batch_name"]),
            str(case["case_index"]),
            str(cores),
        ),
        env={
            dispatch_identity.CAMPAIGN_RUN_ID_ENVIRONMENT: str(manifest["campaign_run_id"]),
            dispatch_identity.SUBMISSION_INDEX_ENVIRONMENT: str(index),
            dispatch_identity.DISPATCH_KEY_ENVIRONMENT: dispatch_identity.dispatch_key(str(record["job_id"]), index),
            _ATTEMPT_INDEX_ENVIRONMENT: str(record["attempt_index"]),
        },
    )


def _publish_dispatch(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    record: dict[str, Any],
    node_job: dict[str, Any],
    *,
    storage_root: Path | str | None,
) -> None:
    """Publish the ticket of a dispatching record exactly once and finalize the record, the cursor, and the intent."""
    node_service.ensure_node_ticket(_node_job_directory(manifest, node_job, storage_root=storage_root), _case_ticket(manifest, campaign, record))
    record["status"] = "dispatched"
    node_job["next_ticket_seq"] = max(int(node_job["next_ticket_seq"]), int(record["ticket_seq"]) + 1)
    manifest["submission_intent"] = None
    _set_ledger_state(manifest, "active")
    _persist_manifest(manifest, storage_root=storage_root)


def _dispatch_case(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    node_job: dict[str, Any],
    *,
    mode: str,
    attempt_index: int,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist one dispatch intent, publish its ticket in the node job, and finalize the record."""
    if mode not in {"initial", "resume", "license_retry", "restart"}:
        message = f"Unsupported campaign dispatch mode: {mode!r}."
        raise ValueError(message)
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        message = "Campaign dispatch attempt index must be a positive integer."
        raise ValueError(message)
    index = len(manifest["submissions"]) + 1
    _validate_binding_source(_execution_binding(manifest))
    record = {
        "submission_index": index,
        "mode": mode,
        "recorded_at": _utc_now(),
        "case": _task_payload(task),
        "node_job_token": node_job["token"],
        "job_id": node_job["job_id"],
        "ticket_seq": int(node_job["next_ticket_seq"]),
        "step_name": _step_name(campaign, task, submission_index=index),
        "attempt_index": attempt_index,
        "status": "dispatching",
        "error": None,
    }
    manifest["submissions"].append(record)
    manifest["submission_intent"] = index
    _set_ledger_state(manifest, "submitting")
    source_service.validate_admitted_source_before_publication()
    _persist_manifest(manifest, storage_root=storage_root)
    _publish_dispatch(manifest, campaign, record, node_job, storage_root=storage_root)
    return manifest


def _recover_dispatch_intent(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> None:
    """Finish an interrupted dispatch: publish the exact ticket again, or mark the dispatch not started.

    A node job that ended without the ticket can no longer run it, so the case starts again without an attempt.
    """
    intent = manifest["submission_intent"]
    if intent is None:
        return
    record = manifest["submissions"][int(intent) - 1]
    node_job = next(item for item in manifest["node_jobs"] if item["token"] == record["node_job_token"])
    directory = _node_job_directory(manifest, node_job, storage_root=storage_root)
    ticket_published = node_service.ticket_path(directory, int(record["ticket_seq"])).is_file()
    if _node_job_liveness(node_job, scheduler, directory) == "terminal" and not ticket_published:
        record["status"] = "not_started"
        record["error"] = "the node job ended before the ticket was published"
        manifest["submission_intent"] = None
        _set_ledger_state(manifest, "active")
        _persist_manifest(manifest, storage_root=storage_root)
        return
    _publish_dispatch(manifest, campaign, record, node_job, storage_root=storage_root)


def _publish_stop_ticket(manifest: dict[str, Any], node_job: dict[str, Any], *, storage_root: Path | str | None) -> None:
    """Publish the stop ticket of a draining node job at its cursor and advance the cursor past it."""
    sequence = int(node_job["next_ticket_seq"])
    node_service.ensure_node_ticket(
        _node_job_directory(manifest, node_job, storage_root=storage_root),
        node_service.NodeTicket(seq=sequence, kind=node_service.STOP_KIND),
    )
    node_job["stop_ticket_seq"] = sequence
    node_job["next_ticket_seq"] = sequence + 1
    _persist_manifest(manifest, storage_root=storage_root)


def _drain_node_job(manifest: dict[str, Any], node_job: dict[str, Any], *, storage_root: Path | str | None) -> None:
    """Mark a node job draining, then publish its stop ticket: it takes no more dispatches and ends after its running steps."""
    node_job["draining"] = True
    _persist_manifest(manifest, storage_root=storage_root)
    _publish_stop_ticket(manifest, node_job, storage_root=storage_root)


def _finish_interrupted_drains(manifest: dict[str, Any], *, storage_root: Path | str | None) -> None:
    """Publish the stop ticket of every node job that was marked draining before a feeder crash."""
    for node_job in manifest["node_jobs"]:
        if node_job["status"] == "submitted" and node_job["draining"] and node_job["stop_ticket_seq"] is None:
            _publish_stop_ticket(manifest, node_job, storage_root=storage_root)


def _maintain_node_jobs(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> None:
    """Bring the node jobs of one feed cycle to a known state: ended jobs, pending leases, drains, and interrupted dispatches."""
    _observe_terminal_node_jobs(manifest, scheduler, storage_root=storage_root)
    _lease_pending_node_jobs(manifest, storage_root=storage_root)
    _finish_interrupted_drains(manifest, storage_root=storage_root)
    _recover_dispatch_intent(manifest, campaign, scheduler, storage_root=storage_root)


def _record_dispatch_key(record: Mapping[str, Any]) -> str:
    """Return the dispatch key ``<allocation>-<submission_index>`` of one dispatch record."""
    return dispatch_identity.dispatch_key(str(record["job_id"]), int(record["submission_index"]))


def _age_seconds(timestamp: object, *, now: datetime, label: str) -> float:
    """Return the seconds between one persisted timestamp and ``now``."""
    return (now - _parse_utc_timestamp(str(timestamp), label=label)).total_seconds()


def _format_queue_age(seconds: float) -> str:
    """Return one ``H:MM:SS`` waiting time in the layout of the squeue elapsed field."""
    whole = max(int(seconds), 0)
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _dispatch_left_execution_evidence(manifest: Mapping[str, Any], record: Mapping[str, Any], *, storage_root: Path | str | None) -> bool:
    """Return whether the case process of a dispatch demonstrably ran: its progress or interruption receipt exists."""
    execution = campaign_evidence.campaign_execution_directory(str(manifest["campaign_run_id"]), storage_root=storage_root)
    key = _record_dispatch_key(record)
    return (common.paths.resolve_generation_slurm_directory(execution) / f"{key}.json").exists() or (
        common.paths.resolve_generation_interruptions_directory(execution) / f"{key}.json"
    ).exists()


def _unstarted_reason(
    node_job: Mapping[str, Any],
    liveness: str,
    record: Mapping[str, Any],
    directory: Path,
    *,
    now: datetime,
) -> str | None:
    """Return why a dispatch without step accounting provably never started, or None while that is not proven.

    Proof is a node job that ended at least the grace period ago, or a finished receipt of the step older than the
    grace period: step accounting lags the receipt by seconds, not minutes.
    """
    observed = node_job["terminal_observed_at"]
    if (
        liveness == "terminal"
        and observed is not None
        and _age_seconds(observed, now=now, label="Node-job terminal_observed_at") >= _NOT_STARTED_GRACE_SECONDS
    ):
        return "the node job ended before the step started"
    finished = node_service.read_node_receipt(directory, int(record["ticket_seq"]), "finished")
    if finished is not None and _age_seconds(finished["ended_at"], now=now, label="Step receipt ended_at") >= _NOT_STARTED_GRACE_SECONDS:
        return "the step never started: its finished receipt has no step accounting"
    return None


def _dispatch_scheduler_rows(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    now: datetime,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[tuple[dict[str, Any], str]]]:
    """Map every dispatched record to a synthetic scheduler row in the layout of today's allocation rows.

    A terminal step row becomes an accounted row with its own state, exit code, times, and node; a running step row
    an active RUNNING row. Without a step row the dispatch is not started when that is proven, else waiting
    (PENDING) in an allocation that is alive, else unknown (no row).
    """
    node_jobs = {str(job["token"]): job for job in manifest["node_jobs"]}
    liveness: dict[str, str] = {}
    active: dict[str, list[str]] = {}
    accounted: dict[str, list[str]] = {}
    not_started: list[tuple[dict[str, Any], str]] = []
    for record in manifest["submissions"]:
        if record["status"] != "dispatched":
            continue
        key = _record_dispatch_key(record)
        step = evidence["steps"].get((str(record["job_id"]), str(record["step_name"])))
        if step is not None:
            if step["state"] in _TERMINAL_ACCOUNTING_STATES:
                accounted[key] = [
                    key,
                    step["state"],
                    step["exit_code"],
                    step["submit"],
                    step["start"],
                    step["end"],
                    step["elapsed"],
                    step["node"],
                    step["cpus"],
                    step["partition"],
                ]
            else:
                active[key] = [key, "RUNNING", "None", step["node"], step["submit"], step["start"], step["elapsed"]]
            continue
        token = str(record["node_job_token"])
        node_job = node_jobs[token]
        directory = _node_job_directory(manifest, node_job, storage_root=storage_root)
        state = liveness.setdefault(token, _node_job_liveness(node_job, evidence, directory))
        reason = _unstarted_reason(node_job, state, record, directory, now=now)
        if reason is not None and _dispatch_left_execution_evidence(manifest, record, storage_root=storage_root):
            # The case ran but its step accounting is missing (lag or retention): never call it not started; once the
            # proof window has passed it is an interrupted execution that consumed its attempt.
            accounted[key] = [key, "NODE_FAIL", "", str(record["recorded_at"]), "", "", "", "", "", ""]
        elif reason is not None:
            not_started.append((record, reason))
        elif state == "alive":
            waiting = _format_queue_age(_age_seconds(record["recorded_at"], now=now, label="Dispatch recorded_at"))
            active[key] = [key, _ACTIVE_PENDING_STATE, "NodeJob", "", str(record["recorded_at"]), "N/A", waiting]
    return active, accounted, not_started


def _dispatch_evidence(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return the scheduler evidence extended with the per-dispatch rows and the dispatches proven never started."""
    active, accounted, not_started = _dispatch_scheduler_rows(manifest, evidence, storage_root=storage_root, now=datetime.now(UTC))
    return {**evidence, "active": active, "accounted": accounted, "not_started": not_started}


def _persist_not_started_dispatches(
    manifest: dict[str, Any],
    evidence: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist the dispatches proven never started so their cases restart without an attempt; return the refreshed evidence."""
    if not evidence["not_started"]:
        return dict(evidence)
    for record, reason in evidence["not_started"]:
        record["status"] = "not_started"
        record["error"] = reason
    _persist_manifest(manifest, storage_root=storage_root)
    return _dispatch_evidence(manifest, evidence, storage_root=storage_root)


def _manifest_with_not_started(manifest: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only view of the manifest in which the dispatches proven never started already carry that status."""
    if not evidence["not_started"]:
        return manifest
    reasons = {int(record["submission_index"]): reason for record, reason in evidence["not_started"]}
    submissions = [
        {**record, "status": "not_started", "error": reasons[int(record["submission_index"])]}
        if int(record["submission_index"]) in reasons
        else record
        for record in manifest["submissions"]
    ]
    return {**manifest, "submissions": submissions}


def _unresolved_dispatch_counts(
    manifest: Mapping[str, Any],
    open_jobs: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, int]:
    """Count per open node job the dispatches without a terminal step state; a dispatch the scheduler does not answer for counts."""
    counts = {str(job["token"]): 0 for job in open_jobs}
    for record in manifest["submissions"]:
        token = str(record["node_job_token"])
        if token in counts and record["status"] in {"dispatching", "dispatched"} and _record_dispatch_key(record) not in evidence["accounted"]:
            counts[token] += 1
    return counts


def _dispatch_target(
    scheduler: Mapping[str, Any],
    open_jobs: Sequence[dict[str, Any]],
    counts: Mapping[str, int],
    *,
    capacity: int,
    submitted_now: Collection[str],
) -> dict[str, Any] | None:
    """Return the queued or running, leased, not draining node job with the most free ticket capacity, if any has some.

    A node job submitted during this feed cycle is not in the cycle's queue read yet and counts as queued.
    """
    best: dict[str, Any] | None = None
    best_free = 0
    for job in open_jobs:
        row = scheduler["allocations"]["active"].get(str(job["job_id"]))
        alive = (row is not None and _scheduler_state(row[1]) in _ALIVE_NODE_JOB_STATES) or str(job["token"]) in submitted_now
        if job["draining"] or job["lease"] != "ok" or not alive:
            continue
        free = capacity - counts[str(job["token"])]
        if free > best_free:
            best, best_free = job, free
    return best


def _drain_idle_node_jobs(
    manifest: dict[str, Any],
    open_jobs: Sequence[dict[str, Any]],
    counts: Mapping[str, int],
    *,
    settled: bool,
    storage_root: Path | str | None,
) -> None:
    """Drain open node jobs that have nothing left to run: every one when the campaign is settled, else the idle ones."""
    for job in open_jobs:
        if not job["draining"] and job["lease"] == "ok" and (settled or counts[str(job["token"])] == 0):
            _drain_node_job(manifest, job, storage_root=storage_root)


def _settle_campaign(
    manifest: dict[str, Any],
    state: str,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Drain every open node job of a settled campaign and persist its final state."""
    open_jobs = _open_node_jobs(manifest, scheduler, storage_root=storage_root)
    _drain_idle_node_jobs(manifest, open_jobs, {str(job["token"]): 0 for job in open_jobs}, settled=True, storage_root=storage_root)
    return _set_campaign_state(manifest, state, storage_root=storage_root)


def _submission_index(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], tuple[Mapping[str, Any], ...]]:
    """Index persisted submissions once for one campaign operation."""
    records: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in manifest["submissions"]:
        case = record["case"]
        key = str(case["batch_id"]), int(case["case_index"])
        records.setdefault(key, []).append(record)
    return {key: tuple(values) for key, values in records.items()}


def _task_submissions(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    *,
    submission_index: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return submission records for one exact case in attempt order."""
    expected = _task_payload(task)
    candidates = manifest["submissions"] if submission_index is None else submission_index.get((task.batch_id, task.case_index), ())
    return tuple(record for record in candidates if record["case"] == expected)


def _started_dispatches(submissions: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Return the dispatches whose ticket is published; an in-flight or never started dispatch owns no case state."""
    return tuple(record for record in submissions if record["status"] == "dispatched")


def _dispatch_ownership(
    dispatched: Sequence[Mapping[str, Any]],
    scheduler: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Return the dispatches of one case whose step is active, and the latest one if the scheduler does not answer for it.

    More than one active step is a conflict wherever it is; a missing answer matters only for the dispatch that owns
    the case, because a newer dispatch supersedes every older one.
    """
    active = [record for record in dispatched if _record_dispatch_key(record) in scheduler["active"]]
    latest = dispatched[-1:]
    unknown = [
        record
        for record in latest
        if _record_dispatch_key(record) not in scheduler["active"] and _record_dispatch_key(record) not in scheduler["accounted"]
    ]
    return active, unknown


def _require_unambiguous_active_task_ownership(
    task: cluster_service.CampaignTask,
    active_records: Sequence[Mapping[str, Any]],
) -> None:
    """Reject more than one active Slurm step for one exact case."""
    if len(active_records) <= 1:
        return
    keys = sorted(_record_dispatch_key(record) for record in active_records)
    message = f"Conflicting active Slurm step ownership for {task.batch_id}/{task.case_id}: {keys}"
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
    """Return named scheduler evidence for one task's latest started dispatch; ``latest_job_id`` is its dispatch key."""
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
    job_id = _record_dispatch_key(latest_submission)
    job_name_value = latest_submission.get("step_name")
    job_name = job_name_value if isinstance(job_name_value, str) else None
    active = scheduler["active"].get(job_id)
    accounted = scheduler["accounted"].get(job_id)
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
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    latest_submission: Mapping[str, Any] | None,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return best-effort progress for one task's latest started dispatch."""
    if latest_submission is None:
        return {
            "availability": "unavailable",
            "reason": "no_job",
            "age_seconds": None,
            "stale": None,
        }
    return progress_service.load_runtime_progress(
        str(manifest["campaign_run_id"]),
        dispatch_identity.dispatch_key(str(latest_submission["job_id"]), int(latest_submission["submission_index"])),
        _task_payload(task),
        campaign=campaign,
        storage_root=storage_root,
        manifest=manifest,
    )


def _admitted_case_attempt(
    manifest: Mapping[str, Any],
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    input_reference: admission_service.InputCaseReference,
    storage_root: Path | str | None,
) -> AdmittedCaseAttempt | None:
    """Return the newest attempt after exact campaign and case identity checks; it is current only for the latest started dispatch."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    current_run_id = str(manifest["campaign_run_id"])
    attempt = attempt_service.latest_case_attempt(
        batch,
        task.case_index,
        storage_root=storage,
    )
    if attempt is None:
        return None
    persisted_input_generation_id = str(attempt.payload["input_generation_id"])
    if persisted_input_generation_id != input_reference.source_id:
        input_reference = input_service.admit_persisted_input_case(
            batch,
            task.case_index,
            persisted_input_generation_id,
            storage_root=storage,
            validation_depth="evidence",
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
        attempt.payload.get("campaign_run_id"),
        label="attempt campaign_run_id",
    )
    current_run_identity_valid = (
        attempt.payload.get("campaign_run_id") == attempt_run_id
        and attempt_run_id == current_run_id
        and attempt.payload.get("solver_git_commit") == manifest["git_commit"]
    )
    started = _started_dispatches(_task_submissions(manifest, task))
    owned: list[Mapping[str, Any]] = []
    if common.paths.campaign_run_parts(current_run_id) is not None:
        owned = [
            record
            for record in started
            if record["job_id"] == attempt.payload.get("job_id") and record["submission_index"] == attempt.payload.get("submission_index")
        ]
        current_run_identity_valid = (
            attempt_run_id == current_run_id
            and len(owned) == 1
            and attempt.payload.get("solver_git_commit")
            == _execution_binding(manifest, submission_index=int(owned[0]["submission_index"]))["git_commit"]
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
    return AdmittedCaseAttempt(attempt=attempt, current=len(owned) == 1 and owned[0] is started[-1])


def _successful_status_path(
    batch: config_service.GenerationConfig,
    task: cluster_service.CampaignTask,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the already-validated processed status used for presentation."""
    return (
        common.paths.resolve_generation_result_evidence_directory(
            batch_runtime.processed_case_directory(
                batch,
                task.case_index,
                storage_root=storage_root,
            )
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
        canonical_case_path = (
            batch_runtime.processed_case_directory(batch, task.case_index, storage_root=storage_root) / common.paths.GENERATION_RESULT_HDF5_FILENAME
        )
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


def _license_retry_is_active(
    state: str,
    latest_submission: Mapping[str, Any] | None,
    in_allocation_license_window: Mapping[str, Any] | None,
) -> bool:
    """Return whether the current pre-solver job is a license retry."""
    return bool(
        latest_submission is not None
        and latest_submission.get("mode") == "license_retry"
        and state in {"pending", "active", "scheduler_unknown"}
        and not _in_allocation_window_proves_solver_started(in_allocation_license_window)
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
        parsed = datetime.fromisoformat(original.replace("Z", "+00:00"))  # noqa: FURB162 -- retain exact persisted timestamp input normalization and rejection behavior
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
            normalized_utc_value=parsed.astimezone(UTC).isoformat(),
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
        normalized_utc_value=localized.astimezone(UTC).isoformat(),
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
        common.paths.resolve_generation_result_evidence_directory(
            batch_runtime.processed_case_directory(
                batch,
                task.case_index,
                storage_root=storage_root,
            )
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
_TERMINAL_TIMESTAMP_SCHEMA_VERSION: Final = 2
_TERMINAL_TIMESTAMP_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "dispatch_key",
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
    dispatch_key: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one exact operational Slurm-End projection receipt path of a dispatch."""
    dispatch_identity.parse_dispatch_key(dispatch_key)
    execution_directory = campaign_evidence.campaign_execution_directory(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    case_directory = common.paths.resolve_generation_reconciliation_case_directory(execution_directory, task.execution_key, task.case_id)
    return case_directory / f"{dispatch_key}.terminal_timestamp.json"


def _validate_terminal_timestamp_evidence(
    payload: object,
    *,
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    dispatch_key: str,
    path: Path,
) -> dict[str, Any]:
    """Validate one exact schema-version-2 external timestamp projection of a dispatch."""
    if (
        not isinstance(payload, dict)
        or set(payload) != _TERMINAL_TIMESTAMP_KEYS
        or payload.get("schema_kind") != _TERMINAL_TIMESTAMP_SCHEMA_KIND
        or payload.get("schema_version") != _TERMINAL_TIMESTAMP_SCHEMA_VERSION
        or payload.get("campaign_run_id") != manifest["campaign_run_id"]
        or payload.get("batch_id") != task.batch_id
        or payload.get("case_id") != task.case_id
        or payload.get("dispatch_key") != dispatch_key
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
            "dispatch_key": dispatch_key,
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
    dispatch_key: str,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load one existing external timestamp projection without scheduler I/O."""
    path = _terminal_timestamp_evidence_path(
        manifest,
        task,
        dispatch_key,
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
        dispatch_key=dispatch_key,
        path=path,
    )


def _record_terminal_timestamp_evidence(
    manifest: Mapping[str, Any],
    task: cluster_service.CampaignTask,
    dispatch_key: str,
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
        dispatch_key,
        storage_root=storage_root,
    )
    payload = {
        "schema_kind": _TERMINAL_TIMESTAMP_SCHEMA_KIND,
        "schema_version": _TERMINAL_TIMESTAMP_SCHEMA_VERSION,
        "campaign_run_id": manifest["campaign_run_id"],
        "batch_id": task.batch_id,
        "case_id": task.case_id,
        "dispatch_key": dispatch_key,
        "source_field": "sacct.End",
        "source_sha256": common.serialization.canonical_json_sha256(
            {
                "dispatch_key": dispatch_key,
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
        dispatch_key=dispatch_key,
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
            dispatch_key,
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
        dispatch_key=dispatch_key,
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
    execution_directory = campaign_evidence.campaign_execution_directory(
        str(manifest["campaign_run_id"]),
        storage_root=storage_root,
    )
    dependency = _case_reconciliation_dependency_sha256()
    receipt_directory = common.paths.resolve_generation_reconciliation_failures_directory(execution_directory, task.execution_key, task.case_id)
    return receipt_directory / f"{source_sha256}.{dependency}.json"


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
    receipt_directory = common.paths.resolve_generation_reconciliation_failures_directory(
        campaign_evidence.campaign_execution_directory(str(manifest["campaign_run_id"]), storage_root=storage_root),
        task.execution_key,
        task.case_id,
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
        return "never_started", str(submissions[-1]["error"] or submissions[-1]["status"])
    if (task.batch_id, task.case_index) in _admission_reservation_keys(manifest):
        return "admission_waiting", "admission_reserved"
    return "never_started", "not_submitted"


def _task_state(  # noqa: C901, PLR0912, PLR0915 -- centralized case evidence reconciliation
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    *,
    batch: config_service.GenerationConfig,
    input_reference: admission_service.InputCaseReference,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
    cached_case_reconciliation: tuple[Mapping[str, Any], Path | None] | None = None,
    submission_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconcile one case from processed, attempt, and exact scheduler evidence."""
    submissions = tuple(submission_records) if submission_records is not None else _task_submissions(manifest, task)
    dispatched = _started_dispatches(submissions)
    latest_submission = dispatched[-1] if dispatched else None
    active_records, unknown_records = _dispatch_ownership(dispatched, scheduler)
    _require_unambiguous_active_task_ownership(task, active_records)
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
        input_reference=input_reference,
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
            (
                scheduler["accounted"][_record_dispatch_key(record)]
                for record in reversed(dispatched)
                if _record_dispatch_key(record) in scheduler["accounted"]
            ),
            None,
        )
        if active_records:
            active_state = _scheduler_state(scheduler["active"][_record_dispatch_key(active_records[-1])][1])
            state = "pending" if active_state == _ACTIVE_PENDING_STATE else "active"
            reason = active_state
        elif unknown_records:
            state = "scheduler_unknown"
            reason = _record_dispatch_key(unknown_records[-1])
        else:
            if latest_submission is not None:
                latest_dispatch_key = dispatch_identity.dispatch_key(
                    str(latest_submission["job_id"]),
                    int(latest_submission["submission_index"]),
                )
                retry_attempt = license_service.latest_wait_for_dispatch(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    dispatch_key=latest_dispatch_key,
                    storage_root=storage_root,
                )
                allocation_window = license_service.load_in_allocation_license_window(
                    batch,
                    task.case_index,
                    campaign_run_id=str(manifest["campaign_run_id"]),
                    dispatch_key=latest_dispatch_key,
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
                input_reference=input_reference,
                storage_root=storage_root,
            )
            # An attempt of an older dispatch never overrides the terminal step state of the case's latest started dispatch.
            attempt = admitted_attempt.attempt if admitted_attempt is not None and admitted_attempt.current else None
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
            elif admitted_attempt is None and batch_runtime.case_failure_is_recorded(
                batch,
                task.case_index,
                storage_root=storage_root,
                execution_run_id=str(manifest["campaign_run_id"]),
                git_commit=str(_execution_binding(manifest, submission_index=int(latest_submission["submission_index"]))["git_commit"])
                if latest_submission is not None
                else str(manifest["git_commit"]),
            ):
                state = "failed"
                reason = "case_failure_evidence"
                failure_stage = "solver"
                pipeline["solver_state"] = "failed"
            elif latest_accounted is not None:
                state, reason = _scheduler_terminal_case_state(_scheduler_state(latest_accounted[1]))
                if state != "interrupted":
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
        campaign,
        task,
        latest_submission=latest_submission,
        storage_root=storage_root,
    )
    active_submission = active_records[-1] if active_records else None
    if active_submission is not None:
        allocation_window = license_service.load_in_allocation_license_window(
            batch,
            task.case_index,
            campaign_run_id=str(manifest["campaign_run_id"]),
            dispatch_key=dispatch_identity.dispatch_key(
                str(active_submission["job_id"]),
                int(active_submission["submission_index"]),
            ),
            storage_root=storage_root,
        )
    license_retry_active = _license_retry_is_active(
        state,
        active_submission if state == "active" else latest_submission,
        allocation_window,
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
    terminal_dispatch_key = scheduler_view["latest_job_id"]
    if state == "successful" and isinstance(terminal_dispatch_key, str):
        if compact_license_attempt_payloads and scheduler_terminal_timestamp.original_value is not None:
            recorded_path = _record_terminal_timestamp_evidence(
                manifest,
                task,
                terminal_dispatch_key,
                scheduler_terminal_timestamp,
                storage_root=storage_root,
            )
            terminal_timestamp_evidence_path = None if recorded_path is None else str(recorded_path)
        existing_timestamp = _load_terminal_timestamp_evidence(
            manifest,
            task,
            terminal_dispatch_key,
            storage_root=storage_root,
        )
        if existing_timestamp is not None:
            if (
                scheduler_terminal_timestamp.original_value is not None
                and existing_timestamp["original_value"] != scheduler_terminal_timestamp.original_value
            ):
                message = f"Terminal timestamp source changed for one completed dispatch: {terminal_dispatch_key}"
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
                    terminal_dispatch_key,
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
            dispatch_key=scheduler_view["latest_job_id"],
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
            license_retry_eligible
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
    task_views: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Publish ready batch manifests from one reconciliation or one bounded fallback."""
    all_complete = True
    for batch in campaign.batches:
        expected = {(batch.batch_id, case_index) for case_index in batch.case_indices}
        if task_views is None:
            references = input_service.admit_configured_input_references(
                batch,
                storage_root=storage_root,
                validation_depth="evidence",
            )
            complete = all(
                batch_runtime.completed_case_is_valid(
                    batch,
                    case_index,
                    storage_root=storage_root,
                    input_reference=references[case_index],
                )
                for case_index in batch.case_indices
            )
        else:
            matching = {(str(view["batch_id"]), int(view["case_index"])): view for view in task_views if view.get("batch_id") == batch.batch_id}
            complete = set(matching) == expected and all(view["state"] == "successful" for view in matching.values())
        if complete:
            batch_runtime.finalize_batch(batch, storage_root=storage_root)
        else:
            all_complete = False
    return all_complete


def _case_reconciliation_failed_view(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    scheduler: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_path: Path | None,
    *,
    submission_records: Sequence[Mapping[str, Any]] | None = None,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Return one isolated lifecycle view while unrelated cases continue."""
    batch = campaign.batch(task.batch_name)
    submissions = tuple(submission_records) if submission_records is not None else _task_submissions(manifest, task)
    dispatched = _started_dispatches(submissions)
    latest_submission = dispatched[-1] if dispatched else None
    active_records, unknown_records = _dispatch_ownership(dispatched, scheduler)
    _require_unambiguous_active_task_ownership(task, active_records)
    if active_records:
        active_state = _scheduler_state(scheduler["active"][_record_dispatch_key(active_records[-1])][1])
        state = "pending" if active_state == _ACTIVE_PENDING_STATE else "active"
    elif unknown_records:
        state = "scheduler_unknown"
    else:
        state = str(evidence["resulting_state"])
    scheduler_view = _task_scheduler_view(latest_submission, scheduler)
    runtime_progress = _task_runtime_progress_view(
        manifest,
        campaign,
        task,
        latest_submission=latest_submission,
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


def _campaign_input_references(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None,
    input_sources: Mapping[str, Mapping[str, Any]],
    admitted: Mapping[str, Mapping[int, admission_service.InputCaseReference]] | None = None,
) -> dict[str, dict[int, admission_service.InputCaseReference]]:
    """Admit each immutable input batch once for one campaign operation, reusing the references already ``admitted``."""
    canonical = campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION
    expected_batches = {
        batch.batch_name for batch in campaign.batches if not (canonical and isinstance(batch.scientific_values.get("supplemental_design"), dict))
    }
    if set(input_sources) != expected_batches:
        message = "Campaign input-source batch membership disagrees with the active campaign."
        raise RuntimeError(message)
    references: dict[str, dict[int, admission_service.InputCaseReference]] = {}
    for batch in campaign.batches:
        if admitted is not None and batch.batch_name in admitted:
            references[batch.batch_name] = dict(admitted[batch.batch_name])
            continue
        if canonical and batch.batch_name not in input_sources:
            references[batch.batch_name] = input_service.admit_configured_input_references(
                batch,
                storage_root=storage_root,
                validation_depth="evidence",
                git_commit=git_commit,
            )
            continue
        selected_source = input_sources[batch.batch_name]
        case_indices = selected_source.get("case_indices")
        if not isinstance(case_indices, list) or tuple(case_indices) != batch.case_indices:
            message = f"Campaign input-source case membership disagrees for batch {batch.batch_name!r}."
            raise RuntimeError(message)
        references[batch.batch_name] = input_service.admit_configured_input_references(
            batch,
            storage_root=storage_root,
            validation_depth="evidence",
            git_commit=git_commit,
            input_generation_id=str(selected_source["input_generation_id"]),
            input_source_git_commit=str(selected_source["source_git_commit"]),
        )
    return references


def _persisted_input_sources(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the complete persisted input-source map."""
    raw_batches = manifest.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        message = "Campaign input-source ownership requires a non-empty batch list."
        raise RuntimeError(message)
    source_fields = ("input_generation_id", "input_source_git_commit", "input_case_indices")
    batches: list[Mapping[str, Any]] = []
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, Mapping):
            message = "Campaign input-source ownership contains a malformed batch record."
            raise TypeError(message)
        batches.append(raw_batch)
    if any(any(batch.get(field) is None for field in source_fields) for batch in batches):
        message = "Campaign input-source ownership is incomplete or corrupt."
        raise RuntimeError(message)

    selected: dict[str, dict[str, Any]] = {}
    for batch in batches:
        batch_name = common.paths.validate_logical_name(batch.get("batch_name"), label="campaign input-source batch_name")
        input_generation_id = common.paths.validate_logical_name(
            batch["input_generation_id"],
            label="campaign input_generation_id",
        )
        if re.fullmatch(r"input-[0-9a-f]{24}", input_generation_id) is None:
            message = "Campaign input_generation_id is malformed."
            raise RuntimeError(message)
        source_git_commit = source_service.validate_git_commit(batch["input_source_git_commit"])
        case_indices = batch["input_case_indices"]
        if (
            not isinstance(case_indices, list)
            or not case_indices
            or any(type(case_index) is not int or case_index < 1 for case_index in case_indices)
            or len(case_indices) != len(set(case_indices))
        ):
            message = "Campaign input-source case membership is malformed."
            raise RuntimeError(message)
        if batch_name in selected:
            message = "Campaign input-source batch ownership is duplicated."
            raise RuntimeError(message)
        selected[batch_name] = {
            "input_generation_id": input_generation_id,
            "source_git_commit": source_git_commit,
            "case_indices": list(case_indices),
        }
    return selected


def _reconciled(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    compact_license_attempt_payloads: bool = False,
    input_references: Mapping[str, Mapping[int, admission_service.InputCaseReference]] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    tasks: Sequence[cluster_service.CampaignTask] | None = None,
    submission_index: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Reconcile every case while isolating only typed case-local defects."""
    progress_started_at = monotonic() if progress is not None else None
    last_progress_at: float | None = None
    task_views: list[dict[str, Any]] = []
    batches = {batch.batch_name: batch for batch in campaign.batches}
    campaign_tasks = tuple(tasks) if tasks is not None else cluster_service.campaign_tasks(campaign)
    submissions_by_task = _submission_index(manifest) if submission_index is None else submission_index
    for task in campaign_tasks:
        dispatched = _started_dispatches(submissions_by_task.get((task.batch_id, task.case_index), ()))
        _require_unambiguous_active_task_ownership(task, _dispatch_ownership(dispatched, scheduler)[0])
    references = _campaign_input_references(
        campaign,
        git_commit=str(manifest["git_commit"]),
        storage_root=storage_root,
        input_sources=_persisted_input_sources(manifest),
        admitted=input_references,
    )
    states = [_scheduler_state(fields[1]) for fields in scheduler["active"].values()]
    pending_jobs = sum(state == _ACTIVE_PENDING_STATE for state in states)
    running_jobs = len(states) - pending_jobs
    terminal_jobs = len(scheduler["accounted"])

    def report_progress() -> None:
        nonlocal last_progress_at
        if progress is None or progress_started_at is None:
            return
        now = monotonic()
        if now - progress_started_at < _RECONCILIATION_PROGRESS_DELAY_SECONDS:
            return
        if last_progress_at is not None and now - last_progress_at < _RECONCILIATION_PROGRESS_HEARTBEAT_SECONDS:
            return
        last_progress_at = now
        progress(
            {
                "operation": "reconciling campaign metadata and scheduler state",
                "cases_completed": len(task_views),
                "cases_total": len(campaign_tasks),
                "jobs_running": running_jobs,
                "jobs_pending": pending_jobs,
                "jobs_terminal": terminal_jobs,
                "heartbeat": "active",
                "eta": "unavailable",
            }
        )

    for task in campaign_tasks:
        batch = batches[task.batch_name]
        input_reference = references[task.batch_name][task.case_index]
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
                submission_records=submissions_by_task.get((task.batch_id, task.case_index), ()),
                storage_root=storage_root,
            )
            task_views.append(view)
            report_progress()
            continue
        try:
            view = _task_state(
                manifest,
                campaign,
                task,
                scheduler,
                batch=batch,
                input_reference=input_reference,
                storage_root=storage_root,
                compact_license_attempt_payloads=(compact_license_attempt_payloads),
                cached_case_reconciliation=cached,
                submission_records=submissions_by_task.get((task.batch_id, task.case_index), ()),
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
                    batch=batch,
                    input_reference=input_reference,
                    storage_root=storage_root,
                    compact_license_attempt_payloads=(compact_license_attempt_payloads),
                    cached_case_reconciliation=(evidence, evidence_path),
                    submission_records=submissions_by_task.get((task.batch_id, task.case_index), ()),
                )
            else:
                view = _case_reconciliation_failed_view(
                    manifest,
                    campaign,
                    task,
                    scheduler,
                    evidence,
                    evidence_path,
                    submission_records=submissions_by_task.get((task.batch_id, task.case_index), ()),
                    storage_root=storage_root,
                )
        task_views.append(view)
        report_progress()
    reservation_keys = _admission_reservation_keys(manifest)
    waiting_keys = {_task_identity_key(view) for view in task_views if view["state"] == "admission_waiting"}
    if waiting_keys != reservation_keys:
        message = "Campaign admission reservations conflict with reconciled case evidence."
        raise RuntimeError(message)
    return task_views, pending_jobs, running_jobs


def _unexpected_operational_failure_views(
    campaign: config_service.CampaignConfig,
    candidate_records: Sequence[Mapping[str, Any]],
    task_views: Sequence[Mapping[str, Any]],
    scheduler: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Find latest dispatches with an unresolved worker-side operational error."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- canonical execution-batch owner

    views = {(str(view["batch_id"]), str(view["case_id"])): view for view in task_views}
    failures: list[Mapping[str, Any]] = []
    for row in candidate_records:
        attempts = row["attempts"]
        if row["decision"] is not None or not attempts:
            continue
        attempt = attempts[-1]
        if attempt["class"] not in {"controller", "publication"} or attempt["retryable"] is not True:
            continue
        reservation = row["reservation"]
        view = views.get((lifecycle_service.execution_batch(campaign, reservation).batch_id, str(reservation["case_id"])))
        if view is None:
            msg = "Operational attempt has no campaign task view."
            raise RuntimeError(msg)
        evidence = attempt["evidence"]
        key = evidence.get("dispatch_key") if isinstance(evidence, Mapping) else None
        if not isinstance(key, str) or dispatch_identity.DISPATCH_KEY_PATTERN.fullmatch(key) is None:
            msg = "Operational attempt lacks its exact Slurm dispatch identity."
            raise RuntimeError(msg)
        if view["latest_job_id"] != key or view["state"] == "successful":
            continue
        if key in scheduler["active"] or key in scheduler["accounted"]:
            failures.append(view)
    return tuple(failures)


def _compute_required_continuation(manifest: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    """Return an explicit heavy-work request without persisting a new lifecycle state."""
    return {**manifest, "continuation_requires_compute": True, "continuation_reason": reason}


def _prepare_supplemental_inputs(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None,
    metadata_only: bool,
) -> tuple[bool, str | None]:
    """Admit prepared replacements or defer their spatial input generation."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- canonical feeder boundary
    from src.generation.cases import generation_cases_selection as selection_service  # noqa: PLC0415 -- replacement deficit owner

    for replacement in (batch for batch in campaign.batches if isinstance(batch.scientific_values.get("supplemental_design"), dict)):
        if metadata_only:
            try:
                input_service.admit_configured_input_references(
                    replacement,
                    storage_root=storage_root,
                    validation_depth="evidence",
                    git_commit=str(manifest["git_commit"]),
                )
            except FileNotFoundError:
                return False, "supplemental_inputs_missing"
        else:
            input_service.generate_input_cases(replacement, len(replacement.case_indices), storage_root=storage_root)
    originals = tuple(batch for batch in campaign.batches if not isinstance(batch.scientific_values.get("supplemental_design"), dict))
    waiting_for_capacity = False
    for original in originals:
        while True:
            records = [
                row
                for row in lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
                if row["reservation"]["batch_id"] == original.batch_id
            ]
            primaries = [row for row in records if row["reservation"]["origin"] == "primary"]
            if len(primaries) != len(original.case_indices) or any(row["decision"] is None for row in primaries):
                break
            deficit = selection_service.replacement_deficit(records, target=len(original.case_indices))
            if deficit < 1:
                break
            used = sum(row["reservation"]["origin"] == "replacement" for row in records)
            high_water = lifecycle_service.replacement_pool_high_water(
                campaign.profile.id, campaign.source_name, original.batch_id, storage_root=storage_root
            )
            if used >= high_water:
                waiting_for_capacity = True
                break
            if metadata_only:
                return False, "replacement_input_generation"
            replacement = lifecycle_service.materialize_replacement_batch(campaign, original.batch_id, storage_root=storage_root)
            input_service.generate_input_cases(replacement, len(replacement.case_indices), storage_root=storage_root)
    return waiting_for_capacity, None


def _reconcile_success_decisions(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    task_views: Sequence[Mapping[str, Any]],
    candidate_records: tuple[dict[str, Any], ...],
    scheduler: Mapping[str, Any],
    *,
    metadata_only: bool,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], ...] | None:
    """Record the success decision of every validly published case whose step ended without writing it; return the current records.

    A step that fails after its valid publication (for example on scratch cleanup) never reaches its own decision. The
    case view is ``successful`` by the same admission as the status; a decided case is never touched, and a case whose
    latest step has not ended stays with that step, which still decides it. A decision hashes the case HDF5, so a
    metadata-only feed writes none: it returns ``None`` at the first case that needs one.
    """
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- canonical execution-batch owner

    undecided = [row for row in candidate_records if row["decision"] is None]
    if not undecided:
        return candidate_records
    views = {(str(view["batch_id"]), str(view["case_id"])): view for view in task_views}
    recorded = False
    for row in undecided:
        reservation = row["reservation"]
        view = views.get((lifecycle_service.execution_batch(campaign, reservation).batch_id, str(reservation["case_id"])))
        if view is None or view["state"] != "successful":
            continue
        if view["latest_job_id"] is not None and view["latest_job_id"] not in scheduler["accounted"]:
            continue
        if metadata_only:
            return None
        task = cluster_service.require_campaign_task(campaign, batch_name=str(view["batch_name"]), case_index=int(view["case_index"]))
        dispatched = _started_dispatches(_task_submissions(manifest, task))
        latest = dispatched[-1] if dispatched else None
        resources = None
        if latest is not None:
            binding = _execution_binding(manifest, submission_index=int(latest["submission_index"]))
            resources = {
                "dispatch_key": _record_dispatch_key(latest),
                "slurm_job_id": str(latest["job_id"]),
                "slurm_step_id": None,
                "submission_index": int(latest["submission_index"]),
                "allocated_cores": int(binding["submission_config"]["cores_per_case"]),
            }
        _record_success_decision(campaign, task, resources=resources, storage_root=storage_root)
        recorded = True
    if not recorded:
        return candidate_records
    return lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)


def _reconcile_runtime_failure_decisions(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    candidate_records: tuple[dict[str, Any], ...],
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> bool:
    """Record the failed decision of every undecided candidate whose solve the runtime stopped itself; return whether any was recorded.

    Only a non-retryable solver attempt with a runtime reason (stalled, timed out, solver error) qualifies, and only when
    it belongs to the case's latest started dispatch, that dispatch has terminal step accounting, and no dispatch of the
    case is active. Anything else, an unclassified solver failure included, waits for an explicit decision.
    """
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 terminal decision owner

    if manifest["submission_intent"] is not None or manifest["node_job_intent"] is not None:
        return False
    recorded = False
    for row in candidate_records:
        attempt = row["attempts"][-1] if row["attempts"] else None
        if row["decision"] is not None or attempt is None or attempt["reason_code"] not in lifecycle_service.RUNTIME_SOLVER_REASONS:
            continue
        dispatches = _case_dispatches(manifest, lifecycle_service.execution_batch(campaign, row["reservation"]), str(row["reservation"]["case_id"]))
        key = attempt["evidence"].get("dispatch_key")
        if not dispatches or _record_dispatch_key(dispatches[-1]) != key or key not in scheduler["accounted"]:
            continue
        if attempt["class"] == "solver" and _failure_decision_blocker(manifest, campaign, row, scheduler) is None:
            _record_failure_decision(
                manifest, campaign, row, reason_code=str(attempt["reason_code"]), detail=str(attempt["detail"]), storage_root=storage_root
            )
            recorded = True
    return recorded


def feed_campaign(  # noqa: PLR0911 -- the sole feeder returns once per settled, deferred, or dispatched outcome
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    verified_repository_commit: str | None = None,
    input_references: Mapping[str, Mapping[int, admission_service.InputCaseReference]] | None = None,
    resolved_campaign: config_service.CampaignConfig | None = None,
    explicit_submission: bool = False,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Reconcile jobs; metadata-only continuation defers input generation and batch admission.

    A compute-required return is transient command evidence, never a persisted
    campaign state. The scheduled caller re-enters this same sole feeder lock.
    """
    lock_path = common.paths.resolve_generation_submission_lock_path(campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root))
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        manifest = campaign_evidence.load_campaign_run(
            run_id,
            storage_root=storage_root,
        )
        if manifest["state"] in _CANCEL_STATES:
            _apply_cancellation(manifest, resolved_campaign, force=False, explicit=False, storage_root=storage_root)
            return manifest
        campaign = campaign_evidence.campaign_from_manifest(manifest) if resolved_campaign is None else resolved_campaign
        blocked = False
        if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
            from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 append-only feeder boundary

            waiting_for_capacity, compute_reason = _prepare_supplemental_inputs(
                manifest,
                campaign,
                storage_root=storage_root,
                metadata_only=metadata_only,
            )
            if compute_reason is not None:
                return _compute_required_continuation(manifest, reason=compute_reason)
            campaign = lifecycle_service.campaign_from_frozen_run(manifest)
            binding = _execution_binding(manifest)
            if "execution_values" in binding:
                campaign = lifecycle_service.campaign_with_execution(campaign, binding["execution_values"])
            blocked = waiting_for_capacity and not any(
                row["decision"] is None
                for row in lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
            )
        tasks = cluster_service.campaign_tasks(campaign)
        if not _recover_node_job_intent(manifest, storage_root=storage_root):
            return manifest
        scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
        _require_scheduler_evidence(scheduler)
        _maintain_node_jobs(manifest, campaign, scheduler, storage_root=storage_root)
        scheduler = _persist_not_started_dispatches(
            manifest,
            _dispatch_evidence(manifest, scheduler, storage_root=storage_root),
            storage_root=storage_root,
        )
        if blocked:
            # Every candidate is decided and no replacement remains: drain the open node jobs as a completion does.
            return _settle_campaign(manifest, "blocked", scheduler, storage_root=storage_root)
        submission_index = _submission_index(manifest)
        task_views, pending_jobs, running_jobs = _reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=storage_root,
            compact_license_attempt_payloads=True,
            input_references=input_references,
            progress=progress,
            tasks=tasks,
            submission_index=submission_index,
        )
        successful = [view for view in task_views if view["state"] == "successful"]
        operational_retries: tuple[Mapping[str, Any], ...] = ()
        if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
            reconciled = _reconcile_success_decisions(
                manifest,
                campaign,
                task_views,
                lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root),
                scheduler,
                metadata_only=metadata_only,
                storage_root=storage_root,
            )
            if reconciled is None:
                return _compute_required_continuation(manifest, reason="success_decision_reconciliation")
            # A new failure decision frees its slot. The next feed plans the replacement (or ends blocked) before it can
            # dispatch, so this feed neither drains an idle node job nor settles.
            if _reconcile_runtime_failure_decisions(manifest, campaign, reconciled, scheduler, storage_root=storage_root):
                return manifest
            candidate_records = reconciled
            originals = tuple(batch for batch in campaign.batches if not isinstance(batch.scientific_values.get("supplemental_design"), dict))
            complete = all(row["decision"] is not None for row in candidate_records) and all(
                sum(row["reservation"]["origin"] == "primary" and row["reservation"]["batch_id"] == batch.batch_id for row in candidate_records)
                == len(batch.case_indices)
                and sum(
                    row["decision"] is not None and row["decision"]["outcome"] == "successful" and row["reservation"]["batch_id"] == batch.batch_id
                    for row in candidate_records
                )
                >= len(batch.case_indices)
                for batch in originals
            )
            if complete:
                from .publication import generation_publication_catalog as catalog_service  # noqa: PLC0415 -- derived view of the settled campaign

                lifecycle_service.seal_selected_pool(campaign.profile.id, campaign.source_name, storage_root=storage_root)
                common.serialization.atomic_write_text(
                    common.paths.resolve_generation_campaign_directory(campaign.profile.id, campaign.source_name, storage_root=storage_root)
                    / common.paths.GENERATION_CASES_VIEW_FILENAME,
                    catalog_service.campaign_cases_csv(campaign.profile.id, campaign.source_name, storage_root=storage_root),
                )
                return _settle_campaign(manifest, "complete", scheduler, storage_root=storage_root)
        if campaign.schema_version != config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION and len(successful) == len(task_views):
            if metadata_only:
                return _compute_required_continuation(manifest, reason="batch_artifact_admission")
            _finalize_completed_batches(
                campaign,
                storage_root=storage_root,
                task_views=task_views,
            )
            return _settle_campaign(manifest, "complete", scheduler, storage_root=storage_root)
        if any(view["state"] == "scheduler_unknown" for view in task_views):
            return _set_campaign_state(manifest, "scheduler_unknown", storage_root=storage_root)
        if any(view["state"] == "license_blocked" and view["license_wait_exhausted"] is True for view in task_views):
            return _settle_campaign(manifest, "license_wait_exhausted", scheduler, storage_root=storage_root)
        if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
            operational_retries = _unexpected_operational_failure_views(campaign, candidate_records, task_views, scheduler)
            if operational_retries and not explicit_submission:
                first = operational_retries[0]
                message = (
                    "Campaign stopped automatic feeding after an unexpected controller or publication error "
                    f"for {first['case_id']} in Slurm step dispatch {first['latest_job_id']}. "
                    "Inspect the retained attempt, then explicitly rerun the same campaign config to retry."
                )
                raise RuntimeError(message)
            if explicit_submission and any(str(view["latest_job_id"]) not in scheduler["accounted"] for view in operational_retries):
                msg = "An explicit operational retry requires terminal step accounting for its previous dispatch."
                raise RuntimeError(msg)

        return _fill_submission_capacity(
            manifest,
            campaign,
            task_views,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            scheduler=scheduler,
            storage_root=storage_root,
            progress=progress,
            verified_repository_commit=verified_repository_commit,
            tasks=tasks,
            submission_index=submission_index,
            operational_retries=operational_retries,
        )


def _in_allocation_window_proves_solver_started(
    in_allocation_license_window: Mapping[str, Any] | None,
) -> bool:
    """Return whether one validated current-job receipt proves solver startup."""
    return bool(
        in_allocation_license_window is not None
        and in_allocation_license_window.get("solver_progress_started") is True
        and in_allocation_license_window.get("outcome") == "solver_progress_started"
        and in_allocation_license_window.get("reason") == "solver_progress_started"
    )


def _task_identity_key(view: Mapping[str, Any]) -> tuple[str, int]:
    """Return one logical campaign-case identity from a reconciled view."""
    return str(view["batch_id"]), int(view["case_index"])


def _view_consumes_admission(view: Mapping[str, Any]) -> bool:
    """Return whether one logical case currently occupies admission."""
    state = view.get("state")
    if state == "active":
        return not _in_allocation_window_proves_solver_started(view.get("in_allocation_license_window"))
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
    if mode not in {"initial", "resume", "license_retry", "restart"}:
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


def _fresh_admission_case_keys(manifest: Mapping[str, Any]) -> frozenset[tuple[str, int]]:
    """Count exact fresh candidates ever admitted by this campaign run."""
    keys = {
        (str(record["case"]["batch_id"]), int(record["case"]["case_index"]))
        for record in manifest.get("submissions", ())
        if record.get("mode") == "initial"
    }
    keys.update(_admission_reservation_keys(manifest))
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
    maximum = int(_active_submission_config(manifest)["max_admission_cases"])
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
) -> tuple[bool, str | None]:
    """Return whether eligible fresh work is prevented from admission."""
    restart_available = any(view["state"] in {"cancelled", "interrupted"} for view in task_views)
    fresh_available = any(view["state"] == "never_started" for view in task_views)
    if not restart_available and not fresh_available:
        return False, None
    submission_config = _active_submission_config(manifest)
    if manifest.get("submission_intent") is not None or manifest.get("node_job_intent") is not None:
        return True, "submission_intent_unresolved"
    if sum(view["state"] in {"pending", "active"} for view in task_views) >= _maximum_dispatch_capacity(submission_config):
        return True, "node_capacity_reached"
    admission_count = len(_logical_admission_case_keys(manifest, task_views))
    maximum_admission_cases = int(submission_config["max_admission_cases"])
    if admission_count > maximum_admission_cases or len(_fresh_admission_case_keys(manifest)) > maximum_admission_cases:
        return True, "max_admission_cases_exceeded"
    if fresh_available and len(_fresh_admission_case_keys(manifest)) >= maximum_admission_cases:
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
    return timestamp.astimezone(UTC)


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


def _claim_restart_admission(
    key: tuple[str, int],
    admission_keys: set[tuple[str, int]],
    operational_keys: set[tuple[str, int]],
    *,
    maximum: int,
    readmitted_keys: Collection[tuple[str, int]] = (),
) -> bool:
    """Keep an explicit operational restart, or a dispatch that never started, inside the frozen admission limit."""
    if key in admission_keys:
        return True
    if key not in operational_keys and key not in readmitted_keys:
        message = "Retry or resume case no longer owns logical admission."
        raise RuntimeError(message)
    if len(admission_keys) >= maximum:
        return False
    admission_keys.add(key)
    return True


def _release_initial_launch_reservation(
    manifest: dict[str, Any],
    *,
    key: tuple[str, int],
) -> None:
    """Release one reservation as its durable submission intent takes over."""
    manifest["admission_reservations"] = [
        record for record in manifest.get("admission_reservations", []) if (str(record["batch_id"]), int(record["case_index"])) != key
    ]


def _set_campaign_state(
    manifest: dict[str, Any],
    state: str,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Persist one campaign state only when its value changes."""
    if manifest["state"] == state:
        return manifest
    manifest["state"] = state
    common.serialization.atomic_write_json(
        campaign_evidence.campaign_run_manifest_path(
            str(manifest["campaign_run_id"]),
            storage_root=storage_root,
        ),
        manifest,
    )
    return manifest


def _verify_admission_source(manifest: Mapping[str, Any], repository_verified: bool) -> bool:
    """Verify the mutable checkout against the admitted execution source once per operation before any scheduler action."""
    if not repository_verified:
        current_commit = _repository_commit()
        binding = _execution_binding(manifest)
        _validate_binding_source(binding)
        expected_commit = binding["git_commit"]
        if current_commit != expected_commit:
            message = f"Solver admission requires admitted execution source commit {expected_commit}, got {current_commit}."
            raise RuntimeError(message)
    return True


def _submitted_attempt_counts(
    manifest: Mapping[str, Any],
    *,
    submission_index: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]] | None = None,
) -> dict[tuple[str, int], int]:
    """Count the dispatches that really started once per logical case; a dispatch that never started consumes no attempt."""
    records_by_task = _submission_index(manifest) if submission_index is None else submission_index
    return {key: len(_started_dispatches(records)) for key, records in records_by_task.items()}


def _preverified_repository_commit(
    manifest: Mapping[str, Any],
    candidate: str | None,
) -> bool:
    """Validate one operation-local source check inherited by submission."""
    if candidate is None:
        return False
    if source_service.validate_git_commit(candidate) != _execution_binding(manifest)["git_commit"]:
        message = "Prevalidated repository commit disagrees with the admitted execution source."
        raise RuntimeError(message)
    return True


def _settled_admission_state(
    campaign: config_service.CampaignConfig,
    task_views: Sequence[Mapping[str, Any]],
    *,
    fresh_admission_cases: int,
    max_admission_cases: int,
) -> str:
    """Distinguish a successful capped subset from unresolved case failures."""
    staged = (
        campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION
        and fresh_admission_cases >= max_admission_cases
        and any(view["state"] == "successful" for view in task_views)
        and any(view["state"] == "never_started" for view in task_views)
        and all(view["state"] in {"successful", "never_started"} for view in task_views)
    )
    return "staged" if staged else "completed_with_failures"


def _dispatch_mode(
    view: Mapping[str, Any],
    *,
    operational_keys: Collection[tuple[str, int]],
    not_started_keys: Collection[tuple[str, int]],
) -> str:
    """Return the dispatch mode of one candidate case view."""
    key = _task_identity_key(view)
    if key in operational_keys or view["state"] in {"cancelled", "interrupted"}:
        return "resume"
    if key in not_started_keys:
        return "restart"
    return "license_retry" if view["state"] == "license_blocked" else "initial"


def _fill_submission_capacity(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig,
    task_views: Sequence[Mapping[str, Any]],
    *,
    pending_jobs: int,
    running_jobs: int,
    scheduler: Mapping[str, Any],
    storage_root: Path | str | None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    verified_repository_commit: str | None = None,
    tasks: Sequence[cluster_service.CampaignTask] | None = None,
    submission_index: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]] | None = None,
    operational_retries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Dispatch owned retries and admit fresh cases within one logical pool, submitting a node job on a free seat when none has room."""
    effective_submission = _active_submission_config(manifest)
    max_admission_cases = int(effective_submission["max_admission_cases"])
    node_job_limit = int(effective_submission["node_jobs"])
    ticket_capacity = _node_ticket_capacity(effective_submission)
    maximum_failed_cases = int(effective_submission["maximum_failed_cases"])
    repository_verified = _preverified_repository_commit(
        manifest,
        verified_repository_commit,
    )
    admission_keys = set(_logical_admission_case_keys(manifest, task_views))
    fresh_admission_keys = set(_fresh_admission_case_keys(manifest))
    campaign_tasks = tuple(tasks) if tasks is not None else cluster_service.campaign_tasks(campaign)
    task_index = {(task.batch_id, task.case_index): task for task in campaign_tasks}
    plan_order = {(task.batch_id, task.case_index): index for index, task in enumerate(campaign_tasks)}
    submissions_by_task = _submission_index(manifest) if submission_index is None else submission_index
    submitted_attempt_counts = _submitted_attempt_counts(manifest, submission_index=submissions_by_task)
    not_started_keys = frozenset(key for key, records in submissions_by_task.items() if records[-1]["status"] == "not_started")
    open_jobs = _open_node_jobs(manifest, scheduler, storage_root=storage_root)
    unresolved = _unresolved_dispatch_counts(manifest, open_jobs, scheduler)
    submitted_now: set[str] = set()

    def report_submission_progress() -> None:
        if progress is None:
            return
        progress(
            {
                "operation": "scheduler_submission",
                "work_units_admitted": len(admission_keys),
                "work_units_total": len(campaign_tasks),
                "scheduler_submissions": len(manifest.get("submissions", ())),
                "eta": "unavailable",
            }
        )

    reservation_keys = _admission_reservation_keys(manifest)
    operational_keys = {_task_identity_key(view) for view in operational_retries}
    retries = deque(
        sorted(
            (
                view
                for view in task_views
                if view["state"] == "license_blocked" and view["license_retry_eligible"] is True and _task_identity_key(view) not in operational_keys
            ),
            key=lambda view: _retry_candidate_sort_key(
                view,
                plan_order=plan_order,
            ),
        )
    )
    restarts = deque(
        [
            *operational_retries,
            *(view for view in task_views if view["state"] in {"cancelled", "interrupted"} and _task_identity_key(view) not in operational_keys),
            *(view for view in task_views if view["state"] == "never_started" and _task_identity_key(view) in not_started_keys),
        ]
    )
    reserved = deque(
        view for view in task_views if view["state"] in {"admission_waiting", "never_started"} and _task_identity_key(view) in reservation_keys
    )
    unsent = deque(
        view
        for view in task_views
        if view["state"] == "never_started" and _task_identity_key(view) not in reservation_keys and _task_identity_key(view) not in not_started_keys
    )
    circuit_breaker_tripped = _solver_failure_threshold_exceeded(
        task_views,
        maximum_failed_cases=maximum_failed_cases,
    )
    has_license_blocked = any(view["state"] == "license_blocked" for view in task_views)
    if len(admission_keys) > max_admission_cases or len(fresh_admission_keys) > max_admission_cases:
        next_state = "active" if pending_jobs or running_jobs else "license_blocked" if has_license_blocked else "active"
        return _set_campaign_state(manifest, next_state, storage_root=storage_root)

    while True:
        next_unsent = unsent[0] if unsent else None
        fresh_allowed = next_unsent is not None and not circuit_breaker_tripped and len(fresh_admission_keys) < max_admission_cases
        if not (retries or restarts or reserved or fresh_allowed):
            break
        target = _dispatch_target(scheduler, open_jobs, unresolved, capacity=ticket_capacity, submitted_now=submitted_now)
        if target is None:
            seat = _free_seat(effective_submission) if len(open_jobs) < node_job_limit else None
            if seat is None:
                return _set_campaign_state(manifest, "active", storage_root=storage_root)
            repository_verified = _verify_admission_source(manifest, repository_verified)
            target = _submit_node_job(manifest, campaign, seat=seat, storage_root=storage_root)
            open_jobs.append(target)
            unresolved[str(target["token"])] = 0
            submitted_now.add(str(target["token"]))
            continue
        next_view = (retries or restarts or reserved or unsent).popleft()
        task = _task_from_payload(
            campaign,
            next_view,
            task_index=task_index,
        )
        key = (task.batch_id, task.case_index)
        mode = _dispatch_mode(next_view, operational_keys=operational_keys, not_started_keys=not_started_keys)
        if mode == "initial":
            _reserve_initial_launch(
                manifest,
                task,
                admission_keys,
                reservation_keys,
                maximum=max_admission_cases,
            )
            fresh_admission_keys.add(key)
        elif not _claim_restart_admission(
            key,
            admission_keys,
            operational_keys,
            maximum=max_admission_cases,
            readmitted_keys=not_started_keys,
        ):
            return _set_campaign_state(manifest, "active", storage_root=storage_root)

        if mode == "initial":
            _release_initial_launch_reservation(manifest, key=key)
        repository_verified = _verify_admission_source(manifest, repository_verified)
        manifest = _dispatch_case(
            manifest,
            campaign,
            task,
            target,
            mode=mode,
            attempt_index=submitted_attempt_counts.get(key, 0) + 1,
            storage_root=storage_root,
        )
        unresolved[str(target["token"])] += 1
        pending_jobs += 1
        submitted_attempt_counts[key] = submitted_attempt_counts.get(key, 0) + 1
        report_submission_progress()

    if pending_jobs or running_jobs or (admission_keys and not has_license_blocked):
        next_state = "active"
    elif has_license_blocked:
        next_state = "license_blocked"
    elif circuit_breaker_tripped and next_unsent is not None:
        next_state = "failure_threshold_reached"
    else:
        next_state = _settled_admission_state(
            campaign,
            task_views,
            fresh_admission_cases=len(fresh_admission_keys),
            max_admission_cases=max_admission_cases,
        )
    _drain_idle_node_jobs(manifest, open_jobs, unresolved, settled=next_state in _SETTLED_CAMPAIGN_STATES, storage_root=storage_root)
    return _set_campaign_state(manifest, next_state, storage_root=storage_root)


def _require_terminal_campaign_jobs(manifest: Mapping[str, Any], *, action: str) -> None:
    """Require every owned node job to have ended before an explicit transition."""
    if manifest["submission_intent"] is not None or manifest["node_job_intent"] is not None:
        message = f"Cannot {action} while a submission intent is unresolved."
        raise RuntimeError(message)
    scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
    _require_scheduler_evidence(scheduler)
    allocations = scheduler["allocations"]
    accounted_states = (_scheduler_field(allocations["accounted"].get(job_id), 1) for job_id in manifest["slurm_job_ids"])
    if allocations["active"] or any(state is None or _scheduler_state(state) not in _TERMINAL_ACCOUNTING_STATES for state in accounted_states):
        message = f"Cannot {action} until all previous node jobs have terminal scheduler evidence."
        raise RuntimeError(message)


def _require_prepared_metadata_submission(*, metadata_only: bool, inputs_prepared: bool) -> None:
    """Keep fresh metadata-only submissions outside scientific input preparation."""
    if metadata_only and not inputs_prepared:
        message = "Fresh metadata-only submission requires prepare-campaign-inputs followed by --inputs-prepared."
        raise ValueError(message)


def submit_campaign(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    inputs_prepared: bool = False,
    max_admission_cases: int | None = None,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Create durable campaign state and fill its safe submission capacity."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"Shared repository commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    run_id = campaign_run_id(campaign, git_commit=requested_commit)
    if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
        if campaign.total_case_count != sum(campaign.selected_targets.values()):
            message = "Canonical campaign submission requires its complete material success-target plan."
            raise ValueError(message)
        from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- versioned campaign creation boundary

        lifecycle_service.create_campaign(campaign, storage_root=storage_root)
    if max_admission_cases is not None:
        if isinstance(max_admission_cases, bool) or max_admission_cases < 1:
            msg = "max_admission_cases override must be a positive integer."
            raise ValueError(msg)
        execution = copy.deepcopy(campaign.execution_values)
        execution["submission"]["max_admission_cases"] = max_admission_cases
        campaign = replace(campaign, execution_values=execution)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root)
    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
    existing = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root) if path.exists() else None
    prepared_input_references: dict[str, dict[int, admission_service.InputCaseReference]] | None = None
    if existing is not None:
        input_sources: Mapping[str, Mapping[str, Any]] = _persisted_input_sources(existing)
    else:
        _require_prepared_metadata_submission(metadata_only=metadata_only, inputs_prepared=inputs_prepared)
        if inputs_prepared:
            prepared_inputs, prepared_input_references = input_service.admit_campaign_inputs_with_references(
                campaign,
                git_commit=requested_commit,
                storage_root=storage_root,
            )
        else:
            prepared_inputs = input_service.prepare_campaign_inputs(
                campaign,
                git_commit=requested_commit,
                storage_root=storage_root,
                progress=progress,
                existing_validation_depth="evidence",
            )
        input_sources = {
            batch.batch_name: {
                **item,
                "source_git_commit": str(item.get("source_git_commit", requested_commit)),
                "case_indices": list(item.get("case_indices", batch.case_indices)),
            }
            for batch, item in zip(campaign.batches, prepared_inputs["batches"], strict=True)
        }
    input_references = (
        prepared_input_references
        if prepared_input_references is not None
        else _campaign_input_references(
            campaign,
            git_commit=requested_commit if existing is None else str(existing["git_commit"]),
            storage_root=storage_root,
            input_sources=input_sources,
        )
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    scheduler_log_directory = common.paths.get_runtime_root().resolve() / "logs" / "generation"
    scheduler_log_directory.mkdir(parents=True, exist_ok=True)
    lock_path = common.paths.resolve_generation_submission_lock_path(run_directory)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        intent = _new_campaign_manifest(
            campaign,
            run_id=run_id,
            requested_commit=requested_commit if existing is None else str(existing["git_commit"]),
            run_directory=run_directory,
            scheduler_log_directory=scheduler_log_directory,
            storage_root=storage_root,
            input_sources=input_sources,
        )
        if path.exists():
            existing = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
            immutable_keys = set(intent).difference(
                {
                    "dataset_packages",
                    "scheduler_log_directory",
                    "slurm_job_ids",
                    "node_jobs",
                    "node_job_intent",
                    "submissions",
                    "submission_intent",
                    "admission_reservations",
                    "state",
                    "execution_config_digest",
                    "submission_config",
                }
            )
            if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
                immutable_keys.discard("source_snapshot")  # Origin stays frozen; new sources belong to execution epochs.
                immutable_keys.discard("campaign_config")  # Authored plan location is recovery evidence, independent of science.
            if any(existing[key] != intent[key] for key in immutable_keys):
                message = f"Existing campaign-run manifest conflicts with {run_id!r}."
                raise FileExistsError(message)
            cancelled_continuation = existing["state"] in {"cancel_requested", "force_cancel_requested"}
            if cancelled_continuation:
                _require_terminal_campaign_jobs(existing, action="continue a cancelled campaign")
            if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
                requested = _requested_execution_binding(campaign, requested_commit)
                previous = _execution_binding(existing)
                from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- exact v2 continuation evidence

                frozen = lifecycle_service.load_campaign_plan(campaign.profile.id, campaign.source_name, storage_root=storage_root)
                _assert_execution_resource_compatibility(frozen["execution_values"], campaign.execution_values)
                if int(requested["submission_config"]["max_admission_cases"]) < int(previous["submission_config"]["max_admission_cases"]):
                    msg = "Campaign admission capacity cannot decrease below its latest recorded execution epoch."
                    raise ValueError(msg)
                source_changed = any(
                    requested[key] != previous[key] for key in ("git_commit", "source_sha256", "source_snapshot", "source_snapshot_sha256")
                )
                resources_changed = requested["execution_config_digest"] != previous["execution_config_digest"]
                if (source_changed or resources_changed) and existing["state"] != "complete":
                    _require_explicit_source_transition(run_id, source_changed=source_changed)
                    if not cancelled_continuation:
                        _require_terminal_campaign_jobs(existing, action="change execution binding")
                    source_service.validate_admitted_source_before_publication()
                    lifecycle_service.append_execution_epoch(existing, requested)
            if cancelled_continuation:
                source_service.validate_admitted_source_before_publication()
                existing["state"] = "active"
                common.serialization.atomic_write_json(path, existing)
        else:
            source_service.validate_admitted_source_before_publication()
            common.serialization.atomic_write_json(path, intent)
            if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
                lifecycle_service.append_execution_epoch(intent, _requested_execution_binding(campaign, requested_commit))
    return feed_campaign(
        run_id,
        storage_root=storage_root,
        progress=progress,
        verified_repository_commit=requested_commit,
        input_references=input_references,
        resolved_campaign=campaign,
        explicit_submission=True,
        metadata_only=metadata_only,
    )


def resume_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    status_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Continue one canonical frozen campaign through its sole scheduler feeder."""
    if common.paths.campaign_run_parts(run_id) is None:
        message = "Campaign resume requires a canonical source campaign execution handle."
        raise ValueError(message)
    result = feed_campaign(run_id, storage_root=storage_root, progress=progress, metadata_only=metadata_only)
    if status_callback is not None:
        status_callback(campaign_status(run_id, storage_root=storage_root))
    return result


def declare_candidate_failure(
    run_id: str,
    material: str,
    case_id: str,
    *,
    reason_code: str,
    detail: str,
    storage_root: Path | str | None = None,
) -> Path:
    """Explicitly terminalize one failed solver/validation candidate after exact admission."""
    if common.paths.campaign_run_parts(run_id) is None:
        message = "Explicit candidate failure decisions require a canonical v2 campaign."
        raise ValueError(message)
    if re.fullmatch(r"[a-z][a-z0-9_]*", reason_code) is None or not detail.strip():
        message = "Terminal candidate failure requires a stable reason code and non-empty detail."
        raise ValueError(message)
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    if manifest["submission_intent"] is not None or manifest["node_job_intent"] is not None or manifest["state"] == "complete":
        message = "A candidate decision cannot be declared during uncertain submission or after campaign completion."
        raise RuntimeError(message)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 terminal decision owner

    records = [
        row
        for row in lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
        if row["reservation"]["material_family"] == material and row["reservation"]["case_id"] == case_id
    ]
    if len(records) != 1:
        message = "Terminal candidate decision does not identify one reserved campaign-local case."
        raise ValueError(message)
    record = records[0]
    decision_path = common.paths.resolve_generation_case_decision_path(record["case_directory"])
    if record["decision"] is not None:
        decision = record["decision"]
        if decision["outcome"] == "failed" and decision["reason_code"] == reason_code and decision["detail"] == detail:
            return decision_path
        message = "Candidate already has a different immutable terminal decision."
        raise FileExistsError(message)
    scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
    _require_scheduler_evidence(scheduler)
    blocker = _failure_decision_blocker(manifest, campaign, record, _dispatch_evidence(manifest, scheduler, storage_root=storage_root))
    if blocker is not None:
        raise RuntimeError(blocker)
    return _record_failure_decision(manifest, campaign, record, reason_code=reason_code, detail=detail, storage_root=storage_root)


def _case_dispatches(manifest: Mapping[str, Any], batch: config_service.GenerationConfig, case_id: str) -> list[Mapping[str, Any]]:
    """Return the started dispatches of one case of its execution batch in submission order."""
    return [
        item
        for item in _started_dispatches(manifest["submissions"])
        if item["case"]["case_id"] == case_id and item["case"]["batch_id"] == batch.batch_id
    ]


def _failure_decision_blocker(
    manifest: Mapping[str, Any], campaign: config_service.CampaignConfig, record: Mapping[str, Any], scheduler: Mapping[str, Any]
) -> str | None:
    """Return why one undecided candidate cannot take a failed decision now, or ``None`` when its exhausted attempt allows one."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 terminal decision owner

    attempts = record["attempts"]
    if not attempts or attempts[-1]["class"] not in {"solver", "validation"} or attempts[-1]["retryable"] is not False:
        return "Candidate has no exhausted solver/validation attempt eligible for an explicit terminal decision."
    # A replacement reservation names its parent batch; its own one-case batch carries scheduler and input identity.
    batch = lifecycle_service.execution_batch(campaign, record["reservation"])
    if any(_record_dispatch_key(item) in scheduler["active"] for item in _case_dispatches(manifest, batch, str(record["reservation"]["case_id"]))):
        return "Candidate still has an active Slurm step."
    return None


def _record_failure_decision(
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    record: Mapping[str, Any],
    *,
    reason_code: str,
    detail: str,
    storage_root: Path | str | None,
) -> Path:
    """Commit the failed terminal decision of one admitted candidate with its input, attempt, and provenance evidence."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 terminal decision owner

    material, case_id = str(record["reservation"]["material_family"]), str(record["reservation"]["case_id"])
    attempts = record["attempts"]
    batch = lifecycle_service.execution_batch(campaign, record["reservation"])
    matching_indices = [index for index in batch.case_indices if batch.case_id(index) == case_id]
    if len(matching_indices) != 1:
        message = "Candidate reservation has no unique case index in its frozen scientific batch."
        raise ValueError(message)
    index = matching_indices[0]
    reference = input_service.admit_configured_input_references(
        batch, storage_root=storage_root, validation_depth="evidence", git_commit=str(manifest["git_commit"])
    )[index]
    payload = reference.case_payload()
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    input_manifest = (
        common.paths.resolve_generation_input_generation_metadata_directory(batch.batch_storage_name, reference.source_id, storage_root=storage)
        / "input_generation_manifest.json"
    )
    attempt_receipts = [
        {
            "relative_path": common.paths.resolve_generation_case_attempt_record_path(record["case_directory"], number)
            .relative_to(storage)
            .as_posix(),
            "sha256": common.serialization.file_sha256(common.paths.resolve_generation_case_attempt_record_path(record["case_directory"], number)),
        }
        for number in range(1, len(attempts) + 1)
    ]
    return lifecycle_service.decide_candidate(
        campaign.profile.id,
        campaign.source_name,
        material,
        case_id,
        outcome="failed",
        stage=str(attempts[-1]["stage"]),
        failure_class=str(attempts[-1]["class"]),
        reason_code=reason_code,
        detail=detail,
        case_input_id=reference.case_input_id,
        simulation_case_id=reference.simulation_case_id,
        case_hdf5_relative=None,
        case_hdf5_sha256=None,
        evidence={
            "input_manifest_relative": input_manifest.relative_to(storage).as_posix(),
            "parameters": payload.get("sampled_values"),
            "fields": payload.get("spatial_diagnostics"),
            "schedules": payload.get("schedule_diagnostics"),
            "diagnostics": attempts[-1],
            "attempt_receipts": attempt_receipts,
            "source_provenance": {
                "git_commit": manifest["git_commit"],
                "source_snapshot": manifest.get("source_snapshot"),
            },
            "resources": attempts[-1].get("evidence"),
        },
        storage_root=storage_root,
    )


def amend_replacement_capacity(
    run_id: str,
    material: str,
    regime: str,
    *,
    capacity: int,
    reason: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Explicitly raise one original batch's total replacement capacity for a campaign run.

    Dataset plans pre-register their capacity at campaign creation; this operation
    serves campaigns without one and records its amendment as explicit.
    """
    parts = common.paths.campaign_run_parts(run_id)
    if parts is None:
        message = "Replacement capacity amendments require a canonical v2 campaign."
        raise ValueError(message)
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- v2 capacity amendment owner

    task, campaign_name = parts
    plan = lifecycle_service.load_campaign_plan(task, campaign_name, storage_root=storage_root)
    matches = [batch for batch in plan["batches"] if batch["material_family"] == material and batch["sampling_regime"] == regime]
    if len(matches) != 1:
        message = f"Campaign {campaign_name!r} has no unique original batch for material {material!r} and regime {regime!r}."
        raise ValueError(message)
    batch_id = str(matches[0]["batch_id"])
    previous = lifecycle_service.replacement_pool_high_water(task, campaign_name, batch_id, storage_root=storage_root)
    high_water = lifecycle_service.amend_replacement_pool(
        task, campaign_name, batch_id, high_water=capacity, reason=reason, storage_root=storage_root
    )
    return {
        "campaign_run_id": run_id,
        "material_family": material,
        "sampling_regime": regime,
        "batch_id": batch_id,
        "previous_capacity": previous,
        "capacity": high_water,
    }


def _load_cancellation_receipt(run_id: str, *, storage_root: Path | str | None) -> tuple[Path, list[dict[str, Any]]]:
    """Return the cancellation receipt path and its validated attempts; a missing receipt has none."""
    path = common.paths.resolve_generation_cancellations_path(campaign_evidence.campaign_execution_directory(run_id, storage_root=storage_root))
    if not path.exists():
        return path, []
    raw = campaign_evidence.load_json_object(
        path,
        label="campaign cancellation receipt",
    )
    if (
        raw.get("schema_kind") != "generation_campaign_cancellations"
        or raw.get("schema_version") != _CANCELLATION_SCHEMA_VERSION
        or raw.get("campaign_run_id") != run_id
        or not isinstance(raw.get("attempts"), list)
    ):
        message = f"Campaign cancellation receipt is malformed: {path}"
        raise ValueError(message)
    return path, raw["attempts"]


def _cancel_commands(mode: str, pending_ids: Sequence[str], running_ids: Sequence[str]) -> list[list[str]]:
    """Return the scancel argument vectors of one cancellation attempt: queued allocations end, running ones forward TERM to their steps."""
    if mode == "force":
        ids = [*pending_ids, *running_ids]
        return [["scancel", "--signal=KILL", "--full", *ids]] if ids else []
    commands: list[list[str]] = []
    if pending_ids:
        commands.append(["scancel", *pending_ids])
    if running_ids:
        commands.append(["scancel", "--signal=TERM", "--batch", *running_ids])
    return commands


def _apply_cancellation(
    manifest: dict[str, Any],
    campaign: config_service.CampaignConfig | None,
    *,
    force: bool,
    explicit: bool,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Apply the persisted cancellation to every node job; an explicit call also records the request.

    Cancellation is a desired state replayed by every later feed: unresolved intents are resolved first, then every
    node job that has not ended gets its cancel marker and, unless an earlier attempt already commanded it
    successfully in this mode, a scancel (queued allocations end, running ones forward TERM to their steps; force
    kills). The receipt is written before any command. Returns the receipt, or None when a replay had nothing to do.
    """
    run_id = str(manifest["campaign_run_id"])
    _recover_node_job_intent(manifest, storage_root=storage_root)
    scheduler = _scheduler_evidence(manifest["slurm_job_ids"])
    _require_scheduler_evidence(scheduler)
    _observe_terminal_node_jobs(manifest, scheduler, storage_root=storage_root)
    if manifest["submission_intent"] is not None:
        resolved = campaign_evidence.campaign_from_manifest(manifest) if campaign is None else campaign
        _recover_dispatch_intent(manifest, resolved, scheduler, storage_root=storage_root)
    effective_force = force or manifest["state"] == "force_cancel_requested"
    mode = "force" if effective_force else "graceful"
    receipt_path, existing = _load_cancellation_receipt(run_id, storage_root=storage_root)
    commanded = {
        job_id
        for attempt in existing
        if attempt["mode"] == mode or (mode == "graceful" and attempt["mode"] == "force")
        for job_id in attempt["succeeded_job_ids"]
    }
    allocation_rows = scheduler["allocations"]["active"]
    active_ids = [job_id for job_id in manifest["slurm_job_ids"] if job_id in allocation_rows and (explicit or job_id not in commanded)]
    pending_ids = [job_id for job_id in active_ids if _scheduler_state(allocation_rows[job_id][1]) == _ACTIVE_PENDING_STATE]
    running_ids = [job_id for job_id in active_ids if job_id not in pending_ids]
    open_jobs = _open_node_jobs(manifest, scheduler, storage_root=storage_root)
    unmarked = [job for job in open_jobs if not job["cancel_published"]]
    if not explicit and not active_ids and not unmarked:
        return None
    attempt: dict[str, Any] = {
        "recorded_at": _utc_now(),
        "mode": mode,
        "pending_job_ids": pending_ids,
        "running_job_ids": running_ids,
        "node_job_tokens": [str(job["token"]) for job in unmarked],
        "commands": [],
        "succeeded_job_ids": [],
    }
    receipt = {
        "schema_kind": "generation_campaign_cancellations",
        "schema_version": _CANCELLATION_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        "attempts": [*existing, attempt],
    }
    common.serialization.atomic_write_json(receipt_path, receipt)
    target_state = "force_cancel_requested" if effective_force else "cancel_requested"
    if manifest["state"] != target_state:
        manifest["state"] = target_state
        _persist_manifest(manifest, storage_root=storage_root)
    for job in unmarked:
        node_service.publish_node_cancel(_node_job_directory(manifest, job, storage_root=storage_root))
        job["cancel_published"] = True
    if unmarked:
        _persist_manifest(manifest, storage_root=storage_root)
    command_results: list[dict[str, Any]] = []
    failed_commands: list[str] = []
    for command in _cancel_commands(mode, pending_ids, running_ids):
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
        if result.returncode == 0:
            attempt["succeeded_job_ids"].extend(argument for argument in command[1:] if _JOB_ID_PATTERN.fullmatch(argument) is not None)
        else:
            failed_commands.append(result.stderr.strip() or f"{command[0]} exit {result.returncode}")
    attempt["commands"] = command_results
    common.serialization.atomic_write_json(receipt_path, receipt)
    if failed_commands:
        message = f"Slurm cancellation failed after its request was persisted: {failed_commands}"
        raise RuntimeError(message)
    return receipt


def cancel_campaign(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Persist the cancellation request and apply it to every node job; every later feed replays it until the node jobs end."""
    if not isinstance(force, bool):
        message = "force must be boolean."
        raise TypeError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage_root,
    )
    lock_path = common.paths.resolve_generation_submission_lock_path(run_directory)
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        manifest = campaign_evidence.load_campaign_run(
            run_id,
            storage_root=storage_root,
        )
        receipt = _apply_cancellation(manifest, None, force=force, explicit=True, storage_root=storage_root)
        if receipt is None:
            message = "An explicit cancellation always records its attempt."
            raise RuntimeError(message)
        return receipt


def campaign_accounting(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return the exact squeue, allocation sacct, and step sacct commands and their current output."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    evidence = _scheduler_evidence(manifest["slurm_job_ids"])
    return {
        "campaign_run_id": run_id,
        "squeue": evidence["squeue"],
        "sacct": evidence["sacct"],
        "sacct_steps": evidence["sacct_steps"],
    }


def record_worker_interruption(
    run_id: str,
    *,
    storage_root: Path | str | None,
    signal_name: str,
    exit_code: int,
) -> Path:
    """Persist one best-effort interruption receipt of the running case step, keyed by its dispatch."""
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    dispatch = dispatch_identity.current_case_dispatch()
    if dispatch is None:
        message = "Worker interruption evidence requires the dispatch environment of its step."
        raise ValueError(message)
    directory = common.paths.resolve_generation_interruptions_directory(
        campaign_evidence.campaign_execution_directory(run_id, storage_root=storage_root)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dispatch.dispatch_key}.json"
    payload = {
        "schema_kind": "generation_worker_interruption",
        "schema_version": 2,
        "campaign_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "recorded_at": _utc_now(),
        "signal": signal_name,
        "exit_code": exit_code,
        **_dispatch_evidence_fields(dispatch),
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
    state_root = _batch_state_directory(
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


def _node_job_status(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> list[dict[str, Any]]:
    """Return one public line of state per node-job record: seat, job id, queue state, node, draining, and unresolved dispatches."""
    submitted = [job for job in manifest["node_jobs"] if job["status"] == "submitted"]
    unresolved = _unresolved_dispatch_counts(manifest, submitted, scheduler)
    lines: list[dict[str, Any]] = []
    for job in manifest["node_jobs"]:
        open_job = False
        state = str(job["status"])
        node: str | None = None
        if job["status"] == "submitted":
            job_id = str(job["job_id"])
            row = scheduler["allocations"]["active"].get(job_id)
            accounted = scheduler["allocations"]["accounted"].get(job_id)
            open_job = _node_job_liveness(job, scheduler, _node_job_directory(manifest, job, storage_root=storage_root)) != "terminal"
            raw_state = _scheduler_field(row, 1) or _scheduler_field(accounted, 1)
            state = _scheduler_state(raw_state) if raw_state is not None else "UNKNOWN"
            node = _scheduler_field(row, 3) if row is not None else _scheduler_field(accounted, 7)
        lines.append(
            {
                "token": job["token"],
                "seat": job["seat"],
                "job_id": job["job_id"],
                "status": job["status"],
                "lease": job["lease"],
                "state": state,
                "node": node,
                "open": open_job,
                "draining": job["draining"],
                "stop_ticket_seq": job["stop_ticket_seq"],
                "cancel_published": job["cancel_published"],
                "unresolved_dispatches": unresolved.get(str(job["token"]), 0),
                "terminal_observed_at": job["terminal_observed_at"],
            }
        )
    return lines


def _persisted_time_window(batch: config_service.GenerationConfig, *, storage_root: Path | str | None) -> dict[str, float | None]:
    """Return the physical start and horizon in hours from a batch's persisted resolved configuration; both ``None`` before it exists."""
    path = batch_runtime.batch_meta_directory(batch, storage_root=storage_root) / "resolved_generation_config.json"
    try:
        time = json.loads(path.read_text(encoding="utf-8"))["time"]
        return {"physical_start_hours": float(time["start"]), "physical_horizon_hours": config_service.physical_horizon_hours(time)}
    except (OSError, ValueError, KeyError, TypeError):
        return {"physical_start_hours": None, "physical_horizon_hours": None}


def _candidate_lineage(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a candidate's decision outcome, exact replaced failed case and typed failure reason: the decision's, else its latest attempt's."""
    design = None if record is None else record["reservation"].get("design_evidence")
    decision = None if record is None else record["decision"]
    if decision is not None:
        failure = decision if decision["outcome"] == "failed" else None
    else:
        failure = record["attempts"][-1] if record is not None and record["attempts"] else None
    return {
        "terminal_decision": None if decision is None else str(decision["outcome"]),
        "parent_failed_case_id": design.get("parent_failed_case_id") if isinstance(design, dict) else None,
        "failure_reason": None if failure is None else failure["reason_code"],
        "failure_detail": None if failure is None else failure["detail"],
    }


def _public_task_view(view: Mapping[str, Any], terminal_decision: str | None) -> dict[str, Any]:
    """Return one work unit in the shared public state vocabulary with its recorded terminal decision outcome, when any."""
    classified_state = str(view["state"])
    if classified_state == "active":
        state = "running"
    elif classified_state in {"pending", "scheduler_unknown"}:
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
    return {**view, "state": state, "classified_state": classified_state, "terminal_decision": terminal_decision}


def _campaign_status_from_snapshot(
    run_id: str,
    *,
    manifest: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    scheduler: Mapping[str, Any],
    task_views: Sequence[Mapping[str, Any]],
    pending_jobs: int,
    running_jobs: int,
    node_jobs: Sequence[Mapping[str, Any]],
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Project one already acquired reconciliation snapshot into public status."""
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
    }
    unknown_cases = sum(view["state"] == "scheduler_unknown" for view in task_views)
    license_retry_eligible_cases = sum(view["state"] == "license_blocked" and view["license_retry_eligible"] is True for view in task_views)
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
    maximum_failed_cases = int(_active_submission_config(manifest)["maximum_failed_cases"])
    circuit_breaker_tripped = _solver_failure_threshold_exceeded(
        task_views,
        maximum_failed_cases=maximum_failed_cases,
    )
    admission_blocked, admission_block_reason = _normal_admission_status(manifest, task_views)
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
    elif manifest["submission_intent"] is not None or manifest["node_job_intent"] is not None or unknown_cases:
        state = "submission_pending_or_unknown"
        next_command = f"status {run_id}"
    elif count_states["license_blocked"]:
        state = "license_blocked"
        next_command = f"status {run_id}"
    elif all_successful and publication_complete:
        state = "successful"
        next_command = f"status {run_id}"
    elif cancellation_requested:
        state = "cancelled"
        next_command = f"run {manifest['campaign_config']}"
    elif count_states["never_started"] or count_states["cancelled"] or count_states["interrupted"] or license_retry_eligible_cases:
        state = "feeding"
        next_command = f"resume {run_id}"
    elif unresolved_cases:
        state = "completed_with_failures"
        next_command = f"resume {run_id}"
    else:
        state = "feeding"
        next_command = f"feed-campaign {run_id}"

    failed_cases = unresolved_cases
    public_task_views = [_public_task_view(view, None) for view in task_views]
    admission = _admission_summary(manifest, task_views)
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
        "git_commit": manifest["git_commit"],
        "execution_config_digest": manifest.get("execution_config_digest"),
        "slurm_job_ids": manifest["slurm_job_ids"],
        "node_jobs": list(node_jobs),
        "open_node_jobs": sum(job["open"] for job in node_jobs),
        "node_job_intent": manifest["node_job_intent"],
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
        "failure_counts": failure_counts,
        "unresolved_solver_failed_cases": unresolved_solver_failures,
        "admission_blocked": admission_blocked,
        "admission_block_reason": admission_block_reason,
        "maximum_failed_cases": maximum_failed_cases,
        "failure_circuit_breaker_tripped": circuit_breaker_tripped,
        "cancellation_requested": cancellation_requested,
        "squeue": manifest_scheduler_view(scheduler["squeue"]),
        "sacct": manifest_scheduler_view(scheduler["sacct"]),
        "sacct_steps": manifest_scheduler_view(scheduler["sacct_steps"]),
        "batches": batches,
        "cases": public_task_views,
        "work_unit_counts": work_unit_counts,
        "remote_storage_root": manifest["remote_storage_root"],
        "campaign_meta_directory": manifest["campaign_meta_directory"],
        "suggested_next_command": next_command,
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
    scheduler = _scheduler_evidence(manifest["slurm_job_ids"] if query_scheduler else ())
    if query_scheduler:
        _require_scheduler_evidence(scheduler)
    scheduler = _dispatch_evidence(manifest, scheduler, storage_root=storage_root)
    task_views, pending_jobs, running_jobs = _reconciled(
        _manifest_with_not_started(manifest, scheduler),
        campaign,
        scheduler,
        storage_root=storage_root,
    )
    node_jobs = _node_job_status(manifest, scheduler, storage_root=storage_root)
    if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
        from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- canonical status evidence
        from src.generation.cases import generation_cases_selection as selection_service  # noqa: PLC0415 -- canonical selected counts

        records = lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
        by_case = {(row["reservation"]["material_family"], row["reservation"]["case_id"]): row for row in records}
        originals = tuple(batch for batch in campaign.batches if not isinstance(batch.scientific_values.get("supplemental_design"), dict))
        by_batch = {
            batch.batch_id: selection_service.candidate_summary(
                [row for row in records if row["reservation"]["batch_id"] == batch.batch_id], target=len(batch.case_indices)
            )
            for batch in originals
        }
        counts = {
            key: sum(summary[key] for summary in by_batch.values())
            for key in (
                "primary",
                "replacement",
                "successful",
                "terminal_failed",
                "active",
                "retryable",
                "pending_validation",
                "active_or_retryable",
                "selected",
                "surplus",
                "replacement_deficit",
            )
        }
        windows = {batch.batch_name: _persisted_time_window(batch, storage_root=storage_root) for batch in campaign.batches}
        return {
            "campaign_run_id": run_id,
            "campaign_state": manifest["state"],
            "campaign_name": campaign.campaign_name,
            "source_name": campaign.source_name,
            "campaign_purpose": campaign.campaign_purpose,
            "root_seed": campaign.root_seed,
            "counts": counts,
            "batches": by_batch,
            "cases": [
                {
                    **_public_task_view(view, None),
                    **_candidate_lineage(by_case.get((view["material"], view["case_id"]))),  # carries the recorded terminal_decision
                    **windows[view["batch_name"]],
                }
                for view in task_views
            ],
            "pending_jobs": pending_jobs,
            "running_jobs": running_jobs,
            "slurm_job_ids": list(manifest["slurm_job_ids"]),
            "node_jobs": node_jobs,
            "open_node_jobs": sum(job["open"] for job in node_jobs),
            "node_job_intent": manifest["node_job_intent"],
            "submission_unknown": manifest["submission_intent"] is not None
            or manifest["node_job_intent"] is not None
            or manifest["state"] == "submission_unknown",
            "suggested_next_command": (
                f"./scripts/generation status {run_id}"
                if campaign.authored_value is not None
                else f"./scripts/generation run {manifest['campaign_config']}"
            ),
        }
    return _campaign_status_from_snapshot(
        run_id,
        manifest=manifest,
        campaign=campaign,
        scheduler=scheduler,
        task_views=task_views,
        pending_jobs=pending_jobs,
        running_jobs=running_jobs,
        node_jobs=node_jobs,
        storage_root=storage_root,
    )


def manifest_scheduler_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return scheduler output without duplicating parsed internal maps."""
    return {
        "command": list(value["command"]),
        "output": value["output"],
        "error": value["error"],
    }


def _dispatch_evidence_fields(dispatch: dispatch_identity.CaseDispatch) -> dict[str, Any]:
    """Return the dispatch identity every worker-side evidence record of one case step carries."""
    return {
        "dispatch_key": dispatch.dispatch_key,
        "slurm_job_id": dispatch.allocation_id,
        "slurm_step_id": dispatch.step_id,
        "submission_index": dispatch.submission_index,
    }


def _bound_dispatch(
    run_id: str,
    dispatch: dispatch_identity.CaseDispatch,
    batch_name: str,
    case_index: int,
    *,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the running step to its dispatch record, waiting for the record to become visible on this node.

    The feeder publishes the ticket before it finalizes the record, and the run record can reach this node over a
    shared file system a little later, so the record is read again for a bounded time before the step fails closed.
    """
    deadline = monotonic() + _DISPATCH_BINDING_WAIT_SECONDS
    while True:
        manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
        records = [record for record in manifest["submissions"] if record["submission_index"] == dispatch.submission_index]
        if len(records) == 1 and records[0]["status"] in {"dispatching", "dispatched"}:
            record = records[0]
            expected_attempt = os.environ.get(_ATTEMPT_INDEX_ENVIRONMENT)
            if (
                record["job_id"] != dispatch.allocation_id
                or record["case"]["batch_name"] != batch_name
                or record["case"]["case_index"] != case_index
                or expected_attempt != str(record["attempt_index"])
            ):
                message = "Current Slurm step is not bound to this exact campaign case."
                raise RuntimeError(message)
            return manifest, record
        if monotonic() >= deadline:
            message = f"Dispatch {dispatch.dispatch_key} has no live record in the campaign run after {_DISPATCH_BINDING_WAIT_SECONDS:.0f} s."
            raise RuntimeError(message)
        sleep(_DISPATCH_BINDING_POLL_SECONDS)


def _record_success_decision(
    campaign: config_service.CampaignConfig,
    task: cluster_service.CampaignTask,
    *,
    resources: Mapping[str, Any] | None,
    storage_root: Path | str | None,
) -> Path:
    """Commit the success decision of one validly published case from its re-validated publication evidence."""
    from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- campaign case evidence boundary

    batch = campaign.batch(task.batch_name)
    provenance = batch_runtime.validate_completed_case(batch, task.case_index, storage_root=storage_root, validation_depth="routine")
    reference = input_service.admit_persisted_input_case(
        batch, task.case_index, str(provenance["input_generation_id"]), storage_root=storage_root, validation_depth="evidence"
    )
    case_payload = reference.case_payload()
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    artifact = batch_runtime.processed_case_directory(batch, task.case_index, storage_root=storage) / common.paths.GENERATION_RESULT_HDF5_FILENAME
    input_manifest = (
        common.paths.resolve_generation_input_generation_metadata_directory(batch.batch_storage_name, reference.source_id, storage_root=storage)
        / "input_generation_manifest.json"
    )
    return lifecycle_service.decide_candidate(
        campaign.profile.id,
        campaign.source_name,
        batch.material_family,
        task.case_id,
        outcome="successful",
        stage="publication",
        failure_class=None,
        reason_code=None,
        detail="Validated canonical case publication",
        case_input_id=str(provenance["case_input_id"]),
        simulation_case_id=str(provenance["simulation_case_id"]),
        case_hdf5_relative=artifact.relative_to(storage).as_posix(),
        case_hdf5_sha256=str(provenance["artifacts"][common.paths.GENERATION_RESULT_HDF5_FILENAME]["sha256"]),
        evidence={
            "input_manifest_relative": input_manifest.relative_to(storage).as_posix(),
            "parameters": case_payload["sampled_values"],
            "fields": case_payload["spatial_diagnostics"],
            "schedules": case_payload.get("schedule_diagnostics"),
            "diagnostics": provenance.get("validation"),
            "source_provenance": provenance,
            "resources": None if resources is None else dict(resources),
        },
        storage_root=storage_root,
    )


def run_campaign_case_job(
    run_id: str,
    batch_name: str,
    case_index: int,
    *,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> batch_runtime.CaseRunOutcome:
    """Execute the exact campaign case bound to the current Slurm step through its dispatch record."""
    dispatch = dispatch_identity.current_case_dispatch()
    if dispatch is None:
        message = "Campaign case execution requires the dispatch environment of its ticket."
        raise RuntimeError(message)
    manifest, dispatch_record = _bound_dispatch(run_id, dispatch, batch_name, case_index, storage_root=storage_root)
    binding = _execution_binding(manifest, submission_index=int(dispatch_record["submission_index"]))
    _validate_binding_source(binding)
    current_commit = _repository_commit()
    if current_commit != binding["git_commit"] or source_service.required_git_commit() != binding["git_commit"]:
        message = "Campaign case worker checkout does not match its admitted execution source."
        raise RuntimeError(message)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION and "execution_values" in binding:
        from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- exact worker execution binding

        campaign = lifecycle_service.campaign_with_execution(campaign, binding["execution_values"])
    task = cluster_service.require_campaign_task(
        campaign,
        batch_name=batch_name,
        case_index=case_index,
    )
    cores = int(binding["submission_config"]["cores_per_case"])
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is None or not allocated.isdigit() or int(allocated) != cores:
        message = f"Campaign case allocation must equal {cores} cores, got {allocated!r}."
        raise RuntimeError(message)
    if dispatch_record["case"] != _task_payload(task):
        message = "Current Slurm step is not bound to this exact campaign case."
        raise RuntimeError(message)
    batch = campaign.batch(task.batch_name)
    persisted_inputs = _persisted_input_sources(manifest)
    selected_input = persisted_inputs.get(task.batch_name)
    if selected_input is None:
        input_references = input_service.admit_configured_input_references(
            batch, storage_root=storage_root, validation_depth="evidence", git_commit=str(binding["git_commit"])
        )
    else:
        input_references = input_service.admit_configured_input_references(
            batch,
            storage_root=storage_root,
            validation_depth="evidence",
            git_commit=str(manifest["git_commit"]),
            input_generation_id=str(selected_input["input_generation_id"]),
            input_source_git_commit=str(selected_input["source_git_commit"]),
        )
    try:
        outcome = cluster_service.run_campaign_case(
            campaign,
            task,
            cores_per_case=cores,
            scheduler_kind="slurm",
            storage_root=storage_root,
            work_root=work_root,
            input_references=input_references,
        )
    except Exception as error:
        if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION:
            from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- campaign case evidence boundary

            attempts = [
                row
                for row in lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
                if row["reservation"]["material_family"] == batch.material_family and row["reservation"]["case_id"] == task.case_id
            ]
            if len(attempts) != 1:
                message = "Campaign candidate reservation is missing during failed case reconciliation."
                raise RuntimeError(message) from error
            failure_stage = str(getattr(error, "failure_stage", "controller"))
            failure_class = (
                "solver"
                if failure_stage == "solver"
                else "validation"
                if failure_stage in {"exports", "conversion"}
                else "publication"
                if failure_stage == "publication"
                else "controller"
            )
            # A solver attempt records how the runtime classified the stop; the exception type stays in the evidence.
            solver_reason = getattr(error, "runtime_reason", None) or lifecycle_service.UNCLASSIFIED_SOLVER_REASON
            lifecycle_service.record_candidate_attempt(
                campaign.profile.id,
                campaign.source_name,
                batch.material_family,
                task.case_id,
                attempt_index=len(attempts[0]["attempts"]) + 1,
                stage=failure_stage,
                failure_class=failure_class,
                reason_code=solver_reason if failure_class == "solver" else type(error).__name__,
                retryable=failure_class in {"controller", "publication"},
                detail=str(error) or type(error).__name__,
                evidence={**_dispatch_evidence_fields(dispatch), "exception_type": type(error).__name__},
                storage_root=storage_root,
            )
        raise
    if campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION and outcome.status in {"completed", "skipped"}:
        _record_success_decision(
            campaign, task, resources={**_dispatch_evidence_fields(dispatch), "allocated_cores": cores}, storage_root=storage_root
        )
    elif campaign.schema_version == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION and outcome.status == "license_blocked":
        from src.generation import generation_campaign_lifecycle as lifecycle_service  # noqa: PLC0415 -- campaign case evidence boundary

        records = [
            row
            for row in lifecycle_service.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
            if row["reservation"]["material_family"] == batch.material_family and row["reservation"]["case_id"] == task.case_id
        ]
        if len(records) != 1:
            message = "Campaign candidate reservation is missing during license retry reconciliation."
            raise RuntimeError(message)
        lifecycle_service.record_candidate_attempt(
            campaign.profile.id,
            campaign.source_name,
            batch.material_family,
            task.case_id,
            attempt_index=len(records[0]["attempts"]) + 1,
            stage="solver",
            failure_class="license",
            reason_code="license_capacity",
            retryable=True,
            detail=outcome.message or "Temporary license capacity unavailable",
            evidence=_dispatch_evidence_fields(dispatch),
            storage_root=storage_root,
        )
    return outcome
