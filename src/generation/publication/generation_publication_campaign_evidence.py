"""Own current campaign-run identity and persisted scheduler evidence.

Responsibilities:
  - Resolve canonical campaign, execution-record, and node-job directory paths, invert a node directory to its campaign, and validate run manifests
  - Reconstruct the exact resolved campaign bound to a launched run
  - Admit exact node-job and case-dispatch ownership without loading scientific payloads

Design principles:
  - Campaign state is independent of purpose-specific terminal evidence
  - Frozen scientific plans own continuation while authored locators are validated separately
  - Node jobs own the allocations and dispatch records own the cases, so both validate against each other

This module does NOT:
  - Submit scheduler jobs, publish tickets, or classify scientific case outcomes
  - Publish Dataset membership or execute solver work

Notes:
  Persisted contract (run.json schema version 1):
    - ``node_jobs`` records one Slurm node job each; ``slurm_job_ids`` lists their allocation ids in record order.
    - ``submissions`` records one dispatch (one ticket for one case step) each and references a node job by token.
    - ``node_job_intent`` names the single unresolved ``submitting`` node job; ``submission_intent`` the single ``dispatching`` record.

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_source as source_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
RUN_MANIFEST_SCHEMA_VERSION: Final = 1
NODE_JOBS_DIRNAME: Final = "node_jobs"
NODE_JOB_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]{16}")
_NODE_SEATS: Final = tuple(range(1, config_service.MAXIMUM_NODE_JOBS + 1))
_NODE_JOB_STATUSES: Final = frozenset({"submitting", "submitted", "submission_failed"})
_NODE_JOB_LEASES: Final = frozenset({"pending", "ok", "failed"})
_NODE_JOB_KEYS: Final = frozenset(
    {
        "token",
        "seat",
        "node_directory",
        "job_id",
        "status",
        "lease",
        "draining",
        "stop_ticket_seq",
        "next_ticket_seq",
        "cancel_published",
        "recorded_at",
        "terminal_observed_at",
        "error",
    }
)
_DISPATCH_KEYS: Final = frozenset(
    {
        "submission_index",
        "mode",
        "recorded_at",
        "case",
        "node_job_token",
        "job_id",
        "ticket_seq",
        "step_name",
        "attempt_index",
        "status",
        "error",
    }
)
_DISPATCH_MODES: Final = frozenset({"initial", "resume", "license_retry", "restart"})
_DISPATCH_STATUSES: Final = frozenset({"dispatching", "dispatched", "not_started"})
STEP_NAME_PATTERN: Final = re.compile(r"[a-z]{2}-[a-z0-9]+-c[0-9]{4,}-s[1-9][0-9]*")
MAX_STEP_NAME_LENGTH: Final = 48
_RUN_MANIFEST_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "campaign_name",
        "source_name",
        "campaign_id",
        "campaign_digest",
        "campaign_config",
        "simulation_profile",
        "selected_batch_names",
        "git_commit",
        "execution_config_digest",
        "slurm_job_ids",
        "node_jobs",
        "node_job_intent",
        "scheduler_log_directory",
        "submission_config",
        "submissions",
        "submission_intent",
        "admission_reservations",
        "remote_storage_root",
        "campaign_meta_directory",
        "batches",
        "state",
    }
)
_RUN_MANIFEST_OPTIONAL_KEYS: Final = frozenset({"source_snapshot"})


def campaign_run_directory(
    run_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one persistent campaign-run metadata directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    if (parts := common.paths.campaign_run_parts(safe_id)) is not None:
        task, campaign_name = parts
        return common.paths.resolve_generation_campaign_directory(task, campaign_name, storage_root=storage_root)
    message = "Campaign-run identity must be one canonical Source handle."
    raise ValueError(message)


def campaign_execution_directory(
    run_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the directory that holds every operational record of one campaign run, below its admitted ``_record/``."""
    campaign_directory = campaign_run_directory(run_id, storage_root=storage_root)
    common.paths.admit_generation_campaign_record_directory(campaign_directory)
    return common.paths.resolve_generation_execution_directory(campaign_directory)


def campaign_run_manifest_path(
    run_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the persistent campaign-run manifest path."""
    return common.paths.resolve_generation_run_manifest_path(campaign_execution_directory(run_id, storage_root=storage_root))


def node_job_relative_directory(token: str) -> str:
    """Return the run-directory-relative node directory persisted for one node-job token."""
    if not isinstance(token, str) or NODE_JOB_TOKEN_PATTERN.fullmatch(token) is None:
        message = f"Node-job token must match {NODE_JOB_TOKEN_PATTERN.pattern!r}, got {token!r}."
        raise ValueError(message)
    return (common.paths.resolve_generation_execution_directory(Path()) / NODE_JOBS_DIRNAME / token).as_posix()


def node_job_campaign_directory(node_directory: Path | str) -> Path | None:
    """Invert one campaign node directory to its campaign directory, or return ``None`` for any other node directory."""
    directory = Path(node_directory)
    if NODE_JOB_TOKEN_PATTERN.fullmatch(directory.name) is None:
        return None
    relative = Path(node_job_relative_directory(directory.name))
    campaign = Path(*directory.parts[: -len(relative.parts)])
    return campaign if len(directory.parts) > len(relative.parts) and campaign / relative == directory else None


def campaign_node_job_directory(
    run_id: str,
    token: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the node directory (tickets, receipts, control, worker records) of one node job."""
    return campaign_run_directory(run_id, storage_root=storage_root) / node_job_relative_directory(token)


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
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("configs",) or relative.suffix != ".yaml":
        message = f"Campaign-run campaign_config is not a canonical generation campaign path: {value!r}."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    candidate = repository / relative
    resolved = candidate.resolve()
    if resolved != candidate:
        message = f"Campaign-run campaign_config contains a symlink: {value!r}."
        raise ValueError(message)
    try:
        resolved.relative_to(repository)
    except ValueError as error:
        message = f"Campaign-run campaign_config escapes the repository: {value!r}."
        raise ValueError(message) from error
    if not resolved.is_file() or resolved.is_symlink():
        message = f"Campaign-run campaign_config is missing or unsafe: {resolved}."
        raise FileNotFoundError(message)
    return resolved


def resolve_frozen_campaign_config_locator(value: Any) -> Path:
    """Validate an authored locator while reading its durable frozen plan."""
    if not isinstance(value, str) or not value:
        message = "Canonical campaign configuration locator must be non-empty text."
        raise TypeError(message)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("configs",) or relative.suffix != ".yaml":
        message = "Canonical campaign configuration locator is unsafe."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    candidate = repository / relative
    if candidate.resolve(strict=False) != candidate:
        message = "Canonical campaign configuration locator escapes its repository."
        raise ValueError(message)
    return candidate


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
        or not common.serialization.is_schema_version(manifest.get("schema_version"), RUN_MANIFEST_SCHEMA_VERSION)
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
        "license_blocked",
        "completed_with_failures",
        "staged",
        "blocked",
        "complete",
        "license_wait_exhausted",
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
    """Validate the persisted feeder and node-job allocation contract."""
    submission = manifest.get("submission_config")
    if not isinstance(submission, dict):
        message = f"Campaign-run submission configuration is malformed: {run_id}."
        raise TypeError(message)
    if set(submission) != {
        "max_admission_cases",
        "poll_interval_seconds",
        "node_jobs",
        "cores_per_case",
        "concurrent_cases_per_node",
        "memory_per_case",
        "node_job_memory",
        "node_job_wall_time",
        "license_wait",
        "partition",
        "scheduler_options",
    }:
        message = f"Campaign-run submission configuration is malformed: {run_id}."
        raise ValueError(message)
    for key in ("max_admission_cases", "poll_interval_seconds", "cores_per_case", "node_jobs", "concurrent_cases_per_node"):
        value = submission[key]
        # A null cap is derived from the plan rates; the campaign's effective cap refuses it without a plan rate.
        if (value is not None or key != "max_admission_cases") and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            message = f"Campaign-run submission configuration {key!r} is malformed: {run_id}."
            raise ValueError(message)
    try:
        for key in ("memory_per_case", "node_job_memory"):
            config_service.parse_slurm_memory_mib(submission[key], label=key)
        config_service.parse_wall_time_seconds(submission["node_job_wall_time"], label="node_job_wall_time")
    except config_service.GenerationConfigError as error:
        message = f"Campaign-run submission configuration node-job resources are malformed: {run_id}."
        raise ValueError(message) from error
    wait = submission["license_wait"]
    in_allocation = wait.get("in_allocation_retry") if isinstance(wait, dict) else None
    if (
        not isinstance(wait, dict)
        or set(wait)
        != {
            "enabled",
            "initial_delay_seconds",
            "maximum_delay_seconds",
            "maximum_wait_seconds",
            "in_allocation_retry",
        }
        or not isinstance(wait.get("enabled"), bool)
        or any(
            isinstance(wait.get(key), bool) or not isinstance(wait.get(key), (int, float)) or not 0.0 < float(wait[key]) < float("inf")
            for key in (
                "initial_delay_seconds",
                "maximum_delay_seconds",
                "maximum_wait_seconds",
            )
        )
        or wait["maximum_delay_seconds"] < wait["initial_delay_seconds"]
        or wait["maximum_wait_seconds"] < wait["initial_delay_seconds"]
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
        message = f"Campaign-run license-wait configuration is malformed: {run_id}."
        raise ValueError(message)
    options = submission["scheduler_options"]
    if not isinstance(options, list) or not all(isinstance(option, str) and option.startswith("--") for option in options):
        message = f"Campaign-run scheduler options are malformed: {run_id}."
        raise ValueError(message)


def _optional_text(value: object) -> bool:
    """Return whether one persisted field is None or a non-empty string."""
    return value is None or (isinstance(value, str) and bool(value))


def _positive_int(value: object) -> bool:
    """Return whether one persisted field is a positive non-boolean integer."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _validate_node_job_record(record: object, *, index: int, run_id: str) -> dict[str, Any]:
    """Validate one persisted node-job record and return it."""
    if (
        not isinstance(record, dict)
        or set(record) != _NODE_JOB_KEYS
        or not isinstance(record.get("token"), str)
        or NODE_JOB_TOKEN_PATTERN.fullmatch(record["token"]) is None
        or isinstance(record.get("seat"), bool)
        or record.get("seat") not in _NODE_SEATS
        or record.get("node_directory") != node_job_relative_directory(record["token"])
        or record.get("status") not in _NODE_JOB_STATUSES
        or record.get("lease") not in _NODE_JOB_LEASES
        or not isinstance(record.get("draining"), bool)
        or not (record.get("stop_ticket_seq") is None or _positive_int(record["stop_ticket_seq"]))
        or not _positive_int(record.get("next_ticket_seq"))
        or not isinstance(record.get("cancel_published"), bool)
        or not isinstance(record.get("recorded_at"), str)
        or not _optional_text(record.get("terminal_observed_at"))
        or not _optional_text(record.get("error"))
    ):
        message = f"Campaign-run node-job record {index} is malformed: {run_id}."
        raise ValueError(message)
    job_id = record["job_id"]
    submitted = record["status"] == "submitted"
    unsubmitted_shape = not submitted and (
        job_id is not None
        or record["lease"] != "pending"
        or record["draining"]
        or record["cancel_published"]
        or record["terminal_observed_at"] is not None
    )
    if (
        (submitted and (not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None))
        or unsubmitted_shape
        or (record["status"] == "submission_failed") != (record["error"] is not None and not submitted)
        or (submitted and record["error"] is not None and record["lease"] != "failed")
        or (record["stop_ticket_seq"] is not None and (not record["draining"] or record["stop_ticket_seq"] >= record["next_ticket_seq"]))
    ):
        message = f"Campaign-run node-job record {index} is inconsistent: {run_id}."
        raise ValueError(message)
    return record


def _validate_node_job_records(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    job_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate every node-job record and the node-job intent; return the records by token."""
    records = manifest.get("node_jobs")
    if not isinstance(records, list):
        message = f"Campaign-run node jobs are malformed: {run_id}."
        raise TypeError(message)
    by_token: dict[str, dict[str, Any]] = {}
    known: list[str] = []
    unresolved: list[str] = []
    for index, raw in enumerate(records, start=1):
        record = _validate_node_job_record(raw, index=index, run_id=run_id)
        if record["token"] in by_token:
            message = f"Campaign-run node-job token is duplicated: {run_id}."
            raise ValueError(message)
        by_token[record["token"]] = record
        if record["job_id"] is not None:
            known.append(record["job_id"])
        if record["status"] == "submitting":
            unresolved.append(record["token"])
    if known != list(job_ids):
        message = f"Campaign-run job IDs disagree with node-job records: {run_id}."
        raise ValueError(message)
    intent = manifest.get("node_job_intent")
    if (intent is None and unresolved) or (intent is not None and unresolved != [intent]):
        message = f"Campaign-run durable node-job intent is inconsistent: {run_id}."
        raise ValueError(message)
    return by_token


def _validate_dispatch_record(
    record: object,
    *,
    index: int,
    run_id: str,
    node_jobs: Mapping[str, Mapping[str, Any]],
) -> str:
    """Validate one persisted dispatch record against its node job and return its status."""
    case = record.get("case") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != _DISPATCH_KEYS
        or record.get("submission_index") != index
        or record.get("mode") not in _DISPATCH_MODES
        or not isinstance(record.get("recorded_at"), str)
        or not isinstance(case, dict)
        or set(case) != {"batch_name", "batch_id", "case_index", "case_id"}
        or not _positive_int(case.get("case_index"))
        or not isinstance(record.get("node_job_token"), str)
        or not isinstance(record.get("job_id"), str)
        or _JOB_ID_PATTERN.fullmatch(record["job_id"]) is None
        or not _positive_int(record.get("ticket_seq"))
        or not isinstance(record.get("step_name"), str)
        or len(record["step_name"]) > MAX_STEP_NAME_LENGTH
        or STEP_NAME_PATTERN.fullmatch(record["step_name"]) is None
        or not record["step_name"].endswith(f"-c{case['case_index']:04d}-s{index}")
        or not _positive_int(record.get("attempt_index"))
        or record.get("status") not in _DISPATCH_STATUSES
        or (record["status"] == "not_started") != isinstance(record.get("error"), str)
        or record.get("error") == ""
    ):
        message = f"Campaign-run dispatch record {index} is malformed: {run_id}."
        raise ValueError(message)
    for key in ("batch_name", "batch_id", "case_id"):
        common.paths.validate_logical_name(case.get(key), label=f"dispatch case {key}")
    node_job = node_jobs.get(record["node_job_token"])
    if node_job is None or node_job["status"] != "submitted" or node_job["job_id"] != record["job_id"]:
        message = f"Campaign-run dispatch record {index} references no submitted node job of its allocation: {run_id}."
        raise ValueError(message)
    sequence, cursor = record["ticket_seq"], node_job["next_ticket_seq"]
    if (record["status"] == "dispatching" and sequence != cursor) or (record["status"] == "dispatched" and sequence >= cursor) or sequence > cursor:
        message = f"Campaign-run dispatch record {index} disagrees with its node job's ticket cursor: {run_id}."
        raise ValueError(message)
    return str(record["status"])


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
    node_jobs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate every dispatch record against the node jobs, the ticket and step-name uniqueness rules, and the dispatch intent."""
    submissions = manifest.get("submissions")
    if not isinstance(submissions, list):
        message = f"Campaign-run submissions are malformed: {run_id}."
        raise TypeError(message)
    unresolved: list[int] = []
    tickets: set[tuple[str, int]] = set()
    steps: set[str] = set()
    for index, record in enumerate(submissions, start=1):
        status = _validate_dispatch_record(record, index=index, run_id=run_id, node_jobs=node_jobs)
        if status != "not_started":
            ticket = (record["node_job_token"], record["ticket_seq"])
            if ticket in tickets:
                message = f"Campaign-run dispatch ticket is duplicated: {run_id}."
                raise ValueError(message)
            tickets.add(ticket)
        if record["step_name"] in steps:
            message = f"Campaign-run dispatch step name is duplicated: {run_id}."
            raise ValueError(message)
        steps.add(record["step_name"])
        if status == "dispatching":
            unresolved.append(index)
    for token, node_job in node_jobs.items():
        if node_job["stop_ticket_seq"] is not None and (token, node_job["stop_ticket_seq"]) in tickets:
            message = f"Campaign-run stop ticket collides with a dispatch ticket: {run_id}."
            raise ValueError(message)
    intent = manifest.get("submission_intent")
    if (intent is None and unresolved) or (intent is not None and unresolved != [intent]):
        message = f"Campaign-run durable submission intent is inconsistent: {run_id}."
        raise ValueError(message)


def _snapshot_path_for_storage(
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    storage: Path,
) -> Path:
    """Map one frozen storage locator into an explicit read root without rewriting it."""
    stored_root = manifest.get("remote_storage_root")
    stored_path = snapshot.get("path")
    if not isinstance(stored_root, str) or not isinstance(stored_path, str):
        message = "Canonical campaign source snapshot locator is malformed."
        raise ValueError(message)  # noqa: TRY004 -- persisted JSON evidence retains the ValueError admission contract
    origin = Path(stored_root)
    path = Path(stored_path)
    if (
        not origin.is_absolute()
        or origin.parent == origin
        or ".." in origin.parts
        or not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(common.paths.get_source_snapshots_root(storage_root=origin))
    ):
        message = "Canonical campaign source snapshot locator escapes its persisted storage root."
        raise ValueError(message)
    return storage / path.relative_to(origin)


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
    node_jobs = _validate_node_job_records(manifest, run_id=run_id, job_ids=job_ids)
    _validate_submission_records(manifest, run_id=run_id, node_jobs=node_jobs)
    resolve_frozen_campaign_config_locator(manifest.get("campaign_config"))
    if "source_snapshot" in manifest:
        snapshot = manifest["source_snapshot"]
        storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
        root = common.paths.get_source_snapshots_root(storage_root=storage)
        if not isinstance(snapshot, dict) or set(snapshot) != {"path", "sha256", "source_fingerprint_sha256"}:
            message = "Canonical campaign source snapshot binding is malformed."
            raise ValueError(message)
        path = _snapshot_path_for_storage(snapshot, manifest, storage=storage)
        if (
            not path.is_absolute()
            or not path.is_relative_to(root)
            or not path.resolve(strict=False).is_relative_to(root)
            or path.is_symlink()
            or not path.is_file()
            or re.fullmatch(r"[0-9a-f]{64}", str(snapshot["sha256"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(snapshot["source_fingerprint_sha256"])) is None
            or common.serialization.file_sha256(path) != snapshot["sha256"]
        ):
            message = "Canonical campaign source snapshot is missing, unsafe, or changed."
            raise ValueError(message)
    return manifest


def load_campaign_scheduler_job_ids(
    run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> tuple[str, ...]:
    """
    Load only the exact persisted node-job allocations of one campaign run.

    Raises
    ------
    RuntimeError
        If a node-job submission is unresolved: its allocation may exist without being recorded.
    ValueError
        If the manifest is malformed or inconsistent.

    """
    path = campaign_run_manifest_path(run_id, storage_root=storage_root)
    if not path.is_file() or path.is_symlink():
        message = f"Campaign-run manifest is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    manifest = load_json_object(path, label="campaign-run manifest")
    if (
        manifest.get("schema_kind") != "generation_campaign_run"
        or not common.serialization.is_schema_version(manifest.get("schema_version"), RUN_MANIFEST_SCHEMA_VERSION)
        or manifest.get("campaign_run_id") != run_id
    ):
        message = f"Unsupported or malformed campaign-run manifest: {run_id}."
        raise ValueError(message)
    job_ids = _validate_campaign_scheduler_job_ids(
        manifest.get("slurm_job_ids"),
        run_id=run_id,
    )
    node_jobs = _validate_node_job_records(manifest, run_id=run_id, job_ids=job_ids)
    _validate_submission_records(manifest, run_id=run_id, node_jobs=node_jobs)
    if manifest["node_job_intent"] is not None:
        message = f"Campaign-run node-job submission is unresolved, so its allocations are not fully known: {run_id}."
        raise RuntimeError(message)
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
    """Reconstruct the executable campaign from its immutable frozen source plan."""
    if common.paths.campaign_run_parts(str(manifest.get("campaign_run_id", ""))) is None:
        message = "Campaign execution requires a canonical source campaign execution handle."
        raise ValueError(message)
    from src.generation import generation_campaign_lifecycle as lifecycle  # noqa: PLC0415 -- frozen source plan owner

    campaign = lifecycle.campaign_from_frozen_run(manifest, require_executable=require_executable)
    _validate_admission_reservations_for_campaign(manifest, campaign)
    return campaign


def campaign_from_manifest(
    manifest: Mapping[str, Any],
    *,
    require_executable: bool = True,
) -> config_service.CampaignConfig:
    """Resolve the frozen source campaign for continuation or inspection."""
    return current_campaign_from_manifest(manifest, require_executable=require_executable)
