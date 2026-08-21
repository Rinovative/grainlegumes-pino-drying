"""
generation_runtime_license.py

Classify and persist retry evidence for temporary COMSOL license capacity.
Responsibilities:
  - Recognize conservative floating-license capacity signatures in captured logs
  - Derive capped exponential retry delays from resolved execution policy
  - Persist and validate one mutable compact wait record per blocked work unit
Design principles:
  - License exit codes never replace captured-text classification
  - Retry history is operational provenance and preserves scientific case identity
  - Scheduler allocations end before controller-side backoff begins
This module does NOT:
  - Submit scheduler jobs, sleep inside allocations, or classify other failures
  - Change COMSOL models, scientific inputs, or case-publication admission
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.generation.cases import generation_cases_config as config_contract

TEMPORARY_LICENSE_CAPACITY: Final = "temporary_license_capacity"
EXHAUSTED_REASON: Final = "temporary COMSOL license capacity remained unavailable for the configured maximum wait"
_FEATURE_PATTERN: Final = re.compile(
    r"Could\s+not\s+obtain\s+license\s+for\s+['\"](?P<feature>[^'\"\r\n]+)['\"]",
    flags=re.IGNORECASE,
)
_USERS_REACHED_PATTERN: Final = re.compile(
    r"Licensed\s+number\s+of\s+users\s+already\s+reached",
    flags=re.IGNORECASE,
)
_LICENSE_MINUS_FOUR_PATTERN: Final = re.compile(
    r"License\s+error\s*:?\s*-4(?:\D|$)",
    flags=re.IGNORECASE,
)
_FLEXNET_MINUS_FOUR_PATTERN: Final = re.compile(
    r"FlexNet\s+Licensing\s+error\s*:\s*(?P<code>-4(?:,\d+)?)",
    flags=re.IGNORECASE,
)
_NON_CAPACITY_LICENSE_PATTERN: Final = re.compile(
    r"(?:invalid\s+license\s+file|fundamentally\s+not\s+licensed|"
    r"product\s+is\s+not\s+licensed|license\s+server\s+configuration)",
    flags=re.IGNORECASE,
)
_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_SOLVER_PROGRESS_PATTERN: Final = re.compile(
    r"(?:Time-Dependent Solver|Stationary Solver|Solution time|^\s*Step\s+Time(?:\s|$))",
    flags=re.IGNORECASE | re.MULTILINE,
)
_WAIT_SCHEMA_KIND: Final = "generation_temporary_license_wait"
_WAIT_SCHEMA_VERSION: Final = 1
_WAIT_FILENAME: Final = "license_wait.json"
_MAX_RAW_EXCERPT_CHARACTERS: Final = 4096
_MAX_RECENT_JOB_IDS: Final = 16
_WAIT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "work_unit_id",
        "scientific_config_digest",
        "classification",
        "feature",
        "error_code",
        "matched_signatures",
        "comsol_exit_code",
        "solver_progress_started",
        "expected_exports_exist",
        "first_blocked_at",
        "last_blocked_at",
        "retry_count",
        "latest_job_id",
        "recent_job_ids",
        "hostname",
        "raw_excerpt",
        "delay_before_next_attempt_seconds",
        "cumulative_wait_seconds",
        "retry_budget_remaining",
        "next_retry_at",
    }
)


@dataclass(frozen=True, slots=True)
class TemporaryLicenseCapacityClassification:
    """Conservative evidence for one retryable pre-solver capacity event."""

    classification: str
    feature: str
    license_code: str | None
    matched_signatures: tuple[str, ...]
    raw_excerpt: str


class TemporaryLicenseCapacityError(RuntimeError):
    """Report one retryable pre-solve COMSOL floating-license capacity event."""

    def __init__(
        self,
        message: str,
        *,
        work_directory: Path,
        command: tuple[str, ...],
        exit_code: int | None,
        evidence: TemporaryLicenseCapacityClassification,
        solver_progress_started: bool = False,
        expected_exports_exist: bool = False,
    ) -> None:
        """Initialize one classified temporary infrastructure failure."""
        super().__init__(message)
        self.work_directory = work_directory
        self.cwd = work_directory
        self.command = command
        self.exit_code = exit_code
        self.timed_out = False
        self.missing_or_invalid_artifacts: tuple[str, ...] = ()
        self.failure_stage = "solver"
        self.evidence = evidence
        self.solver_progress_started = solver_progress_started
        self.expected_exports_exist = expected_exports_exist


def _bounded_raw_excerpt(captured_text: str) -> str:
    """Return bounded head-and-tail raw COMSOL evidence."""
    normalized = captured_text.replace("\x00", "").strip()
    if len(normalized) <= _MAX_RAW_EXCERPT_CHARACTERS:
        return normalized
    marker = "\n... raw license evidence middle omitted ...\n"
    available = _MAX_RAW_EXCERPT_CHARACTERS - len(marker)
    head = available // 2
    tail = available - head
    return normalized[:head] + marker + normalized[-tail:]


def solver_progress_started(captured_text: str) -> bool:
    """Return whether bounded COMSOL evidence proves solver progress began."""
    if not isinstance(captured_text, str):
        message = "Captured COMSOL progress evidence must be text."
        raise TypeError(message)
    return _SOLVER_PROGRESS_PATTERN.search(captured_text) is not None


def classify_temporary_license_capacity(
    captured_text: str,
) -> TemporaryLicenseCapacityClassification | None:
    """Classify only strong COMSOL floating-license capacity evidence."""
    if not isinstance(captured_text, str):
        message = "Captured COMSOL license evidence must be text."
        raise TypeError(message)
    if _NON_CAPACITY_LICENSE_PATTERN.search(captured_text) is not None:
        return None
    feature_match = _FEATURE_PATTERN.search(captured_text)
    if feature_match is None:
        return None
    exact_signatures: list[str] = [feature_match.group(0).strip()]
    users_match = _USERS_REACHED_PATTERN.search(captured_text)
    license_match = _LICENSE_MINUS_FOUR_PATTERN.search(captured_text)
    flexnet_match = _FLEXNET_MINUS_FOUR_PATTERN.search(captured_text)
    exact_signatures.extend(match.group(0).strip() for match in (users_match, license_match, flexnet_match) if match is not None)
    if users_match is None and license_match is None and flexnet_match is None:
        return None
    license_code = str(flexnet_match.group("code")) if flexnet_match is not None else "-4" if license_match is not None else None
    return TemporaryLicenseCapacityClassification(
        classification=TEMPORARY_LICENSE_CAPACITY,
        feature=feature_match.group("feature").strip(),
        license_code=license_code,
        matched_signatures=tuple(exact_signatures),
        raw_excerpt=_bounded_raw_excerpt(captured_text),
    )


def controller_owned_window_deadline_evidence() -> TemporaryLicenseCapacityClassification:
    """Return structured retry evidence for the owned acquisition deadline."""
    reason = "controller_owned_in_allocation_license_window_deadline"
    return TemporaryLicenseCapacityClassification(
        classification=TEMPORARY_LICENSE_CAPACITY,
        feature="COMSOL license acquisition",
        license_code=None,
        matched_signatures=(reason,),
        raw_excerpt=reason,
    )


def bounded_retry_delay_seconds(
    policy: Mapping[str, Any],
    *,
    attempt_index: int,
    cumulative_wait_seconds: float,
) -> float:
    """Return the next capped delay, optionally bounded by cumulative wait."""
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        message = "License retry attempt_index must be a positive integer."
        raise ValueError(message)
    if isinstance(cumulative_wait_seconds, bool) or not isinstance(
        cumulative_wait_seconds,
        (int, float),
    ):
        message = "License retry cumulative wait must be a finite non-negative number."
        raise TypeError(message)
    cumulative = float(cumulative_wait_seconds)
    if not 0.0 <= cumulative < float("inf"):
        message = "License retry cumulative wait must be a finite non-negative number."
        raise ValueError(message)
    initial = float(policy["initial_delay_seconds"])
    maximum = float(policy["maximum_delay_seconds"])
    delay = initial
    for _index in range(1, attempt_index):
        if delay >= maximum:
            delay = maximum
            break
        delay = min(maximum, delay * 2.0)
    maximum_wait = policy["maximum_wait_seconds"]
    if maximum_wait is None:
        return delay
    remaining = max(0.0, float(maximum_wait) - cumulative)
    return min(delay, remaining)


def temporary_license_wait_directory(
    campaign_run_id: str,
    batch_id: str,
    case_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the canonical per-case operational retry-evidence directory."""
    run_directory = campaign_evidence.campaign_run_directory(
        campaign_run_id,
        storage_root=storage_root,
    )
    safe_batch = common.paths.validate_logical_name(batch_id, label="batch_id")
    safe_case = common.paths.validate_logical_name(case_id, label="case_id")
    return run_directory / "license_retries" / safe_batch / safe_case


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    """Return one timezone-aware persisted timestamp."""
    if not isinstance(value, str):
        message = f"{label} must be a timezone-aware timestamp."
        raise TypeError(message)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        message = f"{label} must be a timezone-aware timestamp."
        raise ValueError(message) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        message = f"{label} must be a timezone-aware timestamp."
        raise ValueError(message)
    return parsed.astimezone(timezone.utc)


def _validate_wait_record(
    payload: object,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
    campaign_run_id: str,
) -> dict[str, Any]:
    """Validate one mutable compact wait record against exact identities."""
    case_id = config.case_id(case_index)
    if (
        not isinstance(payload, dict)
        or set(payload) != _WAIT_KEYS
        or payload.get("schema_kind") != _WAIT_SCHEMA_KIND
        or payload.get("schema_version") != _WAIT_SCHEMA_VERSION
        or payload.get("campaign_run_id") != campaign_run_id
        or payload.get("batch_id") != config.batch_id
        or payload.get("case_id") != case_id
        or payload.get("work_unit_id") != f"{config.batch_id}/{case_id}"
        or payload.get("scientific_config_digest") != config.scientific_config_digest
        or payload.get("classification") != TEMPORARY_LICENSE_CAPACITY
        or not isinstance(payload.get("feature"), str)
        or not payload["feature"]
        or (payload.get("error_code") is not None and not isinstance(payload.get("error_code"), str))
        or not isinstance(payload.get("matched_signatures"), list)
        or not payload["matched_signatures"]
        or not all(isinstance(value, str) and value for value in payload["matched_signatures"])
        or not isinstance(payload.get("hostname"), str)
        or not payload["hostname"]
        or not isinstance(payload.get("raw_excerpt"), str)
        or not payload["raw_excerpt"]
        or len(payload["raw_excerpt"]) > _MAX_RAW_EXCERPT_CHARACTERS
        or not isinstance(payload.get("retry_budget_remaining"), bool)
        or payload.get("solver_progress_started") is not False
        or payload.get("expected_exports_exist") is not False
    ):
        message = f"Temporary-license wait evidence is malformed for {config.batch_id}/{case_id}."
        raise ValueError(message)
    exit_code = payload.get("comsol_exit_code")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        message = f"Temporary-license COMSOL exit code is malformed for {case_id}."
        raise ValueError(message)
    retry_count = payload.get("retry_count")
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 1:
        message = f"Temporary-license retry count is malformed for {case_id}."
        raise ValueError(message)
    latest_job_id = payload.get("latest_job_id")
    recent_job_ids = payload.get("recent_job_ids")
    if (
        not isinstance(latest_job_id, str)
        or _JOB_ID_PATTERN.fullmatch(latest_job_id) is None
        or not isinstance(recent_job_ids, list)
        or not recent_job_ids
        or len(recent_job_ids) > _MAX_RECENT_JOB_IDS
        or len(recent_job_ids) != len(set(recent_job_ids))
        or not all(isinstance(value, str) and _JOB_ID_PATTERN.fullmatch(value) is not None for value in recent_job_ids)
        or recent_job_ids[-1] != latest_job_id
    ):
        message = f"Temporary-license job history is malformed for {case_id}."
        raise ValueError(message)
    first = _parse_timestamp(payload.get("first_blocked_at"), label="temporary-license first blocked timestamp")
    last = _parse_timestamp(payload.get("last_blocked_at"), label="temporary-license last blocked timestamp")
    if last < first:
        message = f"Temporary-license blocked timestamps are inconsistent for {case_id}."
        raise ValueError(message)
    delay = payload.get("delay_before_next_attempt_seconds")
    cumulative = payload.get("cumulative_wait_seconds")
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not 0.0 <= float(delay) < float("inf")
        or isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
        or not 0.0 <= float(cumulative) < float("inf")
    ):
        message = f"Temporary-license retry delays are malformed for {case_id}."
        raise ValueError(message)
    prior_cumulative = float(cumulative) - float(delay)
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    expected_delay = bounded_retry_delay_seconds(
        policy,
        attempt_index=retry_count,
        cumulative_wait_seconds=prior_cumulative,
    )
    if prior_cumulative < 0.0 or float(delay) != expected_delay:
        message = f"Temporary-license retry budget is inconsistent for {case_id}."
        raise ValueError(message)
    remaining = bool(payload["retry_budget_remaining"])
    next_retry = payload.get("next_retry_at")
    if remaining:
        retry_at = _parse_timestamp(next_retry, label="temporary-license next retry timestamp")
        if float(delay) <= 0.0 or retry_at != last + timedelta(seconds=float(delay)):
            message = f"Temporary-license next retry is inconsistent for {case_id}."
            raise ValueError(message)
    elif next_retry is not None or float(delay) != 0.0:
        message = f"Exhausted temporary-license wait evidence is inconsistent for {case_id}."
        raise ValueError(message)
    return payload


def _owned_atomic_temporary_entry(path: Path, *, destination: Path) -> bool:
    """Return whether a safe present or just-published atomic sibling is owned."""
    if common.serialization.atomic_write_temporary_destination(path) != destination:
        return False
    try:
        entry_status = path.lstat()
    except FileNotFoundError:
        return True
    return stat.S_ISREG(entry_status.st_mode)


def load_temporary_license_wait(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load the optional compact wait record for one blocked work unit."""
    directory = temporary_license_wait_directory(
        campaign_run_id,
        config.batch_id,
        config.case_id(case_index),
        storage_root=storage_root,
    )
    if not directory.exists():
        return None
    if not directory.is_dir() or directory.is_symlink():
        message = f"Temporary-license retry evidence directory is unsafe: {directory}"
        raise ValueError(message)
    entries = sorted(directory.iterdir())
    wait_path = directory / _WAIT_FILENAME
    for entry in entries:
        if entry == wait_path:
            if entry.is_symlink() or not entry.is_file():
                message = f"Temporary-license retry evidence directory contains unexpected entries: {directory}"
                raise ValueError(message)
            continue
        if _owned_atomic_temporary_entry(entry, destination=wait_path):
            continue
        message = f"Temporary-license retry evidence directory contains unexpected entries: {directory}"
        raise ValueError(message)
    if not wait_path.exists():
        return None
    if wait_path.is_symlink() or not wait_path.is_file():
        message = f"Temporary-license retry evidence directory contains unexpected entries: {directory}"
        raise ValueError(message)
    try:
        payload = json.loads(wait_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load temporary-license wait evidence: {wait_path}"
        raise ValueError(message) from error
    return _validate_wait_record(
        payload,
        config=config,
        case_index=case_index,
        campaign_run_id=campaign_run_id,
    )


def record_temporary_license_wait(
    config: config_contract.GenerationConfig,
    case_index: int,
    error: TemporaryLicenseCapacityError,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Update one compact per-work-unit wait record before scratch is reclaimed."""
    campaign_run_id = common.paths.validate_logical_name(
        os.environ.get("GENERATION_CAMPAIGN_RUN_ID"),
        label="campaign_run_id",
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Temporary-license retry evidence requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    current = load_temporary_license_wait(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    if current is None:
        retry_count = 1
        prior_cumulative = 0.0
        first_blocked_at = datetime.now(timezone.utc).isoformat()
        recent_job_ids: list[str] = []
    else:
        retry_count = int(current["retry_count"]) + 1
        prior_cumulative = float(current["cumulative_wait_seconds"])
        first_blocked_at = str(current["first_blocked_at"])
        recent_job_ids = [str(value) for value in current["recent_job_ids"]]
    if job_id in recent_job_ids:
        message = f"Temporary-license retry evidence already includes Slurm job {job_id}."
        raise FileExistsError(message)
    recent_job_ids.append(job_id)
    recent_job_ids = recent_job_ids[-_MAX_RECENT_JOB_IDS:]
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    delay = bounded_retry_delay_seconds(
        policy,
        attempt_index=retry_count,
        cumulative_wait_seconds=prior_cumulative,
    )
    cumulative = prior_cumulative + delay
    timestamp = datetime.now(timezone.utc)
    retry_remaining = delay > 0.0
    case_id = config.case_id(case_index)
    payload = {
        "schema_kind": _WAIT_SCHEMA_KIND,
        "schema_version": _WAIT_SCHEMA_VERSION,
        "campaign_run_id": campaign_run_id,
        "batch_id": config.batch_id,
        "case_id": case_id,
        "work_unit_id": f"{config.batch_id}/{case_id}",
        "scientific_config_digest": config.scientific_config_digest,
        "classification": error.evidence.classification,
        "feature": error.evidence.feature,
        "error_code": error.evidence.license_code,
        "matched_signatures": list(error.evidence.matched_signatures),
        "comsol_exit_code": error.exit_code,
        "solver_progress_started": error.solver_progress_started,
        "expected_exports_exist": error.expected_exports_exist,
        "first_blocked_at": first_blocked_at,
        "last_blocked_at": timestamp.isoformat(),
        "retry_count": retry_count,
        "latest_job_id": job_id,
        "recent_job_ids": recent_job_ids,
        "hostname": socket.gethostname(),
        "raw_excerpt": error.evidence.raw_excerpt,
        "delay_before_next_attempt_seconds": delay,
        "cumulative_wait_seconds": cumulative,
        "retry_budget_remaining": retry_remaining,
        "next_retry_at": ((timestamp + timedelta(seconds=delay)).isoformat() if retry_remaining else None),
    }
    directory = temporary_license_wait_directory(
        campaign_run_id,
        config.batch_id,
        case_id,
        storage_root=storage_root,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _WAIT_FILENAME
    common.serialization.atomic_write_json(path, payload)
    admitted = load_temporary_license_wait(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    if admitted != payload:
        message = f"Temporary-license wait evidence did not re-admit after publication: {path}"
        raise RuntimeError(message)
    return path


def latest_wait_for_job(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Return canonical wait evidence only when it belongs to the latest job."""
    wait = load_temporary_license_wait(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    if wait is None:
        return None
    return wait if wait["latest_job_id"] == job_id else None


def wait_record_is_eligible(
    attempt: Mapping[str, Any],
    *,
    at: datetime | None = None,
) -> bool:
    """Return whether a retryable wait record reached its eligibility time."""
    if not bool(attempt["retry_budget_remaining"]):
        return False
    next_retry_at = attempt["next_retry_at"]
    eligible = _parse_timestamp(
        next_retry_at,
        label="temporary-license next retry eligibility",
    )
    current = datetime.now(timezone.utc) if at is None else at
    if current.tzinfo is None or current.utcoffset() is None:
        message = "Retry eligibility comparison requires a timezone-aware time."
        raise ValueError(message)
    return current.astimezone(timezone.utc) >= eligible


_WINDOW_SCHEMA_KIND: Final = "generation_in_allocation_license_window"
_WINDOW_SCHEMA_VERSION: Final = 1
_WINDOW_DIRECTORY_NAME: Final = "license_retry_windows"
_MAX_RECENT_CHECKOUT_SUMMARIES: Final = 8
_MAX_CHECKOUT_EXCERPT_CHARACTERS: Final = 512
_WINDOW_OUTCOMES: Final = frozenset({"solver_progress_started", "window_exhausted"})
_SCIENTIFIC_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class InAllocationLicenseCheckoutSummary:
    """One compact completed COMSOL checkout attempt within an allocation."""

    checkout_index: int
    started_at: datetime
    ended_at: datetime
    started_monotonic_seconds: float
    ended_monotonic_seconds: float
    process_exit_code: int | None
    classification: TemporaryLicenseCapacityClassification | None
    solver_progress_started: bool

    def __post_init__(self) -> None:
        """Validate bounded, monotonic checkout evidence at construction."""
        if isinstance(self.checkout_index, bool) or self.checkout_index < 1:
            message = "In-allocation checkout index must be a positive integer."
            raise ValueError(message)
        for label, monotonic_value in (
            ("checkout start", self.started_monotonic_seconds),
            ("checkout end", self.ended_monotonic_seconds),
        ):
            if isinstance(monotonic_value, bool) or not isinstance(monotonic_value, (int, float)) or not 0.0 <= float(monotonic_value) < float("inf"):
                message = f"In-allocation {label} monotonic time must be finite and non-negative."
                raise ValueError(message)
        if self.ended_monotonic_seconds < self.started_monotonic_seconds:
            message = "In-allocation checkout end cannot precede its start."
            raise ValueError(message)
        for label, timestamp_value in (("checkout start", self.started_at), ("checkout end", self.ended_at)):
            if timestamp_value.tzinfo is None or timestamp_value.utcoffset() is None:
                message = f"In-allocation {label} timestamp must be timezone-aware."
                raise ValueError(message)
        if self.ended_at < self.started_at:
            message = "In-allocation checkout timestamp end cannot precede its start."
            raise ValueError(message)
        if self.process_exit_code is not None and (isinstance(self.process_exit_code, bool) or not isinstance(self.process_exit_code, int)):
            message = "In-allocation checkout exit code must be an integer or null."
            raise TypeError(message)
        if self.solver_progress_started and self.classification is not None:
            message = "Solver-progress checkout evidence cannot be temporary-capacity classified."
            raise ValueError(message)

    @property
    def duration_seconds(self) -> float:
        """Return the monotonic duration of this completed checkout."""
        return float(self.ended_monotonic_seconds - self.started_monotonic_seconds)


@dataclass(frozen=True, slots=True)
class InAllocationLicenseWindowResult:
    """One completed bounded license-acquisition window for one Slurm allocation."""

    campaign_run_id: str
    batch_id: str
    case_id: str
    work_unit_id: str
    scientific_config_digest: str
    job_id: str
    hostname: str
    window_started_at: datetime
    window_ended_at: datetime
    window_started_monotonic_seconds: float
    window_ended_monotonic_seconds: float
    configured_window_seconds: float
    checkout_summaries: tuple[InAllocationLicenseCheckoutSummary, ...]
    solver_progress_started: bool
    outcome: str
    reason: str
    next_controller_retry_basis: str | None

    def __post_init__(self) -> None:
        """Validate the immutable operational result before persistence."""
        if _JOB_ID_PATTERN.fullmatch(self.job_id) is None:
            message = "In-allocation license window requires one numeric Slurm job ID."
            raise ValueError(message)
        for label, text_value in (
            ("campaign_run_id", self.campaign_run_id),
            ("batch_id", self.batch_id),
            ("case_id", self.case_id),
            ("hostname", self.hostname),
        ):
            if not isinstance(text_value, str) or not text_value:
                message = f"In-allocation license window {label} must be non-empty text."
                raise ValueError(message)
        if self.work_unit_id != f"{self.batch_id}/{self.case_id}":
            message = "In-allocation license window work-unit identity is inconsistent."
            raise ValueError(message)
        if not isinstance(self.scientific_config_digest, str) or _SCIENTIFIC_DIGEST_PATTERN.fullmatch(self.scientific_config_digest) is None:
            message = "In-allocation license window scientific digest is malformed."
            raise ValueError(message)
        for label, timestamp_value in (("window start", self.window_started_at), ("window end", self.window_ended_at)):
            if timestamp_value.tzinfo is None or timestamp_value.utcoffset() is None:
                message = f"In-allocation {label} timestamp must be timezone-aware."
                raise ValueError(message)
        if self.window_ended_at < self.window_started_at:
            message = "In-allocation window end cannot precede its start."
            raise ValueError(message)
        for label, numeric_value in (
            ("window start", self.window_started_monotonic_seconds),
            ("window end", self.window_ended_monotonic_seconds),
            ("configured window", self.configured_window_seconds),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, (int, float)) or not 0.0 < float(numeric_value) < float("inf"):
                message = f"In-allocation {label} duration must be finite and positive."
                raise ValueError(message)
        if self.window_ended_monotonic_seconds < self.window_started_monotonic_seconds:
            message = "In-allocation window monotonic end cannot precede its start."
            raise ValueError(message)
        indexes = tuple(summary.checkout_index for summary in self.checkout_summaries)
        if indexes != tuple(range(1, len(indexes) + 1)):
            message = "In-allocation checkout summaries must use contiguous indexes."
            raise ValueError(message)
        if self.outcome not in _WINDOW_OUTCOMES:
            message = "In-allocation license window outcome is unsupported."
            raise ValueError(message)
        if self.outcome == "solver_progress_started":
            if not self.solver_progress_started or self.reason != "solver_progress_started" or self.next_controller_retry_basis is not None:
                message = "Solver-progress license window outcome is inconsistent."
                raise ValueError(message)
        elif (
            self.solver_progress_started
            or self.reason != "in_allocation_license_window_exhausted"
            or self.next_controller_retry_basis != "controller_temporary_license_retry"
        ):
            message = "Exhausted license window outcome is inconsistent."
            raise ValueError(message)

    @property
    def realised_window_seconds(self) -> float:
        """Return the complete monotonic license-acquisition window duration."""
        return float(self.window_ended_monotonic_seconds - self.window_started_monotonic_seconds)


def in_allocation_license_window_directory(
    campaign_run_id: str,
    batch_id: str,
    case_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return the strict append-only receipt directory for one logical case."""
    run_directory = campaign_evidence.campaign_run_directory(campaign_run_id, storage_root=storage_root)
    safe_batch = common.paths.validate_logical_name(batch_id, label="batch_id")
    safe_case = common.paths.validate_logical_name(case_id, label="case_id")
    return run_directory / _WINDOW_DIRECTORY_NAME / safe_batch / safe_case


def _checkout_summary_payload(summary: InAllocationLicenseCheckoutSummary) -> dict[str, Any]:
    """Return one bounded JSON-safe checkout summary."""
    classification = summary.classification
    excerpt = None if classification is None else _bounded_raw_excerpt(classification.raw_excerpt)[-_MAX_CHECKOUT_EXCERPT_CHARACTERS:]
    return {
        "checkout_index": summary.checkout_index,
        "started_at": summary.started_at.astimezone(timezone.utc).isoformat(),
        "ended_at": summary.ended_at.astimezone(timezone.utc).isoformat(),
        "duration_seconds": summary.duration_seconds,
        "process_exit_code": summary.process_exit_code,
        "classification": None if classification is None else classification.classification,
        "feature": None if classification is None else classification.feature,
        "error_code": None if classification is None else classification.license_code,
        "solver_progress_started": summary.solver_progress_started,
        "raw_excerpt_sha256": None if excerpt is None else hashlib.sha256(excerpt.encode()).hexdigest(),
        "raw_excerpt": excerpt,
    }


def in_allocation_license_window_result(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    hostname: str,
    window_started_at: datetime,
    window_ended_at: datetime,
    window_started_monotonic_seconds: float,
    window_ended_monotonic_seconds: float,
    checkout_summaries: tuple[InAllocationLicenseCheckoutSummary, ...],
    solver_progress_started: bool,
    outcome: str,
) -> InAllocationLicenseWindowResult:
    """Build one validated final result from injected worker lifecycle evidence."""
    policy = config.execution_values["runtime"]["temporary_license_retry"]["in_allocation_retry"]
    case_id = config.case_id(case_index)
    exhausted = outcome == "window_exhausted"
    return InAllocationLicenseWindowResult(
        campaign_run_id=common.paths.validate_logical_name(campaign_run_id, label="campaign_run_id"),
        batch_id=config.batch_id,
        case_id=case_id,
        work_unit_id=f"{config.batch_id}/{case_id}",
        scientific_config_digest=config.scientific_config_digest,
        job_id=job_id,
        hostname=hostname,
        window_started_at=window_started_at,
        window_ended_at=window_ended_at,
        window_started_monotonic_seconds=window_started_monotonic_seconds,
        window_ended_monotonic_seconds=window_ended_monotonic_seconds,
        configured_window_seconds=float(policy["maximum_window_seconds"]),
        checkout_summaries=checkout_summaries,
        solver_progress_started=solver_progress_started,
        outcome=outcome,
        reason="in_allocation_license_window_exhausted" if exhausted else "solver_progress_started",
        next_controller_retry_basis="controller_temporary_license_retry" if exhausted else None,
    )


def _window_payload(result: InAllocationLicenseWindowResult) -> dict[str, Any]:
    """Return one deterministic bounded allocation-window receipt payload."""
    summaries = result.checkout_summaries[-_MAX_RECENT_CHECKOUT_SUMMARIES:]
    features = sorted({summary.classification.feature for summary in result.checkout_summaries if summary.classification is not None})
    error_codes = sorted(
        {
            summary.classification.license_code
            for summary in result.checkout_summaries
            if summary.classification is not None and summary.classification.license_code is not None
        }
    )
    return {
        "schema_kind": _WINDOW_SCHEMA_KIND,
        "schema_version": _WINDOW_SCHEMA_VERSION,
        "campaign_run_id": result.campaign_run_id,
        "batch_id": result.batch_id,
        "case_id": result.case_id,
        "work_unit_id": result.work_unit_id,
        "scientific_config_digest": result.scientific_config_digest,
        "slurm_job_id": result.job_id,
        "hostname": result.hostname,
        "window_started_at": result.window_started_at.astimezone(timezone.utc).isoformat(),
        "window_ended_at": result.window_ended_at.astimezone(timezone.utc).isoformat(),
        "configured_window_seconds": result.configured_window_seconds,
        "realised_window_seconds": result.realised_window_seconds,
        "checkout_attempt_count": len(result.checkout_summaries),
        "checkout_capacity_failure_count": sum(summary.classification is not None for summary in result.checkout_summaries),
        "observed_features": features,
        "observed_error_codes": error_codes,
        "first_checkout_started_at": None
        if not result.checkout_summaries
        else result.checkout_summaries[0].started_at.astimezone(timezone.utc).isoformat(),
        "last_checkout_ended_at": None
        if not result.checkout_summaries
        else result.checkout_summaries[-1].ended_at.astimezone(timezone.utc).isoformat(),
        "solver_progress_started": result.solver_progress_started,
        "outcome": result.outcome,
        "reason": result.reason,
        "next_controller_retry_basis": result.next_controller_retry_basis,
        "controller_retry_increment": result.outcome == "window_exhausted",
        "recent_checkout_summaries": [_checkout_summary_payload(summary) for summary in summaries],
    }


def _validate_checkout_summary_payload(payload: object) -> dict[str, Any]:
    """Validate one bounded recent checkout summary from persisted evidence."""
    keys = {
        "checkout_index",
        "started_at",
        "ended_at",
        "duration_seconds",
        "process_exit_code",
        "classification",
        "feature",
        "error_code",
        "solver_progress_started",
        "raw_excerpt_sha256",
        "raw_excerpt",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        message = "In-allocation checkout summary is malformed."
        raise ValueError(message)
    index = payload["checkout_index"]
    duration = payload["duration_seconds"]
    exit_code = payload["process_exit_code"]
    started = _parse_timestamp(payload["started_at"], label="in-allocation checkout start")
    ended = _parse_timestamp(payload["ended_at"], label="in-allocation checkout end")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 1
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not 0.0 <= float(duration) < float("inf")
        or ended < started
        or (exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)))
        or not isinstance(payload["solver_progress_started"], bool)
    ):
        message = "In-allocation checkout timing or process evidence is malformed."
        raise ValueError(message)
    classification = payload["classification"]
    feature = payload["feature"]
    error_code = payload["error_code"]
    excerpt_digest = payload["raw_excerpt_sha256"]
    excerpt = payload["raw_excerpt"]
    if classification is None:
        if any(value is not None for value in (feature, error_code, excerpt_digest, excerpt)):
            message = "Unclassified in-allocation checkout retains inconsistent license evidence."
            raise ValueError(message)
    elif (
        classification != TEMPORARY_LICENSE_CAPACITY
        or not isinstance(feature, str)
        or not feature
        or (error_code is not None and not isinstance(error_code, str))
        or not isinstance(excerpt, str)
        or not excerpt
        or len(excerpt) > _MAX_CHECKOUT_EXCERPT_CHARACTERS
        or not isinstance(excerpt_digest, str)
        or _SCIENTIFIC_DIGEST_PATTERN.fullmatch(excerpt_digest) is None
        or hashlib.sha256(excerpt.encode()).hexdigest() != excerpt_digest
        or payload["solver_progress_started"] is True
    ):
        message = "Temporary-capacity checkout summary is malformed."
        raise ValueError(message)
    return payload


def load_in_allocation_license_window(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load one immutable allocation-window receipt for its exact Slurm job."""
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "In-allocation license window lookup requires one numeric Slurm job ID."
        raise ValueError(message)
    case_id = config.case_id(case_index)
    directory = in_allocation_license_window_directory(
        campaign_run_id,
        config.batch_id,
        case_id,
        storage_root=storage_root,
    )
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        message = f"In-allocation license window directory is unsafe: {directory}"
        raise ValueError(message)
    path = directory / f"{job_id}.json"
    entries = tuple(directory.iterdir())
    if any(
        entry.is_symlink() or not entry.is_file() or _JOB_ID_PATTERN.fullmatch(entry.stem) is None or entry.suffix != ".json" for entry in entries
    ):
        message = f"In-allocation license window directory contains unexpected entries: {directory}"
        raise ValueError(message)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load in-allocation license window evidence: {path}"
        raise ValueError(message) from error
    required = {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "work_unit_id",
        "scientific_config_digest",
        "slurm_job_id",
        "hostname",
        "window_started_at",
        "window_ended_at",
        "configured_window_seconds",
        "realised_window_seconds",
        "checkout_attempt_count",
        "checkout_capacity_failure_count",
        "observed_features",
        "observed_error_codes",
        "first_checkout_started_at",
        "last_checkout_ended_at",
        "solver_progress_started",
        "outcome",
        "reason",
        "next_controller_retry_basis",
        "controller_retry_increment",
        "recent_checkout_summaries",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        message = f"In-allocation license window evidence is malformed for {config.batch_id}/{case_id}."
        raise ValueError(message)
    if (
        payload["schema_kind"] != _WINDOW_SCHEMA_KIND
        or payload["schema_version"] != _WINDOW_SCHEMA_VERSION
        or payload["campaign_run_id"] != campaign_run_id
        or payload["batch_id"] != config.batch_id
        or payload["case_id"] != case_id
        or payload["work_unit_id"] != f"{config.batch_id}/{case_id}"
        or payload["scientific_config_digest"] != config.scientific_config_digest
        or payload["slurm_job_id"] != job_id
        or not isinstance(payload["hostname"], str)
        or not payload["hostname"]
        or payload["outcome"] not in _WINDOW_OUTCOMES
        or not isinstance(payload["solver_progress_started"], bool)
        or not isinstance(payload["controller_retry_increment"], bool)
    ):
        message = f"In-allocation license window identity is malformed for {config.batch_id}/{case_id}."
        raise ValueError(message)
    start = _parse_timestamp(payload["window_started_at"], label="in-allocation license window start")
    end = _parse_timestamp(payload["window_ended_at"], label="in-allocation license window end")
    if end < start:
        message = f"In-allocation license window timestamps are inconsistent for {case_id}."
        raise ValueError(message)
    numeric_keys = ("configured_window_seconds", "realised_window_seconds")
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], (int, float)) or float(payload[key]) < 0.0 for key in numeric_keys):
        message = f"In-allocation license window durations are malformed for {case_id}."
        raise ValueError(message)
    expected_window = config.execution_values["runtime"]["temporary_license_retry"]["in_allocation_retry"]["maximum_window_seconds"]
    if float(payload["configured_window_seconds"]) != float(expected_window):
        message = f"In-allocation license window policy is inconsistent for {case_id}."
        raise ValueError(message)
    summaries = payload["recent_checkout_summaries"]
    attempts = payload["checkout_attempt_count"]
    failures = payload["checkout_capacity_failure_count"]
    features = payload["observed_features"]
    error_codes = payload["observed_error_codes"]
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or isinstance(failures, bool)
        or not isinstance(failures, int)
        or not 0 <= failures <= attempts
        or not isinstance(summaries, list)
        or len(summaries) != min(attempts, _MAX_RECENT_CHECKOUT_SUMMARIES)
        or not isinstance(features, list)
        or features != sorted(set(features))
        or not all(isinstance(value, str) and value for value in features)
        or not isinstance(error_codes, list)
        or error_codes != sorted(set(error_codes))
        or not all(isinstance(value, str) and value for value in error_codes)
    ):
        message = f"In-allocation checkout counts or aggregates are malformed for {case_id}."
        raise ValueError(message)
    validated_summaries = [_validate_checkout_summary_payload(summary) for summary in summaries]
    expected_indexes = list(range(attempts - len(validated_summaries) + 1, attempts + 1))
    if [summary["checkout_index"] for summary in validated_summaries] != expected_indexes:
        message = f"In-allocation recent checkout ordering is malformed for {case_id}."
        raise ValueError(message)
    first_checkout = _parse_timestamp(payload["first_checkout_started_at"], label="first in-allocation checkout start")
    last_checkout = _parse_timestamp(payload["last_checkout_ended_at"], label="last in-allocation checkout end")
    if last_checkout < first_checkout or first_checkout < start or last_checkout > end:
        message = f"In-allocation checkout timestamps escape their window for {case_id}."
        raise ValueError(message)
    exhausted = payload["outcome"] == "window_exhausted"
    if (
        exhausted != payload["controller_retry_increment"]
        or (
            exhausted
            and (
                failures != attempts
                or any(summary["classification"] != TEMPORARY_LICENSE_CAPACITY for summary in validated_summaries)
                or payload["solver_progress_started"]
                or payload["reason"] != "in_allocation_license_window_exhausted"
                or payload["next_controller_retry_basis"] != "controller_temporary_license_retry"
            )
        )
        or (
            not exhausted
            and (
                failures >= attempts
                or validated_summaries[-1]["solver_progress_started"] is not True
                or not payload["solver_progress_started"]
                or payload["reason"] != "solver_progress_started"
                or payload["next_controller_retry_basis"] is not None
            )
        )
    ):
        message = f"In-allocation license window outcome is inconsistent for {case_id}."
        raise ValueError(message)
    return payload


def record_in_allocation_license_window(
    config: config_contract.GenerationConfig,
    case_index: int,
    result: InAllocationLicenseWindowResult,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Append one immutable per-numeric-job allocation-window receipt."""
    case_id = config.case_id(case_index)
    if result.batch_id != config.batch_id or result.case_id != case_id or result.scientific_config_digest != config.scientific_config_digest:
        message = "In-allocation license window result does not bind the configured case."
        raise ValueError(message)
    directory = in_allocation_license_window_directory(
        result.campaign_run_id,
        result.batch_id,
        result.case_id,
        storage_root=storage_root,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        message = f"In-allocation license window directory is unsafe: {directory}"
        raise ValueError(message)
    path = directory / f"{result.job_id}.json"
    payload = _window_payload(result)
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        message = f"In-allocation license window receipt already exists for Slurm job {result.job_id}."
        raise FileExistsError(message) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


_STATUS_RECOVERY_SCHEMA_KIND: Final = "generation_in_allocation_status_artifact_recovery"
_STATUS_RECOVERY_SCHEMA_VERSION: Final = 1
_STATUS_RECOVERY_DIRECTORY_NAME: Final = "license_status_recoveries"
_STATUS_RECOVERY_FILENAME_PATTERN: Final = re.compile(r"checkout_(?P<index>[0-9]{4,})[.]json")
_STATUS_RECOVERY_STATES: Final = frozenset({"pending", "complete"})
_MAX_STATUS_RECOVERY_EXCERPT_BYTES: Final = 160
_STATUS_RECOVERY_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "work_unit_id",
        "scientific_config_digest",
        "slurm_job_id",
        "checkout_index",
        "hostname",
        "process_id",
        "process_exit_code",
        "checkout_started_at",
        "checkout_ended_at",
        "prelaunch_checked_at",
        "prelaunch_path_existed",
        "status_relative_path",
        "status_content_class",
        "status_state",
        "status_timestamp_milliseconds",
        "status_size_bytes",
        "status_sha256",
        "status_excerpt",
        "status_file_device",
        "status_file_inode",
        "status_file_user_id",
        "status_file_modified_nanoseconds",
        "temporary_capacity_classification",
        "feature",
        "error_code",
        "license_evidence_sha256",
        "solver_progress_started",
        "required_exports_exist",
        "scientific_result_exists",
        "cleanup_state",
        "recorded_at",
        "cleanup_completed_at",
    }
)


def in_allocation_status_recovery_directory(
    campaign_run_id: str,
    batch_id: str,
    case_id: str,
    job_id: str,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one exact per-job status-artifact recovery directory."""
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Status-artifact recovery requires one numeric Slurm job ID."
        raise ValueError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        campaign_run_id,
        storage_root=storage_root,
    )
    safe_batch = common.paths.validate_logical_name(batch_id, label="batch_id")
    safe_case = common.paths.validate_logical_name(case_id, label="case_id")
    return run_directory / _STATUS_RECOVERY_DIRECTORY_NAME / safe_batch / safe_case / job_id


def _status_recovery_payload(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    checkout_started_at: datetime,
    checkout_ended_at: datetime,
    hostname: str,
    artifact: Any,
    classification: TemporaryLicenseCapacityClassification,
    cleanup_state: str,
    recorded_at: str,
    cleanup_completed_at: str | None,
) -> dict[str, Any]:
    """Return one exact bounded capacity status-artifact recovery receipt."""
    case_id = config.case_id(case_index)
    classification_identity = {
        "classification": classification.classification,
        "feature": classification.feature,
        "error_code": classification.license_code,
        "matched_signatures": list(classification.matched_signatures),
    }
    return {
        "schema_kind": _STATUS_RECOVERY_SCHEMA_KIND,
        "schema_version": _STATUS_RECOVERY_SCHEMA_VERSION,
        "campaign_run_id": common.paths.validate_logical_name(campaign_run_id, label="campaign_run_id"),
        "batch_id": config.batch_id,
        "case_id": case_id,
        "work_unit_id": f"{config.batch_id}/{case_id}",
        "scientific_config_digest": config.scientific_config_digest,
        "slurm_job_id": job_id,
        "checkout_index": artifact.prelaunch.checkout_index,
        "hostname": hostname,
        "process_id": artifact.process_id,
        "process_exit_code": artifact.process_exit_code,
        "checkout_started_at": checkout_started_at.astimezone(timezone.utc).isoformat(),
        "checkout_ended_at": checkout_ended_at.astimezone(timezone.utc).isoformat(),
        "prelaunch_checked_at": artifact.prelaunch.checked_at,
        "prelaunch_path_existed": False,
        "status_relative_path": "solved.mph.status",
        "status_content_class": "comsol_batch_status_timestamp_and_state",
        "status_state": artifact.status_state,
        "status_timestamp_milliseconds": artifact.status_timestamp_milliseconds,
        "status_size_bytes": artifact.file_size_bytes,
        "status_sha256": artifact.content_sha256,
        "status_excerpt": artifact.content_excerpt,
        "status_file_device": artifact.file_device,
        "status_file_inode": artifact.file_inode,
        "status_file_user_id": artifact.file_user_id,
        "status_file_modified_nanoseconds": artifact.file_modified_nanoseconds,
        "temporary_capacity_classification": classification.classification,
        "feature": classification.feature,
        "error_code": classification.license_code,
        "license_evidence_sha256": common.serialization.canonical_json_sha256(classification_identity),
        "solver_progress_started": False,
        "required_exports_exist": False,
        "scientific_result_exists": False,
        "cleanup_state": cleanup_state,
        "recorded_at": recorded_at,
        "cleanup_completed_at": cleanup_completed_at,
    }


def _validate_status_recovery_payload(
    payload: object,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
    campaign_run_id: str,
    job_id: str,
    path: Path,
) -> dict[str, Any]:
    """Validate one exact schema-version-1 status-artifact recovery receipt."""
    case_id = config.case_id(case_index)
    if not isinstance(payload, dict) or set(payload) != _STATUS_RECOVERY_KEYS:
        message = f"Status-artifact recovery receipt is malformed: {path}"
        raise ValueError(message)
    index = payload.get("checkout_index")
    integer_fields = (
        "process_id",
        "status_timestamp_milliseconds",
        "status_size_bytes",
        "status_file_device",
        "status_file_inode",
        "status_file_user_id",
        "status_file_modified_nanoseconds",
    )
    if (
        payload.get("schema_kind") != _STATUS_RECOVERY_SCHEMA_KIND
        or payload.get("schema_version") != _STATUS_RECOVERY_SCHEMA_VERSION
        or payload.get("campaign_run_id") != campaign_run_id
        or payload.get("batch_id") != config.batch_id
        or payload.get("case_id") != case_id
        or payload.get("work_unit_id") != f"{config.batch_id}/{case_id}"
        or payload.get("scientific_config_digest") != config.scientific_config_digest
        or payload.get("slurm_job_id") != job_id
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 1
        or not isinstance(payload.get("hostname"), str)
        or not payload["hostname"]
        or any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) or int(payload[key]) < 0 for key in integer_fields)
        or isinstance(payload.get("process_exit_code"), bool)
        or not isinstance(payload.get("process_exit_code"), int)
        or payload.get("prelaunch_path_existed") is not False
        or payload.get("status_relative_path") != "solved.mph.status"
        or payload.get("status_content_class") != "comsol_batch_status_timestamp_and_state"
        or payload.get("status_state") not in {"Running", "Done", "Failed", "Error"}
        or not isinstance(payload.get("status_excerpt"), str)
        or len(payload["status_excerpt"].encode("utf-8")) > _MAX_STATUS_RECOVERY_EXCERPT_BYTES
        or not isinstance(payload.get("status_sha256"), str)
        or _SCIENTIFIC_DIGEST_PATTERN.fullmatch(payload["status_sha256"]) is None
        or payload.get("temporary_capacity_classification") != TEMPORARY_LICENSE_CAPACITY
        or not isinstance(payload.get("feature"), str)
        or not payload["feature"]
        or (payload.get("error_code") is not None and not isinstance(payload["error_code"], str))
        or not isinstance(payload.get("license_evidence_sha256"), str)
        or _SCIENTIFIC_DIGEST_PATTERN.fullmatch(payload["license_evidence_sha256"]) is None
        or payload.get("solver_progress_started") is not False
        or payload.get("required_exports_exist") is not False
        or payload.get("scientific_result_exists") is not False
        or payload.get("cleanup_state") not in _STATUS_RECOVERY_STATES
    ):
        message = f"Status-artifact recovery identity is malformed: {path}"
        raise ValueError(message)
    started = _parse_timestamp(payload.get("checkout_started_at"), label="status-recovery checkout start")
    ended = _parse_timestamp(payload.get("checkout_ended_at"), label="status-recovery checkout end")
    _parse_timestamp(payload.get("prelaunch_checked_at"), label="status-recovery prelaunch check")
    _parse_timestamp(payload.get("recorded_at"), label="status-recovery record")
    completed = payload.get("cleanup_completed_at")
    if (
        ended < started
        or (payload["cleanup_state"] == "pending" and completed is not None)
        or (payload["cleanup_state"] == "complete" and completed is None)
    ):
        message = f"Status-artifact recovery lifecycle is malformed: {path}"
        raise ValueError(message)
    if completed is not None:
        _parse_timestamp(completed, label="status-recovery completion")
    expected_name = f"checkout_{index:04d}.json"
    if path.name != expected_name:
        message = f"Status-artifact recovery filename is malformed: {path}"
        raise ValueError(message)
    return payload


def record_in_allocation_status_artifact_recovery(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    checkout_started_at: datetime,
    checkout_ended_at: datetime,
    hostname: str,
    artifact: Any,
    classification: TemporaryLicenseCapacityClassification,
    cleanup_state: str,
    storage_root: Path | str | None,
) -> Path:
    """Persist pending evidence before cleanup, then complete the same exact receipt."""
    for label, value in (
        ("checkout_started_at", checkout_started_at),
        ("checkout_ended_at", checkout_ended_at),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            message = f"Status-artifact recovery {label} must be timezone-aware."
            raise ValueError(message)
    if cleanup_state not in _STATUS_RECOVERY_STATES:
        message = "Status-artifact recovery cleanup state must be pending or complete."
        raise ValueError(message)
    directory = in_allocation_status_recovery_directory(
        campaign_run_id,
        config.batch_id,
        config.case_id(case_index),
        job_id,
        storage_root=storage_root,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        message = f"Status-artifact recovery directory is unsafe: {directory}"
        raise ValueError(message)
    path = directory / f"checkout_{artifact.prelaunch.checkout_index:04d}.json"
    existing: dict[str, Any] | None = None
    if path.exists():
        if path.is_symlink() or not path.is_file():
            message = f"Status-artifact recovery receipt is unsafe: {path}"
            raise ValueError(message)
        try:
            raw_existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Could not load status-artifact recovery receipt: {path}"
            raise ValueError(message) from error
        existing = _validate_status_recovery_payload(
            raw_existing,
            config=config,
            case_index=case_index,
            campaign_run_id=campaign_run_id,
            job_id=job_id,
            path=path,
        )
    if existing is None and cleanup_state == "complete":
        message = "Status-artifact cleanup completion requires its prior pending receipt."
        raise RuntimeError(message)
    recorded_at = datetime.now(timezone.utc).isoformat() if existing is None else str(existing["recorded_at"])
    payload = _status_recovery_payload(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        job_id=job_id,
        checkout_started_at=checkout_started_at,
        checkout_ended_at=checkout_ended_at,
        hostname=hostname,
        artifact=artifact,
        classification=classification,
        cleanup_state=cleanup_state,
        recorded_at=recorded_at,
        cleanup_completed_at=(datetime.now(timezone.utc).isoformat() if cleanup_state == "complete" else None),
    )
    if existing is not None:
        comparable = dict(payload)
        comparable["cleanup_state"] = existing["cleanup_state"]
        comparable["cleanup_completed_at"] = existing["cleanup_completed_at"]
        if comparable != existing:
            message = f"Status-artifact recovery receipt conflicts with its exact checkout: {path}"
            raise FileExistsError(message)
        if existing["cleanup_state"] == "complete" or cleanup_state == "pending":
            return path
    common.serialization.atomic_write_json(path, payload)
    _validate_status_recovery_payload(
        payload,
        config=config,
        case_index=case_index,
        campaign_run_id=campaign_run_id,
        job_id=job_id,
        path=path,
    )
    return path


def load_in_allocation_status_artifact_recoveries(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], ...]:
    """Load every exact status-artifact recovery receipt for one Slurm job."""
    directory = in_allocation_status_recovery_directory(
        campaign_run_id,
        config.batch_id,
        config.case_id(case_index),
        job_id,
        storage_root=storage_root,
    )
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        message = f"Status-artifact recovery directory is unsafe: {directory}"
        raise ValueError(message)
    records: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        temporary_destination = common.serialization.atomic_write_temporary_destination(path)
        if (
            temporary_destination is not None
            and _STATUS_RECOVERY_FILENAME_PATTERN.fullmatch(temporary_destination.name) is not None
            and _owned_atomic_temporary_entry(path, destination=temporary_destination)
        ):
            continue
        match = _STATUS_RECOVERY_FILENAME_PATTERN.fullmatch(path.name)
        if path.is_symlink() or not path.is_file() or match is None:
            message = f"Status-artifact recovery directory contains an unsafe entry: {path}"
            raise ValueError(message)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Could not load status-artifact recovery receipt: {path}"
            raise ValueError(message) from error
        record = _validate_status_recovery_payload(
            payload,
            config=config,
            case_index=case_index,
            campaign_run_id=campaign_run_id,
            job_id=job_id,
            path=path,
        )
        if int(match.group("index")) != int(record["checkout_index"]):
            message = f"Status-artifact recovery checkout identity is malformed: {path}"
            raise ValueError(message)
        records.append(record)
    return tuple(records)
