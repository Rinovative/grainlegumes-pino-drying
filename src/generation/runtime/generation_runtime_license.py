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

import json
import os
import re
import socket
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
    """Conservative classification extracted from captured COMSOL text."""

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
    if not entries:
        return None
    wait_path = directory / _WAIT_FILENAME
    if len(entries) != 1 or entries[0] != wait_path or wait_path.is_symlink() or not wait_path.is_file():
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
