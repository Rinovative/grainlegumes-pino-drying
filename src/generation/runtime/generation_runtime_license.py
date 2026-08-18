"""
generation_runtime_license.py

Classify and persist retry evidence for temporary COMSOL license capacity.
Responsibilities:
  - Recognize conservative floating-license capacity signatures in captured logs
  - Derive capped exponential retry delays from resolved execution policy
  - Persist and validate immutable per-case license-attempt evidence
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
ATTEMPT_SCHEMA_KIND: Final = "generation_temporary_license_capacity_attempt"
ATTEMPT_SCHEMA_VERSION: Final = 1
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
_ATTEMPT_NAME_PATTERN: Final = re.compile(r"attempt-(?P<index>[0-9]{4,})\.json")
_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_ATTEMPT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "scientific_config_digest",
        "attempt_index",
        "slurm_job_id",
        "timestamp",
        "hostname",
        "classification",
        "detected_feature",
        "detected_license_code",
        "matched_signatures",
        "delay_before_next_attempt_seconds",
        "cumulative_wait_seconds",
        "retry_budget_remaining",
        "next_eligible_at",
    }
)


@dataclass(frozen=True, slots=True)
class TemporaryLicenseCapacityClassification:
    """Conservative classification extracted from captured COMSOL text."""

    classification: str
    feature: str
    license_code: str | None
    matched_signatures: tuple[str, ...]


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
    matched: list[str] = ["could_not_obtain_license"]
    users_reached = _USERS_REACHED_PATTERN.search(captured_text) is not None
    license_minus_four = _LICENSE_MINUS_FOUR_PATTERN.search(captured_text) is not None
    flexnet_match = _FLEXNET_MINUS_FOUR_PATTERN.search(captured_text)
    if users_reached:
        matched.append("licensed_users_reached")
    if license_minus_four:
        matched.append("license_error_minus_four")
    if flexnet_match is not None:
        matched.append("flexnet_error_minus_four")
    if not (users_reached or license_minus_four or flexnet_match is not None):
        return None
    license_code = str(flexnet_match.group("code")) if flexnet_match is not None else "-4" if license_minus_four else None
    return TemporaryLicenseCapacityClassification(
        classification=TEMPORARY_LICENSE_CAPACITY,
        feature=feature_match.group("feature").strip(),
        license_code=license_code,
        matched_signatures=tuple(matched),
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


def temporary_license_attempt_directory(
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


def _validate_attempt(
    payload: object,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
    campaign_run_id: str,
    expected_index: int,
) -> dict[str, Any]:
    """Validate one immutable retry receipt against exact current identities."""
    case_id = config.case_id(case_index)
    if (
        not isinstance(payload, dict)
        or set(payload) != _ATTEMPT_KEYS
        or payload.get("schema_kind") != ATTEMPT_SCHEMA_KIND
        or payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or payload.get("campaign_run_id") != campaign_run_id
        or payload.get("batch_id") != config.batch_id
        or payload.get("case_id") != case_id
        or payload.get("scientific_config_digest") != config.scientific_config_digest
        or payload.get("attempt_index") != expected_index
        or payload.get("classification") != TEMPORARY_LICENSE_CAPACITY
        or not isinstance(payload.get("hostname"), str)
        or not payload["hostname"]
        or not isinstance(payload.get("detected_feature"), str)
        or not payload["detected_feature"]
        or (payload.get("detected_license_code") is not None and not isinstance(payload.get("detected_license_code"), str))
        or not isinstance(payload.get("matched_signatures"), list)
        or not payload["matched_signatures"]
        or not all(isinstance(value, str) and value for value in payload["matched_signatures"])
        or not isinstance(payload.get("retry_budget_remaining"), bool)
    ):
        message = f"Temporary-license retry evidence is malformed for {config.batch_id}/{case_id}."
        raise ValueError(message)
    job_id = payload.get("slurm_job_id")
    if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = f"Temporary-license retry evidence has a malformed Slurm job ID for {case_id}."
        raise ValueError(message)
    timestamp = _parse_timestamp(
        payload.get("timestamp"),
        label="temporary-license retry timestamp",
    )
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
    remaining = bool(payload["retry_budget_remaining"])
    next_eligible = payload.get("next_eligible_at")
    if remaining:
        eligible_time = _parse_timestamp(
            next_eligible,
            label="temporary-license next eligibility",
        )
        if float(delay) <= 0.0 or eligible_time != timestamp + timedelta(seconds=float(delay)):
            message = f"Temporary-license next eligibility is inconsistent for {case_id}."
            raise ValueError(message)
    elif next_eligible is not None or float(delay) != 0.0:
        message = f"Exhausted temporary-license retry evidence is inconsistent for {case_id}."
        raise ValueError(message)
    return payload


def load_temporary_license_attempts(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], ...]:
    """Load and validate the complete immutable retry history for one case."""
    directory = temporary_license_attempt_directory(
        campaign_run_id,
        config.batch_id,
        config.case_id(case_index),
        storage_root=storage_root,
    )
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        message = f"Temporary-license retry evidence directory is unsafe: {directory}"
        raise ValueError(message)
    entries = sorted(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() or _ATTEMPT_NAME_PATTERN.fullmatch(path.name) is None for path in entries):
        message = f"Temporary-license retry evidence directory contains unexpected entries: {directory}"
        raise ValueError(message)
    attempts: list[dict[str, Any]] = []
    prior_cumulative = 0.0
    seen_jobs: set[str] = set()
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    for expected_index, path in enumerate(entries, start=1):
        match = _ATTEMPT_NAME_PATTERN.fullmatch(path.name)
        if match is None or int(match.group("index")) != expected_index:
            message = f"Temporary-license retry attempt ordering is invalid: {directory}"
            raise ValueError(message)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Could not load temporary-license retry evidence: {path}"
            raise ValueError(message) from error
        attempt = _validate_attempt(
            payload,
            config=config,
            case_index=case_index,
            campaign_run_id=campaign_run_id,
            expected_index=expected_index,
        )
        job_id = str(attempt["slurm_job_id"])
        if job_id in seen_jobs:
            message = f"Temporary-license retry history repeats Slurm job {job_id}."
            raise ValueError(message)
        seen_jobs.add(job_id)
        expected_delay = bounded_retry_delay_seconds(
            policy,
            attempt_index=expected_index,
            cumulative_wait_seconds=prior_cumulative,
        )
        observed_delay = float(attempt["delay_before_next_attempt_seconds"])
        observed_cumulative = float(attempt["cumulative_wait_seconds"])
        if (
            observed_delay != expected_delay
            or observed_cumulative != prior_cumulative + expected_delay
            or bool(attempt["retry_budget_remaining"]) != (expected_delay > 0.0)
        ):
            message = f"Temporary-license retry budget chain is inconsistent: {path}"
            raise ValueError(message)
        prior_cumulative = observed_cumulative
        attempts.append(attempt)
    return tuple(attempts)


def record_temporary_license_capacity_attempt(
    config: config_contract.GenerationConfig,
    case_index: int,
    error: TemporaryLicenseCapacityError,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Persist one compact retry receipt before case scratch is reclaimed."""
    campaign_run_id = common.paths.validate_logical_name(
        os.environ.get("GENERATION_CAMPAIGN_RUN_ID"),
        label="campaign_run_id",
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Temporary-license retry evidence requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    attempts = load_temporary_license_attempts(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    if any(attempt["slurm_job_id"] == job_id for attempt in attempts):
        message = f"Temporary-license retry evidence already exists for Slurm job {job_id}."
        raise FileExistsError(message)
    attempt_index = len(attempts) + 1
    prior_cumulative = 0.0 if not attempts else float(attempts[-1]["cumulative_wait_seconds"])
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    delay = bounded_retry_delay_seconds(
        policy,
        attempt_index=attempt_index,
        cumulative_wait_seconds=prior_cumulative,
    )
    cumulative = prior_cumulative + delay
    timestamp = datetime.now(timezone.utc)
    retry_remaining = delay > 0.0
    payload = {
        "schema_kind": ATTEMPT_SCHEMA_KIND,
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "campaign_run_id": campaign_run_id,
        "batch_id": config.batch_id,
        "case_id": config.case_id(case_index),
        "scientific_config_digest": config.scientific_config_digest,
        "attempt_index": attempt_index,
        "slurm_job_id": job_id,
        "timestamp": timestamp.isoformat(),
        "hostname": socket.gethostname(),
        "classification": error.evidence.classification,
        "detected_feature": error.evidence.feature,
        "detected_license_code": error.evidence.license_code,
        "matched_signatures": list(error.evidence.matched_signatures),
        "delay_before_next_attempt_seconds": delay,
        "cumulative_wait_seconds": cumulative,
        "retry_budget_remaining": retry_remaining,
        "next_eligible_at": ((timestamp + timedelta(seconds=delay)).isoformat() if retry_remaining else None),
    }
    directory = temporary_license_attempt_directory(
        campaign_run_id,
        config.batch_id,
        config.case_id(case_index),
        storage_root=storage_root,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"attempt-{attempt_index:04d}.json"
    if path.exists() or path.is_symlink():
        message = f"Temporary-license retry evidence already exists: {path}"
        raise FileExistsError(message)
    common.serialization.atomic_write_json(path, payload)
    load_temporary_license_attempts(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    return path


def latest_attempt_for_job(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    job_id: str,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Return retry evidence only when it belongs to the latest submitted job."""
    attempts = load_temporary_license_attempts(
        config,
        case_index,
        campaign_run_id=campaign_run_id,
        storage_root=storage_root,
    )
    matches = [attempt for attempt in attempts if attempt["slurm_job_id"] == job_id]
    if len(matches) > 1:
        message = f"Temporary-license retry history duplicates Slurm job {job_id}."
        raise ValueError(message)
    if not matches:
        return None
    return {
        **matches[0],
        "first_blocked_at": attempts[0]["timestamp"],
    }


def retry_attempt_is_eligible(
    attempt: Mapping[str, Any],
    *,
    at: datetime | None = None,
) -> bool:
    """Return whether a retryable attempt has reached its eligibility time."""
    if not bool(attempt["retry_budget_remaining"]):
        return False
    eligible = _parse_timestamp(
        attempt["next_eligible_at"],
        label="temporary-license next eligibility",
    )
    current = datetime.now(timezone.utc) if at is None else at
    if current.tzinfo is None or current.utcoffset() is None:
        message = "Retry eligibility comparison requires a timezone-aware time."
        raise ValueError(message)
    return current.astimezone(timezone.utc) >= eligible
