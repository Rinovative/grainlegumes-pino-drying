"""Own the write-once member records of one Generation field test: attempts, outcomes, seals, and locks.

Responsibilities:
  - Locate member, attempt, node-job, analysis, and lock paths below one field-test directory, refusing symbolic links
  - Publish attempt submissions and outcomes once, and read them back as one attempt history
  - Decide from that history whether a member's disposition is final, link a success's case.h5, and seal it exactly once
  - Return the members a verdict judges after verifying each base replacement against its failed gate's seal
  - Hold the member lock across a whole attempt and the dispatch lock across submit, resume, and cancel

Design principles:
  - Every record is published once through a hard link and never replaced; a second writer must hold identical content
  - The seal is the only terminal state; a failure is sealed when it reproduces or the planned attempt limit is reached
  - A member case.h5 exists only behind a committed succeeded outcome: the attempt keeps its output until then
  - No replacement exists: a member keeps its intervention and base for every attempt

This module does NOT:
  - Run cases, query the scheduler, or choose which members are dispatched

Notes:
  Layout below STORAGE_ROOT/04_field_tests/TASK/TEST_ID/: members/MEMBER/{seal.json, case.h5, attempts/N/{submission.json,
  outcome.json, case.h5, bundle/, runtime/}}, nodes/TOKEN/ (node directory, job.json, and job.recovered.json), analysis/,
  locks/MEMBER.lock and locks/dispatch.lock; no component below the test directory may be a symbolic link. A submission
  holds node_token, ticket_seq, step_label, submitted_at, and its binding roster_version, roster_sha256 and member_sha256.
  An outcome's classification is a disposition or not_started (a ticket that never ran), which does not count as an
  attempt; its notes list runtime evidence a step could not copy into the attempt (null before execution). Settling a
  succeeded outcome hard-links its attempt's case.h5 into the member directory and verifies the
  outcome's case_h5_sha256 before the seal. Seal fields beyond the interface: schema_kind and input_comparison
  {relative_tolerance, max_abs_diff per input array, touched (arrays the intervention changed)}; a gate's inputs_equal
  means reproduced within that tolerance, and its gate record adds output_divergences (null when no output was compared).
  Fields that need a generated bundle or a node (bundle_sha256, the two acceptance attempts, cpu_model, input_comparison)
  are null when the member was sealed without one.

"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from src import common

from . import generation_field_tests_roster as roster_service

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

SEAL_SCHEMA_KIND: Final = "field_test_member_seal"
OUTCOME_SCHEMA_KIND: Final = "field_test_attempt_outcome"
NOT_STARTED: Final = "not_started"
CLASSIFICATIONS: Final = (*roster_service.DISPOSITIONS, NOT_STARTED)
# Dispositions a single attempt settles; a solver failure or infrastructure stop is retried within the planned limit.
_FINAL_AT_ONCE: Final = frozenset({"succeeded", "gate_blocked", "binding_refused", "infeasible"})
_REPRODUCED_SOLVER_FAILURES: Final = 2
_OUTCOME_KEYS: Final = frozenset(
    {
        "schema_kind",
        "member_id",
        "attempt",
        "classification",
        "failure",
        "bundle_sha256",
        "spatial_support_attempt",
        "schedule_acceptance_attempt",
        "cpu_model",
        "host",
        "slurm_job_id",
        "slurm_step_id",
        "started_at",
        "finished_at",
        "case_h5_sha256",
        "gate",
        "input_comparison",
        "notes",
        "recorded_by",
    }
)


def utc_now() -> str:
    """Return the current UTC time in ISO form."""
    return datetime.now(UTC).isoformat()


def record_path(directory: Path, *parts: str) -> Path:
    """
    Return one path below a field-test directory after checking every existing component with lstat.

    Raises
    ------
    ValueError
        If a component is not a logical name or is a symbolic link, which could redirect a record out of the field test.

    """
    path = directory
    for part in parts:
        path = path / common.paths.validate_logical_name(part, label="field-test record component")
        if path.is_symlink():
            message = f"Field-test record path component is a symbolic link: {path}"
            raise ValueError(message)
    return path


def member_directory(directory: Path, member_id: str) -> Path:
    """Return the record directory of one member."""
    return record_path(directory, "members", member_id)


def attempt_directory(directory: Path, member_id: str, attempt: int) -> Path:
    """Return the evidence directory of one numbered attempt."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        message = f"Attempt number must be a positive integer, got {attempt!r}."
        raise ValueError(message)
    return record_path(directory, "members", member_id, "attempts", str(attempt))


def node_directory(directory: Path, token: str) -> Path:
    """Return the node directory of one field-test node job."""
    return record_path(directory, "nodes", token)


@contextmanager
def member_lock(directory: Path, member_id: str) -> Iterator[Path]:
    """Hold the member's exclusive lock without waiting; a held lock means another process owns the member now."""
    with common.locking.exclusive_file_lock(record_path(directory, "locks", f"{member_id}.lock"), blocking=False) as path:
        yield path


@contextmanager
def dispatch_lock(directory: Path) -> Iterator[Path]:
    """Serialize submit, resume, and cancel of one field test."""
    with common.locking.exclusive_file_lock(record_path(directory, "locks", "dispatch.lock"), blocking=True) as path:
        yield path


def _read_object(path: Path) -> dict[str, Any] | None:
    """Return one JSON object, or None while it does not exist."""
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        message = f"Field-test record is not a regular file: {path}"
        raise ValueError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Field-test record must hold a JSON object: {path}"
        raise TypeError(message)
    return payload


def read_seal(directory: Path, member_id: str) -> dict[str, Any] | None:
    """Return one member's seal, or None while it is unsealed."""
    seal = _read_object(member_directory(directory, member_id) / "seal.json")
    if seal is not None and (
        seal.get("schema_kind") != SEAL_SCHEMA_KIND
        or seal.get("member_id") != member_id
        or seal.get("disposition") not in roster_service.DISPOSITIONS
    ):
        message = f"Seal of member {member_id!r} is malformed."
        raise ValueError(message)
    return seal


def gate_succeeded(seal: Mapping[str, Any]) -> bool:
    """Return whether one gate seal records success with base inputs reproduced within the last-digit tolerance."""
    return seal["disposition"] == "succeeded" and bool((seal.get("gate") or {}).get("inputs_equal"))


def attempt_numbers(directory: Path, member_id: str) -> tuple[int, ...]:
    """Return the attempt numbers of one member in ascending order."""
    root = record_path(directory, "members", member_id, "attempts")
    if not root.is_dir():
        return ()
    return tuple(sorted(int(path.name) for path in root.iterdir() if path.name.isdigit()))


def read_submission(directory: Path, member_id: str, attempt: int) -> dict[str, Any] | None:
    """Return the submission identity of one attempt, or None."""
    return _read_object(attempt_directory(directory, member_id, attempt) / "submission.json")


def read_outcome(directory: Path, member_id: str, attempt: int) -> dict[str, Any] | None:
    """Return the outcome of one attempt, or None while the attempt has not ended."""
    outcome = _read_object(attempt_directory(directory, member_id, attempt) / "outcome.json")
    if outcome is not None and (set(outcome) != _OUTCOME_KEYS or outcome["classification"] not in CLASSIFICATIONS):
        message = f"Outcome of {member_id!r} attempt {attempt} is malformed."
        raise ValueError(message)
    return outcome


def write_submission(directory: Path, member_id: str, attempt: int, payload: Mapping[str, Any]) -> None:
    """Publish the submission identity of one attempt once, before its node job is submitted."""
    roster_service.publish_once(attempt_directory(directory, member_id, attempt) / "submission.json", payload)


def new_outcome(member_id: str, attempt: int, classification: str, *, recorded_by: str, **fields: Any) -> dict[str, Any]:
    """Return one complete outcome record; omitted evidence fields are null."""
    outcome: dict[str, Any] = dict.fromkeys(_OUTCOME_KEYS)
    outcome.update(
        schema_kind=OUTCOME_SCHEMA_KIND,
        member_id=member_id,
        attempt=attempt,
        classification=classification,
        recorded_by=recorded_by,
        finished_at=utc_now(),
    )
    unknown = set(fields) - _OUTCOME_KEYS
    if unknown or classification not in CLASSIFICATIONS:
        message = f"Outcome has unknown fields {sorted(unknown)} or classification {classification!r}."
        raise ValueError(message)
    outcome.update(fields)
    return outcome


def seal_due(classifications: list[str], *, max_attempts: int) -> bool:
    """Return whether an attempt history is final: a final disposition, a reproduced solver failure, or the attempt limit."""
    counted = [item for item in classifications if item != NOT_STARTED]
    if not counted:
        return False
    return (
        counted[-1] in _FINAL_AT_ONCE
        or (counted[-1] == "solver_failure" and counted.count("solver_failure") >= _REPRODUCED_SOLVER_FAILURES)
        or len(counted) >= max_attempts
    )


def _link_case(directory: Path, member_id: str, outcome: Mapping[str, Any]) -> None:
    """Hard-link the case.h5 a succeeded outcome names (its attempt's output) into the member record; idempotent, digest-checked."""
    target = member_directory(directory, member_id) / "case.h5"
    if not target.exists() and not target.is_symlink():
        os.link(attempt_directory(directory, member_id, int(outcome["attempt"])) / "case.h5", target)
    if target.is_symlink() or not target.is_file() or common.serialization.file_sha256(target) != outcome["case_h5_sha256"]:
        message = f"Member case.h5 of {member_id!r} does not hold the case_h5_sha256 its succeeded outcome records."
        raise ValueError(message)


def write_seal(directory: Path, member: Mapping[str, Any], outcome: Mapping[str, Any], *, attempts: int) -> dict[str, Any]:
    """Seal one member once from its final outcome, linking a success's case.h5 first; the caller holds the member lock."""
    seal: dict[str, Any] = {
        "schema_kind": SEAL_SCHEMA_KIND,
        "member_id": member["member_id"],
        "disposition": outcome["classification"],
        "attempts": attempts,
        **{
            key: outcome[key]
            for key in (
                "bundle_sha256",
                "spatial_support_attempt",
                "schedule_acceptance_attempt",
                "cpu_model",
                "started_at",
                "finished_at",
                "failure",
                "input_comparison",
            )
        },
        "case_h5_sha256": outcome["case_h5_sha256"] if outcome["classification"] == "succeeded" else None,
    }
    if member["role"] == "gate":
        seal["gate"] = outcome["gate"]
    if seal["disposition"] == "succeeded":
        _link_case(directory, str(member["member_id"]), outcome)
    roster_service.publish_once(member_directory(directory, str(member["member_id"])) / "seal.json", seal)
    return seal


def settle(directory: Path, roster: Mapping[str, Any], member: Mapping[str, Any]) -> dict[str, Any] | None:
    """Seal one unsealed member when its recorded attempt history is final and return the seal; the caller holds its lock."""
    existing = read_seal(directory, str(member["member_id"]))
    if existing is not None:
        return existing
    outcomes = [read_outcome(directory, str(member["member_id"]), number) for number in attempt_numbers(directory, str(member["member_id"]))]
    finished = [outcome for outcome in outcomes if outcome is not None]
    classifications = [str(outcome["classification"]) for outcome in finished]
    if not seal_due(classifications, max_attempts=int(roster["max_attempts"])):
        return None
    counted = [outcome for outcome in finished if outcome["classification"] != NOT_STARTED]
    return write_seal(directory, member, counted[-1], attempts=len(counted))


def record_outcome(directory: Path, roster: Mapping[str, Any], member: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any] | None:
    """Publish one attempt outcome once, then seal the member when its history is final; the caller holds the member lock."""
    path = attempt_directory(directory, str(member["member_id"]), int(outcome["attempt"])) / "outcome.json"
    roster_service.publish_once(path, outcome)
    return settle(directory, roster, member)


def seal_without_attempt(directory: Path, member: Mapping[str, Any], disposition: str, *, reason_code: str, message: str) -> dict[str, Any]:
    """Seal one member that needs no step (a failed gate blocks it), with zero attempts; the caller holds its lock."""
    outcome = new_outcome(
        str(member["member_id"]),
        1,
        disposition,
        recorded_by="dispatch",
        failure={"reason_code": reason_code, "message": message},
        started_at=utc_now(),
    )
    return write_seal(directory, member, outcome, attempts=0)


def effective_members(directory: Path, roster: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Return the members a verdict judges: a verified replaced base keeps its sealed records but leaves every verdict.

    Raises
    ------
    ValueError
        If a replaced base's gate is unsealed, sealed successful, or sealed with another disposition than recorded.

    """
    replaced: set[str] = set()
    for item in roster.get("replacements", []):
        seal = read_seal(directory, str(item["failed_gate_member_id"]))
        if seal is None or gate_succeeded(seal) or seal["disposition"] != item["failed_disposition"]:
            message = f"Replacement of gate {item['failed_gate_member_id']!r} is not backed by its sealed failure."
            raise ValueError(message)
        replaced.add(str(item["failed_gate_member_id"]))
    return [item for item in roster["members"] if (item["member_id"] if item["role"] == "gate" else item["gate_member_id"]) not in replaced]
