"""Derive campaign success deficits and selected membership from candidate evidence.

Responsibilities:
  - Count successful, terminal-failed, active, retryable, and surplus candidates
  - Determine replacement capacity without oversubmitting active work
  - Select exact successful targets in stable design order

Design principles:
  - An operational attempt failure does not create a replacement deficit
  - A candidate's immutable local ID is distinct from its target success slot
  - Selection is independent of job completion order

This module does NOT:
  - Persist cases, create designs, submit jobs, or infer solver success
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _design_order(record: Mapping[str, Any]) -> tuple[int, int, str]:
    """Order original design before supplemental candidates."""
    reservation = record["reservation"]
    return (
        0 if reservation["origin"] == "primary" else 1,
        int(reservation["candidate_position"]),
        str(reservation["case_id"]),
    )


def select_successes(records: Sequence[Mapping[str, Any]], *, target: int) -> tuple[Mapping[str, Any], ...]:
    """Return exact target successes only after all candidate outcomes settle."""
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        msg = "Selected-success target must be positive."
        raise ValueError(msg)
    if any(record["decision"] is None for record in records):
        msg = "Cannot finalize selection while reserved candidates remain active or retryable."
        raise RuntimeError(msg)
    successful = [record for record in records if record["decision"]["outcome"] == "successful"]
    successful.sort(key=_design_order)
    if len(successful) < target:
        msg = f"Only {len(successful)} successful candidates satisfy a target of {target}."
        raise RuntimeError(msg)
    return tuple(successful[:target])


def replacement_deficit(records: Sequence[Mapping[str, Any]], *, target: int) -> int:
    """Count unfilled success slots after accounting for all pending candidates."""
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        msg = "Selected-success target must be positive."
        raise ValueError(msg)
    primary = [record for record in records if record["reservation"]["origin"] == "primary"]
    if len(primary) < target:
        return 0
    successful = sum(record["decision"] is not None and record["decision"]["outcome"] == "successful" for record in records)
    pending = sum(record["decision"] is None for record in records)
    return max(0, target - successful - pending)


def candidate_summary(records: Sequence[Mapping[str, Any]], *, target: int) -> dict[str, Any]:
    """Report candidate fates, selection and safe replacement capacity."""
    selected: tuple[Mapping[str, Any], ...]
    try:
        selected = select_successes(records, target=target)
    except RuntimeError:
        selected = ()
    counts = {
        "primary": 0,
        "replacement": 0,
        "successful": 0,
        "terminal_failed": 0,
        "active": 0,
        "retryable": 0,
        "pending_validation": 0,
        "active_or_retryable": 0,
        "selected": len(selected),
        "surplus": 0,
        "replacement_deficit": replacement_deficit(records, target=target),
    }
    for record in records:
        reservation = record["reservation"]
        decision = record["decision"]
        counts[reservation["origin"]] += 1
        if decision is None:
            counts["active_or_retryable"] += 1
            attempts = record.get("attempts", ())
            latest = attempts[-1] if attempts else None
            if latest is not None and latest.get("retryable") is True:
                counts["retryable"] += 1
            elif latest is not None and latest.get("stage") == "validation":
                counts["pending_validation"] += 1
            else:
                counts["active"] += 1
        elif decision["outcome"] == "successful":
            counts["successful"] += 1
        else:
            counts["terminal_failed"] += 1
    counts["surplus"] = max(0, counts["successful"] - target)
    return counts
