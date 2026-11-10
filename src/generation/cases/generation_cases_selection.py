"""Derive campaign success deficits, success-rate stops and selected membership from candidate evidence.

Responsibilities:
  - Count successful, terminal-failed, active, retryable, and surplus candidates
  - Determine the replacement deficit without oversubmitting active work
  - Bound a batch's success rate from its decided candidates and decide whether the batch is stopped
  - Derive the most failures a batch can have and still reach its count, and the default admission cap they sum to
  - Select exact successful targets in stable design order

Design principles:
  - An operational attempt failure does not create a replacement deficit
  - Only terminal decisions count towards the success rate; undecided candidates never do
  - A candidate's immutable local ID is distinct from its target success slot
  - Selection is independent of job completion order

This module does NOT:
  - Persist cases, create designs, submit jobs, or infer solver success
  - Own the minimum success rate itself, which the campaign records as amendments

Notes:
  Success-rate stop:
    - k is the number of successful and n the number of decided (successful or failed) candidates of one material,
      pooled over all its designs; the campaign lifecycle passes that material's records.
    - The bound is the one-sided Clopper-Pearson upper bound at SUCCESS_RATE_CONFIDENCE: Beta quantile (k + 1, n - k), 1 for k == n.
    - A batch is stopped while n >= 1 and the bound is below its minimum success rate s.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from scipy.stats import beta as beta_distribution

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SUCCESS_RATE_CONFIDENCE: Final = 0.95
# A plan admits count + f fresh candidates per batch; f beyond a million is no realistic campaign and a misconfigured rate.
MAXIMUM_TOLERATED_FAILURES_BRACKET: Final = 2**20


def _design_order(record: Mapping[str, Any]) -> tuple[int, int, str]:
    """Order the original design points before their replacements, each by candidate position."""
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


def decided_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Return ``(k, n)``: the successful and the decided (successful or failed) candidates of one batch."""
    outcomes = [record["decision"]["outcome"] for record in records if record["decision"] is not None]
    return outcomes.count("successful"), len(outcomes)


def success_rate_upper_bound(successful: int, decided: int) -> float:
    """Return the one-sided Clopper-Pearson upper bound of a success rate from ``successful`` of ``decided`` candidates."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (successful, decided)) or not 0 <= successful <= decided:
        msg = "Success-rate counts must be integers with 0 <= successful <= decided."
        raise ValueError(msg)
    if successful == decided:
        return 1.0
    return float(beta_distribution.ppf(SUCCESS_RATE_CONFIDENCE, successful + 1, decided - successful))


def success_rate_stopped(successful: int, decided: int, *, minimum_success_rate: float) -> bool:
    """Return whether a batch with ``successful`` of ``decided`` candidates falls below its minimum success rate."""
    return decided >= 1 and success_rate_upper_bound(successful, decided) < minimum_success_rate


def maximum_tolerated_failures(count: int, minimum_success_rate: float) -> int:
    """Return the most failures f with which a batch of ``count`` originals can still complete.

    The batch completes when its last success is admitted at ``count - 1`` successes and f failures, so f is the largest
    failure count whose bound over those ``count - 1 + f`` decided candidates stays at or above ``minimum_success_rate``.
    The bound falls monotonically in f, so a doubling search brackets f and a bisection finds it; a rate whose f reaches
    ``MAXIMUM_TOLERATED_FAILURES_BRACKET`` raises ``ValueError``.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not 0.0 < minimum_success_rate <= 1.0:
        msg = "A batch count must be a positive integer and its minimum success rate lie in (0, 1]."
        raise ValueError(msg)

    def admits(failures: int) -> bool:
        return success_rate_upper_bound(count - 1, count - 1 + failures) >= minimum_success_rate

    low, high = 0, 1
    while admits(high):
        if high >= MAXIMUM_TOLERATED_FAILURES_BRACKET:
            msg = (
                f"Minimum success rate {minimum_success_rate} is too small for a bounded admission cap: "
                f"a batch of {count} originals would tolerate at least {high} failures."
            )
            raise ValueError(msg)
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if admits(middle) else (low, middle)
    return low


def default_admission_cap(batches: Iterable[tuple[int, float]]) -> int:
    """Return the fresh candidates a plan admits by default: each ``(count, rate)`` batch's count plus the failures it tolerates."""
    return sum(count + maximum_tolerated_failures(count, rate) for count, rate in batches)


def candidate_summary(records: Sequence[Mapping[str, Any]], *, target: int) -> dict[str, Any]:
    """Report candidate fates, selection and the replacement deficit."""
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
