"""
generation_campaign_status.py

Render concise human and workflow-monitor views of canonical campaign status.
Responsibilities:
  - Format deterministic per-case scheduler and runtime progress evidence
  - Derive stable state and solver-advancement render signatures
  - Bound automatic active-case output without hiding explicit status detail
Design principles:
  - Presentation consumes canonical status without changing campaign decisions
  - Partial solver values remain raw evidence rather than convergence claims
  - Missing monitoring evidence is explicit and non-fatal
This module does NOT:
  - Query Slurm, parse COMSOL logs, or validate terminal case success
  - Persist monitoring evidence or authorize publication and cleanup
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src import common

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_MONITOR_RECORD_KIND = "campaign-monitor"
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_PENDING_CASE_STATES = frozenset(
    {
        "pending",
        "never_started",
        "cancelled",
        "interrupted",
        "license_blocked",
        "scheduler_unknown",
    }
)
_FAILED_CASE_STATES = frozenset(
    {
        "failed",
        "timed_out",
        "exports_failed",
        "conversion_failed",
        "publication_failed",
    }
)


def _text(value: object) -> str:
    """Return concise explicit text for one optional value."""
    if value is None or value == "":
        return "unavailable"
    return str(value)


def _float_value(value: object) -> float | None:
    """Return one finite numeric value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _format_simulated_time(value: object) -> str:
    """Return physical simulated time in hours."""
    seconds = _float_value(value)
    return "unavailable" if seconds is None else f"{seconds / 3_600:.4g} h"


def _format_step_size(value: object) -> str:
    """Return a compact adaptive step size with a suitable unit."""
    seconds = _float_value(value)
    if seconds is None:
        return "unavailable"
    magnitude = abs(seconds)
    if magnitude >= _SECONDS_PER_HOUR:
        return f"{seconds / _SECONDS_PER_HOUR:.4g} h"
    if magnitude >= _SECONDS_PER_MINUTE:
        return f"{seconds / _SECONDS_PER_MINUTE:.4g} min"
    return f"{seconds:.4g} s"


def _format_comsol_stage(value: object) -> str:
    """Return raw COMSOL stage percentage or explicit unavailability."""
    numeric = _float_value(value)
    return "unavailable" if numeric is None else f"{numeric:g}%"


def _format_age(runtime: Mapping[str, Any]) -> str:
    """Return explicit runtime-progress freshness text."""
    age = _float_value(runtime.get("age_seconds"))
    if age is None:
        return "unavailable"
    if age < _SECONDS_PER_MINUTE:
        rendered = f"{int(age)} s ago"
    elif age < _SECONDS_PER_HOUR:
        rendered = f"{int(age // _SECONDS_PER_MINUTE)} min ago"
    else:
        rendered = f"{int(age // _SECONDS_PER_HOUR)} h ago"
    return f"{rendered} (stale)" if runtime.get("stale") is True else rendered


def _runtime_view(case: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one well-shaped runtime view or an unavailable substitute."""
    runtime = case.get("runtime_progress")
    if isinstance(runtime, dict):
        return runtime
    return {"availability": "unavailable", "reason": "not_reported"}


def _case_bucket(case: Mapping[str, Any]) -> str:
    """Return one aggregate presentation bucket for a canonical case state."""
    state = str(case.get("state"))
    if state == "successful":
        return "completed"
    if state in _FAILED_CASE_STATES:
        return "failed"
    if state == "active":
        return "active"
    if state in _PENDING_CASE_STATES:
        return "pending"
    return "failed"


def _case_heading(case: Mapping[str, Any], *, show_batch: bool) -> str:
    """Return one compact exact case, job, node, and elapsed line."""
    parts = [str(case["case_id"])]
    if show_batch:
        parts.append(f"batch={case['batch_name']}")
    parts.extend(
        (
            f"job={_text(case.get('latest_job_id'))}",
            f"node={_text(case.get('node'))}",
            f"elapsed={_text(case.get('elapsed'))}",
        )
    )
    return "  ".join(parts)


def _active_case_lines(case: Mapping[str, Any], *, show_batch: bool) -> list[str]:
    """Return compact multiline raw evidence for one active case."""
    lines = [_case_heading(case, show_batch=show_batch)]
    runtime = _runtime_view(case)
    if runtime.get("availability") != "available":
        reason = _text(runtime.get("reason"))
        lines.append(f"  phase=unavailable  runtime_progress={reason}")
        return lines
    phase = _text(runtime.get("phase"))
    parser_state = runtime.get("parser_state")
    if parser_state == "available" and runtime.get("simulated_time_seconds") is not None:
        lines.append(
            "  "
            f"phase={phase}  sim_time={_format_simulated_time(runtime.get('simulated_time_seconds'))}  "
            f"step={_format_step_size(runtime.get('step_size_seconds'))}"
        )
        lines.append(
            "  "
            f"order={_text(runtime.get('order'))}  Tfail={_text(runtime.get('time_failures'))}  "
            f"NLfail={_text(runtime.get('nonlinear_failures'))}  updated={_format_age(runtime)}"
        )
    elif parser_state == "available" and runtime.get("nonlinear_iteration") is not None:
        lines.append(
            "  "
            f"phase={phase}  nonlinear_iteration={_text(runtime.get('nonlinear_iteration'))}  "
            f"COMSOL_stage={_format_comsol_stage(runtime.get('comsol_progress_percent'))}  "
            f"updated={_format_age(runtime)}"
        )
    else:
        lines.append(f"  phase={phase}  solver=unavailable  updated={_format_age(runtime)}")
    return lines


def _nonactive_case_line(case: Mapping[str, Any], *, show_batch: bool) -> str:
    """Return one concise state line for a non-active case."""
    heading = _case_heading(case, show_batch=show_batch)
    details = [
        f"state={_text(case.get('state'))}",
        f"reason={_text(case.get('reason'))}",
    ]
    if case.get("failure_stage") is not None:
        details.extend(
            (
                f"solver={_text(case.get('solver_state'))}",
                f"failure_stage={_text(case.get('failure_stage'))}",
            )
        )
    return f"{heading}  {'  '.join(details)}"


def _case_inventory(status: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return the canonical deterministic case inventory."""
    cases = status.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        return ()
    return tuple(cases)


def _append_section(lines: list[str], title: str, records: Sequence[str]) -> None:
    """Append one separated human-summary section when it has records."""
    if not records:
        return
    lines.extend(("", f"{title}:"))
    lines.extend(records)


def format_campaign_status_summary(
    status: Mapping[str, Any],
    *,
    max_active_cases: int | None = None,
) -> str:
    """
    Format canonical campaign status as a concise human summary.

    Parameters
    ----------
    status : Mapping[str, Any]
        Result returned by the canonical campaign-status service.
    max_active_cases : int | None, optional
        Maximum active cases to render. ``None`` renders every active case.

    Returns
    -------
    str
        Multiline human-readable campaign and per-case evidence.

    Notes
    -----
    Runtime progress is observational. Successful state remains owned by
    canonical completed-case validation.

    """
    if max_active_cases is not None and (isinstance(max_active_cases, bool) or max_active_cases < 1):
        message = "max_active_cases must be a positive integer or None."
        raise ValueError(message)
    cases = _case_inventory(status)
    buckets = {name: [case for case in cases if _case_bucket(case) == name] for name in ("completed", "active", "pending", "failed")}
    lines = [
        f"Campaign: {_text(status.get('campaign_run_id'))}",
        f"State: {_text(status.get('campaign_state'))}",
        (
            f"Cases: {len(buckets['completed'])}/{len(cases)} completed, "
            f"{len(buckets['active'])} active, {len(buckets['pending'])} pending, "
            f"{len(buckets['failed'])} failed"
        ),
    ]
    show_batch = len({str(case.get("batch_name")) for case in cases}) > 1
    active = buckets["active"]
    visible_active = active if max_active_cases is None else active[:max_active_cases]
    active_lines: list[str] = []
    for index, case in enumerate(visible_active):
        if index:
            active_lines.append("")
        active_lines.extend(_active_case_lines(case, show_batch=show_batch))
    omitted = len(active) - len(visible_active)
    if omitted:
        active_lines.append(f"... {omitted} additional active case(s) omitted")
    _append_section(lines, "Active cases", active_lines)
    license_blocked_records = [
        _nonactive_case_line(case, show_batch=show_batch) for case in buckets["pending"] if case.get("state") == "license_blocked"
    ]
    pending_records = [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["pending"] if case.get("state") != "license_blocked"]
    _append_section(lines, "License-blocked cases", license_blocked_records)
    _append_section(lines, "Pending cases", pending_records)
    _append_section(
        lines,
        "Failed cases",
        [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["failed"]],
    )
    _append_section(
        lines,
        "Completed cases",
        [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["completed"]],
    )
    return "\n".join(lines)


def campaign_monitor_signatures(status: Mapping[str, Any]) -> tuple[str, str]:
    """
    Derive state and solver-advancement signatures for rate-limited rendering.

    Parameters
    ----------
    status : Mapping[str, Any]
        Result returned by the canonical campaign-status service.

    Returns
    -------
    tuple[str, str]
        Urgent state/identity signature and detailed solver-progress signature.

    """
    urgent_cases: list[dict[str, Any]] = []
    progress_cases: list[dict[str, Any]] = []
    for case in _case_inventory(status):
        runtime = _runtime_view(case)
        urgent = {
            "batch_name": case.get("batch_name"),
            "case_id": case.get("case_id"),
            "state": case.get("state"),
            "latest_job_id": case.get("latest_job_id"),
            "scheduler_state": case.get("scheduler_state"),
            "node": case.get("node"),
            "phase": runtime.get("phase"),
            "terminal": runtime.get("terminal"),
        }
        urgent_cases.append(urgent)
        progress_cases.append(
            {
                **urgent,
                "parser_state": runtime.get("parser_state"),
                "comsol_progress_percent": runtime.get("comsol_progress_percent"),
                "step_index": runtime.get("step_index"),
                "simulated_time_seconds": runtime.get("simulated_time_seconds"),
                "step_size_seconds": runtime.get("step_size_seconds"),
                "order": runtime.get("order"),
                "time_failures": runtime.get("time_failures"),
                "nonlinear_failures": runtime.get("nonlinear_failures"),
                "nonlinear_iteration": runtime.get("nonlinear_iteration"),
            }
        )
    urgent_payload = {"campaign_state": status.get("campaign_state"), "cases": urgent_cases}
    progress_payload = {"campaign_state": status.get("campaign_state"), "cases": progress_cases}
    return (
        common.serialization.canonical_json_sha256(urgent_payload),
        common.serialization.canonical_json_sha256(progress_payload),
    )


def format_campaign_monitor(
    status: Mapping[str, Any],
    *,
    max_active_cases: int,
) -> str:
    """
    Format one machine header followed by the reusable human summary.

    Parameters
    ----------
    status : Mapping[str, Any]
        Result returned by the canonical campaign-status service.
    max_active_cases : int
        Positive bound for active cases in automatic workflow output.

    Returns
    -------
    str
        One tab-delimited control header and the human summary.

    """
    urgent_signature, progress_signature = campaign_monitor_signatures(status)
    header = "\t".join(
        (
            _MONITOR_RECORD_KIND,
            str(status["campaign_state"]),
            urgent_signature,
            progress_signature,
        )
    )
    summary = format_campaign_status_summary(status, max_active_cases=max_active_cases)
    return f"{header}\n{summary}"
