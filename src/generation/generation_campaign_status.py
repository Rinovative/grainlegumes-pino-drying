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
from datetime import datetime, timezone
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from src import common

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_MONITOR_RECORD_KIND = "campaign-monitor"
_SOURCE_MONITOR_RECORD_KIND = "source-monitor"
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_MAX_ACTIONABLE_DETAILS = 20
_MAX_RECENT_FAILURES = 3
_MAX_OLDER_FAILURE_GROUPS = 8
_MAX_REASON_CHARACTERS = 160
_MAX_EVIDENCE_PATH_CHARACTERS = 80
_STORAGE_DOMAIN_NAMES = frozenset({"01_generation", "02_datasets", "03_experiments"})
_UNSAFE_CAPACITY_STATUS_PREFIX = "Unsafe COMSOL capacity-checkout status artifact:"
_CASE_SUMMARY_LABELS = (
    "successful",
    "running",
    "scheduler_pending",
    "license_blocked",
    "not_admitted",
    "failed",
)
_ADMISSION_SUMMARY_LABELS = (
    "pending",
    "starting",
    "acquiring_license",
    "license_waiting",
)
_FAILED_CASE_STATES = frozenset(
    {
        "failed",
        "timed_out",
        "exports_failed",
        "conversion_failed",
        "publication_failed",
        "case_reconciliation_failed",
    }
)
_PRE_SOLVER_RUNTIME_PHASES = frozenset(
    {
        "preparing",
        "acquiring_comsol_license",
        "starting_solver",
    }
)


def _text(value: object) -> str:
    """Return concise explicit text for one optional value."""
    if value is None or value == "":
        return "unavailable"
    return str(value)


def _available_text(value: object) -> str | None:
    """Return displayable scalar text while suppressing routine empty values."""
    if value is None or value is False or isinstance(value, (dict, list, tuple, set)):
        return None
    if value in {"", "unavailable", "not_applicable"}:
        return None
    return str(value)


def _bounded_text(value: object, *, maximum: int) -> str | None:
    """Return one whitespace-normalized scalar bounded for terminal display."""
    text = _available_text(value)
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= maximum:
        return compact
    return f"{compact[: maximum - 3].rstrip()}..."


def _compact_evidence_path(value: object) -> str | None:
    """Return one compact evidence path relative to its storage domain."""
    text = _available_text(value)
    if text is None:
        return None
    path = PurePath(text)
    relative_parts = [part for part in path.parts if part not in {path.anchor, "/"}]
    domain_index = next(
        (index for index, part in enumerate(relative_parts) if part in _STORAGE_DOMAIN_NAMES),
        None,
    )
    if domain_index is not None:
        compact = PurePath(*relative_parts[domain_index:]).as_posix()
    else:
        compact = PurePath(*relative_parts[-3:]).as_posix()
        if path.is_absolute():
            compact = f".../{compact}"
    return _bounded_text(compact, maximum=_MAX_EVIDENCE_PATH_CHARACTERS)


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
        return "successful"
    if state in _FAILED_CASE_STATES:
        return "failed"
    if state in {"running", "active"}:
        return "running"
    if state == "license_blocked":
        return "license_blocked"
    if state in {"scheduler_pending", "pending", "scheduler_unknown"}:
        return "scheduler_pending"
    if state == "admission_waiting":
        # A reservation has entered admission; the Admission line retains its
        # precise starting/acquiring/waiting occupancy projection.
        return "running"
    if state == "never_started":
        return "not_admitted"
    return "failed"


def _active_case_sort_key(case: Mapping[str, Any]) -> tuple[int, str, str]:
    """Group useful execution before pre-solver waiting with stable identity."""
    runtime = _runtime_view(case)
    phase = runtime.get("phase")
    waiting = runtime.get("availability") != "available" or phase is None or phase in _PRE_SOLVER_RUNTIME_PHASES
    batch_identity = str(case.get("batch_id") or case.get("batch_name") or "")
    return int(waiting), batch_identity, str(case.get("case_id") or "")


def _case_heading(case: Mapping[str, Any], *, include_runtime: bool = True) -> str:
    """Return one compact work-unit identity and scheduler heading."""
    parts = [str(case["case_id"])]
    label = _group_label(case)
    if label != "unavailable":
        parts.append(f"batch={label}")
    variant = _available_text(case.get("variant_id"))
    role = _available_text(case.get("case_role"))
    if variant is not None and variant != label:
        parts.append(f"variant={variant}")
    if role is not None:
        parts.append(f"role={role}")
    for name, value in (("job", case.get("latest_job_id")), ("node", case.get("node"))):
        rendered = _available_text(value)
        if rendered is not None:
            parts.append(f"{name}={rendered}")
    if include_runtime:
        elapsed = _available_text(case.get("elapsed"))
        if elapsed is not None:
            parts.append(f"elapsed={elapsed}")
    return "  ".join(parts)


def _active_case_lines(case: Mapping[str, Any]) -> list[str]:
    """Return compact multiline progress for one running work unit."""
    lines = [_case_heading(case)]
    runtime = _runtime_view(case)
    if runtime.get("availability") != "available":
        reason = _bounded_text(runtime.get("reason"), maximum=_MAX_REASON_CHARACTERS)
        if reason is not None:
            lines.append(f"  runtime_progress={reason}")
        return lines
    phase = _available_text(runtime.get("phase"))
    if phase is not None:
        displayed_phase = "acquiring_comsol_license" if phase in {"starting_solver", "acquiring_comsol_license"} else phase
        phase_parts = [f"phase={displayed_phase}"]
        if phase in {"stationary_airflow", "transient_drying"}:
            progress = _available_text(_format_comsol_stage(runtime.get("comsol_progress_percent")))
            if progress is not None:
                phase_parts.append(f"progress={progress}")
        lines.append(f"  {'  '.join(phase_parts)}")
    if phase in {"starting_solver", "acquiring_comsol_license"}:
        window = _float_value(runtime.get("license_window_seconds"))
        limit = _float_value(runtime.get("license_window_limit_seconds"))
        checkouts = _available_text(runtime.get("license_checkout_attempt_count"))
        acquisition_parts = []
        if window is not None and limit is not None:
            acquisition_parts.append(f"window={window:.0f} s / {limit:g} s")
        if checkouts is not None:
            acquisition_parts.append(f"checkouts={checkouts}")
        if acquisition_parts:
            lines.append(f"  {'  '.join(acquisition_parts)}")
        last_result = _bounded_text(runtime.get("last_license_result"), maximum=_MAX_REASON_CHARACTERS)
        if last_result is not None:
            lines.append(f"  checkout_result={last_result}")
        recovered = case.get("status_artifact_recovery_count")
        if isinstance(recovered, int) and not isinstance(recovered, bool) and recovered > 0:
            lines.append("  status_artifact=recovered")
    if phase in {"stationary_airflow", "transient_drying"} and runtime.get("parser_state") == "available":
        solver_values = (
            ("simulated_time", _format_simulated_time(runtime.get("simulated_time_seconds"))),
            ("step", runtime.get("step_index")),
            ("step_size", _format_step_size(runtime.get("step_size_seconds"))),
        )
        solver_parts = [f"{name}={rendered}" for name, value in solver_values if (rendered := _available_text(value)) is not None]
        if solver_parts:
            lines.append(f"  {'  '.join(solver_parts)}")
        failure_parts = []
        for name, value in (("Tfail", runtime.get("time_failures")), ("NLfail", runtime.get("nonlinear_failures"))):
            rendered = _available_text(value)
            if rendered is not None:
                failure_parts.append(f"{name}={rendered}")
        if failure_parts:
            lines.append(f"  {'  '.join(failure_parts)}")
        update_parts = []
        updated = _available_text(runtime.get("last_solver_log_update_at"))
        if updated is not None:
            update_parts.append(f"last_solver_update={updated}")
        age = _available_text(_format_age(runtime))
        if age is not None:
            update_parts.append(f"age={age}")
        if update_parts:
            lines.append(f"  {'  '.join(update_parts)}")
    return lines


def _scheduler_pending_lines(case: Mapping[str, Any]) -> list[str]:
    """Return one compact scheduler-pending work-unit block."""
    lines = [_case_heading(case, include_runtime=False)]
    details = []
    classified_state = case.get("classified_state", case.get("state"))
    reason = "scheduler_visibility_unknown" if classified_state == "scheduler_unknown" else case.get("scheduler_state") or case.get("reason")
    for name, value in (
        ("queue_age", case.get("queue_age")),
        ("reason", reason),
        ("cores", case.get("requested_cores")),
    ):
        rendered = _bounded_text(value, maximum=_MAX_REASON_CHARACTERS)
        if rendered is not None:
            details.append(f"{name}={rendered}")
    if details:
        lines.append(f"  {'  '.join(details)}")
    return lines


def _license_blocked_lines(case: Mapping[str, Any]) -> list[str]:
    """Return concise operational license retry detail without raw evidence."""
    lines = [_case_heading(case, include_runtime=False), "  state=license_blocked"]
    reason = _bounded_text(case.get("reason"), maximum=_MAX_REASON_CHARACTERS)
    if reason is not None:
        lines.append(f"  reason={reason}")
    window = case.get("in_allocation_license_window")
    if isinstance(window, dict):
        realised = _float_value(window.get("realised_window_seconds"))
        checkouts = _available_text(window.get("checkout_attempt_count"))
        window_parts = []
        if realised is not None:
            window_parts.append(f"window={realised:.0f} s")
        if checkouts is not None:
            window_parts.append(f"checkouts={checkouts}")
        if window_parts:
            lines.append(f"  {'  '.join(window_parts)}")
    recovered = case.get("status_artifact_recovery_count")
    if isinstance(recovered, int) and not isinstance(recovered, bool) and recovered > 0:
        lines.append(f"  status_artifacts_recovered={recovered}")
    retry = case.get("temporary_license_retry")
    if isinstance(retry, dict):
        retry_parts = []
        for name, value in (
            ("retry", retry.get("retry_count")),
            ("next_retry", retry.get("next_retry_at")),
        ):
            rendered = _available_text(value)
            if rendered is not None:
                retry_parts.append(f"{name}={rendered}")
        if retry_parts:
            lines.append(f"  {'  '.join(retry_parts)}")
        wait = _available_text(retry.get("cumulative_wait_seconds"))
        if wait is not None:
            lines.append(f"  cumulative_wait={wait} s")
    return lines


def _replay_state(case: Mapping[str, Any]) -> str:
    """Return the compact operational replay state for a failed case."""
    if case.get("replay_running") is True:
        return "running"
    if case.get("replay_blocked") is True:
        return "blocked"
    if case.get("replay_eligible") is True:
        return "eligible"
    return "unavailable"


def _failed_case_lines(case: Mapping[str, Any]) -> list[str]:
    """Return compact actionable failure detail with bounded evidence references."""
    lines = [_case_heading(case)]
    state = _available_text(case.get("classified_state")) or _text(case.get("state"))
    details = [f"state={state}"]
    for name, value in (("stage", case.get("failure_stage")), ("solver", case.get("solver_state"))):
        rendered = _available_text(value)
        if rendered is not None:
            details.append(f"{name}={rendered}")
    lines.append(f"  {'  '.join(details)}")
    lines.append(f"  replay={_replay_state(case)}")
    raw_reason = case.get("reason")
    reason: str | None
    if isinstance(raw_reason, str) and raw_reason.startswith(_UNSAFE_CAPACITY_STATUS_PREFIX):
        reason = "Unowned or unsupported COMSOL status artifact."
    else:
        reason = _bounded_text(raw_reason, maximum=_MAX_REASON_CHARACTERS)
    if reason is not None:
        lines.append(f'  reason="{reason}"')
    reconciliation = case.get("case_reconciliation")
    if isinstance(reconciliation, dict):
        category = _bounded_text(
            reconciliation.get("failure_category"),
            maximum=_MAX_REASON_CHARACTERS,
        )
        parts = [] if category is None else [f"category={category}"]
        parts.extend(
            (
                f"scientific_success_valid={str(reconciliation.get('scientific_success_valid') is True).lower()}",
                f"admission_continues={str(reconciliation.get('admission_continues') is True).lower()}",
            )
        )
        lines.append(f"  {'  '.join(parts)}")
    evidence = _compact_evidence_path(case.get("case_reconciliation_evidence_path") or case.get("evidence_path") or case.get("replay_evidence_path"))
    if evidence is not None:
        lines.append(f"  evidence={evidence}")
    return lines


def _completed_case_lines(case: Mapping[str, Any]) -> list[str]:
    """Return compact successful-case terminal evidence."""
    lines = [_case_heading(case, include_runtime=False)]
    details = ["state=successful"]
    reason = _bounded_text(case.get("reason"), maximum=_MAX_REASON_CHARACTERS)
    if reason is not None:
        details.append(f"reason={reason}")
    lines.append(f"  {'  '.join(details)}")
    terminal = []
    simulated_end = _float_value(case.get("simulated_end_time"))
    simulated_end_unit = _available_text(case.get("simulated_end_time_unit"))
    if simulated_end is not None and simulated_end_unit is not None:
        terminal.append(f"simulated_end={simulated_end:.2f} {simulated_end_unit}")
    bulk_moisture = _float_value(case.get("final_bulk_moisture_wb"))
    if bulk_moisture is not None:
        terminal.append(f"bulk_moisture={100.0 * bulk_moisture:.1f}% wb")
    target = _float_value(case.get("target_moisture_wb"))
    if target is not None:
        terminal.append(f"target={100.0 * target:.1f}% wb")
    if terminal:
        lines.append(f"  {'  '.join(terminal)}")
    reconciliation = case.get("case_reconciliation")
    if isinstance(reconciliation, dict):
        category = _bounded_text(
            reconciliation.get("failure_category"),
            maximum=_MAX_REASON_CHARACTERS,
        )
        evidence = _compact_evidence_path(case.get("case_reconciliation_evidence_path"))
        parts = []
        if category is not None:
            parts.append(f"presentation={category}")
        parts.append(f"scientific_success_valid={str(reconciliation.get('scientific_success_valid') is True).lower()}")
        parts.append(f"admission_continues={str(reconciliation.get('admission_continues') is True).lower()}")
        if evidence is not None:
            parts.append(f"evidence={evidence}")
        lines.append(f"  {'  '.join(parts)}")
    return lines


def _terminal_timestamp(
    case: Mapping[str, Any],
    *,
    field: str,
) -> datetime | None:
    """Return one comparable aware terminal timestamp when safely usable."""
    value = case.get(field)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _terminal_case_identity(case: Mapping[str, Any]) -> tuple[str, str]:
    """Return deterministic existing batch/case identity for ordering ties."""
    return (
        str(case.get("batch_id") or case.get("batch_name") or ""),
        str(case.get("case_id") or ""),
    )


def _ordered_terminal_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> tuple[Mapping[str, Any], ...]:
    """Order usable timestamps newest-first and unavailable times by identity."""
    projected = [(case, _terminal_timestamp(case, field=field)) for case in cases]
    timestamped = [(case, timestamp) for case, timestamp in projected if timestamp is not None]
    timestamped.sort(key=lambda item: _terminal_case_identity(item[0]))
    timestamped.sort(key=lambda item: item[1], reverse=True)
    unavailable = sorted(
        (case for case, timestamp in projected if timestamp is None),
        key=_terminal_case_identity,
    )
    return tuple([case for case, _timestamp in timestamped] + unavailable)


def _recent_completed_cases(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return at most three newest completions with deterministic fallback."""
    return _ordered_terminal_cases(cases, field="completed_at")[:3]


def _recent_and_older_failed_cases(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Return latest-three terminal failures and the remaining population."""
    ordered = _ordered_terminal_cases(cases, field="failed_at")
    return ordered[:_MAX_RECENT_FAILURES], ordered[_MAX_RECENT_FAILURES:]


def _normalized_failure_class(case: Mapping[str, Any]) -> str:
    """Return one bounded stable failure category without raw diagnostics."""
    state = _available_text(case.get("classified_state")) or _text(case.get("state"))
    stage = _available_text(case.get("failure_stage"))
    if state == "case_reconciliation_failed":
        return state
    if stage is None:
        return state
    return f"{stage}:{state}"


def _older_failure_rows(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Group all older failures by normalized classification with a hard bound."""
    counts: dict[str, int] = {}
    for case in cases:
        classification = _normalized_failure_class(case)
        counts[classification] = counts.get(classification, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: item[0])
    visible = ordered[:_MAX_OLDER_FAILURE_GROUPS]
    hidden_count = sum(count for _name, count in ordered[_MAX_OLDER_FAILURE_GROUPS:])
    rows = [f"  {name}: {count}" for name, count in visible]
    if hidden_count:
        rows.append(f"  other_classifications: {hidden_count}")
    rows.append(f"  total: {len(cases)}")
    return rows


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


def _group_label(case: Mapping[str, Any]) -> str:
    """Return the authoritative material label, falling back to batch identity."""
    material = case.get("material")
    return str(material) if isinstance(material, str) and material else _text(case.get("batch_name"))


def _grouped_population_rows(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return deterministic batch or material counts for a non-actionable population."""
    counts: dict[str, int] = {}
    for case in cases:
        label = _group_label(case)
        counts[label] = counts.get(label, 0) + 1
    return [*(f"  {label}: {count}" for label, count in counts.items()), f"  total: {len(cases)}"]


def _actionable_section(
    lines: list[str],
    title: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    maximum: int | None,
    renderer: Any,
) -> None:
    """Append bounded deterministic detail for one actionable presentation state."""
    if not cases:
        return
    records: list[str] = []
    visible = cases if maximum is None else cases[:maximum]
    for index, case in enumerate(visible):
        if index:
            records.append("")
        records.extend(renderer(case))
    displayed = len(visible)
    if maximum is not None and displayed < len(cases):
        records.insert(0, f"displayed={displayed}  additional={len(cases) - displayed}")
    _append_section(lines, title, records)


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
    buckets = {name: [case for case in cases if _case_bucket(case) == name] for name in _CASE_SUMMARY_LABELS}
    submission = status.get("submission_config")
    submission_values = submission if isinstance(submission, dict) else {}
    admission = status.get("admission")
    admission_values = admission if isinstance(admission, dict) else {}
    components = admission_values.get("components")
    component_values = components if isinstance(components, dict) else {}
    admission_counts = {name: component_values.get(name, 0) for name in _ADMISSION_SUMMARY_LABELS}
    admission_count = admission_values.get(
        "count",
        sum(value for value in admission_counts.values() if isinstance(value, int) and not isinstance(value, bool)),
    )
    admission_maximum = admission_values.get(
        "maximum",
        submission_values.get("max_admission_cases", 0),
    )
    case_counts = {name: len(buckets[name]) for name in _CASE_SUMMARY_LABELS}
    max_running = submission_values.get("max_running_cases")
    max_running_text = "unlimited" if max_running is None else _text(max_running)
    lines = [
        f"Campaign: {_text(status.get('campaign_run_id'))}",
        f"State: {_text(status.get('campaign_state'))}",
        (f"Execution: commit={_text(status.get('git_commit'))}  config_digest={_text(status.get('execution_config_digest'))}"),
        (
            "Resources: "
            f"cores_per_case={_text(submission_values.get('cores_per_case'))}  "
            f"max_admission_cases={_text(submission_values.get('max_admission_cases'))}  "
            f"max_running_cases={max_running_text}"
        ),
        ("Cases: " + "  ".join(f"{name}={case_counts[name]}" for name in _CASE_SUMMARY_LABELS) + f"  total={len(cases)}"),
        f"Admission: {_text(admission_count)}/{_text(admission_maximum)}",
        ("  " + "  ".join(f"{name}={_text(admission_counts[name])}" for name in _ADMISSION_SUMMARY_LABELS)),
    ]
    runnable_work_remains = any(
        buckets[name]
        for name in (
            "running",
            "scheduler_pending",
            "license_blocked",
            "not_admitted",
        )
    ) or any(case.get("replay_eligible") is True for case in buckets["failed"])
    if status.get("admission_blocked") is True:
        block_reason = _bounded_text(
            status.get("admission_block_reason"),
            maximum=_MAX_REASON_CHARACTERS,
        )
        block_parts = ["state=blocked"]
        if block_reason is not None:
            block_parts.append(f"reason={block_reason}")
        lines.append(f"  {'  '.join(block_parts)}")
    if buckets["failed"] and runnable_work_remains:
        lines.append(f"Failures: {len(buckets['failed'])} terminal, continuing remaining work")
    detail_limit = _MAX_ACTIONABLE_DETAILS if max_active_cases is None else max_active_cases
    active_cases = sorted(buckets["running"], key=_active_case_sort_key)
    _actionable_section(lines, "Active cases", active_cases, maximum=max_active_cases, renderer=_active_case_lines)
    _actionable_section(lines, "License-blocked cases", buckets["license_blocked"], maximum=detail_limit, renderer=_license_blocked_lines)
    _actionable_section(lines, "Scheduler-pending cases", buckets["scheduler_pending"], maximum=None, renderer=_scheduler_pending_lines)
    recent_failed, older_failed = _recent_and_older_failed_cases(buckets["failed"])
    _actionable_section(
        lines,
        (f"Recently failed cases (latest {len(recent_failed)} of {len(buckets['failed'])})"),
        recent_failed,
        maximum=None,
        renderer=_failed_case_lines,
    )
    if older_failed:
        _append_section(
            lines,
            f"Older failures ({len(older_failed)})",
            _older_failure_rows(older_failed),
        )
    recent_completed = _recent_completed_cases(buckets["successful"])
    _actionable_section(
        lines,
        f"Recently completed cases (latest {len(recent_completed)} of {len(buckets['successful'])})",
        recent_completed,
        maximum=None,
        renderer=_completed_case_lines,
    )
    if buckets["not_admitted"]:
        _append_section(
            lines,
            "Not admitted",
            _grouped_population_rows(buckets["not_admitted"]),
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
            "attempt_index": case.get("attempt_index"),
            "attempt_campaign_run_id": case.get("attempt_campaign_run_id"),
            "postprocessing_state": case.get("postprocessing_state"),
            "replay_eligible": case.get("replay_eligible"),
            "replay_blocked": case.get("replay_blocked"),
            "replay_block_reason": case.get("replay_block_reason"),
            "replay_attempt_count": case.get("replay_attempt_count"),
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
    urgent_payload = {
        "campaign_state": status.get("campaign_state"),
        "admission_blocked": status.get("admission_blocked"),
        "admission_block_reason": status.get("admission_block_reason"),
        "cases": urgent_cases,
    }
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


def format_workflow_monitor(
    status: Mapping[str, Any],
    source_status: Mapping[str, Any],
    *,
    max_active_cases: int,
) -> str:
    """Format one campaign header, one source header, and one bounded summary."""
    campaign_record = format_campaign_monitor(
        status,
        max_active_cases=max_active_cases,
    )
    campaign_header, summary = campaign_record.split("\n", 1)
    source_header = "\t".join(
        (
            _SOURCE_MONITOR_RECORD_KIND,
            str(source_status["campaign_run_id"]),
            str(source_status["campaign_state"]),
            str(source_status["source_state"]),
            ("unavailable" if source_status.get("reclaimable_bytes") is None else str(source_status["reclaimable_bytes"])),
            str(source_status["cleanup_eligibility"]),
            str(source_status["active_slurm"]),
        )
    )
    return f"{campaign_header}\n{source_header}\n{summary}"


def format_benchmark_status_summary(
    status: Mapping[str, Any],
    *,
    max_active_cases: int | None = None,
) -> str:
    """Format benchmark status through the common bounded work-unit model."""
    raw_units = status.get("work_units")
    units = raw_units if isinstance(raw_units, list) and all(isinstance(item, dict) for item in raw_units) else []
    normalized = [
        {
            **unit,
            "case_id": str(unit.get("work_unit_id", f"{unit.get('variant_id', 'unavailable')} {unit.get('case_role', 'unavailable')}")),
            "batch_name": str(unit.get("variant_id", "unavailable")),
        }
        for unit in units
    ]
    categories = _CASE_SUMMARY_LABELS
    buckets = {name: [unit for unit in normalized if _case_bucket(unit) == name] for name in categories}
    wave = status.get("current_wave")
    wave_values = wave if isinstance(wave, dict) else {}
    current_variant = wave_values.get("variant_id")
    current_units = [unit for unit in normalized if unit.get("variant_id") == current_variant]
    current_counts = {name: sum(_case_bucket(unit) == name for unit in current_units) for name in categories}
    lines = [
        f"Benchmark: {_text(status.get('suite_name'))}",
        f"Run: {_text(status.get('benchmark_run_id'))}",
        f"State: {_text(status.get('state'))}",
        f"Wave: {_text(wave_values.get('wave_position'))}/{_text(status.get('wave_count'))}",
        f"Current cores_per_case: {_text(wave_values.get('cores_per_case'))}",
        f"Measurements: {len(buckets['successful'])}/{len(normalized)} successful",
        "Work units: " + "  ".join(f"{name}={len(buckets[name])}" for name in categories) + f"  total={len(normalized)}",
    ]
    if current_units:
        lines.extend(
            (
                "",
                "Current wave:",
                f"  successful={current_counts['successful']}/{len(current_units)}",
                f"  running={current_counts['running']}",
                f"  scheduler_pending={current_counts['scheduler_pending']}",
                f"  license_blocked={current_counts['license_blocked']}",
                f"  elapsed={_text(status.get('current_wave_elapsed'))}",
                f"  last_progress_timestamp={_text(status.get('last_progress_timestamp'))}",
                f"  eta={_text(status.get('eta'))}",
            )
        )
    detail_limit = _MAX_ACTIONABLE_DETAILS if max_active_cases is None else max_active_cases
    _actionable_section(lines, "Running work units", buckets["running"], maximum=max_active_cases, renderer=_active_case_lines)
    _actionable_section(lines, "Scheduler-pending work units", buckets["scheduler_pending"], maximum=None, renderer=_scheduler_pending_lines)
    _actionable_section(lines, "License-blocked work units", buckets["license_blocked"], maximum=detail_limit, renderer=_license_blocked_lines)
    _actionable_section(lines, "Failed work units", buckets["failed"], maximum=detail_limit, renderer=_failed_case_lines)
    _actionable_section(lines, "Completed work units", buckets["successful"], maximum=None, renderer=_completed_case_lines)
    if buckets["not_admitted"]:
        _append_section(lines, "Not admitted", _grouped_population_rows(buckets["not_admitted"]))
    partial = status.get("partial_evaluation")
    if isinstance(partial, dict):
        lines.extend(("", "Partial wave evaluation (provisional):"))
        variants = partial.get("variants")
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                lines.extend(
                    (
                        f"  {variant.get('variant_id')}:",
                        f"    cores_per_case={_text(variant.get('cores_per_case'))}",
                        f"    successful_measurement_count={_text(variant.get('successful_measurement_count'))}",
                        f"    median_comsol_process_seconds={_text(variant.get('median_comsol_process_seconds'))}",
                        f"    minimum_comsol_process_seconds={_text(variant.get('minimum_comsol_process_seconds'))}",
                        f"    maximum_comsol_process_seconds={_text(variant.get('maximum_comsol_process_seconds'))}",
                        f"    median_core_hours_per_case={_text(variant.get('median_core_hours_per_case'))}",
                        f"    estimated_cases_per_node={_text(variant.get('estimated_cases_per_node'))}",
                        f"    estimated_compute_only_cases_per_node_hour={_text(variant.get('estimated_cases_per_node_hour'))}",
                        f"    scheduler_queue_seconds={_text(variant.get('scheduler_queue_seconds'))}",
                        f"    license_wait_seconds={_text(variant.get('license_wait_seconds'))}",
                        f"    license_probe_seconds={_text(variant.get('license_probe_seconds'))}",
                        f"    observed_solver_concurrency={_text(variant.get('observed_peak_solver_concurrency'))}",
                        f"    peak_memory_per_case_bytes={_text(variant.get('peak_memory_per_case_bytes'))}",
                        f"    peak_scratch_per_case_bytes={_text(variant.get('peak_scratch_per_case_bytes'))}",
                    )
                )
        lines.append(f"  provisional_recommendation={_text(partial.get('recommended_cores_per_case'))}")
    final_summary = status.get("final_summary")
    if isinstance(final_summary, dict):
        markdown = final_summary.get("markdown")
        if isinstance(markdown, str) and markdown:
            lines.extend(("", f"Final validated benchmark summary ({_text(final_summary.get('path'))}):", markdown.rstrip()))
        else:
            lines.extend(
                (
                    "",
                    f"Final benchmark summary: {_text(final_summary.get('path'))}",
                    f"  fastest_single_case_cores={_text(final_summary.get('fastest_single_case_cores'))}",
                    f"  lowest_core_hours_cores={_text(final_summary.get('lowest_core_hours_cores'))}",
                    f"  recommended_cores_per_case={_text(final_summary.get('recommended_cores_per_case'))}",
                    f"  estimated_cases_per_node={_text(final_summary.get('estimated_cases_per_node'))}",
                    (f"  estimated_compute_only_cases_per_node_hour={_text(final_summary.get('estimated_compute_only_cases_per_node_hour'))}"),
                    f"  license_qualification={_text(final_summary.get('license_qualification'))}",
                )
            )
    return "\n".join(lines)


def format_benchmark_monitor(status: Mapping[str, Any], *, max_active_cases: int) -> str:
    """Format a rate-limit header and shared benchmark summary."""
    raw_units = status.get("work_units")
    units = raw_units if isinstance(raw_units, list) else []
    normalized = [
        {
            **unit,
            "case_id": unit.get("work_unit_id"),
            "batch_name": unit.get("variant_id"),
        }
        for unit in units
        if isinstance(unit, dict)
    ]
    common_status = {"campaign_state": status.get("state"), "cases": normalized}
    state_signature, progress_signature = campaign_monitor_signatures(common_status)
    benchmark_state = common.serialization.canonical_json_sha256(
        {
            "common_state_signature": state_signature,
            "current_wave": status.get("current_wave"),
            "waves": status.get("waves"),
            "partial_evaluation": status.get("partial_evaluation"),
        }
    )
    header = "\t".join((_MONITOR_RECORD_KIND, _text(status.get("state")), benchmark_state, progress_signature))
    return f"{header}\n{format_benchmark_status_summary(status, max_active_cases=max_active_cases)}"
