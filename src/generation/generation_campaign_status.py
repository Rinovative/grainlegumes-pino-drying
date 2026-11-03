"""Render concise human and campaign-monitor views of canonical campaign status.

Responsibilities:
  - Format deterministic per-case scheduler and runtime progress evidence and one line per campaign node job
  - Derive stable state and solver-advancement render signatures
  - Bound automatic active-case output without hiding explicit status detail
  - Render the driver's live board: one line per case, counts, and an estimated remaining time from the status the driver already reads

Design principles:
  - Presentation consumes canonical status without changing campaign decisions
  - Partial solver values remain raw evidence rather than convergence claims
  - Missing monitoring evidence is explicit and non-fatal
  - The board keeps only in-memory observation times, including unverified-failure sightings; estimates say unknown without enough evidence
  - Replacement lineage comes only from the status's exact parent link, never from batch names or case order alone

This module does NOT:
  - Query Slurm, parse COMSOL logs, or validate terminal case success
  - Persist monitoring evidence or authorize publication and cleanup
"""

from __future__ import annotations

import heapq
import math
import re
import statistics
from datetime import UTC, datetime
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from src import common

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_MONITOR_RECORD_KIND = "campaign-monitor"
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
_BOARD_REPEAT_SECONDS = 300  # the board is repeated this often while nothing changed, so a long solve still shows its progress
_BOARD_STALE_SECONDS = 600  # a running case whose solver evidence stayed unchanged this long is marked as stuck
_SLURM_ELAPSED = re.compile(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d{2})\Z")
_WAITING_WORDS = {
    "scheduler_pending": "queued in Slurm",
    "pending": "queued in Slurm",
    "license_blocked": "waiting for a COMSOL license",
    "admission_waiting": "waiting for admission",
    "never_started": "not started yet",
}
# The campaign's reason for a case whose scheduler step ended COMPLETED without result evidence visible to the driver's node.
_UNVERIFIED_FAILURE_REASON = "completed_without_valid_case_evidence"
_VERIFYING_TEXT = "step ended, result not yet visible"


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
    """Return one aggregate presentation bucket for a canonical case state; a case without a state is unknown, not failed."""
    state = str(case.get("state") or "scheduler_unknown")
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
    if state in {"never_started", "skipped"}:
        # A benchmark candidate skipped after its variant reached the success target never ran.
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


def _failure_reason(case: Mapping[str, Any]) -> str | None:
    """Return the bounded reason of one failed work unit in words."""
    raw_reason = case.get("reason")
    if isinstance(raw_reason, str) and raw_reason.startswith(_UNSAFE_CAPACITY_STATUS_PREFIX):
        return "Unowned or unsupported COMSOL status artifact."
    return _bounded_text(raw_reason, maximum=_MAX_REASON_CHARACTERS)


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
    reason = _failure_reason(case)
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
    evidence = _compact_evidence_path(case.get("case_reconciliation_evidence_path") or case.get("evidence_path"))
    if evidence is not None:
        lines.append(f"  evidence={evidence}")
    return lines


def _terminal_value_parts(case: Mapping[str, Any]) -> list[str]:
    """Return the simulated end and the bulk moisture against its target of one successful case, as far as reported."""
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
    return terminal


def _completed_case_lines(case: Mapping[str, Any]) -> list[str]:
    """Return compact successful-case terminal evidence."""
    lines = [_case_heading(case, include_runtime=False)]
    details = ["state=successful"]
    reason = _bounded_text(case.get("reason"), maximum=_MAX_REASON_CHARACTERS)
    if reason is not None:
        details.append(f"reason={reason}")
    lines.append(f"  {'  '.join(details)}")
    terminal = _terminal_value_parts(case)
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
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 -- retain exact persisted timestamp input normalization and rejection behavior
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


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


def _node_job_lines(status: Mapping[str, Any]) -> list[str]:
    """Return one line per node job and one for an unresolved node-job submission."""
    lines: list[str] = []
    jobs = status.get("node_jobs")
    for job in jobs if isinstance(jobs, list) else ():
        parts = [
            f"seat={_text(job.get('seat'))}",
            f"job={_text(job.get('job_id'))}",
            f"state={_text(job.get('state'))}",
        ]
        node = _available_text(job.get("node"))
        if node is not None:
            parts.append(f"node={node}")
        parts.append(f"unresolved={_text(job.get('unresolved_dispatches'))}")
        if job.get("draining") is True:
            parts.append("draining")
        if job.get("cancel_published") is True:
            parts.append("cancel_published")
        if job.get("lease") not in {None, "ok"}:
            parts.append(f"lease={_text(job.get('lease'))}")
        lines.append(f"Node job: {'  '.join(parts)}")
    if status.get("node_job_intent") is not None:
        lines.append(f"Node job submission unresolved: token={_text(status.get('node_job_intent'))}")
    return lines


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
    lines = [
        f"Campaign: {_text(status.get('campaign_run_id'))}",
        f"State: {_text(status.get('campaign_state'))}",
        (f"Execution: commit={_text(status.get('git_commit'))}  config_digest={_text(status.get('execution_config_digest'))}"),
        (
            "Resources: "
            f"cores_per_case={_text(submission_values.get('cores_per_case'))}  "
            f"node_jobs={_text(submission_values.get('node_jobs'))}  "
            f"concurrent_cases_per_node={_text(submission_values.get('concurrent_cases_per_node'))}  "
            f"max_admission_cases={_text(submission_values.get('max_admission_cases'))}"
        ),
        *_node_job_lines(status),
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
    )
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


def _case_signature_parts(case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the state and identity fields and the state plus solver-advancement fields of one case."""
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
        "phase": runtime.get("phase"),
        "terminal": runtime.get("terminal"),
        # A decision or its lineage recorded while another case runs changes what the board shows at once.
        "terminal_decision": case.get("terminal_decision"),
        "failure_reason": case.get("failure_reason"),
        "failure_detail": case.get("failure_detail"),
        "parent_failed_case_id": case.get("parent_failed_case_id"),
    }
    progress = {
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
    return urgent, progress


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
        urgent, progress = _case_signature_parts(case)
        urgent_cases.append(urgent)
        progress_cases.append(progress)
    urgent_payload = {
        "campaign_state": status.get("campaign_state"),
        "admission_blocked": status.get("admission_blocked"),
        "admission_block_reason": status.get("admission_block_reason"),
        "node_jobs": [
            {key: job.get(key) for key in ("seat", "job_id", "state", "draining", "unresolved_dispatches")}
            for job in status.get("node_jobs") or ()
            if isinstance(job, dict)
        ],
        "node_job_intent": status.get("node_job_intent"),
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


def format_duration(seconds: float) -> str:
    """Return a coarse human duration: seconds below a minute, minutes below an hour, else hours and minutes."""
    whole = int(seconds)
    if whole < _SECONDS_PER_MINUTE:
        return f"{whole} s"
    if whole < _SECONDS_PER_HOUR:
        return f"{whole // _SECONDS_PER_MINUTE} min"
    return f"{whole // _SECONDS_PER_HOUR} h {whole % _SECONDS_PER_HOUR // _SECONDS_PER_MINUTE:02d} min"


def _elapsed_seconds(value: object) -> float | None:
    """Return the seconds of one Slurm ``[D-][HH:]MM:SS`` elapsed field."""
    match = _SLURM_ELAPSED.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return float(((days * 24 + hours) * _SECONDS_PER_MINUTE + minutes) * _SECONDS_PER_MINUTE + seconds)


def _case_key(case: Mapping[str, Any]) -> tuple[str, str]:
    """Return the batch and case identity under which the board tracks one case."""
    return str(case.get("batch_name") or _group_label(case)), str(case.get("case_id"))


def _board_group(case: Mapping[str, Any]) -> str:
    """Return the board group of one case: done, running, waiting or failed."""
    bucket = _case_bucket(case)
    if bucket == "running" and case.get("state") != "admission_waiting":
        return "running"
    return {"successful": "done", "failed": "failed"}.get(bucket, "waiting")


def _running_parts(case: Mapping[str, Any]) -> tuple[list[str], float | None]:
    """Return the board fields of one running case and its estimated remaining solving seconds.

    Progress is the solver time past the case's own ``physical_start_hours`` against its ``physical_horizon_hours``,
    both from the status. The estimate extrapolates that progress; a drying case that reaches its target moisture
    earlier ends before the horizon, so the estimate is an upper bound.
    """
    runtime = _runtime_view(case)
    simulated = _float_value(runtime.get("simulated_time_seconds"))
    start_hours = _float_value(case.get("physical_start_hours"))
    horizon_hours = _float_value(case.get("physical_horizon_hours"))
    eta = None
    if runtime.get("availability") != "available":
        parts = [f"no runtime report ({_text(runtime.get('reason'))})"]
    elif runtime.get("parser_state") != "available" or simulated is None or start_hours is None or not horizon_hours:
        # A running case's batch always has its persisted time window, so only the solver phase can be missing here.
        parts = [(_available_text(runtime.get("phase")) or "starting").replace("_", " ")]
    else:
        progressed, horizon_seconds = simulated - start_hours * _SECONDS_PER_HOUR, horizon_hours * _SECONDS_PER_HOUR
        parts = [f"{100.0 * progressed / horizon_seconds:.0f} %", f"{_format_simulated_time(progressed)} / {horizon_hours:g} h"]
        elapsed = _float_value(runtime.get("elapsed_seconds"))
        if progressed > 0 and elapsed is not None:
            eta = max(0.0, elapsed * (horizon_seconds - progressed) / progressed)
        parts.append(f"step {_format_step_size(runtime.get('step_size_seconds'))}")
        parts.append(f"Tfail {_text(runtime.get('time_failures'))}  NLfail {_text(runtime.get('nonlinear_failures'))}")
        parts.append(f"updated {_format_age(runtime)}")
    parts.append("ETA unknown" if eta is None else f"ETA ~{format_duration(eta)}")
    return parts, eta


def _remaining_seconds(running_etas: Sequence[float | None], waiting: int, finished_seconds: Sequence[float], seats: int) -> float | None:
    """Estimate the seconds until every case ended, or ``None`` without evidence for each running or waiting case.

    Running cases keep their seats; each waiting case, taking the median duration of the finished cases, goes whole
    to the seat that frees first, and the estimate is the latest finish.
    """
    known = [eta for eta in running_etas if eta is not None]
    if len(known) < len(running_etas) or (waiting and not finished_seconds):
        return None
    typical = statistics.median(finished_seconds) if waiting else 0.0
    finishes = sorted(known)[-seats:] + [0.0] * max(0, seats - len(known))
    heapq.heapify(finishes)
    for _case in range(waiting):
        heapq.heappush(finishes, heapq.heappop(finishes) + typical)
    return max(finishes)


def _replacements(cases: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """Map each failed case to the replacement that took its slot, from the status's exact ``parent_failed_case_id``.

    Every replacement names its slot's failed primary, so the successor of a failed case is the first later case of
    its material that names the same primary (the failed case itself when it is the primary).
    """
    pairs: dict[tuple[str, str], str] = {}
    for failed in (case for case in cases if _board_group(case) == "failed"):
        material, case_id = _group_label(failed), str(failed.get("case_id"))
        slot = failed.get("parent_failed_case_id") or case_id
        successors = sorted(
            str(case.get("case_id"))
            for case in cases
            if _group_label(case) == material and case.get("parent_failed_case_id") == slot and str(case.get("case_id")) > case_id
        )
        if successors:
            pairs[material, case_id] = successors[0]
    return pairs


def _failed_parts(case: Mapping[str, Any], replaced: Mapping[tuple[str, str], str]) -> list[str]:
    """Return the board fields of one failed case: its typed reason and detail and what became of its slot."""
    code, detail = _available_text(case.get("failure_reason")), _available_text(case.get("failure_detail"))
    typed = detail if code is None or (detail or "").startswith(code) else code if detail is None else f"{code}: {detail}"
    reason = _bounded_text(typed, maximum=_MAX_REASON_CHARACTERS) or _failure_reason(case) or _text(case.get("classified_state")).replace("_", " ")
    replacement = replaced.get((_group_label(case), str(case.get("case_id"))))
    if case.get("terminal_decision") is None:
        return [reason, "no decision recorded"]
    return [reason, f"replaced by {replacement}" if replacement else "decision recorded"]


def _is_unverified(case: Mapping[str, Any]) -> bool:
    """Return whether one case failed for a COMPLETED step whose result evidence may still become visible on another node.

    The status names that classification in ``reason``; a recorded failed decision settles the case, a recorded success
    (the step's own decision, seen before its result files) does not.
    """
    return (
        _board_group(case) == "failed"
        and _available_text(case.get("reason")) == _UNVERIFIED_FAILURE_REASON
        and case.get("terminal_decision") != "failed"
    )


class CampaignBoard:
    """Render the live overview of one running campaign from the status payload the driver reads at every poll."""

    def __init__(self, *, started: float, seats: int, verify_seconds: float) -> None:
        """Observe from clock reading ``started``; ``seats`` is the number of parallel cases.

        ``verify_seconds`` is how long an unverified failure reads as verifying before it reads as failed; the driver passes
        its own failure settle window, so the board and the driver agree on it.
        """
        self._started = started
        self._seats = max(1, seats)
        self._verify_seconds = verify_seconds
        self._signature: str | None = None
        self._printed_at = started
        self._evidence: dict[tuple[str, str], tuple[str, float]] = {}
        self._unverified_since: dict[tuple[str, str], float] = {}
        self._verifying_until: float | None = None

    def observe(self, status: Mapping[str, Any], *, now: float) -> str | None:
        """Record solver-evidence changes and first sightings of unverified failures; return the board when it is due, else ``None``.

        The board is due at the first call, when the state signature of the campaign changed, after the repeat interval, and
        when a case printed as verifying leaves that state because its window expired.
        """
        cases = _case_inventory(status)
        self._unverified_since = {_case_key(case): self._unverified_since.get(_case_key(case), now) for case in cases if _is_unverified(case)}
        if not cases:
            return None
        for case in cases:
            fingerprint = common.serialization.canonical_json_sha256(_case_signature_parts(case)[1])
            key = _case_key(case)
            previous = self._evidence.get(key)
            if previous is None or previous[0] != fingerprint:
                self._evidence[key] = (fingerprint, now)
        signature = campaign_monitor_signatures(status)[0]
        window_closed = self._verifying_until is not None and now >= self._verifying_until
        if signature == self._signature and now - self._printed_at < _BOARD_REPEAT_SECONDS and not window_closed:
            return None
        self._signature, self._printed_at = signature, now
        ends = [since + self._verify_seconds for since in self._unverified_since.values()]
        self._verifying_until = min((end for end in ends if end > now), default=None)
        return self._render(status, cases, now)

    def _group(self, case: Mapping[str, Any], now: float) -> str:
        """Return the board group of one case at ``now``: an unverified failure inside its window is verifying, else its status group."""
        since = self._unverified_since.get(_case_key(case))
        if since is not None and now - since < self._verify_seconds:
            return "verifying"
        return _board_group(case)

    def _render(self, status: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], now: float) -> str:
        """Return the header with the counts and the estimate, the node jobs, and one line per case."""
        grouped = [(case, self._group(case, now)) for case in cases]
        groups = {name: [case for case, group in grouped if group == name] for name in ("done", "running", "waiting", "failed", "verifying")}
        running = {_case_key(case): _running_parts(case) for case in groups["running"]}
        finished = [seconds for case in groups["done"] if (seconds := _elapsed_seconds(case.get("elapsed"))) is not None]
        remaining = _remaining_seconds([eta for _parts, eta in running.values()], len(groups["waiting"]), finished, self._seats)
        active = bool(running or groups["waiting"])
        estimate = "unknown" if remaining is None else f"~{format_duration(remaining)}" if active else "none"
        counts = "  ".join(f"{name} {len(members)}" for name, members in groups.items() if name != "verifying" or members)
        lines = [f"Board  {counts}  of {len(cases)}  elapsed {format_duration(now - self._started)}  remaining {estimate}", *_node_job_lines(status)]
        replaced = _replacements(cases)
        width = max((len(_group_label(case)) for case in cases), default=0)
        for case in sorted(cases, key=lambda item: (_group_label(item), str(item.get("case_id")))):
            group = self._group(case, now)
            key = _case_key(case)
            parts = [_WAITING_WORDS.get(str(case.get("state")), _text(case.get("state")).replace("_", " "))]
            if group == "running":
                parts = list(running[key][0])
                stale = now - self._evidence[key][1]
                if stale > _BOARD_STALE_SECONDS:
                    parts.append(f"no progress for {int(stale // _SECONDS_PER_MINUTE)} min")
            elif group == "done":
                took = _elapsed_seconds(case.get("elapsed"))
                parts = [*_terminal_value_parts(case), *([] if took is None else [f"took {format_duration(took)}"])]
            elif group == "verifying":
                parts = [_VERIFYING_TEXT]
            elif group == "failed":
                parts = _failed_parts(case, replaced)
            lines.append(f"{_group_label(case):<{width}}  {key[1]}  {group:<7}  {'  '.join(parts)}")
        return "\n".join(lines)


def format_benchmark_status_summary(
    status: Mapping[str, Any],
    *,
    max_active_cases: int | None = None,
) -> str:
    """Format benchmark sweep status through the common bounded work-unit model."""
    raw_units = status.get("work_units")
    units = raw_units if isinstance(raw_units, list) and all(isinstance(item, dict) for item in raw_units) else []
    normalized = [
        {
            **unit,
            "case_id": str(unit.get("work_unit_id", f"{unit.get('variant_id', 'unavailable')} {unit.get('case_role', 'unavailable')}")),
            "batch_name": f"{unit.get('sweep_id', 'unavailable')}/{unit.get('variant_id', 'unavailable')}",
        }
        for unit in units
    ]
    categories = _CASE_SUMMARY_LABELS
    buckets = {name: [unit for unit in normalized if _case_bucket(unit) == name] for name in categories}
    lines = [
        f"Benchmark: {_text(status.get('suite_name'))}",
        f"Run: {_text(status.get('benchmark_run_id'))}",
        f"State: {_text(status.get('state'))}",
        f"Measurements: {len(buckets['successful'])}/{len(normalized)} successful",
        "Work units: " + "  ".join(f"{name}={len(buckets[name])}" for name in categories) + f"  total={len(normalized)}",
    ]
    sweeps = status.get("sweeps")
    if isinstance(sweeps, list):
        lines.append("")
        lines.append("Sweeps:")
        for sweep in sweeps:
            if not isinstance(sweep, dict):
                continue
            lines.append(
                f"  {_text(sweep.get('sweep_id'))} ({_text(sweep.get('order'))}): state={_text(sweep.get('state'))} "
                f"variants={_text(sweep.get('measured_variants'))}/{_text(sweep.get('total_variants'))} "
                f"current_cores_per_case={_text(sweep.get('current_cores_per_case'))} "
                f"node_jobs={','.join(str(item) for item in sweep.get('node_job_ids') or []) or 'none'}"
            )
    lines.append(f"Last progress: {_text(status.get('last_progress_timestamp'))}  eta={_text(status.get('eta'))}")
    detail_limit = _MAX_ACTIONABLE_DETAILS if max_active_cases is None else max_active_cases
    _actionable_section(lines, "Running work units", buckets["running"], maximum=max_active_cases, renderer=_active_case_lines)
    _actionable_section(lines, "Queued work units", buckets["scheduler_pending"], maximum=None, renderer=_scheduler_pending_lines)
    _actionable_section(lines, "Failed or interrupted work units", buckets["failed"], maximum=detail_limit, renderer=_failed_case_lines)
    _actionable_section(lines, "Completed work units", buckets["successful"], maximum=None, renderer=_completed_case_lines)
    if buckets["not_admitted"]:
        _append_section(lines, "Not yet dispatched", _grouped_population_rows(buckets["not_admitted"]))
    partial = status.get("partial_evaluation")
    if isinstance(partial, dict):
        lines.extend(("", "Measured sweep variants (provisional):"))
        measurements = partial.get("measurements")
        for row in measurements if isinstance(measurements, list) else []:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  {_text(row.get('sweep_id'))} k={_text(row.get('cores_per_case'))}: node={_text(row.get('node'))} "
                f"successful={_text(row.get('successful_cases'))}/{_text(row.get('cases'))} "
                f"makespan_s={_text(row.get('makespan_seconds'))} cases_per_node_hour={_text(row.get('throughput_cases_per_node_hour'))}"
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
                    f"  recommended_cores_per_case={_text(final_summary.get('recommended_cores_per_case'))}",
                    f"  recommended_throughput_cases_per_node_hour={_text(final_summary.get('recommended_throughput_cases_per_node_hour'))}",
                )
            )
    return "\n".join(lines)


def format_benchmark_monitor(status: Mapping[str, Any], *, max_active_cases: int) -> str:
    """Format a rate-limit header and the shared benchmark summary."""
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
            "sweeps": status.get("sweeps"),
            "partial_evaluation": status.get("partial_evaluation"),
        }
    )
    header = "\t".join((_MONITOR_RECORD_KIND, _text(status.get("state")), benchmark_state, progress_signature))
    return f"{header}\n{format_benchmark_status_summary(status, max_active_cases=max_active_cases)}"
