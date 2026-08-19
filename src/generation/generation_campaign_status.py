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
        return "successful"
    if state in _FAILED_CASE_STATES:
        return "failed"
    if state in {"running", "active"}:
        return "running"
    if state == "license_blocked":
        return "license_blocked"
    if state in {"scheduler_pending", "pending"}:
        return "scheduler_pending"
    if state == "never_started":
        return "never_started"
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
    """Return compact multiline raw evidence for one running work unit."""
    lines = [_case_heading(case, show_batch=show_batch)]
    runtime = _runtime_view(case)
    if runtime.get("availability") != "available":
        reason = _text(runtime.get("reason"))
        lines.append(f"  phase=unavailable  parser_state={reason}  last_progress_update=unavailable")
        return lines
    phase = _text(runtime.get("phase"))
    parser_state = _text(runtime.get("parser_state"))
    progress = _format_comsol_stage(runtime.get("comsol_progress_percent")) if phase in {"stationary_airflow", "transient_drying"} else "unavailable"
    lines.append(f"  state=running  phase={phase}  parser_state={parser_state}  progress={progress}")
    if phase in {"stationary_airflow", "transient_drying"} and runtime.get("parser_state") == "available":
        lines.append(
            "  "
            f"simulated_time={_format_simulated_time(runtime.get('simulated_time_seconds'))}  "
            f"step={_text(runtime.get('step_index'))}  "
            f"step_size={_format_step_size(runtime.get('step_size_seconds'))}"
        )
    else:
        lines.append("  simulated_time=unavailable  step=unavailable  step_size=unavailable")
    lines.append(
        "  "
        f"Tfail={_text(runtime.get('time_failures'))}  NLfail={_text(runtime.get('nonlinear_failures'))}  "
        f"last_solver_update={_text(runtime.get('last_solver_log_update_at'))}  "
        f"last_progress_update={_text(runtime.get('updated_at'))}  age={_format_age(runtime)}"
    )
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
    buckets = {
        name: [case for case in cases if _case_bucket(case) == name]
        for name in ("successful", "running", "scheduler_pending", "license_blocked", "never_started", "failed")
    }
    submission = status.get("submission_config")
    submission_values = submission if isinstance(submission, dict) else {}
    max_running = submission_values.get("max_running_cases")
    max_running_text = "unlimited" if max_running is None else _text(max_running)
    lines = [
        f"Campaign: {_text(status.get('campaign_run_id'))}",
        f"State: {_text(status.get('campaign_state'))}",
        (f"Execution: commit={_text(status.get('git_commit'))}  config_digest={_text(status.get('execution_config_digest'))}"),
        (
            "Resources: "
            f"cores_per_case={_text(submission_values.get('cores_per_case'))}  "
            f"pending_buffer={_text(submission_values.get('pending_buffer'))}  "
            f"max_running_cases={max_running_text}"
        ),
        (
            "Cases: "
            f"successful={len(buckets['successful'])}  running={len(buckets['running'])}  "
            f"scheduler_pending={len(buckets['scheduler_pending'])}  license_blocked={len(buckets['license_blocked'])}  "
            f"never_started={len(buckets['never_started'])}  failed={len(buckets['failed'])}  total={len(cases)} "
            f"({len(buckets['failed'])} failed)"
        ),
    ]
    show_batch = len({str(case.get("batch_name")) for case in cases}) > 1
    active = buckets["running"]
    visible_active = active if max_active_cases is None else active[:max_active_cases]
    active_lines: list[str] = []
    for index, case in enumerate(visible_active):
        if index:
            active_lines.append("")
        active_lines.extend(_active_case_lines(case, show_batch=show_batch))
    omitted = len(active) - len(visible_active)
    if omitted:
        active_lines.append(f"... {omitted} additional active case(s) omitted")
    _append_section(lines, "Running cases", active_lines)
    license_blocked_records = [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["license_blocked"]]
    pending_records = [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["scheduler_pending"]]
    never_started_records = [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["never_started"]]
    _append_section(lines, "License-blocked cases", license_blocked_records)
    _append_section(lines, "Scheduler-pending cases", pending_records)
    _append_section(lines, "Never-started cases", never_started_records)
    _append_section(
        lines,
        "Failed cases",
        [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["failed"]],
    )
    _append_section(
        lines,
        "Completed cases",
        [_nonactive_case_line(case, show_batch=show_batch) for case in buckets["successful"]],
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


def format_benchmark_status_summary(
    status: Mapping[str, Any],
    *,
    max_active_cases: int | None = None,
) -> str:
    """Format benchmark status through the common work-unit presentation model."""
    raw_units = status.get("work_units")
    units = raw_units if isinstance(raw_units, list) and all(isinstance(item, dict) for item in raw_units) else []
    normalized = [
        {
            **unit,
            "case_id": f"{unit.get('variant_id', 'unavailable')} {unit.get('case_role', 'unavailable')}",
            "batch_name": str(unit.get("variant_id", "unavailable")),
        }
        for unit in units
    ]
    categories = ("successful", "running", "scheduler_pending", "license_blocked", "never_started", "failed")
    counts = {name: sum(unit.get("state") == name for unit in normalized) for name in categories}
    wave = status.get("current_wave")
    wave_values = wave if isinstance(wave, dict) else {}
    current_variant = wave_values.get("variant_id")
    current_units = [unit for unit in normalized if unit.get("variant_id") == current_variant]
    current_counts = {name: sum(unit.get("state") == name for unit in current_units) for name in categories}
    lines = [
        f"Benchmark: {_text(status.get('suite_name'))}",
        f"Run: {_text(status.get('benchmark_run_id'))}",
        f"State: {_text(status.get('state'))}",
        f"Wave: {_text(wave_values.get('wave_position'))}/{_text(status.get('wave_count'))}",
        f"Current cores_per_case: {_text(wave_values.get('cores_per_case'))}",
        f"Measurements: {counts['successful']}/{len(normalized)} successful",
        "Work units: " + "  ".join(f"{name}={counts[name]}" for name in categories) + f"  total={len(normalized)}",
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
    running = [unit for unit in normalized if unit.get("state") == "running"]
    visible_running = running if max_active_cases is None else running[:max_active_cases]
    detail: list[str] = []
    for index, unit in enumerate(visible_running):
        if index:
            detail.append("")
        detail.extend(_active_case_lines(unit, show_batch=False))
    omitted = len(running) - len(visible_running)
    if omitted:
        detail.append(f"... {omitted} additional running work unit(s) omitted")
    _append_section(lines, "Running work units", detail)
    wave_order = status.get("waves")
    if isinstance(wave_order, list):
        completed = [f"  {item.get('variant_id')}: complete" for item in wave_order if isinstance(item, dict) and item.get("state") == "complete"]
        future = [
            f"  {item.get('variant_id')}: never_started" for item in wave_order if isinstance(item, dict) and item.get("state") == "never_started"
        ]
        _append_section(lines, "Completed waves", completed)
        _append_section(lines, "Future waves", future)
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
