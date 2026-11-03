"""Interpret core benchmark sweep measurements and render stable reports.

Responsibilities:
  - Calculate measured node throughput, case times, and memory per sweep and core-count variant
  - Compare variants within each sweep (one node) and across sweeps, and recommend a core count
  - Serialize per-work-unit CSV evidence and render the benchmark Markdown conclusion

Design principles:
  - Throughput is measured from step receipts of one node job, never extrapolated from single cases
  - Failed candidates cost node time inside the makespan, so failures are priced into throughput, not excluded
  - A variant is recommendable only when it reached the success target in every sweep and its node job fits the memory limit
  - Upper bounds are labelled as bounds; unavailable values stay None

This module does NOT:
  - Submit jobs or validate persisted evidence chains
  - Publish or repair report artifacts

Notes:
  A measurement is one variant attempt of one sweep: its candidates ran as steps of one node job with floor(cores/k)
  cases at a time until the success target was reached (a failed candidate is replaced by the next one; the rest
  are skipped). Makespan is the time from the first step start to the last step end of the executed candidates;
  measured throughput is successful cases per node-hour over that makespan.

"""

from __future__ import annotations

import csv
import statistics
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Final

from src.generation.cases import generation_cases_config as config_service

from . import generation_benchmark_config as benchmark_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_SUMMARY_SCHEMA_KIND: Final = "generation_core_scaling_summary"
WORK_UNIT_TIMING_FIELDS: Final = frozenset(
    {
        "scheduler_queue_seconds",
        "license_wait_seconds",
        "license_probe_seconds",
        "canonical_input_preparation_seconds",
        "comsol_process_seconds",
        "export_conversion_seconds",
        "publication_seconds",
        "total_controller_elapsed_seconds",
    }
)
MAX_BENCHMARK_LOG_EXCERPT_BYTES: Final = 8 * 1024
SUMMARY_METRIC_FIELDS: Final = (
    "suite_name",
    "suite_digest",
    "benchmark_mode",
    "case_pool_size",
    "target_successful_cases",
    "cores_per_node",
    "sweeps",
    "measurements",
    "variants",
    "recommended_cores_per_case",
    "recommended_throughput_cases_per_node_hour",
    "recommendation_basis",
    "production_interpretation",
    "production_configuration_modified",
    "dataset_membership",
    "complete",
)
_SECONDS_PER_HOUR: Final = 3600.0
_CSV_FIELDS: Final = (
    "sweep_id",
    "variant_id",
    "cores_per_case",
    "variant_attempt",
    "case_role",
    "work_unit_id",
    "outcome",
    "node",
    "cpu_model",
    "slurm_job_id",
    "slurm_step_id",
    "step_started_at",
    "step_ended_at",
    "case_wall_seconds",
    "comsol_process_seconds",
    "peak_memory_bytes",
    "step_max_rss_bytes",
    "error_type",
)


def parse_benchmark_timestamp(value: object, *, label: str) -> datetime:
    """
    Parse one timezone-aware benchmark scheduler timestamp.

    Parameters
    ----------
    value : object
        ISO timestamp value from benchmark evidence.
    label : str
        Evidence field name used in validation errors.

    Returns
    -------
    datetime
        Timestamp with an explicit timezone.

    """
    if not isinstance(value, str):
        message = f"{label} must be one ISO timestamp."
        raise TypeError(message)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        message = f"{label} must be one ISO timestamp."
        raise ValueError(message) from error
    if parsed.tzinfo is None:
        message = f"{label} must include a timezone."
        raise ValueError(message)
    return parsed


def _production_interpretation(suite: benchmark_config.CoreBenchmarkSuite) -> dict[str, Any]:
    """Resolve the current production core count and its authoritative owner."""
    campaign = config_service.load_campaign_config(
        suite.production_campaign_path,
        require_executable=False,
    )
    cluster = campaign.execution_values["cluster"]
    return {
        "campaign_config": benchmark_config.repository_relative(suite.production_campaign_path),
        "campaign_total_cases": campaign.total_case_count,
        "current_production_cores_per_case": suite.production_cores_per_case,
        "current_concurrent_cases_per_node": cluster["concurrent_cases_per_node"],
        "current_node_jobs": cluster["node_jobs"],
        "cores_config": benchmark_config.repository_relative(suite.production_cores_config_path),
        "cores_key": suite.production_cores_key,
    }


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _unit_wall_seconds(unit: Mapping[str, Any]) -> float | None:
    """Return the step wall time of one unit from its worker receipts, or None when a receipt is missing."""
    if not unit.get("started_at") or not unit.get("ended_at"):
        return None
    started = parse_benchmark_timestamp(unit["started_at"], label="step started_at")
    ended = parse_benchmark_timestamp(unit["ended_at"], label="step ended_at")
    return (ended - started).total_seconds()


def _unit_peak_bytes(unit: Mapping[str, Any]) -> int | None:
    """Return one case's peak memory: the step's sacct MaxRSS, else the worker's child peak, else None (unknown)."""
    if unit.get("step_max_rss_bytes") is not None:
        return int(unit["step_max_rss_bytes"])
    record = unit.get("record") or {}
    peak = (record.get("resource") or {}).get("peak_memory_bytes")
    return None if peak is None else int(peak)


def _measurement_metrics(suite: benchmark_config.CoreBenchmarkSuite, measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Return the metrics of one sweep variant measurement."""
    variant = suite.variant(str(measurement["variant_id"]))
    units = [unit for unit in measurement["units"] if unit["outcome"] != "skipped"]
    successes = [unit for unit in units if unit["outcome"] == "success"]
    starts = [parse_benchmark_timestamp(unit["started_at"], label="step started_at") for unit in units if unit.get("started_at")]
    ends = [parse_benchmark_timestamp(unit["ended_at"], label="step ended_at") for unit in units if unit.get("ended_at")]
    makespan = (max(ends) - min(starts)).total_seconds() if starts and ends and len(ends) == len(units) else None
    walls = [value for unit in units if (value := _unit_wall_seconds(unit)) is not None]
    comsol = [float(unit["record"]["timings_seconds"]["comsol_process_seconds"]) for unit in successes]
    peaks = [_unit_peak_bytes(unit) for unit in units]
    known_peaks = [value for value in peaks if value is not None]
    concurrency = suite.resources(variant).concurrent_cases
    return {
        "sweep_id": measurement["sweep_id"],
        "order": measurement["order"],
        "variant_id": variant.variant_id,
        "cores_per_case": variant.cores_per_case,
        "variant_attempt": measurement["variant_attempt"],
        "concurrent_cases": concurrency,
        "node_job_id": measurement["node_job_id"],
        "node": measurement["node"],
        "cpu_model": measurement["cpu_model"],
        "foreign_jobs_at_start": measurement["foreign_jobs_at_start"],
        "foreign_jobs_at_end": measurement["foreign_jobs_at_end"],
        "cases": len(units),
        "successful_cases": len(successes),
        "reached_target": len(successes) >= suite.target_successful_cases,
        "failed_case_roles": sorted(str(unit["case_role"]) for unit in units if unit["outcome"] != "success"),
        "makespan_seconds": makespan,
        "throughput_cases_per_node_hour": (len(successes) * _SECONDS_PER_HOUR / makespan if makespan and makespan > 0.0 else None),
        "median_case_wall_seconds": _median(walls),
        "max_case_wall_seconds": max(walls) if walls else None,
        "median_comsol_process_seconds": _median(comsol),
        "max_comsol_process_seconds": max(comsol) if comsol else None,
        "max_case_peak_memory_bytes": max(known_peaks) if known_peaks and len(known_peaks) == len(units) else None,
        "node_peak_memory_upper_bound_bytes": concurrency * max(known_peaks) if known_peaks and len(known_peaks) == len(units) else None,
        "node_job_memory": suite.resources(variant).node_job_memory,
    }


def summarize_core_benchmark_results(
    suite: benchmark_config.CoreBenchmarkSuite,
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Summarize measured sweep variants and recommend the core count with the highest node throughput.

    Parameters
    ----------
    suite : CoreBenchmarkSuite
        Resolved suite the measurements belong to.
    measurements : Sequence[Mapping[str, Any]]
        One entry per measured sweep variant: identity, node evidence, and its units with outcome, step
        receipt times, and validated result record (or None for a step that ended without evidence).

    Returns
    -------
    dict[str, Any]
        Deterministic metrics, per-variant comparison across sweeps, and the recommendation.

    """
    measured = [_measurement_metrics(suite, measurement) for measurement in measurements]
    # A case that failed in every measurement fails for case-intrinsic reasons and disqualifies no single variant.
    shared_failures = set.intersection(*(set(row["failed_case_roles"]) for row in measured)) if measured else set()
    variants: list[dict[str, Any]] = []
    for variant in sorted(suite.variants, key=lambda item: item.cores_per_case):
        rows = [row for row in measured if row["variant_id"] == variant.variant_id]
        throughputs = [float(row["throughput_cases_per_node_hour"]) for row in rows if row["throughput_cases_per_node_hour"] is not None]
        failed = sorted({role for row in rows for role in row["failed_case_roles"]})
        memory_fits = config_service.parse_slurm_memory_mib(suite.resources(variant).node_job_memory, label="node_job_memory") <= (
            config_service.parse_slurm_memory_mib(suite.node_memory_limit, label="node_memory_limit")
        )
        variants.append(
            {
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
                "concurrent_cases": suite.resources(variant).concurrent_cases,
                "measured_sweeps": [row["sweep_id"] for row in rows],
                "throughput_by_sweep": {row["sweep_id"]: row["throughput_cases_per_node_hour"] for row in rows},
                "mean_throughput_cases_per_node_hour": statistics.fmean(throughputs) if throughputs else None,
                "failed_case_roles": failed,
                "variant_specific_failed_case_roles": sorted(set(failed) - shared_failures),
                "recommendable": (
                    bool(rows)
                    and len(rows) == len(suite.sweeps)
                    and memory_fits
                    and len(throughputs) == len(rows)
                    and all(row["reached_target"] for row in rows)
                ),
                "production": variant.cores_per_case == suite.production_cores_per_case,
            }
        )
    candidates = [item for item in variants if item["recommendable"]]
    best = max(candidates, key=lambda item: (item["mean_throughput_cases_per_node_hour"], -item["cores_per_case"]), default=None)
    complete = len(measured) == len(suite.sweeps) * len(suite.variants)
    return {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": benchmark_config.BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "benchmark_mode": "core_selection",
        "case_pool_size": suite.representative_case_count,
        "target_successful_cases": suite.target_successful_cases,
        "cores_per_node": suite.cores_per_node,
        "sweeps": [
            {"sweep_id": sweep.sweep_id, "order": sweep.order, "variant_order": [variant.variant_id for variant in suite.sweep_variants(sweep)]}
            for sweep in suite.sweeps
        ],
        "measurements": measured,
        "variants": variants,
        "recommended_cores_per_case": None if best is None else best["cores_per_case"],
        "recommended_throughput_cases_per_node_hour": None if best is None else best["mean_throughput_cases_per_node_hour"],
        "recommendation_basis": (
            "highest mean measured node throughput (successful cases per node-hour, one node job per sweep; failed candidates "
            "were replaced and their node time counts) among variants that reached the success target in every sweep "
            "within the node memory limit; ties prefer fewer cores"
        ),
        "production_interpretation": _production_interpretation(suite),
        "production_configuration_modified": False,
        "dataset_membership": "none",
        "complete": complete,
    }


def results_csv(measurements: Sequence[Mapping[str, Any]]) -> str:
    """Return one CSV row per work unit of every measured sweep variant."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(_CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for measurement in measurements:
        for unit in measurement["units"]:
            record = unit.get("record") or {}
            resource = record.get("resource") or {}
            timings = record.get("timings_seconds") or {}
            error = record.get("error") or {}
            writer.writerow(
                {
                    "sweep_id": measurement["sweep_id"],
                    "variant_id": measurement["variant_id"],
                    "cores_per_case": measurement["cores_per_case"],
                    "variant_attempt": measurement["variant_attempt"],
                    "case_role": unit["case_role"],
                    "work_unit_id": unit["work_unit_id"],
                    "outcome": unit["outcome"],
                    "node": resource.get("node", measurement["node"]),
                    "cpu_model": resource.get("cpu_model", measurement["cpu_model"]),
                    "slurm_job_id": resource.get("slurm_job_id", measurement["node_job_id"]),
                    "slurm_step_id": resource.get("slurm_step_id"),
                    "step_started_at": unit.get("started_at"),
                    "step_ended_at": unit.get("ended_at"),
                    "case_wall_seconds": _unit_wall_seconds(unit),
                    "comsol_process_seconds": timings.get("comsol_process_seconds"),
                    "peak_memory_bytes": resource.get("peak_memory_bytes"),
                    "step_max_rss_bytes": unit.get("step_max_rss_bytes"),
                    "error_type": error.get("type"),
                }
            )
    return buffer.getvalue()


def _format_metric(value: object) -> str:
    """Return one stable Markdown cell."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def core_benchmark_markdown(summary: Mapping[str, Any]) -> str:
    """Render the benchmark conclusion: per-sweep measurements, per-variant comparison, and the recommendation."""
    lines = [
        f"# Core benchmark: {summary['suite_name']}",
        "",
        (
            f"Success target per variant: {summary['target_successful_cases']} of a {summary['case_pool_size']}-case pool "
            f"(failed candidates are replaced); node cores: {summary['cores_per_node']}; complete: {summary['complete']}."
        ),
        "",
        "## Measurements (one node job per sweep, variants in sweep order on the same node)",
        "",
        "| sweep | k | concurrent | node | CPU | successful/tried | makespan s | cases per node-hour | median case s | max case s "
        "| max case peak GiB | node peak bound GiB | failed |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["measurements"]:
        peak = row["max_case_peak_memory_bytes"]
        bound = row["node_peak_memory_upper_bound_bytes"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["sweep_id"]),
                    str(row["cores_per_case"]),
                    str(row["concurrent_cases"]),
                    _format_metric(row["node"]),
                    _format_metric(row["cpu_model"]),
                    f"{row['successful_cases']}/{row['cases']}",
                    _format_metric(row["makespan_seconds"]),
                    _format_metric(row["throughput_cases_per_node_hour"]),
                    _format_metric(row["median_case_wall_seconds"]),
                    _format_metric(row["max_case_wall_seconds"]),
                    _format_metric(None if peak is None else peak / 2**30),
                    _format_metric(None if bound is None else bound / 2**30),
                    ", ".join(row["failed_case_roles"]) or "none",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Variants across sweeps",
            "",
            "| k | concurrent | throughput by sweep | mean cases per node-hour | recommendable | production |",
            "| ---: | ---: | --- | ---: | --- | --- |",
        )
    )
    for variant in summary["variants"]:
        by_sweep = ", ".join(f"{sweep}: {_format_metric(value)}" for sweep, value in variant["throughput_by_sweep"].items()) or "unavailable"
        lines.append(
            f"| {variant['cores_per_case']} | {variant['concurrent_cases']} | {by_sweep} | "
            f"{_format_metric(variant['mean_throughput_cases_per_node_hour'])} | {variant['recommendable']} | {variant['production']} |"
        )
    lines.extend(
        (
            "",
            "## Recommendation",
            "",
            f"Recommended cores_per_case: {_format_metric(summary['recommended_cores_per_case'])} "
            f"({_format_metric(summary['recommended_throughput_cases_per_node_hour'])} cases per node-hour).",
            f"Basis: {summary['recommendation_basis']}.",
            f"Current production cores_per_case: {summary['production_interpretation']['current_production_cores_per_case']}; "
            "the benchmark does not change the production configuration.",
            "The node peak memory column is an upper bound (concurrent cases times the largest single-case peak).",
            "Variant-specific failed roles (failing in some but not all measurements) per k: "
            + (
                "; ".join(
                    f"{variant['cores_per_case']}: {', '.join(variant['variant_specific_failed_case_roles']) or 'none'}"
                    for variant in summary["variants"]
                )
                or "none"
            )
            + ".",
            "",
        )
    )
    return "\n".join(lines)
