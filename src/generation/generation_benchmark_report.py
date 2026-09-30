"""
generation_benchmark_report.py

Interpret core benchmark measurements and render stable reports.

Responsibilities:
  - Calculate runtime, throughput, resource, and license metrics
  - Serialize per-work-unit CSV evidence
  - Render the benchmark Markdown conclusion

Design principles:
  - Keep compute ranking separate from operational waits
  - Preserve report field order and persisted payload semantics
  - Consume immutable suite configuration without lifecycle imports

This module does NOT:
  - Submit jobs or validate persisted evidence chains
  - Publish or repair report artifacts
"""

from __future__ import annotations

import csv
import math
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
    """Resolve current production count and authoritative core-setting owner."""
    campaign = config_service.load_campaign_config(
        suite.production_campaign_path,
        require_executable=False,
    )
    execution = benchmark_config.load_yaml_mapping(
        suite.production_cores_config_path,
        label="benchmark production execution config",
    )
    cluster = benchmark_config.require_mapping(execution.get("cluster"), label="benchmark production execution cluster")
    cores = benchmark_config.require_positive_integer(
        cluster.get("cores_per_case"),
        label="benchmark production cores_per_case",
    )
    if cores != campaign.execution_values["cluster"]["cores_per_case"]:
        message = "Current production campaign and core-setting config disagree."
        raise ValueError(message)
    return {
        "campaign_config": benchmark_config.repository_relative(suite.production_campaign_path),
        "campaign_total_cases": campaign.total_case_count,
        "current_production_cores_per_case": cores,
        "current_estimated_cases_per_node": suite.cores_per_node // cores,
        "current_max_running_cases": campaign.execution_values["submission"]["max_running_cases"],
        "cores_config": benchmark_config.repository_relative(suite.production_cores_config_path),
        "cores_key": suite.production_cores_key,
    }


def _solver_overlap_metrics(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, bool]:
    """Return peak successful solver concurrency and two-case overlap."""
    intervals = [
        (
            parse_benchmark_timestamp(record["solver_interval"]["started_at"], label="solver started_at"),
            parse_benchmark_timestamp(record["solver_interval"]["ended_at"], label="solver ended_at"),
        )
        for record in records
    ]
    if not intervals:
        return 0, False
    peak = max(sum(start <= instant < end for start, end in intervals) for instant in (start for start, _end in intervals))
    overlapped = len(intervals) == len(benchmark_config.BENCHMARK_REPRESENTATIVE_CASE_ROLES) and max(intervals[0][0], intervals[1][0]) < min(
        intervals[0][1], intervals[1][1]
    )
    return max(1, peak), overlapped


def _projected_resource_feasibility(
    estimate: int,
    limit: int | None,
) -> str:
    """Classify one projected node resource against an authoritative limit."""
    if limit is None:
        return "operator_review_required"
    return "pass" if estimate <= limit else "fail"


def _variant_resource_feasibility(memory: str, scratch: str) -> str:
    """Combine independent memory and scratch feasibility classifications."""
    if "fail" in {memory, scratch}:
        return "fail"
    if "operator_review_required" in {memory, scratch}:
        return "operator_review_required"
    return "pass"


def _ordered_unique_text(values: Sequence[object]) -> list[str]:
    """Return non-empty text values once in stable encounter order."""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value in result:
            continue
        result.append(value)
    return result


def summarize_core_benchmark_results(
    suite: benchmark_config.CoreBenchmarkSuite,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Calculate separated runtime, throughput, resource, and license metrics.

    Parameters
    ----------
    suite : benchmark_config.CoreBenchmarkSuite
        Immutable suite defining the measured variants and cases.
    records : Sequence[Mapping[str, Any]]
        One validated result record for every suite work unit.

    Returns
    -------
    dict[str, Any]
        Stable summary payload with compute-only recommendations.

    """
    expected_count = len(suite.variants) * suite.representative_case_count
    if len(records) != expected_count:
        message = f"Benchmark summary requires {expected_count} work-unit records, got {len(records)}."
        raise ValueError(message)
    production = _production_interpretation(suite)
    by_variant: list[dict[str, Any]] = []
    for variant in suite.variants:
        selected = [record for record in records if record.get("variant_id") == variant.variant_id]
        if len(selected) != suite.representative_case_count:
            message = f"Benchmark records do not cover both cases for {variant.variant_id!r}."
            raise ValueError(message)
        successes = [record for record in selected if record.get("status") == "success"]
        failures = [record for record in selected if record.get("status") == "failed"]
        pending = [record for record in selected if record.get("status") == "pending"]
        solve_times = [float(record["timings_seconds"]["comsol_process_seconds"]) for record in successes]
        if any(not math.isfinite(value) or value <= 0.0 for value in solve_times):
            message = f"Benchmark summary received invalid successful COMSOL timings for {variant.variant_id!r}."
            raise ValueError(message)
        queue_values = [record["timings_seconds"]["scheduler_queue_seconds"] for record in successes]
        queue_wait = None if any(value is None for value in queue_values) else sum(float(value) for value in queue_values)
        license_wait = sum(float(record["timings_seconds"]["license_wait_seconds"]) for record in successes)
        license_probe = sum(float(record["timings_seconds"]["license_probe_seconds"]) for record in successes)
        conversion = sum(float(record["timings_seconds"]["export_conversion_seconds"]) for record in successes)
        publication = sum(float(record["timings_seconds"]["publication_seconds"]) for record in successes)
        preparation = sum(float(record["timings_seconds"]["canonical_input_preparation_seconds"]) for record in successes)
        controller_elapsed = sum(float(record["timings_seconds"]["total_controller_elapsed_seconds"]) for record in successes)
        operational_values = (
            license_wait,
            license_probe,
            conversion,
            publication,
            preparation,
            controller_elapsed,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in operational_values):
            message = f"Benchmark summary received invalid separated timing evidence for {variant.variant_id!r}."
            raise ValueError(message)
        license_records = [record["license"] for record in successes]
        blocked_count = sum(int(value["license_blocked_submission_count"]) for value in license_records)
        resources = [record["resource"] for record in successes]
        peak_memory = max((int(value["peak_memory_bytes"]) for value in resources), default=0)
        peak_scratch = max((int(value["peak_scratch_bytes"]) for value in resources), default=0)
        cases_per_node = suite.cores_per_node // variant.cores_per_case
        estimated_memory = cases_per_node * peak_memory
        estimated_scratch = cases_per_node * peak_scratch
        memory_feasibility = _projected_resource_feasibility(estimated_memory, suite.node_memory_limit_bytes)
        scratch_feasibility = _projected_resource_feasibility(estimated_scratch, suite.node_scratch_limit_bytes)
        resource_feasibility = _variant_resource_feasibility(memory_feasibility, scratch_feasibility)
        concurrency, overlapped = _solver_overlap_metrics(successes)
        if solve_times:
            median_solve = float(statistics.median(solve_times))
            median_core_hours = variant.cores_per_case * median_solve / 3600.0
            node_throughput = cases_per_node * 3600.0 / median_solve
            minimum_solve = min(solve_times)
            maximum_solve = max(solve_times)
        else:
            median_solve = None
            median_core_hours = None
            node_throughput = None
            minimum_solve = None
            maximum_solve = None
        raw_excerpts = _ordered_unique_text([value["raw_excerpt"] for value in license_records])
        by_variant.append(
            {
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "cores_per_case": variant.cores_per_case,
                "successful_measurement_count": len(successes),
                "failed_measurement_count": len(failures),
                "pending_measurement_count": len(pending),
                "individual_comsol_process_seconds": solve_times,
                "median_comsol_process_seconds": median_solve,
                "minimum_comsol_process_seconds": minimum_solve,
                "maximum_comsol_process_seconds": maximum_solve,
                "median_core_hours_per_case": median_core_hours,
                "estimated_cases_per_node": cases_per_node,
                "estimated_cases_per_node_hour": node_throughput,
                "throughput_label": "compute-only estimated node throughput",
                "peak_memory_per_case_bytes": peak_memory,
                "estimated_peak_memory_per_node_bytes": estimated_memory,
                "peak_scratch_per_case_bytes": peak_scratch,
                "estimated_peak_scratch_per_node_bytes": estimated_scratch,
                "memory_feasibility": memory_feasibility,
                "scratch_feasibility": scratch_feasibility,
                "resource_feasibility": resource_feasibility,
                "scheduler_queue_seconds": queue_wait,
                "license_wait_seconds": license_wait,
                "license_probe_seconds": license_probe,
                "canonical_input_preparation_seconds": preparation,
                "export_conversion_seconds": conversion,
                "publication_seconds": publication,
                "total_controller_elapsed_seconds": controller_elapsed,
                "license_blocked_submission_count": blocked_count,
                "detected_features": _ordered_unique_text([value["detected_feature"] for value in license_records]),
                "detected_comsol_flexnet_codes": _ordered_unique_text([value["detected_error_code"] for value in license_records]),
                "matched_signatures": _ordered_unique_text([signature for value in license_records for signature in value["matched_signatures"]]),
                "bounded_raw_excerpts": [value[:MAX_BENCHMARK_LOG_EXCERPT_BYTES] for value in raw_excerpts],
                "observed_peak_solver_concurrency": concurrency,
                "requested_cases_overlapped_in_solver_execution": overlapped,
            }
        )
    complete = [
        record
        for record in by_variant
        if record["successful_measurement_count"] == suite.representative_case_count
        and record["failed_measurement_count"] == 0
        and record["pending_measurement_count"] == 0
    ]
    fastest_measurement = min(
        (
            (runtime, int(record["cores_per_case"]), str(record["variant_id"]))
            for record in complete
            for runtime in record["individual_comsol_process_seconds"]
        ),
        default=None,
    )
    fastest_cores = None if fastest_measurement is None else fastest_measurement[1]
    core_efficient = (
        min(
            complete,
            key=lambda record: (
                float(record["median_core_hours_per_case"]),
                int(record["cores_per_case"]),
            ),
        )
        if complete
        else None
    )
    lowest_core_hours_cores = None if core_efficient is None else int(core_efficient["cores_per_case"])
    feasible = [record for record in complete if record["resource_feasibility"] != "fail"]
    if feasible:
        best_throughput = max(float(record["estimated_cases_per_node_hour"]) for record in feasible)
        throughput_ties = [record for record in feasible if float(record["estimated_cases_per_node_hour"]) >= 0.95 * best_throughput]
        recommended = min(
            throughput_ties,
            key=lambda record: (
                float(record["median_core_hours_per_case"]),
                int(record["cores_per_case"]),
            ),
        )
    else:
        recommended = None
    recommended_cores = None if recommended is None else int(recommended["cores_per_case"])
    recommended_cases_per_node = None if recommended is None else int(recommended["estimated_cases_per_node"])
    incomplete_license_concurrency = any(
        record["license_blocked_submission_count"] > 0 and not record["requested_cases_overlapped_in_solver_execution"] for record in by_variant
    )
    all_overlap_observed = bool(by_variant) and all(record["requested_cases_overlapped_in_solver_execution"] for record in by_variant)
    if incomplete_license_concurrency:
        license_qualification = "compute recommendation valid; concurrent-license observation incomplete"
    elif all_overlap_observed:
        license_qualification = "compute recommendation valid; concurrent-license execution observed"
    else:
        license_qualification = "compute recommendation valid; requested solver overlap not observed"
    production["recommended_difference_from_current_cores_per_case"] = (
        None if recommended_cores is None else recommended_cores - int(production["current_production_cores_per_case"])
    )
    production["recommended_differs_from_current"] = (
        None if recommended_cores is None else recommended_cores != int(production["current_production_cores_per_case"])
    )
    recommended_detail = (
        None
        if recommended is None
        else {
            "variant_id": recommended["variant_id"],
            "cores_per_case": recommended_cores,
            "estimated_cases_per_node": recommended_cases_per_node,
            "estimated_cases_per_node_hour": recommended["estimated_cases_per_node_hour"],
            "median_comsol_process_seconds": recommended["median_comsol_process_seconds"],
            "median_core_hours_per_case": recommended["median_core_hours_per_case"],
            "resource_feasibility": recommended["resource_feasibility"],
            "license_qualification": license_qualification,
            "proposed_configuration": {
                "cores_per_case": recommended_cores,
                "cases_per_node": recommended_cases_per_node,
                "max_running_cases": production["current_max_running_cases"],
            },
            "manual_review_required": True,
        }
    )
    canary = suite.canary_variant()
    return {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": benchmark_config.BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "benchmark_mode": "core_selection",
        "representative_cases": suite.case_selections(),
        "cases_per_variant": suite.representative_case_count,
        "required_successful_measurements": expected_count,
        "cores_per_node": suite.cores_per_node,
        "variants": by_variant,
        "fastest_single_case_cores": fastest_cores,
        "fastest_single_case": (
            None
            if fastest_measurement is None
            else {
                "cores_per_case": fastest_measurement[1],
                "variant_id": fastest_measurement[2],
                "comsol_process_seconds": fastest_measurement[0],
            }
        ),
        "lowest_core_hours_cores": lowest_core_hours_cores,
        "lowest_core_hours": (
            None
            if core_efficient is None
            else {
                "cores_per_case": lowest_core_hours_cores,
                "variant_id": core_efficient["variant_id"],
                "median_core_hours_per_case": core_efficient["median_core_hours_per_case"],
            }
        ),
        "recommended_cores_per_case": recommended_cores,
        "recommended_estimated_cases_per_node": recommended_cases_per_node,
        "recommended_production": recommended_detail,
        "recommendation_basis": (
            "maximize compute-only estimated cases per node-hour among resource-feasible variants; "
            "within 5% prefer lower median core-hours, then fewer cores"
        ),
        "timing_contract": {
            "primary_runtime": "successful comsol_process_seconds only",
            "excluded_from_ranking": [
                "scheduler_queue_seconds",
                "license_wait_seconds",
                "license_probe_seconds",
                "canonical_input_preparation_seconds",
                "export_conversion_seconds",
                "publication_seconds",
                "total_controller_elapsed_seconds",
            ],
            "license_only_attempts_contribute_successful_runtime_observations": 0,
        },
        "resource_limits": {
            "node_memory_limit_bytes": suite.node_memory_limit_bytes,
            "node_scratch_limit_bytes": suite.node_scratch_limit_bytes,
            "missing_limit_policy": "operator_review_required",
        },
        "license_qualification": license_qualification,
        "production_interpretation": production,
        "production_configuration_modified": False,
        "dataset_membership": "none",
        "canary_wave": {
            "variant_id": canary.variant_id,
            "cores_per_case": canary.cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "included_in_final_measurements": True,
            "additional_canary_work_units": 0,
        },
    }


def results_csv(
    records: Sequence[Mapping[str, Any]],
    *,
    queue_by_work_unit: Mapping[str, float | None] | None = None,
) -> str:
    """
    Serialize stable per-work-unit timing, resource, and license evidence.

    Parameters
    ----------
    records : Sequence[Mapping[str, Any]]
        Validated result records in persisted work-unit order.
    queue_by_work_unit : Mapping[str, float | None] | None, optional
        Reconciled scheduler queue time for each work unit.

    Returns
    -------
    str
        Deterministic CSV content with a fixed column order.

    """
    stream = StringIO(newline="")
    fields: tuple[str, ...] = (
        "variant_id",
        "cores_per_case",
        "case_position",
        "case_role",
        "work_unit_id",
        "status",
        "attempt",
        "scheduler_queue_seconds",
        "license_wait_seconds",
        "license_probe_seconds",
        "canonical_input_preparation_seconds",
        "comsol_process_seconds",
        "export_conversion_seconds",
        "publication_seconds",
        "total_controller_elapsed_seconds",
        "core_hours",
        "solver_started_at",
        "solver_ended_at",
        "node",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
        "peak_memory_bytes",
        "peak_scratch_bytes",
        "license_blocked_submission_count",
        "detected_feature",
        "detected_comsol_flexnet_code",
        "matched_signatures",
        "license_raw_excerpt",
        "solver_log_sha256",
        "solver_log_size_bytes",
        "solver_log_excerpt",
        "hdf5_sha256",
        "hdf5_size_bytes",
    )
    writer: csv.DictWriter[str] = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        timings_value = record.get("timings_seconds")
        timings = timings_value if isinstance(timings_value, dict) else {}
        resource_value = record.get("resource")
        resource = resource_value if isinstance(resource_value, dict) else {}
        license_value = record.get("license")
        license_evidence = license_value if isinstance(license_value, dict) else {}
        interval_value = record.get("solver_interval")
        interval = interval_value if isinstance(interval_value, dict) else {}
        solver_log_value = record.get("solver_log")
        solver_log = solver_log_value if isinstance(solver_log_value, dict) else {}
        hdf5_value = record.get("hdf5")
        hdf5 = hdf5_value if isinstance(hdf5_value, dict) else {}
        cores = record.get("cores_per_case")
        solve = timings.get("comsol_process_seconds")
        core_hours = (
            float(solve) * int(cores) / 3600.0
            if isinstance(solve, (int, float)) and not isinstance(solve, bool) and isinstance(cores, int) and not isinstance(cores, bool)
            else None
        )
        writer.writerow(
            {
                "variant_id": record.get("variant_id"),
                "cores_per_case": cores,
                "case_position": record.get("case_position"),
                "case_role": record.get("case_role"),
                "work_unit_id": record.get("work_unit_id"),
                "status": record.get("status"),
                "attempt": record.get("attempt"),
                **{
                    field: (
                        queue_by_work_unit.get(str(record.get("work_unit_id")), timings.get(field))
                        if field == "scheduler_queue_seconds" and queue_by_work_unit is not None
                        else timings.get(field)
                    )
                    for field in WORK_UNIT_TIMING_FIELDS
                },
                "core_hours": core_hours,
                "solver_started_at": interval.get("started_at"),
                "solver_ended_at": interval.get("ended_at"),
                "node": resource.get("node"),
                "partition": resource.get("partition"),
                "requested_cpus": resource.get("requested_cpus"),
                "allocated_cpus": resource.get("allocated_cpus"),
                "comsol_np": resource.get("comsol_np"),
                "slurm_job_id": resource.get("slurm_job_id"),
                "peak_memory_bytes": resource.get("peak_memory_bytes"),
                "peak_scratch_bytes": resource.get("peak_scratch_bytes"),
                "license_blocked_submission_count": license_evidence.get("license_blocked_submission_count"),
                "detected_feature": license_evidence.get("detected_feature"),
                "detected_comsol_flexnet_code": license_evidence.get("detected_error_code"),
                "matched_signatures": ",".join(license_evidence.get("matched_signatures", [])),
                "license_raw_excerpt": license_evidence.get("raw_excerpt"),
                "solver_log_sha256": solver_log.get("sha256"),
                "solver_log_size_bytes": solver_log.get("size_bytes"),
                "solver_log_excerpt": solver_log.get("excerpt"),
                "hdf5_sha256": hdf5.get("sha256"),
                "hdf5_size_bytes": hdf5.get("size_bytes"),
            }
        )
    return stream.getvalue()


def _format_metric(value: object) -> str:
    """Format one optional finite numeric metric compactly for Markdown."""
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"Benchmark metric must be numeric or null, got {value!r}."
        raise TypeError(message)
    return f"{float(value):.6g}"


def core_benchmark_markdown(summary: Mapping[str, Any]) -> str:
    """
    Render the fast core-selection evidence without mixing operational waits.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Validated benchmark summary payload.

    Returns
    -------
    str
        Deterministic Markdown report content.

    """
    lines = [
        f"# Core-selection benchmark: {summary['suite_name']}",
        "",
        "Two deterministic scientific cases are measured concurrently in each of four sequential core-count waves.",
        "The first production-core wave is both the canary and two final measurements; there is no additional canary or second phase.",
        "Queue, license wait, and license-probe time are reported separately and do not affect compute ranking.",
        "",
        (
            "| cores | successes | median COMSOL (s) | min (s) | max (s) | "
            "core-hours/case | cases/node | cases/node-hour | queue (s) | license wait (s) | "
            "memory/case (bytes) | scratch/case (bytes) | peak solver concurrency | overlap | feasibility |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for record in summary["variants"]:
        values = (
            record["cores_per_case"],
            record["successful_measurement_count"],
            _format_metric(record["median_comsol_process_seconds"]),
            _format_metric(record["minimum_comsol_process_seconds"]),
            _format_metric(record["maximum_comsol_process_seconds"]),
            _format_metric(record["median_core_hours_per_case"]),
            record["estimated_cases_per_node"],
            _format_metric(record["estimated_cases_per_node_hour"]),
            _format_metric(record["scheduler_queue_seconds"]),
            _format_metric(record["license_wait_seconds"]),
            record["peak_memory_per_case_bytes"],
            record["peak_scratch_per_case_bytes"],
            record["observed_peak_solver_concurrency"],
            "yes" if record["requested_cases_overlapped_in_solver_execution"] else "no",
            record["resource_feasibility"],
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    recommended = summary["recommended_production"]
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Fastest individual solve: {summary['fastest_single_case_cores']} cores per case.",
            f"- Lowest median core-hours: {summary['lowest_core_hours_cores']} cores per case.",
        ]
    )
    if recommended is None:
        lines.append("- Production recommendation: unavailable until all eight measurements and resource checks are valid.")
    else:
        proposal = recommended["proposed_configuration"]
        lines.extend(
            [
                f"- Compute-based production recommendation: {recommended['cores_per_case']} cores per case.",
                f"- Estimated cases per node: {recommended['estimated_cases_per_node']}.",
                f"- Proposed cores_per_case: {proposal['cores_per_case']}.",
                f"- Proposed cases_per_node: {proposal['cases_per_node']}.",
                f"- Proposed max_running_cases: {proposal['max_running_cases']}.",
                "- Apply only after manual review; no production configuration was edited.",
            ]
        )
    lines.extend(
        [
            f"- License qualification: {summary['license_qualification']}.",
            f"- Basis: {summary['recommendation_basis']}.",
            "- Throughput is a compute-only estimate from per-case solver time, not a fully packed-node measurement.",
            "- Dataset membership: none.",
            "",
        ]
    )
    return chr(10).join(lines)
