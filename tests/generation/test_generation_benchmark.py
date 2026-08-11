# ruff: noqa: S101, SLF001, PLR2004
"""Isolated transient core-benchmark identity, execution, and metric contracts."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import common, generation

_COMMIT = "a" * 40
_SMOKE_DIGEST = "b" * 64
_BENCHMARK_RUN_ID = "core_scaling_transient__0123456789abcdef"
_CASE_INPUT_ID = "c" * 64
_SIMULATION_CASE_ID = "d" * 64
_HDF5_SHA256 = "e" * 64


def _repository() -> Path:
    """Return the repository root used by maintained benchmark references."""
    return Path(__file__).resolve().parents[2]


def _suite() -> Any:
    """Load the maintained suite without requiring native COMSOL executables."""
    return generation.benchmark.load_core_benchmark_suite(
        _repository() / "configs/generation/benchmarks/transient_core_scaling/suite.yaml",
        require_executable=False,
    )


def _success_record(suite: Any, variant: Any, repetition: int) -> dict[str, Any]:
    """Return one minimal valid successful-repetition record."""
    return {
        "schema_kind": generation.benchmark.BENCHMARK_RESULT_SCHEMA_KIND,
        "schema_version": generation.benchmark.BENCHMARK_SCHEMA_VERSION,
        "status": "success",
        "benchmark_run_id": _BENCHMARK_RUN_ID,
        "git_commit": _COMMIT,
        "suite_digest": suite.suite_digest,
        "variant_id": variant.variant_id,
        "execution_id": suite.execution_id(variant),
        "repetition": repetition,
        "repetition_id": suite.repetition_id(variant, repetition),
        "cores_per_case": variant.cores_per_case,
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "case_input_id": _CASE_INPUT_ID,
        "simulation_case_id": _SIMULATION_CASE_ID,
        "attempt": 1,
        "resource": {
            "node": "node-a",
            "partition": suite.partition,
            "requested_cpus": variant.cores_per_case,
            "allocated_cpus": variant.cores_per_case,
            "comsol_np": variant.cores_per_case,
            "slurm_job_id": str(1000 + repetition),
        },
        "scheduler_timing": {
            "submit_time": "2026-01-01T00:00:00+00:00",
            "start_time": "2026-01-01T00:00:05+00:00",
            "completion_time": "2026-01-01T00:00:20+00:00",
            "queue_wait_s": 5.0,
            "turnaround_s": 20.0,
        },
        "timings_s": {
            "case_materialization": 0.5,
            "comsol_process": 1.0,
            "export_conversion": 0.25,
            "hdf5_admission": 0.1,
            "complete_case": 1.85,
        },
        "hdf5": {
            "sha256": _HDF5_SHA256,
            "size_bytes": 1024,
            "identity": {
                "case_input_id": _CASE_INPUT_ID,
                "simulation_case_id": _SIMULATION_CASE_ID,
            },
            "retained_as_scientific_case": False,
        },
    }


def test_maintained_suite_is_config_owned_same_case_and_serially_isolated(tmp_path: Path) -> None:
    """Protect four YAML-owned variants, one shared case, and one measured solve at a time."""
    suite = _suite()
    authored = yaml.safe_load(suite.source_path.read_text(encoding="utf-8"))
    assert suite.repetitions == authored["repetitions"]
    assert len(suite.variants) == 4
    assert len({variant.variant_id for variant in suite.variants}) == 4
    assert len({variant.cores_per_case for variant in suite.variants}) == 4
    assert all(0 < variant.cores_per_case <= suite.cores_per_node for variant in suite.variants)
    assert [variant.cores_per_case for variant in suite.variants] == [
        yaml.safe_load(variant.source_path.read_text(encoding="utf-8"))["cores_per_case"] for variant in suite.variants
    ]
    assert suite.case_campaign.dataset_packages == ()
    case = suite.case_selection()
    assert case["assignment"]["pilot_case_kind"] == "nominal_reference"
    assert case["simulation_profile"] == "transient_drying"
    first = suite.variants[0]
    changed_cores = replace(first, cores_per_case=first.cores_per_case + 1)
    assert suite.execution_id(changed_cores) != suite.execution_id(first)
    assert suite.repetition_id(first, 1) != suite.repetition_id(first, 2)
    assert suite.case_selection() == case

    run_id = generation.benchmark.core_benchmark_run_id(
        suite,
        git_commit=_COMMIT,
        smoke_gate_digest=_SMOKE_DIGEST,
    )
    sequence = generation.benchmark._measured_sequence(suite)
    assert [(variant.cores_per_case, repetition) for variant, repetition in sequence] == [
        (variant.cores_per_case, repetition) for repetition in range(1, suite.repetitions + 1) for variant in suite.variants
    ]
    for variant, repetition in sequence:
        command = generation.benchmark.build_core_benchmark_slurm_command(
            suite,
            run_id=run_id,
            storage_root=tmp_path.resolve(),
            log_directory=(tmp_path / "logs").resolve(),
            role="measure",
            variant=variant,
            repetition=repetition,
        )
        assert "--nodes=1" in command
        assert "--ntasks=1" in command
        assert f"--cpus-per-task={variant.cores_per_case}" in command
        assert not any(argument.startswith("--array") for argument in command)
        assert not any(argument.startswith("--dependency") for argument in command)
        assert "--exclusive" not in command
        assert f"--output={(tmp_path / 'logs').resolve()}/slurm-%j.out" in command
        assert variant.variant_id in command[-1]
        assert str(repetition) in command[-1]


def test_benchmark_rejects_nonordinary_scheduler_options() -> None:
    """Prevent benchmark YAML from requesting exclusivity or reservations."""
    for option in ("--exclusive", "--reservation=reserved", "--nodelist=node-a"):
        with pytest.raises(ValueError, match="owned by the launcher"):
            generation.benchmark._scheduler_options([option])
    assert generation.benchmark._scheduler_options(["--qos=shared"]) == ("--qos=shared",)


def test_current_checkout_must_match_persisted_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject resume or execution from a checkout other than the launch commit."""
    manifest = {"git_commit": _COMMIT}
    monkeypatch.setattr(generation.benchmark, "_repository_commit", lambda: _COMMIT)
    generation.benchmark._require_current_checkout(manifest)

    monkeypatch.setattr(generation.benchmark, "_repository_commit", lambda: "f" * 40)
    with pytest.raises(RuntimeError, match="does not match persisted commit"):
        generation.benchmark._require_current_checkout(manifest)


def test_benchmark_timing_uses_scheduler_owned_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the exact Slurm allocation start rather than worker entry time."""
    monkeypatch.setenv("SLURM_JOB_START_TIME", "1767225605")
    assert generation.benchmark._slurm_scheduler_start_time() == "2026-01-01T00:00:05+00:00"

    monkeypatch.delenv("SLURM_JOB_START_TIME")
    with pytest.raises(RuntimeError, match="SLURM_JOB_START_TIME"):
        generation.benchmark._slurm_scheduler_start_time()


def test_all_variants_materialize_identical_scientific_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove resource variants neither resample nor alter transient case inputs."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", _COMMIT)
    suite = _suite()
    bundles = [
        generation.cases.case.generate_case_input_bundle(
            suite.case_config,
            suite.case_index,
            tmp_path / variant.variant_id,
        )
        for variant in suite.variants
    ]
    reference = bundles[0]
    reference_files = {path.name: path.read_bytes() for path in (*reference.input_paths, reference.directory / "case.json")}
    for bundle in bundles[1:]:
        assert bundle.case_input_id == reference.case_input_id
        assert bundle.simulation_case_id == reference.simulation_case_id
        assert bundle.case_payload == reference.case_payload
        assert {path.name: path.read_bytes() for path in (*bundle.input_paths, bundle.directory / "case.json")} == reference_files
    assert {path.name for path in reference.input_paths} == {
        "fields.csv",
        "scalars.csv",
        "schedule.csv",
    }
    assert len({suite.execution_id(variant) for variant in suite.variants}) == 4
    variant = suite.variants[0]
    repetition_ids = {suite.repetition_id(variant, repetition) for repetition in range(1, suite.repetitions + 1)}
    assert len(repetition_ids) == suite.repetitions
    assert all(reference.case_input_id not in repetition_id for repetition_id in repetition_ids)


def test_successful_repetitions_skip_and_failed_attempts_remain_retryable(tmp_path: Path) -> None:
    """Protect immutable success reuse while leaving failed repetitions pending."""
    suite = _suite()
    variant = suite.variants[0]
    directory = tmp_path / "benchmark"
    assert generation.benchmark._pending_repetitions(directory, suite, variant) == tuple(range(1, suite.repetitions + 1))

    first_path = generation.benchmark._success_path(directory, suite, variant, 1)
    first_path.parent.mkdir(parents=True)
    common.serialization.atomic_write_json(first_path, _success_record(suite, variant, 1))
    assert generation.benchmark._pending_repetitions(directory, suite, variant) == tuple(range(2, suite.repetitions + 1))

    second_path = generation.benchmark._success_path(directory, suite, variant, 2)
    second_path.parent.mkdir(parents=True)
    common.serialization.atomic_write_json(
        second_path.parent / "attempt-0001.json",
        {"status": "failed", "error": {"message": "synthetic"}},
    )
    assert 2 in generation.benchmark._pending_repetitions(directory, suite, variant)
    common.serialization.atomic_write_json(second_path, _success_record(suite, variant, 2))
    assert generation.benchmark._pending_repetitions(directory, suite, variant) == tuple(range(3, suite.repetitions + 1))


def test_deterministic_metrics_and_benchmark_namespace_do_not_mutate_production(tmp_path: Path) -> None:
    """Protect aggregate scaling math, recommendation labels, and dataset isolation."""
    maintained = _suite()
    synthetic_cores = (1, 2, 4, 8)
    variants = tuple(replace(variant, cores_per_case=cores) for variant, cores in zip(maintained.variants, synthetic_cores, strict=True))
    suite = replace(maintained, variants=variants, cores_per_node=8)
    solve_times = (
        (10.0, 12.0, 14.0),
        (4.0, 5.0, 6.0),
        (2.0, 3.0, 4.0),
        (1.0, 2.0, 3.0),
    )
    overheads = (3.0, 1.0, 1.0, 1.0)
    records = [
        {
            "status": "success",
            "variant_id": variant.variant_id,
            "timings_s": {
                "comsol_process": solve,
                "complete_case": solve + overhead,
            },
            "scheduler_timing": {
                "queue_wait_s": float(repetition),
                "turnaround_s": solve + overhead + float(repetition),
            },
        }
        for variant, times, overhead in zip(suite.variants, solve_times, overheads, strict=True)
        for repetition, solve in enumerate(times, start=1)
    ]
    summary = generation.benchmark.summarize_core_benchmark_results(suite, records)
    reference = summary["variants"][0]
    second = summary["variants"][1]
    last = summary["variants"][-1]
    assert reference["individual_comsol_wall_s"] == [10.0, 12.0, 14.0]
    assert reference["median_comsol_wall_s"] == 12.0
    assert reference["mean_comsol_wall_s"] == 12.0
    assert reference["minimum_comsol_wall_s"] == 10.0
    assert reference["maximum_comsol_wall_s"] == 14.0
    assert reference["sample_standard_deviation_comsol_wall_s"] == statistics.stdev((10.0, 12.0, 14.0))
    assert reference["median_total_case_wall_s"] == 15.0
    assert reference["speedup"] == 1.0
    assert reference["parallel_efficiency"] == 1.0
    assert math.isclose(second["speedup"], 12.0 / 5.0)
    assert math.isclose(second["parallel_efficiency"], (12.0 / 5.0) / 2.0)
    assert second["median_queue_wait_s"] == 2.0
    assert second["median_turnaround_s"] == 8.0
    assert math.isclose(second["median_core_hours_per_case"], 2.0 * 5.0 / 3600.0)
    assert math.isclose(second["cases_per_100_core_hours"], 36000.0)
    assert math.isclose(
        second["estimated_production_campaign_core_hours"],
        660.0 * 2.0 * 5.0 / 3600.0,
    )
    assert summary["fastest_single_case"]["variant_id"] == last["variant_id"]
    assert summary["fastest_single_case_cores_per_case"] == last["cores_per_case"]
    assert summary["best_parallel_efficiency"]["variant_id"] == second["variant_id"]
    assert summary["best_parallel_efficiency_cores_per_case"] == second["cores_per_case"]
    assert summary["recommended_production_cores_per_case"] == second["cores_per_case"]
    assert summary["recommendation_basis"].startswith("lowest median COMSOL core-hours")
    assert summary["recommended_production"] == {
        "variant_id": second["variant_id"],
        "cores_per_case": second["cores_per_case"],
        "median_comsol_wall_s": second["median_comsol_wall_s"],
        "median_total_case_wall_s": second["median_total_case_wall_s"],
        "median_core_hours_per_case": second["median_core_hours_per_case"],
        "speedup": second["speedup"],
        "parallel_efficiency": second["parallel_efficiency"],
        "current_production_cores_per_case": 16,
        "difference_from_current_cores_per_case": second["cores_per_case"] - 16,
        "fastest_single_case_cores_per_case": last["cores_per_case"],
        "differs_from_fastest_single_case": True,
    }
    assert summary["production_interpretation"]["campaign_total_cases"] == 660
    assert summary["production_interpretation"]["cores_key"] == "cluster.cores_per_case"
    assert summary["production_configuration_modified"] is False
    assert summary["dataset_membership"] == "none"

    benchmark_directory = generation.benchmark.core_benchmark_directory(
        "core_scaling_transient__0123456789abcdef",
        storage_root=tmp_path,
    )
    relative = benchmark_directory.relative_to(tmp_path)
    assert relative.parts[:4] == (
        "01_generation",
        "meta",
        "performance_benchmarks",
        "core_scaling",
    )
    discovered = generation.cases.config.discover_campaign_configs(_repository())
    discovered_paths = {campaign.source_path for campaign in discovered}
    assert maintained.source_path not in discovered_paths
    assert all("benchmarks" not in path.parts for path in discovered_paths)


def test_resume_does_not_duplicate_active_scheduler_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a persisted active Slurm allocation authoritative during resume."""
    suite = _suite()
    directory = generation.benchmark.core_benchmark_directory(
        _BENCHMARK_RUN_ID,
        storage_root=tmp_path,
    )
    directory.mkdir(parents=True)
    manifest = {
        "benchmark_run_id": _BENCHMARK_RUN_ID,
        "git_commit": _COMMIT,
        "state": "submitting",
        "preparation_job_ids": ["1001"],
        "measured_job_ids": ["1002"],
        "submission_history": [
            {
                "submitted_at": "2026-01-01T00:00:00+00:00",
                "role": "measure",
                "variant_id": suite.variants[0].variant_id,
                "repetitions": [1],
                "command": ["sbatch"],
                "job_id": "1002",
            }
        ],
    }
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "1002_1|RUNNING|node-a", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
        },
    )

    duplicate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_duplicate(*args: Any, **kwargs: Any) -> str:
        duplicate_calls.append((args, kwargs))
        return "9999"

    monkeypatch.setattr(generation.benchmark, "_submit", record_duplicate)
    returned = generation.benchmark._submit_pending(
        manifest,
        suite,
        storage=tmp_path,
        variant_id=None,
    )
    assert returned["state"] == "submitted"
    assert duplicate_calls == []
    persisted = generation.benchmark._load_json(
        directory / "benchmark_manifest.json",
        label="persisted benchmark manifest",
    )
    assert persisted["measured_job_ids"] == ["1002"]

    (directory / "canonical_case.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
        },
    )
    submitted: list[list[str]] = []

    def submit_one(command: list[str], **_kwargs: Any) -> str:
        submitted.append(command)
        return "1003"

    monkeypatch.setattr(generation.benchmark, "_submit", submit_one)
    accounting_unknown = generation.benchmark._submit_pending(
        returned,
        suite,
        storage=tmp_path,
        variant_id=None,
    )
    assert accounting_unknown["state"] == "scheduler_unknown"
    assert accounting_unknown["measured_job_ids"] == ["1002"]
    assert submitted == []

    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "1002|FAILED|1:0", "error": None},
        },
    )
    advanced = generation.benchmark._submit_pending(
        accounting_unknown,
        suite,
        storage=tmp_path,
        variant_id=None,
    )
    assert advanced["measured_job_ids"] == ["1002", "1003"]
    assert advanced["submission_history"][-1]["variant_id"] == suite.variants[0].variant_id
    assert advanced["submission_history"][-1]["repetitions"] == [1]
    assert len(submitted) == 1

    all_success = [_success_record(suite, variant, repetition) for variant in suite.variants for repetition in range(1, suite.repetitions + 1)]
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_manifest",
        lambda *_args, **_kwargs: (manifest, suite),
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args, **_kwargs: all_success,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "1003|RUNNING|node-a", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
        },
    )
    status = generation.benchmark.core_benchmark_status(
        _BENCHMARK_RUN_ID,
        storage_root=tmp_path,
    )
    assert status["state"] == "running"
    assert status["retry_repetitions"] == []

    failed_records = [*all_success]
    failed_records[0] = {
        "status": "failed",
        "variant_id": suite.variants[0].variant_id,
        "repetition": 1,
        "repetition_id": suite.repetition_id(suite.variants[0], 1),
    }
    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args, **_kwargs: failed_records,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "1002|FAILED|1:0", "error": None},
        },
    )
    retry = generation.benchmark.core_benchmark_status(
        _BENCHMARK_RUN_ID,
        storage_root=tmp_path,
    )
    assert retry["state"] == "retry_required"
    assert retry["retry_repetitions"] == [
        {
            "variant_id": suite.variants[0].variant_id,
            "repetition": 1,
            "repetition_id": suite.repetition_id(suite.variants[0], 1),
            "evidence_status": "failed",
        }
    ]

    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args, **_kwargs: all_success,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "1003|RUNNING|node-a", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
        },
    )
    with pytest.raises(RuntimeError, match="Slurm jobs remain active"):
        generation.benchmark.finalize_core_benchmark(
            _BENCHMARK_RUN_ID,
            storage_root=tmp_path,
        )


def test_transfer_publication_is_bound_to_pretransfer_inventory() -> None:
    """Reject staged evidence whose digest, file count, or bytes changed in transit."""
    inventory = {
        "inventory_sha256": _HDF5_SHA256,
        "file_count": 17,
        "size_bytes": 4096,
    }
    generation.benchmark._validate_expected_transfer_inventory(
        inventory,
        expected_sha256=_HDF5_SHA256,
        expected_file_count=17,
        expected_size_bytes=4096,
    )
    with pytest.raises(RuntimeError, match="differs from the pre-transfer"):
        generation.benchmark._validate_expected_transfer_inventory(
            {**inventory, "size_bytes": 4097},
            expected_sha256=_HDF5_SHA256,
            expected_file_count=17,
            expected_size_bytes=4096,
        )


def test_summary_view_retry_repairs_only_missing_deterministic_files(
    tmp_path: Path,
) -> None:
    """Repair an interrupted summary write while rejecting conflicting output."""
    suite = _suite()
    records = [_success_record(suite, variant, repetition) for variant in suite.variants for repetition in range(1, suite.repetitions + 1)]
    summary = generation.benchmark.summarize_core_benchmark_results(suite, records)
    assert generation.benchmark._summary_metrics_match(summary, suite, records)
    stale_interpretation = {
        **summary,
        "production_interpretation": {
            **summary["production_interpretation"],
            "campaign_total_cases": 1,
        },
    }
    assert not generation.benchmark._summary_metrics_match(stale_interpretation, suite, records)
    generation.benchmark._validate_or_repair_summary_outputs(
        tmp_path,
        summary,
        records,
        repair_missing=True,
    )
    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "summary.md").is_file()

    (tmp_path / "runs.csv").write_text("conflicting evidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or unsafe"):
        generation.benchmark._validate_or_repair_summary_outputs(
            tmp_path,
            summary,
            records,
            repair_missing=True,
        )
