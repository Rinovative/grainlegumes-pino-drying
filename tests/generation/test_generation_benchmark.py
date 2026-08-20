# ruff: noqa: PLR2004, S101, SLF001
"""Fast core-selection benchmark contracts and compact evidence."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src import common, generation

if TYPE_CHECKING:
    from collections.abc import Mapping

_COMMIT = "a" * 40
_RUN_ID = "core_scaling_transient__0123456789abcdef"


def _synthetic_suite(
    generation_config_factory: Any,
) -> generation.benchmark.CoreBenchmarkSuite:
    """Return two representative cases and four resource-only variants."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        scheduler_kind="slurm",
        natural_count=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    case_config = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    variants = tuple(
        generation.benchmark.CoreBenchmarkVariant(
            source_path=config_path.parent / f"cores_{cores:02d}.yaml",
            variant_id=f"cores_{cores:02d}",
            cores_per_case=cores,
        )
        for cores in (4, 8, 16, 32)
    )
    return generation.benchmark.CoreBenchmarkSuite(
        source_path=config_path.parent / "suite.yaml",
        suite_name="synthetic_core_selection",
        suite_digest="c" * 64,
        case_campaign_path=config_path,
        case_campaign=campaign,
        case_config=case_config,
        representative_cases=(
            generation.benchmark.CoreBenchmarkRepresentativeCase(
                case_role="nominal",
                case_index=1,
            ),
            generation.benchmark.CoreBenchmarkRepresentativeCase(
                case_role="natural",
                case_index=2,
            ),
        ),
        maximum_work_unit_attempts=2,
        variants=variants,
        cores_per_node=32,
        partition="test",
        wall_time="00:30:00",
        scheduler_options=(),
        production_campaign_path=config_path,
        production_cores_config_path=config_path.parent / "execution.yaml",
        production_cores_key="cluster.cores_per_case",
        production_cores_per_case=16,
    )


def _test_owned_pilot_campaign(
    generation_config_factory: Any,
    *,
    cases_per_material: int,
) -> Path:
    """Return a valid mutable pilot configuration owned entirely by one test."""
    path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=cases_per_material,
        campaign_purpose="technical_runtime_smoke",
    )
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["campaign_purpose"] = "pilot_check"
    campaign.pop("paired_equivalence_seed")
    campaign["sampling"] = {
        "method": "lhs",
        "seed_base": 123456,
        "cases_per_material": cases_per_material,
        "case_semantics": {
            "first": "nominal_reference",
            "remaining": "natural_pilot",
        },
    }
    campaign["dataset_packages"] = []
    path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

    execution_path = path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["retention"]["pilot_check"] = "full"
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    return path


def _production() -> dict[str, Any]:
    """Return compact production interpretation for summary tests."""
    return {
        "campaign_config": "configs/generation/campaigns/transient.yaml",
        "campaign_total_cases": 600,
        "current_production_cores_per_case": 16,
        "current_estimated_cases_per_node": 2,
        "current_max_running_cases": None,
        "cores_config": "configs/generation/execution/cluster_cpu.yaml",
        "cores_key": "cluster.cores_per_case",
    }


def _success_records(
    suite: generation.benchmark.CoreBenchmarkSuite,
    runtimes: Mapping[int, tuple[float, float]],
    *,
    queue_seconds: float = 0.0,
    license_wait_seconds: float = 0.0,
    license_probe_seconds: float = 0.0,
    license_blocks: int = 0,
    overlap: bool = True,
    peak_memory_bytes: int = 20,
    peak_scratch_bytes: int = 10,
) -> list[dict[str, Any]]:
    """Return eight small successful measurement records."""
    records: list[dict[str, Any]] = []
    origin = datetime(2026, 8, 19, tzinfo=timezone.utc)
    for wave, variant in enumerate(suite.variants):
        values = runtimes[variant.cores_per_case]
        first_start = origin + timedelta(hours=wave)
        first_end = first_start + timedelta(seconds=values[0])
        second_start = first_start + timedelta(seconds=1) if overlap else first_end + timedelta(seconds=1)
        starts = (first_start, second_start)
        for case_position, runtime in enumerate(values, start=1):
            representative = suite.representative_case(case_position)
            start = starts[case_position - 1]
            end = start + timedelta(seconds=runtime)
            records.append(
                {
                    "status": "success",
                    "variant_id": variant.variant_id,
                    "cores_per_case": variant.cores_per_case,
                    "case_position": case_position,
                    "case_role": representative.case_role,
                    "work_unit_id": suite.work_unit_id(variant, case_position),
                    "timings_seconds": {
                        "scheduler_queue_seconds": queue_seconds,
                        "license_wait_seconds": license_wait_seconds,
                        "license_probe_seconds": license_probe_seconds,
                        "canonical_input_preparation_seconds": 0.25,
                        "comsol_process_seconds": runtime,
                        "export_conversion_seconds": 1.0,
                        "publication_seconds": 0.5,
                        "total_controller_elapsed_seconds": (runtime + queue_seconds + license_wait_seconds + license_probe_seconds + 1.75),
                    },
                    "solver_interval": {
                        "started_at": start.isoformat(),
                        "ended_at": end.isoformat(),
                    },
                    "resource": {
                        "peak_memory_bytes": peak_memory_bytes,
                        "peak_scratch_bytes": peak_scratch_bytes,
                        "slurm_job_id": str(1000 + len(records)),
                        "node": "node-a",
                        "requested_cpus": variant.cores_per_case,
                    },
                    "license": {
                        "license_blocked_submission_count": license_blocks,
                        "license_wait_seconds": license_wait_seconds,
                        "license_probe_seconds": license_probe_seconds,
                        "scheduler_queue_seconds_before_success": 0.0,
                        "detected_feature": "COMSOL" if license_blocks else None,
                        "detected_error_code": "-4" if license_blocks else None,
                        "matched_signatures": (["licensed number of users already reached"] if license_blocks else []),
                        "raw_excerpt": "temporary capacity" if license_blocks else None,
                        "successful_artifacts_override_prior_warning": True,
                    },
                }
            )
    return records


def _pending_records(
    suite: generation.benchmark.CoreBenchmarkSuite,
    *,
    successful: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    """Return one compact result-state record for each benchmark work unit."""
    return [
        {
            "status": ("success" if (variant.variant_id, suite.representative_case(case_position).case_role) in successful else "pending"),
            "variant_id": variant.variant_id,
            "case_position": case_position,
            "case_role": suite.representative_case(case_position).case_role,
            "work_unit_id": suite.work_unit_id(variant, case_position),
            "cores_per_case": variant.cores_per_case,
        }
        for variant in suite.variants
        for case_position in range(1, suite.representative_case_count + 1)
    ]


def _minimal_manifest(run_id: str = _RUN_ID) -> dict[str, Any]:
    """Return mutable orchestration state needed by synthetic submit steps."""
    return {
        "benchmark_run_id": run_id,
        "git_commit": _COMMIT,
        "measured_job_ids": [],
        "submission_history": [],
        "state": "inputs_ready",
    }


def _prepare_submission_test(
    suite: generation.benchmark.CoreBenchmarkSuite,
    storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Create proof placeholders and isolate submission from external Slurm."""
    directory = generation.benchmark.core_benchmark_directory(
        _RUN_ID,
        storage_root=storage,
    )
    for representative in suite.representative_cases:
        proof = directory / "canonical_cases" / f"{representative.case_role}.json"
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        generation.benchmark,
        "_load_case_proofs",
        lambda *_args, **_kwargs: {"nominal": {}, "natural": {}},
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_validate_records_against_proof",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "", "error": None},
        },
    )
    monkeypatch.setattr(
        generation.benchmark,
        "build_core_benchmark_slurm_command",
        lambda *_args, variant, case_position, **_kwargs: [
            "sbatch",
            variant.variant_id,
            str(case_position),
        ],
    )
    return directory


@pytest.mark.parametrize(
    ("value", "status", "started_at"),
    [
        (None, "unavailable", None),
        ("1787097600", "available", "2026-08-19T00:00:00+00:00"),
        ("not-a-timestamp", "invalid", None),
        ("0", "invalid", None),
    ],
)
def test_slurm_start_time_environment_is_optional_bounded_evidence(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    status: str,
    started_at: str | None,
) -> None:
    """Keep worker execution independent of optional Slurm start-time environment."""
    if value is None:
        monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_START_TIME", value)
    evidence = generation.benchmark._slurm_scheduler_start_time_evidence()
    assert evidence["status"] == status
    assert evidence["started_at"] == started_at
    assert evidence["raw_value"] == value


def test_oversized_numeric_slurm_start_time_is_bounded_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject arbitrarily long numeric environment text without aborting the worker."""
    monkeypatch.setenv("SLURM_JOB_START_TIME", "9" * 10000)
    evidence = generation.benchmark._slurm_scheduler_start_time_evidence()
    assert evidence["status"] == "invalid"
    assert evidence["started_at"] is None
    assert evidence["raw_value"] == "9" * generation.benchmark._MAX_SLURM_ENV_VALUE_CHARACTERS


def test_summary_successful_measurements_reject_tampering_and_truncation(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind optional controller evidence exactly to terminal success records."""
    suite = _synthetic_suite(generation_config_factory)
    records = [
        {
            "status": "success",
            "work_unit_id": "unit-1",
            "variant_id": "cores_04",
            "case_role": "nominal",
            "resource": {
                "slurm_job_id": "101",
                "node": "node01",
                "requested_cpus": 4,
            },
            "timings_seconds": {
                "comsol_process_seconds": 12.5,
                "license_wait_seconds": 0.0,
                "export_conversion_seconds": 1.0,
                "publication_seconds": 0.5,
            },
        }
    ]
    manifest = {
        "git_commit": _COMMIT,
        "preflight": {"receipt_sha256": "b" * 64},
        "submission_history": [
            {"role": "measure", "variant_id": "cores_04", "case_role": "nominal", "job_id": "100"},
            {"role": "measure", "variant_id": "cores_04", "case_role": "nominal", "job_id": "101"},
        ],
    }
    scheduler_accounting = {
        "output": ("100|COMPLETED|0:0|2026-08-19T00:00:00|2026-08-19T00:00:03\n101|COMPLETED|0:0|2026-08-19T00:01:00|2026-08-19T00:01:05"),
        "error": None,
    }
    measurements = generation.benchmark._controller_scheduler_queue_accounting(
        manifest,
        records,
        {"sacct": scheduler_accounting},
    )
    monkeypatch.setattr(generation.benchmark, "_proof_identities", lambda *_args: [])
    summary = {
        "schema_kind": generation.benchmark.BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": 1,
        "benchmark_run_id": _RUN_ID,
        "git_commit": _COMMIT,
        "template_sha256": suite.case_config.template_sha256,
        "representative_case_identities": [],
        "preflight": manifest["preflight"],
        "result_set_digest": common.serialization.canonical_json_sha256(records),
        "scheduler_accounting": scheduler_accounting,
        "successful_measurements": measurements,
        "generated_at": "2026-08-19T00:00:00+00:00",
    }
    generation.benchmark._validate_summary_identity(
        summary,
        run_id=_RUN_ID,
        suite=suite,
        manifest=manifest,
        proofs={},
        records=records,
    )
    tampered = {**summary, "successful_measurements": [dict(measurements[0])]}
    tampered["successful_measurements"][0]["comsol_process_seconds"] = 99.0
    with pytest.raises(ValueError, match="stale or incomplete"):
        generation.benchmark._validate_summary_identity(
            tampered,
            run_id=_RUN_ID,
            suite=suite,
            manifest=manifest,
            proofs={},
            records=records,
        )
    truncated = {**summary, "successful_measurements": []}
    with pytest.raises(ValueError, match="stale or incomplete"):
        generation.benchmark._validate_summary_identity(
            truncated,
            run_id=_RUN_ID,
            suite=suite,
            manifest=manifest,
            proofs={},
            records=records,
        )


def test_summary_accounting_reconciliation_only_upgrades_availability() -> None:
    """Upgrade derived queue evidence without regressing or inventing history."""
    unavailable = [{"work_unit_id": "unit-1", "scheduler_queue_availability": "unavailable"}]
    available = [{"work_unit_id": "unit-1", "scheduler_queue_availability": "available"}]
    assert generation.benchmark._has_strictly_more_available_queue_measurements(
        unavailable,
        available,
    )
    assert not generation.benchmark._has_strictly_more_available_queue_measurements(
        available,
        unavailable,
    )
    assert not generation.benchmark._has_strictly_more_available_queue_measurements(
        None,
        available,
    )


def test_finalize_same_result_refreshes_only_improved_queue_accounting(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refresh derived evidence when terminal accounting becomes more complete."""
    suite = _synthetic_suite(generation_config_factory)
    records = _success_records(
        suite,
        {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)},
    )
    submissions: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        job_id = str(100 + index)
        record["resource"].update(
            {
                "slurm_job_id": job_id,
                "node": "node01",
                "requested_cpus": record["cores_per_case"],
            }
        )
        submissions.append(
            {
                "role": "measure",
                "variant_id": record["variant_id"],
                "case_role": record["case_role"],
                "job_id": job_id,
            }
        )
    manifest = {
        "benchmark_run_id": _RUN_ID,
        "git_commit": _COMMIT,
        "preflight": {"receipt_sha256": "a" * 64},
        "measured_job_ids": [item["job_id"] for item in submissions],
        "submission_history": submissions,
        "state": "running",
    }
    directory = generation.benchmark.core_benchmark_directory(_RUN_ID, storage_root=tmp_path)
    directory.mkdir(parents=True)
    immutable_success = directory / "runs" / "immutable-success.json"
    immutable_success.parent.mkdir()
    immutable_success.write_text('{"immutable": true}\n', encoding="utf-8")
    unavailable_scheduler = {"sacct": {"output": "", "error": "not yet available"}}
    unavailable_measurements = generation.benchmark._controller_scheduler_queue_accounting(
        manifest,
        records,
        unavailable_scheduler,
    )
    initial = generation.benchmark.summarize_core_benchmark_results(suite, records)
    generation.benchmark._apply_controller_queue_to_summary(
        initial,
        unavailable_measurements,
    )
    initial.update(
        {
            "schema_kind": generation.benchmark.BENCHMARK_SUMMARY_SCHEMA_KIND,
            "schema_version": 1,
            "benchmark_run_id": _RUN_ID,
            "git_commit": _COMMIT,
            "template_sha256": suite.case_config.template_sha256,
            "representative_case_identities": [],
            "preflight": manifest["preflight"],
            "scheduler_accounting": unavailable_scheduler["sacct"],
            "successful_measurements": unavailable_measurements,
            "result_set_digest": common.serialization.canonical_json_sha256(records),
            "generated_at": "2026-08-19T00:00:00+00:00",
        }
    )
    common.serialization.atomic_write_json(directory / "summary.json", initial)
    common.serialization.atomic_write_text(
        directory / "runs.csv",
        generation.benchmark._results_csv(
            records,
            queue_by_work_unit=generation.benchmark._controller_queue_by_work_unit(
                unavailable_measurements,
            ),
        ),
    )
    common.serialization.atomic_write_text(
        directory / "summary.md",
        generation.benchmark.core_benchmark_markdown(initial),
    )
    scheduler_rows = "\n".join(f"{item['job_id']}|COMPLETED|0:0|2026-08-19T00:00:00|2026-08-19T00:00:05" for item in submissions)
    scheduler = {
        "squeue": {"output": "", "error": None},
        "sacct": {"output": scheduler_rows, "error": None},
    }
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_manifest",
        lambda *_args, **_kwargs: (manifest, suite),
    )
    monkeypatch.setattr(generation.benchmark, "_scheduler_evidence", lambda *_args: scheduler)
    monkeypatch.setattr(generation.benchmark, "_load_case_proofs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generation.benchmark, "_result_records", lambda *_args: records)
    monkeypatch.setattr(generation.benchmark, "_validate_records_against_proof", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark, "_validate_summary_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark, "_persist_manifest", lambda *_args, **_kwargs: None)
    refreshed = generation.benchmark.finalize_core_benchmark(_RUN_ID, storage_root=tmp_path)
    persisted = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert refreshed == persisted
    assert all(item["scheduler_queue_availability"] == "available" for item in persisted["successful_measurements"])
    assert persisted["result_set_digest"] == initial["result_set_digest"]
    assert persisted["schema_version"] == 1
    assert immutable_success.read_bytes() == b'{"immutable": true}\n'
    for before, after in zip(initial["variants"], persisted["variants"], strict=True):
        assert {key: value for key, value in before.items() if key != "scheduler_queue_seconds"} == {
            key: value for key, value in after.items() if key != "scheduler_queue_seconds"
        }
    history = directory / "summary_history" / "revision-0001"
    assert (history / "summary.json").is_file()
    assert (history / "runs.csv").is_file()
    assert (history / "summary.md").is_file()
    stable_bytes = (directory / "summary.json").read_bytes()
    scheduler["sacct"] = {"output": "", "error": "temporarily unavailable"}
    assert generation.benchmark.finalize_core_benchmark(_RUN_ID, storage_root=tmp_path) == persisted
    assert (directory / "summary.json").read_bytes() == stable_bytes
    assert len(list((directory / "summary_history").iterdir())) == 1


def test_controller_accounting_includes_prior_license_submission_queues() -> None:
    """Derive queue duration only from exact root accounting rows."""
    scheduler = {
        "sacct": {
            "error": None,
            "output": (
                "101|COMPLETED|0:0|2026-08-19T00:00:00|2026-08-19T00:00:05|2026-08-19T00:01:00\n"
                "102|COMPLETED|0:0|2026-08-19T00:02:00|2026-08-19T00:02:07|2026-08-19T00:03:00\n"
                "102.batch|COMPLETED|0:0|2026-08-19T00:02:00|2026-08-19T00:02:07|2026-08-19T00:03:00\n"
                "103|RUNNING|0:0|2026-08-19T00:04:00|2026-08-19T00:04:02|Unknown"
            ),
        }
    }
    assert generation.benchmark._accounted_queue_seconds(scheduler, ["101", "102"]) == 12.0
    assert generation.benchmark._accounted_queue_seconds(scheduler, ["101", "999"]) is None
    assert generation.benchmark._accounted_queue_seconds(scheduler, ["103"]) is None


def test_historical_success_attempt_without_success_record_remains_pending(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require published success evidence before counting a terminal attempt as success."""
    suite = _synthetic_suite(generation_config_factory)
    variant = suite.canary_variant()
    directory = generation.benchmark.core_benchmark_directory(_RUN_ID, storage_root=tmp_path)
    attempt = directory / "runs" / suite.execution_id(variant) / suite.work_unit_id(variant, 1) / "attempt-0001.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text('{"status": "success"}\n', encoding="utf-8")
    monkeypatch.setattr(generation.benchmark, "_validate_success_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark, "_load_benchmark_license_wait", lambda *_args, **_kwargs: None)
    records = generation.benchmark._result_records(directory, suite)
    record = next(item for item in records if item["work_unit_id"] == suite.work_unit_id(variant, 1))
    assert record["status"] == "pending"


def test_benchmark_worker_calls_shared_executor_and_keeps_its_duration(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Admit a successful measurement without a Slurm start-time environment."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    variant = suite.canary_variant()
    representative = suite.representative_case(1)
    proof = {
        "case_input_id": "d" * 64,
        "simulation_case_id": "e" * 64,
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "canonical_input_preparation_seconds": 0.25,
    }
    manifest = {
        "benchmark_run_id": _RUN_ID,
        "git_commit": _COMMIT,
        "preflight": {"receipt_sha256": "f" * 64},
        "submission_history": [
            {
                "role": "measure",
                "job_id": "123",
                "variant_id": variant.variant_id,
                "case_role": representative.case_role,
                "work_unit_id": suite.work_unit_id(variant, 1),
                "submitted_at": "2026-08-19T00:00:00+00:00",
            }
        ],
    }
    output = tmp_path / "case.h5"
    output.write_bytes(b"synthetic")
    log = tmp_path / "solver.log"
    log.write_text("solver completed\n", encoding="utf-8")
    prepared = SimpleNamespace(work_directory=tmp_path / "work")
    result = SimpleNamespace(
        canonical_case=SimpleNamespace(path=output),
        command=("comsol", "-np", str(variant.cores_per_case)),
        timing={
            "runtime_s": 99.0,
            "comsol_process_seconds": 12.5,
            "export_conversion_s": 7.0,
            "export_conversion_seconds": 1.0,
            "scratch_peak_bytes": 2,
            "started_at": "2026-08-19T00:00:10+00:00",
            "ended_at": "2026-08-19T00:00:22+00:00",
        },
        solver_log=log,
    )
    calls: list[object] = []
    monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(variant.cores_per_case))
    monkeypatch.setattr(generation.benchmark, "load_core_benchmark_manifest", lambda *_args, **_kwargs: (manifest, suite))
    monkeypatch.setattr(generation.benchmark, "_require_current_checkout", lambda *_args: None)
    monkeypatch.setattr(generation.benchmark.source_service, "required_git_commit", lambda: _COMMIT)
    monkeypatch.setattr(generation.benchmark, "_load_case_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(generation.benchmark.preparation_service, "prepare_case_work_directory", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(generation.benchmark, "_proof_payload", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(generation.benchmark, "_validate_materialized_case_proof", lambda *_args: None)
    monkeypatch.setattr(generation.benchmark.runtime_service, "execute_prepared_case", lambda *_args, **_kwargs: calls.append(1) or result)
    monkeypatch.setattr(generation.benchmark.storage_service, "validate_case_hdf5", lambda *_args, **_kwargs: dict(proof))
    monkeypatch.setattr(generation.benchmark, "_validate_hdf5_scientific_identity", lambda *_args: None)
    monkeypatch.setattr(generation.benchmark, "_benchmark_peak_child_memory_bytes", lambda: 0)
    monkeypatch.setattr(generation.benchmark, "_cleanup_prepared", lambda *_args, **_kwargs: 0)
    record = generation.benchmark.run_core_benchmark_case(
        _RUN_ID,
        variant.variant_id,
        representative.case_role,
        storage_root=storage,
        work_root=tmp_path / "work-root",
    )
    assert calls == [1]
    assert record["timings_seconds"]["comsol_process_seconds"] == 12.5
    assert record["timings_seconds"]["export_conversion_seconds"] == 1.0
    assert record["timings_seconds"]["scheduler_queue_seconds"] is None
    accounting = generation.benchmark._controller_scheduler_queue_accounting(
        manifest,
        [record],
        {"sacct": {"error": None, "output": "123|COMPLETED|0:0|2026-08-19T00:00:00|2026-08-19T00:00:05"}},
    )
    assert accounting[0]["slurm_job_id"] == "123"
    assert record["worker_interval"]["slurm_job_start_time"]["status"] == "unavailable"


def test_resource_change_preserves_both_case_identities(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Keep resource variants outside both canonical scientific identities."""
    suite = _synthetic_suite(generation_config_factory)
    baseline_variant = suite.variant("cores_04")
    changed_variant = replace(baseline_variant, cores_per_case=8)
    for position, representative in enumerate(suite.representative_cases, start=1):
        baseline = generation.cases.case.generate_case_input_bundle(
            suite.case_config,
            representative.case_index,
            tmp_path / f"baseline-{position}",
        )
        changed = generation.cases.case.generate_case_input_bundle(
            suite.case_config,
            representative.case_index,
            tmp_path / f"changed-{position}",
        )
        assert changed.case_input_id == baseline.case_input_id
        assert changed.simulation_case_id == baseline.simulation_case_id
        assert changed.case_payload == baseline.case_payload
    assert suite.execution_id(changed_variant) != suite.execution_id(baseline_variant)


@pytest.mark.parametrize("cases_per_material", [1, 3, 5])
def test_maintained_benchmark_is_independent_of_mutable_pilot_count(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    cases_per_material: int,
) -> None:
    """Resolve the same benchmark owner around valid test-owned pilot counts."""
    test_project_root = common.paths.get_project_root()
    repository = Path(__file__).resolve().parents[2]
    suite_path = repository / "configs/generation/benchmarks/transient_core_scaling/suite.yaml"
    monkeypatch.setenv("PROJECT_ROOT", str(repository))
    before = generation.benchmark.load_core_benchmark_suite(
        suite_path,
        require_executable=False,
    )
    pilot_path = _test_owned_pilot_campaign(
        generation_config_factory,
        cases_per_material=cases_per_material,
    )
    monkeypatch.setenv("PROJECT_ROOT", str(test_project_root))
    pilot = generation.cases.config.load_campaign_config(
        pilot_path,
        require_executable=False,
    )
    monkeypatch.setenv("PROJECT_ROOT", str(repository))
    after = generation.benchmark.load_core_benchmark_suite(
        suite_path,
        require_executable=False,
    )

    assert pilot.total_case_count == cases_per_material
    assert before.suite_digest == after.suite_digest
    assert before.case_campaign_path == after.case_campaign_path
    assert before.case_campaign_path != pilot.source_path
    assert [case.case_index for case in after.representative_cases] == [1, 2]


def test_sequence_and_slurm_jobs_use_same_cases_in_each_wave(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Order both production cases first and encode readable case-role jobs."""
    suite = _synthetic_suite(generation_config_factory)
    sequence = generation.benchmark._measured_sequence(suite)
    assert [variant.cores_per_case for variant, _position in sequence] == [
        16,
        16,
        4,
        4,
        8,
        8,
        32,
        32,
    ]
    assert [position for _variant, position in sequence] == [1, 2] * 4
    project_root = common.paths.get_project_root()
    launcher = project_root / "scripts/generation_benchmark_node.sh"
    source_launcher = Path(__file__).resolve().parents[2] / "scripts/generation_benchmark_node.sh"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_launcher, launcher)
    command = generation.benchmark.build_core_benchmark_slurm_command(
        suite,
        run_id=_RUN_ID,
        storage_root=tmp_path.resolve(),
        log_directory=(tmp_path / "logs").resolve(),
        variant=suite.canary_variant(),
        case_position=2,
    )
    assert any(value.startswith("--job-name=td-bench-c16-nat-") for value in command)
    assert not any(value.startswith("--licenses") for value in command)
    wrapped = shlex.split(command[-1].removeprefix("--wrap="))
    assert wrapped[-5:] == [
        str(launcher),
        str(project_root),
        _RUN_ID,
        "cores_16",
        "natural",
    ]


def test_benchmark_attempt_history_is_append_only_and_hash_chained(
    tmp_path: Path,
) -> None:
    """Bind scientific retries to the immutable immediately preceding receipt."""
    stable = {
        "schema_version": 1,
        "benchmark_run_id": _RUN_ID,
        "suite_digest": "1" * 64,
        "variant_id": "cores_16",
        "execution_id": "execution-16",
        "case_position": 1,
        "case_role": "nominal",
        "work_unit_id": "work-unit",
        "git_commit": _COMMIT,
        "case_input_id": "2" * 64,
        "simulation_case_id": "3" * 64,
        "scientific_config_digest": "4" * 64,
        "template_sha256": "5" * 64,
        "benchmark_preflight_sha256": "6" * 64,
        "cores_per_case": 16,
    }
    first_path = common.serialization.atomic_write_json(
        tmp_path / "attempt-0001.json",
        {**stable, "attempt": 1, "previous_attempt": None},
    )
    second_path = common.serialization.atomic_write_json(
        tmp_path / "attempt-0002.json",
        {
            **stable,
            "attempt": 2,
            "previous_attempt": {
                "attempt": 1,
                "receipt_sha256": common.serialization.file_sha256(first_path),
            },
        },
    )
    generation.benchmark._validate_benchmark_attempt_chain((first_path, second_path))
    changed = json.loads(second_path.read_text(encoding="utf-8"))
    changed["case_input_id"] = "9" * 64
    common.serialization.atomic_write_json(second_path, changed)
    with pytest.raises(ValueError, match="identity changed"):
        generation.benchmark._validate_benchmark_attempt_chain((first_path, second_path))


def test_preflight_is_standalone_and_plans_eight_commands(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run benchmark-owned preflight without Smoke or a preparation job."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    scratch = tmp_path / "scratch"
    storage.mkdir()
    scratch.mkdir()
    executable = tmp_path / "comsol"
    executable.write_text("synthetic", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(generation.benchmark, "_repository_commit", lambda: _COMMIT)
    monkeypatch.setattr(generation.benchmark, "_require_clean_repository", lambda: None)
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_suite",
        lambda *_args, **_kwargs: suite,
    )
    planned_commands: list[tuple[str, int]] = []

    def command(
        *_args: Any,
        variant: Any,
        case_position: int,
        **_kwargs: Any,
    ) -> list[str]:
        planned_commands.append((variant.variant_id, case_position))
        return ["sbatch", variant.variant_id, str(case_position)]

    monkeypatch.setattr(
        generation.benchmark,
        "build_core_benchmark_slurm_command",
        command,
    )
    receipt = generation.benchmark.preflight_core_benchmark(
        suite.source_path,
        git_commit=_COMMIT,
        storage_root=storage,
        scratch_root=scratch,
        comsol_version_output="COMSOL Multiphysics 6.4",
        comsol_executable_path=executable,
    )
    assert receipt["checks"]["clean_exact_source"] == "pass"
    assert len(planned_commands) == 8
    assert len(set(planned_commands)) == 8
    assert not (storage / "01_generation/meta/real_smoke").exists()


def test_login_preparation_creates_two_proofs_and_submits_nothing(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialize both canonical inputs on the login side before submission."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    wave_order = suite.variant_wave_order()
    plan = {
        "benchmark_run_id": _RUN_ID,
        "git_commit": _COMMIT,
        "preflight": {},
        "variant_wave_order": [variant.variant_id for variant in wave_order],
        "canary_wave": {},
        "measurement_waves": [],
        "required_successful_measurements": 8,
    }
    monkeypatch.setattr(generation.benchmark, "plan_core_benchmark", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_suite",
        lambda *_args, **_kwargs: suite,
    )
    monkeypatch.setattr(generation.benchmark, "_repository_relative", lambda path: path.name)

    def materialize(
        observed_run_id: str,
        *,
        storage_root: Path,
        work_root: Path,
    ) -> tuple[Path, ...]:
        assert observed_run_id == _RUN_ID
        assert Path(work_root) == tmp_path / "scratch"
        directory = generation.benchmark.core_benchmark_directory(
            observed_run_id,
            storage_root=storage_root,
        )
        paths = []
        for representative in suite.representative_cases:
            target = directory / "canonical_cases" / f"{representative.case_role}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
            paths.append(target)
        return tuple(paths)

    monkeypatch.setattr(generation.benchmark, "_materialize_core_benchmark_inputs", materialize)
    monkeypatch.setattr(
        generation.benchmark,
        "_submit",
        lambda *_args, **_kwargs: pytest.fail("input preparation submitted Slurm work"),
    )
    manifest = generation.benchmark.prepare_core_benchmark(
        suite.source_path,
        git_commit=_COMMIT,
        storage_root=storage,
        scratch_root=tmp_path / "scratch",
        comsol_version_output="COMSOL Multiphysics 6.4",
        comsol_executable_path=tmp_path / "comsol",
    )
    assert manifest["state"] == "inputs_ready"
    assert manifest["measured_job_ids"] == []
    assert manifest["submission_history"] == []
    assert len(tuple((storage / "01_generation/meta/performance_benchmarks/core_scaling" / _RUN_ID / "canonical_cases").glob("*.json"))) == 2


def test_submitter_runs_two_cases_concurrently_and_holds_later_waves(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit both canary cases, then only the next wave after both succeed."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    _prepare_submission_test(suite, storage, monkeypatch)
    records = _pending_records(suite)
    monkeypatch.setattr(generation.benchmark, "_result_records", lambda *_args: records)
    job_ids = iter(("501", "502", "503", "504"))
    monkeypatch.setattr(generation.benchmark, "_submit", lambda *_args, **_kwargs: next(job_ids))
    manifest = generation.benchmark._submit_pending(
        _minimal_manifest(),
        suite,
        storage=storage,
    )
    assert [item["case_role"] for item in manifest["submission_history"]] == [
        "nominal",
        "natural",
    ]
    assert {item["variant_id"] for item in manifest["submission_history"]} == {"cores_16"}
    successful = frozenset({("cores_16", "nominal"), ("cores_16", "natural")})
    records[:] = _pending_records(suite, successful=successful)
    manifest = generation.benchmark._submit_pending(manifest, suite, storage=storage)
    assert [item["variant_id"] for item in manifest["submission_history"]] == [
        "cores_16",
        "cores_16",
        "cores_04",
        "cores_04",
    ]


def test_partial_wave_resume_submits_only_missing_case(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse a valid success and submit no duplicate work unit."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    _prepare_submission_test(suite, storage, monkeypatch)
    successful = frozenset({("cores_16", "nominal")})
    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args: _pending_records(suite, successful=successful),
    )
    submitted: list[str] = []

    def submit(command: list[str], **_kwargs: Any) -> str:
        submitted.append(command[-1])
        return "701"

    monkeypatch.setattr(generation.benchmark, "_submit", submit)
    manifest = _minimal_manifest()
    nominal = suite.representative_case(1)
    manifest["measured_job_ids"] = ["700"]
    manifest["submission_history"] = [
        {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "role": "measure",
            "variant_id": "cores_16",
            "case_position": 1,
            "case_role": nominal.case_role,
            "work_unit_id": suite.work_unit_id(suite.canary_variant(), 1),
            "command": ["sbatch"],
            "job_id": "700",
        }
    ]
    result = generation.benchmark._submit_pending(manifest, suite, storage=storage)
    assert submitted == ["2"]
    assert [item["case_role"] for item in result["submission_history"]] == [
        "nominal",
        "natural",
    ]


def test_first_wave_failure_is_canary_failure_without_extra_solve(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block later waves after either final canary measurement exhausts attempts."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    directory = _prepare_submission_test(suite, storage, monkeypatch)
    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args: _pending_records(suite),
    )
    failed = directory / "runs" / suite.execution_id(suite.canary_variant()) / suite.work_unit_id(suite.canary_variant(), 1) / "attempt-0001.json"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        generation.benchmark,
        "_scientific_failure_count",
        lambda attempts: suite.maximum_work_unit_attempts if attempts else 0,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_submit",
        lambda *_args, **_kwargs: pytest.fail("submitted work after canary failure"),
    )
    manifest = generation.benchmark._submit_pending(
        _minimal_manifest(),
        suite,
        storage=storage,
    )
    assert manifest["state"] == "canary_failed"
    assert manifest["submission_history"] == []


def test_license_block_waits_and_retries_without_manual_variant(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep capacity waits operational and retry both eligible work units."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    storage.mkdir()
    _prepare_submission_test(suite, storage, monkeypatch)
    monkeypatch.setattr(
        generation.benchmark,
        "_result_records",
        lambda *_args: _pending_records(suite),
    )
    wait = {"retry_budget_remaining": True}
    monkeypatch.setattr(
        generation.benchmark,
        "_load_benchmark_license_wait",
        lambda *_args, **_kwargs: wait,
    )
    monkeypatch.setattr(
        generation.benchmark.license_service,
        "wait_record_is_eligible",
        lambda _wait: False,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_submit",
        lambda *_args, **_kwargs: pytest.fail("submitted before license wait elapsed"),
    )
    manifest = generation.benchmark._submit_pending(
        _minimal_manifest(),
        suite,
        storage=storage,
    )
    assert manifest["state"] == "license_blocked"
    monkeypatch.setattr(
        generation.benchmark.license_service,
        "wait_record_is_eligible",
        lambda _wait: True,
    )
    job_ids = iter(("801", "802"))
    monkeypatch.setattr(generation.benchmark, "_submit", lambda *_args, **_kwargs: next(job_ids))
    resumed = generation.benchmark._submit_pending(manifest, suite, storage=storage)
    assert [item["case_role"] for item in resumed["submission_history"]] == [
        "nominal",
        "natural",
    ]


def test_license_only_attempt_retains_wait_not_runtime_attempt(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain probe and queue time without creating a successful observation."""
    suite = _synthetic_suite(generation_config_factory)
    directory = generation.benchmark.core_benchmark_directory(
        _RUN_ID,
        storage_root=tmp_path,
    )
    work_unit = directory / "runs" / suite.execution_id(suite.canary_variant()) / suite.work_unit_id(suite.canary_variant(), 1)
    work_unit.mkdir(parents=True)
    monkeypatch.setattr(
        generation.benchmark.license_service,
        "bounded_retry_delay_seconds",
        lambda _policy, **_kwargs: 0.0,
    )
    evidence = generation.runtime.license.TemporaryLicenseCapacityClassification(
        classification=generation.runtime.license.TEMPORARY_LICENSE_CAPACITY,
        feature="COMSOL",
        license_code="-4",
        matched_signatures=("licensed number of users already reached",),
        raw_excerpt="temporary capacity",
    )
    error = generation.runtime.license.TemporaryLicenseCapacityError(
        "capacity unavailable",
        work_directory=tmp_path / "work",
        command=("comsol",),
        evidence=evidence,
        exit_code=1,
        solver_progress_started=False,
        expected_exports_exist=False,
    )
    wait = generation.benchmark._record_benchmark_license_wait(
        directory,
        suite,
        suite.canary_variant(),
        1,
        error,
        run_id=_RUN_ID,
        job_id="1234",
        license_probe_seconds=2.5,
        scheduler_queue_seconds=7.0,
    )
    assert wait["cumulative_probe_seconds"] == 2.5
    assert wait["cumulative_scheduler_queue_seconds"] == 7.0
    assert wait["retry_budget_remaining"] is False
    assert not tuple(work_unit.glob("attempt-*.json"))


def test_summary_uses_only_successful_comsol_runtime_for_ranking(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep queue, license, conversion, and publication out of recommendation."""
    suite = _synthetic_suite(generation_config_factory)
    monkeypatch.setattr(generation.benchmark, "_production_interpretation", lambda _suite: _production())
    runtimes = {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)}
    baseline = generation.benchmark.summarize_core_benchmark_results(
        suite,
        _success_records(suite, runtimes),
    )
    delayed = generation.benchmark.summarize_core_benchmark_results(
        suite,
        _success_records(
            suite,
            runtimes,
            queue_seconds=10000.0,
            license_wait_seconds=5000.0,
            license_probe_seconds=100.0,
            license_blocks=3,
        ),
    )
    assert baseline["fastest_single_case_cores"] == 32
    assert baseline["lowest_core_hours_cores"] == 4
    assert baseline["recommended_cores_per_case"] == 4
    assert baseline["recommended_estimated_cases_per_node"] == 8
    assert delayed["recommended_cores_per_case"] == baseline["recommended_cores_per_case"]
    four = baseline["variants"][0]
    assert four["median_core_hours_per_case"] == pytest.approx(4 * 100 / 3600)
    assert four["estimated_cases_per_node_hour"] == pytest.approx(8 * 3600 / 100)
    assert four["resource_feasibility"] == "operator_review_required"


def test_resource_infeasible_variant_cannot_be_recommended(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude projected node packing that exceeds an authoritative limit."""
    suite = replace(
        _synthetic_suite(generation_config_factory),
        node_memory_limit_bytes=100,
        node_scratch_limit_bytes=1000,
    )
    monkeypatch.setattr(generation.benchmark, "_production_interpretation", lambda _suite: _production())
    runtimes = {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)}
    summary = generation.benchmark.summarize_core_benchmark_results(
        suite,
        _success_records(suite, runtimes, peak_memory_bytes=20),
    )
    assert summary["variants"][0]["memory_feasibility"] == "fail"
    assert summary["recommended_cores_per_case"] == 8


def test_five_percent_tie_is_deterministic(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break exact throughput and core-hour ties toward fewer cores."""
    suite = _synthetic_suite(generation_config_factory)
    monkeypatch.setattr(generation.benchmark, "_production_interpretation", lambda _suite: _production())
    runtimes = {4: (100.0, 100.0), 8: (50.0, 50.0), 16: (40.0, 40.0), 32: (30.0, 30.0)}
    summary = generation.benchmark.summarize_core_benchmark_results(
        suite,
        _success_records(suite, runtimes),
    )
    assert summary["variants"][0]["estimated_cases_per_node_hour"] == summary["variants"][1]["estimated_cases_per_node_hour"]
    assert summary["variants"][0]["median_core_hours_per_case"] == summary["variants"][1]["median_core_hours_per_case"]
    assert summary["recommended_cores_per_case"] == 4


def test_license_overlap_qualification_does_not_change_compute_metrics(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report absent overlap separately from compute-only performance."""
    suite = _synthetic_suite(generation_config_factory)
    monkeypatch.setattr(generation.benchmark, "_production_interpretation", lambda _suite: _production())
    runtimes = {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)}
    summary = generation.benchmark.summarize_core_benchmark_results(
        suite,
        _success_records(
            suite,
            runtimes,
            license_blocks=1,
            license_wait_seconds=30.0,
            overlap=False,
        ),
    )
    assert summary["recommended_cores_per_case"] == 4
    assert summary["license_qualification"] == ("compute recommendation valid; concurrent-license observation incomplete")
    assert all(record["observed_peak_solver_concurrency"] == 1 for record in summary["variants"])


def test_markdown_summary_loader_reads_only_persisted_canonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Print the validated Markdown artifact and fail closed on drift or absence."""
    directory = generation.benchmark.core_benchmark_directory(
        _RUN_ID,
        storage_root=tmp_path,
    )
    directory.mkdir(parents=True)
    summary = {"benchmark_run_id": _RUN_ID}
    expected = "# Persisted benchmark summary\n"
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_summary",
        lambda *_args, **_kwargs: summary,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "core_benchmark_markdown",
        lambda _summary: expected,
    )
    summary_path = directory / "summary.md"
    summary_path.write_text(expected, encoding="utf-8")

    assert (
        generation.benchmark.load_core_benchmark_markdown(
            _RUN_ID,
            storage_root=tmp_path,
        )
        == expected
    )

    summary_path.write_text("# Drifted summary\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        generation.benchmark.load_core_benchmark_markdown(
            _RUN_ID,
            storage_root=tmp_path,
        )
    summary_path.unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        generation.benchmark.load_core_benchmark_markdown(
            _RUN_ID,
            storage_root=tmp_path,
        )


def test_completed_status_exposes_the_persisted_validated_summary(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project final persisted benchmark evidence through read-only status."""
    suite = _synthetic_suite(generation_config_factory)
    storage = tmp_path / "storage"
    directory = generation.benchmark.core_benchmark_directory(
        _RUN_ID,
        storage_root=storage,
    )
    for representative in suite.representative_cases:
        proof = generation.benchmark._canonical_case_proof_path(
            directory,
            representative,
        )
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text("{}\n", encoding="utf-8")
    markdown = "# Validated benchmark\n\nRecommended: 4 cores.\n"
    (directory / "summary.md").write_text(markdown, encoding="utf-8")
    records = _success_records(
        suite,
        {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)},
    )
    manifest = {
        **_minimal_manifest(),
        "state": "complete",
        "suite_config": str(suite.source_path),
    }
    summary = {
        "fastest_single_case_cores": 32,
        "lowest_core_hours_cores": 4,
        "recommended_cores_per_case": 4,
        "recommended_production": {
            "estimated_cases_per_node": 8,
            "estimated_cases_per_node_hour": 288.0,
        },
        "license_qualification": "concurrent-license execution observed",
    }
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_manifest",
        lambda *_args, **_kwargs: (manifest, suite),
    )
    monkeypatch.setattr(generation.benchmark, "_result_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        generation.benchmark,
        "_benchmark_work_unit_views",
        lambda *_args, **_kwargs: [{**record, "state": "successful", "runtime_progress": None} for record in records],
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_controller_scheduler_queue_accounting",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_completed_wave_evaluation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_summary",
        lambda *_args, **_kwargs: summary,
    )
    monkeypatch.setattr(
        generation.benchmark,
        "core_benchmark_markdown",
        lambda _summary: markdown,
    )

    status = generation.benchmark.core_benchmark_status(
        _RUN_ID,
        storage_root=storage,
        query_scheduler=False,
    )

    assert status["state"] == "complete"
    assert status["final_summary"] == {
        "path": str(directory / "summary.md"),
        "markdown": markdown,
        "fastest_single_case_cores": 32,
        "lowest_core_hours_cores": 4,
        "recommended_cores_per_case": 4,
        "estimated_cases_per_node": 8,
        "estimated_compute_only_cases_per_node_hour": 288.0,
        "license_qualification": "concurrent-license execution observed",
    }

    (directory / "summary.md").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        generation.benchmark.core_benchmark_status(
            _RUN_ID,
            storage_root=storage,
            query_scheduler=False,
        )


def test_valid_success_evidence_overrides_prior_license_warning() -> None:
    """Admit bounded warning evidence only when terminal success is authoritative."""
    result = {
        "solver_interval": {
            "started_at": "2026-08-19T00:00:00+00:00",
            "ended_at": "2026-08-19T00:00:01+00:00",
        },
        "license": {
            "license_blocked_submission_count": 1,
            "license_wait_seconds": 60.0,
            "license_probe_seconds": 1.0,
            "scheduler_queue_seconds_before_success": 2.0,
            "detected_feature": "COMSOL",
            "detected_error_code": "-4",
            "matched_signatures": ["licensed number of users already reached"],
            "raw_excerpt": "temporary capacity",
            "successful_artifacts_override_prior_warning": True,
        },
        "solver_log": {
            "sha256": "a" * 64,
            "size_bytes": 10,
            "excerpt": "warning then success",
            "excerpt_truncated": False,
        },
    }
    generation.benchmark._validate_success_support_evidence(
        result,
        work_unit_id="synthetic",
    )
    result["license"]["successful_artifacts_override_prior_warning"] = False
    with pytest.raises(ValueError, match="success precedence"):
        generation.benchmark._validate_success_support_evidence(
            result,
            work_unit_id="synthetic",
        )


def test_benchmark_cancellation_targets_only_owned_active_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist cancellation before targeting exact pending and running jobs."""
    storage = tmp_path / "storage"
    directory = generation.benchmark.core_benchmark_directory(
        _RUN_ID,
        storage_root=storage,
    )
    directory.mkdir(parents=True)
    manifest = _minimal_manifest()
    manifest["measured_job_ids"] = ["501", "502"]
    monkeypatch.setattr(
        generation.benchmark,
        "load_core_benchmark_manifest",
        lambda *_args, **_kwargs: (manifest, object()),
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {
                "output": "501|PENDING|Resources\n502|RUNNING|node01",
                "error": None,
            },
            "sacct": {"output": "", "error": None},
        },
    )
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert (directory / "cancellations.json").is_file()
        assert manifest["state"] == "cancel_requested"
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(generation.benchmark.subprocess, "run", run)
    generation.benchmark.cancel_core_benchmark(_RUN_ID, storage_root=storage)
    assert commands == [
        ["scancel", "501"],
        ["scancel", "--signal=TERM", "--batch", "502"],
    ]


def test_benchmark_work_units_use_disjoint_scheduler_and_submission_states(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Separate running jobs, Slurm PENDING jobs, and unsubmitted future waves."""
    suite = _synthetic_suite(generation_config_factory)
    records = _pending_records(suite)
    current = suite.variant_wave_order()[0]
    submissions = []
    rows = []
    for position, job_id in enumerate(("501", "502"), start=1):
        representative = suite.representative_case(position)
        submissions.append(
            {
                "role": "measure",
                "job_id": job_id,
                "variant_id": current.variant_id,
                "case_role": representative.case_role,
                "work_unit_id": suite.work_unit_id(current, position),
            }
        )
        state = "RUNNING" if position == 1 else "PENDING"
        rows.append(f"{job_id}|{state}|None|node-a|00:01:00")
    manifest = {"benchmark_run_id": _RUN_ID, "submission_history": submissions}
    scheduler = {"squeue": {"output": "\n".join(rows), "error": None}, "sacct": {"output": "", "error": None}}
    directory = tmp_path / "benchmark"
    directory.mkdir()

    views = generation.benchmark._benchmark_work_unit_views(manifest, records, scheduler, directory=directory, suite=suite)
    counts = {
        state: sum(view["state"] == state for view in views)
        for state in ("successful", "running", "scheduler_pending", "license_blocked", "never_started", "failed")
    }
    assert counts == {
        "successful": 0,
        "running": 1,
        "scheduler_pending": 1,
        "license_blocked": 0,
        "never_started": 6,
        "failed": 0,
    }
    assert sum(counts.values()) == len(views) == 8
    running = next(view for view in views if view["state"] == "running")
    assert running["runtime_progress"]["reason"] == "not_reported"


def test_partial_evaluation_uses_only_complete_waves(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep incomplete waves out of provisional and final recommendations."""
    suite = _synthetic_suite(generation_config_factory)
    monkeypatch.setattr(generation.benchmark, "_production_interpretation", lambda _suite: _production())
    runtimes = {4: (100.0, 100.0), 8: (60.0, 60.0), 16: (40.0, 40.0), 32: (30.0, 30.0)}
    successes = _success_records(suite, runtimes, queue_seconds=999.0, license_wait_seconds=888.0)
    successful_by_unit = {record["work_unit_id"]: record for record in successes}
    pending = _pending_records(suite)
    manifest = {"submission_history": []}
    scheduler = {"squeue": {"output": "", "error": None}, "sacct": {"output": "", "error": None}}
    assert generation.benchmark._completed_wave_evaluation(suite, manifest, pending, scheduler) is None

    first_two = {variant.variant_id for variant in suite.variant_wave_order()[:2]}
    one_complete = [
        successful_by_unit[record["work_unit_id"]] if record["variant_id"] == suite.variant_wave_order()[0].variant_id else record
        for record in pending
    ]
    one = generation.benchmark._completed_wave_evaluation(suite, manifest, one_complete, scheduler)
    assert one is not None
    assert one["completed_wave_count"] == 1
    assert len(one["variants"]) == 1
    assert one["final_recommendation_available"] is False

    two_complete = [successful_by_unit[record["work_unit_id"]] if record["variant_id"] in first_two else record for record in pending]
    two = generation.benchmark._completed_wave_evaluation(suite, manifest, two_complete, scheduler)
    assert two is not None
    assert two["completed_wave_count"] == 2
    assert {record["variant_id"] for record in two["variants"]} == first_two
    assert two["recommended_cores_per_case"] in {variant.cores_per_case for variant in suite.variant_wave_order()[:2]}
    assert two["final_recommendation_available"] is False

    complete = generation.benchmark._completed_wave_evaluation(suite, manifest, successes, scheduler)
    assert complete is not None
    assert complete["completed_wave_count"] == 4
    assert complete["final_recommendation_available"] is True
