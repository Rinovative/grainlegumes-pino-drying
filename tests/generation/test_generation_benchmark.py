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
from typing import TYPE_CHECKING, Any

import pytest

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


def test_maintained_suite_is_two_cases_four_waves_and_eight_measurements() -> None:
    """Resolve the sole fast benchmark with production cores first."""
    repository = common.paths.get_project_root()
    suite_path = repository / "configs/generation/benchmarks/transient_core_scaling/suite.yaml"
    suite = generation.benchmark.load_core_benchmark_suite(
        suite_path,
        require_executable=False,
    )
    inspection = generation.benchmark.inspect_core_benchmark(
        suite_path,
        require_executable=False,
    )
    assert [case["case_role"] for case in inspection["representative_cases"]] == [
        "nominal",
        "natural",
    ]
    assert [case["case_index"] for case in inspection["representative_cases"]] == [1, 2]
    wave_cores = [wave["cores_per_case"] for wave in inspection["variant_waves"]]
    assert wave_cores[0] == suite.production_cores_per_case
    assert wave_cores[1:] == sorted(variant.cores_per_case for variant in suite.variants if variant.cores_per_case != suite.production_cores_per_case)
    assert inspection["parallel_cases_per_variant"] == 2
    assert inspection["required_successful_measurements"] == 8
    assert inspection["canary_wave"]["included_in_final_measurements"] is True


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
