# ruff: noqa: S101, SLF001
"""Core-benchmark resources remain outside generated scientific identity."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation

if TYPE_CHECKING:
    from collections.abc import Mapping

_COMMIT = "a" * 40


def test_resource_change_preserves_case_identity(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep scheduler resources operational while generated science stays stable."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", _COMMIT)
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    case_config = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    baseline_variant = generation.benchmark.CoreBenchmarkVariant(
        source_path=config_path.parent / "one_core.yaml",
        variant_id="one_core",
        cores_per_case=1,
    )
    suite = generation.benchmark.CoreBenchmarkSuite(
        source_path=config_path.parent / "suite.yaml",
        suite_name="synthetic_core_scaling",
        suite_digest="c" * 64,
        case_campaign_path=config_path,
        case_campaign=campaign,
        case_config=case_config,
        case_index=1,
        repetitions=1,
        variants=(baseline_variant,),
        cores_per_node=24,
        partition="test",
        wall_time=None,
        scheduler_options=(),
        production_campaign_path=config_path,
        production_cores_config_path=config_path.parent / "execution.yaml",
        production_cores_key="cluster.cores_per_case",
        production_cores_per_case=1,
    )
    changed_variant = replace(baseline_variant, cores_per_case=2)

    baseline = generation.cases.case.generate_case_input_bundle(
        case_config,
        suite.case_index,
        tmp_path / "baseline",
    )
    changed = generation.cases.case.generate_case_input_bundle(
        case_config,
        suite.case_index,
        tmp_path / "changed-resource",
    )

    assert suite.execution_id(changed_variant) != suite.execution_id(baseline_variant)
    assert changed.case_input_id == baseline.case_input_id
    assert changed.simulation_case_id == baseline.simulation_case_id
    assert changed.case_payload == baseline.case_payload


def _synthetic_suite(
    generation_config_factory: Any,
) -> generation.benchmark.CoreBenchmarkSuite:
    """Return a four-variant suite with a unique production-core canary."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        scheduler_kind="slurm",
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
        suite_name="synthetic_core_scaling",
        suite_digest="c" * 64,
        case_campaign_path=config_path,
        case_campaign=campaign,
        case_config=case_config,
        case_index=1,
        repetitions=3,
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


def test_standalone_identity_sequence_and_readable_jobs(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Bind only benchmark dependencies and put production repetition one first."""
    suite = _synthetic_suite(generation_config_factory)
    version = generation.benchmark._comsol_version_evidence(
        "COMSOL Multiphysics 6.4",
        configured_executable=suite.resource_contract()["comsol_executable"],
    )
    identity = generation.benchmark._benchmark_identity(
        suite,
        git_commit=_COMMIT,
        comsol_version=version,
    )
    serialized = json.dumps(identity, sort_keys=True).lower()
    assert identity["schema_version"] == 1
    assert "smoke" not in serialized
    assert "dataset" not in serialized
    assert "storage_root" not in identity
    assert "background" not in identity

    sequence = generation.benchmark._measured_sequence(suite)
    assert sequence[0] == (suite.canary_variant(), 1)
    assert len(sequence) == len(suite.variants) * suite.repetitions
    assert len(set(sequence)) == len(sequence)
    selected = generation.benchmark._measured_sequence(
        suite,
        variant_id="cores_08",
    )
    assert selected == tuple((suite.variant("cores_08"), repetition) for repetition in range(1, 4))
    duplicate = replace(
        suite,
        variants=(*suite.variants, replace(suite.variants[0], variant_id="cores_16_duplicate", cores_per_case=16)),
    )
    with pytest.raises(ValueError, match="exactly one variant"):
        duplicate.canary_variant()

    project_root = common.paths.get_project_root()
    launcher = project_root / "scripts/generation_benchmark_node.sh"
    source_launcher = Path(__file__).resolve().parents[2] / "scripts/generation_benchmark_node.sh"
    shutil.copy2(source_launcher, launcher)
    command = generation.benchmark.build_core_benchmark_slurm_command(
        suite,
        run_id="core_scaling_transient__0123456789abcdef",
        storage_root=tmp_path.resolve(),
        log_directory=(tmp_path / "logs").resolve(),
        role="measure",
        variant=suite.variant("cores_16"),
        repetition=2,
    )
    assert "--job-name=td-bench-c16-r02-0123" in command
    assert not any(argument.startswith("--licenses") for argument in command)
    assert "-usebatchlic" not in command[-1]
    wrapped = shlex.split(command[-1].removeprefix("--wrap="))
    assert wrapped[-6:] == [
        str(launcher),
        str(project_root),
        "core_scaling_transient__0123456789abcdef",
        "measure",
        "cores_16",
        "2",
    ]


def test_benchmark_attempt_history_is_append_only_and_hash_chained(
    tmp_path: Path,
) -> None:
    """Bind every retry to the immutable immediately preceding receipt."""
    stable = {
        "schema_version": 1,
        "benchmark_run_id": "core_scaling_transient__0123456789abcdef",
        "suite_digest": "1" * 64,
        "variant_id": "cores_16",
        "execution_id": "execution-16",
        "repetition": 1,
        "repetition_id": "repetition-01",
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
    attempts = (first_path, second_path)

    generation.benchmark._validate_benchmark_attempt_chain(attempts)
    immutable_first = first_path.read_bytes()
    corrupted = json.loads(first_path.read_text(encoding="utf-8"))
    corrupted["schema_version"] = 1
    corrupted["status"] = "rewritten"
    common.serialization.atomic_write_json(first_path, corrupted)
    with pytest.raises(ValueError, match="predecessor chain"):
        generation.benchmark._validate_benchmark_attempt_chain(attempts)
    first_path.write_bytes(immutable_first)

    changed_identity = json.loads(second_path.read_text(encoding="utf-8"))
    changed_identity["case_input_id"] = "9" * 64
    common.serialization.atomic_write_json(second_path, changed_identity)
    with pytest.raises(ValueError, match="identity changed"):
        generation.benchmark._validate_benchmark_attempt_chain(attempts)


def test_standalone_preflight_requires_clean_source_without_smoke(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run benchmark-owned preflight with no Real-Smoke directory and reject dirt."""
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
    monkeypatch.setattr(generation.benchmark, "load_core_benchmark_suite", lambda *_args, **_kwargs: suite)
    monkeypatch.setattr(
        generation.benchmark,
        "build_core_benchmark_slurm_command",
        lambda *_args, role, variant=None, repetition=None, **_kwargs: [
            "sbatch",
            role,
            "-" if variant is None else variant.variant_id,
            "-" if repetition is None else str(repetition),
        ],
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
    assert receipt["schema_version"] == 1
    assert not (storage / "01_generation/meta/real_smoke").exists()

    monkeypatch.undo()
    monkeypatch.setattr(
        generation.benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["git", "status"],
            0,
            stdout=" M src/generation/generation_benchmark.py\n",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="clean exact committed source"):
        generation.benchmark._require_clean_repository()


def _minimal_manifest(run_id: str) -> dict[str, Any]:
    """Return mutable orchestration state needed by one synthetic submit step."""
    return {
        "benchmark_run_id": run_id,
        "git_commit": _COMMIT,
        "preparation_job_ids": [],
        "measured_job_ids": [],
        "submission_history": [],
        "state": "ready",
    }


def test_canary_gate_blocks_remaining_measurements_and_scientific_failure(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit only the measured canary until success and stop on real failure."""
    suite = _synthetic_suite(generation_config_factory)
    run_id = "core_scaling_transient__0123456789abcdef"
    storage = tmp_path / "storage"
    directory = generation.benchmark.core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True)
    (directory / "canonical_case.json").write_text("{}\n", encoding="utf-8")
    proof: Mapping[str, Any] = {}
    monkeypatch.setattr(generation.benchmark, "_load_case_proof", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(generation.benchmark, "_validated_canary_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark, "_canary_attempts", lambda *_args, **_kwargs: ())
    submitted: list[tuple[str | None, int | None]] = []

    def command(
        _suite: Any,
        *,
        variant: Any = None,
        repetition: int | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        submitted.append((None if variant is None else variant.variant_id, repetition))
        return ["sbatch"]

    monkeypatch.setattr(generation.benchmark, "build_core_benchmark_slurm_command", command)
    monkeypatch.setattr(generation.benchmark, "_submit", lambda *_args, **_kwargs: "501")
    manifest = generation.benchmark._submit_pending(
        _minimal_manifest(run_id),
        suite,
        storage=storage,
        variant_id=None,
    )
    assert submitted == [(suite.canary_variant().variant_id, 1)]
    assert manifest["submission_history"][0]["repetitions"] == [1]
    assert (suite.canary_variant(), 1) in generation.benchmark._measured_sequence(suite)

    submitted.clear()
    monkeypatch.setattr(
        generation.benchmark,
        "_validated_canary_result",
        lambda *_args, **_kwargs: {"status": "success"},
    )
    monkeypatch.setattr(
        generation.benchmark,
        "_pending_repetitions",
        lambda _directory, selected_suite, variant: () if variant == selected_suite.canary_variant() else (1,),
    )
    generation.benchmark._submit_pending(
        _minimal_manifest(run_id),
        suite,
        storage=storage,
        variant_id=None,
    )
    assert submitted == [(suite.variants[0].variant_id, 1)]

    monkeypatch.setattr(
        generation.benchmark,
        "_validated_canary_result",
        lambda *_args, **_kwargs: None,
    )
    failed_path = tmp_path / "attempt-0001.json"
    failed_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    monkeypatch.setattr(generation.benchmark, "_canary_attempts", lambda *_args, **_kwargs: (failed_path,))
    monkeypatch.setattr(generation.benchmark, "_validate_failure_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generation.benchmark,
        "_submit",
        lambda *_args, **_kwargs: pytest.fail("remaining measurement submitted after canary failure"),
    )
    failed = generation.benchmark._submit_pending(
        _minimal_manifest(run_id),
        suite,
        storage=storage,
        variant_id=None,
    )
    assert failed["state"] == "canary_failed"


def test_canary_license_capacity_remains_blocked_not_failed(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep temporary canary capacity operationally blocked without further jobs."""
    suite = _synthetic_suite(generation_config_factory)
    run_id = "core_scaling_transient__fedcba9876543210"
    storage = tmp_path / "storage"
    directory = generation.benchmark.core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True)
    (directory / "canonical_case.json").write_text("{}\n", encoding="utf-8")
    pending_path = tmp_path / "attempt-0001.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "pending",
                "temporary_license_retry": {"retry_budget_remaining": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(generation.benchmark, "_load_case_proof", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generation.benchmark, "_validated_canary_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark, "_canary_attempts", lambda *_args, **_kwargs: (pending_path,))
    monkeypatch.setattr(generation.benchmark, "_validate_pending_license_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation.benchmark.license_service, "retry_attempt_is_eligible", lambda _retry: False)
    monkeypatch.setattr(
        generation.benchmark,
        "_submit",
        lambda *_args, **_kwargs: pytest.fail("license-blocked canary submitted prematurely"),
    )
    blocked = generation.benchmark._submit_pending(
        _minimal_manifest(run_id),
        suite,
        storage=storage,
        variant_id=None,
    )
    assert blocked["state"] == "license_blocked"


def test_exhausted_benchmark_license_capacity_remains_operationally_pending(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite wait boundary must not turn capacity into scientific failure."""
    suite = _synthetic_suite(generation_config_factory)
    monkeypatch.setattr(
        generation.benchmark.license_service,
        "bounded_retry_delay_seconds",
        lambda _policy, **_kwargs: 0.0,
    )
    attempt = {
        "attempt": 1,
        "previous_attempt": None,
        "status": "pending",
        "recorded_at": "2026-08-18T15:45:01+00:00",
        "benchmark_run_id": "core_scaling_transient__0123456789abcdef",
        "suite_digest": suite.suite_digest,
        "variant_id": suite.canary_variant().variant_id,
        "execution_id": suite.execution_id(suite.canary_variant()),
        "repetition": 1,
        "repetition_id": suite.repetition_id(suite.canary_variant(), 1),
        "git_commit": _COMMIT,
        "case_input_id": "1" * 64,
        "simulation_case_id": "2" * 64,
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "benchmark_preflight_sha256": "3" * 64,
        "cores_per_case": suite.canary_variant().cores_per_case,
        "temporary_license_retry": {
            "classification": generation.runtime.license.TEMPORARY_LICENSE_CAPACITY,
            "detected_feature": "COMSOL",
            "detected_license_code": "-4",
            "matched_signatures": ["licensed_users_reached"],
            "retry_attempt_index": 1,
            "delay_before_next_attempt_seconds": 0.0,
            "cumulative_wait_seconds": 0.0,
            "retry_budget_remaining": False,
            "next_eligible_at": None,
        },
    }
    path = common.serialization.atomic_write_json(
        tmp_path / "attempt-0001.json",
        attempt,
    )

    assert generation.benchmark._validated_benchmark_license_retry_history(
        suite.case_config,
        (path,),
        repetition_label=suite.repetition_id(suite.canary_variant(), 1),
    ) == (1, 0.0, True)


def test_smoke_bound_preflight_shape_is_rejected_without_legacy_reading(
    generation_config_factory: Any,
) -> None:
    """Keep schema-v1 standalone evidence strict instead of interpreting Smoke-era keys."""
    suite = _synthetic_suite(generation_config_factory)
    old_smoke_bound = {
        "schema_kind": generation.benchmark.BENCHMARK_PREFLIGHT_SCHEMA_KIND,
        "schema_version": 1,
        "smoke_gate_digest": "f" * 64,
    }
    with pytest.raises(ValueError, match="preflight receipt is malformed"):
        generation.benchmark._validate_preflight_payload(
            old_smoke_bound,
            run_id="core_scaling_transient__0123456789abcdef",
            suite=suite,
            git_commit=_COMMIT,
        )
