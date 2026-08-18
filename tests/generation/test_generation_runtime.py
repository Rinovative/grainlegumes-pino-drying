# ruff: noqa: S101, PLR2004, SLF001
"""Canonical conversion, provenance separation, failure, and locking contracts."""

from __future__ import annotations

import json
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src import common, generation
from src.generation.cli import cli_generation as cli_service
from src.generation.contracts import generation_contracts_profiles as profile_contract
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_workspace as workspace

if TYPE_CHECKING:
    from pathlib import Path


def test_processed_publication_layout_rejects_extra_declared_artifact(
    tmp_path: Path,
) -> None:
    """Keep arbitrary retained payloads outside canonical processed cases."""
    directory = tmp_path / "processed-case"
    exports = directory / "comsol_exports"
    exports.mkdir(parents=True)
    required = {
        "case.h5",
        "solver.log",
        "timing.json",
        "status.json",
        "execution_provenance.json",
        "processing_provenance.json",
    }
    for name in {"_SUCCESS", "provenance.json", *required}:
        (directory / name).write_text("evidence\n", encoding="utf-8")
    (exports / "fields.csv").write_text("export\n", encoding="utf-8")
    artifact_names = {*required, "comsol_exports/fields.csv"}

    runtime_service._require_processed_publication_layout(
        directory,
        artifact_names=artifact_names,
        required=required,
    )
    (directory / "unexpected.bin").write_bytes(b"unexpected")
    artifact_names.add("unexpected.bin")
    with pytest.raises(RuntimeError, match="top-level membership"):
        runtime_service._require_processed_publication_layout(
            directory,
            artifact_names=artifact_names,
            required=required,
        )


def _natural_batch_name(simulation_profile: str) -> str:
    """Return one canonical synthetic natural-batch selector."""
    return generation.cases.config.build_batch_name(
        simulation_profile,
        "lentil",
        "natural",
    )


def _prepare_canonical_inputs(
    config: Any,
    storage: Path,
    *,
    case_count: int = 1,
) -> None:
    """Publish exact canonical inputs before exercising worker runtime."""
    generation.cases.input_generation.generate_input_cases(
        config,
        case_count,
        storage_root=storage,
    )


def _record_synthetic_failure(
    config: Any,
    storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    git_commit: str,
    execution_run_id: str,
) -> Path:
    """Record one compact test-owned execution failure."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", git_commit)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", execution_run_id)
    _prepare_canonical_inputs(config, storage)
    return generation.runtime.record_case_failure(
        config,
        1,
        RuntimeError("synthetic case failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )


def _latest_attempt(
    config: Any,
    storage: Path,
    *,
    run_id: str | None = None,
) -> Any:
    """Return one required latest synthetic attempt."""
    attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        config.batch_id if run_id is None else run_id,
        storage_root=storage,
    )
    assert attempt is not None
    return attempt


def test_float32_conversion_requires_explicit_tolerance() -> None:
    """Protect validated conversion rather than silent precision loss."""
    values = np.asarray([1.0, 1.0e-9, 123.456789], dtype=np.float64)
    converted = generation.publication.storage.validate_float32_conversion(values, rtol=1e-6, atol=1e-12, label="synthetic")
    assert converted.dtype == np.float32
    with pytest.raises(ValueError, match="exceeds configured tolerance"):
        generation.publication.storage.validate_float32_conversion(values, rtol=0.0, atol=0.0, label="synthetic")


def _bulk_moisture_arrays(
    discrepancy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a three-time, four-node X_wb_bulk=0.1 consistency fixture."""
    static = np.zeros(
        (len(profile_contract.TRANSIENT_STATIC_FIELD_NAMES), 2, 2),
        dtype=np.float64,
    )
    static[
        profile_contract.TRANSIENT_STATIC_FIELD_NAMES.index(
            "rho_bu_dry",
        )
    ] = 9.0
    time_axis = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
    states = np.zeros(
        (
            time_axis.size,
            len(profile_contract.TRANSIENT_FIELD_NAMES),
            2,
            2,
        ),
        dtype=np.float64,
    )
    for name in ("w_surf", "w_int"):
        states[
            :,
            profile_contract.TRANSIENT_FIELD_NAMES.index(name),
        ] = 1.0
    globals_ = np.zeros(
        (
            time_axis.size,
            len(profile_contract.GLOBAL_FIELD_NAMES),
        ),
        dtype=np.float64,
    )
    globals_[
        :,
        profile_contract.GLOBAL_FIELD_NAMES.index("t"),
    ] = time_axis
    globals_[
        :,
        profile_contract.GLOBAL_FIELD_NAMES.index("X_wb_bulk"),
    ] = 0.1 + discrepancy
    return static, time_axis, states, globals_


@pytest.mark.parametrize(
    ("discrepancy", "expected"),
    [
        (0.0, True),
        (1.8912875530963102e-7, True),
        (5.0e-7, True),
        (1.1e-5, False),
        (1.0e-2, False),
    ],
)
def test_transient_bulk_moisture_semantic_tolerance_is_strict(
    generation_config_factory: Any,
    discrepancy: float,
    expected: bool,
) -> None:
    """Admit solver-scale disagreement and reject material mismatches."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    static, time_axis, states, globals_ = _bulk_moisture_arrays(
        discrepancy,
    )
    result = generation.publication.storage.evaluate_transient_bulk_moisture_consistency(
        config,
        static,
        time_axis,
        states,
        None,
        globals_,
        f_surf=0.4,
        time_tolerance=1.0e-12,
    )

    assert generation.publication.storage.transient_bulk_moisture_tolerance(
        config,
    ) == (1.0e-5, 1.0e-9)
    assert result.matches is expected
    if discrepancy == 1.8912875530963102e-7:
        assert np.max(
            np.abs(result.exported - result.reconstructed),
        ) == pytest.approx(discrepancy)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_bulk_moisture_consistency_fails_closed_on_non_finite_globals(
    generation_config_factory: Any,
    invalid: float,
) -> None:
    """Reject NaN and Inf before semantic comparison."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    static, time_axis, states, globals_ = _bulk_moisture_arrays(0.0)
    globals_[
        1,
        profile_contract.GLOBAL_FIELD_NAMES.index("X_wb_bulk"),
    ] = invalid

    with pytest.raises(ValueError, match="non-finite"):
        (
            generation.publication.storage.evaluate_transient_bulk_moisture_consistency(
                config,
                static,
                time_axis,
                states,
                None,
                globals_,
                f_surf=0.4,
                time_tolerance=1.0e-12,
            )
        )


def test_bulk_moisture_consistency_preserves_strict_time_alignment(
    generation_config_factory: Any,
) -> None:
    """Reject misaligned global time without weakening temporal admission."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    static, time_axis, states, globals_ = _bulk_moisture_arrays(0.0)
    globals_[
        1,
        profile_contract.GLOBAL_FIELD_NAMES.index("t"),
    ] = 1.1

    with pytest.raises(ValueError, match="exactly one row"):
        (
            generation.publication.storage.evaluate_transient_bulk_moisture_consistency(
                config,
                static,
                time_axis,
                states,
                None,
                globals_,
                f_surf=0.4,
                time_tolerance=1.0e-12,
            )
        )


def test_resolved_science_and_execution_are_persisted_separately(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect scientific identity from site and resource settings."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    scientific_path = generation.runtime.initialize_batch_metadata(config, storage_root=tmp_path / "storage")
    assert scientific_path.name == "resolved_generation_config.json"
    scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    serialized = json.dumps(scientific, sort_keys=True)
    assert all(term not in serialized for term in ("max_nodes", "cores_per_case", "partition", "timeout_seconds", "wall_time", "cpu_host"))
    execution_files = list((scientific_path.parent / "execution_configs").glob("*.json"))
    assert len(execution_files) == 1
    execution = json.loads(execution_files[0].read_text(encoding="utf-8"))
    assert execution == config.execution_values
    assert generation.cases.config.compute_scientific_config_digest(scientific) == config.scientific_config_digest


@pytest.mark.parametrize(
    "argument",
    [
        "-job",
        "-job=other",
        "-nosave",
        "-pname",
        "-pname=T_amb",
        "-plist",
        "-plist=298.15[K]",
        "-pindex",
        "-pindex=1",
        "-paramfile",
        "-paramfile=parameters.txt",
    ],
)
def test_extra_arguments_cannot_override_scalar_handoff_flags(
    generation_config_factory: Any,
    argument: str,
) -> None:
    """Keep every COMSOL parameter-injection flag runtime-owned."""
    config_path, _template = generation_config_factory(extra_arguments=(argument,))
    with pytest.raises(generation.cases.config.GenerationConfigError, match="cannot override"):
        generation.cases.config.load_generation_config(
            config_path,
            only_batch=_natural_batch_name("transient_drying"),
        )


def test_comsol_commands_use_the_exact_admitted_runtime_scalar_vector(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protect exact runtime injection, local/Slurm order, and CLI parity."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        extra_arguments=("-recover",),
    )
    batch_name = _natural_batch_name("transient_drying")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    bundle = generation.cases.case.generate_case_input_bundle(
        config,
        1,
        tmp_path / "command bundle",
    )
    admission = bundle.scalar_handoff
    assert admission is not None
    with pytest.raises(ValueError, match="require one admitted scalar handoff"):
        comsol_service.build_comsol_command(config, cores_per_case=2)
    entries = admission.entries
    assert len(entries) == 12
    assert tuple(entry.name for entry in entries) == profile_contract.TRANSIENT_SCALAR_INPUT_FIELDS
    assert all(entry.owner == "case_dependent" for entry in entries)
    assert {"T_init", "T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}.isdisjoint(entry.name for entry in entries)

    local = comsol_service.build_comsol_command(
        config,
        cores_per_case=2,
        scalar_handoff=admission,
    )
    parameter_start = local.index("-pname")
    assert local[parameter_start : parameter_start + 6 : 2] == ["-pname", "-plist", "-pindex"]
    assert local[parameter_start + 1].split(",") == [entry.name for entry in entries]
    expected_values = [scalar_handoff_contract.format_comsol_parameter(entry) for entry in entries]
    assert local[parameter_start + 3].split(",") == expected_values
    assert "[1]" not in local[parameter_start + 3]
    assert local[parameter_start + 5] == ",".join(str(index) for index in range(1, 13))
    assert local[parameter_start + 6 :] == ["-np", "2", "-recover"]

    slurm = comsol_service.build_comsol_command(
        config,
        cores_per_case=2,
        scalar_handoff=admission,
        scheduler_kind="slurm",
    )
    assert slurm == local
    assert "srun" not in slurm
    assert "--exclusive" not in slurm

    status = cli_service.main(
        [
            "print-command",
            str(config_path),
            "--only-batch",
            batch_name,
            "1",
            "--cores-per-case",
            "2",
        ]
    )
    assert status == 0
    assert shlex.split(capsys.readouterr().out.strip()) == local


def test_comsol_builder_owns_fixed_job_and_controlled_stop_output(
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    """Bind every run to solved.mph so its documented status path is fixed."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )

    command = comsol_service.build_comsol_command(config, cores_per_case=16)

    assert command == [
        str(fake_comsol),
        "batch",
        "-inputfile",
        "model.mph",
        "-job",
        "b1",
        "-outputfile",
        "solved.mph",
        "-np",
        "16",
    ]


def test_comsol_builder_rejects_invalid_retention_state(
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    """Fail closed instead of constructing a command without one save mode."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    execution = {**config.execution_values, "retention_policy": None}
    invalid = replace(config, execution_values=execution)

    with pytest.raises(TypeError, match="retention_policy must be resolved"):
        comsol_service.build_comsol_command(invalid, cores_per_case=16)


def test_scalar_source_validation_precedes_evidence_and_process_start(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject changed scalar bytes before provenance or Popen can attest them."""
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "scalar validation storage"
    _prepare_canonical_inputs(config, storage)
    prepared = runtime_service.prepare_case_work_directory(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "scalar validation work",
    )
    admission = prepared.bundle.scalar_handoff
    assert admission is not None
    admission.source_path.write_bytes(admission.source_path.read_bytes() + b"changed")
    process_started = False

    def reject_process_start(*_args: Any, **_kwargs: Any) -> None:
        nonlocal process_started
        process_started = True
        message = "Popen must not be reached"
        raise AssertionError(message)

    monkeypatch.setattr(runtime_service.subprocess, "Popen", reject_process_start)
    with pytest.raises(RuntimeError, match="bytes changed"):
        runtime_service.execute_prepared_case(
            config,
            prepared,
            cores_per_case=1,
            worker_slot=0,
        )
    assert process_started is False
    assert not (prepared.runtime_directory / "execution_provenance.json").exists()


def test_missing_canonical_inputs_never_trigger_worker_generation(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail worker preparation without generating absent canonical inputs."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "missing canonical inputs"
    generation_called = False

    def reject_generation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal generation_called
        generation_called = True
        message = "worker attempted input generation"
        raise AssertionError(message)

    monkeypatch.setattr(
        generation.cases.input_generation,
        "generate_input_cases",
        reject_generation,
    )
    with pytest.raises(
        generation.runtime.CasePreparationError,
        match="Canonical input readiness is required",
    ):
        runtime_service.prepare_case_work_directory(
            config,
            1,
            storage_root=storage,
            work_root=tmp_path / "missing canonical work",
        )

    assert generation_called is False
    assert not tuple(storage.rglob("input_generation_manifest.json"))


def test_steady_command_is_parameter_free(
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    """Keep the steady template authoritative for its fixed conditioning."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    command = comsol_service.build_comsol_command(config, cores_per_case=1)
    assert not {"-pname", "-plist", "-pindex"}.intersection(command)


@pytest.mark.integration
def test_preparation_failure_is_recorded_without_a_work_directory(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect durable status evidence when case preparation itself fails."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "storage"
    _prepare_canonical_inputs(config, storage)

    def reject_preparation(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic preparation failure"
        raise OSError(message)

    monkeypatch.setattr(runtime_service, "prepare_case_work_directory", reject_preparation)
    with pytest.raises(OSError, match="synthetic preparation failure"):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    assert generation.runtime.case_failure_is_recorded(config, 1, storage_root=storage)
    attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        config.batch_id,
        storage_root=storage,
    )
    assert attempt is not None
    assert attempt.payload["case_state"] == "failed"
    assert attempt.payload["failure_stage"] == "input"
    assert "synthetic preparation failure" in attempt.payload["reason"]
    cleanup = json.loads((attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "not_created"


def test_attempt_from_another_campaign_is_not_current(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope append-only attempt evidence to its exact campaign run."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "campaign-scoped attempt storage"
    old_run_id = "old-campaign__0123456789abcdef"
    current_run_id = "current-campaign__0123456789abcdef"
    _record_synthetic_failure(
        config,
        storage,
        monkeypatch,
        git_commit="a" * 40,
        execution_run_id=old_run_id,
    )
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", current_run_id)

    assert not generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
        execution_run_id=current_run_id,
        git_commit="a" * 40,
    )
    old_attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        old_run_id,
        storage_root=storage,
    )
    assert old_attempt is not None


def test_malformed_or_symlinked_failure_receipt_fails_closed(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject unreadable current-state objects and unsafe receipt paths."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    malformed_storage = tmp_path / "malformed storage"
    malformed = generation.runtime.case_failure_path(
        config,
        1,
        storage_root=malformed_storage,
    )
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read case failure evidence"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=malformed_storage,
        )

    symlink_storage = tmp_path / "symlink storage"
    symlink = generation.runtime.case_failure_path(
        config,
        1,
        storage_root=symlink_storage,
    )
    symlink.parent.mkdir(parents=True)
    target = tmp_path / "foreign failure.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="evidence is unsafe"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=symlink_storage,
        )


@pytest.mark.integration
def test_attempt_history_is_append_only_across_failure_and_success(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve old campaign attempts after a newer failure or successful run."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    old_run_id = "old-campaign__0123456789abcdef"
    current_run_id = "current-campaign__0123456789abcdef"
    commit = "a" * 40

    failed_storage = tmp_path / "append-only failure storage"
    old_path = _record_synthetic_failure(
        config,
        failed_storage,
        monkeypatch,
        git_commit=commit,
        execution_run_id=old_run_id,
    )
    old_bytes = old_path.read_bytes()
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", current_run_id)
    with monkeypatch.context() as scoped:

        def reject_preparation(*_args: Any, **_kwargs: Any) -> None:
            message = "new synthetic preparation failure"
            raise OSError(message)

        scoped.setattr(
            runtime_service,
            "prepare_case_work_directory",
            reject_preparation,
        )
        with pytest.raises(OSError, match="new synthetic preparation failure"):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=failed_storage,
                work_root=tmp_path / "new failure work",
            )
    assert old_path.read_bytes() == old_bytes
    current_attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        current_run_id,
        storage_root=failed_storage,
    )
    assert current_attempt is not None
    assert current_attempt.payload["case_state"] == "failed"

    success_storage = tmp_path / "append-only recovery storage"
    success_path = _record_synthetic_failure(
        config,
        success_storage,
        monkeypatch,
        git_commit=commit,
        execution_run_id=old_run_id,
    )
    success_bytes = success_path.read_bytes()
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", current_run_id)
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=success_storage,
        work_root=tmp_path / "recovery work",
    )
    assert outcome.status == "completed"
    assert success_path.read_bytes() == success_bytes
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=success_storage,
    )
    assert not generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=success_storage,
        execution_run_id=current_run_id,
        git_commit=commit,
    )


@pytest.mark.integration
def test_failure_timeout_missing_export_and_case_lock(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect evidence-first cleanup, timeout handling, exports, and locking."""
    config_path, _template = generation_config_factory(executable=fake_comsol, timeout=0.1)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    _prepare_canonical_inputs(config, storage)
    _prepare_canonical_inputs(config, tmp_path / "timeout-storage")

    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    with pytest.raises(generation.runtime.CaseExecutionError) as failed:
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=storage, work_root=work)
    assert not failed.value.work_directory.exists()
    failed_attempt = _latest_attempt(config, storage)
    assert failed_attempt.payload["case_state"] == "failed"
    assert failed_attempt.payload["failure_stage"] == "solver"
    assert failed_attempt.payload["process_exit_code"] == 7
    assert failed_attempt.payload["template"]["sha256"] == config.template_sha256
    assert failed_attempt.payload["retention_policy"] == "full"
    assert "payload/runtime/solver.log" in failed_attempt.payload["retained_inventory"]
    cleanup = json.loads((failed_attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "complete"

    monkeypatch.setenv("FAKE_COMSOL_MODE", "timeout")
    with pytest.raises(generation.runtime.CaseExecutionError, match="timeout") as timed_out:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=tmp_path / "timeout-storage",
            work_root=tmp_path / "timeout-work",
        )
    assert not timed_out.value.work_directory.exists()
    timeout_attempt = _latest_attempt(config, tmp_path / "timeout-storage")
    assert timeout_attempt.payload["case_state"] == "timed_out"
    assert timeout_attempt.payload["timed_out"] is True

    monkeypatch.setenv("FAKE_COMSOL_MODE", "success")
    prepared = runtime_service.prepare_case_work_directory(config, 1, storage_root=storage, work_root=work)
    with pytest.raises(FileNotFoundError, match=r"airflow\.csv"):
        generation.runtime.collect_exports(config, prepared)
    workspace.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=prepared.work_root,
        storage_root=storage.resolve(),
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
    )

    _prepare_canonical_inputs(config, tmp_path / "locked")
    lock_path = generation.runtime.case_lock_path(
        config,
        1,
        storage_root=tmp_path / "locked",
    )
    with common.locking.exclusive_file_lock(lock_path, blocking=False), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generation.runtime.run_case,
            config,
            1,
            cores_per_case=1,
            storage_root=tmp_path / "locked",
            work_root=tmp_path / "locked-work",
            blocking_lock=False,
        )
        with pytest.raises(common.locking.FileLockUnavailableError):
            future.result()


@pytest.mark.integration
def test_compact_case_uses_solved_output_but_omits_model_from_publication(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Use solved.mph for stop control while keeping compact publication bounded."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        retain_solved_model=False,
        retain_raw_csv=False,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "no-save storage"
    _prepare_canonical_inputs(config, storage)
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=16,
        storage_root=storage,
        work_root=tmp_path / "no-save work",
    )

    assert outcome.status == "completed"
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
    assert not (outcome.processed_directory / "solved.mph").exists()
    execution = json.loads((outcome.processed_directory / "execution_provenance.json").read_text(encoding="utf-8"))
    arguments = execution["invocation"]["arguments"]
    assert arguments[:10] == [
        str(fake_comsol),
        "batch",
        "-inputfile",
        "model.mph",
        "-job",
        "b1",
        "-outputfile",
        "solved.mph",
        "-np",
        "16",
    ]
    assert "-nosave" not in arguments
    assert execution["result"]["solved_model"]["canonical_relative_path"] == "solved.mph"
    assert (outcome.processed_directory / "case.h5").is_file()
    processing = json.loads((outcome.processed_directory / "processing_provenance.json").read_text(encoding="utf-8"))
    assert processing["mode"] == "initial"
    assert processing["solver_git_commit"] == processing["processing_git_commit"]


@pytest.mark.integration
def test_suffixed_solver_output_is_published_canonically_before_cleanup(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish a valid canonical solved model before removing case scratch."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        retain_solved_model=True,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "storage with spaces"
    _prepare_canonical_inputs(config, storage)
    cleanup_observations: list[bool] = []
    original_cleanup = workspace.cleanup_case_workspace

    def cleanup_after_publication(*args: Any, **kwargs: Any) -> int:
        cleanup_observations.append(
            generation.runtime.completed_case_is_valid(
                config,
                1,
                storage_root=storage,
            )
        )
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(
        workspace,
        "cleanup_case_workspace",
        cleanup_after_publication,
    )
    monkeypatch.setenv("FAKE_COMSOL_SOLVED_MODEL_MODE", "suffixed")

    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "work with spaces",
    )

    assert outcome.status == "completed"
    assert cleanup_observations == [True]
    assert outcome.work_directory is not None
    assert not outcome.work_directory.exists()
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
    solved_model = outcome.processed_directory / "solved.mph"
    assert solved_model.is_file()
    assert solved_model.stat().st_size > 0


@pytest.mark.integration
def test_solver_rejects_unchanged_stale_and_ambiguous_solved_outputs(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exactly one valid solver-produced output after a zero exit."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        retain_solved_model=True,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage_root = tmp_path / "invalid solved storage"
    work_root = tmp_path / "invalid solved work"
    _prepare_canonical_inputs(config, storage_root, case_count=len(config.case_indices))
    scenarios = (
        ("missing", True, "observed none", ("solved*.mph",)),
        ("multiple", False, "solved_1.mph, solved_2.mph", ("solved_1.mph", "solved_2.mph")),
        ("empty", False, "unsafe or empty", ("solved.mph",)),
        ("symlink", False, "unsafe or empty", ("solved.mph",)),
    )
    for worker_slot, (mode, create_stale, match, artifacts) in enumerate(scenarios):
        case_index = config.case_indices[worker_slot % len(config.case_indices)]
        prepared = runtime_service.prepare_case_work_directory(
            config,
            case_index,
            storage_root=storage_root,
            work_root=work_root,
        )
        stale_payload = b"stale solved model\n"
        if create_stale:
            (prepared.work_directory / "solved.mph").write_bytes(stale_payload)
        monkeypatch.setenv("FAKE_COMSOL_SOLVED_MODEL_MODE", mode)
        with pytest.raises(generation.runtime.CaseExecutionError, match=match) as caught:
            runtime_service.execute_prepared_case(
                config,
                prepared,
                cores_per_case=1,
                worker_slot=worker_slot,
            )
        assert caught.value.exit_code == 0
        assert caught.value.missing_or_invalid_artifacts == artifacts
        if create_stale:
            assert (prepared.work_directory / "solved.mph").read_bytes() == stale_payload
        execution = json.loads((prepared.runtime_directory / "execution_provenance.json").read_text(encoding="utf-8"))
        assert execution["result"]["state"] == "succeeded"
        assert execution["result"]["solved_model"] is None
        rejected_output = prepared.work_directory / "solved.mph"
        if mode == "symlink":
            assert rejected_output.is_symlink()
            rejected_output.unlink()
        workspace.cleanup_case_workspace(
            prepared.work_directory,
            allowed_root=prepared.work_root,
            storage_root=storage_root.resolve(),
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
        )


@pytest.mark.integration
def test_two_concurrent_cases_keep_inputs_exports_and_workspaces_isolated(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect real case-workspace isolation under concurrent fake solvers."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=2,
        retain_raw_csv=True,
        timeout=60.0,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "concurrent storage"
    work = tmp_path / "concurrent work"
    _prepare_canonical_inputs(config, storage, case_count=2)
    tracker = tmp_path / "fake-comsol-tracker.json"
    monkeypatch.setenv("FAKE_COMSOL_TRACKER", str(tracker))
    monkeypatch.setenv("FAKE_COMSOL_EXPECT_STARTS", "2")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                generation.runtime.run_case,
                config,
                case_index,
                cores_per_case=1,
                storage_root=storage,
                work_root=work,
            )
            for case_index in (1, 2)
        ]
        outcomes = [future.result() for future in futures]
    assert {outcome.case_id for outcome in outcomes} == {"case_0001", "case_0002"}
    work_directories = {outcome.work_directory for outcome in outcomes}
    assert None not in work_directories
    assert len(work_directories) == 2
    assert all(not directory.exists() for directory in work_directories if directory is not None)
    retained_inputs = [
        generation.runtime.raw_case_directory(
            config,
            case_index,
            storage_root=storage,
        )
        / "inputs/fields.csv"
        for case_index in (1, 2)
    ]
    assert all(path.is_file() for path in retained_inputs)
    assert retained_inputs[0].read_bytes() != retained_inputs[1].read_bytes()
    for case_index in (1, 2):
        generation.runtime.validate_completed_case(
            config,
            case_index,
            storage_root=storage,
        )
    concurrency = json.loads(tracker.read_text(encoding="utf-8"))
    assert concurrency["starts"] == 2
    assert concurrency["maximum"] == 2


@pytest.mark.integration
def test_publication_failure_records_evidence_before_scratch_cleanup(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect fail-closed publication, evidence-first cleanup, and no completion."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "publication failure storage"
    _prepare_canonical_inputs(config, storage)
    original_cleanup = workspace.cleanup_case_workspace
    observed = {"publication_called": False, "failure_before_cleanup": False}

    def fail_publication(*_args: Any, **_kwargs: Any) -> None:
        observed["publication_called"] = True
        message = "synthetic publication validation failure"
        raise RuntimeError(message)

    def evidence_first_cleanup(*args: Any, **kwargs: Any) -> int:
        attempt = _latest_attempt(config, storage)
        observed["failure_before_cleanup"] = (
            attempt.payload["case_state"] == "publication_failed"
            and attempt.payload["failure_stage"] == "publication"
            and not (attempt.directory / "cleanup.json").exists()
        )
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(
        runtime_service,
        "publish_completed_case",
        fail_publication,
    )
    monkeypatch.setattr(
        workspace,
        "cleanup_case_workspace",
        evidence_first_cleanup,
    )
    with pytest.raises(RuntimeError, match="synthetic publication validation failure"):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "publication failure work",
        )
    assert observed == {
        "publication_called": True,
        "failure_before_cleanup": True,
    }
    attempt = _latest_attempt(config, storage)
    assert attempt.payload["case_state"] == "publication_failed"
    assert attempt.payload["solver_state"] == "succeeded"
    assert attempt.payload["conversion_state"] == "succeeded"
    assert attempt.payload["publication_state"] == "failed"
    cleanup = json.loads((attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )


@pytest.mark.integration
def test_runtime_cancellation_terminates_solver_and_persists_cancelled_case(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect controlled-stop cancellation, attempt evidence, and cleanup."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        timeout=5.0,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "cancelled storage"
    _prepare_canonical_inputs(config, storage)
    tracker = tmp_path / "cancelled tracker.json"
    monkeypatch.setenv("FAKE_COMSOL_MODE", "timeout")
    monkeypatch.setenv("FAKE_COMSOL_TRACKER", str(tracker))
    generation.runtime.reset_runtime_cancellation()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                generation.runtime.run_case,
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "cancelled work",
            )
            deadline = time.monotonic() + 5.0
            while not tracker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert tracker.exists()
            generation.runtime.request_runtime_cancellation()
            with pytest.raises(generation.runtime.CaseInterruptedError):
                future.result(timeout=10.0)
    finally:
        generation.runtime.reset_runtime_cancellation()
    attempt = _latest_attempt(config, storage)
    assert attempt.payload["case_state"] == "cancelled"
    assert attempt.payload["failure_stage"] == "solver"
    assert attempt.payload["process_exit_code"] is not None
    assert (attempt.directory / "payload" / "solved.mph.status").read_text(encoding="utf-8") == "Stop 2\n"
    stop = json.loads((attempt.directory / "payload/runtime/stop.json").read_text(encoding="utf-8"))
    assert stop["reason"] == "cancelled"
    cleanup = json.loads((attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
