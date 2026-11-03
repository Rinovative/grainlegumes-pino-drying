"""Canonical conversion, provenance separation, failure, and locking contracts.

Responsibilities:
  - Check conversion, attempt history, exact solver argv, publication, locks, and cancellation

Design principles:
  - Scientific and operational provenance stay separate; failures retain evidence

This module does NOT:
  - Execute licensed production solvers or generate missing canonical inputs
"""

from __future__ import annotations

import json
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cli import cli_generation as cli_service
from src.generation.contracts import generation_contracts_profiles as profile_contract
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity
from src.generation.runtime import generation_runtime_workspace as workspace


def test_processed_publication_layout_rejects_extra_declared_artifact(
    tmp_path: Path,
) -> None:
    """Keep arbitrary retained payloads outside canonical result directories."""
    directory = tmp_path / "result"
    evidence = directory / "evidence"
    exports = evidence / "comsol_exports"
    exports.mkdir(parents=True)
    required = {
        "case.h5",
        "evidence/solver.log",
        "evidence/timing.json",
        "evidence/status.json",
        "evidence/execution_provenance.json",
        "evidence/processing_provenance.json",
    }
    for name in {"_SUCCESS", "provenance.json", *required}:
        (directory / name).write_text("evidence\n", encoding="utf-8")
    (exports / "fields.csv").write_text("export\n", encoding="utf-8")
    (evidence / "solved.mph").write_bytes(b"model")
    artifact_names = {*required, "evidence/comsol_exports/fields.csv", "evidence/solved.mph"}

    runtime_service._require_processed_publication_layout(
        directory,
        artifact_names=artifact_names,
        required=required,
        retention_policy="full",
    )
    for unexpected in (directory / "unexpected.bin", evidence / "unexpected.bin"):
        unexpected.write_bytes(b"unexpected")
        artifact_names.add(unexpected.relative_to(directory).as_posix())
        with pytest.raises(RuntimeError, match="top-level membership"):
            runtime_service._require_processed_publication_layout(
                directory,
                artifact_names=artifact_names,
                required=required,
                retention_policy="full",
            )
        unexpected.unlink()
        artifact_names.discard(unexpected.relative_to(directory).as_posix())


def _bind_campaign_run(
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    *,
    allocation_id: str = "4242",
    submission_index: int = 1,
    step_id: str = "0",
) -> None:
    """Make the process one campaign case step with a complete dispatch identity."""
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    monkeypatch.setenv("SLURM_JOB_ID", allocation_id)
    monkeypatch.setenv("SLURM_STEP_ID", step_id)
    monkeypatch.setenv("GENERATION_SUBMISSION_INDEX", str(submission_index))
    monkeypatch.setenv("GENERATION_DISPATCH_KEY", dispatch_identity.dispatch_key(allocation_id, submission_index))


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
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    lifecycle.create_campaign(campaign, storage_root=storage)
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
    _bind_campaign_run(monkeypatch, execution_run_id)
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


def _latest_attempt(config: Any, storage: Path) -> Any:
    """Return one required latest synthetic attempt."""
    attempt = generation.publication.attempt.latest_case_attempt(config, 1, storage_root=storage)
    assert attempt is not None
    return attempt


def _current_run_id(config: Any, *, git_commit: str = "a" * 40) -> str:
    """Resolve the run ID bound to the authored source campaign."""
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    return generation.campaign.campaign_run_id(campaign, git_commit=git_commit)


def _canonical_failed_attempt_config(generation_config_factory: Any, tmp_path: Path) -> tuple[Any, Path, str]:
    """Prepare one current source input for unsuccessful attempt publication."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "canonical-attempt-storage"
    generation.cases.input_generation.prepare_campaign_inputs(campaign, git_commit="a" * 40, storage_root=storage)
    return campaign.batches[0], storage, generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)


def test_canonical_unsuccessful_attempt_publishes_and_loads_by_exact_locator(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep failed v2 runtime evidence beside its reserved material case."""
    config, storage, run_id = _canonical_failed_attempt_config(generation_config_factory, tmp_path)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    _bind_campaign_run(monkeypatch, run_id)
    receipt = runtime_service.record_case_failure(
        config,
        1,
        RuntimeError("synthetic input rejection"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    expected = common.paths.resolve_generation_attempt_run_directory(config.batch_storage_name, config.case_id(1), 1, storage_root=storage)
    assert receipt == expected / "attempt.json"
    loaded = generation.publication.attempt.load_attempt(expected)
    assert loaded.payload["case_state"] == "failed"
    assert (loaded.payload["job_id"], loaded.payload["slurm_step_id"], loaded.payload["submission_index"]) == ("4242", "0", 1)
    assert generation.publication.attempt.latest_case_attempt(config, 1, storage_root=storage) == loaded
    wrong_target = storage / "01_generation" / "unrelated" / "run_0002"
    with pytest.raises(ValueError, match="authoritative storage locator"):
        generation.publication.attempt._safe_attempt_target(
            wrong_target,
            storage_root=storage,
            batch_storage_name=config.batch_storage_name,
            case_id=config.case_id(1),
            attempt_index=2,
        )
    linked_history = storage / "01_generation" / "linked-history"
    linked_history.symlink_to(tmp_path / "outside-history", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        generation.publication.attempt._safe_attempt_history_root(linked_history / "run", storage_root=storage)
    next_target = common.paths.resolve_generation_attempt_run_directory(config.batch_storage_name, config.case_id(1), 2, storage_root=storage)
    next_target.symlink_to(tmp_path / "outside-attempt")
    with pytest.raises(ValueError, match="not one ordinary directory"):
        runtime_service.record_case_failure(
            config,
            1,
            RuntimeError("another synthetic input rejection"),
            worker_slot=0,
            scheduler_kind="slurm",
            allocated_node="node01",
            work_directory=None,
            storage_root=storage,
            scratch_cleanup_status="not_created",
            failure_stage="input",
        )
    assert not (tmp_path / "outside-attempt").exists()


def test_failed_attempt_retention_respects_full_and_compact_policy(generation_config_factory: Any, tmp_path: Path) -> None:
    """Retain present owned scientific files and complete logs only under full policy."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("steady_flow"))
    work = tmp_path / "case"
    (work / "runtime").mkdir(parents=True)
    log = b"accepted solver progress\n" * 50_000
    (work / "runtime" / "solver.log").write_bytes(log)
    (work / "model.mph").write_bytes(b"input model")
    (work / "solved.mph").write_bytes(b"partial solved model")
    (work / "runtime" / "case.h5").write_bytes(b"present converted bytes")

    full_config = replace(config, execution_values={**config.execution_values, "retention_policy": "full"})
    full_policy, full_paths, full_omitted = generation.publication.attempt._retention_paths(full_config, work_directory=work)
    assert full_policy == "full"
    assert full_omitted == []
    assert {Path("model.mph"), Path("solved.mph"), Path("runtime/case.h5")} <= set(full_paths)
    full_inventory = generation.publication.attempt._copy_retained(work, tmp_path / "full", full_paths, full_retention=True)
    assert full_inventory["payload/runtime/solver.log"]["size_bytes"] == len(log)
    assert (tmp_path / "full" / "payload" / "runtime" / "solver.log").read_bytes() == log

    compact_config = replace(config, execution_values={**config.execution_values, "retention_policy": "compact"})
    compact_policy, compact_paths, compact_omitted = generation.publication.attempt._retention_paths(compact_config, work_directory=work)
    assert compact_policy == "compact"
    assert {"model.mph", "solved.mph", "runtime/case.h5"} <= set(compact_omitted)
    assert {Path("model.mph"), Path("solved.mph"), Path("runtime/case.h5")}.isdisjoint(compact_paths)
    compact_inventory = generation.publication.attempt._copy_retained(work, tmp_path / "compact", compact_paths, full_retention=False)
    assert compact_inventory["payload/runtime/solver.log"]["size_bytes"] < len(log)


def test_attempt_index_rejects_symlinked_history_before_listing(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never inspect attempt names through a relocated canonical history."""
    config, storage, run_id = _canonical_failed_attempt_config(generation_config_factory, tmp_path)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    _bind_campaign_run(monkeypatch, run_id)
    target = common.paths.resolve_generation_attempt_run_directory(config.batch_storage_name, config.case_id(1), 1, storage_root=storage)
    outside = tmp_path / "outside-history"
    outside.mkdir()
    target.parent.parent.mkdir(parents=True, exist_ok=True)
    target.parent.symlink_to(outside, target_is_directory=True)
    history = target.parent
    original_iterdir = Path.iterdir

    def reject_outside_listing(path: Path) -> Any:
        if path == history:
            pytest.fail("Attempt index listed a symlinked history before containment validation")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", reject_outside_listing)
    with pytest.raises(ValueError, match="symbolic link"):
        runtime_service.record_case_failure(
            config,
            1,
            RuntimeError("synthetic input rejection"),
            worker_slot=0,
            scheduler_kind="slurm",
            allocated_node="node01",
            work_directory=None,
            storage_root=storage,
            scratch_cleanup_status="not_created",
            failure_stage="input",
        )


def test_outer_worker_cleanup_retains_unpublished_case_scratch(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only inner case cleanup may remove a case child after durable publication."""
    storage = tmp_path / "storage"
    storage.mkdir()
    worker = tmp_path / "scratch" / "vp2-generation-42.test"
    worker.mkdir(parents=True)
    run_id = "campaign-v2~steady_flow~seen_lentil-n1__lhs-s9910"
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    workspace.initialize_worker_workspace(worker, run_id=run_id, storage_root=storage)
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("steady_flow"))
    child, root, _marker = workspace.create_case_workspace(config, case_id="case_0001", storage_root=storage, work_root=worker)
    assert root == worker
    (child / "solver.log").write_text("unpublished evidence\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preserving unpublished evidence"):
        workspace.cleanup_worker_workspace(worker, run_id=run_id, storage_root=storage, allow_active_job_id="42")
    assert (child / "solver.log").read_text(encoding="utf-8") == "unpublished evidence\n"
    assert (worker / workspace.WORKER_WORKSPACE_MARKER).is_file()

    assert (
        workspace.cleanup_case_workspace(
            child,
            allowed_root=worker,
            storage_root=storage,
            expected_run_id=run_id,
            expected_case_id="case_0001",
            allow_active_job_id="42",
        )
        > 0
    )
    assert workspace.cleanup_worker_workspace(worker, run_id=run_id, storage_root=storage, allow_active_job_id="42") > 0
    assert not worker.exists()


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


def test_resolved_batch_science_preserves_campaign_execution_provenance(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Bind batch science while the Source plan retains full execution choices."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    config = campaign.batch(_natural_batch_name("transient_drying"))
    root = lifecycle.create_campaign(campaign, storage_root=tmp_path / "storage")
    scientific_path = generation.runtime.initialize_batch_metadata(config, storage_root=tmp_path / "storage")
    assert scientific_path.name == "resolved_generation_config.json"
    scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    serialized = json.dumps(scientific, sort_keys=True)
    assert all(term not in serialized for term in ("max_nodes", "cores_per_case", "partition", "timeout_seconds", "wall_time", "cpu_host"))
    plan = json.loads(common.paths.resolve_generation_campaign_plan_path(root).read_text(encoding="utf-8"))
    assert plan["execution_values"] == config.execution_values
    assert not (scientific_path.parent / "execution_configs").exists()
    assert generation.runtime.initialize_batch_metadata(config, storage_root=tmp_path / "storage") == scientific_path
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
        "-batchlog",
        "-batchlog=other.log",
        "-batchlogout",
        "-batchlogout=true",
        "-BATCHLOG=other.log",
        "-usebatchlic",
        "-ckl",
        "-keeplicenses",
        "-USEBATCHLIC",
    ],
)
def test_extra_arguments_cannot_override_runtime_owned_comsol_flags(
    generation_config_factory: Any,
    argument: str,
) -> None:
    """Keep every command, parameter, batch-log, and licence flag runtime-owned."""
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
    assert local[1:7] == ["-autosave", "off", "-recoverydir", "comsol/recovery", "-tmpdir", "comsol/tmp"]
    assert local[7] == "batch"
    parameter_start = local.index("-pname")
    assert local[parameter_start : parameter_start + 6 : 2] == ["-pname", "-plist", "-pindex"]
    assert local[parameter_start + 1].split(",") == [*(entry.name for entry in entries), "Nx", "Ny", "Lx", "Ly", "Lz", "dt_out", "t_max"]
    expected_values = [scalar_handoff_contract.format_comsol_parameter(entry) for entry in entries]
    assert local[parameter_start + 3].split(",") == [*expected_values, "17", "11", "1.2[m]", "0.75[m]", "0.80000000000000004[m]", "1[h]", "168[h]"]
    assert "[1]" not in local[parameter_start + 3]
    assert local[parameter_start + 5] == ",".join(str(index) for index in range(1, 20))
    assert local[parameter_start + 6 :] == [
        "-np",
        "2",
        "-batchlog",
        "comsol_batch.log",
        "-batchlogout",
        "-usebatchlic",
        "-recover",
    ]

    slurm = comsol_service.build_comsol_command(
        config,
        cores_per_case=2,
        scalar_handoff=admission,
        scheduler_kind="slurm",
    )
    assert slurm == local
    assert local.count("-usebatchlic") == 1
    assert slurm.count("-usebatchlic") == 1
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
        "-autosave",
        "off",
        "-recoverydir",
        "comsol/recovery",
        "-tmpdir",
        "comsol/tmp",
        "batch",
        "-inputfile",
        "model.mph",
        "-job",
        "b1",
        "-outputfile",
        "solved.mph",
        "-np",
        "16",
        "-batchlog",
        "comsol_batch.log",
        "-batchlogout",
        "-usebatchlic",
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


def test_comsol_recovery_and_temporary_files_stay_inside_the_case_workspace(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Create the case-local COMSOL directories and name them before the batch target."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol)
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("steady_flow"))
    storage = tmp_path / "storage"
    _prepare_canonical_inputs(config, storage)
    prepared = runtime_service.prepare_case_work_directory(config, 1, storage_root=storage, work_root=tmp_path / "work")
    assert prepared.comsol_directory == prepared.work_directory / "comsol"
    assert (prepared.comsol_directory / "recovery").is_dir()
    assert (prepared.comsol_directory / "tmp").is_dir()
    command = comsol_service.build_comsol_command(config, cores_per_case=1, scratch_directory=str(prepared.comsol_directory))
    assert command[1:8] == [
        "-autosave",
        "off",
        "-recoverydir",
        str(prepared.comsol_directory / "recovery"),
        "-tmpdir",
        str(prepared.comsol_directory / "tmp"),
        "batch",
    ]
    workspace.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=prepared.work_root,
        storage_root=storage.resolve(),
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
    )


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


def test_supplied_raw_case_bundle_replaces_canonical_admission_and_stays_byte_verified(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """A caller-owned bundle is copied and re-verified without canonical storage, an input-generation identity, or another case."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("transient_drying"))
    storage = tmp_path / "supplied bundle storage"
    work = tmp_path / "supplied bundle work"
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "supplied-bundle")

    with pytest.raises(ValueError, match="mutually exclusive"):
        runtime_service.prepare_case_work_directory(
            config,
            1,
            storage_root=storage,
            work_root=work,
            input_generation_id="input-" + "a" * 24,
            raw_case_directory=bundle.directory,
        )
    prepared = runtime_service.prepare_case_work_directory(config, 1, storage_root=storage, work_root=work, raw_case_directory=bundle.directory)
    assert prepared.canonical_raw_directory == bundle.directory.resolve()
    assert prepared.input_generation_id == generation.cases.input_generation.configured_input_generation_id(config)
    assert prepared.bundle.simulation_case_id == bundle.simulation_case_id
    assert {path.name for path in prepared.bundle.input_paths} == {path.name for path in bundle.input_paths}
    assert all((prepared.work_directory / path.name).read_bytes() == path.read_bytes() for path in bundle.input_paths)
    assert not tuple(storage.rglob("input_generation_manifest.json"))

    with pytest.raises(generation.runtime.CasePreparationError, match="belongs to"):
        runtime_service.prepare_case_work_directory(config, 2, storage_root=storage, work_root=work, raw_case_directory=bundle.directory)
    victim = bundle.input_paths[0]
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(generation.runtime.CasePreparationError, match="Raw case input changed"):
        runtime_service.prepare_case_work_directory(config, 1, storage_root=storage, work_root=work, raw_case_directory=bundle.directory)


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


def test_transient_smoke_discretization_reaches_comsol_as_config_owned_parameters(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """A smaller grid and the fully supported time plan control the template."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        grid_overrides={"nx": 33, "ny": 21},
    )
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("transient_drying"))
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "smoke-bundle")
    assert bundle.scalar_handoff is not None
    command = comsol_service.build_comsol_command(config, cores_per_case=2, scalar_handoff=bundle.scalar_handoff)
    names = command[command.index("-pname") + 1].split(",")
    values = command[command.index("-plist") + 1].split(",")
    assert names[:12] == list(profile_contract.TRANSIENT_SCALAR_INPUT_FIELDS)
    assert dict(zip(names[12:], values[12:], strict=True)) == {
        "Nx": "33",
        "Ny": "21",
        "Lx": "1.2[m]",
        "Ly": "0.75[m]",
        "Lz": "0.80000000000000004[m]",
        "dt_out": "1[h]",
        "t_max": "168[h]",
    }


def test_solver_provenance_tracks_current_source_while_reusing_older_canonical_input(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New solver execution keeps original input bytes and its distinct source commit."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=1)
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("steady_flow"))
    storage = tmp_path / "storage"
    input_commit, solver_commit = "a" * 40, "b" * 40
    monkeypatch.setenv("GENERATION_GIT_COMMIT", input_commit)
    _prepare_canonical_inputs(config, storage)
    reference = generation.cases.input_generation.admit_configured_input_references(config, storage_root=storage, git_commit=input_commit)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", solver_commit)
    outcome = generation.runtime.run_case(
        config, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work", input_reference=reference[1]
    )
    assert outcome.status == "completed"
    directory = outcome.processed_directory
    publication = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "evidence" / "execution_provenance.json").read_text(encoding="utf-8"))
    processing = json.loads((directory / "evidence" / "processing_provenance.json").read_text(encoding="utf-8"))
    timing = json.loads((directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert publication["git_commit"] == input_commit
    assert execution["git_commit"] == processing["solver_git_commit"] == timing["git_commit"] == solver_commit
    assert processing["processing_git_commit"] == solver_commit
    generation.runtime.finalize_batch(config, storage_root=storage, input_references=reference)
    assert generation.runtime.validate_terminal_batch(config, storage_root=storage, input_references=reference)["git_commit"] == input_commit


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

    _bind_campaign_run(monkeypatch, _current_run_id(config))

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
    attempt = generation.publication.attempt.latest_case_attempt(config, 1, storage_root=storage)
    assert attempt is not None
    assert attempt.payload["case_state"] == "failed"
    assert attempt.payload["failure_stage"] == "input"
    assert "synthetic preparation failure" in attempt.payload["reason"]
    cleanup = json.loads((attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "not_created"


@pytest.mark.integration
def test_attempt_history_is_append_only_across_failure_and_success(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve earlier attempts after a newer failure or successful retry."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    source_run_id = _current_run_id(config)
    current_run_id = source_run_id
    commit = "a" * 40

    failed_storage = tmp_path / "append-only failure storage"
    source_path = _record_synthetic_failure(
        config,
        failed_storage,
        monkeypatch,
        git_commit=commit,
        execution_run_id=source_run_id,
    )
    source_bytes = source_path.read_bytes()
    _bind_campaign_run(monkeypatch, current_run_id)
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
    assert source_path.read_bytes() == source_bytes
    current_attempt = generation.publication.attempt.latest_case_attempt(config, 1, storage_root=failed_storage)
    assert current_attempt is not None
    assert current_attempt.payload["case_state"] == "failed"

    success_storage = tmp_path / "append-only recovery storage"
    success_path = _record_synthetic_failure(
        config,
        success_storage,
        monkeypatch,
        git_commit=commit,
        execution_run_id=source_run_id,
    )
    success_bytes = success_path.read_bytes()
    _bind_campaign_run(monkeypatch, current_run_id)
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
    assert generation.runtime.case_failure_is_recorded(
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
    config_path, _template = generation_config_factory(retain_raw_csv=True, retain_solved_model=True, executable=fake_comsol, timeout=1.0)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    _prepare_canonical_inputs(config, storage)
    _prepare_canonical_inputs(config, tmp_path / "timeout-storage")
    _bind_campaign_run(monkeypatch, _current_run_id(config))

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
    steps = (failed_attempt.directory / "payload/runtime/solver_steps.csv").read_text(encoding="utf-8").splitlines()
    assert steps[1] == "1,1.6355,0.42589,18,6,18,1,0,0,4.8e-16,4.6e-16"
    cleanup = json.loads((failed_attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "complete"

    monkeypatch.setenv("FAKE_COMSOL_MODE", "timeout")
    assert failed.value.runtime_reason == "solver_error"
    with pytest.raises(generation.runtime.CaseExecutionError) as timed_out:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=tmp_path / "timeout-storage",
            work_root=tmp_path / "timeout-work",
        )
    assert not timed_out.value.work_directory.exists()
    assert timed_out.value.runtime_reason == "timed_out"
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
        allow_active_job_id="4242",
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
@pytest.mark.parametrize(
    ("mode", "launcher_status", "expected_reason"),
    [("external_sigterm", None, None), ("failure", 137, None), ("failure", 255, "solver_error")],
)
def test_signal_exit_is_unclassified_and_other_statuses_stay_solver_error(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    launcher_status: int | None,
    expected_reason: str | None,
) -> None:
    """Leave a signal exit unclassified so the campaign never replaces a memory-limit kill automatically; 255 is no signal."""
    executable = fake_comsol
    expected_status = -15 if launcher_status is None else launcher_status
    if launcher_status is not None:  # a launcher shell reports a signal death as 128 + signal
        executable = tmp_path / "launcher.sh"
        script = f'#!/bin/sh\n{shlex.quote(str(fake_comsol))} "$@"\nstatus=$?\ntest "$status" -eq 7 && exit {launcher_status}\nexit "$status"\n'
        executable.write_text(script, encoding="utf-8")
        executable.chmod(0o755)
    config_path, _template = generation_config_factory(retain_raw_csv=True, retain_solved_model=True, executable=executable)
    config = generation.cases.config.load_generation_config(config_path, only_batch=_natural_batch_name("transient_drying"))
    storage = tmp_path / "storage"
    _prepare_canonical_inputs(config, storage)
    _bind_campaign_run(monkeypatch, _current_run_id(config))
    monkeypatch.setenv("FAKE_COMSOL_MODE", mode)

    with pytest.raises(generation.runtime.CaseExecutionError) as failed:
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work")

    assert failed.value.exit_code == expected_status
    assert failed.value.runtime_reason == expected_reason
    assert _latest_attempt(config, storage).payload["case_state"] == "failed"


@pytest.mark.integration
def test_transient_case_publishes_structural_solver_timing(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Persist transient scientific timing without retaining its complete batch log."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "transient timing storage"
    _prepare_canonical_inputs(config, storage)

    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=8,
        storage_root=storage,
        work_root=tmp_path / "transient timing work",
    )

    assert outcome.status == "completed"
    timing = json.loads((outcome.processed_directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert timing["comsol_stationary_airflow_seconds"] == 3.5
    assert timing["comsol_transient_drying_seconds"] == 7.25
    assert timing["comsol_scientific_solver_seconds"] == 10.75
    assert timing["comsol_solver_timing"]["status"] == "complete"
    assert timing["comsol_solver_timing"]["source_kind"] == "comsol_batch_log"
    assert timing["comsol_process_seconds"] == timing["runtime_s"]
    stall_rule = config.execution_values["runtime"]["stall_stop"]
    assert timing["stall_progress"] == {"minimum_simulated_seconds": float(stall_rule["minimum_simulated_seconds"]), "reached_after_seconds": None}
    assert not (outcome.processed_directory / "evidence" / "comsol_batch.log").exists()


@pytest.mark.integration
def test_successful_transient_case_keeps_missing_timing_nonfatal(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep scientific success when the finalized log lacks a required phase timing."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    storage = tmp_path / "missing transient timing storage"
    _prepare_canonical_inputs(config, storage)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "success_missing_transient_timing")

    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "missing transient timing work",
    )

    assert outcome.status == "completed"
    timing = json.loads((outcome.processed_directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert timing["comsol_stationary_airflow_seconds"] == 3.5
    assert timing["comsol_transient_drying_seconds"] is None
    assert timing["comsol_scientific_solver_seconds"] is None
    assert timing["comsol_solver_timing"]["status"] == "missing"
    assert generation.runtime.completed_case_is_valid(config, 1, storage_root=storage)


@pytest.mark.integration
def test_compact_case_uses_solved_output_but_omits_model_from_publication(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    evidence = outcome.processed_directory / "evidence"
    assert not (evidence / "solved.mph").exists()
    assert not (evidence / "comsol_exports").exists()
    assert not tuple(outcome.processed_directory.rglob("*.csv"))
    assert {path.name for path in outcome.processed_directory.iterdir()} == {"_SUCCESS", "case.h5", "evidence", "provenance.json"}
    assert {path.name for path in evidence.iterdir()} == {
        "execution_provenance.json",
        "processing_provenance.json",
        "solver.log",
        "status.json",
        "timing.json",
    }
    execution = json.loads((outcome.processed_directory / "evidence" / "execution_provenance.json").read_text(encoding="utf-8"))
    arguments = execution["invocation"]["arguments"]
    scratch = Path(execution["invocation"]["working_directory"]) / "comsol"
    assert arguments[:16] == [
        str(fake_comsol),
        "-autosave",
        "off",
        "-recoverydir",
        str(scratch / "recovery"),
        "-tmpdir",
        str(scratch / "tmp"),
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
    assert arguments.count("-batchlog") == 1
    assert arguments.count("-batchlogout") == 1
    assert arguments.count("-usebatchlic") == 1
    assert arguments[arguments.index("-batchlog") + 1].endswith("/runtime/comsol_batch.log")
    assert execution["result"]["solved_model"]["canonical_relative_path"] == "solved.mph"
    case_hdf5 = outcome.processed_directory / "case.h5"
    assert case_hdf5.is_file()
    with h5py.File(case_hdf5, "r") as handle:
        source_export_dataset = handle.get("provenance/source_exports_json")
        assert isinstance(source_export_dataset, h5py.Dataset)
        raw_source_exports: Any = source_export_dataset[()]
    source_export_text = raw_source_exports.decode("utf-8") if isinstance(raw_source_exports, bytes) else str(raw_source_exports)
    source_exports = json.loads(source_export_text)
    assert source_exports
    for configured_name, evidence in source_exports.items():
        assert evidence == {
            "logical_role": evidence["logical_role"],
            "configured_relative_name": configured_name,
            "sha256": evidence["sha256"],
            "size_bytes": evidence["size_bytes"],
            "row_count": evidence["row_count"],
            "column_count": evidence["column_count"],
            "conversion_status": "converted",
            "retained": False,
        }
        assert evidence["size_bytes"] > 0
        assert evidence["row_count"] > 0
        assert evidence["column_count"] > 0
    timing = json.loads((outcome.processed_directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert timing["comsol_stationary_airflow_seconds"] == 3.5
    assert timing["comsol_transient_drying_seconds"] is None
    assert timing["comsol_scientific_solver_seconds"] == 3.5
    assert timing["comsol_solver_timing"]["status"] == "complete"
    assert timing["comsol_process_seconds"] == timing["runtime_s"]
    assert timing["source_export_bytes"] > 0
    assert timing["solved_model_scratch_bytes"] > 0
    assert timing["case_hdf5_bytes"] == case_hdf5.stat().st_size
    assert timing["persistent_case_bytes"] > 0
    assert timing["direct_exports_retained_bytes"] == 0
    assert timing["solved_model_retained_bytes"] == 0
    assert timing["scratch_peak_bytes"] >= timing["source_export_bytes"]
    processing = json.loads((outcome.processed_directory / "evidence" / "processing_provenance.json").read_text(encoding="utf-8"))
    assert processing["mode"] == "initial"
    assert processing["solver_git_commit"] == processing["processing_git_commit"]

    hashed_paths: list[Path] = []
    original_file_sha256 = common.serialization.file_sha256

    def record_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_file_sha256(path)

    monkeypatch.setattr(common.serialization, "file_sha256", record_hash)
    generation.runtime.validate_completed_case(
        config,
        1,
        storage_root=storage,
        validation_depth="routine",
    )
    assert case_hdf5 not in hashed_paths
    hashed_paths.clear()
    generation.runtime.validate_completed_case(
        config,
        1,
        storage_root=storage,
        validation_depth="deep",
    )
    assert case_hdf5 in hashed_paths


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
    solved_model = outcome.processed_directory / "evidence" / "solved.mph"
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
        / "fields.csv"
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
    _bind_campaign_run(monkeypatch, _current_run_id(config))
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
        retain_raw_csv=True,
        retain_solved_model=True,
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
    _bind_campaign_run(monkeypatch, _current_run_id(config))
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
    assert attempt.payload["retention_policy"] == "full"
    assert not (attempt.directory / "payload" / "solved.mph.status").exists()
    assert "payload/runtime/stop.json" in attempt.payload["retained_inventory"]
    stop = json.loads((attempt.directory / "payload/runtime/stop.json").read_text(encoding="utf-8"))
    assert stop["reason"] == "cancelled"
    cleanup = json.loads((attempt.directory / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )


def test_attempt_payload_records_dispatch_step_and_submission_index(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind each campaign attempt to its allocation, step, and submission index."""
    config, storage, run_id = _canonical_failed_attempt_config(generation_config_factory, tmp_path)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    _bind_campaign_run(monkeypatch, run_id, allocation_id="777", submission_index=12, step_id="5")
    receipt = runtime_service.record_case_failure(
        config,
        1,
        RuntimeError("campaign failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    attempt = generation.publication.attempt.load_attempt(receipt.parent)
    assert attempt.payload["campaign_run_id"] == run_id
    assert (attempt.payload["job_id"], attempt.payload["slurm_step_id"], attempt.payload["submission_index"]) == ("777", "5", 12)


def test_attempt_dispatch_identity_is_null_outside_campaigns_and_validated(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record no dispatch identity for a non-campaign attempt and reject inconsistent persisted identities."""
    config, storage, _run_id = _canonical_failed_attempt_config(generation_config_factory, tmp_path)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID", raising=False)
    receipt = runtime_service.record_case_failure(
        config,
        1,
        RuntimeError("local failure"),
        worker_slot=0,
        scheduler_kind="local",
        allocated_node=None,
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    attempt = generation.publication.attempt.load_attempt(receipt.parent)
    assert attempt.payload["slurm_step_id"] is None
    assert attempt.payload["submission_index"] is None

    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    for step_id, submission_index in (("0", None), (None, 3), ("x", 3), ("0", 0), ("0", True)):
        receipt.write_text(json.dumps({**persisted, "slurm_step_id": step_id, "submission_index": submission_index}), encoding="utf-8")
        with pytest.raises(ValueError, match="dispatch identity is malformed"):
            generation.publication.attempt.load_attempt(receipt.parent)


def _dispatch_environment(**overrides: str | None) -> dict[str, str]:
    """Return one complete campaign case-step environment with optional replaced or removed entries."""
    environment: dict[str, str | None] = {
        "GENERATION_CAMPAIGN_RUN_ID": "campaign-run",
        "SLURM_JOB_ID": "5001",
        "SLURM_STEP_ID": "3",
        "GENERATION_SUBMISSION_INDEX": "17",
        "GENERATION_DISPATCH_KEY": "5001-17",
        **overrides,
    }
    return {name: value for name, value in environment.items() if value is not None}


def test_dispatch_key_round_trips_and_rejects_malformed_values() -> None:
    """Keep one canonical key syntax for building and parsing."""
    assert dispatch_identity.dispatch_key("5001", 17) == "5001-17"
    assert dispatch_identity.parse_dispatch_key("5001-17") == ("5001", 17)
    assert dispatch_identity.DISPATCH_KEY_PATTERN.fullmatch("5001-17") is not None
    for malformed in ("5001", "5001-0", "5001-017", "-3", "a-3", "5001-3 ", "5001-3-4", "5001_3", "", "5001.0-3"):
        assert dispatch_identity.DISPATCH_KEY_PATTERN.fullmatch(malformed) is None
        with pytest.raises(ValueError, match="Dispatch key"):
            dispatch_identity.parse_dispatch_key(malformed)
    with pytest.raises(ValueError, match="numeric Slurm job id"):
        dispatch_identity.dispatch_key("12_3", 1)
    with pytest.raises(ValueError, match="positive"):
        dispatch_identity.dispatch_key("12", 0)
    with pytest.raises(TypeError, match="integer"):
        dispatch_identity.dispatch_key("12", True)
    with pytest.raises(ValueError, match="does not match"):
        dispatch_identity.CaseDispatch(allocation_id="12", step_id="0", submission_index=2, dispatch_key="12-3")


def test_current_case_dispatch_resolves_a_complete_campaign_step() -> None:
    """Resolve allocation, step, submission index, and key of the running case step."""
    assert dispatch_identity.current_case_dispatch(_dispatch_environment()) == dispatch_identity.CaseDispatch(
        allocation_id="5001",
        step_id="3",
        submission_index=17,
        dispatch_key="5001-17",
    )


def test_current_case_dispatch_is_none_outside_campaign_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invent no identity for a Slurm step without a campaign run, such as a benchmark step."""
    assert dispatch_identity.current_case_dispatch(_dispatch_environment(GENERATION_CAMPAIGN_RUN_ID=None)) is None
    assert dispatch_identity.current_case_dispatch({"SLURM_JOB_ID": "5001", "SLURM_STEP_ID": "3"}) is None
    assert dispatch_identity.current_case_dispatch({}) is None
    monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID", raising=False)
    assert dispatch_identity.current_case_dispatch() is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SLURM_STEP_ID": None}, "SLURM_STEP_ID"),
        ({"SLURM_STEP_ID": "batch"}, "SLURM_STEP_ID"),
        ({"SLURM_JOB_ID": None}, "SLURM_JOB_ID"),
        ({"SLURM_JOB_ID": "50x1"}, "SLURM_JOB_ID"),
        ({"GENERATION_SUBMISSION_INDEX": None}, "GENERATION_SUBMISSION_INDEX"),
        ({"GENERATION_SUBMISSION_INDEX": "0"}, "GENERATION_SUBMISSION_INDEX"),
        ({"GENERATION_SUBMISSION_INDEX": "017"}, "GENERATION_SUBMISSION_INDEX"),
        ({"GENERATION_DISPATCH_KEY": None}, "does not match"),
        ({"GENERATION_DISPATCH_KEY": "5001-18"}, "does not match"),
        ({"GENERATION_DISPATCH_KEY": "5002-17"}, "does not match"),
    ],
)
def test_current_case_dispatch_fails_closed_on_an_incomplete_or_contradictory_campaign_step(overrides: dict[str, str | None], message: str) -> None:
    """Refuse a campaign context whose step identity is missing, malformed, or inconsistent."""
    with pytest.raises(RuntimeError, match=message):
        dispatch_identity.current_case_dispatch(_dispatch_environment(**overrides))
