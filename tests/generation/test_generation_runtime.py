# ruff: noqa: S101, PLR2004
"""Canonical conversion, provenance separation, failure, and locking contracts."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src import common, generation
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_workspace as workspace

if TYPE_CHECKING:
    from pathlib import Path


def _natural_batch_name(simulation_profile: str) -> str:
    """Return one canonical synthetic natural-batch selector."""
    return generation.cases.config.build_batch_name(
        simulation_profile,
        "lentil",
        "natural",
    )


def test_float32_conversion_requires_explicit_tolerance() -> None:
    """Protect validated conversion rather than silent precision loss."""
    values = np.asarray([1.0, 1.0e-9, 123.456789], dtype=np.float64)
    converted = generation.publication.storage.validate_float32_conversion(values, rtol=1e-6, atol=1e-12, label="synthetic")
    assert converted.dtype == np.float32
    with pytest.raises(ValueError, match="exceeds configured tolerance"):
        generation.publication.storage.validate_float32_conversion(values, rtol=0.0, atol=0.0, label="synthetic")


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
    assert common.serialization.canonical_json_sha256(scientific) == config.scientific_config_digest


def test_terminal_case_identity_binds_persisted_input_configuration(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject case identities recomputed around an arbitrary input-config digest."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "case identity")
    payload = bundle.case_payload
    manifest = {
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "git_commit": "a" * 40,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
    }
    record = {
        "case_id": bundle.case_id,
        "case_index": 1,
        "case_input_id": bundle.case_input_id,
        "simulation_case_id": bundle.simulation_case_id,
        "material_family": config.material_family,
    }
    runtime_service._require_case_matches_terminal(  # noqa: SLF001
        payload,
        manifest=manifest,
        scientific=config.scientific_values,
        record=record,
        directory=bundle.directory,
    )

    tampered = json.loads(json.dumps(payload))
    tampered["case_input_config_digest"] = "0" * 64
    assert tampered["case_input_config_digest"] != config.case_input_config_digest
    tampered["case_input_id"] = generation.cases.case.compute_case_input_id(tampered)
    tampered["simulation_case_id"] = generation.cases.case.compute_simulation_case_id(tampered)
    tampered_record = {
        **record,
        "case_input_id": tampered["case_input_id"],
        "simulation_case_id": tampered["simulation_case_id"],
    }
    with pytest.raises(RuntimeError, match="metadata disagrees"):
        runtime_service._require_case_matches_terminal(  # noqa: SLF001
            tampered,
            manifest=manifest,
            scientific=config.scientific_values,
            record=tampered_record,
            directory=bundle.directory,
        )


def test_pilot_terminal_admission_uses_canonical_semantic_batch_kind(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Protect pilot terminal identity when sampling_regime remains natural."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
    )
    original = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("transient_drying"),
    )
    scientific = json.loads(json.dumps(original.scientific_values))
    assignment = {
        "case_index": 1,
        "regime_index": 0,
        "material_family": "lentil",
        "material_role": "seen",
        "evaluation_regime": generation.cases.config.NO_EVALUATION_REGIME,
        "sampling_regime": "natural",
        "assignment_role": "nominal_reference",
        "ood_group": None,
        "ood_unit_id": None,
        "ood_units_per_case": 0,
        "pilot_case_kind": "nominal_reference",
    }
    scientific.update(
        {
            "campaign_purpose": generation.cases.config.PILOT_CAMPAIGN_PURPOSE,
            "evaluation_regime": generation.cases.config.NO_EVALUATION_REGIME,
            "case_count": 1,
            "assignments": [assignment],
            "pilot_check": {
                "cases_per_material": 1,
                "case_kinds": list(generation.cases.config.PILOT_CASE_KINDS),
                "case_semantics": {"first": "nominal_reference", "remaining": "natural_pilot"},
                "nominal_case_index": 1,
                "dataset_membership": "none",
                "sampling_semantics": "one_explicit_nominal_then_ordinary_natural_support",
            },
        }
    )
    scientific.pop("paired_equivalence_seed", None)
    scientific_digest = common.serialization.canonical_json_sha256(scientific)
    case_input_digest = generation.cases.config.compute_case_input_config_digest(scientific)
    batch_name = generation.cases.config.build_batch_name(
        original.profile.id,
        original.material_family,
        generation.cases.config.PILOT_CAMPAIGN_PURPOSE,
    )
    batch = replace(
        original,
        evaluation_regime=generation.cases.config.NO_EVALUATION_REGIME,
        batch_name=batch_name,
        scientific_values=scientific,
        case_indices=(1,),
        assignments={1: assignment},
        scientific_config_digest=scientific_digest,
        case_input_config_digest=case_input_digest,
        batch_identity=scientific_digest,
        batch_id=generation.cases.config.build_batch_id(batch_name, scientific_digest),
    )
    storage = tmp_path / "pilot terminal storage"
    generation.runtime.run_case(
        batch,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "pilot terminal work",
    )
    generation.runtime.finalize_batch(batch, storage_root=storage)
    admitted = generation.runtime.admit_terminal_batch(batch.batch_id, storage_root=storage)
    assert admitted.batch_name == batch_name
    assert admitted.sampling_regime == "natural"
    assert admitted.scientific_config_payload()["campaign_purpose"] == "pilot_check"


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
    failure = json.loads(generation.runtime.case_failure_path(config, 1, storage_root=storage).read_text(encoding="utf-8"))
    assert failure["error"]["type"] == "OSError"
    assert failure["work_directory"] is None


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

    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    with pytest.raises(generation.runtime.CaseExecutionError) as failed:
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=storage, work_root=work)
    assert not failed.value.work_directory.exists()
    failure_record = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert failure_record["execution"]["cwd"] == str(failed.value.work_directory)
    assert failure_record["execution"]["exit_code"] == 7
    assert failure_record["execution"]["command"]
    assert failure_record["input_files"]["declared"] == failure_record["input_files"]["observed"]
    assert failure_record["template_sha256"] == config.template_sha256
    assert failure_record["scratch_cleanup"]["status"] == "complete"
    assert failure_record["log_tail"]["source"] == "solver.log"

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

    lock_path = generation.runtime.case_lock_path(config, 1, storage_root=tmp_path / "locked")
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


def test_solver_receives_exact_isolated_cwd_and_relative_files(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect explicit cwd, relative model arguments, and cleanup ordering."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage = tmp_path / "storage with spaces"
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
    timing = json.loads((outcome.processed_directory / "timing.json").read_text(encoding="utf-8"))
    execution = json.loads((outcome.processed_directory / "execution_provenance.json").read_text(encoding="utf-8"))
    assert timing["working_directory"] == str(outcome.work_directory)
    assert execution["invocation"]["working_directory"] == str(outcome.work_directory)
    assert execution["result"]["state"] == "succeeded"
    assert execution["result"]["exit_code"] == 0
    arguments = execution["invocation"]["arguments"]
    assert arguments[arguments.index("-inputfile") + 1] == "model.mph"
    assert arguments[arguments.index("-outputfile") + 1] == "solved.mph"
    case_payload = json.loads((outcome.processed_directory / "case.json").read_text(encoding="utf-8"))
    assert set(case_payload["input_files"]) == {"fields.csv"}
    assert "scalar_handoff" not in case_payload


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
        / "raw_csv/inputs/fields.csv"
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
    original_cleanup = workspace.cleanup_case_workspace
    observed = {"publication_called": False, "failure_before_cleanup": False}

    def fail_publication(*_args: Any, **_kwargs: Any) -> None:
        observed["publication_called"] = True
        message = "synthetic publication validation failure"
        raise RuntimeError(message)

    def evidence_first_cleanup(*args: Any, **kwargs: Any) -> int:
        failure_path = generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        )
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        observed["failure_before_cleanup"] = failure["scratch_cleanup"]["status"] == "pending"
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
    failure = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert failure["scratch_cleanup"]["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )


def test_runtime_cancellation_terminates_solver_and_persists_cancelled_case(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect cooperative TERM propagation, evidence, cleanup, and rerun state."""
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
    failure = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert failure["state"] == "cancelled"
    assert failure["execution"]["exit_code"] is not None
    assert failure["scratch_cleanup"]["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
