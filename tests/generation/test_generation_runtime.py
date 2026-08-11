# ruff: noqa: S101, PLR2004
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


@pytest.mark.parametrize(
    "argument",
    [
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
        runtime_service.build_comsol_command(config, cores_per_case=2)
    entries = admission.entries
    assert len(entries) == 12
    assert tuple(entry.name for entry in entries) == profile_contract.TRANSIENT_SCALAR_INPUT_FIELDS
    assert all(entry.owner == "case_dependent" for entry in entries)
    assert {"T_init", "T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}.isdisjoint(entry.name for entry in entries)

    local = runtime_service.build_comsol_command(
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

    slurm = runtime_service.build_comsol_command(
        config,
        cores_per_case=2,
        scalar_handoff=admission,
        scheduler_kind="slurm",
        node_hostname="node-a",
    )
    assert slurm[:7] == [
        "srun",
        "--exclusive",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--cpu-bind=cores",
        "--nodelist=node-a",
    ]
    assert slurm[7:] == local

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
    prepared = runtime_service.prepare_case_work_directory(
        config,
        1,
        storage_root=tmp_path / "scalar validation storage",
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
    command = runtime_service.build_comsol_command(config, cores_per_case=1)
    assert not {"-pname", "-plist", "-pindex"}.intersection(command)


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
    outcome = generation.runtime.run_case(
        batch,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "pilot terminal work",
    )
    assert outcome.work_directory is not None
    execution = json.loads((outcome.processed_directory / "execution_provenance.json").read_text(encoding="utf-8"))
    case_payload = json.loads((outcome.processed_directory / "case.json").read_text(encoding="utf-8"))
    scalar_binding = execution["scalar_handoff"]
    runtime_entries = [entry for entry in case_payload["scalar_handoff"]["entries"] if entry["owner"] == "case_dependent"]
    assert scalar_binding["state"] == "applied"
    assert scalar_binding["mechanism"] == "comsol_cli_pname_plist"
    assert scalar_binding["entries"] == case_payload["scalar_handoff"]["entries"]
    assert scalar_binding["contract_sha256"] == scalar_handoff_contract.TRANSIENT_SCALAR_HANDOFF_CONTRACT_SHA256
    assert scalar_binding["source"]["filename"] == "scalars.csv"
    assert scalar_binding["source"]["sha256"] == case_payload["input_files"]["scalars.csv"]["sha256"]
    assert scalar_binding["source"]["size_bytes"] == case_payload["input_files"]["scalars.csv"]["size_bytes"]
    assert scalar_binding["source"]["path"] == str(outcome.work_directory / "scalars.csv")
    assert scalar_binding["runtime_override_names"] == [entry["name"] for entry in runtime_entries]
    assert scalar_binding["runtime_override_values"] == [entry["value"] for entry in runtime_entries]
    arguments = execution["invocation"]["arguments"]
    assert scalar_binding["formatted_plist_expressions"] == arguments[arguments.index("-plist") + 1].split(",")
    assert scalar_binding["pindex_values"] == list(range(1, 13))
    assert scalar_binding["original_comsol_output_filename"] == "solved.mph"
    assert scalar_binding["canonical_solved_model_filename"] == "solved.mph"
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


def test_solver_receives_relative_files_and_canonicalizes_suffixed_output(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect relative invocation plus atomic admission of one suffixed output."""
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
    timing = json.loads((outcome.processed_directory / "timing.json").read_text(encoding="utf-8"))
    execution = json.loads((outcome.processed_directory / "execution_provenance.json").read_text(encoding="utf-8"))
    assert timing["working_directory"] == str(outcome.work_directory)
    assert execution["invocation"]["working_directory"] == str(outcome.work_directory)
    assert execution["schema_version"] == 1
    assert execution["result"]["state"] == "succeeded"
    assert execution["result"]["exit_code"] == 0
    solved_model = outcome.processed_directory / "solved.mph"
    assert execution["scalar_handoff"] == {
        "state": "not_applicable",
        "mechanism": "parameter_free",
        "reason": "steady_flow_has_no_transient_scalar_runtime_overrides",
        "original_comsol_output_filename": "solved_1.mph",
        "canonical_solved_model_filename": "solved.mph",
    }
    solved_evidence = execution["result"]["solved_model"]
    assert solved_evidence == {
        "requested_relative_path": "solved.mph",
        "observed_relative_path": "solved_1.mph",
        "canonical_relative_path": "solved.mph",
        "disposition": "new",
        "canonicalized": True,
        "size_bytes": solved_model.stat().st_size,
        "sha256": common.serialization.file_sha256(solved_model),
    }
    publication = json.loads((outcome.processed_directory / "provenance.json").read_text(encoding="utf-8"))
    assert publication["artifacts"]["solved.mph"] == {
        "sha256": solved_evidence["sha256"],
        "size_bytes": solved_evidence["size_bytes"],
    }
    arguments = execution["invocation"]["arguments"]
    assert arguments[arguments.index("-inputfile") + 1] == "model.mph"
    assert arguments[arguments.index("-outputfile") + 1] == "solved.mph"
    case_payload = json.loads((outcome.processed_directory / "case.json").read_text(encoding="utf-8"))
    assert set(case_payload["input_files"]) == {"fields.csv"}
    assert "scalar_handoff" not in case_payload


def test_solved_model_inventory_accepts_one_replaced_exact_candidate(tmp_path: Path) -> None:
    """Distinguish a solver replacement from an unchanged stale exact output."""
    work_directory = tmp_path / "replaced solved model"
    work_directory.mkdir()
    model = work_directory / "model.mph"
    model.write_bytes(b"template model must never be selected\n")
    solved_model = work_directory / "solved.mph"
    solved_model.write_bytes(b"stale model\n")
    before = runtime_service._solved_model_inventory(work_directory)  # noqa: SLF001
    assert set(before) == {"solved.mph"}
    solved_model.write_bytes(b"replacement model\n")

    canonical, evidence = runtime_service._canonicalize_solved_model(  # noqa: SLF001
        work_directory,
        before,
    )

    assert canonical == solved_model
    assert canonical.read_bytes() == b"replacement model\n"
    assert model.read_bytes() == b"template model must never be selected\n"
    assert evidence["observed_relative_path"] == "solved.mph"
    assert evidence["disposition"] == "replaced"
    assert evidence["canonicalized"] is False


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
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name("steady_flow"),
    )
    storage_root = tmp_path / "invalid solved storage"
    work_root = tmp_path / "invalid solved work"
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
