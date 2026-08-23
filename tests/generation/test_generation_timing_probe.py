# ruff: noqa: D103, PLR2004, S101, SLF001
"""Focused contracts for one normal-path COMSOL timing probe."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.generation import generation_timing_probe as probe
from src.generation.cli import cli_generation
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_workspace as workspace_service


def test_parser_retains_candidates_without_claiming_real_comsol_grammar() -> None:
    data = b"Stationary solver\nSolution time: 2 min\nTransient solver\nElapsed time: 3.5e1 s\n"
    records = probe.parse_solution_times(data)
    summary = probe.summarize_solution_times(records)

    assert [record["timing_expression"] for record in records] == ["solution_time", "elapsed_time"]
    assert [record["converted_seconds"] for record in records] == [120.0, 35.0]
    assert [record["line_number"] for record in records] == [2, 4]
    assert [record["byte_offset"] for record in records] == [
        len(b"Stationary solver\n"),
        len(data) - len(b"Elapsed time: 3.5e1 s\n"),
    ]
    assert all(len(record["context"].encode()) <= probe._MAX_CONTEXT_BYTES for record in records)
    assert summary["phases"]["stationary"]["status"] == "single_candidate"
    assert summary["phases"]["transient"]["status"] == "single_candidate"
    assert summary["diagnostic_candidate_sum_seconds"] == 155.0
    assert summary["method_status"] == "candidate_evidence_available"
    assert summary["real_comsol_grammar_validated"] is False
    assert summary["diagnostic_only"] is True
    assert "comsol_scientific_solver_seconds" not in summary


def test_parser_reports_computation_duplicate_nested_and_malformed_candidates() -> None:
    data = (
        b"Stationary solver\n"
        b"  Computation time: 1 s\n"
        b"Computation time: 1 s\n"
        b"Computation time: 1 s\n"
        b"Transient solver\n"
        b"Elapsed time: bananas\n"
        b"Solution time: 3 fortnights\n"
    )
    records = probe.parse_solution_times(data)
    stationary = [record for record in records if record["detected_phase"] == "stationary"]
    transient = [record for record in records if record["detected_phase"] == "transient"]
    summary = probe.summarize_solution_times(records)

    assert {record["timing_expression"] for record in stationary} == {"computation_time"}
    assert stationary[0]["classification"] == "nested"
    assert stationary[1]["duplicate_classification"] == "duplicate"
    assert stationary[2]["duplicate_classification"] == "duplicate"
    assert [record["parse_status"] for record in transient] == ["malformed", "unsupported_format"]
    assert summary["phases"]["stationary"]["status"] == "ambiguous"
    assert summary["phases"]["transient"]["status"] == "ambiguous"
    assert summary["diagnostic_candidate_sum_seconds"] is None


def test_incremental_reader_persists_offsets_partial_lines_and_resets(tmp_path: Path) -> None:
    path = tmp_path / "comsol_batch.log"
    path.write_bytes(b"Stationary\nSolution time: 1 s")
    state, first = probe.observe_appended_bytes(
        path,
        probe.ProbeObservationState(),
        observed_monotonic_ns=1,
        observed_utc="2026-01-01T00:00:00+00:00",
    )
    restored = probe._state_from_payload(probe._state_payload(state))

    assert [event["event_kind"] for event in first] == ["stationary_phase_first_observed"]
    assert restored.partial == b"Solution time: 1 s"
    assert restored.offset == path.stat().st_size
    assert restored.partial_offset == len(b"Stationary\n")
    assert restored.next_line_number == 2

    path.write_bytes(path.read_bytes() + b"\nTransient\nElapsed time: 2 s\n")
    state, second = probe.observe_appended_bytes(
        path,
        restored,
        observed_monotonic_ns=2,
        observed_utc="2026-01-01T00:00:01+00:00",
    )
    assert [event["event_kind"] for event in second] == [
        "stationary_completion_observed",
        "transient_phase_first_observed",
        "transient_completion_observed",
    ]
    assert second[0]["byte_offset"] == len(b"Stationary\n")
    assert second[0]["line_number"] == 2
    assert not state.partial

    replacement_state = probe.ProbeObservationState(
        offset=state.offset,
        partial=state.partial,
        partial_offset=state.partial_offset,
        device=-1,
        inode=-1,
        next_line_number=state.next_line_number,
        phase=state.phase,
        generation=state.generation,
    )
    path.write_bytes(b"Transient\nComputation time: 4 s\n")
    _, replaced = probe.observe_appended_bytes(path, replacement_state, observed_monotonic_ns=3)
    assert replaced[0]["event_kind"] == "log_replaced"
    assert replaced[1]["line_number"] == 1
    assert replaced[-1]["byte_offset"] == len(b"Transient\n")

    truncated_state = probe.ProbeObservationState(
        offset=10_000,
        device=path.stat().st_dev,
        inode=path.stat().st_ino,
        next_line_number=20,
        phase="transient",
    )
    _, truncated = probe.observe_appended_bytes(path, truncated_state, observed_monotonic_ns=4)
    assert truncated[0]["event_kind"] == "log_truncated"


def _command_config(extra_arguments: list[str]) -> Any:
    return SimpleNamespace(
        profile=SimpleNamespace(id=profiles.STEADY_FLOW_PROFILE),
        execution_values={
            "retention_policy": "full",
            "runtime": {"executable": "/opt/comsol/bin/comsol", "extra_arguments": extra_arguments},
        },
    )


def test_probe_batchlog_is_single_runtime_owned_flag_pair() -> None:
    standard = comsol_service.build_comsol_command(_command_config([]), cores_per_case=4)
    command = comsol_service.build_comsol_command(
        _command_config([]),
        cores_per_case=4,
        diagnostic_batchlog="/attempt/runtime/comsol_batch.log",
    )

    assert "-batchlog" not in standard
    assert "-batchlogout" not in standard
    assert command.count("-batchlog") == 1
    assert command.count("-batchlogout") == 1
    assert command[command.index("-batchlog") + 1] == "/attempt/runtime/comsol_batch.log"
    assert command[command.index("-np") + 1] == "4"


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ["-batchlog", "/other/log"],
        ["-batchlog=/other/log"],
        ["-batchlogout"],
        ["-BATCHLOG=/other/log"],
    ],
)
def test_probe_batchlog_rejects_configured_conflicts(extra_arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="batch logging conflicts"):
        comsol_service.build_comsol_command(
            _command_config(extra_arguments),
            cores_per_case=1,
            diagnostic_batchlog="/attempt/runtime/comsol_batch.log",
        )


def test_observed_wall_and_method_verdicts_remain_diagnostic() -> None:
    events = [
        {"event_kind": "stationary_phase_first_observed", "observed_monotonic_ns": 10},
        {"event_kind": "stationary_completion_observed", "observed_monotonic_ns": 20},
        {"event_kind": "transient_phase_first_observed", "observed_monotonic_ns": 30},
        {"event_kind": "transient_completion_observed", "observed_monotonic_ns": 50},
    ]
    observed = probe._observed_wall(events)
    verdicts = probe._method_verdicts(
        {
            "method_status": "candidate_evidence_available",
            "diagnostic_only": True,
            "real_comsol_grammar_validated": False,
            "phases": {},
        },
        observed,
    )

    assert observed["stationary_airflow_observed_wall_seconds"] == pytest.approx(1e-8)
    assert observed["transient_drying_observed_wall_seconds"] == pytest.approx(2e-8)
    assert verdicts["same_process_method"]["method_status"] == "not_implementable_from_current_source_boundary"
    assert verdicts["same_process_method"]["exact_boundary_proven"] is False
    assert verdicts["recommendation"] == "unresolved_pending_real_probe_review"
    assert verdicts["real_comsol_grammar_validated"] is False
    assert verdicts["production_timing_fields_updated"] is False


def _bind_fake_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "synthetic-cpu-node")


def _successful_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    campaign_path, _ = generation_config_factory(
        simulation_profile=profiles.TRANSIENT_DRYING_PROFILE,
        executable=fake_comsol,
        natural_count=2,
        campaign_purpose="technical_runtime_smoke",
        scheduler_kind="slurm",
        license_retry_enabled=False,
    )
    _bind_fake_slurm(monkeypatch)
    storage = tmp_path / "cpu-storage"
    work = tmp_path / "normal-work"
    work.mkdir()
    (work / "user-sentinel.txt").write_text("preserve me\n", encoding="utf-8")
    tracker = tmp_path / "fake-comsol-tracker.json"
    monkeypatch.setenv("FAKE_COMSOL_TRACKER", str(tracker))
    result = probe.run_timing_probe(campaign_path, storage_root=storage, work_root=work)
    return result, campaign_path, storage, work


def test_probe_runs_exactly_one_case_through_normal_generation_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    generated_calls: list[dict[str, Any]] = []
    run_calls: list[dict[str, Any]] = []
    original_generate = probe.input_service.generate_input_cases
    original_run = probe.batch_service.run_case

    def tracked_generate(config: Any, case_count: int, **kwargs: Any) -> Any:
        generated_calls.append({"batch_id": config.batch_id, "case_count": case_count, **kwargs})
        return original_generate(config, case_count, **kwargs)

    def tracked_run(config: Any, case_index: int, **kwargs: Any) -> Any:
        run_calls.append({"batch_id": config.batch_id, "case_index": case_index, **kwargs})
        return original_run(config, case_index, **kwargs)

    monkeypatch.setattr(probe.input_service, "generate_input_cases", tracked_generate)
    monkeypatch.setattr(probe.batch_service, "run_case", tracked_run)
    result, campaign_path, storage, work = _successful_probe(
        tmp_path,
        monkeypatch,
        generation_config_factory,
        fake_comsol,
    )
    bundle = Path(result["probe_cpu_bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    exact_command = json.loads((bundle / "exact_command.json").read_text(encoding="utf-8"))
    candidates = json.loads((bundle / "batch_log_candidates.json").read_text(encoding="utf-8"))
    environment = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
    tracker = json.loads((tmp_path / "fake-comsol-tracker.json").read_text(encoding="utf-8"))
    plan = probe.resolve_timing_probe_plan(campaign_path)

    assert result["probe_case_state"] == "successful"
    assert result["probe_case_exit_code"] == 0
    assert len(generated_calls) == 1
    assert generated_calls[0]["case_count"] == 1
    assert generated_calls[0]["case_start"] == plan["case_index"]
    assert len(run_calls) == 1
    assert run_calls[0]["case_index"] == plan["case_index"]
    assert run_calls[0]["cores_per_case"] == plan["resources"]["cores_per_case"] == 1
    assert run_calls[0]["scheduler_kind"] == "slurm"
    assert run_calls[0]["storage_root"] != storage
    assert callable(run_calls[0]["diagnostic_observer"])
    assert tracker == {"active": 0, "maximum": 1, "starts": 1}
    assert manifest["normal_generation_path"]["requested_case_count"] == 1
    assert manifest["normal_generation_path"]["case_runner"] == "generation_runtime_batch.run_case"
    assert manifest["resources"]["cores_per_case"] == 1
    assert manifest["case_index"] == plan["case_index"]
    assert manifest["normal_case_evidence"]["status"] == "completed"
    assert manifest["normal_case_evidence"]["failure_attempt"] is None
    assert manifest["normal_case_evidence"]["license_wait"] is None
    assert manifest["normal_case_evidence"]["publication_files"]["_SUCCESS"]["payload"]["case_id"] == manifest["case_id"]
    assert set(manifest["canonical_input_files"]) == set(manifest["workspace_input_files"])
    assert "fields.csv" in manifest["workspace_input_files"]
    assert all(Path(value["workspace_path"]).name == name for name, value in manifest["workspace_input_files"].items())
    assert exact_command["argv"].count("-batchlog") == 1
    assert exact_command["argv"].count("-batchlogout") == 1
    assert exact_command["argv"][exact_command["argv"].index("-np") + 1] == "1"
    assert exact_command["working_directory"] == run_calls[0]["diagnostic_observer"].session["work_path"]
    assert {record["timing_expression"] for record in candidates["records"]} == {"solution_time", "elapsed_time"}
    assert candidates["summary"]["real_comsol_grammar_validated"] is False
    assert environment["comsol_version"]["status"] == "recorded"
    assert environment["slurm_job_id"] == "12345"
    assert probe.validate_probe_bundle(bundle)["valid"] is True
    assert {path.name for path in bundle.iterdir()} == probe._EXACT_BUNDLE_INVENTORY
    assert not any(path.suffix == ".h5" for path in bundle.iterdir())
    assert not (storage / "01_generation").exists()
    assert not (storage / "02_datasets").exists()
    assert sorted(path.name for path in work.iterdir()) == ["user-sentinel.txt"]
    assert not (bundle.parent / ".active").exists()
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    assert not bundle.stat().st_mode & write_bits
    assert all(not path.stat().st_mode & write_bits for path in bundle.iterdir())


def test_probe_rejects_execution_outside_slurm_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    campaign_path, _ = generation_config_factory(
        simulation_profile=profiles.TRANSIENT_DRYING_PROFILE,
        executable=fake_comsol,
        natural_count=2,
        campaign_purpose="technical_runtime_smoke",
        scheduler_kind="slurm",
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    storage = tmp_path / "cpu-storage"

    with pytest.raises(RuntimeError, match="only inside one numeric Slurm job allocation"):
        probe.run_timing_probe(campaign_path, storage_root=storage)

    assert not storage.exists()


def test_failed_normal_case_still_publishes_compact_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    campaign_path, _ = generation_config_factory(
        simulation_profile=profiles.TRANSIENT_DRYING_PROFILE,
        executable=fake_comsol,
        natural_count=2,
        campaign_purpose="technical_runtime_smoke",
        scheduler_kind="slurm",
        license_retry_enabled=False,
    )
    _bind_fake_slurm(monkeypatch)
    storage = tmp_path / "cpu-storage"
    work = tmp_path / "normal-work"
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")

    result = probe.run_timing_probe(campaign_path, storage_root=storage, work_root=work)
    bundle = Path(result["probe_cpu_bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    attempt = manifest["normal_case_evidence"]["failure_attempt"]

    assert result["probe_case_state"] == "failed"
    assert result["probe_case_exit_code"] == 7
    assert manifest["exit_status"]["exit_code"] == 7
    assert manifest["normal_case_evidence"]["status"] == "failed"
    assert manifest["normal_case_evidence"]["license_wait"] is None
    assert attempt["receipt"]["payload"]["campaign_run_id"] == result["probe_id"]
    assert attempt["receipt"]["payload"]["case_state"] == "failed"
    assert attempt["receipt"]["payload"]["failure_stage"] == "solver"
    assert attempt["receipt"]["payload"]["process_exit_code"] == 7
    assert attempt["cleanup"]["payload"]["status"] == "complete"
    assert b"synthetic failure" in (bundle / "stderr.log").read_bytes()
    assert probe.validate_probe_bundle(bundle)["valid"] is True
    assert not (storage / "01_generation").exists()
    assert not (bundle.parent / ".active").exists()
    assert not work.exists() or not any(work.iterdir())


def test_temporary_license_deferral_retains_authoritative_normal_wait_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    campaign_path, _ = generation_config_factory(
        simulation_profile=profiles.TRANSIENT_DRYING_PROFILE,
        executable=fake_comsol,
        natural_count=2,
        campaign_purpose="technical_runtime_smoke",
        scheduler_kind="slurm",
        in_allocation_retry_enabled=False,
    )
    _bind_fake_slurm(monkeypatch)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")
    result = probe.run_timing_probe(
        campaign_path,
        storage_root=tmp_path / "cpu-storage",
        work_root=tmp_path / "normal-work",
    )
    bundle = Path(result["probe_cpu_bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    evidence = manifest["normal_case_evidence"]
    wait = evidence["license_wait"]["receipt"]["payload"]

    assert result["probe_case_state"] == "failed"
    assert result["probe_case_exit_code"] == 1
    assert evidence["status"] == "license_blocked"
    assert evidence["failure_attempt"] is None
    assert wait["schema_kind"] == "generation_temporary_license_wait"
    assert wait["campaign_run_id"] == result["probe_id"]
    assert wait["case_id"] == manifest["case_id"]
    assert wait["classification"] == "temporary_license_capacity"
    assert probe.validate_probe_bundle(bundle)["valid"] is True
    assert "GENERATION_CAMPAIGN_RUN_ID" not in os.environ


def _stage_probe_bundle(storage: Path, bundle: Path, probe_id: str) -> Path:
    staging = workspace_service.create_transfer_staging(storage_root=storage, run_id=probe_id)
    shutil.copytree(bundle, staging / probe_id)
    return staging


def test_transferred_probe_publication_is_exact_idempotent_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    result, _, _, _ = _successful_probe(tmp_path, monkeypatch, generation_config_factory, fake_comsol)
    bundle = Path(result["probe_cpu_bundle"])
    probe_id = str(result["probe_id"])
    gpu_storage = tmp_path / "gpu-storage"

    staging = _stage_probe_bundle(gpu_storage, bundle, probe_id)
    published = probe.publish_transferred_probe_bundle(
        probe_id,
        staging_root=staging,
        destination_root=gpu_storage,
    )
    workspace_service.cleanup_transfer_staging(staging, storage_root=gpu_storage, run_id=probe_id)
    target = Path(published["probe_bundle"])

    assert published["reused"] is False
    assert target == gpu_storage / "03_experiments" / probe.PROBE_SCOPE / probe_id
    assert probe.validate_probe_bundle(target)["valid"] is True
    assert all((target / name).read_bytes() == (bundle / name).read_bytes() for name in probe._EXACT_BUNDLE_INVENTORY)

    repeated_staging = _stage_probe_bundle(gpu_storage, bundle, probe_id)
    repeated = probe.publish_transferred_probe_bundle(
        probe_id,
        staging_root=repeated_staging,
        destination_root=gpu_storage,
    )
    workspace_service.cleanup_transfer_staging(repeated_staging, storage_root=gpu_storage, run_id=probe_id)
    assert repeated["reused"] is True

    target.chmod(0o755)
    readme = target / "README.md"
    readme.chmod(0o644)
    readme.write_text(readme.read_text(encoding="utf-8") + "corrupt\n", encoding="utf-8")
    conflicting_staging = _stage_probe_bundle(gpu_storage, bundle, probe_id)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        probe.publish_transferred_probe_bundle(
            probe_id,
            staging_root=conflicting_staging,
            destination_root=gpu_storage,
        )


def test_transferred_probe_rejects_unexpected_hdf5_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
    fake_comsol: Path,
) -> None:
    result, _, _, _ = _successful_probe(tmp_path, monkeypatch, generation_config_factory, fake_comsol)
    bundle = Path(result["probe_cpu_bundle"])
    probe_id = str(result["probe_id"])
    gpu_storage = tmp_path / "gpu-storage"
    staging = _stage_probe_bundle(gpu_storage, bundle, probe_id)
    staged_bundle = staging / probe_id
    staged_bundle.chmod(0o755)
    (staged_bundle / "case.h5").write_bytes(b"not allowed")

    with pytest.raises(ValueError, match="inventory differs"):
        probe.publish_transferred_probe_bundle(
            probe_id,
            staging_root=staging,
            destination_root=gpu_storage,
        )


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [("successful", 0), ("failed", 7)],
)
def test_timing_probe_cli_announces_identity_once_and_reports_retained_bundle(
    state: str,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(_config: Path, **kwargs: Any) -> dict[str, Any]:
        kwargs["announce"]({"probe_id": "probe-20260823T000000Z-0123456789ab"})
        return {
            "probe_id": "probe-20260823T000000Z-0123456789ab",
            "probe_case_state": state,
            "probe_case_exit_code": exit_code,
            "probe_cpu_bundle": "/storage/probe-bundle",
        }

    monkeypatch.setattr(cli_generation.timing_probe_service, "run_timing_probe", fake_run)
    code = cli_generation.main(
        [
            "timing-probe",
            str(tmp_path / "campaign.yaml"),
            "--storage-root",
            str(tmp_path / "storage"),
        ]
    )

    assert code == exit_code
    assert capsys.readouterr().out.splitlines() == [
        "PROBE_ID=probe-20260823T000000Z-0123456789ab",
        f"PROBE_CASE_STATE={state}",
        f"PROBE_CASE_EXIT_CODE={exit_code}",
        "PROBE_CPU_BUNDLE=/storage/probe-bundle",
    ]
