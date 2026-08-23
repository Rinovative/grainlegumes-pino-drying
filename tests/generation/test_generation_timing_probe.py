# ruff: noqa: D103, EM101, PLR2004, S101, SLF001, TRY003
"""Focused software contracts for the temporary COMSOL timing probe."""

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common
from src.generation import generation_timing_probe as probe
from src.generation.cli import cli_generation
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.runtime import generation_runtime_comsol as comsol_service


def test_parser_classifies_supported_candidates_without_claiming_real_grammar() -> None:
    data = b"Stationary solver\nSolution time: 2 min\nTransient solver\nSolution time: 3.5e1 s\n"
    records = probe.parse_solution_times(data)
    summary = probe.summarize_solution_times(records)

    assert [record["converted_seconds"] for record in records] == [120.0, 35.0]
    assert [record["line_number"] for record in records] == [2, 4]
    assert [record["byte_offset"] for record in records] == [len(b"Stationary solver\n"), len(data) - len(b"Solution time: 3.5e1 s\n")]
    assert all(len(record["context"].encode()) <= probe._MAX_CONTEXT_BYTES for record in records)
    assert summary["phases"]["stationary"]["status"] == "confirmed"
    assert summary["phases"]["transient"]["status"] == "confirmed"
    assert summary["candidate_scientific_sum_seconds"] == 155.0
    assert summary["diagnostic_only"] is True
    assert "comsol_scientific_solver_seconds" not in summary


def test_parser_reports_duplicate_nested_malformed_and_unsupported_candidates() -> None:
    data = (
        b"Stationary solver\n"
        b"  Solution time: 1 s\n"
        b"Solution time: 1 s\n"
        b"Solution time: 1 s\n"
        b"Transient solver\n"
        b"Solution time: bananas\n"
        b"Solution time: 3 fortnights\n"
    )
    records = probe.parse_solution_times(data)
    stationary = [record for record in records if record["detected_phase"] == "stationary"]
    transient = [record for record in records if record["detected_phase"] == "transient"]
    summary = probe.summarize_solution_times(records)

    assert stationary[0]["classification"] == "nested"
    assert stationary[1]["duplicate_classification"] == "duplicate"
    assert stationary[2]["duplicate_classification"] == "duplicate"
    assert [record["parse_status"] for record in transient] == ["malformed", "unsupported_format"]
    assert summary["phases"]["stationary"]["status"] == "ambiguous"
    assert summary["phases"]["transient"]["status"] == "ambiguous"
    assert summary["candidate_scientific_sum_seconds"] is None


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

    path.write_bytes(path.read_bytes() + b"\nTransient\nSolution time: 2 s\n")
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
    path.write_bytes(b"Transient\nSolution time: 4 s\n")
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
    command = comsol_service.build_comsol_command(
        _command_config([]),
        cores_per_case=4,
        diagnostic_batchlog="/attempt/runtime/comsol_batch.log",
    )
    assert command.count("-batchlog") == 1
    assert command.count("-batchlogout") == 1
    assert command[command.index("-batchlog") + 1] == "/attempt/runtime/comsol_batch.log"
    assert "-np" in command


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


def test_pid_liveness_reaps_finished_direct_child(monkeypatch: pytest.MonkeyPatch) -> None:
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(probe.os, "waitpid", lambda pid, _options: (pid, 0))
    monkeypatch.setattr(probe.os, "kill", lambda pid, signal: kill_calls.append((pid, signal)))

    assert probe._pid_is_alive(4_321) is False
    assert kill_calls == []


def test_observed_wall_and_same_process_verdicts_remain_diagnostic() -> None:
    events = [
        {"event_kind": "stationary_phase_first_observed", "observed_monotonic_ns": 10},
        {"event_kind": "stationary_completion_observed", "observed_monotonic_ns": 20},
        {"event_kind": "transient_phase_first_observed", "observed_monotonic_ns": 30},
        {"event_kind": "transient_completion_observed", "observed_monotonic_ns": 50},
    ]
    observed = probe._observed_wall(events)
    batch_summary = {
        "method_status": "unresolved",
        "diagnostic_only": True,
        "phases": {},
    }
    verdicts = probe._method_verdicts(batch_summary, observed)

    assert observed["stationary_airflow_observed_wall_seconds"] == pytest.approx(1e-8)
    assert observed["transient_drying_observed_wall_seconds"] == pytest.approx(2e-8)
    assert observed["diagnostic_only"] is True
    assert verdicts["same_process_method"]["method_status"] == "not_implementable_from_current_source_boundary"
    assert verdicts["same_process_method"]["exact_boundary_proven"] is False
    assert verdicts["recommendation"] == "unresolved"
    assert verdicts["production_timing_fields_updated"] is False


def _install_fake_probe_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
) -> tuple[Path, Path]:
    fake_script = tmp_path / "timing_probe_fake_comsol.py"
    fake_script.write_text(
        r"""#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time

if sys.argv[1:] == ["-version"]:
    print("COMSOL Multiphysics 6.4.0.293")
    raise SystemExit(0)

arguments = sys.argv[1:]
if arguments.count("-batchlog") != 1 or arguments.count("-batchlogout") != 1:
    raise RuntimeError("timing-probe fake COMSOL requires one batch-log owner")
batch_log = Path(arguments[arguments.index("-batchlog") + 1])
counter = Path(os.environ["GENERATION_TIMING_PROBE_TEST_COUNTER"])
count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
counter.write_text(str(count), encoding="utf-8")
with batch_log.open("ab") as stream:
    stream.write(b"Stationary solver\n")
    stream.flush()
print("Stationary solver", flush=True)
time.sleep(0.1)
with batch_log.open("ab") as stream:
    stream.write(b"Solution time: 1.25 s\nTransient solver\n")
    stream.flush()
print("Solution time: 1.25 s", flush=True)
print("Transient solver", flush=True)
time.sleep(0.1)
with batch_log.open("ab") as stream:
    stream.write(b"Solution time: 2.5 s\n")
    stream.flush()
print("Solution time: 2.5 s", flush=True)
""",
        encoding="utf-8",
    )
    fake_script.chmod(fake_script.stat().st_mode | stat.S_IXUSR)
    counter = tmp_path / "timing_probe_fake_comsol_count.txt"
    repository_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "1" * 40)
    monkeypatch.setenv("GENERATION_TIMING_PROBE_TEST_COUNTER", str(counter))
    monkeypatch.setenv("PYTHONPATH", str(repository_root))
    campaign_path, _template_path = generation_config_factory(
        simulation_profile=profiles.TRANSIENT_DRYING_PROFILE,
        executable=fake_script,
        natural_count=1,
        campaign_purpose="technical_runtime_smoke",
    )
    campaign = probe.config_service.load_campaign_config(campaign_path)
    if len(campaign.batches) != 1:
        raise AssertionError("timing-probe fixture requires one synthetic batch")
    batch = campaign.batches[0]
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        json.dumps(
            {
                "schema_kind": "generation_timing_probe",
                "schema_version": 1,
                "campaign_config": str(campaign_path),
                "batch_name": batch.batch_name,
                "case_index": batch.case_indices[0],
                "cores_per_case": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, counter


def test_probe_resumes_one_child_and_publishes_exact_immutable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
) -> None:
    config_path, counter = _install_fake_probe_runtime(tmp_path, monkeypatch, generation_config_factory)
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    original_observer = probe._observe_until_exit
    announcements: list[dict[str, str]] = []

    def interrupt_controller(_active: Path, _session: Any, _child_pid: int) -> Any:
        raise RuntimeError("synthetic controller interruption")

    monkeypatch.setattr(probe, "_observe_until_exit", interrupt_controller)
    with pytest.raises(RuntimeError, match="controller interruption"):
        probe.run_timing_probe(
            config_path,
            storage_root=storage,
            work_root=work,
            announce=lambda payload: announcements.append(dict(payload)),
        )
    assert announcements[0]["probe_id"]
    active = Path(announcements[0]["probe_active"])
    assert active.is_dir()
    prepared_session = json.loads((active / "session.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(probe, "_observe_until_exit", original_observer)
    result = probe.run_timing_probe(
        config_path,
        storage_root=storage,
        work_root=work,
        announce=lambda payload: announcements.append(dict(payload)),
    )
    bundle = Path(result["probe_bundle"])

    assert result["resumed"] is True
    assert announcements[0]["probe_id"] == announcements[1]["probe_id"] == result["probe_id"]
    assert counter.read_text(encoding="utf-8") == "1"
    assert not Path(announcements[0]["probe_active"]).exists()
    assert not (storage / "01_generation").exists()
    assert not (storage / "02_datasets").exists()
    assert probe.validate_probe_bundle(bundle)["valid"] is True
    assert {path.name for path in bundle.iterdir()} == probe._EXACT_BUNDLE_INVENTORY
    assert not (bundle / "in_process_probe_timing.json").exists()
    assert (bundle / "comsol_batch.log").read_bytes() == (b"Stationary solver\nSolution time: 1.25 s\nTransient solver\nSolution time: 2.5 s\n")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    verdicts = json.loads((bundle / "method_verdicts.json").read_text(encoding="utf-8"))
    observed = json.loads((bundle / "observed_wall_timing.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "1" * 40
    assert manifest["case_input_id"] == prepared_session["case_input_id"]
    assert manifest["simulation_case_id"] == prepared_session["simulation_case_id"]
    assert manifest["attempt_id"]
    assert manifest["files"]["comsol_batch.log"]["size_bytes"] == (bundle / "comsol_batch.log").stat().st_size
    assert verdicts["batch_log_method"]["method_status"] == "confirmed"
    assert verdicts["same_process_method"]["method_status"] == "not_implementable_from_current_source_boundary"
    assert verdicts["recommendation"] == "comsol_batch_log_solution_time"
    assert observed["diagnostic_only"] is True
    assert "stationary_airflow_observed_wall_seconds" in observed
    assert "transient_drying_observed_wall_seconds" in observed
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    assert not bundle.stat().st_mode & write_bits
    assert all(not path.stat().st_mode & write_bits for path in bundle.iterdir())

    bundle.chmod(0o755)
    manifest_path = bundle / "manifest.json"
    manifest_path.chmod(0o644)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        probe.validate_probe_bundle(bundle)


@pytest.mark.parametrize("tamper_kind", ["control_only", "coordinated_command", "case_bundle"])
def test_probe_tampering_fails_before_child_execution(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
) -> None:
    config_path, counter = _install_fake_probe_runtime(tmp_path, monkeypatch, generation_config_factory)
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    original_start = probe._start_or_admit_child
    announcements: list[dict[str, str]] = []

    def pause_before_child(_active: Path, _session: Any) -> int:
        raise RuntimeError("prepared pause")

    monkeypatch.setattr(probe, "_start_or_admit_child", pause_before_child)
    with pytest.raises(RuntimeError, match="prepared pause"):
        probe.run_timing_probe(
            config_path,
            storage_root=storage,
            work_root=work,
            announce=lambda payload: announcements.append(dict(payload)),
        )
    active = Path(announcements[0]["probe_active"])
    control_path = active / "child_control.json"
    session_path = active / "session.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if tamper_kind == "control_only":
        control["command"] = [sys.executable, "-c", "raise SystemExit(77)"]
        control_path.write_text(json.dumps(control) + "\n", encoding="utf-8")
    elif tamper_kind == "coordinated_command":
        command = [sys.executable, "-c", "raise SystemExit(78)"]
        attempt_id = common.serialization.canonical_json_sha256(
            {
                "probe_id": session["probe_id"],
                "source_commit": session["source_commit"],
                "case_input_id": session["case_input_id"],
                "simulation_case_id": session["simulation_case_id"],
                "command": command,
            }
        )
        session["command"] = command
        session["attempt_id"] = attempt_id
        control["command"] = command
        control["attempt_id"] = attempt_id
        session["control_sha256"] = common.serialization.canonical_json_sha256(control)
        control_path.write_text(json.dumps(control) + "\n", encoding="utf-8")
        session_path.write_text(json.dumps(session) + "\n", encoding="utf-8")
    else:
        case_path = Path(session["work_path"]) / "case.json"
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_payload["simulation_case_id"] = "9" * 64
        case_path.write_text(json.dumps(case_payload) + "\n", encoding="utf-8")

    popen_calls: list[list[str]] = []

    def forbidden_popen(command: list[str], **_kwargs: Any) -> Any:
        popen_calls.append(command)
        raise AssertionError("tampered timing probe reached subprocess execution")

    monkeypatch.setattr(probe, "_start_or_admit_child", original_start)
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden_popen)
    with pytest.raises(RuntimeError, match=r"control|command|case|session"):
        probe._execute_child(control_path)
    assert popen_calls == []
    with pytest.raises(RuntimeError, match=r"control|command|case|session"):
        probe.run_timing_probe(
            config_path,
            storage_root=storage,
            work_root=work,
        )
    assert popen_calls == []
    assert not counter.exists()
    assert active.is_dir()


def test_monitor_failure_is_evidence_and_does_not_kill_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_config_factory: Any,
) -> None:
    config_path, counter = _install_fake_probe_runtime(tmp_path, monkeypatch, generation_config_factory)
    original_observe = probe.observe_appended_bytes
    calls = {"count": 0}

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("synthetic observer failure")
        return original_observe(*args, **kwargs)

    monkeypatch.setattr(probe, "observe_appended_bytes", fail_once)
    result = probe.run_timing_probe(
        config_path,
        storage_root=tmp_path / "storage",
        work_root=tmp_path / "work",
    )
    bundle = Path(result["probe_bundle"])
    events = [json.loads(line) for line in (bundle / "phase_events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["exit_code"] == 0
    assert counter.read_text(encoding="utf-8") == "1"
    assert any(event["event_kind"] == "monitor_error" for event in events)
    assert json.loads((bundle / "parser_summary.json").read_text(encoding="utf-8"))["monitor_error_count"] == 1
    assert probe.validate_probe_bundle(bundle)["valid"] is True


def test_timing_probe_cli_announces_identity_before_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(_config: Path, **kwargs: Any) -> dict[str, Any]:
        kwargs["announce"](
            {
                "probe_id": "probe-test",
                "probe_active": "/storage/03_experiments/comsol_phase_timing_probe/.active/key",
                "probe_work": "/work/probe-test",
            }
        )
        return {"probe_id": "probe-test", "probe_bundle": "/storage/probe-test", "exit_code": 0}

    monkeypatch.setattr(cli_generation.timing_probe_service, "run_timing_probe", fake_run)
    code = cli_generation.main(
        [
            "timing-probe",
            str(tmp_path / "probe.yaml"),
            "--storage-root",
            str(tmp_path / "storage"),
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines == [
        "PROBE_ID=probe-test",
        "PROBE_ACTIVE=/storage/03_experiments/comsol_phase_timing_probe/.active/key",
        "PROBE_WORK=/work/probe-test",
        "PROBE_BUNDLE=/storage/probe-test",
    ]
