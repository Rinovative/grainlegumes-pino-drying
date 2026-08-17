# ruff: noqa: PLR2004, S101
"""Focused contracts for non-authoritative case runtime progress."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest

from src import generation
from src.generation.runtime import generation_runtime_progress as progress
from src.generation.runtime import generation_runtime_workspace as workspace

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict[str, Any]:
    """Return one exact synthetic campaign submission manifest."""
    return {
        "batches": [{"batch_name": "batch", "batch_id": "batch-id"}],
        "submissions": [
            {
                "job_id": "12",
                "case": {
                    "batch_name": "batch",
                    "batch_id": "batch-id",
                    "case_index": 1,
                    "case_id": "case_0001",
                },
            }
        ],
    }


def _expected_identity() -> dict[str, Any]:
    """Return the exact synthetic case identity."""
    return {
        "batch_name": "batch",
        "batch_id": "batch-id",
        "case_index": 1,
        "case_id": "case_0001",
    }


def _bind_campaign(
    monkeypatch: pytest.MonkeyPatch,
    run_directory: Path,
) -> None:
    """Bind progress persistence to one test-owned campaign directory."""
    monkeypatch.setattr(progress.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: _manifest())
    monkeypatch.setattr(progress.campaign_evidence, "campaign_run_directory", lambda *_args, **_kwargs: run_directory)


def test_transient_parser_uses_only_complete_supported_numeric_rows() -> None:
    """Parse the last complete transient row without modifying source lines."""
    lines = [
        "<---- Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution",
        "Step Time Stepsize Res Jac Sol Order Tfail NLfail LinErr LinRes",
        "1 360 0.075 2 3 4 1 0 5 1e-8 2e-9",
        "2 720 0.00938 5 6 7 2 1 15 2e-8 3e-9",
        "3 1080 0.01",
    ]
    original = list(lines)
    parser = progress.ComsolProgressParser()

    result = parser.consume(lines)

    assert lines == original
    assert result["parser_state"] == "available"
    assert result["comsol_section"] == "transient_drying"
    assert result["step_index"] == 2
    assert result["simulated_time_seconds"] == 720.0
    assert result["step_size_seconds"] == 0.00938
    assert result["residual_evaluations"] == 5
    assert result["jacobian_evaluations"] == 6
    assert result["linear_solves"] == 7
    assert result["order"] == 2
    assert result["time_failures"] == 1
    assert result["nonlinear_failures"] == 15
    unsupported = progress.ComsolProgressParser().consume(["2 720 0.00938 5 6 7 2 1 15 2e-8 3e-9"])
    assert unsupported == {"parser_state": "unavailable"}


def test_stationary_parser_reports_iteration_and_raw_comsol_stage() -> None:
    """Recognize the retained stationary heading and nonlinear table."""
    parser = progress.ComsolProgressParser()

    result = parser.consume(
        [
            "<---- Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution",
            "Iter SolEst ResEst Damping Stepsize #Res #Jac #Sol LinErr LinRes",
            "4 3.6e-13 1.7e+04 1.0 5.8e-13 6 4 8 2.7e-14 6e-16",
            "---------- Current Progress: 100 % - Assembling residual",
            "malformed stationary row",
        ]
    )

    assert result["parser_state"] == "available"
    assert result["comsol_section"] == "stationary_airflow"
    assert result["nonlinear_iteration"] == 4
    assert result["comsol_progress_percent"] == 100
    assert "simulated_time_seconds" not in result


def test_reporter_is_incremental_atomic_rate_limited_and_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve partial lines and write changes, heartbeat, and terminal state."""
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = iter(
        (
            started,
            started + timedelta(seconds=20),
            started + timedelta(seconds=40),
            started + timedelta(seconds=321),
            started + timedelta(seconds=322),
        )
    )
    monkeypatch.setattr(progress, "_utc_now", lambda: next(clock))
    stdout = tmp_path / "stdout.log"
    initial = b"Step Time Stepsize Res Jac Sol Order Tfail NLfail LinErr LinRes\n1 0.1"
    stdout.write_bytes(initial)
    receipt = tmp_path / "receipt.json"
    reporter = progress.RuntimeProgressReporter(
        {"campaign_run_id": "run"},
        receipt,
        stdout_path=stdout,
        started_at=started,
    )

    assert reporter.update(phase="starting_solver", force=True)
    first = json.loads(receipt.read_text(encoding="utf-8"))
    first_inode = receipt.stat().st_ino
    assert first["phase"] == "transient_drying"
    assert first["step_index"] is None

    appended = b" 0.1 2 3 4 1 0 0 0 0\n"
    with stdout.open("ab") as stream:
        stream.write(appended)
    assert reporter.update(phase="starting_solver")
    assert receipt.stat().st_ino != first_inode
    assert json.loads(receipt.read_text(encoding="utf-8"))["step_index"] == 1
    assert reporter.update(phase="starting_solver") is False
    assert reporter.update(phase="starting_solver")
    assert reporter.update(phase="completed", terminal=True)
    terminal = json.loads(receipt.read_text(encoding="utf-8"))
    assert terminal["phase"] == "completed"
    assert terminal["terminal"] is True
    assert stdout.read_bytes() == initial + appended


def test_progress_write_failure_is_nonfatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return unavailable progress rather than propagating a write failure."""
    reporter = progress.RuntimeProgressReporter(
        {"campaign_run_id": "run"},
        tmp_path / "receipt.json",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    def fail_write(*_args: object, **_kwargs: object) -> None:
        message = "synthetic progress write failure"
        raise OSError(message)

    monkeypatch.setattr(progress.common.serialization, "atomic_write_json", fail_write)

    assert reporter.update(phase="preparing", force=True) is False


def test_strict_factory_rejects_conflicts_and_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exact campaign/job/case binding and symlink-free ownership."""
    run = tmp_path / "run"
    run.mkdir()
    _bind_campaign(monkeypatch, run)

    reporter = progress.RuntimeProgressReporter.create(
        "run",
        **_expected_identity(),
        slurm_job_id="12",
        hostname="node-a",
    )
    assert reporter.path == run / "progress" / "12.json"
    reporter.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity conflicts"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            slurm_job_id="12",
            hostname="node-a",
        )
    reporter.path.unlink()
    (run / "progress").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "progress").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="directory is unsafe"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            slurm_job_id="12",
            hostname="node-a",
        )
    with pytest.raises(ValueError, match="not bound"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            slurm_job_id="13",
            hostname="node-a",
        )

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(
        progress.campaign_evidence,
        "campaign_run_directory",
        lambda *_args, **_kwargs: linked_root / "run",
    )
    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            slurm_job_id="12",
            hostname="node-a",
        )


def test_loader_is_read_only_and_marks_missing_malformed_and_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat all optional receipt faults as explicit unavailable evidence."""
    run = tmp_path / "run"
    run.mkdir()
    _bind_campaign(monkeypatch, run)
    expected = _expected_identity()

    missing = progress.load_runtime_progress("run", "12", expected, manifest=_manifest())
    assert missing["reason"] == "not_reported"
    assert not (run / "progress").exists()

    recorded_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(progress, "_utc_now", lambda: recorded_at)
    reporter = progress.RuntimeProgressReporter.create(
        "run",
        **expected,
        slurm_job_id="12",
        hostname="node-a",
    )
    assert reporter.update(phase="preparing", force=True)
    fresh = progress.load_runtime_progress(
        "run",
        "12",
        expected,
        manifest=_manifest(),
        now=recorded_at + timedelta(seconds=5),
    )
    assert fresh["availability"] == "available"
    assert fresh["age_seconds"] == 5.0
    assert fresh["stale"] is False
    stale = progress.load_runtime_progress(
        "run",
        "12",
        expected,
        manifest=_manifest(),
        now=recorded_at + timedelta(seconds=601),
    )
    assert stale["stale"] is True

    invalid_optional = json.loads(reporter.path.read_text(encoding="utf-8"))
    invalid_optional["time_failures"] = "not-an-integer"
    reporter.path.write_text(json.dumps(invalid_optional), encoding="utf-8")
    unsupported = progress.load_runtime_progress("run", "12", expected, manifest=_manifest())
    assert unsupported["reason"] == "invalid_or_unsupported"

    reporter.path.write_text("not json\n", encoding="utf-8")
    malformed = progress.load_runtime_progress("run", "12", expected, manifest=_manifest())
    assert malformed["availability"] == "unavailable"


@pytest.mark.integration
def test_monitoring_failure_does_not_change_the_simulated_case_path(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Keep a fake solver execution successful when every monitor hook fails."""
    config_path, _template = generation_config_factory(executable=fake_comsol)
    campaign = generation.cases.config.load_campaign_config(config_path)
    config = campaign.batches[0]
    storage = tmp_path / "storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    prepared = generation.runtime.prepare_case_work_directory(
        config,
        config.case_indices[0],
        storage_root=storage,
        work_root=tmp_path / "work",
    )

    class BrokenReporter(progress.RuntimeProgressReporter):
        """Raise from every optional monitoring hook."""

        def __init__(self) -> None:
            """Initialize without operational persistence state."""

        def bind_stdout(self, _path: Path) -> None:
            message = "synthetic parser setup failure"
            raise RuntimeError(message)

        def update(self, *, phase: str, terminal: bool = False, force: bool = False) -> bool:
            _ = (phase, terminal, force)
            message = "synthetic progress write failure"
            raise OSError(message)

    result = generation.runtime.execute_prepared_case(
        config,
        prepared,
        cores_per_case=1,
        worker_slot=0,
        progress_reporter=BrokenReporter(),
    )

    assert result.canonical_case.path.is_file()
    assert (prepared.runtime_directory / "stdout.log").is_file()
    workspace.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=prepared.work_root,
        storage_root=(tmp_path / "storage").resolve(),
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
    )
