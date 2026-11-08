"""Focused contracts for non-authoritative case runtime progress.

Responsibilities:
  - Check incremental progress, atomic reporting, strict paths, freshness, and log resets

Design principles:
  - Progress remains non-authoritative and monitoring failures are nonfatal

This module does NOT:
  - Decide scientific success or replace persisted terminal evidence
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final

import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity
from src.generation.runtime import generation_runtime_progress as progress
from src.generation.runtime import generation_runtime_workspace as workspace

if TYPE_CHECKING:
    from pathlib import Path

_CAMPAIGN: Final[Any] = SimpleNamespace(batches=[SimpleNamespace(batch_name="batch", batch_id="batch-id")])


def _record(*, submission_index: int, allocation: str, case_index: int) -> dict[str, Any]:
    """Return the dispatch-record fields that bind progress to one case."""
    return {
        "submission_index": submission_index,
        "job_id": allocation,
        "case": _expected_identity(case_index),
    }


def _manifest(*records: dict[str, Any]) -> dict[str, Any]:
    """Return one synthetic campaign manifest; the default is one dispatch 12-3 of case 1."""
    return {"submissions": list(records) or [_record(submission_index=3, allocation="12", case_index=1)]}


def _expected_identity(case_index: int = 1) -> dict[str, Any]:
    """Return the exact synthetic case identity."""
    return {
        "batch_name": "batch",
        "batch_id": "batch-id",
        "case_index": case_index,
        "case_id": f"case_{case_index:04d}",
    }


def _dispatch(allocation: str = "12", submission_index: int = 3, step_id: str = "4") -> dispatch_identity.CaseDispatch:
    """Return one synthetic running case-step dispatch."""
    return dispatch_identity.CaseDispatch(
        allocation_id=allocation,
        step_id=step_id,
        submission_index=submission_index,
        dispatch_key=dispatch_identity.dispatch_key(allocation, submission_index),
    )


def _bind_campaign(
    monkeypatch: pytest.MonkeyPatch,
    run_directory: Path,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Bind progress persistence to one test-owned campaign directory."""
    bound = manifest if manifest is not None else _manifest()
    monkeypatch.setattr(progress.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: bound)
    monkeypatch.setattr(progress.campaign_evidence, "campaign_from_manifest", lambda *_args, **_kwargs: _CAMPAIGN)
    monkeypatch.setattr(progress.campaign_evidence, "campaign_run_directory", lambda *_args, **_kwargs: run_directory)
    common.paths.resolve_generation_execution_directory(run_directory).mkdir(parents=True, exist_ok=True)


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


def test_step_history_keeps_every_complete_transient_row_and_skips_other_tables() -> None:
    """The finished log yields the whole time-stepping history in order, without stationary or partial rows."""
    lines = [
        "Iter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes",
        "   1          64     3.2e+04   0.0100000          65    2    1    2  2.3e-13  3.1e-15",
        "Step        Time    Stepsize      Res  Jac  Sol Order Tfail NLfail   LinErr   LinRes",
        "           Current Progress:   4 % - Assembling residual",
        "   1      1.6355     0.42589       18    6   18     1     0      0  4.8e-16  4.6e-16",
        "   2      2.0614     0.42589       20    7   20     1     0      1  4.8e-16  4.6e-16",
        "   3      2.4",
    ]

    rows = progress.transient_step_history(lines)

    assert len(progress.TRANSIENT_STEP_HISTORY_COLUMNS) == len(rows[0])
    assert rows == [(1, 1.6355, 0.42589, 18, 6, 18, 1, 0, 0, 4.8e-16, 4.6e-16), (2, 2.0614, 0.42589, 20, 7, 20, 1, 0, 1, 4.8e-16, 4.6e-16)]
    assert progress.transient_step_history(lines[:2]) == []


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
    started = datetime(2026, 1, 1, tzinfo=UTC)
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
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
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
        dispatch=_dispatch(),
        hostname="node-a",
    )
    slurm = common.paths.resolve_generation_slurm_directory(common.paths.resolve_generation_execution_directory(run))
    assert reporter.path == slurm / "12-3.json"
    reporter.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(progress.RuntimeProgressIdentityError, match="identity conflicts"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            dispatch=_dispatch(),
            hostname="node-a",
        )
    reporter.path.unlink()
    slurm.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    slurm.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="directory is unsafe"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            dispatch=_dispatch(),
            hostname="node-a",
        )
    with pytest.raises(progress.RuntimeProgressIdentityError, match="not bound"):
        progress.RuntimeProgressReporter.create(
            "run",
            **_expected_identity(),
            dispatch=_dispatch(allocation="13"),
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
            dispatch=_dispatch(),
            hostname="node-a",
        )


def test_progress_binds_by_submission_index_allocation_and_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind each dispatch of one allocation to exactly its own run.json record and case."""
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(
        _record(submission_index=3, allocation="12", case_index=1),
        _record(submission_index=5, allocation="12", case_index=2),
        _record(submission_index=6, allocation="13", case_index=1),
    )
    _bind_campaign(monkeypatch, run, manifest)
    slurm = common.paths.resolve_generation_slurm_directory(common.paths.resolve_generation_execution_directory(run))

    first = progress.RuntimeProgressReporter.create("run", **_expected_identity(1), dispatch=_dispatch("12", 3, "4"), hostname="node-a")
    second = progress.RuntimeProgressReporter.create("run", **_expected_identity(2), dispatch=_dispatch("12", 5, "7"), hostname="node-a")
    assert (first.path, second.path) == (slurm / "12-3.json", slurm / "12-5.json")
    assert first.update(phase="preparing", force=True)
    assert second.update(phase="preparing", force=True)
    for dispatch, case_index in ((_dispatch("12", 3, "4"), 1), (_dispatch("12", 5, "7"), 2)):
        loaded = progress.load_runtime_progress("run", dispatch.dispatch_key, _expected_identity(case_index), campaign=_CAMPAIGN, manifest=manifest)
        assert loaded["availability"] == "available"
        assert loaded["dispatch_key"] == dispatch.dispatch_key
        assert loaded["slurm_job_id"] == "12"
        assert loaded["slurm_step_id"] == dispatch.step_id
        assert loaded["case_index"] == case_index

    for wrong_dispatch, case_index in (
        (_dispatch("12", 5), 1),
        (_dispatch("12", 4), 1),
        (_dispatch("13", 3), 1),
        (_dispatch("12", 6), 1),
    ):
        with pytest.raises(progress.RuntimeProgressIdentityError, match="not bound"):
            progress.RuntimeProgressReporter.create("run", **_expected_identity(case_index), dispatch=wrong_dispatch, hostname="node-a")
        unavailable = progress.load_runtime_progress(
            "run", wrong_dispatch.dispatch_key, _expected_identity(case_index), campaign=_CAMPAIGN, manifest=manifest
        )
        assert unavailable["availability"] == "unavailable"

    foreign = {**_expected_identity(1), "batch_id": "other-batch-id"}
    with pytest.raises(progress.RuntimeProgressIdentityError, match="batch identity is not bound"):
        progress.RuntimeProgressReporter.create("run", **foreign, dispatch=_dispatch("12", 3), hostname="node-a")
    assert progress.load_runtime_progress("run", "12-3", foreign, campaign=_CAMPAIGN, manifest=manifest)["availability"] == "unavailable"

    duplicated = _manifest(
        _record(submission_index=3, allocation="12", case_index=1),
        _record(submission_index=3, allocation="12", case_index=1),
    )
    _bind_campaign(monkeypatch, run, duplicated)
    with pytest.raises(progress.RuntimeProgressIdentityError, match="not bound"):
        progress.RuntimeProgressReporter.create("run", **_expected_identity(1), dispatch=_dispatch("12", 3), hostname="node-b")

    other_step = json.loads(first.path.read_text(encoding="utf-8"))
    other_step["slurm_step_id"] = "9"
    first.path.write_text(json.dumps(other_step), encoding="utf-8")
    _bind_campaign(monkeypatch, run, manifest)
    with pytest.raises(progress.RuntimeProgressIdentityError, match="identity conflicts"):
        progress.RuntimeProgressReporter.create("run", **_expected_identity(1), dispatch=_dispatch("12", 3, "4"), hostname="node-a")


def test_loader_is_read_only_and_marks_missing_malformed_and_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat all optional receipt faults as explicit unavailable evidence."""
    run = tmp_path / "run"
    run.mkdir()
    _bind_campaign(monkeypatch, run)
    expected = _expected_identity()

    missing = progress.load_runtime_progress("run", "12-3", expected, campaign=_CAMPAIGN, manifest=_manifest())
    assert missing["reason"] == "not_reported"
    assert not common.paths.resolve_generation_slurm_directory(common.paths.resolve_generation_execution_directory(run)).exists()

    recorded_at = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(progress, "_utc_now", lambda: recorded_at)
    reporter = progress.RuntimeProgressReporter.create(
        "run",
        **expected,
        dispatch=_dispatch(),
        hostname="node-a",
    )
    assert reporter.update(phase="preparing", force=True)
    fresh = progress.load_runtime_progress(
        "run",
        "12-3",
        expected,
        campaign=_CAMPAIGN,
        manifest=_manifest(),
        now=recorded_at + timedelta(seconds=5),
    )
    assert fresh["availability"] == "available"
    assert fresh["age_seconds"] == 5.0
    assert fresh["stale"] is False
    stale = progress.load_runtime_progress(
        "run",
        "12-3",
        expected,
        campaign=_CAMPAIGN,
        manifest=_manifest(),
        now=recorded_at + timedelta(seconds=601),
    )
    assert stale["stale"] is True

    invalid_optional = json.loads(reporter.path.read_text(encoding="utf-8"))
    invalid_optional["time_failures"] = "not-an-integer"
    reporter.path.write_text(json.dumps(invalid_optional), encoding="utf-8")
    unsupported = progress.load_runtime_progress("run", "12-3", expected, campaign=_CAMPAIGN, manifest=_manifest())
    assert unsupported["reason"] == "invalid_or_unsupported"

    without_step = json.loads(reporter.path.read_text(encoding="utf-8"))
    without_step["time_failures"] = None
    del without_step["slurm_step_id"]
    reporter.path.write_text(json.dumps(without_step), encoding="utf-8")
    assert progress.load_runtime_progress("run", "12-3", expected, campaign=_CAMPAIGN, manifest=_manifest())["reason"] == "invalid_or_unsupported"
    assert progress.load_runtime_progress("run", "12", expected, campaign=_CAMPAIGN, manifest=_manifest())["reason"] == "invalid_or_unavailable"

    reporter.path.write_text("not json\n", encoding="utf-8")
    malformed = progress.load_runtime_progress("run", "12-3", expected, campaign=_CAMPAIGN, manifest=_manifest())
    assert malformed["availability"] == "unavailable"


def test_checkout_log_reset_cannot_reuse_stale_solver_progress(tmp_path: Path) -> None:
    """Reset parser offsets and state when the owned startup log is truncated."""
    receipt = tmp_path / "42.json"
    stdout = tmp_path / "stdout.log"
    stdout.write_text("Time-Dependent Solver 1 in Transient Drying\n", encoding="utf-8")
    reporter = progress.RuntimeProgressReporter({"campaign_run_id": "campaign__0123456789abcdef"}, receipt, stdout_path=stdout)
    assert reporter.update(phase="starting_solver", force=True)
    assert json.loads(receipt.read_text(encoding="utf-8"))["phase"] == "transient_drying"

    stdout.write_text("Licensed number of users already reached\n", encoding="utf-8")
    reporter.reset_stdout(stdout)
    assert reporter.update_license_acquisition(
        window_seconds=3.0,
        window_limit_seconds=17.0,
        checkout_attempt_count=2,
        last_result="temporary_license_capacity",
        force=True,
    )

    loaded = json.loads(receipt.read_text(encoding="utf-8"))
    assert loaded["phase"] == "acquiring_comsol_license"
    assert loaded["parser_state"] == "unavailable"
    assert loaded["license_checkout_attempt_count"] == 2


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
    lifecycle.create_campaign(campaign, storage_root=storage)
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
