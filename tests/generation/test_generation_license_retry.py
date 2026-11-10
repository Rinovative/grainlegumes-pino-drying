"""Temporary COMSOL floating-license capacity retry contracts.

Responsibilities:
  - Check capacity classification, bounded waits, recovery receipts, cancellation, and retry
    retention

Design principles:
  - License acquisition stays separate from scientific attempts and solver failure

This module does NOT:
  - Acquire real licenses or suppress genuine solver errors
"""

from __future__ import annotations

import json
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from queue import Queue
from typing import TYPE_CHECKING, Any, cast

import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.runtime import generation_runtime_batch as batch_runtime
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_stop as stop_service

if TYPE_CHECKING:
    from pathlib import Path

_OBSERVED_CAPACITY_TEXT = """
Could not obtain license for 'Brinkman Equations (br)'.
Required product: CFD Module.
License error: -4.
Licensed number of users already reached.
Feature: COMSOL
FlexNet Licensing error:-4,132
"""


class _Clock:
    """Small monotonic clock advanced by bounded fake-process waits."""

    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _WaitingProcess:
    """Fake process that remains alive until the controller terminates it."""

    pid = 4172

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    @staticmethod
    def poll() -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            pytest.fail("Startup wait must remain bounded.")
        self.clock.advance(timeout)
        command = "comsol"
        raise subprocess.TimeoutExpired(command, timeout)


def _capacity_error(work_directory: Path) -> license_service.TemporaryLicenseCapacityError:
    """Return one classified synthetic capacity exception."""
    evidence = license_service.classify_temporary_license_capacity(_OBSERVED_CAPACITY_TEXT)
    assert evidence is not None
    return license_service.TemporaryLicenseCapacityError(
        "synthetic temporary license capacity",
        work_directory=work_directory,
        command=("comsol", "batch"),
        exit_code=0,
        evidence=evidence,
    )


def _source_run_id(config: Any) -> str:
    """Return the current source campaign identity for one resolved batch."""
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    return generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)


def _dispatch_environment(monkeypatch: pytest.MonkeyPatch, allocation_id: str, submission_index: int, *, step_id: str = "0") -> str:
    """Bind the process to one campaign case dispatch and return its key."""
    key = dispatch_identity.dispatch_key(allocation_id, submission_index)
    monkeypatch.setenv("SLURM_JOB_ID", allocation_id)
    monkeypatch.setenv("SLURM_STEP_ID", step_id)
    monkeypatch.setenv("GENERATION_SUBMISSION_INDEX", str(submission_index))
    monkeypatch.setenv("GENERATION_DISPATCH_KEY", key)
    return key


def _prepare_current_source_inputs(config: Any, count: int, storage: Path) -> Any:
    """Admit the original design before running a bounded candidate fixture."""
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    lifecycle.create_campaign(campaign, storage_root=storage)
    return generation.cases.input_generation.generate_input_cases(config, count, storage_root=storage)


def test_license_capacity_classifier_is_strong_and_conservative() -> None:
    """Recognize capacity evidence without broad license/model false positives."""
    observed = license_service.classify_temporary_license_capacity(_OBSERVED_CAPACITY_TEXT)
    assert observed is not None
    assert observed.classification == "temporary_license_capacity"
    assert observed.feature == "Brinkman Equations (br)"
    assert observed.license_code == "-4,132"

    add_on = license_service.classify_temporary_license_capacity(
        "Could not obtain license for 'Heat Transfer Module'.\nLicensed number of users already reached."
    )
    assert add_on is not None
    assert add_on.feature == "Heat Transfer Module"

    terminal_messages = (
        "Required product is not licensed for this installation.",
        ("Could not obtain license for 'CFD Module': invalid license file. License error: -4."),
        ("Could not obtain license for 'CFD Module': product is not licensed. License error: -4."),
        "License server configuration error: host unreachable.",
        "Nonlinear solver did not converge.",
        "Mesh generation failed.",
        "Configured export is missing.",
        "HDF5 conversion validation failed.",
    )
    assert all(license_service.classify_temporary_license_capacity(message) is None for message in terminal_messages)


def test_startup_waiter_preserves_owned_deadline_and_cancellation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate a silent checkout at its deadline without masking cancellation."""
    clock = _Clock()
    process = _WaitingProcess(clock)
    terminated: list[_WaitingProcess] = []

    monkeypatch.setattr(batch_runtime.time, "monotonic", clock)
    monkeypatch.setattr(batch_runtime, "_captured_startup_text", lambda _prepared: "")
    monkeypatch.setattr(batch_runtime, "runtime_cancellation_requested", lambda: False)

    def terminate(owned: _WaitingProcess) -> int:
        terminated.append(owned)
        return -signal.SIGTERM

    monkeypatch.setattr(batch_runtime, "_terminate_solver_and_wait", terminate)
    prepared: Any = object()
    outcome, exit_code = batch_runtime._wait_for_solver_start_or_exit(
        cast("subprocess.Popen[str]", process),
        prepared,
        deadline=10.5,
        window_started_monotonic=10.0,
        window_limit_seconds=0.5,
        checkout_attempt_count=1,
        last_result=None,
        progress_reporter=None,
    )

    assert outcome == "window_deadline"
    assert exit_code == -signal.SIGTERM
    assert terminated == [process]
    assert clock() == 10.5

    cancelled_process = _WaitingProcess(clock)
    monkeypatch.setattr(batch_runtime, "runtime_cancellation_requested", lambda: True)
    cancelled, cancelled_exit_code = batch_runtime._wait_for_solver_start_or_exit(
        cast("subprocess.Popen[str]", cancelled_process),
        prepared,
        deadline=clock(),
        window_started_monotonic=10.0,
        window_limit_seconds=0.5,
        checkout_attempt_count=1,
        last_result=None,
        progress_reporter=None,
    )

    assert cancelled == "cancelled"
    assert cancelled_exit_code == -signal.SIGTERM
    assert terminated == [process, cancelled_process]


def test_in_allocation_window_receipt_is_immutable_and_controller_scoped(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist one bounded exhausted allocation window without changing wait authority."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 17.0)
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    evidence = license_service.classify_temporary_license_capacity(_OBSERVED_CAPACITY_TEXT)
    assert evidence is not None
    started_at = datetime(2026, 1, 2, tzinfo=UTC)
    summaries = tuple(
        license_service.InAllocationLicenseCheckoutSummary(
            checkout_index=index,
            started_at=started_at + timedelta(seconds=index - 1),
            ended_at=started_at + timedelta(seconds=index),
            started_monotonic_seconds=39.0 + index,
            ended_monotonic_seconds=40.0 + index,
            process_exit_code=0,
            classification=evidence,
            solver_progress_started=False,
        )
        for index in range(1, 11)
    )
    result = license_service.in_allocation_license_window_result(
        config,
        1,
        campaign_run_id="source~transient_drying~window-receipt__0123456789abcdef",
        dispatch_key="1201-4",
        hostname="test-host",
        window_started_at=started_at,
        window_ended_at=started_at + timedelta(seconds=17),
        window_started_monotonic_seconds=40.0,
        window_ended_monotonic_seconds=57.0,
        checkout_summaries=summaries,
        solver_progress_started=False,
        outcome="window_exhausted",
    )

    receipt_path = license_service.record_in_allocation_license_window(
        config,
        1,
        result,
        storage_root=tmp_path / "storage",
    )
    receipt = license_service.load_in_allocation_license_window(
        config,
        1,
        campaign_run_id=result.campaign_run_id,
        dispatch_key="1201-4",
        storage_root=tmp_path / "storage",
    )
    assert receipt_path.name == "1201-4.json"
    assert receipt is not None
    assert receipt["controller_retry_increment"] is True
    assert receipt["checkout_attempt_count"] == 10
    assert receipt["checkout_capacity_failure_count"] == 10
    assert receipt["configured_window_seconds"] == 17.0
    assert len(receipt["recent_checkout_summaries"]) == 8
    assert [summary["checkout_index"] for summary in receipt["recent_checkout_summaries"]] == list(range(3, 11))
    assert all(summary["duration_seconds"] == 1.0 for summary in receipt["recent_checkout_summaries"])
    assert receipt_path.stat().st_size < 20_000
    assert receipt["dispatch_key"] == "1201-4"
    with pytest.raises(FileExistsError, match="already exists"):
        license_service.record_in_allocation_license_window(
            config,
            1,
            result,
            storage_root=tmp_path / "storage",
        )

    sibling_path = license_service.record_in_allocation_license_window(
        config,
        1,
        replace(result, dispatch_key="1201-5"),
        storage_root=tmp_path / "storage",
    )
    assert sibling_path.parent == receipt_path.parent
    assert sibling_path.name == "1201-5.json"
    # A writer killed during publication leaves only an owned atomic temporary, which readers skip; nothing else is tolerated.
    (receipt_path.parent / ".1201-6.json.abc123.tmp").write_text("{", encoding="utf-8")
    assert not any(path.name.endswith(".tmp") and not path.name.startswith(".1201-6") for path in receipt_path.parent.iterdir())
    for key in ("1201-4", "1201-5"):
        loaded = license_service.load_in_allocation_license_window(
            config,
            1,
            campaign_run_id=result.campaign_run_id,
            dispatch_key=key,
            storage_root=tmp_path / "storage",
        )
        assert loaded is not None
        assert loaded["dispatch_key"] == key
    with pytest.raises(ValueError, match="dispatch key"):
        license_service.load_in_allocation_license_window(
            config,
            1,
            campaign_run_id=result.campaign_run_id,
            dispatch_key="1201",
            storage_root=tmp_path / "storage",
        )


def test_status_artifact_recovery_receipt_precedes_and_completes_cleanup(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Persist exact pending evidence before unlink and complete it afterward."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    work_directory = tmp_path / "owned-work"
    work_directory.mkdir()
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=3,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text("1787228251108\nError", encoding="utf-8")
    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=7731,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )
    classification = license_service.classify_temporary_license_capacity(
        _OBSERVED_CAPACITY_TEXT,
    )
    assert artifact is not None
    assert classification is not None
    started_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(seconds=1)

    with pytest.raises(ValueError, match="must be timezone-aware"):
        license_service.record_in_allocation_status_artifact_recovery(
            config,
            1,
            campaign_run_id="source~transient_drying~status-recovery__0123456789abcdef",
            dispatch_key="633014-2",
            checkout_started_at=started_at.replace(tzinfo=None),
            checkout_ended_at=ended_at,
            hostname="synthetic-node",
            artifact=artifact,
            classification=classification,
            cleanup_state="pending",
            storage_root=tmp_path / "storage",
        )

    receipt_path = license_service.record_in_allocation_status_artifact_recovery(
        config,
        1,
        campaign_run_id="source~transient_drying~status-recovery__0123456789abcdef",
        dispatch_key="633014-2",
        checkout_started_at=started_at,
        checkout_ended_at=ended_at,
        hostname="synthetic-node",
        artifact=artifact,
        classification=classification,
        cleanup_state="pending",
        storage_root=tmp_path / "storage",
    )
    pending = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert pending["cleanup_state"] == "pending"
    assert pending["status_state"] == "Error"
    assert pending["status_sha256"] == artifact.content_sha256
    assert status_path.exists()

    stop_service.remove_capacity_checkout_status(artifact)
    license_service.record_in_allocation_status_artifact_recovery(
        config,
        1,
        campaign_run_id="source~transient_drying~status-recovery__0123456789abcdef",
        dispatch_key="633014-2",
        checkout_started_at=started_at,
        checkout_ended_at=ended_at,
        hostname="synthetic-node",
        artifact=artifact,
        classification=classification,
        cleanup_state="complete",
        storage_root=tmp_path / "storage",
    )
    records = license_service.load_in_allocation_status_artifact_recoveries(
        config,
        1,
        campaign_run_id="source~transient_drying~status-recovery__0123456789abcdef",
        dispatch_key="633014-2",
        storage_root=tmp_path / "storage",
    )
    assert not status_path.exists()
    assert len(records) == 1
    assert records[0]["dispatch_key"] == "633014-2"
    assert records[0]["cleanup_state"] == "complete"
    assert records[0]["checkout_index"] == 3
    assert records[0]["solver_progress_started"] is False
    assert records[0]["required_exports_exist"] is False


def _status_recovery_artifact(work_directory: Path, *, checkout_index: int) -> tuple[Any, Any]:
    """Return one synthetic COMSOL status artifact and capacity classification."""
    work_directory.mkdir(parents=True)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=checkout_index,
    )
    (work_directory / stop_service.STOP_STATUS_FILENAME).write_text(
        "1787228251108\nError",
        encoding="utf-8",
    )
    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=7731 + checkout_index,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )
    classification = license_service.classify_temporary_license_capacity(
        _OBSERVED_CAPACITY_TEXT,
    )
    assert artifact is not None
    assert classification is not None
    return artifact, classification


def test_status_recovery_loader_admits_atomic_publication_and_lifecycle(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read no partial receipt, then preserve pending-to-complete publication."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    run_id = "source~transient_drying~concurrent_status_recovery__0123456789abcdef"
    dispatch_key = "638283-1"
    artifact, classification = _status_recovery_artifact(
        tmp_path / "status-work",
        checkout_index=3,
    )
    started_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(seconds=1)
    staged: Queue[Path] = Queue()
    releases: Queue[None] = Queue()
    original_fsync = common.serialization._fsync_file

    def gated_fsync(temporary_path: Path) -> None:
        original_fsync(temporary_path)
        staged.put(temporary_path)
        releases.get(timeout=10.0)

    monkeypatch.setattr(common.serialization, "_fsync_file", gated_fsync)
    publication = {
        "campaign_run_id": run_id,
        "dispatch_key": dispatch_key,
        "checkout_started_at": started_at,
        "checkout_ended_at": ended_at,
        "hostname": "synthetic-node",
        "artifact": artifact,
        "classification": classification,
        "storage_root": storage,
    }
    expected = (
        license_service.in_allocation_status_recovery_directory(
            run_id,
            config.batch_storage_name,
            config.case_id(1),
            dispatch_key,
            storage_root=storage,
        )
        / "checkout_0003.json"
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_future = executor.submit(
            license_service.record_in_allocation_status_artifact_recovery,
            config,
            1,
            **publication,
            cleanup_state="pending",
        )
        try:
            pending_temporary = staged.get(timeout=10.0)
            assert pending_temporary.is_file()
            assert common.serialization.atomic_write_temporary_destination(pending_temporary) == expected
            assert not expected.exists()
            assert (
                license_service.load_in_allocation_status_artifact_recoveries(
                    config,
                    1,
                    campaign_run_id=run_id,
                    dispatch_key=dispatch_key,
                    storage_root=storage,
                )
                == ()
            )
        finally:
            releases.put(None)
        assert pending_future.result(timeout=10.0) == expected

        pending_records = license_service.load_in_allocation_status_artifact_recoveries(
            config,
            1,
            campaign_run_id=run_id,
            dispatch_key=dispatch_key,
            storage_root=storage,
        )
        assert [record["cleanup_state"] for record in pending_records] == ["pending"]
        stop_service.remove_capacity_checkout_status(artifact)

        complete_future = executor.submit(
            license_service.record_in_allocation_status_artifact_recovery,
            config,
            1,
            **publication,
            cleanup_state="complete",
        )
        try:
            complete_temporary = staged.get(timeout=10.0)
            assert complete_temporary.is_file()
            assert common.serialization.atomic_write_temporary_destination(complete_temporary) == expected
            during_completion = license_service.load_in_allocation_status_artifact_recoveries(
                config,
                1,
                campaign_run_id=run_id,
                dispatch_key=dispatch_key,
                storage_root=storage,
            )
            assert [record["cleanup_state"] for record in during_completion] == ["pending"]
        finally:
            releases.put(None)
        assert complete_future.result(timeout=10.0) == expected

    complete_records = license_service.load_in_allocation_status_artifact_recoveries(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key=dispatch_key,
        storage_root=storage,
    )
    assert [record["cleanup_state"] for record in complete_records] == ["complete"]
    assert complete_records[0]["checkout_index"] == 3


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("unexpected_temporary", "unsafe entry"),
        ("atomic_temporary_symlink", "unsafe entry"),
        ("malformed_final", "Could not load"),
        ("conflicting_checkout", "filename is malformed"),
    ],
)
def test_status_recovery_directory_remains_fail_closed(
    mutation: str,
    error_match: str,
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject every non-owned, unsafe, malformed, or identity-conflicting entry."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    run_id = f"source~transient_drying~fail_closed_{mutation}__0123456789abcdef"
    dispatch_key = "638284-1"
    artifact, classification = _status_recovery_artifact(
        tmp_path / "status-work",
        checkout_index=3,
    )
    started_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    receipt = license_service.record_in_allocation_status_artifact_recovery(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key=dispatch_key,
        checkout_started_at=started_at,
        checkout_ended_at=started_at + timedelta(seconds=1),
        hostname="synthetic-node",
        artifact=artifact,
        classification=classification,
        cleanup_state="pending",
        storage_root=storage,
    )
    if mutation == "unexpected_temporary":
        (receipt.parent / ".checkout_0003.json.foreign.extra.tmp").write_text(
            "foreign",
            encoding="utf-8",
        )
    elif mutation == "atomic_temporary_symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (receipt.parent / ".checkout_0004.json.abcdefgh.tmp").symlink_to(outside)
    elif mutation == "malformed_final":
        receipt.write_text("{", encoding="utf-8")
    else:
        conflicting = json.loads(receipt.read_text(encoding="utf-8"))
        conflicting["checkout_index"] = 4
        common.serialization.atomic_write_json(receipt, conflicting)

    with pytest.raises(ValueError, match=error_match):
        license_service.load_in_allocation_status_artifact_recoveries(
            config,
            1,
            campaign_run_id=run_id,
            dispatch_key=dispatch_key,
            storage_root=storage,
        )


def test_concurrent_status_recovery_directories_remain_independent(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconcile two actively publishing case/job directories independently."""
    config_path, _template = generation_config_factory(
        natural_count=2,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    run_id = "source~transient_drying~parallel_status_recovery__0123456789abcdef"
    started_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    classification = license_service.classify_temporary_license_capacity(
        _OBSERVED_CAPACITY_TEXT,
    )
    assert classification is not None
    publications: list[tuple[int, str, Any]] = []
    for case_index, dispatch_key in ((1, "638285-1"), (2, "638286-2")):
        artifact, _classification = _status_recovery_artifact(
            tmp_path / f"status-work-{case_index}",
            checkout_index=1,
        )
        publications.append((case_index, dispatch_key, artifact))

    staged: Queue[Path] = Queue()
    releases: Queue[None] = Queue()
    original_fsync = common.serialization._fsync_file

    def gated_fsync(temporary_path: Path) -> None:
        original_fsync(temporary_path)
        staged.put(temporary_path)
        releases.get(timeout=10.0)

    monkeypatch.setattr(common.serialization, "_fsync_file", gated_fsync)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                license_service.record_in_allocation_status_artifact_recovery,
                config,
                case_index,
                campaign_run_id=run_id,
                dispatch_key=dispatch_key,
                checkout_started_at=started_at,
                checkout_ended_at=started_at + timedelta(seconds=1),
                hostname=f"synthetic-node-{case_index}",
                artifact=artifact,
                classification=classification,
                cleanup_state="pending",
                storage_root=storage,
            )
            for case_index, dispatch_key, artifact in publications
        ]
        try:
            temporary_paths = {staged.get(timeout=10.0), staged.get(timeout=10.0)}
            expected_destinations = {
                license_service.in_allocation_status_recovery_directory(
                    run_id,
                    config.batch_storage_name,
                    config.case_id(case_index),
                    dispatch_key,
                    storage_root=storage,
                )
                / "checkout_0001.json"
                for case_index, dispatch_key, _artifact in publications
            }
            assert {common.serialization.atomic_write_temporary_destination(path) for path in temporary_paths} == expected_destinations
            for case_index, dispatch_key, _artifact in publications:
                assert (
                    license_service.load_in_allocation_status_artifact_recoveries(
                        config,
                        case_index,
                        campaign_run_id=run_id,
                        dispatch_key=dispatch_key,
                        storage_root=storage,
                    )
                    == ()
                )
        finally:
            releases.put(None)
            releases.put(None)
        assert {future.result(timeout=10.0) for future in futures} == expected_destinations

    for case_index, dispatch_key, _artifact in publications:
        records = license_service.load_in_allocation_status_artifact_recoveries(
            config,
            case_index,
            campaign_run_id=run_id,
            dispatch_key=dispatch_key,
            storage_root=storage,
        )
        assert len(records) == 1
        assert records[0]["case_id"] == config.case_id(case_index)
        assert records[0]["dispatch_key"] == dispatch_key


def test_temporary_license_wait_loader_admits_atomic_update(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore only the exact mutable wait receipt's in-flight atomic sibling."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    run_id = "source~transient_drying~concurrent_license_wait__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "638287", 1)
    staged: Queue[Path] = Queue()
    releases: Queue[None] = Queue()
    original_fsync = common.serialization._fsync_file

    def gated_fsync(temporary_path: Path) -> None:
        original_fsync(temporary_path)
        staged.put(temporary_path)
        releases.get(timeout=10.0)

    monkeypatch.setattr(common.serialization, "_fsync_file", gated_fsync)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            license_service.record_temporary_license_wait,
            config,
            1,
            _capacity_error(tmp_path / "license-work"),
            storage_root=storage,
        )
        try:
            temporary_path = staged.get(timeout=10.0)
            expected = (
                license_service.temporary_license_wait_directory(
                    run_id,
                    config.batch_storage_name,
                    config.case_id(1),
                    storage_root=storage,
                )
                / "wait.json"
            )
            assert temporary_path.is_file()
            assert common.serialization.atomic_write_temporary_destination(temporary_path) == expected
            assert (
                license_service.load_temporary_license_wait(
                    config,
                    1,
                    campaign_run_id=run_id,
                    storage_root=storage,
                )
                is None
            )
        finally:
            releases.put(None)
        assert future.result(timeout=10.0) == expected

    wait = license_service.load_temporary_license_wait(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert wait is not None
    assert wait["latest_dispatch_key"] == "638287-1"
    assert wait["retry_count"] == 1


def test_license_retry_backoff_is_exponential_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Increase a test-owned retry delay without exceeding its cap."""
    monkeypatch.setattr(license_service, "INITIAL_BACKOFF_SECONDS", 7.0)
    monkeypatch.setattr(license_service, "MAXIMUM_BACKOFF_SECONDS", 11.0)
    policy = {"enabled": True, "maximum_wait_seconds": 50.0}
    cumulative = 0.0
    delays: list[float] = []
    for attempt_index in range(1, 7):
        delay = license_service.bounded_retry_delay_seconds(
            policy,
            attempt_index=attempt_index,
            cumulative_wait_seconds=cumulative,
        )
        delays.append(delay)
        cumulative += delay

    assert delays == [7.0, 11.0, 11.0, 11.0, 10.0, 0.0]
    assert cumulative == 50.0


def test_unbounded_license_retry_wait_never_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interpret a null maximum wait as indefinitely controller-retryable."""
    monkeypatch.setattr(license_service, "INITIAL_BACKOFF_SECONDS", 3.0)
    monkeypatch.setattr(license_service, "MAXIMUM_BACKOFF_SECONDS", 8.0)
    policy = {"enabled": True, "maximum_wait_seconds": None}

    assert (
        license_service.bounded_retry_delay_seconds(
            policy,
            attempt_index=1,
            cumulative_wait_seconds=0.0,
        )
        == 3.0
    )
    assert (
        license_service.bounded_retry_delay_seconds(
            policy,
            attempt_index=100,
            cumulative_wait_seconds=10_000_000.0,
        )
        == 8.0
    )


@pytest.mark.integration
def test_zero_exit_license_attempt_releases_scratch_and_later_succeeds(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat zero-exit capacity as retryable, then publish the same case normally."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 0.5)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "701", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")
    monkeypatch.setenv("FAKE_COMSOL_CAPACITY_STATUS_STATE", "Error")

    blocked = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=work,
    )

    assert blocked.status == "license_blocked"
    assert blocked.message == "in_allocation_license_window_exhausted"
    assert blocked.work_directory is None
    assert not generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )
    wait = license_service.load_temporary_license_wait(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert wait is not None
    assert wait["latest_dispatch_key"] == "701-1"
    assert wait["recent_dispatch_keys"] == ["701-1"]
    assert wait["retry_count"] == 1
    assert wait["delay_before_next_attempt_seconds"] == license_service.INITIAL_BACKOFF_SECONDS
    assert wait["retry_budget_remaining"] is True
    assert wait["feature"] in {"Brinkman Equations (br)", "COMSOL license acquisition"}
    assert wait["error_code"] == ("-4,132" if wait["feature"] == "Brinkman Equations (br)" else None)
    assert wait["solver_progress_started"] is False
    assert wait["expected_exports_exist"] is False
    assert (
        "Licensed number of users already reached" in wait["raw_excerpt"]
        if wait["feature"] == "Brinkman Equations (br)"
        else wait["raw_excerpt"] == "controller_owned_in_allocation_license_window_deadline"
    )
    window = license_service.load_in_allocation_license_window(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key="701-1",
        storage_root=storage,
    )
    assert window is not None
    assert window["outcome"] == "window_exhausted"
    assert window["reason"] == "in_allocation_license_window_exhausted"
    assert window["checkout_attempt_count"] >= 1
    assert window["checkout_capacity_failure_count"] == window["checkout_attempt_count"]
    assert "Brinkman Equations (br)" in window["observed_features"]
    assert "-4,132" in window["observed_error_codes"]
    assert any(summary["process_exit_code"] == 0 for summary in window["recent_checkout_summaries"])
    status_recoveries = license_service.load_in_allocation_status_artifact_recoveries(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key="701-1",
        storage_root=storage,
    )
    assert status_recoveries
    assert len(status_recoveries) <= window["checkout_capacity_failure_count"]
    assert all(record["cleanup_state"] == "complete" for record in status_recoveries)
    assert not (blocked.processed_directory / "_SUCCESS").exists()
    assert not (blocked.processed_directory / "case.h5").exists()
    assert attempt_service.latest_case_attempt(config, 1, storage_root=storage) is None

    monkeypatch.delenv("FAKE_COMSOL_MODE")
    monkeypatch.delenv("FAKE_COMSOL_CAPACITY_STATUS_STATE")
    _dispatch_environment(monkeypatch, "702", 2)
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=work,
    )
    assert outcome.status == "completed"
    timing = json.loads((outcome.processed_directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert timing["license_wait_seconds"] == license_service.INITIAL_BACKOFF_SECONDS
    assert attempt_service.latest_case_attempt(config, 1, storage_root=storage) is None
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
    assert (
        license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        is not None
    )


@pytest.mark.integration
def test_silent_controller_deadline_becomes_license_blocked_without_failure(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route the controller-owned TERM status through existing operational retry."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 0.05)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "703", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "silent_startup")

    blocked = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=tmp_path / "work",
    )

    assert blocked.status == "license_blocked"
    assert blocked.message == "in_allocation_license_window_exhausted"
    assert not generation.runtime.case_failure_is_recorded(config, 1, storage_root=storage)
    assert attempt_service.latest_case_attempt(config, 1, storage_root=storage) is None
    assert not generation.runtime.completed_case_is_valid(config, 1, storage_root=storage)
    wait = license_service.load_temporary_license_wait(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert wait is not None
    assert wait["comsol_exit_code"] == -signal.SIGTERM
    assert wait["feature"] == "COMSOL license acquisition"
    assert wait["matched_signatures"] == ["controller_owned_in_allocation_license_window_deadline"]
    assert wait["retry_budget_remaining"] is True
    window = license_service.load_in_allocation_license_window(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key="703-1",
        storage_root=storage,
    )
    assert window is not None
    assert window["outcome"] == "window_exhausted"
    assert window["reason"] == "in_allocation_license_window_exhausted"
    assert window["recent_checkout_summaries"][-1]["process_exit_code"] == -signal.SIGTERM


@pytest.mark.integration
def test_cancellation_between_capacity_checkouts_remains_cancelled(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give cancellation priority after a capacity checkout and before retry."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 1.0)
    monkeypatch.setattr(license_service, "IN_ALLOCATION_PAUSE_SECONDS", 0.1)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "705", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")
    pauses: list[float] = []
    real_sleep = batch_runtime.time.sleep

    def cancel_during_pause(seconds: float) -> None:
        if seconds == 0.1:
            pauses.append(seconds)
            generation.runtime.request_runtime_cancellation()
            return
        real_sleep(seconds)

    monkeypatch.setattr(batch_runtime.time, "sleep", cancel_during_pause)
    generation.runtime.reset_runtime_cancellation()
    try:
        with pytest.raises(generation.runtime.CaseInterruptedError):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                scheduler_kind="slurm",
                storage_root=storage,
                work_root=tmp_path / "work",
            )
    finally:
        generation.runtime.reset_runtime_cancellation()

    assert pauses == [0.1]
    assert (
        license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        is None
    )
    attempt = attempt_service.latest_case_attempt(config, 1, storage_root=storage)
    assert attempt is not None
    assert attempt.payload["case_state"] == "cancelled"


@pytest.mark.integration
def test_unowned_sigterm_remains_a_genuine_solver_failure(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reinterpret an arbitrary process status minus fifteen as capacity."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 1.0)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "704", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "external_sigterm")

    with pytest.raises(generation.runtime.CaseExecutionError) as caught:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            scheduler_kind="slurm",
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    assert caught.value.exit_code == -signal.SIGTERM
    assert generation.runtime.case_failure_is_recorded(config, 1, storage_root=storage)
    assert (
        license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        is None
    )


@pytest.mark.integration
@pytest.mark.parametrize("entry_path", ["authored_rule", "no_retry_flag"])
def test_disabled_license_protection_fails_the_case_without_wait_or_window(
    entry_path: str,
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the protection off, a capacity failure is an ordinary solver failure: no wait receipt, no allocation window."""
    config_path, _template = generation_config_factory(executable=fake_comsol, license_retry_enabled=entry_path != "authored_rule")
    overrides = {"no_retry": True} if entry_path == "no_retry_flag" else None
    config = generation.cases.config.load_campaign_config(config_path, execution_overrides=overrides).batch("transient_drying__lentil__test")
    assert config.execution_values["runtime"]["license_wait"]["enabled"] is False
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "706", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")

    with pytest.raises(license_service.TemporaryLicenseCapacityError):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            scheduler_kind="slurm",
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    attempt = attempt_service.latest_case_attempt(config, 1, storage_root=storage)
    assert attempt is not None
    assert (attempt.payload["case_state"], attempt.payload["failure_stage"]) == ("failed", "solver")
    assert license_service.load_temporary_license_wait(config, 1, campaign_run_id=run_id, storage_root=storage) is None
    assert license_service.load_in_allocation_license_window(config, 1, campaign_run_id=run_id, dispatch_key="706-1", storage_root=storage) is None


@pytest.mark.integration
def test_capacity_failures_then_solver_start_keep_one_allocation_and_process(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry twice, then keep the third process alive for the scientific solve."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 1.0)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "711", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity_twice_then_success")
    monkeypatch.setenv("FAKE_COMSOL_CAPACITY_STATUS_STATE", "Error")
    monkeypatch.setenv("FAKE_COMSOL_DELAY", "0.5")

    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=tmp_path / "work",
    )

    assert outcome.status == "completed"
    receipt = license_service.load_in_allocation_license_window(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key="711-1",
        storage_root=storage,
    )
    assert receipt is not None
    assert receipt["outcome"] == "solver_progress_started"
    assert receipt["checkout_attempt_count"] == 3
    assert receipt["checkout_capacity_failure_count"] == 2
    assert [summary["process_exit_code"] for summary in receipt["recent_checkout_summaries"]] == [0, 0, None]
    assert receipt["recent_checkout_summaries"][-1]["solver_progress_started"] is True
    status_recoveries = license_service.load_in_allocation_status_artifact_recoveries(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key="711-1",
        storage_root=storage,
    )
    assert [record["checkout_index"] for record in status_recoveries] == [1, 2]
    assert all(record["cleanup_state"] == "complete" for record in status_recoveries)
    assert (
        license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        is None
    )
    timing = json.loads((outcome.processed_directory / "evidence" / "timing.json").read_text(encoding="utf-8"))
    assert timing["in_allocation_checkout_attempt_count"] == 3
    assert timing["status_artifact_recovery_count"] == 2
    assert timing["in_allocation_capacity_pause_seconds"] == pytest.approx(0.02)
    assert timing["comsol_process_seconds"] < timing["complete_execution_s"]
    assert generation.runtime.completed_case_is_valid(config, 1, storage_root=storage)


@pytest.mark.integration
def test_two_workers_search_for_capacity_without_a_global_launch_gate(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow two independent allocations to run checkout processes concurrently."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 0.6)
    config_path, _template = generation_config_factory(executable=fake_comsol, natural_count=2)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 2, storage)
    prepared = tuple(
        generation.runtime.prepare_case_work_directory(
            config,
            case_index,
            storage_root=storage,
            work_root=tmp_path / f"work-{case_index}",
        )
        for case_index in (1, 2)
    )
    tracker = tmp_path / "checkout-tracker.json"
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity_delayed")
    monkeypatch.setenv("FAKE_COMSOL_TRACKER", str(tracker))
    monkeypatch.setenv("FAKE_COMSOL_EXPECT_STARTS", "2")

    def execute(item: Any) -> BaseException:
        try:
            generation.runtime.execute_prepared_case(
                config,
                item,
                cores_per_case=1,
                worker_slot=int(item.bundle.case_payload["case_index"]) - 1,
                scheduler_kind="slurm",
            )
        except BaseException as error:  # noqa: BLE001 -- concurrent exception is the asserted outcome
            return error
        message = "Synthetic capacity-only worker unexpectedly completed."
        return AssertionError(message)

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = tuple(executor.map(execute, prepared))

    assert all(isinstance(error, license_service.TemporaryLicenseCapacityError) for error in errors)
    tracker_payload = json.loads(tracker.read_text(encoding="utf-8"))
    assert tracker_payload["maximum"] == 2
    assert tracker_payload["active"] == 0
    assert tracker_payload["starts"] >= 4
    assert not any(
        generation.runtime.case_failure_is_recorded(
            config,
            case_index,
            storage_root=storage,
            execution_run_id=_source_run_id(config),
        )
        for case_index in (1, 2)
    )
    for item in prepared:
        generation.runtime.workspace.cleanup_case_workspace(
            item.work_directory,
            allowed_root=item.work_root,
            storage_root=storage.resolve(),
            expected_run_id=item.workspace_run_id,
            expected_case_id=item.bundle.case_id,
        )


@pytest.mark.integration
def test_valid_success_overrides_an_earlier_license_warning(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a complete validated result successful despite an earlier warning."""
    monkeypatch.setattr(license_service, "IN_ALLOCATION_WINDOW_SECONDS", 2.0)
    config_path, _template = generation_config_factory(executable=fake_comsol)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "751", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "success_with_license_warning")

    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=tmp_path / "work",
    )

    assert outcome.status == "completed"
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
    assert not license_service.load_temporary_license_wait(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )


def test_persisted_retry_budget_exhaustion_is_terminal_evidence(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct cumulative wait receipts and close a test-owned budget."""
    config_path, _template = generation_config_factory(license_maximum_wait_seconds=12.0)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    policy = config.execution_values["runtime"]["license_wait"]
    run_id = _source_run_id(config)
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    error = _capacity_error(tmp_path)
    submission_index = 1
    observed: list[dict[str, Any]] = []
    for _retry in range(100):
        _dispatch_environment(monkeypatch, "801", submission_index)
        license_service.record_temporary_license_wait(
            config,
            1,
            error,
            storage_root=storage,
        )
        wait = license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        assert wait is not None
        observed.append(dict(wait))
        if not wait["retry_budget_remaining"]:
            break
        submission_index += 1
    else:
        pytest.fail("Synthetic retry policy did not exhaust within 100 attempts")

    assert observed[-2]["cumulative_wait_seconds"] == policy["maximum_wait_seconds"]
    assert observed[-2]["retry_budget_remaining"] is True
    assert observed[-1]["delay_before_next_attempt_seconds"] == 0.0
    assert observed[-1]["cumulative_wait_seconds"] == policy["maximum_wait_seconds"]
    assert observed[-1]["retry_budget_remaining"] is False
    assert observed[-1]["next_retry_at"] is None
    assert wait["recent_dispatch_keys"][-1] == f"801-{submission_index}"
    latest = license_service.latest_wait_for_dispatch(
        config,
        1,
        campaign_run_id=run_id,
        dispatch_key=wait["latest_dispatch_key"],
        storage_root=storage,
    )
    assert latest is not None
    assert latest["first_blocked_at"] == observed[0]["first_blocked_at"]
    assert (
        license_service.latest_wait_for_dispatch(
            config,
            1,
            campaign_run_id=run_id,
            dispatch_key="801-1",
            storage_root=storage,
        )
        is None
    )


def test_wait_uniqueness_is_per_dispatch_key_within_one_allocation(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept two dispatches of one allocation, but never the same dispatch twice."""
    config_path, _template = generation_config_factory()
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    run_id = _source_run_id(config)
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    error = _capacity_error(tmp_path)

    _dispatch_environment(monkeypatch, "900", 4, step_id="2")
    license_service.record_temporary_license_wait(config, 1, error, storage_root=storage)
    _dispatch_environment(monkeypatch, "900", 7, step_id="3")
    license_service.record_temporary_license_wait(config, 1, error, storage_root=storage)

    wait = license_service.load_temporary_license_wait(config, 1, campaign_run_id=run_id, storage_root=storage)
    assert wait is not None
    assert wait["retry_count"] == 2
    assert wait["recent_dispatch_keys"] == ["900-4", "900-7"]
    assert wait["latest_dispatch_key"] == "900-7"
    with pytest.raises(FileExistsError, match="900-7"):
        license_service.record_temporary_license_wait(config, 1, error, storage_root=storage)

    monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID")
    with pytest.raises(ValueError, match="campaign_run_id"):
        license_service.record_temporary_license_wait(config, 1, error, storage_root=storage)


@pytest.mark.integration
def test_license_retry_cleanup_failure_becomes_terminal(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not auto-retry a capacity attempt whose marked scratch did not clean."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__test",
    )
    storage = tmp_path / "storage"
    _prepare_current_source_inputs(config, 1, storage)
    run_id = _source_run_id(config)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    _dispatch_environment(monkeypatch, "991", 1)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")

    def fail_cleanup(*_args: Any, **_kwargs: Any) -> int:
        message = "synthetic retry cleanup failure"
        raise OSError(message)

    monkeypatch.setattr(
        generation.runtime.batch,
        "_cleanup_case_attempt",
        fail_cleanup,
    )
    with pytest.raises(
        generation.runtime.CaseCleanupError,
        match="synthetic retry cleanup failure",
    ):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            scheduler_kind="slurm",
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    assert generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )
    receipt = license_service.load_temporary_license_wait(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert receipt is not None
