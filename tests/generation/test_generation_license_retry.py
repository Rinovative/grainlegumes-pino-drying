# ruff: noqa: S101, PLR2004
"""Temporary COMSOL floating-license capacity retry contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src import generation
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.runtime import generation_runtime_license as license_service

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


def test_in_allocation_window_receipt_is_immutable_and_controller_scoped(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Persist one bounded exhausted allocation window without changing wait authority."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["runtime"]["temporary_license_retry"]["in_allocation_retry"] = {
        "enabled": True,
        "maximum_window_seconds": 17.0,
        "pause_after_capacity_failure_seconds": 3.0,
    }
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    evidence = license_service.classify_temporary_license_capacity(_OBSERVED_CAPACITY_TEXT)
    assert evidence is not None
    started_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
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
        campaign_run_id="window-receipt__0123456789abcdef",
        job_id="1201",
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
        job_id="1201",
        storage_root=tmp_path / "storage",
    )
    assert receipt_path.name == "1201.json"
    assert receipt is not None
    assert receipt["controller_retry_increment"] is True
    assert receipt["checkout_attempt_count"] == 10
    assert receipt["checkout_capacity_failure_count"] == 10
    assert receipt["configured_window_seconds"] == 17.0
    assert len(receipt["recent_checkout_summaries"]) == 8
    assert [summary["checkout_index"] for summary in receipt["recent_checkout_summaries"]] == list(range(3, 11))
    assert all(summary["duration_seconds"] == 1.0 for summary in receipt["recent_checkout_summaries"])
    assert receipt_path.stat().st_size < 20_000
    with pytest.raises(FileExistsError, match="already exists"):
        license_service.record_in_allocation_license_window(
            config,
            1,
            result,
            storage_root=tmp_path / "storage",
        )


def test_license_retry_backoff_is_exponential_and_bounded() -> None:
    """Increase a test-owned retry delay without exceeding its cap."""
    policy = {
        "enabled": True,
        "initial_delay_seconds": 7.0,
        "maximum_delay_seconds": 11.0,
        "maximum_wait_seconds": 50.0,
    }
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


def test_unbounded_license_retry_wait_never_exhausts() -> None:
    """Interpret a null maximum wait as indefinitely controller-retryable."""
    policy = {
        "enabled": True,
        "initial_delay_seconds": 3.0,
        "maximum_delay_seconds": 8.0,
        "maximum_wait_seconds": None,
    }

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
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        scheduler_kind="slurm",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    run_id = "license-retry__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    monkeypatch.setenv("SLURM_JOB_ID", "701")
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity")

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
    retry_policy = config.execution_values["runtime"]["temporary_license_retry"]
    assert wait["latest_job_id"] == "701"
    assert wait["recent_job_ids"] == ["701"]
    assert wait["retry_count"] == 1
    assert wait["delay_before_next_attempt_seconds"] == retry_policy["initial_delay_seconds"]
    assert wait["retry_budget_remaining"] is True
    assert wait["feature"] == "Brinkman Equations (br)"
    assert wait["error_code"] == "-4,132"
    assert wait["solver_progress_started"] is False
    assert wait["expected_exports_exist"] is False
    assert "Licensed number of users already reached" in wait["raw_excerpt"]
    window = license_service.load_in_allocation_license_window(
        config,
        1,
        campaign_run_id=run_id,
        job_id="701",
        storage_root=storage,
    )
    assert window is not None
    assert window["outcome"] == "window_exhausted"
    assert window["reason"] == "in_allocation_license_window_exhausted"
    assert window["checkout_attempt_count"] >= 1
    assert window["checkout_capacity_failure_count"] == window["checkout_attempt_count"]
    assert all(summary["process_exit_code"] == 0 for summary in window["recent_checkout_summaries"])
    assert not (blocked.processed_directory / "_SUCCESS").exists()
    assert not (blocked.processed_directory / "case.h5").exists()
    assert (
        attempt_service.latest_case_attempt(
            config,
            1,
            run_id,
            storage_root=storage,
        )
        is None
    )

    monkeypatch.delenv("FAKE_COMSOL_MODE")
    monkeypatch.setenv("SLURM_JOB_ID", "702")
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        scheduler_kind="slurm",
        storage_root=storage,
        work_root=work,
    )
    assert outcome.status == "completed"
    timing = json.loads((outcome.processed_directory / "timing.json").read_text(encoding="utf-8"))
    assert timing["license_wait_seconds"] == retry_policy["initial_delay_seconds"]
    assert (
        attempt_service.latest_case_attempt(
            config,
            1,
            run_id,
            storage_root=storage,
        )
        is None
    )
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
def test_capacity_failures_then_solver_start_keep_one_allocation_and_process(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry twice, then keep the third process alive for the scientific solve."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        scheduler_kind="slurm",
        in_allocation_maximum_window_seconds=1.0,
        in_allocation_pause_after_capacity_failure_seconds=0.01,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    storage = tmp_path / "storage"
    generation.cases.input_generation.generate_input_cases(config, 1, storage_root=storage)
    run_id = "license-acquired__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    monkeypatch.setenv("SLURM_JOB_ID", "711")
    monkeypatch.setenv("FAKE_COMSOL_MODE", "license_capacity_twice_then_success")
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
        job_id="711",
        storage_root=storage,
    )
    assert receipt is not None
    assert receipt["outcome"] == "solver_progress_started"
    assert receipt["checkout_attempt_count"] == 3
    assert receipt["checkout_capacity_failure_count"] == 2
    assert [summary["process_exit_code"] for summary in receipt["recent_checkout_summaries"]] == [0, 0, None]
    assert receipt["recent_checkout_summaries"][-1]["solver_progress_started"] is True
    assert (
        license_service.load_temporary_license_wait(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        is None
    )
    timing = json.loads((outcome.processed_directory / "timing.json").read_text(encoding="utf-8"))
    assert timing["in_allocation_checkout_attempt_count"] == 3
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
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        scheduler_kind="slurm",
        natural_count=2,
        in_allocation_maximum_window_seconds=0.35,
        in_allocation_pause_after_capacity_failure_seconds=0.01,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    storage = tmp_path / "storage"
    generation.cases.input_generation.generate_input_cases(config, 2, storage_root=storage)
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
    assert not any(generation.runtime.case_failure_is_recorded(config, case_index, storage_root=storage) for case_index in (1, 2))
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
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        scheduler_kind="slurm",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    storage = tmp_path / "storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    run_id = "license-warning__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    monkeypatch.setenv("SLURM_JOB_ID", "751")
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
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        license_initial_delay_seconds=2.0,
        license_maximum_delay_seconds=5.0,
        license_maximum_wait_seconds=12.0,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    run_id = "license-budget__0123456789abcdef"
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    error = _capacity_error(tmp_path)
    next_job = 801
    observed: list[dict[str, Any]] = []
    for _retry in range(100):
        monkeypatch.setenv("SLURM_JOB_ID", str(next_job))
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
        next_job += 1
    else:
        pytest.fail("Synthetic retry policy did not exhaust within 100 attempts")

    assert observed[-2]["cumulative_wait_seconds"] == policy["maximum_wait_seconds"]
    assert observed[-2]["retry_budget_remaining"] is True
    assert observed[-1]["delay_before_next_attempt_seconds"] == 0.0
    assert observed[-1]["cumulative_wait_seconds"] == policy["maximum_wait_seconds"]
    assert observed[-1]["retry_budget_remaining"] is False
    assert observed[-1]["next_retry_at"] is None
    latest = license_service.latest_wait_for_job(
        config,
        1,
        campaign_run_id=run_id,
        job_id=wait["latest_job_id"],
        storage_root=storage,
    )
    assert latest is not None
    assert latest["first_blocked_at"] == observed[0]["first_blocked_at"]


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
        scheduler_kind="slurm",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    storage = tmp_path / "storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        1,
        storage_root=storage,
    )
    run_id = "license-cleanup__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    monkeypatch.setenv("SLURM_JOB_ID", "991")
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
