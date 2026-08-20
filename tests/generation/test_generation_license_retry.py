# ruff: noqa: S101, PLR2004
"""Temporary COMSOL floating-license capacity retry contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

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

    with pytest.raises(
        license_service.TemporaryLicenseCapacityError,
    ) as caught:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            scheduler_kind="slurm",
            storage_root=storage,
            work_root=work,
        )

    assert caught.value.exit_code == 0
    assert not caught.value.work_directory.exists()
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
