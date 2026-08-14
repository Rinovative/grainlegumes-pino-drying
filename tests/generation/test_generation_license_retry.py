# ruff: noqa: S101, PLR2004
"""Temporary COMSOL floating-license capacity retry contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src import generation
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
    """Use 60/120/240/300 backoff and stop exactly at the wait budget."""
    policy = {
        "enabled": True,
        "initial_delay_seconds": 60.0,
        "maximum_delay_seconds": 300.0,
        "maximum_wait_seconds": 3600.0,
    }
    cumulative = 0.0
    delays: list[float] = []
    for attempt_index in range(1, 16):
        delay = license_service.bounded_retry_delay_seconds(
            policy,
            attempt_index=attempt_index,
            cumulative_wait_seconds=cumulative,
        )
        delays.append(delay)
        cumulative += delay

    assert delays[:6] == [60.0, 120.0, 240.0, 300.0, 300.0, 300.0]
    assert delays[-2:] == [180.0, 0.0]
    assert cumulative == 3600.0
    assert all(delay >= 0.0 for delay in delays)


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
    attempts = license_service.load_temporary_license_attempts(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert len(attempts) == 1
    assert attempts[0]["slurm_job_id"] == "701"
    assert attempts[0]["delay_before_next_attempt_seconds"] == 60.0
    assert attempts[0]["retry_budget_remaining"] is True
    assert not generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    ).exists()

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
    assert generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
    assert (
        len(
            license_service.load_temporary_license_attempts(
                config,
                1,
                campaign_run_id=run_id,
                storage_root=storage,
            )
        )
        == 1
    )


def test_persisted_retry_budget_exhaustion_is_terminal_evidence(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct cumulative wait from receipts and close the 3600-second budget."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    run_id = "license-budget__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    error = _capacity_error(tmp_path)
    for job in range(801, 816):
        monkeypatch.setenv("SLURM_JOB_ID", str(job))
        license_service.record_temporary_license_capacity_attempt(
            config,
            1,
            error,
            storage_root=tmp_path / "storage",
        )

    attempts = license_service.load_temporary_license_attempts(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=tmp_path / "storage",
    )
    assert len(attempts) == 15
    assert attempts[-2]["cumulative_wait_seconds"] == 3600.0
    assert attempts[-2]["retry_budget_remaining"] is True
    assert attempts[-1]["delay_before_next_attempt_seconds"] == 0.0
    assert attempts[-1]["cumulative_wait_seconds"] == 3600.0
    assert attempts[-1]["retry_budget_remaining"] is False
    assert attempts[-1]["next_eligible_at"] is None


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
    receipt = license_service.load_temporary_license_attempts(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    assert len(receipt) == 1
