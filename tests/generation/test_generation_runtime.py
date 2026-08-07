# ruff: noqa: S101, PLR2004
"""Fake-COMSOL profile execution, publication, resume, and integrity contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("simulation_profile", ["steady_flow", "transient_drying"])
def test_fake_comsol_profile_publication_resume_and_integrity(
    simulation_profile: str,
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Protect both profiles through one complete publication and resume lifecycle."""
    config_path, template = generation_config_factory(simulation_profile=simulation_profile, executable=fake_comsol)
    config = generation.config.load_generation_config(config_path)
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    template_digest = common.serialization.file_sha256(template)
    incomplete = generation.runtime.processed_case_directory(config, 1, storage_root=storage)
    incomplete.mkdir(parents=True)
    (incomplete / "partial.txt").write_text("interrupted", encoding="utf-8")

    outcome = generation.runtime.run_case(config, 1, cores_per_case=2, storage_root=storage, work_root=work)
    assert outcome.status == "completed"
    assert common.serialization.file_sha256(template) == template_digest
    assert outcome.work_directory is not None
    assert not outcome.work_directory.exists()
    completed = outcome.processed_directory
    assert (completed / "exports" / "airflow.csv").is_file()
    assert (completed / "learning_views" / "steady_flow" / "fields.csv").is_file()
    assert (completed / "exports" / "transient_000.csv").is_file() is (simulation_profile == "transient_drying")
    timing = json.loads((completed / "timing.json").read_text(encoding="utf-8"))
    assert timing["simulation_profile"] == simulation_profile
    assert timing["requested_cores"] == 2
    assert timing["arguments"][1:7] == ["batch", "-inputfile", "model.mph", "-outputfile", "solved.mph", "-np"]
    assert not (completed / "solved.mph").exists()
    state = storage / "01_generation" / ".state" / simulation_profile / config.batch_id
    assert list((state / "quarantine").iterdir())
    generation.runtime.validate_completed_case(config, 1, storage_root=storage)
    assert generation.runtime.run_case(config, 1, cores_per_case=2, storage_root=storage, work_root=work).status == "skipped"
    generation.runtime.finalize_batch(config, storage_root=storage)
    manifest = generation.runtime.validate_terminal_batch(config, storage_root=storage)
    assert manifest["simulation_profile"] == simulation_profile
    assert manifest["available_learning_views"] == list(config.profile.available_learning_views)
    assert manifest["airflow_source"] == config.profile.airflow_source

    exported = completed / "exports" / "airflow.csv"
    exported.write_text("x;y;p;u;v\n0;0;999;0;0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity failure"):
        generation.runtime.validate_completed_case(config, 1, storage_root=storage)


def test_repeated_stationary_airflow_is_canonicalized_and_variation_fails(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect explicit stationarity tolerance and the canonical static airflow view."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        repeated_airflow_times=True,
    )
    config = generation.config.load_generation_config(config_path)
    monkeypatch.setenv("FAKE_COMSOL_REPEAT_AIRFLOW", "1")
    outcome = generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=tmp_path / "ok", work_root=tmp_path / "work-ok")
    canonical = outcome.processed_directory / "learning_views" / "steady_flow" / "fields.csv"
    assert canonical.read_text(encoding="utf-8").splitlines()[0] == "x;y;p;u;v"

    monkeypatch.setenv("FAKE_COMSOL_VARY_AIRFLOW", "1")
    with pytest.raises(generation.runtime.CaseExecutionError, match="varies beyond tolerance"):
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=tmp_path / "bad", work_root=tmp_path / "work-bad")


def test_failure_timeout_export_validation_and_case_lock(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect failed-work retention, timeout cleanup, export admission, and locking."""
    config_path, _template = generation_config_factory(executable=fake_comsol, timeout=0.1)
    config = generation.config.load_generation_config(config_path)
    storage = tmp_path / "storage"
    work = tmp_path / "work"

    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    with pytest.raises(generation.runtime.CaseExecutionError) as failed:
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=storage, work_root=work)
    assert failed.value.work_directory.is_dir()

    monkeypatch.setenv("FAKE_COMSOL_MODE", "timeout")
    with pytest.raises(generation.runtime.CaseExecutionError, match="timeout") as timed_out:
        generation.runtime.run_case(config, 1, cores_per_case=1, storage_root=storage, work_root=work, cleanup_failed=True)
    assert not timed_out.value.work_directory.exists()

    monkeypatch.setenv("FAKE_COMSOL_MODE", "success")
    prepared = generation.case.prepare_case_work_directory(config, 1, storage_root=storage, work_root=work)
    with pytest.raises(FileNotFoundError, match="airflow"):
        generation.runtime.collect_exports(config, prepared)
    (prepared.exports_directory / "airflow.csv").write_text("", encoding="utf-8")
    (prepared.exports_directory / "transient_000.csv").write_text("x;y\n0;0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        generation.runtime.collect_exports(config, prepared)

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
