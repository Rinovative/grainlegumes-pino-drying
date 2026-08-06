# ruff: noqa: S101
"""Protect the sole storage root and its numbered scientific lifecycle areas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import configs

from src import common, experiments

if TYPE_CHECKING:
    from pathlib import Path


def test_storage_root_derives_numbered_lifecycle_areas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every implicit scientific path descends from the sole storage root."""
    storage_root = tmp_path / "scientific storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    generation_root = storage_root / "01_generation"
    datasets_root = storage_root / "02_datasets"
    experiments_root = storage_root / "03_experiments"

    assert common.paths.get_storage_root() == storage_root
    assert common.paths.get_generation_root() == generation_root
    assert common.paths.get_generation_meta_root() == generation_root / "meta"
    assert common.paths.get_generation_raw_root() == generation_root / "raw"
    assert common.paths.get_generation_processed_root() == generation_root / "processed"
    assert common.paths.get_generation_state_root() == generation_root / ".state"
    assert common.paths.get_datasets_root() == datasets_root
    assert common.paths.get_dataset_metadata_root() == datasets_root / "meta"
    assert common.paths.get_dataset_payload_root() == datasets_root / "raw"
    assert common.paths.get_dataset_state_root() == datasets_root / ".state"
    assert common.paths.get_experiments_root() == experiments_root
    assert common.paths.get_experiment_state_root() == experiments_root / ".state"
    assert common.paths.get_dataset_build_locks_root() == datasets_root / ".state/dataset-builds/locks"
    assert common.paths.get_dataset_build_transactions_root() == datasets_root / ".state/dataset-builds/transactions"
    assert common.paths.get_run_locks_root() == experiments_root / ".state/runs/locks"
    assert common.paths.resolve_queue_log_dir("steady_flow") == experiments_root / "steady_flow/logs/queue"

    assert common.paths.resolve_generated_batch_dir("tiny", stage="raw") == generation_root / "raw/tiny"
    assert common.paths.resolve_generated_batch_dir("tiny", stage="processed") == generation_root / "processed/tiny"
    assert common.paths.resolve_dataset_path("tiny") == datasets_root / "raw/tiny/tiny.pt"
    assert common.paths.resolve_dataset_metadata_dir("tiny") == datasets_root / "meta/tiny"
    assert common.paths.resolve_dataset_build_lock_path("tiny") == datasets_root / ".state/dataset-builds/locks/dataset-tiny.lock"
    assert common.paths.resolve_dataset_build_transaction_path("tiny") == (datasets_root / ".state/dataset-builds/transactions/dataset-tiny.json")
    run_dir = experiments_root / "steady_flow/runs/run"
    run_lock = common.paths.resolve_run_lock_path(run_dir)
    artifact_lock = common.paths.resolve_artifact_lock_path(run_dir / "analysis/id")
    assert run_lock.parent == experiments_root / ".state/runs/locks"
    assert run_lock.name.startswith("run-")
    assert run_lock.suffix == ".lock"
    assert artifact_lock.parent == experiments_root / ".state/runs/locks"
    assert artifact_lock.name.startswith("artifact-")
    assert artifact_lock.suffix == ".lock"
    assert common.paths.resolve_run_output_dir("steady_flow", "run") == run_dir
    assert common.paths.resolve_study_dir("steady_flow", "study") == experiments_root / "steady_flow/studies/study"


def test_default_storage_is_repository_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The portable default is the storage directory beside the repository."""
    project_root = tmp_path / "repository"
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("STORAGE_ROOT", raising=False)

    assert common.paths.get_storage_root() == tmp_path / "storage"
    assert common.paths.get_generation_root() == tmp_path / "storage/01_generation"
    assert common.paths.get_datasets_root() == tmp_path / "storage/02_datasets"
    assert common.paths.get_experiments_root() == tmp_path / "storage/03_experiments"


def test_explicit_storage_argument_is_bounded_to_one_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled explicit root overrides the environment for one derivation."""
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("STORAGE_ROOT", str(environment_root))

    assert common.paths.get_storage_root(storage_root=explicit_root) == explicit_root
    assert common.paths.get_generation_root(storage_root=explicit_root) == explicit_root / "01_generation"
    assert common.paths.get_datasets_root(storage_root=explicit_root) == explicit_root / "02_datasets"
    assert common.paths.get_experiments_root(storage_root=explicit_root) == explicit_root / "03_experiments"
    assert common.paths.get_storage_root() == environment_root


def test_resolved_training_config_records_unified_storage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved run provenance records the root and relevant derived areas."""
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    config = experiments.config.loader.resolve_config(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )

    assert config["paths"] == {
        "project_root": str(common.paths.get_project_root()),
        "storage_root": str(storage_root),
        "dataset_metadata_root": str(storage_root / "02_datasets/meta"),
        "dataset_root": str(storage_root / "02_datasets/raw"),
        "output_root": str(storage_root / "03_experiments"),
    }


def test_output_override_cannot_relocate_dataset_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded output override moves runs but not immutable dataset inputs."""
    storage_root = tmp_path / "storage"
    second_output_root = tmp_path / "bounded outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    config = experiments.config.loader.resolve_config(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )
    dataset_before = common.paths.resolve_dataset_path(
        config["data"]["train_dataset"],
        dataset_root=config["paths"]["dataset_root"],
    )
    run_before = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=config["paths"]["output_root"],
    )

    config["paths"]["output_root"] = str(second_output_root)
    dataset_after = common.paths.resolve_dataset_path(
        config["data"]["train_dataset"],
        dataset_root=config["paths"]["dataset_root"],
    )
    run_after = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=config["paths"]["output_root"],
    )

    assert dataset_before == dataset_after
    assert run_before != run_after
    assert dataset_after.is_relative_to(storage_root / "02_datasets/raw")
    assert run_after.is_relative_to(second_output_root)


_INVALID_LOGICAL_NAMES = (
    "",
    ".",
    "..",
    "../escape",
    "nested/name",
    "nested\\name",
    "/outside/escape",
    " trailing",
)


def test_logical_name_validator_rejects_unsafe_components() -> None:
    """Empty, traversal, separator, absolute, and untrimmed names are rejected."""
    for invalid_name in _INVALID_LOGICAL_NAMES:
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.validate_logical_name(invalid_name, label="logical name")


def test_owned_path_resolvers_apply_logical_name_validation(tmp_path: Path) -> None:
    """Every public resolver rejects traversal at its ownership boundary."""
    invalid_name = "../escape"
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_path(invalid_name, dataset_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_metadata_dir(invalid_name, metadata_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_build_lock_path(invalid_name, storage_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_build_transaction_path(invalid_name, storage_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_generated_batch_dir(invalid_name, stage="raw", storage_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_run_output_dir("steady_flow", invalid_name, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_run_output_dir(invalid_name, "run", output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_optuna_trial_dir("steady_flow", invalid_name, "run", output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_optuna_trial_dir("steady_flow", "study", invalid_name, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_runs_root(invalid_name, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_queue_log_dir(invalid_name, storage_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_ood_analysis_dir(tmp_path / "run", invalid_name)
