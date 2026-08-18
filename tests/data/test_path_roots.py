# ruff: noqa: S101
"""Protect the sole storage root and its numbered scientific lifecycle areas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import configs

from src import common, experiments

if TYPE_CHECKING:
    from pathlib import Path


def test_dataset_packages_root_is_canonical_and_preserves_logical_id(tmp_path: Path) -> None:
    """Relocating the lifecycle root must not alter one logical Dataset ID."""
    dataset_id = "steady_flow__lentil__id"
    first_root = common.paths.get_dataset_packages_root(storage_root=tmp_path / "first")
    second_root = common.paths.get_dataset_packages_root(storage_root=tmp_path / "second")

    first_path = common.paths.resolve_dataset_path(dataset_id, dataset_root=first_root)
    second_path = common.paths.resolve_dataset_path(dataset_id, dataset_root=second_root)

    assert first_root == tmp_path / "first/02_datasets/packages"
    assert second_root == tmp_path / "second/02_datasets/packages"
    assert first_path.relative_to(first_root) == second_path.relative_to(second_root)
    assert first_path.parent.name == second_path.parent.name == dataset_id


def test_populated_legacy_dataset_root_requires_explicit_move(tmp_path: Path) -> None:
    """Never read or migrate a populated legacy Dataset root implicitly."""
    legacy = tmp_path / "02_datasets/raw"
    legacy.mkdir(parents=True)
    (legacy / "existing-package").mkdir()

    with pytest.raises(RuntimeError) as caught:
        common.paths.get_dataset_packages_root(storage_root=tmp_path)

    expected = (
        "Legacy dataset packages require a one-time explicit move:\n"
        'mv -- "$STORAGE_ROOT/02_datasets/raw" \\\n'
        '       "$STORAGE_ROOT/02_datasets/packages"'
    )
    assert str(caught.value) == expected


def test_populated_legacy_and_current_dataset_roots_fail_closed(tmp_path: Path) -> None:
    """Never merge or silently select between two populated Dataset roots."""
    legacy = tmp_path / "02_datasets/raw"
    current = tmp_path / "02_datasets/packages"
    legacy.mkdir(parents=True)
    current.mkdir(parents=True)
    (legacy / "legacy-package").mkdir()
    (current / "current-package").mkdir()

    with pytest.raises(RuntimeError, match="Both legacy and current dataset roots contain data"):
        common.paths.get_dataset_packages_root(storage_root=tmp_path)


def test_empty_legacy_dataset_root_is_ignored_narrowly(tmp_path: Path) -> None:
    """An empty ordinary legacy directory does not become a second root."""
    legacy = tmp_path / "02_datasets/raw"
    legacy.mkdir(parents=True)

    assert common.paths.get_dataset_packages_root(storage_root=tmp_path) == tmp_path / "02_datasets/packages"


def test_unsafe_legacy_dataset_root_is_rejected(tmp_path: Path) -> None:
    """A legacy symlink cannot bypass lifecycle-root detection."""
    target = tmp_path / "legacy-target"
    target.mkdir()
    legacy = tmp_path / "02_datasets/raw"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="legacy dataset raw root is not a safe directory"):
        common.paths.get_dataset_packages_root(storage_root=tmp_path)


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
    assert dataset_after.is_relative_to(storage_root / "02_datasets/packages")
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
