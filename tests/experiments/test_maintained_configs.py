"""Admit every maintained learning configuration and Dataset plan through its loader.

Responsibilities:
  - Preflight each YAML under configs/03_learning with a synthetic train Dataset name
  - Validate each Optuna study against the study contract (shared cadence, pruner, search-space candidates)
  - Load each Dataset plan under configs/02_datasets and derive its Generation campaign
  - Resolve the documented matched A0/B/A1 command for every transient experiment
  - Pass the B target that matched experiments leave open, as the documented command does

Design principles:
  - Maintained configurations are mutable scientific inputs: only admission is checked,
    never their current values

This module does NOT:
  - Read or write scientific storage, resolve real Dataset IDs, or start training
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from support import optuna_trials

from src import common
from src.datasets import dataset_plan
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_naming as naming
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as plan_service
from src.experiments.tuning import experiments_tuning_optuna as optuna

if TYPE_CHECKING:
    from collections.abc import Callable

_REPOSITORY = Path(__file__).resolve().parents[2]


def _admission_failures(paths: list[Path], admit: Callable[[Path], object]) -> list[str]:
    """Return one readable failure per configuration that its loader rejects."""
    failures = []
    for path in paths:
        try:
            admit(path)
        except Exception as error:  # noqa: BLE001 -- report every rejected configuration at once
            failures.append(f"{path.relative_to(_REPOSITORY)}: {type(error).__name__}: {error}")
    return failures


def _preflight_with_documented_overrides(path: Path) -> preflight.ConfigPreflight:
    """Preflight one YAML with the train Dataset and, for matched plans, the B target the YAML leaves open."""
    matched = plan_service.is_matched_a1_b_config(loader.load_authored_config(path))
    return preflight.inspect_config(path, train_dataset="synthetic_dataset", train_seconds=600.0 if matched else None)


def _admit_within_the_name_limit(path: Path) -> None:
    """Preflight one YAML, then check that its file name and the longest run name of every stage fit one path component."""
    result = _preflight_with_documented_overrides(path)
    names = [path.stem, result.run_label]
    raw = loader.load_authored_config(path)
    if plan_service.is_matched_a1_b_config(raw):
        requested = loader.apply_training_overrides(raw, train_dataset="synthetic_dataset", train_seconds=600.0)
        plan = plan_service.resolve_matched_a1_b_plan(requested, source=naming.config_source_from_path(path))
        names += [stage["run"]["name"] for stage in plan.stages]
    for name in names:
        common.paths.validate_logical_name(name, label=f"{path.name} name")


def test_every_learning_config_passes_preflight() -> None:
    """Experiment and Optuna YAMLs resolve with the documented overrides and keep their names within one path component."""
    paths = sorted(path for path in (_REPOSITORY / "configs/03_learning").rglob("*.yaml") if path.name != "base.yaml")

    assert paths
    assert _admission_failures(paths, _admit_within_the_name_limit) == []


def test_every_optuna_study_passes_its_study_contract() -> None:
    """Each study's cadence, pruner and every declared search-space candidate validate before any trial exists."""
    paths = sorted((_REPOSITORY / "configs/03_learning").glob("*/optuna/*.yaml"))

    assert paths
    assert (
        _admission_failures(
            paths,
            lambda path: optuna.build_study_signature(
                optuna_trials.with_configured_identities(optuna.load_optuna_study_config(path, train_dataset="synthetic_dataset"))
            ),
        )
        == []
    )


def test_every_dataset_plan_loads_and_derives_its_campaign() -> None:
    """Dataset plans load and project to a Generation campaign without storage access."""
    paths = sorted((_REPOSITORY / "configs/02_datasets").rglob("*.yaml"))

    assert paths
    assert _admission_failures(paths, lambda path: dataset_plan.authored_campaign(dataset_plan.load_dataset_plan(path))) == []


def test_documented_matched_command_resolves_for_every_transient_experiment() -> None:
    """The matched A0/B/A1 overrides shown in docs/05_training.md stay admissible."""
    paths = sorted((_REPOSITORY / "configs/03_learning/transient_drying/experiments").glob("*/*.yaml"))

    assert paths
    assert (
        _admission_failures(
            paths,
            lambda path: preflight.inspect_config(
                path,
                train_dataset="synthetic_dataset",
                epochs=16,
                train_seconds=600.0,
                rollout_length=4,
            ),
        )
        == []
    )
