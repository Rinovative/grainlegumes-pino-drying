"""Protect config sources, model labels, config-name validation, pair keys, trial leaves and run names.

Responsibilities:
  - Check the (task, family, name) source derived from canonical, Optuna and external config paths
  - Check model labels, the config-name rule, pair keys, trial leaves and run names, including the epoch-budget tokens of seed groups and studies
  - Check that revisions and the derived layout reach the resolved config as a pure function of its inputs
  - Check that the live layout identity is recomputed only when validation requests it

Design principles:
  - Names are pure functions of the resolved config, its source and its revision

This module does NOT:
  - Allocate runs, check occupancy or exercise the Optuna lifecycle
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src import common, learning
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_naming as naming

if TYPE_CHECKING:
    from pathlib import Path

_TREE = "configs/03_learning"


def _fd_config() -> dict[str, Any]:
    """Return a resolved physics-enabled config using central finite differences."""
    return configs.resolve(configs.direct_config(physics_enabled=True))


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            f"{_TREE}/steady_flow/experiments/architecture_comparison/fno_m4x5_h8_l2.yaml",
            ("steady_flow", "architecture_comparison", "fno_m4x5_h8_l2"),
        ),
        (f"{_TREE}/transient_drying/optuna/uno_a0_search.yaml", ("transient_drying", "optuna", "uno_a0_search")),
    ],
)
def test_canonical_paths_name_task_family_and_stem(tmp_path: Path, relative: str, expected: tuple[str, str, str]) -> None:
    """The canonical tree alone decides task, family and name; the file need not exist."""
    assert naming.config_source_from_path(tmp_path / relative) == naming.ConfigSource(*expected)


def test_external_yaml_takes_its_task_from_the_yaml(tmp_path: Path) -> None:
    """A YAML outside the canonical tree is the explicit external family, labelled by its stem."""
    experiment = configs.write_yaml(tmp_path / "request.yaml", {"task": "steady_flow"})
    study = configs.write_yaml(tmp_path / "study.yaml", {"experiment": {"task": "transient_drying"}, "search_space": {}})
    taskless = configs.write_yaml(tmp_path / "taskless.yaml", {"model": {}})

    assert naming.config_source_from_path(experiment) == naming.ConfigSource("steady_flow", "external", "request")
    assert naming.config_source_from_path(study) == naming.ConfigSource("transient_drying", "external", "study")
    with pytest.raises(loader.ConfigError, match="must declare a task"):
        naming.config_source_from_path(taskless)
    with pytest.raises(FileNotFoundError):
        naming.config_source_from_path(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "relative",
    [
        f"{_TREE}/steady_flow/base.yaml",
        f"{_TREE}/steady_flow/experiments/plain.yaml",
        f"{_TREE}/steady_flow/experiments/family/nested/plain.yaml",
        f"{_TREE}/steady_flow/experiments/optuna/plain.yaml",
        f"{_TREE}/steady_flow/experiments/external/plain.yaml",
        f"{_TREE}/steady_flow/experiments/family/plain.yml",
        f"{_TREE}/steady_flow/optuna/nested/study.yaml",
        f"{_TREE}/steady_flow/optuna/study.json",
        f"{_TREE}/steady_flow/shared.yaml",
        f"{_TREE}/{_TREE}/steady_flow/experiments/family/plain.yaml",
    ],
)
def test_other_shapes_below_the_canonical_tree_are_rejected(tmp_path: Path, relative: str) -> None:
    """Base files, missing or nested families, reserved families and repeated trees are not executable configs."""
    with pytest.raises(loader.ConfigError, match=r"must be configs/03_learning|ambiguous"):
        naming.config_source_from_path(tmp_path / relative)


def test_model_label_marks_physics_and_derivative_family() -> None:
    """Plain models use the core label; physics adds the pi- prefix and the fd or spec suffix."""
    plain = configs.resolve(configs.direct_config())
    assert naming.model_label(plain) == "fno_m4x5_h8_l2"
    assert naming.model_family(plain) == "fno"

    finite_difference = _fd_config()
    assert naming.model_label(finite_difference) == "pi-fno_m4x5_h8_l2__fd"
    assert naming.model_family(finite_difference) == "pi-fno"

    raw = configs.direct_config(model_kind="uno", physics_enabled=True)
    raw["loss"]["physics"]["derivatives"] = {"kind": "spectral"}
    spectral = configs.resolve(raw)
    assert naming.model_label(spectral).startswith("pi-uno_m4x5_h8_l5_s")
    assert naming.model_label(spectral).endswith("__spec")


def test_model_label_excludes_switches_that_the_display_label_keeps() -> None:
    """The core label ignores non-default model switches, which only the display label shows."""
    raw = configs.direct_config()
    raw["model"]["params"]["channel_mlp_skip"] = "linear"
    config = configs.resolve(raw)

    assert naming.model_label(config) == "fno_m4x5_h8_l2"
    assert learning.models.naming.architecture_label(config) == "fno_m4x5_h8_l2__channel-mlp-skip-linear"


@pytest.mark.parametrize("name", ["fno_m4x5_h8_l2", "fno_m4x5_h8_l2__lr-low", "fno_m4x5_h8_l2__a__b"])
def test_config_name_may_equal_or_extend_the_model_label(name: str) -> None:
    """The model label, optionally followed by a double-underscore variant, names a config."""
    config = configs.resolve(configs.direct_config())

    naming.validate_config_name(naming.ConfigSource("steady_flow", "architecture_comparison", name), config)


@pytest.mark.parametrize("name", ["fno_m4x5_h8_l2_lr-low", "fno_m4x5_h8_l3", "pi-fno_m4x5_h8_l2__fd", "lr-low__fno_m4x5_h8_l2", "request"])
def test_config_name_must_start_with_the_model_label(name: str) -> None:
    """A name that does not equal or extend the label is rejected, naming the expected prefix."""
    config = configs.resolve(configs.direct_config())

    with pytest.raises(loader.ConfigError, match=r"fno_m4x5_h8_l2__"):
        naming.validate_config_name(naming.ConfigSource("steady_flow", "architecture_comparison", name), config)


def test_physics_configs_need_their_physics_label_and_studies_are_exempt() -> None:
    """A physics-enabled config must be named pi-...; an Optuna study may carry any name."""
    config = _fd_config()

    naming.validate_config_name(naming.ConfigSource("steady_flow", "architecture_comparison", "pi-fno_m4x5_h8_l2__fd__lr-low"), config)
    with pytest.raises(loader.ConfigError, match="pi-fno_m4x5_h8_l2__fd"):
        naming.validate_config_name(naming.ConfigSource("steady_flow", "architecture_comparison", "fno_m4x5_h8_l2"), config)
    naming.validate_config_name(naming.ConfigSource("steady_flow", "optuna", "pi-fno_search"), config)


def _plan_children(*, b_scale: float = 1.0, a1_scale: float = 1.0, seconds: float = 0.25) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the resolved B and A1 children of a matched plan with explicit pair settings."""
    raw = configs.transient_matched_config()
    raw["training"]["b"].update({"lr_scale": b_scale, "train_seconds": seconds})
    raw["training"]["a1"]["lr_scale"] = a1_scale
    plan = configs.resolve_plan(raw)
    return dict(plan.b), dict(plan.a1)


def test_pair_key_names_budget_rollout_and_one_shared_learning_rate_scale() -> None:
    """Equal B and A1 scales share one lr token; the rollout is B's window length."""
    b, a1 = _plan_children(b_scale=0.5, a1_scale=0.5, seconds=600.0)

    assert naming.pair_key(b, a1) == "b-a1__budget-600s__rollout-4__lr-0p5"
    assert b["run"]["layout"]["pair"] == a1["run"]["layout"]["pair"] == naming.pair_key(b, a1)


def test_pair_key_separates_unequal_learning_rate_scales() -> None:
    """Different B and A1 scales are named separately; a fractional budget keeps its decimal."""
    b, a1 = _plan_children(b_scale=1.0, a1_scale=0.25, seconds=0.25)

    assert naming.pair_key(b, a1) == "b-a1__budget-0p25s__rollout-4__lr-b-1__lr-a1-0p25"


def test_pair_key_requires_the_b_child_first() -> None:
    """The key reads B's budget and window, so swapped children are refused."""
    b, a1 = _plan_children()

    with pytest.raises(ValueError, match="B child first"):
        naming.pair_key(a1, b)


def test_trial_leaf_formats_number_label_and_sorted_values() -> None:
    """Zero-padded number, model label, then one name-value token per sampled parameter sorted by name."""
    config = configs.resolve(configs.direct_config())
    values = {"lr": 0.00038836, "batch_size": 16, "warmup": True, "momentum_weight": 123456.0, "scheduler": "Plateau X"}

    assert naming.trial_leaf(7, config, {}) == "0007__fno_m4x5_h8_l2"
    assert naming.trial_leaf(7, config, values) == (
        "0007__fno_m4x5_h8_l2__batch_size-16__lr-0p000388__momentum_weight-123000__scheduler-plateau-x__warmup-yes"
    )
    assert naming.trial_leaf(12345, config, {}).startswith("12345__")
    with pytest.raises(ValueError, match="non-negative integer"):
        naming.trial_leaf(-1, config, {})
    with pytest.raises(ValueError, match="finite"):
        naming.trial_leaf(1, config, {"lr": float("nan")})


def test_run_name_follows_steady_stage_pair_revision_and_trial() -> None:
    """Steady runs omit the stage; staged runs append pair and stage; trials use study and trial number."""
    base = {"config_name": "fno_m4x5_h8_l2", "seed": 9, "revision": None, "budget_epochs": None, "stage": "single", "pair": None, "trial": None}
    pair = "b-a1__budget-600s__rollout-4__lr-1"

    assert naming.run_name(base) == "fno_m4x5_h8_l2__seed-9"
    assert naming.run_name({**base, "revision": "fixed"}) == "fno_m4x5_h8_l2__seed-9__rev-fixed"
    assert naming.run_name({**base, "stage": "a0"}) == "fno_m4x5_h8_l2__seed-9__a0"
    assert naming.run_name({**base, "stage": "a1", "pair": pair, "revision": "fixed"}) == f"fno_m4x5_h8_l2__seed-9__rev-fixed__{pair}__a1"
    assert naming.run_name({**base, "config_name": "fno_search", "trial": "0003__fno_m4x5_h8_l2"}) == "fno_search__trial-0003"
    assert naming.parent_label({**base, "family": "tests", "revision": "fixed"}) == "tests__fno_m4x5_h8_l2__seed-9__rev-fixed"


def test_budget_epochs_name_the_transient_seed_group_and_the_study_before_the_revision() -> None:
    """A transient seed group carries its A0 epoch budget and a study its trial budget; a revision label stays last."""
    base = {"config_name": "fno_m4x5_h8_l2", "seed": 9, "revision": None, "budget_epochs": 40, "stage": "a0", "pair": None, "trial": None}
    pair = "b-a1__budget-600s__rollout-4__lr-1"
    study = {**base, "family": "optuna", "config_name": "fno_search", "stage": "single", "trial": "0003__fno_m4x5_h8_l2"}

    assert naming.run_name(base) == "fno_m4x5_h8_l2__seed-9__a0ep-40__a0"
    assert naming.run_name({**base, "stage": "a1", "pair": pair, "revision": "fixed"}) == f"fno_m4x5_h8_l2__seed-9__a0ep-40__rev-fixed__{pair}__a1"
    assert naming.parent_label({**base, "family": "tests", "revision": "fixed"}) == "tests__fno_m4x5_h8_l2__seed-9__a0ep-40__rev-fixed"
    assert naming.run_name(study) == "fno_search__trial-0003"
    assert naming.parent_label({**study, "revision": "fixed"}) == "optuna__fno_search__ep-40__rev-fixed__seed-9"
    assert common.paths.seed_group_name(9) == "seed-9"
    assert common.paths.study_group_name("fno_search", 40) == "fno_search__ep-40"
    for invalid in (0, -1, 2.0, True, None):
        with pytest.raises(ValueError, match="positive integer"):
            common.paths.study_group_name("fno_search", invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        common.paths.seed_group_name(9, a0_epochs=0)


def test_resolved_layout_is_a_pure_function_of_config_source_and_revision(tmp_path: Path) -> None:
    """The same inputs give the same layout; only the revision and the source change it."""
    raw = configs.direct_config()
    source = configs.source_for(raw)
    first = loader.resolve_config(copy.deepcopy(raw), source=source, output_root=tmp_path)
    second = loader.resolve_config(copy.deepcopy(raw), source=source, output_root=tmp_path)
    revised = loader.resolve_config(loader.apply_training_overrides(raw, revision="fixed-loss"), source=source, output_root=tmp_path)

    assert first["run"] == second["run"]
    assert first["run"]["layout"] == {
        "task": "steady_flow",
        "dataset": "synthetic_train",
        "family": "tests",
        "config_name": "fno_m4x5_h8_l2",
        "model_label": "fno_m4x5_h8_l2",
        "seed": 17,
        "revision": None,
        "budget_epochs": None,
        "stage": "single",
        "pair": None,
        "trial": None,
        "a0_id": first["run"]["layout"]["a0_id"],
        "branch_id": None,
    }
    assert first["run"]["name"] == "fno_m4x5_h8_l2__seed-17"
    assert revised["run"]["revision"] == revised["run"]["layout"]["revision"] == "fixed-loss"
    assert revised["run"]["name"] == "fno_m4x5_h8_l2__seed-17__rev-fixed-loss"
    assert revised["run"]["layout"]["a0_id"] == first["run"]["layout"]["a0_id"]
    assert common.paths.resolve_config_run_dir(revised) == common.paths.resolve_config_run_dir(first).with_name("seed-17__rev-fixed-loss")


def test_resolver_rejects_a_mismatching_source_task_and_config_name() -> None:
    """The source task must equal the config task and an experiment name must match its model label."""
    raw = configs.direct_config()
    label = configs.source_for(raw).name

    with pytest.raises(loader.ConfigError, match="Task config path mismatch"):
        loader.resolve_config(copy.deepcopy(raw), source=naming.ConfigSource("transient_drying", "tests", label))
    with pytest.raises(loader.ConfigError, match=r"must equal its model label 'fno_m4x5_h8_l2'"):
        loader.resolve_config(copy.deepcopy(raw), source=naming.ConfigSource("steady_flow", "tests", "other"))


def test_saved_layout_must_match_its_config_for_execution() -> None:
    """Execution rejects a layout whose dataset, revision or model label contradicts the saved config."""
    config = configs.resolve(configs.direct_config())
    loader.validate_resolved_config(config)

    for key, value, message in (
        ("dataset", "other_dataset", "seed, revision and epoch budget"),
        ("revision", "x", "seed, revision and epoch budget"),
        ("budget_epochs", 5, "seed, revision and epoch budget"),
        ("model_label", "fno_m9x9_h8_l2", "Model label contradicts"),
    ):
        tampered = copy.deepcopy(config)
        tampered["run"]["layout"][key] = value
        with pytest.raises(loader.ConfigError, match=message):
            loader.validate_resolved_config(tampered)


def test_layout_identity_is_recomputed_only_when_verification_is_requested() -> None:
    """An initial-stage identity that the running implementation does not reproduce fails execution but not structural validation."""
    config = configs.resolve(configs.direct_config())
    drifted = copy.deepcopy(config)
    drifted["run"]["layout"]["a0_id"] = "f" * 16

    with pytest.raises(loader.ConfigError, match="Initial run layout identity"):
        loader.validate_resolved_config(drifted)
    structural = loader.validate_resolved_config(drifted, verify_layout_identity=False)
    assert structural["run"]["layout"]["a0_id"] == "f" * 16
    drifted["run"]["layout"]["dataset"] = "other_dataset"
    with pytest.raises(loader.ConfigError, match="seed, revision and epoch budget"):
        loader.validate_resolved_config(drifted, verify_layout_identity=False)
