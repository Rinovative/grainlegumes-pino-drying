"""Protect command-line training overrides applied to authored configurations.

Responsibilities:
  - Check which budget a matched plan, a stage-A-only A0, and a steady run receive from --epochs
  - Check that B rollout window and continuation scale reach exactly their stages
  - Check that matched-only overrides are rejected for every other configuration kind
  - Check that only a non-negative integer is admitted as the replicate seed

Design principles:
  - Overrides copy the authored request and the result must still resolve through its loader
  - Overrides only bind values; the matched plan resolver rejects invalid ones before any run identity exists

This module does NOT:
  - Parse command-line text or dispatch training
  - Resolve saved-effective resume, which refuses every semantic override
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from support import configs

from src.experiments.config import experiments_config_loader as loader

_EPOCHS = 6
_MATCHED_ONLY = ({"rollout_length": 4}, {"lr_scale": 0.5})


def test_matched_epochs_override_sets_the_one_k_of_a0_and_b() -> None:
    """--epochs binds A0's K, which also sizes B (K x N transitions); A1 matches B's receipt, and B/A1 settings stay."""
    raw = configs.transient_matched_config()
    before = copy.deepcopy(raw)

    changed = loader.apply_training_overrides(raw, epochs=_EPOCHS)
    plan = configs.resolve_plan(changed)

    assert raw == before
    assert changed["training"]["a0"]["budget_epochs"] == _EPOCHS
    assert changed["training"]["b"] == raw["training"]["b"]
    assert changed["training"]["a1"] == raw["training"]["a1"]
    assert plan.a0["training"]["budget_epochs"] == _EPOCHS
    assert {config["run"]["layout"]["budget_epochs"] for config in plan.stages} == {_EPOCHS}
    assert {config["training"]["stage_schedule"]["a0_budget_epochs"] for config in (plan.b, plan.a1)} == {_EPOCHS}


def test_stage_a_only_epochs_override_keeps_schedule_budget_equal_to_the_run_budget() -> None:
    """The schedule's and the run's budget_epochs are one budget, so the override must move both."""
    raw = configs.transient_stage_a_only_config()

    changed = loader.apply_training_overrides(raw, epochs=_EPOCHS)
    resolved = configs.resolve(changed)

    assert changed["training"]["budget_epochs"] == changed["training"]["stage_schedule"]["budget_epochs"] == _EPOCHS
    assert raw["training"]["budget_epochs"] == raw["training"]["stage_schedule"]["budget_epochs"] != _EPOCHS
    assert resolved["training"]["budget_epochs"] == resolved["training"]["stage_schedule"]["budget_epochs"] == _EPOCHS


def test_steady_epochs_override_changes_only_the_epoch_budget() -> None:
    """Preserve every scientific section of a single-stage request."""
    raw = configs.direct_config()
    before = copy.deepcopy(raw)

    changed = loader.apply_training_overrides(raw, epochs=2)

    assert raw == before
    assert changed["training"]["budget_epochs"] == 2
    assert {key: value for key, value in changed.items() if key != "training"} == {key: value for key, value in raw.items() if key != "training"}


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5])
def test_epochs_override_must_be_a_positive_integer(epochs: object) -> None:
    """Reject zero, negative, boolean, and fractional budgets for every configuration kind."""
    with pytest.raises(loader.ConfigError, match="--epochs must be a positive integer"):
        loader.apply_epoch_budget(configs.transient_matched_config(), epochs)  # type: ignore[arg-type]


def test_seed_override_must_be_a_non_negative_integer() -> None:
    """Reject a negative replicate seed before any layout or seed plan derives from it."""
    with pytest.raises(loader.ConfigError, match="--seed must be a non-negative integer"):
        loader.apply_training_overrides(configs.direct_config(), seed=-1)


def test_lr_scale_override_sets_both_continuations() -> None:
    """B and A1 share one factor so a paired sweep compares equal learning-rate starts."""
    raw = configs.transient_matched_config()

    changed = loader.apply_training_overrides(raw, lr_scale=0.5)
    plan = configs.resolve_plan(changed)

    assert changed["training"]["b"]["lr_scale"] == changed["training"]["a1"]["lr_scale"] == pytest.approx(0.5)
    assert raw["training"]["b"]["lr_scale"] == raw["training"]["a1"]["lr_scale"] == pytest.approx(1.0)
    assert plan.b["training"]["matched_a1_b"]["lr_scale"] == plan.a1["training"]["matched_a1_b"]["lr_scale"] == pytest.approx(0.5)
    assert plan.a0["training"]["matched_a1_b"]["lr_scale"] is None


@pytest.mark.parametrize("value", [0, -0.5, float("nan"), float("inf"), True])
def test_lr_scale_override_must_be_positive_finite(value: object) -> None:
    """Reject factors that would zero, invert, or poison the restored learning rate when the plan resolves."""
    changed = loader.apply_training_overrides(configs.transient_matched_config(), lr_scale=value)  # type: ignore[arg-type]
    with pytest.raises(loader.ConfigError, match="lr_scale must be a positive finite"):
        configs.resolve_plan(changed)


@pytest.mark.parametrize(
    ("model_kind", "rollout_length", "sampling", "lengths", "fractions"),
    [
        ("fno", 2, {"mode": "rollout_window", "rollout_length": 2, "window_stride": 2, "window_offset": 0}, [2], [0.0]),
        ("fno", 8, {"mode": "rollout_window", "rollout_length": 8, "window_stride": 8, "window_offset": 0}, [2, 4, 8], [0.0, 1 / 3, 2 / 3]),
        ("rno", 2, {"mode": "trajectory", "rollout_length": 2}, [2], [0.0]),
    ],
)
def test_rollout_length_override_resizes_the_b_window_and_truncates_its_curriculum(
    model_kind: str,
    rollout_length: int,
    sampling: dict[str, Any],
    lengths: list[int],
    fractions: list[float],
) -> None:
    """B follows the requested horizon; curriculum lengths stay within it and keep even milestone spacing."""
    raw = configs.transient_matched_config(model_kind=model_kind)
    before = copy.deepcopy(raw)

    changed = loader.apply_training_overrides(raw, rollout_length=rollout_length)
    plan = configs.resolve_plan(changed)

    b = changed["training"]["b"]
    assert raw == before
    assert b["sampling"] == sampling
    assert b["curriculum"]["lengths"] == lengths
    assert b["curriculum"]["milestone_fractions"] == pytest.approx(fractions)
    assert changed["training"]["a0"] == raw["training"]["a0"]
    assert plan.b["temporal"]["sampling"] == sampling
    assert plan.a1["temporal"]["sampling"] == plan.a0["temporal"]["sampling"]


def test_rollout_length_override_admits_only_curriculum_horizons() -> None:
    """Horizons outside the admitted ladder are rejected rather than rounded."""
    with pytest.raises(loader.ConfigError, match="admitted curriculum horizon"):
        loader.apply_training_overrides(configs.transient_matched_config(), rollout_length=3)


@pytest.mark.parametrize(
    "raw_factory",
    [configs.transient_stage_a_only_config, configs.direct_config],
    ids=["stage-a-only", "steady-single-stage"],
)
@pytest.mark.parametrize("override", _MATCHED_ONLY, ids=["rollout-length", "lr-scale"])
def test_matched_only_overrides_are_rejected_for_other_configurations(raw_factory: Any, override: dict[str, Any]) -> None:
    """B window and continuation scale have no meaning without a matched plan."""
    raw = raw_factory()
    before = copy.deepcopy(raw)

    with pytest.raises(loader.ConfigError, match="require a transient matched_a1_b plan"):
        loader.apply_training_overrides(raw, **override)

    assert raw == before
