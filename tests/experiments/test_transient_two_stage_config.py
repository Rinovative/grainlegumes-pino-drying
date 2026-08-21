# ruff: noqa: S101
"""Protect strict authored transient two-stage plan resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as transient_plan

_EXPERIMENTS = Path("configs/learning/transient_drying/experiments")
_FNO = _EXPERIMENTS / "fno_m128x160_h64_l3__lentil_chickpea__s9.yaml"
_ROUNDING_TOTAL = 7
_ROUNDED_STAGE_A = 4
_ROUNDED_STAGE_B = 3
_FIXED_TOTAL = 200
_MAINTAINED = (
    ("fno_m128x160_h64_l3__lentil_chickpea__s9.yaml", "fno"),
    ("rno_m24x24_h16_l3__lentil_chickpea__s9.yaml", "rno"),
    ("uno_m64x64_h32_l7_s1-05-05-1-1-2-2_r0p495__lentil_chickpea__s9.yaml", "uno"),
)


def test_plan_derives_distinct_stage_names_and_teacher_binding() -> None:
    """Derive independent validated A0/B configs with runtime-visible stage identity."""
    plan = transient_plan.load_and_resolve_transient_training_plan(_FNO)

    assert plan.stage_a["training"]["stage"] == "a"
    assert plan.stage_a["training"]["comparison_arm"] == "a0"
    assert plan.stage_b["training"]["stage"] == "b"
    assert plan.stage_b["training"]["comparison_arm"] == "b"
    assert plan.stage_a["run"]["name"] != plan.stage_b["run"]["name"]
    assert "stage_a0" in plan.stage_a["run"]["name"]
    assert "stage_b" in plan.stage_b["run"]["name"]
    assert plan.stage_b["training"]["teacher_handoff"] == {"source_run_name": plan.stage_a["run"]["name"]}
    assert loader.validate_resolved_config(plan.stage("a"))["run"]["name"] == plan.stage_a["run"]["name"]
    assert loader.validate_resolved_config(plan.stage("b"))["run"]["name"] == plan.stage_b["run"]["name"]
    with pytest.raises(ValueError, match="Unknown transient training stage"):
        plan.stage("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["partial", "mixed"])
def test_plan_rejects_partial_or_mixed_training_schema(mutation: str) -> None:
    """Fail closed when authored training mixes or omits mandatory stage branches."""
    raw: dict[str, Any] = loader.load_yaml(_FNO)
    broken = copy.deepcopy(raw)
    if mutation == "partial":
        del broken["training"]["stage_b"]
    else:
        broken["training"]["epochs"] = 1

    with pytest.raises(loader.ConfigError, match="training keys must be exactly"):
        transient_plan.resolve_transient_training_plan(broken)


@pytest.mark.parametrize(
    ("stage", "arm"),
    [("invalid", "a0"), ("a", "b"), ("b", "a0")],
)
def test_resolved_config_rejects_inconsistent_stage_and_arm(stage: str, arm: str) -> None:
    """Keep the stage and comparison-arm pair as one strict semantic contract."""
    config = copy.deepcopy(dict(transient_plan.load_and_resolve_transient_training_plan(_FNO).stage_a))
    config["training"]["stage"] = stage
    config["training"]["comparison_arm"] = arm

    with pytest.raises(loader.ConfigError, match="stage and comparison_arm"):
        loader.validate_resolved_config(config)


def test_plan_rejects_unknown_stage_schema_key() -> None:
    """Reject unknown authored stage keys before the single-stage loader is called."""
    broken = loader.load_yaml(_FNO)
    broken["training"]["stage_b"]["teacher_handoff"] = None

    with pytest.raises(loader.ConfigError, match=r"training\.stage_b keys must be exactly"):
        transient_plan.resolve_transient_training_plan(broken)


@pytest.mark.parametrize(("filename", "model_kind"), _MAINTAINED)
def test_maintained_two_stage_configs_preflight(filename: str, model_kind: str) -> None:
    """Preflight each maintained architecture-first authored plan through its A0 child."""
    result = preflight.inspect_config(_EXPERIMENTS / filename)

    assert result.family == preflight.EXPERIMENT_FAMILY
    assert result.task == "transient_drying"
    assert result.model_kind == model_kind
    assert result.physics_enabled is False


def test_stage_schedule_allocates_complementary_epochs_with_half_up_rounding() -> None:
    """Keep the authored allocation owner deterministic and complementary."""
    allocation = transient_plan.resolve_stage_epoch_allocation(
        {"mode": "joint_ab", "budget_unit": "epochs", "total_epochs": _ROUNDING_TOTAL, "stage_a_fraction": 0.5}
    )

    assert allocation["stage_a_epochs"] == _ROUNDED_STAGE_A
    assert allocation["stage_b_epochs"] == _ROUNDED_STAGE_B
    assert allocation["stage_a_epochs"] + allocation["stage_b_epochs"] == allocation["total_epochs"]


@pytest.mark.parametrize(("filename", "model_kind"), _MAINTAINED)
def test_fixed_plans_share_one_authored_epoch_schedule(filename: str, model_kind: str) -> None:
    """Derive both fixed stages from one shared authored schedule owner."""
    del model_kind
    raw = loader.load_yaml(_EXPERIMENTS / filename)
    plan = transient_plan.resolve_transient_training_plan(raw)

    assert raw["training"]["stage_schedule"]["total_epochs"] == _FIXED_TOTAL
    assert plan.stage_a["training"]["stage_schedule"] == plan.stage_b["training"]["stage_schedule"]
    assert plan.stage_a["training"]["epochs"] + plan.stage_b["training"]["epochs"] == _FIXED_TOTAL
    assert all(value is None for value in plan.stage_b["training"]["matched_compute"].values())
