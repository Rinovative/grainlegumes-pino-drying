# ruff: noqa: S101
"""Protect strict authored transient two-stage plan resolution."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src import learning
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as transient_plan

if TYPE_CHECKING:
    from pathlib import Path

_ROUNDING_TOTAL = 7
_ROUNDED_STAGE_A = 4
_ROUNDED_STAGE_B = 3
_FIXED_TOTAL = 7
_STRIDE_TWO = 2
_MODEL_KINDS = ("fno", "rno", "uno")


def _raw(model_kind: str = "fno") -> dict[str, Any]:
    """Return one isolated test-owned authored transient plan."""
    return configs.transient_two_stage_config(model_kind=model_kind)


def _plan(model_kind: str = "fno") -> transient_plan.TransientTrainingPlan:
    """Resolve one isolated test-owned authored transient plan."""
    return transient_plan.resolve_transient_training_plan(_raw(model_kind))


def test_plan_derives_distinct_stage_names_and_teacher_binding() -> None:
    """Derive independent validated A0/B configs with runtime-visible stage identity."""
    plan = _plan()

    assert plan.stage_a["training"]["stage"] == "a"
    assert plan.stage_a["training"]["comparison_arm"] == "a0"
    assert plan.stage_b["training"]["stage"] == "b"
    assert plan.stage_b["training"]["comparison_arm"] == "b"
    assert plan.stage_a["run"]["name"] != plan.stage_b["run"]["name"]
    assert plan.stage_a["run"]["name"].endswith("_a0")
    assert plan.stage_b["run"]["name"].endswith("_b")
    assert plan.stage_b["training"]["teacher_handoff"] == {"source_run_name": plan.stage_a["run"]["name"]}
    assert loader.validate_resolved_config(plan.stage("a"))["run"]["name"] == plan.stage_a["run"]["name"]
    assert loader.validate_resolved_config(plan.stage("b"))["run"]["name"] == plan.stage_b["run"]["name"]
    with pytest.raises(ValueError, match="Unknown transient training stage"):
        plan.stage("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["partial", "mixed"])
def test_plan_rejects_partial_or_mixed_training_schema(mutation: str) -> None:
    """Fail closed when authored training mixes or omits mandatory stage branches."""
    broken = copy.deepcopy(_raw())
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
    config = copy.deepcopy(dict(_plan().stage_a))
    config["training"]["stage"] = stage
    config["training"]["comparison_arm"] = arm

    with pytest.raises(loader.ConfigError, match="stage and comparison_arm"):
        loader.validate_resolved_config(config)


def test_plan_rejects_unknown_stage_schema_key() -> None:
    """Reject unknown authored stage keys before the single-stage loader is called."""
    broken = _raw()
    broken["training"]["stage_b"]["teacher_handoff"] = None

    with pytest.raises(loader.ConfigError, match=r"training\.stage_b keys must be exactly"):
        transient_plan.resolve_transient_training_plan(broken)


@pytest.mark.parametrize("model_kind", _MODEL_KINDS)
def test_test_owned_two_stage_configs_preflight(tmp_path: Path, model_kind: str) -> None:
    """Preflight each supported architecture through its test-owned A0 child."""
    path = configs.write_yaml(
        tmp_path / "configs" / "learning" / "transient_drying" / "experiments" / f"{model_kind}.yaml",
        _raw(model_kind),
    )

    result = preflight.inspect_config(path)

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


@pytest.mark.parametrize("model_kind", _MODEL_KINDS)
def test_fixed_plans_share_one_authored_epoch_schedule(model_kind: str) -> None:
    """Derive both fixed stages from one shared authored schedule owner."""
    raw = _raw(model_kind)
    plan = transient_plan.resolve_transient_training_plan(raw)

    assert raw["training"]["stage_schedule"]["total_epochs"] == _FIXED_TOTAL
    assert plan.stage_a["training"]["stage_schedule"] == plan.stage_b["training"]["stage_schedule"]
    assert plan.stage_a["training"]["epochs"] + plan.stage_b["training"]["epochs"] == _FIXED_TOTAL
    assert all(value is None for value in plan.stage_b["training"]["matched_compute"].values())


def test_spatial_stride_defaults_to_one_and_changes_run_identity_only_when_explicit() -> None:
    """Resolve legacy omission at full resolution and distinguish stride-two runs."""
    stride_one_raw = _raw()
    stride_one_raw["data"].pop("spatial_stride", None)
    stride_one = transient_plan.resolve_transient_training_plan(stride_one_raw)

    stride_two_raw = copy.deepcopy(stride_one_raw)
    stride_two_raw["data"]["spatial_stride"] = _STRIDE_TWO
    stride_two = transient_plan.resolve_transient_training_plan(stride_two_raw)

    assert stride_one.stage_a["data"]["spatial_stride"] == 1
    assert stride_one.stage_b["data"]["spatial_stride"] == 1
    assert stride_two.stage_a["data"]["spatial_stride"] == _STRIDE_TWO
    assert "_stride2_" in stride_two.stage_a["run"]["name"]
    assert stride_one.stage_a["run"]["name"] != stride_two.stage_a["run"]["name"]
    assert learning.training.checkpoint.config_digest(stride_one.stage_a) != learning.training.checkpoint.config_digest(stride_two.stage_a)
    assert learning.training.checkpoint.resume_contract_digest(stride_one.stage_a) != learning.training.checkpoint.resume_contract_digest(
        stride_two.stage_a
    )


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_spatial_stride_rejects_non_integer_or_non_positive_values(value: object) -> None:
    """Reject coercion and boundary-invalid stride values during config resolution."""
    raw = _raw()
    raw["data"]["spatial_stride"] = value

    with pytest.raises((TypeError, ValueError), match="spatial_stride"):
        transient_plan.resolve_transient_training_plan(raw)


def test_transient_persistent_workers_require_worker_processes() -> None:
    """Fail fast when a transient config enables persistence without workers."""
    raw = _raw()
    raw["data"].update({"num_workers": 0, "persistent_workers": True})

    with pytest.raises(loader.ConfigError, match="persistent_workers"):
        transient_plan.resolve_transient_training_plan(raw)
