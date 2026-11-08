"""Protect strict authored matched A0/B/A1 plan resolution and run identity.

Responsibilities:
  - Check exact authored schemas, derived stage names, teacher binding, and inherited A1 protocol
  - Check budget, continuation-scale, worker, stride, and sampling admission
  - Check the stage-A-only A0 contract that remains outside the matched plan
  - Check which authored values enter A0 identity and which only the branch identity
  - Check that the A0 epoch budget names the seed group shared by all stages

Design principles:
  - Derived stages preserve explicit arm identity; invalid authored values fail before any run exists
  - Test-owned plans only: maintained configurations are covered by their own admission test

This module does NOT:
  - Execute A0/B/A1 sequencing, training, or receipts
  - Apply command-line overrides
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src import common, learning
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as plan_service

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_MODEL_KINDS = ("fno", "rno", "uno")
_STRIDE_TWO = 2
_TEACHER_FORCED_FIELDS = ("evaluation_interval", "additional_evaluation_interval", "gradient_accumulation_steps", "curriculum")


def _resolve(raw: dict[str, Any]) -> plan_service.MatchedA1BPlan:
    """Resolve one test-owned authored plan without checking occupied output names."""
    return configs.resolve_plan(raw)


def test_plan_derives_ordered_children_with_teacher_and_receipt_binding() -> None:
    """Derive independent validated A0/B/A1 configs whose names carry their stage."""
    plan = _resolve(configs.transient_matched_config())

    assert tuple(config["run"]["name"].rsplit("_", 1)[-1] for config in plan.stages) == ("a0", "b", "a1")
    assert [config["training"]["comparison_arm"] for config in plan.stages] == ["a0", "b", "a1"]
    assert [config["training"]["stage"] for config in plan.stages] == ["a", "b", "a"]
    assert plan.a0["training"]["teacher_handoff"] is None
    assert plan.b["training"]["teacher_handoff"] == plan.a1["training"]["teacher_handoff"] == {"source_run_name": plan.a0["run"]["name"]}
    assert plan.a1["training"]["matched_a1_b"]["receipt_source_run_name"] == plan.b["run"]["name"]
    assert plan.b["training"]["matched_a1_b"]["train_seconds"] == pytest.approx(0.25)
    assert plan.a0["training"]["matched_a1_b"]["train_seconds"] is plan.a1["training"]["matched_a1_b"]["train_seconds"] is None
    assert [config["training"]["matched_a1_b"]["evaluation_points"] for config in plan.stages] == [None, 4, 4]
    for name, config in zip(("a0", "b", "a1"), plan.stages, strict=True):
        assert plan.stage(name) is config
        assert loader.validate_resolved_config(config)["run"]["name"] == config["run"]["name"]
    with pytest.raises(ValueError, match="Unknown matched training stage"):
        plan.stage("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("model_kind", ["fno", "rno"])
def test_a1_inherits_the_teacher_forced_protocol_of_a0(model_kind: str) -> None:
    """A1 continues A0 teacher-forced: only its epoch cap and learning-rate scale are its own."""
    raw = configs.transient_matched_config(model_kind=model_kind)
    raw["training"]["a0"].update({"evaluation_interval": 2, "additional_evaluation_interval": 3, "gradient_accumulation_steps": 2})
    raw["training"]["a1"]["max_epochs"] = 9
    plan = _resolve(raw)

    assert plan.a1["temporal"]["sampling"] == plan.a0["temporal"]["sampling"] == raw["training"]["a0"]["sampling"]
    for field in _TEACHER_FORCED_FIELDS:
        assert plan.a1["training"][field] == plan.a0["training"][field] == raw["training"]["a0"][field]
    assert plan.a1["training"]["epochs"] == 9
    assert plan.a0["training"]["epochs"] == raw["training"]["a0"]["max_epochs"]
    assert plan.b["temporal"]["sampling"] == raw["training"]["b"]["sampling"]
    assert plan.b["training"]["curriculum"] == raw["training"]["b"]["curriculum"]


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda raw: raw["training"].pop("b"), "training"),
        (lambda raw: raw["training"].update(epochs=1), "training"),
        (lambda raw: raw["training"]["a0"].update(lr_scale=1.0), r"training\.a0"),
        (lambda raw: raw["training"]["b"].update(teacher_handoff=None), r"training\.b"),
        (lambda raw: raw["training"]["a1"].update(sampling={"mode": "one_step_transition"}), r"training\.a1"),
        (lambda raw: raw["temporal"].update(sampling={"mode": "one_step_transition"}), "temporal"),
    ],
    ids=["missing-b", "extra-training-key", "a0-extra-key", "b-extra-key", "a1-extra-key", "temporal-extra-key"],
)
def test_authored_plan_schema_is_exact(mutate: Callable[[dict[str, Any]], object], path: str) -> None:
    """Fail closed when an authored node omits or adds keys; A1 cannot author its own sampling."""
    raw = configs.transient_matched_config()
    mutate(raw)

    with pytest.raises(loader.ConfigError, match=rf"{path} keys must be exactly"):
        _resolve(raw)


def test_b_budget_must_be_authored_or_passed() -> None:
    """A null B target is an incomplete plan, not an implicit default."""
    raw = configs.transient_matched_config()
    raw["training"]["b"]["train_seconds"] = None

    with pytest.raises(loader.ConfigError, match=r"training\.b\.train_seconds is required"):
        _resolve(raw)


@pytest.mark.parametrize("stage", ["b", "a1"])
@pytest.mark.parametrize("value", [0, -0.5, float("nan"), float("inf"), True])
def test_continuation_lr_scale_must_be_positive_finite(stage: str, value: object) -> None:
    """Reject non-positive, non-finite, and boolean learning-rate factors for both continuations."""
    raw = configs.transient_matched_config()
    raw["training"][stage]["lr_scale"] = value

    with pytest.raises(loader.ConfigError, match=rf"training\.{stage}\.lr_scale must be a positive finite"):
        _resolve(raw)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw["training"]["b"].update(train_seconds=0.0), "B train_seconds must be positive"),
        (lambda raw: raw["training"]["a1"].update(match_to="b.best_epoch_seconds"), "B's committed train_seconds"),
        (lambda raw: raw["training"]["a0"].update(max_epochs=0), r"training\.a0\.max_epochs must be a positive integer"),
        (lambda raw: raw["training"].update(continuation_evaluations=0), "continuation_evaluations must be a positive"),
        (lambda raw: raw["data"].update(num_workers=2), "num_workers=0"),
        (lambda raw: raw["data"].update(persistent_workers=True), "persistent_workers"),
        (lambda raw: raw.update(task="steady_flow"), "only for transient_drying"),
    ],
    ids=["zero-seconds", "unmatched-receipt", "zero-epochs", "zero-evaluation-points", "worker-processes", "persistent-workers", "other-task"],
)
def test_matched_plan_rejects_unsupported_budgets_workers_and_tasks(mutate: Callable[[dict[str, Any]], object], match: str) -> None:
    """Keep the exact partial-epoch continuation prerequisites explicit and fail-closed."""
    raw = configs.transient_matched_config()
    mutate(raw)

    with pytest.raises(loader.ConfigError, match=match):
        _resolve(raw)


@pytest.mark.parametrize(
    ("stage", "arm"),
    [("invalid", "a0"), ("a", "b"), ("b", "a0")],
)
def test_resolved_config_rejects_inconsistent_stage_and_arm(stage: str, arm: str) -> None:
    """Keep the stage and comparison-arm pair as one strict semantic contract."""
    config = copy.deepcopy(dict(_resolve(configs.transient_matched_config()).a0))
    config["training"]["stage"] = stage
    config["training"]["comparison_arm"] = arm

    with pytest.raises(loader.ConfigError, match="stage and comparison_arm"):
        loader.validate_resolved_config(config)


@pytest.mark.parametrize(
    ("model_kind", "sampling"),
    [
        ("rno", {"mode": "rollout_window", "rollout_length": 4, "window_stride": 4, "window_offset": 0}),
        ("fno", {"mode": "trajectory", "rollout_length": 4}),
    ],
)
def test_sampling_follows_model_recurrence(model_kind: str, sampling: dict[str, Any]) -> None:
    """The RNO must see whole trajectories; stateless models must not."""
    raw = configs.transient_matched_config(model_kind=model_kind)
    raw["training"]["b"]["sampling"] = sampling

    with pytest.raises(loader.ConfigError, match="trajectory"):
        _resolve(raw)


def test_rno_trajectory_sampling_requires_its_window_length() -> None:
    """The RNO trains consecutive windows of whole trajectories, so its sampling must name their length."""
    raw = configs.transient_matched_config(model_kind="rno")
    raw["training"]["b"]["sampling"] = {"mode": "trajectory"}

    with pytest.raises(loader.ConfigError, match="rollout_length"):
        _resolve(raw)


@pytest.mark.parametrize("model_kind", _MODEL_KINDS)
def test_test_owned_matched_configs_preflight(tmp_path: Path, model_kind: str) -> None:
    """Preflight each supported architecture through its test-owned A0 child."""
    path = configs.write_config(tmp_path, configs.transient_matched_config(model_kind=model_kind), canonical_root=tmp_path)

    result = preflight.inspect_config(path)

    assert result.family == preflight.EXPERIMENT_FAMILY
    assert result.task == "transient_drying"
    assert result.model_kind == model_kind
    assert result.physics_enabled is False


def test_spatial_stride_is_required_authored_and_resolved_and_distinguishes_runs() -> None:
    """Refuse an authored or resolved config without the stride, and distinguish stride-two runs."""
    stride_one_raw = configs.transient_matched_config()
    missing_raw = copy.deepcopy(stride_one_raw)
    del missing_raw["data"]["spatial_stride"]
    with pytest.raises(loader.ConfigError, match=r"data\.spatial_stride is required"):
        _resolve(missing_raw)
    stride_one = _resolve(stride_one_raw)

    stride_two_raw = copy.deepcopy(stride_one_raw)
    stride_two_raw["data"]["spatial_stride"] = _STRIDE_TWO
    stride_two = _resolve(stride_two_raw)

    persisted_without_stride = copy.deepcopy(dict(stride_one.a0))
    persisted_without_stride["data"].pop("spatial_stride")
    checkpoint = learning.training.checkpoint

    with pytest.raises(loader.ConfigError, match=r"data\.spatial_stride is required"):
        loader.validate_resolved_config(persisted_without_stride)
    assert stride_two.a0["data"]["spatial_stride"] == _STRIDE_TWO
    assert stride_two.a0["run"]["layout"]["a0_id"] != stride_one.a0["run"]["layout"]["a0_id"]
    assert checkpoint.config_digest(stride_one.a0) != checkpoint.config_digest(stride_two.a0)
    assert checkpoint.resume_contract_digest(stride_one.a0) != checkpoint.resume_contract_digest(stride_two.a0)


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_spatial_stride_rejects_non_integer_or_non_positive_values(value: object) -> None:
    """Reject coercion and boundary-invalid stride values during config resolution."""
    raw = configs.transient_matched_config()
    raw["data"]["spatial_stride"] = value

    with pytest.raises((TypeError, ValueError), match="spatial_stride"):
        _resolve(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["training"]["b"]["curriculum"].update(seed=9),
        lambda raw: raw["data"].update(hdf5_cache_size=0),
        lambda raw: raw["tracking"]["wandb"].update(metric_schema_version=2),
        lambda raw: raw["training"].update(mixed_precision=False),
    ],
    ids=["curriculum-seed", "hdf5-cache-size", "metric-schema-version", "mixed-precision"],
)
def test_removed_keys_and_other_schema_versions_are_rejected(mutate: Callable[[dict[str, Any]], object]) -> None:
    """Reject the retired curriculum seed, HDF5 cache and mixed-precision keys and any metric schema version other than 1."""
    raw = configs.transient_matched_config()
    mutate(raw)

    with pytest.raises(loader.ConfigError):
        _resolve(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["training"]["b"].update(train_seconds=2.0),
        lambda raw: raw["training"]["b"].update(lr_scale=0.5),
        lambda raw: raw["training"]["a1"].update(lr_scale=0.5),
        lambda raw: raw["training"].update(continuation_evaluations=8),
    ],
    ids=["train-seconds", "b-lr-scale", "a1-lr-scale", "evaluation-points"],
)
def test_branch_settings_change_the_branch_but_never_the_shared_a0(mutate: Callable[[dict[str, Any]], object]) -> None:
    """B/A1 budget and continuation scale select a new branch of the same A0 checkpoint."""
    raw = configs.transient_matched_config()
    reference = _resolve(raw)
    mutate(raw)
    changed = _resolve(raw)

    assert changed.a0["run"]["layout"] == reference.a0["run"]["layout"]
    assert learning.training.checkpoint.config_digest(changed.a0) == learning.training.checkpoint.config_digest(reference.a0)
    assert changed.b["run"]["layout"]["a0_id"] == reference.b["run"]["layout"]["a0_id"]
    assert changed.b["run"]["layout"]["branch_id"] != reference.b["run"]["layout"]["branch_id"]
    assert changed.b["run"]["layout"]["branch_id"] == changed.a1["run"]["layout"]["branch_id"]


def test_seed_remains_part_of_the_shared_a0_identity() -> None:
    """A different seed trains a different A0 and therefore a different branch group."""
    first = _resolve(configs.transient_matched_config(seed=17))
    second = _resolve(configs.transient_matched_config(seed=23))

    assert first.a0["run"]["layout"]["a0_id"] != second.a0["run"]["layout"]["a0_id"]
    assert first.b["run"]["layout"]["branch_id"] != second.b["run"]["layout"]["branch_id"]


def test_every_matched_stage_shares_the_seed_group_named_by_the_a0_epoch_budget() -> None:
    """B and A1 keep their own epoch caps but sit in the seed group of their A0's budget, so another A0 budget is another group."""
    plan = _resolve(configs.transient_matched_config(seed=17))
    longer_raw = configs.transient_matched_config(seed=17)
    longer_raw["training"]["a0"]["max_epochs"] += 1
    longer = _resolve(longer_raw)
    budget = plan.a0["training"]["epochs"]

    assert plan.b["training"]["epochs"] != budget
    assert {config["run"]["layout"]["budget_epochs"] for config in plan.stages} == {budget}
    assert {common.paths.resolve_config_experiment_dir(config).name for config in plan.stages} == {f"seed-17__a0ep-{budget}"}
    assert all(f"__seed-17__a0ep-{budget}__" in f"{config['run']['name']}__" for config in plan.stages)
    assert common.paths.resolve_config_experiment_dir(longer.a0).name == f"seed-17__a0ep-{budget + 1}"
    assert common.paths.resolve_config_experiment_dir(longer.a0) != common.paths.resolve_config_experiment_dir(plan.a0)
    assert longer.a0["run"]["layout"]["a0_id"] != plan.a0["run"]["layout"]["a0_id"]


def test_stage_a_only_config_resolves_as_the_single_stage_a0() -> None:
    """The Optuna base protocol is one teacher-forced A0 whose schedule total is its epoch budget."""
    resolved = configs.resolve(configs.transient_stage_a_only_config())

    assert resolved["training"]["stage"] == "a"
    assert resolved["training"]["comparison_arm"] == "a0"
    assert resolved["training"]["teacher_handoff"] is None
    assert "matched_a1_b" not in resolved["training"]
    assert resolved["training"]["epochs"] == resolved["training"]["stage_schedule"]["total_epochs"]
    assert resolved["run"]["layout"]["budget_epochs"] == resolved["training"]["epochs"]
    assert common.paths.resolve_config_experiment_dir(resolved).name == f"seed-17__a0ep-{resolved['training']['epochs']}"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda training: training.update(epochs=training["epochs"] + 1), "training.epochs must equal"),
        (lambda training: training["stage_schedule"].update(unexpected=1), "stage_a_only"),
        (lambda training: training.update(comparison_arm="a1", teacher_handoff={"source_run_name": "a0"}), "Outside matched_a1_b"),
    ],
    ids=["epochs-differ-from-total", "schedule-extra-key", "continuation-arm"],
)
def test_stage_a_only_protocol_is_exact(mutate: Callable[[dict[str, Any]], object], match: str) -> None:
    """Outside a matched plan only the exact stage-A-only A0 protocol is admitted."""
    raw = configs.transient_stage_a_only_config()
    mutate(raw["training"])

    with pytest.raises(loader.ConfigError, match=match):
        configs.resolve(raw)
