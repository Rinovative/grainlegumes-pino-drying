# ruff: noqa: S101
"""Protect the bounded current transient W&B projection and grouping."""

from __future__ import annotations

from support import configs

from src.experiments import experiments_wandb_schema as schema
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_transient_plan as transient_plan

_OBJECTIVE = "normalized_drying_group_macro_rmse"
_CURATED_CUDA_COUNT = 25
_CURATED_CUDA_WITH_AUX_COUNT = 26


def test_current_cuda_history_is_unique_curated_and_approximately_25() -> None:
    """Keep only the required interpretive series and no redundant task prefix."""
    projections = schema.curated_transient_metric_projections(
        objective_id=_OBJECTIVE,
        state_aux_enabled=False,
        cuda_enabled=True,
    )
    sources = [projection.source_key for projection in projections]
    destinations = [projection.wandb_key for projection in projections]

    assert len(projections) == _CURATED_CUDA_COUNT
    assert len(sources) == len(set(sources))
    assert len(destinations) == len(set(destinations))
    assert all(not key.startswith("Transient/") for key in destinations)
    assert {
        "Overview/train_loss",
        "Overview/id_objective",
        "Overview/ood_objective",
        "Overview/generalization_gap",
        "Optimization/learning_rate",
        "Accuracy/ID/grain_moisture",
        "Accuracy/OOD/grain_moisture",
        "Curriculum/horizon",
        "Performance/cuda_peak_memory_gib",
    }.issubset(destinations)
    assert "Loss/state_aux" not in destinations
    assert not any("optimizer_steps" in key or "draw_index" in key for key in destinations)
    assert destinations.count("Overview/id_objective") == 1
    assert destinations.count("Overview/ood_objective") == 1


def test_optional_series_and_cuda_unit_projection_follow_runtime_semantics() -> None:
    """Include state aux only when enabled and convert bytes only at presentation."""
    projections = schema.curated_transient_metric_projections(
        objective_id=_OBJECTIVE,
        state_aux_enabled=True,
        cuda_enabled=True,
    )
    by_destination = {projection.wandb_key: projection for projection in projections}

    assert len(projections) == _CURATED_CUDA_WITH_AUX_COUNT
    assert by_destination["Loss/state_aux"].source_key == "train/loss_state_aux"
    assert by_destination["Performance/cuda_peak_memory_gib"].multiplier * 1024**3 == 1.0
    cpu_destinations = {
        projection.wandb_key
        for projection in schema.curated_transient_metric_projections(
            objective_id=_OBJECTIVE,
            state_aux_enabled=False,
            cuda_enabled=False,
        )
    }
    assert "Performance/cuda_peak_memory_gib" not in cpu_destinations


def test_current_stage_children_share_parent_group_and_use_distinct_job_types() -> None:
    """Use presentation labels for grouping while leaving opaque IDs authoritative."""
    raw = configs.transient_two_stage_config()
    raw["data"]["train_dataset"] = "transient_drying__synthetic_family__id__0123456789abcdef"
    plan = transient_plan.resolve_transient_training_plan(raw)
    group_a, job_a = schema.transient_run_organization(
        plan.stage_a,
        workflow="train",
        metric_schema_version=2,
    )
    group_b, job_b = schema.transient_run_organization(
        plan.stage_b,
        workflow="train",
        metric_schema_version=2,
    )

    assert group_a is not None
    assert group_a == group_b == loader.generate_parent_experiment_label(plan.stage_a)
    assert job_a == "stage_a0"
    assert job_b == "stage_b"
    assert plan.stage_a["data"]["train_dataset"] not in group_a
    assert "transient_drying" not in group_a
    assert "0123456789abcdef" not in group_a
    assert schema.transient_run_organization(plan.stage_a, workflow="train", metric_schema_version=1) == (None, "train")
