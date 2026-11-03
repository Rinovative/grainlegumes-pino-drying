"""Protect the bounded current transient W&B projection and grouping.

Responsibilities:
  - Check bounded CUDA history projection, optional series, parent grouping, and stage job types

Design principles:
  - Current semantic series remain unique and runtime units are explicit

This module does NOT:
  - Contact W&B or define model optimization behavior
"""

from __future__ import annotations

from support import configs

from src.experiments import experiments_wandb_schema as schema
from src.experiments.config import experiments_config_naming as naming

_OBJECTIVE = "normalized_drying_group_macro_rmse"


def test_current_cuda_history_has_unique_semantic_series() -> None:
    """Keep only the required interpretive series and no redundant task prefix."""
    projections = schema.curated_transient_metric_projections(
        objective_id=_OBJECTIVE,
        state_aux_enabled=False,
        cuda_enabled=True,
    )
    sources = [projection.source_key for projection in projections]
    destinations = [projection.wandb_key for projection in projections]

    assert len(sources) == len(set(sources))
    assert len(destinations) == len(set(destinations))
    assert all(not key.startswith("Transient/") for key in destinations)
    assert {
        "Overview/train_loss",
        "Overview/id_objective",
        "Overview/additional_evaluation_objective",
        "Overview/additional_evaluation_difference",
        "Optimization/learning_rate",
        "Accuracy/ID/grain_moisture",
        "Accuracy/AdditionalEvaluation/grain_moisture",
        "Curriculum/horizon",
        "Performance/cuda_peak_memory_gib",
    }.issubset(destinations)
    assert "Loss/state_aux" not in destinations
    assert not any("optimizer_steps" in key or "draw_index" in key for key in destinations)
    assert destinations.count("Overview/id_objective") == 1
    assert destinations.count("Overview/additional_evaluation_objective") == 1


def test_optional_series_and_cuda_unit_projection_follow_runtime_semantics() -> None:
    """Include state aux only when enabled and convert bytes only at presentation."""
    projections = schema.curated_transient_metric_projections(
        objective_id=_OBJECTIVE,
        state_aux_enabled=True,
        cuda_enabled=True,
    )
    by_destination = {projection.wandb_key: projection for projection in projections}

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
    raw = configs.transient_matched_config()
    raw["data"]["train_dataset"] = "transient_drying__synthetic_family__id__0123456789abcdef"
    plan = configs.resolve_plan(raw)
    organization = {name: schema.transient_run_organization(plan.stage(name), workflow="train") for name in ("a0", "b", "a1")}
    group = organization["a0"][0]

    assert group is not None
    assert {group_label for group_label, _job in organization.values()} == {group}
    assert group == naming.parent_label(plan.a0["run"]["layout"])
    assert {name: job for name, (_group, job) in organization.items()} == {"a0": "stage_a0", "b": "stage_b", "a1": "stage_a1"}
    assert plan.a0["data"]["train_dataset"] not in group
    assert "transient_drying" not in group
    assert "0123456789abcdef" not in group
