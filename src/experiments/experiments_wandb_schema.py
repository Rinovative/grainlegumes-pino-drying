"""
experiments_wandb_schema.py

Define the curated transient W&B history projection and display organization.

Responsibilities:
  - Map authoritative completed-epoch metrics to concise W&B history namespaces
  - Select only scientifically useful transient series for the current schema
  - Apply presentation-only unit projections without recomputing scientific values
  - Derive current Stage A0/B group and job-type presentation from resolved config

Design principles:
  - Local training and evaluation metrics remain authoritative
  - W&B history is a bounded projection, not a mirror of all telemetry
  - One maintained projection defines current transient presentation

This module does NOT:
  - Initialize W&B, persist opaque run IDs, or perform network operations
  - Recompute objectives, normalized errors, physical errors, or channel metrics
  - Change local training history, checkpoints, or summaries
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .config import experiments_config_loader as config_loader

_GIB: Final = float(1024**3)


@dataclass(frozen=True, slots=True)
class MetricProjection:
    """Bind one authoritative source key to one current W&B history key."""

    source_key: str
    wandb_key: str
    multiplier: float = 1.0


def curated_transient_metric_projections(
    *,
    objective_id: str,
    state_aux_enabled: bool,
    cuda_enabled: bool,
) -> tuple[MetricProjection, ...]:
    """Return the approximately 25-series current transient history contract."""
    if not isinstance(objective_id, str) or not objective_id:
        message = "Transient W&B projection requires a non-empty objective identifier."
        raise ValueError(message)
    projections = [
        MetricProjection("train/loss_total", "Overview/train_loss"),
        MetricProjection(f"id/{objective_id}", "Overview/id_objective"),
        MetricProjection(f"ood/{objective_id}", "Overview/ood_objective"),
        MetricProjection("generalization/objective_gap", "Overview/generalization_gap"),
        MetricProjection("optimization/learning_rate", "Optimization/learning_rate"),
        MetricProjection("train/loss_data_T", "Loss/T"),
        MetricProjection("train/loss_data_phi", "Loss/phi"),
        MetricProjection("train/loss_data_w_surf", "Loss/w_surf"),
        MetricProjection("train/loss_data_w_int", "Loss/w_int"),
    ]
    if state_aux_enabled:
        projections.append(MetricProjection("train/loss_state_aux", "Loss/state_aux"))
    for role, namespace in (("id", "ID"), ("ood", "OOD")):
        projections.extend(
            (
                MetricProjection(f"{role}/{objective_id}/component/T", f"Accuracy/{namespace}/T"),
                MetricProjection(f"{role}/{objective_id}/component/phi", f"Accuracy/{namespace}/phi"),
                MetricProjection(f"{role}/{objective_id}/component/w_surf", f"Accuracy/{namespace}/w_surf"),
                MetricProjection(f"{role}/{objective_id}/component/w_int", f"Accuracy/{namespace}/w_int"),
                MetricProjection(
                    f"{role}/{objective_id}/component/grain_moisture_error",
                    f"Accuracy/{namespace}/grain_moisture",
                ),
            )
        )
    projections.extend(
        (
            MetricProjection("transient/curriculum_max_horizon", "Curriculum/horizon"),
            MetricProjection("transient/self_fed_stage", "Curriculum/self_fed_stage"),
            MetricProjection("system/train_duration_seconds", "Performance/train_seconds"),
            MetricProjection("system/epoch_duration_seconds", "Performance/epoch_seconds"),
            MetricProjection("system/train_samples_per_second", "Performance/samples_per_second"),
        )
    )
    if cuda_enabled:
        projections.append(
            MetricProjection(
                "system/cuda_peak_memory_allocated_bytes",
                "Performance/cuda_peak_memory_gib",
                multiplier=1.0 / _GIB,
            )
        )
    source_keys = [projection.source_key for projection in projections]
    destinations = [projection.wandb_key for projection in projections]
    if len(source_keys) != len(set(source_keys)) or len(destinations) != len(set(destinations)):
        message = "Curated transient W&B projections must have unique source and destination keys."
        raise RuntimeError(message)
    return tuple(projections)


def transient_run_organization(
    config: Mapping[str, Any],
    *,
    workflow: str,
) -> tuple[str | None, str]:
    """Return the maintained W&B group and job type for one transient child."""
    if workflow != "train":
        return None, workflow
    training = config.get("training")
    if not isinstance(training, Mapping):
        message = "Transient W&B organization requires resolved training identity."
        raise TypeError(message)
    arm = training.get("comparison_arm")
    stage = training.get("stage")
    if stage == "a" and arm == "a0":
        job_type = "stage_a0"
    elif stage == "a" and arm == "a_plus":
        job_type = "stage_a_plus"
    elif stage == "b" and arm == "b":
        job_type = "stage_b"
    else:
        message = f"Unsupported transient training stage/arm for W&B organization: {stage!r}/{arm!r}."
        raise ValueError(message)
    return config_loader.generate_parent_experiment_label(config), job_type
