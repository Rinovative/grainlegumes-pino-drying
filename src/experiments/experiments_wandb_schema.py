"""Define the curated W&B history projection and run organization for both tasks.

Responsibilities:
  - Map authoritative completed-epoch metrics to the shared W&B sections Overview, Loss, Accuracy, Physics, Curriculum, Performance and Optuna
  - Select only scientifically useful series for the task and the run's loss, monitor, OOD and device settings
  - Apply presentation-only unit projections without recomputing scientific values
  - Derive group, job type and name of a run from its saved layout and training stage

Design principles:
  - Local training and evaluation metrics remain authoritative
  - W&B history is a bounded projection, not a mirror of all telemetry
  - One maintained projection list gives both tasks the same section and key names

This module does NOT:
  - Initialize W&B, persist opaque run IDs, or perform network operations
  - Recompute objectives, normalized errors, physical errors, or channel metrics
  - Change local training history, checkpoints, or summaries
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from src import common

_GIB: Final = float(1024**3)
_ROLE_SECTIONS: Final = (("id", "Validation"), ("additional_evaluation", "OOD"))
_TRANSIENT_FIELDS: Final = ("T", "phi", "w_surf", "w_int")
_TRANSIENT_ACCURACY: Final = (*((name, name) for name in _TRANSIENT_FIELDS), ("grain_moisture_error", "grain_moisture"))
_PHYSICS_TERMS: Final = ("momentum", "continuity", "boundary")
_STAGE_JOB_TYPES: Final = {("a", "a0"): "stage_a0", ("b", "b"): "stage_b", ("a", "a1"): "stage_a1"}

# Source suffix below ``physics/id/`` to W&B label; the monitor producer owns the suffixes.
PHYSICS_VALIDATION_METRICS: Final = {
    "momentum_residual_mse": "momentum",
    "flux_balance_mse": "flux_balance",
    "pressure_boundary_mse": "pressure_boundary",
}


@dataclass(frozen=True, slots=True)
class MetricProjection:
    """Bind one authoritative source key to one W&B history key; only ``in_summary`` series keep their last value in the run summary."""

    source_key: str
    wandb_key: str
    multiplier: float = 1.0
    in_summary: bool = False


def _accuracy_metric_ids(evaluation_metrics: Sequence[Mapping[str, Any]], *, objective_id: str) -> tuple[str, ...]:
    """Select the ordered predictive diagnostics of a steady task from resolved metric semantics."""
    selected: list[str] = []
    for metric in evaluation_metrics:
        fields = metric["fields"]
        field_count = len(fields) if isinstance(fields, Sequence) and not isinstance(fields, str) else 0
        kind = str(metric["kind"])
        diagnostic = kind in {"group_rmse", "vector_rmse", "relative_l2", "relative_h1"} or (kind == "rmse" and field_count == 1)
        if diagnostic and str(metric["id"]) != objective_id:
            selected.append(str(metric["id"]))
    return tuple(selected)


def curated_metric_projections(
    task_id: str | None,
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    physics_monitor_enabled: bool,
    state_aux_enabled: bool,
    ood_enabled: bool,
    cuda_enabled: bool,
    optuna_trial: bool,
) -> tuple[MetricProjection, ...]:
    """Return the ordered curated history of one run; a series no producer fills is not listed."""
    if not isinstance(objective_id, str) or not objective_id:
        message = "W&B projection requires a non-empty objective identifier."
        raise ValueError(message)
    transient = task_id == "transient_drying"
    roles = _ROLE_SECTIONS if ood_enabled else _ROLE_SECTIONS[:1]
    pairs = [
        ("train/loss_total", "Overview/train_loss"),
        ("optimization/learning_rate", "Overview/learning_rate"),
        (f"id/{objective_id}", "Overview/objective_validation"),
    ]
    if ood_enabled:
        pairs += [
            (f"additional_evaluation/{objective_id}", "Overview/objective_ood"),
            ("additional_evaluation/objective_difference", "Overview/ood_gap"),
        ]
    if transient:
        pairs += [(f"train/loss_data_{name}", f"Loss/{name}") for name in _TRANSIENT_FIELDS]
        if state_aux_enabled:
            pairs.append(("train/loss_state_aux", "Loss/state_aux"))
        for role, section in roles:
            pairs += [(f"{role}/{objective_id}/component/{source}", f"Accuracy/{section}/{label}") for source, label in _TRANSIENT_ACCURACY]
        pairs += [("transient/curriculum_max_horizon", "Curriculum/horizon"), ("transient/self_fed_stage", "Curriculum/self_fed_stage")]
    else:
        pairs.append(("train/loss_data", "Loss/data"))
        if physics_training_enabled:
            pairs += [(f"physics/train/loss_{term}", f"Loss/{term}") for term in _PHYSICS_TERMS]
        accuracy_ids = _accuracy_metric_ids(evaluation_metrics, objective_id=objective_id)
        for role, section in roles:
            pairs += [(f"{role}/{metric_id}", f"Accuracy/{section}/{metric_id}") for metric_id in accuracy_ids]
        if physics_monitor_enabled:
            pairs += [(f"physics/id/{suffix}", f"Physics/Validation/{label}") for suffix, label in PHYSICS_VALIDATION_METRICS.items()]
        if physics_training_enabled:
            pairs += [(f"physics/train/{term}_weight", f"Physics/Weight/{term}") for term in _PHYSICS_TERMS]
    pairs += [
        ("system/epoch_duration_seconds", "Performance/epoch_seconds"),
        ("system/train_duration_seconds", "Performance/train_seconds"),
        ("system/train_samples_per_second", "Performance/samples_per_second"),
    ]
    projections = [MetricProjection(source, key) for source, key in pairs]
    if cuda_enabled:
        projections.append(MetricProjection("system/cuda_peak_memory_allocated_bytes", "Performance/cuda_peak_memory_gib", multiplier=1.0 / _GIB))
    if optuna_trial:
        projections += [
            MetricProjection("optuna/objective", "Optuna/objective", in_summary=True),
            MetricProjection("optuna/best_objective", "Optuna/best_objective", in_summary=True),
        ]
    source_keys = [projection.source_key for projection in projections]
    destinations = [projection.wandb_key for projection in projections]
    if len(source_keys) != len(set(source_keys)) or len(destinations) != len(set(destinations)):
        message = "Curated W&B projections must have unique source and destination keys."
        raise RuntimeError(message)
    return tuple(projections)


def run_organization(config: Mapping[str, Any], *, workflow: str) -> tuple[str | None, str, str]:
    """Return the W&B group, job type and display name of one run.

    Training runs of both tasks share ``<family>/<config name>`` across seeds, stages and
    pairs; an Optuna study is ``<family>/<study directory>``, whose budget and revision keep
    studies apart. The job type is the role: ``train`` (steady), ``stage_a0``, ``stage_b``,
    ``stage_a1`` (transient) or ``optuna_trial``. Other workflows have no group.
    """
    layout = config["run"]["layout"]
    name = str(config["run"]["name"])
    if workflow == "optuna_trial":
        study = common.paths.study_group_name(layout["config_name"], layout["budget_epochs"], layout["revision"])
        return f"{layout['family']}/{study}", workflow, f"{study}__trial-{str(layout['trial']).split('__', 1)[0]}"
    if workflow != "train":
        return None, workflow, name
    group = f"{layout['family']}/{layout['config_name']}"
    if config["task"] != "transient_drying":
        return group, workflow, name
    training = config["training"]
    job_type = _STAGE_JOB_TYPES.get((training.get("stage"), training.get("comparison_arm")))
    if job_type is None:
        message = f"Unsupported transient training stage/arm for W&B organization: {training.get('stage')!r}/{training.get('comparison_arm')!r}."
        raise ValueError(message)
    return group, job_type, name
