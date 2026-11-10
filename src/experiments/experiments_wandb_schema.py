"""Define the curated W&B history projection and run organization for both tasks.

Responsibilities:
  - Map authoritative completed-epoch metrics to the W&B sections of each task: Overview, Performance and Optuna in both,
    Training, Physics, Fields, Physical and Norms (steady), Training, Loss, Fields, Physical, Moisture and OneStep (transient)
  - Select the series that change during a run or compare ID with OOD, for the task and the run's loss, monitor, OOD and device settings
  - Derive the terminal-summary key that a projected evaluation series keeps
  - Apply presentation-only unit projections without recomputing scientific values
  - Derive group, job type and name of a run from its saved layout and training stage

Design principles:
  - Local training and evaluation metrics remain authoritative
  - W&B history is a bounded projection, not a mirror of all telemetry
  - Keys have one slash; a section is a sequence of pages of six panels in a 3 x 2 grid, and the order of the names after the slash
    equals the position order, row-major and page after page, under byte-wise, case-insensitive and punctuation-ignoring collation
    (lower-case labels, no two names first differing at ``-`` versus ``_``); projections are emitted in that same order
  - A page needs all three quantities of its comparison section; the maintained configs keep them all
  - A comparison page pairs ID with one OOD part, ID above OOD (``<part>-id-<q>`` over ``<part>-ood-<q>``); the ID series
    feeds one destination per page, and an unconfigured OOD part drops its page, and without any OOD part the ID series stand alone
  - Performance and Optuna keys carry no role token

This module does NOT:
  - Initialize W&B, persist opaque run IDs, or perform network operations
  - Recompute objectives, normalized errors, physical errors, or channel metrics
  - Change local training history, checkpoints, or summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from src import common

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

_GIB: Final = float(1024**3)
# Source role of the evaluation loaders to the label of its series in the W&B keys.
_ROLES: Final = (("id", "id"), ("material_ood", "material"), ("parameter_ood", "parameter"))
# Transient state fields as (lower-case panel label, field id), in label order.
_TRANSIENT_FIELDS: Final = (("phi", "phi"), ("temperature", "T"), ("w_surf", "w_surf"))
_PHYSICS_TERMS: Final = ("boundary", "continuity", "momentum")
_STAGE_JOB_TYPES: Final = {("a", "a0"): "stage_a0", ("b", "b"): "stage_b", ("a", "a1"): "stage_a1"}
# Compared evaluation series per section as (name, source below the role); a source without a slash is a metric id and appears
# only when the resolved metrics contain it, so the three quantities of a page stay together only for configs that keep them all.
_STEADY_COMPARISONS: Final = {
    "Fields": tuple((field, f"normalized_rmse_{field}") for field in "puv"),
    "Physical": tuple((field, f"physical_rmse_{field}") for field in "puv"),
    "Norms": (
        ("relative_h1", "normalized_relative_h1"),
        ("relative_l2", "normalized_relative_l2"),
        ("velocity", "normalized_velocity_vector_rmse"),
    ),
}
# The same for transient; ``{objective}`` is the objective id.
_TRANSIENT_COMPARISONS: Final = {
    "Fields": tuple((label, f"{{objective}}/component/{field}") for label, field in _TRANSIENT_FIELDS),
    "Physical": tuple((label, f"physical_rmse_{field}") for label, field in _TRANSIENT_FIELDS),
    "Moisture": (
        ("grain_moisture", "{objective}/component/grain_moisture_error"),
        ("w_gr_physical", "physical/w_gr_rmse"),
        ("w_int", "{objective}/component/w_int"),
    ),
}

# Source suffix below ``physics/id/`` to the term it monitors; the monitor producer owns the suffixes.
PHYSICS_VALIDATION_METRICS: Final = {
    "momentum_residual_mse": "momentum",
    "flux_balance_mse": "continuity",
    "pressure_boundary_mse": "boundary",
}


@dataclass(frozen=True, slots=True)
class MetricProjection:
    """Bind one authoritative source key to one W&B history key; only ``in_summary`` series keep their last value in the run summary."""

    source_key: str
    wandb_key: str
    multiplier: float = 1.0
    in_summary: bool = False


def selected_summary_key(source_key: str) -> str | None:
    """Return the ``selected/`` summary key under which the checkpoint evaluation publishes a projected source, if it does."""
    if source_key.startswith(tuple(f"{role}/" for role, _label in _ROLES)):
        return f"selected/{source_key}"
    if source_key.startswith("physics/id/"):
        return f"selected/physics/{source_key.removeprefix('physics/id/')}"
    if source_key.startswith("physics/train/") and source_key.endswith("_weight"):
        return f"selected/training/{source_key.removeprefix('physics/train/')}"
    return None


def _comparison_pairs(section: str, quantities: Sequence[tuple[str, str]], ood_roles: Collection[str]) -> list[tuple[str, str]]:
    """Return one page per configured OOD part: its ID series, then its OOD series, each over the section's quantities.

    Without any OOD part, the ID series stand alone as ``id-<q>``.
    """
    if not any(role in ood_roles for role, _part in _ROLES[1:]):
        return [(f"id/{source}", f"{section}/id-{name}") for name, source in quantities]
    return [
        (f"{role}/{source}", f"{section}/{part}-{label}-{name}")
        for ood_role, part in _ROLES[1:]
        if ood_role in ood_roles
        for role, label in (("id", "id"), (ood_role, "ood"))
        for name, source in quantities
    ]


def curated_metric_projections(
    task_id: str | None,
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    physics_monitor_enabled: bool,
    state_aux_enabled: bool,
    ood_roles: Collection[str],
    cuda_enabled: bool,
    optuna_trial: bool,
) -> tuple[MetricProjection, ...]:
    """Return the ordered curated history of one run; a series no producer fills is not listed.

    ``ood_roles`` names the configured (non-empty) OOD parts, ``material_ood`` and/or ``parameter_ood``; an empty part
    adds no Overview series and no comparison page;
    with neither part, each comparison section shows the ID series alone. One source may feed several destinations, the ID series of each page.

    The transient objective and metrics are configurable (``evaluation.objective.id``, ``evaluation.metrics``), so the transient series
    that name a metric id (Physical, OneStep) are filtered by the resolved metrics like the steady ones; component and moisture sources
    come from the objective's own components and the evaluation loop and are not. The three-per-row page alignment holds when a
    section keeps all three quantities, as the maintained configs do; a config that drops a metric shortens its page.
    """
    if not isinstance(objective_id, str) or not objective_id:
        message = "W&B projection requires a non-empty objective identifier."
        raise ValueError(message)
    unknown = set(ood_roles).difference(role for role, _label in _ROLES[1:])
    if unknown:
        message = f"W&B projection names unknown OOD roles {sorted(unknown)}."
        raise ValueError(message)
    transient = task_id == "transient_drying"
    roles = [(role, label) for role, label in _ROLES if role == "id" or role in ood_roles]
    pairs = [(f"{role}/{objective_id}", f"Overview/objective-{label}") for role, label in roles]
    pairs.append(("train/loss_total", "Overview/train-loss_total"))
    pairs += [(f"{role}/objective_difference", f"Overview/transfer_gap-{label}") for role, label in roles[1:]]
    if transient:
        pairs += [
            ("transient/curriculum_max_horizon", "Training/curriculum-horizon"),
            ("transient/curriculum_progress", "Training/curriculum-progress"),
            ("transient/forward_transitions", "Training/forward_transitions"),
            ("optimization/learning_rate", "Training/learning_rate"),
            ("global_step", "Training/optimizer_steps"),
            ("transient/processed_target_transitions", "Training/transitions"),
        ]
        pairs += [(f"train/loss_data_{field}", f"Loss/data-{label}") for label, field in _TRANSIENT_FIELDS[:2]]
        pairs.append(("train/loss_data", "Loss/data-total"))
        pairs += [(f"train/loss_data_{field}", f"Loss/data-{field}") for field in ("w_int", "w_surf")]
        if state_aux_enabled:
            pairs.append(("train/loss_state_aux", "Loss/state_aux"))
    else:
        pairs.append(("optimization/learning_rate", "Training/learning_rate"))
        if physics_training_enabled:
            pairs.append(("train/loss_data", "Training/loss_data"))
        pairs.append(("global_step", "Training/optimizer_steps"))
        if physics_training_enabled:
            pairs += [(f"physics/train/{term}_weight", f"Training/weight-{term}") for term in _PHYSICS_TERMS]
            pairs += [(f"physics/train/loss_{term}", f"Physics/train-{term}") for term in _PHYSICS_TERMS]
        if physics_monitor_enabled:
            suffixes = {term: suffix for suffix, term in PHYSICS_VALIDATION_METRICS.items()}
            pairs += [(f"physics/id/{suffixes[term]}", f"Physics/val-{term}") for term in _PHYSICS_TERMS]
    available = {str(metric["id"]) for metric in evaluation_metrics} - {objective_id}
    for section, quantities in (_TRANSIENT_COMPARISONS if transient else _STEADY_COMPARISONS).items():
        shown = [(name, source.format(objective=objective_id)) for name, source in quantities]
        pairs += _comparison_pairs(section, [(name, source) for name, source in shown if "/" in source or source in available], ood_roles)
    if transient:
        pairs += [
            (f"{role}/{template.format(field)}", f"OneStep/{label}-{kind}-{name}")
            for role, label in roles
            for kind, template in (("one_step", "guardrail/one_step/normalized_rmse_{}"), ("rollout", "normalized_rmse_{}"))
            for name, field in _TRANSIENT_FIELDS
            if f"normalized_rmse_{field}" in available
        ]
    pairs += [
        ("system/epoch_duration_seconds", "Performance/duration-epoch"),
        ("system/train_duration_seconds", "Performance/duration-train"),
        ("system/id_evaluation_duration_seconds", "Performance/duration-validation"),
    ]
    projections = [MetricProjection(source, key) for source, key in pairs]
    if cuda_enabled:
        projections.append(MetricProjection("system/cuda_peak_memory_allocated_bytes", "Performance/memory-cuda_peak_gib", multiplier=1.0 / _GIB))
    progress = (
        ("transient/wall_seconds", "Performance/progress-wall_seconds")
        if transient
        else ("system/estimated_remaining_seconds", "Performance/progress-remaining_seconds")
    )
    projections += [MetricProjection(*progress), MetricProjection("system/train_samples_per_second", "Performance/rate-samples_per_second")]
    if optuna_trial:
        projections.append(MetricProjection("optuna/best_objective", "Optuna/best_objective", in_summary=True))
    destinations = [projection.wandb_key for projection in projections]
    if len(destinations) != len(set(destinations)):
        message = "Curated W&B projections must have unique destination keys."
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
