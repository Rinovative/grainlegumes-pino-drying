"""Define the curated W&B history projection and run organization for both tasks.

Responsibilities:
  - Map authoritative training-event metrics to the W&B sections of both tasks, in this order: Overview, Training, Loss, Residuals,
    Fields, Physical, Norms, Moisture and OneStep (transient only), Performance, Optuna; a section appears when its series exist
  - Name the summary-only keys that W&B receives as hidden metrics, so end values make no panel and no section
  - Select the series that change during a run, for the task and the run's loss, monitor and device settings
  - Derive the terminal-summary key that a projected evaluation series keeps
  - Apply presentation-only unit projections without recomputing scientific values
  - Derive group, job type and name of a run from its saved layout and training stage

Design principles:
  - Local training and evaluation metrics remain authoritative
  - W&B history is a bounded projection, not a mirror of all telemetry
  - Keys have one slash; a comparison section is a sequence of pages with one column per quantity and one row, the ID validation
    series (3 columns in most sections, 4 in steady Fields, 2 in Norms), set as that many columns in the saved view; the order of
    the names after the slash equals the position order, row-major, under byte-wise, case-insensitive and punctuation-ignoring
    collation (lower-case labels, no two names first differing at ``-`` versus ``_``); projections are emitted in that same order
  - A page needs all quantities of its comparison section; the maintained configs keep them all
  - The same name means the same logic in both tasks; the ID validation series is labelled ``id_validation``, never ``id``, which
    names the ID test of the evaluation artifacts
  - Training and Optuna evaluate no held-out role, so the history holds no OOD series; the artifact step evaluates OOD
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
    from collections.abc import Mapping, Sequence

_GIB: Final = float(1024**3)
# Keys the tracker writes only to the run summary, never to history; W&B gets them as hidden metrics (a trailing ``*`` is a prefix glob),
# so they stay in the run table and make no panel or section.
SUMMARY_ONLY_KEYS: Final = (
    "config/*",
    "data/*",
    "model/*",
    "objective/*",
    "run/*",
    "selected/*",
    "task/*",
    "terminal/*",
    "tracking/*",
    "training_work*",
    "tuning/*",
    "Optuna/pruned",
    "Optuna/state",
    "Optuna/trial_duration_seconds",
    "Optuna/trial_number",
)
# Training outcomes and run statuses that prune an Optuna trial.
PRUNED_OUTCOMES: Final = frozenset({"pruned", "nonfinite_pruned", "oom_pruned", "duplicate_pruned", "diverged", "no_eligible_checkpoint"})
# Transient state fields as (lower-case panel label, field id), in label order.
_TRANSIENT_FIELDS: Final = (("phi", "phi"), ("temperature", "T"), ("w_surf", "w_surf"))
_PHYSICS_TERMS: Final = ("boundary", "continuity", "momentum")
_STAGE_JOB_TYPES: Final = {("a", "a0"): "stage_a0", ("b", "b"): "stage_b", ("a", "a1"): "stage_a1"}
# The relative norms both tasks compare under Norms, and only there, as (name, metric id).
_RELATIVE_NORMS: Final = (("relative_h1", "physical_relative_h1"), ("relative_l2", "physical_relative_l2"))
# Compared evaluation series per section as (name, source below the role); a source without a slash is a metric id and appears
# only when the resolved metrics contain it, so the quantities of a page stay together only for configs that keep them all.
# Steady Fields holds every Train-scaled field error: p, u, v and the velocity vector of the objective's velocity group.
_STEADY_COMPARISONS: Final = {
    "Fields": (*((field, f"normalized_rmse_{field}") for field in "puv"), ("velocity", "normalized_velocity_vector_rmse")),
    "Physical": tuple((field, f"physical_rmse_{field}") for field in "puv"),
    "Norms": _RELATIVE_NORMS,
}
# The same for transient; ``{objective}`` is the objective id.
_TRANSIENT_COMPARISONS: Final = {
    "Fields": tuple((label, f"{{objective}}/component/{field}") for label, field in _TRANSIENT_FIELDS),
    "Physical": tuple((label, f"physical_rmse_{field}") for label, field in _TRANSIENT_FIELDS),
    "Norms": _RELATIVE_NORMS,
    "Moisture": (
        ("grain_moisture", "{objective}/component/grain_moisture"),
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
    """Bind one authoritative source key to one W&B history key; only ``in_summary`` series keep their last value in the run summary.

    ``validation`` marks a series that exists only in validation events (every evaluation loader, the physics monitor and the
    trial's best value); the tracker deduplicates those by point index and every other series by global step.
    """

    source_key: str
    wandb_key: str
    multiplier: float = 1.0
    in_summary: bool = False
    validation: bool = False


def _is_validation_source(source_key: str) -> bool:
    """Return whether a source key is filled only at validation points."""
    prefixes = ("id/", "physics/id/", "system/id_evaluation_")
    return source_key.startswith(prefixes) or source_key == "optuna/best_objective"


def selected_summary_key(source_key: str) -> str | None:
    """Return the ``selected/`` summary key under which the checkpoint evaluation publishes a projected source, if it does."""
    if source_key.startswith("id/"):
        return f"selected/{source_key}"
    if source_key.startswith("physics/id/"):
        return f"selected/physics/{source_key.removeprefix('physics/id/')}"
    if source_key.startswith("physics/train/") and source_key.endswith("_weight"):
        return f"selected/training/{source_key.removeprefix('physics/train/')}"
    return None


def curated_metric_projections(
    task_id: str | None,
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    physics_monitor_enabled: bool,
    cuda_enabled: bool,
    optuna_trial: bool,
    rollout_curriculum: bool,
) -> tuple[MetricProjection, ...]:
    """Return the ordered curated history of one run; a series no producer fills is not listed.

    Each comparison section shows the ID validation series alone, one column per quantity.
    ``rollout_curriculum`` marks a transient run whose stage draws rollout horizons (Stage B); only it fills the curriculum series.

    The transient objective and metrics are configurable (``evaluation.objective.id``, ``evaluation.metrics``), so the transient series
    that name a metric id (Physical, OneStep) are filtered by the resolved metrics like the steady ones; component and moisture sources
    come from the objective's own components and the evaluation loop and are not. The three-per-row page alignment holds when a
    section keeps all three quantities, as the maintained configs do; a config that drops a metric shortens its page.
    """
    if not isinstance(objective_id, str) or not objective_id:
        message = "W&B projection requires a non-empty objective identifier."
        raise ValueError(message)
    transient = task_id == "transient_drying"
    pairs = [(f"id/{objective_id}", "Overview/objective-id_validation"), ("train/loss_total", "Overview/train-loss_total")]
    # The budget fraction drives the schedule and the validation points; the committed supervised transitions are the x-axis
    # of every chart (a steady case is one), and the optimisation seconds, which budget nothing, are their own curve.
    pairs.append(("budget/progress", "Training/budget-progress"))
    if transient:
        if rollout_curriculum:
            pairs += [
                ("transient/curriculum_max_horizon", "Training/curriculum-horizon"),
                ("transient/curriculum_progress", "Training/curriculum-progress"),
            ]
        pairs += [
            ("transient/forward_transitions", "Training/forward_transitions"),
            ("optimization/learning_rate", "Training/learning_rate"),
            ("budget/train_seconds", "Training/optimisation_seconds"),
            ("global_step", "Training/optimizer_steps"),
            ("transient/processed_target_transitions", "Training/transitions"),
        ]
        pairs += [(f"train/loss_data_{field}", f"Loss/data-{label}") for label, field in _TRANSIENT_FIELDS[:2]]
        pairs.append(("train/loss_data", "Loss/data-total"))
        pairs += [(f"train/loss_data_{field}", f"Loss/data-{field}") for field in ("w_int", "w_surf")]
    else:
        pairs.append(("optimization/learning_rate", "Training/learning_rate"))
        pairs.append(("budget/train_seconds", "Training/optimisation_seconds"))
        pairs.append(("global_step", "Training/optimizer_steps"))
        pairs.append(("train/loss_data_p", "Loss/data-p"))
        pairs.append(("train/loss_data", "Loss/data-total"))
        pairs += [(f"train/loss_data_{field}", f"Loss/data-{field}") for field in ("u", "v", "velocity")]
        if physics_training_enabled:
            pairs += [(f"physics/train/{term}_weight", f"Training/weight-{term}") for term in _PHYSICS_TERMS]
            pairs += [(f"physics/train/loss_{term}", f"Loss/physics-{term}") for term in _PHYSICS_TERMS]
        if physics_monitor_enabled:
            suffixes = {term: suffix for suffix, term in PHYSICS_VALIDATION_METRICS.items()}
            pairs += [(f"physics/id/{suffixes[term]}", f"Residuals/val-{term}") for term in _PHYSICS_TERMS]
    available = {str(metric["id"]) for metric in evaluation_metrics} - {objective_id}
    for section, quantities in (_TRANSIENT_COMPARISONS if transient else _STEADY_COMPARISONS).items():
        shown = [(name, source.format(objective=objective_id)) for name, source in quantities]
        pairs += [(f"id/{source}", f"{section}/id_validation-{name}") for name, source in shown if "/" in source or source in available]
    if transient:
        pairs += [
            (f"id/{template.format(field)}", f"OneStep/id_validation-{kind}-{name}")
            for kind, template in (("one_step", "guardrail/one_step/normalized_rmse_{}"), ("rollout", "normalized_rmse_{}"))
            for name, field in _TRANSIENT_FIELDS
            if f"normalized_rmse_{field}" in available
        ]
    pairs += [
        ("system/event_duration_seconds", "Performance/duration-event"),
        ("system/train_duration_seconds", "Performance/duration-train"),
        ("system/id_evaluation_duration_seconds", "Performance/duration-validation"),
    ]
    projections = [MetricProjection(source, key, validation=_is_validation_source(source)) for source, key in pairs]
    if cuda_enabled:
        projections.append(MetricProjection("system/cuda_peak_memory_allocated_bytes", "Performance/memory-cuda_peak_gib", multiplier=1.0 / _GIB))
    projections += [
        MetricProjection("system/estimated_remaining_seconds", "Performance/progress-remaining_seconds"),
        MetricProjection("system/train_samples_per_second", "Performance/rate-units_per_second"),
    ]
    if optuna_trial:
        projections.append(MetricProjection("optuna/best_objective", "Optuna/best_objective", in_summary=True, validation=True))
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
