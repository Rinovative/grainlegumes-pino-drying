"""Define generic runtime defaults around task-owned semantic contracts.

Responsibilities:
  - Provide the code defaults of the keys a task's base.yaml does not state
  - Name the config paths each task's base.yaml owns, which carry no code default
  - Project task metric definitions into generic config defaults
  - Return serializable defaults for strict experiment resolution

Design principles:
  - A value stated in a task's base.yaml has one authority: the config chain, never a code default
  - Dataset selections are explicit exact immutable IDs
  - Task-fixed scientific semantics come exclusively from the registered TaskSpec
  - Runtime defaults remain independent of concrete task field names
  - Semantic identifiers are distinct from Python implementation class names

This module does NOT:
  - Parse or strictly validate user config. ``experiments.config.loader`` owns admission
  - Construct models, losses, metrics, or physics. Their registries own construction
  - Define dataset storage schemas or lifecycle behavior
"""

from __future__ import annotations

from typing import Any

from src import domain

# Config paths each task's base.yaml owns without a code default: a chain that omits one is rejected with its path.
REQUIRED_PATHS: dict[str, tuple[str, ...]] = {
    "steady_flow": (
        "run.seed",
        "run.device",
        "data.material_ood_datasets",
        "data.batch_size",
        "data.num_workers",
        "data.pin_memory",
        "data.persistent_workers",
        "loss.data.kind",
        "loss.data.space",
        "loss.data.weight",
        "optimizer.kind",
        "optimizer.weight_decay",
        "optimizer.betas",
        "optimizer.second_moment_floor",
        "scheduler.kind",
        "tracking.wandb.mode",
        "tracking.wandb.workflow",
        "tracking.wandb.monitor.interval",
    ),
    "transient_drying": (
        "run.seed",
        "run.device",
        "data.material_ood_datasets",
        "data.batch_size",
        "data.num_workers",
        "data.persistent_workers",
        "loss.data.kind",
        "loss.data.space",
        "loss.data.weight",
        "loss.data.beta",
        "loss.data.channel_weights",
        "loss.data.state_aux_weight",
        "loss.physics.enabled",
        "optimizer.kind",
        "optimizer.weight_decay",
        "scheduler.kind",
        "tracking.wandb.mode",
        "tracking.wandb.workflow",
        "tracking.wandb.monitor.enabled",
        "tracking.wandb.monitor.interval",
        "tracking.wandb.monitor.max_cases",
    ),
}

RUN_DEFAULTS: dict[str, Any] = {"revision": None}

PHYSICS_LOSS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "derivatives": {"kind": "physical"},
    "interior_crop": 2,
    "momentum_weight": {
        "target": 1.0,
        "warmup": {"kind": "linear", "epochs": 50},
    },
    "continuity_weight": {
        "target": 1.0e3,
        "warmup": {"kind": "linear", "epochs": 50},
    },
    "boundary_weight": {
        "target": 1.0e2,
        "warmup": {"kind": "linear", "epochs": 50},
    },
}

WANDB_ENTITY = "Rinovative-Hub"
WANDB_TASK_PROJECTS = {
    "steady_flow": "grainlegumes-pino-airflow",
    "transient_drying": "grainlegumes-pino-drying",
}
WANDB_MAX_TAGS = 2
WANDB_WORKFLOWS = (
    "train",
    "optuna_trial",
    "tracking_validation",
)

TRACKING_DEFAULTS: dict[str, Any] = {
    "wandb": {
        "project": WANDB_TASK_PROJECTS["steady_flow"],
        "entity": WANDB_ENTITY,
        "tags": [],
        "upload": {
            "evaluation_artifacts": False,
        },
    }
}


def get_task_defaults(task_id: str) -> dict[str, Any]:
    """
    Project one registered task into generic experiment defaults.

    Parameters
    ----------
    task_id : str
        Exact registered task identifier.

    Returns
    -------
    dict[str, Any]
        Serializable defaults of the keys the task's base.yaml does not state;
        :data:`REQUIRED_PATHS` names the ones it owns.

    Raises
    ------
    ValueError
        If `task_id` is not registered.

    """
    task = domain.tasks.registry.get_task(task_id)
    if task_id == "transient_drying":
        return {
            "run": dict(RUN_DEFAULTS),
            "data": {"pin_memory": True},
            "temporal": {"sampling": {"mode": "one_step_transition"}},
            "evaluation": {"metrics": [metric.as_dict(all_fields=task.metric_names) for metric in task.default_metrics]},
            "optimizer": {"betas": [0.9, 0.999], "second_moment_floor": 1.0e-6},
            "training": {
                "stage": "a",
                "comparison_arm": "a0",
                "gradient_accumulation_steps": 1,
                "curriculum": {"lengths": [1], "milestone_fractions": [0.0]},
                "teacher_handoff": None,
            },
            "tracking": {"wandb": {**TRACKING_DEFAULTS["wandb"], "project": WANDB_TASK_PROJECTS[task_id]}},
        }
    return {
        "run": dict(RUN_DEFAULTS),
        "loss": {"physics": dict(PHYSICS_LOSS_DEFAULTS)},
        "evaluation": {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
        },
        "training": {},
        "tracking": {"wandb": {**TRACKING_DEFAULTS["wandb"], "monitor": {"enabled": True, "max_cases": 4}}},
    }
