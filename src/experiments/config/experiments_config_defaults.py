"""Define generic runtime defaults around task-owned semantic contracts.

Responsibilities:
  - Provide generic run, data-loader, optimizer, and scheduler defaults
  - Project task losses and metric definitions into generic config defaults
  - Return serializable defaults for strict experiment resolution

Design principles:
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

RUN_DEFAULTS: dict[str, Any] = {
    "seed": 9,
    "revision": None,
    "device": "auto",
}

DATA_RUNTIME_DEFAULTS: dict[str, Any] = {
    "additional_evaluation_datasets": [],
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
}

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

OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    "adamw": {
        "kind": "adamw",
        "betas": [0.9, 0.999],
        "second_moment_floor": 1.0e-6,
    }
}

SCHEDULER_DEFAULTS: dict[str, dict[str, Any]] = {
    "reduce_on_plateau": {"kind": "reduce_on_plateau"},
}

WANDB_ENTITY = "Rinovative-Hub"
WANDB_TASK_PROJECTS = {
    "steady_flow": "grainlegumes-pino-airflow",
    "transient_drying": "grainlegumes-pino-drying",
}
WANDB_MAX_TAGS = 2
WANDB_METRIC_SCHEMA_VERSION = 1
WANDB_WORKFLOWS = (
    "train",
    "optuna_trial",
    "tracking_validation",
)

TRACKING_DEFAULTS: dict[str, Any] = {
    "wandb": {
        "mode": "online",
        "workflow": "train",
        "project": WANDB_TASK_PROJECTS["steady_flow"],
        "entity": WANDB_ENTITY,
        "tags": [],
        "monitor": {
            "enabled": True,
            "interval": 5,
            "max_cases": 4,
        },
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
        Serializable runtime defaults with task-owned scientific semantics.

    Raises
    ------
    ValueError
        If `task_id` is not registered.

    """
    task = domain.tasks.registry.get_task(task_id)
    if task_id == "transient_drying":
        return {
            "run": RUN_DEFAULTS,
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": True,
                "persistent_workers": False,
            },
            "input_profile": task.primary_input_profile,
            "temporal": {"sampling": {"mode": "one_step_transition"}, "temporal_conditioning": {"kind": "none"}},
            "scaling": {"mode": "state_std"},
            "loss": {
                "data": {
                    "kind": "huber",
                    "space": "scaled_increment",
                    "weight": 1.0,
                    "beta": 1.0,
                    "channel_weights": [1.0, 1.0, 1.0, 1.0],
                    "state_aux_weight": 0.0,
                },
                "physics": {"enabled": False},
            },
            "evaluation": {"metrics": [metric.as_dict(all_fields=task.metric_names) for metric in task.default_metrics]},
            "optimizer": OPTIMIZER_DEFAULTS["adamw"],
            "scheduler": SCHEDULER_DEFAULTS["reduce_on_plateau"],
            "training": {
                "stage": "a",
                "comparison_arm": "a0",
                "gradient_accumulation_steps": 1,
                "curriculum": {"lengths": [1], "milestone_fractions": [0.0]},
                "teacher_handoff": None,
            },
            "tracking": {
                "wandb": {
                    **TRACKING_DEFAULTS["wandb"],
                    "project": WANDB_TASK_PROJECTS[task_id],
                    "monitor": {"enabled": False, "interval": 1, "max_cases": 1},
                },
            },
        }
    return {
        "run": RUN_DEFAULTS,
        "data": dict(DATA_RUNTIME_DEFAULTS),
        "loss": {
            "data": {
                "kind": task.data_losses[0],
                "space": "normalized",
                "weight": 1.0,
            },
            "physics": dict(PHYSICS_LOSS_DEFAULTS),
        },
        "evaluation": {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
        },
        "optimizer": OPTIMIZER_DEFAULTS["adamw"],
        "scheduler": SCHEDULER_DEFAULTS["reduce_on_plateau"],
        "training": {},
        "tracking": TRACKING_DEFAULTS,
    }
