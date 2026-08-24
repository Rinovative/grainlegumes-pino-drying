"""
experiments_config_defaults.py

Define generic runtime defaults around task-owned semantic contracts.

Responsibilities:
  - Provide generic run, data-loader, optimizer, scheduler, and training defaults
  - Project task losses and metric definitions into generic config defaults
  - Return serializable defaults for strict experiment resolution

Design principles:
  - Exact training and evaluation Dataset IDs are explicit experiment inputs
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
    "deterministic": True,
    "device": "auto",
    "suffix": None,
}

DATA_RUNTIME_DEFAULTS: dict[str, Any] = {
    "train_ratio": 0.8,
    "ood_fraction": 0.2,
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
}

PHYSICS_LOSS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "derivatives": {
        "kind": "spectral",
        "extension": "reflect",
    },
    "interior_crop": 2,
    "residual_weight": {
        "target": 1.0e-4,
        "warmup": {"kind": "linear", "epochs": 50},
    },
    "boundary_weight": {
        "target": 5.0e-4,
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
    "reduce_on_plateau": {
        "kind": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 20,
        "min_lr": 1.0e-8,
    }
}

TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 600,
    "evaluation_interval": 5,
    "ood_evaluation_interval": 5,
    "mixed_precision": False,
}

WANDB_ENTITY = "Rinovative-Hub"
WANDB_REPOSITORY_PROJECT = "grainlegumes-pino-drying"
WANDB_TASK_PROJECTS = {
    "steady_flow": WANDB_REPOSITORY_PROJECT,
    "transient_drying": "grainlegumes-pino-drying-transient",
}
WANDB_MAX_TAGS = 2
WANDB_WORKFLOWS = (
    "train",
    "optuna_trial",
    "gpu_smoke",
    "cpu_acceptance",
    "tracking_validation",
)

TRACKING_DEFAULTS: dict[str, Any] = {
    "wandb": {
        "mode": "online",
        "workflow": "train",
        "study": None,
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
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "spatial_stride": 1,
                "transient_backend_preference": "pt_shards",
                "transient_backend_required": False,
                "hdf5_cache_size": 0,
                "allow_technical_smoke": False,
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
                "physics": {"enabled": False, "continuity": "none"},
            },
            "evaluation": {"metrics": [metric.as_dict(all_fields=task.metric_names) for metric in task.default_metrics]},
            "optimizer": OPTIMIZER_DEFAULTS["adamw"],
            "scheduler": SCHEDULER_DEFAULTS["reduce_on_plateau"],
            "training": {
                "epochs": 100,
                "evaluation_interval": 1,
                "ood_evaluation_interval": 1,
                "mixed_precision": False,
                "stage": "a",
                "comparison_arm": "a0",
                "gradient_accumulation_steps": 1,
                "fixed_evaluation_horizon": 1,
                "curriculum": {"lengths": [1], "milestone_fractions": [0.0], "seed": 9},
                "matched_compute": {
                    "planned_seconds": None,
                    "planned_steps": None,
                    "rollout_reference_seconds": None,
                    "rollout_reference_steps": None,
                },
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
            "physics": {
                **PHYSICS_LOSS_DEFAULTS,
                "continuity": task.physics.continuity,
            },
        },
        "evaluation": {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
        },
        "optimizer": OPTIMIZER_DEFAULTS["adamw"],
        "scheduler": SCHEDULER_DEFAULTS["reduce_on_plateau"],
        "training": TRAINING_DEFAULTS,
        "tracking": TRACKING_DEFAULTS,
    }
