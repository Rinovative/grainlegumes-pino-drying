"""Build compact artificial configuration requests owned by the tests.

Responsibilities:
  - Build direct, matched transient, stage-A-only transient, steady and transient-A0 Optuna requests and write temporary YAML files
  - Derive the test-owned config source whose name equals a request's model label and resolve requests with it

Design principles:
  - Each builder returns explicit test-owned values independently of production experiment files
  - Resolution helpers only supply the source; the production loader owns every other decision

This module does NOT:
  - Allocate runs or start training
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from src.experiments.config import experiments_config_defaults as config_defaults
from src.experiments.config import experiments_config_loader as config_loader
from src.experiments.config import experiments_config_naming as config_naming
from src.experiments.config import experiments_config_transient_plan as transient_plan

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

TEST_FAMILY = "tests"


def direct_config(
    *,
    model_kind: str = "fno",
    physics_enabled: bool = False,
    device: str = "cpu",
    wandb_mode: str = "disabled",
    workflow: str = "train",
) -> dict[str, Any]:
    """Return a complete, artificial direct-training request owned by the tests."""
    if model_kind == "fno":
        model_params: dict[str, Any] = {
            "n_modes": [4, 5],
            "hidden_channels": 8,
            "n_layers": 2,
        }
    elif model_kind == "uno":
        model_params = {
            "modes_x": 4,
            "modes_y": 5,
            "hidden_channels": 8,
            "n_layers": 5,
            "mode_ratio": 0.5,
        }
    else:
        message = f"Unsupported synthetic model kind: {model_kind!r}"
        raise ValueError(message)

    physics: dict[str, Any] = {"enabled": False}
    if physics_enabled:
        physics = {
            "enabled": True,
            "derivatives": {"kind": "physical", "extension": "none"},
            "interior_crop": 1,
            "momentum_weight": {
                "target": 2.0e-3,
                "warmup": {"kind": "linear", "epochs": 2},
            },
            "continuity_weight": {
                "target": 4.0e-3,
                "warmup": {"kind": "linear", "epochs": 2},
            },
            "boundary_weight": {
                "target": 3.0e-3,
                "warmup": {"kind": "linear", "epochs": 2},
            },
        }

    return {
        "task": "steady_flow",
        "run": {
            "seed": 17,
            "deterministic": True,
            "device": device,
        },
        "data": {
            "train_dataset": "synthetic_train",
            "additional_evaluation_datasets": ["synthetic_ood"],
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"kind": model_kind, "params": model_params},
        "loss": {
            "data": {"kind": "relative_h1", "space": "normalized", "weight": 1.0},
            "physics": physics,
        },
        "evaluation": {"objective": {"id": "normalized_group_macro_rmse"}},
        "optimizer": {"kind": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
        "scheduler": {"kind": "reduce_on_plateau", "factor": 0.5, "patience": 2},
        "training": {
            "epochs": 3,
            "evaluation_interval": 1,
            "additional_evaluation_interval": 1,
            "mixed_precision": False,
        },
        "tracking": {
            "wandb": {
                "mode": wandb_mode,
                "workflow": workflow,
                "monitor": {"interval": 1, "max_cases": 2},
                "upload": {"evaluation_artifacts": False},
            }
        },
    }


def _transient_model_params(model_kind: str) -> dict[str, Any]:
    """Return the tiny test-owned transient architecture parameters."""
    if model_kind == "fno":
        return {
            "n_modes": [4, 5],
            "hidden_channels": 8,
            "n_layers": 2,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "fno_skip": "linear",
            "channel_mlp_skip": "soft-gating",
            "implementation": "factorized",
            "positional_embedding": None,
        }
    if model_kind == "rno":
        return {
            "n_modes": [4, 5],
            "hidden_channels": 8,
            "n_layers": 2,
            "rno_skip": False,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "channel_mlp_skip": "soft-gating",
            "fno_skip": "linear",
            "return_sequences": False,
            "positional_embedding": None,
        }
    if model_kind == "uno":
        return {
            "modes_x": 4,
            "modes_y": 5,
            "hidden_channels": 8,
            "n_layers": 5,
            "mode_ratio": 0.5,
        }
    message = f"Unsupported synthetic transient model kind: {model_kind!r}"
    raise ValueError(message)


def _transient_base(*, model_kind: str, seed: int, temporal: dict[str, Any]) -> dict[str, Any]:
    """Return the transient sections shared by the matched and stage-A-only builders."""
    return {
        "task": "transient_drying",
        "run": {
            "seed": seed,
            "deterministic": True,
            "device": "cpu",
        },
        "data": {
            "train_dataset": "synthetic_transient_train",
            "additional_evaluation_datasets": ["synthetic_transient_ood"],
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "transient_backend_preference": "pt_shards",
            "transient_backend_required": True,
            "hdf5_cache_size": 0,
        },
        "input_profile": "canonical_physics_complete_v1",
        "temporal": temporal,
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
        "evaluation": {
            "objective": {
                "id": "normalized_drying_group_macro_rmse",
            }
        },
        "optimizer": {"kind": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
        "scheduler": {
            "kind": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1.0e-6,
        },
        "tracking": {
            "wandb": {
                "mode": "disabled",
                "workflow": "train",
                "monitor": {"enabled": False, "interval": 1, "max_cases": 1},
            }
        },
        "model": {
            "kind": model_kind,
            "params": _transient_model_params(model_kind),
        },
    }


def _teacher_forced_stage(*, model_kind: str, seed: int) -> dict[str, Any]:
    """Return the A0 sampling and curriculum that A1 inherits."""
    if model_kind == "rno":
        return {
            "sampling": {"mode": "trajectory", "rollout_length": 4},
            "curriculum": {"lengths": [2, 4], "milestone_fractions": [0.0, 0.5], "seed": seed},
        }
    return {
        "sampling": {"mode": "one_step_transition"},
        "curriculum": {"lengths": [1], "milestone_fractions": [0.0], "seed": seed},
    }


def transient_matched_config(
    *,
    model_kind: str = "fno",
    seed: int = 17,
) -> dict[str, Any]:
    """Return a compact artificial authored matched A0 -> {B, A1} plan owned by the tests."""
    a0 = {
        "max_epochs": 4,
        "evaluation_interval": 1,
        "additional_evaluation_interval": 1,
        "gradient_accumulation_steps": 1,
        **_teacher_forced_stage(model_kind=model_kind, seed=seed),
    }
    b_sampling: dict[str, Any] = {"mode": "rollout_window", "rollout_length": 4, "window_stride": 4, "window_offset": 0}
    if model_kind == "rno":
        b_sampling = {"mode": "trajectory", "rollout_length": 4}
    b = {
        "max_epochs": 3,
        "train_seconds": 0.25,
        "lr_scale": 1.0,
        "evaluation_interval": 1,
        "additional_evaluation_interval": 1,
        "gradient_accumulation_steps": 1,
        "sampling": b_sampling,
        "curriculum": {"lengths": [2, 4], "milestone_fractions": [0.0, 0.5], "seed": seed},
    }
    config = _transient_base(model_kind=model_kind, seed=seed, temporal={"temporal_conditioning": {"kind": "none"}})
    config["training"] = {
        "mode": "matched_a1_b",
        "mixed_precision": False,
        "continuation_evaluations": 4,
        "a0": a0,
        "b": b,
        "a1": {"max_epochs": 3, "match_to": "b.train_seconds", "lr_scale": 1.0},
    }
    return config


def transient_stage_a_only_config(
    *,
    model_kind: str = "fno",
    seed: int = 17,
) -> dict[str, Any]:
    """Return a compact artificial single-stage A0 request, as an Optuna study base owns it."""
    a0 = _teacher_forced_stage(model_kind=model_kind, seed=seed)
    epochs = 4
    config = _transient_base(
        model_kind=model_kind,
        seed=seed,
        temporal={"sampling": a0["sampling"], "temporal_conditioning": {"kind": "none"}},
    )
    config["training"] = {
        "epochs": epochs,
        "stage_schedule": {"mode": "stage_a_only", "budget_unit": "epochs", "total_epochs": epochs},
        "evaluation_interval": 1,
        "additional_evaluation_interval": 1,
        "mixed_precision": False,
        "stage": "a",
        "comparison_arm": "a0",
        "gradient_accumulation_steps": 1,
        "curriculum": a0["curriculum"],
        "teacher_handoff": None,
    }
    return config


def optuna_config(
    *,
    model_kind: str = "fno",
    physics_enabled: bool = False,
    multivariate: bool = True,
    role: str | None = None,
) -> dict[str, Any]:
    """Return a compact artificial Optuna request with explicit search semantics."""
    study: dict[str, Any] = {
        "seed": 29,
        "n_trials": 2,
        "sampler": {"kind": "tpe", "multivariate": multivariate},
        "pruner": {
            "kind": "median",
            "n_startup_trials": 1,
            "n_warmup_steps": 1,
            "interval_steps": 1,
        },
    }
    if role is not None:
        study["role"] = role

    experiment = direct_config(
        model_kind=model_kind,
        physics_enabled=physics_enabled,
        workflow="optuna_trial",
    )
    return {
        "study": study,
        "experiment": experiment,
        "search_space": {
            "model.params.hidden_channels": {
                "name": "hidden_channels",
                "kind": "categorical",
                "values": [8, 12],
            },
            "optimizer.lr": {
                "name": "learning_rate",
                "kind": "float",
                "low": 1.0e-4,
                "high": 1.0e-2,
                "log": True,
            },
        },
    }


def optuna_transient_a0_config() -> dict[str, Any]:
    """Return a compact artificial transient A0 Optuna request: the stage_a_only study mode over a single-stage base."""
    request = optuna_config()
    request["study"]["mode"] = "stage_a_only"
    request["experiment"] = transient_stage_a_only_config()
    request["experiment"]["data"]["additional_evaluation_datasets"] = []
    request["experiment"]["tracking"]["wandb"]["workflow"] = "optuna_trial"
    return request


def source_for(raw: Mapping[str, Any]) -> config_naming.ConfigSource:
    """Return the test-family source named exactly like the request's model label.

    A request too incomplete to label gets a placeholder source: the resolver reports
    that request's own schema error before it ever checks the config name.
    """
    task = raw.get("task")
    try:
        effective = config_loader.deep_merge(config_defaults.get_task_defaults(str(task)), raw)
        return config_naming.ConfigSource(str(task), TEST_FAMILY, config_naming.model_label(effective))
    except (KeyError, TypeError, ValueError):
        return config_naming.ConfigSource(str(task), TEST_FAMILY, "unlabelled")


def resolve(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Resolve a test request under its test-family source."""
    return config_loader.resolve_config(raw, source=source_for(raw), **kwargs)


def resolve_plan(raw: Mapping[str, Any], **kwargs: Any) -> transient_plan.MatchedA1BPlan:
    """Resolve a test matched plan under its test-family source."""
    return transient_plan.resolve_matched_a1_b_plan(raw, source=source_for(raw), **kwargs)


def write_config(directory: Path, payload: Mapping[str, Any], *, canonical_root: Path | None = None) -> Path:
    """Write an executable request into ``directory`` under its model-label file name.

    With ``canonical_root`` the file lands in that root's canonical
    ``configs/03_learning/<task>/experiments/tests/`` tree instead.
    """
    source = source_for(payload)
    if canonical_root is not None:
        directory = canonical_root / "configs" / "03_learning" / source.task / "experiments" / source.family
    return write_yaml(directory / f"{source.name}.yaml", payload)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one test-owned YAML request and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path
