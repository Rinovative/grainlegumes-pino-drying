"""Build compact artificial configuration requests owned by the tests.

Responsibilities:
  - Build direct, matched transient, stage-A-only transient, steady and transient-A0 Optuna requests and write temporary YAML files
  - Derive the in-memory test-owned config source named like a request's model label and resolve requests with it

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
            "positional_embedding": None,
            "domain_padding": 0.1,
        }
    elif model_kind == "uno":
        model_params = {
            "modes_x": 4,
            "modes_y": 5,
            "hidden_channels": 8,
            "n_layers": 5,
            "mode_ratio": 0.5,
            "positional_embedding": None,
            "domain_padding": 0.1,
        }
    else:
        message = f"Unsupported synthetic model kind: {model_kind!r}"
        raise ValueError(message)

    physics: dict[str, Any] = {"enabled": False}
    if physics_enabled:
        physics = {
            "enabled": True,
            "derivatives": {"kind": "physical"},
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
            "material_ood_datasets": ["synthetic_ood"],
            "train_subset": None,
            "evaluation_batch_size": 2,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "spatial_stride": 1,
        },
        "model": {"kind": model_kind, "params": model_params},
        "loss": {
            "data": {"kind": "h1", "weight": 1.0},
            "physics": physics,
        },
        "evaluation": {"objective": {"id": "normalized_group_macro_rmse"}},
        "optimizer": {"kind": "adamw", "lr": 1.0e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "second_moment_floor": 1.0e-6},
        "scheduler": {"kind": "cosine"},
        "training": {
            "budget_epochs": 3,
            "units_per_update": 2,
            "memory_batch": 2,
        },
        "tracking": {
            "wandb": {
                "mode": wandb_mode,
                "workflow": workflow,
                "monitor": {"max_cases": 2},
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
            "domain_padding": 0.1,
        }
    if model_kind == "rno":
        return {
            "n_modes": [4, 5],
            "hidden_channels": 8,
            "n_layers": 2,
            "candidate_activation": "tanh",
            "rno_skip": False,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "channel_mlp_skip": "soft-gating",
            "fno_skip": "linear",
            "return_sequences": False,
            "positional_embedding": None,
            "domain_padding": 0.1,
        }
    if model_kind == "uno":
        return {
            "modes_x": 4,
            "modes_y": 5,
            "hidden_channels": 8,
            "n_layers": 5,
            "mode_ratio": 0.5,
            "positional_embedding": None,
            "domain_padding": 0.1,
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
            "material_ood_datasets": ["synthetic_transient_ood"],
            "train_subset": None,
            "evaluation_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "spatial_stride": 1,
        },
        "input_profile": "canonical_physics_complete",
        "temporal": temporal,
        "loss": {
            "data": {"kind": "mse", "weight": 1.0},
            "physics": {"enabled": False},
        },
        "evaluation": {
            "objective": {
                "id": "normalized_drying_group_macro_rmse",
            }
        },
        "optimizer": {"kind": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
        "scheduler": {"kind": "cosine"},
        "tracking": {
            "wandb": {
                "mode": "disabled",
                "workflow": "train",
                "monitor": {"enabled": False, "max_cases": 1},
            }
        },
        "model": {
            "kind": model_kind,
            "params": _transient_model_params(model_kind),
        },
    }


def _teacher_forced_sampling(*, model_kind: str) -> dict[str, Any]:
    """Return the A0 sampling that A1 inherits; teacher forcing draws no rollout horizon, so neither stage has a curriculum."""
    if model_kind == "rno":
        return {"mode": "trajectory", "rollout_length": 4}
    return {"mode": "one_step_transition"}


def transient_matched_config(
    *,
    model_kind: str = "fno",
    seed: int = 17,
) -> dict[str, Any]:
    """Return a compact artificial authored matched A0 -> {B, A1} plan owned by the tests."""
    a0 = {"budget_epochs": 4, "memory_batch": 2, "sampling": _teacher_forced_sampling(model_kind=model_kind)}
    b_sampling: dict[str, Any] = {"mode": "rollout_window", "rollout_length": 4, "window_stride": 4, "window_offset": 0}
    if model_kind == "rno":
        b_sampling = {"mode": "trajectory", "rollout_length": 4}
    b = {
        "lr_scale": 1.0,
        "memory_batch": 1,
        "sampling": b_sampling,
        "curriculum": {"lengths": [2, 4], "milestone_fractions": [0.0, 0.5]},
    }
    config = _transient_base(model_kind=model_kind, seed=seed, temporal={"temporal_conditioning": {"kind": "none"}})
    config["training"] = {
        "mode": "matched_a1_b",
        "units_per_update": 2,
        "a0": a0,
        "b": b,
        "a1": {"lr_scale": 1.0, "memory_batch": 2},
    }
    return config


def transient_stage_a_only_config(
    *,
    model_kind: str = "fno",
    seed: int = 17,
) -> dict[str, Any]:
    """Return a compact artificial single-stage A0 request, the form an Optuna study projects from its matched plan."""
    epochs = 4
    config = _transient_base(
        model_kind=model_kind,
        seed=seed,
        temporal={"sampling": _teacher_forced_sampling(model_kind=model_kind), "temporal_conditioning": {"kind": "none"}},
    )
    config["training"] = {
        "budget_epochs": epochs,
        "stage_schedule": {"mode": "stage_a_only", "budget_unit": "units", "budget_epochs": epochs},
        "stage": "a",
        "comparison_arm": "a0",
        "units_per_update": 2,
        "memory_batch": 2,
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
    """Return a compact artificial Optuna request with explicit search semantics; its trials train the whole budget K."""
    study: dict[str, Any] = {
        "seed": 29,
        "n_trials": 2,
        "budget_fraction": 1.0,
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


def optuna_transient_a0_config(*, model_kind: str = "fno") -> dict[str, Any]:
    """Return a compact artificial transient A0 Optuna request: the stage_a_only study mode over a matched plan."""
    request = optuna_config()
    request["study"]["mode"] = "stage_a_only"
    request["experiment"] = transient_matched_config(model_kind=model_kind)
    request["experiment"]["data"]["material_ood_datasets"] = []
    request["experiment"]["tracking"]["wandb"]["workflow"] = "optuna_trial"
    return request


def source_for(raw: Mapping[str, Any]) -> config_naming.ConfigSource:
    """Return the in-memory test-family source named like the file ``write_config`` writes for the request.

    That name is the model label and the learning rate, the two leaves such a file sets. A source without
    its file is not name-checked. A request too incomplete to label gets a placeholder source.
    """
    task = raw.get("task")
    try:
        effective = config_loader.deep_merge(config_defaults.get_task_defaults(str(task)), raw)
        name = config_naming.config_name(effective, {"model": raw["model"], "optimizer": {"lr": raw["optimizer"]["lr"]}})
        return config_naming.ConfigSource(str(task), TEST_FAMILY, name)
    except (KeyError, TypeError, ValueError):
        return config_naming.ConfigSource(str(task), TEST_FAMILY, "unlabelled")


def resolve(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Resolve a test request under its test-family source."""
    return config_loader.resolve_config(raw, source=source_for(raw), **kwargs)


def resolve_plan(raw: Mapping[str, Any], **kwargs: Any) -> transient_plan.MatchedA1BPlan:
    """Resolve a test matched plan under its test-family source."""
    return transient_plan.resolve_matched_a1_b_plan(raw, source=source_for(raw), **kwargs)


def write_config(directory: Path, payload: Mapping[str, Any], *, canonical_root: Path | None = None, family: str = TEST_FAMILY) -> Path:
    """Write an executable request as a maintained config below ``canonical_root`` (default ``directory``).

    The request lands in that root's canonical ``configs/03_learning/<task>/`` tree: a task ``base.yaml``
    with everything except the model and the learning rate, and an ``experiments/<family>/`` file that
    sets only these two, named after them as the loader requires of every config file.
    """
    source = source_for(payload)
    tree = (directory if canonical_root is None else canonical_root) / "configs" / "03_learning" / source.task
    shared = {key: value for key, value in payload.items() if key not in {"task", "model"}}
    shared["optimizer"] = {key: value for key, value in payload["optimizer"].items() if key != "lr"}
    write_yaml(tree / "base.yaml", shared)
    authored = {"model": payload["model"], "optimizer": {"lr": payload["optimizer"]["lr"]}}
    return write_yaml(tree / "experiments" / family / f"{source.name}.yaml", {"task": source.task, "base": "../../base.yaml", **authored})


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one test-owned YAML request and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path
