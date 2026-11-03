"""Persist and restore exact training checkpoints at admitted continuation boundaries.

Responsibilities:
  - Define the current checkpoint schema and run-identity binding
  - Capture model, optimizer, scheduler, scaler, loss, RNG, loader, and optional adapter state
  - Decide from the admitted config whether a run's best keeps the full training state
  - Validate matched-comparison batch progress and its exact sampler continuation
  - Validate checkpoint identity before mutating runtime objects
  - Publish best and last checkpoints through atomic replacement
  - Declare the per-task training code projections behind the scientific implementation digest

Design principles:
  - ``last_checkpoint.pt`` is the only continuation source and always carries the full state
  - ``best_checkpoint.pt`` is the inference and artifact source; it carries the full state only
    as the matched-A0 teacher handoff source and evaluation state otherwise
  - Ordinary resume uses completed epochs; matched comparisons may persist validated batch boundaries
  - Missing, extra, or mismatched state fails closed before runtime restoration

This module does NOT:
  - Allocate run directories, choose resume policy, or validate mutable run status
  - Construct models, optimizers, schedulers, scalers, losses, or dataloaders
  - Repair or reinterpret malformed checkpoints
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import random
import sys
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import torch
from torch import nn

from src import common, datasets

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

CheckpointRole = Literal["best", "last"]
CheckpointPayloadKind = Literal["full", "evaluation"]
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_KIND = "experiment_training_checkpoint"
_SHA256_LENGTH: Final = 64
_CHECKPOINT_IDENTITY_KEYS: Final = frozenset(
    {
        "identity_contract_sha256",
        "task",
        "task_contract_digest",
        "effective_config_digest",
        "resume_contract_digest",
        "dataset_ids",
        "dataset_fingerprints",
        "split_contract_digest",
        "split_membership_digests",
        "normalizer_sha256",
        "numerical_environment",
        "objective",
    }
)
_TRAINING_IDENTITY_CONTRACT: Final = {
    "schema_kind": "training_semantic_identity",
    "schema_version": 1,
    "model_construction": "resolved_model_kind_and_parameters",
    "loss_construction": "resolved_data_and_physics_loss",
    "optimization": "resolved_optimizer_scheduler_and_training",
    "preprocessing": "fitted_normalizer_artifact_sha256",
    "selection": "resolved_evaluation_objective_and_ordered_split",
}
TRAINING_IDENTITY_CONTRACT_DIGEST: Final = common.serialization.canonical_json_sha256(_TRAINING_IDENTITY_CONTRACT)
_CONFIG_SEMANTIC_ROOT_KEYS: Final = frozenset(
    {
        "task",
        "task_contract",
        "run",
        "data",
        "model",
        "loss",
        "evaluation",
        "optimizer",
        "scheduler",
        "training",
    }
)
_CONFIG_PROVENANCE_ROOT_KEYS: Final = frozenset({"paths", "tracking"})
_RUN_SEMANTIC_KEYS: Final = frozenset({"seed", "deterministic"})
_RUN_PROVENANCE_KEYS: Final = frozenset({"device", "name", "revision", "layout", "naming_schema_version"})
_DATA_SEMANTIC_KEYS: Final = frozenset({"train_dataset", "additional_evaluation_datasets", "batch_size"})
_DATA_EXECUTION_KEYS: Final = frozenset({"num_workers", "pin_memory", "persistent_workers"})
_TRAINING_DIRECTORIES: Final = (
    "datasets/contracts",
    "datasets/preprocessing",
    "datasets/runtime",
    "learning/losses",
    "learning/metrics",
    "learning/models",
    "learning/training",
)
_TRAINING_FILES: Final = (
    "common/common_code_identity.py",
    "common/common_random.py",
    "common/common_serialization.py",
    "datasets/packages/dataset_packages_manifest.py",
    "datasets/packages/dataset_packages_reproduction.py",
    "domain/domain_field_sets.py",
    "domain/domain_fields.py",
    "domain/tasks/domain_task_registry.py",
    "domain/tasks/domain_task_spec.py",
    "experiments/config/experiments_config_loader.py",
    "experiments/experiments_run.py",
    "generation/contracts/generation_contracts_descriptors.py",
    "generation/contracts/generation_contracts_materials.py",
    "generation/contracts/generation_contracts_porosity.py",
    "generation/contracts/generation_contracts_profiles.py",
    "generation/contracts/generation_contracts_registry.py",
    "generation/contracts/generation_contracts_vocabulary.py",
)
_TRAINING_EXEMPTIONS: Final = {
    "datasets/packages/dataset_packages_builder.py": "Dataset publication is bound by the Dataset content identity",
    "datasets/packages/dataset_packages_selection.py": "Dataset membership is bound by the Dataset content identity",
    "experiments/config/experiments_config_defaults.py": "config defaults act through the resolved-config identity",
    "experiments/config/experiments_config_naming.py": "naming acts through the resolved-config identity",
    "experiments/config/experiments_config_transient_plan.py": "matched-plan config acts through the resolved-config identity",
    "generation/cases/generation_cases_config.py": "Generation config reaches training only through Dataset content",
    "generation/contracts/generation_contracts_source.py": "producer provenance capture, not training computation",
    "learning/learning_device.py": "device choice; the numerical environment is bound separately",
    "learning/learning_device_policy.py": "device choice; the numerical environment is bound separately",
}
TRAINING_PROJECTIONS: Final = {
    "steady_flow": common.code_identity.CodeProjection(
        name="training_steady_flow",
        directories=(*_TRAINING_DIRECTORIES, "domain/physics"),
        files=(*_TRAINING_FILES, "domain/tasks/domain_task_steady_flow.py"),
        excluded_markers=("transient",),
        exemptions={
            **_TRAINING_EXEMPTIONS,
            "experiments/experiments_matched_budget.py": "transient matched-budget protocol, reached through the run module dispatch",
            "learning/learning_temporal.py": "transient time-conditioning contract, reached through the loader's transient config validation",
        },
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="training_transient_drying",
        directories=(*_TRAINING_DIRECTORIES, "learning/transient"),
        files=(
            *_TRAINING_FILES,
            "datasets/packages/dataset_packages_trajectory.py",
            "datasets/packages/dataset_packages_transient_shards.py",
            "domain/physics/domain_physics_contracts.py",
            "domain/tasks/domain_task_transient_drying.py",
            "experiments/experiments_matched_budget.py",
            "learning/learning_temporal.py",
        ),
        excluded_markers=("steady", "_pino"),
        exemptions={
            **_TRAINING_EXEMPTIONS,
            "domain/physics/domain_physics_derivatives.py": "steady PINO derivative operators, reached through the loss factory dispatch",
        },
    ),
}
_NEURALOPERATOR_PROJECTION: Final = common.code_identity.CodeProjection(name="neuraloperator", directories=(".",))
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "checkpoint_role",
        "payload_kind",
        "identity",
        "completed_epoch",
        "next_epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_enabled",
        "scaler_state_dict",
        "loss_state_dict",
        "best_metric",
        "best_epoch",
        "objective_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_loader_generator_state",
        "train_sampler_state_dict",
    }
)
# Only a full payload continues optimization. An evaluation payload keeps the remaining keys:
# completed_epoch and global_step are then the epoch (possibly a budget-ended partial one) and
# the optimizer step at which its best was evaluated, never a continuation boundary.
_CONTINUATION_KEYS: Final = frozenset(
    {
        "next_epoch",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_loader_generator_state",
        "train_sampler_state_dict",
    }
)


def _selected_config_mapping(
    value: Any,
    *,
    semantic_keys: frozenset[str],
    ignored_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    """Select declared semantic fields and reject unclassified config keys."""
    if not isinstance(value, Mapping):
        message = f"Training identity {label} must be a mapping."
        raise TypeError(message)
    unknown = sorted(set(value).difference(semantic_keys | ignored_keys))
    missing = sorted(semantic_keys.difference(value))
    if missing or unknown:
        message = f"Training identity {label} fields are unclassified: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    return {key: copy.deepcopy(value[key]) for key in sorted(semantic_keys)}


def scientific_implementation_digest(task_id: str) -> str:
    """Bind the declared task-owned scientific implementation, excluding presentation.

    The task's training projection names its declaration modules, shared model,
    loss, metric, training, preprocessing and runtime packages, and the task-specific
    files of that task; files owned by the other task do not participate. A changed
    digest detects changed implementation, without proving mathematical
    nonequivalence or completeness; the projection tests classify every static
    import of a member. The shared run and loader modules execute scientific choices
    and are hashed conservatively in full, so unrelated edits within those two
    modules may also invalidate compatibility. Separate CLI, Analysis and
    documentation modules remain outside this projection.

    Raises
    ------
    ValueError
        If ``task_id`` has no declared training projection.

    """
    projection = TRAINING_PROJECTIONS.get(task_id)
    if projection is None:
        message = f"No training implementation projection is declared for task {task_id!r}."
        raise ValueError(message)
    return projection.sha256()


def effective_config_identity_payload(
    config: Mapping[str, Any],
    *,
    include_epochs: bool = True,
) -> dict[str, Any]:
    """Return positive model-training dependencies without paths or reporting."""
    keys = set(config)
    if keys == {"task", "temporal"}:
        temporal = _required_mapping(config["temporal"], label="config temporal")
        return {
            "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
            "task": copy.deepcopy(config["task"]),
            "temporal": copy.deepcopy(dict(temporal)),
        }
    is_transient = config.get("task") == "transient_drying"
    transient_roots = frozenset({"input_profile", "temporal", "scaling"}) if is_transient else frozenset()
    allowed_roots = _CONFIG_SEMANTIC_ROOT_KEYS | _CONFIG_PROVENANCE_ROOT_KEYS | transient_roots
    required_roots = _CONFIG_SEMANTIC_ROOT_KEYS | transient_roots
    unknown = sorted(keys.difference(allowed_roots))
    missing = sorted(required_roots.difference(keys))
    if missing or unknown:
        message = f"Resolved training config fields are unclassified: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    run = _selected_config_mapping(
        config["run"],
        semantic_keys=_RUN_SEMANTIC_KEYS,
        ignored_keys=_RUN_PROVENANCE_KEYS,
        label="run",
    )
    data = _selected_config_mapping(
        config["data"],
        semantic_keys=(
            frozenset(
                {
                    "train_dataset",
                    "additional_evaluation_datasets",
                    "additional_evaluation_fraction",
                    "batch_size",
                }
            )
            if is_transient
            else _DATA_SEMANTIC_KEYS
        ),
        ignored_keys=(
            frozenset(
                {
                    "num_workers",
                    "pin_memory",
                    "persistent_workers",
                    "spatial_stride",
                    "transient_backend_preference",
                    "transient_backend_required",
                    "hdf5_cache_size",
                }
            )
            if is_transient
            else _DATA_EXECUTION_KEYS
        ),
        label="data",
    )
    configured_spatial_stride = config["data"].get("spatial_stride", 1) if is_transient else 1
    if is_transient and configured_spatial_stride != 1:
        data["spatial_stride"] = configured_spatial_stride
    task_contract = _required_mapping(config["task_contract"], label="config task_contract")
    required_task_contract = {"schema_version", "task", "digest", "data_contract_digest"}
    missing_contract = sorted(required_task_contract.difference(task_contract))
    if missing_contract:
        message = f"Training task contract is missing identity fields {missing_contract}."
        raise ValueError(message)
    training = copy.deepcopy(dict(_required_mapping(config["training"], label="config training")))
    if is_transient and training.get("stage") == "a" and training.get("comparison_arm") == "a0":
        # A0 optimization is independent of which self-fed/teacher-forced branch follows it.
        training.pop("stage_schedule", None)
        training.pop("matched_a1_b", None)
        training.pop("teacher_handoff", None)
    if not include_epochs:
        training.pop("epochs", None)
        schedule = training.get("stage_schedule")
        if isinstance(schedule, dict) and schedule.get("mode") == "matched_a1_b":
            for key in ("a0_max_epochs", "b_max_epochs", "a1_max_epochs"):
                schedule.pop(key, None)
    payload = {
        "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": copy.deepcopy(config["task"]),
        "implementation_sha256": scientific_implementation_digest(str(config["task"])),
        "task_contract": {key: copy.deepcopy(task_contract[key]) for key in sorted(required_task_contract)},
        "run": run,
        "data": data,
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
        "optimizer": copy.deepcopy(config["optimizer"]),
        "scheduler": copy.deepcopy(config["scheduler"]),
        "training": training,
    }
    if is_transient:
        payload["input_profile"] = copy.deepcopy(config["input_profile"])
        payload["temporal"] = copy.deepcopy(config["temporal"])
        payload["scaling"] = copy.deepcopy(config["scaling"])
    return payload


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the dependency-scoped identity of a completed training config."""
    return common.serialization.canonical_json_sha256(effective_config_identity_payload(config))


def resume_contract_digest(config: Mapping[str, Any]) -> str:
    """Return training semantics fixed across exact resume, excluding duration."""
    return common.serialization.canonical_json_sha256(effective_config_identity_payload(config, include_epochs=False))


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one mapping value or raise with checkpoint context."""
    if not isinstance(value, Mapping):
        msg = f"Checkpoint {label} must be a mapping."
        raise TypeError(msg)
    return value


def _required_sha256(value: Any, *, label: str) -> str:
    """Return one strict lowercase SHA-256 checkpoint identity field."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        message = f"Checkpoint {label} must be one lowercase SHA-256 digest."
        raise ValueError(message)
    return value


_OBJECTIVE_KEYS = frozenset({"id", "kind", "space", "fields", "reduction", "direction"})


def _validate_objective(value: Any, *, label: str) -> dict[str, Any]:
    """
    Validate and isolate one exact persisted objective identity.

    All six canonical fields are required, unknown fields are rejected, ordered
    output fields must be unique, and direction is limited to minimize/maximize.
    The returned deep copy prevents caller mutation from altering admitted state.
    """
    objective = _required_mapping(value, label=label)
    missing = sorted(_OBJECTIVE_KEYS.difference(objective))
    unknown = sorted(set(objective).difference(_OBJECTIVE_KEYS))
    if missing or unknown:
        msg = f"Checkpoint {label} objective keys do not match. Missing: {missing}. Unknown: {unknown}."
        raise ValueError(msg)
    for key in ("id", "kind", "space", "reduction"):
        if not isinstance(objective[key], str) or not objective[key]:
            msg = f"Checkpoint {label}.{key} must be a non-empty string."
            raise TypeError(msg)
    fields = objective["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) and field for field in fields):
        msg = f"Checkpoint {label}.fields must be a non-empty exact field list."
        raise TypeError(msg)
    if len(fields) != len(set(fields)):
        msg = f"Checkpoint {label}.fields contains duplicates."
        raise ValueError(msg)
    if objective["direction"] not in {"minimize", "maximize"}:
        msg = f"Checkpoint {label}.direction must be 'minimize' or 'maximize'."
        raise ValueError(msg)
    return copy.deepcopy(dict(objective))


def _validate_checkpoint_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Validate one current dependency-scoped checkpoint identity."""
    identity = _required_mapping(value, label=label)
    keys = set(identity)
    if keys != _CHECKPOINT_IDENTITY_KEYS:
        missing = sorted(_CHECKPOINT_IDENTITY_KEYS.difference(keys))
        unknown = sorted(keys.difference(_CHECKPOINT_IDENTITY_KEYS))
        message = f"Checkpoint {label} keys do not match. Missing: {missing}. Unknown: {unknown}."
        raise ValueError(message)
    if not isinstance(identity["task"], str) or not identity["task"]:
        message = f"Checkpoint {label}.task must be a non-empty string."
        raise TypeError(message)
    for key in ("task_contract_digest", "effective_config_digest", "resume_contract_digest"):
        _required_sha256(identity[key], label=f"{label}.{key}")
    environment = _required_mapping(identity["numerical_environment"], label=f"{label}.numerical_environment")
    if set(environment) != {
        "python",
        "numpy",
        "torch",
        "neuraloperator",
        "scipy",
        "h5py",
        "torch_cuda_build",
        "neuraloperator_implementation_sha256",
        "sif_sha256",
    } or any(value is not None and (not isinstance(value, str) or not value) for value in environment.values()):
        message = f"Checkpoint {label}.numerical_environment is invalid."
        raise ValueError(message)
    objective = _validate_objective(identity["objective"], label=f"{label}.objective")
    normalized_roles: dict[str, dict[str, Any]] = {}
    is_transient = identity["task"] == "transient_drying"
    # Steady additional evaluation is optional but must appear in every role mapping together.
    has_additional = is_transient or "additional_evaluation" in _required_mapping(
        identity["dataset_fingerprints"], label=f"{label}.dataset_fingerprints"
    )
    additional_roles = {"additional_evaluation"} if has_additional else set()
    membership_roles = (
        {"train", "scaling_train_one_step", "validation", "id_test", "additional_evaluation"}
        if is_transient
        else {"train", "validation", "id_test", *additional_roles}
    )
    dataset_roles = {"train", *additional_roles}
    for key, roles in (
        ("dataset_fingerprints", dataset_roles),
        ("split_membership_digests", membership_roles),
    ):
        values = _required_mapping(identity[key], label=f"{label}.{key}")
        if set(values) != roles:
            message = f"Checkpoint {label}.{key} must contain exactly values for {sorted(roles)}."
            raise ValueError(message)
        for role, digest in values.items():
            _required_sha256(digest, label=f"{label}.{key}.{role}")
        normalized_roles[key] = copy.deepcopy(dict(values))
    result = copy.deepcopy(dict(identity))
    result["objective"] = objective
    result.update(normalized_roles)
    contract_digest = _required_sha256(
        identity["identity_contract_sha256"],
        label=f"{label}.identity_contract_sha256",
    )
    if contract_digest != TRAINING_IDENTITY_CONTRACT_DIGEST:
        message = f"Checkpoint {label}.identity_contract_sha256 is not the maintained training contract."
        raise ValueError(message)
    dataset_ids = _required_mapping(identity["dataset_ids"], label=f"{label}.dataset_ids")
    if set(dataset_ids) != dataset_roles or not all(isinstance(item, str) and item for item in dataset_ids.values()):
        message = f"Checkpoint {label}.dataset_ids must contain non-empty {sorted(dataset_roles)} identifiers."
        raise ValueError(message)
    result["dataset_ids"] = copy.deepcopy(dict(dataset_ids))
    _required_sha256(identity["split_contract_digest"], label=f"{label}.split_contract_digest")
    _required_sha256(identity["normalizer_sha256"], label=f"{label}.normalizer_sha256")
    return result


def numerical_environment_identity() -> dict[str, str | None]:
    """Record the actual numerical packages used by checkpoint continuation.

    This runtime evidence is separate from scientific run paths, which may be
    resolved in a different preflight environment. It never probes CUDA hardware.
    """
    versions: dict[str, str | None] = {"python": ".".join(str(part) for part in sys.version_info[:3])}
    for name in ("numpy", "torch", "neuraloperator", "scipy", "h5py"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["torch_cuda_build"] = torch.version.cuda
    model_spec = importlib.util.find_spec("neuralop")
    if model_spec is not None and model_spec.origin is not None:
        model_root = Path(model_spec.origin).resolve().parent
        versions["neuraloperator_implementation_sha256"] = _NEURALOPERATOR_PROJECTION.sha256(model_root)
    else:
        versions["neuraloperator_implementation_sha256"] = None
    provenance = os.environ.get("ML_SLURM_PROVENANCE")
    sif_digest = json.loads(provenance).get("sif_sha256") if provenance is not None else None
    if sif_digest is not None:
        _required_sha256(sif_digest, label="runtime SIF")
    versions["sif_sha256"] = sif_digest
    return versions


def build_checkpoint_identity(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    *,
    normalizer_sha256: str,
    persisted_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build current task, config, Dataset, split, normalizer, and objective identity."""
    task = config.get("task")
    task_contract = _required_mapping(config.get("task_contract"), label="config task_contract")
    evaluation = _required_mapping(config.get("evaluation"), label="config evaluation")
    objective = _validate_objective(
        evaluation.get("objective"),
        label="config evaluation.objective",
    )
    task_digest = task_contract.get("digest")
    if not isinstance(task, str) or not task:
        message = "Config task must be a non-empty string for checkpoint identity."
        raise TypeError(message)
    task_digest = _required_sha256(task_digest, label="config task_contract.digest")
    normalizer_digest = _required_sha256(normalizer_sha256, label="normalizer_sha256")
    if task == "transient_drying":
        base_required = {
            "schema_kind",
            "schema_version",
            "task",
            "task_contract_digest",
            "data_contract_digest",
            "tensorizer",
            "sampling",
            "evaluation_sampling",
            "dataset_identity",
            "additional_evaluation_fraction",
            "case_partition",
            "case_partition_digest",
            "evaluation_assignments",
            "split_seed",
            "source_case_ids",
            "source_case_materials",
            "spatial_view",
            "roles",
            "runtime_provenance",
        }
        split_version = split_indices.get("schema_version")
        configured_stride = config["data"].get("spatial_stride", 1)
        if (
            split_indices.get("schema_kind") != datasets.runtime.transient_training.SPLIT_SCHEMA_KIND
            or type(split_version) is not int
            or split_version != datasets.runtime.transient_training.SPLIT_SCHEMA_VERSION
        ):
            message = "Historical or malformed transient checkpoint split schema is unsupported."
            raise ValueError(message)
        required = base_required
        spatial_view = _required_mapping(split_indices.get("spatial_view"), label="transient spatial_view")
        if spatial_view.get("spatial_stride") != configured_stride:
            message = "Transient checkpoint split spatial stride does not match the config."
            raise ValueError(message)
        if set(split_indices) != required:
            message = "Transient checkpoint split does not match the maintained training-split schema."
            raise ValueError(message)
        if split_indices["task"] != task or split_indices["task_contract_digest"] != task_digest:
            message = "Transient checkpoint split task identity does not match the config contract."
            raise ValueError(message)
        identities = _required_mapping(split_indices["dataset_identity"], label="transient dataset_identity")
        roles = _required_mapping(split_indices["roles"], label="transient roles")
        train_identity = _required_mapping(identities.get("train"), label="transient train identity")
        additional_identity = identities.get("additional_evaluation")
        if not isinstance(additional_identity, list):
            message = "Transient checkpoint additional evaluation Dataset identities must be a list."
            raise ValueError(message)

        def membership(role: str) -> str:
            if role == "additional_evaluation":
                added_roles = _required_mapping(roles.get(role), label="transient additional evaluation roles").get("parts")
                if not isinstance(added_roles, list):
                    message = "Transient checkpoint additional evaluation membership evidence is invalid."
                    raise ValueError(message)
                return common.serialization.canonical_json_sha256(added_roles)
            evidence = _required_mapping(roles.get(role), label=f"transient {role} role")
            return _required_sha256(evidence.get("membership_digest"), label=f"transient {role} membership_digest")

        membership_values = {role: membership(role) for role in ("train", "scaling_train_one_step", "validation", "id_test", "additional_evaluation")}
        dataset_ids = {
            "train": str(train_identity.get("dataset_id", "")),
            "additional_evaluation": common.serialization.canonical_json_sha256(additional_identity),
        }
        fingerprints = {
            "train": _required_sha256(train_identity.get("index_digest"), label="transient train index_digest"),
            "additional_evaluation": common.serialization.canonical_json_sha256(additional_identity),
        }
        scientific_split = dict(split_indices)
        scientific_split.pop("runtime_provenance")
        persisted = dict(persisted_config or config)
        return _validate_checkpoint_identity(
            {
                "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
                "task": task,
                "task_contract_digest": task_digest,
                "effective_config_digest": config_digest(persisted),
                "resume_contract_digest": resume_contract_digest(persisted),
                "dataset_ids": dataset_ids,
                "dataset_fingerprints": fingerprints,
                "split_contract_digest": common.serialization.canonical_json_sha256(scientific_split),
                "split_membership_digests": membership_values,
                "normalizer_sha256": normalizer_digest,
                "numerical_environment": numerical_environment_identity(),
                "objective": objective,
            },
            label="built identity",
        )
    split_contract = datasets.preprocessing.splits.admit_split_contract(split_indices)
    if split_contract.task != task or split_contract.task_contract_digest != task_digest:
        message = "Split task identity must match the checkpoint config contract."
        raise ValueError(message)
    sources = split_contract.sources()
    dataset_ids = {role: source.dataset_id for role, source in sources.items()}
    fingerprints = {role: source.fingerprint for role, source in sources.items()}
    split_payload = split_contract.scientific_payload()
    membership_values = split_payload["membership_digests"]
    persisted = dict(persisted_config or config)
    return _validate_checkpoint_identity(
        {
            "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
            "task": task,
            "task_contract_digest": task_digest,
            "effective_config_digest": config_digest(persisted),
            "resume_contract_digest": resume_contract_digest(persisted),
            "dataset_ids": dataset_ids,
            "dataset_fingerprints": fingerprints,
            "split_contract_digest": common.serialization.canonical_json_sha256(split_payload),
            "split_membership_digests": membership_values,
            "normalizer_sha256": normalizer_digest,
            "numerical_environment": numerical_environment_identity(),
            "objective": objective,
        },
        label="built identity",
    )


def _train_loader_generator(train_loader: DataLoader[Any]) -> torch.Generator:
    """Return the explicit generator controlling shuffled train membership."""
    generator = getattr(train_loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        sampler = getattr(train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
    if not isinstance(generator, torch.Generator):
        msg = "Exact resume requires an explicit torch.Generator on the training DataLoader."
        raise TypeError(msg)
    return generator


def _stateful_sampler(train_loader: DataLoader[Any]) -> Any | None:
    """Return the loader's state-carrying batch or item sampler, if it has one."""
    for sampler in (getattr(train_loader, "batch_sampler", None), getattr(train_loader, "sampler", None)):
        if callable(getattr(sampler, "state_dict", None)):
            return sampler
    return None


def _sampler_state(train_loader: DataLoader[Any]) -> Mapping[str, Any] | None:
    """Return optional explicit sampler state without inventing sampler state."""
    sampler = _stateful_sampler(train_loader)
    if sampler is None:
        return None
    state = sampler.state_dict()
    if not isinstance(state, Mapping):
        msg = "Training sampler state_dict() must return a mapping."
        raise TypeError(msg)
    return dict(state)


def _cuda_device_index(device: torch.device) -> int:
    """Return the already concrete CUDA index without querying runtime defaults."""
    if device.type != "cuda" or device.index is None:
        msg = f"Checkpoint CUDA state requires an indexed concrete CUDA device, got {device}."
        raise ValueError(msg)
    return device.index


def _active_cuda_devices(
    model: nn.Module,
    *,
    runtime_device: torch.device | None = None,
) -> tuple[torch.device, ...]:
    """
    Discover sorted concrete CUDA devices owned by this runtime or model.

    Only an explicitly supplied indexed runtime device and devices already
    owning model parameters or buffers participate. The helper never queries a
    process-default CUDA device or initializes availability-based fallback.
    """
    indices: set[int] = set()
    if runtime_device is not None:
        if not isinstance(runtime_device, torch.device):
            msg = f"runtime_device must be a concrete torch.device, got {runtime_device!r}."
            raise TypeError(msg)
        if runtime_device.type == "cuda":
            indices.add(_cuda_device_index(runtime_device))
    for tensors in (model.parameters(), model.buffers()):
        for tensor in tensors:
            if tensor.device.type == "cuda":
                indices.add(_cuda_device_index(tensor.device))
    return tuple(torch.device("cuda", index) for index in sorted(indices))


def _capture_cuda_rng_states(devices: tuple[torch.device, ...]) -> list[torch.Tensor]:
    """Capture RNG state only for CUDA devices used by this runtime."""
    return [torch.cuda.get_rng_state(device).clone() for device in devices]


def _set_cuda_rng_states(states: list[torch.Tensor], devices: tuple[torch.device, ...]) -> None:
    """Restore ordered CUDA RNG states onto the current active devices."""
    for state, device in zip(states, devices, strict=True):
        torch.cuda.set_rng_state(state, device=device)


def _attempt_rollback(
    label: str,
    action: Callable[[], Any],
    errors: list[str],
) -> None:
    """
    Attempt one transactional rollback action and retain any failure.

    Arbitrary component exceptions are converted to bounded diagnostic strings
    so later rollback actions still run. The original restore error remains the
    primary exception unless the collected failures make rollback incomplete.
    """
    try:
        action()
    except Exception as error:  # noqa: BLE001 -- rollback must retain arbitrary component failures
        errors.append(f"{label}: {type(error).__name__}: {error}")


def is_teacher_handoff_source(config: Mapping[str, Any]) -> bool:
    """
    Return whether an admitted run's best checkpoint is the teacher handoff source.

    Exactly the transient A0 of a ``matched_a1_b`` schedule publishes its best for
    B/A1 continuation, so only its best keeps the full training state. Steady runs,
    B, A1, and stage-A-only A0 never branch. The decision reads the admitted
    configuration, never a checkpoint payload.
    """
    if config["task"] != "transient_drying":
        return False
    training = config["training"]
    return bool(training["comparison_arm"] == "a0" and training["stage_schedule"]["mode"] == "matched_a1_b")


def best_payload_kind(config: Mapping[str, Any]) -> CheckpointPayloadKind:
    """Return the payload kind every best checkpoint of one admitted run must carry."""
    return "full" if is_teacher_handoff_source(config) else "evaluation"


def make_checkpoint(
    *,
    role: CheckpointRole,
    payload_kind: CheckpointPayloadKind,
    identity: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    loss: nn.Module,
    best_metric: float | None,
    best_epoch: int | None,
    objective_history: list[dict[str, Any]],
    train_loader: DataLoader[Any],
    runtime_device: torch.device,
    adapter: Any | None = None,
    batch_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Capture a full continuation or an evaluation-only checkpoint payload.

    Parameters
    ----------
    role : {"best", "last"}
        Enforced checkpoint lifecycle role.
    payload_kind : {"full", "evaluation"}
        ``"full"`` for every last checkpoint and the best of
        :func:`is_teacher_handoff_source` runs; ``"evaluation"`` omits optimizer,
        scheduler, scaler, RNG, loader, ``next_epoch``, and batch progress.
    identity : Mapping[str, Any]
        Immutable run identity from :func:`build_checkpoint_identity`.
    completed_epoch : int
        Positive one-based epoch that completed fully.
    global_step : int
        Number of completed optimizer steps.
    model, optimizer, scheduler, scaler, loss : Any
        Stateful runtime objects.
    amp_enabled : bool
        Whether mixed-precision scaling is active.
    best_metric : float | None
        Best finite objective so far, if any.
    best_epoch : int | None
        One-based epoch owning ``best_metric``.
    objective_history : list[dict[str, Any]]
        Ordered finite evaluation history.
    train_loader : DataLoader
        Shuffled loader whose generator state controls the next epoch.
    runtime_device : torch.device
        Active concrete execution device. CUDA RNG is captured only for this device and
        any CUDA devices that actually own model parameters or buffers.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with data_processor where accepted.
    batch_progress : Mapping[str, Any] | None, optional
        Exact completed optimizer and partial-epoch progress for adapter
        continuation; only a full payload records it.

    Returns
    -------
    dict[str, Any]
        Complete validated checkpoint.

    """
    if role not in {"best", "last"}:
        msg = f"Unknown checkpoint role {role!r}."
        raise ValueError(msg)
    if payload_kind not in {"full", "evaluation"} or (role == "last" and payload_kind != "full"):
        msg = f"Checkpoint payload kind {payload_kind!r} is invalid for role {role!r}; a last checkpoint is always full."
        raise ValueError(msg)
    if payload_kind == "evaluation" and batch_progress is not None:
        msg = "An evaluation checkpoint records no batch continuation progress."
        raise ValueError(msg)
    if completed_epoch <= 0:
        msg = f"completed_epoch must be positive, got {completed_epoch}."
        raise ValueError(msg)
    if global_step < 0:
        msg = f"global_step must be non-negative, got {global_step}."
        raise ValueError(msg)
    if (best_metric is None) != (best_epoch is None):
        msg = "best_metric and best_epoch must either both be present or both be absent."
        raise ValueError(msg)
    if best_metric is not None and not math.isfinite(float(best_metric)):
        msg = f"best_metric must be finite, got {best_metric}."
        raise ValueError(msg)
    if role == "best" and best_metric is None:
        msg = "A best checkpoint requires a finite best objective."
        raise ValueError(msg)
    if not isinstance(runtime_device, torch.device) or runtime_device.type not in {"cpu", "cuda"}:
        msg = f"Checkpoint capture requires one concrete CPU or CUDA torch.device, got {runtime_device!r}."
        raise TypeError(msg)
    if amp_enabled and scaler is None:
        msg = "AMP-enabled checkpoints require a scaler."
        raise ValueError(msg)
    if amp_enabled and runtime_device.type != "cuda":
        msg = "AMP-enabled checkpoints require a concrete CUDA runtime device."
        raise ValueError(msg)

    validated_identity = _validate_checkpoint_identity(identity, label="identity")
    adapter_state: dict[str, Any] | None = None
    if adapter is not None:
        raw_adapter_state = adapter.state_dict()
        if not isinstance(raw_adapter_state, Mapping):
            msg = "Checkpoint adapter.state_dict() must return a mapping."
            raise TypeError(msg)
        adapter_state = dict(raw_adapter_state)
    payload: dict[str, Any] = {
        "schema_kind": CHECKPOINT_SCHEMA_KIND,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "payload_kind": payload_kind,
        "identity": validated_identity,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "amp_enabled": bool(amp_enabled),
        "loss_state_dict": loss.state_dict(),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": best_epoch,
        "objective_history": copy.deepcopy(objective_history),
    }
    if payload_kind == "full":
        generator = _train_loader_generator(train_loader)
        active_cuda_devices = _active_cuda_devices(model, runtime_device=runtime_device)
        payload |= {
            "next_epoch": completed_epoch + (1 if batch_progress is None or batch_progress["complete"] else 0),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),  # noqa: NPY002 -- exact process-global RNG state
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": _capture_cuda_rng_states(active_cuda_devices),
            "train_loader_generator_state": generator.get_state(),
            "train_sampler_state_dict": _sampler_state(train_loader),
        }
    if adapter_state is not None:
        payload["adapter_state_dict"] = copy.deepcopy(adapter_state)
    if batch_progress is not None:
        payload["batch_progress"] = copy.deepcopy(dict(batch_progress))
    return payload


def records_completed_budget(payload: Mapping[str, Any]) -> bool:
    """
    Return whether a checkpoint's adapter controller records a completed stage budget.

    A completed budget is terminal: such a checkpoint finalizes its run and
    never continues optimization.
    """
    adapter_state = payload.get("adapter_state_dict")
    controller = adapter_state.get("controller") if isinstance(adapter_state, Mapping) else None
    return isinstance(controller, Mapping) and controller.get("budget_complete") is True


def recorded_history_epochs(payload: Mapping[str, Any]) -> int:
    """
    Return how many epoch history records precede one validated last checkpoint.

    An operational partial epoch publishes its checkpoint before any history
    record; a terminal budget epoch records its history before its checkpoints.
    """
    completed_epoch = int(payload["completed_epoch"])
    progress = payload.get("batch_progress")
    if progress is not None and not progress["complete"] and not records_completed_budget(payload):
        return completed_epoch - 1
    return completed_epoch


def validate_checkpoint(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    payload_kind: CheckpointPayloadKind,
    scheduler_expected: bool,
    amp_expected: bool,
    require_best: bool,
    adapter_expected: bool = False,
) -> dict[str, Any]:
    """
    Validate a checkpoint completely before runtime state restoration.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Untrusted loaded checkpoint mapping.
    expected_identity : Mapping[str, Any]
        Exact task/config/dataset/split/objective identity for the current run.
    expected_role : {"best", "last"}
        Required lifecycle role.
    payload_kind : {"full", "evaluation"}
        Required payload kind derived from the admitted config: ``"full"`` for
        every last checkpoint, :func:`best_payload_kind` for a best checkpoint.
    scheduler_expected : bool
        Whether scheduler state must be present.
    amp_expected : bool
        Whether AMP and scaler state must be present.
    require_best : bool
        Whether finite best-objective state is mandatory.

    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.

    Returns
    -------
    dict[str, Any]
        Isolated shallow mapping admitted under the current schema.

    Raises
    ------
    TypeError
        If required mappings, scalar types, or RNG/sampler state shapes are invalid.
    ValueError
        If schema, identity, role, progress, best selection, or component presence
        contradicts the expected run contract.

    Notes
    -----
    Validation mutates no runtime object. RNG payloads are tested on temporary
    generators before the checkpoint can reach transactional restoration. An
    evaluation payload keeps identity, role, objective-history, and selected-best
    checks; only continuation checks are absent because it has no continuation state.

    """
    schema_version = payload.get("schema_version")
    if payload.get("schema_kind") != CHECKPOINT_SCHEMA_KIND or type(schema_version) is not int or schema_version != CHECKPOINT_SCHEMA_VERSION:
        msg = (
            "Historical or malformed checkpoint schema_kind/schema_version is unsupported: "
            f"got {payload.get('schema_kind')!r}/{schema_version!r}, admitted {CHECKPOINT_SCHEMA_KIND!r}/{CHECKPOINT_SCHEMA_VERSION}. "
            "Regenerate the run."
        )
        raise ValueError(msg)
    if payload.get("checkpoint_role") != expected_role:
        msg = f"Checkpoint role mismatch: expected {expected_role!r}, got {payload.get('checkpoint_role')!r}."
        raise ValueError(msg)
    if payload_kind not in {"full", "evaluation"} or (expected_role == "last" and payload_kind != "full"):
        msg = f"Expected checkpoint payload kind {payload_kind!r} is invalid for role {expected_role!r}; a last checkpoint is always full."
        raise ValueError(msg)
    if payload.get("payload_kind") != payload_kind:
        msg = f"Checkpoint payload_kind mismatch: expected {payload_kind!r}, got {payload.get('payload_kind')!r}."
        raise ValueError(msg)
    full = payload_kind == "full"
    has_batch_progress = full and "batch_progress" in payload
    expected_keys = (
        (_CHECKPOINT_KEYS if full else _CHECKPOINT_KEYS - _CONTINUATION_KEYS)
        | ({"adapter_state_dict"} if adapter_expected else set())
        | ({"batch_progress"} if has_batch_progress else set())
    )
    missing = sorted(expected_keys.difference(payload))
    unknown = sorted(set(payload).difference(expected_keys))
    if missing or unknown:
        msg = f"Checkpoint {payload_kind} payload keys mismatch. Missing keys: {missing}. Unknown keys: {unknown}."
        raise ValueError(msg)
    if adapter_expected and not isinstance(payload["adapter_state_dict"], Mapping):
        msg = "Transient checkpoint adapter_state_dict must be a mapping."
        raise TypeError(msg)
    identity = _validate_checkpoint_identity(payload["identity"], label="identity")
    expected = _validate_checkpoint_identity(expected_identity, label="expected identity")
    comparison_identity = copy.deepcopy(identity)
    comparison_expected = copy.deepcopy(expected)
    actual_sif = comparison_expected["numerical_environment"]["sif_sha256"]
    saved_sif = comparison_identity["numerical_environment"]["sif_sha256"]
    if actual_sif is None or saved_sif is None:
        comparison_identity["numerical_environment"]["sif_sha256"] = None
        comparison_expected["numerical_environment"]["sif_sha256"] = None
    if comparison_identity != comparison_expected:
        msg = "Checkpoint run identity is incompatible with config, task, dataset, split, normalizer, or objective state."
        raise ValueError(msg)

    completed_epoch = payload["completed_epoch"]
    global_step = payload["global_step"]
    integer_fields = [("completed_epoch", completed_epoch, 1), ("global_step", global_step, 0)]
    if full:
        integer_fields.append(("next_epoch", payload["next_epoch"], 1 if has_batch_progress else 2))
    for label, value, minimum in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            msg = f"Checkpoint {label} must be an integer >= {minimum}."
            raise TypeError(msg)
    progress = payload.get("batch_progress")
    if has_batch_progress:
        if not isinstance(progress, Mapping) or set(progress) != {"epoch", "cursor", "size", "complete"}:
            msg_0 = "Matched batch progress schema is invalid."
            raise ValueError(msg_0)
        # Windowed trajectories consume their loader items only after the last window, so a
        # budget crossing inside the epoch's first loader batch leaves a terminal zero cursor.
        cursor_floor = 0 if records_completed_budget(payload) else 1
        if (
            progress["epoch"] != completed_epoch
            or type(progress["cursor"]) is not int
            or type(progress["size"]) is not int
            or not cursor_floor <= progress["cursor"] <= progress["size"]
            or type(progress["complete"]) is not bool
            or progress["complete"] != (progress["cursor"] == progress["size"])
        ):
            msg_0 = "Matched batch progress does not describe one valid epoch cursor."
            raise ValueError(msg_0)
    if full and payload["next_epoch"] != completed_epoch + (1 if progress is None or progress["complete"] else 0):
        msg = "Checkpoint next_epoch conflicts with its completed or partial batch boundary."
        raise ValueError(msg)

    for key in ("model_state_dict", "loss_state_dict", *(("optimizer_state_dict",) if full else ())):
        _required_mapping(payload[key], label=key)
    if full and scheduler_expected != isinstance(payload["scheduler_state_dict"], Mapping):
        msg = "Checkpoint scheduler state is absent or present inconsistently with the resolved config."
        raise ValueError(msg)
    amp_enabled = payload["amp_enabled"]
    if not isinstance(amp_enabled, bool) or amp_enabled != amp_expected:
        msg = f"Checkpoint AMP mode mismatch: expected {amp_expected}, got {amp_enabled!r}."
        raise ValueError(msg)
    if full and amp_expected != isinstance(payload["scaler_state_dict"], Mapping):
        msg = "Checkpoint scaler state is absent or present inconsistently with active AMP."
        raise ValueError(msg)

    best_metric = payload["best_metric"]
    best_epoch = payload["best_epoch"]
    if (best_metric is None) != (best_epoch is None):
        msg = "Checkpoint best_metric/best_epoch presence is inconsistent."
        raise ValueError(msg)
    if require_best and best_metric is None:
        msg = "Checkpoint requires a finite best objective but none is stored."
        raise ValueError(msg)
    if best_metric is not None:
        if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)) or not math.isfinite(float(best_metric)):
            msg = "Checkpoint best_metric must be finite."
            raise ValueError(msg)
        if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 1 <= best_epoch <= completed_epoch:
            msg = "Checkpoint best_epoch must identify a completed epoch."
            raise ValueError(msg)

    history = payload["objective_history"]
    if not isinstance(history, list):
        msg = "Checkpoint objective_history must be a list."
        raise TypeError(msg)
    history_values: list[tuple[int, float]] = []
    previous_epoch = 0
    for entry in history:
        if not isinstance(entry, Mapping) or set(entry) != {"epoch", "objective_id", "value"}:
            msg = "Checkpoint objective_history entries must contain exactly epoch, objective_id, and value."
            raise ValueError(msg)
        value = entry["value"]
        epoch = entry["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not previous_epoch < epoch <= completed_epoch:
            msg = "Checkpoint objective_history epochs must be strictly increasing completed epochs."
            raise ValueError(msg)
        if entry["objective_id"] != identity["objective"]["id"]:
            msg = "Checkpoint objective_history id does not match checkpoint identity."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = "Checkpoint objective_history values must be finite."
            raise ValueError(msg)
        history_values.append((epoch, float(value)))
        previous_epoch = epoch
    if (best_metric is None) != (not history_values):
        msg = "Checkpoint best state and objective_history presence are inconsistent."
        raise ValueError(msg)
    if best_metric is not None:
        selected = (
            min(history_values, key=lambda item: item[1])
            if identity["objective"]["direction"] == "minimize"
            else max(history_values, key=lambda item: item[1])
        )
        if selected != (best_epoch, float(best_metric)):
            msg = "Checkpoint best state does not match its direction-aware objective history."
            raise ValueError(msg)
        if expected_role == "best" and completed_epoch != best_epoch:
            msg = "A best checkpoint must be captured at its selected best epoch."
            raise ValueError(msg)
    if not full:
        return dict(payload)

    python_rng_state = payload["python_rng_state"]
    numpy_rng_state = payload["numpy_rng_state"]
    if not isinstance(python_rng_state, tuple):
        msg = "Checkpoint Python RNG state must be a tuple."
        raise TypeError(msg)
    if not isinstance(numpy_rng_state, tuple):
        msg = "Checkpoint NumPy RNG state must be a tuple."
        raise TypeError(msg)
    try:
        random.Random().setstate(python_rng_state)  # noqa: S311 -- validate saved non-cryptographic RNG state
        np.random.RandomState().set_state(numpy_rng_state)
    except (TypeError, ValueError) as error:
        msg = "Checkpoint Python or NumPy RNG state is invalid."
        raise ValueError(msg) from error
    for key in ("torch_cpu_rng_state", "train_loader_generator_state"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
            msg = f"Checkpoint {key} must be a one-dimensional CPU byte tensor."
            raise TypeError(msg)
        try:
            torch.Generator().set_state(value)
        except RuntimeError as error:
            msg = f"Checkpoint {key} is not a valid Torch generator state."
            raise ValueError(msg) from error
    cuda_states = payload["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.device.type == "cpu" and state.dtype == torch.uint8 and state.ndim == 1 for state in cuda_states
    ):
        msg = "Checkpoint CUDA RNG states must be one-dimensional CPU byte tensors."
        raise TypeError(msg)
    sampler_state = payload["train_sampler_state_dict"]
    if sampler_state is not None and not isinstance(sampler_state, Mapping):
        msg = "Checkpoint train_sampler_state_dict must be a mapping or None."
        raise TypeError(msg)
    if has_batch_progress and (
        not isinstance(progress, Mapping)
        or not isinstance(sampler_state, Mapping)
        or sampler_state.get("cursor") != progress["cursor"]
        or sampler_state.get("size") != progress["size"]
        or sampler_state.get("epoch") != completed_epoch - 1
    ):
        msg_0 = "Matched checkpoint batch progress disagrees with its exact sampler cursor."
        raise ValueError(msg_0)
    return dict(payload)


def save_checkpoint(payload: Mapping[str, Any], path: Path | str) -> Path:
    """
    Atomically publish one already validated checkpoint payload.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Complete best- or last-checkpoint mapping.
    path : Path | str
        Exact role-qualified checkpoint path.

    Returns
    -------
    Path
        Atomically published checkpoint path.

    """
    return common.serialization.atomic_torch_save(dict(payload), path)


def load_checkpoint(
    path: Path | str,
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    payload_kind: CheckpointPayloadKind,
    scheduler_expected: bool,
    amp_expected: bool,
    require_best: bool,
    adapter_expected: bool = False,
) -> dict[str, Any]:
    """
    Load and validate one checkpoint under the strict saved schema.

    Parameters
    ----------
    path : Path | str
        Required checkpoint path.
    expected_identity : Mapping[str, Any]
        Exact task/config/dataset/split/objective identity.
    expected_role : {"best", "last"}
        Required lifecycle role.
    payload_kind : {"full", "evaluation"}
        Required payload kind; see :func:`validate_checkpoint`.
    scheduler_expected : bool
        Whether scheduler state is required.
    amp_expected : bool
        Whether scaler state is required.
    require_best : bool
        Whether finite selected-best state must already exist.

    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.

    Returns
    -------
    dict[str, Any]
        Fully validated checkpoint mapping.

    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        msg = f"Required {expected_role} checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        msg = f"Checkpoint must contain a mapping: {checkpoint_path}"
        raise TypeError(msg)
    return validate_checkpoint(
        payload,
        expected_identity=expected_identity,
        expected_role=expected_role,
        payload_kind=payload_kind,
        scheduler_expected=scheduler_expected,
        amp_expected=amp_expected,
        require_best=require_best,
        adapter_expected=adapter_expected,
    )


def restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    loss: nn.Module,
    train_loader: DataLoader[Any],
    adapter: Any | None = None,
    adapter_expected: bool = False,
) -> dict[str, Any]:
    """
    Restore a validated last checkpoint transactionally.

    Schema, identity, runtime capability, and rollback-state checks happen
    before mutation. If any component, loader, sampler, or RNG restore fails,
    every supplied runtime object and process RNG is restored to its exact
    pre-call state before the original failure is re-raised.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Candidate strict ``last`` checkpoint.
    expected_identity : Mapping[str, Any]
        Immutable identity required by the current resumed run.
    model, optimizer, scheduler, scaler, loss : Any
        Current runtime components whose state may be replaced.
    amp_enabled : bool
        Whether scaler state is required by the resumed runtime.
    train_loader : DataLoader
        Loader owning the explicit shuffle generator and optional sampler state.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with data_processor where accepted.
    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.

    Returns
    -------
    dict[str, Any]
        Restored progress, best-objective state, and isolated objective history.

    Raises
    ------
    TypeError
        If checkpoint or current loader/sampler capabilities cannot support exact restore.
    RuntimeError
        If CUDA state cardinality differs or rollback after a restore failure is incomplete.

    Notes
    -----
    Process-global Python, NumPy, Torch CPU, and active-device CUDA RNG states
    are part of both the restore transaction and its rollback snapshot.

    """
    checkpoint = validate_checkpoint(
        payload,
        expected_identity=expected_identity,
        expected_role="last",
        payload_kind="full",
        scheduler_expected=scheduler is not None,
        amp_expected=amp_enabled,
        require_best=False,
        adapter_expected=adapter_expected,
    )
    saved_sif = checkpoint["identity"]["numerical_environment"]["sif_sha256"]
    actual_sif = expected_identity["numerical_environment"]["sif_sha256"]
    if saved_sif is not None and actual_sif != saved_sif:
        message = "Exact resume requires the retained producing SIF identity in the actual training environment."
        raise ValueError(message)
    if adapter_expected and adapter is None:
        msg = "Transient checkpoint restoration requires an adapter."
        raise ValueError(msg)
    generator = _train_loader_generator(train_loader)
    sampler_state = checkpoint["train_sampler_state_dict"]
    sampler = _stateful_sampler(train_loader)
    sampler_load_state = getattr(sampler, "load_state_dict", None)
    current_sampler_state = _sampler_state(train_loader)
    if sampler_state is not None:
        if not callable(sampler_load_state):
            msg = "Checkpoint contains sampler state but the current sampler cannot restore it."
            raise TypeError(msg)
        if current_sampler_state is None:
            msg = "Checkpoint sampler restore requires a snapshot-capable current sampler."
            raise TypeError(msg)

    active_cuda_devices = _active_cuda_devices(model)
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if active_cuda_devices and cuda_states and len(cuda_states) != len(active_cuda_devices):
        msg = "Checkpoint CUDA RNG state count does not match the model runtime devices."
        raise RuntimeError(msg)

    snapshot = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "scaler_state_dict": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "loss_state_dict": copy.deepcopy(loss.state_dict()),
        "adapter_state_dict": copy.deepcopy(adapter.state_dict()) if adapter is not None else None,
        "train_loader_generator_state": generator.get_state().clone(),
        "train_sampler_state_dict": copy.deepcopy(current_sampler_state),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),  # noqa: NPY002 -- transactional process RNG snapshot
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": _capture_cuda_rng_states(active_cuda_devices),
    }

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        loss.load_state_dict(checkpoint["loss_state_dict"], strict=True)
        if adapter_expected and adapter is not None:
            adapter.load_state_dict(dict(checkpoint["adapter_state_dict"]))

        generator.set_state(checkpoint["train_loader_generator_state"])
        if sampler_state is not None and callable(sampler_load_state):
            sampler_load_state(sampler_state)

        random.setstate(cast("tuple[Any, ...]", checkpoint["python_rng_state"]))
        np.random.set_state(  # noqa: NPY002 -- restore the exact process-global RNG state
            cast("tuple[Any, ...]", checkpoint["numpy_rng_state"]),
        )
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if active_cuda_devices and cuda_states:
            _set_cuda_rng_states(cuda_states, active_cuda_devices)
    except BaseException as error:
        rollback_errors: list[str] = []
        _attempt_rollback(
            "model",
            lambda: model.load_state_dict(snapshot["model_state_dict"], strict=True),
            rollback_errors,
        )
        _attempt_rollback(
            "optimizer",
            lambda: optimizer.load_state_dict(snapshot["optimizer_state_dict"]),
            rollback_errors,
        )
        if scheduler is not None:
            _attempt_rollback(
                "scheduler",
                lambda: scheduler.load_state_dict(snapshot["scheduler_state_dict"]),
                rollback_errors,
            )
        if scaler is not None:
            _attempt_rollback(
                "scaler",
                lambda: scaler.load_state_dict(snapshot["scaler_state_dict"]),
                rollback_errors,
            )
        _attempt_rollback(
            "loss",
            lambda: loss.load_state_dict(snapshot["loss_state_dict"], strict=True),
            rollback_errors,
        )
        if adapter is not None:
            _attempt_rollback(
                "adapter",
                lambda: adapter.load_state_dict(cast("dict[str, Any]", snapshot["adapter_state_dict"])),
                rollback_errors,
            )
        _attempt_rollback(
            "train loader generator",
            lambda: generator.set_state(snapshot["train_loader_generator_state"]),
            rollback_errors,
        )
        if current_sampler_state is not None and callable(sampler_load_state):
            _attempt_rollback(
                "train sampler",
                lambda: sampler_load_state(snapshot["train_sampler_state_dict"]),
                rollback_errors,
            )
        _attempt_rollback(
            "Python RNG",
            lambda: random.setstate(cast("tuple[Any, ...]", snapshot["python_rng_state"])),
            rollback_errors,
        )
        _attempt_rollback(
            "NumPy RNG",
            lambda: np.random.set_state(  # noqa: NPY002 -- transactional process RNG rollback
                cast("tuple[Any, ...]", snapshot["numpy_rng_state"]),
            ),
            rollback_errors,
        )
        _attempt_rollback(
            "Torch CPU RNG",
            lambda: torch.set_rng_state(snapshot["torch_cpu_rng_state"]),
            rollback_errors,
        )
        _attempt_rollback(
            "Torch CUDA RNG",
            lambda: _set_cuda_rng_states(snapshot["torch_cuda_rng_states"], active_cuda_devices),
            rollback_errors,
        )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            msg = f"Checkpoint restore failed and runtime rollback was incomplete: {details}"
            raise RuntimeError(msg) from error
        raise

    return {
        "completed_epoch": checkpoint["completed_epoch"],
        "next_epoch": checkpoint["next_epoch"],
        "global_step": checkpoint["global_step"],
        "best_metric": checkpoint["best_metric"],
        "best_epoch": checkpoint["best_epoch"],
        "objective_history": copy.deepcopy(checkpoint["objective_history"]),
        "batch_progress": copy.deepcopy(checkpoint.get("batch_progress")),
    }


def restore_handoff_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_source_identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    loss: nn.Module,
    train_loader: DataLoader[Any],
) -> dict[str, Any]:
    """
    Restore the common state from an immutable transient teacher best checkpoint.

    The source is the best of an :func:`is_teacher_handoff_source` run, so it must
    be a full payload. The source adapter state is required and validated but
    deliberately not applied: the target arm owns a fresh controller and curriculum.
    """
    runtime_source_identity = copy.deepcopy(dict(expected_source_identity))
    runtime_source_identity["numerical_environment"] = numerical_environment_identity()
    source = validate_checkpoint(
        payload,
        expected_identity=runtime_source_identity,
        expected_role="best",
        payload_kind="full",
        scheduler_expected=scheduler is not None,
        amp_expected=amp_enabled,
        require_best=True,
        adapter_expected=True,
    )
    common_payload = dict(source)
    common_payload.pop("adapter_state_dict")
    common_payload.pop("batch_progress", None)
    common_payload["checkpoint_role"] = "last"
    restored = restore_checkpoint(
        common_payload,
        expected_identity=runtime_source_identity,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        amp_enabled=amp_enabled,
        loss=loss,
        train_loader=train_loader,
        adapter=None,
        adapter_expected=False,
    )
    return {
        "source_completed_epoch": source["completed_epoch"],
        "source_global_step": source["global_step"],
        "source_best_metric": source["best_metric"],
        "source_best_epoch": source["best_epoch"],
        "source_identity": copy.deepcopy(source["identity"]),
        "restored_common_state": restored,
    }
