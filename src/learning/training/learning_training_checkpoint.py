"""Persist and restore exact training checkpoints at admitted continuation boundaries.

Responsibilities:
  - Define the current checkpoint schema and run-identity binding
  - Capture model, optimizer, schedule, loss, RNG, unit sampler, stage budget, and optional adapter state
  - Decide from the admitted config whether a run's best keeps the full training state
  - Keep schedule state out of a teacher handoff source, so a continuation inherits only optimizer state
  - Validate the unit sampler's epoch order and committed group position as the exact continuation point
  - Validate checkpoint identity before mutating runtime objects
  - Admit a checkpoint only in the numerical environment it was produced in, a continuation source in its SIF too
    (native only natively) while reading admits a native and a container checkpoint alike, and refuse another
    environment as an environment error when every other identity field matches
  - Carry the committed training-work counts across resume; the global step equals their optimizer steps
  - Validate the objective history: a finite value exactly at eligible validation points, the best among them
  - Record a stage's divergence in its terminal last checkpoint, so a resume finalizes it instead of training on
  - Publish best and last checkpoints through atomic replacement, a new best together with its last; a failed write
    before their renames keeps the previous pair and stays an ``OSError``, also one torch hides in its archive writer,
    while a failure between their renames, which leaves a pair no resume admits, is terminal
  - Declare the per-task model projection and the training projection that includes it
  - Bind run identity to the producer digests a resolved config records, never to the running code

Design principles:
  - ``last_checkpoint.pt`` is the only continuation source and always carries the full state
  - ``best_checkpoint.pt`` is the inference and artifact source; it carries the full state only
    as the matched-A0 teacher handoff source and evaluation state otherwise
  - Resume continues at the saved update-group boundary, an epoch end or inside an epoch
  - Missing, extra, or mismatched state fails closed before runtime restoration; an unreadable file is a contract error

This module does NOT:
  - Allocate run directories, choose resume policy, or validate mutable run status
  - Construct models, optimizers, schedulers, losses, or dataloaders
  - Repair or reinterpret malformed checkpoints
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import os
import random
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import torch
from torch import nn

from src import common, datasets

from . import learning_training_budget as budget_state

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

    from .learning_training_budget import StageBudget

CheckpointRole = Literal["best", "last"]
CheckpointPayloadKind = Literal["full", "evaluation"]
CHECKPOINT_SCHEMA_KIND = "experiment_training_checkpoint"
# The RuntimeError text torch's CPU allocator raises when the host refuses an allocation: CAFFE_ENFORCE in
# c10/core/impl/alloc_cpu.cpp (torch/lib/libc10.so), verified in the installed torch 2.4.1.
_CPU_ALLOCATOR_OUT_OF_MEMORY: Final = "DefaultCPUAllocator: can't allocate memory"
_SHA256_LENGTH: Final = 64
_CHECKPOINT_IDENTITY_KEYS: Final = frozenset(
    {
        "identity_contract_sha256",
        "task",
        "task_contract_digest",
        "effective_config_digest",
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
    "model_construction": "resolved_model_kind_and_parameters",
    "loss_construction": "resolved_data_and_physics_loss",
    # The budget belongs to the identity: a run is never extended, so resume and identity are one digest.
    "optimization": "resolved_optimizer_scheduler_and_training_including_budget",
    "preprocessing": "fitted_normalizer_artifact_sha256",
    "selection": "resolved_evaluation_objective_and_ordered_split",
    # The evaluation gate: runs of the earlier protocol, whose training also evaluated OOD, carry another digest.
    "held_out_evaluation": "never_in_training_or_tuning; id_test_and_ood_only_through_the_explicit_artifacts_call",
}
TRAINING_IDENTITY_CONTRACT_DIGEST: Final = common.serialization.canonical_json_sha256(_TRAINING_IDENTITY_CONTRACT)
_CONFIG_SEMANTIC_ROOT_KEYS: Final = frozenset(
    {
        "task",
        "task_contract",
        "producer",
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
_RUN_PROVENANCE_KEYS: Final = frozenset({"device", "name", "revision", "layout"})
_DATA_SEMANTIC_KEYS: Final = frozenset({"train_dataset", "train_subset", "material_ood_datasets", "spatial_stride"})
# Evaluation accumulates sufficient statistics over the whole role, so its batch size changes no selected value.
_DATA_EXECUTION_KEYS: Final = frozenset({"evaluation_batch_size", "num_workers", "pin_memory", "persistent_workers"})
PRODUCER_KEYS: Final = frozenset({"training_sha256", "model_sha256"})
_MODEL_DIRECTORIES: Final = ("datasets/contracts", "datasets/preprocessing", "datasets/runtime")
_MODEL_FILES: Final = (
    "common/common_code_identity.py",
    "common/common_serialization.py",
    "datasets/packages/dataset_packages_manifest.py",
    "domain/domain_field_sets.py",
    "domain/domain_fields.py",
    "domain/tasks/domain_task_registry.py",
    "domain/tasks/domain_task_spec.py",
    "generation/contracts/generation_contracts_descriptors.py",
    "generation/contracts/generation_contracts_materials.py",
    "generation/contracts/generation_contracts_porosity.py",
    "generation/contracts/generation_contracts_profiles.py",
    "generation/contracts/generation_contracts_registry.py",
    "generation/contracts/generation_contracts_vocabulary.py",
    "learning/models/learning_models_factory.py",
)
_MODEL_EXEMPTIONS: Final = {
    "datasets/packages/dataset_packages_builder.py": "Dataset publication is bound by the Dataset content identity",
    "datasets/packages/dataset_packages_reproduction.py": "Dataset producer evidence is validated as evidence, not used for prediction",
    "datasets/packages/dataset_packages_selection.py": "Dataset membership is bound by the Dataset content identity",
    "generation/cases/generation_cases_config.py": "Generation config reaches a model only through Dataset content",
    "generation/contracts/generation_contracts_source.py": "producer provenance capture, not prediction",
    "learning/learning_device.py": "device choice; the numerical environment is bound separately",
    "learning/learning_device_policy.py": "device choice; the numerical environment is bound separately",
}
_STEADY_REFERENCE_EXEMPTION: Final = "steady mean-inlet-pressure reference of the shared scaling, reached only by steady encoding"
MODEL_PROJECTIONS: Final[dict[str, common.code_identity.CodeProjection]] = {
    "steady_flow": common.code_identity.CodeProjection(
        name="model_steady_flow",
        directories=_MODEL_DIRECTORIES,
        files=(
            *_MODEL_FILES,
            # The steady scaling divides by the mean inlet pressure that the Brinkman module defines.
            "domain/physics/domain_physics_boundary.py",
            "domain/physics/domain_physics_brinkman.py",
            "domain/physics/domain_physics_contracts.py",
            "domain/physics/domain_physics_derivatives.py",
            "domain/tasks/domain_task_steady_flow.py",
        ),
        excluded_markers=("transient",),
        exemptions={
            **_MODEL_EXEMPTIONS,
            "learning/models/learning_models_rno_cell.py": "transient-only RNO cell, reached through the model factory's kind dispatch",
        },
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="model_transient_drying",
        directories=_MODEL_DIRECTORIES,
        files=(
            *_MODEL_FILES,
            "datasets/packages/dataset_packages_trajectory.py",
            "datasets/packages/dataset_packages_transient_shards.py",
            "domain/domain_moisture.py",
            "domain/tasks/domain_task_transient_drying.py",
            "learning/learning_temporal.py",
            "learning/models/learning_models_rno_cell.py",
            "learning/transient/learning_transient_contracts.py",
            "learning/transient/learning_transient_rollout.py",
            "learning/transient/learning_transient_scaling.py",
            "learning/transient/learning_transient_tensorizer.py",
        ),
        excluded_markers=("steady", "_pino"),
        exemptions={**_MODEL_EXEMPTIONS, "domain/physics/domain_physics_brinkman.py": _STEADY_REFERENCE_EXEMPTION},
    ),
}
_TRAINING_DIRECTORIES: Final = ("learning/losses", "learning/metrics", "learning/models", "learning/training")
_TRAINING_FILES: Final = (
    "common/common_random.py",
    "datasets/packages/dataset_packages_reproduction.py",
    "experiments/config/experiments_config_loader.py",
    "experiments/experiments_run.py",
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
TRAINING_PROJECTIONS: Final[dict[str, common.code_identity.CodeProjection]] = {
    "steady_flow": common.code_identity.CodeProjection(
        name="training_steady_flow",
        directories=(*_TRAINING_DIRECTORIES, "domain/physics"),
        files=_TRAINING_FILES,
        excluded_markers=("transient",),
        exemptions={
            **_TRAINING_EXEMPTIONS,
            "experiments/experiments_matched_budget.py": "transient matched-budget protocol, reached through the run module dispatch",
            "learning/learning_temporal.py": "transient time-conditioning contract, reached through the loader's transient config validation",
        },
        includes=(MODEL_PROJECTIONS["steady_flow"],),
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="training_transient_drying",
        directories=(*_TRAINING_DIRECTORIES, "learning/transient"),
        files=(*_TRAINING_FILES, "domain/physics/domain_physics_contracts.py", "experiments/experiments_matched_budget.py"),
        excluded_markers=("steady", "_pino"),
        exemptions={
            **_TRAINING_EXEMPTIONS,
            "domain/physics/domain_physics_derivatives.py": "steady PINO derivative operators, reached through the loss factory dispatch",
            "domain/physics/domain_physics_brinkman.py": _STEADY_REFERENCE_EXEMPTION,
        },
        includes=(MODEL_PROJECTIONS["transient_drying"],),
    ),
}
_NEURALOPERATOR_PROJECTION: Final = common.code_identity.CodeProjection(name="neuraloperator", directories=(".",))
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_kind",
        "checkpoint_role",
        "payload_kind",
        "identity",
        "completed_epoch",
        "next_epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "loss_state_dict",
        "best_metric",
        "best_epoch",
        "objective_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_sampler_state_dict",
        "training_work",
        "update_work",
        "budget_state_dict",
        "divergence",
    }
)
# Committed optimizer groups, supervised predictions with a gradient, units, and distinct cases summed over updates.
TRAINING_WORK_KEYS: Final = frozenset({"optimizer_steps", "gradient_predictions", "units", "update_distinct_cases"})
# One row per committed update, in order; its columns sum to the matching training_work counts.
UPDATE_WORK_COLUMNS: Final = ("units", "distinct_cases", "supervised_transitions")
_UPDATE_WORK_TOTALS: Final = (("units", "units"), ("distinct_cases", "update_distinct_cases"), ("supervised_transitions", "gradient_predictions"))
# Only a full payload continues optimization. An evaluation payload keeps the remaining keys:
# completed_epoch and global_step are then the epoch (possibly a budget-ended partial one) and
# the optimizer step at which its best was evaluated, never a continuation boundary.
_CONTINUATION_KEYS: Final = frozenset(
    {
        "next_epoch",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_sampler_state_dict",
        "divergence",
    }
)


class ResumeEnvironmentError(ValueError):
    """Refuse a checkpoint in another numerical environment than its producing one, which admits it; resuming binds the image too."""


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
    """Bind the declared task-owned training implementation, excluding presentation.

    The task's training projection is its model projection plus the loss, metric,
    model-naming and training packages, the task's other training machinery and the
    shared run and loader modules; files owned by the other task do not participate.
    A changed digest detects changed implementation, without proving mathematical
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


def model_implementation_digest(task_id: str) -> str:
    """Bind the code that turns persisted weights and fitted scaling into predictions of one task.

    The task's model projection names the Dataset contracts, preprocessing and runtime
    readers with their scaling fit callers, the Generation interpretation contracts,
    the model factory and the task's tensorizer and rollout files; losses, metrics and
    the training lifecycle do not participate. The training projection includes it.

    Raises
    ------
    ValueError
        If ``task_id`` has no declared model projection.

    """
    projection = MODEL_PROJECTIONS.get(task_id)
    if projection is None:
        message = f"No model implementation projection is declared for task {task_id!r}."
        raise ValueError(message)
    return projection.sha256()


def effective_config_identity_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return positive model-training dependencies without paths or reporting.

    The implementation identity is the config's recorded ``producer`` evidence, never
    the running code, so a persisted config keeps its identity under later code changes.
    The budget, k and the schedule belong to it, so a resume continues exactly this
    identity and another budget is another run. A matched B or A1 also binds the
    ``a0_id`` of the A0 it starts from. Execution settings that change no
    selected value (loader workers, evaluation batch, ``training.memory_batch``) stay
    outside it, so a resume may change them.
    """
    keys = set(config)
    if keys == {"task", "temporal"}:
        temporal = _required_mapping(config["temporal"], label="config temporal")
        return {
            "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
            "task": copy.deepcopy(config["task"]),
            "temporal": copy.deepcopy(dict(temporal)),
        }
    is_transient = config.get("task") == "transient_drying"
    transient_roots = frozenset({"input_profile", "temporal"}) if is_transient else frozenset()
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
    data = _selected_config_mapping(config["data"], semantic_keys=_DATA_SEMANTIC_KEYS, ignored_keys=_DATA_EXECUTION_KEYS, label="data")
    task_contract = _required_mapping(config["task_contract"], label="config task_contract")
    required_task_contract = {"task", "digest", "data_contract_digest"}
    missing_contract = sorted(required_task_contract.difference(task_contract))
    if missing_contract:
        message = f"Training task contract is missing identity fields {missing_contract}."
        raise ValueError(message)
    training = copy.deepcopy(dict(_required_mapping(config["training"], label="config training")))
    # The memory split of an update changes only floating-point rounding, so it is execution, not science.
    training.pop("memory_batch", None)
    if is_transient and training.get("stage") == "a" and training.get("comparison_arm") == "a0":
        # A0 optimization is independent of which self-fed/teacher-forced branch follows it.
        training.pop("stage_schedule", None)
        training.pop("matched_a1_b", None)
        training.pop("teacher_handoff", None)
    payload = {
        "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": copy.deepcopy(config["task"]),
        "producer": _selected_config_mapping(config["producer"], semantic_keys=PRODUCER_KEYS, ignored_keys=frozenset(), label="producer"),
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
        layout = _required_mapping(config["run"], label="config run").get("layout")
        if isinstance(layout, Mapping) and layout.get("stage") in {"b", "a1"}:
            # A continuation is only the same run when it starts from the same A0 lineage.
            payload["a0_id"] = layout["a0_id"]
    return payload


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the dependency-scoped identity of a training config, the one contract of resume and admission."""
    return common.serialization.canonical_json_sha256(effective_config_identity_payload(config))


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


def _require_divergence(value: object, *, role: str) -> None:
    """Admit a divergence record: none, or the non-empty message only a last checkpoint carries."""
    if value is not None and (role != "last" or not isinstance(value, str) or not value):
        msg = "Checkpoint divergence must be None, or the non-empty message of a terminal last checkpoint."
        raise ValueError(msg)


def _validate_bound_violations(value: object) -> None:
    """Admit one point's per-case bound evidence: state channel to simulation case id to the finite hours of its first violation."""
    if not isinstance(value, Mapping) or not all(
        isinstance(channel, str)
        and channel
        and isinstance(cases, Mapping)
        and cases
        and all(
            isinstance(case_id, str) and case_id and not isinstance(hours, bool) and isinstance(hours, (int, float)) and math.isfinite(float(hours))
            for case_id, hours in cases.items()
        )
        for channel, cases in value.items()
    ):
        msg = "Checkpoint objective_history bound_violations must map non-empty channels to non-empty case ids with finite hours."
        raise ValueError(msg)


def _validate_training_work(value: Any) -> dict[str, int]:
    """Return an isolated copy of the non-negative committed training-work counts."""
    if (
        not isinstance(value, Mapping)
        or set(value) != TRAINING_WORK_KEYS
        or any(type(value[key]) is not int or value[key] < 0 for key in TRAINING_WORK_KEYS)
    ):
        message = f"Checkpoint training_work must hold exactly the non-negative integers {sorted(TRAINING_WORK_KEYS)}."
        raise ValueError(message)
    return {key: int(value[key]) for key in sorted(TRAINING_WORK_KEYS)}


def validate_update_work(value: Any, *, training_work: Mapping[str, int]) -> list[list[int]]:
    """
    Return an isolated copy of the per-update :data:`UPDATE_WORK_COLUMNS` rows.

    Every committed update has one row of a positive unit count, a distinct-case count
    between one and its units, and non-negative supervised transitions; the rows number
    ``training_work["optimizer_steps"]`` and their columns sum to the matching counts.

    Raises
    ------
    ValueError
        If a row is malformed or the rows disagree with ``training_work``.

    """
    rows = value if isinstance(value, list) else None
    if rows is None or any(
        not isinstance(row, list)
        or len(row) != len(UPDATE_WORK_COLUMNS)
        or any(type(item) is not int for item in row)
        or not 1 <= row[1] <= row[0]
        or row[2] < 0
        for row in rows
    ):
        message = f"Checkpoint update_work must hold one {list(UPDATE_WORK_COLUMNS)} integer row per committed update."
        raise ValueError(message)
    totals = {column: sum(row[index] for row in rows) for index, column in enumerate(UPDATE_WORK_COLUMNS)}
    if len(rows) != training_work["optimizer_steps"] or any(totals[column] != training_work[key] for column, key in _UPDATE_WORK_TOTALS):
        message = "Checkpoint update_work disagrees with its training_work counts."
        raise ValueError(message)
    return [list(row) for row in rows]


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


def validate_checkpoint_identity(value: Any, *, label: str) -> dict[str, Any]:
    """
    Validate one current dependency-scoped checkpoint identity and return an isolated copy.

    The one schema of the identity every checkpoint carries and every finished run summary records for its producer.

    Raises
    ------
    TypeError
        If the identity, a nested mapping, the task or an objective field has the wrong type.
    ValueError
        If its keys, digests, roles, numerical environment, objective or identity contract differ from the schema.

    """
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
    for key in ("task_contract_digest", "effective_config_digest"):
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
    # Both tasks bind the ordered sources of each OOD Evaluation role as one digest, an empty list included.
    membership_roles = {"train", "validation", "id_test", "material_ood", "parameter_ood"}
    dataset_roles = {"train", "material_ood", "parameter_ood"}
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
    split = datasets.preprocessing.splits.admit_split(split_indices)
    if split.task != task or split.task_contract_digest != task_digest:
        message = "Split task identity must match the checkpoint config contract."
        raise ValueError(message)
    if split.extension is not None and split.extension["spatial_view"]["spatial_stride"] != config["data"]["spatial_stride"]:
        message = "Transient checkpoint split spatial stride does not match the config."
        raise ValueError(message)
    digest = common.serialization.canonical_json_sha256
    memberships = split.membership_digests()
    ood_roles = datasets.preprocessing.splits.OOD_SPLIT_ROLES
    persisted = dict(persisted_config or config)
    return validate_checkpoint_identity(
        {
            "identity_contract_sha256": TRAINING_IDENTITY_CONTRACT_DIGEST,
            "task": task,
            "task_contract_digest": task_digest,
            "effective_config_digest": config_digest(persisted),
            "dataset_ids": {
                "train": split.train.dataset_id,
                **{role: digest([part.dataset_id for part in split.ood_parts(role)]) for role in ood_roles},
            },
            "dataset_fingerprints": {
                "train": digest(dict(split.train.dataset)),
                **{role: digest([dict(part.dataset) for part in split.ood_parts(role)]) for role in ood_roles},
            },
            "split_contract_digest": digest(split.as_payload()),
            "split_membership_digests": {**memberships, **{role: digest(memberships[role]) for role in ood_roles}},
            "normalizer_sha256": normalizer_digest,
            "numerical_environment": numerical_environment_identity(),
            "objective": objective,
        },
        label="built identity",
    )


def _unit_sampler(train_loader: DataLoader[Any]) -> datasets.runtime.units.UnitSampler:
    """Return the resumable unit sampler whose order, position and generator a full checkpoint carries."""
    sampler = getattr(train_loader, "batch_sampler", None)
    if not isinstance(sampler, datasets.runtime.units.UnitSampler):
        msg = "Exact resume requires a training DataLoader driven by a UnitSampler."
        raise TypeError(msg)
    return sampler


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
    loss: nn.Module,
    best_metric: float | None,
    best_epoch: int | None,
    objective_history: list[dict[str, Any]],
    train_loader: DataLoader[Any],
    runtime_device: torch.device,
    training_work: Mapping[str, int],
    update_work: list[list[int]],
    budget: StageBudget,
    adapter: Any | None = None,
    divergence: str | None = None,
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
        schedule, RNG, unit sampler and ``next_epoch``. Only a last checkpoint
        carries schedule state: a full best is a teacher handoff source, whose
        continuation starts its own schedule.
    identity : Mapping[str, Any]
        Immutable run identity from :func:`build_checkpoint_identity`.
    completed_epoch : int
        Positive one-based epoch in which the checkpoint is captured, fully or at a committed group inside it.
    global_step : int
        Number of completed optimizer steps.
    model, optimizer, scheduler, loss : Any
        Stateful runtime objects.
    best_metric : float | None
        Best finite objective so far, if any.
    best_epoch : int | None
        One-based epoch owning ``best_metric``.
    objective_history : list[dict[str, Any]]
        Ordered finite evaluation history, one entry per validation point.
    train_loader : DataLoader
        Loader whose unit sampler's epoch order, committed position and generator continue training;
        a position inside its epoch makes ``next_epoch`` that same epoch.
    runtime_device : torch.device
        Active concrete execution device. CUDA RNG is captured only for this device and
        any CUDA devices that actually own model parameters or buffers.
    training_work : Mapping[str, int]
        Committed :data:`TRAINING_WORK_KEYS` counts of this run so far.
    update_work : list[list[int]]
        One :data:`UPDATE_WORK_COLUMNS` row per committed update of this run so far.
    budget : StageBudget
        The stage budget with its committed units, optimisation seconds and validation points.
    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with a loop scaling where accepted.
    divergence : str | None, optional
        The non-finite training loss that ended the stage; only the terminal last checkpoint of a
        diverged stage carries it, so a resume finalizes the stage instead of training on.

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
    _require_divergence(divergence, role=role)
    if not isinstance(runtime_device, torch.device) or runtime_device.type not in {"cpu", "cuda"}:
        msg = f"Checkpoint capture requires one concrete CPU or CUDA torch.device, got {runtime_device!r}."
        raise TypeError(msg)

    validated_identity = validate_checkpoint_identity(identity, label="identity")
    adapter_state: dict[str, Any] | None = None
    if adapter is not None:
        raw_adapter_state = adapter.state_dict()
        if not isinstance(raw_adapter_state, Mapping):
            msg = "Checkpoint adapter.state_dict() must return a mapping."
            raise TypeError(msg)
        adapter_state = dict(raw_adapter_state)
    payload: dict[str, Any] = {
        "schema_kind": CHECKPOINT_SCHEMA_KIND,
        "checkpoint_role": role,
        "payload_kind": payload_kind,
        "identity": validated_identity,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "loss_state_dict": loss.state_dict(),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": best_epoch,
        "objective_history": copy.deepcopy(objective_history),
        "training_work": _validate_training_work(training_work),
        "update_work": validate_update_work(update_work, training_work=training_work),
        "budget_state_dict": budget.state_dict(),
    }
    if payload_kind == "full":
        sampler = _unit_sampler(train_loader)
        active_cuda_devices = _active_cuda_devices(model, runtime_device=runtime_device)
        payload |= {
            "divergence": divergence,
            "next_epoch": completed_epoch + (1 if sampler.epoch_complete else 0),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and role == "last" else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),  # noqa: NPY002 -- exact process-global RNG state
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": _capture_cuda_rng_states(active_cuda_devices),
            "train_sampler_state_dict": sampler.state_dict(),
        }
    if adapter_state is not None:
        payload["adapter_state_dict"] = copy.deepcopy(adapter_state)
    return payload


def _partial_epoch(payload: Mapping[str, Any]) -> bool:
    """Return whether a full checkpoint stops inside its epoch's unit order."""
    state = payload["train_sampler_state_dict"]
    return bool(state["cursor"] != state["size"])


def records_completed_budget(payload: Mapping[str, Any]) -> bool:
    """
    Return whether a validated checkpoint records a completed stage budget.

    A completed budget is terminal: such a checkpoint finalizes its run and
    never continues optimization.
    """
    state = budget_state.validate_state(payload["budget_state_dict"])
    return budget_state.StageBudget(**state).complete


def recorded_history_epochs(payload: Mapping[str, Any]) -> int:
    """
    Return how many epoch history records precede one validated last checkpoint.

    An operational partial epoch publishes its checkpoint before any history
    record; a terminal epoch, ended by its budget or a divergence, records its
    history before its checkpoints.
    """
    completed_epoch = int(payload["completed_epoch"])
    if _partial_epoch(payload) and payload["divergence"] is None and not records_completed_budget(payload):
        return completed_epoch - 1
    return completed_epoch


def validate_checkpoint(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    payload_kind: CheckpointPayloadKind,
    scheduler_expected: bool,
    require_best: bool,
    adapter_expected: bool = False,
    resume: bool = False,
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
    require_best : bool
        Whether finite best-objective state is mandatory.

    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.
    resume : bool, optional
        Admit the checkpoint as a continuation source: its numerical environment must
        equal the expected one exactly, ``sif_sha256`` ``None`` (native) matching only
        ``None``. Otherwise a native and a container identity are compared without
        their SIF digest.

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
    ResumeEnvironmentError
        A ``ValueError`` if a checkpoint that matches every other identity field was
        produced in another numerical environment, for a continuation source its image
        included: the environment, not the checkpoint, differs.

    Notes
    -----
    Validation mutates no runtime object. RNG payloads are tested on temporary
    generators before the checkpoint can reach transactional restoration. An
    evaluation payload keeps identity, role, objective-history, and selected-best
    checks; only continuation checks are absent because it has no continuation state.

    """
    if payload.get("schema_kind") != CHECKPOINT_SCHEMA_KIND:
        msg = (
            "Malformed or unsupported checkpoint schema_kind: "
            f"got {payload.get('schema_kind')!r}, admitted {CHECKPOINT_SCHEMA_KIND!r}. "
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
    expected_keys = (_CHECKPOINT_KEYS if full else _CHECKPOINT_KEYS - _CONTINUATION_KEYS) | ({"adapter_state_dict"} if adapter_expected else set())
    missing = sorted(expected_keys.difference(payload))
    unknown = sorted(set(payload).difference(expected_keys))
    if missing or unknown:
        msg = f"Checkpoint {payload_kind} payload keys mismatch. Missing keys: {missing}. Unknown keys: {unknown}."
        raise ValueError(msg)
    if adapter_expected and not isinstance(payload["adapter_state_dict"], Mapping):
        msg = "Transient checkpoint adapter_state_dict must be a mapping."
        raise TypeError(msg)
    identity = validate_checkpoint_identity(payload["identity"], label="identity")
    expected = validate_checkpoint_identity(expected_identity, label="expected identity")
    comparison_identity = copy.deepcopy(identity)
    comparison_expected = copy.deepcopy(expected)
    saved_environment = comparison_identity.pop("numerical_environment")
    actual_environment = comparison_expected.pop("numerical_environment")
    incompatible = "Checkpoint run identity is incompatible with config, task, dataset, split, normalizer, or objective state."
    # Every other field is the run's own identity: a difference there contradicts the run, whatever its environment.
    if comparison_identity != comparison_expected:
        raise ValueError(incompatible)
    # Reading admits a native and a container run alike; two different containers stay apart.
    if not resume and None in (saved_environment["sif_sha256"], actual_environment["sif_sha256"]):
        saved_environment["sif_sha256"] = actual_environment["sif_sha256"] = None
    differing = ", ".join(key for key in sorted(saved_environment) if saved_environment[key] != actual_environment[key])
    if differing:
        # A checkpoint is admitted only in the numerical environment it was produced in, a continuation source in its
        # container too, the rule of the history's identity digest: the environment differs, and the producing one admits it.
        use, rule = ("Exact resume", "a native run resumes only natively") if resume else ("Reading", "a native and a container run read alike")
        msg = f"{use} requires the producing numerical environment, which differs in {differing}; {rule}."
        raise ResumeEnvironmentError(msg)

    completed_epoch = payload["completed_epoch"]
    global_step = payload["global_step"]
    integer_fields = [("completed_epoch", completed_epoch, 1), ("global_step", global_step, 0)]
    if full:
        integer_fields.append(("next_epoch", payload["next_epoch"], 1))
    for label, value, minimum in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            msg = f"Checkpoint {label} must be an integer >= {minimum}."
            raise TypeError(msg)
    training_work = _validate_training_work(payload["training_work"])
    validate_update_work(payload["update_work"], training_work=training_work)
    # The loop starts both at zero for a run or stage and advances both by every committed update.
    if global_step != training_work["optimizer_steps"]:
        msg = "Checkpoint global_step must equal its committed optimizer steps."
        raise ValueError(msg)
    budget = budget_state.validate_state(payload["budget_state_dict"])
    if full:
        _require_divergence(payload["divergence"], role=expected_role)
        sampler_state = payload["train_sampler_state_dict"]
        datasets.runtime.units.validate_state(sampler_state)
        # A checkpoint follows at least one committed group of its epoch, at the epoch's end or inside it.
        if sampler_state["epoch"] != completed_epoch - 1 or sampler_state["cursor"] < 1:
            msg = "Checkpoint unit sampler state does not belong to a committed group of its completed epoch."
            raise ValueError(msg)
        if payload["next_epoch"] != completed_epoch + (0 if _partial_epoch(payload) else 1):
            msg = "Checkpoint next_epoch conflicts with its completed or partial unit-sampler epoch."
            raise ValueError(msg)
        # Only a matched A1, the one budget that replays B's validations, stops inside an epoch after its remaining transitions.
        remaining = budget["total"] - budget["supervised_transitions"]
        if "stop" in sampler_state and ("replay" not in budget or sampler_state["stop"] != sampler_state["cursor"] + remaining):
            msg = "Checkpoint sampler stop is not the exact end of a matched A1 of transition units at its remaining budget."
            raise ValueError(msg)

    for key in ("model_state_dict", "loss_state_dict", *(("optimizer_state_dict",) if full else ())):
        _required_mapping(payload[key], label=key)
    # A full best is a teacher handoff source: its continuation starts its own schedule, so it carries none.
    if full and (scheduler_expected and expected_role == "last") != isinstance(payload["scheduler_state_dict"], Mapping):
        msg = "Checkpoint schedule state is absent or present inconsistently with the resolved config and role."
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
    previous_epoch = previous_point = 0
    for entry in history:
        if not isinstance(entry, Mapping) or set(entry) != {"point", "epoch", "objective_id", "value", "bound_violations"}:
            msg = "Checkpoint objective_history entries must contain exactly point, epoch, objective_id, value, and bound_violations."
            raise ValueError(msg)
        _validate_bound_violations(entry["bound_violations"])
        value = entry["value"]
        epoch = entry["epoch"]
        point = entry["point"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not max(1, previous_epoch) <= epoch <= completed_epoch:
            msg = "Checkpoint objective_history epochs must be non-decreasing epochs up to the captured one."
            raise ValueError(msg)
        if isinstance(point, bool) or not isinstance(point, int) or not previous_point < point <= budget["validated_point"]:
            msg = "Checkpoint objective_history points must be strictly increasing validated budget points."
            raise ValueError(msg)
        previous_point = point
        if entry["objective_id"] != identity["objective"]["id"]:
            msg = "Checkpoint objective_history id does not match checkpoint identity."
            raise ValueError(msg)
        previous_epoch = epoch
        # Only an eligible validation holds a value, always finite; a non-eligible point holds none.
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = "Checkpoint objective_history values must be finite."
            raise ValueError(msg)
        history_values.append((epoch, float(value)))
    if sorted([entry["point"] for entry in history] + budget["skipped_points"]) != list(range(1, budget["validated_point"] + 1)):
        msg = "Checkpoint objective_history and skipped budget points do not cover exactly the validated points."
        raise ValueError(msg)
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
    value = payload["torch_cpu_rng_state"]
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        msg = "Checkpoint torch_cpu_rng_state must be a one-dimensional CPU byte tensor."
        raise TypeError(msg)
    try:
        torch.Generator().set_state(value)
    except RuntimeError as error:
        msg = "Checkpoint torch_cpu_rng_state is not a valid Torch generator state."
        raise ValueError(msg) from error
    cuda_states = payload["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.device.type == "cpu" and state.dtype == torch.uint8 and state.ndim == 1 for state in cuda_states
    ):
        msg = "Checkpoint CUDA RNG states must be one-dimensional CPU byte tensors."
        raise TypeError(msg)
    return dict(payload)


def _staged_path(path: Path) -> Path:
    """Return the fixed hidden sibling a checkpoint is staged at before its rename."""
    return path.with_name(f".{path.name}.staged")


def _fsync_directory(directory: Path) -> None:
    """Best-effort flush of renamed directory entries; content atomicity holds without it."""
    with suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def save_checkpoints(*publications: tuple[Mapping[str, Any], Path | str]) -> None:
    """
    Atomically publish already validated checkpoint payloads as one ordered group.

    Parameters
    ----------
    *publications : tuple[Mapping[str, Any], Path | str]
        Complete best- or last-checkpoint mappings with their exact role-qualified
        paths, in rename order.

    Raises
    ------
    OSError
        If a write fails before the first rename, also inside the archive torch writes, which
        torch reports as a ``RuntimeError`` with that ``OSError`` in its context: every
        destination keeps its previous checkpoint.
    RuntimeError
        If anything ends the publication between its first and its last rename: the replaced
        checkpoint disagrees with the kept one (a best newer than its last), which no resume admits.

    Notes
    -----
    Every payload is first published atomically to its fixed staged sibling
    (``.<name>.staged``); only then are the staged files renamed back to back, so
    a failure before the renames leaves every destination unchanged. A handled
    failure removes all staged files; a crash leaves at most one staged file per
    destination, which ``discard_staged_checkpoints`` removes before the next run.
    A failure between the renames does not restore the previous pair; it is terminal.

    """
    staged = [(_staged_path(Path(path)), Path(path)) for _payload, path in publications]
    renamed = 0
    try:
        for (payload, _path), (temporary, _final) in zip(publications, staged, strict=True):
            common.serialization.atomic_torch_save(dict(payload), temporary)
        for temporary, final in staged:
            temporary.replace(final)
            renamed += 1
    except BaseException as error:
        with suppress(OSError):  # a failed cleanup must not replace the publication error
            discard_staged_checkpoints(*(final for _temporary, final in staged))
        if 0 < renamed < len(staged):
            msg = (
                f"{staged[renamed - 1][1].name} was replaced but {staged[renamed][1].name} was not, so they disagree "
                f"and no resume admits them: {type(error).__name__}: {error}"
            )
            raise RuntimeError(msg) from error
        # torch's archive writer reports a failed write as a RuntimeError ("unexpected pos") with the OSError in its context.
        written: BaseException | None = error if isinstance(error, Exception) else None
        seen: set[int] = set()
        while written is not None and not isinstance(written, OSError) and id(written) not in seen:
            seen.add(id(written))
            written = written.__cause__ or written.__context__
        if isinstance(written, OSError) and written is not error:
            raise type(written)(*written.args) from error
        raise
    if staged:
        _fsync_directory(staged[-1][1].parent)


def discard_staged_checkpoints(*paths: Path) -> None:
    """Remove staged siblings, and the atomic writer's temporaries of them, left by a failure or crash."""
    for path in paths:
        staged = _staged_path(path)
        staged.unlink(missing_ok=True)
        # A kill inside atomic_torch_save leaves its randomly named temporary of the staged file.
        for candidate in staged.parent.glob(f".{staged.name}.*"):
            if common.serialization.atomic_write_temporary_destination(candidate) == staged:
                candidate.unlink(missing_ok=True)


def load_checkpoint(
    path: Path | str,
    *,
    expected_identity: Mapping[str, Any],
    expected_role: CheckpointRole,
    payload_kind: CheckpointPayloadKind,
    scheduler_expected: bool,
    require_best: bool,
    adapter_expected: bool = False,
    resume: bool = False,
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
    require_best : bool
        Whether finite selected-best state must already exist.

    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.
    resume : bool, optional
        Admit it as a continuation source; see :func:`validate_checkpoint`.

    Returns
    -------
    dict[str, Any]
        Fully validated checkpoint mapping.

    Raises
    ------
    FileNotFoundError
        If the checkpoint is absent.
    ValueError
        If its bytes are no readable checkpoint (empty, truncated or other content), or by :func:`validate_checkpoint`.
    OSError
        If reading the file fails: the job's environment, not the checkpoint, ended the load.
    MemoryError
        If decoding runs out of CPU memory, also as torch's CPU allocator reports it: the job's environment as well.

    Notes
    -----
    The file is read whole before ``torch.load`` decodes it, so every other decoding failure reads as an unreadable
    checkpoint.

    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        msg = f"Required {expected_role} checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    # A failed read or allocation is the job's environment; once the bytes are read, whatever else ends their decoding is content.
    content = checkpoint_path.read_bytes()
    try:
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    except MemoryError:
        raise
    except Exception as error:
        if isinstance(error, RuntimeError) and _CPU_ALLOCATOR_OUT_OF_MEMORY in str(error):
            raise MemoryError(str(error)) from error
        msg = f"Checkpoint is unreadable: {checkpoint_path}."
        raise ValueError(msg) from error
    if not isinstance(payload, Mapping):
        msg = f"Checkpoint must contain a mapping: {checkpoint_path}"
        raise TypeError(msg)
    return validate_checkpoint(
        payload,
        expected_identity=expected_identity,
        expected_role=expected_role,
        payload_kind=payload_kind,
        scheduler_expected=scheduler_expected,
        require_best=require_best,
        adapter_expected=adapter_expected,
        resume=resume,
    )


def restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    loss: nn.Module,
    train_loader: DataLoader[Any],
    budget: StageBudget | None,
    adapter: Any | None = None,
    adapter_expected: bool = False,
    continue_sampler_generator: bool = False,
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
    model, optimizer, scheduler, loss : Any
        Current runtime components whose state may be replaced.
    train_loader : DataLoader
        Loader whose unit sampler receives the saved epoch order, position and generator.
    budget : StageBudget | None
        The resumed stage's budget, which receives the saved progress; ``None`` for a
        teacher handoff, whose target starts its own budget.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with a loop scaling where accepted.
    adapter_expected : bool, optional
        Require task-owned adapter continuation evidence in the checkpoint.
    continue_sampler_generator : bool, optional
        Adopt only the saved sampler generator at a fresh epoch, for a teacher handoff whose
        target trains on its own units.

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
        require_best=False,
        adapter_expected=adapter_expected,
        resume=True,
    )
    if adapter_expected and adapter is None:
        msg = "Transient checkpoint restoration requires an adapter."
        raise ValueError(msg)
    sampler = _unit_sampler(train_loader)
    sampler_state = checkpoint["train_sampler_state_dict"]

    active_cuda_devices = _active_cuda_devices(model)
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if active_cuda_devices and cuda_states and len(cuda_states) != len(active_cuda_devices):
        msg = "Checkpoint CUDA RNG state count does not match the model runtime devices."
        raise RuntimeError(msg)

    snapshot = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "loss_state_dict": copy.deepcopy(loss.state_dict()),
        "adapter_state_dict": copy.deepcopy(adapter.state_dict()) if adapter is not None else None,
        "train_sampler_state_dict": sampler.state_dict(),
        "budget_state_dict": budget.state_dict() if budget is not None else None,
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
        loss.load_state_dict(checkpoint["loss_state_dict"], strict=True)
        if adapter_expected and adapter is not None:
            adapter.load_state_dict(dict(checkpoint["adapter_state_dict"]))
        if budget is not None:
            budget.load_state_dict(checkpoint["budget_state_dict"])

        if continue_sampler_generator:
            sampler.continue_generator(sampler_state)
        else:
            sampler.load_state_dict(sampler_state)

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
            "train unit sampler",
            lambda: sampler.load_state_dict(cast("dict[str, Any]", snapshot["train_sampler_state_dict"])),
            rollback_errors,
        )
        if budget is not None:
            _attempt_rollback(
                "stage budget",
                lambda: budget.load_state_dict(cast("dict[str, Any]", snapshot["budget_state_dict"])),
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
        "training_work": _validate_training_work(checkpoint["training_work"]),
        "update_work": validate_update_work(checkpoint["update_work"], training_work=checkpoint["training_work"]),
        "divergence": checkpoint["divergence"],
    }


def restore_handoff_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_source_identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    loss: nn.Module,
    train_loader: DataLoader[Any],
) -> dict[str, Any]:
    """
    Restore the common state from an immutable transient teacher best checkpoint.

    The source is the best of an :func:`is_teacher_handoff_source` run, so it must
    be a full payload without schedule state; a source with schedule state is
    rejected. The target inherits the model, loss and optimizer state (AdamW moments
    and step counters) and builds its own schedule and budget. The source adapter
    state is required and validated but deliberately not applied: the target arm owns
    a fresh controller and curriculum. The target's unit sampler continues only the
    source's generator stream at a fresh epoch, because the target trains on its own units.
    """
    runtime_source_identity = copy.deepcopy(dict(expected_source_identity))
    runtime_source_identity["numerical_environment"] = numerical_environment_identity()
    source = validate_checkpoint(
        payload,
        expected_identity=runtime_source_identity,
        expected_role="best",
        payload_kind="full",
        scheduler_expected=False,
        require_best=True,
        adapter_expected=True,
        resume=True,
    )
    common_payload = dict(source)
    common_payload.pop("adapter_state_dict")
    common_payload["checkpoint_role"] = "last"
    restored = restore_checkpoint(
        common_payload,
        expected_identity=runtime_source_identity,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss=loss,
        train_loader=train_loader,
        budget=None,
        adapter=None,
        adapter_expected=False,
        continue_sampler_generator=True,
    )
    return {
        "source_completed_epoch": source["completed_epoch"],
        "source_global_step": source["global_step"],
        "source_best_metric": source["best_metric"],
        "source_best_epoch": source["best_epoch"],
        "source_identity": copy.deepcopy(source["identity"]),
        "restored_common_state": restored,
    }
