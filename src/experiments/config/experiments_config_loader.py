"""Load and strictly resolve semantic experiment configurations.

Responsibilities:
  - Parse YAML mappings under the strict experiment schema
  - Merge an authored experiment's optional shared task base before validation
  - Reject unknown keys, identifiers, fields, and contradictory settings
  - Derive task-fixed channels, defaults, objective, and task-contract digest
  - Construct dataloaders from an already resolved configuration, capping their workers at the CPUs available to the process
  - Atomically save resolved YAML and derive the run layout and observer organization from the config source

Design principles:
  - Semantic resolution is strict, path-aware, deterministic, and side-effect free
  - Task-fixed semantics and full metric definitions come only from domain.tasks
  - Executable YAMLs use one canonical section order and explicitly select their objective
  - Saved configuration identifiers never depend on Python class names

This module does NOT:
  - Define dataset storage or fingerprints. Dataset objects enforce those contracts
  - Implement physics equations or metric mathematics. Domain and learning modules do
  - Own checkpoint, resume, run-directory, or artifact lifecycle

Notes:
  Transient epoch allocation:
    - ``stage_a_only`` trains A0 alone; ``training.epochs`` equals the schedule's ``total_epochs``.
    - ``matched_a1_b`` preserves independent positive per-stage maximum epochs with ``budget_unit: train_seconds``; A0's cap is its epoch
      budget, B stops at its measured seconds and A1 at B's receipt.

"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any, NoReturn

import yaml

from src import common, datasets, domain, learning
from src.learning.losses import learning_losses_factory as loss_factory
from src.learning.metrics import learning_metrics as metric_registry
from src.learning.models import learning_models_factory as model_factory
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum

from . import experiments_config_defaults as config_defaults
from . import experiments_config_naming as config_naming


class ConfigError(ValueError):
    """
    Represent a path-qualified semantic configuration violation.

    This error is raised at raw-schema and resolved-config boundaries. Registry
    errors are wrapped as ``ConfigError`` when their meaning belongs to a YAML
    path, while unknown standalone registry lookups may still raise ``ValueError``.
    """


CANONICAL_EXPERIMENT_SECTION_ORDER = (
    "task",
    "run",
    "data",
    "input_profile",
    "temporal",
    "scaling",
    "model",
    "loss",
    "evaluation",
    "optimizer",
    "scheduler",
    "training",
    "tracking",
)
_ROOT_KEYS = frozenset(CANONICAL_EXPERIMENT_SECTION_ORDER)
EXPERIMENT_ROOT_KEYS = _ROOT_KEYS
# An authored experiment may name its task's shared base; the base owns values every maintained file shares.
AUTHORED_BASE_KEY = "base"
AUTHORED_BASE_FILENAME = "base.yaml"
_TASK_FIXED_KEYS = frozenset(
    {
        "input_fields",
        "output_fields",
        "in_channels",
        "out_channels",
        "task_contract",
        "preprocessing",
        "physics",
        "paths",
    }
)
_ADAM_BETA_COUNT = 2
_STEADY_DATA_KEYS = frozenset({"train_dataset", "additional_evaluation_datasets", "batch_size", "num_workers", "pin_memory", "persistent_workers"})
_PHYSICS_WEIGHT_NAMES = ("momentum_weight", "continuity_weight", "boundary_weight")
_RESOLVED_PATH_KEYS = frozenset(
    {
        "project_root",
        "storage_root",
        "dataset_root",
        "output_root",
    }
)
_SECTION_KEYS = {
    "run": frozenset({"seed", "revision", "deterministic", "device", "name", "naming_schema_version", "layout"}),
    "data": frozenset(
        {
            "train_dataset",
            "additional_evaluation_datasets",
            "additional_evaluation_fraction",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "spatial_stride",
            "transient_backend_preference",
            "transient_backend_required",
            "hdf5_cache_size",
        }
    ),
    "model": frozenset({"kind", "params"}),
    "loss": frozenset({"data", "physics"}),
    "evaluation": frozenset({"metrics", "objective"}),
    "optimizer": frozenset({"kind", "lr", "weight_decay", "betas", "second_moment_floor"}),
    "scheduler": frozenset({"kind", "factor", "patience", "min_lr"}),
    "training": frozenset(
        {
            "epochs",
            "stage_schedule",
            "evaluation_interval",
            "additional_evaluation_interval",
            "mixed_precision",
            "stage",
            "comparison_arm",
            "gradient_accumulation_steps",
            "curriculum",
            "teacher_handoff",
            "matched_a1_b",
        }
    ),
    "tracking": frozenset({"wandb"}),
}


def _as_mapping(value: Any, *, path: str) -> dict[str, Any]:
    """Return a mutable mapping copy with path-rich type errors."""
    if not isinstance(value, Mapping):
        msg = f"{path} must be a mapping, got {type(value).__name__}."
        raise ConfigError(msg)
    return dict(value)


def _is_positive_finite(value: object) -> bool:
    """Return whether one YAML value is a positive finite real number, rejecting booleans."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def _non_negative_number(value: object, *, path: str) -> float:
    """Return one finite non-negative YAML number; reject booleans and numeric text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        msg = f"{path} must be a finite non-negative number, got {value!r}."
        raise ConfigError(msg)
    return float(value)


def _non_negative_integer(value: object, *, path: str) -> int:
    """Return one non-negative YAML integer; reject booleans, floats, and numeric text."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{path} must be a non-negative integer, got {value!r}."
        raise ConfigError(msg)
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    """Reject keys outside one strict schema node."""
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        msg = f"{path} contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ConfigError(msg)


def _validate_input_schema(user_config: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912
    """
    Reject noncanonical task-fixed overrides and unknown nested keys.

    Validation walks every user-addressable schema node before defaults are
    merged. It admits only semantic selectors, rejects derived task contracts
    and channel counts, and reports the exact dotted path of an unsupported key.
    The supplied mapping is inspected without mutation.
    """
    fixed = sorted(set(user_config).intersection(_TASK_FIXED_KEYS))
    if fixed:
        msg = f"Task-fixed config key(s) cannot be overridden: {fixed}. Select a registered task instead."
        raise ConfigError(msg)
    _reject_unknown(user_config, _ROOT_KEYS, path="config")

    for section, allowed in _SECTION_KEYS.items():
        if section not in user_config or user_config[section] is None:
            continue
        section_mapping = _as_mapping(user_config[section], path=section)
        _reject_unknown(section_mapping, allowed, path=section)

    raw_run = _as_mapping(user_config.get("run"), path="run")
    derived_run_keys = sorted({"name", "naming_schema_version", "layout"}.intersection(raw_run))
    if derived_run_keys:
        msg = f"Derived run key(s) must not be supplied by an executable request: {derived_run_keys}."
        raise ConfigError(msg)

    model = _as_mapping(user_config.get("model"), path="model")
    params = _as_mapping(model.get("params"), path="model.params")
    fixed_channels = sorted({"in_channels", "out_channels"}.intersection(params))
    if fixed_channels:
        msg = f"model.params task-fixed channel key(s) cannot be overridden: {fixed_channels}."
        raise ConfigError(msg)

    if "loss" in user_config:
        loss = _as_mapping(user_config["loss"], path="loss")
        if "data" in loss:
            data_loss = _as_mapping(loss["data"], path="loss.data")
            _reject_unknown(data_loss, frozenset({"kind", "space", "weight", "beta", "channel_weights", "state_aux_weight"}), path="loss.data")
        if "physics" in loss:
            physics = _as_mapping(loss["physics"], path="loss.physics")
            _reject_unknown(
                physics,
                frozenset(
                    {
                        "enabled",
                        "derivatives",
                        "interior_crop",
                        "momentum_weight",
                        "continuity_weight",
                        "boundary_weight",
                    }
                ),
                path="loss.physics",
            )
            if "derivatives" in physics:
                derivatives = _as_mapping(physics["derivatives"], path="loss.physics.derivatives")
                _reject_unknown(
                    derivatives,
                    frozenset({"kind", "extension"}),
                    path="loss.physics.derivatives",
                )
            for weight_name in _PHYSICS_WEIGHT_NAMES:
                if weight_name not in physics:
                    continue
                weight = _as_mapping(physics[weight_name], path=f"loss.physics.{weight_name}")
                _reject_unknown(
                    weight,
                    frozenset({"target", "warmup"}),
                    path=f"loss.physics.{weight_name}",
                )
                if "warmup" in weight:
                    warmup = _as_mapping(weight["warmup"], path=f"loss.physics.{weight_name}.warmup")
                    _reject_unknown(
                        warmup,
                        frozenset({"kind", "epochs"}),
                        path=f"loss.physics.{weight_name}.warmup",
                    )

    if "tracking" in user_config:
        tracking = _as_mapping(user_config["tracking"], path="tracking")
        if "wandb" in tracking:
            wandb = _as_mapping(tracking["wandb"], path="tracking.wandb")
            _reject_unknown(
                wandb,
                frozenset(
                    {
                        "mode",
                        "workflow",
                        "study",
                        "monitor",
                        "upload",
                        "metric_schema_version",
                    }
                ),
                path="tracking.wandb",
            )
            if "monitor" in wandb:
                monitor = _as_mapping(
                    wandb["monitor"],
                    path="tracking.wandb.monitor",
                )
                _reject_unknown(
                    monitor,
                    frozenset({"enabled", "interval", "max_cases"}),
                    path="tracking.wandb.monitor",
                )
            if "upload" in wandb:
                upload = _as_mapping(
                    wandb["upload"],
                    path="tracking.wandb.upload",
                )
                _reject_unknown(
                    upload,
                    frozenset({"evaluation_artifacts"}),
                    path="tracking.wandb.upload",
                )

    if "evaluation" not in user_config:
        msg = "evaluation.objective is required for every executable request."
        raise ConfigError(msg)
    evaluation = _as_mapping(user_config["evaluation"], path="evaluation")
    if "metrics" in evaluation:
        metrics = evaluation["metrics"]
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
            msg = "evaluation.metrics must be a list of metric mappings."
            raise ConfigError(msg)
        for index, raw_metric in enumerate(metrics):
            metric = _as_mapping(raw_metric, path=f"evaluation.metrics[{index}]")
            _reject_unknown(
                metric,
                frozenset({"id", "kind", "space", "fields", "reduction"}),
                path=f"evaluation.metrics[{index}]",
            )
    if "objective" not in evaluation:
        msg = "evaluation.objective is required for every executable request."
        raise ConfigError(msg)
    objective = _as_mapping(evaluation["objective"], path="evaluation.objective")
    _reject_unknown(
        objective,
        frozenset({"id"}),
        path="evaluation.objective",
    )
    objective_id = objective.get("id")
    if not isinstance(objective_id, str) or not objective_id:
        msg = "evaluation.objective.id must be a non-empty string."
        raise ConfigError(msg)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """
    Load one YAML experiment mapping under the strict schema.

    Parameters
    ----------
    path : Path or str
        YAML source path.

    Returns
    -------
    dict[str, Any]
        Raw semantic experiment mapping.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ConfigError
        If the YAML root is not a mapping.

    """
    source_path = Path(path)
    if not source_path.exists():
        msg = f"Config file not found: {source_path}"
        raise FileNotFoundError(msg)
    with source_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        msg = f"YAML root must be a mapping: {source_path}"
        raise ConfigError(msg)
    return dict(payload)


def save_yaml(config: dict[str, Any], path: Path | str) -> None:
    """
    Save a resolved semantic config mapping.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved semantic configuration.
    path : Path or str
        Destination YAML path.

    Notes
    -----
    Serialization is assembled in memory and published through atomic text
    replacement. Callers never observe a partially written config.

    """
    destination = Path(path)
    stream = StringIO()
    yaml.dump(config, stream, default_flow_style=False, sort_keys=False)
    common.serialization.atomic_write_text(destination, stream.getvalue())


def apply_authored_base(mapping: Mapping[str, Any], *, origin: Path | str) -> dict[str, Any]:
    """
    Merge the optional shared task base named by one authored experiment mapping.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Authored experiment mapping, optionally with a ``base`` key.
    origin : Path or str
        YAML file the mapping was read from; ``base`` is relative to it.

    Returns
    -------
    dict[str, Any]
        Independent mapping without ``base``; authored values override the
        base values through ``deep_merge``.

    Raises
    ------
    ConfigError
        If ``base`` does not name ``base.yaml`` in the authored file's directory
        or an enclosing directory named after the mapping's task, or if that
        base names another base.

    """
    authored = dict(mapping)
    if AUTHORED_BASE_KEY not in authored:
        return authored
    reference = authored.pop(AUTHORED_BASE_KEY)
    source = Path(origin).expanduser().resolve()
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        msg = f"{source}: base must be a relative path to the task's base.yaml."
        raise ConfigError(msg)
    base_path = (source.parent / reference).resolve()
    task = authored.get("task")
    if (
        base_path.name != AUTHORED_BASE_FILENAME
        or base_path.parent not in (source.parent, *source.parent.parents)
        or base_path.parent.name != task
        or not base_path.is_file()
    ):
        msg = f"{source}: base {reference!r} must name base.yaml in the enclosing {task!r} task directory."
        raise ConfigError(msg)
    shared = load_yaml(base_path)
    if AUTHORED_BASE_KEY in shared:
        msg = f"{base_path}: a task base cannot name another base."
        raise ConfigError(msg)
    return deep_merge(shared, authored)


def load_authored_config(path: Path | str) -> dict[str, Any]:
    """Load one authored experiment YAML with its optional shared task base merged."""
    return apply_authored_base(load_yaml(path), origin=path)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge mappings while replacing scalar and list leaves.

    Parameters
    ----------
    base : dict[str, Any]
        Base mapping copied before merge.
    override : Mapping[str, Any]
        Values that override matching base paths.

    Returns
    -------
    dict[str, Any]
        Independent merged mapping.

    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_loss(config: dict[str, Any], *, task: domain.tasks.spec.TaskSpec) -> None:
    """
    Resolve loss semantics against the registered task contract in place.

    The helper validates supervised space/weight, derivative/extension
    compatibility, crop size, and the three physics-term linear warmup
    schedules. Canonical numeric values are written back into ``config`` only
    after their path-qualified constraints have been checked.
    """
    loss = _as_mapping(config["loss"], path="loss")
    data_loss = _as_mapping(loss["data"], path="loss.data")
    kind = str(data_loss["kind"])
    if kind not in task.data_losses:
        msg = f"loss.data.kind {kind!r} is not allowed by task {task.id!r}: {list(task.data_losses)}."
        raise ConfigError(msg)
    try:
        loss_factory.validate_data_loss_semantics(kind, space=str(data_loss["space"]))
    except ValueError as error:
        msg = f"loss.data: {error}"
        raise ConfigError(msg) from error
    data_loss["weight"] = _non_negative_number(data_loss["weight"], path="loss.data.weight")

    physics = _as_mapping(loss["physics"], path="loss.physics")
    if not isinstance(physics["enabled"], bool):
        msg = "loss.physics.enabled must be a boolean."
        raise ConfigError(msg)
    selected_physics = domain.tasks.registry.resolve_physics(task.physics.kind)
    if selected_physics != task.physics:
        msg = f"Task {task.id!r} physics registry entry does not match its task contract."
        raise ConfigError(msg)
    derivatives = _as_mapping(physics["derivatives"], path="loss.physics.derivatives")
    try:
        domain.physics.contracts.validate_derivative_kind(
            str(derivatives["kind"]),
            extension=str(derivatives["extension"]),
        )
    except ValueError as error:
        msg = f"loss.physics.derivatives: {error}"
        raise ConfigError(msg) from error
    physics["interior_crop"] = _non_negative_integer(physics["interior_crop"], path="loss.physics.interior_crop")

    for weight_name in _PHYSICS_WEIGHT_NAMES:
        weight_config = _as_mapping(physics[weight_name], path=f"loss.physics.{weight_name}")
        weight_config["target"] = _non_negative_number(weight_config["target"], path=f"loss.physics.{weight_name}.target")
        warmup = _as_mapping(weight_config["warmup"], path=f"loss.physics.{weight_name}.warmup")
        if warmup["kind"] != "linear":
            msg = f"Unknown warmup identifier {warmup['kind']!r} at loss.physics.{weight_name}.warmup.kind. Expected 'linear'."
            raise ConfigError(msg)
        warmup["epochs"] = _non_negative_integer(warmup["epochs"], path=f"loss.physics.{weight_name}.warmup.epochs")
        weight_config["warmup"] = warmup
        physics[weight_name] = weight_config

    loss["data"] = data_loss
    loss["physics"] = physics
    config["loss"] = loss


def _metric_fields(
    metric: dict[str, Any],
    *,
    task: domain.tasks.spec.TaskSpec,
    path: str,
) -> tuple[str, ...]:
    """
    Validate and canonicalize one metric's output-field selection.

    ``all`` expands in exact TaskSpec output order. Empty, duplicate, or unknown
    fields raise ``ConfigError`` at ``path``.
    """
    available_fields = task.metric_names if task.id == "transient_drying" else task.output_names
    raw_fields = metric.get("fields", "all")
    if raw_fields == "all":
        fields = available_fields
        metric["fields"] = list(fields)
        return fields
    if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, Sequence):
        msg = f"{path}.fields must be 'all' or a non-empty list of output fields."
        raise ConfigError(msg)
    fields = tuple(str(field) for field in raw_fields)
    if not fields:
        msg = f"{path}.fields must not be empty."
        raise ConfigError(msg)
    if len(fields) != len(set(fields)):
        msg = f"{path}.fields contains duplicates: {list(fields)}."
        raise ConfigError(msg)
    unknown = [field for field in fields if field not in available_fields]
    if unknown:
        msg = f"{path}.fields references unknown task metric field(s): {unknown}. Available fields: {list(available_fields)}."
        raise ConfigError(msg)
    metric["fields"] = list(fields)
    return fields


def _validate_resolved_metric_keys(config: Mapping[str, Any]) -> None:
    """
    Require every resolved metric to use the exact six-field canonical schema.

    This fail-closed pass prevents unresolved shorthand or omitted direction from
    entering saved configs before metric/objective revalidation.
    """
    evaluation = _as_mapping(config.get("evaluation"), path="evaluation")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        return
    expected_keys = {"id", "kind", "space", "fields", "reduction", "direction"}
    for index, raw_metric in enumerate(metrics):
        path = f"evaluation.metrics[{index}]"
        metric = _as_mapping(raw_metric, path=path)
        if set(metric) != expected_keys:
            msg = f"Resolved {path} must contain exactly {sorted(expected_keys)}, got {sorted(metric)}."
            raise ConfigError(msg)


def _validate_evaluation(config: dict[str, Any], *, task: domain.tasks.spec.TaskSpec) -> None:
    """
    Resolve metric declarations and materialize one complete objective in place.

    Each metric is bound to an exact tensor space, ordered field set, reduction,
    direction, and unit-compatible task contract. The selected objective becomes
    a full copy of exactly one declared metric. Partial or contradictory resolved
    objective mappings fail closed.
    """
    evaluation = _as_mapping(config["evaluation"], path="evaluation")
    raw_metrics = evaluation["metrics"]
    if not isinstance(raw_metrics, list) or not raw_metrics:
        msg = "evaluation.metrics must be a non-empty list."
        raise ConfigError(msg)

    metrics: list[dict[str, Any]] = []
    metric_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_metric in enumerate(raw_metrics):
        path = f"evaluation.metrics[{index}]"
        metric = _as_mapping(raw_metric, path=path)
        metric_id = metric.get("id")
        if not isinstance(metric_id, str) or not metric_id:
            msg = f"{path}.id must be a non-empty string."
            raise ConfigError(msg)
        if metric_id in metric_by_id:
            msg = f"Duplicate evaluation metric id {metric_id!r} at {path}."
            raise ConfigError(msg)
        kind = str(metric.get("kind"))
        space = str(metric.get("space"))
        reduction = str(metric.get("reduction"))
        try:
            metric_kind = metric_registry.validate_metric_semantics(
                kind,
                space=space,
                reduction=reduction,
            )
        except ValueError as error:
            msg = f"{path}: {error}"
            raise ConfigError(msg) from error
        fields = _metric_fields(metric, task=task, path=path)
        group_kinds = {"group_macro_rmse", "group_rmse", "vector_rmse"}
        if kind == "group_macro_rmse":
            if not task.output_groups or fields != task.output_names:
                msg = f"{path} group_macro_rmse must select every grouped TaskSpec output in declared order: {list(task.output_names)}."
                raise ConfigError(msg)
        elif kind in {"group_rmse", "vector_rmse"}:
            matches = [group for group in task.output_groups if group.fields == fields]
            if len(matches) != 1:
                available = {group.id: list(group.fields) for group in task.output_groups}
                msg = f"{path} fields must match one complete TaskSpec output group. Available groups: {available}."
                raise ConfigError(msg)
        if space == "physical" and len(fields) != 1 and kind not in group_kinds:
            units = {task.field(field).unit for field in fields}
            if len(units) > 1:
                msg = f"{path} cannot aggregate physical fields with incompatible units: {sorted(units)}."
                raise ConfigError(msg)
            msg = f"{path} physical metrics must select exactly one output field."
            raise ConfigError(msg)
        if kind == "vector_rmse":
            units = {task.field(field).unit for field in fields}
            if len(units) != 1:
                msg = f"{path} vector_rmse fields must share one physical unit, got {sorted(units)}."
                raise ConfigError(msg)
        requested_direction = metric.get("direction", metric_kind.direction)
        if requested_direction != metric_kind.direction:
            msg = f"{path}.direction {requested_direction!r} contradicts metric {kind!r} direction {metric_kind.direction!r}."
            raise ConfigError(msg)
        metric["direction"] = metric_kind.direction
        metrics.append(metric)
        metric_by_id[metric_id] = metric

    selection = _as_mapping(evaluation["objective"], path="evaluation.objective")
    objective_id = selection.get("id")
    if not isinstance(objective_id, str) or not objective_id:
        msg = "evaluation.objective.id must be a non-empty metric identifier."
        raise ConfigError(msg)
    if objective_id not in metric_by_id:
        msg = f"evaluation.objective.id {objective_id!r} is not a declared metric id. Available ids: {sorted(metric_by_id)}."
        raise ConfigError(msg)

    selected = metric_by_id[objective_id]
    objective = {
        "id": selected["id"],
        "kind": selected["kind"],
        "space": selected["space"],
        "fields": copy.deepcopy(selected["fields"]),
        "reduction": selected["reduction"],
        "direction": selected["direction"],
    }
    selection_keys = set(selection)
    if selection_keys != {"id"} and selection != objective:
        msg = f"Resolved evaluation.objective must exactly equal its selected metric definition. Expected {objective!r}, received {selection!r}."
        raise ConfigError(msg)

    evaluation["metrics"] = metrics
    evaluation["objective"] = objective
    config["evaluation"] = evaluation


def get_resolved_objective(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the complete canonical objective from a resolved experiment config.

    Parameters
    ----------
    config : Mapping[str, Any]
        Candidate resolved config containing ``evaluation.objective`` and the
        corresponding metric declaration.

    Returns
    -------
    dict[str, Any]
        Isolated id, kind, space, ordered fields, reduction, and direction.

    Raises
    ------
    ConfigError
        If the objective is partial, unsupported, or inconsistent with metrics.

    """
    evaluation = _as_mapping(config.get("evaluation"), path="evaluation")
    objective = _as_mapping(evaluation.get("objective"), path="evaluation.objective")
    expected_keys = {"id", "kind", "space", "fields", "reduction", "direction"}
    if set(objective) != expected_keys:
        msg = f"Resolved evaluation.objective must contain exactly {sorted(expected_keys)}, got {sorted(objective)}."
        raise ConfigError(msg)
    for key in ("id", "kind", "space", "reduction"):
        if not isinstance(objective[key], str) or not objective[key]:
            msg = f"Resolved evaluation.objective.{key} must be a non-empty string."
            raise ConfigError(msg)
    fields = objective["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) and field for field in fields):
        msg = "Resolved evaluation.objective.fields must be a non-empty exact field list."
        raise ConfigError(msg)
    if len(fields) != len(set(fields)):
        msg = f"Resolved evaluation.objective.fields contains duplicates: {fields!r}."
        raise ConfigError(msg)
    if objective["direction"] not in {"minimize", "maximize"}:
        msg = "Resolved evaluation.objective.direction must be 'minimize' or 'maximize'."
        raise ConfigError(msg)

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        msg = "Resolved evaluation.metrics must be a list."
        raise ConfigError(msg)
    selected = [metric for metric in metrics if isinstance(metric, Mapping) and metric.get("id") == objective["id"]]
    if len(selected) != 1:
        msg = f"Resolved objective id {objective['id']!r} must select exactly one evaluation metric."
        raise ConfigError(msg)
    selected_objective = {key: copy.deepcopy(selected[0].get(key)) for key in expected_keys}
    if selected_objective != objective:
        msg = "Resolved evaluation.objective does not exactly match its evaluation metric definition."
        raise ConfigError(msg)
    return copy.deepcopy(objective)


def _configured_dataset_list(data: Mapping[str, Any], *, path: str, key: str, allow_empty: bool = False) -> tuple[str, ...]:
    """Return distinct configured Dataset IDs for one task-owned selection key."""
    value = data.get(key, [] if allow_empty else None)
    if not isinstance(value, list) or (not value and not allow_empty):
        msg = f"{path}.{key} must contain one or more logical dataset ids."
        raise ConfigError(msg)
    try:
        dataset_ids = tuple(common.paths.validate_logical_name(dataset_id, label=f"{path}.{key}[{index}]") for index, dataset_id in enumerate(value))
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if len(dataset_ids) != len(set(dataset_ids)):
        msg = f"{path}.{key} must not contain duplicates."
        raise ConfigError(msg)
    return dataset_ids


def _resolve_dataset_selections(config: dict[str, Any], *, authored_data: Mapping[str, Any]) -> None:
    """Bind exact Dataset IDs and explicit Evaluation assignments before training."""
    data = _as_mapping(config["data"], path="data")
    try:
        data["train_dataset"] = common.paths.validate_logical_name(authored_data["train_dataset"], label="data.train_dataset")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    key = "additional_evaluation_datasets"
    data[key] = list(_configured_dataset_list(authored_data, path="data", key=key, allow_empty=True))
    config["data"] = data


def resolved_model_variant(config: Mapping[str, Any]) -> str:
    """Return the shared architecture label, separately from physics and stage."""
    return learning.models.naming.architecture_label(config)


def derive_wandb_organization(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the canonical project, optional study, and minimal base tags."""
    tracking = _as_mapping(config.get("tracking"), path="tracking")
    wandb = _as_mapping(tracking.get("wandb"), path="tracking.wandb")
    workflow = wandb.get("workflow")
    if workflow not in config_defaults.WANDB_WORKFLOWS:
        msg = f"tracking.wandb.workflow must be one of {list(config_defaults.WANDB_WORKFLOWS)}, got {workflow!r}."
        raise ConfigError(msg)

    task = common.paths.validate_logical_name(config.get("task"), label="task")
    variant = config_naming.model_family(config)
    raw_study = wandb.get("study")
    if raw_study is not None:
        try:
            study = common.paths.validate_logical_name(raw_study, label="tracking.wandb.study")
        except ValueError as error:
            raise ConfigError(str(error)) from error
    else:
        study = None

    if workflow == "train":
        if study is not None:
            msg = "tracking.wandb.study is valid only for workflow='optuna_trial'."
            raise ConfigError(msg)
        tags = [variant]
    elif workflow == "optuna_trial":
        if study is None:
            msg = "tracking.wandb.study is required for workflow='optuna_trial'."
            raise ConfigError(msg)
        tags = [variant, "optuna"]
    else:
        if study is not None:
            msg = "Acceptance workflows do not accept tracking.wandb.study."
            raise ConfigError(msg)
        tags = []

    if len(tags) > config_defaults.WANDB_MAX_TAGS or len(tags) != len(set(tags)):
        msg = f"Derived W&B tags must be unique and contain at most {config_defaults.WANDB_MAX_TAGS} values."
        raise ConfigError(msg)
    return {
        "project": config_defaults.WANDB_TASK_PROJECTS[task],
        "entity": config_defaults.WANDB_ENTITY,
        "study": study,
        "tags": tags,
    }


def _validate_tracking(config: dict[str, Any], *, require_derived: bool) -> None:
    """Validate one central W&B mode and canonical derived organization."""
    tracking = _as_mapping(config["tracking"], path="tracking")
    _reject_unknown(tracking, frozenset({"wandb"}), path="tracking")
    wandb = _as_mapping(tracking.get("wandb"), path="tracking.wandb")
    _reject_unknown(
        wandb,
        frozenset(
            {
                "mode",
                "workflow",
                "study",
                "project",
                "entity",
                "tags",
                "monitor",
                "upload",
                "metric_schema_version",
            }
        ),
        path="tracking.wandb",
    )
    metric_schema_version = wandb.get("metric_schema_version")
    if metric_schema_version is None:
        if require_derived:
            msg = "Current resolved configs must persist tracking.wandb.metric_schema_version."
            raise ConfigError(msg)
        wandb["metric_schema_version"] = config_defaults.WANDB_METRIC_SCHEMA_VERSION
    elif (
        isinstance(metric_schema_version, bool)
        or not isinstance(metric_schema_version, int)
        or metric_schema_version != config_defaults.WANDB_METRIC_SCHEMA_VERSION
    ):
        msg = "tracking.wandb.metric_schema_version is unsupported."
        raise ConfigError(msg)

    mode = wandb.get("mode")
    if mode not in {"online", "offline", "disabled"}:
        msg = "tracking.wandb.mode must be 'online', 'offline', or 'disabled'."
        raise ConfigError(msg)

    monitor = _as_mapping(wandb.get("monitor"), path="tracking.wandb.monitor")
    _reject_unknown(monitor, frozenset({"enabled", "interval", "max_cases"}), path="tracking.wandb.monitor")
    if type(monitor.get("enabled")) is not bool:
        msg = "tracking.wandb.monitor.enabled must be boolean."
        raise ConfigError(msg)
    for key in ("interval", "max_cases"):
        value = monitor.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            msg = f"tracking.wandb.monitor.{key} must be a positive integer."
            raise ConfigError(msg)

    upload = _as_mapping(wandb.get("upload"), path="tracking.wandb.upload")
    _reject_unknown(upload, frozenset({"evaluation_artifacts"}), path="tracking.wandb.upload")
    if type(upload.get("evaluation_artifacts")) is not bool:
        msg = "tracking.wandb.upload.evaluation_artifacts must be boolean."
        raise ConfigError(msg)

    expected = derive_wandb_organization({**config, "tracking": {"wandb": wandb}})
    if require_derived:
        differences = sorted(key for key, value in expected.items() if wandb.get(key) != value)
        if differences:
            msg = f"Resolved W&B organization is noncanonical at key(s): {differences}."
            raise ConfigError(msg)
    else:
        wandb.update(expected)
    wandb["monitor"] = monitor
    wandb["upload"] = upload
    tracking["wandb"] = wandb
    config["tracking"] = tracking


def _validate_current_run_identity_schema(run: Mapping[str, Any]) -> None:
    """Validate the persisted run-naming schema and its required identity evidence."""
    if type(run.get("naming_schema_version")) is not int or run["naming_schema_version"] != RUN_NAMING_SCHEMA_VERSION:
        msg = "run.naming_schema_version is unsupported."
        raise ConfigError(msg)

    if type(run.get("seed")) is not int or run["seed"] < 0:
        msg = "run.seed must be a non-negative integer."
        raise ConfigError(msg)
    if type(run.get("deterministic")) is not bool:
        msg = "run.deterministic must be boolean."
        raise ConfigError(msg)


def _validate_runtime_sections(config: dict[str, Any], *, require_derived_tracking: bool) -> None:
    """
    Validate all generic runtime sections after semantic resolution.

    This final pass checks optimizer/scheduler identifiers, positive duration,
    mixed-precision type, logical dataset names, optional tracking policy, and
    the exact requested device vocabulary. It validates policy only: concrete
    device resolution remains a top-level execution-service responsibility.
    """
    optimizer = _as_mapping(config["optimizer"], path="optimizer")
    if optimizer["kind"] != "adamw":
        msg = f"Unknown optimizer identifier {optimizer['kind']!r}. Available optimizers: adamw."
        raise ConfigError(msg)
    if "lr" not in optimizer:
        msg = "optimizer.lr is required."
        raise ConfigError(msg)
    betas = optimizer["betas"]
    if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) != _ADAM_BETA_COUNT:
        msg = f"optimizer.betas must contain exactly two values, got {betas!r}."
        raise ConfigError(msg)

    scheduler = config.get("scheduler")
    if scheduler is not None:
        scheduler_mapping = _as_mapping(scheduler, path="scheduler")
        if scheduler_mapping["kind"] != "reduce_on_plateau":
            msg = f"Unknown scheduler identifier {scheduler_mapping['kind']!r}. Available schedulers: reduce_on_plateau."
            raise ConfigError(msg)

    training = _as_mapping(config["training"], path="training")
    if type(training["mixed_precision"]) is not bool:
        msg = f"training.mixed_precision must be boolean, got {training['mixed_precision']!r}."
        raise ConfigError(msg)
    if int(training["epochs"]) <= 0:
        msg = "training.epochs must be positive."
        raise ConfigError(msg)
    if int(training["evaluation_interval"]) <= 0:
        msg = "training.evaluation_interval must be positive."
        raise ConfigError(msg)
    if int(training["additional_evaluation_interval"]) <= 0:
        msg = "training.additional_evaluation_interval must be positive."
        raise ConfigError(msg)
    data = _as_mapping(config["data"], path="data")
    batch_size = data.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        msg = "data.batch_size must be a positive integer."
        raise ConfigError(msg)
    num_workers = data.get("num_workers")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        msg = "data.num_workers must be an integer >= 0."
        raise ConfigError(msg)
    if type(data.get("pin_memory")) is not bool or type(data.get("persistent_workers")) is not bool:
        msg = "data.pin_memory and data.persistent_workers must be boolean."
        raise ConfigError(msg)
    if config.get("task") == "transient_drying" and num_workers == 0 and data["persistent_workers"]:
        msg = "Transient data.persistent_workers requires data.num_workers > 0."
        raise ConfigError(msg)
    try:
        common.paths.validate_logical_name(data["train_dataset"], label="data.train_dataset")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    _configured_dataset_list(data, path="data", key="additional_evaluation_datasets", allow_empty=True)
    if config.get("task") == "steady_flow" and set(data) != _STEADY_DATA_KEYS:
        msg = f"Steady data keys must be exactly {sorted(_STEADY_DATA_KEYS)}; membership comes from the Dataset partition."
        raise ConfigError(msg)

    _validate_tracking(config, require_derived=require_derived_tracking)

    run = _as_mapping(config["run"], path="run")
    _validate_current_run_identity_schema(run)
    try:
        run["device"] = learning.device_policy.validate_device_policy(
            run.get("device"),
            path="run.device",
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    for key in ("name", "revision"):
        value = run.get(key)
        if value is None:
            continue
        try:
            common.paths.validate_logical_name(value, label=f"run.{key}")
        except ValueError as error:
            raise ConfigError(str(error)) from error


RUN_NAMING_SCHEMA_VERSION = 1
_EXACT_DATASET_DIGEST_TOKEN = re.compile(r"[0-9a-f]{16,64}\Z")


def allocation_includes_epochs(config: Mapping[str, Any]) -> bool:
    """Return whether the epoch budget is part of a run's experiment identity.

    A transient A0 (also as a stage-A-only Optuna trial) schedules its rollout
    curriculum and budget controller over its epoch budget, so another budget is
    another experiment. Every Optuna trial (``run.layout.trial`` is set) belongs to
    a study of one fixed trial budget and never extends it, so its budget is part
    of its identity too. Every other run treats epochs as training progress that a
    rerun may extend.
    """
    is_trial = config["run"]["layout"]["trial"] is not None
    return is_trial or (config["task"] == "transient_drying" and config["training"].get("comparison_arm") == "a0")


def layout_budget_epochs(config: Mapping[str, Any], *, family: str) -> int | None:
    """Return the epoch budget that names a config's output unit, else ``None``.

    An Optuna trial lives in a study directory named after the study's trial epoch
    budget. Every stage of a transient seed group (A0, B and A1) carries the A0
    epoch budget, because the A0's curriculum and controller identity depend on it.
    A steady run has none: its epochs extend the run in place.
    """
    if family == common.paths.OPTUNA_FAMILY:
        return config["training"]["epochs"]
    if config["task"] != "transient_drying":
        return None
    schedule = config["training"]["stage_schedule"]
    return schedule["a0_max_epochs"] if schedule["mode"] == "matched_a1_b" else schedule["total_epochs"]


def allocation_digest(config: Mapping[str, Any], *, stage: str | None = None) -> str:
    """Identify one experiment independently of labels, paths and, where allowed, epochs.

    Equal digests mean the same experiment: a later start may only continue or
    extend the run that owns it. The first 16 characters are the layout's
    ``a0_id`` of single runs and A0. An A0 excludes the branch-planning metadata
    of the B/A1 pairs that may follow it and keeps its epoch budget
    (:func:`allocation_includes_epochs`); B and A1 include the ``a0_id`` they
    start from.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved configuration.
    stage : str | None, optional
        Layout stage; defaults to ``config["run"]["layout"]["stage"]``.

    Returns
    -------
    str
        Lowercase SHA-256 digest of the scientific identity payload.

    """
    stage = config["run"]["layout"]["stage"] if stage is None else stage
    payload = learning.training.checkpoint.effective_config_identity_payload(config, include_epochs=allocation_includes_epochs(config))
    if stage == "a0":
        training = payload["training"]
        training.pop("stage_schedule", None)
        training.pop("teacher_handoff", None)
        training.pop("matched_a1_b", None)
    elif stage in {"b", "a1"}:
        # A continuation is only the same experiment when it starts from the same A0.
        payload["a0_id"] = config["run"]["layout"]["a0_id"]
    return common.serialization.canonical_json_sha256(payload)


def _derive_layout(
    config: Mapping[str, Any],
    source: config_naming.ConfigSource,
    *,
    a0_id: str | None,
    branch_id: str | None,
    stage: str | None,
    pair: str | None,
    trial: str | None,
) -> dict[str, Any]:
    """Derive the complete layout and run name as a pure function of config, source and revision."""
    resolved = copy.deepcopy(dict(config))
    run = resolved["run"]
    stage = "single" if stage is None else stage
    if source.family == common.paths.OPTUNA_FAMILY and trial is None:
        # The study's base config is a template; every sampled trial re-finalizes with its own leaf.
        trial = config_naming.trial_leaf(0, resolved, {})
    run["layout"] = {
        "task": resolved["task"],
        "dataset": resolved["data"]["train_dataset"],
        "family": source.family,
        "config_name": source.name,
        "model_label": config_naming.model_label(resolved),
        "seed": run["seed"],
        "revision": run.get("revision"),
        "budget_epochs": layout_budget_epochs(resolved, family=source.family),
        "stage": stage,
        "pair": pair,
        "trial": trial,
        "a0_id": a0_id,
        "branch_id": branch_id,
    }
    if a0_id is None:
        run["layout"]["a0_id"] = allocation_digest(resolved)[:16]
    run["name"] = config_naming.run_name(run["layout"])
    return resolved


def finalize_run_layout(
    config: Mapping[str, Any],
    *,
    a0_id: str | None = None,
    branch_id: str | None = None,
    stage: str | None = None,
    pair: str | None = None,
    trial: str | None = None,
) -> dict[str, Any]:
    """Re-derive a resolved config's layout and run name, keeping its config source.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved config whose ``run.layout`` already names its source (task, family
        and config name), as produced by :func:`resolve_config`.
    a0_id, branch_id : str | None, optional
        Exact A0 and branch identities. ``a0_id`` defaults to the epoch-free allocation digest.
    stage : str | None, optional
        ``single`` (default), ``a0``, ``b`` or ``a1``.
    pair : str | None, optional
        Pair key shared by a matched plan's B and A1 stages.
    trial : str | None, optional
        Optuna trial leaf; an Optuna-family config defaults to the study template leaf.

    Returns
    -------
    dict[str, Any]
        Independent copy with the re-derived ``run.layout`` and ``run.name``.

    """
    layout = _as_mapping(_as_mapping(config.get("run"), path="run").get("layout"), path="run.layout")
    try:
        source = config_naming.ConfigSource(layout["task"], layout["family"], layout["config_name"])
    except KeyError as error:
        msg = "run.layout must name its config source (task, family, config_name) before it is re-derived."
        raise ConfigError(msg) from error
    return _derive_layout(config, source, a0_id=a0_id, branch_id=branch_id, stage=stage, pair=pair, trial=trial)


def _validate_run_layout(config: Mapping[str, Any], *, verify_identity: bool) -> dict[str, Any]:
    """Share lexical validation while reserving live identity checks for execution."""
    run = _as_mapping(config.get("run"), path="run")
    layout = dict(_as_mapping(run.get("layout"), path="run.layout"))
    common.paths.resolve_config_run_dir(config)
    if (
        layout["task"] != config["task"]
        or layout["dataset"] != config["data"]["train_dataset"]
        or layout["seed"] != run["seed"]
        or layout["revision"] != run.get("revision")
        or layout["budget_epochs"] != layout_budget_epochs(config, family=layout["family"])
    ):
        msg = "Run layout task, dataset, seed, revision and epoch budget must match its saved configuration."
        raise ConfigError(msg)
    if verify_identity:
        if layout["model_label"] != config_naming.model_label(config):
            msg = "Model label contradicts the resolved model and physics-loss settings."
            raise ConfigError(msg)
        if layout["stage"] in {"single", "a0"}:
            expected = allocation_digest(config)[:16]
            if layout["a0_id"] != expected:
                msg = "Initial run layout identity does not match its scientific dependencies."
                raise ConfigError(msg)
    return layout


_TRANSIENT_STATE_CHANNELS = 4


def _raise_config_error(message: str) -> NoReturn:
    """Raise one path-owned configuration violation from a constructed message."""
    raise ConfigError(message)


def _validate_transient_loss(config: dict[str, Any]) -> None:
    """Validate the data-only scaled-increment loss branch."""
    loss = _as_mapping(config["loss"], path="loss")
    data = _as_mapping(loss["data"], path="loss.data")
    required = {"kind", "space", "weight", "beta", "channel_weights", "state_aux_weight"}
    if set(data) != required or data["kind"] not in {"huber", "mse"} or data["space"] != "scaled_increment":
        _raise_config_error("Transient loss.data must be the complete scaled-increment huber/mse mapping.")
    if any(
        isinstance(data[key], bool) or not isinstance(data[key], (int, float)) or not math.isfinite(float(data[key])) or float(data[key]) < 0
        for key in ("weight", "state_aux_weight")
    ):
        _raise_config_error("Transient loss weights must be non-negative finite numbers.")
    if (
        isinstance(data["beta"], bool)
        or not isinstance(data["beta"], (int, float))
        or not math.isfinite(float(data["beta"]))
        or float(data["beta"]) <= 0
    ):
        _raise_config_error("Transient loss.data.beta must be positive and finite.")
    weights = data["channel_weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != _TRANSIENT_STATE_CHANNELS
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in weights
        )
        or sum(float(value) for value in weights) <= 0
    ):
        _raise_config_error("Transient loss.data.channel_weights must contain four non-negative finite values with positive sum.")
    physics = _as_mapping(loss["physics"], path="loss.physics")
    if physics != {"enabled": False}:
        _raise_config_error("Transient physics must be exactly disabled.")
    config["loss"] = {
        "data": {
            **data,
            "weight": float(data["weight"]),
            "beta": float(data["beta"]),
            "channel_weights": [float(value) for value in weights],
            "state_aux_weight": float(data["state_aux_weight"]),
        },
        "physics": physics,
    }


def resolve_transient_stage_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve canonical transient epoch-allocation evidence."""
    raw = _as_mapping(schedule, path="training.stage_schedule")
    mode = raw.get("mode")
    if mode == "matched_a1_b":
        expected_keys = {"mode", "budget_unit", "a0_max_epochs", "b_max_epochs", "a1_max_epochs"}
        if set(raw) != expected_keys or raw.get("budget_unit") != "train_seconds":
            _raise_config_error("matched_a1_b stage_schedule requires exact train_seconds and per-stage max epochs.")
        if any(type(raw[key]) is not int or raw[key] < 1 for key in ("a0_max_epochs", "b_max_epochs", "a1_max_epochs")):
            _raise_config_error("matched_a1_b max epochs must be positive integers.")
        return dict(raw)
    if mode != "stage_a_only" or set(raw) != {"mode", "budget_unit", "total_epochs"} or raw.get("budget_unit") != "epochs":
        _raise_config_error("training.stage_schedule must be matched_a1_b or exactly {mode: stage_a_only, budget_unit: epochs, total_epochs}.")
    total = raw.get("total_epochs")
    if type(total) is not int or total < 1:
        _raise_config_error("training.stage_schedule.total_epochs must be a positive integer.")
    return dict(raw)


def _resolve_temporal_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the exact temporal branch of a transient experiment: both explicit Dataset sampling and Learning conditioning choices."""
    required = {"sampling", "temporal_conditioning"}
    if set(value) != required:
        message = f"Transient temporal configuration keys must be exactly {sorted(required)}."
        raise ValueError(message)
    return {
        "sampling": datasets.contracts.transient.TransientSamplingSpec.from_mapping(value["sampling"]).as_dict(),
        "temporal_conditioning": learning.temporal.TemporalConditioningSpec.from_mapping(value["temporal_conditioning"]).as_dict(),
    }


def _validate_transient_extensions(  # noqa: C901, PLR0912, PLR0915
    config: dict[str, Any], *, authored: Mapping[str, Any] | None = None
) -> None:
    """Resolve strict transient-only temporal, loader, and curriculum semantics."""
    if authored is not None:
        for key in ("input_profile", "temporal", "scaling"):
            if key not in authored:
                _raise_config_error(f"Transient config requires top-level {key!r}.")
    task = domain.tasks.registry.get_task("transient_drying")
    profile = config.get("input_profile")
    if not isinstance(profile, str):
        _raise_config_error("input_profile must be a registered transient profile identifier.")
    try:
        task.input_profile(profile)
        temporal = _resolve_temporal_config(_as_mapping(config["temporal"], path="temporal"))
    except (TypeError, ValueError) as error:
        msg = f"temporal/input_profile: {error}"
        raise ConfigError(msg) from error
    scaling = _as_mapping(config.get("scaling"), path="scaling")
    if set(scaling) != {"mode"} or scaling["mode"] not in {"state_std", "delta_rms"}:
        _raise_config_error("scaling must be exactly {'mode': 'state_std' | 'delta_rms'}.")
    tensorizer = TransientTensorizerSpec.from_mapping({"input_profile": profile, "temporal_conditioning": temporal["temporal_conditioning"]})
    config["input_profile"] = profile
    config["temporal"] = temporal
    config["scaling"] = {"mode": scaling["mode"]}
    data = _as_mapping(config["data"], path="data")
    if "spatial_stride" not in data:
        data["spatial_stride"] = 1
    required_data = {
        "train_dataset",
        "additional_evaluation_datasets",
        "additional_evaluation_fraction",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "spatial_stride",
        "transient_backend_preference",
        "transient_backend_required",
        "hdf5_cache_size",
    }
    if set(data) != required_data:
        _raise_config_error("Transient data keys do not match the loader contract.")
    fraction = data["additional_evaluation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction != 1.0:
        _raise_config_error("Transient data.additional_evaluation_fraction must equal 1; Dataset-owned OOD membership is fixed.")
    if data["transient_backend_preference"] not in {"pt_shards", "canonical_hdf5"} or type(data["transient_backend_required"]) is not bool:
        _raise_config_error("Transient backend settings are invalid.")
    if isinstance(data["hdf5_cache_size"], bool) or not isinstance(data["hdf5_cache_size"], int) or data["hdf5_cache_size"] < 0:
        _raise_config_error("data.hdf5_cache_size must be an integer >= 0.")
    try:
        data["spatial_stride"] = datasets.contracts.transient.validate_spatial_stride(data["spatial_stride"])
    except (TypeError, ValueError) as error:
        message = f"data.spatial_stride: {error}"
        raise ConfigError(message) from error
    config["data"] = data
    training = _as_mapping(config["training"], path="training")
    required_training = {
        "epochs",
        "stage_schedule",
        "evaluation_interval",
        "additional_evaluation_interval",
        "mixed_precision",
        "stage",
        "comparison_arm",
        "gradient_accumulation_steps",
        "curriculum",
        "teacher_handoff",
    }
    if set(training) not in (required_training, required_training | {"matched_a1_b"}):
        _raise_config_error("Transient training keys do not match the curriculum contract.")
    schedule = resolve_transient_stage_schedule(_as_mapping(training["stage_schedule"], path="training.stage_schedule"))
    stage, arm = training["stage"], training["comparison_arm"]
    if stage not in {"a", "b"} or arm not in {"a0", "a1", "b"} or (stage == "a" and arm not in {"a0", "a1"}) or (stage == "b" and arm != "b"):
        _raise_config_error("Transient training stage and comparison_arm are inconsistent.")
    matched_protocol = schedule["mode"] == "matched_a1_b"
    expected_epochs = (
        schedule[{"a0": "a0_max_epochs", "b": "b_max_epochs", "a1": "a1_max_epochs"}[arm]] if matched_protocol else schedule["total_epochs"]
    )
    if training["epochs"] != expected_epochs:
        _raise_config_error("training.epochs must equal the resolved stage_schedule allocation.")
    accumulation = training["gradient_accumulation_steps"]
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        _raise_config_error("training.gradient_accumulation_steps must be a positive integer.")
    curriculum = _as_mapping(training["curriculum"], path="training.curriculum")
    if (
        set(curriculum) != {"lengths", "milestone_fractions", "seed"}
        or not isinstance(curriculum["lengths"], list)
        or not curriculum["lengths"]
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in curriculum["lengths"])
        or curriculum["lengths"] != sorted(set(curriculum["lengths"]))
    ):
        _raise_config_error("training.curriculum.lengths must be ascending unique positive integers.")
    fractions = curriculum["milestone_fractions"]
    if (
        not isinstance(fractions, list)
        or len(fractions) != len(curriculum["lengths"])
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not 0 <= float(item) <= 1 for item in fractions)
        or [float(item) for item in fractions] != sorted(float(item) for item in fractions)
    ):
        _raise_config_error("training.curriculum.milestone_fractions must align with lengths and be ascending in [0, 1].")
    if isinstance(curriculum["seed"], bool) or not isinstance(curriculum["seed"], int) or curriculum["seed"] < 0:
        _raise_config_error("training.curriculum.seed must be a non-negative integer.")
    try:
        resolved_curriculum = RolloutCurriculum(
            lengths=tuple(curriculum["lengths"]),
            milestone_fractions=tuple(float(item) for item in fractions),
        )
    except (TypeError, ValueError) as error:
        message = f"training.curriculum: {error}"
        raise ConfigError(message) from error
    sampling = temporal["sampling"]
    recurrent = config["model"]["kind"] == "rno"
    if recurrent != (sampling["mode"] == "trajectory"):
        _raise_config_error(
            "RNO carries its hidden state through whole trajectories and requires temporal.sampling.mode='trajectory'; "
            "stateless models train on one-step transitions or rollout windows."
        )
    if stage == "b" and sampling["mode"] == "one_step_transition":
        _raise_config_error("Stage B self-feeds rollout windows and cannot use one-step sampling.")
    if recurrent and sampling.get("rollout_length") is None:
        _raise_config_error("RNO trajectory sampling requires rollout_length, the window across which its hidden state is carried.")
    rollout_length = 1 if sampling["mode"] == "one_step_transition" else sampling["rollout_length"]
    if resolved_curriculum.lengths[-1] > rollout_length:
        _raise_config_error("training.curriculum lengths must fit the temporal.sampling rollout length.")
    handoff = training["teacher_handoff"]
    if handoff is not None and (
        not isinstance(handoff, Mapping)
        or set(handoff) != {"source_run_name"}
        or not isinstance(handoff["source_run_name"], str)
        or not handoff["source_run_name"]
    ):
        _raise_config_error("training.teacher_handoff must be null or contain only source_run_name.")
    requires_handoff = arm in {"b", "a1"}
    if requires_handoff != (handoff is not None):
        _raise_config_error("B and A1 require a teacher handoff; A0 forbids one.")
    protocol = training.get("matched_a1_b")
    if matched_protocol:
        if (
            arm not in {"a0", "b", "a1"}
            or not isinstance(protocol, Mapping)
            or set(protocol) != {"role", "train_seconds", "receipt_source_run_name", "initialization", "lr_scale", "evaluation_points"}
        ):
            _raise_config_error("matched_a1_b requires its exact stage-owned protocol mapping.")
        if protocol["role"] != arm or protocol["initialization"] != "a0_best_full_common_fresh_controller":
            _raise_config_error("matched_a1_b role or branch initialization is invalid.")
        seconds = protocol["train_seconds"]
        if (arm == "b") != _is_positive_finite(seconds):
            _raise_config_error("matched_a1_b B requires a positive train_seconds budget; A0/A1 derive no numeric target from YAML.")
        if (arm == "a1") != (isinstance(protocol["receipt_source_run_name"], str) and bool(protocol["receipt_source_run_name"])):
            _raise_config_error("matched_a1_b A1 must name its B receipt source, other stages must not.")
        if (arm != "a0") != _is_positive_finite(protocol["lr_scale"]):
            _raise_config_error("matched_a1_b B and A1 require a positive continuation lr_scale; A0 must not declare one.")
        points = protocol["evaluation_points"]
        if (arm != "a0") != (type(points) is int and points >= 1):
            _raise_config_error("matched_a1_b B and A1 require positive budget evaluation_points; A0 must not declare them.")
    elif protocol is not None or arm != "a0":
        _raise_config_error("Outside matched_a1_b, transient training is the stage-A-only A0 protocol.")
    config["training"] = {
        **training,
        "curriculum": {
            "lengths": list(curriculum["lengths"]),
            "milestone_fractions": [float(item) for item in fractions],
            "seed": curriculum["seed"],
        },
        "teacher_handoff": None if handoff is None else {"source_run_name": handoff["source_run_name"]},
    }
    config["_transient_tensorizer"] = tensorizer


def _reject_transient_roots_for_steady(config: Mapping[str, Any]) -> None:
    """Keep the steady configuration schema byte-for-byte semantic compatible."""
    unexpected = sorted({"input_profile", "temporal", "scaling"}.intersection(config))
    if unexpected:
        _raise_config_error(f"Steady-flow config does not admit transient root key(s): {unexpected}.")


def apply_epoch_budget(raw: Mapping[str, Any], epochs: int | None) -> dict[str, Any]:
    """Copy an authored request and apply an explicit epoch budget.

    Steady single-stage training uses ``training.epochs``. A stage-A-only A0 run
    sets ``training.epochs`` and its schedule total together. A matched plan sets
    A0's epoch budget ``training.a0.max_epochs``; B stops at its measured
    seconds and A1 at B's receipt, so their caps stay operational bounds.
    """
    requested = copy.deepcopy(dict(raw))
    if epochs is None:
        return requested
    if type(epochs) is not int or epochs < 1:
        msg = "--epochs must be a positive integer."
        raise ConfigError(msg)
    training = _as_mapping(requested.get("training"), path="training")
    schedule = training.get("stage_schedule")
    if training.get("mode") == "matched_a1_b":
        training["a0"] = {**_as_mapping(training.get("a0"), path="training.a0"), "max_epochs": epochs}
    elif isinstance(schedule, Mapping):
        training["stage_schedule"] = {**schedule, "total_epochs": epochs}
        training["epochs"] = epochs
    else:
        training["epochs"] = epochs
    requested["training"] = training
    return requested


def apply_training_overrides(
    raw: Mapping[str, Any],
    *,
    epochs: int | None = None,
    train_dataset: str | None = None,
    train_seconds: float | None = None,
    rollout_length: int | None = None,
    lr_scale: float | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Copy and bind explicit Dataset, matched-protocol, revision and epoch-budget overrides.

    ``rollout_length``, ``train_seconds`` and ``lr_scale`` apply only to an authored
    matched plan: the B window and its truncated curriculum, B's measured target,
    and the continuation learning-rate factor shared by B and A1 for a paired sweep.
    ``revision`` names a deliberately separate output location for otherwise identical
    science and is stored as ``run.revision``.
    """
    requested = copy.deepcopy(dict(raw))
    if revision is not None:
        try:
            validated_revision = common.paths.validate_logical_name(revision, label="--revision")
        except ValueError as error:
            raise ConfigError(str(error)) from error
        requested["run"] = {**_as_mapping(requested.get("run", {}), path="run"), "revision": validated_revision}
    if train_dataset is not None:
        requested["data"]["train_dataset"] = common.paths.validate_logical_name(train_dataset, label="--train-dataset")
    matched = (
        requested.get("task") == "transient_drying"
        and isinstance(requested.get("training"), Mapping)
        and (requested["training"].get("mode") == "matched_a1_b")
    )
    if not matched and any(value is not None for value in (rollout_length, train_seconds, lr_scale)):
        msg = "--rollout-length, --train-seconds and --lr-scale require a transient matched_a1_b plan."
        raise ConfigError(msg)
    if rollout_length is not None:
        if type(rollout_length) is not int or rollout_length not in {1, 2, 4, 8, 16, 32, 64}:
            msg = "--rollout-length must be an admitted curriculum horizon: 1, 2, 4, 8, 16, 32 or 64."
            raise ConfigError(msg)
        stage_b = requested["training"]["b"]
        stage_b["sampling"] = (
            {"mode": "trajectory", "rollout_length": rollout_length}
            if stage_b["sampling"]["mode"] == "trajectory"
            else {"mode": "rollout_window", "rollout_length": rollout_length, "window_stride": rollout_length, "window_offset": 0}
        )
        curriculum = stage_b["curriculum"]
        lengths = sorted({length for length in curriculum["lengths"] if length <= rollout_length} | {rollout_length})
        curriculum["lengths"] = lengths
        curriculum["milestone_fractions"] = [index / len(lengths) for index in range(len(lengths))]
    # The matched plan resolver validates both values after binding; they are copied unchanged here.
    if train_seconds is not None:
        requested["training"]["b"]["train_seconds"] = train_seconds
    if lr_scale is not None:
        requested["training"]["b"]["lr_scale"] = lr_scale
        requested["training"]["a1"]["lr_scale"] = lr_scale
    return apply_epoch_budget(requested, epochs)


def resolve_config(
    user_config: dict[str, Any],
    *,
    source: config_naming.ConfigSource,
    storage_root: Path | str | None = None,
    output_root: Path | str | None = None,
    naming_schema_version: int = RUN_NAMING_SCHEMA_VERSION,
) -> dict[str, Any]:
    """
    Strictly resolve one semantic experiment configuration.

    Parameters
    ----------
    user_config : dict[str, Any]
        Raw semantic experiment mapping.
    source : config_naming.ConfigSource
        Task, family and name of the YAML the mapping came from. The task must equal
        the config task, and an experiment family's name must extend the model label.
    storage_root : Path or str or None, optional
        Explicit Dataset-reference and resolved-path root for controlled callers.
    output_root : Path or str or None, optional
        Override the experiments root that run destinations descend from.
    naming_schema_version : int, optional
        Sole current derived run-label schema.

    Returns
    -------
    dict[str, Any]
        Fully resolved configuration with task contract, digest, channels, and paths.

    Raises
    ------
    ConfigError
        If the schema, identifiers, fields, settings or config name violate the contract.
    ValueError
        If a referenced semantic identifier is not registered.

    Notes
    -----
    Exact Dataset IDs and resolved Evaluation assignments are persisted before
    training; manifests and payloads are admitted by the Dataset owner. The layout
    is a pure function of the resolved config, ``source`` and ``run.revision``.

    """
    if not isinstance(user_config, Mapping):
        msg = "Experiment config must be a mapping."
        raise ConfigError(msg)
    _validate_input_schema(user_config)
    task_id = user_config.get("task")
    if not isinstance(task_id, str) or not task_id:
        msg = "Missing required non-empty config task identifier."
        raise ConfigError(msg)
    try:
        task = domain.tasks.registry.get_task(task_id)
    except ValueError as error:
        msg = f"config.task: {error}"
        raise ConfigError(msg) from error
    if source.task != task_id:
        msg = (
            f"Task config path mismatch for {source.family}/{source.name}: source task is {source.task!r}, "
            f"but config task is {task_id!r}. Move the YAML or set its task to {source.task!r}."
        )
        raise ConfigError(msg)
    authored_data = _as_mapping(user_config.get("data"), path="data")
    required_selection_keys = ("train_dataset",)
    missing_dataset_ids = [key for key in required_selection_keys if key not in authored_data]
    if missing_dataset_ids:
        msg = f"Experiment data must explicitly select Dataset identities; missing={missing_dataset_ids}."
        raise ConfigError(msg)
    if type(naming_schema_version) is not int or naming_schema_version != RUN_NAMING_SCHEMA_VERSION:
        msg = f"Unsupported requested run naming schema version: {naming_schema_version!r}."
        raise ConfigError(msg)

    effective = deep_merge(config_defaults.get_task_defaults(task_id), user_config)
    effective["task"] = task_id
    run = _as_mapping(effective["run"], path="run")
    run["naming_schema_version"] = naming_schema_version
    effective["run"] = run
    _resolve_dataset_selections(
        effective,
        authored_data=authored_data,
    )
    is_transient = task_id == "transient_drying"
    if is_transient:
        _validate_transient_extensions(effective, authored=user_config)
    else:
        _reject_transient_roots_for_steady(user_config)

    model = _as_mapping(effective["model"], path="model")
    kind = model.get("kind")
    if not isinstance(kind, str):
        msg = "model.kind is required and must be a semantic string identifier."
        raise ConfigError(msg)
    params = _as_mapping(model.get("params"), path="model.params")
    try:
        params = deep_merge(model_factory.model_defaults(kind), params)
        model_factory.validate_model_params(
            kind,
            params,
            require_channels=False,
            operator_dimensionality=task.operator_dimensionality,
        )
    except ValueError as error:
        msg = f"model: {error}"
        raise ConfigError(msg) from error
    if is_transient:
        tensorizer = effective.pop("_transient_tensorizer")
        if params.get("positional_embedding", None) is not None:
            msg = "Transient models require model.params.positional_embedding: null because x/y are explicit channels."
            raise ConfigError(msg)
        params["in_channels"] = tensorizer.in_channels
        params["out_channels"] = 4
        params["positional_embedding"] = None
    else:
        params["in_channels"] = task.in_channels
        params["out_channels"] = task.out_channels
    model["params"] = params
    effective["model"] = model

    optimizer = _as_mapping(effective["optimizer"], path="optimizer")
    optimizer_kind = str(optimizer.get("kind"))
    if optimizer_kind not in config_defaults.OPTIMIZER_DEFAULTS:
        msg = f"Unknown optimizer identifier {optimizer_kind!r}. Available optimizers: {sorted(config_defaults.OPTIMIZER_DEFAULTS)}."
        raise ConfigError(msg)
    effective["optimizer"] = deep_merge(config_defaults.OPTIMIZER_DEFAULTS[optimizer_kind], optimizer)

    scheduler = effective.get("scheduler")
    if scheduler is not None:
        scheduler_mapping = _as_mapping(scheduler, path="scheduler")
        scheduler_kind = str(scheduler_mapping.get("kind"))
        if scheduler_kind not in config_defaults.SCHEDULER_DEFAULTS:
            msg = f"Unknown scheduler identifier {scheduler_kind!r}. Available schedulers: {sorted(config_defaults.SCHEDULER_DEFAULTS)}."
            raise ConfigError(msg)
        effective["scheduler"] = deep_merge(config_defaults.SCHEDULER_DEFAULTS[scheduler_kind], scheduler_mapping)

    if is_transient:
        _validate_transient_loss(effective)
    else:
        _validate_loss(effective, task=task)
    _validate_evaluation(effective, task=task)
    _validate_runtime_sections(effective, require_derived_tracking=False)

    effective["task_contract"] = task.resolved_contract()
    effective["paths"] = {
        "project_root": str(common.paths.get_project_root()),
        "storage_root": str(common.paths.get_storage_root(storage_root=storage_root)),
        "dataset_root": str(common.paths.get_datasets_root(storage_root=storage_root)),
        "output_root": str(common.paths.get_experiments_root(storage_root=storage_root)),
    }
    if output_root is not None:
        effective["paths"]["output_root"] = str(Path(output_root).expanduser())
    config_naming.validate_config_name(source, effective)
    effective = _derive_layout(effective, source, a0_id=None, branch_id=None, stage=None, pair=None, trial=None)
    run = effective["run"]
    try:
        common.paths.validate_logical_name(run["name"], label="run.name")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    effective["run"] = run
    return effective


def validate_resolved_task_contract(config: Mapping[str, Any]) -> domain.tasks.spec.TaskSpec:
    """
    Validate the persisted task contract in an effective configuration.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved or saved semantic configuration.

    Returns
    -------
    domain.tasks.spec.TaskSpec
        Registered task matching the saved identifier and digest.

    Raises
    ------
    ConfigError
        If the complete task contract does not exactly match the registered task.
    ValueError
        If the task identifier is unknown.

    """
    task_id = config.get("task")
    if not isinstance(task_id, str):
        msg = "Resolved config must contain a string task identifier."
        raise ConfigError(msg)
    task = domain.tasks.registry.get_task(task_id)
    contract = config.get("task_contract")
    if not isinstance(contract, Mapping):
        msg = "Resolved config must contain the current task_contract."
        raise ConfigError(msg)
    expected_contract = task.resolved_contract()
    schema_version = contract.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != domain.tasks.spec.TASK_SCHEMA_VERSION:
        msg = (
            f"Resolved task contract does not exactly match registered task {task_id!r}: "
            f"schema_version must be integer {domain.tasks.spec.TASK_SCHEMA_VERSION}."
        )
        raise ConfigError(msg)
    if dict(contract) != expected_contract:
        msg = f"Resolved task contract does not exactly match registered task {task_id!r}."
        raise ConfigError(msg)
    return task


def validate_resolved_config(config: Mapping[str, Any], *, verify_layout_identity: bool = True) -> dict[str, Any]:
    """
    Validate and return an isolated canonical resolved experiment config.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved candidate, commonly loaded from a saved ``config.yaml``.
    verify_layout_identity : bool, optional
        Recompute the layout identity (model label, ``a0_id`` against the allocation
        digest) with the running scientific implementation. ``False`` keeps the
        structural checks only; use it where the implementation that will execute the
        config is not yet known, never to admit execution.

    Returns
    -------
    dict[str, Any]
        Deep-copied canonical config whose task contract, paths, model, loss,
        metrics, objective, runtime sections, and tracking policy all agree.

    Raises
    ------
    ConfigError
        If required derived values are absent or any semantic section drifts.

    Notes
    -----
    Validation is read-only: it allocates no run, dataset loader, tracker, or
    device and never rewrites the supplied mapping.

    """
    return _validate_resolved_config(config, verify_layout_identity=verify_layout_identity)


def validate_saved_artifact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted configuration for reading already-published predictions.

    Parameters
    ----------
    config : Mapping[str, Any]
        Saved resolved configuration whose exact bytes are bound by its run bundle.

    Returns
    -------
    dict[str, Any]
        Isolated configuration with admitted task, schema, semantic sections,
        paths and structural layout. The saved run validator additionally binds
        the initial layout to the persisted checkpoint identity.

    Raises
    ------
    ConfigError
        If a saved semantic or structural contract is inconsistent.

    Notes
    -----
    This entrypoint does not admit model execution. Resume and new inference
    require the current scientific implementation through validate_resolved_config.
    No historical implementation digest or run layout is regenerated or rewritten.

    """
    return _validate_resolved_config(config, verify_layout_identity=False)


def _validate_resolved_config(config: Mapping[str, Any], *, verify_layout_identity: bool) -> dict[str, Any]:
    """Validate shared saved schema/semantics and, on request, the live layout identity."""
    if not isinstance(config, Mapping):
        msg = "Resolved experiment config must be a mapping."
        raise ConfigError(msg)
    effective = copy.deepcopy(dict(config))
    task_id = effective.get("task")
    transient_roots = frozenset({"input_profile", "temporal", "scaling"})
    roots = _ROOT_KEYS if task_id == "transient_drying" else _ROOT_KEYS.difference(transient_roots)
    allowed = roots.union({"task_contract", "paths"})
    missing = sorted(allowed.difference(effective))
    unknown = sorted(set(effective).difference(allowed))
    if missing or unknown:
        msg = f"Resolved config keys do not match. Missing: {missing}. Unknown: {unknown}."
        raise ConfigError(msg)

    task = validate_resolved_task_contract(effective)
    is_transient = task.id == "transient_drying"
    if is_transient:
        _validate_transient_extensions(effective)
        tensorizer = effective.pop("_transient_tensorizer")
    else:
        _reject_transient_roots_for_steady(effective)
        tensorizer = None
    model = _as_mapping(effective["model"], path="model")
    kind = model.get("kind")
    if not isinstance(kind, str) or not kind:
        msg = "Resolved model.kind must be a non-empty semantic identifier."
        raise ConfigError(msg)
    params = _as_mapping(model.get("params"), path="model.params")
    try:
        model_factory.validate_model_params(
            kind,
            params,
            require_channels=True,
            operator_dimensionality=task.operator_dimensionality,
        )
    except ValueError as error:
        msg = f"model: {error}"
        raise ConfigError(msg) from error
    expected_in_channels = tensorizer.in_channels if tensorizer is not None else task.in_channels
    expected_out_channels = 4 if tensorizer is not None else task.out_channels
    if params.get("in_channels") != expected_in_channels or params.get("out_channels") != expected_out_channels:
        msg = "Resolved model channels do not match the task contract."
        raise ConfigError(msg)
    if tensorizer is not None and params.get("positional_embedding") is not None:
        msg = "Transient models require null positional_embedding because x/y are explicit channels."
        raise ConfigError(msg)
    model["params"] = params
    effective["model"] = model

    if is_transient:
        _validate_transient_loss(effective)
    else:
        _validate_loss(effective, task=task)
    _validate_resolved_metric_keys(effective)
    _validate_evaluation(effective, task=task)
    _validate_runtime_sections(effective, require_derived_tracking=True)
    get_resolved_objective(effective)
    _validate_run_layout(effective, verify_identity=verify_layout_identity)
    expected_run_name = config_naming.run_name(effective["run"]["layout"])
    if effective["run"]["name"] != expected_run_name:
        msg = f"Resolved run.name must equal the canonical generated leaf {expected_run_name!r}."
        raise ConfigError(msg)
    paths = _as_mapping(effective["paths"], path="paths")
    missing_paths = sorted(_RESOLVED_PATH_KEYS.difference(paths))
    unknown_paths = sorted(set(paths).difference(_RESOLVED_PATH_KEYS))
    if missing_paths or unknown_paths:
        msg = f"Resolved paths do not match the two-domain contract. Missing: {missing_paths}. Unknown: {unknown_paths}."
        raise ConfigError(msg)
    invalid_paths = sorted(key for key, value in paths.items() if not isinstance(value, str) or not value)
    if invalid_paths:
        msg = f"Resolved paths must contain non-empty strings. Invalid key(s): {invalid_paths}."
        raise ConfigError(msg)
    effective["paths"] = paths
    return effective


def load_and_resolve_config(yaml_path: Path | str) -> dict[str, Any]:
    """
    Load and strictly resolve one experiment YAML.

    Parameters
    ----------
    yaml_path : Path or str
        Semantic experiment YAML path.

    Returns
    -------
    dict[str, Any]
        Fully resolved semantic configuration.

    """
    raw = load_authored_config(yaml_path)
    return resolve_config(raw, source=config_naming.config_source_from_path(yaml_path))


def _effective_num_workers(configured: int) -> int:
    """Cap the configured loader workers at the CPUs this process may use and report the cap once when it applies."""
    available = len(os.sched_getaffinity(0))
    effective = min(configured, available)
    if effective < configured:
        print(f"data workers {effective} of {configured} configured ({available} CPUs)", flush=True)
    return effective


def create_dataloaders_from_config(
    config: dict[str, Any],
    *,
    split_indices: dict[str, Any] | None = None,
    data_processor: Any | None = None,
    seed_plan: Mapping[str, int] | None = None,
    startup_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    split_persist_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Create current dataloaders after validating the resolved task contract.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved experiment configuration.
    split_indices : dict[str, Any] or None, optional
        Existing split membership to reuse.
    data_processor : Any or None, optional
        Existing data processor passed through to the current loader.
    seed_plan : Mapping[str, int] | None, optional
        Stable labeled ``split``, ``loader``, and ``worker`` seeds. Defaults to
        ``run.seed`` for isolated direct loader callers.
    startup_callback : Callable or None, optional
        Presentation-owned receiver for bounded semantic startup events.
    split_persist_callback : Callable or None, optional
        Persist resolved Experiment membership before fitting train-only scalers.

    Returns
    -------
    dict[str, Any]
        Train/evaluation loaders, data processor, and resolved split membership.

    Raises
    ------
    ConfigError
        If the task contract or required data settings are invalid.

    Notes
    -----
    The shared task dataset factory validates schema and fingerprint before
    splitting. Loader construction may read dataset files, fit preprocessing
    state for a fresh run, or reuse caller-supplied split and processor state.
    Persistence remains the run lifecycle's responsibility.

    """
    task = validate_resolved_task_contract(config)
    data_cfg = _as_mapping(config.get("data"), path="data")
    num_workers = _effective_num_workers(data_cfg["num_workers"])
    if task.id == "transient_drying":
        tensorizer = TransientTensorizerSpec.from_mapping(
            {"input_profile": config["input_profile"], "temporal_conditioning": config["temporal"]["temporal_conditioning"]}
        )
        seeds = dict(seed_plan or {})
        run_seed = int(config["run"]["seed"])
        settings = datasets.runtime.factory.LoaderSettings(
            batch_size=data_cfg["batch_size"],
            num_workers=num_workers,
            pin_memory=data_cfg["pin_memory"],
            persistent_workers=data_cfg["persistent_workers"],
            hdf5_cache_size=data_cfg["hdf5_cache_size"],
        )
        loaders = datasets.runtime.transient_training.create_transient_training_loaders(
            train_dataset_id=data_cfg["train_dataset"],
            additional_evaluation_dataset_ids=data_cfg["additional_evaluation_datasets"],
            additional_evaluation_fraction=data_cfg["additional_evaluation_fraction"],
            tensorizer=tensorizer,
            train_sampling=datasets.contracts.transient.TransientSamplingSpec.from_mapping(config["temporal"]["sampling"]),
            loader_settings=settings,
            storage_root=config["paths"]["storage_root"],
            transient_backend_preference=data_cfg["transient_backend_preference"],
            transient_backend_required=data_cfg["transient_backend_required"],
            spatial_stride=data_cfg["spatial_stride"],
            scale_mode=config["scaling"]["mode"],
            loader_seed=seeds.get("loader", run_seed),
            worker_seed=seeds.get("worker", run_seed),
            saved_split=split_indices,
            restored_scaling_artifact=data_processor,
            startup_callback=startup_callback,
            split_persist_callback=split_persist_callback,
        )
        return {
            "train": loaders.train,
            "eval": loaders.evaluation,
            "additional_evaluation": loaders.additional_evaluation,
            "id_test": loaders.id_test,
            "data_processor": loaders.scaling_artifact,
            "split_indices": loaders.split,
            "dataset_identity": loaders.dataset_identity,
            "spatial_view": loaders.spatial_view,
            "runtime_provenance": loaders.runtime_provenance,
            "tensorizer": tensorizer,
        }
    seeds = dict(seed_plan or {})
    run_seed = int(config["run"]["seed"])
    train_loader, evaluation_loaders, normalizer, split_indices = datasets.runtime.training.create_dataloaders(
        train_dataset_id=common.paths.validate_logical_name(data_cfg["train_dataset"], label="data.train_dataset"),
        additional_evaluation_dataset_ids=tuple(data_cfg["additional_evaluation_datasets"]),
        task=task,
        storage_root=common.paths.storage_root_for_datasets_root(config["paths"]["dataset_root"]),
        batch_size=data_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg["persistent_workers"],
        loader_seed=seeds.get("loader", run_seed),
        worker_seed=seeds.get("worker", run_seed),
        split_indices=split_indices,
        data_processor=data_processor,
        split_persist_callback=split_persist_callback,
    )
    return {
        "train": train_loader,
        "eval": evaluation_loaders["eval"],
        "id_test": evaluation_loaders["id_test"],
        "additional_evaluation": evaluation_loaders["additional_evaluation"],
        "data_processor": normalizer,
        "split_indices": split_indices,
    }
