"""
experiments_config_transient_plan.py

Resolve authored two-stage transient-training plans into canonical run configurations.

Responsibilities:
  - Detect and strictly validate the authored transient two-stage training schema
  - Derive independent canonical A0 and B configurations through the normal loader
  - Bind Stage B teacher provenance to the derived Stage A run identity
  - Preserve task-directory identity validation for authored plan files

Design principles:
  - The existing resolved experiment schema remains the single runtime contract
  - Authored plan inputs are copied before derivation and never mutated
  - Stage identity is derived into run names instead of duplicated in authored YAML

This module does NOT:
  - Execute training, allocate run directories, or publish handoffs
  - Change Optuna, saved-config, resume, or matched-config schemas
  - Define temporal sampling or curriculum semantics
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from . import experiments_config_loader as loader

if TYPE_CHECKING:
    from pathlib import Path

StageName = Literal["a", "b"]
_STAGE_KEYS = frozenset({"mixed_precision", "stage_schedule", "stage_a", "stage_b"})
_STAGE_A_KEYS = frozenset(
    {
        "evaluation_interval",
        "ood_evaluation_interval",
        "gradient_accumulation_steps",
        "sampling",
        "fixed_evaluation_horizon",
        "curriculum",
    }
)
_STAGE_B_KEYS = _STAGE_A_KEYS.union({"matched_compute"})
_TEMPORAL_KEYS = frozenset({"temporal_conditioning"})
_MATCHED_KEYS = frozenset(
    {
        "planned_seconds",
        "planned_steps",
        "rollout_reference_seconds",
        "rollout_reference_steps",
    }
)


@dataclass(frozen=True, slots=True)
class TransientTrainingPlan:
    """Contain immutable references to one derived transient A0/B training pair."""

    stage_a: Mapping[str, Any]
    stage_b: Mapping[str, Any]

    def stage(self, name: StageName) -> Mapping[str, Any]:
        """Return the resolved configuration for one named training stage."""
        if name == "a":
            return self.stage_a
        if name == "b":
            return self.stage_b
        message = f"Unknown transient training stage {name!r}; expected 'a' or 'b'."
        raise ValueError(message)

    @property
    def stages(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Return derived configurations in mandatory execution order."""
        return (self.stage_a, self.stage_b)


def is_transient_two_stage_config(raw: Mapping[str, Any]) -> bool:
    """Return whether a raw mapping structurally declares authored stage branches."""
    training = raw.get("training")
    return isinstance(training, Mapping) and bool({"stage_a", "stage_b"}.intersection(training))


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    """Return an isolated mapping or raise the loader-owned configuration error."""
    if not isinstance(value, Mapping):
        message = f"{path} must be a mapping."
        raise loader.ConfigError(message)
    return copy.deepcopy(dict(value))


def _validate_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, path: str) -> None:
    """Require one authored node to contain its complete and exact schema."""
    actual = set(value)
    if actual != expected:
        message = f"{path} keys must be exactly {sorted(expected)}, got {sorted(actual)}."
        raise loader.ConfigError(message)


def _null_matched_compute() -> dict[str, None]:
    """Return the exact no-budget resolved A0 matched-compute mapping."""
    return dict.fromkeys(sorted(_MATCHED_KEYS))


def resolve_stage_epoch_allocation(stage_schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the loader-owned canonical transient epoch allocation."""
    return loader.resolve_transient_stage_schedule(stage_schedule)


def _derive_child(
    raw: Mapping[str, Any],
    *,
    stage: StageName,
    stage_config: Mapping[str, Any],
    stage_schedule: Mapping[str, Any],
    source_run_name: str | None,
    naming_schema_version: int,
) -> dict[str, Any]:
    """Build one ordinary raw child config from the authored shared plan mapping."""
    child = copy.deepcopy(dict(raw))
    temporal = _mapping(child.get("temporal"), path="temporal")
    temporal["sampling"] = copy.deepcopy(stage_config["sampling"])
    child["temporal"] = temporal
    run = _mapping(child.get("run"), path="run")
    if naming_schema_version == 1:
        stage_suffix = "stage_a0" if stage == "a" else "stage_b"
        run["suffix"] = stage_suffix if run.get("suffix") is None else f"{run['suffix']}_{stage_suffix}"
    child["run"] = run
    training: dict[str, Any] = {
        "epochs": stage_schedule["stage_a_epochs"] if stage == "a" else stage_schedule["stage_b_epochs"],
        "stage_schedule": copy.deepcopy(dict(stage_schedule)),
        "evaluation_interval": stage_config["evaluation_interval"],
        "ood_evaluation_interval": stage_config["ood_evaluation_interval"],
        "mixed_precision": raw["training"]["mixed_precision"],
        "stage": stage,
        "comparison_arm": "a0" if stage == "a" else "b",
        "gradient_accumulation_steps": stage_config["gradient_accumulation_steps"],
        "fixed_evaluation_horizon": stage_config["fixed_evaluation_horizon"],
        "curriculum": copy.deepcopy(stage_config["curriculum"]),
        "matched_compute": _null_matched_compute() if stage == "a" else copy.deepcopy(stage_config["matched_compute"]),
        "teacher_handoff": None if stage == "a" else {"source_run_name": source_run_name},
    }
    child["training"] = training
    return child


def resolve_transient_training_plan(
    raw: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
    naming_schema_version: int = loader.RUN_NAMING_SCHEMA_VERSION,
    pinned_dataset_references: Mapping[str, Any] | None = None,
) -> TransientTrainingPlan:
    """
    Resolve one authored transient two-stage plan into canonical A0 and B configs.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Authored transient plan with shared semantic sections and exact stage maps.
    storage_root : Path or str or None, optional
        Explicit root for controlled logical Dataset-reference resolution.
    naming_schema_version : int, optional
        Current concise or legacy saved-run naming schema.
    pinned_dataset_references : Mapping[str, Any] or None, optional
        Saved reference evidence reused without chasing current aliases.

    Returns
    -------
    TransientTrainingPlan
        Fully resolved, independently validated A0 and B configurations.

    Raises
    ------
    loader.ConfigError
        If authored plan structure or either derived child violates the runtime contract.

    """
    authored = copy.deepcopy(dict(raw))
    if authored.get("task") != "transient_drying":
        message = "Authored two-stage plans are supported only for task='transient_drying'."
        raise loader.ConfigError(message)
    training = _mapping(authored.get("training"), path="training")
    _validate_exact_keys(training, _STAGE_KEYS, path="training")
    temporal = _mapping(authored.get("temporal"), path="temporal")
    _validate_exact_keys(temporal, _TEMPORAL_KEYS, path="temporal")
    stage_schedule = resolve_stage_epoch_allocation(_mapping(training.get("stage_schedule"), path="training.stage_schedule"))
    stage_a = _mapping(training.get("stage_a"), path="training.stage_a")
    stage_b = _mapping(training.get("stage_b"), path="training.stage_b")
    _validate_exact_keys(stage_a, _STAGE_A_KEYS, path="training.stage_a")
    _validate_exact_keys(stage_b, _STAGE_B_KEYS, path="training.stage_b")
    _validate_exact_keys(
        _mapping(stage_b["matched_compute"], path="training.stage_b.matched_compute"), _MATCHED_KEYS, path="training.stage_b.matched_compute"
    )

    a_raw = _derive_child(
        authored,
        stage="a",
        stage_config=stage_a,
        stage_schedule=stage_schedule,
        source_run_name=None,
        naming_schema_version=naming_schema_version,
    )
    resolved_a = loader.resolve_config(
        a_raw,
        storage_root=storage_root,
        naming_schema_version=naming_schema_version,
        pinned_dataset_references=pinned_dataset_references,
    )
    resolved_a = loader.validate_resolved_config(resolved_a)
    b_raw = _derive_child(
        authored,
        stage="b",
        stage_config=stage_b,
        stage_schedule=stage_schedule,
        source_run_name=resolved_a["run"]["name"],
        naming_schema_version=naming_schema_version,
    )
    resolved_b = loader.resolve_config(
        b_raw,
        storage_root=storage_root,
        naming_schema_version=naming_schema_version,
        pinned_dataset_references=pinned_dataset_references,
    )
    resolved_b = loader.validate_resolved_config(resolved_b)
    handoff = resolved_b["training"]["teacher_handoff"]
    if handoff != {"source_run_name": resolved_a["run"]["name"]}:
        message = "Derived Stage B teacher handoff must bind exactly to the derived Stage A run name."
        raise loader.ConfigError(message)
    return TransientTrainingPlan(
        stage_a=MappingProxyType(resolved_a),
        stage_b=MappingProxyType(resolved_b),
    )


def load_and_resolve_transient_training_plan(
    yaml_path: Path | str,
    *,
    validate_task_identity: bool = True,
) -> TransientTrainingPlan:
    """Load, resolve, and optionally validate one authored transient two-stage plan file."""
    raw = loader.load_yaml(yaml_path)
    plan = resolve_transient_training_plan(raw)
    if validate_task_identity:
        loader.validate_task_directory_identity(
            yaml_path,
            raw_task=raw.get("task"),
            resolved_task=plan.stage_a.get("task"),
        )
    return plan
