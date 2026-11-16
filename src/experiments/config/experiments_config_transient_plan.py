"""Resolve authored matched A0/B/A1 transient plans into canonical child run configurations.

Responsibilities:
  - Detect and strictly validate the authored matched A0 -> {B, A1} training schema
  - Derive the A0, B and A1 child configurations through the normal loader
  - Derive A1's teacher-forced protocol from A0 so both arms share one sampling contract
  - Bind B and A1 teacher provenance, the shared branch identity and the pair key to the derived A0
  - Project a plan's A0 with another budget and A0's memory batch into the stage-A-only experiment an Optuna study tunes

Design principles:
  - The existing resolved experiment schema remains the single runtime contract
  - Authored plan inputs are copied before derivation and never mutated
  - Stage identity is derived into run names instead of duplicated in authored YAML
  - Only B draws rollout horizons, so only B has a curriculum; A0 and A1 have none

This module does NOT:
  - Execute training, allocate run directories, or publish handoffs
  - Choose an Optuna trial budget or change saved-config and resume schemas
  - Define temporal sampling or curriculum semantics
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from src import common, learning

from . import experiments_config_loader as loader
from . import experiments_config_naming as config_naming

if TYPE_CHECKING:
    from pathlib import Path

MatchedStageName = Literal["a0", "b", "a1"]
_TEMPORAL_KEYS = frozenset({"temporal_conditioning"})
_MATCHED_PLAN_KEYS = frozenset({"mode", "units_per_update", "a0", "b", "a1"})
# Every stage authors its own memory batch: an execution setting outside identity that changes only floating-point rounding.
_MATCHED_A0_KEYS = frozenset({"budget_epochs", "sampling", "memory_batch"})
_MATCHED_B_KEYS = frozenset({"sampling", "curriculum", "lr_scale", "memory_batch"})
_MATCHED_A1_KEYS = frozenset({"lr_scale", "memory_batch"})
_MATCHED_INITIALIZATION = "a0_best_full_common_fresh_controller"


@dataclass(frozen=True, slots=True)
class MatchedA1BPlan:
    """Hold the three independently resolved children of one matched comparison."""

    a0: Mapping[str, Any]
    b: Mapping[str, Any]
    a1: Mapping[str, Any]

    def stage(self, name: MatchedStageName) -> Mapping[str, Any]:
        """Return one exact child by its public stage token."""
        if name not in {"a0", "b", "a1"}:
            msg = f"Unknown matched training stage {name!r}."
            raise ValueError(msg)
        return getattr(self, name)

    @property
    def stages(self) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        """Return operational order without changing the scientific branch topology."""
        return self.a0, self.b, self.a1


def is_matched_a1_b_config(raw: Mapping[str, Any]) -> bool:
    """Identify the explicitly authored three-stage mode."""
    training = raw.get("training")
    return isinstance(training, Mapping) and training.get("mode") == "matched_a1_b"


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


def _positive_finite(value: object) -> bool:
    """Return whether one authored value is a positive finite real number."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def stage_a_only_experiment(raw: Mapping[str, Any], *, budget_epochs: int) -> dict[str, Any]:
    """
    Return the authored stage-A-only A0 of a matched plan, trained for another epoch budget.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Authored matched plan, its task base merged.
    budget_epochs : int
        Budget of the single A0 stage in passes over the full Train set.

    Returns
    -------
    dict[str, Any]
        Independent experiment with the plan's shared sections, k, A0's memory batch and A0
        sampling, and a ``stage_a_only`` schedule over ``budget_epochs``; B and A1 are dropped.

    Raises
    ------
    loader.ConfigError
        If the plan's shared or A0 schema is invalid or ``budget_epochs`` is not a positive integer.

    Notes
    -----
    This is the A0 an Optuna study tunes: it needs no B or A1 setting, so only the plan's
    shared keys and its A0 are validated here; the plan is validated whole when it trains.

    """
    authored = copy.deepcopy(dict(raw))
    training = _mapping(authored.get("training"), path="training")
    if authored.get("task") != "transient_drying" or training.get("mode") != "matched_a1_b":
        msg = "A stage-A-only experiment derives from a transient_drying matched_a1_b plan."
        raise loader.ConfigError(msg)
    _validate_exact_keys(training, _MATCHED_PLAN_KEYS, path="training")
    _validate_exact_keys(_mapping(authored.get("temporal"), path="temporal"), _TEMPORAL_KEYS, path="temporal")
    a0 = _mapping(training["a0"], path="training.a0")
    _validate_exact_keys(a0, _MATCHED_A0_KEYS, path="training.a0")
    if type(budget_epochs) is not int or budget_epochs < 1:
        msg = "A stage-A-only budget must be a positive integer number of epochs."
        raise loader.ConfigError(msg)
    authored["temporal"]["sampling"] = a0["sampling"]
    authored["training"] = {
        "budget_epochs": budget_epochs,
        "stage_schedule": {"mode": "stage_a_only", "budget_unit": "units", "budget_epochs": budget_epochs},
        "units_per_update": training["units_per_update"],
        "memory_batch": a0["memory_batch"],
    }
    return authored


def resolve_matched_a1_b_plan(
    raw: Mapping[str, Any],
    *,
    source: config_naming.ConfigSource,
    storage_root: Path | str | None = None,
    output_root: Path | str | None = None,
) -> MatchedA1BPlan:
    """
    Validate authored stage budgets and derive ordinary independent run configs.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Authored matched plan with shared semantic sections and exact ``training.a0/b/a1`` maps.
    source : config_naming.ConfigSource
        Task, family and name of the authored plan YAML; every stage shares it.
    storage_root : Path or str or None, optional
        Explicit root for controlled logical Dataset-reference resolution.
    output_root : Path or str or None, optional
        Override the experiments root that stage destinations descend from.

    Returns
    -------
    MatchedA1BPlan
        Fully resolved, independently validated A0, B and A1 configurations.

    Raises
    ------
    loader.ConfigError
        If the authored plan or any derived child violates the runtime contract.

    Notes
    -----
    A1 is A0's teacher-forced protocol continued from A0's selected checkpoint, so its
    sampling is taken from ``a0``; ``units_per_update`` is authored once and shared by all
    three stages, the memory batch once per stage; only A1's learning-rate scale and memory
    batch are authored. Only B has a curriculum: teacher forcing draws no rollout horizon.
    A0 trains for ``budget_epochs`` K of the full Train set, B for K x N supervised
    transitions (N the full Train set's), A1 for B's committed transitions from its receipt;
    every stage validates at the same budget fractions and starts its own schedule.

    """
    authored = copy.deepcopy(dict(raw))
    if authored.get("task") != "transient_drying":
        msg = "matched_a1_b is supported only for transient_drying."
        raise loader.ConfigError(msg)
    training = _mapping(authored.get("training"), path="training")
    protocol_key = "matched_a1_b"
    _validate_exact_keys(training, _MATCHED_PLAN_KEYS, path="training")
    temporal = _mapping(authored.get("temporal"), path="temporal")
    _validate_exact_keys(temporal, _TEMPORAL_KEYS, path="temporal")
    a0 = _mapping(training["a0"], path="training.a0")
    b = _mapping(training["b"], path="training.b")
    a1_authored = _mapping(training["a1"], path="training.a1")
    _validate_exact_keys(a0, _MATCHED_A0_KEYS, path="training.a0")
    _validate_exact_keys(b, _MATCHED_B_KEYS, path="training.b")
    _validate_exact_keys(a1_authored, _MATCHED_A1_KEYS, path="training.a1")
    if type(a0["budget_epochs"]) is not int or a0["budget_epochs"] < 1:
        msg = "training.a0.budget_epochs must be a positive integer."
        raise loader.ConfigError(msg)
    for name, stage in (("b", b), ("a1", a1_authored)):
        if not _positive_finite(stage["lr_scale"]):
            msg = f"training.{name}.lr_scale must be a positive finite continuation learning-rate factor."
            raise loader.ConfigError(msg)
    stages = {"a0": dict(a0), "b": b, "a1": {"sampling": a0["sampling"], "memory_batch": a1_authored["memory_batch"]}}
    lr_scales = {"a0": None, "b": float(b["lr_scale"]), "a1": float(a1_authored["lr_scale"])}
    schedule = {"mode": protocol_key, "budget_unit": "supervised_transitions", "a0_budget_epochs": a0["budget_epochs"]}

    def derive(name: MatchedStageName, *, a0_name: str | None, b_name: str | None) -> dict[str, Any]:
        child = copy.deepcopy(authored)
        child["temporal"]["sampling"] = copy.deepcopy(stages[name]["sampling"])
        stage = stages[name]
        child["training"] = {
            # Only A0 states epochs; B trains K x N supervised transitions with A0's K and A1 B's receipt.
            **({"budget_epochs": stage["budget_epochs"]} if name == "a0" else {}),
            "stage_schedule": copy.deepcopy(schedule),
            "stage": "b" if name == "b" else "a",
            "comparison_arm": name,
            # One k serves every stage, each counting it in its own unit; the memory batch is the stage's own.
            "units_per_update": training["units_per_update"],
            "memory_batch": stage["memory_batch"],
            **({"curriculum": copy.deepcopy(stage["curriculum"])} if name == "b" else {}),
            "teacher_handoff": None if name == "a0" else {"source_run_name": a0_name},
            protocol_key: {
                "role": name,
                "receipt_source_run_name": b_name if name == "a1" else None,
                "initialization": _MATCHED_INITIALIZATION,
                "lr_scale": lr_scales[name],
            },
        }
        resolved = loader.resolve_config(
            child,
            source=source,
            storage_root=storage_root,
            output_root=output_root,
        )
        return loader.validate_resolved_config(resolved)

    resolved_a0 = loader.finalize_run_layout(derive("a0", a0_name=None, b_name=None), stage="a0")
    resolved_b = derive("b", a0_name=resolved_a0["run"]["name"], b_name=None)
    resolved_a1 = derive("a1", a0_name=resolved_a0["run"]["name"], b_name=resolved_b["run"]["name"])
    identity_b = learning.training.checkpoint.effective_config_identity_payload(resolved_b)
    identity_a1 = learning.training.checkpoint.effective_config_identity_payload(resolved_a1)
    identity_a1["training"][protocol_key]["receipt_source_run_name"] = "b"
    branch_id = common.serialization.canonical_json_sha256({"protocol": protocol_key, "b": identity_b, "a1": identity_a1})[:16]
    a0_id = resolved_a0["run"]["layout"]["a0_id"]
    pair = config_naming.pair_key(resolved_b, resolved_a1, source.authored)
    resolved_b = loader.finalize_run_layout(resolved_b, a0_id=a0_id, branch_id=branch_id, stage="b", pair=pair)
    resolved_a1["training"][protocol_key]["receipt_source_run_name"] = resolved_b["run"]["name"]
    resolved_a1 = loader.finalize_run_layout(resolved_a1, a0_id=a0_id, branch_id=branch_id, stage="a1", pair=pair)
    return MatchedA1BPlan(MappingProxyType(resolved_a0), MappingProxyType(resolved_b), MappingProxyType(resolved_a1))
