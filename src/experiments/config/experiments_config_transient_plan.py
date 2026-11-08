"""Resolve authored matched A0/B/A1 transient plans into canonical child run configurations.

Responsibilities:
  - Detect and strictly validate the authored matched A0 -> {B, A1} training schema
  - Derive the A0, B and A1 child configurations through the normal loader
  - Derive A1's teacher-forced protocol from A0 so both arms share one sampling contract
  - Bind B and A1 teacher provenance, the shared branch identity and the pair key to the derived A0

Design principles:
  - The existing resolved experiment schema remains the single runtime contract
  - Authored plan inputs are copied before derivation and never mutated
  - Stage identity is derived into run names instead of duplicated in authored YAML

This module does NOT:
  - Execute training, allocate run directories, or publish handoffs
  - Change Optuna, saved-config, or resume schemas
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
_MATCHED_PLAN_KEYS = frozenset({"mode", "continuation_evaluations", "a0", "b", "a1"})
_MATCHED_A0_KEYS = frozenset(
    {
        "max_epochs",
        "evaluation_interval",
        "additional_evaluation_interval",
        "gradient_accumulation_steps",
        "sampling",
        "curriculum",
    }
)
_MATCHED_B_KEYS = _MATCHED_A0_KEYS | {"train_seconds", "lr_scale"}
_MATCHED_A1_KEYS = frozenset({"max_epochs", "match_to", "lr_scale"})
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


def resolve_matched_a1_b_plan(
    raw: Mapping[str, Any],
    *,
    source: config_naming.ConfigSource,
    storage_root: Path | str | None = None,
    output_root: Path | str | None = None,
    naming_schema_version: int = loader.RUN_NAMING_SCHEMA_VERSION,
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
    naming_schema_version : int, optional
        Sole current saved-run naming schema.

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
    sampling, curriculum, evaluation cadence and accumulation are taken from ``a0``;
    only its operational epoch cap, receipt binding and learning-rate scale are authored.
    B and A1 select checkpoints only at epoch ends that cross one of the
    ``continuation_evaluations`` equally spaced points of their measured budget, so both
    arms get the same number of selection and scheduler opportunities.

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
    for name, stage in (("a0", a0), ("b", b), ("a1", a1_authored)):
        if type(stage["max_epochs"]) is not int or stage["max_epochs"] < 1:
            msg = f"training.{name}.max_epochs must be a positive integer."
            raise loader.ConfigError(msg)
    if a1_authored["match_to"] != "b.train_seconds":
        msg = "A1 must explicitly match B's committed train_seconds receipt."
        raise loader.ConfigError(msg)
    seconds = b["train_seconds"]
    if seconds is None:
        msg = "training.b.train_seconds is required; set it in the experiment config or pass --train-seconds."
        raise loader.ConfigError(msg)
    if not _positive_finite(seconds):
        msg = "B train_seconds must be positive and finite."
        raise loader.ConfigError(msg)
    evaluation_points = training["continuation_evaluations"]
    if type(evaluation_points) is not int or evaluation_points < 1:
        msg = "training.continuation_evaluations must be a positive number of equally spaced B/A1 budget points."
        raise loader.ConfigError(msg)
    for name, stage in (("b", b), ("a1", a1_authored)):
        if not _positive_finite(stage["lr_scale"]):
            msg = f"training.{name}.lr_scale must be a positive finite continuation learning-rate factor."
            raise loader.ConfigError(msg)
    stages = {name: dict(a0) for name in ("a0", "a1")} | {"b": b}
    stages["a1"]["max_epochs"] = a1_authored["max_epochs"]
    lr_scales = {"a0": None, "b": float(b["lr_scale"]), "a1": float(a1_authored["lr_scale"])}
    schedule = {
        "mode": protocol_key,
        "budget_unit": "train_seconds",
        "a0_max_epochs": a0["max_epochs"],
        "b_max_epochs": b["max_epochs"],
        "a1_max_epochs": a1_authored["max_epochs"],
    }

    def derive(name: MatchedStageName, *, a0_name: str | None, b_name: str | None) -> dict[str, Any]:
        child = copy.deepcopy(authored)
        child["temporal"]["sampling"] = copy.deepcopy(stages[name]["sampling"])
        stage = stages[name]
        child["training"] = {
            "epochs": stage["max_epochs"],
            "stage_schedule": copy.deepcopy(schedule),
            "evaluation_interval": stage["evaluation_interval"],
            "additional_evaluation_interval": stage["additional_evaluation_interval"],
            "stage": "b" if name == "b" else "a",
            "comparison_arm": name,
            "gradient_accumulation_steps": stage["gradient_accumulation_steps"],
            "curriculum": copy.deepcopy(stage["curriculum"]),
            "teacher_handoff": None if name == "a0" else {"source_run_name": a0_name},
            protocol_key: {
                "role": name,
                "train_seconds": float(seconds) if name == "b" else None,
                "receipt_source_run_name": b_name if name == "a1" else None,
                "initialization": _MATCHED_INITIALIZATION,
                "lr_scale": lr_scales[name],
                "evaluation_points": None if name == "a0" else evaluation_points,
            },
        }
        resolved = loader.resolve_config(
            child,
            source=source,
            storage_root=storage_root,
            output_root=output_root,
            naming_schema_version=naming_schema_version,
        )
        return loader.validate_resolved_config(resolved)

    resolved_a0 = loader.finalize_run_layout(derive("a0", a0_name=None, b_name=None), stage="a0")
    resolved_b = derive("b", a0_name=resolved_a0["run"]["name"], b_name=None)
    resolved_a1 = derive("a1", a0_name=resolved_a0["run"]["name"], b_name=resolved_b["run"]["name"])
    identity_b = learning.training.checkpoint.effective_config_identity_payload(resolved_b, include_epochs=False)
    identity_a1 = learning.training.checkpoint.effective_config_identity_payload(resolved_a1, include_epochs=False)
    identity_a1["training"][protocol_key]["receipt_source_run_name"] = "b"
    branch_id = common.serialization.canonical_json_sha256({"protocol": protocol_key, "b": identity_b, "a1": identity_a1})[:16]
    a0_id = resolved_a0["run"]["layout"]["a0_id"]
    pair = config_naming.pair_key(resolved_b, resolved_a1)
    resolved_b = loader.finalize_run_layout(resolved_b, a0_id=a0_id, branch_id=branch_id, stage="b", pair=pair)
    resolved_a1["training"][protocol_key]["receipt_source_run_name"] = resolved_b["run"]["name"]
    resolved_a1 = loader.finalize_run_layout(resolved_a1, a0_id=a0_id, branch_id=branch_id, stage="a1", pair=pair)
    if resolved_b["data"]["num_workers"] != 0:
        msg = "matched_a1_b requires data.num_workers=0 for exact partial-epoch continuation."
        raise loader.ConfigError(msg)
    return MatchedA1BPlan(MappingProxyType(resolved_a0), MappingProxyType(resolved_b), MappingProxyType(resolved_a1))
