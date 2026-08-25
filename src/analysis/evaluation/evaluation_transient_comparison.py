"""
evaluation_transient_comparison.py

Validate transient training lineage and paired airflow-to-drying evidence.

Responsibilities:
  - Admit immutable A0, A+, and B lineage evidence for comparison
  - Verify matched post-handoff compute and safe-boundary completion
  - Validate paired B/C drying conditions and truthful unavailable C evidence

Design principles:
  - The primary method comparison is B versus A+ at matched compute
  - Immutable compatibility evidence must agree exactly across arms
  - Airflow substitution changes only the declared airflow fields

This module does NOT:
  - Load checkpoints, run models, or calculate downstream error metrics
  - Treat absent compatible Airflow evidence as zero degradation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

_COMPUTE_TOLERANCE = 1.0e-9
_TRAINING_ARM_COUNT = 3
_MINIMUM_SHARED_MEMBERSHIP_RUNS = 2


@dataclass(frozen=True, slots=True)
class TrainingArmEvidence:
    """Store immutable lineage and post-handoff compute evidence for one training arm."""

    arm: str
    parent_id: str
    architecture_id: str
    dataset_id: str
    split_id: str
    input_profile_id: str
    scaling_id: str
    boundary_representation_id: str
    strategy: str
    planned_post_handoff_compute: float
    actual_post_handoff_compute: float
    safe_boundary_overrun: float
    clock_kind: str = "cuda_device_seconds"
    budget_complete: bool = True
    best_within_budget_checkpoint_id: str = "best_within_budget"


@dataclass(frozen=True, slots=True)
class SharedRoleMembership:
    """Store exact transient dataset membership evidence for one selected role."""

    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    membership_digests: tuple[str, ...]
    case_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineageComparison:
    """Store admitted lineage arms and explicit primary/separate comparisons."""

    a0: TrainingArmEvidence
    a_plus: TrainingArmEvidence
    b: TrainingArmEvidence
    primary_comparison: str
    separate_comparison: str


@dataclass(frozen=True, slots=True)
class AirflowDryingCondition:
    """Store frozen drying inputs for one paired pipeline condition."""

    case_id: str
    drying_checkpoint_id: str
    scaling_id: str
    input_profile_id: str
    initial_state_id: str
    boundary_schedule_id: str
    startup_support_id: str
    material_parameters_id: str
    grid_id: str
    horizon_id: str
    precision: str
    postprocessing_id: str
    airflow_fields: tuple[str, ...]
    training_airflow_source: str = "comsol_reference"


@dataclass(frozen=True, slots=True)
class AirflowDryingComparison:
    """Store paired A/B/C evidence and truthful C availability state."""

    reference_case_id: str
    drying_on_comsol: AirflowDryingCondition
    drying_on_airflow_no: AirflowDryingCondition | None
    c_available: bool
    c_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class AirflowDegradationMetrics:
    """Store B/A, C/A, and C/B downstream errors without causal overclaiming."""

    metric_id: str
    drying_surrogate_error: float
    complete_pipeline_error: float | None
    airflow_substitution_discrepancy: float | None
    signed_airflow_degradation: float | None
    airflow_degradation_ratio: float | None
    upstream_airflow_error: float | None
    c_available: bool
    c_unavailable_reason: str | None


def _positive(value: float, *, label: str) -> float:
    """Validate one finite non-negative compute duration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real duration."
        raise TypeError(msg)
    result = float(value)
    if result < 0.0 or not math.isfinite(result):
        msg = f"{label} must be finite and non-negative."
        raise ValueError(msg)
    return result


def validate_shared_role_membership(
    *,
    role: str,
    memberships: tuple[SharedRoleMembership, ...],
) -> SharedRoleMembership:
    """Require every transient comparison run to evaluate identical role membership."""
    if not isinstance(role, str) or not role:
        msg = "Transient comparison role must be non-empty text."
        raise ValueError(msg)
    if len(memberships) < _MINIMUM_SHARED_MEMBERSHIP_RUNS:
        msg = "Transient shared-membership validation requires at least two runs."
        raise ValueError(msg)
    reference = memberships[0]
    for membership in memberships:
        if (
            not isinstance(membership.dataset_name, str)
            or not membership.dataset_name
            or not membership.source_dataset_ids
            or not all(isinstance(value, str) and value for value in membership.source_dataset_ids)
            or not membership.membership_digests
            or not all(isinstance(value, str) and value for value in membership.membership_digests)
            or not membership.case_ids
            or not all(isinstance(value, str) and value for value in membership.case_ids)
            or membership.case_ids != tuple(sorted(set(membership.case_ids)))
        ):
            msg = f"Transient {role!r} membership evidence is malformed."
            raise ValueError(msg)
        if membership != reference:
            msg = f"Transient comparison runs disagree on exact {role!r} dataset membership."
            raise ValueError(msg)
    return reference


def validate_lineage_comparison(
    *,
    a0: TrainingArmEvidence,
    a_plus: TrainingArmEvidence,
    b: TrainingArmEvidence,
) -> LineageComparison:
    """Validate A0/A+/B compatibility and matched post-handoff compute evidence."""
    arms = {item.arm: item for item in (a0, a_plus, b)}
    if set(arms) != {"A0", "A+", "B"} or len(arms) != _TRAINING_ARM_COUNT:
        msg = "Lineage comparison requires exactly A0, A+, and B arms."
        raise ValueError(msg)
    immutable_names = ("parent_id", "architecture_id", "dataset_id", "split_id", "input_profile_id", "scaling_id", "boundary_representation_id")
    for name in immutable_names:
        values = {getattr(item, name) for item in (a0, a_plus, b)}
        if len(values) != 1 or not next(iter(values)):
            msg = f"A0/A+/B immutable lineage evidence disagrees for {name}."
            raise ValueError(msg)
    if not all(item.strategy for item in (a0, a_plus, b)) or a_plus.strategy in (a0.strategy, b.strategy) or a0.strategy == b.strategy:
        msg = "A0, A+, and B require distinct non-empty training strategies."
        raise ValueError(msg)
    if a0.clock_kind not in {"cuda_device_seconds", "optimizer_steps"}:
        msg = "A0 matched-compute clock is unsupported."
        raise ValueError(msg)
    if (
        _positive(a0.planned_post_handoff_compute, label="A0 planned compute") != 0.0
        or _positive(a0.actual_post_handoff_compute, label="A0 actual compute") != 0.0
        or _positive(a0.safe_boundary_overrun, label="A0 safe-boundary overrun") != 0.0
        or not a0.budget_complete
        or not a0.best_within_budget_checkpoint_id
    ):
        msg = "A0 must be the admitted Stage-A handoff with zero post-handoff compute."
        raise ValueError(msg)
    for item in (a_plus, b):
        planned = _positive(item.planned_post_handoff_compute, label=f"{item.arm} planned compute")
        actual = _positive(item.actual_post_handoff_compute, label=f"{item.arm} actual compute")
        overrun = _positive(item.safe_boundary_overrun, label=f"{item.arm} safe-boundary overrun")
        if item.clock_kind not in {"cuda_device_seconds", "optimizer_steps"}:
            msg = f"{item.arm} matched-compute clock is unsupported."
            raise ValueError(msg)
        if not item.budget_complete or not item.best_within_budget_checkpoint_id:
            msg = f"{item.arm} requires budget-complete and best-within-budget evidence."
            raise ValueError(msg)
        if actual < planned or abs(overrun - (actual - planned)) > _COMPUTE_TOLERANCE:
            msg = f"{item.arm} safe-boundary overrun does not match completed compute."
            raise ValueError(msg)
        if item.clock_kind == "optimizer_steps" and (actual != planned or overrun != 0.0):
            msg = f"{item.arm} optimizer-step budget must complete without overrun."
            raise ValueError(msg)
    if a_plus.clock_kind != b.clock_kind or a_plus.planned_post_handoff_compute != b.actual_post_handoff_compute:
        msg = "A+ must use B's completed terminal compute as its matched post-handoff budget on the same clock."
        raise ValueError(msg)
    return LineageComparison(a0, a_plus, b, "B_vs_A_plus", "B_vs_A0")


def validate_airflow_drying_comparison(
    *,
    reference_case_id: str,
    drying_on_comsol: AirflowDryingCondition,
    drying_on_airflow_no: AirflowDryingCondition | None,
    unavailable_reason: str | None = None,
) -> AirflowDryingComparison:
    """Validate a paired B/C airflow substitution or preserve an explicit absent-C reason."""
    if not reference_case_id or drying_on_comsol.case_id != reference_case_id:
        msg = "Reference A and B must have the identical non-empty case identity."
        raise ValueError(msg)
    if drying_on_airflow_no is None:
        if not unavailable_reason:
            msg = "Unavailable C requires one exact missing-evidence reason."
            raise ValueError(msg)
        return AirflowDryingComparison(reference_case_id, drying_on_comsol, None, False, unavailable_reason)
    if unavailable_reason is not None:
        msg = "Available C must not carry an unavailable reason."
        raise ValueError(msg)
    if drying_on_airflow_no.case_id != reference_case_id:
        msg = "A, B, and C require the same case mapping."
        raise ValueError(msg)
    if drying_on_comsol.training_airflow_source != "comsol_reference" or drying_on_airflow_no.training_airflow_source != "comsol_reference":
        msg = "The frozen drying checkpoint must have training_airflow_source='comsol_reference'."
        raise ValueError(msg)
    frozen_names = (
        "drying_checkpoint_id",
        "scaling_id",
        "input_profile_id",
        "initial_state_id",
        "boundary_schedule_id",
        "startup_support_id",
        "material_parameters_id",
        "grid_id",
        "horizon_id",
        "precision",
        "postprocessing_id",
    )
    for name in frozen_names:
        left, right = getattr(drying_on_comsol, name), getattr(drying_on_airflow_no, name)
        if not left or left != right:
            msg = f"B/C must retain identical {name}."
            raise ValueError(msg)
    if tuple(drying_on_comsol.airflow_fields) != ("u", "v", "p") or tuple(drying_on_airflow_no.airflow_fields) != ("u", "v", "p"):
        msg = "B/C airflow substitution must contain exactly u, v, and p in order."
        raise ValueError(msg)
    return AirflowDryingComparison(reference_case_id, drying_on_comsol, drying_on_airflow_no, True, None)


def build_airflow_degradation_metrics(
    *,
    metric_id: str,
    drying_surrogate_error: float,
    complete_pipeline_error: float | None,
    airflow_substitution_discrepancy: float | None,
    upstream_airflow_error: float | None,
    epsilon: float = 1.0e-12,
    unavailable_reason: str | None = None,
) -> AirflowDegradationMetrics:
    """Calculate paired downstream degradation or retain exact unavailable-C evidence."""
    if not metric_id:
        msg = "Pipeline degradation requires one explicit metric identifier."
        raise ValueError(msg)
    baseline = _positive(drying_surrogate_error, label="drying surrogate error")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or not 0.0 < float(epsilon) < float("inf"):
        msg = "Pipeline degradation epsilon must be finite and positive."
        raise ValueError(msg)
    upstream = None if upstream_airflow_error is None else _positive(upstream_airflow_error, label="upstream airflow error")
    if complete_pipeline_error is None or airflow_substitution_discrepancy is None:
        if complete_pipeline_error is not None or airflow_substitution_discrepancy is not None or not unavailable_reason:
            msg = "Unavailable C requires both C/A and C/B to be absent with one exact reason."
            raise ValueError(msg)
        return AirflowDegradationMetrics(
            metric_id,
            baseline,
            None,
            None,
            None,
            None,
            upstream,
            False,
            unavailable_reason,
        )
    if unavailable_reason is not None:
        msg = "Available C metrics must not carry an unavailable reason."
        raise ValueError(msg)
    complete = _positive(complete_pipeline_error, label="complete pipeline error")
    discrepancy = _positive(airflow_substitution_discrepancy, label="airflow substitution discrepancy")
    return AirflowDegradationMetrics(
        metric_id,
        baseline,
        complete,
        discrepancy,
        complete - baseline,
        complete / max(baseline, float(epsilon)),
        upstream,
        True,
        None,
    )
