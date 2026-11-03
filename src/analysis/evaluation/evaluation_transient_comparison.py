"""Validate transient training lineage and paired airflow-to-drying evidence.

Responsibilities:
  - Admit immutable A0, B, and A1 lineage evidence for comparison
  - Verify receipt-matched training time and safe-boundary completion
  - Calculate paired airflow-degradation metrics and retain truthful unavailable C evidence

Design principles:
  - The primary method comparison is B versus A1 at matched measured training time
  - Immutable compatibility evidence must agree exactly across arms

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
    parent_experiment_sha256: str | None = None
    receipt_sha256: str | None = None
    a0_handoff_sha256: str | None = None


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
    a1: TrainingArmEvidence
    b: TrainingArmEvidence
    primary_comparison: str
    separate_comparison: str


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


def validate_matched_a1_b_lineage_comparison(
    *,
    a0: TrainingArmEvidence,
    a1: TrainingArmEvidence,
    b: TrainingArmEvidence,
) -> LineageComparison:
    """Admit the real A0/B/A1 branch and B-receipt-matched training time."""
    if (a0.arm, a1.arm, b.arm) != ("A0", "A1", "B"):
        msg = "Matched lineage comparison requires distinct A0, A1, and B arms."
        raise ValueError(msg)
    for name in (
        "parent_id",
        "architecture_id",
        "dataset_id",
        "split_id",
        "input_profile_id",
        "scaling_id",
        "boundary_representation_id",
    ):
        values = {getattr(item, name) for item in (a0, a1, b)}
        if len(values) != 1 or not next(iter(values)):
            msg = f"Matched A0/B/A1 immutable lineage evidence disagrees for {name}."
            raise ValueError(msg)
    if not b.parent_experiment_sha256 or b.parent_experiment_sha256 != a1.parent_experiment_sha256:
        msg = "Matched B/A1 must share their exact branch parent identity."
        raise ValueError(msg)
    if a0.parent_experiment_sha256 is not None:
        msg = "Shared A0 identity must remain independent of a B/A1 branch."
        raise ValueError(msg)
    if a0.clock_kind not in {"cuda_device_seconds", "optimizer_steps"} or (
        _positive(a0.planned_post_handoff_compute, label="A0 planned compute") != 0.0
        or _positive(a0.actual_post_handoff_compute, label="A0 actual compute") != 0.0
        or _positive(a0.safe_boundary_overrun, label="A0 safe-boundary overrun") != 0.0
        or not a0.budget_complete
        or not a0.best_within_budget_checkpoint_id
    ):
        msg = "Matched A0 must be the completed common handoff with zero post-handoff compute."
        raise ValueError(msg)
    if a0.receipt_sha256 is not None or a0.a0_handoff_sha256 is not None:
        msg = "Matched A0 cannot carry a B train budget receipt."
        raise ValueError(msg)
    if not b.receipt_sha256 or b.receipt_sha256 != a1.receipt_sha256 or not b.a0_handoff_sha256 or b.a0_handoff_sha256 != a1.a0_handoff_sha256:
        msg = "Matched B/A1 must share the exact immutable B receipt and A0 handoff."
        raise ValueError(msg)
    if not a0.strategy or not a1.strategy or not b.strategy or a1.strategy == b.strategy:
        msg = "Matched B and A1 require distinct rollout and teacher-forced strategies."
        raise ValueError(msg)
    for item in (b, a1):
        planned = _positive(item.planned_post_handoff_compute, label=f"{item.arm} planned train_seconds")
        actual = _positive(item.actual_post_handoff_compute, label=f"{item.arm} actual train_seconds")
        overrun = _positive(item.safe_boundary_overrun, label=f"{item.arm} train_seconds overrun")
        if (
            item.clock_kind != "train_seconds"
            or planned <= 0.0
            or actual < planned
            or abs(overrun - (actual - planned)) > _COMPUTE_TOLERANCE
            or not item.budget_complete
            or not item.best_within_budget_checkpoint_id
        ):
            msg = f"{item.arm} lacks completed matched train_seconds evidence."
            raise ValueError(msg)
    if b.actual_post_handoff_compute != a1.planned_post_handoff_compute:
        msg = "A1 planned train_seconds must equal B's receipt-backed committed train_seconds."
        raise ValueError(msg)
    return LineageComparison(a0, a1, b, "B_vs_A1", "B_vs_A0")


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
