"""Validate transient training lineage and paired airflow-to-drying evidence.

Responsibilities:
  - Admit immutable A0, B, and A1 lineage evidence for comparison
  - Verify receipt-matched training time and safe-boundary completion, or a stage's divergence with its signed shortfall
  - Admit a diverged A0 with its best eligible checkpoint as the common handoff, reporting its end like the other arms'
  - Admit a B or A1 without an eligible checkpoint by its end status, budget progress and eligible points, without a result
  - Bind every B and A1 arm to the one shared receipt and to the selected A0's handoff
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
# A stage's end: its budget with an eligible checkpoint, a non-finite training loss, or no eligible checkpoint.
_END_STATUSES = frozenset({"completed", "diverged", "no_eligible_checkpoint"})


@dataclass(frozen=True, slots=True)
class TrainingArmEvidence:
    """
    Store immutable lineage and post-handoff compute evidence for one training arm with a result.

    ``safe_boundary_overrun`` is the signed difference actual - planned; it is negative
    only for a diverged arm, which ends before its budget. B's status, budget progress and
    eligible points come from its receipt, the other arms' from their own training outcome.
    """

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
    clock_kind: str = "units"
    budget_complete: bool = True
    training_status: str = "completed"
    budget_progress: float = 1.0
    eligible_points: int = 1
    best_within_budget_checkpoint_id: str = "best_within_budget"
    parent_experiment_sha256: str | None = None
    receipt_sha256: str | None = None
    a0_handoff_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class UnavailableArm:
    """
    Store a B or A1 arm without an eligible checkpoint: its end and the receipt it shares, but no result.

    Such an arm has no artifact, so its evidence is B's receipt (B) or its own training
    outcome (A1); it enters the comparison with an unavailable result instead of a refusal.
    Like an arm with a result, it is bound to the shared receipt and to the selected A0's
    handoff that the receipt names.
    """

    arm: str
    training_status: str
    budget_progress: float
    eligible_points: int
    receipt_sha256: str
    a0_handoff_sha256: str


@dataclass(frozen=True, slots=True)
class SharedRoleMembership:
    """Store exact transient dataset membership evidence for one selected role."""

    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    membership_digests: tuple[str, ...]
    case_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineageComparison:
    """Store admitted lineage arms, each with its end status, and the explicit primary/separate comparisons."""

    a0: TrainingArmEvidence
    a1: TrainingArmEvidence | UnavailableArm
    b: TrainingArmEvidence | UnavailableArm
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
    """Require every comparison run of either task to evaluate identical role membership."""
    if not isinstance(role, str) or not role:
        msg = "Comparison role must be non-empty text."
        raise ValueError(msg)
    if len(memberships) < _MINIMUM_SHARED_MEMBERSHIP_RUNS:
        msg = "Shared-membership validation requires at least two runs."
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
            msg = f"Comparison {role!r} membership evidence is malformed."
            raise ValueError(msg)
        if membership != reference:
            msg = f"Comparison runs disagree on exact {role!r} dataset membership; select runs evaluated on the same cases."
            raise ValueError(msg)
    return reference


def matched_lineage_mismatch(
    *,
    a0: TrainingArmEvidence,
    a1: TrainingArmEvidence | UnavailableArm,
    b: TrainingArmEvidence | UnavailableArm,
) -> str | None:
    """Return why three valid arms are not one matched A0/A1/B set, or ``None`` when they are; unavailable arms bind by receipt."""
    if (a0.arm, a1.arm, b.arm) != ("A0", "A1", "B"):
        return "Matched lineage comparison requires distinct A0, A1, and B arms."
    available = [item for item in (a0, a1, b) if isinstance(item, TrainingArmEvidence)]
    for name in (
        "parent_id",
        "architecture_id",
        "dataset_id",
        "split_id",
        "input_profile_id",
        "scaling_id",
        "boundary_representation_id",
    ):
        values = {getattr(item, name) for item in available}
        if len(values) != 1 or not next(iter(values)):
            return f"Matched A0/B/A1 immutable lineage evidence disagrees for {name}."
    if (
        isinstance(b, TrainingArmEvidence)
        and isinstance(a1, TrainingArmEvidence)
        and (not b.parent_experiment_sha256 or b.parent_experiment_sha256 != a1.parent_experiment_sha256)
    ):
        return "Matched B/A1 must share their exact branch parent identity."
    return None


def _validate_arm_end(item: TrainingArmEvidence | UnavailableArm) -> None:
    """Admit one arm's end: a result exactly with eligible points, never a result after no eligible checkpoint."""
    eligible_points, status = item.eligible_points, item.training_status
    has_result = isinstance(item, TrainingArmEvidence)
    if (
        status not in _END_STATUSES
        or isinstance(eligible_points, bool)
        or not isinstance(eligible_points, int)
        or not 0.0 <= float(item.budget_progress) <= 1.0
        or (eligible_points > 0) != has_result
        or (has_result and status == "no_eligible_checkpoint")
        or (not has_result and status == "completed")
    ):
        msg = f"{item.arm} end is inconsistent: a result needs eligible points, and no eligible checkpoint means no result."
        raise ValueError(msg)


def validate_matched_a1_b_lineage_comparison(
    *,
    a0: TrainingArmEvidence,
    a1: TrainingArmEvidence | UnavailableArm,
    b: TrainingArmEvidence | UnavailableArm,
) -> LineageComparison:
    """
    Admit the real A0/B/A1 branch and B-receipt-matched training time, reporting each arm's end.

    A diverged arm keeps its best eligible checkpoint and a negative actual - planned; a
    diverged A0 stays the common handoff with zero post-handoff compute and reports its status.
    A B or A1 without an eligible checkpoint enters as an :class:`UnavailableArm` bound by the
    shared receipt, so the comparison reports its status instead of refusing the set.
    """
    mismatch = matched_lineage_mismatch(a0=a0, a1=a1, b=b)
    if mismatch is not None:
        raise ValueError(mismatch)
    if a0.parent_experiment_sha256 is not None:
        msg = "Shared A0 identity must remain independent of a B/A1 branch."
        raise ValueError(msg)
    if a0.clock_kind != "units" or (
        _positive(a0.planned_post_handoff_compute, label="A0 planned compute") != 0.0
        or _positive(a0.actual_post_handoff_compute, label="A0 actual compute") != 0.0
        or _positive(a0.safe_boundary_overrun, label="A0 safe-boundary overrun") != 0.0
        or not (a0.budget_complete or a0.training_status == "diverged")
        or not a0.best_within_budget_checkpoint_id
    ):
        msg = "Matched A0 must be the common handoff of a completed or diverged A0 with zero post-handoff compute."
        raise ValueError(msg)
    if a0.receipt_sha256 is not None or not a0.a0_handoff_sha256:
        msg = "Matched A0 must carry its own handoff and no B train budget receipt."
        raise ValueError(msg)
    available = [item for item in (b, a1) if isinstance(item, TrainingArmEvidence)]
    # Every B/A1 arm, with or without a result, names the selected A0's handoff through the one shared receipt.
    if not b.receipt_sha256 or b.receipt_sha256 != a1.receipt_sha256 or any(item.a0_handoff_sha256 != a0.a0_handoff_sha256 for item in (b, a1)):
        msg = "Matched B/A1 must share the exact immutable B receipt and the selected A0's handoff."
        raise ValueError(msg)
    if (
        not a0.strategy
        or any(not item.strategy for item in available)
        or (len(available) == len((b, a1)) and available[0].strategy == available[1].strategy)
    ):
        msg = "Matched B and A1 require distinct rollout and teacher-forced strategies."
        raise ValueError(msg)
    for item in (a0, b, a1):
        _validate_arm_end(item)
    for item in available:
        planned = _positive(item.planned_post_handoff_compute, label=f"{item.arm} planned supervised_transitions")
        actual = _positive(item.actual_post_handoff_compute, label=f"{item.arm} actual supervised_transitions")
        overrun = float(item.safe_boundary_overrun)
        # A diverged arm ends before its budget, so only its signed difference actual - planned may be negative.
        ended = item.budget_complete and actual >= planned
        if (
            item.clock_kind != "supervised_transitions"
            or not math.isfinite(overrun)
            or abs(overrun - (actual - planned)) > _COMPUTE_TOLERANCE
            or not (ended or item.training_status == "diverged")
            or not item.best_within_budget_checkpoint_id
        ):
            msg = f"{item.arm} lacks completed or diverged matched supervised-transition evidence."
            raise ValueError(msg)
    if (
        isinstance(b, TrainingArmEvidence)
        and isinstance(a1, TrainingArmEvidence)
        and b.actual_post_handoff_compute != a1.planned_post_handoff_compute
    ):
        msg = "A1 planned supervised transitions must equal B's receipt-backed committed supervised transitions."
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
