"""
Persist and admit sequence-aware transient-drying Evaluation artifacts.

This module owns the variable-length sequence payload and its strict identity.
It does not reconstruct models, select Dataset membership, calculate metrics,
or publish an artifact target into a completed run.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

from src import common
from src.analysis.artifacts import analysis_artifact_performance as artifact_performance
from src.analysis.artifacts import contracts
from src.analysis.evaluation import evaluation_transient_metrics as transient_metrics
from src.datasets.contracts import dataset_contracts_transient as transient_contract

TRANSIENT_SEQUENCE_SCHEMA_KIND: Final = "transient_drying_evaluation_sequence"
TRANSIENT_SEQUENCE_SCHEMA_VERSION: Final = 2
TRANSIENT_ARTIFACT_KIND: Final = "transient_sequence"
_STEP_CONTRACT: Final = transient_contract.TRANSIENT_STEP_CONTRACT
STATE_ORDER: Final = tuple(field.name for field in _STEP_CONTRACT.dynamic_state)
STATE_UNITS: Final = tuple(field.unit for field in _STEP_CONTRACT.dynamic_state)
STATIC_ORDER: Final = tuple(field.name for field in _STEP_CONTRACT.static_spatial_conditioning)
BOUNDARY_ORDER: Final = tuple(field.name for field in _STEP_CONTRACT.step_boundary_conditioning)
SCALAR_ORDER: Final = tuple(field.name for field in _STEP_CONTRACT.scalar_conditioning)
FIXED_HORIZONS: Final = (1, 2, 4, 8, 16, 32, 64, 128)
TARGET_CRITERION: Final = "dry_solid_mass_fraction_with_local_X_wb_above_X_target_wb_le_f_wet_dm_max"
_MINIMUM_TRAJECTORY_STATES: Final = 2
_SHA256_HEX_LENGTH: Final = 64

EvaluationMode = Literal["teacher_forced_one_step", "autonomous_full", "rolling_origin"]
DatasetRole = Literal["id", "ood"]
RequestedHorizon = int | Literal["full"]
EVALUATION_MODES: Final[tuple[EvaluationMode, ...]] = (
    "teacher_forced_one_step",
    "autonomous_full",
    "rolling_origin",
)
ROLLING_ORIGIN_POLICY: Final = "early_middle_late_unique"

_IDENTITY_KEYS: Final = {
    "case_id",
    "dataset_identity",
    "dataset_role",
    "material_family",
    "simulation_identity",
    "model_kind",
    "model_parameters",
    "checkpoint_identity",
    "input_profile",
    "coordinate_policy",
    "boundary_representation",
    "scaling_identity",
    "training_airflow_source",
    "inference_airflow_source",
    "stage_identity",
    "training_strategy",
    "curriculum_identity",
    "parent_checkpoint",
    "stage_a_handoff",
    "matched_compute_manifest",
    "dataset_backend",
    "pt_payload_identity",
    "evaluation_config_identity",
    "timing_evidence_identity",
}
_TARGET_KEYS: Final = {
    "criterion",
    "limit",
    "reference_evidence_scope",
    "predicted_evidence_scope",
    "reference_available",
    "predicted_available",
    "reference_unavailable_reason",
    "predicted_unavailable_reason",
    "reference_reached",
    "predicted_reached",
    "reference_time_to_target",
    "predicted_time_to_target",
    "reference_censored",
    "predicted_censored",
    "reference_final_gap",
    "predicted_final_gap",
    "reference_final_time",
    "predicted_final_time",
}
_EXCLUSION_KEYS: Final = {"excluded", "reason"}


class TransientSequenceArtifactError(ValueError):
    """Signal malformed or contradictory transient sequence evidence."""


def _json_copy(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Return a detached JSON-compatible mapping or fail with its semantic path."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        message = f"{label} must contain only finite JSON-compatible evidence."
        raise TypeError(message) from error
    if not isinstance(result, dict):
        message = f"{label} did not round-trip as a mapping."
        raise TypeError(message)
    return result


def evaluation_protocol_identity(evaluation_config: Mapping[str, Any]) -> str:
    """Return the exact transient Evaluation configuration and protocol digest."""
    evaluation = _json_copy(evaluation_config, label="transient evaluation configuration")
    return common.serialization.canonical_json_sha256(
        {
            "evaluation": evaluation,
            "modes": list(EVALUATION_MODES),
            "fixed_horizons": list(FIXED_HORIZONS),
            "rolling_origin_policy": ROLLING_ORIGIN_POLICY,
        }
    )


def _finite_array(
    value: Any,
    *,
    label: str,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    """Return one owned finite array with an exact rank."""
    array = np.asarray(value)
    if array.ndim != ndim:
        message = f"{label} must have rank {ndim}, got shape {array.shape}."
        raise TransientSequenceArtifactError(message)
    if array.dtype.kind not in {"f", "i", "u"}:
        message = f"{label} must be a real numeric array."
        raise TypeError(message)
    converted = np.asarray(array, dtype=dtype)
    if not np.isfinite(converted).all():
        message = f"{label} contains non-finite values."
        raise TransientSequenceArtifactError(message)
    return np.ascontiguousarray(converted)


def _mask_array(value: Any, *, label: str, ndim: int) -> np.ndarray:
    """Return one owned boolean mask without truth-value coercion."""
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != ndim:
        message = f"{label} must be one rank-{ndim} boolean array."
        raise TypeError(message)
    return np.ascontiguousarray(array)


def _nonempty_text(value: Any, *, label: str) -> str:
    """Return stripped non-empty text."""
    if not isinstance(value, str) or not value.strip():
        message = f"{label} must be non-blank text."
        raise TypeError(message)
    return value.strip()


def _nonnegative_int(value: Any, *, label: str) -> int:
    """Return a non-negative integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        message = f"{label} must be a non-negative integer."
        raise TypeError(message)
    return int(value)


def _validate_identity(value: Mapping[str, Any], *, case_id: str, dataset_role: DatasetRole) -> dict[str, Any]:
    """Admit exact checkpoint, Dataset, tensorizer, lineage, and backend identity."""
    identity = _json_copy(value, label="sequence identity")
    if set(identity) != _IDENTITY_KEYS:
        missing = sorted(_IDENTITY_KEYS.difference(identity))
        unexpected = sorted(set(identity).difference(_IDENTITY_KEYS))
        message = f"Sequence identity fields mismatch: missing={missing}, unexpected={unexpected}."
        raise TransientSequenceArtifactError(message)
    if identity["case_id"] != case_id or identity["dataset_role"] != dataset_role:
        message = "Sequence identity contradicts its case or Dataset role."
        raise TransientSequenceArtifactError(message)
    if identity["model_kind"] not in {"fno", "uno", "rno"}:
        message = "Sequence model kind must be exactly 'fno', 'uno', or 'rno'."
        raise TransientSequenceArtifactError(message)
    if identity["training_airflow_source"] != "comsol_reference":
        message = "Transient Evaluation requires Training airflow source 'comsol_reference'."
        raise TransientSequenceArtifactError(message)
    if identity["inference_airflow_source"] not in {"comsol_reference", "external"}:
        message = "Inference airflow source must be 'comsol_reference' or 'external'."
        raise TransientSequenceArtifactError(message)
    if identity["dataset_backend"] not in {"canonical_hdf5", "pt_shards"}:
        message = "Dataset backend must be canonical_hdf5 or pt_shards."
        raise TransientSequenceArtifactError(message)
    required_text = (
        "case_id",
        "material_family",
        "input_profile",
        "coordinate_policy",
        "boundary_representation",
        "training_strategy",
        "dataset_backend",
        "evaluation_config_identity",
        "timing_evidence_identity",
    )
    for key in required_text:
        _nonempty_text(identity[key], label=f"sequence identity.{key}")
    for key in (
        "dataset_identity",
        "simulation_identity",
        "model_parameters",
        "checkpoint_identity",
        "scaling_identity",
        "stage_identity",
        "curriculum_identity",
        "parent_checkpoint",
        "stage_a_handoff",
        "matched_compute_manifest",
    ):
        if not isinstance(identity[key], dict) or not identity[key]:
            message = f"Sequence identity.{key} must be a non-empty mapping."
            raise TransientSequenceArtifactError(message)
    if identity["parent_checkpoint"] != identity["stage_a_handoff"]:
        message = "Sequence parent checkpoint and Stage-A handoff identity disagree."
        raise TransientSequenceArtifactError(message)
    return identity


def _validate_target(value: Mapping[str, Any]) -> dict[str, Any]:
    """Admit distinct canonical-reference and regular-grid prediction evidence."""
    target = _json_copy(value, label="target evidence")
    if set(target) != _TARGET_KEYS:
        message = "Target evidence fields do not match the transient sequence schema."
        raise TransientSequenceArtifactError(message)
    if target["criterion"] != TARGET_CRITERION:
        message = "Target evidence must use the canonical transient stopping criterion."
        raise TransientSequenceArtifactError(message)
    limit = target["limit"]
    if isinstance(limit, bool) or not isinstance(limit, Real) or not math.isfinite(float(limit)) or not 0.0 < float(limit) < 1.0:
        message = "Target limit must be finite and lie strictly inside (0, 1)."
        raise ValueError(message)
    scopes = {
        "reference": {"canonical_completed_case", "unavailable_partial_interval"},
        "predicted": {"regular_sequence_grid"},
    }
    for prefix, allowed_scopes in scopes.items():
        scope = target[f"{prefix}_evidence_scope"]
        if scope not in allowed_scopes:
            message = f"{prefix} target evidence scope is unsupported."
            raise TransientSequenceArtifactError(message)
        available = target[f"{prefix}_available"]
        unavailable_reason = target[f"{prefix}_unavailable_reason"]
        reached = target[f"{prefix}_reached"]
        censored = target[f"{prefix}_censored"]
        target_time = target[f"{prefix}_time_to_target"]
        final_time = target[f"{prefix}_final_time"]
        gap = target[f"{prefix}_final_gap"]
        if not isinstance(available, bool) or not isinstance(reached, bool) or not isinstance(censored, bool):
            message = f"{prefix} target availability, reached, and censoring states must be boolean."
            raise TransientSequenceArtifactError(message)
        if not available:
            _nonempty_text(unavailable_reason, label=f"{prefix} target unavailable reason")
            if reached or censored or target_time is not None or final_time is not None or gap is not None:
                message = f"Unavailable {prefix} target evidence must not fabricate status, time, censoring, or gap."
                raise TransientSequenceArtifactError(message)
            continue
        if unavailable_reason is not None:
            message = f"Available {prefix} target evidence must have a null unavailable reason."
            raise TransientSequenceArtifactError(message)
        if reached is censored:
            message = f"Available {prefix} reached and censored states must be complementary."
            raise TransientSequenceArtifactError(message)
        if isinstance(final_time, bool) or not isinstance(final_time, Real) or not math.isfinite(float(final_time)):
            message = f"Available {prefix} target evidence requires one finite final time."
            raise TransientSequenceArtifactError(message)
        if reached:
            if isinstance(target_time, bool) or not isinstance(target_time, Real) or not math.isfinite(float(target_time)):
                message = f"{prefix} reached target requires one finite time."
                raise TransientSequenceArtifactError(message)
            if float(target_time) > float(final_time):
                message = f"{prefix} target time cannot exceed its evidence endpoint."
                raise TransientSequenceArtifactError(message)
        elif target_time is not None:
            message = f"{prefix} censored target time must be null."
            raise TransientSequenceArtifactError(message)
        if isinstance(gap, bool) or not isinstance(gap, Real) or not math.isfinite(float(gap)):
            message = f"{prefix} final target gap must be finite."
            raise TransientSequenceArtifactError(message)
    return target


def _validate_record_target_scope(
    target: Mapping[str, Any],
    *,
    times: np.ndarray,
    mode: EvaluationMode,
    origin_index: int,
    requested_horizon: RequestedHorizon,
) -> None:
    """Bind target evidence scopes and endpoints to one exact sequence record."""
    predicted_time = target["predicted_time_to_target"]
    if predicted_time is not None and not bool(np.any(times == float(predicted_time))):
        message = "Predicted target time must be one exact physical time in the regular sequence."
        raise TransientSequenceArtifactError(message)
    predicted_final_time = target["predicted_final_time"]
    if target["predicted_available"] and predicted_final_time != float(times[-1]):
        message = "Predicted target endpoint must equal the exact final regular sequence time."
        raise TransientSequenceArtifactError(message)

    full_autonomous = mode == "autonomous_full" and origin_index == 0 and requested_horizon == "full"
    if full_autonomous:
        reference_final_time = target["reference_final_time"]
        if target["reference_evidence_scope"] != "canonical_completed_case" or target["reference_available"] is not True:
            message = "Full-autonomous records require canonical completed-case reference target evidence."
            raise TransientSequenceArtifactError(message)
        if not isinstance(reference_final_time, Real):
            message = "Canonical completed-case reference target endpoint must be numeric."
            raise TransientSequenceArtifactError(message)
        regular_step = float(times[-1] - times[-2])
        if float(reference_final_time) < float(times[-1]) or float(reference_final_time) > float(times[-1]) + regular_step + 1.0e-6:
            message = "Canonical reference target endpoint contradicts the regular/exact-stop interval."
            raise TransientSequenceArtifactError(message)
        if target["reference_reached"]:
            reference_time = target["reference_time_to_target"]
            if not isinstance(reference_time, Real) or float(reference_time) != float(reference_final_time):
                message = "Canonical reference target time must equal its completed-case endpoint."
                raise TransientSequenceArtifactError(message)
        reference_gap = target["reference_final_gap"]
        if not isinstance(reference_gap, Real) or bool(target["reference_reached"]) != (float(reference_gap) <= 0.0):
            message = "Canonical reference target status contradicts its signed final gap."
            raise TransientSequenceArtifactError(message)
        return
    if target["reference_evidence_scope"] != "unavailable_partial_interval" or target["reference_available"] is not False:
        message = "Partial sequence records must keep completed-case reference target evidence unavailable."
        raise TransientSequenceArtifactError(message)


def _validate_exclusion(value: Mapping[str, Any]) -> dict[str, Any]:
    """Admit explicit inclusion or exclusion status."""
    exclusion = _json_copy(value, label="exclusion evidence")
    if set(exclusion) != _EXCLUSION_KEYS or not isinstance(exclusion["excluded"], bool):
        message = "Exclusion evidence requires exactly excluded and reason fields."
        raise TransientSequenceArtifactError(message)
    reason = exclusion["reason"]
    if exclusion["excluded"]:
        _nonempty_text(reason, label="exclusion reason")
    elif reason is not None:
        message = "Included sequence records must have a null exclusion reason."
        raise TransientSequenceArtifactError(message)
    return exclusion


@dataclass(frozen=True, slots=True)
class TransientSequenceRecord:
    """Hold one validated case, origin, mode, and horizon prediction sequence."""

    mode: EvaluationMode
    case_id: str
    dataset_role: DatasetRole
    origin_index: int
    requested_horizon: RequestedHorizon
    available_horizon: int
    trajectory_length: int
    physical_times: np.ndarray
    transition_indices: np.ndarray
    reference_states: np.ndarray
    predicted_states: np.ndarray
    reference_increments: np.ndarray | None
    predicted_increments: np.ndarray | None
    spatial_mask: np.ndarray
    temporal_mask: np.ndarray
    static_conditioning: np.ndarray
    boundary_conditioning: np.ndarray
    scalar_conditioning: np.ndarray
    identity: Mapping[str, Any]
    target: Mapping[str, Any]
    timing: Mapping[str, Any]
    exclusion: Mapping[str, Any]

    @property
    def record_id(self) -> str:
        """Return a stable content-independent semantic record identifier."""
        payload = {
            "schema": TRANSIENT_SEQUENCE_SCHEMA_VERSION,
            "mode": self.mode,
            "case_id": self.case_id,
            "dataset_role": self.dataset_role,
            "origin_index": self.origin_index,
            "requested_horizon": self.requested_horizon,
            "inference_airflow_source": self.identity["inference_airflow_source"],
            "checkpoint_identity": self.identity["checkpoint_identity"],
        }
        return common.serialization.canonical_json_sha256(payload)

    @property
    def elapsed_physical_time(self) -> float:
        """Return exact final minus origin time for this available sequence."""
        return float(self.physical_times[-1] - self.physical_times[0])

    def validated(self) -> TransientSequenceRecord:  # noqa: C901
        """Return an owned, schema-normalized copy after fail-closed validation."""
        if self.mode not in {"teacher_forced_one_step", "autonomous_full", "rolling_origin"}:
            message = f"Unsupported transient Evaluation mode {self.mode!r}."
            raise TransientSequenceArtifactError(message)
        case_id = _nonempty_text(self.case_id, label="case_id")
        if self.dataset_role not in {"id", "ood"}:
            message = "Dataset role must be 'id' or 'ood'."
            raise TransientSequenceArtifactError(message)
        origin_index = _nonnegative_int(self.origin_index, label="origin_index")
        available = _nonnegative_int(self.available_horizon, label="available_horizon")
        trajectory_length = _nonnegative_int(self.trajectory_length, label="trajectory_length")
        if available < 1 or trajectory_length < _MINIMUM_TRAJECTORY_STATES or origin_index + available >= trajectory_length:
            message = "Origin and available horizon must describe a non-empty interval within the complete trajectory."
            raise TransientSequenceArtifactError(message)
        requested = self.requested_horizon
        if requested != "full":
            requested = _nonnegative_int(requested, label="requested_horizon")
            if requested < 1 or requested != available:
                message = "Available records for a fixed horizon must retain the exact requested horizon."
                raise TransientSequenceArtifactError(message)
        elif self.mode == "teacher_forced_one_step":
            message = "Teacher-forced records use the explicit one-step horizon."
            raise TransientSequenceArtifactError(message)

        times = _finite_array(self.physical_times, label="physical_times", dtype=np.dtype(np.float64), ndim=1)
        transitions = np.asarray(self.transition_indices)
        if transitions.ndim != 1 or transitions.dtype.kind not in {"i", "u"}:
            message = "transition_indices must be one integer vector."
            raise TypeError(message)
        transitions = np.ascontiguousarray(transitions, dtype=np.int64)
        if times.shape != (available + 1,) or transitions.shape != (available,):
            message = "Time/state and transition axes do not match available_horizon."
            raise TransientSequenceArtifactError(message)
        if not np.all(np.diff(times) > 0.0):
            message = "Physical times must be strictly increasing."
            raise TransientSequenceArtifactError(message)
        expected_transitions = np.arange(origin_index, origin_index + available, dtype=np.int64)
        if not np.array_equal(transitions, expected_transitions):
            message = "Transition indices must form the exact contiguous chain from rollout origin."
            raise TransientSequenceArtifactError(message)

        reference = _finite_array(self.reference_states, label="reference_states", dtype=np.dtype(np.float32), ndim=4)
        predicted = _finite_array(self.predicted_states, label="predicted_states", dtype=np.dtype(np.float32), ndim=4)
        if reference.shape != predicted.shape or reference.shape[0:2] != (available + 1, len(STATE_ORDER)):
            message = "Reference and predicted states must share [available+1,4,Y,X] shape."
            raise TransientSequenceArtifactError(message)
        height, width = reference.shape[-2:]
        if min(height, width) < 1:
            message = "Sequence state grids must be non-empty."
            raise TransientSequenceArtifactError(message)
        spatial_mask = _mask_array(self.spatial_mask, label="spatial_mask", ndim=2)
        temporal_mask = _mask_array(self.temporal_mask, label="temporal_mask", ndim=1)
        if spatial_mask.shape != (height, width) or temporal_mask.shape != (available + 1,):
            message = "Spatial and temporal masks do not match the sequence axes."
            raise TransientSequenceArtifactError(message)
        if not spatial_mask.any() or not temporal_mask.all():
            message = "Persisted available sequences require a non-empty spatial mask and fully valid temporal axis."
            raise TransientSequenceArtifactError(message)

        static = _finite_array(self.static_conditioning, label="static_conditioning", dtype=np.dtype(np.float32), ndim=3)
        boundary = _finite_array(self.boundary_conditioning, label="boundary_conditioning", dtype=np.dtype(np.float32), ndim=2)
        scalars = _finite_array(self.scalar_conditioning, label="scalar_conditioning", dtype=np.dtype(np.float32), ndim=1)
        if static.shape != (len(STATIC_ORDER), height, width):
            message = "Static conditioning must have [7,Y,X] shape."
            raise TransientSequenceArtifactError(message)
        if boundary.shape != (available, len(BOUNDARY_ORDER)):
            message = "Boundary conditioning must retain both endpoints and startup evidence as [L,9]."
            raise TransientSequenceArtifactError(message)
        if scalars.shape != (len(SCALAR_ORDER),):
            message = "Scalar conditioning must contain the exact eight material parameters."
            raise TransientSequenceArtifactError(message)

        def increments(value: np.ndarray | None, *, label: str) -> np.ndarray | None:
            if value is None:
                return None
            result = _finite_array(value, label=label, dtype=np.dtype(np.float32), ndim=4)
            if result.shape != (available, len(STATE_ORDER), height, width):
                message = f"{label} must have [available,4,Y,X] shape."
                raise TransientSequenceArtifactError(message)
            return result

        reference_increments = increments(self.reference_increments, label="reference_increments")
        predicted_increments = increments(self.predicted_increments, label="predicted_increments")
        if reference_increments is not None and not np.allclose(reference_increments, np.diff(reference, axis=0), rtol=0.0, atol=2.0e-5):
            message = "Reference increments contradict reference absolute states."
            raise TransientSequenceArtifactError(message)
        if predicted_increments is not None and not np.allclose(predicted_increments, np.diff(predicted, axis=0), rtol=0.0, atol=2.0e-5):
            message = "Predicted increments contradict predicted absolute states."
            raise TransientSequenceArtifactError(message)

        identity = _validate_identity(self.identity, case_id=case_id, dataset_role=self.dataset_role)
        target = _validate_target(self.target)
        _validate_record_target_scope(
            target,
            times=times,
            mode=self.mode,
            origin_index=origin_index,
            requested_horizon=cast("RequestedHorizon", requested),
        )
        timing = _json_copy(self.timing, label="timing evidence")
        exclusion = _validate_exclusion(self.exclusion)
        if exclusion["excluded"]:
            message = "Excluded horizons are availability records, not persisted sequence payloads."
            raise TransientSequenceArtifactError(message)
        return replace(
            self,
            case_id=case_id,
            origin_index=origin_index,
            requested_horizon=cast("RequestedHorizon", requested),
            available_horizon=available,
            trajectory_length=trajectory_length,
            physical_times=times,
            transition_indices=transitions,
            reference_states=reference,
            predicted_states=predicted,
            reference_increments=reference_increments,
            predicted_increments=predicted_increments,
            spatial_mask=spatial_mask,
            temporal_mask=temporal_mask,
            static_conditioning=static,
            boundary_conditioning=boundary,
            scalar_conditioning=scalars,
            identity=identity,
            target=target,
            timing=timing,
            exclusion=exclusion,
        )


@dataclass(frozen=True, slots=True)
class UnavailableHorizon:
    """Describe one requested fixed horizon that a case/origin cannot support."""

    case_id: str
    dataset_role: DatasetRole
    origin_index: int
    requested_horizon: int
    available_transitions: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return strict JSON-compatible availability evidence."""
        case_id = _nonempty_text(self.case_id, label="unavailable case_id")
        origin = _nonnegative_int(self.origin_index, label="unavailable origin_index")
        requested = _nonnegative_int(self.requested_horizon, label="unavailable requested_horizon")
        available = _nonnegative_int(self.available_transitions, label="unavailable available_transitions")
        reason = _nonempty_text(self.reason, label="unavailable reason")
        if self.dataset_role not in {"id", "ood"} or requested < 1 or available >= requested:
            message = "Unavailable horizon evidence must identify a genuinely unsupported fixed horizon."
            raise TransientSequenceArtifactError(message)
        return {
            "case_id": case_id,
            "dataset_role": self.dataset_role,
            "origin_index": origin,
            "requested_horizon": requested,
            "available_transitions": available,
            "reason": reason,
        }


@dataclass(frozen=True, slots=True)
class LoadedTransientSequenceArtifact:
    """Return fully admitted sequence payloads and their compact summary table."""

    root: Path
    dataset_name: str
    dataset_role: DatasetRole
    records: tuple[TransientSequenceRecord, ...]
    unavailable_horizons: tuple[dict[str, Any], ...]
    frame: pd.DataFrame
    provenance: Mapping[str, Any]
    identity_sha256: str


def _scientific_identity_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return artifact identity evidence without operational performance telemetry."""
    result = {key: value for key, value in document.items() if key not in {"outputs", "identity_sha256"}}
    runtime = result.get("runtime")
    if isinstance(runtime, Mapping) and "operational_performance" in runtime:
        result["runtime"] = {key: value for key, value in runtime.items() if key != "operational_performance"}
    return result


def _chain_id_from_coordinates(*, mode: str, origin_index: int) -> str:
    """Return one case-local prediction-chain identity from exact coordinates."""
    return common.serialization.canonical_json_sha256({"mode": mode, "origin_index": origin_index})[:24]


def _chain_id(record: TransientSequenceRecord) -> str:
    """Return one case-local prediction-chain identity independent of fixed prefixes."""
    return _chain_id_from_coordinates(
        mode=record.mode,
        origin_index=record.origin_index,
    )


def _record_row(
    record: TransientSequenceRecord,
    *,
    payload_path: str,
    payload_sha256: str,
    metric_statistics: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, Any]:
    """Build one bounded Parquet row that refers to a case-local numerical bundle."""
    target = record.target
    return {
        "schema_kind": TRANSIENT_SEQUENCE_SCHEMA_KIND,
        "schema_version": TRANSIENT_SEQUENCE_SCHEMA_VERSION,
        "record_id": record.record_id,
        "case_id": record.case_id,
        "dataset_role": record.dataset_role,
        "mode": record.mode,
        "origin_index": record.origin_index,
        "origin_time": float(record.physical_times[0]),
        "requested_horizon": str(record.requested_horizon),
        "available_horizon": record.available_horizon,
        "trajectory_length": record.trajectory_length,
        "elapsed_physical_time": record.elapsed_physical_time,
        "inference_airflow_source": record.identity["inference_airflow_source"],
        "model_kind": record.identity["model_kind"],
        "dataset_backend": record.identity["dataset_backend"],
        "reference_reached": target["reference_reached"],
        "predicted_reached": target["predicted_reached"],
        "reference_censored": target["reference_censored"],
        "predicted_censored": target["predicted_censored"],
        "reference_time_to_target": target["reference_time_to_target"],
        "predicted_time_to_target": target["predicted_time_to_target"],
        "payload_path": payload_path,
        "payload_sha256": payload_sha256,
        "chain_id": _chain_id(record),
        "metric_statistics_json": json.dumps(metric_statistics, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if metric_statistics is not None
        else None,
        "identity_json": json.dumps(record.identity, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True),
        "target_json": json.dumps(record.target, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True),
        "timing_json": json.dumps(record.timing, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True),
        "exclusion_json": json.dumps(record.exclusion, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True),
    }


def _case_payload_path(case_id: str) -> Path:
    """Return a safe deterministic bundle path without using case text as a filename."""
    digest = common.serialization.canonical_json_sha256({"case_id": case_id})
    return Path("npz") / f"case-{digest}.npz"


def _validate_repository_evidence(
    value: Any,
    *,
    label: str,
    required: bool,
) -> dict[str, Any] | None:
    """Admit bounded Git commit and dirty-state evidence."""
    if value is None and not required:
        return None
    if not isinstance(value, dict) or set(value) != {"commit", "dirty"}:
        message = f"{label} must contain exactly commit and dirty evidence."
        raise TransientSequenceArtifactError(message)
    commit = value["commit"]
    dirty = value["dirty"]
    if commit is not None and (
        not isinstance(commit, str) or len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit)
    ):
        message = f"{label} commit must be one Git object identity or null."
        raise TransientSequenceArtifactError(message)
    if dirty is not None and not isinstance(dirty, bool):
        message = f"{label} dirty state must be boolean or null."
        raise TransientSequenceArtifactError(message)
    return {"commit": commit, "dirty": dirty}


def _validate_parent_experiment_evidence(value: Any) -> dict[str, Any]:
    """Admit exact grouped-parent evidence or explicit legacy parent absence."""
    evidence = _json_copy(value, label="run parent_experiment")
    kind = evidence.get("kind")
    if kind == "grouped":
        required = {
            "kind",
            "parent_available",
            "parent_label",
            "parent_identity_sha256",
            "run_revision",
            "source_repository",
            "child_source_repository",
        }
        if set(evidence) != required or evidence["parent_available"] is not True:
            message = "Grouped parent experiment evidence is incomplete."
            raise TransientSequenceArtifactError(message)
        common.paths.validate_logical_name(
            evidence["parent_label"],
            label="parent experiment label",
        )
        digest = evidence["parent_identity_sha256"]
        if not isinstance(digest, str) or len(digest) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in digest):
            message = "Grouped parent experiment identity digest is invalid."
            raise TransientSequenceArtifactError(message)
        _nonnegative_int(
            evidence["run_revision"],
            label="parent experiment run_revision",
        )
        evidence["source_repository"] = _validate_repository_evidence(
            evidence["source_repository"],
            label="parent source_repository",
            required=True,
        )
    elif kind == "legacy":
        required = {
            "kind",
            "parent_available",
            "reason",
            "child_source_repository",
        }
        if set(evidence) != required or evidence["parent_available"] is not False:
            message = "Legacy parent experiment absence evidence is incomplete."
            raise TransientSequenceArtifactError(message)
        _nonempty_text(
            evidence["reason"],
            label="legacy parent absence reason",
        )
    else:
        message = "Run parent experiment evidence must be grouped or legacy."
        raise TransientSequenceArtifactError(message)
    evidence["child_source_repository"] = _validate_repository_evidence(
        evidence["child_source_repository"],
        label="child source_repository",
        required=False,
    )
    return evidence


def _write_case_bundle(path: Path, records: Sequence[TransientSequenceRecord]) -> None:
    """Write one deduplicated complete-case bundle and unique prediction chains."""
    if not records:
        message = "Transient case bundle requires at least one record."
        raise ValueError(message)
    reference_increment_flags = {record.reference_increments is not None for record in records}
    if len(reference_increment_flags) != 1:
        message = "Case records disagree on reference-increment availability."
        raise TransientSequenceArtifactError(message)
    exemplar = records[0]
    trajectory_length = exemplar.trajectory_length
    shared_times = np.empty(trajectory_length, dtype=np.float64)
    shared_reference = np.empty((trajectory_length, *exemplar.reference_states.shape[1:]), dtype=np.float32)
    shared_temporal = np.empty(trajectory_length, dtype=bool)
    shared_boundary = np.empty((trajectory_length - 1, *exemplar.boundary_conditioning.shape[1:]), dtype=np.float32)
    covered_states = np.zeros(trajectory_length, dtype=bool)
    covered_transitions = np.zeros(trajectory_length - 1, dtype=bool)
    payload: dict[str, np.ndarray] = {
        "spatial_mask": exemplar.spatial_mask,
        "static_conditioning": exemplar.static_conditioning,
        "scalar_conditioning": exemplar.scalar_conditioning,
        "state_order": np.asarray(STATE_ORDER, dtype="U16"),
        "state_units": np.asarray(STATE_UNITS, dtype="U16"),
        "static_order": np.asarray(STATIC_ORDER, dtype="U24"),
        "boundary_order": np.asarray(BOUNDARY_ORDER, dtype="U40"),
        "scalar_order": np.asarray(SCALAR_ORDER, dtype="U24"),
    }
    chains: dict[str, TransientSequenceRecord] = {}
    chain_increment_flags: dict[str, bool] = {}
    for record in records:
        if (
            record.case_id != exemplar.case_id
            or record.trajectory_length != trajectory_length
            or not np.array_equal(record.spatial_mask, exemplar.spatial_mask)
            or not np.array_equal(record.static_conditioning, exemplar.static_conditioning)
            or not np.array_equal(record.scalar_conditioning, exemplar.scalar_conditioning)
        ):
            message = "Case-local shared transient evidence is contradictory."
            raise TransientSequenceArtifactError(message)
        start = record.origin_index
        stop = start + record.available_horizon + 1
        transition_stop = stop - 1
        if covered_states[start:stop].any() and (
            not np.array_equal(shared_times[start:stop][covered_states[start:stop]], record.physical_times[covered_states[start:stop]])
            or not np.array_equal(shared_reference[start:stop][covered_states[start:stop]], record.reference_states[covered_states[start:stop]])
            or not np.array_equal(shared_temporal[start:stop][covered_states[start:stop]], record.temporal_mask[covered_states[start:stop]])
        ):
            message = "Case records contradict shared reference/time/mask evidence."
            raise TransientSequenceArtifactError(message)
        if covered_transitions[start:transition_stop].any() and not np.array_equal(
            shared_boundary[start:transition_stop][covered_transitions[start:transition_stop]],
            record.boundary_conditioning[covered_transitions[start:transition_stop]],
        ):
            message = "Case records contradict shared boundary evidence."
            raise TransientSequenceArtifactError(message)
        shared_times[start:stop] = record.physical_times
        shared_reference[start:stop] = record.reference_states
        shared_temporal[start:stop] = record.temporal_mask
        shared_boundary[start:transition_stop] = record.boundary_conditioning
        covered_states[start:stop] = True
        covered_transitions[start:transition_stop] = True
        chain = _chain_id(record)
        increment_available = record.predicted_increments is not None
        prior_availability = chain_increment_flags.setdefault(
            chain,
            increment_available,
        )
        if prior_availability != increment_available:
            message = "One transient prediction chain disagrees on increment availability."
            raise TransientSequenceArtifactError(message)
        current = chains.get(chain)
        if current is None or record.available_horizon > current.available_horizon:
            chains[chain] = record
        elif record.available_horizon == current.available_horizon and (
            not np.array_equal(record.predicted_states, current.predicted_states)
            or (
                record.predicted_increments is not None
                and current.predicted_increments is not None
                and not np.array_equal(
                    record.predicted_increments,
                    current.predicted_increments,
                )
            )
        ):
            message = "One transient prediction chain has contradictory duplicate evidence."
            raise TransientSequenceArtifactError(message)
    if not covered_states.all() or not covered_transitions.all():
        message = "Case bundle lacks complete shared trajectory evidence."
        raise TransientSequenceArtifactError(message)
    payload.update(
        {
            "physical_times": shared_times,
            "reference_states": shared_reference,
            "temporal_mask": shared_temporal,
            "boundary_conditioning": shared_boundary,
        }
    )
    if reference_increment_flags == {True}:
        payload["reference_increments"] = np.diff(shared_reference, axis=0)
    for chain, record in chains.items():
        prefix = f"chain_{chain}_"
        payload[f"{prefix}predicted_states"] = record.predicted_states
        if record.predicted_increments is not None:
            payload[f"{prefix}predicted_increments"] = record.predicted_increments
    cast("Any", np.savez)(path, **payload)


class TransientSequenceArtifactStager:
    """Stream exact case bundles into one private incomplete artifact root."""

    def __init__(
        self,
        root: Path | str,
        *,
        dataset_name: str,
        dataset_role: DatasetRole,
    ) -> None:
        """Create one empty staging root without publishing a completion marker."""
        artifact_root = Path(root)
        if artifact_root.exists() and any(artifact_root.iterdir()):
            message = f"Transient artifact staging root must be empty: {artifact_root}"
            raise FileExistsError(message)
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "npz").mkdir()
        self.root = artifact_root.resolve()
        self.dataset_name = common.paths.validate_logical_name(
            dataset_name,
            label="dataset_name",
        )
        if dataset_role not in {"id", "ood"}:
            message = "dataset_role must be 'id' or 'ood'."
            raise ValueError(message)
        self.dataset_role = dataset_role
        self._rows: list[dict[str, Any]] = []
        self._unavailable: list[dict[str, Any]] = []
        self._case_payloads: dict[str, tuple[str, str]] = {}
        self._record_ids: set[str] = set()
        self._finalized = False

    def write_case(
        self,
        records: Sequence[TransientSequenceRecord],
        *,
        unavailable_horizons: Sequence[UnavailableHorizon] = (),
        metric_statistics: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    ) -> None:
        """Validate, persist, and release one exact case-local record group."""
        if self._finalized:
            message = "Transient artifact staging is already finalized."
            raise RuntimeError(message)
        admitted = tuple(record.validated() for record in records)
        case_ids = {record.case_id for record in admitted}
        record_ids = {record.record_id for record in admitted}
        if not admitted or len(case_ids) != 1 or len(record_ids) != len(admitted):
            message = "Transient case staging requires one non-empty case with unique available records."
            raise ValueError(message)
        case_id = next(iter(case_ids))
        if case_id in self._case_payloads or self._record_ids.intersection(record_ids):
            message = f"Transient case {case_id!r} was already staged."
            raise ValueError(message)
        if any(record.dataset_role != self.dataset_role for record in admitted):
            message = "Transient sequence records do not share the artifact Dataset role."
            raise ValueError(message)
        if metric_statistics is not None and set(metric_statistics) != record_ids:
            message = "Transient metric statistics must cover every case record exactly when supplied."
            raise ValueError(message)
        unavailable = tuple(item.as_dict() for item in unavailable_horizons)
        if any(item["case_id"] != case_id or item["dataset_role"] != self.dataset_role for item in unavailable):
            message = "Transient unavailable horizons must belong to the staged case and role."
            raise ValueError(message)
        relative = _case_payload_path(case_id)
        payload_path = self.root / relative
        _write_case_bundle(payload_path, admitted)
        payload_digest = common.serialization.file_sha256(payload_path)
        payload = (relative.as_posix(), payload_digest)
        rows = [
            _record_row(
                record,
                payload_path=payload[0],
                payload_sha256=payload[1],
                metric_statistics=(None if metric_statistics is None else metric_statistics.get(record.record_id)),
            )
            for record in admitted
        ]
        self._case_payloads[case_id] = payload
        self._record_ids.update(record_ids)
        self._rows.extend(rows)
        self._unavailable.extend(unavailable)

    def finalize(
        self,
        *,
        provenance: Mapping[str, Any],
    ) -> TransientSequenceArtifactIndex:
        """Publish compact summary/provenance last and return an index without payload rereads."""
        if self._finalized:
            message = "Transient artifact staging is already finalized."
            raise RuntimeError(message)
        if not self._rows:
            message = "Transient sequence artifacts require at least one staged case."
            raise ValueError(message)
        frame = pd.DataFrame.from_records(self._rows)
        parquet_path = self.root / f"{self.dataset_name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        admitted_provenance = _json_copy(
            provenance,
            label="transient artifact provenance",
        )
        required_provenance = {"run", "dataset", "evaluation", "runtime", "lineage"}
        if set(admitted_provenance) != required_provenance:
            message = "Transient artifact provenance must contain exactly run, dataset, evaluation, runtime, and lineage."
            raise ValueError(message)
        run = admitted_provenance["run"]
        if not isinstance(run, dict) or "parent_experiment" not in run:
            message = "Transient artifact run provenance requires parent_experiment evidence."
            raise ValueError(message)
        run["parent_experiment"] = _validate_parent_experiment_evidence(
            run["parent_experiment"],
        )
        outputs = {
            "parquet": {
                "path": parquet_path.relative_to(self.root).as_posix(),
                "sha256": common.serialization.file_sha256(parquet_path),
            },
            "npz": [{"path": relative, "sha256": digest} for relative, digest in sorted(self._case_payloads.values())],
        }
        document = {
            "provenance_schema_version": contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "artifact_schema_version": TRANSIENT_SEQUENCE_SCHEMA_VERSION,
            "artifact_kind": TRANSIENT_ARTIFACT_KIND,
            "task": "transient_drying",
            **admitted_provenance,
            "availability": {
                "available_record_count": len(self._rows),
                "unavailable_horizon_count": len(self._unavailable),
                "unavailable_horizons": list(self._unavailable),
            },
            "outputs": outputs,
        }
        document["identity_sha256"] = common.serialization.canonical_json_sha256(
            _scientific_identity_document(document),
        )
        common.serialization.atomic_write_json(
            contracts.artifact_provenance_path(self.root),
            document,
        )
        index = _index_from_admitted_frame(
            self.root,
            provenance=document,
            frame=frame,
            cache_limit=1,
        )
        validate_transient_sequence_payload_inventory(index)
        self._finalized = True
        return index


def write_transient_sequence_artifact(
    root: Path | str,
    *,
    dataset_name: str,
    dataset_role: DatasetRole,
    records: Sequence[TransientSequenceRecord],
    unavailable_horizons: Sequence[UnavailableHorizon] = (),
    provenance: Mapping[str, Any],
    metric_statistics: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> TransientSequenceArtifactIndex:
    """Write one artifact through the bounded case-staging owner."""
    admitted = tuple(records)
    grouped: dict[str, list[TransientSequenceRecord]] = {}
    for record in admitted:
        grouped.setdefault(record.case_id, []).append(record)
    stager = TransientSequenceArtifactStager(
        root,
        dataset_name=dataset_name,
        dataset_role=dataset_role,
    )
    unavailable_by_case: dict[str, list[UnavailableHorizon]] = {}
    for item in unavailable_horizons:
        unavailable_by_case.setdefault(item.case_id, []).append(item)
    for case_id, case_records in grouped.items():
        case_record_ids = {record.record_id for record in case_records}
        case_statistics = (
            None
            if metric_statistics is None
            else {record_id: metric_statistics[record_id] for record_id in case_record_ids if record_id in metric_statistics}
        )
        stager.write_case(
            case_records,
            unavailable_horizons=unavailable_by_case.pop(case_id, ()),
            metric_statistics=case_statistics,
        )
    if unavailable_by_case:
        message = "Transient unavailable horizons contain cases without staged records."
        raise ValueError(message)
    return stager.finalize(provenance=provenance)


def publish_transient_operational_performance(
    root: Path | str,
    value: Mapping[str, Any],
) -> None:
    """Publish validated non-scientific telemetry without changing artifact identity."""
    artifact_root = Path(root).expanduser().resolve(strict=True)
    marker = contracts.artifact_provenance_path(artifact_root)
    if not marker.is_file() or marker.is_symlink():
        message = f"Transient artifact completion marker is missing or unsafe: {marker}"
        raise FileNotFoundError(message)
    admitted = artifact_performance.validate_operational_performance(value)
    with marker.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        message = "Transient artifact provenance must be a mapping."
        raise TypeError(message)
    identity = document.get("identity_sha256")
    if identity != common.serialization.canonical_json_sha256(_scientific_identity_document(document)):
        message = "Transient artifact identity digest is invalid before telemetry publication."
        raise TransientSequenceArtifactError(message)
    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        message = "Transient artifact runtime provenance must be a mapping."
        raise TypeError(message)
    runtime["operational_performance"] = admitted
    if identity != common.serialization.canonical_json_sha256(_scientific_identity_document(document)):
        message = "Operational telemetry changed scientific artifact identity."
        raise RuntimeError(message)
    common.serialization.atomic_write_json(marker, document)


def _require_relative_payload(root: Path, relative: Any) -> Path:
    """Resolve one safe regular NPZ file below the artifact root."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        message = "Transient artifact contains an unsafe payload path."
        raise TransientSequenceArtifactError(message)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or path.suffix != ".npz":
        message = "Transient payload must be one regular NPZ below the artifact root."
        raise TransientSequenceArtifactError(message)
    return path


class _TransientCasePayload(Mapping[str, np.ndarray]):
    """Own one open case bundle and materialize requested arrays on demand."""

    def __init__(self, path: Path, *, expected_sha256: str) -> None:
        if common.serialization.file_sha256(path) != expected_sha256:
            message = f"Transient case payload digest changed: {path}"
            raise TransientSequenceArtifactError(message)
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            message = f"Transient case payload cannot be opened: {path}"
            raise TransientSequenceArtifactError(message) from error
        if not hasattr(archive, "files") or not hasattr(archive, "close"):
            close = getattr(archive, "close", None)
            if callable(close):
                close()
            message = f"Transient case payload is not one NPZ archive: {path}"
            raise TransientSequenceArtifactError(message)
        self._archive: Any = archive
        self._fields = frozenset(str(name) for name in archive.files)
        self._arrays: dict[str, np.ndarray] = {}
        self._closed = False

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one cached array, decompressing only its selected NPZ member."""
        if key not in self._fields:
            raise KeyError(key)
        if self._closed:
            message = "Transient case payload was accessed after cache eviction."
            raise RuntimeError(message)
        cached = self._arrays.get(key)
        if cached is not None:
            return cached
        try:
            value = np.asarray(self._archive[key])
        except (OSError, ValueError, KeyError) as error:
            message = f"Transient case payload member cannot be read: {key!r}"
            raise TransientSequenceArtifactError(message) from error
        self._arrays[key] = value
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate payload field names without decompressing numerical arrays."""
        return iter(self._fields)

    def __len__(self) -> int:
        """Return the persisted field count without decompressing arrays."""
        return len(self._fields)

    def __contains__(self, key: object) -> bool:
        """Test field availability without materializing its array."""
        return isinstance(key, str) and key in self._fields

    def close(self) -> None:
        """Close the archive and release the bounded decompressed-array cache."""
        if self._closed:
            return
        self._archive.close()
        self._arrays.clear()
        self._closed = True


def _load_case_payload(root: Path, row: Mapping[str, Any]) -> _TransientCasePayload:
    """Open one digest-validated case bundle without eagerly decompressing its fields."""
    path = _require_relative_payload(root, row["payload_path"])
    return _TransientCasePayload(path, expected_sha256=str(row["payload_sha256"]))


def _record_from_row(root: Path, row: Mapping[str, Any], payload: Mapping[str, np.ndarray] | None = None) -> TransientSequenceRecord:
    """Reconstruct one exact row record from its selected case bundle prefix."""
    if payload is None:
        owned_payload = _load_case_payload(root, row)
        try:
            return _record_from_row(root, row, owned_payload)
        finally:
            owned_payload.close()
    required_common = {
        "physical_times",
        "reference_states",
        "temporal_mask",
        "boundary_conditioning",
        "spatial_mask",
        "static_conditioning",
        "scalar_conditioning",
        "state_order",
        "state_units",
        "static_order",
        "boundary_order",
        "scalar_order",
    }
    if not required_common.issubset(payload):
        message = "Transient case bundle lacks shared evidence."
        raise TransientSequenceArtifactError(message)
    expected_orders = {
        "state_order": STATE_ORDER,
        "state_units": STATE_UNITS,
        "static_order": STATIC_ORDER,
        "boundary_order": BOUNDARY_ORDER,
        "scalar_order": SCALAR_ORDER,
    }
    for key, expected in expected_orders.items():
        if tuple(str(item) for item in payload[key].tolist()) != expected:
            message = f"Transient case bundle {key} contradicts the canonical contract."
            raise TransientSequenceArtifactError(message)
    chain = _nonempty_text(row["chain_id"], label="summary chain_id")
    prefix = f"chain_{chain}_"
    needed = {f"{prefix}predicted_states"}
    if not needed.issubset(payload):
        message = "Transient case bundle lacks the selected prediction chain."
        raise TransientSequenceArtifactError(message)
    available = int(row["available_horizon"])
    state_count = available + 1
    try:
        requested: RequestedHorizon = "full" if row["requested_horizon"] == "full" else int(row["requested_horizon"])
        record = TransientSequenceRecord(
            mode=cast("EvaluationMode", row["mode"]),
            case_id=str(row["case_id"]),
            dataset_role=cast("DatasetRole", row["dataset_role"]),
            origin_index=int(row["origin_index"]),
            requested_horizon=requested,
            available_horizon=available,
            trajectory_length=int(row["trajectory_length"]),
            physical_times=payload["physical_times"][int(row["origin_index"]) : int(row["origin_index"]) + state_count],
            transition_indices=np.arange(int(row["origin_index"]), int(row["origin_index"]) + available, dtype=np.int64),
            reference_states=payload["reference_states"][int(row["origin_index"]) : int(row["origin_index"]) + state_count],
            predicted_states=payload[f"{prefix}predicted_states"][:state_count],
            reference_increments=(
                None
                if "reference_increments" not in payload
                else payload["reference_increments"][int(row["origin_index"]) : int(row["origin_index"]) + available]
            ),
            predicted_increments=(None if f"{prefix}predicted_increments" not in payload else payload[f"{prefix}predicted_increments"][:available]),
            spatial_mask=payload["spatial_mask"],
            temporal_mask=payload["temporal_mask"][int(row["origin_index"]) : int(row["origin_index"]) + state_count],
            static_conditioning=payload["static_conditioning"],
            boundary_conditioning=payload["boundary_conditioning"][int(row["origin_index"]) : int(row["origin_index"]) + available],
            scalar_conditioning=payload["scalar_conditioning"],
            identity=json.loads(str(row["identity_json"])),
            target=json.loads(str(row["target_json"])),
            timing=json.loads(str(row["timing_json"])),
            exclusion=json.loads(str(row["exclusion_json"])),
        ).validated()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = f"Transient sequence summary row is invalid: {error}"
        raise TransientSequenceArtifactError(message) from error
    if record.record_id != row["record_id"]:
        message = "Transient sequence record identity contradicts its summary row."
        raise TransientSequenceArtifactError(message)
    return record


@dataclass(frozen=True, slots=True)
class TransientSequenceRecordSummary:
    """Describe one case-bundle record without opening numerical arrays."""

    record_id: str
    mode: EvaluationMode
    case_id: str
    dataset_role: DatasetRole
    origin_index: int
    requested_horizon: RequestedHorizon
    available_horizon: int
    trajectory_length: int
    origin_time: float
    elapsed_physical_time: float
    payload_path: str
    payload_sha256: str
    chain_id: str
    metric_statistics: Mapping[str, Mapping[str, object]] | None
    identity: Mapping[str, Any]
    target: Mapping[str, Any]
    timing: Mapping[str, Any]
    exclusion: Mapping[str, Any]


@dataclass(slots=True)
class TransientSequenceArtifactIndex:
    """Hold a validated manifest with bounded on-demand case-bundle loading."""

    root: Path
    dataset_name: str
    dataset_role: DatasetRole
    summaries: tuple[TransientSequenceRecordSummary, ...]
    unavailable_horizons: tuple[dict[str, Any], ...]
    frame: pd.DataFrame
    provenance: Mapping[str, Any]
    identity_sha256: str
    cache_limit: int = 1
    _rows: dict[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    _case_cache: OrderedDict[str, _TransientCasePayload] = field(default_factory=OrderedDict, init=False, repr=False)
    _record_cache: dict[str, TransientSequenceRecord] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the bounded cache and manifest-record correspondence."""
        if isinstance(self.cache_limit, bool) or not isinstance(self.cache_limit, int) or self.cache_limit < 1:
            message = "Transient sequence cache_limit must be a positive integer."
            raise TypeError(message)
        record_ids = tuple(summary.record_id for summary in self.summaries)
        if not record_ids or len(record_ids) != len(set(record_ids)) or set(self._rows) != set(record_ids):
            message = "Transient sequence index rows contradict its summary inventory."
            raise TransientSequenceArtifactError(message)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """Return persisted semantic record identities in manifest order."""
        return tuple(summary.record_id for summary in self.summaries)

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return exact case identities in first manifest appearance order."""
        return tuple(dict.fromkeys(summary.case_id for summary in self.summaries))

    @property
    def cache_size(self) -> int:
        """Return the number of currently decompressed case bundles."""
        return len(self._case_cache)

    def _payload(self, row: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
        case_id = str(row["case_id"])
        cached = self._case_cache.pop(case_id, None)
        if cached is not None:
            self._case_cache[case_id] = cached
            return cached
        payload = _load_case_payload(self.root, row)
        self._case_cache[case_id] = payload
        while len(self._case_cache) > self.cache_limit:
            evicted, evicted_payload = self._case_cache.popitem(last=False)
            evicted_payload.close()
            for record_id, candidate in tuple(self._rows.items()):
                if str(candidate["case_id"]) == evicted:
                    self._record_cache.pop(record_id, None)
        return payload

    def record(self, record_id: str) -> TransientSequenceRecord:
        """Reconstruct one row from its bounded cached case bundle."""
        cached = self._record_cache.get(record_id)
        if cached is not None:
            return cached
        try:
            row = self._rows[record_id]
        except KeyError as error:
            message = f"Unknown transient sequence record {record_id!r}."
            raise KeyError(message) from error
        record = _record_from_row(self.root, row, self._payload(row))
        self._record_cache[record_id] = record
        return record

    def records(self, *, case_id: str | None = None) -> tuple[TransientSequenceRecord, ...]:
        """Return all records or one selected case without reopening its bundle."""
        if case_id is not None and case_id not in self.case_ids:
            message = f"Unknown transient artifact case {case_id!r}."
            raise KeyError(message)
        selected = self.summaries if case_id is None else tuple(summary for summary in self.summaries if summary.case_id == case_id)
        return tuple(self.record(summary.record_id) for summary in selected)

    def close(self) -> None:
        """Close cached NPZ handles and release reconstructed records."""
        for payload in self._case_cache.values():
            payload.close()
        self._case_cache.clear()
        self._record_cache.clear()


_SUMMARY_COLUMNS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "record_id",
        "case_id",
        "dataset_role",
        "mode",
        "origin_index",
        "origin_time",
        "requested_horizon",
        "available_horizon",
        "trajectory_length",
        "elapsed_physical_time",
        "inference_airflow_source",
        "model_kind",
        "dataset_backend",
        "reference_reached",
        "predicted_reached",
        "reference_censored",
        "predicted_censored",
        "reference_time_to_target",
        "predicted_time_to_target",
        "payload_path",
        "payload_sha256",
        "chain_id",
        "metric_statistics_json",
        "identity_json",
        "target_json",
        "timing_json",
        "exclusion_json",
    }
)


def _summary_from_row(root: Path, row: Mapping[str, Any], *, dataset_role: DatasetRole) -> TransientSequenceRecordSummary:
    """Validate one compact record row without decompressing its case bundle."""
    if set(row) != _SUMMARY_COLUMNS:
        message = "Transient sequence summary columns do not match the current schema."
        raise TransientSequenceArtifactError(message)
    if (
        row["schema_kind"] != TRANSIENT_SEQUENCE_SCHEMA_KIND
        or row["schema_version"] != TRANSIENT_SEQUENCE_SCHEMA_VERSION
        or row["dataset_role"] != dataset_role
    ):
        message = "Transient sequence summary schema or Dataset role is incompatible."
        raise TransientSequenceArtifactError(message)
    mode = row["mode"]
    if mode not in {"teacher_forced_one_step", "autonomous_full", "rolling_origin"}:
        message = "Transient sequence summary contains an unsupported mode."
        raise TransientSequenceArtifactError(message)
    case_id = _nonempty_text(row["case_id"], label="summary case_id")
    origin = _nonnegative_int(row["origin_index"], label="summary origin_index")
    available = _nonnegative_int(row["available_horizon"], label="summary available_horizon")
    trajectory_length = _nonnegative_int(row["trajectory_length"], label="summary trajectory_length")
    requested: RequestedHorizon = (
        "full" if row["requested_horizon"] == "full" else _nonnegative_int(int(row["requested_horizon"]), label="summary requested_horizon")
    )
    if (
        available < 1
        or trajectory_length < _MINIMUM_TRAJECTORY_STATES
        or origin + available >= trajectory_length
        or requested not in ("full", available)
        or (mode == "teacher_forced_one_step" and requested != 1)
    ):
        message = "Transient sequence summary horizon evidence is contradictory."
        raise TransientSequenceArtifactError(message)
    _require_relative_payload(root, row["payload_path"])
    payload_digest = _nonempty_text(
        row["payload_sha256"],
        label="summary payload_sha256",
    )
    try:
        identity = _validate_identity(json.loads(str(row["identity_json"])), case_id=case_id, dataset_role=dataset_role)
        target = _validate_target(json.loads(str(row["target_json"])))
        timing = _json_copy(json.loads(str(row["timing_json"])), label="sequence timing")
        exclusion = _validate_exclusion(json.loads(str(row["exclusion_json"])))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        message = f"Transient sequence summary JSON is invalid: {error}"
        raise TransientSequenceArtifactError(message) from error
    record_id = _nonempty_text(row["record_id"], label="summary record_id")
    expected = common.serialization.canonical_json_sha256(
        {
            "schema": TRANSIENT_SEQUENCE_SCHEMA_VERSION,
            "mode": mode,
            "case_id": case_id,
            "dataset_role": dataset_role,
            "origin_index": origin,
            "requested_horizon": requested,
            "inference_airflow_source": identity["inference_airflow_source"],
            "checkpoint_identity": identity["checkpoint_identity"],
        }
    )
    if record_id != expected:
        message = "Transient sequence summary record identity is invalid."
        raise TransientSequenceArtifactError(message)
    if (
        row["inference_airflow_source"] != identity["inference_airflow_source"]
        or row["model_kind"] != identity["model_kind"]
        or row["dataset_backend"] != identity["dataset_backend"]
    ):
        message = "Transient sequence summary duplicates contradict identity JSON."
        raise TransientSequenceArtifactError(message)
    chain_id = _nonempty_text(row["chain_id"], label="summary chain_id")
    expected_chain_id = _chain_id_from_coordinates(
        mode=str(mode),
        origin_index=origin,
    )
    if chain_id != expected_chain_id:
        message = "Transient sequence prediction-chain identity is invalid."
        raise TransientSequenceArtifactError(message)
    raw_statistics = row["metric_statistics_json"]
    metric_statistics: Mapping[str, Mapping[str, object]] | None = None
    if raw_statistics is not None:
        try:
            parsed_statistics = json.loads(str(raw_statistics))
        except json.JSONDecodeError as error:
            message = "Transient record metric statistics are not valid JSON."
            raise TransientSequenceArtifactError(message) from error
        if not isinstance(parsed_statistics, dict) or set(parsed_statistics) != {
            "cumulative",
            "endpoint",
            "diagnostics",
        }:
            message = "Transient record metric statistics are invalid."
            raise TransientSequenceArtifactError(message)
        for scope in ("cumulative", "endpoint"):
            state = parsed_statistics[scope]
            if not isinstance(state, dict):
                message = "Transient record metric sufficient statistics must be mappings."
                raise TransientSequenceArtifactError(message)
            transient_metrics.TransientMetricAccumulator.from_state_dict(state)
        diagnostics = parsed_statistics["diagnostics"]
        diagnostic_fields = {
            "plausibility": {
                "inspected_values",
                "nonfinite_values",
                "negative_moisture_values",
                "relative_humidity_bound_violations",
                "temperature_range_violations",
            },
            "stability": {
                "increment_count",
                "nonfinite_increment_count",
                "oscillatory_increment_count",
                "abnormal_growth_count",
            },
        }
        if not isinstance(diagnostics, dict) or set(diagnostics) != set(diagnostic_fields):
            message = "Transient record diagnostic statistics are invalid."
            raise TransientSequenceArtifactError(message)
        for label, fields in diagnostic_fields.items():
            values = diagnostics[label]
            if (
                not isinstance(values, dict)
                or set(values) != fields
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values())
            ):
                message = f"Transient record {label} statistics are invalid."
                raise TransientSequenceArtifactError(message)
        metric_statistics = cast(
            "Mapping[str, Mapping[str, object]]",
            parsed_statistics,
        )
    return TransientSequenceRecordSummary(
        record_id,
        cast("EvaluationMode", mode),
        case_id,
        dataset_role,
        origin,
        requested,
        available,
        trajectory_length,
        float(row["origin_time"]),
        float(row["elapsed_physical_time"]),
        str(row["payload_path"]),
        payload_digest,
        chain_id,
        metric_statistics,
        identity,
        target,
        timing,
        exclusion,
    )


def validate_transient_sequence_payload_inventory(
    index: TransientSequenceArtifactIndex,
) -> None:
    """Validate case-bundle field inventories without decompressing numerical arrays."""
    if not isinstance(index, TransientSequenceArtifactIndex):
        message = "Transient payload inventory validation requires one admitted index."
        raise TypeError(message)
    common_fields = {
        "physical_times",
        "reference_states",
        "temporal_mask",
        "boundary_conditioning",
        "spatial_mask",
        "static_conditioning",
        "scalar_conditioning",
        "state_order",
        "state_units",
        "static_order",
        "boundary_order",
        "scalar_order",
    }
    for case_id in index.case_ids:
        summaries = tuple(summary for summary in index.summaries if summary.case_id == case_id)
        path = _require_relative_payload(index.root, summaries[0].payload_path)
        chain_fields = {f"chain_{summary.chain_id}_predicted_states" for summary in summaries}
        optional_fields = {
            "reference_increments",
            *(f"chain_{summary.chain_id}_predicted_increments" for summary in summaries),
        }
        with np.load(path, allow_pickle=False) as loaded:
            actual = set(loaded.files)
        allowed = common_fields.union(chain_fields, optional_fields)
        if not common_fields.union(chain_fields).issubset(actual) or actual.difference(allowed):
            message = f"Transient case payload field inventory contradicts its summary for case {case_id!r}."
            raise TransientSequenceArtifactError(message)


def _manifest_npz_digests(outputs: Any) -> dict[str, str]:
    """Return the already-verified output-manifest digest for each NPZ payload."""
    if not isinstance(outputs, Mapping) or set(outputs) != {"parquet", "npz"}:
        message = "Transient artifact output manifest has an unsupported schema."
        raise TransientSequenceArtifactError(message)
    npz_entries = outputs["npz"]
    if not isinstance(npz_entries, list):
        message = "Transient artifact output manifest NPZ entries must be a list."
        raise TransientSequenceArtifactError(message)
    digests: dict[str, str] = {}
    for entry in npz_entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            message = "Transient artifact output manifest NPZ entry is invalid."
            raise TransientSequenceArtifactError(message)
        path = entry["path"]
        digest = entry["sha256"]
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
            or path in digests
        ):
            message = "Transient artifact output manifest NPZ entry is unsafe."
            raise TransientSequenceArtifactError(message)
        digests[path] = digest
    return digests


def _validate_case_payload_manifest(
    root: Path,
    paths_by_case: Mapping[str, set[tuple[str, str]]],
    outputs: Any,
) -> None:
    """Bind each case summary payload digest to the already-verified output manifest."""
    manifest_npz_digests = _manifest_npz_digests(outputs)
    case_paths = {next(iter(values))[0] for values in paths_by_case.values()}
    if set(manifest_npz_digests) != case_paths:
        message = "Transient artifact output manifest does not match its case payload inventory."
        raise TransientSequenceArtifactError(message)
    for case_id, values in paths_by_case.items():
        payload_path, payload_digest = next(iter(values))
        if payload_path != _case_payload_path(case_id).as_posix():
            message = "Transient case payload path contradicts its exact case identity."
            raise TransientSequenceArtifactError(message)
        _require_relative_payload(root, payload_path)
        if manifest_npz_digests[payload_path] != payload_digest:
            message = "Transient case payload digest contradicts the output manifest."
            raise TransientSequenceArtifactError(message)


def _validate_runtime_performance(provenance: Mapping[str, Any]) -> None:
    """Admit optional operational telemetry outside scientific identity."""
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        message = "Transient artifact runtime provenance must be a mapping."
        raise TransientSequenceArtifactError(message)
    operational_performance = runtime.get("operational_performance")
    if operational_performance is None:
        return
    try:
        artifact_performance.validate_operational_performance(operational_performance)
    except (TypeError, ValueError) as error:
        message = "Transient artifact operational performance is invalid."
        raise TransientSequenceArtifactError(message) from error


def _index_from_admitted_frame(
    artifact_root: Path,
    *,
    provenance: Mapping[str, Any],
    frame: pd.DataFrame,
    cache_limit: int,
) -> TransientSequenceArtifactIndex:
    """Build one strict lazy index from already-admitted output digests and rows."""
    dataset = provenance["dataset"]
    if not isinstance(dataset, Mapping):
        message = "Transient artifact Dataset provenance must be a mapping."
        raise TypeError(message)
    dataset_name = common.paths.validate_logical_name(
        dataset.get("name"),
        label="artifact dataset.name",
    )
    dataset_role = dataset.get("role")
    if dataset_role not in {"id", "ood"}:
        message = "Transient artifact Dataset role must be id or ood."
        raise TransientSequenceArtifactError(message)
    if frame.empty or "record_id" not in frame or frame["record_id"].duplicated().any():
        message = "Transient artifact summary must contain unique non-empty records."
        raise TransientSequenceArtifactError(message)
    rows = tuple(frame.to_dict(orient="records"))
    admitted_role = cast("DatasetRole", dataset_role)
    summaries = tuple(
        _summary_from_row(
            artifact_root,
            row,
            dataset_role=admitted_role,
        )
        for row in rows
    )
    paths_by_case: dict[str, set[tuple[str, str]]] = {}
    for summary in summaries:
        paths_by_case.setdefault(summary.case_id, set()).add(
            (summary.payload_path, summary.payload_sha256),
        )
    if any(len(paths) != 1 for paths in paths_by_case.values()):
        message = "One exact transient case must share exactly one case payload path and digest."
        raise TransientSequenceArtifactError(message)
    if len({summary.payload_path for summary in summaries}) != len(paths_by_case):
        message = "Transient case payload path is shared across distinct cases."
        raise TransientSequenceArtifactError(message)
    _validate_case_payload_manifest(
        artifact_root,
        paths_by_case,
        provenance["outputs"],
    )
    statistic_capabilities = {summary.metric_statistics is not None for summary in summaries}
    if len(statistic_capabilities) != 1:
        message = "Transient metric-statistic capability must be present for all records or none."
        raise TransientSequenceArtifactError(message)
    availability = provenance["availability"]
    if not isinstance(availability, Mapping) or set(availability) != {
        "available_record_count",
        "unavailable_horizon_count",
        "unavailable_horizons",
    }:
        message = "Transient artifact availability evidence is invalid."
        raise TransientSequenceArtifactError(message)
    unavailable = availability["unavailable_horizons"]
    if (
        availability["available_record_count"] != len(summaries)
        or not isinstance(unavailable, list)
        or availability["unavailable_horizon_count"] != len(unavailable)
    ):
        message = "Transient artifact availability counts contradict payload contents."
        raise TransientSequenceArtifactError(message)
    validated_unavailable = tuple(
        UnavailableHorizon(
            case_id=item["case_id"],
            dataset_role=item["dataset_role"],
            origin_index=item["origin_index"],
            requested_horizon=item["requested_horizon"],
            available_transitions=item["available_transitions"],
            reason=item["reason"],
        ).as_dict()
        for item in unavailable
    )
    return TransientSequenceArtifactIndex(
        artifact_root,
        dataset_name,
        admitted_role,
        summaries,
        validated_unavailable,
        frame,
        provenance,
        str(provenance["identity_sha256"]),
        cache_limit,
        {summary.record_id: row for summary, row in zip(summaries, rows, strict=True)},
    )


def load_transient_sequence_artifact_index(root: Path | str, *, cache_limit: int = 1) -> TransientSequenceArtifactIndex:
    """Admit a complete sequence manifest without opening case numerical payloads."""
    artifact_root = Path(root).expanduser().resolve(strict=True)
    marker = contracts.artifact_provenance_path(artifact_root)
    if not marker.is_file() or marker.is_symlink():
        message = f"Transient artifact completion marker is missing or unsafe: {marker}"
        raise FileNotFoundError(message)
    with marker.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    required = {
        "provenance_schema_version",
        "artifact_schema_version",
        "artifact_kind",
        "task",
        "run",
        "dataset",
        "evaluation",
        "runtime",
        "lineage",
        "availability",
        "outputs",
        "identity_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        message = "Transient artifact provenance fields do not match the schema."
        raise TransientSequenceArtifactError(message)
    expected_static = {
        "provenance_schema_version": contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": TRANSIENT_SEQUENCE_SCHEMA_VERSION,
        "artifact_kind": TRANSIENT_ARTIFACT_KIND,
        "task": "transient_drying",
    }
    if any(provenance[key] != value for key, value in expected_static.items()):
        message = "Transient artifact schema or task identity is unsupported."
        raise TransientSequenceArtifactError(message)
    if provenance["outputs"] != contracts.artifact_output_manifest(artifact_root):
        message = "Transient artifact output manifest does not match current payload bytes."
        raise TransientSequenceArtifactError(message)
    identity_document = _scientific_identity_document(provenance)
    if provenance["identity_sha256"] != common.serialization.canonical_json_sha256(identity_document):
        message = "Transient artifact identity digest is invalid."
        raise TransientSequenceArtifactError(message)
    _validate_runtime_performance(provenance)
    run = provenance["run"]
    if not isinstance(run, dict) or "parent_experiment" not in run:
        message = "Transient artifact run provenance lacks parent experiment evidence."
        raise TransientSequenceArtifactError(message)
    _validate_parent_experiment_evidence(run["parent_experiment"])
    dataset = provenance["dataset"]
    if not isinstance(dataset, dict):
        message = "Transient artifact Dataset provenance must be a mapping."
        raise TypeError(message)
    dataset_name = common.paths.validate_logical_name(dataset.get("name"), label="artifact dataset.name")
    dataset_role = dataset.get("role")
    if dataset_role not in {"id", "ood"}:
        message = "Transient artifact Dataset role must be id or ood."
        raise TransientSequenceArtifactError(message)
    parquet_files = sorted(artifact_root.glob("*.parquet"))
    if len(parquet_files) != 1 or parquet_files[0].name != f"{dataset_name}.parquet":
        message = "Transient artifact Parquet filename contradicts its Dataset identity."
        raise TransientSequenceArtifactError(message)
    frame = pd.read_parquet(parquet_files[0])
    return _index_from_admitted_frame(
        artifact_root,
        provenance=provenance,
        frame=frame,
        cache_limit=cache_limit,
    )


def load_transient_sequence_artifact(root: Path | str) -> LoadedTransientSequenceArtifact:
    """Read every record reconstructed from strict indexed case bundles."""
    index = load_transient_sequence_artifact_index(root)
    records = index.records()
    if any(record.dataset_role != index.dataset_role for record in records):
        message = "Transient artifact records contradict the role-level Dataset identity."
        raise TransientSequenceArtifactError(message)
    return LoadedTransientSequenceArtifact(
        index.root, index.dataset_name, index.dataset_role, records, index.unavailable_horizons, index.frame, index.provenance, index.identity_sha256
    )
