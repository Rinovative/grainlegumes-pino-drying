"""
evaluation_transient_session.py

Summarize validated transient sequence artifacts without refitting state.

Responsibilities:
  - Admit transient-sequence frames and their persisted scaling evidence
  - Aggregate complete-dataset endpoint and cumulative error statistics
  - Expose bounded case, dataset, and flat tracking-summary views

Design principles:
  - Persisted scaling is reconstructed exactly and never refitted
  - Origin states are excluded from every error reduction
  - Fixed horizons remain exact semantic values, not display labels

This module does NOT:
  - Load files, run inference, mutate frames, or publish artifacts
  - Average finalized RMSE values across cases or batches
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch

from src import common
from src.learning.transient import learning_transient_scaling as scaling

from . import evaluation_transient_artifact as sequence_artifact
from . import evaluation_transient_comparison as comparison
from . import evaluation_transient_metrics as metrics
from . import evaluation_transient_timing as timing

_DEFAULT_MODES = ("teacher_forced_one_step", "autonomous_full", "rolling_origin")
_MAX_TRACKING_SUMMARY_ENTRIES = 470
_MAX_PUBLICATION_METADATA_ENTRIES = 42
_MAX_PUBLICATION_SUMMARY_ENTRIES = 512
_MAX_TRACKING_TEXT_LENGTH = 512
_PUBLICATION_TIMING_COMPONENTS = (
    "comsol_transient_drying_seconds",
    "drying_no_rollout_model_seconds",
)
_ORIGIN_PLUS_TRANSITION_COUNT = 2
_SPATIAL_MASK_RANK = 2
_STATE_ARRAY_RANK = 4
_STATE_CHANNEL_COUNT = len(sequence_artifact.STATE_ORDER)
_RHO_BULK_DENSITY_INDEX = sequence_artifact.STATIC_ORDER.index("rho_bu_dry")
_REQUIRED_FRAME_ATTRS = frozenset(
    {
        "artifact_kind",
        "transient_unavailable_horizons",
        "transient_scaling_state",
    }
)
_TransientMetricSummary = metrics.TransientMetricSummary
_TargetCensoringDiagnostics = metrics.TargetCensoringDiagnostics
_PlausibilityDiagnostics = metrics.PlausibilityDiagnostics
_StabilityDiagnostics = metrics.StabilityDiagnostics


@dataclass(frozen=True, slots=True)
class TransientSessionSummary:
    """Store one complete-dataset frame/material/mode/horizon/scope reduction."""

    frame: str
    material_family: str | None
    mode: str
    requested_horizon: int | str
    scope: str
    metrics: _TransientMetricSummary
    contributing_record_count: int
    contributing_case_count: int
    unavailable_case_count: int
    elapsed_physical_time_min: float
    elapsed_physical_time_median: float
    elapsed_physical_time_max: float
    origin_time_min: float
    origin_time_max: float
    target: _TargetCensoringDiagnostics
    target_time_error_count: int
    target_time_error_mean: float | None
    target_time_error_mae: float | None
    target_gap_count: int
    predicted_final_target_gap_mean: float | None
    reference_final_target_gap_mean: float | None
    target_final_gap_error_mean: float | None
    target_final_gap_error_mae: float | None
    plausibility: _PlausibilityDiagnostics
    stability: _StabilityDiagnostics


@dataclass(frozen=True, slots=True)
class TransientCaseInventoryEntry:
    """Bind one exact case to its material, Dataset, and owning artifact."""

    frame_name: str
    case_id: str
    case_label: str
    material_family: str
    dataset_role: Literal["id", "ood"]
    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    membership_digests: tuple[str, ...]
    artifact_root: Path
    artifact_identity_sha256: str
    dataset_identity: Mapping[str, Any]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one mapping or raise one contract-focused error."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    return value


def _record_value(record: Any, name: str) -> Any:
    """Read one record field from a validated object or mapping."""
    if isinstance(record, Mapping):
        try:
            return record[name]
        except KeyError as error:
            message = f"Transient sequence record lacks {name!r}."
            raise ValueError(message) from error
    try:
        return getattr(record, name)
    except AttributeError as error:
        message = f"Transient sequence record lacks {name!r}."
        raise ValueError(message) from error


def _sequence_records(frame: pd.DataFrame) -> tuple[Any, ...]:
    """Admit immutable transient-record references from one frame attribute."""
    if not isinstance(frame, pd.DataFrame):
        msg = "Transient session frames must be pandas DataFrames."
        raise TypeError(msg)
    attrs = frame.attrs
    if not _REQUIRED_FRAME_ATTRS.issubset(attrs):
        missing = sorted(_REQUIRED_FRAME_ATTRS.difference(attrs))
        message = f"Transient sequence frame lacks required attrs {missing}."
        raise ValueError(message)
    if attrs["artifact_kind"] != "transient_sequence":
        msg = "Transient session requires artifact_kind='transient_sequence'."
        raise ValueError(msg)
    records = attrs.get("transient_sequence_records")
    if isinstance(records, tuple) and records:
        return records
    index = attrs.get("transient_sequence_index")
    if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
        return index.summaries
    msg = "Transient frame requires eager records or one validated sequence index."
    raise ValueError(msg)


def _validate_sequence_source(frame: pd.DataFrame) -> None:
    """Validate eager or indexed sequence availability without loading arrays."""
    attrs = frame.attrs
    if not _REQUIRED_FRAME_ATTRS.issubset(attrs):
        missing = sorted(_REQUIRED_FRAME_ATTRS.difference(attrs))
        message = f"Transient sequence frame lacks required attrs {missing}."
        raise ValueError(message)
    if attrs["artifact_kind"] != "transient_sequence":
        msg = "Transient session requires artifact_kind='transient_sequence'."
        raise ValueError(msg)
    records = attrs.get("transient_sequence_records")
    index = attrs.get("transient_sequence_index")
    if isinstance(records, tuple) and records:
        return
    if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex) and index.summaries:
        return
    msg = "Transient frame requires eager records or one non-empty validated index."
    raise ValueError(msg)


def _scale(frame: pd.DataFrame) -> scaling.TransientScalingArtifact:
    """Reconstruct one persisted train-only scale artifact on CPU."""
    state = _require_mapping(frame.attrs["transient_scaling_state"], label="transient_scaling_state")
    return scaling.TransientScalingArtifact.from_state_dict(state)


def _requested_horizon(record: Any) -> int | str:
    """Return one preserved fixed or full horizon value."""
    value = _record_value(record, "requested_horizon")
    if value == "full":
        return "full"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = "Transient record requested_horizon must be positive integer or 'full'."
        raise ValueError(msg)
    return value


def _spatial_mask(record: Any, *, count: int) -> np.ndarray:
    """Expand one persisted spatial mask across reconstructed state channels."""
    mask = np.asarray(_record_value(record, "spatial_mask"))
    if mask.dtype != np.bool_ or mask.ndim != _SPATIAL_MASK_RANK or not bool(mask.any()):
        msg = "Transient record spatial_mask must be non-empty boolean [Y,X]."
        raise ValueError(msg)
    return np.broadcast_to(mask[None, None], (count, _STATE_CHANNEL_COUNT, *mask.shape)).copy()


def _bulk_moisture_inputs(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical dry density and masked trapezoidal cell weights."""
    mask = _spatial_mask(record, count=1)[0, 0]
    static = np.asarray(_record_value(record, "static_conditioning"), dtype=np.float64)
    expected_shape = (len(sequence_artifact.STATIC_ORDER), *mask.shape)
    if static.shape != expected_shape or not np.isfinite(static).all():
        msg = "Transient record static_conditioning must match the canonical finite Dataset fields."
        raise ValueError(msg)
    return static[_RHO_BULK_DENSITY_INDEX], metrics.trapezoidal_cell_weights(mask)


def _normalized_states(
    artifact: scaling.TransientScalingArtifact,
    states: np.ndarray,
) -> np.ndarray:
    """Encode physical absolute states with admitted saved scaling only."""
    raw = np.asarray(states, dtype=np.float32)
    if raw.ndim != _STATE_ARRAY_RANK or raw.shape[1] != _STATE_CHANNEL_COUNT or not np.isfinite(raw).all():
        msg = "Transient record states must be finite [time,4,Y,X] arrays."
        raise ValueError(msg)
    encoded = artifact.encode_state(torch.from_numpy(np.ascontiguousarray(raw)))
    return encoded.detach().cpu().numpy()


def _required_pipeline_metric(values: Mapping[str, Any], name: str) -> float:
    """Return one finite non-negative pipeline metric without filling absence."""
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"Pipeline metric {name!r} must be a real scalar."
        raise TypeError(msg)
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        msg = f"Pipeline metric {name!r} must be finite and non-negative."
        raise ValueError(msg)
    return result


def _target_values(
    records: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], list[float], list[float]]:
    """Return target status, reached-time error, and time-aligned final-gap evidence."""
    predicted: list[bool | None] = []
    reference: list[bool | None] = []
    time_errors: list[float] = []
    predicted_gaps: list[float] = []
    reference_gaps: list[float] = []
    gap_errors: list[float] = []
    for record in records:
        target = _require_mapping(_record_value(record, "target"), label="record target")
        pred_available = target.get("predicted_available") is True
        ref_available = target.get("reference_available") is True
        predicted.append(target.get("predicted_reached") if pred_available else None)
        reference.append(target.get("reference_reached") if ref_available else None)
        pred_time, ref_time = target.get("predicted_time_to_target"), target.get("reference_time_to_target")
        if pred_available and ref_available and isinstance(pred_time, Real) and isinstance(ref_time, Real):
            time_errors.append(float(pred_time) - float(ref_time))

        pred_gap = target.get("predicted_final_gap")
        ref_gap = target.get("reference_final_gap")
        admitted_prediction: float | None = None
        admitted_reference: float | None = None
        if pred_available:
            if isinstance(pred_gap, bool) or not isinstance(pred_gap, Real) or not np.isfinite(float(pred_gap)):
                msg = "Available predicted target evidence requires one finite final gap."
                raise ValueError(msg)
            admitted_prediction = float(pred_gap)
            predicted_gaps.append(admitted_prediction)
        if ref_available:
            if isinstance(ref_gap, bool) or not isinstance(ref_gap, Real) or not np.isfinite(float(ref_gap)):
                msg = "Available reference target evidence requires one finite final gap."
                raise ValueError(msg)
            admitted_reference = float(ref_gap)
            reference_gaps.append(admitted_reference)
        pred_final_time = target.get("predicted_final_time")
        ref_final_time = target.get("reference_final_time")
        if (
            admitted_prediction is not None
            and admitted_reference is not None
            and isinstance(pred_final_time, Real)
            and not isinstance(pred_final_time, bool)
            and isinstance(ref_final_time, Real)
            and not isinstance(ref_final_time, bool)
            and float(pred_final_time) == float(ref_final_time)
        ):
            gap_errors.append(admitted_prediction - admitted_reference)
    return np.asarray(predicted, dtype=object), np.asarray(reference, dtype=object), time_errors, predicted_gaps, reference_gaps, gap_errors


def _empty_plausibility() -> metrics.PlausibilityDiagnostics:
    """Return zero diagnostics when a summary has no contributing records."""
    return metrics.PlausibilityDiagnostics(0, 0, 0, 0, 0)


def _empty_stability() -> metrics.StabilityDiagnostics:
    """Return zero stability diagnostics when a summary has no records."""
    return metrics.StabilityDiagnostics(0, 0, 0, 0)


def _persisted_diagnostics(
    value: Mapping[str, object],
) -> tuple[metrics.PlausibilityDiagnostics, metrics.StabilityDiagnostics]:
    """Reconstruct already-admitted integer diagnostics from one summary row."""
    plausibility = _require_mapping(
        value.get("plausibility"),
        label="metric_statistics.diagnostics.plausibility",
    )
    stability = _require_mapping(
        value.get("stability"),
        label="metric_statistics.diagnostics.stability",
    )
    return (
        metrics.PlausibilityDiagnostics(
            inspected_values=int(plausibility["inspected_values"]),
            nonfinite_values=int(plausibility["nonfinite_values"]),
            negative_moisture_values=int(plausibility["negative_moisture_values"]),
            relative_humidity_bound_violations=int(plausibility["relative_humidity_bound_violations"]),
            temperature_range_violations=int(plausibility["temperature_range_violations"]),
        ),
        metrics.StabilityDiagnostics(
            increment_count=int(stability["increment_count"]),
            nonfinite_increment_count=int(stability["nonfinite_increment_count"]),
            oscillatory_increment_count=int(stability["oscillatory_increment_count"]),
            abnormal_growth_count=int(stability["abnormal_growth_count"]),
        ),
    )


def _add_optional_tracking_value(
    values: dict[str, float | int],
    key: str,
    value: float | None,
) -> None:
    """Append one available finite aggregate without inventing missing evidence."""
    if value is not None:
        values[key] = value


def _summary_row(summary: TransientSessionSummary) -> dict[str, Any]:
    """Flatten one complete summary into bounded semantic scalar columns."""
    row: dict[str, Any] = {
        "frame": summary.frame,
        "material_family": summary.material_family,
        "mode": summary.mode,
        "requested_horizon": summary.requested_horizon,
        "scope": summary.scope,
        "normalized_drying_group_macro_rmse": summary.metrics.normalized_drying_group_macro_rmse,
        "physical_w_gr_rmse": summary.metrics.physical_w_gr_rmse,
        "physical_w_gr_mae": summary.metrics.physical_w_gr_mae,
        "bulk_dry_basis_rmse": summary.metrics.bulk_dry_basis_rmse,
        "bulk_dry_basis_mae": summary.metrics.bulk_dry_basis_mae,
        "bulk_wet_basis_rmse": summary.metrics.bulk_wet_basis_rmse,
        "bulk_wet_basis_mae": summary.metrics.bulk_wet_basis_mae,
        "predicted_bulk_dry_basis_mean": summary.metrics.predicted_bulk_dry_basis_mean,
        "reference_bulk_dry_basis_mean": summary.metrics.reference_bulk_dry_basis_mean,
        "predicted_bulk_wet_basis_mean": summary.metrics.predicted_bulk_wet_basis_mean,
        "reference_bulk_wet_basis_mean": summary.metrics.reference_bulk_wet_basis_mean,
        "bulk_moisture_valid_count": summary.metrics.bulk_moisture_valid_count,
        "bulk_moisture_unavailable_count": summary.metrics.bulk_moisture_unavailable_count,
        "contributing_record_count": summary.contributing_record_count,
        "contributing_case_count": summary.contributing_case_count,
        "unavailable_case_count": summary.unavailable_case_count,
        "elapsed_physical_time_min": summary.elapsed_physical_time_min,
        "elapsed_physical_time_median": summary.elapsed_physical_time_median,
        "elapsed_physical_time_max": summary.elapsed_physical_time_max,
        "origin_time_min": summary.origin_time_min,
        "origin_time_max": summary.origin_time_max,
        "target_available_count": summary.target.available_count,
        "predicted_reached_count": summary.target.predicted_reached_count,
        "reference_reached_count": summary.target.reference_reached_count,
        "predicted_right_censored_count": summary.target.predicted_right_censored_count,
        "reference_right_censored_count": summary.target.reference_right_censored_count,
        "target_agreement_count": summary.target.agreement_count,
        "target_time_error_count": summary.target_time_error_count,
        "target_time_error_mean": summary.target_time_error_mean,
        "target_time_error_mae": summary.target_time_error_mae,
        "target_gap_count": summary.target_gap_count,
        "predicted_final_target_gap_mean": summary.predicted_final_target_gap_mean,
        "reference_final_target_gap_mean": summary.reference_final_target_gap_mean,
        "target_final_gap_error_mean": summary.target_final_gap_error_mean,
        "target_final_gap_error_mae": summary.target_final_gap_error_mae,
        "nonfinite_values": summary.plausibility.nonfinite_values,
        "negative_moisture_values": summary.plausibility.negative_moisture_values,
        "relative_humidity_bound_violations": summary.plausibility.relative_humidity_bound_violations,
        "temperature_range_violations": summary.plausibility.temperature_range_violations,
        "oscillatory_increment_count": summary.stability.oscillatory_increment_count,
        "abnormal_increment_growth_count": summary.stability.abnormal_growth_count,
    }
    for field, value in summary.metrics.normalized_rmse.items():
        row[f"normalized_rmse_{field}"] = value
    for field, value in summary.metrics.physical_rmse.items():
        row[f"physical_rmse_{field}"] = value
    for field, value in summary.metrics.physical_mae.items():
        row[f"physical_mae_{field}"] = value
    for field, value in summary.metrics.relative_l2.items():
        row[f"relative_l2_{field}"] = value
    return row


def _tracking_text(values: Sequence[Any], *, label: str) -> str:
    """Return one bounded stable text identity from non-empty scalar evidence."""
    admitted: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            msg = f"{label} must contain non-empty text identities."
            raise TypeError(msg)
        admitted.add(value)
    if not admitted:
        msg = f"{label} must contain at least one identity."
        raise ValueError(msg)
    result = "|".join(sorted(admitted))
    if len(result) > _MAX_TRACKING_TEXT_LENGTH:
        msg = f"{label} exceeds the bounded tracking text contract."
        raise ValueError(msg)
    return result


def _record_identity(record: Any) -> Mapping[str, Any]:
    """Return one admitted sequence-record identity mapping."""
    return _require_mapping(_record_value(record, "identity"), label="sequence record identity")


def _material_family(record: Any) -> str:
    """Return one authoritative material identity without parsing a case label."""
    material = _record_identity(record).get("material_family")
    if not isinstance(material, str) or not material:
        msg = "Transient sequence record identity requires material_family."
        raise ValueError(msg)
    return material


def _nonnegative_tracking_duration(value: Any, *, label: str) -> float:
    """Return one finite non-negative locally aggregated duration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real duration."
        raise TypeError(msg)
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        msg = f"{label} must be finite and non-negative."
        raise ValueError(msg)
    return result


def _add_publication_timing_metadata(
    metadata: dict[str, bool | float | int | str],
    *,
    timing_cases: Sequence[timing.TransientTimingCase],
    materialization_seconds: Sequence[float],
    hardware_evidence: Sequence[Mapping[str, Any]],
) -> None:
    """Append bounded cross-case timing aggregates without case-level payloads."""
    combined_report = timing.build_transient_timing_report(timing_cases)
    metadata["evaluation/timing/device"] = _tracking_text(
        [case.device for case in combined_report.cases],
        label="timing device identity",
    )
    metadata["evaluation/timing/precision"] = _tracking_text(
        [case.precision for case in combined_report.cases],
        label="timing precision identity",
    )
    warmup_passes = {case.warmup_passes for case in combined_report.cases}
    if len(warmup_passes) != 1:
        msg = "Transient publication timing cases must share one warm-up contract."
        raise ValueError(msg)
    metadata["evaluation/timing/warmup_passes"] = next(iter(warmup_passes))
    metadata["evaluation/timing/hardware_identity_sha256"] = common.serialization.canonical_json_sha256(hardware_evidence)
    metadata["evaluation/timing/component_composed"] = combined_report.component_composed

    for name, summary in combined_report.speedups.items():
        prefix = f"evaluation/timing/speedup/{name}"
        metadata[f"{prefix}/available_count"] = summary.available_count
        metadata[f"{prefix}/unavailable_count"] = summary.unavailable_count
        if summary.ratio_of_sums is not None:
            metadata[f"{prefix}/ratio_of_sums"] = summary.ratio_of_sums
    for component in _PUBLICATION_TIMING_COMPONENTS:
        case_medians = timing.component_case_medians(combined_report, component)
        prefix = f"evaluation/timing/component/{component}"
        metadata[f"{prefix}/available_count"] = len(case_medians)
        if case_medians:
            metadata[f"{prefix}/median_seconds"] = float(np.median(np.asarray(tuple(case_medians.values()), dtype=np.float64)))
    materialization = np.asarray(materialization_seconds, dtype=np.float64)
    metadata["evaluation/timing/component/dataset_materialization_seconds/available_count"] = int(materialization.size)
    metadata["evaluation/timing/component/dataset_materialization_seconds/median_seconds"] = float(np.median(materialization))


class TransientEvaluationSession:
    """
    Hold bounded read-only reductions over one mapping of transient artifact frames.

    Parameters
    ----------
    frames : Mapping[str, pandas.DataFrame]
        Named already-validated transient artifact frames. Every frame carries the
        persisted transient scaling state used to normalize absolute states.

    """

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        """Validate frame contracts and reconstruct their persisted scale artifacts."""
        if not isinstance(frames, Mapping) or not frames:
            msg = "Transient evaluation session requires one non-empty frame mapping."
            raise ValueError(msg)
        admitted: dict[str, tuple[pd.DataFrame, scaling.TransientScalingArtifact]] = {}
        for name, frame in frames.items():
            if not isinstance(name, str) or not name:
                msg = "Transient session frame names must be non-empty text."
                raise ValueError(msg)
            _validate_sequence_source(frame)
            admitted[name] = (frame, _scale(frame))
        self._frames = MappingProxyType(admitted)
        self._case_dataframe_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._summary_cache: dict[tuple[str, ...], tuple[TransientSessionSummary, ...]] = {}
        self._dataset_dataframe_cache: dict[tuple[str, ...], pd.DataFrame] = {}

    @property
    def frame_names(self) -> tuple[str, ...]:
        """Return stable named frame order."""
        return tuple(self._frames)

    @property
    def canonical_frames(self) -> Mapping[str, pd.DataFrame]:
        """Return live admitted frames without exposing mutable session ownership."""
        return MappingProxyType({name: frame for name, (frame, _artifact) in self._frames.items()})

    @property
    def artifact_roots(self) -> tuple[Path, ...]:
        """Return exact artifact roots for report-output separation checks."""
        roots: list[Path] = []
        for frame, _artifact in self._frames.values():
            value = frame.attrs.get("artifact_root")
            if not isinstance(value, str) or not value:
                msg = "Transient report rendering requires each artifact_root attribute."
                raise ValueError(msg)
            roots.append(Path(value).resolve())
        return tuple(roots)

    def records(self, frame_name: str) -> tuple[Any, ...]:
        """Return admitted sequence records for one named frame."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        if isinstance(
            index,
            sequence_artifact.TransientSequenceArtifactIndex,
        ):
            return index.records()
        return _sequence_records(frame)

    def material_families(self, frame_name: str) -> tuple[str, ...]:
        """Return exact material identities without opening indexed payloads."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        return tuple(dict.fromkeys(_material_family(record) for record in _sequence_records(frame)))

    def dataset_role(self, frame_name: str) -> Literal["id", "ood"]:
        """Return the exact run-relative Dataset role for one artifact frame."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
            return index.dataset_role
        provenance = frame.attrs.get("artifact_provenance")
        dataset = provenance.get("dataset") if isinstance(provenance, Mapping) else None
        role = dataset.get("role") if isinstance(dataset, Mapping) else None
        if role in {"id", "ood"}:
            return cast("Literal['id', 'ood']", role)
        roles = {_record_value(record, "dataset_role") for record in _sequence_records(frame)}
        if roles == {"id"}:
            return "id"
        if roles == {"ood"}:
            return "ood"
        msg = f"Transient artifact frame {frame_name!r} has inconsistent Dataset roles."
        raise ValueError(msg)

    def case_inventory(self) -> tuple[TransientCaseInventoryEntry, ...]:
        """Return every exact case with immutable owning-artifact provenance."""
        entries: list[TransientCaseInventoryEntry] = []
        for frame_name, (frame, _artifact) in self._frames.items():
            index = frame.attrs.get("transient_sequence_index")
            if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
                summaries = index.summaries
                case_ids = index.case_ids
                provenance = _require_mapping(index.provenance, label="transient artifact provenance")
                dataset_name = index.dataset_name
                artifact_root = index.root
                artifact_identity = index.identity_sha256
            else:
                summaries = _sequence_records(frame)
                case_ids = tuple(dict.fromkeys(str(_record_value(summary, "case_id")) for summary in summaries))
                provenance = _require_mapping(
                    frame.attrs.get("artifact_provenance"),
                    label="transient artifact provenance",
                )
                dataset_value = _require_mapping(
                    provenance.get("dataset"),
                    label="transient artifact Dataset provenance",
                )
                dataset_name_value = dataset_value.get("name")
                artifact_root_value = frame.attrs.get("artifact_root")
                if (
                    not isinstance(dataset_name_value, str)
                    or not dataset_name_value
                    or not isinstance(artifact_root_value, str)
                    or not artifact_root_value
                ):
                    msg = "Transient case inventory requires exact Dataset and artifact-root identity."
                    raise ValueError(msg)
                dataset_name = dataset_name_value
                artifact_root = Path(artifact_root_value).resolve()
                artifact_identity_value = provenance.get("identity_sha256")
                artifact_identity = (
                    artifact_identity_value
                    if isinstance(artifact_identity_value, str) and artifact_identity_value
                    else common.serialization.canonical_json_sha256(provenance)
                )
            dataset = _require_mapping(provenance.get("dataset"), label="transient artifact Dataset provenance")
            source_dataset_ids = dataset.get("source_dataset_ids")
            membership_digests = dataset.get("membership_digests")
            if (
                not isinstance(source_dataset_ids, list)
                or not source_dataset_ids
                or any(not isinstance(value, str) or not value for value in source_dataset_ids)
                or not isinstance(membership_digests, list)
                or not membership_digests
                or any(not isinstance(value, str) or not value for value in membership_digests)
            ):
                msg = "Transient case inventory requires exact Dataset source and membership identities."
                raise ValueError(msg)
            first_by_case = {str(_record_value(summary, "case_id")): summary for summary in summaries}
            for case_id in case_ids:
                summary = first_by_case[case_id]
                identity = _record_identity(summary)
                dataset_identity_value = identity.get("dataset_identity")
                dataset_identity = dataset_identity_value if isinstance(dataset_identity_value, Mapping) else {"artifact_dataset_name": dataset_name}
                simulation_identity = identity.get("simulation_identity")
                compact_case_id = simulation_identity.get("generation_case_id") if isinstance(simulation_identity, Mapping) else None
                if not isinstance(compact_case_id, str) or not compact_case_id:
                    compact_case_id = case_id
                entries.append(
                    TransientCaseInventoryEntry(
                        frame_name=frame_name,
                        case_id=case_id,
                        case_label=compact_case_id,
                        material_family=_material_family(summary),
                        dataset_role=self.dataset_role(frame_name),
                        dataset_name=dataset_name,
                        source_dataset_ids=tuple(source_dataset_ids),
                        membership_digests=tuple(membership_digests),
                        artifact_root=artifact_root,
                        artifact_identity_sha256=artifact_identity,
                        dataset_identity=MappingProxyType(dict(dataset_identity)),
                    )
                )
        return tuple(entries)

    def partitioned_case_inventory(self) -> tuple[TransientCaseInventoryEntry, ...]:
        """Return a disjoint multi-artifact case union for one selected run."""
        inventory = self.case_inventory()
        case_ids = tuple(entry.case_id for entry in inventory)
        if not case_ids:
            msg = "Single-model Evaluation requires at least one exact case identity."
            raise ValueError(msg)
        if len(case_ids) != len(set(case_ids)):
            msg = "Single-model Evaluation artifact partitions contain duplicate exact case identities."
            raise ValueError(msg)
        return inventory

    def _frame_provenance(self, frame_name: str) -> dict[str, Any]:
        """Return exact artifact, Dataset, checkpoint, run, and protocol ownership."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        if not isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
            provenance = frame.attrs.get("artifact_provenance")
            dataset_value = provenance.get("dataset") if isinstance(provenance, Mapping) else None
            run_value = provenance.get("run") if isinstance(provenance, Mapping) else None
            artifact_root = frame.attrs.get("artifact_root")
            if (
                not isinstance(provenance, Mapping)
                or not isinstance(dataset_value, Mapping)
                or dataset_value.get("role") not in {"id", "ood"}
                or not isinstance(artifact_root, str)
                or not artifact_root
            ):
                return {}
            run = run_value if isinstance(run_value, Mapping) else {}
            identity = _record_identity(_sequence_records(frame)[0])
            lineage = identity.get("stage_identity")
            identity_value = provenance.get("identity_sha256")
            artifact_identity = (
                identity_value if isinstance(identity_value, str) and identity_value else common.serialization.canonical_json_sha256(provenance)
            )
            return {
                "dataset_role": self.dataset_role(frame_name),
                "dataset_name": dataset_value.get("name"),
                "source_dataset_ids": tuple(dataset_value.get("source_dataset_ids", ())),
                "membership_digests": tuple(dataset_value.get("membership_digests", ())),
                "artifact_root": str(Path(artifact_root).resolve()),
                "artifact_identity_sha256": artifact_identity,
                "checkpoint_sha256": run.get("best_checkpoint_sha256"),
                "checkpoint_epoch": run.get("best_checkpoint_epoch"),
                "run_name": run.get("name"),
                "stage_identity": MappingProxyType(dict(lineage)) if isinstance(lineage, Mapping) else None,
                "evaluation_config_identity": identity.get("evaluation_config_identity"),
            }
        provenance = index.provenance
        dataset_value = provenance.get("dataset") if isinstance(provenance, Mapping) else None
        run_value = provenance.get("run") if isinstance(provenance, Mapping) else None
        dataset = dataset_value if isinstance(dataset_value, Mapping) else {}
        run = run_value if isinstance(run_value, Mapping) else {}
        identity = _record_identity(index.summaries[0])
        lineage = identity.get("stage_identity")
        return {
            "dataset_role": self.dataset_role(frame_name),
            "dataset_name": index.dataset_name,
            "source_dataset_ids": tuple(dataset.get("source_dataset_ids", ())),
            "membership_digests": tuple(dataset.get("membership_digests", ())),
            "artifact_root": str(index.root),
            "artifact_identity_sha256": index.identity_sha256,
            "checkpoint_sha256": run.get("best_checkpoint_sha256"),
            "checkpoint_epoch": run.get("best_checkpoint_epoch"),
            "run_name": run.get("name"),
            "stage_identity": MappingProxyType(dict(lineage)) if isinstance(lineage, Mapping) else None,
            "evaluation_config_identity": identity.get("evaluation_config_identity"),
        }

    def case_ids(
        self,
        frame_name: str,
        *,
        material_family: str | None = None,
    ) -> tuple[str, ...]:
        """Return exact optionally material-filtered case IDs without opening payloads."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        records = _sequence_records(frame)
        if material_family is not None:
            if not isinstance(material_family, str) or not material_family:
                msg = "material_family must be non-empty text or None."
                raise ValueError(msg)
            records = tuple(record for record in records if _material_family(record) == material_family)
        return tuple(dict.fromkeys(str(_record_value(record, "case_id")) for record in records))

    def record_summaries_for_case(
        self,
        frame_name: str,
        case_id: str,
    ) -> tuple[Any, ...]:
        """Return case-local sequence coordinates without opening numerical arrays."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        records = index.summaries if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex) else _sequence_records(frame)
        selected = tuple(record for record in records if _record_value(record, "case_id") == case_id)
        if not selected:
            msg = f"Unknown transient artifact case {case_id!r}."
            raise KeyError(msg)
        return selected

    def record_for_coordinates(
        self,
        frame_name: str,
        case_id: str,
        *,
        mode: str,
        origin_index: int,
        requested_horizon: int | str,
    ) -> Any:
        """Load exactly one sequence selected from its compact case summary."""
        summaries = self.record_summaries_for_case(frame_name, case_id)
        matches = tuple(
            record
            for record in summaries
            if _record_value(record, "mode") == mode
            and _record_value(record, "origin_index") == origin_index
            and _requested_horizon(record) == requested_horizon
        )
        if len(matches) != 1:
            msg = "Transient protocol, origin, and horizon must identify exactly one sequence."
            raise ValueError(msg)
        summary = matches[0]
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
            return index.record(str(_record_value(summary, "record_id")))
        return summary

    def records_for_case(
        self,
        frame_name: str,
        case_id: str,
    ) -> tuple[Any, ...]:
        """Load only one selected case from an indexed transient artifact."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        index = frame.attrs.get("transient_sequence_index")
        if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
            return index.records(case_id=case_id)
        records = tuple(record for record in _sequence_records(frame) if _record_value(record, "case_id") == case_id)
        if not records:
            msg = f"Unknown transient artifact case {case_id!r}."
            raise KeyError(msg)
        return records

    def full_autonomous_summaries(
        self,
        frame_name: str | None = None,
        *,
        material_family: str | None = None,
    ) -> tuple[Any, ...]:
        """Return compact complete-rollout summaries without opening case payloads."""
        frame_names = self.frame_names if frame_name is None else (frame_name,)
        selected: list[Any] = []
        case_keys: set[tuple[str, str]] = set()
        for name in frame_names:
            try:
                frame, _artifact = self._frames[name]
            except KeyError as error:
                msg = f"Unknown transient session frame {name!r}."
                raise KeyError(msg) from error
            for summary in _sequence_records(frame):
                if _record_value(summary, "mode") != "autonomous_full" or _requested_horizon(summary) != "full":
                    continue
                if material_family is not None and _material_family(summary) != material_family:
                    continue
                case_id = _record_value(summary, "case_id")
                if not isinstance(case_id, str) or not case_id:
                    msg = "Full-autonomous transient summaries require non-empty case IDs."
                    raise ValueError(msg)
                key = (name, case_id)
                if key in case_keys:
                    msg = "Each transient case must have exactly one full-autonomous summary."
                    raise ValueError(msg)
                case_keys.add(key)
                selected.append(summary)
        if not selected:
            msg = "No full-autonomous transient summaries satisfy the requested context."
            raise ValueError(msg)
        return tuple(selected)

    def full_autonomous_records(
        self,
        frame_name: str | None = None,
        *,
        material_family: str | None = None,
    ) -> tuple[Any, ...]:
        """Return one optionally material-filtered full-autonomous record per case."""
        frame_names = self.frame_names if frame_name is None else (frame_name,)
        selected: list[Any] = []
        case_keys: set[tuple[str, str]] = set()
        for name in frame_names:
            try:
                frame, _artifact = self._frames[name]
            except KeyError as error:
                msg = f"Unknown transient session frame {name!r}."
                raise KeyError(msg) from error
            index = frame.attrs.get("transient_sequence_index")
            summaries = _sequence_records(frame)
            for summary in summaries:
                if _record_value(summary, "mode") != "autonomous_full" or _requested_horizon(summary) != "full":
                    continue
                if material_family is not None and _material_family(summary) != material_family:
                    continue
                case_id = _record_value(summary, "case_id")
                if not isinstance(case_id, str) or not case_id:
                    msg = "Full-autonomous transient records require non-empty case IDs."
                    raise ValueError(msg)
                key = (name, case_id)
                if key in case_keys:
                    msg = f"Transient frame {name!r} contains duplicate full-autonomous case {case_id!r}."
                    raise ValueError(msg)
                case_keys.add(key)
                selected.append(
                    index.record(str(_record_value(summary, "record_id")))
                    if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex)
                    else summary
                )
        if not selected:
            msg = "Transient target evidence requires full-autonomous, full-horizon records."
            raise ValueError(msg)
        return tuple(selected)

    def scaling_state(self, frame_name: str) -> Mapping[str, Any]:
        """Return the persisted train-only scaling state for one frame."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        return _require_mapping(
            frame.attrs["transient_scaling_state"],
            label="transient_scaling_state",
        )

    def timing_report(self, frame_name: str) -> timing.TransientTimingReport:
        """Return one recomputed timing report from admitted artifact provenance."""
        try:
            frame, _artifact = self._frames[frame_name]
        except KeyError as error:
            msg = f"Unknown transient session frame {frame_name!r}."
            raise KeyError(msg) from error
        provenance = _require_mapping(
            frame.attrs.get("artifact_provenance"),
            label="artifact_provenance",
        )
        evaluation = _require_mapping(
            provenance.get("evaluation"),
            label="artifact_provenance.evaluation",
        )
        return timing.admit_transient_timing_report(
            _require_mapping(
                evaluation.get("timing_report"),
                label="artifact_provenance.evaluation.timing_report",
            )
        )

    def pipeline_degradation(self) -> tuple[comparison.AirflowDegradationMetrics, ...]:
        """Return B/A evidence and truthful available or unavailable C evidence per frame."""
        full_cumulative = {
            summary.frame: summary
            for summary in self._all_material_summaries(modes=("autonomous_full",))
            if summary.scope == "cumulative" and summary.requested_horizon == "full"
        }
        results: list[comparison.AirflowDegradationMetrics] = []
        metric_id = metrics.NORMALIZED_DRYING_GROUP_MACRO_RMSE
        for frame_name, (frame, _artifact) in self._frames.items():
            summary = full_cumulative.get(frame_name)
            if summary is None:
                msg = f"Transient frame {frame_name!r} lacks full autonomous cumulative evidence."
                raise RuntimeError(msg)
            provenance = _require_mapping(frame.attrs.get("artifact_provenance"), label="artifact_provenance")
            evaluation = _require_mapping(provenance.get("evaluation"), label="artifact_provenance.evaluation")
            pipeline = _require_mapping(evaluation.get("pipeline_analysis"), label="evaluation.pipeline_analysis")
            c_available = pipeline.get("c_available")
            if not isinstance(c_available, bool) or pipeline.get("fabricated_prediction_count") != 0:
                msg = "Pipeline provenance requires boolean C availability and zero fabricated predictions."
                raise ValueError(msg)
            upstream = pipeline.get("upstream_airflow_error")
            upstream_error = float(upstream) if isinstance(upstream, Real) and not isinstance(upstream, bool) else None
            if not c_available:
                reasons = pipeline.get("c_unavailable_reasons")
                if not isinstance(reasons, list) or not reasons or any(not isinstance(reason, str) or not reason for reason in reasons):
                    msg = "Unavailable pipeline C requires exact non-empty reasons."
                    raise ValueError(msg)
                results.append(
                    comparison.build_airflow_degradation_metrics(
                        metric_id=metric_id,
                        drying_surrogate_error=summary.metrics.normalized_drying_group_macro_rmse,
                        complete_pipeline_error=None,
                        airflow_substitution_discrepancy=None,
                        upstream_airflow_error=upstream_error,
                        unavailable_reason="; ".join(reasons),
                    )
                )
                continue
            raw_metrics = _require_mapping(pipeline.get("metrics"), label="evaluation.pipeline_analysis.metrics")
            raw_metric = _require_mapping(raw_metrics.get(metric_id), label=f"pipeline metric {metric_id}")
            results.append(
                comparison.build_airflow_degradation_metrics(
                    metric_id=metric_id,
                    drying_surrogate_error=summary.metrics.normalized_drying_group_macro_rmse,
                    complete_pipeline_error=_required_pipeline_metric(raw_metric, "complete_pipeline_error"),
                    airflow_substitution_discrepancy=_required_pipeline_metric(
                        raw_metric,
                        "airflow_substitution_discrepancy",
                    ),
                    upstream_airflow_error=upstream_error,
                )
            )
        return tuple(results)

    def close(self) -> None:
        """Release frame, scaling, payload, and derived-table references."""
        for frame, _artifact in self._frames.values():
            index = frame.attrs.get("transient_sequence_index")
            if isinstance(index, sequence_artifact.TransientSequenceArtifactIndex):
                index.close()
        self._frames = MappingProxyType({})
        self._case_dataframe_cache.clear()
        self._summary_cache.clear()
        self._dataset_dataframe_cache.clear()

    @property
    def cache_sizes(self) -> Mapping[str, int]:
        """Return bounded immutable cache counts for notebook diagnostics."""
        return MappingProxyType(
            {
                "case_dataframes": len(self._case_dataframe_cache),
                "summaries": len(self._summary_cache),
                "dataset_dataframes": len(self._dataset_dataframe_cache),
                "payload_records": sum(
                    index.cache_size
                    for frame, _artifact in self._frames.values()
                    if isinstance(
                        index := frame.attrs.get("transient_sequence_index"),
                        sequence_artifact.TransientSequenceArtifactIndex,
                    )
                ),
            }
        )

    def case_dataframe(self, *, modes: Sequence[str] = _DEFAULT_MODES) -> pd.DataFrame:
        """Return cached cumulative and endpoint rows for every available sequence."""
        selected = _validate_modes(modes)
        cached = self._case_dataframe_cache.get(selected)
        if cached is not None:
            return cached.copy(deep=True)
        rows: list[dict[str, Any]] = []
        for frame_name, (frame, artifact) in self._frames.items():
            for record in _sequence_records(frame):
                mode = _record_value(record, "mode")
                if mode not in selected:
                    continue
                horizon = _requested_horizon(record)
                for scope in ("cumulative", "endpoint"):
                    summary = _summarize_group(
                        (record,),
                        frame=frame_name,
                        material_family=_material_family(record),
                        artifact=artifact,
                        mode=mode,
                        horizon=horizon,
                        scope=scope,
                        unavailable_case_count=0,
                    )
                    row = _summary_row(summary)
                    row.update(self._frame_provenance(frame_name))
                    row.update(
                        {
                            "case_id": _record_value(record, "case_id"),
                            "origin_index": _record_value(record, "origin_index"),
                        }
                    )
                    rows.append(row)
        result = pd.DataFrame.from_records(rows)
        self._case_dataframe_cache[selected] = result
        return result.copy(deep=True)

    def _reduce_summaries(
        self,
        *,
        modes: tuple[str, ...],
        group_by_material: bool,
    ) -> tuple[TransientSessionSummary, ...]:
        """Reduce exact record groups either per material or over full frame membership."""
        results: list[TransientSessionSummary] = []
        for frame_name, (frame, artifact) in self._frames.items():
            records = _sequence_records(frame)
            groups: dict[tuple[str | None, str, int | str], list[Any]] = {}
            for record in records:
                mode = _record_value(record, "mode")
                if mode in modes:
                    material = _material_family(record) if group_by_material else None
                    groups.setdefault((material, mode, _requested_horizon(record)), []).append(record)
            unavailable = frame.attrs["transient_unavailable_horizons"]
            if not isinstance(unavailable, tuple):
                msg = "transient_unavailable_horizons must be one tuple."
                raise TypeError(msg)
            for (material, mode, horizon), grouped in groups.items():
                case_ids = {str(_record_value(record, "case_id")) for record in grouped}
                unavailable_cases = _unavailable_cases(
                    unavailable,
                    mode=mode,
                    horizon=horizon,
                    case_ids=case_ids,
                )
                results.extend(
                    _summarize_group(
                        grouped,
                        frame=frame_name,
                        material_family=material,
                        artifact=artifact,
                        mode=mode,
                        horizon=horizon,
                        scope=scope,
                        unavailable_case_count=unavailable_cases,
                    )
                    for scope in ("cumulative", "endpoint")
                )
        return tuple(results)

    def _all_material_summaries(
        self,
        *,
        modes: Sequence[str] = _DEFAULT_MODES,
    ) -> tuple[TransientSessionSummary, ...]:
        """Return full-frame reductions reserved for frame-level provenance consumers."""
        return self._reduce_summaries(
            modes=_validate_modes(modes),
            group_by_material=False,
        )

    def summaries(self, *, modes: Sequence[str] = _DEFAULT_MODES) -> tuple[TransientSessionSummary, ...]:
        """Return cached reductions for each material, mode, and horizon group."""
        selected = _validate_modes(modes)
        cached = self._summary_cache.get(selected)
        if cached is not None:
            return cached
        resolved = self._reduce_summaries(
            modes=selected,
            group_by_material=True,
        )
        self._summary_cache[selected] = resolved
        return resolved

    def dataset_dataframe(self, *, modes: Sequence[str] = _DEFAULT_MODES) -> pd.DataFrame:
        """Return one cached scalar dataset summary table for plotting."""
        selected = _validate_modes(modes)
        cached = self._dataset_dataframe_cache.get(selected)
        if cached is not None:
            return cached.copy(deep=True)
        result = pd.DataFrame.from_records([_summary_row(summary) for summary in self.summaries(modes=selected)])
        if not result.empty:
            provenance_by_frame = {frame_name: self._frame_provenance(frame_name) for frame_name in self.frame_names}
            for column in next(iter(provenance_by_frame.values())):
                result[column] = result["frame"].map(lambda frame_name, key=column: provenance_by_frame[str(frame_name)][key])
        self._dataset_dataframe_cache[selected] = result
        return result.copy(deep=True)

    def wandb_summary(self, *, modes: Sequence[str] = _DEFAULT_MODES) -> dict[str, float | int]:
        """Return bounded aggregate tracking evidence without arrays or W&B imports."""
        selected_modes = _validate_modes(modes)
        values: dict[str, float | int] = {}
        for index, summary in enumerate(self._all_material_summaries(modes=selected_modes)):
            prefix = f"evaluation/{index}/{summary.frame}/{summary.mode}/{summary.requested_horizon}/{summary.scope}"
            values[f"{prefix}/normalized_drying_group_macro_rmse"] = summary.metrics.normalized_drying_group_macro_rmse
            for field, value in summary.metrics.normalized_rmse.items():
                values[f"{prefix}/normalized_rmse_{field}"] = value
            values[f"{prefix}/contributing_record_count"] = summary.contributing_record_count
            values[f"{prefix}/contributing_case_count"] = summary.contributing_case_count
            values[f"{prefix}/unavailable_case_count"] = summary.unavailable_case_count
            values[f"{prefix}/elapsed_physical_time_median"] = summary.elapsed_physical_time_median
            if summary.mode == "autonomous_full" and summary.scope == "cumulative":
                for field, value in summary.metrics.physical_rmse.items():
                    values[f"{prefix}/physical_rmse_{field}"] = value
                for field, value in summary.metrics.physical_mae.items():
                    values[f"{prefix}/physical_mae_{field}"] = value
                values[f"{prefix}/physical_w_gr_rmse"] = summary.metrics.physical_w_gr_rmse
                values[f"{prefix}/physical_w_gr_mae"] = summary.metrics.physical_w_gr_mae
                _add_optional_tracking_value(
                    values,
                    f"{prefix}/bulk_dry_basis_rmse",
                    summary.metrics.bulk_dry_basis_rmse,
                )
                _add_optional_tracking_value(
                    values,
                    f"{prefix}/bulk_wet_basis_rmse",
                    summary.metrics.bulk_wet_basis_rmse,
                )
                values[f"{prefix}/bulk_moisture_valid_count"] = summary.metrics.bulk_moisture_valid_count
                values[f"{prefix}/bulk_moisture_unavailable_count"] = summary.metrics.bulk_moisture_unavailable_count
                values[f"{prefix}/target_available_count"] = summary.target.available_count
                values[f"{prefix}/predicted_right_censored_count"] = summary.target.predicted_right_censored_count
                values[f"{prefix}/target_time_error_count"] = summary.target_time_error_count
                if summary.target_time_error_mean is not None:
                    values[f"{prefix}/target_time_error_mean"] = summary.target_time_error_mean
                if summary.target_time_error_mae is not None:
                    values[f"{prefix}/target_time_error_mae"] = summary.target_time_error_mae
                values[f"{prefix}/target_gap_count"] = summary.target_gap_count
                if summary.predicted_final_target_gap_mean is not None:
                    values[f"{prefix}/predicted_final_target_gap_mean"] = summary.predicted_final_target_gap_mean
                if summary.reference_final_target_gap_mean is not None:
                    values[f"{prefix}/reference_final_target_gap_mean"] = summary.reference_final_target_gap_mean
                if summary.target_final_gap_error_mean is not None:
                    values[f"{prefix}/target_final_gap_error_mean"] = summary.target_final_gap_error_mean
                if summary.target_final_gap_error_mae is not None:
                    values[f"{prefix}/target_final_gap_error_mae"] = summary.target_final_gap_error_mae
                values[f"{prefix}/nonfinite_values"] = summary.plausibility.nonfinite_values
                values[f"{prefix}/negative_moisture_values"] = summary.plausibility.negative_moisture_values
                values[f"{prefix}/abnormal_increment_growth_count"] = summary.stability.abnormal_growth_count
        if "autonomous_full" in selected_modes:
            for index, pipeline in enumerate(self.pipeline_degradation()):
                prefix = f"evaluation/pipeline/{index}"
                values[f"{prefix}/drying_surrogate_error"] = pipeline.drying_surrogate_error
                values[f"{prefix}/c_available"] = int(pipeline.c_available)
                if pipeline.complete_pipeline_error is not None:
                    values[f"{prefix}/complete_pipeline_error"] = pipeline.complete_pipeline_error
                if pipeline.airflow_substitution_discrepancy is not None:
                    values[f"{prefix}/airflow_substitution_discrepancy"] = pipeline.airflow_substitution_discrepancy
                if pipeline.signed_airflow_degradation is not None:
                    values[f"{prefix}/signed_airflow_degradation"] = pipeline.signed_airflow_degradation
                if pipeline.airflow_degradation_ratio is not None:
                    values[f"{prefix}/airflow_degradation_ratio"] = pipeline.airflow_degradation_ratio
                if pipeline.upstream_airflow_error is not None:
                    values[f"{prefix}/upstream_airflow_error"] = pipeline.upstream_airflow_error
        if len(values) > _MAX_TRACKING_SUMMARY_ENTRIES:
            msg = "Transient Evaluation tracking summary exceeds its bounded scalar inventory."
            raise RuntimeError(msg)
        return values

    def wandb_publication_summary(self) -> dict[str, bool | float | int | str]:
        """Return bounded aggregate and identity evidence for opt-in publication."""
        values: dict[str, bool | float | int | str] = dict(self.wandb_summary())
        metadata = self._wandb_publication_metadata()
        overlap = set(values).intersection(metadata)
        if overlap:
            msg = f"Transient Evaluation publication summary contains duplicate keys: {sorted(overlap)}."
            raise RuntimeError(msg)
        if len(metadata) > _MAX_PUBLICATION_METADATA_ENTRIES:
            msg = "Transient Evaluation publication metadata exceeds its bounded inventory."
            raise RuntimeError(msg)
        values.update(metadata)
        if len(values) > _MAX_PUBLICATION_SUMMARY_ENTRIES:
            msg = "Transient Evaluation publication summary exceeds the W&B scalar inventory."
            raise RuntimeError(msg)
        return values

    def _wandb_publication_metadata(self) -> dict[str, bool | float | int | str]:
        """Reduce validated role provenance and raw clocks to bounded identities."""
        if "id" not in self._frames or set(self._frames).difference({"id", "ood"}):
            msg = "Transient W&B publication requires an ID frame and at most one OOD frame."
            raise ValueError(msg)

        run_evidence: list[Mapping[str, Any]] = []
        lineage_evidence: list[Mapping[str, Any]] = []
        record_identities: list[Mapping[str, Any]] = []
        timing_cases: list[timing.TransientTimingCase] = []
        materialization_seconds: list[float] = []
        hardware_evidence: list[dict[str, Any]] = []
        dataset_evidence: dict[str, Mapping[str, Any]] = {}
        metadata: dict[str, bool | float | int | str] = {}

        for frame_name, (frame, _artifact) in self._frames.items():
            provenance = _require_mapping(frame.attrs.get("artifact_provenance"), label="artifact_provenance")
            run = _require_mapping(provenance.get("run"), label="artifact_provenance.run")
            dataset = _require_mapping(provenance.get("dataset"), label="artifact_provenance.dataset")
            evaluation = _require_mapping(provenance.get("evaluation"), label="artifact_provenance.evaluation")
            lineage = _require_mapping(provenance.get("lineage"), label="artifact_provenance.lineage")
            dataset_name = dataset.get("name")
            if dataset.get("role") != frame_name or not isinstance(dataset_name, str) or not dataset_name:
                msg = f"Transient publication frame {frame_name!r} contradicts its Dataset identity."
                raise ValueError(msg)
            metadata[f"evaluation/identity/dataset/{frame_name}/name"] = dataset_name
            dataset_evidence[frame_name] = dataset
            run_evidence.append(run)
            lineage_evidence.append(lineage)

            full_records = self.full_autonomous_records(frame_name)
            case_ids = {_record_value(record, "case_id") for record in full_records}
            if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
                msg = "Transient publication requires non-empty full-autonomous case identities."
                raise ValueError(msg)
            identities = [_record_identity(record) for record in full_records]
            record_identities.extend(identities)

            report = self.timing_report(frame_name)
            if {case.case_id for case in report.cases} != case_ids:
                msg = f"Transient timing report for {frame_name!r} does not cover exact target-case membership."
                raise ValueError(msg)
            for case in report.cases:
                timing_cases.append(replace(case, case_id=f"{frame_name}:{case.case_id}"))
                hardware_evidence.append(
                    {
                        "device": case.device,
                        "cpu": case.cpu,
                        "gpu": case.gpu,
                        "precision": case.precision,
                        "dataset_backend": case.dataset_backend,
                        "batch_size": case.batch_size,
                        "software_versions": dict(case.software_versions),
                    }
                )

            component_evidence = _require_mapping(
                evaluation.get("component_availability"),
                label="artifact_provenance.evaluation.component_availability",
            )
            if set(component_evidence) != case_ids:
                msg = f"Transient component timing for {frame_name!r} does not cover exact target-case membership."
                raise ValueError(msg)
            for case_id in sorted(case_ids):
                case_timing = _require_mapping(
                    component_evidence[case_id],
                    label=f"component_availability.{case_id}",
                )
                materialization_seconds.append(
                    _nonnegative_tracking_duration(
                        case_timing.get("dataset_materialization_seconds"),
                        label=f"component_availability.{case_id}.dataset_materialization_seconds",
                    )
                )

        run_digests = {common.serialization.canonical_json_sha256(value) for value in run_evidence}
        lineage_digests = {common.serialization.canonical_json_sha256(value) for value in lineage_evidence}
        if len(run_digests) != 1 or len(lineage_digests) != 1:
            msg = "Transient ID/OOD publication evidence must share one run and Training lineage."
            raise ValueError(msg)
        run = run_evidence[0]
        checkpoint_epoch = run.get("best_checkpoint_epoch")
        if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 0:
            msg = "Transient publication checkpoint epoch must be a non-negative integer."
            raise TypeError(msg)
        metadata["evaluation/identity/checkpoint_sha256"] = _tracking_text(
            [run.get("best_checkpoint_sha256")],
            label="checkpoint identity",
        )
        metadata["evaluation/identity/checkpoint_epoch"] = checkpoint_epoch
        metadata["evaluation/identity/input_profile"] = _tracking_text(
            [identity.get("input_profile") for identity in record_identities],
            label="input-profile identity",
        )
        metadata["evaluation/identity/model_kind"] = _tracking_text(
            [identity.get("model_kind") for identity in record_identities],
            label="model identity",
        )
        metadata["evaluation/identity/backend"] = _tracking_text(
            [identity.get("dataset_backend") for identity in record_identities],
            label="Dataset backend identity",
        )
        metadata["evaluation/identity/timing_evidence_sha256"] = common.serialization.canonical_json_sha256(
            sorted(_tracking_text([identity.get("timing_evidence_identity")], label="timing evidence identity") for identity in record_identities)
        )
        metadata["evaluation/identity/dataset_sha256"] = common.serialization.canonical_json_sha256(dataset_evidence)

        timing_backends = [case.dataset_backend for case in timing_cases]
        record_backends = [identity.get("dataset_backend") for identity in record_identities]
        if _tracking_text(timing_backends, label="timing backend identity") != _tracking_text(
            record_backends,
            label="record backend identity",
        ):
            msg = "Transient timing and sequence Dataset backend identities disagree."
            raise ValueError(msg)
        _add_publication_timing_metadata(
            metadata,
            timing_cases=timing_cases,
            materialization_seconds=materialization_seconds,
            hardware_evidence=hardware_evidence,
        )
        return metadata


def _validate_modes(modes: Sequence[str]) -> tuple[str, ...]:
    """Return unique declared default or caller-selected evaluation modes."""
    result = tuple(modes)
    if not result or any(mode not in _DEFAULT_MODES for mode in result) or len(set(result)) != len(result):
        msg = "modes must be unique supported transient evaluation modes."
        raise ValueError(msg)
    return result


def _unavailable_cases(
    unavailable: Sequence[Any],
    *,
    mode: str,
    horizon: int | str,
    case_ids: set[str],
) -> int:
    """Count group-local cases with explicit unsupported fixed rolling horizons."""
    if mode != "rolling_origin" or not isinstance(horizon, int):
        return 0
    cases: set[str] = set()
    for item in unavailable:
        mapping = _require_mapping(item, label="transient unavailable horizon")
        if mapping.get("requested_horizon") == horizon:
            case_id = mapping.get("case_id")
            if isinstance(case_id, str) and case_id in case_ids:
                cases.add(case_id)
    return len(cases)


def _summarize_group(
    records: Sequence[Any],
    *,
    frame: str,
    material_family: str | None,
    artifact: scaling.TransientScalingArtifact,
    mode: str,
    horizon: int | str,
    scope: str,
    unavailable_case_count: int,
) -> TransientSessionSummary:
    """Accumulate one exact material/mode/horizon/scope sufficient-statistic group."""
    materials = {_material_family(record) for record in records}
    if material_family is not None and materials != {material_family}:
        msg = "Transient summary group mixes material identities."
        raise ValueError(msg)
    if records and isinstance(
        records[0],
        sequence_artifact.TransientSequenceRecordSummary,
    ):
        summaries = cast(
            "Sequence[sequence_artifact.TransientSequenceRecordSummary]",
            records,
        )
        states: list[Mapping[str, object]] = []
        plausibility = _empty_plausibility()
        stability = _empty_stability()
        for record in summaries:
            statistics = record.metric_statistics
            if statistics is None:
                message = "Indexed transient summaries require persisted metric statistics."
                raise ValueError(message)
            states.append(statistics[scope])
            current_plausibility, current_stability = _persisted_diagnostics(statistics["diagnostics"])
            plausibility = metrics.PlausibilityDiagnostics(
                plausibility.inspected_values + current_plausibility.inspected_values,
                plausibility.nonfinite_values + current_plausibility.nonfinite_values,
                plausibility.negative_moisture_values + current_plausibility.negative_moisture_values,
                plausibility.relative_humidity_bound_violations + current_plausibility.relative_humidity_bound_violations,
                plausibility.temperature_range_violations + current_plausibility.temperature_range_violations,
            )
            stability = metrics.StabilityDiagnostics(
                stability.increment_count + current_stability.increment_count,
                stability.nonfinite_increment_count + current_stability.nonfinite_increment_count,
                stability.oscillatory_increment_count + current_stability.oscillatory_increment_count,
                stability.abnormal_growth_count + current_stability.abnormal_growth_count,
            )
        accumulator = metrics.TransientMetricAccumulator.merged(states)
        (
            predicted,
            reference,
            target_errors,
            predicted_gaps,
            reference_gaps,
            gap_errors,
        ) = _target_values(summaries)
        target = metrics.derive_target_censoring_diagnostics(
            predicted_reached=predicted,
            reference_reached=reference,
        )
        error_array = np.asarray(target_errors, dtype=np.float64)
        predicted_gap_array = np.asarray(predicted_gaps, dtype=np.float64)
        reference_gap_array = np.asarray(reference_gaps, dtype=np.float64)
        gap_error_array = np.asarray(gap_errors, dtype=np.float64)
        elapsed_values = np.asarray(
            [record.elapsed_physical_time for record in summaries],
            dtype=np.float64,
        )
        origin_values = np.asarray(
            [record.origin_time for record in summaries],
            dtype=np.float64,
        )
        return TransientSessionSummary(
            frame=frame,
            material_family=material_family,
            mode=mode,
            requested_horizon=horizon,
            scope=scope,
            metrics=accumulator.finalize(),
            contributing_record_count=len(summaries),
            contributing_case_count=len({record.case_id for record in summaries}),
            unavailable_case_count=unavailable_case_count,
            elapsed_physical_time_min=float(elapsed_values.min()),
            elapsed_physical_time_median=float(np.median(elapsed_values)),
            elapsed_physical_time_max=float(elapsed_values.max()),
            origin_time_min=float(origin_values.min()),
            origin_time_max=float(origin_values.max()),
            target=target,
            target_time_error_count=int(error_array.size),
            target_time_error_mean=(float(np.mean(error_array)) if error_array.size else None),
            target_time_error_mae=(float(np.mean(np.abs(error_array))) if error_array.size else None),
            target_gap_count=int(gap_error_array.size),
            predicted_final_target_gap_mean=(float(np.mean(predicted_gap_array)) if predicted_gap_array.size else None),
            reference_final_target_gap_mean=(float(np.mean(reference_gap_array)) if reference_gap_array.size else None),
            target_final_gap_error_mean=(float(np.mean(gap_error_array)) if gap_error_array.size else None),
            target_final_gap_error_mae=(float(np.mean(np.abs(gap_error_array))) if gap_error_array.size else None),
            plausibility=plausibility,
            stability=stability,
        )
    accumulator = metrics.TransientMetricAccumulator(scope=scope)
    elapsed: list[float] = []
    origins: list[float] = []
    plausibility = _empty_plausibility()
    stability = _empty_stability()
    for record in records:
        prediction = np.asarray(_record_value(record, "predicted_states"), dtype=np.float32)
        reference = np.asarray(_record_value(record, "reference_states"), dtype=np.float32)
        if prediction.shape != reference.shape or prediction.shape[0] < _ORIGIN_PLUS_TRANSITION_COUNT:
            msg = "Transient records require matching origin-plus-transition state arrays."
            raise ValueError(msg)
        selected_prediction = prediction[1:] if scope == "cumulative" else prediction[-1:]
        selected_reference = reference[1:] if scope == "cumulative" else reference[-1:]
        normalized_prediction = _normalized_states(artifact, selected_prediction)
        normalized_reference = _normalized_states(artifact, selected_reference)
        scalar = np.asarray(_record_value(record, "scalar_conditioning"), dtype=np.float64)
        if scalar.shape != (len(sequence_artifact.SCALAR_ORDER),):
            msg = "Transient record scalar_conditioning must retain the canonical Dataset scalar fields."
            raise ValueError(msg)
        dry_density, cell_weights = _bulk_moisture_inputs(record)
        accumulator.update(
            normalized_prediction=normalized_prediction,
            normalized_reference=normalized_reference,
            physical_prediction=selected_prediction,
            physical_reference=selected_reference,
            f_surf=np.asarray(scalar[2]),
            rho_bu_dry=dry_density,
            cell_weights=cell_weights,
            valid_mask=_spatial_mask(record, count=selected_prediction.shape[0]),
        )
        times = np.asarray(_record_value(record, "physical_times"), dtype=np.float64)
        elapsed.append(float(times[-1] - times[0]))
        origins.append(float(times[0]))
        current_plausibility = metrics.derive_plausibility_diagnostics(
            prediction[1:],
            temperature_range=metrics.TEMPERATURE_PLAUSIBILITY_RANGE_K,
        )
        plausibility = metrics.PlausibilityDiagnostics(
            plausibility.inspected_values + current_plausibility.inspected_values,
            plausibility.nonfinite_values + current_plausibility.nonfinite_values,
            plausibility.negative_moisture_values + current_plausibility.negative_moisture_values,
            plausibility.relative_humidity_bound_violations + current_plausibility.relative_humidity_bound_violations,
            plausibility.temperature_range_violations + current_plausibility.temperature_range_violations,
        )
        current_stability = metrics.derive_stability_diagnostics(prediction)
        stability = metrics.StabilityDiagnostics(
            stability.increment_count + current_stability.increment_count,
            stability.nonfinite_increment_count + current_stability.nonfinite_increment_count,
            stability.oscillatory_increment_count + current_stability.oscillatory_increment_count,
            stability.abnormal_growth_count + current_stability.abnormal_growth_count,
        )
    predicted, reference, target_errors, predicted_gaps, reference_gaps, gap_errors = _target_values(records)
    target = metrics.derive_target_censoring_diagnostics(predicted_reached=predicted, reference_reached=reference)
    error_array = np.asarray(target_errors, dtype=np.float64)
    predicted_gap_array = np.asarray(predicted_gaps, dtype=np.float64)
    reference_gap_array = np.asarray(reference_gaps, dtype=np.float64)
    gap_error_array = np.asarray(gap_errors, dtype=np.float64)
    return TransientSessionSummary(
        frame=frame,
        material_family=material_family,
        mode=mode,
        requested_horizon=horizon,
        scope=scope,
        metrics=accumulator.finalize(),
        contributing_record_count=len(records),
        contributing_case_count=len({_record_value(record, "case_id") for record in records}),
        unavailable_case_count=unavailable_case_count,
        elapsed_physical_time_min=float(np.min(elapsed)),
        elapsed_physical_time_median=float(np.median(elapsed)),
        elapsed_physical_time_max=float(np.max(elapsed)),
        origin_time_min=float(np.min(origins)),
        origin_time_max=float(np.max(origins)),
        target=target,
        target_time_error_count=int(error_array.size),
        target_time_error_mean=float(np.mean(error_array)) if error_array.size else None,
        target_time_error_mae=float(np.mean(np.abs(error_array))) if error_array.size else None,
        target_gap_count=int(gap_error_array.size),
        predicted_final_target_gap_mean=float(np.mean(predicted_gap_array)) if predicted_gap_array.size else None,
        reference_final_target_gap_mean=float(np.mean(reference_gap_array)) if reference_gap_array.size else None,
        target_final_gap_error_mean=float(np.mean(gap_error_array)) if gap_error_array.size else None,
        target_final_gap_error_mae=float(np.mean(np.abs(gap_error_array))) if gap_error_array.size else None,
        plausibility=plausibility,
        stability=stability,
    )
