"""Summarize validated transient sequence artifacts without refitting state.

Responsibilities:
  - Admit transient-sequence frames and their persisted scaling evidence
  - Refuse frames that do not record one common valid analysis producer
  - Aggregate complete-dataset endpoint and cumulative error statistics
  - Rank frames by the selection objective of their complete autonomous rollouts
  - Reduce the relative norms of complete rollouts per physical time, or per drying-phase point of ``time_ui.phase_cohort``,
    with their supported case counts, and classify each full rollout's reference completion from its persisted target evidence
  - Expose bounded case and dataset views
  - Resolve the series styles of the comparison once over all loaded frames, for every transient view

Design principles:
  - Persisted scaling is reconstructed exactly and never refitted
  - Origin states are excluded from every error reduction over hours; a drying-phase point may resolve to the origin
  - Fixed horizons remain exact semantic values, not display labels
  - A group with any non-finite prediction value reports support counts instead of error metrics
  - The objective and the per-field RMSE, MAE and relative norms of every table, ranking, and B/A pipeline error come from
    learning.metrics.metrics and weigh the cases of a material and the materials equally, as checkpoint selection does;
    a horizon relative norm sums each case's energies over its states first; grain-water and bulk errors follow the
    same case-equal rule

This module does NOT:
  - Load files, run inference, mutate frames, or publish artifacts
  - Average finalized RMSE values across cases or batches
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd

from src import common, datasets, domain, learning
from src.analysis.artifacts import contracts
from src.analysis.ui import analysis_ui_time as time_ui

from . import evaluation_prediction_validity as validity_contract
from . import evaluation_transient_artifact as sequence_artifact
from . import evaluation_transient_comparison as comparison
from . import evaluation_transient_metrics as metrics
from . import evaluation_transient_timing as timing
from .evaluation_plot import evaluation_plot_series as series

_DEFAULT_MODES = ("teacher_forced_one_step", "autonomous_full", "rolling_origin")
_FIELD_METRIC_IDS = tuple(spec.id for spec in domain.tasks.spec.field_metric_specs(sequence_artifact.STATE_ORDER))
_RELATIVE_METRIC_IDS = tuple(metric_id for metric_id in _FIELD_METRIC_IDS if metric_id.startswith("physical_relative_"))
_NORMALIZED_COLUMNS = (metrics.NORMALIZED_DRYING_GROUP_MACRO_RMSE, *_FIELD_METRIC_IDS)
_SUMMARY_METRIC_COLUMNS = (
    "physical_w_gr_rmse",
    "physical_w_gr_mae",
    "bulk_dry_basis_rmse",
    "bulk_dry_basis_mae",
    "bulk_wet_basis_rmse",
    "bulk_wet_basis_mae",
    "predicted_bulk_dry_basis_mean",
    "reference_bulk_dry_basis_mean",
    "predicted_bulk_wet_basis_mean",
    "reference_bulk_wet_basis_mean",
    "bulk_moisture_valid_count",
    "bulk_moisture_unavailable_count",
)
_ORIGIN_PLUS_TRANSITION_COUNT = 2
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
    material_family: str
    mode: str
    requested_horizon: int | str
    scope: str
    # ``None`` when any contributing value is non-finite: never reduced over the finite subset.
    metrics: _TransientMetricSummary | None
    # Physical field statistics of every case, summed over its records and states; empty with ``metrics`` ``None``.
    case_statistics: tuple[learning.metrics.metrics.CaseStatistics, ...]
    # The drying objective and every field metric id over ``case_statistics``, each case weighing equally; an unavailable
    # relative norm is ``None``. ``None`` with ``metrics``.
    field_metrics: Mapping[str, float | None] | None
    # Cases below the reference floor of every relative metric id; ``None`` with ``metrics``.
    relative_unavailable_cases: Mapping[str, int] | None
    metric_support: Mapping[str, Any]
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
    dataset_role: sequence_artifact.DatasetRole
    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    membership_digests: tuple[str, ...]
    artifact_root: Path
    artifact_identity_sha256: str
    dataset_identity: Mapping[str, Any]
    evaluation_assignment: Mapping[str, Any] | None = None


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


def _scale(frame: pd.DataFrame) -> datasets.preprocessing.normalization.TrainScaling:
    """Reconstruct one persisted train-only scale artifact on CPU."""
    state = _require_mapping(frame.attrs["transient_scaling_state"], label="transient_scaling_state")
    return datasets.preprocessing.normalization.TrainScaling.from_state_dict(state)


def _requested_horizon(record: Any) -> int | str:
    """Return one preserved fixed or full horizon value."""
    value = _record_value(record, "requested_horizon")
    if value == "full":
        return "full"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = "Transient record requested_horizon must be positive integer or 'full'."
        raise ValueError(msg)
    return value


def _static_conditioning(record: Any, *, grid_shape: tuple[int, ...]) -> np.ndarray:
    """Return the record's finite canonical static fields on its state grid."""
    static = np.asarray(_record_value(record, "static_conditioning"), dtype=np.float64)
    if static.shape != (len(sequence_artifact.STATIC_ORDER), *grid_shape) or not np.isfinite(static).all():
        msg = "Transient record static_conditioning must match the canonical finite Dataset fields."
        raise ValueError(msg)
    return static


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


def reference_completion(record: Any) -> time_ui.Completion:
    """Return a full rollout's reference completion from its persisted target evidence: reached, censored or unknown."""
    target = _require_mapping(_record_value(record, "target"), label="record target")
    return time_ui.completion_status(available=target.get("reference_available") is True, reached=target.get("reference_reached"))


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


def _summary_row(summary: TransientSessionSummary) -> dict[str, Any]:
    """Flatten one summary into bounded scalar columns; unavailable error metrics keep only their support counts."""
    row: dict[str, Any] = {
        "frame": summary.frame,
        "material_family": summary.material_family,
        "mode": summary.mode,
        "requested_horizon": summary.requested_horizon,
        "scope": summary.scope,
        **{name: summary.metric_support[name] for name in validity_contract.SUPPORT_COUNT_FIELDS},
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
    for name in _NORMALIZED_COLUMNS:
        row[name] = None if summary.field_metrics is None else summary.field_metrics[name]
    for name in _RELATIVE_METRIC_IDS:
        row[f"{name}_unavailable_cases"] = None if summary.relative_unavailable_cases is None else summary.relative_unavailable_cases[name]
    result = None if summary.metrics is None else asdict(summary.metrics)
    for name in _SUMMARY_METRIC_COLUMNS:
        row[name] = None if result is None else result[name]
    return row


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
        admitted: dict[str, tuple[pd.DataFrame, datasets.preprocessing.normalization.TrainScaling]] = {}
        for name, frame in frames.items():
            if not isinstance(name, str) or not name:
                msg = "Transient session frame names must be non-empty text."
                raise ValueError(msg)
            _validate_sequence_source(frame)
            admitted[name] = (frame, _scale(frame))
        # Every session view compares its frames, so they must come from one recorded analysis implementation.
        contracts.require_common_analysis_producer({name: frame.attrs.get("artifact_provenance") for name, frame in frames.items()})
        self._frames = MappingProxyType(admitted)
        self._case_dataframe_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._summary_cache: dict[tuple[str, ...], tuple[TransientSessionSummary, ...]] = {}
        self._dataset_dataframe_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._coordinate_records: dict[tuple[str, str], tuple[sequence_artifact.TransientSequenceArtifactIndex, Any]] = {}

    @property
    def frame_names(self) -> tuple[str, ...]:
        """Return stable named frame order."""
        return tuple(self._frames)

    def require_common_comparison_grids(self) -> None:
        """Compare every matched case's persisted exact grid without loading arrays."""
        baseline: dict[tuple[str, str], Mapping[str, Any]] | None = None
        baseline_name = ""
        for name, (frame, _artifact) in self._frames.items():
            grids: dict[tuple[str, str], Mapping[str, Any]] = {}
            for record in _sequence_records(frame):
                key = (str(_record_value(record, "dataset_role")), str(_record_value(record, "case_id")))
                spatial = _record_identity(record).get("spatial_representation")
                if not isinstance(spatial, Mapping) or not spatial.get("grid_identity_sha256"):
                    msg = f"Transient model {name!r} case {key!r} lacks exact spatial identity."
                    raise ValueError(msg)
                if key in grids and grids[key] != spatial:
                    msg = f"Transient model {name!r} changes Evaluation grid within case {key!r}."
                    raise ValueError(msg)
                grids[key] = spatial
            if baseline is None:
                baseline, baseline_name = grids, name
                continue
            if grids.keys() != baseline.keys():
                msg = f"Transient models {baseline_name!r} and {name!r} have different exact case memberships."
                raise ValueError(msg)
            for key, spatial in grids.items():
                if spatial != baseline[key]:
                    msg = f"Transient models {baseline_name!r} and {name!r} have incompatible Evaluation grids for case {key!r}."
                    raise ValueError(msg)

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

    def dataset_role(self, frame_name: str) -> sequence_artifact.DatasetRole:
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
        if role in sequence_artifact.DATASET_ROLES:
            return cast("sequence_artifact.DatasetRole", role)
        roles = {_record_value(record, "dataset_role") for record in _sequence_records(frame)}
        if len(roles) == 1 and (only := next(iter(roles))) in sequence_artifact.DATASET_ROLES:
            return cast("sequence_artifact.DatasetRole", only)
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
                        evaluation_assignment=(
                            MappingProxyType(dict(identity["evaluation_assignment"]))
                            if isinstance(identity.get("evaluation_assignment"), Mapping)
                            else None
                        ),
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
                or dataset_value.get("role") not in sequence_artifact.DATASET_ROLES
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
                "evaluation_assignments": tuple(dataset_value.get("evaluation_assignments", ())),
                "artifact_root": str(Path(artifact_root).resolve()),
                "artifact_identity_sha256": artifact_identity,
                "checkpoint_sha256": run.get("best_checkpoint_sha256"),
                "checkpoint_epoch": run.get("best_checkpoint_epoch"),
                "run_name": run.get("name"),
                "model_kind": identity.get("model_kind"),
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
            "evaluation_assignments": tuple(dataset.get("evaluation_assignments", ())),
            "artifact_root": str(index.root),
            "artifact_identity_sha256": index.identity_sha256,
            "checkpoint_sha256": run.get("best_checkpoint_sha256"),
            "checkpoint_epoch": run.get("best_checkpoint_epoch"),
            "run_name": run.get("name"),
            "model_kind": identity.get("model_kind"),
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
            return self._shared_record(frame_name, index, str(_record_value(summary, "record_id")))
        return summary

    def _shared_record(self, frame_name: str, index: sequence_artifact.TransientSequenceArtifactIndex, record_id: str) -> Any:
        """Return one record opened at most once per session for the index object that loaded it."""
        # Views of this session share each opened record; the index evicts per case and would reload it per view.
        # A replaced index on the frame never serves a record opened from its predecessor.
        key = (frame_name, record_id)
        cached = self._coordinate_records.get(key)
        if cached is not None and cached[0] is index:
            return cached[1]
        record = index.record(record_id)
        self._coordinate_records[key] = (index, record)
        return record

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
                    self._shared_record(name, index, str(_record_value(summary, "record_id")))
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
        timing_block = _require_mapping(provenance.get("timing"), label="artifact_provenance.timing")
        return timing.admit_transient_timing_report(_require_mapping(timing_block.get("report"), label="artifact_provenance.timing.report"))

    def pipeline_degradation(self) -> tuple[comparison.AirflowDegradationMetrics, ...]:
        """
        Return B/A evidence and truthful available or unavailable C evidence per frame whose B/A error is available.

        The B/A drying-surrogate error is the frame's :meth:`selection_objective`, so it
        reduces cases exactly as checkpoint selection and the comparison ranking do.
        """
        results: list[comparison.AirflowDegradationMetrics] = []
        metric_id = metrics.NORMALIZED_DRYING_GROUP_MACRO_RMSE
        for frame_name, (frame, _artifact) in self._frames.items():
            surrogate_error = self.selection_objective(frame_name)
            if not math.isfinite(surrogate_error):
                continue
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
                        drying_surrogate_error=surrogate_error,
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
                    drying_surrogate_error=surrogate_error,
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
        self._coordinate_records.clear()

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
                    identity = _record_identity(record)
                    assignment = identity.get("evaluation_assignment")
                    row["declared_evaluation_role"] = assignment["role"] if isinstance(assignment, Mapping) else "id"
                    row["evaluation_relation"] = assignment["relation"] if isinstance(assignment, Mapping) else None
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

    def selection_objective(self, frame_name: str) -> float:
        """Return the selection objective of one frame's complete autonomous rollouts, NaN when any case is unavailable."""
        return self.selection_errors(frame_name)[metrics.NORMALIZED_DRYING_GROUP_MACRO_RMSE]

    def selection_errors(self, frame_name: str) -> dict[str, float]:
        """
        Return the selection objective and every field metric of one frame's complete autonomous rollouts.

        Each case enters with its physical field statistics summed over its
        cumulative full rollout, as its material summary holds them; the errors
        average equally over the cases of a material and over the materials, as
        checkpoint selection does. A frame with any unavailable case returns NaN
        everywhere, so partial evidence never ranks; an unavailable relative norm is NaN.
        """
        _frame, artifact = self._frames[frame_name]
        materials = [
            summary
            for summary in self.summaries(modes=("autonomous_full",))
            if summary.frame == frame_name and summary.requested_horizon == "full" and summary.scope == "cumulative"
        ]
        if not materials or any(summary.field_metrics is None for summary in materials):
            return dict.fromkeys(_NORMALIZED_COLUMNS, math.nan)
        values, _unavailable = _population_metrics([case for summary in materials for case in summary.case_statistics], artifact)
        return {name: math.nan if value is None else value for name, value in values.items()}

    def series_styles(self) -> series.SeriesStyles:
        """
        Return the fixed style of every series this comparison can draw, resolved once over all its loaded frames.

        Each frame contributes its run (model kind and selected checkpoint), Dataset role and materials from its provenance
        (``series.comparison_styles``); every transient view takes these styles, so a run keeps its colour in a view that holds
        evidence for only some runs. Each frame's whole series is also held under the frame name, for views drawn per frame.
        """
        members: dict[str, list[tuple[Any, ...]]] = {}
        for frame_name in self.frame_names:
            provenance = self._frame_provenance(frame_name)
            facts = (provenance.get("model_kind"), provenance.get("dataset_role"), provenance.get("checkpoint_sha256"))
            members[frame_name] = [(facts[0], facts[1], material, facts[2]) for material in self.material_families(frame_name)]
        styles = series.comparison_styles(member for frame_members in members.values() for member in frame_members)
        styles.update(series.select_styles(members, styles))
        return styles

    def relative_norm_curve(self, frame_name: str, *, time_axis: str = "hours") -> pd.DataFrame:
        """
        Return the relative L2 and H1 of one frame's complete autonomous rollouts at every exact physical time or phase point.

        At each post-origin time every case present enters the material-case mean of its
        per-case norm, as the horizon value does. A case whose predicted state at that time
        is non-finite, or whose reference lies below the field's floor, stays in ``cases``
        and only lowers ``supported_cases``; its other times count. Columns:
        ``physical_time``, ``metric_id``, ``value`` (NaN when a material has no supported
        case), ``cases`` and ``supported_cases``. With ``time_axis="phase"`` the clock column is
        ``phase``, every grid point of ``time_ui.phase_cohort``, and ``attrs["phase_label"]`` names the cohort.
        """
        phase = time_ui.is_phase(time_axis)
        reduction = learning.metrics.metrics
        floors = {field: domain.tasks.registry.get_task("transient_drying").reference_floor(field) for field in sequence_artifact.STATE_ORDER}
        # Per exact time or phase point: (material, field statistics of the state, or None when the predicted state is non-finite).
        by_time: dict[float, list[tuple[str, np.ndarray | None]]] = {}
        # Loaded artifacts hold array-free summaries; the full records carry the states.
        records = self.full_autonomous_records(frame_name)
        # Each pooled record with the states it contributes: every post-origin state, or its phase states (the origin a candidate).
        pooled: list[tuple[Any, np.ndarray | slice]] = [(record, slice(1, None)) for record in records]
        attrs: dict[str, str] = {}
        if phase:
            completion: list[time_ui.Completion] = [reference_completion(record) for record in records]
            cohort = time_ui.phase_cohort([_record_value(record, "physical_times") for record in records], completion)
            pooled = [(records[case], states) for case, states in zip(cohort.cases, cohort.states, strict=True)]
            by_time, attrs = {float(point): [] for point in time_ui.PHASE_GRID}, {"phase_label": cohort.label()}
        for record, states in pooled:
            prediction = np.asarray(_record_value(record, "predicted_states"), dtype=np.float64)[states]
            reference = np.asarray(_record_value(record, "reference_states"), dtype=np.float64)[states]
            spacing = metrics.grid_spacing(_static_conditioning(record, grid_shape=reference.shape[-2:]))
            rows = metrics.state_field_statistics(prediction, reference, spacing=spacing)
            times = time_ui.PHASE_GRID if phase else np.asarray(_record_value(record, "physical_times"), dtype=np.float64)[states]
            material = _material_family(record)
            for time, state, sums in zip(times, prediction, rows, strict=True):
                by_time.setdefault(float(time), []).append((material, sums if np.isfinite(state).all() else None))
        clock = "phase" if phase else "physical_time"
        result: list[dict[str, Any]] = []
        for time, cases in sorted(by_time.items()):
            for norm in reduction.RELATIVE_NORMS:
                for index, field in enumerate(sequence_artifact.STATE_ORDER):
                    per_case = [
                        (material, None if sums is None else reduction.case_relative_norm(sums[index], norm=norm, floor=floors[field]))
                        for material, sums in cases
                    ]
                    value = reduction.material_case_mean(per_case) if per_case else None  # a phase point of a frame without a reached case
                    result.append(
                        {
                            clock: time,
                            "metric_id": f"physical_relative_{norm}_{field}",
                            "value": math.nan if value is None else value,
                            "cases": len(cases),
                            "supported_cases": sum(case_value is not None for _material, case_value in per_case),
                        }
                    )
        curve = pd.DataFrame.from_records(result, columns=[clock, "metric_id", "value", "cases", "supported_cases"])
        curve.attrs.update(attrs)
        return curve

    def _reduce_summaries(self, *, modes: tuple[str, ...]) -> tuple[TransientSessionSummary, ...]:
        """Reduce exact record groups per material, mode, and horizon."""
        results: list[TransientSessionSummary] = []
        for frame_name, (frame, artifact) in self._frames.items():
            records = _sequence_records(frame)
            groups: dict[tuple[str, str, int | str], list[Any]] = {}
            for record in records:
                mode = _record_value(record, "mode")
                if mode in modes:
                    groups.setdefault((_material_family(record), mode, _requested_horizon(record)), []).append(record)
            unavailable = frame.attrs["transient_unavailable_horizons"]
            if not isinstance(unavailable, tuple):
                msg = "transient_unavailable_horizons must be one tuple."
                raise TypeError(msg)
            if "rolling_origin" in modes:
                # Every material shows each rolling horizon that any case reaches or requests, so its missing cases stay visible.
                requested = {_require_mapping(item, label="transient unavailable horizon").get("requested_horizon") for item in unavailable}
                horizons = {horizon for _material, mode, horizon in groups if mode == "rolling_origin"} | {
                    value for value in requested if isinstance(value, int) and not isinstance(value, bool)
                }
                for material in {_material_family(record) for record in records}:
                    for horizon in horizons:
                        groups.setdefault((material, "rolling_origin", horizon), [])
            for (material, mode, horizon), grouped in groups.items():
                unavailable_cases = _unavailable_cases(
                    unavailable,
                    mode=mode,
                    horizon=horizon,
                    case_ids=set(self.case_ids(frame_name, material_family=material)),
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

    def summaries(self, *, modes: Sequence[str] = _DEFAULT_MODES) -> tuple[TransientSessionSummary, ...]:
        """Return cached reductions for each material, mode, and horizon group."""
        selected = _validate_modes(modes)
        cached = self._summary_cache.get(selected)
        if cached is not None:
            return cached
        resolved = self._reduce_summaries(modes=selected)
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
    """Count the material's cases, whatever records exist at the horizon, with explicit unsupported fixed rolling horizons."""
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


def _train_scales(artifact: datasets.preprocessing.normalization.TrainScaling) -> dict[str, float]:
    """Return the run's Train state scale of every canonical state field."""
    return contracts.output_scales(artifact, output_fields=sequence_artifact.STATE_ORDER)


def _population_metrics(
    cases: Sequence[learning.metrics.metrics.CaseStatistics],
    artifact: datasets.preprocessing.normalization.TrainScaling,
) -> tuple[dict[str, float | None], dict[str, int]]:
    """
    Return the drying objective and every field metric of one case population, with the unavailable counts.

    :mod:`learning.metrics.metrics` averages each case's statistics equally over the cases
    of a material and over the materials, as checkpoint selection does: RMSE takes its root
    after the means, a relative norm divides within each case first.
    """
    task = domain.tasks.registry.get_task("transient_drying")
    scales = _train_scales(artifact)
    reduction = learning.metrics.metrics
    objective = reduction.group_objective((case.errors() for case in cases), terms=task.error_terms, train_scales=scales)
    floors = {field: task.reference_floor(field) for field in sequence_artifact.STATE_ORDER}
    values, unavailable = reduction.field_metric_values(cases, train_scales=scales, reference_floors=floors, output_groups=task.output_groups)
    return {metrics.NORMALIZED_DRYING_GROUP_MACRO_RMSE: objective.value, **values}, unavailable


def _group_errors(
    case_states: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    material_family: str,
    artifact: datasets.preprocessing.normalization.TrainScaling,
) -> tuple[_TransientMetricSummary, tuple[learning.metrics.metrics.CaseStatistics, ...], dict[str, float | None], dict[str, int]]:
    """Return the moisture metrics, per-case field statistics, and field metrics of one material group, each case weighing equally."""
    merged = {case_id: metrics.TransientMetricAccumulator.merged(states) for case_id, states in case_states.items()}
    moisture = metrics.summarize_moisture((material_family, accumulator.case_moisture()) for accumulator in merged.values())
    cases = tuple(accumulator.case_statistics(case_id, material_family) for case_id, accumulator in merged.items())
    values, unavailable = _population_metrics(cases, artifact)
    return moisture, cases, values, unavailable


def _summarize_group(
    records: Sequence[Any],
    *,
    frame: str,
    material_family: str,
    artifact: datasets.preprocessing.normalization.TrainScaling,
    mode: str,
    horizon: int | str,
    scope: str,
    unavailable_case_count: int,
) -> TransientSessionSummary:
    """Accumulate one exact material/mode/horizon/scope sufficient-statistic group; a group without records is unavailable."""
    if not records:
        none_support = validity_contract.support_evidence(required=0, computed=0, finite=0)
        no_values = np.empty(0)
        return TransientSessionSummary(
            frame=frame,
            material_family=material_family,
            mode=mode,
            requested_horizon=horizon,
            scope=scope,
            metrics=None,
            case_statistics=(),
            field_metrics=None,
            relative_unavailable_cases=None,
            metric_support={**none_support, "available": False, "unavailable_reason": "no_records_at_horizon"},
            contributing_record_count=0,
            contributing_case_count=0,
            unavailable_case_count=unavailable_case_count,
            elapsed_physical_time_min=math.nan,
            elapsed_physical_time_median=math.nan,
            elapsed_physical_time_max=math.nan,
            origin_time_min=math.nan,
            origin_time_max=math.nan,
            target=metrics.derive_target_censoring_diagnostics(predicted_reached=no_values, reference_reached=no_values),
            target_time_error_count=0,
            target_time_error_mean=None,
            target_time_error_mae=None,
            target_gap_count=0,
            predicted_final_target_gap_mean=None,
            reference_final_target_gap_mean=None,
            target_final_gap_error_mean=None,
            target_final_gap_error_mae=None,
            plausibility=_empty_plausibility(),
            stability=_empty_stability(),
        )
    materials = {_material_family(record) for record in records}
    if materials != {material_family}:
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
        case_states: dict[str, list[Mapping[str, object]]] = {}
        support = dict.fromkeys(("required", "computed", "finite"), 0)
        plausibility = _empty_plausibility()
        stability = _empty_stability()
        for record in summaries:
            statistics = record.metric_statistics
            if statistics is None:
                message = "Indexed transient summaries require persisted metric statistics."
                raise ValueError(message)
            metric_evidence = _require_mapping(
                statistics[scope],
                label=f"metric_statistics.{scope}",
            )
            for name in support:
                support[name] += int(metric_evidence[f"{name}_value_count"])
            if metric_evidence.get("available") is True:
                case_states.setdefault(record.case_id, []).append(
                    _require_mapping(metric_evidence.get("statistics"), label=f"metric_statistics.{scope}.statistics")
                )
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
        metric_support = validity_contract.support_evidence(**support)
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
        moisture, case_statistics, field_metrics, unavailable = (
            _group_errors(case_states, material_family=material_family, artifact=artifact) if metric_support["available"] else (None, (), None, None)
        )
        return TransientSessionSummary(
            frame=frame,
            material_family=material_family,
            mode=mode,
            requested_horizon=horizon,
            scope=scope,
            metrics=moisture,
            case_statistics=case_statistics,
            field_metrics=field_metrics,
            relative_unavailable_cases=unavailable,
            metric_support=metric_support,
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
    accumulators: dict[str, metrics.TransientMetricAccumulator] = {}
    required_count = finite_count = 0
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
        scalar = np.asarray(_record_value(record, "scalar_conditioning"), dtype=np.float64)
        if scalar.shape != (len(sequence_artifact.SCALAR_ORDER),):
            msg = "Transient record scalar_conditioning must retain the canonical Dataset scalar fields."
            raise ValueError(msg)
        static = _static_conditioning(record, grid_shape=prediction.shape[-2:])
        required_count += selected_prediction.size
        finite_count += int(np.isfinite(selected_prediction).sum())
        if finite_count == required_count:
            case_id = str(_record_value(record, "case_id"))
            accumulators.setdefault(case_id, metrics.TransientMetricAccumulator(scope=scope)).update(
                physical_prediction=selected_prediction,
                physical_reference=selected_reference,
                f_surf=np.asarray(scalar[2]),
                rho_bu_dry=static[_RHO_BULK_DENSITY_INDEX],
                spacing=metrics.grid_spacing(static),
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
    metric_support = validity_contract.support_evidence(required=required_count, computed=required_count, finite=finite_count)
    moisture, case_statistics, field_metrics, unavailable = (
        _group_errors(
            {case_id: [accumulator.state_dict()] for case_id, accumulator in accumulators.items()},
            material_family=material_family,
            artifact=artifact,
        )
        if metric_support["available"]
        else (None, (), None, None)
    )
    return TransientSessionSummary(
        frame=frame,
        material_family=material_family,
        mode=mode,
        requested_horizon=horizon,
        scope=scope,
        metrics=moisture,
        case_statistics=case_statistics,
        field_metrics=field_metrics,
        relative_unavailable_cases=unavailable,
        metric_support=metric_support,
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
