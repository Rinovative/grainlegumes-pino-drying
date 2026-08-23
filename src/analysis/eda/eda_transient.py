"""
eda_transient.py

Define semantic completed-output exploratory analysis for transient drying.
Responsibilities:
  - Discover canonical transient fields without duplicating the Training profile
  - Select exact physical-time states and summarize physical trajectories
  - Report target attainment, censoring, schedules, parameters, and runtime evidence
  - Assemble backend-neutral scientific views from validated transient runtime items
Design principles:
  - Canonical HDF5 remains the completed-case provenance and diagnostic authority
  - PT/HDF5 runtime items share one physical-value adapter independent of storage
  - Incompatible physical channels remain separate throughout every reduction
This module does NOT:
  - Run Training or inference, calculate surrogate errors, or derive speedup
  - Parse Generation sidecars, solver logs, scheduler output, or source CSV files
  - Treat exact-stop diagnostic states as learned transitions or rollout targets
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final

import numpy as np
import pandas as pd

from src import domain, generation
from src.datasets.contracts import dataset_contracts_transient as transient_contract

_TRANSIENT_TASK_ID: Final = transient_contract.TRANSIENT_PROFILE_ID
_REQUIRED_COLUMNS: Final = (
    "state_trajectories",
    "static_fields",
    "boundary_intervals",
    "scalar_conditioning",
    "time",
    "meta",
)
_TARGET_GAP_SIGN: Final = "positive_still_too_wet_negative_below_target"
_MINIMUM_STATE_COUNT: Final = 2
_MINIMUM_SCHEDULE_NODE_COUNT: Final = 2
_SPATIAL_STATE_NDIM: Final = 3
_ROLLOUT_TARGET_NDIM: Final = 4
_BOUNDARY_SEQUENCE_NDIM: Final = 2
_FLOAT32_TIME_TOLERANCE_MULTIPLIER: Final = 4.0
_FRACTION_ROUNDOFF_TOLERANCE: Final = 4.0 * math.ulp(1.0)


def _finite_real_value(value: Any) -> float | None:
    """Return one finite real scalar without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _within_unit_interval(value: float) -> bool:
    """Accept a physical fraction plus only machine-scale boundary roundoff."""
    lower_valid = value >= 0.0 or math.isclose(
        value,
        0.0,
        rel_tol=0.0,
        abs_tol=_FRACTION_ROUNDOFF_TOLERANCE,
    )
    upper_valid = value <= 1.0 or math.isclose(
        value,
        1.0,
        rel_tol=0.0,
        abs_tol=_FRACTION_ROUNDOFF_TOLERANCE,
    )
    return lower_valid and upper_valid


class PhysicalTimeUnavailableError(ValueError):
    """Report an exact physical-time request absent from one case trajectory."""


@dataclass(frozen=True, slots=True)
class TransientSnapshot:
    """Hold one exact physical state snapshot and its display semantics."""

    case_id: str
    physical_time_hours: float
    channels: tuple[str, ...]
    fields: dict[str, np.ndarray]
    units: dict[str, str]
    diagnostic_exact_stop: bool


@dataclass(frozen=True, slots=True)
class SupportedScheduleSeries:
    """Hold one linearly evaluated schedule on exact simulated case support."""

    quantity: str
    unit: str
    physical_time_hours: np.ndarray
    values: np.ndarray
    final_time_hours: float

    def value_at(self, physical_time_hours: float) -> float:
        """Evaluate the maintained linear schedule without extrapolation."""
        if isinstance(physical_time_hours, bool):
            message = "Schedule evaluation time must be one finite scalar."
            raise TypeError(message)
        requested = float(physical_time_hours)
        if not math.isfinite(requested):
            message = "Schedule evaluation time must be one finite scalar."
            raise ValueError(message)
        lower = float(self.physical_time_hours[0])
        if requested < lower or requested > self.final_time_hours:
            message = f"Schedule evaluation time {requested:g} h is outside simulated support [{lower:g}, {self.final_time_hours:g}] h."
            raise ValueError(message)
        return float(
            np.interp(
                requested,
                self.physical_time_hours,
                self.values,
            )
        )


@dataclass(frozen=True, slots=True)
class TargetAttainmentDiagnostic:
    """Hold summary, case, reached-duration, group, and exclusion evidence."""

    summary: dict[str, Any]
    cases: pd.DataFrame
    reached_distribution: dict[str, Any]
    groups: pd.DataFrame
    exclusion_reasons: dict[str, int]


@dataclass(frozen=True, slots=True)
class CompletionTargetAnalysis:
    """Hold per-dataset outcome shares, exact cases, and omission accounting."""

    outcomes: pd.DataFrame
    cases: pd.DataFrame
    omissions: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FieldDiscovery:
    """Describe one EDA field through its current authoritative contract."""

    name: str
    unit: str
    category: str
    availability: str


def _state_contract() -> tuple[tuple[str, ...], dict[str, str]]:
    """Return canonical state order and units from the Dataset contract."""
    fields = transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state
    return tuple(field.name for field in fields), {field.name: field.unit for field in fields}


def _task_contract_digest() -> str:
    """Return the currently registered transient TaskSpec digest."""
    return domain.tasks.registry.get_task(_TRANSIENT_TASK_ID).contract_digest


def _semantic_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    """Narrow one validated nested semantic value to a read-only mapping."""
    if not isinstance(value, Mapping):
        message = f"Transient EDA {label} must be a semantic mapping."
        raise TypeError(message)
    return value


def validate_transient_frame(frame: pd.DataFrame) -> None:
    """Require one task-bound nested transient EDA frame."""
    if not isinstance(frame, pd.DataFrame):
        msg = "Transient EDA input must be a pandas DataFrame."
        raise TypeError(msg)
    if frame.attrs.get("task_id") != _TRANSIENT_TASK_ID:
        msg = "Transient EDA requires a transient_drying frame."
        raise ValueError(msg)
    if frame.attrs.get("task_contract_digest") != _task_contract_digest():
        msg = "Transient EDA frame TaskSpec identity is stale or inconsistent."
        raise ValueError(msg)
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"Transient EDA frame lacks required semantic columns: {missing}."
        raise ValueError(msg)
    state_names, _units = _state_contract()
    for case_id, row in frame.iterrows():
        states = _semantic_mapping(
            row["state_trajectories"],
            label=f"case {case_id!r} state trajectories",
        )
        _semantic_mapping(row["static_fields"], label=f"case {case_id!r} static fields")
        _semantic_mapping(
            row["boundary_intervals"],
            label=f"case {case_id!r} boundary intervals",
        )
        _semantic_mapping(
            row["scalar_conditioning"],
            label=f"case {case_id!r} scalar conditioning",
        )
        time = _semantic_mapping(row["time"], label=f"case {case_id!r} time evidence")
        if tuple(states) != state_names:
            msg = f"Transient case {case_id!r} state order is not canonical."
            raise ValueError(msg)
        regular = np.asarray(time.get("regular_state_hours"), dtype=np.float64)
        valid_mask = np.asarray(time.get("valid_state_mask"))
        trajectory_length = time.get("trajectory_length")
        if regular.ndim != 1 or regular.size < _MINIMUM_STATE_COUNT or not np.isfinite(regular).all():
            msg = f"Transient case {case_id!r} has an invalid physical-time axis."
            raise ValueError(msg)
        if valid_mask.dtype != np.dtype(bool) or valid_mask.shape != regular.shape or trajectory_length != int(np.count_nonzero(valid_mask)):
            msg = f"Transient case {case_id!r} has invalid trajectory-length or mask evidence."
            raise ValueError(msg)
        shapes = [np.asarray(states[name]).shape for name in state_names]
        if any(len(shape) != _SPATIAL_STATE_NDIM or shape[0] != regular.size for shape in shapes) or len(set(shapes)) != 1:
            msg = f"Transient case {case_id!r} state trajectories are not aligned."
            raise ValueError(msg)


def discover_fields(frame: pd.DataFrame) -> dict[str, tuple[FieldDiscovery, ...]]:
    """Discover canonical state, static, boundary, scalar, and schedule fields."""
    validate_transient_frame(frame)
    contract = transient_contract.TRANSIENT_STEP_CONTRACT
    profile = generation.contracts.get_profile_contract(_TRANSIENT_TASK_ID)
    retained_static: set[str] = set()
    retained_schedule: set[str] = set()
    if not frame.empty:
        retained_static = set(frame.iloc[0]["static_fields"])
        schedule = frame.iloc[0].get("schedule")
        retained_schedule = set(schedule) if isinstance(schedule, Mapping) else set()
    training_static = {field.name for field in contract.static_spatial_conditioning}
    archived = {field.name for field in contract.archived_ablation_fields}
    return {
        "dynamic_state": tuple(FieldDiscovery(field.name, field.unit, "dynamic_state", "available") for field in contract.dynamic_state),
        "static_spatial": tuple(
            FieldDiscovery(
                field.name,
                field.unit,
                "training_static" if field.name in training_static else "archive_only",
                "available" if field.name in retained_static else "unavailable",
            )
            for field in (*profile.coordinate_fields, *profile.static_fields)
            if field.name in training_static or field.name in archived
        ),
        "boundary_interval": tuple(
            FieldDiscovery(field.name, field.unit, "boundary_interval", "available") for field in contract.step_boundary_conditioning
        ),
        "scalar_material": tuple(FieldDiscovery(field.name, field.unit, "scalar_material", "available") for field in contract.scalar_conditioning),
        "complete_schedule": tuple(
            FieldDiscovery(
                field.name,
                field.unit,
                "complete_schedule",
                "available" if field.name in retained_schedule else "unavailable",
            )
            for field in profile.schedule_fields
        ),
    }


def resolve_dynamic_channels(
    frame: pd.DataFrame,
    channels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return a canonical ordered channel selection, defaulting to all states."""
    validate_transient_frame(frame)
    canonical, _units = _state_contract()
    if channels is None:
        return canonical
    if isinstance(channels, str):
        msg = "Dynamic channel selection must be a sequence, not one string."
        raise TypeError(msg)
    requested = tuple(channels)
    if not requested or len(requested) != len(set(requested)):
        msg = "Dynamic channel selection must be non-empty and unique."
        raise ValueError(msg)
    unknown = set(requested).difference(canonical)
    if unknown:
        msg = f"Unknown transient state channels: {sorted(unknown)}."
        raise ValueError(msg)
    requested_set = set(requested)
    return tuple(name for name in canonical if name in requested_set)


def _case_row(frame: pd.DataFrame, case_id: str) -> pd.Series:
    """Return one exact case row by its public frame identity."""
    validate_transient_frame(frame)
    matches = np.flatnonzero(frame.index.astype(str).to_numpy() == str(case_id))
    if matches.size != 1:
        msg = f"Transient EDA frame has no unique case {case_id!r}."
        raise KeyError(msg)
    return frame.iloc[int(matches[0])]


def available_physical_times(
    frame: pd.DataFrame,
    case_id: str,
    *,
    include_exact_stop: bool = True,
) -> tuple[float, ...]:
    """Return exact selectable state times without nearest-time synthesis."""
    row = _case_row(frame, case_id)
    values = [float(value) for value in np.asarray(row["time"]["regular_state_hours"])]
    exact_stop = row.get("exact_stop")
    if include_exact_stop and isinstance(exact_stop, Mapping) and isinstance(exact_stop.get("state"), Mapping):
        values.append(float(exact_stop["time_hours"]))
    return tuple(values)


def final_physical_time_hours(
    frame: pd.DataFrame,
    case_id: str,
) -> float:
    """Return one case's exact authoritative final simulated physical time."""
    row = _case_row(frame, case_id)
    completion = _semantic_mapping(
        row.get("completion"),
        label=f"case {case_id!r} completion evidence",
    )
    final_time = _finite_real_value(completion.get("physical_duration_hours"))
    if final_time is None or final_time < 0.0:
        message = f"Transient case {case_id!r} lacks a valid final physical time."
        raise ValueError(message)
    available = available_physical_times(frame, case_id)
    time = _semantic_mapping(
        row["time"],
        label=f"case {case_id!r} time evidence",
    )
    tolerance = _finite_real_value(time.get("classification_tolerance_hours"))
    if tolerance is None or tolerance < 0.0:
        message = f"Transient case {case_id!r} lacks a valid time-classification tolerance."
        raise ValueError(message)
    if not math.isclose(
        final_time,
        available[-1],
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        message = f"Transient case {case_id!r} final duration disagrees with stored state support."
        raise ValueError(message)
    return final_time


def supported_schedule_series(
    frame: pd.DataFrame,
    case_id: str,
    quantity: str,
) -> SupportedScheduleSeries:
    """Return one complete schedule clipped to exact simulated case support."""
    row = _case_row(frame, case_id)
    schedule = _semantic_mapping(
        row.get("schedule"),
        label=f"case {case_id!r} complete schedule",
    )
    profile = generation.contracts.get_profile_contract(_TRANSIENT_TASK_ID)
    fields = {field.name: field for field in profile.schedule_fields if field.name != "t"}
    try:
        field = fields[quantity]
    except KeyError as error:
        message = f"Unknown transient schedule quantity: {quantity!r}."
        raise ValueError(message) from error
    times = np.asarray(schedule.get("t"), dtype=np.float64)
    values = np.asarray(schedule.get(quantity), dtype=np.float64)
    if (
        times.ndim != 1
        or times.size < _MINIMUM_SCHEDULE_NODE_COUNT
        or values.shape != times.shape
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        message = f"Transient case {case_id!r} has invalid schedule evidence for {quantity!r}."
        raise ValueError(message)
    final_time = final_physical_time_hours(frame, case_id)
    if final_time < times[0] or final_time > times[-1]:
        message = (
            f"Transient case {case_id!r} final time {final_time:g} h is outside the configured schedule support [{times[0]:g}, {times[-1]:g}] h."
        )
        raise ValueError(message)
    stop = int(np.searchsorted(times, final_time, side="right"))
    supported_times = np.array(times[:stop], copy=True)
    supported_values = np.array(values[:stop], copy=True)
    if supported_times.size == 0 or supported_times[-1] != final_time:
        supported_times = np.append(supported_times, final_time)
        supported_values = np.append(
            supported_values,
            np.interp(final_time, times, values),
        )
    return SupportedScheduleSeries(
        quantity=quantity,
        unit=field.unit,
        physical_time_hours=np.ascontiguousarray(supported_times),
        values=np.ascontiguousarray(supported_values),
        final_time_hours=final_time,
    )


def _exact_time_index(values: np.ndarray, requested: float, tolerance: float) -> int | None:
    """Return one exact classified time index or report ambiguity."""
    matches = np.flatnonzero(np.isclose(values, requested, rtol=0.0, atol=tolerance))
    if matches.size > 1:
        msg = "Physical-time classification is ambiguous within its persisted tolerance."
        raise ValueError(msg)
    return None if matches.size == 0 else int(matches[0])


def select_state_snapshot(
    frame: pd.DataFrame,
    case_id: str,
    physical_time_hours: float,
    *,
    channels: Sequence[str] | None = None,
) -> TransientSnapshot:
    """Select one exact regular or diagnostic state at a physical time."""
    if isinstance(physical_time_hours, bool) or not isinstance(physical_time_hours, (int, float)) or not math.isfinite(float(physical_time_hours)):
        msg = "physical_time_hours must be one finite real scalar."
        raise TypeError(msg)
    row = _case_row(frame, case_id)
    selected = resolve_dynamic_channels(frame, channels)
    requested = float(physical_time_hours)
    time = _semantic_mapping(row["time"], label="case time evidence")
    tolerance = float(time["classification_tolerance_hours"])
    regular = np.asarray(time["regular_state_hours"], dtype=np.float64)
    index = _exact_time_index(regular, requested, tolerance)
    diagnostic = False
    if index is not None:
        states = _semantic_mapping(row["state_trajectories"], label="state trajectories")
        fields = {name: np.ascontiguousarray(np.asarray(states[name][index])) for name in selected}
    else:
        exact = row.get("exact_stop")
        exact_time = exact.get("time_hours") if isinstance(exact, Mapping) else None
        exact_state = exact.get("state") if isinstance(exact, Mapping) else None
        if (
            not isinstance(exact_time, (int, float))
            or not isinstance(exact_state, Mapping)
            or not math.isclose(requested, float(exact_time), rel_tol=0.0, abs_tol=tolerance)
        ):
            available = available_physical_times(frame, case_id)
            msg = f"Physical time {requested:g} h is unavailable for {case_id!r}; available times are {available}."
            raise PhysicalTimeUnavailableError(msg)
        fields = {name: np.ascontiguousarray(np.asarray(exact_state[name])) for name in selected}
        diagnostic = True
    _names, units = _state_contract()
    return TransientSnapshot(
        case_id=str(case_id),
        physical_time_hours=requested,
        channels=selected,
        fields=fields,
        units={name: units[name] for name in selected},
        diagnostic_exact_stop=diagnostic,
    )


def trajectory_table(
    frame: pd.DataFrame,
    case_id: str,
    *,
    channels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return per-channel spatial summaries over the full physical trajectory."""
    row = _case_row(frame, case_id)
    selected = resolve_dynamic_channels(frame, channels)
    physical_time = np.asarray(row["time"]["regular_state_hours"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    _names, units = _state_contract()
    for name in selected:
        values = np.asarray(row["state_trajectories"][name], dtype=np.float64)
        for index, time_value in enumerate(physical_time):
            field = values[index]
            records.append(
                {
                    "case_id": str(case_id),
                    "physical_time_hours": float(time_value),
                    "channel": name,
                    "unit": units[name],
                    "spatial_mean": float(np.mean(field)),
                    "spatial_minimum": float(np.min(field)),
                    "spatial_maximum": float(np.max(field)),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.attrs["time_control"] = "none_full_trajectory_uses_x_axis"
    result.attrs["channel_selection"] = "multi_default_all"
    return result


def fixed_time_summary(
    frame: pd.DataFrame,
    physical_time_hours: float,
    *,
    channels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Align exact physical-time summaries across cases and disclose exclusions."""
    selected = resolve_dynamic_channels(frame, channels)
    records: list[dict[str, Any]] = []
    unavailable: list[str] = []
    shorter: list[str] = []
    right_censored_cases = 0
    right_censored_contributors = 0
    requested_time = float(physical_time_hours)
    for case_id, row in frame.iterrows():
        completion = row.get("completion")
        is_right_censored = isinstance(completion, Mapping) and completion.get("right_censored") is True
        right_censored_cases += int(is_right_censored)
        try:
            snapshot = select_state_snapshot(
                frame,
                str(case_id),
                requested_time,
                channels=selected,
            )
        except PhysicalTimeUnavailableError:
            unavailable.append(str(case_id))
            time = _semantic_mapping(row["time"], label=f"case {case_id!r} time evidence")
            tolerance = float(time["classification_tolerance_hours"])
            available = available_physical_times(frame, str(case_id))
            if available and requested_time > max(available) + tolerance:
                shorter.append(str(case_id))
            continue
        right_censored_contributors += int(is_right_censored)
        for name in selected:
            field = np.asarray(snapshot.fields[name], dtype=np.float64)
            records.append(
                {
                    "case_id": str(case_id),
                    "physical_time_hours": snapshot.physical_time_hours,
                    "channel": name,
                    "unit": snapshot.units[name],
                    "spatial_mean": float(np.mean(field)),
                    "spatial_minimum": float(np.min(field)),
                    "spatial_maximum": float(np.max(field)),
                    "diagnostic_exact_stop": snapshot.diagnostic_exact_stop,
                }
            )
    result = pd.DataFrame.from_records(records)
    result.attrs.update(
        {
            "contributing_case_count": len(frame) - len(unavailable),
            "unavailable_case_count": len(unavailable),
            "unavailable_case_ids": tuple(unavailable),
            "shorter_trajectory_case_count": len(shorter),
            "shorter_trajectory_case_ids": tuple(shorter),
            "right_censored_case_count": right_censored_cases,
            "right_censored_contributor_count": right_censored_contributors,
            "failed_case_count": int(frame.attrs.get("failed_case_count", 0)),
            "incomplete_case_count": int(frame.attrs.get("incomplete_case_count", 0)),
            "alignment": "exact_physical_time_no_nearest_substitution",
        }
    )
    return result


def boundary_interval_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all per-transition boundary endpoints and startup support evidence."""
    validate_transient_frame(frame)
    fields = transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning
    records: list[dict[str, Any]] = []
    for case_id, row in frame.iterrows():
        time = row["time"]
        t_n = np.asarray(time["transition_t_n_hours"], dtype=np.float64)
        t_np1 = np.asarray(time["transition_t_n_plus_1_hours"], dtype=np.float64)
        boundary = row["boundary_intervals"]
        for index, (start, stop) in enumerate(zip(t_n, t_np1, strict=True)):
            record: dict[str, Any] = {
                "case_id": str(case_id),
                "transition_index": index,
                "t_n_hours": float(start),
                "t_n_plus_1_hours": float(stop),
            }
            for field in fields:
                record[field.name] = float(np.asarray(boundary[field.name])[index])
            records.append(record)
    result = pd.DataFrame.from_records(records)
    result.attrs["interpolation"] = transient_contract.TRANSIENT_STEP_CONTRACT.boundary_interval_interpolation
    result.attrs["time_control"] = "none_fixed_transition_evidence"
    return result


def schedule_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete schedules without flattening their source arrays."""
    validate_transient_frame(frame)
    profile = generation.contracts.get_profile_contract(_TRANSIENT_TASK_ID)
    records: list[dict[str, Any]] = []
    for case_id, row in frame.iterrows():
        schedule = row.get("schedule")
        if not isinstance(schedule, Mapping):
            continue
        for field in profile.schedule_fields:
            if field.name == "t":
                continue
            values = np.asarray(schedule[field.name], dtype=np.float64)
            records.append(
                {
                    "case_id": str(case_id),
                    "quantity": field.name,
                    "unit": field.unit,
                    "initial": float(values[0]),
                    "final": float(values[-1]),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "amplitude": float(np.max(values) - np.min(values)),
                    "endpoint_change": float(values[-1] - values[0]),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.attrs["source"] = "complete_generation_schedule"
    result.attrs["time_control"] = "none_full_schedule_uses_x_axis"
    return result


def scalar_parameter_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return realized completed-case material-conditioning parameters."""
    validate_transient_frame(frame)
    fields = transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning
    records: list[dict[str, Any]] = []
    for case_id, row in frame.iterrows():
        values = _semantic_mapping(row["scalar_conditioning"], label="scalar conditioning")
        meta = _semantic_mapping(row["meta"], label="case metadata")
        records.extend(
            {
                "case_id": str(case_id),
                "material_family": meta.get("material_family"),
                "parameter": field.name,
                "unit": field.unit,
                "value": float(values[field.name]),
                "distribution_kind": "realized_completed_case",
            }
            for field in fields
        )
    result = pd.DataFrame.from_records(records)
    result.attrs["time_control"] = "none_static_parameter"
    return result


def runtime_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose separated physical-duration and operational runtime evidence."""
    validate_transient_frame(frame)
    records: list[dict[str, Any]] = []
    for case_id, row in frame.iterrows():
        runtime = row.get("runtime")
        completion = row.get("completion")
        if not isinstance(runtime, Mapping) or not isinstance(completion, Mapping):
            continue
        availability = runtime.get("component_timing_availability")
        availability = availability if isinstance(availability, Mapping) else {}
        records.append(
            {
                "case_id": str(case_id),
                "physical_drying_duration_hours": completion.get("physical_duration_hours"),
                "time_to_target_hours": completion.get("time_to_target_hours"),
                "stationary_airflow_solver_seconds": runtime.get("stationary_airflow_solver_seconds"),
                "transient_drying_solver_seconds": runtime.get("transient_drying_solver_seconds"),
                "scientific_solver_seconds": runtime.get("scientific_solver_seconds"),
                "comsol_process_seconds": runtime.get("comsol_process_seconds"),
                "queue_wait_seconds": runtime.get("queue_wait_seconds"),
                "licence_wait_seconds": runtime.get("licence_wait_seconds"),
                "generation_compute_end_to_end_seconds": runtime.get("generation_compute_end_to_end_seconds"),
                "complete_execution_seconds": runtime.get("complete_execution_seconds"),
                "component_timing_availability": dict(availability),
            }
        )
    result = pd.DataFrame.from_records(records)
    result.attrs["runtime_is_model_input"] = False
    result.attrs["runtime_is_dataset_identity"] = False
    return result


def _distribution(values: np.ndarray, *, configured_horizon_hours: float | None) -> dict[str, Any]:
    """Return the required reached-only duration statistics."""
    if values.size == 0:
        return {
            "unit": "h",
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "configured_horizon_hours": configured_horizon_hours,
        }
    return {
        "unit": "h",
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "configured_horizon_hours": configured_horizon_hours,
    }


def target_attainment_diagnostic(  # noqa: PLR0912 -- one complete diagnostic admission pass
    frame: pd.DataFrame,
    *,
    group_by: str = "material_family",
) -> TargetAttainmentDiagnostic:
    """Report canonical target attainment, censoring, duration, and signed gap."""
    validate_transient_frame(frame)
    records: list[dict[str, Any]] = []
    exclusions: dict[str, int] = dict(frame.attrs.get("exclusion_reasons", {}))
    for case_id, row in frame.iterrows():
        completion = row.get("completion")
        if not isinstance(completion, Mapping):
            exclusions["completion_evidence_unavailable"] = exclusions.get("completion_evidence_unavailable", 0) + 1
            continue
        reached = completion.get("target_reached")
        censored = completion.get("right_censored")
        duration_value = _finite_real_value(completion.get("physical_duration_hours"))
        target_time = completion.get("time_to_target_hours")
        final_wet_value = _finite_real_value(completion.get("final_wet_fraction"))
        target_limit_value = _finite_real_value(completion.get("target_wet_fraction_limit"))
        final_bulk_moisture_value = _finite_real_value(completion.get("final_bulk_moisture_wb"))
        target_moisture_value = _finite_real_value(completion.get("target_moisture_wb"))
        if (
            not isinstance(reached, bool)
            or not isinstance(censored, bool)
            or duration_value is None
            or final_wet_value is None
            or target_limit_value is None
            or final_bulk_moisture_value is None
            or target_moisture_value is None
            or duration_value < 0.0
            or not _within_unit_interval(final_wet_value)
            or not _within_unit_interval(target_limit_value)
            or not _within_unit_interval(final_bulk_moisture_value)
            or not 0.0 < target_moisture_value < 1.0
        ):
            exclusions["invalid_completion_evidence"] = exclusions.get("invalid_completion_evidence", 0) + 1
            continue
        if reached is censored:
            msg = f"Case {case_id!r} target and censoring states are inconsistent."
            raise ValueError(msg)

        time = _semantic_mapping(row["time"], label="case time evidence")
        tolerance_value = _finite_real_value(time.get("classification_tolerance_hours", 0.0))
        if tolerance_value is None or tolerance_value < 0.0:
            exclusions["invalid_completion_evidence"] = exclusions.get("invalid_completion_evidence", 0) + 1
            continue
        target_time_value: float | None = None
        if reached:
            target_time_value = _finite_real_value(target_time)
            outside_terminal_interval = (
                target_time_value is None
                or target_time_value < 0.0
                or (
                    target_time_value > duration_value
                    and not math.isclose(
                        target_time_value,
                        duration_value,
                        rel_tol=0.0,
                        abs_tol=tolerance_value,
                    )
                )
            )
            if outside_terminal_interval:
                exclusions["invalid_completion_evidence"] = exclusions.get("invalid_completion_evidence", 0) + 1
                continue
        elif target_time is not None:
            msg = f"Unreached case {case_id!r} fabricates a time-to-target value."
            raise ValueError(msg)

        gap = final_wet_value - target_limit_value
        if reached != (gap <= 0.0):
            msg = f"Case {case_id!r} target state disagrees with the canonical criterion."
            raise ValueError(msg)
        meta = _semantic_mapping(row["meta"], label="case metadata")
        runtime_value = row.get("runtime")
        runtime = runtime_value if isinstance(runtime_value, Mapping) else None
        configured_horizon = _finite_real_value(time.get("configured_horizon_hours"))
        if reached:
            terminal_time_kind = "target_attainment_exact_stop" if isinstance(row.get("exact_stop"), Mapping) else "target_attainment"
        elif configured_horizon is not None and math.isclose(
            duration_value,
            configured_horizon,
            rel_tol=0.0,
            abs_tol=tolerance_value,
        ):
            terminal_time_kind = "configured_maximum_duration"
        else:
            terminal_time_kind = "right_censoring_time"
        records.append(
            {
                "case_id": str(case_id),
                "material_family": meta.get("material_family"),
                "dataset_role": meta.get("dataset_role"),
                "case_family": meta.get("case_family"),
                "completion_state": ("target_reached" if reached else "right_censored"),
                "terminal_time_kind": terminal_time_kind,
                "target_reached": reached,
                "right_censored": censored,
                "physical_duration_hours": duration_value,
                "time_to_target_hours": target_time_value,
                "configured_horizon_hours": configured_horizon,
                "final_bulk_moisture_wb": final_bulk_moisture_value,
                "target_moisture_wb": target_moisture_value,
                "final_wet_fraction": final_wet_value,
                "target_wet_fraction_limit": target_limit_value,
                "final_target_gap": gap,
                "final_target_gap_unit": "1",
                "final_target_gap_sign": _TARGET_GAP_SIGN,
                "stationary_airflow_solver_seconds": (None if runtime is None else runtime.get("stationary_airflow_solver_seconds")),
                "transient_drying_solver_seconds": (None if runtime is None else runtime.get("transient_drying_solver_seconds")),
                "scientific_solver_seconds": (None if runtime is None else runtime.get("scientific_solver_seconds")),
                "comsol_process_seconds": (None if runtime is None else runtime.get("comsol_process_seconds")),
                "queue_wait_seconds": (None if runtime is None else runtime.get("queue_wait_seconds")),
                "licence_wait_seconds": (None if runtime is None else runtime.get("licence_wait_seconds")),
                "generation_compute_end_to_end_seconds": (None if runtime is None else runtime.get("generation_compute_end_to_end_seconds")),
                "complete_execution_seconds": (None if runtime is None else runtime.get("complete_execution_seconds")),
            }
        )
    cases = pd.DataFrame.from_records(records)
    valid = len(cases)
    reached_count = int(cases["target_reached"].sum()) if valid else 0
    unreached_count = valid - reached_count
    total_discovered = int(frame.attrs.get("total_discovered_case_count", len(frame)))
    failed = int(frame.attrs.get("failed_case_count", 0))
    incomplete = int(frame.attrs.get("incomplete_case_count", 0))
    excluded = total_discovered - valid - failed - incomplete
    if excluded < 0:
        msg = "Transient case accounting exceeds the discovered-case count."
        raise ValueError(msg)
    accounted_exclusions = sum(exclusions.values())
    if accounted_exclusions < excluded:
        exclusions["bounded_or_unselected"] = excluded - accounted_exclusions
    denominator = valid
    omitted = total_discovered - denominator
    omission_reasons = dict(exclusions)
    if failed:
        omission_reasons["failed_simulation"] = failed
    if incomplete:
        omission_reasons["incomplete_output"] = incomplete
    if sum(omission_reasons.values()) != omitted:
        msg = "Transient omission reasons do not match the non-eligible case count."
        raise ValueError(msg)
    summary = {
        "total_discovered_cases": total_discovered,
        "case_accounting_scope": frame.attrs.get(
            "case_accounting_scope",
            "validated_runtime_item_selection",
        ),
        "eligible_case_count": valid,
        "valid_completed_cases": valid,
        "outcome_percentage_denominator": "eligible_target_diagnostic_cases",
        "outcome_categories_mutually_exclusive": True,
        "reached_count": reached_count,
        "reached_percentage": None if denominator == 0 else 100.0 * reached_count / denominator,
        "reached_percentage_denominator": "eligible_target_diagnostic_cases",
        "unreached_count": unreached_count,
        "unreached_percentage": None if denominator == 0 else 100.0 * unreached_count / denominator,
        "unreached_percentage_denominator": "eligible_target_diagnostic_cases",
        "failed_count": failed,
        "incomplete_count": incomplete,
        "excluded_count": excluded,
        "omitted_count": omitted,
        "omission_reasons": omission_reasons,
        "final_target_gap_sign": _TARGET_GAP_SIGN,
        "physical_duration_unit": "h",
        "runtime_unit": "s",
    }
    horizons: set[float] = set()
    for _case_id, row in frame.iterrows():
        time = _semantic_mapping(row["time"], label="case time evidence")
        value = time.get("configured_horizon_hours")
        if isinstance(value, (int, float)):
            horizons.add(float(value))
    configured_horizon = next(iter(horizons)) if len(horizons) == 1 else None
    reached_values = (
        cases.loc[cases["target_reached"], "time_to_target_hours"].to_numpy(dtype=np.float64) if valid else np.asarray([], dtype=np.float64)
    )
    if valid:
        group_values = cases[group_by] if group_by in cases.columns else pd.Series([None] * valid)
        group_frame = cases.assign(_group=group_values.fillna("unavailable"))
        groups = (
            group_frame.groupby("_group", sort=True, dropna=False)
            .agg(
                case_count=("case_id", "size"),
                reached_count=("target_reached", "sum"),
                right_censored_count=("right_censored", "sum"),
            )
            .reset_index()
            .rename(columns={"_group": group_by})
        )
    else:
        groups = pd.DataFrame(columns=[group_by, "case_count", "reached_count", "right_censored_count"])
    return TargetAttainmentDiagnostic(
        summary=summary,
        cases=cases,
        reached_distribution=_distribution(
            reached_values,
            configured_horizon_hours=configured_horizon,
        ),
        groups=groups,
        exclusion_reasons=exclusions,
    )


def completion_target_analysis(
    datasets: Mapping[str, pd.DataFrame],
) -> CompletionTargetAnalysis:
    """Assemble independent-denominator outcomes and exact eligible-case details."""
    if not datasets:
        message = "Completion-target analysis requires at least one dataset."
        raise ValueError(message)
    outcome_records: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    omission_records: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        diagnostic = target_attainment_diagnostic(frame)
        summary = diagnostic.summary
        denominator = int(summary["eligible_case_count"])
        categories = (
            ("Reached target", int(summary["reached_count"])),
            ("Right-censored", int(summary["unreached_count"])),
        )
        for outcome, count in categories:
            outcome_records.append(
                {
                    "dataset": label,
                    "outcome": outcome,
                    "count": count,
                    "eligible_case_count": denominator,
                    "percentage": (None if denominator == 0 else 100.0 * count / denominator),
                }
            )
        if not diagnostic.cases.empty:
            case_frames.append(diagnostic.cases.assign(dataset=label).loc[:, ["dataset", *diagnostic.cases.columns]])
        omission_records.append(
            {
                "dataset": label,
                "eligible_case_count": denominator,
                "omitted_case_count": int(summary["omitted_count"]),
                "total_discovered_cases": int(summary["total_discovered_cases"]),
                "omission_reasons": dict(summary["omission_reasons"]),
            }
        )
    cases = (
        pd.concat(case_frames, ignore_index=True)
        if case_frames
        else pd.DataFrame(
            columns=(
                "dataset",
                "case_id",
                "completion_state",
                "terminal_time_kind",
                "physical_duration_hours",
                "time_to_target_hours",
                "configured_horizon_hours",
                "final_bulk_moisture_wb",
                "target_moisture_wb",
                "final_wet_fraction",
                "target_wet_fraction_limit",
                "final_target_gap",
                "stationary_airflow_solver_seconds",
                "transient_drying_solver_seconds",
                "scientific_solver_seconds",
                "comsol_process_seconds",
                "queue_wait_seconds",
                "licence_wait_seconds",
                "generation_compute_end_to_end_seconds",
                "complete_execution_seconds",
            )
        )
    )
    return CompletionTargetAnalysis(
        outcomes=pd.DataFrame.from_records(outcome_records),
        cases=cases,
        omissions=pd.DataFrame.from_records(omission_records),
    )


def _numpy(value: Any) -> np.ndarray:
    """Convert one semantic tensor or array to detached CPU NumPy evidence."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        msg = "Transient runtime item contains non-finite scientific values."
        raise ValueError(msg)
    return result


def _runtime_transitions(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize one one-step or rollout item into physical transitions."""
    state = _numpy(item["state"]).astype(np.float32, copy=False)
    target = _numpy(item["target"]).astype(np.float32, copy=False)
    boundary = _numpy(item["boundary"]).astype(np.float32, copy=False)
    if target.ndim == _SPATIAL_STATE_NDIM:
        target = target[np.newaxis, ...]
    if boundary.ndim == 1:
        boundary = boundary[np.newaxis, ...]
    time = item["time"]
    t_n = np.atleast_1d(_numpy(time["t_n"]).astype(np.float64, copy=False))
    t_np1 = np.atleast_1d(_numpy(time["t_n_plus_1"]).astype(np.float64, copy=False))
    dt = np.atleast_1d(_numpy(time["dt"]).astype(np.float64, copy=False))
    length = target.shape[0]
    if (
        state.ndim != _SPATIAL_STATE_NDIM
        or target.ndim != _ROLLOUT_TARGET_NDIM
        or boundary.ndim != _BOUNDARY_SEQUENCE_NDIM
        or boundary.shape[0] != length
        or t_n.shape != (length,)
        or t_np1.shape != (length,)
        or dt.shape != (length,)
    ):
        msg = "Transient runtime item tensor and time shapes are inconsistent."
        raise ValueError(msg)
    records = []
    current = state
    for index in range(length):
        following = current + target[index]
        records.append(
            {
                "t_n": float(t_n[index]),
                "t_n_plus_1": float(t_np1[index]),
                "dt": float(dt[index]),
                "state": np.ascontiguousarray(current),
                "next_state": np.ascontiguousarray(following),
                "boundary": np.ascontiguousarray(boundary[index]),
            }
        )
        current = following
    return records


def frame_from_transient_items(
    items: Sequence[Mapping[str, Any]],
    *,
    backend: str,
) -> pd.DataFrame:
    """Assemble backend-neutral EDA cases from validated runtime item values."""
    if not items or not isinstance(backend, str) or not backend:
        msg = "Transient item EDA requires non-empty items and a backend label."
        raise ValueError(msg)
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            msg = "Transient runtime item lacks metadata."
            raise TypeError(msg)
        case_id = str(metadata["simulation_case_id"])
        entry = grouped.setdefault(
            case_id,
            {
                "metadata": dict(metadata),
                "static": _numpy(item["static"]).astype(np.float32, copy=False),
                "scalars": _numpy(item["scalars"]).astype(np.float32, copy=False),
                "transitions": {},
            },
        )
        if not np.array_equal(entry["static"], _numpy(item["static"])) or not np.array_equal(entry["scalars"], _numpy(item["scalars"])):
            msg = f"Transient runtime items disagree on static evidence for {case_id!r}."
            raise ValueError(msg)
        for transition in _runtime_transitions(item):
            key = (transition["t_n"], transition["t_n_plus_1"])
            previous = entry["transitions"].get(key)
            if previous is not None:
                for name in ("state", "next_state", "boundary"):
                    if not np.array_equal(previous[name], transition[name]):
                        msg = f"Duplicate transient runtime transition disagrees for {case_id!r}."
                        raise ValueError(msg)
            else:
                entry["transitions"][key] = transition
    contract = transient_contract.TRANSIENT_STEP_CONTRACT
    state_names = tuple(field.name for field in contract.dynamic_state)
    static_names = tuple(field.name for field in contract.static_spatial_conditioning)
    boundary_names = tuple(field.name for field in contract.step_boundary_conditioning)
    scalar_names = tuple(field.name for field in contract.scalar_conditioning)
    rows: list[dict[str, Any]] = []
    case_ids: list[str] = []
    for case_id, entry in sorted(grouped.items()):
        transitions = [entry["transitions"][key] for key in sorted(entry["transitions"])]
        if not transitions:
            msg = f"Transient runtime item EDA requires at least one transition for {case_id!r}."
            raise ValueError(msg)
        runtime_times = np.asarray(
            [value for transition in transitions for value in (transition["t_n"], transition["t_n_plus_1"], transition["dt"])],
            dtype=np.float64,
        )
        if not np.isfinite(runtime_times).all():
            msg = f"Transient runtime time evidence is non-finite for {case_id!r}."
            raise ValueError(msg)
        time_tolerance = float(np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(runtime_times)))) * _FLOAT32_TIME_TOLERANCE_MULTIPLIER)
        for index, transition in enumerate(transitions):
            elapsed = transition["t_n_plus_1"] - transition["t_n"]
            if (
                elapsed <= 0.0
                or transition["dt"] <= 0.0
                or not math.isclose(
                    elapsed,
                    transition["dt"],
                    rel_tol=0.0,
                    abs_tol=time_tolerance,
                )
            ):
                msg = f"Transient runtime transition time evidence is invalid for {case_id!r}."
                raise ValueError(msg)
            if index == 0:
                continue
            previous = transitions[index - 1]
            if not math.isclose(
                previous["t_n_plus_1"],
                transition["t_n"],
                rel_tol=0.0,
                abs_tol=time_tolerance,
            ):
                msg = f"Transient runtime item EDA requires one contiguous transition chain for {case_id!r}."
                raise ValueError(msg)
            if not np.array_equal(previous["next_state"], transition["state"]):
                msg = f"Transient runtime state disagrees across adjacent transitions for {case_id!r}."
                raise ValueError(msg)
        state_times = np.asarray(
            [transitions[0]["t_n"], *(transition["t_n_plus_1"] for transition in transitions)],
            dtype=np.float64,
        )
        states = np.stack([transitions[0]["state"], *(transition["next_state"] for transition in transitions)])
        static = entry["static"]
        scalars = entry["scalars"]
        metadata = entry["metadata"]
        if static.shape[0] != len(static_names) or scalars.shape != (len(scalar_names),):
            msg = f"Transient runtime conditioning order is invalid for {case_id!r}."
            raise ValueError(msg)
        rows.append(
            {
                "state_trajectories": {name: np.ascontiguousarray(states[:, index]) for index, name in enumerate(state_names)},
                "static_fields": {name: np.ascontiguousarray(static[index]) for index, name in enumerate(static_names)},
                "boundary_intervals": {
                    name: np.asarray(
                        [transition["boundary"][index] for transition in transitions],
                        dtype=np.float32,
                    )
                    for index, name in enumerate(boundary_names)
                },
                "scalar_conditioning": {name: float(scalars[index]) for index, name in enumerate(scalar_names)},
                "schedule": None,
                "time": {
                    "regular_state_hours": state_times,
                    "valid_state_mask": np.ones(state_times.shape, dtype=bool),
                    "trajectory_length": int(state_times.size),
                    "transition_t_n_hours": np.asarray(
                        [transition["t_n"] for transition in transitions],
                        dtype=np.float64,
                    ),
                    "transition_t_n_plus_1_hours": np.asarray(
                        [transition["t_n_plus_1"] for transition in transitions],
                        dtype=np.float64,
                    ),
                    "transition_dt_hours": np.asarray(
                        [transition["dt"] for transition in transitions],
                        dtype=np.float64,
                    ),
                    "configured_horizon_hours": None,
                    "classification_tolerance_hours": time_tolerance,
                    "classification_tolerance_source": "float32_runtime_item_resolution",
                    "unit": contract.time_unit,
                },
                "exact_stop": None,
                "global_series": None,
                "final_status": None,
                "completion": None,
                "runtime": None,
                "meta": {
                    "case_id": case_id,
                    "case_input_id": metadata.get("case_input_id"),
                    "simulation_case_id": case_id,
                    "material_family": metadata.get("material_family"),
                    "dataset_role": metadata.get("split"),
                    "case_family": metadata.get("evaluation_regime"),
                    "simulation_profile": metadata.get("source_simulation_profile"),
                },
                "source": {
                    "dataset_id": metadata.get("dataset_id"),
                    "source_batch_id": metadata.get("source_batch_id"),
                    "backend": backend,
                },
            }
        )
        case_ids.append(case_id)
    task = domain.tasks.registry.get_task(_TRANSIENT_TASK_ID)
    frame = pd.DataFrame(rows, index=pd.Index(case_ids, name="sample_id"))
    frame.attrs.update(
        {
            "task_id": task.id,
            "task_contract_digest": task.contract_digest,
            "dataset_backend": backend,
            "analysis_representation": "transient_runtime_item_semantics",
            "loaded_case_count": len(frame),
            "available_case_count": len(frame),
            "total_discovered_case_count": len(frame),
            "failed_case_count": 0,
            "incomplete_case_count": 0,
            "exclusion_reasons": {},
            "field_names": state_names,
            "field_units": {field.name: field.unit for field in contract.dynamic_state},
            "spatial_shape": None if frame.empty else next(iter(frame.iloc[0]["state_trajectories"].values())).shape[1:],
        }
    )
    validate_transient_frame(frame)
    return frame
