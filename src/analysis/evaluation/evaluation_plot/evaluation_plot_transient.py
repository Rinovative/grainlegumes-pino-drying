"""Render sequence-aware transient-drying Evaluation evidence.

Responsibilities:
  - Compare reference, prediction, and signed-error fields at exact physical times
  - Plot physical state trajectories and exact grain-moisture derivations
  - Plot central rollout error across time, horizons, and reduction scopes
  - Present target, matched-compute, pipeline-degradation, and timing evidence

Design principles:
  - Dynamic states follow the shared EDA labels, units, and display conversions
  - Physical times are selected exactly and never nearest-neighbour substituted
  - Endpoint and cumulative reductions remain visually and semantically distinct

This module does NOT:
  - Load artifacts, run inference, refit scaling, or calculate speedup formulas
  - Alter steady Evaluation plots or silently fill unavailable evidence
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import domain
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_axis
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_STATE_ARRAY_RANK = 4
_SPATIAL_MAP_ERROR_COLUMN = 2
_MINIMUM_TRAJECTORY_TIMES = 2
_TARGET_MAGNITUDE_BIN_COUNT = 12
_TARGET_MAGNITUDE_POINT_LIMIT = 50_000
_GRAIN_MOISTURE_FIELD = "w_gr"
_MAX_ERROR_BATCH_INPUT_BYTES = 64 * 1024 * 1024
_ERROR_PAIR_MASK_BYTES = 17
_MIN_BAND_CASES = 2


def _value(record: Any, name: str) -> Any:
    """Read one field from a validated sequence object or mapping."""
    if isinstance(record, Mapping):
        if name not in record:
            msg = f"Transient plot record lacks {name!r}."
            raise ValueError(msg)
        return record[name]
    if not hasattr(record, name):
        msg = f"Transient plot record lacks {name!r}."
        raise ValueError(msg)
    return getattr(record, name)


def _material_label(value: Any) -> str:
    """Return one shared human material label from authoritative identity evidence."""
    if not isinstance(value, str) or not value:
        msg = "Transient plot material_family must be non-empty text."
        raise ValueError(msg)
    return display_labels.material_display_label(value)


def _material_role_label(
    frame_name: Any,
    material_family: Any,
    dataset_role: Any,
    *,
    include_frame: bool,
) -> str:
    """Return material-first role context with model identity only when needed."""
    material = _material_label(material_family)
    label = (
        display_labels.material_role_display_label(material_family, dataset_role)
        if isinstance(dataset_role, str) and dataset_role in {"id", "additional_evaluation"}
        else material
    )
    return f"{label} · {frame_name}" if include_frame else label


def _selected_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Return unique compatible stored and derived states in canonical order."""
    canonical = (*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD)
    requested = set(artifact.STATE_ORDER if fields is None else fields)
    if not requested or requested.difference(canonical):
        msg = "Transient state selection must contain known stored or derived states."
        raise ValueError(msg)
    return tuple(field for field in canonical if field in requested)


def _time_index(record: Any, physical_time: float | None) -> tuple[int, float]:
    """Resolve only an exact stored physical time."""
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        msg = "Transient plot times must be finite and strictly increasing."
        raise ValueError(msg)
    selected = float(times[-1]) if physical_time is None else float(physical_time)
    matches = np.flatnonzero(times == selected)
    if matches.size != 1:
        msg = f"Requested physical time {selected:g} is unavailable; valid times are {times.tolist()}."
        raise ValueError(msg)
    return int(matches[0]), selected


def _state_unit(field: str) -> str:
    """Return the exact physical unit for one persisted or derived state."""
    if field == _GRAIN_MOISTURE_FIELD:
        surface_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_surf")]
        internal_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_int")]
        if surface_unit != internal_unit:
            msg = "Surface and internal moisture units must agree for grain moisture."
            raise ValueError(msg)
        return surface_unit
    try:
        return artifact.STATE_UNITS[artifact.STATE_ORDER.index(field)]
    except ValueError as error:
        msg = f"Unknown transient state field {field!r}."
        raise ValueError(msg) from error


def _display_state_values(
    field: str,
    values: np.ndarray,
    *,
    error: bool,
) -> np.ndarray:
    """Apply the shared EDA display conversion without changing artifact arrays."""
    return field_labels.display_values(
        values,
        _state_unit(field),
        quantity_kind="difference" if error else "absolute",
    )


def _grain_moisture(record: Any, states: np.ndarray) -> np.ndarray:
    """Derive exact grain moisture from persisted states and material evidence."""
    if states.ndim != _STATE_ARRAY_RANK or states.shape[1] != len(artifact.STATE_ORDER):
        msg = "Grain-moisture trajectories require [time,4,Y,X] state arrays."
        raise ValueError(msg)
    scalars = np.asarray(_value(record, "scalar_conditioning"), dtype=np.float64)
    if scalars.shape != (len(artifact.SCALAR_ORDER),):
        msg = "Grain-moisture trajectories require exact scalar conditioning."
        raise ValueError(msg)
    fraction = scalars[artifact.SCALAR_ORDER.index("f_surf")]
    return domain.moisture.granular_water_content(
        states[:, artifact.STATE_ORDER.index("w_surf")],
        states[:, artifact.STATE_ORDER.index("w_int")],
        fraction,
    )


def plot_state_maps(
    record: Any,
    *,
    state_fields: Sequence[str] | None = None,
    physical_time: float | None = None,
    lock_scale: bool = True,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    reference_scales: Mapping[str, error_diagnostics.ReferenceScale | str] | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot reference, prediction, and one selected full-grid error mode."""
    fields = _selected_fields(state_fields)
    if not isinstance(lock_scale, bool):
        msg = "lock_scale must be boolean."
        raise TypeError(msg)
    index, selected_time = _time_index(record, physical_time)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK or reference.shape[1] != len(artifact.STATE_ORDER):
        msg = "Transient state maps require matching [time,4,Y,X] arrays."
        raise ValueError(msg)
    if static.shape != (len(artifact.STATIC_ORDER), *reference.shape[-2:]):
        msg = "Transient state maps require exact [7,Y,X] static conditioning."
        raise ValueError(msg)
    x_values, y_values = static[0], static[1]
    indices = display_grid.display_indices(reference.shape[-2:], display_spatial_stride)
    x_display = display_grid.display_sample(x_values, indices)
    y_display = display_grid.display_sample(y_values, indices)
    figure, axes = layout.map_subplots(rows=len(fields), columns=3)
    column_titles = ("Reference", "Prediction", error_mode)
    for row, field in enumerate(fields):
        if field == _GRAIN_MOISTURE_FIELD:
            raw_reference = _grain_moisture(record, reference)[index]
            raw_prediction = _grain_moisture(record, prediction)[index]
        else:
            channel = artifact.STATE_ORDER.index(field)
            raw_reference = reference[index, channel]
            raw_prediction = prediction[index, channel]
        if error_mode not in error_diagnostics.TRANSIENT_ERROR_MODES:
            msg = f"Transient error mode must be one of {error_diagnostics.TRANSIENT_ERROR_MODES}."
            raise ValueError(msg)
        physical_unit = _state_unit(field)
        scale = None if reference_scales is None else reference_scales.get(field)
        error_result = error_diagnostics.selected_error_map(
            raw_reference,
            raw_prediction,
            mode=error_mode,
            field=field,
            physical_unit=physical_unit,
            scale=scale,
            support=np.asarray(_value(record, "spatial_mask"), dtype=bool),
        )
        reference_display = _display_state_values(field, raw_reference, error=False)
        prediction_display = _display_state_values(field, raw_prediction, error=False)
        error_display = (
            None
            if error_result.map_values is None
            else error_result.map_values
            if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
            else _display_state_values(field, error_result.map_values, error=True)
        )
        values = (reference_display, prediction_display, error_display)
        finite_reference = reference_display[np.isfinite(reference_display)]
        finite_prediction = prediction_display[np.isfinite(prediction_display)]
        if not finite_reference.size:
            msg = "Transient reference map has no finite support."
            raise ValueError(msg)
        reference_norm = (
            layout.linear_norm(finite_reference, finite_prediction) if lock_scale and finite_prediction.size else layout.linear_norm(finite_reference)
        )
        prediction_norm = reference_norm if lock_scale or not finite_prediction.size else layout.linear_norm(finite_prediction)
        for column, (name, value) in enumerate(zip(column_titles, values, strict=True)):
            axis = axes[row, column]
            is_error = column == _SPATIAL_MAP_ERROR_COLUMN
            if value is None:
                axis.text(0.5, 0.5, f"Unavailable: {error_result.reason}", ha="center", va="center", wrap=True, transform=axis.transAxes)
                axis.set_title(name)
                axis.set_aspect("equal")
                continue
            if is_error:
                semantic = field_maps.FieldMapSemantic.SIGNED_ERROR if error_mode == "Signed error" else field_maps.FieldMapSemantic.ABSOLUTE_ERROR
                definition_values = np.asarray(value)[np.isfinite(value)]
                if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
                    definition_values = np.concatenate((definition_values, np.array((0.0, 1.0))))
                definition = field_maps.build_field_map(
                    definition_values,
                    semantic=semantic,
                    field=field,
                    unit=(
                        "1"
                        if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
                        else field_labels.display_unit(physical_unit, quantity_kind="difference")
                    ),
                    display_transform=error_mode,
                    locked_boundaries=np.linspace(0.0, 1.0, 12) if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR else None,
                    bound_policy="fixed 0..1 with extension" if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR else "full selected field",
                )
            image = axis.pcolormesh(
                x_display,
                y_display,
                np.ma.masked_invalid(display_grid.display_sample(np.asarray(value), indices)),
                shading="auto",
                cmap=(definition.colormap if is_error else visual_semantics.field_visual_semantics(field).colormap),
                norm=(definition.normalizer if is_error else (reference_norm if name == "Reference" else prediction_norm)),
            )
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(name)
            if not is_error and not np.isfinite(value).all():
                axis.text(0.02, 0.02, "Non-finite cells masked", transform=axis.transAxes, fontsize=8)
            layout.add_map_colorbar(
                figure,
                image,
                axis,
                label=(
                    "1 (train state_std; >1 extended)"
                    if is_error and error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
                    else field_labels.display_unit(physical_unit, quantity_kind="difference" if is_error else "absolute")
                ),
            )
        layout.add_channel_row_label(
            axes[row, 0],
            field_labels.field_label_with_unit(
                field,
                _state_unit(field),
                mathtext=True,
            ),
        )
    layout.apply_map_grid_axis_labels(axes, x_label="x [m]", y_label="y [m]")
    layout.set_suptitle_over_axes(
        figure,
        f"Transient state comparison at t={selected_time:g} h",
        axes.flat,
    )
    return figure


def _trajectory_values(
    record: Any,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact time, mask, reference, and prediction values for one field."""
    if field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Transient trajectory field is unsupported: {field!r}."
        raise ValueError(msg)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    mask = np.asarray(_value(record, "spatial_mask"))
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or mask.dtype != np.bool_
        or mask.shape != reference.shape[-2:]
        or not mask.any()
        or times.shape != (reference.shape[0],)
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
    ):
        msg = "Transient trajectories require aligned state arrays, mask, and physical times."
        raise ValueError(msg)
    if field == _GRAIN_MOISTURE_FIELD:
        reference_values = _grain_moisture(record, reference)
        prediction_values = _grain_moisture(record, prediction)
    else:
        channel = artifact.STATE_ORDER.index(field)
        reference_values = reference[:, channel]
        prediction_values = prediction[:, channel]
    return times, mask, reference_values, prediction_values


def _union_record_times(records: Sequence[Any], *, exclude_origin: bool) -> np.ndarray:
    """Return every exact supported physical time, without filling short tails."""
    admitted = tuple(records)
    if not admitted:
        msg = "Transient trajectory summaries require at least one record."
        raise ValueError(msg)
    support: set[float] = set()
    for record in admitted:
        values = np.asarray(_value(record, "physical_times"), dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0.0):
            msg = "Transient physical times must be finite and strictly increasing."
            raise ValueError(msg)
        support.update(float(value) for value in values[1 if exclude_origin else 0 :])
    times = np.asarray(sorted(support), dtype=np.float64)
    if not times.size:
        msg = "Transient trajectory records contain no required exact physical times."
        raise ValueError(msg)
    return times


@dataclass(frozen=True, slots=True)
class _ExactTimeSummary:
    """One exact-time cohort summary with separate presence and invalid counts."""

    times: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    count: np.ndarray
    invalid_count: np.ndarray


def _aggregate_exact_time_values(rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> _ExactTimeSummary:
    """Pool only present exact times; one nonfinite contributor voids that score."""
    by_time: dict[float, list[float]] = {}
    for times, values in rows:
        if times.ndim != 1 or values.shape != times.shape or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
            msg = "Exact-time aggregate rows require aligned strictly increasing times and values."
            raise ValueError(msg)
        for time, value in zip(times, values, strict=True):
            by_time.setdefault(float(time), []).append(float(value))
    if not by_time:
        msg = "Exact-time aggregate requires at least one supported physical time."
        raise ValueError(msg)
    times = np.asarray(sorted(by_time), dtype=np.float64)
    mean = np.full(times.shape, np.nan, dtype=np.float64)
    lower = np.full(times.shape, np.nan, dtype=np.float64)
    upper = np.full(times.shape, np.nan, dtype=np.float64)
    count = np.zeros(times.shape, dtype=np.int64)
    invalid = np.zeros(times.shape, dtype=np.int64)
    for index, time in enumerate(times):
        values = np.asarray(by_time[float(time)], dtype=np.float64)
        count[index] = values.size
        invalid[index] = int(np.count_nonzero(~np.isfinite(values)))
        if invalid[index]:
            continue
        mean[index] = float(np.mean(values))
        if values.size >= _MIN_BAND_CASES:
            lower[index], upper[index] = np.quantile(values, (0.1, 0.9))
    return _ExactTimeSummary(times, mean, lower, upper, count, invalid)


def plot_state_trajectory_summary(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
) -> Figure:
    """Plot material-aware aggregate or single-case trajectories on exact times."""
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not label or not records for label, records in admitted.items()):
        msg = "Trajectory summaries require labelled non-empty record groups."
        raise ValueError(msg)
    all_times = np.concatenate([_union_record_times(records, exclude_origin=False) for records in admitted.values()])
    display_time = time_axis.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(9.2, max(4.8, 3.8 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        support_axis = axis.twinx() if row == 0 else None
        for series_index, (label, records) in enumerate(admitted.items()):
            reference_rows: list[tuple[np.ndarray, np.ndarray]] = []
            prediction_rows: list[tuple[np.ndarray, np.ndarray]] = []
            for record in records:
                record_times, mask, reference, prediction = _trajectory_values(record, field)
                reference_display = _display_state_values(field, reference, error=False)
                prediction_display = _display_state_values(field, prediction, error=False)
                reference_rows.append((record_times, np.mean(reference_display[:, mask], axis=1)))
                prediction_rows.append((record_times, np.mean(prediction_display[:, mask], axis=1)))
            reference_summary = _aggregate_exact_time_values(reference_rows)
            prediction_summary = _aggregate_exact_time_values(prediction_rows)
            plotted_time = display_time.values(reference_summary.times)
            color = colors(series_index % colors.N)
            axis.plot(
                plotted_time,
                reference_summary.mean,
                color=color,
                linewidth=2.0,
                label=f"{label} · Reference",
            )
            axis.plot(
                plotted_time,
                prediction_summary.mean,
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=f"{label} · Prediction",
            )
            axis.fill_between(plotted_time, reference_summary.lower, reference_summary.upper, color=color, alpha=0.08)
            axis.fill_between(plotted_time, prediction_summary.lower, prediction_summary.upper, color=color, alpha=0.08)
            if support_axis is not None:
                support_axis.step(plotted_time, reference_summary.count, where="mid", color=color, alpha=0.55, label=f"{label} n(t)")
                unavailable = (reference_summary.invalid_count + prediction_summary.invalid_count) > 0
                if unavailable.any():
                    support_axis.scatter(plotted_time[unavailable], reference_summary.count[unavailable], color=color, marker="x", s=28)
        display_time.configure(axis)
        axis.set_ylabel(field_labels.field_label_with_unit(field, _state_unit(field), mathtext=True))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
        if support_axis is not None:
            support_axis.set_ylabel("Exact cases n(t); x marker = non-finite unavailable")
            support_axis.set_ylim(bottom=0)
            support_axis.legend(loc="lower right", frameon=False, fontsize="small")
    figure.suptitle("Reference vs prediction trajectories — exact-time spatial means; bands require n(t) ≥ 2")
    return figure


def _normalized_error_trajectory(
    record: Any,
    *,
    scaling_state: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    """Return saved-scale per-channel RMSE at exact post-origin times."""
    selected_fields = tuple(fields)
    if not selected_fields or set(selected_fields).difference(artifact.STATE_ORDER):
        msg = "Normalized error trajectories require stored transient states."
        raise ValueError(msg)
    scaling = TransientScalingArtifact.from_state_dict(scaling_state)
    reference = torch.as_tensor(np.asarray(_value(record, "reference_states"), dtype=np.float32))
    prediction = torch.as_tensor(np.asarray(_value(record, "predicted_states"), dtype=np.float32))
    mask = np.asarray(_value(record, "spatial_mask"))
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK or mask.dtype != np.bool_:
        msg = "Normalized error trajectories require aligned state arrays and a boolean mask."
        raise ValueError(msg)
    normalized_error = (scaling.encode_state(prediction) - scaling.encode_state(reference)).detach().cpu().numpy()
    result = {
        field: np.sqrt(
            np.mean(
                np.square(normalized_error[1:, artifact.STATE_ORDER.index(field), mask], dtype=np.float64),
                axis=1,
            )
        )
        for field in selected_fields
    }
    return times[1:], result


def plot_error_over_physical_time(
    series: Mapping[str, Sequence[Any]],
    *,
    scaling_states: Mapping[str, Mapping[str, Any]],
    state_fields: Sequence[str],
) -> Figure:
    """Plot saved-scale channel errors over exact physical time by context."""
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not fields or not admitted or set(admitted) != set(scaling_states):
        msg = "Error-over-time series require stored fields and matching scaling contexts."
        raise ValueError(msg)
    all_times = np.concatenate([_union_record_times(records, exclude_origin=True) for records in admitted.values()])
    display_time = time_axis.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(9.2, max(4.8, 3.5 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        support_axis = axis.twinx() if row == 0 else None
        for series_index, (label, records) in enumerate(admitted.items()):
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            for record in records:
                record_times, values = _normalized_error_trajectory(
                    record,
                    scaling_state=scaling_states[label],
                    fields=fields,
                )
                rows.append((record_times, values[field]))
            summary = _aggregate_exact_time_values(rows)
            plotted_time = display_time.values(summary.times)
            color = colors(series_index % colors.N)
            axis.plot(plotted_time, summary.mean, color=color, linewidth=2.0, label=label)
            axis.fill_between(plotted_time, summary.lower, summary.upper, color=color, alpha=0.1)
            if support_axis is not None:
                support_axis.step(plotted_time, summary.count, where="mid", color=color, alpha=0.55, label=f"{label} n(t)")
                unavailable = summary.invalid_count > 0
                if unavailable.any():
                    support_axis.scatter(plotted_time[unavailable], summary.count[unavailable], color=color, marker="x", s=28)
        display_time.configure(axis)
        axis.set_ylabel(f"{field_labels.field_label(field)} normalized RMSE [1]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
        if support_axis is not None:
            support_axis.set_ylabel("Exact cases n(t); x marker = non-finite unavailable")
            support_axis.set_ylim(bottom=0)
            support_axis.legend(loc="lower right", frameon=False, fontsize="small")
    figure.suptitle("Error over physical time — exact-time case means; 10th-90th bands require n(t) ≥ 2")
    return figure


def plot_central_error_vs_time(
    record: Any,
    *,
    scaling_state: Mapping[str, Any],
) -> Figure:
    """Plot central and per-channel normalized errors against physical time."""
    scaling = TransientScalingArtifact.from_state_dict(scaling_state)
    reference = torch.as_tensor(
        np.asarray(_value(record, "reference_states"), dtype=np.float32),
        dtype=torch.float32,
    )
    prediction = torch.as_tensor(
        np.asarray(_value(record, "predicted_states"), dtype=np.float32),
        dtype=torch.float32,
    )
    mask = np.asarray(_value(record, "spatial_mask"))
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK or mask.dtype != np.bool_:
        msg = "Transient error curves require aligned state arrays and a boolean mask."
        raise ValueError(msg)
    normalized_error = (scaling.encode_state(prediction) - scaling.encode_state(reference)).detach().cpu().numpy()
    selected = normalized_error[1:, :, mask]
    channel_rmse = np.sqrt(np.mean(np.square(selected, dtype=np.float64), axis=2))
    weights = np.asarray((1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0))
    central = channel_rmse @ weights
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)[1:]
    display_time = time_axis.physical_time_display(times, preferred_unit="auto")
    plotted_time = display_time.values(times)
    figure, axis = plt.subplots(figsize=(9.5, 5.5), layout=None)
    figure.subplots_adjust(left=0.11, right=0.78, bottom=0.13, top=0.92)
    axis.plot(
        plotted_time,
        central,
        marker="o",
        linewidth=2.2,
        label="Central Drying metric",
    )
    for channel, field in enumerate(artifact.STATE_ORDER):
        axis.plot(
            plotted_time,
            channel_rmse[:, channel],
            alpha=0.75,
            label=field,
        )
    display_time.configure(axis)
    axis.set_ylabel("Normalized RMSE [1]")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _aggregate_snapshot_evidence(  # noqa: PLR0915 -- one bounded physical snapshot reduction
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
    compute_device: str = "cpu",
    device_evidence: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool exact-time case evidence without stacking complete trajectories."""
    if state_field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Aggregate transient snapshot field is unsupported: {state_field!r}."
        raise ValueError(msg)
    admitted = tuple(records)
    if not admitted:
        msg = "Aggregate transient snapshots require at least one complete-rollout record."
        raise ValueError(msg)
    error_sum: np.ndarray | None = None
    absolute_error_sum: np.ndarray | None = None
    error_square_sum: np.ndarray | None = None
    counts: np.ndarray | None = None
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    reference_means: list[float] = []
    prediction_means: list[float] = []
    cuda_reference: list[np.ndarray] = []
    cuda_prediction: list[np.ndarray] = []
    cuda_masks: list[np.ndarray] = []
    selected_device: compute.AnalysisDevice | None = None
    batch_cases = 1
    routes: set[str] = set()

    def flush_cuda_batch() -> None:
        nonlocal error_sum, absolute_error_sum, error_square_sum, counts, selected_device
        if not cuda_reference:
            return
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        reduced = compute.reduce_error_batch(
            np.stack(cuda_reference),
            np.stack(cuda_prediction),
            np.stack(cuda_masks),
            requested=compute_device,
        )
        error_sum += reduced.signed_sum
        absolute_error_sum += reduced.absolute_sum
        error_square_sum += reduced.squared_sum
        counts += reduced.support_count
        routes.add(reduced.device.label)
        selected_device = reduced.device
        cuda_reference.clear()
        cuda_prediction.clear()
        cuda_masks.clear()

    for record in admitted:
        if _value(record, "mode") != "autonomous_full" or _value(record, "requested_horizon") != "full":
            msg = "Aggregate transient snapshots require complete autonomous rollouts only."
            raise ValueError(msg)
        time_index, _selected_time = _time_index(record, physical_time)
        reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
        prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
        mask = np.asarray(_value(record, "spatial_mask"))
        static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
        if (
            reference.shape != prediction.shape
            or reference.ndim != _STATE_ARRAY_RANK
            or reference.shape[1] != len(artifact.STATE_ORDER)
            or mask.dtype != np.bool_
            or mask.shape != reference.shape[-2:]
            or not mask.any()
            or static.shape != (len(artifact.STATIC_ORDER), *mask.shape)
        ):
            msg = "Aggregate transient snapshots require aligned state, mask, and coordinate arrays."
            raise ValueError(msg)
        current_x = static[0]
        current_y = static[1]
        if x_values is None:
            x_values = current_x
            y_values = current_y
            error_sum = np.zeros(mask.shape, dtype=np.float64)
            absolute_error_sum = np.zeros(mask.shape, dtype=np.float64)
            error_square_sum = np.zeros(mask.shape, dtype=np.float64)
            counts = np.zeros(mask.shape, dtype=np.int64)
            batch_cases = max(1, _MAX_ERROR_BATCH_INPUT_BYTES // (mask.size * _ERROR_PAIR_MASK_BYTES))
            selected_device = compute.resolve_analysis_device(compute_device, elements=min(len(admitted), batch_cases) * mask.size)
        elif y_values is None or not np.array_equal(current_x, x_values) or not np.array_equal(current_y, y_values):
            msg = "Aggregate transient spatial maps require one exact shared coordinate grid."
            raise ValueError(msg)
        if state_field == _GRAIN_MOISTURE_FIELD:
            reference_snapshot = _grain_moisture(record, reference)[time_index]
            prediction_snapshot = _grain_moisture(record, prediction)[time_index]
        else:
            channel = artifact.STATE_ORDER.index(state_field)
            reference_snapshot = reference[time_index, channel]
            prediction_snapshot = prediction[time_index, channel]
        if not np.isfinite(reference_snapshot[mask]).all() or not np.isfinite(prediction_snapshot[mask]).all():
            msg = "Aggregate transient snapshots require finite values inside every spatial mask."
            raise ValueError(msg)
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        if selected_device is not None and selected_device.backend == "cuda":
            cuda_reference.append(reference_snapshot)
            cuda_prediction.append(prediction_snapshot)
            cuda_masks.append(mask)
            if len(cuda_reference) >= batch_cases:
                flush_cuda_batch()
        else:
            error = prediction_snapshot - reference_snapshot
            error_sum[mask] += error[mask]
            absolute_error_sum[mask] += np.abs(error[mask])
            error_square_sum[mask] += np.square(error[mask], dtype=np.float64)
            counts[mask] += 1
            if selected_device is not None:
                routes.add(selected_device.label)
        reference_means.append(float(np.mean(reference_snapshot[mask], dtype=np.float64)))
        prediction_means.append(float(np.mean(prediction_snapshot[mask], dtype=np.float64)))
    if x_values is None or y_values is None or error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
        msg = "Aggregate transient snapshot evidence was not initialized."
        raise RuntimeError(msg)
    flush_cuda_batch()
    if device_evidence is not None:
        device_evidence["route"] = "; ".join(sorted(routes))
    if not all(np.isfinite(values[counts > 0]).all() for values in (error_sum, absolute_error_sum, error_square_sum)):
        msg = "Aggregate transient error statistics are non-finite on required support."
        raise ValueError(msg)
    mean_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_absolute_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_square_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    np.divide(error_sum, counts, out=mean_error, where=counts > 0)
    np.divide(absolute_error_sum, counts, out=mean_absolute_error, where=counts > 0)
    np.divide(error_square_sum, counts, out=mean_square_error, where=counts > 0)
    error_variance = np.maximum(mean_square_error - np.square(mean_error), 0.0)
    return (
        x_values,
        y_values,
        mean_absolute_error,
        np.sqrt(error_variance),
        np.asarray(reference_means, dtype=np.float64),
        np.asarray(prediction_means, dtype=np.float64),
    )


def _labelled_record_series(
    value: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    allow_empty_contexts: bool = False,
) -> dict[str, tuple[Any, ...]]:
    """Normalize one sequence or labelled multi-context aggregate."""
    if isinstance(value, Mapping):
        series: dict[str, tuple[Any, ...]] = {}
        for label, records in value.items():
            if not isinstance(label, str) or not label:
                msg = "Transient aggregate context labels must be non-empty text."
                raise ValueError(msg)
            admitted = tuple(records)
            if not admitted and not allow_empty_contexts:
                msg = "Transient aggregate contexts must contain at least one record."
                raise ValueError(msg)
            series[label] = admitted
    else:
        admitted = tuple(value)
        if not admitted:
            msg = "Transient aggregate plots require at least one record."
            raise ValueError(msg)
        series = {"": admitted}
    if not series or not any(series.values()):
        msg = "Transient aggregate plots require at least one context."
        raise ValueError(msg)
    return series


def plot_aggregate_error_maps(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
    statistic: str,
    compute_device: str = "cpu",
) -> Figure:
    """Plot exact-time mean or standard-deviation error maps by context."""
    fields = _selected_fields(state_fields)
    if statistic not in {"mean", "std"}:
        msg = "Aggregate error-map statistic must be 'mean' or 'std'."
        raise ValueError(msg)
    series = _labelled_record_series(records, allow_empty_contexts=True)
    figure, axes = layout.map_subplots(
        rows=len(fields),
        columns=len(series),
    )
    total_cases = 0
    routes: set[str] = set()
    for column, (context, current_records) in enumerate(series.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
            if not current_records:
                axis = axes[row, column]
                axis.text(0.5, 0.5, "No case at this exact time", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    axis.set_title(context)
                continue
            device_evidence: dict[str, str] = {}
            (
                x_values,
                y_values,
                mean_error,
                std_error,
                _reference_means,
                _prediction_means,
            ) = _aggregate_snapshot_evidence(
                current_records,
                state_field=field,
                physical_time=physical_time,
                compute_device=compute_device,
                device_evidence=device_evidence,
            )
            routes.update(device_evidence.values())
            raw_values = mean_error if statistic == "mean" else std_error
            displayed = _display_state_values(field, raw_values, error=True)
            axis = axes[row, column]
            image = axis.pcolormesh(
                x_values,
                y_values,
                displayed,
                shading="auto",
                cmap=visual_semantics.field_visual_semantics(
                    field,
                    role="absolute_error",
                ).colormap,
                norm=layout.linear_norm(displayed),
            )
            axis.set_aspect("equal")
            if row == 0 and context:
                axis.set_title(context)
            layout.add_map_colorbar(
                figure,
                image,
                axis,
                label=field_labels.display_unit(
                    _state_unit(field),
                    quantity_kind="difference",
                ),
            )
            if column == 0:
                layout.add_channel_row_label(
                    axis,
                    field_labels.field_label_with_unit(
                        field,
                        _state_unit(field),
                        mathtext=True,
                    ),
                )
    layout.apply_map_grid_axis_labels(
        axes,
        x_label="x [m]",
        y_label="y [m]",
    )
    statistic_label = "Mean absolute" if statistic == "mean" else "Standard deviation of signed"
    layout.set_suptitle_over_axes(
        figure,
        (f"{statistic_label} error at t={physical_time:g} h — {total_cases} case-context observations\n" + "; ".join(sorted(routes))),
        axes.flat,
    )
    return figure


def plot_predicted_vs_reference_channels(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
) -> Figure:
    """Plot exact-time case spatial means by channel and material context."""
    fields = _selected_fields(state_fields)
    series = _labelled_record_series(records, allow_empty_contexts=True)
    figure, raw_axes = plt.subplots(
        len(fields),
        len(series),
        figsize=(
            max(7.2, 5.8 * len(series)),
            max(5.2, 4.6 * len(fields)),
        ),
        layout="constrained",
        squeeze=False,
    )
    total_cases = 0
    for column, (context, current_records) in enumerate(series.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
            if not current_records:
                axis = raw_axes[row, column]
                axis.text(0.5, 0.5, "No case at this exact time", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    axis.set_title(context)
                continue
            (
                _x,
                _y,
                _mean_error,
                _std_error,
                reference_means,
                prediction_means,
            ) = _aggregate_snapshot_evidence(
                current_records,
                state_field=field,
                physical_time=physical_time,
            )
            reference_display = _display_state_values(
                field,
                reference_means,
                error=False,
            )
            prediction_display = _display_state_values(
                field,
                prediction_means,
                error=False,
            )
            lower = float(
                min(
                    np.min(reference_display),
                    np.min(prediction_display),
                )
            )
            upper = float(
                max(
                    np.max(reference_display),
                    np.max(prediction_display),
                )
            )
            if lower == upper:
                padding = max(abs(lower) * 0.05, 1.0)
                lower -= padding
                upper += padding
            axis = raw_axes[row, column]
            axis.scatter(
                reference_display,
                prediction_display,
                s=55,
                alpha=0.8,
            )
            axis.plot(
                (lower, upper),
                (lower, upper),
                color="black",
                linestyle="--",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            label = field_labels.field_label_with_unit(
                field,
                _state_unit(field),
                mathtext=True,
            )
            axis.set_xlabel(f"Reference case spatial mean — {label}")
            axis.set_ylabel(f"Predicted case spatial mean — {label}")
            if row == 0 and context:
                axis.set_title(context)
            axis.grid(alpha=0.25)
    figure.suptitle(f"Prediction versus reference at t={physical_time:g} h — {total_cases} case-context observations")
    return figure


def plot_error_distributions(
    case_frame: pd.DataFrame,
    *,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Show complete-rollout case-error distributions by stored channel."""
    selected = _full_autonomous_cumulative(case_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient error distributions require at least one stored channel metric."
        raise ValueError(msg)
    values = []
    for field in fields:
        column = f"normalized_rmse_{field}"
        if column not in selected:
            msg = f"Transient error distributions lack {column!r}."
            raise ValueError(msg)
        current = selected[column].to_numpy(dtype=np.float64)
        if not np.isfinite(current).all():
            msg = "Transient case errors must be finite."
            raise ValueError(msg)
        values.append(current)
    figure, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    axis.boxplot(values, tick_labels=tuple(field_labels.field_label(field) for field in fields), showfliers=True)
    axis.set_ylabel("Normalized RMSE [1]")
    axis.set_title("Full-rollout case error distributions")
    axis.grid(axis="y", alpha=0.25)
    return figure


def plot_error_vs_target_magnitude(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
) -> Figure:
    """Plot bounded trajectory-element error trends against target magnitude."""
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not records for records in admitted.values()):
        msg = "Target-magnitude error trends require labelled non-empty record groups."
        raise ValueError(msg)
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(8.8, max(4.8, 3.6 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        for series_index, (label, records) in enumerate(admitted.items()):
            target_chunks: list[np.ndarray] = []
            error_chunks: list[np.ndarray] = []
            for record in records:
                _times, mask, reference, prediction = _trajectory_values(record, field)
                target = np.abs(_display_state_values(field, reference[1:, mask], error=False).ravel())
                error = np.abs(_display_state_values(field, prediction[1:, mask] - reference[1:, mask], error=True).ravel())
                stride = max(1, int(np.ceil(target.size / _TARGET_MAGNITUDE_POINT_LIMIT)))
                target_chunks.append(target[::stride])
                error_chunks.append(error[::stride])
            targets = np.concatenate(target_chunks)
            errors = np.concatenate(error_chunks)
            edges = np.unique(np.quantile(targets, np.linspace(0.0, 1.0, _TARGET_MAGNITUDE_BIN_COUNT + 1)))
            if len(edges) < _MINIMUM_TRAJECTORY_TIMES:
                centers = np.asarray([float(np.median(targets))])
                medians = np.asarray([float(np.median(errors))])
            else:
                bins = np.clip(np.digitize(targets, edges[1:-1], right=True), 0, len(edges) - 2)
                centers = np.asarray([float(np.median(targets[bins == index])) for index in range(len(edges) - 1) if np.any(bins == index)])
                medians = np.asarray([float(np.median(errors[bins == index])) for index in range(len(edges) - 1) if np.any(bins == index)])
            axis.plot(
                centers,
                medians,
                marker="o",
                linewidth=2.0,
                color=colors(series_index % colors.N),
                label=label,
            )
        unit = field_labels.display_unit(_state_unit(field), quantity_kind="difference")
        field_name = field_labels.field_label(field, mathtext=True)
        axis.set_xlabel(f"|Reference {field_name}| [{unit}]")
        axis.set_ylabel(f"Median absolute error [{unit}]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    figure.suptitle("Error vs |GT| magnitude — all stored post-origin times; deterministic bounded bins")
    return figure


def maximum_error_physical_time(
    record: Any,
    *,
    state_fields: Sequence[str],
) -> float:
    """Return the stored post-origin time with largest scale-normalized field error."""
    fields = _selected_fields(state_fields)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    mask = np.asarray(_value(record, "spatial_mask"))
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or mask.dtype != np.bool_
        or mask.shape != reference.shape[-2:]
        or times.shape != (reference.shape[0],)
        or len(times) < _MINIMUM_TRAJECTORY_TIMES
    ):
        msg = "Error-maximizing time requires aligned non-empty transient evidence."
        raise ValueError(msg)
    scores = np.zeros(len(times) - 1, dtype=np.float64)
    for field in fields:
        if field == _GRAIN_MOISTURE_FIELD:
            reference_values = _grain_moisture(record, reference)
            prediction_values = _grain_moisture(record, prediction)
        else:
            channel = artifact.STATE_ORDER.index(field)
            reference_values = reference[:, channel]
            prediction_values = prediction[:, channel]
        selected_reference = reference_values[:, mask]
        selected_error = prediction_values[:, mask] - selected_reference
        scale = max(float(np.std(selected_reference, dtype=np.float64)), np.finfo(np.float64).eps)
        scores += np.sqrt(np.mean(np.square(selected_error[1:], dtype=np.float64), axis=1)) / scale
    return float(times[1 + int(np.argmax(scores))])


def _full_autonomous_cumulative(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the comparable complete-rollout cumulative rows."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
        msg = "Transient aggregate plot requires complete-rollout summary columns."
        raise ValueError(msg)
    selected = frame.loc[(frame["mode"] == "autonomous_full") & (frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")].copy()
    if selected.empty:
        msg = "Transient aggregate plot requires full-autonomous cumulative evidence."
        raise ValueError(msg)
    return selected


def plot_channel_error(
    summary_frame: pd.DataFrame,
    *,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Compare complete-rollout normalized RMSE for selected stored states."""
    selected = _full_autonomous_cumulative(summary_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient channel-error plots require at least one stored state."
        raise ValueError(msg)
    columns = tuple(f"normalized_rmse_{field}" for field in fields)
    if any(column not in selected for column in columns):
        msg = "Transient channel-error plot lacks per-state normalized RMSE."
        raise ValueError(msg)
    labels = tuple(field_labels.field_label(field) for field in fields)
    positions = np.arange(len(columns), dtype=np.float64)
    width = 0.8 / len(selected)
    figure, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    role_count = selected["dataset_role"].nunique() if "dataset_role" in selected else 0
    include_frame = selected["frame"].nunique() > role_count
    for offset, (_, row) in enumerate(selected.iterrows()):
        values = row.loc[list(columns)].to_numpy(dtype=np.float64)
        axis.bar(
            positions - 0.4 + width / 2.0 + offset * width,
            values,
            width=width,
            label=_material_role_label(
                row["frame"],
                row["material_family"],
                row.get("dataset_role"),
                include_frame=include_frame,
            ),
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Normalized RMSE [1]")
    axis.set_title("Full-rollout per-state error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    return figure


def plot_horizon_error(summary_frame: pd.DataFrame) -> Figure:
    """Plot rollout-index and physical-time views without conflating axes."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
        "elapsed_physical_time_median",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Horizon-error plot requires a transient dataset summary frame."
        raise ValueError(msg)
    selected = summary_frame.loc[summary_frame["mode"] == "rolling_origin"].copy()
    if selected.empty:
        msg = "Horizon-error plot requires rolling-origin summaries."
        raise ValueError(msg)
    horizons = tuple(
        sorted(
            selected["requested_horizon"].unique(),
            key=lambda value: float("inf") if value == "full" else int(value),
        )
    )
    horizon_positions = {value: position for position, value in enumerate(horizons)}
    physical_times = selected["elapsed_physical_time_median"].to_numpy(dtype=np.float64)
    display_time = time_axis.physical_time_display(
        physical_times,
        preferred_unit="auto",
    )
    figure, axes_grid, legend_axis = layout.subplots_with_legend_column(
        rows=1,
        columns=2,
        column_width=5.2,
        row_height=5.0,
        legend_width=3.0,
    )
    axes = axes_grid[0]
    handles = []
    role_count = selected["dataset_role"].nunique() if "dataset_role" in selected else 0
    include_frame = selected["frame"].nunique() > role_count
    role_by_context = (
        selected.groupby(["frame", "material_family"], sort=False)["dataset_role"].first().to_dict() if "dataset_role" in selected else {}
    )
    for (frame, material, scope), group in selected.groupby(
        ["frame", "material_family", "scope"],
        sort=False,
    ):
        ordered = group.assign(_position=group["requested_horizon"].map(horizon_positions)).sort_values("_position")
        label = f"{_material_role_label(frame, material, role_by_context.get((frame, material)), include_frame=include_frame)} — {scope}"
        (handle,) = axes[0].plot(
            ordered["_position"],
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            label=label,
        )
        handles.append(handle)
        axes[1].plot(
            display_time.values(ordered["elapsed_physical_time_median"]),
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            color=handle.get_color(),
        )
    axes[0].set_xticks(
        np.arange(len(horizons)),
        [str(value) for value in horizons],
    )
    axes[0].set_xlabel("Requested horizon [transitions]")
    display_time.configure(axes[1])
    axes[1].set_xlabel(f"Median elapsed physical time [{display_time.unit}]")
    for axis in axes:
        axis.set_ylabel("Normalized Drying group macro RMSE [1]")
        axis.grid(axis="y", alpha=0.25)
    legend_axis.legend(handles=handles, loc="upper left", frameon=False)
    return figure


def plot_endpoint_vs_cumulative(summary_frame: pd.DataFrame) -> Figure:
    """Compare endpoint and cumulative central error without conflating scopes."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Endpoint comparison requires transient summary columns."
        raise ValueError(msg)
    pivot = summary_frame.pivot_table(
        index=["frame", "material_family", "mode", "requested_horizon"],
        columns="scope",
        values="normalized_drying_group_macro_rmse",
        aggfunc="first",
    ).dropna(subset=["cumulative", "endpoint"])
    if pivot.empty:
        msg = "Endpoint comparison requires paired cumulative and endpoint evidence."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    for frame_name, group in pivot.groupby(level="frame", sort=False):
        axis.scatter(group["cumulative"], group["endpoint"], label=str(frame_name))
    axis.legend()
    maximum = float(np.max(pivot[["cumulative", "endpoint"]].to_numpy()))
    axis.plot((0.0, maximum), (0.0, maximum), linestyle="--", color="black")
    axis.set_xlabel("Cumulative central error [1]")
    axis.set_ylabel("Endpoint central error [1]")
    axis.grid(alpha=0.25)
    return figure


def plot_target_time(records: Sequence[Any] | Mapping[str, Sequence[Any]]) -> Figure:
    """Plot reached target pairs by saved model and retain censored/unavailable counts."""
    groups = records if isinstance(records, Mapping) else {"Selected model": records}
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    notices = []
    all_times: list[float] = []
    for label, selected in groups.items():
        reference: list[float] = []
        prediction: list[float] = []
        censored = unavailable = 0
        for record in selected:
            target = _value(record, "target")
            if not isinstance(target, Mapping):
                msg = "Transient target plot requires mapping evidence."
                raise TypeError(msg)
            if target.get("reference_available") is not True or target.get("predicted_available") is not True:
                unavailable += 1
                continue
            ref_time = target.get("reference_time_to_target")
            pred_time = target.get("predicted_time_to_target")
            if isinstance(ref_time, Real) and isinstance(pred_time, Real):
                reference.append(float(ref_time))
                prediction.append(float(pred_time))
            else:
                censored += 1
        pairs[str(label)] = (reference, prediction)
        all_times.extend((*reference, *prediction))
        notices.append(f"{label}: paired reached {len(reference)}, censored {censored}, unavailable {unavailable}")
    figure, axis = plt.subplots(figsize=(8.5, 6.0), layout="constrained")
    display_time = time_axis.physical_time_display(tuple(all_times) if all_times else (0.0,), preferred_unit="auto")
    for label, (reference, prediction) in pairs.items():
        if reference:
            axis.scatter(display_time.values(reference), display_time.values(prediction), label=label)
    if all_times:
        lower, upper = display_time.values((min(all_times), max(all_times)))
        axis.plot((lower, upper), (lower, upper), linestyle="--", color="black")
        axis.legend()
    display_time.configure(axis)
    display_time.configure(axis, dimension="y")
    axis.set_xlabel(f"Reference time to target [{display_time.unit}]")
    axis.set_ylabel(f"Predicted time to target [{display_time.unit}]")
    axis.set_title("\n".join(notices), fontsize="small")
    axis.grid(alpha=0.25)
    return figure


def plot_training_performance_vs_compute(performance: pd.DataFrame) -> Figure:
    """Plot admitted legacy or receipt-matched training arms in their actual clock."""
    required = {
        "comparison_arm",
        "material_family",
        "optimizer_device_compute",
        "normalized_drying_group_macro_rmse",
        "checkpoint_role",
        "mode",
        "requested_horizon",
        "scope",
    }
    if not isinstance(performance, pd.DataFrame) or not required.issubset(performance):
        msg = "Matched-compute plot requires arm, compute, metric, checkpoint, mode, horizon, and scope."
        raise ValueError(msg)
    arms = set(performance["comparison_arm"])
    if not arms or not arms.issubset({"A0", "A1", "B"}):
        msg = "Matched-compute plot requires the A0/A1/B arm family."
        raise ValueError(msg)
    clocks = set(performance.loc[performance["comparison_arm"].isin(("A1", "B")), "compute_clock"]) if "compute_clock" in performance else set()
    if clocks != {"train_seconds"}:
        msg = "Matched A1/B plot requires persisted committed train_seconds for both arms."
        raise ValueError(msg)
    selected = performance.loc[performance["scope"] == "cumulative"].copy()
    if selected.empty:
        msg = "Matched-compute plot requires cumulative Evaluation evidence."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(10.0, 6.0), layout="constrained")
    for (material, mode, horizon), group in selected.groupby(
        ["material_family", "mode", "requested_horizon"],
        sort=False,
    ):
        if group["comparison_arm"].duplicated().any():
            msg = "Matched-compute series requires at most one value per arm, material, mode, and horizon."
            raise ValueError(msg)
        ordered = group.assign(_arm_order=group["comparison_arm"].map({"A0": 0, "A1": 1, "B": 2})).sort_values("_arm_order")
        axis.plot(
            ordered["training_compute"] if "training_compute" in ordered else ordered["optimizer_device_compute"],
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            label=f"{_material_label(material)} · {mode} / {horizon}",
        )
        for row in ordered.itertuples(index=False):
            axis.annotate(
                str(row.comparison_arm),
                (row.training_compute if hasattr(row, "training_compute") else row.optimizer_device_compute, row.normalized_drying_group_macro_rmse),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    axis.set_xlabel("Committed training time [s]")
    axis.set_ylabel("Cumulative normalized Drying group macro RMSE [1]")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure


def plot_pipeline_degradation(
    evidence: Sequence[comparison.AirflowDegradationMetrics],
) -> Figure:
    """Plot B/A, C/A, and C/B errors while retaining unavailable C evidence."""
    values = tuple(evidence)
    if not values:
        msg = "Pipeline degradation plot requires metric evidence."
        raise ValueError(msg)
    figure, axes = plt.subplots(
        len(values),
        1,
        figsize=(8.5, 3.8 * len(values)),
        squeeze=False,
        layout="constrained",
    )
    for axis, item in zip(axes[:, 0], values, strict=True):
        heights = [
            item.drying_surrogate_error,
            np.nan if item.complete_pipeline_error is None else item.complete_pipeline_error,
            np.nan if item.airflow_substitution_discrepancy is None else item.airflow_substitution_discrepancy,
        ]
        axis.bar(("B/A", "C/A", "C/B"), heights)
        axis.set_ylabel(item.metric_id)
        if not item.c_available:
            axis.text(
                0.98,
                0.95,
                f"C unavailable: {item.c_unavailable_reason}",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
    return figure


def plot_timing_speedups(report: timing.TransientTimingReport) -> Figure:
    """Plot paired per-case values for every component-composed speedup."""
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    names = tuple(report.speedups)
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(10.0, 3.6 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[:, 0], names, strict=True):
        summary = report.speedups[name]
        available = [(case.case_id, case.speedup) for case in summary.cases if case.speedup is not None]
        axis.bar(
            [case_id for case_id, _speedup in available],
            [speedup for _case_id, speedup in available],
        )
        axis.set_ylabel("Duration ratio [1]")
        axis.set_title(
            f"{summary.reference_boundary} /\n{summary.surrogate_boundary}\n"
            f"{summary.available_count}/{summary.total_count} paired; ratio of sums={summary.ratio_of_sums}",
            fontsize="small",
        )
        axis.tick_params(axis="x", rotation=45)
    return figure


def plot_timing_distributions(report: timing.TransientTimingReport) -> Figure:
    """Plot paired reference and surrogate runtime distributions per formula."""
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing distribution plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    names = tuple(report.speedups)
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(10.0, 3.8 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[:, 0], names, strict=True):
        summary = report.speedups[name]
        pairs = [
            (case.reference_seconds, case.surrogate_seconds)
            for case in summary.cases
            if case.reference_seconds is not None and case.surrogate_seconds is not None
        ]
        if pairs:
            reference = [pair[0] for pair in pairs]
            surrogate = [pair[1] for pair in pairs]
            axis.boxplot((reference, surrogate), positions=(0, 1), widths=0.5)
            for reference_seconds, surrogate_seconds in pairs:
                axis.plot(
                    (0, 1),
                    (reference_seconds, surrogate_seconds),
                    marker="o",
                    alpha=0.45,
                )
        else:
            axis.text(
                0.5,
                0.5,
                "No paired runtime evidence available",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_xticks((0, 1), (summary.reference_boundary, summary.surrogate_boundary))
        axis.set_ylabel("Recorded duration [s]")
        axis.set_title(f"{summary.comparison_scope} — {summary.available_count}/{summary.total_count} paired cases", fontsize="small")
        axis.grid(axis="y", alpha=0.25)
    return figure


def _case_accuracy_values(accuracy: pd.DataFrame, *, label: str) -> Mapping[str, float]:
    """Admit one finite non-negative central-accuracy value per case."""
    required = {"case_id", "normalized_drying_group_macro_rmse"}
    if not isinstance(accuracy, pd.DataFrame) or not required.issubset(accuracy):
        msg = f"{label} requires case identity and central metric."
        raise ValueError(msg)
    values: dict[str, float] = {}
    for case_id, value in zip(
        accuracy["case_id"],
        accuracy["normalized_drying_group_macro_rmse"],
        strict=True,
    ):
        if not isinstance(case_id, str) or not case_id or case_id in values:
            msg = f"{label} requires unique non-empty case identities."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = f"{label} central metrics must be real scalars."
            raise TypeError(msg)
        admitted = float(value)
        if not np.isfinite(admitted) or admitted < 0.0:
            msg = f"{label} central metrics must be finite and non-negative."
            raise ValueError(msg)
        values[case_id] = admitted
    return values


def plot_accuracy_runtime_comparison(
    accuracy: pd.DataFrame,
    reports: Mapping[str, timing.TransientTimingReport],
    *,
    measure: str = "prepared_wall",
) -> Figure:
    """Compare saved models using exact paired case accuracy and a named measured clock."""
    labels = {
        "prepared_wall": "Prepared rollout request wall, warmed median [s]",
        "model": "Rollout model calls only, warmed median [s]",
        "ratio": "COMSOL reported drying solution / model-call time [ratio]",
    }
    if measure not in labels or "frame" not in accuracy:
        msg = "Accuracy/runtime comparison requires a named clock and frame identity."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(10.0, 6.5), layout="constrained")
    notices = []
    for index, (frame_name, report) in enumerate(reports.items()):
        selected = accuracy.loc[accuracy["frame"] == frame_name]
        values = _case_accuracy_values(selected, label=f"Accuracy/runtime comparison {frame_name}")
        durations: Mapping[str, float]
        if measure == "ratio":
            durations = {case.case_id: case.speedup for case in report.speedups["drying_only_solver_speedup"].cases if case.speedup is not None}
        else:
            component = "drying_no_prepared_rollout_wall_seconds" if measure == "prepared_wall" else "drying_no_rollout_model_seconds"
            durations = timing.component_case_medians(report, component)
        paired = [(durations[case_id], value) for case_id, value in values.items() if case_id in durations]
        if paired:
            axis.scatter(
                [point[0] for point in paired],
                [point[1] for point in paired],
                color=plt.colormaps["tab10"](index % 10),
                label=f"{frame_name} ({len(paired)} cases)",
            )
        notices.append(f"{frame_name}: {len(paired)}/{len(values)} paired")
    axis.set_xlabel(labels[measure])
    axis.set_ylabel("Cumulative full-rollout normalized Drying group macro RMSE [1]")
    axis.set_title("\n".join(notices), fontsize="small")
    if axis.collections:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No paired saved accuracy/runtime evidence", transform=axis.transAxes, ha="center")
    axis.grid(alpha=0.25)
    return figure
