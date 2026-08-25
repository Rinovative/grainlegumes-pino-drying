"""
evaluation_plot_transient.py

Render sequence-aware transient-drying Evaluation evidence.

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
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_axis
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_STATE_ARRAY_RANK = 4
_MINIMUM_TRAJECTORY_TIMES = 2
_TARGET_MAGNITUDE_BIN_COUNT = 12
_TARGET_MAGNITUDE_POINT_LIMIT = 50_000
_GRAIN_MOISTURE_FIELD = "w_gr"


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
        if isinstance(dataset_role, str) and dataset_role in {"id", "ood"}
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
) -> Figure:
    """Plot EDA-aligned reference, prediction, and signed-error maps."""
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
    figure, axes = layout.map_subplots(rows=len(fields), columns=3)
    column_titles = ("Reference", "Prediction", "Signed error")
    for row, field in enumerate(fields):
        if field == _GRAIN_MOISTURE_FIELD:
            raw_reference = _grain_moisture(record, reference)[index]
            raw_prediction = _grain_moisture(record, prediction)[index]
        else:
            channel = artifact.STATE_ORDER.index(field)
            raw_reference = reference[index, channel]
            raw_prediction = prediction[index, channel]
        values = (
            _display_state_values(field, raw_reference, error=False),
            _display_state_values(field, raw_prediction, error=False),
            _display_state_values(
                field,
                raw_prediction - raw_reference,
                error=True,
            ),
        )
        reference_norm = layout.linear_norm(values[0], values[1]) if lock_scale else layout.linear_norm(values[0])
        prediction_norm = reference_norm if lock_scale else layout.linear_norm(values[1])
        error_norm = layout.signed_norm(values[2])
        for column, (name, value) in enumerate(zip(column_titles, values, strict=True)):
            axis = axes[row, column]
            is_error = name == "Signed error"
            image = axis.pcolormesh(
                x_values,
                y_values,
                value,
                shading="auto",
                cmap=(
                    visual_semantics.field_visual_semantics(
                        field,
                        role="signed_error",
                    ).colormap
                    if is_error
                    else visual_semantics.field_visual_semantics(field).colormap
                ),
                norm=(error_norm if is_error else (reference_norm if name == "Reference" else prediction_norm)),
            )
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(name)
            layout.add_map_colorbar(
                figure,
                image,
                axis,
                label=field_labels.display_unit(
                    _state_unit(field),
                    quantity_kind="difference" if is_error else "absolute",
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


def plot_state_trajectory(
    record: Any,
    *,
    state_field: str,
    include_envelope: bool = False,
) -> Figure:
    """Plot reference and prediction spatial summaries over exact physical time."""
    if state_field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Transient trajectory field is unsupported: {state_field!r}."
        raise ValueError(msg)
    if not isinstance(include_envelope, bool):
        msg = "include_envelope must be boolean."
        raise TypeError(msg)
    times, mask, reference_values, prediction_values = _trajectory_values(record, state_field)
    reference_display = _display_state_values(
        state_field,
        reference_values,
        error=False,
    )
    prediction_display = _display_state_values(
        state_field,
        prediction_values,
        error=False,
    )
    reference_selected = reference_display[:, mask]
    prediction_selected = prediction_display[:, mask]
    display_time = time_axis.physical_time_display(times, preferred_unit="auto")
    plotted_time = display_time.values(times)
    figure, axis = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    axis.plot(
        plotted_time,
        reference_selected.mean(axis=1),
        color="black",
        linewidth=2.2,
        label="Reference",
    )
    axis.plot(
        plotted_time,
        prediction_selected.mean(axis=1),
        color="#1f77b4",
        linewidth=2.2,
        linestyle="--",
        label="Prediction",
    )
    if include_envelope:
        reference_bounds = np.quantile(reference_selected, (0.1, 0.9), axis=1)
        prediction_bounds = np.quantile(prediction_selected, (0.1, 0.9), axis=1)
        axis.fill_between(
            plotted_time,
            reference_bounds[0],
            reference_bounds[1],
            color="black",
            alpha=0.12,
        )
        axis.fill_between(
            plotted_time,
            prediction_bounds[0],
            prediction_bounds[1],
            color="#1f77b4",
            alpha=0.12,
        )
    display_time.configure(axis)
    axis.set_ylabel(
        field_labels.field_label_with_unit(
            state_field,
            _state_unit(state_field),
            mathtext=True,
        )
    )
    identity = _value(record, "identity")
    if not isinstance(identity, Mapping):
        msg = "Transient trajectory record identity must be a mapping."
        raise TypeError(msg)
    axis.set_title(f"Spatial mean — {_material_label(identity.get('material_family'))} · {_value(record, 'case_id')}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    return figure


def _shared_record_times(records: Sequence[Any], *, exclude_origin: bool) -> np.ndarray:
    """Return exact physical times shared by one non-empty record group."""
    admitted = tuple(records)
    if not admitted:
        msg = "Transient trajectory summaries require at least one record."
        raise ValueError(msg)
    shared = {float(value) for value in np.asarray(_value(admitted[0], "physical_times"), dtype=np.float64)}
    for record in admitted[1:]:
        shared.intersection_update(float(value) for value in np.asarray(_value(record, "physical_times"), dtype=np.float64))
    times = np.asarray(sorted(shared), dtype=np.float64)
    if exclude_origin and times.size:
        times = times[1:]
    if not times.size:
        msg = "Transient trajectory records share no required exact physical times."
        raise ValueError(msg)
    return times


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
    all_times = np.concatenate([_shared_record_times(records, exclude_origin=False) for records in admitted.values()])
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
        for series_index, (label, records) in enumerate(admitted.items()):
            times = _shared_record_times(records, exclude_origin=False)
            reference_rows: list[np.ndarray] = []
            prediction_rows: list[np.ndarray] = []
            for record in records:
                record_times, mask, reference, prediction = _trajectory_values(record, field)
                indices = np.asarray([int(np.flatnonzero(record_times == value)[0]) for value in times])
                reference_display = _display_state_values(field, reference[indices], error=False)
                prediction_display = _display_state_values(field, prediction[indices], error=False)
                reference_rows.append(np.mean(reference_display[:, mask], axis=1))
                prediction_rows.append(np.mean(prediction_display[:, mask], axis=1))
            reference_matrix = np.stack(reference_rows)
            prediction_matrix = np.stack(prediction_rows)
            plotted_time = display_time.values(times)
            color = colors(series_index % colors.N)
            axis.plot(
                plotted_time,
                reference_matrix.mean(axis=0),
                color=color,
                linewidth=2.0,
                label=f"{label} · Reference",
            )
            axis.plot(
                plotted_time,
                prediction_matrix.mean(axis=0),
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=f"{label} · Prediction",
            )
            if len(records) > 1:
                reference_bounds = np.quantile(reference_matrix, (0.1, 0.9), axis=0)
                prediction_bounds = np.quantile(prediction_matrix, (0.1, 0.9), axis=0)
                axis.fill_between(plotted_time, reference_bounds[0], reference_bounds[1], color=color, alpha=0.08)
                axis.fill_between(plotted_time, prediction_bounds[0], prediction_bounds[1], color=color, alpha=0.08)
        display_time.configure(axis)
        axis.set_ylabel(field_labels.field_label_with_unit(field, _state_unit(field), mathtext=True))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    figure.suptitle("Reference vs prediction trajectories — spatial mean; aggregate bands show case 10th-90th percentiles")
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
    all_times = np.concatenate([_shared_record_times(records, exclude_origin=True) for records in admitted.values()])
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
        for series_index, (label, records) in enumerate(admitted.items()):
            times = _shared_record_times(records, exclude_origin=True)
            rows = []
            for record in records:
                record_times, values = _normalized_error_trajectory(
                    record,
                    scaling_state=scaling_states[label],
                    fields=fields,
                )
                indices = np.asarray([int(np.flatnonzero(record_times == value)[0]) for value in times])
                rows.append(values[field][indices])
            matrix = np.stack(rows)
            plotted_time = display_time.values(times)
            color = colors(series_index % colors.N)
            axis.plot(plotted_time, matrix.mean(axis=0), color=color, linewidth=2.0, label=label)
            if len(records) > 1:
                bounds = np.quantile(matrix, (0.1, 0.9), axis=0)
                axis.fill_between(plotted_time, bounds[0], bounds[1], color=color, alpha=0.1)
        display_time.configure(axis)
        axis.set_ylabel(f"{field_labels.field_label(field)} normalized RMSE [1]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    figure.suptitle("Error over physical time — aggregate lines are case means with 10th-90th percentile bands")
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


def _aggregate_snapshot_evidence(
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
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
        error = prediction_snapshot - reference_snapshot
        error_sum[mask] += error[mask]
        absolute_error_sum[mask] += np.abs(error[mask])
        error_square_sum[mask] += np.square(error[mask], dtype=np.float64)
        counts[mask] += 1
        reference_means.append(float(np.mean(reference_snapshot[mask], dtype=np.float64)))
        prediction_means.append(float(np.mean(prediction_snapshot[mask], dtype=np.float64)))
    if x_values is None or y_values is None or error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
        msg = "Aggregate transient snapshot evidence was not initialized."
        raise RuntimeError(msg)
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


def plot_aggregate_spatial_error(
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
) -> Figure:
    """Plot mean absolute spatial error at one exact shared physical time."""
    x_values, y_values, mean_absolute_error, _std_error, reference_means, _prediction_means = _aggregate_snapshot_evidence(
        records,
        state_field=state_field,
        physical_time=physical_time,
    )
    displayed_error = _display_state_values(
        state_field,
        mean_absolute_error,
        error=True,
    )
    figure, axes = layout.map_subplots(rows=1, columns=1)
    axis = axes[0, 0]
    image = axis.pcolormesh(
        x_values,
        y_values,
        displayed_error,
        shading="auto",
        cmap=visual_semantics.field_visual_semantics(
            state_field,
            role="absolute_error",
        ).colormap,
        norm=layout.linear_norm(displayed_error),
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"Mean absolute error — {state_field} — t={physical_time:g} h — {len(reference_means)} cases")
    layout.add_map_colorbar(
        figure,
        image,
        axis,
        label=field_labels.display_unit(
            _state_unit(state_field),
            quantity_kind="difference",
        ),
    )
    return figure


def plot_predicted_vs_reference(
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
) -> Figure:
    """Plot bounded case-level spatial means for prediction versus reference."""
    _x_values, _y_values, _mean_absolute_error, _std_error, reference_means, prediction_means = _aggregate_snapshot_evidence(
        records,
        state_field=state_field,
        physical_time=physical_time,
    )
    reference_display = _display_state_values(
        state_field,
        reference_means,
        error=False,
    )
    prediction_display = _display_state_values(
        state_field,
        prediction_means,
        error=False,
    )
    lower = float(min(np.min(reference_display), np.min(prediction_display)))
    upper = float(max(np.max(reference_display), np.max(prediction_display)))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    unit = field_labels.display_unit(
        _state_unit(state_field),
        quantity_kind="absolute",
    )
    label = field_labels.field_label(state_field)
    figure, axis = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    axis.scatter(reference_display, prediction_display, s=55, alpha=0.8)
    axis.plot((lower, upper), (lower, upper), color="black", linestyle="--")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel(f"Reference case spatial mean — {label} [{unit}]")
    axis.set_ylabel(f"Predicted case spatial mean — {label} [{unit}]")
    axis.set_title(f"Prediction versus reference — t={physical_time:g} h — {len(reference_means)} cases")
    axis.grid(alpha=0.25)
    return figure


def _labelled_record_series(
    value: Sequence[Any] | Mapping[str, Sequence[Any]],
) -> dict[str, tuple[Any, ...]]:
    """Normalize one sequence or labelled multi-context aggregate."""
    if isinstance(value, Mapping):
        series: dict[str, tuple[Any, ...]] = {}
        for label, records in value.items():
            if not isinstance(label, str) or not label:
                msg = "Transient aggregate context labels must be non-empty text."
                raise ValueError(msg)
            admitted = tuple(records)
            if not admitted:
                msg = "Transient aggregate contexts must contain at least one record."
                raise ValueError(msg)
            series[label] = admitted
    else:
        admitted = tuple(value)
        if not admitted:
            msg = "Transient aggregate plots require at least one record."
            raise ValueError(msg)
        series = {"": admitted}
    if not series:
        msg = "Transient aggregate plots require at least one context."
        raise ValueError(msg)
    return series


def plot_aggregate_error_maps(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
    statistic: str,
) -> Figure:
    """Plot exact-time mean or standard-deviation error maps by context."""
    fields = _selected_fields(state_fields)
    if statistic not in {"mean", "std"}:
        msg = "Aggregate error-map statistic must be 'mean' or 'std'."
        raise ValueError(msg)
    series = _labelled_record_series(records)
    figure, axes = layout.map_subplots(
        rows=len(fields),
        columns=len(series),
    )
    total_cases = 0
    for column, (context, current_records) in enumerate(series.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
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
            )
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
        (f"{statistic_label} error at t={physical_time:g} h — {total_cases} case-context observations"),
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
    series = _labelled_record_series(records)
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


def plot_worst_case_errors(
    case_frame: pd.DataFrame,
    *,
    top_k: int = 5,
) -> Figure:
    """Show the largest complete-rollout case errors without changing ranking."""
    required = {
        "frame",
        "material_family",
        "case_id",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(case_frame, pd.DataFrame) or not required.issubset(case_frame):
        msg = "Transient worst-case plot requires case-level summary columns."
        raise ValueError(msg)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        msg = "top_k must be a positive integer."
        raise TypeError(msg)
    selected = case_frame.loc[
        (case_frame["mode"] == "autonomous_full") & (case_frame["requested_horizon"] == "full") & (case_frame["scope"] == "cumulative")
    ].nlargest(top_k, "normalized_drying_group_macro_rmse")
    if selected.empty:
        msg = "Transient worst-case plot requires complete-rollout case evidence."
        raise ValueError(msg)
    role_count = selected["dataset_role"].nunique() if "dataset_role" in selected else 0
    include_frame = selected["frame"].nunique() > role_count
    labels = tuple(
        f"{_material_role_label(frame, material, role, include_frame=include_frame)} · {case_id}"
        for frame, material, role, case_id in zip(
            selected["frame"],
            selected["material_family"],
            selected.get("dataset_role", pd.Series((None,) * len(selected), index=selected.index)),
            selected["case_id"],
            strict=True,
        )
    )
    values = selected["normalized_drying_group_macro_rmse"].to_numpy(dtype=np.float64)
    figure, axis = plt.subplots(
        figsize=(9.0, max(3.5, 0.55 * len(selected) + 1.5)),
        layout="constrained",
    )
    axis.barh(np.arange(len(selected)), values, color="#c44e52")
    axis.set_yticks(np.arange(len(selected)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Normalized Drying group macro RMSE [1]")
    axis.set_title(f"Worst {len(selected)} complete-rollout cases")
    axis.grid(axis="x", alpha=0.25)
    return figure


def plot_id_ood_generalization(summary_frame: pd.DataFrame) -> Figure:
    """Compare matched ID and near-family OOD complete-rollout errors."""
    selected = _full_autonomous_cumulative(summary_frame)
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in selected.iterrows():
        frame_name = str(row["frame"])
        material = _material_label(row["material_family"])
        if frame_name.endswith(" ID"):
            model, role = frame_name[:-3], "ID"
        elif frame_name.endswith(" OOD"):
            model, role = frame_name[:-4], "OOD"
        else:
            continue
        metric_value = row["normalized_drying_group_macro_rmse"]
        if isinstance(metric_value, bool) or not isinstance(metric_value, Real):
            msg = "ID/OOD generalization requires finite scalar error metrics."
            raise TypeError(msg)
        admitted_value = float(metric_value)
        if not np.isfinite(admitted_value):
            msg = "ID/OOD generalization requires finite scalar error metrics."
            raise ValueError(msg)
        rows.setdefault((model, material), {})[role] = admitted_value
    paired = tuple((model, material, values["ID"], values["OOD"]) for (model, material), values in rows.items() if set(values) == {"ID", "OOD"})
    if not paired:
        msg = "ID/OOD generalization requires paired role artifacts per model."
        raise ValueError(msg)
    id_error = np.asarray([value[2] for value in paired], dtype=np.float64)
    ood_error = np.asarray([value[3] for value in paired], dtype=np.float64)
    maximum = float(max(np.max(id_error), np.max(ood_error)))
    figure, axis = plt.subplots(figsize=(6.3, 6.0), layout="constrained")
    axis.scatter(id_error, ood_error, s=70)
    axis.plot((0.0, maximum), (0.0, maximum), color="black", linestyle="--")
    for model, material, x_value, y_value in paired:
        axis.annotate(
            f"{model} · {material}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("ID normalized Drying group macro RMSE [1]")
    axis.set_ylabel("Near-family OOD normalized Drying group macro RMSE [1]")
    axis.set_title("ID-to-OOD generalization")
    axis.grid(alpha=0.25)
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
    axis.scatter(pivot["cumulative"], pivot["endpoint"])
    maximum = float(np.max(pivot[["cumulative", "endpoint"]].to_numpy()))
    axis.plot((0.0, maximum), (0.0, maximum), linestyle="--", color="black")
    axis.set_xlabel("Cumulative central error [1]")
    axis.set_ylabel("Endpoint central error [1]")
    axis.grid(alpha=0.25)
    return figure


def plot_target_time(records: Sequence[Any]) -> Figure:
    """Plot paired reached target times and disclose censored records."""
    reference: list[float] = []
    prediction: list[float] = []
    censored = 0
    unavailable = 0
    for record in records:
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
    figure, axis = plt.subplots(figsize=(6.5, 6.0), layout=None)
    figure.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.94)
    if reference:
        display_time = time_axis.physical_time_display(
            (*reference, *prediction),
            preferred_unit="auto",
        )
        reference_display = display_time.values(reference)
        prediction_display = display_time.values(prediction)
        axis.scatter(reference_display, prediction_display)
        lower = min(reference_display.min(), prediction_display.min())
        upper = max(reference_display.max(), prediction_display.max())
        axis.plot((lower, upper), (lower, upper), linestyle="--", color="black")
        display_time.configure(axis)
        display_time.configure(axis, dimension="y")
        axis.set_xlabel(f"Reference time to target [{display_time.unit}]")
        axis.set_ylabel(f"Predicted time to target [{display_time.unit}]")
    else:
        axis.set_xlabel("Reference time to target [h]")
        axis.set_ylabel("Predicted time to target [h]")
    axis.text(
        0.02,
        0.98,
        f"Paired reached: {len(reference)}\nCensored: {censored}\nUnavailable: {unavailable}",
        transform=axis.transAxes,
        va="top",
    )
    axis.grid(alpha=0.25)
    return figure


def plot_training_performance_vs_compute(performance: pd.DataFrame) -> Figure:
    """Plot matched A0, A+, and B performance by cumulative rollout semantics."""
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
    if not arms or not arms.issubset({"A0", "A+", "B"}):
        msg = "Matched-compute plot supports only A0, A+, and B arms."
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
        ordered = group.assign(_arm_order=group["comparison_arm"].map({"A0": 0, "A+": 1, "B": 2})).sort_values("_arm_order")
        axis.plot(
            ordered["optimizer_device_compute"],
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            label=f"{_material_label(material)} · {mode} / {horizon}",
        )
        for row in ordered.itertuples(index=False):
            axis.annotate(
                str(row.comparison_arm),
                (row.optimizer_device_compute, row.normalized_drying_group_macro_rmse),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    axis.set_xlabel("Optimizer-device compute [persisted clock unit]")
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
        axis.set_ylabel("Speedup [ratio]")
        axis.set_title(f"{name} — {summary.available_count}/{summary.total_count} available; ratio of sums={summary.ratio_of_sums}")
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
        axis.set_xticks((0, 1), ("Reference", "Surrogate"))
        axis.set_ylabel("Warmed runtime [s]")
        axis.set_title(f"{name} — {summary.available_count}/{summary.total_count} paired cases")
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


def _plot_unavailable_accuracy_pairing(*, x_label: str, reason: str) -> Figure:
    """Render explicit absence without fabricating an accuracy/timing pair."""
    figure, axis = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    axis.text(0.5, 0.5, reason, transform=axis.transAxes, ha="center", va="center")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Normalized Drying group macro RMSE [1]")
    axis.grid(alpha=0.25)
    return figure


def plot_accuracy_vs_inference_time(
    accuracy: pd.DataFrame,
    report: timing.TransientTimingReport,
    *,
    component_name: str = "drying_no_end_to_end_seconds",
) -> Figure:
    """Pair case-level central accuracy with warmed measured inference time."""
    values = _case_accuracy_values(accuracy, label="Accuracy-inference-time plot")
    durations = timing.component_case_medians(report, component_name)
    paired = tuple(
        (case.case_id, durations[case.case_id], values[case.case_id]) for case in report.cases if case.case_id in durations and case.case_id in values
    )
    x_label = f"{component_name} warmed median [s]"
    if not paired:
        return _plot_unavailable_accuracy_pairing(
            x_label=x_label,
            reason="No paired case-level accuracy and inference-time evidence is available.",
        )
    figure, axis = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    axis.scatter(
        [duration for _case_id, duration, _value in paired],
        [value for _case_id, _duration, value in paired],
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Normalized Drying group macro RMSE [1]")
    axis.set_title(f"Paired cases: {len(paired)}")
    axis.grid(alpha=0.25)
    return figure


def plot_accuracy_vs_speedup(
    accuracy: pd.DataFrame,
    report: timing.TransientTimingReport,
    *,
    speedup_name: str = "drying_only_solver_speedup",
) -> Figure:
    """Pair case-level central accuracy with one admitted speedup definition."""
    values = _case_accuracy_values(accuracy, label="Accuracy-speedup plot")
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Accuracy-speedup plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    if speedup_name not in report.speedups:
        msg = f"Unknown transient speedup {speedup_name!r}."
        raise ValueError(msg)
    speedups: dict[str, float] = {}
    for case in report.speedups[speedup_name].cases:
        if case.speedup is not None:
            speedups[case.case_id] = case.speedup
    paired = tuple(
        (case.case_id, speedups[case.case_id], values[case.case_id])
        for case in report.speedups[speedup_name].cases
        if case.case_id in speedups and case.case_id in values
    )
    if not paired:
        return _plot_unavailable_accuracy_pairing(
            x_label=f"{speedup_name} [ratio]",
            reason="No paired case-level accuracy and speedup evidence is available.",
        )
    figure, axis = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    axis.scatter(
        [speedup for _case_id, speedup, _value in paired],
        [value for _case_id, _speedup, value in paired],
    )
    axis.set_xlabel(f"{speedup_name} [ratio]")
    axis.set_ylabel("Normalized Drying group macro RMSE [1]")
    axis.set_title(f"Paired cases: {len(paired)}")
    axis.grid(alpha=0.25)
    return figure
