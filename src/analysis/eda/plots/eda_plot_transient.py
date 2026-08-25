"""
eda_plot_transient.py

Plot transient completed-output exploratory diagnostics.
Responsibilities:
  - Render exact physical-time state and static-field snapshots
  - Plot physical trajectories, simulated-support schedules, and realized parameters
  - Consolidate target outcomes, terminal timing, final moisture, and exact evidence
Design principles:
  - Every incompatible physical channel owns a separately labelled axis
  - Plot reductions delegate to the transient EDA semantic owner
  - Missing timing or physical-time evidence remains visibly unavailable
This module does NOT:
  - Change established steady-flow plot functions or presentation styling
  - Run model inference, calculate prediction error, or derive speedup
  - Add nearest-time fallbacks or average incompatible raw units
"""

from __future__ import annotations

import math
from numbers import Integral
from textwrap import wrap
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src import generation
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_transient as transient
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_axis
from src.datasets.contracts import dataset_contracts_transient as transient_contract

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpecBase

_SPATIAL_SUPTITLE_Y: Final = 0.985
_SPATIAL_GRID_TOP: Final = _SPATIAL_SUPTITLE_Y - (_SPATIAL_SUPTITLE_Y - 0.91) / 8.0
_SPATIAL_MAP_ROW_SPACING: Final = 0.24 * 3.0 / 4.0 / 2.0 / 2.0
_SPATIAL_MAP_COLUMN_SPACING: Final = 0.34 / 2.0
_SCHEDULE_SPATIAL_GRID_TOP: Final = _SPATIAL_SUPTITLE_Y - (_SPATIAL_SUPTITLE_Y - 0.96) / 3.0
_LINE_ROW_HEIGHT_INCHES: Final = 1.65 * 4.0 / 3.0
_MAP_ROW_HEIGHT_INCHES: Final = 3.45
_PRESSURE_SCHEDULE_GROUP_SPACING: Final = 0.24 * 1.25
_SCHEDULE_ROW_SPACING: Final = 0.24
_LINE_MAP_SPACER_INCHES: Final = 0.75 / 2.0 * 1.25 * 1.25
_LINE_LEGEND_COLUMN_SPACING: Final = 0.30 * 0.25
_MAP_ROW_SPACING: Final = 0.24 / 2.0 / 2.0
_MAP_COLUMN_SPACING: Final = 0.34 * 3.0 / 4.0 / 2.0
_TRANSIENT_ROW_LABEL_X: Final = 0.035
_TRAJECTORY_ROW_LABEL_X: Final = -0.28 * 0.5
_TRAJECTORY_ROW_HEIGHT_INCHES: Final = 4.0 * 0.75
_EARLY_OPERATION_END_HOURS: Final = 1.0
_HOURS_PER_DAY: Final = 24.0
_RANGE_SERIES_VALUE_COUNT: Final = 3
_DATASET_HEADER_WRAP_WIDTH: Final = 38
_MAXIMUM_DATASET_HEADER_LINES: Final = 2
_OUTCOME_LABELS: Final = ("Reached target", "Right-censored")
_COMPLETION_FIGURE_WIDTH_INCHES: Final = 14.0
_COMPLETION_GRID_WIDTH_FRACTION: Final = 0.90
_COMPLETION_BASE_RATIO_SUM: Final = 2.75
_COMPLETION_BASE_COLUMN_INCHES: Final = _COMPLETION_FIGURE_WIDTH_INCHES * _COMPLETION_GRID_WIDTH_FRACTION / _COMPLETION_BASE_RATIO_SUM
_COMPLETION_PLOT_WIDTH_INCREMENT_INCHES: Final = _COMPLETION_FIGURE_WIDTH_INCHES / 16.0
_COMPLETION_SCIENCE_COLUMN_RATIO: Final = 1.0 + _COMPLETION_PLOT_WIDTH_INCREMENT_INCHES / (2.0 * _COMPLETION_BASE_COLUMN_INCHES)
_COMPLETION_LEGEND_COLUMN_RATIO: Final = 0.75 - _COMPLETION_PLOT_WIDTH_INCREMENT_INCHES / _COMPLETION_BASE_COLUMN_INCHES


def _dataset_color_map(datasets: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """Return stable colorblind-friendly colors keyed by concise dataset label."""
    identities = tuple(
        visual_semantics.DatasetVisualIdentity(
            canonical_identity=label,
            label=label,
        )
        for label in datasets
    )
    colors = visual_semantics.dataset_colors(identities)
    return {label: colors[label] for label in datasets}


def _integer_count(value: object, *, label: str) -> int:
    """Return one exact non-negative plotted case count."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        message = f"{label} must be one non-negative integer count."
        raise TypeError(message)
    return int(value)


def _field_axes(count: int, *, width: float = 4.6, height: float = 3.8) -> tuple[Figure, np.ndarray]:
    """Create one compact row-wrapped field-axis grid."""
    columns = min(2, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
        layout=None,
    )
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.10, top=0.88, wspace=0.30, hspace=0.38)
    flat = axes.ravel()
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat


def _day_display(
    physical_hours: Sequence[float] | np.ndarray,
    *,
    right_margin_hours: float = 0.0,
    include_zero: bool = True,
) -> time_axis.PhysicalTimeDisplay:
    """Return a display-only day axis from authoritative physical hours."""
    return time_axis.physical_time_display(
        physical_hours,
        preferred_unit="d",
        include_zero=include_zero,
        right_margin_hours=right_margin_hours,
        major_interval_hours=24.0,
        minor_interval_hours=None,
    )


def _schedule_fields() -> tuple[generation.contracts.FieldContract, ...]:
    """Return authoritative time-varying schedule quantities in schema order."""
    profile = generation.contracts.get_profile_contract("transient_drying")
    return tuple(field for field in profile.schedule_fields if field.name != "t")


def _case_time_resolutions(
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    master_time_hours: float,
) -> dict[str, time_axis.ResolvedPhysicalTime]:
    """Resolve exact/latest-prior/final stored time for each transient case."""
    return {
        label: time_axis.resolve_master_physical_time(
            transient.available_physical_times(frame, case_ids[label]),
            master_time_hours,
        )
        for label, frame in datasets.items()
        if capabilities.is_transient_frame(frame)
    }


def _draw_exact_schedule_split_axes(
    main_axes: Sequence[Axes],
    startup_axes: Sequence[Axes],
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    colors: Mapping[str, str],
    master_time_hours: float,
    case_times: Mapping[str, time_axis.ResolvedPhysicalTime],
) -> None:
    """Draw exact clipped schedules in day-scale main and hour-scale startup views."""
    fields = _schedule_fields()
    if len(main_axes) != len(fields) or len(startup_axes) != len(fields):
        message = "Split schedule axes must match authoritative schedule quantities."
        raise ValueError(message)
    main_support_by_row: list[list[np.ndarray]] = [[] for _field in fields]
    startup_support_by_row: list[list[np.ndarray]] = [[] for _field in fields]
    for label, frame in datasets.items():
        for field_index, field in enumerate(fields):
            series = transient.supported_schedule_series(
                frame,
                case_ids[label],
                field.name,
            )
            display_values = field_labels.display_values(
                series.values,
                field.unit,
            )
            main, startup = _split_physical_time_support(
                series.physical_time_hours,
                display_values,
            )
            main_times, (main_values,) = main
            startup_times, (startup_values,) = startup
            if main_times.size:
                main_axes[field_index].plot(
                    main_times / _HOURS_PER_DAY,
                    main_values,
                    color=colors[label],
                    marker=None,
                )
                main_support_by_row[field_index].append(main_times)
            if startup_times.size:
                startup_axes[field_index].plot(
                    startup_times,
                    startup_values,
                    color=colors[label],
                    marker=None,
                )
                startup_support_by_row[field_index].append(startup_times)
            actual = case_times.get(label)
            if actual is None:
                message = f"Current schedule marker lacks resolved case time for {label!r}."
                raise ValueError(message)
            marker_value = field_labels.display_values(
                (series.value_at(actual.physical_time),),
                field.unit,
            )[0]
            if actual.physical_time < _EARLY_OPERATION_END_HOURS:
                marker_axis = startup_axes[field_index]
                marker_time = actual.physical_time
            else:
                marker_axis = main_axes[field_index]
                marker_time = actual.physical_time / _HOURS_PER_DAY
            marker_axis.scatter(
                (marker_time,),
                (marker_value,),
                color=colors[label],
                marker="o",
                s=28,
                zorder=4,
            )
    for field_index, (main_axis, startup_axis) in enumerate(zip(main_axes, startup_axes, strict=True)):
        _configure_split_time_axes(
            main_axis,
            startup_axis,
            main_support_by_row[field_index],
            startup_support_by_row[field_index],
        )
        if master_time_hours < _EARLY_OPERATION_END_HOURS:
            startup_axis.axvline(
                master_time_hours,
                color="black",
                linestyle=":",
                linewidth=1.2,
            )
        else:
            main_axis.axvline(
                master_time_hours / _HOURS_PER_DAY,
                color="black",
                linestyle=":",
                linewidth=1.2,
            )
        main_axis.grid(axis="y", alpha=0.20)
        startup_axis.grid(axis="y", alpha=0.20)
    for axis in (*main_axes[:-1], *startup_axes[:-1]):
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelbottom=False)


def _compact_two_line_label(value: str) -> str:
    """Wrap one concise header to at most two lines without removing text."""
    lines = wrap(
        value,
        width=_DATASET_HEADER_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) <= _MAXIMUM_DATASET_HEADER_LINES:
        return "\n".join(lines)
    return "\n".join((lines[0], " ".join(lines[1:])))


def _section_three_dataset_header(label: str) -> str:
    """Remove the redundant task prefix from section-3 plot headings only."""
    return label.removeprefix("Drying · ")


def _display_case_time(
    resolved: time_axis.ResolvedPhysicalTime | None,
) -> str:
    """Return current-time text with full-hour formatting for final holds."""
    if resolved is None:
        return ""
    if resolved.final_hold:
        final_time = time_axis.format_terminal_physical_time_hours(
            resolved.physical_time,
        )
        return f" (t = {final_time}, final)"
    return f" (t = {resolved.physical_time:g} h)"


def _apply_eda_map_axis_labels(map_axes: np.ndarray) -> None:
    """Keep coordinate labels on the first actual map and bottom map row."""
    rows = tuple(tuple(axis for axis in row) for row in map_axes)
    bottom = len(rows) - 1
    for row_index, row in enumerate(rows):
        visible = tuple(axis for axis in row if axis.get_visible() and axis.axison)
        for axis in visible:
            is_bottom = row_index == bottom
            axis.set_xlabel("x [m]" if is_bottom else "")
            axis.tick_params(
                axis="x",
                labelbottom=is_bottom,
                labelsize=layout.MAP_LAYOUT.tick_size,
            )
            axis.set_ylabel("")
            axis.tick_params(
                axis="y",
                labelleft=False,
                labelsize=layout.MAP_LAYOUT.tick_size,
            )
        if visible:
            visible[0].set_ylabel("y [m]")
            visible[0].tick_params(axis="y", labelleft=True)


def _draw_inlet_pressure_axis(
    axis: Axes,
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    colors: Mapping[str, str],
) -> None:
    """Draw each exact spatial inlet-pressure profile on its inlet coordinate."""
    for label, frame in datasets.items():
        row = capabilities.case_row(frame, case_ids[label])
        inlet_coordinate, pressure = capabilities.inlet_pressure_boundary(
            frame,
            row,
        )
        axis.plot(
            inlet_coordinate,
            pressure,
            color=colors[label],
            marker=None,
            label="_nolegend_",
        )
    axis.set_xlim(0.0, 1.2)
    axis.set_xlabel("x [m]")
    axis.grid(axis="y", alpha=0.20)


def _spatial_map_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    requested_fields: Sequence[str] | None,
    field_view: capabilities.FieldView,
    lock_scale: bool,
    physical_time_hours: float | None,
    include_schedules: bool,
    title: str,
) -> Figure:
    """Render one union-capability map grid with optional exact split schedules."""
    if not datasets or tuple(case_ids) != tuple(datasets):
        message = "Spatial-map datasets and case IDs must match in order."
        raise ValueError(message)
    if not isinstance(lock_scale, bool):
        message = "lock_scale must be boolean."
        raise TypeError(message)
    resolution = capabilities.resolve_fields(
        datasets,
        view=field_view,
        requested=requested_fields,
    )
    needs_time = any(
        capabilities.is_dynamic_field(datasets[label], field) for field in resolution.fields for label in resolution.datasets_by_field[field]
    )
    if needs_time and physical_time_hours is None:
        message = "Dynamic spatial fields require one selected master physical time."
        raise ValueError(message)
    master_time = None if physical_time_hours is None else float(physical_time_hours)
    case_times = {} if master_time is None else _case_time_resolutions(datasets, case_ids, master_time)
    schedule_fields = _schedule_fields() if include_schedules else ()
    line_rows = 1 + len(schedule_fields) if include_schedules else 0
    map_rows = len(resolution.fields)
    dataset_count = len(datasets)
    figure_height = line_rows * _LINE_ROW_HEIGHT_INCHES + (_LINE_MAP_SPACER_INCHES if include_schedules else 0.0) + map_rows * _MAP_ROW_HEIGHT_INCHES
    figure = plt.figure(
        figsize=(5.2 * dataset_count + 1.0, figure_height),
        layout=None,
    )
    scientific_axes: list[Axes] = []
    map_grid: GridSpecBase
    if include_schedules:
        colors = _dataset_color_map(datasets)
        outer = figure.add_gridspec(
            3,
            1,
            height_ratios=(
                line_rows * _LINE_ROW_HEIGHT_INCHES,
                _LINE_MAP_SPACER_INCHES,
                map_rows * _MAP_ROW_HEIGHT_INCHES,
            ),
            left=0.10,
            right=0.98,
            bottom=0.07,
            top=_SCHEDULE_SPATIAL_GRID_TOP,
            hspace=0.0,
        )
        line_block = outer[0].subgridspec(
            1,
            2,
            width_ratios=(1.34, 0.30),
            wspace=_LINE_LEGEND_COLUMN_SPACING,
        )
        scientific_line_block = line_block[0, 0].subgridspec(
            2,
            1,
            height_ratios=(1.0, float(len(schedule_fields))),
            hspace=_PRESSURE_SCHEDULE_GROUP_SPACING,
        )
        pressure_axis = figure.add_subplot(scientific_line_block[0, 0])
        _draw_inlet_pressure_axis(
            pressure_axis,
            datasets=datasets,
            case_ids=case_ids,
            colors=colors,
        )
        reference_frame = next(iter(datasets.values()))
        layout.add_channel_row_label(
            pressure_axis,
            capabilities.field_quantity_label(
                reference_frame,
                "p_in_bc",
                mathtext=True,
            ),
            figure_x=_TRANSIENT_ROW_LABEL_X,
        )
        schedule_grid = scientific_line_block[1, 0].subgridspec(
            len(schedule_fields),
            2,
            width_ratios=(1.0, 0.34),
            wspace=0.30,
            hspace=_SCHEDULE_ROW_SPACING,
        )
        main_schedule_axes = tuple(figure.add_subplot(schedule_grid[row, 0]) for row in range(len(schedule_fields)))
        startup_schedule_axes = tuple(figure.add_subplot(schedule_grid[row, 1]) for row in range(len(schedule_fields)))
        for axis, schedule_field in zip(main_schedule_axes, schedule_fields, strict=True):
            layout.add_channel_row_label(
                axis,
                field_labels.field_label_with_unit(
                    schedule_field.name,
                    schedule_field.unit,
                    mathtext=True,
                ),
                figure_x=_TRANSIENT_ROW_LABEL_X,
            )
        if master_time is None:
            message = "Transient schedule maps require one selected master time."
            raise ValueError(message)
        _draw_exact_schedule_split_axes(
            main_schedule_axes,
            startup_schedule_axes,
            datasets=datasets,
            case_ids=case_ids,
            colors=colors,
            master_time_hours=master_time,
            case_times=case_times,
        )
        if main_schedule_axes:
            main_schedule_axes[0].set_title("Physical time from 1 h to final support")
            startup_schedule_axes[0].set_title("Startup: 0-1 h")
        legend_axis = figure.add_subplot(line_block[0, 1])
        legend_axis.set_axis_off()
        legend_axis.legend(
            handles=[Line2D([], [], color=colors[label], label=label) for label in datasets],
            loc="upper left",
            frameon=False,
        )
        map_grid = outer[2].subgridspec(
            map_rows,
            dataset_count,
            wspace=_MAP_COLUMN_SPACING,
            hspace=_MAP_ROW_SPACING,
        )
        scientific_axes.extend((pressure_axis, *main_schedule_axes, *startup_schedule_axes))
    else:
        map_grid = figure.add_gridspec(
            map_rows,
            dataset_count,
            left=0.10,
            right=0.98,
            bottom=0.07,
            top=_SPATIAL_GRID_TOP,
            wspace=_SPATIAL_MAP_COLUMN_SPACING,
            hspace=_SPATIAL_MAP_ROW_SPACING,
        )

    map_axes = np.empty((map_rows, dataset_count), dtype=object)
    dataset_labels = tuple(datasets)
    for field_index, field in enumerate(resolution.fields):
        compatible = set(resolution.datasets_by_field[field])
        units = {capabilities.field_unit(datasets[label], field) for label in compatible}
        if len(units) != 1:
            message = f"Spatial field {field!r} has incompatible physical units."
            raise ValueError(message)
        reference_label = resolution.datasets_by_field[field][0]
        values_by_label: dict[str, np.ndarray] = {}
        for label in compatible:
            frame = datasets[label]
            row = capabilities.case_row(frame, case_ids[label])
            if capabilities.is_dynamic_field(frame, field):
                resolved_time = case_times[label]
                values = capabilities.field_values_at_physical_time(
                    frame,
                    row,
                    field,
                    resolved_time.physical_time,
                )
            else:
                values = capabilities.field_values(frame, row, field)
            values_by_label[label] = capabilities.field_display_values(frame, field, values)
        shared_norm = layout.linear_norm(*values_by_label.values()) if lock_scale and values_by_label else None
        rightmost_compatible_index = max(index for index, label in enumerate(dataset_labels) if label in compatible)
        for dataset_index, (label, frame) in enumerate(datasets.items()):
            axis = figure.add_subplot(map_grid[field_index, dataset_index])
            map_axes[field_index, dataset_index] = axis
            scientific_axes.append(axis)
            if field_index == 0:
                header_label = _section_three_dataset_header(label) if include_schedules else label
                header = f"{header_label}{_display_case_time(case_times.get(label))}"
                axis.text(
                    0.5,
                    1.035,
                    _compact_two_line_label(header),
                    ha="center",
                    va="bottom",
                    transform=axis.transAxes,
                    clip_on=False,
                )
            if label not in compatible:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    f"{field} unavailable",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue
            values = values_by_label[label]
            normalization = shared_norm if shared_norm is not None else layout.linear_norm(values)
            row = capabilities.case_row(frame, case_ids[label])
            x, y = capabilities.spatial_coordinates(frame, row)
            image = axis.pcolormesh(
                x,
                y,
                values,
                shading="auto",
                cmap=visual_semantics.field_visual_semantics(field).colormap,
                norm=normalization,
            )
            axis.set_aspect("equal")
            layout.add_map_colorbar(
                figure,
                image,
                axis,
                label=(
                    "[" + field_labels.display_unit(capabilities.field_display_unit(frame, field)) + "]"
                    if dataset_index == rightmost_compatible_index
                    else ""
                ),
            )
        layout.add_channel_row_label(
            map_axes[field_index, 0],
            capabilities.field_quantity_label(
                datasets[reference_label],
                field,
                mathtext=True,
            ),
            figure_x=(_TRANSIENT_ROW_LABEL_X if include_schedules else None),
        )
    _apply_eda_map_axis_labels(map_axes)
    availability = capabilities.availability_note(resolution)
    if availability:
        figure.text(
            0.035,
            0.015,
            availability,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    layout.set_suptitle_over_axes(
        figure,
        title,
        scientific_axes,
        y=_SPATIAL_SUPTITLE_Y,
    )
    return figure


def plot_spatial_fields(
    *,
    frame: pd.DataFrame,
    case_id: str,
    fields: Sequence[str] | None = None,
    dataset_name: str = "Dataset",
    lock_scale: bool = False,
    physical_time_hours: float | None = None,
) -> Figure:
    """Plot one case's spatial inputs, outputs, and transient state fields."""
    return plot_spatial_field_comparison(
        datasets={dataset_name: frame},
        case_ids={dataset_name: case_id},
        fields=fields,
        lock_scale=lock_scale,
        physical_time_hours=physical_time_hours,
    )


def plot_spatial_field_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    fields: Sequence[str] | None = None,
    lock_scale: bool = False,
    physical_time_hours: float | None = None,
) -> Figure:
    """Compare compatible spatial fields across steady and transient datasets."""
    return _spatial_map_figure(
        datasets=datasets,
        case_ids=case_ids,
        requested_fields=fields,
        field_view="spatial_map",
        lock_scale=lock_scale,
        physical_time_hours=physical_time_hours,
        include_schedules=False,
        title="Retained spatial fields",
    )


def plot_state_snapshot(
    *,
    frame: pd.DataFrame,
    case_id: str,
    physical_time_hours: float,
    channels: Sequence[str] | None = None,
    dataset_name: str = "Dataset",
    lock_scale: bool = False,
) -> Figure:
    """Plot one transient case with schedule context at one master time."""
    return plot_state_snapshot_comparison(
        datasets={dataset_name: frame},
        case_ids={dataset_name: case_id},
        physical_time_hours=physical_time_hours,
        channels=channels,
        lock_scale=lock_scale,
    )


def plot_state_snapshot_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    physical_time_hours: float,
    channels: Sequence[str] | None = None,
    lock_scale: bool = False,
) -> Figure:
    """Compare exact/latest-prior transient states with inlet schedules."""
    return _spatial_map_figure(
        datasets=datasets,
        case_ids=case_ids,
        requested_fields=channels,
        field_view="state_snapshot",
        lock_scale=lock_scale,
        physical_time_hours=physical_time_hours,
        include_schedules=True,
        title=f"Transient states and schedules at master t = {physical_time_hours:g} h",
    )


def _aggregate_supported_schedule(
    frame: pd.DataFrame,
    case_ids: Sequence[str],
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one schedule over only cases still within simulated support."""
    series = tuple(transient.supported_schedule_series(frame, case_id, quantity) for case_id in case_ids)
    if not series:
        message = "Aggregate schedule plotting requires at least one selected case."
        raise ValueError(message)
    physical_time = np.asarray(
        sorted({float(value) for item in series for value in item.physical_time_hours}),
        dtype=np.float64,
    )
    q10 = np.empty_like(physical_time)
    median = np.empty_like(physical_time)
    q90 = np.empty_like(physical_time)
    contributor_count = np.empty(physical_time.shape, dtype=np.int64)
    for index, time_value in enumerate(physical_time):
        contributors = tuple(
            item.value_at(float(time_value)) for item in series if item.physical_time_hours[0] <= time_value <= item.final_time_hours
        )
        if not contributors:
            message = "Aggregate schedule timeline contains unsupported physical time."
            raise RuntimeError(message)
        q10[index], median[index], q90[index] = np.quantile(
            np.asarray(contributors, dtype=np.float64),
            (0.1, 0.5, 0.9),
        )
        contributor_count[index] = len(contributors)
    return physical_time, q10, median, q90, contributor_count


def _split_physical_time_support(
    physical_time_hours: Sequence[float] | np.ndarray,
    *values: Sequence[float] | np.ndarray,
) -> tuple[
    tuple[np.ndarray, tuple[np.ndarray, ...]],
    tuple[np.ndarray, tuple[np.ndarray, ...]],
]:
    """Split exact stored support into 1-hour-to-final and 0-to-1-hour views."""
    physical_time = np.asarray(physical_time_hours, dtype=np.float64)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if (
        physical_time.ndim != 1
        or physical_time.size == 0
        or not np.isfinite(physical_time).all()
        or np.any(np.diff(physical_time) <= 0.0)
        or any(value.shape != physical_time.shape or not np.isfinite(value).all() for value in arrays)
    ):
        message = "Physical-time split requires aligned finite increasing series."
        raise ValueError(message)
    main_mask = physical_time >= _EARLY_OPERATION_END_HOURS
    startup_mask = physical_time <= _EARLY_OPERATION_END_HOURS
    return (
        (
            np.ascontiguousarray(physical_time[main_mask]),
            tuple(np.ascontiguousarray(value[main_mask]) for value in arrays),
        ),
        (
            np.ascontiguousarray(physical_time[startup_mask]),
            tuple(np.ascontiguousarray(value[startup_mask]) for value in arrays),
        ),
    )


def _plot_split_time_series(
    main_axis: Axes,
    startup_axis: Axes,
    physical_time_hours: np.ndarray,
    centre: np.ndarray,
    *,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    color: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot exact support in day-scale main and hour-scale startup views."""
    values = (centre,) if lower is None or upper is None else (centre, lower, upper)
    main, startup = _split_physical_time_support(
        physical_time_hours,
        *values,
    )
    for axis, (times, split_values), divisor in (
        (main_axis, main, _HOURS_PER_DAY),
        (startup_axis, startup, 1.0),
    ):
        if times.size == 0:
            continue
        display_times = times / divisor
        axis.plot(display_times, split_values[0], color=color, marker=None)
        if len(split_values) == _RANGE_SERIES_VALUE_COUNT:
            axis.fill_between(
                display_times,
                split_values[1],
                split_values[2],
                color=color,
                alpha=0.18,
            )
    return main[0], startup[0]


def _configure_split_time_axes(
    main_axis: Axes,
    startup_axis: Axes,
    main_times: Sequence[np.ndarray],
    startup_times: Sequence[np.ndarray],
) -> None:
    """Configure a day-scale main axis and exact 0-to-1-hour startup axis."""
    if main_times:
        combined_main = np.concatenate(tuple(main_times))
        main_display = _day_display(combined_main, include_zero=False)
        main_display.configure(main_axis)
        current_limits = main_axis.get_xlim()
        main_axis.set_xlim(
            left=_EARLY_OPERATION_END_HOURS / _HOURS_PER_DAY,
            right=max(
                current_limits[1],
                _EARLY_OPERATION_END_HOURS / _HOURS_PER_DAY,
            ),
        )
    else:
        main_axis.set_xlim(
            _EARLY_OPERATION_END_HOURS / _HOURS_PER_DAY,
            (_EARLY_OPERATION_END_HOURS + 0.05) / _HOURS_PER_DAY,
        )
        main_axis.set_xlabel("Time [d]")
    if startup_times:
        combined_startup = np.concatenate(tuple(startup_times))
        startup_display = time_axis.physical_time_display(
            combined_startup,
            preferred_unit="h",
            include_zero=True,
            major_interval_hours=None,
            minor_interval_hours=None,
        )
        startup_display.configure(startup_axis)
    else:
        startup_axis.set_xlabel("Time [h]")
    startup_axis.set_xlim(0.0, _EARLY_OPERATION_END_HOURS)


def _trajectory_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    channels: Sequence[str] | None,
    case_ids: Mapping[str, str] | None,
    max_cases: int | None,
) -> Figure:
    """Render exact-case or exact-coordinate aggregate schedules and trajectories."""
    if not datasets or (case_ids is None) == (max_cases is None):
        message = "Trajectory plotting requires exactly one case map or aggregate bound."
        raise ValueError(message)
    resolution = capabilities.resolve_fields(
        datasets,
        view="state_trajectory",
        requested=channels,
    )
    colors = _dataset_color_map(datasets)
    schedule_fields = _schedule_fields()
    rows = len(schedule_fields) + len(resolution.fields)
    figure = plt.figure(
        figsize=(11.8, _TRAJECTORY_ROW_HEIGHT_INCHES * rows),
        layout=None,
    )
    grid = figure.add_gridspec(
        rows,
        3,
        width_ratios=(1.0, 0.34, 0.30),
        height_ratios=([1.0] * rows),
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.95,
        wspace=0.30,
        hspace=0.38 / 4.0,
    )
    main_axes = tuple(figure.add_subplot(grid[row, 0]) for row in range(rows))
    startup_axes = tuple(figure.add_subplot(grid[row, 1]) for row in range(rows))
    for index, schedule_field in enumerate(schedule_fields):
        layout.add_channel_row_label(
            main_axes[index],
            field_labels.field_label_with_unit(
                schedule_field.name,
                schedule_field.unit,
                mathtext=True,
            ),
            axis_x=_TRAJECTORY_ROW_LABEL_X,
        )
        main_times: list[np.ndarray] = []
        startup_times: list[np.ndarray] = []
        for label, frame in datasets.items():
            if case_ids is not None:
                series = transient.supported_schedule_series(
                    frame,
                    case_ids[label],
                    schedule_field.name,
                )
                physical_time = series.physical_time_hours
                q10 = None
                median = field_labels.display_values(
                    series.values,
                    schedule_field.unit,
                )
                q90 = None
            else:
                if max_cases is None:
                    message = "Aggregate trajectory plotting lost its case bound."
                    raise RuntimeError(message)
                selected_case_ids = tuple(str(value) for value in frame.index[:max_cases])
                physical_time, q10, median, q90, _contributors = _aggregate_supported_schedule(
                    frame,
                    selected_case_ids,
                    schedule_field.name,
                )
                q10 = field_labels.display_values(q10, schedule_field.unit)
                median = field_labels.display_values(median, schedule_field.unit)
                q90 = field_labels.display_values(q90, schedule_field.unit)
            main_support, startup_support = _plot_split_time_series(
                main_axes[index],
                startup_axes[index],
                physical_time,
                median,
                lower=q10,
                upper=q90,
                color=colors[label],
            )
            if main_support.size:
                main_times.append(main_support)
            if startup_support.size:
                startup_times.append(startup_support)
        _configure_split_time_axes(
            main_axes[index],
            startup_axes[index],
            main_times,
            startup_times,
        )
        main_axes[index].grid(axis="y", alpha=0.20)
        startup_axes[index].grid(axis="y", alpha=0.20)

    trajectory_offset = len(schedule_fields)
    for field_index, field in enumerate(resolution.fields):
        row_index = trajectory_offset + field_index
        compatible = capabilities.compatible_frames(
            datasets,
            resolution,
            field,
        )
        reference = next(iter(compatible.values()))
        layout.add_channel_row_label(
            main_axes[row_index],
            capabilities.field_quantity_label(
                reference,
                field,
                mathtext=True,
            ),
            axis_x=_TRAJECTORY_ROW_LABEL_X,
        )
        main_times = []
        startup_times = []
        for label, frame in compatible.items():
            if case_ids is not None:
                table = transient.trajectory_table(
                    frame,
                    case_ids[label],
                    channels=(field,),
                )
                physical_time = table["physical_time_hours"].to_numpy(dtype=float)
                median = capabilities.field_display_values(
                    frame,
                    field,
                    table["spatial_mean"].to_numpy(dtype=float),
                )
                lower = capabilities.field_display_values(
                    frame,
                    field,
                    table["spatial_minimum"].to_numpy(dtype=float),
                )
                upper = capabilities.field_display_values(
                    frame,
                    field,
                    table["spatial_maximum"].to_numpy(dtype=float),
                )
            else:
                if max_cases is None:
                    message = "Aggregate trajectory plotting lost its case bound."
                    raise RuntimeError(message)
                times_by_case = []
                means_by_case = []
                for case_id in tuple(str(value) for value in frame.index[:max_cases]):
                    table = transient.trajectory_table(
                        frame,
                        case_id,
                        channels=(field,),
                    )
                    times_by_case.append(table["physical_time_hours"].to_numpy(dtype=float))
                    means_by_case.append(
                        capabilities.field_display_values(
                            frame,
                            field,
                            table["spatial_mean"].to_numpy(dtype=float),
                        )
                    )
                shared_times = time_axis.ordered_time_intersection(times_by_case)
                if not shared_times:
                    continue
                physical_time = np.asarray(shared_times, dtype=float)
                stacked = np.stack(
                    [
                        values[np.searchsorted(times, physical_time)]
                        for times, values in zip(
                            times_by_case,
                            means_by_case,
                            strict=True,
                        )
                    ]
                )
                lower, median, upper = np.quantile(
                    stacked,
                    (0.1, 0.5, 0.9),
                    axis=0,
                )
            main_support, startup_support = _plot_split_time_series(
                main_axes[row_index],
                startup_axes[row_index],
                physical_time,
                median,
                lower=lower,
                upper=upper,
                color=colors[label],
            )
            if main_support.size:
                main_times.append(main_support)
            if startup_support.size:
                startup_times.append(startup_support)
        _configure_split_time_axes(
            main_axes[row_index],
            startup_axes[row_index],
            main_times,
            startup_times,
        )
        if not main_times and not startup_times:
            main_axes[row_index].text(
                0.5,
                0.5,
                "Trajectory unavailable on shared exact times",
                ha="center",
                va="center",
                transform=main_axes[row_index].transAxes,
            )
        main_axes[row_index].grid(axis="y", alpha=0.20)
        startup_axes[row_index].grid(axis="y", alpha=0.20)

    main_axes[0].set_title("Physical time from 1 h to final support")
    startup_axes[0].set_title("Startup: 0-1 h")
    for axis in (*main_axes[:-1], *startup_axes[:-1]):
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelbottom=False)
    legend_axis = figure.add_subplot(grid[:, 2])
    legend_axis.set_axis_off()
    handles: list[object] = [
        Line2D(
            [],
            [],
            color=colors[label],
            label=label,
        )
        for label in datasets
    ]
    handles.extend(
        (
            Line2D([], [], color="black", label="Mean / median"),
            Patch(
                facecolor="black",
                alpha=0.18,
                label="Range / q10-q90",
            ),
        )
    )
    legend_axis.legend(handles=handles, loc="upper left", frameon=False)
    scope = "single exact cases" if case_ids is not None else f"first {max_cases} cases"
    layout.set_suptitle_over_axes(
        figure,
        f"Schedules and reference physical trajectories — {scope}",
        (*main_axes, *startup_axes),
        y=0.98,
    )
    return figure


def plot_state_trajectories(
    *,
    frame: pd.DataFrame,
    case_id: str,
    channels: Sequence[str] | None = None,
    dataset_name: str = "Dataset",
) -> Figure:
    """Plot one exact case's schedules and physical state trajectories."""
    return plot_state_trajectory_comparison(
        datasets={dataset_name: frame},
        case_ids={dataset_name: case_id},
        channels=channels,
    )


def plot_state_trajectory_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    channels: Sequence[str] | None = None,
) -> Figure:
    """Compare exact case schedules and trajectories without resampling."""
    if tuple(case_ids) != tuple(datasets):
        message = "Transient trajectory datasets and case IDs must match in order."
        raise ValueError(message)
    return _trajectory_figure(
        datasets=datasets,
        channels=channels,
        case_ids=case_ids,
        max_cases=None,
    )


def plot_state_trajectory_summary(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    channels: Sequence[str] | None = None,
) -> Figure:
    """Aggregate schedules and states only on exact shared stored times."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        message = "max_cases must be a positive integer."
        raise ValueError(message)
    return _trajectory_figure(
        datasets=datasets,
        channels=channels,
        case_ids=None,
        max_cases=max_cases,
    )


def plot_realized_parameters(
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> Figure:
    """Plot completed-case distributions for each material conditioning scalar."""
    if not datasets:
        message = "Realized-parameter plotting requires at least one dataset."
        raise ValueError(message)
    fields = transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning
    figure, axes = _field_axes(len(fields), width=4.4, height=3.5)
    for axis, field in zip(axes, fields, strict=False):
        values_by_dataset = []
        labels = []
        for label, frame in datasets.items():
            table = transient.scalar_parameter_table(frame)
            values = table.loc[table["parameter"] == field.name, "value"].to_numpy(dtype=float)
            if values.size:
                values_by_dataset.append(field_labels.display_values(values, field.unit))
                labels.append(label)
        axis.boxplot(values_by_dataset, tick_labels=labels, showmeans=True)
        axis.set(
            title=field.name,
            ylabel=field_labels.field_label_with_unit(field.name, field.unit),
        )
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    layout.set_suptitle_over_axes(
        figure,
        "Realized completed-case material parameters",
        axes,
    )
    return figure


def completion_target_detail_table(
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return exact selected eligible-case values for the companion table."""
    analysis = transient.completion_target_analysis(datasets)
    columns = {
        "dataset": "Dataset",
        "case_id": "Case",
        "completion_state": "Completion state",
        "terminal_time_kind": "Terminal-time meaning",
        "time_to_target_hours": "Target time [h]",
        "physical_duration_hours": "Final physical time [h]",
        "configured_horizon_hours": "Configured maximum [h]",
        "final_bulk_moisture_wb": "Final bulk moisture [wb]",
        "target_moisture_wb": "Target moisture [wb]",
        "final_wet_fraction": "Final dry-matter wet fraction [1]",
        "target_wet_fraction_limit": "Allowed wet-fraction limit [1]",
        "final_target_gap": "Wet-fraction target gap [1]",
        "stationary_airflow_solver_seconds": "Stationary airflow solver [s]",
        "transient_drying_solver_seconds": "Transient drying solver [s]",
        "scientific_solver_seconds": "Scientific solver [s]",
        "comsol_process_seconds": "COMSOL process [s]",
        "queue_wait_seconds": "Queue wait [s]",
        "licence_wait_seconds": "Licence wait [s]",
        "generation_compute_end_to_end_seconds": "Generation compute end-to-end [s]",
        "complete_execution_seconds": "Complete execution [s]",
    }
    available = [name for name in columns if name in analysis.cases]
    result = analysis.cases.loc[:, available].rename(columns=columns).copy()
    for column in (
        "Target time [h]",
        "Final physical time [h]",
        "Configured maximum [h]",
    ):
        if column not in result:
            continue
        result[column] = result[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else time_axis.format_terminal_physical_time_hours(
                    float(value),
                    include_unit=False,
                )
            )
        )
    return result


def _completion_time_headroom(maximum_hours: float) -> float:
    """Return a small bounded margin beyond one physical-time maximum."""
    if not np.isfinite(maximum_hours) or maximum_hours < 0.0:
        message = "Completion-time maximum must be finite and non-negative."
        raise ValueError(message)
    return min(6.0, max(1.0, 0.035 * max(maximum_hours, 1.0)))


def plot_completion_target_analysis(
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> Figure:
    """Plot distinct outcome-share, timing, and final-state questions."""
    analysis = transient.completion_target_analysis(datasets)
    colors = _dataset_color_map(datasets)
    figure = plt.figure(figsize=(_COMPLETION_FIGURE_WIDTH_INCHES, 9.4), layout=None)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(
            _COMPLETION_SCIENCE_COLUMN_RATIO,
            _COMPLETION_SCIENCE_COLUMN_RATIO,
            _COMPLETION_LEGEND_COLUMN_RATIO,
        ),
        height_ratios=(0.95, 1.05),
        left=0.08,
        right=0.98,
        bottom=0.045 + 1.25 * ((0.22 + 0.045) / 2.0 - 0.045),
        top=0.90,
        wspace=0.34,
        hspace=0.42 / 2.0,
    )
    outcome_axis = figure.add_subplot(grid[0, :2])
    time_plot_axis = figure.add_subplot(grid[1, 0])
    final_grid = grid[1, 1].subgridspec(2, 1, hspace=0.52)
    moisture_axis = figure.add_subplot(final_grid[0, 0])
    wet_fraction_axis = figure.add_subplot(final_grid[1, 0], sharex=moisture_axis)
    legend_axis = figure.add_subplot(grid[:, 2])
    legend_axis.set_axis_off()

    category_positions = np.arange(len(_OUTCOME_LABELS), dtype=np.float64)
    dataset_count = len(datasets)
    bar_width = 0.80 / max(dataset_count, 1)
    zero_eligible: list[str] = []
    for dataset_index, label in enumerate(datasets):
        rows = analysis.outcomes.loc[analysis.outcomes["dataset"] == label]
        denominator = int(rows["eligible_case_count"].iloc[0])
        if denominator == 0:
            zero_eligible.append(label)
            continue
        percentages = rows["percentage"].to_numpy(dtype=np.float64)
        counts = rows["count"].to_numpy(dtype=int)
        offsets = category_positions + (dataset_index - (dataset_count - 1) / 2.0) * bar_width
        bars = outcome_axis.bar(
            offsets,
            percentages,
            width=bar_width * 0.92,
            color=colors[label],
            alpha=0.88,
        )
        for bar, percentage, count in zip(
            bars,
            percentages,
            counts,
            strict=True,
        ):
            outcome_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(percentage + 2.0, 106.0),
                f"{percentage:.1f}% ({count}/{denominator})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=0,
            )
    outcome_axis.set_xticks(category_positions, _OUTCOME_LABELS)
    outcome_axis.set_ylim(0.0, 110.0)
    outcome_axis.set_yticks(np.arange(0.0, 101.0, 20.0))
    outcome_axis.set_ylabel("Cases [%]")
    outcome_axis.set_title("Outcome share among eligible target-diagnostic cases")
    outcome_axis.grid(axis="y", alpha=0.25)
    if zero_eligible:
        outcome_axis.text(
            0.5,
            0.04,
            "No eligible cases: " + ", ".join(zero_eligible),
            transform=outcome_axis.transAxes,
            ha="center",
            va="bottom",
        )

    eligible_cases = analysis.cases
    if eligible_cases.empty:
        for axis, message in (
            (time_plot_axis, "No eligible terminal-time evidence."),
            (moisture_axis, "No eligible final-moisture evidence."),
            (wet_fraction_axis, "No eligible wet-fraction evidence."),
        ):
            axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
    else:
        terminal_values = np.where(
            eligible_cases["target_reached"].to_numpy(dtype=bool),
            eligible_cases["time_to_target_hours"].to_numpy(dtype=float),
            eligible_cases["physical_duration_hours"].to_numpy(dtype=float),
        )
        configured_horizons = np.asarray(
            pd.to_numeric(
                eligible_cases.loc[:, "configured_horizon_hours"],
                errors="coerce",
            ),
            dtype=float,
        )
        finite_horizons = configured_horizons[np.isfinite(configured_horizons)]
        extent_values = np.concatenate((terminal_values, finite_horizons)) if finite_horizons.size else terminal_values
        maximum_hours = float(np.max(extent_values))
        display_time = _day_display(
            extent_values,
            right_margin_hours=_completion_time_headroom(maximum_hours),
        )
        for dataset_index, label in enumerate(datasets):
            rows = eligible_cases.loc[eligible_cases["dataset"] == label]
            if rows.empty:
                continue
            offsets = np.linspace(-0.16, 0.16, len(rows)) if len(rows) > 1 else np.asarray([0.0])
            for offset, (_, row) in zip(offsets, rows.iterrows(), strict=True):
                reached = bool(row["target_reached"])
                value = float(row["time_to_target_hours"]) if reached else float(row["physical_duration_hours"])
                terminal_kind = str(row["terminal_time_kind"])
                marker = "o" if reached else ("s" if terminal_kind == "configured_maximum_duration" else "x")
                time_plot_axis.scatter(
                    display_time.values((value,))[0],
                    dataset_index + offset,
                    color=colors[label],
                    marker=marker,
                    alpha=0.85,
                )
        display_time.configure(time_plot_axis)
        time_plot_axis.set_yticks([])
        time_plot_axis.set_ylabel("Dataset rows (see legend)")
        time_plot_axis.set_title("Target-attainment and censoring times")
        time_plot_axis.grid(axis="y", alpha=0.15)

        position = 0
        for label in datasets:
            rows = eligible_cases.loc[eligible_cases["dataset"] == label]
            count = len(rows)
            if count == 0:
                continue
            positions = np.arange(position, position + count)
            moisture_axis.scatter(
                positions,
                rows["final_bulk_moisture_wb"],
                color=colors[label],
                marker="o",
                alpha=0.85,
            )
            moisture_axis.scatter(
                positions,
                rows["target_moisture_wb"],
                color=colors[label],
                marker="_",
                linewidths=2.0,
            )
            wet_fraction_axis.scatter(
                positions,
                rows["final_wet_fraction"],
                color=colors[label],
                marker="o",
                alpha=0.85,
            )
            wet_fraction_axis.scatter(
                positions,
                rows["target_wet_fraction_limit"],
                color=colors[label],
                marker="_",
                linewidths=2.0,
            )
            position += count
        moisture_axis.tick_params(axis="x", labelbottom=False)
        moisture_axis.set_ylabel("Moisture [wb]")
        moisture_axis.set_title("Final bulk moisture and target")
        moisture_axis.grid(alpha=0.25)
        wet_fraction_axis.set_xlabel("Eligible case position")
        wet_fraction_axis.set_ylabel("Wet fraction [1]")
        wet_fraction_axis.set_title("Final wet fraction and allowed limit")
        wet_fraction_axis.grid(alpha=0.25)

    dataset_handles = [
        Line2D(
            [],
            [],
            color=colors[label],
            marker="o",
            linestyle="None",
            label=label,
        )
        for label in datasets
    ]
    semantic_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="Target time / final value"),
        Line2D([], [], color="black", marker="s", linestyle="None", label="Maximum-duration censoring"),
        Line2D([], [], color="black", marker="_", linestyle="None", label="Target / allowed limit"),
    ]
    legend_axis.legend(
        handles=[*dataset_handles, *semantic_handles],
        loc="upper left",
        frameon=False,
    )
    omission_lines = []
    for _, row in analysis.omissions.iterrows():
        reasons = dict(row["omission_reasons"])
        reason_text = ", ".join(f"{name.replace('_', ' ')}={count}" for name, count in sorted(reasons.items()))
        suffix = f" ({reason_text})" if reason_text else ""
        eligible_count = _integer_count(
            row["eligible_case_count"],
            label="Eligible case count",
        )
        omitted_count = _integer_count(
            row["omitted_case_count"],
            label="Omitted case count",
        )
        omission_lines.append(f"{row['dataset']} — eligible cases: {eligible_count}; non-eligible omitted: {omitted_count}{suffix}.")
    figure.text(
        0.08,
        0.045,
        "\n".join(omission_lines),
        ha="left",
        va="bottom",
        linespacing=1.25,
        fontsize=8.8,
    )
    layout.set_suptitle_over_axes(
        figure,
        "Completion and target attainment",
        (outcome_axis, time_plot_axis, moisture_axis, wet_fraction_axis),
    )
    return figure
