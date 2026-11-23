"""Plot transient completed-output exploratory diagnostics.

Responsibilities:
  - Render exact physical-time state and static-field snapshots
  - Plot physical trajectories and simulated-support schedules in a main and a start-up time window (the shared windows of the ui time module);
    the aggregate state rows alternatively over the drying phase of ``analysis_ui_time.phase_cohort``, its cohort labels in the note
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

import itertools
from numbers import Integral
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src import generation
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_transient as transient
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_moisture_display as moisture_display
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_axis

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LINE_ROW_HEIGHT_INCHES: Final = 1.65 * 4.0 / 3.0
_PRESSURE_SCHEDULE_GROUP_SPACING: Final = 0.24 * 1.25
_SCHEDULE_ROW_SPACING: Final = 0.24
_LEGEND_WIDTH_INCHES: Final = 2.4
LEGEND_WRAP_WIDTH: Final = 26  # characters per line of a dataset legend entry
_ORDER_COLORS: Final = ("#000000", "#F0E442", "#7F7F7F", "#332288", "#882255", "#AA4499", "#DDCC77", "#44AA99")  # no material color
_BATCH_LINESTYLES: Final[tuple[visual_semantics.LineStyle, ...]] = ("-", "--", ":", "-.")  # batches of one material, by sorted label rank
_HATCH_COLOR: Final = "white"  # Matplotlib draws a hatch in the edge color; white shows on every dataset color
_BATCH_HATCHES: Final = ("", "//", "..", "xx")  # the same rank as bar fill pattern
FOOTNOTE_INCHES: Final = (0.1, 0.08)  # lower-left corner of a figure's disclosure line
_TRAJECTORY_ROW_LABEL_X: Final = -0.28 * 0.5
_TRAJECTORY_ROW_HEIGHT_INCHES: Final = 4.0 * 0.75
_TRAJECTORY_WIDTH_INCHES: Final = 9.0
_EARLY_OPERATION_END_HOURS: Final = time_axis.STARTUP_WINDOWS[1].stop_hours
_RANGE_SERIES_VALUE_COUNT: Final = 3
_MINIMUM_DISPERSION_SUPPORT: Final = 2
_DATASET_HEADER_WRAP_WIDTH: Final = 38
_MAXIMUM_DATASET_HEADER_LINES: Final = 2
_OUTCOME_LABELS: Final = ("Reached target", "Right-censored")
_COMPLETION_FIGURE_WIDTH_INCHES: Final = 10.5


def batch_styles(materials: Mapping[str, str | None]) -> dict[str, visual_semantics.SeriesStyle]:
    """
    Return the line color, style and bar hatch of every dataset, keyed by concise dataset label; the one mapping of the EDA plots.

    ``materials`` maps every dataset label to its one registered material or ``None``. A dataset with a registered material always
    owns that material's color. Every other dataset takes, by its label's sorted rank, a solid line in a color of ``_ORDER_COLORS``,
    which no material uses. Datasets that share one material tell apart by line style and hatch, taken by the sorted rank of their
    labels; the first stays solid and unhatched. Viewers resolve this once over every loaded dataset, so a style never follows a selection.
    """
    materials = {label: material if material in visual_semantics.MATERIAL_COLORS else None for label, material in materials.items()}
    unmatched = sorted(label for label, material in materials.items() if material is None)
    styles: dict[str, visual_semantics.SeriesStyle] = {}
    for label, material in materials.items():
        if material is not None:
            same = sorted(other for other, candidate in materials.items() if candidate == material)
            rank = same.index(label)
            styles[label] = visual_semantics.SeriesStyle(
                visual_semantics.material_color(material),
                _BATCH_LINESTYLES[rank % len(_BATCH_LINESTYLES)],
                None,
                _BATCH_HATCHES[rank % len(_BATCH_HATCHES)],
            )
        else:
            styles[label] = visual_semantics.SeriesStyle(_ORDER_COLORS[unmatched.index(label) % len(_ORDER_COLORS)], "-", None)
    return styles


def _integer_count(value: object, *, label: str) -> int:
    """Return one exact non-negative plotted case count."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        message = f"{label} must be one non-negative integer count."
        raise TypeError(message)
    return int(value)


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
    linestyles: Mapping[str, str],
    master_time_hours: float,
    case_times: Mapping[str, time_axis.ResolvedPhysicalTime],
) -> None:
    """Draw exact clipped schedules in day-scale main and hour-scale startup views."""
    fields = _schedule_fields()
    if len(main_axes) != len(fields) or len(startup_axes) != len(fields):
        message = "Split schedule axes must match authoritative schedule quantities."
        raise ValueError(message)
    main_support_by_row: list[list[np.ndarray]] = [[] for _field in fields]
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
                field=field.name,
            )
            main, startup = _split_physical_time_support(
                series.physical_time_hours,
                display_values,
            )
            main_times, (main_values,) = main
            startup_times, (startup_values,) = startup
            if main_times.size:
                main_axes[field_index].plot(
                    main_times,
                    main_values,
                    color=colors[label],
                    linestyle=linestyles[label],
                    marker=None,
                )
                main_support_by_row[field_index].append(main_times)
            if startup_times.size:
                startup_axes[field_index].plot(
                    startup_times,
                    startup_values,
                    color=colors[label],
                    linestyle=linestyles[label],
                    marker=None,
                )
            actual = case_times.get(label)
            if actual is None:
                message = f"Current schedule marker lacks resolved case time for {label!r}."
                raise ValueError(message)
            marker_value = field_labels.display_values(
                (series.value_at(actual.physical_time),),
                field.unit,
                field=field.name,
            )[0]
            marker_axis = startup_axes[field_index] if actual.physical_time < _EARLY_OPERATION_END_HOURS else main_axes[field_index]
            marker_axis.scatter(
                (actual.physical_time,),
                (marker_value,),
                color=colors[label],
                marker="o",
                s=28,
                zorder=4,
            )
    for field_index, (main_axis, startup_axis) in enumerate(zip(main_axes, startup_axes, strict=True)):
        _configure_split_time_axes(main_axis, startup_axis, main_support_by_row[field_index], bottom=field_index == len(fields) - 1)
        (startup_axis if master_time_hours < _EARLY_OPERATION_END_HOURS else main_axis).axvline(
            master_time_hours,
            color="black",
            linestyle=":",
            linewidth=1.2,
        )
        main_axis.grid(axis="y", alpha=0.20)
        startup_axis.grid(axis="y", alpha=0.20)


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


def _draw_inlet_pressure_axis(
    axis: Axes,
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    colors: Mapping[str, str],
    linestyles: Mapping[str, str],
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
            linestyle=linestyles[label],
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
    styles: Mapping[str, visual_semantics.SeriesStyle],
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render one union-capability map grid with optional exact split schedules, whose lines take their ``styles``."""
    if not datasets or tuple(case_ids) != tuple(datasets):
        message = "Spatial-map datasets and case IDs must match in order."
        raise ValueError(message)
    if not isinstance(lock_scale, bool):
        message = "lock_scale must be boolean."
        raise TypeError(message)
    if include_schedules and not _has_complete_schedules(datasets):
        include_schedules = False
        title = "Reference states; full authored schedules unavailable in this Dataset view"
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
    used = {label for field in resolution.fields for label in resolution.datasets_by_field[field]}
    rows = {label: capabilities.case_row(datasets[label], case_ids[label]) for label in datasets if label in used}
    coordinates = {label: capabilities.spatial_coordinates(datasets[label], row) for label, row in rows.items()}
    cells: dict[tuple[str, str], tuple[np.ndarray, field_maps.FieldMapDefinition]] = {}
    for field in resolution.fields:
        compatible = [label for label in datasets if label in resolution.datasets_by_field[field]]
        units = {capabilities.field_value_unit(datasets[label], field) for label in compatible}
        representations = {capabilities.field_representation(datasets[label], field) for label in compatible}
        if len(units) != 1 or len(representations) != 1:
            message = f"Spatial field {field!r} has incompatible units or representations."
            raise ValueError(message)
        values_by_label: dict[str, np.ndarray] = {}
        for label in compatible:
            frame = datasets[label]
            values = (
                capabilities.field_values_at_physical_time(frame, rows[label], field, case_times[label].physical_time)
                if capabilities.is_dynamic_field(frame, field)
                else capabilities.field_values(frame, rows[label], field)
            )
            values_by_label[label] = capabilities.field_display_values(frame, rows[label], field, values)
        unit, representation = next(iter(units)), next(iter(representations))
        shared = spatial_field_scale(field, unit, representation, *values_by_label.values()) if lock_scale else None
        for label, values in values_by_label.items():
            cells[field, label] = (values, shared if shared is not None else spatial_field_scale(field, unit, representation, values))
    extents = [display_grid.physical_extent(*coordinates[label]) for label in coordinates]
    extent = (min(e[0] for e in extents), max(e[1] for e in extents), min(e[2] for e in extents), max(e[3] for e in extents))
    schedule_fields = _schedule_fields() if include_schedules else ()
    geometry = layout.plan_geometry(
        len(resolution.fields),
        len(datasets),
        (extent[1] - extent[0], extent[3] - extent[2]),
        width_cap=None,
        above_in=(1 + len(schedule_fields)) * _LINE_ROW_HEIGHT_INCHES if include_schedules else 0.0,
        right_in=_LEGEND_WIDTH_INCHES if include_schedules else 0.0,
    )
    figure, axes, slots = layout.map_grid(geometry, extent=extent)
    try:
        if include_schedules:
            _draw_line_band(figure, geometry, datasets, case_ids, schedule_fields, master_time, case_times, styles)
        for (field_index, field), (column, (label, frame)) in itertools.product(enumerate(resolution.fields), enumerate(datasets.items())):
            if (field, label) not in cells:
                layout.empty_cell(axes[field_index, column], f"{field} unavailable")
                continue
            values, scale = cells[field, label]
            indices = display_grid.display_indices(values.shape, display_spatial_stride)
            x, y = coordinates[label]
            image = axes[field_index, column].pcolormesh(
                display_grid.display_sample(x, indices),
                display_grid.display_sample(y, indices),
                display_grid.display_sample(values, indices),
                shading="auto",
                **scale.pcolormesh_kwargs(),
            )
            unit_label = f"[{capabilities.field_display_unit(frame, field)}]"
            layout.add_field_map_colorbar(figure, image, slots[field_index, column], scale, label=unit_label)
        for field_index, field in enumerate(resolution.fields):  # an empty first cell hands its y labels to the row's first map
            first = next((column for column, label in enumerate(datasets) if (field, label) in cells), 0)
            if first:
                axes[field_index, first].set_ylabel("y [m]", fontsize=figure_layout.LABEL_PT)
                axes[field_index, first].tick_params(labelleft=True)
        names = [_section_three_dataset_header(label) if include_schedules else label for label in datasets]
        headers = [
            display_labels.wrapped_dataset_display_label(
                f"{name}{_display_case_time(case_times.get(label))}", width=_DATASET_HEADER_WRAP_WIDTH, max_lines=_MAXIMUM_DATASET_HEADER_LINES
            )
            for name, label in zip(names, datasets, strict=True)
        ]
        layout.set_column_headers(axes, headers, ())
        row_labels = [
            capabilities.field_display_name(next(iter(capabilities.compatible_frames(datasets, resolution, field).values())), field, mathtext=True)
            for field in resolution.fields
        ]
        layout.set_row_labels(axes, row_labels)
        layout.set_map_title(figure, f"{title} · {next(iter(case_ids.values()))}" if len(datasets) == 1 else title)
        if availability := capabilities.availability_note(resolution):
            figure.text(
                FOOTNOTE_INCHES[0] / figure.get_figwidth(), FOOTNOTE_INCHES[1] / figure.get_figheight(), availability, va="bottom", fontsize=8
            )
    except Exception:
        plt.close(figure)
        raise
    return figure


def _draw_line_band(
    figure: Figure,
    geometry: layout.MapGeometry,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    schedule_fields: Sequence[generation.contracts.FieldContract],
    master_time: float | None,
    case_times: Mapping[str, time_axis.ResolvedPhysicalTime],
    styles: Mapping[str, visual_semantics.SeriesStyle],
) -> None:
    """Draw the inlet pressure and the split schedules in the band above the maps, the dataset legend right of it from the top."""
    if master_time is None:
        message = "Transient schedule maps require one selected master time."
        raise ValueError(message)
    colors = {label: style.color for label, style in styles.items()}
    linestyles = {label: style.linestyle for label, style in styles.items()}
    left, bottom, width, height = layout.band_rect(geometry, "above")
    outer = figure.add_gridspec(
        2,
        1,
        left=left,
        right=left + width,
        bottom=bottom,
        top=bottom + height,
        height_ratios=(1.0, float(len(schedule_fields))),
        hspace=_PRESSURE_SCHEDULE_GROUP_SPACING,
    )
    pressure_axis = figure.add_subplot(outer[0, 0])
    _draw_inlet_pressure_axis(pressure_axis, datasets=datasets, case_ids=case_ids, colors=colors, linestyles=linestyles)
    schedule_axes = layout.window_axes(
        figure, outer[1, 0], time_axis.STARTUP_WINDOWS, [1.0] * len(schedule_fields), wspace=0.30, hspace=_SCHEDULE_ROW_SPACING
    )
    main_axes, startup_axes = (tuple(schedule_axes[:, column]) for column in range(len(time_axis.STARTUP_WINDOWS)))
    row_axes = np.empty((len(main_axes) + 1, 1), dtype=object)
    row_axes[:, 0] = (pressure_axis, *main_axes)
    layout.set_row_labels(
        row_axes,
        [
            capabilities.field_quantity_label(next(iter(datasets.values())), "p_in_bc", mathtext=True),
            *(field_labels.field_label_with_unit(field.name, field.unit, mathtext=True) for field in schedule_fields),
        ],
    )
    _draw_exact_schedule_split_axes(
        main_axes,
        startup_axes,
        datasets=datasets,
        case_ids=case_ids,
        colors=colors,
        linestyles=linestyles,
        master_time_hours=master_time,
        case_times=case_times,
    )
    for axis, window in zip(schedule_axes[0], time_axis.STARTUP_WINDOWS, strict=True):
        axis.set_title(window.title)
    handles = [
        Line2D([], [], color=colors[label], label=display_labels.wrapped_dataset_display_label(label, width=LEGEND_WRAP_WIDTH)) for label in datasets
    ]
    for handle, label in zip(handles, datasets, strict=True):
        handle.set(linestyle=linestyles[label])
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(layout.band_rect(geometry, "right")[0], bottom + height),
        borderaxespad=0.0,
        frameon=False,
    )


def _has_complete_schedules(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Keep runtime interval conditioning distinct from a complete source schedule."""
    return all(frame.attrs.get("complete_schedule_available", True) for frame in datasets.values())


def spatial_field_scale(field: str, unit: str, representation: str, *values: np.ndarray) -> field_maps.FieldMapDefinition:
    """
    Build one data-derived scale (round bands; geometric for permeability) from full arrays; masked unphysical water cells stay off it.

    Only a quantity signed by meaning (``Kxy``, a ``delta_`` field, a centred visual semantic) is diverging about 0; any other
    one, also with negative values (log10 permeability, a pressure undershoot), is sequential over its own range.
    """
    combined = np.concatenate(tuple(np.ravel(value) for value in values))
    if moisture_display.is_moisture_field(field):
        combined = combined[np.isfinite(combined)]
    semantic = field_maps.FieldMapSemantic.LINEAR_SEQUENTIAL if np.any(combined < 0.0) else field_maps.FieldMapSemantic.LINEAR_POSITIVE
    if field in {"Kxx", "Kyy", "K_min", "K_max"} and unit == "m^2":
        semantic = field_maps.FieldMapSemantic.LOG_POSITIVE
    elif field == "Kxy" or field.startswith("delta_") or visual_semantics.field_visual_semantics(field).centered:
        semantic = field_maps.FieldMapSemantic.LINEAR_SIGNED
    return field_maps.build_field_map(combined, semantic=semantic, field=field, unit=unit, display_transform=representation)


def plot_spatial_field_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    fields: Sequence[str] | None = None,
    lock_scale: bool = False,
    physical_time_hours: float | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Compare compatible spatial fields across steady and transient datasets."""
    return _spatial_map_figure(
        datasets=datasets,
        case_ids=case_ids,
        requested_fields=fields,
        field_view="spatial_map",
        lock_scale=lock_scale,
        display_spatial_stride=display_spatial_stride,
        physical_time_hours=physical_time_hours,
        include_schedules=False,
        title="Retained spatial fields",
        styles={},  # no lines are drawn
    )


def plot_state_snapshot_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    physical_time_hours: float,
    styles: Mapping[str, visual_semantics.SeriesStyle],
    channels: Sequence[str] | None = None,
    lock_scale: bool = False,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Compare exact/latest-prior transient states with inlet schedules, drawn in the dataset ``styles`` of the viewer."""
    return _spatial_map_figure(
        datasets=datasets,
        case_ids=case_ids,
        requested_fields=channels,
        field_view="state_snapshot",
        lock_scale=lock_scale,
        display_spatial_stride=display_spatial_stride,
        physical_time_hours=physical_time_hours,
        include_schedules=True,
        title=f"Transient states and schedules at master t = {physical_time_hours:g} h",
        styles=styles,
    )


def _aggregate_supported_schedule(
    frame: pd.DataFrame,
    case_ids: Sequence[str],
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate authored piecewise-linear schedules within their simulated support."""
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
    """Split exact stored support into the main (1 hour to final) and start-up (0 to 1 hour) windows."""
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
    main_mask, startup_mask = (window.select(physical_time) for window in time_axis.STARTUP_WINDOWS)
    return (
        (np.ascontiguousarray(physical_time[main_mask]), tuple(np.ascontiguousarray(value[main_mask]) for value in arrays)),
        (np.ascontiguousarray(physical_time[startup_mask]), tuple(np.ascontiguousarray(value[startup_mask]) for value in arrays)),
    )


def _plot_split_time_series(
    main_axis: Axes,
    startup_axis: Axes,
    physical_time_hours: np.ndarray,
    centre: np.ndarray,
    *,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    style: visual_semantics.SeriesStyle,
    contributor_count: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot exact support, in hours, in the main and startup windows in the color and line style of ``style``."""
    color = style.color
    values = (centre,) if lower is None or upper is None else (centre, lower, upper)
    has_range = len(values) == _RANGE_SERIES_VALUE_COUNT
    if contributor_count is not None:
        values = (*values, contributor_count)
    main, startup = _split_physical_time_support(
        physical_time_hours,
        *values,
    )
    for axis, (times, split_values) in ((main_axis, main), (startup_axis, startup)):
        if times.size == 0:
            continue
        axis.plot(times, split_values[0], color=color, linestyle=style.linestyle, marker=None)
        if has_range:
            axis.fill_between(
                times,
                split_values[1],
                split_values[2],
                color=color,
                alpha=0.18,
                where=split_values[-1] >= _MINIMUM_DISPERSION_SUPPORT if contributor_count is not None else None,
            )
    return main[0], startup[0]


def _configure_split_time_axes(main_axis: Axes, startup_axis: Axes, main_times: Sequence[np.ndarray], *, bottom: bool) -> None:
    """Style the main axis to the end of its shown support and the startup axis; ``bottom`` rows only carry time numbers and the label."""
    main_window, startup_window = time_axis.STARTUP_WINDOWS
    shown_end = max((float(times.max()) for times in main_times), default=None)
    time_axis.style_window_axis(main_axis, main_window, shown_end_hours=shown_end, bottom=bottom)
    time_axis.style_window_axis(startup_axis, startup_window, bottom=bottom)


def _contributor_axes(axes: Sequence[Axes], maximum: int) -> tuple[Axes, ...]:
    """Keep exact contributor counts distinct from each physical quantity."""
    result = tuple(axis.twinx() for axis in axes)
    for axis in result:
        axis.set_ylim(0.0, maximum + 0.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        axis.set_ylabel("n(t)")
    return result


def _plot_contributors(
    main_axis: Axes,
    startup_axis: Axes,
    physical_time_hours: np.ndarray,
    counts: np.ndarray,
    *,
    style: visual_semantics.SeriesStyle,
) -> None:
    """Disclose every stored-time count as a step line in the dataset's color and line style, without implying interpolated support."""
    main, startup = _split_physical_time_support(physical_time_hours, counts)
    for axis, (times, values) in ((main_axis, main), (startup_axis, startup)):
        axis.plot(times, values[0], color=style.color, linestyle=style.linestyle, drawstyle="steps-post", alpha=0.5)


def _case_series(frame: pd.DataFrame, field: str, max_cases: int | None) -> tuple[list[np.ndarray], list[np.ndarray], list[time_axis.Completion]]:
    """Return the stored hours, displayed spatial means and completion of the frame's first ``max_cases`` cases."""
    cases = [capabilities.case_row(frame, str(case_id)) for case_id in frame.index[:max_cases]]
    trajectories = [capabilities.field_display_trajectory(frame, case, field)[:2] for case in cases]
    return [times for times, _means in trajectories], [means for _times, means in trajectories], [transient.case_completion(case) for case in cases]


def _plot_phase_row(
    axes: tuple[Axes, Axes, Axes, Axes],
    series: Mapping[str, tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[time_axis.Completion]]],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    *,
    first: bool,
    bottom: bool,
) -> dict[str, str]:
    """Draw one state row's pooled median, q10-q90 band and n(τ) per dataset over the phase, hiding the start-up axes; return the cohort labels."""
    axis, count_axis, *hidden = axes
    for unused in hidden:
        unused.set_visible(False)
    count_axis.set_ylabel("n(τ)")
    time_axis.style_window_axis(axis, time_axis.PHASE_WINDOW, bottom=bottom)
    axis.grid(axis="y", alpha=0.20)
    if first:
        figure_layout.set_panel_title(axis, time_axis.PHASE_WINDOW.title)
    labels = {}
    for label, (times_by_case, means_by_case, completion) in series.items():
        lower, median, upper, cohort = transient.aggregate_phase_series(times_by_case, means_by_case, completion)
        style, grid, count = styles[label], time_axis.PHASE_GRID, len(cohort.cases)
        axis.plot(grid, median, color=style.color, linestyle=style.linestyle)
        if count >= _MINIMUM_DISPERSION_SUPPORT:
            axis.fill_between(grid, lower, upper, color=style.color, alpha=0.18)
        count_axis.plot(grid, np.full(grid.shape, count), color=style.color, linestyle=style.linestyle, drawstyle="steps-post", alpha=0.5)
        labels[label] = cohort.label()
    return labels


def _trajectory_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    channels: Sequence[str] | None,
    case_ids: Mapping[str, str] | None,
    max_cases: int | None,
    styles: Mapping[str, visual_semantics.SeriesStyle],
    clock: str = "hours",
) -> Figure:
    """Render exact-case or exact-coordinate aggregate schedules and trajectories (aggregate states optionally over the phase) in ``styles``."""
    if not datasets or (case_ids is None) == (max_cases is None):
        message = "Trajectory plotting requires exactly one case map or aggregate bound."
        raise ValueError(message)
    phase = time_axis.is_phase(clock)
    if phase and (lacking := [label for label, frame in datasets.items() if not transient.has_completion_evidence(frame)]):
        message = f"The drying phase pools reached cases, but datasets {lacking} lack completion evidence (published Datasets have none)."
        raise ValueError(message)
    phase_notes: dict[str, str] = {}
    resolution = capabilities.resolve_fields(
        datasets,
        view="state_trajectory",
        requested=channels,
    )
    schedule_fields = _schedule_fields() if _has_complete_schedules(datasets) else ()
    rows = len(schedule_fields) + len(resolution.fields)
    figure = plt.figure(figsize=(_TRAJECTORY_WIDTH_INCHES, _TRAJECTORY_ROW_HEIGHT_INCHES * rows), layout=figure_layout.panel_engine())
    maximum_count = max(min(max_cases, len(frame)) for frame in datasets.values()) if max_cases is not None else 0
    window_grid = layout.window_axes(figure, figure.add_gridspec(1, 1)[0, 0], time_axis.STARTUP_WINDOWS, [1.0] * rows)
    main_axes, startup_axes = (tuple(window_grid[:, column]) for column in range(len(time_axis.STARTUP_WINDOWS)))
    for axis, window in zip(window_grid[0], time_axis.STARTUP_WINDOWS, strict=True):
        figure_layout.set_panel_title(axis, window.title)  # a first state row over the phase retitles its main axis
    legend_panels = window_grid
    main_counts = _contributor_axes(main_axes, maximum_count) if maximum_count else ()
    startup_counts = _contributor_axes(startup_axes, maximum_count) if maximum_count else ()
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
        for label, frame in datasets.items():
            contributors = None
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
                    field=schedule_field.name,
                )
                q90 = None
            else:
                if max_cases is None:
                    message = "Aggregate trajectory plotting lost its case bound."
                    raise RuntimeError(message)
                selected_case_ids = tuple(str(value) for value in frame.index[:max_cases])
                physical_time, q10, median, q90, contributors = _aggregate_supported_schedule(
                    frame,
                    selected_case_ids,
                    schedule_field.name,
                )
                q10, median, q90 = (
                    field_labels.display_values(values, schedule_field.unit, field=schedule_field.name) for values in (q10, median, q90)
                )
            if contributors is not None:
                _plot_contributors(main_counts[index], startup_counts[index], physical_time, contributors, style=styles[label])
            main_support, _startup_support = _plot_split_time_series(
                main_axes[index],
                startup_axes[index],
                physical_time,
                median,
                lower=q10,
                upper=q90,
                style=styles[label],
                contributor_count=contributors,
            )
            if main_support.size:
                main_times.append(main_support)
        # Over the phase the state rows read τ, so the last schedule row carries the clock's numbers and label.
        bottom = index == rows - 1 or (phase and index == len(schedule_fields) - 1)
        _configure_split_time_axes(main_axes[index], startup_axes[index], main_times, bottom=bottom)
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
        if phase:
            row_axes = (main_axes[row_index], main_counts[row_index], startup_axes[row_index], startup_counts[row_index])
            phase_series = {label: _case_series(frame, field, max_cases) for label, frame in compatible.items()}
            phase_notes |= _plot_phase_row(row_axes, phase_series, styles, first=field_index == 0, bottom=row_index == rows - 1)
            continue
        main_times = []
        startup_times = []
        for label, frame in compatible.items():
            contributors = None
            if case_ids is not None:
                row = capabilities.case_row(frame, case_ids[label])
                physical_time, median, lower, upper = capabilities.field_display_trajectory(frame, row, field)
            else:
                if max_cases is None:
                    message = "Aggregate trajectory plotting lost its case bound."
                    raise RuntimeError(message)
                times_by_case, means_by_case, _completion = _case_series(frame, field, max_cases)
                physical_time, lower, median, upper, contributors = transient.aggregate_exact_series(
                    times_by_case,
                    means_by_case,
                )
                _plot_contributors(main_counts[row_index], startup_counts[row_index], physical_time, contributors, style=styles[label])
            main_support, startup_support = _plot_split_time_series(
                main_axes[row_index],
                startup_axes[row_index],
                physical_time,
                median,
                lower=lower,
                upper=upper,
                style=styles[label],
                contributor_count=contributors,
            )
            if main_support.size:
                main_times.append(main_support)
            if startup_support.size:
                startup_times.append(startup_support)
        _configure_split_time_axes(main_axes[row_index], startup_axes[row_index], main_times, bottom=row_index == rows - 1)
        if not main_times and not startup_times:
            main_axes[row_index].text(
                0.5,
                0.5,
                "Trajectory unavailable on stored physical times",
                ha="center",
                va="center",
                transform=main_axes[row_index].transAxes,
            )
        main_axes[row_index].grid(axis="y", alpha=0.20)
        startup_axes[row_index].grid(axis="y", alpha=0.20)

    figure_layout.apply_text_sizes(figure.axes)
    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=styles[label].color,
            linestyle=styles[label].linestyle,
            label=display_labels.wrapped_dataset_display_label(
                label if max_cases is None else f"{label} ({min(max_cases, len(frame))} cases)", width=LEGEND_WRAP_WIDTH
            ),
        )
        for label, frame in datasets.items()
    ]
    dataset_labels = [str(handle.get_label()) for handle in handles]
    handles.extend(
        (
            Line2D([], [], color="black", label="Mean / median"),
            Patch(facecolor="black", alpha=0.18, label="Range / q10-q90"),
            *_temporal_support_legend(aggregate=max_cases is not None, schedules=bool(schedule_fields)),
        )
    )
    scope = "single exact cases" if case_ids is not None else "available exact-time support"
    scope = "states over drying phase, reached cases only" if phase else scope
    figure_layout.set_figure_title(
        figure,
        window_grid,
        (
            f"Schedules and reference physical trajectories — {scope}"
            if schedule_fields
            else f"Reference physical trajectories — {scope}; full authored schedules unavailable"
        ),
        note=time_axis.phase_note(phase_notes),
    )
    figure_layout.add_legend(figure, legend_panels, handles, [str(handle.get_label()) for handle in handles], groups={"Datasets": dataset_labels})
    return figure


def _temporal_support_legend(*, aggregate: bool, schedules: bool) -> list[Line2D]:
    """Disclose distinct support semantics beside the dataset legend, one single-line entry each."""
    handles = []
    if aggregate:
        handles.append(Line2D([], [], color="black", drawstyle="steps-post", alpha=0.5, label="n(t): contributors (right axes)"))
    if schedules:
        handles.extend(Line2D([], [], linestyle="none", label=text) for text in ("Schedules: authored, running cases", "States: observed times only"))
    return handles


def plot_state_trajectory_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    channels: Sequence[str] | None = None,
) -> Figure:
    """Compare exact case schedules and trajectories without resampling, in the dataset ``styles`` of the viewer."""
    if tuple(case_ids) != tuple(datasets):
        message = "Transient trajectory datasets and case IDs must match in order."
        raise ValueError(message)
    return _trajectory_figure(
        datasets=datasets,
        channels=channels,
        case_ids=case_ids,
        max_cases=None,
        styles=styles,
    )


def plot_state_trajectory_summary(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    styles: Mapping[str, visual_semantics.SeriesStyle],
    channels: Sequence[str] | None = None,
    time_axis: str = "hours",
) -> Figure:
    """
    Aggregate observed state times and explicitly authored schedule functions, in the dataset ``styles`` of the viewer.

    ``time_axis="phase"`` pools states by ``analysis_ui_time.phase_cohort`` (schedules stay in hours), refusing datasets without completion evidence.
    """
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        message = "max_cases must be a positive integer."
        raise ValueError(message)
    return _trajectory_figure(
        datasets=datasets,
        channels=channels,
        case_ids=None,
        max_cases=max_cases,
        styles=styles,
        clock=time_axis,
    )


def _completion_time_headroom(maximum_hours: float) -> float:
    """Return a small bounded margin beyond one physical-time maximum."""
    if not np.isfinite(maximum_hours) or maximum_hours < 0.0:
        message = "Completion-time maximum must be finite and non-negative."
        raise ValueError(message)
    return min(6.0, max(1.0, 0.035 * max(maximum_hours, 1.0)))


def plot_completion_target_analysis(
    *,
    datasets: Mapping[str, pd.DataFrame],
    styles: Mapping[str, visual_semantics.SeriesStyle],
) -> Figure:
    """Plot distinct outcome-share, timing, and final-state questions in the dataset ``styles`` of the viewer."""
    analysis = transient.completion_target_analysis(datasets)
    colors = {label: style.color for label, style in styles.items()}
    figure = plt.figure(figsize=(_COMPLETION_FIGURE_WIDTH_INCHES, 9.4), layout=figure_layout.panel_engine())
    grid = figure.add_gridspec(2, 2, height_ratios=(0.95, 1.05))
    outcome_axis = figure.add_subplot(grid[0, :])
    time_plot_axis = figure.add_subplot(grid[1, 0])
    final_grid = grid[1, 1].subgridspec(2, 1)
    moisture_axis = figure.add_subplot(final_grid[0, 0])
    wet_fraction_axis = figure.add_subplot(final_grid[1, 0], sharex=moisture_axis)
    panels = np.array([[outcome_axis, time_plot_axis], [moisture_axis, wet_fraction_axis]], dtype=object)

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
            edgecolor=_HATCH_COLOR,
            hatch=styles[label].hatch,
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
    figure_layout.set_panel_title(outcome_axis, "Outcome share among eligible target-diagnostic cases")
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
        time_plot_axis.set_yticks(range(len(datasets)), list(datasets))  # each row names its dataset
        time_plot_axis.set_ylabel("Dataset")
        figure_layout.set_panel_title(time_plot_axis, "Target-attainment and censoring times")
        time_plot_axis.grid(axis="y", alpha=0.15)

        position = 0
        block_centres: dict[str, float] = {}
        for label in datasets:
            rows = eligible_cases.loc[eligible_cases["dataset"] == label]
            count = len(rows)
            if count == 0:
                continue
            positions = np.arange(position, position + count)
            block_centres[label] = float(positions.mean())
            moisture_axis.scatter(
                positions,
                field_labels.display_values(rows["final_bulk_moisture_wb"].to_numpy(dtype=float), "1", field="final_bulk_moisture_wb"),
                color=colors[label],
                marker="o",
                alpha=0.85,
            )
            moisture_axis.scatter(
                positions,
                field_labels.display_values(rows["target_moisture_wb"].to_numpy(dtype=float), "1", field="target_moisture_wb"),
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
        moisture_axis.set_ylabel(
            f"{moisture_display.display_name('w_gr', mathtext=True)} [{field_labels.display_unit('1', field='final_bulk_moisture_wb')}]"
        )
        figure_layout.set_panel_title(moisture_axis, "Final bulk moisture and target")
        moisture_axis.grid(alpha=0.25)
        wet_fraction_axis.set_xticks(list(block_centres.values()), list(block_centres), rotation=20, ha="right")  # the blocks name their datasets
        wet_fraction_axis.set_xlabel("Dataset (eligible cases in order)")
        wet_fraction_axis.set_ylabel("Wet fraction [-]")
        figure_layout.set_panel_title(wet_fraction_axis, "Final wet fraction and allowed limit")
        wet_fraction_axis.grid(alpha=0.25)

    dataset_handles = [
        Patch(
            facecolor=colors[label],
            edgecolor=_HATCH_COLOR,
            alpha=0.88,
            hatch=styles[label].hatch,
            label=display_labels.wrapped_dataset_display_label(label, width=LEGEND_WRAP_WIDTH),
        )
        for label in datasets
    ]
    semantic_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="Target time / final value"),
        Line2D([], [], color="black", marker="s", linestyle="None", label="Maximum-duration censoring"),
        Line2D([], [], color="black", marker="x", linestyle="None", label="Right-censoring time"),
        Line2D([], [], color="black", marker="_", linestyle="None", label="Target / allowed limit"),
    ]
    handles = [*dataset_handles, *semantic_handles]
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
    figure.supxlabel("\n".join(omission_lines), x=0.01, ha="left", linespacing=1.25, fontsize=8.8)
    figure_layout.apply_text_sizes(figure.axes)
    figure_layout.set_figure_title(figure, panels, "Completion and target attainment")
    figure_layout.add_legend(figure, panels, handles, [str(handle.get_label()) for handle in handles])
    return figure
