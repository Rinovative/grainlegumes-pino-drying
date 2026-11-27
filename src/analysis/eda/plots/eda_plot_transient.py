"""Plot transient completed-output exploratory diagnostics.

Responsibilities:
  - Render exact physical-time state and static-field snapshots, a one-signed velocity state on the velocity colormap of the steady
    plots; cells within round-off of 0 count as 0 for its sign and scale only, the map and its range note keep the stored values
  - Plot state trajectories in the documented drying windows of the ui time module (one below 72 h, else the first 24 h, the
    middle in days and the last 24 h), the aggregate state rows alternatively over the drying phase of
    ``analysis_ui_time.phase_cohort``, its cohort labels in the note; authored schedules keep their start-up and
    exact-support clock up to each exact stop, in the trajectory view and in the band over the snapshots
  - Count the cases each dataset draws, and the eligible ones a bound leaves out, for the titles and notes of the EDA summaries
  - Consolidate target outcomes, terminal timing, final moisture, and exact evidence

Design principles:
  - Every incompatible physical channel owns a separately labelled axis
  - Plot reductions delegate to the transient EDA semantic owner
  - Missing timing or physical-time evidence remains visibly unavailable
  - Figure text names a dataset by its plotted label (``eda_capabilities.plotted_labels``); frames, styles and case ids stay
    keyed by the dataset key

This module does NOT:
  - Change established steady-flow plot functions or presentation styling
  - Run model inference, calculate prediction error, or derive speedup
  - Add nearest-time fallbacks or average incompatible raw units
"""

from __future__ import annotations

import itertools
from numbers import Integral
from typing import TYPE_CHECKING, Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src import domain, generation
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
_HATCH_COLOR: Final = "white"  # Matplotlib draws a hatch in the edge color; white shows on every dataset color
_SINGLETON_MARKER: Final = "o"  # a series with one stored time in a window shows as a point, since a line needs two
_MAX_SIGNIFICANT_DIGITS: Final = 17  # significant digits that tell any two distinct float64 values apart
FOOTNOTE_INCHES: Final = (0.1, 0.08)  # lower-left corner of a figure's disclosure line
_TRAJECTORY_ROW_HEIGHT_INCHES: Final = 4.0 * 0.75
_TRAJECTORY_WIDTH_INCHES: Final = 9.0
_EARLY_OPERATION_END_HOURS: Final = time_axis.STARTUP_WINDOWS[1].stop_hours
_MINIMUM_DISPERSION_SUPPORT: Final = 2
_DATASET_HEADER_WRAP_WIDTH: Final = 38
_MAXIMUM_DATASET_HEADER_LINES: Final = 2
_OUTCOME_LABELS: Final = ("Reached target", "Right-censored")
_COMPLETION_FIGURE_WIDTH_INCHES: Final = 10.5
_COUNT_STRIP_HEIGHT_RATIO: Final = 0.3  # a contributor-count strip relative to the quantity row above it
_NEAR_UNIFORM_RELATIVE_SPAN: Final = 0.01  # a map whose observed span is below this share of its magnitude names its range
_WRAPPED_MAP_COLUMNS: Final = 2  # the fields of one dataset's retained maps fill this many columns
_NOTE_LINE_INCHES: Final = 0.2  # the rise of a column header over its note line (14 pt)
_VELOCITY_COMPONENTS: Final = next(group.fields for group in domain.tasks.registry.get_task("steady_flow").output_groups if group.id == "velocity")
_VELOCITY_COLORMAP: Final = "turbo"  # docs/06: u and v of the steady plots use turbo
_ROUNDOFF_RELATIVE: Final = 1e-9  # a velocity cell this small against the field's peak is round-off (v holds -1e-13 on the inlet row)

# One window of a plotted row: coordinates, centre, range (lower, upper) or none, and the contributing cases of an aggregate.
type _Series = tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]


def batch_styles(materials: Mapping[str, str | None], roles: Mapping[str, str | None] | None = None) -> dict[str, visual_semantics.SeriesStyle]:
    """
    Return the line color, style and bar hatch of every dataset, keyed by concise dataset label; the one mapping of the EDA plots.

    ``materials`` maps every dataset label to its one registered material or ``None``. A dataset with a registered material always
    owns that material's color; every other dataset takes, by its label's sorted rank, a color of ``_ORDER_COLORS``, which no
    material uses. Line style and hatch are those of the dataset's evaluation role in ``roles`` (``analysis_visual_semantics``); a
    dataset without a role is solid and unhatched, whatever its label. Viewers resolve this once over every loaded dataset, so a
    style never follows a selection.
    """
    materials = {label: material if material in visual_semantics.MATERIAL_COLORS else None for label, material in materials.items()}
    unmatched = sorted(label for label, material in materials.items() if material is None)
    styles: dict[str, visual_semantics.SeriesStyle] = {}
    for label, material in materials.items():
        role = None if roles is None else roles.get(label)
        color = _ORDER_COLORS[unmatched.index(label) % len(_ORDER_COLORS)] if material is None else visual_semantics.material_color(material)
        linestyle: visual_semantics.LineStyle = "-" if role is None else visual_semantics.ROLE_LINESTYLES[role]
        styles[label] = visual_semantics.SeriesStyle(color, linestyle, None, "" if role is None else visual_semantics.ROLE_HATCHES[role])
    return styles


def case_count_text(datasets: Mapping[str, pd.DataFrame], max_cases: int | None, *, plotted: Mapping[str, str]) -> tuple[str, str]:
    """
    Return a title's count of the cases each dataset draws (its first ``max_cases``, all without a bound) and a note of those left out.

    ``datasets`` are only those that contribute to the view; ``plotted`` maps each to the label the figure writes. The count
    reads «N cases» for one dataset, «N cases per Dataset» when every dataset draws N, else «label: N cases» for each. The note
    names, for every dataset the bound leaves cases of, the drawn of its eligible cases (the frame's ``available_case_count``,
    known before the load: the package or material membership, the successful cases of a source batch); else it is empty.
    """
    drawn = {label: len(frame) if max_cases is None else min(max_cases, len(frame)) for label, frame in datasets.items()}
    counts = set(drawn.values())
    if len(counts) > 1:
        count = ", ".join(f"{plotted[label]}: {number} cases" for label, number in drawn.items())
    else:
        count = f"{next(iter(counts), 0)} cases{' per Dataset' if len(drawn) > 1 else ''}"
    eligible = {label: frame.attrs.get("available_case_count") for label, frame in datasets.items()}
    bounded = [f"{plotted[label]}: {drawn[label]} of {total}" for label, total in eligible.items() if isinstance(total, int) and total > drawn[label]]
    return count, f"First cases drawn of the eligible ones — {'; '.join(bounded)}." if bounded else ""


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
                    **_singleton_point(main_times.size),
                )
                main_support_by_row[field_index].append(main_times)
            if startup_times.size:
                startup_axes[field_index].plot(
                    startup_times,
                    startup_values,
                    color=colors[label],
                    linestyle=linestyles[label],
                    **_singleton_point(startup_times.size),
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
                clip_on=False,  # whole also on a window's frame, where 1 h, the main window's start, puts it
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


def _without_roundoff(values: np.ndarray) -> np.ndarray:
    """Return a copy of ``values`` for the scale only, each cell within solver round-off of 0 (against the peak) at 0: noise gives no second sign."""
    peak = float(np.max(np.abs(values), initial=0.0))
    return np.where(np.abs(values) <= _ROUNDOFF_RELATIVE * peak, 0.0, values)


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
    case_number: int | None = None,
) -> Figure:
    """Render one union-capability map grid with optional exact split schedules, whose lines take their ``styles``; ``case_number`` ends the title."""
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
    plotted = capabilities.plotted_labels(datasets)
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
        # Round-off cells of a velocity count as 0 for its scale (sign and bands) only; the map and its range note show stored values.
        scaled = {label: _without_roundoff(values) if field in _VELOCITY_COMPONENTS else values for label, values in values_by_label.items()}
        shared = spatial_field_scale(field, unit, representation, *scaled.values(), velocity_state=True) if lock_scale else None
        for label, values in values_by_label.items():
            scale = shared if shared is not None else spatial_field_scale(field, unit, representation, scaled[label], velocity_state=True)
            cells[field, label] = (values, scale)
    extents = [display_grid.physical_extent(*coordinates[label]) for label in coordinates]
    extent = (min(e[0] for e in extents), max(e[1] for e in extents), min(e[2] for e in extents), max(e[3] for e in extents))
    schedule_fields = _schedule_fields() if include_schedules else ()
    # One dataset's retained maps fill two columns, a field per cell, instead of one tall column.
    wrapped = len(datasets) == 1 and not include_schedules and len(resolution.fields) > 1
    if wrapped:
        positions = {(field, label): divmod(index, _WRAPPED_MAP_COLUMNS) for index, field in enumerate(resolution.fields) for label in datasets}
        shape = (-(-len(resolution.fields) // _WRAPPED_MAP_COLUMNS), _WRAPPED_MAP_COLUMNS)
    else:
        positions = {
            (field, label): (row, column) for (row, field), (column, label) in itertools.product(enumerate(resolution.fields), enumerate(datasets))
        }
        shape = (len(resolution.fields), len(datasets))
    notes = {  # the range of a near-uniform map, on a note line above it
        positions[key]: note
        for key, (values, _scale) in cells.items()
        if (note := _near_uniform_range_note(values, capabilities.field_display_unit(datasets[key[1]], key[0])))
    }
    # A note under a first-row header raises the header by a line, so a schedule band above stands that line higher.
    clearance_in = _NOTE_LINE_INCHES if include_schedules and any(row == 0 for row, _column in notes) else 0.0
    geometry = layout.plan_geometry(
        *shape,
        (extent[1] - extent[0], extent[3] - extent[2]),
        row_labels=not wrapped,
        width_cap=None,
        above_in=(1 + len(schedule_fields)) * _LINE_ROW_HEIGHT_INCHES + clearance_in if include_schedules else 0.0,
        right_in=_LEGEND_WIDTH_INCHES if include_schedules else 0.0,
    )
    figure, axes, slots = layout.map_grid(geometry, extent=extent)
    try:
        if include_schedules:
            _draw_line_band(
                figure, geometry, datasets, case_ids, schedule_fields, master_time, case_times, styles, plotted=plotted, clearance_in=clearance_in
            )
        for (field, label), cell in positions.items():
            if (field, label) not in cells:
                layout.empty_cell(axes[cell], f"{field} unavailable")
                continue
            values, scale = cells[field, label]
            indices = display_grid.display_indices(values.shape, display_spatial_stride)
            x, y = coordinates[label]
            image = axes[cell].pcolormesh(
                display_grid.display_sample(x, indices),
                display_grid.display_sample(y, indices),
                display_grid.display_sample(values, indices),
                shading="auto",
                **scale.pcolormesh_kwargs(),
            )
            layout.add_field_map_colorbar(figure, image, slots[cell], scale, label=f"[{capabilities.field_display_unit(datasets[label], field)}]")
        field_names = [
            capabilities.field_display_name(next(iter(capabilities.compatible_frames(datasets, resolution, field).values())), field, mathtext=True)
            for field in resolution.fields
        ]
        if wrapped:
            _label_wrapped_cells(axes, field_names, notes)
        else:
            for field_index, field in enumerate(resolution.fields):  # an empty first cell hands its y labels to the row's first map
                first = next((column for column, label in enumerate(datasets) if (field, label) in cells), 0)
                if first:
                    axes[field_index, first].set_ylabel("y [m]", fontsize=figure_layout.LABEL_PT)
                    axes[field_index, first].tick_params(labelleft=True)
            names = [_section_three_dataset_header(plotted[label]) if include_schedules else plotted[label] for label in datasets]
            headers = [
                display_labels.wrapped_dataset_display_label(
                    f"{name}{_display_case_time(case_times.get(label))}", width=_DATASET_HEADER_WRAP_WIDTH, max_lines=_MAXIMUM_DATASET_HEADER_LINES
                )
                for name, label in zip(names, datasets, strict=True)
            ]
            _label_grid_cells(axes, headers, field_names, notes)
        layout.set_map_title(figure, title if case_number is None else f"{title} — Case {case_number}")
        details = _single_case_details(plotted, case_ids, case_times, wrapped=wrapped)
        if footnote := "; ".join(part for part in (details, capabilities.availability_note(resolution, plotted=plotted)) if part):
            figure.text(FOOTNOTE_INCHES[0] / figure.get_figwidth(), FOOTNOTE_INCHES[1] / figure.get_figheight(), footnote, va="bottom", fontsize=8)
    except Exception:
        plt.close(figure)
        raise
    return figure


def range_text(lower: float, upper: float, *, digits: int) -> str:
    """Write ``lower to upper`` with ``digits`` significant digits, or as many more as tell unequal endpoints apart."""
    while lower != upper and digits < _MAX_SIGNIFICANT_DIGITS and f"{lower:.{digits}g}" == f"{upper:.{digits}g}":
        digits += 1
    return f"{lower:.{digits}g} to {upper:.{digits}g}"


def _near_uniform_range_note(values: np.ndarray, unit: str) -> str:
    """Return the observed full-support range of a near-uniform map in its display ``unit``, or ``""`` for any other map."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    lower, upper = float(finite.min()), float(finite.max())
    reach = max(abs(lower), abs(upper))
    if reach == 0.0 or upper - lower > _NEAR_UNIFORM_RELATIVE_SPAN * reach:
        return ""
    suffix = "" if unit == field_labels.display_unit("1") else f" {unit}"
    return f"constant {lower:.6g}{suffix}" if lower == upper else f"range {range_text(lower, upper, digits=6)}{suffix}"


def _label_grid_cells(axes: np.ndarray, headers: Sequence[str], field_names: Sequence[str], notes: Mapping[tuple[int, int], str]) -> None:
    """Head the dataset columns, label the field rows, and write every ``notes`` line above its map, in the first row under the header."""
    layout.set_column_headers(axes, headers, [notes.get((0, column), "") for column in range(axes.shape[1])])
    layout.set_row_labels(axes, field_names)
    for (row, column), note in notes.items():
        if row:  # a lower row's note line stands in the gap above its map
            layout.add_map_note(axes[row, column], note)


def _label_wrapped_cells(axes: np.ndarray, field_names: Sequence[str], notes: Mapping[tuple[int, int], str]) -> None:
    """Head every wrapped cell with its field and its ``notes`` line, blank the cells after the last, and give a map above a blank one x labels."""
    columns = axes.shape[1]
    for row in range(axes.shape[0]):
        row_notes = [notes.get((row, column), "") for column in range(columns)]
        layout.set_column_headers(axes[row : row + 1], field_names[row * columns : (row + 1) * columns], row_notes)
    for index in range(len(field_names), axes.size):
        row, column = divmod(index, columns)
        axes[row, column].set_axis_off()
        if row:
            axes[row - 1, column].tick_params(labelbottom=True)
            axes[row - 1, column].set_xlabel("x [m]", fontsize=figure_layout.LABEL_PT)


def _single_case_details(
    plotted: Mapping[str, str],
    case_ids: Mapping[str, str],
    case_times: Mapping[str, time_axis.ResolvedPhysicalTime],
    *,
    wrapped: bool,
) -> str:
    """Return the exact case of a one-dataset figure, plus its ``plotted`` label and time when no column header names them; ``""`` for several."""
    if len(plotted) != 1:
        return ""
    label = next(iter(plotted))
    return f"{plotted[label]}{_display_case_time(case_times.get(label))} · case {case_ids[label]}" if wrapped else f"Case {case_ids[label]}"


def _draw_line_band(
    figure: Figure,
    geometry: layout.MapGeometry,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    schedule_fields: Sequence[generation.contracts.FieldContract],
    master_time: float | None,
    case_times: Mapping[str, time_axis.ResolvedPhysicalTime],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    *,
    plotted: Mapping[str, str],
    clearance_in: float,
) -> None:
    """Draw the inlet pressure and the split schedules above the maps, its lowest ``clearance_in`` left free, the ``plotted`` legend right of it."""
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
        bottom=bottom + clearance_in / geometry.figure_height,
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
        Line2D([], [], color=colors[label], label=display_labels.wrapped_dataset_display_label(plotted[label], width=LEGEND_WRAP_WIDTH))
        for label in datasets
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


def spatial_field_scale(
    field: str, unit: str, representation: str, *values: np.ndarray, velocity_state: bool = False
) -> field_maps.FieldMapDefinition:
    """
    Build one data-derived scale (round bands; geometric for permeability) from full arrays; masked unphysical water cells stay off it.

    Only a quantity signed by meaning (``Kxy``, a ``delta_`` field, a centred visual semantic) is diverging about 0; any other
    one, also with negative values (log10 permeability, a pressure undershoot), is sequential over its own range. With
    ``velocity_state`` (a retained map), a velocity component of one sign is a sequential state on the velocity colormap of
    the steady plots; one of both signs keeps the diverging scale that shows its direction.
    """
    combined = np.concatenate(tuple(np.ravel(value) for value in values))
    if moisture_display.is_moisture_field(field):
        combined = combined[np.isfinite(combined)]
    semantic = field_maps.FieldMapSemantic.LINEAR_SEQUENTIAL if np.any(combined < 0.0) else field_maps.FieldMapSemantic.LINEAR_POSITIVE
    colormap = None
    if field in {"Kxx", "Kyy", "K_min", "K_max"} and unit == "m^2":
        semantic = field_maps.FieldMapSemantic.LOG_POSITIVE
    elif velocity_state and field in _VELOCITY_COMPONENTS and (np.all(combined >= 0.0) or np.all(combined <= 0.0)):
        colormap = _VELOCITY_COLORMAP
    elif field == "Kxy" or field.startswith("delta_") or visual_semantics.field_visual_semantics(field).centered:
        semantic = field_maps.FieldMapSemantic.LINEAR_SIGNED
    return field_maps.build_field_map(combined, semantic=semantic, field=field, unit=unit, display_transform=representation, colormap_name=colormap)


def plot_spatial_field_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    fields: Sequence[str] | None = None,
    lock_scale: bool = False,
    physical_time_hours: float | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
    case_number: int | None = None,
) -> Figure:
    """Compare compatible spatial fields across steady and transient datasets, the title naming the shared ``case_number``."""
    return _spatial_map_figure(
        case_number=case_number,
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
    case_number: int | None = None,
) -> Figure:
    """Compare exact/latest-prior transient states with inlet schedules in the viewer's dataset ``styles``; ``case_number`` ends the title."""
    return _spatial_map_figure(
        case_number=case_number,
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


def _window_points(window: time_axis.TimeWindow, times: np.ndarray, end: float | None, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the window coordinates of the stored ``times`` inside ``window`` (of a case ending at ``end``) and the ``values`` there."""
    inside = window.select(times, end_hours=end)
    return (window.hours(times[inside], end_hours=end), *(value[inside] for value in values))


def _schedule_points(window: time_axis.TimeWindow, item: transient.SupportedScheduleSeries) -> np.ndarray:
    """Return one schedule's nodes inside the clock ``window`` plus the window edges inside its support, drawn on both sides of an edge."""
    support = (float(item.physical_time_hours[0]), item.final_time_hours)
    edges = np.asarray([edge for edge in (window.start_hours, window.stop_hours) if support[0] <= edge <= support[1]], dtype=np.float64)
    return np.unique(np.concatenate([_window_points(window, item.physical_time_hours, None)[0], window.hours(edges)]))


def _aggregate_supported_schedule(
    series: Sequence[transient.SupportedScheduleSeries], window: time_axis.TimeWindow
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate authored piecewise-linear schedules within their simulated support at every node inside one clock window.

    The coordinates are the cases' nodes inside ``window``, their exact stops included; at each, every case whose support holds
    that time contributes its value, so a stopped case drops out without a forward fill. Returns the coordinates, q10, median,
    q90 and the contributing case count.
    """
    if not series:
        message = "Aggregate schedule plotting requires at least one selected case."
        raise ValueError(message)
    coordinates = np.unique(np.concatenate([_schedule_points(window, item) for item in series]))
    q10, median, q90 = (np.empty_like(coordinates) for _quantile in range(3))
    contributor_count = np.empty(coordinates.shape, dtype=np.int64)
    for index, time_value in enumerate(coordinates):
        contributors = [item.value_at(float(time_value)) for item in series if item.physical_time_hours[0] <= time_value <= item.final_time_hours]
        if not contributors:
            message = "Aggregate schedule timeline contains unsupported physical time."
            raise RuntimeError(message)
        q10[index], median[index], q90[index] = np.quantile(np.asarray(contributors, dtype=np.float64), (0.1, 0.5, 0.9))
        contributor_count[index] = len(contributors)
    return coordinates, q10, median, q90, contributor_count


def _schedule_window(
    series: Sequence[transient.SupportedScheduleSeries], field: generation.contracts.FieldContract, window: time_axis.TimeWindow, *, aggregate: bool
) -> _Series | None:
    """Return one clock window of authored schedules at nodes and window edges: the exact case, else the median, q10-q90 and count."""
    if aggregate:
        times, q10, median, q90, counts = _aggregate_supported_schedule(series, window)
        lower, centre, upper = (field_labels.display_values(values, field.unit, field=field.name) for values in (q10, median, q90))
        return (times, centre, lower, upper, counts) if times.size else None
    times = _schedule_points(window, series[0])
    values = np.asarray([series[0].value_at(float(time)) for time in times], dtype=np.float64)
    return (times, field_labels.display_values(values, field.unit, field=field.name), None, None, None) if times.size else None


def _state_window(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    completion: Sequence[time_axis.Completion],
    ends: np.ndarray,
    window: time_axis.TimeWindow,
    *,
    aggregate: bool,
) -> _Series | None:
    """
    Return one window of a state row: an exact case's spatial mean and min-max, else the median and q10-q90 of the cases' means.

    In hours the cases meet on exactly equal window coordinates (``transient.aggregate_exact_series``); the drying-phase window
    pools the reached cases (``transient.aggregate_phase_series``). ``None`` when no stored time lies in the window.
    """
    if window.unit == "1":
        times_by_case, means_by_case = [item[0] for item in trajectories], [item[1] for item in trajectories]
        q10, median, q90, cohort = transient.aggregate_phase_series(times_by_case, means_by_case, completion)
        return time_axis.PHASE_GRID, median, q10, q90, np.full(time_axis.PHASE_GRID.shape, len(cohort.cases))
    shown = [_window_points(window, times, float(end), *values) for (times, *values), end in zip(trajectories, ends, strict=True)]
    if not aggregate:
        ((times, means, lower, upper),) = shown
        return (times, means, lower, upper, None) if times.size else None
    present = [(times, means) for times, means, *_extrema in shown if times.size]
    if not present:
        return None
    times, q10, median, q90, counts = transient.aggregate_exact_series([item[0] for item in present], [item[1] for item in present])
    return times, median, q10, q90, counts


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


def _singleton_point(count: int) -> dict[str, Any]:
    """
    Return the plot keywords of a series with ``count`` stored times in one window: one time shows as a point, more add nothing.

    A line needs two times, and a lone time often sits on the window's frame (an open window ends at its last shown time, the
    start-up window at 0 and 1 hour), so the point is drawn unclipped to show its whole disc without padding the shared limits.
    """
    return {"marker": _SINGLETON_MARKER, "clip_on": False} if count == 1 else {}


def _draw_range(
    axis: Axes, times: np.ndarray, lower: np.ndarray, upper: np.ndarray, supported: np.ndarray, style: visual_semantics.SeriesStyle
) -> None:
    """
    Draw one window's range at its ``supported`` times: shaded between adjacent supported times, else a vertical interval.

    A band of zero width shows nothing, so a supported time without a supported neighbour (a window's only time among them)
    shows its range as an unclipped interval in the series' color and line style.
    """
    joined = supported & (np.r_[False, supported[:-1]] | np.r_[supported[1:], False])
    if joined.any():
        axis.fill_between(times, lower, upper, color=style.color, alpha=0.18, where=joined)
    if (isolated := supported & ~joined).any():
        axis.vlines(
            times[isolated], lower[isolated], upper[isolated], colors=style.color, linestyles=style.linestyle, alpha=0.4, linewidth=2.0, clip_on=False
        )


def _draw_counts(axis: Axes, times: np.ndarray, counts: np.ndarray, style: visual_semantics.SeriesStyle) -> None:
    """Disclose every stored-time count as a step line in the series' color and line style, without implying interpolated support."""
    axis.plot(times, counts, color=style.color, linestyle=style.linestyle, drawstyle="steps-post", alpha=0.5, **_singleton_point(times.size))


def _draw_series(axis: Axes, series: _Series, style: visual_semantics.SeriesStyle, strip: Axes | None = None) -> None:
    """
    Draw one window of a series in ``style``: its centre line and its range, and its contributing cases on the count ``strip``.

    The range shows where at least two cases contribute (at every time of a series without counts), as ``_draw_range`` says.
    """
    times, centre, lower, upper, counts = series
    axis.plot(times, centre, color=style.color, linestyle=style.linestyle, **_singleton_point(times.size))
    if lower is not None and upper is not None:
        _draw_range(axis, times, lower, upper, np.ones(times.shape, dtype=bool) if counts is None else counts >= _MINIMUM_DISPERSION_SUPPORT, style)
    if strip is not None and counts is not None:
        _draw_counts(strip, times, counts, style)


def _configure_split_time_axes(main_axis: Axes, startup_axis: Axes, main_times: Sequence[np.ndarray], *, bottom: bool) -> None:
    """Style the main axis to the end of its shown support and the startup axis; ``bottom`` rows only carry time numbers and the label."""
    main_window, startup_window = time_axis.STARTUP_WINDOWS
    shown_end = max((float(times.max()) for times in main_times), default=None)
    time_axis.style_window_axis(main_axis, main_window, shown_end_hours=shown_end, bottom=bottom)
    time_axis.style_window_axis(startup_axis, startup_window, bottom=bottom)


def _style_count_strips(strips: np.ndarray, maximum: int, label: str) -> None:
    """Give every thin count strip ``[row, window]`` below a quantity row the integer range 0 to ``maximum`` and its ``label``."""
    for axis in strips[:, 0]:  # the windows of a row share their y axis
        axis.set_ylim(0.0, maximum + 0.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        axis.set_ylabel(label, rotation=0, ha="right", va="center")
        axis.grid(axis="y", alpha=0.20)


def _style_window_block(
    data: np.ndarray, strips: np.ndarray | None, windows: Sequence[time_axis.TimeWindow], maximum: int, shown_end: float | None
) -> None:
    """
    Title the ``windows`` over the first row of one block, fit every axis to its window, the time numbers under the block only.

    An open window (the schedules' main one) ends at ``shown_end``, the last time the block draws.
    """
    last = data.shape[0] - 1
    for row, axes in enumerate(data):
        for column, axis in enumerate(axes):
            time_axis.style_window_axis(axis, windows[column], shown_end_hours=shown_end, bottom=row == last and strips is None)
            axis.grid(axis="y", alpha=0.20)
            if strips is not None:
                time_axis.style_window_axis(strips[row, column], windows[column], shown_end_hours=shown_end, bottom=row == last)
    for axis, window in zip(data[0], windows, strict=True):
        figure_layout.set_panel_title(axis, window.title)
    if strips is not None:
        _style_count_strips(strips, maximum, "n(τ)" if windows[0].unit == "1" else "n(t)")


def _trajectory_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    channels: Sequence[str] | None,
    case_ids: Mapping[str, str] | None,
    max_cases: int | None,
    styles: Mapping[str, visual_semantics.SeriesStyle],
    clock: str = "hours",
    case_number: int | None = None,
) -> Figure:
    """
    Render exact-case or aggregate schedules and state trajectories in ``styles``; exact cases are titled by ``case_number``.

    The authored schedules keep their start-up and exact-support clock (``analysis_ui_time.STARTUP_WINDOWS``) up to each
    exact stop, in a block above the states. The state rows take the drying windows of the longest case
    (``analysis_ui_time.drying_windows``): one below 72 h, else the first 24 h in hours, the middle in days and the last
    24 h in hours, counted before each case's end (its last stored state) when the cases' ends differ at all; over the
    drying phase they take its one window.
    """
    if not datasets or (case_ids is None) == (max_cases is None):
        message = "Trajectory plotting requires exactly one case map or aggregate bound."
        raise ValueError(message)
    phase = time_axis.is_phase(clock)
    if phase and (lacking := [label for label, frame in datasets.items() if not transient.has_completion_evidence(frame)]):
        message = f"The drying phase pools reached cases, but datasets {lacking} lack completion evidence (published Datasets have none)."
        raise ValueError(message)
    resolution = capabilities.resolve_fields(datasets, view="state_trajectory", requested=channels)
    plotted = capabilities.plotted_labels(datasets)
    schedules = {field.name: field for field in _schedule_fields()} if _has_complete_schedules(datasets) else {}
    aggregate = case_ids is None
    cases = {label: (case_ids[label],) if case_ids is not None else tuple(map(str, frame.index[:max_cases])) for label, frame in datasets.items()}
    rows = {label: [capabilities.case_row(datasets[label], case) for case in ids] for label, ids in cases.items()}
    completion: dict[str, list[time_axis.Completion]] = {
        label: [transient.case_completion(row) for row in case_rows] for label, case_rows in rows.items()
    }
    ends = {  # a case's end is its last stored state, at most one interval before its exact stop, as the evaluation's
        label: np.asarray([transient.available_physical_times(datasets[label], case, include_exact_stop=False)[-1] for case in ids])
        for label, ids in cases.items()
    }
    longest, shortest = max(float(end.max()) for end in ends.values()), min(float(end.min()) for end in ends.values())
    states = (time_axis.PHASE_WINDOW,) if phase else time_axis.drying_windows(longest, before_end=longest != shortest)
    blocks = [(names, windows) for names, windows in ((tuple(schedules), time_axis.STARTUP_WINDOWS), (resolution.fields, states)) if names]
    heights = [1.0, _COUNT_STRIP_HEIGHT_RATIO] if aggregate else [1.0]  # an aggregate row keeps its case counts in a thin strip below it
    figure = plt.figure(
        figsize=(_TRAJECTORY_WIDTH_INCHES, _TRAJECTORY_ROW_HEIGHT_INCHES * sum(heights) * sum(len(names) for names, _windows in blocks)),
        layout=figure_layout.panel_engine(),
    )
    outer = figure.add_gridspec(len(blocks), 1, height_ratios=[len(names) for names, _windows in blocks])
    notes: dict[str, str] = {}
    row_labels: list[tuple[Axes, str]] = []
    for block, (names, windows) in enumerate(blocks):
        grid = layout.window_axes(figure, outer[block, 0], windows, heights * len(names))
        data, strips = (grid[0::2], grid[1::2]) if aggregate else (grid, None)
        shown_ends: list[float] = []
        for row, name in enumerate(names):
            shown: dict[str, list[_Series | None]] = {}
            if name in schedules:
                row_label = field_labels.field_label_with_unit(name, schedules[name].unit, mathtext=True)
                for label, frame in datasets.items():
                    series = [transient.supported_schedule_series(frame, case, name) for case in cases[label]]
                    shown[label] = [_schedule_window(series, schedules[name], window, aggregate=aggregate) for window in windows]
            else:
                compatible = capabilities.compatible_frames(datasets, resolution, name)
                row_label = capabilities.field_quantity_label(next(iter(compatible.values())), name, mathtext=True)
                for label, frame in compatible.items():
                    trajectories = [capabilities.field_display_trajectory(frame, case_row, name) for case_row in rows[label]]
                    shown[label] = [_state_window(trajectories, completion[label], ends[label], window, aggregate=aggregate) for window in windows]
                    if phase:  # the note's lines name their dataset by its plotted label
                        notes[plotted[label]] = time_axis.phase_cohort([item[0] for item in trajectories], completion[label]).label()
            row_labels.append((data[row, 0], row_label))
            for label, items in shown.items():
                for column, item in enumerate(items):
                    if item is not None:
                        _draw_series(data[row, column], item, styles[label], None if strips is None else strips[row, column])
                        shown_ends.append(float(item[0][-1]))
        _style_window_block(data, strips, windows, max(map(len, cases.values())), max(shown_ends, default=None))
    figure_layout.apply_text_sizes(figure.axes)
    for axis, text in row_labels:  # the y label of the first column: the layout puts it left of that column's tick labels
        axis.set_ylabel(figure_layout.bold_label(text), fontsize=figure_layout.PANEL_TITLE_PT, fontweight="bold")
        axis.yaxis.label.set_gid("channel-row-label")
    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=styles[label].color,
            linestyle=styles[label].linestyle,
            label=display_labels.wrapped_dataset_display_label(
                f"{plotted[label]} ({len(ids)} cases)" if aggregate else plotted[label], width=LEGEND_WRAP_WIDTH
            ),
        )
        for label, ids in cases.items()
    ]
    dataset_handles = list(handles)  # the Dataset group owns these entries by identity
    centre, spread = ("Median over cases", "q10-q90 over cases") if aggregate else ("State: spatial mean", "State: spatial min-max")
    counts = ", ".join(dict.fromkeys("n(τ)" if windows[0].unit == "1" else "n(t)" for _names, windows in blocks)) if aggregate else ""
    handles += [Line2D([], [], color="black", label=centre), Patch(facecolor="black", alpha=0.18, label=spread)]
    handles += _temporal_support_legend(counts=counts, schedules=bool(schedules))
    scope = ("single exact cases" if case_number is None else f"Case {case_number}") if case_ids is not None else "available exact-time support"
    scope = "states over drying phase, reached cases only" if phase else scope
    panels = np.asarray(figure.axes, dtype=object)[np.newaxis, :]  # the blocks' grids differ in their window count
    figure_layout.set_figure_title(
        figure,
        panels,
        (
            f"Schedules and reference physical trajectories — {scope}"
            if schedules
            else f"Reference physical trajectories — {scope}; full authored schedules unavailable"
        ),
        note=time_axis.phase_note(notes),
    )
    figure_layout.add_legend(figure, panels, handles, [str(handle.get_label()) for handle in handles], groups={"Dataset": dataset_handles})
    return figure


def _temporal_support_legend(*, counts: str, schedules: bool) -> list[Line2D]:
    """Disclose distinct support semantics beside the dataset legend, one single-line entry each; ``counts`` names the strips' symbols."""
    handles = []
    if counts:
        handles.append(Line2D([], [], color="black", drawstyle="steps-post", alpha=0.5, label=f"{counts}: contributors (strip below each row)"))
    if schedules:
        handles.extend(Line2D([], [], linestyle="none", label=text) for text in ("Schedules: authored, running cases", "States: observed times only"))
    return handles


def plot_state_trajectory_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    channels: Sequence[str] | None = None,
    case_number: int | None = None,
) -> Figure:
    """Compare exact case schedules and trajectories without resampling, in the dataset ``styles`` of the viewer, titled by ``case_number``."""
    if tuple(case_ids) != tuple(datasets):
        message = "Transient trajectory datasets and case IDs must match in order."
        raise ValueError(message)
    return _trajectory_figure(
        datasets=datasets,
        channels=channels,
        case_ids=case_ids,
        max_cases=None,
        styles=styles,
        case_number=case_number,
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
    plotted = capabilities.plotted_labels(datasets)
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
            "No eligible cases: " + ", ".join(plotted[label] for label in zero_eligible),
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
        time_plot_axis.set_yticks(range(len(datasets)), [plotted[label] for label in datasets])  # each row names its dataset
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
        blocks = [plotted[label] for label in block_centres]  # the blocks name their datasets
        wet_fraction_axis.set_xticks(list(block_centres.values()), blocks, rotation=20, ha="right")
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
            label=display_labels.wrapped_dataset_display_label(plotted[label], width=LEGEND_WRAP_WIDTH),
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
        omission_lines.append(f"{plotted[row['dataset']]} — eligible cases: {eligible_count}; non-eligible omitted: {omitted_count}{suffix}.")
    figure.supxlabel("\n".join(omission_lines), x=0.01, ha="left", linespacing=1.25, fontsize=8.8)
    figure_layout.apply_text_sizes(figure.axes)
    figure_layout.set_figure_title(figure, panels, "Completion and target attainment")
    figure_layout.add_legend(figure, panels, handles, [str(handle.get_label()) for handle in handles])
    return figure
