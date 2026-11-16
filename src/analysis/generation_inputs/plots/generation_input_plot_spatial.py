"""Render coordinated A/B/B-minus-A blocks for generation-input fields.

Responsibilities:
  - Group maintained channels as rows of the shared map grid under three columns headed once: Case A, Case B, and
    Case B minus Case A, each row named once at its left and the figure title carrying the scale state
  - Use independent A/B physical scales unless the local scale lock is enabled
  - Give every map its own unit-bearing slot colour bar as tall as the map
  - Compose a lower band under the maps with one distribution per field, pressure lines, embedded summaries and
    optional side plots, whose legends stand inside their axes

Design principles:
  - B-minus-A remains physical, symmetric, and exactly centred at zero; maps, differences and distributions use the
    shared display scales (relative humidity in %, its differences in pp) under the active label style
  - The lower band's panels are as wide as the maps together and never take part in the map grid
  - Summary-table numbers are written compactly and in columns wide enough that they never overlap
  - Incompatible grids are reported without interpolation or resampling

This module does NOT:
  - Render single-case alternatives, derive fields, or load input files
  - Pool unrelated fields or display dataset-mean maps
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Final

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.ui import analysis_ui_plot_layout as shared_layout
from src.analysis.ui import tables

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

_MAP_COLUMNS: Final = 3  # Case A, Case B, Case B minus Case A
_ROW_LABEL_WIDTH: Final = 16  # characters per line of a row label: a long field name stays within its map's height
_MATH_SPAN: Final = re.compile(r"\$[^$]*\$")  # a mathtext span, which a row label never splits
_MINIMUM_SIDE_PLOT_ROWS: Final = 2
_SUMMARY_VALUE_COLUMN_COUNT: Final = 4
# The summary column of the lower band: its longest row label (2.0 in) and its widest number (-1.234e-10, 0.61 in) each fit
# their cell inside the cell padding at the table font.
_SUMMARY_WIDTH_IN: Final = 6.0
_SUMMARY_ROW_HEIGHT_IN: Final = 0.34  # every row of the summary table, the two-line header included; the table sits at the top of its column
_LOWER_GAP_IN: Final = 0.5  # the lower band's plots to its summary column
_SUMMARY_LABEL_COLUMN_FRACTION: Final = 0.45
_SUMMARY_VALUE_COLUMN_FRACTION: Final = (1.0 - _SUMMARY_LABEL_COLUMN_FRACTION) / _SUMMARY_VALUE_COLUMN_COUNT
_SUMMARY_COLUMN_WIDTHS: Final = (
    _SUMMARY_LABEL_COLUMN_FRACTION,
    *(_SUMMARY_VALUE_COLUMN_FRACTION,) * _SUMMARY_VALUE_COLUMN_COUNT,
)


def _validate_quantities(
    record: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
) -> tuple[str, ...]:
    """Return one non-empty tuple of profile-available display fields."""
    selected = tuple(quantities)
    if not selected:
        msg = "A spatial comparison block requires at least one field."
        raise ValueError(msg)
    unavailable = tuple(quantity for quantity in selected if quantity not in diagnostics.display_field_names(record))
    if unavailable:
        msg = f"Spatial fields are unavailable for the selected profile: {unavailable}."
        raise ValueError(msg)
    return selected


def _histogram_bounds(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    """Return finite shared histogram bounds with a constant-field margin."""
    combined = np.concatenate((first.ravel(), second.ravel()))
    if not np.isfinite(combined).all():
        msg = "Spatial distributions require finite physical values."
        raise ValueError(msg)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if lower == upper:
        padding = max(abs(lower), 1.0) * 1.0e-6
        lower -= padding
        upper += padding
    return lower, upper


def _case_plot_label(
    record: diagnostics.GenerationInputDiagnostics,
    dataset_label: str,
) -> str:
    """Return one actual case number with its A/B dataset role."""
    return f"Case {record.case.case_index} ({dataset_label})"


def _draw_distribution(
    axis: Axes,
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    quantity: str,
    *,
    same_dataset: bool,
) -> None:
    """Draw A/B field distributions and unambiguous dataset-mean markers on the display scale."""
    first_values, second_values = layout.display_field(first, quantity), layout.display_field(second, quantity)
    exponent = layout.histogram_exponent(first_values, second_values)
    scale = 10.0**exponent
    first_values, second_values = first_values / scale, second_values / scale
    lower, upper = _histogram_bounds(first_values, second_values)
    bins = tuple(float(value) for value in np.linspace(lower, upper, 31))
    histograms.plot_histogram(
        axis,
        first_values.ravel(),
        bins=bins,
        alpha=0.44,
        color=layout.DATASET_A_COLOR,
        label=_case_plot_label(first, "A"),
    )
    histograms.plot_histogram(
        axis,
        second_values.ravel(),
        bins=bins,
        alpha=0.44,
        color=layout.DATASET_B_COLOR,
        label=_case_plot_label(second, "B"),
    )
    first_mean, second_mean = (
        diagnostics.display_value(quantity, dataset.field_summary_means[(quantity, "mean")], diagnostics.FIELD_UNITS[quantity]) / scale
        for dataset in (mean_a, mean_b)
    )
    if same_dataset:
        axis.axvline(
            first_mean,
            color=layout.DATASET_MEAN_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"Dataset mean, n = {mean_a.case_count}",
        )
    else:
        axis.axvline(
            first_mean,
            color=layout.DATASET_A_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"Mean A, n = {mean_a.case_count}",
        )
        axis.axvline(
            second_mean,
            color=layout.DATASET_B_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"Mean B, n = {mean_b.case_count}",
        )
    axis.set_xlabel(
        layout.quantity_axis_label(quantity, exponent),
        fontsize=layout.FIGURE_LAYOUT.label_size,
    )
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)  # the power of ten stands in the unit bracket, never as offset text
    axis.set_ylabel("Cell count", fontsize=layout.FIGURE_LAYOUT.label_size)
    axis.set_title(
        f"{layout.quantity_label(quantity)} distribution",
        fontsize=layout.FIGURE_LAYOUT.axis_title_size,
    )
    axis.tick_params(labelsize=layout.FIGURE_LAYOUT.tick_size)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=layout.FIGURE_LAYOUT.legend_size, ncols=2)


def _formatted_table_value(value: object) -> str:
    """Return one compact figure-table value without changing its meaning."""
    if isinstance(value, (int, float, np.number)) and not isinstance(
        value,
        (bool, np.bool_),
    ):
        return f"{float(value):.4g}"
    return str(value)


def _draw_summary_table(
    axis: Axes,
    table: pd.DataFrame,
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> None:
    """Draw one compact four-value summary inside a dedicated figure column."""
    if not isinstance(table, pd.DataFrame):
        msg = "Spatial summary tables require a pandas DataFrame."
        raise TypeError(msg)
    axis.set_axis_off()
    row_labels = tuple(label for _section, _category, label in table.index)
    cell_text = tuple(
        (
            row_label,
            *(_formatted_table_value(value) for value in row),
        )
        for row_label, row in zip(
            row_labels,
            table.itertuples(index=False, name=None),
            strict=True,
        )
    )
    column_labels = (
        "Summary",
        f"Case {first.case.case_index}\n(A)",
        f"Mean A\n(n={mean_a.case_count})",
        f"Case {second.case.case_index}\n(B)",
        f"Mean B\n(n={mean_b.case_count})",
    )
    figure = axis.get_figure()
    if figure is None:
        msg = "The summary table needs an axis on a figure."
        raise RuntimeError(msg)
    axis_height_in = axis.get_position().height * figure.bbox.height / figure.dpi
    height = min(1.0, (len(cell_text) + 1) * _SUMMARY_ROW_HEIGHT_IN / axis_height_in)
    artist = axis.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=_SUMMARY_COLUMN_WIDTHS,
        bbox=Bbox.from_bounds(0.0, 1.0 - height, 1.0, height),
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(layout.FIGURE_LAYOUT.table_size)
    value_colors = tables.row_local_color_matrix(table)
    for (row, column), cell in artist.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold", color="#1f2933")
        elif column == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(ha="left", color="#1f2933")
        else:
            cell_colors = value_colors.iloc[row - 1, column - 1]
            if isinstance(cell_colors, tables.TableCellColors):
                cell.set_facecolor(cell_colors.background)
                cell.set_text_props(color=cell_colors.foreground)


def _row_label(label: str) -> str:
    """Wrap a row label between words to lines of ``_ROW_LABEL_WIDTH`` characters, never inside a mathtext span."""
    protected = _MATH_SPAN.sub(lambda match: match.group(0).replace(" ", "\0"), label)
    return textwrap.fill(protected, _ROW_LABEL_WIDTH, break_long_words=False, break_on_hyphens=False).replace("\0", " ")


def _spatial_incompatibility(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
) -> str | None:
    """Return the first exact grid incompatibility across selected fields."""
    for quantity in quantities:
        message = diagnostics.spatial_difference_compatibility(
            first,
            second,
            quantity,
        )
        if message is not None:
            return message
    return None


def _draw_comparison_map_row(
    figure: Figure,
    axes: np.ndarray,
    slots: np.ndarray,
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantity: str,
    *,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> None:
    """Draw one actual-case A/B/difference row of the shared map grid, every map with its own unit-bearing colour bar."""
    difference = field_labels.display_values(
        diagnostics.compatible_field_difference(first, second, quantity),
        diagnostics.FIELD_UNITS[quantity],
        field=quantity,
        quantity_kind="difference",
    )
    first_values, second_values = layout.display_field(first, quantity), layout.display_field(second, quantity)
    first_norm, second_norm = layout.comparison_norms(quantity, first_values, second_values, lock_scale=lock_scale)
    unit = layout.quantity_colorbar_label(quantity)
    for axis, slot, values, norm, label in (
        (axes[0], slots[0], first_values, first_norm, unit),
        (axes[1], slots[1], second_values, second_norm, unit),
        (axes[2], slots[2], difference, layout.signed_norm(difference), layout.quantity_colorbar_label(quantity, difference=True)),
    ):
        image = layout.draw_field(axis, first, values, norm, display_spatial_stride=display_spatial_stride)
        shared_layout.add_field_map_colorbar(figure, image, slot, norm, label=label)


def _draw_map_rows(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
    *,
    lower_rows: int,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride,
) -> tuple[Figure, GridSpec]:
    """
    Draw the A, B and B-minus-A map rows on the shared map grid and return the figure with the grid of its lower band.

    The columns are headed once above the first row and each row is named once at its left. The lower band below the maps
    holds ``lower_rows`` rows of ``FIGURE_LAYOUT.distribution_row_height`` as two columns: wide plots on the left, the
    summary column of ``_SUMMARY_WIDTH_IN`` on the right, together as wide as the maps.
    """
    extent = display_grid.physical_extent(first.fields["x"], first.fields["y"])
    row_height = layout.FIGURE_LAYOUT.distribution_row_height
    below_in = lower_rows * row_height + (lower_rows - 1) * shared_layout.BAND_GAP_V_IN
    geometry = shared_layout.plan_geometry(len(quantities), _MAP_COLUMNS, (extent[1] - extent[0], extent[3] - extent[2]), below_in=below_in)
    figure, axes, slots = shared_layout.map_grid(geometry, extent=extent)
    for row, quantity in enumerate(quantities):
        _draw_comparison_map_row(
            figure, axes[row], slots[row], first, second, quantity, lock_scale=lock_scale, display_spatial_stride=display_spatial_stride
        )
    case_a, case_b = first.case.case_index, second.case.case_index
    difference_header = f"Case {case_b} (B) \N{MINUS SIGN} Case {case_a} (A)"
    shared_layout.set_column_headers(axes, (f"Case {case_a} (A)", f"Case {case_b} (B)", difference_header), ())
    shared_layout.set_row_labels(axes, [_row_label(layout.quantity_label(quantity)) for quantity in quantities])
    left, bottom, width, height = shared_layout.band_rect(geometry, "below")
    lower = figure.add_gridspec(
        lower_rows,
        2,
        left=left,
        right=left + width,
        bottom=bottom,
        top=bottom + height,
        width_ratios=(geometry.grid_width - _SUMMARY_WIDTH_IN - _LOWER_GAP_IN, _SUMMARY_WIDTH_IN),
        wspace=2.0 * _LOWER_GAP_IN / (geometry.grid_width - _LOWER_GAP_IN),
        hspace=shared_layout.BAND_GAP_V_IN / row_height,
    )
    return figure, lower


def _pressure_line(
    record: diagnostics.GenerationInputDiagnostics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the diagnostics-owned inlet-pressure boundary support."""
    return diagnostics.inlet_pressure_boundary(record)


def _dataset_pressure_line(
    dataset: diagnostics.DatasetDiagnostics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact pointwise empirical inlet-pressure mean."""
    supports_and_values = tuple(_pressure_line(record) for record in dataset.records)
    support = supports_and_values[0][0]
    if any(not np.array_equal(other_support, support) for other_support, _values in supports_and_values[1:]):
        msg = "Dataset inlet-pressure mean requires identical persisted x support."
        raise ValueError(msg)
    values = np.mean(
        np.stack(tuple(item[1] for item in supports_and_values)),
        axis=0,
    )
    return support, values


def _draw_pressure_comparison(
    axis: Axes,
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    same_dataset: bool,
) -> None:
    """Draw case pressure boundaries and exact pointwise dataset means."""
    first_x, first_values = _pressure_line(first)
    second_x, second_values = _pressure_line(second)
    if not np.array_equal(first_x, second_x):
        msg = "Case pressure-line comparison requires identical persisted x support."
        raise ValueError(msg)
    axis.plot(
        first_x,
        first_values,
        color=layout.DATASET_A_COLOR,
        linewidth=layout.FIGURE_LAYOUT.line_width,
        label=_case_plot_label(first, "A"),
    )
    axis.plot(
        second_x,
        second_values,
        color=layout.DATASET_B_COLOR,
        linewidth=layout.FIGURE_LAYOUT.line_width,
        label=_case_plot_label(second, "B"),
    )
    mean_a_x, mean_a_values = _dataset_pressure_line(mean_a)
    if not np.array_equal(mean_a_x, first_x):
        msg = "Dataset A pressure mean must match the selected case support exactly."
        raise ValueError(msg)
    if same_dataset:
        axis.plot(
            mean_a_x,
            mean_a_values,
            color=layout.DATASET_MEAN_COLOR,
            linestyle="--",
            linewidth=layout.FIGURE_LAYOUT.line_width,
            label=f"Dataset mean, n = {mean_a.case_count}",
        )
    else:
        mean_b_x, mean_b_values = _dataset_pressure_line(mean_b)
        if not np.array_equal(mean_b_x, second_x):
            msg = "Dataset B pressure mean must match the selected case support exactly."
            raise ValueError(msg)
        axis.plot(
            mean_a_x,
            mean_a_values,
            color=layout.DATASET_A_COLOR,
            linestyle="--",
            linewidth=layout.FIGURE_LAYOUT.line_width,
            label=f"Mean A, n = {mean_a.case_count}",
        )
        axis.plot(
            mean_b_x,
            mean_b_values,
            color=layout.DATASET_B_COLOR,
            linestyle="--",
            linewidth=layout.FIGURE_LAYOUT.line_width,
            label=f"Mean B, n = {mean_b.case_count}",
        )
    axis.set_xlabel("x [m]", fontsize=layout.FIGURE_LAYOUT.label_size)
    axis.set_ylabel(
        "Inlet pressure [Pa]",
        fontsize=layout.FIGURE_LAYOUT.label_size,
    )
    axis.set_title(
        "Inlet pressure boundary",
        fontsize=layout.FIGURE_LAYOUT.axis_title_size,
    )
    axis.tick_params(labelsize=layout.FIGURE_LAYOUT.tick_size)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=layout.FIGURE_LAYOUT.legend_size, ncols=2)


def basic_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure | widgets.HTML:
    """Render a porosity map row with a lower band of the compact distribution, the pressure line, and the table."""
    quantities = ("eps_bed", "p_in_bc")
    _validate_quantities(first, quantities)
    _validate_quantities(second, quantities)
    incompatibility = _spatial_incompatibility(first, second, quantities)
    if incompatibility is not None:
        return widgets.HTML(f"<p><b>Spatial comparison unavailable.</b> {incompatibility} Scalar field summaries remain available.</p>")
    figure, lower = _draw_map_rows(first, second, ("eps_bed",), lower_rows=2, lock_scale=lock_scale, display_spatial_stride=display_spatial_stride)
    same_dataset = mean_a.batch_identity == mean_b.batch_identity
    _draw_distribution(figure.add_subplot(lower[0, 0]), first, mean_a, second, mean_b, "eps_bed", same_dataset=same_dataset)
    _draw_pressure_comparison(figure.add_subplot(lower[1, 0]), first, mean_a, second, mean_b, same_dataset=same_dataset)
    table = diagnostics.field_summary_comparison_table(first, mean_a, second, mean_b, quantities=quantities)
    _draw_summary_table(figure.add_subplot(lower[:, 1]), table, first, mean_a, second, mean_b)
    scale_state = "locked A/B scales" if lock_scale else "independent A/B scales"
    shared_layout.set_map_title(figure, f"Basic spatial inputs — {scale_state}")
    return figure


def comparison_block(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    quantities: Sequence[str],
    *,
    title: str,
    lock_scale: bool,
    side_plot: Callable[[Axes], None] | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure | widgets.HTML:
    """
    Render A/B/difference map rows above a lower band of one distribution per field and a summary table.

    Each map has its own unit-bearing colour bar. Optional side content is drawn in the lower band's summary column below
    the compact summary, with its legend inside its axis.
    """
    selected = _validate_quantities(first, quantities)
    _validate_quantities(second, selected)
    incompatibility = _spatial_incompatibility(first, second, selected)
    if incompatibility is not None:
        return widgets.HTML(f"<p><b>Spatial comparison unavailable.</b> {incompatibility} Scalar field summaries remain available.</p>")
    if side_plot is not None and len(selected) < _MINIMUM_SIDE_PLOT_ROWS:
        msg = "A side plot requires at least two lower distribution rows."
        raise ValueError(msg)
    figure, lower = _draw_map_rows(
        first, second, selected, lower_rows=len(selected), lock_scale=lock_scale, display_spatial_stride=display_spatial_stride
    )
    same_dataset = mean_a.batch_identity == mean_b.batch_identity
    for row, quantity in enumerate(selected):
        _draw_distribution(figure.add_subplot(lower[row, 0]), first, mean_a, second, mean_b, quantity, same_dataset=same_dataset)
    if side_plot is None:
        summary_axis = figure.add_subplot(lower[:, 1])
    else:
        summary_axis = figure.add_subplot(lower[0, 1])
        side_plot(figure.add_subplot(lower[1:, 1]))
    table = diagnostics.field_summary_comparison_table(first, mean_a, second, mean_b, quantities=selected)
    _draw_summary_table(summary_axis, table, first, mean_a, second, mean_b)
    scale_state = "locked A/B scales" if lock_scale else "independent A/B scales"
    shared_layout.set_map_title(figure, f"{title} — {scale_state}")
    return figure
