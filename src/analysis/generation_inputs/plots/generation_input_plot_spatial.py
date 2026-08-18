"""
generation_input_plot_spatial.py

Render coordinated A/B/B-minus-A blocks for generation-input fields.
Responsibilities:
  - Group maintained channels as rows under three fixed map columns
  - Use independent A/B physical scales unless the local scale lock is enabled
  - Give every map its own axes-coupled colorbar
  - Compose two-column distributions, pressure lines, and embedded summaries
Design principles:
  - B-minus-A remains physical, symmetric, and exactly centred at zero
  - Distribution axes never participate in map colorbar geometry
  - Incompatible grids are reported without interpolation or resampling
This module does NOT:
  - Render single-case alternatives, derive fields, or load input files
  - Pool unrelated fields or display dataset-mean maps
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.ui import tables

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_MINIMUM_SIDE_PLOT_ROWS: Final = 2
_SUMMARY_VALUE_COLUMN_COUNT: Final = 4
_SUMMARY_LABEL_COLUMN_FRACTION: Final = 0.52
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
    """Draw A/B field distributions and unambiguous dataset-mean markers."""
    lower, upper = _histogram_bounds(
        first.fields[quantity],
        second.fields[quantity],
    )
    bins = tuple(float(value) for value in np.linspace(lower, upper, 31))
    axis.hist(
        first.fields[quantity].ravel(),
        bins=bins,
        alpha=0.44,
        color=layout.DATASET_A_COLOR,
        label=_case_plot_label(first, "A"),
    )
    axis.hist(
        second.fields[quantity].ravel(),
        bins=bins,
        alpha=0.44,
        color=layout.DATASET_B_COLOR,
        label=_case_plot_label(second, "B"),
    )
    first_mean = mean_a.field_summary_means[(quantity, "mean")]
    second_mean = mean_b.field_summary_means[(quantity, "mean")]
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
        layout.quantity_axis_label(quantity),
        fontsize=layout.MAP_LAYOUT.label_size,
    )
    axis.set_ylabel("Cell count", fontsize=layout.MAP_LAYOUT.label_size)
    axis.set_title(
        f"{diagnostics.FIELD_LABELS[quantity]} distribution",
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )
    axis.tick_params(labelsize=layout.MAP_LAYOUT.tick_size)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=layout.MAP_LAYOUT.legend_size, ncols=2)


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
    artist = axis.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=_SUMMARY_COLUMN_WIDTHS,
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0),
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(layout.MAP_LAYOUT.table_size)
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


def _apply_map_axis_labels(
    axes: Sequence[Sequence[Axes]],
) -> None:
    """Keep coordinate labels only on the outer edges of one map grid."""
    rows = tuple(tuple(row) for row in axes)
    for row_index, row in enumerate(rows):
        for column_index, axis in enumerate(row):
            bottom = row_index == len(rows) - 1
            left = column_index == 0
            axis.set_xlabel(
                "x [m]" if bottom else "",
                fontsize=layout.MAP_LAYOUT.label_size,
            )
            axis.set_ylabel(
                "y [m]" if left else "",
                fontsize=layout.MAP_LAYOUT.label_size,
            )
            axis.tick_params(
                axis="x",
                labelbottom=bottom,
                labelsize=layout.MAP_LAYOUT.tick_size,
            )
            axis.tick_params(
                axis="y",
                labelleft=left,
                labelsize=layout.MAP_LAYOUT.tick_size,
            )


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
    axes: tuple[Axes, Axes, Axes],
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantity: str,
    *,
    lock_scale: bool,
) -> None:
    """Draw one actual-case A/B/difference row with independent colorbars."""
    difference = diagnostics.compatible_field_difference(
        first,
        second,
        quantity,
    )
    first_norm, second_norm = layout.comparison_norms(
        quantity,
        first.fields[quantity],
        second.fields[quantity],
        lock_scale=lock_scale,
    )
    difference_norm = layout.signed_norm(difference)
    first_image = layout.draw_map(
        axes[0],
        first,
        quantity,
        title=(f"Case {first.case.case_index}\n{diagnostics.FIELD_LABELS[quantity]}"),
        norm=first_norm,
    )
    second_image = layout.draw_map(
        axes[1],
        second,
        quantity,
        title=(f"Case {second.case.case_index}\n{diagnostics.FIELD_LABELS[quantity]}"),
        norm=second_norm,
    )
    difference_image = layout.draw_array_map(
        axes[2],
        first.fields["x"],
        first.fields["y"],
        difference,
        title=(f"Case {second.case.case_index} - Case {first.case.case_index}\n{diagnostics.FIELD_LABELS[quantity]}"),
        norm=difference_norm,
    )
    if lock_scale and first_image.norm is not second_image.norm:
        msg = "Locked A/B maps must share the exact normalization object."
        raise RuntimeError(msg)
    if not lock_scale and first_image.norm is second_image.norm:
        msg = "Unlocked A/B maps must retain independent normalizations."
        raise RuntimeError(msg)
    colorbar_label = layout.quantity_colorbar_label(quantity)
    layout.add_map_colorbar(
        figure,
        first_image,
        axes[0],
        label=colorbar_label,
    )
    layout.add_map_colorbar(
        figure,
        second_image,
        axes[1],
        label=colorbar_label,
    )
    layout.add_map_colorbar(
        figure,
        difference_image,
        axes[2],
        label=colorbar_label,
    )


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
        linewidth=layout.MAP_LAYOUT.line_width,
        label=_case_plot_label(first, "A"),
    )
    axis.plot(
        second_x,
        second_values,
        color=layout.DATASET_B_COLOR,
        linewidth=layout.MAP_LAYOUT.line_width,
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
            linewidth=layout.MAP_LAYOUT.line_width,
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
            linewidth=layout.MAP_LAYOUT.line_width,
            label=f"Mean A, n = {mean_a.case_count}",
        )
        axis.plot(
            mean_b_x,
            mean_b_values,
            color=layout.DATASET_B_COLOR,
            linestyle="--",
            linewidth=layout.MAP_LAYOUT.line_width,
            label=f"Mean B, n = {mean_b.case_count}",
        )
    axis.set_xlabel("x [m]", fontsize=layout.MAP_LAYOUT.label_size)
    axis.set_ylabel(
        "Inlet pressure [Pa]",
        fontsize=layout.MAP_LAYOUT.label_size,
    )
    axis.set_title(
        "Inlet pressure boundary",
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )
    axis.tick_params(labelsize=layout.MAP_LAYOUT.tick_size)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=layout.MAP_LAYOUT.legend_size, ncols=2)


def basic_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
) -> Figure | widgets.HTML:
    """Render a porosity map row, compact distribution, pressure line, and table."""
    quantities = ("eps_bed", "p_in_bc")
    _validate_quantities(first, quantities)
    _validate_quantities(second, quantities)
    incompatibility = _spatial_incompatibility(first, second, quantities)
    if incompatibility is not None:
        return widgets.HTML(f"<p><b>Spatial comparison unavailable.</b> {incompatibility} Scalar field summaries remain available.</p>")
    figure, grid = layout.figure_grid(
        columns=3,
        row_heights=(
            layout.MAP_LAYOUT.composite_map_row_height,
            layout.MAP_LAYOUT.distribution_row_height,
            layout.MAP_LAYOUT.distribution_row_height,
        ),
    )
    map_axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
    )
    _draw_comparison_map_row(
        figure,
        map_axes,
        first,
        second,
        "eps_bed",
        lock_scale=lock_scale,
    )
    _apply_map_axis_labels((map_axes,))
    same_dataset = mean_a.batch_identity == mean_b.batch_identity
    distribution_axis = figure.add_subplot(grid[1, :2])
    _draw_distribution(
        distribution_axis,
        first,
        mean_a,
        second,
        mean_b,
        "eps_bed",
        same_dataset=same_dataset,
    )
    pressure_axis = figure.add_subplot(grid[2, :2])
    _draw_pressure_comparison(
        pressure_axis,
        first,
        mean_a,
        second,
        mean_b,
        same_dataset=same_dataset,
    )
    summary_axis = figure.add_subplot(grid[1:, 2])
    _draw_summary_table(
        summary_axis,
        diagnostics.field_summary_comparison_table(
            first,
            mean_a,
            second,
            mean_b,
            quantities=quantities,
        ),
        first,
        mean_a,
        second,
        mean_b,
    )
    scale_state = "locked A/B scales" if lock_scale else "independent A/B scales"
    figure.suptitle(
        f"Basic spatial inputs — {scale_state}",
        fontsize=layout.MAP_LAYOUT.title_size,
    )
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
    include_distributions: bool = True,
    side_plot: Callable[[Axes, Axes], None] | None = None,
) -> Figure | widgets.HTML:
    """
    Render A/B/difference maps above two-column distributions and a summary.

    Each map retains its own axes-coupled colorbar. Optional side content is
    placed below the compact summary within the third lower column.
    """
    selected = _validate_quantities(first, quantities)
    _validate_quantities(second, selected)
    incompatibility = _spatial_incompatibility(first, second, selected)
    if incompatibility is not None:
        return widgets.HTML(f"<p><b>Spatial comparison unavailable.</b> {incompatibility} Scalar field summaries remain available.</p>")
    lower_row_count = len(selected) if include_distributions else 1
    row_heights = (layout.MAP_LAYOUT.composite_map_row_height,) * len(selected) + (layout.MAP_LAYOUT.distribution_row_height,) * lower_row_count
    figure, grid = layout.figure_grid(
        columns=3,
        row_heights=row_heights,
    )
    map_axes = []
    for row, quantity in enumerate(selected):
        axes = (
            figure.add_subplot(grid[row, 0]),
            figure.add_subplot(grid[row, 1]),
            figure.add_subplot(grid[row, 2]),
        )
        map_axes.append(axes)
        _draw_comparison_map_row(
            figure,
            axes,
            first,
            second,
            quantity,
            lock_scale=lock_scale,
        )
    _apply_map_axis_labels(map_axes)
    lower_start = len(selected)
    same_dataset = mean_a.batch_identity == mean_b.batch_identity
    distribution_axes: list[Axes] = []
    if include_distributions:
        for offset, quantity in enumerate(selected):
            axis = figure.add_subplot(grid[lower_start + offset, :2])
            distribution_axes.append(axis)
            _draw_distribution(
                axis,
                first,
                mean_a,
                second,
                mean_b,
                quantity,
                same_dataset=same_dataset,
            )
    else:
        empty_axis = figure.add_subplot(grid[lower_start, :2])
        empty_axis.set_axis_off()
    if side_plot is None:
        summary_axis = figure.add_subplot(grid[lower_start:, 2])
    else:
        if not include_distributions or lower_row_count < _MINIMUM_SIDE_PLOT_ROWS:
            msg = "A side plot requires at least two lower distribution rows."
            raise ValueError(msg)
        summary_axis = figure.add_subplot(grid[lower_start, 2])
        relation_grid = grid[lower_start + 1 :, 2].subgridspec(
            2,
            1,
            height_ratios=layout.MAP_LAYOUT.relation_height_ratios,
            hspace=layout.MAP_LAYOUT.relation_vertical_spacing,
        )
        side_axis = figure.add_subplot(relation_grid[0, 0])
        legend_axis = figure.add_subplot(relation_grid[1, 0])
        legend_axis.set_axis_off()
        side_plot(side_axis, legend_axis)
    _draw_summary_table(
        summary_axis,
        diagnostics.field_summary_comparison_table(
            first,
            mean_a,
            second,
            mean_b,
            quantities=selected,
        ),
        first,
        mean_a,
        second,
        mean_b,
    )
    if side_plot is not None:
        layout.align_axis_to_references(
            figure,
            summary_axis,
            distribution_axes[0],
            right_reference=side_axis,
        )
    scale_state = "locked A/B scales" if lock_scale else "independent A/B scales"
    figure.suptitle(
        f"{title} — {scale_state}",
        fontsize=layout.MAP_LAYOUT.title_size,
    )
    return figure
