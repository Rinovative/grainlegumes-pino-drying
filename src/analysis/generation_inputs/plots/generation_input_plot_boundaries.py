"""
===============================================================================
generation_input_plot_boundaries.py
===============================================================================
Render unified boundary tables and transient schedule comparisons.
Responsibilities:
  - Present boundary scalars as Case A, Mean A, Case B, and Mean B
  - Plot every transient channel over operating and first-hour intervals
  - Draw case and empirical-mean schedules on exact persisted supports
  - Report unavailable pointwise means when dataset supports differ
Design principles:
  - Cases are solid and dataset means are dashed with consistent A/B colors
  - Relative humidity is averaged directly as its persisted channel
  - No schedule is interpolated, resampled, or reconstructed for comparison
This module does NOT:
  - Render single-case schedules or separate case-versus-mean views
  - Modify startup semantics, psychrometric logic, or generation inputs
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.ui import tables
from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ipywidgets as widgets
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_EARLY_OPERATION_END_H = 1.0


def boundary_comparison_table(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render the exact four-column boundary and scalar-summary table."""
    table = diagnostics.boundary_comparison_table(
        first,
        mean_a,
        second,
        mean_b,
    )
    return tables.grouped_styled_dataframes(
        diagnostics.grouped_table_sections(table),
        title=(f"Boundary conditions — Mean A n = {mean_a.case_count}; Mean B n = {mean_b.case_count}"),
        columns=2,
        shade_constant=True,
        row_local=True,
    )


def _schedule_axes() -> tuple[Figure, tuple[tuple[Axes, Axes], ...]]:
    """Create three channel rows with operating and first-hour columns."""
    figure, grid = layout.figure_grid(
        columns=2,
        row_heights=(layout.MAP_LAYOUT.schedule_row_height,) * 3,
    )
    axes = tuple(
        (
            figure.add_subplot(grid[row, 0]),
            figure.add_subplot(grid[row, 1]),
        )
        for row in range(3)
    )
    return figure, axes


def _early_operation_ticks(*supports: Iterable[float]) -> tuple[float, ...]:
    """Return exact first-hour support ticks in display minutes."""
    return tuple(sorted({60.0 * float(value) for support in supports for value in support if 0.0 <= float(value) <= _EARLY_OPERATION_END_H}))


def _matching_startup_duration(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
) -> float:
    """Return the compared persisted startup duration or reject mismatched semantics."""
    first_duration = diagnostics.transient_evidence(first)[3].duration_h
    second_duration = diagnostics.transient_evidence(second)[3].duration_h
    if first_duration != second_duration:
        msg = f"Compared transient schedules have different persisted startup durations: A={first_duration:g} h, B={second_duration:g} h."
        raise ValueError(msg)
    return first_duration


def _plot_schedule(
    axis: Axes,
    schedule: np.ndarray,
    *,
    column: int,
    name: str,
    unit: str,
    startup_duration_h: float,
    early_operation: bool,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot exact operating or first-hour rows without changing the source schedule."""
    window_end_h = _EARLY_OPERATION_END_H if early_operation else startup_duration_h
    rows = diagnostics.schedule_window_rows(
        schedule,
        window_end_h,
        startup_only=early_operation,
    )
    times = 60.0 * rows[:, 0] if early_operation else rows[:, 0]

    axis.plot(
        times,
        diagnostics.display_value(name, rows[:, column], unit),
        color=color,
        linestyle=linestyle,
        linewidth=layout.MAP_LAYOUT.line_width,
        label=label,
    )


def schedule_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    same_dataset: bool,
) -> Figure:
    """
    Plot Case A, Mean A, Case B, and Mean B in every transient channel.

    Mean schedules are included only when all unique cases in the corresponding
    dataset share the exact persisted support. If supports differ, case lines
    remain and the figure reports why the pointwise mean is unavailable.
    """
    first_schedule = diagnostics.transient_evidence(first)[0]
    second_schedule = diagnostics.transient_evidence(second)[0]
    startup_duration_h = _matching_startup_duration(first, second)

    figure, axes = _schedule_axes()

    channel_contract = tuple(
        zip(
            range(1, len(profiles.SCHEDULE_FIELDS)),
            profiles.SCHEDULE_FIELDS[1:],
            profiles.SCHEDULE_UNITS[1:],
            strict=True,
        )
    )

    supports = [
        diagnostics.schedule_window_rows(
            first_schedule,
            _EARLY_OPERATION_END_H,
            startup_only=True,
        )[:, 0],
        diagnostics.schedule_window_rows(
            second_schedule,
            _EARLY_OPERATION_END_H,
            startup_only=True,
        )[:, 0],
    ]

    if mean_a.schedule_mean is not None:
        mean_times = mean_a.schedule_mean[:, 0]
        supports.append(mean_times[(mean_times >= 0.0) & (mean_times <= _EARLY_OPERATION_END_H)])

    if not same_dataset and mean_b.schedule_mean is not None:
        mean_times = mean_b.schedule_mean[:, 0]
        supports.append(mean_times[(mean_times >= 0.0) & (mean_times <= _EARLY_OPERATION_END_H)])

    early_operation_ticks = _early_operation_ticks(*supports)

    for row, (column, name, unit) in enumerate(channel_contract):
        for axis, early_operation in (
            (axes[row][0], False),
            (axes[row][1], True),
        ):
            _plot_schedule(
                axis,
                first_schedule,
                column=column,
                name=name,
                unit=unit,
                startup_duration_h=startup_duration_h,
                early_operation=early_operation,
                color=layout.DATASET_A_COLOR,
                linestyle="-",
                label=f"Case {first.case.case_index} (A)",
            )

            if mean_a.schedule_mean is not None:
                mean_label = f"Dataset mean, n = {mean_a.case_count}" if same_dataset else f"Mean A, n = {mean_a.case_count}"
                _plot_schedule(
                    axis,
                    mean_a.schedule_mean,
                    column=column,
                    name=name,
                    unit=unit,
                    startup_duration_h=startup_duration_h,
                    early_operation=early_operation,
                    color=(layout.DATASET_MEAN_COLOR if same_dataset else layout.DATASET_A_COLOR),
                    linestyle="--",
                    label=mean_label,
                )

            _plot_schedule(
                axis,
                second_schedule,
                column=column,
                name=name,
                unit=unit,
                startup_duration_h=startup_duration_h,
                early_operation=early_operation,
                color=layout.DATASET_B_COLOR,
                linestyle="-",
                label=f"Case {second.case.case_index} (B)",
            )

            if not same_dataset and mean_b.schedule_mean is not None:
                _plot_schedule(
                    axis,
                    mean_b.schedule_mean,
                    column=column,
                    name=name,
                    unit=unit,
                    startup_duration_h=startup_duration_h,
                    early_operation=early_operation,
                    color=layout.DATASET_B_COLOR,
                    linestyle="--",
                    label=f"Mean B, n = {mean_b.case_count}",
                )

            axis.set_ylabel(
                f"{name} [{diagnostics.display_unit(name, unit)}]",
                fontsize=layout.MAP_LAYOUT.label_size,
            )
            axis.tick_params(labelsize=layout.MAP_LAYOUT.tick_size)
            axis.grid(alpha=0.24)

        axes[row][1].set_xlim(
            0.0,
            60.0 * _EARLY_OPERATION_END_H,
        )
        axes[row][1].set_xticks(early_operation_ticks)

    axes[0][0].set_title(
        f"Operating schedule: {60.0 * startup_duration_h:g} min onward",
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )
    axes[0][1].set_title(
        (f"Startup and early operation: 0-{60.0 * _EARLY_OPERATION_END_H:g} min"),
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )

    axes[-1][0].set_xlabel(
        "time [h]",
        fontsize=layout.MAP_LAYOUT.label_size,
    )
    axes[-1][1].set_xlabel(
        "time [min]",
        fontsize=layout.MAP_LAYOUT.label_size,
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=min(4, len(labels)),
        fontsize=layout.MAP_LAYOUT.legend_size,
    )

    unavailable = []
    if mean_a.schedule_mean_unavailable is not None:
        unavailable.append(f"Mean A: {mean_a.schedule_mean_unavailable}")
    if not same_dataset and mean_b.schedule_mean_unavailable is not None:
        unavailable.append(f"Mean B: {mean_b.schedule_mean_unavailable}")

    if unavailable:
        figure.text(
            0.01,
            0.01,
            " ".join(unavailable),
            fontsize=layout.MAP_LAYOUT.tick_size,
            ha="left",
            va="bottom",
        )

    figure.suptitle(
        (f"Transient schedules — Case {first.case.case_index} (A) vs Case {second.case.case_index} (B)"),
        fontsize=layout.MAP_LAYOUT.title_size,
    )

    return figure
