"""Render unified boundary tables and transient schedule comparisons.

Responsibilities:
  - Present boundary scalars as Case A, Mean A, Case B, and Mean B, unshaded: a larger value ranks nothing
  - Plot primitive and derived boundary quantities over operation and the first hour
  - Derive RH after display-time interpolation and average per-case curves
  - Report unavailable pointwise means when dataset supports differ

Design principles:
  - Cases are solid and dataset means are dashed in the colour of their case, keyed by the case captions; a one-case
    dataset draws no mean
  - Temperature and humidity ratio remain the only persisted schedule channels
  - Display interpolation never mutates or adds persisted schedule support

This module does NOT:
  - Render single-case schedules or separate case-versus-mean views
  - Modify startup semantics, psychrometric logic, or generation inputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import tables

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ipywidgets as widgets
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
    )


def _schedule_axes() -> tuple[Figure, np.ndarray]:
    """Create three channel rows with operating and first-hour columns of equal panels."""
    return figure_layout.panel_figure(3, 2, panel_in=(layout.FIGURE_LAYOUT.column_width, layout.FIGURE_LAYOUT.schedule_row_height))


def _early_operation_ticks(*supports: Iterable[float]) -> tuple[float, ...]:
    """Return exact first-hour support ticks in display minutes."""
    return tuple(sorted({60.0 * float(value) for support in supports for value in support if 0.0 <= float(value) <= _EARLY_OPERATION_END_H}))


def _matching_startup_policy(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
) -> tuple[bool, float]:
    """Return matching persisted startup enablement and duration."""
    first_startup = diagnostics.transient_evidence(first)[3]
    second_startup = diagnostics.transient_evidence(second)[3]
    if (first_startup.enabled, first_startup.duration_h) != (
        second_startup.enabled,
        second_startup.duration_h,
    ):
        msg = (
            "Compared transient schedules have different persisted startup policies: "
            f"A=({first_startup.enabled}, {first_startup.duration_h:g} h), "
            f"B=({second_startup.enabled}, {second_startup.duration_h:g} h)."
        )
        raise ValueError(msg)
    return first_startup.enabled, first_startup.duration_h


def _display_times(
    schedule: np.ndarray,
    *,
    startup_enabled: bool,
    startup_duration_h: float,
    early_operation: bool,
) -> np.ndarray:
    """Return display-only times without mutating primitive support."""
    if early_operation:
        return np.linspace(0.0, _EARLY_OPERATION_END_H, 61, dtype=np.float64)
    start_h = startup_duration_h if startup_enabled else 0.0
    return np.asarray(schedule[schedule[:, 0] >= start_h, 0], dtype=np.float64)


def _plot_schedule(
    axis: Axes,
    values: np.ndarray,
    *,
    column: int,
    name: str,
    unit: str,
    early_operation: bool,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot one evaluated primitive-or-derived boundary curve."""
    times = 60.0 * values[:, 0] if early_operation else values[:, 0]
    axis.plot(
        times,
        diagnostics.display_value(name, values[:, column], unit),
        color=color,
        linestyle=linestyle,
        linewidth=layout.FIGURE_LAYOUT.line_width,
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
    Plot primitive boundaries and derived inlet RH for cases and datasets.

    Temperature and humidity ratio are interpolated piecewise-linearly at every
    display time. Relative humidity is then derived per case. Dataset RH means
    average those per-case nonlinear curves. The mean of a one-case dataset is
    that case's own curve, so it is neither drawn nor keyed.
    """
    first_schedule = diagnostics.transient_evidence(first)[0]
    second_schedule = diagnostics.transient_evidence(second)[0]
    startup_enabled, startup_duration_h = _matching_startup_policy(first, second)
    figure, axes = _schedule_axes()
    channel_contract = (
        (1, "T_in_bc", "K"),
        (2, "omega_in_bc", "kg/kg"),
        (3, "phi_in_bc", "1"),
    )
    schedule_mean_a = mean_a.schedule_mean if mean_a.case_count > 1 else None
    schedule_mean_b = mean_b.schedule_mean if not same_dataset and mean_b.case_count > 1 else None

    color_a, color_b = layout.case_colors(mean_a, mean_b)
    caption_a, caption_b = layout.case_caption(first, mean_a, "A"), layout.case_caption(second, mean_b, "B")

    supports = [
        first_schedule[(first_schedule[:, 0] >= 0.0) & (first_schedule[:, 0] <= _EARLY_OPERATION_END_H), 0],
        second_schedule[(second_schedule[:, 0] >= 0.0) & (second_schedule[:, 0] <= _EARLY_OPERATION_END_H), 0],
    ]
    if schedule_mean_a is not None:
        mean_times = schedule_mean_a[:, 0]
        supports.append(mean_times[(mean_times >= 0.0) & (mean_times <= _EARLY_OPERATION_END_H)])
    if schedule_mean_b is not None:
        mean_times = schedule_mean_b[:, 0]
        supports.append(mean_times[(mean_times >= 0.0) & (mean_times <= _EARLY_OPERATION_END_H)])
    early_operation_ticks = _early_operation_ticks(*supports)

    for axis_pair, early_operation in ((0, False), (1, True)):
        first_times = _display_times(
            first_schedule,
            startup_enabled=startup_enabled,
            startup_duration_h=startup_duration_h,
            early_operation=early_operation,
        )
        second_times = _display_times(
            second_schedule,
            startup_enabled=startup_enabled,
            startup_duration_h=startup_duration_h,
            early_operation=early_operation,
        )
        first_values = diagnostics.case_boundary_schedule(first, first_times)
        second_values = diagnostics.case_boundary_schedule(second, second_times)
        mean_a_values = (
            None
            if schedule_mean_a is None
            else diagnostics.dataset_boundary_schedule(
                mean_a,
                _display_times(
                    schedule_mean_a,
                    startup_enabled=startup_enabled,
                    startup_duration_h=startup_duration_h,
                    early_operation=early_operation,
                ),
            )
        )
        mean_b_values = (
            None
            if schedule_mean_b is None
            else diagnostics.dataset_boundary_schedule(
                mean_b,
                _display_times(
                    schedule_mean_b,
                    startup_enabled=startup_enabled,
                    startup_duration_h=startup_duration_h,
                    early_operation=early_operation,
                ),
            )
        )
        for row, (column, name, unit) in enumerate(channel_contract):
            axis = axes[row][axis_pair]
            _plot_schedule(
                axis,
                first_values,
                column=column,
                name=name,
                unit=unit,
                early_operation=early_operation,
                color=color_a,
                linestyle="-",
                label=caption_a,
            )
            if mean_a_values is not None:
                mean_label = f"Dataset mean, n = {mean_a.case_count}" if same_dataset else f"Mean A, n = {mean_a.case_count}"
                _plot_schedule(
                    axis,
                    mean_a_values,
                    column=column,
                    name=name,
                    unit=unit,
                    early_operation=early_operation,
                    color=(layout.DATASET_MEAN_COLOR if same_dataset else color_a),
                    linestyle="--",
                    label=mean_label,
                )
            _plot_schedule(
                axis,
                second_values,
                column=column,
                name=name,
                unit=unit,
                early_operation=early_operation,
                color=color_b,
                linestyle="-",
                label=caption_b,
            )
            if mean_b_values is not None:
                _plot_schedule(
                    axis,
                    mean_b_values,
                    column=column,
                    name=name,
                    unit=unit,
                    early_operation=early_operation,
                    color=color_b,
                    linestyle="--",
                    label=f"Mean B, n = {mean_b.case_count}",
                )
            if axis_pair == 0:  # both columns show the same channel, so its label stands once per row
                axis.set_ylabel(f"{field_labels.field_label(name, mathtext=True)} [{diagnostics.display_unit(name, unit)}]")
            axis.grid(alpha=0.24)

    for row in range(3):
        axes[row][1].set_xlim(0.0, 60.0 * _EARLY_OPERATION_END_H)
        axes[row][1].set_xticks(early_operation_ticks)

    figure_layout.set_panel_title(
        axes[0][0], f"Operating schedule: {60.0 * startup_duration_h:g} min onward" if startup_enabled else "Operating schedule"
    )
    figure_layout.set_panel_title(
        axes[0][1],
        (
            f"Startup and early operation: 0-{60.0 * _EARLY_OPERATION_END_H:g} min"
            if startup_enabled
            else f"Early operation: 0-{60.0 * _EARLY_OPERATION_END_H:g} min"
        ),
    )
    axes[-1][0].set_xlabel("Time [h]")
    axes[-1][1].set_xlabel("Time [min]")

    unavailable = []
    if mean_a.schedule_mean_unavailable is not None:
        unavailable.append(f"Mean A: {mean_a.schedule_mean_unavailable}")
    if not same_dataset and mean_b.schedule_mean_unavailable is not None:
        unavailable.append(f"Mean B: {mean_b.schedule_mean_unavailable}")
    figure_layout.set_figure_title(
        figure,
        axes,
        f"Transient schedules — Case {first.case.case_index} (A) vs Case {second.case.case_index} (B)",  # the key names material and regime
        note=" ".join(unavailable) or None,
    )
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure_layout.add_legend(figure, axes, handles, legend_labels)
    return figure
