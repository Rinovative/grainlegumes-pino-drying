"""
===============================================================================
generation_input_plot_boundaries.py
===============================================================================
Render unified boundary tables and transient schedule comparisons.
Responsibilities:
  - Present boundary scalars as Case A, Mean A, Case B, and Mean B
  - Plot every transient channel over full and startup intervals
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

import numpy as np

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.ui import tables
from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
    """Create three channel rows with full-horizon and startup columns."""
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


def _startup_ticks(*supports: Iterable[float]) -> tuple[float, ...]:
    """Return exact persisted support ticks inside the startup interval."""
    return tuple(sorted({float(value) for support in supports for value in support if 0.0 <= float(value) <= 1.0}))


def _plot_schedule(
    axis: Axes,
    schedule: np.ndarray,
    *,
    column: int,
    limit: float,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot one persisted schedule channel without support transformation."""
    rows = schedule[schedule[:, 0] <= limit]
    axis.plot(
        rows[:, 0],
        rows[:, column],
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
    figure, axes = _schedule_axes()
    channel_contract = tuple(
        zip(
            range(1, len(profiles.SCHEDULE_FIELDS)),
            profiles.SCHEDULE_FIELDS[1:],
            profiles.SCHEDULE_UNITS[1:],
            strict=True,
        )
    )
    supports = [first_schedule[:, 0], second_schedule[:, 0]]
    if mean_a.schedule_mean is not None:
        supports.append(mean_a.schedule_mean[:, 0])
    if not same_dataset and mean_b.schedule_mean is not None:
        supports.append(mean_b.schedule_mean[:, 0])
    startup_ticks = _startup_ticks(*supports)
    for row, (column, name, unit) in enumerate(channel_contract):
        for axis, startup_only in (
            (axes[row][0], False),
            (axes[row][1], True),
        ):
            limit = 1.0 if startup_only else np.inf
            _plot_schedule(
                axis,
                first_schedule,
                column=column,
                limit=limit,
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
                    limit=limit,
                    color=(layout.DATASET_MEAN_COLOR if same_dataset else layout.DATASET_A_COLOR),
                    linestyle="--",
                    label=mean_label,
                )
            _plot_schedule(
                axis,
                second_schedule,
                column=column,
                limit=limit,
                color=layout.DATASET_B_COLOR,
                linestyle="-",
                label=f"Case {second.case.case_index} (B)",
            )
            if not same_dataset and mean_b.schedule_mean is not None:
                _plot_schedule(
                    axis,
                    mean_b.schedule_mean,
                    column=column,
                    limit=limit,
                    color=layout.DATASET_B_COLOR,
                    linestyle="--",
                    label=f"Mean B, n = {mean_b.case_count}",
                )
            axis.set_ylabel(
                f"{name} [{unit}]",
                fontsize=layout.MAP_LAYOUT.label_size,
            )
            axis.tick_params(labelsize=layout.MAP_LAYOUT.tick_size)
            axis.grid(alpha=0.24)
        axes[row][1].set_xlim(0.0, 1.0)
        axes[row][1].set_xticks(startup_ticks)
    axes[0][0].set_title(
        "Full horizon",
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )
    axes[0][1].set_title(
        "Startup interval: 0-1 h",
        fontsize=layout.MAP_LAYOUT.axis_title_size,
    )
    axes[-1][0].set_xlabel(
        "time [h]",
        fontsize=layout.MAP_LAYOUT.label_size,
    )
    axes[-1][1].set_xlabel(
        "time [h]",
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
