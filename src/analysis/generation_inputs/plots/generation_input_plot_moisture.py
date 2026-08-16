"""
===============================================================================
generation_input_plot_moisture.py
===============================================================================
Compose transient moisture, sorption, and inlet-bed A/B diagnostics.
Responsibilities:
  - Compare initial moisture and equilibrium-RH maps as A/B/B-minus-A rows
  - Retain separate A/B distributions with dataset-mean markers
  - Compare inlet startup RH with case and empirical bed-equilibrium summaries
Design principles:
  - Equilibrium RH comes only from the canonical per-case inverse-Oswin result
  - Dataset reference values average per-case nonlinear diagnostics
  - Moisture and humidity remain separate physical quantities
This module does NOT:
  - Render single-case alternatives, copy sorption equations, or resample fields
  - Display dataset-mean maps or alter generation acceptance
===============================================================================
"""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.lines import Line2D

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics

from . import generation_input_plot_layout as layout
from . import generation_input_plot_spatial as spatial

if TYPE_CHECKING:
    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

RELATION_MARKER_SIZE: Final = 38.0
RELATION_RANGE_LINE_WIDTH: Final = 6.0
RELATION_PHASE_COLORS: Final = MappingProxyType(
    {
        "Bed median (q05-q95 range)": "tab:purple",
        "Inlet start": "tab:green",
        "Inlet startup end": "tab:red",
    }
)
RELATION_PHASE_MARKERS: Final = (
    ("Bed median (q05-q95 range)", "o"),
    ("Inlet start", "^"),
    ("Inlet startup end", "v"),
)


def moisture_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
) -> Figure | widgets.HTML:
    """Render moisture and equilibrium RH maps with retained distributions."""
    diagnostics.transient_evidence(first)
    diagnostics.transient_evidence(second)
    return spatial.comparison_block(
        first,
        mean_a,
        second,
        mean_b,
        diagnostics.MOISTURE_FIELD_NAMES,
        title="Initial moisture and sorption",
        lock_scale=lock_scale,
        include_distributions=True,
        side_plot=partial(
            _draw_inlet_bed_comparison,
            first=first,
            mean_a=mean_a,
            second=second,
            mean_b=mean_b,
            same_dataset=(mean_a.batch_identity == mean_b.batch_identity),
        ),
    )


def _case_relationship(
    record: diagnostics.GenerationInputDiagnostics,
) -> tuple[float, float, float, float, float]:
    """Return bed q05/median/q95 and inlet startup endpoints for one case."""
    statistics = diagnostics.field_statistics(record).loc["phi_eq"]
    startup = diagnostics.transient_evidence(record)[3]
    inlet = startup.variables["phi_in_bc"]
    return (
        float(statistics["q05"]),
        float(statistics["median"]),
        float(statistics["q95"]),
        inlet.start,
        inlet.end,
    )


def _mean_relationship(
    dataset: diagnostics.DatasetDiagnostics,
) -> tuple[float, float, float, float, float]:
    """Return empirical means of per-case bed and inlet scalar summaries."""
    return (
        dataset.field_summary_means[("phi_eq", "q05")],
        dataset.field_summary_means[("phi_eq", "median")],
        dataset.field_summary_means[("phi_eq", "q95")],
        float(dataset.boundary_means["phi_in_bc start"]),
        float(dataset.boundary_means["phi_in_bc startup end"]),
    )


def _draw_inlet_bed_comparison(
    axis: Axes,
    legend_axis: Axes,
    *,
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    same_dataset: bool,
) -> None:
    """Draw inlet and bed phases using only phase colors and x-position roles."""
    diagnostics.transient_evidence(first)
    diagnostics.transient_evidence(second)
    labels: tuple[str, ...]
    values: tuple[tuple[float, float, float, float, float], ...]
    if same_dataset:
        labels = (
            f"Case {first.case.case_index}\n(A)",
            f"Dataset mean\n(n={mean_a.case_count})",
            f"Case {second.case.case_index}\n(B)",
        )
        values = (
            _case_relationship(first),
            _mean_relationship(mean_a),
            _case_relationship(second),
        )
    else:
        labels = (
            f"Case {first.case.case_index}\n(A)",
            f"Mean A\n(n={mean_a.case_count})",
            f"Case {second.case.case_index}\n(B)",
            f"Mean B\n(n={mean_b.case_count})",
        )
        values = (
            _case_relationship(first),
            _mean_relationship(mean_a),
            _case_relationship(second),
            _mean_relationship(mean_b),
        )
    positions = np.arange(len(values), dtype=np.float64)
    for position, value in zip(
        positions,
        values,
        strict=True,
    ):
        q05, median, q95, inlet_start, inlet_end = value
        axis.vlines(
            position,
            q05,
            q95,
            color=RELATION_PHASE_COLORS["Bed median (q05-q95 range)"],
            linewidth=RELATION_RANGE_LINE_WIDTH,
            alpha=0.28,
        )
        for offset, phase_value, marker, phase_color in (
            (
                0.0,
                median,
                "o",
                RELATION_PHASE_COLORS["Bed median (q05-q95 range)"],
            ),
            (-0.07, inlet_start, "^", RELATION_PHASE_COLORS["Inlet start"]),
            (
                0.07,
                inlet_end,
                "v",
                RELATION_PHASE_COLORS["Inlet startup end"],
            ),
        ):
            axis.scatter(
                position + offset,
                phase_value,
                color=phase_color,
                edgecolors="none",
                linewidths=0.0,
                marker=marker,
                s=RELATION_MARKER_SIZE,
                zorder=3,
            )
    phase_handles = tuple(
        Line2D(
            (),
            (),
            color="none",
            marker=marker,
            markerfacecolor=RELATION_PHASE_COLORS[label],
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=float(np.sqrt(RELATION_MARKER_SIZE)),
            label=label,
        )
        for label, marker in RELATION_PHASE_MARKERS
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(
        "Relative humidity [1]",
        fontsize=layout.MAP_LAYOUT.label_size,
    )
    axis.set_title("Inlet vs bed-equilibrium RH", fontsize=layout.MAP_LAYOUT.axis_title_size)
    axis.tick_params(labelsize=layout.MAP_LAYOUT.tick_size)
    axis.grid(axis="y", alpha=0.22)
    legend_axis.legend(
        handles=phase_handles,
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        borderaxespad=0.0,
        mode="expand",
        fontsize=layout.MAP_LAYOUT.legend_size,
        ncols=len(phase_handles),
    )
