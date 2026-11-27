"""Compose transient moisture, sorption, and inlet-bed A/B diagnostics.

Responsibilities:
  - Compare initial moisture and equilibrium-RH maps as A/B/B-minus-A rows
  - Retain separate A/B distributions with dataset-mean markers
  - Compare inlet startup RH with case and empirical bed-equilibrium summaries on one relative-humidity axis, its phase
    key inside its panel

Design principles:
  - Equilibrium RH comes only from the canonical per-case inverse-Oswin result
  - Dataset reference values average per-case nonlinear diagnostics
  - Moisture and humidity remain separate physical quantities; the inlet and the bed share the relative-humidity axis

This module does NOT:
  - Render single-case alternatives, copy sorption equations, or resample fields
  - Display dataset-mean maps or alter generation acceptance
"""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from matplotlib.lines import Line2D

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

from . import generation_input_plot_layout as layout
from . import generation_input_plot_spatial as spatial

if TYPE_CHECKING:
    from typing import SupportsFloat

    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.presentation import analysis_display_grid as display_grid

RELATION_MARKER_SIZE: Final = 38.0
RELATION_RANGE_LINE_WIDTH: Final = 1.5
RELATION_RANGE_CAP_SIZE: Final = 5.0
RELATION_LABEL_PAD_PT: Final = 8.0  # the relative-humidity label clear of its tick labels
RELATION_PHASE_COLORS: Final = MappingProxyType(
    {
        "Bed median (q05\N{EN DASH}q95)": "tab:purple",
        "Inlet start": "tab:green",
        "Inlet startup end": "tab:red",
    }
)
RELATION_PHASE_MARKERS: Final = (
    ("Bed median (q05\N{EN DASH}q95)", "o"),
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
    display_spatial_stride: display_grid.DisplayStride = "auto",
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
        display_spatial_stride=display_spatial_stride,
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
    return _relative_humidity_display(
        (
            float(cast("SupportsFloat", statistics["q05"])),
            float(cast("SupportsFloat", statistics["median"])),
            float(cast("SupportsFloat", statistics["q95"])),
        ),
        (inlet.start, inlet.end),
    )


def _relative_humidity_display(bed: tuple[float, float, float], inlet: tuple[float, float]) -> tuple[float, float, float, float, float]:
    """Return bed q05/median/q95 and inlet startup endpoints in the displayed % RH."""
    unit = diagnostics.FIELD_UNITS["phi_eq"]  # bed and inlet relative humidity share the stored fraction unit
    q05, median, q95 = (float(value) for value in diagnostics.display_value("phi_eq", np.asarray(bed), unit))
    start, end = (float(value) for value in diagnostics.display_value("phi_in_bc", np.asarray(inlet), unit))
    return q05, median, q95, start, end


def _mean_relationship(
    dataset: diagnostics.DatasetDiagnostics,
) -> tuple[float, float, float, float, float]:
    """Return empirical means of per-case bed and inlet scalar summaries."""
    return _relative_humidity_display(
        (
            dataset.field_summary_means[("phi_eq", "q05")],
            dataset.field_summary_means[("phi_eq", "median")],
            dataset.field_summary_means[("phi_eq", "q95")],
        ),
        (float(dataset.boundary_means["phi_in_bc start"]), float(dataset.boundary_means["phi_in_bc startup end"])),
    )


def _draw_inlet_bed_comparison(
    axis: Axes,
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
    bed_color, start_color, end_color = RELATION_PHASE_COLORS.values()
    for position, value in zip(
        positions,
        values,
        strict=True,
    ):
        q05, median, q95, inlet_start, inlet_end = value
        # Opaque with end caps: a bed's q05-q95 spread is a few percent of the axis, which a faint bar hid behind the median marker.
        axis.errorbar(
            position,
            median,
            yerr=np.array(((median - q05,), (q95 - median,))),
            fmt="none",
            ecolor=bed_color,
            elinewidth=RELATION_RANGE_LINE_WIDTH,
            capsize=RELATION_RANGE_CAP_SIZE,
            zorder=2,
        )
        for offset, phase_value, marker, phase_color in (
            (0.0, median, "o", bed_color),
            (-0.07, inlet_start, "^", start_color),
            (0.07, inlet_end, "v", end_color),
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
    # One axis for the inlet and the bed: it is labelled as relative humidity, not as either of them.
    axis.set_ylabel(
        field_labels.field_label_with_unit("phi", diagnostics.FIELD_UNITS["phi_eq"], mathtext=True),
        fontsize=layout.FIGURE_LAYOUT.label_size,
        labelpad=RELATION_LABEL_PAD_PT,
    )
    axis.set_title("Inlet vs bed-equilibrium RH", fontsize=layout.FIGURE_LAYOUT.axis_title_size)
    axis.tick_params(labelsize=layout.FIGURE_LAYOUT.tick_size)
    axis.grid(axis="y", alpha=0.22)
    figure = axis.get_figure(root=True)
    if figure is None:
        msg = "The inlet-bed comparison needs an axis on a figure."
        raise RuntimeError(msg)
    panel = np.empty((1, 1), dtype=object)
    panel[0, 0] = axis
    figure_layout.add_legend(figure, panel, phase_handles, [label for label, _marker in RELATION_PHASE_MARKERS])  # the panel's own key stays inside
