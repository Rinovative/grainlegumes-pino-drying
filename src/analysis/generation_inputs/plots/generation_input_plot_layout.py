"""Define the schedule and table figure grids and the field scales and drawing of generation-input maps.

Responsibilities:
  - Create the bounded generation-input composite figure grid of the schedule figure
  - Draw one structured-grid field on its shared discrete scale
  - Centralize independent, locked, and signed physical scales: log-positive principal permeabilities, signed Kxy, and
    round state bands for every other field

Design principles:
  - Map geometry, headers and colour bars belong to the shared fixed-inch map grid; this module only fills it
  - Presentation helpers never clip displayed scientific values; fields reach maps only through the shared display
    conversion (relative humidity in %), and labelled in the active label style with one unit rule for axes and colour bars

This module does NOT:
  - Load generation inputs, derive fields, or select cases
  - Plan map grids, label colour bars, or import evaluation internals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec


@dataclass(frozen=True, slots=True)
class GenerationInputFigureLayout:
    """Define one centrally configured generation-input notebook layout."""

    notebook_width: float = shared_layout.MAP_MAX_WIDTH_IN
    column_width: float = 5.8  # one column of a schedule figure; maps size themselves on the shared map grid
    distribution_row_height: float = 2.75
    schedule_row_height: float = 2.75
    horizontal_spacing: float = 0.135
    vertical_spacing: float = 0.11
    title_size: float = 15.0
    axis_title_size: float = 12.5
    label_size: float = 11.25
    tick_size: float = 10.0
    legend_size: float = 9.0
    table_size: float = 8.0
    line_width: float = 1.4

    def figure_width(self, columns: int) -> float:
        """Return a positive notebook-bounded figure width of ``columns`` schedule or table columns."""
        if columns <= 0:
            message = "Generation-input figures require at least one column."
            raise ValueError(message)
        return min(self.notebook_width, self.column_width * columns)

    def figure_height(self, row_heights: Sequence[float]) -> float:
        """Return the central height for one explicit row composition."""
        heights = tuple(float(value) for value in row_heights)
        if not heights or not np.isfinite(heights).all() or any(value <= 0.0 for value in heights):
            message = "Generation-input figure row heights must be positive and finite."
            raise ValueError(message)
        return float(sum(heights) + self.vertical_spacing * (len(heights) - 1))


FIGURE_LAYOUT: Final = GenerationInputFigureLayout()
DATASET_A_COLOR: Final = "tab:blue"
DATASET_B_COLOR: Final = "tab:orange"
DATASET_MEAN_COLOR: Final = "0.30"


def figure_grid(
    *,
    columns: int,
    row_heights: Sequence[float],
) -> tuple[Figure, GridSpec]:
    """Create one constrained notebook-width figure and explicit GridSpec."""
    heights = tuple(float(value) for value in row_heights)
    figure = plt.figure(
        figsize=(FIGURE_LAYOUT.figure_width(columns), FIGURE_LAYOUT.figure_height(heights)),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        len(heights),
        columns,
        height_ratios=heights,
        wspace=FIGURE_LAYOUT.horizontal_spacing,
        hspace=FIGURE_LAYOUT.vertical_spacing,
    )
    return figure, grid


def signed_norm(*values: np.ndarray) -> field_maps.FieldMapDefinition:
    """Use the shared zero-centred round signed-difference scale."""
    return field_maps.build_field_map(
        np.concatenate(tuple(np.ravel(value) for value in values)),
        semantic=field_maps.FieldMapSemantic.SIGNED_ERROR,
    )


def physical_norm(quantity: str, *values: np.ndarray) -> field_maps.FieldMapDefinition:
    """
    Build one shared discrete physical scale from complete source arrays.

    The principal permeabilities are log-positive and Kxy is signed. Every other field is a state on the round state bands
    (``field_maps.state_boundaries``), so a near-uniform one is widened. A state's bar, Kxy's included, carries the power of
    ten of its values' size (``field_maps.state_exponent``), none for values between 10^-3 and 10^4.
    """
    combined = np.concatenate(tuple(np.ravel(value) for value in values))
    semantic, locked = field_maps.FieldMapSemantic.LINEAR_POSITIVE, None
    if quantity == "Kxy":
        semantic = field_maps.FieldMapSemantic.LINEAR_SIGNED
    elif quantity in {"Kxx", "Kyy", "K_min", "K_max"}:
        semantic = field_maps.FieldMapSemantic.LOG_POSITIVE
    elif np.ptp(combined) > 0.0:  # a constant keeps its truthful one-value scale
        locked = field_maps.state_boundaries(float(np.min(combined)), float(np.max(combined)))
    return field_maps.build_field_map(combined, semantic=semantic, field=quantity, unit=diagnostics.FIELD_UNITS[quantity], locked_boundaries=locked)


def comparison_norms(
    quantity: str,
    first: np.ndarray,
    second: np.ndarray,
    *,
    lock_scale: bool,
) -> tuple[field_maps.FieldMapDefinition, field_maps.FieldMapDefinition]:
    """
    Return independent or exactly shared A/B physical normalizations.

    Signed Kxy fields remain zero-centred. Locking uses the same normalization
    object for both cases; unlocked maps use separate data-driven objects.
    """
    if lock_scale:
        shared = physical_norm(quantity, first, second)
        return shared, shared
    return physical_norm(quantity, first), physical_norm(quantity, second)


def draw_field(
    axis: Axes,
    record: diagnostics.GenerationInputDiagnostics,
    values: np.ndarray,
    scale: field_maps.FieldMapDefinition,
    *,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> QuadMesh:
    """Draw one exact structured-grid array of ``record`` on ``scale``, sampled for display only."""
    indices = display_grid.display_indices(values.shape, display_spatial_stride)
    return axis.pcolormesh(
        display_grid.display_sample(record.fields["x"], indices),
        display_grid.display_sample(record.fields["y"], indices),
        display_grid.display_sample(values, indices),
        shading="auto",
        **scale.pcolormesh_kwargs(),
    )


def display_field(record: diagnostics.GenerationInputDiagnostics, quantity: str) -> np.ndarray:
    """Return one generated field read-only on its display scale, such as equilibrium relative humidity in % RH."""
    stored = record.fields[quantity]
    values = np.asarray(diagnostics.display_value(quantity, stored, diagnostics.FIELD_UNITS[quantity]))
    if values is not stored:  # a converted copy; the stored evidence array keeps its own flags
        values.setflags(write=False)
    return values


def quantity_label(quantity: str) -> str:
    """Return one quantity's label in the active style; a quantity without a shared symbol keeps its concise name."""
    if field_labels.has_declared_field_metadata(quantity):
        return field_labels.field_label(quantity, mathtext=True)
    return diagnostics.field_name(quantity)


def quantity_colorbar_label(quantity: str, *, difference: bool = False) -> str:
    """Return only the display unit of a map's values or differences, as each map header owns the quantity; one unit rule for axes and bars."""
    unit = diagnostics.FIELD_UNITS[quantity]
    return f"[{field_labels.display_unit(unit, field=quantity, quantity_kind='difference' if difference else 'absolute')}]" if unit else ""


def quantity_axis_label(quantity: str, exponent: int = 0) -> str:
    """Return one concise physical quantity and display unit label; a nonzero ``exponent`` is written into the unit bracket as on the colour bars."""
    unit = quantity_colorbar_label(quantity)
    if exponent and unit:
        unit = shared_layout.label_with_exponent(unit, exponent)
    return f"{quantity_label(quantity)} {unit}".rstrip()


def histogram_exponent(*values: np.ndarray) -> int:
    """Return the power of ten the value axis of a histogram of ``values`` is written in (0 for values between 10^-3 and 10^4)."""
    combined = np.concatenate(tuple(np.ravel(value) for value in values))
    return field_maps.state_exponent(float(np.min(combined)), float(np.max(combined)))
