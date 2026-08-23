"""
analysis_ui_plot_layout.py

Define shared notebook-scale physical-map and colorbar presentation.

Responsibilities:
  - Create equal-sized physical-map subplot grids
  - Attach proportionate axes-coupled colorbars
  - Reserve a dedicated legend column without constrained-layout coupling
  - Normalize finite constant and varying physical fields predictably

Design principles:
  - Layout never changes scientific arrays or their physical coordinates
  - One immutable sizing contract serves EDA and Evaluation snapshot maps
  - Every colorbar remains explicitly bound to its map axis

This module does NOT:
  - Choose scientific fields, units, colormaps, or comparison semantics
  - Load data, construct notebook controls, or manage figure export
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class MapLayoutContract:
    """Define one consistent notebook-scale physical-map layout."""

    column_width: float = 5.8
    row_height: float = 3.6
    column_spacing: float = 0.18
    row_spacing: float = 0.10
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.025
    label_size: float = 11.25
    tick_size: float = 10.0

    def figure_size(self, *, rows: int, columns: int) -> tuple[float, float]:
        """Return the shared positive map-grid dimensions."""
        if rows <= 0 or columns <= 0:
            message = "Physical map grids require positive row and column counts."
            raise ValueError(message)
        return self.column_width * columns, self.row_height * rows


MAP_LAYOUT = MapLayoutContract()
_MINIMUM_GRID_TOP = 0.10


def linear_norm(*values: np.ndarray) -> Normalize:
    """Return one finite linear normalization with stable constant-field limits."""
    if not values:
        message = "Map normalization requires at least one array."
        raise ValueError(message)
    combined = np.concatenate(tuple(np.ravel(np.asarray(value, dtype=np.float64)) for value in values))
    if not np.isfinite(combined).all():
        message = "Map normalization requires finite physical values."
        raise ValueError(message)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if np.isclose(lower, upper, rtol=0.0, atol=np.finfo(np.float64).eps):
        padding = max(abs(lower), 1.0) * 1.0e-12
        lower -= padding
        upper += padding
    return Normalize(vmin=lower, vmax=upper)


def signed_norm(*values: np.ndarray) -> TwoSlopeNorm:
    """Return one symmetric finite normalization centered exactly at zero."""
    if not values:
        message = "Signed map normalization requires at least one array."
        raise ValueError(message)
    combined = np.concatenate(tuple(np.ravel(np.asarray(value, dtype=np.float64)) for value in values))
    if not np.isfinite(combined).all():
        message = "Signed map normalization requires finite physical values."
        raise ValueError(message)
    limit = max(
        float(np.max(np.abs(combined))),
        float(np.finfo(np.float64).eps),
    )
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def map_subplots(
    *,
    rows: int,
    columns: int,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Create one equal-column map grid under the shared layout contract."""
    owned = {"figsize", "squeeze", "layout", "gridspec_kw"}
    overlap = owned.intersection(kwargs)
    if overlap:
        message = f"Shared map grids own figure size, squeezing, layout, and spacing; received {sorted(overlap)}."
        raise ValueError(message)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=MAP_LAYOUT.figure_size(rows=rows, columns=columns),
        squeeze=False,
        layout=None,
        **kwargs,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.93,
        bottom=0.09,
        top=0.91,
        wspace=MAP_LAYOUT.column_spacing + 0.18,
        hspace=MAP_LAYOUT.row_spacing + 0.18,
    )
    return figure, axes


def subplots_with_legend_column(
    *,
    rows: int,
    columns: int,
    column_width: float = 4.8,
    row_height: float = 3.6,
    legend_width: float = 2.4,
    sharex: bool = False,
    top: float = 0.90,
    hspace: float = 0.34,
) -> tuple[Figure, np.ndarray, Axes]:
    """Create a manually spaced subplot grid with one reserved legend rail."""
    if rows <= 0 or columns <= 0:
        message = "Legend-column grids require positive row and column counts."
        raise ValueError(message)
    if min(column_width, row_height, legend_width) <= 0.0:
        message = "Legend-column grid dimensions must be positive."
        raise ValueError(message)
    if not _MINIMUM_GRID_TOP < top < 1.0 or hspace < 0.0:
        message = "Legend-column grid top and row spacing are invalid."
        raise ValueError(message)
    figure = plt.figure(
        figsize=(column_width * columns + legend_width, row_height * rows),
        layout=None,
    )
    grid = figure.add_gridspec(
        rows,
        columns + 1,
        width_ratios=(*([1.0] * columns), legend_width / column_width),
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=top,
        wspace=0.34,
        hspace=hspace,
    )
    axes = np.empty((rows, columns), dtype=object)
    for row in range(rows):
        for column in range(columns):
            shared_axis = axes[0, column] if sharex and row > 0 else None
            axes[row, column] = figure.add_subplot(
                grid[row, column],
                sharex=shared_axis,
            )
    legend_axis = figure.add_subplot(grid[:, -1])
    legend_axis.set_axis_off()
    return figure, axes, legend_axis


def add_channel_row_label(
    axis: Axes,
    label: str,
    *,
    axis_x: float = -0.28,
    figure_x: float | None = None,
) -> None:
    """Place one shared formula-and-unit heading at the far-left row margin."""
    if not isinstance(label, str) or not label.strip():
        message = "Channel row labels must be non-empty text."
        raise ValueError(message)
    if figure_x is None:
        if isinstance(axis_x, bool) or not np.isfinite(float(axis_x)):
            message = "Axes-coordinate channel labels require one finite x-position."
            raise ValueError(message)
        location = (float(axis_x), 0.5)
        coordinates: str | Any = "axes fraction"
    else:
        if isinstance(figure_x, bool) or not np.isfinite(float(figure_x)) or not 0.0 <= float(figure_x) <= 1.0:
            message = "Figure-coordinate channel labels require a finite x-position in [0, 1]."
            raise ValueError(message)
        position = axis.get_position()
        location = (float(figure_x), (position.y0 + position.y1) / 2.0)
        coordinates = axis.figure.transFigure
    annotation = axis.annotate(
        label,
        xy=location,
        xycoords=coordinates,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
        annotation_clip=False,
    )
    annotation.set_gid("channel-row-label")


def set_suptitle_over_axes(
    figure: Figure,
    title: str,
    axes: Iterable[Axes],
    **kwargs: Any,
) -> Any:
    """Centre one title over explicit scientific axes, excluding side rails."""
    scientific_axes = tuple(axis for axis in axes if axis.get_visible())
    if not scientific_axes:
        message = "A scientific-block title requires at least one visible axis."
        raise ValueError(message)
    if not isinstance(title, str) or not title.strip():
        message = "Scientific-block titles must be non-empty text."
        raise ValueError(message)
    left = min(axis.get_position().x0 for axis in scientific_axes)
    right = max(axis.get_position().x1 for axis in scientific_axes)
    return figure.suptitle(title, x=(left + right) / 2.0, **kwargs)


def configure_bottom_occupied_row_xlabels(
    axes: Iterable[Axes],
    *,
    columns: int,
    label: str | None = None,
    hide_upper_tick_labels: bool,
) -> None:
    """Keep x-axis text on the bottom occupied row of one row-major grid."""
    occupied = tuple(axes)
    if not occupied:
        message = "Bottom-row x-axis formatting requires at least one occupied axis."
        raise ValueError(message)
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        message = "Bottom-row x-axis formatting requires a positive column count."
        raise ValueError(message)
    if label is not None and (not isinstance(label, str) or not label.strip()):
        message = "Bottom-row x-axis labels must be non-empty text when supplied."
        raise ValueError(message)
    if not isinstance(hide_upper_tick_labels, bool):
        message = "Bottom-row x-axis tick visibility must be boolean."
        raise TypeError(message)
    bottom_row = (len(occupied) - 1) // columns
    for index, axis in enumerate(occupied):
        is_bottom = index // columns == bottom_row
        if is_bottom:
            if label is not None:
                axis.set_xlabel(label)
        else:
            axis.set_xlabel("")
        if hide_upper_tick_labels:
            axis.tick_params(axis="x", labelbottom=is_bottom)


def add_map_colorbar(
    figure: Figure,
    image: ScalarMappable,
    map_axis: Axes,
    *,
    label: str = "",
) -> Colorbar:
    """Attach one proportionate colorbar and record its map-axis ownership."""
    divider = make_axes_locatable(map_axis)
    colorbar_axis = divider.append_axes(
        "right",
        size=f"{100.0 * MAP_LAYOUT.colorbar_fraction:g}%",
        pad=f"{100.0 * MAP_LAYOUT.colorbar_pad:g}%",
    )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(label, fontsize=MAP_LAYOUT.label_size)
    colorbar.ax.tick_params(labelsize=MAP_LAYOUT.tick_size)
    return colorbar


def apply_map_grid_axis_labels(
    axes: Iterable[Iterable[Axes]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    """Place coordinate labels and tick labels only on visible grid edges."""
    rows = tuple(tuple(row) for row in axes)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        message = "Map-axis grids must be non-empty and rectangular."
        raise ValueError(message)
    bottom = len(rows) - 1
    for row_index, row in enumerate(rows):
        for column_index, axis in enumerate(row):
            if not axis.get_visible() or not axis.axison:
                continue
            is_bottom = row_index == bottom
            is_left = column_index == 0
            axis.set_xlabel(x_label if is_bottom else "")
            axis.set_ylabel(y_label if is_left else "")
            axis.tick_params(
                axis="x",
                labelbottom=is_bottom,
                labelsize=MAP_LAYOUT.tick_size,
            )
            axis.tick_params(
                axis="y",
                labelleft=is_left,
                labelsize=MAP_LAYOUT.tick_size,
            )
