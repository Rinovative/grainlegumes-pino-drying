"""Define shared notebook-scale physical-map and colorbar presentation.

Responsibilities:
  - Create equal-sized, width-capped physical-map subplot grids under constrained layout
  - Plan fixed-inch map grids: equal maps with the domain's aspect, colour bars as tall as their maps, fixed gaps
  - Assemble a fixed map grid's map axes and hidden colour-bar slots in one place
  - Write map-grid column headers with notes, row labels and one figure title once each
  - Attach proportionate axes-coupled colorbars or fill reserved fixed-inch colour-bar slots
  - Reserve a dedicated legend column inside the same constrained layout
  - Normalize finite constant and varying physical fields predictably
  - Choose round discrete colour-scale boundaries: steps of 1, 2 or 5 times a power of ten
  - Write a power of ten into the unit bracket of a colour-bar label

Design principles:
  - Layout never changes scientific arrays or their physical coordinates
  - One immutable sizing contract serves EDA, Evaluation and generation-input maps
  - Every colorbar remains explicitly bound to its map axis
  - A fixed-inch grid's figure size is a pure function of rows, columns and domain and uses no layout engine,
    so a reused figure keeps its pixel size and axes positions

This module does NOT:
  - Choose scientific fields, units, colormaps, or comparison semantics
  - Load data, construct notebook controls, or manage figure export
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import Size, make_axes_locatable  # type: ignore[import-untyped]

from src.analysis.presentation import analysis_field_maps as field_maps

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

    from src.analysis.presentation.analysis_field_maps import FieldMapExtend

MAP_WIDTH_IN = 4.2  # target width of one drawn map
MAP_MAX_WIDTH_IN = 17.4  # figure width cap: all maps shrink uniformly to stay under it
COLORBAR_WIDTH_IN = 0.14
COLORBAR_PAD_IN = 0.08  # map to its colour bar
GAP_H_IN = 1.10  # colour bar to the next map: the bar's tick labels and unit label sit here
GAP_V_IN = 0.40  # between rows: x tick labels stand on the bottom row only
MARGIN_LEFT_IN = 1.50  # row labels, y tick labels and "y [m]"
MARGIN_RIGHT_IN = 1.10  # tick labels and unit label of the last colour bar
MARGIN_TOP_IN = 0.95  # figure title and column headers with one note line
MARGIN_BOTTOM_IN = 0.75  # x tick labels and "x [m]"
_TITLE_FROM_TOP_IN = 0.12
_ROW_LABEL_OFFSET_PT = 62.0  # left of the first column's y tick labels and "y [m]"
_HEADER_PAD_PT = 6.0  # Matplotlib's default title pad
_HEADER_NOTE_LINE_PT = 14.0  # one 10 pt note line under a header
_TICK_SIZE_PT = 9.0
_LABEL_SIZE_PT = 10.0
_HEADER_SIZE_PT = 11.0
_TITLE_SIZE_PT = 12.0
_NOTE_SIZE_PT = 10.0  # note lines carry per-map metrics, readable in a 60-dpi GIF
_NOTE_COLOUR = "0.20"
_BAND_MANTISSAS = (1.0, 2.0, 2.5, 5.0)  # round band widths: these times a power of ten
_PREFERRED_BANDS = 11  # the band count that wins a tie in the covered span
_MAX_STEP_DECIMALS = 15
# Relative: far below the gap between round steps, above the roundoff of a step taken as a difference of large boundaries.
_STEP_TOLERANCE = 1e-6
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_UNIT_BRACKET = re.compile(r"\[([^\]]*)\]\s*$")
_MATH_SPAN = re.compile(r"\$([^$]+)\$")


@dataclass(frozen=True, slots=True)
class MapLayoutContract:
    """Define one consistent notebook-scale physical-map layout."""

    column_width: float = 5.8
    row_height: float = 3.6
    maximum_width: float = 17.4
    column_spacing: float = 0.18
    row_spacing: float = 0.10
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.025
    label_size: float = 11.25
    tick_size: float = 10.0

    def figure_size(self, *, rows: int, columns: int) -> tuple[float, float]:
        """Return the shared positive map-grid dimensions, shrinking columns to the width cap."""
        if rows <= 0 or columns <= 0:
            message = "Physical map grids require positive row and column counts."
            raise ValueError(message)
        return min(self.column_width * columns, self.maximum_width), self.row_height * rows


MAP_LAYOUT = MapLayoutContract()


@dataclass(frozen=True, slots=True)
class MapGeometry:
    """Hold the resolved inch geometry of one fixed map grid with row labels and a figure title."""

    rows: int
    columns: int
    map_width: float
    map_height: float

    @property
    def cell_width(self) -> float:
        """Return the width of one map plus its colour-bar pad and slot."""
        return self.map_width + COLORBAR_PAD_IN + COLORBAR_WIDTH_IN

    @property
    def figure_width(self) -> float:
        """Return the figure width: margins, cells and the gaps between them."""
        return MARGIN_LEFT_IN + MARGIN_RIGHT_IN + self.columns * self.cell_width + (self.columns - 1) * GAP_H_IN

    @property
    def figure_height(self) -> float:
        """Return the figure height: margins, maps and the gaps between rows."""
        return MARGIN_TOP_IN + MARGIN_BOTTOM_IN + self.rows * self.map_height + (self.rows - 1) * GAP_V_IN


@contextmanager
def eda_typography() -> Iterator[None]:
    """Moderately enlarge EDA figure text for an opt-in plotting scope."""
    with plt.rc_context(
        {
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    ):
        yield


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


def step_decimals(step: float) -> int:
    """Return the decimals the round ``step`` needs exactly (0.25 gives 2, 2.5 gives 1, 25 gives 0)."""
    for decimals in range(_MAX_STEP_DECIMALS):
        if abs(round(step, decimals) - step) <= _STEP_TOLERANCE * step:
            return decimals
    return _MAX_STEP_DECIMALS


def _best_band(width: float, bands_from: Callable[[float], int | None], max_decimals: int | None) -> tuple[float, int] | None:
    """Return the (step, band count) of the round candidate covering the least span, ties to the count nearest 11, then the larger step."""
    base = math.floor(math.log10(width / field_maps.MAX_BANDS))
    best: tuple[tuple[float, int, float], float, int] | None = None
    for power in range(base, base + 3):
        for mantissa in _BAND_MANTISSAS:
            step = mantissa * 10.0**power
            count = None if max_decimals is not None and step_decimals(step) > max_decimals else bands_from(step)
            if count is None:
                continue
            key = (round(count * step / width, 9), abs(count - _PREFERRED_BANDS), -step)
            if best is None or key < best[0]:
                best = (key, step, count)
    return None if best is None else best[1:]


def rounded_boundaries(lower: float, upper: float, *, max_decimals: int | None = None) -> np.ndarray | None:
    """
    Return the round boundaries of 9 to 13 equal bands from a multiple of the step at or below ``lower`` that reach ``upper``.

    The step is m times a power of ten with m in 1, 2, 2.5 or 5. Among the candidates that reach ``upper`` the one with the
    smallest covered span wins; a tie goes to the band count nearest 11, then to the larger step. With ``max_decimals``
    candidates whose step needs more decimals are skipped, and a range whose best step is finer than the last decimal is too
    narrow for it: ``None``, as for ``upper`` not above ``lower`` (or a width that underflows or overflows). The boundaries
    are the doubles nearest their decimal values (``_multiples``).
    """
    width = upper - lower
    if not (math.isfinite(width) and width / field_maps.MAX_BANDS > 0.0):
        return None

    def bands_from(step: float) -> int | None:
        first = _first_multiple(lower, step)
        count = max(field_maps.MIN_BANDS, math.ceil(upper / step - first - 1e-9))
        count += 1 if _multiples(first + count, step) < upper else 0  # the top itself is checked: a rounding step below ``upper`` would cut it short
        return count if count <= field_maps.MAX_BANDS else None

    found = _best_band(width, bands_from, None)
    if found is not None and max_decimals is not None and step_decimals(found[0]) > max_decimals:
        found = None if found[0] < 10.0**-max_decimals * (1.0 - 1e-9) else _best_band(width, bands_from, max_decimals)
    if found is None:
        return None
    step, count = found
    return _multiples(_first_multiple(lower, step) + np.arange(count + 1), step)


def _multiples(indices: np.ndarray | int, step: float) -> np.ndarray:
    """Return ``indices`` times the round ``step`` as the doubles nearest their decimal values (3 times 0.1 gives 0.3)."""
    exponent = math.floor(math.log10(step)) - 1  # m times 10^k with m in 1, 2, 2.5 or 5 is a whole number times 10^(k-1)
    whole = np.asarray(indices, dtype=np.float64) * round(step / 10.0**exponent)
    return whole / 10.0**-exponent if exponent < 0 else whole * 10.0**exponent


def _first_multiple(lower: float, step: float) -> int:
    """Return the index of the largest multiple of ``step`` at or below ``lower`` (a quotient rounded up to a whole number steps back)."""
    first = math.floor(lower / step + 1e-9)
    return first - 1 if _multiples(first, step) > lower else first


def symmetric_boundaries(reach: float) -> np.ndarray | None:
    """
    Return round boundaries of an odd number (9, 11 or 13) of equal bands of width w with ``count / 2 * w`` at least ``reach``.

    The middle band is centred on 0. The candidates and the choice among them are those of ``rounded_boundaries``.
    ``None`` for a reach that is not positive (or underflows), or whose span overflows.
    """
    if not (math.isfinite(2.0 * reach) and reach / field_maps.MAX_BANDS > 0.0):
        return None

    def bands_from(step: float) -> int | None:
        count = max(field_maps.MIN_BANDS, math.ceil(2.0 * reach / step - 1e-9))
        count += 1 - count % 2
        count += 2 if count / 2.0 * step < reach else 0
        return count if count <= field_maps.MAX_BANDS else None

    found = _best_band(2.0 * reach, bands_from, None)
    if found is None:
        return None
    step, count = found
    return (np.arange(count + 1) - count / 2.0) * step


def map_subplots(
    *,
    rows: int,
    columns: int,
    figure: Figure | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """
    Create one equal-column map grid under the shared layout contract.

    A given ``figure`` (created here) is cleared and reused with its current size and
    layout engine, so animation frames keep identical pixel dimensions.
    """
    owned = {"figsize", "squeeze", "layout", "gridspec_kw"}
    overlap = owned.intersection(kwargs)
    if overlap:
        message = f"Shared map grids own figure size, squeezing, layout, and spacing; received {sorted(overlap)}."
        raise ValueError(message)
    spacing = {"wspace": MAP_LAYOUT.column_spacing, "hspace": MAP_LAYOUT.row_spacing}
    if figure is not None:
        figure.clear()
        return figure, figure.subplots(rows, columns, squeeze=False, gridspec_kw=spacing, **kwargs)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=MAP_LAYOUT.figure_size(rows=rows, columns=columns),
        squeeze=False,
        layout="compressed",
        gridspec_kw=spacing,
        **kwargs,
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
    hspace: float = 0.34,
) -> tuple[Figure, np.ndarray, Axes]:
    """Create a constrained subplot grid with one reserved legend rail."""
    if rows <= 0 or columns <= 0:
        message = "Legend-column grids require positive row and column counts."
        raise ValueError(message)
    if min(column_width, row_height, legend_width) <= 0.0:
        message = "Legend-column grid dimensions must be positive."
        raise ValueError(message)
    if hspace < 0.0:
        message = "Legend-column grid row spacing is invalid."
        raise ValueError(message)
    figure = plt.figure(
        figsize=(column_width * columns + legend_width, row_height * rows),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        rows,
        columns + 1,
        width_ratios=(*([1.0] * columns), legend_width / column_width),
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
    """
    Title one figure over its scientific axes.

    A manually laid-out figure centres the title over the given axes, excluding
    side rails. A constrained figure positions the title itself, so ``y`` is
    dropped and the title is centred on the figure.
    """
    scientific_axes = tuple(axis for axis in axes if axis.get_visible())
    if not scientific_axes:
        message = "A scientific-block title requires at least one visible axis."
        raise ValueError(message)
    if not isinstance(title, str) or not title.strip():
        message = "Scientific-block titles must be non-empty text."
        raise ValueError(message)
    if figure.get_layout_engine() is not None:
        kwargs.pop("y", None)
        return figure.suptitle(title, **kwargs)
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
    colorbar = figure.colorbar(image, ax=map_axis, fraction=MAP_LAYOUT.colorbar_fraction, pad=MAP_LAYOUT.colorbar_pad)
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


def label_with_exponent(label: str, exponent: int) -> str:
    """
    Write the power of ten into the unit bracket that ends ``label``: ``[kg/m³]`` becomes ``[10⁻³ kg/m³]``, ``[-]`` becomes ``[10⁻³]``.

    A ``label`` without a unit bracket raises ``ValueError``; the exponent 0 returns it unchanged.
    """
    unit = _UNIT_BRACKET.search(label)
    if unit is None:
        message = f"A colour-bar label must end in a unit bracket such as [K] or [-], got {label!r}."
        raise ValueError(message)
    if exponent == 0:
        return label
    power = power_of_ten(exponent)
    return f"{label[: unit.start()]}[{power if unit.group(1) in {'', '-'} else f'{power} {unit.group(1)}'}]"


def power_of_ten(exponent: int) -> str:
    """Return ten to ``exponent`` with a superscript exponent, as a bar's unit writes it (-4 gives 10⁻⁴)."""
    return f"10{str(exponent).translate(_SUPERSCRIPT)}"


def _bold_label(text: str) -> str:
    """Return ``text`` with its mathtext spans set bold; ``fontweight`` alone never bolds mathtext."""
    return _MATH_SPAN.sub(lambda match: "$\\mathbf{" + match.group(1).replace("\\mathrm{", "\\mathbf{") + "}$", text)


def plan_geometry(rows: int, columns: int, domain: tuple[float, float]) -> MapGeometry:
    """
    Return the fixed-inch geometry of one map grid with row labels and a figure title.

    Every map has the same width (the target width, shrunk uniformly when the columns would exceed the width
    cap) and the physical aspect of ``domain`` (width, height); the figure size follows from rows, columns and
    domain alone. Non-positive counts or a non-finite or non-positive domain raise ``ValueError``.
    """
    if rows <= 0 or columns <= 0:
        message = "Map grids need positive row and column counts."
        raise ValueError(message)
    width, height = (float(value) for value in domain)
    if not (np.isfinite(width) and np.isfinite(height) and width > 0.0 and height > 0.0):
        message = f"Map grids need a finite positive physical domain, got {domain!r}."
        raise ValueError(message)
    fixed = MARGIN_LEFT_IN + MARGIN_RIGHT_IN + columns * (COLORBAR_PAD_IN + COLORBAR_WIDTH_IN) + (columns - 1) * GAP_H_IN
    map_width = min(MAP_WIDTH_IN, (MAP_MAX_WIDTH_IN - fixed) / columns)
    return MapGeometry(rows, columns, map_width, map_width * height / width)


def map_figure(geometry: MapGeometry, *, figure: Figure | None = None) -> tuple[Figure, GridSpec]:
    """
    Return a figure of exactly the geometry's size without a layout engine and its grid of map cells.

    Each cell is one map plus its colour-bar slot wide and one map tall. A given ``figure`` (made here for the
    same geometry) is cleared and reused, so animation frames keep one pixel size and identical axes positions;
    a figure of another size or with a layout engine that would re-solve the positions raises ``ValueError``.
    """
    size = (geometry.figure_width, geometry.figure_height)
    if figure is None:
        figure = plt.figure(figsize=size, layout="none")
    elif figure.get_layout_engine() is not None or not np.allclose(figure.get_size_inches(), size):
        message = "A reused map figure needs the geometry's size and no layout engine."
        raise ValueError(message)
    else:
        figure.clear()
    grid = figure.add_gridspec(
        geometry.rows,
        geometry.columns,
        left=MARGIN_LEFT_IN / geometry.figure_width,
        right=1.0 - MARGIN_RIGHT_IN / geometry.figure_width,
        bottom=MARGIN_BOTTOM_IN / geometry.figure_height,
        top=1.0 - MARGIN_TOP_IN / geometry.figure_height,
        wspace=GAP_H_IN / geometry.cell_width,
        hspace=GAP_V_IN / geometry.map_height,
    )
    return figure, grid


def prepare_map_axes(axis: Axes, *, extent: tuple[float, float, float, float], bottom: bool, left: bool) -> Axes:
    """
    Turn one grid-cell axes into a map axes and return its hidden colour-bar slot.

    The map gets equal aspect on the physical ``extent`` (x0, x1, y0, y1); the slot of fixed width and pad sits
    right of it and is exactly as tall as the drawn map. Tick and coordinate labels stand only on the bottom
    row (``bottom``) and the left column (``left``).
    """
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    slot = make_axes_locatable(axis).append_axes("right", size=Size.Fixed(COLORBAR_WIDTH_IN), pad=Size.Fixed(COLORBAR_PAD_IN))
    slot.set_visible(False)
    axis.tick_params(labelbottom=bottom, labelleft=left, labelsize=_TICK_SIZE_PT)
    axis.set_xlabel("x [m]" if bottom else "")
    axis.set_ylabel("y [m]" if left else "")
    return slot


def map_grid(
    geometry: MapGeometry, *, extent: tuple[float, float, float, float], figure: Figure | None = None
) -> tuple[Figure, np.ndarray, np.ndarray]:
    """
    Return the figure of ``map_figure`` with its map axes and their hidden colour-bar slots, each [rows, columns].

    Every map is prepared on the physical ``extent`` (``prepare_map_axes``). The slots are added after all maps, so the
    figure lists the maps first in row-major order, then the slots.
    """
    figure, grid = map_figure(geometry, figure=figure)
    axes = np.empty((geometry.rows, geometry.columns), dtype=object)
    for row, column in np.ndindex(axes.shape):
        axes[row, column] = figure.add_subplot(grid[row, column])
    slots = np.empty_like(axes)
    for (row, column), axis in np.ndenumerate(axes):
        slots[row, column] = prepare_map_axes(axis, extent=extent, bottom=row == geometry.rows - 1, left=column == 0)
    return figure, axes, slots


def add_slot_colorbar(
    figure: Figure,
    image: ScalarMappable,
    slot: Axes,
    *,
    label: str,
    extend: FieldMapExtend = "neither",
) -> Colorbar:
    """
    Fill one reserved colour-bar slot: as tall as its map, width and pad fixed in inches, labelled with ``label``.

    ``label`` must end in a unit bracket such as ``[K]`` or ``[-]``, else ``ValueError``; ``extend`` draws an arrow
    only at the ends the caller found values beyond.
    """
    label_with_exponent(label, 0)
    slot.set_visible(True)
    colorbar = figure.colorbar(image, cax=slot, extend=extend)
    colorbar.ax.tick_params(labelsize=_TICK_SIZE_PT)
    colorbar.set_label(label, size=_LABEL_SIZE_PT)
    return colorbar


def add_map_note(axis: Axes, text: str) -> None:
    """Write one note line centred just above a map: the line under a column header, or the same line above a lower row's map."""
    axis.annotate(
        text,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, 3.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=_NOTE_SIZE_PT,
        color=_NOTE_COLOUR,
    )


def set_column_headers(axes: np.ndarray, titles: Sequence[str], notes: Sequence[str]) -> None:
    """
    Write each column header once above the first row; a non-empty note is a line under its header (``add_map_note``).

    When any column has a note, every header rises by one note line so that the headers share one baseline.
    """
    raised = any(notes)
    for column, text in enumerate(titles):
        axis = axes[0, column]
        axis.set_title(
            _bold_label(text),
            fontsize=_HEADER_SIZE_PT,
            fontweight="bold",
            pad=_HEADER_PAD_PT + (_HEADER_NOTE_LINE_PT if raised else 0.0),
        )
        if column < len(notes) and notes[column]:
            add_map_note(axis, notes[column])


def set_row_labels(axes: np.ndarray, labels: Sequence[str]) -> None:
    """Write each row label once, left of the first column's tick and coordinate labels."""
    for row, text in enumerate(labels):
        annotation = axes[row, 0].annotate(
            _bold_label(text),
            xy=(0.0, 0.5),
            xycoords="axes fraction",
            xytext=(-_ROW_LABEL_OFFSET_PT, 0.0),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="center",
            fontsize=_HEADER_SIZE_PT,
            fontweight="bold",
        )
        annotation.set_gid("channel-row-label")


def set_figure_title(figure: Figure, text: str) -> None:
    """Write the one bold figure title of a fixed-inch map grid into its top margin."""
    figure.suptitle(text, x=0.5, y=1.0 - _TITLE_FROM_TOP_IN / figure.get_figheight(), fontsize=_TITLE_SIZE_PT, fontweight="bold")
