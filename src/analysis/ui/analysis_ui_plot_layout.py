"""Define the shared fixed-inch map grid and the figure helpers kept beside it.

Responsibilities:
  - Plan fixed-inch map grids: equal maps with the domain's aspect, colour bars as tall as their maps, fixed gaps, the
    margins of row labels and a figure title, a width cap or none, and non-map bands above, below or right of the maps
  - Assemble a fixed map grid's map axes and hidden colour-bar slots in one place and give each band's figure rectangle
  - Write map-grid column headers with notes, row labels, one figure title and the text of an empty cell
  - Fill reserved fixed-inch colour-bar slots
  - Label the colour bar of a field-map scale from its kind: a constant's value, a cyclic scale's boundaries, a signed
    one's band centres, a wide log scale's decades, the power of ten in the unit bracket from the size of a state's
    values and from the step of an error, every label thinned to the bar's real tick spacing
  - Lay out a line figure's time windows as columns of one grid inside a given grid cell, rows sharing their y axis
    (``window_axes``), and head a row of such a figure with a formula and unit (``add_channel_row_label``)
  - Normalize finite constant and varying physical fields predictably

Design principles:
  - Layout never changes scientific arrays or their physical coordinates
  - One fixed-inch map grid serves EDA, Evaluation and generation-input maps
  - Every colorbar remains explicitly bound to its map's slot
  - A fixed-inch grid's figure size is a pure function of its plan (rows, columns, domain, margins and bands) and uses
    no layout engine, so a reused figure keeps its pixel size and axes positions
  - Every text of a fixed-inch map figure sets its size, taken from ``analysis_ui_figure_layout``, so no rc setting changes a map
  - Colour-bar labels are thinned when drawn, from the drawn tick positions, so every layout gets the real spacing

This module does NOT:
  - Choose scientific fields, units, colormaps, colour-scale boundaries or comparison semantics
  - Load data, construct notebook controls, or manage figure export
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis, YAxis
from matplotlib.colors import Normalize
from matplotlib.ticker import Formatter, LogFormatterSciNotation, NullFormatter
from mpl_toolkits.axes_grid1 import Size, make_axes_locatable  # type: ignore[import-untyped]

from src.analysis.presentation import analysis_field_maps as field_maps

from . import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec, SubplotSpec

    from src.analysis.presentation.analysis_field_maps import FieldMapExtend
    from src.analysis.ui.analysis_ui_time import TimeWindow

MAP_WIDTH_IN = 4.2  # target width of one drawn map
MAP_MAX_WIDTH_IN = 17.4  # default figure width cap: all maps shrink uniformly to stay under it
COLORBAR_WIDTH_IN = 0.14
COLORBAR_PAD_IN = 0.08  # map to its colour bar
GAP_H_IN = 1.10  # colour bar to the next map or to the right band: the bar's tick labels and unit label sit here
GAP_V_IN = 0.40  # between rows: x tick labels stand on the bottom row only
BAND_GAP_V_IN = 0.85  # maps to a band above or below: x tick labels, x label and the next panel title or column header
MARGIN_LEFT_IN = 0.95  # y tick labels and "y [m]"
MARGIN_LEFT_ROW_LABEL_IN = 1.50  # plus the row labels
MARGIN_RIGHT_IN = 1.10  # tick labels and unit label of the last colour bar
MARGIN_TOP_IN = 0.50  # column headers with one note line
MARGIN_TOP_TITLE_IN = 0.95  # plus the figure title
MARGIN_BOTTOM_IN = 0.75  # x tick labels and "x [m]"
TICK_LABEL_SPACING_MIN_IN = 0.16  # colour-bar labels closer than this are thinned; every tick keeps its mark
_LOG_DECADE_RATIO = 30.0  # a geometric scale wider than this factor ...
_LOG_DECADES_MIN = 4  # ... that holds at least this many decades is labelled at its decades
_MINIMUM_TICKS = 2
_GEOMETRIC_SIGNIFICANT_DIGITS = 3  # a narrow geometric scale (no decade labels) writes its smallest boundary with this many digits
_CYCLIC_COLORMAPS = frozenset({"twilight", "twilight_shifted", "hsv"})  # Matplotlib's cyclic colormaps: their two ends meet
_TITLE_FROM_TOP_IN = 0.12
_ROW_LABEL_OFFSET_PT = 62.0  # left of the first column's y tick labels and "y [m]"
_HEADER_PAD_PT = 4.0  # header to its map: slightly below Matplotlib's 6 pt
_HEADER_NOTE_LINE_PT = 14.0  # one 10 pt note line under a header
_NOTE_COLOUR = "0.20"
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_UNIT_BRACKET = re.compile(r"\[([^\]]*)\]\s*$")
_MATH_SPAN = re.compile(r"\$([^$]+)\$")


@dataclass(frozen=True, slots=True)
class MapGeometry:
    """Hold the resolved inch geometry of one fixed map grid: its maps, left and top margins and the non-map bands beside it."""

    rows: int
    columns: int
    map_width: float
    map_height: float
    left: float
    top: float
    above_in: float
    below_in: float
    right_in: float

    @property
    def cell_width(self) -> float:
        """Return the width of one map plus its colour-bar pad and slot."""
        return self.map_width + COLORBAR_PAD_IN + COLORBAR_WIDTH_IN

    @property
    def grid_width(self) -> float:
        """Return the width of the map columns: cells and the gaps between them."""
        return self.columns * self.cell_width + (self.columns - 1) * GAP_H_IN

    @property
    def grid_height(self) -> float:
        """Return the height of the map rows: maps and the gaps between them."""
        return self.rows * self.map_height + (self.rows - 1) * GAP_V_IN

    @property
    def figure_width(self) -> float:
        """Return the figure width: margins, map columns and a right band behind its gap."""
        return self.left + self.grid_width + _band_extent(self.right_in, GAP_H_IN) + MARGIN_RIGHT_IN

    @property
    def figure_height(self) -> float:
        """Return the figure height: margins, map rows and the bands above and below, each behind its gap."""
        return (
            self.top + _band_extent(self.above_in, BAND_GAP_V_IN) + self.grid_height + _band_extent(self.below_in, BAND_GAP_V_IN) + MARGIN_BOTTOM_IN
        )


@dataclass(frozen=True, slots=True)
class TickFormat:
    """The labels of a colour bar: ``decimals`` after the point, the ticks written divided by 10**``exponent`` (the factor stands in the unit)."""

    decimals: int = 1
    exponent: int = 0


class _BarLabels(Formatter):
    """Write colour-bar tick labels in one fixed format, thinned when drawn to the real spacing of the ticks on the bar."""

    def __init__(self, shown: TickFormat) -> None:
        self._shown = shown

    def __call__(self, x: float, pos: int | None = None) -> str:  # noqa: ARG002 -- Matplotlib's formatter signature
        """Return one label; rounding first turns a small negative value into +0.0, so no label reads "-0.0"."""
        value = round(x / 10.0**self._shown.exponent, self._shown.decimals) + 0.0
        return f"{value:.{self._shown.decimals}f}".replace("-", "\N{MINUS SIGN}")

    def format_ticks(self, values: list[float]) -> list[str]:
        """Return the labels of all tick ``values``, empty for the ticks thinned out (``_labelled_ticks``)."""
        ticks = np.asarray(values, dtype=np.float64)
        kept = _labelled_ticks(ticks, _tick_gaps_in(self.axis, ticks))
        return [self(value) if index in kept else "" for index, value in enumerate(ticks)]


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


def _band_extent(size: float, gap: float) -> float:
    """Return the room a band of ``size`` inches takes beside the maps: the band and its gap, nothing for an absent band."""
    return size + gap if size > 0.0 else 0.0


def _grid_bottom(geometry: MapGeometry) -> float:
    """Return the height in inches of the lowest map row's bottom edge above the figure's bottom edge."""
    return MARGIN_BOTTOM_IN + _band_extent(geometry.below_in, BAND_GAP_V_IN)


def _tick_gaps_in(axis: object, ticks: np.ndarray) -> np.ndarray:
    """Return the drawn distances in inches between neighbouring ``ticks`` on ``axis``; unknown distances are infinite."""
    if not isinstance(axis, Axis) or ticks.size < _MINIMUM_TICKS or (figure := axis.axes.get_figure(root=True)) is None:
        return np.full(max(ticks.size - 1, 0), np.inf)
    along = 1 if isinstance(axis, YAxis) else 0
    points = np.zeros((ticks.size, 2))
    points[:, along] = ticks
    return np.abs(np.diff(axis.axes.transData.transform(points)[:, along])) / figure.dpi


def _labelled_ticks(ticks: np.ndarray, gaps_in: np.ndarray) -> set[int]:
    """
    Return the indices of the ``ticks`` that keep a label when neighbouring ticks stand ``gaps_in`` inches apart.

    Every tick is labelled while each gap holds one label (``TICK_LABEL_SPACING_MIN_IN``). Otherwise every k-th tick keeps
    its label, k the smallest count of gaps that holds one: both ends always, no interior label closer than k gaps to an
    end, and a scale symmetric about 0 thinned symmetrically (an odd count from its centre tick, an even one from both
    ends). An odd symmetric scale always keeps its centre 0: with both ends when each half holds one label, else alone.
    Any other bar too short for both end labels keeps only its top label.
    """
    last = ticks.size - 1
    smallest = float(np.min(gaps_in)) if last >= 1 else math.inf
    if smallest >= TICK_LABEL_SPACING_MIN_IN:
        return set(range(ticks.size))
    step = math.ceil(TICK_LABEL_SPACING_MIN_IN / smallest) if smallest > 0.0 else last
    symmetric = bool(np.allclose(ticks, -ticks[::-1], rtol=0.0, atol=1e-9 * float(np.max(np.abs(ticks)))))
    if symmetric and last % 2 == 0:
        centre = last // 2
        if float(np.sum(gaps_in[:centre])) < TICK_LABEL_SPACING_MIN_IN:
            return {centre}
        core = {index for index in range(last + 1) if (index - centre) % step == 0}
        return {0, centre, last} | {index for index in core if min(index, last - index) >= step}
    if float(np.sum(gaps_in)) < TICK_LABEL_SPACING_MIN_IN:
        return {last}
    if symmetric:
        core = {index for index in range(last + 1) if min(index, last - index) % step == 0}
    else:
        core = {index for index in range(last + 1) if (last - index) % step == 0}
    return {0, last} | {index for index in core if min(index, last - index) >= step}


def _decades(boundaries: np.ndarray) -> list[float] | None:
    """Return the decades that label a wide geometric scale (a ratio above 30 holding at least four decades), else ``None``."""
    if boundaries.size < _MINIMUM_TICKS or boundaries[0] <= 0.0 or boundaries[-1] / boundaries[0] <= _LOG_DECADE_RATIO:
        return None
    steps = np.diff(np.log(boundaries))
    low, high = math.ceil(math.log10(boundaries[0]) - 1e-9), math.floor(math.log10(boundaries[-1]) + 1e-9)
    if not np.allclose(steps, steps[0]) or high - low + 1 < _LOG_DECADES_MIN:
        return None
    return [10.0**power for power in range(low, high + 1)]


def _geometric_format(boundaries: np.ndarray) -> TickFormat:
    """
    Return the format that writes every boundary of a narrow geometric scale with the significant digits of its smallest one.

    At least three; more when the smallest relative step r between adjacent boundaries needs them (1 - floor(log10 r)), so
    every absolute step is at least one unit of the last decimal and adjacent boundaries never read alike.
    """
    exponent = math.floor(math.log10(float(boundaries[-1])))
    smallest = float(boundaries[0]) / 10.0**exponent
    digits = max(_GEOMETRIC_SIGNIFICANT_DIGITS, 1 - math.floor(math.log10(float(np.min(np.diff(boundaries) / boundaries[:-1])))))
    return TickFormat(decimals=max(0, digits - 1 - math.floor(math.log10(smallest))), exponent=exponent)


def error_ticks(boundaries: np.ndarray, *, signed: bool) -> np.ndarray:
    """Return the labelled ticks of an error bar: the band boundaries, or for a signed scale the band centres 0, ±w, ±2w ... (w the band width)."""
    return (boundaries[:-1] + boundaries[1:]) / 2.0 if signed else boundaries


def tick_step(ticks: np.ndarray) -> float | None:
    """Return the smallest step between the ticks (``None`` for fewer than two or equal ones)."""
    step = float(np.min(np.diff(ticks))) if ticks.size >= _MINIMUM_TICKS else 0.0
    return step if step > 0.0 else None


def tick_format(step: float | None) -> TickFormat:
    """
    Return the labels of an error bar whose smallest round tick step is ``step``: exactly the decimals the step needs.

    A step that would need more than three decimals is written as a multiple of its power of ten instead, which the unit
    names. The error bars are centred on 0; a state bar uses ``state_format``, whose factor follows the size of its values.
    """
    if step is None:
        return TickFormat()
    decimals = field_maps.step_decimals(step)
    if decimals <= field_maps.MAX_TICK_DECIMALS:
        return TickFormat(decimals)
    exponent = math.floor(math.log10(step) + 1e-9)
    return TickFormat(field_maps.step_decimals(step / 10.0**exponent), exponent)


def state_format(steps: Iterable[float | None], exponent: int = 0) -> TickFormat:
    """
    Return the labels of a state bar written in 10**``exponent``: the decimals its frames' round tick steps need there, at most three.

    ``exponent`` is the bar's ``field_maps.state_exponent``, which follows the size of its values, never its step; 0 (no
    factor) for the states whose values lie between 10^-3 and 10^4. A frame whose step would need more decimals is
    near-uniform and does not set them (``field_maps.widened_range`` serves it); ``None`` steps (constant frames) are
    ignored. Without any frame that fits, the bar gets the three decimals.
    """
    needed = [field_maps.step_decimals(step / 10.0**exponent) for step in steps if step is not None]
    fitting = [decimals for decimals in needed if decimals <= field_maps.MAX_TICK_DECIMALS]
    return TickFormat(max(fitting) if fitting else field_maps.MAX_TICK_DECIMALS if needed else 1, exponent)


def error_format(boundaries: np.ndarray | None, *, signed: bool) -> TickFormat:
    """Return the label format of an error bar over ``boundaries``, the one ``set_fixed_decimal_ticks`` derives from its ticks."""
    return TickFormat() if boundaries is None else tick_format(tick_step(error_ticks(boundaries, signed=signed)))


def is_finer_than_decimals(step: float | None, decimals: int) -> bool:
    """Return whether a round tick ``step`` is below the unit of the last decimal ``decimals``."""
    return step is not None and step < 10.0**-decimals * (1.0 - 1e-9)


def set_fixed_decimal_ticks(colorbar: Colorbar, ticks: np.ndarray, fixed: TickFormat | None = None) -> None:
    """
    Label ``colorbar`` at ``ticks`` in the format ``fixed`` (default: the one its own ticks need), with the true minus sign and no "-0".

    A power of ten is written into the unit bracket of the bar's label. A caller that draws several frames passes the format
    that fits all of them, so the labels and the unit never change between frames. When the bar is drawn, labels closer
    than ``TICK_LABEL_SPACING_MIN_IN`` are thinned (``_labelled_ticks``); every tick keeps its mark.
    """
    shown = tick_format(tick_step(ticks)) if fixed is None else fixed
    colorbar.set_ticks(ticks.tolist())
    if shown.exponent:
        colorbar.set_label(label_with_exponent(colorbar.ax.get_ylabel(), shown.exponent))
    colorbar.ax.yaxis.set_major_formatter(_BarLabels(shown))


def window_axes(
    figure: Figure,
    spec: SubplotSpec,
    windows: Sequence[TimeWindow],
    row_heights: Sequence[float],
    **grid_kwargs: Any,
) -> np.ndarray:
    """
    Return the axes ``[row, window]`` of a grid in ``spec``: one column per time window, as wide as the window asks.

    A row shares its y axis over the windows, whose y tick labels stay on the first column. ``grid_kwargs`` are the
    spacings of the grid (``wspace``, ``hspace``).
    """
    grid = spec.subgridspec(
        len(row_heights), len(windows), width_ratios=[window.width_ratio for window in windows], height_ratios=list(row_heights), **grid_kwargs
    )
    axes = np.empty((len(row_heights), len(windows)), dtype=object)
    for row, column in np.ndindex(axes.shape):
        axis = axes[row, column] = figure.add_subplot(grid[row, column], sharey=axes[row, 0] if column else None)
        if column:
            axis.tick_params(axis="y", labelleft=False)
    return axes


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
        fontsize=figure_layout.PANEL_TITLE_PT,
        fontweight="bold",
        annotation_clip=False,
    )
    annotation.set_gid("channel-row-label")


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


def plan_geometry(
    rows: int,
    columns: int,
    domain: tuple[float, float],
    *,
    row_labels: bool = True,
    title: bool = True,
    width_cap: float | None = MAP_MAX_WIDTH_IN,
    above_in: float = 0.0,
    below_in: float = 0.0,
    right_in: float = 0.0,
) -> MapGeometry:
    """
    Return the fixed-inch geometry of one map grid.

    Every map has the same width and the physical aspect of ``domain`` (width, height). The left margin holds the row
    labels (``row_labels``), the top margin the figure title (``title``) above the column headers. ``above_in``,
    ``below_in`` and ``right_in`` reserve non-map bands of that height or width beside the maps, each behind one fixed
    gap (``band_rect``). The maps shrink uniformly when the figure would be wider than ``width_cap``; ``None`` never
    shrinks them, so the figure grows with its columns. Non-positive counts, a non-finite or non-positive domain, a
    negative or non-finite band, or a cap that leaves no room for the maps raise ``ValueError``.
    """
    if rows <= 0 or columns <= 0:
        message = "Map grids need positive row and column counts."
        raise ValueError(message)
    width, height = (float(value) for value in domain)
    if not (np.isfinite(width) and np.isfinite(height) and width > 0.0 and height > 0.0):
        message = f"Map grids need a finite positive physical domain, got {domain!r}."
        raise ValueError(message)
    bands = (float(above_in), float(below_in), float(right_in))
    if not all(math.isfinite(band) and band >= 0.0 for band in bands):
        message = f"Map-figure bands need finite non-negative sizes, got {bands!r}."
        raise ValueError(message)
    left = MARGIN_LEFT_ROW_LABEL_IN if row_labels else MARGIN_LEFT_IN
    map_width = MAP_WIDTH_IN
    if width_cap is not None:
        fixed = left + MARGIN_RIGHT_IN + _band_extent(bands[2], GAP_H_IN) + columns * (COLORBAR_PAD_IN + COLORBAR_WIDTH_IN) + (columns - 1) * GAP_H_IN
        available = (width_cap - fixed) / columns
        if not (math.isfinite(available) and available > 0.0):
            message = f"A width cap of {width_cap!r} in leaves no room for {columns} maps."
            raise ValueError(message)
        map_width = min(MAP_WIDTH_IN, available)
    return MapGeometry(rows, columns, map_width, map_width * height / width, left, MARGIN_TOP_TITLE_IN if title else MARGIN_TOP_IN, *bands)


def band_rect(geometry: MapGeometry, where: Literal["above", "below", "right"]) -> tuple[float, float, float, float]:
    """
    Return the (left, bottom, width, height) figure fractions of one non-map band the geometry reserves.

    The bands above and below span the map columns; the right band spans the map rows only, so an axes there shares the
    maps' vertical extent. A band the geometry does not reserve, or an unknown ``where``, raises ``ValueError``.
    """
    sizes = {"above": geometry.above_in, "below": geometry.below_in, "right": geometry.right_in}
    if sizes.get(where, 0.0) <= 0.0:
        message = f"The map geometry reserves no {where!r} band."
        raise ValueError(message)
    bottom = _grid_bottom(geometry)
    x0, y0, width, height = {
        "above": (geometry.left, bottom + geometry.grid_height + BAND_GAP_V_IN, geometry.grid_width, geometry.above_in),
        "below": (geometry.left, MARGIN_BOTTOM_IN, geometry.grid_width, geometry.below_in),
        "right": (geometry.left + geometry.grid_width + GAP_H_IN, bottom, geometry.right_in, geometry.grid_height),
    }[where]
    return x0 / geometry.figure_width, y0 / geometry.figure_height, width / geometry.figure_width, height / geometry.figure_height


def map_figure(geometry: MapGeometry, *, figure: Figure | None = None) -> tuple[Figure, GridSpec]:
    """
    Return a figure of exactly the geometry's size without a layout engine and its grid of map cells.

    Each cell is one map plus its colour-bar slot wide and one map tall; the bands stay free (``band_rect``). A given
    ``figure`` (made here for the same geometry) is cleared and reused, so animation frames keep one pixel size and
    identical axes positions; a figure of another size or with a layout engine that would re-solve the positions raises
    ``ValueError``.
    """
    size = (geometry.figure_width, geometry.figure_height)
    if figure is None:
        figure = plt.figure(figsize=size, layout="none")
    elif figure.get_layout_engine() is not None or not np.allclose(figure.get_size_inches(), size):
        message = "A reused map figure needs the geometry's size and no layout engine."
        raise ValueError(message)
    else:
        figure.clear()
    bottom = _grid_bottom(geometry)
    grid = figure.add_gridspec(
        geometry.rows,
        geometry.columns,
        left=geometry.left / geometry.figure_width,
        right=(geometry.left + geometry.grid_width) / geometry.figure_width,
        bottom=bottom / geometry.figure_height,
        top=(bottom + geometry.grid_height) / geometry.figure_height,
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
    axis.tick_params(labelbottom=bottom, labelleft=left, labelsize=figure_layout.TICK_PT)
    axis.set_xlabel("x [m]" if bottom else "", fontsize=figure_layout.LABEL_PT)
    axis.set_ylabel("y [m]" if left else "", fontsize=figure_layout.LABEL_PT)
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
    only at the ends the caller found values beyond. The caller labels the ticks: ``add_field_map_colorbar`` for the bar
    of a field-map scale, ``set_fixed_decimal_ticks`` for ticks of its own.
    """
    label_with_exponent(label, 0)
    slot.set_visible(True)
    colorbar = figure.colorbar(image, cax=slot, extend=extend)
    colorbar.ax.tick_params(labelsize=figure_layout.TICK_PT)
    colorbar.set_label(label, size=figure_layout.LABEL_PT)
    return colorbar


def add_field_map_colorbar(
    figure: Figure,
    image: ScalarMappable,
    slot: Axes,
    scale: field_maps.FieldMapDefinition,
    *,
    label: str,
    extend: FieldMapExtend | None = None,
) -> Colorbar:
    """
    Fill one map's slot with the bar of ``scale`` (``add_slot_colorbar``) and label it from the scale's kind, the one place that does.

    - A constant scale has one tick at its value.
    - A cyclic scale (a colormap whose ends meet, ``_CYCLIC_COLORMAPS``) is labelled at its boundaries, so both wrap ends are named.
    - Any other signed scale (``field_maps.SIGNED_SEMANTICS``) is labelled at its band centres 0, ±w, ±2w ...
    - A log-positive scale is labelled at its decades when it is wide (``_decades``), every boundary a minor tick, else at its
      boundaries with the significant digits of the smallest one that keep adjacent ones distinct, at least three (``_geometric_format``).
    - Where the ticks stand and how they are written are separate: a linear state, positive or signed, has at most three decimals
      in the power of ten of its boundaries' size (``field_maps.state_exponent``, the one its state bands were built in; none
      between 10^-3 and 10^4) in the unit bracket (``state_format``); an error (``field_maps.ERROR_SEMANTICS``), signed or not,
      gets exactly the decimals its step needs, a finer step as a power of ten in the unit bracket.

    Labels are thinned to the bar's real height. ``extend`` replaces the scale's arrows for one of several maps that share a locked
    scale but not its values beyond. A categorical scale raises ``ValueError``, and so does a linear or narrow geometric scale whose ticks, written
    as the bar writes them, would repeat a label (locked boundaries off the grid of their format).
    """
    if scale.semantic is field_maps.FieldMapSemantic.CATEGORICAL:
        message = "A categorical field map has no slot colour bar."
        raise ValueError(message)
    colorbar = add_slot_colorbar(figure, image, slot, label=label, extend=scale.extend if extend is None else extend)
    boundaries = np.asarray(scale.boundaries, dtype=np.float64)
    if scale.semantic is field_maps.FieldMapSemantic.CONSTANT:
        colorbar.set_ticks(boundaries.tolist())
        return colorbar
    if scale.semantic is field_maps.FieldMapSemantic.LOG_POSITIVE:
        decades = _decades(boundaries)
        if decades is None:
            set_fixed_decimal_ticks(colorbar, boundaries, _geometric_format(boundaries))
            _require_distinct_labels(colorbar, boundaries, label)
            return colorbar
        colorbar.set_ticks(decades)
        colorbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        colorbar.set_ticks(boundaries.tolist(), minor=True)
        colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
        return colorbar
    centred = scale.semantic in field_maps.SIGNED_SEMANTICS and scale.colormap.name not in _CYCLIC_COLORMAPS
    ticks = error_ticks(boundaries, signed=centred)
    error = scale.semantic in field_maps.ERROR_SEMANTICS
    exponent = field_maps.state_exponent(float(boundaries[0]), float(boundaries[-1]))
    set_fixed_decimal_ticks(colorbar, ticks, None if error else state_format([tick_step(ticks)], exponent))
    _require_distinct_labels(colorbar, ticks, label)
    return colorbar


def _require_distinct_labels(colorbar: Colorbar, ticks: np.ndarray, label: str) -> None:
    """Raise ``ValueError`` naming ``label`` when two ``ticks``, written by the bar's own formatter before thinning, read alike."""
    texts = [colorbar.ax.yaxis.get_major_formatter()(tick) for tick in ticks]
    if repeated := sorted({text for text in texts if texts.count(text) > 1}):
        message = f"The colour bar {label!r} would repeat the tick labels {repeated}."
        raise ValueError(message)


def empty_cell(axis: Axes, text: str) -> None:
    """Show one grid cell as empty: no frame and no ticks, ``text`` centred in it."""
    axis.set_axis_off()
    axis.text(0.5, 0.5, text, ha="center", va="center", wrap=True, transform=axis.transAxes, fontsize=figure_layout.LABEL_PT)


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
        fontsize=figure_layout.NOTE_PT,
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
            fontsize=figure_layout.PANEL_TITLE_PT,
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
            fontsize=figure_layout.PANEL_TITLE_PT,
            fontweight="bold",
        )
        annotation.set_gid("channel-row-label")


def set_map_title(figure: Figure, text: str) -> None:
    """Write the one bold figure title of a fixed-inch map grid into its top margin."""
    figure.suptitle(text, x=0.5, y=1.0 - _TITLE_FROM_TOP_IN / figure.get_figheight(), fontsize=figure_layout.TITLE_PT, fontweight="bold")
