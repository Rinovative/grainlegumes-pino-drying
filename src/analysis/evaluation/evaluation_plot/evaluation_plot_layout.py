"""Define shared current-native layout contracts for evaluation field maps.

Responsibilities:
  - Keep equivalent evaluation maps at one readable notebook-native scale
  - Keep map colorbars proportionate and consistently padded
  - Keep comparable map axes equal-sized with row/column-aware axis labels
  - Size composite line-and-map figures without plot-local magic numbers
  - Center figure titles over data axes when a right-side column is present
  - Mark panels whose bounded prefix has no finite case as unavailable
  - Choose the round error bands over a percentile reach, label colour-bar ticks at fixed decimals (a finer
    step as a multiple of a power of ten in the unit) and write the one colour-scale note and the mean-error line
    above each error map of a fixed-inch map grid

Design principles:
  - Layout is presentation-only and never changes scientific values
  - Physical aspect ratios remain owned by the plotting axes
  - One immutable contract serves predictive, physics, sample, and outlier maps

This module does NOT:
  - Load artifacts, select cases, calculate statistics, or clip scientific data
  - Choose colormaps, labels, coordinates, or plot-specific panel composition
  - Own notebook controls, display calls, exports, or lazy rendering
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Sized

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.text import Text


@dataclass(frozen=True, slots=True)
class MapLayoutContract:
    """Define one consistent notebook-scale physical-map layout."""

    column_width: float = shared_layout.MAP_LAYOUT.column_width
    row_height: float = shared_layout.MAP_LAYOUT.row_height
    maximum_width: float = shared_layout.MAP_LAYOUT.maximum_width
    map_column_spacing: float = 0.18
    map_row_spacing: float = 0.10
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.025
    composite_line_height: float = 3.25
    composite_vertical_gap: float = 0.25
    composite_height_ratios: tuple[float, float] = (1.0, 1.35)
    composite_hspace: float = 0.30
    composite_group_spacing: float = 0.28

    def figure_size(self, *, rows: int, columns: int) -> tuple[float, float]:
        """Return a readable size for an all-map grid."""
        if rows <= 0 or columns <= 0:
            msg = "Map grids require positive row and column counts."
            raise ValueError(msg)
        return min(self.column_width * columns, self.maximum_width), self.row_height * rows

    def composite_size(self, *, map_columns: int) -> tuple[float, float]:
        """Return size for one line region above one row of maps."""
        if map_columns <= 0:
            msg = "Composite figures require at least one map column."
            raise ValueError(msg)
        return min(self.column_width * map_columns, self.maximum_width), self.composite_line_height + self.composite_vertical_gap + self.row_height


MAP_LAYOUT = MapLayoutContract()
_NOTE_FROM_TOP_IN = 0.36  # the scale note's top below the figure top: under the 12 pt title of a fixed-inch map grid
_NOTE_SIZE_PT = 9.0
_NOTE_COLOUR = "0.20"
_MINIMUM_TICKS = 2
MAX_TICK_DECIMALS = 3  # an error bar's finer step moves its power of ten into the unit; a state bar never does (owner 14.11.)


def error_boundaries(values: np.ndarray, *, signed: bool) -> np.ndarray:
    """
    Return the round error bands of the finite error ``values``: from 0 to the round top over their reach, or symmetric about 0 if signed.

    The reach is the 95th percentile (signed: the larger absolute one of the 5th and 95th). The bands are
    the round bands of ``shared_layout.rounded_boundaries`` (9 to 13 of m times a power of ten, m in 1, 2, 2.5, 5); a signed
    scale has an odd count with the neutral band centred on 0 (``shared_layout.symmetric_boundaries``).
    A map without any error (reach 0) gets the scale of reach 1, so it never degenerates to one colour.
    """
    scale = field_maps.percentile_scale(values, "signed" if signed else "magnitude")
    if scale is None:
        msg = "No round error bands exist for values without a finite error."
        raise ValueError(msg)
    reach = scale[1] if scale[1] > 0.0 else 1.0
    boundaries = shared_layout.symmetric_boundaries(reach) if signed else shared_layout.rounded_boundaries(0.0, reach)
    if boundaries is None:
        msg = f"No round error bands exist for a reach of {reach!r}."
        raise ValueError(msg)
    # BoundaryNorm sends a value equal to its last boundary to the over colour; an error exactly at the round top belongs to the scale.
    boundaries[-1] = np.nextafter(boundaries[-1], np.inf)
    return boundaries


def state_boundaries(lower: float, upper: float) -> np.ndarray:
    """
    Return the round state boundaries (9 to 13 bands) from ``lower`` that reach ``upper``, never cut short.

    A constant value spans its magnitude (at least 1) upward, so it fills the lowest band of a readable bar. A near-uniform
    range whose round step is finer than the three decimals a state bar may show is widened to 11 bands of 0.001 about its centre.
    """
    top = upper if upper > lower else lower + max(abs(lower), 1.0)
    boundaries = shared_layout.rounded_boundaries(lower, top, max_decimals=MAX_TICK_DECIMALS)
    if boundaries is None:
        return np.linspace(*widened_range(lower, top, MAX_TICK_DECIMALS), field_maps.N_COLOR_BINS + 1)
    return boundaries


def error_ticks(boundaries: np.ndarray, *, signed: bool) -> np.ndarray:
    """Return the labelled ticks of an error bar: the band boundaries, or for a signed scale the band centres 0, ±w, ±2w ... (w the band width)."""
    return (boundaries[:-1] + boundaries[1:]) / 2.0 if signed else boundaries


@dataclass(frozen=True, slots=True)
class TickFormat:
    """The labels of a colour bar: ``decimals`` after the point, the ticks written divided by 10**``exponent`` (the factor stands in the unit)."""

    decimals: int = 1
    exponent: int = 0


def tick_step(ticks: np.ndarray) -> float | None:
    """Return the smallest step between the ticks (``None`` for fewer than two or equal ones)."""
    step = float(np.min(np.diff(ticks))) if ticks.size >= _MINIMUM_TICKS else 0.0
    return step if step > 0.0 else None


def tick_format(step: float | None) -> TickFormat:
    """
    Return the labels of an error bar whose smallest round tick step is ``step``: exactly the decimals the step needs.

    A step that would need more than three decimals is written as a multiple of its power of ten instead, which the unit
    names. The error bars are centred on 0; a state bar uses ``state_format`` and never gets a factor.
    """
    if step is None:
        return TickFormat()
    decimals = shared_layout.step_decimals(step)
    if decimals <= MAX_TICK_DECIMALS:
        return TickFormat(decimals)
    exponent = math.floor(math.log10(step) + 1e-9)
    return TickFormat(shared_layout.step_decimals(step / 10.0**exponent), exponent)


def state_format(steps: Iterable[float | None]) -> TickFormat:
    """
    Return the labels of a state bar, which never carries a power of ten: the decimals its frames' round tick steps need, at most three.

    A frame whose step would need more decimals is near-uniform and does not set them (``widened_range`` serves it);
    ``None`` steps (constant frames) are ignored. Without any frame that fits, the bar gets the three decimals.
    """
    needed = [shared_layout.step_decimals(step) for step in steps if step is not None]
    fitting = [decimals for decimals in needed if decimals <= MAX_TICK_DECIMALS]
    return TickFormat(max(fitting) if fitting else MAX_TICK_DECIMALS if needed else 1)


def is_finer_than_decimals(step: float | None, decimals: int) -> bool:
    """Return whether a round tick ``step`` is below the unit of the last decimal ``decimals``."""
    return step is not None and step < 10.0**-decimals * (1.0 - 1e-9)


def widened_range(lower: float, upper: float, decimals: int) -> tuple[float, float]:
    """Return a range widened symmetrically about its centre to 11 steps of 10**-``decimals``, starting on a whole step."""
    unit = 10.0**-decimals
    start = round(((lower + upper) / 2.0 - 5.5 * unit) / unit) * unit
    return start, start + 11.0 * unit


def set_fixed_decimal_ticks(colorbar: Colorbar, ticks: np.ndarray, fixed: TickFormat | None = None) -> None:
    """
    Label ``colorbar`` at ``ticks`` in the format ``fixed`` (default: the one its own ticks need), with the true minus sign and no "-0".

    A power of ten is written into the unit bracket of the bar's label. A caller that draws several frames passes the format
    that fits all of them, so the labels and the unit never change between frames.
    """
    shown = tick_format(tick_step(ticks)) if fixed is None else fixed
    colorbar.set_ticks(ticks.tolist())
    if shown.exponent:
        colorbar.set_label(shared_layout.label_with_exponent(colorbar.ax.get_ylabel(), shown.exponent))
    # Rounding first turns a small negative value into +0.0, so no label reads "-0.0".
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: f"{round(value / 10.0**shown.exponent, shown.decimals) + 0.0:.{shown.decimals}f}".replace("-", "\N{MINUS SIGN}")
        )
    )


def add_scale_note(figure: Figure, *, shared: bool = True) -> None:
    """Write the one colour-scale note under the title of a fixed-inch map grid (``shared``: reference scale)."""
    reference = "prediction uses the reference scale; " if shared else ""
    figure.text(
        0.5,
        1.0 - _NOTE_FROM_TOP_IN / figure.get_figheight(),
        f"Colour scales: 5th\N{EN DASH}95th percentile; {reference}arrows mark values outside",
        ha="center",
        va="top",
        fontsize=_NOTE_SIZE_PT,
        color=_NOTE_COLOUR,
    )


def error_format(boundaries: np.ndarray | None, *, signed: bool) -> TickFormat:
    """Return the label format of an error bar over ``boundaries``, the one ``set_fixed_decimal_ticks`` derives from its ticks."""
    return TickFormat() if boundaries is None else tick_format(tick_step(error_ticks(boundaries, signed=signed)))


def mean_error_text(values: np.ndarray | None, *, unit: str | None, exponent: int = 0) -> str:
    """
    Return the mean-error line of one error map: mean |e| of its finite ``values`` in ``unit`` (``None``: normalized), else "n/a".

    The mean runs over every finite cell of the field, independent of the display sampling. A bar that writes its labels as
    multiples of a power of ten (``exponent``, from ``error_format``) gets the mean as a multiple of that same power. The
    mean keeps three significant digits and is always written positionally, never in e-notation.
    """
    finite = None if values is None else np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite is None or finite.size == 0:
        return "mean |e| n/a"
    power = f" \N{MULTIPLICATION SIGN} {shared_layout.power_of_ten(exponent)}" if exponent else ""
    digits = np.format_float_positional(float(np.mean(np.abs(finite))) / 10.0**exponent, precision=3, unique=False, fractional=False, trim="-")
    mean = f"{digits}{power}"
    return f"mean |e| = {mean} {unit}" if unit is not None else f"mean {error_diagnostics.NORMALIZED_ERROR_SYMBOL} = {mean}"


def set_error_headers(axes: np.ndarray, titles: Sequence[str], error_texts: Sequence[str]) -> None:
    """Write the column headers and, above each row's error map (last column), its mean-error line; the top row's stands under its header."""
    shared_layout.set_column_headers(axes, titles, ("",) * (len(titles) - 1) + (error_texts[0],))
    for row in range(1, len(error_texts)):
        shared_layout.add_map_note(axes[row, -1], error_texts[row])


def case_title(
    title: str,
    *,
    case_count: int | None = None,
    case_number: int | None = None,
) -> str:
    """Append exactly one aggregate-count or visible-case suffix."""
    if (case_count is None) == (case_number is None):
        msg = "Supply exactly one of case_count or case_number."
        raise ValueError(msg)
    value = case_count if case_count is not None else case_number
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        msg = "Case counts and visible case numbers must be positive integers."
        raise ValueError(msg)
    numeric = int(value)
    suffix = f"{numeric} {'case' if numeric == 1 else 'cases'}" if case_count is not None else f"Case {numeric}"
    return f"{title} — {suffix}"


def effective_case_counts(
    datasets: Mapping[str, Sized],
    *,
    max_cases: int | None = None,
) -> dict[str, int]:
    """Return actual per-dataset counts used by a complete or bounded view."""
    if not datasets:
        msg = "Case-based titles require at least one labelled dataset."
        raise ValueError(msg)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, Integral) or int(max_cases) <= 0):
        msg = "max_cases must be a positive integer when supplied."
        raise ValueError(msg)
    return {label: len(dataset) if max_cases is None else min(int(max_cases), len(dataset)) for label, dataset in datasets.items()}


def aggregate_title_context(title: str, case_counts: Mapping[str, int]) -> tuple[str, dict[str, str]]:
    """Return an honest figure title and dataset headings for aggregate evidence."""
    if not case_counts:
        msg = "Aggregate title context requires at least one dataset count."
        raise ValueError(msg)
    validated = {label: case_title(label, case_count=count) for label, count in case_counts.items()}
    unique_counts = set(case_counts.values())
    if len(unique_counts) == 1:
        return case_title(title, case_count=next(iter(unique_counts))), dict.fromkeys(case_counts, "")
    return title, validated


def empty_prefix_text(case_count: int) -> str:
    """Return how a bounded-prefix view states that no finite case remains, never a number and never zero."""
    return f"unavailable (0 finite of {case_count} prefix cases)"


def mark_empty_prefix(*axes: Axes, case_count: int) -> None:
    """Show on every given panel that its bounded prefix has no finite case, instead of drawing or failing."""
    for axis in axes:
        axis.text(0.5, 0.5, empty_prefix_text(case_count), ha="center", va="center", transform=axis.transAxes)


def map_subplots(*, rows: int, columns: int, **kwargs: Any) -> tuple[Figure, Any]:
    """Create one equal-column map grid under the shared layout contract."""
    if "figsize" in kwargs or "squeeze" in kwargs or "layout" in kwargs or "gridspec_kw" in kwargs:
        msg = "Shared map grids own figsize, squeeze, layout, and GridSpec spacing."
        raise ValueError(msg)
    return plt.subplots(
        rows,
        columns,
        figsize=MAP_LAYOUT.figure_size(rows=rows, columns=columns),
        squeeze=False,
        layout="compressed",
        gridspec_kw={"wspace": MAP_LAYOUT.map_column_spacing, "hspace": MAP_LAYOUT.map_row_spacing},
        **kwargs,
    )


def add_map_colorbar(
    figure: Figure,
    image: ScalarMappable,
    axis: Axes,
    *,
    label: str | None = None,
    colorbar_axis: Axes | None = None,
) -> Colorbar:
    """Attach one proportionate colorbar under the shared sizing contract."""
    if colorbar_axis is None:
        colorbar = figure.colorbar(image, ax=axis, fraction=MAP_LAYOUT.colorbar_fraction, pad=MAP_LAYOUT.colorbar_pad)
    else:
        colorbar = figure.colorbar(image, cax=colorbar_axis)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def _map_axis_grid(axes: Iterable[Iterable[Axes]]) -> tuple[tuple[Axes, ...], ...]:
    """Normalize and validate one rectangular map-axis grid."""
    rows = tuple(tuple(row) for row in axes)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        msg = "Map-axis grids must be non-empty and rectangular."
        raise ValueError(msg)
    return rows


def apply_map_grid_axis_labels(
    axes: Iterable[Iterable[Axes]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    """
    Apply coordinate and tick labels only at visible map-grid edges.

    Parameters
    ----------
    axes : Iterable[Iterable[Axes]]
        Rectangular rows of map axes. Invisible or disabled axes are ignored.
    x_label : str
        Label applied to active maps in the global bottom row.
    y_label : str
        Label applied to active maps in the global leftmost column.

    Raises
    ------
    ValueError
        If the map grid is empty or not rectangular.

    Notes
    -----
    Tick positions are preserved. Repeated tick-label artists are hidden without
    changing map extents, and a standalone active map keeps complete axes.

    """
    rows = _map_axis_grid(axes)
    active = tuple(tuple(axis.get_visible() and axis.axison for axis in row) for row in rows)
    bottom_row = len(rows) - 1
    for row, axis_row in enumerate(rows):
        for column, axis in enumerate(axis_row):
            if not active[row][column]:
                continue
            is_bottom = row == bottom_row
            is_left = column == 0
            axis.set_xlabel(x_label if is_bottom else "")
            axis.set_ylabel(y_label if is_left else "")
            axis.tick_params(axis="x", labelbottom=is_bottom)
            axis.tick_params(axis="y", labelleft=is_left)


def _center_text_artist(figure: Figure, artist: Text, plot_axes: tuple[Axes, ...]) -> None:
    """Keep one text artist centered over its visible plotting axes."""

    def center_artist(_event: object = None) -> None:
        """Follow final Matplotlib positions without forcing an eager draw."""
        positions = [axis.get_position() for axis in plot_axes if axis.get_visible()]
        if positions:
            left = min(position.x0 for position in positions)
            right = max(position.x1 for position in positions)
            artist.set_x((left + right) / 2.0)

    center_artist()
    figure.canvas.mpl_connect("draw_event", center_artist)


def title_over_axes(
    figure: Figure,
    title: str,
    axes: Iterable[Axes],
    **kwargs: Any,
) -> Text:
    """Keep one figure title centered over its data axes after layout."""
    plot_axes = tuple(axes)
    if not plot_axes:
        msg = "Title centering requires at least one plotting axis."
        raise ValueError(msg)
    title_artist = figure.suptitle(title, **kwargs)
    _center_text_artist(figure, title_artist, plot_axes)
    return title_artist


def text_over_axes(
    figure: Figure,
    text: str,
    axes: Iterable[Axes],
    *,
    y: float,
    **kwargs: Any,
) -> Text:
    """Place figure-level explanatory text above and centered on data axes."""
    plot_axes = tuple(axes)
    if not plot_axes:
        msg = "Explanatory-text centering requires at least one plotting axis."
        raise ValueError(msg)
    text_artist = figure.text(0.5, y, text, **kwargs)
    _center_text_artist(figure, text_artist, plot_axes)
    return text_artist
