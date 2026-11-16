"""Own the geometry, text sizes, legends and titles of line, bar and distribution figures.

Responsibilities:
  - Build constrained panel grids whose panels have one fixed size, one shared gap and the shared text sizes
  - Put an axis on a log scale with non-overlapping tick labels (``set_log_scale``)
  - Collect the legend entries of a figure's axes
  - Place the legend of one panel inside it and the legend of several panels outside right, top aligned with the top plot frame
  - Write the y label once per row, row headers above the panels and the figure title with its note over the data axes

Design principles:
  - The text sizes are the ones the map figures use (``analysis_ui_plot_layout`` imports them), so every figure type reads alike
  - A legend outside the panels grows the figure; it never shrinks or moves a panel, and no entry wraps
  - A long note wraps to the figure width instead of widening the figure past its panels
  - Geometry is derived from the panel grid and the artists, never from the number of curves

This module does NOT:
  - Choose colours, line styles or the entities a legend names
  - Draw maps or colour bars; ``analysis_ui_plot_layout`` owns those
  - Own scientific labels or units
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure

TICK_PT = 9.0
LABEL_PT = 10.0
PANEL_TITLE_PT = 11.0
TITLE_PT = 12.0
NOTE_PT = 10.0

PANEL_PAD_IN = 0.03  # padding around every panel in inches; Matplotlib's default is 3/72 (0.042)
LEGEND_GAP_IN = 0.15  # panels to the legend column
LEGEND_STACK_GAP_IN = 0.1  # between stacked legend groups
ROW_HEADER_RISE_IN = 0.3  # row header above the first panel's top edge, clear of its panel title
NOTE_GAP_PT = 2.0  # title to its note, smaller than a full line
_LINE_HEIGHT = 1.25  # text line height in font sizes
_WRAP_SHRINK = 0.9  # columns kept per retry while a wrapped note is still wider than the figure
_MINOR_LABEL_ORDER = (5.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0)  # mantissas a log axis may label, most preferred first
_MINOR_LABEL_GAP = 0.3  # least distance of two labelled log ticks, as a share of the view's span in decades
_MINIMUM_LOG_LABELS = 2  # labels a log view should keep; below it the gap halves once
_MAX_LABEL_DIGITS = 17  # significant digits that tell any two distinct float64 values apart
_POINTS_PER_INCH = 72.0
_GRID_NDIM = 2  # rows x columns


class _CenteredText(Text):
    """Text centred over the visible panels of its figure, recomputed at every draw so a later layout change never leaves it stale."""

    def __init__(self, panels: Sequence[Axes], *args: Any, **kwargs: Any) -> None:
        """Keep the panels to centre over."""
        super().__init__(*args, **kwargs)
        self._panels = tuple(panels)

    def draw(self, renderer: RendererBase) -> None:
        """Centre over the final positions, then draw."""
        positions = [axis.get_position() for axis in self._panels if axis.get_visible()]
        if positions:
            centre = (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2.0
            half = min(0.5, self.get_window_extent(renderer).width / 2.0 / self.get_figure().bbox.width)  # type: ignore[union-attr]
            self.set_x(min(max(centre, half), 1.0 - half))  # shifted inward only where the text would pass the figure edge
        super().draw(renderer)


def _blocks(entries: Mapping[str, Artist], title: str | None, groups: Mapping[str, Sequence[str]] | None) -> list[tuple[str | None, list[str]]]:
    """Return the legend blocks as (title, labels): one block, or the groups followed by the unlisted labels."""
    if groups is None:
        return [(title, list(entries))]
    blocks: list[tuple[str | None, list[str]]] = [(name, [label for label in names if label in entries]) for name, names in groups.items()]
    listed = {label for _name, names in blocks for label in names}
    blocks.append((None, [label for label in entries if label not in listed]))
    return [(name, names) for name, names in blocks if names]


def _grid(axes: np.ndarray) -> np.ndarray:
    """Return the two-dimensional axes array of ``panel_figure``; any other shape raises ``ValueError``."""
    grid = np.asarray(axes, dtype=object)
    if grid.ndim != _GRID_NDIM:
        message = f"Expected the two-dimensional axes array of panel_figure, got {grid.ndim} dimension(s)."
        raise ValueError(message)
    return grid


def _engine(figure: Figure) -> ConstrainedLayoutEngine:
    """Return the constrained layout engine of a figure from ``panel_figure``."""
    engine = figure.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        message = "Expected a figure from panel_figure with a constrained layout."
        raise TypeError(message)
    return engine


def _grow(figure: Figure, *, right_in: float = 0.0, top_in: float = 0.0, bottom_in: float = 0.0) -> None:
    """Add a strip right of, above and below the panels; the panels' rectangle keeps its size in inches."""
    engine = _engine(figure)
    width, height = figure.get_size_inches()
    new_width, new_height = width + right_in, height + top_in + bottom_in
    left, bottom, rect_width, rect_height = engine.get()["rect"]  # Matplotlib's rect is (left, bottom, width, height)
    engine.set(
        rect=(
            left * width / new_width,
            (bottom * height + bottom_in) / new_height,
            rect_width * width / new_width,
            rect_height * height / new_height,
        )
    )
    figure.set_size_inches(new_width, new_height)


def _note_width_in(figure: Figure, note: str) -> float:
    """Return the rendered width in inches of the widest line of ``note`` at the note size."""
    probe = figure.text(0.0, 0.0, note, fontsize=NOTE_PT)
    try:
        return probe.get_window_extent().width / figure.dpi
    finally:
        probe.remove()


def _wrap_note(figure: Figure, note: str) -> str:
    """Wrap every line of ``note`` wider than the figure at word boundaries; a line with mathtext or one longer word stays whole."""
    width_in = _note_width_in(figure, note)
    if width_in <= figure.get_figwidth():
        return note
    lines = note.split("\n")
    # A mathtext line (``$`` keeps it unsplit, since a break inside it fails to render) or a single word cannot wrap; the
    # widest of them widens the figure anyway, so the other lines may fill that width instead of shrinking to single words.
    unbreakable = [line for line in lines if "$" in line] + [word for line in lines if "$" not in line for word in line.split()]
    limit_in = max(figure.get_figwidth(), _note_width_in(figure, "\n".join(unbreakable)))
    # The widest line gives a first mean character width; glyph widths differ, so the columns shrink until the rendered
    # note fits.
    columns = max(1, int(max(len(line) for line in lines) * limit_in / width_in))
    while True:
        wrapped = "\n".join(line if "$" in line else textwrap.fill(line, columns, break_long_words=False, break_on_hyphens=False) for line in lines)
        if columns == 1 or _note_width_in(figure, wrapped) <= limit_in:
            return wrapped
        columns = max(1, int(columns * _WRAP_SHRINK))


def panel_engine() -> ConstrainedLayoutEngine:
    """Return a new constrained layout engine with the panel gap of ``panel_figure``, for figures that build their own gridspec."""
    return ConstrainedLayoutEngine(w_pad=PANEL_PAD_IN, h_pad=PANEL_PAD_IN, wspace=0.0, hspace=0.0)


def panel_figure(
    rows: int,
    columns: int,
    *,
    panel_in: tuple[float, float] = (4.2, 3.0),
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    sharex: bool = False,
    sharey: Literal["none", "row", "all"] = "none",
) -> tuple[Figure, np.ndarray]:
    """
    Return a constrained figure of ``rows`` x ``columns`` panels of the fixed cell size ``panel_in`` and the shared text sizes.

    The cell is the panel with its tick and axis labels; ``width_ratios`` and ``height_ratios`` scale single columns and rows
    relative to it. Shared axes hide the inner tick labels. The axes come back as a two-dimensional array whose tick labels, axis
    labels already have ``TICK_PT`` and ``LABEL_PT``, also for text set later; titles go through ``set_panel_title``.
    """
    columns_in = panel_in[0] * (columns if width_ratios is None else sum(width_ratios))
    rows_in = panel_in[1] * (rows if height_ratios is None else sum(height_ratios))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns_in, rows_in),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        layout=panel_engine(),
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )
    apply_text_sizes(axes.flat)
    return figure, axes


def apply_text_sizes(axes: Iterable[Axes]) -> None:
    """Give the tick labels and axis labels of ``axes`` the shared sizes, also for label text set later."""
    for axis in axes:
        axis.tick_params(labelsize=TICK_PT)
        axis.xaxis.label.set_fontsize(LABEL_PT)
        axis.yaxis.label.set_fontsize(LABEL_PT)


def _mantissa_exponent(value: float) -> tuple[float, int]:
    """Split a positive ``value`` into its mantissa in [1, 10) and its power of ten."""
    exponent = int(np.floor(np.log10(value)))
    return value / 10.0**exponent, exponent


def _labelled_minor_ticks(low: float, high: float, ticks: Iterable[float]) -> np.ndarray:
    """
    Return the minor ``ticks`` a log axis labels in the view from ``low`` to ``high``.

    Every decade in the view counts as labelled. A minor tick joins in order of preference (mantissa 5, then 2, 3, 4, 6 to 9,
    then any other by position: the linear ticks Matplotlib places in a view without such a mantissa, as from 1.1 to 1.2) and
    only at ``_MINOR_LABEL_GAP`` of the view's span or more from every labelled tick, so a view inside one decade still gets
    several labels and a view over many decades gets none. A view that would keep fewer than two labels (few ticks, as from
    8 to 16) is chosen again at half the gap.
    """
    if not 0.0 < low < high:
        return np.empty(0)
    log_low, log_high = float(np.log10(low)), float(np.log10(high))

    def preference(tick: float) -> tuple[int, float]:
        mantissa = _mantissa_exponent(tick)[0]
        ranks = [rank for rank, preferred in enumerate(_MINOR_LABEL_ORDER) if np.isclose(mantissa, preferred)]
        return (ranks[0] if ranks else len(_MINOR_LABEL_ORDER)), tick

    candidates = sorted((float(tick) for tick in ticks if low <= tick <= high), key=preference)
    decades = [float(decade) for decade in range(int(np.ceil(log_low)), int(np.floor(log_high)) + 1)]
    for share in (_MINOR_LABEL_GAP, _MINOR_LABEL_GAP / 2.0):
        gap = share * (log_high - log_low)
        taken, labelled = list(decades), []
        for tick in candidates:
            position = float(np.log10(tick))
            if all(abs(position - other) >= gap for other in taken):
                taken.append(position)
                labelled.append(tick)
        if len(taken) >= _MINIMUM_LOG_LABELS:
            break
    return np.asarray(labelled)


def set_log_scale(axis: Axes, which: Literal["x", "y"]) -> None:
    """Put one axis on a log scale whose tick labels never overlap.

    Decades are labelled as ``LogFormatterSciNotation`` does; minor ticks are labelled as ``_labelled_minor_ticks`` chooses
    for the view and ticks read at draw time, as mantissa times a power of ten, with three significant digits or as many
    more as keep the labels distinct. Matplotlib's own
    minor formatter is not used: it rounds the mantissa to an integer and keeps a tick subset that can drop 5.
    """
    data_axis = axis.xaxis if which == "x" else axis.yaxis
    get_limits = axis.get_xlim if which == "x" else axis.get_ylim
    (axis.set_xscale if which == "x" else axis.set_yscale)("log")

    def _minor_label(value: float, _pos: int | None = None) -> str:
        low, high = sorted(get_limits())
        labelled = _labelled_minor_ticks(low, high, data_axis.get_minorticklocs())
        if value not in set(labelled.tolist()):  # the axis formats exactly the locator's values, so a near tick never joins
            return ""
        # Three significant digits, or as many more as keep the labels of a very narrow view distinct.
        digits = next((n for n in range(3, _MAX_LABEL_DIGITS) if len({f"{tick:.{n}g}" for tick in labelled}) == len(labelled)), _MAX_LABEL_DIGITS)
        mantissa, exponent = _mantissa_exponent(value)
        return rf"$\mathdefault{{{mantissa:.{digits}g}\times10^{{{exponent}}}}}$"

    data_axis.set_major_formatter(LogFormatterSciNotation())
    data_axis.set_minor_formatter(FuncFormatter(_minor_label))


def set_panel_title(axis: Axes, text: str, **kwargs: Any) -> Text:
    """Set the title of one panel at ``PANEL_TITLE_PT`` (Matplotlib's ``set_title`` would reset it to its rc default)."""
    kwargs.setdefault("fontsize", PANEL_TITLE_PT)
    return axis.set_title(text, **kwargs)


def legend_entries(axes: Iterable[Axes]) -> tuple[list[Artist], list[str]]:
    """Return the legend handles and labels of ``axes`` in drawing order; of a repeated label the first handle is kept."""
    entries: dict[str, Artist] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    return list(entries.values()), list(entries)


def add_legend(
    figure: Figure,
    axes: np.ndarray,
    handles: Sequence[Artist],
    labels: Sequence[str],
    *,
    title: str | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    outside: bool = False,
) -> None:
    """
    Write the legend of ``axes``: inside the only visible panel, else outside right of the rightmost panel, top at the top frame.

    ``outside`` places the legend of a single panel outside right as well, for a figure the owner's spec names so. Of entries
    with a repeated label the first is kept. ``groups`` maps a group title to its labels and stacks one titled legend per
    group below each other in the column (inside the single panel: one legend with the entries in group order); labels no group lists
    form a final untitled group. ``title`` titles an ungrouped legend. The column is as wide as its longest entry: the figure grows
    by that width, and by the missing height when the column is taller than the panels, while the panels keep their size.
    """
    entries: dict[str, Artist] = {}
    for label, handle in zip(labels, handles, strict=True):
        entries.setdefault(label, handle)
    if not entries:
        message = "A legend needs at least one entry; a view without series draws none."
        raise ValueError(message)
    visible = [axis for axis in _grid(axes).flat if axis.get_visible()]
    if not visible:
        message = "A legend needs at least one visible panel."
        raise ValueError(message)
    blocks = _blocks(entries, title, groups)
    if len(visible) == 1 and not outside:
        # One block is a plain list (its name titles the legend); several keep their titles as bold heading rows inside the panel,
        # a gap row between the blocks.
        headed = len(blocks) > 1
        flat_handles: list[Artist] = []
        flat_labels: list[str] = []
        heading_rows: list[int] = []
        for position, (name, names) in enumerate(blocks):
            spacer = Line2D([], [], linestyle="none")
            if headed and position:
                flat_handles.append(spacer)
                flat_labels.append(" ")
            if headed and name is not None:
                heading_rows.append(len(flat_labels))
                flat_handles.append(spacer)
                flat_labels.append(name)
            flat_handles.extend(entries[label] for label in names)
            flat_labels.extend(names)
        legend_title = blocks[0][0] if not headed and title is None else title
        legend = visible[0].legend(flat_handles, flat_labels, loc="best", title=legend_title, fontsize=TICK_PT, title_fontsize=TICK_PT)
        for row in heading_rows:
            legend.get_texts()[row].set_fontweight("bold")
        return
    figure.draw_without_rendering()  # the layout runs now, so the positions below are final
    top = max(visible, key=lambda axis: axis.get_position().y1)
    right = max(visible, key=lambda axis: axis.get_position().x1)
    anchor = blended_transform_factory(right.transAxes, top.transAxes)
    tight_right = max(box.x1 for axis in figure.axes if axis.get_visible() and (box := axis.get_tightbbox()) is not None)  # twin axes included
    decorations_in = max(0.0, tight_right - right.get_window_extent().x1) / figure.dpi
    legends = [
        figure.legend(
            [entries[label] for label in names],
            names,
            title=name,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=TICK_PT,
            title_fontsize=TICK_PT,
        )
        for name, names in blocks
    ]
    extents = [legend.get_window_extent() for legend in legends]
    offset_in = 0.0
    for legend, extent in zip(legends, extents, strict=True):
        legend.set_bbox_to_anchor(
            (1.0, 1.0),
            transform=anchor + ScaledTranslation(decorations_in + LEGEND_GAP_IN, -offset_in, figure.dpi_scale_trans),
        )
        offset_in += extent.height / figure.dpi + LEGEND_STACK_GAP_IN
    column_in = offset_in - LEGEND_STACK_GAP_IN
    panels_in = (top.get_window_extent().y1 - min(axis.get_window_extent().y0 for axis in visible)) / figure.dpi
    _grow(figure, right_in=LEGEND_GAP_IN + max(extent.width for extent in extents) / figure.dpi, bottom_in=max(0.0, column_in - panels_in))


def label_rows_once(axes: np.ndarray, ylabel: str) -> None:
    """Write ``ylabel`` on the first visible panel of every row and clear it on the others; tick labels stay where they are."""
    for row in _grid(axes):
        first = next((column for column, axis in enumerate(row) if axis.get_visible()), None)
        for column, axis in enumerate(row):
            axis.set_ylabel(ylabel if column == first else "", fontsize=LABEL_PT)


def set_row_headers(figure: Figure, axes: np.ndarray, headers: Sequence[str]) -> None:
    """
    Write one bold header per row, left aligned above the row's first visible panel and its panel title.

    A panel title of several lines in the row (set before this call) lifts the header by its extra lines.
    """
    for row, header in zip(_grid(axes), headers, strict=True):
        axis = next((candidate for candidate in row if candidate.get_visible()), None)
        if axis is not None:
            extra_lines = max(candidate.get_title().count("\n") for candidate in row if candidate.get_visible())
            rise_in = ROW_HEADER_RISE_IN + extra_lines * PANEL_TITLE_PT * _LINE_HEIGHT / _POINTS_PER_INCH
            axis.text(
                0.0,
                1.0,
                header,
                transform=axis.transAxes + ScaledTranslation(0.0, rise_in, figure.dpi_scale_trans),
                fontsize=PANEL_TITLE_PT,
                fontweight="bold",
                ha="left",
                va="bottom",
            )


def set_figure_title(figure: Figure, axes: np.ndarray, title: str, *, note: str | None = None) -> None:
    """
    Write the bold figure title and, a small gap below it, its smaller note, both centred over the visible panels ``axes``.

    The figure grows by the strip of both texts, which stay anchored to its top edge in inches; the panels keep their size. A text
    wider than the panels shifts inward off the figure edge. A note line wider than the figure wraps at word boundaries; a title,
    or a note line that cannot wrap, wider than the figure widens it. A second call raises ``ValueError``.
    """
    if any(isinstance(artist, _CenteredText) for artist in figure.artists):
        message = "The figure title is already set."
        raise ValueError(message)
    if note is not None:
        note = _wrap_note(figure, note)
    title_in = TITLE_PT * _LINE_HEIGHT * (title.count("\n") + 1) / _POINTS_PER_INCH
    note_in = 0.0 if note is None else (NOTE_GAP_PT + NOTE_PT * _LINE_HEIGHT * (note.count("\n") + 1)) / _POINTS_PER_INCH
    panels = tuple(_grid(axes).flat)
    _grow(figure, top_in=title_in + note_in)
    widest_in = 0.0
    for text, size, weight, drop_in in ((title, TITLE_PT, "bold", 0.0), (note, NOTE_PT, "normal", title_in + NOTE_GAP_PT / _POINTS_PER_INCH)):
        if text is not None:
            artist = _CenteredText(panels, 0.5, 1.0, text, fontsize=size, fontweight=weight, ha="center", va="top")
            figure.add_artist(artist)
            artist.set_transform(figure.transSubfigure + ScaledTranslation(0.0, -drop_in, figure.dpi_scale_trans))
            widest_in = max(widest_in, artist.get_window_extent().width / figure.dpi)
    _grow(figure, right_in=max(0.0, widest_in - figure.get_figwidth()))
