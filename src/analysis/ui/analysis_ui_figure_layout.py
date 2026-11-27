"""Own the geometry, text sizes, legends and titles of line, bar and distribution figures.

Responsibilities:
  - Build constrained panel grids whose panels have one fixed size, one shared gap and the shared text sizes
  - Put an axis on a log scale with non-overlapping tick labels (``set_log_scale``)
  - Collect the legend entries of a figure's axes
  - Place the legend of one panel in one of its corners, clear of the data of the panel and its twin axes through a free
    corner, headroom, more key columns or a larger panel, else outside right of it, and the legend of several panels outside
    right, top aligned with the top plot frame
  - Write the y label once per row, row headers above their row's titles and scientific factors, and the figure title with
    its note over the data axes

Design principles:
  - The text sizes are the ones the map figures use (``analysis_ui_plot_layout`` imports them), so every figure type reads alike
  - Every bold heading writes its symbols bold and upright (``bold_label``), the row headers here and the map headers alike
  - A legend outside the panels grows the figure; it never shrinks or moves a panel, and no entry wraps
  - A legend group owns its entries by identity, never by text or position, so equal texts of two groups keep their own handles
  - Coverage is judged from what the panel draws in display coordinates; headroom is runtime state of the axis it extends, so
    a key added again starts from the limits before it, and autoscaling stays as it was
  - Headroom keeps the scale truthful, inferred from the axes and their data: an axis without a negative limit never extends
    below zero, nor one of non-negative data past the margin autoscaling gave it below zero, whichever way it runs, a parity
    plot (equal x and y limits) extends both axes together, an axis shared with other axes never changes (the other axis
    does), y, the value axis, widens before x, and the data in view before keep 60 % of every widened axis over all passes;
    a line marked ``REFERENCE_GID`` (a parity or target diagonal) stays clear of the key without counting as data for the 60 %
  - Beyond that the key is built again in more columns that fit the panel, a grouped key with whole groups per column, then
    the panel grows once, at most to twice its size each way, and else the key stands outside right as for several panels,
    so it never stays over data; a key being placed takes no part in the layout
  - A long note wraps to the figure's width, again when a legend widens it, instead of widening the figure past its panels;
    a legend first fills the strip a wide title or note reserved, so the order of the two calls never changes the size
  - Title and note keep an inset from the figure edges and the place a tight export measured, so no export clips them; the
    strip above the panels holds the note at its rendered height, mathtext lines taller, with a clearance to the panel titles
  - Geometry is derived from the panel grid and the artists, never from the number of curves

This module does NOT:
  - Choose colours, line styles or the entities a legend names
  - Draw maps or colour bars; ``analysis_ui_plot_layout`` owns those
  - Own scientific labels or units
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import Collection, LineCollection
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
from matplotlib.transforms import Bbox, ScaledTranslation, blended_transform_factory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

TICK_PT = 9.0
LABEL_PT = 10.0
PANEL_TITLE_PT = 11.0
TITLE_PT = 12.0
NOTE_PT = 10.0

PANEL_PAD_IN = 0.03  # padding around every panel in inches; Matplotlib's default is 3/72 (0.042)
LEGEND_GAP_IN = 0.15  # panels to the legend column
LEGEND_STACK_GAP_IN = 0.1  # between stacked legend groups
REFERENCE_GID = "reference-line"  # gid of a parity or target diagonal: a single panel's key clears it, the data's 60 % leave it out
ROW_HEADER_GAP_PT = 4.0  # a row header's bottom above its row's frames, panel titles and y-axis offset texts
ROW_HEADER_CLEARANCE_IN = 0.12  # room the layout keeps free above a row header, so it reads with its own row
NOTE_GAP_PT = 2.0  # title to its note, smaller than a full line
_NOTE_CLEARANCE_PT = 4.0  # the note's last line to the panel titles below it
_TITLE_INSET_IN = 0.12  # figure edges to the title and its note, as on the map figures, so no ink touches the export boundary
_KEY_CLEARANCE_PT = 3.0  # least distance of a single panel's key from the data it must not cover, about a marker's radius
_KeyCorner = Literal["upper right", "upper left", "lower left", "lower right"]
_KEY_CORNERS: tuple[_KeyCorner, ...] = ("upper right", "upper left", "lower left", "lower right")  # Matplotlib's order of preference
_KEY_EDGE_PX = 0.25  # data on the padded key's edge, where headroom puts them, do not cover it
_MIN_DATA_FILL = 0.6  # share of a widened axis the data it showed before still fill
_ZERO_TOLERANCE = 1e-9  # share of an axis's span by which a zero value may read below zero after the display round trip
_GROWTH_SLACK = 1.02  # a grown panel exceeds the size its key needs by this factor, so tick labels the new layout changes keep it clear
_MAX_GROWTH = 2.0  # a single panel grows for its key at most by this factor each way; beyond it the key stands outside right
_HEADROOM = "_key_headroom"  # runtime attribute of an axes: per widened axis (0 x, 1 y) its limits before and after its panel's key
type _Drawn = tuple[Axes, Path, bool, bool]  # an axis, a path it draws in display coordinates, whether connected, whether filled
type _Span = tuple[float, float]  # an axis's data extent in view in scale units before the key, and the lowest limit the key may give it
type _Widening = tuple[float, float, tuple[int, ...]]  # the panel growth, the span factor, the axes (0 x, 1 y) that move, the key's first
_LINE_HEIGHT = 1.25  # text line height in font sizes
_WRAP_SHRINK = 0.9  # columns kept per retry while a wrapped note is still wider than the figure
_MINOR_LABEL_ORDER = (5.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0)  # mantissas a log axis may label, most preferred first
_MINOR_LABEL_GAP = 0.3  # least distance of two labelled log ticks, as a share of the view's span in decades
_MINIMUM_LOG_LABELS = 2  # labels a log view should keep; below it the gap halves once
_MAX_LABEL_DIGITS = 17  # significant digits that tell any two distinct float64 values apart
_POINTS_PER_INCH = 72.0
_GRID_NDIM = 2  # rows x columns
_MATH_SPAN = re.compile(r"\$([^$]+)\$")


class _CenteredText(Text):
    """Text centred over the visible panels of its figure, recomputed at every draw so a later layout change never leaves it stale."""

    def __init__(self, panels: Sequence[Axes], *args: Any, unwrapped: str | None = None, **kwargs: Any) -> None:
        """Keep the panels to centre over and, for a note, its text before wrapping (``_fit_note``)."""
        super().__init__(*args, **kwargs)
        self._panels = tuple(panels)
        self.unwrapped = unwrapped

    def draw(self, renderer: RendererBase) -> None:
        """Centre over the final positions, then draw."""
        positions = [axis.get_position() for axis in self._panels if axis.get_visible()]
        figure = self.get_figure(root=True)
        if positions and figure is not None:
            centre = (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2.0
            # The figure's width in pixels from its fractions: a tight export crops ``figure.bbox`` for the final draw but keeps
            # that scale, so the text stays where the tight bounding box measured it.
            width = figure.transFigure.transform((1.0, 0.0))[0] - figure.transFigure.transform((0.0, 0.0))[0]
            half = min(0.5, (self.get_window_extent(renderer).width / 2.0 + _TITLE_INSET_IN * figure.dpi) / width)
            self.set_x(min(max(centre, half), 1.0 - half))  # shifted inward only where the text would pass the edge's inset
        super().draw(renderer)


class _RowHeader(Text):
    """
    A row header placed, whenever the figure is laid out or drawn, over the frames, panel titles and y-axis offset texts of its row.

    An offset text is the scientific factor of a y axis, such as 1e8. The layout keeps ``ROW_HEADER_CLEARANCE_IN`` free
    above the header, so it reads with its own row.
    """

    def __init__(self, row: Sequence[Axes], *args: Any, **kwargs: Any) -> None:
        """Keep the visible panels of the row; the header stands on the first one's top frame, raised by ``_rise``."""
        super().__init__(*args, **kwargs)
        self._row = tuple(row)

    def _rise(self, renderer: RendererBase | None) -> None:
        """Raise the header from the top frame by the highest panel title or y-axis offset text of the row, in axes fractions."""
        for axis in self._row:  # a panel laid out or drawn later still holds the offset text of its last place
            axis.yaxis.get_tightbbox(renderer)  # brings it up to date, as Matplotlib does before it places a title
        texts = [text for axis in self._row for text in (axis.title, axis.yaxis.get_offset_text()) if text.get_visible() and text.get_text()]
        frame = self._row[0].get_window_extent(renderer)
        self.set_y(1.0 + max([0.0, *(text.get_window_extent(renderer).y1 - frame.y1 for text in texts)]) / frame.height)

    def draw(self, renderer: RendererBase) -> None:
        """Place the header over the row's final decorations, then draw."""
        self._rise(renderer)
        super().draw(renderer)

    def get_tightbbox(self, renderer: RendererBase | None = None) -> Bbox | None:
        """Return the header's box at its place, raised at its top by the clearance."""
        self._rise(renderer)
        box = super().get_tightbbox(renderer)
        figure = self.get_figure(root=True)
        return box if box is None or figure is None else Bbox.from_extents(box.x0, box.y0, box.x1, box.y1 + ROW_HEADER_CLEARANCE_IN * figure.dpi)


def _first(pairs: Iterable[tuple[str, Artist]]) -> dict[str, Artist]:
    """Return the (label, handle) ``pairs`` as entries in order, of a repeated label the first handle."""
    entries: dict[str, Artist] = {}
    for label, handle in pairs:
        entries.setdefault(label, handle)
    return entries


def _owned(pairs: Sequence[tuple[str, Artist]], members: Sequence[Artist]) -> list[tuple[str, Artist]]:
    """
    Return the (label, handle) entries a group lists by their handles; a member without an entry adds nothing.

    A label is no member (``TypeError``): equal text can name two entries, so only the handle identifies one.
    """
    owned: list[tuple[str, Artist]] = []
    for member in members:
        if isinstance(member, str):
            message = f"A legend group lists its entries by their handles, not by the label {member!r}."
            raise TypeError(message)
        owned += [pair for pair in pairs if pair[1] is member][:1]
    return owned


def _blocks(
    pairs: Sequence[tuple[str, Artist]], title: str | None, groups: Mapping[str, Sequence[Artist]] | None
) -> list[tuple[str | None, dict[str, Artist]]]:
    """
    Return the legend blocks as (title, entries): one block, or the groups followed by the entries no group owns.

    A group owns the entries it lists (``_owned``), never by position, so two groups each keep an entry of the same text with
    its own handle (a model named like a material) in any order of the pairs; a repeated label merges within a block only.
    """
    if groups is None:
        return [(title, _first(pairs))]
    owned = {name: _owned(pairs, members) for name, members in groups.items()}
    taken = {id(handle) for entries in owned.values() for _label, handle in entries}
    blocks: list[tuple[str | None, list[tuple[str, Artist]]]] = [*owned.items(), (None, [pair for pair in pairs if id(pair[1]) not in taken])]
    return [(name, _first(entries)) for name, entries in blocks if entries]


def _stack_rows(stack: Sequence[tuple[str | None, dict[str, Artist]]]) -> int:
    """Return the rows a key column of the blocks ``stack`` takes: each block's heading and entries, and a gap row between two."""
    return sum(len(entries) + (name is not None) for name, entries in stack) + len(stack) - 1


def _stacks(blocks: Sequence[tuple[str | None, dict[str, Artist]]], columns: int) -> list[list[tuple[str | None, dict[str, Artist]]]]:
    """
    Split ``blocks`` in order into ``columns`` (at most their number) key columns of whole blocks, the tallest as short as it can be.

    Each height from the tallest block up fills the columns greedily: a column takes the next block while it stays within that
    height, and a block opens a column early where only so every later column still gets one. The first height no column
    exceeds is the least, found in polynomial time.
    """
    height = max(_stack_rows([block]) for block in blocks)
    while True:
        stacks: list[list[tuple[str | None, dict[str, Artist]]]] = []
        for index, block in enumerate(blocks):
            if not stacks or len(blocks) - index <= columns - len(stacks) or (len(stacks) < columns and _stack_rows([*stacks[-1], block]) > height):
                stacks.append([])
            stacks[-1].append(block)
        if max(map(_stack_rows, stacks)) <= height:
            return stacks
        height += 1


def _key_rows(blocks: Sequence[tuple[str | None, dict[str, Artist]]], columns: int) -> tuple[list[Artist], list[str], list[int]]:
    """
    Return the handles, labels and bold heading rows of the key of one panel in ``columns`` columns, filled column by column.

    One block is a plain list, which Matplotlib spreads over the columns. Several keep their titles as heading rows and a gap row
    between two blocks of a column, and each column stacks whole blocks in order (``_stacks``), padded with blank rows to the
    tallest, so a heading always tops its own entries.
    """
    if len(blocks) == 1:
        return list(blocks[0][1].values()), list(blocks[0][1]), []
    spacer = Line2D([], [], linestyle="none")
    stacks = _stacks(blocks, columns)
    height = max(map(_stack_rows, stacks))
    rows: list[tuple[Artist, str]] = []  # (handle, label), column by column
    headings: list[int] = []
    for stack in stacks:
        column: list[tuple[Artist, str]] = []
        for position, (name, entries) in enumerate(stack):
            if position:
                column.append((spacer, " "))
            if name is not None:
                headings.append(len(rows) + len(column))
                column.append((spacer, name))
            column += [(handle, label) for label, handle in entries.items()]
        rows += column + [(spacer, " ")] * (height - len(column))
    return [handle for handle, _label in rows], [label for _handle, label in rows], headings


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


def _grow(figure: Figure, *, right_in: float = 0.0, top_in: float = 0.0, bottom_in: float = 0.0, panels_in: tuple[float, float] = (0.0, 0.0)) -> None:
    """Add a strip right of, above and below the panels' rectangle and ``panels_in`` (width, height) to it; the other strips keep their inches."""
    engine = _engine(figure)
    width, height = figure.get_size_inches()
    new_width, new_height = width + right_in + panels_in[0], height + top_in + bottom_in + panels_in[1]
    left, bottom, rect_width, rect_height = engine.get()["rect"]  # Matplotlib's rect is (left, bottom, width, height)
    engine.set(
        rect=(
            left * width / new_width,
            (bottom * height + bottom_in) / new_height,
            (rect_width * width + panels_in[0]) / new_width,
            (rect_height * height + panels_in[1]) / new_height,
        )
    )
    figure.set_size_inches(new_width, new_height)


def _note_size_in(figure: Figure, note: str) -> tuple[float, float]:
    """Return the rendered width of the widest line of ``note`` and its height, in inches at the note size; mathtext lines are taller."""
    probe = figure.text(0.0, 0.0, note, fontsize=NOTE_PT)
    try:
        box = probe.get_window_extent()
        return box.width / figure.dpi, box.height / figure.dpi
    finally:
        probe.remove()


def _wrap_note(figure: Figure, note: str) -> str:
    """Wrap every line of ``note`` wider than the figure inside its insets at word boundaries; mathtext or one longer word stays whole."""
    room_in = figure.get_figwidth() - 2.0 * _TITLE_INSET_IN
    width_in = _note_size_in(figure, note)[0]
    if width_in <= room_in:
        return note
    lines = note.split("\n")
    # A mathtext line (``$`` keeps it unsplit, since a break inside it fails to render) or a single word cannot wrap; the
    # widest of them widens the figure anyway, so the other lines may fill that width instead of shrinking to single words.
    unbreakable = [line for line in lines if "$" in line] + [word for line in lines if "$" not in line for word in line.split()]
    limit_in = max(room_in, _note_size_in(figure, "\n".join(unbreakable))[0])
    # The widest line gives a first mean character width; glyph widths differ, so the columns shrink until the rendered
    # note fits.
    columns = max(1, int(max(len(line) for line in lines) * limit_in / width_in))
    while True:
        wrapped = "\n".join(line if "$" in line else textwrap.fill(line, columns, break_long_words=False, break_on_hyphens=False) for line in lines)
        if columns == 1 or _note_size_in(figure, wrapped)[0] <= limit_in:
            return wrapped
        columns = max(1, int(columns * _WRAP_SHRINK))


def _fit_note(figure: Figure) -> None:
    """
    Wrap the note of the figure title to the figure's current width and resize the strip above the panels by the rendered height it gains or loses.

    A note line that cannot wrap and is wider than the figure inside its insets widens it.
    """
    for note in figure.artists:
        if isinstance(note, _CenteredText) and note.unwrapped is not None:
            before_in = _note_size_in(figure, note.get_text())[1]
            note.set_text(_wrap_note(figure, note.unwrapped))
            width_in, height_in = _note_size_in(figure, note.get_text())
            _grow(figure, right_in=max(0.0, width_in + 2.0 * _TITLE_INSET_IN - figure.get_figwidth()), top_in=height_in - before_in)


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


def _drawn_data(axes: Sequence[Axes], *, references: bool = True) -> list[_Drawn]:
    """
    Return what ``axes`` draw in data coordinates, as paths in display coordinates with their axis.

    Lines, bars, boxes, error bars and bands are connected paths (bars, boxes and bands filled); markers and scatter points
    are the unconnected vertices of a path. Artists placed partly in axes coordinates (``axhline``, ``axvspan``) and texts
    are not data a key must clear. A line marked ``REFERENCE_GID`` counts through its own transform, so an ``axline`` diagonal,
    which spans the view, is cleared too; ``references`` False leaves the marked artists out.
    """
    drawn: list[_Drawn] = []
    for owner in axes:
        data = owner.transData
        for artist in owner.get_children():
            reference = artist.get_gid() == REFERENCE_GID
            if not artist.get_visible() or (not references and reference):
                continue
            if isinstance(artist, Line2D) and (artist.get_transform() is data or reference):
                drawn.append((owner, artist.get_transform().transform_path(artist.get_path()), artist.get_linestyle() != "None", False))
            elif isinstance(artist, Patch) and artist.get_data_transform() is data:
                drawn.append((owner, artist.get_transform().transform_path(artist.get_path()), True, artist.get_fill()))
            elif isinstance(artist, Collection) and artist.get_offset_transform() is data:
                drawn.append((owner, Path(data.transform(np.asarray(artist.get_offsets(), dtype=float).reshape(-1, 2))), False, False))
            elif isinstance(artist, Collection) and artist.get_transform() is data:
                drawn.extend((owner, data.transform_path(path), True, not isinstance(artist, LineCollection)) for path in artist.get_paths())
    return drawn


def _covers(box: Bbox, drawn: Sequence[_Drawn]) -> bool:
    """Return whether data lie under the display ``box``: a point inside it or a path crossing it; data on its edge keep the clearance."""
    inner = box.padded(-_KEY_EDGE_PX)
    return any(
        path.intersects_bbox(inner, filled=filled) if connected else inner.count_contains(path.vertices) > 0 for _, path, connected, filled in drawn
    )


def _heights(box: Bbox, drawn: Sequence[_Drawn], owner: Axes, along: int = 1) -> np.ndarray:
    """Return the display coordinates along ``along`` (0 x, 1 y) of what ``owner`` draws in ``box``'s other span: its vertices and crossings there."""
    across = 1 - along
    low, high = (box.intervalx, box.intervaly)[across]
    parts = [np.empty(0)]
    for _, path, connected, _filled in (item for item in drawn if item[0] is owner):
        for polyline in path.to_polygons(closed_only=False) if connected else [path.vertices]:
            line = np.asarray(polyline, dtype=float).reshape(-1, 2)
            parts.append(line[(line[:, across] >= low) & (line[:, across] <= high), along])
            if connected:  # the share of each segment at which it meets either side; outside 0 to 1 (or undefined) it misses that side
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = (np.array([[low], [high]]) - line[:-1, across]) / np.diff(line[:, across])
                    parts.append((line[:-1, along] + share * np.diff(line[:, along]))[(share >= 0.0) & (share <= 1.0)])
    heights = np.concatenate(parts)
    return heights[np.isfinite(heights)]


def _shown(frame: Bbox, path: Path, connected: bool) -> np.ndarray:
    """Return display points that bound what ``path`` draws inside ``frame``: its markers there, or where its segments enter and leave it."""
    low, high = frame.get_points()
    if not connected:
        points = np.asarray(path.vertices, dtype=float).reshape(-1, 2)
        return points[np.all((points >= low) & (points <= high), axis=1)]
    parts = [np.empty((0, 2))]
    for polyline in path.to_polygons(closed_only=False):
        line = np.asarray(polyline, dtype=float).reshape(-1, 2)
        start, step = line[:-1], np.diff(line, axis=0)
        parallel = np.where((start >= low) & (start <= high), np.inf, -np.inf)  # a segment parallel to two sides runs between them or never
        with np.errstate(divide="ignore", invalid="ignore"):
            meets = (np.stack([low, high])[:, None, :] - start) / step  # per side, segment and axis the share at which they meet
        enter = np.where(step == 0.0, -parallel, meets.min(axis=0)).max(axis=1).clip(0.0)
        leave = np.where(step == 0.0, parallel, meets.max(axis=0)).min(axis=1).clip(max=1.0)
        kept = enter <= leave
        parts += [start[kept] + share[kept, None] * step[kept] for share in (enter, leave)]
    return np.concatenate(parts)


def _limits(owner: Axes, along: int) -> tuple[float, float]:
    """Return the view limits of ``owner`` along ``along`` (0 x, 1 y)."""
    return (owner.get_xlim, owner.get_ylim)[along]()


def _extent(frame: Bbox, drawn: Sequence[_Drawn], owner: Axes, along: int) -> float:
    """Return the span in scale units along ``along`` (0 x, 1 y) of what ``owner`` draws inside the panel ``frame``; data out of view never count."""
    points = np.concatenate([np.empty((0, 2)), *(_shown(frame, path, connected) for item, path, connected, _filled in drawn if item is owner)])
    ends = (owner.xaxis, owner.yaxis)[along].get_transform().transform(np.asarray(_limits(owner, along), dtype=float))
    return float(np.ptp(points[:, along]) / (frame.width, frame.height)[along] * abs(ends[1] - ends[0])) if len(points) else 0.0


def _shared(owner: Axes, along: int) -> bool:
    """Return whether ``owner`` shares its axis ``along`` (0 x, 1 y) with other axes, whose view a change would move as well."""
    return len((owner.get_shared_x_axes, owner.get_shared_y_axes)[along]().get_siblings(owner)) > 1


def _span(frame: Bbox, data: Sequence[_Drawn], owner: Axes, along: int) -> _Span:
    """
    Return the extent in view (``_extent``) of the ``data`` of ``owner`` along ``along`` (0 x, 1 y) and the lowest limit a key may give that axis.

    The floor is zero for an axis without a negative limit, the lower limit itself where all data are non-negative but an
    autoscale margin already reaches below zero, and unbounded for data with a negative value.
    """
    limits = _limits(owner, along)
    to_data = owner.transData.inverted()
    values = np.concatenate([np.empty(0), *(to_data.transform(path.vertices)[:, along] for item, path, *_ in data if item is owner)])
    values = values[np.isfinite(values)]
    non_negative = min(limits) >= 0.0 or (values.size > 0 and values.min() >= -_ZERO_TOLERANCE * abs(limits[1] - limits[0]))
    return _extent(frame, data, owner, along), (min(*limits, 0.0) if non_negative else -np.inf)


def _cap(owner: Axes, along: int, high: bool, span: _Span) -> float:
    """
    Return the largest factor by which the key may grow the span of ``owner``'s axis ``along`` (0 x, 1 y) on its ``high`` or low side.

    Spans are in scale units. The data that spanned the extent of ``span`` before the key keep ``_MIN_DATA_FILL`` of the widened
    span, and the limit never passes the floor of ``span`` (``_span``): the limit that moves is the one on the key's side, which on
    an inverted axis is the numerically lower one on the high side. At least 1, which leaves the span as it is.
    """
    extent, floor = span
    limits = _limits(owner, along)
    scale = (owner.xaxis, owner.yaxis)[along].get_transform()
    fixed, moving = scale.transform(np.asarray(limits if high else limits[::-1], dtype=float))
    cap = extent / (_MIN_DATA_FILL * abs(moving - fixed))
    if np.isfinite(floor) and moving < fixed:  # the limit moves down toward the floor, which it may reach but not pass
        cap = min(cap, (fixed - scale.transform(floor)) / (fixed - moving))
    return max(1.0, float(cap))


def _widening(box: Bbox, frame: Bbox, heights: np.ndarray, along: int, high: bool, cap: float) -> tuple[float, float]:
    """
    Return the panel growth and the span factor along ``along`` (0 x, 1 y) for data at the display ``heights`` to clear the key ``box``.

    The factor is how much the span, in scale units, must grow on the key's ``high`` or low side of the panel ``frame``; 1 means
    the data clear it already, and data reaching far past the panel's edge there, as bars toward the bottom of a log axis, need
    one beyond any ``cap``. The growth is 1 within ``cap``, else the factor by which the panel's length must grow until the needed
    factor falls to ``cap``, the data keeping their axes fractions and the key its size; infinite where no length suffices.
    """
    if not heights.size:
        return 1.0, 1.0
    low, length = frame.p0[along], frame.p1[along] - frame.p0[along]
    reach, room = (heights.max() - low, box.p0[along] - low) if high else (low + length - heights.min(), low + length - box.p1[along])
    factor = max(1.0, float(reach / room)) if room > 0.0 else np.inf
    if factor <= cap:
        return 1.0, factor
    key = length - room  # grown by g, the panel needs the factor g * reach / (g * length - key)
    return (float(cap * key / (cap * length - reach)) if cap * length > reach else np.inf), factor


def _key_needs(
    figure: Figure, axis: Axes, legend: Legend, corner: _KeyCorner, drawn: Sequence[_Drawn], spans: Mapping[tuple[Axes, int], _Span]
) -> dict[Axes, _Widening]:
    """
    Return per axes whose data lie under the key ``legend`` in its ``corner`` how its view must widen to clear them; empty where none do.

    Of y and x, each where no other axes share it (``_shared``; a parity plot, with equal x and y limits, moves both together),
    y, the value axis, wins within its caps (``_cap``, against the data's ``spans`` before the key), else the one that needs
    the least panel growth (``_widening``), so x widens only where y is shared, capped or cannot clear. An axes that may widen
    neither needs an infinite growth.
    """
    box = legend.get_window_extent().padded(_KEY_CLEARANCE_PT * figure.dpi / _POINTS_PER_INCH)
    frame, sides = axis.get_window_extent(), ("right" in corner, "upper" in corner)
    needs: dict[Axes, _Widening] = {}
    for owner in dict.fromkeys(item[0] for item in drawn):
        own = [item for item in drawn if item[0] is owner]
        if not _covers(box, own):  # an axes whose data the key clears needs nothing, whatever it shares
            continue
        options: list[_Widening] = []
        for along in (1, 0):
            moved = (along, 1 - along) if owner.get_xlim() == owner.get_ylim() else (along,)
            if not any(_shared(owner, other) for other in moved):
                cap = min(_cap(owner, other, sides[along], spans[owner, other]) for other in moved)
                options.append((*_widening(box, frame, _heights(box, own, owner, along), along, sides[along], cap), moved))
        needs[owner] = min(options, key=lambda option: option[0], default=(np.inf, np.inf, (1,)))  # the first, y, on a tie
    return needs


def _growth(needed: Mapping[Axes, _Widening]) -> tuple[float, float]:
    """Return the factors (x, y) by which the panel must grow for every axes in ``needed`` to widen within its caps; (1, 1) for none."""
    wider, taller = (max((growth for growth, _factor, moved in needed.values() if moved[0] == along), default=1.0) for along in (0, 1))
    return wider, taller


def _place_key(
    figure: Figure, axis: Axes, legend: Legend, drawn: Sequence[_Drawn], spans: Mapping[tuple[Axes, int], _Span]
) -> tuple[_KeyCorner, dict[Axes, _Widening], tuple[float, float]]:
    """
    Put ``legend``, built at Matplotlib's best place, in a corner; return the corner, its widening and the panel growth it needs.

    The key takes its first free corner, Matplotlib's best one first, else the one that needs the least widening within the caps
    (``_key_needs``), one that widens y alone before one that moves x, and there needs no growth, (1, 1). Where no corner fits
    inside the panel and clears the data so, it takes the one that needs the least growth (x, y) of the panel for both.
    """
    best, frame = legend.get_window_extent().bounds, axis.get_window_extent()
    legend.set_loc(_KEY_CORNERS[0])
    box = legend.get_window_extent()  # every corner leaves the same border, so one shows whether the key fits inside the panel
    fit = (max(1.0, (frame.x1 - box.x0) / frame.width), max(1.0, (frame.y1 - box.y0) / frame.height))
    options: dict[_KeyCorner, tuple[dict[Axes, _Widening], tuple[float, float]]] = {}
    best_corner: _KeyCorner | None = None
    for corner in _KEY_CORNERS:
        legend.set_loc(corner)
        if legend.get_window_extent().bounds == best:
            best_corner = corner
        needed = _key_needs(figure, axis, legend, corner, drawn, spans)
        growth = _growth(needed)
        options[corner] = (needed, (max(fit[0], growth[0]), max(fit[1], growth[1])))

    def cost(corner: _KeyCorner) -> tuple[float, bool, float, bool]:
        needed, (wider, taller) = options[corner]
        moves_x = any(moved[0] == 0 for _growth, _factor, moved in needed.values())  # a stretched x, where y alone clears another corner
        # Rounded, so a mathematical tie (symmetric data, full-height bars) goes to Matplotlib's corner order, not to float noise.
        return round(wider * taller, 6), moves_x, round(max((option[1] for option in needed.values()), default=0.0), 6), corner != best_corner

    corner = min(options, key=cost)
    legend.set_loc(corner)
    needed, growth = options[corner]
    return corner, needed, growth


def _restore(axes: Sequence[Axes]) -> bool:
    """
    Set every axis of ``axes`` a key widened back to its limits before and forget that state; return whether any limit moved.

    An axis keeps its limits where the caller has set others since, or where it shares them now with other axes, which took them.
    """
    restored = False
    for owner in axes:
        for along, (before, after) in (getattr(owner, _HEADROOM, None) or {}).items():
            if _limits(owner, along) == after and not _shared(owner, along):
                (owner.set_xlim, owner.set_ylim)[along](before, auto=None)
                restored = True
        setattr(owner, _HEADROOM, None)
    return restored


def _widen(
    figure: Figure,
    axis: Axes,
    legend: Legend,
    corner: _KeyCorner,
    axes: Sequence[Axes],
    needed: Mapping[Axes, _Widening],
    spans: Mapping[tuple[Axes, int], _Span],
) -> tuple[float, float]:
    """
    Move the limit on the key's side of every axes in ``needed`` out by its factor, keeping its autoscaling, then again as the drawn layout asks.

    Every move is checked again on the drawn layout. Return (1, 1) where the key clears the data after at most two moves within
    the caps, else the panel growth (``_growth``) the pass that would break one needs, which then moves nothing, or an infinite
    one where data, such as those a move brought into view, still lie under the key after the second. The second pass is
    capped against the data's ``spans`` before the first, so two passes never widen more than one may. Every move is runtime
    state of its axes (``_HEADROOM``).
    """
    sides = ("right" in corner, "upper" in corner)
    for _move in range(2):
        growth = _growth(needed)
        if not needed or growth != (1.0, 1.0):
            return growth
        for owner, (_growth_needed, factor, moved) in needed.items():
            if factor > 1.0:
                along, state = moved[0], getattr(owner, _HEADROOM, None) or {}
                # The value at the axes fraction ``factor`` from the opposite side extends the scale linearly in its units; the
                # axes of a panel map x and y apart, so the other coordinate of the point is irrelevant.
                fraction = factor if sides[along] else 1.0 - factor
                limit = (owner.transAxes + owner.transData.inverted()).transform((fraction, fraction))[along]
                for other in moved:
                    current = _limits(owner, other)
                    (owner.set_xlim, owner.set_ylim)[other]((current[0], limit) if sides[along] else (limit, current[1]), auto=None)
                    state[other] = (state[other][0] if other in state else current, _limits(owner, other))
                setattr(owner, _HEADROOM, state)
        figure.draw_without_rendering()
        needed = _key_needs(figure, axis, legend, corner, _drawn_data(axes), spans)
    return (np.inf, np.inf) if needed else (1.0, 1.0)


def _settle(
    figure: Figure, axis: Axes, legend: Legend, axes: Sequence[Axes], drawn: Sequence[_Drawn], spans: Mapping[tuple[Axes, int], _Span]
) -> tuple[float, float]:
    """Place and clear the key ``legend`` as built (``_place_key``, ``_widen``); return (1, 1), else the panel growth it needs, limits unmoved."""
    corner, needed, growth = _place_key(figure, axis, legend, drawn, spans)
    if growth == (1.0, 1.0):
        growth = _widen(figure, axis, legend, corner, axes, needed, spans)
        if growth != (1.0, 1.0) and _restore(axes):
            figure.draw_without_rendering()
    return growth


def _panel_growth_in(figure: Figure, axis: Axes, growth: tuple[float, float]) -> tuple[float, float] | None:
    """
    Return the inches (width, height) the panels' rectangle must gain for the panel ``axis`` to grow by ``growth`` (x, y) and ``_GROWTH_SLACK``.

    Only a gridspec panel of a constrained layout grows with the rectangle (``_grow``), by its gridspec share of what it gains,
    and at most by ``_MAX_GROWTH`` each way, slack included; None for any other panel and for a growth beyond that.
    """
    spec = axis.get_subplotspec()
    if spec is None or not isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine) or max(growth) * _GROWTH_SLACK > _MAX_GROWTH:
        return None
    grid, frame = spec.get_gridspec(), axis.get_window_extent()
    size = (frame.width / figure.dpi, frame.height / figure.dpi)
    shares = [
        np.asarray(ratios, dtype=float)[span].sum() / np.sum(ratios)
        for ratios, span in ((grid.get_width_ratios(), spec.colspan), (grid.get_height_ratios(), spec.rowspan))
    ]
    wider, taller = (
        (factor * _GROWTH_SLACK - 1.0) * length / share if factor > 1.0 else 0.0 for factor, length, share in zip(growth, size, shares, strict=True)
    )
    return float(wider), float(taller)


def _clear_key(figure: Figure, axis: Axes, key: Callable[[int], Legend], columns: int) -> bool:
    """
    Put the key of a single panel, built by ``key`` in a given number of columns, in a corner clear of the data of the panel and its twin axes.

    Matplotlib's best place stays where it is a corner that covers no data (``_drawn_data``). Otherwise the key takes the first
    free corner in Matplotlib's order, else the corner that needs the least widening, and every axes with data under the key
    moves one limit on that side out until they clear it, keeping its autoscaling (``_place_key``, ``_widen``). Where that would
    break a cap (``_cap``, against data that leave out reference lines), the key is built again with one more column, up to
    ``columns`` while it fits the panel's width; where no count fits and clears, the panel grows once by the least any count
    needs (``_panel_growth_in``). Return whether the key stands clear; where nothing clears the data, it is removed for the
    caller to set outside, with the limits as the caller left them and a tried growth kept. While it is placed, the key takes no
    part in the layout, so a candidate larger than the panel never squeezes it. A key added again starts from the limits before
    its predecessor's widening, unless the caller has set them or shared their axis since.
    """

    def build(count: int) -> Legend:
        legend = key(count)
        legend.set_in_layout(False)
        return legend

    legend = build(1)
    figure.draw_without_rendering()  # the final layout, in which twin axes share the panel's place, and Matplotlib's best place
    place = axis.get_position().bounds
    axes = [other for other in figure.axes if other is axis or (other.get_visible() and other.get_position().bounds == place)]
    if _restore(axes):
        figure.draw_without_rendering()
    frame, drawn, data = axis.get_window_extent(), _drawn_data(axes), _drawn_data(axes, references=False)
    spans = {(owner, along): _span(frame, data, owner, along) for owner in axes for along in (0, 1)}
    needs: list[tuple[float, int, tuple[float, float]]] = []  # per column count that failed: the panel growth it needs
    growth = (np.inf, np.inf)
    for count in range(1, columns + 1):
        if count > 1:
            legend = build(count)
            if legend.get_window_extent().width > frame.width:
                break  # every further column widens it
        growth = _settle(figure, axis, legend, axes, drawn, spans)
        if growth == (1.0, 1.0):
            break
        needs.append((round(growth[0] * growth[1], 6), count, growth))
    if growth != (1.0, 1.0) and (added := _panel_growth_in(figure, axis, min(needs)[2])) is not None:  # the least growth, once
        legend = build(min(needs)[1])
        _grow(figure, panels_in=added)
        _fit_note(figure)
        figure.draw_without_rendering()
        growth = _settle(figure, axis, legend, axes, _drawn_data(axes), spans)
    legend.set_in_layout(True)
    if growth != (1.0, 1.0):
        legend.remove()  # nothing clears the data inside the panel: the caller sets the key outside
    return growth == (1.0, 1.0)


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
    groups: Mapping[str, Sequence[Artist]] | None = None,
    outside: bool = False,
) -> None:
    """
    Write the legend of ``axes``: inside the only visible panel, else outside right of the rightmost panel, top at the top frame.

    Inside the panel the key stands in a corner: its best place where that is a corner clear of the data of the panel and
    its twin axes, else a free corner, else the corner that needs the least headroom, with a limit on that side moved
    out until the key clears the data; where that would leave the data under 60 % of the axis, the key takes more columns,
    and where none clears it, the panel grows, at most to twice its size (``_clear_key``). Where even that fails, as for a
    hand-placed panel, the key stands outside right as ``outside`` places the
    legend of a single panel, for a figure the owner's spec names so. Of entries
    with a repeated label the first is kept. ``groups`` maps a group title to the entries it owns, each by its handle, and
    stacks one titled legend per group below each other in the column (inside the
    single panel: one legend with the entries in group order under bold heading rows, whole groups per key column, ``_key_rows``);
    entries no group owns form a final untitled group. Membership is
    identity, never text or position (``_blocks``), so a model named like a material keeps its own marker beside the material's
    entry and a repeated label merges within its group only. ``title`` titles an ungrouped legend. The column is as wide as its
    longest entry: the figure grows by that width, less a strip right of the panels a wide title or note added already, and by the
    missing height when the column is taller than the panels, while the panels keep their size; a figure title's note wraps again
    to the wider figure. A figure without a constrained layout keeps its size, and a tight export holds the column.
    """
    pairs = list(zip(labels, handles, strict=True))
    if not pairs:
        message = "A legend needs at least one entry; a view without series draws none."
        raise ValueError(message)
    visible = [axis for axis in _grid(axes).flat if axis.get_visible()]
    if not visible:
        message = "A legend needs at least one visible panel."
        raise ValueError(message)
    blocks = _blocks(pairs, title, groups)
    if len(visible) == 1 and not outside:
        # One block is a plain list (its name titles the legend); several keep their titles as bold heading rows (``_key_rows``).
        legend_title = blocks[0][0] if len(blocks) == 1 and title is None else title

        def key(columns: int) -> Legend:
            entry_handles, entry_labels, headings = _key_rows(blocks, columns)
            legend = visible[0].legend(
                entry_handles, entry_labels, loc="best", ncols=columns, title=legend_title, fontsize=TICK_PT, title_fontsize=TICK_PT
            )
            for row in headings:
                legend.get_texts()[row].set_fontweight("bold")
            return legend

        if _clear_key(figure, visible[0], key, len(blocks) if len(blocks) > 1 else len(blocks[0][1])):  # whole groups per column
            return
    figure.draw_without_rendering()  # the layout runs now, so the positions below are final
    top = max(visible, key=lambda axis: axis.get_position().y1)
    right = max(visible, key=lambda axis: axis.get_position().x1)
    anchor = blended_transform_factory(right.transAxes, top.transAxes)
    tight_right = max(box.x1 for axis in figure.axes if axis.get_visible() and (box := axis.get_tightbbox()) is not None)  # twin axes included
    decorations_in = max(0.0, tight_right - right.get_window_extent().x1) / figure.dpi
    legends = [
        figure.legend(
            list(entries.values()),
            list(entries),
            title=name,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=TICK_PT,
            title_fontsize=TICK_PT,
        )
        for name, entries in blocks
    ]
    extents = [legend.get_window_extent() for legend in legends]
    offset_in = 0.0
    for legend, extent in zip(legends, extents, strict=True):
        legend.set_bbox_to_anchor(
            (1.0, 1.0),
            transform=anchor + ScaledTranslation(decorations_in + LEGEND_GAP_IN, -offset_in, figure.dpi_scale_trans),
        )
        offset_in += extent.height / figure.dpi + LEGEND_STACK_GAP_IN
    if not isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine):
        return  # hand-placed panels keep their place in a figure of fixed size
    column_in = offset_in - LEGEND_STACK_GAP_IN
    panels_in = (top.get_window_extent().y1 - min(axis.get_window_extent().y0 for axis in visible)) / figure.dpi
    left, _bottom, width, _height = _engine(figure).get()["rect"]
    reserved_in = (1.0 - left - width) * figure.get_figwidth()  # the strip right of the panels a wide title or note added already
    _grow(
        figure,
        right_in=max(0.0, LEGEND_GAP_IN + max(extent.width for extent in extents) / figure.dpi - reserved_in),
        bottom_in=max(0.0, column_in - panels_in),
    )
    _fit_note(figure)  # a note wrapped before the column existed takes the width it adds


def label_rows_once(axes: np.ndarray, ylabel: str) -> None:
    """Write ``ylabel`` on the first visible panel of every row and clear it on the others; tick labels stay where they are."""
    for row in _grid(axes):
        first = next((column for column, axis in enumerate(row) if axis.get_visible()), None)
        for column, axis in enumerate(row):
            axis.set_ylabel(ylabel if column == first else "", fontsize=LABEL_PT)


def bold_label(text: str) -> str:
    """Return ``text`` with its mathtext spans set bold and upright, symbols included; ``fontweight`` alone never bolds mathtext."""
    return _MATH_SPAN.sub(lambda match: "$\\mathbf{" + match.group(1).replace("\\mathrm{", "\\mathbf{") + "}$", text)


def set_row_headers(figure: Figure, axes: np.ndarray, headers: Sequence[str]) -> None:
    """
    Write one bold header per row (``bold_label``), left aligned just above the row's first visible panel and the row's upper decorations.

    The header stands ``ROW_HEADER_GAP_PT`` above the frames, panel titles and y-axis offset texts (a scientific factor) of
    its row's visible panels, measured whenever the figure is laid out or drawn; the layout keeps ``ROW_HEADER_CLEARANCE_IN``
    free above it, so the header reads with its own row rather than with the x labels of the row above.
    """
    for row, header in zip(_grid(axes), headers, strict=True):
        shown = [axis for axis in row if axis.get_visible()]
        if shown:
            transform = shown[0].transAxes + ScaledTranslation(0.0, ROW_HEADER_GAP_PT / _POINTS_PER_INCH, figure.dpi_scale_trans)
            shown[0].add_artist(
                _RowHeader(
                    shown,
                    0.0,
                    1.0,
                    bold_label(header),
                    transform=transform,
                    fontsize=PANEL_TITLE_PT,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
            )


def set_figure_title(figure: Figure, axes: np.ndarray, title: str, *, note: str | None = None) -> None:
    """
    Write the bold figure title and, a small gap below it, its smaller note, both centred over the visible panels ``axes``.

    The figure grows by the strip of both texts, which stay anchored a fixed inset below its top edge in inches; the panels keep
    their size. A text wider than the panels shifts inward, never closer than that inset to a figure edge, also in a tight
    export. A note line wider than the figure wraps at word boundaries, again when a later legend widens the figure
    (``add_legend``); a title, or a note line that cannot wrap, wider than the figure widens it. A second call raises ``ValueError``.
    """
    if any(isinstance(artist, _CenteredText) for artist in figure.artists):
        message = "The figure title is already set."
        raise ValueError(message)
    panels = tuple(_grid(axes).flat)
    title_in = TITLE_PT * _LINE_HEIGHT * (title.count("\n") + 1) / _POINTS_PER_INCH
    texts = [(_CenteredText(panels, 0.5, 1.0, title, fontsize=TITLE_PT, fontweight="bold", ha="center", va="top"), _TITLE_INSET_IN)]
    note_in = 0.0
    if note is not None:  # the strip holds the unwrapped note as rendered, mathtext lines taller, until ``_fit_note`` wraps it
        note_in = (NOTE_GAP_PT + _NOTE_CLEARANCE_PT) / _POINTS_PER_INCH + _note_size_in(figure, note)[1]
        note_drop_in = _TITLE_INSET_IN + title_in + NOTE_GAP_PT / _POINTS_PER_INCH
        texts.append((_CenteredText(panels, 0.5, 1.0, note, fontsize=NOTE_PT, ha="center", va="top", unwrapped=note), note_drop_in))
    for artist, drop_in in texts:
        figure.add_artist(artist)
        artist.set_transform(figure.transSubfigure + ScaledTranslation(0.0, -drop_in, figure.dpi_scale_trans))
    _grow(
        figure,
        right_in=max(0.0, texts[0][0].get_window_extent().width / figure.dpi + 2.0 * _TITLE_INSET_IN - figure.get_figwidth()),
        top_in=_TITLE_INSET_IN + title_in + note_in,
    )
    _fit_note(figure)
