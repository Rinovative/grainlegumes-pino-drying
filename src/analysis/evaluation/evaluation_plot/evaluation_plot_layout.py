"""Define the title, case-count and text helpers shared by evaluation figures.

Responsibilities:
  - Draw the step ECDF that every distribution figure shares and give every display channel its fixed palette colour
  - Title figures with exact aggregate case counts or visible case numbers, also per dataset panel, a bounded prefix as
    «first 50 of 128 cases», with the short label of one compared run or «A vs B» of two, and with the one dataset whose
    models a comparison's columns show
  - Name the models and the roles of dataset-panel figures in one legend, in their fixed styles (lines, bars or points), the
    materials a pooled series shows as markers, and a model whose points carry several markers as a neutral circle beside a key
    of their shapes
  - Mark panels whose bounded prefix has no finite case as unavailable, the lines of one panel fitted inside it at every draw
  - Write a panel's support or fit lines as a caption under its x label, never over its data
  - Write the one colour-scale note under the title and the mean-error line above each error map of a fixed-inch
    map grid, the mean with three significant digits in the power of ten of its bar

Design principles:
  - Layout is presentation-only and never changes scientific values
  - The title, case-count and unavailable-panel helpers serve every evaluation figure, not only field maps
  - Map geometry, headers and colour bars belong to the shared fixed-inch map grid; this module only adds text to it

This module does NOT:
  - Load artifacts, select cases, calculate statistics, or clip scientific data
  - Choose map colormaps, labels, coordinates, or plot-specific panel composition
  - Plan map grids, choose colour-scale bands (``analysis_field_maps``) or label colour bars (``analysis_ui_plot_layout``)
  - Own notebook controls, display calls, exports, or lazy rendering
"""

from __future__ import annotations

import math
from decimal import Decimal
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Sized

    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_presentation as presentation
    from src.analysis.presentation import analysis_visual_semantics as visual_semantics


_NOTE_FROM_TOP_IN = 0.36  # the scale note's top below the figure top: under the 12 pt title of a fixed-inch map grid
_NOTE_SIZE_PT = 9.0
_NOTE_COLOUR = "0.20"
_EMPTY_PREFIX_SPAN = 0.9  # the unavailable lines never take more of the panel's width or height
_EMPTY_PREFIX_SHRINK = 0.9  # font scale per retry while the unavailable lines pass the panel
_EMPTY_PREFIX_MIN_PT = 5.0  # the smallest font a retry reaches; rows that pass the panel even at it are cut and counted
_CAPTION_GAP_PT = 2.0  # a caption line below the x label or the line before
_SIGNIFICANT_DIGITS = 3  # of a mean-error line and the other statistics written into a figure
_POINT_SIZE = 7.0  # the marker of a point entry in a legend
_TITLED_MODELS = 2  # a title names one compared run or «A vs B»; the legend names more
MARK_EVERY = 0.1  # a marker every tenth of the axes diagonal, so dense lines stay readable
POOLED_NOTE = "Pooled series: the mean of its materials, each material counts equally; the small markers show each material"
# The ends of a state or error scale: round bands over its robust range (field_maps.percentile_scale), widened for a degenerate or
# near-uniform range (field_maps.state_boundaries, error_boundaries), so the ends cover that range but are not its percentiles.
PERCENTILE_SCALE = "5th\N{EN DASH}95th percentile, covered by round bands"


class _EmptyPrefixText(Text):
    """
    The unavailable lines of one panel, fitted to the panel's final size at every draw.

    Each line wraps to the panel width at word boundaries, a wider word at character boundaries; while the rows still pass the
    panel (many models) the font shrinks, and the rows that even the smallest font cannot fit end in one row «… k more», so the
    text stays inside the panel whatever layout ran before the draw.
    """

    def __init__(self, axis: Axes) -> None:
        """Start without lines, centred on ``axis`` and outside the layout, so the panel never grows for its text."""
        super().__init__(0.5, 0.5, "", horizontalalignment="center", verticalalignment="center", transform=axis.transAxes, clip_on=False)
        self.set_in_layout(False)
        self.notes: list[str] = []
        self._panel = axis
        self._size = self.get_fontproperties().get_size_in_points()

    def _wrapped(self, note: str, width: float, renderer: RendererBase) -> list[str]:
        """Return the rows of ``note`` at most ``width`` pixels wide, broken at word boundaries and a wider word at character boundaries."""

        def fits(row: str) -> bool:
            return renderer.get_text_width_height_descent(row, self.get_fontproperties(), ismath=False)[0] <= width

        rows: list[str] = []
        for word in note.split():
            if rows and fits(joined := f"{rows[-1]} {word}"):
                rows[-1] = joined
            elif fits(word):
                rows.append(word)
            else:
                rows.append("")
                for character in word:
                    if rows[-1] and not fits(rows[-1] + character):
                        rows.append("")
                    rows[-1] += character
        return rows

    def draw(self, renderer: RendererBase) -> None:
        """Wrap every line to the panel width, shrink the font until the rows fit the panel, cut those the smallest font cannot fit, then draw."""
        panel = self._panel.get_window_extent(renderer)
        width, height = _EMPTY_PREFIX_SPAN * panel.width, _EMPTY_PREFIX_SPAN * panel.height
        size = self._size
        while True:
            self.set_fontsize(size)
            rows = [row for note in self.notes for row in self._wrapped(note, width, renderer)]
            self.set_text("\n".join(rows))
            extent = self.get_window_extent(renderer)
            if (extent.width <= width and extent.height <= height) or size <= _EMPTY_PREFIX_MIN_PT:
                break
            size = max(_EMPTY_PREFIX_MIN_PT, size * _EMPTY_PREFIX_SHRINK)
        kept = len(rows)
        while kept and self.get_window_extent(renderer).height > height:
            kept -= 1
            self.set_text("\n".join([*rows[:kept], f"\N{HORIZONTAL ELLIPSIS} {len(rows) - kept} more"]))
        super().draw(renderer)


def add_scale_note(figure: Figure, *, shared: bool = True, scale: str = PERCENTILE_SCALE, extra: str | None = None) -> None:
    """
    Write the one colour-scale note under the title of a fixed-inch map grid: what sets the bars' ends and that arrows mark values beyond.

    ``scale`` names the rule of the ends, the robust percentile range by default; ``shared`` adds that the prediction uses the reference
    scale; ``extra`` ends the line, such as the meaning of lines drawn over the maps. Every map figure whose bars can show an arrow
    writes it here.
    """
    reference = "prediction uses the reference scale; " if shared else ""
    figure.text(
        0.5,
        1.0 - _NOTE_FROM_TOP_IN / figure.get_figheight(),
        f"Colour scales: {scale}; {reference}arrows mark values outside" + (f"; {extra}" if extra else ""),
        ha="center",
        va="top",
        fontsize=_NOTE_SIZE_PT,
        color=_NOTE_COLOUR,
    )


def significant(value: float) -> str:
    """
    Return ``value`` with three significant digits, written positionally with its trailing zeros (0.0100, 6.70, 150, 1230), never in e-notation.

    The value is rounded once, to a three-digit scientific string whose digits are then written out without another float conversion:
    a carry such as 9.996 to 10.0 keeps three digits, and a large value such as 1.234e24 gets exact zeros, never a binary tail.
    """
    if not math.isfinite(value):
        return f"{value}"
    return format(Decimal(f"{value:.{_SIGNIFICANT_DIGITS - 1}e}"), "f")


def mean_error_text(values: np.ndarray | None, *, unit: str | None, exponent: int = 0) -> str:
    """
    Return the mean-error line of one error map: mean |e| of its finite ``values`` in ``unit`` (``None``: normalized), else "n/a".

    The mean runs over every finite cell of the field, independent of the display sampling. A bar that writes its labels as
    multiples of a power of ten (``exponent``, from ``shared_layout.error_format``) gets the mean as a multiple of that same power. The
    mean keeps three significant digits, trailing zeros included (``significant``).
    """
    finite = None if values is None else np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite is None or finite.size == 0:
        return "mean |e| n/a"
    power = f" \N{MULTIPLICATION SIGN} {shared_layout.power_of_ten(exponent)}" if exponent else ""
    mean = f"{significant(float(np.mean(np.abs(finite))) / 10.0**exponent)}{power}"
    return f"mean |e| = {mean} {unit}" if unit is not None else f"mean {error_diagnostics.NORMALIZED_ERROR_SYMBOL} = {mean}"


def add_caption(axis: Axes, lines: Sequence[str], colors: Sequence[object] | None = None) -> None:
    """
    Write ``lines`` under the x label of ``axis``, each below the one before and in its colour of ``colors``, never over the data.

    The lines are annotations anchored to the label, so a constrained layout reserves their strip under the panel. They name what a
    panel shows, such as its case support or fit statistics.
    """
    anchor: Artist = axis.xaxis.label
    for index, line in enumerate(lines):
        anchor = axis.annotate(
            line,
            xy=(0.5, 0.0),
            xycoords=anchor,
            xytext=(0.0, -_CAPTION_GAP_PT),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=figure_layout.TICK_PT,
            color="black" if colors is None else colors[index],
        )


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
    eligible: int | None = None,
) -> str:
    """Append exactly one aggregate-count or visible-case suffix; a count below its ``eligible`` cases names the prefix, «first 50 of 128 cases»."""
    if (case_count is None) == (case_number is None):
        msg = "Supply exactly one of case_count or case_number."
        raise ValueError(msg)
    value = case_count if case_count is not None else case_number
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        msg = "Case counts and visible case numbers must be positive integers."
        raise ValueError(msg)
    numeric = int(value)
    suffix = f"{numeric} {'case' if numeric == 1 else 'cases'}" if case_count is not None else f"Case {numeric}"
    if case_count is not None and eligible is not None and eligible > numeric:  # a case-count viewer draws the first cases only
        suffix = f"first {numeric} of {eligible} cases"
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


def models_suffix(models: Iterable[str]) -> str:
    """
    Return the title suffix naming the compared runs ``models``, each once, by their short run labels, the labels the legend uses.

    The compared runs are those whose frames the figure receives, also one it cannot draw (its note names the gap). One run gives
    `` (FNO · B · seed 9)``, exactly two `` (A vs B)``. More runs, which the legend names, or a run without an explicit label (``""``)
    give none, so a figure never names a run it cannot tell from the others.
    """
    runs = tuple(dict.fromkeys(models))
    return f" ({' vs '.join(runs)})" if 0 < len(runs) <= _TITLED_MODELS and all(runs) else ""


def dataset_suffix(datasets: Mapping[str, pd.DataFrame]) -> str:
    """Return `` · <dataset>`` naming the one dataset whose models are the columns of a comparison figure, which no column names; else ""."""
    labels = {datasets.labels[key][0] for key in datasets} if isinstance(datasets, dataset_cuts.DisplayFrames) else set()
    if len(labels) == 1 and (label := next(iter(labels))) not in datasets:
        return f" · {label}"
    return ""


def aggregate_title_context(
    title: str, case_counts: Mapping[str, int], *, models: Iterable[str] = (), eligible: Mapping[str, int] | None = None
) -> tuple[str, dict[str, str]]:
    """
    Return an honest figure title and dataset headings for aggregate evidence; the title ends with ``models_suffix(models)``.

    ``eligible`` gives a dataset's cases before a bounded prefix: a drawn count below it reads «first n of N cases» (``case_title``).
    Datasets alike in drawn and eligible count share the title's count, else each heading carries its own.
    """
    if not case_counts:
        msg = "Aggregate title context requires at least one dataset count."
        raise ValueError(msg)
    pairs = {label: (count, (eligible or {}).get(label)) for label, count in case_counts.items()}
    validated = {label: case_title(label, case_count=count, eligible=total) for label, (count, total) in pairs.items()}
    named = models_suffix(models)
    if len(set(pairs.values())) == 1:
        count, total = next(iter(pairs.values()))
        return case_title(title, case_count=count, eligible=total) + named, dict.fromkeys(case_counts, "")
    return title + named, validated


def empty_prefix_text(case_count: int) -> str:
    """Return how a bounded-prefix view states that no finite case remains, never a number and never zero."""
    return f"unavailable (0 finite of {case_count} prefix cases)"


def mark_empty_prefix(*axes: Axes, case_count: int, model: str | None = None) -> None:
    """
    Show on every given panel that its bounded prefix has no finite case, instead of drawing or failing.

    With ``model`` the line starts with the model's name. The lines of one panel stand together about its centre in one text that
    every draw wraps to the panel width and shrinks while it would pass the panel's edges, its rows beyond the smallest font cut
    to one row «… k more», so it stays inside the panel.
    """
    text = empty_prefix_text(case_count) if model is None else f"{model}: {empty_prefix_text(case_count)}"
    for axis in axes:
        note = next((item for item in axis.texts if isinstance(item, _EmptyPrefixText)), None)
        if note is None:
            note = _EmptyPrefixText(axis)
            axis.add_artist(note)
        note.notes.append(text)
        note.set_text("\n".join(note.notes))  # unwrapped until the first draw fits it to the panel


def plot_ecdf(
    axis: Axes,
    values: np.ndarray,
    *,
    color: object,
    linestyle: str = "-",
    marker: str | None = None,
    label: str | None = None,
    positive_only: bool = False,
) -> None:
    """
    Draw the empirical CDF i/n of ``values`` (finite, non-empty) as a step line, without smoothing.

    ``positive_only`` leaves the exact zeros off, which a log axis cannot place; they keep their share of the CDF, and at
    least one value must be positive. The line
    starts with the rise at the smallest drawn value from the CDF level below it, so a single value is a visible step.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    step = 1.0 / ordered.size
    cumulative = np.arange(1, ordered.size + 1, dtype=float) * step
    if positive_only:
        ordered, cumulative = ordered[ordered > 0.0], cumulative[ordered > 0.0]
    start = cumulative[0] - step  # 0, or the share of the exact zeros left off
    axis.step(
        np.r_[ordered[0], ordered],
        np.r_[start, cumulative],
        where="post",
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker=marker,
        markevery=MARK_EVERY,
        label=label,
    )


def channel_colors(fields: Iterable[presentation.DisplayField]) -> dict[str, tuple[float, float, float, float]]:
    """Return the fixed colour of every display channel by field key, in display order: a channel keeps it whichever channels are drawn."""
    palette = colormaps["tab10"]
    return {field.key: palette(index % palette.N) for index, field in enumerate(fields)}


def dataset_title_context(title: str, datasets: dataset_cuts.DisplayFrames, counts: Mapping[str, int]) -> tuple[str, dict[str, str]]:
    """
    Return ``aggregate_title_context`` for the dataset panels of ``datasets`` (``series.panel_groups`` of its labels).

    ``counts`` is by frame key; a dataset panel counts the cases of its first model's frames (all materials of a pooled series), as the
    models of one dataset are case-matched, and names a bounded prefix by those frames' own cases («first 50 of 128 cases»). The
    title names the models of every frame by their title labels (``DisplayFrames.title_labels``), so a model without an explicit
    label names none.
    """
    first = {dataset: next(iter(models.values())) for dataset, models in series.panel_series(datasets.labels).items()}
    panel_counts = {dataset: sum(counts[key] for key in keys) for dataset, keys in first.items()}
    eligible = {dataset: sum(len(datasets[key]) for key in keys) for dataset, keys in first.items()}
    return aggregate_title_context(title, panel_counts, models=datasets.title_labels.values(), eligible=eligible)


def material_legend(
    datasets: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Mapping[str, tuple[str, ...]]],
    styles: Mapping[str, visual_semantics.SeriesStyle],
) -> dict[str, dict[str, Artist]]:
    """
    Return the ``extra`` entry group of ``add_series_legend`` that names the materials a pooled series shows as markers.

    ``groups`` is ``series.panel_series`` of the figure and ``styles`` the per-material styles of its frames: a material is a colour
    when the figure colours by material, else a marker shape in grey (the colour then names the model). Without a pooled series the
    result is empty.
    """
    entries: dict[str, Artist] = {}
    pooled = [key for models in groups.values() for keys in models.values() if len(keys) > 1 for key in keys]
    for key in pooled:
        style = styles[key]
        entries.setdefault(
            series.material_name(datasets[key]),
            Line2D([0], [0], color="0.35" if style.marker else style.color, marker=style.marker or "o", linestyle="", markersize=6),
        )
    return {"Material": entries} if entries else {}


def series_note(styles: series.SeriesStyles, *notes: str | None) -> str | None:
    """Join the given figure notes, adding «colour = run» when the figure's styles colour by run, not by architecture; ``None`` without any."""
    lines = [note for note in notes if note]
    if styles.run_colours:
        lines.append("colour = run")
    return "\n".join(lines) or None


def add_series_legend(
    figure: Figure,
    axes: np.ndarray,
    frames: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Mapping[str, str]],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    *,
    bars: bool = False,
    points: bool = False,
    extra: Mapping[str, Mapping[str, Artist]] | None = None,
    name_models: bool = True,
    name_roles: bool = True,
) -> None:
    """
    Write the legend of ``axes`` naming the models and, where panels differ in role, the roles; ``extra`` adds titled entry groups first.

    A model gets one entry in the colour of its run (a solid line, a patch for ``bars``, or for ``points`` the marker its points are
    drawn with: the one its series share, else a neutral circle beside a grey key of their material markers, added to a «Material»
    group of ``extra``) only when several models are drawn (one model is told by the panel titles) and ``name_models`` holds; a
    figure that colours something else drops it.
    Roles get a second group with one grey entry per role, in its line style or hatch, only when the panels differ in role, so no
    panel's role stands for the others; ``name_roles`` is off for scatter and point figures, whose points show no line style or hatch.
    Entries of different groups keep their own handles also under equal text, so a model named like a material keeps its own
    marker. Without any entry (a single model of a single role) no legend is written. A single titled group keeps its title also
    inside a single panel.
    """
    first: dict[str, str] = {}
    markers: dict[str, dict[str | None, str]] = {}  # each model's markers, each with the first of its series drawn so
    roles: dict[str, visual_semantics.SeriesStyle] = {}
    for by_model in groups.values():
        for model, key in by_model.items():
            first.setdefault(model, key)
            markers.setdefault(model, {}).setdefault(styles[key].marker, key)
            roles.setdefault(dataframe.dataset_role(frames[key]), styles[key])
    blocks: dict[str, dict[str, Artist]] = {title: dict(group) for title, group in (extra or {}).items()}

    def artist(color: str, linestyle: visual_semantics.LineStyle, hatch: str, *, role: bool, marker: str | None = None) -> Artist:
        if bars:
            return Patch(facecolor="white" if role else color, edgecolor="0.35" if role else "white", hatch=hatch)
        if points:
            return Line2D([0], [0], color=color, marker=marker or "o", linestyle="", markersize=_POINT_SIZE)
        return Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.0)

    if name_models and len(first) > 1:
        mixed = {model: shown for model, shown in markers.items() if points and len(shown) > 1}
        blocks["Model"] = {
            model: artist(styles[key].color, "-", "", role=False, marker=None if model in mixed else styles[key].marker)
            for model, key in first.items()
        }
        shown_markers = [(marker, key) for shown in mixed.values() for marker, key in shown.items() if marker is not None]
        for marker, key in shown_markers:  # the grey key that tells those models' shapes apart
            shapes = blocks.setdefault("Material", {})
            shapes.setdefault(series.material_name(frames[key]), Line2D([0], [0], color="0.35", marker=marker, linestyle="", markersize=_POINT_SIZE))
    if name_roles and len(roles) > 1:
        blocks["Role"] = {role: artist("0.35", style.linestyle, style.hatch, role=True) for role, style in roles.items()}
    if any(blocks.values()):
        labels = [label for group in blocks.values() for label in group]  # a label of two groups once per group, own handle each
        handles = [handle for group in blocks.values() for handle in group.values()]
        named = {title: list(group.values()) for title, group in blocks.items()}  # each group owns its entries by their handles
        figure_layout.add_legend(figure, axes, handles, labels, title=next(iter(named)) if len(named) == 1 else None, groups=named)
