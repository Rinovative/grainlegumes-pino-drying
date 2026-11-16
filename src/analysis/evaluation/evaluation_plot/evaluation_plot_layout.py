"""Define the title, case-count and text helpers shared by evaluation figures.

Responsibilities:
  - Draw the step ECDF that every distribution figure shares and give every display channel its fixed palette colour
  - Title figures with exact aggregate case counts or visible case numbers, also per dataset panel
  - Name the models and the roles of dataset-panel figures in one legend, in their fixed styles
  - Mark panels whose bounded prefix has no finite case as unavailable
  - Write the one colour-scale note under the title and the mean-error line above each error map of a fixed-inch
    map grid, the mean in the power of ten of its bar

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

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Sized

    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_presentation as presentation
    from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
    from src.analysis.presentation import analysis_visual_semantics as visual_semantics


_NOTE_FROM_TOP_IN = 0.36  # the scale note's top below the figure top: under the 12 pt title of a fixed-inch map grid
_NOTE_SIZE_PT = 9.0
_NOTE_COLOUR = "0.20"
_EMPTY_PREFIX_GID = "empty-prefix"
MARK_EVERY = 0.1  # a marker every tenth of the axes diagonal, so dense lines stay readable
PERCENTILE_SCALE = "5th\N{EN DASH}95th percentile"  # the ends of a state or error scale over its robust range (field_maps.percentile_scale)


def add_scale_note(figure: Figure, *, shared: bool = True, scale: str = PERCENTILE_SCALE) -> None:
    """
    Write the one colour-scale note under the title of a fixed-inch map grid: what sets the bars' ends and that arrows mark values beyond.

    ``scale`` names the rule of the ends, the robust percentile range by default; ``shared`` adds that the prediction uses the reference
    scale. Every map figure whose bars can show an arrow writes it here.
    """
    reference = "prediction uses the reference scale; " if shared else ""
    figure.text(
        0.5,
        1.0 - _NOTE_FROM_TOP_IN / figure.get_figheight(),
        f"Colour scales: {scale}; {reference}arrows mark values outside",
        ha="center",
        va="top",
        fontsize=_NOTE_SIZE_PT,
        color=_NOTE_COLOUR,
    )


def mean_error_text(values: np.ndarray | None, *, unit: str | None, exponent: int = 0) -> str:
    """
    Return the mean-error line of one error map: mean |e| of its finite ``values`` in ``unit`` (``None``: normalized), else "n/a".

    The mean runs over every finite cell of the field, independent of the display sampling. A bar that writes its labels as
    multiples of a power of ten (``exponent``, from ``shared_layout.error_format``) gets the mean as a multiple of that same power. The
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


def mark_empty_prefix(*axes: Axes, case_count: int, model: str | None = None) -> None:
    """
    Show on every given panel that its bounded prefix has no finite case, instead of drawing or failing.

    With ``model`` the line starts with the model's name; several models of one panel stack their lines.
    """
    text = empty_prefix_text(case_count) if model is None else f"{model}: {empty_prefix_text(case_count)}"
    for axis in axes:
        earlier = sum(item.get_gid() == _EMPTY_PREFIX_GID for item in axis.texts)
        axis.text(0.5, 0.5 - 0.08 * earlier, text, ha="center", va="center", transform=axis.transAxes, gid=_EMPTY_PREFIX_GID)


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


def dataset_title_context(title: str, groups: Mapping[str, Mapping[str, str]], counts: Mapping[str, int]) -> tuple[str, dict[str, str]]:
    """
    Return ``aggregate_title_context`` for the dataset panels of ``groups`` (``series.panel_groups``).

    ``counts`` is by frame key; a dataset panel counts the cases of its first model's frame, as the models of one dataset are case-matched.
    """
    return aggregate_title_context(title, {dataset: counts[next(iter(models.values()))] for dataset, models in groups.items()})


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
    extra: Mapping[str, Mapping[str, Artist]] | None = None,
    name_models: bool = True,
    name_roles: bool = True,
) -> None:
    """
    Write the legend of ``axes`` naming the models and, where panels differ in role, the roles; ``extra`` adds titled entry groups first.

    A model gets one entry in the colour of its run, a solid line or a patch for ``bars``, only when several models are drawn (one
    model is told by the panel titles) and ``name_models`` holds; a figure that colours something else drops it.
    Roles get a second group with one grey entry per role, in its line style or hatch, only when the panels differ in role, so no
    panel's role stands for the others; ``name_roles`` is off for scatter figures, whose points show no line style or hatch.
    Without any entry (a single model of a single role) no legend is
    written. A single titled group keeps its title also inside a single panel.
    """
    first: dict[str, str] = {}
    roles: dict[str, visual_semantics.SeriesStyle] = {}
    for by_model in groups.values():
        for model, key in by_model.items():
            first.setdefault(model, key)
            roles.setdefault(dataframe.dataset_role(frames[key]), styles[key])
    entries: dict[str, Artist] = {}
    named: dict[str, list[str]] = {}
    for title, group in (extra or {}).items():
        entries |= group
        named[title] = list(group)

    def artist(color: str, linestyle: visual_semantics.LineStyle, hatch: str, *, role: bool) -> Artist:
        if bars:
            return Patch(facecolor="white" if role else color, edgecolor="0.35" if role else "white", hatch=hatch)
        return Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.0)

    if name_models and len(first) > 1:
        entries |= {model: artist(styles[key].color, "-", "", role=False) for model, key in first.items()}
        named["Model"] = list(first)
    if name_roles and len(roles) > 1:
        entries |= {role: artist("0.35", style.linestyle, style.hatch, role=True) for role, style in roles.items()}
        named["Role"] = list(roles)
    if entries:
        figure_layout.add_legend(
            figure, axes, list(entries.values()), list(entries), title=next(iter(named)) if len(named) == 1 else None, groups=named
        )
