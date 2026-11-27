"""Plot case-level statistics and metadata distributions for EDA.

Responsibilities:
  - Plot disjoint generated-case and scalar-material parameter distributions, the sampled parameters titled «group · name» as in
    the generation-input tables
  - Plot case-level field-value summaries
  - Count the drawn cases of the Datasets that hold values of a grid in its title, and the eligible ones a bound leaves out
    in its note
  - Build interactive case-statistics viewers

Design principles:
  - Statistics are aggregated per case from an explicit ordered prefix
  - Incremental caches extend monotonically as notebook case limits increase; a lower limit reads only its case prefix
  - Dataset values show on their shared display scales (°C, %, water as wet-basis moisture with bulk case means) and are
    labelled independently in the active label style
  - Overlaid datasets draw unfilled outlines on the panel's common bins in their line styles; one constant that every shown
    dataset holds is written once with its unit and coverage
  - Outlines nest, an earlier one wider and constant lines alike, so a coincident outline of another colour stays visible;
    datasets of one colour whose outlines coincide are named under the panel's x label, bin edges and counts unchanged
  - Figure text names a dataset by its plotted label (``eda_capabilities.plotted_labels``); caches and styles stay keyed by
    the dataset key

This module does NOT:
  - Compute frequency-domain EDA or model-evaluation diagnostics
  - Own generic case-count navigation and widget rendering
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import same_color
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from src import analysis
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.analysis.generation_inputs import generation_input_table_schema as table_schema
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.presentation import analysis_moisture_display as moisture_display
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as plot_layout
from src.datasets.contracts import dataset_contracts_transient as transient_contract

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.eda.eda_dataframe import PublishedTensorValue
    from src.analysis.presentation.analysis_visual_semantics import SeriesStyle

_STATISTICS_EXCLUDED_PARAMETER_KEYS = frozenset(
    {
        "completion_target_wet_fraction_limit",
        "runtime_target_wet_fraction_limit",
    }
)
_SCALAR_MATERIAL_FIELDS = transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning
_SCALAR_MATERIAL_KEYS = frozenset(field.name for field in _SCALAR_MATERIAL_FIELDS)
_SCALAR_MATERIAL_NORMALIZED_KEYS = frozenset(key.casefold() for key in _SCALAR_MATERIAL_KEYS)
_SCALAR_MATERIAL_UNITS = {field.name: field.unit for field in _SCALAR_MATERIAL_FIELDS}
_SPATIAL_DIMENSIONS = 2
_FOOTNOTE_INCHES = transient_plots.FOOTNOTE_INCHES
_STAT_PANEL_IN = (4.0, 2.8)  # one parameter histogram
_FIELD_PANEL_IN = (4.0, 2.4)  # one min, mean or max histogram of a field
_OUTLINE_WIDTH = 1.2  # a histogram's bar outline, in the dataset's line style
_OUTLINE_NEST_STEP = 1.4  # each earlier dataset's outline is this much wider, so outlines on shared bins stay visible one inside the other
_LEGEND_LINE_WIDTH = 2.0  # thin enough that a dashed legend line shows its dashes
_FIELD_STATISTICS = ("min", "mean", "max")
_LEGEND_WRAP_WIDTH = transient_plots.LEGEND_WRAP_WIDTH
_KDE_MINIMUM_OBSERVATIONS = 20  # a kernel density over fewer cases suggests a smoothness the sample cannot show
_CLIP_MINIMUM_OBSERVATIONS = 100  # 1st-99th percentile display clipping needs at least 100 cases to trim more than the extremes
_HISTOGRAM_X_TICKS = 4  # at most this many x tick intervals, so neighbouring value labels never touch
_NOTE_LINE_PT = 11.0  # one 9 pt note line under a histogram's x label
_NO_PARAMETERS_NOTE = (
    "No sampled Generation parameters in the selected cases: a published Dataset runtime view does not carry them; "
    "its scalar material conditioning has its own view."
)
_NO_SCALARS_NOTE = "No schema-declared scalar material conditioning in the selected cases."


def plot_published_tensor(
    value: PublishedTensorValue,
    *,
    x: np.ndarray,
    y: np.ndarray,
    display_spatial_stride: display_grid.DisplayStride = "auto",
    case_label: str | None = None,
    transition: tuple[float, float] | None = None,
    rho_bu_dry: np.ndarray | None = None,
) -> Figure:
    """
    Show a stored tensor on its actual support and summarize its full grid; the title names the case and the transition in hours.

    Every support shows its values on their display scale (°C, % RH); a water density becomes % w.b. with the case's dry
    bulk density ``rho_bu_dry``, which it therefore requires.
    """
    unit = moisture_display.field_display_unit(value.name, value.unit)
    label = capabilities.representation_display_name(value.name, value.representation, mathtext=True)
    if moisture_display.is_moisture_field(value.name):
        if rho_bu_dry is None:
            message = f"Showing {value.name} as wet-basis moisture requires the case's dry bulk density."
            raise ValueError(message)
        values = moisture_display.wet_basis_percent(value.array, rho_bu_dry)
    else:
        values = field_labels.display_values(value.array, value.unit, field=value.name)
    context = " · ".join(part for part in (case_label, None if transition is None else f"{transition[0]:g} h → {transition[1]:g} h") if part)
    heading = f"{value.group}: {label}"
    footnote = f"{value.representation}; full support min={float(np.min(values)):g}, max={float(np.max(values)):g}"
    if values.ndim == _SPATIAL_DIMENSIONS:
        scale = transient_plots.spatial_field_scale(value.name, value.unit, value.representation, values)
        extent = display_grid.physical_extent(x, y)
        geometry = plot_layout.plan_geometry(1, 1, (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False, title=bool(context))
        figure, axes, slots = plot_layout.map_grid(geometry, extent=extent)
        indices = display_grid.display_indices(values.shape, display_spatial_stride)
        try:
            image = axes[0, 0].pcolormesh(
                display_grid.display_sample(x, indices),
                display_grid.display_sample(y, indices),
                display_grid.display_sample(values, indices),
                shading="auto",
                **scale.pcolormesh_kwargs(),
            )
            plot_layout.add_field_map_colorbar(figure, image, slots[0, 0], scale, label=f"{label} [{unit}]")
            plot_layout.set_column_headers(axes, [heading], [])
            if context:
                plot_layout.set_map_title(figure, context)
            figure.text(_FOOTNOTE_INCHES[0] / figure.get_figwidth(), _FOOTNOTE_INCHES[1] / figure.get_figheight(), footnote, fontsize=8)
        except Exception:
            plt.close(figure)
            raise
        return figure
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(6.0, 4.0))
    axis = axes[0, 0]
    if values.ndim == 1 and value.support == "inlet boundary":
        inlet_x = x[y == np.min(y)]  # the inlet edge y = min(y) runs along x
        order = np.argsort(inlet_x)
        axis.plot(inlet_x[order], values[order])
        axis.set_xlabel("Inlet x [m]")
        axis.set_ylabel(f"{label} [{unit}]")
    elif values.ndim == 0:
        axis.set_axis_off()
        axis.text(0.5, 0.5, f"{label} = {float(values):g} [{unit}]\n{value.support}", ha="center", va="center", transform=axis.transAxes)
    else:
        plt.close(figure)
        message = f"Unsupported published tensor support for {value.name}: {values.shape}."
        raise ValueError(message)
    figure_layout.set_panel_title(axis, f"{context}\n{heading}" if context else heading)
    figure.supxlabel(footnote, fontsize=figure_layout.TICK_PT)
    return figure


# ============================================================================
# TYPES
# ============================================================================


class _StatCache(TypedDict):
    """
    Track incrementally loaded sampled generated-case parameters for one dataset.

    ``loaded`` is the consumed ordered prefix length. ``cases`` retains each case's
    flattened numeric leaves keyed by parameter path, in case order.
    """

    loaded: int
    cases: list[dict[str, float]]
    units: dict[str, str]


class _ParamCache(TypedDict):
    """
    Track incrementally loaded schema-declared material scalars for one dataset.

    This internal notebook transport mirrors :class:`_StatCache` but keeps the
    semantically separate ``scalar_conditioning`` namespace.
    """

    loaded: int
    cases: list[dict[str, float]]


class _FieldCache(TypedDict):
    """
    Track per-field min/mean/max histories for an incrementally loaded prefix.

    Values remain on each task field's stored scale and are partitioned by field
    and statistic so extending a viewer does not recompute earlier cases; a case
    without values holds NaN, so list positions stay case positions.
    """

    loaded: int
    data: dict[str, dict[str, list[float]]]


# ============================================================================
# HELPERS
# ============================================================================


def _as_float(x: Any) -> float | None:
    """
    Convert x to float if possible.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    float or None
        Converted float value, or None if conversion is not possible.

    """
    if x is None:
        return None

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, (list, tuple, np.ndarray)):
        array = np.asarray(x)
        if array.size == 1:
            item = array.item()
            return _as_float(item) if item is not x else None
        return None

    # everything else (str, dict, enum, ...)
    return None


_RNG_IMPLEMENTATION_KEYS = frozenset(
    {
        "generator_state",
        "generator_states",
        "numpy_rng_state",
        "python_rng_state",
        "random_state",
        "rng_counter",
        "rng_counters",
        "rng_state",
        "rng_states",
        "rng_type",
        "torch_rng_state",
    }
)


def _is_rng_implementation_key(key: str) -> bool:
    """Return whether one exact metadata key owns RNG implementation state."""
    normalized = key.strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized in _RNG_IMPLEMENTATION_KEYS


def _flatten_dict_raw(dct: Mapping[str, Any]) -> dict[str, float]:
    """
    Flatten nested dictionary into a single-level dictionary with float values.

    Parameters
    ----------
    dct : dict[str, Any]
        Input nested dictionary.

    Returns
    -------
    dict[str, float]
        Flattened dictionary with float values, excluding exact RNG
        implementation-state branches.

    """
    out: dict[str, float] = {}

    def _rec(key: str, obj: Any) -> None:
        """Flatten nested mappings/sequences while retaining numeric leaf paths."""
        if isinstance(obj, Mapping):
            for child_key, value in obj.items():
                if _is_rng_implementation_key(str(child_key)):
                    continue
                _rec(f"{key}_{child_key}", value)
            return
        if isinstance(obj, np.void) and obj.dtype.names:
            for child_key in obj.dtype.names:
                _rec(f"{key}_{child_key}", obj[child_key])
            return
        asdict_method = getattr(obj, "_asdict", None)
        if callable(asdict_method):
            record = asdict_method()
            if isinstance(record, Mapping):
                _rec(key, record)
                return

        if isinstance(obj, (list, tuple, np.ndarray)) and not np.isscalar(obj):
            for i, v in enumerate(obj):
                _rec(f"{key}_{i}", v)
            return

        val = _as_float(obj)
        if val is not None:
            out[key] = val

    for k, v in dct.items():
        if _is_rng_implementation_key(str(k)):
            continue
        _rec(k, v)

    return out


def _case_parameter_values(meta: Mapping[str, Any]) -> dict[str, float]:
    """Return every numeric generated-case parameter leaf without fitting."""
    parameters = meta.get("parameters")
    return _flatten_dict_raw(parameters) if isinstance(parameters, Mapping) else {}


def _case_parameter_units(meta: Mapping[str, Any]) -> dict[str, str]:
    """Return authoritative generated-case parameter units when retained."""
    parameter_units = meta.get("parameter_units")
    if not isinstance(parameter_units, Mapping):
        return {}
    return {str(key): str(unit) for key, unit in parameter_units.items() if isinstance(unit, str) and unit}


def _metadata_statistic_key_is_visible(key: str) -> bool:
    """Apply Plot 1-1's exact structural and runtime-limit denylist."""
    normalized = key.strip().casefold().replace("-", "_").replace(" ", "_")
    return (
        not normalized.startswith("geometry_")
        and normalized not in _STATISTICS_EXCLUDED_PARAMETER_KEYS
        and normalized not in _SCALAR_MATERIAL_NORMALIZED_KEYS
    )


def _visible_metadata_statistic_keys(keys: Sequence[str]) -> list[str]:
    """Retain every discovered Plot 1-1 field except the explicit denylist."""
    return [key for key in keys if _metadata_statistic_key_is_visible(key)]


def _scalar_material_parameter_keys(keys: Sequence[str]) -> list[str]:
    """Return available schema-declared material scalars in contract order."""
    available = set(keys)
    return [field.name for field in _SCALAR_MATERIAL_FIELDS if field.name in available]


def _prefix_columns(cases: Sequence[Mapping[str, float]], max_cases: int) -> dict[str, list[float]]:
    """Return every key's values over the first ``max_cases`` cached cases in case order, so a lowered control never shows more."""
    columns: dict[str, list[float]] = {}
    for case in cases[:max_cases]:
        for key, value in case.items():
            columns.setdefault(key, []).append(value)
    return columns


def _contributing_frames(
    datasets: Mapping[str, pd.DataFrame], cases: Mapping[str, Sequence[Mapping[str, Any]]], keys: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Return, for each dataset with one, the rows of its drawn ``cases`` that give a finite value of one of ``keys``, for the title count."""
    rows = {
        name: [
            index
            for index, case in enumerate(cases[name])
            if any(isinstance(value := case.get(key), (int, float, np.floating)) and math.isfinite(value) for key in keys)
        ]
        for name in datasets
    }
    return {name: datasets[name].iloc[indices] for name, indices in rows.items() if indices}


def _clip_for_plot(vals: np.ndarray, *, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    """
    Clip extreme values based on percentiles for better plotting.

    Parameters
    ----------
    vals : np.ndarray
        Input values.
    q_low : float
        Lower percentile threshold.
    q_high : float
        Upper percentile threshold.

    Returns
    -------
    np.ndarray
        Clipped values.

    """
    vals = vals[np.isfinite(vals)]
    if vals.size < _CLIP_MINIMUM_OBSERVATIONS:  # a sparse sample's percentiles fall inside its extremes and would drop real cases
        return vals

    lo, hi = np.percentile(vals, [q_low, q_high])
    if lo >= hi:
        return vals

    return vals[(vals >= lo) & (vals <= hi)]


# ============================================================================
# DATA-DRIVEN BINNING + LAYOUT
# ============================================================================


def _infer_bins(vals: np.ndarray, *, min_bins: int = 10, max_bins: int = 80) -> int:
    """
    Infer number of histogram bins using Freedman-Diaconis rule.

    Parameters
    ----------
    vals : np.ndarray
        Input values.
    min_bins : int
        Minimum number of bins.
    max_bins : int
        Maximum number of bins.

    Returns
    -------
    int
        Inferred number of bins.

    """
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:  # noqa: PLR2004
        return 1

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 1
    if vmin == vmax:
        return 1

    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(q75 - q25)

    if iqr <= 0.0:
        return int(np.clip(min_bins, 1, max_bins))

    bw = 2.0 * iqr * vals.size ** (-1.0 / 3.0)
    if not np.isfinite(bw) or bw <= 0.0:
        return int(np.clip(min_bins, 1, max_bins))

    bins = int(np.ceil((vmax - vmin) / bw))
    return int(np.clip(bins, 1, max_bins))


def _infer_ncols(n_items: int, *, max_cols: int = 5) -> int:
    """
    Infer number of columns for subplot grid layout.

    Parameters
    ----------
    n_items : int
        Number of items to plot.
    max_cols : int
        Maximum number of columns.

    Returns
    -------
    int
        Inferred number of columns.

    """
    return min(max_cols, max(1, int(np.ceil(np.sqrt(n_items)))))


# ============================================================================
# HIST GRID (GENERIC)
# ============================================================================


def _draw_dataset_histogram(
    axis: Axes, values: np.ndarray, style: SeriesStyle, bins: np.ndarray | None = None, *, linewidth: float = _OUTLINE_WIDTH
) -> None:
    """
    Draw one dataset's histogram on ``bins`` (else its own) and, for enough non-constant values, its kernel density.

    Bars are unfilled outlines and a constant is one vertical line, both of ``linewidth`` in the dataset's color and line style,
    so overlaid datasets never blend into colors the legend does not show.
    """
    artists = histograms.plot_histogram(
        axis,
        values,
        bins=_infer_bins(values) if bins is None else bins,
        color=style.color,
        fill=False,
        edgecolor=style.color,
        linestyle=style.linestyle,
        linewidth=linewidth,
    )
    if artists.constant_line is not None:
        artists.constant_line.set(linestyle=style.linestyle, linewidth=linewidth)
    elif values.size >= _KDE_MINIMUM_OBSERVATIONS:
        try:
            kde = gaussian_kde(values)
            x = np.linspace(values.min(), values.max(), 300)
            bw = artists.bin_edges[1] - artists.bin_edges[0]
            axis.plot(x, kde(x) * values.size * bw, color=style.color, linestyle=style.linestyle, lw=1.6)
        except LinAlgError:
            pass


def _draw_panel_histograms(
    axis: Axes, series: Mapping[str, np.ndarray], styles: Mapping[str, SeriesStyle], unit: str | None, plotted: Mapping[str, str]
) -> tuple[int, list[tuple[str, str]]]:
    """
    Draw every dataset's values of one panel on common bins and thinned x ticks; return the axis' power of ten and its note lines.

    A known display ``unit`` and the values' common power of ten share the x label's unit bracket (never offset text), so the
    unit stands in one place for every panel; a panel without unit gets a label only for a power of ten. Where several datasets
    hold one and the same constant, the panel writes it once with its unit and the datasets it covers instead of stacking lines.
    Outlines nest, constant lines alike, so a coincident outline of another colour shows around a later one; in one colour they
    merge, and the returned note lines (text, colour) name every such dataset by its ``plotted`` label (``_merged_outlines``).
    """
    shown = {name: values for name, values in series.items() if values.size}
    pooled = np.concatenate(tuple(shown.values())) if shown else np.empty(0)
    if len(shown) > 1 and np.all(pooled == pooled[0]):
        coverage = f"all {len(shown)}" if len(shown) == len(series) else f"{len(shown)} of {len(series)}"
        suffix = "" if unit in {None, field_labels.display_unit("1")} else f" {unit}"
        axis.text(0.5, 0.5, f"constant {pooled[0]:g}{suffix}\nin {coverage} Datasets", transform=axis.transAxes, ha="center", va="center")
        axis.set(xticks=[], yticks=[], xlabel="")
        return 0, []
    exponent = field_maps.state_exponent(float(pooled.min()), float(pooled.max())) if shown else 0
    bins = np.histogram_bin_edges(pooled / 10.0**exponent, bins=_infer_bins(pooled)) if shown else None
    scaled = {name: values / 10.0**exponent for name, values in shown.items()}
    for rank, (name, values) in enumerate(scaled.items()):  # the first outline is the widest and the last the thinnest, drawn over the others
        _draw_dataset_histogram(axis, values, styles[name], bins, linewidth=_OUTLINE_WIDTH + _OUTLINE_NEST_STEP * (len(scaled) - 1 - rank))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=_HISTOGRAM_X_TICKS))
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    if unit is not None or exponent:
        axis.set_xlabel(plot_layout.label_with_exponent(f"Value [{unit or '-'}]", exponent))
    return exponent, [] if bins is None else _merged_outlines(scaled, bins, styles, plotted)


def _merged_outlines(
    series: Mapping[str, np.ndarray], bins: np.ndarray, styles: Mapping[str, SeriesStyle], plotted: Mapping[str, str]
) -> list[tuple[str, str]]:
    """
    Return a note line (text, colour) for every dataset whose outline lies on an earlier dataset's of its colour.

    Nesting cannot show two such outlines apart: two constants at one value, or bars whose counts on ``bins`` equal the other
    dataset's in every bin they fill. The line names both datasets by their ``plotted`` labels, and says «part of» where the
    fuller dataset also fills other bins.
    """
    constants = {name: float(values[0]) for name, values in series.items() if np.all(values == values[0])}
    counts = {name: np.histogram(values, bins)[0] for name, values in series.items() if name not in constants}

    def merged(earlier: str, name: str) -> str | None:
        if earlier in constants or name in constants:
            return f"{plotted[name]} coincides with {plotted[earlier]}" if constants.get(earlier) == constants.get(name) else None
        if np.array_equal(counts[name], counts[earlier]):
            return f"{plotted[name]} coincides with {plotted[earlier]}"
        for sparse, full in ((name, earlier), (earlier, name)):
            if np.all((counts[sparse] == 0) | (counts[sparse] == counts[full])):
                return f"{plotted[sparse]} coincides with part of {plotted[full]}"
        return None

    names = list(series)
    lines = []
    for index, name in enumerate(names):
        same = (other for other in names[:index] if same_color(styles[other].color, styles[name].color))
        note = next((text for other in same if (text := merged(other, name)) is not None), None)
        if note is not None:
            lines.append((note, styles[name].color))
    return lines


def _clipped_tail_lines(
    cohorts: Mapping[str, np.ndarray], shown: Mapping[str, np.ndarray], styles: Mapping[str, SeriesStyle], exponent: int
) -> list[tuple[str, str]]:
    """
    Return a note line (text, colour) per series that display clipping left cases out of: the shown range and the count.

    The range reads in the axis' scaled values: ``exponent`` is the power of ten ``_draw_panel_histograms`` divided them by.
    """
    lines = []
    for name, values in cohorts.items():
        if shown[name].size < values.size:
            lower, upper = (float(value) / 10.0**exponent for value in (shown[name].min(), shown[name].max()))
            left_out = f"{values.size - shown[name].size} of {values.size} cases left out"
            lines.append((f"shown {transient_plots.range_text(lower, upper, digits=3)}; {left_out}", styles[name].color))
    return lines


def _note_below_x_label(axis: Axes, lines: Sequence[tuple[str, str]]) -> None:
    """Write every note line (text, colour) right-aligned below the panel's x label, one under the other."""
    for line, (text, color) in enumerate(lines):
        axis.annotate(
            text,
            xy=(1.0, 0.0),
            xycoords=("axes fraction", axis.xaxis.label),
            xytext=(0.0, -line * _NOTE_LINE_PT),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=figure_layout.TICK_PT,
            color=color,
        )


def _dataset_legend(figure: Figure, axes: np.ndarray, active: Sequence[str], styles: Mapping[str, SeriesStyle], plotted: Mapping[str, str]) -> None:
    """Write the dataset legend of a histogram grid, every dataset under its ``plotted`` label."""
    handles = [Line2D([], [], lw=_LEGEND_LINE_WIDTH, color=styles[name].color, linestyle=styles[name].linestyle) for name in active]
    names = [display_labels.wrapped_dataset_display_label(plotted[name], width=_LEGEND_WRAP_WIDTH) for name in active]
    figure_layout.add_legend(figure, axes, handles, names, title="Dataset")


def _hist_grid(
    *,
    data_by_dataset: dict[str, dict[str, np.ndarray]],
    active_datasets: list[str],
    columns: list[str],
    title: str,
    styles: Mapping[str, SeriesStyle],
    units_by_column: Mapping[str, str] | None = None,
    empty_note: str = _NO_PARAMETERS_NOTE,
    titles: Mapping[str, str] | None = None,
    note: str = "",
    plotted: Mapping[str, str] | None = None,
) -> Figure | str:
    """
    Create a grid of histogram plots for multiple datasets and columns.

    Parameters
    ----------
    data_by_dataset : dict[str, dict[str, np.ndarray]]
        Data organized by dataset name and column name.
    active_datasets : list[str]
        List of active dataset names to plot.
    columns : list[str]
        List of column names to plot.
    title : str
        Title for the entire figure.
    styles : Mapping[str, SeriesStyle]
        Color and line style of every active dataset, resolved by the viewer over all loaded datasets.
    units_by_column : Mapping[str, str] | None, optional
        Authoritative stored units; a column with one shows its symbol as title and its display unit in the x label.
    empty_note : str, optional
        The one-line note that replaces the grid when no column is available.
    titles : Mapping[str, str] | None, optional
        Panel title of every column; without it a column with a unit shows its field symbol, any other its name.
    note : str, optional
        The figure note under the title, such as the drawn of the eligible cases.
    plotted : Mapping[str, str] | None, optional
        The label the figure writes for every active dataset (``eda_capabilities.plotted_labels``); without it, its name.

    Returns
    -------
    Figure | str
        Matplotlib Figure containing the histogram grid ("Counts" labels the first panel of every row), or ``empty_note``.

    """
    if not columns:
        return empty_note
    plotted = {name: name for name in active_datasets} if plotted is None else plotted

    ncols = _infer_ncols(len(columns))
    nrows = math.ceil(len(columns) / ncols)
    figure, grid_axes = figure_layout.panel_figure(nrows, ncols, panel_in=_STAT_PANEL_IN)
    for axis in grid_axes.flat[len(columns) :]:
        axis.set_visible(False)

    for axis in grid_axes.flat[max(0, len(columns) - ncols) : len(columns)]:  # the last panel of every column, also of one that ends a row early
        axis.set_xlabel("Value")
    for axis, key in zip(grid_axes.flat, columns, strict=False):
        unit = None if units_by_column is None else units_by_column.get(key)
        series = {name: data_by_dataset[name].get(key, np.array([])) for name in active_datasets}
        shown_unit = None if unit is None else field_labels.display_unit(unit, field=key)
        finite = {name: vals[np.isfinite(vals)] for name, vals in series.items()}
        _note_below_x_label(axis, _draw_panel_histograms(axis, finite, styles, shown_unit, plotted)[1])
        default_title = key if unit is None else field_labels.field_label(key, mathtext=True)
        figure_layout.set_panel_title(axis, default_title if titles is None else titles[key])
        axis.grid(True, linestyle="--", alpha=0.25)
    figure_layout.label_rows_once(grid_axes, "Counts")
    figure_layout.set_figure_title(figure, grid_axes, title, note=note or None)
    _dataset_legend(figure, grid_axes, active_datasets, styles, plotted)
    return figure


def _parameter_title(key: str) -> str:
    """Return a sampled Generation parameter's panel title «group · name» of the generation-input tables, else its field label."""
    for spec in (table_schema.parameter_row_spec(key), table_schema.boundary_row_spec(key)):
        if spec.section != table_schema.UNCATEGORIZED_SECTION:
            return f"{spec.category} · {spec.label}"
    return field_labels.field_label(key, mathtext=True)


# ============================================================================
# META STATISTICS
# ============================================================================


def plot_meta_statistics(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    resume_cases: int | None = None,
    host: analysis.ui.panel.ViewPanel | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> analysis.ui.panel.ViewPanel:
    """
    Build sampled generated-case parameter distributions.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Labelled EDA frames whose authoritative case metadata contains sampled
        parameter leaves. Frame order defines dataset display order.
    styles : Mapping[str, SeriesStyle]
        Color and line style of every dataset, resolved by the viewer over all loaded datasets.
    resume_cases : int | None, optional
        Case count a former viewer of the same datasets showed; it is kept up to the largest dataset.
    host : ViewPanel | None, optional
        View whose render builds this viewer; the viewer publishes only while that render is current and the host is shown.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ViewPanel
        Case-count control and incrementally cached histograms for the given
        dataset set.

    Raises
    ------
    ValueError
        If no dataset is available.

    Notes
    -----
    Nested mapping/sequence paths are flattened and non-numeric leaves omitted.
    Caches grow monotonically; a lowered case control shows only its case prefix.

    """
    names = list(datasets.keys())
    plotted = capabilities.plotted_labels(datasets)
    cache: dict[str, _StatCache] = {name: {"loaded": 0, "cases": [], "units": {}} for name in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure | str:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.

        Returns
        -------
        Figure | str
            Matplotlib Figure containing the histogram grid, or a one-line note when no parameter is available.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for _, row in df.iloc[entry["loaded"] : max_cases].iterrows():
                meta = row["meta"]
                if not isinstance(meta, dict):
                    message = "EDA case metadata must be a mapping."
                    raise TypeError(message)
                values = _case_parameter_values(meta)
                units = _case_parameter_units(meta)
                entry["cases"].append(values)
                for key in values:
                    unit = units.get(key)
                    if unit is not None:
                        previous = entry["units"].setdefault(key, unit)
                        if previous != unit:
                            message = f"Generated parameter {key!r} has inconsistent units."
                            raise ValueError(message)

            entry["loaded"] = max_cases

        active = list(datasets)
        columns = {name: _prefix_columns(cache[name]["cases"], max_cases) for name in active}
        keys = list(dict.fromkeys(k for n in active for k in columns[n]))

        units_by_column: dict[str, str] = {}
        for key in keys:
            available_units = {cache[name]["units"][key] for name in active if key in cache[name]["units"]}
            if len(available_units) > 1:
                message = f"Generated parameter {key!r} has incompatible units across datasets."
                raise ValueError(message)
            if available_units:
                units_by_column[key] = next(iter(available_units))
        data = {
            name: {
                key: field_labels.display_values(
                    [value for value in columns[name].get(key, []) if isinstance(value, (int, float, np.floating))],
                    units_by_column.get(key, "1"),
                    field=key,
                )
                for key in keys
            }
            for name in active
        }

        visible = _visible_metadata_statistic_keys(keys)
        drawn = _contributing_frames(datasets, {name: cache[name]["cases"][:max_cases] for name in active}, visible)
        count, note = transient_plots.case_count_text(drawn, max_cases, plotted=plotted)
        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=visible,
            title=f"Generated case parameters — {count}",
            styles=styles,
            units_by_column=units_by_column,
            titles={key: _parameter_title(key) for key in visible},
            note=note,
            plotted=plotted,
        )

    return analysis.ui.viewers.case_count_view(
        _plot,
        datasets=cast("Mapping[Hashable, pd.DataFrame]", datasets),
        start_cases=100,
        resume_cases=resume_cases,
        host=host,
        step_size=50,
        export_state=export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
    )


# ============================================================================
# 1-2. META PARAMETERS (AUTO)
# ============================================================================


def plot_meta_parameters(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    resume_cases: int | None = None,
    host: analysis.ui.panel.ViewPanel | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> analysis.ui.panel.ViewPanel:
    """
    Build schema-declared scalar material-conditioning distributions.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Labelled transient EDA frames whose ``scalar_conditioning`` mappings
        contain schema-declared case-global material inputs.
    styles : Mapping[str, SeriesStyle]
        Color and line style of every dataset, resolved by the viewer over all loaded datasets.
    resume_cases : int | None, optional
        Case count a former viewer of the same datasets showed; it is kept up to the largest dataset.
    host : ViewPanel | None, optional
        View whose render builds this viewer; the viewer publishes only while that render is current and the host is shown.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ViewPanel
        Case-count control and material-scalar histograms on display units
        derived from authoritative stored units.

    Raises
    ------
    ValueError
        If no dataset is available.

    Notes
    -----
    Caches grow monotonically with the largest requested prefix; a lowered
    control shows only its case prefix.

    """
    names = list(datasets.keys())
    plotted = capabilities.plotted_labels(datasets)
    cache: dict[str, _ParamCache] = {n: {"loaded": 0, "cases": []} for n in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure | str:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.

        Returns
        -------
        Figure | str
            Matplotlib Figure containing the histogram grid, or a one-line note when no scalar is available.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for _, row in df.iloc[entry["loaded"] : max_cases].iterrows():
                meta = row["meta"]
                if not isinstance(meta, dict):
                    message = "EDA case metadata must be a mapping."
                    raise TypeError(message)
                scalar = row.get("scalar_conditioning")
                flat = _flatten_dict_raw(scalar) if isinstance(scalar, Mapping) else {}
                entry["cases"].append({key: value for key, value in flat.items() if key in _SCALAR_MATERIAL_KEYS})

            entry["loaded"] = max_cases

        active = list(datasets)
        columns = {name: _prefix_columns(cache[name]["cases"], max_cases) for name in active}
        keys = list(dict.fromkeys(k for n in active for k in columns[n]))

        data = {
            name: {
                key: field_labels.display_values(
                    [value for value in columns[name].get(key, []) if isinstance(value, (int, float, np.floating))],
                    _SCALAR_MATERIAL_UNITS[key],
                    field=key,
                )
                for key in keys
            }
            for name in active
        }

        scalars = _scalar_material_parameter_keys(keys)
        drawn = _contributing_frames(datasets, {name: cache[name]["cases"][:max_cases] for name in active}, scalars)
        count, note = transient_plots.case_count_text(drawn, max_cases, plotted=plotted)
        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=scalars,
            title=f"Scalar material conditioning — {count}",
            styles=styles,
            units_by_column=_SCALAR_MATERIAL_UNITS,
            empty_note=_NO_SCALARS_NOTE,
            note=note,
            plotted=plotted,
        )

    return analysis.ui.viewers.case_count_view(
        _plot,
        datasets=cast("Mapping[Hashable, pd.DataFrame]", datasets),
        start_cases=100,
        resume_cases=resume_cases,
        host=host,
        step_size=50,
        export_state=export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
    )


# ============================================================================
# 1-3. FIELD VALUE DISTRIBUTIONS (AUTO)
# ============================================================================


def plot_field_value_distributions(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    resume_cases: int | None = None,
    host: analysis.ui.panel.ViewPanel | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> analysis.ui.panel.ViewPanel:
    """
    Build per-field distributions of case minima, means, and maxima.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Non-empty labelled EDA frames. Every column except ``meta``, ``x``, and
        ``y`` in the first frame is treated as a comparable field array.
    styles : Mapping[str, SeriesStyle]
        Color and line style of every dataset, resolved by the viewer over all loaded datasets.
    resume_cases : int | None, optional
        Case count a former viewer of the same datasets showed; it is kept up to the largest dataset.
    host : ViewPanel | None, optional
        View whose render builds this viewer; the viewer publishes only while that render is current and the host is shown.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ViewPanel
        Case-count control and a field-by-statistic histogram grid backed by
        monotonic ordered-prefix caches.

    Raises
    ------
    ValueError
        If datasets are empty.

    Notes
    -----
    From 100 cases on, values are clipped to the 1st--99th percentile for display only; every
    series that loses cases names the count and its shown range below its panel. Source frames
    are not mutated; a lowered case-count control shows only its case prefix.

    """
    names = list(datasets)
    resolution = capabilities.resolve_fields(
        datasets,
        view="field_statistics",
    )
    field_names = list(resolution.fields)
    cache: dict[str, _FieldCache] = {
        name: {
            "loaded": 0,
            "data": {field: {"min": [], "mean": [], "max": []} for field in field_names if name in resolution.datasets_by_field[field]},
        }
        for name in names
    }
    plotted = capabilities.plotted_labels(datasets)
    availability = capabilities.availability_note(resolution, plotted=plotted)

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Plot function for case count viewer.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Dictionary of dataset names to DataFrames.

        Returns
        -------
        Figure
            Matplotlib Figure containing the histogram grid.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for _, row in df.iloc[entry["loaded"] : max_cases].iterrows():
                for field in entry["data"]:
                    raw = capabilities.field_values(df, row, field)
                    array = capabilities.field_display_values(df, row, field, raw)
                    empty = array.size == 0
                    entry["data"][field]["min"].append(np.nan if empty else float(np.nanmin(array)))
                    entry["data"][field]["mean"].append(np.nan if empty else capabilities.field_display_mean(df, row, field, raw))
                    entry["data"][field]["max"].append(np.nan if empty else float(np.nanmax(array)))

            entry["loaded"] = max_cases

        active = list(datasets)

        figure, axes = figure_layout.panel_figure(len(field_names), len(_FIELD_STATISTICS), panel_in=_FIELD_PANEL_IN)
        for i, f in enumerate(field_names):
            unit = capabilities.field_display_unit(datasets[resolution.datasets_by_field[f][0]], f)
            for j, stat in enumerate(_FIELD_STATISTICS):
                ax = axes[i, j]
                prefixes = {name: np.asarray(cache[name]["data"][f][stat][:max_cases]) for name in active if f in cache[name]["data"]}
                cohorts = {name: values[np.isfinite(values)] for name, values in prefixes.items()}
                panel = {name: _clip_for_plot(values) for name, values in cohorts.items()}
                exponent, notes = _draw_panel_histograms(ax, panel, styles, unit, plotted)
                _note_below_x_label(ax, [*notes, *_clipped_tail_lines(cohorts, panel, styles, exponent)])
                figure_layout.set_panel_title(ax, stat)  # on every row, so a row reads without the one above
                ax.grid(True, linestyle="--", alpha=0.25)
        figure_layout.label_rows_once(axes, "Counts")
        figure_layout.set_row_headers(
            figure,
            axes,
            [capabilities.field_display_name(datasets[resolution.datasets_by_field[f][0]], f, mathtext=True) for f in field_names],
        )
        count, count_note = transient_plots.case_count_text(capabilities.contributing_frames(datasets, resolution), max_cases, plotted=plotted)
        note = " ".join(part for part in (count_note, availability) if part)
        figure_layout.set_figure_title(figure, axes, f"Field value distributions per channel — {count}", note=note or None)
        _dataset_legend(figure, axes, active, styles, plotted)
        return figure

    return analysis.ui.viewers.case_count_view(
        _plot,
        datasets=cast("Mapping[Hashable, pd.DataFrame]", datasets),
        start_cases=100,
        resume_cases=resume_cases,
        host=host,
        step_size=50,
        export_state=export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
    )
