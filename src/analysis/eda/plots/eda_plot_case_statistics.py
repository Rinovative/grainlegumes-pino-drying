"""Plot case-level statistics and metadata distributions for EDA.

Responsibilities:
  - Plot disjoint generated-case and scalar-material parameter distributions
  - Plot case-level field-value summaries
  - Build interactive case-statistics viewers

Design principles:
  - Statistics are aggregated per case from an explicit ordered prefix
  - Incremental caches extend monotonically as notebook case limits increase
  - Dataset values show on their shared display scales (°C, %, water as wet-basis moisture with bulk case means) and are
    labelled independently in the active label style

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
from matplotlib.lines import Line2D
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from src import analysis
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_transient as transient_plots
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
_HISTOGRAM_ALPHA = 0.35
_LEGEND_LINE_WIDTH = 2.0  # thin enough that a dashed legend line shows its dashes
_FIELD_STATISTICS = ("min", "mean", "max")
_LEGEND_WRAP_WIDTH = transient_plots.LEGEND_WRAP_WIDTH


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

    ``loaded`` is the consumed ordered prefix length. ``cols`` retains flattened
    numeric leaf histories keyed by parameter path.
    """

    loaded: int
    cols: dict[str, list[float]]
    units: dict[str, str]


class _ParamCache(TypedDict):
    """
    Track incrementally loaded schema-declared material scalars for one dataset.

    This internal notebook transport mirrors :class:`_StatCache` but keeps the
    semantically separate ``scalar_conditioning`` namespace.
    """

    loaded: int
    cols: dict[str, list[float]]


class _FieldCache(TypedDict):
    """
    Track per-field min/mean/max histories for an incrementally loaded prefix.

    Values remain on each task field's stored scale and are partitioned by field
    and statistic so extending a viewer does not recompute earlier cases.
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
    if vals.size < 2:  # noqa: PLR2004
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


def _draw_dataset_histogram(axis: Axes, values: np.ndarray, style: SeriesStyle) -> None:
    """Draw one dataset's histogram and, for non-constant values, its kernel density in the dataset's color and line style."""
    artists = histograms.plot_histogram(axis, values, bins=_infer_bins(values), color=style.color, alpha=_HISTOGRAM_ALPHA)
    if artists.constant_line is None:
        try:
            kde = gaussian_kde(values)
            x = np.linspace(values.min(), values.max(), 300)
            bw = artists.bin_edges[1] - artists.bin_edges[0]
            axis.plot(x, kde(x) * values.size * bw, color=style.color, linestyle=style.linestyle, lw=1.6)
        except LinAlgError:
            pass


def _draw_panel_histograms(axis: Axes, series: Mapping[str, np.ndarray], styles: Mapping[str, SeriesStyle], unit: str | None) -> None:
    """Draw every dataset's values of one panel; their common power of ten goes into the x label's unit bracket, never into offset text."""
    series = {name: values for name, values in series.items() if values.size}
    exponent = field_maps.state_exponent(min(v.min() for v in series.values()), max(v.max() for v in series.values())) if series else 0
    for name, values in series.items():
        _draw_dataset_histogram(axis, values / 10.0**exponent, styles[name])
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    if exponent:
        axis.set_xlabel(plot_layout.label_with_exponent(f"Value [{unit or '-'}]", exponent))


def _dataset_legend(figure: Figure, axes: np.ndarray, active: Sequence[str], styles: Mapping[str, SeriesStyle]) -> None:
    """Write the dataset legend of a histogram grid."""
    handles = [Line2D([], [], lw=_LEGEND_LINE_WIDTH, color=styles[name].color, linestyle=styles[name].linestyle) for name in active]
    names = [display_labels.wrapped_dataset_display_label(name, width=_LEGEND_WRAP_WIDTH) for name in active]
    figure_layout.add_legend(figure, axes, handles, names, title="Dataset")


def _hist_grid(
    *,
    data_by_dataset: dict[str, dict[str, np.ndarray]],
    active_datasets: list[str],
    columns: list[str],
    title: str,
    styles: Mapping[str, SeriesStyle],
    units_by_column: Mapping[str, str] | None = None,
) -> Figure:
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
        Authoritative units used for unit-bearing subplot headings.

    Returns
    -------
    Figure
        Matplotlib Figure containing the histogram grid; "Counts" labels the first panel of every row.

    """
    if not columns:
        figure, axes = figure_layout.panel_figure(1, 1, panel_in=(8.0, 3.2))
        axes[0, 0].set_axis_off()
        axes[0, 0].text(
            0.5, 0.5, "No numeric parameter evidence is available for the selected cases.", ha="center", va="center", transform=axes[0, 0].transAxes
        )
        figure_layout.set_figure_title(figure, axes, title)
        return figure

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
        _draw_panel_histograms(axis, {name: vals[np.isfinite(vals)] for name, vals in series.items()}, styles, unit)
        figure_layout.set_panel_title(axis, key if unit is None else field_labels.field_label_with_unit(key, unit))
        axis.grid(True, linestyle="--", alpha=0.25)
    figure_layout.label_rows_once(grid_axes, "Counts")
    figure_layout.set_figure_title(figure, grid_axes, title)
    _dataset_legend(figure, grid_axes, active_datasets, styles)
    return figure


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
    Caches grow monotonically. Reducing the case control after loading a larger
    prefix does not discard already accumulated observations.

    """
    names = list(datasets.keys())
    cache: dict[str, _StatCache] = {name: {"loaded": 0, "cols": {}, "units": {}} for name in names}

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
                meta = row["meta"]
                if not isinstance(meta, dict):
                    message = "EDA case metadata must be a mapping."
                    raise TypeError(message)
                values = _case_parameter_values(meta)
                units = _case_parameter_units(meta)
                for key, value in values.items():
                    entry["cols"].setdefault(key, []).append(value)
                    unit = units.get(key)
                    if unit is not None:
                        previous = entry["units"].setdefault(key, unit)
                        if previous != unit:
                            message = f"Generated parameter {key!r} has inconsistent units."
                            raise ValueError(message)

            entry["loaded"] = max_cases

        active = list(datasets)
        keys = list(dict.fromkeys(k for n in active for k in cache[n]["cols"]))

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
                    [value for value in cache[name]["cols"].get(key, []) if isinstance(value, (int, float, np.floating))],
                    units_by_column.get(key, "1"),
                    field=key,
                )
                for key in keys
            }
            for name in active
        }

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=_visible_metadata_statistic_keys(keys),
            title=f"Generated case parameters — {max_cases} cases",
            styles=styles,
            units_by_column=units_by_column,
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
    Caches grow monotonically with the largest requested prefix. Reducing the
    control later reuses, rather than truncates, accumulated observations.

    """
    names = list(datasets.keys())
    cache: dict[str, _ParamCache] = {n: {"loaded": 0, "cols": {}} for n in names}

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
                meta = row["meta"]
                if not isinstance(meta, dict):
                    message = "EDA case metadata must be a mapping."
                    raise TypeError(message)
                scalar = row.get("scalar_conditioning")
                if not isinstance(scalar, Mapping):
                    continue
                for key, value in _flatten_dict_raw(scalar).items():
                    if key in _SCALAR_MATERIAL_KEYS:
                        entry["cols"].setdefault(key, []).append(value)

            entry["loaded"] = max_cases

        active = list(datasets)
        keys = list(dict.fromkeys(k for n in active for k in cache[n]["cols"]))

        data = {
            name: {
                key: field_labels.display_values(
                    [value for value in cache[name]["cols"].get(key, []) if isinstance(value, (int, float, np.floating))],
                    _SCALAR_MATERIAL_UNITS[key],
                    field=key,
                )
                for key in keys
            }
            for name in active
        }

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=_scalar_material_parameter_keys(keys),
            title=f"Scalar material conditioning — {max_cases} cases",
            styles=styles,
            units_by_column=_SCALAR_MATERIAL_UNITS,
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
    Values are clipped to the 1st--99th percentile for display only. Source
    frames are not mutated. Cached observations remain when a user
    lowers the case-count control.

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
    availability = capabilities.availability_note(resolution)

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
                    if array.size == 0:
                        continue
                    entry["data"][field]["min"].append(float(np.nanmin(array)))
                    entry["data"][field]["mean"].append(capabilities.field_display_mean(df, row, field, raw))
                    entry["data"][field]["max"].append(float(np.nanmax(array)))

            entry["loaded"] = max_cases

        active = list(datasets)

        figure, axes = figure_layout.panel_figure(len(field_names), len(_FIELD_STATISTICS), panel_in=_FIELD_PANEL_IN)
        for axis in axes[-1]:
            axis.set_xlabel("Value")
        for i, f in enumerate(field_names):
            unit = capabilities.field_display_unit(datasets[resolution.datasets_by_field[f][0]], f)
            for j, stat in enumerate(_FIELD_STATISTICS):
                ax = axes[i, j]
                panel = {name: _clip_for_plot(np.asarray(cache[name]["data"][f][stat])) for name in active if f in cache[name]["data"]}
                _draw_panel_histograms(ax, panel, styles, unit)
                if i == 0:
                    figure_layout.set_panel_title(ax, stat)
                ax.grid(True, linestyle="--", alpha=0.25)
        figure_layout.label_rows_once(axes, "Counts")
        figure_layout.set_row_headers(
            figure,
            axes,
            [capabilities.field_quantity_label(datasets[resolution.datasets_by_field[f][0]], f, mathtext=True) for f in field_names],
        )
        figure_layout.set_figure_title(figure, axes, f"Field value distributions per channel — {max_cases} cases", note=availability or None)
        _dataset_legend(figure, axes, active, styles)
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
