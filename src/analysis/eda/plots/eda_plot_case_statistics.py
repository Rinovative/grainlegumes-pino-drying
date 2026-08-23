"""
eda_plot_case_statistics.py

Plot case-level statistics and metadata distributions for EDA.

Responsibilities:
  - Plot disjoint generated-case and scalar-material parameter distributions
  - Plot case-level field-value summaries
  - Build interactive case-statistics viewers

Design principles:
  - Statistics are aggregated per case from an explicit ordered prefix
  - Incremental caches extend monotonically as notebook case limits increase
  - Dataset values remain on their stored scales and are labelled independently

This module does NOT:
  - Compute frequency-domain EDA or model-evaluation diagnostics
  - Own generic case-count navigation and widget rendering
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from src import analysis
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as plot_layout
from src.datasets.contracts import dataset_contracts_transient as transient_contract

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    CheckboxGroup = analysis.ui.components.CheckboxGroup

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
_HISTOGRAM_SUPTITLE_Y = 0.98
_SCALAR_MATERIAL_TITLE_GAP = 3.0 * 1.75 * (_HISTOGRAM_SUPTITLE_Y - 0.97)
_SCALAR_MATERIAL_GRID_TOP = _HISTOGRAM_SUPTITLE_Y - _SCALAR_MATERIAL_TITLE_GAP


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
    values: dict[str, float] = {}
    generator = meta.get("generator")
    if isinstance(generator, Mapping):
        for block in generator.values():
            if not isinstance(block, Mapping):
                continue
            parameters = block.get("parameters")
            if isinstance(parameters, Mapping):
                values.update(_flatten_dict_raw(parameters))
    parameters = meta.get("parameters")
    if isinstance(parameters, Mapping):
        values.update(_flatten_dict_raw(parameters))
    return values


def _case_parameter_units(meta: Mapping[str, Any]) -> dict[str, str]:
    """Return authoritative generated-case parameter units when retained."""
    units: dict[str, str] = {}
    generator = meta.get("generator")
    if isinstance(generator, Mapping):
        for block in generator.values():
            if not isinstance(block, Mapping):
                continue
            for unit_key in ("parameter_units", "sampled_units", "units"):
                block_units = block.get(unit_key)
                if isinstance(block_units, Mapping):
                    units.update({str(key): str(unit) for key, unit in block_units.items() if isinstance(unit, str) and unit})
    parameter_units = meta.get("parameter_units")
    if isinstance(parameter_units, Mapping):
        units.update({str(key): str(unit) for key, unit in parameter_units.items() if isinstance(unit, str) and unit})
    return units


def _dataset_colors(dataset_names: Sequence[str]) -> dict[str, str]:
    """Return stable colorblind-friendly colors keyed by concise labels."""
    identities = tuple(
        visual_semantics.DatasetVisualIdentity(
            canonical_identity=name,
            label=name,
        )
        for name in dataset_names
    )
    colors = visual_semantics.dataset_colors(identities)
    return {name: colors[name] for name in dataset_names}


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


def _selected_datasets(dataset_selector: CheckboxGroup) -> list[str]:
    """
    Get list of selected dataset names from the dataset selector widget.

    Parameters
    ----------
    dataset_selector : CheckboxGroup
        Dataset selector widget.

    Returns
    -------
    list[str]
        List of selected dataset names.

    Raises
    ------
    ValueError
        If no dataset is selected.

    """
    active = [n for n, cb in dataset_selector.boxes.items() if cb.value]
    if not active:
        msg = "Select at least one dataset."
        raise ValueError(msg)
    return active


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


def _hist_grid(
    *,
    data_by_dataset: dict[str, dict[str, np.ndarray]],
    active_datasets: list[str],
    columns: list[str],
    title: str,
    units_by_column: Mapping[str, str] | None = None,
    grid_top: float = 0.90,
    row_spacing: float = 0.34,
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
    units_by_column : Mapping[str, str] | None, optional
        Authoritative units used for unit-bearing subplot headings.
    grid_top : float, optional
        Explicit upper GridSpec boundary.
    row_spacing : float, optional
        Explicit GridSpec spacing between histogram rows.

    Returns
    -------
    Figure
        Matplotlib Figure containing the histogram grid.

    """
    dataset_colors = _dataset_colors(active_datasets)
    if not columns:
        figure, axis = plt.subplots(figsize=(8.0, 3.2), layout=None)
        figure.subplots_adjust(left=0.08, right=0.96, bottom=0.15, top=0.82)
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No numeric parameter evidence is available for the selected cases.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        plot_layout.set_suptitle_over_axes(
            figure,
            title,
            (axis,),
            y=_HISTOGRAM_SUPTITLE_Y,
        )
        return figure

    ncols = _infer_ncols(len(columns))
    nrows = math.ceil(len(columns) / ncols)

    fig, grid_axes, ax_leg = plot_layout.subplots_with_legend_column(
        rows=nrows,
        columns=ncols,
        column_width=4.0,
        row_height=2.8,
        legend_width=3.6,
        top=grid_top,
        hspace=row_spacing,
    )

    axes: list[Axes] = []
    for row in range(nrows):
        for column in range(ncols):
            index = row * ncols + column
            axis = grid_axes[row, column]
            if index < len(columns):
                axes.append(axis)
            else:
                axis.set_axis_off()

    legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active_datasets]

    for ax, key in zip(axes, columns, strict=False):
        for name in active_datasets:
            vals = data_by_dataset[name].get(key, np.array([]))
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            color = dataset_colors[name]
            bins = _infer_bins(vals)

            artists = histograms.plot_histogram(
                ax,
                vals,
                bins=bins,
                color=color,
                alpha=0.35,
            )
            if artists.constant_line is None:
                try:
                    kde = gaussian_kde(vals)
                    x = np.linspace(vals.min(), vals.max(), 300)
                    bw = artists.bin_edges[1] - artists.bin_edges[0]
                    ax.plot(x, kde(x) * vals.size * bw, color=color, lw=1.6)
                except LinAlgError:
                    pass

        unit = None if units_by_column is None else units_by_column.get(key)
        ax.set_title(
            key if unit is None else field_labels.field_label_with_unit(key, unit),
            fontsize=10,
        )
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)

    plot_layout.configure_bottom_occupied_row_xlabels(
        axes,
        columns=ncols,
        label="Value",
        hide_upper_tick_labels=False,
    )
    ax_leg.legend(
        legend_handles,
        active_datasets,
        title="Dataset",
        loc="upper left",
    )
    plot_layout.set_suptitle_over_axes(
        fig,
        title,
        axes,
        y=_HISTOGRAM_SUPTITLE_Y,
    )

    return fig


# ============================================================================
# META STATISTICS
# ============================================================================


def plot_meta_statistics(
    *,
    datasets: dict[str, pd.DataFrame],
    allow_dataset_selection: bool = True,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """
    Build sampled generated-case parameter distributions.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Labelled EDA frames whose authoritative case metadata contains sampled
        parameter leaves. Frame order defines dataset display order.
    allow_dataset_selection : bool, optional
        Show the maintained dataset-checkbox group inside this viewer.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ipywidgets.VBox
        Case-count controls, dataset checkboxes, and incrementally cached
        histograms for the selected dataset set.

    Raises
    ------
    ValueError
        If no dataset is available or the user disables every dataset.

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
            units = {cache[name]["units"][key] for name in active if key in cache[name]["units"]}
            if len(units) > 1:
                message = f"Generated parameter {key!r} has incompatible units across datasets."
                raise ValueError(message)
            if units:
                units_by_column[key] = next(iter(units))
        data = {
            name: {
                key: field_labels.display_values(
                    [value for value in cache[name]["cols"].get(key, []) if isinstance(value, (int, float, np.floating))],
                    units_by_column.get(key, "1"),
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
            units_by_column=units_by_column,
            grid_top=0.96,
            row_spacing=0.425,
        )

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        allow_dataset_selection=allow_dataset_selection,
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
    allow_dataset_selection: bool = True,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """
    Build schema-declared scalar material-conditioning distributions.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Labelled transient EDA frames whose ``scalar_conditioning`` mappings
        contain schema-declared case-global material inputs.
    allow_dataset_selection : bool, optional
        Show the maintained dataset-checkbox group inside this viewer.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ipywidgets.VBox
        Case-count controls, dataset checkboxes, and material-scalar histograms
        on display units derived from authoritative stored units.

    Raises
    ------
    ValueError
        If no dataset is available or the user disables every dataset.

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
            units_by_column=_SCALAR_MATERIAL_UNITS,
            grid_top=_SCALAR_MATERIAL_GRID_TOP,
            row_spacing=0.50,
        )

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        allow_dataset_selection=allow_dataset_selection,
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
    allow_dataset_selection: bool = True,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """
    Build per-field distributions of case minima, means, and maxima.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Non-empty labelled EDA frames. Every column except ``meta``, ``x``, and
        ``y`` in the first frame is treated as a comparable field array.
    allow_dataset_selection : bool, optional
        Show the maintained dataset-checkbox group inside this viewer.
    export_state : dict[str, Any] | None, optional
        Explicit panel-local current-figure state.
    export_plot_name, export_title : str | None, optional
        Export identity supplied by the active dropdown entry.

    Returns
    -------
    ipywidgets.VBox
        Case-count controls, dataset checkboxes, and a field-by-statistic
        histogram grid backed by monotonic ordered-prefix caches.

    Raises
    ------
    ValueError
        If datasets are empty or no dataset remains selected.

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
    dataset_colors = _dataset_colors(names)
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
                    array = capabilities.field_display_values(
                        df,
                        field,
                        capabilities.field_values(df, row, field),
                    )
                    if array.size == 0:
                        continue
                    entry["data"][field]["min"].append(float(np.nanmin(array)))
                    entry["data"][field]["mean"].append(float(np.nanmean(array)))
                    entry["data"][field]["max"].append(float(np.nanmax(array)))

            entry["loaded"] = max_cases

        active = list(datasets)

        nrows = len(field_names)
        ncols = 3

        fig, axes, ax_leg = plot_layout.subplots_with_legend_column(
            rows=nrows,
            columns=ncols,
            column_width=4.0,
            row_height=2.4,
            legend_width=3.6,
            top=0.97,
        )

        legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active]

        for i, f in enumerate(field_names):
            for j, stat in enumerate(["min", "mean", "max"]):
                ax = axes[i][j]

                for name in active:
                    field_data = cache[name]["data"].get(f)
                    if field_data is None:
                        continue
                    vals = np.asarray(field_data[stat])
                    vals = _clip_for_plot(vals)
                    if vals.size == 0:
                        continue

                    color = dataset_colors[name]
                    bins = _infer_bins(vals)
                    artists = histograms.plot_histogram(
                        ax,
                        vals,
                        bins=bins,
                        color=color,
                        alpha=0.35,
                    )
                    if artists.constant_line is None:
                        try:
                            kde = gaussian_kde(vals)
                            x = np.linspace(vals.min(), vals.max(), 300)
                            bw = artists.bin_edges[1] - artists.bin_edges[0]
                            ax.plot(x, kde(x) * vals.size * bw, color=color, lw=1.6)
                        except LinAlgError:
                            pass

                if i == 0:
                    ax.set_title(stat)
                if j == 0:
                    ax.set_ylabel("Count")
                    plot_layout.add_channel_row_label(
                        ax,
                        capabilities.field_quantity_label(
                            datasets[resolution.datasets_by_field[f][0]],
                            f,
                            mathtext=True,
                        ),
                    )

                ax.grid(True, linestyle="--", alpha=0.25)

        plot_layout.configure_bottom_occupied_row_xlabels(
            axes.ravel(),
            columns=ncols,
            label="Value",
            hide_upper_tick_labels=False,
        )
        ax_leg.legend(
            legend_handles,
            active,
            title="Dataset",
            loc="upper left",
        )
        plot_layout.set_suptitle_over_axes(
            fig,
            f"Field value distributions per channel — {max_cases} cases",
            axes.ravel(),
        )
        if availability:
            fig.text(0.06, 0.012, availability, ha="left", va="bottom", fontsize=8)
        fig.subplots_adjust(
            top=0.96,
            bottom=0.08 if availability else 0.06,
            left=0.06,
            right=0.97,
        )

        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        allow_dataset_selection=allow_dataset_selection,
        export_state=export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
    )
