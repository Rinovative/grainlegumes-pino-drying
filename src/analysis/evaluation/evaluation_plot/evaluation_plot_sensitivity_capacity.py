"""Separate exact model-capacity evidence from exploratory metadata sensitivity.

Responsibilities:
  - Compare authoritative accuracy with exact trainable parameter counts, one shared count as a single centred tick
  - Compute Spearman associations for supported scientific metadata, a heatmap per run and dataset with the parameters named on
    each row's first shown panel, one colour bar as tall as the panel block
  - Plot quantile-binned metadata/error sensitivity in canonical parameter order, a point per parameter and channel

Design principles:
  - Capacity uses persisted provenance and never a proxy efficiency score
  - Point series get marker entries in their legend, and no line joins categories without an order; a model whose points carry
    several markers is a neutral circle beside a key of the material shapes
  - A pooled dataset weights its materials equally and keeps each material visible as a marker beside the pooled value
  - A pooled point is named the mean per-material error, never the artifact objective or the authoritative aggregate
  - Metadata filtering and labels come from one current-native presentation layer
  - Associations are exploratory and do not imply causality
  - Short, constant, missing, or non-finite evidence is disclosed or rejected

This module does NOT:
  - Infer architecture from run names, paths, or hidden defaults
  - Admit model or dataset identities for comparison
  - Perform hyperparameter search, causal inference, or model selection
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.artifacts import analysis_artifact_contracts as contracts
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_datasets as dataset_cuts
    from src.analysis.presentation import analysis_visual_semantics as visual_semantics

_MINIMUM_SENSITIVITY_CASES = 20
_VARIATION_FLOOR = 1e-12
_MINIMUM_QUANTILE_EDGES = 4
_MINIMUM_BIN_CASES = 2
_MINIMUM_SENSITIVITY_BINS = 3
_HEATMAP_CELL_IN = (0.9, 0.3)  # inches per field column and per parameter row of a heatmap
_HEATMAP_MARGIN_IN = 2.6  # tick labels and gap around the cells of one heatmap panel
_COLORBAR_PAD_IN = 0.1  # the heatmaps' shared colour bar beside the panel block
_CAPACITY_PAD = 2.0  # one shared capacity: the log x range reaches this factor below and above it
_TREND_DODGE = 0.15  # the channels of one parameter row stand this far apart (in rows), so equal sensitivities stay visible


def _trainable_count(frame: pd.DataFrame) -> int:
    """
    Read the exact trainable parameter count.

    Complete provenance is authoritative. A missing or non-positive count raises the comparison exception rather than falling
    back to labels, model objects, or proxy capacity measures.
    """
    provenance = dataframe.require_complete_provenance(frame)
    model = provenance.get("model")
    counts = model.get("parameter_counts") if isinstance(model, Mapping) else None
    if not isinstance(counts, Mapping):
        msg = "Capacity analysis requires model provenance with parameter counts."
        raise dataframe.ComparisonCompatibilityError(msg)
    trainable = counts.get("trainable")
    if isinstance(trainable, bool) or not isinstance(trainable, Integral) or int(trainable) <= 0:
        msg = "Capacity analysis requires an exact positive trainable parameter count."
        raise dataframe.ComparisonCompatibilityError(msg)
    return int(trainable)


def _prefix_objective(frame: pd.DataFrame, max_cases: int) -> tuple[Mapping[str, Any], int]:
    """Finalize the unchanged task objective evidence, available or not, on one admitted saved prefix."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    selected_count = min(max_cases, len(frame))
    selected = frame.iloc[:selected_count]
    aggregate = contracts.aggregate_normalized_group_macro_rmse(
        selected,
        output_groups=dataframe.require_complete_provenance(frame)["evaluator"]["output_groups"],
        train_scales=frame.attrs["train_scales"],
        reference_floors=frame.attrs["reference_floors"],
    )
    return aggregate, selected_count


def _capacity_legend(
    figure: Figure,
    axes: np.ndarray,
    datasets: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Mapping[str, tuple[str, ...]]],
    styles: Mapping[str, visual_semantics.SeriesStyle],
    material_styles: Mapping[str, visual_semantics.SeriesStyle],
) -> None:
    """
    Name the models, when several are drawn, and in a separate group the materials the markers show.

    A model is its colour with the marker its points share, a neutral circle where they carry several. The material key names the
    marker of an unpooled series and every small marker of a pooled one (grey shapes, or circles in the material colour where the
    figure colours by material). A model named like a material keeps its own entry (``figure_layout.add_legend``).
    """
    models: dict[str, tuple[str, set[str]]] = {}
    shapes: dict[str, Line2D] = {}
    for by_model in groups.values():
        for model, keys in by_model.items():
            style = styles[keys[0]]
            models.setdefault(model, (style.color, set()))[1].add(style.marker or "o")
            marked = keys if len(keys) > 1 else keys[:1] if style.marker else ()  # a pooled series' small markers, else its own marker
            for key in marked:
                own = material_styles[key] if len(keys) > 1 else style
                shape = Line2D([0], [0], color="0.35" if own.marker else own.color, marker=own.marker or "o", linestyle="", markersize=6)
                shapes.setdefault(series.material_name(datasets[key]), shape)
    entries = (
        {
            model: Line2D([0], [0], color=color, marker=next(iter(marks)) if len(marks) == 1 else "o", linestyle="", markersize=7)
            for model, (color, marks) in models.items()
        }
        if len(models) > 1
        else {}
    )
    named = {title: list(group.values()) for title, group in (("Model", entries), ("Material", shapes)) if group}
    if named:
        figure_layout.add_legend(figure, axes, [*entries.values(), *shapes.values()], [*entries, *shapes], groups=named)


def plot_capacity_accuracy(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = 50,
) -> Figure:
    """
    Plot exact capacity and accuracy, a panel per dataset with the models as points.

    A pooled dataset (one frame per material) shows one point per model at the equal-weight mean of its materials' objectives
    (``series.material_mean``), and each material's own objective as a small marker in its material style. That mean is not the
    artifact objective of the pooled cases, so title and axis name it the mean per-material normalized group macro RMSE; the
    materials of a pooled series must share one exact grid (``dataframe.validate_pooled_grids``).

    The declared configuration of each model is the Model-family configuration table (``plot_architecture_table``), not a
    second copy beside the panels.
    """
    dataframe.validate_comparison(datasets)
    groups = series.panel_series(datasets.labels)
    pools = [keys for models in groups.values() for keys in models.values() if len(keys) > 1]
    dataframe.validate_pooled_grids(datasets, pools)
    styles, material_styles = series.series_styles(datasets, pools), series.series_styles(datasets)
    # A pooled point is the mean of its materials' objectives, which is no artifact objective, so the figure names it as that mean.
    title, quantity = (
        ("Capacity versus mean per-material normalized group macro RMSE", "Mean per-material normalized group macro RMSE")
        if pools
        else ("Capacity versus authoritative aggregate accuracy", "Normalized group macro RMSE")
    )
    figure_title, count_headings = layout.dataset_title_context(title, datasets, layout.effective_case_counts(datasets, max_cases=max_cases))
    figure, axes = figure_layout.panel_figure(1, len(groups), sharey="row")
    unavailable: list[str] = []
    for axis, (dataset, models) in zip(axes[0], groups.items(), strict=True):
        drawn: set[int] = set()
        for model, keys in models.items():
            style = styles[keys[0]]
            trainable = _trainable_count(datasets[keys[0]])
            aggregates = {key: _prefix_objective(datasets[key], max_cases)[0] for key in keys}
            values = {key: [aggregate["value"] if aggregate["available"] else np.nan] for key, aggregate in aggregates.items()}
            pooled, per_material = series.material_mean(keys, values.__getitem__)
            if np.isfinite(pooled[0]):
                axis.scatter(trainable, pooled[0], marker=style.marker or "o", s=90, color=style.color)
                drawn.add(trainable)
            for key, (value,) in zip(keys, per_material, strict=True):
                if len(keys) > 1 and np.isfinite(value):
                    marker = material_styles[key]
                    axis.scatter(trainable, value, marker=marker.marker or "o", s=30, color=marker.color, edgecolors="black", linewidths=0.5)
                    drawn.add(trainable)  # a material drawn without its pooled point still sets the capacity range
                if not aggregates[key]["available"]:
                    # An unavailable objective is named with its support counts and never placed on the accuracy axis.
                    material = f" ({series.material_name(datasets[key])})" if len(keys) > 1 else ""
                    unavailable.append(f"{model} on {dataset}{material}: {validity_contract.unavailable_text(aggregates[key])}")
        figure_layout.set_log_scale(axis, "x")
        if len(drawn) == 1:  # one drawn capacity: its exact count as the one tick, centred on a log range padded alike on both sides
            (count,) = drawn
            axis.set_xlim(count / _CAPACITY_PAD, count * _CAPACITY_PAD)
            axis.set_xticks([count], [f"{count:,}"])
            axis.set_xticks([], minor=True)
        axis.set_xlabel(f"Trainable parameter count [{field_labels.display_unit('1')}]")
        axis.grid(alpha=0.25, which="both")
        figure_layout.set_panel_title(axis, count_headings[dataset] or dataset)
    figure_layout.label_rows_once(axes, f"{quantity} [{field_labels.display_unit('1')}]\n(lower is better)")
    _capacity_legend(figure, axes, datasets, groups, styles, material_styles)  # one data panel keeps its legend inside
    notes = ["Not plotted: " + "; ".join(unavailable)] if unavailable else []
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles, *notes, layout.POOLED_NOTE if pools else None))
    return figure


def _case_pairs(frame: pd.DataFrame, parameter: str, metric_column: str, *, max_cases: int) -> tuple[np.ndarray, np.ndarray]:
    """Return parameter and metric values of the saved prefix without the cases whose metric is non-finite."""
    prefix = frame.iloc[:max_cases]
    explanatory = np.asarray(pd.to_numeric(prefix[parameter], errors="raise"), dtype=float)
    if explanatory.size == 0 or not np.isfinite(explanatory).all():
        msg = f"Sensitivity parameter {parameter!r} must be finite numeric."
        raise ValueError(msg)
    response = np.asarray(pd.to_numeric(prefix[metric_column], errors="raise"), dtype=float)
    finite = validity_contract.finite_case_mask(response)
    return explanatory[finite], response[finite]


def _selected_fields(
    datasets: Mapping[str, pd.DataFrame],
    channels: tuple[str, ...],
) -> tuple[presentation.DisplayField, ...]:
    """Resolve one non-empty stable channel selection."""
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    by_label = {field.label: field for field in fields}
    magnitude = next((field for field in fields if field.key == "velocity_magnitude"), None)
    if magnitude is not None:
        by_label["U"] = magnitude
    if not channels or any(channel not in by_label for channel in channels):
        msg = f"Select at least one channel from {tuple(by_label)}."  # noqa: S608
        raise ValueError(msg)
    return tuple(by_label[channel] for channel in channels)


def _spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Spearman correlation without invoking SciPy on constant input."""
    x_constant, _x_tolerance = presentation.effectively_constant(x_values)
    y_constant, _y_tolerance = presentation.effectively_constant(y_values)
    if x_constant or y_constant:
        return float("nan")
    return float(pd.Series(x_values).corr(pd.Series(y_values), method="spearman"))


def plot_metadata_error_heatmap(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = 100,
) -> Figure:
    """Render the all-channel metadata association heatmaps: a column per dataset and a row per run (headed by its name when several)."""
    dataframe.validate_comparison(datasets)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    parameters = presentation.metadata_parameters(tuple(datasets.values()), max_cases=max_cases)
    if not parameters:
        msg = "Metadata sensitivity requires supported scientific parameters."
        raise ValueError(msg)
    groups = series.panel_groups(datasets.labels)
    figure_title, count_headings = layout.dataset_title_context(
        "Parameter-error association", datasets, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    models = list(dict.fromkeys(model for by_model in groups.values() for model in by_model))
    figure, axes = figure_layout.panel_figure(
        len(models),
        len(groups),
        panel_in=(_HEATMAP_CELL_IN[0] * len(fields) + _HEATMAP_MARGIN_IN, _HEATMAP_CELL_IN[1] * len(parameters) + _HEATMAP_MARGIN_IN),
        sharey="row",
    )
    parameter_labels = [presentation.metadata_label(parameter) for parameter in parameters]
    for column, (_dataset, by_model) in enumerate(groups.items()):
        for model, key in by_model.items():
            axis = axes[models.index(model), column]
            matrix = np.empty((len(parameters), len(fields)), dtype=float)
            for parameter_index, parameter in enumerate(parameters):
                for field_index, field in enumerate(fields):
                    x_values, y_values = _case_pairs(datasets[key], parameter, field.metric_column, max_cases=max_cases)
                    matrix[parameter_index, field_index] = _spearman(x_values, y_values)
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
            axis.set_xticks(np.arange(len(fields)), [field.metric_label("RMSE") for field in fields])
            axis.set_yticks(np.arange(len(parameters)), parameter_labels)
            for parameter_index, field_index in np.ndindex(matrix.shape):
                if np.isfinite(matrix[parameter_index, field_index]):
                    axis.text(field_index, parameter_index, f"{matrix[parameter_index, field_index]:+.2f}", ha="center", va="center", fontsize=8)
    for axis in axes.flat:
        axis.set_visible(bool(axis.images))
    for row in axes:  # a run without the first dataset names the parameters on its first shown panel, which the shared y hid
        next(axis for axis in row if axis.get_visible()).tick_params(axis="y", labelleft=True)
    for column, (dataset, by_model) in enumerate(groups.items()):  # the dataset heads its column on the top panel it has
        figure_layout.set_panel_title(axes[min(models.index(model) for model in by_model), column], count_headings[dataset] or dataset)
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes, models)
    # One bar as tall as the visible panel block, close beside it (its pad is a fraction of the block's width).
    figure.colorbar(
        image,
        ax=[axis for axis in axes.flat if axis.get_visible()],
        pad=_COLORBAR_PAD_IN / figure.get_figwidth(),
        label=f"Spearman correlation [{field_labels.display_unit('1')}]",
    )
    figure_layout.set_figure_title(figure, axes, figure_title)
    return figure


def _sensitivity(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return P90-P10 of quantile-binned response medians."""
    if x_values.size < _MINIMUM_SENSITIVITY_CASES or np.std(x_values) < _VARIATION_FLOOR or np.std(y_values) < _VARIATION_FLOOR:
        return float("nan")
    edges = np.unique(np.quantile(x_values, np.linspace(0.0, 1.0, 13)))
    if edges.size < _MINIMUM_QUANTILE_EDGES:
        return float("nan")
    assignments = np.clip(np.digitize(x_values, edges[1:-1]), 0, len(edges) - 2)
    medians = [
        float(np.median(y_values[assignments == index]))
        for index in range(len(edges) - 1)
        if np.count_nonzero(assignments == index) >= _MINIMUM_BIN_CASES
    ]
    if len(medians) < _MINIMUM_SENSITIVITY_BINS:
        return float("nan")
    return float(np.percentile(medians, 90) - np.percentile(medians, 10))


def plot_metadata_error_trends(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = 100,
    channels: tuple[str, ...] = ("p", "u", "v", "U"),
) -> Figure:
    """Render registry-order metadata sensitivities with channel checkboxes: a column per dataset, a row per model, a colour per channel."""
    dataframe.validate_comparison(datasets)
    fields = _selected_fields(datasets, channels)
    parameters = presentation.metadata_parameters(tuple(datasets.values()), max_cases=max_cases)
    if not parameters:
        msg = "Metadata trend analysis requires supported scientific parameters."
        raise ValueError(msg)
    groups = series.panel_groups(datasets.labels)
    figure_title, count_headings = layout.dataset_title_context(
        "Parameter sensitivity", datasets, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    # A channel keeps its colour whichever channels are selected, so the figure never depends on the run-colour limit. The
    # parameters are categories without an order between them, so each sensitivity is a point and no line joins them.
    colors = layout.channel_colors(presentation.shared_display_fields(tuple(datasets.values())))
    models = list(dict.fromkeys(model for by_model in groups.values() for model in by_model))
    figure, axes = figure_layout.panel_figure(len(models), len(groups), panel_in=(5.0, max(3.0, 0.3 * len(parameters) + 1.8)), sharey="row")
    y_positions = np.arange(len(parameters))
    for column, (dataset, by_model) in enumerate(groups.items()):
        for model, key in by_model.items():
            axis = axes[models.index(model), column]
            plotted_values: list[float] = []
            for index, field in enumerate(fields):
                values = []
                for parameter in parameters:
                    explanatory, response = _case_pairs(datasets[key], parameter, field.metric_column, max_cases=max_cases)
                    values.append(_sensitivity(explanatory, response))
                plotted_values.extend(values)
                dodge = (index - (len(fields) - 1) / 2.0) * _TREND_DODGE  # the channels of one parameter side by side
                axis.plot(values, y_positions + dodge, color=colors[field.key], marker="o", linestyle="", alpha=0.9)
            positive = np.asarray(plotted_values, dtype=float)
            positive = positive[np.isfinite(positive) & (positive > 0.0)]
            if positive.size:
                figure_layout.set_log_scale(axis, "x")
            else:
                axis.set_xlim(0.0, 1.0)
                axis.text(0.5, 0.5, "No resolved sensitivity for these cases", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlabel(f"Sensitivity on normalized RMSE [{field_labels.display_unit('1')}]\n(P90 minus P10 of binned medians)")
            axis.grid(True, which="both", axis="x", alpha=0.3)
        figure_layout.set_panel_title(axes[0, column], count_headings[dataset] or dataset)
    for row_axes in axes:
        row_axes[0].set_yticks(y_positions, [presentation.metadata_label(parameter) for parameter in parameters])
        row_axes[0].invert_yaxis()
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes, models)
    # The column titles name each dataset and so its role; the key names the channels by their point colour.
    channels_legend = {field.matplotlib_label: Line2D([0], [0], color=colors[field.key], marker="o", linestyle="") for field in fields}
    figure_layout.add_legend(figure, axes, list(channels_legend.values()), list(channels_legend), title="Channel")
    figure_layout.set_figure_title(figure, axes, figure_title)
    return figure
