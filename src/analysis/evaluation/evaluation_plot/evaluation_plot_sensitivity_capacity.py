"""Separate exact model-capacity evidence from exploratory metadata sensitivity.

Responsibilities:
  - Compare authoritative accuracy with exact trainable parameter counts
  - Compute Spearman associations for supported scientific metadata
  - Plot quantile-binned metadata/error sensitivity in canonical parameter order

Design principles:
  - Capacity uses persisted provenance and never a proxy efficiency score
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
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_datasets as dataset_cuts

_MINIMUM_SENSITIVITY_CASES = 20
_VARIATION_FLOOR = 1e-12
_MINIMUM_QUANTILE_EDGES = 4
_MINIMUM_BIN_CASES = 2
_MINIMUM_SENSITIVITY_BINS = 3
_HEATMAP_CELL_IN = (0.9, 0.3)  # inches per field column and per parameter row of a heatmap
_HEATMAP_MARGIN_IN = 2.6  # tick labels and gap around the cells of one heatmap panel


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


def plot_capacity_accuracy(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = 50,
) -> Figure:
    """
    Plot exact capacity and accuracy, a panel per dataset with the models as points.

    The declared configuration of each model is the Model-family configuration table (``plot_architecture_table``), not a
    second copy beside the panels.
    """
    dataframe.validate_comparison(datasets)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    figure_title, count_headings = layout.dataset_title_context(
        "Capacity versus authoritative aggregate accuracy", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    figure, axes = figure_layout.panel_figure(1, len(groups), sharey="row")
    unavailable: list[str] = []
    for axis, (dataset, models) in zip(axes[0], groups.items(), strict=True):
        for model, key in models.items():
            frame, style = datasets[key], styles[key]
            trainable = _trainable_count(frame)
            aggregate, _selected_count = _prefix_objective(frame, max_cases)
            if aggregate["available"]:
                axis.scatter(trainable, aggregate["value"], marker=style.marker or "o", s=90, color=style.color)
            else:
                # An unavailable objective is named with its support counts and never placed on the accuracy axis.
                unavailable.append(f"{model} on {dataset}: {validity_contract.unavailable_text(aggregate)}")
        figure_layout.set_log_scale(axis, "x")
        axis.set_xlabel("exact trainable parameter count")
        axis.grid(alpha=0.25, which="both")
        figure_layout.set_panel_title(axis, count_headings[dataset] or dataset)
    figure_layout.label_rows_once(axes, f"Normalized group macro RMSE [{field_labels.display_unit('1')}]\n(lower is better)")
    layout.add_series_legend(figure, axes, datasets, groups, styles, name_roles=False)  # one data panel keeps its legend inside
    not_plotted = "Not plotted: " + "; ".join(unavailable) if unavailable else None
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles, not_plotted))
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
    """Render the all-channel metadata association heatmaps: datasets side by side, or underneath each other with a column per model."""
    dataframe.validate_comparison(datasets)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    parameters = presentation.metadata_parameters(tuple(datasets.values()), max_cases=max_cases)
    if not parameters:
        msg = "Metadata sensitivity requires supported scientific parameters."
        raise ValueError(msg)
    groups = series.panel_groups(datasets.labels)
    figure_title, count_headings = layout.dataset_title_context(
        "Parameter-error association", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    side_by_side = all(len(models) == 1 for models in groups.values())
    figure, axes = figure_layout.panel_figure(
        1 if side_by_side else len(groups),
        len(groups) if side_by_side else max(len(models) for models in groups.values()),
        panel_in=(_HEATMAP_CELL_IN[0] * len(fields) + _HEATMAP_MARGIN_IN, _HEATMAP_CELL_IN[1] * len(parameters) + _HEATMAP_MARGIN_IN),
        sharey="row",
    )
    parameter_labels = [presentation.metadata_label(parameter) for parameter in parameters]
    for index, (dataset, models) in enumerate(groups.items()):
        for slot, (model, key) in enumerate(models.items()):
            axis = axes[0, index] if side_by_side else axes[index, slot]
            matrix = np.empty((len(parameters), len(fields)), dtype=float)
            for parameter_index, parameter in enumerate(parameters):
                for field_index, field in enumerate(fields):
                    x_values, y_values = _case_pairs(datasets[key], parameter, field.metric_column, max_cases=max_cases)
                    matrix[parameter_index, field_index] = _spearman(x_values, y_values)
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
            axis.set_xticks(np.arange(len(fields)), [field.metric_label("RMSE") for field in fields])
            axis.set_yticks(np.arange(len(parameters)), parameter_labels)
            figure_layout.set_panel_title(axis, (count_headings[dataset] or dataset) if side_by_side else model)
            for row, column in np.ndindex(matrix.shape):
                if np.isfinite(matrix[row, column]):
                    axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    for axis in axes.flat:
        axis.set_visible(bool(axis.images))
    if not side_by_side:
        figure_layout.set_row_headers(figure, axes, [count_headings[dataset] or dataset for dataset in groups])
    figure.colorbar(
        image, ax=[axis for axis in axes.flat if axis.get_visible()], shrink=0.85, label=f"Spearman correlation [{field_labels.display_unit('1')}]"
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
    # Only the role's line style is needed (colour is the channel), so the figure never depends on the run-colour limit.
    styles = {
        key: visual_semantics.series_style(role="id" if (split := str(frame.attrs.get("dataset_role"))) == "eval" else split, pooled=True)
        for key, frame in datasets.items()
    }
    figure_title, count_headings = layout.dataset_title_context(
        "Parameter sensitivity", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    # A channel keeps its colour whichever channels are selected; the line style is the role.
    colors = layout.channel_colors(presentation.shared_display_fields(tuple(datasets.values())))
    models = list(dict.fromkeys(model for by_model in groups.values() for model in by_model))
    figure, axes = figure_layout.panel_figure(len(models), len(groups), panel_in=(5.0, max(3.0, 0.3 * len(parameters) + 1.8)), sharey="row")
    y_positions = np.arange(len(parameters))
    for column, (dataset, by_model) in enumerate(groups.items()):
        for model, key in by_model.items():
            axis = axes[models.index(model), column]
            plotted_values: list[float] = []
            for field in fields:
                values = []
                for parameter in parameters:
                    explanatory, response = _case_pairs(datasets[key], parameter, field.metric_column, max_cases=max_cases)
                    values.append(_sensitivity(explanatory, response))
                plotted_values.extend(values)
                axis.plot(values, y_positions, color=colors[field.key], marker="o", linestyle=styles[key].linestyle, alpha=0.9)
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
    channels_legend = {field.matplotlib_label: Line2D([0], [0], color=colors[field.key], linewidth=2.0) for field in fields}
    layout.add_series_legend(figure, axes, datasets, groups, styles, extra={"Channel": channels_legend}, name_models=False)
    figure_layout.set_figure_title(figure, axes, figure_title)
    return figure
