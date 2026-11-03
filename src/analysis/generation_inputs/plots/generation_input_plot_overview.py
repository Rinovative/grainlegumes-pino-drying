"""Present generation-input comparison and dataset overview tables.

Responsibilities:
  - Render compact A/B dataset, case, identity, and provenance context
  - Render parameter and field summaries as A, Mean A, B, and Mean B
  - Show every unique dataset case with raw parameter values
  - Show compact per-case field summaries without standardization
  - Plot campaign failures by sampled parameter, per material, coloured by the typed failure reason

Design principles:
  - Units appear once inside row or column labels
  - Four-value comparison tables normalize colors independently by row
  - Dataset overview tables place parameters by row and actual cases by column

This module does NOT:
  - Render maps, standardize values, or load input files
  - Expose input-generation IDs or hashes in normal selectors
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.ui import analysis_ui_plot_layout as plot_layout
from src.analysis.ui import tables

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ipywidgets as widgets
    from matplotlib.figure import Figure

_FAILURE_PANEL_COLUMNS: Final = 3


def case_context(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render compact side-by-side scientific and technical context."""
    return tables.styled_dataframe(
        diagnostics.case_context_table(
            first,
            mean_a,
            second,
            mean_b,
        ),
        title="Case context",
        shade_constant=False,
    )


def parameters(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render every scalar parameter under the exact four-column contract."""
    table = diagnostics.parameter_comparison_table(
        first,
        mean_a,
        second,
        mean_b,
    )
    return tables.grouped_styled_dataframes(
        diagnostics.grouped_table_sections(table),
        title=(f"Parameters — Mean A n = {mean_a.case_count}; Mean B n = {mean_b.case_count}"),
        columns=2,
        shade_constant=True,
        row_local=True,
    )


def field_summaries(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render meaningful field statistics under the four-column contract."""
    table = diagnostics.field_summary_comparison_table(
        first,
        mean_a,
        second,
        mean_b,
    )
    return tables.grouped_styled_dataframes(
        diagnostics.grouped_table_sections(table),
        title=(f"Field summaries — Mean A n = {mean_a.case_count}; Mean B n = {mean_b.case_count}"),
        columns=2,
        shade_constant=True,
        row_local=True,
    )


def dataset_cases_parameters(
    dataset: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render raw values for all unique cases in one canonical dataset."""
    table = diagnostics.dataset_parameter_table(dataset)
    return tables.grouped_styled_dataframes(
        diagnostics.grouped_table_sections(table),
        title=(f"{dataset.material_family.replace('_', ' ').title()} · {dataset.sampling_regime} — {dataset.case_count} unique cases"),
        columns=1,
        shade_constant=True,
        row_local=True,
    )


def dataset_field_summaries(
    dataset: diagnostics.DatasetDiagnostics,
) -> widgets.VBox:
    """Render independently colored field summaries for all dataset cases."""
    table = diagnostics.dataset_field_summary_table(dataset)
    return tables.grouped_styled_dataframes(
        diagnostics.grouped_table_sections(table),
        title=(f"Dataset field summaries — {dataset.case_count} unique cases"),
        columns=1,
        shade_constant=True,
        row_local=True,
    )


@plot_layout.eda_typography()
def outcome_parameter_distribution(values: Mapping[str, np.ndarray], *, name: str, unit: str) -> Figure:
    """Compare full observed scalar support without implying a causal effect."""
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    nonempty = [array for array in values.values() if array.size]
    if not nonempty:
        axis.text(0.5, 0.5, "No admitted numeric values for this parameter and unit.", ha="center", transform=axis.transAxes)
        return figure
    combined = np.concatenate(nonempty)
    bins = np.histogram_bin_edges(combined, bins="auto")
    colors = {"successful": "tab:blue", "failed": "tab:red", "pending": "0.5"}
    for outcome, array in values.items():
        if not array.size:
            continue
        label = f"{outcome} (n={array.size})"
        artists = histograms.plot_histogram(axis, array, bins=bins, color=colors[outcome], alpha=0.45, label=label)
        if artists.constant_line is not None:
            artists.constant_line.set_label(label)
    axis.set_xlabel(f"{name} [{unit}]")
    axis.set_ylabel("Candidate count")
    axis.legend()
    axis.set_title("Recorded outcome association; not causal evidence")
    return figure


@plot_layout.eda_typography()
def failures_by_parameter(table: pd.DataFrame, parameters: Sequence[str]) -> Figure:
    """Place every terminal candidate at its sampled value, one panel per parameter and one row per material.

    Successful candidates are grey on the material's line; failed ones sit just above it, coloured by their typed reason.
    """
    terminal = table.loc[table["outcome"].isin(("successful", "failed"))] if "outcome" in table else table.iloc[0:0]
    numeric = [name for name in parameters if name in terminal and pd.to_numeric(terminal[name], errors="coerce").notna().any()]
    if terminal.empty or not numeric:
        figure, axis = plt.subplots(figsize=(8.0, 2.0))
        axis.text(0.5, 0.5, "No terminal candidates with numeric sampled parameters.", ha="center", transform=axis.transAxes)
        axis.set_axis_off()
        return figure
    labels = terminal["reason_code"].fillna("unrecorded").where(terminal["outcome"] == "failed", "successful").astype(str)
    reasons = sorted(set(labels) - {"successful"})
    colors = {"successful": "0.6", **{reason: plt.get_cmap("tab10")(index % 10) for index, reason in enumerate(reasons)}}
    materials = sorted(terminal["material_family"].astype(str).unique())
    rows = terminal["material_family"].astype(str).map({material: index for index, material in enumerate(materials)})
    columns = min(_FAILURE_PANEL_COLUMNS, len(numeric))
    panel_rows = math.ceil(len(numeric) / columns)
    figure, axes = plt.subplots(panel_rows, columns, figsize=(4.5 * columns, (1.0 + 0.5 * len(materials)) * panel_rows), squeeze=False)
    for axis, name in zip(axes.flat, numeric, strict=False):
        values = pd.to_numeric(terminal[name], errors="coerce")
        for label in ("successful", *reasons):
            chosen = labels == label
            offset = 0.0 if label == "successful" else 0.25
            axis.scatter(values[chosen], rows[chosen] + offset, s=16, color=colors[label], label=f"{label} (n={int(chosen.sum())})")
        axis.set_yticks(range(len(materials)), materials)
        axis.set_title(name)
    for axis in axes.flat[len(numeric) :]:
        axis.set_axis_off()
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=min(len(handles), 4))
    figure.suptitle("Failures by parameter: recorded association, not causal evidence")
    return figure
