"""Present generation-input comparison and dataset overview tables.

Responsibilities:
  - Render compact A/B dataset, case, identity, and provenance context
  - Render parameter and field summaries as A, Mean A, B, and Mean B
  - Show every unique dataset case with raw parameter values
  - Show compact per-case field summaries without standardization
  - Group campaign parameters by table section and plot the chosen ones as grids of equal small panels: outcome histograms
    and failures by material coloured by the typed failure reason

Design principles:
  - Units appear once inside row or column labels
  - Four-value comparison tables normalize colors independently by row
  - Dataset overview tables place parameters by row and actual cases by column
  - Legend counts are those of the whole candidate scope; a panel that draws fewer candidates names its own n in its title

This module does NOT:
  - Render maps, standardize values, or load input files
  - Expose input-generation IDs or hashes in normal selectors
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs import generation_input_table_schema as table_schema
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import tables

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_PANEL_COLUMNS: Final = 3
_PANEL_IN: Final = (3.6, 2.4)  # titles are parameter names of up to about 40 characters
_EVIDENCE_NOTE: Final = "Recorded association; not causal evidence"
_OUTCOME_COLORS: Final = {"successful": "tab:blue", "failed": "tab:red", "pending": "0.5"}


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


def _message_figure(message: str) -> Figure:
    """Return a one-panel figure that states why nothing is drawn."""
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(5.0, 1.5))
    axes[0, 0].text(0.5, 0.5, message, ha="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_axis_off()
    return figure


def _parameter_title(axis: Axes, name: str, drawn: int, total: int) -> None:
    """
    Title the panel by its parameter and add its own n when it draws fewer candidates than the legend counts.

    The n sits in the title, not inside the axes, so a legend placed inside a single panel can never cover it.
    """
    figure_layout.set_panel_title(axis, f"{name} (n = {drawn} of {total})" if drawn < total else name)


def _parameter_grid(count: int, *, sharey: Literal["none", "all"] = "none") -> tuple[Figure, np.ndarray]:
    """Return a figure of ``count`` equal panels in rows of ``_PANEL_COLUMNS``; unused cells of the last row are hidden."""
    columns = min(_PANEL_COLUMNS, count)
    figure, axes = figure_layout.panel_figure(math.ceil(count / columns), columns, panel_in=_PANEL_IN, sharey=sharey)
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return figure, axes


def outcome_parameter_distributions(values: Mapping[table_schema.Parameter, Mapping[str, np.ndarray]], counts: Mapping[str, int]) -> Figure:
    """
    Compare the observed support of every chosen parameter by outcome, one panel per parameter, ordered by table section.

    ``values`` holds the finite values in the recorded unit per outcome of every parameter, drawn in the display unit the axis
    names; ``counts`` the candidates per outcome of the whole scope. The legend counts are those of the scope, the same for every
    panel; a panel that draws fewer candidates, because some recorded no value of its parameter in its unit, names its own n in its title.
    """
    ordered = [parameter for group in table_schema.grouped_parameters(values).values() for parameter in group]
    drawn = [parameter for parameter in ordered if any(array.size for array in values[parameter].values())]
    if not drawn:
        return _message_figure("No admitted numeric values for the chosen parameters and units.")
    figure, axes = _parameter_grid(len(drawn))
    for axis, parameter in zip(axes.flat, drawn, strict=False):
        by_outcome = {outcome: diagnostics.display_value(parameter[0], array, parameter[1]) for outcome, array in values[parameter].items()}
        bins = np.histogram_bin_edges(np.concatenate([array for array in by_outcome.values() if array.size]), bins="auto")
        for outcome, array in by_outcome.items():
            if not array.size:
                continue
            label = f"{outcome} (n={counts[outcome]})"
            artists = histograms.plot_histogram(axis, array, bins=bins, color=_OUTCOME_COLORS[outcome], alpha=0.45, label=label)
            if artists.constant_line is not None:
                artists.constant_line.set_label(label)
        _parameter_title(axis, parameter[0], sum(array.size for array in by_outcome.values()), sum(counts.values()))
        axis.set_xlabel(f"[{diagnostics.display_unit(*parameter)}]")
    handles, legend_labels = figure_layout.legend_entries(axes.flat)
    figure_layout.label_rows_once(axes, "Candidate count")
    figure_layout.set_figure_title(figure, axes, "Candidate outcomes by parameter", note=_EVIDENCE_NOTE)
    figure_layout.add_legend(figure, axes, handles, legend_labels)
    return figure


def failures_by_parameter(candidates: Iterable[Mapping[str, Any]], parameters: Iterable[table_schema.Parameter]) -> Figure:
    """
    Place every terminal candidate at its sampled value, one panel per chosen parameter, ordered by table section, one row per material.

    Successful candidates are grey on the material's line; failed ones sit just above it, coloured by their typed reason. The
    values are selected by name and unit as in the distributions. The legend counts are those of all terminal candidates; a panel
    that draws fewer, because some recorded no value of its parameter in its unit, names its own n in its title.
    """
    terminal = [row for row in candidates if row["outcome"] in ("successful", "failed")]
    values_by_parameter = {
        parameter: pd.Series(
            diagnostics.display_value(
                parameter[0], np.array([sources.parameter_value(row, *parameter) for row in terminal], dtype=np.float64), parameter[1]
            )
        )
        for group in table_schema.grouped_parameters(parameters).values()
        for parameter in group
    }
    numeric = [parameter for parameter, values in values_by_parameter.items() if values.notna().any()]
    if not numeric:
        return _message_figure("No terminal candidates with numeric sampled parameters.")
    labels = pd.Series(
        ["successful" if row["outcome"] == "successful" else str((row.get("failure") or {}).get("reason_code") or "unrecorded") for row in terminal]
    )
    reasons = sorted(set(labels) - {"successful"})
    colors = {"successful": "0.6", **{reason: plt.get_cmap("tab10")(index % 10) for index, reason in enumerate(reasons)}}
    materials = sorted({str(row["material_family"]) for row in terminal})
    rows = pd.Series([materials.index(str(row["material_family"])) for row in terminal], dtype=np.float64)
    figure, axes = _parameter_grid(len(numeric), sharey="all")
    for axis, parameter in zip(axes.flat, numeric, strict=False):
        values = values_by_parameter[parameter]
        for label in ("successful", *reasons):
            chosen = labels == label
            offset = 0.0 if label == "successful" else 0.25
            axis.scatter(values[chosen], rows[chosen] + offset, s=16, color=colors[label], label=f"{label} (n={int(chosen.sum())})")
        _parameter_title(axis, parameter[0], int(values.notna().sum()), len(terminal))
        axis.set_xlabel(f"[{diagnostics.display_unit(*parameter)}]")
    axes[0, 0].set_yticks(range(len(materials)), materials)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure_layout.set_figure_title(figure, axes, "Terminal candidates by parameter and material", note=_EVIDENCE_NOTE)
    figure_layout.add_legend(figure, axes, handles, legend_labels)
    return figure
