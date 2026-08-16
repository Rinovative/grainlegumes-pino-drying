"""
===============================================================================
generation_input_plot_overview.py
===============================================================================
Present generation-input comparison and dataset overview tables.
Responsibilities:
  - Render compact A/B dataset, case, identity, and provenance context
  - Render parameter and field summaries as A, Mean A, B, and Mean B
  - Show every unique dataset case with raw parameter values
  - Show compact per-case field summaries without standardization
Design principles:
  - Units appear once inside row or column labels
  - Four-value comparison tables normalize colors independently by row
  - Dataset overview tables place parameters by row and actual cases by column
This module does NOT:
  - Render maps, standardize values, or load input files
  - Expose input-generation IDs or hashes in normal selectors
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.ui import tables

if TYPE_CHECKING:
    import ipywidgets as widgets


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
