"""Present the paired template comparison of two campaigns.

Responsibilities:
  - Draw one parity panel per quantity: Campaign A on x, Campaign B on y, one point per case, coloured by material, on its identity line
  - Name the campaigns and what the pairing left out in the figure note
  - Render the paired success statistics and the paired quantity table

Design principles:
  - Both axes of a panel share one range, so the identity line is its diagonal
  - A panel without a pair successful in both campaigns says so instead of showing an unreadable empty frame

This module does NOT:
  - Pair candidates or compute statistics
  - Choose campaigns or quantities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.analysis.generation_inputs import generation_input_template_comparison as template
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import tables

if TYPE_CHECKING:
    import ipywidgets as widgets
    from matplotlib.figure import Figure

_PANEL_IN: Final = (4.0, 3.6)
_RANGE_MARGIN: Final = 0.05  # of the data range on each side


def parity_figure(comparison: template.TemplateComparison) -> Figure:
    """Draw one parity panel per quantity of ``comparison`` with a legend of the materials and the identity line."""
    figure, axes = figure_layout.panel_figure(1, len(comparison.quantities), panel_in=_PANEL_IN)
    for axis, (quantity, label) in zip(axes.flat, comparison.quantities.items(), strict=True):
        pairs = template.quantity_pairs(comparison, quantity)
        for material, group in pairs.groupby("material_family"):
            axis.scatter(group["a"], group["b"], s=18, color=visual_semantics.material_color(str(material)), label=str(material))
        if pairs.empty:
            axis.text(0.5, 0.5, "No pair successful in both campaigns", ha="center", transform=axis.transAxes)
        else:
            low, high = float(pairs[["a", "b"]].min().min()), float(pairs[["a", "b"]].max().max())
            margin = _RANGE_MARGIN * (high - low) or 1.0
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(low - margin, high + margin)
            axis.axline((low, low), slope=1.0, color="0.4", linewidth=1, label="A = B")
        figure_layout.set_panel_title(axis, label)
        axis.set_xlabel("Campaign A")
    handles, labels = figure_layout.legend_entries(axes.flat)
    figure_layout.label_rows_once(axes, "Campaign B")
    figure_layout.set_figure_title(figure, axes, "Paired template comparison", note=template.context_line(comparison))
    if handles:
        figure_layout.add_legend(figure, axes, handles, labels)
    return figure


def template_comparison(comparison: template.TemplateComparison) -> tuple[Figure, widgets.VBox, widgets.VBox]:
    """Return the parity figure, the paired success table and the paired quantity table."""
    return (
        parity_figure(comparison),
        tables.styled_dataframe(template.success_table(comparison), title="Success of both campaigns, paired", shade_constant=False),
        tables.styled_dataframe(template.quantity_table(comparison), title="Paired quantities", shade_constant=False),
    )
