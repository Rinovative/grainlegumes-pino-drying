"""Present the paired template comparison of two campaigns.

Responsibilities:
  - Draw one parity panel per quantity: Campaign A on x, Campaign B on y, one point per case, coloured by material, on its identity line
  - Count what the pairing left out in a compact figure note, and name the campaigns by their captions in the table titles
  - Render the paired success statistics and the paired quantity table, beside a compact status instead of the figure when
    no pair has a quantity's values in both campaigns

Design principles:
  - Both axes of a panel share one range, so the identity line is its diagonal
  - An empty panel or the status says why instead of showing an unreadable empty frame: no pair successful in both
    campaigns, or none of those with the quantity in both; a missing value is never reported as a failed pairing
  - The table titles name the campaigns by the captions the campaign choice shows, where they wrap, never in a figure note
    as wide as the figure; the exact (task, Source name) campaigns only pair

This module does NOT:
  - Pair candidates or compute statistics
  - Choose campaigns or quantities
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Final

import ipywidgets as widgets

from src.analysis.generation_inputs import generation_input_template_comparison as template
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import tables

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

_PANEL_IN: Final = (4.0, 3.6)
_RANGE_MARGIN: Final = 0.05  # of the data range on each side


def _pairing_note(comparison: template.TemplateComparison) -> str:
    """Return the compact counts of the decided pairs and of what the pairing left out."""
    return (
        f"{comparison.success.pairs} decided pairs; {comparison.unmatched} designs in one campaign only; "
        f"{comparison.unverified} unverified; {comparison.pending} pending"
    )


def _no_pair_text(comparison: template.TemplateComparison, quantity: str) -> str:
    """Return why no pair shows ``quantity``: none is successful in both campaigns, or none of those has its value in both."""
    both = comparison.success.both
    return f"Pairs successful in both campaigns: {both},\nnone with {quantity} in both" if both else "No pair successful in both campaigns"


def parity_figure(comparison: template.TemplateComparison) -> Figure:
    """Draw one parity panel per quantity of ``comparison`` with a legend of the materials and the identity line."""
    figure, axes = figure_layout.panel_figure(1, len(comparison.quantities), panel_in=_PANEL_IN)
    for axis, (quantity, label) in zip(axes.flat, comparison.quantities.items(), strict=True):
        pairs = template.quantity_pairs(comparison, quantity)
        for material, group in pairs.groupby("material_family"):
            name = display_labels.material_display_label(str(material))
            axis.scatter(group["a"], group["b"], s=18, color=visual_semantics.material_color(str(material)), label=name)
        if pairs.empty:
            axis.text(0.5, 0.5, _no_pair_text(comparison, label), ha="center", transform=axis.transAxes)
        else:
            low, high = float(pairs[["a", "b"]].min().min()), float(pairs[["a", "b"]].max().max())
            margin = _RANGE_MARGIN * (high - low) or 1.0
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(low - margin, high + margin)
            axis.axline((low, low), slope=1.0, color="0.4", linewidth=1, label="A = B", gid=figure_layout.REFERENCE_GID)
        figure_layout.set_panel_title(axis, label)
        axis.set_xlabel("Campaign A")
    handles, labels = figure_layout.legend_entries(axes.flat)
    figure_layout.label_rows_once(axes, "Campaign B")
    figure_layout.set_figure_title(figure, axes, "Paired template comparison", note=_pairing_note(comparison))
    if handles:
        figure_layout.add_legend(figure, axes, handles, labels)
    return figure


def template_comparison(
    comparison: template.TemplateComparison, captions: Mapping[template.Campaign, str]
) -> tuple[Figure | widgets.HTML, widgets.VBox, widgets.VBox]:
    """
    Return the parity figure, the paired success table and the paired quantity table, both titled with the campaign captions.

    ``captions`` maps each campaign to the caption the campaign choice shows (``generation_input_labels``), so a title names
    the pair as it was chosen. Without a pair that has some quantity's values in both campaigns the figure would hold only
    empty frames, so a status takes its place: why no pair is drawn (``_no_pair_text``), then the pairing counts.
    """
    names = f"A: {captions[comparison.campaign_a]}; B: {captions[comparison.campaign_b]}"
    drawable = any(not template.quantity_pairs(comparison, quantity).empty for quantity in comparison.quantities)
    status = f"{_no_pair_text(comparison, ' or '.join(comparison.quantities.values()))}. {_pairing_note(comparison)}."
    return (
        parity_figure(comparison) if drawable else widgets.HTML(f"<p>{escape(status)}</p>"),
        tables.styled_dataframe(template.success_table(comparison), title=f"Success of both campaigns, paired — {names}", shade_constant=False),
        tables.styled_dataframe(template.quantity_table(comparison), title=f"Paired quantities — {names}", shade_constant=False),
    )
