"""Paired template comparison of two campaigns: pairing, success statistics, paired quantities and the view.

Responsibilities:
  - Check that only primary candidates of the same design pair, that unmatched designs, undecided and unverified pairs are counted, and
    that a shared key bound to other sampled values or another case id is a conflict
  - Check the Newcombe hybrid-score interval and the exact McNemar test against hand-computed values and their degenerate cases
  - Check that quantities use pairs both campaigns completed successfully, and the parity figure, controls and view gating
  - Check that zero pairs give a status instead of empty frames, that a pair successful in both campaigns but without both
    values is named as such, never as a failed pairing, that the table titles name the campaigns by the captions the
    campaign choice shows, never by their Source names, and that Campaign A opens on the task of Dataset A

Design principles:
  - Pairs are tiny synthetic candidate rows; expected values are hand-computed, the Newcombe ones with exact decimal arithmetic

This module does NOT:
  - Read campaigns from storage or check the other generation-input views
"""

from __future__ import annotations

import math
from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.text import Text

from src.analysis.generation_inputs import generation_input_controls as controls
from src.analysis.generation_inputs import generation_input_labels as labels
from src.analysis.generation_inputs import generation_input_panel as panel
from src.analysis.generation_inputs import generation_input_presentation as presentation
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs import generation_input_template_comparison as template
from src.analysis.generation_inputs.plots import generation_input_plot_template as plot_template
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_panel as ui_panel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_TASK = "transient_drying"
_A = (_TASK, "campaign_a")
_B = (_TASK, "campaign_b")
_QUANTITIES = {"duration_seconds": "Duration [s]"}


def _row(
    campaign: tuple[str, str], material: str, index: int, outcome: str, duration: float | None = None, *, origin: str = "primary", **extra: Any
) -> dict[str, Any]:
    """
    Return one candidate row of ``campaign``; a primary's design key follows its index, a replacement's is its own.

    The sampled values follow material and index, so campaigns agree on a shared design; B lists them in reverse order,
    which must not matter.
    """
    sampled = {"kappa": index + (0.5 if material == "lentil" else 0.25), "zeta": 0.1 * index}
    return {
        "task": campaign[0],
        "source_name": campaign[1],
        "material_family": material,
        "design_key": f"sobol_reference:{index}" if origin == "primary" else f"candidate_{index}",
        "case_id": f"case_{index:04d}",
        "origin": origin,
        "outcome": outcome,
        "duration_seconds": duration,
        "parameters": dict(reversed(sampled.items())) if campaign == _B else sampled,
        "parameter_units": {"kappa": "m^2", "zeta": "1"},
        **extra,
    }


def _selection_state(task: str) -> Any:
    """Return a stand-in selection state whose Dataset A belongs to ``task``; the template view reads only that."""
    return SimpleNamespace(selection=SimpleNamespace(dataset_a_key=(task, "batch")))


def _widgets(roots: Iterable[widgets.Widget]) -> Iterator[widgets.Widget]:
    """Yield ``roots`` and every widget nested in them."""
    for root in roots:
        yield root
        yield from _widgets(getattr(root, "children", ()))


def _html_texts(root: widgets.Widget) -> list[str]:
    """Return the values of every HTML widget in ``root``."""
    return [widget.value for widget in _widgets((root,)) if isinstance(widget, widgets.HTML)]


def _paired_rows() -> list[dict[str, Any]]:
    """Return two campaigns whose designs pair as listed; ``(outcome, duration)`` of A, then of B."""
    designs = {
        ("lentil", 1): (("successful", 10.0), ("successful", 12.0)),
        ("lentil", 2): (("successful", 20.0), ("successful", 18.0)),
        ("lentil", 3): (("successful", 5.0), ("failed", None)),
        ("lentil", 4): (("failed", None), ("successful", 30.0)),
        ("lentil", 5): (("failed", None), ("successful", 6.0)),
        ("lentil", 6): (("pending", None), ("successful", 7.0)),
        ("chickpea", 1): (("successful", 40.0), ("successful", 44.0)),
        ("chickpea", 2): (("successful", 8.0), ("successful", None)),
    }
    rows = [
        _row(campaign, material, index, *outcome)
        for (material, index), outcomes in designs.items()
        for campaign, outcome in zip((_A, _B), outcomes, strict=True)
    ]
    rows += [_row(_A, "lentil", 7, "successful", 9.0), _row(_B, "chickpea", 3, "successful", 9.0)]  # designs of one campaign only
    rows += [_row(campaign, "lentil", 8, "successful", 9.0, origin="replacement") for campaign in (_A, _B)]  # same key in both: never paired
    rows += [_row(("steady_flow", "campaign_a"), "lentil", 1, "successful", 99.0)]  # another task with the name of A
    return rows


# Hand values of the paired durations (10, 12), (20, 18), (40, 44), whose differences B - A are (2, -2, 4): medians 20 and 18, median 2,
# P10 = -2 + 0.2 * 4 = -1.2 and P90 = 2 + 0.8 * 2 = 3.6 by linear interpolation. The pair (8, none) has no B value.
_DURATION_TABLE = {"n paired": 3, "median A": 20.0, "median B": 18.0, "median B - A": 2.0, "P10 B - A": -1.2, "P90 B - A": 3.6}


def test_only_primary_designs_present_in_both_campaigns_pair_and_the_rest_is_counted() -> None:
    """A replacement never pairs, a design in one campaign is dropped and counted, and a pair with a pending candidate is not decided.

    Equal sampled values pair whatever their order.
    """
    comparison = template.compare_templates(_paired_rows(), _A, _B, quantities=_QUANTITIES)

    assert (comparison.unmatched, comparison.pending, comparison.unverified) == (2, 1, 0)
    assert len(comparison.pairs) == 7
    assert not set(comparison.pairs["design_key"]) & {"candidate_8", "sobol_reference:7"}
    success = comparison.success
    assert (success.both, success.only_a, success.only_b, success.neither) == (4, 1, 2, 0)
    assert success.rate_a == pytest.approx(5 / 7)
    assert success.rate_b == pytest.approx(6 / 7)
    assert success.difference == pytest.approx(1 / 7)  # B - A
    assert success.p_value == 1.0  # one discordant pair for A, two for B


def test_quantities_use_only_pairs_both_campaigns_completed_successfully_with_a_value_in_both() -> None:
    """The failed, pending and valueless pairs add no point: the table and the figure keep the three pairs with both durations."""
    comparison = template.compare_templates(_paired_rows(), _A, _B, quantities=_QUANTITIES)
    table = template.quantity_table(comparison).iloc[0]
    for column, value in _DURATION_TABLE.items():
        assert table[column] == pytest.approx(value), column
    pairs = template.quantity_pairs(comparison, "duration_seconds")
    assert sorted(zip(pairs["a"], pairs["b"], strict=True)) == [(10.0, 12.0), (20.0, 18.0), (40.0, 44.0)]


def _replace_in(rows: list[dict[str, Any]], campaign: tuple[str, str], material: str, index: int, **changes: Any) -> list[dict[str, Any]]:
    """Return ``rows`` with the primary of ``campaign`` at ``material`` and ``index`` changed."""
    key = f"sobol_reference:{index}"
    return [
        {**row, **changes} if (row["task"], row["source_name"], row["material_family"], row["design_key"]) == (*campaign, material, key) else row
        for row in rows
    ]


def test_a_shared_design_key_with_different_sampled_values_is_a_conflict_that_refuses_the_comparison_and_the_offer() -> None:
    """Campaigns of other seeds share keys but not inputs: the comparison names the count and the first key, and no partner is offered."""
    rows = _replace_in(_paired_rows(), _B, "lentil", 2, parameters={"kappa": 99.0, "zeta": 0.2})
    rows = _replace_in(rows, _B, "chickpea", 1, parameter_units={"kappa": "kPa", "zeta": "1"})  # another unit is another value

    with pytest.raises(ValueError, match=r"different sampled values: 2; first: (lentil/sobol_reference:2|chickpea/sobol_reference:1)"):
        template.compare_templates(rows, _A, _B, quantities=_QUANTITIES)
    assert template.comparable_campaigns(rows) == {}


def test_the_same_design_under_another_case_id_is_an_identity_conflict_not_two_one_sided_designs() -> None:
    """A design key bound to different case ids in the two campaigns refuses the comparison and the offer."""
    rows = _replace_in(_paired_rows(), _B, "lentil", 1, case_id="case_0099")

    with pytest.raises(ValueError, match=r"different case ids: 1; first: lentil/sobol_reference:1"):
        template.compare_templates(rows, _A, _B, quantities=_QUANTITIES)
    assert template.comparable_campaigns(rows) == {}


def test_pairs_without_sampled_values_are_unverified_left_out_and_named_in_the_pairing_note() -> None:
    """A candidate that carries no sampled values cannot show it is the same design: its pair is counted and excluded, no conflict."""
    rows = _replace_in(_paired_rows(), _A, "lentil", 1, parameters=None)
    rows = _replace_in(rows, _B, "chickpea", 1, parameter_units=None)
    comparison = template.compare_templates(rows, _A, _B, quantities=_QUANTITIES)

    assert (comparison.unverified, comparison.pending, comparison.unmatched) == (2, 1, 2)
    assert len(comparison.pairs) == 5
    assert comparison.success.pairs == 5
    assert "2 unverified" in plot_template._pairing_note(comparison)
    assert len(template.quantity_pairs(comparison, "duration_seconds")) == 1  # only lentil 2 remains with both durations
    assert template.comparable_campaigns(rows)[_A] == (_B,)  # the other shared designs are verified


def test_campaigns_that_share_keys_but_no_verified_design_are_not_offered() -> None:
    """Without one shared design whose candidates carry equal sampled values on both sides, nothing shows the keys name one design."""
    rows = [{**row, "parameters": None, "parameter_units": None} for row in _share_rows()]

    assert template.comparable_campaigns(rows) == {}
    assert template.compare_templates(rows, (_TASK, "a"), (_TASK, "b"), quantities=_QUANTITIES).success.pairs == 0


# The expected intervals are the exact decimal evaluation of Newcombe's formula for (both, only A, only B, neither); the first proportion
# of his difference is B's. Table (9, 1, 3, 7): n = 20, p_A = 0.5, p_B = 0.6, ad - bc = 60 > n / 2, so C = 50 and phi = 50 / sqrt(9600).
# Table (10, 2, 6, 2): ad - bc = 8 <= n / 2, so C = 0 and phi = 0. Table (2, 4, 3, 1): ad - bc = -10 < 0, so C = -10 and phi = -10 / sqrt(600).
@pytest.mark.parametrize(
    ("counts", "difference", "low", "high"),
    [
        ((9, 1, 3, 7), 0.1, -0.10521201037703574, 0.28972765036620984),
        ((10, 2, 6, 2), 0.2, -0.08194787416730849, 0.4445200386835596),
        ((2, 4, 3, 1), -0.1, -0.5159100995993697, 0.36231488895171493),
    ],
)
def test_newcombe_interval_matches_the_hand_computed_value_in_each_branch_of_the_correlation_correction(
    counts: tuple[int, int, int, int], difference: float, low: float, high: float
) -> None:
    """Method 10 of Newcombe (1998) reproduces the hand values for a positive, a corrected-to-zero and a negative correlation."""
    result = template.compare_success(*counts)

    assert result.difference == pytest.approx(difference)
    assert (result.low, result.high) == pytest.approx((low, high), abs=1e-9)


def test_exact_mcnemar_p_is_the_doubled_binomial_tail_of_the_discordant_pairs_capped_at_one() -> None:
    """One against seven discordant pairs: 2 * (1 + 8) / 2^8 = 0.0703125; a tie and no discordant pair cap at one."""
    assert template.mcnemar_exact_p(1, 7) == pytest.approx(18 / 256)
    assert template.mcnemar_exact_p(7, 1) == pytest.approx(18 / 256)
    assert template.mcnemar_exact_p(3, 3) == 1.0
    assert template.mcnemar_exact_p(0, 0) == 1.0


def test_exact_mcnemar_p_of_thousands_of_discordant_pairs_equals_the_exact_integer_tail() -> None:
    """3900 against 4100 discordant pairs: the p value agrees with the exact rational tail to 1e-12."""
    only_a, only_b = 3900, 4100
    discordant = only_a + only_b
    tail, coefficient = 0, 1
    for k in range(only_a + 1):  # exact C(m, k) by the integer recurrence
        coefficient = coefficient * (discordant - k + 1) // k if k else 1
        tail += coefficient

    assert template.mcnemar_exact_p(only_a, only_b) == pytest.approx(float(Fraction(2 * tail, 2**discordant)), rel=1e-12)


def test_zero_pairs_leave_every_statistic_undefined_and_the_quantity_table_empty() -> None:
    """
    Campaigns without a common design give no decided pair: statistics are ``None``, the table has zero pairs and NaN medians.

    Both table titles name the campaigns by their captions, which the sampling regimes tell apart, never by their Source names.
    """
    rows = [_row(_A, "lentil", 1, "successful", 1.0, sampling_regime="train"), _row(_B, "lentil", 2, "successful", 2.0, sampling_regime="test")]
    comparison = template.compare_templates(rows, _A, _B, quantities=_QUANTITIES)

    success = comparison.success
    assert (success.pairs, comparison.unmatched) == (0, 2)
    assert (success.rate_a, success.rate_b, success.difference, success.low, success.high, success.p_value) == (None,) * 6
    table = template.quantity_table(comparison).iloc[0]
    assert table["n paired"] == 0
    assert math.isnan(table["median B - A"])
    figure = plot_template.parity_figure(comparison)
    try:
        assert len(figure.axes) == 1
    finally:
        plt.close(figure)
    captions = labels.campaign_display_labels(rows, (_A, _B))
    status, success_table, quantity_table = plot_template.template_comparison(comparison, captions)  # no figure of empty frames
    assert isinstance(status, widgets.HTML)
    assert "No pair successful in both campaigns" in status.value
    for table in (success_table, quantity_table):
        titles = " ".join(_html_texts(table))
        assert f"A: {captions[_A]}; B: {captions[_B]}" in titles
        assert "campaign_" not in titles


def test_a_pair_successful_in_both_campaigns_without_both_durations_is_not_reported_as_unsuccessful() -> None:
    """The status and the empty panel count the one pair successful in both campaigns and name the duration it lacks."""
    rows = [_row(campaign, "lentil", 1, "successful") for campaign in (_A, _B)]
    comparison = template.compare_templates(rows, _A, _B, quantities=_QUANTITIES)
    assert comparison.success.both == 1
    status = plot_template.template_comparison(comparison, labels.campaign_display_labels(rows, (_A, _B)))[0]
    figure = plot_template.parity_figure(comparison)
    try:
        assert isinstance(status, widgets.HTML)
        for text in (status.value, *(text.get_text() for text in figure.axes[0].texts)):
            assert "No pair successful" not in text
            assert "successful in both campaigns: 1" in text
            assert "Duration [s]" in text
    finally:
        plt.close(figure)


def test_without_discordant_pairs_the_difference_is_zero_p_is_one_and_the_interval_stays_open() -> None:
    """Three pairs successful in both and two failed in both: difference 0, p 1, the symmetric interval of the formula, not a point."""
    result = template.compare_success(3, 0, 0, 2)

    assert (result.difference, result.p_value) == (0.0, 1.0)
    assert (result.low, result.high) == pytest.approx((-0.30732310919421185, 0.30732310919421185), abs=1e-9)  # phi = 3.5 / 6


def test_all_pairs_successful_in_both_campaigns_give_zero_difference_and_the_interval_of_the_wilson_bound() -> None:
    """Four of four in both: phi = 0 because a margin is empty, so the interval is plus and minus one minus the Wilson lower bound of 4/4."""
    result = template.compare_success(4, 0, 0, 0)

    assert (result.rate_a, result.rate_b, result.difference, result.p_value) == (1.0, 1.0, 0.0, 1.0)
    assert (result.low, result.high) == pytest.approx((-0.4898908364545973, 0.4898908364545973), abs=1e-9)  # 1 - 4 / (4 + z^2)


def _share_rows() -> list[dict[str, Any]]:
    """Return campaigns a, b, c of one task where a shares with b and b with c, and one campaign of another task."""
    return [
        _row((_TASK, "a"), "lentil", 1, "successful"),
        _row((_TASK, "a"), "lentil", 2, "successful"),
        _row((_TASK, "b"), "lentil", 2, "successful"),
        _row((_TASK, "b"), "lentil", 3, "successful"),
        _row((_TASK, "c"), "lentil", 3, "successful"),
        _row((_TASK, "c"), "lentil", 9, "successful", origin="replacement"),
        _row(("steady_flow", "b"), "lentil", 2, "successful"),
    ]


def test_campaigns_are_comparable_when_two_of_one_task_share_a_primary_design() -> None:
    """The partners are symmetric, stay within a task, and a shared replacement does not count."""
    partners = template.comparable_campaigns(_share_rows())

    assert partners == {(_TASK, "a"): ((_TASK, "b"),), (_TASK, "b"): ((_TASK, "a"), (_TASK, "c")), (_TASK, "c"): ((_TASK, "b"),)}
    assert template.comparable_campaigns(_share_rows()[:2]) == {}
    assert template.comparable_campaigns([_row(_A, "lentil", 1, "successful"), _row(("steady_flow", "x"), "lentil", 1, "successful")]) == {}
    assert template.comparable_campaigns([_row(campaign, "lentil", 4, "successful", origin="replacement") for campaign in (_A, _B)]) == {}


def test_the_view_is_offered_only_when_two_campaigns_of_one_task_share_a_design() -> None:
    """One campaign hides the entry in the registry and the factories; two campaigns that share a design show it in both."""
    key = "template_comparison"

    def offered(rows: Iterable[dict[str, Any]]) -> tuple[bool, bool]:
        stub = SimpleNamespace(outcomes=sources.CampaignOutcomeCatalog(tuple(rows)))  # the factories only bind the catalog
        listed = key in panel._view_factories(cast("Any", stub), cast("Any", None))
        names = {plot.key for section in presentation.sections_for_profiles((_TASK,), template_comparison=listed) for plot in section.plots}
        return listed, key in names

    assert offered(_share_rows()[:2]) == (False, False)
    assert offered(_share_rows()) == (True, True)
    assert key not in {plot.key for section in presentation.sections_for_profiles((_TASK,)) for plot in section.plots}


def test_campaign_b_follows_campaign_a_and_a_change_reports_once() -> None:
    """Campaign B offers the partners of Campaign A; the selection yields the comparison of the chosen pair."""
    local = controls.TemplateComparisonControls(sources.CampaignOutcomeCatalog(tuple(_share_rows())), template.comparable_campaigns(_share_rows()))
    calls: list[int] = []
    local.set_callback(lambda: calls.append(1))
    assert [value for _label, value in local.campaign_b.options] == [(_TASK, "b")]

    local.campaign_a.value = (_TASK, "b")
    assert [value for _label, value in local.campaign_b.options] == [(_TASK, "a"), (_TASK, "c")]
    assert len(calls) == 1
    local.campaign_b.value = (_TASK, "c")
    assert len(calls) == 2
    assert (local.comparison().campaign_a, local.comparison().campaign_b) == ((_TASK, "b"), (_TASK, "c"))


def test_campaign_a_opens_on_a_campaign_of_the_task_of_dataset_a_when_that_task_has_a_pair() -> None:
    """Pairs in two tasks, the transient one listed first: a steady Dataset A opens on the steady pair, an unpaired task on the first."""
    rows = [_row(campaign, "lentil", 1, "successful") for campaign in ((_TASK, "a"), (_TASK, "b"), ("steady_flow", "x"), ("steady_flow", "y"))]
    partners = template.comparable_campaigns(rows)
    assert next(iter(partners)) == (_TASK, "a")
    for task, expected in (("steady_flow", ("steady_flow", "x")), (_TASK, (_TASK, "a")), ("other_profile", (_TASK, "a"))):
        view = panel._template_factory(sources.CampaignOutcomeCatalog(tuple(rows)), partners, _selection_state(task))()
        try:
            (campaign_a,) = (widget for widget in _widgets(view.rows) if isinstance(widget, widgets.Dropdown) and widget.description == "Campaign A")
            assert campaign_a.value == expected
        finally:
            plt.close("all")


def test_the_parity_figure_has_one_panel_per_quantity_with_one_point_per_pair_on_a_shared_diagonal() -> None:
    """Each panel draws A on x and B on y per material with the fixed material colour, the identity line and equal limits."""
    rows = [{**row, "wall_seconds": None if row["duration_seconds"] is None else 2.0 * row["duration_seconds"]} for row in _paired_rows()]
    comparison = template.compare_templates(rows, _A, _B, quantities={**_QUANTITIES, "wall_seconds": "Wall time [s]"})
    figure = plot_template.parity_figure(comparison)
    try:
        panels = [axis for axis in figure.axes if axis.get_visible()]
        assert len(panels) == 2
        for axis, scale in zip(panels, (1.0, 2.0), strict=True):
            points = sorted(tuple(map(float, point)) for collection in axis.collections for point in np.asarray(collection.get_offsets()))
            assert points == [(scale * a, scale * b) for a, b in [(10.0, 12.0), (20.0, 18.0), (40.0, 44.0)]]
            assert axis.get_xlim() == axis.get_ylim()
            assert len(axis.lines) == 1  # the identity line
        colours = {collection.get_label(): collection.get_facecolor()[0] for collection in panels[0].collections}
        assert colours["Lentil"] == pytest.approx(to_rgba(visual_semantics.material_color("lentil")))
        assert {text.get_text() for text in figure.legends[0].get_texts()} == {"Chickpea", "Lentil", "A = B"}
        assert not any("campaign_" in text.get_text() for text in figure.findobj(Text))  # the names stand in the table titles
    finally:
        plt.close(figure)


def test_the_view_publishes_the_figure_with_the_success_and_quantity_tables() -> None:
    """The cached view renders the selected pair into one figure and two tables titled with the captions the campaign choice shows."""
    rows = [{**row, "sampling_regime": "train" if row["source_name"] == "campaign_a" else "test"} for row in _paired_rows()]
    factory = panel._template_factory(sources.CampaignOutcomeCatalog(tuple(rows)), template.comparable_campaigns(rows), _selection_state(_TASK))
    export: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "template", "title": "Template"}
    view = factory(export_state=export, export_plot_name="template", export_title="Template")
    try:
        assert isinstance(view, ui_panel.ViewPanel)
        assert view.slot is not None
        assert not view.failure.shown
        assert isinstance(export["fig"], Figure)
        chosen = {widget.description: widget.label for widget in _widgets(view.rows) if isinstance(widget, widgets.Dropdown)}
        _figure, *published = view.slot.widget.children
        assert len(published) == 2
        for table in published:
            titles = " ".join(_html_texts(table))
            assert f"A: {chosen['Campaign A']}; B: {chosen['Campaign B']}" in titles
            assert "campaign_" not in titles
    finally:
        plt.close("all")
