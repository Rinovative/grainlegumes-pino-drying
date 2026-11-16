"""Protect the shared physical-map layout contract.

Responsibilities:
  - Check the grid of time windows: column widths, row heights and the y axis shared along a row
  - Check the fixed-inch map geometry: margins, the width cap or none, and the sizes and places of the bands beside the maps
  - Check the slot colour bar's unit bracket and the labels a field-map scale's kind gives it: boundaries, signed band
    centres, decades, a cyclic scale's wrap ends, a constant's value, a state's power of ten from the size of its values
    and an error's from its step, a map's own arrows, and the thinning of its labels to the real bar height

Design principles:
  - Geometry is read from drawn figures, never from layout fractions

This module does NOT:
  - Check scientific field values, colour-scale boundaries or per-plot composition
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_plot_layout as plot_layout
from src.analysis.ui import analysis_ui_time as time_ui


def _slot_bar(domain: tuple[float, float], boundaries: Any, *, semantic: str = "linear_positive", **options: Any) -> tuple[Any, Any]:
    """Return a drawn one-map figure of ``domain`` and the slot bar of the locked ``semantic`` scale on ``boundaries``."""
    figure, axes, slots = plot_layout.map_grid(plot_layout.plan_geometry(1, 1, domain), extent=(0.0, domain[0], 0.0, domain[1]))
    scale = field_maps.build_field_map(boundaries, semantic=semantic, locked_boundaries=boundaries, **options)
    image = axes[0, 0].pcolormesh(np.zeros((2, 2)), **scale.pcolormesh_kwargs())
    colorbar = plot_layout.add_field_map_colorbar(figure, image, slots[0, 0], scale, label="T [K]")
    figure.canvas.draw()
    return figure, colorbar


def _data_bar(values: Any, *, semantic: str, label: str, locked: Any = None) -> tuple[Any, Any]:
    """Return a drawn one-map figure and the slot bar of the ``semantic`` scale of ``values``, on ``locked`` boundaries if given."""
    figure, axes, slots = plot_layout.map_grid(plot_layout.plan_geometry(1, 1, (1.0, 1.0)), extent=(0.0, 1.0, 0.0, 1.0))
    scale = field_maps.build_field_map(values, semantic=semantic, locked_boundaries=locked)
    image = axes[0, 0].pcolormesh(np.zeros((2, 2)), **scale.pcolormesh_kwargs())
    colorbar = plot_layout.add_field_map_colorbar(figure, image, slots[0, 0], scale, label=label)
    figure.canvas.draw()
    return figure, colorbar


def _signed(*texts: str) -> list[str]:
    """Return ``texts`` with the true minus sign a bar writes."""
    return [text.replace("-", "\N{MINUS SIGN}") for text in texts]


def _texts(colorbar: Any) -> list[str]:
    """Return the drawn tick label texts of ``colorbar``, empty where a label is thinned out."""
    return [label.get_text() for label in colorbar.ax.get_yticklabels()]


def test_fixed_map_geometry_follows_rows_columns_and_domain_and_shrinks_all_maps_under_the_width_cap() -> None:
    """One column keeps the target map width; many columns share the cap; the figure has exactly the planned size."""
    single = plot_layout.plan_geometry(2, 1, (1.2, 0.75))
    assert single.map_width == plot_layout.MAP_WIDTH_IN
    assert single.map_height == pytest.approx(plot_layout.MAP_WIDTH_IN * 0.75 / 1.2)
    assert single.figure_height == pytest.approx(
        plot_layout.MARGIN_TOP_TITLE_IN + plot_layout.MARGIN_BOTTOM_IN + 2 * single.map_height + plot_layout.GAP_V_IN
    )
    wide = plot_layout.plan_geometry(1, 4, (1.2, 0.75))
    assert wide.map_width < plot_layout.MAP_WIDTH_IN
    assert wide.figure_width == pytest.approx(plot_layout.MAP_MAX_WIDTH_IN)
    for rows, columns, domain in ((0, 1, (1.0, 1.0)), (1, 1, (0.0, 1.0)), (1, 1, (1.0, float("nan")))):
        with pytest.raises(ValueError, match="Map grids need"):
            plot_layout.plan_geometry(rows, columns, domain)
    figure, _grid = plot_layout.map_figure(wide)
    assert figure.get_layout_engine() is None
    np.testing.assert_allclose(figure.get_size_inches(), (wide.figure_width, wide.figure_height))
    plt.close(figure)


def test_without_a_width_cap_every_map_keeps_the_target_width_and_the_figure_grows_with_its_columns() -> None:
    """Six columns shrink under the default cap; without a cap every map keeps its target width and the figure passes the cap."""
    capped = plot_layout.plan_geometry(1, 6, (1.2, 0.75))
    free = plot_layout.plan_geometry(1, 6, (1.2, 0.75), width_cap=None)
    assert capped.map_width < plot_layout.MAP_WIDTH_IN
    assert free.map_width == plot_layout.MAP_WIDTH_IN
    assert free.figure_width == pytest.approx(
        plot_layout.MARGIN_LEFT_ROW_LABEL_IN + 6 * free.cell_width + 5 * plot_layout.GAP_H_IN + plot_layout.MARGIN_RIGHT_IN
    )
    assert free.figure_width > plot_layout.MAP_MAX_WIDTH_IN


def test_bands_above_below_and_right_take_their_size_and_one_gap_and_the_right_band_spans_the_map_rows() -> None:
    """Each band adds its size and one fixed gap; above and below span the map columns, right spans the drawn map rows."""
    domain = (1.2, 0.75)
    plain = plot_layout.plan_geometry(2, 3, domain, row_labels=False, title=False, width_cap=None)
    banded = plot_layout.plan_geometry(2, 3, domain, row_labels=False, title=False, width_cap=None, above_in=1.7, below_in=2.0, right_in=2.4)
    assert (plain.left, plain.top) == (plot_layout.MARGIN_LEFT_IN, plot_layout.MARGIN_TOP_IN)
    assert banded.figure_width - plain.figure_width == pytest.approx(2.4 + plot_layout.GAP_H_IN)
    assert banded.figure_height - plain.figure_height == pytest.approx(1.7 + 2.0 + 2 * plot_layout.BAND_GAP_V_IN)
    figure, axes, _slots = plot_layout.map_grid(banded, extent=(0.0, domain[0], 0.0, domain[1]))
    figure.canvas.draw()
    width, height = figure.get_size_inches()
    boxes = [axis.get_position() for axis in axes.flat]
    left, right = min(box.x0 for box in boxes) * width, max(box.x1 for box in boxes) * width
    bottom, top = min(box.y0 for box in boxes) * height, max(box.y1 for box in boxes) * height
    inches = {where: np.multiply(plot_layout.band_rect(banded, where), (width, height, width, height)) for where in ("above", "below", "right")}
    np.testing.assert_allclose(inches["above"], (left, top + plot_layout.BAND_GAP_V_IN, banded.grid_width, 1.7), atol=1e-6)
    np.testing.assert_allclose(inches["below"], (left, bottom - plot_layout.BAND_GAP_V_IN - 2.0, banded.grid_width, 2.0), atol=1e-6)
    bar_edge = right + plot_layout.COLORBAR_PAD_IN + plot_layout.COLORBAR_WIDTH_IN
    np.testing.assert_allclose(inches["right"], (bar_edge + plot_layout.GAP_H_IN, bottom, 2.4, top - bottom), atol=1e-6)
    plt.close(figure)
    with pytest.raises(ValueError, match="reserves no 'right' band"):
        plot_layout.band_rect(plain, "right")
    with pytest.raises(ValueError, match="finite non-negative"):
        plot_layout.plan_geometry(1, 1, domain, below_in=-1.0)


def test_a_power_of_ten_moves_into_the_unit_bracket_of_a_colorbar_label_and_a_label_without_a_bracket_is_refused() -> None:
    """The factor stands in the unit bracket (a dimensionless unit keeps only the factor); a slot refuses a label without a unit bracket."""
    assert plot_layout.label_with_exponent("u [kg/m³]", -5) == "u [10⁻⁵ kg/m³]"
    assert plot_layout.label_with_exponent("ε [-]", 3) == "ε [10³]"
    assert plot_layout.label_with_exponent("[K]", 0) == "[K]"
    figure, grid = plot_layout.map_figure(plot_layout.plan_geometry(1, 1, (1.0, 1.0)))
    axis = figure.add_subplot(grid[0, 0])
    slot = plot_layout.prepare_map_axes(axis, extent=(0.0, 1.0, 0.0, 1.0), bottom=True, left=True)
    image = axis.pcolormesh(np.zeros((2, 2)), norm=BoundaryNorm(np.linspace(0.0, 1.0, 12), 11))
    with pytest.raises(ValueError, match="must end in a unit bracket"):
        plot_layout.add_slot_colorbar(figure, image, slot, label="kg/m³")
    plt.close(figure)


def test_a_slot_bar_labels_its_boundaries_its_signed_band_centres_or_the_decades_of_a_wide_geometric_scale() -> None:
    """Plain: every boundary at the decimals of the step; signed: the band centres, a fine step in a power of ten; geometric: decades."""
    plain = 0.25 * np.arange(11)
    figure, bar = _slot_bar((1.0, 1.0), plain)
    np.testing.assert_allclose(bar.get_ticks(), plain)
    assert _texts(bar) == [f"{value:.2f}" for value in plain]
    plt.close(figure)
    signed = (np.arange(12) - 5.5) * 2e-5
    figure, bar = _slot_bar((1.0, 1.0), signed, semantic="signed_error")
    np.testing.assert_allclose(bar.get_ticks(), (signed[:-1] + signed[1:]) / 2.0, atol=1e-15)
    assert _texts(bar) == [str(value).replace("-", "\N{MINUS SIGN}") for value in range(-10, 11, 2)]
    assert bar.ax.get_ylabel() == "T [10⁻⁵ K]"
    plt.close(figure)
    geometric = np.geomspace(1e-5, 1.0, 12)
    figure, bar = _slot_bar((1.0, 1.0), geometric, semantic="log_positive")
    np.testing.assert_allclose(bar.get_ticks(), 10.0 ** np.arange(-5, 1))
    np.testing.assert_allclose(bar.get_ticks(minor=True), geometric[1:-1])  # the other boundaries keep a mark without a label
    assert bar.ax.get_ylabel() == "T [K]"
    plt.close(figure)
    narrow = np.geomspace(3.0e-9, 7.06e-9, 12)  # a Kozeny-Carman permeability band: no decades to label
    figure, bar = _slot_bar((1.0, 1.0), narrow, semantic="log_positive")
    texts = [text for text in _texts(bar) if text]
    assert (texts[0], texts[-1]) == ("3.00", "7.06")
    assert {len(text.partition(".")[2]) for text in texts} == {2}
    assert bar.ax.get_ylabel() == "T [10⁻⁹ K]"
    plt.close(figure)


def test_a_tiny_signed_state_such_as_the_shear_permeability_is_labelled_in_the_power_of_ten_of_its_values() -> None:
    """Kxy of ±2e-10 m²: band centres -2.0 to 2.0 with 10⁻¹⁰ in the unit bracket, not «0.000» on every label."""
    figure, bar = _data_bar(np.linspace(-2.0e-10, 2.0e-10, 101), semantic="linear_signed", label="Kxy [m²]")
    assert _texts(bar) == _signed("-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0")
    assert bar.ax.get_ylabel() == "Kxy [10⁻¹⁰ m²]"
    plt.close(figure)


def test_a_state_between_a_thousandth_and_ten_thousand_keeps_its_own_unit() -> None:
    """A temperature from 293.15 to 303.15 K on its state bands reads 293 to 304 in K, without a factor."""
    figure, bar = _data_bar(np.array([293.15, 303.15]), semantic="linear_positive", label="T [K]", locked=field_maps.state_boundaries(293.15, 303.15))
    assert _texts(bar) == [str(value) for value in range(293, 305)]
    assert bar.ax.get_ylabel() == "T [K]"
    plt.close(figure)


def test_a_near_uniform_tiny_state_is_widened_and_labelled_in_its_power_of_ten() -> None:
    """1.2345e-10 to 1.2346e-10 m² is widened to 11 steps of 0.001 in 10⁻¹⁰ m²: twelve distinct labels 1.229 to 1.240."""
    locked = field_maps.state_boundaries(1.2345e-10, 1.2346e-10)
    figure, bar = _data_bar(np.array([1.2345e-10, 1.2346e-10]), semantic="linear_positive", label="K [m²]", locked=locked)
    assert _texts(bar) == [f"{value:.3f}" for value in np.linspace(1.229, 1.240, 12)]
    assert bar.ax.get_ylabel() == "K [10⁻¹⁰ m²]"
    plt.close(figure)


def test_a_bar_whose_bands_round_up_across_a_power_of_ten_is_banded_and_labelled_in_the_larger_one() -> None:
    """
    A shear reach of 9.96e-11 gets 11 bands of 2e-11 up to 1.1e-10: centres -1.0 to 1.0 in 10⁻¹⁰, not -10 to 10 in 10⁻¹¹.

    A near-uniform state just below 10⁻¹⁰ is widened in 10⁻¹⁰ (steps of 0.001 there), so its 3-decimal labels stay distinct.
    """
    figure, bar = _data_bar(np.linspace(-9.96e-11, 9.96e-11, 101), semantic="linear_signed", label="Kxy [m²]")
    assert _texts(bar) == _signed("-1.0", "-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
    assert bar.ax.get_ylabel() == "Kxy [10⁻¹⁰ m²]"
    plt.close(figure)
    locked = field_maps.state_boundaries(9.9960e-11, 9.9965e-11)
    figure, bar = _data_bar(np.array([9.9960e-11, 9.9965e-11]), semantic="linear_positive", label="K [m²]", locked=locked)
    assert _texts(bar) == [f"{value:.3f}" for value in np.linspace(0.994, 1.005, 12)]
    assert bar.ax.get_ylabel() == "K [10⁻¹⁰ m²]"
    plt.close(figure)


def test_a_data_derived_signed_state_with_a_step_finer_than_a_thousandth_is_widened_to_distinct_labels_symmetric_about_zero() -> None:
    """A velocity of reach 0.0018 m/s gets the 11 bands of 0.001 centred on 0: centres -0.005 to 0.005, never twice «0.001»."""
    figure, bar = _data_bar(np.linspace(-0.0018, 0.0018, 101), semantic="linear_signed", label="u [m/s]")
    assert _texts(bar) == _signed(*(f"{value:.3f}" for value in np.linspace(-0.005, 0.005, 11)))
    assert bar.ax.get_ylabel() == "u [m/s]"
    plt.close(figure)


def test_a_data_derived_shear_of_reach_just_below_its_rounded_power_of_ten_bands_and_labels_in_one_power_of_ten() -> None:
    """A Kxy reach of 9.7e-11 m² rounds up to 1.1e-10: centres -1.0 to 1.0 in 10⁻¹⁰ m², the power of ten of its bands."""
    figure, bar = _data_bar(np.linspace(-9.7e-11, 9.7e-11, 101), semantic="linear_signed", label="Kxy [m²]")
    assert _texts(bar) == _signed("-1.0", "-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
    assert bar.ax.get_ylabel() == "Kxy [10⁻¹⁰ m²]"
    plt.close(figure)


def test_a_data_derived_near_uniform_positive_state_is_widened_to_distinct_labels() -> None:
    """0.009 to 0.0099 would get bands of 0.0001 with duplicate 3-decimal labels: widened to 11 steps of 0.001 from 0.004."""
    figure, bar = _data_bar(np.linspace(0.009, 0.0099, 101), semantic="linear_positive", label="X [kg/kg]")
    assert _texts(bar) == [f"{value:.3f}" for value in np.linspace(0.004, 0.015, 12)]
    assert bar.ax.get_ylabel() == "X [kg/kg]"
    plt.close(figure)


@pytest.mark.parametrize(
    ("boundaries", "options"),
    [
        (0.0005 + 0.001 * np.arange(10), {}),  # three decimals: 0.0005 and 0.0015 round alike
        (0.005 + 0.01 * np.arange(10), {}),  # the step chooses two decimals
        ((np.arange(12) - 5.5) * 0.001, {"semantic": "linear_signed", "colormap_name": "twilight"}),  # cyclic: labelled at ±0.0055
    ],
)
def test_a_bar_whose_written_labels_would_repeat_is_refused_naming_its_label(boundaries: Any, options: dict[str, Any]) -> None:
    """Locked boundaries off the grid of the bar's format would show one label twice: ValueError naming the bar and the text."""
    with pytest.raises(ValueError, match=r"'T \[K\]' would repeat the tick labels"):
        _slot_bar((1.0, 1.0), boundaries, **options)
    plt.close("all")


def test_a_near_uniform_log_positive_bar_writes_enough_significant_digits_for_distinct_labels() -> None:
    """Kxx from 1e-10 to 1.001e-10 m²: three significant digits would write 1.00 eleven times; every boundary reads once."""
    figure, bar = _data_bar(np.array([1.0e-10, 1.001e-10]), semantic="log_positive", label="Kxx [m²]")
    shown = [text for text in _texts(bar) if text]
    assert shown
    assert len(set(shown)) == len(shown)
    assert bar.ax.get_ylabel() == "Kxx [10⁻¹⁰ m²]"
    plt.close(figure)


def test_maintained_state_bands_keep_distinct_labels_on_a_locked_bar() -> None:
    """The state bands widened from 0 and a signed state's own bands, locked again, label every tick once."""
    figure, bar = _slot_bar((1.0, 1.0), field_maps.state_boundaries(0.0011, 0.0012))
    assert _texts(bar) == [f"{value:.3f}" for value in np.linspace(0.0, 0.011, 12)]
    plt.close(figure)
    for reach in (0.0018, 9.7e-11):
        signed = field_maps.build_field_map(np.array([-reach, reach]), semantic="linear_signed").boundaries
        figure, bar = _slot_bar((1.0, 1.0), signed, semantic="linear_signed")
        shown = [text for text in _texts(bar) if text]
        assert len(set(shown)) == len(shown)
        plt.close(figure)


def test_a_constant_zero_shear_keeps_one_tick_at_zero_in_its_own_unit() -> None:
    """A shear permeability of 0 everywhere is a constant bar: one tick at 0 and no power of ten in the unit bracket."""
    figure, bar = _data_bar(np.zeros(4), semantic="linear_signed", label="Kxy [m²]")
    np.testing.assert_allclose(bar.get_ticks(), [0.0])
    assert _texts(bar) == ["0"]
    assert bar.ax.get_ylabel() == "Kxy [m²]"
    plt.close(figure)


def test_a_state_bar_between_a_thousandth_and_ten_thousand_carries_no_power_of_ten_while_an_error_bar_with_a_finer_step_does() -> None:
    """Steps of 0.001: the linear positive state keeps three decimals in its own unit; an absolute error of steps 0.0005 moves 10⁻⁴ into it."""
    figure, state = _slot_bar((1.0, 1.0), 0.4 + 0.001 * np.arange(11))
    assert state.ax.get_ylabel() == "T [K]"
    assert {len(text.partition(".")[2]) for text in _texts(state) if text} == {3}
    plt.close(figure)
    figure, error = _slot_bar((1.0, 1.0), 0.4 + 0.0005 * np.arange(11), semantic="absolute_error")
    assert error.ax.get_ylabel() == "T [10⁻⁴ K]"
    plt.close(figure)
    # A signed state (such as a shear permeability) keeps the signed band centres but stays a state; a signed error stays an error.
    signed = (np.arange(12) - 5.5) * 0.001
    figure, state = _slot_bar((1.0, 1.0), signed, semantic="linear_signed")
    np.testing.assert_allclose(state.get_ticks(), (signed[:-1] + signed[1:]) / 2.0, atol=1e-15)
    assert state.ax.get_ylabel() == "T [K]"
    plt.close(figure)
    figure, error = _slot_bar((1.0, 1.0), signed / 2.0, semantic="signed_error")
    assert error.ax.get_ylabel() == "T [10⁻⁴ K]"
    plt.close(figure)


def test_a_cyclic_bar_is_labelled_at_its_wrap_boundaries_and_a_constant_one_at_its_value() -> None:
    """An axial direction on twilight names -90 and 90, not the band centres; a constant bar has one tick at its value."""
    orientation = np.linspace(-90.0, 90.0, 10)
    figure, bar = _slot_bar((1.0, 1.0), orientation, semantic="linear_signed", colormap_name="twilight")
    np.testing.assert_allclose(bar.get_ticks(), orientation)
    assert (_texts(bar)[0], _texts(bar)[-1]) == ("\N{MINUS SIGN}90", "90")
    plt.close(figure)
    figure, axes, slots = plot_layout.map_grid(plot_layout.plan_geometry(1, 1, (1.0, 1.0)), extent=(0.0, 1.0, 0.0, 1.0))
    constant = field_maps.build_field_map((0.4, 0.4), semantic="linear_positive")
    bar = plot_layout.add_field_map_colorbar(
        figure, axes[0, 0].pcolormesh(np.zeros((2, 2)), **constant.pcolormesh_kwargs()), slots[0, 0], constant, label="ε [-]"
    )
    np.testing.assert_allclose(bar.get_ticks(), [0.4])
    plt.close(figure)


def test_a_map_of_a_shared_scale_draws_its_own_arrows_and_a_categorical_scale_has_no_slot_bar() -> None:
    """``extend`` replaces the shared scale's arrows for one map; a categorical scale is refused."""
    figure, axes, slots = plot_layout.map_grid(plot_layout.plan_geometry(1, 1, (1.0, 1.0)), extent=(0.0, 1.0, 0.0, 1.0))
    shared = field_maps.build_field_map(0.25 * np.arange(11), semantic="linear_positive", locked_boundaries=0.25 * np.arange(11))
    image = axes[0, 0].pcolormesh(np.zeros((2, 2)), **shared.pcolormesh_kwargs())
    assert plot_layout.add_field_map_colorbar(figure, image, slots[0, 0], shared, label="T [K]", extend="max").extend == "max"
    categorical = field_maps.build_field_map(("sand", "clay"), semantic="categorical")
    with pytest.raises(ValueError, match="categorical"):
        plot_layout.add_field_map_colorbar(figure, image, slots[0, 0], categorical, label="[-]")
    plt.close(figure)


def test_a_flat_bar_thins_its_labels_to_the_real_height_keeps_both_ends_and_thins_a_signed_bar_symmetrically() -> None:
    """Bars of 1.05 and 0.42 in keep both end labels and no two kept labels overlap; a signed bar keeps 0; 0.07 in keeps the top."""
    boundaries = 0.1 * np.arange(13)
    for domain in ((4.0, 1.0), (10.0, 1.0)):
        figure, bar = _slot_bar(domain, boundaries)
        kept = [label for label in bar.ax.get_yticklabels() if label.get_text()]
        assert (kept[0].get_text(), kept[-1].get_text()) == ("0.0", "1.2")
        assert len(kept) < boundaries.size
        extents = [label.get_window_extent() for label in kept]
        assert all(lower.y1 <= upper.y0 for lower, upper in pairwise(extents))
        plt.close(figure)
    figure, bar = _slot_bar((4.0, 1.0), (np.arange(12) - 5.5) * 0.2, semantic="linear_signed")
    texts = _texts(bar)
    assert [bool(text) for text in texts] == [bool(text) for text in reversed(texts)]
    assert (texts[0], texts[5], texts[-1]) == ("\N{MINUS SIGN}1.0", "0.0", "1.0")
    assert "" in texts
    plt.close(figure)
    figure, bar = _slot_bar((60.0, 1.0), boundaries)
    assert [text for text in _texts(bar) if text] == ["1.2"]
    plt.close(figure)


def test_a_short_signed_bar_keeps_its_zero_label() -> None:
    """0 stays with both ends while each half of the bar holds one label (0.42 in), and alone on a shorter bar (0.07 in)."""
    signed = (np.arange(12) - 5.5) * 0.2
    figure, bar = _slot_bar((10.0, 1.0), signed, semantic="linear_signed")
    assert [text for text in _texts(bar) if text] == ["\N{MINUS SIGN}1.0", "0.0", "1.0"]
    plt.close(figure)
    figure, bar = _slot_bar((60.0, 1.0), signed, semantic="linear_signed")
    assert [text for text in _texts(bar) if text] == ["0.0"]
    plt.close(figure)


def test_window_axes_are_one_column_per_window_with_a_y_axis_shared_along_each_row() -> None:
    """A window column is as wide as its ratio asks, a row is as high as its height, and y tick labels stay on the first column."""
    figure = plt.figure(figsize=(8.0, 4.0))
    try:
        windows = time_ui.drying_windows(120.0, before_end=False)
        axes = plot_layout.window_axes(figure, figure.add_gridspec(1, 1)[0, 0], windows, [1.0, 0.5], wspace=0.1)
        assert axes.shape == (2, len(windows))
        first, middle, last = (axis.get_position().width for axis in axes[0])
        assert first == pytest.approx(last)
        assert first == pytest.approx(middle * windows[0].width_ratio / windows[1].width_ratio)
        assert axes[1, 0].get_position().height == pytest.approx(0.5 * axes[0, 0].get_position().height)
        for row in axes:
            assert all(row[0].get_shared_y_axes().joined(row[0], axis) for axis in row[1:])
            assert [axis.yaxis.get_tick_params()["labelleft"] for axis in row] == [True, False, False]
        assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])
    finally:
        plt.close(figure)
