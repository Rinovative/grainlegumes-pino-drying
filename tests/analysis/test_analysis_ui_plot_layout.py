"""Protect the shared physical-map layout contract and its two dependent contracts.

Responsibilities:
  - Check the width cap, constrained layout, colorbar binding and title ownership of map grids
  - Check the fixed-inch map geometry and the slot colour bar's power-of-ten and label thinning
  - Check the round colour-scale boundaries: the smallest 1, 2 or 5 step from a multiple of it
  - Check that the Evaluation and generation-input contracts reuse the shared sizes

Design principles:
  - Geometry is read from drawn figures, never from layout fractions

This module does NOT:
  - Check scientific field values, colour scales or per-plot composition
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as evaluation_layout
from src.analysis.generation_inputs.plots import generation_input_plot_layout as generation_layout
from src.analysis.ui import analysis_ui_plot_layout as plot_layout


def test_width_cap_shrinks_columns_only_above_three_columns() -> None:
    """Three columns keep the contract column width; more columns share the cap."""
    contract = plot_layout.MAP_LAYOUT
    assert contract.figure_size(rows=2, columns=3) == (3 * contract.column_width, 2 * contract.row_height)
    width, height = contract.figure_size(rows=2, columns=5)
    assert width == contract.maximum_width
    assert height == 2 * contract.row_height
    figure, axes = plot_layout.map_subplots(rows=2, columns=5)
    assert figure.get_figwidth() == pytest.approx(contract.maximum_width)
    assert figure.get_layout_engine() is not None
    assert axes.shape == (2, 5)
    plt.close(figure)


def test_map_grid_reuses_a_given_figure_and_keeps_its_size() -> None:
    """A reused figure is cleared, keeps its size and receives the requested axes grid."""
    figure, axes = plot_layout.map_subplots(rows=2, columns=3)
    image = axes[0, 0].imshow(np.arange(6.0).reshape(2, 3))
    plot_layout.add_map_colorbar(figure, image, axes[0, 0])
    size = tuple(figure.get_size_inches())
    reused, new_axes = plot_layout.map_subplots(rows=1, columns=2, figure=figure)
    assert reused is figure
    assert tuple(figure.get_size_inches()) == size
    assert new_axes.shape == (1, 2)
    assert set(figure.axes) == set(new_axes.flat)
    assert figure.get_layout_engine() is not None
    plt.close(figure)


def test_colorbar_stays_bound_to_its_own_map_axis_under_constrained_layout() -> None:
    """Each colorbar sits directly right of its own map and left of the next map."""
    figure, axes = plot_layout.map_subplots(rows=1, columns=2)
    colorbars = []
    for axis in axes[0]:
        image = axis.imshow(np.arange(6.0).reshape(2, 3))
        colorbars.append(plot_layout.add_map_colorbar(figure, image, axis, label="K"))
    figure.canvas.draw()
    first, second = axes[0]
    assert first.get_position().x1 <= colorbars[0].ax.get_position().x0 <= colorbars[0].ax.get_position().x1 <= second.get_position().x0
    assert second.get_position().x1 <= colorbars[1].ax.get_position().x0
    plt.close(figure)


def test_suptitle_over_axes_is_positioned_by_constrained_layout_but_centred_for_manual_layout() -> None:
    """A constrained figure owns the title height; a manual figure is centred over the given axes."""
    constrained, constrained_axes = plot_layout.map_subplots(rows=1, columns=2)
    title = plot_layout.set_suptitle_over_axes(constrained, "Maps", constrained_axes.flat, y=0.5)
    constrained.canvas.draw()
    assert title.get_position()[1] > 0.9
    plt.close(constrained)

    manual = plt.figure(layout=None)
    axis = manual.add_axes((0.1, 0.1, 0.4, 0.6))
    title = plot_layout.set_suptitle_over_axes(manual, "Manual", [axis])
    assert title.get_position()[0] == pytest.approx(0.3)
    plt.close(manual)


def test_fixed_map_geometry_follows_rows_columns_and_domain_and_shrinks_all_maps_under_the_width_cap() -> None:
    """One column keeps the target map width; many columns share the cap; the figure has exactly the planned size."""
    single = plot_layout.plan_geometry(2, 1, (1.2, 0.75))
    assert single.map_width == plot_layout.MAP_WIDTH_IN
    assert single.map_height == pytest.approx(plot_layout.MAP_WIDTH_IN * 0.75 / 1.2)
    assert single.figure_height == pytest.approx(
        plot_layout.MARGIN_TOP_IN + plot_layout.MARGIN_BOTTOM_IN + 2 * single.map_height + plot_layout.GAP_V_IN
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


def test_rounded_boundaries_cover_the_range_with_the_least_span_of_9_to_13_round_bands() -> None:
    """The step is m times a power of ten, m in 1, 2, 2.5, 5; the least span wins, from a step multiple at or below the lower end."""
    expected = {
        (0.0, 0.95): (0.0, 0.1, 10),
        (0.0, 2.0): (0.0, 0.2, 10),
        (0.0, 0.3): (0.0, 0.025, 12),
        (-0.47, 0.47): (-0.5, 0.1, 10),
        (16.25, 304.9): (0.0, 25.0, 13),  # case 42 p: not 0 to 550 Pa in steps of 50
        (-0.04119, 0.04502): (-0.05, 0.01, 10),  # case 42 u
        (0.0745, 0.2338): (0.06, 0.02, 9),  # case 42 v
    }
    for (lower, upper), (start, step, bands) in expected.items():
        boundaries = plot_layout.rounded_boundaries(lower, upper)
        assert boundaries is not None
        np.testing.assert_allclose(boundaries, start + step * np.arange(bands + 1), atol=1e-12)
    assert plot_layout.rounded_boundaries(1.0, 1.0) is None


def test_a_signed_scale_is_symmetric_with_an_odd_band_count_and_the_white_band_centred_on_zero() -> None:
    """Band width w with count / 2 * w at least the reach, the least span: reach 0.3 gives 13 bands of 0.05, reach 1 gives 11 of 0.2."""
    for reach, (width, bands) in {0.3: (0.05, 13), 1.0: (0.2, 11)}.items():
        boundaries = plot_layout.symmetric_boundaries(reach)
        assert boundaries is not None
        np.testing.assert_allclose(boundaries, (np.arange(bands + 1) - bands / 2.0) * width, atol=1e-12)
        assert boundaries[bands // 2] < 0.0 < boundaries[bands // 2 + 1]
    assert plot_layout.symmetric_boundaries(0.0) is None


def test_a_state_scale_whose_step_needs_more_than_the_allowed_decimals_skips_those_steps_or_is_too_narrow() -> None:
    """With max_decimals a step like 0.0025 is skipped for 0.005; a range whose best step is finer than the last decimal has no scale."""
    skipped = plot_layout.rounded_boundaries(0.0, 0.0225, max_decimals=3)
    assert skipped is not None
    assert plot_layout.step_decimals(float(skipped[1] - skipped[0])) <= 3
    assert plot_layout.rounded_boundaries(26.85, 26.85 + 5e-5, max_decimals=3) is None


def test_the_decimals_of_a_step_are_exactly_those_it_needs() -> None:
    """A whole-number step needs none, 2.5 one, 0.25 two."""
    assert [plot_layout.step_decimals(step) for step in (50.0, 25.0, 2.5, 0.25, 0.01, 0.002)] == [0, 0, 1, 2, 2, 3]


def test_boundaries_are_decimal_multiples_of_the_step_and_an_unrepresentable_span_has_no_scale() -> None:
    """A lower end on a step multiple starts there, also as 0.1 + 0.2; a step read off large boundaries keeps its decimals."""
    for lower in (0.3, 0.1 + 0.2):
        boundaries = plot_layout.rounded_boundaries(lower, 1.2)
        assert boundaries is not None
        assert np.array_equal(boundaries, [index / 10 for index in range(3, 13)])
    far = plot_layout.rounded_boundaries(1.0e6, 1.0e6 + 0.02, max_decimals=3)
    assert far is not None
    assert plot_layout.step_decimals(float(far[1] - far[0])) == 3
    assert plot_layout.rounded_boundaries(-1.0e308, 1.0e308) is None
    assert plot_layout.symmetric_boundaries(1.0e308) is None


def test_evaluation_and_generation_contracts_reuse_the_shared_sizes() -> None:
    """Column width, row height and the width cap come from one place."""
    shared = plot_layout.MAP_LAYOUT
    assert evaluation_layout.MAP_LAYOUT.figure_size(rows=1, columns=5) == (shared.maximum_width, shared.row_height)
    assert evaluation_layout.MAP_LAYOUT.composite_size(map_columns=5)[0] == shared.maximum_width
    assert generation_layout.MAP_LAYOUT.figure_width(5) == shared.maximum_width
    assert generation_layout.MAP_LAYOUT.map_row_height == shared.row_height
