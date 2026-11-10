"""Protect the shared physical-map layout contract and its two dependent contracts.

Responsibilities:
  - Check the width cap, constrained layout, colorbar binding and title ownership of map grids
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


def test_evaluation_and_generation_contracts_reuse_the_shared_sizes() -> None:
    """Column width, row height and the width cap come from one place."""
    shared = plot_layout.MAP_LAYOUT
    assert evaluation_layout.MAP_LAYOUT.figure_size(rows=1, columns=5) == (shared.maximum_width, shared.row_height)
    assert evaluation_layout.MAP_LAYOUT.composite_size(map_columns=5)[0] == shared.maximum_width
    assert generation_layout.MAP_LAYOUT.figure_width(5) == shared.maximum_width
    assert generation_layout.MAP_LAYOUT.map_row_height == shared.row_height
