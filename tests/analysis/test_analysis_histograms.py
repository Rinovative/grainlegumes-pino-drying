# ruff: noqa: S101, D100, D103, PLR2004
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import BarContainer

from src.analysis.presentation import histograms


def test_nonconstant_histogram_preserves_caller_bins() -> None:
    figure, axis = plt.subplots()
    try:
        artists = histograms.plot_histogram(
            axis,
            np.asarray((0.0, 1.0, 2.0)),
            bins=np.asarray((0.0, 0.5, 1.5, 2.5)),
            color="tab:blue",
        )
        np.testing.assert_allclose(artists.bin_edges, (0.0, 0.5, 1.5, 2.5))
        assert artists.bars is not None
        assert artists.constant_line is None
        assert artists.constant_value is None
    finally:
        plt.close(figure)


def test_exact_constant_histogram_has_only_one_count_line() -> None:
    figure, axis = plt.subplots()
    try:
        artists = histograms.plot_histogram(
            axis,
            np.asarray((2.0, 2.0, 2.0)),
            bins=20,
            color="tab:orange",
        )
        assert artists.bars is None
        assert artists.bin_edges.size == 0
        assert len(axis.patches) == 0
        assert len(axis.lines) == 1
        np.testing.assert_allclose(artists.heights, (3.0,))
        assert artists.constant_value == 2.0
        assert artists.constant_line is not None
        np.testing.assert_allclose(artists.constant_line.get_xdata(), (2.0, 2.0))
        np.testing.assert_allclose(artists.constant_line.get_ydata(), (0.0, 3.0))
    finally:
        plt.close(figure)


def test_exact_constant_histogram_preserves_weight_and_density_semantics() -> None:
    weighted_figure, weighted_axis = plt.subplots()
    density_figure, density_axis = plt.subplots()
    try:
        weighted = histograms.plot_histogram(
            weighted_axis,
            np.asarray((4.0, 4.0, 4.0)),
            bins=30,
            weights=np.asarray((2.0, 3.0, 5.0)),
        )
        density = histograms.plot_histogram(
            density_axis,
            np.asarray((4.0, 4.0, 4.0)),
            bins=30,
            density=True,
        )
        assert weighted.bars is None
        assert density.bars is None
        assert len(weighted_axis.patches) == 0
        assert len(density_axis.patches) == 0
        assert weighted.heights[0] == pytest.approx(10.0)
        assert weighted.constant_line is not None
        assert weighted.constant_line.get_ydata()[-1] == pytest.approx(10.0)
        assert density.constant_line is not None
        assert density.constant_line.get_ydata()[-1] == pytest.approx(density.heights[0])
    finally:
        plt.close(weighted_figure)
        plt.close(density_figure)


def test_near_constant_unequal_values_use_the_normal_histogram_path() -> None:
    figure, axis = plt.subplots()
    values = np.asarray((1.0, 1.0 + 8.0e-15))
    try:
        artists = histograms.plot_histogram(axis, values, bins=4)
        assert artists.constant_line is None
        assert artists.constant_value is None
        assert isinstance(artists.bars, BarContainer)
        assert len(artists.bars.patches) == 4
    finally:
        plt.close(figure)
