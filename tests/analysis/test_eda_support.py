"""Protect unequal trajectory support without synthetic tails or interpolation.

Responsibilities:
  - Check exact-time aggregate support, case weighting, scale contracts, and spectral intervals

Design principles:
  - Unequal trajectories retain their tails without interpolation or synthetic contributors

This module does NOT:
  - Extend source trajectories or fit training normalizers
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from src.analysis.eda import eda_transient as transient
from src.analysis.eda.eda_dataframe import PublishedTensorValue
from src.analysis.eda.plots import eda_plot_case_statistics as statistics
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra
from src.analysis.eda.plots import eda_plot_transient as plots


def test_aggregate_preserves_long_tail_and_equal_case_weight() -> None:
    """Unequal trajectory lengths retain observed tails and unweighted cases."""
    times, low, median, high, counts = transient.aggregate_exact_series(
        [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])],
        [np.array([10.0, 20.0, 80.0]), np.array([30.0, 40.0])],
    )
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(counts, [2, 2, 1])
    np.testing.assert_allclose(median, [20.0, 30.0, 80.0])
    np.testing.assert_allclose(low, [12.0, 22.0, 80.0])
    np.testing.assert_allclose(high, [28.0, 38.0, 80.0])


def test_disjoint_timestamps_do_not_acquire_interpolated_contributors() -> None:
    """Different time coordinates contribute only at their stored times."""
    times, _low, median, _high, counts = transient.aggregate_exact_series(
        [np.array([0.0, 2.0]), np.array([1.0, 3.0])],
        [np.array([4.0, 8.0]), np.array([11.0, 15.0])],
    )
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])
    np.testing.assert_array_equal(median, [4.0, 11.0, 8.0, 15.0])


@pytest.mark.parametrize(
    ("times", "values"),
    [
        (np.array([0.0, 1.0]), np.array([0.0, np.nan])),
        (np.array([0.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([2, 1], dtype=np.uint32), np.array([0.0, 1.0])),
    ],
)
def test_invalid_source_support_fails_instead_of_shrinking_score(
    times: np.ndarray,
    values: np.ndarray,
) -> None:
    """Invalid values or duplicate/out-of-order times fail scientific admission."""
    with pytest.raises(ValueError, match="finite aligned values"):
        transient.aggregate_exact_series([times], [values])


def test_single_contributor_has_line_without_dispersion_band() -> None:
    """One observed value is shown without implying population dispersion."""
    figure, axes = plt.subplots(1, 2)
    try:
        plots._plot_split_time_series(
            axes[0],
            axes[1],
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            lower=np.array([1.0, 2.0, 3.0]),
            upper=np.array([1.0, 2.0, 3.0]),
            color="black",
            contributor_count=np.ones(3, dtype=np.int64),
        )
        assert all(axis.lines for axis in axes)
        assert all(not collection.get_paths() for axis in axes for collection in axis.collections)
    finally:
        plt.close(figure)


def test_count_display_preserves_every_exact_time_and_support_change() -> None:
    """The exported figure retains n(t), including gaps and nonmonotone support."""
    figure, axes = plt.subplots(1, 2)
    try:
        times = np.array([0.0, 1.0, 3.0, 6.0])
        counts = np.array([3, 1, 2, 1])
        plots._plot_contributors(axes[0], axes[1], times, counts, color="black")
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), times[1:] / 24.0)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), counts[1:])
        np.testing.assert_array_equal(axes[1].lines[0].get_xdata(), times[:2])
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), counts[:2])
        assert all(axis.lines[0].get_linestyle() == "None" for axis in axes)
    finally:
        plt.close(figure)


def test_exact_state_sequences_do_not_invert_rounded_float32_increments() -> None:
    """The actual endpoint survives a valid delta whose rounded inverse differs."""
    states = np.asarray([np.full((4, 2, 2), value) for value in (0.35, 0.000123, 0.0001)], dtype=np.float32)
    increments = states[1:] - states[:-1]
    assert not np.array_equal(states[0] + increments[0], states[1])
    item = {
        "state": states[0],
        "state_sequence": states,
        "target": increments,
        "boundary": np.zeros((2, 4), dtype=np.float32),
        "time": {"t_n": np.array([0.0, 1.0]), "t_n_plus_1": np.array([1.0, 2.0]), "dt": np.ones(2)},
    }
    records = transient._runtime_transitions(item)
    np.testing.assert_array_equal(records[0]["next_state"], states[1])
    np.testing.assert_array_equal(records[1]["state"], states[1])
    item["target"] = np.zeros_like(increments)
    with pytest.raises(ValueError, match="stored state sequence"):
        transient._runtime_transitions(item)


def test_published_tensor_scales_follow_signed_and_encoded_field_contracts() -> None:
    """An all-positive signed field stays centered; log-encoded K stays linear."""
    x, y = np.meshgrid(np.arange(3), np.arange(2))
    values = np.arange(1.0, 7.0).reshape(2, 3)
    for name, unit, representation, array, signed in (
        ("Kxy", "1", "dimensionless_cross_component_ratio_to_geometric_mean", values, True),
        ("u", "m/s", "identity", values, True),
        ("delta_T", "K", "physical_increment", values, True),
        ("Kxx", "1", "dimensionless_log10_ratio_to_1_m2", -values, True),
        ("Kxx", "m^2", "identity", values * 1e-9, False),
    ):
        value = PublishedTensorValue("static", "input", name, unit, representation, "spatial", array)
        figure = statistics.plot_published_tensor(value, x=x, y=y, display_spatial_stride=2)
        try:
            image = figure.axes[0].collections[0]
            assert isinstance(image.norm, BoundaryNorm)
            boundaries = image.norm.boundaries
            assert image.cmap.N == 11
            assert len(boundaries) == 12
            if signed:
                assert boundaries[0] == -boundaries[-1]
                assert boundaries[5] < 0 < boundaries[6]
            else:
                np.testing.assert_allclose(np.diff(np.log(boundaries)), np.repeat(np.diff(np.log(boundaries))[0], 11))
        finally:
            plt.close(figure)


def test_spectral_fraction_maps_share_eleven_fixed_logarithmic_intervals() -> None:
    """Both spectral orientations share the established numerical floor and range."""
    scale = spectra._power_fraction_scale()
    assert scale.color_count == 11
    np.testing.assert_allclose(scale.boundaries, np.geomspace(1e-12, 1.0, 12))
    assert scale.is_locked
