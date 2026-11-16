"""Protect physical field-map coordinates, bounds, and colorbar contracts.

Responsibilities:
  - Check discrete boundaries, signed and categorical scales, overflow, and definition isolation
  - Check the round band rules: the least span of 9 to 13 bands of 1, 2, 2.5 or 5 times a power of ten from a step
    multiple, an odd symmetric count for a signed scale, the decimals a step needs, and data-derived linear scales on them
  - Check a state's power of ten from the size of its values and the state bands built in it, also across a power of ten

Design principles:
  - Rendering preserves input values and exact locked color boundaries

This module does NOT:
  - Quantize scientific data or validate model predictions
  - Check colour-bar tick labels or map geometry (see the plot-layout tests)
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm

from src.analysis.presentation import field_maps


def test_a_data_derived_positive_scale_has_one_color_per_round_band_over_the_values() -> None:
    definition = field_maps.build_field_map((0.03, 0.2, 0.79), semantic="linear_positive", field="phi")
    np.testing.assert_array_equal(definition.boundaries, np.arange(10) / 10)  # 9 bands of 0.1 from 0 reach 0.79
    assert definition.color_count == definition.boundaries.size - 1
    assert isinstance(definition.normalizer, BoundaryNorm)
    assert definition.contourf_kwargs()["levels"] is definition.boundaries
    assert definition.contourf_kwargs()["extend"] == "neither"
    colorbar_kwargs = definition.colorbar_kwargs()
    assert colorbar_kwargs["ticks"] is definition.ticks
    assert colorbar_kwargs["extend"] == "neither"
    assert definition.pcolormesh_kwargs()["norm"] is definition.normalizer
    assert definition.imshow_kwargs()["cmap"] is definition.colormap


def test_a_data_derived_signed_scale_has_an_odd_round_band_count_with_zero_in_the_neutral_band() -> None:
    definition = field_maps.build_field_map((-2.0, 0.0, 1.5), semantic="linear_signed", field="u")
    np.testing.assert_allclose(definition.boundaries, (np.arange(10) - 4.5) * 0.5)  # 9 bands of 0.5 reach the magnitude 2
    centre = definition.color_count // 2
    assert definition.color_count % 2 == 1
    assert definition.boundaries[centre] < 0.0 < definition.boundaries[centre + 1]
    assert definition.normalizer(0.0) == centre


def test_a_sequential_scale_bands_the_values_own_range_of_either_sign_without_a_zero_centre() -> None:
    definition = field_maps.build_field_map((-7.4, -7.2, -6.9), semantic="linear_sequential", field="Kxx")
    np.testing.assert_allclose(definition.boundaries, np.linspace(-7.4, -6.9, 11))  # the state bands of the range, no centre
    assert definition.color_count == definition.boundaries.size - 1
    assert definition.normalizer(-7.39) == 0
    assert definition.normalizer(-6.91) == definition.color_count - 1


def test_log_positive_boundaries_have_truthful_geometric_spacing() -> None:
    definition = field_maps.build_field_map((1e-4, 1e-2, 1.0), semantic="log_positive", field="Kxx")
    np.testing.assert_allclose(definition.boundaries, np.geomspace(1e-4, 1.0, 12))  # eleven geometric bands, not round ones
    assert np.all(definition.boundaries > 0.0)
    assert np.allclose(np.diff(np.log(definition.boundaries)), np.diff(np.log(definition.boundaries))[0])
    with pytest.raises(ValueError, match="strictly positive"):
        field_maps.build_field_map((0.0, 1.0), semantic="log_positive")


def test_absolute_and_relative_errors_reject_negative_values() -> None:
    for semantic in ("absolute_error", "relative_error"):
        with pytest.raises(ValueError, match="nonnegative"):
            field_maps.build_field_map((-0.1, 0.2), semantic=semantic)


def test_categorical_definition_uses_actual_categories_without_expansion() -> None:
    source = np.asarray(("bed", "air", "bed"), dtype=object)
    definition = field_maps.build_field_map(source, semantic="categorical")
    assert definition.color_count == 2
    assert definition.category_labels == ("bed", "air")
    assert definition.contourf_kwargs()["levels"] is definition.boundaries
    assert np.array_equal(source, np.asarray(("bed", "air", "bed"), dtype=object))


def test_constant_definition_does_not_fabricate_variation() -> None:
    definition = field_maps.build_field_map((3.5, 3.5), semantic="constant")
    assert definition.constant_value == 3.5
    assert definition.color_count == 1
    assert np.array_equal(definition.boundaries, np.asarray((3.5,)))
    assert "levels" not in definition.contourf_kwargs()
    with pytest.raises(ValueError, match="exactly one observed value"):
        field_maps.build_field_map((3.5, 4.0), semantic="constant")
    with pytest.raises(ValueError, match="10 to 14 finite strictly increasing"):
        field_maps.build_field_map((1.0, np.nextafter(1.0, 2.0)), semantic="absolute_error")
    # The same one-ulp range as a positive state is widened to 11 steps of 0.001 about its centre.
    widened = field_maps.build_field_map((1.0, np.nextafter(1.0, 2.0)), semantic="linear_positive").boundaries
    np.testing.assert_allclose(np.diff(widened), np.full(11, 0.001), rtol=1e-9)
    assert widened[0] < 1.0 < widened[-1]


def test_a_constant_signed_field_takes_the_colour_its_sign_has_on_a_symmetric_scale() -> None:
    """A constant zero shear is the neutral centre colour, never an end colour; a constant negative or positive one takes its end."""
    for value, position in ((0.0, 0.5), (-2.0e-10, 0.0), (3.0e-10, 1.0)):
        definition = field_maps.build_field_map(np.full(4, value), semantic="linear_signed", colormap_name="RdBu_r")
        assert definition.semantic is field_maps.FieldMapSemantic.CONSTANT
        assert definition.colormap.name == "RdBu_r"
        np.testing.assert_allclose(definition.colormap(definition.normalizer(value)), colormaps["RdBu_r"](position), atol=0.01)


def test_locked_definition_preserves_every_exact_boundary() -> None:
    source = field_maps.build_field_map((0.0, 1.0), semantic="linear_positive", comparison_scope="reference-prediction")
    locked = field_maps.build_field_map((0.2, 0.3), semantic="linear_positive", comparison_scope="reference-prediction", locked_boundaries=source)
    assert source.comparison_scope == "reference-prediction"
    assert not source.is_locked
    assert locked.is_locked
    assert locked.boundaries.flags.writeable is False
    assert np.array_equal(locked.boundaries, source.boundaries)
    with pytest.raises(ValueError, match="same scientific semantic"):
        field_maps.build_field_map((-1.0, 1.0), semantic="linear_signed", locked_boundaries=source)


def test_locked_definition_reports_overflow_without_adding_visible_colors() -> None:
    source = field_maps.build_field_map((0.0, 1.0), semantic="linear_positive")
    locked = field_maps.build_field_map((0.25, 1.5), semantic="linear_positive", locked_boundaries=source)
    assert locked.extend == "max"
    assert locked.color_count == source.color_count
    assert np.array_equal(locked.boundaries, source.boundaries)
    assert locked.contourf_kwargs()["extend"] == "max"
    assert locked.colorbar_kwargs()["extend"] == "max"
    assert locked.normalizer.clip is False


def test_locked_round_boundaries_keep_one_color_per_band_from_9_to_13_bands() -> None:
    """A locked scale of 13 bands has 13 colors; a signed one needs an odd count with its neutral band around zero; 14 bands are refused."""
    boundaries = 25.0 * np.arange(14)
    locked = field_maps.build_field_map((10.0, 300.0), semantic="linear_positive", locked_boundaries=boundaries)
    assert (locked.color_count, locked.boundaries.size) == (13, 14)
    signed = field_maps.build_field_map((-0.3, 0.3), semantic="signed_error", locked_boundaries=(np.arange(10) - 4.5) * 0.1)
    assert signed.color_count == 9
    with pytest.raises(ValueError, match="finite strictly increasing"):
        field_maps.build_field_map((10.0, 300.0), semantic="linear_positive", locked_boundaries=25.0 * np.arange(15))
    with pytest.raises(ValueError, match="neutral center"):
        field_maps.build_field_map((-0.3, 0.3), semantic="signed_error", locked_boundaries=(np.arange(11) - 5.0) * 0.1)


def test_repeated_builds_return_isolated_definitions_without_sharing_mutable_renderers() -> None:
    first = field_maps.build_field_map((0.0, 1.0), semantic="linear_positive", field="phi", unit="1", comparison_scope="case-a")
    first.normalizer.clip = False
    first.colormap.colorbar_extend = True
    second = field_maps.build_field_map((0.0, 1.0), semantic="linear_positive", field="phi", unit="1", comparison_scope="case-a")
    different_scope = field_maps.build_field_map((0.0, 1.0), semantic="linear_positive", field="phi", unit="1", comparison_scope="case-b")

    assert first is not second
    assert second.normalizer.clip is True
    assert first.colormap.colorbar_extend is True
    assert second.colormap.colorbar_extend is False
    assert np.array_equal(first.boundaries, second.boundaries)
    with pytest.raises(ValueError, match="WRITEABLE"):
        second.boundaries.setflags(write=True)
    assert different_scope.comparison_scope == "case-b"


def test_continuous_input_remains_unmodified_and_unquantized() -> None:
    values = np.asarray((0.03, 0.26, 0.79), dtype=np.float64)
    original = values.copy()
    definition = field_maps.build_field_map(values, semantic="linear_positive")
    assert np.array_equal(values, original)
    assert not np.array_equal(values, definition.normalizer(values))
    assert np.array_equal(values, original)


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
        boundaries = field_maps.rounded_boundaries(lower, upper)
        assert boundaries is not None
        np.testing.assert_allclose(boundaries, start + step * np.arange(bands + 1), atol=1e-12)
    assert field_maps.rounded_boundaries(1.0, 1.0) is None


def test_rounded_boundaries_reach_a_range_near_the_smallest_doubles() -> None:
    """A width of 1e-300 still gets 9 to 13 round bands (its step's reciprocal power would overflow); a subnormal width gets none."""
    boundaries = field_maps.rounded_boundaries(0.0, 1e-300)
    assert boundaries is not None
    assert 10 <= boundaries.size <= 14
    assert boundaries[0] <= 0.0
    assert boundaries[-1] >= 1e-300
    assert field_maps.rounded_boundaries(0.0, 1e-320) is None
    assert field_maps.symmetric_boundaries(1e-320) is None


def test_a_signed_scale_is_symmetric_with_an_odd_band_count_and_the_white_band_centred_on_zero() -> None:
    """Band width w with count / 2 * w at least the reach, the least span: reach 0.3 gives 13 bands of 0.05, reach 1 gives 11 of 0.2."""
    for reach, (width, bands) in {0.3: (0.05, 13), 1.0: (0.2, 11)}.items():
        boundaries = field_maps.symmetric_boundaries(reach)
        assert boundaries is not None
        np.testing.assert_allclose(boundaries, (np.arange(bands + 1) - bands / 2.0) * width, atol=1e-12)
        assert boundaries[bands // 2] < 0.0 < boundaries[bands // 2 + 1]
    assert field_maps.symmetric_boundaries(0.0) is None


def test_a_state_scale_whose_step_needs_more_than_the_allowed_decimals_skips_those_steps_or_is_too_narrow() -> None:
    """With max_decimals a step like 0.0025 is skipped for 0.005; a range whose best step is finer than the last decimal has no scale."""
    skipped = field_maps.rounded_boundaries(0.0, 0.0225, max_decimals=3)
    assert skipped is not None
    assert field_maps.step_decimals(float(skipped[1] - skipped[0])) <= 3
    assert field_maps.rounded_boundaries(26.85, 26.85 + 5e-5, max_decimals=3) is None


def test_a_state_takes_the_power_of_ten_of_its_largest_magnitude_only_outside_a_thousandth_to_ten_thousand() -> None:
    """Zero and magnitudes in [10^-3, 10^4) read without a factor; a shear permeability of ±2e-10 takes 10^-10, a pressure of 2.5e4 Pa 10^4."""
    cases = {(0.0, 0.0): 0, (293.15, 303.15): 0, (0.0, 1.0e-3): 0, (-9999.0, 0.0): 0, (0.0, 9.9e-4): -4, (-2.0e-10, 2.0e-10): -10, (0.0, 2.5e4): 4}
    assert {bounds: field_maps.state_exponent(*bounds) for bounds in cases} == cases


def test_state_bands_of_tiny_values_follow_every_state_rule_in_their_power_of_ten() -> None:
    """In units of 10^-10: round bands of 0.1, a constant spanning its magnitude, 11 steps of 0.001 about a near-uniform centre."""
    np.testing.assert_allclose(field_maps.state_boundaries(1.0e-10, 2.0e-10), np.arange(10, 21) * 1.0e-11, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(field_maps.state_boundaries(1.0e-10, 1.0e-10), np.arange(10, 21) * 1.0e-11, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(field_maps.state_boundaries(1.2345e-10, 1.2346e-10), np.linspace(1.229e-10, 1.240e-10, 12), rtol=1e-12, atol=0.0)


def test_state_bands_that_round_up_across_a_power_of_ten_are_built_again_in_the_larger_one() -> None:
    """9.9960e-11 to 9.9965e-11 widened in 10^-11 would reach 1.0002e-10: built in 10^-10 instead, 11 steps of 0.001 from 0.994."""
    boundaries = field_maps.state_boundaries(9.9960e-11, 9.9965e-11)
    np.testing.assert_allclose(boundaries, np.linspace(0.994e-10, 1.005e-10, 12), rtol=1e-12, atol=0.0)
    assert field_maps.state_exponent(float(boundaries[0]), float(boundaries[-1])) == -10


def test_data_derived_states_follow_the_state_rule_and_errors_keep_their_own_bands() -> None:
    """
    Signed states band in the power of ten of their values with at most three decimals there, positive ones as ``state_boundaries``.

    Reach 0.0018: the 11 bands of 0.001 centred on 0; reach 9.7e-11: 11 bands of 0.2 in 10^-10, settled there; 0.009 to
    0.0099: widened to 11 steps of 0.001. A signed error of reach 0.0018 keeps its finer round bands.
    """

    def boundaries(values: np.ndarray, semantic: str) -> np.ndarray:
        return field_maps.build_field_map(values, semantic=semantic).boundaries

    np.testing.assert_allclose(boundaries(np.array([-0.0018, 0.0012]), "linear_signed"), (np.arange(12) - 5.5) * 0.001, rtol=1e-12, atol=0.0)
    shear = boundaries(np.array([-9.7e-11, 9.0e-11]), "linear_signed")
    np.testing.assert_allclose(shear, (np.arange(12) - 5.5) * 2.0e-11, rtol=1e-12, atol=0.0)
    assert field_maps.state_exponent(float(shear[0]), float(shear[-1])) == -10
    np.testing.assert_allclose(boundaries(np.array([0.009, 0.0099]), "linear_positive"), np.linspace(0.004, 0.015, 12), rtol=1e-12, atol=0.0)
    error = boundaries(np.array([-0.0018, 0.0012]), "signed_error")
    assert error[-1] < 0.0025
    assert field_maps.step_decimals(float(error[1] - error[0])) > field_maps.MAX_TICK_DECIMALS


def test_a_near_uniform_nonnegative_state_is_widened_from_zero_and_one_reaching_below_zero_about_its_centre() -> None:
    """0.0011 to 0.0012 gets 11 steps of 0.001 from 0, never negative bands; -0.0001 to 0.0002 keeps its centred window."""
    np.testing.assert_allclose(field_maps.state_boundaries(0.0011, 0.0012), np.linspace(0.0, 0.011, 12), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(field_maps.widened_range(-0.0001, 0.0002, 3), (-0.005, 0.006), rtol=0.0, atol=1e-15)


def test_the_decimals_of_a_step_are_exactly_those_it_needs() -> None:
    """A whole-number step needs none, 2.5 one, 0.25 two."""
    assert [field_maps.step_decimals(step) for step in (50.0, 25.0, 2.5, 0.25, 0.01, 0.002)] == [0, 0, 1, 2, 2, 3]


def test_boundaries_are_decimal_multiples_of_the_step_and_an_unrepresentable_span_has_no_scale() -> None:
    """A lower end on a step multiple starts there, also as 0.1 + 0.2; a step read off large boundaries keeps its decimals."""
    for lower in (0.3, 0.1 + 0.2):
        boundaries = field_maps.rounded_boundaries(lower, 1.2)
        assert boundaries is not None
        assert np.array_equal(boundaries, [index / 10 for index in range(3, 13)])
    far = field_maps.rounded_boundaries(1.0e6, 1.0e6 + 0.02, max_decimals=3)
    assert far is not None
    assert field_maps.step_decimals(float(far[1] - far[0])) == 3
    assert field_maps.rounded_boundaries(-1.0e308, 1.0e308) is None
    assert field_maps.symmetric_boundaries(1.0e308) is None
