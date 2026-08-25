# ruff: noqa: D100, D103, PLR2004, S101
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from src.analysis.presentation import field_maps


def test_continuous_definition_has_exact_eleven_colors_and_twelve_boundaries() -> None:
    definition = field_maps.build_field_map((0.0, 0.2, 1.0), semantic="linear_positive", field="phi")
    assert definition.color_count == 11
    assert definition.boundaries.size == 12
    assert np.all(np.diff(definition.boundaries) > 0.0)
    assert isinstance(definition.normalizer, BoundaryNorm)
    assert definition.contourf_kwargs()["levels"] is definition.boundaries
    assert definition.contourf_kwargs()["extend"] == "neither"
    colorbar_kwargs = definition.colorbar_kwargs()
    assert colorbar_kwargs["ticks"] is definition.ticks
    assert colorbar_kwargs["extend"] == "neither"
    assert definition.pcolormesh_kwargs()["norm"] is definition.normalizer
    assert definition.imshow_kwargs()["cmap"] is definition.colormap


def test_signed_definition_has_symmetric_neutral_center_interval() -> None:
    definition = field_maps.build_field_map((-2.0, 0.0, 1.5), semantic="linear_signed", field="u")
    assert np.isclose(definition.boundaries[0], -definition.boundaries[-1])
    assert definition.boundaries[5] < 0.0 < definition.boundaries[6]
    assert definition.normalizer(definition.boundaries[5] / 2.0) == 5


def test_log_positive_boundaries_have_truthful_geometric_spacing() -> None:
    definition = field_maps.build_field_map((1e-4, 1e-2, 1.0), semantic="log_positive", field="Kxx")
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
    assert np.array_equal(definition.encode_categories(source), np.asarray((0, 1, 0)))
    assert np.array_equal(source, np.asarray(("bed", "air", "bed"), dtype=object))


def test_constant_definition_does_not_fabricate_variation() -> None:
    definition = field_maps.build_field_map((3.5, 3.5), semantic="constant")
    assert definition.constant_value == 3.5
    assert definition.color_count == 1
    assert np.array_equal(definition.boundaries, np.asarray((3.5,)))
    assert "levels" not in definition.contourf_kwargs()
    with pytest.raises(ValueError, match="exactly one observed value"):
        field_maps.build_field_map((3.5, 4.0), semantic="constant")
    with pytest.raises(ValueError, match="twelve finite strictly increasing"):
        field_maps.build_field_map((1.0, np.nextafter(1.0, 2.0)), semantic="linear_positive")


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
    assert locked.color_count == 11
    assert locked.boundaries.size == 12
    assert locked.contourf_kwargs()["extend"] == "max"
    assert locked.colorbar_kwargs()["extend"] == "max"
    assert locked.normalizer.clip is False


def test_identical_cache_reuse_is_bounded_without_sharing_mutable_renderers() -> None:
    field_maps.clear_field_map_cache()
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
    cache_info = field_maps.field_map_cache_info()
    assert cache_info.hits == 1
    assert cache_info.currsize <= 128
    assert different_scope.comparison_scope == "case-b"


def test_continuous_input_remains_unmodified_and_unquantized() -> None:
    values = np.asarray((0.03, 0.26, 0.79), dtype=np.float64)
    original = values.copy()
    definition = field_maps.build_field_map(values, semantic="linear_positive")
    assert np.array_equal(values, original)
    assert not np.array_equal(values, definition.normalizer(values))
    assert np.array_equal(values, original)
