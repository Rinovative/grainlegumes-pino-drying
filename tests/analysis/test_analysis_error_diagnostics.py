"""Protect fixed physical-state error scales and selected map semantics.

Responsibilities:
  - Check train-bound fixed scales, normalized error maps, and unavailable invalid diagnostics

Design principles:
  - Physical-state scales retain field and fitted-scaler provenance

This module does NOT:
  - Fit normalizers or select training objectives
"""

import numpy as np
import pytest

from src.analysis.presentation import analysis_error_diagnostics as diagnostics


def _steady_provenance(scale: float = 2.0, dataset: str = "train-dataset", fit_split: str = "train") -> dict:
    return {
        "normalizer": {
            "output_standard_deviations": {"p": scale},
            "fit_split": fit_split,
            "output_normalization": "train_fitted_per_channel_standardization",
            "sha256": "a" * 64,
        },
        "dataset": {"name": dataset},
    }


def test_fixed_state_scale_produces_same_normalized_error_across_frames_and_offsets() -> None:
    """One train-only scale yields the same dimensionless error at shifted references."""
    scale = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    assert isinstance(scale, diagnostics.ReferenceScale)
    first = diagnostics.selected_error_map(
        np.array([[10.0, 20.0]]),
        np.array([[11.0, 22.0]]),
        mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR,
        field="p",
        physical_unit="Pa",
        scale=scale,
    )
    second = diagnostics.selected_error_map(
        np.array([[110.0, 120.0]]),
        np.array([[111.0, 122.0]]),
        mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR,
        field="p",
        physical_unit="Pa",
        scale=scale,
    )
    assert first.map_values is not None
    assert second.map_values is not None
    np.testing.assert_array_equal(first.map_values, second.map_values)
    np.testing.assert_allclose(first.map_values, [[0.5, 1.0]])


def test_original_steady_modes_keep_actual_meanings() -> None:
    """Legacy MAE and reference-RMS percentage retain their definitions."""
    reference = np.array([[3.0, 4.0]])
    prediction = np.array([[4.0, 2.0]])
    mae = diagnostics.selected_error_map(reference, prediction, mode="MAE", field="p", physical_unit="Pa")
    relative = diagnostics.selected_error_map(reference, prediction, mode="Relative [%]", field="p", physical_unit="Pa")
    assert mae.map_values is not None
    assert relative.map_values is not None
    np.testing.assert_array_equal(mae.map_values, [[1.0, 2.0]])
    np.testing.assert_allclose(relative.map_values, np.array([[1.0, 2.0]]) / np.sqrt(12.5) * 100.0, rtol=1e-10)


def test_incompatible_or_missing_scales_and_nonfinite_support_are_unavailable() -> None:
    """Mismatched provenance and invalid support cannot yield a misleading score."""
    first = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    other = diagnostics.steady_reference_scale(_steady_provenance(dataset="other"), field="p", unit="Pa")
    assert isinstance(first, diagnostics.ReferenceScale)
    assert isinstance(other, diagnostics.ReferenceScale)
    with pytest.raises(ValueError, match="incompatible"):
        diagnostics.require_common_reference_scale((first, other))
    missing = diagnostics.steady_reference_scale(_steady_provenance(scale=0.0), field="p", unit="Pa")
    result = diagnostics.selected_error_map(
        np.array([[1.0]]), np.array([[2.0]]), mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR, field="p", physical_unit="Pa", scale=missing
    )
    assert result.map_values is None
    assert result.reason is not None
    invalid = diagnostics.selected_error_map(
        np.array([[1.0]]), np.array([[np.nan]]), mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR, field="p", physical_unit="Pa", scale=first
    )
    assert invalid.map_values is None
    assert "Non-finite" in str(invalid.reason)


def test_transient_state_std_is_used_even_for_delta_rms_training_and_derived_is_unavailable() -> None:
    """Reconstructed state errors use persisted state_std, never delta_rms."""
    state = {
        "state_std": [2.0, 0.25],
        "delta_rms": [0.01, 0.02],
        "scale_mode": "delta_rms",
        "train_membership_digest": "train",
        "state_names": ["T", "phi"],
    }
    temperature = diagnostics.transient_reference_scale(state, field="T", unit="K", state_order=("T", "phi"), dataset_identity="train")
    assert isinstance(temperature, diagnostics.ReferenceScale)
    np.testing.assert_allclose(temperature.value, 2.0)
    contradicted = diagnostics.transient_reference_scale(
        state,
        field="T",
        unit="K",
        state_order=("T", "phi"),
        dataset_identity="train",
        expected_scaler_identity="different",
    )
    assert isinstance(contradicted, str)
    assert "contradicts" in contradicted
    derived = diagnostics.transient_reference_scale(state, field="w_gr", unit="1", state_order=("T", "phi"), dataset_identity="train")
    assert isinstance(derived, str)
    assert "derived" in derived


def test_reference_scale_rejects_non_train_direct_construction_and_field_mismatch() -> None:
    """Public scale objects cannot bypass validated field and train provenance."""
    scale = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    assert isinstance(scale, diagnostics.ReferenceScale)
    with pytest.raises(ValueError, match="train-only"):
        diagnostics.ReferenceScale("p", "Pa", 0.0, "test", "train", "dataset", "scaler")
    with pytest.raises(ValueError, match="train-only"):
        diagnostics.ReferenceScale("p", "Pa", 1.0, "test", "test", "dataset", "scaler")
    for invalid in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="positive finite"):
            diagnostics.ReferenceScale("p", "Pa", invalid, "test", "train", "dataset", "scaler")
    untrained = diagnostics.steady_reference_scale(_steady_provenance(fit_split="test"), field="p", unit="Pa")
    assert isinstance(untrained, str)
    mismatched = diagnostics.selected_error_map(
        np.array([[1.0]]), np.array([[2.0]]), mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR, field="u", physical_unit="Pa", scale=scale
    )
    assert mismatched.map_values is None
    assert "different physical field" in str(mismatched.reason)


def test_equal_numeric_scale_with_different_fitted_scaler_cannot_be_compared() -> None:
    """The saved scaler identity participates in fixed-scale comparison admission."""
    first = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    alternate_provenance = _steady_provenance()
    alternate_provenance["normalizer"]["sha256"] = "b" * 64
    alternate = diagnostics.steady_reference_scale(alternate_provenance, field="p", unit="Pa")
    assert isinstance(first, diagnostics.ReferenceScale)
    assert isinstance(alternate, diagnostics.ReferenceScale)
    with pytest.raises(ValueError, match="incompatible"):
        diagnostics.require_common_reference_scale((first, alternate))


def test_finite_inputs_with_overflowed_selected_error_are_unavailable() -> None:
    """Finite physical inputs cannot produce an infinite map or a false zero score."""
    scale = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    assert isinstance(scale, diagnostics.ReferenceScale)
    overflowed = diagnostics.selected_error_map(
        np.array([[-1e308]]),
        np.array([[1e308]]),
        mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR,
        field="p",
        physical_unit="Pa",
        scale=scale,
    )
    assert overflowed.map_values is None
    assert "non-finite" in str(overflowed.reason)


def test_unsupported_cells_of_a_selected_error_map_are_nan() -> None:
    """Cells outside the required support stay unavailable instead of becoming zero."""
    scale = diagnostics.steady_reference_scale(_steady_provenance(), field="p", unit="Pa")
    assert isinstance(scale, diagnostics.ReferenceScale)
    selected = diagnostics.selected_error_map(
        np.array([[1.0, 1.0, 1.0]]),
        np.array([[2.0, 99.0, 3.0]]),
        mode=diagnostics.NORMALIZED_ABSOLUTE_ERROR,
        field="p",
        physical_unit="Pa",
        scale=scale,
        support=np.array([[True, False, True]]),
    )
    assert selected.map_values is not None
    np.testing.assert_allclose(selected.map_values, [[0.5, np.nan, 1.0]], equal_nan=True)
