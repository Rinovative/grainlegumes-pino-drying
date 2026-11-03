"""Protect exact comparison grids and display-only endpoint sampling.

Responsibilities:
  - Reject mismatched coordinates and source indices; check aligned endpoint-preserving display
    sampling

Design principles:
  - Comparison admission uses physical grid identity rather than array shape

This module does NOT:
  - Resample scientific comparison grids or generate artifacts
"""

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.evaluation import evaluation_transient_session as transient_session
from src.analysis.presentation import analysis_display_grid as grids


def _coordinates(offset: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 6) + offset, np.linspace(0.0, 1.0, 5))
    return np.stack((x, y))


def test_equal_shape_different_coordinates_fail_exact_comparison() -> None:
    """Equal array shapes do not establish an identical physical grid."""
    first = grids.spatial_grid_identity(_coordinates(), coordinate_units=("m", "m"), field_order=("p", "u"))
    other = grids.spatial_grid_identity(_coordinates(0.1), coordinate_units=("m", "m"), field_order=("p", "u"))
    with pytest.raises(ValueError, match="coordinate_digest"):
        grids.require_common_grid(first, other)


def test_display_stride_preserves_endpoints_and_aligned_fields() -> None:
    """Display samples retain endpoints and apply one index map to every field."""
    indices = grids.display_indices((5, 6), 2)
    assert indices[0].tolist() == [0, 2, 4]
    assert indices[1].tolist() == [0, 2, 4, 5]
    full = np.arange(30).reshape(5, 6)
    sampled = grids.display_sample(full, indices)
    assert sampled[-1, -1] == full[-1, -1]
    assert sampled.shape == (3, 4)
    np.testing.assert_array_equal(grids.display_sample(full + 1, indices) - sampled, 1)
    masked = np.ma.array(full, mask=full % 2 == 0)
    sampled_mask = grids.display_sample(masked, indices)
    assert np.ma.isMaskedArray(sampled_mask)
    np.testing.assert_array_equal(np.ma.getmaskarray(sampled_mask), np.ma.getmaskarray(masked)[indices[0][:, None], indices[1]])


def test_same_coordinate_values_with_different_index_map_fail() -> None:
    """Artifact source-index semantics remain part of comparison admission."""
    first = grids.spatial_grid_identity(_coordinates(), coordinate_units=("m", "m"), field_order=("p",))
    alternate = grids.spatial_grid_identity(
        _coordinates(),
        coordinate_units=("m", "m"),
        field_order=("p",),
        source_shape=(5, 11),
        artifact_stride=2,
        axis_indices=(tuple(range(5)), (0, 2, 4, 6, 8, 10)),
    )
    with pytest.raises(ValueError, match="source_shape"):
        grids.require_common_grid(first, alternate)


def test_transient_model_comparison_checks_every_case_grid_identity() -> None:
    """Manifest identities guard all paired cases without opening tensor payloads."""

    def frame(case_grid: str) -> pd.DataFrame:
        result = pd.DataFrame()
        records = tuple(
            SimpleNamespace(
                dataset_role="id",
                case_id=case,
                identity={"spatial_representation": {"grid_identity_sha256": digest}},
            )
            for case, digest in (("case-1", "first"), ("case-2", case_grid))
        )
        result.attrs["artifact_kind"] = "transient_sequence"
        result.attrs["transient_scaling_state"] = {}
        result.attrs["transient_unavailable_horizons"] = ()
        result.attrs["transient_sequence_records"] = records
        return result

    evaluation = object.__new__(transient_session.TransientEvaluationSession)
    evaluation._frames = MappingProxyType({"A": (frame("same"), None), "B": (frame("different"), None)})
    with pytest.raises(ValueError, match="case-2"):
        evaluation.require_common_comparison_grids()
    evaluation._frames = MappingProxyType({"A": (frame("same"), None), "B": (frame("same"), None)})
    evaluation.require_common_comparison_grids()
