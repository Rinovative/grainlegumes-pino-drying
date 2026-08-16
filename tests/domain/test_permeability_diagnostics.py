"""Exact symmetric permeability-tensor diagnostic contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src import domain


def test_symmetric_tensor_diagnostics_recovers_principal_values() -> None:
    """Recover known eigenvalues from a 45-degree rotated tensor."""
    diagnostics = domain.permeability.symmetric_tensor_diagnostics(
        np.full((2, 3), 2.5),
        np.full((2, 3), 1.5),
        np.full((2, 3), 2.5),
    )

    np.testing.assert_allclose(diagnostics.minimum_principal, 1.0)
    np.testing.assert_allclose(diagnostics.maximum_principal, 4.0)
    np.testing.assert_allclose(diagnostics.anisotropy_ratio, 4.0)
    np.testing.assert_allclose(diagnostics.determinant, 4.0)


def test_symmetric_tensor_diagnostics_rejects_non_positive_definite_values() -> None:
    """Reject a tensor at the canonical positive-definiteness boundary."""
    with pytest.raises(ValueError, match="positive definite"):
        domain.permeability.symmetric_tensor_diagnostics(1.0, 1.0, 1.0)
