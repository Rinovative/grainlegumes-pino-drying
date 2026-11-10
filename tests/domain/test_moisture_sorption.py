"""Exact modified-Oswin moisture-equilibrium and granular-water contracts.

Responsibilities:
  - Check modified-Oswin forward/inverse round trips and explicit clipping
  - Check that granular water rejects unphysical states except for model-output evaluation

Design principles:
  - Equilibrium values are compared numerically and invalid clipping contracts fail

This module does NOT:
  - Fit material sorption coefficients or run drying solves
"""

from __future__ import annotations

import numpy as np
import pytest

from src import domain


def test_modified_oswin_forward_inverse_round_trip() -> None:
    """Recover the original RH field through the maintained exact inverse."""
    relative_humidity = np.asarray([[0.2, 0.45], [0.7, 0.9]], dtype=np.float64)
    temperature = np.asarray([[293.15, 298.15], [303.15, 308.15]], dtype=np.float64)
    moisture = domain.moisture.oswin_equilibrium_dry_basis_moisture(
        relative_humidity,
        temperature,
        a_osw=12.06202053,
        b_osw=-0.0573838,
        c_osw=0.34338283,
    )
    reconstructed = domain.moisture.oswin_equilibrium_relative_humidity(
        moisture,
        temperature,
        a_osw=12.06202053,
        b_osw=-0.0573838,
        c_osw=0.34338283,
    )

    np.testing.assert_allclose(reconstructed, relative_humidity, rtol=2.0e-15, atol=2.0e-15)


def test_modified_oswin_clipping_is_explicit() -> None:
    """Apply solver clipping only when the caller supplies its exact bounds."""
    clipped = domain.moisture.oswin_equilibrium_dry_basis_moisture(
        np.asarray([0.0, 1.0]),
        298.15,
        a_osw=12.0,
        b_osw=-0.05,
        c_osw=0.4,
        clip_bounds=(1.0e-6, 0.999),
    )
    np.testing.assert_array_equal(np.isfinite(clipped), np.ones(clipped.shape, dtype=bool))
    with pytest.raises(ValueError, match="below 1"):
        domain.moisture.oswin_equilibrium_dry_basis_moisture(
            1.0,
            298.15,
            a_osw=12.0,
            b_osw=-0.05,
            c_osw=0.4,
        )


def test_granular_water_content_rejects_unphysical_states_unless_evaluating_model_output() -> None:
    """Physical callers reject negative water; model-output evaluation applies the same weighting and keeps NaN."""
    surface = np.asarray([0.2, -0.1, np.nan])
    internal = np.asarray([0.4, -0.1, 0.4])
    with pytest.raises(ValueError, match="non-negative"):
        domain.moisture.granular_water_content(surface, internal, 0.25)
    weighted = domain.moisture.granular_water_content(surface, internal, 0.25, require_physical_states=False)
    np.testing.assert_allclose(weighted[:2], [0.35, -0.1])
    assert np.isnan(weighted[2])
    with pytest.raises(ValueError, match="f_surf"):
        domain.moisture.granular_water_content(surface, internal, 1.0, require_physical_states=False)
