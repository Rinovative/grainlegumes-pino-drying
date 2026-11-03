"""Define permeability tensor ordering and diagnostics.

Responsibilities:
  - Declare internal permeability component names and their canonical order
  - Derive exact principal values and anisotropy for symmetric 2D tensors

Design principles:
  - Permeability contracts are declarative and deterministic
  - Numerical diagnostics fail closed on non-positive-definite tensors

This module does NOT:
  - Normalize permeability for learned-model storage
  - Choose learned model input fields or task channel order
  - Evaluate permeability-dependent physics residuals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# =============================================================================
# Internal canonical representation primitives (symmetric, reduced).
# A TaskSpec decides which components become learned channels.
# 2D follows the steady-flow semantic order (xx, xy, yy).
# 3D groups components in upper-triangular row order.
# =============================================================================

INTERNAL_KAPPA_2D_ORDER: list[str] = [
    "Kxx",
    "Kxy",
    "Kyy",
]

INTERNAL_KAPPA_3D_ORDER: list[str] = [
    "Kxx",
    "Kxy",
    "Kxz",
    "Kyy",
    "Kyz",
    "Kzz",
]

# =============================================================================
# Helpers
# =============================================================================


@dataclass(frozen=True, slots=True)
class PermeabilityTensorDiagnostics:
    """
    Hold exact pointwise diagnostics for a symmetric positive-definite tensor.

    Attributes
    ----------
    minimum_principal, maximum_principal : numpy.ndarray
        Ordered principal permeabilities in square metres.
    anisotropy_ratio : numpy.ndarray
        Dimensionless maximum-to-minimum principal permeability ratio.
    determinant : numpy.ndarray
        Pointwise determinant in square metres to the fourth power.

    """

    minimum_principal: np.ndarray
    maximum_principal: np.ndarray
    anisotropy_ratio: np.ndarray
    determinant: np.ndarray


def symmetric_tensor_diagnostics(Kxx: Any, Kxy: Any, Kyy: Any) -> PermeabilityTensorDiagnostics:
    """
    Derive principal permeability and anisotropy for a symmetric 2D tensor.

    Parameters
    ----------
    Kxx, Kxy, Kyy : Any
        Identically shaped physical permeability components in square metres.

    Returns
    -------
    PermeabilityTensorDiagnostics
        Minimum and maximum principal permeability, their ratio, and determinant.

    Raises
    ------
    ValueError
        If components differ in shape, contain non-finite values, or do not form
        a pointwise positive-definite tensor.

    """
    xx = np.asarray(Kxx, dtype=np.float64)
    xy = np.asarray(Kxy, dtype=np.float64)
    yy = np.asarray(Kyy, dtype=np.float64)
    if xx.shape != xy.shape or xx.shape != yy.shape:
        message = "Symmetric permeability components must have identical shapes."
        raise ValueError(message)
    if not np.isfinite(np.stack((xx, xy, yy))).all():
        message = "Symmetric permeability components must contain only finite values."
        raise ValueError(message)
    determinant = xx * yy - xy**2
    if np.any(xx <= 0.0) or np.any(yy <= 0.0) or np.any(determinant <= 0.0):
        message = "Symmetric permeability tensors must be pointwise positive definite."
        raise ValueError(message)
    midpoint = 0.5 * (xx + yy)
    radius = np.hypot(0.5 * (xx - yy), xy)
    minimum = midpoint - radius
    maximum = midpoint + radius
    if np.any(minimum <= 0.0) or not np.isfinite(np.stack((minimum, maximum))).all():
        message = "Symmetric permeability principal values are invalid."
        raise ValueError(message)
    anisotropy = maximum / minimum
    if not np.isfinite(anisotropy).all() or np.any(anisotropy < 1.0):
        message = "Symmetric permeability anisotropy is invalid."
        raise ValueError(message)
    return PermeabilityTensorDiagnostics(
        minimum_principal=minimum,
        maximum_principal=maximum,
        anisotropy_ratio=anisotropy,
        determinant=determinant,
    )
