"""
===============================================================================
domain_moisture.py
===============================================================================
Define the canonical granular-phase moisture-basis conversions.
Responsibilities:
  - Convert dry-basis and wet-basis moisture without duplicating formulas
  - Derive local moisture from dry-solid and water mass densities
  - Derive bulk wet-basis moisture from integrated dry and water mass
Design principles:
  - Moisture basis is explicit in every public name
  - Invalid mass and moisture domains fail before division
  - Bulk moisture is mass weighted, never an unqualified spatial mean
This module does NOT:
  - Cache derived moisture fields or define dataset normalization
  - Own material ranges, solver expressions, or spatial quadrature rules
===============================================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dry_basis_to_wet_basis(X_db: Any) -> np.ndarray:
    """Convert dry-basis moisture ``X_db`` to wet-basis moisture ``X_wb``."""
    dry_basis = np.asarray(X_db)
    if not np.isfinite(dry_basis).all() or np.any(dry_basis < 0):
        message = "X_db must contain finite non-negative values."
        raise ValueError(message)
    return dry_basis / (1.0 + dry_basis)


def wet_basis_to_dry_basis(X_wb: Any) -> np.ndarray:
    """Convert wet-basis moisture ``X_wb`` to dry-basis moisture ``X_db``."""
    wet_basis = np.asarray(X_wb)
    if not np.isfinite(wet_basis).all() or np.any((wet_basis < 0) | (wet_basis >= 1)):
        message = "X_wb must contain finite values in [0, 1)."
        raise ValueError(message)
    return wet_basis / (1.0 - wet_basis)


def granular_water_content(w_surf: Any, w_int: Any, f_surf: Any) -> np.ndarray:
    """Return weighted granular water f_surf*w_surf + (1-f_surf)*w_int."""
    surface = np.asarray(w_surf)
    internal = np.asarray(w_int)
    fraction = np.asarray(f_surf)
    try:
        surface, internal, fraction = np.broadcast_arrays(surface, internal, fraction)
    except ValueError as error:
        message = "w_surf, w_int, and f_surf must be broadcast-compatible."
        raise ValueError(message) from error
    if (
        not np.isfinite(surface).all()
        or not np.isfinite(internal).all()
        or not np.isfinite(fraction).all()
        or np.any(surface < 0)
        or np.any(internal < 0)
        or np.any((fraction <= 0) | (fraction >= 1))
    ):
        message = "Water states must be finite and non-negative and f_surf must lie strictly inside (0, 1)."
        raise ValueError(message)
    return fraction * surface + (1.0 - fraction) * internal


def dry_basis_moisture(w_gr: Any, rho_bu_dry: Any) -> np.ndarray:
    """Return ``X_db = w_gr / rho_bu_dry`` from local mass densities."""
    water = np.asarray(w_gr)
    dry_mass = np.asarray(rho_bu_dry)
    if water.shape != dry_mass.shape:
        message = "w_gr and rho_bu_dry must have identical shapes."
        raise ValueError(message)
    if not np.isfinite(water).all() or np.any(water < 0) or not np.isfinite(dry_mass).all() or np.any(dry_mass <= 0):
        message = "w_gr must be finite and non-negative and rho_bu_dry must be finite and positive."
        raise ValueError(message)
    return water / dry_mass


def wet_basis_moisture(w_gr: Any, rho_bu_dry: Any) -> np.ndarray:
    """Return ``X_wb = w_gr / (rho_bu_dry + w_gr)`` from mass densities."""
    dry_basis = dry_basis_moisture(w_gr, rho_bu_dry)
    return dry_basis_to_wet_basis(dry_basis)


def bulk_wet_basis_moisture(w_gr: Any, rho_bu_dry: Any, *, cell_weights: Any | None = None) -> float:
    """Return ``X_wb_bulk`` from integrated water and dry-solid masses."""
    water = np.asarray(w_gr, dtype=np.float64)
    dry_mass = np.asarray(rho_bu_dry, dtype=np.float64)
    dry_basis_moisture(water, dry_mass)
    if cell_weights is None:
        weights = np.ones_like(water, dtype=np.float64)
    else:
        weights = np.asarray(cell_weights, dtype=np.float64)
        if weights.shape != water.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            message = "cell_weights must match the field shape and be finite and non-negative."
            raise ValueError(message)
    integrated_water = float(np.sum(water * weights, dtype=np.float64))
    integrated_dry = float(np.sum(dry_mass * weights, dtype=np.float64))
    if integrated_dry <= 0 or integrated_dry + integrated_water <= 0:
        message = "Integrated dry and total masses must be positive."
        raise ValueError(message)
    return integrated_water / (integrated_dry + integrated_water)
