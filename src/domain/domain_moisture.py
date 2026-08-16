"""
===============================================================================
domain_moisture.py
===============================================================================
Define canonical granular-phase moisture and equilibrium-sorption conversions.
Responsibilities:
  - Convert dry-basis and wet-basis moisture without duplicating formulas
  - Evaluate and invert the maintained modified-Oswin equilibrium relation
  - Derive local moisture from dry-solid and water mass densities
  - Derive bulk wet-basis moisture from integrated dry and water mass
Design principles:
  - Moisture basis is explicit in every public name
  - Forward and inverse sorption use one exact temperature convention
  - Invalid mass and moisture domains fail before division
  - Bulk moisture is mass weighted, never an unqualified spatial mean
This module does NOT:
  - Cache derived moisture fields or define dataset normalization
  - Own material coefficients, humidity clipping policy, or spatial quadrature rules
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


def _oswin_temperature_factor(
    temperature: Any,
    *,
    a_osw: float,
    b_osw: float,
    c_osw: float,
) -> tuple[np.ndarray, float]:
    """Return the positive modified-Oswin temperature factor and exponent."""
    values = np.asarray(temperature, dtype=np.float64)
    coefficients = np.asarray((a_osw, b_osw, c_osw), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0) or not np.isfinite(coefficients).all() or c_osw <= 0.0:
        message = "Modified-Oswin inputs require positive finite temperature and C_osw, with finite A_osw and B_osw."
        raise ValueError(message)
    factor = a_osw + b_osw * (values - 273.15)
    if not np.isfinite(factor).all() or np.any(factor <= 0.0):
        message = "Modified-Oswin temperature factor A_osw+B_osw*(T-273.15 K) must remain positive and finite."
        raise ValueError(message)
    return factor, float(c_osw)


def oswin_equilibrium_dry_basis_moisture(
    relative_humidity: Any,
    temperature: Any,
    *,
    a_osw: float,
    b_osw: float,
    c_osw: float,
    clip_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Evaluate the maintained modified-Oswin equilibrium moisture relation.

    Parameters
    ----------
    relative_humidity : Any
        Relative humidity on the unit interval.
    temperature : Any
        Broadcast-compatible absolute temperature in kelvin.
    a_osw, b_osw, c_osw : float
        Maintained modified-Oswin coefficients with units 1, 1/K, and 1.
    clip_bounds : tuple[float, float] | None, optional
        Explicit numerical relative-humidity bounds applied before evaluation.
        Omit them for the unclipped physical relation.

    Returns
    -------
    numpy.ndarray
        Equilibrium dry-basis moisture in kg/kg.

    Raises
    ------
    ValueError
        If inputs are non-finite, incompatible, outside their physical domain,
        or produce a non-finite equilibrium state.

    """
    humidity = np.asarray(relative_humidity, dtype=np.float64)
    factor, exponent = _oswin_temperature_factor(
        temperature,
        a_osw=a_osw,
        b_osw=b_osw,
        c_osw=c_osw,
    )
    try:
        humidity, factor = np.broadcast_arrays(humidity, factor)
    except ValueError as error:
        message = "Relative humidity and temperature must be broadcast-compatible for modified-Oswin evaluation."
        raise ValueError(message) from error
    if not np.isfinite(humidity).all() or np.any((humidity < 0.0) | (humidity > 1.0)):
        message = "Relative humidity must contain finite values in [0, 1] for modified-Oswin evaluation."
        raise ValueError(message)
    effective = humidity
    if clip_bounds is not None:
        lower, upper = clip_bounds
        if not np.isfinite((lower, upper)).all() or not 0.0 < lower < upper < 1.0:
            message = "Modified-Oswin clip bounds must be finite and strictly ordered inside (0, 1)."
            raise ValueError(message)
        effective = np.clip(humidity, lower, upper)
    elif np.any(humidity >= 1.0):
        message = "Unclipped modified-Oswin evaluation requires relative humidity below 1."
        raise ValueError(message)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = 0.01 * factor * np.power(effective / (1.0 - effective), exponent)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        message = "Modified-Oswin equilibrium evaluation produced invalid dry-basis moisture."
        raise ValueError(message)
    return result


def oswin_equilibrium_relative_humidity(
    dry_basis_moisture: Any,
    temperature: Any,
    *,
    a_osw: float,
    b_osw: float,
    c_osw: float,
) -> np.ndarray:
    """
    Invert the maintained modified-Oswin relation for relative humidity.

    Parameters
    ----------
    dry_basis_moisture : Any
        Non-negative dry-basis moisture in kg/kg.
    temperature : Any
        Broadcast-compatible absolute temperature in kelvin.
    a_osw, b_osw, c_osw : float
        Maintained modified-Oswin coefficients with units 1, 1/K, and 1.

    Returns
    -------
    numpy.ndarray
        Exact equilibrium relative humidity on the half-open interval [0, 1).

    Raises
    ------
    ValueError
        If inputs are non-finite, incompatible, outside their physical domain,
        or produce an invalid equilibrium state.

    Notes
    -----
    This is the exact algebraic inverse used by the configured initial-state
    expression; no numerical humidity clipping is applied.

    """
    moisture = np.asarray(dry_basis_moisture, dtype=np.float64)
    factor, exponent = _oswin_temperature_factor(
        temperature,
        a_osw=a_osw,
        b_osw=b_osw,
        c_osw=c_osw,
    )
    try:
        moisture, factor = np.broadcast_arrays(moisture, factor)
    except ValueError as error:
        message = "Dry-basis moisture and temperature must be broadcast-compatible for modified-Oswin inversion."
        raise ValueError(message) from error
    if not np.isfinite(moisture).all() or np.any(moisture < 0.0):
        message = "Dry-basis moisture must contain finite non-negative values for modified-Oswin inversion."
        raise ValueError(message)
    scaled = 100.0 * moisture / factor
    log_ratio = np.full(scaled.shape, -np.inf, dtype=np.float64)
    positive = scaled > 0.0
    log_ratio[positive] = np.log(scaled[positive]) / exponent
    result = np.empty(log_ratio.shape, dtype=np.float64)
    nonnegative = log_ratio >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-log_ratio[nonnegative]))
    exp_ratio = np.exp(log_ratio[~nonnegative])
    result[~nonnegative] = exp_ratio / (1.0 + exp_ratio)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result >= 1.0)):
        message = "Modified-Oswin inversion produced relative humidity outside [0, 1)."
        raise ValueError(message)
    return result


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
