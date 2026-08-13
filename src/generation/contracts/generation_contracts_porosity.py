"""
===============================================================================
generation_contracts_porosity.py
===============================================================================
Resolve fixed material Kozeny-Carman calibration and packing-scatter supports.
Responsibilities:
  - Derive and validate fixed material Kozeny-Carman reference coefficients
  - Resolve joint permeability/porosity supports before case sampling
  - Invert the monotonic global permeability-to-porosity trend robustly
Design principles:
  - Material calibration remains fixed at its canonical reference record
  - Authored supports remain visible beside resolved effective supports
  - Local porosity morphology remains outside this scalar coupling owner
This module does NOT:
  - Sample DOE coordinates, generate local fields, or author material values
  - Impose a pointwise permeability-to-porosity relation
===============================================================================
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any, Final

PACKING_SCATTER_TRUNCATION_LOWER: Final = -3.0
PACKING_SCATTER_TRUNCATION_UPPER: Final = 3.0
_BISECTION_ITERATIONS: Final = 96
_SUPPORT_TOLERANCE: Final = 1e-14
_STANDARD_NORMAL: Final = NormalDist()
_TRUNCATION_CDF_LOWER: Final = _STANDARD_NORMAL.cdf(PACKING_SCATTER_TRUNCATION_LOWER)
_TRUNCATION_CDF_UPPER: Final = _STANDARD_NORMAL.cdf(PACKING_SCATTER_TRUNCATION_UPPER)


def _finite(value: Any, *, label: str) -> float:
    """Return one finite non-boolean scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        msg = f"{label} must be one finite scalar."
        raise ValueError(msg)
    return float(value)


def _interval(value: Mapping[str, Any], *, label: str) -> tuple[float, float]:
    """Return one finite strictly ordered scalar interval."""
    lower = _finite(value.get("lower"), label=f"{label} lower bound")
    upper = _finite(value.get("upper"), label=f"{label} upper bound")
    if not lower < upper:
        msg = f"{label} must be strictly ordered."
        raise ValueError(msg)
    return lower, upper


def kozeny_carman_response(epsilon: float) -> float:
    """Return ``epsilon**3 / (1-epsilon)**2`` in its physical domain."""
    value = _finite(epsilon, label="Kozeny-Carman porosity")
    if not 0.0 < value < 1.0:
        msg = "Kozeny-Carman porosity must lie strictly inside (0, 1)."
        raise ValueError(msg)
    return value**3 / (1.0 - value) ** 2


def derive_reference_coefficient(material_kappa_nominal: float, eps_bed_cal_ref: float) -> float:
    """Derive ``A_KC_reference = kappa_nominal / g(eps_bed_cal_ref)``."""
    permeability = _finite(material_kappa_nominal, label="Material nominal permeability")
    if permeability <= 0.0:
        msg = "Material nominal permeability must be strictly positive."
        raise ValueError(msg)
    coefficient = permeability / kozeny_carman_response(eps_bed_cal_ref)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        msg = "Derived Kozeny-Carman reference coefficient must be finite and positive."
        raise ValueError(msg)
    return coefficient


def _solve_reference_porosity_unchecked(sampled_kappa_mean: float, reference_coefficient: float) -> float:
    """Invert one positive Kozeny-Carman target over its complete physical domain."""
    permeability = _finite(sampled_kappa_mean, label="Sampled mean permeability")
    coefficient = _finite(reference_coefficient, label="Kozeny-Carman reference coefficient")
    if permeability <= 0.0 or coefficient <= 0.0:
        msg = "Kozeny-Carman permeability and coefficient must be strictly positive."
        raise ValueError(msg)
    target = permeability / coefficient
    left, right = 0.0, 1.0
    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (left + right)
        if kozeny_carman_response(middle) > target:
            right = middle
        else:
            left = middle
    result = 0.5 * (left + right)
    if not 0.0 < result < 1.0 or not math.isfinite(result):
        msg = "Kozeny-Carman bisection did not return a finite physical solution."
        raise RuntimeError(msg)
    return result


def solve_reference_porosity(
    sampled_kappa_mean: float,
    reference_coefficient: float,
    *,
    eps_min_global: float,
    eps_max_global: float,
) -> float:
    """Invert one Kozeny-Carman target and enforce configured global guards."""
    lower = _finite(eps_min_global, label="Global porosity lower guard")
    upper = _finite(eps_max_global, label="Global porosity upper guard")
    if not 0.0 < lower < upper < 1.0:
        msg = "Global porosity guards must be ordered strictly inside (0, 1)."
        raise ValueError(msg)
    result = _solve_reference_porosity_unchecked(sampled_kappa_mean, reference_coefficient)
    if math.isclose(result, lower, rel_tol=0.0, abs_tol=_SUPPORT_TOLERANCE):
        return lower
    if math.isclose(result, upper, rel_tol=0.0, abs_tol=_SUPPORT_TOLERANCE):
        return upper
    if not lower < result < upper:
        msg = f"Kozeny-Carman solution {result} lies outside global guards [{lower}, {upper}]."
        raise ValueError(msg)
    return result


def truncated_standard_normal_quantile(unit_coordinate: float) -> float:
    """Map one unit coordinate to the open standard-normal interval ``(-3, 3)``."""
    coordinate = _finite(unit_coordinate, label="Packing-scatter unit coordinate")
    if not 0.0 <= coordinate <= 1.0:
        message = "Packing-scatter unit coordinate must lie in [0, 1]."
        raise ValueError(message)
    probability = _TRUNCATION_CDF_LOWER + coordinate * (_TRUNCATION_CDF_UPPER - _TRUNCATION_CDF_LOWER)
    probability = min(
        max(probability, math.nextafter(_TRUNCATION_CDF_LOWER, _TRUNCATION_CDF_UPPER)),
        math.nextafter(_TRUNCATION_CDF_UPPER, _TRUNCATION_CDF_LOWER),
    )
    result = _STANDARD_NORMAL.inv_cdf(probability)
    if not PACKING_SCATTER_TRUNCATION_LOWER < result < PACKING_SCATTER_TRUNCATION_UPPER:
        message = "Packing scatter quantile must lie strictly inside (-3, 3)."
        raise RuntimeError(message)
    return result


def draw_truncated_standard_normal(seed: int) -> float:
    """Draw one deterministic truncated-normal deviate from exactly one uniform draw."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        message = "Packing-scatter seed must be an integer."
        raise TypeError(message)
    unit_coordinate = random.Random(seed).random()  # noqa: S311 - deterministic scientific stream
    return truncated_standard_normal_quantile(unit_coordinate)


def resolve_porosity_coupling(
    *,
    material_family: str,
    material_kappa_nominal: float,
    eps_bed_cal_ref: float,
    authored_permeability_support: Mapping[str, Any],
    packing_porosity_mean_support: Mapping[str, Any],
    eps_min_global: float,
    eps_max_global: float,
    authored_kappa_ood: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the fixed calibration and KC-compatible authored permeability supports."""
    if not isinstance(material_family, str) or not material_family:
        msg = "Material family must be non-empty text."
        raise ValueError(msg)
    authored_lower, authored_upper = _interval(authored_permeability_support, label=f"Material {material_family} authored permeability support")
    natural_lower, natural_upper = _interval(packing_porosity_mean_support, label="Natural packing-porosity support")
    global_lower = _finite(eps_min_global, label="Global porosity lower guard")
    global_upper = _finite(eps_max_global, label="Global porosity upper guard")
    if not 0.0 < global_lower < natural_lower < natural_upper < global_upper < 1.0:
        msg = "Natural packing-porosity support must lie strictly inside global porosity guards."
        raise ValueError(msg)
    nominal = _finite(material_kappa_nominal, label="Material nominal permeability")
    calibration = _finite(eps_bed_cal_ref, label="Material calibration porosity")
    coefficient = derive_reference_coefficient(nominal, calibration)
    kc_lower = coefficient * kozeny_carman_response(natural_lower)
    kc_upper = coefficient * kozeny_carman_response(natural_upper)
    effective_lower, effective_upper = max(authored_lower, kc_lower), min(authored_upper, kc_upper)
    if not all(math.isfinite(item) for item in (kc_lower, kc_upper, effective_lower, effective_upper)) or not effective_lower < effective_upper:
        msg = (
            f"Material {material_family} authored permeability support [{authored_lower}, {authored_upper}] "
            f"has empty KC-compatible intersection [{kc_lower}, {kc_upper}]."
        )
        raise ValueError(msg)
    if not effective_lower <= nominal <= effective_upper:
        msg = (
            f"Material {material_family} nominal permeability {nominal} lies outside effective joint support [{effective_lower}, {effective_upper}]."
        )
        raise ValueError(msg)
    nominal_identity = solve_reference_porosity(nominal, coefficient, eps_min_global=global_lower, eps_max_global=global_upper)
    if not math.isclose(nominal_identity, calibration, rel_tol=0.0, abs_tol=_SUPPORT_TOLERANCE):
        msg = "Material nominal Kozeny-Carman identity does not recover calibration porosity."
        raise RuntimeError(msg)
    tails: dict[str, dict[str, float]] = {}
    for item in authored_kappa_ood or []:
        lower, upper = _interval(item, label="Authored permeability OOD support")
        direction = "lower" if upper < authored_lower else "upper" if lower > authored_upper else None
        if direction is None or direction in tails:
            msg = (
                f"Material {material_family} authored permeability OOD interval [{lower}, {upper}] must be "
                f"one unique lower or upper interval separated from authored ID support "
                f"[{authored_lower}, {authored_upper}]."
            )
            raise ValueError(msg)
        mapped_lower = _solve_reference_porosity_unchecked(lower, coefficient)
        mapped_upper = _solve_reference_porosity_unchecked(upper, coefficient)
        if mapped_lower < global_lower or mapped_upper > global_upper:
            msg = (
                f"Material {material_family} {direction} permeability OOD interval [{lower}, {upper}] maps to "
                f"porosity [{mapped_lower}, {mapped_upper}] outside global guards [{global_lower}, {global_upper}]; "
                f"natural porosity support is [{natural_lower}, {natural_upper}]."
            )
            raise ValueError(msg)
        if direction == "lower" and not mapped_upper < natural_lower:
            msg = (
                f"Material {material_family} lower permeability OOD interval [{lower}, {upper}] maps to "
                f"porosity [{mapped_lower}, {mapped_upper}], not below natural support "
                f"[{natural_lower}, {natural_upper}] within global guards [{global_lower}, {global_upper}]."
            )
            raise ValueError(msg)
        if direction == "upper" and not mapped_lower > natural_upper:
            msg = (
                f"Material {material_family} upper permeability OOD interval [{lower}, {upper}] maps to "
                f"porosity [{mapped_lower}, {mapped_upper}], not above natural support "
                f"[{natural_lower}, {natural_upper}] within global guards [{global_lower}, {global_upper}]."
            )
            raise ValueError(msg)
        tails[direction] = {
            "kappa_lower": lower,
            "kappa_upper": upper,
            "porosity_lower": mapped_lower,
            "porosity_upper": mapped_upper,
        }
    return {
        "material_family": material_family,
        "material_kappa_nominal": nominal,
        "material_eps_bed_cal_ref": calibration,
        "A_KC_reference": coefficient,
        "authored_permeability_support": {"lower": authored_lower, "upper": authored_upper},
        "natural_porosity_support": {"lower": natural_lower, "upper": natural_upper},
        "kc_compatible_permeability_support": {"lower": kc_lower, "upper": kc_upper},
        "effective_joint_permeability_support": {"lower": effective_lower, "upper": effective_upper},
        "authored_support_narrowed": effective_lower > authored_lower or effective_upper < authored_upper,
        "eps_kc_trend_interval": {
            "lower": solve_reference_porosity(effective_lower, coefficient, eps_min_global=global_lower, eps_max_global=global_upper),
            "upper": solve_reference_porosity(effective_upper, coefficient, eps_min_global=global_lower, eps_max_global=global_upper),
        },
        "kappa_ood_porosity_supports": tails,
    }


def resolve_active_packing_support_for_value(
    coupling: Mapping[str, Any], *, sampled_kappa_mean: float, active_ood_unit: str | None
) -> tuple[str, float, float]:
    """Resolve the active scatter support from the case OOD unit and sampled permeability."""
    if active_ood_unit != "kappa_mean":
        natural_lower, natural_upper = _interval(coupling["natural_porosity_support"], label="Natural porosity support")
        return "natural", natural_lower, natural_upper
    value = _finite(sampled_kappa_mean, label="Sampled mean permeability")
    tails = coupling.get("kappa_ood_porosity_supports")
    if not isinstance(tails, Mapping):
        msg = "kappa_mean OOD has no mapped authored support."
        raise TypeError(msg)
    for direction in ("lower", "upper"):
        tail = tails.get(direction)
        if isinstance(tail, Mapping) and float(tail["kappa_lower"]) <= value <= float(tail["kappa_upper"]):
            return f"kappa_mean_ood_{direction}", float(tail["porosity_lower"]), float(tail["porosity_upper"])
    msg = f"Sampled kappa_mean {value} is outside every authored kappa_mean OOD support."
    raise ValueError(msg)
