"""
===============================================================================
generation_porosity.py
===============================================================================
Resolve the material-calibrated global porosity coupling contract.
Responsibilities:
  - Derive the Kozeny-Carman reference coefficient from configured material data
  - Resolve conditional natural and transformed parameter-OOD factor supports
  - Invert the global coupling inside universal physical porosity guards
Design principles:
  - Material measurements remain in their existing permeability and density owners
  - Conditional supports are deterministic functions of resolved case science
  - Local porosity morphology remains outside this scalar coupling owner
This module does NOT:
  - Generate local fields, sample DOE coordinates, or author material values
  - Derive porosity pointwise from a realized permeability field
===============================================================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOWER_TAIL_OUTER_OFFSET: Final = 0.40
_LOWER_TAIL_INNER_OFFSET: Final = 0.15
_UPPER_TAIL_INNER_OFFSET: Final = 0.15
_UPPER_TAIL_OUTER_OFFSET: Final = 0.40
_BISECTION_ITERATIONS: Final = 96
ANCHOR_PARAMETER_NAME: Final = "porosity.kc_anchor_factor"


def _finite(value: Any, *, label: str) -> float:
    """Return one finite non-boolean scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be one finite scalar."
        raise ValueError(message)
    return float(value)


def kozeny_carman_response(epsilon: float) -> float:
    """Return ``epsilon**3 / (1-epsilon)**2`` inside its physical domain."""
    value = _finite(epsilon, label="Kozeny-Carman porosity")
    if not 0.0 < value < 1.0:
        message = "Kozeny-Carman porosity must lie strictly inside (0, 1)."
        raise ValueError(message)
    return value**3 / (1.0 - value) ** 2


def derive_reference_coefficient(
    material_kappa_nominal: float,
    eps_bed_cal_ref: float,
) -> float:
    """Derive ``A_KC_ref = kappa_nominal / g(eps_bed_cal_ref)``."""
    permeability = _finite(material_kappa_nominal, label="Material nominal permeability")
    if permeability <= 0.0:
        message = "Material nominal permeability must be strictly positive."
        raise ValueError(message)
    coefficient = permeability / kozeny_carman_response(eps_bed_cal_ref)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        message = "Derived Kozeny-Carman reference coefficient must be finite and positive."
        raise ValueError(message)
    return coefficient


def solve_reference_porosity(
    sampled_kappa_mean: float,
    reference_coefficient: float,
    kc_anchor_factor: float,
    *,
    eps_min_global: float,
    eps_max_global: float,
) -> float:
    """
    Invert the material-calibrated Kozeny-Carman relation by bisection.

    The solved relation is
    ``sampled_kappa_mean = kc_anchor_factor * reference_coefficient * g(eps)``.
    Both physical guards are open bounds for the scalar reference level.
    """
    permeability = _finite(sampled_kappa_mean, label="Sampled mean permeability")
    coefficient = _finite(reference_coefficient, label="Kozeny-Carman reference coefficient")
    factor = _finite(kc_anchor_factor, label="Kozeny-Carman anchor factor")
    lower = _finite(eps_min_global, label="Global porosity lower guard")
    upper = _finite(eps_max_global, label="Global porosity upper guard")
    if permeability <= 0.0 or coefficient <= 0.0 or factor <= 0.0:
        message = "Kozeny-Carman permeability, coefficient, and factor must be strictly positive."
        raise ValueError(message)
    if not 0.0 < lower < upper < 1.0:
        message = "Global porosity guards must be ordered strictly inside (0, 1)."
        raise ValueError(message)
    target = permeability / (factor * coefficient)
    response_lower = kozeny_carman_response(lower)
    response_upper = kozeny_carman_response(upper)
    if not response_lower < target < response_upper:
        message = (
            "Kozeny-Carman reference porosity has no solution strictly inside "
            f"the global guards: target={target}, response_bounds=[{response_lower}, {response_upper}]."
        )
        raise ValueError(message)
    left = lower
    right = upper
    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (left + right)
        if kozeny_carman_response(middle) > target:
            right = middle
        else:
            left = middle
    result = 0.5 * (left + right)
    if not lower < result < upper or not math.isfinite(result):
        message = "Kozeny-Carman bisection did not return a finite interior solution."
        raise RuntimeError(message)
    return result


def _tail_record(
    direction: str,
    lower_log: float,
    upper_log: float,
    *,
    id_lower_log: float,
    id_upper_log: float,
    sampled_kappa_mean: float,
    reference_coefficient: float,
    eps_min_global: float,
    eps_max_global: float,
) -> dict[str, Any]:
    """Return one transformed tail and its physical feasibility evidence."""
    lower = math.exp(lower_log)
    upper = math.exp(upper_log)
    id_width = id_upper_log - id_lower_log
    reference_at_lower = solve_reference_porosity(
        sampled_kappa_mean,
        reference_coefficient,
        lower,
        eps_min_global=eps_min_global,
        eps_max_global=eps_max_global,
    )
    reference_at_upper = solve_reference_porosity(
        sampled_kappa_mean,
        reference_coefficient,
        upper,
        eps_min_global=eps_min_global,
        eps_max_global=eps_max_global,
    )
    gap = id_lower_log - upper_log if direction == "lower" else lower_log - id_upper_log
    return {
        "direction": direction,
        "support_kind": f"ood_{direction}",
        "lower": lower,
        "upper": upper,
        "transformed_lower": lower_log,
        "transformed_upper": upper_log,
        "transformed_gap": gap,
        "transformed_width": upper_log - lower_log,
        "transformed_gap_fraction": gap / id_width,
        "transformed_width_fraction": (upper_log - lower_log) / id_width,
        "reference_porosity_range": sorted((reference_at_lower, reference_at_upper)),
        "physical_interpretation": ("looser_higher_porosity_global_packing" if direction == "lower" else "denser_lower_porosity_global_packing"),
    }


def resolve_anchor_factor_support(
    *,
    sampled_kappa_mean: float,
    material_kappa_nominal: float,
    eps_bed_cal_ref: float,
    packing_porosity_mean_support: Mapping[str, Any],
    eps_min_global: float,
    eps_max_global: float,
) -> dict[str, Any]:
    """
    Resolve the conditional natural factor interval and feasible OOD tails.

    The interval is conditioned on the sampled mean permeability and the active
    density-calibration reference. OOD tails are exact relative log-space
    extensions of that interval and are never independently authored values.
    """
    permeability = _finite(sampled_kappa_mean, label="Sampled mean permeability")
    if permeability <= 0.0:
        message = "Sampled mean permeability must be strictly positive."
        raise ValueError(message)
    support_lower = _finite(packing_porosity_mean_support.get("lower"), label="Material packing support lower bound")
    support_upper = _finite(packing_porosity_mean_support.get("upper"), label="Material packing support upper bound")
    global_lower = _finite(eps_min_global, label="Global porosity lower guard")
    global_upper = _finite(eps_max_global, label="Global porosity upper guard")
    if not 0.0 < global_lower < support_lower < support_upper < global_upper < 1.0:
        message = "Material packing support must lie strictly inside the global porosity guards."
        raise ValueError(message)
    reference_coefficient = derive_reference_coefficient(material_kappa_nominal, eps_bed_cal_ref)
    id_lower = permeability / (reference_coefficient * kozeny_carman_response(support_upper))
    id_upper = permeability / (reference_coefficient * kozeny_carman_response(support_lower))
    if not 0.0 < id_lower < id_upper or not all(math.isfinite(value) for value in (id_lower, id_upper)):
        message = "Conditional Kozeny-Carman anchor-factor bounds must be finite, positive, and ordered."
        raise ValueError(message)
    lower_log = math.log(id_lower)
    upper_log = math.log(id_upper)
    width = upper_log - lower_log
    candidate_logs = (
        (
            "lower",
            lower_log - _LOWER_TAIL_OUTER_OFFSET * width,
            lower_log - _LOWER_TAIL_INNER_OFFSET * width,
        ),
        (
            "upper",
            upper_log + _UPPER_TAIL_INNER_OFFSET * width,
            upper_log + _UPPER_TAIL_OUTER_OFFSET * width,
        ),
    )
    available: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for direction, tail_lower, tail_upper in candidate_logs:
        try:
            record = _tail_record(
                direction,
                tail_lower,
                tail_upper,
                id_lower_log=lower_log,
                id_upper_log=upper_log,
                sampled_kappa_mean=permeability,
                reference_coefficient=reference_coefficient,
                eps_min_global=global_lower,
                eps_max_global=global_upper,
            )
        except ValueError as error:
            unavailable.append({"direction": direction, "reason": str(error)})
            continue
        reference_range = record["reference_porosity_range"]
        expected_departure = reference_range[0] > support_upper if direction == "lower" else reference_range[1] < support_lower
        if not expected_departure:
            unavailable.append(
                {
                    "direction": direction,
                    "reason": "tail does not produce the required material-support departure",
                }
            )
            continue
        available.append(record)
    return {
        "support_kind": "conditional",
        "support_resolver": "kozeny_carman_anchor_factor",
        "conditioning_coordinate": "kappa_mean",
        "material_kappa_nominal": _finite(material_kappa_nominal, label="Material nominal permeability"),
        "sampled_kappa_mean": permeability,
        "eps_bed_cal_ref": _finite(eps_bed_cal_ref, label="Density-calibration porosity"),
        "packing_porosity_mean_support": {
            "lower": support_lower,
            "upper": support_upper,
        },
        "A_KC_reference": reference_coefficient,
        "id_interval": {
            "lower": id_lower,
            "upper": id_upper,
            "transformed_lower": lower_log,
            "transformed_upper": upper_log,
            "transformed_width": width,
        },
        "available_ood_tails": available,
        "unavailable_ood_directions": unavailable,
        "ood_basis": "conditional extrapolation beyond material-natural packing support",
        "ood_status": "synthetic_design",
    }


def classify_anchor_factor(value: float, support: Mapping[str, Any]) -> str:
    """Return the exact active natural or OOD support containing one factor."""
    factor = _finite(value, label="Kozeny-Carman anchor factor")
    natural = support["id_interval"]
    if float(natural["lower"]) <= factor <= float(natural["upper"]):
        return "natural"
    for tail in support["available_ood_tails"]:
        if float(tail["lower"]) <= factor <= float(tail["upper"]):
            return str(tail["support_kind"])
    message = "Kozeny-Carman anchor factor lies outside every resolved conditional support."
    raise ValueError(message)
