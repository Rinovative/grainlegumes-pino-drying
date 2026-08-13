"""
===============================================================================
generation_cases_fields.py
===============================================================================
Generate deterministic bed, pressure, and initial-moisture spatial fields.
Responsibilities:
  - Preserve the maintained multiscale permeability and porosity construction
  - Use independent label-derived streams for bed, pressure, and moisture fields
  - Build a smooth bounded dry-basis initial-moisture field without clipping
Design principles:
  - Cartesian orientation, units, and physical invariants fail closed
  - Moisture amplitude means maximum absolute deviation from its configured mean
  - Derived dry-bulk density and initial water follow the finalized formulas
This module does NOT:
  - Define scientific ranges, schedules, COMSOL mappings, or storage semantics
  - Add local or high-frequency perturbations to initial moisture
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy import signal

from src.generation.contracts import generation_contracts_porosity as porosity_service
from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_cases_seeding as seeding

_MINIMUM_AXIS_POINTS = 2
_EQUAL_PROBABILITY = 0.5
_MAX_SPATIAL_SUPPORT_ATTEMPTS = 32


@dataclass(frozen=True, slots=True)
class SpatialFields:
    """One generated Cartesian input set and realized diagnostics."""

    shape: tuple[int, int]
    columns: dict[str, np.ndarray]
    metadata: dict[str, Any]


POROSITY_DIAGNOSTIC_KEYS: Final = frozenset(
    {
        "texture_source",
        "background_field_sha256",
        "material_family",
        "material_kappa_nominal",
        "material_eps_bed_cal_ref",
        "A_KC_reference",
        "authored_permeability_support",
        "kc_compatible_permeability_support",
        "effective_joint_permeability_support",
        "active_kappa_mean_support",
        "natural_porosity_support",
        "sampled_kappa_mean",
        "eps_kc_trend",
        "packing_scatter_seed",
        "packing_scatter_z",
        "packing_scatter_truncation_lower",
        "packing_scatter_truncation_upper",
        "packing_scatter_margin",
        "packing_scatter_sigma",
        "packing_scatter_support_kind",
        "packing_scatter_support_lower",
        "packing_scatter_support_upper",
        "eps_reference",
        "pointwise_guard_lower",
        "pointwise_guard_upper",
        "eps_bed_min",
        "eps_bed_max",
        "eps_bed_mean",
        "eps_bed_std",
        "eps_bed_q05",
        "eps_bed_q50",
        "eps_bed_q95",
        "eps_bed_clipped_fraction",
    }
)


def validate_porosity_diagnostics(value: Mapping[str, Any]) -> None:
    """Validate exact fixed-calibration, packing-scatter, and local-field provenance."""
    missing, unknown = sorted(POROSITY_DIAGNOSTIC_KEYS.difference(value)), sorted(set(value).difference(POROSITY_DIAGNOSTIC_KEYS))
    if missing or unknown:
        msg = f"Porosity diagnostics schema is invalid: missing={missing}, unknown={unknown}."
        raise ValueError(msg)
    if (
        value["packing_scatter_truncation_lower"] != porosity_service.PACKING_SCATTER_TRUNCATION_LOWER
        or value["packing_scatter_truncation_upper"] != porosity_service.PACKING_SCATTER_TRUNCATION_UPPER
    ):
        msg = "Porosity diagnostics truncation bounds must be exactly [-3, 3]."
        raise ValueError(msg)
    z = value["packing_scatter_z"]
    if (
        isinstance(z, bool)
        or not isinstance(z, (int, float))
        or not porosity_service.PACKING_SCATTER_TRUNCATION_LOWER < float(z) < porosity_service.PACKING_SCATTER_TRUNCATION_UPPER
    ):
        msg = "Porosity diagnostics packing scatter must lie strictly inside (-3, 3)."
        raise ValueError(msg)


class PorositySupportError(ValueError):
    """Report a porosity-support failure and whether regeneration may resolve it."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        """Initialize the failure with its deterministic retry classification."""
        super().__init__(message)
        self.retryable = retryable


def _finite(values: Mapping[str, Any], name: str) -> float:
    """Return one required finite non-boolean parameter value."""
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        msg = f"Sampled parameter {name!r} must be finite."
        raise ValueError(msg)
    return float(value)


def _gaussian_kernel(sigma_x: float, sigma_y: float, *, shared_radius: float | None = None) -> np.ndarray:
    """Return one normalized finite Gaussian convolution kernel."""
    if sigma_x <= 0 or sigma_y <= 0:
        msg = "Gaussian correlation widths must be strictly positive."
        raise ValueError(msg)
    radius = math.ceil(6.0 * (max(sigma_x, sigma_y) if shared_radius is None else shared_radius))
    coordinate: np.ndarray = np.arange(-radius, radius + 1, dtype=np.float64)
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    kernel = np.exp(-(x_grid**2 / (2.0 * sigma_x**2) + y_grid**2 / (2.0 * sigma_y**2)))
    return kernel / np.sum(kernel)


def _convolve(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply deterministic zero-padded two-dimensional convolution."""
    return signal.fftconvolve(field, kernel, mode="same")


def _standardize(field: np.ndarray, *, label: str) -> np.ndarray:
    """Return a zero-mean field with sample standard deviation one."""
    standard_deviation = float(np.std(field, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= np.finfo(np.float64).eps:
        msg = f"Generated {label} has degenerate variance."
        raise ValueError(msg)
    return (field - np.mean(field)) / standard_deviation


def _array_sha256(field: np.ndarray) -> str:
    """Hash one canonical float64 field with explicit shape evidence."""
    array = np.ascontiguousarray(field, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(",".join(str(length) for length in array.shape).encode("ascii"))
    digest.update(b"|float64|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _multiscale_background(
    shape: tuple[int, int],
    *,
    resolution: float,
    length_x: float,
    coarse_len_rel: float,
    fine_len_rel: float,
    coarse_weight: float,
    fine_weight: float,
    fine_ani_x: float,
    fine_ani_y: float,
    cross_scale_corr: float,
    random: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correlated coarse, fine, and weighted multiscale fields."""
    if (
        coarse_len_rel <= 0
        or fine_len_rel <= 0
        or fine_ani_x <= 0
        or fine_ani_y <= 0
        or not -1 <= cross_scale_corr <= 1
        or min(coarse_weight, fine_weight) < 0
        or not math.isclose(coarse_weight + fine_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        msg = f"{label} multiscale parameters violate their physical domains."
        raise ValueError(msg)
    denominator = math.sqrt(8.0 * math.log(2.0))
    sigma_coarse = coarse_len_rel * length_x / denominator / resolution
    sigma_fine_x = fine_len_rel * length_x / denominator / resolution * fine_ani_x
    sigma_fine_y = fine_len_rel * length_x / denominator / resolution * fine_ani_y
    shared_radius = max(sigma_coarse, sigma_fine_x, sigma_fine_y)
    coarse_kernel = _gaussian_kernel(sigma_coarse, sigma_coarse, shared_radius=shared_radius)
    fine_kernel = _gaussian_kernel(sigma_fine_x, sigma_fine_y, shared_radius=shared_radius)
    seed_coarse = random.standard_normal(shape)
    independent = random.standard_normal(shape)
    seed_fine = cross_scale_corr * seed_coarse + math.sqrt(max(0.0, 1.0 - cross_scale_corr**2)) * independent
    coarse = _standardize(_convolve(seed_coarse, coarse_kernel), label=f"{label} coarse structure")
    fine = _standardize(_convolve(seed_fine, fine_kernel), label=f"{label} fine structure")
    return coarse, fine, coarse_weight * coarse + fine_weight * fine


def _bed_structure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    resolution: float,
    values: Mapping[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the maintained bed background and optional local perturbations."""
    length_x = float(x_grid[0, -1] - x_grid[0, 0])
    coarse, fine, background = _multiscale_background(
        x_grid.shape,
        resolution=resolution,
        length_x=length_x,
        coarse_len_rel=_finite(values, "bed.structure.coarse_len_rel"),
        fine_len_rel=_finite(values, "bed.structure.fine_len_rel"),
        coarse_weight=_finite(values, "bed.structure.coarse_weight"),
        fine_weight=_finite(values, "bed.structure.fine_weight"),
        fine_ani_x=_finite(values, "bed.structure.fine_ani_x"),
        fine_ani_y=_finite(values, "bed.structure.fine_ani_y"),
        cross_scale_corr=_finite(values, "bed.structure.cross_scale_corr"),
        random=random,
        label="bed",
    )
    local = np.zeros_like(background)
    level = _finite(values, "bed.perturbations.amplitude")
    if level < 0:
        msg = "bed.perturbations.amplitude must be non-negative."
        raise ValueError(msg)
    if level > 0:
        granularity = _finite(values, "bed.perturbations.granularity")
        bias = _finite(values, "bed.perturbations.sign_bias")
        if not 0 <= granularity <= 1 or not 0 <= bias <= 1:
            msg = "Bed perturbation granularity and sign bias must lie in [0, 1]."
            raise ValueError(msg)
        normalized_x = x_grid / length_x
        normalized_y = y_grid / float(y_grid[-1, 0] - y_grid[0, 0])
        length_zero = _finite(values, "bed.structure.coarse_len_rel")
        sigma_min = length_zero / 10.0
        sigma_characteristic = sigma_min * (length_zero / sigma_min) ** (1.0 - granularity)
        mean_count = level / max(math.pi * sigma_characteristic**2, np.finfo(np.float64).eps)
        for _ in range(int(random.poisson(mean_count))):
            center_x, center_y = random.random(2)
            spread = math.log(2.0)
            sigma = sigma_characteristic * math.exp(0.5 * spread * float(random.standard_normal()))
            aspect = math.exp(spread * float(random.standard_normal()))
            sigma_x, sigma_y = (sigma * aspect, sigma) if float(random.random()) < _EQUAL_PROBABILITY else (sigma, sigma * aspect)
            angle = 2.0 * math.pi * float(random.random())
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated_x = cosine * (normalized_x - center_x) + sine * (normalized_y - center_y)
            rotated_y = -sine * (normalized_x - center_x) + cosine * (normalized_y - center_y)
            amplitude = 1.0 if float(random.random()) < bias else -1.0
            local += amplitude * np.exp(-(rotated_x**2 / (2.0 * sigma_x**2) + rotated_y**2 / (2.0 * sigma_y**2)))
        local -= np.mean(local)
        rms = math.sqrt(float(np.mean(local**2)))
        local = level * local / max(rms, np.finfo(np.float64).eps)
    structure = _standardize(background + local, label="final bed structure")
    return (
        structure,
        background,
        {
            "coarse_len_rel": _finite(values, "bed.structure.coarse_len_rel"),
            "fine_len_rel": _finite(values, "bed.structure.fine_len_rel"),
            "coarse_weight": _finite(values, "bed.structure.coarse_weight"),
            "fine_weight": _finite(values, "bed.structure.fine_weight"),
            "cross_scale_corr": _finite(values, "bed.structure.cross_scale_corr"),
            "fine_ani_x": _finite(values, "bed.structure.fine_ani_x"),
            "fine_ani_y": _finite(values, "bed.structure.fine_ani_y"),
            "perturbation_amplitude": level,
            "coarse_mean": float(np.mean(coarse)),
            "fine_mean": float(np.mean(fine)),
        },
    )


def _permeability_fields(
    structure: np.ndarray,
    background: np.ndarray,
    *,
    resolution: float,
    length_x: float,
    values: Mapping[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map bed structure to scalar and symmetric positive-definite permeability."""
    relative_variation = _finite(values, "kappa_cv")
    mean = _finite(values, "kappa_mean")
    if relative_variation < 0 or mean <= 0:
        msg = "Permeability mean must be positive and relative variation non-negative."
        raise ValueError(msg)
    log_standard_deviation = math.sqrt(math.log1p(relative_variation**2))
    kappa = mean * np.exp(log_standard_deviation * structure - 0.5 * log_standard_deviation**2)
    sigma_theta = max(_finite(values, "permeability.orientation.smooth_len_rel") * length_x / resolution, 1.0)
    orientation_kernel = _gaussian_kernel(sigma_theta, sigma_theta)
    normalized_absolute = np.abs(background) / max(float(np.max(np.abs(background))), np.finfo(np.float64).eps)
    ratio = 1.0 + _finite(values, "permeability.anisotropy.strength") * (
        (_finite(values, "permeability.anisotropy.max_ratio") - 1.0) * normalized_absolute ** _finite(values, "permeability.anisotropy.exponent")
    )
    if np.any(ratio <= 0):
        msg = "Permeability anisotropy ratio must remain positive."
        raise ValueError(msg)
    gradient_y, gradient_x = np.gradient(background, resolution, resolution)
    theta_raw = np.arctan2(gradient_y, gradient_x)
    director_x = _convolve(np.cos(2.0 * theta_raw), orientation_kernel)
    director_y = _convolve(np.sin(2.0 * theta_raw), orientation_kernel)
    norm = np.sqrt(director_x**2 + director_y**2)
    director_x /= np.maximum(norm, np.finfo(np.float64).eps)
    director_y /= np.maximum(norm, np.finfo(np.float64).eps)
    jitter = _finite(values, "permeability.orientation.jitter")
    if jitter < 0:
        msg = "permeability.orientation.jitter must be non-negative."
        raise ValueError(msg)
    if jitter > 0:
        director_x += jitter * _convolve(random.standard_normal(director_x.shape), orientation_kernel)
        director_y += jitter * _convolve(random.standard_normal(director_y.shape), orientation_kernel)
        norm = np.sqrt(director_x**2 + director_y**2)
        director_x /= np.maximum(norm, np.finfo(np.float64).eps)
        director_y /= np.maximum(norm, np.finfo(np.float64).eps)
    theta = 0.5 * np.arctan2(director_y, director_x)
    k1 = kappa * ratio
    k2 = kappa / ratio
    cosine = np.cos(theta)
    sine = np.sin(theta)
    kxx = k1 * cosine**2 + k2 * sine**2
    kyy = k1 * sine**2 + k2 * cosine**2
    kxy = (k1 - k2) * sine * cosine
    determinant = kxx * kyy - kxy**2
    if not np.isfinite(np.stack((kxx, kxy, kyy))).all() or np.any(determinant <= 0):
        msg = "Generated permeability tensor is non-finite or not positive definite."
        raise ValueError(msg)
    return (
        kappa,
        kxx,
        kxy,
        kyy,
        {
            "kappa_mean": mean,
            "kappa_cv": relative_variation,
            "log_standard_deviation": log_standard_deviation,
            "determinant_min": float(np.min(determinant)),
            "units": {
                "kappa_mean": "m^2",
                "kappa_cv": "1",
                "log_standard_deviation": "1",
                "determinant_min": "m^4",
            },
        },
    )


def _packing_reference(
    values: Mapping[str, Any],
    *,
    coupling: Mapping[str, Any],
    active_ood_unit: str | None,
    packing_scatter_seed: int,
) -> dict[str, Any]:
    """Resolve one fixed case-level KC trend and bounded packing scatter."""
    minimum = _finite(values, "eps_min_global")
    maximum = _finite(values, "eps_max_global")
    permeability = _finite(values, "kappa_mean")
    coefficient = float(coupling["A_KC_reference"])
    trend = porosity_service.solve_reference_porosity(permeability, coefficient, eps_min_global=minimum, eps_max_global=maximum)
    support_kind, support_lower, support_upper = porosity_service.resolve_active_packing_support_for_value(
        coupling, sampled_kappa_mean=permeability, active_ood_unit=active_ood_unit
    )
    if support_kind == "natural":
        active_kappa_support = copy.deepcopy(coupling["effective_joint_permeability_support"])
    else:
        direction = support_kind.removeprefix("kappa_mean_ood_")
        tail = coupling["kappa_ood_porosity_supports"][direction]
        active_kappa_support = {
            "lower": float(tail["kappa_lower"]),
            "upper": float(tail["kappa_upper"]),
        }
    margin = min(trend - support_lower, support_upper - trend)
    tolerance = 1e-14
    if margin < -tolerance:
        msg = f"Kozeny-Carman trend {trend} lies outside active packing support [{support_lower}, {support_upper}]."
        raise PorositySupportError(
            msg,
            retryable=False,
        )
    if abs(margin) <= tolerance:
        margin = 0.0
    sigma = margin / porosity_service.PACKING_SCATTER_TRUNCATION_UPPER
    z = porosity_service.draw_truncated_standard_normal(packing_scatter_seed)
    reference = trend + sigma * z
    if not support_lower - tolerance <= reference <= support_upper + tolerance:
        msg = "Packing scatter reference lies outside its active support."
        raise RuntimeError(msg)
    return {
        "material_family": str(coupling["material_family"]),
        "material_kappa_nominal": float(coupling["material_kappa_nominal"]),
        "material_eps_bed_cal_ref": float(coupling["material_eps_bed_cal_ref"]),
        "A_KC_reference": coefficient,
        "authored_permeability_support": copy.deepcopy(coupling["authored_permeability_support"]),
        "kc_compatible_permeability_support": copy.deepcopy(coupling["kc_compatible_permeability_support"]),
        "effective_joint_permeability_support": copy.deepcopy(coupling["effective_joint_permeability_support"]),
        "active_kappa_mean_support": active_kappa_support,
        "natural_porosity_support": copy.deepcopy(coupling["natural_porosity_support"]),
        "sampled_kappa_mean": permeability,
        "eps_kc_trend": trend,
        "packing_scatter_seed": packing_scatter_seed,
        "packing_scatter_z": z,
        "packing_scatter_truncation_lower": porosity_service.PACKING_SCATTER_TRUNCATION_LOWER,
        "packing_scatter_truncation_upper": porosity_service.PACKING_SCATTER_TRUNCATION_UPPER,
        "packing_scatter_margin": margin,
        "packing_scatter_sigma": sigma,
        "packing_scatter_support_kind": support_kind,
        "packing_scatter_support_lower": support_lower,
        "packing_scatter_support_upper": support_upper,
        "eps_reference": reference,
    }


def _porosity_field(
    background: np.ndarray, *, resolution: float, length_x: float, values: Mapping[str, Any], reference: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate local porosity texture around one fixed case-level reference."""
    smooth_relative = _finite(values, "porosity.smooth_len_rel")
    if smooth_relative < 0:
        msg = "porosity.smooth_len_rel must be non-negative."
        raise ValueError(msg)
    latent = (
        _convolve(background, _gaussian_kernel(max(smooth_relative * length_x / resolution, 1.0), max(smooth_relative * length_x / resolution, 1.0)))
        if smooth_relative > 0
        else background
    )
    texture = _standardize(latent, label="porosity structure")
    texture -= np.mean(texture)
    texture /= max(math.sqrt(float(np.mean(texture**2))), np.finfo(np.float64).eps)
    minimum, maximum = _finite(values, "eps_min_global"), _finite(values, "eps_max_global")
    unconstrained = float(reference["eps_reference"]) + _finite(values, "porosity.texture_amp") * texture
    clipped = (unconstrained < minimum) | (unconstrained > maximum)
    porosity = np.clip(unconstrained, minimum, maximum)
    if not np.isfinite(porosity).all() or np.any((porosity < minimum) | (porosity > maximum)):
        msg = "Generated porosity violates configured pointwise guards."
        raise ValueError(msg)
    realized_mean = float(np.mean(porosity))
    natural = reference["natural_porosity_support"]
    natural_lower, natural_upper = float(natural["lower"]), float(natural["upper"])
    support_kind = str(reference["packing_scatter_support_kind"])
    if support_kind == "natural":
        retained_state = natural_lower <= realized_mean <= natural_upper
    elif support_kind == "kappa_mean_ood_lower":
        retained_state = realized_mean < natural_lower
    elif support_kind == "kappa_mean_ood_upper":
        retained_state = realized_mean > natural_upper
    else:
        msg = f"Unknown packing scatter support kind {support_kind!r}."
        raise PorositySupportError(msg, retryable=False)
    if not retained_state:
        msg = "Generated mean porosity does not retain its natural or kappa_mean-OOD state."
        raise PorositySupportError(msg, retryable=True)
    diagnostics = {
        "texture_source": "z_background",
        "background_field_sha256": _array_sha256(background),
        **copy.deepcopy(dict(reference)),
        "pointwise_guard_lower": minimum,
        "pointwise_guard_upper": maximum,
        "eps_bed_min": float(np.min(porosity)),
        "eps_bed_max": float(np.max(porosity)),
        "eps_bed_mean": realized_mean,
        "eps_bed_std": float(np.std(porosity)),
        "eps_bed_q05": float(np.quantile(porosity, 0.05)),
        "eps_bed_q50": float(np.quantile(porosity, 0.5)),
        "eps_bed_q95": float(np.quantile(porosity, 0.95)),
        "eps_bed_clipped_fraction": float(np.mean(clipped)),
    }
    return porosity, diagnostics


def _pressure_boundary(x_grid: np.ndarray, *, values: Mapping[str, Any], random: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one-dimensional inlet pressure in its case-owned stream."""
    x_axis = x_grid[0, :]
    normalized_x = (x_axis - np.min(x_axis)) / max(float(np.ptp(x_axis)), np.finfo(np.float64).eps)
    shape = _finite(values, "pressure_bc.sin_amp") * np.sin(
        2.0 * math.pi * _finite(values, "pressure_bc.sin_freq") * normalized_x + _finite(values, "pressure_bc.sin_phase")
    )
    count = int(_finite(values, "pressure_bc.gauss_count"))
    gaussian_amplitude = _finite(values, "pressure_bc.gauss_amp")
    if count > 0 and gaussian_amplitude != 0:
        centers = np.linspace(0.0, 1.0, count + 2)[1:-1]
        sigma_zero = _finite(values, "pressure_bc.gauss_width")
        if sigma_zero <= 0:
            msg = "pressure_bc.gauss_width must be positive when Gaussian pressure terms are active."
            raise ValueError(msg)
        sigmas = sigma_zero * (1.0 + _finite(values, "pressure_bc.gauss_jitter") * random.standard_normal(count))
        sigmas = np.maximum(sigmas, 0.05 * sigma_zero)
        for center, sigma in zip(centers, sigmas, strict=True):
            shape += gaussian_amplitude / count * np.exp(-((normalized_x - center) ** 2) / (2.0 * sigma**2))
    shape += _finite(values, "pressure_bc.linear_amp") * (2.0 * normalized_x - 1.0)
    inlet = _finite(values, "pressure_bc.mean") * (1.0 + shape)
    if not np.isfinite(inlet).all() or np.any(inlet <= 0.0):
        msg = "Generated inlet pressure must remain finite and strictly positive without clipping."
        raise ValueError(msg)
    field = np.zeros_like(x_grid)
    field[0, :] = inlet
    return field, {"minimum": float(np.min(inlet)), "maximum": float(np.max(inlet)), "unit": "Pa"}


def _initial_moisture(
    shape: tuple[int, int],
    *,
    resolution: float,
    length_x: float,
    values: Mapping[str, Any],
    family_bounds: Mapping[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the independent smooth bounded dry-basis moisture realization."""
    _base, _smooth, latent = _multiscale_background(
        shape,
        resolution=resolution,
        length_x=length_x,
        coarse_len_rel=_finite(values, "initial_moisture.structure.coarse_len_rel"),
        fine_len_rel=_finite(values, "initial_moisture.structure.fine_len_rel"),
        coarse_weight=_finite(values, "initial_moisture.structure.coarse_weight"),
        fine_weight=_finite(values, "initial_moisture.structure.fine_weight"),
        fine_ani_x=_finite(values, "initial_moisture.structure.fine_ani_x"),
        fine_ani_y=_finite(values, "initial_moisture.structure.fine_ani_y"),
        cross_scale_corr=_finite(values, "initial_moisture.structure.cross_scale_corr"),
        random=random,
        label="initial moisture",
    )
    latent -= np.mean(latent)
    maximum_absolute = float(np.max(np.abs(latent)))
    if maximum_absolute <= np.finfo(np.float64).eps:
        msg = "Initial-moisture latent field has degenerate amplitude."
        raise ValueError(msg)
    latent /= maximum_absolute
    mean = _finite(values, "initial_moisture.mean_db")
    amplitude = _finite(values, "initial_moisture.amplitude_db")
    lower = float(family_bounds["lower"])
    upper = float(family_bounds["upper"])
    natural_lower = float(family_bounds.get("natural_lower", lower))
    natural_upper = float(family_bounds.get("natural_upper", upper))
    allow_departure = family_bounds.get("natural_support_departure_allowed", False)
    if not isinstance(allow_departure, bool):
        msg = "Initial-moisture natural-support departure state must be boolean."
        raise TypeError(msg)
    permitted = min(mean - lower, upper - mean)
    tolerance = 16.0 * np.finfo(np.float64).eps
    if amplitude < 0 or amplitude > permitted + tolerance:
        msg = "Initial-moisture amplitude violates the active analytical no-clipping bound."
        raise ValueError(msg)
    field = mean + amplitude * latent
    if np.any(field < lower - tolerance) or np.any(field > upper + tolerance):
        msg = "Initial-moisture construction escaped its analytical no-clipping bounds."
        raise RuntimeError(msg)
    realized_mean = float(np.mean(field))
    realized_minimum = float(np.min(field))
    realized_maximum = float(np.max(field))
    within_natural_support = bool(realized_minimum >= natural_lower - tolerance and realized_maximum <= natural_upper + tolerance)
    if not allow_departure and not within_natural_support:
        msg = "Initial-moisture construction escaped the material natural support."
        raise RuntimeError(msg)
    metadata: dict[str, Any] = {
        "configured_mean": mean,
        "configured_max_abs_deviation": amplitude,
        "mean": realized_mean,
        "standard_deviation": float(np.std(field, ddof=1)),
        "minimum": realized_minimum,
        "maximum": realized_maximum,
        "maximum_absolute_deviation": float(np.max(np.abs(field - mean))),
        "latent_mean": float(np.mean(latent)),
        "latent_maximum_absolute": float(np.max(np.abs(latent))),
        "enforced_lower": lower,
        "enforced_upper": upper,
        "material_natural_lower": natural_lower,
        "material_natural_upper": natural_upper,
        "within_material_natural_support": within_natural_support,
        "natural_support_departure_allowed": allow_departure,
        "field_sha256": _array_sha256(field),
        "units": {
            "configured_mean": "kg/kg",
            "configured_max_abs_deviation": "kg/kg",
            "mean": "kg/kg",
            "standard_deviation": "kg/kg",
            "minimum": "kg/kg",
            "maximum": "kg/kg",
            "maximum_absolute_deviation": "kg/kg",
            "latent_mean": "1",
            "latent_maximum_absolute": "1",
            "enforced_lower": "kg/kg",
            "enforced_upper": "kg/kg",
            "material_natural_lower": "kg/kg",
            "material_natural_upper": "kg/kg",
        },
    }
    constraint = family_bounds.get("target_separation_constraint")
    if isinstance(constraint, Mapping):
        metadata["target_separation_constraint"] = {
            "authored_expression": constraint["authored_expression"],
            "minimum_db": constraint["minimum_db"],
            "margin_above_target_db": constraint["margin_above_target_db"],
            "unit": constraint["unit"],
        }
    active_ood_unit = family_bounds.get("active_ood_unit")
    if active_ood_unit is not None:
        metadata["active_ood_unit"] = active_ood_unit
    return field, metadata


def _generate_spatial_fields_once(
    simulation_profile: str,
    grid: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    seeds: Mapping[str, int],
    family_bounds: Mapping[str, Any] | None,
    porosity_reference: Mapping[str, Any],
) -> SpatialFields:
    """Generate one deterministic profile-owned spatial input set."""
    if simulation_profile == profiles.STEADY_FLOW_PROFILE:
        required_seeds = {"bed", "pressure_bc", "packing_scatter"}
    elif simulation_profile == profiles.TRANSIENT_DRYING_PROFILE:
        required_seeds = {"bed", "pressure_bc", "initial_moisture", "packing_scatter"}
    else:
        available = ", ".join(profiles.available_profiles())
        msg = f"Unknown simulation_profile {simulation_profile!r}. Available profiles: {available}."
        raise ValueError(msg)
    if set(seeds) != required_seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1 for seed in seeds.values()
    ):
        msg = f"Spatial generation for {simulation_profile!r} requires exact uint32 seeds {sorted(required_seeds)}."
        raise ValueError(msg)
    length_x = float(grid["Lx"])
    length_y = float(grid["Ly"])
    length_z = float(grid["Lz"])
    resolution_x = float(grid["dx"])
    resolution_y = float(grid["dy"])
    count_x = int(grid["nx"])
    count_y = int(grid["ny"])
    if count_x < _MINIMUM_AXIS_POINTS or count_y < _MINIMUM_AXIS_POINTS or not math.isclose(resolution_x, resolution_y):
        msg = "Spatial grid must contain at least two points per axis with equal resolution."
        raise ValueError(msg)
    if not math.isclose((count_x - 1) * resolution_x, length_x) or not math.isclose((count_y - 1) * resolution_y, length_y):
        msg = "Boundary-inclusive spatial grid dimensions are inconsistent."
        raise ValueError(msg)
    x_axis = np.linspace(0.0, length_x, count_x, dtype=np.float64)
    y_axis = np.linspace(0.0, length_y, count_y, dtype=np.float64)
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    bed_random = np.random.default_rng(seeds["bed"])
    structure, background, structure_metadata = _bed_structure(
        x_grid,
        y_grid,
        resolution=resolution_x,
        values=values,
        random=bed_random,
    )
    _kappa, kxx, kxy, kyy, permeability_metadata = _permeability_fields(
        structure,
        background,
        resolution=resolution_x,
        length_x=length_x,
        values=values,
        random=bed_random,
    )
    porosity, porosity_metadata = _porosity_field(
        background,
        resolution=resolution_x,
        length_x=length_x,
        values=values,
        reference=porosity_reference,
    )
    pressure, pressure_metadata = _pressure_boundary(
        x_grid,
        values=values,
        random=np.random.default_rng(seeds["pressure_bc"]),
    )
    columns = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": kxx,
        "Kxy": kxy,
        "Kyy": kyy,
        "eps_bed": porosity,
        "p_in_bc": pressure,
    }
    metadata: dict[str, Any] = {
        "generator_version": seeding.GENERATOR_VERSION,
        "simulation_profile": simulation_profile,
        "random_stream": "numpy.default_rng",
        "seeds": dict(seeds),
        "geometry": {
            "Lx": length_x,
            "Ly": length_y,
            "Lz": length_z,
            "dx": resolution_x,
            "dy": resolution_y,
            "nx": count_x,
            "ny": count_y,
            "mesh_elements_x": count_x - 1,
            "mesh_elements_y": count_y - 1,
            "boundaries_included": True,
            "units": {
                "Lx": "m",
                "Ly": "m",
                "Lz": "m",
                "dx": "m",
                "dy": "m",
                "nx": "1",
                "ny": "1",
                "mesh_elements_x": "1",
                "mesh_elements_y": "1",
            },
        },
        "structure": structure_metadata,
        "permeability": permeability_metadata,
        "porosity": porosity_metadata,
        "pressure_boundary": pressure_metadata,
        "field_units": {
            "x": "m",
            "y": "m",
            "Kxx": "m^2",
            "Kxy": "m^2",
            "Kyy": "m^2",
            "eps_bed": "1",
            "p_in_bc": "Pa",
        },
    }
    if simulation_profile == profiles.TRANSIENT_DRYING_PROFILE:
        if family_bounds is None:
            msg = "Transient spatial generation requires material-family initial-moisture bounds."
            raise ValueError(msg)
        moisture, moisture_metadata = _initial_moisture(
            x_grid.shape,
            resolution=resolution_x,
            length_x=length_x,
            values=values,
            family_bounds=family_bounds,
            random=np.random.default_rng(seeds["initial_moisture"]),
        )
        calibration_porosity = _finite(values, "eps_bed_cal_ref")
        if not 0 < calibration_porosity < 1:
            msg = "eps_bed_cal_ref must lie strictly inside (0, 1)."
            raise ValueError(msg)
        dry_bulk_density = _finite(values, "rho_bu_dry_ref") * (1.0 - porosity) / (1.0 - calibration_porosity)
        initial_water = dry_bulk_density * moisture
        columns["X_0_db_field"] = moisture
        metadata["initial_moisture"] = moisture_metadata
        metadata["field_units"].update(
            {
                "X_0_db_field": "kg/kg",
                "rho_bu_dry": "kg/m^3",
                "w_gr_0": "kg/m^3",
            }
        )
        metadata["derived_fields"] = {
            "rho_bu_dry_formula": "rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)",
            "w_gr_0_formula": "rho_bu_dry*X_0_db_field",
            "initial_compartment_values": "w_surf(0)=w_int(0)=w_gr_0",
            "rho_bu_dry_sha256": _array_sha256(dry_bulk_density),
            "w_gr_0_sha256": _array_sha256(initial_water),
        }
    expected_columns = profiles.spatial_input_fields(simulation_profile)
    if tuple(columns) != expected_columns:
        msg = f"Generated spatial columns do not match the {simulation_profile!r} contract."
        raise RuntimeError(msg)
    if not all(array.shape == x_grid.shape and np.isfinite(array).all() for array in columns.values()):
        msg = "Generated spatial fields must share one finite Cartesian shape."
        raise ValueError(msg)
    return SpatialFields(shape=x_grid.shape, columns=columns, metadata=metadata)


def _complete_case_retry_seeds(seeds: Mapping[str, int], attempt: int) -> dict[str, int]:
    """Return exact initial seeds or one deterministic complete-case retry plan."""
    if attempt == 1:
        return dict(seeds)
    return {
        name: (seed if name == "packing_scatter" else seeding.derive_seed(seed, "complete_case_support_retry", str(attempt), name))
        for name, seed in seeds.items()
    }


def generate_spatial_fields(
    simulation_profile: str,
    grid: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    seeds: Mapping[str, int],
    family_bounds: Mapping[str, Any] | None,
    porosity_coupling: Mapping[str, Any],
    active_ood_unit: str | None,
) -> SpatialFields:
    """
    Generate one deterministic spatial case with bounded support regeneration.

    Only a realized mean-support failure caused by stochastic texture may retry.
    Every retry regenerates the complete spatial input set from one derived seed
    plan; its accepted attempt also controls remaining case-random inputs.
    Structural coupling or attribution failures remain immediate errors.
    """
    if "packing_scatter" not in seeds:
        msg = "Spatial generation requires one packing_scatter semantic seed."
        raise ValueError(msg)
    porosity_reference = _packing_reference(
        values, coupling=porosity_coupling, active_ood_unit=active_ood_unit, packing_scatter_seed=int(seeds["packing_scatter"])
    )
    rejection_reasons: list[str] = []
    for attempt in range(1, _MAX_SPATIAL_SUPPORT_ATTEMPTS + 1):
        try:
            result = _generate_spatial_fields_once(
                simulation_profile,
                grid,
                values,
                seeds=_complete_case_retry_seeds(seeds, attempt),
                family_bounds=family_bounds,
                porosity_reference=porosity_reference,
            )
        except PorositySupportError as error:
            if not error.retryable:
                raise
            rejection_reasons.append(str(error))
            continue
        metadata = copy.deepcopy(result.metadata)
        accepted_seeds = _complete_case_retry_seeds(seeds, attempt)
        metadata["complete_case_support_retry"] = {
            "maximum_attempts": _MAX_SPATIAL_SUPPORT_ATTEMPTS,
            "acceptance_attempt": attempt,
            "rejection_count": len(rejection_reasons),
            "rejection_reasons": rejection_reasons,
            "accepted_attempt_seeds": accepted_seeds,
            "policy": "reject_and_deterministically_regenerate_complete_case",
        }
        return SpatialFields(
            shape=result.shape,
            columns=result.columns,
            metadata=metadata,
        )
    message = (
        f"No complete spatial realization retained the required material porosity-support state after "
        f"{_MAX_SPATIAL_SUPPORT_ATTEMPTS} deterministic attempts; last_reason={rejection_reasons[-1]!r}."
    )
    raise PorositySupportError(message, retryable=False)
