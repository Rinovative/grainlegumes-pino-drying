"""
===============================================================================
generation_fields.py
===============================================================================
Generate deterministic shared spatial input fields in Python.
Responsibilities:
  - Port the maintained multiscale structure and localized-noise formulas
  - Map structure into permeability tensor, porosity, and pressure-boundary fields
  - Preserve Cartesian orientation and deterministic Fortran-order export ordering
Design principles:
  - Existing scientific formulas and units remain explicit
  - One case seed drives one deterministic NumPy stream
  - Tensor and porosity invariants fail before persistence
This module does NOT:
  - Provide hidden compatibility generators or alternate random streams
  - Define profile scalars, schedules, output channels, or COMSOL behavior
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import signal

from . import generation_config as config_contract

PAIR_SIZE = 2
MIN_AXIS_POINTS = 2
EQUAL_PROBABILITY = 0.5


@dataclass(frozen=True, slots=True)
class SpatialFields:
    """One generated Cartesian field set and its provenance metadata."""

    shape: tuple[int, int]
    columns: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _finite_real(parameters: dict[str, Any], name: str) -> float:
    """Return one finite generator parameter."""
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"generator.parameters.{name} must be a finite real number."
        raise ValueError(message)
    return float(value)


def _pair(parameters: dict[str, Any], name: str) -> tuple[float, float]:
    """Return one finite two-value generator parameter."""
    value = parameters[name]
    if not isinstance(value, (list, tuple)) or len(value) != PAIR_SIZE:
        message = f"generator.parameters.{name} must contain exactly two values."
        raise ValueError(message)
    first = float(value[0])
    second = float(value[1])
    if not math.isfinite(first) or not math.isfinite(second):
        message = f"generator.parameters.{name} must contain finite values."
        raise ValueError(message)
    return first, second


def _gaussian_kernel(sigma_x: float, sigma_y: float, *, shared_radius: float | None = None) -> np.ndarray:
    """Return the normalized finite Gaussian convolution kernel."""
    if sigma_x <= 0 or sigma_y <= 0:
        message = "Gaussian kernel widths must be strictly positive."
        raise ValueError(message)
    radius = math.ceil(6.0 * (max(sigma_x, sigma_y) if shared_radius is None else shared_radius))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    kernel = np.exp(-(x_grid**2 / (2.0 * sigma_x**2) + y_grid**2 / (2.0 * sigma_y**2)))
    return kernel / np.sum(kernel)


def _convolve(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply zero-padded two-dimensional convolution with same-size output."""
    return signal.convolve2d(field, kernel, mode="same", boundary="fill", fillvalue=0)


def _standardize(field: np.ndarray, *, label: str) -> np.ndarray:
    """Return a zero-mean field with sample standard deviation one."""
    standard_deviation = float(np.std(field, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= np.finfo(np.float64).eps:
        message = f"Generated {label} has degenerate variance."
        raise ValueError(message)
    return (field - np.mean(field)) / standard_deviation


def _validate_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Delegate formula-domain validation to the generator schema owner."""
    return config_contract.validate_generator_parameters(parameters)


def _structure_fields(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    resolution: float,
    values: dict[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the maintained base, smooth, localized, and final structure fields."""
    length_x = float(x_grid[0, -1] - x_grid[0, 0])
    base_length = float(values["base_len_rel"]) * length_x
    smooth_length = float(values["smooth_len_rel"]) * length_x
    anisotropy = cast("tuple[float, float]", values["anisotropy"])
    weights = cast("tuple[float, float]", values["ms_weight"])
    denominator = math.sqrt(8.0 * math.log(2.0))
    sigma_smooth_x = smooth_length / denominator / resolution * anisotropy[0]
    sigma_smooth_y = smooth_length / denominator / resolution * anisotropy[1]
    sigma_base = base_length / denominator / resolution
    shared_radius = max(sigma_smooth_x, sigma_smooth_y, sigma_base)
    smooth_kernel = _gaussian_kernel(sigma_smooth_x, sigma_smooth_y, shared_radius=shared_radius)
    base_kernel = _gaussian_kernel(sigma_base, sigma_base, shared_radius=shared_radius)

    seed_base = random.standard_normal(x_grid.shape)
    uncorrelated = random.standard_normal(x_grid.shape)
    coupling = float(values["coupling"])
    seed_smooth = coupling * seed_base + math.sqrt(1.0 - coupling**2) * uncorrelated
    z_base = _standardize(_convolve(seed_base, base_kernel), label="base structure")
    z_smooth = _standardize(_convolve(seed_smooth, smooth_kernel), label="smooth structure")
    z_background = weights[0] * z_base + weights[1] * z_smooth

    z_noises = np.zeros_like(z_background)
    level = float(values["noise_level"])
    if level > 0:
        granularity = float(values["noise_granularity"])
        bias = float(values["noise_bias"])
        normalized_x = x_grid / length_x
        normalized_y = y_grid / float(y_grid[-1, 0] - y_grid[0, 0])
        length_zero = float(values["base_len_rel"])
        sigma_min = length_zero / 10.0
        sigma_max = length_zero
        sigma_characteristic = sigma_min * (sigma_max / sigma_min) ** (1.0 - granularity)
        mean_count = level / max(math.pi * sigma_characteristic**2, np.finfo(np.float64).eps)
        noise_field = np.zeros_like(z_background)
        for _ in range(int(random.poisson(mean_count))):
            center_x, center_y = random.random(2)
            spread = math.log(2.0)
            sigma = sigma_characteristic * math.exp(0.5 * spread * float(random.standard_normal()))
            aspect = math.exp(spread * float(random.standard_normal()))
            if float(random.random()) < EQUAL_PROBABILITY:
                sigma_x, sigma_y = sigma * aspect, sigma
            else:
                sigma_x, sigma_y = sigma, sigma * aspect
            angle = 2.0 * math.pi * float(random.random())
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated_x = cosine * (normalized_x - center_x) + sine * (normalized_y - center_y)
            rotated_y = -sine * (normalized_x - center_x) + cosine * (normalized_y - center_y)
            amplitude = 1.0 if float(random.random()) < bias else -1.0
            noise_field += amplitude * np.exp(-(rotated_x**2 / (2.0 * sigma_x**2) + rotated_y**2 / (2.0 * sigma_y**2)))
        noise_field -= np.mean(noise_field)
        root_mean_square = math.sqrt(float(np.mean(noise_field**2)))
        noise_field /= max(root_mean_square, np.finfo(np.float64).eps)
        z_noises = level * noise_field
    z = _standardize(z_background + z_noises, label="final structure")
    metadata = {
        "base_len_rel": values["base_len_rel"],
        "smooth_len_rel": values["smooth_len_rel"],
        "ms_weight": list(weights),
        "anisotropy": list(anisotropy),
        "coupling": coupling,
        "noise_level": level,
        "noise_granularity": values["noise_granularity"],
        "noise_bias": values["noise_bias"],
    }
    return z, z_background, metadata


def _permeability_fields(
    z: np.ndarray,
    z_background: np.ndarray,
    *,
    resolution: float,
    length_x: float,
    values: dict[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map structure into scalar and symmetric positive-definite permeability."""
    relative_variation = float(values["var_rel"])
    log_standard_deviation = math.sqrt(math.log1p(relative_variation**2))
    kappa = float(values["k_mean"]) * np.exp(log_standard_deviation * z - 0.5 * log_standard_deviation**2)
    sigma_theta = max(float(values["theta_smooth_rel"]) * length_x / resolution, 1.0)
    orientation_kernel = _gaussian_kernel(sigma_theta, sigma_theta)
    maximum_background = float(np.max(np.abs(z_background)))
    normalized_absolute = np.abs(z_background) / max(maximum_background, np.finfo(np.float64).eps)
    ratio = 1.0 + float(values["tensor_strength"]) * ((float(values["a_max"]) - 1.0) * normalized_absolute ** float(values["a_gamma"]))
    gradient_y, gradient_x = np.gradient(z_background, resolution, resolution)
    theta_raw = np.arctan2(gradient_y, gradient_x)
    director_x = _convolve(np.cos(2.0 * theta_raw), orientation_kernel)
    director_y = _convolve(np.sin(2.0 * theta_raw), orientation_kernel)
    norm = np.sqrt(director_x**2 + director_y**2)
    director_x /= np.maximum(norm, np.finfo(np.float64).eps)
    director_y /= np.maximum(norm, np.finfo(np.float64).eps)
    jitter = float(values["theta_jitter"])
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
        message = "Generated permeability tensor is non-finite or not positive definite."
        raise ValueError(message)
    metadata = {
        "k_mean": values["k_mean"],
        "var_rel": relative_variation,
        "log_standard_deviation": log_standard_deviation,
        "determinant_min": float(np.min(determinant)),
    }
    return kappa, kxx, kxy, kyy, metadata


def _porosity_field(
    z_background: np.ndarray,
    *,
    resolution: float,
    length_x: float,
    values: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the maintained globally Kozeny-Carman-anchored porosity field."""
    smooth_relative = float(values["eps_smooth_rel"])
    if smooth_relative > 0:
        sigma = max(smooth_relative * length_x / resolution, 1.0)
        z_porosity = _convolve(z_background, _gaussian_kernel(sigma, sigma))
    else:
        z_porosity = z_background
    centered = z_porosity - np.mean(z_porosity)
    centered /= max(float(np.std(centered, ddof=1)), np.finfo(np.float64).eps)
    texture = centered - np.mean(centered)
    texture /= max(math.sqrt(float(np.mean(texture**2))), np.finfo(np.float64).eps)
    minimum = float(values["eps_min_global"])
    maximum = float(values["eps_max_global"])
    k_mean = float(values["k_mean"])
    material_factor = float(values["A_rel"]) * k_mean
    low = minimum + 1e-6
    high = maximum - 1e-6
    for _ in range(80):
        middle = 0.5 * (low + high)
        mapped = material_factor * middle**3 / max((1.0 - middle) ** 2, np.finfo(np.float64).eps)
        if mapped > k_mean:
            high = middle
        else:
            low = middle
    reference = 0.5 * (low + high)
    porosity = np.clip(reference + float(values["texture_amp"]) * texture, minimum, maximum)
    if not np.isfinite(porosity).all() or np.any(porosity < minimum) or np.any(porosity > maximum):
        message = "Generated porosity violates configured finite physical bounds."
        raise ValueError(message)
    return porosity, {"reference": reference, "A_mat": material_factor, "minimum": minimum, "maximum": maximum}


def _pressure_boundary(
    x_grid: np.ndarray,
    *,
    values: dict[str, Any],
    random: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the maintained one-dimensional inlet pressure and 2-D adapter field."""
    x_axis = x_grid[0, :]
    normalized_x = (x_axis - np.min(x_axis)) / max(float(np.max(x_axis) - np.min(x_axis)), np.finfo(np.float64).eps)
    shape = np.zeros_like(normalized_x)
    sine_amplitude = float(values["a_sin"])
    if sine_amplitude != 0:
        shape += sine_amplitude * np.sin(2.0 * math.pi * float(values["f_sin"]) * normalized_x + float(values["phi_sin"]))
    gaussian_count = int(float(values["k_gauss"]))
    gaussian_amplitude = float(values["a_gauss"])
    if gaussian_count > 0 and gaussian_amplitude != 0:
        centers = np.linspace(0.0, 1.0, gaussian_count + 2)[1:-1]
        sigma_zero = float(values["sigma_gauss"])
        sigmas = sigma_zero * (1.0 + float(values["gauss_jitter"]) * random.standard_normal(gaussian_count))
        sigmas = np.maximum(sigmas, 0.05 * sigma_zero)
        for center, sigma in zip(centers, sigmas, strict=True):
            shape += gaussian_amplitude / max(gaussian_count, 1) * np.exp(-((normalized_x - center) ** 2) / (2.0 * sigma**2))
    linear_amplitude = float(values["a_lin"])
    if linear_amplitude != 0:
        shape += linear_amplitude * (2.0 * normalized_x - 1.0)
    inlet = np.maximum(float(values["p_inlet_mean"]) * (1.0 + shape), 0.0)
    field = np.zeros_like(x_grid)
    field[0, :] = inlet
    return field, {"minimum": float(np.min(inlet)), "maximum": float(np.max(inlet))}


def generate_spatial_fields(
    domain: dict[str, Any],
    parameters: dict[str, Any],
    *,
    seed: int,
) -> SpatialFields:
    """
    Generate one deterministic spatial field set for a configured case.

    Coordinate matrices use ``meshgrid(x, y)`` orientation. Persisted columns
    are flattened later in deterministic Fortran order.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        message = f"Case seed must be an integer in the uint32 range, got {seed!r}."
        raise ValueError(message)
    values = _validate_parameters(parameters)
    length_x = float(domain["length_x_m"])
    length_y = float(domain["length_y_m"])
    resolution = float(domain["resolution_m"])
    count_x = math.floor(length_x / resolution + 0.5) + 1
    count_y = math.floor(length_y / resolution + 0.5) + 1
    if count_x < MIN_AXIS_POINTS or count_y < MIN_AXIS_POINTS:
        message = "Configured spatial grid must contain at least two points on each axis."
        raise ValueError(message)
    x_axis = np.linspace(0.0, length_x, count_x, dtype=np.float64)
    y_axis = np.linspace(0.0, length_y, count_y, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    random = np.random.default_rng(seed)
    z, z_background, structure_metadata = _structure_fields(
        x_grid,
        y_grid,
        resolution=resolution,
        values=values,
        random=random,
    )
    kappa, kxx, kxy, kyy, permeability_metadata = _permeability_fields(
        z,
        z_background,
        resolution=resolution,
        length_x=length_x,
        values=values,
        random=random,
    )
    porosity, porosity_metadata = _porosity_field(
        z_background,
        resolution=resolution,
        length_x=length_x,
        values=values,
    )
    pressure, pressure_metadata = _pressure_boundary(x_grid, values=values, random=random)
    columns = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": kxx,
        "Kxy": kxy,
        "Kyy": kyy,
        "eps": porosity,
        "p_bc": pressure,
        "kappa": kappa,
        "z": z,
        "z_bg": z_background,
    }
    if not all(array.shape == x_grid.shape and np.isfinite(array).all() for array in columns.values()):
        message = "Generated spatial fields must share one finite Cartesian shape."
        raise ValueError(message)
    metadata = {
        "generator_version": config_contract.GENERATOR_VERSION,
        "random_stream": "numpy.default_rng",
        "seed": seed,
        "geometry": {
            "length_x_m": length_x,
            "length_y_m": length_y,
            "resolution_m": resolution,
            "nx": count_x,
            "ny": count_y,
        },
        "structure": structure_metadata,
        "permeability": permeability_metadata,
        "porosity": porosity_metadata,
        "pressure_boundary": pressure_metadata,
    }
    return SpatialFields(shape=x_grid.shape, columns=columns, metadata=metadata)
