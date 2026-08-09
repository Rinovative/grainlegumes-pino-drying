"""
===============================================================================
generation_schedule.py
===============================================================================
Generate one deterministic compositional inlet schedule.
Responsibilities:
  - Combine smooth, event, and trend components through one sparse simplex model
  - Generate temperature and humidity-ratio schedules from independent substreams
  - Derive relative humidity thermodynamically and classify only after generation
Design principles:
  - Schedule class is metadata and never selects a production implementation path
  - All event details and activation decisions are label-derived provenance
  - The planned complete 0..168-hour schedule owns the reference temperature
This module does NOT:
  - Define material ranges, COMSOL interpolation tags, or alternate class generators
  - Infer values from an early solver stop
===============================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import generation_config as config_contract

_EVENT_SELECTION_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Schedule:
    """One planned regular inlet schedule and realized provenance."""

    values: np.ndarray
    metadata: dict[str, Any]
    derived_scalars: dict[str, float]


def saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Return water saturation pressure using the maintained Magnus relation."""
    values = np.asarray(temperature, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        msg = "Temperature must be finite and positive for vapor-pressure conversion."
        raise ValueError(msg)
    temperature_c = values - 273.15
    denominator = temperature_c + 243.04
    if np.any(denominator <= 0):
        msg = "Temperature lies outside the maintained Magnus relation domain."
        raise ValueError(msg)
    return 610.94 * np.exp(17.625 * temperature_c / denominator)


def humidity_ratio_to_relative_humidity(
    humidity_ratio: np.ndarray,
    temperature: np.ndarray,
    *,
    pressure: float,
) -> np.ndarray:
    """Convert humidity ratio to relative humidity at one reference pressure."""
    omega = np.asarray(humidity_ratio, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if omega.shape != temperature.shape or not np.isfinite(omega).all() or np.any(omega < 0):
        msg = "Humidity ratio and temperature must be aligned finite arrays with omega >= 0."
        raise ValueError(msg)
    if not math.isfinite(pressure) or pressure <= 0:
        msg = "Reference pressure must be finite and positive."
        raise ValueError(msg)
    vapor_pressure = pressure * omega / (0.621945 + omega)
    relative_humidity = vapor_pressure / saturation_vapor_pressure(temperature)
    if not np.isfinite(relative_humidity).all():
        msg = "Thermodynamic relative-humidity conversion produced non-finite values."
        raise ValueError(msg)
    return relative_humidity


def _normalized(value: np.ndarray) -> np.ndarray:
    """Return one zero-mean component with maximum absolute value one."""
    centered = np.asarray(value, dtype=np.float64) - np.mean(value)
    maximum = float(np.max(np.abs(centered)))
    if maximum <= np.finfo(np.float64).eps:
        return np.zeros_like(centered)
    return centered / maximum


def _smooth_component(
    time: np.ndarray,
    *,
    timescale_rel: float,
    random: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one smooth harmonic realization and its phases."""
    horizon = float(time[-1] - time[0])
    scale = max(timescale_rel * horizon, 1.0)
    harmonic_count = 3
    phases = random.uniform(0.0, 2.0 * math.pi, size=harmonic_count)
    coefficients = random.standard_normal(harmonic_count) / np.arange(1, harmonic_count + 1, dtype=np.float64)
    component = np.zeros_like(time)
    for index in range(harmonic_count):
        period = scale / (index + 1)
        component += coefficients[index] * np.sin(2.0 * math.pi * time / period + phases[index])
    return _normalized(component), {
        "harmonic_count": harmonic_count,
        "phases": phases.tolist(),
        "coefficients": coefficients.tolist(),
        "timescale": scale,
    }


def _event_component(
    time: np.ndarray,
    *,
    count: int,
    duration_rel: float,
    width_rel: float,
    random: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate deterministic step-or-pulse events within the planned horizon."""
    if count <= 0:
        return np.zeros_like(time), []
    horizon = float(time[-1] - time[0])
    duration = max(duration_rel * horizon, 1.0)
    width = max(width_rel * horizon, 0.25)
    positions = np.sort(random.uniform(time[0] + 1.0, time[-1] - 1.0, size=count))
    component = np.zeros_like(time)
    details: list[dict[str, Any]] = []
    for position in positions:
        sign = 1.0 if float(random.random()) < _EVENT_SELECTION_THRESHOLD else -1.0
        event_type = "step" if float(random.random()) < _EVENT_SELECTION_THRESHOLD else "pulse"
        jitter = float(random.uniform(-0.5, 0.5))
        center = float(np.clip(position + jitter, time[0], time[-1]))
        if event_type == "pulse":
            contribution = np.exp(-0.5 * ((time - center) / width) ** 2)
        else:
            end = min(center + duration, float(time[-1]))
            contribution = 0.5 * (np.tanh((time - center) / width) - np.tanh((time - end) / width))
        component += sign * contribution
        details.append(
            {
                "position": center,
                "sign": int(sign),
                "type": event_type,
                "duration": duration,
                "width": width,
                "local_jitter": jitter,
            }
        )
    return _normalized(component), details


def _components(
    time: np.ndarray,
    values: Mapping[str, Any],
    *,
    random: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate all three potential schedule components in one implementation."""
    smooth, smooth_details = _smooth_component(
        time,
        timescale_rel=float(values["schedule.timescale_rel"]),
        random=random,
    )
    events, event_details = _event_component(
        time,
        count=int(values["schedule.event_count"]),
        duration_rel=float(values["schedule.event_duration_rel"]),
        width_rel=float(values["schedule.event_width_rel"]),
        random=random,
    )
    trend = _normalized(np.linspace(-1.0, 1.0, time.size, dtype=np.float64))
    return {"smooth": smooth, "event": events, "trend": trend}, {
        "smooth": smooth_details,
        "events": event_details,
    }


def _activation(weights: Mapping[str, float], *, random: np.random.Generator, event_count: int) -> dict[str, bool]:
    """Draw one deterministic sparse activation mask from simplex weights."""
    active = {name: bool(weight > 0 and float(random.random()) < weight) for name, weight in weights.items()}
    if event_count <= 0:
        active["event"] = False
    return active


def _compose(
    components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    active: Mapping[str, bool],
) -> np.ndarray:
    """Combine active weighted components and bound the latent magnitude."""
    result = np.zeros_like(next(iter(components.values())))
    for name in ("smooth", "event", "trend"):
        if active[name]:
            result += float(weights[name]) * components[name]
    maximum = float(np.max(np.abs(result)))
    return result if maximum <= 1.0 else result / maximum


def _schedule_class(active: Mapping[str, bool], *, temperature_amplitude: float, humidity_amplitude: float) -> str:
    """Derive the realized schedule class only after component generation."""
    names = [name for name in ("smooth", "event", "trend") if active[name]]
    if (temperature_amplitude == 0 and humidity_amplitude == 0) or not names:
        return "constant"
    if len(names) > 1:
        return "mixed"
    return names[0]


def generate_schedule(
    values: Mapping[str, Any],
    time_contract: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    seeds: Mapping[str, int],
) -> Schedule:
    """Generate the single finalized mixed schedule on regular hourly nodes."""
    if set(seeds) != {"schedule_shared", "schedule_independent"}:
        msg = "Schedule generation requires exact shared and independent seeds."
        raise ValueError(msg)
    time = np.asarray(time_contract["regular_times"], dtype=np.float64)
    if time.shape != (169,) or not np.array_equal(time, np.arange(169, dtype=np.float64)):
        msg = "Schedule time contract must contain regular hourly nodes 0..168 h."
        raise ValueError(msg)
    weights_raw = values["schedule.component_weights"]
    if not isinstance(weights_raw, Mapping) or tuple(weights_raw) != ("smooth", "event", "trend"):
        msg = "Schedule component weights must be the ordered smooth/event/trend simplex."
        raise ValueError(msg)
    weights = {name: float(weights_raw[name]) for name in weights_raw}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        msg = "Schedule component weights must be finite, non-negative, and sum to one."
        raise ValueError(msg)
    temperature_amplitude = float(values["T_in_amp"])
    humidity_amplitude = float(values["omega_in_amp"])
    correlation = float(values["schedule.corr"])
    if temperature_amplitude < 0 or humidity_amplitude < 0 or not -1 <= correlation <= 1:
        msg = "Schedule amplitudes must be non-negative and schedule.corr must lie in [-1, 1]."
        raise ValueError(msg)
    shared_random = np.random.default_rng(seeds["schedule_shared"])
    independent_random = np.random.default_rng(seeds["schedule_independent"])
    active = _activation(weights, random=shared_random, event_count=int(values["schedule.event_count"]))
    shared_components, shared_details = _components(time, values, random=shared_random)
    independent_components, independent_details = _components(time, values, random=independent_random)
    shared_latent = _compose(shared_components, weights, active)
    independent_latent = _compose(independent_components, weights, active)
    humidity_latent = correlation * shared_latent + math.sqrt(max(0.0, 1.0 - correlation**2)) * independent_latent
    humidity_maximum = float(np.max(np.abs(humidity_latent)))
    if humidity_maximum > 1.0:
        humidity_latent /= humidity_maximum
    temperature = float(values["T_in_base"]) + temperature_amplitude * shared_latent
    humidity_ratio = float(values["omega_in_base"]) + humidity_amplitude * humidity_latent
    if not np.isfinite(temperature).all() or not np.isfinite(humidity_ratio).all():
        msg = "Generated schedule contains non-finite values."
        raise ValueError(msg)
    if np.any(temperature > float(fixed["T_in_max"])):
        msg = "Generated inlet temperature exceeds 308.15 K."
        raise ValueError(msg)
    if np.any((humidity_ratio < float(fixed["omega_min"])) | (humidity_ratio > float(fixed["omega_max"]))):
        msg = "Generated inlet humidity ratio violates configured bounds."
        raise ValueError(msg)
    relative_humidity = humidity_ratio_to_relative_humidity(
        humidity_ratio,
        temperature,
        pressure=float(fixed["p_ref"]),
    )
    if np.any((relative_humidity < float(fixed["phi_clip_min"])) | (relative_humidity > float(fixed["phi_clip_max"]))):
        msg = "Thermodynamically derived inlet relative humidity violates configured bounds."
        raise ValueError(msg)
    values_array = np.column_stack((time, temperature, humidity_ratio, relative_humidity)).astype(np.float64, copy=False)
    reference_temperature = float(np.trapezoid(temperature, time) / (time[-1] - time[0]))
    component_amplitudes = {
        "temperature": {name: temperature_amplitude * weights[name] if active[name] else 0.0 for name in weights},
        "humidity_ratio": {name: humidity_amplitude * weights[name] if active[name] else 0.0 for name in weights},
    }
    return Schedule(
        values=values_array,
        metadata={
            "generator_kind": "compositional_mixed",
            "generator_version": config_contract.GENERATOR_VERSION,
            "interpolation": "linear",
            "schedule_class": _schedule_class(
                active,
                temperature_amplitude=temperature_amplitude,
                humidity_amplitude=humidity_amplitude,
            ),
            "component_weights": weights,
            "component_active": active,
            "realized_component_amplitudes": component_amplitudes,
            "schedule.corr": correlation,
            "seeds": dict(seeds),
            "shared_realization": shared_details,
            "independent_realization": independent_details,
            "column_order": ["t", "T_in_bc", "omega_in_bc", "phi_in_bc"],
            "humidity_formula": "phi_in_bc=p_ref*omega_in_bc/(0.621945+omega_in_bc)/p_sat(T_in_bc)",
            "humidity_conversion_owner": "generation_schedule",
            "conversion_pressure": {
                "name": "p_ref",
                "value": float(fixed["p_ref"]),
                "unit": "Pa",
                "owner": "package_fixed",
            },
            "saturation_pressure_formula": "Magnus_610.94_17.625_243.04",
            "planned_interval": [float(time[0]), float(time[-1])],
            "units": {
                "values": {"t": "h", "T_in_bc": "K", "omega_in_bc": "kg/kg", "phi_in_bc": "1"},
                "diagnostics": {
                    "smooth.phases": "rad",
                    "smooth.timescale": "h",
                    "events.position": "h",
                    "events.duration": "h",
                    "events.width": "h",
                    "events.local_jitter": "h",
                    "planned_interval": "h",
                },
            },
        },
        derived_scalars={"T_in_ref": reference_temperature},
    )
