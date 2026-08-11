"""
===============================================================================
generation_cases_schedule.py
===============================================================================
Generate one deterministic compositional inlet schedule.
Responsibilities:
  - Combine smooth, event, and trend components through one sparse simplex model
  - Generate temperature and humidity-ratio schedules from independent substreams
  - Derive relative humidity thermodynamically and classify only after generation
Design principles:
  - Schedule class is metadata and never selects a production implementation path
  - All event details and activation decisions are label-derived provenance
  - The configured complete schedule owns only time-dependent inlet forcing
This module does NOT:
  - Define material ranges, COMSOL interpolation tags, or alternate class generators
  - Infer values from an early solver stop
===============================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_cases_seeding as seeding

_EVENT_SELECTION_THRESHOLD = 0.5
_MAX_SCHEDULE_ATTEMPTS = 32
_MINIMUM_SCHEDULE_NODES = 2
SCHEDULE_DIAGNOSTIC_UNITS: Final = MappingProxyType(
    {
        "min_T_in_bc": "K",
        "max_T_in_bc": "K",
        "min_omega_in_bc": "kg/kg",
        "max_omega_in_bc": "kg/kg",
        "min_phi_in_bc": "1",
        "max_phi_in_bc": "1",
        "min_phi_source_air": "1",
        "max_phi_source_air": "1",
        "min_heater_temperature_rise": "K",
        "schedule_rejection_count": "1",
        "schedule_acceptance_attempt": "1",
    }
)


@dataclass(frozen=True, slots=True)
class Schedule:
    """One planned regular inlet schedule and realized provenance."""

    values: np.ndarray
    metadata: dict[str, Any]

    @property
    def diagnostics(self) -> dict[str, float | int]:
        """Return the canonical generated-schedule diagnostic values."""
        return {name: self.metadata[name] for name in SCHEDULE_DIAGNOSTIC_UNITS}


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


def _attempt_seed(seed: int, *, attempt: int) -> int:
    """Return the original seed first, then label-derived retry streams."""
    if attempt == 1:
        return seed
    return seeding.derive_seed(seed, "schedule_retry", str(attempt))


def _candidate_schedule(
    time: np.ndarray,
    values: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    temperature_amplitude: float,
    humidity_amplitude: float,
    correlation: float,
    seeds: Mapping[str, int],
    attempt: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool], dict[str, Any], dict[str, Any], dict[str, int]]:
    """Realize one complete unclipped schedule candidate from deterministic streams."""
    attempt_seeds = {name: _attempt_seed(seed, attempt=attempt) for name, seed in seeds.items()}
    shared_random = np.random.default_rng(attempt_seeds["schedule_shared"])
    independent_random = np.random.default_rng(attempt_seeds["schedule_independent"])
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
    return temperature, humidity_ratio, active, shared_details, independent_details, attempt_seeds


def _feasibility_reason(
    temperature: np.ndarray,
    humidity_ratio: np.ndarray,
    phi_in: np.ndarray,
    phi_source: np.ndarray,
    *,
    ambient_temperature: float,
    fixed: Mapping[str, Any],
) -> str | None:
    """Return the first complete-schedule heater feasibility violation."""
    arrays = (temperature, humidity_ratio, phi_in, phi_source)
    if not all(np.isfinite(array).all() for array in arrays):
        return "derived schedule quantities are not all finite"
    if np.any((temperature < float(fixed["T_in_min"])) | (temperature > float(fixed["T_in_max"]))):
        return "T_in_bc violates the configured inlet-temperature envelope"
    if np.any((humidity_ratio < float(fixed["omega_min"])) | (humidity_ratio > float(fixed["omega_max"]))):
        return "omega_in_bc violates the selected source-air engineering envelope"
    if np.any(temperature < ambient_temperature):
        return "T_in_bc is below T_amb in a heater-only apparatus"
    if np.any(humidity_ratio <= 0.0):
        return "omega_in_bc is not strictly positive"
    if np.any((phi_source <= 0.0) | (phi_source > 1.0)):
        return "phi_source_air lies outside (0, 1]"
    phi_minimum = float(fixed["phi_operational_min"])
    phi_maximum = float(fixed["phi_operational_max"])
    if np.any((phi_in < phi_minimum) | (phi_in > phi_maximum)):
        return f"phi_in_bc violates the configured operating envelope [{phi_minimum}, {phi_maximum}]"
    return None


def generate_schedule(
    values: Mapping[str, Any],
    time_contract: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    seeds: Mapping[str, int],
) -> Schedule:
    """Generate the single finalized mixed schedule on configured regular nodes."""
    if set(seeds) != {"schedule_shared", "schedule_independent"}:
        msg = "Schedule generation requires exact shared and independent seeds."
        raise ValueError(msg)
    time = np.asarray(time_contract["regular_times"], dtype=np.float64)
    interval = float(time_contract["interval"])
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(interval)) * 16
    if (
        time.ndim != 1
        or time.size < _MINIMUM_SCHEDULE_NODES
        or not np.isfinite(time).all()
        or not np.allclose(np.diff(time), interval, rtol=0.0, atol=tolerance)
    ):
        msg = "Schedule time contract must contain at least two finite, regularly spaced configured nodes."
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
    duration_relative = float(values["schedule.event_duration_rel"])
    width_relative = float(values["schedule.event_width_rel"])
    if temperature_amplitude < 0 or humidity_amplitude < 0 or not -1 <= correlation <= 1:
        msg = "Schedule amplitudes must be non-negative and schedule.corr must lie in [-1, 1]."
        raise ValueError(msg)
    if duration_relative < 2.0 * width_relative:
        msg = "Schedule event duration must be at least twice the transition width."
        raise ValueError(msg)
    ambient_temperature = float(values["T_amb"])
    rejection_reasons: list[str] = []
    for acceptance_attempt in range(1, _MAX_SCHEDULE_ATTEMPTS + 1):
        temperature, humidity_ratio, active, shared_details, independent_details, attempt_seeds = _candidate_schedule(
            time,
            values,
            weights,
            temperature_amplitude=temperature_amplitude,
            humidity_amplitude=humidity_amplitude,
            correlation=correlation,
            seeds=seeds,
            attempt=acceptance_attempt,
        )
        try:
            relative_humidity = humidity_ratio_to_relative_humidity(
                humidity_ratio,
                temperature,
                pressure=float(fixed["p_ref"]),
            )
            source_relative_humidity = humidity_ratio_to_relative_humidity(
                humidity_ratio,
                np.full_like(temperature, ambient_temperature),
                pressure=float(fixed["p_ref"]),
            )
        except ValueError as error:
            rejection_reasons.append(str(error))
            continue
        reason = _feasibility_reason(
            temperature,
            humidity_ratio,
            relative_humidity,
            source_relative_humidity,
            ambient_temperature=ambient_temperature,
            fixed=fixed,
        )
        if reason is None:
            break
        rejection_reasons.append(reason)
    else:
        message = (
            f"No feasible complete heater-only schedule after {_MAX_SCHEDULE_ATTEMPTS} deterministic attempts; last_reason={rejection_reasons[-1]!r}."
        )
        raise ValueError(message)
    values_array = np.column_stack((time, temperature, humidity_ratio, relative_humidity)).astype(np.float64, copy=False)
    component_amplitudes = {
        "temperature": {name: temperature_amplitude * weights[name] if active[name] else 0.0 for name in weights},
        "humidity_ratio": {name: humidity_amplitude * weights[name] if active[name] else 0.0 for name in weights},
    }
    diagnostics = {
        "min_T_in_bc": float(np.min(temperature)),
        "max_T_in_bc": float(np.max(temperature)),
        "min_omega_in_bc": float(np.min(humidity_ratio)),
        "max_omega_in_bc": float(np.max(humidity_ratio)),
        "min_phi_in_bc": float(np.min(relative_humidity)),
        "max_phi_in_bc": float(np.max(relative_humidity)),
        "min_phi_source_air": float(np.min(source_relative_humidity)),
        "max_phi_source_air": float(np.max(source_relative_humidity)),
        "min_heater_temperature_rise": float(np.min(temperature - ambient_temperature)),
        "schedule_rejection_count": len(rejection_reasons),
        "schedule_acceptance_attempt": acceptance_attempt,
    }
    return Schedule(
        values=values_array,
        metadata={
            "generator_kind": "compositional_mixed",
            "generator_version": seeding.GENERATOR_VERSION,
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
            "accepted_attempt_seeds": attempt_seeds,
            "schedule_rejection_reasons": rejection_reasons,
            **diagnostics,
            "shared_realization": shared_details,
            "independent_realization": independent_details,
            "column_order": list(profiles.SCHEDULE_FIELDS),
            "heater_physics": "humidity_ratio_conserved_across_ambient_air_heater",
            "source_air_humidity_ratio": "omega_source_air(t)=omega_in_bc(t)",
            "humidity_formula": "phi_in_bc=p_ref*omega_in_bc/(0.621945+omega_in_bc)/p_sat(T_in_bc)",
            "source_humidity_formula": "phi_source_air=p_ref*omega_in_bc/(0.621945+omega_in_bc)/p_sat(T_amb)",
            "phi_source_air_usage": "validation_and_provenance_only",
            "humidity_conversion_owner": "generation_schedule",
            "conversion_pressure": {
                "name": "p_ref",
                "value": float(fixed["p_ref"]),
                "unit": "Pa",
                "owner": "package_fixed",
            },
            "saturation_pressure_formula": "Magnus_610.94_17.625_243.04",
            "relative_humidity_operational_bounds": [
                float(fixed["phi_operational_min"]),
                float(fixed["phi_operational_max"]),
            ],
            "oswin_numerical_clip_bounds": [
                float(fixed["phi_clip_min"]),
                float(fixed["phi_clip_max"]),
            ],
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
                    **SCHEDULE_DIAGNOSTIC_UNITS,
                },
            },
        },
    )
