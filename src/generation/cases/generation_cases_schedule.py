"""
===============================================================================
generation_cases_schedule.py
===============================================================================
Generate one deterministic grid-resolved temporal inlet schedule.
Responsibilities:
  - Compose dedicated smooth, event, and trend processes on the regular time grid
  - Realize exact temperature and humidity-ratio amplitude and correlation contracts
  - Derive relative humidity thermodynamically and report schedule-quality evidence
  - Build physical COMSOL startup and boundary interpolation tables
Design principles:
  - One temporal process family serves natural, parameter-OOD, and stress supports
  - Simplex weights mean relative component contribution exactly once
  - Temporal generation remains independent of every spatial-field implementation
This module does NOT:
  - Generate pressure or other spatial fields, define supports, or add learned channels
  - Infer values from an early solver stop
===============================================================================
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_cases_seeding as seeding

_MAX_SCHEDULE_ATTEMPTS = 32
_MINIMUM_SCHEDULE_NODES = 2
_TABLE_RANK = 2
_GAUSSIAN_KERNEL_STANDARD_DEVIATIONS = 4.0
_RANDOM_BINARY_THRESHOLD = 0.5
_MINIMUM_LAG1_NODES = 3
_INTERVAL_BOUND_COUNT = 2
_DRY_AIR_TO_WATER_MASS_RATIO = 0.621945
_MAGNUS_BASE_PRESSURE_PA = 610.94
_MAGNUS_EXPONENT = 17.625
_MAGNUS_TEMPERATURE_OFFSET_C = 243.04
_RELATIVE_HUMIDITY_ROOT_TOLERANCE = 256.0 * np.finfo(np.float64).eps
SCHEDULE_GENERATOR_VERSION: Final = 1
COMSOL_BOUNDARY_HANDOFF_VERSION: Final = 1
CORRELATION_TOLERANCE: Final = 2.0e-12
MINIMUM_SMOOTH_SCALE_INTERVALS: Final = 4.0
MINIMUM_EVENT_WIDTH_INTERVALS: Final = 2.0
MINIMUM_EVENT_DURATION_INTERVALS: Final = 4.0
SCHEDULE_DIAGNOSTIC_UNITS: Final = MappingProxyType(
    {
        "mean_T_in_bc": "K",
        "min_T_in_bc": "K",
        "max_T_in_bc": "K",
        "peak_to_peak_T_in_bc": "K",
        "max_abs_deviation_T_in_base": "K",
        "configured_T_in_amp": "K",
        "realized_T_in_amp": "K",
        "T_in_amp_realization_ratio": "1",
        "rms_rate_T_in_bc": "K/h",
        "max_abs_rate_T_in_bc": "K/h",
        "lag1_autocorrelation_T_in_bc": "1",
        "total_variation_per_horizon_T_in_bc": "K/h",
        "mean_omega_in_bc": "kg/kg",
        "min_omega_in_bc": "kg/kg",
        "max_omega_in_bc": "kg/kg",
        "peak_to_peak_omega_in_bc": "kg/kg",
        "max_abs_deviation_omega_in_base": "kg/kg",
        "configured_omega_in_amp": "kg/kg",
        "realized_omega_in_amp": "kg/kg",
        "omega_in_amp_realization_ratio": "1",
        "rms_rate_omega_in_bc": "kg/(kg*h)",
        "max_abs_rate_omega_in_bc": "kg/(kg*h)",
        "lag1_autocorrelation_omega_in_bc": "1",
        "total_variation_per_horizon_omega_in_bc": "kg/(kg*h)",
        "mean_phi_in_bc": "1",
        "min_phi_in_bc": "1",
        "max_phi_in_bc": "1",
        "peak_to_peak_phi_in_bc": "1",
        "rms_rate_phi_in_bc": "1/h",
        "max_abs_rate_phi_in_bc": "1/h",
        "lag1_autocorrelation_phi_in_bc": "1",
        "total_variation_per_horizon_phi_in_bc": "1/h",
        "configured_T_omega_correlation": "1",
        "realized_T_omega_correlation": "1",
        "absolute_T_omega_correlation_error": "1",
        "schedule_interval": "h",
        "smooth_scale_hours": "h",
        "smooth_scale_intervals": "1",
        "minimum_event_width_hours": "h",
        "minimum_event_width_intervals": "1",
        "event_duration_hours": "h",
        "event_duration_intervals": "1",
        "constant_T_in_bc": "1",
        "constant_omega_in_bc": "1",
        "min_phi_source_air": "1",
        "max_phi_source_air": "1",
        "min_heater_temperature_rise": "K",
        "schedule_rejection_count": "1",
        "schedule_acceptance_attempt": "1",
    }
)


class _DegenerateScheduleError(ValueError):
    """Report one retryable zero-variance temporal realization."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """One planned regular inlet schedule and realized provenance."""

    values: np.ndarray
    metadata: dict[str, Any]

    @property
    def diagnostics(self) -> dict[str, float | int | bool | None]:
        """Return the canonical generated-schedule diagnostic values."""
        return {name: self.metadata[name] for name in SCHEDULE_DIAGNOSTIC_UNITS}


@dataclass(frozen=True, slots=True)
class ComsolBoundarySchedule:
    """One final COMSOL interpolation table and its canonical provenance."""

    values: np.ndarray
    metadata: dict[str, Any]


def _startup_ramp_policy(value: Mapping[str, Any], *, regular_interval: float) -> tuple[bool, float]:
    """Return one validated startup-ramp policy."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"enabled", "duration_h"}
        or not isinstance(value["enabled"], bool)
        or isinstance(value["duration_h"], bool)
    ):
        msg = "Startup-ramp policy must contain exact boolean enabled and numeric duration_h values."
        raise ValueError(msg)
    duration_h = float(value["duration_h"])
    if not math.isfinite(duration_h) or not 0.0 < duration_h < regular_interval:
        msg = "Startup-ramp duration_h must be finite, positive, and shorter than the regular interval."
        raise ValueError(msg)
    return value["enabled"], duration_h


def _operational_bounds(metadata: Mapping[str, Any], name: str) -> tuple[float, float]:
    """Return one validated two-sided operational interval from schedule metadata."""
    raw = metadata.get(name)
    if not isinstance(raw, list) or len(raw) != _INTERVAL_BOUND_COUNT or any(isinstance(item, bool) for item in raw):
        msg = f"Schedule metadata {name!r} must contain two numeric bounds."
        raise ValueError(msg)
    lower, upper = (float(item) for item in raw)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        msg = f"Schedule metadata {name!r} must contain a finite increasing interval."
        raise ValueError(msg)
    return lower, upper


def _startup_ramp_metadata(
    canonical_start: np.ndarray,
    *,
    enabled: bool,
    duration_h: float,
    initial_temperature: float,
    pressure: float,
) -> dict[str, Any]:
    """Return truthful startup-ramp policy evidence."""
    canonical_humidity_ratio = float(canonical_start[2])
    boundary_start_temperature = initial_temperature if enabled else float(canonical_start[1])
    startup_phi = float(
        humidity_ratio_to_relative_humidity(
            np.asarray([canonical_humidity_ratio], dtype=np.float64),
            np.asarray([boundary_start_temperature], dtype=np.float64),
            pressure=pressure,
        )[0]
    )
    if not enabled:
        return {
            "enabled": False,
            "duration_h": duration_h,
            "temperature_start_policy": "disabled_retain_canonical_schedule",
            "initial_temperature_K": initial_temperature,
            "canonical_start_humidity_ratio_kg_per_kg": canonical_humidity_ratio,
            "startup_relative_humidity": startup_phi,
            "humidity_start_policy": "disabled_derive_from_retained_canonical_primitives",
            "rejoin_policy": "not_applicable",
        }
    return {
        "enabled": True,
        "duration_h": duration_h,
        "temperature_start_policy": "use_initial_temperature_exactly",
        "initial_temperature_K": initial_temperature,
        "canonical_start_humidity_ratio_kg_per_kg": canonical_humidity_ratio,
        "startup_relative_humidity": startup_phi,
        "humidity_start_policy": "preserve_canonical_omega_in_bc_and_derive_phi_in_bc",
        "rejoin_policy": "interpolate_canonical_temperature_and_humidity_ratio_then_derive_phi_in_bc",
    }


def _boundary_handoff_metadata(
    canonical: Schedule,
    *,
    startup_metadata: Mapping[str, Any],
    regular_interval: float,
    rejoin_row: np.ndarray | None,
) -> dict[str, Any]:
    """Describe how the canonical schedule becomes the COMSOL interpolation table."""
    canonical_values = np.asarray(canonical.values, dtype=np.float64)
    return {
        "handoff_version": COMSOL_BOUNDARY_HANDOFF_VERSION,
        "representation": "comsol_linear_interpolation_table",
        "startup_ramp": dict(startup_metadata),
        "canonical_regular_grid": {
            "start_h": float(canonical_values[0, 0]),
            "stop_h": float(canonical_values[-1, 0]),
            "interval_h": regular_interval,
            "node_count": int(canonical_values.shape[0]),
        },
        "canonical_start_row": [float(value) for value in canonical_values[0]],
        "rejoin_row": None if rejoin_row is None else [float(value) for value in rejoin_row],
        "regular_output_time_policy": "common.time.regular_times_unchanged",
    }


def validate_comsol_boundary_schedule(
    schedule_values: np.ndarray,
    *,
    regular_times: np.ndarray,
    startup_ramp: Mapping[str, Any],
    initial_temperature: float,
    pressure: float,
    metadata: Mapping[str, Any],
) -> None:
    """Validate the primitive COMSOL boundary and continuous derived-RH contract."""
    if not isinstance(metadata, Mapping):
        msg = "COMSOL boundary schedule metadata must be a mapping."
        raise TypeError(msg)
    values = np.asarray(schedule_values, dtype=np.float64)
    regular = np.asarray(regular_times, dtype=np.float64)
    if regular.ndim != 1 or regular.size < _MINIMUM_SCHEDULE_NODES or not np.isfinite(regular).all():
        msg = "COMSOL boundary handoff requires finite canonical regular times."
        raise ValueError(msg)
    differences = np.diff(regular)
    regular_interval = float(differences[0])
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(regular_interval)) * 16
    if (
        not math.isfinite(regular_interval)
        or regular_interval <= 0.0
        or not np.allclose(differences, regular_interval, rtol=0.0, atol=tolerance)
        or regular[0] != 0.0
    ):
        msg = "COMSOL boundary handoff requires a regular canonical grid beginning at t=0."
        raise ValueError(msg)
    enabled, duration_h = _startup_ramp_policy(startup_ramp, regular_interval=regular_interval)
    expected_times = np.concatenate((regular[:1], [duration_h], regular[1:])) if enabled else regular
    if (
        values.ndim != _TABLE_RANK
        or values.shape != (expected_times.size, len(profiles.SCHEDULE_FIELDS))
        or not np.isfinite(values).all()
        or not np.array_equal(values[:, 0], expected_times)
        or np.any(np.diff(values[:, 0]) <= 0.0)
    ):
        msg = "COMSOL boundary schedule has invalid shape, times, ordering, or finite values."
        raise ValueError(msg)
    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        msg = "COMSOL startup temperature source must be finite and physically positive."
        raise ValueError(msg)
    if np.any(values[:, 1] <= 0.0) or np.any(values[:, 2] <= 0.0):
        msg = "COMSOL boundary temperature or humidity ratio is physically invalid."
        raise ValueError(msg)
    temperature_minimum, temperature_maximum = _operational_bounds(
        metadata,
        "temperature_operational_bounds",
    )
    humidity_minimum, humidity_maximum = _operational_bounds(
        metadata,
        "humidity_ratio_operational_bounds",
    )
    phi_minimum, phi_maximum = _operational_bounds(
        metadata,
        "relative_humidity_operational_bounds",
    )
    handoff = metadata.get("boundary_handoff")
    if not isinstance(handoff, Mapping):
        msg = "COMSOL boundary schedule lacks handoff provenance."
        raise TypeError(msg)
    expected_handoff_keys = {
        "handoff_version",
        "representation",
        "startup_ramp",
        "canonical_regular_grid",
        "canonical_start_row",
        "rejoin_row",
        "regular_output_time_policy",
    }
    if set(handoff) != expected_handoff_keys:
        msg = "COMSOL boundary handoff provenance has invalid fields."
        raise ValueError(msg)
    expected_grid_metadata = {
        "start_h": float(regular[0]),
        "stop_h": float(regular[-1]),
        "interval_h": regular_interval,
        "node_count": int(regular.size),
    }
    canonical_start = np.asarray(handoff["canonical_start_row"], dtype=np.float64)
    if (
        handoff["handoff_version"] != COMSOL_BOUNDARY_HANDOFF_VERSION
        or handoff["representation"] != "comsol_linear_interpolation_table"
        or handoff["canonical_regular_grid"] != expected_grid_metadata
        or handoff["regular_output_time_policy"] != "common.time.regular_times_unchanged"
        or canonical_start.shape != (len(profiles.SCHEDULE_FIELDS),)
        or not np.isfinite(canonical_start).all()
        or canonical_start[0] != regular[0]
    ):
        msg = "COMSOL boundary handoff provenance disagrees with the active schedule contract."
        raise ValueError(msg)

    source_phi = humidity_ratio_to_relative_humidity(
        values[:, 2],
        np.full(values.shape[0], initial_temperature, dtype=np.float64),
        pressure=pressure,
    )
    physical_phi_minimum, physical_phi_maximum = derived_relative_humidity_extrema(
        values[:, 1],
        values[:, 2],
        pressure=pressure,
    )
    if (
        np.any((source_phi <= 0.0) | (source_phi > 1.0))
        or np.any(values[:, 1] < initial_temperature)
        or physical_phi_minimum <= 0.0
        or physical_phi_maximum > 1.0
    ):
        msg = "COMSOL boundary schedule violates heater-only or physical humidity constraints."
        raise ValueError(msg)

    if enabled:
        fraction = duration_h / regular_interval
        expected_rejoin = canonical_start + fraction * (values[2] - canonical_start)
        expected_rejoin[0] = duration_h
        rejoin_row = np.asarray(handoff["rejoin_row"], dtype=np.float64)
        expected_startup_metadata = _startup_ramp_metadata(
            canonical_start,
            enabled=True,
            duration_h=duration_h,
            initial_temperature=initial_temperature,
            pressure=pressure,
        )
        if (
            handoff["startup_ramp"] != expected_startup_metadata
            or values[0, 1] != initial_temperature
            or values[0, 2] != canonical_start[2]
            or rejoin_row.shape != (len(profiles.SCHEDULE_FIELDS),)
            or not np.array_equal(values[1], expected_rejoin)
            or not np.array_equal(rejoin_row, expected_rejoin)
            or not np.array_equal(values[2:, 0], regular[1:])
        ):
            msg = "COMSOL startup node, rejoin node, or retained regular nodes are invalid."
            raise ValueError(msg)
        canonical_rows = np.concatenate(
            (canonical_start[np.newaxis, :], values[2:]),
            axis=0,
        )
    else:
        expected_startup_metadata = _startup_ramp_metadata(
            canonical_start,
            enabled=False,
            duration_h=duration_h,
            initial_temperature=initial_temperature,
            pressure=pressure,
        )
        if (
            handoff["startup_ramp"] != expected_startup_metadata
            or handoff["rejoin_row"] is not None
            or not np.array_equal(values[0], canonical_start)
        ):
            msg = "Disabled COMSOL startup ramp must retain the canonical schedule exactly."
            raise ValueError(msg)
        canonical_rows = values

    operating_phi_minimum, operating_phi_maximum = derived_relative_humidity_extrema(
        canonical_rows[:, 1],
        canonical_rows[:, 2],
        pressure=pressure,
    )
    if (
        np.any((canonical_rows[:, 1] < temperature_minimum) | (canonical_rows[:, 1] > temperature_maximum))
        or np.any((canonical_rows[:, 2] < humidity_minimum) | (canonical_rows[:, 2] > humidity_maximum))
        or operating_phi_minimum < phi_minimum
        or operating_phi_maximum > phi_maximum
    ):
        msg = "COMSOL canonical schedule or its continuous derived relative humidity violates persisted operational bounds."
        raise ValueError(msg)


def build_comsol_boundary_schedule(
    canonical: Schedule,
    startup_ramp: Mapping[str, Any],
    *,
    initial_temperature: float,
    pressure: float,
) -> ComsolBoundarySchedule:
    """Apply the configured startup handoff to primitive schedule values."""
    canonical_values = np.asarray(canonical.values, dtype=np.float64)
    if (
        canonical_values.ndim != _TABLE_RANK
        or canonical_values.shape[0] < _MINIMUM_SCHEDULE_NODES
        or canonical_values.shape[1] != len(profiles.SCHEDULE_FIELDS)
        or not np.isfinite(canonical_values).all()
    ):
        msg = "Canonical schedule has invalid shape or values for COMSOL handoff."
        raise ValueError(msg)
    regular_times = canonical_values[:, 0]
    regular_interval = float(regular_times[1] - regular_times[0])
    enabled, duration_h = _startup_ramp_policy(startup_ramp, regular_interval=regular_interval)
    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        msg = "COMSOL startup temperature source must be finite and physically positive."
        raise ValueError(msg)
    rejoin_row: np.ndarray | None = None
    startup_metadata: dict[str, Any]
    if enabled:
        fraction = duration_h / regular_interval
        constructed_rejoin = canonical_values[0] + fraction * (canonical_values[1] - canonical_values[0])
        constructed_rejoin[0] = duration_h
        startup_metadata = _startup_ramp_metadata(
            canonical_values[0],
            enabled=True,
            duration_h=duration_h,
            initial_temperature=initial_temperature,
            pressure=pressure,
        )
        start_row = canonical_values[0].copy()
        start_row[1] = initial_temperature
        handoff_values = np.concatenate(
            (start_row[np.newaxis, :], constructed_rejoin[np.newaxis, :], canonical_values[1:]),
            axis=0,
        )
        rejoin_row = constructed_rejoin
        if not np.array_equal(handoff_values[2:], canonical_values[1:]):
            msg = "COMSOL startup transformation changed retained canonical regular nodes."
            raise RuntimeError(msg)
    else:
        handoff_values = canonical_values.copy()
        startup_metadata = _startup_ramp_metadata(
            canonical_values[0],
            enabled=False,
            duration_h=duration_h,
            initial_temperature=initial_temperature,
            pressure=pressure,
        )
    metadata = copy.deepcopy(canonical.metadata)
    metadata["boundary_handoff"] = _boundary_handoff_metadata(
        canonical,
        startup_metadata=startup_metadata,
        regular_interval=regular_interval,
        rejoin_row=rejoin_row,
    )
    validate_comsol_boundary_schedule(
        handoff_values,
        regular_times=regular_times,
        startup_ramp=startup_ramp,
        initial_temperature=initial_temperature,
        pressure=pressure,
        metadata=metadata,
    )
    return ComsolBoundarySchedule(values=handoff_values, metadata=metadata)


def saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Return water saturation pressure using the maintained Magnus relation."""
    values = np.asarray(temperature, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        msg = "Temperature must be finite and positive for vapor-pressure conversion."
        raise ValueError(msg)
    temperature_c = values - 273.15
    denominator = temperature_c + _MAGNUS_TEMPERATURE_OFFSET_C
    if np.any(denominator <= 0):
        msg = "Temperature lies outside the maintained Magnus relation domain."
        raise ValueError(msg)
    return _MAGNUS_BASE_PRESSURE_PA * np.exp(_MAGNUS_EXPONENT * temperature_c / denominator)


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
    vapor_pressure = pressure * omega / (_DRY_AIR_TO_WATER_MASS_RATIO + omega)
    relative_humidity = vapor_pressure / saturation_vapor_pressure(temperature)
    if not np.isfinite(relative_humidity).all():
        msg = "Thermodynamic relative-humidity conversion produced non-finite values."
        raise ValueError(msg)
    return relative_humidity


def _relative_humidity_stationary_fractions(
    temperature_start: float,
    temperature_end: float,
    humidity_start: float,
    humidity_end: float,
) -> tuple[float, ...]:
    """Return interior extrema fractions for one linear primitive interval."""
    temperature_offset = temperature_start - 273.15 + _MAGNUS_TEMPERATURE_OFFSET_C
    temperature_delta = temperature_end - temperature_start
    humidity_delta = humidity_end - humidity_start
    mass_ratio = _DRY_AIR_TO_WATER_MASS_RATIO
    magnus_slope = _MAGNUS_EXPONENT * _MAGNUS_TEMPERATURE_OFFSET_C
    coefficients = np.asarray(
        (
            mass_ratio * humidity_delta * temperature_offset**2 - magnus_slope * temperature_delta * humidity_start * (mass_ratio + humidity_start),
            2.0 * mass_ratio * humidity_delta * temperature_offset * temperature_delta
            - magnus_slope * temperature_delta * humidity_delta * (mass_ratio + 2.0 * humidity_start),
            mass_ratio * humidity_delta * temperature_delta**2 - magnus_slope * temperature_delta * humidity_delta**2,
        ),
        dtype=np.float64,
    )
    scale = float(np.max(np.abs(coefficients)))
    coefficient_tolerance = _RELATIVE_HUMIDITY_ROOT_TOLERANCE * max(1.0, scale)
    if abs(float(coefficients[2])) <= coefficient_tolerance:
        if abs(float(coefficients[1])) <= coefficient_tolerance:
            return ()
        roots = np.asarray((-float(coefficients[0]) / float(coefficients[1]),), dtype=np.complex128)
    else:
        roots = np.asarray(np.roots(coefficients[::-1]), dtype=np.complex128)
    fractions: list[float] = []
    for root in roots:
        if abs(float(root.imag)) > _RELATIVE_HUMIDITY_ROOT_TOLERANCE * max(1.0, abs(float(root.real))):
            continue
        fraction = float(root.real)
        if -_RELATIVE_HUMIDITY_ROOT_TOLERANCE <= fraction <= 1.0 + _RELATIVE_HUMIDITY_ROOT_TOLERANCE:
            clipped = min(1.0, max(0.0, fraction))
            if 0.0 < clipped < 1.0 and clipped not in fractions:
                fractions.append(clipped)
    return tuple(sorted(fractions))


def derived_relative_humidity_extrema(
    temperature: np.ndarray,
    humidity_ratio: np.ndarray,
    *,
    pressure: float,
) -> tuple[float, float]:
    """
    Return exact-candidate extrema after linear primitive interpolation.

    The logarithmic derivative on each interval reduces to a quadratic.
    Endpoints and every real interior stationary point are evaluated through
    the authoritative psychrometric conversion, without inserting support.
    """
    temperatures = np.asarray(temperature, dtype=np.float64)
    humidity_ratios = np.asarray(humidity_ratio, dtype=np.float64)
    if (
        temperatures.ndim != 1
        or humidity_ratios.shape != temperatures.shape
        or temperatures.size < _MINIMUM_SCHEDULE_NODES
        or not np.isfinite(temperatures).all()
        or not np.isfinite(humidity_ratios).all()
        or np.any(temperatures <= 0.0)
        or np.any(humidity_ratios <= 0.0)
    ):
        msg = "Continuous relative-humidity validation requires aligned positive finite primitive series."
        raise ValueError(msg)
    candidate_temperatures = list(temperatures)
    candidate_humidity_ratios = list(humidity_ratios)
    for index in range(temperatures.size - 1):
        for fraction in _relative_humidity_stationary_fractions(
            float(temperatures[index]),
            float(temperatures[index + 1]),
            float(humidity_ratios[index]),
            float(humidity_ratios[index + 1]),
        ):
            candidate_temperatures.append(float(temperatures[index] + fraction * (temperatures[index + 1] - temperatures[index])))
            candidate_humidity_ratios.append(float(humidity_ratios[index] + fraction * (humidity_ratios[index + 1] - humidity_ratios[index])))
    derived = humidity_ratio_to_relative_humidity(
        np.asarray(candidate_humidity_ratios, dtype=np.float64),
        np.asarray(candidate_temperatures, dtype=np.float64),
        pressure=pressure,
    )
    return float(np.min(derived)), float(np.max(derived))


def relative_humidity_to_humidity_ratio(
    relative_humidity: np.ndarray,
    temperature: np.ndarray,
    *,
    pressure: float,
) -> np.ndarray:
    """Convert relative humidity to humidity ratio at one reference pressure."""
    phi = np.asarray(relative_humidity, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if phi.shape != temperature.shape or not np.isfinite(phi).all() or np.any((phi < 0.0) | (phi > 1.0)):
        msg = "Relative humidity and temperature must be aligned finite arrays with phi in [0, 1]."
        raise ValueError(msg)
    if not math.isfinite(pressure) or pressure <= 0:
        msg = "Reference pressure must be finite and positive."
        raise ValueError(msg)
    vapor_pressure = phi * saturation_vapor_pressure(temperature)
    if np.any(vapor_pressure >= pressure):
        msg = "Relative-humidity conversion requires vapor pressure below reference pressure."
        raise ValueError(msg)
    humidity_ratio = _DRY_AIR_TO_WATER_MASS_RATIO * vapor_pressure / (pressure - vapor_pressure)
    if not np.isfinite(humidity_ratio).all() or np.any(humidity_ratio < 0.0):
        msg = "Thermodynamic humidity-ratio conversion produced invalid values."
        raise ValueError(msg)
    return humidity_ratio


def humidity_ratio_dew_point_temperature(
    humidity_ratio: np.ndarray,
    *,
    pressure: float,
) -> np.ndarray:
    """Return the Magnus dew-point temperature for one humidity ratio."""
    omega = np.asarray(humidity_ratio, dtype=np.float64)
    if not np.isfinite(omega).all() or np.any(omega <= 0.0):
        msg = "Dew-point conversion requires finite strictly positive humidity ratio."
        raise ValueError(msg)
    if not math.isfinite(pressure) or pressure <= 0.0:
        msg = "Reference pressure must be finite and positive."
        raise ValueError(msg)
    vapor_pressure = pressure * omega / (_DRY_AIR_TO_WATER_MASS_RATIO + omega)
    logarithm = np.log(vapor_pressure / _MAGNUS_BASE_PRESSURE_PA)
    denominator = _MAGNUS_EXPONENT - logarithm
    if np.any(denominator <= 0.0):
        msg = "Humidity ratio lies outside the maintained Magnus inverse domain."
        raise ValueError(msg)
    return 273.15 + _MAGNUS_TEMPERATURE_OFFSET_C * logarithm / denominator


def _regular_time(
    time_contract: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Return and validate the configured regular schedule grid."""
    time = np.asarray(time_contract["regular_times"], dtype=np.float64)
    interval = float(time_contract["interval"])
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(interval)) * 16
    if (
        time.ndim != 1
        or time.size < _MINIMUM_SCHEDULE_NODES
        or not math.isfinite(interval)
        or interval <= 0.0
        or not np.isfinite(time).all()
        or not np.allclose(np.diff(time), interval, rtol=0.0, atol=tolerance)
    ):
        msg = "Schedule time contract must contain at least two finite, regularly spaced configured nodes."
        raise ValueError(msg)
    return time, interval


def temporal_resolution(
    values: Mapping[str, Any],
    time_contract: Mapping[str, Any],
) -> dict[str, float]:
    """Return characteristic schedule scales in hours and regular intervals."""
    time, interval = _regular_time(time_contract)
    horizon = float(time[-1] - time[0])
    smooth_scale = float(values["schedule.timescale_rel"]) * horizon
    event_width = float(values["schedule.event_width_rel"]) * horizon
    event_duration = float(values["schedule.event_duration_rel"]) * horizon
    return {
        "schedule_interval": interval,
        "smooth_scale_hours": smooth_scale,
        "smooth_scale_intervals": smooth_scale / interval,
        "minimum_event_width_hours": event_width,
        "minimum_event_width_intervals": event_width / interval,
        "event_duration_hours": event_duration,
        "event_duration_intervals": event_duration / interval,
    }


def validate_temporal_resolution(
    values: Mapping[str, Any],
    time_contract: Mapping[str, Any],
) -> dict[str, float]:
    """Validate every characteristic scale against the configured time grid."""
    resolution = temporal_resolution(values, time_contract)
    if resolution["smooth_scale_intervals"] < MINIMUM_SMOOTH_SCALE_INTERVALS:
        message = f"schedule.timescale_rel resolves below the minimum {MINIMUM_SMOOTH_SCALE_INTERVALS:g} regular intervals."
        raise ValueError(message)
    if resolution["minimum_event_width_intervals"] < MINIMUM_EVENT_WIDTH_INTERVALS:
        message = f"schedule.event_width_rel resolves below the minimum {MINIMUM_EVENT_WIDTH_INTERVALS:g} regular intervals."
        raise ValueError(message)
    if resolution["event_duration_intervals"] < MINIMUM_EVENT_DURATION_INTERVALS:
        message = f"schedule.event_duration_rel resolves below the minimum {MINIMUM_EVENT_DURATION_INTERVALS:g} regular intervals."
        raise ValueError(message)
    if resolution["event_duration_hours"] < 2.0 * resolution["minimum_event_width_hours"]:
        msg = "Schedule event duration must be at least twice the transition width."
        raise ValueError(msg)
    horizon = float(np.asarray(time_contract["regular_times"], dtype=np.float64)[-1]) - float(
        np.asarray(time_contract["regular_times"], dtype=np.float64)[0]
    )
    if resolution["event_duration_hours"] + 2.0 * resolution["minimum_event_width_hours"] >= horizon:
        msg = "Schedule event duration and edge widths must fit within the planned horizon."
        raise ValueError(msg)
    return resolution


def validate_temporal_support_resolution(
    parameter_values: Mapping[str, Mapping[str, Any]],
    time_contract: Mapping[str, Any],
) -> None:
    """Validate every authored temporal support against the regular grid."""
    time, interval = _regular_time(time_contract)
    horizon = float(time[-1] - time[0])
    requirements = {
        "schedule.timescale_rel": MINIMUM_SMOOTH_SCALE_INTERVALS,
        "schedule.event_width_rel": MINIMUM_EVENT_WIDTH_INTERVALS,
        "schedule.event_duration_rel": MINIMUM_EVENT_DURATION_INTERVALS,
    }
    all_bounds: dict[str, list[Mapping[str, Any]]] = {}
    for name, minimum_intervals in requirements.items():
        entry = parameter_values[name]
        bounds = [entry, *entry.get("ood", [])]
        all_bounds[name] = bounds
        for index, support in enumerate(bounds):
            lower_intervals = float(support["lower"]) * horizon / interval
            if lower_intervals < minimum_intervals:
                support_kind = "natural" if index == 0 else f"ood[{index - 1}]"
                message = f"{name} {support_kind} lower bound resolves to {lower_intervals:g} intervals; minimum is {minimum_intervals:g}."
                raise ValueError(message)
    maximum_duration = max(float(bounds["upper"]) for bounds in all_bounds["schedule.event_duration_rel"])
    maximum_width = max(float(bounds["upper"]) for bounds in all_bounds["schedule.event_width_rel"])
    if maximum_duration + 2.0 * maximum_width >= 1.0:
        msg = "Authored event duration and edge-width supports do not fit within the planned horizon."
        raise ValueError(msg)


def _zero_centered(value: np.ndarray) -> np.ndarray:
    """Return one float64 vector centered on the actual schedule nodes."""
    centered = np.asarray(value, dtype=np.float64) - float(np.mean(value))
    return centered - float(np.mean(centered))


def _normalized_component(value: np.ndarray) -> np.ndarray:
    """Return one zero-mean component with maximum absolute value one."""
    centered = _zero_centered(value)
    maximum = float(np.max(np.abs(centered)))
    if maximum <= np.finfo(np.float64).eps:
        return np.zeros_like(centered)
    normalized = centered / maximum
    return _zero_centered(normalized)


def _unit_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    """Return one zero-mean unit-norm vector or reject a degenerate draw."""
    centered = _zero_centered(value)
    norm = float(np.linalg.norm(centered))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps * math.sqrt(centered.size):
        message = f"{label} temporal realization is degenerate."
        raise _DegenerateScheduleError(message)
    return centered / norm


def _amplitude_shape(value: np.ndarray, *, label: str) -> np.ndarray:
    """Return one zero-mean shape with exact unit maximum absolute deviation."""
    centered = _zero_centered(value)
    maximum = float(np.max(np.abs(centered)))
    if not math.isfinite(maximum) or maximum <= np.finfo(np.float64).eps:
        message = f"{label} temporal realization cannot realize a positive amplitude."
        raise _DegenerateScheduleError(message)
    shape = _zero_centered(centered / maximum)
    renormalization = float(np.max(np.abs(shape)))
    if not math.isfinite(renormalization) or renormalization <= np.finfo(np.float64).eps:
        message = f"{label} temporal realization became degenerate during normalization."
        raise _DegenerateScheduleError(message)
    return shape / renormalization


def _gaussian_low_pass(
    excitation: np.ndarray,
    *,
    correlation_intervals: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Apply a reflected Gaussian filter with one e-folding correlation scale."""
    kernel_sigma = correlation_intervals / 2.0
    radius = max(1, math.ceil(_GAUSSIAN_KERNEL_STANDARD_DEVIATIONS * kernel_sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / kernel_sigma) ** 2)
    kernel /= float(np.sum(kernel))
    padded = np.pad(np.asarray(excitation, dtype=np.float64), radius, mode="reflect")
    filtered = np.convolve(padded, kernel, mode="valid")
    if filtered.shape != excitation.shape:
        msg = "Temporal low-pass filtering changed the regular schedule shape."
        raise RuntimeError(msg)
    return filtered, {
        "filter": "reflected_gaussian_convolution",
        "correlation_definition": "e_folding_lag_of_the_ideal_filtered_white_noise_autocorrelation",
        "kernel_sigma_intervals": kernel_sigma,
        "kernel_radius_intervals": radius,
    }


def _smooth_component(
    time: np.ndarray,
    *,
    interval: float,
    timescale_rel: float,
    random: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one dedicated low-pass temporal stochastic realization."""
    horizon = float(time[-1] - time[0])
    scale = timescale_rel * horizon
    correlation_intervals = scale / interval
    excitation = random.standard_normal(time.size)
    filtered, filter_details = _gaussian_low_pass(
        excitation,
        correlation_intervals=correlation_intervals,
    )
    return _normalized_component(filtered), {
        "process": "seeded_white_excitation_then_gaussian_low_pass",
        "correlation_time_hours": scale,
        "correlation_time_intervals": correlation_intervals,
        **filter_details,
    }


def _event_component(
    time: np.ndarray,
    *,
    count: int,
    duration: float,
    width: float,
    random: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate deterministic finite-duration step-like or pulse events."""
    if count <= 0:
        return np.zeros_like(time), []
    start_minimum = float(time[0] + width)
    start_maximum = float(time[-1] - duration - width)
    if start_maximum <= start_minimum:
        msg = "Resolved event duration and widths leave no valid event placement."
        raise ValueError(msg)
    starts = np.sort(random.uniform(start_minimum, start_maximum, size=count))
    component = np.zeros_like(time)
    details: list[dict[str, Any]] = []
    for start in starts:
        end = float(start + duration)
        center = 0.5 * (float(start) + end)
        sign = 1.0 if float(random.random()) < _RANDOM_BINARY_THRESHOLD else -1.0
        event_type = "step" if float(random.random()) < _RANDOM_BINARY_THRESHOLD else "pulse"
        window = 0.5 * (np.tanh((time - float(start)) / width) - np.tanh((time - end) / width))
        contribution = window
        if event_type == "pulse":
            contribution = window * np.exp(-0.5 * ((time - center) / (0.25 * duration)) ** 2)
        component += sign * contribution
        details.append(
            {
                "start": float(start),
                "center": center,
                "end": end,
                "sign": int(sign),
                "type": event_type,
                "duration": duration,
                "width": width,
            }
        )
    normalized = _normalized_component(component)
    if not np.any(normalized):
        msg = "Positive event_count produced a degenerate event realization."
        raise _DegenerateScheduleError(msg)
    return normalized, details


def _trend_component(
    time: np.ndarray,
    *,
    random: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Generate one slow horizon-scale drift without high-frequency structure."""
    coordinate = np.linspace(-1.0, 1.0, time.size, dtype=np.float64)
    direction = 1 if float(random.random()) < _RANDOM_BINARY_THRESHOLD else -1
    curvature = float(random.uniform(-0.25, 0.25))
    component = direction * (coordinate + curvature * (coordinate**2 - float(np.mean(coordinate**2))))
    return _normalized_component(component), {
        "direction": direction,
        "curvature": curvature,
        "scale_fraction_of_horizon": 1.0,
    }


def _components(
    time: np.ndarray,
    values: Mapping[str, Any],
    *,
    interval: float,
    random: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate all dedicated temporal components in one implementation."""
    horizon = float(time[-1] - time[0])
    smooth, smooth_details = _smooth_component(
        time,
        interval=interval,
        timescale_rel=float(values["schedule.timescale_rel"]),
        random=random,
    )
    events, event_details = _event_component(
        time,
        count=int(values["schedule.event_count"]),
        duration=float(values["schedule.event_duration_rel"]) * horizon,
        width=float(values["schedule.event_width_rel"]) * horizon,
        random=random,
    )
    trend, trend_details = _trend_component(time, random=random)
    return {"smooth": smooth, "event": events, "trend": trend}, {
        "smooth": smooth_details,
        "events": event_details,
        "trend": trend_details,
    }


def _component_availability(
    weights: Mapping[str, float],
    *,
    event_count: int,
) -> dict[str, bool]:
    """Return deterministic component availability without stochastic activation."""
    return {
        "smooth": weights["smooth"] > 0.0,
        "event": weights["event"] > 0.0 and event_count > 0,
        "trend": weights["trend"] > 0.0,
    }


def _compose(
    components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine each available normalized component using its simplex weight once."""
    result = np.zeros_like(next(iter(components.values())))
    for name in ("smooth", "event", "trend"):
        result += float(weights[name]) * components[name]
    return _zero_centered(result)


def _correlated_latents(
    shared: np.ndarray,
    independent: np.ndarray,
    *,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | None]]:
    """Construct exact discrete-node Pearson-correlated temporal latents."""
    shared_unit = _unit_vector(shared, label="shared")
    independent_centered = _zero_centered(independent)
    projection = float(np.dot(independent_centered, shared_unit))
    orthogonal = independent_centered - projection * shared_unit
    remainder = max(0.0, 1.0 - correlation**2)
    if remainder <= np.finfo(np.float64).eps:
        humidity_unit = math.copysign(1.0, correlation) * shared_unit
        orthogonal_norm: float | None = None
    else:
        orthogonal_unit = _unit_vector(orthogonal, label="independent orthogonal")
        orthogonal_norm = float(np.linalg.norm(orthogonal))
        humidity_unit = correlation * shared_unit + math.sqrt(remainder) * orthogonal_unit
    return (
        shared_unit,
        _unit_vector(humidity_unit, label="humidity"),
        {
            "method": "discrete_zero_center_standardize_orthogonalize",
            "independent_projection_on_shared": projection,
            "independent_orthogonal_norm": orthogonal_norm,
        },
    )


def _schedule_class(
    available: Mapping[str, bool],
    *,
    temperature_amplitude: float,
    humidity_amplitude: float,
) -> str:
    """Derive a descriptive schedule class after deterministic composition."""
    names = [name for name in ("smooth", "event", "trend") if available[name]]
    if temperature_amplitude == 0.0 and humidity_amplitude == 0.0:
        return "constant"
    if len(names) > 1:
        return "mixed"
    return names[0]


def _attempt_seed(seed: int, *, attempt: int) -> int:
    """Return the original seed first, then schedule-version-bound retry streams."""
    if attempt == 1:
        return seed
    return seeding.derive_seed(
        seed,
        "schedule_algorithm",
        str(SCHEDULE_GENERATOR_VERSION),
        "retry",
        str(attempt),
    )


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
    interval = float(time[1] - time[0])
    attempt_seeds = {name: _attempt_seed(seed, attempt=attempt) for name, seed in seeds.items()}
    shared_random = np.random.default_rng(attempt_seeds["schedule_shared"])
    independent_random = np.random.default_rng(attempt_seeds["schedule_independent"])
    shared_components, shared_details = _components(
        time,
        values,
        interval=interval,
        random=shared_random,
    )
    independent_components, independent_details = _components(
        time,
        values,
        interval=interval,
        random=independent_random,
    )
    shared_latent, humidity_latent, correlation_details = _correlated_latents(
        _compose(shared_components, weights),
        _compose(independent_components, weights),
        correlation=correlation,
    )
    temperature_base = float(values["T_in_base"])
    humidity_base = float(values["omega_in_base"])
    temperature = (
        np.full(time.shape, temperature_base, dtype=np.float64)
        if temperature_amplitude == 0.0
        else temperature_base + temperature_amplitude * _amplitude_shape(shared_latent, label="temperature")
    )
    humidity_ratio = (
        np.full(time.shape, humidity_base, dtype=np.float64)
        if humidity_amplitude == 0.0
        else humidity_base + humidity_amplitude * _amplitude_shape(humidity_latent, label="humidity-ratio")
    )
    available = _component_availability(
        weights,
        event_count=int(values["schedule.event_count"]),
    )
    shared_details["correlation_construction"] = correlation_details
    return (
        temperature,
        humidity_ratio,
        available,
        shared_details,
        independent_details,
        attempt_seeds,
    )


def _discrete_pearson(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    """Return discrete-node Pearson correlation or explicit not-applicable."""
    left_centered = _zero_centered(left)
    right_centered = _zero_centered(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _numeric_tolerance(*values: float) -> float:
    """Return a strict scale-aware float64 tolerance."""
    return np.finfo(np.float64).eps * max(1.0, *(abs(value) for value in values)) * 512.0


def _amplitude_contract_reason(
    signal: np.ndarray,
    *,
    base: float,
    amplitude: float,
    label: str,
) -> str | None:
    """Return one exact mean/amplitude contract violation."""
    if amplitude == 0.0:
        if not np.array_equal(signal, np.full(signal.shape, base, dtype=np.float64)):
            return f"zero {label} amplitude did not produce an exact constant schedule"
        return None
    tolerance = _numeric_tolerance(base, amplitude)
    realized = float(np.max(np.abs(signal - base)))
    if np.array_equal(signal, np.full(signal.shape, signal[0], dtype=np.float64)):
        return f"positive {label} amplitude produced a constant schedule"
    if not math.isclose(float(np.mean(signal)), base, rel_tol=0.0, abs_tol=tolerance):
        return f"{label} temporal mean does not equal its configured base"
    if not math.isclose(realized, amplitude, rel_tol=0.0, abs_tol=tolerance):
        return f"{label} realized amplitude does not equal its configured amplitude"
    return None


def _numerical_contract_reason(
    temperature: np.ndarray,
    humidity_ratio: np.ndarray,
    *,
    temperature_base: float,
    humidity_base: float,
    temperature_amplitude: float,
    humidity_amplitude: float,
    configured_correlation: float,
) -> str | None:
    """Return the first amplitude or correlation invariant violation."""
    for signal, base, amplitude, label in (
        (temperature, temperature_base, temperature_amplitude, "temperature"),
        (humidity_ratio, humidity_base, humidity_amplitude, "humidity-ratio"),
    ):
        reason = _amplitude_contract_reason(
            signal,
            base=base,
            amplitude=amplitude,
            label=label,
        )
        if reason is not None:
            return reason
    realized = _discrete_pearson(temperature, humidity_ratio)
    if temperature_amplitude == 0.0 or humidity_amplitude == 0.0:
        if realized is not None:
            return "zero-variance schedule correlation must be not applicable"
    elif realized is None or abs(realized - configured_correlation) > CORRELATION_TOLERANCE:
        return "realized T-omega correlation does not equal schedule.corr"
    return None


def _feasibility_reason(
    temperature: np.ndarray,
    humidity_ratio: np.ndarray,
    phi_in: np.ndarray,
    phi_source: np.ndarray,
    phi_extrema: tuple[float, float],
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
    if phi_extrema[0] < phi_minimum or phi_extrema[1] > phi_maximum:
        return f"derived phi_in_bc violates the configured continuous operating envelope [{phi_minimum}, {phi_maximum}]"
    return None


def _lag1_autocorrelation(value: np.ndarray) -> float | None:
    """Return lag-one correlation or explicit not-applicable for constants."""
    if value.size < _MINIMUM_LAG1_NODES:
        return None
    return _discrete_pearson(value[:-1], value[1:])


def _series_diagnostics(
    value: np.ndarray,
    *,
    name: str,
    interval: float,
    base_name: str | None = None,
    base: float | None = None,
    amplitude_name: str | None = None,
    configured_amplitude: float | None = None,
) -> dict[str, float | bool | None]:
    """Return the canonical scalar diagnostics for one temporal series."""
    differences = np.diff(value)
    rates = differences / interval
    result: dict[str, float | bool | None] = {
        f"mean_{name}": float(np.mean(value)),
        f"min_{name}": float(np.min(value)),
        f"max_{name}": float(np.max(value)),
        f"peak_to_peak_{name}": float(np.ptp(value)),
        f"rms_rate_{name}": float(np.sqrt(np.mean(rates * rates))),
        f"max_abs_rate_{name}": float(np.max(np.abs(rates))),
        f"lag1_autocorrelation_{name}": _lag1_autocorrelation(value),
        f"total_variation_per_horizon_{name}": float(np.sum(np.abs(differences)) / ((value.size - 1) * interval)),
    }
    if base_name is not None and base is not None and amplitude_name is not None and configured_amplitude is not None:
        realized = float(np.max(np.abs(value - base)))
        result.update(
            {
                f"max_abs_deviation_{base_name}": realized,
                f"configured_{amplitude_name}": configured_amplitude,
                f"realized_{amplitude_name}": realized,
                f"{amplitude_name}_realization_ratio": (None if configured_amplitude == 0.0 else realized / configured_amplitude),
                f"constant_{name}": bool(np.array_equal(value, np.full(value.shape, value[0], dtype=np.float64))),
            }
        )
    return result


def generate_schedule(
    values: Mapping[str, Any],
    time_contract: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    seeds: Mapping[str, int],
) -> Schedule:
    """Generate the single finalized temporal schedule on configured regular nodes."""
    if set(seeds) != {"schedule_shared", "schedule_independent"}:
        msg = "Schedule generation requires exact shared and independent seeds."
        raise ValueError(msg)
    time, interval = _regular_time(time_contract)
    resolution = validate_temporal_resolution(values, time_contract)
    weights_raw = values["schedule.component_weights"]
    if not isinstance(weights_raw, Mapping) or tuple(weights_raw) != ("smooth", "event", "trend"):
        msg = "Schedule component weights must be the ordered smooth/event/trend simplex."
        raise ValueError(msg)
    weights = {name: float(weights_raw[name]) for name in weights_raw}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()) or not math.isclose(
        sum(weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        msg = "Schedule component weights must be finite, non-negative, and sum to one."
        raise ValueError(msg)
    temperature_amplitude = float(values["T_in_amp"])
    humidity_amplitude = float(values["omega_in_amp"])
    correlation = float(values["schedule.corr"])
    event_count_number = float(values["schedule.event_count"])
    if (
        temperature_amplitude < 0.0
        or humidity_amplitude < 0.0
        or not -1.0 <= correlation <= 1.0
        or not event_count_number.is_integer()
        or event_count_number < 0.0
    ):
        msg = "Schedule amplitudes and event count must be non-negative, and schedule.corr must lie in [-1, 1]."
        raise ValueError(msg)
    ambient_temperature = float(values["T_amb"])
    temperature_base = float(values["T_in_base"])
    humidity_base = float(values["omega_in_base"])
    rejection_reasons: list[str] = []
    for acceptance_attempt in range(1, _MAX_SCHEDULE_ATTEMPTS + 1):
        try:
            (
                temperature,
                humidity_ratio,
                available,
                shared_details,
                independent_details,
                attempt_seeds,
            ) = _candidate_schedule(
                time,
                values,
                weights,
                temperature_amplitude=temperature_amplitude,
                humidity_amplitude=humidity_amplitude,
                correlation=correlation,
                seeds=seeds,
                attempt=acceptance_attempt,
            )
        except _DegenerateScheduleError as error:
            rejection_reasons.append(str(error))
            continue
        numerical_reason = _numerical_contract_reason(
            temperature,
            humidity_ratio,
            temperature_base=temperature_base,
            humidity_base=humidity_base,
            temperature_amplitude=temperature_amplitude,
            humidity_amplitude=humidity_amplitude,
            configured_correlation=correlation,
        )
        if numerical_reason is not None:
            message = f"Temporal schedule numerical contract failed: {numerical_reason}."
            raise RuntimeError(message)
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
            relative_humidity_extrema = derived_relative_humidity_extrema(
                temperature,
                humidity_ratio,
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
            relative_humidity_extrema,
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

    values_array = np.column_stack((time, temperature, humidity_ratio)).astype(
        np.float64,
        copy=False,
    )
    realized_correlation = _discrete_pearson(temperature, humidity_ratio)
    correlation_error = None if realized_correlation is None else abs(realized_correlation - correlation)
    relative_humidity_diagnostics = _series_diagnostics(
        relative_humidity,
        name="phi_in_bc",
        interval=interval,
    )
    relative_humidity_diagnostics.update(
        {
            "min_phi_in_bc": relative_humidity_extrema[0],
            "max_phi_in_bc": relative_humidity_extrema[1],
            "peak_to_peak_phi_in_bc": relative_humidity_extrema[1] - relative_humidity_extrema[0],
        }
    )
    diagnostics: dict[str, float | int | bool | None] = {
        **_series_diagnostics(
            temperature,
            name="T_in_bc",
            interval=interval,
            base_name="T_in_base",
            base=temperature_base,
            amplitude_name="T_in_amp",
            configured_amplitude=temperature_amplitude,
        ),
        **_series_diagnostics(
            humidity_ratio,
            name="omega_in_bc",
            interval=interval,
            base_name="omega_in_base",
            base=humidity_base,
            amplitude_name="omega_in_amp",
            configured_amplitude=humidity_amplitude,
        ),
        **relative_humidity_diagnostics,
        "configured_T_omega_correlation": correlation,
        "realized_T_omega_correlation": realized_correlation,
        "absolute_T_omega_correlation_error": correlation_error,
        **resolution,
        "min_phi_source_air": float(np.min(source_relative_humidity)),
        "max_phi_source_air": float(np.max(source_relative_humidity)),
        "min_heater_temperature_rise": float(np.min(temperature - ambient_temperature)),
        "schedule_rejection_count": len(rejection_reasons),
        "schedule_acceptance_attempt": acceptance_attempt,
    }
    if set(diagnostics) != set(SCHEDULE_DIAGNOSTIC_UNITS):
        missing = sorted(set(SCHEDULE_DIAGNOSTIC_UNITS).difference(diagnostics))
        extra = sorted(set(diagnostics).difference(SCHEDULE_DIAGNOSTIC_UNITS))
        message = f"Schedule diagnostic ownership mismatch: missing={missing}, extra={extra}."
        raise RuntimeError(message)
    return Schedule(
        values=values_array,
        metadata={
            "generator_kind": "grid_resolved_correlated_temporal_composition",
            "generator_version": SCHEDULE_GENERATOR_VERSION,
            "interpolation": "linear",
            "schedule_class": _schedule_class(
                available,
                temperature_amplitude=temperature_amplitude,
                humidity_amplitude=humidity_amplitude,
            ),
            "component_weights": weights,
            "component_weight_semantics": "relative_contribution_used_once_before_complete_shape_normalization",
            "component_availability": available,
            "event_presence_semantics": "schedule.event_count_without_hidden_activation_probability",
            "correlation_semantics": "realized_discrete_time_Pearson_on_regular_schedule_nodes",
            "amplitude_semantics": "maximum_absolute_deviation_from_exact_temporal_mean_base",
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
            "relative_humidity_interpolation_contract": "linearly_interpolate_T_in_bc_and_omega_in_bc_then_derive_phi_in_bc",
            "relative_humidity_extrema_method": "interval_endpoints_and_exact_quadratic_stationary_points",
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
            "temperature_operational_bounds": [
                float(fixed["T_in_min"]),
                float(fixed["T_in_max"]),
            ],
            "humidity_ratio_operational_bounds": [
                float(fixed["omega_min"]),
                float(fixed["omega_max"]),
            ],
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
                "values": {
                    "t": "h",
                    "T_in_bc": "K",
                    "omega_in_bc": "kg/kg",
                },
                "diagnostics": {
                    "smooth.correlation_time_hours": "h",
                    "smooth.correlation_time_intervals": "1",
                    "smooth.kernel_sigma_intervals": "1",
                    "smooth.kernel_radius_intervals": "1",
                    "events.start": "h",
                    "events.center": "h",
                    "events.end": "h",
                    "events.duration": "h",
                    "events.width": "h",
                    "planned_interval": "h",
                    **SCHEDULE_DIAGNOSTIC_UNITS,
                },
            },
        },
    )
