"""
===============================================================================
generation_validation_pilot_analysis.py
===============================================================================
Analyze successful transient pilot cases from canonical HDF5 evidence.
Responsibilities:
  - Compute exact physical-bound, duration, balance, trend, and extrema diagnostics
  - Derive storage projections only from measured successful pilot artifacts
  - Preserve configured applicability metadata without inventing scientific domains
Design principles:
  - Every material follows one generic diagnostic implementation
  - Conservation residuals remain measurements with no invented pass tolerance
  - Regular and optional exact-stop states retain their distinct canonical roles
This module does NOT:
  - Execute COMSOL, calibrate parameters, or define empirical realism thresholds
  - Add exports, alter scientific values, or create learning dataset membership
===============================================================================
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final, cast

import h5py
import numpy as np

from src import domain
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_storage as storage_service

_SECONDS_PER_HOUR: Final = 3600.0
_PILOT_ADEQUACY_MIN_DURATION_H: Final = 24.0
_TABLE_RANK: Final = 2
_MINIMUM_BALANCE_STATES: Final = 2


def _finite_vector(values: Any, *, label: str, minimum_length: int = 1) -> np.ndarray:
    """Return one finite float64 vector with a required minimum length."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < minimum_length or not np.isfinite(vector).all():
        message = f"{label} must be one finite vector with at least {minimum_length} values."
        raise ValueError(message)
    return vector


def trapezoidal_interval_integrals(time_h: Any, rate_per_second: Any) -> np.ndarray:
    """Integrate a rate over each actual interval using trapezoidal quadrature."""
    time = _finite_vector(time_h, label="time_h", minimum_length=2)
    rate = _finite_vector(rate_per_second, label="rate_per_second", minimum_length=2)
    if rate.shape != time.shape or np.any(np.diff(time) <= 0.0):
        message = "Trapezoidal integration requires equal-length values and strictly increasing times."
        raise ValueError(message)
    interval_seconds = np.diff(time) * _SECONDS_PER_HOUR
    return 0.5 * (rate[:-1] + rate[1:]) * interval_seconds


def _residual_metrics(residual: np.ndarray, *, scale: float) -> dict[str, Any]:
    """Return exact residual aggregates and a scale-normalized diagnostic."""
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        message = "Residual metrics require one finite vector."
        raise ValueError(message)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
    denominator = float(scale)
    relative = maximum / denominator if denominator > 0.0 else (0.0 if maximum == 0.0 else None)
    return {
        "max_abs_residual_kg": maximum,
        "rms_residual_kg": rms,
        "relative_residual": relative,
        "relative_scale_kg": denominator,
        "acceptance_tolerance": None,
    }


def water_balance_diagnostics(
    time_h: Any,
    series: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute independent total, granular, and gas water-balance residuals."""
    time = _finite_vector(time_h, label="global time", minimum_length=2)
    required = ("m_w_gr", "m_v_gas", "m_dot_evap", "m_dot_v_in", "m_dot_v_out")
    columns: dict[str, np.ndarray] = {}
    for name in required:
        if name not in series:
            return {
                "status": "unavailable",
                "reason": f"missing canonical global quantity {name}",
                "quadrature": "trapezoidal_over_actual_stored_times",
                "acceptance_tolerance": None,
            }
        column = _finite_vector(series[name], label=name, minimum_length=2)
        if column.shape != time.shape:
            message = f"Global series {name!r} does not align with stored times."
            raise ValueError(message)
        columns[name] = column

    m_w_gr = columns["m_w_gr"]
    m_v_gas = columns["m_v_gas"]
    evaporation = columns["m_dot_evap"]
    vapor_in = columns["m_dot_v_in"]
    vapor_out = columns["m_dot_v_out"]
    total_water = m_w_gr + m_v_gas

    external_integral = trapezoidal_interval_integrals(time, vapor_in - vapor_out)
    total_delta = np.diff(total_water)
    total_residual = total_delta - external_integral
    total_scale = max(
        float(np.max(np.abs(total_water))),
        float(np.sum(np.abs(external_integral))),
    )
    total_metrics = _residual_metrics(total_residual, scale=total_scale)
    total_metrics.update(
        {
            "interval_residual_kg": total_residual.tolist(),
            "run_residual_kg": float(total_water[-1] - total_water[0] - np.sum(external_integral)),
            "equation": "delta(m_w_gr+m_v_gas)-integral(m_dot_v_in-m_dot_v_out)=0",
        }
    )

    evaporation_integral = trapezoidal_interval_integrals(time, evaporation)
    granular_residual = np.diff(m_w_gr) + evaporation_integral
    granular_scale = max(
        float(np.max(np.abs(m_w_gr))),
        float(np.sum(np.abs(evaporation_integral))),
    )
    granular_metrics = _residual_metrics(granular_residual, scale=granular_scale)
    granular_metrics.update(
        {
            "interval_residual_kg": granular_residual.tolist(),
            "run_residual_kg": float(m_w_gr[-1] - m_w_gr[0] + np.sum(evaporation_integral)),
            "equation": "delta(m_w_gr)+integral(m_dot_evap)=0",
        }
    )

    gas_integral = trapezoidal_interval_integrals(
        time,
        evaporation + vapor_in - vapor_out,
    )
    gas_residual = np.diff(m_v_gas) - gas_integral
    gas_scale = max(
        float(np.max(np.abs(m_v_gas))),
        float(np.sum(np.abs(gas_integral))),
    )
    gas_metrics = _residual_metrics(gas_residual, scale=gas_scale)
    gas_metrics.update(
        {
            "interval_residual_kg": gas_residual.tolist(),
            "run_residual_kg": float(m_v_gas[-1] - m_v_gas[0] - np.sum(gas_integral)),
            "equation": "delta(m_v_gas)-integral(m_dot_evap+m_dot_v_in-m_dot_v_out)=0",
        }
    )
    return {
        "status": "available",
        "quadrature": "trapezoidal_over_actual_stored_times",
        "time_unit": "h",
        "rate_time_conversion": "hours_multiplied_by_3600_seconds_per_hour",
        "total_water": total_metrics,
        "granular_water": granular_metrics,
        "gas_water": gas_metrics,
        "evaporation_role": "internal_transfer_not_external_total_water_source",
        "acceptance_tolerance": None,
    }


def native_mass_balance_statistics(values: Any) -> dict[str, Any]:
    """Return complete-series statistics for COMSOL's native mass-balance value."""
    vector = _finite_vector(values, label="mt_mass_balance")
    return {
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
        "max_abs": float(np.max(np.abs(vector))),
        "mean_abs": float(np.mean(np.abs(vector))),
        "rms": float(np.sqrt(np.mean(vector**2))),
        "final": float(vector[-1]),
        "unit": "kg/s",
        "acceptance_tolerance": None,
    }


def positive_step_diagnostics(values: Any) -> dict[str, Any]:
    """Describe every exact positive step without imposing a numerical tolerance."""
    vector = _finite_vector(values, label="monotonicity series")
    positive = np.diff(vector)
    positive = positive[positive > 0.0]
    return {
        "positive_step_count": int(positive.size),
        "largest_positive_step": float(np.max(positive)) if positive.size else 0.0,
        "total_positive_excursion": float(np.sum(positive)) if positive.size else 0.0,
        "automatic_failure_tolerance": None,
    }


def monotonicity_diagnostics(series: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect maintained no-rewetting global moisture quantities."""
    return {name: positive_step_diagnostics(series[name]) for name in ("m_w_gr", "X_wb_bulk", "f_wet_dm")}


def duration_diagnostic(
    *,
    case_kind: str,
    target_reached: bool,
    final_time_h: float,
    last_regular_time_h: float,
    final_x_wb_bulk: float,
    final_f_wet_dm: float,
    configured_threshold: float,
    configured_horizon_h: float,
    previous_regular_f_wet_dm: float | None,
) -> dict[str, Any]:
    """Classify duration using a protocol lower bound and configured horizon."""
    numeric = (
        final_time_h,
        last_regular_time_h,
        final_x_wb_bulk,
        final_f_wet_dm,
        configured_threshold,
        configured_horizon_h,
    )
    if case_kind not in config_service.PILOT_CASE_KINDS or not all(math.isfinite(value) for value in numeric) or configured_horizon_h <= 0.0:
        message = "Duration diagnostics received malformed pilot state."
        raise ValueError(message)
    if target_reached:
        drying_time_h: float | None = final_time_h
        stop_reason = "target_stop"
        if case_kind == "nominal_reference":
            if final_time_h < _PILOT_ADEQUACY_MIN_DURATION_H:
                result = "TOO_FAST"
            elif final_time_h <= configured_horizon_h:
                result = "PASS"
            else:
                result = "INVALID_RESULT"
        else:
            result = "TARGET_REACHED"
    else:
        drying_time_h = None
        stop_reason = "time_horizon"
        result = "NOT_DRY_WITHIN_HORIZON" if case_kind == "nominal_reference" else "RIGHT_CENSORED"
    stop_consistent = (final_f_wet_dm <= configured_threshold) if target_reached else (final_f_wet_dm > configured_threshold)
    return {
        "result": result,
        "target_reached": target_reached,
        "drying_time_h": drying_time_h,
        "drying_time_days": None if drying_time_h is None else drying_time_h / 24.0,
        "last_valid_time_h": final_time_h,
        "last_regular_time_h": last_regular_time_h,
        "stop_reason": stop_reason,
        "final_X_wb_bulk": final_x_wb_bulk,
        "final_f_wet_dm": final_f_wet_dm,
        "configured_threshold": configured_threshold,
        "adequacy_window_h": {
            "minimum": _PILOT_ADEQUACY_MIN_DURATION_H,
            "maximum": configured_horizon_h,
            "minimum_basis": "pilot_protocol_lower_diagnostic",
            "maximum_basis": "resolved_case_time_horizon",
        },
        "previous_regular_f_wet_dm": previous_regular_f_wet_dm,
        "stop_consistent": stop_consistent,
        "root_crossing_interpolation": None,
        "right_censored": not target_reached,
    }


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    """Return finiteness and exact extrema for one numeric array."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "finite": bool(finite.all()),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
    }


def _violation(
    problems: list[dict[str, Any]],
    *,
    quantity: str,
    rule: str,
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Append one exact physical-contract violation when its mask is nonempty."""
    count = int(np.count_nonzero(mask))
    if count:
        summary = _array_summary(values)
        problems.append(
            {
                "quantity": quantity,
                "rule": rule,
                "violation_count": count,
                "observed_min": summary["min"],
                "observed_max": summary["max"],
            }
        )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return one deterministic weighted empirical quantile."""
    flat_values = np.asarray(values, dtype=np.float64).ravel()
    flat_weights = np.asarray(weights, dtype=np.float64).ravel()
    if (
        flat_values.shape != flat_weights.shape
        or not np.isfinite(flat_values).all()
        or not np.isfinite(flat_weights).all()
        or np.any(flat_weights <= 0.0)
        or not 0.0 <= quantile <= 1.0
    ):
        message = "Weighted quantile requires aligned finite values and positive weights."
        raise ValueError(message)
    order = np.argsort(flat_values, kind="stable")
    ordered_values = flat_values[order]
    cumulative = np.cumsum(flat_weights[order])
    threshold = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), ordered_values.size - 1)
    return float(ordered_values[index])


def field_and_physical_diagnostics(
    *,
    static: Mapping[str, Any],
    transient_states: Iterable[Mapping[str, Any]],
    globals_by_name: Mapping[str, Any],
    scalars: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect all canonical spatial states through one material-generic path."""
    required_static = (
        *domain.fields.PERMEABILITY_FIELDS,
        *domain.fields.POROSITY_FIELDS,
        "rho_bu_dry",
        *domain.fields.STATE_FIELDS,
    )
    static_arrays = {name: np.asarray(static[name], dtype=np.float64) for name in required_static}
    shape = static_arrays["eps_bed"].shape
    if any(array.shape != shape for array in static_arrays.values()):
        message = "Pilot static fields must share one Cartesian shape."
        raise ValueError(message)
    problems: list[dict[str, Any]] = []
    for name, array in static_arrays.items():
        _violation(
            problems,
            quantity=name,
            rule="finite",
            values=array,
            mask=~np.isfinite(array),
        )
    eps = static_arrays["eps_bed"]
    eps_min = float(scalars["eps_min_global"])
    eps_max = float(scalars["eps_max_global"])
    _violation(
        problems,
        quantity="eps_bed",
        rule=f"{eps_min}<=eps_bed<={eps_max}",
        values=eps,
        mask=(eps < eps_min) | (eps > eps_max),
    )
    determinant = static_arrays["Kxx"] * static_arrays["Kyy"] - static_arrays["Kxy"] ** 2
    _violation(
        problems,
        quantity="permeability_tensor",
        rule="Kxx>0,Kyy>0,determinant>0",
        values=determinant,
        mask=(static_arrays["Kxx"] <= 0.0) | (static_arrays["Kyy"] <= 0.0) | (determinant <= 0.0),
    )
    rho = static_arrays["rho_bu_dry"]
    _violation(
        problems,
        quantity="rho_bu_dry",
        rule="rho_bu_dry>0",
        values=rho,
        mask=rho <= 0.0,
    )

    extrema = {
        "T_min_run": math.inf,
        "T_max_run": -math.inf,
        "phi_min_run": math.inf,
        "phi_max_run": -math.inf,
        "X_wb_min_run": math.inf,
        "X_wb_max_run": -math.inf,
        "w_surf_min_run": math.inf,
        "w_surf_max_run": -math.inf,
        "w_int_min_run": math.inf,
        "w_int_max_run": -math.inf,
    }
    first_x_wb: np.ndarray | None = None
    final_x_wb: np.ndarray | None = None
    final_temperature: np.ndarray | None = None
    final_phi: np.ndarray | None = None
    state_count = 0
    f_surf = float(scalars["f_surf"])
    r_surf = float(scalars["r_surf_0"])
    a_osw = float(scalars["A_osw"])
    b_osw = float(scalars["B_osw"])
    c_osw = float(scalars["C_osw"])
    local_evaporation_min = math.inf
    local_evaporation_max = -math.inf
    local_evaporation_nonfinite_count = 0
    local_evaporation_negative_count = 0
    for state in transient_states:
        arrays = {name: np.asarray(state[name], dtype=np.float64) for name in profiles.TRANSIENT_FIELD_NAMES}
        if any(array.shape != shape for array in arrays.values()):
            message = "Pilot transient fields do not match the static Cartesian shape."
            raise ValueError(message)
        for name, array in arrays.items():
            _violation(
                problems,
                quantity=name,
                rule="finite",
                values=array,
                mask=~np.isfinite(array),
            )
        temperature = arrays["T"]
        phi = arrays["phi"]
        w_surf = arrays["w_surf"]
        w_int = arrays["w_int"]
        _violation(problems, quantity="T", rule="T>0 K", values=temperature, mask=temperature <= 0.0)
        _violation(problems, quantity="phi", rule="0<=phi<=1", values=phi, mask=(phi < 0.0) | (phi > 1.0))
        _violation(problems, quantity="w_surf", rule="w_surf>=0", values=w_surf, mask=w_surf < 0.0)
        _violation(problems, quantity="w_int", rule="w_int>=0", values=w_int, mask=w_int < 0.0)
        w_gr = f_surf * w_surf + (1.0 - f_surf) * w_int
        _violation(problems, quantity="w_gr", rule="w_gr>=0", values=w_gr, mask=w_gr < 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_db = w_gr / rho
            x_wb = w_gr / (rho + w_gr)
        _violation(problems, quantity="X_db", rule="finite", values=x_db, mask=~np.isfinite(x_db))
        _violation(problems, quantity="X_db", rule="X_db>=0", values=x_db, mask=x_db < 0.0)
        _violation(problems, quantity="X_wb", rule="finite", values=x_wb, mask=~np.isfinite(x_wb))
        _violation(problems, quantity="X_wb", rule="0<=X_wb<1", values=x_wb, mask=(x_wb < 0.0) | (x_wb >= 1.0))
        phi_effective = np.clip(phi, 1.0e-6, 0.999)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x_eq_db = 0.01 * (a_osw + b_osw * (temperature - 273.15)) * np.power(phi_effective / (1.0 - phi_effective), c_osw)
            w_eq = rho * x_eq_db
            local_evaporation = f_surf * r_surf * np.maximum(w_surf - w_eq, 0.0)
        local_finite = np.isfinite(local_evaporation)
        local_evaporation_nonfinite_count += int(np.count_nonzero(~local_finite))
        local_evaporation_negative_count += int(np.count_nonzero(local_evaporation < 0.0))
        if np.any(local_finite):
            local_evaporation_min = min(local_evaporation_min, float(np.min(local_evaporation[local_finite])))
            local_evaporation_max = max(local_evaporation_max, float(np.max(local_evaporation[local_finite])))
        _violation(
            problems,
            quantity="m_evap",
            rule="finite",
            values=local_evaporation,
            mask=~local_finite,
        )
        _violation(
            problems,
            quantity="m_evap",
            rule="m_evap>=0",
            values=local_evaporation,
            mask=local_evaporation < 0.0,
        )
        extrema["T_min_run"] = min(extrema["T_min_run"], float(np.nanmin(temperature)))
        extrema["T_max_run"] = max(extrema["T_max_run"], float(np.nanmax(temperature)))
        extrema["phi_min_run"] = min(extrema["phi_min_run"], float(np.nanmin(phi)))
        extrema["phi_max_run"] = max(extrema["phi_max_run"], float(np.nanmax(phi)))
        extrema["X_wb_min_run"] = min(extrema["X_wb_min_run"], float(np.nanmin(x_wb)))
        extrema["X_wb_max_run"] = max(extrema["X_wb_max_run"], float(np.nanmax(x_wb)))
        extrema["w_surf_min_run"] = min(extrema["w_surf_min_run"], float(np.nanmin(w_surf)))
        extrema["w_surf_max_run"] = max(extrema["w_surf_max_run"], float(np.nanmax(w_surf)))
        extrema["w_int_min_run"] = min(extrema["w_int_min_run"], float(np.nanmin(w_int)))
        extrema["w_int_max_run"] = max(extrema["w_int_max_run"], float(np.nanmax(w_int)))
        if first_x_wb is None:
            first_x_wb = x_wb.copy()
        final_x_wb = x_wb.copy()
        final_temperature = temperature.copy()
        final_phi = phi.copy()
        state_count += 1
    if state_count == 0 or first_x_wb is None or final_x_wb is None or final_temperature is None or final_phi is None:
        message = "Pilot physical diagnostics require at least one transient state."
        raise ValueError(message)

    globals_arrays = {name: _finite_vector(values, label=name) for name, values in globals_by_name.items()}
    f_wet = globals_arrays["f_wet_dm"]
    evaporation = globals_arrays["m_dot_evap"]
    _violation(problems, quantity="f_wet_dm", rule="0<=f_wet_dm<=1", values=f_wet, mask=(f_wet < 0.0) | (f_wet > 1.0))
    _violation(problems, quantity="m_dot_evap", rule="m_dot_evap>=0", values=evaporation, mask=evaporation < 0.0)
    q95_mass_final = (
        _weighted_quantile(final_x_wb, rho, 0.95) if np.isfinite(final_x_wb).all() and np.isfinite(rho).all() and np.all(rho > 0.0) else None
    )
    report_extrema: dict[str, float | None] = dict(extrema)
    report_extrema.update(
        {
            "T_min_final": float(np.min(final_temperature)),
            "T_max_final": float(np.max(final_temperature)),
            "phi_min_final": float(np.min(final_phi)),
            "phi_max_final": float(np.max(final_phi)),
            "X_target_wb": float(scalars["X_target_wb"]),
            "X_wb_max_final": float(np.max(final_x_wb)),
            "X_wb_q95_mass_final": q95_mass_final,
            "X_wb_bulk_initial": float(globals_arrays["X_wb_bulk"][0]),
            "X_wb_bulk_final": float(globals_arrays["X_wb_bulk"][-1]),
            "m_dot_evap_min": float(np.min(evaporation)),
            "m_dot_evap_max": float(np.max(evaporation)),
            "T_out_mean_min": float(np.min(globals_arrays["T_out_mean"])),
            "T_out_mean_max": float(np.max(globals_arrays["T_out_mean"])),
            "phi_out_mean_min": float(np.min(globals_arrays["phi_out_mean"])),
            "phi_out_mean_max": float(np.max(globals_arrays["phi_out_mean"])),
            "m_dot_v_in_min": float(np.min(globals_arrays["m_dot_v_in"])),
            "m_dot_v_in_max": float(np.max(globals_arrays["m_dot_v_in"])),
            "m_dot_v_out_min": float(np.min(globals_arrays["m_dot_v_out"])),
            "m_dot_v_out_max": float(np.max(globals_arrays["m_dot_v_out"])),
        }
    )
    velocity = np.hypot(static_arrays["u"], static_arrays["v"])
    airflow = {
        "p_min": float(np.min(static_arrays["p"])),
        "p_max": float(np.max(static_arrays["p"])),
        "u_min": float(np.min(static_arrays["u"])),
        "u_max": float(np.max(static_arrays["u"])),
        "v_min": float(np.min(static_arrays["v"])),
        "v_max": float(np.max(static_arrays["v"])),
        "velocity_magnitude_min": float(np.min(velocity)),
        "velocity_magnitude_mean": float(np.mean(velocity)),
        "velocity_magnitude_max": float(np.max(velocity)),
        "reverse_flow_fraction": float(np.mean(static_arrays["u"] < 0.0)),
        "reverse_flow_failure_threshold": None,
    }
    physical = {
        "status": "violation" if problems else "pass",
        "violations": problems,
        "porosity": {
            **_array_summary(eps),
            "configured_global_min": eps_min,
            "configured_global_max": eps_max,
        },
        "permeability": {
            "representation": "Kxx,Kxy,Kyy_symmetric_tensor",
            "symmetric_as_intended": True,
            "determinant_min": float(np.min(determinant)),
            "positive_definite": bool(np.all(static_arrays["Kxx"] > 0.0) and np.all(static_arrays["Kyy"] > 0.0) and np.all(determinant > 0.0)),
        },
        "rho_bu_dry": _array_summary(rho),
        "local_m_evap": {
            "status": "derived_from_binding_formula_and_canonical_states",
            "formula": "f_surf*r_surf_0*max(w_surf-w_eq,0)",
            "finite": local_evaporation_nonfinite_count == 0,
            "nonfinite_count": local_evaporation_nonfinite_count,
            "nonnegative": local_evaporation_negative_count == 0,
            "negative_count": local_evaporation_negative_count,
            "min": None if local_evaporation_min == math.inf else local_evaporation_min,
            "max": None if local_evaporation_max == -math.inf else local_evaporation_max,
            "aggregate_independent_check": "m_dot_evap",
        },
    }
    return physical, {"run_extrema": report_extrema, "airflow": airflow}


def schedule_diagnostics(
    schedule_values: Any,
    *,
    schedule_metadata: Mapping[str, Any],
    ambient_temperature: float,
    phi_operational_min: float,
    phi_operational_max: float,
) -> dict[str, Any]:
    """Recheck heater-only feasibility from canonical values and retained metadata."""
    values = np.asarray(schedule_values, dtype=np.float64)
    if values.ndim != _TABLE_RANK or values.shape[1] != len(profiles.SCHEDULE_FIELDS) or not np.isfinite(values).all():
        message = "Pilot schedule values are malformed."
        raise ValueError(message)
    columns = {name: values[:, index] for index, name in enumerate(profiles.SCHEDULE_FIELDS)}
    if (
        not math.isfinite(phi_operational_min)
        or not math.isfinite(phi_operational_max)
        or not 0.0 <= phi_operational_min <= phi_operational_max <= 1.0
    ):
        message = "Configured pilot humidity bounds are invalid."
        raise ValueError(message)
    source_min = schedule_metadata.get("min_phi_source_air")
    source_max = schedule_metadata.get("max_phi_source_air")
    heater_rise = schedule_metadata.get("min_heater_temperature_rise")
    metadata_values = (source_min, source_max, heater_rise)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in metadata_values):
        message = "Pilot schedule metadata lacks finite source-air and heater diagnostics."
        raise ValueError(message)
    source_min_value = float(cast("float", source_min))
    source_max_value = float(cast("float", source_max))
    heater_rise_value = float(cast("float", heater_rise))
    checks = {
        "T_in_bc_at_or_above_T_amb": bool(np.all(columns["T_in_bc"] >= ambient_temperature)),
        "omega_in_bc_positive": bool(np.all(columns["omega_in_bc"] > 0.0)),
        "phi_source_air_physical": bool(0.0 < source_min_value <= source_max_value <= 1.0),
        "phi_in_bc_operational": bool(np.all((columns["phi_in_bc"] >= phi_operational_min) & (columns["phi_in_bc"] <= phi_operational_max))),
    }
    return {
        "status": "pass" if all(checks.values()) else "violation",
        "checks": checks,
        "min_T_in_bc": float(np.min(columns["T_in_bc"])),
        "max_T_in_bc": float(np.max(columns["T_in_bc"])),
        "min_omega_in_bc": float(np.min(columns["omega_in_bc"])),
        "max_omega_in_bc": float(np.max(columns["omega_in_bc"])),
        "min_phi_in_bc": float(np.min(columns["phi_in_bc"])),
        "max_phi_in_bc": float(np.max(columns["phi_in_bc"])),
        "min_phi_source_air": source_min_value,
        "max_phi_source_air": source_max_value,
        "min_heater_temperature_rise": heater_rise_value,
        "configured_phi_operational_min": phi_operational_min,
        "configured_phi_operational_max": phi_operational_max,
    }


def _applicability_overlap(provenance: Mapping[str, Any]) -> str:
    """Classify an evidence record without interpreting applicability prose."""
    evidence = str(provenance.get("evidence", ""))
    if "transfer" in evidence:
        return "material_transfer"
    if evidence.startswith("engineering_") or evidence in {
        "calibration_prior",
        "hierarchical_engineering_prior",
        "synthetic_design",
    }:
        return "engineering_extension"
    return "not_evaluable_from_evidence"


def scientific_applicability_diagnostics(
    scientific: Mapping[str, Any],
    *,
    operating_domain: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose record applicability beside observed pilot extrema without parsing prose."""
    material = scientific.get("material")
    registry = material.get("parameter_registry") if isinstance(material, Mapping) else None
    if not isinstance(registry, Mapping):
        message = "Canonical scientific provenance has no material parameter registry."
        raise TypeError(message)
    selected: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for name, entry in registry.items():
        if not isinstance(entry, Mapping):
            continue
        atomic_record = entry.get("atomic_record")
        concern_name = str(atomic_record) if isinstance(atomic_record, str) else str(name)
        if concern_name in seen_records:
            continue
        if concern_name not in {
            "oswin",
            "density_calibration",
            "two_compartment_kinetics",
            "thermal_properties",
        } and name not in {"k_gr", "cp_gr_dry"}:
            continue
        provenance = entry.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        seen_records.add(concern_name)
        selected.append(
            {
                "record": concern_name,
                "representative_parameter": str(name),
                "evidence": provenance.get("evidence"),
                "method": provenance.get("method"),
                "verification": provenance.get("verification"),
                "applicability": provenance.get("applicability"),
                "note": provenance.get("note"),
                "overlap": _applicability_overlap(provenance),
                "classification_basis": "configured_evidence_only",
            }
        )
    return {
        "observed_operating_domain": dict(operating_domain),
        "records": selected,
        "numeric_applicability_parsing": "not_performed",
        "reason": "Configured applicability is exposed verbatim; absent numeric domains are not fabricated.",
    }


def _hdf5_names(dataset: h5py.Dataset) -> tuple[str, ...]:
    """Return exact JSON-encoded field names from one canonical dataset."""
    raw = dataset.attrs["field_names"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list) or not all(isinstance(name, str) for name in parsed):
        message = f"Canonical field metadata is malformed for {dataset.name}."
        raise ValueError(message)
    return tuple(parsed)


def _named_columns(dataset: h5py.Dataset) -> dict[str, np.ndarray]:
    """Materialize one compact canonical table as name-to-column arrays."""
    values = np.asarray(dataset, dtype=np.float64)
    names = _hdf5_names(dataset)
    if values.ndim == 1:
        if values.size != len(names):
            message = f"Canonical vector width is invalid for {dataset.name}."
            raise ValueError(message)
        return {name: np.asarray([values[index]], dtype=np.float64) for index, name in enumerate(names)}
    if values.ndim != _TABLE_RANK or values.shape[1] != len(names):
        message = f"Canonical table width is invalid for {dataset.name}."
        raise ValueError(message)
    return {name: values[:, index] for index, name in enumerate(names)}


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required non-symlink JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"{label} must contain one JSON object: {path}"
        raise TypeError(message)
    return value


def _required_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required canonical HDF5 dataset with an explicit type guard."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        message = f"Canonical pilot HDF5 dataset is missing or malformed: {name}."
        raise TypeError(message)
    return value


def _state_iterator(handle: h5py.File) -> Iterable[dict[str, np.ndarray]]:
    """Yield regular states and the optional final exact-stop state once."""
    dataset = _required_dataset(handle, "transient/fields")
    names = _hdf5_names(dataset)
    for index in range(dataset.shape[0]):
        frame = np.asarray(dataset[index], dtype=np.float64)
        yield {name: frame[position] for position, name in enumerate(names)}
    if "exact_stop" in handle:
        exact = _required_dataset(handle, "exact_stop/fields")
        exact_names = _hdf5_names(exact)
        frame = np.asarray(exact, dtype=np.float64)
        yield {name: frame[position] for position, name in enumerate(exact_names)}


def analyze_successful_case(
    processed_directory: Path | str,
    *,
    case_kind: str,
) -> dict[str, Any]:
    """Analyze one validated successful transient pilot case."""
    directory = Path(processed_directory).expanduser().resolve()
    hdf5_path = directory / "case.h5"
    identity = storage_service.validate_case_hdf5(
        hdf5_path,
        expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
    )
    case_payload = _json_file(directory / "case.json", label="pilot case provenance")
    status = _json_file(directory / "status.json", label="pilot case status")
    technical = case_payload.get("pilot_check")
    if not isinstance(technical, dict) or technical.get("case_kind") != case_kind or technical.get("dataset_membership") != "none":
        message = f"Pilot case kind or technical-only provenance is invalid: {directory}"
        raise ValueError(message)
    with h5py.File(hdf5_path, "r") as handle:
        static_dataset = _required_dataset(handle, "static/fields")
        static_names = _hdf5_names(static_dataset)
        static_values = np.asarray(static_dataset, dtype=np.float64)
        static = {name: static_values[index] for index, name in enumerate(static_names)}
        scalar_columns = _named_columns(_required_dataset(handle, "scalar/values"))
        scalars = {name: float(values[0]) for name, values in scalar_columns.items()}
        scientific_raw = _required_dataset(handle, "provenance/scientific_config_json")[()]
        if isinstance(scientific_raw, bytes):
            scientific_raw = scientific_raw.decode("utf-8")
        scientific = json.loads(str(scientific_raw))
        configured_regular_times = _finite_vector(
            scientific["time"]["regular_times"],
            label="configured regular times",
            minimum_length=2,
        )
        if np.any(np.diff(configured_regular_times) <= 0.0):
            message = "Configured regular times must be strictly increasing."
            raise ValueError(message)
        configured_horizon_h = float(scientific["time"]["stop"])
        fixed = scientific["scientific_fixed_values"]
        scalars.update(
            {
                "eps_min_global": float(fixed["eps_min_global"]),
                "eps_max_global": float(fixed["eps_max_global"]),
            }
        )
        global_dataset = _required_dataset(handle, "global/values")
        global_columns = _named_columns(global_dataset)
        global_time = global_columns["t"]
        physical, spatial = field_and_physical_diagnostics(
            static=static,
            transient_states=_state_iterator(handle),
            globals_by_name=global_columns,
            scalars=scalars,
        )
        schedule = np.asarray(_required_dataset(handle, "schedule/values"), dtype=np.float64)
        regular_time = np.asarray(_required_dataset(handle, "time"), dtype=np.float64)
        exact_time = float(np.asarray(_required_dataset(handle, "exact_stop/time"), dtype=np.float64)[0]) if "exact_stop" in handle else None
        transient_storage_bytes = int(_required_dataset(handle, "transient/fields").id.get_storage_size())
        global_storage_bytes = int(global_dataset.id.get_storage_size())

    target_reached = bool(status["target_reached"])
    final_time_h = float(status["t_stop_exact"])
    final_f_wet = float(status["f_wet_dm_final"])
    threshold = float(fixed["f_wet_dm_max"])
    previous_regular = None
    if global_columns["f_wet_dm"].size >= _MINIMUM_BALANCE_STATES:
        previous_regular = float(global_columns["f_wet_dm"][-2] if exact_time is not None else global_columns["f_wet_dm"][-1])
    duration = duration_diagnostic(
        case_kind=case_kind,
        target_reached=target_reached,
        final_time_h=final_time_h,
        last_regular_time_h=float(status["t_last_regular"]),
        final_x_wb_bulk=float(global_columns["X_wb_bulk"][-1]),
        final_f_wet_dm=final_f_wet,
        configured_threshold=threshold,
        configured_horizon_h=configured_horizon_h,
        previous_regular_f_wet_dm=previous_regular,
    )
    balances = water_balance_diagnostics(global_time, global_columns)
    monotonicity = monotonicity_diagnostics(global_columns)
    native_balance = native_mass_balance_statistics(global_columns["mt_mass_balance"])
    schedule_result = schedule_diagnostics(
        schedule,
        schedule_metadata=case_payload["schedule_diagnostics"],
        ambient_temperature=float(scalars["T_amb"]),
        phi_operational_min=float(fixed["phi_operational_min"]),
        phi_operational_max=float(fixed["phi_operational_max"]),
    )
    hard_checks = {
        "solver_completed": status.get("solver_success") is True,
        "study_1_completed": True,
        "study_1_evidence": "required_stationary_profile_export_canonicalized",
        "study_2_completed": True,
        "study_2_evidence": "required_transient_global_and_final_exports_canonicalized",
        "expected_output_files": all(
            (directory / name).is_file()
            for name in (
                "case.h5",
                "case.json",
                "solver.log",
                "timing.json",
                "status.json",
                "execution_provenance.json",
            )
        ),
        "mapping_header_contract": True,
        "hdf5_conversion": True,
        "time_classification": True,
        "required_shapes": True,
        "finite_required_arrays": status.get("contains_nan_or_inf") is False,
        "ordered_regular_times": bool(regular_time.size >= 1 and np.all(np.diff(regular_time) > 0.0)),
        "regular_times_match_configured_contract": bool(
            regular_time.size <= configured_regular_times.size and np.array_equal(regular_time, configured_regular_times[: regular_time.size])
        ),
        "exact_stop_excluded_from_regular_transitions": exact_time is None or exact_time > regular_time[-1],
        "source_config_template_identity": (
            identity["case_input_id"] == case_payload["case_input_id"]
            and identity["simulation_case_id"] == case_payload["simulation_case_id"]
            and identity["scientific_config_digest"] == case_payload["scientific_config_digest"]
            and identity["template_sha256"] == case_payload["template"]["sha256"]
            and identity["git_commit"] == case_payload["git_commit"]
        ),
        "scalar_handoff": status.get("field_shape_valid") is True and status.get("schedule_valid") is True,
    }
    if not all(value for value in hard_checks.values() if isinstance(value, bool)):
        message = f"Validated canonical case failed a pilot hard-contract reconstruction: {directory}"
        raise ValueError(message)

    applicability = scientific_applicability_diagnostics(
        scientific,
        operating_domain={
            "T_min_run_K": spatial["run_extrema"]["T_min_run"],
            "T_max_run_K": spatial["run_extrema"]["T_max_run"],
            "phi_min_run": spatial["run_extrema"]["phi_min_run"],
            "phi_max_run": spatial["run_extrema"]["phi_max_run"],
            "X_wb_min_run": spatial["run_extrema"]["X_wb_min_run"],
            "X_wb_max_run": spatial["run_extrema"]["X_wb_max_run"],
        },
    )
    warnings: list[str] = []
    if duration["result"] in {
        "TOO_FAST",
        "NOT_DRY_WITHIN_HORIZON",
        "RIGHT_CENSORED",
        "INVALID_RESULT",
    }:
        warnings.append(f"duration:{duration['result']}")
    adequacy_window = duration["adequacy_window_h"]
    if case_kind == "natural_pilot" and target_reached and not (adequacy_window["minimum"] <= final_time_h <= adequacy_window["maximum"]):
        warnings.append("natural_duration_outside_nominal_window")
    if not duration["stop_consistent"]:
        warnings.append("target_stop_inconsistent")
    if schedule_result["status"] != "pass":
        warnings.append("schedule_feasibility_violation")
    if any(record["positive_step_count"] for record in monotonicity.values()):
        warnings.append("positive_moisture_steps_observed")
    if any(record["overlap"] != "not_evaluable_from_evidence" for record in applicability["records"]):
        warnings.append("non_direct_scientific_applicability")
    if physical["status"] == "violation" or not duration["stop_consistent"] or schedule_result["status"] != "pass":
        result_class = "PHYSICAL_CONTRACT_VIOLATION"
    elif duration["result"] == "INVALID_RESULT":
        result_class = "INVALID_RESULT"
    elif case_kind == "nominal_reference" and duration["result"] in {
        "TOO_FAST",
        "NOT_DRY_WITHIN_HORIZON",
    }:
        result_class = duration["result"]
    elif warnings:
        result_class = "PASS_WITH_WARNINGS"
    else:
        result_class = "PASS"
    return {
        "case_id": case_payload["case_id"],
        "case_index": case_payload["case_index"],
        "simulation_case_id": identity["simulation_case_id"],
        "material": case_payload["material_family"],
        "material_role": case_payload["material_role"],
        "case_kind": case_kind,
        "solver_status": "success",
        "result_class": result_class,
        "target_reached": duration["target_reached"],
        "drying_time_h": duration["drying_time_h"],
        "drying_time_days": duration["drying_time_days"],
        "last_valid_time_h": duration["last_valid_time_h"],
        "stop_reason": duration["stop_reason"],
        "failed_stage": None,
        "warning_count": len(warnings),
        "final_X_wb_bulk": duration["final_X_wb_bulk"],
        "final_f_wet_dm": duration["final_f_wet_dm"],
        "hard_contract": hard_checks,
        "duration": duration,
        "physical_bound": physical,
        "conservation_diagnostic": {
            "independent_water_balances": balances,
            "comsol_mt_mass_balance": native_balance,
            "energy_balance": "unavailable_from_current_canonical_outputs",
        },
        "trend_diagnostic": monotonicity,
        "extrema_diagnostic": spatial,
        "schedule_input_sanity": schedule_result,
        "applicability_domain_diagnostic": applicability,
        "numerical_runtime": {
            "runtime_s": status["runtime_s"],
            "regular_state_count": int(status["n_regular_states"]),
            "final_regular_time_h": float(status["t_last_regular"]),
            "has_exact_stop_state": bool(status["has_exact_stop_state"]),
            "exact_stop_state_time_h": status["exact_stop_state_time"],
        },
        "storage": {
            "canonical_hdf5_bytes": hdf5_path.stat().st_size,
            "transient_dataset_storage_bytes": transient_storage_bytes,
            "global_dataset_storage_bytes": global_storage_bytes,
            "regular_state_count": int(status["n_regular_states"]),
            "final_regular_time_h": float(status["t_last_regular"]),
        },
        "warnings": warnings,
        "retained_evidence_path": str(directory),
    }


def production_storage_projection(
    successful_cases: Iterable[Mapping[str, Any]],
    *,
    target_case_count: int,
    regular_state_count: int,
) -> dict[str, Any]:
    """Project configured production storage from measured successful HDF5 files."""
    for value, label in (
        (target_case_count, "target_case_count"),
        (regular_state_count, "regular_state_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            message = f"{label} must be a positive integer."
            raise ValueError(message)
    records = list(successful_cases)
    sizes = [int(record["storage"]["canonical_hdf5_bytes"]) for record in records]
    common = {
        "target_case_count": target_case_count,
        "regular_state_count": regular_state_count,
    }
    if not sizes:
        return {
            **common,
            "status": "unavailable",
            "basis": None,
            "mean_based_bytes": None,
            "median_based_bytes": None,
            "reason": "no successful real pilot HDF5 artifacts",
        }
    mean_size = float(statistics.fmean(sizes))
    median_size = float(statistics.median(sizes))
    full_horizon_cases: list[float] = []
    for record in records:
        storage = record["storage"]
        observed_states = int(storage["regular_state_count"])
        transient_bytes = int(storage["transient_dataset_storage_bytes"])
        global_bytes = int(storage["global_dataset_storage_bytes"])
        size = int(storage["canonical_hdf5_bytes"])
        if observed_states < 1:
            continue
        base_bytes = size - transient_bytes - global_bytes
        if base_bytes < 0:
            continue
        per_state = (transient_bytes + global_bytes) / observed_states
        full_horizon_cases.append(base_bytes + per_state * regular_state_count)
    full_horizon = (
        {
            "status": "available",
            "projected_bytes": round(statistics.fmean(full_horizon_cases) * target_case_count),
            "method": "measured_nontrajectory_bytes_plus_measured_regular_state_storage_scaled_to_configured_state_count",
            "label": "configured_horizon_projection",
            "exact": False,
        }
        if full_horizon_cases
        else {
            "status": "unavailable",
            "reason": "measured per-state dataset storage could not be separated safely",
        }
    )
    return {
        **common,
        "status": "available",
        "basis": "observed_real_pilot_based_estimate",
        "successful_hdf5_case_count": len(sizes),
        "mean_hdf5_bytes_per_case": mean_size,
        "median_hdf5_bytes_per_case": median_size,
        "min_hdf5_bytes_per_case": min(sizes),
        "max_hdf5_bytes_per_case": max(sizes),
        "mean_based_bytes": round(mean_size * target_case_count),
        "median_based_bytes": round(median_size * target_case_count),
        "full_horizon_projection": full_horizon,
        "storage_budget_guard": None,
    }
