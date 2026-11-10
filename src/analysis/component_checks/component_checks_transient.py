"""Diagnose water-mass and sorption balances of transient reference trajectories.

Responsibilities:
  - Close the COMSOL global water balances over every stored output interval
  - Reconstruct grain water, pore-gas vapour, evaporation, and advective boundary
    fluxes from stored fields and compare them with the COMSOL global series
  - Check the inlet temperature and humidity of the fields against the Dataset schedule
  - Evaluate pointwise residuals of the two-compartment sorption equations
  - Report which balance terms the persisted Dataset and artifacts lack
  - Bundle every diagnostic of one case or Dataset for notebook and report consumers

Design principles:
  - Formulas follow the Generation physical-formula contract and domain moisture services
  - Time integrals use the stored output times only; rule differences bound smooth
    quadrature error but cannot separate sub-hourly variation from solver error
  - Out-of-range humidity and water values are counted, never silently accepted

This module does NOT:
  - Change Generation formulas, Dataset contracts, or evaluation artifacts
  - Evaluate the energy balance, diffusive boundary fluxes, or sub-hourly dynamics
  - Read source files itself or render figures

Notes:
  Balances (Generation contract, total_water_balance):
    grain:  d(m_w_gr)/dt = -m_dot_evap
    gas:    d(m_v_gas)/dt = m_dot_evap + m_dot_v_in - m_dot_v_out
    total:  d(m_w_gr + m_v_gas)/dt = m_dot_v_in - m_dot_v_out
  Field reconstruction with the case thickness Lz and trapezoidal cell weights:
    m_w_gr = Lz int(f_surf w_surf + (1 - f_surf) w_int)
    m_v_gas = Lz int(eps M_v phi p_sat(T) / (R T)) with the maintained Magnus p_sat
    m_dot_evap = Lz int(f_surf r_surf max(w_surf - w_eq, 0)), r_surf = r_surf_0
    boundary flux = Lz int(v M_v phi p_sat(T) / (R T)) dx on the inlet and outlet rows
  Reconstruction limits:
    COMSOL evaluates its own moisture-transport saturation concentration
    COMSOL boundary terms are total normal fluxes; the fields give the advective part
  Sorption residuals on [t_n, t_n+1] (trapezoidal rule):
    internal: (1 - f) dw_int/dt + j_int, j_int = (1 - f) r_int (w_int - w_surf)
    surface:  f dw_surf/dt - (j_int - m_evap)
    grain:    dw_gr/dt + m_evap

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from scipy import integrate

from src import datasets, domain, generation
from src.analysis import evaluation
from src.analysis.evaluation import evaluation_transient_validity as transient_validity

from . import component_checks_sources as sources

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .component_checks_sources import Membership, TransientReferenceCase

# CODATA 2018 molar gas constant, as COMSOL's R_const.
MOLAR_GAS_CONSTANT: Final = 8.314462618
SECONDS_PER_HOUR: Final = 3600.0
_TIME_MATCH_TOLERANCE_HOURS = 1e-6


@dataclass(frozen=True, slots=True)
class CaseBalanceDiagnostics:
    """
    Hold every balance diagnostic of one transient reference case.

    Attributes
    ----------
    label : str
        Compact ``material/case`` label.
    integrals, comparison, balance, field_closure, sorption : pandas.DataFrame
        Outputs of :func:`field_integrals`, :func:`compare_with_global`,
        :func:`global_balance`, :func:`field_balance`, and
        :func:`sorption_residuals`.
    summary : dict[str, object]
        Output of :func:`summarize_case`.

    """

    label: str
    integrals: pd.DataFrame
    comparison: pd.DataFrame
    balance: pd.DataFrame
    field_closure: pd.DataFrame
    sorption: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Grid:
    """Hold structured-grid quadrature for one case."""

    x_row: np.ndarray
    cell_area_weights: np.ndarray

    def area_integral(self, field: np.ndarray) -> float:
        """Return the trapezoidal area integral of one ``[y, x]`` field."""
        return float(np.sum(field * self.cell_area_weights, dtype=np.float64))

    def line_integral(self, values: np.ndarray) -> float:
        """Return the trapezoidal x-integral of one row."""
        return float(np.trapezoid(values, self.x_row))


def _grid(case: TransientReferenceCase) -> _Grid:
    """Return quadrature weights from the case coordinates."""
    x = case.static["x"]
    y = case.static["y"]
    x_row = x[0]
    dx = np.diff(x_row)
    dy = np.diff(y[:, 0])
    if np.any(dx <= 0) or np.any(dy <= 0):
        msg = "Transient coordinates must increase along x and y."
        raise ValueError(msg)
    weights = evaluation.transient_metrics.trapezoidal_cell_weights(np.ones(x.shape, dtype=bool)) * float(dx.mean()) * float(dy.mean())
    return _Grid(x_row=x_row, cell_area_weights=weights)


def vapour_density(temperature: np.ndarray, relative_humidity: np.ndarray, *, molar_mass: float) -> np.ndarray:
    """
    Return pore-gas vapour mass density ``M_v phi p_sat(T) / (R T)`` in kg/m^3.

    Parameters
    ----------
    temperature : numpy.ndarray
        Absolute temperature in K.
    relative_humidity : numpy.ndarray
        Relative humidity, unclipped.
    molar_mass : float
        Water-vapour molar mass in kg/mol.

    Returns
    -------
    numpy.ndarray
        Vapour density with the broadcast input shape.

    """
    temperature = np.asarray(temperature, dtype=np.float64)
    saturation = generation.cases.schedule.saturation_vapor_pressure(temperature)
    return molar_mass * np.asarray(relative_humidity, dtype=np.float64) * saturation / (MOLAR_GAS_CONSTANT * temperature)


def _state(case: TransientReferenceCase, name: str, index: int) -> np.ndarray:
    """Return one stored state field in float64."""
    return np.asarray(case.states[name][index], dtype=np.float64)


def evaporation_rate(case: TransientReferenceCase, index: int) -> tuple[np.ndarray, int]:
    """
    Return the local evaporation rate ``m_evap`` at one stored time.

    Parameters
    ----------
    case : TransientReferenceCase
        Reference trajectory with material scalars and case constants.
    index : int
        Stored-state index.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``f_surf r_surf_0 max(w_surf - w_eq, 0)`` in kg/(m^3 s) and the number
        of points whose stored relative humidity lies outside ``[0, 1]``. Such
        values are limited to ``[0, 1]`` before the equilibrium relation, which
        clips them to the recorded ``[phi_clip_min, phi_clip_max]`` anyway.

    """
    phi = _state(case, "phi", index)
    outside = int(np.count_nonzero((phi < 0.0) | (phi > 1.0)))
    scalars = case.scalars
    equilibrium = domain.moisture.oswin_equilibrium_dry_basis_moisture(
        np.clip(phi, 0.0, 1.0),
        _state(case, "T", index),
        a_osw=scalars["A_osw"],
        b_osw=scalars["B_osw"],
        c_osw=scalars["C_osw"],
        clip_bounds=(case.constants["phi_clip_min"], case.constants["phi_clip_max"]),
    )
    w_eq = case.static["rho_bu_dry"] * equilibrium
    rate = scalars["f_surf"] * scalars["r_surf_0"] * np.maximum(_state(case, "w_surf", index) - w_eq, 0.0)
    return rate, outside


def _inlet_conditions(case: TransientReferenceCase) -> tuple[np.ndarray, np.ndarray]:
    """Return scheduled inlet temperature and humidity ratio at every stored time."""
    boundary = case.boundary
    temperature = np.append(boundary["T_in_bc_t_n"], boundary["T_in_bc_t_n_plus_1"][-1])
    humidity_ratio = np.append(boundary["omega_in_bc_t_n"], boundary["omega_in_bc_t_n_plus_1"][-1])
    if temperature.shape != case.time_hours.shape:
        msg = "Transient boundary conditioning does not cover every stored time."
        raise ValueError(msg)
    return temperature, humidity_ratio


def field_integrals(case: TransientReferenceCase) -> pd.DataFrame:
    """
    Reconstruct balance quantities from the stored fields at every stored time.

    Returns
    -------
    pandas.DataFrame
        Per stored time: ``grain_water_kg``, ``gas_vapour_kg``,
        ``evaporation_kg_per_s``, ``inlet_advective_kg_per_s``,
        ``outlet_advective_kg_per_s``; the maximum inlet-row deviations of
        ``T`` (K) and ``phi`` from the schedule, separately for the inlet
        interior and the two inlet-wall corner nodes; ``phi_max``; the counts of
        stored ``phi`` values outside ``[0, 1]`` and of negative water values,
        which enter the grain-water integral as zero; and the count of points
        the maintained evaluation validity rule marks as physically invalid.

    """
    grid = _grid(case)
    length_z = case.constants["Lz"]
    molar_mass = case.constants["M_v"]
    porosity = case.static["eps_bed"]
    velocity_y = case.static["v"]
    f_surf = case.scalars["f_surf"]
    inlet_temperature, inlet_humidity_ratio = _inlet_conditions(case)
    inlet_phi = generation.cases.schedule.humidity_ratio_to_relative_humidity(
        inlet_humidity_ratio, inlet_temperature, pressure=case.constants["p_ref"]
    )
    rows = []
    for index, time in enumerate(case.time_hours):
        temperature = _state(case, "T", index)
        phi = _state(case, "phi", index)
        surface_water = _state(case, "w_surf", index)
        internal_water = _state(case, "w_int", index)
        negative = int(np.count_nonzero(surface_water < 0.0) + np.count_nonzero(internal_water < 0.0))
        # The domain contract rejects negative water; such counted points enter the integral as zero.
        water = domain.moisture.granular_water_content(np.maximum(surface_water, 0.0), np.maximum(internal_water, 0.0), f_surf)
        density = vapour_density(temperature, phi, molar_mass=molar_mass)
        rate, outside = evaporation_rate(case, index)
        temperature_deviation = np.abs(temperature[0] - inlet_temperature[index])
        phi_deviation = np.abs(phi[0] - inlet_phi[index])
        stacked = np.stack([np.asarray(case.states[name][index]) for name in transient_validity.STATE_ORDER])[None]
        rows.append(
            {
                "time_h": float(time),
                "grain_water_kg": length_z * grid.area_integral(water),
                "gas_vapour_kg": length_z * grid.area_integral(porosity * density),
                "evaporation_kg_per_s": length_z * grid.area_integral(rate),
                "inlet_advective_kg_per_s": length_z * grid.line_integral(velocity_y[0] * density[0]),
                "outlet_advective_kg_per_s": length_z * grid.line_integral(velocity_y[-1] * density[-1]),
                "inlet_temperature_deviation_k": float(temperature_deviation[1:-1].max()),
                "inlet_corner_temperature_deviation_k": float(temperature_deviation[[0, -1]].max()),
                "inlet_phi_deviation": float(phi_deviation[1:-1].max()),
                "inlet_corner_phi_deviation": float(phi_deviation[[0, -1]].max()),
                "phi_max": float(phi.max()),
                "phi_outside_unit_interval": outside,
                "negative_water_points": negative,
                "validity_invalid_points": int(transient_validity.prediction_physical_invalid_mask(stacked).sum()),
            }
        )
    return pd.DataFrame(rows)


def _global_frame(case: TransientReferenceCase) -> pd.DataFrame:
    """Return the COMSOL global series as a frame sorted by time."""
    frame = pd.DataFrame(case.global_series).sort_values("t").reset_index(drop=True)
    if frame["t"].duplicated().any():
        msg = "COMSOL global series contain duplicate times."
        raise ValueError(msg)
    return frame


def compare_with_global(case: TransientReferenceCase, integrals: pd.DataFrame) -> pd.DataFrame:
    """
    Pair field reconstructions with the COMSOL global series at stored times.

    Returns
    -------
    pandas.DataFrame
        Per stored time: each field quantity, its COMSOL counterpart, and the
        relative difference ``field / comsol - 1`` (NaN where COMSOL is zero).

    Raises
    ------
    ValueError
        If a stored time has no COMSOL global row.

    """
    global_frame = _global_frame(case)
    global_times = global_frame["t"].to_numpy()
    stored_times = integrals["time_h"].to_numpy()
    upper = np.clip(np.searchsorted(global_times, stored_times), 0, len(global_times) - 1)
    lower = np.clip(upper - 1, 0, len(global_times) - 1)
    nearest = np.where(np.abs(global_times[lower] - stored_times) <= np.abs(global_times[upper] - stored_times), lower, upper)
    matched = global_frame.iloc[nearest].reset_index(drop=True)
    if not np.allclose(matched["t"].to_numpy(), stored_times, rtol=0.0, atol=_TIME_MATCH_TOLERANCE_HOURS):
        msg = "Stored state times are missing from the COMSOL global series."
        raise ValueError(msg)
    pairs = {
        "grain_water": ("grain_water_kg", "m_w_gr"),
        "gas_vapour": ("gas_vapour_kg", "m_v_gas"),
        "evaporation": ("evaporation_kg_per_s", "m_dot_evap"),
        "inlet_flux": ("inlet_advective_kg_per_s", "m_dot_v_in"),
        "outlet_flux": ("outlet_advective_kg_per_s", "m_dot_v_out"),
    }
    frame = pd.DataFrame({"time_h": integrals["time_h"]})
    for label, (field_column, global_column) in pairs.items():
        frame[f"{label}_field"] = integrals[field_column]
        frame[f"{label}_comsol"] = matched[global_column]
        comsol = matched[global_column].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            frame[f"{label}_relative_difference"] = np.where(comsol != 0.0, integrals[field_column].to_numpy() / comsol - 1.0, np.nan)
    return frame


def _integral(rate: np.ndarray, seconds: np.ndarray, *, rule: str) -> np.ndarray:
    """Integrate a sampled rate over every interval with the named rule."""
    if rule == "trapezoid":
        return 0.5 * (rate[:-1] + rate[1:]) * seconds
    if rule == "left":
        return rate[:-1] * seconds
    msg = f"Unknown quadrature rule {rule!r}."
    raise ValueError(msg)


def global_balance(case: TransientReferenceCase) -> pd.DataFrame:
    """
    Close the COMSOL global water balances over every output interval.

    Returns
    -------
    pandas.DataFrame
        Per interval between consecutive global rows (including an exact-stop
        row): start and end time, the storage changes, the trapezoidal and
        left-endpoint flux integrals, the closure of the grain, gas, and total
        balance (storage change minus integrated net source, in kg), and the
        interval mean of COMSOL's ``mt_mass_balance``.

    """
    frame = _global_frame(case)
    seconds = np.diff(frame["t"].to_numpy()) * SECONDS_PER_HOUR
    grain = frame["m_w_gr"].to_numpy()
    gas = frame["m_v_gas"].to_numpy()
    evaporation = frame["m_dot_evap"].to_numpy()
    inflow = frame["m_dot_v_in"].to_numpy()
    outflow = frame["m_dot_v_out"].to_numpy()
    result = pd.DataFrame(
        {
            "start_h": frame["t"].to_numpy()[:-1],
            "end_h": frame["t"].to_numpy()[1:],
            "grain_change_kg": np.diff(grain),
            "gas_change_kg": np.diff(gas),
            "mt_mass_balance_kg_per_s": 0.5 * (frame["mt_mass_balance"].to_numpy()[:-1] + frame["mt_mass_balance"].to_numpy()[1:]),
        }
    )
    for rule in ("trapezoid", "left"):
        evaporated = _integral(evaporation, seconds, rule=rule)
        net_inflow = _integral(inflow - outflow, seconds, rule=rule)
        result[f"evaporated_kg_{rule}"] = evaporated
        result[f"net_inflow_kg_{rule}"] = net_inflow
        result[f"grain_closure_kg_{rule}"] = result["grain_change_kg"] + evaporated
        result[f"gas_closure_kg_{rule}"] = result["gas_change_kg"] - evaporated - net_inflow
        result[f"total_closure_kg_{rule}"] = result["grain_change_kg"] + result["gas_change_kg"] - net_inflow
    return result


def field_balance(integrals: pd.DataFrame) -> pd.DataFrame:
    """
    Close the field-reconstructed total water balance with advective boundary fluxes.

    Returns
    -------
    pandas.DataFrame
        Per stored interval: field storage change, trapezoidal advective net
        inflow, and their closure in kg. Diffusive boundary fluxes are absent.

    """
    seconds = np.diff(integrals["time_h"].to_numpy()) * SECONDS_PER_HOUR
    storage = (integrals["grain_water_kg"] + integrals["gas_vapour_kg"]).to_numpy()
    net = (integrals["inlet_advective_kg_per_s"] - integrals["outlet_advective_kg_per_s"]).to_numpy()
    inflow = _integral(net, seconds, rule="trapezoid")
    return pd.DataFrame(
        {
            "start_h": integrals["time_h"].to_numpy()[:-1],
            "end_h": integrals["time_h"].to_numpy()[1:],
            "storage_change_kg": np.diff(storage),
            "advective_net_inflow_kg": inflow,
            "closure_kg": np.diff(storage) - inflow,
        }
    )


def sorption_residuals(case: TransientReferenceCase) -> pd.DataFrame:
    """
    Evaluate pointwise sorption-equation residuals over every stored interval.

    Returns
    -------
    pandas.DataFrame
        Per interval and quadrature rule: the RMS residual of the internal,
        surface, and grain equations, each also divided by the RMS of its
        storage-rate term (``*_relative``), in kg/(m^3 s).

    Notes
    -----
    The fields are stored hourly, so the residual measures consistency up to
    the quadrature error of the rate integrals and the solver error of the
    stored states. A residual that shrinks from the left-endpoint to the
    trapezoidal rule is quadrature-limited; agreement between the rules does
    not exclude sub-hourly variation, which hourly states cannot resolve.

    """
    scalars = case.scalars
    fraction = scalars["f_surf"]
    exchange_rate = scalars["r_int_surf"] * scalars["r_surf_0"]
    seconds = np.diff(case.time_hours) * SECONDS_PER_HOUR
    rows = []
    previous: dict[str, np.ndarray] | None = None
    for index in range(case.time_hours.size):
        surface = _state(case, "w_surf", index)
        internal = _state(case, "w_int", index)
        evaporation, _ = evaporation_rate(case, index)
        current = {
            "w_surf": surface,
            "w_int": internal,
            "j_int": (1.0 - fraction) * exchange_rate * (internal - surface),
            "m_evap": evaporation,
        }
        if previous is not None:
            step = seconds[index - 1]
            storage = {
                "internal": (1.0 - fraction) * (current["w_int"] - previous["w_int"]) / step,
                "surface": fraction * (current["w_surf"] - previous["w_surf"]) / step,
            }
            storage["grain"] = storage["internal"] + storage["surface"]
            for rule, weights in (("trapezoid", (0.5, 0.5)), ("left", (1.0, 0.0))):
                exchange = weights[0] * previous["j_int"] + weights[1] * current["j_int"]
                evaporated = weights[0] * previous["m_evap"] + weights[1] * current["m_evap"]
                residual = {
                    "internal": storage["internal"] + exchange,
                    "surface": storage["surface"] - exchange + evaporated,
                    "grain": storage["grain"] + evaporated,
                }
                row: dict[str, object] = {"start_h": float(case.time_hours[index - 1]), "end_h": float(case.time_hours[index]), "rule": rule}
                for name, value in residual.items():
                    rms = float(np.sqrt(np.mean(np.square(value))))
                    scale = float(np.sqrt(np.mean(np.square(storage[name]))))
                    row[f"{name}_rms"] = rms
                    row[f"{name}_relative"] = rms / scale if scale > 0 else np.inf
                rows.append(row)
        previous = current
    return pd.DataFrame(rows)


def summarize_case(
    case: TransientReferenceCase,
    *,
    integrals: pd.DataFrame,
    comparison: pd.DataFrame,
    balance: pd.DataFrame,
    reconstructed: pd.DataFrame,
    sorption: pd.DataFrame,
) -> dict[str, object]:
    """
    Reduce one case's balance diagnostics to headline numbers.

    Returns
    -------
    dict[str, object]
        Cumulative closures relative to the water removed from the grain
        (trapezoidal, left-endpoint, and for the grain balance Simpson's rule),
        maximum relative field-versus-COMSOL differences, the largest
        ``mt_mass_balance`` relative to evaporation, inlet deviations, the
        humidity maximum and evaluation-validity counts of the reference, and
        median relative sorption residuals per rule. Ratios with a zero
        denominator, e.g. a trajectory that removes no water, are NaN.

    """
    removed = float(abs(balance["grain_change_kg"].sum()))

    def relative_to_removed(value: float) -> float:
        return value / removed if removed > 0 else math.nan

    summary: dict[str, object] = {
        "case": case.label,
        "end_h": float(balance["end_h"].iloc[-1]),
        "water_removed_kg": removed,
    }
    for rule in ("trapezoid", "left"):
        for name in ("grain", "gas", "total"):
            summary[f"{name}_closure_{rule}"] = relative_to_removed(float(balance[f"{name}_closure_kg_{rule}"].sum()))
    seconds = (np.append(balance["start_h"].to_numpy(), balance["end_h"].iloc[-1])) * SECONDS_PER_HOUR
    evaporation = _global_frame(case)["m_dot_evap"].to_numpy()
    summary["grain_closure_simpson"] = relative_to_removed(float(balance["grain_change_kg"].sum()) + float(integrate.simpson(evaporation, x=seconds)))
    evaporation_scale = float(np.abs(case.global_series["m_dot_evap"]).max())
    largest_balance = float(np.abs(case.global_series["mt_mass_balance"]).max())
    summary["mt_mass_balance_over_evaporation"] = largest_balance / evaporation_scale if evaporation_scale > 0 else math.nan
    summary["field_total_closure"] = relative_to_removed(float(reconstructed["closure_kg"].sum()))
    for label in ("grain_water", "gas_vapour", "evaporation", "inlet_flux", "outlet_flux"):
        summary[f"{label}_max_relative_difference"] = float(comparison[f"{label}_relative_difference"].abs().max())
    for column in (
        "inlet_temperature_deviation_k",
        "inlet_corner_temperature_deviation_k",
        "inlet_phi_deviation",
        "inlet_corner_phi_deviation",
        "phi_max",
    ):
        summary[column] = float(integrals[column].max())
    summary["phi_outside_unit_interval"] = int(integrals["phi_outside_unit_interval"].sum())
    summary["negative_water_points"] = int(integrals["negative_water_points"].sum())
    summary["validity_invalid_points"] = int(integrals["validity_invalid_points"].sum())
    invalid_times = integrals.loc[integrals["validity_invalid_points"] > 0, "time_h"]
    summary["validity_invalid_times_h"] = ", ".join(f"{value:g}" for value in invalid_times) or "none"
    for rule in ("trapezoid", "left"):
        selected = sorption[sorption["rule"] == rule]
        for name in ("internal", "surface", "grain"):
            summary[f"sorption_{name}_{rule}_median"] = float(selected[f"{name}_relative"].median())
    return summary


def evaluate_case(case: TransientReferenceCase) -> CaseBalanceDiagnostics:
    """Evaluate every balance diagnostic of one reference case."""
    integrals = field_integrals(case)
    comparison = compare_with_global(case, integrals)
    balance = global_balance(case)
    closure = field_balance(integrals)
    sorption = sorption_residuals(case)
    summary = summarize_case(case, integrals=integrals, comparison=comparison, balance=balance, reconstructed=closure, sorption=sorption)
    return CaseBalanceDiagnostics(
        label=case.label,
        integrals=integrals,
        comparison=comparison,
        balance=balance,
        field_closure=closure,
        sorption=sorption,
        summary=summary,
    )


def evaluate_dataset(
    dataset_id: str,
    *,
    membership: Membership,
    storage_root: Path | str | None = None,
    max_cases: int | None = None,
) -> list[CaseBalanceDiagnostics]:
    """Evaluate every balance diagnostic for the leading cases of one transient Dataset membership."""
    cases = sources.iter_transient_reference_cases(dataset_id, membership=membership, storage_root=storage_root, max_cases=max_cases)
    return [evaluate_case(case) for case in cases]


def summary_table(cases: Sequence[CaseBalanceDiagnostics]) -> pd.DataFrame:
    """Return the per-case headline numbers with one column per case."""
    return pd.DataFrame([item.summary for item in cases]).set_index("case").T


def balance_term_availability() -> pd.DataFrame:
    """
    Tabulate where each water-balance term is persisted.

    Returns
    -------
    pandas.DataFrame
        One row per balance term: whether the COMSOL global series of the
        source HDF5 carry it, whether the Dataset transient contract carries
        it or the fields it needs, and how a field-level reconstruction is
        limited. Membership is read from the Generation and Dataset contracts.

    """
    global_names = set(generation.contracts.profiles.GLOBAL_FIELD_NAMES)
    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    dataset_names = {field.name for field in (*contract.dynamic_state, *contract.static_spatial_conditioning, *contract.scalar_conditioning)}
    terms = (
        ("grain water m_w_gr", "m_w_gr", ("w_surf", "w_int", "f_surf"), "exact up to quadrature"),
        (
            "evaporation m_dot_evap",
            "m_dot_evap",
            ("w_surf", "T", "phi", "rho_bu_dry", "f_surf", "r_surf_0", "A_osw", "B_osw", "C_osw"),
            "exact up to quadrature",
        ),
        ("pore-gas vapour m_v_gas", "m_v_gas", ("T", "phi", "eps_bed"), "COMSOL saturation concentration not recorded; Magnus used"),
        ("inlet flux m_dot_v_in", "m_dot_v_in", ("v", "T", "phi"), "advective part only; diffusive inlet flux not exported"),
        ("outlet flux m_dot_v_out", "m_dot_v_out", ("v", "T", "phi"), "advective part only"),
        ("COMSOL balance mt_mass_balance", "mt_mass_balance", (), "not reconstructable"),
    )
    return pd.DataFrame(
        [
            {
                "term": label,
                "comsol_global_series": name in global_names,
                "dataset_term": name in dataset_names,
                "dataset_has_required_fields": bool(required) and set(required) <= dataset_names,
                "field_reconstruction": note,
            }
            for label, name, required, note in terms
        ]
    )
