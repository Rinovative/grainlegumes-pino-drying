"""
===============================================================================
generation_input_diagnostics.py
===============================================================================
Build immutable diagnostics from admitted profile-aware generation inputs.
Responsibilities:
  - Preserve exact sampled, scalar, schedule, spatial, and provenance evidence
  - Derive shared permeability and transient moisture diagnostics canonically
  - Preserve exact persisted supports and evaluate display-only boundary curves
  - Provide Celsius-aware A/B comparisons and empirical dataset summaries
Design principles:
  - Generation, profile, and domain owners define every scientific semantic
  - Steady and transient inputs share one model with explicit optional content
  - Every derived array is an independent read-only physical representation
This module does NOT:
  - Read unadmitted files, execute COMSOL, or mark inputs as completed
  - Invent missing profile inputs, mutate evidence, or render notebook widgets
===============================================================================
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src import domain
from src.generation.cases import generation_cases_admission as admission
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.contracts import generation_contracts_profiles as profiles

from . import generation_input_labels as labels
from . import generation_input_table_schema as table_schema

if TYPE_CHECKING:
    from collections.abc import Sequence

SCALAR_NAMES: Final = profiles.TRANSIENT_SCALAR_INPUT_FIELDS
COMMON_DERIVED_FIELD_NAMES: Final = ("K_min", "K_max", "K_anisotropy")
TRANSIENT_DERIVED_FIELD_NAMES: Final = ("rho_bu_dry", "w_gr0", "phi_eq")
PERMEABILITY_FIELD_NAMES: Final = ("Kxx", "Kxy", "Kyy", *COMMON_DERIVED_FIELD_NAMES)
PERMEABILITY_DISTRIBUTION_NAMES: Final = ("K_min", "K_max", "K_anisotropy")
SPATIAL_OVERVIEW_FIELD_NAMES: Final = ("eps_bed", "p_in_bc")
MOISTURE_FIELD_NAMES: Final = ("X_0_db_field", "phi_eq")
MOISTURE_DISTRIBUTION_NAMES: Final = ("X_0_db_field", "phi_eq", "eps_bed")

FIELD_LABELS: Final = MappingProxyType(
    {
        "x": "Horizontal coordinate",
        "y": "Vertical coordinate",
        "Kxx": "Permeability xx component",
        "Kxy": "Permeability xy component",
        "Kyy": "Permeability yy component",
        "eps_bed": "Bed porosity",
        "p_in_bc": "Inlet pressure boundary",
        "X_0_db_field": "Initial dry-basis moisture",
        "K_min": "Minimum principal permeability",
        "K_max": "Maximum principal permeability",
        "K_anisotropy": "Principal permeability anisotropy",
        "rho_bu_dry": "Dry bulk density",
        "w_gr0": "Initial granular water content",
        "phi_eq": "Initial equilibrium relative humidity",
    }
)
FIELD_UNITS: Final = MappingProxyType(
    {
        "x": "m",
        "y": "m",
        "Kxx": "m^2",
        "Kxy": "m^2",
        "Kyy": "m^2",
        "eps_bed": "1",
        "p_in_bc": "Pa",
        "X_0_db_field": "kg/kg",
        "K_min": "m^2",
        "K_max": "m^2",
        "K_anisotropy": "1",
        "rho_bu_dry": "kg/m^3",
        "w_gr0": "kg/m^3",
        "phi_eq": "1",
    }
)
STAT_NAMES: Final = ("min", "q05", "median", "mean", "q95", "max", "std")
COMPARISON_COLUMNS: Final = ("Case A", "Mean A", "Case B", "Mean B")
_PRIMITIVE_SCHEDULE_NAMES: Final = profiles.SCHEDULE_FIELDS[1:]
_PRIMITIVE_SCHEDULE_UNITS: Final = profiles.SCHEDULE_UNITS[1:]
_SCHEDULE_NAMES: Final = (*_PRIMITIVE_SCHEDULE_NAMES, "phi_in_bc")
_SCHEDULE_UNITS: Final = (*_PRIMITIVE_SCHEDULE_UNITS, "1")
_ABSOLUTE_TEMPERATURE_DISPLAY_NAMES: Final = frozenset(
    (
        "T_amb",
        "T_flow_ref",
        "T_in_base",
        "T_init",
        "T_in_bc",
        "T_in_bc start",
        "T_in_bc startup end",
    )
)
_MINIMUM_SCHEDULE_NODES: Final = 2
_SCHEDULE_TABLE_RANK: Final = 2
_SPATIAL_DIMENSIONS: Final = 2
if set(FIELD_UNITS) != set(FIELD_LABELS):
    msg = "Generation-input field labels and units must cover the same inventory."
    raise RuntimeError(msg)
if set(profiles.TRANSIENT_SPATIAL_INPUT_FIELDS).union(
    COMMON_DERIVED_FIELD_NAMES,
    TRANSIENT_DERIVED_FIELD_NAMES,
) != set(FIELD_LABELS):
    msg = "Generation-input field metadata must cover all maintained raw and derived fields."
    raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class StartupVariableDiagnostics:
    """
    Describe one exact boundary variable across the startup handoff interval.

    Attributes
    ----------
    name : str
        Canonical schedule-channel name.
    unit : str
        Physical unit of the primitive or canonically derived channel.
    start, end : float
        Exact values at the first schedule support and the startup endpoint.
    delta : float
        Signed ``end - start`` change across startup support.

    """

    name: str
    unit: str
    start: float
    end: float
    delta: float


@dataclass(frozen=True, slots=True)
class StartupDiagnostics:
    """
    Describe exact transient startup support and psychrometric consistency.

    Attributes
    ----------
    enabled : bool
        Whether the configured startup ramp is enabled.
    duration_h : float
        Configured startup-ramp duration in hours.
    t_start_h, t_end_h : float
        Exact schedule support bounding the startup interval in hours.
    support_times_h : numpy.ndarray
        Read-only final COMSOL schedule support through the persisted duration.
    variables : Mapping[str, StartupVariableDiagnostics]
        Exact endpoint diagnostics for every maintained boundary channel.

    """

    enabled: bool
    duration_h: float
    t_start_h: float
    t_end_h: float
    support_times_h: np.ndarray
    variables: Mapping[str, StartupVariableDiagnostics]


@dataclass(frozen=True, slots=True)
class GenerationInputDiagnostics:
    """
    Expose one immutable scientific view of an admitted generation input.

    Attributes
    ----------
    case : AdmittedInputCase
        Validated persisted source and provenance evidence.
    profile_id : str
        Maintained simulation-profile identifier.
    scalars : Mapping[str, float]
        Exact transient scalar handoff, empty for steady-flow inputs.
    fields : Mapping[str, numpy.ndarray]
        Read-only raw and canonically derived structured-grid fields.
    schedule : numpy.ndarray | None
        Exact primitive COMSOL boundary schedule for transient inputs.
    canonical_schedule : numpy.ndarray | None
        Canonical pre-handoff transient schedule support.
    regular_output_schedule : numpy.ndarray | None
        Configured regular requested state/output schedule.
    startup : StartupDiagnostics | None
        Transient startup diagnostics, absent for steady-flow inputs.

    """

    case: admission.AdmittedInputCase
    profile_id: str
    scalars: Mapping[str, float]
    fields: Mapping[str, np.ndarray]
    schedule: np.ndarray | None
    canonical_schedule: np.ndarray | None
    regular_output_schedule: np.ndarray | None
    startup: StartupDiagnostics | None


@dataclass(frozen=True, slots=True)
class DatasetDiagnostics:
    """Hold empirical summaries over all unique admitted cases in one dataset."""

    profile_id: str
    batch_id: str
    batch_storage_name: str
    batch_identity: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str
    records: tuple[GenerationInputDiagnostics, ...]
    parameter_means: Mapping[str, float | str]
    parameter_units: Mapping[str, str]
    field_summary_means: Mapping[tuple[str, str], float]
    boundary_means: Mapping[str, float | str]
    schedule_mean: np.ndarray | None
    schedule_mean_unavailable: str | None

    @property
    def case_count(self) -> int:
        """Return the number of unique cases represented by every empirical mean."""
        return len(self.records)


def is_transient(record: GenerationInputDiagnostics) -> bool:
    """Return whether a diagnostic record owns transient-only inputs."""
    return record.profile_id == profiles.TRANSIENT_DRYING_PROFILE


def raw_spatial_names(record: GenerationInputDiagnostics) -> tuple[str, ...]:
    """Return the exact profile-owned spatial adapter inventory."""
    return profiles.spatial_input_fields(record.profile_id)


def derived_field_names(record: GenerationInputDiagnostics) -> tuple[str, ...]:
    """Return derived field names available for one profile."""
    if is_transient(record):
        return (*COMMON_DERIVED_FIELD_NAMES, *TRANSIENT_DERIVED_FIELD_NAMES)
    return COMMON_DERIVED_FIELD_NAMES


def display_field_names(record: GenerationInputDiagnostics) -> tuple[str, ...]:
    """Return non-coordinate raw and derived fields available for display."""
    raw = tuple(name for name in raw_spatial_names(record) if name not in {"x", "y"})
    return (*raw, *derived_field_names(record))


def inlet_pressure_boundary(
    record: GenerationInputDiagnostics,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the exact one-dimensional inlet-pressure support and values.

    Raises
    ------
    ValueError
        If the persisted adapter is not two-dimensional, does not match the
        coordinate grid, or contains pressure outside its canonical inlet row.

    """
    x_coordinates = record.fields["x"]
    pressure = record.fields["p_in_bc"]
    if x_coordinates.shape != pressure.shape or pressure.ndim != _SPATIAL_DIMENSIONS:
        msg = "Inlet pressure requires matching two-dimensional coordinate and boundary fields."
        raise ValueError(msg)
    if np.any(pressure[1:, :] != 0.0):
        msg = "Inlet pressure must be zero outside its canonical inlet row."
        raise ValueError(msg)
    return _immutable(x_coordinates[0, :]), _immutable(pressure[0, :])


def field_summary_specs(record: GenerationInputDiagnostics) -> tuple[tuple[str, str], ...]:
    """Return the compact maintained scalar field summaries for one profile."""
    common = (
        ("eps_bed", "min"),
        ("eps_bed", "mean"),
        ("eps_bed", "max"),
        ("p_in_bc", "min"),
        ("p_in_bc", "mean"),
        ("p_in_bc", "max"),
        ("Kxx", "median"),
        ("Kxy", "median"),
        ("Kyy", "median"),
        ("K_min", "median"),
        ("K_max", "median"),
        ("K_anisotropy", "q95"),
    )
    if not is_transient(record):
        return common
    return (
        *common,
        ("X_0_db_field", "min"),
        ("X_0_db_field", "mean"),
        ("X_0_db_field", "max"),
        ("phi_eq", "mean"),
        ("phi_eq", "q95"),
        ("rho_bu_dry", "mean"),
        ("w_gr0", "mean"),
    )


def case_display_label(record: GenerationInputDiagnostics) -> str:
    """Return one short case label independent of persisted directory names."""
    return f"Case {record.case.case_index}"


def _immutable(value: Any) -> np.ndarray:
    """Return one independent read-only float64 array."""
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _plain_json(value: Any) -> Any:
    """Return mutable JSON-compatible evidence for concise table formatting."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _finite_real_scalar(value: object, *, label: str) -> float:
    """Return one finite Python or NumPy real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a finite real scalar."
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{label} must be finite."
        raise ValueError(msg)
    return number


def _numeric_parameter(value: Any) -> float | None:
    """Return one finite scalar parameter or None for structured evidence."""
    try:
        return _finite_real_scalar(value, label="Parameter value")
    except (TypeError, ValueError):
        return None


def _parameter_display(value: Any) -> float | str:
    """Return a numeric cell or stable compact JSON for structured parameters."""
    number = _numeric_parameter(value)
    if number is not None:
        return number
    return json.dumps(_plain_json(value), sort_keys=True, separators=(",", ":"))


def _field_statistics(values: np.ndarray) -> dict[str, float]:
    """Return the maintained finite scalar summary for one physical field."""
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _fixed_value(payload: Mapping[str, Any], name: str) -> float:
    """Return one uniquely persisted finite stationary fixed value."""
    matches = [entry for entry in payload["stationary_fixed_values"] if entry.get("name") == name]
    if len(matches) != 1:
        msg = f"Case payload must contain exactly one stationary fixed value {name!r}."
        raise ValueError(msg)
    value = matches[0].get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        msg = f"Stationary fixed value {name!r} must be finite numeric evidence."
        raise ValueError(msg)
    return float(value)


def _canonical_schedule(
    schedule: np.ndarray,
    handoff: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Reconstruct exact canonical and output rows from persisted handoff evidence."""
    grid = handoff["canonical_regular_grid"]
    start = float(grid["start_h"])
    stop = float(grid["stop_h"])
    interval = float(grid["interval_h"])
    node_count = int(grid["node_count"])
    if node_count < _MINIMUM_SCHEDULE_NODES or not np.isfinite((start, stop, interval)).all() or interval <= 0.0:
        msg = "Canonical regular-grid metadata is invalid."
        raise ValueError(msg)
    times = start + interval * np.arange(node_count, dtype=np.float64)
    if times[-1] != stop:
        msg = "Canonical regular-grid stop does not match its declared node count."
        raise ValueError(msg)

    ramp = handoff["startup_ramp"]
    enabled = bool(ramp["enabled"])
    duration_h = float(ramp["duration_h"])
    expected_rows = node_count + int(enabled)
    if schedule.shape != (expected_rows, len(profiles.SCHEDULE_FIELDS)):
        msg = "Final COMSOL schedule row count disagrees with startup-handoff evidence."
        raise ValueError(msg)
    if enabled:
        if schedule[1, 0] != duration_h or not np.array_equal(
            schedule[1],
            np.asarray(handoff["rejoin_row"], dtype=np.float64),
        ):
            msg = "Final COMSOL schedule does not retain the exact startup rejoin row."
            raise ValueError(msg)
        regular_tail = schedule[2:]
    else:
        if handoff["rejoin_row"] is not None:
            msg = "Disabled startup handoff cannot declare a rejoin row."
            raise ValueError(msg)
        regular_tail = schedule[1:]

    canonical_start = np.asarray(handoff["canonical_start_row"], dtype=np.float64)
    canonical = _immutable(np.vstack((canonical_start, regular_tail)))
    if not np.array_equal(canonical[:, 0], times):
        msg = "Final COMSOL schedule does not retain the exact canonical regular support."
        raise ValueError(msg)
    if handoff["regular_output_time_policy"] != "common.time.regular_times_unchanged":
        msg = "Generation-input EDA requires unchanged regular output times."
        raise ValueError(msg)
    return canonical, _immutable(canonical), enabled, duration_h


def _startup_diagnostics(
    case: admission.AdmittedInputCase,
    schedule: np.ndarray,
    *,
    enabled: bool,
    duration_h: float,
) -> StartupDiagnostics:
    """Build primitive startup values and derived psychrometric endpoints."""
    end_index = 1 if enabled else 0
    variables: dict[str, StartupVariableDiagnostics] = {
        name: StartupVariableDiagnostics(
            name=name,
            unit=unit,
            start=float(schedule[0, column]),
            end=float(schedule[end_index, column]),
            delta=float(schedule[end_index, column] - schedule[0, column]),
        )
        for column, (name, unit) in enumerate(
            zip(_PRIMITIVE_SCHEDULE_NAMES, _PRIMITIVE_SCHEDULE_UNITS, strict=True),
            start=1,
        )
    }
    pressure = _fixed_value(case.payload, "p_ref")
    derived_phi = schedule_service.humidity_ratio_to_relative_humidity(
        np.asarray((variables["omega_in_bc"].start, variables["omega_in_bc"].end)),
        np.asarray((variables["T_in_bc"].start, variables["T_in_bc"].end)),
        pressure=pressure,
    )
    variables["phi_in_bc"] = StartupVariableDiagnostics(
        name="phi_in_bc",
        unit="1",
        start=float(derived_phi[0]),
        end=float(derived_phi[1]),
        delta=float(derived_phi[1] - derived_phi[0]),
    )
    support = _immutable(schedule[schedule[:, 0] <= duration_h, 0])
    return StartupDiagnostics(
        enabled=enabled,
        duration_h=duration_h,
        t_start_h=float(schedule[0, 0]),
        t_end_h=float(schedule[end_index, 0]),
        support_times_h=support,
        variables=MappingProxyType(variables),
    )


def build_case_diagnostics(case: admission.AdmittedInputCase) -> GenerationInputDiagnostics:
    """
    Derive one immutable diagnostic record from an admitted generation input.

    Parameters
    ----------
    case : AdmittedInputCase
        One validated production-generated raw-input bundle.

    Returns
    -------
    GenerationInputDiagnostics
        Raw evidence plus exact profile-appropriate derived diagnostics.

    Raises
    ------
    ValueError
        If admitted evidence disagrees with its profile contract or contains an
        invalid calibration state.

    Notes
    -----
    Shared permeability diagnostics are available for both maintained profiles.
    Transient schedules and moisture diagnostics remain explicitly absent for
    steady inputs rather than being synthesized. Source arrays are not mutated.

    """
    profile_id = case.profile_id
    if case.payload["simulation_profile"] != profile_id:
        msg = "Admitted profile identity disagrees with case.json."
        raise ValueError(msg)
    expected_raw = profiles.spatial_input_fields(profile_id)
    if tuple(case.fields) != expected_raw:
        msg = "Admitted spatial inventory does not match its profile contract."
        raise ValueError(msg)
    raw = {name: _immutable(case.fields[name]) for name in expected_raw}
    principal = domain.permeability.symmetric_tensor_diagnostics(raw["Kxx"], raw["Kxy"], raw["Kyy"])
    fields: dict[str, np.ndarray] = {
        **raw,
        "K_min": _immutable(principal.minimum_principal),
        "K_max": _immutable(principal.maximum_principal),
        "K_anisotropy": _immutable(principal.anisotropy_ratio),
    }

    if profile_id == profiles.STEADY_FLOW_PROFILE:
        if case.scalars or case.schedule is not None:
            msg = "Steady inputs cannot contain transient scalar or schedule adapters."
            raise ValueError(msg)
        return GenerationInputDiagnostics(
            case=case,
            profile_id=profile_id,
            scalars=MappingProxyType({}),
            fields=MappingProxyType(fields),
            schedule=None,
            canonical_schedule=None,
            regular_output_schedule=None,
            startup=None,
        )

    if tuple(case.scalars) != SCALAR_NAMES or case.schedule is None:
        msg = "Transient scalar or schedule inventory does not match its profile contract."
        raise ValueError(msg)
    scalars = MappingProxyType({name: float(case.scalars[name]) for name in SCALAR_NAMES})
    calibration_porosity = scalars["eps_bed_cal_ref"]
    if not 0.0 < calibration_porosity < 1.0:
        msg = "Calibration porosity must lie strictly inside (0, 1)."
        raise ValueError(msg)
    dry_density = scalars["rho_bu_dry_ref"] * (1.0 - raw["eps_bed"]) / (1.0 - calibration_porosity)
    initial_temperature = float(case.payload["sampled_values"]["T_init"])
    equilibrium_humidity = domain.moisture.oswin_equilibrium_relative_humidity(
        raw["X_0_db_field"],
        initial_temperature,
        a_osw=scalars["A_osw"],
        b_osw=scalars["B_osw"],
        c_osw=scalars["C_osw"],
    )
    fields.update(
        {
            "rho_bu_dry": _immutable(dry_density),
            "w_gr0": _immutable(dry_density * raw["X_0_db_field"]),
            "phi_eq": _immutable(equilibrium_humidity),
        }
    )

    schedule = _immutable(case.schedule)
    handoff = case.payload["schedule_diagnostics"]["boundary_handoff"]
    canonical, output, enabled, duration_h = _canonical_schedule(schedule, handoff)
    startup = _startup_diagnostics(case, schedule, enabled=enabled, duration_h=duration_h)
    return GenerationInputDiagnostics(
        case=case,
        profile_id=profile_id,
        scalars=scalars,
        fields=MappingProxyType(fields),
        schedule=schedule,
        canonical_schedule=canonical,
        regular_output_schedule=output,
        startup=startup,
    )


def build_collection_diagnostics(
    cases: Sequence[admission.AdmittedInputCase],
) -> tuple[GenerationInputDiagnostics, ...]:
    """Build immutable diagnostics in caller-supplied stable case order."""
    return tuple(build_case_diagnostics(case) for case in cases)


def _require_transient(
    record: GenerationInputDiagnostics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StartupDiagnostics]:
    """Return complete transient evidence or reject a steady-only request."""
    if (
        not is_transient(record)
        or record.schedule is None
        or record.canonical_schedule is None
        or record.regular_output_schedule is None
        or record.startup is None
    ):
        msg = "This diagnostic is available only for transient-drying inputs."
        raise ValueError(msg)
    return record.schedule, record.canonical_schedule, record.regular_output_schedule, record.startup


def transient_evidence(
    record: GenerationInputDiagnostics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StartupDiagnostics]:
    """Return complete exact transient evidence through the public contract."""
    return _require_transient(record)


def schedule_window_rows(
    schedule: np.ndarray,
    duration_h: float,
    *,
    startup_only: bool,
) -> np.ndarray:
    """Return immutable exact persisted rows for one semantic schedule window."""
    times_h = schedule[:, 0]
    mask = (times_h >= 0.0) & (times_h <= duration_h) if startup_only else times_h >= duration_h
    return _immutable(np.array(schedule[mask], copy=True))


def evaluate_boundary_schedule(
    schedule: np.ndarray,
    times_h: np.ndarray,
    *,
    pressure: float,
) -> np.ndarray:
    """Evaluate primitive interpolation and derive RH at display times."""
    values = np.asarray(schedule, dtype=np.float64)
    display_times = np.asarray(times_h, dtype=np.float64)
    if (
        values.ndim != _SCHEDULE_TABLE_RANK
        or values.shape[1] != len(profiles.SCHEDULE_FIELDS)
        or values.shape[0] < _MINIMUM_SCHEDULE_NODES
        or not np.isfinite(values).all()
        or np.any(np.diff(values[:, 0]) <= 0.0)
    ):
        msg = "Boundary display evaluation requires a finite ordered primitive schedule."
        raise ValueError(msg)
    if (
        display_times.ndim != 1
        or not np.isfinite(display_times).all()
        or np.any(display_times < values[0, 0])
        or np.any(display_times > values[-1, 0])
    ):
        msg = "Boundary display times must be finite and inside persisted support."
        raise ValueError(msg)
    temperature = np.interp(display_times, values[:, 0], values[:, 1])
    humidity_ratio = np.interp(display_times, values[:, 0], values[:, 2])
    relative_humidity = schedule_service.humidity_ratio_to_relative_humidity(
        humidity_ratio,
        temperature,
        pressure=pressure,
    )
    return _immutable(
        np.column_stack(
            (
                display_times,
                temperature,
                humidity_ratio,
                relative_humidity,
            )
        )
    )


def case_boundary_schedule(
    record: GenerationInputDiagnostics,
    times_h: np.ndarray,
) -> np.ndarray:
    """Evaluate one case's primitive schedule and derived inlet RH."""
    schedule, _canonical, _output, _startup = _require_transient(record)
    return evaluate_boundary_schedule(
        schedule,
        times_h,
        pressure=_fixed_value(record.case.payload, "p_ref"),
    )


def dataset_boundary_schedule(
    dataset: DatasetDiagnostics,
    times_h: np.ndarray,
) -> np.ndarray:
    """Average per-case evaluated primitive and derived boundary curves."""
    evaluated = tuple(case_boundary_schedule(record, times_h) for record in dataset.records)
    support = evaluated[0][:, 0]
    if any(not np.array_equal(values[:, 0], support) for values in evaluated[1:]):
        msg = "Dataset boundary display evaluations must share exact display times."
        raise ValueError(msg)
    return _immutable(
        np.column_stack(
            (
                support,
                np.mean(
                    np.stack(tuple(values[:, 1:] for values in evaluated)),
                    axis=0,
                ),
            )
        )
    )


def startup_schedule_rows(record: GenerationInputDiagnostics) -> np.ndarray:
    """Return exact final-schedule rows over the active startup interval."""
    schedule, _canonical, _output, startup = _require_transient(record)
    duration_h = startup.duration_h if startup.enabled else 0.0
    return schedule_window_rows(schedule, duration_h, startup_only=True)


def operating_schedule_rows(record: GenerationInputDiagnostics) -> np.ndarray:
    """Return exact final-schedule rows from active startup rejoin or time zero."""
    schedule, _canonical, _output, startup = _require_transient(record)
    start_h = startup.duration_h if startup.enabled else 0.0
    return schedule_window_rows(schedule, start_h, startup_only=False)


def startup_schedule_minutes(record: GenerationInputDiagnostics) -> np.ndarray:
    """Return startup rows with only their time coordinate converted to minutes."""
    rows = np.array(startup_schedule_rows(record), copy=True)
    rows[:, 0] *= 60.0
    return _immutable(rows)


def field_statistics(record: GenerationInputDiagnostics) -> pd.DataFrame:
    """Return compact statistics over each quantity's meaningful support."""
    rows = []
    for name in display_field_names(record):
        values = inlet_pressure_boundary(record)[1] if name == "p_in_bc" else record.fields[name]
        rows.append(
            {
                "quantity": name,
                "physical meaning": FIELD_LABELS[name],
                "unit": FIELD_UNITS[name],
                **_field_statistics(values),
            }
        )
    return pd.DataFrame(rows).set_index("quantity")


def _homogeneous_records(
    records: Sequence[GenerationInputDiagnostics],
    *,
    minimum: int = 1,
) -> tuple[GenerationInputDiagnostics, ...]:
    """Return a non-empty same-profile record tuple."""
    selected = tuple(records)
    if len(selected) < minimum:
        msg = f"Generation-input comparison requires at least {minimum} case(s)."
        raise ValueError(msg)
    profiles_found = {record.profile_id for record in selected}
    if len(profiles_found) != 1:
        msg = "Generation-input comparison requires cases from one simulation profile."
        raise ValueError(msg)
    return selected


def _schedule_mean(
    records: tuple[GenerationInputDiagnostics, ...],
) -> tuple[np.ndarray | None, str | None]:
    """Return a pointwise mean only when every persisted support is identical."""
    if not is_transient(records[0]):
        return None, None
    schedules = tuple(_require_transient(record)[0] for record in records)
    support = schedules[0][:, 0]
    if any(schedule.shape != schedules[0].shape or not np.array_equal(schedule[:, 0], support) for schedule in schedules[1:]):
        return (
            None,
            "Pointwise dataset mean unavailable: persisted schedule supports differ exactly.",
        )
    mean = np.column_stack(
        (
            support,
            np.mean(
                np.stack(tuple(schedule[:, 1:] for schedule in schedules)),
                axis=0,
            ),
        )
    )
    return _immutable(mean), None


def _empirical_mean(values: Sequence[Any]) -> float | str:
    """Return a numeric empirical mean or one neutral common nonnumeric value."""
    selected = tuple(values)
    numeric = tuple(_numeric_parameter(value) for value in selected)
    if all(value is not None for value in numeric):
        return float(np.mean(np.asarray(numeric, dtype=np.float64)))
    displayed = tuple(_parameter_display(value) for value in selected)
    if displayed and all(value == displayed[0] for value in displayed[1:]):
        return displayed[0]
    return "not available"


def _dataset_boundary_means(
    records: tuple[GenerationInputDiagnostics, ...],
) -> Mapping[str, float | str]:
    """Calculate empirical means for maintained boundary scalar summaries."""
    result: dict[str, float | str] = {}
    for name in profiles.STATIONARY_FIXED_FIELDS:
        result[name] = float(np.mean([_fixed_value(record.case.payload, name) for record in records]))
    for statistic in STAT_NAMES:
        result[f"p_in_bc {statistic}"] = float(
            np.mean(
                [
                    _finite_real_scalar(
                        field_statistics(record).loc["p_in_bc", statistic],
                        label=f"p_in_bc {statistic}",
                    )
                    for record in records
                ]
            )
        )
    if is_transient(records[0]):
        for name in _SCHEDULE_NAMES:
            for label, attribute in (
                ("start", "start"),
                ("startup end", "end"),
                ("startup delta", "delta"),
            ):
                result[f"{name} {label}"] = float(
                    np.mean(
                        [
                            getattr(
                                _require_transient(record)[3].variables[name],
                                attribute,
                            )
                            for record in records
                        ]
                    )
                )
    return MappingProxyType(result)


def build_dataset_diagnostics(
    records: Sequence[GenerationInputDiagnostics],
) -> DatasetDiagnostics:
    """
    Calculate empirical means over every unique admitted case in one dataset.

    Raises
    ------
    ValueError
        If records are empty, duplicated, cross-profile, cross-batch, or expose
        incompatible parameter and field contracts.

    """
    selected = tuple(records)
    if not selected:
        msg_0 = "Dataset diagnostics require at least one admitted case."
        raise ValueError(msg_0)
    identities = tuple(str(record.case.payload["case_input_id"]) for record in selected)
    if len(identities) != len(set(identities)):
        msg_0 = "Dataset diagnostics require unique case-input identities."
        raise ValueError(msg_0)
    first = selected[0]
    payload = first.case.payload
    if first.case.batch_storage_name is None or first.case.campaign_purpose is None:
        msg_0 = "Dataset diagnostics require canonical batch storage and campaign-purpose metadata."
        raise ValueError(msg_0)
    contract = (
        first.profile_id,
        str(payload["batch_id"]),
        first.case.batch_storage_name,
        str(payload["batch_identity"]),
        str(payload["material_family"]),
        str(payload["sampling_regime"]),
        first.case.campaign_purpose,
    )
    for record in selected[1:]:
        other = record.case.payload
        if (
            record.profile_id,
            str(other["batch_id"]),
            record.case.batch_storage_name,
            str(other["batch_identity"]),
            str(other["material_family"]),
            str(other["sampling_regime"]),
            record.case.campaign_purpose,
        ) != contract:
            msg_0 = "Dataset means require one canonical batch dataset."
            raise ValueError(msg_0)
    names = tuple(payload["sampled_values"])
    units = {name: str(payload["sampled_units"][name]) for name in names}
    for record in selected:
        if (
            tuple(record.case.payload["sampled_values"]) != names
            or {name: str(record.case.payload["sampled_units"][name]) for name in names} != units
        ):
            msg_0 = "Dataset parameter inventories and units must match exactly."
            raise ValueError(msg_0)
        if display_field_names(record) != display_field_names(first):
            msg_0 = "Dataset field inventories must match exactly."
            raise ValueError(msg_0)
    parameter_means = MappingProxyType(
        {name: _empirical_mean([record.case.payload["sampled_values"][name] for record in selected]) for name in names}
    )
    summaries = {id(record): field_statistics(record) for record in selected}
    field_means = MappingProxyType(
        {
            (quantity, statistic): float(
                np.mean(
                    [
                        _finite_real_scalar(
                            summaries[id(record)].loc[quantity, statistic],
                            label=f"{quantity} {statistic}",
                        )
                        for record in selected
                    ]
                )
            )
            for quantity in display_field_names(first)
            for statistic in STAT_NAMES
        }
    )
    schedule_mean, schedule_message = _schedule_mean(selected)
    return DatasetDiagnostics(
        profile_id=contract[0],
        batch_id=contract[1],
        batch_storage_name=contract[2],
        batch_identity=contract[3],
        material_family=contract[4],
        sampling_regime=contract[5],
        campaign_purpose=contract[6],
        records=selected,
        parameter_means=parameter_means,
        parameter_units=MappingProxyType(units),
        field_summary_means=field_means,
        boundary_means=_dataset_boundary_means(selected),
        schedule_mean=schedule_mean,
        schedule_mean_unavailable=schedule_message,
    )


def display_unit(name: str, unit: str) -> str:
    """Return the EDA display unit without changing persisted physical units."""
    return "°C" if name in _ABSOLUTE_TEMPERATURE_DISPLAY_NAMES and unit == "K" else unit


def display_value(name: str, value: Any, unit: str) -> Any:
    """Return one EDA display value without mutating persisted source evidence."""
    if name in _ABSOLUTE_TEMPERATURE_DISPLAY_NAMES and unit == "K":
        return value - 273.15
    return value


def _unit_label(name: str, unit: str) -> str:
    """Return one canonical row label with its EDA display unit included once."""
    return f"{name} [{unit}]" if unit else name


def _comparison_frame(
    rows: Sequence[tuple[tuple[str, str, str], Any, Any, Any, Any]],
    *,
    index_name: str,
) -> pd.DataFrame:
    """Build the exact four-column comparison table without altering values."""
    index = pd.MultiIndex.from_tuples(
        tuple(row[0] for row in rows),
        names=("Section", "Category", index_name),
    )
    frame = pd.DataFrame(
        tuple(row[1:] for row in rows),
        index=index,
        columns=COMPARISON_COLUMNS,
    )
    if tuple(frame.columns) != COMPARISON_COLUMNS:
        msg_0 = "Generation-input comparison columns changed unexpectedly."
        raise RuntimeError(msg_0)
    return frame


def grouped_table_sections(
    table: pd.DataFrame,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Split one presentation-indexed table into stable top-level sections."""
    if not isinstance(table.index, pd.MultiIndex) or tuple(table.index.names) != (
        "Section",
        "Category",
        table.index.names[-1],
    ):
        msg_0 = "Grouped generation-input tables require a Section/Category/item index."
        raise ValueError(msg_0)
    sections = tuple(dict.fromkeys(str(value) for value in table.index.get_level_values("Section")))
    result: list[tuple[str, pd.DataFrame]] = []
    for section in sections:
        selected = table.xs(section, level="Section", drop_level=True)
        if not isinstance(selected, pd.DataFrame):
            msg_0 = f"Grouped table section {section!r} did not preserve a DataFrame."
            raise TypeError(msg_0)
        subset = selected.copy()
        subset.attrs = dict(table.attrs)
        result.append((section, subset))
    return tuple(result)


def _component_weights(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the exact canonical schedule-component mapping."""
    value = values.get("schedule.component_weights")
    if not isinstance(value, Mapping) or set(value) != set(table_schema.COMPONENT_WEIGHT_NAMES):
        msg_0 = "schedule.component_weights must contain smooth, event, and trend values."
        raise ValueError(msg_0)
    return value


def _expanded_parameter_names(values: Mapping[str, Any]) -> tuple[str, ...]:
    """Return category-ordered parameters with component weights expanded."""
    available = set(values)
    ordered = []
    consumed = set()
    for name in table_schema.PARAMETER_ORDER:
        if name.startswith("schedule.component_weights."):
            if "schedule.component_weights" in available:
                _component_weights(values)
                ordered.append(name)
                consumed.add("schedule.component_weights")
        elif name in available:
            ordered.append(name)
            consumed.add(name)
    ordered.extend(name for name in values if name not in consumed)
    return tuple(ordered)


def _parameter_root(name: str) -> str:
    """Return the persisted parameter name owning one displayed row."""
    prefix = "schedule.component_weights."
    return "schedule.component_weights" if name.startswith(prefix) else name


def _parameter_value(values: Mapping[str, Any], name: str) -> float | str:
    """Return one scalar display value, including an expanded component."""
    root = _parameter_root(name)
    if root == "schedule.component_weights":
        component = name.rsplit(".", maxsplit=1)[-1]
        numeric = _numeric_parameter(_component_weights(values)[component])
        if numeric is None:
            msg_0 = f"Schedule component weight {component!r} must be finite numeric evidence."
            raise ValueError(msg_0)
        return numeric
    return _parameter_display(values[root])


def _parameter_mean(dataset: DatasetDiagnostics, name: str) -> float | str:
    """Return one empirical parameter mean, including an expanded component."""
    root = _parameter_root(name)
    if root != "schedule.component_weights":
        return dataset.parameter_means[root]
    component = name.rsplit(".", maxsplit=1)[-1]
    return _empirical_mean(tuple(_component_weights(record.case.payload["sampled_values"])[component] for record in dataset.records))


def _parameter_row_index(name: str, unit: str) -> tuple[str, str, str]:
    """Return one grouped and EDA-display-unit-bearing parameter index."""
    spec = table_schema.parameter_row_spec(name)
    return spec.section, spec.category, _unit_label(spec.label, display_unit(name, unit))


def _field_row_index(
    quantity: str,
    statistic: str,
) -> tuple[str, str, str]:
    """Return one grouped and unit-bearing field-summary index."""
    spec = table_schema.field_summary_row_spec(quantity, statistic)
    return (
        spec.section,
        spec.category,
        _unit_label(spec.label, FIELD_UNITS[quantity]),
    )


def _boundary_row_index(name: str, unit: str) -> tuple[str, str, str]:
    """Return one grouped and EDA-display-unit-bearing boundary-summary index."""
    spec = table_schema.boundary_row_spec(name)
    return spec.section, spec.category, _unit_label(spec.label, display_unit(name, unit))


def case_context_table(
    first: GenerationInputDiagnostics,
    mean_a: DatasetDiagnostics,
    second: GenerationInputDiagnostics,
    mean_b: DatasetDiagnostics,
) -> pd.DataFrame:
    """Return compact side-by-side dataset, case, identity, and provenance evidence."""
    if first.profile_id != second.profile_id:
        msg_0 = "Case context comparison requires one simulation profile."
        raise ValueError(msg_0)
    values = {
        "dataset": (
            f"{mean_a.material_family.replace('_', ' ').title()} · {mean_a.sampling_regime}",
            f"{mean_b.material_family.replace('_', ' ').title()} · {mean_b.sampling_regime}",
        ),
        "material": (mean_a.material_family, mean_b.material_family),
        "regime": (mean_a.sampling_regime, mean_b.sampling_regime),
        "campaign purpose": (
            mean_a.campaign_purpose,
            mean_b.campaign_purpose,
        ),
        "simulation profile": (
            mean_a.profile_id,
            mean_b.profile_id,
        ),
        "profile label": (
            labels.profile_display_label(mean_a.profile_id),
            labels.profile_display_label(mean_b.profile_id),
        ),
        "status": (
            "input only; not executed",
            "input only; not executed",
        ),
        "case number": (first.case.case_index, second.case.case_index),
        "dataset case count": (mean_a.case_count, mean_b.case_count),
        "canonical batch": (mean_a.batch_id, mean_b.batch_id),
        "batch storage name": (
            mean_a.batch_storage_name,
            mean_b.batch_storage_name,
        ),
        "batch identity": (mean_a.batch_identity, mean_b.batch_identity),
        "case input identity": (
            first.case.payload["case_input_id"],
            second.case.payload["case_input_id"],
        ),
        "simulation case identity": (
            first.case.payload["simulation_case_id"],
            second.case.payload["simulation_case_id"],
        ),
        "input-generation source": (first.case.source_id, second.case.source_id),
        "git commit": (
            first.case.payload["git_commit"],
            second.case.payload["git_commit"],
        ),
        "available files": (
            ", ".join(first.case.payload["input_files"]),
            ", ".join(second.case.payload["input_files"]),
        ),
        "persistent directory": (
            str(first.case.directory),
            str(second.case.directory),
        ),
    }
    return pd.DataFrame(
        ((name, first_value, second_value) for name, (first_value, second_value) in values.items()),
        columns=("Item", "Case A", "Case B"),
    ).set_index("Item")


def parameter_comparison_table(
    first: GenerationInputDiagnostics,
    mean_a: DatasetDiagnostics,
    second: GenerationInputDiagnostics,
    mean_b: DatasetDiagnostics,
) -> pd.DataFrame:
    """Return every scalar parameter under the common four-column contract."""
    if first.profile_id != second.profile_id or mean_a.profile_id != mean_b.profile_id:
        msg_0 = "Parameter comparison requires compatible simulation profiles."
        raise ValueError(msg_0)
    first_values = first.case.payload["sampled_values"]
    second_values = second.case.payload["sampled_values"]
    if (
        tuple(first_values) != tuple(second_values)
        or tuple(first_values) != tuple(mean_a.parameter_means)
        or tuple(first_values) != tuple(mean_b.parameter_means)
    ):
        msg_0 = "Compared parameter inventories must match exactly."
        raise ValueError(msg_0)
    rows = []
    for name in _expanded_parameter_names(first_values):
        root = _parameter_root(name)
        unit = str(first.case.payload["sampled_units"][root])
        if unit != str(second.case.payload["sampled_units"][root]) or unit != mean_a.parameter_units[root] or unit != mean_b.parameter_units[root]:
            msg_0 = f"Compared parameter unit for {root!r} does not match exactly."
            raise ValueError(msg_0)
        rows.append(
            (
                _parameter_row_index(name, unit),
                display_value(name, _parameter_value(first_values, name), unit),
                display_value(name, _parameter_mean(mean_a, name), unit),
                display_value(name, _parameter_value(second_values, name), unit),
                display_value(name, _parameter_mean(mean_b, name), unit),
            )
        )
    frame = _comparison_frame(rows, index_name="Parameter")
    frame.attrs["mean_case_counts"] = (
        mean_a.case_count,
        mean_b.case_count,
    )
    return frame


def field_summary_comparison_table(
    first: GenerationInputDiagnostics,
    mean_a: DatasetDiagnostics,
    second: GenerationInputDiagnostics,
    mean_b: DatasetDiagnostics,
    *,
    quantities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return maintained field summaries under the common four-column contract."""
    if first.profile_id != second.profile_id:
        msg_0 = "Field-summary comparison requires one simulation profile."
        raise ValueError(msg_0)
    selected_quantities = None if quantities is None else frozenset(quantities)
    specs = tuple(spec for spec in field_summary_specs(first) if selected_quantities is None or spec[0] in selected_quantities)
    first_stats = field_statistics(first)
    second_stats = field_statistics(second)
    rows = [
        (
            _field_row_index(quantity, statistic),
            _finite_real_scalar(first_stats.loc[quantity, statistic], label=f"{quantity} {statistic} Case A"),
            mean_a.field_summary_means[(quantity, statistic)],
            _finite_real_scalar(second_stats.loc[quantity, statistic], label=f"{quantity} {statistic} Case B"),
            mean_b.field_summary_means[(quantity, statistic)],
        )
        for quantity, statistic in specs
    ]
    frame = _comparison_frame(rows, index_name="Field / statistic")
    frame.attrs["mean_case_counts"] = (
        mean_a.case_count,
        mean_b.case_count,
    )
    return frame


def _boundary_case_values(
    record: GenerationInputDiagnostics,
) -> dict[str, float | str]:
    """Return maintained boundary scalars for one case."""
    result: dict[str, float | str] = {name: _fixed_value(record.case.payload, name) for name in profiles.STATIONARY_FIXED_FIELDS}
    statistics = field_statistics(record).loc["p_in_bc"]
    result.update(
        {
            f"p_in_bc {statistic}": _finite_real_scalar(
                statistics[statistic],
                label=f"p_in_bc {statistic}",
            )
            for statistic in STAT_NAMES
        }
    )
    if is_transient(record):
        startup = _require_transient(record)[3]
        for name in _SCHEDULE_NAMES:
            variable = startup.variables[name]
            result.update(
                {
                    f"{name} start": variable.start,
                    f"{name} startup end": variable.end,
                    f"{name} startup delta": variable.delta,
                }
            )
    return result


def boundary_comparison_table(
    first: GenerationInputDiagnostics,
    mean_a: DatasetDiagnostics,
    second: GenerationInputDiagnostics,
    mean_b: DatasetDiagnostics,
) -> pd.DataFrame:
    """Return stationary and transient boundaries under one table contract."""
    if first.profile_id != second.profile_id:
        msg_0 = "Boundary comparison requires one simulation profile."
        raise ValueError(msg_0)
    first_values = _boundary_case_values(first)
    second_values = _boundary_case_values(second)
    if tuple(first_values) != tuple(second_values):
        msg_0 = "Compared boundary inventories must match exactly."
        raise ValueError(msg_0)
    units: dict[str, str] = {entry["name"]: str(entry["unit"]) for entry in first.case.payload["stationary_fixed_values"]}
    units.update({f"p_in_bc {statistic}": FIELD_UNITS["p_in_bc"] for statistic in STAT_NAMES})
    if is_transient(first):
        units.update(
            {
                f"{name} {label}": unit
                for name, unit in zip(
                    _SCHEDULE_NAMES,
                    _SCHEDULE_UNITS,
                    strict=True,
                )
                for label in ("start", "startup end", "startup delta")
            }
        )
    rows = [
        (
            _boundary_row_index(name, units[name]),
            display_value(name, first_values[name], units[name]),
            display_value(name, mean_a.boundary_means[name], units[name]),
            display_value(name, second_values[name], units[name]),
            display_value(name, mean_b.boundary_means[name], units[name]),
        )
        for name in first_values
    ]
    frame = _comparison_frame(rows, index_name="Boundary / statistic")
    frame.attrs["mean_case_counts"] = (
        mean_a.case_count,
        mean_b.case_count,
    )
    return frame


def dataset_parameter_table(dataset: DatasetDiagnostics) -> pd.DataFrame:
    """Return category-grouped parameter rows by actual dataset case number."""
    first_values = dataset.records[0].case.payload["sampled_values"]
    names = _expanded_parameter_names(first_values)
    index = pd.MultiIndex.from_tuples(
        tuple(
            _parameter_row_index(
                name,
                dataset.parameter_units[_parameter_root(name)],
            )
            for name in names
        ),
        names=("Section", "Category", "Parameter"),
    )
    columns = tuple(f"Case {record.case.case_index}" for record in dataset.records)
    values = tuple(
        tuple(
            display_value(
                name,
                _parameter_value(record.case.payload["sampled_values"], name),
                dataset.parameter_units[_parameter_root(name)],
            )
            for record in dataset.records
        )
        for name in names
    )
    return pd.DataFrame(values, index=index, columns=columns)


def dataset_field_summary_table(
    dataset: DatasetDiagnostics,
) -> pd.DataFrame:
    """Return category-grouped field-summary rows by dataset case number."""
    specs = field_summary_specs(dataset.records[0])
    summaries = tuple(field_statistics(record) for record in dataset.records)
    index = pd.MultiIndex.from_tuples(
        tuple(_field_row_index(quantity, statistic) for quantity, statistic in specs),
        names=("Section", "Category", "Field / statistic"),
    )
    columns = tuple(f"Case {record.case.case_index}" for record in dataset.records)
    values = tuple(
        tuple(
            _finite_real_scalar(
                summary.loc[quantity, statistic],
                label=f"{quantity} {statistic}",
            )
            for summary in summaries
        )
        for quantity, statistic in specs
    )
    return pd.DataFrame(values, index=index, columns=columns)


def spatial_difference_compatibility(
    first: GenerationInputDiagnostics,
    second: GenerationInputDiagnostics,
    quantity: str,
) -> str | None:
    """Return one exact spatial-difference incompatibility or None."""
    if first.profile_id != second.profile_id:
        return "Generation-input comparison requires cases from one simulation profile."
    if quantity not in display_field_names(first) or quantity not in display_field_names(second):
        return f"Field {quantity!r} is not available in both cases."
    if first.fields[quantity].shape != second.fields[quantity].shape:
        return f"Field shapes differ: A={first.fields[quantity].shape}, B={second.fields[quantity].shape}."
    for coordinate in ("x", "y"):
        if not np.array_equal(
            first.fields[coordinate],
            second.fields[coordinate],
        ):
            return f"Physical coordinate grid {coordinate!r} differs exactly between A and B."
    return None


def compatible_field_difference(
    first: GenerationInputDiagnostics,
    second: GenerationInputDiagnostics,
    quantity: str,
) -> np.ndarray:
    """Return immutable B-minus-A values after exact profile and grid checks."""
    incompatibility = spatial_difference_compatibility(
        first,
        second,
        quantity,
    )
    if incompatibility is not None:
        raise ValueError(incompatibility)
    return _immutable(second.fields[quantity] - first.fields[quantity])
