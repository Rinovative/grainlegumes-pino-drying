"""
===============================================================================
generation_publication_storage.py
===============================================================================
Convert validated COMSOL tables into one canonical HDF5 case payload.
Responsibilities:
  - Map explicit profile-config headers to stable logical fields
  - Validate Cartesian static/transient data, schedules, globals, and final status
  - Validate float32 conversion before atomic compressed HDF5 publication
Design principles:
  - CSV is a temporary adapter; case.h5 is the sole canonical case payload
  - No COMSOL tag, expression, sign convention, or filename is inferred
  - HDF5 field ordering, units, hashes, and identities are explicit metadata
This module does NOT:
  - Execute COMSOL, define scientific ranges, or register learning tasks
  - Preserve a parallel canonical CSV learning view
===============================================================================
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from src import common, domain
from src.generation.cases import generation_cases_admission as input_admission
from src.generation.cases import generation_cases_config as config_contract
from src.generation.cases import generation_cases_fields as fields_service
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.contracts import generation_contracts_comsol_spreadsheet as spreadsheet_contract
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.generation.cases.generation_cases_config import GenerationConfig

HDF5_SCHEMA_KIND = "vp2_canonical_case"
HDF5_SCHEMA_VERSION = config_contract.CANONICAL_HDF5_SCHEMA_VERSION
HDF5_CONVERTER_VERSION = config_contract.CANONICAL_HDF5_CONVERTER_VERSION
_MINIMUM_AXIS_POINTS = 2
_TABLE_RANK = 2
_COORDINATE_ATOL = 1e-12
_STATIONARITY_RTOL = profiles.STATIONARITY_TOLERANCE
_TIME_CLASSIFICATION_FACTOR = 16.0
_THERMODYNAMIC_ROUNDTRIP_ATOL = 64.0 * np.finfo(np.float64).eps
_UNIT_INTERVAL_ROUNDOFF_ATOL = 64.0 * np.finfo(np.float64).eps
_FLOAT32_VALIDATION_CHUNK_VALUES = 1_000_000
_SHA256_HEX_LENGTH = 64
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
STATUS_SCHEMA_VERSION = 1
_CASE_SCIENTIFIC_PROVENANCE_SCHEMA_KIND = "vp2_case_scientific_provenance"
_CASE_SCIENTIFIC_PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CanonicalCase:
    """One validated HDF5 payload and derived solver status."""

    path: Path
    status_path: Path
    status: dict[str, Any]
    source_export_hashes: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TransientBulkMoistureConsistency:
    """Canonical exported-versus-reconstructed transient bulk moisture."""

    time: np.ndarray
    exported: np.ndarray
    reconstructed: np.ndarray
    rtol: float
    atol: float
    matches: bool


def time_classification_tolerance(time_contract: Mapping[str, Any]) -> float:
    """Return the numerical state-time classification tolerance."""
    scale = max(
        1.0,
        abs(float(time_contract["start"])),
        abs(float(time_contract["stop"])),
        abs(float(time_contract["interval"])),
    )
    return _TIME_CLASSIFICATION_FACTOR * np.finfo(np.float64).eps * scale


def _time_classification_basis(time_contract: Mapping[str, Any]) -> str:
    """Describe the numerical state-time classification tolerance."""
    stop = float(time_contract["stop"])
    return f"{_TIME_CLASSIFICATION_FACTOR:g}*float64_epsilon*{stop:g}h; numerical classification only"


def _compression_matches(dataset: h5py.Dataset, storage: Mapping[str, Any]) -> bool:
    """Return whether one HDF5 dataset uses the configured filter contract."""
    return (
        dataset.compression == storage["compression"]
        and dataset.compression_opts == int(storage["compression_level"])
        and dataset.shuffle is bool(storage["shuffle"])
    )


def _contract(config: GenerationConfig, role: str) -> dict[str, Any]:
    """Return one exact resolved export mapping by logical role."""
    matches = [item for item in config.scientific_values["output_contract"]["exports"] if item["role"] == role]
    if len(matches) != 1:
        msg = f"Resolved profile requires exactly one mapping for export role {role!r}."
        raise RuntimeError(msg)
    return matches[0]


def _role_paths(exports: Sequence[Any], role: str) -> list[Path]:
    """Return collected source paths for one explicit role."""
    return [Path(item.source_path) for item in exports if item.role == role]


def _mapped_table(
    paths: Sequence[Path],
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Read explicitly mapped fields after rejecting probe-state columns."""
    role = str(contract["role"])
    expected_logical = tuple(contract["units"])
    if tuple(contract["columns"]) != expected_logical:
        unresolved = [logical for logical in expected_logical if logical not in contract["columns"]]
        message = f"Export role {role!r} cannot be ingested until mappings are confirmed for {unresolved}."
        raise RuntimeError(message)
    if not paths:
        message = f"Required export role {role!r} produced no files."
        raise FileNotFoundError(message)
    expected_units = {source: str(contract["units"][logical]) for logical, source in contract["columns"].items()}
    collected: dict[str, list[np.ndarray]] = {name: [] for name in expected_logical}
    for path in paths:
        table = spreadsheet_contract.read_comsol_spreadsheet(
            path,
            delimiter=str(contract["delimiter"]),
            expected_units=expected_units,
        )
        header = list(table.canonical_header)
        values = table.values
        if values is None:
            message = f"COMSOL Spreadsheet admission did not load numeric values: {path}"
            raise RuntimeError(message)
        missing = [source for source in contract["columns"].values() if source not in header]
        if missing:
            message = f"Export {path} is missing explicitly configured headers {missing}."
            raise ValueError(message)
        for logical, source in contract["columns"].items():
            collected[logical].append(values[:, header.index(source)])
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def _expected_axes(config: GenerationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return authoritative boundary-inclusive axes from the resolved grid."""
    grid = config.scientific_values["grid"]
    return (
        np.linspace(0.0, float(grid["Lx"]), int(grid["nx"]), dtype=np.float64),
        np.linspace(0.0, float(grid["Ly"]), int(grid["ny"]), dtype=np.float64),
    )


def _static_fields(
    config: GenerationConfig,
    exports: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize one possibly repeated stationary export onto y-by-x arrays."""
    contract = _contract(config, profiles.STEADY_FLOW_EXPORT_ROLE)
    mapped = _mapped_table(_role_paths(exports, profiles.STEADY_FLOW_EXPORT_ROLE), contract)
    x_axis, y_axis = _expected_axes(config)
    x_values = mapped["x"]
    y_values = mapped["y"]
    coordinate_rows: dict[tuple[int, int], list[int]] = {}
    for row, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        x_index = _axis_index(float(x_value), x_axis, label="static x")
        y_index = _axis_index(float(y_value), y_axis, label="static y")
        coordinate_rows.setdefault((y_index, x_index), []).append(row)
    if len(coordinate_rows) != x_axis.size * y_axis.size:
        msg = "Static export does not contain one complete Cartesian grid within coordinate tolerance."
        raise ValueError(msg)
    repeated_allowed = False
    field_names = profiles.static_field_names(config.profile.id)
    arrays: np.ndarray = np.empty((len(field_names), y_axis.size, x_axis.size), dtype=np.float64)
    for coordinate, rows in coordinate_rows.items():
        if len(rows) != 1 and not repeated_allowed:
            msg = f"Static export repeats coordinate index {coordinate} without configured time ownership."
            raise ValueError(msg)
        y_index, x_index = coordinate
        for field_index, name in enumerate(field_names):
            candidates = mapped[name][rows]
            if not np.allclose(candidates, candidates[0], rtol=_STATIONARITY_RTOL, atol=_STATIONARITY_RTOL):
                msg = f"Supposedly stationary field {name!r} varies beyond tolerance at {coordinate}."
                raise ValueError(msg)
            arrays[field_index, y_index, x_index] = candidates[0]
    determinant = arrays[0] * arrays[2] - arrays[1] ** 2
    if np.any(arrays[0] <= 0) or np.any(arrays[2] <= 0) or np.any(determinant <= 0):
        msg = "Canonical static permeability tensor is not positive definite."
        raise ValueError(msg)
    porosity = arrays[field_names.index("eps_bed")]
    if np.any((porosity <= 0) | (porosity >= 1)):
        msg = "Canonical static porosity must lie strictly inside (0, 1)."
        raise ValueError(msg)
    return x_axis, y_axis, arrays


def _classify_transient_times(
    times: np.ndarray,
    time_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Classify solver times as configured regular nodes plus one exact stop."""
    values = np.asarray(times, dtype=np.float64)
    start = float(time_contract["start"])
    stop = float(time_contract["stop"])
    interval = float(time_contract["interval"])
    tolerance = time_classification_tolerance(time_contract)
    maximum_index = len(time_contract["regular_times"]) - 1
    if (
        values.ndim != 1
        or values.size < 1
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0.0)
        or abs(float(values[0]) - start) > tolerance
        or float(values[-1]) > stop + tolerance
    ):
        msg = "Transient state times must be finite, increasing, begin at the configured start, and not exceed the configured stop."
        raise ValueError(msg)
    regular_by_index: dict[int, int] = {}
    irregular_positions: list[int] = []
    for position, value in enumerate(values):
        regular_index = round((float(value) - start) / interval)
        regular_value = start + regular_index * interval
        if 0 <= regular_index <= maximum_index and abs(float(value) - regular_value) <= tolerance:
            if regular_index in regular_by_index:
                msg = f"Transient export contains duplicate regular state index {regular_index}."
                raise ValueError(msg)
            regular_by_index[regular_index] = position
        else:
            irregular_positions.append(position)
    if len(irregular_positions) > 1:
        msg = "Transient export may contain at most one exact irregular stop state."
        raise ValueError(msg)
    if not regular_by_index or 0 not in regular_by_index:
        msg = "Transient export must contain at least the configured initial state."
        raise ValueError(msg)
    regular_indices = sorted(regular_by_index)
    if regular_indices != list(range(regular_indices[-1] + 1)):
        msg = "Regular transient states must form a contiguous configured prefix."
        raise ValueError(msg)
    regular_positions = np.asarray([regular_by_index[index] for index in regular_indices], dtype=np.int64)
    regular_times = start + np.asarray(regular_indices, dtype=np.float64) * interval
    irregular_position = irregular_positions[0] if irregular_positions else None
    if irregular_position is not None:
        irregular_time = float(values[irregular_position])
        if irregular_position != values.size - 1 or irregular_time <= float(regular_times[-1]) + tolerance:
            msg = "The optional irregular state must be final and follow the last regular state."
            raise ValueError(msg)
    return regular_times, regular_positions, irregular_position


def _axis_index(value: float, axis: np.ndarray, *, label: str) -> int:
    """Return one authoritative coordinate index within canonical tolerance."""
    insertion = int(np.searchsorted(axis, value))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < axis.size]
    if not candidates:
        message = f"Export {label} coordinate {value!r} lies outside the authoritative grid."
        raise ValueError(message)
    index = min(candidates, key=lambda candidate: abs(float(axis[candidate]) - value))
    if abs(float(axis[index]) - value) > _COORDINATE_ATOL:
        message = f"Export {label} coordinate {value!r} lies outside the authoritative grid."
        raise ValueError(message)
    return index


def _transient_fields(
    config: GenerationConfig,
    exports: Sequence[Any],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray | None]:
    """Stream one native wide export into regular and diagnostic state arrays."""
    contract = _contract(config, profiles.TRANSIENT_RAW_EXPORT_ROLE)
    paths = _role_paths(exports, profiles.TRANSIENT_RAW_EXPORT_ROLE)
    if len(paths) != 1:
        message = "Native wide transient admission requires exactly one configured Spreadsheet export."
        raise ValueError(message)
    expected_logical = tuple(contract["units"])
    if tuple(contract["columns"]) != expected_logical:
        unresolved = [logical for logical in expected_logical if logical not in contract["columns"]]
        message = f"Export role {profiles.TRANSIENT_RAW_EXPORT_ROLE!r} cannot be ingested until mappings are confirmed for {unresolved}."
        raise RuntimeError(message)
    source_by_logical = {logical: str(source) for logical, source in contract["columns"].items()}
    expected_units = {source_by_logical[logical]: str(contract["units"][logical]) for logical in expected_logical}
    path = paths[0]
    table = spreadsheet_contract.read_comsol_spreadsheet(
        path,
        delimiter=str(contract["delimiter"]),
        include_values=False,
    )
    groups = spreadsheet_contract.group_temporal_columns(
        table.raw_header,
        expected_units=expected_units,
    )
    numeric_rows = spreadsheet_contract.iter_comsol_spreadsheet_numeric_rows(
        path,
        delimiter=str(contract["delimiter"]),
        width=table.column_count,
    )
    try:
        first_row = next(numeric_rows)
    except StopIteration as error:
        message = "Native wide transient export contains no numeric state-time evidence."
        raise ValueError(message) from error
    source_t = source_by_logical["t"]
    observed_numeric_times = np.asarray(
        [float(first_row[group.column_index(source_t)]) for group in groups],
        dtype=np.float64,
    )
    regular_times, regular_positions, irregular_position = _classify_transient_times(
        observed_numeric_times,
        config.scientific_values["time"],
    )
    expected_rows = x_axis.size * y_axis.size
    if table.row_count != expected_rows:
        message = (
            "Native wide transient export must contain exactly one row per authoritative "
            f"grid node; expected={expected_rows}, observed={table.row_count}."
        )
        raise ValueError(message)
    regular_fields = np.full(
        (regular_times.size, len(profiles.TRANSIENT_FIELD_NAMES), y_axis.size, x_axis.size),
        np.nan,
        dtype=np.float64,
    )
    exact_stop_fields = (
        None
        if irregular_position is None
        else np.full(
            (len(profiles.TRANSIENT_FIELD_NAMES), y_axis.size, x_axis.size),
            np.nan,
            dtype=np.float64,
        )
    )
    regular_target = {int(position): index for index, position in enumerate(regular_positions)}
    occupied = np.zeros((y_axis.size, x_axis.size), dtype=bool)
    time_tolerance = time_classification_tolerance(config.scientific_values["time"])
    row_count = 0
    for row in chain((first_row,), numeric_rows):
        row_count += 1
        first = groups[0]
        reference_x = float(row[first.column_index(source_by_logical["x"])])
        reference_y = float(row[first.column_index(source_by_logical["y"])])
        x_index = _axis_index(reference_x, x_axis, label="x")
        y_index = _axis_index(reference_y, y_axis, label="y")
        if occupied[y_index, x_index]:
            message = f"Native wide transient export repeats grid coordinate {(reference_x, reference_y)}."
            raise ValueError(message)
        occupied[y_index, x_index] = True
        row_numeric_times = np.asarray(
            [float(row[group.column_index(source_by_logical["t"])]) for group in groups],
            dtype=np.float64,
        )
        if not np.allclose(row_numeric_times, observed_numeric_times, rtol=0.0, atol=time_tolerance):
            message = "Native wide transient numeric t columns must be spatially constant within canonical tolerance."
            raise ValueError(message)
        for state_index, group in enumerate(groups):
            numeric_time = float(row_numeric_times[state_index])
            header_tolerance = max(column.state_time_text_atol for column in group.columns)
            if abs(numeric_time - group.state_time) > max(time_tolerance, header_tolerance):
                message = f"Native wide transient numeric t disagrees with header-owned time {group.state_time:g}: observed={numeric_time!r}."
                raise ValueError(message)
            state_x = float(row[group.column_index(source_by_logical["x"])])
            state_y = float(row[group.column_index(source_by_logical["y"])])
            if abs(state_x - reference_x) > _COORDINATE_ATOL or abs(state_y - reference_y) > _COORDINATE_ATOL:
                message = f"Native wide transient x/y coordinates disagree across state {group.state_time:g}."
                raise ValueError(message)
            if state_index in regular_target:
                target = regular_fields[regular_target[state_index]]
            else:
                if state_index != irregular_position or exact_stop_fields is None:
                    message = "Transient state ownership is inconsistent with regular/exact-stop classification."
                    raise RuntimeError(message)
                target = exact_stop_fields
            for field_index, logical in enumerate(profiles.TRANSIENT_FIELD_NAMES):
                target[field_index, y_index, x_index] = row[group.column_index(source_by_logical[logical])]
    if row_count != table.row_count:
        message = f"Native wide transient streaming lost numeric rows: inspected={table.row_count}, converted={row_count}."
        raise RuntimeError(message)
    if not occupied.all() or not np.isfinite(regular_fields).all() or (exact_stop_fields is not None and not np.isfinite(exact_stop_fields).all()):
        message = "Native wide transient export contains an incomplete grid or non-finite canonical fields."
        raise ValueError(message)
    numeric_regular = observed_numeric_times[regular_positions]
    if not np.allclose(numeric_regular, regular_times, rtol=0.0, atol=time_tolerance):
        message = "Native wide transient regular numeric times disagree with the configured schedule prefix."
        raise ValueError(message)
    exact_stop_time = None if irregular_position is None else float(observed_numeric_times[irregular_position])
    return regular_times, regular_fields, exact_stop_time, exact_stop_fields


def _ordered_values(config: GenerationConfig, exports: Sequence[Any], role: str, names: tuple[str, ...]) -> np.ndarray:
    """Return one exact ordered global or final-status table."""
    contract = _contract(config, role)
    mapped = _mapped_table(_role_paths(exports, role), contract)
    lengths = {array.size for array in mapped.values()}
    if len(lengths) != 1:
        msg = f"Mapped export role {role!r} has inconsistent field lengths."
        raise ValueError(msg)
    values = np.column_stack([mapped[name] for name in names]).astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        msg = f"Mapped export role {role!r} contains non-finite values."
        raise ValueError(msg)
    return values


def _combined_state_time(
    regular_time: np.ndarray,
    exact_stop_time: float | None,
    exact_stop_fields: np.ndarray | None,
) -> np.ndarray:
    """Return the complete state-time axis without copying full state fields."""
    if (exact_stop_time is None) != (exact_stop_fields is None):
        message = "Exact-stop time and fields must either both be present or both be absent."
        raise ValueError(message)
    if exact_stop_time is None:
        return regular_time
    return np.concatenate((regular_time, np.asarray([exact_stop_time], dtype=np.float64)))


def transient_bulk_moisture_tolerance(
    config: GenerationConfig,
) -> tuple[float, float]:
    """Return the configured semantic tolerance for transient bulk moisture."""
    tolerance = config.scientific_values["validation"]["transient_bulk_moisture"]
    return float(tolerance["rtol"]), float(tolerance["atol"])


def evaluate_transient_bulk_moisture_consistency(
    config: GenerationConfig,
    static_fields: np.ndarray,
    state_time: np.ndarray,
    regular_fields: np.ndarray,
    exact_stop_fields: np.ndarray | None,
    global_values: np.ndarray,
    *,
    f_surf: float,
    time_tolerance: float,
) -> TransientBulkMoistureConsistency:
    """Reconstruct and compare the canonical transient bulk-moisture series."""
    rho_bu_dry = static_fields[profiles.TRANSIENT_STATIC_FIELD_NAMES.index("rho_bu_dry")]
    surface_index = profiles.TRANSIENT_FIELD_NAMES.index("w_surf")
    interior_index = profiles.TRANSIENT_FIELD_NAMES.index("w_int")
    states = iter(regular_fields) if exact_stop_fields is None else chain(iter(regular_fields), (exact_stop_fields,))
    weights = np.ones_like(rho_bu_dry, dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    expected = np.asarray(
        [
            domain.moisture.bulk_wet_basis_moisture(
                domain.moisture.granular_water_content(
                    state[surface_index],
                    state[interior_index],
                    f_surf,
                ),
                rho_bu_dry,
                cell_weights=weights,
            )
            for state in states
        ],
        dtype=np.float64,
    )
    global_time = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("t")]
    if global_time.shape != state_time.shape or not np.allclose(
        global_time,
        state_time,
        rtol=0.0,
        atol=time_tolerance,
    ):
        message = "Global diagnostics must contain exactly one row for every regular and optional exact-stop state."
        raise ValueError(message)
    column = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("X_wb_bulk")]
    if not np.isfinite(expected).all():
        message = "Reconstructed X_wb_bulk contains non-finite values."
        raise ValueError(message)
    if not np.isfinite(column).all():
        message = "Exported X_wb_bulk contains non-finite values."
        raise ValueError(message)
    rtol, atol = transient_bulk_moisture_tolerance(config)
    return TransientBulkMoistureConsistency(
        time=np.asarray(state_time, dtype=np.float64),
        exported=np.asarray(column, dtype=np.float64),
        reconstructed=expected,
        rtol=rtol,
        atol=atol,
        matches=bool(np.allclose(column, expected, rtol=rtol, atol=atol)),
    )


def _validate_global_bulk_moisture(
    config: GenerationConfig,
    static_fields: np.ndarray,
    state_time: np.ndarray,
    regular_fields: np.ndarray,
    exact_stop_fields: np.ndarray | None,
    global_values: np.ndarray,
    *,
    f_surf: float,
    time_tolerance: float,
) -> None:
    """Validate exported bulk moisture against integrated dry and water mass."""
    result = evaluate_transient_bulk_moisture_consistency(
        config,
        static_fields,
        state_time,
        regular_fields,
        exact_stop_fields,
        global_values,
        f_surf=f_surf,
        time_tolerance=time_tolerance,
    )
    if not result.matches:
        maximum = float(np.max(np.abs(result.exported - result.reconstructed)))
        message = f"Exported X_wb_bulk disagrees with weighted integrated dry and water mass; maximum error={maximum}."
        raise ValueError(message)


def _stationary_fixed_values(
    case_payload: Mapping[str, Any],
    fixed_values: Mapping[str, Any],
) -> np.ndarray:
    """Validate configured package-fixed conditioning without claiming file input."""
    entries = case_payload.get("stationary_fixed_values")
    ownership = case_payload.get("stationary_fixed_ownership")
    expected_entries = [
        {
            "name": name,
            "value": fixed_values[name],
            "unit": unit,
            "owner": "package_fixed",
            "runtime_source": "canonical_template",
        }
        for name, unit in zip(
            profiles.STATIONARY_FIXED_FIELDS,
            profiles.STATIONARY_FIXED_UNITS,
            strict=True,
        )
    ]
    expected_ownership = {
        name: {
            "owner": "package_fixed",
            "unit": unit,
            "fixed_value": fixed_values[name],
        }
        for name, unit in zip(
            profiles.STATIONARY_FIXED_FIELDS,
            profiles.STATIONARY_FIXED_UNITS,
            strict=True,
        )
    }
    if entries != expected_entries or ownership != expected_ownership:
        msg = "Stationary package-fixed provenance disagrees with the configured template contract."
        raise ValueError(msg)
    sampled = case_payload.get("sampled_values")
    profile_id = case_payload.get("simulation_profile")
    if not isinstance(sampled, Mapping):
        message = "Case sampled-value provenance must be a mapping."
        raise TypeError(message)
    if profile_id == profiles.STEADY_FLOW_PROFILE:
        if any(sampled.get(name) != fixed_values[name] for name in profiles.STATIONARY_FIXED_FIELDS):
            message = "Stationary package-fixed values are missing from steady case identity provenance."
            raise ValueError(message)
    elif profile_id == profiles.TRANSIENT_DRYING_PROFILE:
        if any(name in sampled for name in (*profiles.STATIONARY_FIXED_FIELDS, "f_wet_dm_max")):
            message = "Template-fixed values cannot be duplicated into transient sampled-value provenance."
            raise ValueError(message)
    else:
        message = f"Unsupported simulation profile in fixed-value provenance: {profile_id!r}."
        raise ValueError(message)
    return np.asarray(
        [fixed_values[name] for name in profiles.STATIONARY_FIXED_FIELDS],
        dtype=np.float64,
    )


def _schedule_values(
    case_payload: Mapping[str, Any],
    work_directory: Path,
    *,
    p_ref: float,
    time_contract: Mapping[str, Any],
    startup_ramp: Mapping[str, Any],
    initial_temperature: float,
) -> np.ndarray:
    """Read and revalidate the configured schedule bound by case.json."""
    spec = case_payload["input_contract"]["schedule"]
    path = work_directory / spec["filename"]
    identity = case_payload["input_files"][path.name]
    if common.serialization.file_sha256(path) != identity["sha256"] or path.stat().st_size != identity["size_bytes"]:
        msg = "Schedule adapter bytes changed after case-input identity was computed."
        raise RuntimeError(msg)
    header, values = input_admission.read_input_adapter_table(path, delimiter=spec["delimiter"])
    if header != list(profiles.SCHEDULE_FIELDS):
        msg = "Schedule adapter does not match the configured primitive-column contract."
        raise ValueError(msg)
    metadata = case_payload.get("schedule_diagnostics")
    if not isinstance(metadata, Mapping):
        msg = "Schedule handoff provenance must be a mapping."
        raise TypeError(msg)
    conversion = metadata.get("conversion_pressure")
    if conversion != {"name": "p_ref", "value": p_ref, "unit": "Pa", "owner": "package_fixed"}:
        msg = "Schedule conversion-pressure provenance is missing or inconsistent."
        raise ValueError(msg)
    schedule_service.validate_comsol_boundary_schedule(
        values,
        regular_times=np.asarray(time_contract["regular_times"], dtype=np.float64),
        startup_ramp=startup_ramp,
        initial_temperature=initial_temperature,
        pressure=p_ref,
        metadata=metadata,
    )
    return values


def _outside_unit_interval(values: np.ndarray) -> np.ndarray:
    """Return values outside [0, 1] beyond binary64 roundoff residue."""
    return (values < -_UNIT_INTERVAL_ROUNDOFF_ATOL) | (values > 1.0 + _UNIT_INTERVAL_ROUNDOFF_ATOL)


def transient_initial_state_tolerance(config: GenerationConfig) -> tuple[float, float]:
    """Return the configured semantic tolerance for transient initial states."""
    tolerance = config.scientific_values["validation"]["transient_initial_state"]
    return float(tolerance["rtol"]), float(tolerance["atol"])


def transient_initial_state_matches(
    config: GenerationConfig,
    actual: np.ndarray,
    expected: np.ndarray,
) -> bool:
    """Return whether a solved transient initial state is semantically canonical."""
    rtol, atol = transient_initial_state_tolerance(config)
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol))


def _validate_transient_outputs(
    config: GenerationConfig,
    static_fields: np.ndarray,
    regular_time: np.ndarray,
    regular_fields: np.ndarray,
    exact_stop_time: float | None,
    exact_stop_fields: np.ndarray | None,
    global_values: np.ndarray,
    final_status: np.ndarray,
    *,
    f_surf: float,
) -> None:
    """Validate final weighted moisture, diagnostic signs, and exact-stop alignment."""
    time_tolerance = time_classification_tolerance(config.scientific_values["time"])
    state_time = _combined_state_time(regular_time, exact_stop_time, exact_stop_fields)
    if global_values.shape != (state_time.size, len(profiles.GLOBAL_FIELD_NAMES)):
        message = "Global diagnostics must contain one complete row per exported solution state."
        raise ValueError(message)
    global_time = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("t")]
    if not np.allclose(global_time, state_time, rtol=0.0, atol=time_tolerance):
        message = "Global diagnostic times do not align with the complete exported state axis."
        raise ValueError(message)
    f_wet = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("f_wet_dm")]
    evaporation = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("m_dot_evap")]
    vapor_in = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("m_dot_v_in")]
    vapor_out = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("m_dot_v_out")]
    if np.any(_outside_unit_interval(f_wet)):
        message = "Global f_wet_dm must lie in [0, 1] within binary64 roundoff."
        raise ValueError(message)
    if np.any(evaporation < 0.0) or np.any(vapor_in < 0.0) or np.any(vapor_out < 0.0):
        message = "Canonical evaporation, inlet, and outlet mass-flow signs must be externally non-negative."
        raise ValueError(message)
    if final_status.shape != (1, len(profiles.FINAL_STATUS_FIELDS)):
        message = "Final status export must contain exactly one complete row."
        raise ValueError(message)
    final = dict(zip(profiles.FINAL_STATUS_FIELDS, final_status[0], strict=True))
    if abs(float(final["t_final"]) - float(state_time[-1])) > time_tolerance:
        message = "Final Status t_final must identify the actual last exported solution state."
        raise ValueError(message)
    if not np.isclose(float(final["f_wet_dm_final"]), float(f_wet[-1]), rtol=1e-6, atol=1e-9):
        message = "Final Status f_wet_dm_final disagrees with the complete global series."
        raise ValueError(message)
    if not np.isclose(
        float(final["X_wb_bulk_final"]),
        float(global_values[-1, profiles.GLOBAL_FIELD_NAMES.index("X_wb_bulk")]),
        rtol=1e-6,
        atol=1e-9,
    ):
        message = "Final Status X_wb_bulk_final disagrees with the complete global series."
        raise ValueError(message)
    static_names = profiles.TRANSIENT_STATIC_FIELD_NAMES
    rho_bu_dry = static_fields[static_names.index("rho_bu_dry")]
    x_initial = static_fields[static_names.index("X_0_db_field")]
    initial_water = rho_bu_dry * x_initial
    state_names = profiles.TRANSIENT_FIELD_NAMES
    for name in ("w_surf", "w_int"):
        if not transient_initial_state_matches(
            config,
            regular_fields[0, state_names.index(name)],
            initial_water,
        ):
            message = f"Initial {name} must equal rho_bu_dry*X_0_db_field without compartment splitting."
            raise ValueError(message)
    final_state = regular_fields[-1] if exact_stop_fields is None else exact_stop_fields
    final_water = domain.moisture.granular_water_content(
        final_state[state_names.index("w_surf")],
        final_state[state_names.index("w_int")],
        f_surf,
    )
    final_x_wb = domain.moisture.wet_basis_moisture(final_water, rho_bu_dry)
    if not np.isclose(float(final["X_wb_max_final"]), float(np.max(final_x_wb)), rtol=1e-6, atol=1e-9):
        message = "Final Status X_wb_max_final disagrees with the weighted Python-derived field."
        raise ValueError(message)
    final_temperature = final_state[state_names.index("T")]
    final_phi = final_state[state_names.index("phi")]
    extrema = {
        "T_min_final": float(np.min(final_temperature)),
        "T_max_final": float(np.max(final_temperature)),
        "phi_min_final": float(np.min(final_phi)),
        "phi_max_final": float(np.max(final_phi)),
    }
    if (
        not np.isclose(float(final["T_min_final"]), extrema["T_min_final"], rtol=1e-6, atol=1e-9)
        or not np.isclose(float(final["T_max_final"]), extrema["T_max_final"], rtol=1e-6, atol=1e-9)
        or not np.isclose(float(final["phi_min_final"]), extrema["phi_min_final"], rtol=1e-6, atol=1e-9)
        or not np.isclose(float(final["phi_max_final"]), extrema["phi_max_final"], rtol=1e-6, atol=1e-9)
    ):
        message = "Final Status temperature or relative-humidity extrema disagree with the actual final state."
        raise ValueError(message)
    _validate_global_bulk_moisture(
        config,
        static_fields,
        state_time,
        regular_fields,
        exact_stop_fields,
        global_values,
        f_surf=f_surf,
        time_tolerance=time_tolerance,
    )


def validate_float32_conversion(values: np.ndarray, *, rtol: float, atol: float, label: str) -> np.ndarray:
    """Convert large fields only after explicit finite allclose validation."""
    source = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source).all():
        msg = f"{label} contains non-finite values before float32 conversion."
        raise ValueError(msg)
    converted = source.astype(np.float32)
    source_flat = source.reshape(-1)
    converted_flat = converted.reshape(-1)
    maximum_error = 0.0
    for start in range(0, source_flat.size, _FLOAT32_VALIDATION_CHUNK_VALUES):
        stop = min(start + _FLOAT32_VALIDATION_CHUNK_VALUES, source_flat.size)
        restored = converted_flat[start:stop].astype(np.float64)
        source_chunk = source_flat[start:stop]
        if not np.allclose(source_chunk, restored, rtol=rtol, atol=atol):
            maximum_error = max(maximum_error, float(np.max(np.abs(source_chunk - restored))))
    if maximum_error > 0.0:
        msg = f"{label} float32 conversion exceeds configured tolerance; maximum absolute error={maximum_error}."
        raise ValueError(msg)
    return converted


def _json_attribute(value: Any) -> str:
    """Serialize structured HDF5 metadata canonically."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _source_hashes(exports: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Return relative-path, role, hash, and size evidence for every adapter export."""
    return {
        item.relative_path.as_posix(): {
            "role": item.role,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in exports
    }


def _status(
    config: GenerationConfig,
    *,
    transient_time: np.ndarray | None,
    exact_stop_time: float | None,
    final_status: np.ndarray | None,
    global_values: np.ndarray | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Derive status while retaining observed QA without a pass tolerance."""
    if transient_time is None or final_status is None or global_values is None:
        t_final = None
        wet_final = None
        target_reached = None
        hit_t_max = None
        regular_states = 0
        last_regular = None
        mass_balance = None
    else:
        values = dict(zip(profiles.FINAL_STATUS_FIELDS, final_status[0], strict=True))
        t_final = float(values["t_final"])
        wet_final = float(values["f_wet_dm_final"])
        limit = float(config.scientific_values["scientific_fixed_values"]["f_wet_dm_max"])
        target_reached = wet_final <= limit
        hit_t_max = t_final >= float(config.scientific_values["time"]["stop"])
        regular_states = int(transient_time.size)
        last_regular = float(transient_time[-1])
        residual = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("mt_mass_balance")]
        maximum_position = int(np.argmax(np.abs(residual)))
        mass_balance = {
            "maximum_absolute": float(abs(residual[maximum_position])),
            "time_of_maximum_absolute": float(global_values[maximum_position, 0]),
            "final": float(residual[-1]),
            "unit": "kg/s",
            "acceptance_tolerance": None,
        }
    return {
        "schema_kind": "simulation_case_status",
        "schema_version": STATUS_SCHEMA_VERSION,
        "solver_success": True,
        "target_reached": target_reached,
        "hit_t_max": hit_t_max,
        "t_stop_exact": t_final,
        "t_last_regular": last_regular,
        "has_exact_stop_state": exact_stop_time is not None,
        "exact_stop_state_time": exact_stop_time,
        "n_regular_states": regular_states,
        "n_regular_steps": max(regular_states - 1, 0),
        "f_wet_dm_final": wet_final,
        "runtime_s": float(runtime_seconds),
        "units": {
            "t_stop_exact": "h",
            "t_last_regular": "h",
            "exact_stop_state_time": "h",
            "f_wet_dm_final": "1",
            "runtime_s": "s",
        },
        "mass_balance_qa": mass_balance,
        "contains_nan_or_inf": False,
        "field_shape_valid": True,
        "schedule_valid": None if transient_time is None else True,
    }


def _write_json_dataset(group: h5py.Group, name: str, value: Any) -> None:
    """Write one canonical JSON provenance record as UTF-8 text."""
    group.create_dataset(name, data=_json_attribute(value), dtype=h5py.string_dtype(encoding="utf-8"))


def _case_scientific_provenance(case_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return realized case science required inside the canonical HDF5 payload."""
    required = (
        "case_id",
        "case_index",
        "case_input_id",
        "simulation_case_id",
        "material_family",
        "material_role",
        "evaluation_regime",
        "sampling_regime",
        "natural_support_state",
        "seed_evidence",
        "block_provenance",
        "sampled_values",
        "sampled_units",
        "coupled_selections",
        "ood",
        "spatial_diagnostics",
    )
    missing = [name for name in required if name not in case_payload]
    if missing:
        message = f"Case payload is missing realized HDF5 scientific provenance {missing}."
        raise ValueError(message)
    result = {
        "schema_kind": _CASE_SCIENTIFIC_PROVENANCE_SCHEMA_KIND,
        "schema_version": _CASE_SCIENTIFIC_PROVENANCE_SCHEMA_VERSION,
        **{name: case_payload[name] for name in required},
    }
    for name in ("schedule_diagnostics", "pilot_check"):
        if name in case_payload:
            result[name] = case_payload[name]
    return result


def _hdf5_scalar_contract(
    profile_id: str,
    admission: scalar_handoff_contract.ScalarHandoffAdmission | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exact scalar names and units after profile-specific admission."""
    if profile_id == profiles.STEADY_FLOW_PROFILE:
        if admission is not None:
            message = "Steady HDF5 publication cannot receive a transient scalar handoff."
            raise ValueError(message)
        return profiles.STATIONARY_FIXED_FIELDS, profiles.STATIONARY_FIXED_UNITS
    if admission is None:
        message = "Transient HDF5 publication requires admitted scalar metadata."
        raise ValueError(message)
    return admission.field_names, admission.units


def _write_hdf5(
    path: Path,
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    static_fields: np.ndarray,
    scalar_values: np.ndarray,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None,
    transient_time: np.ndarray | None,
    transient_fields: np.ndarray | None,
    exact_stop_time: float | None,
    exact_stop_fields: np.ndarray | None,
    schedule_values: np.ndarray | None,
    global_values: np.ndarray | None,
    final_status: np.ndarray | None,
    source_hashes: Mapping[str, Any],
) -> None:
    """Write one complete schema-v1 case payload to a private temporary path."""
    storage = config.scientific_values["storage"]
    static_float32 = validate_float32_conversion(
        static_fields,
        rtol=float(storage["float32_rtol"]),
        atol=float(storage["float32_atol"]),
        label="static fields",
    )
    transient_float32 = None
    if transient_fields is not None:
        transient_float32 = validate_float32_conversion(
            transient_fields,
            rtol=float(storage["float32_rtol"]),
            atol=float(storage["float32_atol"]),
            label="regular transient fields",
        )
    exact_stop_float32 = None
    if exact_stop_fields is not None:
        exact_stop_float32 = validate_float32_conversion(
            exact_stop_fields,
            rtol=float(storage["float32_rtol"]),
            atol=float(storage["float32_atol"]),
            label="exact-stop transient fields",
        )
    compression = {
        "compression": storage["compression"],
        "compression_opts": int(storage["compression_level"]),
        "shuffle": bool(storage["shuffle"]),
    }
    static_names = profiles.static_field_names(config.profile.id)
    static_units = profiles.static_field_units(config.profile.id)
    scalar_names, scalar_units = _hdf5_scalar_contract(
        config.profile.id,
        scalar_handoff,
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_kind"] = HDF5_SCHEMA_KIND
        handle.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        handle.attrs["converter_version"] = storage["converter_version"]
        for key in (
            "simulation_profile",
            "case_id",
            "material_family",
            "material_role",
            "evaluation_regime",
            "sampling_regime",
            "natural_support_state",
            "case_input_id",
            "simulation_case_id",
            "scientific_config_digest",
            "export_contract_sha256",
            "airflow_source",
        ):
            handle.attrs[key] = case_payload[key]
        case_index = case_payload["case_index"]
        if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 1:
            message = "Case index must be a positive integer before HDF5 publication."
            raise ValueError(message)
        handle.attrs["case_index"] = case_index
        handle.attrs["template_sha256"] = case_payload["template"]["sha256"]
        handle.attrs["available_learning_views"] = _json_attribute(case_payload["available_learning_views"])
        provenance = handle.create_group("provenance")
        _write_json_dataset(
            provenance,
            "scientific_config_json",
            config_contract.scientific_config_identity_payload(config.scientific_values),
        )
        _write_json_dataset(
            provenance,
            "case_scientific_provenance_json",
            _case_scientific_provenance(case_payload),
        )
        _write_json_dataset(provenance, "input_files_json", case_payload["input_files"])
        _write_json_dataset(provenance, "source_exports_json", source_hashes)
        _write_json_dataset(
            provenance,
            "template_json",
            {
                "sha256": case_payload["template"]["sha256"],
                "sha256_validation": "pass",
                "comsol_internal_contract": "runtime_validation_required",
            },
        )
        if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
            _write_json_dataset(
                provenance,
                "scalar_handoff_json",
                case_payload["scalar_handoff"],
            )
        _write_json_dataset(
            provenance,
            "stationary_fixed_ownership_json",
            case_payload["stationary_fixed_ownership"],
        )
        coordinates = handle.create_group("coords")
        x_dataset = coordinates.create_dataset("x", data=np.asarray(x_axis, dtype=np.float64))
        y_dataset = coordinates.create_dataset("y", data=np.asarray(y_axis, dtype=np.float64))
        x_dataset.attrs["unit"] = "m"
        y_dataset.attrs["unit"] = "m"
        static = handle.create_group("static")
        static_dataset = static.create_dataset(
            "fields",
            data=static_float32,
            chunks=(1, min(int(storage["chunk_y"]), y_axis.size), min(int(storage["chunk_x"]), x_axis.size)),
            **compression,
        )
        static_dataset.attrs["field_names"] = _json_attribute(list(static_names))
        static_dataset.attrs["units"] = _json_attribute(list(static_units))
        parameter_group_name = "stationary_fixed" if config.profile.id == profiles.STEADY_FLOW_PROFILE else "scalar"
        scalar_group = handle.create_group(parameter_group_name)
        scalar_dataset = scalar_group.create_dataset(
            "values",
            data=np.asarray(scalar_values, dtype=np.float64),
        )
        scalar_dataset.attrs["field_names"] = _json_attribute(list(scalar_names))
        scalar_dataset.attrs["units"] = _json_attribute(list(scalar_units))
        if config.profile.id == profiles.STEADY_FLOW_PROFILE:
            scalar_dataset.attrs["ownership"] = _json_attribute(["package_fixed"] * len(scalar_names))
            scalar_dataset.attrs["runtime_source"] = "canonical_template"
        else:
            if scalar_handoff is None:
                message = "Transient HDF5 publication requires admitted scalar ownership."
                raise ValueError(message)
            scalar_dataset.attrs["ownership"] = _json_attribute(list(scalar_handoff.ownership))
        if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
            if transient_time is None or transient_float32 is None or schedule_values is None or global_values is None or final_status is None:
                msg = "Transient HDF5 publication requires every regular, schedule, global, and final-status value."
                raise ValueError(msg)
            time_dataset = handle.create_dataset("time", data=np.asarray(transient_time, dtype=np.float64))
            time_dataset.attrs["unit"] = "h"
            time_contract = config.scientific_values["time"]
            time_dataset.attrs["classification_atol"] = time_classification_tolerance(time_contract)
            time_dataset.attrs["classification_basis"] = _time_classification_basis(time_contract)
            transient = handle.create_group("transient")
            transient_dataset = transient.create_dataset(
                "fields",
                data=transient_float32,
                chunks=(
                    min(int(storage["chunk_time"]), transient_time.size),
                    1,
                    min(int(storage["chunk_y"]), y_axis.size),
                    min(int(storage["chunk_x"]), x_axis.size),
                ),
                **compression,
            )
            transient_dataset.attrs["field_names"] = _json_attribute(list(profiles.TRANSIENT_FIELD_NAMES))
            transient_dataset.attrs["units"] = _json_attribute(list(profiles.TRANSIENT_FIELD_UNITS))
            if exact_stop_time is not None and exact_stop_float32 is not None:
                exact_stop = handle.create_group("exact_stop")
                exact_time_dataset = exact_stop.create_dataset("time", data=np.asarray([exact_stop_time], dtype=np.float64))
                exact_time_dataset.attrs["unit"] = "h"
                exact_fields_dataset = exact_stop.create_dataset(
                    "fields",
                    data=exact_stop_float32,
                    chunks=(1, min(int(storage["chunk_y"]), y_axis.size), min(int(storage["chunk_x"]), x_axis.size)),
                    **compression,
                )
                exact_fields_dataset.attrs["field_names"] = _json_attribute(list(profiles.TRANSIENT_FIELD_NAMES))
                exact_fields_dataset.attrs["units"] = _json_attribute(list(profiles.TRANSIENT_FIELD_UNITS))
                exact_stop.attrs["usage"] = "diagnostic_only_no_training_transition"
            schedule = handle.create_group("schedule")
            schedule_dataset = schedule.create_dataset("values", data=np.asarray(schedule_values, dtype=np.float64), **compression)
            schedule_dataset.attrs["field_names"] = _json_attribute(list(profiles.SCHEDULE_FIELDS))
            schedule_dataset.attrs["units"] = _json_attribute(list(profiles.SCHEDULE_UNITS))
            schedule_dataset.attrs["conversion_pressure"] = _json_attribute(case_payload["schedule_diagnostics"]["conversion_pressure"])
            schedule_dataset.attrs["humidity_conversion_owner"] = "generation_schedule"
            schedule_dataset.attrs["boundary_handoff"] = _json_attribute(case_payload["schedule_diagnostics"]["boundary_handoff"])
            global_group = handle.create_group("global")
            global_dataset = global_group.create_dataset("values", data=np.asarray(global_values, dtype=np.float64), **compression)
            global_dataset.attrs["field_names"] = _json_attribute(list(profiles.GLOBAL_FIELD_NAMES))
            global_dataset.attrs["units"] = _json_attribute(list(profiles.GLOBAL_FIELD_UNITS))
            final_group = handle.create_group("final_status")
            final_dataset = final_group.create_dataset("values", data=np.asarray(final_status[0], dtype=np.float64))
            final_dataset.attrs["field_names"] = _json_attribute(list(profiles.FINAL_STATUS_FIELDS))
            final_dataset.attrs["units"] = _json_attribute(list(profiles.FINAL_STATUS_UNITS))
        handle.flush()


def _hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required HDF5 dataset with a path-rich type error."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        msg = f"Canonical HDF5 member {name!r} must be a dataset."
        raise TypeError(msg)
    return value


def _hdf5_text_attribute(value: Any, *, label: str) -> str:
    """Return one required textual HDF5 attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        msg = f"Canonical HDF5 attribute {label!r} must be text."
        raise TypeError(msg)
    return value


def _hdf5_json_dataset(handle: h5py.File, name: str) -> Any:
    """Load one canonical JSON text dataset."""
    dataset = _hdf5_dataset(handle, name)
    if dataset.shape != ():
        msg = f"Canonical JSON provenance member {name!r} must be scalar text."
        raise ValueError(msg)
    return json.loads(_hdf5_text_attribute(dataset[()], label=name))


def _is_sha256(value: Any) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""
    return isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH and set(value).issubset(_SHA256_CHARACTERS)


def _safe_relative_path(value: Any, *, label: str) -> str:
    """Validate one portable repository- or artifact-relative path."""
    if not isinstance(value, str) or not value or "\\" in value:
        msg = f"{label} must be a non-empty POSIX relative path."
        raise ValueError(msg)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != value:
        msg = f"{label} must be a safe normalized POSIX relative path."
        raise ValueError(msg)
    return value


def _validate_hash_record(value: Any, *, label: str, include_role: bool) -> str | None:
    """Validate one exact portable file hash/size record and return its role."""
    expected = {"sha256", "size_bytes", "role"} if include_role else {"sha256", "size_bytes"}
    if not isinstance(value, dict) or set(value) != expected or not _is_sha256(value.get("sha256")):
        msg = f"{label} has an invalid hash record."
        raise ValueError(msg)
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        msg = f"{label}.size_bytes must be a positive integer."
        raise ValueError(msg)
    role = value.get("role")
    if include_role and (not isinstance(role, str) or not role):
        msg = f"{label}.role must be non-empty text."
        raise ValueError(msg)
    return role if isinstance(role, str) else None


def _validate_input_file_provenance(value: Any, *, profile: str) -> None:
    """Validate exact profile-specific input-file identities."""
    expected = {"fields.csv"}
    if profile == profiles.TRANSIENT_DRYING_PROFILE:
        expected.update({"scalars.csv", "schedule.csv"})
    if not isinstance(value, dict) or set(value) != expected:
        msg = f"Canonical input-file provenance must contain exactly {sorted(expected)}."
        raise ValueError(msg)
    for filename, record in value.items():
        _validate_hash_record(record, label=f"input_files[{filename!r}]", include_role=False)


def _validate_source_export_provenance(
    value: Any,
    *,
    profile_contract: profiles.SimulationProfile,
) -> None:
    """Validate exact source-export roles with portable hashes and multiplicity."""
    if not isinstance(value, dict) or not value:
        msg = "Canonical source-export provenance must be one non-empty mapping."
        raise ValueError(msg)
    role_counts = {spec.role: 0 for spec in profile_contract.export_roles}
    for relative_path, record in value.items():
        _safe_relative_path(relative_path, label="source export path")
        role = _validate_hash_record(record, label=f"source_exports[{relative_path!r}]", include_role=True)
        if role not in role_counts:
            msg = f"Canonical source export declares an unknown role {role!r}."
            raise ValueError(msg)
        role_counts[role] += 1
    for spec in profile_contract.export_roles:
        count = role_counts[spec.role]
        if (spec.required and count < 1) or (not spec.allow_multiple and count > 1):
            msg = f"Canonical source-export multiplicity is invalid for role {spec.role!r}: {count}."
            raise ValueError(msg)


def _dataset_is_finite(dataset: h5py.Dataset) -> bool:
    """Check a field dataset incrementally without materializing a trajectory."""
    if not dataset.shape:
        return bool(np.isfinite(np.asarray(dataset)).all())
    return all(np.isfinite(np.asarray(dataset[index])).all() for index in range(dataset.shape[0]))


def _require_group_members(handle: h5py.File, name: str, expected: set[str]) -> h5py.Group:
    """Return one group only when its direct membership is exact."""
    group = handle.get(name)
    if not isinstance(group, h5py.Group) or set(group) != expected:
        msg = f"Canonical HDF5 group {name!r} must contain exactly {sorted(expected)}."
        raise ValueError(msg)
    return group


def _dataset_contract(dataset: h5py.Dataset, *, names: tuple[str, ...], units: tuple[str, ...], label: str) -> None:
    """Validate exact ordered name and unit metadata for one dataset."""
    observed_names = json.loads(_hdf5_text_attribute(dataset.attrs.get("field_names", ""), label=f"{label}.field_names"))
    observed_units = json.loads(_hdf5_text_attribute(dataset.attrs.get("units", ""), label=f"{label}.units"))
    if observed_names != list(names) or observed_units != list(units):
        msg = f"Canonical {label} field-name or unit metadata is invalid."
        raise ValueError(msg)


def _validate_hdf5_members(handle: h5py.File, profile: str) -> None:
    """Require exact root membership for one source profile."""
    expected = {"coords", "static", "provenance"}
    if profile == profiles.STEADY_FLOW_PROFILE:
        expected.add("stationary_fixed")
    if profile == profiles.TRANSIENT_DRYING_PROFILE:
        expected |= {"scalar", "time", "transient", "schedule", "global", "final_status"}
        if "exact_stop" in handle:
            expected.add("exact_stop")
    if set(handle) != expected:
        msg = f"Canonical HDF5 group membership mismatch for profile {profile!r}: {sorted(handle)}"
        raise ValueError(msg)


def _validate_hdf5_provenance(
    handle: h5py.File,
    profile: str,
    profile_contract: profiles.SimulationProfile,
) -> tuple[dict[str, Any], str | None, scalar_handoff_contract.ScalarHandoffAdmission | None, dict[str, Any]]:
    """Validate complete scientific, input, conditioning, and template provenance."""
    expected = {
        "scientific_config_json",
        "case_scientific_provenance_json",
        "input_files_json",
        "source_exports_json",
        "template_json",
        "stationary_fixed_ownership_json",
    }
    if profile == profiles.TRANSIENT_DRYING_PROFILE:
        expected.add("scalar_handoff_json")
    _require_group_members(handle, "provenance", expected)
    scientific = _hdf5_json_dataset(handle, "provenance/scientific_config_json")
    case_scientific = _hdf5_json_dataset(
        handle,
        "provenance/case_scientific_provenance_json",
    )
    input_files = _hdf5_json_dataset(handle, "provenance/input_files_json")
    source_exports = _hdf5_json_dataset(handle, "provenance/source_exports_json")
    template = _hdf5_json_dataset(handle, "provenance/template_json")
    fixed_ownership = _hdf5_json_dataset(
        handle,
        "provenance/stationary_fixed_ownership_json",
    )
    expected_fixed = {
        name: {
            "owner": "package_fixed",
            "unit": unit,
            "fixed_value": scientific["scientific_fixed_values"][name],
        }
        for name, unit in zip(
            profiles.STATIONARY_FIXED_FIELDS,
            profiles.STATIONARY_FIXED_UNITS,
            strict=True,
        )
    }
    if (
        not isinstance(scientific, dict)
        or scientific.get("schema_version") != config_contract.CONFIG_SCHEMA_VERSION
        or scientific.get("simulation_profile") != profile
        or fixed_ownership != expected_fixed
    ):
        msg = "Canonical HDF5 scientific or stationary-conditioning provenance is invalid."
        raise ValueError(msg)
    required_case_keys = {
        "schema_kind",
        "schema_version",
        "case_id",
        "case_index",
        "case_input_id",
        "simulation_case_id",
        "material_family",
        "material_role",
        "evaluation_regime",
        "sampling_regime",
        "natural_support_state",
        "seed_evidence",
        "block_provenance",
        "sampled_values",
        "sampled_units",
        "coupled_selections",
        "ood",
        "spatial_diagnostics",
    }
    optional_case_keys = {"schedule_diagnostics", "pilot_check"}
    if (
        not isinstance(case_scientific, dict)
        or not required_case_keys.issubset(case_scientific)
        or set(case_scientific).difference(required_case_keys | optional_case_keys)
        or case_scientific.get("schema_kind") != _CASE_SCIENTIFIC_PROVENANCE_SCHEMA_KIND
        or case_scientific.get("schema_version") != _CASE_SCIENTIFIC_PROVENANCE_SCHEMA_VERSION
    ):
        msg = "Canonical HDF5 realized case-scientific provenance schema is invalid."
        raise ValueError(msg)
    matched_attributes = (
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "material_family",
        "material_role",
        "evaluation_regime",
        "sampling_regime",
        "natural_support_state",
    )
    if any(case_scientific[name] != handle.attrs.get(name) for name in matched_attributes) or case_scientific["case_index"] != handle.attrs.get(
        "case_index"
    ):
        msg = "Canonical HDF5 case-scientific provenance disagrees with identity attributes."
        raise ValueError(msg)
    sampled_values = case_scientific["sampled_values"]
    sampled_units = case_scientific["sampled_units"]
    block_provenance = case_scientific["block_provenance"]
    ood = case_scientific["ood"]
    if (
        not isinstance(sampled_values, dict)
        or not sampled_values
        or not isinstance(sampled_units, dict)
        or set(sampled_values) != set(sampled_units)
        or not isinstance(block_provenance, dict)
        or not block_provenance
        or not isinstance(case_scientific["seed_evidence"], dict)
        or not isinstance(case_scientific["coupled_selections"], dict)
        or not isinstance(case_scientific["spatial_diagnostics"], dict)
        or not isinstance(ood, dict)
        or ood.get("natural_support_state") != case_scientific["natural_support_state"]
    ):
        msg = "Canonical HDF5 realized values, units, seeds, blocks, diagnostics, or OOD provenance are invalid."
        raise ValueError(msg)
    diagnostics = case_scientific["spatial_diagnostics"]
    try:
        fields_service.validate_porosity_diagnostics(diagnostics["porosity"])
    except (KeyError, TypeError, ValueError) as error:
        msg_0 = "Canonical HDF5 porosity diagnostics are invalid."
        raise ValueError(msg_0) from error
    material_contract = scientific.get("material")
    if isinstance(material_contract, dict):
        active_names = material_contract.get("active_coordinate_names")
        active_blocks = material_contract.get("active_sampling_blocks")
        registry = material_contract.get("parameter_registry")
        if (
            not isinstance(active_names, list)
            or not set(active_names).issubset(sampled_values)
            or not isinstance(active_blocks, list)
            or set(active_blocks) != set(block_provenance)
            or not isinstance(registry, dict)
        ):
            msg = "Canonical HDF5 realized provenance does not cover every profile-active coordinate and block."
            raise ValueError(msg)
    if profile == profiles.TRANSIENT_DRYING_PROFILE and "schedule_diagnostics" not in case_scientific:
        msg = "Canonical transient HDF5 is missing realized schedule diagnostics."
        raise ValueError(msg)
    _validate_input_file_provenance(input_files, profile=profile)
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None = None
    if profile == profiles.TRANSIENT_DRYING_PROFILE:
        raw_handoff = _hdf5_json_dataset(
            handle,
            "provenance/scalar_handoff_json",
        )
        if (
            not isinstance(raw_handoff, dict)
            or set(raw_handoff)
            != {
                "mechanism",
                "filename",
                "fresh_per_case",
                "runtime_validation",
                "entries",
            }
            or raw_handoff.get("mechanism") != "case_local_long_form_csv"
            or raw_handoff.get("filename") != "scalars.csv"
            or raw_handoff.get("fresh_per_case") is not True
            or raw_handoff.get("runtime_validation") != "required"
        ):
            msg = "Canonical transient scalar-handoff provenance is invalid."
            raise ValueError(msg)
        scalar_identity = input_files[raw_handoff["filename"]]
        scalar_handoff = scalar_handoff_contract.admit_transient_scalar_handoff(
            raw_handoff["entries"],
            source_path=Path(raw_handoff["filename"]),
            source_filename=raw_handoff["filename"],
            sha256=scalar_identity["sha256"],
            size_bytes=scalar_identity["size_bytes"],
        )
    _validate_source_export_provenance(
        source_exports,
        profile_contract=profile_contract,
    )
    template_sha256 = _hdf5_text_attribute(
        handle.attrs.get("template_sha256", ""),
        label="template_sha256",
    )
    expected_template = {
        "sha256": template_sha256,
        "sha256_validation": "pass",
        "comsol_internal_contract": "runtime_validation_required",
    }
    reference_template = scientific.get("reference_template")
    if "template_relative_path" in handle.attrs or reference_template != {"sha256": template_sha256}:
        msg = "Canonical HDF5 template identity must contain bytes only."
        raise ValueError(msg)
    if not _is_sha256(template_sha256) or template != expected_template:
        msg = "Canonical HDF5 persisted template identity or runtime-validation provenance is invalid."
        raise ValueError(msg)
    return scientific, None, scalar_handoff, fixed_ownership


def _validate_hdf5_static_and_parameters(
    handle: h5py.File,
    profile: str,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None,
    scientific: Mapping[str, Any],
) -> tuple[h5py.Dataset, tuple[str, ...]]:
    """Validate configured grid, static fields, and parameter provenance."""
    _require_group_members(handle, "coords", {"x", "y"})
    _require_group_members(handle, "static", {"fields"})
    x_axis = _hdf5_dataset(handle, "coords/x")
    y_axis = _hdf5_dataset(handle, "coords/y")
    static = _hdf5_dataset(handle, "static/fields")
    if x_axis.dtype != np.float64 or y_axis.dtype != np.float64 or static.dtype != np.float32:
        msg = "Canonical coordinate/static dtypes are invalid."
        raise ValueError(msg)
    x_unit = _hdf5_text_attribute(x_axis.attrs.get("unit", ""), label="coords.x.unit")
    y_unit = _hdf5_text_attribute(y_axis.attrs.get("unit", ""), label="coords.y.unit")
    if x_unit != "m" or y_unit != "m":
        msg = "Canonical coordinate units must be explicit metres."
        raise ValueError(msg)
    grid = scientific["grid"]
    storage = scientific["storage"]
    fixed_values = scientific["scientific_fixed_values"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    expected_x = np.linspace(0.0, float(grid["Lx"]), nx, dtype=np.float64)
    expected_y = np.linspace(0.0, float(grid["Ly"]), ny, dtype=np.float64)
    x_values = np.asarray(x_axis, dtype=np.float64)
    y_values = np.asarray(y_axis, dtype=np.float64)
    if (
        x_values.shape != expected_x.shape
        or y_values.shape != expected_y.shape
        or not np.isfinite(x_values).all()
        or not np.isfinite(y_values).all()
        or not np.allclose(x_values, expected_x, rtol=0.0, atol=_COORDINATE_ATOL)
        or not np.allclose(y_values, expected_y, rtol=0.0, atol=_COORDINATE_ATOL)
    ):
        msg = "Canonical coordinate axes disagree with the embedded configured grid."
        raise ValueError(msg)
    static_names = profiles.static_field_names(profile)
    static_units = profiles.static_field_units(profile)
    if static.shape != (len(static_names), ny, nx) or not _dataset_is_finite(static):
        msg = "Canonical static field shape or finiteness is invalid."
        raise ValueError(msg)
    expected_static_chunks = (
        1,
        min(int(storage["chunk_y"]), ny),
        min(int(storage["chunk_x"]), nx),
    )
    if not _compression_matches(static, storage) or static.chunks != expected_static_chunks:
        msg = "Canonical static compression or chunking contract is invalid."
        raise ValueError(msg)
    _dataset_contract(static, names=static_names, units=static_units, label="static")
    scalar_names: tuple[str, ...]
    scalar_units: tuple[str, ...]
    if profile == profiles.STEADY_FLOW_PROFILE:
        _require_group_members(handle, "stationary_fixed", {"values"})
        scalar = _hdf5_dataset(handle, "stationary_fixed/values")
        scalar_names = profiles.STATIONARY_FIXED_FIELDS
        scalar_units = profiles.STATIONARY_FIXED_UNITS
        scalar_values = np.asarray(scalar, dtype=np.float64)
        ownership = json.loads(
            _hdf5_text_attribute(
                scalar.attrs.get("ownership", ""),
                label="stationary_fixed.ownership",
            )
        )
        runtime_source = _hdf5_text_attribute(
            scalar.attrs.get("runtime_source", ""),
            label="stationary_fixed.runtime_source",
        )
        expected_values = np.asarray(
            [fixed_values[name] for name in scalar_names],
            dtype=np.float64,
        )
        if (
            scalar_handoff is not None
            or scalar.dtype != np.float64
            or scalar.shape != (len(scalar_names),)
            or ownership != ["package_fixed"] * len(scalar_names)
            or runtime_source != "canonical_template"
            or not np.array_equal(scalar_values, expected_values)
        ):
            msg = "Canonical configured stationary values or provenance are invalid."
            raise ValueError(msg)
        _dataset_contract(
            scalar,
            names=scalar_names,
            units=scalar_units,
            label="stationary_fixed",
        )
        return scalar, scalar_names

    if scalar_handoff is None:
        message = "Canonical transient HDF5 is missing admitted scalar provenance."
        raise ValueError(message)
    _require_group_members(handle, "scalar", {"values"})
    scalar = _hdf5_dataset(handle, "scalar/values")
    scalar_names = scalar_handoff.field_names
    scalar_units = scalar_handoff.units
    scalar_values = np.asarray(scalar, dtype=np.float64)
    ownership = json.loads(
        _hdf5_text_attribute(
            scalar.attrs.get("ownership", ""),
            label="scalar.ownership",
        )
    )
    expected_values = np.asarray(scalar_handoff.values, dtype=np.float64)
    if (
        scalar.dtype != np.float64
        or scalar.shape != (len(profiles.TRANSIENT_SCALAR_INPUT_FIELDS),)
        or scalar_names != profiles.TRANSIENT_SCALAR_INPUT_FIELDS
        or scalar_units != profiles.TRANSIENT_SCALAR_INPUT_UNITS
        or ownership != list(scalar_handoff.ownership)
        or not np.isfinite(scalar_values).all()
        or not np.array_equal(scalar_values, expected_values)
    ):
        message = "Canonical transient scalar values or handoff provenance are invalid."
        raise ValueError(message)
    _dataset_contract(
        scalar,
        names=scalar_names,
        units=scalar_units,
        label="scalar",
    )
    return scalar, scalar_names


def _validate_hdf5_regular_trajectory(
    handle: h5py.File,
    scientific: Mapping[str, Any],
) -> np.ndarray:
    """Validate the configured regular-only axis and absolute trajectory."""
    _require_group_members(handle, "transient", {"fields"})
    time = _hdf5_dataset(handle, "time")
    transient = _hdf5_dataset(handle, "transient/fields")
    time_values = np.asarray(time, dtype=np.float64)
    time_contract = scientific["time"]
    grid = scientific["grid"]
    storage = scientific["storage"]
    configured_times = np.asarray(time_contract["regular_times"], dtype=np.float64)
    tolerance = time_classification_tolerance(time_contract)
    if (
        time.dtype != np.float64
        or transient.dtype != np.float32
        or time_values.ndim != 1
        or time_values.size < 1
        or time_values.size > configured_times.size
        or not np.array_equal(time_values, configured_times[: time_values.size])
        or _hdf5_text_attribute(time.attrs.get("unit", ""), label="time.unit") != "h"
        or float(time.attrs.get("classification_atol", -1.0)) != tolerance
        or _hdf5_text_attribute(time.attrs.get("classification_basis", ""), label="time.classification_basis")
        != _time_classification_basis(time_contract)
        or not _dataset_is_finite(transient)
    ):
        msg = "Canonical regular time axis, dtypes, or classification provenance is invalid."
        raise ValueError(msg)
    time_size = time.size
    if time_size is None:
        msg = "Canonical regular time dataset must have a finite element count."
        raise ValueError(msg)
    expected_shape = (
        time_size,
        len(profiles.TRANSIENT_FIELD_NAMES),
        int(grid["ny"]),
        int(grid["nx"]),
    )
    if transient.shape != expected_shape:
        msg = "Canonical regular transient field shape is invalid."
        raise ValueError(msg)
    _dataset_contract(
        transient,
        names=profiles.TRANSIENT_FIELD_NAMES,
        units=profiles.TRANSIENT_FIELD_UNITS,
        label="transient",
    )
    expected_chunks = (
        min(int(storage["chunk_time"]), time_size),
        1,
        min(int(storage["chunk_y"]), int(grid["ny"])),
        min(int(storage["chunk_x"]), int(grid["nx"])),
    )
    if not _compression_matches(transient, storage) or transient.chunks != expected_chunks:
        msg = "Canonical transient compression or chunking contract is invalid."
        raise ValueError(msg)
    return time_values


def _validate_hdf5_schedule(
    handle: h5py.File,
    scientific: Mapping[str, Any],
) -> None:
    """Validate the exact COMSOL boundary table without changing regular output time."""
    _require_group_members(handle, "schedule", {"values"})
    schedule = _hdf5_dataset(handle, "schedule/values")
    values = np.asarray(schedule, dtype=np.float64)
    storage = scientific["storage"]
    if (
        schedule.dtype != np.float64
        or values.ndim != _TABLE_RANK
        or values.shape[1] != len(profiles.SCHEDULE_FIELDS)
        or not np.isfinite(values).all()
        or not _compression_matches(schedule, storage)
    ):
        msg = "COMSOL boundary schedule dtype, shape, finiteness, or compression is invalid."
        raise ValueError(msg)
    _dataset_contract(
        schedule,
        names=profiles.SCHEDULE_FIELDS,
        units=profiles.SCHEDULE_UNITS,
        label="schedule",
    )
    p_ref = float(scientific["scientific_fixed_values"]["p_ref"])
    conversion = json.loads(
        _hdf5_text_attribute(
            schedule.attrs.get("conversion_pressure", ""),
            label="schedule.conversion_pressure",
        )
    )
    conversion_owner = _hdf5_text_attribute(
        schedule.attrs.get("humidity_conversion_owner", ""),
        label="schedule.humidity_conversion_owner",
    )
    case_scientific = _hdf5_json_dataset(handle, "provenance/case_scientific_provenance_json")
    metadata = case_scientific.get("schedule_diagnostics")
    sampled_values = case_scientific.get("sampled_values")
    boundary_handoff = json.loads(
        _hdf5_text_attribute(
            schedule.attrs.get("boundary_handoff", ""),
            label="schedule.boundary_handoff",
        )
    )
    handoff_ramp = boundary_handoff.get("startup_ramp") if isinstance(boundary_handoff, Mapping) else None
    if (
        conversion
        != {
            "name": "p_ref",
            "value": p_ref,
            "unit": "Pa",
            "owner": "package_fixed",
        }
        or conversion_owner != "generation_schedule"
        or not isinstance(metadata, Mapping)
        or not isinstance(sampled_values, Mapping)
        or not isinstance(handoff_ramp, Mapping)
        or boundary_handoff != metadata.get("boundary_handoff")
    ):
        msg = "COMSOL boundary schedule provenance is invalid."
        raise ValueError(msg)
    enabled = handoff_ramp.get("enabled")
    duration_h = handoff_ramp.get("duration_h")
    scientific_ramp = {"enabled": enabled}
    if enabled is True:
        scientific_ramp["duration_h"] = duration_h
    if scientific.get("boundary_schedule") != {"startup_ramp": scientific_ramp}:
        msg = "COMSOL boundary handoff disagrees with active scientific startup identity."
        raise ValueError(msg)
    schedule_service.validate_comsol_boundary_schedule(
        values,
        regular_times=np.asarray(scientific["time"]["regular_times"], dtype=np.float64),
        startup_ramp={"enabled": enabled, "duration_h": duration_h},
        initial_temperature=float(sampled_values["T_init"]),
        pressure=p_ref,
        metadata=metadata,
    )


def _complete_hdf5_time(
    handle: h5py.File,
    regular_time: np.ndarray,
    scientific: Mapping[str, Any],
) -> np.ndarray:
    """Validate and append one optional configured diagnostic exact-stop time."""
    if "exact_stop" not in handle:
        return regular_time
    exact_group = _require_group_members(handle, "exact_stop", {"time", "fields"})
    exact_time = _hdf5_dataset(handle, "exact_stop/time")
    exact_fields = _hdf5_dataset(handle, "exact_stop/fields")
    values = np.asarray(exact_time, dtype=np.float64)
    unit = _hdf5_text_attribute(exact_time.attrs.get("unit", ""), label="exact_stop.time.unit")
    time_contract = scientific["time"]
    grid = scientific["grid"]
    storage = scientific["storage"]
    tolerance = time_classification_tolerance(time_contract)
    configured_times = np.asarray(time_contract["regular_times"], dtype=np.float64)
    separated_from_regular = values.shape == (1,) and np.min(np.abs(configured_times - values[0])) > tolerance
    expected_shape = (
        len(profiles.TRANSIENT_FIELD_NAMES),
        int(grid["ny"]),
        int(grid["nx"]),
    )
    expected_chunks = (
        1,
        min(int(storage["chunk_y"]), int(grid["ny"])),
        min(int(storage["chunk_x"]), int(grid["nx"])),
    )
    if (
        exact_time.dtype != np.float64
        or values.shape != (1,)
        or values[0] <= regular_time[-1] + tolerance
        or values[0] > float(time_contract["stop"]) + tolerance
        or not separated_from_regular
        or exact_fields.dtype != np.float32
        or exact_fields.shape != expected_shape
        or not _compression_matches(exact_fields, storage)
        or exact_fields.chunks != expected_chunks
        or not _dataset_is_finite(exact_fields)
        or unit != "h"
        or exact_group.attrs.get("usage") != "diagnostic_only_no_training_transition"
    ):
        msg = "Canonical exact-stop diagnostic contract is invalid."
        raise ValueError(msg)
    _dataset_contract(
        exact_fields,
        names=profiles.TRANSIENT_FIELD_NAMES,
        units=profiles.TRANSIENT_FIELD_UNITS,
        label="exact_stop.fields",
    )
    return np.concatenate((regular_time, values))


def _validate_hdf5_diagnostics(
    handle: h5py.File,
    complete_time: np.ndarray,
    scientific: Mapping[str, Any],
) -> None:
    """Validate complete global diagnostics and actual Final Status."""
    _require_group_members(handle, "global", {"values"})
    _require_group_members(handle, "final_status", {"values"})
    global_values = _hdf5_dataset(handle, "global/values")
    final_status = _hdf5_dataset(handle, "final_status/values")
    global_array = np.asarray(global_values, dtype=np.float64)
    final_array = np.asarray(final_status, dtype=np.float64)
    storage = scientific["storage"]
    tolerance = time_classification_tolerance(scientific["time"])
    if (
        global_values.dtype != np.float64
        or global_array.shape != (complete_time.size, len(profiles.GLOBAL_FIELD_NAMES))
        or not np.isfinite(global_array).all()
        or not _compression_matches(global_values, storage)
        or not np.allclose(global_array[:, 0], complete_time, rtol=0.0, atol=tolerance)
    ):
        msg = "Canonical complete global diagnostic values or time semantics are invalid."
        raise ValueError(msg)
    _dataset_contract(
        global_values,
        names=profiles.GLOBAL_FIELD_NAMES,
        units=profiles.GLOBAL_FIELD_UNITS,
        label="global",
    )
    if (
        final_status.dtype != np.float64
        or final_array.shape != (len(profiles.FINAL_STATUS_FIELDS),)
        or not np.isfinite(final_array).all()
        or abs(final_array[0] - complete_time[-1]) > tolerance
    ):
        msg = "Canonical final-status values or actual final time are invalid."
        raise ValueError(msg)
    _dataset_contract(
        final_status,
        names=profiles.FINAL_STATUS_FIELDS,
        units=profiles.FINAL_STATUS_UNITS,
        label="final_status",
    )


def _validate_hdf5_transient(
    handle: h5py.File,
    scientific: Mapping[str, Any],
) -> None:
    """Validate all transient-only schema-v1 members."""
    regular_time = _validate_hdf5_regular_trajectory(handle, scientific)
    _validate_hdf5_schedule(handle, scientific)
    complete_time = _complete_hdf5_time(handle, regular_time, scientific)
    _validate_hdf5_diagnostics(handle, complete_time, scientific)


def validate_case_hdf5(path: Path, *, expected_profile: str | None = None) -> dict[str, Any]:
    """Validate exact schema-v1 layout, provenance, units, shapes, and identities."""
    if not path.is_file() or path.is_symlink():
        msg = f"Canonical case HDF5 is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    with h5py.File(path, "r") as handle:
        if (
            handle.attrs.get("schema_kind") != HDF5_SCHEMA_KIND
            or int(handle.attrs.get("schema_version", -1)) != HDF5_SCHEMA_VERSION
            or handle.attrs.get("converter_version") != HDF5_CONVERTER_VERSION
        ):
            msg = f"Unsupported canonical case HDF5 schema or converter identity: {path}"
            raise ValueError(msg)
        profile = _hdf5_text_attribute(handle.attrs.get("simulation_profile", ""), label="simulation_profile")
        if expected_profile is not None and profile != expected_profile:
            msg = f"Canonical case profile mismatch: expected {expected_profile!r}, got {profile!r}."
            raise ValueError(msg)
        profile_contract = profiles.resolve_profile(profile)
        _validate_hdf5_members(handle, profile)
        (
            scientific,
            template_relative_path,
            scalar_handoff,
            _fixed_ownership,
        ) = _validate_hdf5_provenance(
            handle,
            profile,
            profile_contract,
        )
        _validate_hdf5_static_and_parameters(
            handle,
            profile,
            scalar_handoff,
            scientific,
        )
        if profile == profiles.TRANSIENT_DRYING_PROFILE:
            _validate_hdf5_transient(handle, scientific)
        identities = {
            name: _hdf5_text_attribute(handle.attrs[name], label=name)
            for name in ("case_input_id", "simulation_case_id", "scientific_config_digest", "template_sha256")
        }
        if any(not _is_sha256(value) for value in identities.values()):
            msg = "Canonical HDF5 identity attributes are malformed."
            raise ValueError(msg)
        export_contract_sha256 = _hdf5_text_attribute(
            handle.attrs.get("export_contract_sha256", ""),
            label="export_contract_sha256",
        )
        available_views = json.loads(_hdf5_text_attribute(handle.attrs.get("available_learning_views", ""), label="available_learning_views"))
        airflow_source = _hdf5_text_attribute(handle.attrs.get("airflow_source", ""), label="airflow_source")
        if (
            not _is_sha256(export_contract_sha256)
            or available_views != list(profile_contract.available_learning_views)
            or airflow_source != profile_contract.airflow_source
        ):
            msg = "Canonical HDF5 export, learning-view, or airflow provenance is invalid."
            raise ValueError(msg)
        if config_contract.compute_scientific_config_digest(scientific) != identities["scientific_config_digest"]:
            msg = "Canonical HDF5 scientific provenance digest is inconsistent."
            raise ValueError(msg)
        if "git_commit" in handle.attrs:
            message = "Canonical HDF5 scientific content must not embed execution Git provenance."
            raise ValueError(message)
        return {
            "simulation_profile": profile,
            "git_commit": None,
            "template_relative_path": template_relative_path,
            "export_contract_sha256": export_contract_sha256,
            "available_learning_views": tuple(available_views),
            "airflow_source": airflow_source,
            **identities,
        }


@dataclass(frozen=True, slots=True)
class _DiagnosticExport:
    """Minimal raw-export descriptor accepted by production parsers."""

    source_path: Path
    role: str


@dataclass(frozen=True, slots=True)
class ReconstructedTransientInitialState:
    """Production-parsed transient initial state and source-byte evidence."""

    stationary_fields: np.ndarray
    transient_states: np.ndarray
    time: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission
    source_artifacts: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReconstructedTransientBulkMoisture:
    """Production-parsed bulk-moisture comparison and source evidence."""

    consistency: TransientBulkMoistureConsistency
    source_artifacts: dict[str, dict[str, Any]]


def reconstruct_transient_initial_state(
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    *,
    stationary_export: Path | str,
    transient_export: Path | str,
    work_directory: Path | str,
) -> ReconstructedTransientInitialState:
    """
    Reconstruct the production canonical initial state from two raw transient exports.

    Parameters
    ----------
    config : GenerationConfig
        Resolved transient generation configuration.
    case_payload : Mapping[str, Any]
        Prepared case provenance and scalar-handoff contract.
    stationary_export : Path | str
        Configured stationary-fields Spreadsheet export.
    transient_export : Path | str
        Configured wide transient-states Spreadsheet export.
    work_directory : Path | str
        Case-local input workspace containing the admitted scalar handoff.

    Returns
    -------
    ReconstructedTransientInitialState
        Canonically ordered stationary fields, regular transient states, grid,
        time axis, scalar handoff, and source identities.

    Raises
    ------
    ValueError
        If the supplied paths cannot satisfy the exact production parser contract.

    Notes
    -----
    This bridge deliberately stops before output validation and HDF5 publication,
    allowing conversion-failure diagnostics to inspect the exact arrays that the
    production validator would otherwise reject.

    """
    if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Initial-state reconstruction requires the transient_drying profile."
        raise ValueError(message)
    stationary_path = Path(stationary_export)
    transient_path = Path(transient_export)
    for path, role in ((stationary_path, profiles.STEADY_FLOW_EXPORT_ROLE), (transient_path, profiles.TRANSIENT_RAW_EXPORT_ROLE)):
        if not path.is_file() or path.is_symlink():
            message = f"Initial-state reconstruction requires one safe {role!r} export: {path}"
            raise FileNotFoundError(message)
    exports = (
        _DiagnosticExport(stationary_path, profiles.STEADY_FLOW_EXPORT_ROLE),
        _DiagnosticExport(transient_path, profiles.TRANSIENT_RAW_EXPORT_ROLE),
    )
    x_axis, y_axis, stationary_fields = _static_fields(config, exports)
    regular_time, transient_states, exact_stop_time, _exact_stop_fields = _transient_fields(
        config,
        exports,
        x_axis=x_axis,
        y_axis=y_axis,
    )
    exported_time = regular_time if exact_stop_time is None else np.concatenate((regular_time, np.asarray((exact_stop_time,), dtype=np.float64)))
    work_root = Path(work_directory).resolve()
    source_paths = {
        "stationary_fields": stationary_path.resolve(),
        "transient_states": transient_path.resolve(),
    }
    if any(not source.is_relative_to(work_root) for source in source_paths.values()):
        message = "Initial-state diagnostic exports must remain inside the case work directory."
        raise ValueError(message)
    scalar_handoff = scalar_handoff_contract.admit_case_scalar_handoff(case_payload, work_directory)
    return ReconstructedTransientInitialState(
        stationary_fields=stationary_fields,
        transient_states=transient_states,
        time=exported_time,
        x_axis=x_axis,
        y_axis=y_axis,
        scalar_handoff=scalar_handoff,
        source_artifacts={
            name: {
                "relative_path": source.relative_to(work_root).as_posix(),
                "sha256": common.serialization.file_sha256(source),
                "size_bytes": source.stat().st_size,
            }
            for name, source in source_paths.items()
        },
    )


def reconstruct_transient_bulk_moisture(
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    *,
    stationary_export: Path | str,
    transient_export: Path | str,
    global_export: Path | str,
    work_directory: Path | str,
) -> ReconstructedTransientBulkMoisture:
    """Reconstruct bulk moisture through the canonical production parsers."""
    if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Bulk-moisture reconstruction requires transient_drying."
        raise ValueError(message)
    paths = {
        "stationary_fields": Path(stationary_export),
        "transient_states": Path(transient_export),
        "global_time_series": Path(global_export),
    }
    roles = {
        "stationary_fields": profiles.STEADY_FLOW_EXPORT_ROLE,
        "transient_states": profiles.TRANSIENT_RAW_EXPORT_ROLE,
        "global_time_series": profiles.GLOBAL_EXPORT_ROLE,
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            message = f"Bulk-moisture reconstruction requires one safe {roles[name]!r} export: {path}"
            raise FileNotFoundError(message)
    exports = tuple(_DiagnosticExport(paths[name], roles[name]) for name in paths)
    x_axis, y_axis, stationary_fields = _static_fields(config, exports)
    regular_time, regular_fields, exact_stop_time, exact_stop_fields = _transient_fields(
        config,
        exports,
        x_axis=x_axis,
        y_axis=y_axis,
    )
    global_values = _ordered_values(
        config,
        exports,
        profiles.GLOBAL_EXPORT_ROLE,
        profiles.GLOBAL_FIELD_NAMES,
    )
    scalar_handoff = scalar_handoff_contract.admit_case_scalar_handoff(
        case_payload,
        work_directory,
    )
    scalar_names = profiles.scalar_input_fields(config.profile.id)
    f_surf = float(scalar_handoff.values[scalar_names.index("f_surf")])
    state_time = _combined_state_time(
        regular_time,
        exact_stop_time,
        exact_stop_fields,
    )
    consistency = evaluate_transient_bulk_moisture_consistency(
        config,
        stationary_fields,
        state_time,
        regular_fields,
        exact_stop_fields,
        global_values,
        f_surf=f_surf,
        time_tolerance=time_classification_tolerance(config.scientific_values["time"]),
    )
    work_root = Path(work_directory).resolve()
    resolved = {name: path.resolve() for name, path in paths.items()}
    if any(not source.is_relative_to(work_root) for source in resolved.values()):
        message = "Bulk-moisture diagnostic exports must remain inside the case work directory."
        raise ValueError(message)
    return ReconstructedTransientBulkMoisture(
        consistency=consistency,
        source_artifacts={
            name: {
                "relative_path": source.relative_to(work_root).as_posix(),
                "sha256": common.serialization.file_sha256(source),
                "size_bytes": source.stat().st_size,
            }
            for name, source in resolved.items()
        },
    )


def convert_exports_to_hdf5(
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    exports: Sequence[Any],
    *,
    work_directory: Path,
    runtime_directory: Path,
    runtime_seconds: float,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None = None,
) -> CanonicalCase:
    """Validate all adapters and atomically create the sole canonical payload."""
    x_axis, y_axis, static_fields = _static_fields(config, exports)
    stationary_values = _stationary_fixed_values(
        case_payload,
        config.scientific_values["scientific_fixed_values"],
    )
    if config.profile.id == profiles.STEADY_FLOW_PROFILE:
        if scalar_handoff is not None:
            message = "Steady HDF5 publication cannot receive a transient scalar handoff."
            raise ValueError(message)
        scalar_values = stationary_values
    else:
        admitted = scalar_handoff_contract.admit_case_scalar_handoff(
            case_payload,
            work_directory,
        )
        if scalar_handoff is not None and scalar_handoff != admitted:
            message = "Runtime and publication scalar admissions disagree."
            raise ValueError(message)
        scalar_handoff = admitted
        scalar_values = np.asarray(admitted.values, dtype=np.float64)
    transient_time: np.ndarray | None = None
    transient_fields: np.ndarray | None = None
    exact_stop_time: float | None = None
    exact_stop_fields: np.ndarray | None = None
    schedule_values: np.ndarray | None = None
    global_values: np.ndarray | None = None
    final_status: np.ndarray | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        transient_time, transient_fields, exact_stop_time, exact_stop_fields = _transient_fields(
            config,
            exports,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        scalar_names = profiles.scalar_input_fields(config.profile.id)
        p_ref = float(config.scientific_values["scientific_fixed_values"]["p_ref"])
        f_surf = float(scalar_values[scalar_names.index("f_surf")])
        schedule_values = _schedule_values(
            case_payload,
            work_directory,
            p_ref=p_ref,
            time_contract=config.scientific_values["time"],
            startup_ramp=config.scientific_values["boundary_schedule"]["startup_ramp"],
            initial_temperature=float(case_payload["sampled_values"]["T_init"]),
        )
        global_values = _ordered_values(config, exports, profiles.GLOBAL_EXPORT_ROLE, profiles.GLOBAL_FIELD_NAMES)
        if np.any(np.diff(global_values[:, 0]) <= 0):
            msg = "Global diagnostic time must be strictly increasing."
            raise ValueError(msg)
        final_status = _ordered_values(config, exports, profiles.FINAL_STATUS_EXPORT_ROLE, profiles.FINAL_STATUS_FIELDS)
        _validate_transient_outputs(
            config,
            static_fields,
            transient_time,
            transient_fields,
            exact_stop_time,
            exact_stop_fields,
            global_values,
            final_status,
            f_surf=f_surf,
        )
    source_hashes = _source_hashes(exports)
    runtime_directory.mkdir(parents=True, exist_ok=True)
    destination = runtime_directory / "case.h5"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".case.", suffix=".h5.tmp", dir=runtime_directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_hdf5(
            temporary,
            config,
            case_payload,
            x_axis=x_axis,
            y_axis=y_axis,
            static_fields=static_fields,
            scalar_values=scalar_values,
            scalar_handoff=scalar_handoff,
            transient_time=transient_time,
            transient_fields=transient_fields,
            exact_stop_time=exact_stop_time,
            exact_stop_fields=exact_stop_fields,
            schedule_values=schedule_values,
            global_values=global_values,
            final_status=final_status,
            source_hashes=source_hashes,
        )
        validate_case_hdf5(temporary, expected_profile=config.profile.id)
        temporary.replace(destination)
        validate_case_hdf5(destination, expected_profile=config.profile.id)
    finally:
        temporary.unlink(missing_ok=True)
    status = _status(
        config,
        transient_time=transient_time,
        exact_stop_time=exact_stop_time,
        final_status=final_status,
        global_values=global_values,
        runtime_seconds=runtime_seconds,
    )
    status_path = common.serialization.atomic_write_json(runtime_directory / "status.json", status)
    return CanonicalCase(
        path=destination,
        status_path=status_path,
        status=status,
        source_export_hashes=source_hashes,
    )
