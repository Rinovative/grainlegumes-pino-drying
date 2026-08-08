"""
===============================================================================
generation_storage.py
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

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from src import common, domain

from . import generation_profiles as profiles
from . import generation_source as source_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .generation_config import GenerationConfig

HDF5_SCHEMA_KIND = "vp2_canonical_case"
HDF5_SCHEMA_VERSION = 2
_MINIMUM_TABLE_ROWS = 2
_MINIMUM_AXIS_POINTS = 2
_TABLE_RANK = 2
_COORDINATE_ATOL = 1e-12
_STATIONARITY_RTOL = profiles.STATIONARITY_TOLERANCE
_TIME_HORIZON = 168.0
_HDF5_COMPRESSION_LEVEL = 4
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class CanonicalCase:
    """One validated HDF5 payload and derived solver status."""

    path: Path
    status_path: Path
    status: dict[str, Any]
    source_export_hashes: dict[str, dict[str, Any]]


def _read_table(path: Path, *, delimiter: str) -> tuple[list[str], np.ndarray]:
    """Read one finite rectangular numeric table with a unique header."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith(("%", "#"))]
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Configured COMSOL export is not readable text: {path}"
        raise ValueError(msg) from error
    rows = list(csv.reader(lines, delimiter=delimiter))
    if len(rows) < _MINIMUM_TABLE_ROWS:
        msg = f"Numeric export must contain a header and data: {path}"
        raise ValueError(msg)
    header = [item.strip() for item in rows[0]]
    if not header or len(header) != len(set(header)) or any(len(row) != len(header) for row in rows[1:]):
        msg = f"Numeric export has duplicate headers or inconsistent row widths: {path}"
        raise ValueError(msg)
    try:
        values = np.asarray([[float(item.strip()) for item in row] for row in rows[1:]], dtype=np.float64)
    except ValueError as error:
        msg = f"Numeric export contains malformed values: {path}"
        raise ValueError(msg) from error
    if not np.isfinite(values).all():
        msg = f"Numeric export contains NaN or infinity: {path}"
        raise ValueError(msg)
    return header, values


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


def _mapped_table(paths: Sequence[Path], contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Read and concatenate explicitly mapped logical fields."""
    if not paths:
        msg = f"Required export role {contract['role']!r} produced no files."
        raise FileNotFoundError(msg)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in contract["columns"]}
    time_column = contract.get("time_column")
    if time_column is not None:
        collected["__stationary_time__"] = []
    for path in paths:
        header, values = _read_table(path, delimiter=contract["delimiter"])
        missing = [source for source in contract["columns"].values() if source not in header]
        if time_column is not None and time_column not in header:
            missing.append(time_column)
        if missing:
            msg = f"Export {path} is missing explicitly configured headers {missing}."
            raise ValueError(msg)
        for logical, source in contract["columns"].items():
            collected[logical].append(values[:, header.index(source)])
        if time_column is not None:
            collected["__stationary_time__"].append(values[:, header.index(time_column)])
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
    actual_x = np.unique(x_values)
    actual_y = np.unique(y_values)
    if (
        actual_x.size != x_axis.size
        or actual_y.size != y_axis.size
        or not np.allclose(actual_x, x_axis, rtol=0.0, atol=_COORDINATE_ATOL)
        or not np.allclose(actual_y, y_axis, rtol=0.0, atol=_COORDINATE_ATOL)
    ):
        msg = "Static export coordinates do not match the authoritative 401x251 grid."
        raise ValueError(msg)
    coordinate_rows: dict[tuple[float, float], list[int]] = {}
    for row, coordinate in enumerate(zip(x_values, y_values, strict=True)):
        coordinate_rows.setdefault((float(coordinate[0]), float(coordinate[1])), []).append(row)
    if len(coordinate_rows) != x_axis.size * y_axis.size:
        msg = "Static export does not contain one complete Cartesian grid."
        raise ValueError(msg)
    repeated_allowed = contract.get("time_column") is not None
    arrays: np.ndarray = np.empty((len(profiles.STATIC_FIELD_NAMES), y_axis.size, x_axis.size), dtype=np.float64)
    x_lookup = {float(value): index for index, value in enumerate(actual_x)}
    y_lookup = {float(value): index for index, value in enumerate(actual_y)}
    for coordinate, rows in coordinate_rows.items():
        if len(rows) != 1 and not repeated_allowed:
            msg = f"Static export repeats coordinate {coordinate} without configured time ownership."
            raise ValueError(msg)
        y_index = y_lookup[coordinate[1]]
        x_index = x_lookup[coordinate[0]]
        for field_index, name in enumerate(profiles.STATIC_FIELD_NAMES):
            candidates = mapped[name][rows]
            if not np.allclose(candidates, candidates[0], rtol=_STATIONARITY_RTOL, atol=_STATIONARITY_RTOL):
                msg = f"Supposedly stationary field {name!r} varies beyond tolerance at {coordinate}."
                raise ValueError(msg)
            arrays[field_index, y_index, x_index] = candidates[0]
    determinant = arrays[0] * arrays[2] - arrays[1] ** 2
    if np.any(arrays[0] <= 0) or np.any(arrays[2] <= 0) or np.any(determinant <= 0):
        msg = "Canonical static permeability tensor is not positive definite."
        raise ValueError(msg)
    porosity = arrays[profiles.STATIC_FIELD_NAMES.index("eps_bed")]
    if np.any((porosity <= 0) | (porosity >= 1)):
        msg = "Canonical static porosity must lie strictly inside (0, 1)."
        raise ValueError(msg)
    return x_axis, y_axis, arrays


def _transient_fields(
    config: GenerationConfig,
    exports: Sequence[Any],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize regular transient states onto time-channel-y-x arrays."""
    contract = _contract(config, profiles.TRANSIENT_RAW_EXPORT_ROLE)
    mapped = _mapped_table(_role_paths(exports, profiles.TRANSIENT_RAW_EXPORT_ROLE), contract)
    times = np.unique(mapped["t"])
    if times.size < 1 or times[0] != 0.0 or np.any(np.diff(times) != 1.0) or times[-1] > _TIME_HORIZON:
        msg = "Transient export times must be a contiguous regular hourly prefix of 0..168 h."
        raise ValueError(msg)
    actual_x = np.unique(mapped["x"])
    actual_y = np.unique(mapped["y"])
    expected_rows = times.size * x_axis.size * y_axis.size
    if (
        mapped["x"].size != expected_rows
        or actual_x.size != x_axis.size
        or actual_y.size != y_axis.size
        or not np.allclose(actual_x, x_axis, rtol=0.0, atol=_COORDINATE_ATOL)
        or not np.allclose(actual_y, y_axis, rtol=0.0, atol=_COORDINATE_ATOL)
    ):
        msg = "Transient export does not contain one complete authoritative Cartesian state per regular time."
        raise ValueError(msg)
    arrays = np.full(
        (times.size, len(profiles.TRANSIENT_FIELD_NAMES), y_axis.size, x_axis.size),
        np.nan,
        dtype=np.float64,
    )
    time_lookup = {float(value): index for index, value in enumerate(times)}
    x_lookup = {float(value): index for index, value in enumerate(actual_x)}
    y_lookup = {float(value): index for index, value in enumerate(actual_y)}
    occupied: set[tuple[int, int, int]] = set()
    for row, (time_value, x_value, y_value) in enumerate(zip(mapped["t"], mapped["x"], mapped["y"], strict=True)):
        try:
            coordinate = time_lookup[float(time_value)], y_lookup[float(y_value)], x_lookup[float(x_value)]
        except KeyError as error:
            msg = "Transient export contains a coordinate outside the authoritative grid."
            raise ValueError(msg) from error
        if coordinate in occupied:
            msg = f"Transient export repeats state coordinate {coordinate}."
            raise ValueError(msg)
        occupied.add(coordinate)
        for field_index, name in enumerate(profiles.TRANSIENT_FIELD_NAMES):
            arrays[coordinate[0], field_index, coordinate[1], coordinate[2]] = mapped[name][row]
    if not np.isfinite(arrays).all():
        msg = "Transient export contains missing or non-finite canonical fields."
        raise ValueError(msg)
    return times, arrays


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


def _validate_global_bulk_moisture(
    static_fields: np.ndarray,
    transient_time: np.ndarray,
    transient_fields: np.ndarray,
    global_values: np.ndarray,
) -> None:
    """Validate exported X_wb_bulk against integrated dry and water mass."""
    rho_bu_dry = static_fields[profiles.STATIC_FIELD_NAMES.index("rho_bu_dry")]
    water = domain.moisture.granular_water_content(
        transient_fields[:, profiles.TRANSIENT_FIELD_NAMES.index("w_surf")],
        transient_fields[:, profiles.TRANSIENT_FIELD_NAMES.index("w_int")],
    )
    weights = np.ones_like(rho_bu_dry, dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    expected = np.asarray(
        [domain.moisture.bulk_wet_basis_moisture(state, rho_bu_dry, cell_weights=weights) for state in water],
        dtype=np.float64,
    )
    global_time = global_values[:, profiles.GLOBAL_FIELD_NAMES.index("t")]
    positions = [np.flatnonzero(np.isclose(global_time, time, rtol=0.0, atol=1e-12)) for time in transient_time]
    if any(position.size != 1 for position in positions):
        message = "Global diagnostics must contain one X_wb_bulk value for every regular transient state."
        raise ValueError(message)
    column = global_values[[int(position[0]) for position in positions], profiles.GLOBAL_FIELD_NAMES.index("X_wb_bulk")]
    if column.shape != expected.shape or not np.allclose(column, expected, rtol=1e-6, atol=1e-9):
        maximum = math.inf if column.shape != expected.shape else float(np.max(np.abs(column - expected)))
        message = f"Exported X_wb_bulk disagrees with integrated dry and water mass; maximum error={maximum}."
        raise ValueError(message)


def _schedule_values(case_payload: Mapping[str, Any], work_directory: Path) -> np.ndarray:
    """Read and revalidate the generated schedule adapter bound by case.json."""
    spec = case_payload["input_contract"]["schedule"]
    path = work_directory / spec["filename"]
    identity = case_payload["input_files"][path.name]
    if common.serialization.file_sha256(path) != identity["sha256"] or path.stat().st_size != identity["size_bytes"]:
        msg = "Schedule adapter bytes changed after case-input identity was computed."
        raise RuntimeError(msg)
    header, values = _read_table(path, delimiter=spec["delimiter"])
    if header != list(profiles.SCHEDULE_FIELDS) or values.shape != (169, len(profiles.SCHEDULE_FIELDS)):
        msg = "Schedule adapter does not match the exact regular 0..168-hour contract."
        raise ValueError(msg)
    if not np.array_equal(values[:, 0], np.arange(169, dtype=np.float64)):
        msg = "Schedule adapter time must be the exact hourly 0..168 sequence."
        raise ValueError(msg)
    return values


def validate_float32_conversion(values: np.ndarray, *, rtol: float, atol: float, label: str) -> np.ndarray:
    """Convert large fields only after explicit finite allclose validation."""
    source = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source).all():
        msg = f"{label} contains non-finite values before float32 conversion."
        raise ValueError(msg)
    converted = source.astype(np.float32)
    restored = converted.astype(np.float64)
    if not np.allclose(source, restored, rtol=rtol, atol=atol):
        maximum_error = float(np.max(np.abs(source - restored)))
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
    case_payload: Mapping[str, Any],
    *,
    transient_time: np.ndarray | None,
    final_status: np.ndarray | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Derive the maintained Python status schema without guessing mass-flow signs."""
    if transient_time is None or final_status is None:
        t_final = None
        wet_final = None
        target_reached = None
        hit_t_max = None
        regular_states = 0
        last_regular = None
    else:
        values = dict(zip(profiles.FINAL_STATUS_FIELDS, final_status[0], strict=True))
        t_final = float(values["t_final"])
        wet_final = float(values["f_wet_dm_final"])
        limit = float(case_payload["sampled_values"]["f_wet_dm_max"])
        target_reached = wet_final <= limit
        hit_t_max = t_final >= float(config.scientific_values["time"]["stop"])
        regular_states = int(transient_time.size)
        last_regular = float(transient_time[-1])
    return {
        "schema_kind": "simulation_case_status",
        "schema_version": 1,
        "solver_success": True,
        "target_reached": target_reached,
        "hit_t_max": hit_t_max,
        "t_stop_exact": t_final,
        "t_last_regular": last_regular,
        "n_regular_states": regular_states,
        "n_regular_steps": max(regular_states - 1, 0),
        "f_wet_dm_final": wet_final,
        "runtime_s": float(runtime_seconds),
        "units": {"t_stop_exact": "h", "t_last_regular": "h", "f_wet_dm_final": "1", "runtime_s": "s"},
        "mass_balance_error": None,
        "contains_nan_or_inf": False,
        "field_shape_valid": True,
        "schedule_valid": True,
    }


def _write_hdf5(
    path: Path,
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    static_fields: np.ndarray,
    transient_time: np.ndarray | None,
    transient_fields: np.ndarray | None,
    schedule_values: np.ndarray | None,
    global_values: np.ndarray | None,
    source_hashes: Mapping[str, Any],
) -> None:
    """Write one complete case payload to a private temporary path."""
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
            label="transient fields",
        )
    compression = {
        "compression": storage["compression"],
        "compression_opts": int(storage["compression_level"]),
        "shuffle": bool(storage["shuffle"]),
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_kind"] = HDF5_SCHEMA_KIND
        handle.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        handle.attrs["converter_version"] = storage["converter_version"]
        for key in (
            "simulation_profile",
            "material_family",
            "sampling_regime",
            "case_input_id",
            "simulation_case_id",
            "scientific_config_digest",
            "export_contract_sha256",
            "airflow_source",
            "git_commit",
        ):
            handle.attrs[key] = case_payload[key]
        handle.attrs["template_sha256"] = case_payload["template"]["sha256"]
        handle.attrs["available_learning_views"] = _json_attribute(case_payload["available_learning_views"])
        handle.attrs["source_export_hashes"] = _json_attribute(source_hashes)
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
        static_dataset.attrs["field_names"] = _json_attribute(list(profiles.STATIC_FIELD_NAMES))
        static_dataset.attrs["units"] = _json_attribute(list(profiles.STATIC_FIELD_UNITS))
        scalar_entries = case_payload["scalars"]
        if (
            not isinstance(scalar_entries, list)
            or [entry.get("name") for entry in scalar_entries] != list(profiles.SCALAR_INPUT_FIELDS)
            or [entry.get("unit") for entry in scalar_entries] != list(profiles.SCALAR_INPUT_UNITS)
        ):
            msg = "Case scalar provenance does not match the canonical scalar adapter contract."
            raise ValueError(msg)
        scalar_group = handle.create_group("scalar")
        scalar_dataset = scalar_group.create_dataset(
            "values",
            data=np.asarray([entry["value"] for entry in scalar_entries], dtype=np.float64),
        )
        scalar_dataset.attrs["field_names"] = _json_attribute(list(profiles.SCALAR_INPUT_FIELDS))
        scalar_dataset.attrs["units"] = _json_attribute([entry["unit"] for entry in scalar_entries])
        if transient_float32 is not None and transient_time is not None and schedule_values is not None and global_values is not None:
            time_dataset = handle.create_dataset("time", data=np.asarray(transient_time, dtype=np.float64))
            time_dataset.attrs["unit"] = "h"
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
            schedule = handle.create_group("schedule")
            schedule_dataset = schedule.create_dataset("values", data=np.asarray(schedule_values, dtype=np.float64), **compression)
            schedule_dataset.attrs["field_names"] = _json_attribute(list(profiles.SCHEDULE_FIELDS))
            schedule_dataset.attrs["units"] = _json_attribute(list(profiles.SCHEDULE_UNITS))
            global_group = handle.create_group("global")
            global_dataset = global_group.create_dataset("values", data=np.asarray(global_values, dtype=np.float64), **compression)
            global_dataset.attrs["field_names"] = _json_attribute(list(profiles.GLOBAL_FIELD_NAMES))
            global_dataset.attrs["units"] = _json_attribute(list(profiles.GLOBAL_FIELD_UNITS))
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


def validate_case_hdf5(path: Path, *, expected_profile: str | None = None) -> dict[str, Any]:
    """Validate canonical case layout, dtypes, compression, chunks, and identities."""
    if not path.is_file() or path.is_symlink():
        msg = f"Canonical case HDF5 is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_kind") != HDF5_SCHEMA_KIND or int(handle.attrs.get("schema_version", -1)) != HDF5_SCHEMA_VERSION:
            msg = f"Unsupported canonical case HDF5 schema: {path}"
            raise ValueError(msg)
        profile = _hdf5_text_attribute(handle.attrs.get("simulation_profile", ""), label="simulation_profile")
        if expected_profile is not None and profile != expected_profile:
            msg = f"Canonical case profile mismatch: expected {expected_profile!r}, got {profile!r}."
            raise ValueError(msg)
        expected_keys = {"coords", "static", "scalar"}
        if profile == profiles.TRANSIENT_DRYING_PROFILE:
            expected_keys |= {"time", "transient", "schedule", "global"}
        if set(handle) != expected_keys:
            msg = f"Canonical HDF5 group membership mismatch for profile {profile!r}: {sorted(handle)}"
            raise ValueError(msg)
        x_axis = _hdf5_dataset(handle, "coords/x")
        y_axis = _hdf5_dataset(handle, "coords/y")
        static = _hdf5_dataset(handle, "static/fields")
        if x_axis.dtype != np.float64 or y_axis.dtype != np.float64 or static.dtype != np.float32:
            msg = "Canonical coordinate/static dtypes are invalid."
            raise ValueError(msg)
        if (
            _hdf5_text_attribute(x_axis.attrs.get("unit", ""), label="coords.x.unit") != "m"
            or _hdf5_text_attribute(y_axis.attrs.get("unit", ""), label="coords.y.unit") != "m"
        ):
            msg = "Canonical coordinate units must be explicit metres."
            raise ValueError(msg)
        x_values = np.asarray(x_axis, dtype=np.float64)
        y_values = np.asarray(y_axis, dtype=np.float64)
        if (
            x_values.ndim != 1
            or y_values.ndim != 1
            or x_values.size < _MINIMUM_AXIS_POINTS
            or y_values.size < _MINIMUM_AXIS_POINTS
            or not np.isfinite(x_values).all()
            or not np.isfinite(y_values).all()
            or np.any(np.diff(x_values) <= 0.0)
            or np.any(np.diff(y_values) <= 0.0)
        ):
            msg = "Canonical coordinate axes must be finite and strictly increasing."
            raise ValueError(msg)
        if static.shape != (len(profiles.STATIC_FIELD_NAMES), y_axis.size, x_axis.size):
            msg = "Canonical static field shape is invalid."
            raise ValueError(msg)
        if static.compression != "gzip" or static.compression_opts != _HDF5_COMPRESSION_LEVEL or not static.shuffle or static.chunks is None:
            msg = "Canonical static compression or chunking contract is invalid."
            raise ValueError(msg)
        static_names = json.loads(_hdf5_text_attribute(static.attrs["field_names"], label="static.field_names"))
        static_units = json.loads(_hdf5_text_attribute(static.attrs["units"], label="static.units"))
        if static_names != list(profiles.STATIC_FIELD_NAMES) or static_units != list(profiles.STATIC_FIELD_UNITS):
            msg = "Canonical static field-name or unit metadata is invalid."
            raise ValueError(msg)
        scalar = _hdf5_dataset(handle, "scalar/values")
        scalar_names = json.loads(_hdf5_text_attribute(scalar.attrs["field_names"], label="scalar.field_names"))
        scalar_units = json.loads(_hdf5_text_attribute(scalar.attrs["units"], label="scalar.units"))
        if (
            scalar.dtype != np.float64
            or scalar.shape != (len(profiles.SCALAR_INPUT_FIELDS),)
            or scalar_names != list(profiles.SCALAR_INPUT_FIELDS)
            or scalar_units != list(profiles.SCALAR_INPUT_UNITS)
            or not np.isfinite(np.asarray(scalar, dtype=np.float64)).all()
        ):
            msg = "Canonical scalar values, names, or units are invalid."
            raise ValueError(msg)
        if profile == profiles.TRANSIENT_DRYING_PROFILE:
            time = _hdf5_dataset(handle, "time")
            transient = _hdf5_dataset(handle, "transient/fields")
            schedule = _hdf5_dataset(handle, "schedule/values")
            global_values = _hdf5_dataset(handle, "global/values")
            if time.dtype != np.float64 or schedule.dtype != np.float64 or global_values.dtype != np.float64 or transient.dtype != np.float32:
                msg = "Canonical transient, schedule, or global dtypes are invalid."
                raise ValueError(msg)
            transient_names = json.loads(_hdf5_text_attribute(transient.attrs["field_names"], label="transient.field_names"))
            transient_units = json.loads(_hdf5_text_attribute(transient.attrs["units"], label="transient.units"))
            schedule_names = json.loads(_hdf5_text_attribute(schedule.attrs["field_names"], label="schedule.field_names"))
            schedule_units = json.loads(_hdf5_text_attribute(schedule.attrs["units"], label="schedule.units"))
            global_names = json.loads(_hdf5_text_attribute(global_values.attrs["field_names"], label="global.field_names"))
            global_units = json.loads(_hdf5_text_attribute(global_values.attrs["units"], label="global.units"))
            time_values = np.asarray(time, dtype=np.float64)
            schedule_values = np.asarray(schedule, dtype=np.float64)
            global_array = np.asarray(global_values, dtype=np.float64)
            if (
                transient.shape != (time.size, len(profiles.TRANSIENT_FIELD_NAMES), y_axis.size, x_axis.size)
                or transient_names != list(profiles.TRANSIENT_FIELD_NAMES)
                or transient_units != list(profiles.TRANSIENT_FIELD_UNITS)
                or _hdf5_text_attribute(time.attrs.get("unit", ""), label="time.unit") != "h"
                or time_values.ndim != 1
                or time_values.size < 1
                or not np.isfinite(time_values).all()
                or np.any(np.diff(time_values) <= 0.0)
            ):
                msg = "Canonical transient field, time, name, or unit contract is invalid."
                raise ValueError(msg)
            if (
                schedule.shape != (169, len(profiles.SCHEDULE_FIELDS))
                or schedule_names != list(profiles.SCHEDULE_FIELDS)
                or schedule_units != list(profiles.SCHEDULE_UNITS)
                or not np.isfinite(schedule_values).all()
                or not np.array_equal(schedule_values[:, 0], np.arange(169, dtype=np.float64))
            ):
                msg = "Canonical schedule shape, values, names, or units are invalid."
                raise ValueError(msg)
            if (
                global_array.ndim != _TABLE_RANK
                or global_array.shape[0] < 1
                or global_array.shape[1] != len(profiles.GLOBAL_FIELD_NAMES)
                or global_names != list(profiles.GLOBAL_FIELD_NAMES)
                or global_units != list(profiles.GLOBAL_FIELD_UNITS)
                or not np.isfinite(global_array).all()
                or np.any(np.diff(global_array[:, 0]) <= 0.0)
            ):
                msg = "Canonical global diagnostic shape, values, names, or units are invalid."
                raise ValueError(msg)
            if (
                transient.compression != "gzip"
                or transient.compression_opts != _HDF5_COMPRESSION_LEVEL
                or not transient.shuffle
                or transient.chunks is None
            ):
                msg = "Canonical transient compression or chunking contract is invalid."
                raise ValueError(msg)
        identities = {
            name: _hdf5_text_attribute(handle.attrs[name], label=name)
            for name in ("case_input_id", "simulation_case_id", "scientific_config_digest", "template_sha256")
        }
        if any(len(value) != _SHA256_HEX_LENGTH for value in identities.values()):
            msg = "Canonical HDF5 identity attributes are malformed."
            raise ValueError(msg)
        git_commit = source_service.validate_git_commit(_hdf5_text_attribute(handle.attrs.get("git_commit", ""), label="git_commit"))
        return {"simulation_profile": profile, "git_commit": git_commit, **identities}


def convert_exports_to_hdf5(
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    exports: Sequence[Any],
    *,
    work_directory: Path,
    runtime_directory: Path,
    runtime_seconds: float,
) -> CanonicalCase:
    """Validate all raw adapters and atomically create the sole canonical payload."""
    x_axis, y_axis, static_fields = _static_fields(config, exports)
    transient_time: np.ndarray | None = None
    transient_fields: np.ndarray | None = None
    schedule_values: np.ndarray | None = None
    global_values: np.ndarray | None = None
    final_status: np.ndarray | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        transient_time, transient_fields = _transient_fields(config, exports, x_axis=x_axis, y_axis=y_axis)
        schedule_values = _schedule_values(case_payload, work_directory)
        global_values = _ordered_values(config, exports, profiles.GLOBAL_EXPORT_ROLE, profiles.GLOBAL_FIELD_NAMES)
        if np.any(np.diff(global_values[:, 0]) <= 0):
            msg = "Global diagnostic time must be strictly increasing."
            raise ValueError(msg)
        _validate_global_bulk_moisture(static_fields, transient_time, transient_fields, global_values)
        final_status = _ordered_values(config, exports, profiles.FINAL_STATUS_EXPORT_ROLE, profiles.FINAL_STATUS_FIELDS)
        if final_status.shape != (1, len(profiles.FINAL_STATUS_FIELDS)):
            msg = "Final status export must contain exactly one complete row."
            raise ValueError(msg)
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
            transient_time=transient_time,
            transient_fields=transient_fields,
            schedule_values=schedule_values,
            global_values=global_values,
            source_hashes=source_hashes,
        )
        validate_case_hdf5(temporary, expected_profile=config.profile.id)
        temporary.replace(destination)
        validate_case_hdf5(destination, expected_profile=config.profile.id)
    finally:
        temporary.unlink(missing_ok=True)
    status = _status(
        config,
        case_payload,
        transient_time=transient_time,
        final_status=final_status,
        runtime_seconds=runtime_seconds,
    )
    status_path = common.serialization.atomic_write_json(runtime_directory / "status.json", status)
    return CanonicalCase(
        path=destination,
        status_path=status_path,
        status=status,
        source_export_hashes=source_hashes,
    )
