"""
===============================================================================
dataset_transient.py
===============================================================================
Build and load physical-unit one-hour transient-drying transition pairs.
Responsibilities:
  - Index consecutive regular hourly states from canonical case.h5 archives
  - Derive delta targets at dataset construction and retain static references
  - Load dynamic, static, boundary, scalar, target, and provenance tensors
Design principles:
  - Absolute states remain canonical; increments are derived learning targets
  - Irregular exact-stop states and transitions beyond termination are excluded
  - No normalization, task registration, or training semantics enter this layer
This module does NOT:
  - Register a transient TaskSpec, fit normalization, train models, or roll out
  - Copy large static fields into every indexed transition
===============================================================================
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src import common
from src.generation import generation_profiles as profiles
from src.generation import generation_storage as storage

from .dataset_transient_contract import TRANSIENT_STEP_CONTRACT

if TYPE_CHECKING:
    from collections.abc import Sequence

TRANSIENT_INDEX_SCHEMA_KIND = "vp2_transient_physical_index"
TRANSIENT_INDEX_SCHEMA_VERSION = 1
_EVALUATION_REGIMES: Final = frozenset({"id", "parameter_ood", "near_family_ood", "far_family_ood"})


def _json_attribute(value: Any, *, label: str) -> list[str]:
    """Decode one JSON list HDF5 attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        message = f"HDF5 attribute {label!r} must contain JSON text."
        raise TypeError(message)
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        message = f"HDF5 attribute {label!r} must contain a string list."
        raise TypeError(message)
    return decoded


def _text_attribute(value: Any, *, label: str) -> str:
    """Return one required textual HDF5 attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str) or not value:
        message = f"HDF5 attribute {label!r} must be non-empty text."
        raise TypeError(message)
    return value


def _regular_transition_indices(time: np.ndarray) -> tuple[int, ...]:
    """Return only consecutive integer-hour transitions."""
    if time.ndim != 1 or not np.isfinite(time).all() or time.size < 1:
        message = "Transient time must be one finite one-dimensional sequence."
        raise ValueError(message)
    return tuple(
        index
        for index in range(time.size - 1)
        if math.isclose(float(time[index]), round(float(time[index])), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(time[index + 1]), round(float(time[index + 1])), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(time[index + 1] - time[index]), TRANSIENT_STEP_CONTRACT.time_step, rel_tol=0.0, abs_tol=1e-12)
    )


def _case_record(path: Path, evaluation_regime: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one canonical case and return case/sample index records."""
    if evaluation_regime not in _EVALUATION_REGIMES:
        message = f"Unsupported transient evaluation regime: {evaluation_regime!r}."
        raise ValueError(message)
    storage.validate_case_hdf5(path, expected_profile=profiles.TRANSIENT_DRYING_PROFILE)
    with h5py.File(path, "r") as handle:
        time = np.asarray(handle["time"], dtype=np.float64)
        transitions = _regular_transition_indices(time)
        case_input_id = _text_attribute(handle.attrs["case_input_id"], label="case_input_id")
        simulation_case_id = _text_attribute(handle.attrs["simulation_case_id"], label="simulation_case_id")
        material_family = _text_attribute(handle.attrs["material_family"], label="material_family")
        transient_names = _json_attribute(handle["transient/fields"].attrs["field_names"], label="transient.field_names")
        static_names = _json_attribute(handle["static/fields"].attrs["field_names"], label="static.field_names")
        schedule_dataset = handle["schedule/values"]
        scalar_dataset = handle["scalar/values"]
        schedule_names = _json_attribute(schedule_dataset.attrs["field_names"], label="schedule.field_names")
        scalar_names = _json_attribute(scalar_dataset.attrs["field_names"], label="scalar.field_names")
        schedule = np.asarray(schedule_dataset, dtype=np.float64)
        scalars = np.asarray(scalar_dataset, dtype=np.float64)
        if transient_names != list(profiles.TRANSIENT_FIELD_NAMES):
            message = f"Transient fields are not canonical in {path}."
            raise ValueError(message)
        required_static = {field.name for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning}.difference({"x", "y"})
        if not required_static.issubset(static_names):
            message = f"Transient static conditioning is incomplete in {path}."
            raise ValueError(message)
        if schedule_names != list(profiles.SCHEDULE_FIELDS) or scalar_names != list(profiles.SCALAR_INPUT_FIELDS):
            message = f"Transient boundary or scalar conditioning is not canonical in {path}."
            raise ValueError(message)
        record = {
            "case_hdf5": str(path),
            "case_hdf5_sha256": common.serialization.file_sha256(path),
            "case_input_id": case_input_id,
            "simulation_case_id": simulation_case_id,
            "material_family": material_family,
            "evaluation_regime": evaluation_regime,
            "sequence_length": int(time.size),
            "transition_count": len(transitions),
            "time": time.tolist(),
            "static_field_reference": "static/fields",
        }
        schedule_time = schedule[:, schedule_names.index("t")]
        scalar_lookup = {name: float(scalars[position]) for position, name in enumerate(scalar_names)}
        samples: list[dict[str, Any]] = []
        for index in transitions:
            time_value = float(time[index])
            next_time = time_value + TRANSIENT_STEP_CONTRACT.time_step
            current_matches = np.flatnonzero(np.isclose(schedule_time, time_value, rtol=0.0, atol=1e-12))
            next_matches = np.flatnonzero(np.isclose(schedule_time, next_time, rtol=0.0, atol=1e-12))
            if current_matches.size != 1 or next_matches.size != 1:
                message = f"Schedule lacks exact endpoints for {time_value} h in {path}."
                raise ValueError(message)
            current = int(current_matches[0])
            following = int(next_matches[0])
            samples.append(
                {
                    "time_index": index,
                    "time": time_value,
                    "dt": TRANSIENT_STEP_CONTRACT.time_step,
                    "schedule_values": {
                        "T_in_t_n": float(schedule[current, schedule_names.index("T_in")]),
                        "T_in_t_np1": float(schedule[following, schedule_names.index("T_in")]),
                        "phi_in_t_n": float(schedule[current, schedule_names.index("phi_in")]),
                        "phi_in_t_np1": float(schedule[following, schedule_names.index("phi_in")]),
                        "T_amb": scalar_lookup["T_amb"],
                    },
                    "target_available": True,
                }
            )
    return record, samples


def _contract_payload() -> dict[str, Any]:
    """Return the exact persisted physical transition contract."""
    return {
        "time_step": TRANSIENT_STEP_CONTRACT.time_step,
        "time_unit": TRANSIENT_STEP_CONTRACT.time_unit,
        "dynamic_state": [field.name for field in TRANSIENT_STEP_CONTRACT.dynamic_state],
        "static_spatial_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
        "step_boundary_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
        "scalar_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning],
        "target_increments": [field.name for field in TRANSIENT_STEP_CONTRACT.target_increments],
    }


def _identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return path-independent content used for transient dataset identity."""
    cases = [{key: value for key, value in case.items() if key != "case_hdf5"} for case in payload["cases"]]
    return {
        "schema_kind": payload["schema_kind"],
        "schema_version": payload["schema_version"],
        "dataset_name": payload["dataset_name"],
        "cases": cases,
        "samples": payload["samples"],
        "contract": payload["contract"],
        "builder": payload["builder"],
        "source_provenance": payload["source_provenance"],
    }


def build_transient_index(
    sources: Sequence[tuple[Path | str, str]],
    destination: Path | str | None,
    *,
    dataset_name: str,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable physical-unit transition index from canonical cases."""
    if not sources:
        message = "Transient index construction requires at least one source case."
        raise ValueError(message)
    if not dataset_name or "__" not in dataset_name:
        message = "Transient dataset_name must use the canonical human-readable grammar."
        raise ValueError(message)
    cases: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    identities: set[str] = set()
    for case_index, (raw_path, evaluation_regime) in enumerate(sources):
        path = Path(raw_path).expanduser().resolve()
        record, case_samples = _case_record(path, evaluation_regime)
        identity = str(record["simulation_case_id"])
        if identity in identities:
            message = f"Transient source simulation identity is duplicated: {identity}."
            raise ValueError(message)
        identities.add(identity)
        cases.append(record)
        samples.extend({"case_index": case_index, **sample} for sample in case_samples)
    if not samples:
        message = "Transient source cases contain no consecutive regular one-hour transitions."
        raise ValueError(message)
    payload_base = {
        "schema_kind": TRANSIENT_INDEX_SCHEMA_KIND,
        "schema_version": TRANSIENT_INDEX_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "cases": cases,
        "samples": samples,
        "contract": _contract_payload(),
        "builder": "src.datasets.dataset_transient.build_transient_index",
        "source_provenance": deepcopy(source_provenance or {}),
    }
    dataset_digest = common.serialization.canonical_json_sha256(_identity_payload(payload_base))
    payload = {
        **payload_base,
        "dataset_id": f"{dataset_name}__{dataset_digest[:16]}",
        "dataset_digest": dataset_digest,
        "sample_count": len(samples),
    }
    if destination is None:
        return payload
    destination_path = Path(destination).expanduser().resolve()
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if destination_path.exists():
        if not destination_path.is_file() or destination_path.read_text(encoding="utf-8") != serialized:
            message = f"Existing transient index conflicts with requested identity: {destination_path}."
            raise FileExistsError(message)
        return payload
    common.serialization.atomic_write_text(destination_path, serialized)
    return payload


class TransientPhysicalDataset(Dataset[dict[str, Any]]):
    """Load indexed one-hour transient pairs as unnormalized physical tensors."""

    def __init__(self, index_path: Path | str) -> None:
        """Load and minimally validate one immutable transient index."""
        self.index_path = Path(index_path).expanduser().resolve()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Could not read transient dataset index: {self.index_path}"
            raise ValueError(message) from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_kind") != TRANSIENT_INDEX_SCHEMA_KIND
            or payload.get("schema_version") != TRANSIENT_INDEX_SCHEMA_VERSION
            or not isinstance(payload.get("cases"), list)
            or not isinstance(payload.get("samples"), list)
            or payload.get("sample_count") != len(payload.get("samples", []))
        ):
            message = f"Transient dataset index schema is invalid: {self.index_path}"
            raise ValueError(message)
        if payload.get("contract") != _contract_payload():
            message = f"Transient dataset contract is invalid: {self.index_path}"
            raise ValueError(message)
        digest = common.serialization.canonical_json_sha256(_identity_payload(payload))
        if payload.get("dataset_digest") != digest or payload.get("dataset_id") != f"{payload['dataset_name']}__{digest[:16]}":
            message = f"Transient dataset identity mismatch: {self.index_path}"
            raise ValueError(message)
        self.payload = payload

    def __len__(self) -> int:
        """Return indexed one-hour transition count."""
        return len(self.payload["samples"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one physical-unit conditioning/target tensor bundle."""
        sample = self.payload["samples"][index]
        case = self.payload["cases"][sample["case_index"]]
        path = Path(case["case_hdf5"])
        if common.serialization.file_sha256(path) != case["case_hdf5_sha256"]:
            message = f"Transient source case changed after indexing: {path}."
            raise RuntimeError(message)
        time_index = int(sample["time_index"])
        with h5py.File(path, "r") as handle:
            transient_names = _json_attribute(handle["transient/fields"].attrs["field_names"], label="transient.field_names")
            static_names = _json_attribute(handle["static/fields"].attrs["field_names"], label="static.field_names")
            schedule_names = _json_attribute(handle["schedule/values"].attrs["field_names"], label="schedule.field_names")
            scalar_names = _json_attribute(handle["scalar/values"].attrs["field_names"], label="scalar.field_names")
            transient = np.asarray(handle["transient/fields"], dtype=np.float32)
            static = np.asarray(handle["static/fields"], dtype=np.float32)
            schedule = np.asarray(handle["schedule/values"], dtype=np.float64)
            scalars = np.asarray(handle["scalar/values"], dtype=np.float64)
            x_axis = np.asarray(handle["coords/x"], dtype=np.float64)
            y_axis = np.asarray(handle["coords/y"], dtype=np.float64)

        dynamic_indices = [transient_names.index(field.name) for field in TRANSIENT_STEP_CONTRACT.dynamic_state]
        dynamic = transient[time_index, dynamic_indices]
        target = transient[time_index + 1, dynamic_indices] - dynamic

        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        static_arrays = {
            "x": x_grid.astype(np.float32),
            "y": y_grid.astype(np.float32),
            **{name: static[static_names.index(name)] for name in static_names},
        }
        static_conditioning = np.stack(
            [static_arrays[field.name] for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
            axis=0,
        )
        time = float(sample["time"])
        schedule_time = schedule[:, schedule_names.index("t")]
        current_matches = np.flatnonzero(np.isclose(schedule_time, time, rtol=0.0, atol=1e-12))
        next_matches = np.flatnonzero(
            np.isclose(
                schedule_time,
                time + TRANSIENT_STEP_CONTRACT.time_step,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if current_matches.size != 1 or next_matches.size != 1:
            message = f"Schedule lacks exact boundary endpoints for transition {time} h in {path}."
            raise ValueError(message)
        current = int(current_matches[0])
        following = int(next_matches[0])
        scalar_lookup = {name: float(scalars[position]) for position, name in enumerate(scalar_names)}
        observed_schedule_values = {
            "T_in_t_n": float(schedule[current, schedule_names.index("T_in")]),
            "T_in_t_np1": float(schedule[following, schedule_names.index("T_in")]),
            "phi_in_t_n": float(schedule[current, schedule_names.index("phi_in")]),
            "phi_in_t_np1": float(schedule[following, schedule_names.index("phi_in")]),
            "T_amb": scalar_lookup["T_amb"],
        }
        if observed_schedule_values != sample["schedule_values"]:
            message = f"Indexed schedule values disagree with canonical source {path}."
            raise RuntimeError(message)
        boundary = np.asarray(
            [observed_schedule_values[field.name] for field in TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
            dtype=np.float32,
        )
        scalar_conditioning = np.asarray(
            [scalar_lookup[field.name] for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning],
            dtype=np.float32,
        )
        return {
            "dynamic_state": torch.from_numpy(np.ascontiguousarray(dynamic)),
            "static_spatial_conditioning": torch.from_numpy(np.ascontiguousarray(static_conditioning)),
            "step_boundary_conditioning": torch.from_numpy(boundary),
            "scalar_conditioning": torch.from_numpy(scalar_conditioning),
            "target_increments": torch.from_numpy(np.ascontiguousarray(target)),
            "field_names": {
                "dynamic_state": [field.name for field in TRANSIENT_STEP_CONTRACT.dynamic_state],
                "static_spatial_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
                "step_boundary_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
                "scalar_conditioning": [field.name for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning],
                "target_increments": [field.name for field in TRANSIENT_STEP_CONTRACT.target_increments],
            },
            "units": {
                "dynamic_state": [field.unit for field in TRANSIENT_STEP_CONTRACT.dynamic_state],
                "static_spatial_conditioning": [field.unit for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
                "step_boundary_conditioning": [field.unit for field in TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
                "scalar_conditioning": [field.unit for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning],
                "target_increments": [field.unit for field in TRANSIENT_STEP_CONTRACT.target_increments],
            },
            "meta": deepcopy(
                {
                    "source_case_identity": case["simulation_case_id"],
                    "source_case_input_identity": case["case_input_id"],
                    "material_family": case["material_family"],
                    "source_dataset_name": self.payload["dataset_name"],
                    "source_dataset_id": self.payload["dataset_id"],
                    "source_evaluation_regime": case["evaluation_regime"],
                    "time_index": time_index,
                    "time": time,
                    "time_unit": TRANSIENT_STEP_CONTRACT.time_unit,
                    "sequence_length": case["sequence_length"],
                    "dt": sample["dt"],
                    "static_field_reference": case["static_field_reference"],
                    "schedule_values": deepcopy(sample["schedule_values"]),
                    "target_available": sample["target_available"],
                }
            ),
        }
