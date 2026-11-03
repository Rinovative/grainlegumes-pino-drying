"""Interpret exact admitted Generation source cases as task-owned analysis views.

Responsibilities:
  - Preserve the established steady-flow tensor and fingerprint representation
  - Materialize canonical transient case trajectories without Dataset publication

Design principles:
  - Generation alone admits source cases, artifacts, and HDF5 identity
  - Dataset trajectory contracts own transient field, time, and handoff semantics
  - Complete trajectories remain nested case evidence rather than flattened samples

This module does NOT:
  - Validate Generation publication schemas, membership, or artifact hashes
  - Publish Dataset packages, derive target increments, or fit preprocessing state
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch

from src import generation
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract

from . import dataset_packages_trajectory as trajectory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec
    from src.generation.runtime.generation_runtime_batch import TerminalCaseEvidence


def _hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required canonical HDF5 dataset."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        msg = f"Canonical HDF5 member {name!r} must be a dataset."
        raise TypeError(msg)
    return value


def _string_list_attribute(dataset: h5py.Dataset, name: str) -> list[str]:
    """Decode one required JSON string-list dataset attribute."""
    value = dataset.attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        msg = f"Canonical HDF5 attribute {name!r} must be text."
        raise TypeError(msg)
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        msg = f"Canonical HDF5 attribute {name!r} must contain a JSON string list."
        raise TypeError(msg)
    return decoded


def _steady_flow_fields(
    static: Mapping[str, np.ndarray],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    task: TaskSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the canonical steady-flow channel and permeability contract."""
    if task.id != "steady_flow":
        msg = f"Generated airflow views support only steady_flow, got {task.id!r}."
        raise ValueError(msg)
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    raw_kxx = static["Kxx"]
    raw_kxy = static["Kxy"]
    raw_kyy = static["Kyy"]
    determinant = raw_kxx * raw_kyy - raw_kxy**2
    if np.any(raw_kxx <= 0) or np.any(raw_kyy <= 0) or np.any(determinant <= 0):
        msg = "Canonical permeability tensor must be positive definite at every point."
        raise ValueError(msg)
    values: dict[str, np.ndarray] = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": np.log10(raw_kxx),
        "Kxy": raw_kxy / np.sqrt(raw_kxx * raw_kyy),
        "Kyy": np.log10(raw_kyy),
        "eps_bed": static["eps_bed"],
        "p_in_bc": static["p_in_bc"],
        "p": static["p"],
        "u": static["u"],
        "v": static["v"],
    }
    if np.any((values["eps_bed"] <= 0) | (values["eps_bed"] > 1)):
        msg = "Canonical porosity must satisfy 0 < eps <= 1."
        raise ValueError(msg)
    expected = set(task.input_names) | set(task.output_names)
    if set(values) != expected:
        msg = "Current steady_flow TaskSpec no longer matches the canonical HDF5 view."
        raise RuntimeError(msg)
    converted = {name: np.asarray(value, dtype=np.float32).copy() for name, value in values.items()}
    if not all(np.isfinite(value).all() for value in converted.values()):
        msg = "Steady-flow learning view is non-finite after float32 conversion."
        raise ValueError(msg)
    return (
        {name: converted[name] for name in task.input_names},
        {name: converted[name] for name in task.output_names},
    )


def _required_named_array(
    handle: h5py.File,
    name: str,
    *,
    expected_names: tuple[str, ...],
    expected_units: tuple[str, ...],
    field_axis: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Read one finite canonical named array with exact field and unit order."""
    dataset = _hdf5_dataset(handle, name)
    names = tuple(_string_list_attribute(dataset, "field_names"))
    units = tuple(_string_list_attribute(dataset, "units"))
    if names != expected_names or units != expected_units:
        msg = f"Canonical HDF5 field or unit order is invalid for {name!r}."
        raise ValueError(msg)
    values = np.asarray(dataset, dtype=dtype)
    if values.ndim == 0 or values.shape[field_axis] != len(expected_names):
        msg = f"Canonical HDF5 shape is invalid for {name!r}: {values.shape}."
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = f"Canonical HDF5 values are non-finite for {name!r}."
        raise ValueError(msg)
    return np.ascontiguousarray(values)


def interpret_admitted_transient_source_case(admitted: Any, *, task: TaskSpec) -> dict[str, Any]:
    """Interpret an exact successful Generation source without a composed batch."""
    case = admitted.case
    source_identity = admitted.source_identity
    case_payload = case.metadata_payload()
    if task.id != transient_contract.TRANSIENT_PROFILE_ID:
        message = f"Transient source interpretation requires transient_drying, got {task.id!r}."
        raise ValueError(message)
    if source_identity["case_id"] != case.case_id or source_identity["case_hdf5_sha256"] != case.case_hdf5_sha256:
        msg = "Admitted transient source identity disagrees with Generation evidence."
        raise ValueError(msg)
    if task.id not in case_payload["available_learning_views"]:
        msg = "Admitted source case does not expose the transient learning view."
        raise ValueError(msg)
    hdf5_artifact = case.artifact("processed", "case.h5")
    if admitted.hdf5_path != hdf5_artifact.path:
        msg = "Admitted transient HDF5 path disagrees with exact case evidence."
        raise ValueError(msg)
    seed_evidence = case_payload.get("seed_evidence")
    spatial = case_payload.get("spatial_diagnostics")
    schedule = case_payload.get("schedule_diagnostics")
    metadata = {
        "case_id": case.case_id,
        "case_index": case.case_index,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "material_family": case.material_family,
        "domain_profile": source_identity["domain_profile"],
        "sampling_regime": case_payload["sampling_regime"],
        "simulation_profile": case_payload["simulation_profile"],
        "available_learning_views": list(case_payload["available_learning_views"]),
        "airflow_source": case_payload["airflow_source"],
        "template_sha256": case_payload["template"]["sha256"],
        "generator_version": case_payload["generator_version"],
        "seed": seed_evidence.get("case_seed") if isinstance(seed_evidence, dict) else None,
        "parameters": case_payload.get("sampled_values"),
        "parameter_units": case_payload.get("sampled_units"),
        "geometry": spatial.get("geometry") if isinstance(spatial, dict) else None,
        "schedule_class": schedule.get("schedule_class") if isinstance(schedule, dict) else None,
    }
    return _interpret_transient_case(
        case,
        task=task,
        hdf5_path=admitted.hdf5_path,
        profile_name=case_payload["simulation_profile"],
        metadata=metadata,
        source={**source_identity, "case_hdf5": hdf5_artifact.as_dict(), "design_provenance": admitted.design_provenance},
        timing=admitted.load_timing().as_dict(),
    )


def _interpret_transient_case(
    case: TerminalCaseEvidence,
    *,
    task: TaskSpec,
    hdf5_path: Path,
    profile_name: str,
    metadata: dict[str, Any],
    source: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Check complete physical trajectory, protocol, and admitted timing evidence."""
    if task.id != transient_contract.TRANSIENT_PROFILE_ID:
        msg = "Transient case interpretation requires transient_drying."
        raise ValueError(msg)
    profile = generation.contracts.get_profile_contract(profile_name)
    if profile.id != transient_contract.TRANSIENT_PROFILE_ID:
        msg = f"Case {case.case_id!r} is not a canonical transient_drying source."
        raise ValueError(msg)
    state_names = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    state_units = tuple(field.unit for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    boundary_names = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning)
    with h5py.File(hdf5_path, "r") as handle:
        layout = trajectory.resolve_transient_case_layout(
            handle,
            hdf5_path,
            sample_id_prefix=case.case_id,
        )
        case_index = {
            "sequence_length": layout.regular_state_count,
            "irregular_stop_time": layout.exact_stop_time,
        }
        arrays = trajectory.read_transient_case_arrays(
            handle,
            hdf5_path,
            case_index,
            layout.samples,
            expected_regular_horizon=layout.configured_horizon,
            complete_case=True,
        )
        static_names = tuple(field.name for field in profile.static_fields)
        static_units = tuple(field.unit for field in profile.static_fields)
        static_values = _required_named_array(
            handle,
            "static/fields",
            expected_names=static_names,
            expected_units=static_units,
            field_axis=0,
            dtype=np.dtype(np.float32),
        )
        schedule_names = tuple(field.name for field in profile.schedule_fields)
        schedule_units = tuple(field.unit for field in profile.schedule_fields)
        schedule_values = _required_named_array(
            handle,
            "schedule/values",
            expected_names=schedule_names,
            expected_units=schedule_units,
            field_axis=1,
            dtype=np.dtype(np.float64),
        )
        scalar_names = tuple(field.name for field in profile.scalar_inputs)
        scalar_units = tuple(field.unit for field in profile.scalar_inputs)
        scalar_values = _required_named_array(
            handle,
            "scalar/values",
            expected_names=scalar_names,
            expected_units=scalar_units,
            field_axis=0,
            dtype=np.dtype(np.float64),
        )
        global_names = generation.contracts.profiles.GLOBAL_FIELD_NAMES
        global_units = generation.contracts.profiles.GLOBAL_FIELD_UNITS
        global_values = _required_named_array(
            handle,
            "global/values",
            expected_names=global_names,
            expected_units=global_units,
            field_axis=1,
            dtype=np.dtype(np.float64),
        )
        final_names = generation.contracts.profiles.FINAL_STATUS_FIELDS
        final_units = generation.contracts.profiles.FINAL_STATUS_UNITS
        final_values = _required_named_array(
            handle,
            "final_status/values",
            expected_names=final_names,
            expected_units=final_units,
            field_axis=0,
            dtype=np.dtype(np.float64),
        )
        x_axis = np.asarray(_hdf5_dataset(handle, "coords/x"), dtype=np.float32)
        y_axis = np.asarray(_hdf5_dataset(handle, "coords/y"), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    static_fields = {
        "x": np.ascontiguousarray(x_grid),
        "y": np.ascontiguousarray(y_grid),
        **{name: np.ascontiguousarray(static_values[index]) for index, name in enumerate(static_names)},
    }
    state_trajectories = {name: np.ascontiguousarray(arrays.states[:, index]) for index, name in enumerate(state_names)}
    boundary_intervals = {name: np.ascontiguousarray(arrays.boundary[:, index]) for index, name in enumerate(boundary_names)}
    scalar_conditioning = {name: float(scalar_values[index]) for index, name in enumerate(scalar_names)}
    schedule = {name: np.ascontiguousarray(schedule_values[:, index]) for index, name in enumerate(schedule_names)}
    global_series = {name: np.ascontiguousarray(global_values[:, index]) for index, name in enumerate(global_names)}
    final_status = {name: float(final_values[index]) for index, name in enumerate(final_names)}
    exact_stop = None
    if arrays.exact_stop_time is not None:
        if arrays.exact_stop_state is None:
            msg = "Transient exact-stop time lacks its diagnostic state."
            raise RuntimeError(msg)
        exact_stop = {
            "time_hours": arrays.exact_stop_time,
            "state": {name: np.ascontiguousarray(arrays.exact_stop_state[index]) for index, name in enumerate(state_names)},
            "usage": "diagnostic_only_no_training_transition_or_rollout",
        }
    expected_duration = float(arrays.time[-1]) if arrays.exact_stop_time is None else arrays.exact_stop_time
    if not np.isclose(
        timing["physical_duration_hours"],
        expected_duration,
        rtol=0.0,
        atol=layout.tolerance,
    ):
        msg = "Transient status duration disagrees with canonical HDF5 time evidence."
        raise ValueError(msg)
    if not np.isclose(
        timing["final_wet_fraction"],
        final_status["f_wet_dm_final"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        msg = "Transient status final wet fraction disagrees with canonical HDF5 evidence."
        raise ValueError(msg)
    transition_t_n = np.asarray(
        [arrays.time[int(sample["time_index_n"])] for sample in layout.samples],
        dtype=np.float64,
    )
    transition_t_n_plus_1 = np.asarray(
        [arrays.time[int(sample["time_index_n_plus_1"])] for sample in layout.samples],
        dtype=np.float64,
    )
    return {
        "state_trajectories": state_trajectories,
        "static_fields": static_fields,
        "boundary_intervals": boundary_intervals,
        "scalar_conditioning": scalar_conditioning,
        "schedule": schedule,
        "time": {
            "regular_state_hours": arrays.time.copy(),
            "valid_state_mask": np.ones(arrays.time.shape, dtype=bool),
            "trajectory_length": int(arrays.time.size),
            "transition_t_n_hours": transition_t_n,
            "transition_t_n_plus_1_hours": transition_t_n_plus_1,
            "transition_dt_hours": transition_t_n_plus_1 - transition_t_n,
            "configured_horizon_hours": layout.configured_horizon,
            "classification_tolerance_hours": layout.tolerance,
            "unit": transient_contract.TRANSIENT_STEP_CONTRACT.time_unit,
        },
        "exact_stop": exact_stop,
        "global_series": global_series,
        "final_status": final_status,
        "completion": {
            **{
                key: timing[key]
                for key in (
                    "physical_duration_hours",
                    "time_to_target_hours",
                    "target_reached",
                    "right_censored",
                    "final_wet_fraction",
                    "target_wet_fraction_limit",
                    "physical_duration_availability",
                    "target_wet_fraction_limit_availability",
                )
            },
            "final_bulk_moisture_wb": final_status["X_wb_bulk_final"],
            "target_moisture_wb": scalar_conditioning["X_target_wb"],
        },
        "runtime": timing,
        "meta": metadata,
        "source": source,
        "contracts": {
            "state_names": state_names,
            "state_units": state_units,
            "static_names": ("x", "y", *static_names),
            "static_units": ("m", "m", *static_units),
            "boundary_names": boundary_names,
            "boundary_units": tuple(field.unit for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning),
            "scalar_names": scalar_names,
            "scalar_units": scalar_units,
            "schedule_names": schedule_names,
            "schedule_units": schedule_units,
            "global_names": global_names,
            "global_units": global_units,
            "final_status_names": final_names,
            "final_status_units": final_units,
        },
    }


def interpret_admitted_source_case(
    admitted: Any,
    *,
    package_case_id: str,
    task: TaskSpec,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Convert one exact admitted source case without fabricating a Generation batch."""
    case = admitted.case
    source = admitted.source_identity
    if source["case_id"] != case.case_id or source["case_hdf5_sha256"] != case.case_hdf5_sha256:
        msg = "Admitted source case identity disagrees with Generation evidence."
        raise ValueError(msg)
    case_payload = case.metadata_payload()
    if "steady_flow" not in case_payload["available_learning_views"]:
        msg = "Source case does not expose the steady-flow learning view."
        raise ValueError(msg)
    with h5py.File(admitted.hdf5_path, "r") as handle:
        x_axis = np.asarray(_hdf5_dataset(handle, "coords/x"), dtype=np.float64)
        y_axis = np.asarray(_hdf5_dataset(handle, "coords/y"), dtype=np.float64)
        static_dataset = _hdf5_dataset(handle, "static/fields")
        static_values = np.asarray(static_dataset, dtype=np.float32)
        names = _string_list_attribute(static_dataset, "field_names")
    profile = generation.contracts.get_profile_contract(case_payload["simulation_profile"])
    if names != [field.name for field in profile.static_fields]:
        msg = "Canonical steady-flow static HDF5 fields have invalid order."
        raise ValueError(msg)
    static = {name: static_values[index] for index, name in enumerate(names)}
    case_inputs, case_outputs = _steady_flow_fields(static, x_axis=x_axis, y_axis=y_axis, task=task)
    inputs = torch.stack([torch.from_numpy(case_inputs[name]) for name in task.input_names])
    outputs = torch.stack([torch.from_numpy(case_outputs[name]) for name in task.output_names])
    metadata = {
        "case_id": package_case_id,
        "source_case_id": case.case_id,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "material_family": case.material_family,
        "sampling_regime": case_payload["sampling_regime"],
        "simulation_profile": case_payload["simulation_profile"],
        "domain_profile": source["domain_profile"],
        "available_learning_views": list(case_payload["available_learning_views"]),
        "airflow_source": case_payload["airflow_source"],
        "template_sha256": case_payload["template"]["sha256"],
        "generator_version": case_payload["generator_version"],
        "seed": case_payload["seed_evidence"]["case_seed"],
        "parameters": case_payload["sampled_values"],
        "parameter_units": case_payload["sampled_units"],
        "geometry": case_payload["spatial_diagnostics"]["geometry"],
        "schedule_class": case_payload.get("schedule_diagnostics", {}).get("schedule_class"),
    }
    physical_source = {"package_case_id": package_case_id, **source}
    fingerprint = identity.compute_case_fingerprint(
        task=task,
        case_id=package_case_id,
        source_identity=physical_source,
        source_metadata=metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return (y_axis.size, x_axis.size), inputs, outputs, metadata, physical_source, fingerprint
