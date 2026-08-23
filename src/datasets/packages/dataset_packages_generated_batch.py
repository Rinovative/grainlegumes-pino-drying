"""
dataset_packages_generated_batch.py

Interpret admitted Generation HDF5 cases as task-owned analysis views.
Responsibilities:
  - Preserve the established steady-flow tensor and fingerprint representation
  - Materialize canonical transient case trajectories without Dataset publication
  - Load bounded manifest prefixes with terminal provenance and runtime evidence
Design principles:
  - Generation alone admits manifests, publications, artifacts, and HDF5 identity
  - Dataset trajectory contracts own transient field, time, and handoff semantics
  - Complete trajectories remain nested case evidence rather than flattened samples
This module does NOT:
  - Validate Generation publication schemas, membership, or artifact hashes
  - Publish Dataset packages, derive target increments, or fit preprocessing state
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src import domain, generation
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract

from . import dataset_packages_trajectory as trajectory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec
    from src.generation.runtime.generation_runtime_batch import TerminalCaseEvidence


class _CompletedBatchEvidence(Protocol):
    """Describe an admitted complete or partial batch case collection."""

    @property
    def simulation_profile(self) -> str:
        """Return the canonical simulation profile."""
        ...

    @property
    def available_learning_views(self) -> tuple[str, ...]:
        """Return task views declared by the Generation profile."""
        ...

    @property
    def airflow_source(self) -> str:
        """Return the admitted airflow source identity."""
        ...

    @property
    def batch_id(self) -> str:
        """Return the immutable scientific batch identity."""
        ...

    @property
    def sampling_regime(self) -> str:
        """Return the configured sampling regime."""
        ...

    @property
    def template_sha256(self) -> str:
        """Return the admitted solver-template digest."""
        ...

    @property
    def git_commit(self) -> str:
        """Return the persisted source commit."""
        ...

    @property
    def cases(self) -> tuple[TerminalCaseEvidence, ...]:
        """Return ordered admitted completed cases."""
        ...

    def case(self, case_id: str) -> TerminalCaseEvidence:
        """Return one exact admitted case member."""
        ...

    def scientific_config_payload(self) -> dict[str, Any]:
        """Return independent resolved scientific configuration."""
        ...


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


def interpret_generated_case(
    batch: _CompletedBatchEvidence,
    case: TerminalCaseEvidence,
    *,
    task: TaskSpec,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Interpret one admitted canonical case as a steady-flow TaskSpec view."""
    if case.case_id not in {item.case_id for item in batch.cases} or batch.case(case.case_id) != case:
        msg = f"Case {case.case_id!r} is not bound to admitted batch {batch.batch_id!r}."
        raise ValueError(msg)
    if "steady_flow" not in batch.available_learning_views:
        msg = f"Batch {batch.batch_id!r} does not advertise a steady_flow view."
        raise ValueError(msg)
    hdf5_artifact = case.artifact_evidence("processed", "case.h5")
    with h5py.File(hdf5_artifact.path, "r") as handle:
        x_axis = np.asarray(_hdf5_dataset(handle, "coords/x"), dtype=np.float64)
        y_axis = np.asarray(_hdf5_dataset(handle, "coords/y"), dtype=np.float64)
        static_dataset = _hdf5_dataset(handle, "static/fields")
        static_values = np.asarray(static_dataset, dtype=np.float32)
        names = _string_list_attribute(static_dataset, "field_names")
    profile = generation.contracts.get_profile_contract(batch.simulation_profile)
    if names != [field.name for field in profile.static_fields]:
        msg = f"Canonical static HDF5 field order is invalid for {case.case_id}."
        raise ValueError(msg)
    static = {name: static_values[index] for index, name in enumerate(names)}
    case_inputs, case_outputs = _steady_flow_fields(static, x_axis=x_axis, y_axis=y_axis, task=task)
    inputs = torch.stack([torch.from_numpy(case_inputs[name]) for name in task.input_names])
    outputs = torch.stack([torch.from_numpy(case_outputs[name]) for name in task.output_names])
    case_payload = case.metadata_payload()
    metadata = {
        "case_id": case.case_id,
        "case_index": case.case_index,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "material_family": case.material_family,
        "sampling_regime": batch.sampling_regime,
        "ood": case_payload["ood"],
        "simulation_profile": batch.simulation_profile,
        "available_learning_views": list(batch.available_learning_views),
        "airflow_source": batch.airflow_source,
        "template_sha256": batch.template_sha256,
        "generator_version": case_payload["generator_version"],
        "seed": case_payload["seed_evidence"]["case_seed"],
        "parameters": case_payload["sampled_values"],
        "parameter_units": case_payload["sampled_units"],
        "geometry": case_payload["spatial_diagnostics"]["geometry"],
        "schedule_class": (case_payload["schedule_diagnostics"]["schedule_class"] if "schedule_diagnostics" in case_payload else None),
    }
    source = {
        "case_id": case.case_id,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "simulation_profile": batch.simulation_profile,
        "template_sha256": batch.template_sha256,
        "airflow_source": batch.airflow_source,
        "case_hdf5": hdf5_artifact.as_dict(),
    }
    fingerprint = identity.compute_case_fingerprint(
        task=task,
        case_id=case.case_id,
        source_identity=source,
        source_metadata=metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return (y_axis.size, x_axis.size), inputs, outputs, metadata, source, fingerprint


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


def _transient_case_metadata(
    batch: _CompletedBatchEvidence,
    case: TerminalCaseEvidence,
) -> dict[str, Any]:
    """Return stable completed-case metadata shared by transient EDA views."""
    payload = case.metadata_payload()
    seed_evidence = payload.get("seed_evidence")
    seed = seed_evidence.get("case_seed") if isinstance(seed_evidence, dict) else None
    spatial = payload.get("spatial_diagnostics")
    geometry = spatial.get("geometry") if isinstance(spatial, dict) else None
    schedule = payload.get("schedule_diagnostics")
    return {
        "case_id": case.case_id,
        "case_index": case.case_index,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "material_family": case.material_family,
        "material_role": payload.get("material_role"),
        "evaluation_regime": payload.get("evaluation_regime"),
        "natural_support_state": payload.get("natural_support_state"),
        "dataset_role": payload.get("dataset_role"),
        "case_family": payload.get("case_family", payload.get("evaluation_regime")),
        "sampling_regime": batch.sampling_regime,
        "ood": payload.get("ood"),
        "simulation_profile": batch.simulation_profile,
        "available_learning_views": list(batch.available_learning_views),
        "airflow_source": batch.airflow_source,
        "template_sha256": batch.template_sha256,
        "generator_version": payload.get("generator_version"),
        "seed": seed,
        "parameters": payload.get("sampled_values"),
        "parameter_units": payload.get("sampled_units"),
        "geometry": geometry,
        "schedule_class": (schedule.get("schedule_class") if isinstance(schedule, dict) else None),
    }


def interpret_generated_transient_case(
    batch: _CompletedBatchEvidence,
    case: TerminalCaseEvidence,
    *,
    task: TaskSpec,
) -> dict[str, Any]:
    """Interpret one admitted canonical case as complete transient EDA evidence."""
    if case.case_id not in {item.case_id for item in batch.cases} or batch.case(case.case_id) != case:
        msg = f"Case {case.case_id!r} is not bound to admitted batch {batch.batch_id!r}."
        raise ValueError(msg)
    if task.id != transient_contract.TRANSIENT_PROFILE_ID:
        msg = f"Transient generated-case interpretation requires transient_drying, got {task.id!r}."
        raise ValueError(msg)
    if task.id not in batch.available_learning_views:
        msg = f"Batch {batch.batch_id!r} does not advertise a {task.id} view."
        raise ValueError(msg)
    profile = generation.contracts.get_profile_contract(batch.simulation_profile)
    if profile.id != transient_contract.TRANSIENT_PROFILE_ID:
        msg = f"Batch {batch.batch_id!r} is not a canonical transient_drying source."
        raise ValueError(msg)
    state_names = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    state_units = tuple(field.unit for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    boundary_names = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning)
    hdf5_artifact = case.artifact("processed", "case.h5")
    with h5py.File(hdf5_artifact.path, "r") as handle:
        layout = trajectory.resolve_transient_case_layout(
            handle,
            hdf5_artifact.path,
            sample_id_prefix=case.case_id,
        )
        case_index = {
            "sequence_length": layout.regular_state_count,
            "irregular_stop_time": layout.exact_stop_time,
        }
        arrays = trajectory.read_transient_case_arrays(
            handle,
            hdf5_artifact.path,
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
    timing = generation.runtime.timing.load_case_timing(case, batch=batch).as_dict()
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
        "meta": _transient_case_metadata(batch, case),
        "source": {
            "case_id": case.case_id,
            "case_input_id": case.case_input_id,
            "simulation_case_id": case.simulation_case_id,
            "simulation_profile": batch.simulation_profile,
            "template_sha256": batch.template_sha256,
            "airflow_source": batch.airflow_source,
            "case_hdf5": hdf5_artifact.as_dict(),
        },
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


def load_generated_batch(
    batch_name: str,
    *,
    task_id: str = "steady_flow",
    storage_root: Path | str | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Load a validated generated-batch prefix without publishing tensors."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id not in {"steady_flow", transient_contract.TRANSIENT_PROFILE_ID}:
        msg = f"Generated analysis views do not support task {task.id!r}."
        raise ValueError(msg)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        msg = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(msg)
    batch = generation.runtime.admit_terminal_batch(
        batch_name,
        storage_root=storage_root,
        validation_depth="routine",
    )
    selected = batch.cases if max_cases is None else batch.cases[:max_cases]
    generated_identity = identity.build_generated_batch_identity(batch.manifest_payload())
    rows: list[dict[str, Any]] = []
    iterator = tqdm(
        selected,
        desc=f"Loading {batch_name}",
        unit="case",
        disable=not show_progress,
    )
    for case in iterator:
        if task.id == "steady_flow":
            _shape, inputs, outputs, metadata, _source, _fingerprint = interpret_generated_case(
                batch,
                case,
                task=task,
            )
            rows.append(
                {
                    **{name: inputs[index].numpy() for index, name in enumerate(task.input_names)},
                    **{name: outputs[index].numpy() for index, name in enumerate(task.output_names)},
                    "meta": metadata,
                }
            )
        else:
            rows.append(
                interpret_generated_transient_case(
                    batch,
                    case,
                    task=task,
                )
            )
    return {
        "batch_name": batch_name,
        "generation_root": batch.generation_root,
        "manifest_path": batch.manifest_path,
        "manifest_sha256": batch.manifest_sha256,
        "generated_batch_identity": generated_identity,
        "sample_ids": [case.case_id for case in selected],
        "available_case_count": len(batch.cases),
        "rows": rows,
        "task": task,
        "analysis_representation": ("steady_field_rows" if task.id == "steady_flow" else "transient_complete_case_rows"),
        "dataset_backend": "canonical_generation_hdf5",
    }
