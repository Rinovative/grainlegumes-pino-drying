"""
dataset_packages_generated_batch.py

Interpret admitted Generation HDF5 cases as task-owned steady-flow tensors.
Responsibilities:
  - Apply the steady-flow TaskSpec field order and tensor representations
  - Build Dataset provenance and fingerprints from typed terminal evidence
  - Load bounded EDA prefixes without publishing training datasets
Design principles:
  - Generation alone admits manifests, publications, artifacts, and HDF5 identity
  - Dataset interpretation consumes immutable admitted paths and identities
  - Scientific fields fail closed before tensor construction
This module does NOT:
  - Validate Generation publication schemas, membership, or artifact hashes
  - Admit CSV views, publish packages, or define transient tensor semantics
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src import domain, generation
from src.datasets.contracts import dataset_contracts_identity as identity

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec


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
    batch: generation.runtime.TerminalBatchEvidence,
    case: generation.runtime.TerminalCaseEvidence,
    *,
    task: TaskSpec,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Interpret one admitted canonical case as a steady-flow TaskSpec view."""
    if case.case_id not in {item.case_id for item in batch.cases} or batch.case(case.case_id) != case:
        msg = f"Case {case.case_id!r} is not bound to terminal batch {batch.batch_id!r}."
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
    if task.id != "steady_flow":
        msg = f"Generated airflow views support only steady_flow, got {task.id!r}."
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
    iterator = tqdm(selected, desc=f"Loading {batch_name}", unit="case", disable=not show_progress)
    for case in iterator:
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
    }
