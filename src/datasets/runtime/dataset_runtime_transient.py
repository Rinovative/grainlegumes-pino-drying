"""
===============================================================================
dataset_runtime_transient.py
===============================================================================
Lazily materialize physical-unit transient transitions from admitted indexes.
Responsibilities:
  - Resolve and revalidate immutable source HDF5 evidence
  - Slice required state, static, boundary, and scalar values on demand
  - Own process-local read-only handles and a bounded LRU cache
Design principles:
  - One admitted trajectory index drives every runtime selection
  - Delta targets are derived from canonical absolute endpoint states
  - Worker serialization never carries open HDF5 resources
This module does NOT:
  - Build or publish transient indexes
  - Fit normalization, train models, or perform rollouts
===============================================================================
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src import common
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.packages import dataset_packages_trajectory as trajectory


class TransientMetadata(TypedDict):
    """Describe one indexed physical transition without model conditioning."""

    dataset_id: str
    dataset_name: str
    sample_id: str
    simulation_case_id: str
    case_input_id: str
    source_batch_id: str
    source_simulation_profile: str
    material_family: str
    evaluation_regime: str
    split: str
    time_index: int
    t_n: float
    t_np1: float
    sequence_length: int


class TransientItem(TypedDict):
    """Expose one unnormalized physical-unit transition for default collation."""

    state: torch.Tensor
    static: torch.Tensor
    boundary: torch.Tensor
    scalars: torch.Tensor
    target: torch.Tensor
    dt: torch.Tensor
    metadata: TransientMetadata


TransientTransform = Callable[[TransientItem], TransientItem]


class TransientPhysicalDataset(Dataset[TransientItem]):
    """Lazily expose indexed one-hour transitions in unnormalized physical units."""

    def __init__(
        self,
        index_path: Path | str,
        *,
        source_root: Path | str | None = None,
        hdf5_cache_size: int = 0,
        sample_indices: Sequence[int] | None = None,
        transform: TransientTransform | None = None,
    ) -> None:
        """Validate index/source identities without opening persistent HDF5 handles."""
        if isinstance(hdf5_cache_size, bool) or not isinstance(hdf5_cache_size, int) or hdf5_cache_size < 0:
            message = "hdf5_cache_size must be a non-negative integer."
            raise ValueError(message)
        self.index_path = Path(index_path).expanduser().resolve()
        self.source_root = common.paths.get_storage_root(storage_root=source_root).expanduser().resolve()
        self.payload = trajectory.load_transient_index(self.index_path)
        self.hdf5_cache_size = hdf5_cache_size
        self.transform = transform
        if sample_indices is None:
            selected = tuple(range(len(self.payload["samples"])))
        else:
            selected = tuple(sample_indices)
            if (
                any(isinstance(index, bool) or not isinstance(index, int) for index in selected)
                or any(index < 0 or index >= len(self.payload["samples"]) for index in selected)
                or len(selected) != len(set(selected))
            ):
                message = "Transient sample_indices must be unique valid integer positions."
                raise ValueError(message)
        self.sample_indices = selected
        self._process_id: int | None = None
        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self._source_paths = tuple(self._validate_case_source(case) for case in self.payload["cases"])
        self._source_stats = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in self._source_paths)

    @property
    def dataset_id(self) -> str:
        """Return the immutable package dataset identifier."""
        return str(self.payload["dataset_id"])

    def _validate_case_source(self, case: Mapping[str, Any]) -> Path:
        """Resolve one safe relative locator and validate its published content hash."""
        relative = case.get("source_relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            message = f"Transient source locator is unsafe in {self.index_path}."
            raise trajectory.TransientDataContractError(message)
        path = (self.source_root / relative).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as error:
            message = f"Transient source escapes storage root: {relative!r}."
            raise trajectory.TransientDataContractError(message) from error
        if not path.is_file() or path.is_symlink():
            message = f"Transient source is missing or unsafe: {path}."
            raise FileNotFoundError(message)
        if common.serialization.file_sha256(path) != case.get("source_hdf5_sha256"):
            message = f"Transient source changed after package publication: {path}."
            raise trajectory.TransientDataContractError(message)
        return path

    def __len__(self) -> int:
        """Return the selected deterministic transition count."""
        return len(self.sample_indices)

    def _validate_runtime_source(self, case_index: int) -> Path:
        """Reject a source whose stat or hash changed after dataset admission."""
        path = self._source_paths[case_index]
        stat = path.stat()
        observed_stat = (stat.st_size, stat.st_mtime_ns)
        if observed_stat != self._source_stats[case_index]:
            expected_hash = self.payload["cases"][case_index]["source_hdf5_sha256"]
            if common.serialization.file_sha256(path) != expected_hash:
                message = f"Transient source changed after dataset admission: {path}."
                raise RuntimeError(message)
            self._source_stats = tuple(observed_stat if index == case_index else value for index, value in enumerate(self._source_stats))
        return path

    def _close_handles(self) -> None:
        """Close every process-local cached HDF5 handle."""
        for handle in self._handles.values():
            with suppress(OSError, RuntimeError):
                handle.close()
        self._handles.clear()

    def close(self) -> None:
        """Close bounded runtime resources owned by the current process."""
        self._close_handles()
        self._process_id = None

    def _ensure_process(self) -> None:
        """Discard inherited runtime state before opening a handle in this PID."""
        process_id = os.getpid()
        if self._process_id != process_id:
            self._close_handles()
            self._process_id = process_id

    def _open_handle(self, path: Path) -> tuple[h5py.File, bool]:
        """Return one lazy read-only handle and whether the caller must close it."""
        self._ensure_process()
        if self.hdf5_cache_size == 0:
            return h5py.File(path, "r"), True
        cached = self._handles.pop(path, None)
        if cached is not None:
            self._handles[path] = cached
            return cached, False
        handle = h5py.File(path, "r")
        self._handles[path] = handle
        while len(self._handles) > self.hdf5_cache_size:
            _evicted_path, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle, False

    @staticmethod
    def _finite_tensor(array: np.ndarray, *, label: str, path: Path) -> torch.Tensor:
        """Convert one finite float32 array into a contiguous tensor."""
        converted = np.asarray(array, dtype=np.float32)
        if not np.isfinite(converted).all():
            message = f"Transient {label} contains non-finite selected values: {path}."
            raise trajectory.TransientDataContractError(message)
        return torch.from_numpy(np.ascontiguousarray(converted))

    def __getitem__(self, index: int) -> TransientItem:
        """Read only the HDF5 slices required by one indexed transition."""
        payload_index = self.sample_indices[index]
        sample = self.payload["samples"][payload_index]
        case_index = int(sample["case_index"])
        case = self.payload["cases"][case_index]
        path = self._validate_runtime_source(case_index)
        handle, close_after = self._open_handle(path)
        try:
            transient_dataset = trajectory.require_hdf5_dataset(handle, "transient/fields")
            static_dataset = trajectory.require_hdf5_dataset(handle, "static/fields")
            schedule_dataset = trajectory.require_hdf5_dataset(handle, "schedule/values")
            scalar_dataset = trajectory.require_hdf5_dataset(handle, "scalar/values")
            transient_names = trajectory.decode_json_string_list(transient_dataset.attrs["field_names"], label="transient.field_names")
            static_names = trajectory.decode_json_string_list(static_dataset.attrs["field_names"], label="static.field_names")
            schedule_names = trajectory.decode_json_string_list(schedule_dataset.attrs["field_names"], label="schedule.field_names")
            scalar_names = trajectory.decode_json_string_list(scalar_dataset.attrs["field_names"], label="scalar.field_names")
            time_index = int(sample["time_index"])
            time_dataset = trajectory.require_hdf5_dataset(handle, "time")
            time = np.asarray(time_dataset, dtype=np.float64)
            tolerance = float(time_dataset.attrs.get("classification_atol", math.nan))
            next_matches = np.flatnonzero(np.isclose(time, float(sample["t_np1"]), rtol=0.0, atol=tolerance))
            if next_matches.size != 1:
                message = f"Transient source lacks indexed next state {sample['t_np1']!r}: {path}."
                raise trajectory.TransientDataContractError(message)
            next_time_index = int(next_matches[0])
            dynamic_indices = [transient_names.index(field.name) for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state]
            state_array = np.asarray(transient_dataset[time_index, dynamic_indices, :, :], dtype=np.float32)
            next_array = np.asarray(transient_dataset[next_time_index, dynamic_indices, :, :], dtype=np.float32)
            x_axis = np.asarray(trajectory.require_hdf5_dataset(handle, "coords/x")[:], dtype=np.float32)
            y_axis = np.asarray(trajectory.require_hdf5_dataset(handle, "coords/y")[:], dtype=np.float32)
            x_grid, y_grid = np.meshgrid(x_axis, y_axis)
            static_values: dict[str, np.ndarray] = {"x": x_grid, "y": y_grid}
            for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning:
                if field.name not in static_values:
                    static_values[field.name] = np.asarray(static_dataset[static_names.index(field.name), :, :], dtype=np.float32)
            static_array = np.stack(
                [static_values[field.name] for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
                axis=0,
            )
            schedule_n = int(sample["schedule_index_n"])
            schedule_np1 = int(sample["schedule_index_np1"])
            scalar_lookup = {
                field.name: float(scalar_dataset[scalar_names.index(field.name)])
                for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning
            }
            scalar_lookup["T_amb"] = float(scalar_dataset[scalar_names.index("T_amb")])
            boundary_values = {
                "T_in_bc_t_n": float(schedule_dataset[schedule_n, schedule_names.index("T_in_bc")]),
                "T_in_bc_t_np1": float(schedule_dataset[schedule_np1, schedule_names.index("T_in_bc")]),
                "phi_in_bc_t_n": float(schedule_dataset[schedule_n, schedule_names.index("phi_in_bc")]),
                "phi_in_bc_t_np1": float(schedule_dataset[schedule_np1, schedule_names.index("phi_in_bc")]),
                "T_amb": scalar_lookup["T_amb"],
            }
            boundary_array = np.asarray(
                [boundary_values[field.name] for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
                dtype=np.float32,
            )
            scalar_array = np.asarray(
                [scalar_lookup[field.name] for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning],
                dtype=np.float32,
            )
        finally:
            if close_after:
                handle.close()
        state = self._finite_tensor(state_array, label="state", path=path)
        next_state = self._finite_tensor(next_array, label="next state", path=path)
        item: TransientItem = {
            "state": state,
            "static": self._finite_tensor(static_array, label="static conditioning", path=path),
            "boundary": self._finite_tensor(boundary_array, label="boundary conditioning", path=path),
            "scalars": self._finite_tensor(scalar_array, label="scalar conditioning", path=path),
            "target": next_state - state,
            "dt": torch.tensor(transient_contract.TRANSIENT_STEP_CONTRACT.time_step, dtype=torch.float32),
            "metadata": {
                "dataset_id": str(self.payload["dataset_id"]),
                "dataset_name": str(self.payload["dataset_name"]),
                "sample_id": str(sample["sample_id"]),
                "simulation_case_id": str(case["simulation_case_id"]),
                "case_input_id": str(case["case_input_id"]),
                "source_batch_id": str(case["source_batch_id"]),
                "source_simulation_profile": str(case["source_simulation_profile"]),
                "material_family": str(case["material_family"]),
                "evaluation_regime": str(case["evaluation_regime"]),
                "split": str(case["membership"]),
                "time_index": time_index,
                "t_n": float(sample["t_n"]),
                "t_np1": float(sample["t_np1"]),
                "sequence_length": int(case["sequence_length"]),
            },
        }
        return item if self.transform is None else self.transform(item)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude open HDF5 handles from worker serialization."""
        state = dict(self.__dict__)
        state["_process_id"] = None
        state["_handles"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a worker-local dataset without runtime HDF5 resources."""
        self.__dict__.update(state)
        self._process_id = None
        self._handles = OrderedDict()

    def __del__(self) -> None:
        """Best-effort close of process-local read-only HDF5 handles."""
        with suppress(AttributeError, OSError, RuntimeError):
            self._close_handles()


def select_transient_sample_indices(
    payload: Mapping[str, Any],
    *,
    membership: str | None,
    ood_group: str | None,
) -> tuple[int, ...]:
    """Select transition positions from case membership and optional OOD group."""
    cases = payload.get("cases")
    samples = payload.get("samples")
    if not isinstance(cases, list) or not isinstance(samples, list):
        message = "Transient selector requires one validated index payload."
        raise TypeError(message)
    selected_cases = {
        index
        for index, case in enumerate(cases)
        if (membership is None or case.get("membership") == membership) and (ood_group is None or case.get("ood_group") == ood_group)
    }
    indices = tuple(index for index, sample in enumerate(samples) if sample.get("case_index") in selected_cases)
    if not indices:
        selector = {"membership": membership, "ood_group": ood_group}
        message = f"Transient package has no samples for selector {selector}."
        raise ValueError(message)
    return indices
