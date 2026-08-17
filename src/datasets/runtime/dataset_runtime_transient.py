"""
===============================================================================
dataset_runtime_transient.py
===============================================================================
Lazily materialize physical-unit transient samples from admitted indexes.
Responsibilities:
  - Resolve and revalidate immutable source HDF5 and temporal evidence
  - Materialize explicit one-step transitions or deterministic rollout windows
  - Slice required state, static, boundary, scalar, and regular-time values
  - Own one process-local read-only handle and bounded LRU-cache implementation
Design principles:
  - One admitted trajectory index drives every runtime sampling mode
  - Delta targets are derived from canonical absolute endpoint states
  - Worker serialization never carries open HDF5 resources
This module does NOT:
  - Build or publish transient indexes
  - Normalize time, train models, or execute autoregressive model rollouts
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src import common
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.packages import dataset_packages_trajectory as trajectory


class TransientTime(TypedDict):
    """Expose authoritative HDF5 regular-time coordinates as tensors."""

    t_n: torch.Tensor
    t_n_plus_1: torch.Tensor
    dt: torch.Tensor


class TransientMetadata(TypedDict):
    """Describe one indexed physical sample without duplicating time values."""

    dataset_id: str
    dataset_name: str
    sample_id: str
    sample_mode: transient_contract.TransientSampleMode
    rollout_length: int
    simulation_case_id: str
    case_input_id: str
    source_batch_id: str
    source_simulation_profile: str
    material_family: str
    evaluation_regime: str
    split: str
    time_index_n: int
    time_index_n_plus_1: int
    sequence_length: int
    stored_state_count: int
    has_exact_stop_state: bool
    t_stop_exact: float


class TransientItem(TypedDict):
    """Expose one unnormalized physical transient sample for default collation."""

    state: torch.Tensor
    static: torch.Tensor
    boundary: torch.Tensor
    scalars: torch.Tensor
    time: TransientTime
    target: torch.Tensor
    metadata: TransientMetadata


TransientTransform = Callable[[TransientItem], TransientItem]


@dataclass(frozen=True, slots=True)
class _ItemReference:
    """Bind one runtime item to consecutive transition-index positions."""

    case_index: int
    sample_positions: tuple[int, ...]


class TransientPhysicalDataset(Dataset[TransientItem]):
    """Lazily expose explicit transient samples in unnormalized physical units."""

    def __init__(
        self,
        index_path: Path | str,
        *,
        sampling: transient_contract.TransientSamplingSpec,
        source_root: Path | str | None = None,
        hdf5_cache_size: int = 0,
        sample_indices: Sequence[int] | None = None,
        transform: TransientTransform | None = None,
    ) -> None:
        """Validate index/source identities without retaining open HDF5 handles."""
        if not isinstance(sampling, transient_contract.TransientSamplingSpec):
            message = "sampling must be one validated TransientSamplingSpec."
            raise TypeError(message)
        if isinstance(hdf5_cache_size, bool) or not isinstance(hdf5_cache_size, int) or hdf5_cache_size < 0:
            message = "hdf5_cache_size must be a non-negative integer."
            raise ValueError(message)
        self.index_path = Path(index_path).expanduser().resolve()
        self.source_root = common.paths.get_storage_root(storage_root=source_root).expanduser().resolve()
        self.payload = trajectory.load_transient_index(self.index_path)
        self.sampling = sampling
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
                or selected != tuple(sorted(selected))
            ):
                message = "Transient sample_indices must be unique ordered valid integer positions."
                raise ValueError(message)
        self.sample_indices = selected
        self._item_references = self._build_item_references()
        self._process_id: int | None = None
        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self._source_paths = tuple(self._validate_case_source(case) for case in self.payload["cases"])
        self._source_stats = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in self._source_paths)

    @property
    def dataset_id(self) -> str:
        """Return the immutable package dataset identifier."""
        return str(self.payload["dataset_id"])

    @property
    def configured_regular_horizon(self) -> float:
        """Return the Generation-configured temporal normalization horizon."""
        return float(self.payload["configured_regular_horizon"]["value"])

    def _build_item_references(self) -> tuple[_ItemReference, ...]:
        """Resolve deterministic item references from one shared transition index."""
        if not self.sample_indices:
            message = "Transient runtime selection contains no indexed transitions."
            raise ValueError(message)
        if self.sampling.mode == "one_step_transition":
            return tuple(
                _ItemReference(
                    case_index=int(self.payload["samples"][position]["case_index"]),
                    sample_positions=(position,),
                )
                for position in self.sample_indices
            )
        rollout_length = self.sampling.rollout_length
        window_stride = self.sampling.window_stride
        window_offset = self.sampling.window_offset
        if rollout_length is None or window_stride is None or window_offset is None:
            message = "Validated rollout sampling is missing required window values."
            raise RuntimeError(message)
        grouped: dict[int, list[int]] = {}
        for position in self.sample_indices:
            case_index = int(self.payload["samples"][position]["case_index"])
            grouped.setdefault(case_index, []).append(position)
        references: list[_ItemReference] = []
        for case_index, positions in grouped.items():
            final_start = len(positions) - rollout_length
            for start in range(window_offset, final_start + 1, window_stride):
                window = tuple(positions[start : start + rollout_length])
                samples = [self.payload["samples"][position] for position in window]
                if any(
                    current["time_index_n_plus_1"] != following["time_index_n"] or current["schedule_index_n_plus_1"] != following["schedule_index_n"]
                    for current, following in pairwise(samples)
                ):
                    message = "Transient rollout windows require consecutive regular transitions from one case."
                    raise trajectory.TransientDataContractError(message)
                references.append(_ItemReference(case_index=case_index, sample_positions=window))
        if not references:
            message = (
                f"Transient selection has no complete rollout window of length {rollout_length} at offset {window_offset} and stride {window_stride}."
            )
            raise ValueError(message)
        return tuple(references)

    def _validate_case_source(self, case: Mapping[str, Any]) -> Path:
        """Resolve one safe source and validate its hash and configured horizon."""
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
        with h5py.File(path, "r") as handle:
            time_dataset = trajectory.require_hdf5_dataset(handle, "time")
            time = np.asarray(time_dataset, dtype=np.float64)
            tolerance = float(time_dataset.attrs.get("classification_atol", math.nan))
            observed_horizon = trajectory.configured_regular_horizon(
                handle,
                time,
                path=path,
                tolerance=tolerance,
            )
        if time.shape != (case.get("sequence_length"),) or not math.isclose(
            observed_horizon,
            self.configured_regular_horizon,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            message = f"Transient source temporal identity disagrees with its index: {path}."
            raise trajectory.TransientDataContractError(message)
        return path

    def __len__(self) -> int:
        """Return the deterministic item count for the explicit sampling mode."""
        return len(self._item_references)

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
    def _finite_tensor(array: Any, *, label: str, path: Path) -> torch.Tensor:
        """Convert one finite float32 value or array into a contiguous tensor."""
        converted = np.asarray(array, dtype=np.float32)
        if not np.isfinite(converted).all():
            message = f"Transient {label} contains non-finite selected values: {path}."
            raise trajectory.TransientDataContractError(message)
        return torch.from_numpy(np.ascontiguousarray(converted))

    def __getitem__(self, index: int) -> TransientItem:
        """Read all HDF5 slices for one one-step sample or rollout window."""
        reference = self._item_references[index]
        samples = [self.payload["samples"][position] for position in reference.sample_positions]
        case_index = reference.case_index
        case = self.payload["cases"][case_index]
        path = self._validate_runtime_source(case_index)
        handle, close_after = self._open_handle(path)
        try:
            transient_dataset = trajectory.require_hdf5_dataset(handle, "transient/fields")
            static_dataset = trajectory.require_hdf5_dataset(handle, "static/fields")
            schedule_dataset = trajectory.require_hdf5_dataset(handle, "schedule/values")
            scalar_dataset = trajectory.require_hdf5_dataset(handle, "scalar/values")
            time_dataset = trajectory.require_hdf5_dataset(handle, "time")
            transient_names = trajectory.decode_json_string_list(transient_dataset.attrs["field_names"], label="transient.field_names")
            static_names = trajectory.decode_json_string_list(static_dataset.attrs["field_names"], label="static.field_names")
            schedule_names = trajectory.decode_json_string_list(schedule_dataset.attrs["field_names"], label="schedule.field_names")
            scalar_names = trajectory.decode_json_string_list(scalar_dataset.attrs["field_names"], label="scalar.field_names")
            state_indices = [int(samples[0]["time_index_n"]), *[int(sample["time_index_n_plus_1"]) for sample in samples]]
            time_values = np.asarray(time_dataset[state_indices], dtype=np.float64)
            tolerance = float(time_dataset.attrs.get("classification_atol", math.nan))
            time_unit = time_dataset.attrs.get("unit")
            if isinstance(time_unit, bytes):
                time_unit = time_unit.decode("utf-8")
            dt_values = np.diff(time_values)
            if (
                time_values.shape != (len(samples) + 1,)
                or not np.isfinite(time_values).all()
                or not math.isfinite(tolerance)
                or tolerance <= 0.0
                or time_unit != transient_contract.TRANSIENT_STEP_CONTRACT.time_unit
                or not np.allclose(
                    dt_values,
                    transient_contract.TRANSIENT_STEP_CONTRACT.time_step,
                    rtol=0.0,
                    atol=tolerance,
                )
            ):
                message = f"Transient indexed time coordinates are invalid: {path}."
                raise trajectory.TransientDataContractError(message)
            dynamic_indices = [transient_names.index(field.name) for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state]
            states_array = np.stack(
                [np.asarray(transient_dataset[time_index, dynamic_indices, :, :], dtype=np.float32) for time_index in state_indices],
                axis=0,
            )
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
            scalar_lookup = {
                field.name: float(scalar_dataset[scalar_names.index(field.name)])
                for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning
            }
            scalar_lookup["T_amb"] = float(scalar_dataset[scalar_names.index("T_amb")])
            raw_handoff = schedule_dataset.attrs.get("boundary_handoff")
            if isinstance(raw_handoff, bytes):
                raw_handoff = raw_handoff.decode("utf-8")
            if not isinstance(raw_handoff, str):
                message = f"Transient source lacks boundary-handoff metadata: {path}."
                raise trajectory.TransientDataContractError(message)
            try:
                boundary_handoff = json.loads(raw_handoff)
            except json.JSONDecodeError as error:
                message = f"Transient source boundary-handoff metadata is invalid: {path}."
                raise trajectory.TransientDataContractError(message) from error
            startup = boundary_handoff.get("startup_ramp") if isinstance(boundary_handoff, dict) else None
            if not isinstance(startup, dict) or not isinstance(startup.get("enabled"), bool):
                message = f"Transient source startup-handoff metadata is invalid: {path}."
                raise trajectory.TransientDataContractError(message)
            startup_time = startup.get("duration_h")
            if isinstance(startup_time, bool) or not isinstance(startup_time, (int, float)) or not math.isfinite(float(startup_time)):
                message = f"Transient source startup duration is invalid: {path}."
                raise trajectory.TransientDataContractError(message)
            startup_time_value = float(startup_time)
            boundary_rows: list[list[float]] = []
            for sample in samples:
                schedule_n = int(sample["schedule_index_n"])
                schedule_n_plus_1 = int(sample["schedule_index_n_plus_1"])
                raw_support_index = sample["schedule_support_index"]
                schedule_time_n = float(schedule_dataset[schedule_n, schedule_names.index("t")])
                schedule_time_n_plus_1 = float(schedule_dataset[schedule_n_plus_1, schedule_names.index("t")])
                support_required = bool(startup["enabled"] and schedule_time_n < startup_time_value < schedule_time_n_plus_1)
                if support_required != (raw_support_index is not None):
                    message = f"Transient index startup support disagrees with source handoff: {path}."
                    raise trajectory.TransientDataContractError(message)
                support_index = schedule_n if raw_support_index is None else int(raw_support_index)
                if raw_support_index is not None and not math.isclose(
                    float(schedule_dataset[support_index, schedule_names.index("t")]),
                    startup_time_value,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    message = f"Transient index startup support points to the wrong source row: {path}."
                    raise trajectory.TransientDataContractError(message)
                support_time_offset = (
                    0.0
                    if raw_support_index is None
                    else float(schedule_dataset[support_index, schedule_names.index("t")])
                    - float(schedule_dataset[schedule_n, schedule_names.index("t")])
                )
                boundary_values = {
                    "T_in_bc_t_n": float(schedule_dataset[schedule_n, schedule_names.index("T_in_bc")]),
                    "T_in_bc_t_n_plus_1": float(schedule_dataset[schedule_n_plus_1, schedule_names.index("T_in_bc")]),
                    "omega_in_bc_t_n": float(schedule_dataset[schedule_n, schedule_names.index("omega_in_bc")]),
                    "omega_in_bc_t_n_plus_1": float(schedule_dataset[schedule_n_plus_1, schedule_names.index("omega_in_bc")]),
                    "T_amb": scalar_lookup["T_amb"],
                    "startup_support_time_offset": support_time_offset,
                    "T_in_bc_startup_support": float(schedule_dataset[support_index, schedule_names.index("T_in_bc")]),
                    "omega_in_bc_startup_support": float(schedule_dataset[support_index, schedule_names.index("omega_in_bc")]),
                    "startup_support_present": float(raw_support_index is not None),
                }
                boundary_rows.append([boundary_values[field.name] for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning])
            boundary_array = np.asarray(boundary_rows, dtype=np.float32)
            scalar_array = np.asarray(
                [scalar_lookup[field.name] for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning],
                dtype=np.float32,
            )
        finally:
            if close_after:
                handle.close()
        states = self._finite_tensor(states_array, label="state sequence", path=path)
        target_sequence = states[1:] - states[:-1]
        boundary = self._finite_tensor(boundary_array, label="boundary conditioning", path=path)
        t_n = self._finite_tensor(time_values[:-1], label="current time", path=path)
        t_n_plus_1 = self._finite_tensor(time_values[1:], label="next time", path=path)
        dt = self._finite_tensor(dt_values, label="time increment", path=path)
        one_step = self.sampling.mode == "one_step_transition"
        first_time_index = int(samples[0]["time_index_n"])
        last_time_index = int(samples[-1]["time_index_n_plus_1"])
        sample_id = str(samples[0]["sample_id"]) if one_step else f"{case['package_case_id']}__window_{first_time_index:04d}_{last_time_index:04d}"
        item: TransientItem = {
            "state": states[0],
            "static": self._finite_tensor(static_array, label="static conditioning", path=path),
            "boundary": boundary[0] if one_step else boundary,
            "scalars": self._finite_tensor(scalar_array, label="scalar conditioning", path=path),
            "time": {
                "t_n": t_n[0] if one_step else t_n,
                "t_n_plus_1": t_n_plus_1[0] if one_step else t_n_plus_1,
                "dt": dt[0] if one_step else dt,
            },
            "target": target_sequence[0] if one_step else target_sequence,
            "metadata": {
                "dataset_id": str(self.payload["dataset_id"]),
                "dataset_name": str(self.payload["dataset_name"]),
                "sample_id": sample_id,
                "sample_mode": self.sampling.mode,
                "rollout_length": len(samples),
                "simulation_case_id": str(case["simulation_case_id"]),
                "case_input_id": str(case["case_input_id"]),
                "source_batch_id": str(case["source_batch_id"]),
                "source_simulation_profile": str(case["source_simulation_profile"]),
                "material_family": str(case["material_family"]),
                "evaluation_regime": str(case["evaluation_regime"]),
                "split": str(case["membership"]),
                "time_index_n": first_time_index,
                "time_index_n_plus_1": last_time_index,
                "sequence_length": int(case["sequence_length"]),
                "stored_state_count": int(case["stored_state_count"]),
                "has_exact_stop_state": case["irregular_stop_time"] is not None,
                "t_stop_exact": float(case["t_stop_exact"]),
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
