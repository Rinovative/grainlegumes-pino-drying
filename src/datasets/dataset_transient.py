"""
===============================================================================
dataset_transient.py
===============================================================================
Build and lazily load physical-unit transient-drying transition packages.
Responsibilities:
  - Validate canonical case time, termination, channel, and source identities
  - Publish portable case-level membership and deterministic transition indexes
  - Slice only required HDF5 state, static, boundary, and scalar values at runtime
Design principles:
  - Absolute HDF5 states remain canonical and delta targets are derived on access
  - Every transition inherits one immutable case-level membership
  - Read-only HDF5 handles are process-local, lazy, and explicitly LRU-bounded
This module does NOT:
  - Register a transient task, fit normalization, train a model, or roll out
  - Duplicate complete cases, static grids, or scalar fields per transition
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src import common
from src.generation import generation_profiles as profiles
from src.generation import generation_storage as storage

from .dataset_transient_contract import TRANSIENT_STEP_CONTRACT
from .dataset_views import ID_MEMBERSHIPS, PACKAGE_REGIMES

TRANSIENT_INDEX_SCHEMA_KIND: Final = "vp2_transient_transition_index"
TRANSIENT_INDEX_SCHEMA_VERSION: Final = 2
_TIME_TOLERANCE: Final = 1e-12


class TransientDataContractError(ValueError):
    """Report one actionable transient package or runtime contract violation."""


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


@dataclass(frozen=True, slots=True)
class TransientSourceCase:
    """Bind one admitted canonical source case to package-owned metadata."""

    path: Path
    package_case_id: str
    source_batch_id: str
    membership: str
    evaluation_regime: str
    expected_sha256: str
    expected_case_input_id: str
    expected_simulation_case_id: str
    material_family: str
    ood_group: str | None
    ood_parameters: tuple[str, ...]
    ood_evidence: dict[str, Any]


TransientTransform = Callable[[TransientItem], TransientItem]


def _json_string_list(value: Any, *, label: str) -> list[str]:
    """Decode one HDF5 JSON string-list attribute."""
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
    """Return one required non-empty HDF5 text attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        message = f"HDF5 attribute {label!r} must be non-empty text."
        raise TypeError(message)
    return value


def _hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required HDF5 dataset with an exact contextual error."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        message = f"Transient HDF5 entry {name!r} must be a dataset: {handle.filename}."
        raise TransientDataContractError(message)
    return value


def transient_contract_payload() -> dict[str, Any]:
    """Return the exact persisted transient tensor names, units, and step."""
    contract = TRANSIENT_STEP_CONTRACT
    return {
        "state": [{"name": field.name, "unit": field.unit} for field in contract.dynamic_state],
        "static": [{"name": field.name, "unit": field.unit} for field in contract.static_spatial_conditioning],
        "boundary": [{"name": field.name, "unit": field.unit} for field in contract.step_boundary_conditioning],
        "scalars": [{"name": field.name, "unit": field.unit} for field in contract.scalar_conditioning],
        "target": [{"name": field.name, "unit": field.unit} for field in contract.target_increments],
        "dt": {"value": contract.time_step, "unit": contract.time_unit},
        "storage": contract.canonical_storage_representation,
        "target_derivation": contract.target_derivation_stage,
        "material_family_usage": contract.material_family_usage,
    }


def _regular_time_contract(time: np.ndarray, *, path: Path) -> tuple[int, float | None]:
    """Return regular-prefix length and an optional final irregular diagnostic."""
    if time.ndim != 1 or time.size < 1 or not np.isfinite(time).all():
        message = f"Transient time must be one non-empty finite sequence: {path}."
        raise TransientDataContractError(message)
    if np.any(np.diff(time) <= 0.0):
        message = f"Transient time must be strictly increasing: {path}."
        raise TransientDataContractError(message)
    regular_count = 0
    irregular_stop: float | None = None
    for index, raw_value in enumerate(time):
        value = float(raw_value)
        expected = index * TRANSIENT_STEP_CONTRACT.time_step
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=_TIME_TOLERANCE):
            regular_count += 1
            continue
        previous = (index - 1) * TRANSIENT_STEP_CONTRACT.time_step
        if index == time.size - 1 and index > 0 and previous < value < expected:
            irregular_stop = value
            break
        message = (
            f"Transient states must form a regular 0..N one-hour prefix with at most one final "
            f"irregular diagnostic state; index={index}, time={value}, source={path}."
        )
        raise TransientDataContractError(message)
    return regular_count, irregular_stop


def _status_evidence(path: Path, time: np.ndarray, regular_count: int) -> dict[str, Any]:
    """Validate an adjacent canonical status sidecar when it is available."""
    status_path = path.with_name("status.json")
    if not status_path.exists():
        return {
            "status_sha256": None,
            "t_stop_exact": None,
            "t_last_regular": float(time[regular_count - 1]),
        }
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Transient status sidecar is unreadable: {status_path}."
        raise TransientDataContractError(message) from error
    if not isinstance(status, dict) or status.get("schema_kind") != "simulation_case_status" or status.get("schema_version") != 1:
        message = f"Transient status sidecar schema is invalid: {status_path}."
        raise TransientDataContractError(message)
    last_regular = float(time[regular_count - 1])
    if (
        status.get("solver_success") is not True
        or status.get("contains_nan_or_inf") is not False
        or status.get("field_shape_valid") is not True
        or status.get("schedule_valid") is not True
        or status.get("n_regular_states") != regular_count
        or not math.isclose(float(status.get("t_last_regular", math.nan)), last_regular, rel_tol=0.0, abs_tol=_TIME_TOLERANCE)
    ):
        message = f"Transient status sidecar disagrees with its regular HDF5 sequence: {status_path}."
        raise TransientDataContractError(message)
    stop = float(status.get("t_stop_exact", math.nan))
    if not math.isfinite(stop) or stop < last_regular - _TIME_TOLERANCE:
        message = f"Transient exact stop precedes the last regular state: {status_path}."
        raise TransientDataContractError(message)
    return {
        "status_sha256": common.serialization.file_sha256(status_path),
        "t_stop_exact": stop,
        "t_last_regular": last_regular,
    }


def _safe_relative_source(path: Path, source_root: Path) -> str:
    """Return one portable source locator below the authoritative storage root."""
    resolved = path.expanduser().resolve()
    root = source_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        message = f"Transient source {resolved} is outside storage root {root}."
        raise TransientDataContractError(message) from error
    if resolved.is_symlink() or not resolved.is_file():
        message = f"Transient source is missing or unsafe: {resolved}."
        raise TransientDataContractError(message)
    return relative.as_posix()


def _validate_source_metadata(source: TransientSourceCase, handle: h5py.File) -> str:
    """Require source-bound identities and return its simulation profile."""
    profile = _text_attribute(handle.attrs.get("simulation_profile"), label="simulation_profile")
    observed = {
        "case_input_id": _text_attribute(handle.attrs.get("case_input_id"), label="case_input_id"),
        "simulation_case_id": _text_attribute(handle.attrs.get("simulation_case_id"), label="simulation_case_id"),
        "material_family": _text_attribute(handle.attrs.get("material_family"), label="material_family"),
    }
    expected = {
        "case_input_id": source.expected_case_input_id,
        "simulation_case_id": source.expected_simulation_case_id,
        "material_family": source.material_family,
    }
    if profile != profiles.TRANSIENT_DRYING_PROFILE or observed != expected:
        message = f"Transient source identity disagrees with package admission: {source.path}."
        raise TransientDataContractError(message)
    return profile


def _case_record(
    source: TransientSourceCase,
    *,
    source_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one source and return compact case and transition records."""
    path = source.path.expanduser().resolve()
    if source.evaluation_regime not in PACKAGE_REGIMES:
        message = f"Unsupported transient evaluation regime: {source.evaluation_regime!r}."
        raise TransientDataContractError(message)
    valid_memberships = ID_MEMBERSHIPS if source.evaluation_regime == "id" else (source.evaluation_regime,)
    if source.membership not in valid_memberships:
        message = f"Membership {source.membership!r} is invalid for {source.evaluation_regime!r}."
        raise TransientDataContractError(message)
    observed_sha256 = common.serialization.file_sha256(path)
    if observed_sha256 != source.expected_sha256:
        message = f"Transient source content differs from package admission: {path}."
        raise TransientDataContractError(message)
    storage.validate_case_hdf5(path, expected_profile=profiles.TRANSIENT_DRYING_PROFILE)
    with h5py.File(path, "r") as handle:
        profile = _validate_source_metadata(source, handle)
        time = np.asarray(_hdf5_dataset(handle, "time"), dtype=np.float64)
        regular_count, irregular_stop = _regular_time_contract(time, path=path)
        transient_dataset = _hdf5_dataset(handle, "transient/fields")
        static_dataset = _hdf5_dataset(handle, "static/fields")
        schedule_dataset = _hdf5_dataset(handle, "schedule/values")
        scalar_dataset = _hdf5_dataset(handle, "scalar/values")
        transient_names = _json_string_list(transient_dataset.attrs["field_names"], label="transient.field_names")
        static_names = _json_string_list(static_dataset.attrs["field_names"], label="static.field_names")
        schedule_names = _json_string_list(schedule_dataset.attrs["field_names"], label="schedule.field_names")
        scalar_names = _json_string_list(scalar_dataset.attrs["field_names"], label="scalar.field_names")
        if transient_names != list(profiles.TRANSIENT_FIELD_NAMES):
            message = f"Transient fields are not canonical: {path}."
            raise TransientDataContractError(message)
        required_static = {field.name for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning}.difference({"x", "y"})
        if not required_static.issubset(static_names):
            message = f"Transient static conditioning is incomplete: {path}."
            raise TransientDataContractError(message)
        if schedule_names != list(profiles.SCHEDULE_FIELDS) or scalar_names != list(profiles.SCALAR_INPUT_FIELDS):
            message = f"Transient boundary or scalar conditioning is not canonical: {path}."
            raise TransientDataContractError(message)
        schedule_time = np.asarray(schedule_dataset[:, schedule_names.index("t")], dtype=np.float64)
        samples: list[dict[str, Any]] = []
        for time_index in range(max(regular_count - 1, 0)):
            t_n = float(time[time_index])
            t_np1 = float(time[time_index + 1])
            current = np.flatnonzero(np.isclose(schedule_time, t_n, rtol=0.0, atol=_TIME_TOLERANCE))
            following = np.flatnonzero(np.isclose(schedule_time, t_np1, rtol=0.0, atol=_TIME_TOLERANCE))
            if current.size != 1 or following.size != 1:
                message = f"Schedule lacks exact endpoints for transition {t_n} -> {t_np1} h: {path}."
                raise TransientDataContractError(message)
            samples.append(
                {
                    "sample_id": f"{source.package_case_id}__step_{time_index:04d}",
                    "time_index": time_index,
                    "t_n": t_n,
                    "t_np1": t_np1,
                    "schedule_index_n": int(current[0]),
                    "schedule_index_np1": int(following[0]),
                }
            )
    if not samples:
        message = f"Transient source has no complete one-hour transition: {path}."
        raise TransientDataContractError(message)
    status = _status_evidence(path, time, regular_count)
    record = {
        "package_case_id": source.package_case_id,
        "source_relative_path": _safe_relative_source(path, source_root),
        "source_hdf5_sha256": observed_sha256,
        "source_status_sha256": status["status_sha256"],
        "case_input_id": source.expected_case_input_id,
        "simulation_case_id": source.expected_simulation_case_id,
        "source_batch_id": source.source_batch_id,
        "source_simulation_profile": profile,
        "material_family": source.material_family,
        "evaluation_regime": source.evaluation_regime,
        "membership": source.membership,
        "ood_group": source.ood_group,
        "ood_parameters": list(source.ood_parameters),
        "ood_evidence": deepcopy(source.ood_evidence),
        "sequence_length": regular_count,
        "stored_state_count": int(time.size),
        "transition_count": len(samples),
        "t_stop_exact": status["t_stop_exact"],
        "irregular_stop_time": irregular_stop,
    }
    return record, samples


def _index_digest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return transition identity without operational source locator paths."""
    result = {key: deepcopy(value) for key, value in payload.items() if key != "index_digest"}
    cases = result.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                case.pop("source_relative_path", None)
    return result


def build_transient_index(
    sources: Sequence[TransientSourceCase],
    destination: Path | str | None,
    *,
    dataset_name: str,
    dataset_id: str,
    evaluation_regime: str,
    contract_digest: str,
    source_root: Path | str,
) -> dict[str, Any]:
    """Build one deterministic compact transition index from canonical cases."""
    if not sources:
        message = "Transient index construction requires at least one source case."
        raise TransientDataContractError(message)
    if evaluation_regime not in PACKAGE_REGIMES or not dataset_name or not dataset_id:
        message = "Transient dataset name, ID, or evaluation regime is invalid."
        raise TransientDataContractError(message)
    root = Path(source_root)
    cases: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    simulation_ids: set[str] = set()
    sample_ids: set[str] = set()
    for case_index, source in enumerate(sources):
        record, case_samples = _case_record(source, source_root=root)
        simulation_id = str(record["simulation_case_id"])
        if simulation_id in simulation_ids:
            message = f"Transient source simulation identity is duplicated: {simulation_id}."
            raise TransientDataContractError(message)
        simulation_ids.add(simulation_id)
        cases.append(record)
        for sample in case_samples:
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                message = f"Transient sample identity is duplicated: {sample_id}."
                raise TransientDataContractError(message)
            sample_ids.add(sample_id)
            samples.append({"case_index": case_index, **sample})
    payload: dict[str, Any] = {
        "schema_kind": TRANSIENT_INDEX_SCHEMA_KIND,
        "schema_version": TRANSIENT_INDEX_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_view": "transient_drying",
        "evaluation_regime": evaluation_regime,
        "contract_digest": contract_digest,
        "contract": transient_contract_payload(),
        "source_locator_root": "storage_root",
        "cases": cases,
        "samples": samples,
        "source_case_count": len(cases),
        "sample_count": len(samples),
        "transition_count": len(samples),
    }
    payload["index_digest"] = common.serialization.canonical_json_sha256(_index_digest_payload(payload))
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


def _load_index(path: Path) -> dict[str, Any]:
    """Load and strictly validate one current transition-index payload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read transient dataset index: {path}."
        raise TransientDataContractError(message) from error
    required = {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "dataset_id",
        "dataset_view",
        "evaluation_regime",
        "contract_digest",
        "contract",
        "source_locator_root",
        "cases",
        "samples",
        "source_case_count",
        "sample_count",
        "transition_count",
        "index_digest",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        message = f"Transient dataset index keys do not match the current schema: {path}."
        raise TransientDataContractError(message)
    if (
        payload["schema_kind"] != TRANSIENT_INDEX_SCHEMA_KIND
        or payload["schema_version"] != TRANSIENT_INDEX_SCHEMA_VERSION
        or payload["dataset_view"] != "transient_drying"
        or payload["evaluation_regime"] not in PACKAGE_REGIMES
        or payload["contract"] != transient_contract_payload()
        or payload["source_locator_root"] != "storage_root"
        or not isinstance(payload["cases"], list)
        or not isinstance(payload["samples"], list)
        or payload["source_case_count"] != len(payload["cases"])
        or payload["sample_count"] != len(payload["samples"])
        or payload["transition_count"] != len(payload["samples"])
        or payload["index_digest"] != common.serialization.canonical_json_sha256(_index_digest_payload(payload))
    ):
        message = f"Transient dataset index contract or identity is invalid: {path}."
        raise TransientDataContractError(message)
    case_count = len(payload["cases"])
    sample_ids: set[str] = set()
    for sample in payload["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("case_index"), int) or not 0 <= sample["case_index"] < case_count:
            message = f"Transient sample references an invalid source case: {path}."
            raise TransientDataContractError(message)
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            message = f"Transient sample identities are invalid or duplicated: {path}."
            raise TransientDataContractError(message)
        sample_ids.add(sample_id)
    return payload


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
        self.payload = _load_index(self.index_path)
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
            raise TransientDataContractError(message)
        path = (self.source_root / relative).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as error:
            message = f"Transient source escapes storage root: {relative!r}."
            raise TransientDataContractError(message) from error
        if not path.is_file() or path.is_symlink():
            message = f"Transient source is missing or unsafe: {path}."
            raise FileNotFoundError(message)
        if common.serialization.file_sha256(path) != case.get("source_hdf5_sha256"):
            message = f"Transient source changed after package publication: {path}."
            raise TransientDataContractError(message)
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
            raise TransientDataContractError(message)
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
            transient_dataset = _hdf5_dataset(handle, "transient/fields")
            static_dataset = _hdf5_dataset(handle, "static/fields")
            schedule_dataset = _hdf5_dataset(handle, "schedule/values")
            scalar_dataset = _hdf5_dataset(handle, "scalar/values")
            transient_names = _json_string_list(transient_dataset.attrs["field_names"], label="transient.field_names")
            static_names = _json_string_list(static_dataset.attrs["field_names"], label="static.field_names")
            schedule_names = _json_string_list(schedule_dataset.attrs["field_names"], label="schedule.field_names")
            scalar_names = _json_string_list(scalar_dataset.attrs["field_names"], label="scalar.field_names")
            time_index = int(sample["time_index"])
            dynamic_indices = [transient_names.index(field.name) for field in TRANSIENT_STEP_CONTRACT.dynamic_state]
            state_array = np.asarray(transient_dataset[time_index, dynamic_indices, :, :], dtype=np.float32)
            next_array = np.asarray(transient_dataset[time_index + 1, dynamic_indices, :, :], dtype=np.float32)
            x_axis = np.asarray(_hdf5_dataset(handle, "coords/x")[:], dtype=np.float32)
            y_axis = np.asarray(_hdf5_dataset(handle, "coords/y")[:], dtype=np.float32)
            x_grid, y_grid = np.meshgrid(x_axis, y_axis)
            static_values: dict[str, np.ndarray] = {"x": x_grid, "y": y_grid}
            for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning:
                if field.name not in static_values:
                    static_values[field.name] = np.asarray(static_dataset[static_names.index(field.name), :, :], dtype=np.float32)
            static_array = np.stack(
                [static_values[field.name] for field in TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
                axis=0,
            )
            schedule_n = int(sample["schedule_index_n"])
            schedule_np1 = int(sample["schedule_index_np1"])
            scalar_lookup = {
                field.name: float(scalar_dataset[scalar_names.index(field.name)]) for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning
            }
            scalar_lookup["T_amb"] = float(scalar_dataset[scalar_names.index("T_amb")])
            boundary_values = {
                "T_in_t_n": float(schedule_dataset[schedule_n, schedule_names.index("T_in")]),
                "T_in_t_np1": float(schedule_dataset[schedule_np1, schedule_names.index("T_in")]),
                "phi_in_t_n": float(schedule_dataset[schedule_n, schedule_names.index("phi_in")]),
                "phi_in_t_np1": float(schedule_dataset[schedule_np1, schedule_names.index("phi_in")]),
                "T_amb": scalar_lookup["T_amb"],
            }
            boundary_array = np.asarray(
                [boundary_values[field.name] for field in TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
                dtype=np.float32,
            )
            scalar_array = np.asarray(
                [scalar_lookup[field.name] for field in TRANSIENT_STEP_CONTRACT.scalar_conditioning],
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
            "dt": torch.tensor(TRANSIENT_STEP_CONTRACT.time_step, dtype=torch.float32),
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
