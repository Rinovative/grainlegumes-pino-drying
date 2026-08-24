"""
dataset_packages_transient_shards.py

Build and admit Dataset-bound derived PT shards for transient trajectories.
Responsibilities:
  - Pack complete canonical cases into soft-target immutable PyTorch shards
  - Bind derived payloads to Dataset, index, source, and GPU publication identity
  - Atomically publish and strictly validate shard receipts and tensor contents
Design principles:
  - Canonical HDF5 remains authoritative and sufficient for exact reconstruction
  - One case belongs to exactly one shard and exact-stop data remains diagnostic
  - Scientific Dataset identity is independent of derived packing policy
This module does NOT:
  - Change package manifests, Dataset IDs, split membership, or Training semantics
  - Delete, rewrite, or recover canonical Generation HDF5 artifacts
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import h5py
import torch

from src import common
from src.datasets.contracts import dataset_contracts_transient as transient_contract

from . import DEFAULT_TRANSIENT_PT_SHARD_BYTES
from . import dataset_packages_manifest as package_manifest
from . import dataset_packages_trajectory as trajectory

if TYPE_CHECKING:
    from collections.abc import Sequence

TRANSIENT_PT_SHARD_SCHEMA_KIND: Final = "transient_pt_shard_payload"
TRANSIENT_PT_SHARD_SCHEMA_VERSION: Final = 1
TRANSIENT_PT_RECEIPT_SCHEMA_KIND: Final = "transient_pt_shard_receipt"
TRANSIENT_PT_RECEIPT_SCHEMA_VERSION: Final = 1
TRANSIENT_PT_SHARD_DIRECTORY: Final = "transient_pt_shards"
TRANSIENT_PT_RECEIPT_FILENAME: Final = "receipt.json"
_MINIMUM_STATE_COUNT: Final = 2
_TRANSIENT_TENSOR_RANK: Final = 4
_STATIC_TENSOR_RANK: Final = 3
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_TERMINAL_PUBLICATION_IDENTITY_KEYS: Final = {
    "campaign_run_id",
    "campaign_id",
    "git_commit",
    "campaign_terminal_sha256",
    "transfer_inventory_sha256",
}
_COMPOSITE_PUBLICATION_IDENTITY_KEYS: Final = {
    "completion_id",
    "parent_run_id",
    "parent_partial_sha256",
    "completion_receipt_sha256",
    "combined_inventory_sha256",
}
_PUBLICATION_IDENTITY_KEY_SETS: Final = (
    _TERMINAL_PUBLICATION_IDENTITY_KEYS,
    _COMPOSITE_PUBLICATION_IDENTITY_KEYS,
)
ProgressCallback = Callable[[Mapping[str, Any]], None]
ValidationDepth = Literal["evidence", "full"]


class TransientShardContractError(ValueError):
    """Report one actionable derived transient-shard contract violation."""


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def transient_shard_directory(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the immutable derived-payload directory for one Dataset."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    return common.paths.get_dataset_packages_root(storage_root=storage_root) / logical_id / TRANSIENT_PT_SHARD_DIRECTORY


def transient_shard_receipt_path(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the readiness receipt path for one derived payload."""
    return transient_shard_directory(dataset_id, storage_root=storage_root) / TRANSIENT_PT_RECEIPT_FILENAME


def _validate_publication_identity(value: Any) -> dict[str, str]:
    """Validate stable canonical GPU publication binding fields."""
    if not isinstance(value, dict) or set(value) not in _PUBLICATION_IDENTITY_KEY_SETS:
        message = "Transient shard publication identity keys are invalid."
        raise TransientShardContractError(message)
    result: dict[str, str] = {}
    for key in sorted(value):
        item = value[key]
        if not isinstance(item, str) or not item:
            message = f"Transient shard GPU publication identity {key!r} is invalid."
            raise TransientShardContractError(message)
        if (key.endswith("_sha256") or key == "combined_inventory_sha256") and _SHA256_PATTERN.fullmatch(item) is None:
            message = f"Transient shard GPU publication digest {key!r} is invalid."
            raise TransientShardContractError(message)
        result[key] = item
    return result


def _package_context(
    dataset_id: str,
    *,
    storage_root: Path | str | None,
    validation_depth: ValidationDepth = "full",
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    """Return an admitted transient manifest, index, and manifest digest."""
    if validation_depth not in {"evidence", "full"}:
        message = f"Unsupported transient shard validation depth: {validation_depth!r}."
        raise ValueError(message)
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    manifest_loader = package_manifest.load_package_manifest if validation_depth == "full" else package_manifest.load_package_manifest_evidence
    manifest = manifest_loader(dataset_id, storage_root=storage)
    if manifest["dataset_view"] != "transient_drying":
        message = f"PT shards apply only to transient_drying packages: {dataset_id!r}."
        raise TransientShardContractError(message)
    manifest_path = common.paths.get_dataset_metadata_root(storage_root=storage) / dataset_id / "dataset_manifest.json"
    index_path = common.paths.get_dataset_packages_root(storage_root=storage) / dataset_id / str(manifest["payload_filename"])
    index = trajectory.load_transient_index(index_path)
    if (
        index["dataset_id"] != manifest["dataset_id"]
        or index["dataset_name"] != manifest["dataset_name"]
        or index["index_digest"] == ""
        or index["sample_count"] != manifest["sample_count"]
        or index["source_case_count"] != manifest["source_case_count"]
        or index["contract_digest"] != manifest["channel_contract_digest"]
    ):
        message = f"Transient package index does not bind its manifest: {index_path}."
        raise TransientShardContractError(message)
    return storage, manifest, index, common.serialization.file_sha256(manifest_path)


def _canonical_payload_stat_identity(
    storage: Path,
    manifest: Mapping[str, Any],
) -> tuple[int, dict[str, int]]:
    """Return the cheap immutable identity of one admitted package payload."""
    path = common.paths.get_dataset_packages_root(storage_root=storage) / str(manifest["dataset_id"]) / str(manifest["payload_filename"])
    return _file_stat_identity(path)


def _case_samples(index: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Group ordered transition records by their exact case index."""
    grouped: list[list[dict[str, Any]]] = [[] for _ in index["cases"]]
    for sample in index["samples"]:
        grouped[int(sample["case_index"])].append(dict(sample))
    if any(not values for values in grouped):
        message = "Transient shard source index contains a case without transitions."
        raise TransientShardContractError(message)
    return tuple(tuple(values) for values in grouped)


def _resolve_source(
    storage: Path,
    case: Mapping[str, Any],
) -> Path:
    """Resolve one canonical GPU HDF5 locator and verify its exact digest."""
    relative_value = case.get("source_relative_path")
    relative = Path(relative_value) if isinstance(relative_value, str) else None
    if relative is None or not relative_value or relative.is_absolute() or ".." in relative.parts:
        message = "Transient shard source locator is unsafe."
        raise TransientShardContractError(message)
    path = (storage / relative).resolve()
    if not path.is_relative_to(storage) or not path.is_file() or path.is_symlink():
        message = f"Canonical GPU transient source is missing or unsafe: {path}."
        raise FileNotFoundError(message)
    if common.serialization.file_sha256(path) != case.get("source_hdf5_sha256"):
        message = f"Canonical GPU transient source changed after Dataset publication: {path}."
        raise TransientShardContractError(message)
    return path


def _case_payload(
    *,
    path: Path,
    index: Mapping[str, Any],
    case_index: int,
    case: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize one complete canonical case as simple tensors and metadata."""
    with h5py.File(path, "r") as handle:
        arrays = trajectory.read_transient_case_arrays(
            handle,
            path,
            case,
            samples,
            expected_regular_horizon=float(index["configured_regular_horizon"]["value"]),
            complete_case=True,
        )
    return {
        "case_index": case_index,
        "case_record": dict(case),
        "samples": [dict(sample) for sample in samples],
        "field_order": {
            "dynamic": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state],
            "static": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
            "boundary": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
            "scalars": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning],
        },
        "states": torch.from_numpy(arrays.states),
        "static": torch.from_numpy(arrays.static),
        "boundary": torch.from_numpy(arrays.boundary),
        "scalars": torch.from_numpy(arrays.scalars),
        "state_time": torch.from_numpy(arrays.time),
        "valid_state_count": int(arrays.states.shape[0]),
        "valid_transition_count": len(samples),
        "exact_stop_time": arrays.exact_stop_time,
        "exact_stop_state": (None if arrays.exact_stop_state is None else torch.from_numpy(arrays.exact_stop_state)),
    }


def _tensor_bytes(value: Any) -> int:
    """Return exact tensor storage bytes in one simple shard payload tree."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _file_stat_identity(path: Path) -> tuple[int, dict[str, int]]:
    """Return cheap local evidence that changes on write or replacement."""
    observed = path.stat()
    return (
        observed.st_size,
        {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mtime_ns": observed.st_mtime_ns,
            "ctime_ns": observed.st_ctime_ns,
        },
    )


def _planned_case_tensor_bytes(
    path: Path,
    case: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> int:
    """Return exact output tensor bytes from bounded HDF5 shape metadata."""
    with h5py.File(path, "r") as handle:
        transient = trajectory.require_hdf5_dataset(handle, "transient/fields")
        static = trajectory.require_hdf5_dataset(handle, "static/fields")
        if transient.ndim != _TRANSIENT_TENSOR_RANK or static.ndim != _STATIC_TENSOR_RANK:
            message = f"Transient shard source tensor rank is invalid: {path}."
            raise TransientShardContractError(message)
        state_count = case.get("sequence_length")
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or state_count < _MINIMUM_STATE_COUNT
            or transient.shape[0] != state_count
            or transient.shape[2:] != static.shape[1:]
        ):
            message = f"Transient shard source shapes disagree with its index: {path}."
            raise TransientShardContractError(message)
        grid_values = int(transient.shape[2]) * int(transient.shape[3])
        if grid_values < 1:
            message = f"Transient shard source grid is empty: {path}."
            raise TransientShardContractError(message)
    dynamic_count = len(transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    static_count = len(transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning)
    boundary_count = len(transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning)
    scalar_count = len(transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning)
    float32_bytes = 4
    float64_bytes = 8
    total = state_count * dynamic_count * grid_values * float32_bytes
    total += static_count * grid_values * float32_bytes
    total += len(samples) * boundary_count * float32_bytes
    total += scalar_count * float32_bytes
    total += state_count * float64_bytes
    if case.get("irregular_stop_time") is not None:
        total += dynamic_count * grid_values * float32_bytes
    return total


def _plan_case_groups(
    case_sizes: Sequence[int],
    *,
    target_shard_bytes: int,
) -> tuple[tuple[int, ...], ...]:
    """Group ordered whole cases with the existing soft byte target."""
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_bytes = 0
    for case_index, case_bytes in enumerate(case_sizes):
        if current and current_bytes + case_bytes > target_shard_bytes:
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(case_index)
        current_bytes += case_bytes
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _derived_payload_id(
    *,
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    publication_identity: Mapping[str, str],
    target_shard_bytes: int,
    case_groups: Sequence[Sequence[str]],
) -> str:
    """Return the operational identity affected only by derived packing policy."""
    payload = {
        "schema_kind": TRANSIENT_PT_SHARD_SCHEMA_KIND,
        "schema_version": TRANSIENT_PT_SHARD_SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "dataset_digest": manifest["dataset_digest"],
        "index_digest": index["index_digest"],
        "transient_contract_digest": index["contract_digest"],
        "publication_identity": dict(publication_identity),
        "target_shard_bytes": target_shard_bytes,
        "case_groups": [list(group) for group in case_groups],
    }
    return common.serialization.canonical_json_sha256(payload)


def _emit(
    callback: ProgressCallback | None,
    **values: Any,
) -> None:
    """Emit one bounded progress event when a caller owns presentation."""
    if callback is not None:
        callback(values)


def _save_shards(
    *,
    directory: Path,
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    publication_identity: Mapping[str, str],
    target_shard_bytes: int,
    case_groups: Sequence[Sequence[int]],
    case_sizes: Sequence[int],
    source_paths: Sequence[Path],
    grouped_samples: Sequence[Sequence[Mapping[str, Any]]],
    callback: ProgressCallback | None,
) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, int]], int]:
    """Materialize and serialize one bounded whole-case shard at a time."""
    case_id_groups = [[str(index["cases"][case_index]["package_case_id"]) for case_index in group] for group in case_groups]
    derived_id = _derived_payload_id(
        manifest=manifest,
        index=index,
        publication_identity=publication_identity,
        target_shard_bytes=target_shard_bytes,
        case_groups=case_id_groups,
    )
    shards: list[dict[str, Any]] = []
    locator: dict[str, dict[str, int]] = {}
    total_size = 0
    source_bytes_read = 0
    cases_packed = 0
    for shard_index, group in enumerate(case_groups):
        packed_cases: list[dict[str, Any]] = []
        for case_index in group:
            case = index["cases"][case_index]
            samples = grouped_samples[case_index]
            payload = _case_payload(
                path=source_paths[case_index],
                index=index,
                case_index=case_index,
                case=case,
                samples=samples,
            )
            observed_case_bytes = _tensor_bytes(payload)
            if observed_case_bytes != case_sizes[case_index]:
                message = f"Transient shard tensor-byte plan changed for case {case_index}."
                raise RuntimeError(message)
            packed_cases.append(payload)
            cases_packed += 1
            source_bytes_read += source_paths[case_index].stat().st_size
            _emit(
                callback,
                operation="shard_case_materialization",
                cases_packed=cases_packed,
                cases_total=len(index["cases"]),
                source_bytes_read=source_bytes_read,
                eta="unavailable",
            )
        filename = f"shard_{shard_index:05d}.pt"
        case_ids = case_id_groups[shard_index]
        shard_payload = {
            "schema_kind": TRANSIENT_PT_SHARD_SCHEMA_KIND,
            "schema_version": TRANSIENT_PT_SHARD_SCHEMA_VERSION,
            "dataset_id": manifest["dataset_id"],
            "dataset_digest": manifest["dataset_digest"],
            "index_digest": index["index_digest"],
            "transient_contract_digest": index["contract_digest"],
            "derived_payload_id": derived_id,
            "publication_identity": dict(publication_identity),
            "shard_index": shard_index,
            "case_ids": case_ids,
            "cases": packed_cases,
        }
        path = directory / filename
        common.serialization.atomic_torch_save(shard_payload, path)
        size_bytes, stat_identity = _file_stat_identity(path)
        case_bytes = sum(case_sizes[case_index] for case_index in group)
        record = {
            "filename": filename,
            "shard_index": shard_index,
            "case_ids": case_ids,
            "case_count": len(packed_cases),
            "transition_count": sum(int(case["valid_transition_count"]) for case in packed_cases),
            "tensor_bytes": case_bytes,
            "size_bytes": size_bytes,
            "stat_identity": stat_identity,
            "sha256": common.serialization.file_sha256(path),
            "oversized_single_case": len(packed_cases) == 1 and case_bytes > target_shard_bytes,
            "oversized_reason": ("complete_case_exceeds_soft_target" if len(packed_cases) == 1 and case_bytes > target_shard_bytes else None),
        }
        shards.append(record)
        total_size += size_bytes
        for case_position, case in enumerate(packed_cases):
            case_id = str(case["case_record"]["package_case_id"])
            if case_id in locator:
                message = f"Transient shard packing duplicated case {case_id!r}."
                raise RuntimeError(message)
            locator[case_id] = {
                "case_index": int(case["case_index"]),
                "shard_index": shard_index,
                "case_position": case_position,
            }
        _emit(
            callback,
            operation="shard_building",
            shards_completed=shard_index + 1,
            shards_total=len(case_groups),
            bytes_written=total_size,
            eta="unavailable",
        )
        del shard_payload, packed_cases
    return derived_id, shards, locator, total_size


def _expected_field_order() -> dict[str, list[str]]:
    """Return the exact transient Training field ordering."""
    return {
        "dynamic": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state],
        "static": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning],
        "boundary": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.step_boundary_conditioning],
        "scalars": [field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.scalar_conditioning],
    }


def _finite_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
    label: str,
    validate_values: bool = True,
) -> torch.Tensor:
    """Validate one CPU tensor shape and, on first process access, values."""
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or value.ndim != len(shape)
        or any(expected is not None and value.shape[index] != expected for index, expected in enumerate(shape))
        or (validate_values and not bool(torch.isfinite(value).all()))
    ):
        message = f"Transient shard tensor {label!r} has an invalid dtype, shape, device, or value."
        raise TransientShardContractError(message)
    return value


def _validate_case_payload(
    value: Any,
    *,
    expected_case_index: int,
    expected_case: Mapping[str, Any],
    expected_samples: Sequence[Mapping[str, Any]],
    validate_values: bool = True,
) -> None:
    """Validate one complete case payload against its scientific index."""
    required = {
        "case_index",
        "case_record",
        "samples",
        "field_order",
        "states",
        "static",
        "boundary",
        "scalars",
        "state_time",
        "valid_state_count",
        "valid_transition_count",
        "exact_stop_time",
        "exact_stop_state",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["case_index"] != expected_case_index
        or value["case_record"] != dict(expected_case)
        or value["samples"] != [dict(sample) for sample in expected_samples]
        or value["field_order"] != _expected_field_order()
        or value["valid_state_count"] != expected_case["sequence_length"]
        or value["valid_transition_count"] != expected_case["transition_count"]
    ):
        message = f"Transient shard case metadata is invalid at index {expected_case_index}."
        raise TransientShardContractError(message)
    states = _finite_tensor(
        value["states"],
        dtype=torch.float32,
        shape=(int(expected_case["sequence_length"]), len(_expected_field_order()["dynamic"]), None, None),
        label="states",
        validate_values=validate_values,
    )
    _finite_tensor(
        value["static"],
        dtype=torch.float32,
        shape=(len(_expected_field_order()["static"]), states.shape[2], states.shape[3]),
        label="static",
        validate_values=validate_values,
    )
    _finite_tensor(
        value["boundary"],
        dtype=torch.float32,
        shape=(len(expected_samples), len(_expected_field_order()["boundary"])),
        label="boundary",
        validate_values=validate_values,
    )
    _finite_tensor(
        value["scalars"],
        dtype=torch.float32,
        shape=(len(_expected_field_order()["scalars"]),),
        label="scalars",
        validate_values=validate_values,
    )
    time = _finite_tensor(
        value["state_time"],
        dtype=torch.float64,
        shape=(int(expected_case["sequence_length"]),),
        label="state_time",
        validate_values=validate_values,
    )
    if time.numel() < _MINIMUM_STATE_COUNT or not bool(torch.all(time[1:] > time[:-1])):
        message = f"Transient shard state times are not strictly increasing at case {expected_case_index}."
        raise TransientShardContractError(message)
    exact_time = value["exact_stop_time"]
    exact_state = value["exact_stop_state"]
    expected_exact = expected_case["irregular_stop_time"]
    if expected_exact is None:
        if exact_time is not None or exact_state is not None:
            message = f"Transient shard has unexpected exact-stop data at case {expected_case_index}."
            raise TransientShardContractError(message)
    else:
        if (
            isinstance(exact_time, bool)
            or not isinstance(exact_time, (int, float))
            or not math.isclose(float(exact_time), float(expected_exact), rel_tol=0.0, abs_tol=1.0e-12)
        ):
            message = f"Transient shard exact-stop time is invalid at case {expected_case_index}."
            raise TransientShardContractError(message)
        _finite_tensor(
            exact_state,
            dtype=torch.float32,
            shape=(states.shape[1], states.shape[2], states.shape[3]),
            label="exact_stop_state",
            validate_values=validate_values,
        )


def _load_transient_shard_payload(path: Path) -> dict[str, Any]:
    """Safely deserialize one shard whose byte identity is already admitted."""
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"Transient PT shard is unreadable: {path}."
        raise TransientShardContractError(message) from error
    if not isinstance(payload, dict):
        message = f"Transient PT shard must contain one mapping: {path}."
        raise TransientShardContractError(message)
    return payload


def load_transient_shard_file(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Hash and safely load one shard file for full admission or first use."""
    if not path.is_file() or path.is_symlink():
        message = f"Transient PT shard is missing or unsafe: {path}."
        raise FileNotFoundError(message)
    if common.serialization.file_sha256(path) != expected_sha256:
        message = f"Transient PT shard content changed after publication: {path}."
        raise TransientShardContractError(message)
    return _load_transient_shard_payload(path)


def _validate_loaded_shard(
    payload: Mapping[str, Any],
    *,
    shard_path: Path,
    shard_index: int,
    record: Mapping[str, Any],
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    publication_identity: Mapping[str, str],
    expected_grouped: Sequence[Sequence[Mapping[str, Any]]],
    locator: Mapping[str, Any],
    validate_values: bool = True,
) -> dict[str, Any]:
    """Validate one safely loaded shard against receipt and scientific index."""
    required = {
        "schema_kind",
        "schema_version",
        "dataset_id",
        "dataset_digest",
        "index_digest",
        "transient_contract_digest",
        "derived_payload_id",
        "publication_identity",
        "shard_index",
        "case_ids",
        "cases",
    }
    if (
        set(payload) != required
        or payload["schema_kind"] != TRANSIENT_PT_SHARD_SCHEMA_KIND
        or payload["schema_version"] != TRANSIENT_PT_SHARD_SCHEMA_VERSION
        or payload["dataset_id"] != manifest["dataset_id"]
        or payload["dataset_digest"] != manifest["dataset_digest"]
        or payload["index_digest"] != index["index_digest"]
        or payload["transient_contract_digest"] != index["contract_digest"]
        or payload["derived_payload_id"] != receipt["derived_payload_id"]
        or payload["publication_identity"] != dict(publication_identity)
        or payload["shard_index"] != shard_index
        or payload["case_ids"] != record["case_ids"]
        or not isinstance(payload["cases"], list)
        or len(payload["cases"]) != record["case_count"]
    ):
        message = f"Transient PT shard payload identity is invalid: {shard_path}."
        raise TransientShardContractError(message)
    transition_count = 0
    tensor_bytes = 0
    for case_position, case_payload in enumerate(payload["cases"]):
        case_id = record["case_ids"][case_position]
        location = locator.get(case_id)
        if (
            not isinstance(case_payload, dict)
            or not isinstance(location, dict)
            or location
            != {
                "case_index": int(case_payload.get("case_index", -1)),
                "shard_index": shard_index,
                "case_position": case_position,
            }
        ):
            message = f"Transient PT shard case locator is invalid for {case_id!r}."
            raise TransientShardContractError(message)
        case_index = int(location["case_index"])
        if not 0 <= case_index < len(index["cases"]):
            message = f"Transient PT shard case index is invalid for {case_id!r}."
            raise TransientShardContractError(message)
        _validate_case_payload(
            case_payload,
            expected_case_index=case_index,
            expected_case=index["cases"][case_index],
            expected_samples=expected_grouped[case_index],
            validate_values=validate_values,
        )
        transition_count += int(case_payload["valid_transition_count"])
        tensor_bytes += _tensor_bytes(case_payload)
    if transition_count != record["transition_count"] or tensor_bytes != record["tensor_bytes"]:
        message = f"Transient PT shard byte or transition totals are invalid: {shard_path}."
        raise TransientShardContractError(message)
    return dict(payload)


def _validate_receipt_directory(
    directory: Path,
    receipt: Any,
    *,
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest_sha256: str,
    canonical_payload_size_bytes: int,
    canonical_payload_stat_identity: Mapping[str, int],
    publication_identity: Mapping[str, str] | None,
    validation_depth: ValidationDepth,
    content_hash_verified: bool = False,
) -> dict[str, Any]:
    """Validate one derived directory against package and optional GPU identity."""
    required = {
        "schema_kind",
        "schema_version",
        "status",
        "created_at",
        "dataset_id",
        "dataset_digest",
        "index_digest",
        "transient_contract_digest",
        "package_manifest_sha256",
        "canonical_payload_sha256",
        "canonical_payload_size_bytes",
        "canonical_payload_stat_identity",
        "publication_identity",
        "target_shard_bytes",
        "derived_payload_id",
        "shard_count",
        "case_count",
        "transition_count",
        "total_size_bytes",
        "case_locator",
        "shards",
        "shard_inventory_digest",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        message = f"Transient PT shard receipt schema is invalid: {directory}."
        raise TransientShardContractError(message)
    observed_publication = _validate_publication_identity(receipt["publication_identity"])
    if publication_identity is not None and observed_publication != dict(publication_identity):
        message = f"Transient PT shard GPU publication identity conflicts: {directory}."
        raise TransientShardContractError(message)
    shards = receipt["shards"]
    locator = receipt["case_locator"]
    canonical_stat_identity = receipt["canonical_payload_stat_identity"]
    valid_canonical_stat_identity = (
        isinstance(canonical_stat_identity, dict)
        and set(canonical_stat_identity) == {"device", "inode", "mtime_ns", "ctime_ns"}
        and all(not isinstance(value, bool) and isinstance(value, int) and value >= 0 for value in canonical_stat_identity.values())
    )
    if not isinstance(shards, list) or not isinstance(locator, dict):
        message = f"Transient PT shard receipt inventory is malformed: {directory}."
        raise TransientShardContractError(message)
    case_groups = cast("list[list[str]]", [record.get("case_ids") for record in shards if isinstance(record, dict)])
    target_bytes = receipt["target_shard_bytes"]
    expected_derived_id = (
        _derived_payload_id(
            manifest=manifest,
            index=index,
            publication_identity=observed_publication,
            target_shard_bytes=target_bytes,
            case_groups=case_groups,
        )
        if isinstance(target_bytes, int) and not isinstance(target_bytes, bool) and target_bytes > 0
        else None
    )
    if (
        receipt["schema_kind"] != TRANSIENT_PT_RECEIPT_SCHEMA_KIND
        or receipt["schema_version"] != TRANSIENT_PT_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "complete"
        or not isinstance(receipt["created_at"], str)
        or not receipt["created_at"]
        or receipt["dataset_id"] != manifest["dataset_id"]
        or receipt["dataset_digest"] != manifest["dataset_digest"]
        or receipt["index_digest"] != index["index_digest"]
        or receipt["transient_contract_digest"] != index["contract_digest"]
        or receipt["package_manifest_sha256"] != manifest_sha256
        or receipt["canonical_payload_sha256"] != manifest["payload_sha256"]
        or isinstance(receipt["canonical_payload_size_bytes"], bool)
        or not isinstance(receipt["canonical_payload_size_bytes"], int)
        or receipt["canonical_payload_size_bytes"] != canonical_payload_size_bytes
        or not valid_canonical_stat_identity
        or canonical_stat_identity != dict(canonical_payload_stat_identity)
        or expected_derived_id is None
        or receipt["derived_payload_id"] != expected_derived_id
        or receipt["shard_count"] != len(shards)
        or receipt["case_count"] != len(index["cases"])
        or receipt["transition_count"] != len(index["samples"])
        or receipt["shard_inventory_digest"] != common.serialization.canonical_json_sha256(shards)
    ):
        message = f"Transient PT shard receipt identity is invalid: {directory}."
        raise TransientShardContractError(message)
    expected_grouped = _case_samples(index)
    observed_case_ids: list[str] = []
    observed_bytes = 0
    for shard_index, record in enumerate(shards):
        shard_required = {
            "filename",
            "shard_index",
            "case_ids",
            "case_count",
            "transition_count",
            "tensor_bytes",
            "size_bytes",
            "stat_identity",
            "sha256",
            "oversized_single_case",
            "oversized_reason",
        }
        filename = f"shard_{shard_index:05d}.pt"
        stat_identity = record.get("stat_identity") if isinstance(record, dict) else None
        valid_stat_identity = (
            isinstance(stat_identity, dict)
            and set(stat_identity) == {"device", "inode", "mtime_ns", "ctime_ns"}
            and all(not isinstance(value, bool) and isinstance(value, int) and value >= 0 for value in stat_identity.values())
        )
        if (
            not isinstance(record, dict)
            or set(record) != shard_required
            or record["filename"] != filename
            or record["shard_index"] != shard_index
            or not isinstance(record["case_ids"], list)
            or record["case_count"] != len(record["case_ids"])
            or record["case_count"] < 1
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 1
            or not valid_stat_identity
            or _SHA256_PATTERN.fullmatch(str(record["sha256"])) is None
            or record["oversized_single_case"] is not (record["case_count"] == 1 and record["tensor_bytes"] > target_bytes)
            or record["oversized_reason"] != ("complete_case_exceeds_soft_target" if record["oversized_single_case"] else None)
        ):
            message = f"Transient PT shard inventory record is invalid: {directory / filename}."
            raise TransientShardContractError(message)
        shard_path = directory / filename
        if not shard_path.is_file() or shard_path.is_symlink():
            message = f"Transient PT shard is missing or unsafe: {shard_path}."
            raise FileNotFoundError(message)
        observed_size, observed_stat_identity = _file_stat_identity(shard_path)
        if observed_size != record["size_bytes"] or observed_stat_identity != stat_identity:
            message = f"Transient PT shard changed after immutable publication: {shard_path}."
            raise TransientShardContractError(message)
        observed_case_ids.extend(record["case_ids"])
        observed_bytes += record["size_bytes"]
        if validation_depth == "evidence":
            continue
        payload = (
            _load_transient_shard_payload(shard_path)
            if content_hash_verified
            else load_transient_shard_file(
                shard_path,
                expected_sha256=str(record["sha256"]),
            )
        )
        _validate_loaded_shard(
            payload,
            shard_path=shard_path,
            shard_index=shard_index,
            record=record,
            receipt=receipt,
            manifest=manifest,
            index=index,
            publication_identity=observed_publication,
            expected_grouped=expected_grouped,
            locator=locator,
        )
    expected_case_ids = [str(case["package_case_id"]) for case in index["cases"]]
    if (
        observed_case_ids != expected_case_ids
        or len(set(observed_case_ids)) != len(observed_case_ids)
        or set(locator) != set(expected_case_ids)
        or observed_bytes != receipt["total_size_bytes"]
    ):
        message = f"Transient PT shard case membership or byte totals are invalid: {directory}."
        raise TransientShardContractError(message)
    return dict(receipt)


def _load_transient_shard_receipt_directory(
    directory: Path,
    *,
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest_sha256: str,
    canonical_payload_size_bytes: int,
    canonical_payload_stat_identity: Mapping[str, int],
    publication_identity: Mapping[str, str] | None,
    validation_depth: ValidationDepth,
) -> dict[str, Any]:
    """Load one shard receipt using an already admitted package context."""
    receipt_path = directory / TRANSIENT_PT_RECEIPT_FILENAME
    if not receipt_path.is_file() or receipt_path.is_symlink():
        message = f"Transient PT shard receipt is missing or unsafe: {receipt_path}."
        raise FileNotFoundError(message)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        message = f"Transient PT shard receipt is unreadable: {receipt_path}."
        raise TransientShardContractError(message) from error
    return _validate_receipt_directory(
        directory,
        receipt,
        manifest=manifest,
        index=index,
        manifest_sha256=manifest_sha256,
        canonical_payload_size_bytes=canonical_payload_size_bytes,
        canonical_payload_stat_identity=canonical_payload_stat_identity,
        publication_identity=publication_identity,
        validation_depth=validation_depth,
    )


def load_transient_shard_receipt(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
    publication_identity: Mapping[str, str] | None = None,
    validation_depth: ValidationDepth = "full",
) -> dict[str, Any]:
    """Load one package-bound shard receipt with evidence or full validation."""
    if validation_depth not in {"evidence", "full"}:
        message = f"Unsupported transient shard validation depth: {validation_depth!r}."
        raise ValueError(message)
    storage, manifest, index, manifest_sha256 = _package_context(
        dataset_id,
        storage_root=storage_root,
        validation_depth=validation_depth,
    )
    canonical_size, canonical_stat = _canonical_payload_stat_identity(
        storage,
        manifest,
    )
    expected_publication = None if publication_identity is None else _validate_publication_identity(dict(publication_identity))
    return _load_transient_shard_receipt_directory(
        transient_shard_directory(dataset_id, storage_root=storage),
        manifest=manifest,
        index=index,
        manifest_sha256=manifest_sha256,
        canonical_payload_size_bytes=canonical_size,
        canonical_payload_stat_identity=canonical_stat,
        publication_identity=expected_publication,
        validation_depth=validation_depth,
    )


def load_transient_shard_payload(
    dataset_id: str,
    shard_index: int,
    *,
    storage_root: Path | str | None = None,
    receipt: Mapping[str, Any] | None = None,
    validate_values: bool = True,
) -> dict[str, Any]:
    """Stat-admit and mmap-load one shard, scanning values once per process."""
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        message = "shard_index must be a non-negative integer."
        raise ValueError(message)
    if type(validate_values) is not bool:
        message = "validate_values must be boolean."
        raise TypeError(message)
    storage, manifest, index, _manifest_sha256 = _package_context(
        dataset_id,
        storage_root=storage_root,
        validation_depth="evidence",
    )
    if receipt is None:
        canonical_size, canonical_stat = _canonical_payload_stat_identity(
            storage,
            manifest,
        )
        admitted = _load_transient_shard_receipt_directory(
            transient_shard_directory(dataset_id, storage_root=storage),
            manifest=manifest,
            index=index,
            manifest_sha256=_manifest_sha256,
            canonical_payload_size_bytes=canonical_size,
            canonical_payload_stat_identity=canonical_stat,
            publication_identity=None,
            validation_depth="evidence",
        )
    else:
        admitted = dict(receipt)
    shards = admitted.get("shards")
    if admitted.get("dataset_id") != dataset_id or not isinstance(shards, list) or shard_index >= len(shards):
        message = f"Transient PT shard index {shard_index} is outside the admitted receipt."
        raise TransientShardContractError(message)
    record = shards[shard_index]
    if not isinstance(record, dict):
        message = f"Transient PT shard record {shard_index} is malformed."
        raise TransientShardContractError(message)
    shard_path = transient_shard_directory(dataset_id, storage_root=storage) / str(record["filename"])
    if not shard_path.is_file() or shard_path.is_symlink():
        message = f"Transient PT shard is missing or unsafe: {shard_path}."
        raise FileNotFoundError(message)
    observed_size, observed_stat_identity = _file_stat_identity(shard_path)
    if observed_size != record.get("size_bytes") or observed_stat_identity != record.get("stat_identity"):
        message = f"Transient PT shard changed after immutable publication: {shard_path}."
        raise TransientShardContractError(message)
    payload = _load_transient_shard_payload(shard_path)
    publication_identity = _validate_publication_identity(admitted["publication_identity"])
    locator = admitted.get("case_locator")
    if not isinstance(locator, dict):
        message = "Transient PT shard receipt case locator is malformed."
        raise TransientShardContractError(message)
    return _validate_loaded_shard(
        payload,
        shard_path=shard_path,
        shard_index=shard_index,
        record=record,
        receipt=admitted,
        manifest=manifest,
        index=index,
        publication_identity=publication_identity,
        expected_grouped=_case_samples(index),
        locator=locator,
        validate_values=validate_values,
    )


def _raw_shard_destination_conflicts(
    destination: Path,
    *,
    dataset_id: str,
    publication_identity: Mapping[str, str],
    target_shard_bytes: int,
) -> bool:
    """Return whether invalid shard evidence claims another immutable owner."""
    receipt_path = destination / TRANSIENT_PT_RECEIPT_FILENAME
    if receipt_path.is_symlink() or (receipt_path.exists() and not receipt_path.is_file()):
        return True
    if not receipt_path.is_file():
        return False
    try:
        raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw_receipt, dict):
        return False
    if raw_receipt.get("dataset_id") not in {None, dataset_id}:
        return True
    if raw_receipt.get("target_shard_bytes") not in {None, target_shard_bytes}:
        return True
    raw_publication = raw_receipt.get("publication_identity")
    if not isinstance(raw_publication, dict):
        return False
    observed_keys = set(raw_publication)
    expected_keys = set(publication_identity)
    if observed_keys in _PUBLICATION_IDENTITY_KEY_SETS and observed_keys != expected_keys:
        return True
    return any(key in raw_publication and raw_publication[key] != value for key, value in publication_identity.items())


def _admit_existing_shard_destination(
    dataset_id: str,
    *,
    destination: Path,
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest_sha256: str,
    canonical_payload_size_bytes: int,
    canonical_payload_stat_identity: Mapping[str, int],
    publication_identity: Mapping[str, str],
    target_shard_bytes: int,
    validation_depth: ValidationDepth,
    rebuild_invalid: bool,
    case_count: int,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Return reusable evidence or authorize one atomic replacement."""
    if not destination.exists():
        return None, False
    if destination.is_symlink() or not destination.is_dir():
        message = f"Transient PT shard destination is unsafe: {destination}."
        raise TransientShardContractError(message)
    try:
        receipt = _load_transient_shard_receipt_directory(
            destination,
            manifest=manifest,
            index=index,
            manifest_sha256=manifest_sha256,
            canonical_payload_size_bytes=canonical_payload_size_bytes,
            canonical_payload_stat_identity=canonical_payload_stat_identity,
            publication_identity=None,
            validation_depth=validation_depth,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        if _raw_shard_destination_conflicts(
            destination,
            dataset_id=dataset_id,
            publication_identity=publication_identity,
            target_shard_bytes=target_shard_bytes,
        ):
            message = f"Invalid transient PT shard evidence claims a different immutable owner for {dataset_id!r}."
            raise FileExistsError(message) from error
        if not rebuild_invalid:
            raise
        _emit(
            progress,
            operation="shard_rebuild",
            cases_packed=0,
            cases_total=case_count,
            shards_completed=0,
            bytes_written=0,
            eta="unavailable",
        )
        return None, True
    if receipt["publication_identity"] != dict(publication_identity):
        message = f"Existing transient PT shard publication identity conflicts for {dataset_id!r}."
        raise FileExistsError(message)
    if receipt["target_shard_bytes"] != target_shard_bytes:
        message = f"Existing transient PT shard packing identity conflicts for {dataset_id!r}."
        raise FileExistsError(message)
    _emit(
        progress,
        operation="shard_reuse",
        cases_packed=receipt["case_count"],
        cases_total=receipt["case_count"],
        shards_completed=receipt["shard_count"],
        shards_total=receipt["shard_count"],
        bytes_written=0,
        eta="unavailable",
    )
    return receipt, False


def _remove_stale_shard_attempts(state_root: Path, dataset_id: str) -> None:
    """Remove only inactive shard staging and backup paths for one locked Dataset."""
    if not state_root.exists():
        return
    prefix = f".{dataset_id}.transient-pt."
    for path in state_root.iterdir():
        staging = path.name.startswith(prefix) and path.name.endswith(".tmp")
        backup = path.name.startswith(f"{prefix}invalid-") and path.name.endswith(".backup")
        if not staging and not backup:
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)


def build_transient_shards(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
    publication_identity: Mapping[str, str],
    target_shard_bytes: int = DEFAULT_TRANSIENT_PT_SHARD_BYTES,
    existing_validation_depth: ValidationDepth = "evidence",
    rebuild_invalid: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build, validate, and atomically publish one derived PT payload."""
    if isinstance(target_shard_bytes, bool) or not isinstance(target_shard_bytes, int) or target_shard_bytes < 1:
        message = "target_shard_bytes must be a positive integer."
        raise ValueError(message)
    if existing_validation_depth not in {"evidence", "full"}:
        message = f"Unsupported transient shard validation depth: {existing_validation_depth!r}."
        raise ValueError(message)
    if not isinstance(rebuild_invalid, bool):
        message = "rebuild_invalid must be boolean."
        raise TypeError(message)
    stable_publication = _validate_publication_identity(dict(publication_identity))
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    destination = transient_shard_directory(dataset_id, storage_root=storage)
    context_depth: ValidationDepth = existing_validation_depth if destination.exists() else "full"
    storage, manifest, index, manifest_sha256 = _package_context(
        dataset_id,
        storage_root=storage,
        validation_depth=context_depth,
    )
    canonical_size, canonical_stat = _canonical_payload_stat_identity(
        storage,
        manifest,
    )
    lock_path = common.paths.resolve_dataset_build_lock_path(
        dataset_id,
        storage_root=storage,
    )
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        state_root = common.paths.get_dataset_state_root(storage_root=storage)
        existing, replace_invalid = _admit_existing_shard_destination(
            dataset_id,
            destination=destination,
            manifest=manifest,
            index=index,
            manifest_sha256=manifest_sha256,
            canonical_payload_size_bytes=canonical_size,
            canonical_payload_stat_identity=canonical_stat,
            publication_identity=stable_publication,
            target_shard_bytes=target_shard_bytes,
            validation_depth=context_depth,
            rebuild_invalid=rebuild_invalid,
            case_count=len(index["cases"]),
            progress=progress,
        )
        if existing is not None:
            _remove_stale_shard_attempts(state_root, dataset_id)
            return {
                "status": "reused",
                "receipt_path": destination / TRANSIENT_PT_RECEIPT_FILENAME,
                "receipt": existing,
            }
        if context_depth != "full":
            storage, manifest, index, manifest_sha256 = _package_context(
                dataset_id,
                storage_root=storage,
                validation_depth="full",
            )
            canonical_size, canonical_stat = _canonical_payload_stat_identity(
                storage,
                manifest,
            )
        state_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{dataset_id}.transient-pt.",
                suffix=".tmp",
                dir=state_root,
            )
        )
        try:
            grouped_samples = _case_samples(index)
            source_paths: list[Path] = []
            case_sizes: list[int] = []
            source_bytes_validated = 0
            for case_index, (case, samples) in enumerate(
                zip(index["cases"], grouped_samples, strict=True),
            ):
                source_path = _resolve_source(storage, case)
                source_paths.append(source_path)
                case_sizes.append(
                    _planned_case_tensor_bytes(
                        source_path,
                        case,
                        samples,
                    )
                )
                source_bytes_validated += source_path.stat().st_size
                _emit(
                    progress,
                    operation="shard_source_validation",
                    cases_completed=case_index + 1,
                    cases_total=len(index["cases"]),
                    source_bytes_validated=source_bytes_validated,
                    eta="unavailable",
                )
            case_groups = _plan_case_groups(
                case_sizes,
                target_shard_bytes=target_shard_bytes,
            )
            derived_id, shards, locator, total_size = _save_shards(
                directory=staging,
                manifest=manifest,
                index=index,
                publication_identity=stable_publication,
                target_shard_bytes=target_shard_bytes,
                case_groups=case_groups,
                case_sizes=case_sizes,
                source_paths=source_paths,
                grouped_samples=grouped_samples,
                callback=progress,
            )
            receipt = {
                "schema_kind": TRANSIENT_PT_RECEIPT_SCHEMA_KIND,
                "schema_version": TRANSIENT_PT_RECEIPT_SCHEMA_VERSION,
                "status": "complete",
                "created_at": _utc_now(),
                "dataset_id": manifest["dataset_id"],
                "dataset_digest": manifest["dataset_digest"],
                "index_digest": index["index_digest"],
                "transient_contract_digest": index["contract_digest"],
                "package_manifest_sha256": manifest_sha256,
                "canonical_payload_sha256": manifest["payload_sha256"],
                "canonical_payload_size_bytes": canonical_size,
                "canonical_payload_stat_identity": canonical_stat,
                "publication_identity": stable_publication,
                "target_shard_bytes": target_shard_bytes,
                "derived_payload_id": derived_id,
                "shard_count": len(shards),
                "case_count": len(index["cases"]),
                "transition_count": len(index["samples"]),
                "total_size_bytes": total_size,
                "case_locator": locator,
                "shards": shards,
                "shard_inventory_digest": common.serialization.canonical_json_sha256(shards),
            }
            common.serialization.atomic_write_json(
                staging / TRANSIENT_PT_RECEIPT_FILENAME,
                receipt,
            )
            _validate_receipt_directory(
                staging,
                receipt,
                manifest=manifest,
                index=index,
                manifest_sha256=manifest_sha256,
                canonical_payload_size_bytes=canonical_size,
                canonical_payload_stat_identity=canonical_stat,
                publication_identity=stable_publication,
                validation_depth="full",
                content_hash_verified=True,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if replace_invalid:
                backup = state_root / (f".{dataset_id}.transient-pt.invalid-{uuid.uuid4().hex}.backup")
                destination.replace(backup)
                try:
                    staging.replace(destination)
                except BaseException:
                    if not destination.exists() and backup.exists():
                        backup.replace(destination)
                    raise
                else:
                    shutil.rmtree(backup)
            else:
                staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        validated = _load_transient_shard_receipt_directory(
            destination,
            manifest=manifest,
            index=index,
            manifest_sha256=manifest_sha256,
            canonical_payload_size_bytes=canonical_size,
            canonical_payload_stat_identity=canonical_stat,
            publication_identity=stable_publication,
            validation_depth="evidence",
        )
        _remove_stale_shard_attempts(state_root, dataset_id)
    return {
        "status": "rebuilt" if replace_invalid else "complete",
        "receipt_path": destination / TRANSIENT_PT_RECEIPT_FILENAME,
        "receipt": validated,
    }
