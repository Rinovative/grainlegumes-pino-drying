"""Construct storage-neutral transient training loaders from published packages.

Responsibilities:
  - Resolve experiment-owned transient case roles before expanding transitions
  - Build and replay strict case and item membership evidence
  - Fit transient scaling from one-step Train evidence only
  - Build deterministically seeded training and evaluation DataLoaders

Design principles:
  - Complete case roles are selected before transitions or rollout windows expand
  - Scaling uses one-step Train evidence and exact restored or cache-bound state
  - Storage backend choices preserve the same membership and sample contract

This module does NOT:
  - Read HDF5 directly or create an alternative transient storage runtime
  - Choose transient tensor channels, losses, or training stages
  - Publish run-owned split or normalizer artifacts
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src import common, domain
from src.datasets import packages
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.runtime import dataset_runtime_factory as factory
from src.datasets.runtime import dataset_runtime_transient as transient
from src.learning.transient import learning_transient_scaling as scaling
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from pathlib import Path

SPLIT_SCHEMA_KIND: Final = "experiment_transient_case_split"
SPLIT_SCHEMA_VERSION: Final = 2
# Every evaluation role yields whole trajectories: checkpoint selection rolls each one out autonomously from its start.
EVALUATION_SAMPLING: Final = transient_contract.TransientSamplingSpec(mode="trajectory")
_TASK: Final = "transient_drying"
_SCALER_CACHE_SCHEMA_KIND: Final = "transient_drying_scaler_cache"
_SCALER_CACHE_SCHEMA_VERSION: Final = 1
_SHA256_LENGTH: Final = 64
_MINIMUM_SPATIAL_AXIS_LENGTH: Final = 2
_MINIMUM_SPLIT_CASES: Final = 3
StartupCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TransientTrainingLoaders:
    """Return loaders and immutable evidence for one transient training run."""

    train: DataLoader[Any]
    evaluation: DataLoader[Any]
    additional_evaluation: DataLoader[Any] | None
    id_test: DataLoader[Any]
    scaling_artifact: scaling.TransientScalingArtifact
    split: dict[str, Any]
    dataset_identity: dict[str, Any]
    spatial_view: dict[str, int]
    runtime_provenance: dict[str, Any]


def _sha256(value: object) -> str:
    """Return one canonical SHA-256 digest for JSON-compatible evidence."""
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _emit_startup(
    callback: StartupCallback | None,
    event: str,
    **values: Any,
) -> None:
    """Forward one bounded semantic startup event to its presentation owner."""
    if callback is not None:
        callback(event, values)


def _spatial_view_evidence(
    dataset: transient.TransientPhysicalDataset,
) -> dict[str, int]:
    """Return the exact canonical and configured effective spatial view."""
    canonical, effective = dataset.spatial_view()
    return {
        "spatial_stride": dataset.spatial_stride,
        "canonical_ny": canonical[0],
        "canonical_nx": canonical[1],
        "effective_ny": effective[0],
        "effective_nx": effective[1],
    }


def _admit_spatial_view(
    value: Any,
    *,
    expected_stride: int,
) -> dict[str, int]:
    """Admit one boundary-preserving persisted spatial-view identity."""
    required = {
        "spatial_stride",
        "canonical_ny",
        "canonical_nx",
        "effective_ny",
        "effective_nx",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        message = "Transient split spatial_view has unexpected fields."
        raise ValueError(message)
    stride = transient_contract.validate_spatial_stride(value["spatial_stride"])
    if stride != transient_contract.validate_spatial_stride(expected_stride):
        message = "Transient split spatial stride does not match the resolved configuration."
        raise ValueError(message)
    axes = tuple(value[key] for key in ("canonical_ny", "canonical_nx", "effective_ny", "effective_nx"))
    if any(isinstance(axis, bool) or not isinstance(axis, int) or axis < _MINIMUM_SPATIAL_AXIS_LENGTH for axis in axes):
        message = "Transient split spatial-view axes must be integers >= 2."
        raise ValueError(message)
    canonical = cast("tuple[int, int]", axes[:2])
    effective = cast("tuple[int, int]", axes[2:])
    if transient_contract.resolve_spatial_view(canonical, stride) != effective:
        message = "Transient split effective grid does not match its canonical grid and stride."
        raise ValueError(message)
    return {
        "spatial_stride": stride,
        "canonical_ny": canonical[0],
        "canonical_nx": canonical[1],
        "effective_ny": effective[0],
        "effective_nx": effective[1],
    }


def _require_fraction(value: float) -> float:
    """Validate the requested case-owned additional evaluation selection fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        message = "additional_evaluation_fraction must be one finite fraction in (0, 1]."
        raise ValueError(message)
    return float(value)


def _worker_initializer(base_seed: int) -> Callable[[int], None]:
    """Create deterministic Python, NumPy, and Torch worker initialization."""

    def initialize(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))  # noqa: NPY002 -- worker process RNG
        torch.manual_seed(worker_seed)

    return initialize


def _case_positions(dataset: transient.TransientPhysicalDataset) -> dict[str, tuple[int, ...]]:
    """Return ordered transition-index positions grouped by immutable package case."""
    positions: dict[str, list[int]] = {}
    for position in dataset.sample_indices:
        sample = dataset.payload["samples"][position]
        case = dataset.payload["cases"][int(sample["case_index"])]
        case_id = str(case["package_case_id"])
        positions.setdefault(case_id, []).append(position)
    return {case_id: tuple(values) for case_id, values in positions.items()}


def _select_cases(
    dataset: transient.TransientPhysicalDataset,
    case_ids: Sequence[str],
) -> transient.TransientPhysicalDataset:
    """Return one factory-derived Dataset restricted to complete ordered cases."""
    available = _case_positions(dataset)
    selected_ids = tuple(case_ids)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)) or any(case_id not in available for case_id in selected_ids):
        message = "Transient case selection must contain unique available package case IDs."
        raise ValueError(message)
    return transient.select_transient_cases(dataset, selected_ids)


def _dataset_identity(dataset: transient.TransientPhysicalDataset) -> dict[str, Any]:
    """Return index and package evidence sufficient to reject dataset drift."""
    payload = dataset.payload
    required = ("dataset_id", "contract_digest", "index_digest", "configured_regular_horizon")
    if any(key not in payload for key in required):
        message = "Transient runtime payload lacks published identity evidence."
        raise ValueError(message)
    return {
        "dataset_id": str(payload["dataset_id"]),
        "data_contract_digest": str(payload["contract_digest"]),
        "index_digest": str(payload["index_digest"]),
        "configured_regular_horizon": dict(payload["configured_regular_horizon"]),
    }


def _scaling_dataset_identity(
    dataset: transient.TransientPhysicalDataset,
    spatial_view: Mapping[str, int],
) -> dict[str, Any]:
    """Bind scaler evidence to the full package manifest and spatial view."""
    manifest_digest = dataset.dataset_manifest_digest
    if manifest_digest is None:
        message = "Transient scaler identity requires a factory-admitted Dataset manifest digest."
        raise ValueError(message)
    return {
        **_dataset_identity(dataset),
        "dataset_manifest_digest": manifest_digest,
        "spatial_stride": spatial_view["spatial_stride"],
        "canonical_spatial_shape": [
            spatial_view["canonical_ny"],
            spatial_view["canonical_nx"],
        ],
        "effective_spatial_shape": [
            spatial_view["effective_ny"],
            spatial_view["effective_nx"],
        ],
    }


def _scaler_cache_identity(
    *,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    horizon: float,
) -> dict[str, Any]:
    """Return the complete scientific identity of one reusable scaler fit."""
    task = domain.tasks.registry.get_task("transient_drying")
    return {
        "task_contract_digest": task.contract_digest,
        "data_contract_digest": task.data_contract_digest,
        "tensorizer": tensorizer.as_dict(),
        "dataset_identity": copy.deepcopy(dict(dataset_identity)),
        "train_membership_digest": train_membership_digest,
        "scale_mode": scale_mode,
        "horizon": float(horizon),
    }


def _validate_scaling_artifact(
    artifact: scaling.TransientScalingArtifact,
    *,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    spatial_view: Mapping[str, int],
    horizon: float,
) -> None:
    """Require exact current scaler identity and Train membership."""
    expected_identity = dict(dataset_identity)
    identity_matches = artifact.dataset_identity == expected_identity
    expected_shape = (
        spatial_view["effective_ny"],
        spatial_view["effective_nx"],
    )
    if (
        artifact.tensorizer != tensorizer
        or not identity_matches
        or artifact.train_membership_digest != train_membership_digest
        or artifact.scale_mode != scale_mode
        or artifact.spatial_shape != expected_shape
        or artifact.horizon != float(horizon)
    ):
        message = "Transient scaling artifact does not match the exact package membership, spatial view, tensorizer, horizon, or scale mode."
        raise ValueError(message)


def _load_scaler_cache_entry(
    path: Path,
    *,
    cache_identity: Mapping[str, Any],
    cache_digest: str,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    spatial_view: Mapping[str, int],
    horizon: float,
) -> scaling.TransientScalingArtifact:
    """Strictly admit one immutable identity-addressed scaler cache entry."""
    if path.is_symlink() or not path.is_file():
        message = f"Transient scaler cache entry is missing or unsafe: {path}."
        raise ValueError(message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Transient scaler cache entry is malformed: {path}."
        raise ValueError(message) from error
    required = {
        "schema_kind",
        "schema_version",
        "cache_identity",
        "cache_identity_sha256",
        "artifact_sha256",
        "artifact",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["schema_kind"] != _SCALER_CACHE_SCHEMA_KIND
        or payload["schema_version"] != _SCALER_CACHE_SCHEMA_VERSION
        or payload["cache_identity"] != dict(cache_identity)
        or payload["cache_identity_sha256"] != cache_digest
        or _sha256(payload["cache_identity"]) != cache_digest
        or not isinstance(payload["artifact_sha256"], str)
        or not isinstance(payload["artifact"], Mapping)
    ):
        message = f"Transient scaler cache identity is invalid: {path}."
        raise ValueError(message)
    artifact = scaling.TransientScalingArtifact.from_state_dict(payload["artifact"])
    if artifact.digest != payload["artifact_sha256"]:
        message = f"Transient scaler cache artifact digest is invalid: {path}."
        raise ValueError(message)
    _validate_scaling_artifact(
        artifact,
        tensorizer=tensorizer,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        scale_mode=scale_mode,
        spatial_view=spatial_view,
        horizon=horizon,
    )
    return artifact


def _fit_scaling_artifact(
    dataset: transient.TransientPhysicalDataset,
    *,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    startup_callback: StartupCallback | None,
) -> scaling.TransientScalingArtifact:
    """Fit once, using shard-grouped iteration and bounded shard progress."""
    shard_count = dataset.shard_count if isinstance(dataset, transient.TransientPTShardDataset) else None
    _emit_startup(
        startup_callback,
        "scaler_fit_start",
        train_transitions=len(dataset),
        shards=shard_count,
    )
    started = time.perf_counter()
    if isinstance(dataset, transient.TransientPTShardDataset):
        items = dataset.iter_scaling_items(
            progress_callback=lambda processed, total: _emit_startup(
                startup_callback,
                "scaler_fit_progress",
                shards_processed=processed,
                shards_total=total,
            )
        )
    else:
        items = (dataset[index] for index in range(len(dataset)))
    artifact = scaling.fit_transient_scaling(
        items,
        tensorizer=tensorizer,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        horizon=dataset.configured_regular_horizon,
        scale_mode=scale_mode,
    )
    _emit_startup(
        startup_callback,
        "scaler_fit_complete",
        seconds=time.perf_counter() - started,
        train_transitions=len(dataset),
        shards=shard_count,
    )
    return artifact


def _fit_or_load_scaling_artifact(
    dataset: transient.TransientPhysicalDataset,
    *,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    spatial_view: Mapping[str, int],
    storage_root: str | None,
    startup_callback: StartupCallback | None,
) -> scaling.TransientScalingArtifact:
    """Reuse or atomically publish one exact transient Train scaler."""
    cache_identity = _scaler_cache_identity(
        tensorizer=tensorizer,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        scale_mode=scale_mode,
        horizon=dataset.configured_regular_horizon,
    )
    cache_digest = _sha256(cache_identity)
    cache_root = common.paths.get_transient_scaler_cache_root(storage_root=storage_root)
    cache_path = cache_root / f"{cache_digest}.json"
    lock_path = cache_root / "locks" / f"{cache_digest}.lock"
    stack = ExitStack()
    try:
        stack.enter_context(
            common.locking.exclusive_file_lock(
                lock_path,
                blocking=True,
            )
        )
    except OSError as error:
        stack.close()
        _emit_startup(
            startup_callback,
            "scaler_cache",
            state="unavailable",
            reason=type(error).__name__,
        )
        return _fit_scaling_artifact(
            dataset,
            tensorizer=tensorizer,
            dataset_identity=dataset_identity,
            train_membership_digest=train_membership_digest,
            scale_mode=scale_mode,
            startup_callback=startup_callback,
        )
    with stack:
        if cache_path.exists() or cache_path.is_symlink():
            try:
                artifact = _load_scaler_cache_entry(
                    cache_path,
                    cache_identity=cache_identity,
                    cache_digest=cache_digest,
                    tensorizer=tensorizer,
                    dataset_identity=dataset_identity,
                    train_membership_digest=train_membership_digest,
                    scale_mode=scale_mode,
                    spatial_view=spatial_view,
                    horizon=dataset.configured_regular_horizon,
                )
            except OSError as error:
                _emit_startup(
                    startup_callback,
                    "scaler_cache",
                    state="unavailable",
                    reason=type(error).__name__,
                )
                return _fit_scaling_artifact(
                    dataset,
                    tensorizer=tensorizer,
                    dataset_identity=dataset_identity,
                    train_membership_digest=train_membership_digest,
                    scale_mode=scale_mode,
                    startup_callback=startup_callback,
                )
            _emit_startup(
                startup_callback,
                "scaler_cache",
                state="hit",
                identity=cache_digest,
            )
            return artifact

        _emit_startup(
            startup_callback,
            "scaler_cache",
            state="miss",
            identity=cache_digest,
        )
        artifact = _fit_scaling_artifact(
            dataset,
            tensorizer=tensorizer,
            dataset_identity=dataset_identity,
            train_membership_digest=train_membership_digest,
            scale_mode=scale_mode,
            startup_callback=startup_callback,
        )
        cache_payload = {
            "schema_kind": _SCALER_CACHE_SCHEMA_KIND,
            "schema_version": _SCALER_CACHE_SCHEMA_VERSION,
            "cache_identity": cache_identity,
            "cache_identity_sha256": cache_digest,
            "artifact_sha256": artifact.digest,
            "artifact": artifact.state_dict(),
        }
        try:
            common.serialization.atomic_write_json(cache_path, cache_payload)
        except OSError as error:
            _emit_startup(
                startup_callback,
                "scaler_cache",
                state="unavailable",
                reason=type(error).__name__,
            )
        return artifact


def _item_ids(dataset: transient.TransientPhysicalDataset) -> tuple[str, ...]:
    """Return deterministic runtime item IDs without materializing arrays."""
    return dataset.runtime_item_ids()


def _role_evidence(dataset: transient.TransientPhysicalDataset) -> dict[str, Any]:
    """Serialize complete case-owned membership evidence for one loader role."""
    case_ids = tuple(_case_positions(dataset))
    item_ids = _item_ids(dataset)
    return {
        "case_ids": list(case_ids),
        "item_ids": list(item_ids),
        "membership_digest": _sha256({"case_ids": case_ids, "item_ids": item_ids}),
    }


def _admit_role_evidence(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Admit one strict case-owned expanded runtime membership."""
    required = {"case_ids", "item_ids", "membership_digest"}
    if not isinstance(value, Mapping) or set(value) != required:
        message = f"Transient split {label} evidence has unexpected fields."
        raise ValueError(message)
    case_ids = value["case_ids"]
    item_ids = value["item_ids"]
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(item, str) and item for item in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        message = f"Transient split {label} case IDs are invalid."
        raise ValueError(message)
    if (
        not isinstance(item_ids, list)
        or not item_ids
        or not all(isinstance(item, str) and item for item in item_ids)
        or len(item_ids) != len(set(item_ids))
    ):
        message = f"Transient split {label} runtime item IDs are invalid."
        raise ValueError(message)
    expected_digest = _sha256(
        {
            "case_ids": tuple(case_ids),
            "item_ids": tuple(item_ids),
        }
    )
    if value["membership_digest"] != expected_digest:
        message = f"Transient split {label} membership digest is invalid."
        raise ValueError(message)
    return {
        "case_ids": list(case_ids),
        "item_ids": list(item_ids),
        "membership_digest": expected_digest,
    }


def _admit_dataset_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Admit one published transient package identity."""
    required = {
        "dataset_id",
        "data_contract_digest",
        "index_digest",
        "configured_regular_horizon",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        message = f"Transient split {label} dataset identity is invalid."
        raise ValueError(message)
    if not isinstance(value["dataset_id"], str) or not value["dataset_id"] or not isinstance(value["configured_regular_horizon"], Mapping):
        message = f"Transient split {label} dataset identity is incomplete."
        raise ValueError(message)
    for key in ("data_contract_digest", "index_digest"):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in digest):
            message = f"Transient split {label} {key} is not SHA-256."
            raise ValueError(message)
    return {
        "dataset_id": value["dataset_id"],
        "data_contract_digest": value["data_contract_digest"],
        "index_digest": value["index_digest"],
        "configured_regular_horizon": dict(value["configured_regular_horizon"]),
    }


def admit_transient_training_split(
    value: Mapping[str, Any],
    *,
    tensorizer: TransientTensorizerSpec,
    sampling: transient_contract.TransientSamplingSpec,
    additional_evaluation_fraction: float,
    split_seed: int | None = None,
    spatial_stride: int = 1,
) -> dict[str, Any]:
    """Admit Dataset fixed membership and the run-owned Train and evaluation item expansion evidence."""
    base_required = {
        "schema_kind",
        "schema_version",
        "task",
        "task_contract_digest",
        "data_contract_digest",
        "tensorizer",
        "sampling",
        "evaluation_sampling",
        "dataset_identity",
        "additional_evaluation_fraction",
        "split_seed",
        "source_case_ids",
        "source_case_materials",
        "case_partition",
        "case_partition_digest",
        "evaluation_assignments",
        "spatial_view",
        "roles",
        "runtime_provenance",
    }
    if not isinstance(value, Mapping):
        message = "Transient training split must be a mapping."
        raise TypeError(message)
    if (
        value.get("schema_kind") != SPLIT_SCHEMA_KIND
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != SPLIT_SCHEMA_VERSION
        or set(value) != base_required
    ):
        msg = "Historical or malformed transient split contract is unsupported."
        raise ValueError(msg)
    effective_seed = value["case_partition"]["seed"]
    if split_seed is not None and split_seed != effective_seed:
        msg_0 = "Expected split seed disagrees with Dataset fixed partition."
        raise ValueError(msg_0)
    split_seed = effective_seed
    admitted_spatial_view = _admit_spatial_view(value["spatial_view"], expected_stride=spatial_stride)
    task = domain.tasks.registry.get_task("transient_drying")
    expected_static = {
        "schema_kind": SPLIT_SCHEMA_KIND,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "data_contract_digest": task.data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": sampling.as_dict(),
        "evaluation_sampling": EVALUATION_SAMPLING.as_dict(),
        "additional_evaluation_fraction": _require_fraction(additional_evaluation_fraction),
        "split_seed": split_seed,
        "source_case_ids": list(value["source_case_ids"]),
        "source_case_materials": dict(value["source_case_materials"]),
        "case_partition": copy.deepcopy(dict(value["case_partition"])),
        "case_partition_digest": partition_contract.partition_digest(value["case_partition"]),
        "evaluation_assignments": copy.deepcopy(list(value["evaluation_assignments"])),
    }
    mismatches = [key for key, expected in expected_static.items() if value[key] != expected]
    if mismatches:
        message = f"Transient training split identity mismatch at {sorted(mismatches)}."
        raise ValueError(message)
    identity = value["dataset_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"train", "additional_evaluation"}:
        message = "Transient split dataset_identity has unexpected fields."
        raise ValueError(message)
    train_identity = _admit_dataset_identity(
        identity["train"],
        label="train",
    )
    raw_additional_evaluation_identity = identity["additional_evaluation"]
    if not isinstance(raw_additional_evaluation_identity, list):
        message = "Transient split additional evaluation Dataset identities must be a list."
        raise TypeError(message)
    additional_evaluation_identity = [
        _admit_dataset_identity(item, label=f"additional_evaluation[{index}]") for index, item in enumerate(raw_additional_evaluation_identity)
    ]

    runtime_provenance = value["runtime_provenance"]
    expected_provenance_keys = {"train", "scaling_train_one_step", "validation", "id_test", "additional_evaluation"}
    if not isinstance(runtime_provenance, Mapping) or set(runtime_provenance) != expected_provenance_keys:
        message = "Transient split runtime_provenance has unexpected fields."
        raise ValueError(message)
    scalar_roles = ("train", "scaling_train_one_step", "validation", "id_test")
    if any(runtime_provenance[role] not in {"canonical_hdf5", "pt_shards"} for role in scalar_roles):
        message = "Transient split runtime backend provenance is invalid."
        raise ValueError(message)
    additional_evaluation_backends = runtime_provenance["additional_evaluation"]
    if not isinstance(additional_evaluation_backends, list) or any(
        backend not in {"canonical_hdf5", "pt_shards"} for backend in additional_evaluation_backends
    ):
        message = "Transient split additional evaluation backend provenance is invalid."
        raise ValueError(message)

    roles = value["roles"]
    role_keys = {
        "train",
        "scaling_train_one_step",
        "validation",
        "id_test",
        "additional_evaluation",
    }
    if not isinstance(roles, Mapping) or set(roles) != role_keys:
        message = "Transient split role evidence has unexpected fields."
        raise ValueError(message)
    admitted_roles = {
        role: _admit_role_evidence(roles[role], label=role)
        for role in (
            "train",
            "scaling_train_one_step",
            "validation",
            "id_test",
        )
    }
    raw_additional_evaluation = roles["additional_evaluation"]
    if not isinstance(raw_additional_evaluation, Mapping) or set(raw_additional_evaluation) != {"parts"}:
        message = "Transient split additional evaluation role evidence is invalid."
        raise ValueError(message)
    raw_parts = raw_additional_evaluation["parts"]
    if not isinstance(raw_parts, list) or len(raw_parts) != len(additional_evaluation_identity):
        message = "Transient split additional evaluation role/Dataset counts disagree."
        raise ValueError(message)
    admitted_additional_evaluation = [
        _admit_role_evidence(item, label=f"additional_evaluation.parts[{index}]") for index, item in enumerate(raw_parts)
    ]
    if admitted_roles["train"]["case_ids"] != admitted_roles["scaling_train_one_step"]["case_ids"]:
        message = "Transient split Train and scaling case memberships differ."
        raise ValueError(message)
    expected_partition = partition_contract.validate_fixed_partition(
        value["case_partition"],
        case_ids=value["source_case_ids"],
        material_by_case=value["source_case_materials"],
    )
    if expected_partition["seed"] != split_seed:
        msg_0 = "Saved split seed disagrees with Dataset fixed partition."
        raise ValueError(msg_0)
    if (
        admitted_roles["train"]["case_ids"] != list(expected_partition["train"])
        or admitted_roles["validation"]["case_ids"] != list(expected_partition["validation"])
        or admitted_roles["id_test"]["case_ids"] != list(expected_partition["id_test"])
    ):
        msg = "Transient split roles disagree with the declared case partition."
        raise ValueError(msg)
    _admit_evaluation_assignments(
        value["evaluation_assignments"],
        [item["dataset_id"] for item in additional_evaluation_identity],
    )
    id_case_sets = [set(admitted_roles[role]["case_ids"]) for role in ("train", "validation", "id_test")]
    if any(left.intersection(right) for index, left in enumerate(id_case_sets) for right in id_case_sets[index + 1 :]):
        message = "Transient Train/validation/ID-test case memberships overlap."
        raise ValueError(message)
    result = {
        **expected_static,
        "dataset_identity": {
            "train": train_identity,
            "additional_evaluation": additional_evaluation_identity,
        },
        "roles": {
            **admitted_roles,
            "additional_evaluation": {"parts": admitted_additional_evaluation},
        },
        "runtime_provenance": {role: runtime_provenance[role] for role in (*scalar_roles, "additional_evaluation")},
    }
    result["spatial_view"] = admitted_spatial_view
    return result


def _scientific_split(value: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only runtime backend choice from replay-scientific identity."""
    result = dict(value)
    result.pop("runtime_provenance", None)
    return result


def _admit_saved_split(saved: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Reject scientific replay drift while retaining historical backend evidence."""
    if not isinstance(saved, Mapping):
        message = "saved_split must be a mapping."
        raise TypeError(message)
    tensorizer = TransientTensorizerSpec.from_mapping(
        {
            "input_profile": current["tensorizer"]["input_profile"],
            "temporal_conditioning": current["tensorizer"]["temporal_conditioning"],
        }
    )
    sampling = transient_contract.TransientSamplingSpec.from_mapping(current["sampling"])
    current_spatial_view = _admit_spatial_view(
        current.get("spatial_view"),
        expected_stride=current["spatial_view"]["spatial_stride"],
    )
    stride = current_spatial_view["spatial_stride"]
    materialized = admit_transient_training_split(
        saved,
        tensorizer=tensorizer,
        sampling=sampling,
        additional_evaluation_fraction=current["additional_evaluation_fraction"],
        split_seed=current["split_seed"],
        spatial_stride=stride,
    )
    admitted_current = admit_transient_training_split(
        current,
        tensorizer=tensorizer,
        sampling=sampling,
        additional_evaluation_fraction=current["additional_evaluation_fraction"],
        split_seed=current["split_seed"],
        spatial_stride=stride,
    )
    if _scientific_split(materialized) != _scientific_split(admitted_current):
        message = "Saved transient split does not exactly match current package, sampling, or membership evidence."
        raise ValueError(message)
    return materialized


def _batch_keys(dataset: Dataset[Any]) -> tuple[tuple[int, int], ...] | None:
    """Return per-item ``(part, transitions)`` keys when batches must not mix them, else ``None``."""
    trajectories = isinstance(dataset, transient.TransientPhysicalDataset) and dataset.sampling.mode == "trajectory"
    if not isinstance(dataset, ConcatDataset) and not trajectories:
        return None
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else (dataset,)
    keys: list[tuple[int, int]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, transient.TransientPhysicalDataset):
            msg = "Grouped transient batches require transient Dataset parts."
            raise TypeError(msg)
        keys.extend((index, count) for count in part.item_transition_counts())
    return tuple(keys)


def _loader(
    dataset: Dataset[Any],
    settings: factory.LoaderSettings,
    *,
    shuffle: bool,
    loader_seed: int,
    worker_seed: int,
) -> DataLoader[Any]:
    """Build one deterministic runtime loader whose batches never mix trajectory lengths or evaluation parts."""
    keys = _batch_keys(dataset)
    configured = factory.LoaderSettings(
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers if settings.num_workers else False,
        # Grouped batches own their shuffling.
        shuffle=shuffle and keys is None,
        hdf5_cache_size=settings.hdf5_cache_size,
    )
    generator = torch.Generator().manual_seed(loader_seed)
    # Whole trajectories end at their own drying stop, so only equal lengths of one part share a batch.
    batches: list[list[int]] | factory.GroupedBatchSampler | None = None
    if keys is not None:
        batches = (
            factory.GroupedBatchSampler(keys, configured.batch_size, generator)
            if shuffle
            else factory.group_batches(range(len(keys)), keys, configured.batch_size)
        )
    return factory.make_data_loader(
        dataset,
        configured,
        batches=batches,
        generator=generator,
        worker_init_fn=_worker_initializer(worker_seed),
    )


def _admit_evaluation_assignments(
    assignments: Sequence[Mapping[str, Any]],
    dataset_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Admit explicit experiment evaluation roles without inferring roles from Dataset names."""
    if len(assignments) != len(dataset_ids):
        msg = "Experiment evaluation assignments must align with selected Datasets."
        raise ValueError(msg)
    allowed = {"ood", "id", "near_family_ood", "far_family_ood", "parameter_ood", "schedule_ood", "stress"}
    result: list[dict[str, Any]] = []
    for index, (assignment, dataset_id) in enumerate(zip(assignments, dataset_ids, strict=True)):
        if not isinstance(assignment, Mapping) or set(assignment) != {"dataset_id", "role", "relation", "assumptions"}:
            msg = f"Experiment evaluation assignment {index} has invalid keys."
            raise ValueError(msg)
        if assignment["dataset_id"] != dataset_id or assignment["role"] not in allowed:
            msg = f"Experiment evaluation assignment {index} disagrees with selected Dataset."
            raise ValueError(msg)
        if not isinstance(assignment["relation"], str) or not assignment["relation"].strip():
            msg = "Experiment evaluation relation must explain the declared comparison."
            raise ValueError(msg)
        assumptions = assignment["assumptions"]
        if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip() for item in assumptions):
            msg = "Experiment evaluation assumptions must be a list of non-empty statements."
            raise ValueError(msg)
        result.append({"dataset_id": dataset_id, "role": assignment["role"], "relation": assignment["relation"], "assumptions": list(assumptions)})
    return result


def _validate_material_holdouts(
    train_dataset: transient.TransientPhysicalDataset,
    assignments: Sequence[Mapping[str, Any]],
    parts: Sequence[transient.TransientPhysicalDataset],
) -> None:
    """Reject same-material cases falsely declared as a material holdout."""
    train_materials = {
        str(case["material_family"]) for case in train_dataset.payload["cases"] if case["package_case_id"] in _case_positions(train_dataset)
    }
    for assignment, part in zip(assignments, parts, strict=True):
        if assignment["role"] in {"ood", "near_family_ood", "far_family_ood"}:
            materials = {str(case["material_family"]) for case in part.payload["cases"] if case["package_case_id"] in _case_positions(part)}
            if train_materials.intersection(materials):
                msg = "Declared family-OOD evaluation reuses a training material family."
                raise ValueError(msg)


def _dataset_fixed_partition(dataset: transient.TransientPhysicalDataset) -> dict[str, Any]:
    """Require the manifest-owned partition before expanding runtime items."""
    fixed = dataset.dataset_case_partition
    if fixed is None or not fixed["train"]:
        msg = "Training requires a Dataset-owned fixed ID partition; build the normal Dataset plan first."
        raise ValueError(msg)
    return partition_contract.validate_fixed_partition(
        fixed,
        case_ids=tuple(_case_positions(dataset)),
        material_by_case={str(case["package_case_id"]): str(case["material_family"]) for case in dataset.payload["cases"]},
    )


def create_transient_training_loaders(
    *,
    train_dataset_id: str,
    additional_evaluation_dataset_ids: Sequence[str],
    tensorizer: TransientTensorizerSpec,
    train_sampling: transient_contract.TransientSamplingSpec,
    loader_settings: factory.LoaderSettings,
    storage_root: str | None = None,
    transient_backend_preference: factory.TransientBackendPreference = "pt_shards",
    transient_backend_required: bool = False,
    spatial_stride: int = 1,
    scale_mode: scaling.TransientScaleMode = "state_std",
    additional_evaluation_fraction: float = 1.0,
    loader_seed: int | None = None,
    worker_seed: int | None = None,
    saved_split: Mapping[str, Any] | None = None,
    restored_scaling_artifact: scaling.TransientScalingArtifact | None = None,
    startup_callback: StartupCallback | None = None,
    split_persist_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> TransientTrainingLoaders:
    """
    Create experiment-role loaders and exact reusable Train-only scaling evidence.

    The Train, validation, and ID-test roles use immutable Dataset cases before
    their items are expanded. Train items follow ``train_sampling``; validation,
    ID-test, and additional-evaluation items are whole trajectories
    (``EVALUATION_SAMPLING``), so checkpoint selection does not depend on the
    training sample shape. Additional evaluation fractions
    likewise choose complete cases before runtime item expansion. Saved split
    evidence is replayed exactly. A restored scaler or integrity-bound cache hit
    avoids recomputing identical Train-only statistics.
    """
    if not isinstance(tensorizer, TransientTensorizerSpec):
        message = "tensorizer must be a TransientTensorizerSpec."
        raise TypeError(message)
    if not isinstance(train_sampling, transient_contract.TransientSamplingSpec):
        message = "train_sampling must be a TransientSamplingSpec."
        raise TypeError(message)
    resolved_spatial_stride = transient_contract.validate_spatial_stride(spatial_stride)
    fraction = _require_fraction(additional_evaluation_fraction)
    if fraction != 1.0:
        msg = "Dataset-owned OOD membership is fixed; additional_evaluation_fraction must remain 1.0."
        raise ValueError(msg)
    resolved_loader_seed = 9 if loader_seed is None else loader_seed
    resolved_worker_seed = resolved_loader_seed if worker_seed is None else worker_seed
    if isinstance(resolved_loader_seed, bool) or not isinstance(resolved_loader_seed, int):
        message = "loader_seed must be an integer."
        raise TypeError(message)
    if isinstance(resolved_worker_seed, bool) or not isinstance(resolved_worker_seed, int):
        message = "worker_seed must be an integer."
        raise TypeError(message)

    def dataset_for(
        dataset_id: str,
        *,
        sampling: transient_contract.TransientSamplingSpec | None = None,
    ) -> transient.TransientPhysicalDataset:
        request = factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view=_TASK,
            transient_sampling=train_sampling if sampling is None else sampling,
            storage_root=storage_root,
            transient_backend_preference=transient_backend_preference,
            transient_backend_required=transient_backend_required,
            spatial_stride=resolved_spatial_stride,
        )
        value = factory.create_dataset(request, hdf5_cache_size=loader_settings.hdf5_cache_size)
        if not isinstance(value, transient.TransientPhysicalDataset):
            msg = "Transient Dataset factory returned a non-transient Dataset."
            raise TypeError(msg)
        return value

    full_training_dataset = dataset_for(train_dataset_id)
    full_evaluation_dataset = dataset_for(train_dataset_id, sampling=EVALUATION_SAMPLING)
    source_case_materials = {str(case["package_case_id"]): str(case["material_family"]) for case in full_training_dataset.payload["cases"]}
    fixed = _dataset_fixed_partition(full_training_dataset)
    split_seed = fixed["seed"]
    case_partition = fixed
    partition = {role: tuple(fixed[role]) for role in ("train", "validation", "id_test")}
    train_dataset = _select_cases(full_training_dataset, partition["train"])
    # A Train case without any item would stay declared Train while never being trained on.
    uncovered = train_dataset.cases_without_items()
    if uncovered:
        msg = (
            f"Train cases {list(uncovered)} are shorter than the training sampling {train_sampling.as_dict()} and would be "
            "silently excluded; choose a sampling that fits them or exclude them explicitly in the Dataset plan."
        )
        raise ValueError(msg)
    evaluation_dataset = _select_cases(full_evaluation_dataset, partition["validation"])
    id_test_dataset = _select_cases(full_evaluation_dataset, partition["id_test"])
    additional_evaluation_ids = tuple(additional_evaluation_dataset_ids)
    if len(additional_evaluation_ids) != len(set(additional_evaluation_ids)):
        msg = "Experiment evaluation Dataset IDs must be unique."
        raise ValueError(msg)
    admitted_assignments = packages.declared_evaluation_assignments(_TASK, train_dataset_id, additional_evaluation_ids, storage_root=storage_root)
    additional_evaluation_parts = [dataset_for(dataset_id, sampling=EVALUATION_SAMPLING) for dataset_id in additional_evaluation_ids]
    selected_additional_evaluation_parts: list[transient.TransientPhysicalDataset] = []
    for source in additional_evaluation_parts:
        declared = source.dataset_case_partition
        if declared is None or not declared["ood"]:
            msg_0 = "Additional evaluation Dataset must declare its fixed OOD cases in the Dataset plan."
            raise ValueError(msg_0)
        selected_additional_evaluation_parts.append(_select_cases(source, declared["ood"]))
    if fixed["ood"]:
        selected_additional_evaluation_parts.append(_select_cases(full_evaluation_dataset, fixed["ood"]))
    _reject_role_leakage(train_dataset, evaluation_dataset, id_test_dataset, selected_additional_evaluation_parts)
    _validate_material_holdouts(train_dataset, admitted_assignments, selected_additional_evaluation_parts)
    additional_evaluation_dataset: Dataset[Any] | None = (
        selected_additional_evaluation_parts[0]
        if len(selected_additional_evaluation_parts) == 1
        else ConcatDataset(selected_additional_evaluation_parts)
        if selected_additional_evaluation_parts
        else None
    )

    one_step = transient_contract.TransientSamplingSpec(mode="one_step_transition")
    fitting_dataset = (
        train_dataset
        if train_sampling.mode == "one_step_transition"
        else _select_cases(dataset_for(train_dataset_id, sampling=one_step), partition["train"])
    )

    spatial_view = _spatial_view_evidence(train_dataset)
    view_datasets = (
        ("validation", evaluation_dataset),
        ("id_test", id_test_dataset),
        ("scaling_train_one_step", fitting_dataset),
        *((f"additional_evaluation[{index}]", part) for index, part in enumerate(selected_additional_evaluation_parts)),
    )
    for label, dataset in view_datasets:
        if _spatial_view_evidence(dataset) != spatial_view:
            message = f"Transient {label} Dataset does not share the Train spatial view."
            raise ValueError(message)

    _emit_startup(
        startup_callback,
        "dataset_resolved",
        dataset_id=train_dataset.dataset_id,
        backend=train_dataset.storage_backend,
        shards=(train_dataset.shard_count if isinstance(train_dataset, transient.TransientPTShardDataset) else None),
        transitions=len(train_dataset.sample_indices),
        canonical_grid=f"{spatial_view['canonical_ny']}x{spatial_view['canonical_nx']}",
        spatial_stride=spatial_view["spatial_stride"],
        effective_grid=f"{spatial_view['effective_ny']}x{spatial_view['effective_nx']}",
        roles="train,validation,id_test",
    )
    for part in selected_additional_evaluation_parts:
        _emit_startup(
            startup_callback,
            "dataset_resolved",
            dataset_id=part.dataset_id,
            backend=part.storage_backend,
            shards=(part.shard_count if isinstance(part, transient.TransientPTShardDataset) else None),
            transitions=len(part.sample_indices),
            canonical_grid=f"{spatial_view['canonical_ny']}x{spatial_view['canonical_nx']}",
            spatial_stride=spatial_view["spatial_stride"],
            effective_grid=f"{spatial_view['effective_ny']}x{spatial_view['effective_nx']}",
            roles="additional_evaluation",
        )

    train_identity = _dataset_identity(train_dataset)
    additional_evaluation_identities = [_dataset_identity(part) for part in selected_additional_evaluation_parts]
    identity: dict[str, Any] = {
        "train": train_identity,
        "additional_evaluation": additional_evaluation_identities,
    }
    current_split: dict[str, Any] = {
        "schema_kind": SPLIT_SCHEMA_KIND,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": "transient_drying",
        "task_contract_digest": domain.tasks.registry.get_task("transient_drying").contract_digest,
        "data_contract_digest": domain.tasks.registry.get_task("transient_drying").data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": train_sampling.as_dict(),
        "evaluation_sampling": EVALUATION_SAMPLING.as_dict(),
        "dataset_identity": identity,
        "additional_evaluation_fraction": fraction,
        "split_seed": split_seed,
        "source_case_ids": list(_case_positions(full_training_dataset)),
        "source_case_materials": source_case_materials,
        "case_partition": copy.deepcopy(dict(case_partition)),
        "case_partition_digest": partition_contract.partition_digest(case_partition),
        "evaluation_assignments": admitted_assignments,
        "spatial_view": spatial_view,
        "runtime_provenance": {
            "train": train_dataset.storage_backend,
            "scaling_train_one_step": fitting_dataset.storage_backend,
            "validation": evaluation_dataset.storage_backend,
            "id_test": id_test_dataset.storage_backend,
            "additional_evaluation": [part.storage_backend for part in selected_additional_evaluation_parts],
        },
        "roles": {
            "train": _role_evidence(train_dataset),
            "scaling_train_one_step": _role_evidence(fitting_dataset),
            "validation": _role_evidence(evaluation_dataset),
            "id_test": _role_evidence(id_test_dataset),
            "additional_evaluation": {"parts": [_role_evidence(part) for part in selected_additional_evaluation_parts]},
        },
    }
    split = admit_transient_training_split(
        current_split if saved_split is None else _admit_saved_split(saved_split, current_split),
        tensorizer=tensorizer,
        sampling=train_sampling,
        additional_evaluation_fraction=fraction,
        split_seed=split_seed,
        spatial_stride=resolved_spatial_stride,
    )
    if saved_split is None and split_persist_callback is not None:
        split_persist_callback(split)

    train_digest = str(current_split["roles"]["scaling_train_one_step"]["membership_digest"])
    scaling_identity = _scaling_dataset_identity(fitting_dataset, spatial_view)
    if restored_scaling_artifact is None:
        artifact = _fit_or_load_scaling_artifact(
            fitting_dataset,
            tensorizer=tensorizer,
            dataset_identity=scaling_identity,
            train_membership_digest=train_digest,
            scale_mode=scale_mode,
            spatial_view=spatial_view,
            storage_root=storage_root,
            startup_callback=startup_callback,
        )
    else:
        artifact = restored_scaling_artifact
        _validate_scaling_artifact(
            artifact,
            tensorizer=tensorizer,
            dataset_identity=scaling_identity,
            train_membership_digest=train_digest,
            scale_mode=scale_mode,
            spatial_view=spatial_view,
            horizon=fitting_dataset.configured_regular_horizon,
        )
        _emit_startup(
            startup_callback,
            "scaler_cache",
            state="hit",
            source="restored_run_or_stage_handoff",
        )

    loader_started = time.perf_counter()
    train_loader = _loader(
        train_dataset,
        loader_settings,
        shuffle=True,
        loader_seed=resolved_loader_seed,
        worker_seed=resolved_worker_seed,
    )
    evaluation_loader = _loader(
        evaluation_dataset,
        loader_settings,
        shuffle=False,
        loader_seed=resolved_loader_seed + 1,
        worker_seed=resolved_worker_seed + 1,
    )
    additional_evaluation_loader = (
        _loader(
            additional_evaluation_dataset,
            loader_settings,
            shuffle=False,
            loader_seed=resolved_loader_seed + 2,
            worker_seed=resolved_worker_seed + 2,
        )
        if additional_evaluation_dataset is not None
        else None
    )
    id_test_loader = _loader(
        id_test_dataset,
        loader_settings,
        shuffle=False,
        loader_seed=resolved_loader_seed + 3,
        worker_seed=resolved_worker_seed + 3,
    )
    _emit_startup(
        startup_callback,
        "dataloader_ready",
        workers=loader_settings.num_workers,
        pin_memory=loader_settings.pin_memory,
        persistent_workers=loader_settings.persistent_workers,
        seconds=time.perf_counter() - loader_started,
    )
    return TransientTrainingLoaders(
        train=train_loader,
        evaluation=evaluation_loader,
        additional_evaluation=additional_evaluation_loader,
        id_test=id_test_loader,
        scaling_artifact=artifact,
        split=split,
        dataset_identity=identity,
        spatial_view=copy.deepcopy(spatial_view),
        runtime_provenance=copy.deepcopy(current_split["runtime_provenance"]),
    )


_resolve_case_partition = partition_contract.resolve_case_partition


def _physical_case_ids(dataset: transient.TransientPhysicalDataset) -> set[tuple[str, str]]:
    """Bind selected cases to physical inputs and simulations across aliases."""
    selected = set(_case_positions(dataset))
    return {(str(case["case_input_id"]), str(case["simulation_case_id"])) for case in dataset.payload["cases"] if case["package_case_id"] in selected}


def _reject_role_leakage(
    train: transient.TransientPhysicalDataset,
    validation: transient.TransientPhysicalDataset,
    test: transient.TransientPhysicalDataset,
    additional_evaluation: Sequence[transient.TransientPhysicalDataset],
) -> None:
    """Reject any repeated physical source case across experiment roles."""
    seen: set[str] = set()
    role_datasets = (
        ("train", train),
        ("validation", validation),
        ("id_test", test),
        *((f"additional_evaluation[{index}]", item) for index, item in enumerate(additional_evaluation)),
    )
    for label, dataset in role_datasets:
        identities = _physical_case_ids(dataset)
        tokens = {token for pair in identities for token in pair}
        if seen.intersection(tokens):
            msg = f"Experiment {label} reuses a physical source case from another role."
            raise ValueError(msg)
        seen.update(tokens)
