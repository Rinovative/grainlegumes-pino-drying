"""
dataset_runtime_transient_training.py

Construct storage-neutral transient training loaders from published packages.

Responsibilities:
  - Select package-owned transient case roles without splitting transitions
  - Persist and replay strict case and item membership evidence
  - Fit transient scaling from one-step Train evidence only
  - Build deterministically seeded training and evaluation DataLoaders

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
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.runtime import dataset_runtime_factory as factory
from src.datasets.runtime import dataset_runtime_transient as transient
from src.learning.transient import learning_transient_scaling as scaling
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from pathlib import Path

_SPLIT_SCHEMA_KIND: Final = "transient_drying_training_split"
_SPLIT_SCHEMA_VERSION: Final = 2
_LEGACY_SPLIT_SCHEMA_VERSION: Final = 1
_SCALER_CACHE_SCHEMA_KIND: Final = "transient_drying_scaler_cache"
_SCALER_CACHE_SCHEMA_VERSION: Final = 1
_SHA256_LENGTH: Final = 64
_MINIMUM_SPATIAL_AXIS_LENGTH: Final = 2
StartupCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TransientTrainingLoaders:
    """Return loaders and immutable evidence for one transient training run."""

    train: DataLoader[Any]
    evaluation: DataLoader[Any]
    ood: DataLoader[Any]
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
    """Validate the requested case-owned OOD selection fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        message = "ood_fraction must be one finite fraction in (0, 1]."
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
    legacy_dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    scale_mode: scaling.TransientScaleMode,
    spatial_view: Mapping[str, int],
    horizon: float,
    allow_legacy_stride_one: bool,
) -> None:
    """Require exact scaler semantics, optionally admitting legacy stride one."""
    expected_identity = dict(dataset_identity)
    identity_matches = artifact.dataset_identity == expected_identity
    legacy_matches = allow_legacy_stride_one and spatial_view["spatial_stride"] == 1 and artifact.dataset_identity == dict(legacy_dataset_identity)
    expected_shape = (
        spatial_view["effective_ny"],
        spatial_view["effective_nx"],
    )
    if (
        artifact.tensorizer != tensorizer
        or not (identity_matches or legacy_matches)
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
    legacy_dataset_identity: Mapping[str, Any],
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
        legacy_dataset_identity=legacy_dataset_identity,
        train_membership_digest=train_membership_digest,
        scale_mode=scale_mode,
        spatial_view=spatial_view,
        horizon=horizon,
        allow_legacy_stride_one=False,
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
    legacy_dataset_identity: Mapping[str, Any],
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
                    legacy_dataset_identity=legacy_dataset_identity,
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
    ood_fraction: float,
    split_seed: int,
    spatial_stride: int = 1,
) -> dict[str, Any]:
    """Admit one current or stride-one legacy transient training split."""
    base_required = {
        "schema_kind",
        "schema_version",
        "task",
        "task_contract_digest",
        "data_contract_digest",
        "tensorizer",
        "sampling",
        "dataset_identity",
        "ood_fraction",
        "split_seed",
        "roles",
        "runtime_provenance",
    }
    if not isinstance(value, Mapping):
        message = "Transient training split must be a mapping."
        raise TypeError(message)
    version = value.get("schema_version")
    if version == _LEGACY_SPLIT_SCHEMA_VERSION:
        if transient_contract.validate_spatial_stride(spatial_stride) != 1:
            message = "Legacy transient split evidence is valid only for spatial_stride=1."
            raise ValueError(message)
        required = base_required
        admitted_spatial_view = None
    elif version == _SPLIT_SCHEMA_VERSION:
        required = base_required.union({"spatial_view"})
        admitted_spatial_view = _admit_spatial_view(
            value.get("spatial_view"),
            expected_stride=spatial_stride,
        )
    else:
        message = f"Transient training split schema version {version!r} is unsupported."
        raise ValueError(message)
    if set(value) != required:
        message = "Transient training split keys do not match the schema."
        raise ValueError(message)
    task = domain.tasks.registry.get_task("transient_drying")
    expected_static = {
        "schema_kind": _SPLIT_SCHEMA_KIND,
        "schema_version": version,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "data_contract_digest": task.data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": sampling.as_dict(),
        "ood_fraction": _require_fraction(ood_fraction),
        "split_seed": split_seed,
    }
    mismatches = [key for key, expected in expected_static.items() if value[key] != expected]
    if mismatches:
        message = f"Transient training split identity mismatch at {sorted(mismatches)}."
        raise ValueError(message)
    identity = value["dataset_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"train", "ood"}:
        message = "Transient split dataset_identity has unexpected fields."
        raise ValueError(message)
    train_identity = _admit_dataset_identity(
        identity["train"],
        label="train",
    )
    raw_ood_identity = identity["ood"]
    if not isinstance(raw_ood_identity, list) or not raw_ood_identity:
        message = "Transient split requires OOD package identities."
        raise ValueError(message)
    ood_identity = [_admit_dataset_identity(item, label=f"ood[{index}]") for index, item in enumerate(raw_ood_identity)]

    runtime_provenance = value["runtime_provenance"]
    expected_provenance_keys = {"train", "scaling_train_one_step", "evaluation", "id_test", "ood"}
    if not isinstance(runtime_provenance, Mapping) or set(runtime_provenance) != expected_provenance_keys:
        message = "Transient split runtime_provenance has unexpected fields."
        raise ValueError(message)
    scalar_roles = ("train", "scaling_train_one_step", "evaluation", "id_test")
    if any(runtime_provenance[role] not in {"canonical_hdf5", "pt_shards"} for role in scalar_roles):
        message = "Transient split runtime backend provenance is invalid."
        raise ValueError(message)
    ood_backends = runtime_provenance["ood"]
    if not isinstance(ood_backends, list) or not ood_backends or any(backend not in {"canonical_hdf5", "pt_shards"} for backend in ood_backends):
        message = "Transient split OOD backend provenance is invalid."
        raise ValueError(message)

    roles = value["roles"]
    role_keys = {
        "train",
        "scaling_train_one_step",
        "evaluation",
        "id_test",
        "ood",
    }
    if not isinstance(roles, Mapping) or set(roles) != role_keys:
        message = "Transient split role evidence has unexpected fields."
        raise ValueError(message)
    admitted_roles = {
        role: _admit_role_evidence(roles[role], label=role)
        for role in (
            "train",
            "scaling_train_one_step",
            "evaluation",
            "id_test",
        )
    }
    raw_ood = roles["ood"]
    if not isinstance(raw_ood, Mapping) or set(raw_ood) != {"parts"}:
        message = "Transient split OOD role evidence is invalid."
        raise ValueError(message)
    raw_parts = raw_ood["parts"]
    if not isinstance(raw_parts, list) or len(raw_parts) != len(ood_identity):
        message = "Transient split OOD role/package counts disagree."
        raise ValueError(message)
    admitted_ood = [_admit_role_evidence(item, label=f"ood.parts[{index}]") for index, item in enumerate(raw_parts)]
    if admitted_roles["train"]["case_ids"] != admitted_roles["scaling_train_one_step"]["case_ids"]:
        message = "Transient split Train and scaling case memberships differ."
        raise ValueError(message)
    id_case_sets = [set(admitted_roles[role]["case_ids"]) for role in ("train", "evaluation", "id_test")]
    if any(left.intersection(right) for index, left in enumerate(id_case_sets) for right in id_case_sets[index + 1 :]):
        message = "Transient Train/evaluation/ID-test case memberships overlap."
        raise ValueError(message)
    result = {
        **expected_static,
        "dataset_identity": {
            "train": train_identity,
            "ood": ood_identity,
        },
        "roles": {
            **admitted_roles,
            "ood": {"parts": admitted_ood},
        },
        "runtime_provenance": {role: runtime_provenance[role] for role in (*scalar_roles, "ood")},
    }
    if admitted_spatial_view is not None:
        result["spatial_view"] = admitted_spatial_view
    return result


def _scientific_split(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return replay evidence normalized across legacy stride-one storage."""
    result = dict(value)
    result.pop("runtime_provenance", None)
    spatial_view = result.get("spatial_view")
    if isinstance(spatial_view, Mapping) and spatial_view.get("spatial_stride") == 1:
        result.pop("spatial_view")
        result["schema_version"] = _LEGACY_SPLIT_SCHEMA_VERSION
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
        ood_fraction=current["ood_fraction"],
        split_seed=current["split_seed"],
        spatial_stride=stride,
    )
    admitted_current = admit_transient_training_split(
        current,
        tensorizer=tensorizer,
        sampling=sampling,
        ood_fraction=current["ood_fraction"],
        split_seed=current["split_seed"],
        spatial_stride=stride,
    )
    if _scientific_split(materialized) != _scientific_split(admitted_current):
        message = "Saved transient split does not exactly match current package, sampling, or membership evidence."
        raise ValueError(message)
    return materialized


def _loader(
    dataset: Dataset[Any],
    settings: factory.LoaderSettings,
    *,
    shuffle: bool,
    loader_seed: int,
    worker_seed: int,
) -> DataLoader[Any]:
    """Build one deterministic runtime loader using the shared factory."""
    configured = factory.LoaderSettings(
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers if settings.num_workers else False,
        prefetch_factor=settings.prefetch_factor if settings.num_workers else None,
        shuffle=shuffle,
        drop_last=settings.drop_last if shuffle else False,
        hdf5_cache_size=settings.hdf5_cache_size,
    )
    return factory.make_data_loader(
        dataset,
        configured,
        generator=torch.Generator().manual_seed(loader_seed),
        worker_init_fn=_worker_initializer(worker_seed),
    )


def _ood_package_regime(
    dataset_id: str,
    *,
    storage_root: str | None,
) -> views.PackageRegime:
    """Return the immutable non-ID regime owned by one transient package manifest."""
    manifest = package_manifest.load_package_manifest(
        dataset_id,
        storage_root=storage_root,
    )
    if manifest["dataset_view"] != "transient_drying":
        message = f"Transient Training OOD dataset {dataset_id!r} has incompatible view {manifest['dataset_view']!r}."
        raise ValueError(message)
    regime = cast("views.PackageRegime", manifest["evaluation_regime"])
    if regime == "id":
        message = f"Transient Training OOD dataset {dataset_id!r} resolves to an ID package."
        raise ValueError(message)
    return regime


def create_transient_training_loaders(
    *,
    train_dataset_id: str,
    ood_dataset_ids: str | Sequence[str],
    tensorizer: TransientTensorizerSpec,
    train_sampling: transient_contract.TransientSamplingSpec,
    loader_settings: factory.LoaderSettings,
    storage_root: str | None = None,
    transient_backend_preference: factory.TransientBackendPreference = "pt_shards",
    transient_backend_required: bool = False,
    spatial_stride: int = 1,
    scale_mode: scaling.TransientScaleMode = "state_std",
    ood_fraction: float = 1.0,
    split_seed: int = 9,
    loader_seed: int | None = None,
    worker_seed: int | None = None,
    saved_split: Mapping[str, Any] | None = None,
    restored_scaling_artifact: scaling.TransientScalingArtifact | None = None,
    allow_technical_smoke: bool = False,
    startup_callback: StartupCallback | None = None,
) -> TransientTrainingLoaders:
    """
    Create package-role loaders and exact reusable Train-only scaling evidence.

    The Train, validation, and ID-test roles are selected by package case before
    either one-step transitions or rollout windows are expanded. OOD fractions
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
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        message = "split_seed must be an integer."
        raise TypeError(message)
    resolved_spatial_stride = transient_contract.validate_spatial_stride(spatial_stride)
    fraction = _require_fraction(ood_fraction)
    resolved_loader_seed = split_seed if loader_seed is None else loader_seed
    resolved_worker_seed = resolved_loader_seed if worker_seed is None else worker_seed
    if isinstance(resolved_loader_seed, bool) or not isinstance(resolved_loader_seed, int):
        message = "loader_seed must be an integer."
        raise TypeError(message)
    if isinstance(resolved_worker_seed, bool) or not isinstance(resolved_worker_seed, int):
        message = "worker_seed must be an integer."
        raise TypeError(message)

    def request(
        dataset_id: str,
        regime: views.PackageRegime,
        membership: str | None = None,
        *,
        sampling: transient_contract.TransientSamplingSpec | None = None,
    ) -> factory.DatasetRequest:
        return factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view="transient_drying",
            evaluation_regime=regime,
            membership=membership,  # type: ignore[arg-type]
            transient_sampling=train_sampling if sampling is None else sampling,
            storage_root=storage_root,
            allow_technical_smoke=allow_technical_smoke,
            transient_backend_preference=transient_backend_preference,
            transient_backend_required=transient_backend_required,
            spatial_stride=resolved_spatial_stride,
        )

    def dataset_for(
        dataset_id: str,
        regime: views.PackageRegime,
        membership: str | None = None,
        *,
        sampling: transient_contract.TransientSamplingSpec | None = None,
    ) -> transient.TransientPhysicalDataset:
        value = factory.create_dataset(
            request(
                dataset_id,
                regime,
                membership,
                sampling=sampling,
            ),
            hdf5_cache_size=loader_settings.hdf5_cache_size,
        )
        if not isinstance(value, transient.TransientPhysicalDataset):
            message = "Transient Dataset factory returned a non-transient Dataset."
            raise TypeError(message)
        return value

    train_dataset = dataset_for(train_dataset_id, "id", "train")
    evaluation_dataset = dataset_for(train_dataset_id, "id", "validation")
    id_test_dataset = dataset_for(train_dataset_id, "id", "id_test")
    ood_ids = (ood_dataset_ids,) if isinstance(ood_dataset_ids, str) else tuple(ood_dataset_ids)
    if not ood_ids:
        message = "ood_dataset_ids must be non-empty."
        raise ValueError(message)
    ood_parts = [
        dataset_for(
            dataset_id,
            _ood_package_regime(dataset_id, storage_root=storage_root),
        )
        for dataset_id in ood_ids
    ]
    selected_ood_parts: list[transient.TransientPhysicalDataset] = []
    for offset, source in enumerate(ood_parts):
        case_ids = tuple(_case_positions(source))
        count = int(len(case_ids) * fraction)
        if count < 1:
            message = "ood_fraction selects no complete OOD cases."
            raise ValueError(message)
        shuffled = list(case_ids)
        random.Random(split_seed + offset).shuffle(shuffled)  # noqa: S311 -- reproducible experimental case selection
        selected_ood_parts.append(_select_cases(source, tuple(shuffled[:count])))
    ood_dataset: Dataset[Any] = selected_ood_parts[0] if len(selected_ood_parts) == 1 else ConcatDataset(selected_ood_parts)

    one_step = transient_contract.TransientSamplingSpec(mode="one_step_transition")
    fitting_dataset = (
        train_dataset
        if train_sampling.mode == "one_step_transition"
        else dataset_for(
            train_dataset_id,
            "id",
            "train",
            sampling=one_step,
        )
    )

    spatial_view = _spatial_view_evidence(train_dataset)
    view_datasets = (
        ("evaluation", evaluation_dataset),
        ("id_test", id_test_dataset),
        ("scaling_train_one_step", fitting_dataset),
        *((f"ood[{index}]", part) for index, part in enumerate(selected_ood_parts)),
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
    for part in selected_ood_parts:
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
            roles="ood",
        )

    train_identity = _dataset_identity(train_dataset)
    ood_identities = [_dataset_identity(part) for part in selected_ood_parts]
    identity: dict[str, Any] = {
        "train": train_identity,
        "ood": ood_identities,
    }
    current_split: dict[str, Any] = {
        "schema_kind": _SPLIT_SCHEMA_KIND,
        "schema_version": _SPLIT_SCHEMA_VERSION,
        "task": "transient_drying",
        "task_contract_digest": domain.tasks.registry.get_task("transient_drying").contract_digest,
        "data_contract_digest": domain.tasks.registry.get_task("transient_drying").data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": train_sampling.as_dict(),
        "dataset_identity": identity,
        "ood_fraction": fraction,
        "split_seed": split_seed,
        "spatial_view": spatial_view,
        "runtime_provenance": {
            "train": train_dataset.storage_backend,
            "scaling_train_one_step": fitting_dataset.storage_backend,
            "evaluation": evaluation_dataset.storage_backend,
            "id_test": id_test_dataset.storage_backend,
            "ood": [part.storage_backend for part in selected_ood_parts],
        },
        "roles": {
            "train": _role_evidence(train_dataset),
            "scaling_train_one_step": _role_evidence(fitting_dataset),
            "evaluation": _role_evidence(evaluation_dataset),
            "id_test": _role_evidence(id_test_dataset),
            "ood": {"parts": [_role_evidence(part) for part in selected_ood_parts]},
        },
    }
    split = current_split if saved_split is None else _admit_saved_split(saved_split, current_split)

    train_digest = str(current_split["roles"]["scaling_train_one_step"]["membership_digest"])
    scaling_identity = _scaling_dataset_identity(fitting_dataset, spatial_view)
    if restored_scaling_artifact is None:
        artifact = _fit_or_load_scaling_artifact(
            fitting_dataset,
            tensorizer=tensorizer,
            dataset_identity=scaling_identity,
            legacy_dataset_identity=train_identity,
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
            legacy_dataset_identity=train_identity,
            train_membership_digest=train_digest,
            scale_mode=scale_mode,
            spatial_view=spatial_view,
            horizon=fitting_dataset.configured_regular_horizon,
            allow_legacy_stride_one=True,
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
    ood_loader = _loader(
        ood_dataset,
        loader_settings,
        shuffle=False,
        loader_seed=resolved_loader_seed + 2,
        worker_seed=resolved_worker_seed + 2,
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
        ood=ood_loader,
        id_test=id_test_loader,
        scaling_artifact=artifact,
        split=split,
        dataset_identity=identity,
        spatial_view=copy.deepcopy(spatial_view),
        runtime_provenance=copy.deepcopy(current_split["runtime_provenance"]),
    )
