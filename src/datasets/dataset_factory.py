"""
===============================================================================
dataset_factory.py
===============================================================================
Resolve typed dataset packages, selectors, runtime objects, and DataLoaders.
Responsibilities:
  - Load the sole steady tensor contract and transient lazy HDF5 contract
  - Apply explicit ID-membership and parameter-OOD group selectors
  - Validate worker, sampler, shuffle, prefetch, and HDF5-cache settings
Design principles:
  - Package identity and ordering remain independent of DataLoader behavior
  - Worker-only options are passed only when worker processes exist
  - Transient values remain physical and expose one optional transform hook
This module does NOT:
  - Fit normalization, choose experiment splits, register tasks, or train models
  - Add transient normalization, distributed training, or storage backends
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from src import common, domain

from . import dataset_identity as identity
from . import dataset_packages as packages
from . import dataset_transient as transient
from . import dataset_views as views

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class SteadyFlowMetadata(TypedDict, total=False):
    """Describe the source identity required on new steady package items."""

    dataset_id: str
    simulation_case_id: str
    case_input_id: str
    source_batch_id: str
    source_simulation_profile: str
    material_family: str
    evaluation_regime: str
    dataset_membership: str
    source_hdf5_sha256: str
    generator: dict[str, Any]


class SteadyFlowItem(TypedDict):
    """Expose one steady input/target pair and isolated source metadata."""

    x: torch.Tensor
    y: torch.Tensor
    meta: SteadyFlowMetadata


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Select one immutable package view, regime, and optional submembership."""

    dataset_id: str
    dataset_view: views.DatasetViewId
    evaluation_regime: views.PackageRegime
    membership: views.IdMembership | None = None
    ood_group: views.OodGroup | None = None
    storage_root: Path | str | None = None

    def __post_init__(self) -> None:
        """Reject selectors that are ambiguous or invalid for their regime."""
        common.paths.validate_logical_name(self.dataset_id, label="dataset_id")
        views.get_view(self.dataset_view)
        if self.membership is not None and self.membership not in views.ID_MEMBERSHIPS:
            message = f"Unsupported ID membership selector: {self.membership!r}."
            raise ValueError(message)
        if self.ood_group is not None and self.ood_group not in views.OOD_GROUPS:
            message = f"Unsupported parameter-OOD group selector: {self.ood_group!r}."
            raise ValueError(message)
        if self.evaluation_regime not in views.PACKAGE_REGIMES:
            message = f"Unsupported package regime: {self.evaluation_regime!r}."
            raise ValueError(message)
        if self.evaluation_regime == "id":
            if self.ood_group is not None:
                message = "ID package selection cannot include an OOD group."
                raise ValueError(message)
        elif self.membership is not None:
            message = "OOD package selection cannot include an ID membership."
            raise ValueError(message)
        if self.evaluation_regime != "parameter_ood" and self.ood_group is not None:
            message = "OOD group selectors apply only to parameter_ood packages."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class LoaderSettings:
    """Validate the common DataLoader and bounded HDF5-cache surface."""

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    shuffle: bool = False
    drop_last: bool = False
    hdf5_cache_size: int = 0

    def __post_init__(self) -> None:
        """Reject invalid or worker-inapplicable runtime settings."""
        for label, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("num_workers", self.num_workers, 0),
            ("hdf5_cache_size", self.hdf5_cache_size, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                message = f"{label} must be an integer >= {minimum}."
                raise ValueError(message)
        if not all(isinstance(value, bool) for value in (self.pin_memory, self.persistent_workers, self.shuffle, self.drop_last)):
            message = "pin_memory, persistent_workers, shuffle, and drop_last must be boolean."
            raise TypeError(message)
        if self.num_workers == 0 and (self.persistent_workers or self.prefetch_factor is not None):
            message = "persistent_workers and prefetch_factor require num_workers > 0."
            raise ValueError(message)
        if self.prefetch_factor is not None and (
            isinstance(self.prefetch_factor, bool) or not isinstance(self.prefetch_factor, int) or self.prefetch_factor < 1
        ):
            message = "prefetch_factor must be a positive integer or None."
            raise ValueError(message)


class SteadyFlowDataset(Dataset[SteadyFlowItem]):
    """Load and verify one steady tensor payload in authoritative task order."""

    def __init__(self, data_path: Path | str, *, task: TaskSpec) -> None:
        """Load one exact payload and verify its complete content fingerprint."""
        path = Path(data_path)
        if not path.is_file() or path.is_symlink():
            message = f"Steady training dataset is missing or unsafe: {path}."
            raise FileNotFoundError(message)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            message = f"Steady training dataset must contain one dictionary payload: {path}."
            raise TypeError(message)
        self.path = path
        self.task = task
        self.input_fields = list(task.input_names)
        self.output_fields = list(task.output_names)
        self.data = payload
        self.identity = identity.validate_training_dataset_payload(
            payload,
            task=task,
            verify_content=True,
        )
        self.inputs = payload["inputs"]
        self.outputs = payload["outputs"]

    def __len__(self) -> int:
        """Return the verified ordered steady sample count."""
        return self.identity.sample_count

    def __getitem__(self, index: int) -> SteadyFlowItem:
        """Return one task-ordered pair and a defensive metadata copy."""
        raw_meta = self.data["source_metadata"][index]
        if not isinstance(raw_meta, Mapping):
            message = f"Steady source_metadata[{index}] must be a mapping: {self.path}."
            raise TypeError(message)
        return {
            "x": self.inputs[index],
            "y": self.outputs[index],
            "meta": cast("SteadyFlowMetadata", deepcopy(dict(raw_meta))),
        }


class SteadyDatasetSelection(Dataset[SteadyFlowItem]):
    """Expose one identity-bound steady case selection without tensor copies."""

    def __init__(
        self,
        source: SteadyFlowDataset,
        indices: Sequence[int],
        *,
        selector: str,
    ) -> None:
        """Validate an ordered unique selection and derive its exact identity."""
        selected = tuple(indices)
        if not selected or len(selected) != len(set(selected)) or min(selected) < 0 or max(selected) >= len(source):
            message = f"Steady selector {selector!r} must resolve non-empty unique in-range positions."
            raise ValueError(message)
        self.source = source
        self.indices = selected
        self.input_fields = source.input_fields
        self.output_fields = source.output_fields
        sample_ids = tuple(source.identity.sample_ids[index] for index in selected)
        fingerprint_payload = json.dumps(
            {
                "source_fingerprint": source.identity.fingerprint,
                "selector": selector,
                "sample_ids": sample_ids,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        source_metadata = source.identity.source_metadata
        self.identity = identity.DatasetIdentity(
            dataset_id=source.identity.dataset_id,
            task=source.identity.task,
            data_contract_digest=source.identity.data_contract_digest,
            fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=source.identity.spatial_shape,
            source_metadata=(None if source_metadata is None else tuple(source_metadata[index] for index in selected)),
            source_provenance=source.identity.source_provenance,
        )

    def __len__(self) -> int:
        """Return selected case count."""
        return len(self.indices)

    def __getitem__(self, index: int) -> SteadyFlowItem:
        """Return one selected source sample."""
        return self.source[self.indices[index]]


def create_steady_dataset(
    data_path: Path | str,
    *,
    task: TaskSpec | None = None,
) -> SteadyFlowDataset:
    """Construct the sole verified steady runtime dataset implementation."""
    resolved_task = domain.tasks.registry.get_task("steady_flow") if task is None else task
    return SteadyFlowDataset(data_path, task=resolved_task)


def _payload_path(manifest: Mapping[str, Any], *, storage_root: Path | str | None) -> Path:
    """Resolve a manifest-bound payload under the dataset lifecycle root."""
    return common.paths.get_dataset_payload_root(storage_root=storage_root) / str(manifest["dataset_id"]) / str(manifest["payload_filename"])


def _case_positions(
    dataset: SteadyFlowDataset,
    package_case_ids: Sequence[str],
    *,
    selector: str,
) -> tuple[int, ...]:
    """Resolve package case IDs to ordered steady payload positions."""
    position_by_id = {sample_id: index for index, sample_id in enumerate(dataset.identity.sample_ids)}
    if len(position_by_id) != len(dataset.identity.sample_ids):
        message = "Steady dataset sample identities are duplicated."
        raise ValueError(message)
    try:
        positions = tuple(position_by_id[case_id] for case_id in package_case_ids)
    except KeyError as error:
        message = f"Manifest selector {selector!r} references a missing steady sample {error.args[0]!r}."
        raise ValueError(message) from error
    if not positions:
        message = f"Manifest selector {selector!r} is unavailable in this steady package."
        raise ValueError(message)
    return positions


def _select_steady(
    dataset: SteadyFlowDataset,
    manifest: Mapping[str, Any],
    request: DatasetRequest,
) -> Dataset[SteadyFlowItem]:
    """Apply an explicit manifest-owned steady membership or group selector."""
    package_case_ids: Sequence[str] | None = None
    selector = "all"
    if request.evaluation_regime == "id" and request.membership is not None:
        split_membership = manifest["split_membership"]
        package_case_ids = split_membership.get(request.membership)
        selector = f"id/{request.membership}"
    elif request.evaluation_regime == "parameter_ood" and request.ood_group is not None:
        available = manifest["available_ood_groups"]
        if request.ood_group not in available:
            message = f"Steady-flow parameter-OOD group {request.ood_group!r} is unavailable; available selectors are {available}."
            raise ValueError(message)
        package_case_ids = manifest["ood_group_indexes"].get(request.ood_group)
        selector = f"parameter_ood/{request.ood_group}"
    if package_case_ids is None:
        return dataset
    if not isinstance(package_case_ids, list) or not all(isinstance(value, str) for value in package_case_ids):
        message = f"Manifest selector {selector!r} is malformed."
        raise TypeError(message)
    return SteadyDatasetSelection(
        dataset,
        _case_positions(dataset, package_case_ids, selector=selector),
        selector=selector,
    )


def _select_transient(
    dataset: transient.TransientPhysicalDataset,
    request: DatasetRequest,
) -> transient.TransientPhysicalDataset:
    """Return a transient dataset with case-owned membership/group positions."""
    membership = request.membership if request.evaluation_regime == "id" else None
    ood_group = request.ood_group if request.evaluation_regime == "parameter_ood" else None
    if membership is None and ood_group is None:
        return dataset
    indices = transient.select_transient_sample_indices(
        dataset.payload,
        membership=membership,
        ood_group=ood_group,
    )
    dataset.close()
    return transient.TransientPhysicalDataset(
        dataset.index_path,
        source_root=dataset.source_root,
        hdf5_cache_size=dataset.hdf5_cache_size,
        sample_indices=indices,
        transform=dataset.transform,
    )


def create_dataset(
    request: DatasetRequest,
    *,
    hdf5_cache_size: int = 0,
    transient_transform: transient.TransientTransform | None = None,
) -> Dataset[SteadyFlowItem] | transient.TransientPhysicalDataset:
    """Resolve, validate, and select one package-backed runtime dataset."""
    manifest = packages.load_package_manifest(
        request.dataset_id,
        storage_root=request.storage_root,
    )
    if manifest["dataset_view"] != request.dataset_view or manifest["evaluation_regime"] != request.evaluation_regime:
        message = (
            f"Dataset request {request.dataset_view!r}/{request.evaluation_regime!r} "
            f"does not match package {manifest['dataset_view']!r}/{manifest['evaluation_regime']!r}."
        )
        raise ValueError(message)
    payload_path = _payload_path(manifest, storage_root=request.storage_root)
    if request.dataset_view == "steady_flow":
        if transient_transform is not None:
            message = "Transient transforms cannot be applied to steady-flow datasets."
            raise ValueError(message)
        steady_dataset = create_steady_dataset(payload_path)
        return _select_steady(steady_dataset, manifest, request)
    if request.ood_group is not None and request.ood_group not in manifest["available_ood_groups"]:
        message = f"Transient parameter-OOD group {request.ood_group!r} is unavailable; available selectors are {manifest['available_ood_groups']}."
        raise ValueError(message)
    transient_dataset = transient.TransientPhysicalDataset(
        payload_path,
        source_root=request.storage_root,
        hdf5_cache_size=hdf5_cache_size,
        transform=transient_transform,
    )
    if (
        transient_dataset.payload["dataset_id"] != manifest["dataset_id"]
        or transient_dataset.payload["evaluation_regime"] != manifest["evaluation_regime"]
        or transient_dataset.payload["contract_digest"] != manifest["channel_contract_digest"]
        or transient_dataset.payload["sample_count"] != manifest["sample_count"]
    ):
        message = f"Transient index does not bind its package manifest: {payload_path}."
        raise ValueError(message)
    return _select_transient(transient_dataset, request)


def _collate_steady_metadata(values: Sequence[Any]) -> Any:
    """Collate nested steady metadata while preserving explicit absent values."""
    try:
        return default_collate(list(values))
    except (TypeError, RuntimeError):
        pass
    if values and all(isinstance(value, Mapping) for value in values):
        mappings = [cast("Mapping[str, Any]", value) for value in values]
        keys = tuple(mappings[0])
        if all(tuple(value) == keys for value in mappings[1:]):
            return {key: _collate_steady_metadata([value[key] for value in mappings]) for key in keys}
    if values and all(isinstance(value, Sequence) and not isinstance(value, str | bytes) for value in values):
        sequences = [cast("Sequence[Any]", value) for value in values]
        lengths = {len(value) for value in sequences}
        if len(lengths) == 1:
            return [_collate_steady_metadata([value[index] for value in sequences]) for index in range(len(sequences[0]))]
    return deepcopy(list(values))


def _collate_runtime_batch(batch: list[Any]) -> Any:
    """Use default collation, with a narrow fallback for rich steady metadata."""
    try:
        return default_collate(batch)
    except (TypeError, RuntimeError):
        if not batch or not all(isinstance(item, Mapping) and set(item) == {"x", "y", "meta"} for item in batch):
            raise
        items = [cast("Mapping[str, Any]", item) for item in batch]
        return {
            "x": default_collate([item["x"] for item in items]),
            "y": default_collate([item["y"] for item in items]),
            "meta": _collate_steady_metadata([item["meta"] for item in items]),
        }


def make_data_loader(
    dataset: Dataset[Any],
    settings: LoaderSettings,
    *,
    sampler: Sampler[Any] | None = None,
    generator: torch.Generator | None = None,
    worker_init_fn: Any | None = None,
) -> DataLoader[Any]:
    """Construct one DataLoader while enforcing sampler/shuffle ownership."""
    if sampler is not None and settings.shuffle:
        message = "DataLoader shuffle and an explicit sampler are mutually exclusive."
        raise ValueError(message)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": settings.batch_size,
        "shuffle": settings.shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "drop_last": settings.drop_last,
        "collate_fn": _collate_runtime_batch,
    }
    if generator is not None:
        arguments["generator"] = generator
    if worker_init_fn is not None:
        arguments["worker_init_fn"] = worker_init_fn
    if settings.num_workers > 0:
        arguments["persistent_workers"] = settings.persistent_workers
        if settings.prefetch_factor is not None:
            arguments["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(**arguments)


def create_data_loader(
    request: DatasetRequest,
    settings: LoaderSettings,
    *,
    sampler: Sampler[Any] | None = None,
    transient_transform: transient.TransientTransform | None = None,
) -> DataLoader[Any]:
    """Resolve one package request and construct its validated DataLoader."""
    dataset = create_dataset(
        request,
        hdf5_cache_size=settings.hdf5_cache_size,
        transient_transform=transient_transform,
    )
    return make_data_loader(dataset, settings, sampler=sampler)


def typed_dataset_id(dataset: Dataset[Any]) -> str:
    """Return one runtime dataset ID across the two view implementations."""
    if isinstance(dataset, transient.TransientPhysicalDataset):
        return dataset.dataset_id
    dataset_identity = getattr(dataset, "identity", None)
    if not isinstance(dataset_identity, identity.DatasetIdentity):
        message = "Runtime dataset does not expose a recognized immutable identity."
        raise TypeError(message)
    return cast("identity.DatasetIdentity", dataset_identity).dataset_id
