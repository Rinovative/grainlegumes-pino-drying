"""
dataset_runtime_factory.py

Resolve package requests and orchestrate generic Dataset runtime loading.
Responsibilities:
  - Validate package-view, selector, and DataLoader requests
  - Dispatch steady and transient packages to their responsible runtimes
  - Construct DataLoaders with safe worker and collation settings
Design principles:
  - View runtimes own their data contracts and selection behavior
  - Package identity and ordering remain independent of DataLoader behavior
  - Worker-only options are passed only when worker processes exist
This module does NOT:
  - Implement steady or transient Dataset storage access
  - Fit normalization, choose experiment splits, or train models
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from src import common
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.packages import dataset_packages_transient_shards as transient_shards

from . import dataset_runtime_steady as steady
from . import dataset_runtime_transient as transient

if TYPE_CHECKING:
    from pathlib import Path

    import torch


TransientBackendPreference = Literal["pt_shards", "canonical_hdf5"]


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Select one immutable package view, regime, and optional submembership."""

    dataset_id: str
    dataset_view: views.DatasetViewId
    evaluation_regime: views.PackageRegime
    membership: views.IdMembership | None = None
    ood_group: str | None = None
    transient_sampling: transient_contract.TransientSamplingSpec | None = None
    storage_root: Path | str | None = None
    allow_technical_smoke: bool = False
    transient_backend_preference: TransientBackendPreference = "pt_shards"
    transient_backend_required: bool = False

    def __post_init__(self) -> None:
        """Reject selectors that are ambiguous or invalid for their regime."""
        common.paths.validate_logical_name(self.dataset_id, label="dataset_id")
        views.get_view(self.dataset_view)
        if self.dataset_view == "transient_drying":
            if not isinstance(self.transient_sampling, transient_contract.TransientSamplingSpec):
                message = "Transient Dataset requests require one explicit transient_sampling specification."
                raise TypeError(message)
        elif self.transient_sampling is not None:
            message = "Steady-flow Dataset requests cannot include transient_sampling."
            raise ValueError(message)
        if self.membership is not None and self.membership not in views.ID_MEMBERSHIPS:
            message = f"Unsupported ID membership selector: {self.membership!r}."
            raise ValueError(message)
        if self.ood_group is not None and (not isinstance(self.ood_group, str) or not self.ood_group):
            message = "Parameter-OOD group selectors must be non-empty strings."
            raise ValueError(message)
        if self.evaluation_regime not in views.PACKAGE_REGIMES:
            message = f"Unsupported package regime: {self.evaluation_regime!r}."
            raise ValueError(message)
        if not isinstance(self.allow_technical_smoke, bool):
            message = "allow_technical_smoke must be boolean."
            raise TypeError(message)
        if self.transient_backend_preference not in {"pt_shards", "canonical_hdf5"}:
            message = "transient_backend_preference must be 'pt_shards' or 'canonical_hdf5'."
            raise ValueError(message)
        if not isinstance(self.transient_backend_required, bool):
            message = "transient_backend_required must be boolean."
            raise TypeError(message)
        if self.dataset_view != "transient_drying" and self.transient_backend_required:
            message = "Steady-flow Dataset requests cannot require a transient backend."
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


def _payload_path(manifest: Mapping[str, Any], *, storage_root: Path | str | None) -> Path:
    """Resolve a manifest-bound payload under the dataset lifecycle root."""
    return common.paths.get_dataset_packages_root(storage_root=storage_root) / str(manifest["dataset_id"]) / str(manifest["payload_filename"])


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
    case_ids = tuple(
        dict.fromkeys(
            str(dataset.payload["cases"][int(dataset.payload["samples"][position]["case_index"])]["package_case_id"]) for position in indices
        )
    )
    return transient.select_transient_cases(dataset, case_ids)


def _create_dataset_from_manifest(
    request: DatasetRequest,
    manifest: Mapping[str, Any],
    *,
    hdf5_cache_size: int = 0,
    transient_transform: transient.TransientTransform | None = None,
) -> Dataset[steady.SteadyFlowItem] | transient.TransientPhysicalDataset:
    """Construct one runtime from package evidence admitted by its caller."""
    if manifest["dataset_view"] != request.dataset_view or manifest["evaluation_regime"] != request.evaluation_regime:
        message = (
            f"Dataset request {request.dataset_view!r}/{request.evaluation_regime!r} "
            f"does not match package {manifest['dataset_view']!r}/{manifest['evaluation_regime']!r}."
        )
        raise ValueError(message)
    if manifest.get("campaign_purpose") == "technical_runtime_smoke" and not request.allow_technical_smoke:
        message = "Technical-smoke package admission requires allow_technical_smoke=True."
        raise ValueError(message)
    payload_path = _payload_path(manifest, storage_root=request.storage_root)
    if request.dataset_view == "steady_flow":
        if transient_transform is not None:
            message = "Transient transforms cannot be applied to steady-flow datasets."
            raise ValueError(message)
        steady_dataset = steady.create_dataset(payload_path)
        return steady.select_dataset(
            steady_dataset,
            manifest,
            evaluation_regime=request.evaluation_regime,
            membership=request.membership,
            ood_group=request.ood_group,
        )
    if request.ood_group is not None and request.ood_group not in manifest["available_ood_groups"]:
        message = f"Transient parameter-OOD group {request.ood_group!r} is unavailable; available selectors are {manifest['available_ood_groups']}."
        raise ValueError(message)
    sampling = request.transient_sampling
    if sampling is None:
        message = "Validated transient Dataset request lost its explicit sampling specification."
        raise RuntimeError(message)
    shard_directory = transient_shards.transient_shard_directory(
        request.dataset_id,
        storage_root=request.storage_root,
    )
    transient_dataset: transient.TransientPhysicalDataset
    if request.transient_backend_preference == "pt_shards" and shard_directory.exists():
        transient_dataset = transient.TransientPTShardDataset(
            payload_path,
            sampling=sampling,
            source_root=request.storage_root,
            hdf5_cache_size=hdf5_cache_size,
            transform=transient_transform,
        )
    elif request.transient_backend_preference == "pt_shards" and request.transient_backend_required:
        message = f"Required transient PT shard payload is missing for {request.dataset_id!r}."
        raise FileNotFoundError(message)
    else:
        transient_dataset = transient.TransientPhysicalDataset(
            payload_path,
            sampling=sampling,
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


def create_dataset(
    request: DatasetRequest,
    *,
    hdf5_cache_size: int = 0,
    transient_transform: transient.TransientTransform | None = None,
) -> Dataset[steady.SteadyFlowItem] | transient.TransientPhysicalDataset:
    """Resolve, validate, and select one package-backed runtime dataset."""
    manifest = package_manifest.load_package_manifest(
        request.dataset_id,
        storage_root=request.storage_root,
    )
    return _create_dataset_from_manifest(
        request,
        manifest,
        hdf5_cache_size=hdf5_cache_size,
        transient_transform=transient_transform,
    )


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
