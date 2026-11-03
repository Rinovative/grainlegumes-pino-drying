"""Resolve package requests and orchestrate generic Dataset runtime loading.

Responsibilities:
  - Validate package-view, exact-case, and DataLoader requests
  - Dispatch steady and transient packages to their responsible runtimes
  - Construct DataLoaders with safe worker and collation settings
  - Form batches whose items share one key, such as a whole-trajectory length

Design principles:
  - View runtimes own their data contracts and selection behavior
  - Package identity and ordering remain independent of DataLoader behavior
  - Worker-only options are passed only when worker processes exist

This module does NOT:
  - Implement steady or transient Dataset storage access
  - Fit normalization, choose experiment splits, or train models
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from src import common
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.packages import dataset_packages_transient_shards as transient_shards

from . import dataset_runtime_steady as steady
from . import dataset_runtime_transient as transient

if TYPE_CHECKING:
    from pathlib import Path


TransientBackendPreference = Literal["pt_shards", "canonical_hdf5"]


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Select one immutable package view and optional exact case membership."""

    dataset_id: str
    dataset_view: views.DatasetViewId
    case_ids: tuple[str, ...] | None = None
    transient_sampling: transient_contract.TransientSamplingSpec | None = None
    storage_root: Path | str | None = None
    transient_backend_preference: TransientBackendPreference = "pt_shards"
    transient_backend_required: bool = False
    spatial_stride: int = 1

    def __post_init__(self) -> None:
        """Reject malformed view, source-case, or runtime requests."""
        common.paths.validate_logical_name(self.dataset_id, label="dataset_id")
        views.get_view(self.dataset_view)
        if self.dataset_view == "transient_drying":
            if not isinstance(self.transient_sampling, transient_contract.TransientSamplingSpec):
                message = "Transient Dataset requests require one explicit transient_sampling specification."
                raise TypeError(message)
            transient_contract.validate_spatial_stride(self.spatial_stride)
        elif self.transient_sampling is not None or self.spatial_stride != 1:
            message = "Steady-flow Dataset requests cannot include transient_sampling or a non-unit spatial_stride."
            raise ValueError(message)
        if self.case_ids is not None and (
            not isinstance(self.case_ids, tuple)
            or not self.case_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in self.case_ids)
            or len(self.case_ids) != len(set(self.case_ids))
        ):
            message = "Dataset case_ids must be an ordered unique non-empty tuple."
            raise ValueError(message)
        if self.transient_backend_preference not in {"pt_shards", "canonical_hdf5"}:
            message = "transient_backend_preference must be 'pt_shards' or 'canonical_hdf5'."
            raise ValueError(message)
        if not isinstance(self.transient_backend_required, bool):
            message = "transient_backend_required must be boolean."
            raise TypeError(message)
        if self.dataset_view != "transient_drying" and self.transient_backend_required:
            message = "Steady-flow Dataset requests cannot require a transient backend."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class LoaderSettings:
    """Validate the common DataLoader and bounded HDF5-cache surface."""

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
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
        if not all(isinstance(value, bool) for value in (self.pin_memory, self.persistent_workers, self.shuffle)):
            message = "pin_memory, persistent_workers, and shuffle must be boolean."
            raise TypeError(message)
        if self.num_workers == 0 and self.persistent_workers:
            message = "persistent_workers requires num_workers > 0."
            raise ValueError(message)


class GroupedBatchSampler(Sampler[list[int]]):
    """Shuffle all items once per epoch and batch only items that share one key."""

    def __init__(self, keys: Sequence[Hashable], batch_size: int, generator: torch.Generator) -> None:
        """Bind per-item keys, the batch size, and the explicit shuffle generator."""
        if not keys:
            message = "Grouped batching requires at least one item."
            raise ValueError(message)
        if not isinstance(generator, torch.Generator):
            message = "Grouped batching requires an explicit torch.Generator."
            raise TypeError(message)
        self.keys = tuple(keys)
        self.batch_size = batch_size
        self.generator = generator
        self._batch_count = len(group_batches(range(len(self.keys)), self.keys, batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of key-homogeneous batches from a fresh permutation."""
        order = torch.randperm(len(self.keys), generator=self.generator).tolist()
        return iter(group_batches(order, self.keys, self.batch_size))

    def __len__(self) -> int:
        """Return the per-epoch batch count, which does not depend on the permutation."""
        return self._batch_count


def _payload_path(manifest: Mapping[str, Any], *, storage_root: Path | str | None) -> Path:
    """Resolve a manifest-bound payload under the dataset lifecycle root."""
    return common.paths.resolve_dataset_path(
        manifest["dataset_view"], str(manifest["dataset_id"]), dataset_root=common.paths.get_datasets_root(storage_root=storage_root)
    )


def _select_transient(
    dataset: transient.TransientPhysicalDataset,
    request: DatasetRequest,
) -> transient.TransientPhysicalDataset:
    """Return a transient dataset restricted to exact experiment case IDs."""
    if request.case_ids is None:
        return dataset
    return transient.select_transient_cases(dataset, request.case_ids)


def _create_dataset_from_manifest(
    request: DatasetRequest,
    manifest: Mapping[str, Any],
    *,
    hdf5_cache_size: int = 0,
) -> Dataset[steady.SteadyFlowItem] | transient.TransientPhysicalDataset:
    """Construct one runtime from package evidence admitted by its caller."""
    if manifest["dataset_view"] != request.dataset_view:
        message = f"Dataset request view {request.dataset_view!r} does not match package {manifest['dataset_view']!r}."
        raise ValueError(message)
    payload_path = _payload_path(manifest, storage_root=request.storage_root)
    if request.dataset_view == "steady_flow":
        steady_dataset = steady.create_dataset(payload_path)
        return steady.select_dataset(
            steady_dataset,
            manifest,
            case_ids=request.case_ids,
        )
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
            spatial_stride=request.spatial_stride,
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
            spatial_stride=request.spatial_stride,
        )
    if (
        transient_dataset.payload["dataset_id"] != manifest["dataset_id"]
        or transient_dataset.payload["contract_digest"] != manifest["channel_contract_digest"]
        or transient_dataset.payload["sample_count"] != manifest["sample_count"]
    ):
        message = f"Transient index does not bind its package manifest: {payload_path}."
        raise ValueError(message)
    selected = _select_transient(transient_dataset, request)
    selected.bind_dataset_manifest_digest(str(manifest["dataset_digest"]))
    selected.dataset_case_partition = manifest["case_partition"]
    return selected


def create_dataset(
    request: DatasetRequest,
    *,
    hdf5_cache_size: int = 0,
) -> Dataset[steady.SteadyFlowItem] | transient.TransientPhysicalDataset:
    """Resolve, validate, and select one package-backed runtime dataset."""
    manifest = package_manifest.load_package_manifest(
        request.dataset_view,
        request.dataset_id,
        storage_root=request.storage_root,
    )
    return _create_dataset_from_manifest(
        request,
        manifest,
        hdf5_cache_size=hdf5_cache_size,
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
    """Use default collation, with a narrow fallback for rich steady metadata.

    The fallback admits mappings that share one key set including ``meta`` (training
    items, and evaluation items that also carry their split indices): ``meta`` keeps
    absent values through steady-metadata collation, every other key collates normally.
    """
    try:
        return default_collate(batch)
    except (TypeError, RuntimeError):
        if not batch or not all(isinstance(item, Mapping) and "meta" in item for item in batch):
            raise
        items = [cast("Mapping[str, Any]", item) for item in batch]
        keys = tuple(items[0])
        if any(tuple(item) != keys for item in items[1:]):
            raise
        return {
            key: _collate_steady_metadata([item[key] for item in items]) if key == "meta" else default_collate([item[key] for item in items])
            for key in keys
        }


def group_batches(order: Sequence[int], keys: Sequence[Hashable] | None, batch_size: int) -> list[list[int]]:
    """
    Partition an item order into batches whose items share one key.

    Parameters
    ----------
    order : Sequence[int]
        Item indices in the order they should be consumed.
    keys : Sequence[Hashable] or None
        Key of every item index; ``None`` puts all items under one key.
    batch_size : int
        Maximum items per batch.

    Returns
    -------
    list[list[int]]
        Batches that keep the relative item order within each key; a batch
        closes at ``batch_size`` items or at the end, and batches are ordered
        by their first item. With one key this is consecutive chunking.

    Raises
    ------
    ValueError
        If ``batch_size`` is not a positive integer or an index has no key.

    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        message = "batch_size must be a positive integer."
        raise ValueError(message)
    items = list(order)
    if keys is None:
        return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    if any(item < 0 or item >= len(keys) for item in items):
        message = "Every batched item index needs one key."
        raise ValueError(message)
    open_batches: dict[Hashable, tuple[int, list[int]]] = {}
    closed: list[tuple[int, list[int]]] = []
    for position, item in enumerate(items):
        first, batch = open_batches.setdefault(keys[item], (position, []))
        batch.append(item)
        if len(batch) == batch_size:
            closed.append((first, batch))
            del open_batches[keys[item]]
    closed.extend(open_batches.values())
    return [batch for _, batch in sorted(closed, key=lambda entry: entry[0])]


def make_data_loader(
    dataset: Dataset[Any],
    settings: LoaderSettings,
    *,
    sampler: Sampler[Any] | None = None,
    batches: Sequence[Sequence[int]] | Sampler[list[int]] | None = None,
    generator: torch.Generator | None = None,
    worker_init_fn: Any | None = None,
) -> DataLoader[Any]:
    """Construct one DataLoader while enforcing sampler/shuffle ownership; ``batches`` fixes every batch or samples them."""
    if sampler is not None and settings.shuffle:
        message = "DataLoader shuffle and an explicit sampler are mutually exclusive."
        raise ValueError(message)
    if batches is not None and (sampler is not None or settings.shuffle):
        message = "Explicit batches exclude a sampler and shuffling."
        raise ValueError(message)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "collate_fn": _collate_runtime_batch,
    }
    if batches is None:
        arguments.update(
            {
                "batch_size": settings.batch_size,
                "shuffle": settings.shuffle if sampler is None else False,
                "sampler": sampler,
            }
        )
    else:
        arguments["batch_sampler"] = batches if isinstance(batches, Sampler) else [list(batch) for batch in batches]
    if generator is not None:
        arguments["generator"] = generator
    if worker_init_fn is not None:
        arguments["worker_init_fn"] = worker_init_fn
    if settings.num_workers > 0:
        arguments["persistent_workers"] = settings.persistent_workers
    return DataLoader(**arguments)
