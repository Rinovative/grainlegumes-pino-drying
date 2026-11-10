"""Construct storage-neutral transient training loaders from published packages.

Responsibilities:
  - Resolve the Dataset-owned case roles, the run's Train subset and the declared material-OOD and parameter-OOD cases
    before expanding items
  - Build the run split through the shared split builder, with the transient item expansion as its task extension
  - Obtain one-step Train scaling through the shared scaling service
  - Build deterministically seeded training and evaluation DataLoaders

Design principles:
  - Complete case roles are selected before transitions or rollout windows expand
  - Scaling uses one-step Train items only, through the shared identity-bound cache or a restored run artifact
  - The scaling cache is keyed by the identity and the run's recorded model digest; the artifact identity stays data-only
  - Every role reads the Dataset's PT shards, so membership and samples do not depend on a storage choice

This module does NOT:
  - Read HDF5 directly or create an alternative transient storage runtime
  - Define split or scaling artifact schemas
  - Choose transient tensor channels, losses, or training stages
  - Publish run-owned split or scaling artifacts
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.datasets import packages
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.preprocessing import dataset_preprocessing_scaling as train_scaling
from src.datasets.preprocessing import dataset_preprocessing_splits as splits
from src.datasets.runtime import dataset_runtime_factory as factory
from src.datasets.runtime import dataset_runtime_transient as transient
from src.learning.transient import learning_transient_scaling as scaling
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from src.datasets.contracts import dataset_contracts_identity as identity

_SCALING_SAMPLING: Final = transient_contract.TransientSamplingSpec(mode="one_step_transition")
_TASK: Final = "transient_drying"


@dataclass(frozen=True, slots=True)
class TransientTrainingLoaders:
    """Return loaders and immutable evidence for one transient training run."""

    train: DataLoader[Any]
    evaluation: DataLoader[Any]
    material_ood: DataLoader[Any] | None
    parameter_ood: DataLoader[Any] | None
    id_test: DataLoader[Any]
    data_processor: scaling.TransientScalingArtifact
    scaling_artifact: dict[str, Any]
    split: dict[str, Any]
    spatial_view: dict[str, int]


def _emit_startup(
    callback: train_scaling.StartupCallback | None,
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


def _binding(dataset: transient.TransientPhysicalDataset) -> identity.ManifestBinding:
    """Return the manifest binding the factory attached to a transient runtime."""
    return cast("identity.ManifestBinding", dataset.manifest_binding)


def _split_identity(dataset: transient.TransientPhysicalDataset) -> dict[str, Any]:
    """Return the persisted split identity of one transient runtime."""
    return splits.split_dataset_identity(_binding(dataset), dataset.payload_identity)


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


def _fit_state(
    dataset: transient.TransientPhysicalDataset,
    *,
    tensorizer: TransientTensorizerSpec,
    scaling_identity: Mapping[str, Any],
    scale_mode: scaling.TransientScaleMode,
    startup_callback: train_scaling.StartupCallback | None,
) -> dict[str, Any]:
    """Fit the transient scaling state once from one-step Train items, with bounded shard progress."""
    if not isinstance(dataset, transient.TransientPTShardDataset):
        msg = "Transient scaling is fitted from the Dataset's PT shards only."
        raise TypeError(msg)
    items = dataset.iter_scaling_items(
        progress_callback=lambda processed, total: _emit_startup(
            startup_callback,
            "scaler_fit_progress",
            shards_processed=processed,
            shards_total=total,
        )
    )
    return scaling.fit_transient_scaling(
        items,
        tensorizer=tensorizer,
        dataset_identity=scaling_identity["dataset"],
        train_membership_digest=scaling_identity["train_membership_digest"],
        horizon=dataset.configured_regular_horizon,
        scale_mode=scale_mode,
    ).state_dict()


def create_transient_training_loaders(
    *,
    train_dataset_id: str,
    material_ood_dataset_ids: Sequence[str],
    train_subset: int | None,
    tensorizer: TransientTensorizerSpec,
    train_sampling: transient_contract.TransientSamplingSpec,
    loader_settings: factory.LoaderSettings,
    model_sha256: str,
    storage_root: str | None = None,
    spatial_stride: int = 1,
    scale_mode: scaling.TransientScaleMode = "state_std",
    loader_seed: int | None = None,
    worker_seed: int | None = None,
    saved_split: Mapping[str, Any] | None = None,
    scaling_artifact: Mapping[str, Any] | None = None,
    startup_callback: train_scaling.StartupCallback | None = None,
    split_persist_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> TransientTrainingLoaders:
    """
    Create experiment-role loaders and exact reusable Train-only scaling evidence.

    The Train, validation, and ID-test roles use immutable Dataset cases before
    their items are expanded; Train is the whole Train role or, with ``train_subset`` N, the first N Train cases
    per ID material, and scaling fits on exactly those cases. Train items follow ``train_sampling``; validation,
    ID-test, material-OOD and parameter-OOD items are whole trajectories
    (``transient_contract.EVALUATION_SAMPLING``), so checkpoint selection does not depend on the
    training sample shape. A saved split is replayed only when it equals the
    rebuilt one. A restored scaling artifact or a cache hit under the same
    scaling identity and ``model_sha256`` (the run's recorded model
    implementation digest, which binds the cache key and not the artifact
    identity) avoids recomputing identical Train-only statistics.
    """
    if not isinstance(tensorizer, TransientTensorizerSpec):
        message = "tensorizer must be a TransientTensorizerSpec."
        raise TypeError(message)
    if not isinstance(train_sampling, transient_contract.TransientSamplingSpec):
        message = "train_sampling must be a TransientSamplingSpec."
        raise TypeError(message)
    resolved_spatial_stride = transient_contract.validate_spatial_stride(spatial_stride)
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
            spatial_stride=resolved_spatial_stride,
        )
        value = factory.create_dataset(request)
        if not isinstance(value, transient.TransientPhysicalDataset):
            msg = "Transient Dataset factory returned a non-transient Dataset."
            raise TypeError(msg)
        return value

    full_training_dataset = dataset_for(train_dataset_id)
    full_evaluation_dataset = dataset_for(train_dataset_id, sampling=transient_contract.EVALUATION_SAMPLING)
    binding = _binding(full_training_dataset)
    fixed = partition_contract.validate_training_partition(
        binding.case_partition,
        dataset_id=train_dataset_id,
        case_ids=binding.case_ids,
        material_by_case=binding.materials,
        design_kind_by_case=binding.design_kinds,
    )
    train_cases = partition_contract.train_subset_cases(fixed, material_by_case=binding.materials, train_subset=train_subset)
    train_dataset = _select_cases(full_training_dataset, train_cases)
    # A Train case without any item would stay declared Train while never being trained on.
    uncovered = train_dataset.cases_without_items()
    if uncovered:
        msg = (
            f"Train cases {list(uncovered)} are shorter than the training sampling {train_sampling.as_dict()} and would be "
            "silently excluded; choose a sampling that fits them or exclude them explicitly in the Dataset plan."
        )
        raise ValueError(msg)
    evaluation_dataset = _select_cases(full_evaluation_dataset, fixed["validation"])
    id_test_dataset = _select_cases(full_evaluation_dataset, fixed["id_test"])
    material_ood_ids = tuple(material_ood_dataset_ids)
    assignments = packages.declared_evaluation_assignments(_TASK, train_dataset_id, material_ood_ids, storage_root=storage_root)
    loaded = {dataset_id: dataset_for(dataset_id, sampling=transient_contract.EVALUATION_SAMPLING) for dataset_id in material_ood_ids}
    loaded[train_dataset_id] = full_evaluation_dataset
    declared_parts = {
        role: [
            (source, partition_contract.evaluation_cases(cast("Mapping[str, Any]", _binding(source).case_partition), role))
            for source in (loaded[assignment["dataset_id"]] for assignment in assignments[role])
        ]
        for role in splits.OOD_SPLIT_ROLES
    }
    splits.reject_role_leakage(
        binding, fixed, {role: [(_binding(source), case_ids) for source, case_ids in declared_parts[role]] for role in splits.OOD_SPLIT_ROLES}
    )
    parts = {role: [_select_cases(source, case_ids) for source, case_ids in declared_parts[role]] for role in declared_parts}
    ood_datasets: dict[str, Dataset[Any] | None] = {
        role: role_parts[0] if len(role_parts) == 1 else ConcatDataset(role_parts) if role_parts else None for role, role_parts in parts.items()
    }

    fitting_dataset = (
        train_dataset
        if train_sampling.mode == _SCALING_SAMPLING.mode
        else _select_cases(dataset_for(train_dataset_id, sampling=_SCALING_SAMPLING), train_cases)
    )
    spatial_view = _spatial_view_evidence(train_dataset)
    view_datasets = (
        ("validation", evaluation_dataset),
        ("id_test", id_test_dataset),
        ("scaling", fitting_dataset),
        *((f"{role}[{index}]", part) for role, role_parts in parts.items() for index, part in enumerate(role_parts)),
    )
    for label, dataset in view_datasets:
        if _spatial_view_evidence(dataset) != spatial_view:
            message = f"Transient {label} Dataset does not share the Train spatial view."
            raise ValueError(message)

    rebuilt = splits.build_split(
        task=_TASK,
        train_dataset=_split_identity(full_training_dataset),
        case_partition=fixed,
        train_subset=train_subset,
        material_by_case=binding.materials,
        ood_parts={
            role: [
                (_split_identity(source), case_ids, assignment)
                for (source, case_ids), assignment in zip(declared_parts[role], assignments[role], strict=True)
            ]
            for role in splits.OOD_SPLIT_ROLES
        },
        extension={
            "tensorizer": tensorizer.selection_dict(),
            "sampling": train_sampling.as_dict(),
            "evaluation_sampling": transient_contract.EVALUATION_SAMPLING.as_dict(),
            "spatial_view": spatial_view,
        },
    )
    split = splits.resolve_split(rebuilt, saved_split)
    _emit_startup(
        startup_callback,
        "dataset_resolved",
        **splits.resolved_event(split, items=len(train_dataset), grid=(spatial_view["effective_ny"], spatial_view["effective_nx"])),
    )
    split_payload = split.as_payload()
    if saved_split is None and split_persist_callback is not None:
        split_persist_callback(split_payload)

    scaling_identity = train_scaling.scaling_identity(
        split, scaling.preprocessing_spec(tensorizer=tensorizer, scale_mode=scale_mode, spatial_view=spatial_view)
    )
    envelope = train_scaling.fit_or_load_scaling(
        scaling_identity,
        model_sha256=model_sha256,
        fit=lambda: _fit_state(
            fitting_dataset, tensorizer=tensorizer, scaling_identity=scaling_identity, scale_mode=scale_mode, startup_callback=startup_callback
        ),
        storage_root=storage_root,
        restored=scaling_artifact,
        startup_callback=startup_callback,
    )

    loader_started = time.perf_counter()
    train_loader = _loader(train_dataset, loader_settings, shuffle=True, loader_seed=resolved_loader_seed, worker_seed=resolved_worker_seed)
    evaluation_loader = _loader(
        evaluation_dataset, loader_settings, shuffle=False, loader_seed=resolved_loader_seed + 1, worker_seed=resolved_worker_seed + 1
    )
    ood_loaders = {
        role: None
        if dataset is None
        else _loader(dataset, loader_settings, shuffle=False, loader_seed=resolved_loader_seed + offset, worker_seed=resolved_worker_seed + offset)
        for (role, dataset), offset in zip(ood_datasets.items(), (2, 4), strict=True)
    }
    id_test_loader = _loader(
        id_test_dataset, loader_settings, shuffle=False, loader_seed=resolved_loader_seed + 3, worker_seed=resolved_worker_seed + 3
    )
    _emit_startup(startup_callback, "dataloader_ready", workers=loader_settings.num_workers, seconds=time.perf_counter() - loader_started)
    return TransientTrainingLoaders(
        train=train_loader,
        evaluation=evaluation_loader,
        material_ood=ood_loaders["material_ood"],
        parameter_ood=ood_loaders["parameter_ood"],
        id_test=id_test_loader,
        data_processor=scaling.admit_transient_scaling_state(envelope["state"], identity=scaling_identity),
        scaling_artifact=envelope,
        split=split_payload,
        spatial_view=copy.deepcopy(spatial_view),
    )
