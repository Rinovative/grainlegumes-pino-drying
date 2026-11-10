"""Orchestrate Dataset-partitioned steady-flow preprocessing and DataLoaders.

Responsibilities:
  - Resolve verified steady package runtimes for training and evaluation Datasets
  - Bind Train (or its ``train_subset``), validation and ID-test to the Dataset-owned fixed case partition, material OOD to
    the declared OOD cases of the training and material-OOD Datasets and parameter OOD to the training Dataset's
    parameter-OOD cases
  - Report the startup phases both tasks share and obtain Train-only scaling through the shared scaling service
  - Seed workers and construct train, validation, ID-test, material-OOD and parameter-OOD loaders (none for an empty part)

Design principles:
  - Membership comes from Dataset manifests; run seeds never repartition data
  - Normalizers are fitted on Train membership only and reused unchanged
  - The scaling cache is keyed by the identity and the run's recorded model digest; the artifact identity stays data-only
  - Loader and worker seeds do not alter persisted membership identity

This module does NOT:
  - Define split schemas or scaling artifact schemas
  - Persist run artifacts or resolve experiment paths
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.preprocessing import dataset_preprocessing_normalization as normalization
from src.datasets.preprocessing import dataset_preprocessing_scaling as scaling
from src.datasets.preprocessing import dataset_preprocessing_splits as splits
from src.datasets.runtime import dataset_runtime_factory as runtime_factory
from src.datasets.runtime import dataset_runtime_steady as steady

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from neuralop.data.transforms.data_processors import DefaultDataProcessor

    from src.domain.tasks.domain_task_spec import TaskSpec


_STEADY_TASK: Final = "steady_flow"


def _normalized_seed(value: Any, *, label: str) -> int:
    """Return one integer loader or worker seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    return value


def _make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Create a worker_init_fn for deterministic DataLoader worker seeding.

    When num_workers > 0, PyTorch spawns worker processes. Each worker
    must have its RNG seeded independently but deterministically.

    Parameters
    ----------
    base_seed : int
        Base seed for the worker pool.

    Returns
    -------
    callable
        Function to pass as worker_init_fn to DataLoader.

    """

    def worker_init_fn(worker_id: int) -> None:
        """Seed the worker's random state."""
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))  # noqa: NPY002 -- worker process RNG
        torch.manual_seed(worker_seed)

    return worker_init_fn


def _steady_dataset(dataset_id: str, *, storage_root: Path | str | None) -> steady.SteadyFlowDataset:
    """Load one complete verified steady package runtime."""
    request = runtime_factory.DatasetRequest(dataset_id=dataset_id, dataset_view=_STEADY_TASK, storage_root=storage_root)
    value = runtime_factory.create_dataset(request)
    if not isinstance(value, steady.SteadyFlowDataset):
        msg = "Steady Dataset factory returned an unexpected runtime."
        raise TypeError(msg)
    return value


def _binding(dataset: steady.SteadyFlowDataset) -> identity.ManifestBinding:
    """Return the manifest binding the factory attached to a steady runtime."""
    return cast("identity.ManifestBinding", dataset.manifest_binding)


def _split_identity(dataset: steady.SteadyFlowDataset) -> dict[str, Any]:
    """Return the persisted split identity of one steady runtime."""
    return splits.split_dataset_identity(_binding(dataset), dataset.payload_identity)


def _ood_dataset(parts: Sequence[tuple[steady.SteadyFlowDataset, Sequence[str]]], *, role: str) -> Dataset[Any] | None:
    """Combine one OOD role's evaluation packages in split order and select their declared cases; ``None`` when empty."""
    if not parts:
        return None
    combined = splits.combine_identity_datasets([dataset for dataset, _ in parts])
    positions: list[int] = []
    offset = 0
    for dataset, case_ids in parts:
        positions.extend(offset + position for position in splits.role_positions(dataset.identity, case_ids, label=role))
        offset += dataset.identity.sample_count
    return Subset(combined, positions)


def create_dataloaders(
    *,
    train_dataset_id: str,
    material_ood_dataset_ids: Sequence[str],
    train_subset: int | None,
    task: TaskSpec,
    model_sha256: str,
    storage_root: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    loader_seed: int = 9,
    worker_seed: int | None = None,
    split_indices: Mapping[str, Any] | None = None,
    scaling_artifact: Mapping[str, Any] | None = None,
    split_persist_callback: Callable[[Mapping[str, Any]], None] | None = None,
    startup_callback: scaling.StartupCallback | None = None,
) -> tuple[DataLoader[Any], dict[str, DataLoader[Any] | None], DefaultDataProcessor, dict[str, Any], dict[str, Any]]:
    """
    Create steady-flow loaders from the Dataset-owned fixed case partition.

    Parameters
    ----------
    train_dataset_id : str
        Published steady Dataset with a fixed Train/validation/ID-test partition.
    material_ood_dataset_ids : Sequence[str]
        Published Datasets whose declared OOD cases join the training Dataset's in the material-OOD part.
    train_subset : int | None
        ``data.train_subset``: ``None`` trains on every Train case, N on the first N per ID material; scalers fit on it.
    task : TaskSpec
        Authoritative task contract.
    model_sha256 : str
        The run's recorded model implementation digest; it binds the scaling cache key, so changed fit or application code
        refits once while a loss-only change reuses the entry.
    storage_root : Path | str | None, optional
        Explicit storage root; the default project storage is used when omitted.
    batch_size : int, optional
        Batch size for all loaders.
    num_workers : int, optional
        Training DataLoader worker count.
    pin_memory : bool, optional
        Whether the training loader pins host memory.
    persistent_workers : bool, optional
        Whether nonzero training workers persist across epochs.
    loader_seed : int, optional
        Shuffled training-loader generator seed; it never changes membership.
    worker_seed : int | None, optional
        Base worker seed; worker ``i`` receives ``worker_seed + i``. Defaults to
        ``loader_seed``.
    split_indices : Mapping[str, Any] | None, optional
        Saved split; it must equal the split rebuilt from the Datasets.
    scaling_artifact : Mapping[str, Any] | None, optional
        Restored scaling artifact, admitted under this run's scaling identity. When omitted, the normalizers come from
        the shared scaling cache or are fitted on Train only.
    split_persist_callback : Callable or None, optional
        Persist a newly resolved split before normalizer fitting.
    startup_callback : Callable or None, optional
        Receives the startup phase events both tasks report.

    Returns
    -------
    tuple[DataLoader, dict[str, DataLoader | None], DefaultDataProcessor, dict[str, Any], dict[str, Any]]
        A shuffled train loader, the ``eval`` (validation), ``id_test``, ``material_ood`` and ``parameter_ood`` loaders
        (``None`` for an empty OOD part), the processor, the split payload and the scaling artifact.

    Raises
    ------
    TypeError
        If seeds, restored split state, or runtimes have incompatible types.
    ValueError
        If a Dataset lacks its fixed partition, OOD declarations are missing,
        family holdouts reuse training materials, or saved membership or scaling drifts.

    """
    loader_seed = _normalized_seed(loader_seed, label="loader_seed")
    worker_seed = _normalized_seed(loader_seed if worker_seed is None else worker_seed, label="worker_seed")
    if num_workers == 0:
        persistent_workers = False

    full_train = _steady_dataset(train_dataset_id, storage_root=storage_root)
    binding = _binding(full_train)
    if full_train.identity.task != task.id:
        msg = f"Training Dataset {train_dataset_id!r} does not belong to task {task.id!r}."
        raise ValueError(msg)
    fixed = partition_contract.validate_training_partition(
        binding.case_partition,
        dataset_id=train_dataset_id,
        case_ids=binding.case_ids,
        material_by_case=binding.materials,
        design_kind_by_case=binding.design_kinds,
    )
    assignments = package_manifest.declared_evaluation_assignments(
        _STEADY_TASK, train_dataset_id, tuple(material_ood_dataset_ids), storage_root=storage_root
    )
    loaded = {dataset_id: _steady_dataset(dataset_id, storage_root=storage_root) for dataset_id in material_ood_dataset_ids}
    loaded[train_dataset_id] = full_train
    parts = {
        role: [
            (source, partition_contract.evaluation_cases(cast("Mapping[str, Any]", _binding(source).case_partition), role))
            for source in (loaded[assignment["dataset_id"]] for assignment in assignments[role])
        ]
        for role in splits.OOD_SPLIT_ROLES
    }
    splits.reject_role_leakage(
        binding, fixed, {role: [(_binding(source), case_ids) for source, case_ids in parts[role]] for role in splits.OOD_SPLIT_ROLES}
    )
    rebuilt = splits.build_split(
        task=task.id,
        train_dataset=_split_identity(full_train),
        case_partition=fixed,
        train_subset=train_subset,
        material_by_case=binding.materials,
        ood_parts={
            role: [
                (_split_identity(source), case_ids, assignment) for (source, case_ids), assignment in zip(parts[role], assignments[role], strict=True)
            ]
            for role in splits.OOD_SPLIT_ROLES
        },
        extension=None,
    )
    split_contract = splits.resolve_split(rebuilt, split_indices)
    split_info = split_contract.as_payload()
    if startup_callback is not None:
        resolved = splits.resolved_event(split_contract, items=split_contract.train.count, grid=full_train.identity.spatial_shape)
        startup_callback("dataset_resolved", resolved)
    if split_indices is None and split_persist_callback is not None:
        split_persist_callback(split_info)

    train_set = Subset(full_train, list(split_contract.train.index_values))

    def fit() -> dict[str, Any]:
        fitting_loader = runtime_factory.make_data_loader(train_set, runtime_factory.LoaderSettings(batch_size=batch_size))
        return normalization.processor_state(normalization.fit_data_processor(fitting_loader, task=task))

    artifact = scaling.fit_or_load_scaling(
        scaling.scaling_identity(split_contract, normalization.preprocessing_spec(task)),
        model_sha256=model_sha256,
        fit=fit,
        storage_root=storage_root,
        restored=scaling_artifact,
        startup_callback=startup_callback,
    )
    data_processor = normalization.data_processor_from_state(artifact["state"], device="cpu")

    loader_started = time.perf_counter()
    generator = torch.Generator().manual_seed(loader_seed)
    worker_init = _make_worker_init_fn(worker_seed) if num_workers > 0 else None
    train_loader = runtime_factory.make_data_loader(
        train_set,
        runtime_factory.LoaderSettings(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            shuffle=True,
        ),
        generator=generator,
        worker_init_fn=worker_init,
    )
    evaluation_settings = runtime_factory.LoaderSettings(batch_size=batch_size)
    evaluation_loaders: dict[str, DataLoader[Any] | None] = {
        "eval": runtime_factory.make_data_loader(Subset(full_train, list(split_contract.validation.index_values)), evaluation_settings),
        "id_test": runtime_factory.make_data_loader(Subset(full_train, list(split_contract.id_test.index_values)), evaluation_settings),
    }
    for role in splits.OOD_SPLIT_ROLES:
        ood_dataset = _ood_dataset(parts[role], role=role)
        evaluation_loaders[role] = None if ood_dataset is None else runtime_factory.make_data_loader(ood_dataset, evaluation_settings)
    if startup_callback is not None:
        startup_callback("dataloader_ready", {"workers": num_workers, "seconds": time.perf_counter() - loader_started})
    return train_loader, evaluation_loaders, data_processor, split_info, artifact
