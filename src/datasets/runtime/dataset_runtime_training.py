"""Orchestrate Dataset-partitioned steady-flow preprocessing and DataLoaders.

Responsibilities:
  - Resolve verified steady package runtimes for training and evaluation Datasets
  - Bind Train/validation/ID-test to the Dataset-owned fixed case partition
  - Seed workers and construct train, validation, ID-test and additional evaluation loaders

Design principles:
  - Membership comes from Dataset manifests; run seeds never repartition data
  - Normalizers are fitted on Train membership only and reused unchanged
  - Loader and worker seeds do not alter persisted membership identity

This module does NOT:
  - Define split schemas or normalizer artifact schemas
  - Persist run artifacts or resolve experiment paths
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.preprocessing import dataset_preprocessing_normalization as normalization
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


def _fixed_partition(manifest: Mapping[str, Any], *, dataset_id: str) -> dict[str, Any]:
    """Require and validate the manifest-owned fixed case partition."""
    fixed = manifest["case_partition"]
    if fixed is None:
        msg = f"Steady Dataset {dataset_id!r} has no fixed case partition; build it from a Dataset plan."
        raise ValueError(msg)
    materials = _materials_by_case(manifest)
    return partition_contract.validate_fixed_partition(fixed, case_ids=tuple(materials), material_by_case=materials)


def _materials_by_case(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the source material family of every package case."""
    return {str(entry["package_case_id"]): str(entry["source"]["material_family"]) for entry in manifest["source_case_identities"]}


def additional_evaluation_components(
    train_dataset_id: str,
    additional_evaluation_dataset_ids: Sequence[str],
    *,
    storage_root: Path | str | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """
    Return the ordered evaluation packages and their declared OOD case IDs.

    Each selected evaluation Dataset contributes its fixed ``ood`` cases; the
    training Dataset follows when its own partition holds material-family cases
    out. Family holdouts must not reuse a training material.
    """
    dataset_ids = tuple(additional_evaluation_dataset_ids)
    if len(dataset_ids) != len(set(dataset_ids)) or train_dataset_id in dataset_ids:
        msg = "Additional evaluation Dataset IDs must be unique and differ from the training Dataset."
        raise ValueError(msg)
    package_manifest.declared_evaluation_assignments(_STEADY_TASK, train_dataset_id, dataset_ids, storage_root=storage_root)
    train_manifest = package_manifest.load_package_manifest_evidence(_STEADY_TASK, train_dataset_id, storage_root=storage_root)
    train_partition = _fixed_partition(train_manifest, dataset_id=train_dataset_id)
    train_materials = {_materials_by_case(train_manifest)[case] for case in train_partition["train"]}
    components: list[tuple[str, tuple[str, ...]]] = []
    for dataset_id in dataset_ids:
        manifest = package_manifest.load_package_manifest_evidence(_STEADY_TASK, dataset_id, storage_root=storage_root)
        partition = _fixed_partition(manifest, dataset_id=dataset_id)
        materials = _materials_by_case(manifest)
        if train_materials.intersection(materials[case] for case in partition["ood"]):
            msg = "Declared family-OOD evaluation reuses a training material family."
            raise ValueError(msg)
        components.append((dataset_id, tuple(partition["ood"])))
    if train_partition["ood"]:
        components.append((train_dataset_id, tuple(train_partition["ood"])))
    return tuple(components)


def additional_evaluation_dataset(
    components: Sequence[tuple[str, Sequence[str]]],
    *,
    storage_root: Path | str | None = None,
    loaded: Mapping[str, steady.SteadyFlowDataset] | None = None,
) -> tuple[Dataset[Any] | None, tuple[int, ...]]:
    """Combine the evaluation packages and locate their declared OOD cases."""
    if not components:
        return None, ()
    parts = [
        (dict(loaded or {}).get(dataset_id) or _steady_dataset(dataset_id, storage_root=storage_root), case_ids)
        for dataset_id, case_ids in components
    ]
    combined = splits.combine_identity_datasets([dataset for dataset, _ in parts])
    positions: list[int] = []
    offset = 0
    for dataset, case_ids in parts:
        positions.extend(offset + position for position in splits.role_positions(dataset.identity, case_ids, label="additional_evaluation"))
        offset += dataset.identity.sample_count
    return combined, tuple(positions)


def create_dataloaders(
    *,
    train_dataset_id: str,
    additional_evaluation_dataset_ids: Sequence[str],
    task: TaskSpec,
    storage_root: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    loader_seed: int = 9,
    worker_seed: int | None = None,
    split_indices: Mapping[str, Any] | None = None,
    data_processor: DefaultDataProcessor | None = None,
    split_persist_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[DataLoader[Any], dict[str, DataLoader[Any] | None], DefaultDataProcessor, dict[str, Any]]:
    """
    Create steady-flow loaders from the Dataset-owned fixed case partition.

    Parameters
    ----------
    train_dataset_id : str
        Published steady Dataset with a fixed Train/validation/ID-test partition.
    additional_evaluation_dataset_ids : Sequence[str]
        Published Datasets whose declared OOD cases are evaluated in addition.
    task : TaskSpec
        Authoritative task contract.
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
        Saved membership; it must equal the membership rebuilt from the Datasets.
    data_processor : DefaultDataProcessor | None, optional
        Restored processor. When omitted, normalizers are fitted on Train only.
    split_persist_callback : Callable or None, optional
        Persist a newly resolved split before normalizer fitting.

    Returns
    -------
    tuple[DataLoader, dict[str, DataLoader | None], DefaultDataProcessor, dict[str, Any]]
        A shuffled train loader, the ``eval`` (validation), ``id_test`` and
        ``additional_evaluation`` loaders, the processor, and the split payload.

    Raises
    ------
    TypeError
        If seeds, restored split state, or runtimes have incompatible types.
    ValueError
        If a Dataset lacks its fixed partition, OOD declarations are missing,
        family holdouts reuse training materials, or saved membership drifts.

    """
    loader_seed = _normalized_seed(loader_seed, label="loader_seed")
    worker_seed = _normalized_seed(loader_seed if worker_seed is None else worker_seed, label="worker_seed")
    if num_workers == 0:
        persistent_workers = False

    train_manifest = package_manifest.load_package_manifest_evidence(_STEADY_TASK, train_dataset_id, storage_root=storage_root)
    fixed = _fixed_partition(train_manifest, dataset_id=train_dataset_id)
    full_train = _steady_dataset(train_dataset_id, storage_root=storage_root)
    if full_train.identity.task != task.id:
        msg = f"Training Dataset {train_dataset_id!r} does not belong to task {task.id!r}."
        raise ValueError(msg)
    additional_dataset, additional_positions = additional_evaluation_dataset(
        additional_evaluation_components(train_dataset_id, additional_evaluation_dataset_ids, storage_root=storage_root),
        storage_root=storage_root,
        loaded={train_dataset_id: full_train},
    )
    additional_identity = cast("identity.DatasetIdentity | None", getattr(additional_dataset, "identity", None))
    evidence = {
        role: splits.role_evidence(role, full_train.identity, splits.role_positions(full_train.identity, fixed[role], label=role))
        for role in splits.ID_SPLIT_ROLES
    }
    rebuilt = splits.SplitContract(
        schema_version=splits.SPLIT_SCHEMA_VERSION,
        task=task.id,
        task_contract_digest=task.contract_digest,
        split_seed=int(fixed["seed"]),
        case_partition=fixed,
        case_partition_digest=partition_contract.partition_digest(fixed),
        train=evidence["train"],
        validation=evidence["validation"],
        id_test=evidence["id_test"],
        additional_evaluation=None
        if additional_identity is None
        else splits.role_evidence("additional_evaluation", additional_identity, additional_positions),
    )
    split_contract = splits.admit_split_contract(
        rebuilt.as_payload() if split_indices is None else split_indices,
        train_identity=full_train.identity,
        additional_evaluation_identity=additional_identity,
        expected_split_seed=rebuilt.split_seed,
    )
    if split_contract != rebuilt:
        msg = "Saved steady split does not match the current Dataset partition and evaluation membership."
        raise ValueError(msg)
    split_info = split_contract.as_payload()
    if split_indices is None and split_persist_callback is not None:
        split_persist_callback(split_info)

    train_set = Subset(full_train, list(split_contract.train.index_values))
    if data_processor is None:
        fitting_loader = runtime_factory.make_data_loader(train_set, runtime_factory.LoaderSettings(batch_size=batch_size))
        data_processor = normalization.fit_data_processor(fitting_loader, task=task)

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
        "additional_evaluation": None
        if additional_dataset is None
        else runtime_factory.make_data_loader(Subset(additional_dataset, list(additional_positions)), evaluation_settings),
    }
    return train_loader, evaluation_loaders, cast("DefaultDataProcessor", data_processor), split_info
