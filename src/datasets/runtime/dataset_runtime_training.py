"""
===============================================================================
dataset_runtime_training.py
===============================================================================
Orchestrate deterministic steady-data splitting, preprocessing, and DataLoaders.
Responsibilities:
  - Resolve verified steady package runtimes for train and OOD roles
  - Create or replay identity-bound split membership
  - Seed workers and construct train, evaluation, OOD, and ID-test loaders
Design principles:
  - Split admission and normalizer fitting remain delegated to their owners
  - Loader and worker seeds do not alter persisted membership identity
  - The established DataLoader signature and returned payload remain stable
This module does NOT:
  - Define split schemas or normalizer artifact schemas
  - Persist run artifacts or resolve experiment paths
===============================================================================
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.preprocessing import dataset_preprocessing_normalization as normalization
from src.datasets.preprocessing import dataset_preprocessing_splits as splits
from src.datasets.runtime import dataset_runtime_factory as runtime_factory
from src.datasets.runtime import dataset_runtime_steady as steady

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence, Sized

    from neuralop.data.transforms.data_processors import DefaultDataProcessor

    from src.domain.tasks.domain_task_spec import TaskSpec


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


def create_dataloaders(
    path_train: str,
    path_test_ood: str | Sequence[str],
    *,
    task: TaskSpec,
    train_dataset_id: str,
    ood_dataset_id: str | Sequence[str],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    ood_fraction: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    split_seed: int = 9,
    loader_seed: int | None = None,
    worker_seed: int | None = None,
    split_indices: Mapping[str, Any] | None = None,
    data_processor: DefaultDataProcessor | None = None,
) -> tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]:
    """
    Create task-aware dataloaders bound to exact dataset identity.

    Parameters
    ----------
    path_train : str
        Current resolved training dataset path.
    path_test_ood : str or Sequence[str]
        One or more independently resolved OOD package paths.
    task : TaskSpec
        Authoritative task contract.
    train_dataset_id : str
        Expected logical training dataset identifier.
    ood_dataset_id : str or Sequence[str]
        Expected logical OOD package identifiers in path order.
    batch_size : int, optional
        Batch size for all loaders.
    train_ratio : float, optional
        Fraction in ``(0, 1)`` assigned to training. The remainder is evaluation.
        Counts use ``int(train_ratio * full_count)``.
    ood_fraction : float, optional
        Fraction in ``(0, 1]`` selected from the OOD dataset, also rounded down
        with ``int``.
    num_workers : int, optional
        Training DataLoader worker count.
    pin_memory : bool, optional
        Whether the training loader pins host memory.
    persistent_workers : bool, optional
        Whether nonzero training workers persist across epochs.
    split_seed : int, optional
        Deterministic train/eval and OOD membership seed.
    loader_seed : int | None, optional
        Shuffled training-loader generator seed. Defaults to ``split_seed`` but
        does not change membership.
    worker_seed : int | None, optional
        Base Python/NumPy/PyTorch worker seed. Worker ``i`` receives
        ``worker_seed + i``. The default is ``loader_seed``.
    split_indices : Mapping[str, Any] | None, optional
        Saved exact membership to validate against datasets, settings, and
        membership digests. Omission creates deterministic new membership.
    data_processor : DefaultDataProcessor | None, optional
        Restored processor. When omitted, input/output normalizers are fit only
        on the selected training subset over TaskSpec normalization axes.

    Returns
    -------
    tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]
        A shuffled train loader, non-shuffled ``eval`` and ``ood`` loaders, a fitted
        or supplied processor, and the complete current split contract. This
        function does not persist the contract or processor.

    Raises
    ------
    TypeError
        If seeds/settings, restored split state, or factory datasets violate
        required types or fail to expose verified ``DatasetIdentity`` objects.
    ValueError
        If ratios select an empty split, logical IDs disagree with payloads, or
        saved membership/settings/identity fail strict validation.

    Notes
    -----
    ``num_workers=0`` forces ``persistent_workers=False``. Evaluation and OOD
    loaders always use the main process without pinned memory. Fitting a new
    processor materializes the complete selected training tensors in memory.
    Caller-supplied processors are reused without refitting.

    """
    split_settings = splits.admit_split_settings(
        train_ratio=train_ratio,
        ood_fraction=ood_fraction,
        split_seed=split_seed,
    )
    train_ratio = split_settings.train_ratio
    ood_fraction = split_settings.ood_fraction
    split_seed = split_settings.split_seed
    loader_seed = _normalized_seed(split_seed if loader_seed is None else loader_seed, label="loader_seed")
    worker_seed = _normalized_seed(loader_seed if worker_seed is None else worker_seed, label="worker_seed")
    if num_workers == 0:
        persistent_workers = False

    raw_train = steady.create_dataset(path_train, task=task)
    raw_train_identity = getattr(raw_train, "identity", None)
    if not isinstance(raw_train_identity, identity.DatasetIdentity):
        msg = "Task dataset factory must expose a verified DatasetIdentity."
        raise TypeError(msg)
    if raw_train_identity.dataset_id != train_dataset_id:
        msg = f"Resolved training dataset identifier does not match its payload: {raw_train_identity.dataset_id!r}/{train_dataset_id!r}."
        raise ValueError(msg)

    ood_paths = (path_test_ood,) if isinstance(path_test_ood, str) else tuple(path_test_ood)
    ood_ids = (ood_dataset_id,) if isinstance(ood_dataset_id, str) else tuple(ood_dataset_id)
    if not ood_paths or len(ood_paths) != len(ood_ids):
        msg = "OOD package paths and identifiers must be non-empty and aligned."
        raise ValueError(msg)
    ood_sources = [steady.create_dataset(path, task=task) for path in ood_paths]
    for expected_id, source_dataset in zip(ood_ids, ood_sources, strict=True):
        source_identity = getattr(source_dataset, "identity", None)
        if not isinstance(source_identity, identity.DatasetIdentity) or source_identity.dataset_id != expected_id:
            msg = f"Resolved OOD package identifier does not match its payload: {expected_id!r}."
            raise ValueError(msg)
    ood_full = splits.combine_identity_datasets(ood_sources)
    ood_identity = getattr(ood_full, "identity", None)
    if not isinstance(ood_identity, identity.DatasetIdentity):
        msg = "Combined OOD packages did not expose a verified DatasetIdentity."
        raise TypeError(msg)

    package_membership = splits.package_id_membership(raw_train)
    id_test_set: Dataset[Mapping[str, Any]] | None = None
    if package_membership is None:
        full_train: Dataset[Mapping[str, Any]] = raw_train
        explicit_train_count = None
    else:
        train_positions = package_membership["train"]
        validation_positions = package_membership["validation"]
        full_train = splits.select_identity_dataset(
            raw_train,
            [*train_positions, *validation_positions],
            label="package train+validation",
        )
        id_test_set = splits.select_identity_dataset(
            raw_train,
            package_membership["id_test"],
            label="package id_test",
        )
        explicit_train_count = len(train_positions)
        train_ratio = explicit_train_count / len(cast("Sized", full_train))

    train_identity = getattr(full_train, "identity", None)
    if not isinstance(train_identity, identity.DatasetIdentity):
        msg = "Resolved training view did not expose a verified DatasetIdentity."
        raise TypeError(msg)
    n_train_full = len(cast("Sized", full_train))
    n_ood_full = len(cast("Sized", ood_full))
    if split_indices is None:
        n_train = int(train_ratio * n_train_full)
        n_eval = n_train_full - n_train
        n_ood = int(ood_fraction * n_ood_full)
        if min(n_train, n_eval, n_ood) <= 0:
            msg = f"Split settings must select non-empty train/eval/OOD sets. Received train={n_train}, eval={n_eval}, ood={n_ood}."
            raise ValueError(msg)
        if explicit_train_count is None:
            train_random, eval_random = random_split(
                full_train,
                [n_train, n_eval],
                generator=torch.Generator().manual_seed(split_seed),
            )
            train_indices = torch.tensor(train_random.indices, dtype=torch.long)
            eval_indices = torch.tensor(eval_random.indices, dtype=torch.long)
        else:
            n_train = explicit_train_count
            n_eval = n_train_full - n_train
            train_indices = torch.arange(n_train, dtype=torch.long)
            eval_indices = torch.arange(n_train, n_train_full, dtype=torch.long)
        ood_random, _ = random_split(
            ood_full,
            [n_ood, n_ood_full - n_ood],
            generator=torch.Generator().manual_seed(split_seed),
        )
        ood_indices = torch.tensor(ood_random.indices, dtype=torch.long)
        membership_digests = {
            "train": identity.membership_digest(
                role="train",
                dataset_fingerprint=train_identity.fingerprint,
                sample_ids=train_identity.sample_ids,
                indices=[int(value) for value in train_indices.tolist()],
            ),
            "eval": identity.membership_digest(
                role="eval",
                dataset_fingerprint=train_identity.fingerprint,
                sample_ids=train_identity.sample_ids,
                indices=[int(value) for value in eval_indices.tolist()],
            ),
            "ood": identity.membership_digest(
                role="ood",
                dataset_fingerprint=ood_identity.fingerprint,
                sample_ids=ood_identity.sample_ids,
                indices=[int(value) for value in ood_indices.tolist()],
            ),
        }
        split_info: dict[str, Any] = {
            "schema_version": splits.SPLIT_SCHEMA_VERSION,
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "ood_indices": ood_indices,
            "metadata": {
                "datasets": {
                    "train": train_identity.as_dict(),
                    "ood": ood_identity.as_dict(),
                },
                "n_train_full": n_train_full,
                "n_train": n_train,
                "n_eval": n_eval,
                "n_ood_full": n_ood_full,
                "n_ood": n_ood,
                "train_ratio": train_ratio,
                "ood_fraction": ood_fraction,
                "split_seed": split_seed,
                "membership_digests": membership_digests,
            },
        }
    else:
        split_info = dict(split_indices)

    split_contract = splits.admit_split_contract(
        split_info,
        train_identity=train_identity,
        ood_identity=ood_identity,
        expected_train_ratio=train_ratio,
        expected_ood_fraction=ood_fraction,
        expected_split_seed=split_seed,
    )
    split_info = split_contract.as_payload()
    train_indices = split_contract.role("train").indices
    eval_indices = split_contract.role("eval").indices
    ood_indices = split_contract.role("ood").indices
    train_set = Subset(full_train, train_indices.tolist())
    eval_set = Subset(full_train, eval_indices.tolist())
    ood_subset = Subset(ood_full, ood_indices.tolist())

    if data_processor is None:
        fitting_loader = runtime_factory.make_data_loader(
            train_set,
            runtime_factory.LoaderSettings(batch_size=batch_size),
        )
        data_processor = normalization.fit_data_processor(
            fitting_loader,
            task=task,
        )

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
    eval_loader = runtime_factory.make_data_loader(eval_set, evaluation_settings)
    ood_loader = runtime_factory.make_data_loader(ood_subset, evaluation_settings)
    test_loaders = {"eval": eval_loader, "ood": ood_loader}
    if id_test_set is not None:
        test_loaders["id_test"] = runtime_factory.make_data_loader(
            id_test_set,
            evaluation_settings,
        )
    return train_loader, test_loaders, cast("DefaultDataProcessor", data_processor), split_info
