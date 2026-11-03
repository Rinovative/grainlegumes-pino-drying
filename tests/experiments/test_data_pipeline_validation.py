"""Exercise the explicit complete-data validator with production-shaped datasets.

Responsibilities:
  - Check split persistence before fitting and complete Dataset/normalizer/loader validation
  - Check that loader construction caps the worker count at the CPUs available to the process

Design principles:
  - Tiny fixtures preserve real admission boundaries while replacing external I/O

This module does NOT:
  - Load production datasets or benchmark loader throughput
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src import common, datasets, experiments

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

validation = experiments.validation.data_pipeline


class _SyntheticDataset(Dataset[dict[str, Any]]):
    """Expose the production dataset surface over small in-memory BCHW tensors."""

    def __init__(
        self,
        *,
        dataset_id: str,
        inputs: Tensor,
        outputs: Tensor,
        task_id: str,
        data_digest: str,
        path: Path,
    ) -> None:
        """Store tensors, ordered field names, metadata, and exact identity."""
        sample_ids = tuple(f"{dataset_id}_case_{index}" for index in range(len(inputs)))
        self.data: dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "source_metadata": [{"case_id": case_id} for case_id in sample_ids],
        }
        self.input_fields = ["input"]
        self.output_fields = ["output"]
        self.path = path
        self.identity = datasets.contracts.identity.DatasetIdentity(
            dataset_id=dataset_id,
            task=task_id,
            data_contract_digest=data_digest,
            fingerprint=("a" if dataset_id == "id_data" else "b") * 64,
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=tuple(inputs.shape[2:]),
        )

    def __len__(self) -> int:
        """Return the complete source sample count."""
        return len(self.identity.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one production-shaped raw sample with admitted case identity."""
        return {
            "x": self.data["inputs"][index],
            "y": self.data["outputs"][index],
            "meta": {"case_id": self.identity.sample_ids[index]},
        }


def _task() -> SimpleNamespace:
    """Return the TaskSpec attributes consumed by complete validation."""
    return SimpleNamespace(
        id="test_flow",
        contract_digest="d" * 64,
        data_contract_digest="d" * 64,
        tensor_layout=("batch", "channel", "y", "x"),
        normalization_axes=(0, 2, 3),
        preprocessing=SimpleNamespace(fit_split="train"),
        in_channels=1,
        out_channels=1,
        input_names=("input",),
        output_names=("output",),
    )


def _normalizer_state(inputs: Tensor, outputs: Tensor, indices: Tensor) -> dict[str, Tensor]:
    """Fit the exact sample-standard-deviation state on ID training membership."""
    selected_inputs = inputs[indices]
    selected_outputs = outputs[indices]
    return {
        "in_normalizer.mean": selected_inputs.mean(dim=(0, 2, 3), keepdim=True),
        "in_normalizer.std": selected_inputs.std(dim=(0, 2, 3), correction=1, keepdim=True),
        "out_normalizer.mean": selected_outputs.mean(dim=(0, 2, 3), keepdim=True),
        "out_normalizer.std": selected_outputs.std(dim=(0, 2, 3), correction=1, keepdim=True),
    }


def _partition(id_source: _SyntheticDataset) -> dict[str, Any]:
    """Build the Dataset-owned fixed 2/1/1 partition through the production contract."""
    case_ids = list(id_source.identity.sample_ids)
    return datasets.contracts.partition.build_fixed_partition(
        case_ids,
        material_by_case=dict.fromkeys(case_ids, "lentil"),
        material_roles={"lentil": "id"},
        recipe={"kind": "exact", "train": case_ids[:2], "validation": case_ids[2:3], "id_test": case_ids[3:]},
        seed_input=9,
        seed=common.random.derive_subseed(9, "split"),
    )


def _split_info(
    *,
    task: SimpleNamespace,
    id_source: _SyntheticDataset,
    ood_source: _SyntheticDataset,
) -> dict[str, Any]:
    """Build the current persisted split payload for a fixed 2/1/1 partition and two OOD cases."""
    splits = datasets.preprocessing.splits
    partition = _partition(id_source)
    evidence = {
        role: splits.role_evidence(role, id_source.identity, splits.role_positions(id_source.identity, partition[role], label=role))
        for role in splits.ID_SPLIT_ROLES
    }
    return splits.SplitContract(
        schema_version=splits.SPLIT_SCHEMA_VERSION,
        task=task.id,
        task_contract_digest=task.contract_digest,
        split_seed=partition["seed"],
        case_partition=partition,
        case_partition_digest=datasets.contracts.partition.partition_digest(partition),
        train=evidence["train"],
        validation=evidence["validation"],
        id_test=evidence["id_test"],
        additional_evaluation=splits.role_evidence("additional_evaluation", ood_source.identity, (0, 2)),
    ).as_payload()


def _manifest(source: _SyntheticDataset, *, material: str, partition: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest evidence fields read by steady loader construction."""
    return {
        "case_partition": partition,
        "source_case_identities": [{"package_case_id": case_id, "source": {"material_family": material}} for case_id in source.identity.sample_ids],
    }


def test_steady_experiment_persists_dataset_partition_before_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist the Dataset-owned membership before fitting Train-only normalizers."""
    task = _task()
    train = _SyntheticDataset(
        dataset_id="id_data",
        inputs=torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2),
        outputs=torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2),
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "id_data.pt",
    )
    ood = _SyntheticDataset(
        dataset_id="ood_data",
        inputs=torch.arange(16, 32, dtype=torch.float32).reshape(4, 1, 2, 2),
        outputs=torch.arange(16, 32, dtype=torch.float32).reshape(4, 1, 2, 2),
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "ood_data.pt",
    )
    ood_ids = list(ood.identity.sample_ids)
    ood_partition = datasets.contracts.partition.build_fixed_partition(
        ood_ids,
        material_by_case=dict.fromkeys(ood_ids, "chickpea"),
        material_roles={"chickpea": "ood"},
        recipe={"kind": "exact", "train": [], "validation": [], "id_test": []},
        seed_input=9,
        seed=common.random.derive_subseed(9, "split"),
    )
    manifests = {
        "id_data": _manifest(train, material="lentil", partition=_partition(train)),
        "ood_data": _manifest(ood, material="chickpea", partition=ood_partition),
    }
    training = datasets.runtime.training
    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: task)

    def load_evidence(_task: str, dataset_id: str, **_kwargs: object) -> dict[str, Any]:
        return manifests[dataset_id]

    monkeypatch.setattr(training.package_manifest, "load_package_manifest_evidence", load_evidence)
    monkeypatch.setattr(training.package_manifest, "declared_evaluation_assignments", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(training, "_steady_dataset", lambda dataset_id, *, storage_root: {"id_data": train, "ood_data": ood}[dataset_id])  # noqa: ARG005
    events: list[str] = []
    split_path = tmp_path / "run" / "split_indices.pt"
    split_path.parent.mkdir()

    def persist(split: Mapping[str, Any]) -> None:
        common.serialization.atomic_torch_save(split, split_path)
        events.append("persist")

    def fit(_loader: DataLoader, *, task: Any) -> object:
        assert task.id == "test_flow"
        assert split_path.is_file()
        persisted = torch.load(split_path, map_location="cpu", weights_only=True)
        assert persisted["train_indices"].tolist() == [0, 1]
        assert persisted["validation_indices"].tolist() == [2]
        assert persisted["id_test_indices"].tolist() == [3]
        assert persisted["additional_evaluation_indices"].tolist() == [0, 1, 2, 3]
        events.append("fit")
        return object()

    monkeypatch.setattr(datasets.preprocessing.normalization, "fit_data_processor", fit)
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "additional_evaluation_datasets": ["ood_data"],
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "paths": {"dataset_root": str(tmp_path / "02_datasets")},
    }
    loaders = experiments.config.loader.create_dataloaders_from_config(config, split_persist_callback=persist)
    assert events == ["persist", "fit"]
    assert {"train", "eval", "id_test", "additional_evaluation"}.issubset(loaders)
    assert loaders["split_indices"]["metadata"]["case_partition"] == manifests["id_data"]["case_partition"]


@pytest.mark.parametrize(("configured", "available", "effective"), [(8, 4, 4), (0, 4, 0)])
def test_dataloader_workers_never_exceed_the_cpus_available_to_the_process(
    configured: int,
    available: int,
    effective: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cap the configured worker count at the process affinity and report the cap only when it applies."""
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> tuple[None, dict[str, None], None, dict[str, Any]]:
        seen.update(kwargs)
        return None, {"eval": None, "id_test": None, "additional_evaluation": None}, None, {}

    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: _task())
    monkeypatch.setattr(datasets.runtime.training, "create_dataloaders", create)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(available)))
    config = {
        "task": "test_flow",
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "additional_evaluation_datasets": [],
            "batch_size": 2,
            "num_workers": configured,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "paths": {"dataset_root": str(tmp_path / "02_datasets")},
    }

    experiments.config.loader.create_dataloaders_from_config(config)

    assert seen["num_workers"] == effective
    capped = f"data workers {effective} of {configured} configured ({available} CPUs)"
    assert (capped in capsys.readouterr().out) is (effective < configured)
    assert config["data"]["num_workers"] == configured


def test_full_validator_checks_complete_metadata_split_normalizer_and_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return typed PASS evidence while using the maintained scientific owners."""
    task = _task()
    id_inputs = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    id_outputs = id_inputs.mul(2).add(3)
    ood_inputs = torch.arange(32, 48, dtype=torch.float32).reshape(4, 1, 2, 2)
    ood_outputs = ood_inputs.mul(2).add(3)
    id_source = _SyntheticDataset(
        dataset_id="id_data",
        inputs=id_inputs,
        outputs=id_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "id_data.pt",
    )
    ood_source = _SyntheticDataset(
        dataset_id="ood_data",
        inputs=ood_inputs,
        outputs=ood_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        path=tmp_path / "ood_data.pt",
    )
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "additional_evaluation_datasets": ["ood_data"],
        },
        "paths": {"dataset_root": str(tmp_path / "datasets")},
    }
    build_calls: list[dict[str, int]] = []

    def build_loaders(_config: dict[str, Any], *, seed_plan: dict[str, int]) -> dict[str, Any]:
        """Create fresh loaders, processor, and split state for each production call."""
        build_calls.append(seed_plan)
        split = _split_info(task=task, id_source=id_source, ood_source=ood_source)
        state = _normalizer_state(id_inputs, id_outputs, split["train_indices"])
        processor = datasets.preprocessing.normalization.data_processor_from_state(state, device="cpu")
        train_generator = torch.Generator().manual_seed(seed_plan["loader"])
        return {
            "train": DataLoader(
                Subset(id_source, split["train_indices"].tolist()),
                batch_size=2,
                shuffle=True,
                generator=train_generator,
            ),
            "eval": DataLoader(
                Subset(id_source, split["validation_indices"].tolist()),
                batch_size=2,
                shuffle=False,
            ),
            "id_test": DataLoader(
                Subset(id_source, split["id_test_indices"].tolist()),
                batch_size=2,
                shuffle=False,
            ),
            "additional_evaluation": DataLoader(
                Subset(ood_source, split["additional_evaluation_indices"].tolist()),
                batch_size=2,
                shuffle=False,
            ),
            "data_processor": processor,
            "split_indices": split,
        }

    metadata_calls: list[tuple[str, Path, Path]] = []

    def load_metadata(
        dataset_id: str,
        *,
        dataset_identity: Any,
        metadata_root: Path,
        dataset_path: Path,
    ) -> dict[str, str]:
        """Record complete identity/path binding at the mounted metadata boundary."""
        assert dataset_identity.dataset_id == dataset_id
        metadata_calls.append((dataset_id, metadata_root, dataset_path))
        return {"dataset_id": dataset_id}

    monkeypatch.setattr(validation.config_loader, "validate_resolved_config", dict)
    monkeypatch.setattr(validation.config_loader, "validate_resolved_task_contract", lambda _value: task)
    monkeypatch.setattr(validation.config_loader, "create_dataloaders_from_config", build_loaders)
    monkeypatch.setattr(datasets.contracts.metadata, "load_dataset_metadata", load_metadata)

    result = validation.validate_full_data_pipeline(config)

    assert isinstance(result, validation.FullDataValidationResult)
    assert len(build_calls) == 2
    assert metadata_calls == [
        ("id_data", tmp_path / "datasets", tmp_path / "id_data.pt"),
        ("ood_data", tmp_path / "datasets", tmp_path / "ood_data.pt"),
    ]
    assert len(result.dataset_membership) == 5
    assert len(result.channels) == 2
    assert len(result.coverage) == 4
    assert {record.result for record in result.dataset_membership} == {"PASS"}
    assert {record.result for record in result.channels} == {"PASS"}
    assert {record.result for record in result.coverage} == {"PASS"}
    assert {record.loader for record in result.coverage} == {"ID train", "ID validation", "ID test", "Additional evaluation"}
    assert all(record.inverse_checked for record in result.coverage)
    assert [record.result for record in result.overall[-2:]] == ["INFO", "INFO"]
