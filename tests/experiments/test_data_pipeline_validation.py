"""Exercise the explicit complete-data validator with production-shaped datasets.

Responsibilities:
  - Check split persistence before fitting and complete Dataset/normalizer/loader validation
  - Check that loader construction caps the worker count at the CPUs available to the process
  - Check that the validator refuses a transient request instead of checking it partially
  - Check that a training run gets no ID-test or OOD loader and only the explicit acceptance asks for them

Design principles:
  - Tiny fixtures preserve real admission boundaries while replacing external I/O

This module does NOT:
  - Load production datasets or benchmark loader throughput
"""

from __future__ import annotations

import copy
import dataclasses
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import configs
from support.scaling import steady_inputs, steady_scaling
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src import common, datasets, domain, experiments
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

validation = experiments.validation.data_pipeline


def _close_dataset(dataset: Dataset[Any]) -> None:
    """Close the transient physical dataset a training loader holds."""
    assert isinstance(dataset, datasets.runtime.transient.TransientPhysicalDataset)
    dataset.close()


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
        input_fields: tuple[str, ...] = ("input",),
        output_fields: tuple[str, ...] = ("output",),
    ) -> None:
        """Store tensors, ordered field names, metadata, and exact identity."""
        sample_ids = tuple(f"{dataset_id}_case_{index}" for index in range(len(inputs)))
        self.data: dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "source_metadata": [{"case_id": case_id} for case_id in sample_ids],
        }
        self.input_fields = list(input_fields)
        self.output_fields = list(output_fields)
        self.identity = datasets.contracts.identity.DatasetIdentity(
            dataset_id=dataset_id,
            task=task_id,
            data_contract_digest=data_digest,
            fingerprint=("a" if dataset_id == "id_data" else "b") * 64,
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=tuple(inputs.shape[2:]),
        )
        self.manifest_binding: datasets.contracts.identity.ManifestBinding | None = None

    @property
    def payload_identity(self) -> dict[str, Any]:
        """Return the identity the payload states about itself, as the production runtime does."""
        return {"fingerprint": self.identity.fingerprint, "spatial_shape": list(self.identity.spatial_shape)}

    def spatial_representation(self) -> datasets.contracts.grid.SpatialRepresentation:
        """Return the full-grid view the runtime serves."""
        height, width = self.identity.spatial_shape
        return datasets.contracts.grid.resolve_spatial_representation((height, width), 1)

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
    # The split schema binds the registered steady task; the channel counts stay fixture-sized.
    return SimpleNamespace(
        id="steady_flow",
        contract_digest=domain.tasks.registry.get_task("steady_flow").contract_digest,
        data_contract_digest="d" * 64,
        tensor_layout=("batch", "channel", "y", "x"),
        normalization_axes=(0, 2, 3),
        preprocessing=SimpleNamespace(fit_split="train"),
        in_channels=1,
        out_channels=1,
        input_names=("input",),
        output_names=("output",),
    )


def _steady_tensors(task: domain.tasks.spec.TaskSpec, *, offset: float, inlet_pressure: float | None = None) -> tuple[Tensor, Tensor]:
    """
    Return four 2 x 2 steady cases on a coordinate grid with distinct static fields, positive inlet pressures and outputs.

    With ``inlet_pressure`` every case has that constant P and a velocity ``v`` whose physical Train std (about 2e-4)
    lies above the scale floor while its dimensionless std (about 2e-9) lies below it.
    """
    inputs = steady_inputs(4, (2, 2))
    values = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) + offset
    for factor, name in enumerate(("Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc"), start=1):
        inputs[:, task.input_names.index(name)] = 1.0 + factor * values
    outputs = torch.stack([values.mul(2).add(3), values.mul(0.5), values.neg()], dim=1)
    if inlet_pressure is not None:
        inputs[:, task.input_names.index("p_in_bc")] = inlet_pressure
        outputs[:, task.output_names.index("v")] = 1.0e-4 * values
    return inputs, outputs


def _fit_scaling(source: _SyntheticDataset, indices: tuple[int, ...]) -> datasets.preprocessing.normalization.TrainScaling:
    """Fit the production steady scaling on the given Train membership of one steady-shaped source."""
    normalization = datasets.preprocessing.normalization
    view = source.spatial_representation().view_evidence()
    return normalization.fit_steady_scaling(
        (source[index] for index in indices),
        preprocessing=normalization.preprocessing_spec("steady_flow", spatial_view=view),
        dataset_identity={"dataset_id": source.identity.dataset_id},
        train_membership_digest="c" * 64,
    )


def _design_kinds(source: _SyntheticDataset, *, material: str) -> dict[str, str]:
    """Return the design kind of each case: Train, Train, validation, test for the ID material, test for the OOD one."""
    case_ids = list(source.identity.sample_ids)
    kinds = ["train", "train", "validation", "test"] if material == "lentil" else ["test"] * len(case_ids)
    return dict(zip(case_ids, kinds, strict=True))


def _partition(id_source: _SyntheticDataset) -> dict[str, Any]:
    """Build the Dataset-owned fixed 2/1/1 partition from the design kinds through the production contract."""
    case_ids = list(id_source.identity.sample_ids)
    return datasets.contracts.partition.build_fixed_partition(
        case_ids,
        material_by_case=dict.fromkeys(case_ids, "lentil"),
        design_kind_by_case=_design_kinds(id_source, material="lentil"),
        material_roles={"lentil": "id"},
    )


def _ood_partition(ood_source: _SyntheticDataset) -> dict[str, Any]:
    """Build the fixed partition of an evaluation-only Dataset whose cases are all declared OOD."""
    case_ids = list(ood_source.identity.sample_ids)
    return datasets.contracts.partition.build_fixed_partition(
        case_ids,
        material_by_case=dict.fromkeys(case_ids, "chickpea"),
        design_kind_by_case=_design_kinds(ood_source, material="chickpea"),
        material_roles={"chickpea": "ood"},
    )


_ASSIGNMENT = {"dataset_id": "ood_data", "role": "material_ood", "relation": "Dataset-declared material holdout", "assumptions": []}
_PRODUCER = {"training_sha256": "a" * 64, "model_sha256": "b" * 64}
_HELD_OUT_ROLES = ("id_test", "material_ood", "parameter_ood")


def _bind(source: _SyntheticDataset, *, material: str, partition: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the manifest binding the Dataset factory gives a runtime and return the manifest evidence it reads."""
    manifest = {
        "dataset_id": source.identity.dataset_id,
        "dataset_digest": ("1" if material == "lentil" else "2") * 64,
        "channel_contract_digest": datasets.contracts.views.get_view("steady_flow").contract_digest,
        "case_partition": partition,
        "grid": {"shape": [4, 4], "axes_sha256": "4" * 64},
        "source_case_identities": [
            {
                "package_case_id": case_id,
                "source": {"material_family": material, "case_input_id": f"input_{case_id}", "simulation_case_id": f"simulation_{case_id}"},
                "design": {"design_kind": kind},
            }
            for case_id, kind in _design_kinds(source, material=material).items()
        ],
    }
    source.manifest_binding = datasets.contracts.identity.manifest_binding(manifest)
    return manifest


def _split_info(*, id_source: _SyntheticDataset, ood_source: _SyntheticDataset) -> dict[str, Any]:
    """Build the current persisted split payload for a fixed 2/1/1 partition and one declared OOD source."""
    splits = datasets.preprocessing.splits
    _bind(id_source, material="lentil", partition=_partition(id_source))
    _bind(ood_source, material="chickpea", partition=_ood_partition(ood_source))

    def identity(source: _SyntheticDataset) -> dict[str, Any]:
        assert source.manifest_binding is not None
        return splits.split_dataset_identity(source.manifest_binding, source.payload_identity)

    ood_cases = _ood_partition(ood_source)["ood"]
    return splits.build_split(
        task="steady_flow",
        train_dataset=identity(id_source),
        case_partition=_partition(id_source),
        train_subset=None,
        material_by_case=dict.fromkeys(id_source.identity.sample_ids, "lentil"),
        ood_parts={"material_ood": [(identity(ood_source), ood_cases, _ASSIGNMENT)]},
        extension=None,
    ).as_payload()


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
    )
    ood = _SyntheticDataset(
        dataset_id="ood_data",
        inputs=torch.arange(16, 32, dtype=torch.float32).reshape(4, 1, 2, 2),
        outputs=torch.arange(16, 32, dtype=torch.float32).reshape(4, 1, 2, 2),
        task_id=task.id,
        data_digest=task.data_contract_digest,
    )
    manifests = {
        "id_data": _bind(train, material="lentil", partition=_partition(train)),
        "ood_data": _bind(ood, material="chickpea", partition=_ood_partition(ood)),
    }
    training = datasets.runtime.training
    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: task)

    def load_evidence(_task: str, dataset_id: str, **_kwargs: object) -> dict[str, Any]:
        return manifests[dataset_id]

    monkeypatch.setattr(training.package_manifest, "load_package_manifest_evidence", load_evidence)
    declared = {"material_ood": [_ASSIGNMENT], "parameter_ood": []}
    monkeypatch.setattr(training.package_manifest, "declared_evaluation_assignments", lambda *_args, **_kwargs: declared)

    def steady_dataset(dataset_id: str, **_kwargs: object) -> _SyntheticDataset:
        return {"id_data": train, "ood_data": ood}[dataset_id]

    monkeypatch.setattr(training, "_steady_dataset", steady_dataset)
    events: list[str] = []
    split_path = tmp_path / "run" / "split_indices.pt"
    split_path.parent.mkdir()

    def persist(split: Mapping[str, Any]) -> None:
        common.serialization.atomic_torch_save(split, split_path)
        events.append("persist")

    def fit(items: Any, **identity: Any) -> datasets.preprocessing.normalization.TrainScaling:
        assert split_path.is_file()
        persisted = datasets.preprocessing.splits.admit_split(torch.load(split_path, map_location="cpu", weights_only=True))
        assert persisted.train.index_values == (0, 1)
        assert persisted.validation.index_values == (2,)
        assert persisted.id_test.index_values == (3,)
        assert [part.index_values for part in persisted.material_ood] == [(0, 1, 2, 3)]
        assert len(list(items)) == len(persisted.train.index_values)
        events.append("fit")
        # The fixture source has one channel; the registered steady scaling only has to carry the fitted identity.
        return dataclasses.replace(steady_scaling((2, 2)), **identity)

    monkeypatch.setattr(datasets.preprocessing.normalization, "fit_steady_scaling", fit)
    create = training.create_dataloaders
    built_roles: list[set[str]] = []

    def record_roles(**kwargs: Any) -> Any:
        built = create(**kwargs)
        built_roles.append(set(built[1]))
        return built

    monkeypatch.setattr(training, "create_dataloaders", record_roles)
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "material_ood_datasets": ["ood_data"],
            "train_subset": None,
            "evaluation_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "spatial_stride": 1,
        },
        "training": {"units_per_update": 2, "memory_batch": 2},
        "paths": {"dataset_root": str(tmp_path / "02_datasets")},
        "producer": _PRODUCER,
    }
    loaders = experiments.config.loader.create_dataloaders_from_config(
        config,
        seed_plan=experiments.run.build_seed_plan(9, stage=None),
        split_persist_callback=persist,
    )
    assert events == ["persist", "fit"]
    # A training run gets Train and validation loaders only; ID-test and OOD stay membership in the saved split.
    assert {"train", "eval", "scaling_artifact"}.issubset(loaders)
    assert not set(loaders).intersection(_HELD_OUT_ROLES)
    assert loaders["split_indices"]["case_partition"] == manifests["id_data"]["case_partition"]
    # Only the explicit data-pipeline acceptance asks for the held-out loaders.
    held_out = experiments.config.loader.create_dataloaders_from_config(
        config,
        seed_plan=experiments.run.build_seed_plan(9, stage=None),
        split_indices=loaders["split_indices"],
        scaling_artifact=loaders["scaling_artifact"],
        held_out_loaders=True,
    )
    assert held_out["id_test"] is not None
    assert held_out["material_ood"] is not None
    assert held_out["parameter_ood"] is None
    # The factory itself builds the held-out loaders only on that explicit request.
    assert built_roles == [{"eval"}, {"eval", *_HELD_OUT_ROLES}]


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

    def create(**kwargs: Any) -> tuple[None, dict[str, None], None, dict[str, Any], dict[str, Any]]:
        seen.update(kwargs)
        return None, {"eval": None, "id_test": None, "material_ood": None, "parameter_ood": None}, None, {}, {}

    monkeypatch.setattr(experiments.config.loader, "validate_resolved_task_contract", lambda _config: _task())
    monkeypatch.setattr(datasets.runtime.training, "create_dataloaders", create)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(available)))
    config = {
        "task": "test_flow",
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "material_ood_datasets": [],
            "train_subset": None,
            "evaluation_batch_size": 2,
            "num_workers": configured,
            "pin_memory": False,
            "persistent_workers": False,
            "spatial_stride": 1,
        },
        "training": {"units_per_update": 2, "memory_batch": 2},
        "paths": {"dataset_root": str(tmp_path / "02_datasets")},
        "producer": _PRODUCER,
    }

    experiments.config.loader.create_dataloaders_from_config(config, seed_plan=experiments.run.build_seed_plan(9, stage=None))

    assert seen["num_workers"] == effective
    capped = f"data workers {effective} of {configured} configured ({available} CPUs)"
    assert (capped in capsys.readouterr().out) is (effective < configured)
    assert config["data"]["num_workers"] == configured


@pytest.mark.parametrize("inlet_pressure", [None, 1.0e5], ids=("varying-inlet-pressure", "dimensionless-v-below-the-floor"))
def test_full_validator_checks_complete_datasets_split_normalizer_and_loaders(monkeypatch: pytest.MonkeyPatch, inlet_pressure: float | None) -> None:
    """
    Return typed PASS evidence while using the maintained scientific owners.

    The second case checks the normalized outputs against the dimensionless statistics that encode them: its
    v is degenerate after the division by P although its physical std lies above the floor.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    id_inputs, id_outputs = _steady_tensors(task, offset=0.0, inlet_pressure=inlet_pressure)
    ood_inputs, ood_outputs = _steady_tensors(task, offset=200.0, inlet_pressure=inlet_pressure)
    fields = {"input_fields": task.input_names, "output_fields": task.output_names}
    id_source = _SyntheticDataset(
        dataset_id="id_data",
        inputs=id_inputs,
        outputs=id_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        **fields,
    )
    ood_source = _SyntheticDataset(
        dataset_id="ood_data",
        inputs=ood_inputs,
        outputs=ood_outputs,
        task_id=task.id,
        data_digest=task.data_contract_digest,
        **fields,
    )
    config = {
        "task": task.id,
        "run": {"seed": 9},
        "data": {
            "train_dataset": "id_data",
            "material_ood_datasets": ["ood_data"],
        },
    }
    build_calls: list[dict[str, int]] = []
    held_out_requests: list[bool] = []

    def build_loaders(_config: dict[str, Any], *, seed_plan: dict[str, int], held_out_loaders: bool = False) -> dict[str, Any]:
        """Create fresh loaders, Train scaling, and split state for each production call."""
        held_out_requests.append(held_out_loaders)
        build_calls.append(seed_plan)
        split = _split_info(id_source=id_source, ood_source=ood_source)
        contract = datasets.preprocessing.splits.admit_split(split)
        scaling = _fit_scaling(id_source, contract.train.index_values)
        train_cases = list(contract.train.index_values)
        sampler = datasets.runtime.units.UnitSampler(
            unit_kind="case", cases=range(len(train_cases)), keys=None, units_per_update=2, memory_batch=2, seed=seed_plan["loader"]
        )
        return {
            "train": DataLoader(Subset(id_source, train_cases), batch_sampler=sampler),
            "eval": DataLoader(
                Subset(id_source, list(contract.validation.index_values)),
                batch_size=2,
                shuffle=False,
            ),
            "id_test": DataLoader(
                Subset(id_source, list(contract.id_test.index_values)),
                batch_size=2,
                shuffle=False,
            ),
            "material_ood": DataLoader(
                Subset(ood_source, list(contract.material_ood[0].index_values)),
                batch_size=2,
                shuffle=False,
            ),
            "parameter_ood": None,
            "scaling": scaling,
            "split_indices": split,
        }

    monkeypatch.setattr(validation.config_loader, "validate_resolved_config", dict)
    monkeypatch.setattr(validation.config_loader, "validate_resolved_task_contract", lambda _value: task)
    monkeypatch.setattr(validation.config_loader, "create_dataloaders_from_config", build_loaders)

    result = validation.validate_full_data_pipeline(config)

    assert isinstance(result, validation.FullDataValidationResult)
    assert len(build_calls) == 2
    # The acceptance checks every role's loader coverage, so it asks for the held-out loaders explicitly; its repeat checks membership only.
    assert held_out_requests == [True, False]
    assert len(result.dataset_membership) == 5
    assert len(result.channels) == task.in_channels + task.out_channels
    assert len(result.coverage) == 4
    assert {record.result for record in result.dataset_membership} == {"PASS"}
    assert {record.result for record in result.channels} == {"PASS"}
    assert {record.result for record in result.coverage} == {"PASS"}
    assert {record.loader for record in result.coverage} == {"ID train", "ID validation", "ID test", "Material OOD"}
    assert all(record.inverse_checked for record in result.coverage)
    assert [record.result for record in result.overall[-2:]] == ["INFO", "INFO"]


def test_full_validator_refuses_a_transient_request_before_loading_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The validator recomputes per-sample standardization, which the transient pipeline does not use; it refuses instead of half-checking."""

    def load_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("a refused request must not load data")

    monkeypatch.setattr(validation.config_loader, "create_dataloaders_from_config", load_forbidden)

    with pytest.raises(ValueError, match="transient_drying"):
        validation.validate_full_data_pipeline(configs.resolve(configs.transient_stage_a_only_config()))


def test_transient_training_factory_opens_only_the_training_dataset_and_builds_no_held_out_loader(
    transient_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transient factory opens the training package only and returns Train and validation loaders; ID-test stays split membership."""
    factory = datasets.runtime.transient_training.factory
    create = factory.create_dataset
    opened: list[str] = []

    def record_open(request: Any) -> Any:
        opened.append(request.dataset_id)
        return create(request)

    monkeypatch.setattr(factory, "create_dataset", record_open)
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=transient_package["train_dataset"],
        material_ood_dataset_ids=[],
        train_subset=None,
        tensorizer=TransientTensorizerSpec(input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("none")),
        train_sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        units_per_update=2,
        memory_batch=1,
        model_sha256="a" * 64,
        storage_root=str(transient_package["root"]),
    )
    _close_dataset(loaders.train.dataset)
    _close_dataset(loaders.evaluation.dataset)

    assert set(opened) == {transient_package["train_dataset"]}
    assert not {field.name for field in dataclasses.fields(loaders)}.intersection(_HELD_OUT_ROLES)
    assert loaders.split["roles"]["id_test"]["case_ids"] == transient_package["case_partition"]["id_test"]


def test_transient_training_gets_no_held_out_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient run gets Train and validation loaders only; a request for held-out loaders is refused before loading data."""
    config = configs.resolve(configs.transient_stage_a_only_config())
    seed_plan = {"loader": 1, "worker": 2}
    monkeypatch.setattr(
        datasets.runtime.transient_training,
        "create_transient_training_loaders",
        lambda **_kwargs: pytest.fail("a refused held-out request loaded data"),
    )
    with pytest.raises(ValueError, match="held-out"):
        experiments.config.loader.create_dataloaders_from_config(config, seed_plan=seed_plan, held_out_loaders=True)

    built = SimpleNamespace(
        train="train",
        evaluation="validation",
        scaling="scaling",
        scaling_artifact={},
        split={},
        spatial_view={},
    )
    monkeypatch.setattr(datasets.runtime.transient_training, "create_transient_training_loaders", lambda **_kwargs: built)
    loaders = experiments.config.loader.create_dataloaders_from_config(config, seed_plan=seed_plan)

    assert (loaders["train"], loaders["eval"]) == ("train", "validation")
    assert not set(loaders).intersection(_HELD_OUT_ROLES)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [("dataset_identity", {"dataset_id": "foreign"}), ("train_membership_digest", "b" * 64)],
)
def test_saved_transient_scaling_state_must_repeat_the_envelope_identity(transient_package: dict[str, Any], field: str, foreign: Any) -> None:
    """Admit the saved scaling state of a matching envelope only when its repeated identity fields equal the envelope identity."""
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=transient_package["train_dataset"],
        material_ood_dataset_ids=[],
        train_subset=None,
        tensorizer=TransientTensorizerSpec(input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("none")),
        train_sampling=sampling,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        units_per_update=2,
        memory_batch=1,
        model_sha256="a" * 64,
        storage_root=str(transient_package["root"]),
    )
    for loader in (loaders.train, loaders.evaluation):
        _close_dataset(loader.dataset)
    config = {
        "task": "transient_drying",
        "task_contract": domain.tasks.registry.get_task("transient_drying").resolved_contract(),
        "data": {
            "train_dataset": transient_package["train_dataset"],
            "material_ood_datasets": [],
            "train_subset": None,
            "spatial_stride": 1,
        },
        "input_profile": "canonical_physics_complete",
        "temporal": {"temporal_conditioning": {"kind": "none"}, "sampling": sampling.as_dict()},
    }
    split = datasets.preprocessing.splits.admit_split(loaders.split)
    envelope = copy.deepcopy(loaders.scaling_artifact)

    admitted: Any = experiments.run._validate_saved_data_contract(config, split, envelope)
    assert admitted.digest == loaders.scaling.digest
    envelope["state"][field] = foreign
    with pytest.raises(experiments.run.RunLifecycleError, match=field):
        experiments.run._validate_saved_data_contract(config, split, envelope)
