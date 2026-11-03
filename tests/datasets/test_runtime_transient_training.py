"""Protect transient training Dataset assembly, normalization, and split boundaries.

Responsibilities:
  - Check stratified partitions, exact source membership, spatial views, loaders, and scaler caches

Design principles:
  - Case splits precede transition expansion and reject physical-source leakage

This module does NOT:
  - Train transient models or alter canonical source cases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from src import common, domain
from src.datasets.contracts import dataset_contracts_partition as partition
from src.datasets.contracts.dataset_contracts_transient import TransientSamplingSpec, rollout_window_starts
from src.datasets.runtime import dataset_runtime_transient as transient
from src.datasets.runtime import dataset_runtime_transient_training as training
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import (
    TransientTensorizerSpec,
)

if TYPE_CHECKING:
    from src.learning.transient.learning_transient_scaling import TransientScaleMode


def test_experiment_partition_preserves_material_counts_and_rejects_small_strata() -> None:
    research_cases = tuple(f"{material}_{index}" for material in ("lentil", "chickpea") for index in range(50))
    materials = {case_id: case_id.split("_", 1)[0] for case_id in research_cases}
    recipe = {"kind": "seeded_material_fraction", "train_fraction": 0.8, "validation_fraction": 0.1, "id_test_fraction": 0.1}
    first = training._resolve_case_partition(research_cases, recipe, split_seed=9, material_by_case=materials)
    second = training._resolve_case_partition(research_cases, recipe, split_seed=9, material_by_case=materials)
    assert first == second
    for material in ("lentil", "chickpea"):
        assert [sum(materials[case_id] == material for case_id in first[role]) for role in ("train", "validation", "id_test")] == [40, 5, 5]
    assert {case_id for cases in first.values() for case_id in cases} == set(research_cases)

    smoke_cases = tuple(f"lentil_{index}" for index in range(4))
    smoke_materials = dict.fromkeys(smoke_cases, "lentil")
    smoke_recipe = {"kind": "seeded_material_fraction", "train_fraction": 0.5, "validation_fraction": 0.25, "id_test_fraction": 0.25}
    smoke = training._resolve_case_partition(smoke_cases, smoke_recipe, split_seed=9, material_by_case=smoke_materials)
    assert [len(smoke[role]) for role in ("train", "validation", "id_test")] == [2, 1, 1]
    with pytest.raises(ValueError, match="without a complete case"):
        training._resolve_case_partition(smoke_cases[:3], smoke_recipe, split_seed=9, material_by_case=dict.fromkeys(smoke_cases[:3], "lentil"))


def test_experiment_partition_rejects_incomplete_source_evidence_and_nonfinite_recipe() -> None:
    cases = ("a", "b", "c", "d")
    recipe = {"kind": "seeded_material_fraction", "train_fraction": 0.5, "validation_fraction": 0.25, "id_test_fraction": 0.25}
    with pytest.raises(ValueError, match="exact source material"):
        training._resolve_case_partition(cases, recipe, split_seed=9, material_by_case={"a": "lentil"})
    with pytest.raises(ValueError, match="finite"):
        training._resolve_case_partition(
            cases,
            {**recipe, "train_fraction": float("nan")},
            split_seed=9,
            material_by_case=dict.fromkeys(cases, "lentil"),
        )


def test_partition_contract_preserves_seeded_order_and_builds_fixed_id_ood_membership() -> None:
    cases = tuple(f"{material}_{index}" for material in ("lentil", "chickpea") for index in range(5))
    materials = {case_id: case_id.split("_", 1)[0] for case_id in cases}
    recipe = {"kind": "seeded_material_fraction", "train_fraction": 0.6, "validation_fraction": 0.2, "id_test_fraction": 0.2}
    expected = {
        "train": ("lentil_0", "lentil_3", "lentil_4", "chickpea_0", "chickpea_4", "chickpea_3"),
        "validation": ("lentil_2", "chickpea_1"),
        "id_test": ("lentil_1", "chickpea_2"),
    }
    assert partition.resolve_case_partition(cases, recipe, split_seed=9, material_by_case=materials) == expected
    assert partition.resolve_case_partition(cases, recipe, split_seed=9, material_by_case=materials) == training._resolve_case_partition(
        cases, recipe, split_seed=9, material_by_case=materials
    )

    fixed_cases = ("lentil_a", "lentil_b", "lentil_c", "wheat_ood", "lentil_d")
    fixed_materials = {case_id: "wheat" if case_id == "wheat_ood" else "lentil" for case_id in fixed_cases}
    fixed_recipe = {"kind": "seeded_material_fraction", "train_fraction": 1 / 3, "validation_fraction": 1 / 3, "id_test_fraction": 1 / 3}
    value = partition.build_fixed_partition(
        fixed_cases,
        material_by_case=fixed_materials,
        material_roles={"lentil": "id", "wheat": "ood"},
        recipe=fixed_recipe,
        seed_input=9,
        seed=1028622135437308793,
    )
    assert value["ood"] == ["wheat_ood"]
    assert set(value["train"] + value["validation"] + value["id_test"]) == set(fixed_cases) - {"wheat_ood"}
    assert partition.validate_fixed_partition(value, case_ids=fixed_cases, material_by_case=fixed_materials) == value
    assert partition.partition_digest(value) == partition.partition_digest(dict(value))
    corrupted = {**value, "ood": ["lentil_a"]}
    with pytest.raises(ValueError, match="cover source cases once"):
        partition.validate_fixed_partition(corrupted, case_ids=fixed_cases, material_by_case=fixed_materials)

    too_small_recipe = {"kind": "seeded_material_fraction", "train_fraction": 0.8, "validation_fraction": 0.1, "id_test_fraction": 0.1}
    with pytest.raises(ValueError, match="without a complete case"):
        partition.build_fixed_partition(
            ("lentil_a", "lentil_b", "lentil_c"),
            material_by_case={"lentil_a": "lentil", "lentil_b": "lentil", "lentil_c": "lentil"},
            material_roles={"lentil": "id"},
            recipe=too_small_recipe,
            seed_input=9,
            seed=1028622135437308793,
        )


def _dataset(*, mode: str) -> transient.TransientPhysicalDataset:
    dataset = object.__new__(transient.TransientPhysicalDataset)
    dataset.sample_indices = (0, 1, 2, 3)
    dataset.payload = {
        "cases": [
            {"package_case_id": "case_b"},
            {"package_case_id": "case_a"},
        ],
        "samples": [
            {"case_index": 0, "sample_id": "case_b__step_0000", "time_index_n": 0, "time_index_n_plus_1": 1},
            {"case_index": 0, "sample_id": "case_b__step_0001", "time_index_n": 1, "time_index_n_plus_1": 2},
            {"case_index": 1, "sample_id": "case_a__step_0000", "time_index_n": 0, "time_index_n_plus_1": 1},
            {"case_index": 1, "sample_id": "case_a__step_0001", "time_index_n": 1, "time_index_n_plus_1": 2},
        ],
    }
    if mode == "one_step_transition":
        dataset.sampling = TransientSamplingSpec(mode="one_step_transition")
        dataset._item_references = ()
    else:
        dataset.sampling = TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0)
        dataset._item_references = (
            transient._ItemReference(case_index=0, sample_positions=(0, 1)),
            transient._ItemReference(case_index=1, sample_positions=(2, 3)),
        )
    return dataset


def test_case_arrays_requires_complete_admitted_hdf5_case_and_preserves_reader_contract(monkeypatch: Any) -> None:
    dataset = _dataset(mode="one_step_transition")
    dataset.storage_backend = "canonical_hdf5"
    dataset.spatial_stride = 2
    dataset.payload["configured_regular_horizon"] = {"value": 8.0}
    dataset.sample_indices = (0, 2, 3)
    with pytest.raises(ValueError, match="not fully admitted"):
        dataset.read_case_arrays("case_b")
    dataset.sample_indices = (0, 1, 2)
    with pytest.raises(ValueError, match="does not identify exactly one"):
        dataset.read_case_arrays("absent")
    dataset.storage_backend = "pt_shards"
    with pytest.raises(NotImplementedError, match="canonical HDF5"):
        dataset.read_case_arrays("case_b")
    dataset.storage_backend = "canonical_hdf5"

    class Handle:
        closed = False

        def close(self) -> None:
            self.closed = True

    handle = Handle()
    expected = object()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(dataset, "_validate_runtime_source", lambda case_index: Path(f"/source/{case_index}.h5"))
    monkeypatch.setattr(dataset, "_open_handle", lambda _path: (handle, True))

    def read_arrays(*args: Any, **kwargs: Any) -> object:
        calls.append((*args, kwargs))
        return expected

    monkeypatch.setattr(transient.trajectory, "read_transient_case_arrays", read_arrays)
    assert dataset.read_case_arrays("case_b") is expected
    assert handle.closed
    assert len(calls) == 1
    assert calls[0][2] == dataset.payload["cases"][0]
    assert calls[0][3] == dataset.payload["samples"][:2]
    assert calls[0][4] == {"expected_regular_horizon": 8.0, "spatial_stride": 2}


class _SelectionDataset(transient.TransientPhysicalDataset):
    """Test-only backend-preserving case-selection runtime."""

    def __init__(
        self,
        index_path: str,
        *,
        sampling: TransientSamplingSpec,
        source_root: str,
        hdf5_cache_size: int,
        spatial_stride: int,
        sample_indices: tuple[int, ...],
    ) -> None:
        self.index_path = index_path
        self.sampling = sampling
        self.source_root = source_root
        self.hdf5_cache_size = hdf5_cache_size
        self.spatial_stride = spatial_stride
        self.dataset_manifest_digest = "f" * 64
        self.sample_indices = sample_indices
        self.storage_backend = "pt_shards"
        self.payload = {
            "cases": [{"package_case_id": "b"}, {"package_case_id": "a"}],
            "samples": [
                {"case_index": 0},
                {"case_index": 0},
                {"case_index": 1},
            ],
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_case_selection_reconstructs_same_backend_before_closing_source() -> None:
    source = _SelectionDataset(
        "index.json",
        sampling=TransientSamplingSpec(mode="one_step_transition"),
        source_root="root",
        hdf5_cache_size=3,
        spatial_stride=1,
        sample_indices=(0, 1, 2),
    )

    selected = transient.select_transient_cases(source, ("a",))

    assert isinstance(selected, _SelectionDataset)
    assert selected.storage_backend == "pt_shards"
    assert selected.sample_indices == (2,)
    assert source.closed is True


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (2.0, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_spatial_stride_contract_rejects_invalid_values(value: object, error: type[Exception]) -> None:
    """Reject implicit coercion while keeping the stride surface exact."""
    with pytest.raises(error):
        training.transient_contract.validate_spatial_stride(value)


def test_boundary_preserving_spatial_view_resolves_canonical_grid() -> None:
    """Keep both physical boundaries when subsampling 251x401 by two."""
    assert training.transient_contract.resolve_spatial_view((251, 401), 1) == (251, 401)
    assert training.transient_contract.resolve_spatial_view((251, 401), 2) == (126, 201)
    representation = training.transient_contract.resolve_spatial_representation((251, 401), 2)
    assert representation.source_shape == (251, 401)
    assert representation.represented_shape == (126, 201)
    assert representation.y_indices == tuple(range(0, 251, 2))
    assert representation.x_indices == tuple(range(0, 401, 2))
    assert representation.y_indices[-1] == 250
    assert representation.x_indices[-1] == 400
    assert representation.as_dict()["index_identity_sha256"] == representation.index_identity_sha256
    assert len(representation.index_identity_sha256) == 64
    assert training.transient_contract.resolve_spatial_representation((251, 401), 1).index_identity_sha256 != representation.index_identity_sha256
    with pytest.raises(ValueError, match="preserve both boundaries"):
        training.transient_contract.resolve_spatial_view((250, 401), 2)


def test_role_evidence_keeps_case_membership_before_item_expansion() -> None:
    dataset = _dataset(mode="rollout_window")

    evidence = training._role_evidence(dataset)

    assert evidence["case_ids"] == ["case_b", "case_a"]
    assert evidence["item_ids"] == ["case_b__window_0000_0002", "case_a__window_0000_0002"]
    assert len(evidence["membership_digest"]) == 64


def test_one_step_evidence_records_transition_ids_for_scaling() -> None:
    evidence = training._role_evidence(_dataset(mode="one_step_transition"))

    assert evidence["item_ids"] == [
        "case_b__step_0000",
        "case_b__step_0001",
        "case_a__step_0000",
        "case_a__step_0001",
    ]


def test_saved_split_replay_requires_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensorizer = TransientTensorizerSpec(
        input_profile="canonical_physics_complete_v1",
        temporal_conditioning=TemporalConditioningSpec("normalized_current_time"),
    )
    current = {
        "schema_kind": "experiment_transient_case_split",
        "tensorizer": tensorizer.selection_dict(),
        "sampling": TransientSamplingSpec(mode="one_step_transition").as_dict(),
        "evaluation_sampling": training.EVALUATION_SAMPLING.as_dict(),
        "additional_evaluation_fraction": 1.0,
        "split_seed": 9,
        "spatial_view": {
            "spatial_stride": 1,
            "canonical_ny": 251,
            "canonical_nx": 401,
            "effective_ny": 251,
            "effective_nx": 401,
        },
    }
    monkeypatch.setattr(
        training,
        "admit_transient_training_split",
        lambda value, **_kwargs: dict(value),
    )

    assert training._admit_saved_split(current, current) == current
    with pytest.raises(ValueError, match="does not exactly match"):
        training._admit_saved_split({**current, "split_seed": 10}, current)


def test_standalone_split_admission_recomputes_membership_digests() -> None:
    tensorizer = TransientTensorizerSpec(
        input_profile="canonical_physics_complete_v1",
        temporal_conditioning=TemporalConditioningSpec("normalized_current_time"),
    )
    sampling = TransientSamplingSpec(mode="one_step_transition")
    task = domain.tasks.registry.get_task("transient_drying")

    def role(case_id: str, item_id: str) -> dict[str, object]:
        evidence = {"case_ids": (case_id,), "item_ids": (item_id,)}
        return {
            "case_ids": [case_id],
            "item_ids": [item_id],
            "membership_digest": training._sha256(evidence),
        }

    identity = {
        "dataset_id": "synthetic",
        "data_contract_digest": task.data_contract_digest,
        "index_digest": "1" * 64,
        "configured_regular_horizon": {"value": 32.0, "unit": "h"},
    }
    split = {
        "schema_kind": "experiment_transient_case_split",
        "schema_version": training.SPLIT_SCHEMA_VERSION,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "data_contract_digest": task.data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": sampling.as_dict(),
        "evaluation_sampling": training.EVALUATION_SAMPLING.as_dict(),
        "dataset_identity": {
            "train": identity,
            "additional_evaluation": [{**identity, "dataset_id": "synthetic_added"}],
        },
        "additional_evaluation_fraction": 1.0,
        "split_seed": 9,
        "source_case_ids": ["train_case", "eval_case", "test_case"],
        "source_case_materials": dict.fromkeys(("train_case", "eval_case", "test_case"), "lentil"),
        "case_partition": {"kind": "exact", "train": ["train_case"], "validation": ["eval_case"], "id_test": ["test_case"]},
        "evaluation_assignments": [
            {"dataset_id": "synthetic_added", "role": "stress", "relation": "fixture comparison", "assumptions": ["fixture stress protocol"]},
        ],
        "spatial_view": {
            "spatial_stride": 1,
            "canonical_ny": 251,
            "canonical_nx": 401,
            "effective_ny": 251,
            "effective_nx": 401,
        },
        "runtime_provenance": {
            "train": "pt_shards",
            "scaling_train_one_step": "pt_shards",
            "validation": "pt_shards",
            "id_test": "pt_shards",
            "additional_evaluation": ["pt_shards"],
        },
        "roles": {
            "train": role("train_case", "train_item"),
            "scaling_train_one_step": role(
                "train_case",
                "scaling_item",
            ),
            "validation": role("eval_case", "eval_item"),
            "id_test": role("test_case", "test_item"),
            "additional_evaluation": {"parts": [role("added_case", "added_item")]},
        },
    }

    split["case_partition"] = {
        "seed_input": 9,
        "seed": 9,
        "recipe": split["case_partition"],
        "material_roles": {"lentil": "id"},
        "train": ["train_case"],
        "validation": ["eval_case"],
        "id_test": ["test_case"],
        "ood": [],
    }
    split["case_partition_digest"] = partition.partition_digest(split["case_partition"])

    admitted = training.admit_transient_training_split(
        split,
        tensorizer=tensorizer,
        sampling=sampling,
        additional_evaluation_fraction=1.0,
        split_seed=9,
    )
    assert admitted == split

    stride_two = {
        **split,
        "spatial_view": {
            "spatial_stride": 2,
            "canonical_ny": 251,
            "canonical_nx": 401,
            "effective_ny": 126,
            "effective_nx": 201,
        },
    }
    assert (
        training.admit_transient_training_split(
            stride_two,
            tensorizer=tensorizer,
            sampling=sampling,
            additional_evaluation_fraction=1.0,
            split_seed=9,
            spatial_stride=2,
        )["spatial_view"]["effective_nx"]
        == 201
    )
    with pytest.raises(ValueError, match="spatial stride"):
        training.admit_transient_training_split(
            split,
            tensorizer=tensorizer,
            sampling=sampling,
            additional_evaluation_fraction=1.0,
            split_seed=9,
            spatial_stride=2,
        )

    historical = {**split, "schema_kind": "transient_drying_training_split", "schema_version": 2}
    with pytest.raises(ValueError, match="Historical"):
        training.admit_transient_training_split(
            historical,
            tensorizer=tensorizer,
            sampling=sampling,
            additional_evaluation_fraction=1.0,
            split_seed=9,
        )

    without_added = {
        **split,
        "dataset_identity": {"train": identity, "additional_evaluation": []},
        "evaluation_assignments": [],
        "runtime_provenance": {**split["runtime_provenance"], "additional_evaluation": []},
        "roles": {**split["roles"], "additional_evaluation": {"parts": []}},
    }
    assert (
        training.admit_transient_training_split(
            without_added,
            tensorizer=tensorizer,
            sampling=sampling,
            additional_evaluation_fraction=1.0,
            split_seed=9,
        )
        == without_added
    )

    split["roles"]["train"]["membership_digest"] = "0" * 64
    with pytest.raises(ValueError, match="membership digest"):
        training.admit_transient_training_split(
            split,
            tensorizer=tensorizer,
            sampling=sampling,
            additional_evaluation_fraction=1.0,
            split_seed=9,
        )


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_transient_loader_workers_preserve_order_and_runtime_flags(workers: int) -> None:
    """Exercise safe CPU loader semantics for the bounded operator candidates."""
    settings = training.factory.LoaderSettings(
        batch_size=2,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    loader = training._loader(
        TensorDataset(torch.arange(6)),
        settings,
        shuffle=False,
        loader_seed=11,
        worker_seed=12,
    )

    observed = torch.cat([batch[0] for batch in loader]).tolist()
    assert observed == list(range(6))
    assert loader.num_workers == workers
    assert loader.pin_memory is True
    assert loader.persistent_workers is (workers > 0)
    iterator = loader._iterator
    if iterator is not None:
        cast("Any", iterator)._shutdown_workers()


def test_additional_evaluation_assignments_preserve_declared_id_ood_and_stress() -> None:
    assignments = [
        {"dataset_id": "near", "role": "near_family_ood", "relation": "held-out family", "assumptions": ["family support is separate"]},
        {"dataset_id": "id", "role": "id", "relation": "replicate under training support", "assumptions": []},
        {"dataset_id": "stress", "role": "stress", "relation": "elevated schedule", "assumptions": ["within declared bounds"]},
    ]
    assert training._admit_evaluation_assignments(assignments, ["near", "id", "stress"]) == assignments
    assert training._admit_evaluation_assignments([], []) == []


def test_additional_evaluation_assignments_reject_implicit_or_mismatched_roles() -> None:
    assignment = {"dataset_id": "added", "role": "stress", "relation": "elevated schedule", "assumptions": []}
    with pytest.raises(ValueError, match="align"):
        training._admit_evaluation_assignments([assignment], [])
    with pytest.raises(ValueError, match="disagrees"):
        training._admit_evaluation_assignments([assignment], ["different"])
    with pytest.raises(ValueError, match="disagrees"):
        training._admit_evaluation_assignments([{**assignment, "role": "inferred_ood"}], ["added"])
    with pytest.raises(ValueError, match="relation"):
        training._admit_evaluation_assignments([{**assignment, "relation": ""}], ["added"])


def test_experiment_role_leakage_rejects_repeated_physical_source() -> None:
    def selected(case_id: str, physical_id: str) -> transient.TransientPhysicalDataset:
        dataset = object.__new__(transient.TransientPhysicalDataset)
        dataset.sample_indices = (0,)
        dataset.payload = {
            "cases": [{"package_case_id": case_id, "case_input_id": physical_id, "simulation_case_id": physical_id}],
            "samples": [{"case_index": 0}],
        }
        return dataset

    train = selected("train_alias", "source_a")
    validation = selected("validation_alias", "source_b")
    test = selected("test_alias", "source_c")
    training._reject_role_leakage(train, validation, test, [selected("stress", "source_d")])
    with pytest.raises(ValueError, match="reuses a physical source"):
        training._reject_role_leakage(train, validation, test, [selected("duplicate_alias", "source_a")])


def test_scaler_cache_reuses_exact_identity_and_rejects_corruption_or_stride_handoff(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit once per scientific view and fail closed on cache or handoff drift."""
    root = Path(str(tmp_path))
    tensorizer = TransientTensorizerSpec(
        input_profile="canonical_physics_complete_v1",
        temporal_conditioning=TemporalConditioningSpec("none"),
    )
    dataset = object.__new__(transient.TransientPhysicalDataset)
    dataset.payload = {"configured_regular_horizon": {"value": 10.0}}
    membership = "1" * 64
    base_source_identity = {"dataset_id": "synthetic", "index_digest": "2" * 64}
    base_identity = {
        **base_source_identity,
        "dataset_manifest_digest": "3" * 64,
        "spatial_stride": 1,
        "canonical_spatial_shape": [3, 3],
        "effective_spatial_shape": [3, 3],
    }
    view_one = {
        "spatial_stride": 1,
        "canonical_ny": 3,
        "canonical_nx": 3,
        "effective_ny": 3,
        "effective_nx": 3,
    }
    fitted: list[training.scaling.TransientScalingArtifact] = []

    def fit_stub(
        _dataset: transient.TransientPhysicalDataset,
        **kwargs: object,
    ) -> training.scaling.TransientScalingArtifact:
        identity = dict(cast("dict[str, object]", kwargs["dataset_identity"]))
        ny, nx = cast("list[int]", identity["effective_spatial_shape"])

        def item(index: int) -> dict[str, object]:
            state = torch.arange(4 * ny * nx, dtype=torch.float32).reshape(4, ny, nx) + index
            return {
                "state": state,
                "target": torch.ones_like(state),
                "static": torch.arange(7 * ny * nx, dtype=torch.float32).reshape(7, ny, nx),
                "boundary": torch.tensor([300.0, 301.0, 0.1, 0.2, 295.0, 0.5, 302.0, 0.3, 0.0]),
                "scalars": torch.arange(8, dtype=torch.float32),
                "time": {
                    "t_n": torch.tensor(float(index)),
                    "t_n_plus_1": torch.tensor(float(index + 1)),
                    "dt": torch.tensor(1.0),
                },
                "metadata": {
                    "simulation_case_id": "case",
                    "time_index_n": index,
                    "time_index_n_plus_1": index + 1,
                },
            }

        artifact = training.scaling.fit_transient_scaling(
            [item(0), item(1)],
            tensorizer=cast("TransientTensorizerSpec", kwargs["tensorizer"]),
            dataset_identity=identity,
            train_membership_digest=cast("str", kwargs["train_membership_digest"]),
            horizon=10.0,
            scale_mode=cast("TransientScaleMode", kwargs["scale_mode"]),
        )
        fitted.append(artifact)
        return artifact

    monkeypatch.setattr(training, "_fit_scaling_artifact", fit_stub)
    events: list[tuple[str, dict[str, object]]] = []

    def load(
        identity: dict[str, object],
        digest: str,
        mode: TransientScaleMode,
        view: dict[str, int],
    ) -> training.scaling.TransientScalingArtifact:
        return training._fit_or_load_scaling_artifact(
            dataset,
            tensorizer=tensorizer,
            dataset_identity=identity,
            train_membership_digest=digest,
            scale_mode=mode,
            spatial_view=view,
            storage_root=str(root),
            startup_callback=lambda event, values: events.append((event, dict(values))),
        )

    first = load(base_identity, membership, "state_std", view_one)
    warm = load(base_identity, membership, "state_std", view_one)
    assert warm.state_dict() == first.state_dict()
    assert len(fitted) == 1
    assert [values["state"] for event, values in events if event == "scaler_cache"][:2] == ["miss", "hit"]

    changed_dataset = {**base_identity, "dataset_manifest_digest": "4" * 64}
    load(changed_dataset, membership, "state_std", view_one)
    load(base_identity, "5" * 64, "state_std", view_one)
    view_two = {
        "spatial_stride": 2,
        "canonical_ny": 3,
        "canonical_nx": 3,
        "effective_ny": 2,
        "effective_nx": 2,
    }
    stride_two_identity = {
        **base_identity,
        "spatial_stride": 2,
        "effective_spatial_shape": [2, 2],
    }
    load(stride_two_identity, membership, "state_std", view_two)
    load(base_identity, membership, "delta_rms", view_one)
    assert len(fitted) == 5

    with pytest.raises(ValueError, match="does not match"):
        training._validate_scaling_artifact(
            first,
            tensorizer=tensorizer,
            dataset_identity=stride_two_identity,
            train_membership_digest=membership,
            scale_mode="state_std",
            spatial_view=view_two,
            horizon=10.0,
        )

    cache_identity = training._scaler_cache_identity(
        tensorizer=tensorizer,
        dataset_identity=base_identity,
        train_membership_digest=membership,
        scale_mode="state_std",
        horizon=10.0,
    )
    cache_path = common.paths.get_transient_scaler_cache_root(storage_root=root) / f"{training._sha256(cache_identity)}.json"
    cache_path.write_text(json.dumps({"broken": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache identity"):
        load(base_identity, membership, "state_std", view_one)


def test_dataset_declared_ood_rejects_same_material_even_for_distinct_cases() -> None:
    train = _dataset(mode="one_step_transition")
    additional = _dataset(mode="one_step_transition")
    for case in train.payload["cases"]:
        case["material_family"] = "lentil"
    for case in additional.payload["cases"]:
        case["package_case_id"] = "additional_" + case["package_case_id"]
        case["material_family"] = "chickpea"
    assignments = [{"dataset_id": "additional", "role": "ood", "relation": "Dataset-declared material holdout", "assumptions": []}]
    assert not set(training._case_positions(train)).intersection(training._case_positions(additional))
    training._validate_material_holdouts(train, assignments, [additional])
    for case in additional.payload["cases"]:
        case["material_family"] = "lentil"
    with pytest.raises(ValueError, match="reuses a training material"):
        training._validate_material_holdouts(train, assignments, [additional])


def test_fixed_ood_fraction_override_is_rejected_before_loading_any_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a requested fraction that would otherwise leave own-Dataset OOD unchanged."""
    monkeypatch.setattr(training.factory, "create_dataset", lambda *_args, **_kwargs: pytest.fail("fraction rejected after Dataset access"))
    with pytest.raises(ValueError, match="membership is fixed"):
        training.create_transient_training_loaders(
            train_dataset_id="unopened",
            additional_evaluation_dataset_ids=(),
            tensorizer=TransientTensorizerSpec(
                input_profile="canonical_physics_complete_v1",
                temporal_conditioning=TemporalConditioningSpec("normalized_current_time"),
            ),
            train_sampling=TransientSamplingSpec(mode="one_step_transition"),
            loader_settings=training.factory.LoaderSettings(),
            additional_evaluation_fraction=0.5,
        )


def test_runtime_uses_shared_declared_assignments_before_ood_payload_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the public projection seam without loading tensors or fitting scalers."""
    dataset = object.__new__(transient.TransientPhysicalDataset)
    dataset.payload = {
        "cases": [{"package_case_id": case, "material_family": "lentil"} for case in ("a", "b", "c")],
        "samples": [{"case_index": index} for index in range(3)],
    }
    dataset.sample_indices = (0, 1, 2)
    dataset.sampling = TransientSamplingSpec(mode="one_step_transition")
    dataset._item_references = dataset._build_item_references()
    dataset.dataset_case_partition = partition.build_fixed_partition(
        ("a", "b", "c"),
        material_by_case=dict.fromkeys(("a", "b", "c"), "lentil"),
        material_roles={"lentil": "id"},
        recipe={"kind": "exact", "train": ["a"], "validation": ["b"], "id_test": ["c"]},
        seed_input=9,
        seed=1028622135437308793,
    )
    loaded: list[str] = []

    def create(request: Any, **_kwargs: Any) -> transient.TransientPhysicalDataset:
        loaded.append(request.dataset_id)
        return dataset

    projected: list[tuple[str, str, tuple[str, ...], str | None]] = []

    def declared(task: str, train_id: str, extra_ids: Any, *, storage_root: str | None = None) -> list[dict[str, Any]]:
        projected.append((task, train_id, tuple(extra_ids), storage_root))
        message = "metadata projection seam reached"
        raise RuntimeError(message)

    monkeypatch.setattr(training.factory, "create_dataset", create)
    monkeypatch.setattr(training, "_select_cases", lambda source, _ids: source)
    monkeypatch.setattr(training.packages, "declared_evaluation_assignments", declared)
    with pytest.raises(RuntimeError, match="metadata projection seam reached"):
        training.create_transient_training_loaders(
            train_dataset_id="fixed-primary",
            additional_evaluation_dataset_ids=("external",),
            tensorizer=TransientTensorizerSpec(
                input_profile="canonical_physics_complete_v1", temporal_conditioning=TemporalConditioningSpec("normalized_current_time")
            ),
            train_sampling=TransientSamplingSpec(mode="one_step_transition"),
            loader_settings=training.factory.LoaderSettings(),
            storage_root="metadata-only",
        )
    assert projected == [("transient_drying", "fixed-primary", ("external",), "metadata-only")]
    # The primary Dataset opens once per sample shape: Train items and whole evaluation trajectories.
    assert loaded == ["fixed-primary", "fixed-primary"]


def _trajectory_dataset(sampling: TransientSamplingSpec, *, lengths: tuple[int, ...]) -> transient.TransientPhysicalDataset:
    """Index consecutive regular transitions of several cases without opening any source."""
    dataset = object.__new__(transient.TransientPhysicalDataset)
    samples = [
        {"case_index": case, "time_index_n": step, "time_index_n_plus_1": step + 1, "schedule_index_n": step, "schedule_index_n_plus_1": step + 1}
        for case, length in enumerate(lengths)
        for step in range(length)
    ]
    dataset.payload = {"cases": [{"package_case_id": f"case_{case}"} for case in range(len(lengths))], "samples": samples}
    dataset.sample_indices = tuple(range(len(samples)))
    dataset.sampling = sampling
    dataset._item_references = dataset._build_item_references()
    return dataset


def test_rollout_windows_cover_the_trajectory_end_and_trajectories_stay_whole() -> None:
    """Add one end-aligned window when the stride stops short; trajectory items keep every transition."""
    assert rollout_window_starts(168, rollout_length=32, window_stride=32, window_offset=0) == (0, 32, 64, 96, 128, 136)
    assert rollout_window_starts(64, rollout_length=32, window_stride=32, window_offset=0) == (0, 32)
    assert rollout_window_starts(20, rollout_length=32, window_stride=32, window_offset=0) == ()

    windows = _trajectory_dataset(
        TransientSamplingSpec(mode="rollout_window", rollout_length=4, window_stride=4, window_offset=0), lengths=(6, 6)
    )._build_item_references()
    assert [reference.sample_positions for reference in windows] == [(0, 1, 2, 3), (2, 3, 4, 5), (6, 7, 8, 9), (8, 9, 10, 11)]

    trajectories = _trajectory_dataset(TransientSamplingSpec(mode="trajectory", rollout_length=4), lengths=(6, 6))._build_item_references()
    assert [reference.sample_positions for reference in trajectories] == [tuple(range(6)), tuple(range(6, 12))]
    # Cases stop at their own drying target, so whole trajectories of one selection may differ in length.
    unequal = _trajectory_dataset(TransientSamplingSpec(mode="trajectory", rollout_length=4), lengths=(6, 5))
    assert unequal.item_transition_counts() == (6, 5)
    assert unequal.cases_without_items() == ()
    # A case shorter than the training window yields no item; the training loaders reject such a Train membership.
    short = _trajectory_dataset(TransientSamplingSpec(mode="rollout_window", rollout_length=4, window_stride=4, window_offset=0), lengths=(6, 3))
    assert short.cases_without_items() == ("case_1",)

    # Evaluation sets no trainer window, so even a trajectory shorter than any window stays one item.
    assert training.EVALUATION_SAMPLING.as_dict() == {"mode": "trajectory"}
    short = _trajectory_dataset(training.EVALUATION_SAMPLING, lengths=(3, 3))._build_item_references()
    assert [reference.sample_positions for reference in short] == [(0, 1, 2), (3, 4, 5)]
    with pytest.raises(ValueError, match="no complete rollout window"):
        _trajectory_dataset(TransientSamplingSpec(mode="trajectory", rollout_length=4), lengths=(3, 3))._build_item_references()
    with pytest.raises(ValueError, match="no rollout windows"):
        training.EVALUATION_SAMPLING.window_starts(6)
    with pytest.raises(ValueError, match="accepts only rollout_length"):
        TransientSamplingSpec(mode="trajectory", rollout_length=4, window_stride=4)


def _batches(loader: Any) -> list[list[int]]:
    """Return one epoch of a loader's index batches."""
    assert loader.batch_sampler is not None
    return [list(batch) for batch in loader.batch_sampler]


def test_trajectory_batches_never_mix_lengths_or_evaluation_parts() -> None:
    """
    Batch whole trajectories that stop at different drying times.

    Evaluation batches keep their item order within one length of one Dataset
    part; shuffled Train batches hold one length, cover every item once per
    epoch and replay from the loader seed. Equal-length windows keep the
    ordinary DataLoader batching.
    """
    settings = training.factory.LoaderSettings(batch_size=2)
    first = _trajectory_dataset(training.EVALUATION_SAMPLING, lengths=(6, 5, 6, 6))
    second = _trajectory_dataset(training.EVALUATION_SAMPLING, lengths=(6,))
    evaluation = training._loader(ConcatDataset([first, second]), settings, shuffle=False, loader_seed=0, worker_seed=0)
    assert _batches(evaluation) == [[0, 2], [1], [3], [4]]

    train_sampling = TransientSamplingSpec(mode="trajectory", rollout_length=2)
    lengths = (6, 5, 6, 5, 6)

    def train_epochs(seed: int) -> list[list[list[int]]]:
        loader = training._loader(_trajectory_dataset(train_sampling, lengths=lengths), settings, shuffle=True, loader_seed=seed, worker_seed=0)
        return [_batches(loader) for _ in range(2)]

    epochs = train_epochs(5)
    for batches in epochs:
        assert sorted(item for batch in batches for item in batch) == list(range(5))
        assert all(len({lengths[item] for item in batch}) == 1 and len(batch) <= 2 for batch in batches)
    assert train_epochs(5) == epochs

    windows = _trajectory_dataset(TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=2, window_offset=0), lengths=(6, 5))
    plain = training._loader(windows, settings, shuffle=True, loader_seed=0, worker_seed=0)
    assert plain.batch_size == 2
