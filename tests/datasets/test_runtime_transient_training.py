"""Protect transient training Dataset assembly and case-role boundaries.

Responsibilities:
  - Check that the persisted Dataset contract names the tensors a runtime item carries
  - Check design-kind partitions, Train subsets, exact source membership, spatial views, loaders, and declared OOD assignments
  - Check the full Train role's unit and transition counts N that the stage budgets read, over unequal trajectory lengths

Design principles:
  - Case splits precede transition expansion; role leakage is the shared split contract's test

This module does NOT:
  - Train transient models or alter canonical source cases
  - Test the shared split schema or scaling cache, which both tasks share
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch.utils.data import TensorDataset

from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.contracts.dataset_contracts_transient import EVALUATION_SAMPLING, TransientSamplingSpec, consecutive_windows, rollout_windows
from src.datasets.runtime import dataset_runtime_transient as transient
from src.datasets.runtime import dataset_runtime_transient_training as training
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import (
    TransientTensorizerSpec,
)

if TYPE_CHECKING:
    from typing import Self


_CASES = {
    "lentil_t0": ("lentil", "train"),
    "lentil_t1": ("lentil", "train"),
    "lentil_v0": ("lentil", "validation"),
    "lentil_s0": ("lentil", "test"),
    "lentil_p0": ("lentil", "parameter_ood"),
    "chickpea_t0": ("chickpea", "train"),
    "chickpea_t1": ("chickpea", "train"),
    "chickpea_v0": ("chickpea", "validation"),
    "chickpea_s0": ("chickpea", "test"),
    "wheat_s0": ("wheat", "test"),
}
_MATERIALS = {case_id: material for case_id, (material, _kind) in _CASES.items()}
_KINDS = {case_id: kind for case_id, (_material, kind) in _CASES.items()}
_ROLES = {"lentil": "id", "chickpea": "id", "wheat": "ood"}


def test_the_persisted_contract_names_exactly_the_tensors_a_runtime_item_carries() -> None:
    """The Dataset contract lists every field group of a runtime item; the reference is the stored next state over the state fields."""
    payload = transient_contract.transient_contract_payload()
    field_groups = {name for name, value in payload.items() if isinstance(value, list)}
    assert field_groups | {"time", "metadata"} == set(transient.TransientItem.__annotations__)
    assert payload["next_state"] == payload["state"]
    inspection = views.inspect_contract("transient_drying")
    reference = inspection.group("next_state")
    assert reference.purpose == "target"
    assert [(field.name, field.unit) for field in reference.fields] == [(field["name"], field["unit"]) for field in payload["state"]]
    assert inspection.target_semantics == "stored_next_state"


def test_partition_follows_the_design_kinds_in_source_order() -> None:
    """Each case's role is its material role and design kind; Train keeps source order; validation rebuilds and compares."""
    value = partition.build_fixed_partition(tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case=_KINDS, material_roles=_ROLES)
    assert {role: value[role] for role in partition.CASE_ROLES} == {
        "train": ["lentil_t0", "lentil_t1", "chickpea_t0", "chickpea_t1"],
        "validation": ["lentil_v0", "chickpea_v0"],
        "id_test": ["lentil_s0", "chickpea_s0"],
        "parameter_ood": ["lentil_p0"],
        "ood": ["wheat_s0"],
    }
    assert partition.evaluation_cases(value, "material_ood") == ("wheat_s0",)
    assert partition.evaluation_cases(value, "parameter_ood") == ("lentil_p0",)
    validated = partition.validate_fixed_partition(value, case_ids=tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case=_KINDS)
    assert validated == value
    assert partition.partition_digest(value) == partition.partition_digest(dict(value))
    with pytest.raises(ValueError, match="roles must be disjoint"):
        partition.validate_fixed_partition(
            {**value, "ood": ["lentil_t0"]}, case_ids=tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case=_KINDS
        )
    swapped = {**value, "train": ["lentil_t0", "lentil_v0", "chickpea_t0", "chickpea_t1"], "validation": ["lentil_t1", "chickpea_v0"]}
    with pytest.raises(ValueError, match="membership does not match"):
        partition.validate_fixed_partition(swapped, case_ids=tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case=_KINDS)


def test_partition_refuses_a_design_kind_its_material_role_cannot_hold_and_incomplete_evidence() -> None:
    """An OOD material holds only its test design; every case needs its material and design kind."""
    with pytest.raises(ValueError, match="which a ood material cannot hold"):
        partition.build_fixed_partition(
            ("wheat_t0",), material_by_case={"wheat_t0": "wheat"}, design_kind_by_case={"wheat_t0": "train"}, material_roles={"wheat": "ood"}
        )
    with pytest.raises(ValueError, match="exact source design kind"):
        partition.build_fixed_partition(tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case={}, material_roles=_ROLES)


def test_train_subset_takes_the_first_train_cases_of_every_id_material() -> None:
    """A Train subset is a per-material prefix in stored order; a size beyond any material's Train cases is refused."""
    value = partition.build_fixed_partition(tuple(_CASES), material_by_case=_MATERIALS, design_kind_by_case=_KINDS, material_roles=_ROLES)
    assert partition.train_subset_cases(value, material_by_case=_MATERIALS, train_subset=None) == tuple(value["train"])
    assert partition.train_subset_cases(value, material_by_case=_MATERIALS, train_subset=1) == ("lentil_t0", "chickpea_t0")
    with pytest.raises(ValueError, match=r"data.train_subset 4 exceeds the Train cases of \['chickpea', 'lentil'\]"):
        partition.train_subset_cases(value, material_by_case=_MATERIALS, train_subset=4)


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

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.closed = True

    handle = Handle()
    expected = object()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(dataset, "_validate_runtime_source", lambda case_index: Path(f"/source/{case_index}.h5"))
    monkeypatch.setattr(transient.h5py, "File", lambda _path, _mode: handle)

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
        spatial_stride: int,
        sample_indices: tuple[int, ...],
    ) -> None:
        self.index_path = index_path
        self.sampling = sampling
        self.source_root = source_root
        self.spatial_stride = spatial_stride
        self.manifest_binding = None
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


def test_pt_dataset_keeps_every_opened_shard_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alternating reads and scaling iteration load each shard once; a new process reloads them."""
    loads: list[int] = []
    cases = [{"package_case_id": "a"}, {"package_case_id": "b"}]

    def load_payload(_dataset_id: str, shard_index: int, **_kwargs: Any) -> dict[str, Any]:
        loads.append(shard_index)
        states = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1) + 10.0 * shard_index
        case = {"case_index": shard_index, "case_record": cases[shard_index], "states": states, "static": torch.zeros(1, 1, 1)}
        return {"shard": shard_index, "cases": [case]}

    monkeypatch.setattr(transient.transient_shards, "load_transient_shard_payload", load_payload)
    dataset = object.__new__(transient.TransientPTShardDataset)
    dataset.payload = {"dataset_id": "dataset", "cases": cases}
    dataset.source_root = Path("root")
    dataset._shard_receipt = {"case_locator": {"a": {"shard_index": 0, "case_position": 0}, "b": {"shard_index": 1, "case_position": 0}}}
    dataset._shard_cache = {}
    dataset._validated_shards = set()
    dataset._process_id = None

    assert [dataset._load_shard(index)["shard"] for index in (0, 1, 0, 1, 0)] == [0, 1, 0, 1, 0]
    assert loads == [0, 1]
    dataset.sampling = TransientSamplingSpec(mode="one_step_transition")
    dataset.spatial_stride = 1
    dataset._item_references = cast("Any", (SimpleNamespace(case_index=0), SimpleNamespace(case_index=1)))
    monkeypatch.setattr(dataset, "_admit_spatial_view", lambda _payload: None)
    monkeypatch.setattr(dataset, "_materialize_shard_reference", lambda _reference, shard, **kwargs: {"shard": shard["shard"], **kwargs})
    # Each item is materialized from its own case's stored states, without reloading a shard.
    items = [cast("dict[str, Any]", item) for item in dataset.iter_scaling_items()]
    assert [(item["shard"], float(item["spatial_states"][-1])) for item in items] == [(0, 2.0), (1, 12.0)]
    assert loads == [0, 1]
    dataset.__setstate__(dataset.__getstate__())
    dataset._load_shard(0)
    assert loads == [0, 1, 0]


def test_case_selection_reconstructs_same_backend_before_closing_source() -> None:
    source = _SelectionDataset(
        "index.json",
        sampling=TransientSamplingSpec(mode="one_step_transition"),
        source_root="root",
        spatial_stride=1,
        sample_indices=(0, 1, 2),
    )
    binding = cast("Any", object())
    source.manifest_binding = binding

    selected = transient.select_transient_cases(source, ("a",))

    assert isinstance(selected, _SelectionDataset)
    assert selected.storage_backend == "pt_shards"
    assert selected.sample_indices == (2,)
    assert selected.manifest_binding is binding
    assert source.closed is True


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (2.0, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_spatial_stride_contract_rejects_invalid_values(value: object, error: type[Exception]) -> None:
    """Reject implicit coercion while keeping the stride surface exact."""
    with pytest.raises(error):
        training.grid_contract.validate_spatial_stride(value)


def test_boundary_preserving_spatial_view_resolves_canonical_grid() -> None:
    """Keep both physical boundaries when subsampling 251x401 by two."""
    assert training.grid_contract.resolve_spatial_view((251, 401), 1) == (251, 401)
    assert training.grid_contract.resolve_spatial_view((251, 401), 2) == (126, 201)
    representation = training.grid_contract.resolve_spatial_representation((251, 401), 2)
    assert representation.source_shape == (251, 401)
    assert representation.represented_shape == (126, 201)
    assert representation.y_indices == tuple(range(0, 251, 2))
    assert representation.x_indices == tuple(range(0, 401, 2))
    assert representation.y_indices[-1] == 250
    assert representation.x_indices[-1] == 400
    assert representation.as_dict()["index_identity_sha256"] == representation.index_identity_sha256
    assert len(representation.index_identity_sha256) == 64
    assert training.grid_contract.resolve_spatial_representation((251, 401), 1).index_identity_sha256 != representation.index_identity_sha256
    with pytest.raises(ValueError, match="does not tile the canonical 251 x 401 grid"):
        training.grid_contract.resolve_spatial_view((251, 401), 4)


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
        batches=None,
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
    materials: dict[str, str] = dict.fromkeys(("a", "b", "c"), "lentil")
    kinds = {"a": "train", "b": "validation", "c": "test"}
    dataset.manifest_binding = identity.ManifestBinding(
        dataset_id="fixed-primary",
        dataset_digest="1" * 64,
        channel_contract_digest="2" * 64,
        case_ids=("a", "b", "c"),
        physical_sources={case_id: (case_id, case_id) for case_id in ("a", "b", "c")},
        materials=materials,
        design_kinds=kinds,
        case_partition=partition.build_fixed_partition(
            ("a", "b", "c"), material_by_case=materials, design_kind_by_case=kinds, material_roles={"lentil": "id"}
        ),
        grid_axes_sha256="3" * 64,
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
            material_ood_dataset_ids=("external",),
            train_subset=None,
            tensorizer=TransientTensorizerSpec(
                input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("normalized_current_time")
            ),
            train_sampling=TransientSamplingSpec(mode="one_step_transition"),
            loader_settings=training.factory.LoaderSettings(),
            units_per_update=2,
            memory_batch=1,
            model_sha256="a" * 64,
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


def test_windows_cover_every_case_and_trajectories_stay_whole() -> None:
    """
    Place end-aligned B windows, consecutive rest windows, and one window for a short case.

    The end-aligned window that closes a trajectory overlaps its predecessor; a case
    shorter than the window forms one window over its whole trajectory, so every Train
    case keeps at least one item. Trajectory items keep every transition.
    """
    assert rollout_windows(168, rollout_length=32, window_stride=32, window_offset=0) == (
        (0, 32), (32, 32), (64, 32), (96, 32), (128, 32), (136, 32),
    )  # fmt: skip
    assert rollout_windows(64, rollout_length=32, window_stride=32, window_offset=0) == ((0, 32), (32, 32))
    assert rollout_windows(20, rollout_length=32, window_stride=32, window_offset=0) == ((0, 20),)
    assert consecutive_windows(70, window=32) == ((0, 32), (32, 32), (64, 6))
    assert consecutive_windows(20, window=32) == ((0, 20),)

    window_sampling = TransientSamplingSpec(mode="rollout_window", rollout_length=4, window_stride=4, window_offset=0)
    windows = _trajectory_dataset(window_sampling, lengths=(6, 3))
    assert [reference.sample_positions for reference in windows._build_item_references()] == [(0, 1, 2, 3), (2, 3, 4, 5), (6, 7, 8)]
    assert windows.item_case_indices() == (0, 0, 1)
    assert windows.item_transition_counts() == (4, 4, 3)

    # Cases stop at their own drying target, so whole trajectories of one selection may differ in length; a short one stays whole.
    trajectories = _trajectory_dataset(TransientSamplingSpec(mode="trajectory", rollout_length=4), lengths=(6, 3))
    assert [reference.sample_positions for reference in trajectories._build_item_references()] == [tuple(range(6)), (6, 7, 8)]
    assert trajectories.item_transition_counts() == (6, 3)

    # Evaluation sets no trainer window, so even a trajectory shorter than any window stays one item.
    assert EVALUATION_SAMPLING.as_dict() == {"mode": "trajectory"}
    with pytest.raises(ValueError, match="place no rollout windows"):
        EVALUATION_SAMPLING.windows(6)
    with pytest.raises(ValueError, match="accepts only rollout_length"):
        TransientSamplingSpec(mode="trajectory", rollout_length=4, window_stride=4)


def _batches(loader: Any) -> list[list[int]]:
    """Return one epoch of a loader's index batches."""
    assert loader.batch_sampler is not None
    return [list(batch) for batch in loader.batch_sampler]


def test_trajectory_batches_never_mix_lengths() -> None:
    """
    Batch whole trajectories that stop at different drying times.

    Evaluation batches keep their item order within one length; Train
    microbatches of the unit sampler hold one length, cover every item once per
    epoch and replay from the sampler seed. The sampler names the unit kind of
    the sampling mode.
    """
    settings = training.factory.LoaderSettings(batch_size=2)
    validation = _trajectory_dataset(EVALUATION_SAMPLING, lengths=(6, 5, 6, 6, 6))
    evaluation = training._loader(validation, settings, batches=None, loader_seed=0, worker_seed=0)
    assert _batches(evaluation) == [[0, 2], [1], [3, 4]]

    train_sampling = TransientSamplingSpec(mode="trajectory", rollout_length=2)
    lengths = (6, 5, 6, 5, 6)

    def train_epochs(seed: int) -> list[list[list[int]]]:
        dataset = _trajectory_dataset(train_sampling, lengths=lengths)
        sampler = training._unit_sampler(dataset, full_set=dataset, units_per_update=4, memory_batch=2, seed=seed)
        epochs = []
        for epoch in range(2):
            sampler.begin_epoch(epoch)
            epochs.append([list(batch) for batch in sampler])
            for group in sampler.groups():
                sampler.advance(len(group.items))
        return epochs

    epochs = train_epochs(5)
    for batches in epochs:
        assert sorted(item for batch in batches for item in batch) == list(range(5))
        assert all(len({lengths[item] for item in batch}) == 1 and len(batch) <= 2 for batch in batches)
    assert train_epochs(5) == epochs
    trajectories = _trajectory_dataset(train_sampling, lengths=lengths)
    sampler = training._unit_sampler(trajectories, full_set=trajectories, units_per_update=4, memory_batch=2, seed=0)
    assert sampler.unit_kind == "trajectory"
    windows = _trajectory_dataset(TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=2, window_offset=0), lengths=(6, 5))
    assert training._unit_sampler(windows, full_set=windows, units_per_update=4, memory_batch=2, seed=0).unit_kind == "window"


def test_the_transition_count_n_sums_unequal_trajectory_lengths_whatever_the_sampling_cuts() -> None:
    """
    N of B's budget K x N is the full Train role's regular transitions, sum L_i over unequal lengths.

    Whole trajectories and end-aligned windows (whose overlap repeats transitions) give the same N;
    a Train subset keeps the full role's units and N, so a learning-curve point trains as long as a full run.
    """
    lengths = (6, 5, 3)
    trajectories = _trajectory_dataset(TransientSamplingSpec(mode="trajectory", rollout_length=2), lengths=lengths)
    windows = _trajectory_dataset(TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=2, window_offset=0), lengths=lengths)
    subset = _trajectory_dataset(TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=2, window_offset=0), lengths=(6,))

    def counts(dataset: transient.TransientPhysicalDataset, full_set: transient.TransientPhysicalDataset) -> tuple[int, int]:
        sampler = training._unit_sampler(dataset, full_set=full_set, units_per_update=1, memory_batch=1, seed=0)
        return sampler.full_set_units, sampler.full_set_transitions

    # (6, 5, 3) cut into windows of 2 at stride 2 end-aligned: 3 + 3 + 2 windows hold 16 transitions, 2 of them repeated.
    assert counts(trajectories, trajectories) == (3, 14)
    assert counts(windows, windows) == (8, 14)
    assert counts(subset, windows) == (8, 14)
