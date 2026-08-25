from __future__ import annotations

# ruff: noqa: D100, D103, PLR2004, S101, SLF001
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch.utils.data import TensorDataset

from src import common, domain
from src.datasets.contracts.dataset_contracts_transient import TransientSamplingSpec
from src.datasets.runtime import dataset_runtime_transient as transient
from src.datasets.runtime import dataset_runtime_transient_training as training
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import (
    TransientTensorizerSpec,
)

if TYPE_CHECKING:
    from src.learning.transient.learning_transient_scaling import TransientScaleMode


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
        transform: object,
    ) -> None:
        self.index_path = index_path
        self.sampling = sampling
        self.source_root = source_root
        self.hdf5_cache_size = hdf5_cache_size
        self.spatial_stride = spatial_stride
        self.dataset_manifest_digest = "f" * 64
        self.sample_indices = sample_indices
        self.transform = transform
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
        transform=None,
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
        "schema_kind": "transient_drying_training_split",
        "tensorizer": tensorizer.selection_dict(),
        "sampling": TransientSamplingSpec(mode="one_step_transition").as_dict(),
        "ood_fraction": 1.0,
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
        "schema_kind": "transient_drying_training_split",
        "schema_version": 1,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "data_contract_digest": task.data_contract_digest,
        "tensorizer": tensorizer.selection_dict(),
        "sampling": sampling.as_dict(),
        "dataset_identity": {
            "train": identity,
            "ood": [{**identity, "dataset_id": "synthetic_ood"}],
        },
        "ood_fraction": 1.0,
        "split_seed": 9,
        "runtime_provenance": {
            "train": "pt_shards",
            "scaling_train_one_step": "pt_shards",
            "evaluation": "pt_shards",
            "id_test": "pt_shards",
            "ood": ["pt_shards"],
        },
        "roles": {
            "train": role("train_case", "train_item"),
            "scaling_train_one_step": role(
                "train_case",
                "scaling_item",
            ),
            "evaluation": role("eval_case", "eval_item"),
            "id_test": role("test_case", "test_item"),
            "ood": {"parts": [role("ood_case", "ood_item")]},
        },
    }

    admitted = training.admit_transient_training_split(
        split,
        tensorizer=tensorizer,
        sampling=sampling,
        ood_fraction=1.0,
        split_seed=9,
    )
    assert admitted == split

    stride_two = {
        **split,
        "schema_version": 2,
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
            ood_fraction=1.0,
            split_seed=9,
            spatial_stride=2,
        )["spatial_view"]["effective_nx"]
        == 201
    )
    with pytest.raises(ValueError, match="Legacy"):
        training.admit_transient_training_split(
            split,
            tensorizer=tensorizer,
            sampling=sampling,
            ood_fraction=1.0,
            split_seed=9,
            spatial_stride=2,
        )

    split["roles"]["train"]["membership_digest"] = "0" * 64
    with pytest.raises(ValueError, match="membership digest"):
        training.admit_transient_training_split(
            split,
            tensorizer=tensorizer,
            sampling=sampling,
            ood_fraction=1.0,
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


def test_ood_package_regime_uses_immutable_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regimes = {
        "parameter": "parameter_ood",
        "near": "near_family_ood",
        "far": "far_family_ood",
        "extreme": "extreme_family_ood",
        "id": "id",
    }
    calls: list[tuple[str, str | None]] = []

    def load_manifest(dataset_id: str, *, storage_root: str | None) -> dict[str, str]:
        calls.append((dataset_id, storage_root))
        return {
            "dataset_view": "transient_drying",
            "evaluation_regime": regimes[dataset_id],
        }

    monkeypatch.setattr(training.package_manifest, "load_package_manifest", load_manifest)

    for dataset_id, regime in tuple(regimes.items())[:-1]:
        assert training._ood_package_regime(dataset_id, storage_root="/storage") == regime
    with pytest.raises(ValueError, match="resolves to an ID package"):
        training._ood_package_regime("id", storage_root="/storage")

    assert calls == [(dataset_id, "/storage") for dataset_id in regimes]


def test_ood_package_regime_rejects_non_transient_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training.package_manifest,
        "load_package_manifest",
        lambda *_args, **_kwargs: {
            "dataset_view": "steady_flow",
            "evaluation_regime": "near_family_ood",
        },
    )

    with pytest.raises(ValueError, match="incompatible view"):
        training._ood_package_regime("steady-ood", storage_root=None)


def test_training_loader_dispatches_ordered_manifest_ood_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ood_regimes = {
        "ood_parameter": "parameter_ood",
        "ood_near": "near_family_ood",
        "ood_far": "far_family_ood",
        "ood_extreme": "extreme_family_ood",
    }
    sampling = TransientSamplingSpec(mode="one_step_transition")

    def runtime_dataset(dataset_id: str) -> transient.TransientPhysicalDataset:
        dataset = object.__new__(transient.TransientPhysicalDataset)
        dataset.sample_indices = (0,)
        dataset.sampling = sampling
        dataset._item_references = (transient._ItemReference(case_index=0, sample_positions=(0,)),)
        dataset.storage_backend = "pt_shards"
        dataset.spatial_stride = 1
        dataset.dataset_manifest_digest = "f" * 64
        dataset._spatial_view = ((251, 401), (251, 401))
        dataset.payload = {
            "dataset_id": dataset_id,
            "contract_digest": "a" * 64,
            "index_digest": dataset_id.ljust(64, "0"),
            "configured_regular_horizon": {"value": 32.0, "unit": "h"},
            "cases": [{"package_case_id": f"{dataset_id}_case"}],
            "samples": [
                {
                    "case_index": 0,
                    "sample_id": f"{dataset_id}_case__step_0000",
                }
            ],
        }
        return dataset

    datasets = {dataset_id: runtime_dataset(dataset_id) for dataset_id in ("train_id", *ood_regimes)}
    requests: list[tuple[str, str]] = []

    def create_dataset(
        request: training.factory.DatasetRequest,
        **_kwargs: object,
    ) -> transient.TransientPhysicalDataset:
        dataset_id = request.dataset_id
        requests.append((dataset_id, request.evaluation_regime))
        return datasets[dataset_id]

    monkeypatch.setattr(
        training.package_manifest,
        "load_package_manifest",
        lambda dataset_id, **_kwargs: {
            "dataset_view": "transient_drying",
            "evaluation_regime": ood_regimes[dataset_id],
        },
    )
    monkeypatch.setattr(training.factory, "create_dataset", create_dataset)
    monkeypatch.setattr(training, "_select_cases", lambda dataset, _case_ids: dataset)
    monkeypatch.setattr(training, "_fit_or_load_scaling_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(training, "_loader", lambda dataset, *_args, **_kwargs: dataset)

    loaders = training.create_transient_training_loaders(
        train_dataset_id="train_id",
        ood_dataset_ids=tuple(ood_regimes),
        tensorizer=TransientTensorizerSpec(
            input_profile="canonical_physics_complete_v1",
            temporal_conditioning=TemporalConditioningSpec("none"),
        ),
        train_sampling=sampling,
        loader_settings=training.factory.LoaderSettings(batch_size=1),
    )

    ood_requests = [request for request in requests if request[0] in ood_regimes]
    assert ood_requests == list(ood_regimes.items())
    assert [identity["dataset_id"] for identity in loaders.dataset_identity["ood"]] == list(ood_regimes)
    assert [part["case_ids"] for part in loaders.split["roles"]["ood"]["parts"]] == [[f"{dataset_id}_case"] for dataset_id in ood_regimes]


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
    legacy_identity = {"dataset_id": "synthetic", "index_digest": "2" * 64}
    base_identity = {
        **legacy_identity,
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
            legacy_dataset_identity=legacy_identity,
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
            legacy_dataset_identity=legacy_identity,
            train_membership_digest=membership,
            scale_mode="state_std",
            spatial_view=view_two,
            horizon=10.0,
            allow_legacy_stride_one=True,
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
