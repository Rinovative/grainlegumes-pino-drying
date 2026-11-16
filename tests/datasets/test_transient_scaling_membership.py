"""Protect Train-only transient scaling through the public training-loader boundary.

Responsibilities:
  - Fit the scaler through ``create_transient_training_loaders`` on the compact published transient package
  - Compare the fitted case-equal state moments and case count with Train-only values recomputed from the published PT shards
  - Require an identity-bound scaler cache hit to restore the identical fitted evidence
  - Fit a ``train_subset`` run on its Train prefix only and record the subset in the split

Design principles:
  - The package's held-out cases move the state moments away from the Train-only values, so fitting on any other membership is visible
  - Expected statistics use the documented case-equal definition on test-read tensors and never call scaler code
  - Both Train sampling branches (direct one-step items and a separately selected one-step fitting Dataset) must give the same Train-only fit

This module does NOT:
  - Train models, judge physical accuracy, or cover scaler cache identity drift and restored-scaler handoff
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch

from src import datasets
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling

_STATE_TOLERANCE = 1.0e-4
_SAMPLING = {
    "one_step": datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
    "rollout": datasets.contracts.transient.TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0),
}


def _published_states(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Read every case's regular state sequence [time, channel, y, x] from the published PT shards."""
    shards = datasets.packages.transient_shards
    root, dataset_id = package["root"], package["train_dataset"]
    receipt = shards.load_transient_shard_receipt(dataset_id, storage_root=root, validation_depth="evidence")
    states: dict[str, torch.Tensor] = {}
    for shard_index in range(receipt["shard_count"]):
        payload = shards.load_transient_shard_payload(dataset_id, shard_index, storage_root=root, receipt=receipt)
        for case in payload["cases"]:
            states[str(case["case_record"]["package_case_id"])] = case["states"].clone()
    return states


def _channel_moments(states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-channel mean and population std in which every case weighs one and its states weigh equally."""
    blocks = [sequence.to(torch.float64).transpose(0, 1).reshape(sequence.shape[1], -1) for sequence in states]
    mean = torch.stack([block.mean(dim=1) for block in blocks]).mean(dim=0)
    variance = torch.stack([(block - mean[:, None]).square().mean(dim=1) for block in blocks]).mean(dim=0)
    return mean, variance.sqrt()


def _create(package: dict[str, Any], sampling: Any, *, train_subset: int | None = None, **extra: Any) -> Any:
    """Create the public transient training loaders on the compact package."""
    return datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=package["train_dataset"],
        material_ood_dataset_ids=[],
        train_subset=train_subset,
        tensorizer=TransientTensorizerSpec(input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("none")),
        train_sampling=sampling,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        units_per_update=2,
        memory_batch=1,
        model_sha256="a" * 64,
        storage_root=str(package["root"]),
        **extra,
    )


def _close(loaders: Any) -> None:
    """Close every loader Dataset the training loaders own."""
    for loader in (loaders.train, loaders.evaluation):
        loader.dataset.close()


def _fit(package: dict[str, Any], sampling: Any, *, train_subset: int | None = None) -> tuple[TrainScaling, dict[str, Any], list[str]]:
    """Create the public loaders once and return the fitted scaler, split evidence and scaler cache outcomes."""
    cache_outcomes: list[str] = []

    def record(event: str, values: Any) -> None:
        if event == "scaler_cache":
            cache_outcomes.append(str(values["state"]))

    loaders = _create(package, sampling, train_subset=train_subset, startup_callback=record)
    try:
        return loaders.scaling, loaders.split, cache_outcomes
    finally:
        _close(loaders)


@pytest.mark.parametrize("sampling", list(_SAMPLING.values()), ids=list(_SAMPLING))
def test_scaler_statistics_come_from_train_cases_only(transient_package: dict[str, Any], sampling: Any) -> None:
    """Fit case-equal state moments on Train cases only, and restore them identically from the cache.

    The held-out cases move the case-equal state mean by more than 1 % in at least one channel, so a fit that admits
    validation or ID-test evidence changes the statistics beyond tolerance.
    """
    partition = transient_package["case_partition"]
    by_case = _published_states(transient_package)
    train = [by_case[case_id] for case_id in partition["train"]]
    held_out = [by_case[case_id] for role in ("validation", "id_test") for case_id in partition[role]]
    assert train
    assert held_out

    train_mean, train_std = _channel_moments(train)
    all_mean, _ = _channel_moments(list(by_case.values()))
    shifted = (all_mean - train_mean).abs() > 0.01 * train_mean.abs()
    assert bool(shifted.any()), "fixture premise: Train-only moments differ from the whole Dataset"

    artifact, split, first_cache = _fit(transient_package, sampling)

    torch.testing.assert_close(artifact.group("state").mean.to(torch.float64), train_mean, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    torch.testing.assert_close(artifact.group("state").std.to(torch.float64), train_std, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    assert split["roles"]["train"]["case_ids"] == partition["train"]
    assert artifact.train_case_count == len(train)

    restored, _, second_cache = _fit(transient_package, sampling)

    assert first_cache == ["miss"]
    assert second_cache == ["hit"]
    assert restored.digest == artifact.digest
    for name, statistics in artifact.groups.items():
        assert torch.equal(restored.group(name).mean, statistics.mean)
        assert torch.equal(restored.group(name).std, statistics.std)


def test_a_train_subset_fits_its_scaler_on_the_train_prefix_only(transient_package: dict[str, Any]) -> None:
    """``train_subset`` 1 trains and fits on the first Train case of the ID material and records the subset in the split."""
    partition = transient_package["case_partition"]
    by_case = _published_states(transient_package)
    assert len(partition["train"]) == 2
    prefix = [by_case[partition["train"][0]]]
    whole_mean, _ = _channel_moments([by_case[case_id] for case_id in partition["train"]])
    prefix_mean, prefix_std = _channel_moments(prefix)
    assert bool(((whole_mean - prefix_mean).abs() > 0.01 * prefix_mean.abs()).any()), "fixture premise: the prefix differs from all Train"

    artifact, split, _ = _fit(transient_package, _SAMPLING["one_step"], train_subset=1)

    assert split["train_subset"] == 1
    assert split["roles"]["train"]["case_ids"] == partition["train"][:1]
    torch.testing.assert_close(artifact.group("state").mean.to(torch.float64), prefix_mean, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    torch.testing.assert_close(artifact.group("state").std.to(torch.float64), prefix_std, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    assert artifact.train_case_count == 1
    with pytest.raises(ValueError, match=r"data.train_subset 4 exceeds the Train cases of \['lentil'\]"):
        _fit(transient_package, _SAMPLING["one_step"], train_subset=4)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [("dataset_identity", {"dataset_id": "foreign"}), ("train_membership_digest", "b" * 64)],
)
def test_restored_scaling_state_with_a_foreign_identity_is_refused(transient_package: dict[str, Any], field: str, foreign: Any) -> None:
    """Refuse a restored artifact whose envelope identity matches while its fitted state repeats another Dataset or membership."""
    sampling = _SAMPLING["one_step"]
    first = _create(transient_package, sampling)
    try:
        envelope = copy.deepcopy(first.scaling_artifact)
    finally:
        _close(first)
    envelope["state"][field] = foreign

    with pytest.raises(ValueError, match=field):
        _create(transient_package, sampling, scaling_artifact=envelope)
