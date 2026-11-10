"""Protect Train-only transient scaling through the public training-loader boundary.

Responsibilities:
  - Fit the scaler through ``create_transient_training_loaders`` on the compact published transient package
  - Compare the fitted state moments, increment RMS and counts with Train-only values recomputed from the published PT shards
  - Require an identity-bound scaler cache hit to restore the identical fitted evidence
  - Fit a ``train_subset`` run on its Train prefix only and record the subset in the split

Design principles:
  - The package's held-out cases move the state moments away from the Train-only values, so fitting on any other membership is visible
  - Expected statistics use the documented definition on test-read tensors and never call scaler code
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
    from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

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
    """Return population mean and standard deviation per channel over all states of the given cases."""
    pooled = torch.cat([sequence.to(torch.float64).transpose(0, 1).reshape(sequence.shape[1], -1) for sequence in states], dim=1)
    return pooled.mean(dim=1), pooled.std(dim=1, correction=0)


def _increment_rms(states: list[torch.Tensor]) -> torch.Tensor:
    """Return the per-channel RMS of float32 one-step increments over the given cases."""
    increments = [(sequence[1:] - sequence[:-1]).to(torch.float64).transpose(0, 1).reshape(sequence.shape[1], -1) for sequence in states]
    return torch.cat(increments, dim=1).square().mean(dim=1).sqrt()


def _create(package: dict[str, Any], sampling: Any, *, train_subset: int | None = None, **extra: Any) -> Any:
    """Create the public transient training loaders on the compact package."""
    return datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=package["train_dataset"],
        material_ood_dataset_ids=[],
        train_subset=train_subset,
        tensorizer=TransientTensorizerSpec(input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("none")),
        train_sampling=sampling,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        model_sha256="a" * 64,
        storage_root=str(package["root"]),
        **extra,
    )


def _close(loaders: Any) -> None:
    """Close every loader Dataset the training loaders own."""
    owned: tuple[Any, ...] = (loaders.train, loaders.evaluation, loaders.material_ood, loaders.parameter_ood, loaders.id_test)
    for loader in owned:
        if loader is not None:
            loader.dataset.close()


def _fit(package: dict[str, Any], sampling: Any, *, train_subset: int | None = None) -> tuple[TransientScalingArtifact, dict[str, Any], list[str]]:
    """Create the public loaders once and return the fitted scaler, split evidence and scaler cache outcomes."""
    cache_outcomes: list[str] = []

    def record(event: str, values: Any) -> None:
        if event == "scaler_cache":
            cache_outcomes.append(str(values["state"]))

    loaders = _create(package, sampling, train_subset=train_subset, startup_callback=record)
    try:
        return loaders.data_processor, loaders.split, cache_outcomes
    finally:
        _close(loaders)


@pytest.mark.parametrize("sampling", list(_SAMPLING.values()), ids=list(_SAMPLING))
def test_scaler_statistics_come_from_train_cases_only(transient_package: dict[str, Any], sampling: Any) -> None:
    """Fit state moments and increment RMS on Train cases only, and restore them identically from the cache.

    The held-out cases move the pooled state mean by more than 1 % in at least one channel, so a fit that admits
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

    torch.testing.assert_close(artifact.state_mean.to(torch.float64), train_mean, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    torch.testing.assert_close(artifact.state_std.to(torch.float64), train_std, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    torch.testing.assert_close(artifact.delta_rms.to(torch.float64), _increment_rms(train), rtol=1.0e-5, atol=1.0e-6)
    assert split["roles"]["train"]["case_ids"] == partition["train"]
    assert artifact.unique_train_state_count == sum(sequence.shape[0] for sequence in train)
    assert artifact.unique_transition_count == artifact.transition_count == sum(sequence.shape[0] - 1 for sequence in train)

    restored, _, second_cache = _fit(transient_package, sampling)

    assert first_cache == ["miss"]
    assert second_cache == ["hit"]
    assert restored.digest == artifact.digest
    for name in ("state_mean", "state_std", "delta_rms", "increment_scale"):
        assert torch.equal(getattr(restored, name), getattr(artifact, name))


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
    torch.testing.assert_close(artifact.state_mean.to(torch.float64), prefix_mean, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    torch.testing.assert_close(artifact.state_std.to(torch.float64), prefix_std, rtol=_STATE_TOLERANCE, atol=_STATE_TOLERANCE)
    assert artifact.unique_train_state_count == prefix[0].shape[0]
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
