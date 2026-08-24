# ruff: noqa: S101
"""Protect transient tensorization and Train-only scaling contracts."""

from __future__ import annotations

import copy
import hashlib
from typing import TYPE_CHECKING, Any

import pytest
import torch

from src.learning import learning_temporal
from src.learning.transient.learning_transient_contracts import (
    TransientTensorizerSpec,
)
from src.learning.transient.learning_transient_scaling import (
    TransientScalingArtifact,
    fit_transient_scaling,
)
from src.learning.transient.learning_transient_tensorizer import (
    TransientTensorizer,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_EXPECTED_UNIQUE_STATES = 3
_EXPECTED_TRANSITIONS = 2
_STREAMING_TRANSITIONS = 20
_STREAMING_STATES = _STREAMING_TRANSITIONS + 1


def _digest(label: str) -> str:
    """Return one deterministic test-owned SHA-256 digest."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _item(
    index: int = 0,
    *,
    startup: float = 0.0,
    regular_omega: tuple[float, float] = (0.1, 0.2),
) -> dict[str, Any]:
    """Return one tiny raw physical one-step Train item."""
    state = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) + index
    boundary = torch.tensor(
        [
            300.0,
            301.0,
            regular_omega[0],
            regular_omega[1],
            295.0,
            0.5,
            302.0,
            0.3,
            startup,
        ],
        dtype=torch.float32,
    )
    return {
        "state": state,
        "target": torch.ones_like(state),
        "static": torch.arange(28, dtype=torch.float32).reshape(7, 2, 2),
        "boundary": boundary,
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


def _spec(
    profile: str = "canonical_physics_complete_v1",
    temporal: learning_temporal.TemporalConditioningKind = "normalized_current_time",
) -> TransientTensorizerSpec:
    """Return one authoritative tensorizer selection."""
    return TransientTensorizerSpec(
        input_profile=profile,
        temporal_conditioning=learning_temporal.TemporalConditioningSpec(temporal),
    )


def _artifact(
    *,
    startup: float = 1.0,
) -> TransientScalingArtifact:
    """Fit one tiny valid artifact."""
    return fit_transient_scaling(
        [_item(0, startup=startup), _item(1, startup=startup)],
        tensorizer=_spec(),
        dataset_identity={
            "dataset_id": "synthetic",
            "index_digest": _digest("index"),
        },
        train_membership_digest=_digest("train-membership"),
        horizon=10.0,
    )


def _batch(
    rollout_length: int = 1,
    *,
    startup: float = 0.0,
) -> dict[str, Any]:
    """Collate one tiny physical one-step or rollout batch."""
    item = _item(0, startup=startup)
    target = item["target"].unsqueeze(0)
    boundary = item["boundary"].unsqueeze(0)
    if rollout_length > 1:
        target = target.repeat(1, rollout_length, 1, 1, 1)
        boundary = boundary.repeat(1, rollout_length, 1)
    return {
        "state": item["state"].unsqueeze(0),
        "target": target,
        "static": item["static"].unsqueeze(0),
        "boundary": boundary,
        "scalars": item["scalars"].unsqueeze(0),
        "time": {
            "t_n": torch.arange(
                rollout_length,
                dtype=torch.float32,
            ).reshape(1, rollout_length),
            "t_n_plus_1": torch.arange(
                1,
                rollout_length + 1,
                dtype=torch.float32,
            ).reshape(1, rollout_length),
            "dt": torch.ones(1, rollout_length),
        },
    }


@pytest.mark.parametrize(
    ("profile", "temporal", "expected_channels"),
    [
        ("canonical_physics_complete_v1", "none", 28),
        ("canonical_physics_complete_v1", "normalized_current_time", 29),
    ],
)
def test_profiles_derive_exact_channels(
    profile: str,
    temporal: learning_temporal.TemporalConditioningKind,
    expected_channels: int,
) -> None:
    """Keep optional time outside the exact task-owned profile."""
    spec = _spec(profile, temporal)
    assert spec.in_channels == expected_channels
    assert spec.positional_embedding is None
    assert ("normalized_current_time" in spec.model_channel_names) == (temporal == "normalized_current_time")


def test_tensorizer_normalizes_one_step_and_teacher_forced_rollout() -> None:
    """Build BCHW and BLCHW inputs from exact reference current states."""
    artifact = _artifact()
    tensorizer = TransientTensorizer(_spec(), artifact)
    one_step = tensorizer.tensorize(_batch())
    rollout = tensorizer.tensorize(_batch(2))

    assert one_step.step_input.shape == (1, 29, 2, 2)
    assert one_step.scaled_target.shape == (1, 1, 4, 2, 2)
    assert rollout.sequence_input.shape == (1, 2, 29, 2, 2)
    expected_second = artifact.encode_state(rollout.state + rollout.target[:, 0])
    assert torch.allclose(
        rollout.sequence_input[:, 1, 0:4],
        expected_second,
    )


def test_endpoints_remain_independent_and_absent_startup_is_neutral() -> None:
    """Preserve both endpoints while masking absent startup placeholders."""
    tensorizer = TransientTensorizer(_spec(), _artifact())
    base = _batch()
    endpoint_n = copy.deepcopy(base)
    endpoint_next = copy.deepcopy(base)
    endpoint_n["boundary"][0, 0] += 2.0
    endpoint_next["boundary"][0, 1] += 3.0

    base_input = tensorizer.tensorize(base).step_input
    n_input = tensorizer.tensorize(endpoint_n).step_input
    next_input = tensorizer.tensorize(endpoint_next).step_input
    assert not torch.equal(base_input[:, 11], n_input[:, 11])
    assert torch.equal(base_input[:, 12], n_input[:, 12])
    assert not torch.equal(base_input[:, 12], next_input[:, 12])
    assert torch.equal(base_input[:, 11], next_input[:, 11])

    altered = _batch(startup=0.0)
    altered["boundary"][0, 5:8] = 999.0
    altered_input = tensorizer.tensorize(altered).step_input
    assert torch.equal(base_input[:, 16:19], altered_input[:, 16:19])


def test_scaling_deduplicates_states_and_ignores_external_eval_values() -> None:
    """Fit unique Train states without accepting evaluation data as input."""
    train_items = [_item(0, startup=0.0), _item(1, startup=0.0)]
    evaluation = _item(2)
    first = fit_transient_scaling(
        train_items,
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("membership"),
        horizon=10.0,
    )
    evaluation["state"].fill_(1.0e9)
    second = fit_transient_scaling(
        train_items,
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("membership"),
        horizon=10.0,
    )
    assert first.unique_train_state_count == _EXPECTED_UNIQUE_STATES
    assert first.transition_count == _EXPECTED_TRANSITIONS
    assert first.digest == second.digest


def test_scaling_admits_exact_float32_delta_when_addition_does_not_round_trip() -> None:
    """Prefer direct endpoints while validating the exact stored delta."""
    first = _item(0)
    second = _item(1)
    first_state = torch.full_like(first["state"], 0.3)
    second_state = torch.full_like(second["state"], 0.1)
    terminal_state = torch.full_like(second["state"], 0.4)
    first["state"] = first_state
    first["target"] = second_state - first_state
    second["state"] = second_state
    second["target"] = terminal_state - second_state
    assert not torch.equal(first_state + first["target"], second_state)

    artifact = fit_transient_scaling(
        [first, second],
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("round-trip"),
        horizon=10.0,
    )

    terminal_reconstructed = second_state + second["target"]
    expected_matrix = torch.stack((first_state, second_state, terminal_reconstructed)).permute(1, 0, 2, 3).reshape(4, -1)
    expected_std = expected_matrix.to(dtype=torch.float64).std(dim=1, unbiased=False).to(dtype=torch.float32)
    assert torch.equal(artifact.state_std, expected_std)
    assert artifact.unique_train_state_count == _EXPECTED_UNIQUE_STATES


def test_scaling_rejects_noncanonical_case_time_order() -> None:
    """Require the immutable package index's grouped consecutive order."""
    first = _item(0)
    second = _item(1)
    with pytest.raises(ValueError, match="canonical zero-time"):
        fit_transient_scaling(
            [second, first],
            tensorizer=_spec(),
            dataset_identity={"dataset_id": "synthetic"},
            train_membership_digest=_digest("reversed-order"),
            horizon=10.0,
        )


def test_scaling_streams_one_shot_items_against_float64_population_reference() -> None:
    """Match stable analytical moments without materializing production tensors."""
    items = []
    for index in range(_STREAMING_TRANSITIONS):
        item = _item(
            index,
            regular_omega=(0.1 + index * 0.01, 0.2 + index * 0.02),
        )
        item["static"] = item["static"] + index * 0.5
        item["scalars"] = item["scalars"] + index * 0.25
        items.append(item)

    artifact = fit_transient_scaling(
        (item for item in items),
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("streaming-reference"),
        horizon=25.0,
    )

    def population(values: tuple[torch.Tensor, ...], channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = torch.stack(values).permute(1, 0, 2, 3).reshape(channels, -1).to(dtype=torch.float64)
        return matrix.mean(dim=1).to(dtype=torch.float32), matrix.std(dim=1, unbiased=False).to(dtype=torch.float32)

    state_values = [item["state"] for item in items]
    state_values.append(items[-1]["state"] + items[-1]["target"])
    state_mean, state_std = population(tuple(state_values), 4)
    static_mean, static_std = population(tuple(item["static"] for item in items), 7)
    scalar_matrix = torch.stack(tuple(item["scalars"] for item in items)).transpose(0, 1).to(dtype=torch.float64)
    omega_matrix = torch.cat(tuple(item["boundary"][2:4] for item in items)).reshape(1, -1).to(dtype=torch.float64)

    def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)

    assert_close(artifact.state_mean, state_mean)
    assert_close(artifact.state_std, state_std)
    assert_close(artifact.delta_rms, torch.ones(4))
    assert_close(artifact.static_mean, static_mean)
    assert_close(artifact.static_std, static_std.clamp_min(artifact.numerical_floor))
    assert_close(
        artifact.scalar_mean,
        scalar_matrix.mean(dim=1).to(dtype=torch.float32),
    )
    assert_close(
        artifact.scalar_std,
        scalar_matrix.std(dim=1, unbiased=False).to(dtype=torch.float32).clamp_min(artifact.numerical_floor),
    )
    assert_close(
        artifact.omega_boundary_mean,
        omega_matrix.mean().to(dtype=torch.float32),
    )
    assert_close(
        artifact.omega_boundary_std,
        omega_matrix.std(unbiased=False).to(dtype=torch.float32).clamp_min(artifact.numerical_floor),
    )
    assert artifact.unique_train_state_count == _STREAMING_STATES
    assert artifact.unique_transition_count == artifact.transition_count == _STREAMING_TRANSITIONS


def test_scaling_isolates_reused_one_shot_buffers() -> None:
    """Snapshot yielded tensors before a one-shot iterator reuses its buffers."""
    state_buffer = torch.empty_like(_item()["state"])
    target_buffer = torch.empty_like(_item()["target"])

    def reused_items() -> Iterator[dict[str, Any]]:
        first = _item(0)
        state_buffer.zero_()
        target_buffer.fill_(1.0)
        first["state"] = state_buffer
        first["target"] = target_buffer
        yield first

        second = _item(1)
        state_buffer.fill_(1.0)
        target_buffer.fill_(2.0)
        second["state"] = state_buffer
        second["target"] = target_buffer
        yield second

    artifact = fit_transient_scaling(
        reused_items(),
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("reused-buffers"),
        horizon=10.0,
    )

    torch.testing.assert_close(
        artifact.state_mean,
        torch.full((4,), 4.0 / 3.0),
    )
    torch.testing.assert_close(
        artifact.delta_rms,
        torch.sqrt(torch.tensor(2.5)).repeat(4),
    )
    assert artifact.unique_train_state_count == _EXPECTED_UNIQUE_STATES
    assert artifact.unique_transition_count == artifact.transition_count == _EXPECTED_TRANSITIONS


def test_scaling_rejects_direct_state_that_disagrees_with_exact_delta() -> None:
    """Keep direct endpoint admission bound to the stored float32 delta."""
    first = _item(0)
    second = _item(1)
    first["state"].fill_(0.3)
    second["state"].fill_(0.1)
    first["target"] = second["state"] - first["state"]
    second["state"].fill_(0.2)

    with pytest.raises(ValueError, match="exact transition target"):
        fit_transient_scaling(
            [first, second],
            tensorizer=_spec(),
            dataset_identity={"dataset_id": "synthetic"},
            train_membership_digest=_digest("conflicting-direct-endpoint"),
            horizon=10.0,
        )


def test_scaling_keeps_direct_and_transition_duplicates_exact() -> None:
    """Reject duplicate direct states and targets even when additions coincide."""
    original = _item(0)
    changed_state = copy.deepcopy(original)
    changed_state["state"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="direct transient state"):
        fit_transient_scaling(
            [original, changed_state],
            tensorizer=_spec(),
            dataset_identity={"dataset_id": "synthetic"},
            train_membership_digest=_digest("conflicting-direct-duplicate"),
            horizon=10.0,
        )

    changed_target = copy.deepcopy(original)
    changed_target["target"].zero_()
    changed_target["target"][0, 0, 1] = torch.nextafter(
        torch.tensor(0.0),
        torch.tensor(float("inf")),
    )
    original["target"].zero_()
    assert torch.equal(original["state"] + original["target"], original["state"] + changed_target["target"])
    with pytest.raises(ValueError, match="transition evidence"):
        fit_transient_scaling(
            [original, changed_target],
            tensorizer=_spec(),
            dataset_identity={"dataset_id": "synthetic"},
            train_membership_digest=_digest("conflicting-transition-duplicate"),
            horizon=10.0,
        )


def test_scaling_persists_raw_zero_statistics_and_selects_delta_rms() -> None:
    """Keep exact-zero statistics while using the persisted floor operationally."""
    zero_item = _item(0)
    zero_item["state"].zero_()
    zero_item["target"].zero_()
    state_std = fit_transient_scaling(
        [zero_item],
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("zero-state"),
        horizon=10.0,
    )
    delta_rms = fit_transient_scaling(
        [zero_item],
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("zero-delta"),
        horizon=10.0,
        scale_mode="delta_rms",
    )

    assert torch.equal(state_std.state_std, torch.zeros(4))
    assert torch.equal(state_std.delta_rms, torch.zeros(4))
    assert torch.equal(state_std.increment_scale, torch.full((4,), state_std.numerical_floor))
    assert delta_rms.scale_mode == "delta_rms"
    assert torch.equal(delta_rms.increment_scale, torch.full((4,), delta_rms.numerical_floor))


def test_delta_rms_mode_uses_raw_increment_rms_for_increment_scale() -> None:
    """Select raw delta RMS rather than the primary absolute-state standard deviation."""
    artifact = fit_transient_scaling(
        [_item(0), _item(1)],
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("delta-rms"),
        horizon=10.0,
        scale_mode="delta_rms",
    )

    assert torch.equal(artifact.increment_scale, artifact.delta_rms.clamp_min(artifact.numerical_floor))


def test_regular_omega_fits_when_every_startup_flag_is_absent() -> None:
    """Fit shared omega statistics from regular endpoints unconditionally."""
    artifact = fit_transient_scaling(
        [
            _item(0, startup=0.0, regular_omega=(0.1, 0.3)),
            _item(1, startup=0.0, regular_omega=(0.5, 0.7)),
        ],
        tensorizer=_spec(),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest=_digest("membership"),
        horizon=10.0,
    )
    assert artifact.omega_boundary_mean.item() == pytest.approx(0.4)
    assert artifact.omega_boundary_std.item() > 0.0


def test_scaling_serialization_is_strict_and_device_independent() -> None:
    """Reject tensorizer tampering and keep runtime transfer hash-stable."""
    artifact = _artifact()
    state = artifact.state_dict()
    assert state["schema_version"] == 1
    restored = TransientScalingArtifact.from_state_dict(state)
    on_cpu = restored.to(torch.device("cpu"))

    assert restored.digest == artifact.digest == on_cpu.digest
    assert restored.spatial_shape == artifact.spatial_shape
    assert all(
        getattr(on_cpu, name).device == torch.device("cpu")
        for name in (
            "state_mean",
            "state_std",
            "delta_rms",
            "static_mean",
            "static_std",
            "scalar_mean",
            "scalar_std",
        )
    )

    changed_shape = copy.deepcopy(state)
    changed_shape["spatial_shape"] = [0, 2]
    with pytest.raises(ValueError, match="spatial_shape"):
        TransientScalingArtifact.from_state_dict(changed_shape)

    changed_channels = copy.deepcopy(state)
    changed_channels["tensorizer"]["model_channel_names"][0] = "wrong"
    with pytest.raises(ValueError, match="tensorizer identity"):
        TransientScalingArtifact.from_state_dict(changed_channels)

    changed_embedding = copy.deepcopy(state)
    changed_embedding["tensorizer"]["positional_embedding"] = "grid"
    with pytest.raises(ValueError, match="tensorizer identity"):
        TransientScalingArtifact.from_state_dict(changed_embedding)


def test_zero_delta_reconstruction_is_exact_and_differentiable() -> None:
    """Preserve exact zeros and gradient flow through physical reconstruction."""
    artifact = _artifact()
    tensorizer = TransientTensorizer(_spec(), artifact)
    current = _batch()["state"].clone().requires_grad_(True)
    scaled_delta = torch.zeros_like(current, requires_grad=True)

    physical = tensorizer.reconstruct_next_state(current, scaled_delta)
    assert torch.equal(physical, current)
    physical.sum().backward()
    assert current.grad is not None
    assert scaled_delta.grad is not None


def test_invalid_time_flag_shape_and_finiteness_fail_closed() -> None:
    """Reject scientific and tensor-contract drift before model execution."""
    tensorizer = TransientTensorizer(_spec(), _artifact())

    wrong_dt = _batch()
    wrong_dt["time"]["dt"][0, 0] = 2.0
    with pytest.raises(ValueError, match="1h"):
        tensorizer.tensorize(wrong_dt)

    invalid_flag = _batch()
    invalid_flag["boundary"][0, 8] = 0.5
    with pytest.raises(ValueError, match="binary"):
        tensorizer.tensorize(invalid_flag)

    nonfinite = _batch()
    nonfinite["state"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        tensorizer.tensorize(nonfinite)

    with pytest.raises(ValueError, match="current_state"):
        tensorizer.assemble_step(
            current_state=torch.zeros(1, 4, 2),
            static=torch.zeros(1, 7, 2, 2),
            boundary=torch.zeros(1, 9),
            scalars=torch.zeros(1, 8),
            t_n=torch.zeros(1),
        )
