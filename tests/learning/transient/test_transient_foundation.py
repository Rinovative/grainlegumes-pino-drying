"""Protect transient tensorization and the case-equal transient Train scaling fit.

Responsibilities:
  - Check tensorization, the case-equal streamed fit on stored states, serialization, and
    reconstruction

Design principles:
  - Duplicate transitions, case order and zero statistics preserve authoritative float32 evidence

This module does NOT:
  - Fit scales from external evaluation data or train transient models
"""

from __future__ import annotations

import copy
import hashlib
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support.scaling import spatial_view

from src import datasets
from src.learning import learning_temporal
from src.learning.transient.learning_transient_contracts import (
    TransientTensorizerSpec,
)
from src.learning.transient.learning_transient_scaling import fit_transient_scaling
from src.learning.transient.learning_transient_tensorizer import (
    TransientTensorizer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling

_F_SURF_INDEX = 2
_W_SURF_INDEX = 2
_W_INT_INDEX = 3


def _digest(label: str) -> str:
    """Return one deterministic test-owned SHA-256 digest."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _item(
    index: int = 0,
    *,
    startup: float = 0.0,
    regular_omega: tuple[float, float] = (0.1, 0.2),
    f_surf: float = 0.25,
) -> dict[str, Any]:
    """Return one tiny raw physical one-step Train item whose stored next state is the next item's state."""
    state = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) + index
    scalars = torch.arange(8, dtype=torch.float32)
    scalars[_F_SURF_INDEX] = f_surf
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
        "next_state": state + 1.0,
        "static": torch.arange(28, dtype=torch.float32).reshape(7, 2, 2),
        "boundary": boundary,
        "scalars": scalars,
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
    profile: str = "canonical_physics_complete",
    temporal: learning_temporal.TemporalConditioningKind = "normalized_current_time",
) -> TransientTensorizerSpec:
    """Return one authoritative tensorizer selection."""
    return TransientTensorizerSpec(
        input_profile=profile,
        temporal_conditioning=learning_temporal.TemporalConditioningSpec(temporal),
    )


def _fit(items: Iterable[dict[str, Any]], label: str = "membership", *, horizon: float = 10.0) -> TrainScaling:
    """Fit the transient scaling of the default tensorizer on a 2 x 2 grid."""
    return fit_transient_scaling(
        items,
        preprocessing=datasets.preprocessing.normalization.preprocessing_spec(
            "transient_drying", spatial_view=spatial_view((2, 2)), tensorizer=_spec().as_dict()
        ),
        dataset_identity={"dataset_id": "synthetic", "index_digest": _digest("index")},
        train_membership_digest=_digest(label),
        horizon=horizon,
    )


def _artifact(
    *,
    startup: float = 1.0,
) -> TrainScaling:
    """Fit one tiny valid scaling."""
    return _fit([_item(0, startup=startup), _item(1, startup=startup)], "train-membership")


def _batch(
    rollout_length: int = 1,
    *,
    startup: float = 0.0,
) -> dict[str, Any]:
    """Collate one tiny physical one-step or rollout batch."""
    item = _item(0, startup=startup)
    next_state = item["next_state"].unsqueeze(0)
    boundary = item["boundary"].unsqueeze(0)
    if rollout_length > 1:
        next_state = torch.stack([item["state"] + step for step in range(1, rollout_length + 1)]).unsqueeze(0)
        boundary = boundary.repeat(1, rollout_length, 1)
    return {
        "state": item["state"].unsqueeze(0),
        "next_state": next_state,
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
        ("canonical_physics_complete", "none", 28),
        ("canonical_physics_complete", "normalized_current_time", 29),
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
    """Build BCHW and BLCHW inputs from the stored current states."""
    artifact = _artifact()
    tensorizer = TransientTensorizer(_spec(), artifact)
    one_step = tensorizer.tensorize(_batch())
    rollout = tensorizer.tensorize(_batch(2))

    assert one_step.step_input.shape == (1, 29, 2, 2)
    assert one_step.next_state.shape == (1, 1, 4, 2, 2)
    assert rollout.sequence_input.shape == (1, 2, 29, 2, 2)
    assert torch.equal(rollout.sequence_input[:, 1, 0:4], artifact.encode_state(rollout.next_state[:, 0]))


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


def _case(name: str, transitions: int, *, offset: float, f_surf: float) -> list[dict[str, Any]]:
    """Return one canonical case of consecutive items whose static fields and scalars repeat within the case."""
    items = [_item(index, regular_omega=(0.1 + offset, 0.3 + offset), f_surf=f_surf) for index in range(transitions)]
    for item in items:
        item["state"] = item["state"] + offset
        item["next_state"] = item["next_state"] + offset
        item["static"] = item["static"] + offset
        item["scalars"] = item["scalars"] + offset
        item["scalars"][_F_SURF_INDEX] = f_surf
        item["metadata"]["simulation_case_id"] = name
    return items


def _case_equal(values_by_case: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the direct float64 mean and population std of ``[channel, value]`` blocks in which every case weighs one."""
    weights = torch.cat(
        [torch.full((block.shape[1],), 1.0 / (len(values_by_case) * block.shape[1]), dtype=torch.float64) for block in values_by_case]
    )
    matrix = torch.cat([block.to(torch.float64) for block in values_by_case], dim=1)
    mean = (matrix * weights).sum(dim=1)
    variance = ((matrix - mean[:, None]).square() * weights).sum(dim=1)
    return mean, variance.sqrt()


def _grain(state: torch.Tensor, f_surf: float) -> torch.Tensor:
    """Return the direct float64 grain moisture of one [4,Y,X] state."""
    fraction = float(torch.tensor(f_surf, dtype=torch.float32))
    return fraction * state[_W_SURF_INDEX].to(torch.float64) + (1.0 - fraction) * state[_W_INT_INDEX].to(torch.float64)


def test_streamed_fit_equals_the_direct_case_equal_computation() -> None:
    """Every group's streamed moments equal the direct computation in which every Train case weighs one."""
    cases = [_case("a", 1, offset=0.0, f_surf=0.25), _case("b", 2, offset=3.0, f_surf=0.6), _case("c", 4, offset=-1.5, f_surf=0.4)]
    artifact = _fit(item for case in cases for item in case)

    def states(case: list[dict[str, Any]]) -> list[torch.Tensor]:
        return [item["state"] for item in case] + [case[-1]["next_state"]]

    f_surfs = (0.25, 0.6, 0.4)
    expected = {
        "state": [torch.stack(states(case), dim=1).reshape(4, -1) for case in cases],
        "grain_moisture": [torch.stack([_grain(state, f) for state in states(case)]).reshape(1, -1) for case, f in zip(cases, f_surfs, strict=True)],
        "static": [case[0]["static"].reshape(7, -1) for case in cases],
        "scalar": [case[0]["scalars"].reshape(8, 1) for case in cases],
        "omega_boundary": [torch.stack([item["boundary"][2:4] for item in case]).reshape(1, -1) for case in cases],
    }
    assert artifact.train_case_count == len(cases)
    for name, blocks in expected.items():
        mean, std = _case_equal(blocks)
        torch.testing.assert_close(artifact.group(name).mean, mean.to(torch.float32), rtol=1.0e-6, atol=1.0e-6)
        torch.testing.assert_close(artifact.group(name).std, std.to(torch.float32), rtol=1.0e-6, atol=1.0e-6)


def test_a_long_and_a_short_case_weigh_the_same() -> None:
    """A case with many transitions moves the Train mean exactly as far as a case with one transition."""
    short, long = _case("short", 1, offset=0.0, f_surf=0.25), _case("long", 6, offset=10.0, f_surf=0.25)
    artifact = _fit([*short, *long])

    def case_mean(case: list[dict[str, Any]]) -> torch.Tensor:
        values = [item["state"] for item in case] + [case[-1]["next_state"]]
        return torch.stack(values).to(torch.float64).mean(dim=(0, 2, 3))

    expected = (case_mean(short) + case_mean(long)) / 2.0
    torch.testing.assert_close(artifact.group("state").mean, expected.to(torch.float32))


def test_the_terminal_state_is_the_stored_state_not_a_float32_reconstruction() -> None:
    """Fit a case's stored terminal state, which ``state + increment`` loses once float32 rounds the increment.

    From 1.0 to 1e-9 the float32 increment rounds to -1.0, so a reconstruction gives 0.0; the +1 and -1 pixels of the
    first two channels cancel, which leaves the stored 1e-9 visible in their means. The water channels stay zero, because
    grain moisture refuses negative water.
    """
    item = _item(0)
    state = torch.zeros(4, 2, 2)
    state[0:2, 0] = torch.tensor([1.0, -1.0])
    stored_next = torch.zeros(4, 2, 2)
    stored_next[0:2, 0, 0] = 1.0e-9
    item["state"], item["next_state"] = state, stored_next
    assert not torch.equal(state + (stored_next - state), stored_next)

    artifact = _fit([item], "stored-terminal")

    expected = {
        "state": torch.stack((state, stored_next), dim=1).reshape(4, -1),
        "grain_moisture": torch.stack([_grain(value, 0.25) for value in (state, stored_next)]).reshape(1, -1),
    }
    for name, block in expected.items():
        mean, std = _case_equal([block])
        torch.testing.assert_close(artifact.group(name).mean, mean.to(torch.float32), rtol=1.0e-6, atol=1.0e-12)
        torch.testing.assert_close(artifact.group(name).std, std.to(torch.float32), rtol=1.0e-6, atol=1.0e-12)


def test_scaling_ignores_external_eval_values() -> None:
    """Fit Train items only, so later changes to an evaluation item leave the digest unchanged."""
    train_items = [_item(0, startup=0.0), _item(1, startup=0.0)]
    evaluation = _item(2)
    first = _fit(train_items)
    evaluation["state"].fill_(1.0e9)
    second = _fit(train_items)
    assert first.train_case_count == 1
    assert first.digest == second.digest


def test_scaling_rejects_noncanonical_case_time_order_and_varying_static_fields() -> None:
    """Require the package index's grouped consecutive order and static evidence that repeats within a case."""
    with pytest.raises(ValueError, match="canonical zero-time"):
        _fit([_item(1), _item(0)], "reversed-order")
    changed = _item(1)
    changed["static"] = changed["static"] + 1.0
    with pytest.raises(ValueError, match="repeat exactly"):
        _fit([_item(0), changed], "changed-static")


def test_scaling_isolates_reused_one_shot_buffers() -> None:
    """Snapshot yielded tensors before a one-shot iterator reuses its buffers."""
    state_buffer = torch.empty_like(_item()["state"])
    next_buffer = torch.empty_like(_item()["next_state"])

    def reused_items() -> Iterator[dict[str, Any]]:
        first = _item(0)
        state_buffer.zero_()
        next_buffer.fill_(1.0)
        first["state"] = state_buffer
        first["next_state"] = next_buffer
        yield first

        second = _item(1)
        state_buffer.fill_(1.0)
        next_buffer.fill_(3.0)
        second["state"] = state_buffer
        second["next_state"] = next_buffer
        yield second

    artifact = _fit(reused_items(), "reused-buffers")

    torch.testing.assert_close(artifact.group("state").mean, torch.full((4,), 4.0 / 3.0))
    assert artifact.train_case_count == 1


def test_scaling_rejects_a_transition_that_does_not_start_at_the_stored_previous_end() -> None:
    """Bind every transition's start state exactly to the stored state its predecessor ended at."""
    first = _item(0)
    second = _item(1)
    second["state"] = torch.nextafter(second["state"], torch.full_like(second["state"], float("inf")))

    with pytest.raises(ValueError, match="stored state its predecessor ended at"):
        _fit([first, second], "conflicting-direct-endpoint")


def test_scaling_keeps_direct_and_transition_duplicates_exact() -> None:
    """Reject a duplicate whose stored start or end state differs, however slightly."""
    original = _item(0)
    changed_state = copy.deepcopy(original)
    changed_state["state"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="direct transient state"):
        _fit([original, changed_state], "conflicting-direct-duplicate")

    changed_next = copy.deepcopy(original)
    changed_next["next_state"][0, 0, 1] = torch.nextafter(changed_next["next_state"][0, 0, 1], torch.tensor(float("inf")))
    with pytest.raises(ValueError, match="transition evidence"):
        _fit([original, changed_next], "conflicting-transition-duplicate")


def test_scaling_persists_raw_zero_statistics_and_floors_the_operational_scale() -> None:
    """Keep exact-zero statistics while every encoding divides by the floored scale."""
    zero_item = _item(0)
    zero_item["state"].zero_()
    zero_item["next_state"].zero_()
    artifact = _fit([zero_item], "zero-state")

    assert torch.equal(artifact.group("state").std, torch.zeros(4))
    floor = datasets.preprocessing.normalization.SCALE_FLOOR
    assert torch.equal(artifact.state_scale, torch.full((4,), floor))


def test_regular_omega_fits_when_every_startup_flag_is_absent() -> None:
    """Fit shared omega statistics from regular endpoints unconditionally."""
    artifact = _fit(
        [
            _item(0, startup=0.0, regular_omega=(0.1, 0.3)),
            _item(1, startup=0.0, regular_omega=(0.5, 0.7)),
        ]
    )
    omega = artifact.group("omega_boundary")
    assert omega.mean.item() == pytest.approx(0.4)
    assert omega.std.item() > 0.0


def test_scaling_serialization_is_strict_and_device_independent() -> None:
    """Keep runtime transfer hash-stable and let the tensorizer refuse a tampered identity."""
    artifact = _artifact()
    state = artifact.state_dict()
    restored = type(artifact).from_state_dict(state)
    on_cpu = restored.to(torch.device("cpu"))

    assert restored.digest == artifact.digest == on_cpu.digest
    assert restored.spatial_shape == artifact.spatial_shape
    assert all(statistics.mean.device == torch.device("cpu") for statistics in on_cpu.groups.values())

    changed_shape = copy.deepcopy(state)
    changed_shape["spatial_shape"] = [0, 2]
    with pytest.raises(ValueError, match="spatial_shape"):
        type(artifact).from_state_dict(changed_shape)

    for key, value in (("model_channel_names", ["wrong"]), ("positional_embedding", "grid")):
        changed = copy.deepcopy(state)
        changed["preprocessing"]["tensorizer"][key] = value
        with pytest.raises(ValueError, match="identities disagree"):
            TransientTensorizer(_spec(), type(artifact).from_state_dict(changed))


def _expected_identity(artifact: TrainScaling) -> dict[str, Any]:
    """Return the envelope scaling identity that exactly matches one scaling's repeated identity fields."""
    return {
        "task": artifact.task,
        "task_contract_digest": artifact.task_contract_digest,
        "dataset": copy.deepcopy(dict(artifact.dataset_identity)),
        "train_membership_digest": artifact.train_membership_digest,
        "preprocessing": copy.deepcopy(dict(artifact.preprocessing)),
    }


@pytest.mark.parametrize(
    ("field", "reported"),
    [
        ("dataset_identity", "dataset_identity"),
        ("train_membership_digest", "train_membership_digest"),
        ("tensorizer", "preprocessing"),
        ("spatial_view", "preprocessing"),
    ],
)
def test_scaling_state_is_admitted_only_under_every_repeated_envelope_identity(field: str, reported: str) -> None:
    """Refuse a state whose repeated identity field differs from the envelope identity, naming the field."""
    artifact = _artifact()
    identity = _expected_identity(artifact)
    normalization = datasets.preprocessing.normalization
    assert normalization.admit_scaling_state(artifact.state_dict(), identity=identity).digest == artifact.digest

    changed = copy.deepcopy(identity)
    if field == "dataset_identity":
        changed["dataset"]["dataset_id"] = "foreign"
    elif field == "train_membership_digest":
        changed["train_membership_digest"] = _digest("foreign-membership")
    elif field == "tensorizer":
        changed["preprocessing"]["tensorizer"] = _spec(temporal="none").as_dict()
    else:
        changed["preprocessing"]["spatial_view"]["effective_nx"] += 1
    with pytest.raises(ValueError, match=reported):
        normalization.admit_scaling_state(artifact.state_dict(), identity=changed)


def test_scaling_rejects_direct_runtime_dtype_mismatch() -> None:
    """Keep scaling dtype admission strict for callers that bypass preparation."""
    artifact = _artifact()
    with pytest.raises(ValueError, match="scaling device/dtype"):
        artifact.encode_state(torch.zeros((1, 4, 2, 2), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_scaling_rejects_direct_cpu_state() -> None:
    """Keep scaling device admission strict for an unprepared CPU state."""
    device = torch.device("cuda", torch.cuda.current_device())
    artifact = _artifact().to(device)
    with pytest.raises(ValueError, match="scaling device/dtype"):
        artifact.encode_state(torch.zeros((1, 4, 2, 2), dtype=torch.float32))


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
