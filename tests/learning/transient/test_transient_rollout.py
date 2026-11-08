"""Protect explicit transient teacher-forced and autonomous rollout behavior.

Responsibilities:
  - Check RNO hidden-state boundaries and differentiable autonomous feedback
  - Check carried, detached RNO state across truncated windows and per-sample state assembly

Design principles:
  - Hidden state persists within a rollout, continues only from an explicit carried state

This module does NOT:
  - Select training budgets or publish evaluation artifacts
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from src.learning.transient.learning_transient_rollout import (
    detach_hidden,
    select_hidden_per_sample,
    self_fed_rollout,
    teacher_forced_rollout,
)
from src.learning.transient.learning_transient_tensorizer import TransientBatch


class _Scaling:
    def encode_state(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Tensorizer:
    scaling = _Scaling()

    def reconstruct_next_state(self, current_state: torch.Tensor, scaled_delta: torch.Tensor) -> torch.Tensor:
        return current_state + scaled_delta

    def assemble_step(
        self, current_state: torch.Tensor, static: torch.Tensor, boundary: torch.Tensor, scalars: torch.Tensor, t_n: torch.Tensor
    ) -> torch.Tensor:
        del static, boundary, scalars, t_n
        return torch.cat((current_state, torch.ones_like(current_state[:, :1])), dim=1)


def _batch(length: int = 3) -> TransientBatch:
    state = torch.zeros(1, 4, 1, 1)
    target = torch.ones(1, length, 4, 1, 1)
    sequence = torch.stack([torch.cat((torch.full_like(state, float(step)), torch.ones_like(state[:, :1])), dim=1) for step in range(length)], dim=1)
    return TransientBatch(
        state=state,
        target=target,
        static=torch.zeros(1, 7, 1, 1),
        boundary=torch.zeros(1, length, 9),
        scalars=torch.zeros(1, 8),
        t_n=torch.arange(length, dtype=torch.float32).view(1, length),
        t_n_plus_1=torch.arange(1, length + 1, dtype=torch.float32).view(1, length),
        dt=torch.ones(1, length),
        step_input=sequence[:, 0],
        sequence_input=sequence,
        scaled_target=target,
    )


class _RNO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.hidden_inputs: list[object | None] = []

    def forward(self, value: torch.Tensor, *, init_hidden_states: object | None, return_hidden_states: bool, keep_states_padded: bool) -> Any:
        assert value.shape[1] == 1
        assert return_hidden_states is True
        assert keep_states_padded is True
        self.hidden_inputs.append(init_hidden_states)
        if init_hidden_states is not None and not isinstance(init_hidden_states, int):
            message = "Synthetic RNO hidden state must be an integer or null."
            raise TypeError(message)
        prediction = value[:, 0, :4] * self.weight
        next_hidden = 0 if init_hidden_states is None else init_hidden_states + 1
        return prediction, next_hidden


class _Operator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :4] * self.weight + 1.0


def test_rno_teacher_forcing_carries_hidden_within_and_resets_between_windows() -> None:
    """Use one explicit RNO call per transition rather than sequence-return mode."""
    model = _RNO()
    batch = _batch()
    first = teacher_forced_rollout(model, batch, _Tensorizer(), model_kind="rno")
    second = teacher_forced_rollout(model, batch, _Tensorizer(), model_kind="rno")
    assert first.scaled_delta.shape == (1, 3, 4, 1, 1)
    assert second.scaled_delta.shape == first.scaled_delta.shape
    assert model.hidden_inputs == [None, 0, 1, None, 0, 1]
    first.scaled_delta.sum().backward()
    assert model.weight.grad is not None


def test_self_fed_rollout_keeps_feedback_differentiable() -> None:
    """Feed reconstructed predictions into later transitions without detach."""
    model = _Operator()
    rollout = self_fed_rollout(model, _batch(), _Tensorizer(), model_kind="fno")
    assert rollout.scaled_delta.shape == (1, 3, 4, 1, 1)
    assert torch.allclose(rollout.scaled_delta[:, 1], torch.full((1, 4, 1, 1), 1.5))
    rollout.physical_prediction[:, -1].sum().backward()
    assert model.weight.grad is not None
    # With w = 0.5 and c0 = 0 each channel runs d_i = w * c_i + 1 and c_{i+1} = c_i + d_i, so c = (0, 1, 2.5), and the
    # terminal state is d_0 + d_1 + d_2. Differentiating through the fed-back state gives dc_1/dw = 0,
    # dd_1/dw = c_1 + w * dc_1/dw = 1, dc_2/dw = 1 and dd_2/dw = c_2 + w * dc_2/dw = 3, so 0 + 1 + 3 = 4 per channel
    # and 16 over four channels. Detached feedback would give dd_2/dw = c_2 = 2.5 and 14.
    assert model.weight.grad.item() == pytest.approx(16.0)


def _slice(batch: TransientBatch, start: int, length: int) -> TransientBatch:
    """Return one consecutive window whose initial state is the reference state at ``start``."""
    stop = start + length
    return TransientBatch(
        state=batch.state + batch.target[:, :start].sum(dim=1),
        target=batch.target[:, start:stop],
        static=batch.static,
        boundary=batch.boundary[:, start:stop],
        scalars=batch.scalars,
        t_n=batch.t_n[:, start:stop],
        t_n_plus_1=batch.t_n_plus_1[:, start:stop],
        dt=batch.dt[:, start:stop],
        step_input=batch.sequence_input[:, start],
        sequence_input=batch.sequence_input[:, start:stop],
        scaled_target=batch.scaled_target[:, start:stop],
    )


class _StateRNO(nn.Module):
    """Accumulate weighted inputs in a one-layer tensor state; each prediction is the new state."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor, *, init_hidden_states: object | None, return_hidden_states: bool, keep_states_padded: bool) -> Any:
        assert return_hidden_states is True
        assert keep_states_padded is True
        step = value[:, 0, :4] * self.weight
        state = step if init_hidden_states is None else cast("list[torch.Tensor]", init_hidden_states)[0] + step
        return state, [state]


def test_rno_state_carries_across_windows_with_truncated_gradients() -> None:
    """Continuing the captured state reproduces one long pass while the carried state has no history."""
    model = _StateRNO()
    full = teacher_forced_rollout(model, _batch(4), _Tensorizer(), model_kind="rno")
    first = teacher_forced_rollout(model, _slice(_batch(4), 0, 2), _Tensorizer(), model_kind="rno", capture=(2,))
    carried = detach_hidden(first.captured_hidden[2])
    second = teacher_forced_rollout(model, _slice(_batch(4), 2, 2), _Tensorizer(), model_kind="rno", initial_hidden=carried)

    assert torch.allclose(torch.cat((first.scaled_delta, second.scaled_delta), dim=1), full.scaled_delta)
    assert all(not layer.requires_grad for layer in cast("list[torch.Tensor]", carried))
    second.scaled_delta.sum().backward()
    assert model.weight.grad is not None
    # Outputs c + w*x2 and c + w*x2 + w*x3 over four channels with inputs 2 and 3 give 4 * (2*2 + 3) = 28;
    # an untruncated state c = w*(0 + 1) would add 8.
    assert model.weight.grad.item() == pytest.approx(28.0)


def test_per_sample_states_come_from_their_own_positions_and_never_mix_with_the_initial_state() -> None:
    """Assemble one batch state from different capture positions and reject the unrepresentable mix."""
    snapshots: dict[int, object | None] = {0: None, 1: [torch.zeros(2, 1)], 2: [torch.ones(2, 1)]}
    state = cast("list[torch.Tensor]", select_hidden_per_sample(snapshots, torch.tensor([2, 1])))
    assert state[0].flatten().tolist() == [1.0, 0.0]
    assert select_hidden_per_sample(snapshots, torch.tensor([0, 0])) is None
    with pytest.raises(ValueError, match="mix the initial"):
        select_hidden_per_sample(snapshots, torch.tensor([0, 1]))
