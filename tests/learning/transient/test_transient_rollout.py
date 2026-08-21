# ruff: noqa: S101, ANN202
"""Protect explicit transient teacher-forced and autonomous rollout behavior."""

from __future__ import annotations

import torch
from torch import nn

from src.learning.transient.learning_transient_rollout import (
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

    def forward(self, value: torch.Tensor, *, init_hidden_states: object | None, return_hidden_states: bool):
        assert value.shape[1] == 1
        assert return_hidden_states is True
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
    assert model.weight.grad.item() != 0.0
