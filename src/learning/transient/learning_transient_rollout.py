# ruff: noqa: EM101, EM102, TRY003, PLR2004, TC001
"""
learning_transient_rollout.py

Execute differentiable teacher-forced and self-fed transient model rollouts.

Responsibilities:
  - Run every Stage-A transition from its reference physical current state
  - Carry official RNO hidden state only within one independent rollout window
  - Reconstruct self-fed Stage-B states without detaching model predictions
  - Expose aligned normalized and physical rollout evidence

Design principles:
  - Tensorizer-owned channel assembly remains authoritative
  - Rollout windows never pad, wrap, or substitute unavailable transitions
  - Shape and finiteness checks fail before silent metric corruption

This module does NOT:
  - Select curricula, calculate losses, or mutate optimizer state
  - Use neuraloperator RNO sequence-return mode
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 - Public annotations must resolve through get_type_hints.
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn

from .learning_transient_tensorizer import TransientBatch

ModelKind = Literal["fno", "uno", "rno"]


class _StateScaling(Protocol):
    """Expose state encoding needed by rollout evidence construction."""

    def encode_state(self, value: torch.Tensor) -> torch.Tensor: ...


class _RolloutTensorizer(Protocol):
    """Expose the minimal tensorizer surface used by differentiable rollouts."""

    @property
    def scaling(self) -> _StateScaling: ...

    def reconstruct_next_state(self, current_state: torch.Tensor, scaled_delta: torch.Tensor) -> torch.Tensor: ...

    def assemble_step(
        self,
        current_state: torch.Tensor,
        static: torch.Tensor,
        boundary: torch.Tensor,
        scalars: torch.Tensor,
        t_n: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class TransientRollout:
    """Hold aligned increment and reconstructed state evidence for one window."""

    scaled_delta: torch.Tensor
    physical_prediction: torch.Tensor
    physical_target: torch.Tensor
    normalized_prediction: torch.Tensor
    normalized_target: torch.Tensor

    @property
    def horizon(self) -> int:
        """Return the exact unpadded rollout transition count."""
        return int(self.scaled_delta.shape[1])


def _require_model_kind(model_kind: str) -> ModelKind:
    """Validate one transient-capable model kind."""
    if model_kind not in {"fno", "uno", "rno"}:
        raise ValueError(f"Transient rollout model kind must be 'fno', 'uno', or 'rno', got {model_kind!r}.")
    return model_kind  # type: ignore[return-value]


def _require_prediction(value: object, *, batch_size: int, height: int, width: int) -> torch.Tensor:
    """Require one finite real floating scaled increment with the expected BCHW shape."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        raise TypeError("Transient model prediction must be one real floating-point tensor.")
    if value.shape != (batch_size, 4, height, width):
        raise ValueError("Transient model prediction must have shape [B,4,Y,X].")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("Transient model prediction contains non-finite values.")
    return value


def predict_step(
    model: nn.Module,
    step_input: torch.Tensor,
    *,
    model_kind: str,
    hidden: object | None,
    model_call: Callable[[Callable[[], object]], object] | None = None,
) -> tuple[torch.Tensor, object | None]:
    """
    Predict one transient scaled increment with model-kind-specific recurrence.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded transient neural operator.
    step_input : torch.Tensor
        Finite model-ready input with shape ``[B,C,Y,X]``.
    model_kind : {"fno", "uno", "rno"}
        Persisted semantic model identifier.
    hidden : object or None
        Official RNO hidden state from the preceding step in this request.
    model_call : callable, optional
        Wrapper receiving one zero-argument model invocation. The wrapper may
        measure only that invocation; prediction validation remains outside it.

    Returns
    -------
    tuple[torch.Tensor, object | None]
        Finite scaled increment with shape ``[B,4,Y,X]`` and the next RNO
        hidden state. Stateless models always return ``None``.

    """
    kind = _require_model_kind(model_kind)
    if not isinstance(model, nn.Module):
        raise TypeError("Transient prediction requires one torch.nn.Module.")
    if not isinstance(step_input, torch.Tensor) or not step_input.is_floating_point() or step_input.is_complex():
        raise TypeError("Transient step input must be one real floating-point tensor.")
    if step_input.ndim != 4 or min(step_input.shape[0], step_input.shape[1], step_input.shape[2], step_input.shape[3]) < 1:
        raise ValueError("Transient step input must have non-empty shape [B,C,Y,X].")
    if not bool(torch.isfinite(step_input).all().item()):
        raise FloatingPointError("Transient step input contains non-finite values.")
    batch_size, _, height, width = step_input.shape

    def invoke_model() -> object:
        """Dispatch exactly one semantic model call without output validation."""
        if kind == "rno":
            return model(step_input.unsqueeze(1), init_hidden_states=hidden, return_hidden_states=True)
        return model(step_input)

    result = invoke_model() if model_call is None else model_call(invoke_model)
    if kind == "rno":
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Official RNO must return (prediction, hidden_states) when return_hidden_states=True.")
        prediction, next_hidden = result
    else:
        prediction = result
        next_hidden = None
    return _require_prediction(prediction, batch_size=batch_size, height=height, width=width), next_hidden


def _predict_steps(model: nn.Module, inputs: torch.Tensor, *, model_kind: ModelKind) -> torch.Tensor:
    """Predict a contiguous teacher-forced sequence with RNO hidden carry."""
    if inputs.ndim != 5:
        raise ValueError("Transient teacher-forced inputs must have shape [B,L,C,Y,X].")
    batch_size, length, _, height, width = inputs.shape
    if model_kind != "rno":
        flat = inputs.reshape(batch_size * length, *inputs.shape[2:])
        prediction, _ = predict_step(model, flat, model_kind=model_kind, hidden=None)
        return prediction.reshape(batch_size, length, 4, height, width)
    hidden: object | None = None
    predictions: list[torch.Tensor] = []
    for index in range(length):
        prediction, hidden = predict_step(model, inputs[:, index], model_kind=model_kind, hidden=hidden)
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def _targets(batch: TransientBatch) -> torch.Tensor:
    """Reconstruct exact reference physical states for every target transition."""
    return batch.state.unsqueeze(1) + batch.target.cumsum(dim=1)


def _rollout_from_deltas(batch: TransientBatch, tensorizer: _RolloutTensorizer, scaled_delta: torch.Tensor) -> TransientRollout:
    """Reconstruct aligned state views from predicted scaled increments."""
    batch_size, length, _, height, width = scaled_delta.shape
    if scaled_delta.shape != (batch_size, batch.rollout_length, 4, height, width):
        raise ValueError("Transient rollout delta shape drifted from the admitted target window.")
    current = batch.state
    states: list[torch.Tensor] = []
    for index in range(length):
        current = tensorizer.reconstruct_next_state(current, scaled_delta[:, index])
        states.append(current)
    physical_prediction = torch.stack(states, dim=1)
    physical_target = _targets(batch)
    normalized_prediction = tensorizer.scaling.encode_state(physical_prediction)
    normalized_target = tensorizer.scaling.encode_state(physical_target)
    return TransientRollout(
        scaled_delta=scaled_delta,
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        normalized_prediction=normalized_prediction,
        normalized_target=normalized_target,
    )


def teacher_forced_rollout(model: nn.Module, batch: TransientBatch, tensorizer: _RolloutTensorizer, *, model_kind: str) -> TransientRollout:
    """Predict every reference transition and reconstruct its cumulative state view."""
    kind = _require_model_kind(model_kind)
    scaled_delta = _predict_steps(model, batch.sequence_input, model_kind=kind)
    return _rollout_from_deltas(batch, tensorizer, scaled_delta)


def self_fed_rollout(
    model: nn.Module, batch: TransientBatch, tensorizer: _RolloutTensorizer, *, model_kind: str, truncation_length: int | None = None
) -> TransientRollout:
    """Autoregressively reconstruct every admitted transition without detaching state."""
    kind = _require_model_kind(model_kind)
    if truncation_length is not None:
        raise ValueError("Transient self-fed rollouts default to no truncation; truncation is not admitted.")
    batch_size, length, _, height, width = batch.target.shape
    current = batch.state
    hidden: object | None = None
    deltas: list[torch.Tensor] = []
    for index in range(length):
        step_input = tensorizer.assemble_step(current, batch.static, batch.boundary[:, index], batch.scalars, batch.t_n[:, index])
        delta, hidden = predict_step(model, step_input, model_kind=kind, hidden=hidden)
        if delta.shape != (batch_size, 4, height, width):
            raise RuntimeError("Transient step prediction shape drifted from the admitted rollout.")
        deltas.append(delta)
        current = tensorizer.reconstruct_next_state(current, delta)
    return _rollout_from_deltas(batch, tensorizer, torch.stack(deltas, dim=1))
