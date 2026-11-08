"""Execute differentiable teacher-forced and self-fed transient model rollouts.

Responsibilities:
  - Run every Stage-A transition from its reference physical current state and score its one-step state
  - Continue official RNO hidden state from a caller-supplied state and capture it at chosen steps
  - Reconstruct self-fed Stage-B states without detaching model predictions
  - Detach and per-sample assemble RNO states at truncation boundaries
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

from collections.abc import Callable, Iterable, Mapping  # noqa: TC003 - Public annotations must resolve through get_type_hints.
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, cast

import torch
from torch import nn

from .learning_transient_tensorizer import TransientBatch  # noqa: TC001 -- public annotations resolve through get_type_hints

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
    captured_hidden: Mapping[int, object | None] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def horizon(self) -> int:
        """Return the exact unpadded rollout transition count."""
        return int(self.scaled_delta.shape[1])


_STEP_INPUT_RANK = 4
_SEQUENCE_INPUT_RANK = 5
_RECURRENT_OUTPUT_COUNT = 2


def _require_model_kind(model_kind: str) -> ModelKind:
    """Validate one transient-capable model kind."""
    if model_kind not in {"fno", "uno", "rno"}:
        msg = f"Transient rollout model kind must be 'fno', 'uno', or 'rno', got {model_kind!r}."
        raise ValueError(msg)
    return model_kind  # type: ignore[return-value]


def _require_prediction(
    value: object,
    *,
    batch_size: int,
    height: int,
    width: int,
    require_finite: bool = True,
) -> torch.Tensor:
    """Require one real floating scaled increment under an explicit finite policy."""
    if not isinstance(require_finite, bool):
        msg = "Transient prediction require_finite must be boolean."
        raise TypeError(msg)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        msg = "Transient model prediction must be one real floating-point tensor."
        raise TypeError(msg)
    if value.shape != (batch_size, 4, height, width):
        msg = "Transient model prediction must have shape [B,4,Y,X]."
        raise ValueError(msg)
    if require_finite and not bool(torch.isfinite(value).all().item()):
        msg = "Transient model prediction contains non-finite values."
        raise FloatingPointError(msg)
    return value


def predict_step(
    model: nn.Module,
    step_input: torch.Tensor,
    *,
    model_kind: str,
    hidden: object | None,
    model_call: Callable[[Callable[[], object]], object] | None = None,
    require_finite_output: bool = True,
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
    require_finite_output : bool, optional
        Require finite model values. Artifact diagnostics alone may disable this
        after strict source-input admission to preserve the returned IEEE values.

    Returns
    -------
    tuple[torch.Tensor, object | None]
        Finite scaled increment with shape ``[B,4,Y,X]`` and the next RNO
        hidden state. Stateless models always return ``None``.

    """
    kind = _require_model_kind(model_kind)
    if not isinstance(require_finite_output, bool):
        msg = "require_finite_output must be boolean."
        raise TypeError(msg)
    if not isinstance(model, nn.Module):
        msg = "Transient prediction requires one torch.nn.Module."
        raise TypeError(msg)
    if not isinstance(step_input, torch.Tensor) or not step_input.is_floating_point() or step_input.is_complex():
        msg = "Transient step input must be one real floating-point tensor."
        raise TypeError(msg)
    if step_input.ndim != _STEP_INPUT_RANK or min(step_input.shape[0], step_input.shape[1], step_input.shape[2], step_input.shape[3]) < 1:
        msg = "Transient step input must have non-empty shape [B,C,Y,X]."
        raise ValueError(msg)
    if not bool(torch.isfinite(step_input).all().item()):
        msg = "Transient step input contains non-finite values."
        raise FloatingPointError(msg)
    batch_size, _, height, width = step_input.shape

    def invoke_model() -> object:
        """Dispatch exactly one semantic model call without output validation."""
        if kind == "rno":
            # Padded states keep the padding region's recurrence; unpadding and re-zeroing it each step would break the carry.
            return model(step_input.unsqueeze(1), init_hidden_states=hidden, return_hidden_states=True, keep_states_padded=True)
        return model(step_input)

    result = invoke_model() if model_call is None else model_call(invoke_model)
    if kind == "rno":
        if not isinstance(result, tuple) or len(result) != _RECURRENT_OUTPUT_COUNT:
            msg = "Official RNO must return (prediction, hidden_states) when return_hidden_states=True."
            raise TypeError(msg)
        prediction, next_hidden = result
    else:
        prediction = result
        next_hidden = None
    admitted = (
        _require_prediction(
            prediction,
            batch_size=batch_size,
            height=height,
            width=width,
        )
        if require_finite_output
        else _require_prediction(
            prediction,
            batch_size=batch_size,
            height=height,
            width=width,
            require_finite=False,
        )
    )
    return admitted, next_hidden


def detach_hidden(hidden: object | None) -> object | None:
    """
    Return one recurrent state without autograd history.

    Detaching at a window boundary is the truncation of truncated
    backpropagation through time: the state value carries over while gradients
    stop. ``None`` (no state yet) passes through unchanged.
    """
    if hidden is None:
        return None
    if not isinstance(hidden, list) or not all(isinstance(layer, torch.Tensor) for layer in hidden):
        msg = "Recurrent hidden state must be one list of per-layer tensors."
        raise TypeError(msg)
    return [layer.detach() for layer in hidden]


def select_hidden_per_sample(
    snapshots: Mapping[int, object | None],
    positions: torch.Tensor,
    rows: torch.Tensor | None = None,
) -> object | None:
    """
    Assemble one batch state from per-step snapshots at sample-specific positions.

    Parameters
    ----------
    snapshots : Mapping[int, object | None]
        Full-batch states captured after the given number of steps.
    positions : torch.Tensor
        One captured step count per full-batch sample.
    rows : torch.Tensor or None, optional
        Full-batch sample indices to assemble, in output order. Defaults to all.

    Returns
    -------
    object | None
        Per-layer tensors whose output sample ``i`` is sample ``rows[i]`` of
        ``snapshots[positions[rows[i]]]``, or ``None`` when every selected
        snapshot is still the model's own initial state.

    Raises
    ------
    ValueError
        If a position was not captured or the selection mixes the initial state
        with populated states, which one batched call cannot represent.

    """
    selected_rows = list(range(int(positions.shape[0]))) if rows is None else [int(value) for value in rows.tolist()]
    requested = [int(positions[row].item()) for row in selected_rows]
    if any(position not in snapshots for position in requested):
        msg = "Recurrent state was not captured at every requested position."
        raise ValueError(msg)
    selected = [snapshots[position] for position in requested]
    if all(state is None for state in selected):
        return None
    if any(state is None for state in selected):
        msg = "Recurrent snapshots mix the initial and populated states within one batch."
        raise ValueError(msg)
    layers = cast("list[list[torch.Tensor]]", selected)
    return [torch.stack([layers[index][layer][row] for index, row in enumerate(selected_rows)]) for layer in range(len(layers[0]))]


def _predict_steps(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    model_kind: ModelKind,
    initial_hidden: object | None,
    capture: frozenset[int],
) -> tuple[torch.Tensor, dict[int, object | None]]:
    """Predict a contiguous teacher-forced sequence with RNO hidden carry and snapshots."""
    if inputs.ndim != _SEQUENCE_INPUT_RANK:
        msg = "Transient teacher-forced inputs must have shape [B,L,C,Y,X]."
        raise ValueError(msg)
    batch_size, length, _, height, width = inputs.shape
    if model_kind != "rno":
        if initial_hidden is not None or capture:
            msg = "Stateless transient models neither accept nor expose a hidden state."
            raise ValueError(msg)
        flat = inputs.reshape(batch_size * length, *inputs.shape[2:])
        prediction, _ = predict_step(model, flat, model_kind=model_kind, hidden=None)
        return prediction.reshape(batch_size, length, 4, height, width), {}
    hidden = initial_hidden
    captured: dict[int, object | None] = {0: hidden} if 0 in capture else {}
    predictions: list[torch.Tensor] = []
    for index in range(length):
        prediction, hidden = predict_step(model, inputs[:, index], model_kind=model_kind, hidden=hidden)
        predictions.append(prediction)
        if index + 1 in capture:
            captured[index + 1] = hidden
    return torch.stack(predictions, dim=1), captured


def _require_capture(capture: Iterable[int], *, length: int) -> frozenset[int]:
    """Validate requested snapshot step counts against one rollout length."""
    resolved = frozenset(capture)
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= length for value in resolved):
        msg = "Recurrent snapshot positions must be step counts within the rollout."
        raise ValueError(msg)
    return resolved


def _targets(batch: TransientBatch) -> torch.Tensor:
    """Reconstruct exact reference physical states for every target transition."""
    return batch.state.unsqueeze(1) + batch.target.cumsum(dim=1)


def _rollout_from_deltas(
    batch: TransientBatch,
    tensorizer: _RolloutTensorizer,
    scaled_delta: torch.Tensor,
    *,
    teacher_forced: bool,
    captured_hidden: Mapping[int, object | None],
) -> TransientRollout:
    """
    Reconstruct aligned state views from predicted scaled increments.

    Teacher-forced views add each increment to its reference current state, so
    every entry is a one-step prediction. Self-fed views chain the increments
    from the window's initial state, as the autonomous model does.
    """
    batch_size, length, _, height, width = scaled_delta.shape
    if scaled_delta.shape != (batch_size, batch.rollout_length, 4, height, width):
        msg = "Transient rollout delta shape drifted from the admitted target window."
        raise ValueError(msg)
    physical_target = _targets(batch)
    current = batch.state
    states: list[torch.Tensor] = []
    for index in range(length):
        if teacher_forced and index > 0:
            current = physical_target[:, index - 1]
        current = tensorizer.reconstruct_next_state(current, scaled_delta[:, index])
        states.append(current)
    physical_prediction = torch.stack(states, dim=1)
    normalized_prediction = tensorizer.scaling.encode_state(physical_prediction)
    normalized_target = tensorizer.scaling.encode_state(physical_target)
    return TransientRollout(
        scaled_delta=scaled_delta,
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        normalized_prediction=normalized_prediction,
        normalized_target=normalized_target,
        captured_hidden=MappingProxyType(dict(captured_hidden)),
    )


def teacher_forced_rollout(
    model: nn.Module,
    batch: TransientBatch,
    tensorizer: _RolloutTensorizer,
    *,
    model_kind: str,
    initial_hidden: object | None = None,
    capture: Iterable[int] = (),
) -> TransientRollout:
    """
    Predict every reference transition and reconstruct its one-step state view.

    ``initial_hidden`` continues an RNO state from the preceding transitions of
    the same trajectory. ``capture`` lists step counts after which the RNO state
    is returned in ``captured_hidden``; ``0`` is the initial state.
    """
    kind = _require_model_kind(model_kind)
    scaled_delta, captured = _predict_steps(
        model,
        batch.sequence_input,
        model_kind=kind,
        initial_hidden=initial_hidden,
        capture=_require_capture(capture, length=batch.rollout_length),
    )
    return _rollout_from_deltas(batch, tensorizer, scaled_delta, teacher_forced=True, captured_hidden=captured)


def self_fed_rollout(
    model: nn.Module,
    batch: TransientBatch,
    tensorizer: _RolloutTensorizer,
    *,
    model_kind: str,
    initial_hidden: object | None = None,
) -> TransientRollout:
    """Autoregressively reconstruct every admitted transition without detaching state."""
    kind = _require_model_kind(model_kind)
    if kind != "rno" and initial_hidden is not None:
        msg = "Stateless transient models cannot start from a hidden state."
        raise ValueError(msg)
    batch_size, length, _, height, width = batch.target.shape
    current = batch.state
    hidden = initial_hidden
    deltas: list[torch.Tensor] = []
    for index in range(length):
        step_input = tensorizer.assemble_step(current, batch.static, batch.boundary[:, index], batch.scalars, batch.t_n[:, index])
        delta, hidden = predict_step(model, step_input, model_kind=kind, hidden=hidden)
        if delta.shape != (batch_size, 4, height, width):
            msg = "Transient step prediction shape drifted from the admitted rollout."
            raise RuntimeError(msg)
        deltas.append(delta)
        current = tensorizer.reconstruct_next_state(current, delta)
    return _rollout_from_deltas(batch, tensorizer, torch.stack(deltas, dim=1), teacher_forced=False, captured_hidden={})
