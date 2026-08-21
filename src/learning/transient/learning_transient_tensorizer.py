"""
learning_transient_tensorizer.py

Tensorize admitted physical transient batches into task-owned model inputs.

Responsibilities:
  - Validate and normalize one-step and rollout batch shapes
  - Assemble exact profile channels and optional normalized time
  - Build teacher-forced sequence inputs from reference increments
  - Reconstruct absolute physical state from predicted increments

Design principles:
  - The tensorizer is the sole owner of channel order and broadcast semantics
  - Both boundary endpoints remain independent model features
  - Absent startup support becomes an exact neutral encoded triple
  - Physical evidence remains available beside scaled model tensors

This module does NOT:
  - Fit preprocessing statistics or invoke neuraloperator models
  - Create transitions, timesteps, or Dataset samples
  - Choose teacher-forcing or rollout curricula
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from src.learning.learning_temporal import apply_temporal_conditioning

from .learning_transient_contracts import TransientTensorizerSpec
from .learning_transient_scaling import TransientScalingArtifact

_STATE_CHANNELS = 4
_STATIC_CHANNELS = 7
_BOUNDARY_CHANNELS = 9
_SCALAR_CHANNELS = 8
_SPATIAL_BATCH_RANK = 4
_SEQUENCE_BATCH_RANK = 5
_VECTOR_BATCH_RANK = 2


@dataclass(frozen=True, slots=True)
class TransientBatch:
    """Hold validated physical evidence and model-ready transient tensors."""

    state: torch.Tensor
    target: torch.Tensor
    static: torch.Tensor
    boundary: torch.Tensor
    scalars: torch.Tensor
    t_n: torch.Tensor
    t_n_plus_1: torch.Tensor
    dt: torch.Tensor
    step_input: torch.Tensor
    sequence_input: torch.Tensor
    scaled_target: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the physical sample count."""
        return int(self.state.shape[0])

    @property
    def rollout_length(self) -> int:
        """Return the normalized transition-window length."""
        return int(self.target.shape[1])


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    """Require one finite real floating tensor without copying it."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"Transient {label} must be a real floating-point tensor."
        raise TypeError(message)
    if not bool(torch.isfinite(value).all().item()):
        message = f"Transient {label} contains non-finite values."
        raise ValueError(message)
    return value


def _require_shared_placement(
    values: tuple[tuple[str, torch.Tensor], ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Require every physical tensor to share scaler device and dtype."""
    for label, value in values:
        if value.device != device or value.dtype != dtype:
            message = f"Transient {label} must use scaler device/dtype {device}/{dtype}, got {value.device}/{value.dtype}."
            raise ValueError(message)


class TransientTensorizer:
    """Own exact transient profile assembly and physical reconstruction."""

    def __init__(
        self,
        spec: TransientTensorizerSpec,
        scaling: TransientScalingArtifact,
    ) -> None:
        """Bind a task-owned profile to matching Train-only scaling evidence."""
        if not isinstance(spec, TransientTensorizerSpec):
            message = "spec must be a TransientTensorizerSpec."
            raise TypeError(message)
        if not isinstance(scaling, TransientScalingArtifact):
            message = "scaling must be a TransientScalingArtifact."
            raise TypeError(message)
        if scaling.tensorizer != spec:
            message = "Tensorizer and scaling artifact identities disagree."
            raise ValueError(message)
        self.spec = spec
        self.scaling = scaling

    def tensorize(self, raw: Mapping[str, Any]) -> TransientBatch:
        """Validate collated one-step/rollout evidence and assemble model tensors."""
        if not isinstance(raw, Mapping):
            message = "Transient batch must be a mapping."
            raise TypeError(message)
        state = _finite_tensor(raw.get("state"), label="state")
        target = _finite_tensor(raw.get("target"), label="target")
        static = _finite_tensor(raw.get("static"), label="static")
        boundary = _finite_tensor(raw.get("boundary"), label="boundary")
        scalars = _finite_tensor(raw.get("scalars"), label="scalars")
        time = raw.get("time")
        if not isinstance(time, Mapping):
            message = "Transient batch time must be a mapping."
            raise TypeError(message)
        t_n = _finite_tensor(time.get("t_n"), label="time.t_n")
        t_next = _finite_tensor(
            time.get("t_n_plus_1"),
            label="time.t_n_plus_1",
        )
        dt = _finite_tensor(time.get("dt"), label="time.dt")

        if state.ndim != _SPATIAL_BATCH_RANK or state.shape[1] != _STATE_CHANNELS:
            message = "Transient state must have shape [B,4,Y,X]."
            raise ValueError(message)
        batch_size, _, height, width = state.shape
        if min(batch_size, height, width) < 1:
            message = "Transient state axes must be non-empty."
            raise ValueError(message)
        if static.shape != (
            batch_size,
            _STATIC_CHANNELS,
            height,
            width,
        ):
            message = "Transient static conditioning must have shape [B,7,Y,X]."
            raise ValueError(message)
        if scalars.shape != (batch_size, _SCALAR_CHANNELS):
            message = "Transient scalar conditioning must have shape [B,8]."
            raise ValueError(message)

        if target.ndim == _SPATIAL_BATCH_RANK:
            target = target.unsqueeze(1)
        if boundary.ndim == _VECTOR_BATCH_RANK:
            boundary = boundary.unsqueeze(1)
        if target.ndim != _SEQUENCE_BATCH_RANK:
            message = "Transient target must have shape [B,L,4,Y,X]."
            raise ValueError(message)
        rollout_length = int(target.shape[1])
        if target.shape != (
            batch_size,
            rollout_length,
            _STATE_CHANNELS,
            height,
            width,
        ):
            message = "Transient target channel or spatial shape is invalid."
            raise ValueError(message)
        if boundary.shape != (
            batch_size,
            rollout_length,
            _BOUNDARY_CHANNELS,
        ):
            message = "Transient boundary conditioning must have shape [B,L,9]."
            raise ValueError(message)
        if rollout_length < 1:
            message = "Transient target rollout length must be positive."
            raise ValueError(message)

        t_n = self._time_values(
            t_n,
            batch_size=batch_size,
            rollout_length=rollout_length,
            label="t_n",
        )
        t_next = self._time_values(
            t_next,
            batch_size=batch_size,
            rollout_length=rollout_length,
            label="t_n_plus_1",
        )
        dt = self._time_values(
            dt,
            batch_size=batch_size,
            rollout_length=rollout_length,
            label="dt",
        )
        placements = (
            ("state", state),
            ("target", target),
            ("static", static),
            ("boundary", boundary),
            ("scalars", scalars),
            ("time.t_n", t_n),
            ("time.t_n_plus_1", t_next),
            ("time.dt", dt),
        )
        _require_shared_placement(
            placements,
            device=self.scaling.device,
            dtype=self.scaling.dtype,
        )
        if not torch.allclose(
            dt,
            torch.ones_like(dt),
            rtol=0.0,
            atol=1.0e-6,
        ) or not torch.allclose(
            t_next - t_n,
            dt,
            rtol=0.0,
            atol=1.0e-6,
        ):
            message = "Transient tensorizer requires regular fixed 1h time evidence."
            raise ValueError(message)

        sequence_input = self._teacher_forced_sequence_inputs(
            initial_state=state,
            static=static,
            boundary=boundary,
            scalars=scalars,
            t_n=t_n,
            target=target,
        )
        scaled_target = self.scaling.encode_delta(target)
        return TransientBatch(
            state=state,
            target=target,
            static=static,
            boundary=boundary,
            scalars=scalars,
            t_n=t_n,
            t_n_plus_1=t_next,
            dt=dt,
            step_input=sequence_input[:, 0],
            sequence_input=sequence_input,
            scaled_target=scaled_target,
        )

    @staticmethod
    def _time_values(
        value: torch.Tensor,
        *,
        batch_size: int,
        rollout_length: int,
        label: str,
    ) -> torch.Tensor:
        """Normalize collated scalar/sequence time values to [B,L]."""
        if value.ndim == 1 and value.shape == (batch_size,):
            value = value.unsqueeze(1)
        expected = (batch_size, rollout_length)
        if value.shape != expected:
            message = f"Transient {label} must have shape {expected}, got {tuple(value.shape)}."
            raise ValueError(message)
        return value

    def _teacher_forced_sequence_inputs(
        self,
        *,
        initial_state: torch.Tensor,
        static: torch.Tensor,
        boundary: torch.Tensor,
        scalars: torch.Tensor,
        t_n: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble every transition from its reference physical current state."""
        zero_increment = torch.zeros_like(target[:, :1])
        preceding = torch.cat((zero_increment, target[:, :-1]), dim=1)
        current_states = initial_state.unsqueeze(1) + preceding.cumsum(dim=1)
        inputs = [
            self.assemble_step(
                current_state=current_states[:, step],
                static=static,
                boundary=boundary[:, step],
                scalars=scalars,
                t_n=t_n[:, step],
            )
            for step in range(target.shape[1])
        ]
        return torch.stack(inputs, dim=1)

    def assemble_step(
        self,
        current_state: torch.Tensor,
        static: torch.Tensor,
        boundary: torch.Tensor,
        scalars: torch.Tensor,
        t_n: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble one model step from physical current-interval evidence."""
        current_state = _finite_tensor(
            current_state,
            label="current_state",
        )
        static = _finite_tensor(static, label="static")
        boundary = _finite_tensor(boundary, label="boundary")
        scalars = _finite_tensor(scalars, label="scalars")
        t_n = _finite_tensor(t_n, label="t_n")
        if current_state.ndim != _SPATIAL_BATCH_RANK or current_state.shape[1] != _STATE_CHANNELS:
            message = "current_state must have shape [B,4,Y,X]."
            raise ValueError(message)
        batch_size, _, height, width = current_state.shape
        if static.shape != (
            batch_size,
            _STATIC_CHANNELS,
            height,
            width,
        ):
            message = "static must have shape [B,7,Y,X]."
            raise ValueError(message)
        if boundary.shape != (batch_size, _BOUNDARY_CHANNELS):
            message = "boundary must have shape [B,9]."
            raise ValueError(message)
        if scalars.shape != (batch_size, _SCALAR_CHANNELS):
            message = "scalars must have shape [B,8]."
            raise ValueError(message)
        if t_n.shape != (batch_size,):
            message = "t_n must have shape [B]."
            raise ValueError(message)
        _require_shared_placement(
            (
                ("current_state", current_state),
                ("static", static),
                ("boundary", boundary),
                ("scalars", scalars),
                ("t_n", t_n),
            ),
            device=self.scaling.device,
            dtype=self.scaling.dtype,
        )

        state_encoded = self.scaling.encode_state(current_state)
        static_encoded = self.scaling.encode_static(static)
        scalar_encoded = self.scaling.encode_scalars(scalars)
        boundary_encoded = self.scaling.encode_boundary(boundary)
        named: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.scaling.state_names):
            named[name] = state_encoded[:, index]
        for index, name in enumerate(self.scaling.static_names):
            named[name] = static_encoded[:, index]
        for index, name in enumerate(self.scaling.boundary_names):
            named[name] = (
                boundary_encoded[:, index]
                .view(
                    batch_size,
                    1,
                    1,
                )
                .expand(batch_size, height, width)
            )
        for index, name in enumerate(self.scaling.scalar_names):
            named[name] = (
                scalar_encoded[:, index]
                .view(
                    batch_size,
                    1,
                    1,
                )
                .expand(batch_size, height, width)
            )

        normalized_time = apply_temporal_conditioning(
            t_n,
            self.spec.temporal_conditioning,
            configured_regular_horizon=self.scaling.horizon,
        )
        if normalized_time is not None:
            named["normalized_current_time"] = normalized_time.view(
                batch_size,
                1,
                1,
            ).expand(batch_size, height, width)

        return torch.stack(
            [named[name] for name in self.spec.model_channel_names],
            dim=1,
        )

    def reconstruct_next_state(
        self,
        current_state: torch.Tensor,
        scaled_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Return finite physical next state from a scaled predicted increment."""
        current = _finite_tensor(current_state, label="current state")
        delta = _finite_tensor(scaled_delta, label="scaled delta")
        if current.ndim != _SPATIAL_BATCH_RANK or current.shape[1] != _STATE_CHANNELS or delta.shape != current.shape:
            message = "Transient reconstruction requires matching [B,4,Y,X] tensors."
            raise ValueError(message)
        _require_shared_placement(
            (("current state", current), ("scaled delta", delta)),
            device=self.scaling.device,
            dtype=self.scaling.dtype,
        )
        result = current + self.scaling.decode_delta(delta)
        if not bool(torch.isfinite(result).all().item()):
            message = "Transient reconstruction produced non-finite state."
            raise FloatingPointError(message)
        return result
