"""
learning_losses_transient.py

Compute transient drying loss over scaled state increments.

Responsibilities:
  - Apply configurable Huber or mean-squared error by output channel
  - Reduce explicitly across batch, horizon, and spatial axes
  - Expose named data, optional reconstructed-state auxiliary, and total terms
  - Preserve four task-owned channel weights in checkpoint state

Design principles:
  - Increment loss operates only in zero-preserving scaled-increment space
  - Channel weights are normalized by their sum before aggregation
  - Reconstructed-state auxiliary loss is disabled by default
  - No steady-flow residual or boundary evaluator is constructed

This module does NOT:
  - Reconstruct state, choose curriculum, or run model recurrence
  - Normalize tensors or fit channel scales
  - Implement physical residual losses
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Literal

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Sequence

TransientDataLossKind = Literal["huber", "mse"]

_STATE_CHANNELS: Final = 4
_MIN_TRANSIENT_RANK: Final = 4


def _finite_weight(value: Any, *, label: str) -> float:
    """Return one non-negative finite scalar weight."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) < 0.0:
        message = f"{label} must be a non-negative finite number."
        raise ValueError(message)
    return float(value)


class TransientIncrementLoss(nn.Module):
    """Compose scaled-increment data loss and optional state auxiliary loss."""

    physics_enabled = False
    continuity = "none"
    component_names = ("total", "data", "data_T", "data_phi", "data_w_surf", "data_w_int", "state_aux")

    def __init__(
        self,
        *,
        kind: TransientDataLossKind,
        channel_weights: Sequence[float],
        data_weight: float = 1.0,
        huber_beta: float = 1.0,
        state_aux_weight: float = 0.0,
    ) -> None:
        """Validate immutable loss semantics and register channel weights."""
        super().__init__()
        if kind not in {"huber", "mse"}:
            message = f"Unknown transient data loss {kind!r}."
            raise ValueError(message)
        if (
            isinstance(huber_beta, bool)
            or not isinstance(huber_beta, int | float)
            or not math.isfinite(float(huber_beta))
            or float(huber_beta) <= 0.0
        ):
            message = "huber_beta must be a positive finite number."
            raise ValueError(message)
        values = tuple(_finite_weight(value, label="channel weight") for value in channel_weights)
        if len(values) != _STATE_CHANNELS or sum(values) <= 0.0:
            message = "channel_weights must contain four values with positive sum."
            raise ValueError(message)
        self.kind = kind
        self.data_weight = _finite_weight(data_weight, label="data_weight")
        self.huber_beta = float(huber_beta)
        self.state_aux_weight = _finite_weight(
            state_aux_weight,
            label="state_aux_weight",
        )
        self.register_buffer(
            "channel_weights",
            torch.tensor(values, dtype=torch.float32),
            persistent=True,
        )
        self.last_components: dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_pair(
        prediction: Any,
        target: Any,
        *,
        label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Require matching finite BCHW or BLCHW four-channel tensors."""
        if not isinstance(prediction, torch.Tensor) or not isinstance(
            target,
            torch.Tensor,
        ):
            message = f"{label} prediction and target must be tensors."
            raise TypeError(message)
        if prediction.shape != target.shape or prediction.ndim < _MIN_TRANSIENT_RANK or prediction.shape[-3] != _STATE_CHANNELS:
            message = f"{label} tensors must share BCHW or BLCHW shape with four channels at axis -3."
            raise ValueError(message)
        if not prediction.is_floating_point() or not target.is_floating_point() or prediction.is_complex() or target.is_complex():
            message = f"{label} tensors must be real floating point."
            raise TypeError(message)
        if prediction.device != target.device:
            message = f"{label} tensors must share one device."
            raise ValueError(message)
        if not bool(torch.isfinite(prediction).all().item()) or not bool(
            torch.isfinite(target).all().item(),
        ):
            message = f"{label} tensors must contain only finite values."
            raise FloatingPointError(message)
        return prediction, target

    def _channel_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Return one mean per channel over every non-channel axis."""
        channel_first = values.movedim(-3, 0)
        return channel_first.reshape(_STATE_CHANNELS, -1).mean(dim=1)

    def _increment_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return four applied weighted scaled-increment channel components."""
        if self.kind == "huber":
            elementwise = functional.huber_loss(
                prediction,
                target,
                reduction="none",
                delta=self.huber_beta,
            )
        else:
            elementwise = (prediction - target).square()
        weights = self.channel_weights.to(
            device=elementwise.device,
            dtype=elementwise.dtype,
        )
        return self.data_weight * self._channel_mean(elementwise) * weights / weights.sum()

    def compute_components(
        self,
        prediction: torch.Tensor,
        *,
        y: torch.Tensor,
        predicted_state: torch.Tensor | None = None,
        target_state: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return named applied loss components for one transient microbatch."""
        del x, epoch
        predicted_delta, target_delta = self._validate_pair(
            prediction,
            y,
            label="scaled increment",
        )
        channel_data = self._increment_components(predicted_delta, target_delta)
        data = channel_data.sum()
        if self.state_aux_weight == 0.0:
            state_aux = data.new_zeros(())
        else:
            predicted_absolute, target_absolute = self._validate_pair(
                predicted_state,
                target_state,
                label="normalized reconstructed state",
            )
            state_aux = self.state_aux_weight * (predicted_absolute - target_absolute).square().mean()
        total = data + state_aux
        components = {
            "total": total,
            "data": data,
            "data_T": channel_data[0],
            "data_phi": channel_data[1],
            "data_w_surf": channel_data[2],
            "data_w_int": channel_data[3],
            "state_aux": state_aux,
        }
        self.last_components = {name: value.detach() for name, value in components.items()}
        return components

    def forward(
        self,
        prediction: torch.Tensor,
        y: torch.Tensor,
        *,
        predicted_state: torch.Tensor | None = None,
        target_state: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> torch.Tensor:
        """Return the differentiable total transient loss."""
        return self.compute_components(
            prediction,
            y=y,
            predicted_state=predicted_state,
            target_state=target_state,
            x=x,
            epoch=epoch,
        )["total"]
