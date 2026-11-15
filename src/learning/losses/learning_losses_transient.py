"""Compose the data-only transient drying loss on reconstructed Train-scaled states.

Responsibilities:
  - Apply the shared group-weighted state loss to predicted and reference states
  - Apply one explicit data weight
  - Expose named total, data, and per-term data components for every training unit

Design principles:
  - The data term is the state error alone; no increment or auxiliary term exists beside it
  - Under teacher forcing the state error equals the Train-scaled increment error; self-fed states accumulate theirs
  - No steady-flow residual or boundary evaluator is constructed

This module does NOT:
  - Reconstruct states, choose curricula, or run model recurrence
  - Define the state loss or its weights; ``losses.state`` and the task's error terms own them
  - Implement physical residual losses
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from .learning_losses_state import StateErrorLoss, StateScaling


class TransientStateLoss(nn.Module):
    """Weight the shared state loss of reconstructed drying states and expose one component per error term."""

    physics_enabled = False

    def __init__(self, *, data_loss: StateErrorLoss, data_weight: float = 1.0) -> None:
        """Validate the data weight and retain the shared state loss."""
        super().__init__()
        if isinstance(data_weight, bool) or not isinstance(data_weight, int | float) or not math.isfinite(data_weight) or data_weight < 0.0:
            message = "data_weight must be a non-negative finite number."
            raise ValueError(message)
        self.data_loss = data_loss
        self.data_weight = float(data_weight)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return the exact ordered named loss-component interface."""
        return ("total", "data", *(f"data_{term.id}" for term in self.data_loss.terms))

    def set_scaling(self, scaling: StateScaling) -> None:
        """Bind the run's Train state scales to the shared state loss."""
        self.data_loss.set_scaling(scaling)

    def unit_components(self, prediction: torch.Tensor, *, y: torch.Tensor, x: Any = None) -> dict[str, torch.Tensor]:
        """Return every named applied component ``[U]`` of Train-scaled predicted and reference states ``[U, L, 4, Y, X]``."""
        del x
        terms = {f"data_{name}": self.data_weight * value for name, value in self.data_loss.unit_term_contributions(prediction, y).items()}
        data = torch.stack(list(terms.values())).sum(dim=0)
        return {"total": data, "data": data, **terms}

    def forward(self, prediction: torch.Tensor, y: torch.Tensor, *, x: Any = None) -> torch.Tensor:
        """Return the differentiable total transient loss averaged over the units."""
        return self.unit_components(prediction, y=y, x=x)["total"].mean()
