"""Define task-owned adaptation contracts for the generic training lifecycle.

Responsibilities:
  - Describe the per-unit loss shares a task computes for one memory microbatch
  - Describe the task part of the one update path: fixing a group's draws and computing unit losses
  - Expose reconstructed evaluation views for semantic metrics
  - Describe completed optimizer work that the loop reports back to adapters

Design principles:
  - The generic loop owns grouping, unit weights, accumulation, optimization and lifecycle sequencing
  - Tasks report per-unit losses and logical work without mutating optimizer state

This module does NOT:
  - Implement a task rollout, optimizer, checkpoint format, or metric
  - Move arbitrary batches without task-owned validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import torch
    from torch import nn

    from src.datasets.runtime.dataset_runtime_units import UnitGroup


@dataclass(frozen=True, slots=True)
class UnitLosses:
    """
    Return the differentiable loss shares of the units of one memory microbatch.

    ``components`` maps every named component, ``total`` included, to one tensor
    ``[U]`` in the microbatch's unit order. The shares one unit receives over all
    yielded results sum to its loss averaged over its supervised transitions and
    grid; a recurrent unit yields one share per window, so each window's graph is
    released before the next. ``supervised_transitions`` counts the predictions
    that entered the loss (a steady case counts one); ``forward_transitions``
    adds the gradient-free teacher-forced warm-up transitions.
    """

    components: dict[str, torch.Tensor]
    supervised_transitions: int
    forward_transitions: int


class UnitLossTask(Protocol):
    """Compute the task-specific part of the one update path: a unit's loss."""

    def begin_group(self, group: UnitGroup, *, progress: float) -> None:
        """Fix every random draw of one update group at the committed budget fraction, before any of its microbatches is loaded."""
        ...

    def unit_losses(self, model: nn.Module, raw_batch: Any, loss: nn.Module, *, positions: tuple[int, ...]) -> Iterator[UnitLosses]:
        """Yield the loss shares of the microbatch whose units sit at ``positions`` of the current group."""
        ...


@dataclass(frozen=True, slots=True)
class AdapterEvaluationViews:
    """
    Return aligned normalized and physical evidence for metric accumulation.

    The primary views drive checkpoint selection. The one-step views predict the
    same targets with every transition started from its reference state and feed
    the one-step guardrail, a diagnostic. ``case_complete`` ``[B]`` marks the cases
    whose autonomous prediction is finite over the whole trajectory; an incomplete
    case holds its non-finite evidence and NaN after its stop. ``bound_violations``
    maps every state channel to the batch's diagnostic of finite autonomous values
    outside the task's physical bounds: row-aligned with the batch, each case's
    physical time in hours of its first violation (``None`` without one).
    """

    normalized_prediction: torch.Tensor
    normalized_target: torch.Tensor
    physical_prediction: torch.Tensor
    physical_target: torch.Tensor
    one_step_normalized_prediction: torch.Tensor
    one_step_physical_prediction: torch.Tensor
    sample_count: int
    processed_target_transitions: int
    case_complete: torch.Tensor
    bound_violations: Mapping[str, tuple[float | None, ...]]
    valid_mask: torch.Tensor | None = None
    f_surf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OptimizerWork:
    """Describe a completed optimizer group's work; its time is on the optimisation clock of the stage budget only."""

    microbatches: int
    processed_target_transitions: int
    forward_transitions: int
    peak_cuda_memory_bytes: int | None = None
