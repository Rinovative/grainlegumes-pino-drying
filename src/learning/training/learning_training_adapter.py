"""Define task-owned adaptation contracts for the generic training lifecycle.

Responsibilities:
  - Describe differentiable task-specific microbatch results
  - Expose reconstructed evaluation views for semantic metrics
  - Describe completed optimizer work that the loop reports back to adapters

Design principles:
  - The generic loop owns optimization and lifecycle sequencing
  - Adapters report logical and measured work without mutating optimizer state

This module does NOT:
  - Implement a task rollout, optimizer, checkpoint format, or metric
  - Move arbitrary batches without task-owned validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class AdapterTrainingStep:
    """
    Return one differentiable microbatch result and exact logical work.

    One loader batch may yield several steps, for example consecutive windows
    of whole trajectories. ``consumed_items`` is the number of loader items the
    step completes: the batch size on its last step and zero before.
    """

    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    sample_count: int
    processed_target_transitions: int
    forward_transitions: int
    consumed_items: int


@dataclass(frozen=True, slots=True)
class AdapterEvaluationViews:
    """
    Return aligned normalized and physical evidence for metric accumulation.

    The primary views drive checkpoint selection. The one-step views predict the
    same targets with every transition started from its reference state and feed
    the one-step guardrail.
    """

    normalized_prediction: torch.Tensor
    normalized_target: torch.Tensor
    physical_prediction: torch.Tensor
    physical_target: torch.Tensor
    one_step_normalized_prediction: torch.Tensor
    one_step_physical_prediction: torch.Tensor
    sample_count: int
    processed_target_transitions: int
    valid_mask: torch.Tensor | None = None
    f_surf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OptimizerWork:
    """Describe a completed optimizer group."""

    microbatches: int
    processed_target_transitions: int
    forward_transitions: int
    optimizer_device_seconds: float | None
    wall_seconds: float
    peak_cuda_memory_bytes: int | None = None
    train_seconds: float | None = None
    batch_cursor: int | None = None
