"""
learning_training_adapter.py

Define task-owned adaptation contracts for the generic training lifecycle.

Responsibilities:
  - Describe differentiable task-specific microbatch results
  - Expose reconstructed evaluation views for semantic metrics
  - Define checkpointable adapter and matched-compute protocol surfaces

Design principles:
  - The generic loop owns optimization and lifecycle sequencing
  - Adapters report logical and measured work without mutating optimizer state
  - Persisted adapter state is explicit and strict

This module does NOT:
  - Implement a task rollout, optimizer, checkpoint format, or metric
  - Move arbitrary batches without task-owned validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import nn


@dataclass(frozen=True, slots=True)
class AdapterTrainingStep:
    """Return one differentiable microbatch result and exact logical work."""

    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    sample_count: int
    processed_target_transitions: int
    forward_transitions: int


@dataclass(frozen=True, slots=True)
class AdapterEvaluationViews:
    """Return aligned normalized and physical evidence for metric accumulation."""

    normalized_prediction: torch.Tensor
    normalized_target: torch.Tensor
    physical_prediction: torch.Tensor
    physical_target: torch.Tensor
    sample_count: int
    processed_target_transitions: int
    valid_mask: torch.Tensor | None = None
    f_surf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OptimizerWork:
    """Describe a completed optimizer group; ``successful`` means parameters changed."""

    successful: bool
    microbatches: int
    processed_target_transitions: int
    forward_transitions: int
    optimizer_device_seconds: float | None
    wall_seconds: float
    peak_cuda_memory_bytes: int | None = None


@runtime_checkable
class CheckpointAdapter(Protocol):
    """Expose strict epoch-boundary continuation state."""

    def state_dict(self) -> dict[str, Any]:
        """Return exact JSON-compatible task-owned continuation state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore already validated task-owned continuation state."""
        ...


@runtime_checkable
class TrainingAdapter(CheckpointAdapter, Protocol):
    """Adapt task-owned physical batches to the generic training lifecycle."""

    @property
    def gradient_accumulation_steps(self) -> int:
        """Return configured microbatches per optimizer update."""
        ...

    def begin_epoch(self, *, epoch_index: int, total_epochs: int) -> None:
        """Prepare deterministic adapter state for one completed epoch."""
        ...

    def prepare_batch(self, raw_batch: Any, *, device: torch.device, training: bool) -> Any:
        """Move and validate one task-owned batch on the resolved device."""
        ...

    def training_step(self, model: nn.Module, batch: Any, loss: nn.Module) -> AdapterTrainingStep:
        """Return one differentiable task-owned training microbatch."""
        ...

    def evaluation_step(self, model: nn.Module, batch: Any) -> AdapterEvaluationViews:
        """Return reconstructed task-owned metric views."""
        ...

    def record_optimizer_work(self, work: OptimizerWork) -> None:
        """Record only optimizer work that the loop reports as completed."""
        ...

    def record_validation_work(self, seconds: float) -> None:
        """Record completed validation wall time excluded from the primary budget."""
        ...

    def budget_state(self) -> dict[str, Any]:
        """Return strict matched-compute stop and remaining-budget evidence."""
        ...

    def should_stop(self) -> bool:
        """Return whether the sticky matched-compute budget has completed."""
        ...

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one eligible completed-epoch selection metric."""
        ...

    def validate_terminal_state(self, *, best_metric: float | None, best_epoch: int | None) -> None:
        """Bind terminal matched-compute evidence to the selected checkpoint."""
        ...

    def telemetry_state(self) -> dict[str, float | int | None]:
        """Return current task, curriculum, and compute telemetry."""
        ...
