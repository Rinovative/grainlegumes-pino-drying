"""Build small Train scalings of both tasks with chosen constant statistics for tests.

Responsibilities:
  - Construct one current-format steady or transient TrainScaling on an explicit grid
  - Construct physical steady inputs whose mean inlet pressure P is chosen

Design principles:
  - Statistics are test-owned constants, so encoded values are known exactly

This module does NOT:
  - Fit statistics from data or write normalizer files
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src import datasets, domain
from src.learning.transient.learning_transient_contracts import TemporalConditioningSpec, TransientTensorizerSpec

if TYPE_CHECKING:
    from src.learning.learning_temporal import TemporalConditioningKind


def spatial_view(shape: tuple[int, int], *, stride: int = 1, canonical: tuple[int, int] | None = None) -> dict[str, int]:
    """Return the spatial-view evidence of one effective grid, canonical equal to effective by default."""
    source = shape if canonical is None else canonical
    return {"spatial_stride": stride, "canonical_ny": source[0], "canonical_nx": source[1], "effective_ny": shape[0], "effective_nx": shape[1]}


def _scaling(
    task_id: str,
    *,
    view: dict[str, int],
    state_mean: float,
    state_std: float,
    membership_digest: str,
    dataset_identity: dict[str, Any],
    tensorizer: dict[str, Any] | None = None,
    horizon: float | None = None,
) -> datasets.preprocessing.normalization.TrainScaling:
    """Return one scaling with constant (encoded and physical) state statistics and zero-mean unit statistics elsewhere."""
    normalization = datasets.preprocessing.normalization
    task = domain.tasks.registry.get_task(task_id)
    groups = {}
    for name, names in normalization.expected_groups(task_id).items():
        mean, std = (state_mean, state_std) if name in {"state", "physical_state"} else (0.0, 1.0)
        groups[name] = normalization.ChannelStatistics(names=names, mean=torch.full((len(names),), mean), std=torch.full((len(names),), std))
    return normalization.TrainScaling(
        task=task_id,
        task_contract_digest=task.contract_digest,
        data_contract_digest=task.data_contract_digest,
        preprocessing=normalization.preprocessing_spec(task_id, spatial_view=view, tensorizer=tensorizer),
        dataset_identity=dataset_identity,
        train_membership_digest=membership_digest,
        train_case_count=1,
        spatial_shape=(view["effective_ny"], view["effective_nx"]),
        groups=groups,
        horizon=horizon,
    )


def steady_scaling(
    shape: tuple[int, int],
    *,
    state_mean: float = 0.0,
    state_std: float = 1.0,
    membership_digest: str = "a" * 64,
) -> datasets.preprocessing.normalization.TrainScaling:
    """Return a steady scaling with constant state statistics and zero-mean unit static statistics."""
    return _scaling(
        datasets.preprocessing.normalization.STEADY_TASK,
        view=spatial_view(shape),
        state_mean=state_mean,
        state_std=state_std,
        membership_digest=membership_digest,
        dataset_identity={"dataset_id": "synthetic"},
    )


def steady_inputs(batch: int, shape: tuple[int, int], *, inlet_pressure: float = 1.0) -> torch.Tensor:
    """Return physical steady inputs ``[B, C, Y, X]`` on a unit coordinate grid with constant ``p_in_bc``, so P is ``inlet_pressure``."""
    task = domain.tasks.registry.get_task(datasets.preprocessing.normalization.STEADY_TASK)
    y, x = torch.meshgrid(torch.linspace(0.0, 1.0, shape[0]), torch.linspace(0.0, 1.0, shape[1]), indexing="ij")
    fields = {"x": x, "y": y, "p_in_bc": torch.full(shape, inlet_pressure)}
    case = torch.stack([fields.get(name, torch.zeros(shape)) for name in task.input_names])
    return case.expand(batch, -1, -1, -1).clone()


def transient_scaling(
    view: dict[str, int],
    *,
    state_std: float = 1.0,
    membership_digest: str = "a" * 64,
    dataset_identity: dict[str, Any] | None = None,
    temporal_conditioning: TemporalConditioningKind = "none",
    horizon: float = 8.0,
) -> datasets.preprocessing.normalization.TrainScaling:
    """Return a transient scaling of the canonical complete profile with zero means and unit scales outside the state."""
    spec = TransientTensorizerSpec("canonical_physics_complete", TemporalConditioningSpec(temporal_conditioning))
    return _scaling(
        datasets.preprocessing.normalization.TRANSIENT_TASK,
        view=view,
        state_mean=0.0,
        state_std=state_std,
        membership_digest=membership_digest,
        dataset_identity={"synthetic": "transient"} if dataset_identity is None else dataset_identity,
        tensorizer=spec.as_dict(),
        horizon=horizon,
    )
