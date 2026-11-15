"""Build semantic supervised and physics-informed loss compositions.

Responsibilities:
  - Resolve the semantic data-loss identifiers shared by both tasks
  - Build the task's group-weighted state loss with its error terms and, for H1, its physical grid
  - Build one named composition interface for supervised and physics training

Design principles:
  - Resolved semantic identifiers select implementations without exposing class names
  - Construction is explicit, task-aware, and placed on the caller-selected device
  - One state-loss definition and one config shape serve both tasks

This module does NOT:
  - Define task contracts, error terms or allowed physics choices. ``domain.tasks`` owns them
  - Implement derivatives, equations, residuals, or diagnostics. ``domain.physics`` does
  - Define evaluation metrics or dataset aggregation. ``learning.metrics`` does
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from src import domain

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from . import learning_losses_pino as pino
    from . import learning_losses_state as state
    from . import learning_losses_transient as transient

DATA_LOSS_KINDS: tuple[str, ...] = ("h1", "mse")


def resolve_data_loss_kind(kind: str) -> str:
    """Return an exact semantic data-loss identifier or raise naming the admitted ones."""
    if kind not in DATA_LOSS_KINDS:
        msg = f"Unknown loss identifier {kind!r}. Available losses: {', '.join(DATA_LOSS_KINDS)}."
        raise ValueError(msg)
    return kind


def build_data_loss(task: domain.tasks.spec.TaskSpec, kind: str, *, grid_spacing: Sequence[float] | None = None) -> state.StateErrorLoss:
    """
    Build the group-weighted state loss of one task.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Task whose error terms weight the state fields in metric-field order.
    kind : str
        ``"mse"`` or ``"h1"``.
    grid_spacing : Sequence[float] | None, optional
        Physical grid spacing in metres, rows then columns; required for ``"h1"``.

    Returns
    -------
    learning_losses_state.StateErrorLoss
        Unweighted state loss of Train-scaled states.

    """
    from . import learning_losses_state as state  # noqa: PLC0415

    return state.StateErrorLoss(
        kind=cast("state.StateLossKind", resolve_data_loss_kind(kind)),
        terms=task.error_terms,
        fields=task.metric_names,
        grid_spacing=grid_spacing,
    )


def _resolved_weight(config: dict[str, Any], name: str) -> pino.LinearWarmup:
    """
    Build one validated linear warmup from a resolved physics-weight node.

    Missing mappings, unknown warmup kinds, and malformed scalar values fail at
    the semantic weight name instead of silently producing a schedule. The
    returned object owns non-negative target and epoch validation.
    """
    from . import learning_losses_pino as pino  # noqa: PLC0415

    weight = config.get(name)
    if not isinstance(weight, dict):
        msg = f"loss.physics.{name} must be a mapping."
        raise TypeError(msg)
    warmup = weight.get("warmup")
    if not isinstance(warmup, dict):
        msg = f"loss.physics.{name}.warmup must be a mapping."
        raise TypeError(msg)
    if warmup.get("kind") != "linear":
        msg = f"Unknown warmup identifier {warmup.get('kind')!r}. Expected 'linear'."
        raise ValueError(msg)
    return pino.LinearWarmup(
        target=float(weight["target"]),
        epochs=int(warmup["epochs"]),
    )


def build_training_loss(
    config: dict[str, Any], *, device: torch.device, grid_spacing: Sequence[float] | None = None
) -> pino.SemanticComposedLoss | transient.TransientStateLoss:
    """
    Build the unified semantic training loss from resolved configuration.

    Both tasks train the same group-weighted state loss. Supervised and
    physics-informed steady configurations return the same type and named
    component interface; the task contract owns the equation, continuity, and
    boundary formulations. The caller supplies the already resolved concrete
    runtime device and binds the Train scaling with ``set_scaling``. This factory
    never performs availability fallback.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved task, loss, and physics configuration.
    device : torch.device
        Concrete CPU or CUDA device selected by the service boundary.
    grid_spacing : Sequence[float] | None, optional
        Physical grid spacing in metres, rows then columns; required when the
        data loss is ``h1``.

    Returns
    -------
    learning_losses_pino.SemanticComposedLoss | learning_losses_transient.TransientStateLoss
        Device-bound composition with a stable named-component interface.

    Raises
    ------
    TypeError
        If the device or required resolved mappings have invalid types.
    ValueError
        If task physics, derivatives, weights, or crop settings contradict
        their registered semantic contracts.

    """
    import torch  # noqa: PLC0415

    from . import learning_losses_pino as pino  # noqa: PLC0415

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Loss construction requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    task = domain.tasks.registry.get_task(str(config["task"]))
    loss_config = config.get("loss")
    if not isinstance(loss_config, dict):
        msg = "Resolved config must contain a loss mapping."
        raise TypeError(msg)
    data_config = loss_config.get("data")
    physics_config = loss_config.get("physics")
    if not isinstance(data_config, dict) or not isinstance(physics_config, dict):
        msg = "Resolved loss config must contain data and physics mappings."
        raise TypeError(msg)
    data_loss = build_data_loss(task, str(data_config["kind"]), grid_spacing=grid_spacing)

    if task.id == "transient_drying":
        from . import learning_losses_transient as transient  # noqa: PLC0415

        if physics_config.get("enabled") is not False:
            msg = "Transient drying supports data-only training; loss.physics.enabled must be false."
            raise ValueError(msg)
        return transient.TransientStateLoss(data_loss=data_loss, data_weight=float(data_config["weight"])).to(device)

    task_physics = domain.tasks.registry.resolve_physics(task.physics.kind)
    derivatives_config = physics_config.get("derivatives")
    if not isinstance(derivatives_config, dict):
        msg = "loss.physics.derivatives must be a mapping."
        raise TypeError(msg)
    derivatives = domain.physics.derivatives.build_derivative_operator(str(derivatives_config["kind"]))
    steady_loss = pino.SemanticComposedLoss(
        data_loss=data_loss,
        data_weight=float(data_config["weight"]),
        physics_enabled=bool(physics_config["enabled"]),
        physics_kind=task_physics.kind,
        input_fields=task.input_names,
        output_fields=task.output_names,
        boundary=task_physics.boundary,
        derivatives=derivatives,
        momentum_weight=_resolved_weight(physics_config, "momentum_weight"),
        continuity_weight=_resolved_weight(physics_config, "continuity_weight"),
        boundary_weight=_resolved_weight(physics_config, "boundary_weight"),
        interior_crop=int(physics_config["interior_crop"]),
    )
    return steady_loss.to(device)
