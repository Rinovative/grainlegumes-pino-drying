"""Construct optimizers and schedulers for training.

Responsibilities:
  - Build AdamW optimizers from resolved configs
  - Build ReduceLROnPlateau schedulers from resolved configs
  - Return no scheduler when scheduler config is absent
  - Scale a restored continuation learning-rate schedule (rates and plateau floors) by one factor

Design principles:
  - YAML semantics map directly to optimizer arguments
  - Backend imports occur only when the corresponding factory is called
  - Unsupported types fail fast

This module does NOT:
  - Execute gradients or epochs. ``learning.training.loop`` owns training
  - Persist optimizer or scheduler state. Checkpoint services own restoration
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from torch import nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.optimizer import Optimizer

_ADAM_BETA_COUNT = 2


def build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> Optimizer:
    """
    Build optimizer from configuration.

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    config : dict[str, Any]
        Optimizer config with keys:
        - optimizer.kind: optimizer type (default: "adamw")
        - optimizer.lr: learning rate
        - optimizer.weight_decay: weight decay
        - optimizer.betas: betas for AdamW (optional)
        - optimizer.second_moment_floor: Adam second-moment denominator floor (optional)

    Returns
    -------
    Optimizer
        Initialized optimizer

    Raises
    ------
    ValueError
        If optimizer type is unknown or required parameters missing

    """
    from neuralop.training import AdamW  # noqa: PLC0415

    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("kind", "adamw").lower()

    if opt_type != "adamw":
        msg = f"Only 'adamw' optimizer is supported, got: {opt_type}"
        raise ValueError(msg)

    lr = opt_config.get("lr")
    if lr is None:
        msg = "Missing required 'optimizer.lr' in config"
        raise ValueError(msg)

    weight_decay = opt_config.get("weight_decay", 0.0)
    betas_config = opt_config.get("betas", [0.9, 0.999])
    if not isinstance(betas_config, Sequence) or isinstance(betas_config, (str, bytes)) or len(betas_config) != _ADAM_BETA_COUNT:
        msg = f"optimizer.betas must contain exactly two numeric values, got: {betas_config!r}"
        raise ValueError(msg)
    betas = (float(betas_config[0]), float(betas_config[1]))
    second_moment_floor = opt_config.get("second_moment_floor", 1e-6)

    return AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=betas,
        eps=float(second_moment_floor),
    )


def build_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
) -> ReduceLROnPlateau | None:
    """
    Build learning rate scheduler from configuration.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule
    config : dict[str, Any]
        Scheduler config with keys:
        - scheduler.kind: scheduler type (default: "reduce_on_plateau")
        - scheduler.factor: LR reduction factor (default: 0.5)
        - scheduler.patience: epochs before reduction (default: 20)
        - scheduler.min_lr: minimum LR (default: 1e-8)

    Returns
    -------
    ReduceLROnPlateau | None
        Scheduler if configured, None otherwise

    Raises
    ------
    ValueError
        If scheduler type is unknown

    """
    sched_config = config.get("scheduler")
    if sched_config is None:
        return None

    sched_type = sched_config.get("kind", "reduce_on_plateau").lower()

    if sched_type == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: PLC0415

        direction = config["evaluation"]["objective"]["direction"]
        mode_by_direction = {"minimize": "min", "maximize": "max"}
        if direction not in mode_by_direction:
            msg = f"evaluation.objective.direction must be 'minimize' or 'maximize', got: {direction!r}"
            raise ValueError(msg)
        mode_value = cast('Literal["min", "max"]', mode_by_direction[direction])
        factor = sched_config.get("factor", 0.5)
        patience = sched_config.get("patience", 20)
        min_lr = sched_config.get("min_lr", 1e-8)

        return ReduceLROnPlateau(
            optimizer,
            mode=mode_value,
            factor=float(factor),
            patience=int(patience),
            min_lr=float(min_lr),
        )
    msg = f"Only 'reduce_on_plateau' scheduler is supported, got: {sched_type}"
    raise ValueError(msg)


def scale_restored_learning_rates(
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | None,
    *,
    scale: float,
) -> list[float]:
    """
    Scale the restored continuation learning-rate schedule by one factor.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose state was just restored from a teacher handoff checkpoint.
    scheduler : ReduceLROnPlateau | None
        Restored scheduler; its floors are scaled with the rates.
    scale : float
        Positive finite continuation factor; ``1`` keeps the inherited state exactly.

    Returns
    -------
    list[float]
        Effective per-group starting learning rates.

    Raises
    ------
    ValueError
        If the factor is not positive and finite.

    Notes
    -----
    Rates and plateau floors are multiplied together, so the continuation follows the
    inherited schedule shape at a different level: a rate already at its floor stays at
    the scaled floor and later reductions remain possible. AdamW moments, step counters
    and the scheduler's best value, bad-epoch counter and cooldown stay as restored; its
    last-rate bookkeeping follows the scaled rates so saved checkpoints stay consistent.

    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        msg = f"Continuation lr_scale must be positive and finite, got {scale!r}."
        raise ValueError(msg)
    current = [float(group["lr"]) for group in optimizer.param_groups]
    if scale == 1:
        return current
    scaled = [rate * float(scale) for rate in current]
    for group, rate in zip(optimizer.param_groups, scaled, strict=True):
        group["lr"] = rate
    if scheduler is not None:
        scheduler.min_lrs = [float(floor) * float(scale) for floor in scheduler.min_lrs]
        scheduler._last_lr = list(scaled)  # noqa: SLF001 -- keep PyTorch's saved last-rate evidence aligned with the scaled groups
    return scaled
