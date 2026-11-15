"""Construct optimizers and per-stage learning-rate schedules for training.

Responsibilities:
  - Build AdamW optimizers from resolved configs
  - Start every stage at its own start rate: optimizer.lr, times lr_scale for B and A1
  - Anneal the rate by a half cosine over the completed stage budget fraction
  - Provide the halve-on-plateau comparison variant stepped at validation points
  - Return no schedule for a constant rate (``scheduler: null``)

Design principles:
  - YAML semantics map directly to optimizer arguments
  - A stage never inherits a schedule: each builds its own from its start rate
  - Backend imports occur only when the corresponding factory is called
  - Unsupported types fail fast

This module does NOT:
  - Execute gradients or epochs. ``learning.training.loop`` owns training
  - Measure budget progress. ``learning.training.budget`` owns it
  - Persist optimizer or schedule state. Checkpoint services own restoration
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:
    from torch import nn
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
        - optimizer.kind: optimizer type
        - optimizer.lr: learning rate
        - optimizer.weight_decay: weight decay
        - optimizer.betas: betas for AdamW
        - optimizer.second_moment_floor: Adam second-moment denominator floor

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

    opt_config = config["optimizer"]
    opt_type = opt_config["kind"].lower()

    if opt_type != "adamw":
        msg = f"Only 'adamw' optimizer is supported, got: {opt_type}"
        raise ValueError(msg)

    lr = opt_config.get("lr")
    if lr is None:
        msg = "Missing required 'optimizer.lr' in config"
        raise ValueError(msg)

    weight_decay = opt_config["weight_decay"]
    betas_config = opt_config["betas"]
    if not isinstance(betas_config, Sequence) or isinstance(betas_config, (str, bytes)) or len(betas_config) != _ADAM_BETA_COUNT:
        msg = f"optimizer.betas must contain exactly two numeric values, got: {betas_config!r}"
        raise ValueError(msg)
    betas = (float(betas_config[0]), float(betas_config[1]))
    second_moment_floor = opt_config["second_moment_floor"]

    return AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=betas,
        eps=float(second_moment_floor),
    )


def cosine_factor(progress: float) -> float:
    """Return (1 + cos πp) / 2 for the completed budget fraction p, clipped at 1."""
    if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or progress < 0:
        msg = f"Budget progress must be finite and non-negative, got {progress!r}."
        raise ValueError(msg)
    return (1.0 + math.cos(math.pi * min(1.0, float(progress)))) / 2.0


class CosineSchedule:
    """
    Anneal every rate as η(p) = η₀ · (1 + cos πp) / 2 over the stage budget fraction p.

    The schedule has no warm-up, no restart within a stage and no floor: it reaches 0 at
    the budget end. It is a function of committed budget progress only, so it advances
    independently of validation and a resume restores it from the budget state alone.
    """

    kind: Final = "cosine"

    def __init__(self, optimizer: Optimizer, *, start_rate: float) -> None:
        """Bind the optimizer and the stage start rate η₀."""
        self.optimizer = optimizer
        self.start_rate = start_rate

    def advance(self, progress: float) -> None:
        """Set every rate for the next optimizer group at committed progress p."""
        rate = self.start_rate * cosine_factor(progress)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def validated(self, metric: float) -> None:
        """Ignore validation: the cosine depends on budget progress only."""
        del metric

    def state_dict(self) -> dict[str, Any]:
        """Return the schedule identity; the budget state carries its progress."""
        return {"kind": self.kind, "start_rate": self.start_rate}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Accept only this schedule's own identity."""
        if dict(state) != self.state_dict():
            msg = f"Saved learning-rate schedule {dict(state)!r} differs from this stage's {self.state_dict()!r}."
            raise ValueError(msg)


class PlateauSchedule:
    """
    Halve-on-plateau comparison variant: reduce every rate after validation points without improvement.

    It exists only until the scheduler comparison of the research plan decides between it,
    the cosine and a constant rate. It is stepped at the stage's validation points with
    the selection objective, has no floor, and every stage starts it fresh at its own
    start rate.
    """

    kind: Final = "reduce_on_plateau"

    def __init__(self, optimizer: Optimizer, *, start_rate: float, factor: float, patience: int, mode: Literal["min", "max"]) -> None:
        """Bind a fresh PyTorch plateau scheduler without a floor to the optimizer."""
        from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: PLC0415

        self.start_rate = start_rate
        # eps=0 too: PyTorch's default skips reductions smaller than 1e-8, a hidden floor.
        self._plateau = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=0.0, eps=0.0)

    def advance(self, progress: float) -> None:
        """Ignore budget progress: the plateau variant moves only at validation points."""
        del progress

    def validated(self, metric: float) -> None:
        """Step the plateau rule with one validation point's objective."""
        self._plateau.step(metric)

    def state_dict(self) -> dict[str, Any]:
        """Return the start rate and the plateau bookkeeping."""
        return {"kind": self.kind, "start_rate": self.start_rate, "plateau": self._plateau.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the plateau bookkeeping of this stage's own schedule."""
        if not isinstance(state, Mapping) or set(state) != {"kind", "start_rate", "plateau"}:
            msg = "Saved plateau schedule state must hold exactly kind, start_rate and plateau."
            raise ValueError(msg)
        if (state["kind"], state["start_rate"]) != (self.kind, self.start_rate):
            msg = "Saved plateau schedule belongs to another schedule or start rate."
            raise ValueError(msg)
        self._plateau.load_state_dict(dict(state["plateau"]))


LearningRateSchedule = CosineSchedule | PlateauSchedule


def build_scheduler(optimizer: Optimizer, config: Mapping[str, Any], *, lr_scale: float = 1.0) -> LearningRateSchedule | None:
    """
    Start one stage's learning-rate schedule at its start rate.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer of the stage; a continuation passes it after its A0 moments and step counters were restored.
    config : Mapping[str, Any]
        Resolved config with ``optimizer.lr`` and ``scheduler`` (``kind: cosine``, ``kind: reduce_on_plateau``
        with ``factor`` and ``patience``, or ``null`` for a constant rate).
    lr_scale : float, optional
        Factor on the A0 start rate ``optimizer.lr``; B and A1 pass their configured ``lr_scale``.

    Returns
    -------
    LearningRateSchedule | None
        The stage schedule, or ``None`` for a constant rate.

    Raises
    ------
    ValueError
        If the scale is not positive and finite, the kind is unknown or the objective direction is invalid.

    Notes
    -----
    Every parameter group is set to the start rate ``lr_scale · optimizer.lr`` first, so a
    continuation never carries A0's scheduled rate or scheduler state. A resume then
    restores the saved optimizer and schedule state over it.

    """
    if isinstance(lr_scale, bool) or not isinstance(lr_scale, (int, float)) or not math.isfinite(lr_scale) or lr_scale <= 0:
        msg = f"Continuation lr_scale must be positive and finite, got {lr_scale!r}."
        raise ValueError(msg)
    start_rate = float(config["optimizer"]["lr"]) * float(lr_scale)
    for group in optimizer.param_groups:
        group["lr"] = start_rate
    sched_config = config.get("scheduler")
    if sched_config is None:
        return None
    if sched_config["kind"] == CosineSchedule.kind:
        return CosineSchedule(optimizer, start_rate=start_rate)
    if sched_config["kind"] == PlateauSchedule.kind:
        mode_by_direction: dict[str, Literal["min", "max"]] = {"minimize": "min", "maximize": "max"}
        direction = config["evaluation"]["objective"]["direction"]
        if direction not in mode_by_direction:
            msg = f"evaluation.objective.direction must be 'minimize' or 'maximize', got: {direction!r}"
            raise ValueError(msg)
        return PlateauSchedule(
            optimizer,
            start_rate=start_rate,
            factor=float(sched_config["factor"]),
            patience=int(sched_config["patience"]),
            mode=mode_by_direction[direction],
        )
    msg = f"Unknown scheduler kind {sched_config['kind']!r}; supported: cosine, reduce_on_plateau."
    raise ValueError(msg)
