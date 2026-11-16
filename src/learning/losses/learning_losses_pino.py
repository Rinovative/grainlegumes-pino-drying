"""Compose supervised and task-owned physics-informed loss components.

Responsibilities:
  - Combine named data, momentum, continuity, and boundary contributions
  - Apply one explicit weight and deterministic epoch warmup per physics term
  - Construct physical tensor views once for domain-owned physics evaluators
  - Expose named components for every case of a batch and reusable domain diagnostics

Design principles:
  - Named loss components remain stable across supervised and physics-enabled runs
  - Normalized predictions are converted to physical views explicitly and once
  - Warmup is deterministic at completed-epoch boundaries

This module does NOT:
  - Define equations, derivatives, residuals, or boundaries. ``domain`` owns them
  - Parse semantic config or select implementations. ``losses.factory`` owns selection
  - Define or aggregate dataset metrics. ``learning.metrics`` owns evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from src import domain

from .learning_losses_state import StateErrorLoss, StateScaling

PHYSICS_TERMS = ("momentum", "continuity", "boundary")


class PhysicalScaling(StateScaling, Protocol):
    """Define the Train-scaling surface that the data and physics terms need."""

    def decode_inputs(self, value: Tensor, *, reference: Tensor | None = None) -> Tensor:
        """Return physical model inputs from standardized ones and the per-case reference."""
        ...

    def decode_state(self, value: Tensor, *, reference: Tensor | None = None) -> Tensor:
        """Return physical states from standardized ones and the per-case reference."""
        ...


@dataclass(frozen=True, slots=True)
class LinearWarmup:
    """
    Define an immutable deterministic zero-to-target epoch schedule.

    Parameters
    ----------
    target : float
        Non-negative terminal component weight.
    epochs : int
        Non-negative number of zero-based epoch positions needed to reach it.

    Raises
    ------
    ValueError
        If either schedule value is negative.

    """

    target: float
    epochs: int

    def __post_init__(self) -> None:
        """Validate non-negative schedule values."""
        if self.target < 0:
            msg = f"Warmup target must be non-negative, got {self.target}."
            raise ValueError(msg)
        if self.epochs < 0:
            msg = f"Warmup epochs must be non-negative, got {self.epochs}."
            raise ValueError(msg)

    def fraction(self, epoch: int) -> float:
        """
        Return the applied zero-to-one fraction at a zero-based epoch position.

        A zero-length schedule returns ``1.0`` immediately. Active schedules
        start at zero and clamp at one. Negative epoch positions raise
        ``ValueError``.
        """
        if epoch < 0:
            msg = f"Warmup epoch must be non-negative, got {epoch}."
            raise ValueError(msg)
        if self.epochs == 0:
            return 1.0
        return min(float(epoch) / float(self.epochs), 1.0)

    def value(self, epoch: int) -> float:
        """
        Return the scheduled weight for a zero-based epoch position.

        Epoch zero has zero physics weight when warmup is active. ``epochs``
        and later use the complete target. A zero-length warmup uses the target
        immediately.
        """
        return self.target * self.fraction(epoch)


class SemanticComposedLoss(nn.Module):
    """
    Compose one supervised or physics-informed semantic training objective.

    Named returned contributions are already weighted, so ``total`` is exactly
    the sum of ``data``, ``momentum``, ``continuity``, and ``boundary``. The
    physics terms are dimensionless domain diagnostics, each with its own
    schedule. Disabled physics components remain present as scalar zeros,
    giving supervised and physics-informed training one stable interface.

    Parameters
    ----------
    data_loss : StateErrorLoss
        Group-weighted state loss of Train-scaled fields.
    data_weight : float
        Explicit supervised contribution weight.
    physics_enabled : bool
        Whether to evaluate domain physics.
    physics_kind : str
        Task-owned domain physics identifier.
    input_fields, output_fields : tuple[str, ...]
        Exact task field declarations used for name-based domain binding.
    boundary : str
        Task-owned boundary formulation identifier.
    derivatives : domain.physics.derivatives.DerivativeOperator
        Explicit numerical derivative backend.
    momentum_weight, continuity_weight, boundary_weight : LinearWarmup
        Deterministic schedules of the three physics terms.
    interior_crop : int
        Interior crop applied by the domain momentum reduction.

    """

    def __init__(
        self,
        *,
        data_loss: StateErrorLoss,
        data_weight: float,
        physics_enabled: bool,
        physics_kind: str,
        input_fields: tuple[str, ...],
        output_fields: tuple[str, ...],
        boundary: str,
        derivatives: domain.physics.derivatives.DerivativeOperator,
        momentum_weight: LinearWarmup,
        continuity_weight: LinearWarmup,
        boundary_weight: LinearWarmup,
        interior_crop: int,
    ) -> None:
        """
        Validate composition inputs, resolve physics, and register epoch state.

        The zero-based warmup position becomes persistent module state, whereas
        the Train scaling remains runtime-only.
        """
        super().__init__()
        if data_weight < 0:
            msg = f"data_weight must be non-negative, got {data_weight}."
            raise ValueError(msg)
        if interior_crop < 0:
            msg = f"interior_crop must be non-negative, got {interior_crop}."
            raise ValueError(msg)
        if boundary != domain.physics.contracts.PRESSURE_BOUNDARY_KIND:
            msg = f"Unknown pressure boundary identifier {boundary!r}. Expected {domain.physics.contracts.PRESSURE_BOUNDARY_KIND!r}."
            raise ValueError(msg)
        self.data_loss = data_loss
        self.data_weight = float(data_weight)
        self.physics_enabled = bool(physics_enabled)
        self.physics_kind = physics_kind
        self.input_fields = tuple(input_fields)
        self.output_fields = tuple(output_fields)
        self.boundary = boundary
        self.derivatives = derivatives
        self.schedules = {"momentum": momentum_weight, "continuity": continuity_weight, "boundary": boundary_weight}
        self.interior_crop = int(interior_crop)
        self._physics_evaluator = domain.physics.brinkman.resolve_physics_evaluator(physics_kind)
        self.scaling: PhysicalScaling | None = None
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long), persistent=True)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return the exact ordered named loss-component interface."""
        terms = self.data_loss.terms
        shares = [field for term in terms if len(term.fields) > 1 for field in term.fields]
        return ("total", "data", *(f"data_{term.id}" for term in terms), *(f"data_{field}" for field in shares), *PHYSICS_TERMS)

    def set_scaling(self, scaling: PhysicalScaling) -> None:
        """
        Attach the run's Train scaling: it weighs the velocity vector in the data term and rebuilds the physical views of the physics terms.

        Parameters
        ----------
        scaling : PhysicalScaling
            Admitted Train scaling of the run on the training device.

        """
        self.data_loss.set_scaling(scaling)
        self.scaling = scaling

    def set_epoch(self, epoch: int) -> None:
        """
        Mutate the persistent zero-based warmup position.

        The scalar buffer is checkpointed with the loss module. Booleans and
        negative or non-integer positions raise ``ValueError``.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            msg = f"epoch must be a non-negative integer, got {epoch!r}."
            raise ValueError(msg)
        self.current_epoch.fill_(epoch)

    def _position(self, epoch: int | None) -> int:
        """Return the explicit or checkpointed zero-based warmup position."""
        return int(self.current_epoch.item()) if epoch is None else epoch

    def component_weights(self, *, epoch: int | None = None) -> dict[str, float]:
        """
        Return explicit weights for every named non-total component.

        ``epoch`` overrides the checkpointed warmup position for this query.
        Disabled physics keeps its stable component keys with zero weights.
        """
        position = self._position(epoch)
        weights = {"data": self.data_weight}
        for term in PHYSICS_TERMS:
            weights[term] = self.schedules[term].value(position) if self.physics_enabled else 0.0
        return weights

    def telemetry_state(self, *, epoch: int | None = None) -> dict[str, float]:
        """
        Return applied physics weights and warmup fractions for one epoch.

        Supervised configurations return no synthetic physics telemetry.
        """
        if not self.physics_enabled:
            return {}
        position = self._position(epoch)
        telemetry: dict[str, float] = {}
        for term in PHYSICS_TERMS:
            telemetry[f"weight_{term}"] = self.schedules[term].value(position)
            telemetry[f"warmup_{term}_fraction"] = self.schedules[term].fraction(position)
        return telemetry

    def compute_physics_diagnostics(
        self,
        pred: Tensor,
        *,
        x: Tensor,
        reference: Tensor | None = None,
    ) -> domain.physics.brinkman.BrinkmanDiagnostics:
        """
        Evaluate domain-owned physics on the physical views of normalized model tensors.

        Parameters
        ----------
        pred : torch.Tensor
            Normalized model outputs.
        x : torch.Tensor
            Normalized model inputs.
        reference : torch.Tensor or None, optional
            Per-case reference ``[B]`` the batch was encoded with; the steady scaling requires it.

        Returns
        -------
        domain.physics.brinkman.BrinkmanDiagnostics
            Reusable full-field and dimensionless scalar diagnostics.

        """
        if self.scaling is None:
            msg = "Physics diagnostics require the run's Train scaling."
            raise RuntimeError(msg)
        inputs_physical = self.scaling.decode_inputs(x, reference=reference)
        outputs_physical = self.scaling.decode_state(pred, reference=reference)
        return self._physics_evaluator(
            inputs_physical,
            outputs_physical,
            input_fields=self.input_fields,
            output_fields=self.output_fields,
            derivatives=self.derivatives,
            boundary=self.boundary,
            interior_crop=self.interior_crop,
        )

    def unit_components(
        self,
        pred: Tensor,
        *,
        x: Tensor | None,
        y: Tensor,
        epoch: int | None = None,
        reference: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute named weighted loss components for every case of a batch.

        Parameters
        ----------
        pred : torch.Tensor
            Normalized prediction tensor ``[U, C, Y, X]``, one case per unit.
        x : torch.Tensor or None
            Normalized task inputs, required only when physics is enabled.
        y : torch.Tensor
            Normalized supervised target tensor.
        reference : torch.Tensor or None, optional
            Per-case reference ``[U]`` of the batch's encoding, used only to decode for the physics terms.
        epoch : int or None, optional
            Explicit warmup position. Defaults to ``current_epoch``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``total``, ``data``, the informational data parts (``data_<term id>`` and the field shares of vector terms, each
            weighted by the data weight and not counted again in ``total``), ``momentum``, ``continuity``, and ``boundary``
            contributions ``[U]``. The data term is each case's own value; a
            physics term is the batch mean given to every case, so with equal case
            weights its sum over the cases equals the sum of the per-case values.

        """
        if pred.shape != y.shape:
            msg = f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(y.shape)}."
            raise ValueError(msg)
        weights = self.component_weights(epoch=epoch)
        data_terms, data_shares = self.data_loss.unit_contributions_with_shares(pred, y)
        components = {"data": weights["data"] * torch.stack(list(data_terms.values())).sum(dim=0)}
        # Informational parts of the data loss: they are not added to the total.
        components |= {f"data_{name}": weights["data"] * value for name, value in data_terms.items()}
        components |= {f"data_{field}": weights["data"] * share for shares in data_shares.values() for field, share in shares.items()}
        units = pred.shape[0]
        terms = dict.fromkeys(PHYSICS_TERMS, pred.new_zeros(units))
        if self.physics_enabled:
            if x is None:
                msg = "Physics-informed loss requires the normalized input tensor x."
                raise ValueError(msg)
            diagnostics = self.compute_physics_diagnostics(pred, x=x, reference=reference)
            terms = {
                "momentum": diagnostics.momentum_mse.expand(units),
                "continuity": diagnostics.continuity_mse.expand(units),
                "boundary": diagnostics.boundary_mse.expand(units),
            }
        for term in PHYSICS_TERMS:
            components[term] = weights[term] * terms[term]
        total = components["data"] + components["momentum"] + components["continuity"] + components["boundary"]
        return {"total": total, **components}

    def forward(
        self,
        pred: Tensor,
        y: Tensor | None = None,
        *,
        x: Tensor | None = None,
        epoch: int | None = None,
        reference: Tensor | None = None,
    ) -> Tensor:
        """Return the scalar total averaged over the batch; ``y`` is mandatory and physics also needs ``x`` and ``reference``."""
        if y is None:
            msg = "SemanticComposedLoss requires a target tensor y."
            raise ValueError(msg)
        return self.unit_components(pred, x=x, y=y, epoch=epoch, reference=reference)["total"].mean()
