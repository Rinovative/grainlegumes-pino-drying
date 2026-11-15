"""Compute the group-weighted state loss of both tasks: MSE, or absolute H1, on Train-scaled states.

Responsibilities:
  - Weight per-field mean squared errors of Train-scaled states with the task-owned error terms
  - Combine the components of a vector term with their squared Train scales, as the selection objective does
  - Add the optional absolute H1 term from squared non-periodic spatial gradient errors in metres
  - Expose the weighted contribution of every term, per training unit and averaged over units

Design principles:
  - One definition serves steady states and transient reconstructed states; the leading axis is the training unit
  - A unit's value averages over all its other axes (its transitions and grid), so per-unit weights act on units
  - Errors are differences of Train-scaled states, so the state centring cancels
  - The quadrature is normalised to the spatial mean: every grid point weighs equally

This module does NOT:
  - Reconstruct, scale or fit states
  - Compose data weights or physics terms; the task compositions own them

Notes:
  For a term with fields F, Train scales s_f and Train-scaled errors e_f, the term value is
  sum_f s_f^2 <e_f^2> / sum_f s_f^2 (MSE), plus l^2 sum_f s_f^2 <|grad e_f|^2> / sum_f s_f^2 (H1),
  with l = 1 m and <.> the mean over every element of the field within one unit; a unit's loss is the weighted
  sum of its terms, and the mean over the equally sized units of one tensor is the mean over all its elements.
  Spatial derivatives are central inside and one-sided at the boundary, per metre of the physical grid.

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.tasks.domain_task_spec import ErrorTermSpec

StateLossKind = Literal["mse", "h1"]
H1_LENGTH_SCALE_M: Final = 1.0
_SPATIAL_AXES: Final = 2
_MIN_STATE_RANK: Final = 4


class _ScaledGroup(Protocol):
    """Expose one scaling group's channel names and scales."""

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ordered channel names."""
        ...

    @property
    def scale(self) -> torch.Tensor:
        """Return the operational Train scale of every channel."""
        ...


class StateScaling(Protocol):
    """Expose the Train state scale and its channel names."""

    def group(self, name: str) -> _ScaledGroup:
        """Return one named scaling group."""
        ...


class StateErrorLoss(nn.Module):
    """
    Compute the group-weighted MSE or absolute H1 error of Train-scaled states.

    Parameters
    ----------
    kind : {"mse", "h1"}
        Mean squared error, or that plus the absolute squared gradient error.
    terms : Sequence[ErrorTermSpec]
        Task-owned weighted error terms.
    fields : Sequence[str]
        State channel names in tensor order.
    grid_spacing : Sequence[float] | None, optional
        Physical grid spacing in metres, rows then columns; required for ``"h1"``.

    Raises
    ------
    ValueError
        If the kind is unknown, a term names an unknown field, or ``"h1"`` lacks a valid spacing.

    """

    def __init__(
        self,
        *,
        kind: StateLossKind,
        terms: Sequence[ErrorTermSpec],
        fields: Sequence[str],
        grid_spacing: Sequence[float] | None = None,
    ) -> None:
        """Validate the loss semantics and fix the one-field term combinations."""
        super().__init__()
        if kind not in {"mse", "h1"}:
            msg = f"Unknown state loss {kind!r}. Expected 'mse' or 'h1'."
            raise ValueError(msg)
        self.kind = kind
        self.terms = tuple(terms)
        self.fields = tuple(fields)
        unknown = sorted({field for term in self.terms for field in term.fields} - set(self.fields))
        if not self.terms or unknown:
            msg = f"State loss terms must name known state fields {list(self.fields)}, got unknown {unknown}."
            raise ValueError(msg)
        self.grid_spacing: tuple[float, float] | None = None
        if kind == "h1":
            spacing = tuple(float(value) for value in grid_spacing or ())
            if len(spacing) != _SPATIAL_AXES or not all(math.isfinite(value) and value > 0.0 for value in spacing):
                msg = f"The H1 state loss requires two positive finite grid spacings in metres (rows, then columns), got {grid_spacing!r}."
                raise ValueError(msg)
            self.grid_spacing = (spacing[0], spacing[1])
        self._indices = tuple(tuple(self.fields.index(field) for field in term.fields) for term in self.terms)
        # A one-field term needs no scale; a vector term weighs its components by their squared Train scales (set_scaling).
        self._combinations: tuple[tuple[float, ...] | None, ...] = tuple((1.0,) if len(term.fields) == 1 else None for term in self.terms)

    def set_scaling(self, scaling: StateScaling) -> None:
        """
        Bind the run's Train state scales that combine the components of vector terms.

        Parameters
        ----------
        scaling : StateScaling
            Admitted Train scaling whose state channels equal ``fields``.

        Raises
        ------
        ValueError
            If the scaling's state channels differ from the loss fields.

        """
        state = scaling.group("state")
        if tuple(state.names) != self.fields:
            msg = f"Train scaling state channels {list(state.names)} differ from the loss fields {list(self.fields)}."
            raise ValueError(msg)
        squares = [float(value) ** 2 for value in state.scale.detach().double().reshape(-1).tolist()]
        self._combinations = tuple(tuple(squares[index] / sum(squares[i] for i in indices) for index in indices) for indices in self._indices)

    def _field_means(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return each unit's per-field mean ``[U, F]`` of the squared error, plus the squared gradient error for H1."""
        if prediction.shape != target.shape or prediction.ndim < _MIN_STATE_RANK or prediction.shape[-3] != len(self.fields):
            msg = (
                f"State loss tensors must share one [U, ..., {len(self.fields)}, Y, X] shape with a leading unit axis, "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
            raise ValueError(msg)
        error = prediction - target
        energy = error.square()
        if self.grid_spacing is not None:
            gradients = torch.gradient(error, spacing=self.grid_spacing, dim=(-2, -1))
            energy = energy + H1_LENGTH_SCALE_M**2 * (gradients[0].square() + gradients[1].square())
        return energy.movedim(-3, 1).reshape(energy.shape[0], len(self.fields), -1).mean(dim=2)

    def unit_term_contributions(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Return the weighted contribution of every error term for every unit.

        Parameters
        ----------
        prediction, target : torch.Tensor
            Train-scaled states ``[U, ..., C, Y, X]``: one leading unit axis, the state channels at axis ``-3``.

        Returns
        -------
        dict[str, torch.Tensor]
            Term id to its weight times its value ``[U]``, each unit averaged over all its other axes;
            a unit's loss is the sum over terms.

        Raises
        ------
        RuntimeError
            If a vector term is evaluated before :meth:`set_scaling`.

        """
        return self.unit_contributions_with_shares(prediction, target)[0]

    def unit_contributions_with_shares(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """
        Return :meth:`unit_term_contributions` plus the per-field shares of every vector term, from one set of field means.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]
            The term contributions ``[U]`` and, for each term over several fields, its field name to that field's weighted
            share ``[U]``; the shares of a term sum to its contribution. A one-field term has no separate share.

        Raises
        ------
        RuntimeError
            If a vector term is evaluated before :meth:`set_scaling`.

        """
        means = self._field_means(prediction, target)
        contributions: dict[str, torch.Tensor] = {}
        shares: dict[str, dict[str, torch.Tensor]] = {}
        for term, indices, combination in zip(self.terms, self._indices, self._combinations, strict=True):
            if combination is None:
                msg = f"Vector term {term.id!r} requires the Train scaling; call set_scaling first."
                raise RuntimeError(msg)
            parts = [factor * means[:, index] for factor, index in zip(combination, indices, strict=True)]
            contributions[term.id] = term.weight * sum(parts, means.new_zeros(means.shape[0]))
            if len(parts) > 1:
                shares[term.id] = {field: term.weight * part for field, part in zip(term.fields, parts, strict=True)}
        return contributions, shares

    def term_contributions(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every term's contribution averaged over the equally sized units of one tensor."""
        return {name: value.mean() for name, value in self.unit_term_contributions(prediction, target).items()}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the weighted sum of the term values over all units."""
        return torch.stack(list(self.term_contributions(prediction, target).values())).sum()
