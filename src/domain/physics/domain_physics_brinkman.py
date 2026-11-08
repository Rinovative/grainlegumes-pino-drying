"""Provide reusable steady Darcy-Brinkman equations and dimensionless diagnostics.

Responsibilities:
  - Compute Darcy-Brinkman momentum residuals with the stationary COMSOL stress
  - Compute the cross-section volume-flux balance implied by div(u) = 0
  - Derive per-sample characteristic scales from task inputs only
  - Bind the steady-flow task fields outside generic numerical kernels
  - Return full-grid fields and dimensionless scalar loss terms

Design principles:
  - Numerical kernels accept named physical quantities, never channel indices
  - Scales come from prescribed inputs, never from predictions or targets
  - Semantic physics and pressure-boundary identifiers fail closed

This module does NOT:
  - Normalize model tensors or choose physics-loss weights and warmup schedules
  - Own derivative discretization, checkpointing, W&B logging, or Optuna policy
  - Infer task channel order without explicit field-name declarations

Notes:
  Scales per sample, with zero outlet gauge pressure and flow from y-min to y-max:
    dp = mean prescribed inlet pressure, G = dp / L_y, U = K_g G / mu, Q = U L_x,
    where K_g is the geometric mean of sqrt(Kxx Kyy) over the grid.
  Momentum residual, as in COMSOL's Brinkman interface for incompressible flow:
    R = -grad(p) + div(tau) - mu K^-1 u,
    tau = (mu/eps)(grad u + grad u^T - (2/3) div(u) I).
  Dimensionless terms:
    momentum: mean over the cropped interior of (Rx^2 + Ry^2) / G^2
    continuity: mean over rows of ((q(y) - mean_y q) / Q)^2 with q(y) = int v dx
    boundary: mean_inlet((p - p_in_bc)^2) / dp^2 + (mean_outlet p)^2 / dp^2
  The flux balance replaces pointwise divergence because the COMSOL FEM solution,
  interpolated onto the regular grid, is not pointwise divergence-free there.
  It is the solved mass balance integrated over a strip when the side walls carry
  no flux.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from . import domain_physics_contracts as physics_contracts
from .domain_physics_boundary import PressureBoundaryResiduals, pressure_boundary_residuals
from .domain_physics_derivatives import DerivativeOperator, crop_interior, infer_uniform_spacing

# COMSOL "Air" eta(T) evaluated at the stationary template's T_flow_ref = 300.65 K
# (27.5 degC); the template takes viscosity from the material at that temperature.
AIR_DYNAMIC_VISCOSITY = 1.85075e-5
# The stationary COMSOL template's stress K = (mu/eps)(grad u + grad u^T - (2/3) div(u) I).
STRESS_TRACE_COEFFICIENT = 2.0 / 3.0
POROSITY_FLOOR = 1e-6
PERMEABILITY_SCALE_FLOOR = 1e-30
PERMEABILITY_DETERMINANT_FLOOR = 1e-4
PERMEABILITY_CROSS_RATIO_CLIP = 0.999
_MIN_TASK_TENSOR_RANK = 3
_SPATIAL_AXES = (-2, -1)


@dataclass(frozen=True, slots=True)
class MomentumResiduals:
    """
    Hold physical x- and y-momentum residual fields.

    ``x`` and ``y`` retain the input tensor's batch/spatial shape and have units
    of pressure gradient, Pa/m. No crop or scalar reduction is implied.

    Attributes
    ----------
    x, y : torch.Tensor
        Full-grid x- and y-momentum equation residuals in Pa/m.

    """

    x: Tensor
    y: Tensor


@dataclass(frozen=True, slots=True)
class BrinkmanScales:
    """
    Hold per-sample characteristic scales derived from task inputs only.

    Every attribute is detached and has shape ``[batch]``.

    Attributes
    ----------
    pressure : torch.Tensor
        Mean prescribed inlet pressure above the zero outlet gauge, in Pa.
    pressure_gradient : torch.Tensor
        ``pressure / L_y`` in Pa/m.
    velocity : torch.Tensor
        Darcy velocity scale ``K_g * pressure_gradient / mu`` in m/s.
    flux : torch.Tensor
        Cross-section volume-flux scale ``velocity * L_x`` in m²/s.

    """

    pressure: Tensor
    pressure_gradient: Tensor
    velocity: Tensor
    flux: Tensor


@dataclass(frozen=True, slots=True)
class BrinkmanDiagnostics:
    """
    Hold full residual fields and dimensionless scalar loss terms.

    Attributes
    ----------
    momentum : MomentumResiduals
        Full-grid physical momentum residuals in Pa/m.
    flux_balance : torch.Tensor
        Dimensionless per-row deviation ``(q(y) - mean_y q) / Q`` with shape
        ``[batch, y]``.
    boundary : PressureBoundaryResiduals
        Physical pressure-boundary values and per-sample reductions.
    scales : BrinkmanScales
        Input-derived scales used by every dimensionless term.
    momentum_mse : torch.Tensor
        Mean of ``(Rx² + Ry²) / G²`` over the cropped interior.
    continuity_mse : torch.Tensor
        Mean square of ``flux_balance`` over rows and samples.
    pressure_inlet_mse, pressure_outlet_mean_square : torch.Tensor
        Per-sample inlet mismatch and squared outlet gauge, each divided by
        ``dp²`` before the batch mean.
    interior_crop : int
        Cells removed from every spatial edge for the momentum term.

    """

    momentum: MomentumResiduals
    flux_balance: Tensor
    boundary: PressureBoundaryResiduals
    scales: BrinkmanScales
    momentum_mse: Tensor
    continuity_mse: Tensor
    pressure_inlet_mse: Tensor
    pressure_outlet_mean_square: Tensor
    interior_crop: int

    @property
    def boundary_mse(self) -> Tensor:
        """Return the dimensionless inlet-plus-outlet pressure term."""
        return self.pressure_inlet_mse + self.pressure_outlet_mean_square

    def as_dict(self) -> dict[str, Tensor]:
        """
        Return dimensionless residual fields and scalar diagnostics.

        Returns
        -------
        dict[str, torch.Tensor]
            ``Rx`` and ``Ry`` are full-grid momentum residuals divided by the
            per-sample pressure gradient, with a singleton channel axis.
            ``flux_balance`` is the per-row flux deviation. All scalars are
            dimensionless and carry no loss weights.

        """
        gradient = self.scales.pressure_gradient[:, None, None, None]
        return {
            "Rx": self.momentum.x.unsqueeze(1) / gradient,
            "Ry": self.momentum.y.unsqueeze(1) / gradient,
            "flux_balance": self.flux_balance,
            "momentum_residual_mse": self.momentum_mse,
            "flux_balance_mse": self.continuity_mse,
            "pressure_boundary_mse": self.boundary_mse,
            "pressure_inlet_mse": self.pressure_inlet_mse,
            "pressure_outlet_mean_square": self.pressure_outlet_mean_square,
        }


def brinkman_scales(
    prescribed_pressure: Tensor,
    inlet_mask: Tensor,
    permeability_xx: Tensor,
    permeability_yy: Tensor,
    x_coordinate: Tensor,
    y_coordinate: Tensor,
    *,
    viscosity: float = AIR_DYNAMIC_VISCOSITY,
) -> BrinkmanScales:
    """
    Derive per-sample Darcy scales from prescribed inputs.

    Parameters
    ----------
    prescribed_pressure : torch.Tensor
        Physical inlet-pressure field in Pa, shaped ``[batch, y, x]``.
    inlet_mask : torch.Tensor
        Boolean y-min inlet mask with the same shape.
    permeability_xx, permeability_yy : torch.Tensor
        Positive diagonal permeability components in m².
    x_coordinate, y_coordinate : torch.Tensor
        Physical coordinate fields in m.
    viscosity : float, optional
        Dynamic viscosity in Pa s.

    Returns
    -------
    BrinkmanScales
        Detached per-sample scales.

    Raises
    ------
    ValueError
        If shapes differ, viscosity is not positive, or any sample has a
        non-positive or non-finite inlet pressure, extent, or velocity scale.

    """
    fields = (inlet_mask, permeability_xx, permeability_yy, x_coordinate, y_coordinate)
    if prescribed_pressure.ndim != _MIN_TASK_TENSOR_RANK or any(field.shape != prescribed_pressure.shape for field in fields):
        msg = "Brinkman scale inputs must share one [batch, y, x] shape."
        raise ValueError(msg)
    if viscosity <= 0:
        msg = f"viscosity must be positive, got {viscosity}."
        raise ValueError(msg)
    with torch.no_grad():
        inlet = inlet_mask.to(prescribed_pressure.dtype)
        pressure = (prescribed_pressure * inlet).sum(dim=_SPATIAL_AXES) / inlet.sum(dim=_SPATIAL_AXES)
        length_x = x_coordinate.amax(dim=_SPATIAL_AXES) - x_coordinate.amin(dim=_SPATIAL_AXES)
        length_y = y_coordinate.amax(dim=_SPATIAL_AXES) - y_coordinate.amin(dim=_SPATIAL_AXES)
        log_permeability = 0.5 * (torch.log(permeability_xx) + torch.log(permeability_yy))
        geometric_permeability = torch.exp(log_permeability.mean(dim=_SPATIAL_AXES))
        pressure_gradient = pressure / length_y
        velocity = geometric_permeability * pressure_gradient / viscosity
        flux = velocity * length_x
    for name, value in (("inlet pressure", pressure), ("x extent", length_x), ("y extent", length_y), ("velocity scale", velocity)):
        if not bool(torch.isfinite(value).all().item()) or not bool((value > 0).all().item()):
            msg = f"Brinkman {name} must be finite and positive for every sample."
            raise ValueError(msg)
    return BrinkmanScales(pressure=pressure, pressure_gradient=pressure_gradient, velocity=velocity, flux=flux)


def cross_section_flux_balance(velocity_y: Tensor, x_coordinate: Tensor, flux_scale: Tensor) -> Tensor:
    """
    Return the dimensionless deviation of every cross-section volume flux.

    Parameters
    ----------
    velocity_y : torch.Tensor
        Physical superficial y-velocity in m/s, shaped ``[batch, y, x]``.
    x_coordinate : torch.Tensor
        Physical x-coordinate field in m with the same shape.
    flux_scale : torch.Tensor
        Positive per-sample flux scale in m²/s, shaped ``[batch]``.

    Returns
    -------
    torch.Tensor
        ``(q(y) - mean_y q) / flux_scale`` with ``q(y)`` the trapezoidal
        integral of ``v`` across x, shaped ``[batch, y]``.

    Raises
    ------
    ValueError
        If field shapes differ or the flux scale does not match the batch.

    """
    if velocity_y.ndim != _MIN_TASK_TENSOR_RANK or velocity_y.shape != x_coordinate.shape:
        msg = "Flux balance requires matching [batch, y, x] velocity and x-coordinate fields."
        raise ValueError(msg)
    if flux_scale.shape != velocity_y.shape[:1]:
        msg = "Flux scale must hold exactly one value per sample."
        raise ValueError(msg)
    flux = torch.trapezoid(velocity_y, x_coordinate, dim=-1)
    return (flux - flux.mean(dim=-1, keepdim=True)) / flux_scale[:, None]


def _inverse_permeability_components(
    permeability_xx: Tensor,
    permeability_xy_ratio: Tensor,
    permeability_yy: Tensor,
    *,
    permeability_scale_floor: float,
    determinant_floor: float,
    cross_ratio_clip: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Invert the symmetric 2D permeability representation with explicit floors.

    Diagonal components are physical m² values. The stored cross channel is the
    dimensionless ratio ``Kxy / sqrt(Kxx*Kyy)``. It is clipped before inversion.
    The geometric scale and normalized determinant are floored independently,
    yielding ``(K^-1_xx, K^-1_xy, K^-1_yy)`` in 1/m² without materializing a
    matrix tensor.

    Raises
    ------
    ValueError
        If the three permeability component shapes differ.

    """
    if permeability_xx.shape != permeability_yy.shape or permeability_xx.shape != permeability_xy_ratio.shape:
        msg = "Permeability component shapes must match."
        raise ValueError(msg)
    scale = torch.sqrt((permeability_xx * permeability_yy).clamp_min(permeability_scale_floor))
    normalized_xx = permeability_xx / scale
    normalized_yy = permeability_yy / scale
    normalized_xy = permeability_xy_ratio.clamp(-cross_ratio_clip, cross_ratio_clip)
    determinant = (normalized_xx * normalized_yy - normalized_xy.square()).clamp_min(determinant_floor)
    inverse_xx = normalized_yy / determinant / scale
    inverse_xy = -normalized_xy / determinant / scale
    inverse_yy = normalized_xx / determinant / scale
    return inverse_xx, inverse_xy, inverse_yy


def brinkman_momentum_residuals(
    pressure: Tensor,
    velocity_x: Tensor,
    velocity_y: Tensor,
    porosity: Tensor,
    permeability_xx: Tensor,
    permeability_xy_ratio: Tensor,
    permeability_yy: Tensor,
    derivatives: DerivativeOperator,
    spacing_x: float | Tensor,
    spacing_y: float | Tensor,
    *,
    viscosity: float = AIR_DYNAMIC_VISCOSITY,
    porosity_floor: float = POROSITY_FLOOR,
    permeability_scale_floor: float = PERMEABILITY_SCALE_FLOOR,
    determinant_floor: float = PERMEABILITY_DETERMINANT_FLOOR,
    cross_ratio_clip: float = PERMEABILITY_CROSS_RATIO_CLIP,
) -> MomentumResiduals:
    """
    Compute the steady two-dimensional Darcy-Brinkman momentum residual.

    The implemented convention is ``-grad(p) + div(tau) - mu K^-1 u`` with
    ``tau = (mu/eps) * (grad(u) + grad(u)^T - (2/3) div(u) I)``, the stress
    of COMSOL's Brinkman interface, which keeps the trace term also for
    incompressible flow.

    Parameters
    ----------
    pressure, velocity_x, velocity_y : torch.Tensor
        Physical pressure in Pa and velocity components in m/s with identical
        sample/spatial shapes, normally ``[batch, y, x]``.
    porosity : torch.Tensor
        Dimensionless porosity field.
    permeability_xx, permeability_yy : torch.Tensor
        Positive physical diagonal permeability components in m^2.
    permeability_xy_ratio : torch.Tensor
        Dimensionless cross component relative to the geometric mean.
    derivatives : DerivativeOperator
        Explicit numerical derivative backend.
    spacing_x, spacing_y : float or torch.Tensor
        Positive physical grid spacing.
    viscosity : float, optional
        Dynamic viscosity in Pa s.
    porosity_floor : float, optional
        Lower clamp applied to dimensionless porosity. The canonical default is
        positive.
    permeability_scale_floor : float, optional
        Lower clamp for the permeability geometric mean in m².
    determinant_floor : float, optional
        Lower clamp for the normalized permeability determinant.
    cross_ratio_clip : float, optional
        Symmetric absolute clamp for the dimensionless cross-permeability ratio.

    Returns
    -------
    MomentumResiduals
        Full-grid x- and y-momentum residual fields in Pa/m.

    Raises
    ------
    TypeError
        If the derivative backend rejects a field or spacing dtype.
    ValueError
        If physical field shapes differ, viscosity is not positive, or the
        derivative backend rejects field geometry or spacing.

    Notes
    -----
    ``u`` is the superficial (Darcy) velocity. The trace term vanishes where
    ``div(u) = 0``; there is no inertial or Forchheimer contribution.
    Stabilization clamps affect porosity and permeability inversion only.
    Inputs are not mutated.

    """
    fields = (
        pressure,
        velocity_x,
        velocity_y,
        porosity,
        permeability_xx,
        permeability_xy_ratio,
        permeability_yy,
    )
    if any(field.shape != pressure.shape for field in fields[1:]):
        msg = "All Brinkman physical fields must have identical shapes."
        raise ValueError(msg)
    if viscosity <= 0:
        msg = f"viscosity must be positive, got {viscosity}."
        raise ValueError(msg)

    safe_porosity = porosity.clamp_min(porosity_floor)
    pressure_x, pressure_y = derivatives.gradient(pressure, spacing_x, spacing_y)
    velocity_x_x, velocity_x_y = derivatives.gradient(velocity_x, spacing_x, spacing_y)
    velocity_y_x, velocity_y_y = derivatives.gradient(velocity_y, spacing_x, spacing_y)

    coefficient = viscosity / safe_porosity
    trace = STRESS_TRACE_COEFFICIENT * (velocity_x_x + velocity_y_y)
    stress_xx = coefficient * (2.0 * velocity_x_x - trace)
    stress_yy = coefficient * (2.0 * velocity_y_y - trace)
    stress_xy = coefficient * (velocity_x_y + velocity_y_x)
    stress_divergence_x = derivatives.divergence(stress_xx, stress_xy, spacing_x, spacing_y)
    stress_divergence_y = derivatives.divergence(stress_xy, stress_yy, spacing_x, spacing_y)

    inverse_xx, inverse_xy, inverse_yy = _inverse_permeability_components(
        permeability_xx,
        permeability_xy_ratio,
        permeability_yy,
        permeability_scale_floor=permeability_scale_floor,
        determinant_floor=determinant_floor,
        cross_ratio_clip=cross_ratio_clip,
    )
    drag_x = viscosity * (inverse_xx * velocity_x + inverse_xy * velocity_y)
    drag_y = viscosity * (inverse_xy * velocity_x + inverse_yy * velocity_y)
    return MomentumResiduals(
        x=-pressure_x + stress_divergence_x - drag_x,
        y=-pressure_y + stress_divergence_y - drag_y,
    )


def _field_mapping(
    tensor: Tensor,
    fields: Sequence[str],
    required: tuple[str, ...],
    *,
    label: str,
) -> dict[str, Tensor]:
    """
    Bind required task names to channel views without reordering the source.

    ``tensor`` must expose batch and channel as its first two axes. The complete
    declared ``fields`` sequence must match the channel count and be duplicate
    free. Every required steady-flow role is returned as ``tensor[:, index]``,
    preserving batch/spatial storage as a view.

    Raises
    ------
    ValueError
        If tensor rank/channel count, field uniqueness, or required membership
        violates the task-binding contract.

    """
    if tensor.ndim < _MIN_TASK_TENSOR_RANK:
        msg = f"{label} tensor must have batch, channel, and spatial axes."
        raise ValueError(msg)
    if tensor.shape[1] != len(fields):
        msg = f"{label} tensor has {tensor.shape[1]} channels but {len(fields)} field names."
        raise ValueError(msg)
    if len(fields) != len(set(fields)):
        msg = f"{label} field declaration contains duplicate names: {list(fields)}."
        raise ValueError(msg)
    missing = [name for name in required if name not in fields]
    if missing:
        msg = f"{label} fields are missing required steady-flow roles: {missing}."
        raise ValueError(msg)
    indices = {name: index for index, name in enumerate(fields)}
    return {name: tensor[:, indices[name]] for name in required}


def evaluate_steady_2d_brinkman(
    inputs: Tensor,
    outputs: Tensor,
    *,
    input_fields: Sequence[str],
    output_fields: Sequence[str],
    derivatives: DerivativeOperator,
    boundary: str,
    interior_crop: int = 0,
) -> BrinkmanDiagnostics:
    """
    Evaluate task-bound steady-flow residuals on physical tensor views.

    Parameters
    ----------
    inputs, outputs : torch.Tensor
        Task-ordered BCHW tensors. Output channels are physical ``p``/``u``/``v``
        values. Input ``x``, ``y``, ``eps``, and ``p_in_bc`` retain physical/stored
        values. Diagonal permeability channels are log10 ratios to 1 m² and are
        exponentiated, while ``kxy`` is the dimensionless geometric-mean ratio.
    input_fields, output_fields : Sequence[str]
        Exact task-owned field declarations used for name-based binding.
    derivatives : DerivativeOperator
        Explicit physical or spectral derivative backend.
    boundary : str
        Semantic pressure boundary formulation.
    interior_crop : int, optional
        Cells removed from every spatial edge before the momentum reduction.
        The flux balance and the pressure boundary always use the full grid.

    Returns
    -------
    BrinkmanDiagnostics
        Full-grid fields, input-derived scales, and dimensionless loss terms.

    Raises
    ------
    TypeError
        If tensor/coordinate dtypes violate derivative or boundary contracts.
    ValueError
        If semantic identifiers, channel declarations, BCHW geometry, Cartesian
        spacing, crop, scales, or physical field shapes violate the contract.

    """
    if boundary != physics_contracts.PRESSURE_BOUNDARY_KIND:
        msg = f"Unknown pressure boundary identifier {boundary!r}. Expected {physics_contracts.PRESSURE_BOUNDARY_KIND!r}."
        raise ValueError(msg)
    input_values = _field_mapping(
        inputs,
        input_fields,
        ("x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc"),
        label="input",
    )
    output_values = _field_mapping(
        outputs,
        output_fields,
        ("p", "u", "v"),
        label="output",
    )
    spacing_x, spacing_y = infer_uniform_spacing(
        input_values["x"],
        input_values["y"],
        axes=_SPATIAL_AXES,
    )
    permeability_xx = torch.pow(10.0, input_values["Kxx"])
    permeability_yy = torch.pow(10.0, input_values["Kyy"])
    momentum = brinkman_momentum_residuals(
        output_values["p"],
        output_values["u"],
        output_values["v"],
        input_values["eps_bed"],
        permeability_xx,
        input_values["Kxy"],
        permeability_yy,
        derivatives,
        spacing_x,
        spacing_y,
    )
    pressure_boundary = pressure_boundary_residuals(
        output_values["p"],
        input_values["p_in_bc"],
        input_values["y"],
        spacing_y,
        spatial_axes=_SPATIAL_AXES,
    )
    scales = brinkman_scales(
        input_values["p_in_bc"],
        pressure_boundary.masks.inlet,
        permeability_xx,
        permeability_yy,
        input_values["x"],
        input_values["y"],
    )
    flux_balance = cross_section_flux_balance(output_values["v"], input_values["x"], scales.flux)
    gradient = scales.pressure_gradient[:, None, None]
    momentum_x_interior = crop_interior(momentum.x / gradient, interior_crop, axes=_SPATIAL_AXES)
    momentum_y_interior = crop_interior(momentum.y / gradient, interior_crop, axes=_SPATIAL_AXES)
    return BrinkmanDiagnostics(
        momentum=momentum,
        flux_balance=flux_balance,
        boundary=pressure_boundary,
        scales=scales,
        momentum_mse=(momentum_x_interior.square() + momentum_y_interior.square()).mean(),
        continuity_mse=flux_balance.square().mean(),
        pressure_inlet_mse=(pressure_boundary.inlet_sample_mse / scales.pressure.square()).mean(),
        pressure_outlet_mean_square=(pressure_boundary.outlet_sample_mean / scales.pressure).square().mean(),
        interior_crop=interior_crop,
    )


PhysicsEvaluator = Callable[..., BrinkmanDiagnostics]
_PHYSICS_EVALUATORS = MappingProxyType({physics_contracts.STEADY_BRINKMAN_KIND: evaluate_steady_2d_brinkman})


def resolve_physics_evaluator(kind: str) -> PhysicsEvaluator:
    """
    Resolve the exact task-selected equation-set evaluator.

    Parameters
    ----------
    kind : str
        Canonical domain physics identifier.

    Returns
    -------
    PhysicsEvaluator
        Registered evaluator callable. Resolution performs no computation.

    Raises
    ------
    ValueError
        If ``kind`` has no registered evaluator.

    """
    try:
        return _PHYSICS_EVALUATORS[kind]
    except KeyError as error:
        available = ", ".join(sorted(_PHYSICS_EVALUATORS))
        msg = f"Unknown domain physics identifier {kind!r}. Available physics: {available}."
        raise ValueError(msg) from error
