"""Provide reusable spatial derivative operators for physical tensor fields.

Responsibilities:
  - Infer uniform-grid spacing from explicit coordinate tensors
  - Compute gradients and divergences through neuralop FiniteDiff (physical) and FourierDiff (spectral)
  - Fix spatial-axis order, Fourier-continuation parameters, and interior-crop semantics

Design principles:
  - Grid spacing and spatial axes are explicit at every numerical boundary
  - Cartesian-grid admission fails closed before an equation is evaluated
  - Library axis order is mapped to the project's (y, x) convention at each library call

This module does NOT:
  - Bind task field names, transform normalized data, or select loss weights
  - Impose equation-specific interior crops or scalar residual reductions
  - Infer nonuniform, curvilinear, or unstructured-grid derivatives
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from . import domain_physics_contracts as physics_contracts

SpatialAxes = tuple[int, int]
_MIN_SPATIAL_POINTS = 2
# Fourier-continuation contract: Legendre fit through 5 boundary points per side
# and 50 smooth continuation points per axis, passed explicitly to FourierDiff.
FOURIER_CONTINUATION_MATCHING_POINTS = 5
FOURIER_CONTINUATION_ADDITIONAL_POINTS = 50
# FiniteDiff one-sided edge stencils read four consecutive points.
_FINITE_DIFFERENCE_MIN_POINTS = 4
_SPATIAL_AXIS_COUNT = 2
_UNIFORM_TOLERANCE = 1e-5
_RECTILINEAR_EPSILON_FACTOR = 32.0
_UNIFORM_ROUNDOFF_FACTOR = 4.0


class DerivativeOperator(Protocol):
    """
    Define the reusable physical-field derivative interface used by equations.

    Implementations accept real floating tensors with explicit ``(y, x)`` axes
    and positive physical spacing. ``gradient`` returns ``(d/dx, d/dy)``.
    ``divergence`` returns ``d(field_x)/dx + d(field_y)/dy`` without cropping or
    scalar reduction.
    """

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(d/dx, d/dy)`` for one scalar field."""
        ...

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return ``d(field_x)/dx + d(field_y)/dy``."""
        ...


def normalize_spatial_axes(ndim: int, axes: SpatialAxes) -> SpatialAxes:
    """
    Normalize the declared ``(y, x)`` axes for a tensor rank.

    Negative indices are translated relative to ``ndim`` while order is
    preserved. Duplicate or out-of-range axes are rejected before tensor
    movement, differentiation, or cropping.

    Raises
    ------
    ValueError
        If the declaration does not resolve to two distinct in-range axes.

    """
    normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    if len(set(normalized)) != _SPATIAL_AXIS_COUNT or any(axis < 0 or axis >= ndim for axis in normalized):
        msg = f"Spatial axes {axes!r} are invalid for tensor rank {ndim}."
        raise ValueError(msg)
    return cast("SpatialAxes", normalized)


def _validate_field(field: Tensor, axes: SpatialAxes) -> SpatialAxes:
    """
    Validate a differentiable real field and return normalized spatial axes.

    The function does not copy data. Each declared spatial dimension must have
    at least two points; operators check their larger stencil minimum separately.

    Raises
    ------
    TypeError
        If ``field`` is not a float32 or float64 ``torch.Tensor``.
    ValueError
        If the axes are invalid or either spatial dimension is shorter than two.

    """
    if not isinstance(field, Tensor):
        msg = f"Derivative fields must be torch.Tensor instances, got {type(field).__name__}."
        raise TypeError(msg)
    if field.dtype not in {torch.float32, torch.float64}:
        msg = f"Derivative fields must be float32 or float64, got {field.dtype}."
        raise TypeError(msg)
    normalized = normalize_spatial_axes(field.ndim, axes)
    if any(field.shape[axis] < _MIN_SPATIAL_POINTS for axis in normalized):
        msg = f"Derivative spatial axes require at least two points, got shape {tuple(field.shape)}."
        raise ValueError(msg)
    return normalized


def _spacing_float(spacing: float | Tensor, *, reference: Tensor, label: str) -> float:
    """
    Return one validated physical spacing as the host scalar the library operators take.

    Conversion follows the reference dtype and device. Transferring the scalar
    to the host does not move or copy the differentiated field.

    Raises
    ------
    ValueError
        If conversion does not produce exactly one finite, strictly positive
        value.

    """
    value = torch.as_tensor(spacing, dtype=reference.dtype, device=reference.device)
    if value.numel() != 1:
        msg = f"{label} must be scalar, got shape {tuple(value.shape)}."
        raise ValueError(msg)
    host_value = float(value.detach().cpu().item())
    if not math.isfinite(host_value) or host_value <= 0.0:
        msg = f"{label} must be finite and positive, got {host_value}."
        raise ValueError(msg)
    return host_value


def _divergence(
    operator: DerivativeOperator,
    field_x: Tensor,
    field_y: Tensor,
    spacing_x: float | Tensor,
    spacing_y: float | Tensor,
) -> Tensor:
    """
    Return ``d(field_x)/dx + d(field_y)/dy`` through the operator's ``gradient``.

    Parameters
    ----------
    operator : DerivativeOperator
        Operator whose ``gradient`` differentiates each component.
    field_x, field_y : torch.Tensor
        Real floating vector components with identical shapes and arbitrary
        leading dimensions.
    spacing_x, spacing_y : float or torch.Tensor
        Finite positive scalar physical spacings shared by both components.

    Returns
    -------
    torch.Tensor
        Divergence with the component shape, dtype, and device. Units are
        component units per spacing unit when both components share units.

    Raises
    ------
    TypeError
        If either component is not a real floating tensor.
    ValueError
        If component shapes, axes, spatial lengths, or spacings are invalid.

    """
    if field_x.shape != field_y.shape:
        msg = f"Vector component shapes must match, got {tuple(field_x.shape)} and {tuple(field_y.shape)}."
        raise ValueError(msg)
    derivative_x, _ = operator.gradient(field_x, spacing_x, spacing_y)
    _, derivative_y = operator.gradient(field_y, spacing_x, spacing_y)
    return derivative_x + derivative_y


def infer_uniform_spacing(
    x_coordinate: Tensor,
    y_coordinate: Tensor,
    *,
    axes: SpatialAxes = (-2, -1),
) -> tuple[Tensor, Tensor]:
    """
    Infer positive spacing from finite, increasing, uniform Cartesian grids.

    Parameters
    ----------
    x_coordinate, y_coordinate : torch.Tensor
        Rectilinear coordinate fields with matching shapes.
    axes : tuple[int, int], optional
        ``(y_axis, x_axis)`` in the coordinate tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean scalar ``(dx, dy)`` tensors on the corresponding coordinate dtype
        and device.

    Raises
    ------
    TypeError
        If coordinates are not floating tensors.
    ValueError
        If shapes/axes differ, values are non-finite, the grid is not increasing
        rectilinear Cartesian data, or spacing is not uniform.

    Notes
    -----
    Cross-axis coordinate variation is admitted only at a small dtype-scaled
    roundoff tolerance. Along each varying axis, strict monotonicity remains
    mandatory. Uniformity admits a fixed relative deviation from mean spacing
    plus four source-dtype epsilons at the coordinate scale, which covers
    subtraction of independently rounded coordinate endpoints and mean-spacing
    reduction without treating materially nonuniform grids as Cartesian.

    """
    y_axis, x_axis = _validate_field(x_coordinate, axes)
    _validate_field(y_coordinate, axes)
    if x_coordinate.shape != y_coordinate.shape:
        msg = f"Coordinate shapes must match, got {tuple(x_coordinate.shape)} and {tuple(y_coordinate.shape)}."
        raise ValueError(msg)
    if not bool(torch.isfinite(x_coordinate).all().item()) or not bool(torch.isfinite(y_coordinate).all().item()):
        msg = "Coordinate grids must contain only finite values."
        raise ValueError(msg)

    x_cross_differences = torch.diff(x_coordinate, dim=y_axis)
    y_cross_differences = torch.diff(y_coordinate, dim=x_axis)
    epsilon = torch.finfo(x_coordinate.dtype).eps
    x_cross_tolerance = _RECTILINEAR_EPSILON_FACTOR * epsilon * x_coordinate.abs().amax().clamp_min(1.0)
    y_cross_tolerance = _RECTILINEAR_EPSILON_FACTOR * epsilon * y_coordinate.abs().amax().clamp_min(1.0)
    if bool((x_cross_differences.abs().amax() > x_cross_tolerance).item()):
        msg = "x-coordinate must be constant along the y-axis on a Cartesian grid."
        raise ValueError(msg)
    if bool((y_cross_differences.abs().amax() > y_cross_tolerance).item()):
        msg = "y-coordinate must be constant along the x-axis on a Cartesian grid."
        raise ValueError(msg)

    # Validate in float64 so the admission decision measures quantization in the
    # persisted source coordinates rather than adding float32 reduction error.
    # The source dtype still determines the permitted roundoff allowance.
    x_work = x_coordinate.to(torch.float64)
    y_work = y_coordinate.to(torch.float64)
    x_differences = torch.diff(x_work, dim=x_axis)
    y_differences = torch.diff(y_work, dim=y_axis)
    if not bool((x_differences > 0).all().item()):
        msg = "x-coordinate must be strictly increasing along the x-axis."
        raise ValueError(msg)
    if not bool((y_differences > 0).all().item()):
        msg = "y-coordinate must be strictly increasing along the y-axis."
        raise ValueError(msg)
    dx_work = x_differences.mean()
    dy_work = y_differences.mean()
    for label, coordinate, differences, mean in (
        ("x", x_coordinate, x_differences, dx_work),
        ("y", y_coordinate, y_differences, dy_work),
    ):
        source_epsilon = torch.finfo(coordinate.dtype).eps
        coordinate_scale = coordinate.detach().to(torch.float64).abs().amax().clamp_min(mean.abs())
        roundoff_allowance = _UNIFORM_ROUNDOFF_FACTOR * source_epsilon * coordinate_scale
        relative_allowance = _UNIFORM_TOLERANCE * mean.abs()
        maximum_deviation = (differences - mean).abs().amax()
        allowed_deviation = relative_allowance + roundoff_allowance
        if bool((maximum_deviation > allowed_deviation).item()):
            msg = (
                f"{label}-coordinate spacing is not uniform: maximum absolute deviation "
                f"{float(maximum_deviation.detach().cpu().item()):.12g} exceeds the combined "
                f"relative ({_UNIFORM_TOLERANCE:.12g}) and {coordinate.dtype} roundoff allowance "
                f"{float(allowed_deviation.detach().cpu().item()):.12g}."
            )
            raise ValueError(msg)
    return dx_work.to(dtype=x_coordinate.dtype), dy_work.to(dtype=y_coordinate.dtype)


def crop_interior(field: Tensor, crop: int, *, axes: SpatialAxes = (-2, -1)) -> Tensor:
    """
    Crop an equal number of cells from every spatial boundary.

    Parameters
    ----------
    field : torch.Tensor
        Tensor containing the spatial axes.
    crop : int
        Non-negative number of cells removed from each side.
    axes : tuple[int, int], optional
        Spatial axes to crop.

    Returns
    -------
    torch.Tensor
        A view of the requested interior. ``crop=0`` returns ``field`` itself.

    Raises
    ------
    TypeError
        If ``crop`` is not an integer or is a boolean.
    ValueError
        If ``crop`` is negative, axes are invalid, or cropping would remove a
        complete spatial dimension.

    """
    if isinstance(crop, bool) or not isinstance(crop, int):
        msg = f"crop must be an integer, got {type(crop).__name__}."
        raise TypeError(msg)
    if crop < 0:
        msg = f"crop must be non-negative, got {crop}."
        raise ValueError(msg)
    if crop == 0:
        return field
    normalized = normalize_spatial_axes(field.ndim, axes)
    if any(2 * crop >= field.shape[axis] for axis in normalized):
        msg = f"crop={crop} removes the complete spatial domain from shape {tuple(field.shape)}."
        raise ValueError(msg)
    slices = [slice(None)] * field.ndim
    for axis in normalized:
        slices[axis] = slice(crop, -crop)
    return field[tuple(slices)]


@dataclass(frozen=True, slots=True)
class PhysicalDerivatives:
    """
    Compute finite-difference physical-space derivatives with ``neuralop.FiniteDiff``.

    The frozen operator preserves input shape, dtype, and device, uses explicit
    x/y spacing, and supports no boundary extension. Grid admission and interior
    cropping remain caller responsibilities.

    Attributes
    ----------
    axes : tuple[int, int]
        Declared ``(y, x)`` axes, normalized and validated when an operation runs.

    """

    axes: SpatialAxes = (-2, -1)

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return the physical-space gradient in ``(d/dx, d/dy)`` order.

        Parameters
        ----------
        field : torch.Tensor
            Real float32 or float64 scalar field with at least four points on
            each declared spatial axis and arbitrary leading dimensions.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(d(field)/dx, d(field)/dy)`` with the input shape, dtype, and
            device. Derivative units are field units per spacing unit.

        Raises
        ------
        TypeError
            If ``field`` is not a float32 or float64 tensor.
        ValueError
            If axes, spatial lengths, or either spacing violate the operator
            contract.

        Notes
        -----
        ``FiniteDiff`` supplies second-order centered interior differences and
        third-order one-sided edge stencils. No boundary cells are cropped.

        """
        # neuraloperator is imported lazily so dependency-light domain consumers,
        # such as Generation contracts, do not load the ML stack on import.
        from neuralop.losses.differentiation import FiniteDiff  # noqa: PLC0415

        normalized_axes = _validate_field(field, self.axes)
        dx = _spacing_float(spacing_x, reference=field, label="spacing_x")
        dy = _spacing_float(spacing_y, reference=field, label="spacing_y")
        moved = field.movedim(normalized_axes, (-2, -1))
        if min(moved.shape[-2:]) < _FINITE_DIFFERENCE_MIN_POINTS:
            msg = f"Finite differences require at least {_FINITE_DIFFERENCE_MIN_POINTS} points per spatial axis, got shape {tuple(field.shape)}."
            raise ValueError(msg)
        # FiniteDiff differentiates the first spatial axis with ``dx`` (our y).
        operator = FiniteDiff(dim=2, h=(dy, dx), periodic_in_x=False, periodic_in_y=False)  # pyright: ignore[reportArgumentType] -- neuralop is unannotated; h accepts one spacing per axis
        return (
            operator.dy(moved).movedim((-2, -1), normalized_axes),
            operator.dx(moved).movedim((-2, -1), normalized_axes),
        )

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return the physical divergence ``d(field_x)/dx + d(field_y)/dy``; see ``_divergence``."""
        return _divergence(self, field_x, field_y, spacing_x, spacing_y)


@dataclass(frozen=True, slots=True)
class SpectralDerivatives:
    """
    Compute FFT derivatives on Fourier-continued fields with ``neuralop.FourierDiff``.

    Frequencies are scaled by physical spacing. Fourier continuation appends a
    smooth Legendre-fitted continuation to both spatial axes, so non-periodic
    fields become smooth periodic signals without jumps or slope kinks, and
    crops back to the original domain. The result keeps the caller dtype/device.

    Attributes
    ----------
    axes : tuple[int, int]
        Declared ``(y, x)`` axes, normalized and validated per operation.

    """

    axes: SpatialAxes = (-2, -1)

    def gradient(
        self,
        field: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return the spectral gradient in ``(d/dx, d/dy)`` order.

        Parameters
        ----------
        field : torch.Tensor
            Real float32 or float64 scalar field with arbitrary leading dimensions.
        spacing_x, spacing_y : float or torch.Tensor
            Finite positive scalar physical spacings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Physically scaled derivatives with the input shape, dtype, and device.

        Raises
        ------
        TypeError
            If ``field`` is not a float32 or float64 tensor.
        ValueError
            If axes, spatial lengths, or either spacing violate the spectral
            operator contract.

        Notes
        -----
        Arbitrary spatial axes move to trailing y/x positions internally and are
        restored on return. Fourier continuation requires at least
        ``2 * FOURIER_CONTINUATION_MATCHING_POINTS`` points per axis and fits the
        boundary samples of each axis. ``FourierDiff`` spaces its grid by
        ``L / n``, so the domain length passed is ``n * spacing``. It builds its
        wavenumbers in torch's default dtype, so float64 fields carry float32
        wavenumber rounding (relative about 1e-7, far below the discretization
        error of the continuation).

        """
        # neuraloperator is imported lazily so dependency-light domain consumers,
        # such as Generation contracts, do not load the ML stack on import.
        from neuralop.losses.differentiation import FourierDiff  # noqa: PLC0415

        normalized_axes = _validate_field(field, self.axes)
        dx = _spacing_float(spacing_x, reference=field, label="spacing_x")
        dy = _spacing_float(spacing_y, reference=field, label="spacing_y")
        moved = field.movedim(normalized_axes, (-2, -1))
        minimum = 2 * FOURIER_CONTINUATION_MATCHING_POINTS
        if min(moved.shape[-2:]) < minimum:
            msg = f"Fourier continuation requires at least {minimum} points per spatial axis, got shape {tuple(field.shape)}."
            raise ValueError(msg)
        height, width = moved.shape[-2:]
        operator = FourierDiff(
            dim=2,
            L=(height * dy, width * dx),
            use_fc="legendre",  # pyright: ignore[reportArgumentType] -- neuralop is unannotated; use_fc accepts a continuation name
            fc_degree=FOURIER_CONTINUATION_MATCHING_POINTS,
            fc_n_additional_pts=FOURIER_CONTINUATION_ADDITIONAL_POINTS,
        )
        # FourierDiff differentiates the first spatial axis with ``dx`` (our y).
        derivative_x = operator.dy(moved).to(field.dtype)  # pyright: ignore[reportOptionalMemberAccess] -- dim=2 always returns a tensor
        derivative_y = operator.dx(moved).to(field.dtype)  # pyright: ignore[reportOptionalMemberAccess] -- dim=2 always returns a tensor
        return (
            derivative_x.movedim((-2, -1), normalized_axes),
            derivative_y.movedim((-2, -1), normalized_axes),
        )

    def divergence(
        self,
        field_x: Tensor,
        field_y: Tensor,
        spacing_x: float | Tensor,
        spacing_y: float | Tensor,
    ) -> Tensor:
        """Return the FFT divergence ``d(field_x)/dx + d(field_y)/dy``; see ``_divergence``."""
        return _divergence(self, field_x, field_y, spacing_x, spacing_y)


def build_derivative_operator(
    kind: str,
    *,
    axes: SpatialAxes = (-2, -1),
) -> DerivativeOperator:
    """
    Build a derivative operator from a semantic identifier.

    Parameters
    ----------
    kind : str
        ``"physical"`` (finite differences) or ``"spectral"`` (FFT with
        Fourier continuation).
    axes : tuple[int, int], optional
        ``(y_axis, x_axis)`` used by the operator.

    Returns
    -------
    DerivativeOperator
        Frozen reusable derivative backend with the requested canonical policy.

    Raises
    ------
    ValueError
        If ``kind`` is unsupported.

    """
    if physics_contracts.validate_derivative_kind(kind) == "physical":
        return PhysicalDerivatives(axes=axes)
    return SpectralDerivatives(axes=axes)
