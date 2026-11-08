"""Verify reusable physical and spectral derivative operators on analytic fields.

Responsibilities:
  - Check physical/spectral derivatives, axis order, crop, spacing, and kind admission

Design principles:
  - Manufactured linear and smooth non-periodic fields isolate derivative behavior

This module does NOT:
  - Assemble Brinkman equations or claim production-field accuracy
"""

from __future__ import annotations

import pytest
import torch

from src import domain


def test_physical_derivatives_match_linear_field_and_crop() -> None:
    """
    Differentiate a linear float64 field on a rectangular uniform physical grid.

    Both analytic gradients and a valid interior crop must be exact, while an
    overlarge crop must fail so finite-difference axes and region semantics stay fixed.
    """
    x_values = torch.linspace(-2.0, 3.0, 17, dtype=torch.float64)
    y_values = torch.linspace(1.0, 4.0, 13, dtype=torch.float64)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    field = (3.0 * x_grid - 2.0 * y_grid + 5.0).unsqueeze(0)
    operator = domain.physics.derivatives.PhysicalDerivatives()
    derivative_x, derivative_y = operator.gradient(
        field,
        x_values[1] - x_values[0],
        y_values[1] - y_values[0],
    )

    assert torch.allclose(derivative_x, torch.full_like(field, 3.0), atol=1e-12)
    assert torch.allclose(derivative_y, torch.full_like(field, -2.0), atol=1e-12)
    assert domain.physics.derivatives.crop_interior(field, 2).shape == (1, 9, 13)
    with pytest.raises(ValueError, match="complete spatial domain"):
        domain.physics.derivatives.crop_interior(field, 7)


def test_uniform_spacing_rejects_invalid_coordinate_grids() -> None:
    """
    Infer spacing from one valid grid, then vary finiteness, order, spacing, and Cartesian form.

    The valid grid must yield exact unit spacing and every malformed family must
    fail explicitly before a derivative operator consumes it.
    """
    y_grid, x_grid = torch.meshgrid(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
        indexing="ij",
    )
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(x_grid, y_grid)
    assert spacing_x.item() == pytest.approx(1.0)
    assert spacing_y.item() == pytest.approx(1.0)

    nonfinite = x_grid.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        domain.physics.derivatives.infer_uniform_spacing(nonfinite, y_grid)

    with pytest.raises(ValueError, match="strictly increasing"):
        domain.physics.derivatives.infer_uniform_spacing(x_grid.flip(-1), y_grid)

    nonuniform = x_grid.clone()
    nonuniform[:, 2:] += 1.0
    with pytest.raises(ValueError, match="not uniform"):
        domain.physics.derivatives.infer_uniform_spacing(nonuniform, y_grid)

    noncartesian = x_grid.clone()
    noncartesian[1] += 0.1
    with pytest.raises(ValueError, match="constant along the y-axis"):
        domain.physics.derivatives.infer_uniform_spacing(noncartesian, y_grid)


def test_uniform_spacing_accepts_float32_quantized_cartesian_grid() -> None:
    """Admit the real-grid geometry after ordinary float32 coordinate quantization."""
    x_values = torch.linspace(0.0, 1.2, 401, dtype=torch.float32)
    y_values = torch.linspace(0.0, 0.75, 251, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    x_differences = torch.diff(x_grid.to(torch.float64), dim=-1)
    y_differences = torch.diff(y_grid.to(torch.float64), dim=-2)
    x_relative_only = ((x_differences - x_differences.mean()).abs() / x_differences.mean()).amax()
    y_relative_only = ((y_differences - y_differences.mean()).abs() / y_differences.mean()).amax()

    assert x_relative_only.item() > 1e-5
    assert y_relative_only.item() > 1e-5
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(x_grid, y_grid)
    assert spacing_x.dtype == torch.float32
    assert spacing_y.dtype == torch.float32
    assert spacing_x.item() == pytest.approx(0.003, rel=1e-6)
    assert spacing_y.item() == pytest.approx(0.003, rel=1e-6)


def test_fourier_continuation_differentiates_non_periodic_fields() -> None:
    """
    Differentiate fields whose boundary values differ on a non-periodic rectangle.

    A linear field and a cosine mode with non-zero boundary slope must both be
    resolved to high accuracy up to the boundary rows, which neither a periodic
    wrap (jump) nor an even reflection (slope kink) can achieve.
    """
    length_x, length_y = 1.2, 0.75
    x_values = torch.linspace(0.0, length_x, 41, dtype=torch.float64)
    y_values = torch.linspace(0.0, length_y, 25, dtype=torch.float64)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(x_grid, y_grid)
    operator = domain.physics.derivatives.SpectralDerivatives()

    linear = (2.0 * x_grid - 3.0 * y_grid).unsqueeze(0)
    linear_x, linear_y = operator.gradient(linear, spacing_x, spacing_y)
    assert torch.allclose(linear_x, torch.full_like(linear_x, 2.0), rtol=1e-3)
    assert torch.allclose(linear_y, torch.full_like(linear_y, -3.0), rtol=1e-3)

    field = torch.sin(0.5 * torch.pi * y_grid / length_y).unsqueeze(0)
    _, derivative_y = operator.gradient(field, spacing_x, spacing_y)
    expected_y = (0.5 * torch.pi / length_y * torch.cos(0.5 * torch.pi * y_grid / length_y)).unsqueeze(0)
    assert torch.allclose(derivative_y, expected_y, atol=1e-3 * expected_y.abs().max().item())

    with pytest.raises(ValueError, match="at least 10 points"):
        operator.gradient(linear[..., :9, :], spacing_x, spacing_y)


# Probe on this 41 x 61 grid at crop 2 (max error relative to max |analytic|):
# physical 6e-4 / 5e-5, spectral 1e-5 / 4e-5 for d/dx / d/dy.
@pytest.mark.parametrize(("kind", "tolerance"), [("physical", 2e-3), ("spectral", 2e-4)])
def test_operators_match_analytic_non_periodic_field_in_interior(kind: str, tolerance: float) -> None:
    """Differentiate a smooth non-periodic field and compare both gradients inside a two-cell crop."""
    ny, nx, length_y, length_x = 41, 61, 0.75, 1.2
    y = torch.linspace(0.0, length_y, ny, dtype=torch.float64).reshape(-1, 1)
    x = torch.linspace(0.0, length_x, nx, dtype=torch.float64).reshape(1, -1)
    field = (torch.sin(3 * x) * torch.exp(y) + x**2 * y).expand(2, ny, nx)
    expected_x = (3 * torch.cos(3 * x) * torch.exp(y) + 2 * x * y).expand(2, ny, nx)
    expected_y = (torch.sin(3 * x) * torch.exp(y) + x**2).expand(2, ny, nx)
    operator = domain.physics.derivatives.build_derivative_operator(kind)

    derivative_x, derivative_y = operator.gradient(field, length_x / (nx - 1), length_y / (ny - 1))

    for estimate, expected in ((derivative_x, expected_x), (derivative_y, expected_y)):
        interior = domain.physics.derivatives.crop_interior(estimate - expected, 2)
        assert interior.abs().max().item() <= tolerance * expected.abs().max().item()


@pytest.mark.parametrize("kind", ["physical", "spectral"])
def test_gradient_follows_declared_axes_and_spacing(kind: str) -> None:
    """
    Differentiate fields that vary along one axis only, in default and moved layouts.

    x must run along columns and y along rows by default, and declared axes
    must move the same derivatives to other tensor positions; the two spacings differ
    so an x/y swap in the library call would change the values.
    """
    n_rows, n_columns, spacing_x, spacing_y = 12, 16, 0.5, 0.25
    x = torch.arange(n_columns, dtype=torch.float64) * spacing_x
    y = torch.arange(n_rows, dtype=torch.float64) * spacing_y
    only_x = x.expand(n_rows, n_columns)
    only_y = y.reshape(-1, 1).expand(n_rows, n_columns)
    operator = domain.physics.derivatives.build_derivative_operator(kind)

    for field, expected in ((only_x, (1.0, 0.0)), (only_y, (0.0, 1.0))):
        derivative_x, derivative_y = operator.gradient(field, spacing_x, spacing_y)
        assert torch.allclose(derivative_x, torch.full_like(field, expected[0]), atol=1e-3)
        assert torch.allclose(derivative_y, torch.full_like(field, expected[1]), atol=1e-3)

    moved = domain.physics.derivatives.build_derivative_operator(kind, axes=(0, 2))
    derivative_x, derivative_y = moved.gradient(only_x.unsqueeze(1), spacing_x, spacing_y)
    assert derivative_x.shape == (n_rows, 1, n_columns)
    assert torch.allclose(derivative_x, torch.ones_like(derivative_x), atol=1e-3)
    assert torch.allclose(derivative_y, torch.zeros_like(derivative_y), atol=1e-3)


@pytest.mark.parametrize("kind", ["physical", "spectral"])
def test_float32_gradients_carry_autograd_without_mutating_the_field(kind: str) -> None:
    """Backpropagate through both derivatives of a float32 field, as the physics loss does in training."""
    y = torch.linspace(0.0, 0.75, 21).reshape(-1, 1)
    x = torch.linspace(0.0, 1.2, 31).reshape(1, -1)
    field = (torch.sin(3 * x) * torch.exp(y)).requires_grad_()
    original = field.detach().clone()
    operator = domain.physics.derivatives.build_derivative_operator(kind)

    derivative_x, derivative_y = operator.gradient(field, 1.2 / 30, 0.75 / 20)
    (derivative_x.square().mean() + derivative_y.square().mean()).backward()

    assert derivative_x.dtype == derivative_y.dtype == torch.float32
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert field.grad.abs().max().item() > 0.0
    assert torch.equal(field.detach(), original)


@pytest.mark.parametrize("spacing", [0.0, float("nan"), torch.tensor([1.0, 1.0])])
def test_operators_reject_invalid_spacing(spacing: float | torch.Tensor) -> None:
    """Reject zero, non-finite and non-scalar spacing before the library runs."""
    field = torch.zeros(12, 12, dtype=torch.float64)
    for kind in ("physical", "spectral"):
        with pytest.raises(ValueError, match="spacing_x"):
            domain.physics.derivatives.build_derivative_operator(kind).gradient(field, spacing, 1.0)


def test_operators_reject_unsupported_inputs() -> None:
    """Reject half precision and too few stencil points before the library runs."""
    physical = domain.physics.derivatives.PhysicalDerivatives()
    field = torch.zeros(6, 6, dtype=torch.float64)
    with pytest.raises(TypeError, match="float32 or float64"):
        physical.gradient(field.to(torch.float16), 1.0, 1.0)
    with pytest.raises(ValueError, match="at least 4 points"):
        physical.gradient(field[:3], 1.0, 1.0)


def test_derivative_semantics_fail_clearly() -> None:
    """
    Request an unknown derivative kind.

    The contradiction must fail during construction, preventing a hidden
    fallback from changing scientific meaning.
    """
    with pytest.raises(ValueError, match="Unknown derivative identifier"):
        domain.physics.derivatives.build_derivative_operator("unsupported")
