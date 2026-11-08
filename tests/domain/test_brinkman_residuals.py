"""Verify manufactured Darcy-Brinkman momentum, flux-balance, and boundary equations.

Responsibilities:
  - Check manufactured momentum, flux balance, pressure boundaries, and named channels
  - Check the drag against the inverse of an anisotropic permeability tensor with a cross ratio
  - Check input-derived scales for heterogeneous permeability and the invariance of the dimensionless terms
  - Bind the viscous stress form and viscosity to the stationary COMSOL formulation

Design principles:
  - Analytic uniform, linear, and quadratic fields isolate equation assembly

This module does NOT:
  - Validate derivative backends or select loss weights
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from src import domain

_PERMEABILITY = 1e-8
_INLET_PRESSURE = 3.0


def _grid(height: int = 9, width: int = 11, *, length_x: float = 2.0, length_y: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build batched float64 coordinate planes on x in [0, length_x] and y in [0, length_y].

    Each field has shape ``(1, height, width)`` with ``ij`` indexing, making
    the physical grid spacing explicit for analytic residual assertions.
    """
    y_values = torch.linspace(0.0, length_y, height, dtype=torch.float64)
    x_values = torch.linspace(0.0, length_x, width, dtype=torch.float64)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    return x_grid.unsqueeze(0), y_grid.unsqueeze(0)


def _steady_tensors(*, permeability: float = _PERMEABILITY) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    """Build zero outputs and isotropic inputs with a prescribed inlet pressure, in task order."""
    task = domain.tasks.registry.get_task("steady_flow")
    x_grid, y_grid = _grid()
    zeros = torch.zeros_like(x_grid)
    inlet_pressure = torch.where(y_grid == y_grid.amin(), _INLET_PRESSURE, 0.0)
    input_by_field = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": torch.full_like(x_grid, math.log10(permeability)),
        "Kxy": zeros,
        "Kyy": torch.full_like(x_grid, math.log10(permeability)),
        "eps_bed": torch.full_like(x_grid, 0.5),
        "p_in_bc": inlet_pressure,
    }
    inputs = torch.stack([input_by_field[field] for field in task.input_names], dim=1)
    outputs = torch.zeros(
        (x_grid.shape[0], task.out_channels, *x_grid.shape[1:]),
        dtype=x_grid.dtype,
        device=x_grid.device,
    )
    return inputs, outputs, task.input_names, task.output_names


def _evaluate(
    inputs: torch.Tensor, outputs: torch.Tensor, input_fields: tuple[str, ...], output_fields: tuple[str, ...]
) -> domain.physics.brinkman.BrinkmanDiagnostics:
    """Evaluate the steady task physics with physical derivatives and crop one."""
    return domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs,
        outputs,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=domain.physics.derivatives.PhysicalDerivatives(),
        boundary="pressure_inlet_zero_pressure_outlet",
        interior_crop=1,
    )


def _uniform_darcy_flow() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    """Return the exact uniform Darcy solution between the prescribed inlet and zero outlet."""
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    y_grid = inputs[:, input_fields.index("y")]
    gradient = _INLET_PRESSURE / 1.0
    outputs[:, output_fields.index("p")] = _INLET_PRESSURE * (1.0 - y_grid)
    outputs[:, output_fields.index("v")] = _PERMEABILITY * gradient / domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
    return inputs, outputs, input_fields, output_fields


def test_uniform_darcy_flow_satisfies_every_term_exactly() -> None:
    """
    Evaluate the uniform Darcy solution of an isotropic bed.

    Pressure gradient balances drag, every cross-section carries the same flux,
    and both pressure boundaries hold, so all three terms vanish; the scales
    follow the prescribed inlet pressure, bed height, and permeability.
    """
    inputs, outputs, input_fields, output_fields = _uniform_darcy_flow()
    diagnostics = _evaluate(inputs, outputs, input_fields, output_fields)
    velocity_scale = _PERMEABILITY * _INLET_PRESSURE / domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY

    assert diagnostics.momentum.x.shape == outputs[:, 0].shape
    assert diagnostics.flux_balance.shape == outputs[:, 0, :, 0].shape
    assert diagnostics.scales.pressure.tolist() == pytest.approx([_INLET_PRESSURE])
    assert diagnostics.scales.pressure_gradient.tolist() == pytest.approx([_INLET_PRESSURE])
    assert diagnostics.scales.velocity.tolist() == pytest.approx([velocity_scale])
    assert diagnostics.scales.flux.tolist() == pytest.approx([2.0 * velocity_scale])
    assert diagnostics.momentum_mse.item() == pytest.approx(0.0, abs=1e-20)
    assert diagnostics.continuity_mse.item() == pytest.approx(0.0, abs=1e-24)
    assert diagnostics.boundary_mse.item() == pytest.approx(0.0, abs=1e-24)


def test_scales_use_each_samples_geometric_permeability_pressure_and_extents() -> None:
    """
    Evaluate two samples with distinct heterogeneous permeability fields and inlet pressures.

    On a grid that is neither unit wide nor unit tall, each sample must scale by
    its own mean inlet pressure, its own geometric mean of ``sqrt(Kxx*Kyy)``,
    and the domain extents, so batch pooling or an arithmetic mean cannot pass.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    length_x = 3.0
    length_y = 0.5
    x_grid, y_grid = _grid(height=7, width=9, length_x=length_x, length_y=length_y)
    x_grid = x_grid.repeat(2, 1, 1)
    y_grid = y_grid.repeat(2, 1, 1)
    zeros = torch.zeros_like(x_grid)

    def per_sample(first: float, second: float) -> torch.Tensor:
        return torch.tensor([first, second], dtype=torch.float64).view(2, 1, 1)

    # Log10 permeabilities vary linearly along x and y with per-sample slopes, so the
    # arithmetic and the geometric mean of sqrt(Kxx * Kyy) differ.
    log_kxx = per_sample(-9.0, -7.5) + per_sample(2.0, -1.0) * x_grid / length_x
    log_kyy = per_sample(-8.0, -10.0) + per_sample(1.5, 2.5) * y_grid / length_y
    # A pressure ramp along the inlet makes the mean inlet pressure differ from its end values.
    inlet_ramp = per_sample(2.0, 5.0) * (1.0 + 0.5 * x_grid / length_x)
    inlet_pressure = torch.where(y_grid == y_grid.amin(dim=(-2, -1), keepdim=True), inlet_ramp, zeros)
    input_by_field = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": log_kxx,
        "Kxy": zeros,
        "Kyy": log_kyy,
        "eps_bed": torch.full_like(x_grid, 0.5),
        "p_in_bc": inlet_pressure,
    }
    inputs = torch.stack([input_by_field[field] for field in task.input_names], dim=1)
    outputs = torch.zeros((2, task.out_channels, *x_grid.shape[1:]), dtype=torch.float64)

    scales = _evaluate(inputs, outputs, task.input_names, task.output_names).scales

    viscosity = domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
    expected_pressure, expected_gradient, expected_velocity, expected_flux = [], [], [], []
    for sample in range(2):
        pressure = inlet_pressure[sample, 0].mean().item()
        # exp of the mean log equals 10 ** (mean of 0.5 * (log10 Kxx + log10 Kyy)).
        geometric_permeability = 10.0 ** (0.5 * (log_kxx[sample].mean().item() + log_kyy[sample].mean().item()))
        gradient = pressure / length_y
        velocity = geometric_permeability * gradient / viscosity
        expected_pressure.append(pressure)
        expected_gradient.append(gradient)
        expected_velocity.append(velocity)
        expected_flux.append(velocity * length_x)

    assert scales.pressure.tolist() == pytest.approx(expected_pressure, rel=1e-9)
    assert scales.pressure_gradient.tolist() == pytest.approx(expected_gradient, rel=1e-9)
    assert scales.velocity.tolist() == pytest.approx(expected_velocity, rel=1e-9)
    assert scales.flux.tolist() == pytest.approx(expected_flux, rel=1e-9)


def test_dimensionless_terms_do_not_depend_on_the_pressure_level() -> None:
    """
    Scale inlet pressure, pressure, and velocity of a perturbed state together.

    Every residual is linear in that common factor and so is every scale, so the
    three dimensionless terms must stay unchanged and nonzero.
    """
    inputs, outputs, input_fields, output_fields = _uniform_darcy_flow()
    x_grid = inputs[:, input_fields.index("x")]
    y_grid = inputs[:, input_fields.index("y")]
    outputs[:, output_fields.index("p")] += 0.1 * torch.sin(math.pi * x_grid)
    outputs[:, output_fields.index("v")] *= 1.0 + 0.2 * y_grid
    reference = _evaluate(inputs, outputs, input_fields, output_fields)
    factor = 7.0
    scaled_inputs = inputs.clone()
    scaled_inputs[:, input_fields.index("p_in_bc")] *= factor
    scaled = _evaluate(scaled_inputs, factor * outputs, input_fields, output_fields)

    for name in ("momentum_mse", "continuity_mse", "boundary_mse"):
        assert getattr(reference, name).item() > 0.0
        assert getattr(scaled, name).item() == pytest.approx(getattr(reference, name).item(), rel=1e-10)


def test_linear_state_matches_analytic_brinkman_residuals() -> None:
    """
    Evaluate Brinkman diagnostics on linear pressure and velocity fields.

    The computed momentum residuals must equal the analytic pressure-gradient,
    viscous, and permeability-drag terms, protecting signs, units, and tensor axes,
    and the flux balance must equal the analytic row-flux deviation.
    """
    inputs, outputs, input_fields, output_fields = _steady_tensors(permeability=1.0)
    x_grid = inputs[:, input_fields.index("x")]
    y_grid = inputs[:, input_fields.index("y")]
    velocity_scale = 1e-5
    outputs[:, output_fields.index("p")] = x_grid + 2.0 * y_grid
    outputs[:, output_fields.index("u")] = velocity_scale * x_grid
    outputs[:, output_fields.index("v")] = -velocity_scale * y_grid
    diagnostics = _evaluate(inputs, outputs, input_fields, output_fields)
    viscosity = domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
    expected_x = -torch.ones_like(x_grid) - viscosity * velocity_scale * x_grid
    expected_y = -2.0 * torch.ones_like(y_grid) + viscosity * velocity_scale * y_grid
    rows = y_grid[:, :, 0]
    # q(y) = -c y L_x and Q = (K dp / (L_y mu)) L_x with K = 1 m^2 and L_y = 1 m.
    expected_flux_balance = -velocity_scale * (rows - rows.mean()) * viscosity / _INLET_PRESSURE

    assert torch.allclose(diagnostics.momentum.x, expected_x, atol=1e-12, rtol=1e-12)
    assert torch.allclose(diagnostics.momentum.y, expected_y, atol=1e-12, rtol=1e-12)
    assert torch.allclose(diagnostics.flux_balance, expected_flux_balance, atol=1e-24, rtol=1e-10)


def test_anisotropic_permeability_drag_matches_the_inverse_tensor() -> None:
    """
    Evaluate constant velocity under a linear pressure in an anisotropic permeability.

    Constant velocity has no viscous stress and linear pressure a constant gradient,
    so the residual must equal ``-grad(p) - mu K^-1 u`` with ``K`` assembled from
    unequal diagonals and a cross ratio, protecting component order and sign.
    """
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    x_grid = inputs[:, input_fields.index("x")]
    y_grid = inputs[:, input_fields.index("y")]
    permeability_xx = 4e-9
    permeability_yy = 1e-8
    cross_ratio = 0.6
    pressure_gradient = torch.tensor([1.5, -2.5], dtype=torch.float64)
    velocity = torch.tensor([3e-4, -5e-4], dtype=torch.float64)
    inputs[:, input_fields.index("Kxx")] = math.log10(permeability_xx)
    inputs[:, input_fields.index("Kyy")] = math.log10(permeability_yy)
    # The stored cross channel is the ratio Kxy / sqrt(Kxx * Kyy), not a permeability.
    inputs[:, input_fields.index("Kxy")] = cross_ratio
    outputs[:, output_fields.index("p")] = pressure_gradient[0] * x_grid + pressure_gradient[1] * y_grid
    outputs[:, output_fields.index("u")] = velocity[0]
    outputs[:, output_fields.index("v")] = velocity[1]

    diagnostics = _evaluate(inputs, outputs, input_fields, output_fields)

    cross_permeability = cross_ratio * math.sqrt(permeability_xx * permeability_yy)
    tensor = torch.tensor(
        [[permeability_xx, cross_permeability], [cross_permeability, permeability_yy]],
        dtype=torch.float64,
    )
    expected = -pressure_gradient - domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY * torch.linalg.inv(tensor) @ velocity

    assert torch.allclose(diagnostics.momentum.x, torch.full_like(x_grid, expected[0].item()), rtol=1e-9, atol=1e-9)
    assert torch.allclose(diagnostics.momentum.y, torch.full_like(y_grid, expected[1].item()), rtol=1e-9, atol=1e-9)


def test_flux_balance_accepts_heterogeneous_divergence_free_flow() -> None:
    """
    Let v vary across the bed width but not along the flow direction.

    Such a field is divergence-free, so every row carries the same flux and the
    balance must vanish although v itself is far from uniform.
    """
    x_grid, _ = _grid()
    velocity_y = 1.0 + 0.5 * torch.sin(math.pi * x_grid / 2.0)
    balance = domain.physics.brinkman.cross_section_flux_balance(velocity_y, x_grid, torch.ones(1, dtype=torch.float64))

    assert torch.allclose(balance, torch.zeros_like(balance), atol=1e-14)


def test_viscous_stress_matches_the_comsol_brinkman_stress() -> None:
    """
    Evaluate a divergent velocity whose trace term varies in both directions, with heterogeneous porosity.

    With ``u = v = x y`` and ``mu/eps = mu (a + b x)`` the COMSOL stress
    ``(mu/eps)(grad u + grad u^T - (2/3) div(u) I)`` gives
    ``div(tau)_x = mu (b (4y - 2x)/3 + (a + b x)/3)`` and ``div(tau)_y = mu (b (x + y) + (a + b x)/3)``.
    Dropping the trace from either normal stress, a wrong trace factor or sign, or mu/eps
    outside the divergence changes at least one component.
    """
    x_grid, y_grid = _grid()
    zeros = torch.zeros_like(x_grid)
    ones = torch.ones_like(x_grid)
    inverse_porosity = 2.0 + 0.5 * x_grid
    velocity = x_grid * y_grid
    momentum = domain.physics.brinkman.brinkman_momentum_residuals(
        zeros,
        velocity,
        velocity,
        1.0 / inverse_porosity,
        ones,
        zeros,
        ones,
        domain.physics.derivatives.PhysicalDerivatives(),
        0.2,
        0.125,
    )
    viscosity = domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
    expected_x = viscosity * (0.5 * (4.0 * y_grid - 2.0 * x_grid) / 3.0 + inverse_porosity / 3.0 - velocity)
    expected_y = viscosity * (0.5 * (x_grid + y_grid) + inverse_porosity / 3.0 - velocity)
    # One-sided edge stencils are inexact for quadratics; two cells inward only central ones remain.
    interior = (slice(None), slice(2, -2), slice(2, -2))

    assert torch.allclose(momentum.x[interior], expected_x[interior], rtol=0.0, atol=1e-12 * viscosity)
    assert torch.allclose(momentum.y[interior], expected_y[interior], rtol=0.0, atol=1e-12 * viscosity)


def test_viscosity_matches_comsol_air_at_flow_reference_temperature() -> None:
    """Bind the residual viscosity to COMSOL's Air eta(T) at the Generation T_flow_ref."""
    common_path = Path(__file__).resolve().parents[2] / "configs/01_generation/physical_model/common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    temperature = float(common["scientific_fixed_values"]["T_flow_ref"]["value"])
    # COMSOL Material Library "Air" dynamic viscosity polynomial in K, as stored in the stationary template.
    coefficients = (-8.38278e-7, 8.35717342e-8, -7.69429583e-11, 4.6437266e-14, -1.06585607e-17)
    comsol_viscosity = sum(coefficient * temperature**power for power, coefficient in enumerate(coefficients))

    assert pytest.approx(comsol_viscosity, rel=1e-5) == domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY


def test_scales_reject_a_missing_inlet_pressure() -> None:
    """A zero prescribed inlet pressure leaves no scale, so evaluation must fail closed."""
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    inputs[:, input_fields.index("p_in_bc")] = 0.0

    with pytest.raises(ValueError, match="inlet pressure must be finite and positive"):
        _evaluate(inputs, outputs, input_fields, output_fields)


def test_outlet_pressure_gauge_is_reduced_per_sample() -> None:
    """
    Give two samples equal-magnitude, opposite-sign outlet pressure gauges.

    Per-sample means must remain distinct, preventing cross-sample
    cancellation from hiding boundary error.
    """
    _, y_grid = _grid()
    y_grid = y_grid.repeat(2, 1, 1)
    pressure = torch.zeros_like(y_grid)
    outlet = y_grid == y_grid.amax(dim=(-2, -1), keepdim=True)
    pressure[0][outlet[0]] = 1.0
    pressure[1][outlet[1]] = -1.0
    residuals = domain.physics.boundary.pressure_boundary_residuals(
        pressure,
        torch.zeros_like(pressure),
        y_grid,
        0.125,
    )

    assert residuals.outlet_sample_mean.tolist() == pytest.approx([1.0, -1.0])


def test_pressure_boundary_residual_and_field_map_invariance() -> None:
    """
    Satisfy the inlet gauge, then reorder every input and output channel with its label.

    Boundary error must remain zero and all residual arrays invariant, proving
    physics binds semantic field names rather than positional conventions.
    """
    inputs, outputs, input_fields, output_fields = _uniform_darcy_flow()
    outputs[:, output_fields.index("u")] = 1e-3 * inputs[:, input_fields.index("x")]
    reference = _evaluate(inputs, outputs, input_fields, output_fields)

    input_order = (6, 0, 5, 2, 1, 4, 3)
    output_order = (2, 0, 1)
    reordered = _evaluate(
        inputs[:, input_order],
        outputs[:, output_order],
        tuple(input_fields[index] for index in input_order),
        tuple(output_fields[index] for index in output_order),
    )

    assert reference.boundary_mse.item() == pytest.approx(0.0, abs=1e-24)
    assert torch.allclose(reference.momentum.x, reordered.momentum.x)
    assert torch.allclose(reference.momentum.y, reordered.momentum.y)
    assert torch.allclose(reference.flux_balance, reordered.flux_balance)
