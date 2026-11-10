"""Verify component checks: membership, residual diagnostics, derivatives, spectra, strides, increments, timing, and criteria.

Responsibilities:
  - Bind steady per-case reductions and the term decomposition to the domain residual
  - Check the 2/3 trace part and the pointwise-divergence scaling on manufactured fields
  - Check float64 coordinate rebuilding, boundary bands, edge profiles, and perturbations
  - Check transient field integrals, global and sorption closures, and validity counts
  - Check the physics-setting inventory of authored steady configurations
  - Check the Train/validation membership filter, its refusal of test and OOD roles, and that
    only selected cases reach the steady reference, transient cases, balances, and scaling fit
  - Check manufactured-field derivative errors, mode truncation, stride views, increment statistics, timing rows, and criteria

Design principles:
  - Small synthetic grids with analytic answers; no published Datasets or solvers
  - Dataset runtimes are replaced by recording stand-ins at the factory boundary

This module does NOT:
  - Re-test domain derivative operators or Brinkman equations covered in tests/domain
  - Measure real timings or real reference fields
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src import common, datasets, domain, experiments, generation
from src.analysis.component_checks import component_checks_increments as increments
from src.analysis.component_checks import component_checks_plots as plots
from src.analysis.component_checks import component_checks_sources as sources
from src.analysis.component_checks import component_checks_spectral as spectral
from src.analysis.component_checks import component_checks_steady as steady
from src.analysis.component_checks import component_checks_summary as summary
from src.analysis.component_checks import component_checks_timing as timing
from src.analysis.component_checks import component_checks_transient as transient

if TYPE_CHECKING:
    from collections.abc import Iterator

_VISCOSITY = domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
_INLET_PRESSURE = 3.0
_PERMEABILITY = 1e-8
_FD = steady.backend_by_label("fd")


def _reference(height: int = 12, width: int = 16, *, cases: int = 2) -> sources.SteadyReference:
    """Return uniform Darcy flow between a prescribed inlet and a zero outlet on a 2 m x 1 m bed."""
    task = domain.tasks.registry.get_task("steady_flow")
    y_values = torch.linspace(0.0, 1.0, height, dtype=torch.float32)
    x_values = torch.linspace(0.0, 2.0, width, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    by_input = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": torch.full_like(x_grid, math.log10(_PERMEABILITY)),
        "Kxy": torch.zeros_like(x_grid),
        "Kyy": torch.full_like(x_grid, math.log10(_PERMEABILITY)),
        "eps_bed": torch.full_like(x_grid, 0.5),
        "p_in_bc": torch.where(y_grid == 0.0, _INLET_PRESSURE, 0.0),
    }
    by_output = {
        "p": _INLET_PRESSURE * (1.0 - y_grid),
        "u": torch.zeros_like(x_grid),
        "v": torch.full_like(x_grid, _PERMEABILITY * _INLET_PRESSURE / _VISCOSITY),
    }
    inputs = torch.stack([by_input[name] for name in task.input_names]).expand(cases, -1, -1, -1).clone()
    outputs = torch.stack([by_output[name] for name in task.output_names]).expand(cases, -1, -1, -1).clone()
    return sources.SteadyReference(
        dataset_id="synthetic",
        membership="train",
        case_ids=tuple(f"lentil__case_{index:04d}__digest" for index in range(cases)),
        material_families=("lentil",) * cases,
        input_fields=tuple(task.input_names),
        output_fields=tuple(task.output_names),
        inputs=inputs,
        outputs=outputs,
    )


def _with_outputs(reference: sources.SteadyReference, **fields: torch.Tensor) -> sources.SteadyReference:
    """Return a reference copy whose named output channels are replaced."""
    outputs = reference.outputs.clone()
    for name, value in fields.items():
        outputs[:, reference.output_fields.index(name)] = value
    return sources.SteadyReference(
        dataset_id=reference.dataset_id,
        membership=reference.membership,
        case_ids=reference.case_ids,
        material_families=reference.material_families,
        input_fields=reference.input_fields,
        output_fields=reference.output_fields,
        inputs=reference.inputs,
        outputs=outputs,
    )


def _fields(reference: sources.SteadyReference) -> steady.SteadyResidualFields:
    """Evaluate float64 FD residual fields of a reference."""
    inputs, outputs = reference.tensors(torch.float64)
    return steady.evaluate_residual_fields(inputs, outputs, input_fields=reference.input_fields, output_fields=reference.output_fields, backend=_FD)


def test_uniform_darcy_flow_has_zero_terms_and_reductions_bind_the_domain_loss() -> None:
    """
    Evaluate exact Darcy flow, then a state with a sinusoidal cross velocity.

    The exact state leaves only float32 rounding in momentum; the perturbed state
    reproduces the domain momentum MSE at every crop, and the pressure and drag
    parts equal -grad(p)/G and -mu K^-1 u / G.
    """
    reference = _reference()
    exact = steady.residual_rows(_fields(reference), crop=1)
    assert np.allclose(exact["momentum"], 0.0, atol=1e-10)
    for name in ("trace_rms", "pointwise_divergence", "porous_divergence", "flux_balance", "boundary"):
        assert np.allclose(exact[name], 0.0, atol=1e-20)

    x_grid = reference.inputs[:, reference.input_fields.index("x")].to(torch.float64)
    perturbed = _with_outputs(reference, u=(1e-3 * torch.sin(math.pi * x_grid)).to(torch.float32))
    inputs, outputs = perturbed.tensors(torch.float64)
    fields = _fields(perturbed)
    for crop in (0, 2):
        domain_loss = domain.physics.brinkman.evaluate_steady_2d_brinkman(
            inputs,
            outputs,
            input_fields=perturbed.input_fields,
            output_fields=perturbed.output_fields,
            derivatives=_FD.build(),
            boundary=domain.physics.contracts.PRESSURE_BOUNDARY_KIND,
            interior_crop=crop,
        )
        assert steady.residual_rows(fields, crop)["momentum"].mean() == pytest.approx(domain_loss.momentum_mse.item(), rel=1e-12)
    gradient = fields.scales.pressure_gradient[:, None, None]
    velocity_x = outputs[:, perturbed.output_fields.index("u")]
    velocity_y = outputs[:, perturbed.output_fields.index("v")]
    assert torch.allclose(fields.drag[:, 0], -_VISCOSITY * velocity_x / _PERMEABILITY / gradient, rtol=1e-6, atol=1e-12)
    assert torch.allclose(fields.drag[:, 1], -_VISCOSITY * velocity_y / _PERMEABILITY / gradient, rtol=1e-6)
    assert torch.allclose(fields.pressure[:, 1], torch.ones_like(fields.pressure[:, 1]), rtol=1e-5)


def test_trace_part_is_contained_in_the_viscous_term_of_a_divergent_field() -> None:
    """
    For u = v = x y the symmetric stress part gives div(tau) = mu / eps in both components.

    The 2/3 trace part adds -(2/3) mu / eps per component, so the viscous term, which
    contains it, is mu / (3 eps).
    """
    reference = _reference()
    x_grid = reference.inputs[:, reference.input_fields.index("x")]
    y_grid = reference.inputs[:, reference.input_fields.index("y")]
    velocity = x_grid * y_grid
    fields = _fields(_with_outputs(reference, p=torch.zeros_like(x_grid), u=velocity, v=velocity))
    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))
    gradient = fields.scales.pressure_gradient[:, None, None, None]
    symmetric = _VISCOSITY / 0.5 / gradient
    assert torch.allclose(fields.viscous[interior], (symmetric / 3.0).expand_as(fields.viscous[interior]), rtol=1e-4)
    assert torch.allclose(fields.trace[interior], (-2.0 * symmetric / 3.0).expand_as(fields.trace[interior]), rtol=1e-4)


def test_pointwise_divergence_is_scaled_by_the_darcy_velocity_over_the_bed_height() -> None:
    """A linear v = c y has div(u) = c everywhere, reported as c L_y / U."""
    reference = _reference()
    y_grid = reference.inputs[:, reference.input_fields.index("y")]
    slope = 2e-4
    fields = _fields(_with_outputs(reference, v=slope * y_grid))
    velocity = fields.scales.velocity[:, None, None]
    assert torch.allclose(fields.divergence, (slope * 1.0 / velocity).expand_as(fields.divergence), rtol=1e-5)


def test_float64_coordinates_rebuild_uniform_grids_and_reject_nonuniform_ones() -> None:
    """Rebuilt float64 grids are exactly uniform within one float32 ulp of the stored values; a distorted grid fails closed."""
    reference = _reference()
    inputs, _ = reference.tensors(torch.float64)
    x_index = reference.input_fields.index("x")
    assert inputs.dtype == torch.float64
    ulp = torch.finfo(torch.float32).eps * 2.0
    assert (inputs[:, x_index] - reference.inputs[:, x_index].to(torch.float64)).abs().max().item() <= ulp
    assert torch.allclose(torch.diff(inputs[:, x_index], dim=-1), torch.full((2, 12, 15), 2.0 / 15.0, dtype=torch.float64), rtol=1e-12)
    distorted = reference.inputs[:, x_index].clone()
    distorted[..., 3] += 1e-3
    with pytest.raises(ValueError, match="not a uniform grid"):
        sources.uniform_float64_coordinate(distorted, axis=-1)


def test_boundary_bands_and_edge_profiles_follow_cell_distance() -> None:
    """Corners lie in band zero, the centre in the interior band, and constant fields give constant profiles."""
    distance = steady.edge_distance((41, 41))
    bands = steady.band_index((41, 41))
    assert distance[0, 0] == 0
    assert distance[20, 20] == 20
    assert steady.BAND_LABELS[bands[20, 20]] == ">=17"
    assert steady.BAND_LABELS[bands[4, 20]] == "3-4"
    profiles = steady.edge_profiles(np.full((2, 41, 41), 3.0), max_distance=4)
    assert set(profiles["edge"]) == set(steady.EDGES)
    assert np.allclose(profiles["rms"], 3.0)


def test_perturbations_are_seeded_scaled_and_expose_the_flux_balance_blind_spot() -> None:
    """Smooth errors have the requested relative RMS per seed; a uniform velocity scaling leaves the flux balance at zero."""
    reference = _reference(height=16, width=20)
    smooth = steady.Perturbation("smooth", ("v",), 0.01)
    _, first = steady.perturb_outputs(reference, smooth, dtype=torch.float64, seed=7)
    _, again = steady.perturb_outputs(reference, smooth, dtype=torch.float64, seed=7)
    _, other = steady.perturb_outputs(reference, smooth, dtype=torch.float64, seed=8)
    channel = reference.output_fields.index("v")
    base = reference.outputs[:, channel].to(torch.float64)
    error = first[:, channel] - base
    relative = error.square().mean(dim=(-2, -1)).sqrt() / base.square().mean(dim=(-2, -1)).sqrt()
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert torch.allclose(relative, torch.full_like(relative, 0.01), rtol=1e-6)

    inputs, scaled = steady.perturb_outputs(reference, steady.Perturbation("velocity_scale", (), 0.01), dtype=torch.float64, seed=0)
    fields = steady.evaluate_residual_fields(inputs, scaled, input_fields=reference.input_fields, output_fields=reference.output_fields, backend=_FD)
    rows = steady.residual_rows(fields, crop=1)
    assert np.allclose(rows["flux_balance"], 0.0, atol=1e-24)
    assert np.allclose(rows["momentum"], 0.01**2, rtol=1e-3)
    with pytest.raises(ValueError, match="unknown output fields"):
        steady.perturb_outputs(reference, steady.Perturbation("noise", ("T",), 0.01), dtype=torch.float64, seed=0)


def test_sensitivity_reports_ratios_against_the_reference_floor() -> None:
    """A pressure offset leaves momentum unchanged and shifts inlet and outlet by one percent of the inlet pressure."""
    reference = _with_outputs(_reference(), u=1e-4 * torch.ones(2, 12, 16))
    frame = steady.evaluate_sensitivity(
        reference,
        settings=(("fd", 1),),
        perturbations=(steady.Perturbation("pressure_offset", (), 0.01),),
        seeds=(1,),
        dtype=torch.float64,
    )
    assert len(frame) == 2
    assert np.allclose(frame["momentum_ratio"], 1.0)
    assert np.allclose(frame["boundary"], 2 * 0.01**2, rtol=1e-5)
    assert frame["seed"].isna().all()


def _transient_case(*, phi: float = 0.5, steps: int = 3) -> sources.TransientReferenceCase:
    """Return a uniform 6 x 7 transient case on a 0.6 m x 0.5 m bed with exact COMSOL globals."""
    height, width = 6, 7
    y_grid, x_grid = np.meshgrid(np.linspace(0.0, 0.5, height), np.linspace(0.0, 0.6, width), indexing="ij")
    temperature = 300.0
    w_eq = 600.0 * domain.moisture.oswin_equilibrium_dry_basis_moisture(
        phi, temperature, a_osw=12.0, b_osw=-0.05, c_osw=0.35, clip_bounds=(1e-6, 0.999)
    )
    shape = (steps, height, width)
    states = {
        "T": np.full(shape, temperature, dtype=np.float32),
        "phi": np.full(shape, phi, dtype=np.float32),
        "w_surf": np.full(shape, float(w_eq) * 0.9, dtype=np.float32),
        "w_int": np.full(shape, float(w_eq) * 0.9, dtype=np.float32),
    }
    times = np.arange(steps, dtype=np.float64)
    grain = np.full(steps, 0.8 * 0.3 * float(w_eq) * 0.9)
    return sources.TransientReferenceCase(
        dataset_id="synthetic",
        case_id="lentil__case_0001__digest",
        material_family="lentil",
        membership="train",
        time_hours=times,
        states=states,
        static={
            "x": x_grid,
            "y": y_grid,
            "u": np.zeros((height, width)),
            "v": np.full((height, width), 0.2),
            "p": np.zeros((height, width)),
            "eps_bed": np.full((height, width), 0.4),
            "rho_bu_dry": np.full((height, width), 600.0),
        },
        boundary={
            "T_in_bc_t_n": np.full(steps - 1, temperature),
            "T_in_bc_t_n_plus_1": np.full(steps - 1, temperature),
            "omega_in_bc_t_n": np.full(steps - 1, 0.01),
            "omega_in_bc_t_n_plus_1": np.full(steps - 1, 0.01),
        },
        scalars={"r_surf_0": 1e-5, "r_int_surf": 0.2, "f_surf": 0.4, "A_osw": 12.0, "B_osw": -0.05, "C_osw": 0.35},
        global_series={
            "t": times,
            "m_w_gr": grain,
            "m_v_gas": np.full(steps, 1e-3),
            "m_dot_evap": np.zeros(steps),
            "m_dot_v_in": np.full(steps, 2e-3),
            "m_dot_v_out": np.full(steps, 2e-3),
            "mt_mass_balance": np.zeros(steps),
        },
        constants={"Lz": 0.8, "M_v": 0.018, "p_ref": 101325.0, "phi_clip_min": 1e-6, "phi_clip_max": 0.999},
        exact_stop_time=None,
    )


def test_field_integrals_use_trapezoidal_quadrature_and_the_case_thickness() -> None:
    """Uniform fields integrate to value times area times Lz; sub-equilibrium surface water does not evaporate."""
    case = _transient_case()
    integrals = transient.field_integrals(case)
    water = 0.4 * float(case.states["w_surf"][0, 0, 0]) + 0.6 * float(case.states["w_int"][0, 0, 0])
    density = 0.018 * 0.5 * generation.cases.schedule.saturation_vapor_pressure(np.asarray(300.0)) / (transient.MOLAR_GAS_CONSTANT * 300.0)
    area = 0.6 * 0.5
    assert integrals["grain_water_kg"].to_numpy() == pytest.approx(np.full(3, 0.8 * area * water), rel=1e-6)
    assert integrals["gas_vapour_kg"].to_numpy() == pytest.approx(np.full(3, 0.8 * area * 0.4 * density), rel=1e-6)
    assert integrals["inlet_advective_kg_per_s"].to_numpy() == pytest.approx(np.full(3, 0.8 * 0.6 * 0.2 * density), rel=1e-6)
    assert np.all(integrals["evaporation_kg_per_s"] == 0.0)
    assert np.all(integrals["validity_invalid_points"] == 0)


def test_global_balance_closes_exact_series_and_reports_a_leak_with_its_sign() -> None:
    """Constant storage with balanced fluxes closes; a missing outflow appears as a negative total closure."""
    case = _transient_case()
    assert np.allclose(transient.global_balance(case)[["grain_closure_kg_trapezoid", "total_closure_kg_trapezoid"]], 0.0)
    case.global_series["m_dot_v_out"] = np.full(3, 2e-3 - 1e-6)
    leak = transient.global_balance(case)["total_closure_kg_trapezoid"].to_numpy()
    assert leak == pytest.approx(np.full(2, -1e-6 * transient.SECONDS_PER_HOUR))
    missing = {**case.global_series, "t": case.global_series["t"] + 0.5}
    shifted = dataclasses.replace(case, global_series=missing)
    with pytest.raises(ValueError, match="missing from the COMSOL global series"):
        transient.compare_with_global(shifted, transient.field_integrals(shifted))


def test_sorption_residual_vanishes_for_a_crank_nicolson_exchange_step() -> None:
    """A Crank-Nicolson internal step satisfies the trapezoidal residual exactly and not the left-endpoint one."""
    case = _transient_case()
    rate = 0.2 * 1e-5
    step = transient.SECONDS_PER_HOUR
    surface = float(case.states["w_surf"][0, 0, 0])
    internal = [surface + 1000.0]
    for _ in range(2):
        internal.append((internal[-1] * (1.0 - 0.5 * rate * step) + rate * step * surface) / (1.0 + 0.5 * rate * step))
    case.states["w_int"][:] = np.asarray(internal, dtype=np.float64)[:, None, None]
    residuals = transient.sorption_residuals(case)
    trapezoid = residuals[residuals["rule"] == "trapezoid"]["internal_relative"].to_numpy()
    left = residuals[residuals["rule"] == "left"]["internal_relative"].to_numpy()
    assert np.all(trapezoid < 1e-4)
    assert np.all(left > 1e-3)


def test_supersaturated_reference_states_are_counted_against_the_validity_rule() -> None:
    """Relative humidity above one is counted; only values beyond the validity tolerance are flagged as invalid."""
    case = _transient_case(phi=0.5)
    case.states["phi"][1, 2, 3] = 1.02
    case.states["phi"][2, 2, 3] = 1.12
    integrals = transient.field_integrals(case)
    assert integrals["phi_outside_unit_interval"].tolist() == [0, 1, 1]
    assert integrals["validity_invalid_points"].tolist() == [0, 0, 1]
    assert integrals["phi_max"].iloc[2] == pytest.approx(1.12, rel=1e-6)


def test_balance_availability_reads_generation_and_dataset_contracts() -> None:
    """Pore-gas vapour and boundary fluxes are COMSOL globals that the Dataset contract does not carry."""
    table = transient.balance_term_availability().set_index("term")
    assert table.loc["pore-gas vapour m_v_gas", "comsol_global_series"]
    assert not table.loc["pore-gas vapour m_v_gas", "dataset_term"]
    assert table.loc["inlet flux m_dot_v_in", "dataset_has_required_fields"]
    assert not table.loc["COMSOL balance mt_mass_balance", "dataset_has_required_fields"]


def test_physics_setting_inventory_reads_authored_experiments_and_studies(tmp_path: Path) -> None:
    """Enabled physics blocks are listed with their derivative and crop, omitted keys take the loader defaults; disabled ones are skipped."""
    root = tmp_path / "steady_flow"
    (root / "experiments").mkdir(parents=True)
    (root / "optuna").mkdir()
    physics = {
        "enabled": True,
        "derivatives": {"kind": "physical"},
        "interior_crop": 3,
        "momentum_weight": {"target": 2.0},
        "continuity_weight": {"target": 5.0},
        "boundary_weight": {"target": 7.0},
    }
    (root / "base.yaml").write_text(yaml.safe_dump({"loss": {"physics": {**physics, "enabled": False}}}), encoding="utf-8")
    (root / "experiments" / "pi.yaml").write_text(
        yaml.safe_dump({"task": "steady_flow", "base": "../base.yaml", "loss": {"physics": physics}}), encoding="utf-8"
    )
    (root / "experiments" / "plain.yaml").write_text(yaml.safe_dump({"task": "steady_flow", "base": "../base.yaml"}), encoding="utf-8")
    (root / "experiments" / "minimal.yaml").write_text(
        yaml.safe_dump({"task": "steady_flow", "loss": {"physics": {"enabled": True}}}), encoding="utf-8"
    )
    study = {
        "experiment": {"task": "steady_flow", "base": "../base.yaml", "loss": {"physics": {**physics, "interior_crop": 1}}},
        "search_space": {"loss.physics.momentum_weight.target": {"kind": "float"}, "optimizer.lr": {"kind": "float"}},
    }
    (root / "optuna" / "study.yaml").write_text(yaml.safe_dump(study), encoding="utf-8")
    table = steady.maintained_physics_settings(root).set_index("config")
    assert sorted(table.index) == ["experiments/minimal.yaml", "experiments/pi.yaml", "optuna/study.yaml"]
    assert table.loc["experiments/pi.yaml", "interior_crop"] == 3
    defaults = experiments.config.defaults.get_task_defaults("steady_flow")["loss"]["physics"]
    assert table.loc["experiments/minimal.yaml", "derivative"] == defaults["derivatives"]["kind"]
    assert table.loc["experiments/minimal.yaml", "interior_crop"] == defaults["interior_crop"]
    assert table.loc["experiments/minimal.yaml", "momentum_weight"] == pytest.approx(defaults["momentum_weight"]["target"])
    assert table.loc["optuna/study.yaml", "searched_physics_keys"] == "loss.physics.momentum_weight.target"


def test_backend_tables_cover_every_backend_crop_and_band() -> None:
    """Backend evaluation feeds one decomposition row per case, backend and crop, and one band row per case, backend and band."""
    reference = _reference()
    fields = steady.evaluate_backends(reference, backends=(_FD,), dtype=torch.float64)
    terms = steady.momentum_term_table(fields, case_labels=reference.case_labels, crops=(0, 1))
    bands = steady.residual_band_table(fields, case_labels=reference.case_labels)
    assert len(terms) == 2 * 2
    assert len(bands) == 2 * len(steady.BAND_LABELS)
    assert list(steady.band_overview(bands).index) == list(steady.BAND_LABELS)
    assert reference.extent(0) == pytest.approx((0.0, 2.0, 0.0, 1.0))


def test_degenerate_denominators_yield_undefined_values_instead_of_errors() -> None:
    """
    Evaluate terms and cases whose normalizing quantity is zero.

    An unchanged zero term has an undefined (NaN) sensitivity ratio, a trajectory
    that removes no water keeps absolute closures with NaN relative closures, zero
    COMSOL values give NaN relative differences, and a field with a zero baseline
    derivative or no interior band gives NaN derivative differences.
    """
    reference = _reference()
    frame = steady.evaluate_sensitivity(
        reference, settings=(("fd", 1),), perturbations=(steady.Perturbation("pressure_offset", (), 0.01),), seeds=(1,), dtype=torch.float64
    )
    assert frame["flux_balance_ratio"].isna().all()
    assert np.isinf(frame["boundary_ratio"]).all()

    diagnostics = transient.evaluate_case(_transient_case())
    closure = diagnostics.summary["grain_closure_trapezoid"]
    assert isinstance(closure, float)
    assert math.isnan(closure)
    assert diagnostics.balance["grain_closure_kg_trapezoid"].abs().max() == 0.0
    assert diagnostics.comparison["evaporation_relative_difference"].isna().all()

    comparison = steady.compare_derivatives(reference, others=("fc",))
    assert comparison["relative_difference"].isna().all()
    assert list(steady.derivative_overview(comparison).columns) == list(steady.BAND_LABELS)


def test_stored_times_match_the_nearest_global_row_within_tolerance() -> None:
    """A stored time a hair above its COMSOL row still pairs with that row."""
    case = _transient_case()
    shifted = dataclasses.replace(case, time_hours=case.time_hours + np.array([0.0, 1e-9, 0.0]))
    comparison = transient.compare_with_global(shifted, transient.field_integrals(shifted))
    assert comparison["grain_water_comsol"].to_numpy() == pytest.approx(case.global_series["m_w_gr"])


def test_negative_stored_water_is_counted_without_aborting() -> None:
    """A negative interpolated water value is counted, flagged by the validity rule, and integrated as zero."""
    case = _transient_case()
    case.states["w_int"][2, 2, 3] = -1e-6
    integrals = transient.field_integrals(case)
    assert integrals["negative_water_points"].tolist() == [0, 0, 1]
    assert integrals["validity_invalid_points"].tolist() == [0, 0, 1]
    assert integrals["grain_water_kg"].iloc[2] < integrals["grain_water_kg"].iloc[1]


def test_transient_plots_accept_more_cases_than_categorical_colours() -> None:
    """More cases than palette colours fall back to neutral unlabelled lines instead of failing."""
    balance = transient.global_balance(_transient_case())
    balance["grain_change_kg"] = -1.0
    for name in ("grain", "gas", "total"):
        balance[f"{name}_closure_kg_trapezoid"] = 1e-3
    count = len(plots.SERIES_COLORS) + 1
    figure = plots.plot_transient_balances({f"case {index}": balance for index in range(count)})
    assert "cases" in figure.axes[0].get_title()
    plt.close(figure)


def test_changed_source_is_refused_with_the_dataset_admission_message(tmp_path: Path) -> None:
    """A source whose bytes changed after Dataset admission fails with the one message the Dataset owner defines."""
    path = tmp_path / "source.h5"
    path.write_bytes(b"admitted")
    admitted_stat, expected_sha256 = sources._file_stat(path), common.serialization.file_sha256(path)
    path.write_bytes(b"changed after admission")
    with pytest.raises(ValueError, match=r"^Source HDF5 changed after Dataset admission: "):
        sources._read_source_balances(path, expected_sha256=expected_sha256, admitted_stat=admitted_stat)


def test_membership_filter_resolves_roles_before_the_case_limit_and_refuses_test_and_ood() -> None:
    """Train and validation resolve in stored order and are cut after filtering; test, OOD, unknown and absent roles fail closed."""
    partition = {"train": ["t1", "t2", "t3"], "validation": ["v1"], "id_test": ["e1"], "parameter_ood": ["p1"], "ood": ["o1"]}
    assert sources.select_membership_cases(partition, "train", dataset_id="d", max_cases=2) == ("t1", "t2")
    assert sources.select_membership_cases(partition, "validation", dataset_id="d", max_cases=4) == ("v1",)
    for role in ("id_test", "ood", "material_ood", "unknown"):
        with pytest.raises(ValueError, match="test and OOD cases stay unseen"):
            sources.select_membership_cases(partition, role, dataset_id="d")
    for absent in (None, {**partition, "validation": []}):
        with pytest.raises(ValueError, match="has no validation membership"):
            sources.select_membership_cases(absent, "validation", dataset_id="d")


_PARTITION_IDS: dict[str, list[str]] = {
    role: [f"lentil__case_{role}{index}__digest" for index in range(count)] for role, count in (("train", 3), ("validation", 2), ("id_test", 1))
}
_STORED_IDS = tuple(case_id for role in ("validation", "train", "id_test") for case_id in _PARTITION_IDS[role])


def test_steady_reference_reads_only_the_leading_membership_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the first two Train cases are read from the Dataset; validation and test rows stay unread."""
    reference = _reference(cases=len(_STORED_IDS))
    read: list[str] = []

    class FakeSteady:
        manifest_binding = SimpleNamespace(case_partition=_PARTITION_IDS)
        identity = SimpleNamespace(dataset_id="synthetic", sample_ids=_STORED_IDS)

        def __getitem__(self, index: int) -> dict[str, object]:
            read.append(_STORED_IDS[index])
            meta = {"case_id": _STORED_IDS[index], "material_family": "lentil"}
            return {"x": reference.inputs[index], "y": reference.outputs[index], "meta": meta}

    monkeypatch.setattr(datasets.runtime.steady, "SteadyFlowDataset", FakeSteady)
    monkeypatch.setattr(datasets.runtime.factory, "create_dataset", lambda _request: FakeSteady())
    loaded = sources.load_steady_reference("synthetic", membership="train", max_cases=2)
    assert loaded.case_ids == tuple(read) == tuple(_PARTITION_IDS["train"][:2])
    assert loaded.membership == "train"


class _FakeTrajectories:
    """Canonical-HDF5 runtime stand-in that serves one synthetic case under every stored ID and records reads."""

    def __init__(self) -> None:
        self.manifest_binding = SimpleNamespace(case_partition=_PARTITION_IDS)
        self.payload = {
            "cases": [
                {"package_case_id": case_id, "source_relative_path": f"{case_id}.h5", "source_hdf5_sha256": "0" * 64, "material_family": "lentil"}
                for case_id in _STORED_IDS
            ]
        }
        self.source_root = Path("/nonexistent")
        self.read: list[str] = []
        self.closed = False

    def read_case_arrays(self, case_id: str) -> SimpleNamespace:
        self.read.append(case_id)
        case, step = _transient_case(), datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
        return SimpleNamespace(
            time=case.time_hours,
            states=np.stack([case.states[field.name] for field in step.dynamic_state], axis=1),
            static=np.stack([case.static[field.name] for field in step.static_spatial_conditioning]),
            boundary=np.stack([case.boundary.get(field.name, case.time_hours[1:]) for field in step.step_boundary_conditioning], axis=1),
            scalars=np.array([case.scalars.get(field.name, 0.0) for field in step.scalar_conditioning]),
            exact_stop_time=None,
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("consumer", ["reference_cases", "balances"])
def test_transient_cases_and_balances_read_only_the_leading_membership_cases(monkeypatch: pytest.MonkeyPatch, consumer: str) -> None:
    """Reference cases and balance diagnostics read only the first validation case and its source, and close the runtime."""
    runtime, case = _FakeTrajectories(), _transient_case()
    balance_sources: list[str] = []

    def read_balances(path: Path, **_kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        balance_sources.append(path.stem)
        return case.global_series, case.constants

    monkeypatch.setattr(datasets.runtime.factory, "create_hdf5_source_dataset", lambda _request: runtime)
    monkeypatch.setattr(sources, "_file_stat", lambda _path: (0, 0))
    monkeypatch.setattr(sources, "_read_source_balances", read_balances)
    if consumer == "reference_cases":
        cases = list(sources.iter_transient_reference_cases("synthetic", membership="validation", max_cases=1))
        assert [item.membership for item in cases] == ["validation"]
    else:
        labels = [item.label for item in transient.evaluate_dataset("synthetic", membership="validation", max_cases=1)]
        assert labels == ["lentil/case_validation0"]
    assert runtime.read == balance_sources == _PARTITION_IDS["validation"][:1]
    assert runtime.closed


def test_membership_scaling_fits_only_the_leading_membership_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scaling fit consumes the items of the selected validation cases only, and its digest hashes exactly those IDs."""
    consumed: list[str] = []
    fitted: dict[str, object] = {}

    def scaling_items(case_ids: tuple[str, ...]) -> Iterator[dict[str, str]]:
        return iter([{"case_id": case_id} for case_id in case_ids])

    class FakeShards:
        manifest_binding = SimpleNamespace(case_partition=_PARTITION_IDS)
        payload_identity = None

        def iter_scaling_items(self) -> Iterator[dict[str, str]]:
            """Yield one ID-tagged item per stored case, i.e. the unfiltered runtime."""
            return scaling_items(_STORED_IDS)

        def close(self) -> None:
            """Release nothing."""

    def select(_dataset: object, case_ids: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(iter_scaling_items=lambda: scaling_items(case_ids), configured_regular_horizon=1, close=lambda: None)

    def fit(items: Iterator[dict[str, str]], **kwargs: object) -> str:
        consumed.extend(item["case_id"] for item in items)
        fitted.update(kwargs)
        return "scaling"

    monkeypatch.setattr(datasets.runtime.transient, "TransientPTShardDataset", FakeShards)
    monkeypatch.setattr(datasets.runtime.factory, "create_dataset", lambda _request: FakeShards())
    monkeypatch.setattr(datasets.runtime.transient, "select_transient_cases", select)
    monkeypatch.setattr(datasets.preprocessing.splits, "split_dataset_identity", lambda _binding, _identity: {})
    monkeypatch.setattr(increments.transient_contracts.TransientTensorizerSpec, "from_mapping", lambda _mapping: None)
    monkeypatch.setattr(increments.transient_scaling, "fit_transient_scaling", fit)
    config = {"data": {"spatial_stride": 1}, "input_profile": None, "temporal": {"temporal_conditioning": None}, "scaling": {"mode": "state_std"}}
    assert increments.fit_membership_scaling("synthetic", membership="validation", config=config, max_cases=1) == "scaling"
    assert consumed == _PARTITION_IDS["validation"][:1]
    assert fitted["train_membership_digest"] == common.serialization.canonical_json_sha256(_PARTITION_IDS["validation"][:1])


def test_manufactured_fields_bound_backend_errors_and_crop_removes_edge_bands() -> None:
    """Float64 does not raise either backend's error, every error is finite, and a crop column keeps only its bands."""
    errors = steady.manufactured_derivative_errors((0.0, 0.4, 0.0, 0.25), (41, 61))
    assert np.isfinite(errors["relative_error"]).all()
    overview = steady.manufactured_overview(errors)
    crop0 = overview["crop 0"].astype(float)
    for backend in ("fd", "fc"):
        assert crop0[(backend, "float64")] <= crop0[(backend, "float32")] * (1 + 1e-3)
    kept = errors[(errors["backend"] == "fc") & (errors["dtype"] == "float64") & ~errors["band"].isin(["0", "1"])]
    weighted = kept.assign(weight=kept["relative_error"] ** 2 * kept["points"]).groupby(["field", "direction"])[["weight", "points"]].sum()
    fc_crop2 = float(overview["crop 2"].astype(float)[("fc", "float64")])
    assert fc_crop2 == pytest.approx(float(np.sqrt(weighted["weight"] / weighted["points"]).max()))
    assert fc_crop2 < crop0[("fc", "float64")]


def test_mode_truncation_keeps_the_layer_modes_and_never_grows_with_more_modes() -> None:
    """The kept set matches the probed neuralop mapping, all modes reproduce the field, and more modes never lose more."""
    assert int(spectral.kept_mode_mask((20, 30), (8, 10)).sum()) == 8 * (10 // 2 + 1)
    field = torch.randn(1, 2, 24, 36, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    errors = spectral.truncation_errors(field, ("a", "b"), mode_counts=(4, 8, 16, 23), paddings=(0.0,), mode_pairs=[("all", (24, 36))])
    assert errors.loc[errors["axis"] == "config", ["relative_l2", "edge_relative"]].to_numpy() == pytest.approx(0.0, abs=1e-12)
    for (_, axis), group in errors[errors["axis"] != "config"].groupby(["field", "axis"]):
        curve = group.sort_values(f"modes_{axis}")["relative_l2"].to_numpy()
        assert np.all(np.diff(curve) <= 1e-12)
    zero = spectral.truncation_errors(torch.zeros(1, 1, 12, 16), ("z",), mode_counts=(4,), paddings=(0.1,))
    assert (zero[["relative_l2", "edge_relative"]].to_numpy() == 0.0).all()


def test_truncation_keeps_each_sample_mean_so_zero_padding_shows_its_jump() -> None:
    """Constant samples of different values lose energy under zero padding; without padding the kept DC mode reproduces them."""
    fields = torch.stack([torch.full((1, 12, 16), 1.0), torch.full((1, 12, 16), 3.0)])
    errors = spectral.truncation_errors(fields, ("c",), mode_counts=(4,), paddings=(0.0, 0.1))
    assert (errors.loc[errors["padding"] == 0.1, ["relative_l2", "edge_relative"]].to_numpy() > 0.0).all()
    assert errors.loc[errors["padding"] == 0.0, ["relative_l2", "edge_relative"]].to_numpy() == pytest.approx(0.0, abs=1e-12)


def test_stride_views_keep_endpoints_and_rebuild_bilinear_fields_exactly() -> None:
    """Views keep both endpoints, stride one and linear fields rebuild exactly, and a stride that drops an endpoint is refused."""
    grid = torch.arange(9 * 13, dtype=torch.float32).reshape(9, 13)
    view = spectral.stride_view(grid, 2)
    assert view.shape == (5, 7)
    assert float(view[0, 0]) == float(grid[0, 0])
    assert float(view[-1, -1]) == float(grid[-1, -1])
    y_grid, x_grid = np.meshgrid(np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 11), indexing="ij")
    linear = np.stack([(step + 1.0) * x_grid + 2.0 * y_grid for step in range(3)]).astype(np.float32)
    case = dataclasses.replace(_transient_case(), states={name: linear.copy() for name in ("T", "phi", "w_surf", "w_int")})
    floor = spectral.stride_floor(case)
    assert (floor.loc[floor["stride"] == 1, "max_normalized_rmse"] == 0.0).all()
    assert (floor.loc[floor["stride"] == 2, "max_normalized_rmse"] < 1e-6).all()
    assert floor.loc[floor["stride"] == 4, "status"].str.startswith("refused").all()


def test_increment_statistics_count_beta_exceedance_and_zeros_and_flag_nonuniform_steps() -> None:
    """Exact zeros, values above beta, the maximum and equal-weight shares are exact; an irregular step is not uniform."""
    batch = torch.tensor([[[[0.0, 0.5]], [[0.1, 0.1]]], [[[2.0, 0.0]], [[0.1, 0.1]]]])
    stats = increments.scaled_increment_statistics([batch], channels=("a", "b"), beta=1.0)
    assert stats.loc["a", "zero_fraction"] == 0.5
    assert stats.loc["a", "above_beta_fraction"] == 0.25
    assert stats.loc["b", "above_beta_fraction"] == 0.0
    assert stats.loc["a", "max"] == 2.0
    assert stats.loc["b", "q50"] == pytest.approx(0.1, rel=1e-6)
    assert stats.loc["a", "loss_share"] == pytest.approx((1.625 / 4) / (1.625 / 4 + 0.005), rel=1e-6)
    steps = increments.time_step_table({"regular": np.array([0.0, 1.0, 2.0]), "irregular": np.array([0.0, 1.0, 2.5])})
    assert steps["uniform"].to_dict() == {"regular": True, "irregular": False, "all cases": False}


def test_timing_without_cuda_reports_unavailable_gpu_rows_and_measured_cpu_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a GPU the CUDA rows say unavailable instead of substituting hardware; CPU rows carry a positive time."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reference = _reference()
    frame = timing.time_physics_loss(
        reference.inputs,
        reference.outputs,
        input_fields=reference.input_fields,
        output_fields=reference.output_fields,
        backends=steady.DERIVATIVE_BACKENDS,
        crop=1,
        warmups=0,
        repeats=1,
    )
    gpu, cpu = frame[frame["device"] == "cuda"], frame[frame["device"] == "cpu"]
    assert set(gpu["backend"]) == set(cpu["backend"]) == {"fd", "fc"}
    assert (gpu["status"] == "unavailable").all()
    assert (cpu["status"] == "measured").all()
    assert (cpu["median_ms"].astype(float) > 0).all()


def test_summary_applies_each_proposed_criterion() -> None:
    """Each criterion turns constructed tables at or across its threshold into its status; missing tables stay not measured."""
    index = pd.MultiIndex.from_product([("fd", "fc"), (0, 1, 2)], names=["backend", "crop"])
    crop = pd.DataFrame(
        {"momentum_floor_median": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0], "smooth_1pct_ratio_median": [10.0, 100.0, 300.0, 5.0, 50.0, 99.9]}, index=index
    )
    timed = pd.DataFrame({"device": ["cpu", "cpu"], "backend": ["fd", "fc"], "status": ["measured"] * 2, "median_ms": [10.0, 20.0]})
    curves = [
        {"field": "p", "axis": axis, "config": None, "modes_y": modes if axis == "y" else None, "modes_x": modes if axis == "x" else None}
        | {"padding": 0.0, "relative_l2": error, "edge_relative": error}
        for axis in ("y", "x")
        for modes, error in ((8, 2e-3), (16, 1e-3), (32, 1e-4))
    ]
    configured = [
        {"field": "p", "axis": "config", "config": "fno", "modes_y": 16, "modes_x": 16, "padding": padding, "relative_l2": 0.1, "edge_relative": edge}
        for padding, edge in ((0.0, 1.0), (0.1, 0.95))
    ]
    modes = pd.DataFrame(curves + configured)
    stride = pd.DataFrame(
        {"stride": [1, 2, 4], "status": ["admissible", "admissible", "refused: endpoint"], "max_normalized_rmse": [0.0, 1e-3, math.nan]}
    )
    tables = increments.TransientComponentTables(
        spectral=modes, stride=stride, increments=pd.DataFrame({"above_beta_fraction": [0.0, 0.0]}), time_steps=pd.DataFrame()
    )
    status = summary.summary_table(crop=crop, timing=timed, steady_modes=modes, transient=tables)["status"]
    assert status["Derivative backend: FC arm"] == "keep FC arm"
    assert status["Interior crop"] == "fd: crop 1; fc: none"
    assert status["n_modes steady, axis y"] == "padding 0: lower bound 16, sufficient 32"
    assert status["Domain padding steady"] == "no effect"
    assert status["Spatial stride"] == "stride 1: admissible; stride 2: admissible; stride 4: refused"
    assert status["Huber regime"] == "Huber = MSE prior"
    assert (summary.summary_table()["status"] == summary.NOT_MEASURED).all()
    assert summary.summary_table()["criterion"].str.endswith(f"({summary.CRITERIA_STATE})").all()
    assert sources.membership_caption("train", ["a", "b"]) == "train membership · 2 cases"


def test_padding_with_all_zero_edge_errors_has_no_effect() -> None:
    """Edge-band errors that are zero for both paddings agree exactly instead of turning 0/0 into an effect."""
    exact = pd.DataFrame({"padding": [0.0, 0.1], "relative_l2": 0.0, "edge_relative": 0.0})
    exact = exact.assign(field="p", axis="config", config="fno", modes_y=16, modes_x=16)
    assert summary.summary_table(steady_modes=exact).loc["Domain padding steady", "status"] == "no effect"
