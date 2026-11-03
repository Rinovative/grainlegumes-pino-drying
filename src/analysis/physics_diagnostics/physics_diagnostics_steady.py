"""Diagnose steady Darcy-Brinkman residuals of reference fields per derivative backend.

Responsibilities:
  - Evaluate the training-loss terms of reference fields through the domain evaluator
  - Decompose momentum into pressure, viscous, and drag parts and add the 2/3 trace variant
  - Evaluate the pointwise divergence variants removed from the loss
  - Reduce residuals per case over interior crops, boundary bands, and edge profiles
  - Compare finite-difference and spectral derivatives of the same reference fields
  - Measure term sensitivity to controlled smooth, noisy, scaling, and offset errors
  - Inventory the physics-loss settings of the maintained steady configurations
  - Condense per-case tables into case-statistic overviews

Design principles:
  - Domain physics owns the loss equations; variants only add explicitly labelled terms
  - Every term is dimensionless with the input-derived per-case domain scales
  - Perturbation seeds are explicit and reported with every sensitivity row
  - Reference tensors are never mutated

This module does NOT:
  - Change loss equations, weights, crops, or derivative defaults
  - Load Datasets, render figures, or persist results
  - Decide whether a variant matches COMSOL; it reports magnitudes only

Notes:
  Momentum terms, each divided by G = dp / L_y:
    pressure = -grad(p), viscous = div((mu/eps)(grad u + grad u^T)), drag = -mu K^-1 u,
    residual = pressure + viscous + drag (exactly the domain residual),
    trace = -(2/3) grad((mu/eps) div u), the compressible stress term COMSOL omits
    for incompressible flow; residual_with_trace = residual + trace.
  The parts come from the domain residual itself: drag from an evaluation with
  zero derivatives, viscous minus drag from an evaluation with zero pressure,
  and their difference to the full residual must equal -grad(p) to roundoff.
  Pointwise variants: div(u) L_y / U and div(eps u) L_y / (mean(eps) U).
  Boundary bands use the cell distance to the nearest grid edge.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src import common, domain, experiments

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .physics_diagnostics_sources import SteadyReference

PerturbationKind = Literal["smooth", "noise", "velocity_scale", "pressure_offset"]

TRACE_COEFFICIENT: Final = 2.0 / 3.0
DEFAULT_CROPS: Final = (0, 1, 2, 4, 8)
BAND_EDGES: Final = (0, 1, 2, 3, 5, 9, 17)
BAND_LABELS: Final = ("0", "1", "2", "3-4", "5-8", "9-16", ">=17")
EDGES: Final = ("inlet", "outlet", "wall_x_min", "wall_x_max")
SMOOTH_MAX_MODE: Final = 4
DEFAULT_SEED: Final = 20261027
_INTERIOR_BAND = BAND_LABELS[-1]
# Allowed roundoff, in dtype epsilons, between the domain residual and its linear parts.
_DECOMPOSITION_TOLERANCE_FACTOR = 64.0
_RESIDUAL_COLUMNS = (
    "momentum",
    "momentum_with_trace",
    "trace_rms",
    "pointwise_divergence",
    "porous_divergence",
    "flux_balance",
    "boundary_inlet",
    "boundary_outlet",
    "boundary",
)


@dataclass(frozen=True, slots=True)
class DerivativeBackend:
    """
    Name one derivative operator configuration.

    Attributes
    ----------
    label : str
        Short stable identifier used in tables.
    kind, extension : str
        Domain derivative identifiers passed to the operator factory.
    description : str
        Readable description for figures.

    """

    label: str
    kind: str
    extension: str
    description: str

    def build(self) -> domain.physics.derivatives.DerivativeOperator:
        """Return the domain operator for this configuration."""
        return domain.physics.derivatives.build_derivative_operator(self.kind, extension=self.extension)


DERIVATIVE_BACKENDS: Final = (
    DerivativeBackend("fd", "physical", "none", "finite differences (torch.gradient)"),
    DerivativeBackend("fc", "spectral", "fourier_continuation", "spectral, Fourier continuation"),
    DerivativeBackend("periodic", "spectral", "none", "spectral, periodic (no extension)"),
)


@dataclass(frozen=True, slots=True)
class _ZeroDerivatives:
    """Return zero derivatives so the domain residual reduces to its drag term."""

    kind: Literal["physical", "spectral"] = "physical"
    extension: Literal["none", "fourier_continuation"] = "none"
    axes: tuple[int, int] = (-2, -1)

    def gradient(self, field: Tensor, spacing_x: float | Tensor, spacing_y: float | Tensor) -> tuple[Tensor, Tensor]:
        """Return two zero fields shaped like ``field``."""
        del spacing_x, spacing_y
        return torch.zeros_like(field), torch.zeros_like(field)

    def divergence(self, field_x: Tensor, field_y: Tensor, spacing_x: float | Tensor, spacing_y: float | Tensor) -> Tensor:
        """Return one zero field shaped like the components."""
        del field_y, spacing_x, spacing_y
        return torch.zeros_like(field_x)


@dataclass(frozen=True, slots=True)
class SteadyResidualFields:
    """
    Hold full-grid dimensionless residual fields of one backend evaluation.

    Attributes
    ----------
    backend : str
        Derivative backend label.
    pressure, viscous, drag, trace, residual : torch.Tensor
        Momentum terms ``[case, 2, y, x]`` divided by the per-case ``G``.
    divergence, porous_divergence : torch.Tensor
        Pointwise divergence variants ``[case, y, x]``.
    flux_balance : torch.Tensor
        Domain cross-section flux balance ``[case, y]``.
    boundary_inlet, boundary_outlet : torch.Tensor
        Domain per-case dimensionless pressure-boundary terms ``[case]``.
    scales : domain.physics.brinkman.BrinkmanScales
        Input-derived per-case scales.

    """

    backend: str
    pressure: Tensor
    viscous: Tensor
    drag: Tensor
    trace: Tensor
    residual: Tensor
    divergence: Tensor
    porous_divergence: Tensor
    flux_balance: Tensor
    boundary_inlet: Tensor
    boundary_outlet: Tensor
    scales: domain.physics.brinkman.BrinkmanScales

    @property
    def residual_with_trace(self) -> Tensor:
        """Return the momentum residual including the 2/3 trace term."""
        return self.residual + self.trace


@dataclass(frozen=True, slots=True)
class Perturbation:
    """
    Describe one controlled error added to reference outputs.

    Attributes
    ----------
    kind : {"smooth", "noise", "velocity_scale", "pressure_offset"}
        ``smooth`` adds a unit-RMS low-mode cosine field and ``noise`` iid
        Gaussian values, each scaled by ``amplitude`` times the per-case RMS of
        the perturbed channel. ``velocity_scale`` multiplies ``u`` and ``v`` by
        ``1 + amplitude``; ``pressure_offset`` adds ``amplitude * dp`` to ``p``.
    fields : tuple[str, ...]
        Perturbed output channels for ``smooth`` and ``noise``.
    amplitude : float
        Relative error size, e.g. ``0.01`` for one percent.

    """

    kind: PerturbationKind
    fields: tuple[str, ...]
    amplitude: float

    @property
    def stochastic(self) -> bool:
        """Return whether the perturbation depends on a random seed."""
        return self.kind in {"smooth", "noise"}

    @property
    def label(self) -> str:
        """Return a compact human-readable description."""
        percent = f"{100.0 * self.amplitude:g} %"
        if self.stochastic:
            return f"{self.kind} {percent} ({'+'.join(self.fields)})"
        return f"{self.kind} {percent}"


DEFAULT_PERTURBATIONS: Final = (
    Perturbation("smooth", ("p", "u", "v"), 0.001),
    Perturbation("smooth", ("p", "u", "v"), 0.01),
    Perturbation("smooth", ("p", "u", "v"), 0.1),
    Perturbation("smooth", ("p",), 0.01),
    Perturbation("smooth", ("u",), 0.01),
    Perturbation("smooth", ("v",), 0.01),
    Perturbation("noise", ("p", "u", "v"), 0.001),
    Perturbation("noise", ("p", "u", "v"), 0.01),
    Perturbation("noise", ("p", "u", "v"), 0.1),
    Perturbation("velocity_scale", (), 0.01),
    Perturbation("pressure_offset", (), 0.01),
)


def backend_by_label(label: str) -> DerivativeBackend:
    """Return the declared backend with ``label`` or raise ``ValueError``."""
    for backend in DERIVATIVE_BACKENDS:
        if backend.label == label:
            return backend
    available = ", ".join(backend.label for backend in DERIVATIVE_BACKENDS)
    msg = f"Unknown derivative backend {label!r}. Available: {available}."
    raise ValueError(msg)


def _bind(tensor: Tensor, fields: Sequence[str]) -> dict[str, Tensor]:
    """Return channel views of a BCHW tensor keyed by declared field names."""
    if tensor.ndim != 4 or tensor.shape[1] != len(fields):  # noqa: PLR2004 -- BCHW
        msg = f"Expected a BCHW tensor with {len(fields)} channels, got {tuple(tensor.shape)}."
        raise ValueError(msg)
    return {name: tensor[:, index] for index, name in enumerate(fields)}


def input_scales(inputs: Tensor, input_fields: Sequence[str]) -> domain.physics.brinkman.BrinkmanScales:
    """Return the domain per-case scales derived from task inputs only."""
    values = _bind(inputs, input_fields)
    _, spacing_y = domain.physics.derivatives.infer_uniform_spacing(values["x"], values["y"])
    inlet = domain.physics.boundary.pressure_boundary_masks(values["y"], spacing_y).inlet
    return domain.physics.brinkman.brinkman_scales(
        values["p_in_bc"], inlet, torch.pow(10.0, values["Kxx"]), torch.pow(10.0, values["Kyy"]), values["x"], values["y"]
    )


def evaluate_residual_fields(
    inputs: Tensor,
    outputs: Tensor,
    *,
    input_fields: Sequence[str],
    output_fields: Sequence[str],
    backend: DerivativeBackend,
) -> SteadyResidualFields:
    """
    Evaluate every steady residual term and variant on the full grid.

    Parameters
    ----------
    inputs, outputs : torch.Tensor
        Task-ordered physical BCHW tensors as published by the Dataset.
    input_fields, output_fields : Sequence[str]
        Task channel declarations.
    backend : DerivativeBackend
        Derivative configuration applied to every derivative.

    Returns
    -------
    SteadyResidualFields
        Dimensionless full-grid terms and per-case boundary values.

    Raises
    ------
    ValueError
        If channel declarations or tensor geometry violate the task binding,
        or the decomposition does not reproduce the domain residual.

    """
    brinkman = domain.physics.brinkman
    operator = backend.build()
    diagnostics = brinkman.evaluate_steady_2d_brinkman(
        inputs,
        outputs,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=operator,
        boundary=domain.physics.contracts.PRESSURE_BOUNDARY_KIND,
        interior_crop=0,
    )
    values = _bind(inputs, input_fields) | _bind(outputs, output_fields)
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(values["x"], values["y"])
    permeability_xx = torch.pow(10.0, values["Kxx"])
    permeability_yy = torch.pow(10.0, values["Kyy"])
    drag = brinkman.brinkman_momentum_residuals(
        torch.zeros_like(values["p"]),
        values["u"],
        values["v"],
        values["eps_bed"],
        permeability_xx,
        values["Kxy"],
        permeability_yy,
        _ZeroDerivatives(),
        spacing_x,
        spacing_y,
    )
    zero_pressure = brinkman.brinkman_momentum_residuals(
        torch.zeros_like(values["p"]),
        values["u"],
        values["v"],
        values["eps_bed"],
        permeability_xx,
        values["Kxy"],
        permeability_yy,
        operator,
        spacing_x,
        spacing_y,
    )
    pressure_x, pressure_y = operator.gradient(values["p"], spacing_x, spacing_y)
    divergence = operator.divergence(values["u"], values["v"], spacing_x, spacing_y)
    porosity = values["eps_bed"]
    coefficient = brinkman.AIR_DYNAMIC_VISCOSITY / porosity.clamp_min(brinkman.POROSITY_FLOOR)
    trace_x, trace_y = operator.gradient(coefficient * divergence, spacing_x, spacing_y)
    porous_divergence = operator.divergence(porosity * values["u"], porosity * values["v"], spacing_x, spacing_y)

    scales = diagnostics.scales
    gradient = scales.pressure_gradient[:, None, None, None]
    residual = torch.stack((diagnostics.momentum.x, diagnostics.momentum.y), dim=1) / gradient
    pressure = -torch.stack((pressure_x, pressure_y), dim=1) / gradient
    drag_terms = torch.stack((drag.x, drag.y), dim=1) / gradient
    velocity_terms = torch.stack((zero_pressure.x, zero_pressure.y), dim=1) / gradient
    tolerance = _DECOMPOSITION_TOLERANCE_FACTOR * torch.finfo(residual.dtype).eps * max(1.0, float(pressure.abs().max()))
    if not torch.allclose(residual - velocity_terms, pressure, rtol=0.0, atol=tolerance):
        msg = "Momentum decomposition does not reproduce the domain residual: residual minus velocity terms differs from -grad(p)."
        raise ValueError(msg)
    viscous = velocity_terms - drag_terms
    trace = -TRACE_COEFFICIENT * torch.stack((trace_x, trace_y), dim=1) / gradient
    length_y = values["y"].amax(dim=(-2, -1)) - values["y"].amin(dim=(-2, -1))
    divergence_scale = (scales.velocity / length_y)[:, None, None]
    mean_porosity = porosity.mean(dim=(-2, -1))[:, None, None]
    boundary = diagnostics.boundary
    return SteadyResidualFields(
        backend=backend.label,
        pressure=pressure,
        viscous=viscous,
        drag=drag_terms,
        trace=trace,
        residual=residual,
        divergence=divergence / divergence_scale,
        porous_divergence=porous_divergence / (mean_porosity * divergence_scale),
        flux_balance=diagnostics.flux_balance,
        boundary_inlet=boundary.inlet_sample_mse / scales.pressure.square(),
        boundary_outlet=(boundary.outlet_sample_mean / scales.pressure).square(),
        scales=scales,
    )


def _crop(field: Tensor, crop: int) -> Tensor:
    """Crop ``crop`` cells from every spatial edge through the domain helper."""
    return domain.physics.derivatives.crop_interior(field, crop)


def _case_mean_square(vector_field: Tensor, crop: int) -> Tensor:
    """Return the per-case interior mean of the squared vector magnitude."""
    return _crop(vector_field, crop).square().sum(dim=1).mean(dim=(-2, -1))


def residual_rows(fields: SteadyResidualFields, crop: int) -> dict[str, np.ndarray]:
    """
    Reduce residual fields to per-case dimensionless terms at one crop.

    Parameters
    ----------
    fields : SteadyResidualFields
        Full-grid evaluation of one backend.
    crop : int
        Cells removed from every edge for momentum and pointwise divergence;
        the flux balance and the pressure boundary always use the full grid.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per-case values keyed by ``momentum``, ``momentum_with_trace``,
        ``trace_rms``, ``pointwise_divergence``, ``porous_divergence``,
        ``flux_balance``, ``boundary_inlet``, ``boundary_outlet``, ``boundary``.

    """
    values = {
        "momentum": _case_mean_square(fields.residual, crop),
        "momentum_with_trace": _case_mean_square(fields.residual_with_trace, crop),
        "trace_rms": _case_mean_square(fields.trace, crop).sqrt(),
        "pointwise_divergence": _crop(fields.divergence, crop).square().mean(dim=(-2, -1)),
        "porous_divergence": _crop(fields.porous_divergence, crop).square().mean(dim=(-2, -1)),
        "flux_balance": fields.flux_balance.square().mean(dim=-1),
        "boundary_inlet": fields.boundary_inlet,
        "boundary_outlet": fields.boundary_outlet,
        "boundary": fields.boundary_inlet + fields.boundary_outlet,
    }
    return {name: value.detach().to(torch.float64).cpu().numpy() for name, value in values.items()}


def summarize_reference_residuals(
    reference: SteadyReference,
    *,
    backends: Sequence[DerivativeBackend] = DERIVATIVE_BACKENDS,
    crops: Sequence[int] = DEFAULT_CROPS,
    dtypes: Sequence[torch.dtype] = (torch.float32, torch.float64),
) -> pd.DataFrame:
    """
    Tabulate reference residuals per case, backend, precision, and crop.

    Parameters
    ----------
    reference : SteadyReference
        Published reference cases.
    backends : Sequence[DerivativeBackend], optional
        Derivative configurations to evaluate.
    crops : Sequence[int], optional
        Interior crops of the momentum and pointwise-divergence reductions.
    dtypes : Sequence[torch.dtype], optional
        Evaluation precisions; float32 matches training.

    Returns
    -------
    pandas.DataFrame
        One row per case, backend, dtype, and crop with the residual columns
        of :func:`residual_rows`.

    """
    rows: list[dict[str, object]] = []
    for dtype in dtypes:
        inputs, outputs = reference.tensors(dtype)
        for backend in backends:
            fields = evaluate_residual_fields(
                inputs, outputs, input_fields=reference.input_fields, output_fields=reference.output_fields, backend=backend
            )
            for crop in crops:
                reduced = residual_rows(fields, crop)
                for index, case in enumerate(reference.case_labels):
                    row: dict[str, object] = {"case": case, "backend": backend.label, "dtype": str(dtype).removeprefix("torch."), "crop": crop}
                    row.update({name: float(reduced[name][index]) for name in _RESIDUAL_COLUMNS})
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_momentum_terms(fields: SteadyResidualFields, *, case_labels: Sequence[str], crop: int) -> pd.DataFrame:
    """
    Tabulate the RMS of every momentum term over one interior crop.

    Parameters
    ----------
    fields : SteadyResidualFields
        Full-grid evaluation of one backend.
    case_labels : Sequence[str]
        Labels in case order.
    crop : int
        Interior crop of the reduction.

    Returns
    -------
    pandas.DataFrame
        One row per case with ``pressure``, ``viscous``, ``drag``, ``trace``,
        ``residual``, and ``residual_with_trace`` RMS values, all divided by G.

    """
    terms = {
        "pressure": fields.pressure,
        "viscous": fields.viscous,
        "drag": fields.drag,
        "trace": fields.trace,
        "residual": fields.residual,
        "residual_with_trace": fields.residual_with_trace,
    }
    reduced = {name: _case_mean_square(value, crop).sqrt().to(torch.float64).cpu().numpy() for name, value in terms.items()}
    return pd.DataFrame(
        [
            {"case": case, "backend": fields.backend, "crop": crop, **{name: float(values[index]) for name, values in reduced.items()}}
            for index, case in enumerate(case_labels)
        ]
    )


def edge_distance(shape: Sequence[int]) -> np.ndarray:
    """Return the cell distance of every ``(y, x)`` point to the nearest grid edge."""
    height, width = (int(value) for value in shape[-2:])
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(columns, width - 1 - columns))


def band_index(shape: Sequence[int]) -> np.ndarray:
    """Return the boundary-band position of every point for :data:`BAND_LABELS`."""
    return np.searchsorted(np.asarray(BAND_EDGES), edge_distance(shape), side="right") - 1


def _band_rms(magnitude_squared: np.ndarray, bands: np.ndarray) -> list[float]:
    """Return the RMS inside every boundary band; bands without points on a small grid are NaN."""
    return [
        float(np.sqrt(magnitude_squared[..., bands == index].mean())) if bool((bands == index).any()) else math.nan
        for index in range(len(BAND_LABELS))
    ]


def residual_band_profile(fields: SteadyResidualFields, *, case_labels: Sequence[str]) -> pd.DataFrame:
    """
    Tabulate the momentum residual RMS per case and boundary band.

    Returns
    -------
    pandas.DataFrame
        Rows with ``case``, ``backend``, ``band``, ``residual_rms``, and
        ``residual_with_trace_rms``.

    """
    bands = band_index(tuple(fields.residual.shape[-2:]))
    residual = fields.residual.square().sum(dim=1).to(torch.float64).cpu().numpy()
    with_trace = fields.residual_with_trace.square().sum(dim=1).to(torch.float64).cpu().numpy()
    rows: list[dict[str, object]] = []
    for index, case in enumerate(case_labels):
        plain = _band_rms(residual[index], bands)
        traced = _band_rms(with_trace[index], bands)
        rows.extend(
            {"case": case, "backend": fields.backend, "band": label, "residual_rms": plain[position], "residual_with_trace_rms": traced[position]}
            for position, label in enumerate(BAND_LABELS)
        )
    return pd.DataFrame(rows)


def edge_profiles(magnitude: np.ndarray, *, max_distance: int = 16) -> pd.DataFrame:
    """
    Return the RMS of a field along lines parallel to each grid edge.

    Parameters
    ----------
    magnitude : numpy.ndarray
        Non-negative ``[case, y, x]`` magnitudes.
    max_distance : int, optional
        Largest cell distance from the edge. Lines exclude the corner zones
        within ``max_distance + 1`` cells of the two perpendicular edges.

    Returns
    -------
    pandas.DataFrame
        Rows with ``edge``, ``distance``, and ``rms`` over all cases.

    Raises
    ------
    ValueError
        If the grid is too small for the requested distance.

    """
    height, width = magnitude.shape[-2:]
    margin = max_distance + 1
    if min(height, width) <= 2 * margin:
        msg = "Grid is too small for the requested edge-profile distance."
        raise ValueError(msg)
    squared = np.square(magnitude, dtype=np.float64)
    lines = {
        "inlet": lambda d: squared[:, d, margin:-margin],
        "outlet": lambda d: squared[:, height - 1 - d, margin:-margin],
        "wall_x_min": lambda d: squared[:, margin:-margin, d],
        "wall_x_max": lambda d: squared[:, margin:-margin, width - 1 - d],
    }
    return pd.DataFrame(
        [
            {"edge": edge, "distance": distance, "rms": float(np.sqrt(line(distance).mean()))}
            for edge, line in lines.items()
            for distance in range(max_distance + 1)
        ]
    )


def compare_derivatives(
    reference: SteadyReference,
    *,
    dtype: torch.dtype = torch.float64,
    baseline: str = "fd",
    others: Sequence[str] = ("fc", "periodic"),
) -> pd.DataFrame:
    """
    Compare backend derivatives of the reference outputs per boundary band.

    Parameters
    ----------
    reference : SteadyReference
        Published reference cases.
    dtype : torch.dtype, optional
        Evaluation precision.
    baseline : str, optional
        Backend label used as the comparison baseline.
    others : Sequence[str], optional
        Backend labels compared against the baseline.

    Returns
    -------
    pandas.DataFrame
        Rows with ``case``, ``field``, ``direction``, ``backend``, ``band``, and
        ``relative_difference``: the band RMS of the derivative difference
        divided by the RMS of the baseline derivative in the interior band;
        NaN when that RMS is zero or the grid has no interior band. Bands
        without points are omitted.

    """
    inputs, outputs = reference.tensors(dtype)
    values = _bind(inputs, reference.input_fields)
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(values["x"], values["y"])
    bands = band_index(tuple(outputs.shape[-2:]))
    interior = bands == len(BAND_LABELS) - 1

    def derivatives(label: str) -> dict[tuple[str, str], np.ndarray]:
        operator = backend_by_label(label).build()
        result: dict[tuple[str, str], np.ndarray] = {}
        for name, field in _bind(outputs, reference.output_fields).items():
            derivative_x, derivative_y = operator.gradient(field, spacing_x, spacing_y)
            result[name, "x"] = derivative_x.to(torch.float64).cpu().numpy()
            result[name, "y"] = derivative_y.to(torch.float64).cpu().numpy()
        return result

    base = derivatives(baseline)
    rows: list[dict[str, object]] = []
    for label in others:
        other = derivatives(label)
        for (name, direction), base_values in base.items():
            difference = np.square(other[name, direction] - base_values)
            for index, case in enumerate(reference.case_labels):
                scale = math.sqrt(float(np.square(base_values[index][interior]).mean())) if bool(interior.any()) else math.nan
                for position, band in enumerate(BAND_LABELS):
                    if not bool((bands == position).any()):
                        continue
                    band_rms = math.sqrt(float(difference[index][bands == position].mean()))
                    relative = band_rms / scale if scale > 0 else math.nan
                    rows.append(
                        {"case": case, "field": name, "direction": direction, "backend": label, "band": band, "relative_difference": relative}
                    )
    return pd.DataFrame(rows)


def smooth_shape(x: Tensor, y: Tensor, *, generator: np.random.Generator, max_mode: int = SMOOTH_MAX_MODE) -> Tensor:
    """
    Return a unit-RMS low-mode cosine field on one Cartesian grid.

    Parameters
    ----------
    x, y : torch.Tensor
        Coordinate fields ``[y, x]`` of one case.
    generator : numpy.random.Generator
        Explicit random source for amplitudes and phases.
    max_mode : int, optional
        Largest half-wave count per axis.

    Returns
    -------
    torch.Tensor
        ``sum a_mn cos(pi m xi + alpha) cos(pi n eta + beta)`` over all
        ``0 <= m, n <= max_mode`` except ``m = n = 0``, normalized to unit RMS,
        with ``xi`` and ``eta`` the coordinates scaled to ``[0, 1]``.

    """
    xi = ((x - x.min()) / (x.max() - x.min())).to(torch.float64)
    eta = ((y - y.min()) / (y.max() - y.min())).to(torch.float64)
    shape = torch.zeros_like(xi)
    for m in range(max_mode + 1):
        for n in range(max_mode + 1):
            if m == 0 and n == 0:
                continue
            amplitude, phase_x, phase_y = generator.normal(), generator.uniform(0.0, 2.0 * math.pi), generator.uniform(0.0, 2.0 * math.pi)
            shape += amplitude * torch.cos(math.pi * m * xi + phase_x) * torch.cos(math.pi * n * eta + phase_y)
    return (shape / shape.square().mean().sqrt()).to(x.dtype)


def perturb_outputs(
    reference: SteadyReference,
    perturbation: Perturbation,
    *,
    dtype: torch.dtype,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """
    Return reference inputs and perturbed outputs for one controlled error.

    Parameters
    ----------
    reference : SteadyReference
        Published reference cases.
    perturbation : Perturbation
        Error definition.
    dtype : torch.dtype
        Evaluation precision.
    seed : int
        Base seed; case ``c`` and channel ``k`` draw from
        ``SeedSequence([seed, c, k])``. Ignored for deterministic kinds.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Shared reference inputs and new perturbed outputs.

    Raises
    ------
    ValueError
        If the amplitude is not finite and positive or a field is unknown.

    """
    if not math.isfinite(perturbation.amplitude) or perturbation.amplitude <= 0:
        msg = "Perturbation amplitude must be finite and positive."
        raise ValueError(msg)
    inputs, outputs = reference.tensors(dtype)
    perturbed = outputs.clone()
    names = reference.output_fields
    if perturbation.kind == "velocity_scale":
        for name in ("u", "v"):
            perturbed[:, names.index(name)] *= 1.0 + perturbation.amplitude
        return inputs, perturbed
    if perturbation.kind == "pressure_offset":
        pressure_scale = input_scales(inputs, reference.input_fields).pressure
        perturbed[:, names.index("p")] += perturbation.amplitude * pressure_scale[:, None, None]
        return inputs, perturbed
    input_values = _bind(inputs, reference.input_fields)
    unknown = [name for name in perturbation.fields if name not in names]
    if unknown:
        msg = f"Perturbation names unknown output fields: {unknown}."
        raise ValueError(msg)
    for case in range(outputs.shape[0]):
        for name in perturbation.fields:
            channel = names.index(name)
            generator = np.random.default_rng(np.random.SeedSequence([seed, case, channel]))
            field = outputs[case, channel]
            if perturbation.kind == "smooth":
                shape = smooth_shape(input_values["x"][case], input_values["y"][case], generator=generator)
            else:
                shape = torch.from_numpy(generator.standard_normal(tuple(field.shape))).to(field.dtype)
            perturbed[case, channel] += perturbation.amplitude * field.square().mean().sqrt() * shape
    return inputs, perturbed


def _ratio(value: float, floor: float) -> float:
    """Return ``value / floor``; ``inf`` for a new nonzero term and NaN when both are zero."""
    if floor > 0:
        return value / floor
    return math.inf if value > 0 else math.nan


def evaluate_sensitivity(
    reference: SteadyReference,
    *,
    settings: Sequence[tuple[str, int]] = (("fd", 0), ("fd", 2), ("fc", 2)),
    perturbations: Sequence[Perturbation] = DEFAULT_PERTURBATIONS,
    seeds: Sequence[int] = (DEFAULT_SEED, DEFAULT_SEED + 1, DEFAULT_SEED + 2, DEFAULT_SEED + 3),
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    """
    Tabulate loss terms of perturbed reference outputs against the reference floor.

    Parameters
    ----------
    reference : SteadyReference
        Published reference cases.
    settings : Sequence[tuple[str, int]], optional
        ``(backend label, interior crop)`` pairs; the defaults are the two
        maintained steady physics configurations plus FD with crop two.
    perturbations : Sequence[Perturbation], optional
        Controlled errors.
    seeds : Sequence[int], optional
        Seeds of stochastic perturbations; deterministic ones use the first.
    dtype : torch.dtype, optional
        Evaluation precision; float32 matches training.

    Returns
    -------
    pandas.DataFrame
        One row per case, setting, perturbation, and seed with the loss terms
        ``momentum``, ``flux_balance``, and ``boundary`` of the perturbed
        fields, the reference values (``*_reference``), and their ratios
        (``*_ratio``): ``inf`` when a zero reference term becomes nonzero and
        NaN when both are zero, i.e. the perturbation leaves the term unchanged.

    """
    terms = ("momentum", "flux_balance", "boundary")
    backends = sorted({label for label, _ in settings})
    inputs, outputs = reference.tensors(dtype)
    floors: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for label in backends:
        fields = evaluate_residual_fields(
            inputs, outputs, input_fields=reference.input_fields, output_fields=reference.output_fields, backend=backend_by_label(label)
        )
        for setting in settings:
            if setting[0] == label:
                floors[setting] = residual_rows(fields, setting[1])
    rows: list[dict[str, object]] = []
    for perturbation in perturbations:
        for seed in seeds if perturbation.stochastic else seeds[:1]:
            perturbed_inputs, perturbed_outputs = perturb_outputs(reference, perturbation, dtype=dtype, seed=seed)
            for label in backends:
                fields = evaluate_residual_fields(
                    perturbed_inputs,
                    perturbed_outputs,
                    input_fields=reference.input_fields,
                    output_fields=reference.output_fields,
                    backend=backend_by_label(label),
                )
                for setting in settings:
                    if setting[0] != label:
                        continue
                    reduced = residual_rows(fields, setting[1])
                    for index, case in enumerate(reference.case_labels):
                        row: dict[str, object] = {
                            "case": case,
                            "backend": label,
                            "crop": setting[1],
                            "perturbation": perturbation.label,
                            "kind": perturbation.kind,
                            "amplitude": perturbation.amplitude,
                            "seed": seed if perturbation.stochastic else None,
                        }
                        for term in terms:
                            value = float(reduced[term][index])
                            floor = float(floors[setting][term][index])
                            row[term] = value
                            row[f"{term}_reference"] = floor
                            row[f"{term}_ratio"] = _ratio(value, floor)
                        rows.append(row)
    return pd.DataFrame(rows)


def flow_regime(reference: SteadyReference, *, dtype: torch.dtype = torch.float64) -> pd.DataFrame:
    """
    Tabulate per-case scales that bound the size of the viscous terms.

    Returns
    -------
    pandas.DataFrame
        Per case: mean inlet pressure, ``G``, Darcy velocity ``U``, mean ``v``,
        the Brinkman length ``sqrt(K_g / mean(eps))`` with ``K_g = U mu / G``
        the domain geometric-mean permeability, and its ratio to the grid spacing.

    """
    inputs, outputs = reference.tensors(dtype)
    values = _bind(inputs, reference.input_fields) | _bind(outputs, reference.output_fields)
    spacing_x, _ = domain.physics.derivatives.infer_uniform_spacing(values["x"], values["y"])
    scales = input_scales(inputs, reference.input_fields)
    geometric = scales.velocity * domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY / scales.pressure_gradient
    brinkman_length = torch.sqrt(geometric / values["eps_bed"].mean(dim=(-2, -1)))
    return pd.DataFrame(
        {
            "case": list(reference.case_labels),
            "inlet_pressure_pa": scales.pressure.cpu().numpy(),
            "pressure_gradient_pa_per_m": scales.pressure_gradient.cpu().numpy(),
            "darcy_velocity_m_per_s": scales.velocity.cpu().numpy(),
            "mean_v_m_per_s": values["v"].mean(dim=(-2, -1)).cpu().numpy(),
            "brinkman_length_m": brinkman_length.cpu().numpy(),
            "brinkman_length_over_dx": (brinkman_length / spacing_x).cpu().numpy(),
        }
    )


def residual_magnitude(fields: SteadyResidualFields, *, with_trace: bool = False) -> np.ndarray:
    """Return the per-point dimensionless momentum residual magnitude ``[case, y, x]``."""
    residual = fields.residual_with_trace if with_trace else fields.residual
    return residual.square().sum(dim=1).sqrt().to(torch.float64).cpu().numpy()


def maintained_physics_settings(config_root: Path | str | None = None) -> pd.DataFrame:
    """
    Inventory the physics-loss settings of maintained steady configurations.

    Parameters
    ----------
    config_root : pathlib.Path or str, optional
        Steady-flow learning configuration directory; defaults to
        ``configs/03_learning/steady_flow`` of the repository.

    Returns
    -------
    pandas.DataFrame
        One row per experiment or Optuna study with enabled physics: source,
        derivative kind and extension, interior crop, the three target weights,
        and whether the study search space varies any physics setting.

    Notes
    -----
    Experiments are merged with their task base through the authoritative
    authored-config loader; run-time values such as Dataset identities are not
    resolved because the physics block does not depend on them.

    """
    root = Path(config_root) if config_root is not None else common.paths.get_project_root() / "configs" / "03_learning" / "steady_flow"
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.yaml")):
        if path.name == experiments.config.loader.AUTHORED_BASE_FILENAME:
            continue
        raw = experiments.config.loader.load_yaml(path)
        is_study = "experiment" in raw
        config = experiments.config.loader.apply_authored_base(raw["experiment"] if is_study else raw, origin=path)
        physics = config.get("loss", {}).get("physics", {})
        if not physics.get("enabled"):
            continue
        searched = sorted(key for key in raw.get("search_space", {}) if str(key).startswith("loss.physics"))
        rows.append(
            {
                "config": str(path.relative_to(root)),
                "kind": "optuna" if is_study else "experiment",
                "derivative": physics["derivatives"]["kind"],
                "extension": physics["derivatives"]["extension"],
                "interior_crop": int(physics["interior_crop"]),
                "momentum_weight": float(physics["momentum_weight"]["target"]),
                "continuity_weight": float(physics["continuity_weight"]["target"]),
                "boundary_weight": float(physics["boundary_weight"]["target"]),
                "searched_physics_keys": ", ".join(searched) or "none",
            }
        )
    return pd.DataFrame(rows)


def residual_overview(residuals: pd.DataFrame, *, statistic: str = "median") -> pd.DataFrame:
    """
    Return one case statistic of every residual term per backend, precision, and crop.

    Parameters
    ----------
    residuals : pandas.DataFrame
        Output of :func:`summarize_reference_residuals`.
    statistic : str, optional
        Pandas aggregation over cases, e.g. ``"median"`` or ``"max"``.

    Returns
    -------
    pandas.DataFrame
        Index ``(backend, dtype, crop)``; one column per residual term.

    """
    return residuals.groupby(["backend", "dtype", "crop"])[list(_RESIDUAL_COLUMNS)].agg(statistic)


def band_overview(bands: pd.DataFrame) -> pd.DataFrame:
    """Return the case-median momentum-residual RMS per boundary band and backend."""
    table = bands.pivot_table(index="band", columns="backend", values=["residual_rms", "residual_with_trace_rms"], aggfunc="median")
    return table.reindex(list(BAND_LABELS))


def derivative_overview(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the case-median relative derivative difference per backend, field, direction, and band."""
    table = comparison.pivot_table(index=["backend", "field", "direction"], columns="band", values="relative_difference", aggfunc="median")
    return table.reindex(columns=list(BAND_LABELS))


def sensitivity_overview(sensitivity: pd.DataFrame, *, term: str) -> pd.DataFrame:
    """
    Return the median and minimum perturbed-to-reference ratio of one loss term.

    Parameters
    ----------
    sensitivity : pandas.DataFrame
        Output of :func:`evaluate_sensitivity`.
    term : str
        ``momentum``, ``flux_balance``, or ``boundary``.

    Returns
    -------
    pandas.DataFrame
        Index perturbation in evaluation order; columns ``(statistic, backend,
        crop)`` with statistics over cases and seeds.

    """
    order = list(dict.fromkeys(sensitivity["perturbation"]))
    table = sensitivity.pivot_table(index="perturbation", columns=["backend", "crop"], values=f"{term}_ratio", aggfunc=["median", "min"])
    return table.reindex(order)


def evaluate_backends(
    reference: SteadyReference,
    *,
    backends: Sequence[DerivativeBackend] = DERIVATIVE_BACKENDS,
    dtype: torch.dtype = torch.float32,
) -> dict[str, SteadyResidualFields]:
    """Return full-grid residual fields of the reference outputs keyed by backend label."""
    inputs, outputs = reference.tensors(dtype)
    return {
        backend.label: evaluate_residual_fields(
            inputs, outputs, input_fields=reference.input_fields, output_fields=reference.output_fields, backend=backend
        )
        for backend in backends
    }


def momentum_term_table(
    fields: Mapping[str, SteadyResidualFields],
    *,
    case_labels: Sequence[str],
    crops: Sequence[int] = (0, 2),
) -> pd.DataFrame:
    """Return :func:`summarize_momentum_terms` rows for every backend and crop."""
    return pd.concat(
        [summarize_momentum_terms(item, case_labels=case_labels, crop=crop) for item in fields.values() for crop in crops],
        ignore_index=True,
    )


def residual_band_table(fields: Mapping[str, SteadyResidualFields], *, case_labels: Sequence[str]) -> pd.DataFrame:
    """Return :func:`residual_band_profile` rows for every backend."""
    return pd.concat([residual_band_profile(item, case_labels=case_labels) for item in fields.values()], ignore_index=True)
