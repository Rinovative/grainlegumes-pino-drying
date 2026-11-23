"""Compute task-resolved PyTorch metrics for training and evaluation.

Responsibilities:
  - Register semantic metric identifiers and validate their spaces, reductions and field selections
  - Compute the physical field statistics of one case: squared and absolute errors, reference and H1 energies
  - Reduce case statistics equally over the cases of a material and over the materials into per-field RMSE, MAE
    and relative norms, and into the one group objective of checkpoint selection, the scheduler, Optuna and ranking
  - Accumulate these reductions over evaluation batches from physical views

Design principles:
  - One implementation serves training validation, artifacts and notebooks
  - Every case is reduced over its grid and time first; an RMSE takes its root after the case and material
    means, a relative norm divides within each case before them
  - A reference below its declared floor makes a relative value unavailable and counted, never zero or stabilized
  - Metric implementations never own or apply normalizers

This module does NOT:
  - Construct the physical tensor views. Evaluation orchestration owns them
  - Log or persist metric results. Callers own observer and storage side effects

Notes:
  Case statistics are float64 sums per field over the rows of one case (a steady case, or the states of a
  transient case): ``count`` and the sums of squared error, absolute error and squared reference over the full
  grid; ``h1_count`` and the sums of e^2 + l^2 |grad e|^2 and f^2 + l^2 |grad f|^2 over the interior without
  H1_INTERIOR_CROP cells at every edge, with l = 1 m and non-periodic derivatives in metres (central inside,
  one-sided at the boundary, so the interior uses central differences only).
  A relative norm of one case is sqrt(error / reference); it is available when sqrt(reference / count) of the
  same norm reaches the field's reference floor. Summing a case's states before dividing gives the horizon norm.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

import numpy as np
import torch

from src import domain

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal["material_case_mean"]
MetricDirection = Literal["minimize", "maximize"]
RelativeNorm = Literal["l2", "h1"]
RELATIVE_NORMS: Final[tuple[RelativeNorm, ...]] = ("l2", "h1")
H1_LENGTH_SCALE_M: Final = 1.0
H1_INTERIOR_CROP: Final = 2
# Component key of a relative norm's unavailable case count, the longest component name a metric emits (the config loader bounds ids by it).
UNAVAILABLE_CASE_VALUES: Final = "unavailable_case_values"
CASE_STATISTICS: Final = ("count", "squared_error", "absolute_error", "reference_square", "h1_count", "h1_error", "h1_reference")
_COUNT, _SQUARED, _ABSOLUTE, _REFERENCE, _H1_COUNT, _H1_ERROR, _H1_REFERENCE = range(len(CASE_STATISTICS))
_NORM_COLUMNS: Final = {"l2": (_COUNT, _SQUARED, _REFERENCE), "h1": (_H1_COUNT, _H1_ERROR, _H1_REFERENCE)}
_RELATIVE_KINDS: Final = {"relative_l2": "l2", "relative_h1": "h1"}
_GROUP_KINDS: Final = frozenset({"group_macro_rmse", "group_rmse", "vector_rmse"})
_CASE_TENSOR_RANK: Final = 4
_SPATIAL_AXES: Final = 2


@dataclass(frozen=True, slots=True)
class MetricKindSpec:
    """
    Describe schema and optimization semantics for one metric identifier.

    Attributes
    ----------
    kind : str
        Canonical saved configuration identifier.
    spaces : frozenset[MetricSpace]
        Supported tensor spaces.
    reductions : frozenset[MetricReduction]
        Supported reduction semantics.
    direction : MetricDirection
        Required optimization direction.

    """

    kind: str
    spaces: frozenset[MetricSpace]
    reductions: frozenset[MetricReduction]
    direction: MetricDirection


_PHYSICAL: Final[frozenset[MetricSpace]] = frozenset({"physical"})
_BOTH_SPACES: Final[frozenset[MetricSpace]] = frozenset({"normalized", "physical"})
_CASE_MEAN: Final[frozenset[MetricReduction]] = frozenset({"material_case_mean"})
_METRIC_KINDS = MappingProxyType(
    {
        "relative_h1": MetricKindSpec("relative_h1", _PHYSICAL, _CASE_MEAN, "minimize"),
        "relative_l2": MetricKindSpec("relative_l2", _PHYSICAL, _CASE_MEAN, "minimize"),
        "rmse": MetricKindSpec("rmse", _BOTH_SPACES, _CASE_MEAN, "minimize"),
        "mae": MetricKindSpec("mae", _BOTH_SPACES, _CASE_MEAN, "minimize"),
        "group_macro_rmse": MetricKindSpec("group_macro_rmse", _PHYSICAL, _CASE_MEAN, "minimize"),
        "group_rmse": MetricKindSpec("group_rmse", _PHYSICAL, _CASE_MEAN, "minimize"),
        "vector_rmse": MetricKindSpec("vector_rmse", _PHYSICAL, _CASE_MEAN, "minimize"),
    }
)


def available_metric_kinds() -> tuple[str, ...]:
    """
    Return registered semantic metric identifiers.

    Returns
    -------
    tuple[str, ...]
        Exact metric kinds accepted by the registry.

    """
    return tuple(sorted(_METRIC_KINDS))


def resolve_metric_kind(kind: str) -> MetricKindSpec:
    """
    Resolve an exact semantic metric identifier.

    Parameters
    ----------
    kind : str
        Canonical metric kind.

    Returns
    -------
    MetricKindSpec
        Immutable metric-space, reduction, and direction descriptor.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    try:
        return _METRIC_KINDS[kind]
    except KeyError as error:
        available = ", ".join(available_metric_kinds())
        msg = f"Unknown metric identifier {kind!r}. Available metrics: {available}."
        raise ValueError(msg) from error


def validate_metric_semantics(
    kind: str,
    *,
    space: str,
    reduction: str,
) -> MetricKindSpec:
    """
    Validate a metric space and reduction against its registry entry.

    Parameters
    ----------
    kind : str
        Canonical metric kind.
    space : str
        Requested normalized or physical tensor space.
    reduction : str
        Requested dataset reduction identifier.

    Returns
    -------
    MetricKindSpec
        Validated semantic metric descriptor.

    Raises
    ------
    ValueError
        If the metric kind, space, or reduction is unsupported.

    """
    spec = resolve_metric_kind(kind)
    if space not in spec.spaces:
        msg = f"Metric {kind!r} does not support space {space!r}. Expected one of {sorted(spec.spaces)}."
        raise ValueError(msg)
    if reduction not in spec.reductions:
        msg = f"Metric {kind!r} does not support reduction {reduction!r}. Expected one of {sorted(spec.reductions)}."
        raise ValueError(msg)
    return spec


def resolve_metric_groups(
    task: domain.tasks.spec.TaskSpec,
    *,
    kind: str,
    fields: tuple[str, ...],
    metric_id: str,
) -> tuple[domain.tasks.spec.OutputGroupSpec, ...]:
    """
    Validate one metric's field selection against the task and return the output groups it reduces over.

    Group metrics select every metric field (``group_macro_rmse``) or one complete output group
    (``group_rmse``, ``vector_rmse``, the latter of one unit); ``rmse`` and ``mae`` select one field;
    a relative norm selects one field, or every metric field for its group-weighted summary.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task contract.
    kind : str
        Registered metric kind.
    fields : tuple[str, ...]
        Ordered selected metric fields.
    metric_id : str
        Metric identifier for diagnostics.

    Returns
    -------
    tuple[domain.tasks.spec.OutputGroupSpec, ...]
        The task groups of a group metric or of a relative-norm summary; empty for one-field metrics.

    Raises
    ------
    ValueError
        If the selection contradicts the metric kind or the task lacks the groups, units or floors it needs.

    """
    all_fields = fields == task.metric_names
    if not task.output_groups and (kind in _GROUP_KINDS or (kind in _RELATIVE_KINDS and all_fields and len(fields) > 1)):
        msg = f"Metric {metric_id!r} requires task-owned output groups."
        raise ValueError(msg)
    if kind in _RELATIVE_KINDS and not task.reference_floors:
        msg = f"Metric {metric_id!r} requires task-owned reference floors."
        raise ValueError(msg)
    if kind == "group_macro_rmse" or (kind in _RELATIVE_KINDS and len(fields) > 1):
        if not all_fields:
            choice = "one field or " if kind in _RELATIVE_KINDS else ""
            msg = f"Metric {metric_id!r} must select {choice}every TaskSpec metric field in declared order: {list(task.metric_names)}."
            raise ValueError(msg)
        return task.output_groups
    if kind in {"group_rmse", "vector_rmse"}:
        matches = tuple(group for group in task.output_groups if group.fields == fields)
        if len(matches) != 1:
            available = {group.id: list(group.fields) for group in task.output_groups}
            msg = f"Metric {metric_id!r} fields must match one complete task output group. Available groups: {available}."
            raise ValueError(msg)
        units = {task.metric_field(field).unit for field in fields}
        if kind == "vector_rmse" and len(units) != 1:
            msg = f"Physical vector metric {metric_id!r} fields must share one unit, got {sorted(units)}."
            raise ValueError(msg)
        return matches
    if len(fields) != 1:
        selected_units = sorted({task.metric_field(field).unit for field in fields})
        msg = f"Metric {metric_id!r} must select exactly one field; selected units are {selected_units}."
        raise ValueError(msg)
    return ()


# ============================================================================
# Case statistics and their material-case reductions
# ============================================================================


def _validated_spacing(grid_spacing: Sequence[float]) -> tuple[float, float]:
    """Return two positive finite grid spacings in metres."""
    spacing = tuple(float(value) for value in grid_spacing)
    if len(spacing) != _SPATIAL_AXES or not all(math.isfinite(value) and value > 0.0 for value in spacing):
        msg = f"grid_spacing must hold two positive finite spacings in metres (rows, then columns), got {list(grid_spacing)}."
        raise ValueError(msg)
    return spacing[0], spacing[1]


def _h1_energy(values: torch.Tensor, spacing: tuple[float, float]) -> torch.Tensor:
    """Return the interior sum of f^2 + l^2 |grad f|^2 per row and field."""
    rows_gradient, columns_gradient = torch.gradient(values, spacing=spacing, dim=(-2, -1))
    density = values.square() + H1_LENGTH_SCALE_M**2 * (rows_gradient.square() + columns_gradient.square())
    crop = H1_INTERIOR_CROP
    return density[..., crop:-crop, crop:-crop].sum(dim=(-2, -1))


def field_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    grid_spacing: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Return the float64 error and reference sums of every row and field.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Matching physical (or Train-scaled) ``(rows, fields, Y, X)`` tensors; a row is one
        steady case or one state of a transient case.
    grid_spacing : Sequence[float] | None, optional
        Grid spacing in metres, rows then columns. Without it the H1 statistics stay zero
        with ``h1_count`` 0.

    Returns
    -------
    numpy.ndarray
        ``(rows, fields, len(CASE_STATISTICS))`` sums in :data:`CASE_STATISTICS` order.

    Raises
    ------
    ValueError
        If shapes differ, the rank is not four, the spacing is invalid, or the grid has no H1 interior.

    """
    if prediction.shape != target.shape or prediction.ndim != _CASE_TENSOR_RANK:
        msg = f"Field statistics require matching (rows, fields, Y, X) tensors, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        raise ValueError(msg)
    error = prediction.double() - target.double()
    reference = target.double()
    height, width = error.shape[-2:]
    result = torch.zeros((*error.shape[:2], len(CASE_STATISTICS)), dtype=torch.float64, device=error.device)
    result[..., _COUNT] = float(height * width)
    result[..., _SQUARED] = error.square().sum(dim=(-2, -1))
    result[..., _ABSOLUTE] = error.abs().sum(dim=(-2, -1))
    result[..., _REFERENCE] = reference.square().sum(dim=(-2, -1))
    if grid_spacing is not None:
        spacing = _validated_spacing(grid_spacing)
        if min(height, width) <= 2 * H1_INTERIOR_CROP:
            msg = f"The H1 interior needs more than {2 * H1_INTERIOR_CROP} grid points per axis, got {height} x {width}."
            raise ValueError(msg)
        result[..., _H1_COUNT] = float((height - 2 * H1_INTERIOR_CROP) * (width - 2 * H1_INTERIOR_CROP))
        result[..., _H1_ERROR] = _h1_energy(error, spacing)
        result[..., _H1_REFERENCE] = _h1_energy(reference, spacing)
    return result.detach().cpu().numpy()


@dataclass(frozen=True, slots=True)
class CaseErrors:
    """
    Hold the physical mean squared error of every field of one case, averaged over its grid and its time.

    Attributes
    ----------
    case_id, material : str
        Case and material identity.
    field_mse : Mapping[str, float]
        Finite non-negative physical mean squared error per field.

    """

    case_id: str
    material: str
    field_mse: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CaseStatistics:
    """
    Bind the summed field statistics of one case to its identity.

    Attributes
    ----------
    case_id, material : str
        Case and material identity.
    fields : tuple[str, ...]
        Field order of ``sums``.
    sums : numpy.ndarray
        ``(fields, len(CASE_STATISTICS))`` float64 sums over every row of the case.

    """

    case_id: str
    material: str
    fields: tuple[str, ...]
    sums: np.ndarray

    def __post_init__(self) -> None:
        """Require finite non-negative sums with a positive element count for every field."""
        sums = np.asarray(self.sums, dtype=np.float64)
        if sums.shape != (len(self.fields), len(CASE_STATISTICS)) or not np.isfinite(sums).all() or np.any(sums < 0.0):
            msg = f"Case {self.case_id!r} statistics must be finite non-negative ({len(self.fields)}, {len(CASE_STATISTICS)}) sums."
            raise ValueError(msg)
        if np.any(sums[:, _COUNT] <= 0.0):
            msg = f"Case {self.case_id!r} statistics need a positive element count for every field."
            raise ValueError(msg)

    def errors(self) -> CaseErrors:
        """Return the case's physical mean squared error per field."""
        return CaseErrors(self.case_id, self.material, dict(zip(self.fields, (self.sums[:, _SQUARED] / self.sums[:, _COUNT]).tolist(), strict=True)))


def case_relative_norm(sums: np.ndarray, *, norm: RelativeNorm, floor: float) -> float | None:
    """
    Return one case's relative norm of one field, or ``None`` below the reference floor.

    Parameters
    ----------
    sums : numpy.ndarray
        One field's :data:`CASE_STATISTICS` sums of one case (all its rows summed).
    norm : {"l2", "h1"}
        Relative norm to form.
    floor : float
        The field's reference-RMS floor in its physical unit.

    Returns
    -------
    float | None
        ``sqrt(error / reference)``, or ``None`` when ``sqrt(reference / count)`` of this
        norm is below ``floor``.

    Raises
    ------
    ValueError
        If the H1 statistics were not computed.

    """
    count_index, error_index, reference_index = _NORM_COLUMNS[norm]
    count = float(sums[count_index])
    if count <= 0.0:
        msg = f"Relative {norm.upper()} requires its statistics; compute them with the grid spacing."
        raise ValueError(msg)
    reference = float(sums[reference_index])
    if math.sqrt(reference / count) < floor:
        return None
    return math.sqrt(float(sums[error_index]) / reference)


def material_case_mean(values: Iterable[tuple[str, float | None]]) -> float | None:
    """
    Return the mean over materials of each material's mean available case value.

    Parameters
    ----------
    values : Iterable[tuple[str, float | None]]
        ``(material, value)`` of every case; ``None`` marks an unavailable case value.

    Returns
    -------
    float | None
        The equal-material mean, or ``None`` when a material has no available case.

    Raises
    ------
    ValueError
        If no case is given.

    """
    by_material: dict[str, list[float]] = {}
    for material, value in values:
        members = by_material.setdefault(material, [])
        if value is not None:
            members.append(float(value))
    if not by_material:
        msg = "A material-case mean requires at least one case."
        raise ValueError(msg)
    if any(not members for members in by_material.values()):
        return None
    return float(np.mean([np.mean(members) for members in by_material.values()]))


def _population(cases: Iterable[CaseStatistics]) -> tuple[tuple[CaseStatistics, ...], tuple[str, ...]]:
    """Return unique cases of one field order."""
    population = tuple(cases)
    if not population:
        msg = "Field metrics require at least one case."
        raise ValueError(msg)
    if len({case.case_id for case in population}) != len(population):
        msg = "A case enters one field-metric population twice."
        raise ValueError(msg)
    fields = population[0].fields
    if any(case.fields != fields for case in population):
        msg = "Field-metric cases must share one field order."
        raise ValueError(msg)
    return population, fields


@dataclass(frozen=True, slots=True)
class FieldErrors:
    """
    Hold the per-field errors of one evaluation population.

    Attributes
    ----------
    mse, mae : Mapping[str, float]
        Mean squared and mean absolute error per field in the space of the statistics,
        each case reduced over its grid and time, then equal over the cases of a material
        and over the materials; the RMSE is the root of ``mse``.
    case_count, material_count : int
        Population sizes.

    """

    mse: Mapping[str, float]
    mae: Mapping[str, float]
    case_count: int
    material_count: int


def field_errors(cases: Iterable[CaseStatistics]) -> FieldErrors:
    """Return the material-case mean squared and absolute error of every field."""
    population, fields = _population(cases)
    mse: dict[str, float] = {}
    mae: dict[str, float] = {}
    for index, field in enumerate(fields):
        for target, column in ((mse, _SQUARED), (mae, _ABSOLUTE)):
            value = material_case_mean((case.material, float(case.sums[index, column] / case.sums[index, _COUNT])) for case in population)
            target[field] = cast("float", value)
    return FieldErrors(MappingProxyType(mse), MappingProxyType(mae), len(population), len({case.material for case in population}))


@dataclass(frozen=True, slots=True)
class RelativeNorms:
    """
    Hold one relative norm per field of one evaluation population with its support.

    Attributes
    ----------
    norm : {"l2", "h1"}
        The relative norm.
    by_field : Mapping[str, float | None]
        Per-case norm, averaged equally over the supported cases of a material and over
        the materials; ``None`` when a material has no supported case.
    supported, unavailable : Mapping[str, int]
        Cases whose reference reaches, or falls below, the field's floor.

    """

    norm: RelativeNorm
    by_field: Mapping[str, float | None]
    supported: Mapping[str, int]
    unavailable: Mapping[str, int]


def relative_norms(cases: Iterable[CaseStatistics], *, norm: RelativeNorm, floors: Mapping[str, float]) -> RelativeNorms:
    """
    Return the material-case mean of the per-case relative norm of every field.

    Parameters
    ----------
    cases : Iterable[CaseStatistics]
        Unique cases of one evaluation role (or one time point) with physical statistics.
    norm : {"l2", "h1"}
        Relative norm to form.
    floors : Mapping[str, float]
        Reference-RMS floor of every field.

    Returns
    -------
    RelativeNorms
        Values with supported and unavailable case counts.

    """
    population, fields = _population(cases)
    values: dict[str, float | None] = {}
    supported: dict[str, int] = {}
    for index, field in enumerate(fields):
        per_case = [(case.material, case_relative_norm(case.sums[index], norm=norm, floor=floors[field])) for case in population]
        values[field] = material_case_mean(per_case)
        supported[field] = sum(value is not None for _material, value in per_case)
    return RelativeNorms(
        norm=norm,
        by_field=MappingProxyType(values),
        supported=MappingProxyType(supported),
        unavailable=MappingProxyType({field: len(population) - count for field, count in supported.items()}),
    )


def group_weighted_mean(values: Mapping[str, float | None], groups: Sequence[domain.tasks.spec.OutputGroupSpec]) -> float | None:
    """
    Average per-field values within each output group, then the groups with equal group weights.

    Returns ``None`` when any field value is unavailable; a plain field mean would count a
    vector group or a two-compartment group twice.
    """
    if not groups:
        msg = "A group-weighted mean requires at least one output group."
        raise ValueError(msg)
    selected = [values.get(field) for group in groups for field in group.fields]
    if any(value is None for value in selected):
        return None
    return float(sum(np.mean([cast("float", values[field]) for field in group.fields]) for group in groups) / len(groups))


def field_metric_values(
    cases: Iterable[CaseStatistics],
    *,
    train_scales: Mapping[str, float],
    reference_floors: Mapping[str, float],
    output_groups: Sequence[domain.tasks.spec.OutputGroupSpec],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """
    Return every :func:`domain.tasks.spec.field_metric_specs` value of one population with its unavailable counts.

    Parameters
    ----------
    cases : Iterable[CaseStatistics]
        Unique cases of one evaluation role (or one time point) with physical statistics.
    train_scales, reference_floors : Mapping[str, float]
        Train state scale and reference-RMS floor of every field.
    output_groups : Sequence[OutputGroupSpec]
        Groups of the summarising relative norms.

    Returns
    -------
    tuple[dict[str, float | None], dict[str, int]]
        Metric id to value (``None`` when unavailable), and every relative metric id to
        its number of cases below the reference floor.

    """
    population = tuple(cases)
    errors = field_errors(population)
    norms = {norm: relative_norms(population, norm=norm, floors=reference_floors) for norm in RELATIVE_NORMS}
    values: dict[str, float | None] = {}
    unavailable: dict[str, int] = {}
    for field in population[0].fields:
        rmse, mae = math.sqrt(errors.mse[field]), errors.mae[field]
        values.update({f"physical_rmse_{field}": rmse, f"normalized_rmse_{field}": rmse / train_scales[field]})
        values.update({f"physical_mae_{field}": mae, f"normalized_mae_{field}": mae / train_scales[field]})
        for norm, result in norms.items():
            values[f"physical_relative_{norm}_{field}"] = result.by_field[field]
            unavailable[f"physical_relative_{norm}_{field}"] = result.unavailable[field]
    for norm, result in norms.items():
        values[f"physical_relative_{norm}"] = group_weighted_mean(result.by_field, output_groups)
        unavailable[f"physical_relative_{norm}"] = sum(result.unavailable.values())
    return values, unavailable


# ============================================================================
# Group objective
# ============================================================================


@dataclass(frozen=True, slots=True)
class GroupObjective:
    """
    Hold the group-weighted error of one evaluation population.

    Attributes
    ----------
    field_mse : Mapping[str, float]
        Physical mean squared error per field, equal over the cases of a material, then over the materials.
    term_rmse : Mapping[str, float]
        Train-scaled RMSE per error term.
    physical_term_rmse : Mapping[str, float]
        Physical RMSE per error term (the vector RMSE for a vector term).
    value : float
        Weighted sum of the term RMSEs; lower is better.
    case_count, material_count : int
        Population sizes behind the value.

    """

    field_mse: Mapping[str, float]
    term_rmse: Mapping[str, float]
    physical_term_rmse: Mapping[str, float]
    value: float
    case_count: int
    material_count: int


def _positive_scale(train_scales: Mapping[str, float], field: str) -> float:
    """Return one finite positive Train scale."""
    raw = train_scales.get(field)
    if isinstance(raw, bool) or not isinstance(raw, Real) or not np.isfinite(float(raw)) or float(raw) <= 0.0:
        msg = f"Field {field!r} Train scale must be a finite positive real number."
        raise ValueError(msg)
    return float(raw)


def term_objective(
    field_mse: Mapping[str, float],
    *,
    terms: Sequence[domain.tasks.spec.ErrorTermSpec],
    train_scales: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Return the Train-scaled and physical RMSE per error term and their weighted sum.

    Each term takes one root of its fields' summed physical mean squared errors divided
    by the sum of their squared Train scales, so a vector term keeps its Euclidean
    geometry (``(E_u + E_v) / (s_u^2 + s_v^2)``); the value is the weighted sum of the
    scaled term RMSEs.

    Parameters
    ----------
    field_mse : Mapping[str, float]
        Physical mean squared error of every term field.
    terms : Sequence[domain.tasks.spec.ErrorTermSpec]
        Task-owned weighted error terms.
    train_scales : Mapping[str, float]
        Train state scale ``max(sigma, floor)`` of every term field, the scale of the loss.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], float]
        Scaled term RMSE, physical term RMSE, and the weighted value.

    Raises
    ------
    ValueError
        If a field error or scale is missing, negative, or non-finite, or no term is given.

    """
    if not terms:
        msg = "The group objective requires at least one error term."
        raise ValueError(msg)
    scaled: dict[str, float] = {}
    physical: dict[str, float] = {}
    for term in terms:
        errors = [float(field_mse.get(field, math.nan)) for field in term.fields]
        if not all(np.isfinite(error) and error >= 0.0 for error in errors):
            msg = f"Error term {term.id!r} requires finite non-negative mean squared errors of {list(term.fields)}."
            raise ValueError(msg)
        scale_square = sum(_positive_scale(train_scales, field) ** 2 for field in term.fields)
        scaled[term.id] = math.sqrt(sum(errors) / scale_square)
        physical[term.id] = math.sqrt(sum(errors))
    value = sum(term.weight * scaled[term.id] for term in terms)
    if not np.isfinite(value):
        msg = "The group objective finalized to a non-finite value."
        raise ValueError(msg)
    return scaled, physical, float(value)


def group_objective(
    cases: Iterable[CaseErrors],
    *,
    terms: Sequence[domain.tasks.spec.ErrorTermSpec],
    train_scales: Mapping[str, float],
) -> GroupObjective:
    """
    Aggregate case errors into the selection objective shared by checkpoint selection, Optuna and ranking.

    Each case's physical squared errors are already averaged over its grid and time,
    so time counts as ``t / t_stop`` of that case. Field errors are then averaged
    equally over the cases of each material and equally over the materials; only
    then does each term take its root and the terms their weights (training is MSE,
    selection the weighted sum of term RMSEs).

    Parameters
    ----------
    cases : Iterable[CaseErrors]
        Unique cases of one evaluation role, each with errors for every term field.
    terms : Sequence[domain.tasks.spec.ErrorTermSpec]
        Task-owned weighted error terms.
    train_scales : Mapping[str, float]
        Train state scale ``max(sigma, floor)`` of every term field.

    Returns
    -------
    GroupObjective
        Population field errors, term RMSEs, and the objective value.

    Raises
    ------
    ValueError
        If no case is given, a case repeats, or errors or scales are invalid.

    """
    fields = tuple(dict.fromkeys(field for term in terms for field in term.fields))
    population = tuple(cases)
    if len({case.case_id for case in population}) != len(population):
        msg = "A case enters the group objective twice."
        raise ValueError(msg)
    if not population:
        msg = "The group objective requires at least one case."
        raise ValueError(msg)
    field_mse = {
        field: cast("float", material_case_mean((case.material, float(case.field_mse.get(field, math.nan))) for case in population))
        for field in fields
    }
    scaled, physical, value = term_objective(field_mse, terms=terms, train_scales=train_scales)
    return GroupObjective(
        field_mse=MappingProxyType(field_mse),
        term_rmse=MappingProxyType(scaled),
        physical_term_rmse=MappingProxyType(physical),
        value=value,
        case_count=len(population),
        material_count=len({case.material for case in population}),
    )


# ============================================================================
# Explicit-space dataset metric accumulators
# ============================================================================


@dataclass(frozen=True, slots=True)
class ResolvedMetric:
    """
    Describe one task-resolved evaluation metric and its reduction contract.

    Attributes
    ----------
    id, kind : str
        Stable config identifier and registered implementation kind.
    space : {"normalized", "physical"}
        Tensor representation required by accumulation; a normalized RMSE or MAE accumulates physical
        statistics and divides by its Train scale, the definition of :func:`field_metric_values`.
    fields : tuple[str, ...]
        Exact TaskSpec metric fields included in declared order.
    field_indices : tuple[int, ...]
        Corresponding channel indices in the task metric tensor.
    reduction : str
        The material-case reduction.
    direction : {"minimize", "maximize"}
        Selection direction when used as an objective.
    unit : str
        Task-owned physical unit or dimensionless ``1``.
    groups : tuple[domain.tasks.spec.OutputGroupSpec, ...]
        Task-owned groups of a group metric or of a relative-norm summary.
    field_scales : tuple[float, ...]
        Train state scales max(sigma, floor) aligned with ``fields`` for group metrics and normalized field errors.
    reference_floors : tuple[float, ...]
        Reference-RMS floors aligned with ``fields`` for relative norms.

    """

    id: str
    kind: str
    space: MetricSpace
    fields: tuple[str, ...]
    field_indices: tuple[int, ...]
    reduction: MetricReduction
    direction: MetricDirection
    unit: str
    groups: tuple[domain.tasks.spec.OutputGroupSpec, ...]
    field_scales: tuple[float, ...]
    reference_floors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous field declarations before tensor accumulation."""
        if not self.fields or len(self.fields) != len(set(self.fields)):
            msg = f"Metric {self.id!r} fields must be unique and non-empty."
            raise ValueError(msg)
        if len(self.field_indices) != len(self.fields):
            msg = f"Metric {self.id!r} field names and channel indices must have equal length."
            raise ValueError(msg)
        if len(self.field_indices) != len(set(self.field_indices)) or any(index < 0 for index in self.field_indices):
            msg = f"Metric {self.id!r} field indices must be unique non-negative integers."
            raise ValueError(msg)
        if any(values and len(values) != len(self.fields) for values in (self.field_scales, self.reference_floors)):
            msg = f"Metric {self.id!r} Train scales and reference floors must align with selected fields."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CaseMembership:
    """
    Bind every row of one evaluation batch to its case and material.

    Attributes
    ----------
    case_ids, materials : tuple[str, ...]
        Row-aligned simulation case and material identities.

    """

    case_ids: tuple[str, ...]
    materials: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require one non-empty case and material identity per row."""
        if len(self.case_ids) != len(self.materials) or not all(isinstance(value, str) and value for value in (*self.case_ids, *self.materials)):
            msg = "Case membership requires one non-empty case and material identity per batch row."
            raise ValueError(msg)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> CaseMembership:
        """Read the collated ``simulation_case_id`` and ``material_family`` lists of one loader batch."""
        try:
            return cls(tuple(metadata["simulation_case_id"]), tuple(metadata["material_family"]))
        except (KeyError, TypeError) as error:
            msg = "Evaluation batches must carry simulation_case_id and material_family metadata for every case."
            raise ValueError(msg) from error

    def repeat(self, count: int) -> CaseMembership:
        """Return the membership of rows flattened from ``count`` consecutive entries per row, such as the states of one trajectory."""
        return CaseMembership(
            tuple(case for case in self.case_ids for _ in range(count)),
            tuple(material for material in self.materials for _ in range(count)),
        )


class DatasetMetric:
    """
    Accumulate one explicit-space dataset metric over complete evaluation populations.

    Implementations validate tensor space, shape ``(batch, channel, *spatial)``,
    concrete device, finiteness, and TaskSpec field selection on every update.
    Callers must reset once, update with every evaluation batch, and compute only
    after the complete dataset. Batching must not change the final value.

    Parameters
    ----------
    definition : ResolvedMetric
        Immutable semantic fields, space, reduction, direction, and unit.
    device : torch.device
        Concrete device on which every update tensor must reside.

    Raises
    ------
    TypeError
        If ``device`` is not a concrete CPU or CUDA ``torch.device``.

    Notes
    -----
    Accumulators own only case statistics. They never normalize tensors, transfer
    devices, persist values, or infer field/unit semantics. Metrics with
    ``requires_cases`` receive the batch's :class:`CaseMembership`.

    """

    requires_cases: ClassVar[bool] = True

    def __init__(self, definition: ResolvedMetric, *, device: torch.device, grid_spacing: Sequence[float] | None = None) -> None:
        """Validate device ownership and initialize empty case statistics; ``grid_spacing`` enables the H1 statistics."""
        if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
            msg = f"Metric construction requires one concrete CPU or CUDA torch.device, got {device!r}."
            raise TypeError(msg)
        self.definition = definition
        self.device = device
        self._grid_spacing = None if grid_spacing is None else _validated_spacing(grid_spacing)
        self.reset()

    @property
    def id(self) -> str:
        """Return the configured metric identifier."""
        return self.definition.id

    @property
    def space(self) -> MetricSpace:
        """Return the tensor space this metric requires."""
        return self.definition.space

    @property
    def fields(self) -> tuple[str, ...]:
        """Return exact selected task output fields."""
        return self.definition.fields

    @property
    def unit(self) -> str:
        """Return the task-owned physical unit or dimensionless unit ``1``."""
        return self.definition.unit

    def reset(self) -> None:
        """Clear the per-case statistics."""
        self._cases: dict[str, tuple[str, np.ndarray]] = {}

    def _validate_update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate one batch view and select the declared task channels.

        Prediction and target must share shape ``(batch, channel, Y, X)``, concrete
        device, requested tensor space, and finite values. Failures include metric
        and batch identity.
        """
        if space != self.space:
            msg = f"Metric {self.id!r} expects {self.space!r} tensors, got {space!r}."
            raise ValueError(msg)
        if pred.shape != target.shape:
            msg = f"Metric {self.id!r} prediction/target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}."
            raise ValueError(msg)
        if pred.device != self.device or target.device != self.device:
            msg = f"Metric {self.id!r} requires tensors on resolved device {self.device}, got prediction={pred.device} and target={target.device}."
            raise ValueError(msg)
        if pred.ndim != _CASE_TENSOR_RANK:
            msg = f"Metric {self.id!r} requires batch, channel, and two spatial axes."
            raise ValueError(msg)
        if not bool(torch.isfinite(pred).all().item()) or not bool(torch.isfinite(target).all().item()):
            msg = f"Metric {self.id!r} received non-finite values in evaluation batch {batch_index}."
            raise FloatingPointError(msg)
        maximum_index = max(self.definition.field_indices)
        if pred.shape[1] <= maximum_index:
            msg = f"Metric {self.id!r} field index {maximum_index} exceeds {pred.shape[1]} channels."
            raise ValueError(msg)
        indices = torch.tensor(self.definition.field_indices, device=pred.device)
        return pred.index_select(1, indices), target.index_select(1, indices)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
        cases: CaseMembership | None = None,
    ) -> None:
        """Add the field statistics of every row to its case; ``cases`` is required."""
        selected_pred, selected_target = self._validate_update(pred, target, space=space, batch_index=batch_index)
        if not isinstance(cases, CaseMembership) or len(cases.case_ids) != selected_pred.shape[0]:
            msg = f"Metric {self.id!r} requires one case membership per batch row."
            raise ValueError(msg)
        rows = field_statistics(selected_pred, selected_target, grid_spacing=self._grid_spacing)
        if not np.isfinite(rows).all():
            msg = f"Metric {self.id!r} produced non-finite statistics in evaluation batch {batch_index}."
            raise FloatingPointError(msg)
        for row, (case_id, material) in enumerate(zip(cases.case_ids, cases.materials, strict=True)):
            known_material, sums = self._cases.get(case_id, (material, np.zeros(rows.shape[1:])))
            if known_material != material:
                msg = f"Case {case_id!r} changed its material within one evaluation."
                raise ValueError(msg)
            self._cases[case_id] = (material, sums + rows[row])

    def statistics(self) -> tuple[CaseStatistics, ...]:
        """Return the statistics of every case accumulated since the last reset."""
        if not self._cases:
            msg = f"Metric {self.id!r} cannot finalize without samples."
            raise RuntimeError(msg)
        return tuple(CaseStatistics(case_id, material, self.fields, sums) for case_id, (material, sums) in self._cases.items())

    def compute(self) -> float:
        """Finalize one dataset metric after all batches."""
        raise NotImplementedError

    def components(self) -> dict[str, float]:
        """Return named finalized components reported next to the value; none by default."""
        return {}


def finalize_metrics(metrics: Mapping[str, DatasetMetric]) -> dict[str, float]:
    """Return every metric value and, under ``<id>/component/<name>``, its components, after all batches."""
    result = {metric_id: float(metric.compute()) for metric_id, metric in metrics.items()}
    for metric_id, metric in metrics.items():
        result.update({f"{metric_id}/component/{name}": float(value) for name, value in metric.components().items()})
    return result


class FieldRMSEMetric(DatasetMetric):
    """Compute one field's RMSE: per-case mean squared error, equal over cases and materials, then the root."""

    def compute(self) -> float:
        """Return the root of the material-case mean squared error, over the field's Train scale when it has one (normalized)."""
        rmse = math.sqrt(field_errors(self.statistics()).mse[self.fields[0]])
        return rmse / self.definition.field_scales[0] if self.definition.field_scales else rmse


class FieldMAEMetric(DatasetMetric):
    """Compute one field's MAE: per-case mean absolute error, equal over cases and materials."""

    def compute(self) -> float:
        """Return the material-case mean absolute error, over the field's Train scale when it has one (normalized)."""
        mae = field_errors(self.statistics()).mae[self.fields[0]]
        return mae / self.definition.field_scales[0] if self.definition.field_scales else mae


class RelativeNormMetric(DatasetMetric):
    """
    Compute one field's relative L2 or H1 norm, or their group-weighted summary over all fields.

    Each case's norm is formed from its own error and reference energies (summed over its
    states), then averaged equally over the cases of a material and over the materials. A
    case below the field's reference floor is unavailable and counted; when a material has
    no supported case, or the summary lacks a field, the value is NaN.
    """

    def __init__(self, definition: ResolvedMetric, *, device: torch.device, grid_spacing: Sequence[float] | None = None) -> None:
        """Bind the relative norm of the metric kind; H1 requires the grid spacing."""
        self.norm: RelativeNorm = cast("RelativeNorm", _RELATIVE_KINDS[definition.kind])
        if self.norm == "h1" and grid_spacing is None:
            msg = f"Metric {definition.id!r} requires grid_spacing in metres (rows, then columns) from the physical Dataset grid."
            raise ValueError(msg)
        super().__init__(definition, device=device, grid_spacing=grid_spacing if self.norm == "h1" else None)

    def _norms(self) -> RelativeNorms:
        """Return the population norms of the selected fields."""
        return relative_norms(self.statistics(), norm=self.norm, floors=dict(zip(self.fields, self.definition.reference_floors, strict=True)))

    def compute(self) -> float:
        """Return the field value or the group-weighted summary; NaN when unavailable."""
        norms = self._norms()
        value = norms.by_field[self.fields[0]] if len(self.fields) == 1 else group_weighted_mean(norms.by_field, self.definition.groups)
        return math.nan if value is None else value

    def components(self) -> dict[str, float]:
        """Return the number of unavailable case values below the reference floors."""
        return {UNAVAILABLE_CASE_VALUES: float(sum(self._norms().unavailable.values()))}


class _CaseGroupMetric(DatasetMetric):
    """Finalize physical per-case squared errors through :func:`group_objective`."""

    def __init__(self, definition: ResolvedMetric, *, device: torch.device) -> None:
        """Bind the error terms of the selected task-owned output groups."""
        super().__init__(definition, device=device)
        self.terms = domain.tasks.spec.error_terms(definition.groups)

    def objective(self) -> GroupObjective:
        """Return the population objective of every case accumulated since the last reset."""
        return group_objective(
            (case.errors() for case in self.statistics()),
            terms=self.terms,
            train_scales=dict(zip(self.fields, self.definition.field_scales, strict=True)),
        )

    def _single_term(self) -> str:
        """Return the id of the one error term a single-group metric reports."""
        if len(self.terms) != 1:
            msg = f"Metric {self.id!r} requires one vector or one-field output group."
            raise RuntimeError(msg)
        return self.terms[0].id


class GroupMacroRMSEMetric(_CaseGroupMetric):
    """Compute the selection objective: the weighted sum of the task's term RMSEs over the material-case population."""

    def compute(self) -> float:
        """Return the objective value."""
        return self.objective().value

    def components(self) -> dict[str, float]:
        """Return the scaled RMSE of every term and, for a group split into several terms, their weighted group mean."""
        objective = self.objective()
        values = dict(objective.term_rmse)
        for group in self.definition.groups:
            members = [term for term in self.terms if set(term.fields) <= set(group.fields)]
            if len(members) > 1:
                values[group.id] = sum(term.weight * values[term.id] for term in members) / sum(term.weight for term in members)
        return values


class GroupRMSEMetric(_CaseGroupMetric):
    """Compute one Train-scaled output-group RMSE over the material-case population."""

    def compute(self) -> float:
        """Return the group's scaled term RMSE."""
        return self.objective().term_rmse[self._single_term()]


class VectorRMSEMetric(_CaseGroupMetric):
    """Compute one physical vector RMSE over the material-case population."""

    def compute(self) -> float:
        """Return the group's physical term RMSE."""
        return self.objective().physical_term_rmse[self._single_term()]


def compute_grain_moisture(
    w_surf: torch.Tensor,
    w_int: torch.Tensor,
    f_surf: torch.Tensor,
) -> torch.Tensor:
    """Return physical grain moisture from surface and interior moisture fields."""
    if not all(isinstance(value, torch.Tensor) for value in (w_surf, w_int, f_surf)):
        msg = "w_surf, w_int, and f_surf must be tensors."
        raise TypeError(msg)
    if w_surf.shape != w_int.shape or w_surf.shape != f_surf.shape:
        msg = "w_surf, w_int, and f_surf must share one shape."
        raise ValueError(msg)
    if not all(value.is_floating_point() and not value.is_complex() for value in (w_surf, w_int, f_surf)):
        msg = "grain moisture inputs must be real floating-point tensors."
        raise TypeError(msg)
    if not all(value.device == w_surf.device for value in (w_int, f_surf)):
        msg = "grain moisture inputs must share one device."
        raise ValueError(msg)
    if not all(bool(torch.isfinite(value).all().item()) for value in (w_surf, w_int, f_surf)):
        msg = "grain moisture inputs must be finite."
        raise ValueError(msg)
    return f_surf * w_surf + (1.0 - f_surf) * w_int


def _resolved_metric_fields(
    raw_fields: Any,
    *,
    task_fields: tuple[str, ...],
    metric_id: str,
) -> tuple[str, ...]:
    """
    Return the exact ordered fields from an already resolved metric declaration.

    ``all`` expands to TaskSpec output order. Explicit lists must be non-empty,
    unique, and task-known. Failures retain the metric ID for config diagnostics.
    """
    if raw_fields == "all":
        return task_fields
    if not isinstance(raw_fields, list) or not raw_fields or not all(isinstance(field, str) for field in raw_fields):
        msg = f"Evaluation metric {metric_id!r} fields must be 'all' or a non-empty list of strings."
        raise TypeError(msg)
    fields = tuple(raw_fields)
    if len(fields) != len(set(fields)):
        msg = f"Evaluation metric {metric_id!r} contains duplicate fields: {list(fields)}."
        raise ValueError(msg)
    unknown = [field for field in fields if field not in task_fields]
    if unknown:
        msg = f"Evaluation metric {metric_id!r} references unknown output fields: {unknown}."
        raise ValueError(msg)
    return fields


def _resolved_output_scales(
    raw_scales: torch.Tensor | None,
    *,
    task_fields: tuple[str, ...],
) -> dict[str, float]:
    """Return the Train state scale values aligned with TaskSpec output order."""
    if raw_scales is None:
        return {}
    if not isinstance(raw_scales, torch.Tensor):
        msg = "Output scales must be the Train state-scale tensor."
        raise TypeError(msg)
    flattened = raw_scales.detach().reshape(-1).cpu()
    if flattened.numel() != len(task_fields):
        msg = f"Output scales must contain {len(task_fields)} TaskSpec channels, got {flattened.numel()}."
        raise ValueError(msg)
    resolved: dict[str, float] = {}
    for index, field in enumerate(task_fields):
        scale = float(flattened[index].item())
        if not np.isfinite(scale) or scale <= 0.0:
            msg = f"Output field {field!r} Train scale must be finite and positive."
            raise ValueError(msg)
        resolved[field] = scale
    return resolved


_METRIC_CLASSES: Final[Mapping[str, type[DatasetMetric]]] = MappingProxyType(
    {
        "rmse": FieldRMSEMetric,
        "mae": FieldMAEMetric,
        "relative_l2": RelativeNormMetric,
        "relative_h1": RelativeNormMetric,
        "group_macro_rmse": GroupMacroRMSEMetric,
        "group_rmse": GroupRMSEMetric,
        "vector_rmse": VectorRMSEMetric,
    }
)


def build_evaluation_metrics(
    config: dict[str, Any],
    *,
    device: torch.device,
    output_scales: torch.Tensor | None = None,
    grid_spacing: Sequence[float] | None = None,
) -> dict[str, DatasetMetric]:
    """
    Build explicit-space dataset accumulators from semantic config.

    Metric implementations do not receive or own normalizers. Group-aware
    metrics receive only the Train state scales max(sigma, floor) and consume
    physical prediction/target views; relative norms receive the task's
    reference floors. Group membership comes from the resolved task contract
    rather than metric IDs or field-name conditionals.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved task and evaluation configuration.
    device : torch.device
        Concrete device required by all accumulator updates.
    output_scales : torch.Tensor | None, optional
        Train state-scale tensor max(sigma, floor) in TaskSpec output
        order. Required when any configured metric uses physical output groups.
    grid_spacing : Sequence[float] | None, optional
        Physical grid spacing in metres, rows then columns. Required when any
        configured metric is ``relative_h1``.

    Returns
    -------
    dict[str, DatasetMetric]
        Metric-ID keyed fresh accumulators in declaration order.

    Raises
    ------
    TypeError
        If required resolved sections or metric entries have invalid types.
    ValueError
        If IDs, field selections, units, directions, or semantic combinations
        contradict the registered task and metric contracts.

    """
    task = domain.tasks.registry.get_task(str(config["task"]))
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        msg = "Resolved config must contain an evaluation mapping."
        raise TypeError(msg)
    raw_metrics = evaluation.get("metrics")
    if not isinstance(raw_metrics, list):
        msg = "evaluation.metrics must be a list."
        raise TypeError(msg)
    resolved_scales = _resolved_output_scales(output_scales, task_fields=task.metric_names)

    built: dict[str, DatasetMetric] = {}
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            msg = "Each evaluation metric must be a mapping."
            raise TypeError(msg)
        metric_id = str(raw_metric["id"])
        if metric_id in built:
            msg = f"Duplicate evaluation metric id {metric_id!r}."
            raise ValueError(msg)
        kind = str(raw_metric["kind"])
        space = str(raw_metric["space"])
        kind_spec = validate_metric_semantics(kind, space=space, reduction=str(raw_metric["reduction"]))
        fields = _resolved_metric_fields(raw_metric["fields"], task_fields=task.metric_names, metric_id=metric_id)
        groups = resolve_metric_groups(task, kind=kind, fields=fields, metric_id=metric_id)
        grouped = kind in _GROUP_KINDS
        # A normalized field error is the artifacts' one definition: physical statistics, the error over the physical Train scale.
        rescaled = kind in {"rmse", "mae"} and space == "normalized"
        if (grouped or rescaled) and not resolved_scales:
            msg = f"Metric {metric_id!r} requires the Train state scales."
            raise ValueError(msg)
        direction = str(raw_metric.get("direction", kind_spec.direction))
        if direction != kind_spec.direction:
            msg = f"Metric {metric_id!r} direction {direction!r} contradicts {kind!r}."
            raise ValueError(msg)
        physical_unit = kind == "vector_rmse" or (kind in {"rmse", "mae"} and space == "physical")
        definition = ResolvedMetric(
            id=metric_id,
            kind=kind,
            space="physical" if rescaled else cast("MetricSpace", space),
            fields=fields,
            field_indices=tuple(task.metric_names.index(field) for field in fields),
            reduction="material_case_mean",
            direction=cast("MetricDirection", direction),
            unit=task.metric_field(fields[0]).unit if physical_unit else "1",
            groups=groups,
            field_scales=tuple(resolved_scales[field] for field in fields) if grouped or rescaled else (),
            reference_floors=tuple(task.reference_floor(field) for field in fields) if kind in _RELATIVE_KINDS else (),
        )
        metric_class = _METRIC_CLASSES[kind]
        if metric_class is RelativeNormMetric:
            built[metric_id] = RelativeNormMetric(definition, device=device, grid_spacing=grid_spacing)
        else:
            built[metric_id] = metric_class(definition, device=device)
    return built
