"""Protect the shared field metrics, the steady group objective and raw train-scale semantics.

Responsibilities:
  - Check the case-resolved physical-group RMSE, the group weights, partition invariance, and invalid
    scales
  - Check that validation and the artifact aggregate reduce every per-field metric and the objective alike,
    per case, then per material, then over the materials
  - Check per-case relative norms, near-zero references and the cropped H1 interior in metres

Design principles:
  - Task-owned groups use per-case sufficient statistics and raw train scales
  - Expected values come from hand-reduced fixtures, not from the reducers under test

This module does NOT:
  - Route normalized/physical tensors or assess fitted-normalizer quality
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from support import configs

from src import analysis, domain, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_NORMALIZED_VECTOR_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "group_rmse")
_PHYSICAL_VECTOR_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "vector_rmse")
_OBJECTIVE_ID = _OBJECTIVE_METRIC.id
_VECTOR_IDS = (_NORMALIZED_VECTOR_METRIC.id, _PHYSICAL_VECTOR_METRIC.id)
_CASE_METRICS = (learning.metrics.metrics.GroupMacroRMSEMetric, learning.metrics.metrics.GroupRMSEMetric, learning.metrics.metrics.VectorRMSEMetric)


def _metric_config(*metric_ids: str) -> dict[str, Any]:
    """Return a resolved config containing only requested public metrics."""
    config = configs.resolve(configs.direct_config())
    requested = set(metric_ids)
    config["evaluation"]["metrics"] = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in requested]
    return config


def _evaluate_physical_partitions(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scales: torch.Tensor,
    partitions: tuple[int, ...],
) -> dict[str, float]:
    """Accumulate identical physical tensors under prescribed batch boundaries."""
    metrics = learning.metrics.metrics.build_evaluation_metrics(
        _metric_config(_OBJECTIVE_ID, *_VECTOR_IDS),
        device=torch.device("cpu"),
        output_scales=scales,
    )
    start = 0
    for batch_index, size in enumerate(partitions):
        stop = start + size
        rows = range(start, stop)
        cases = learning.metrics.metrics.CaseMembership(tuple(f"case-{row}" for row in rows), tuple("AB"[row % 2] for row in rows))
        for metric in metrics.values():
            assert isinstance(metric, _CASE_METRICS)
            metric.update(
                pred[start:stop],
                target[start:stop],
                batch_index=batch_index,
                cases=cases,
            )
        start = stop
    assert start == pred.shape[0]
    return {metric_id: metric.compute() for metric_id, metric in metrics.items()}


def test_group_objective_uses_equal_physical_groups_and_global_statistics() -> None:
    """
    Evaluate unequal pressure and velocity scales under three batch partitions.

    The objective must average the scalar and vector group errors equally. The
    velocity term combines physical component MSE before one root and divides by
    one root-sum-square raw train scale. A physically small transverse error stays
    visible without receiving independent unit-variance influence.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    groups = {group.id: group for group in task.output_groups}
    pressure = groups["pressure"]
    velocity = groups["velocity"]
    scales_by_field = {
        pressure.fields[0]: 4.0,
        velocity.fields[0]: 0.1,
        velocity.fields[1]: 10.0,
    }
    error_by_field = {
        pressure.fields[0]: 8.0,
        velocity.fields[0]: 0.1,
        velocity.fields[1]: 20.0,
    }
    scales = torch.tensor([scales_by_field[field] for field in task.output_names], dtype=torch.float64)
    errors = torch.tensor(
        [error_by_field[field] for field in task.output_names],
        dtype=torch.float64,
    ).reshape(1, -1, 1, 1)
    target = torch.zeros((5, task.out_channels, 2, 3), dtype=torch.float64)
    pred = target + errors

    partial = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(2, 3))
    separate = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(1, 1, 1, 1, 1))
    combined = _evaluate_physical_partitions(pred, target, scales=scales, partitions=(5,))

    pressure_error = 8.0 / 4.0
    physical_velocity = (0.1**2 + 20.0**2) ** 0.5
    normalized_velocity = physical_velocity / (0.1**2 + 10.0**2) ** 0.5
    expected_objective = 0.5 * pressure_error + 0.5 * normalized_velocity
    assert partial == pytest.approx(
        {
            _OBJECTIVE_ID: expected_objective,
            _NORMALIZED_VECTOR_METRIC.id: normalized_velocity,
            _PHYSICAL_VECTOR_METRIC.id: physical_velocity,
        },
        rel=1e-14,
        abs=1e-14,
    )
    assert partial == pytest.approx(separate, rel=1e-14, abs=1e-14)
    assert partial == pytest.approx(combined, rel=1e-14, abs=1e-14)
    independently_scaled_field_mean = (pressure_error + 1.0 + 2.0) / 3.0
    assert partial[_OBJECTIVE_ID] != pytest.approx(independently_scaled_field_mean)

    zero_errors = dict.fromkeys(task.output_names, 0.0)
    u_only, _, _ = learning.metrics.metrics.term_objective(
        {**zero_errors, velocity.fields[0]: scales_by_field[velocity.fields[0]] ** 2}, terms=task.error_terms, train_scales=scales_by_field
    )
    v_only, _, _ = learning.metrics.metrics.term_objective(
        {**zero_errors, velocity.fields[1]: scales_by_field[velocity.fields[1]] ** 2}, terms=task.error_terms, train_scales=scales_by_field
    )
    shared_variance = scales_by_field[velocity.fields[0]] ** 2 + scales_by_field[velocity.fields[1]] ** 2
    expected_u_only = scales_by_field[velocity.fields[0]] / shared_variance**0.5
    expected_v_only = scales_by_field[velocity.fields[1]] / shared_variance**0.5
    assert u_only[velocity.id] == pytest.approx(expected_u_only)
    assert v_only[velocity.id] == pytest.approx(expected_v_only)
    assert v_only[velocity.id] / u_only[velocity.id] == pytest.approx(scales_by_field[velocity.fields[1]] / scales_by_field[velocity.fields[0]])


def test_field_metrics_reduce_cases_then_materials_alike_in_validation_and_artifacts() -> None:
    """
    Reduce unequal cases of two materials through the validation accumulators and from their artifact rows.

    ``case-1`` enters in two rows, so it holds twice the elements of the others, and material A has two
    cases against one in B. Every per-field metric and the objective must agree between both paths, and
    the RMSE must be the root of the case, then material, then materials mean of the case MSE, not the
    element-pooled RMSE.
    """
    spacing = (0.5, 0.25)
    rows = (("case-1", "A"), ("case-1", "A"), ("case-2", "A"), ("case-3", "B"))
    generator = torch.Generator().manual_seed(7)
    target = 1.0 + torch.rand((len(rows), _TASK.out_channels, 6, 7), generator=generator, dtype=torch.float64)
    prediction = target + 0.2 * torch.randn(target.shape, generator=generator, dtype=torch.float64)
    scales = dict(zip(_TASK.output_names, (4.0, 0.5, 2.0), strict=True))
    scale = torch.tensor(list(scales.values()), dtype=torch.float64).view(1, -1, 1, 1)
    metric_ids = tuple(spec.id for spec in domain.tasks.spec.field_metric_specs(_TASK.output_names))
    metrics = learning.metrics.metrics.build_evaluation_metrics(
        _metric_config(_OBJECTIVE_ID, *metric_ids), device=torch.device("cpu"), output_scales=scale.flatten(), grid_spacing=spacing
    )
    for index, (case_id, material) in enumerate(rows):
        cases = learning.metrics.metrics.CaseMembership((case_id,), (material,))
        for metric in metrics.values():
            row = slice(index, index + 1)
            metric.update(prediction[row], target[row], batch_index=index, cases=cases)
    validation = learning.metrics.metrics.finalize_metrics(metrics)

    sums = learning.metrics.metrics.field_statistics(prediction, target, grid_spacing=spacing)
    by_case: dict[tuple[str, str], np.ndarray] = {}
    for key, row_sums in zip(rows, sums, strict=True):
        by_case[key] = by_case.get(key, np.zeros_like(row_sums)) + row_sums
    floors = {field: _TASK.reference_floor(field) for field in _TASK.output_names}
    records: list[dict[str, Any]] = []
    for (case_id, material), case_sums in by_case.items():
        first_row = rows.index((case_id, material))
        record: dict[str, Any] = {
            "meta": json.dumps({"simulation_case_id": [case_id], "material_family": [material]}),
            "prediction_validity": json.dumps(
                validity_contract.build_field_prediction_validity(prediction[first_row].numpy(), channels=_TASK.output_names)
            ),
        }
        for field_index, field in enumerate(_TASK.output_names):
            record.update(zip(analysis.artifacts.contracts.case_statistic_columns(field), case_sums[field_index], strict=True))
        record.update(
            analysis.artifacts.contracts.case_metric_values(
                case_sums, fields=_TASK.output_names, output_groups=_TASK.output_groups, train_scales=scales, reference_floors=floors
            )
        )
        records.append(record)
    aggregate = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        pd.DataFrame(records), output_groups=_TASK.output_groups, train_scales=scales, reference_floors=floors
    )

    squared = (prediction - target).square()[:, 0]
    case_mse = {key: float(squared[[row == key for row in rows]].mean()) for key in by_case}
    expected = math.sqrt((0.5 * (case_mse[("case-1", "A")] + case_mse[("case-2", "A")]) + case_mse[("case-3", "B")]) / 2.0)
    assert validation[f"physical_rmse_{_TASK.output_names[0]}"] == pytest.approx(expected, rel=1e-12)
    assert expected != pytest.approx(math.sqrt(float(squared.mean())), rel=1e-6)
    assert (aggregate["case_count"], aggregate["material_count"]) == (3, 2)
    assert aggregate["value"] == pytest.approx(validation[_OBJECTIVE_ID], rel=1e-12)
    assert aggregate["field_metrics"] == pytest.approx({metric_id: validation[metric_id] for metric_id in metric_ids}, rel=1e-12)


def _single_field_case(case_id: str, material: str, *, reference: float, error: float) -> learning.metrics.metrics.CaseStatistics:
    """Return the statistics of one constant one-field case on a 6 x 6 grid with H1 sums."""
    target = torch.full((1, 1, 6, 6), reference, dtype=torch.float64)
    sums = learning.metrics.metrics.field_statistics(target + error, target, grid_spacing=(0.1, 0.1))[0]
    return learning.metrics.metrics.CaseStatistics(case_id, material, ("f",), sums)


def test_relative_norm_is_formed_per_case_before_the_mean() -> None:
    """
    Two cases of one material with reference norms 100 and 1 and errors of 10 % and 100 % give 55 %.

    Summing the energies over the cases first would give about 10 % and hide the poor small-norm case.
    """
    cases = (_single_field_case("large", "A", reference=100.0, error=10.0), _single_field_case("small", "A", reference=1.0, error=1.0))
    for norm in learning.metrics.metrics.RELATIVE_NORMS:
        result = learning.metrics.metrics.relative_norms(cases, norm=norm, floors={"f": 1.0e-3})
        assert result.by_field["f"] == pytest.approx(0.55, rel=1e-12)
        assert result.unavailable["f"] == 0


def test_near_zero_reference_is_unavailable_and_counted() -> None:
    """A case whose reference RMS is below the floor has no relative value; a material without any supported case has none either."""
    supported = _single_field_case("supported", "A", reference=1.0, error=0.5)
    near_zero = _single_field_case("near-zero", "A", reference=1.0e-4, error=1.0e-4)
    only_near_zero = _single_field_case("only-near-zero", "B", reference=1.0e-4, error=1.0e-4)

    assert learning.metrics.metrics.case_relative_norm(near_zero.sums[0], norm="l2", floor=1.0e-3) is None
    mixed = learning.metrics.metrics.relative_norms((supported, near_zero), norm="l2", floors={"f": 1.0e-3})
    assert (mixed.by_field["f"], mixed.unavailable["f"]) == (pytest.approx(0.5, rel=1e-12), 1)
    lacking = learning.metrics.metrics.relative_norms((supported, only_near_zero), norm="h1", floors={"f": 1.0e-3})
    assert (lacking.by_field["f"], lacking.unavailable["f"]) == (None, 1)


def test_relative_h1_uses_the_cropped_interior_with_unit_length_scale_in_metres() -> None:
    """
    Compare the relative H1 of quadratic fields on an anisotropic grid with its analytic value.

    Central differences are exact for quadratics, so the discrete gradient in metres equals the analytic one
    on the interior without two cells per edge, where the energy density is f² + (1 m)²·|∇f|².
    """
    spacing = (0.5, 0.25)
    y = (torch.arange(7, dtype=torch.float64) * spacing[0])[:, None].expand(7, 9)
    x = (torch.arange(9, dtype=torch.float64) * spacing[1])[None, :].expand(7, 9)
    target = 1.0 + 2.0 * y + 3.0 * x + y**2 + x * y
    error = 0.1 * (y**2 + 2.0 * x * y + x)
    target_density = target**2 + (2.0 + 2.0 * y + x) ** 2 + (3.0 + y) ** 2
    error_density = error**2 + (0.1 * (2.0 * y + 2.0 * x)) ** 2 + (0.1 * (2.0 * y + 1.0)) ** 2
    interior = (slice(2, -2), slice(2, -2))
    expected = math.sqrt(float(error_density[interior].sum()) / float(target_density[interior].sum()))

    sums = learning.metrics.metrics.field_statistics((target + error)[None, None], target[None, None], grid_spacing=spacing)[0, 0]

    assert learning.metrics.metrics.case_relative_norm(sums, norm="h1", floor=1.0e-3) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="positive finite spacings"):
        learning.metrics.metrics.field_statistics(target[None, None], target[None, None], grid_spacing=(0.5, float("nan")))


def test_group_metric_is_task_owned_and_rejects_invalid_scales_or_statistics(
    synthetic_task: domain.tasks.spec.TaskSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve an alternate grouped task, then exercise scale and statistic failures.

    The generic builder must consume TaskSpec group membership without steady-flow
    field checks. Missing groups, partial groups, empty counts, and invalid raw
    fitted scales fail before an objective can be selected or compared.
    """
    monkeypatch.setattr(
        learning.metrics.metrics.domain.tasks.registry,
        "get_task",
        lambda _task_id: synthetic_task,
    )
    definition = next(
        metric.as_dict(all_fields=synthetic_task.output_names) for metric in synthetic_task.default_metrics if metric.kind == "group_macro_rmse"
    )
    objective_id = str(definition["id"])
    config = {"task": synthetic_task.id, "evaluation": {"metrics": [definition]}}
    scales = torch.tensor([2.0, 6.0])
    metric = learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
        output_scales=scales,
    )[objective_id]
    target = torch.zeros((2, synthetic_task.out_channels, 1, 2), dtype=torch.float64)
    errors = torch.tensor([2.0, 12.0], dtype=torch.float64).reshape(1, -1, 1, 1)
    assert isinstance(metric, learning.metrics.metrics.GroupMacroRMSEMetric)
    metric.update(target + errors, target, batch_index=0, cases=learning.metrics.metrics.CaseMembership(("c1", "c2"), ("A", "A")))

    assert metric.fields == synthetic_task.output_names
    assert metric.compute() == pytest.approx(1.5)

    partial_definition = dict(definition)
    partial_definition["fields"] = [synthetic_task.output_names[0]]
    config["evaluation"]["metrics"] = [partial_definition]
    with pytest.raises(ValueError, match="must select every TaskSpec metric field"):
        learning.metrics.metrics.build_evaluation_metrics(
            config,
            device=torch.device("cpu"),
            output_scales=scales,
        )

    config["evaluation"]["metrics"] = [definition]
    for invalid_scales in (torch.tensor([0.0, 1.0]), torch.tensor([float("inf"), 1.0])):
        with pytest.raises(ValueError, match="finite and positive"):
            learning.metrics.metrics.build_evaluation_metrics(
                config,
                device=torch.device("cpu"),
                output_scales=invalid_scales,
            )
    with pytest.raises(ValueError, match="requires the Train state scales"):
        learning.metrics.metrics.build_evaluation_metrics(config, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="at least one case"):
        learning.metrics.metrics.group_objective((), terms=synthetic_task.error_terms, train_scales={"response_a": 1.0, "response_b": 1.0})
