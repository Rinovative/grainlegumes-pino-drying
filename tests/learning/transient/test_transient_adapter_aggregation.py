"""Protect adapter-owned evaluation aggregation contracts.

Responsibilities:
  - Check rollout scoring with the one-step guardrail view
  - Check that physical moisture telemetry accumulates in float64 and stays finite for finite float32 predictions
  - Check that grain-moisture validation errors are reduced per case, then over cases and materials
  - Check that one non-finite case leaves a validation without metric values, and the per-channel bound diagnostics

Design principles:
  - Fixed evaluation views isolate the aggregation from rollout numerics

This module does NOT:
  - Resolve stage plans or allocate parent experiment records
  - Test the training update path; the unit-update tests own it
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from src.learning.training import learning_training_adapter as adapter_contract
from src.learning.training import learning_training_loop as loop
from src.learning.transient import learning_transient_adapter as transient_adapter


class _Metric:
    id = "metric"
    requires_cases = True

    def reset(self) -> None:
        self.values: list[float] = []
        self.masks: list[torch.Tensor | None] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, batch_index: int, cases: Any, mask: torch.Tensor | None = None) -> None:
        del batch_index, cases
        self.values.extend((pred - target).flatten().tolist())
        self.masks.append(mask)

    def compute(self) -> float:
        return float(sum(abs(value) for value in self.values) / len(self.values))

    def components(self) -> dict[str, float]:
        return {}


def _batch(*case_ids: str, material: str = "lentil") -> dict[str, Any]:
    """Return one loader batch that carries only the case membership of its rows."""
    return {"metadata": {"simulation_case_id": list(case_ids), "material_family": [material] * len(case_ids)}}


class _EvalAdapter:
    def prepare_batch(self, raw: Any, *, device: torch.device, training: bool) -> Any:
        del device, training
        return raw

    def evaluation_step(self, model: nn.Module, batch: Any) -> adapter_contract.AdapterEvaluationViews:
        del model, batch
        target = torch.zeros(2, 2, 4, 1, 1)
        pred = target.clone()
        pred[:, 0] = 1.0
        pred[:, 1] = 9.0
        one_step = target + 1.0
        mask = torch.ones(2, 2, 1, 1, 1, dtype=torch.bool)
        return adapter_contract.AdapterEvaluationViews(
            physical_prediction=pred,
            physical_target=target,
            one_step_physical_prediction=one_step,
            sample_count=2,
            processed_target_transitions=4,
            case_complete=torch.ones(2, dtype=torch.bool),
            bound_violations={},
            valid_mask=mask,
            f_surf=torch.full((2, 2), 0.5),
        )


class _FixedViewsAdapter(_EvalAdapter):
    def __init__(self, views: adapter_contract.AdapterEvaluationViews) -> None:
        self.views = views

    def evaluation_step(self, model: nn.Module, batch: Any) -> adapter_contract.AdapterEvaluationViews:
        del model, batch
        return self.views


def test_adapter_moisture_telemetry_stays_finite_for_finite_float32_predictions() -> None:
    """Squares of finite float32 moisture predictions near 1e20 overflow float32 but not the float64 telemetry."""
    f_surf = torch.full((1, 2), 0.5)
    physical_target = torch.zeros(1, 2, 4, 1, 2)
    physical_target[:, :, 2] = 0.25
    physical_target[:, :, 3] = 0.5
    physical_prediction = physical_target.clone()
    physical_prediction[:, :, 2] = torch.tensor([[[[1.0e20, 2.0e20]], [[3.0e20, 4.0e20]]]])
    physical_prediction[:, :, 3] = 2.0e20
    views = adapter_contract.AdapterEvaluationViews(
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        one_step_physical_prediction=physical_prediction,
        sample_count=1,
        processed_target_transitions=2,
        case_complete=torch.ones(1, dtype=torch.bool),
        bound_violations={},
        valid_mask=torch.ones(1, 2, 1, 1, 1, dtype=torch.bool),
        f_surf=f_surf,
    )

    values = loop._eval_adapter_one_epoch(nn.Identity(), [_batch("a")], {"metric": _Metric()}, torch.device("cpu"), _FixedViewsAdapter(views))

    fraction = f_surf.double().reshape(1, 2, 1, 1)
    expected = (
        fraction * physical_prediction[:, :, 2].double()
        + (1.0 - fraction) * physical_prediction[:, :, 3].double()
        - (fraction * physical_target[:, :, 2].double() + (1.0 - fraction) * physical_target[:, :, 3].double())
    )
    float32_fraction = f_surf.reshape(1, 2, 1, 1)
    float32_difference = (
        float32_fraction * physical_prediction[:, :, 2]
        + (1.0 - float32_fraction) * physical_prediction[:, :, 3]
        - (float32_fraction * physical_target[:, :, 2] + (1.0 - float32_fraction) * physical_target[:, :, 3])
    )
    assert torch.isfinite(float32_difference).all()
    assert torch.isinf(float32_difference.square().sum())
    assert math.isfinite(values["physical/w_gr_rmse"])
    assert math.isfinite(values["physical/w_gr_mae"])
    assert values["physical/w_gr_rmse"] == pytest.approx(float(expected.square().mean().sqrt()), rel=1e-6)
    assert values["physical/w_gr_mae"] == pytest.approx(float(expected.abs().mean()), rel=1e-6)


class _BatchViewsAdapter(_EvalAdapter):
    """Return the evaluation views each test batch carries."""

    def evaluation_step(self, model: nn.Module, batch: Any) -> adapter_contract.AdapterEvaluationViews:
        del model
        return batch["views"]


def _moisture_case(case_id: str, *, length: int, error: float) -> dict[str, Any]:
    """Return one single-case batch whose grain moisture misses the zero reference by ``error`` at every state."""
    target = torch.zeros(1, length, 4, 1, 1)
    prediction = target.clone()
    prediction[:, :, 2:] = error
    views = adapter_contract.AdapterEvaluationViews(
        physical_prediction=prediction,
        physical_target=target,
        one_step_physical_prediction=prediction,
        sample_count=1,
        processed_target_transitions=length,
        case_complete=torch.ones(1, dtype=torch.bool),
        bound_violations={},
        f_surf=torch.full((1, length), 0.5),
    )
    return {**_batch(case_id), "views": views}


def test_grain_moisture_validation_averages_each_case_before_the_cases_like_the_artifacts() -> None:
    """Errors 1 over one state and 3 over three states give MAE 2 and RMSE sqrt(5), not the pooled 2.5 and sqrt(7)."""
    batches = [_moisture_case("short", length=1, error=1.0), _moisture_case("long", length=3, error=3.0)]

    values = loop._eval_adapter_one_epoch(nn.Identity(), batches, {"metric": _Metric()}, torch.device("cpu"), _BatchViewsAdapter())

    assert values["physical/w_gr_mae"] == pytest.approx(2.0)
    assert values["physical/w_gr_rmse"] == pytest.approx(math.sqrt(5.0))


def test_adapter_evaluation_scores_the_rollout_and_guards_with_the_one_step_view() -> None:
    metric = _Metric()
    eval_loader: Any = [_batch("a", "b")]
    values = loop._eval_adapter_one_epoch(nn.Identity(), eval_loader, {"metric": metric}, torch.device("cpu"), _EvalAdapter())
    assert values["metric"] == 5.0
    assert values["guardrail/one_step/metric"] == 1.0
    assert values["nonfinite_cases"] == 0.0
    assert metric.masks[0] is not None


class _IncompleteSecondBatchAdapter(_EvalAdapter):
    """Return a first batch with a moisture violation of row 0 and a second batch with one non-finite case and a violation of row 1."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluation_step(self, model: nn.Module, batch: Any) -> adapter_contract.AdapterEvaluationViews:
        views = super().evaluation_step(model, batch)
        self.calls += 1
        if self.calls == 1:
            return replace(views, bound_violations={"w_surf": (4.0, None), "T": (None, None)})
        return replace(views, case_complete=torch.tensor([True, False]), bound_violations={"w_surf": (None, 2.5), "T": (None, None)})


def test_one_non_finite_case_leaves_the_validation_without_metric_values_but_counts_every_case() -> None:
    """A validation with one non-finite case is not eligible; per-case bound evidence is still collected, and the aggregates derive from it."""
    bound_cases: dict[str, dict[str, float]] = {}
    values = loop._eval_adapter_one_epoch(
        nn.Identity(),
        [_batch("a", "b"), _batch("c", "d")],
        {"metric": _Metric()},
        torch.device("cpu"),
        _IncompleteSecondBatchAdapter(),
        bound_cases=bound_cases,
    )

    assert bound_cases == {"w_surf": {"a": 4.0, "d": 2.5}}
    assert values == {
        "nonfinite_cases": 1.0,
        "bounds/w_surf/violating_cases": 2.0,
        "bounds/w_surf/first_violation_hours": 2.5,
        "bounds/T/violating_cases": 0.0,
    }
    assert not loop.validation_eligible(values, "metric")
    assert not loop.validation_eligible({"nonfinite_cases": 0.0, "metric": math.nan}, "metric")
    assert loop.validation_eligible({"nonfinite_cases": 0.0, "metric": 0.5}, "metric")


class _NonFiniteOneStepAdapter(_EvalAdapter):
    """Return a finite autonomous trajectory whose teacher-forced one-step diagnostic is non-finite."""

    def evaluation_step(self, model: nn.Module, batch: Any) -> adapter_contract.AdapterEvaluationViews:
        views = super().evaluation_step(model, batch)
        one_step = views.one_step_physical_prediction.clone()
        one_step[1, 0] = math.inf
        return replace(views, one_step_physical_prediction=one_step)


def test_a_non_finite_one_step_diagnostic_never_affects_eligibility() -> None:
    """The one-step view only loses its guardrail value; the autonomous trajectory decides completeness."""
    values = loop._eval_adapter_one_epoch(nn.Identity(), [_batch("a", "b")], {"metric": _Metric()}, torch.device("cpu"), _NonFiniteOneStepAdapter())

    assert values["nonfinite_cases"] == 0.0
    assert values["metric"] == 5.0
    assert "guardrail/one_step/metric" not in values
    assert loop.validation_eligible(values, "metric")


def test_bound_violations_return_each_cases_first_time_without_clamping() -> None:
    """Negative moisture in two cases at different times; the corners stay excluded and a third case has no violation."""
    prediction = torch.zeros(3, 3, 4, 5, 5)
    prediction[:, :, 0] = 300.0
    prediction[0, 1:, 2, 2, 2] = -1.0e-6
    prediction[1, 2, 2, 2, 3] = -2.0
    prediction[1, 0, 2, 0, 0] = -5.0  # corner block, excluded
    times = torch.tensor([[1.0, 2.0, 3.0]] * 3)

    violations = transient_adapter._bound_violations(prediction, times)

    assert violations["w_surf"] == (2.0, 3.0, None)
    assert violations["T"] == (None, None, None)
    assert prediction[0, 1, 2, 2, 2].item() == pytest.approx(-1.0e-6)
