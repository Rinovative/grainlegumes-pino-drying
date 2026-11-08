"""Protect adapter-owned lifecycle aggregation contracts.

Responsibilities:
  - Check transition-weighted accumulation, budget stops, and rollout scoring with the one-step guardrail view
  - Check that physical moisture telemetry accumulates in float64 and stays finite for finite float32 predictions

Design principles:
  - The first committed budget-crossing optimizer group ends admission of later work

This module does NOT:
  - Resolve stage plans or allocate parent experiment records
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.optim.sgd import SGD

from src.learning.training import learning_training_adapter as adapter_contract
from src.learning.training import learning_training_loop as loop

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class _Step:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    sample_count: int
    processed_target_transitions: int
    forward_transitions: int
    consumed_items: int


class _Adapter:
    gradient_accumulation_steps = 3

    def __init__(self) -> None:
        self.work: list[adapter_contract.OptimizerWork] = []

    def prepare_batch(self, raw: dict[str, torch.Tensor], *, device: torch.device, training: bool) -> dict[str, torch.Tensor]:
        del training
        return {key: value.to(device) for key, value in raw.items()}

    def training_steps(self, model: nn.Module, batch: dict[str, torch.Tensor], loss: nn.Module) -> Iterator[_Step]:
        del loss
        value = (model(batch["x"]) - batch["y"]).square().mean()
        yield _Step(value, {"total": value, "custom": value * 2.0}, 1, int(batch["transitions"]), 7, 1)

    def record_optimizer_work(self, work: adapter_contract.OptimizerWork) -> None:
        self.work.append(work)

    def should_stop(self) -> bool:
        return False


class _BudgetAdapter(_Adapter):
    gradient_accumulation_steps = 1

    def should_stop(self) -> bool:
        return bool(self.work)


class _WindowAdapter(_Adapter):
    """Yield three windows per loader batch, each computed after the preceding update."""

    gradient_accumulation_steps = 1

    def __init__(self, *, stop_after: int | None = None) -> None:
        super().__init__()
        self.stop_after = stop_after
        self.weights_seen: list[float] = []

    def training_steps(self, model: nn.Module, batch: dict[str, torch.Tensor], loss: nn.Module) -> Iterator[_Step]:
        del loss
        for window in range(3):
            self.weights_seen.append(float(model.weight.detach().item()))
            value = (model(batch["x"]) - batch["y"]).square().mean()
            yield _Step(value, {"total": value}, 1, 1, 1, 1 if window == 2 else 0)

    def should_stop(self) -> bool:
        return self.stop_after is not None and len(self.work) >= self.stop_after


class _Metric:
    id = "metric"
    space = "normalized"

    def reset(self) -> None:
        self.values: list[float] = []
        self.masks: list[torch.Tensor | None] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, space: str, batch_index: int, mask: torch.Tensor | None = None) -> None:
        del space, batch_index
        self.values.extend((pred - target).flatten().tolist())
        self.masks.append(mask)

    def compute(self) -> float:
        return float(sum(abs(value) for value in self.values) / len(self.values))


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
            normalized_prediction=pred,
            normalized_target=target,
            physical_prediction=pred,
            physical_target=target,
            one_step_normalized_prediction=one_step,
            one_step_physical_prediction=one_step,
            sample_count=2,
            processed_target_transitions=4,
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
    normalized = torch.zeros(1, 2, 4, 1, 2)
    physical_target = torch.zeros(1, 2, 4, 1, 2)
    physical_target[:, :, 2] = 0.25
    physical_target[:, :, 3] = 0.5
    physical_prediction = physical_target.clone()
    physical_prediction[:, :, 2] = torch.tensor([[[[1.0e20, 2.0e20]], [[3.0e20, 4.0e20]]]])
    physical_prediction[:, :, 3] = 2.0e20
    views = adapter_contract.AdapterEvaluationViews(
        normalized_prediction=normalized,
        normalized_target=normalized,
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        one_step_normalized_prediction=normalized,
        one_step_physical_prediction=physical_prediction,
        sample_count=1,
        processed_target_transitions=2,
        valid_mask=torch.ones(1, 2, 1, 1, 1, dtype=torch.bool),
        f_surf=f_surf,
    )

    values = loop._eval_adapter_one_epoch(nn.Identity(), [object()], {"metric": _Metric()}, torch.device("cpu"), _FixedViewsAdapter(views))

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


def test_adapter_accumulation_weights_transitions_and_keeps_components() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = SGD(model.parameters(), lr=1.0)
    adapter = _Adapter()
    train_loader: Any = [
        {"x": torch.tensor([[1.0]]), "y": torch.zeros(1, 1), "transitions": torch.tensor(1)},
        {"x": torch.tensor([[2.0]]), "y": torch.zeros(1, 1), "transitions": torch.tensor(3)},
    ]
    values = loop.train_one_epoch(
        model,
        train_loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
        adapter=adapter,
    )
    assert torch.allclose(model.weight.detach(), torch.tensor([[-5.5]]))
    assert values["train/loss_custom"] == 6.5
    assert values["transient/train/processed_target_transitions"] == 4.0
    assert values["transient/train/optimizer_groups"] == 1.0
    # Training work counts the gradient-carrying target transitions, never the no-gradient warm-up forwards.
    assert values["transient/train/forward_transitions"] == 14.0
    assert (values["committed_optimizer_groups"], values["gradient_predictions"]) == (1.0, 4.0)
    assert adapter.work[0].processed_target_transitions == 4
    assert adapter.work[0].optimizer_device_seconds is None


def test_adapter_stops_after_first_budget_crossing_optimizer_group() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = SGD(model.parameters(), lr=0.1)
    adapter = _BudgetAdapter()
    train_loader: Any = [{"x": torch.tensor([[1.0]]), "y": torch.zeros(1, 1), "transitions": torch.tensor(1)} for _ in range(4)]

    values = loop.train_one_epoch(
        model,
        train_loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
        adapter=adapter,
    )

    assert len(adapter.work) == 1
    assert values["transient/train/optimizer_groups"] == 1.0
    assert values["transient/train/microbatches"] == 1.0
    assert values["optimizer_steps"] == 1.0


def test_adapter_evaluation_scores_the_rollout_and_guards_with_the_one_step_view() -> None:
    metric = _Metric()
    eval_loader: Any = [object()]
    values = loop._eval_adapter_one_epoch(nn.Identity(), eval_loader, {"metric": metric}, torch.device("cpu"), _EvalAdapter())
    assert values["metric"] == 5.0
    assert values["guardrail/one_step/metric"] == 1.0
    assert metric.masks[0] is not None


def test_windowed_loader_batches_step_after_every_window_and_stop_at_the_budget() -> None:
    """Each window is computed after the previous update; a completed budget stops mid-batch."""
    for stop_after, expected_groups in ((None, 6), (4, 4)):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        adapter = _WindowAdapter(stop_after=stop_after)
        train_loader: Any = [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1), "transitions": torch.tensor(1)} for _ in range(2)]
        values = loop.train_one_epoch(
            model,
            train_loader,
            SGD(model.parameters(), lr=0.1),
            nn.MSELoss(),
            torch.device("cpu"),
            adapter=adapter,
        )

        assert len(adapter.work) == expected_groups
        assert values["transient/train/microbatches"] == float(expected_groups)
        assert len(set(adapter.weights_seen)) == expected_groups
