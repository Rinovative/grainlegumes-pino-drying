# ruff: noqa: S101, D103, PLR2004, SLF001
"""Protect adapter-owned lifecycle aggregation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim.sgd import SGD

from src.learning.training import learning_training_adapter as adapter_contract
from src.learning.training import learning_training_loop as loop


@dataclass
class _Step:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    sample_count: int
    processed_target_transitions: int
    forward_transitions: int


class _Adapter:
    gradient_accumulation_steps = 3

    def __init__(self) -> None:
        self.work: list[adapter_contract.OptimizerWork] = []

    def prepare_batch(self, raw: dict[str, torch.Tensor], *, device: torch.device, training: bool) -> dict[str, torch.Tensor]:
        del training
        return {key: value.to(device) for key, value in raw.items()}

    def training_step(self, model: nn.Module, batch: dict[str, torch.Tensor], loss: nn.Module) -> _Step:
        del loss
        value = (model(batch["x"]) - batch["y"]).square().mean()
        return _Step(value, {"total": value, "custom": value * 2.0}, 1, int(batch["transitions"]), 7)

    def record_optimizer_work(self, work: adapter_contract.OptimizerWork) -> None:
        self.work.append(work)

    def should_stop(self) -> bool:
        return False


class _BudgetAdapter(_Adapter):
    gradient_accumulation_steps = 1

    def should_stop(self) -> bool:
        return bool(self.work)


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
        mask = torch.ones(2, 2, 1, 1, 1, dtype=torch.bool)
        return adapter_contract.AdapterEvaluationViews(pred, target, pred, target, 2, 4, mask, torch.full((2, 2), 0.5))


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


def test_adapter_evaluation_uses_horizon_and_first_transition_guardrail_masks() -> None:
    metric = _Metric()
    eval_loader: Any = [object()]
    values = loop._eval_adapter_one_epoch(nn.Identity(), eval_loader, {"metric": metric}, torch.device("cpu"), _EvalAdapter())
    assert values["metric"] == 5.0
    assert values["guardrail/one_step/metric"] == 1.0
    assert metric.masks[0] is not None
