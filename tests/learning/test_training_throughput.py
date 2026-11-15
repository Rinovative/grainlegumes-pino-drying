"""Protect optimizer-phase timing and committed-unit training throughput.

Responsibilities:
  - Check optimizer-phase timers, committed unit counts, and invalid-duration rejection

Design principles:
  - Loader iteration and final optimizer work remain inside measured training time

This module does NOT:
  - Benchmark production training or include post-training lifecycle work
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from support import configs
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src import learning
from src.datasets.runtime.dataset_runtime_units import UnitSampler

if TYPE_CHECKING:
    from collections.abc import Iterator

loop = learning.training.loop


class _MappingDataset(Dataset[dict[str, torch.Tensor]]):
    """Provide a configurable number of scalar regression samples."""

    def __init__(self, samples: int) -> None:
        self.inputs = torch.arange(float(samples)).reshape(samples, 1)
        self.targets = 2.0 * self.inputs

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.inputs[index], "y": self.targets[index]}


class _RecordingLoader:
    """Load a unit sampler's microbatches while exposing the first load in the event order."""

    def __init__(self, loader: DataLoader[Any], events: list[str]) -> None:
        self.loader = loader
        self.batch_sampler = loader.batch_sampler
        self.events = events

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self.events.append("loader_iteration")
        yield from self.loader


class _CaseMSE(nn.Module):
    """Return every case's squared error as its loss."""

    def unit_components(self, pred: torch.Tensor, *, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``total`` per case."""
        del x
        return {"total": (pred - y).square().flatten(1).mean(dim=1)}


def _loader(cases: int, *, units_per_update: int, memory_batch: int = 2) -> DataLoader[Any]:
    """Return a loader whose unit sampler has begun its first epoch."""
    sampler = UnitSampler(unit_kind="case", cases=range(cases), keys=None, units_per_update=units_per_update, memory_batch=memory_batch, seed=0)
    sampler.begin_epoch(0)
    return DataLoader(_MappingDataset(cases), batch_sampler=sampler)


def _model_and_optimizer() -> tuple[nn.Module, SGD]:
    """Return a minimal differentiable CPU model and optimizer."""
    model = nn.Linear(1, 1, bias=False)
    return model, SGD(model.parameters(), lr=0.0)


def _train(model: nn.Module, loader: Any, optimizer: SGD, **kwargs: Any) -> dict[str, float]:
    """Train the loader's remaining groups of one epoch on the steady update path, under a budget no epoch reaches."""
    budget = learning.training.budget.StageBudget(unit="units", total=1000)
    steady = loop._SteadyUnits(None, torch.device("cpu"))
    return loop.train_update_groups(model, loader, optimizer, _CaseMSE(), torch.device("cpu"), steady, budget=budget, **kwargs)


def _fixed_clock(monkeypatch: pytest.MonkeyPatch, start: float, stop: float) -> None:
    """Install a monotonic clock that reads ``start`` once and ``stop`` afterwards."""
    readings = [start]
    monkeypatch.setattr(loop.time, "perf_counter", lambda: readings.pop() if readings else stop)


def test_cpu_timer_wraps_loader_iteration_and_final_optimizer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start before loader iteration, stop after optimizer work, and never touch CUDA."""
    events: list[str] = []
    loader = cast("DataLoader[Any]", _RecordingLoader(_loader(5, units_per_update=3), events))
    model, optimizer = _model_and_optimizer()
    original_step = optimizer.step

    def record_step(*args: Any, **kwargs: Any) -> Any:
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    def clock() -> float:
        events.append("clock")
        return 10.0 if events.count("clock") == 1 else 12.0

    def forbidden_sync(_device: torch.device) -> None:
        pytest.fail("CPU training must not synchronize CUDA")

    monkeypatch.setattr(optimizer, "step", record_step)
    monkeypatch.setattr(loop.time, "perf_counter", clock)
    monkeypatch.setattr(loop.torch.cuda, "synchronize", forbidden_sync)

    values = _train(model, loader, optimizer)

    assert events.index("clock") < events.index("loader_iteration")
    assert events[-1] == "clock"
    assert events.count("optimizer_step") == 2
    assert max(index for index, event in enumerate(events) if event == "optimizer_step") < len(events) - 1
    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == 2.5


@pytest.mark.parametrize("units_per_update", [2, 4])
def test_throughput_counts_the_committed_units_of_full_and_last_groups(monkeypatch: pytest.MonkeyPatch, units_per_update: int) -> None:
    """Count the units of every committed group, the shorter last group included."""
    processed_units = 5
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 20.0, 22.0)

    values = _train(model, _loader(processed_units, units_per_update=units_per_update), optimizer)

    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == processed_units / 2.0
    assert values["units"] == processed_units


@pytest.mark.parametrize("stop", [5.0, math.nan, math.inf])
def test_invalid_training_duration_fails_without_publishing_throughput(
    monkeypatch: pytest.MonkeyPatch,
    stop: float,
) -> None:
    """Reject zero and non-finite durations rather than emit misleading values."""
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 5.0, stop)

    with pytest.raises(RuntimeError, match="invalid duration"):
        _train(model, _loader(2, units_per_update=2), optimizer)


@pytest.mark.parametrize("publication_fails", [False, True])
def test_checkpoint_timer_measures_capture_and_publication_outside_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_fails: bool,
) -> None:
    """Measure both checkpoint operations directly and retain partial failure evidence."""
    config = configs.resolve(configs.direct_config())
    config["tracking"]["wandb"]["monitor"]["enabled"] = False
    objective_id = config["evaluation"]["objective"]["id"]
    clock = [10.0]
    events: list[str] = []
    saved: dict[Path, dict[str, Any]] = {}
    timing: dict[str, Any] = {}
    model, optimizer = _model_and_optimizer()
    loader = DataLoader(
        _MappingDataset(1), batch_sampler=UnitSampler(unit_kind="case", cases=(0,), keys=None, units_per_update=1, memory_batch=1, seed=0)
    )
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock[0])

    def train(_model: nn.Module, train_loader: DataLoader[Any], *_args: Any, **kwargs: Any) -> dict[str, Any]:
        clock[0] += 7.0
        loop.train_unit_sampler(train_loader).advance(1)
        kwargs["budget"].record_group(units=1, train_seconds=7.0)
        work = {"optimizer_steps": 1.0, "gradient_predictions": 1.0, "units": 1.0, "update_distinct_cases": 1.0}
        return {"train/loss": 0.5, **work, "update_work": [[1, 1, 1]], "divergence": None}

    def evaluate(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        clock[0] += 11.0
        return {objective_id: 0.25, "nonfinite_cases": 0.0}

    def capture(**kwargs: Any) -> dict[str, Any]:
        events.append(f"capture_{kwargs['role']}")
        clock[0] += 2.0
        return {
            "completed_epoch": kwargs["completed_epoch"],
            "next_epoch": kwargs["completed_epoch"] + 1,
            "best_metric": kwargs["best_metric"],
            "best_epoch": kwargs["best_epoch"],
            "checkpoint_role": kwargs["role"],
        }

    def publish(*publications: tuple[dict[str, Any], Path]) -> None:
        for payload, _path in publications:
            events.append(f"publish_{payload['checkpoint_role']}")
            clock[0] += 3.0
            if publication_fails and payload["checkpoint_role"] == "last":
                message = "bounded checkpoint publication failure"
                raise OSError(message)
        saved.update({path: payload for payload, path in publications})

    def observer(_epoch: int, _values: dict[str, float]) -> None:
        events.append("observer")
        clock[0] += 50.0

    monkeypatch.setattr(loop, "train_update_groups", train)
    monkeypatch.setattr(loop, "eval_one_epoch", evaluate)
    monkeypatch.setattr(loop.checkpoints, "make_checkpoint", capture)
    monkeypatch.setattr(loop.checkpoints, "save_checkpoints", publish)
    monkeypatch.setattr(loop.checkpoints, "load_checkpoint", lambda path, **_kwargs: saved[Path(path)])
    arguments = {
        "config": config,
        "device": torch.device("cpu"),
        "model": model,
        "optimizer": optimizer,
        "train_loader": loader,
        "eval_loader": loader,
        "train_loss": nn.MSELoss(),
        "eval_metrics": {objective_id: object()},
        "save_dir": tmp_path,
        "event_callback": observer,
        "checkpoint_identity": {"test_identity": "synthetic"},
        "operational_timing": timing,
        # One unit is the whole budget, so the first group reaches point 20 and validates once.
        "budget": learning.training.budget.StageBudget(unit="units", total=1),
    }
    if publication_fails:
        with pytest.raises(OSError, match="bounded checkpoint publication failure"):
            loop.train_loop(**arguments)
    else:
        result = loop.train_loop(**arguments)
        assert result["operational_timing"] == timing
        assert events[-1] == "observer"
    assert events[:4] == ["capture_best", "capture_last", "publish_best", "publish_last"]
    assert timing["checkpoint_capture_and_publication_seconds"] == 10.0
    assert timing["checkpoint_capture_and_publication_attempts"] == 2
    assert timing["event_duration_scope"].endswith("excludes_epoch_state_checkpoint_writes_and_observers")
