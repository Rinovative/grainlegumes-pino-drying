"""Protect optimizer-phase timing and actual-sample training throughput.

Responsibilities:
  - Check optimizer-phase timers, actual sample counts, and invalid-duration rejection

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
    """Yield prescribed batches while exposing iterator construction order."""

    def __init__(self, batches: list[dict[str, torch.Tensor]], events: list[str]) -> None:
        self.batches = batches
        self.events = events

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self.events.append("loader_iteration")
        yield from self.batches


def _model_and_optimizer() -> tuple[nn.Module, SGD]:
    """Return a minimal differentiable CPU model and optimizer."""
    model = nn.Linear(1, 1, bias=False)
    return model, SGD(model.parameters(), lr=0.0)


def _fixed_clock(monkeypatch: pytest.MonkeyPatch, start: float, stop: float) -> None:
    """Install one two-reading monotonic clock."""
    ticks = iter((start, stop))
    monkeypatch.setattr(loop.time, "perf_counter", lambda: next(ticks))


def test_cpu_timer_wraps_loader_iteration_and_final_optimizer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start before loader iteration, stop after optimizer work, and never touch CUDA."""
    events: list[str] = []
    batches = [
        {"x": torch.ones(2, 1), "y": torch.zeros(2, 1)},
        {"x": torch.ones(3, 1), "y": torch.zeros(3, 1)},
    ]
    loader = cast("DataLoader[Any]", _RecordingLoader(batches, events))
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

    values = loop.train_one_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
    )

    assert events[0:2] == ["clock", "loader_iteration"]
    assert events[-1] == "clock"
    assert max(index for index, event in enumerate(events) if event == "optimizer_step") < len(events) - 1
    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == 2.5


@pytest.mark.parametrize(
    ("batch_size", "drop_last", "processed_samples"),
    [
        (2, False, 5),
        (4, False, 5),
        (2, True, 4),
        (4, True, 4),
    ],
)
def test_throughput_counts_actual_full_partial_and_retained_samples(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    drop_last: bool,
    processed_samples: int,
) -> None:
    """Use observed batch dimensions across batch sizes and drop-last policies."""
    loader = DataLoader(
        _MappingDataset(5),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
    )
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 20.0, 22.0)

    values = loop.train_one_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
    )

    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == processed_samples / 2.0
    assert values["system/train_samples_per_second"] != len(loader) * batch_size / 2.0 or processed_samples == len(loader) * batch_size


@pytest.mark.parametrize("stop", [5.0, math.nan, math.inf])
def test_invalid_training_duration_fails_without_publishing_throughput(
    monkeypatch: pytest.MonkeyPatch,
    stop: float,
) -> None:
    """Reject zero and non-finite durations rather than emit misleading values."""
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 5.0, stop)
    loader = DataLoader(_MappingDataset(2), batch_size=2)

    with pytest.raises(RuntimeError, match="invalid duration"):
        loop.train_one_epoch(
            model,
            loader,
            optimizer,
            nn.MSELoss(),
            torch.device("cpu"),
        )


@pytest.mark.parametrize("publication_fails", [False, True])
def test_checkpoint_timer_measures_capture_and_publication_outside_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_fails: bool,
) -> None:
    """Measure both checkpoint operations directly and retain partial failure evidence."""
    config = configs.resolve(configs.direct_config())
    config["training"]["epochs"] = 1
    config["tracking"]["wandb"]["monitor"]["enabled"] = False
    objective_id = config["evaluation"]["objective"]["id"]
    clock = [10.0]
    events: list[str] = []
    saved: dict[Path, dict[str, Any]] = {}
    timing: dict[str, Any] = {}
    model, optimizer = _model_and_optimizer()
    loader = DataLoader(_MappingDataset(1), batch_size=1)
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock[0])

    def train(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        clock[0] += 7.0
        return {"train/loss": 0.5, "optimizer_steps": 1.0}

    def evaluate(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        clock[0] += 11.0
        return {objective_id: 0.25}

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

    def publish(payload: dict[str, Any], path: Path) -> None:
        events.append(f"publish_{payload['checkpoint_role']}")
        clock[0] += 3.0
        if publication_fails and payload["checkpoint_role"] == "last":
            message = "bounded checkpoint publication failure"
            raise OSError(message)
        saved[path] = payload

    def observer(_epoch: int, _values: dict[str, float]) -> None:
        events.append("observer")
        clock[0] += 50.0

    monkeypatch.setattr(loop, "train_one_epoch", train)
    monkeypatch.setattr(loop, "eval_one_epoch", evaluate)
    monkeypatch.setattr(loop.checkpoints, "make_checkpoint", capture)
    monkeypatch.setattr(loop.checkpoints, "save_checkpoint", publish)
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
        "epoch_end_callback": observer,
        "checkpoint_identity": {"test_identity": "synthetic"},
        "operational_timing": timing,
    }
    if publication_fails:
        with pytest.raises(OSError, match="bounded checkpoint publication failure"):
            loop.train_loop(**arguments)
    else:
        result = loop.train_loop(**arguments)
        assert result["operational_timing"] == timing
        assert events[-1] == "observer"
    assert events[:4] == ["capture_best", "publish_best", "capture_last", "publish_last"]
    assert timing["checkpoint_capture_and_publication_seconds"] == 10.0
    assert timing["checkpoint_capture_and_publication_attempts"] == 2
    assert timing["epoch_duration_scope"].endswith("excludes_epoch_state_last_checkpoint_and_observers")
