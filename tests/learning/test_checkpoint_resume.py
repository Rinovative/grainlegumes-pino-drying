"""Protect exact epoch-boundary resume and strict best/last checkpoint publication.

Responsibilities:
  - Check exact resumed state, the wall-time last-checkpoint cadence, checkpoint roles and payload
    kinds, transactional restore, evaluation cadence, and objective selection

Design principles:
  - Synthetic deterministic training distinguishes best selection from last resume state

This module does NOT:
  - Allocate run directories or validate high-level inference admission
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from support import configs
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src import common, domain, experiments, learning
from src.datasets.runtime.dataset_runtime_units import UnitSampler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from torch.optim.optimizer import Optimizer

    from src.learning.training.learning_training_checkpoint import CheckpointPayloadKind

_CONTINUATION_KEYS = frozenset(
    {
        "next_epoch",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "train_sampler_state_dict",
    }
)
_NO_WORK = {"optimizer_steps": 0, "gradient_predictions": 0, "units": 0, "update_distinct_cases": 0}


class _MappingDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Provide a fixed twelve-sample regression dataset for exact-resume tests.

    Samples are generated in memory with no normalization or storage behavior.
    deterministic shuffle state is owned by the surrounding DataLoader.
    """

    def __init__(self) -> None:
        self.x = torch.linspace(-1.0, 1.0, 24).reshape(12, 2)
        self.y = torch.stack((self.x[:, 0] - self.x[:, 1], self.x[:, 0] + 0.5 * self.x[:, 1]), dim=1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class _DatasetMSE:
    """
    Accumulate normalized MSE from global squared-error/count evidence.

    The helper mirrors the evaluation metric protocol but omits field selection,
    units, and distributed state so checkpoint tests isolate lifecycle behavior.
    """

    id = "mse"
    space = "normalized"

    def reset(self) -> None:
        """Clear accumulated squared error and element count."""
        self.total = 0.0
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Accumulate one normalized batch using elementwise SSE and count."""
        del batch_index
        assert space == self.space
        difference = pred - target
        self.total += float(torch.sum(difference * difference))
        self.count += difference.numel()

    def compute(self) -> float:
        """Return the global element-mean squared error."""
        return self.total / self.count

    def components(self) -> dict[str, float]:
        """Return no components, like a field metric without unavailable cases."""
        return {}


class _NonFiniteMetric(_DatasetMSE):
    """Metric used to prove a non-finite objective is never eligible."""

    def compute(self) -> float:
        """Return NaN to exercise non-finite objective admission."""
        return math.nan


class _ObjectiveStreamMetric(_DatasetMSE):
    """
    Return a prescribed finite objective sequence across evaluations.

    Parameters
    ----------
    values : list[float]
        One value consumed by each ``compute`` call in evaluation order.

    """

    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def reset(self) -> None:
        """Preserve the prescribed stream across evaluation resets."""

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Validate normalized routing without accumulating batch values."""
        del pred, target, batch_index
        assert space == self.space

    def compute(self) -> float:
        """Consume and return the next prescribed selection objective."""
        return next(self.values)


class _StatefulLoss(nn.Module):
    """MSE with explicit epoch state that exact resume must preserve."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("epoch_index", torch.tensor(-1, dtype=torch.long))

    def set_epoch(self, epoch_index: int) -> None:
        """Persist the completed training epoch in module state."""
        self.epoch_index.fill_(epoch_index)

    def unit_components(self, pred: torch.Tensor, *, x: torch.Tensor, y: torch.Tensor, reference: object = None) -> dict[str, torch.Tensor]:
        """Return every case's mean squared error while retaining serializable epoch state."""
        del x, reference
        return {"total": (pred - y).square().flatten(1).mean(dim=1)}


class _WarmupStatefulLoss(_StatefulLoss):
    """Expose deterministic selected-epoch weights without changing train semantics."""

    def telemetry_state(self, *, epoch: int | None = None) -> dict[str, float]:
        position = int(self.epoch_index.item()) if epoch is None else epoch
        fraction = min(float(position) / 10.0, 1.0)
        return {
            "weight_momentum": fraction,
            "weight_continuity": 3.0 * fraction,
            "weight_boundary": 2.0 * fraction,
        }


class _DivergingLoss(_StatefulLoss):
    """Return NaN from the given training epoch on (set_epoch counts from zero)."""

    def __init__(self, diverge_at_epoch: int) -> None:
        super().__init__()
        self.diverge_at_epoch = diverge_at_epoch

    def unit_components(self, pred: torch.Tensor, *, x: torch.Tensor, y: torch.Tensor, reference: object = None) -> dict[str, torch.Tensor]:
        """Return the MSE until the divergence epoch, then NaN on the prediction graph."""
        components = super().unit_components(pred, x=x, y=y, reference=reference)
        if int(self.epoch_index.item()) >= self.diverge_at_epoch:
            return {"total": components["total"] * math.nan}
        return components


class _NonFiniteTrainingLoss(nn.Module):
    """
    Return one prescribed non-finite value while retaining an autograd path.

    Parameters
    ----------
    value : float
        NaN or signed infinity injected before backward.

    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def unit_components(self, pred: torch.Tensor, *, x: torch.Tensor, y: torch.Tensor, reference: object = None) -> dict[str, torch.Tensor]:
        """Attach the prescribed non-finite value of every case to the prediction graph."""
        del x, y, reference
        return {"total": pred.flatten(1).sum(dim=1) * 0.0 + self.value}


def _objective(*, direction: str = "minimize") -> dict[str, Any]:
    """Return one complete synthetic resolved objective."""
    return {
        "id": "mse",
        "kind": "mse",
        "space": "normalized",
        "fields": ["first", "second"],
        "reduction": "element_mean",
        "direction": direction,
    }


def _config(epochs: int, *, direction: str = "minimize") -> dict[str, Any]:
    """Return the minimal resolved training contract used by loop tests."""
    return {
        "task": "steady_flow",
        "run": {"device": "cpu"},
        "training": {"budget_epochs": epochs},
        "evaluation": {"objective": _objective(direction=direction)},
        "tracking": {
            "wandb": {
                "monitor": {"enabled": False, "max_cases": 1},
            }
        },
    }


def _budget(epochs: int) -> learning.training.budget.StageBudget:
    """
    Return a budget of ``epochs`` passes over the twelve cases.

    The sampler updates once per epoch, so every epoch end crosses at least one of the
    20 validation points and validates exactly once, and the global step equals the epoch.
    """
    return learning.training.budget.StageBudget(unit="units", total=learning.training.budget.passes_budget(epochs, len(_MappingDataset())))


def _identity(run: str = "synthetic", *, direction: str = "minimize") -> dict[str, Any]:
    """
    Build a complete current checkpoint identity namespaced by one run label.

    The task remains steady flow. The run-scoped config, Dataset, split,
    normalizer, and objective dependencies vary together for selection tests.
    """

    def digest(label: str) -> str:
        return hashlib.sha256(f"{run}:{label}".encode()).hexdigest()

    return {
        "identity_contract_sha256": learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "steady_flow",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "dataset_ids": {"train": f"{run}-train-dataset", "material_ood": f"{run}-ood-dataset", "parameter_ood": f"{run}-pood-dataset"},
        "dataset_fingerprints": {"train": digest("train"), "material_ood": digest("ood"), "parameter_ood": digest("pood")},
        "split_contract_digest": digest("split-contract"),
        "split_membership_digests": {
            "train": digest("train-membership"),
            "validation": digest("validation-membership"),
            "id_test": digest("id-test-membership"),
            "material_ood": digest("ood-membership"),
            "parameter_ood": digest("pood-membership"),
        },
        "normalizer_sha256": digest("normalizer"),
        "numerical_environment": learning.training.checkpoint.numerical_environment_identity(),
        "objective": _objective(direction=direction),
    }


def _components(seed: int) -> tuple[nn.Module, Optimizer, Any, nn.Module, DataLoader[Any], DataLoader[Any]]:
    """
    Construct deterministically seeded model, optimizer, scheduler, loss, and loaders.

    The training loader is driven by a unit sampler whose state is part of exact
    checkpoint continuation. All components remain CPU-local and synthetic.
    """
    experiments.run.seed_process(seed, device=torch.device("cpu"))
    dataset = _MappingDataset()
    train_loader = DataLoader(dataset, batch_sampler=_sampler(seed + 1))
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Dropout(0.2), nn.Linear(8, 2))
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.8)
    scheduler = learning.training.optim.CosineSchedule(optimizer, start_rate=0.05)
    return model, optimizer, scheduler, _StatefulLoss(), train_loader, eval_loader


def _sampler(seed: int) -> UnitSampler:
    """Return the unit sampler of the twelve steady cases: one update of all twelve per epoch, in microbatches of four."""
    return UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=12, memory_batch=4, seed=seed)


def _commit_epoch(loader: DataLoader[Any]) -> None:
    """Commit every update group of the sampler's next epoch without training."""
    sampler = loader.batch_sampler
    assert isinstance(sampler, UnitSampler)
    sampler.begin_epoch(sampler.epoch + 1)
    for group in sampler.groups():
        sampler.advance(len(group.items))


def _mse_step(model: nn.Module, optimizer: Optimizer, loss: Any) -> None:
    """Take one optimizer step on the first three cases."""
    dataset = _MappingDataset()
    optimizer.zero_grad()
    loss.unit_components(model(dataset.x[:3]), x=dataset.x[:3], y=dataset.y[:3])["total"].mean().backward()
    optimizer.step()


def _sampler_state(loader: DataLoader[Any]) -> dict[str, Any]:
    """Return the persisted state of a training loader's unit sampler."""
    sampler = loader.batch_sampler
    assert isinstance(sampler, UnitSampler)
    return sampler.state_dict()


def _run(
    run_dir: Path,
    *,
    epochs: int,
    seed: int,
    resume_from: Path | None = None,
    event_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, Any], nn.Module, Optimizer, Any, nn.Module, DataLoader[Any]]:
    """
    Execute one synthetic training segment, optionally from ``last_checkpoint.pt``.

    Checkpoints are published below ``run_dir`` and the returned live components
    expose state for exact comparison with an uninterrupted segment.
    """
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(seed)
    result = learning.training.loop.train_loop(
        config=_config(epochs),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        resume_from=resume_from,
        event_callback=event_callback,
        checkpoint_identity=_identity(),
        budget=_budget(epochs),
    )
    return result, model, optimizer, scheduler, loss, train_loader


def _without_clock(values: dict[str, Any]) -> dict[str, Any]:
    """Drop the measured operational timing and optimisation seconds, which vary between invocations, from a result or budget."""
    kept = {key: value for key, value in values.items() if key not in {"operational_timing", "train_seconds"}}
    if "budget" in kept:
        kept["budget"] = _without_clock(kept["budget"])
    return kept


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Compare nested tensor, NumPy, collection, and scalar state exactly."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_uninterrupted_and_resumed_training_are_state_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Compare one uninterrupted four-epoch run with a run interrupted after epoch 2 and resumed.

    Model, optimizer, cosine rates, budget, loss warmup, loader epoch, RNG, objective, and
    history state must match, proving ``last_checkpoint.pt`` is sufficient continuation.
    """
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    full_dir.mkdir()
    resumed_dir.mkdir()

    full_rates: list[float] = []
    full_result, full_model, full_optimizer, full_scheduler, full_loss, full_loader = _run(
        full_dir, epochs=4, seed=151, event_callback=lambda _step, values: full_rates.append(values["optimization/learning_rate"])
    )
    resume_observations: list[tuple[int, float, float, float]] = []
    resumed_rates: list[float] = []

    def capture_resume_epoch(step: int, values: dict[str, float]) -> None:
        resumed_rates.append(values["optimization/learning_rate"])
        resume_observations.append(
            (
                step,
                values["system/event_duration_seconds"],
                values["system/train_duration_seconds"],
                values["system/train_samples_per_second"],
            )
        )
        if step == 2 and len(resume_observations) == 2:
            raise _StopRunError

    # Every event publishes a last, so the interruption after epoch 2 leaves its exact state.
    monkeypatch.setattr(learning.training.loop, "LAST_CHECKPOINT_INTERVAL_SECONDS", 0.0)
    with pytest.raises(_StopRunError):
        _run(resumed_dir, epochs=4, seed=151, event_callback=capture_resume_epoch)
    # The interrupted last holds the rate of its own committed progress, not the one its last group trained with.
    interrupted = torch.load(resumed_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    interrupted_progress = interrupted["budget_state_dict"]["units"] / interrupted["budget_state_dict"]["total"]
    assert interrupted_progress == 0.5
    assert interrupted["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.05 * learning.training.optim.cosine_factor(interrupted_progress)
    resumed_result, resumed_model, resumed_optimizer, resumed_scheduler, resumed_loss, resumed_loader = _run(
        resumed_dir,
        epochs=4,
        seed=999,
        resume_from=resumed_dir / "last_checkpoint.pt",
        event_callback=capture_resume_epoch,
    )

    assert [step for step, _event_duration, _train_duration, _throughput in resume_observations] == [1, 2, 3, 4]
    assert resumed_rates == full_rates
    # Each event reports η(p) of the committed progress (one epoch of four per event), so the budget end reports 0.
    expected_rates = [0.05 * learning.training.optim.cosine_factor(epoch / 4) for epoch in range(1, 5)]
    assert full_rates == pytest.approx(expected_rates, abs=1e-15)
    assert full_rates[-1] == 0.0
    assert all(earlier > later for earlier, later in itertools.pairwise(full_rates))
    assert all(
        math.isfinite(value) and value > 0.0
        for _epoch, epoch_duration, train_duration, throughput in resume_observations
        for value in (epoch_duration, train_duration, throughput)
    )
    # Direct operational clocks vary between invocations; every scientific
    # result and all model/optimizer/checkpoint/RNG states remain exact below.
    assert resumed_result["operational_timing"]["scope"] == full_result["operational_timing"]["scope"]
    assert _without_clock(resumed_result) == _without_clock(full_result) | {
        "checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "best_checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "last_checkpoint_path": str(resumed_dir / "last_checkpoint.pt"),
    }
    # Four epochs of one twelve-case group; the resumed segment continues the counts of its checkpoint.
    work = resumed_result["training_work"]
    assert work == full_result["training_work"]
    assert (work["optimizer_steps"], work["gradient_predictions"], work["units"]) == (4, 48, 48)
    # Per update: units, distinct cases, supervised transitions; the resumed segment appends to its checkpoint's record.
    assert resumed_result["update_work"] == full_result["update_work"] == [[12, 12, 12]] * 4
    _assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())
    _assert_nested_equal(full_loss.state_dict(), resumed_loss.state_dict())
    _assert_nested_equal(_sampler_state(full_loader), _sampler_state(resumed_loader))

    full_last = torch.load(full_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    resumed_last = torch.load(resumed_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    for key in (
        "completed_epoch",
        "next_epoch",
        "global_step",
        "best_metric",
        "best_epoch",
        "objective_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "train_sampler_state_dict",
        "training_work",
        "update_work",
        "scheduler_state_dict",
    ):
        _assert_nested_equal(full_last[key], resumed_last[key])
    assert _without_clock(full_last["budget_state_dict"]) == _without_clock(resumed_last["budget_state_dict"])
    assert full_last["budget_state_dict"]["validated_point"] == learning.training.budget.VALIDATION_POINTS


def test_caller_owned_attempt_work_holds_the_work_of_the_newest_last_when_an_event_callback_stops_the_run(tmp_path: Path) -> None:
    """A run stopped from its event callback leaves its caller the work it started from and the work its newest last checkpoint holds."""

    def stop_after_the_second_epoch(step: int, _values: dict[str, float]) -> None:
        if step == 2:
            raise _StopRunError

    work = learning.training.loop.AttemptWork()
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(151)
    budget = _budget(4)
    with pytest.raises(_StopRunError):
        learning.training.loop.train_loop(
            config=_config(4),
            device=torch.device("cpu"),
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_loss=loss,
            # Epoch 2 brings no new best, so only epoch 1 has a last checkpoint when the run stops.
            eval_metrics={"mse": _ObjectiveStreamMetric([0.5, 0.6, 0.7, 0.8])},
            save_dir=tmp_path,
            event_callback=stop_after_the_second_epoch,
            checkpoint_identity=_identity("attempt-work"),
            attempt_work=work,
            budget=budget,
        )

    assert work.at_start == {"optimizer_steps": 0, "gradient_predictions": 0, "units": 0, "update_distinct_cases": 0, "train_seconds": 0.0}
    last = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    durable = work.durable
    assert durable is not None
    assert durable == {**last["training_work"], "train_seconds": last["budget_state_dict"]["train_seconds"]}
    # One epoch of one twelve-case group is committed; the second was computed and a resume trains it again.
    assert (last["training_work"]["optimizer_steps"], last["training_work"]["units"]) == (1, 12)
    assert budget.train_seconds > durable["train_seconds"] > 0.0


class _StopRunError(Exception):
    """Stop a synthetic run from its epoch callback."""


@pytest.mark.parametrize(
    ("epochs", "values", "seconds_per_epoch", "best_epochs", "last_epochs"),
    [
        pytest.param(5, [0.5, 0.6, 0.4, 0.45, 0.47], 0.0, [1, 3], [1, 3, 5], id="new-best-and-terminal"),
        pytest.param(4, [0.5, 0.6, 0.7, 0.8], 1000.0, [1], [1, 3, 4], id="interval-elapsed"),
    ],
)
def test_last_is_published_at_new_bests_elapsed_intervals_and_the_terminal_epoch_only(
    epochs: int,
    values: list[float],
    seconds_per_epoch: float,
    best_epochs: list[int],
    last_epochs: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first epoch and every new best get a ``last``; otherwise ``last`` waits for the 30-minute interval or the loop's end."""
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(23)
    published: list[tuple[str, int]] = []
    flags: list[float] = []
    clock = [0.0]
    real_save = learning.training.checkpoint.save_checkpoints

    def record(*publications: tuple[dict[str, Any], Path]) -> None:
        published.extend((payload["checkpoint_role"], payload["completed_epoch"]) for payload, _path in publications)
        real_save(*publications)

    def observe(_epoch: int, metrics: dict[str, float]) -> None:
        flags.append(metrics["checkpoint/last_published"])
        clock[0] += seconds_per_epoch

    monkeypatch.setattr(learning.training.checkpoint, "save_checkpoints", record)
    monkeypatch.setattr(learning.training.loop.time, "monotonic", lambda: clock[0])
    result = learning.training.loop.train_loop(
        config=_config(epochs),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric(values)},
        save_dir=tmp_path,
        event_callback=observe,
        checkpoint_identity=_identity("last-cadence"),
        budget=_budget(epochs),
    )

    assert result["status"] == "completed"
    assert [epoch for role, epoch in published if role == "best"] == best_epochs
    assert [epoch for role, epoch in published if role == "last"] == last_epochs
    assert flags == [float(epoch in last_epochs) for epoch in range(1, epochs + 1)]
    assert result["operational_timing"]["checkpoint_capture_and_publication_attempts"] == len(published)


def test_interrupted_run_resumes_from_its_newest_last_and_trains_later_epochs_again(tmp_path: Path) -> None:
    """
    Interrupt after epoch 3 and resume from the newest ``last``.

    The best is not newer than that ``last``, and the resumed run trains every epoch
    after it again and ends in the uninterrupted run's exact terminal state.
    """
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    full_dir.mkdir()
    resumed_dir.mkdir()
    full_result, full_model, full_optimizer, _scheduler, _loss, _loader = _run(full_dir, epochs=5, seed=151)
    flags: list[float] = []

    def interrupt_after_third(epoch: int, values: dict[str, float]) -> None:
        flags.append(values["checkpoint/last_published"])
        if epoch == 3:
            message = "synthetic interruption"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(resumed_dir, epochs=5, seed=151, event_callback=interrupt_after_third)
    # Epoch 1 publishes the first best with the first last; a later epoch without a new best publishes neither.
    newest_last = max(epoch for epoch, flag in enumerate(flags, start=1) if flag == 1.0)
    assert flags[0] == 1.0
    last = torch.load(resumed_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    best = torch.load(resumed_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert last["completed_epoch"] == newest_last
    assert best["best_epoch"] <= last["completed_epoch"]

    resumed_result, resumed_model, resumed_optimizer, _scheduler, _loss, _loader = _run(
        resumed_dir, epochs=5, seed=999, resume_from=resumed_dir / "last_checkpoint.pt"
    )

    assert _without_clock(resumed_result) == _without_clock(full_result) | {
        "checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "best_checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "last_checkpoint_path": str(resumed_dir / "last_checkpoint.pt"),
    }
    _assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())


def _history_loop(
    run_dir: Path,
    objectives: list[float],
    *,
    units_per_update: int,
    resume: bool = False,
    event_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Train two epochs of ``units_per_update``-case groups through the real history writer, reconciled to ``last`` on resume."""
    identity = _identity("history")
    binding: dict[str, Any] = {"task": "steady_flow", "run_name": "history", "checkpoint_identity": identity}
    last_path = run_dir / "last_checkpoint.pt"
    saved = (
        learning.training.checkpoint.load_checkpoint(
            last_path, expected_identity=identity, expected_role="last", payload_kind="full", scheduler_expected=False, require_best=False
        )
        if resume
        else None
    )
    initial = learning.training.history.reconcile_history(
        run_dir,
        **binding,
        completed_epoch=0 if saved is None else learning.training.checkpoint.recorded_history_epochs(saved),
        validations=[] if saved is None else saved["objective_history"],
    )
    experiments.run.seed_process(41, device=torch.device("cpu"))
    dataset = _MappingDataset()
    sampler = UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=units_per_update, memory_batch=units_per_update, seed=42)
    model = nn.Linear(2, 2)
    return learning.training.loop.train_loop(
        config=_config(2),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.05),
        train_loader=DataLoader(dataset, batch_sampler=sampler),
        eval_loader=DataLoader(dataset, batch_size=4, shuffle=False),
        train_loss=_StatefulLoss(),
        eval_metrics={"mse": _ObjectiveStreamMetric(objectives)},
        save_dir=run_dir,
        resume_from=last_path if resume else None,
        event_callback=event_callback,
        epoch_state_callback=learning.training.history.make_epoch_state_callback(run_dir, **binding, initial_history=initial),
        checkpoint_identity=identity,
        budget=_budget(2),
    )


def _completed_history(run_dir: Path, *, epochs: int, validations: Sequence[Mapping[str, Any]] | Sequence[int]) -> dict[str, Any]:
    """Admit the history of ``_history_loop`` exactly at ``epochs`` epoch records and its objective history or bare points."""
    return learning.training.history.validate_completed_history(
        run_dir, task="steady_flow", run_name="history", checkpoint_identity=_identity("history"), completed_epoch=epochs, validations=validations
    )


@pytest.mark.parametrize(
    ("units_per_update", "points"),
    [(6, [5, 10, 15, 20]), (4, [3, 6, 10, 13, 16, 20])],
    ids=("one-validation-inside-each-epoch", "two-validations-inside-each-epoch"),
)
def test_the_history_records_every_validation_point_once_also_inside_an_epoch(tmp_path: Path, units_per_update: int, points: list[int]) -> None:
    """Every validation point, also one inside an epoch, has one record at its epoch and step beside one record per epoch."""
    result = _history_loop(tmp_path, [1.0] * len(points), units_per_update=units_per_update)

    # The producer's records bind to the objective history the loop returns: points, epochs, eligibility and values.
    history = _completed_history(tmp_path, epochs=2, validations=result["objective_history"])
    assert [(record["validation/point"], record["epoch"]) for record in history["validations"]] == [
        (float(point), 1.0 + (point > 10)) for point in points
    ]
    assert all("id/mse" in record and "global_step" in record for record in history["validations"])
    assert [record["epoch"] for record in history["epochs"]] == [1.0, 2.0]
    assert not any("validation/point" in record for record in history["epochs"])


def test_resume_from_a_last_inside_an_epoch_restores_exactly_the_validations_it_counts(tmp_path: Path) -> None:
    """
    A new best at point 6 publishes ``last`` inside epoch 1 after its second validation; the run stops after point 10.

    Point 10 and epoch 1 are recorded ahead of that ``last``: the resume truncates them, trains them again and
    records every point once.
    """

    def stop_at_the_first_epoch_end(step: int, _values: dict[str, float]) -> None:
        if step == 3:
            raise _StopRunError

    with pytest.raises(_StopRunError):
        _history_loop(tmp_path, [3.0, 2.0, 2.5], units_per_update=4, event_callback=stop_at_the_first_epoch_end)
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    points = [entry["point"] for entry in saved["objective_history"]]
    assert (points, learning.training.checkpoint.recorded_history_epochs(saved)) == ([3, 6], 0)
    _completed_history(tmp_path, epochs=1, validations=[3, 6, 10])

    result = _history_loop(tmp_path, [2.5, 2.4, 2.3, 2.2], units_per_update=4, resume=True)

    history = _completed_history(tmp_path, epochs=2, validations=result["objective_history"])
    assert [(record["validation/point"], record["epoch"]) for record in history["validations"]] == [
        (3.0, 1.0),
        (6.0, 1.0),
        (10.0, 1.0),
        (13.0, 2.0),
        (16.0, 2.0),
        (20.0, 2.0),
    ]


def _streamed_loop(run_dir: Path, values: list[float], **kwargs: Any) -> dict[str, Any]:
    """Train one evaluation-every-epoch segment whose objective stream fixes every new best."""
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(37)
    return learning.training.loop.train_loop(
        config=_config(3),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric(values)},
        save_dir=run_dir,
        checkpoint_identity=_identity("paired-publication"),
        budget=_budget(3),
        **kwargs,
    )


def test_failed_last_write_of_a_new_best_epoch_keeps_the_previous_best_and_last_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fail the last write of an epoch whose new best is already serialized.

    Neither file may change and no staged temporary may remain, so the previous
    pair still agrees on the selected objective and the run resumes to completion.
    """
    previous: dict[str, bytes] = {}
    real_torch_save = torch.save

    def fail_second_last(payload: Any, stream: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(payload, dict) and payload.get("checkpoint_role") == "last" and payload.get("completed_epoch") == 2:
            stream.write(b"partial")
            message = "injected last-write failure"
            raise OSError(message)
        real_torch_save(payload, stream, *args, **kwargs)

    def keep_pair(_epoch: int, _values: dict[str, float]) -> None:
        previous.update({name: (tmp_path / name).read_bytes() for name in ("best_checkpoint.pt", "last_checkpoint.pt")})

    monkeypatch.setattr(torch, "save", fail_second_last)
    with pytest.raises(OSError, match="injected last-write failure"):
        _streamed_loop(tmp_path, [0.5, 0.4], event_callback=keep_pair)
    monkeypatch.undo()

    assert {name: (tmp_path / name).read_bytes() for name in previous} == previous
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".")] == []
    best = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert (best["best_epoch"], best["best_metric"]) == (last["best_epoch"], last["best_metric"]) == (1, 0.5)
    resumed = _streamed_loop(tmp_path, [0.4, 0.3], resume_from=tmp_path / "last_checkpoint.pt")
    assert (resumed["status"], resumed["best_epoch"], resumed["best_metric"]) == ("completed", 3, 0.3)


def test_failure_between_the_two_renames_leaves_a_best_newer_than_last_that_finalization_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fail the rename of ``last`` after the new best of epoch 2 is already renamed.

    This is the one window the paired publication leaves open: ``best`` is newer
    than ``last`` and no staged file remains. Run admission refuses to resume such
    a pair; a loop that reaches completed finalization with it rejects it too.
    """
    real_replace = type(tmp_path).replace
    last_renames: list[Path] = []

    def fail_second_last_rename(source: Path, target: Path | str) -> Path:
        if type(source)(target).name == "last_checkpoint.pt":
            last_renames.append(source)
            if len(last_renames) == 2:
                message = "injected last-rename failure"
                raise OSError(message)
        return real_replace(source, target)

    monkeypatch.setattr(type(tmp_path), "replace", fail_second_last_rename)
    with pytest.raises(OSError, match="injected last-rename failure"):
        _streamed_loop(tmp_path, [0.5, 0.4])
    monkeypatch.undo()

    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".")] == []
    best = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert (best["best_epoch"], last["completed_epoch"], last["best_epoch"]) == (2, 1, 1)
    with pytest.raises(RuntimeError, match="Best and last checkpoints disagree"):
        _streamed_loop(tmp_path, [0.6, 0.7], resume_from=tmp_path / "last_checkpoint.pt")


def _trial_loop(run_dir: Path, values: list[float], **kwargs: Any) -> dict[str, Any]:
    """Train the streamed segment as an Optuna trial, whose config records a trial layout."""
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(37)
    config = _config(3)
    config["run"]["layout"] = {"trial": "0000__study"}
    return learning.training.loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric(values)},
        save_dir=run_dir,
        checkpoint_identity=_identity("trial"),
        budget=_budget(3),
        **kwargs,
    )


def test_trial_publishes_each_new_best_by_one_rename_and_never_a_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial never resumes, so it keeps only its best checkpoint and its result names no last."""
    published: list[tuple[str, int]] = []
    flags: list[float] = []
    real_save = learning.training.checkpoint.save_checkpoints

    def record(*publications: tuple[dict[str, Any], Path]) -> None:
        assert len(publications) == 1
        published.extend((payload["checkpoint_role"], payload["completed_epoch"]) for payload, _path in publications)
        real_save(*publications)

    monkeypatch.setattr(learning.training.checkpoint, "save_checkpoints", record)
    result = _trial_loop(tmp_path, [0.5, 0.4, 0.6], event_callback=lambda _epoch, metrics: flags.append(metrics["checkpoint/last_published"]))

    assert (result["status"], result["best_epoch"], result["last_checkpoint_path"], result["next_epoch"]) == ("completed", 2, None, None)
    assert published == [("best", 1), ("best", 2)]
    assert flags == [0.0, 0.0, 0.0]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best_checkpoint.pt"]


def test_failed_best_rename_of_a_trial_keeps_the_best_it_already_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one rename there is no best-newer-than-last window: a failed publication leaves the previously reported best whole."""
    reported: list[tuple[int, float]] = []
    real_replace = type(tmp_path).replace
    best_renames: list[Path] = []

    def fail_second_best_rename(source: Path, target: Path | str) -> Path:
        if type(source)(target).name == "best_checkpoint.pt":
            best_renames.append(source)
            if len(best_renames) == 2:
                message = "injected best-rename failure"
                raise OSError(message)
        return real_replace(source, target)

    monkeypatch.setattr(type(tmp_path), "replace", fail_second_best_rename)
    with pytest.raises(OSError, match="injected best-rename failure"):
        _trial_loop(tmp_path, [0.5, 0.4], event_callback=lambda epoch, metrics: reported.append((epoch, metrics["id/mse"])))
    monkeypatch.undo()

    best = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert reported == [(1, 0.5)]
    assert (best["best_epoch"], best["best_metric"]) == (1, 0.5)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best_checkpoint.pt"]


def test_failed_staged_cleanup_never_masks_the_publication_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged file that cannot be removed after a failed write leaves the write's own error to propagate."""
    real_torch_save = torch.save

    def fail_last(payload: Any, stream: Any, *args: Any, **kwargs: Any) -> None:
        if payload.get("checkpoint_role") == "last":
            message = "injected last-write failure"
            raise OSError(message)
        real_torch_save(payload, stream, *args, **kwargs)

    def refuse_unlink(_path: Path, *, missing_ok: bool = False) -> None:
        del missing_ok
        message = "injected cleanup failure"
        raise PermissionError(message)

    monkeypatch.setattr(torch, "save", fail_last)
    monkeypatch.setattr(type(tmp_path), "unlink", refuse_unlink)
    with pytest.raises(OSError, match="injected last-write failure"):
        learning.training.checkpoint.save_checkpoints(
            ({"checkpoint_role": "best"}, tmp_path / "best_checkpoint.pt"),
            ({"checkpoint_role": "last"}, tmp_path / "last_checkpoint.pt"),
        )
    assert not (tmp_path / "best_checkpoint.pt").exists()


def test_next_invocation_drops_staged_checkpoints_a_crash_left_unrenamed(tmp_path: Path) -> None:
    """A finalize-only resume publishes nothing, so only the loop-start cleanup can remove crash leftovers."""
    _run(tmp_path, epochs=2, seed=151)
    leftovers = [tmp_path / f".{name}.staged" for name in ("best_checkpoint.pt", "last_checkpoint.pt")]
    # A kill inside the staged write leaves the atomic writer's random temporary instead.
    leftovers += [tmp_path / f".{leftover.name}.k3x9q2.tmp" for leftover in leftovers]
    for leftover in leftovers:
        leftover.write_bytes(b"crash leftover")

    resumed, _model, _optimizer, _scheduler, _loss, _loader = _run(tmp_path, epochs=2, seed=999, resume_from=tmp_path / "last_checkpoint.pt")

    assert resumed["status"] == "completed"
    assert not any(leftover.exists() for leftover in leftovers)


def test_resume_at_the_final_epoch_boundary_finalizes_without_training(tmp_path: Path) -> None:
    """
    Resume a run whose last checkpoint already completed every configured epoch.

    Such a run was interrupted after its terminal checkpoints but before
    finalization; resuming it must validate and return the published state
    without another epoch, optimizer step or epoch event.
    """
    first, _model, _optimizer, _scheduler, _loss, _loader = _run(tmp_path, epochs=2, seed=151)
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    events: list[int] = []

    resumed, model, _optimizer, _scheduler, _loss, _loader = _run(
        tmp_path,
        epochs=2,
        seed=999,
        resume_from=tmp_path / "last_checkpoint.pt",
        event_callback=lambda epoch, _values: events.append(epoch),
    )

    assert resumed["status"] == "completed"
    assert events == []
    for key in ("completed_epoch", "global_step", "best_epoch", "best_metric", "objective_history"):
        assert resumed[key] == first[key]
    _assert_nested_equal(model.state_dict(), saved["model_state_dict"])


def test_best_and_last_have_distinct_enforced_roles(tmp_path: Path) -> None:
    """
    Train three epochs and load both published checkpoint roles explicitly.

    Each file must validate only for its declared role. Treating ``last`` as
    ``best`` must fail so inference selection and continuation remain distinct.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=3, seed=31)
    identity = _identity()

    best = learning.training.checkpoint.load_checkpoint(
        run_dir / "best_checkpoint.pt",
        expected_identity=identity,
        expected_role="best",
        payload_kind="evaluation",
        scheduler_expected=True,
        require_best=True,
    )
    last = learning.training.checkpoint.load_checkpoint(
        run_dir / "last_checkpoint.pt",
        expected_identity=identity,
        expected_role="last",
        payload_kind="full",
        scheduler_expected=True,
        require_best=True,
    )

    assert best["checkpoint_role"] == "best"
    assert last["checkpoint_role"] == "last"
    assert last["completed_epoch"] == 3
    # A steady best is never continued, so it keeps no optimizer, scheduler, RNG, or loader state.
    assert (best["payload_kind"], last["payload_kind"]) == ("evaluation", "full")
    assert _CONTINUATION_KEYS.isdisjoint(best)
    assert _CONTINUATION_KEYS.issubset(last)
    with pytest.raises(ValueError, match="role mismatch"):
        learning.training.checkpoint.load_checkpoint(
            run_dir / "last_checkpoint.pt",
            expected_identity=identity,
            expected_role="best",
            payload_kind="full",
            scheduler_expected=True,
            require_best=True,
        )


def test_the_training_identity_binds_the_held_out_evaluation_gate() -> None:
    """Every checkpoint identity and training fingerprint records that training and tuning never evaluate held-out roles.

    The rule is part of the identity contract, so a checkpoint of the earlier protocol carries another contract digest
    and is refused (``former_resume_contract`` below), and analysis never mixes the two protocols silently.
    """
    contract = learning.training.checkpoint._TRAINING_IDENTITY_CONTRACT
    assert contract["held_out_evaluation"] == "never_in_training_or_tuning; id_test_and_ood_only_through_the_explicit_artifacts_call"
    assert common.serialization.canonical_json_sha256(contract) == learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST


@pytest.mark.parametrize("mismatch", ["config", "numerical_environment", "former_resume_contract", "other_a0_lineage"])
def test_schema_or_identity_failure_precedes_runtime_mutation(tmp_path: Path, mismatch: str) -> None:
    """
    Replace the saved checkpoint identity before restoring into fresh components.

    Validation must fail before any model parameter changes, protecting runtime
    state from partial application of an incompatible continuation payload. A
    checkpoint of the former identity contract, whose resume digest left out the
    epoch budget so that a run could be extended, is such a payload, and so is a
    B checkpoint whose own settings are unchanged but whose A0 lineage differs.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=2, seed=7)
    payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    expected = _identity()
    if mismatch == "config":
        payload["identity"] = _identity("other")
    elif mismatch == "other_a0_lineage":
        saved_b = dict(configs.resolve_plan(configs.transient_matched_config()).b)
        other_b = copy.deepcopy(saved_b)
        other_b["run"]["layout"]["a0_id"] = "f" * 16
        payload["identity"]["effective_config_digest"] = learning.training.checkpoint.config_digest(saved_b)
        expected["effective_config_digest"] = learning.training.checkpoint.config_digest(other_b)
    elif mismatch == "former_resume_contract":
        payload["identity"] = {**payload["identity"], "identity_contract_sha256": "0" * 64, "resume_contract_digest": "1" * 64}
    else:
        payload["identity"]["numerical_environment"]["numpy"] = "changed-numerical-runtime"

    model, optimizer, scheduler, loss, train_loader, _ = _components(99)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match=r"identity is incompatible|identity keys do not match"):
        learning.training.checkpoint.restore_checkpoint(
            payload,
            expected_identity=expected,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            train_loader=train_loader,
            budget=_budget(2),
        )
    _assert_nested_equal(before, model.state_dict())


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf],
    ids=("nan", "infinity"),
)
def test_non_finite_training_loss_precedes_backward(value: float) -> None:
    """
    Vary training loss across NaN and infinity before backward.

    Each non-finite category ends the stretch as a divergence with parameters, gradients,
    and optimizer state untouched: the faulty update is neither applied nor skipped over.
    """
    model, optimizer, _scheduler, _loss, train_loader, _eval_loader = _components(73)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    train_loader.batch_sampler.begin_epoch(0)  # type: ignore[union-attr]
    budget = _budget(4)
    result = learning.training.loop.train_update_groups(
        model,
        train_loader,
        optimizer,
        _NonFiniteTrainingLoss(value),
        torch.device("cpu"),
        learning.training.loop._SteadyUnits(None, torch.device("cpu")),
        budget=budget,
    )

    assert "non-finite before backward" in result["divergence"]
    assert result["optimizer_steps"] == 0.0
    assert budget.units == 0
    _assert_nested_equal(model_before, model.state_dict())
    _assert_nested_equal(optimizer_before, optimizer.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_checkpoint_ignores_unrelated_host_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mock CUDA as available while forbidding its RNG query during CPU checkpoint creation.

    The payload must contain an empty CUDA RNG list, keeping checkpoint contents
    determined by the resolved runtime device rather than unrelated host capability.
    """
    model, optimizer, scheduler, loss, train_loader, _ = _components(81)
    _commit_epoch(train_loader)

    def unexpected_cuda_capture(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        msg = "CPU checkpoint queried CUDA RNG"
        raise AssertionError(msg)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected_cuda_capture)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        payload_kind="full",
        identity=_identity("cpu-portable"),
        completed_epoch=1,
        global_step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=train_loader,
        runtime_device=torch.device("cpu"),
        training_work=_NO_WORK,
        update_work=[],
        budget=_budget(4),
    )

    assert payload["torch_cuda_rng_states"] == []


def test_cpu_restore_is_portable_when_checkpoint_contains_cuda_rng() -> None:
    """
    Add synthetic CUDA RNG evidence to an otherwise CPU-created checkpoint.

    CPU restoration must ignore that optional state and recover model/global-step
    values, preserving portability when checkpoints move away from GPU hosts.
    """
    source_model, source_optimizer, source_scheduler, source_loss, source_loader, _ = _components(82)
    _commit_epoch(source_loader)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        payload_kind="full",
        identity=_identity("device-portable"),
        completed_epoch=1,
        global_step=0,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        loss=source_loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work=_NO_WORK,
        update_work=[],
        budget=_budget(4),
    )
    payload["torch_cuda_rng_states"] = [torch.get_rng_state().clone()]

    model, optimizer, scheduler, loss, train_loader, _ = _components(83)
    restored = learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=_identity("device-portable"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        train_loader=train_loader,
        budget=_budget(4),
    )

    assert restored["global_step"] == 0
    _assert_nested_equal(model.state_dict(), source_model.state_dict())


def test_late_restore_failure_rolls_back_every_runtime_state() -> None:
    """
    Inject an invalid loss state after all earlier checkpoint components validate.

    Restore must roll back model, optimizer, scheduler, loss, unit sampler,
    and every process RNG exactly, proving continuation is transactional.
    """
    source_model, source_optimizer, source_scheduler, source_loss, source_loader, _ = _components(84)
    _commit_epoch(source_loader)
    _mse_step(source_model, source_optimizer, source_loss)
    source_scheduler.advance(0.25)
    source_loss.set_epoch(3)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        payload_kind="full",
        identity=_identity("transactional"),
        completed_epoch=1,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        loss=source_loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work=_NO_WORK,
        update_work=[],
        budget=_budget(4),
    )
    payload["loss_state_dict"] = {"unexpected": torch.tensor(1.0)}

    model, optimizer, scheduler, loss, train_loader, _ = _components(85)
    _commit_epoch(train_loader)
    _mse_step(model, optimizer, loss)
    scheduler.advance(0.5)
    loss.set_epoch(7)
    budget = _budget(4)
    budget.record_group(units=12, supervised_transitions=12, train_seconds=1.0)
    before = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "budget": budget.state_dict(),
        "loss": copy.deepcopy(loss.state_dict()),
        "loader": _sampler_state(train_loader),
        "python_rng": random.getstate(),
        "numpy_rng": copy.deepcopy(np.random.get_state()),  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
        "torch_rng": torch.get_rng_state().clone(),
    }

    with pytest.raises(RuntimeError, match="Unexpected key"):
        learning.training.checkpoint.restore_checkpoint(
            payload,
            expected_identity=_identity("transactional"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            train_loader=train_loader,
            budget=budget,
        )

    _assert_nested_equal(before["model"], model.state_dict())
    _assert_nested_equal(before["optimizer"], optimizer.state_dict())
    _assert_nested_equal(before["scheduler"], scheduler.state_dict())
    _assert_nested_equal(before["budget"], budget.state_dict())
    _assert_nested_equal(before["loss"], loss.state_dict())
    _assert_nested_equal(before["loader"], _sampler_state(train_loader))
    _assert_nested_equal(before["python_rng"], random.getstate())
    _assert_nested_equal(before["numpy_rng"], np.random.get_state())  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
    assert torch.equal(before["torch_rng"], torch.get_rng_state())


def test_non_finite_objective_is_never_eligible_and_training_continues(tmp_path: Path) -> None:
    """
    Evaluate one training epoch with a metric that returns NaN at every validation point.

    No point is eligible, training still runs its whole budget, no best checkpoint is
    published, and the run ends as ``no_eligible_checkpoint`` with zero eligible points.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(5)

    result = learning.training.loop.train_loop(
        config={
            "task": "steady_flow",
            "run": {"device": "cpu"},
            "training": {"budget_epochs": 1},
            "evaluation": {"objective": _objective()},
            "tracking": {"wandb": {"monitor": {"enabled": False, "max_cases": 1}}},
        },
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _NonFiniteMetric()},
        scheduler=scheduler,
        save_dir=run_dir,
        checkpoint_identity=_identity(),
        budget=_budget(1),
    )

    assert result["status"] == "no_eligible_checkpoint"
    assert result["training_outcome"]["eligible_points"] == 0
    assert result["budget"]["validated_point"] == learning.training.budget.VALIDATION_POINTS
    assert result["best_metric"] is None
    assert all(entry["value"] is None for entry in result["objective_history"])
    assert not (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "last_checkpoint.pt").exists()


@pytest.mark.parametrize("eligible", [True, False], ids=("best-eligible", "none-eligible"))
def test_divergence_ends_the_run_at_its_budget_fraction_with_the_best_eligible_checkpoint(tmp_path: Path, eligible: bool) -> None:
    """
    A NaN training loss in the third of four epochs ends the run as diverged at 50 % of its budget.

    Its result is the best eligible checkpoint of the two validated epochs; without an eligible
    point it has none and is reported unsuccessful by its missing best.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, _loss, train_loader, eval_loader = _components(17)
    result = learning.training.loop.train_loop(
        config=_config(4),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=_DivergingLoss(diverge_at_epoch=2),
        eval_metrics={"mse": _DatasetMSE() if eligible else _NonFiniteMetric()},
        scheduler=scheduler,
        save_dir=run_dir,
        checkpoint_identity=_identity(),
        budget=_budget(4),
    )

    outcome = result["training_outcome"]
    assert result["status"] == "diverged"
    assert outcome["budget_progress"] == pytest.approx(0.5)
    assert outcome["eligible_points"] == (2 if eligible else 0)
    assert "non-finite before backward" in outcome["divergence"]
    assert result["global_step"] == 2
    assert (result["best_epoch"] in {1, 2}) is eligible
    assert (run_dir / "best_checkpoint.pt").exists() is eligible
    assert (run_dir / "last_checkpoint.pt").exists()

    # A crash after the terminal last checkpoint and before the summary: the resume finalizes the
    # divergence its last checkpoint records, with a loss that would now train, and runs no update.
    model, optimizer, scheduler, _loss, train_loader, eval_loader = _components(17)
    resumed = learning.training.loop.train_loop(
        config=_config(4),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=_DivergingLoss(diverge_at_epoch=100),
        eval_metrics={"mse": _DatasetMSE() if eligible else _NonFiniteMetric()},
        scheduler=scheduler,
        save_dir=run_dir,
        resume_from=run_dir / "last_checkpoint.pt",
        checkpoint_identity=_identity(),
        budget=_budget(4),
    )
    assert resumed["status"] == "diverged"
    assert resumed["training_outcome"] == outcome
    assert resumed["global_step"] == result["global_step"]
    assert resumed["best_epoch"] == result["best_epoch"]


@pytest.mark.parametrize(
    ("direction", "values", "expected_epoch", "expected_metric"),
    [
        ("minimize", [1.0, 1.0], 1, 1.0),
        ("maximize", [1.0, 2.0], 2, 2.0),
    ],
)
def test_objective_direction_and_strict_ties_control_persisted_best(
    direction: str,
    values: list[float],
    expected_epoch: int,
    expected_metric: float,
    tmp_path: Path,
) -> None:
    """
    Vary objective direction and a two-evaluation value stream including a minimize tie.

    Selection must follow the declared direction, retain the earlier strict tie,
    and persist the same complete objective in checkpoint identity.
    """
    run_dir = tmp_path / direction
    run_dir.mkdir()
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(17)
    objective = _objective(direction=direction)
    identity = _identity(f"selection-{direction}", direction=direction)
    observed_learning_rates: list[float] = []

    result = learning.training.loop.train_loop(
        config=_config(2, direction=direction),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric(values)},
        save_dir=run_dir,
        event_callback=lambda _epoch, metrics: observed_learning_rates.append(metrics["optimization/learning_rate"]),
        checkpoint_identity=identity,
        budget=_budget(2),
    )

    assert observed_learning_rates == [0.05, 0.05]
    assert result["objective"] == objective
    assert result["best_epoch"] == expected_epoch
    assert result["best_metric"] == expected_metric
    payload = learning.training.checkpoint.load_checkpoint(
        run_dir / "last_checkpoint.pt",
        expected_identity=identity,
        expected_role="last",
        payload_kind="full",
        scheduler_expected=False,
        require_best=True,
    )
    assert payload["identity"]["objective"] == objective


def test_checkpoint_requires_its_schema_kind(tmp_path: Path) -> None:
    """Reject a payload that is not a training checkpoint."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=2, seed=42)
    payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    payload["schema_kind"] = "other_payload"

    with pytest.raises(ValueError, match="unsupported checkpoint schema_kind"):
        learning.training.checkpoint.validate_checkpoint(
            payload,
            expected_identity=_identity(),
            expected_role="last",
            payload_kind="full",
            scheduler_expected=True,
            require_best=True,
        )


def test_best_payload_kind_and_exact_key_set_fail_closed(tmp_path: Path) -> None:
    """Reject a best whose kind contradicts the expectation, and a missing or extra key."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=1, seed=23)
    evaluation_best = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model, optimizer, scheduler, loss, train_loader, _ = _components(23)
    _commit_epoch(train_loader)
    full_best = learning.training.checkpoint.make_checkpoint(
        role="best",
        payload_kind="full",
        identity=_identity(),
        completed_epoch=evaluation_best["completed_epoch"],
        global_step=evaluation_best["global_step"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        best_metric=evaluation_best["best_metric"],
        best_epoch=evaluation_best["best_epoch"],
        objective_history=evaluation_best["objective_history"],
        train_loader=train_loader,
        runtime_device=torch.device("cpu"),
        training_work=_NO_WORK,
        update_work=[],
        budget=learning.training.budget.StageBudget(**evaluation_best["budget_state_dict"]),
    )
    missing = {key: value for key, value in evaluation_best.items() if key != "loss_state_dict"}
    extra = {**evaluation_best, "optimizer_state_dict": optimizer.state_dict()}

    def validate(payload: dict[str, Any], kind: CheckpointPayloadKind) -> dict[str, Any]:
        return learning.training.checkpoint.validate_checkpoint(
            payload,
            expected_identity=_identity(),
            expected_role="best",
            payload_kind=kind,
            scheduler_expected=True,
            require_best=True,
        )

    assert validate(evaluation_best, "evaluation")["payload_kind"] == "evaluation"
    assert validate(full_best, "full")["payload_kind"] == "full"
    rejected: tuple[tuple[dict[str, Any], CheckpointPayloadKind, str], ...] = (
        (evaluation_best, "full", "payload_kind mismatch: expected 'full', got 'evaluation'"),
        (full_best, "evaluation", "payload_kind mismatch: expected 'evaluation', got 'full'"),
        (missing, "evaluation", r"Missing keys: \['loss_state_dict'\]"),
        (extra, "evaluation", r"Unknown keys: \['optimizer_state_dict'\]"),
    )
    for payload, kind, match in rejected:
        with pytest.raises(ValueError, match=match):
            validate(payload, kind)


def test_full_payloads_bind_the_unit_sampler_state_and_refuse_loader_generator_payloads() -> None:
    """
    Write a last checkpoint before and after its epoch's first committed group.

    A last checkpoint is always full; its sampler state must lie on a committed group of
    its completed epoch, a payload that carries a loader generator instead of the
    unit sampler state is refused, and its per-update rows must sum to its work counts.
    """
    model, optimizer, scheduler, loss, _train_loader, _ = _components(29)
    train_loader = DataLoader(
        _MappingDataset(), batch_sampler=UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=3, memory_batch=2, seed=30)
    )
    budget = _budget(4)
    budget.record_group(units=3, supervised_transitions=3, train_seconds=1.0)
    assert budget.record_validation() == 1
    common: dict[str, Any] = {
        "identity": _identity(),
        "completed_epoch": 1,
        "global_step": 4,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": loss,
        "best_metric": 0.5,
        "best_epoch": 1,
        "objective_history": [{"point": 1, "epoch": 1, "objective_id": "mse", "value": 0.5, "bound_violations": {"w_surf": {"case_a": 2.5}}}],
        "train_loader": train_loader,
        "runtime_device": torch.device("cpu"),
        "training_work": {"optimizer_steps": 1, "gradient_predictions": 3, "units": 3, "update_distinct_cases": 3},
        "update_work": [[3, 3, 3]],
        "budget": budget,
    }

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        return learning.training.checkpoint.validate_checkpoint(
            payload, expected_identity=_identity(), expected_role="last", payload_kind="full", scheduler_expected=True, require_best=True
        )

    with pytest.raises(ValueError, match="a last checkpoint is always full"):
        learning.training.checkpoint.make_checkpoint(role="last", payload_kind="evaluation", **common)
    with pytest.raises(ValueError, match="does not belong to a committed group"):
        validate(learning.training.checkpoint.make_checkpoint(role="last", payload_kind="full", **common))

    sampler = train_loader.batch_sampler
    assert isinstance(sampler, UnitSampler)
    sampler.begin_epoch(0)
    sampler.advance(3)
    partial = learning.training.checkpoint.make_checkpoint(role="last", payload_kind="full", **common)
    assert validate(partial)["next_epoch"] == 1
    legacy = {key: value for key, value in partial.items() if key != "train_sampler_state_dict"}
    legacy["train_loader_generator_state"] = torch.Generator().get_state()
    with pytest.raises(ValueError, match=r"Missing keys: \['train_sampler_state_dict'\]"):
        validate(legacy)
    with pytest.raises(ValueError, match="disagrees with its training_work"):
        validate({**partial, "update_work": [[3, 3, 3], [3, 3, 3]]})
    # A point's per-case bound evidence is required and holds finite hours only.
    entry = partial["objective_history"][0]
    missing = {key: value for key, value in entry.items() if key != "bound_violations"}
    for history_entry in (missing, {**entry, "bound_violations": {"w_surf": {"case_a": math.nan}}}):
        with pytest.raises(ValueError, match=r"exactly point|bound_violations must map"):
            validate({**partial, "objective_history": [history_entry]})


def test_completed_epoch_callback_preserves_evaluation_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Train a one-epoch budget in four groups of three units, so every group crosses five of the 20 points.

    Each group ends with one validation mid-epoch at the last point it crossed, after the
    safe ``last`` publication; the crossed points before it are recorded once as skipped,
    and point 20 is the budget end.
    """
    monkeypatch.setattr(learning.training.loop, "LAST_CHECKPOINT_INTERVAL_SECONDS", 0)
    run_dir = tmp_path / "callback-cadence"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, _train_loader, eval_loader = _components(28)
    train_loader = DataLoader(
        _MappingDataset(), batch_sampler=UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=3, memory_batch=2, seed=29)
    )
    observed: list[tuple[int, dict[str, float]]] = []

    def capture(step: int, values: dict[str, float]) -> None:
        payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        assert payload["global_step"] == step
        observed.append((step, dict(values)))

    result = learning.training.loop.train_loop(
        config=_config(1),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        event_callback=capture,
        checkpoint_identity=_identity("callback-cadence"),
        budget=_budget(1),
    )

    assert [step for step, _values in observed] == [1, 2, 3, 4]
    assert [values["validation/point"] for _step, values in observed] == [5.0, 10.0, 15.0, 20.0]
    assert [values["epoch"] for _step, values in observed] == [1.0] * 4
    assert all("id/mse" in values and "train/loss_total" in values for _step, values in observed)
    assert [(entry["point"], entry["epoch"]) for entry in result["objective_history"]] == [(5, 1), (10, 1), (15, 1), (20, 1)]
    assert result["budget"]["skipped_points"] == [point for point in range(1, 20) if point % 5 != 0]
    assert result["completed_epoch"] == 1


_HELD_OUT_ROLES = ("id_test", "material_ood", "parameter_ood")


def _held_out_keys(payload: Mapping[str, Any]) -> list[str]:
    """Return the keys of ``payload`` that name a held-out role."""
    return [key for key in payload if any(role in key for role in _HELD_OUT_ROLES)]


def test_validation_and_physics_follow_the_validation_points_and_no_held_out_role_is_evaluated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate and run the physics monitor on validation membership once per point; the loop takes no held-out loader."""
    assert "ood_loaders" not in inspect.signature(learning.training.loop.train_loop).parameters
    run_dir = tmp_path / "shared-points"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(101)
    config = _config(3)
    config["tracking"]["wandb"]["monitor"] = {"enabled": True, "max_cases": 2}
    id_epochs: list[int] = []
    evaluated_loaders: list[Any] = []
    physics_calls = 0
    active_epoch = 0
    observed: dict[int, dict[str, float]] = {}

    def record_eval(
        _model: nn.Module,
        loader: DataLoader[Any],
        _metrics: dict[str, Any],
        _device: torch.device,
        _scaling: Any,
        **_kwargs: Any,
    ) -> dict[str, float]:
        evaluated_loaders.append(loader)
        id_epochs.append(active_epoch)
        return {"mse": float(active_epoch), "nonfinite_cases": 0.0}

    def record_physics(_model: nn.Module, loader: DataLoader[Any], *_args: Any, **_kwargs: Any) -> dict[str, float]:
        nonlocal physics_calls
        evaluated_loaders.append(loader)
        physics_calls += 1
        return {
            "physics/id/momentum_residual_mse": float(physics_calls),
            "physics/id/flux_balance_mse": 2.0,
            "physics/id/pressure_boundary_mse": 4.0,
        }

    def capture(epoch: int, values: dict[str, float]) -> None:
        nonlocal active_epoch
        active_epoch = epoch + 1
        observed[epoch] = dict(values)

    active_epoch = 1
    monkeypatch.setattr(learning.training.loop, "eval_one_epoch", record_eval)
    monkeypatch.setattr(learning.training.loop, "evaluate_physics_monitor", record_physics)
    learning.training.loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        event_callback=capture,
        checkpoint_identity=_identity("shared-points"),
        budget=_budget(3),
    )

    assert id_epochs == [1, 2, 3]
    assert physics_calls == 3
    # Validation and the physics monitor read only the validation membership.
    assert all(loader is eval_loader for loader in evaluated_loaders)
    for key in ("id/mse", "physics/id/momentum_residual_mse"):
        assert [epoch for epoch, payload in observed.items() if key in payload] == [1, 2, 3]
    assert not [key for payload in observed.values() for key in _held_out_keys(payload)]
    assert [observed[epoch]["physics/id/momentum_residual_mse"] for epoch in (1, 2, 3)] == [1.0, 2.0, 3.0]


def test_selected_checkpoint_evaluation_uses_one_best_state_for_all_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload best_checkpoint.pt before validation, physics and selected weights; no held-out role is evaluated."""
    assert "ood_loaders" not in inspect.signature(learning.training.loop.evaluate_selected_checkpoint).parameters
    run_dir = tmp_path / "selected-best"
    run_dir.mkdir()
    model, optimizer, scheduler, _loss, train_loader, eval_loader = _components(43)
    train_loss = _WarmupStatefulLoss()
    result = learning.training.loop.train_loop(
        config=_config(2),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=train_loss,
        eval_metrics={"mse": _ObjectiveStreamMetric([0.1, 0.9])},
        scheduler=scheduler,
        save_dir=run_dir,
        checkpoint_identity=_identity("selected-best"),
        budget=_budget(2),
    )
    assert result["best_epoch"] == 1
    best = learning.training.checkpoint.load_checkpoint(
        run_dir / "best_checkpoint.pt",
        expected_identity=_identity("selected-best"),
        expected_role="best",
        payload_kind="evaluation",
        scheduler_expected=True,
        require_best=True,
    )
    best_state = best["model_state_dict"]
    task = domain.tasks.registry.get_task("steady_flow")
    science_metric_ids = tuple(metric.id for metric in task.default_metrics)
    id_science_metrics = {metric_id: 0.11 + 0.01 * index for index, metric_id in enumerate(science_metric_ids)}

    def assert_best_state(current: nn.Module) -> None:
        for name, value in current.state_dict().items():
            assert torch.equal(value.cpu(), best_state[name])

    def selected_eval(
        current: nn.Module,
        loader: DataLoader[Any],
        _metrics: dict[str, Any],
        _device: torch.device,
        _scaling: Any,
    ) -> dict[str, float]:
        assert_best_state(current)
        assert loader is eval_loader
        return {"mse": 0.2, "nonfinite_cases": 0.0, **id_science_metrics}

    def selected_physics(
        current: nn.Module,
        loader: DataLoader[Any],
        _loss_fn: nn.Module,
        _device: torch.device,
        _scaling: Any,
        *,
        max_cases: int,
    ) -> dict[str, float]:
        assert_best_state(current)
        assert loader is eval_loader
        assert max_cases == 2
        return {
            "physics/id/momentum_residual_mse": 1.0,
            "physics/id/flux_balance_mse": 2.0,
            "physics/id/pressure_boundary_mse": 4.0,
        }

    monkeypatch.setattr(learning.training.loop, "eval_one_epoch", selected_eval)
    monkeypatch.setattr(learning.training.loop, "evaluate_physics_monitor", selected_physics)
    selected_config = _config(2)
    selected_config["loss"] = {
        "physics": {
            "enabled": True,
            "momentum_weight": {"target": 1.0, "warmup": {"epochs": 10}},
            "continuity_weight": {"target": 3.0, "warmup": {"epochs": 10}},
            "boundary_weight": {"target": 2.0, "warmup": {"epochs": 10}},
        }
    }
    selected = learning.training.loop.evaluate_selected_checkpoint(
        config=selected_config,
        model=model,
        train_loss=train_loss,
        eval_loader=eval_loader,
        eval_metrics={
            "mse": _DatasetMSE(),
            **{metric_id: _DatasetMSE() for metric_id in science_metric_ids},
        },
        device=torch.device("cpu"),
        scaling=None,
        checkpoint_identity=_identity("selected-best"),
        best_checkpoint_path=run_dir / "best_checkpoint.pt",
        scheduler_expected=True,
        max_physics_cases=2,
    )
    assert selected["selected_epoch"] == 1
    metrics = selected["selected_metrics"]
    assert metrics["selected/id/mse"] == 0.2
    assert {metric_id: metrics[f"selected/id/{metric_id}"] for metric_id in science_metric_ids} == id_science_metrics
    assert not _held_out_keys(metrics)
    assert metrics["selected/physics/flux_balance_mse"] == 2.0
    for term in ("momentum", "continuity", "boundary"):
        assert metrics[f"selected/training/{term}_weight"] == 0.0

    settled_config = copy.deepcopy(selected_config)
    for term in ("momentum", "continuity", "boundary"):
        settled_config["loss"]["physics"][f"{term}_weight"]["warmup"]["epochs"] = 0
    settled = learning.training.loop.evaluate_selected_checkpoint(
        config=settled_config,
        model=model,
        train_loss=train_loss,
        eval_loader=eval_loader,
        eval_metrics={"mse": _DatasetMSE()},
        device=torch.device("cpu"),
        scaling=None,
        checkpoint_identity=_identity("selected-best"),
        best_checkpoint_path=run_dir / "best_checkpoint.pt",
        scheduler_expected=True,
        max_physics_cases=2,
    )
    settled_metrics = settled["selected_metrics"]
    assert not any(key.startswith("selected/training/") for key in settled_metrics)


def test_training_timer_excludes_post_training_epoch_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude evaluation, monitoring, checkpoints, and W&B from train duration."""
    run_dir = tmp_path / "timing-boundary"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, _train_loader, eval_loader = _components(91)
    # One update of all twelve cases in one microbatch: the update path reads the clock at the group start and around its one load.
    train_loader = DataLoader(
        _MappingDataset(), batch_sampler=UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=12, memory_batch=12, seed=0)
    )
    events: list[str] = []
    ticks = itertools.count()
    observed: dict[str, float] = {}
    real_eval = learning.training.loop.eval_one_epoch
    real_save = learning.training.checkpoint.save_checkpoints

    def clock() -> float:
        events.append("clock")
        return float(next(ticks))

    def record_eval(*args: Any, **kwargs: Any) -> dict[str, float]:
        events.append("evaluation")
        return real_eval(*args, **kwargs)

    def record_checkpoint(*publications: tuple[dict[str, Any], Path]) -> None:
        events.append("checkpoint")
        real_save(*publications)

    def record_event(_step: int, metrics: dict[str, float]) -> None:
        events.extend(("physics_monitor", "wandb_logging"))
        observed.update(metrics)

    monkeypatch.setattr(learning.training.loop.time, "perf_counter", clock)
    monkeypatch.setattr(learning.training.loop, "eval_one_epoch", record_eval)
    monkeypatch.setattr(learning.training.checkpoint, "save_checkpoints", record_checkpoint)

    learning.training.loop.train_loop(
        config=_config(1),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        event_callback=record_event,
        checkpoint_identity=_identity("timing-boundary"),
        budget=_budget(1),
    )

    # Each clock reading returns its own index, so a duration counts the readings it spans.
    readings_before_evaluation = events[: events.index("evaluation")].count("clock")
    readings_before_observers = events[: events.index("physics_monitor")].count("clock")
    train = observed["system/train_duration_seconds"]
    assert 0.0 < train <= readings_before_evaluation - 1
    assert train < observed["system/event_duration_seconds"] <= readings_before_observers
    # The new best is written with its last after the event duration and before the observers.
    assert events.index("evaluation") < events.index("checkpoint") < events.index("physics_monitor")
    assert observed["system/train_samples_per_second"] == 12.0 / train


def test_missing_best_prevents_successful_loop_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Stub best-checkpoint publication away while allowing the last checkpoint to save.

    Final reload validation must fail rather than report success, keeping a run
    without its advertised selection artifact from becoming completed.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(14)
    real_save = learning.training.checkpoint.save_checkpoints

    def drop_best(*publications: tuple[dict[str, Any], Path]) -> None:
        real_save(*(publication for publication in publications if publication[0]["checkpoint_role"] != "best"))

    monkeypatch.setattr(learning.training.checkpoint, "save_checkpoints", drop_best)
    with pytest.raises(FileNotFoundError, match="Required best checkpoint"):
        learning.training.loop.train_loop(
            config={
                "task": "steady_flow",
                "run": {"device": "cpu"},
                "training": {"budget_epochs": 1},
                "evaluation": {"objective": _objective()},
                "tracking": {"wandb": {"monitor": {"enabled": False, "max_cases": 1}}},
            },
            device=torch.device("cpu"),
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_loss=loss,
            eval_metrics={"mse": _DatasetMSE()},
            scheduler=scheduler,
            save_dir=run_dir,
            checkpoint_identity=_identity("missing-best"),
            budget=_budget(1),
        )

    assert not (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "last_checkpoint.pt").is_file()


def _subset_run(
    run_dir: Path,
    positions: list[int],
    *,
    resume_from: Path | None = None,
    stop_after: int | None = None,
    loss: _StatefulLoss | None = None,
) -> dict[str, Any]:
    """Train 12 Train cases of an 18-case full set for two full-set passes in groups of six, recording the warm-up position per event."""
    run_dir.mkdir(exist_ok=True)
    model, optimizer, scheduler, _default_loss, _train_loader, eval_loader = _components(23)
    active_loss = _StatefulLoss() if loss is None else loss
    sampler = UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=6, memory_batch=3, seed=24, full_set_units=18)

    def record(_epoch: int, _values: dict[str, float]) -> None:
        positions.append(int(active_loss.epoch_index.item()))
        if len(positions) == stop_after:
            raise _StopRunError

    return learning.training.loop.train_loop(
        config=_config(2),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(_MappingDataset(), batch_sampler=sampler),
        eval_loader=eval_loader,
        train_loss=active_loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        resume_from=resume_from,
        event_callback=record,
        checkpoint_identity=_identity("subset-warmup"),
        budget=learning.training.budget.StageBudget(unit="units", total=learning.training.budget.passes_budget(2, 18)),
    )


def test_warmup_counts_full_set_passes_per_group_across_a_mid_epoch_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the loss warm-up per group from committed full-set passes; a resume inside a subset epoch applies the same positions."""
    uninterrupted: list[int] = []
    _subset_run(tmp_path / "full", uninterrupted)
    # Six groups of six units start at 0, 6, ..., 30 committed units: 0 or 1 pass of the 18-case full set. The second
    # pass begins inside the second subset epoch, which a sampler-epoch count would miss.
    assert uninterrupted == [0, 0, 0, 1, 1, 1]

    monkeypatch.setattr(learning.training.loop, "LAST_CHECKPOINT_INTERVAL_SECONDS", 0.0)
    resumed: list[int] = []
    with pytest.raises(_StopRunError):
        _subset_run(tmp_path / "resumed", resumed, stop_after=3)
    _subset_run(tmp_path / "resumed", resumed, resume_from=tmp_path / "resumed" / "last_checkpoint.pt")

    assert resumed == uninterrupted


def test_divergence_at_a_warmup_boundary_keeps_the_committed_position(tmp_path: Path) -> None:
    """Keep the warm-up position of the last committed group when the group that starts the second full-set pass diverges."""
    positions: list[int] = []
    # The fourth group starts at 18 committed units, position 1, and diverges there.
    result = _subset_run(tmp_path, positions, loss=_DivergingLoss(diverge_at_epoch=1))

    assert result["training_outcome"]["status"] == "diverged"
    last = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert int(last["loss_state_dict"]["epoch_index"]) == positions[-1] == 0


def test_validation_bound_evidence_survives_checkpoint_resume_and_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Carry each point's per-case bound evidence from validation through the last checkpoint, a resume and the terminal result."""
    real_eval = learning.training.loop.eval_one_epoch
    evaluations = itertools.count(1)

    def eval_with_bounds(*args: Any, bound_cases: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
        values = real_eval(*args)
        # The ID evaluation of each point reports one violating case of its own.
        if bound_cases is not None:
            bound_cases["T"] = {f"case-{next(evaluations)}": 1.5}
        return values

    def stop_after_second_point(step: int, _values: dict[str, float]) -> None:
        if step == 2:
            raise _StopRunError

    monkeypatch.setattr(learning.training.loop, "eval_one_epoch", eval_with_bounds)
    monkeypatch.setattr(learning.training.loop, "LAST_CHECKPOINT_INTERVAL_SECONDS", 0.0)
    with pytest.raises(_StopRunError):
        _run(tmp_path, epochs=4, seed=151, event_callback=stop_after_second_point)
    interrupted = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert [entry["bound_violations"] for entry in interrupted["objective_history"]] == [{"T": {"case-1": 1.5}}, {"T": {"case-2": 1.5}}]

    result, *_ = _run(tmp_path, epochs=4, seed=999, resume_from=tmp_path / "last_checkpoint.pt")

    assert [entry["bound_violations"] for entry in result["objective_history"]] == [{"T": {f"case-{point}": 1.5}} for point in range(1, 5)]
