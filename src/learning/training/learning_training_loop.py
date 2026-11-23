"""Run the custom training and evaluation loop with checkpoint support.

Responsibilities:
  - Train each stage until its budget is complete and evaluate validation and the bounded physics monitor on validation
    membership right after the group that reaches one of the budget's validation points, also inside an epoch
  - Keep the optimizer at the schedule's rate of the committed budget fraction: at stage start, after a
    restore and after every committed group
  - Consume the resolved semantic objective for plateau-schedule and best-metric updates
  - Manage checkpoints, histories, and reproducibility state
  - Run one update path for both tasks: per update group, weight each unit's loss by its case weight,
    accumulate the memory microbatches, average over the group's units, then step once
  - Measure every group on the one optimisation clock and commit its units, supervised transitions and seconds to the
    unit sampler and the stage budget
  - Count each run's committed optimizer groups, units, per-update distinct cases and gradient-carrying predictions
  - Hand the caller an attempt's committed work at its start and in its newest durable state, and its precision
  - Describe the stage's units: kind, Train and full-set counts, full-set transitions, cases seen, k and the last-group rule
  - Invoke optional event callbacks for local lifecycle observers

Design principles:
  - Reproducibility state is explicit in last checkpoints; a best keeps it only as the matched-A0 handoff source
  - Data, model and loss objects are caller-provided
  - The task part of an update is only how a unit's loss is computed
  - Controller behavior stays independent of model architecture
  - Training never sees a held-out role: ID-test and OOD are evaluated only by the separate artifact step

This module does NOT:
  - Load or resolve configs. ``experiments.config.loader`` owns semantic admission
  - Construct models or losses. Caller-selected learning factories own construction
  - Parse CLI arguments. ``experiments.cli`` owns command boundaries
"""

from __future__ import annotations

import copy
import inspect
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import nn

from src import common, datasets, experiments
from src.learning.metrics import learning_metrics as metric_impl

from . import learning_training_checkpoint as checkpoints
from .learning_training_adapter import OptimizerWork, UnitLosses
from .learning_training_budget import StageBudget

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

    from .learning_training_adapter import UnitLossTask
    from .learning_training_optim import LearningRateSchedule

TensorBatch = dict[str, torch.Tensor]
_TRANSIENT_VIEW_RANK = 5
LAST_CHECKPOINT_INTERVAL_SECONDS = 30 * 60  # bounds training lost to a crash; a multi-GB full-state write per event costs more
EventCallback = Callable[[int, dict[str, float]], None]
# The epoch in progress, the training values of an event that ends an epoch, and the complete values of a validating event.
EpochStateCallback = Callable[[int, dict[str, float] | None, dict[str, float] | None], None]
# Every evaluation reports its cases with a non-finite prediction; one of them leaves the evaluation without metric values.
NONFINITE_CASES_KEY = "nonfinite_cases"
# An epoch's last group holds the units left over, and its loss is their mean (see train_update_groups).
LAST_GROUP_RULE: Final = "epoch_remainder_mean_over_its_units"


@dataclass(slots=True)
class AttemptWork:
    """
    Carry one ``train_loop`` invocation's committed work to its caller, also when it ends with an exception.

    Each record holds the ``training_work`` counts and the budget's committed optimisation
    ``train_seconds``. ``at_start`` is the work the invocation restored (zero for a fresh run or
    stage); ``durable`` is the work a continuation restores: that of the newest ``last``
    checkpoint, or, for an Optuna trial, which never resumes, that of its latest event. Both
    stay ``None`` until training starts. Groups after ``durable`` are computed but not
    committed: a resume trains them again. ``interrupted_group_seconds`` holds the optimisation
    seconds of a group an exception or a divergence ended, from its clock start to that end;
    the budget never commits them.
    """

    at_start: dict[str, Any] | None = None
    durable: dict[str, Any] | None = None
    interrupted_group_seconds: float = 0.0


def attempt_precision(model: nn.Module) -> dict[str, Any]:
    """Return the numerics an attempt trains with: parameter dtypes, autocast (never used: ``None``), the TF32 flags and cuDNN autotuning."""
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()})
    return {
        "parameter_dtypes": dtypes,
        "autocast_dtype": None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_autotune": bool(torch.backends.cudnn.benchmark),
    }


def validation_eligible(metrics: Mapping[str, float], objective_id: str) -> bool:
    """
    Return whether one validation makes its checkpoint eligible, the same rule for every task and stage.

    Eligible means complete and finite: every case's autonomous trajectory is finite to its end
    and the objective is finite. The teacher-forced one-step view and physical validity are
    diagnostics, never criteria, and no value over a valid prefix or over only the finite
    cases is ever selected.
    """
    return metrics[NONFINITE_CASES_KEY] == 0 and math.isfinite(float(metrics[objective_id]))


def train_unit_sampler(train_loader: Any) -> datasets.runtime.units.UnitSampler:
    """Return the resumable unit sampler that drives a training loader's update groups."""
    sampler = getattr(train_loader, "batch_sampler", None)
    if not isinstance(sampler, datasets.runtime.units.UnitSampler):
        msg = "Training requires a DataLoader driven by a UnitSampler, the one sampler of every task and stage."
        raise TypeError(msg)
    return sampler


def unit_evidence(sampler: datasets.runtime.units.UnitSampler) -> dict[str, Any]:
    """Return a stage's unit kind, Train and full-set units, full-set transitions, Train and seen cases, k and the last-group rule."""
    return {
        "unit_kind": sampler.unit_kind,
        "train_units": sampler.size,
        "full_set_units": sampler.full_set_units,
        "full_set_transitions": sampler.full_set_transitions,
        "train_cases": sampler.case_count,
        "seen_cases": sampler.seen_cases,
        "units_per_update": sampler.units_per_update,
        "last_group_rule": LAST_GROUP_RULE,
    }


class _SteadyUnits:
    """Compute the steady task's unit losses: one case per unit, standardized by the Train scaling."""

    def __init__(self, scaling: Any | None, device: torch.device) -> None:
        """Bind the Train scaling and the device that receive every microbatch."""
        self.scaling = scaling
        self.device = device

    def begin_group(self, group: datasets.runtime.units.UnitGroup, *, progress: float) -> None:
        """Steady units draw nothing per group."""
        del group, progress

    def unit_losses(self, model: nn.Module, raw_batch: Any, loss: nn.Module, *, positions: tuple[int, ...]) -> Iterator[UnitLosses]:
        """Yield every case's loss of one microbatch from one forward pass."""
        batch = _prepare_batch(raw_batch, self.device, self.scaling, training=True)
        units = _training_batch_size(batch)
        if units != len(positions):
            msg = "A steady microbatch must hold exactly the cases its update group assigned to it."
            raise ValueError(msg)
        components = loss.unit_components(model(batch["x"]), x=batch["x"], y=batch["y"], reference=batch.get("reference"))
        yield UnitLosses(components=dict(components), supervised_transitions=units, forward_transitions=units)


class PhysicsMonitorEvaluationError(RuntimeError):
    """Identify a failure owned by bounded scientific physics evaluation."""


def _synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA only where wall-clock phase timing requires it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loader_case_count(loader: Any) -> int:
    """Return the concrete loader membership size when available."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0
    try:
        return len(dataset)
    except TypeError:
        return 0


@contextmanager
def _training_phase(name: str, completed_epoch: int) -> Iterator[None]:
    """Annotate failures with the active completed-epoch lifecycle phase."""
    try:
        yield
    except BaseException as error:
        with suppress(Exception):
            error.training_phase = name  # type: ignore[attr-defined]
            error.completed_epoch = completed_epoch  # type: ignore[attr-defined]
        raise


def _move_batch_to_device(batch: Any, device: torch.device) -> TensorBatch:
    """Move an existing dataset batch to the target device."""
    if not isinstance(batch, dict):
        msg = f"Expected dataloader batch to be a dict, got: {type(batch).__name__}"
        raise TypeError(msg)

    tensor_batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    if "x" not in tensor_batch or "y" not in tensor_batch:
        msg = "Training batches must contain tensor keys 'x' and 'y'."
        raise KeyError(msg)
    return tensor_batch


def _case_membership(raw_batch: Any, *, key: str, metrics: Iterable[Any]) -> metric_impl.CaseMembership | None:
    """Return the case membership of one loader batch when a metric aggregates over cases."""
    if not any(getattr(metric, "requires_cases", False) for metric in metrics):
        return None
    metadata = raw_batch.get(key) if isinstance(raw_batch, Mapping) else None
    if not isinstance(metadata, Mapping):
        msg = f"Case-aggregated evaluation metrics require the batch's {key!r} metadata mapping."
        raise TypeError(msg)
    return metric_impl.CaseMembership.from_metadata(metadata)


def _prepare_batch(
    raw_batch: Any,
    device: torch.device,
    scaling: Any | None,
    *,
    training: bool,
) -> TensorBatch:
    """
    Move one raw batch to the device and standardize it with the Train scaling when one is given.

    Model inputs are always standardized; targets are standardized for training
    and stay physical for evaluation. The final batch contains tensor ``x`` and ``y`` keys on the device and,
    with a scaling, the per-case ``reference`` P derived once from the physical inputs,
    which every later encode or decode of the batch takes.
    """
    batch = _move_batch_to_device(raw_batch, device)
    if scaling is None:
        return batch
    reference = scaling.state_reference(batch["x"])
    batch["reference"] = reference
    batch["x"] = scaling.encode_inputs(batch["x"], reference=reference)
    if training:
        batch["y"] = scaling.encode_state(batch["y"], reference=reference)
    return batch


def _training_batch_size(batch: TensorBatch) -> int:
    """Return the unambiguous positive sample dimension of one prepared batch."""
    x = batch["x"]
    y = batch["y"]
    if x.ndim == 0 or y.ndim == 0:
        msg = "Training batch tensors must expose a leading sample dimension."
        raise ValueError(msg)
    x_samples = int(x.shape[0])
    y_samples = int(y.shape[0])
    if x_samples != y_samples:
        msg = f"Training batch sample dimensions disagree: x={x_samples}, y={y_samples}."
        raise ValueError(msg)
    if y_samples <= 0:
        msg = "Training batches must contain at least one sample."
        raise ValueError(msg)
    return y_samples


def _require_unit_losses(share: Any, *, units: int) -> dict[str, torch.Tensor]:
    """Validate one task's per-unit loss shares before backward work begins."""
    if not isinstance(share, UnitLosses):
        msg = "A task must yield UnitLosses for every microbatch."
        raise TypeError(msg)
    components = share.components
    if not isinstance(components, Mapping) or "total" not in components:
        msg = "Unit loss components must contain total."
        raise TypeError(msg)
    for name, value in components.items():
        if not isinstance(value, torch.Tensor) or value.shape != (units,):
            msg = f"Unit loss component {name!r} must hold one value per microbatch unit ({units})."
            raise ValueError(msg)
        if not bool(torch.isfinite(value.detach()).all().item()):
            msg = f"Training loss component {name!r} is non-finite before backward: {value.detach()}."
            raise FloatingPointError(msg)
    counts = (share.supervised_transitions, share.forward_transitions)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        msg = "Unit loss work counts must be positive integers."
        raise ValueError(msg)
    return dict(components)


def _next_microbatch(iterator: Iterator[Any]) -> Any:
    """Return the loader's next microbatch of the current update group, which must not end the loader."""
    raw_batch = next(iterator, None)
    if raw_batch is None:
        msg = "Training loader ended inside an update group."
        raise RuntimeError(msg)
    return raw_batch


def _backward_microbatch(
    model: nn.Module,
    raw_batch: Any,
    loss_fn: nn.Module,
    task: UnitLossTask,
    group: datasets.runtime.units.UnitGroup,
    *,
    positions: tuple[int, ...],
    components: dict[str, list[torch.Tensor]],
) -> tuple[int, int]:
    """
    Accumulate one memory microbatch's weighted unit-loss gradients and return its supervised and forward transitions.

    Each unit's loss share is multiplied by its case weight and divided by the group's unit count
    before backward; the detached weighted components are appended to ``components``.
    """
    supervised = forward = 0
    for share in task.unit_losses(model, raw_batch, loss_fn, positions=positions):
        unit_components = _require_unit_losses(share, units=len(positions))
        total = unit_components["total"]
        weights = torch.tensor([group.weights[position] for position in positions], dtype=total.dtype, device=total.device)
        weighted = {name: (weights * value).sum() for name, value in unit_components.items()}
        (weighted["total"] / len(group.items)).backward()
        for name, value in weighted.items():
            components.setdefault(name, []).append(value.detach())
        supervised += share.supervised_transitions
        forward += share.forward_transitions
    return supervised, forward


def _interrupted_group_seconds(device: torch.device, *, group_started: float, loader_wait: float, waiting: float | None) -> float:
    """Return the optimisation seconds an uncommitted group computed up to its end, loader waits excluded."""
    # Queued device work of this group still counts. A failing synchronize returns only once the device
    # stopped executing the queue, so the time read after it covers all work the group computed.
    with suppress(Exception):
        _synchronize_device(device)
    now = time.perf_counter()
    pending_wait = 0.0 if waiting is None else now - waiting
    return max(0.0, now - group_started - loader_wait - pending_wait)


def _set_group_warmup(loss_fn: nn.Module, budget: StageBudget, full_set_units: int) -> dict[str, Any] | None:
    """Set a loss warm-up position for the next group; return the loss state before it, or ``None`` without a warm-up."""
    set_epoch = getattr(loss_fn, "set_epoch", None)
    if not callable(set_epoch):
        return None
    committed = copy.deepcopy(loss_fn.state_dict())
    # The physics warm-up counts committed passes over the full Train set, like the budget, and is set per group:
    # a Train subset ramps over the same budget fraction, and a resume at any group applies the same weights.
    set_epoch(budget.units // full_set_units)
    return committed


def train_update_groups(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    task: UnitLossTask,
    *,
    budget: StageBudget,
    scheduler: LearningRateSchedule | None = None,
    adapter: Any | None = None,
    attempt: AttemptWork | None = None,
) -> dict[str, Any]:
    """
    Train the current epoch's uncommitted update groups up to the next stage event.

    Every group of the train loader's unit sampler is one optimizer update: the group's
    units, their case weights and the task's random draws are fixed, then its memory
    microbatches are loaded, and
    each unit's loss share is multiplied by its weight and divided by the group's unit count
    (``units_per_update``, or the actual count of an epoch's last group) before backward,
    so the microbatch split changes only rounding. After the step the sampler commits the
    group, the budget records its units, supervised transitions and optimisation seconds, the schedule sets the rate
    of the new committed budget fraction and the adapter, if any, records its work.
    Training stops after the group that reaches a validation point or completes the
    budget, or at the epoch end; a sampler stop (``UnitSampler.end_after``) cuts the group that reaches it.

    Parameters
    ----------
    model : torch.nn.Module
        Model mutated through training-mode forward/backward updates.
    train_loader : DataLoader
        Loader driven by a :class:`~src.datasets.runtime.dataset_runtime_units.UnitSampler`
        whose current epoch has begun.
    optimizer : Optimizer
        Optimizer stepped once per update group.
    loss_fn : torch.nn.Module
        Semantic composition exposing ``unit_components``.
    device : torch.device
        Concrete device of the model.
    task : UnitLossTask
        Task part of the update: the transient adapter, or the steady case losses.
    budget : StageBudget
        The stage budget; it must neither be complete nor have a due validation point.
    scheduler : LearningRateSchedule | None, optional
        Stage schedule advanced to the committed budget fraction after every committed group;
        the caller sets it for the progress at entry.
    adapter : Any | None, optional
        Transient compute accounting that receives every committed group's work.
    attempt : AttemptWork | None, optional
        Attempt record that receives the optimisation seconds of a group an exception or a divergence ends.

    Returns
    -------
    dict[str, Any]
        Telemetry of the trained groups: unit-weighted mean loss components, per-update unit,
        distinct-case and supervised-transition means, the last group's unit count, and the
        internal ``optimizer_steps``, ``gradient_predictions``, ``units`` and
        ``update_distinct_cases`` counts and ``update_work`` rows (units, distinct cases and
        supervised transitions of every committed update, in order) the checkpoint lifecycle
        consumes, plus the internal ``divergence``: ``None``,
        or the message of the non-finite loss or training prediction that stopped the stretch
        before its group's step (the loss and per-update keys are then absent when no group
        was committed). A steady case is one gradient prediction, a transient supervised
        transition is one.

    Raises
    ------
    RuntimeError
        If no group is uncommitted, the budget has no room or a validation is due, the
        loader ends inside a group, or physics component telemetry is incomplete.

    Notes
    -----
    The optimisation clock of a group is synchronised wall time from before the curriculum
    draws to after the optimizer step, minus the loader waits; the device is synchronised
    before every wait, so queued device work is never counted as waiting. The schedule
    advance and the conversion of the loss telemetry to Python numbers follow the clock.
    Validation, checkpoints and logging happen outside this function and outside the clock.

    """
    sampler = train_unit_sampler(train_loader)
    groups = sampler.groups()
    if not groups:
        msg = "Training epoch has no uncommitted update group."
        raise RuntimeError(msg)
    if budget.complete or budget.validation_due:
        msg = "Training groups need an incomplete budget without a due validation point."
        raise RuntimeError(msg)
    model.train()
    sums: dict[str, float] = {}
    units = microbatches = optimizer_steps = supervised = forwarded = distinct_cases = 0
    last_group_units = 0
    update_work: list[list[int]] = []
    divergence: str | None = None
    _synchronize_device(device)
    started = time.perf_counter()
    optimizer.zero_grad()
    iterator = iter(train_loader)
    for group in groups:
        committed_loss_state = _set_group_warmup(loss_fn, budget, sampler.full_set_units)
        _synchronize_device(device)
        group_started = time.perf_counter()
        loader_wait = 0.0
        waiting: float | None = None
        # Detached device tensors: converting them to Python numbers waits for the device, so it happens after the clock.
        group_components: dict[str, list[torch.Tensor]] = {}
        group_supervised = group_forward = 0
        try:
            task.begin_group(group, progress=budget.progress)
            for positions in group.microbatches:
                _synchronize_device(device)
                waiting = time.perf_counter()
                raw_batch = _next_microbatch(iterator)
                loader_wait += time.perf_counter() - waiting
                waiting = None
                microbatch_supervised, microbatch_forward = _backward_microbatch(
                    model, raw_batch, loss_fn, task, group, positions=positions, components=group_components
                )
                group_supervised += microbatch_supervised
                group_forward += microbatch_forward
            optimizer.step()
            optimizer.zero_grad()
            _synchronize_device(device)
        except BaseException as error:
            # The attempt computed this group up to its end; the budget never commits it, a divergence included.
            if attempt is not None:
                attempt.interrupted_group_seconds += _interrupted_group_seconds(
                    device, group_started=group_started, loader_wait=loader_wait, waiting=waiting
                )
            if not isinstance(error, FloatingPointError):
                raise
            # Divergence: a non-finite loss or training prediction ends the stage before this group's step;
            # the faulty update is neither applied nor skipped over.
            optimizer.zero_grad()
            if committed_loss_state is not None:
                # The failed group never committed, so the loss keeps the warm-up position of the last committed group.
                loss_fn.load_state_dict(committed_loss_state)
            divergence = str(error)
            break
        microbatches += len(group.microbatches)
        group_wall = time.perf_counter() - group_started
        sampler.advance(len(group.items))
        budget.record_group(units=len(group.items), supervised_transitions=group_supervised, train_seconds=max(0.0, group_wall - loader_wait))
        # The optimizer always holds η(p) of the committed progress: the next group trains with it, and
        # a checkpoint or validation after this group records it.
        if scheduler is not None:
            scheduler.advance(budget.progress)
        for name, values in group_components.items():
            for value in torch.stack(values).tolist():
                sums[name] = sums.get(name, 0.0) + value
        optimizer_steps += 1
        units += len(group.items)
        supervised += group_supervised
        forwarded += group_forward
        distinct_cases += group.distinct_cases
        last_group_units = len(group.items)
        update_work.append([len(group.items), group.distinct_cases, group_supervised])
        if adapter is not None:
            adapter.record_optimizer_work(
                OptimizerWork(
                    microbatches=len(group.microbatches),
                    processed_target_transitions=group_supervised,
                    forward_transitions=group_forward,
                    peak_cuda_memory_bytes=(int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
                )
            )
        if budget.complete or budget.validation_due:
            break
    _synchronize_device(device)
    duration = time.perf_counter() - started
    if not math.isfinite(duration) or duration <= 0.0:
        msg = f"Training phase has invalid duration {duration}."
        raise RuntimeError(msg)
    result: dict[str, Any] = {
        "train/updates": float(optimizer_steps),
        "train/microbatches": float(microbatches),
        "train/forward_transitions": float(forwarded),
        "divergence": divergence,
        "system/train_duration_seconds": duration,
        "system/train_samples_per_second": units / duration,
        "optimizer_steps": float(optimizer_steps),
        "gradient_predictions": float(supervised),
        "units": float(units),
        "update_distinct_cases": float(distinct_cases),
        "update_work": update_work,
    }
    if not optimizer_steps:
        # Only a divergence in this stretch's first group leaves no committed group to describe.
        return result
    averaged = {name: value / units for name, value in sums.items()}
    result.update(
        {
            "train/loss_total": averaged["total"],
            "train/loss_data": averaged.get("data", averaged["total"]),
            "train/units_per_update": units / optimizer_steps,
            "train/distinct_cases_per_update": distinct_cases / optimizer_steps,
            "train/supervised_transitions_per_update": supervised / optimizer_steps,
            "train/last_group_units": float(last_group_units),
        }
    )
    result.update({f"train/loss_{name}": value for name, value in averaged.items()})
    if bool(getattr(loss_fn, "physics_enabled", False)):
        required = {"momentum", "continuity", "boundary"}
        missing = sorted(required.difference(averaged))
        if missing:
            msg = f"Physics-informed epoch aggregation is missing component(s): {missing}."
            raise RuntimeError(msg)
        result.update(
            {
                "physics/train/loss_momentum": averaged["momentum"],
                "physics/train/loss_continuity": averaged["continuity"],
                "physics/train/loss_boundary": averaged["boundary"],
            }
        )
        telemetry_state = getattr(loss_fn, "telemetry_state", None)
        if not callable(telemetry_state):
            msg = "Physics-informed loss does not expose applied weight telemetry."
            raise RuntimeError(msg)
        telemetry = cast("Mapping[str, float]", telemetry_state())
        for term in ("momentum", "continuity", "boundary"):
            result[f"physics/train/{term}_weight"] = float(telemetry[f"weight_{term}"])
    return result


def _finite_physics_monitor_scalar(name: str, value: torch.Tensor) -> float:
    """Return one finite detached diagnostic scalar or reject it scientifically."""
    scalar = float(value.detach().item())
    if not math.isfinite(scalar):
        msg = f"Physics monitor {name!r} is non-finite: {scalar}."
        raise FloatingPointError(msg)
    return scalar


def evaluate_physics_monitor(
    model: nn.Module,
    eval_loader: Iterable[Any],
    loss_fn: nn.Module,
    device: torch.device,
    scaling: Any,
    *,
    max_cases: int,
) -> dict[str, float]:
    """
    Evaluate bounded deterministic physics monitors on the saved eval prefix.

    The caller persists the exact prefix membership. This function consumes no
    more than ``max_cases`` in loader order, reuses the semantic loss's domain
    physics evaluator and the Train scaling, and produces no artifact files.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model evaluated temporarily in evaluation mode.
    eval_loader : Iterable[Any]
        Deterministic saved-evaluation membership in source order.
    loss_fn : torch.nn.Module
        Semantic loss exposing ``compute_physics_diagnostics``.
    device : torch.device
        Concrete device for bounded inference.
    scaling : Any
        Train scaling that standardizes the inputs and restores physical predictions.
    max_cases : int
        Positive upper bound on prefix samples, not batches.

    Returns
    -------
    dict[str, float]
        Sample-weighted dimensionless momentum, flux-balance, and boundary means.

    Raises
    ------
    TypeError
        If batches or the diagnostic interface violate their contracts.
    ValueError
        If ``max_cases`` is not a positive exact integer.
    PhysicsMonitorEvaluationError
        If model prediction, physics diagnostics, or finite-value validation fails.
        The originating exception is retained as the direct cause.
    RuntimeError
        If the selected membership produces no samples.

    Notes
    -----
    The model's incoming training/evaluation state is restored even on failure.

    """
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = f"max_cases must be a positive integer, got {max_cases!r}."
        raise ValueError(msg)
    compute = getattr(loss_fn, "compute_physics_diagnostics", None)
    if not callable(compute):
        msg = "Physics monitoring requires the semantic physics diagnostic adapter."
        raise TypeError(msg)

    totals = {
        "physics/id/momentum_residual_mse": 0.0,
        "physics/id/flux_balance_mse": 0.0,
        "physics/id/pressure_boundary_mse": 0.0,
    }
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in eval_loader:
                if not isinstance(raw_batch, Mapping):
                    msg = f"Expected monitor batch mapping, got {type(raw_batch).__name__}."
                    raise TypeError(msg)
                remaining = max_cases - sample_count
                if remaining <= 0:
                    break
                raw_y = raw_batch.get("y")
                raw_x = raw_batch.get("x")
                if not isinstance(raw_x, torch.Tensor) or not isinstance(raw_y, torch.Tensor):
                    msg = "Monitor batches must contain tensor keys 'x' and 'y'."
                    raise TypeError(msg)
                take = min(remaining, int(raw_y.shape[0]))
                batch = _prepare_batch(
                    {"x": raw_x[:take], "y": raw_y[:take]},
                    device,
                    scaling,
                    training=False,
                )
                try:
                    pred = model(batch["x"])
                    diagnostics = cast("Any", compute(pred, x=batch["x"], reference=batch.get("reference")))
                    values = {
                        "physics/id/momentum_residual_mse": diagnostics.momentum_mse,
                        "physics/id/flux_balance_mse": diagnostics.continuity_mse,
                        "physics/id/pressure_boundary_mse": diagnostics.boundary_mse,
                    }
                    for name, value in values.items():
                        totals[name] += _finite_physics_monitor_scalar(name, value) * take
                except Exception as error:
                    msg = f"Bounded physics-monitor scientific evaluation failed after model-input preparation: {type(error).__name__}: {error}"
                    raise PhysicsMonitorEvaluationError(msg) from error
                sample_count += take
    finally:
        model.train(was_training)

    if sample_count == 0:
        msg = "Physics monitor evaluation membership produced no samples."
        raise RuntimeError(msg)
    return {name: value / sample_count for name, value in totals.items()}


def eval_one_epoch(
    model: nn.Module,
    eval_loader: Iterable[Any],
    eval_metrics: dict[str, Any],
    device: torch.device,
    scaling: Any | None = None,
    adapter: Any | None = None,
    *,
    bound_cases: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """
    Execute evaluation on physical tensors with dataset accumulation.

    Evaluation preprocessing standardizes model inputs while preserving physical
    targets. This function derives physical predictions once by the inverse of
    the Train state scale. Metrics accumulate sufficient statistics and finalize
    only after the loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model evaluated without gradients.
    eval_loader : Iterable[Any]
        Loader providing task ``x`` and ``y`` batches.
    eval_metrics : dict[str, Any]
        Metric-ID keyed physical-space dataset accumulators.
    device : torch.device
        Concrete device receiving prepared batches.
    scaling : Any | None, optional
        Train scaling that decodes the physical predictions; without one (a synthetic task) the model predicts them.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with scaling where accepted.
    bound_cases : dict[str, dict[str, float]] | None, optional
        Mutated in place by the adapter evaluation: state channel to simulation
        case id to the hours of that case's first physical-bound violation, for
        violating cases only. Expected empty on entry; steady evaluation leaves it untouched.

    Returns
    -------
    dict[str, float]
        One finalized dataset value per configured metric ID.

    Raises
    ------
    ValueError
        If metric IDs, spaces (every metric accumulates physical tensors), or tensor views are inconsistent.

    Notes
    -----
    Dataset accumulators are reset at entry and finalized once after all batches.
    No batch-level metric means are averaged.

    """
    if adapter is not None:
        if scaling is not None:
            msg = "Adapter evaluation and a loop scaling are mutually exclusive."
            raise ValueError(msg)
        return _eval_adapter_one_epoch(model, eval_loader, eval_metrics, device, adapter, bound_cases=bound_cases)
    model.eval()
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()

    nonfinite_cases = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            cases = _case_membership(raw_batch, key="meta", metrics=eval_metrics.values())
            batch = _prepare_batch(raw_batch, device, scaling, training=False)
            pred_normalized = model(batch["x"])
            # A finite normalized prediction can still overflow on decoding; either view makes the case non-finite.
            pred_physical = pred_normalized if scaling is None else scaling.decode_state(pred_normalized, reference=batch["reference"])
            finite = pred_normalized.flatten(1).isfinite().all(1) & pred_physical.flatten(1).isfinite().all(1)
            nonfinite_cases += int((~finite).sum().item())
            if nonfinite_cases:
                # An incomplete validation reports no metric value, so the remaining batches are only counted.
                continue
            for metric in eval_metrics.values():
                metric.update(
                    pred_physical,
                    batch["y"],
                    space="physical",
                    batch_index=batch_index,
                    **({"cases": cases} if getattr(metric, "requires_cases", False) else {}),
                )

    if nonfinite_cases:
        return {NONFINITE_CASES_KEY: float(nonfinite_cases)}
    return {**metric_impl.finalize_metrics(eval_metrics), NONFINITE_CASES_KEY: 0.0}


def _grain_moisture_errors(views: Any) -> tuple[list[float], list[float], list[int]]:
    """Return per batch row the summed absolute and squared grain-moisture errors of the autonomous views and their valid element count."""
    fraction = views.f_surf.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, views.physical_prediction.shape[-2], views.physical_prediction.shape[-1])
    predicted_w = metric_impl.compute_grain_moisture(views.physical_prediction[:, :, 2], views.physical_prediction[:, :, 3], fraction)
    target_w = metric_impl.compute_grain_moisture(views.physical_target[:, :, 2], views.physical_target[:, :, 3], fraction)
    # float64 like the official metrics, so finite predictions cannot overflow the telemetry.
    difference = predicted_w.double() - target_w.double()
    valid = torch.ones_like(difference, dtype=torch.bool)
    if views.valid_mask is not None:
        try:
            valid = torch.broadcast_to(views.valid_mask, views.physical_prediction.shape).select(2, 0)
        except RuntimeError as error:
            msg = "Transient valid_mask must broadcast to physical prediction evidence."
            raise ValueError(msg) from error
        if not bool(valid.any().item()):
            msg = "Transient valid_mask has no valid physical moisture telemetry elements."
            raise ValueError(msg)
    selected = torch.where(valid, difference, torch.zeros_like(difference)).flatten(1)
    return selected.abs().sum(1).tolist(), selected.square().sum(1).tolist(), valid.flatten(1).sum(1).tolist()


def _add_case_moisture(moisture: dict[str, tuple[str, float, float, int]], views: Any, cases: metric_impl.CaseMembership | None) -> None:
    """Add one batch's grain-moisture error sums to ``moisture``: per case its material, absolute and squared sums and count."""
    if cases is None:
        msg = "Grain-moisture validation is reduced per case and requires the batch's case membership."
        raise ValueError(msg)
    absolute, squared, counts = _grain_moisture_errors(views)
    for row, (case_id, material) in enumerate(zip(cases.case_ids, cases.materials, strict=True)):
        _material, case_absolute, case_squared, case_count = moisture.get(case_id, (material, 0.0, 0.0, 0))
        moisture[case_id] = (material, case_absolute + absolute[row], case_squared + squared[row], case_count + counts[row])


def _case_moisture_metrics(moisture: Mapping[str, tuple[str, float, float, int]]) -> dict[str, float]:
    """Return the grain-moisture MAE and RMSE: each case's mean, then equal over cases and materials like the artifacts."""
    if not moisture:
        return {}
    mae = metric_impl.material_case_mean((material, absolute / count if count else None) for material, absolute, _, count in moisture.values())
    mse = metric_impl.material_case_mean((material, squared / count if count else None) for material, _, squared, count in moisture.values())
    if mae is None or mse is None:
        return {}
    return {"physical/w_gr_mae": mae, "physical/w_gr_rmse": math.sqrt(mse)}


def _collect_bound_cases(
    bound_cases: dict[str, dict[str, float]],
    channels: dict[str, None],
    violations: Mapping[str, tuple[float | None, ...]],
    raw_batch: Any,
    cases: metric_impl.CaseMembership | None,
) -> None:
    """Add one batch's per-row first-violation hours to ``bound_cases`` under the rows' case ids, which are read only for a violating row."""
    case_ids: tuple[str, ...] | None = None
    for channel, hours_by_row in violations.items():
        channels.setdefault(channel)
        for row, hours in enumerate(hours_by_row):
            if hours is None:
                continue
            if case_ids is None:
                case_ids = (cases or metric_impl.CaseMembership.from_metadata(raw_batch["metadata"])).case_ids
            per_case = bound_cases.setdefault(channel, {})
            per_case[case_ids[row]] = min(hours, per_case.get(case_ids[row], math.inf))


def _eval_adapter_one_epoch(
    model: nn.Module,
    eval_loader: Iterable[Any],
    eval_metrics: dict[str, Any],
    device: torch.device,
    adapter: Any,
    *,
    bound_cases: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Accumulate adapter-provided BLCHW views through resolved metric objects; ``bound_cases``, when given, receives the per-case bound evidence."""
    if bound_cases is None:
        bound_cases = {}
    model.eval()
    guardrails = {metric_id: copy.deepcopy(metric) for metric_id, metric in eval_metrics.items()}
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()
        guardrails[metric_id].reset()
    # Per case: material and the summed absolute and squared grain-moisture errors over its valid elements, and their count.
    moisture: dict[str, tuple[str, float, float, int]] = {}
    nonfinite_cases = 0
    one_step_finite = True
    # The channels the adapter reports keep their counts at zero; ``bound_cases`` holds the violating cases only.
    bound_channels: dict[str, None] = {}
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            cases = _case_membership(raw_batch, key="metadata", metrics=eval_metrics.values())
            views = adapter.evaluation_step(model, adapter.prepare_batch(raw_batch, device=device, training=False))
            _collect_bound_cases(bound_cases, bound_channels, views.bound_violations, raw_batch, cases)
            # The standardized view of a finite physical rollout can still overflow; either view makes the case non-finite.
            nonfinite_cases += int((~(views.case_complete & views.normalized_prediction.flatten(1).isfinite().all(1))).sum().item())
            # The one-step guardrail is a diagnostic: a non-finite one-step value only leaves it without a value.
            one_step_finite = one_step_finite and bool(views.one_step_physical_prediction.isfinite().all().item())
            if nonfinite_cases:
                # An incomplete validation reports no metric value, so the remaining batches are only counted.
                continue
            source_views = (
                views.normalized_prediction,
                views.normalized_target,
                views.physical_prediction,
                views.physical_target,
                views.one_step_normalized_prediction,
                views.one_step_physical_prediction,
            )
            reference = views.normalized_target
            if (
                not isinstance(reference, torch.Tensor)
                or reference.ndim != _TRANSIENT_VIEW_RANK
                or any(not isinstance(value, torch.Tensor) or value.shape != reference.shape for value in source_views)
            ):
                msg = "Adapter evaluation views must be BLCHW tensors of one shape."
                raise ValueError(msg)
            pred, target = views.physical_prediction.flatten(0, 1), views.physical_target.flatten(0, 1)
            for metric_id, metric in eval_metrics.items():
                kwargs: dict[str, Any] = {"space": "physical", "batch_index": batch_index}
                if cases is not None and getattr(metric, "requires_cases", False):
                    kwargs["cases"] = cases.repeat(reference.shape[1])
                mask = views.valid_mask
                if mask is not None and "mask" in inspect.signature(metric.update).parameters:
                    kwargs["mask"] = mask.flatten(0, 1)
                metric.update(pred, target, **kwargs)
                if one_step_finite:
                    guardrails[metric_id].update(views.one_step_physical_prediction.flatten(0, 1), target, **kwargs)
            if views.f_surf is not None:
                _add_case_moisture(moisture, views, cases)
    # Finite physical-bound violations are diagnostics of every validation, never a stop or a selection criterion.
    result: dict[str, float] = {NONFINITE_CASES_KEY: float(nonfinite_cases)}
    # The flat aggregates derive from the per-case evidence: per channel its violating cases and the earliest first violation.
    for channel in bound_channels:
        violating = bound_cases.get(channel, {})
        result[f"bounds/{channel}/violating_cases"] = float(len(violating))
        if violating:
            result[f"bounds/{channel}/first_violation_hours"] = min(violating.values())
    if nonfinite_cases:
        return result
    result.update(metric_impl.finalize_metrics(eval_metrics))
    if one_step_finite:
        result.update({f"guardrail/one_step/{metric_id}": float(metric.compute()) for metric_id, metric in guardrails.items()})
    result.update(_case_moisture_metrics(moisture))
    return result


def evaluate_selected_checkpoint(
    *,
    config: Mapping[str, Any],
    model: nn.Module,
    train_loss: nn.Module,
    eval_loader: DataLoader,
    eval_metrics: dict[str, Any],
    device: torch.device,
    scaling: Any | None,
    checkpoint_identity: Mapping[str, Any],
    best_checkpoint_path: Path | str,
    scheduler_expected: bool,
    max_physics_cases: int,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Load and evaluate the authoritative best checkpoint on one shared science state, on validation membership only."""
    payload = checkpoints.load_checkpoint(
        best_checkpoint_path,
        expected_identity=checkpoint_identity,
        expected_role="best",
        payload_kind=checkpoints.best_payload_kind(config),
        scheduler_expected=scheduler_expected,
        require_best=True,
        adapter_expected=adapter is not None,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    train_loss.load_state_dict(payload["loss_state_dict"], strict=True)

    selected_epoch = int(payload["completed_epoch"])
    if selected_epoch != int(payload["best_epoch"]):
        msg = "Selected best checkpoint epoch disagrees with its best-objective epoch."
        raise RuntimeError(msg)
    extra = () if adapter is None else (adapter,)
    id_metrics = eval_one_epoch(model, eval_loader, eval_metrics, device, scaling, *extra)
    objective_id = str(cast("Mapping[str, Any]", config["evaluation"])["objective"]["id"])
    # The selected checkpoint's ID validation was eligible.
    if not validation_eligible(id_metrics, objective_id):
        msg = f"Selected checkpoint evaluation did not reproduce its eligible ID objective {objective_id!r}."
        raise RuntimeError(msg)

    # One rule for every event and summary: a metric without a finite value is absent.
    selected: dict[str, float] = {f"selected/id/{name}": float(value) for name, value in id_metrics.items() if math.isfinite(value)}
    if adapter is None:
        physics = evaluate_physics_monitor(
            model,
            eval_loader,
            train_loss,
            device,
            scaling,
            max_cases=max_physics_cases,
        )
        for history_key, value in physics.items():
            suffix = history_key.removeprefix("physics/id/")
            selected[f"selected/physics/{suffix}"] = float(value)

    physics_config = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["loss"])["physics"])
    telemetry = getattr(train_loss, "telemetry_state", None)
    if bool(physics_config["enabled"]) and callable(telemetry):
        # The restored loss state holds the warm-up position of the selected checkpoint's last group.
        weights = cast("Mapping[str, float]", telemetry())
        schedules = {term: cast("Mapping[str, Any]", physics_config[f"{term}_weight"]) for term in ("momentum", "continuity", "boundary")}
        applied = {term: float(weights[f"weight_{term}"]) for term in schedules}
        dynamic = any(int(cast("Mapping[str, Any]", schedule["warmup"])["epochs"]) > 0 for schedule in schedules.values())
        differs_from_terminal = any(applied[term] != float(schedule["target"]) for term, schedule in schedules.items())
        if dynamic and differs_from_terminal:
            for term, value in applied.items():
                selected[f"selected/training/{term}_weight"] = value

    return {"selected_epoch": selected_epoch, "selected_metrics": selected}


@contextmanager
def _checkpoint_publication_timer(timing: dict[str, Any], device: torch.device) -> Iterator[None]:
    """Measure capture and publication, including a failed publication attempt."""
    _synchronize_device(device)
    started = time.monotonic()
    try:
        yield
    finally:
        _synchronize_device(device)
        timing["checkpoint_capture_and_publication_seconds"] += time.monotonic() - started
        timing["checkpoint_capture_and_publication_attempts"] += 1


def train_loop(  # noqa: C901, PLR0912, PLR0915
    config: dict[str, Any],
    device: torch.device,
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    train_loss: nn.Module,
    eval_metrics: dict[str, Any],
    scaling: Any | None = None,
    scheduler: LearningRateSchedule | None = None,
    save_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
    event_callback: EventCallback | None = None,
    epoch_state_callback: EpochStateCallback | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    adapter: Any | None = None,
    operational_timing: dict[str, Any] | None = None,
    attempt_work: AttemptWork | None = None,
    budget: StageBudget | None = None,
) -> dict[str, Any]:
    """
    Train one stage over its budget, validating at its budget points, with a bounded ``last`` cadence.

    Training runs epoch after epoch until the stage budget is complete. An event ends
    every stretch of training: the group that reaches a validation point (also inside an
    epoch), the epoch end or the budget end, to which a matched A1 of transition units cuts its last group. At a reached point the
    model is validated once and the bounded physics monitor runs on the validation prefix; points
    the same group also crossed are recorded as skipped in the budget. ``last_checkpoint.pt``
    is published at an event that (1) brought a new best, (2) ends the loop, (3) finds no
    ``last`` in the run directory yet (the first event of a run or stage), or (4) comes at
    least ``LAST_CHECKPOINT_INTERVAL_SECONDS`` of monotonic time after this invocation's
    previous ``last`` (or its start). A new best's ``best_checkpoint.pt`` is
    written together with that ``last``: both files are written and flushed to
    temporary siblings first, then renamed back to back (best, then last), so a
    crash or failure before the renames keeps the previous consistent pair; only
    a crash between the two renames leaves best newer than last, which run
    admission refuses to resume and completed finalization rejects. Every normal
    return therefore has a ``last`` for its terminal event.
    A crash loses the groups since the newest ``last``: resume continues from it at the
    committed group and trains them again.
    An Optuna trial (its config records ``run.layout.trial``) never resumes and so
    never writes a ``last``: a new best is published alone by a single rename, and
    the result reports no ``last_checkpoint_path`` and no ``next_epoch``.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved runtime config retaining the requested device policy.
    device : torch.device
        Concrete indexed runtime device resolved at the service boundary.
    model, optimizer, train_loader, eval_loader, train_loss, eval_metrics : Any
        Already constructed runtime components. Fresh-run seeding must occur
        before their construction; ``eval_loader`` holds the validation membership.
    scaling : Any | None, optional
        Admitted Train scaling of the run (steady); the transient adapter owns its own.
    scheduler : LearningRateSchedule | None, optional
        The stage's own learning-rate schedule; ``None`` keeps the start rate.
    save_dir : Path | str | None, optional
        Canonical run directory. Required for lifecycle execution.
    resume_from : Path | str | None, optional
        Exact ``last_checkpoint.pt`` continuation source.
    event_callback : Callable | None, optional
        Callback invoked with the run's committed optimizer step after every event once its
        evidence is durable: a new best is published, the epoch history is written, and
        ``last`` is published when a new best, the first event, the loop's end, or the
        interval calls for it (``checkpoint/last_published``). ``epoch`` names the epoch in
        progress; validation and physics values and ``validation/point`` are present only at a
        validation point.
    epoch_state_callback : Callable | None, optional
        Callback of either task, invoked once per event that validates or ends an epoch
        (also a budget end or divergence inside an epoch), before that event's
        ``last_checkpoint.pt`` publication, with the epoch, the event's training values
        when it ends the epoch and its complete values when it validated. A ``last``
        therefore never counts a record that was not written
        (:func:`checkpoints.recorded_history_epochs` and its ``objective_history``).
        It owns the durable history, which resume reconciles to the restored checkpoint.
    checkpoint_identity : Mapping[str, Any] | None, optional
        Immutable task/config/dataset/split/objective identity.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with scaling where accepted.
    operational_timing : dict[str, Any] | None, optional
        Attempt-owned measurement sink retained even if checkpoint publication fails.
    attempt_work : AttemptWork | None, optional
        Attempt-owned sink of the committed work at the start and in the newest durable
        state, so the caller can publish it when the run ends with an exception.
    budget : StageBudget
        The stage budget (required); a resume restores its committed state from ``last``.

    Returns
    -------
    dict[str, Any]
        Completed progress, best/last paths, finite objective, history, the
        run's cumulative ``training_work`` counts, its per-update ``update_work`` rows
        and the terminal ``budget`` state.

    Raises
    ------
    ValueError
        If duration, objective, checkpoint identity, or finite-value contracts fail.

    """
    timing = operational_timing if operational_timing is not None else {}
    timing.setdefault("scope", "current_train_loop_invocation")
    timing.update(
        checkpoint_capture_and_publication_seconds=0.0,
        checkpoint_capture_and_publication_attempts=0,
        event_duration_scope=(
            "event_start_through_training_evaluation_best_capture_and_physics; excludes_epoch_state_checkpoint_writes_and_observers"
        ),
    )
    if save_dir is None:
        msg = "Canonical training requires save_dir for best and last checkpoints."
        raise ValueError(msg)
    run_dir = Path(save_dir)
    if not run_dir.is_dir():
        msg = f"Allocated run directory does not exist: {run_dir}"
        raise FileNotFoundError(msg)
    identity = dict(checkpoint_identity or {})
    if not identity:
        msg = "Canonical training requires a non-empty checkpoint_identity."
        raise ValueError(msg)
    if adapter is not None and scaling is not None:
        msg_0 = "Adapter training and a loop scaling are mutually exclusive."
        raise ValueError(msg_0)
    matched_protocol = (
        config.get("task") == "transient_drying"
        and config["training"].get("matched_a1_b") is not None
        and config["training"]["comparison_arm"] in {"b", "a1"}
    )
    if matched_protocol and adapter is None:
        msg_0 = "Matched training requires its transient adapter."
        raise TypeError(msg_0)
    sampler = train_unit_sampler(train_loader)
    # A matched A1 of transition units commits exactly B's transitions; an RNO trajectory unit cannot be cut exactly.
    exact_end = matched_protocol and config["training"]["comparison_arm"] == "a1" and sampler.unit_kind == "transition"
    best_kind = checkpoints.best_payload_kind(config)

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Training requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda" and device.index is None:
        msg = "Training requires an indexed CUDA device resolved by the runtime boundary."
        raise ValueError(msg)
    model = model.to(device)
    if scaling is not None:
        scaling = scaling.to(device)
    task: UnitLossTask = adapter if adapter is not None else _SteadyUnits(scaling, device)

    if not isinstance(budget, StageBudget):
        msg = "Canonical training requires the stage's StageBudget."
        raise TypeError(msg)
    monitor_settings = config["tracking"]["wandb"]["monitor"]
    physics_monitor_enabled = bool(monitor_settings["enabled"])
    physics_monitor_max_cases = int(monitor_settings["max_cases"])
    objective = config["evaluation"]["objective"]
    objective_id = str(objective["id"])
    objective_direction = str(objective["direction"])
    if objective_direction not in {"minimize", "maximize"}:
        msg = f"Unknown objective direction {objective_direction!r}."
        raise ValueError(msg)
    if objective_id not in eval_metrics:
        msg = f"Configured evaluation objective {objective_id!r} is absent from evaluation metrics."
        raise KeyError(msg)

    best_metric: float | None = None
    best_epoch: int | None = None
    objective_history: list[dict[str, Any]] = []
    terminal_epoch_metrics: dict[str, float] = {}
    global_step = 0
    # Committed optimizer groups and the predictions that entered the loss
    # with a gradient; a fresh run or stage starts at zero, a resume continues its last checkpoint's counts.
    training_work = dict.fromkeys(sorted(checkpoints.TRAINING_WORK_KEYS), 0)
    attempt = AttemptWork() if attempt_work is None else attempt_work
    # One checkpoint.UPDATE_WORK_COLUMNS row per committed update, kept with the counts it sums to.
    update_work: list[list[int]] = []
    start_epoch_index = 0
    finalize_only = False
    divergence: str | None = None

    if resume_from is not None:
        resume_path = Path(resume_from)
        last_payload = checkpoints.load_checkpoint(
            resume_path,
            expected_identity=identity,
            expected_role="last",
            payload_kind="full",
            scheduler_expected=scheduler is not None,
            require_best=False,
            adapter_expected=adapter is not None,
            resume=True,
        )
        restored = checkpoints.restore_checkpoint(
            last_payload,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=train_loss,
            train_loader=train_loader,
            budget=budget,
            adapter=adapter,
            adapter_expected=adapter is not None,
        )
        start_epoch_index = int(restored["next_epoch"]) - 1
        global_step = int(restored["global_step"])
        training_work.update(restored["training_work"])
        update_work = restored["update_work"]
        best_metric = restored["best_metric"]
        best_epoch = restored["best_epoch"]
        objective_history = list(restored["objective_history"])
        divergence = restored["divergence"]
        # A completed stage budget or a recorded divergence already published the terminal best and last
        # checkpoints; resuming them only finalizes, because no further optimizer group is admitted.
        finalize_only = budget.complete or divergence is not None

    best_path = common.paths.resolve_best_checkpoint_file(run_dir)
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    # Admission gives the run one writer, so staged files of an earlier crashed publication are ours to drop.
    checkpoints.discard_staged_checkpoints(best_path, last_path)
    trial_run = experiments.run.is_trial_config(config)
    session_started = time.perf_counter()
    last_published_at = time.monotonic()

    def committed_work() -> dict[str, Any]:
        """Return the committed counts and optimisation seconds of the current state."""
        return {**training_work, "train_seconds": budget.train_seconds}

    attempt.at_start, attempt.durable = committed_work(), committed_work()
    progress_at_start = budget.progress
    # Stage start or restore: the optimizer holds η(p) of the committed progress before any group trains.
    if scheduler is not None:
        scheduler.advance(budget.progress)

    def capture(role: checkpoints.CheckpointRole, payload_kind: checkpoints.CheckpointPayloadKind, completed_epoch: int) -> dict[str, Any]:
        """Capture one checkpoint of the current committed state."""
        return checkpoints.make_checkpoint(
            role=role,
            payload_kind=payload_kind,
            identity=identity,
            completed_epoch=completed_epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=train_loss,
            best_metric=best_metric,
            best_epoch=best_epoch,
            objective_history=objective_history,
            train_loader=train_loader,
            runtime_device=device,
            training_work=training_work,
            update_work=update_work,
            budget=budget,
            adapter=adapter,
            divergence=divergence if role == "last" else None,
        )

    def validate(completed_epoch: int, evaluated: dict[str, float]) -> dict[str, Any] | None:
        """Validate once at the reached budget point; return a new best's payload."""
        nonlocal best_metric, best_epoch
        point = budget.record_validation()
        _synchronize_device(device)
        if device.type == "cuda":
            # Item 175: return training's cached blocks before validation's differently shaped batches; the memory
            # admission rule (85 % of the reserved peak) is measured with this release in place.
            torch.cuda.empty_cache()
        id_started = time.perf_counter()
        # The point keeps the validation role's per-case bound evidence.
        bound_cases: dict[str, dict[str, float]] = {}
        with _training_phase("id_evaluation", completed_epoch):
            id_metrics = eval_one_epoch(
                model, eval_loader, eval_metrics, device, scaling, *(() if adapter is None else (adapter,)), bound_cases=bound_cases
            )
        _synchronize_device(device)
        evaluated["system/id_evaluation_duration_seconds"] = time.perf_counter() - id_started
        if adapter is not None:
            adapter.record_validation_work(evaluated["system/id_evaluation_duration_seconds"])
        evaluated["system/id_evaluation_case_count"] = float(_loader_case_count(eval_loader))
        if not id_metrics[NONFINITE_CASES_KEY] and objective_id not in id_metrics:
            msg = f"Configured evaluation objective {objective_id!r} was not produced by ID evaluation metrics."
            raise KeyError(msg)
        eligible = validation_eligible(id_metrics, objective_id)
        current_metric = float(id_metrics[objective_id]) if eligible else None
        # Non-finite values never enter the event: the local history and the observers keep finite evidence only.
        evaluated.update({f"id/{name}": value for name, value in id_metrics.items() if math.isfinite(value)})
        evaluated["validation/point"] = float(point)
        evaluated["validation/skipped_points"] = float(len(budget.skipped_points))
        evaluated["validation/eligible"] = float(eligible)
        # A non-eligible point keeps its entry without a value.
        objective_history.append(
            {"point": point, "epoch": completed_epoch, "objective_id": objective_id, "value": current_metric, "bound_violations": bound_cases}
        )
        if current_metric is not None and adapter is not None:
            # Training stops at the group that completes the budget, so no validation sees post-budget updates.
            adapter.record_within_budget_evaluation(current_metric, epoch_index=completed_epoch - 1)
        if current_metric is not None and scheduler is not None:
            old_learning_rate = float(optimizer.param_groups[0]["lr"])
            with _training_phase("scheduler_update", completed_epoch):
                scheduler.validated(current_metric)
            new_learning_rate = float(optimizer.param_groups[0]["lr"])
            if new_learning_rate != old_learning_rate:
                evaluated["optimization/scheduler_old_learning_rate"] = old_learning_rate
                evaluated["optimization/scheduler_new_learning_rate"] = new_learning_rate
        best_payload = None
        previous_best = best_metric
        # Training continues at a non-eligible point; it never becomes the best.
        is_better = current_metric is not None and (
            best_metric is None or (current_metric < best_metric if objective_direction == "minimize" else current_metric > best_metric)
        )
        evaluated["checkpoint/new_best"] = float(is_better)
        if previous_best is not None:
            evaluated["checkpoint/previous_best_objective"] = float(previous_best)
        if is_better and current_metric is not None:
            best_metric = current_metric
            best_epoch = completed_epoch
            with _checkpoint_publication_timer(timing, device):
                best_payload = capture("best", best_kind, completed_epoch)
        # The steady physics monitor rejects non-finite predictions, so it follows complete ID validations only.
        if physics_monitor_enabled and not id_metrics[NONFINITE_CASES_KEY]:
            _synchronize_device(device)
            physics_started = time.perf_counter()
            with _training_phase("physics_monitor", completed_epoch):
                evaluated.update(evaluate_physics_monitor(model, eval_loader, train_loss, device, scaling, max_cases=physics_monitor_max_cases))
            _synchronize_device(device)
            evaluated["system/physics_monitor_duration_seconds"] = time.perf_counter() - physics_started
            evaluated["system/physics_monitor_case_count"] = float(min(physics_monitor_max_cases, _loader_case_count(eval_loader)))
        return best_payload

    # The terminal epoch of a finalize-only resume is the one its last checkpoint recorded.
    completed_epoch = int(last_payload["completed_epoch"]) if finalize_only else start_epoch_index
    epoch_index = start_epoch_index
    # A zero transitions budget (an A1 matched to a B that diverged before its first committed group) admits no group.
    ends_loop = finalize_only or budget.complete
    while not ends_loop:
        completed_epoch = epoch_index + 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # A resumed partial epoch keeps its committed order and position.
        epoch_start_sampler = sampler.state_dict()
        sampler.begin_epoch(epoch_index)
        epoch_complete = False
        while not (epoch_complete or ends_loop):
            event_started = time.perf_counter()
            if exact_end and -budget.overrun < sampler.size - sampler.cursor:
                # The budget ends inside this epoch: the group reaching it is cut to the remaining transition units.
                sampler.end_after(-budget.overrun)
            with _training_phase("training_epoch", completed_epoch):
                train_metrics = train_update_groups(
                    model,
                    train_loader,
                    optimizer,
                    train_loss,
                    device,
                    task,
                    budget=budget,
                    scheduler=scheduler,
                    adapter=adapter,
                    attempt=attempt,
                )
            if exact_end and budget.overrun > 0:
                msg = f"A matched A1 must commit exactly B's transitions, one per unit; it exceeded them by {budget.overrun}."
                raise RuntimeError(msg)
            optimizer_steps = int(train_metrics.pop("optimizer_steps"))
            global_step += optimizer_steps
            training_work["optimizer_steps"] += optimizer_steps
            for key in ("gradient_predictions", "units", "update_distinct_cases"):
                training_work[key] += int(train_metrics.pop(key))
            update_work.extend(train_metrics.pop("update_work"))
            divergence = train_metrics.pop("divergence")
            epoch_complete = sampler.epoch_complete
            # A divergence before this epoch's first committed group ends the stage at the previous epoch's end,
            # whose own event already recorded it; the sampler returns to that committed state.
            recorded_end = divergence is not None and sampler.cursor == 0
            if recorded_end:
                if epoch_start_sampler["permutation"]:
                    sampler.load_state_dict(epoch_start_sampler)
                completed_epoch -= 1
            # The budget end or a divergence ends the stage; one predicate both exits the loop and forces the
            # terminal last, so the two cannot drift.
            ends_loop = budget.complete or divergence is not None

            evaluated: dict[str, float] = {}
            best_payload = validate(completed_epoch, evaluated) if budget.validation_due else None

            train_metrics["epoch"] = float(completed_epoch)
            train_metrics["global_step"] = float(global_step)
            train_metrics["optimization/learning_rate"] = float(optimizer.param_groups[0]["lr"])
            committed = budget.state_dict()
            train_metrics.update({f"budget/{key}": float(committed[key]) for key in ("units", "supervised_transitions", "train_seconds")})
            train_metrics["budget/progress"] = budget.progress
            event_duration = time.perf_counter() - event_started
            if not math.isfinite(event_duration) or event_duration <= 0.0:
                msg = f"Training event in epoch {completed_epoch} has invalid duration {event_duration}."
                raise RuntimeError(msg)
            train_metrics["system/event_duration_seconds"] = event_duration
            elapsed = time.perf_counter() - session_started
            train_metrics["system/session_elapsed_seconds"] = elapsed
            if budget.progress > progress_at_start:
                train_metrics["system/estimated_remaining_seconds"] = elapsed * (1.0 - budget.progress) / (budget.progress - progress_at_start)
            if device.type == "cuda":
                train_metrics["system/cuda_peak_memory_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
            if adapter is not None:
                train_metrics.update({f"transient/{key}": float(value) for key, value in adapter.telemetry_state().items() if value is not None})
            event_metrics = {**train_metrics, **evaluated}
            # The history holds one record per validation point, also inside an epoch, and one per completed epoch,
            # also for a budget end or divergence inside an epoch; both precede the event's last publication.
            ends_epoch = (epoch_complete or ends_loop) and not recorded_end
            if epoch_state_callback is not None and (ends_epoch or evaluated):
                with _training_phase("epoch_state", completed_epoch):
                    epoch_state_callback(
                        completed_epoch,
                        copy.deepcopy(train_metrics) if ends_epoch else None,
                        copy.deepcopy(event_metrics) if evaluated else None,
                    )

            # A new best is published only together with its last, so a published best is never newer than the
            # newest last; the first event of a run or stage gets one, so an early crash leaves a resumable run.
            # A divergence in the stage's very first group leaves no committed state to checkpoint.
            publish_last = (
                not trial_run
                and completed_epoch >= 1
                and (
                    ends_loop
                    or best_payload is not None
                    or not last_path.exists()
                    or time.monotonic() - last_published_at >= LAST_CHECKPOINT_INTERVAL_SECONDS
                )
            )
            if trial_run and best_payload is not None:
                # A trial never resumes: its new best is the only file, published by one rename.
                with _checkpoint_publication_timer(timing, device), _training_phase("best_checkpoint", completed_epoch):
                    checkpoints.save_checkpoints((best_payload, best_path))
            if publish_last:
                with _checkpoint_publication_timer(timing, device):
                    last_payload = capture("last", "full", completed_epoch)
                    with _training_phase("last_checkpoint", completed_epoch):
                        # Both files are staged before either rename, so a failure keeps the previous pair.
                        best_publication = ((best_payload, best_path),) if best_payload is not None else ()
                        checkpoints.save_checkpoints(*best_publication, (last_payload, last_path))
                last_published_at = time.monotonic()
            if publish_last or trial_run:
                attempt.durable = committed_work()
            event_metrics["checkpoint/last_published"] = float(publish_last)
            terminal_epoch_metrics = event_metrics
            # The step-indexed observers get a point only for a committed step; a divergence in a stretch's first
            # group commits none, and its end reaches them through the run's terminal status.
            if event_callback is not None and optimizer_steps:
                with _training_phase("event_observers", completed_epoch):
                    event_callback(global_step, event_metrics)
        epoch_index += 1

    terminal_epoch = completed_epoch
    # Only a divergence ends a stage before its budget: no stage has a cap that could stop it earlier.
    if divergence is None and not budget.complete:
        msg = "Training ended before its stage budget completed."
        raise RuntimeError(msg)
    if adapter is not None and best_metric is not None:
        adapter.validate_terminal_state(best_metric=best_metric, best_epoch=best_epoch)
    status = "diverged" if divergence is not None else ("completed" if best_metric is not None else "no_eligible_checkpoint")
    # A diverged run's result is its best eligible checkpoint up to the divergence; without one it is unsuccessful.
    training_outcome = {
        "status": status,
        "budget_progress": budget.progress,
        "eligible_points": sum(entry["value"] is not None for entry in objective_history),
        "divergence": divergence,
    }
    if best_metric is None:
        return {
            "completed_epoch": terminal_epoch,
            "next_epoch": None if trial_run or terminal_epoch < 1 else last_payload["next_epoch"],
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "objective": dict(objective),
            "objective_history": objective_history,
            "terminal_epoch": terminal_epoch,
            "terminal_metrics": {key: value for key, value in terminal_epoch_metrics.items() if key.startswith("train/")},
            "checkpoint_path": str(best_path) if best_path.exists() else None,
            "best_checkpoint_path": str(best_path) if best_path.exists() else None,
            "last_checkpoint_path": None if trial_run or terminal_epoch < 1 else str(last_path),
            "status": status,
            "training_outcome": training_outcome,
            "operational_timing": copy.deepcopy(timing),
            "training_work": dict(training_work),
            "update_work": [list(row) for row in update_work],
            "budget": budget.state_dict(),
        }
    best_checkpoint = checkpoints.load_checkpoint(
        best_path,
        expected_identity=identity,
        expected_role="best",
        payload_kind=best_kind,
        scheduler_expected=scheduler is not None,
        require_best=True,
        adapter_expected=adapter is not None,
    )
    if not trial_run:
        last_checkpoint = checkpoints.load_checkpoint(
            last_path,
            expected_identity=identity,
            expected_role="last",
            payload_kind="full",
            scheduler_expected=scheduler is not None,
            require_best=True,
            adapter_expected=adapter is not None,
        )
        if last_checkpoint["completed_epoch"] != terminal_epoch:
            msg = "Last checkpoint does not represent the configured terminal epoch."
            raise RuntimeError(msg)
        if best_checkpoint["best_metric"] != last_checkpoint["best_metric"] or best_checkpoint["best_epoch"] != last_checkpoint["best_epoch"]:
            msg = "Best and last checkpoints disagree about the selected objective state."
            raise RuntimeError(msg)

    return {
        "completed_epoch": terminal_epoch,
        "next_epoch": None if trial_run else last_payload["next_epoch"],
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "objective": dict(objective),
        "objective_history": objective_history,
        "terminal_epoch": terminal_epoch,
        "terminal_metrics": {key: value for key, value in terminal_epoch_metrics.items() if key.startswith("train/")},
        "checkpoint_path": str(best_path),
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": None if trial_run else str(last_path),
        "status": status,
        "training_outcome": training_outcome,
        "operational_timing": copy.deepcopy(timing),
        "training_work": dict(training_work),
        "update_work": [list(row) for row in update_work],
        "budget": budget.state_dict(),
    }
