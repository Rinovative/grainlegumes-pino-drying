"""Run the custom training and evaluation loop with checkpoint support.

Responsibilities:
  - Execute completed training, ID, OOD, and bounded physics events
  - Use one terminal-inclusive predicate with independently resolved phase intervals
  - Consume the resolved semantic objective for scheduler and best-metric updates
  - Manage checkpoints, histories, and reproducibility state
  - Preserve matched-comparison sampler cursors and committed optimizer-group timing
  - Count each run's committed optimizer groups and gradient-carrying predictions at the group commit boundary
  - Invoke optional epoch-end callbacks for local lifecycle observers

Design principles:
  - Reproducibility state is explicit in last checkpoints; a best keeps it only as the matched-A0 handoff source
  - Data, model and loss objects are caller-provided
  - Controller behavior stays independent of model architecture

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
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence, Sized
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from src import common, datasets, experiments
from src.learning.metrics import learning_metrics as metric_impl

from . import learning_training_checkpoint as checkpoints
from . import learning_training_events as training_events
from .learning_training_adapter import OptimizerWork

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.optimizer import Optimizer

TensorBatch = dict[str, torch.Tensor]
_TRANSIENT_VIEW_RANK = 5
LAST_CHECKPOINT_INTERVAL_SECONDS = 30 * 60  # bounds training lost to a crash; a multi-GB full-state write per epoch costs more
EpochEndCallback = Callable[[int, dict[str, float]], None]
EpochStateCallback = Callable[[int, dict[str, float]], None]


class MatchedPermutationSampler(Sampler[list[int]]):
    """Persist the exact shuffled epoch permutation and consumed sample cursor."""

    def __init__(self, size: int, generator: torch.Generator, *, batch_size: int, keys: Sequence[Hashable] | None = None) -> None:
        """Create an empty permutation bound to one explicit loader generator and batch rule."""
        if size < 1:
            msg = "Matched training requires a nonempty train Dataset."
            raise ValueError(msg)
        if keys is not None and len(keys) != size:
            msg = "Matched batch keys must cover every train item."
            raise ValueError(msg)
        self.size = size
        self.generator = generator
        self.batch_size = batch_size
        self.keys = None if keys is None else tuple(keys)
        self.epoch = -1
        self.permutation: list[int] = []
        self.cursor = 0
        self._batch_count = len(datasets.runtime.factory.group_batches(range(size), self.keys, batch_size))

    def begin_epoch(self, epoch: int) -> None:
        """Draw one new permutation only after the previous epoch finished."""
        if epoch == self.epoch and self.permutation:
            return
        if epoch != self.epoch + 1 or (self.permutation and self.cursor != self.size):
            msg = "Matched sampler epoch progression conflicts with durable cursor."
            raise ValueError(msg)
        self.permutation = torch.randperm(self.size, generator=self.generator).tolist()
        self.cursor = 0
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the unconsumed batches of the persisted permutation, independent of loader prefetch."""
        batches = datasets.runtime.factory.group_batches(self.permutation, self.keys, self.batch_size)
        consumed = 0
        for index, batch in enumerate(batches):
            if consumed == self.cursor:
                return iter(batches[index:])
            consumed += len(batch)
        if consumed != self.cursor:
            msg = "Matched sampler cursor does not fall on a batch boundary of its permutation."
            raise ValueError(msg)
        return iter(())

    def __len__(self) -> int:
        """Keep the original full-epoch batch count visible to the loop."""
        return self._batch_count

    def advance(self, samples: int) -> None:
        """Commit only batches whose optimizer group completed."""
        if samples < 1 or self.cursor + samples > self.size:
            msg = "Matched sampler cursor would leave its fixed permutation."
            raise ValueError(msg)
        self.cursor += samples

    def state_dict(self) -> dict[str, Any]:
        """Return the exact mid-epoch continuation contract."""
        return {
            "size": self.size,
            "epoch": self.epoch,
            "permutation": list(self.permutation),
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Reject missing, duplicated, or out-of-range sample positions."""
        keys = {"size", "epoch", "permutation", "cursor"}
        if set(state) != keys or state["size"] != self.size:
            msg = "Matched sampler state schema or Dataset length differs."
            raise ValueError(msg)
        permutation = state["permutation"]
        if not isinstance(permutation, list) or sorted(permutation) != list(range(self.size)):
            msg = "Matched sampler permutation is invalid."
            raise ValueError(msg)
        cursor = state["cursor"]
        epoch = state["epoch"]
        if type(cursor) is not int or not 0 <= cursor <= self.size or type(epoch) is not int or epoch < 0:
            msg = "Matched sampler epoch or cursor is invalid."
            raise ValueError(msg)
        self.permutation = list(permutation)
        self.cursor = cursor
        self.epoch = epoch


def _matched_train_loader(loader: DataLoader[Any]) -> DataLoader[Any]:
    """Preserve loader semantics while making consumed batch order checkpointable."""
    if loader.num_workers != 0:
        msg = "Exact matched partial-epoch resume currently requires data.num_workers=0."
        raise ValueError(msg)
    if loader.drop_last:
        msg = "Exact matched partial-epoch resume requires drop_last=false."
        raise ValueError(msg)
    # Grouped trajectory batches carry their batch size and per-item keys on the batch sampler.
    grouped = loader.batch_sampler if isinstance(loader.batch_sampler, datasets.runtime.factory.GroupedBatchSampler) else None
    batch_size = grouped.batch_size if grouped is not None else loader.batch_size
    if not isinstance(loader.generator, torch.Generator) or batch_size is None:
        msg = "Matched training requires an explicit generator and fixed batch size."
        raise TypeError(msg)
    if not isinstance(loader.dataset, Sized):
        msg = "Matched training requires a sized Dataset."
        raise TypeError(msg)
    sampler = MatchedPermutationSampler(
        len(loader.dataset), loader.generator, batch_size=batch_size, keys=grouped.keys if grouped is not None else None
    )
    return DataLoader(
        loader.dataset,
        batch_sampler=sampler,
        generator=loader.generator,
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
    )


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


def _configured_ood_loaders(ood_loaders: Mapping[str, DataLoader | None] | None) -> dict[str, DataLoader]:
    """Return the non-empty OOD Evaluation loaders in canonical role order; an empty part is never evaluated."""
    supplied = dict(ood_loaders or {})
    unknown = set(supplied).difference(datasets.preprocessing.splits.OOD_SPLIT_ROLES)
    if unknown:
        msg = f"OOD Evaluation loaders name unknown roles {sorted(unknown)}."
        raise ValueError(msg)
    return {role: loader for role in datasets.preprocessing.splits.OOD_SPLIT_ROLES if (loader := supplied.get(role)) is not None}


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


def _prepare_batch(
    raw_batch: Any,
    device: torch.device,
    data_processor: Any | None,
    *,
    training: bool,
) -> TensorBatch:
    """
    Prepare one raw batch with the processor in the requested lifecycle mode.

    When present, the processor is switched to train/eval state and preprocesses
    a copied mapping before tensor transfer. The final batch must contain tensor
    ``x`` and ``y`` keys on the concrete device.
    """
    if data_processor is None:
        return _move_batch_to_device(raw_batch, device)

    if training:
        data_processor.train()
    else:
        data_processor.eval()

    processed = data_processor.preprocess(dict(raw_batch))
    return _move_batch_to_device(processed, device)


def _compute_loss(loss_fn: nn.Module, pred: torch.Tensor, batch: TensorBatch) -> torch.Tensor:
    """Compute one semantic composition or conventional supervised loss."""
    if hasattr(loss_fn, "compute_components"):
        return loss_fn(pred, x=batch["x"], y=batch["y"])
    return loss_fn(pred, batch["y"])


def _require_finite_training_loss(loss: torch.Tensor) -> None:
    """Reject a non-finite batch loss before backward or optimizer mutation."""
    if not bool(torch.isfinite(loss.detach()).all().item()):
        msg = f"Training loss is non-finite before backward: {loss.detach()}."
        raise FloatingPointError(msg)


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


def _require_adapter_step(step: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, int, int, int]:
    """Validate one adapter microbatch result before backward work begins."""
    loss, components = getattr(step, "loss", None), getattr(step, "components", None)
    counts = (getattr(step, "sample_count", None), getattr(step, "processed_target_transitions", None), getattr(step, "forward_transitions", None))
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all().item()):
        msg = "Adapter training loss must be one finite scalar before backward."
        raise FloatingPointError(msg)
    if not isinstance(components, Mapping) or "total" not in components:
        msg = "Adapter training_step components must contain total."
        raise TypeError(msg)
    for name, value in components.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(torch.isfinite(value.detach()).all().item()):
            msg = f"Adapter component {name!r} must be finite and scalar."
            raise FloatingPointError(msg)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        msg = "Adapter work counts must be positive integers."
        raise ValueError(msg)
    consumed = getattr(step, "consumed_items", None)
    if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
        msg = "Adapter consumed_items must be a non-negative integer."
        raise ValueError(msg)
    return loss, dict(components), cast("int", counts[0]), cast("int", counts[1]), cast("int", counts[2]), consumed


def _cuda_timing_events(device: torch.device) -> tuple[Any, Any]:
    """Return a recorded start and an unrecorded end CUDA event, or no events on CPU."""
    if device.type != "cuda":
        return None, None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    return start, end


def _train_adapter_one_epoch(  # noqa: C901, PLR0915
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    adapter: Any,
    *,
    max_optimizer_steps: int | None = None,
    max_train_seconds: float | None = None,
) -> dict[str, float]:
    """
    Train task-owned batches with transition-weighted accumulation.

    Every loader batch yields one or more adapter microbatches; the next one is
    computed only after the preceding optimizer group has been stepped. A
    completed stage budget stops at the exact optimizer group, while operational
    caps and the matched sampler cursor advance only at loader-batch boundaries.
    """
    accumulation = getattr(adapter, "gradient_accumulation_steps", None)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        msg = "Adapter gradient_accumulation_steps must be a positive integer."
        raise ValueError(msg)
    model.train()
    sums: dict[str, float] = {}
    samples = processed = forwarded = microbatches = groups = optimizer_steps = committed_predictions = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad()
    group: list[tuple[int, int, Any, Any, int]] = []
    group_started = 0.0
    group_loader_wait = 0.0
    matched_sampler = (
        train_loader.batch_sampler
        if isinstance(train_loader, DataLoader) and isinstance(train_loader.batch_sampler, MatchedPermutationSampler)
        else None
    )
    cap_reached = False

    def reached_operational_cap() -> bool:
        return bool(
            (max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps)
            or (max_train_seconds is not None and adapter.budget_state()["train_seconds"] >= max_train_seconds)
        )

    def finish() -> None:
        nonlocal groups, optimizer_steps, committed_predictions, group, group_loader_wait
        if not group:
            return
        total = sum(item[0] for item in group)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(total)
        optimizer.step()
        optimizer.zero_grad()
        if device.type == "cuda":
            group[-1][3].record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            seconds = sum(float(start.elapsed_time(end)) / 1000.0 for _, _, start, end, _ in group)
            if not math.isfinite(seconds) or seconds < 0.0:
                msg = "Adapter CUDA timing is invalid."
                raise RuntimeError(msg)
        else:
            seconds = None
        consumed_items = sum(item[4] for item in group)
        if matched_sampler is not None and consumed_items:
            matched_sampler.advance(consumed_items)
        train_seconds = max(0.0, time.perf_counter() - group_started - group_loader_wait) if matched_sampler is not None else None
        adapter.record_optimizer_work(
            OptimizerWork(
                microbatches=len(group),
                processed_target_transitions=total,
                forward_transitions=sum(item[1] for item in group),
                optimizer_device_seconds=seconds,
                wall_seconds=time.perf_counter() - group_started,
                peak_cuda_memory_bytes=(int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
                train_seconds=train_seconds,
                batch_cursor=matched_sampler.cursor if matched_sampler is not None else None,
            )
        )
        groups += 1
        optimizer_steps += 1
        committed_predictions += total
        group = []
        group_loader_wait = 0.0

    resume_generator_state = (
        train_loader.generator.get_state().clone()
        if matched_sampler is not None and matched_sampler.cursor > 0 and isinstance(train_loader.generator, torch.Generator)
        else None
    )
    iterator = iter(train_loader)
    if resume_generator_state is not None and isinstance(train_loader.generator, torch.Generator):
        train_loader.generator.set_state(resume_generator_state)
    while True:
        wait_started = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration:
            if group:
                group_loader_wait += time.perf_counter() - wait_started
            break
        if group:
            group_loader_wait += time.perf_counter() - wait_started
        else:
            group_started = time.perf_counter()
        batch = adapter.prepare_batch(raw_batch, device=device, training=True)
        steps = adapter.training_steps(model, batch, loss_fn)
        budget_complete = False
        while True:
            start_event, end_event = _cuda_timing_events(device)
            step = next(steps, None)
            if step is None:
                break
            loss, components, count, targets, forward, consumed = _require_adapter_step(step)
            (loss * targets).backward()
            if end_event is not None and len(group) + 1 < accumulation:
                end_event.record(torch.cuda.current_stream(device))
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().item()) * targets
            samples += count
            processed += targets
            forwarded += forward
            microbatches += 1
            group.append((targets, forward, start_event, end_event, consumed))
            if len(group) == accumulation:
                finish()
                budget_complete = bool(adapter.should_stop())
                if budget_complete:
                    break
                # The next window of the same loader batch opens the next optimizer group.
                group_started = time.perf_counter()
        if budget_complete:
            break
        cap_reached = reached_operational_cap()
        if cap_reached:
            break
    finish()
    cap_reached = cap_reached or reached_operational_cap()
    if samples == 0:
        msg = "Adapter training loader produced no work."
        raise RuntimeError(msg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    if not math.isfinite(duration) or duration <= 0.0:
        msg = f"Training phase has invalid duration {duration}."
        raise RuntimeError(msg)
    averaged = {name: value / processed for name, value in sums.items()}
    result = {
        "train/loss_total": averaged["total"],
        "train/loss_data": averaged.get("data", averaged["total"]),
        "system/train_duration_seconds": duration,
        "system/train_samples_per_second": samples / duration,
        "optimizer_steps": float(optimizer_steps),
        "committed_optimizer_groups": float(groups),
        "gradient_predictions": float(committed_predictions),
        "transient/train/samples": float(samples),
        "transient/train/processed_target_transitions": float(processed),
        "transient/train/forward_transitions": float(forwarded),
        "transient/train/microbatches": float(microbatches),
        "transient/train/optimizer_groups": float(groups),
    }
    if matched_sampler is not None:
        result["transient/train/operational_cap_reached"] = float(cap_reached)
    result.update({f"train/loss_{name}": value for name, value in averaged.items()})
    return result


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    data_processor: Any | None = None,
    adapter: Any | None = None,
    max_optimizer_steps: int | None = None,
    max_train_seconds: float | None = None,
) -> dict[str, float]:
    """
    Execute one training epoch and sample-weight every named loss component.

    Batch means are multiplied by their actual sample counts before epoch
    reduction. Training duration covers dataloader waiting through final batch
    bookkeeping and optimizer work, while complete epoch duration remains a
    separate outer lifecycle metric. Throughput is actual processed samples
    divided by this training-phase duration. Batch size remains relevant context.
    Supervised runs publish only total/data loss. Physics-informed runs additionally
    publish momentum, continuity, and boundary contributions,
    applied weights, and warmup fractions.

    Parameters
    ----------
    model : torch.nn.Module
        Model mutated through training-mode forward/backward updates.
    train_loader : DataLoader
        Loader providing mapping batches with ``x`` and ``y`` tensors.
    optimizer : Optimizer
        Optimizer stepped once per finite batch.
    loss_fn : torch.nn.Module
        Conventional loss or semantic composition exposing named components.
    device : torch.device
        Concrete device receiving each prepared batch.
    data_processor : Any | None, optional
        Processor switched to training mode before preprocessing each batch.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with data_processor where accepted.
    max_optimizer_steps : int | None, optional
        Stop the adapter phase after this many completed optimizer steps.
    max_train_seconds : float | None, optional
        Bound the adapter run by measured training duration.

    Returns
    -------
    dict[str, float]
        Stable per-epoch training telemetry plus internal optimizer-step,
        committed-group and gradient-prediction counts consumed by the
        checkpoint lifecycle. A steady sample is one gradient prediction, a
        transient target transition is one.

    Raises
    ------
    ValueError
        If an adapter and a data processor are both supplied.
    FloatingPointError
        If any batch loss is non-finite before optimizer mutation.
    RuntimeError
        If the loader is empty or physics component telemetry is incomplete.

    """
    if adapter is not None:
        if data_processor is not None:
            msg_0 = "Adapter training and data_processor are mutually exclusive."
            raise ValueError(msg_0)
        return _train_adapter_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            adapter,
            max_optimizer_steps=max_optimizer_steps,
            max_train_seconds=max_train_seconds,
        )

    model.train()
    component_sums: dict[str, float] = {}
    sample_count = 0
    optimizer_steps = committed_groups = committed_predictions = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    for raw_batch in train_loader:
        batch = _prepare_batch(raw_batch, device, data_processor, training=True)
        batch_samples = _training_batch_size(batch)
        optimizer.zero_grad()

        pred = model(batch["x"])
        loss = _compute_loss(loss_fn, pred, batch)
        _require_finite_training_loss(loss)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        committed_groups += 1
        committed_predictions += batch_samples

        raw_components = getattr(loss_fn, "last_components", {})
        components = (
            dict(raw_components) if isinstance(raw_components, Mapping) and raw_components else {"total": loss.detach(), "data": loss.detach()}
        )
        for name, value in components.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                msg = f"Training loss component {name!r} must be one scalar tensor."
                raise TypeError(msg)
            component_sums[name] = component_sums.get(name, 0.0) + float(value.item()) * batch_samples
        sample_count += batch_samples

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_duration = time.perf_counter() - train_started
    if sample_count == 0:
        msg = "Training loader produced no samples."
        raise RuntimeError(msg)
    if not math.isfinite(train_duration) or train_duration <= 0.0:
        msg = f"Training phase has invalid duration {train_duration}."
        raise RuntimeError(msg)
    train_samples_per_second = sample_count / train_duration
    if not math.isfinite(train_samples_per_second) or train_samples_per_second <= 0.0:
        msg = f"Training phase has invalid throughput {train_samples_per_second}."
        raise RuntimeError(msg)

    averaged = {name: value / sample_count for name, value in component_sums.items()}
    result = {
        "train/loss_total": averaged["total"],
        "train/loss_data": averaged.get("data", averaged["total"]),
        "system/train_duration_seconds": train_duration,
        "system/train_samples_per_second": train_samples_per_second,
        "optimizer_steps": float(optimizer_steps),
        "committed_optimizer_groups": float(committed_groups),
        "gradient_predictions": float(committed_predictions),
    }
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
    data_processor: Any,
    *,
    max_cases: int,
) -> dict[str, float]:
    """
    Evaluate bounded deterministic physics monitors on the saved eval prefix.

    The caller persists the exact prefix membership. This function consumes no
    more than ``max_cases`` in loader order, reuses the semantic loss's domain
    physics evaluator and saved normalizers, and produces no artifact files.

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
    data_processor : Any
        Fitted processor owning the saved normalizers.
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
                    data_processor,
                    training=False,
                )
                try:
                    pred = model(batch["x"])
                    diagnostics = cast("Any", compute(pred, x=batch["x"]))
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
    data_processor: Any | None = None,
    adapter: Any | None = None,
) -> dict[str, float]:
    """
    Execute evaluation with explicit tensor spaces and dataset accumulation.

    Evaluation preprocessing normalizes model inputs while preserving physical
    targets. This function derives normalized targets once with the fitted
    output normalizer and derives physical predictions once by inverse transform.
    Metrics accumulate sufficient statistics and finalize only after the loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model evaluated without gradients.
    eval_loader : Iterable[Any]
        Loader providing task ``x`` and ``y`` batches.
    eval_metrics : dict[str, Any]
        Metric-ID keyed explicit-space dataset accumulators.
    device : torch.device
        Concrete device receiving prepared batches.
    data_processor : Any | None, optional
        Fitted processor needed for normalized/physical view construction.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with data_processor where accepted.

    Returns
    -------
    dict[str, float]
        One finalized dataset value per configured metric ID.

    Raises
    ------
    ValueError
        If metric IDs, spaces, or tensor views are inconsistent.
    RuntimeError
        If required normalizer state or a requested tensor space is unavailable.

    Notes
    -----
    Dataset accumulators are reset at entry and finalized once after all batches.
    No batch-level metric means are averaged.

    """
    if adapter is not None:
        if data_processor is not None:
            msg = "Adapter evaluation and data_processor are mutually exclusive."
            raise ValueError(msg)
        return _eval_adapter_one_epoch(model, eval_loader, eval_metrics, device, adapter)
    model.eval()
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()

    metric_spaces = {str(getattr(metric, "space", "")) for metric in eval_metrics.values()}
    requires_physical = "physical" in metric_spaces
    if requires_physical and data_processor is None:
        msg = "Physical evaluation metrics require a fitted data processor."
        raise RuntimeError(msg)
    out_normalizer = getattr(data_processor, "out_normalizer", None) if data_processor is not None else None
    if data_processor is not None and out_normalizer is None:
        msg = "Evaluation with a data processor requires data_processor.out_normalizer."
        raise RuntimeError(msg)

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            batch = _prepare_batch(raw_batch, device, data_processor, training=False)
            pred_normalized = model(batch["x"])
            if data_processor is None:
                target_normalized = batch["y"]
                target_physical = None
            else:
                if out_normalizer is None:
                    msg = "Evaluation normalized target construction requires an output normalizer."
                    raise RuntimeError(msg)
                target_physical = batch["y"]
                target_normalized = out_normalizer.transform(target_physical)
            views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "normalized": (pred_normalized, target_normalized),
            }
            if requires_physical:
                if out_normalizer is None:
                    msg = "Physical evaluation prediction construction requires an output normalizer."
                    raise RuntimeError(msg)
                pred_physical = out_normalizer.inverse_transform(pred_normalized)
                if target_physical is None:
                    msg = "Physical evaluation target construction requires a fitted data processor."
                    raise RuntimeError(msg)
                views["physical"] = (pred_physical, target_physical)

            for metric in eval_metrics.values():
                space = str(metric.space)
                try:
                    pred_view, target_view = views[space]
                except KeyError as error:
                    msg = f"Metric {metric.id!r} requested unavailable tensor space {space!r}."
                    raise RuntimeError(msg) from error
                metric.update(
                    pred_view,
                    target_view,
                    space=space,
                    batch_index=batch_index,
                )

    return {metric_id: metric.compute() for metric_id, metric in eval_metrics.items()}


def _eval_adapter_one_epoch(
    model: nn.Module,
    eval_loader: Iterable[Any],
    eval_metrics: dict[str, Any],
    device: torch.device,
    adapter: Any,
) -> dict[str, float]:
    """Accumulate adapter-provided BLCHW views through resolved metric objects."""
    model.eval()
    guardrails = {metric_id: copy.deepcopy(metric) for metric_id, metric in eval_metrics.items()}
    for metric_id, metric in eval_metrics.items():
        if getattr(metric, "id", metric_id) != metric_id:
            msg = f"Evaluation metric key {metric_id!r} does not match its resolved id."
            raise ValueError(msg)
        metric.reset()
        guardrails[metric_id].reset()
    w_abs = w_sq = 0.0
    w_count = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(eval_loader):
            views = adapter.evaluation_step(model, adapter.prepare_batch(raw_batch, device=device, training=False))
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
            normalized = (views.normalized_prediction.flatten(0, 1), views.normalized_target.flatten(0, 1))
            physical = (views.physical_prediction.flatten(0, 1), views.physical_target.flatten(0, 1))
            for metric_id, metric in eval_metrics.items():
                pred, target = normalized if metric.space == "normalized" else physical if metric.space == "physical" else (None, None)
                if pred is None:
                    msg = f"Metric {metric.id!r} requested unavailable space."
                    raise RuntimeError(msg)
                kwargs: dict[str, Any] = {"space": metric.space, "batch_index": batch_index}
                mask = views.valid_mask
                if mask is not None and "mask" in inspect.signature(metric.update).parameters:
                    kwargs["mask"] = mask.flatten(0, 1)
                metric.update(pred, target, **kwargs)
                one_step = (
                    views.one_step_normalized_prediction.flatten(0, 1)
                    if metric.space == "normalized"
                    else views.one_step_physical_prediction.flatten(0, 1)
                )
                guardrails[metric_id].update(one_step, target, **kwargs)
            if views.f_surf is not None:
                fraction = (
                    views.f_surf.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, views.physical_prediction.shape[-2], views.physical_prediction.shape[-1])
                )
                predicted_w = metric_impl.compute_grain_moisture(views.physical_prediction[:, :, 2], views.physical_prediction[:, :, 3], fraction)
                target_w = metric_impl.compute_grain_moisture(views.physical_target[:, :, 2], views.physical_target[:, :, 3], fraction)
                # float64 like the official metrics, so finite predictions cannot overflow the telemetry.
                difference = predicted_w.double() - target_w.double()
                moisture_mask = views.valid_mask
                if moisture_mask is not None:
                    try:
                        expanded_mask = torch.broadcast_to(moisture_mask, views.physical_prediction.shape).select(2, 0)
                    except RuntimeError as error:
                        msg_0 = "Transient valid_mask must broadcast to physical prediction evidence."
                        raise ValueError(msg_0) from error
                    if not bool(expanded_mask.any().item()):
                        msg_0 = "Transient valid_mask has no valid physical moisture telemetry elements."
                        raise ValueError(msg_0)
                    selected_difference = difference[expanded_mask]
                else:
                    selected_difference = difference
                w_abs += float(selected_difference.abs().sum().item())
                w_sq += float(selected_difference.square().sum().item())
                w_count += selected_difference.numel()
    result = {metric_id: float(metric.compute()) for metric_id, metric in eval_metrics.items()}
    for metric_id, metric in eval_metrics.items():
        components = getattr(metric, "components", None)
        if callable(components):
            component_values = components()
            if not isinstance(component_values, Mapping):
                msg_0 = "Metric components must return one mapping."
                raise TypeError(msg_0)
            for name, value in component_values.items():
                result[f"{metric_id}/component/{name}"] = float(value)
    result.update({f"guardrail/one_step/{metric_id}": float(metric.compute()) for metric_id, metric in guardrails.items()})
    if w_count:
        result.update({"physical/w_gr_mae": w_abs / w_count, "physical/w_gr_rmse": math.sqrt(w_sq / w_count)})
    return result


def evaluate_selected_checkpoint(
    *,
    config: Mapping[str, Any],
    model: nn.Module,
    train_loss: nn.Module,
    eval_loader: DataLoader,
    ood_loaders: Mapping[str, DataLoader | None] | None = None,
    eval_metrics: dict[str, Any],
    device: torch.device,
    data_processor: Any | None,
    checkpoint_identity: Mapping[str, Any],
    best_checkpoint_path: Path | str,
    scheduler_expected: bool,
    max_physics_cases: int,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Load and evaluate the authoritative best checkpoint on one shared science state."""
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
    id_metrics = eval_one_epoch(model, eval_loader, eval_metrics, device, data_processor, *extra)
    diagnostic_metrics = {
        role: eval_one_epoch(model, loader, eval_metrics, device, data_processor, *extra)
        for role, loader in _configured_ood_loaders(ood_loaders).items()
    }
    objective_id = str(cast("Mapping[str, Any]", config["evaluation"])["objective"]["id"])
    if objective_id not in id_metrics or any(objective_id not in metrics for metrics in diagnostic_metrics.values()):
        msg = f"Selected checkpoint evaluation did not produce objective {objective_id!r} for its configured loaders."
        raise KeyError(msg)

    selected: dict[str, float] = {f"selected/id/{name}": float(value) for name, value in id_metrics.items()}
    for role, metrics in diagnostic_metrics.items():
        selected.update({f"selected/{role}/{name}": float(value) for name, value in metrics.items()})
        selected[f"selected/{role}/objective_difference"] = float(metrics[objective_id]) - float(id_metrics[objective_id])
    if adapter is None:
        physics = evaluate_physics_monitor(
            model,
            eval_loader,
            train_loss,
            device,
            data_processor,
            max_cases=max_physics_cases,
        )
        for history_key, value in physics.items():
            suffix = history_key.removeprefix("physics/id/")
            selected[f"selected/physics/{suffix}"] = float(value)

    physics_config = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["loss"])["physics"])
    telemetry = getattr(train_loss, "telemetry_state", None)
    if bool(physics_config["enabled"]) and callable(telemetry):
        weights = cast("Mapping[str, float]", telemetry(epoch=selected_epoch - 1))
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
    ood_loaders: Mapping[str, DataLoader | None] | None = None,
    data_processor: Any | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    save_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
    epoch_end_callback: EpochEndCallback | None = None,
    epoch_state_callback: EpochStateCallback | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    adapter: Any | None = None,
    max_steps: int | None = None,
    max_train_seconds: float | None = None,
    operational_timing: dict[str, Any] | None = None,
    training_work: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Train through exact completed-epoch checkpoints with a bounded ``last`` cadence.

    ``last_checkpoint.pt`` is published at the end of a completed epoch that (1)
    brought a new best, (2) ends the loop, (3) finds no ``last`` in the run
    directory yet (the first epoch of a run or stage), or (4) comes at least
    ``LAST_CHECKPOINT_INTERVAL_SECONDS`` of monotonic time after this invocation's
    previous ``last`` (or its start). A new best's ``best_checkpoint.pt`` is
    written together with that ``last``: both files are written and flushed to
    temporary siblings first, then renamed back to back (best, then last), so a
    crash or failure before the renames keeps the previous consistent pair; only
    a crash between the two renames leaves best newer than last, which run
    admission refuses to resume and completed finalization rejects. Every normal
    return therefore has a ``last`` for its terminal epoch.
    A crash loses the epochs since the newest ``last`` (one interval plus the
    epoch and write in progress): resume continues from it and trains them again.
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
        before their construction.
    ood_loaders : Mapping[str, torch.utils.data.DataLoader | None] | None, optional
        Saved ``material_ood`` and ``parameter_ood`` memberships; an absent or ``None`` part is not evaluated.
    data_processor : Any | None, optional
        Saved/fitted normalization processor.
    scheduler : ReduceLROnPlateau | None, optional
        Configured objective scheduler.
    save_dir : Path | str | None, optional
        Canonical run directory. Required for lifecycle execution.
    resume_from : Path | str | None, optional
        Exact ``last_checkpoint.pt`` continuation source.
    epoch_end_callback : Callable | None, optional
        Callback invoked after every completed epoch once its evidence is durable:
        a new best is published, adapter history is written, and ``last`` is
        published when a new best, the first epoch, the loop's end, or the interval calls for it
        (``checkpoint/last_published``). ID, OOD, and physics values are present
        only when their shared predicate is due under each phase's independently
        resolved interval.
    epoch_state_callback : Callable | None, optional
        Adapter-only callback invoked after all completed-epoch evidence is known
        and before any ``last_checkpoint.pt`` publication of that epoch. It owns
        durable local history that resume truncates to the restored checkpoint.
    checkpoint_identity : Mapping[str, Any] | None, optional
        Immutable task/config/dataset/split/objective identity.

    adapter : Any | None, optional
        Task-owned batch and continuation-state adapter; mutually exclusive with data_processor where accepted.
    max_steps : int | None, optional
        Bound the adapter run by completed optimizer steps.
    max_train_seconds : float | None, optional
        Bound the adapter run by measured training duration.

    operational_timing : dict[str, Any] | None, optional
        Attempt-owned measurement sink retained even if checkpoint publication fails.
    training_work : dict[str, int] | None, optional
        Attempt-owned sink of the run's cumulative ``training_work`` counts, current at
        every epoch boundary, so the caller can publish them when the run ends with an exception.

    Returns
    -------
    dict[str, Any]
        Completed progress, best/last paths, finite objective, history, and the
        run's cumulative ``training_work`` counts.

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
        epoch_duration_scope=(
            "epoch_start_through_training_evaluation_best_capture_and_physics; excludes_epoch_state_checkpoint_writes_and_observers"
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
    if adapter is not None and data_processor is not None:
        msg_0 = "Adapter training and data_processor are mutually exclusive."
        raise ValueError(msg_0)
    matched_protocol = (
        config.get("task") == "transient_drying"
        and config["training"].get("matched_a1_b") is not None
        and config["training"]["comparison_arm"] in {"b", "a1"}
    )
    if matched_protocol:
        if adapter is None or not isinstance(train_loader, DataLoader):
            msg_0 = "Matched training requires its transient adapter and DataLoader."
            raise TypeError(msg_0)
        train_loader = _matched_train_loader(train_loader)
    matched_sampler = train_loader.batch_sampler if matched_protocol else None
    best_kind = checkpoints.best_payload_kind(config)
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1 or not matched_protocol):
        msg_0 = "max_steps requires a positive matched B/A1 operational cap."
        raise ValueError(msg_0)
    if max_train_seconds is not None and (
        type(max_train_seconds) not in {int, float} or not math.isfinite(max_train_seconds) or max_train_seconds <= 0 or not matched_protocol
    ):
        msg_0 = "max_train_seconds requires a finite positive matched B/A1 operational cap."
        raise ValueError(msg_0)

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Training requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda" and device.index is None:
        msg = "Training requires an indexed CUDA device resolved by the runtime boundary."
        raise ValueError(msg)
    model = model.to(device)
    if data_processor is not None:
        data_processor.to(device)

    n_epochs = int(config["training"]["epochs"])
    eval_interval = int(config["training"]["evaluation_interval"])
    diagnostic_loaders = _configured_ood_loaders(ood_loaders)
    diagnostic_interval = int(config["training"]["ood_evaluation_interval"])
    monitor_settings = config["tracking"]["wandb"]["monitor"]
    physics_monitor_enabled = bool(monitor_settings["enabled"])
    physics_monitor_interval = int(monitor_settings["interval"])
    physics_monitor_max_cases = int(monitor_settings["max_cases"])
    if n_epochs <= 0:
        msg = f"training.epochs must be positive, got: {n_epochs}"
        raise ValueError(msg)
    for label, interval in (
        ("training.evaluation_interval", eval_interval),
        ("training.ood_evaluation_interval", diagnostic_interval),
        ("tracking.wandb.monitor.interval", physics_monitor_interval),
    ):
        if interval <= 0:
            msg = f"{label} must be positive, got: {interval}"
            raise ValueError(msg)
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
    training_work = {} if training_work is None else training_work
    training_work.clear()
    training_work.update(dict.fromkeys(sorted(checkpoints.TRAINING_WORK_KEYS), 0))
    start_epoch_index = 0
    resumed_partial = False
    finalize_only = False

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
        )
        restored = checkpoints.restore_checkpoint(
            last_payload,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=train_loss,
            train_loader=train_loader,
            adapter=adapter,
            adapter_expected=adapter is not None,
        )
        start_epoch_index = int(restored["next_epoch"]) - 1
        global_step = int(restored["global_step"])
        training_work.update(restored["training_work"])
        best_metric = restored["best_metric"]
        best_epoch = restored["best_epoch"]
        objective_history = list(restored["objective_history"])
        resumed_partial = bool(restored["batch_progress"] is not None and not restored["batch_progress"]["complete"])
        # A completed stage budget or the final epoch boundary already published the terminal best and
        # last checkpoints; resuming them only finalizes, because no further optimizer group is admitted.
        finalize_only = checkpoints.records_completed_budget(last_payload) or start_epoch_index == n_epochs
        if not finalize_only and max_steps is not None and global_step >= max_steps:
            msg_0 = "max_steps must increase beyond committed optimizer steps on resume."
            raise ValueError(msg_0)
        if (
            not finalize_only
            and max_train_seconds is not None
            and adapter is not None
            and adapter.budget_state()["train_seconds"] >= max_train_seconds
        ):
            msg_0 = "max_train_seconds must increase beyond committed train seconds on resume."
            raise ValueError(msg_0)
        if not finalize_only and start_epoch_index >= n_epochs:
            msg = (
                f"Resume checkpoint already completed epoch {restored['completed_epoch']}, "
                f"but runtime training.epochs is {n_epochs}. Increase the terminal duration deliberately."
            )
            raise ValueError(msg)

    best_path = common.paths.resolve_best_checkpoint_file(run_dir)
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    # Admission gives the run one writer, so staged files of an earlier crashed publication are ours to drop.
    checkpoints.discard_staged_checkpoints(best_path, last_path)
    trial_run = experiments.run.is_trial_config(config)
    session_started = time.perf_counter()
    last_published_at = time.monotonic()

    # The terminal epoch of a finalize-only resume is the one its last checkpoint recorded.
    completed_epoch = int(last_payload["completed_epoch"]) if finalize_only else start_epoch_index
    for epoch_index in range(start_epoch_index, start_epoch_index if finalize_only else n_epochs):
        completed_epoch = epoch_index + 1
        epoch_started = time.perf_counter()
        parameter_groups = optimizer.param_groups
        if not parameter_groups:
            msg = "Optimizer has no parameter groups before a training epoch."
            raise RuntimeError(msg)
        epoch_learning_rate = float(parameter_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        set_epoch = getattr(train_loss, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch_index)

        if isinstance(matched_sampler, MatchedPermutationSampler):
            matched_sampler.begin_epoch(epoch_index)
        if adapter is not None and not (resumed_partial and epoch_index == start_epoch_index):
            adapter.begin_epoch(epoch_index=epoch_index, total_epochs=n_epochs)
        resumed_partial = False
        with _training_phase("training_epoch", completed_epoch):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                train_loss,
                device,
                data_processor,
                adapter=adapter,
                max_optimizer_steps=(max_steps - global_step if max_steps is not None else None),
                max_train_seconds=max_train_seconds,
            )
        optimizer_steps = int(train_metrics.pop("optimizer_steps"))
        global_step += optimizer_steps
        training_work["optimizer_steps"] += int(train_metrics.pop("committed_optimizer_groups"))
        training_work["gradient_predictions"] += int(train_metrics.pop("gradient_predictions"))

        evaluated: dict[str, float] = {}
        id_metrics: dict[str, float] = {}
        current_metric: float | None = None
        best_payload: dict[str, Any] | None = None
        operational_partial = bool(
            isinstance(matched_sampler, MatchedPermutationSampler)
            and matched_sampler.cursor < matched_sampler.size
            and adapter is not None
            and not adapter.should_stop()
        )
        epoch_event = training_events.is_completed_epoch_event(
            completed_epoch,
            interval=eval_interval,
            target_epoch=n_epochs,
        )
        # Measured-budget arms validate only at epoch ends that crossed a new budget point
        # (or at their epoch cap), so B and A1 get the same number of selection chances.
        budget_gated = adapter is not None and adapter.evaluates_at_budget_points()
        should_evaluate_id = not operational_partial and (
            (adapter is not None and bool(adapter.should_stop()))
            or (epoch_event and (not budget_gated or completed_epoch == n_epochs or (adapter is not None and adapter.budget_evaluation_due())))
        )
        should_evaluate_diagnostic = (
            not operational_partial
            and bool(diagnostic_loaders)
            and training_events.is_completed_epoch_event(
                completed_epoch,
                interval=diagnostic_interval,
                target_epoch=n_epochs,
            )
        )
        should_monitor_physics = (
            not operational_partial
            and physics_monitor_enabled
            and training_events.is_completed_epoch_event(
                completed_epoch,
                interval=physics_monitor_interval,
                target_epoch=n_epochs,
            )
        )

        if should_evaluate_id:
            _synchronize_device(device)
            id_started = time.perf_counter()
            with _training_phase("id_evaluation", completed_epoch):
                if adapter is None:
                    id_metrics = eval_one_epoch(model, eval_loader, eval_metrics, device, data_processor)
                else:
                    id_metrics = eval_one_epoch(model, eval_loader, eval_metrics, device, data_processor, adapter)
            _synchronize_device(device)
            evaluated["system/id_evaluation_duration_seconds"] = time.perf_counter() - id_started
            if adapter is not None:
                adapter.record_validation_work(evaluated["system/id_evaluation_duration_seconds"])
            evaluated["system/id_evaluation_case_count"] = float(_loader_case_count(eval_loader))
            current_metric = id_metrics.get(objective_id)
            if current_metric is None:
                msg = f"Configured evaluation objective {objective_id!r} was not produced by ID evaluation metrics."
                raise KeyError(msg)
            current_metric = float(current_metric)
            if not math.isfinite(current_metric):
                msg = f"ID evaluation objective {objective_id!r} is non-finite at epoch {completed_epoch}: {current_metric}."
                raise FloatingPointError(msg)
            evaluated.update({f"id/{name}": value for name, value in id_metrics.items()})
            objective_history.append(
                {
                    "epoch": completed_epoch,
                    "objective_id": objective_id,
                    "value": current_metric,
                }
            )

            if adapter is not None:
                # Adapter work stops at the first crossing optimizer group, so this
                # evaluation cannot observe additional post-budget parameter updates.
                adapter.record_within_budget_evaluation(current_metric, epoch_index=epoch_index)
                adapter.record_budget_evaluation()
            if scheduler is not None:
                old_learning_rate = float(optimizer.param_groups[0]["lr"])
                with _training_phase("scheduler_update", completed_epoch):
                    scheduler.step(current_metric)
                new_learning_rate = float(optimizer.param_groups[0]["lr"])
                if new_learning_rate != old_learning_rate:
                    evaluated["optimization/scheduler_old_learning_rate"] = old_learning_rate
                    evaluated["optimization/scheduler_new_learning_rate"] = new_learning_rate

            previous_best = best_metric
            is_better = best_metric is None or (current_metric < best_metric if objective_direction == "minimize" else current_metric > best_metric)
            evaluated["checkpoint/new_best"] = float(is_better)
            if previous_best is not None:
                evaluated["checkpoint/previous_best_objective"] = float(previous_best)
            if is_better:
                best_metric = current_metric
                best_epoch = completed_epoch
                with _checkpoint_publication_timer(timing, device):
                    best_payload = checkpoints.make_checkpoint(
                        role="best",
                        payload_kind=best_kind,
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
                        adapter=adapter,
                    )

        for diagnostic_role, diagnostic_loader in (diagnostic_loaders if should_evaluate_diagnostic else {}).items():
            _synchronize_device(device)
            diagnostic_started = time.perf_counter()
            with _training_phase(f"{diagnostic_role}_evaluation", completed_epoch):
                if adapter is None:
                    diagnostic_metrics = eval_one_epoch(model, diagnostic_loader, eval_metrics, device, data_processor)
                else:
                    diagnostic_metrics = eval_one_epoch(model, diagnostic_loader, eval_metrics, device, data_processor, adapter)
            _synchronize_device(device)
            evaluated[f"system/{diagnostic_role}_evaluation_duration_seconds"] = time.perf_counter() - diagnostic_started
            evaluated[f"system/{diagnostic_role}_evaluation_case_count"] = float(_loader_case_count(diagnostic_loader))
            evaluated.update({f"{diagnostic_role}/{name}": value for name, value in diagnostic_metrics.items()})
            diagnostic_objective = diagnostic_metrics.get(objective_id)
            if diagnostic_objective is not None and current_metric is not None:
                evaluated[f"{diagnostic_role}/objective_difference"] = float(diagnostic_objective) - current_metric

        if should_monitor_physics:
            _synchronize_device(device)
            physics_started = time.perf_counter()
            with _training_phase("physics_monitor", completed_epoch):
                physics_metrics = evaluate_physics_monitor(
                    model,
                    eval_loader,
                    train_loss,
                    device,
                    data_processor,
                    max_cases=physics_monitor_max_cases,
                )
            _synchronize_device(device)
            evaluated.update(physics_metrics)
            evaluated["system/physics_monitor_duration_seconds"] = time.perf_counter() - physics_started
            evaluated["system/physics_monitor_case_count"] = float(min(physics_monitor_max_cases, _loader_case_count(eval_loader)))

        train_metrics["global_step"] = float(global_step)
        train_metrics["optimization/learning_rate"] = epoch_learning_rate
        epoch_duration = time.perf_counter() - epoch_started
        if not math.isfinite(epoch_duration) or epoch_duration <= 0.0:
            msg = f"Completed epoch {completed_epoch} has invalid duration {epoch_duration}."
            raise RuntimeError(msg)
        train_metrics["system/epoch_duration_seconds"] = epoch_duration
        elapsed = time.perf_counter() - session_started
        train_metrics["system/session_elapsed_seconds"] = elapsed
        epochs_this_session = completed_epoch - start_epoch_index
        remaining_epochs = n_epochs - completed_epoch
        if epochs_this_session > 0:
            train_metrics["system/estimated_remaining_seconds"] = elapsed / epochs_this_session * remaining_epochs
        if device.type == "cuda":
            train_metrics["system/cuda_peak_memory_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))

        if adapter is not None:
            train_metrics.update({f"transient/{key}": float(value) for key, value in adapter.telemetry_state().items() if value is not None})
            train_metrics["transient/budget_complete"] = float(bool(adapter.should_stop()))
        terminal_epoch_metrics = {**train_metrics, **evaluated}
        if epoch_state_callback is not None and not operational_partial:
            if adapter is None:
                msg_0 = "epoch_state_callback is reserved for task-owned adapter training."
                raise ValueError(msg_0)
            with _training_phase("epoch_state", completed_epoch):
                epoch_state_callback(completed_epoch, copy.deepcopy(terminal_epoch_metrics))

        # One predicate both exits the loop and forces the terminal last, so the two cannot drift.
        ends_loop = (
            completed_epoch == n_epochs
            or operational_partial
            or (adapter is not None and (bool(adapter.should_stop()) or bool(train_metrics.get("transient/train/operational_cap_reached"))))
        )
        # A new best is published only together with its last, so a published best is never newer than the
        # newest last; the first epoch of a run or stage gets one, so an early crash leaves a resumable run.
        publish_last = not trial_run and (
            ends_loop
            or best_payload is not None
            or not last_path.exists()
            or time.monotonic() - last_published_at >= LAST_CHECKPOINT_INTERVAL_SECONDS
        )
        if trial_run and best_payload is not None:
            # A trial never resumes: its new best is the only file, published by one rename.
            with _checkpoint_publication_timer(timing, device), _training_phase("best_checkpoint", completed_epoch):
                checkpoints.save_checkpoints((best_payload, best_path))
        if publish_last:
            with _checkpoint_publication_timer(timing, device):
                last_payload = checkpoints.make_checkpoint(
                    role="last",
                    payload_kind="full",
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
                    adapter=adapter,
                    batch_progress=(
                        {
                            "epoch": completed_epoch,
                            "cursor": matched_sampler.cursor,
                            "size": matched_sampler.size,
                            "complete": matched_sampler.cursor == matched_sampler.size,
                        }
                        if isinstance(matched_sampler, MatchedPermutationSampler)
                        else None
                    ),
                )
                with _training_phase("last_checkpoint", completed_epoch):
                    # Both files are staged before either rename, so a failure keeps the previous pair.
                    best_publication = ((best_payload, best_path),) if best_payload is not None else ()
                    checkpoints.save_checkpoints(*best_publication, (last_payload, last_path))
            last_published_at = time.monotonic()
        terminal_epoch_metrics["checkpoint/last_published"] = float(publish_last)
        if epoch_end_callback is not None and not operational_partial:
            with _training_phase("epoch_observers", completed_epoch):
                epoch_end_callback(completed_epoch, terminal_epoch_metrics)
        if ends_loop:
            break

    terminal_epoch = completed_epoch
    budget_unmet = bool(
        matched_protocol and adapter is not None and not adapter.should_stop() and config["training"]["comparison_arm"] in {"b", "a1"}
    )
    if (best_metric is None and not budget_unmet) or (best_epoch is None and not budget_unmet):
        msg = "Training produced no finite objective and cannot be marked completed."
        raise RuntimeError(msg)
    if adapter is not None and not budget_unmet:
        adapter.validate_terminal_state(best_metric=best_metric, best_epoch=best_epoch)
    if budget_unmet:
        if adapter is None:
            msg_0 = "Matched budget_unmet requires its transient adapter."
            raise AssertionError(msg_0)
        budget_state = adapter.budget_state()
        return {
            "completed_epoch": terminal_epoch,
            "next_epoch": last_payload["next_epoch"],
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "objective": dict(objective),
            "objective_history": objective_history,
            "terminal_epoch": terminal_epoch,
            "terminal_metrics": {key: value for key, value in terminal_epoch_metrics.items() if key.startswith("train/")},
            "checkpoint_path": str(best_path) if best_path.exists() else None,
            "best_checkpoint_path": str(best_path) if best_path.exists() else None,
            "last_checkpoint_path": str(last_path),
            "status": "budget_unmet",
            "operational_timing": copy.deepcopy(timing),
            "training_work": dict(training_work),
            "train_seconds": budget_state.get("train_seconds"),
            "target_train_seconds": budget_state.get("planned_train_seconds"),
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
        "status": "completed",
        "operational_timing": copy.deepcopy(timing),
        "training_work": dict(training_work),
        **(
            {"train_seconds": adapter.budget_state()["train_seconds"], "target_train_seconds": adapter.budget_state()["planned_train_seconds"]}
            if matched_protocol and adapter is not None
            else {}
        ),
    }
