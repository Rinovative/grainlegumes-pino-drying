"""Adapt transient drying physical batches to generic training-step contracts.

Responsibilities:
  - Tensorize physical transient evidence on a caller-selected device
  - Run teacher-forced Stage A and self-fed Stage B differentiable rollouts
  - Train the RNO on consecutive trajectory windows with a carried, detached hidden state
  - Select continuous Stage-B windows without padding or detached feedback
  - Persist curriculum and stage-budget continuation evidence

Design principles:
  - Tensorizer and scaling artifacts remain the channel and reconstruction owners
  - Every stage is validated by the same autonomous rollout of whole trajectories, so A0, A1, and B select checkpoints alike
  - Controller work advances only after generic-loop optimizer confirmation

This module does NOT:
  - Perform optimizer steps, checkpoint publication, or metric aggregation
  - Invent boundary, startup, static, scalar, or time conditioning evidence
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn

from src import datasets
from src.learning.training import learning_training_adapter as training_adapter

from .learning_transient_contracts import TransientTensorizerSpec
from .learning_transient_curriculum import (
    BudgetControl,
    ClockKind,
    ComparisonArm,
    MatchedComputeController,
    RolloutCurriculum,
    RolloutCurriculumState,
    TeacherHandoffIdentity,
    TrainingStage,
    TransientTrainingSpec,
)
from .learning_transient_rollout import (
    TransientRollout,
    detach_hidden,
    select_hidden_per_sample,
    self_fed_rollout,
    teacher_forced_rollout,
)
from .learning_transient_tensorizer import TransientBatch, TransientTensorizer


class TransientTrainingAdapter:
    """Implement task-owned Stage-A and Stage-B training adaptation."""

    def __init__(self, *, tensorizer: TransientTensorizer, spec: TransientTrainingSpec) -> None:
        """Bind immutable tensorization and stage semantics."""
        if not isinstance(tensorizer, TransientTensorizer):
            msg = "tensorizer must be a TransientTensorizer."
            raise TypeError(msg)
        if not isinstance(spec, TransientTrainingSpec):
            msg = "spec must be a TransientTrainingSpec."
            raise TypeError(msg)
        self.tensorizer = tensorizer
        self.spec = spec
        self.curriculum_state = RolloutCurriculumState.create(spec.curriculum, seed=spec.curriculum_seed)
        self._epoch_index = 0
        self._optimizer_events = 0
        self._last_origin_min: int | None = None
        self._last_origin_max: int | None = None

    @property
    def gradient_accumulation_steps(self) -> int:
        """Return the configured microbatch accumulation factor."""
        return self.spec.gradient_accumulation_steps

    def begin_epoch(self, *, epoch_index: int, total_epochs: int) -> None:
        """Bind one epoch index without deriving curriculum progress from epochs."""
        if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
            msg = "epoch_index must be a non-negative integer."
            raise ValueError(msg)
        if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
            msg = "total_epochs must be a positive integer."
            raise ValueError(msg)
        self.spec.controller.begin_epoch(epoch_index=epoch_index, total_epochs=total_epochs)
        self._epoch_index = epoch_index
        self._optimizer_events = 0

    def prepare_batch(self, raw_batch: Any, *, device: torch.device, training: bool) -> TransientBatch:
        """Recursively move physical evidence and tensorize it on the scaler device."""
        del training
        if not isinstance(device, torch.device):
            msg = "device must be one concrete torch.device."
            raise TypeError(msg)
        if device != self.tensorizer.scaling.device:
            msg = "Transient adapter device must match its scaling artifact device."
            raise ValueError(msg)
        moved = _move_to_device(raw_batch, device)
        if not isinstance(moved, Mapping):
            msg = "Transient raw batch must remain a mapping after device transfer."
            raise TypeError(msg)
        return self.tensorizer.tensorize(moved)

    def training_steps(self, model: nn.Module, batch: TransientBatch, loss: nn.Module) -> Iterator[training_adapter.AdapterTrainingStep]:
        """
        Yield the differentiable Stage-A or Stage-B microbatches of one loader batch.

        Stateless models train on the batch as one microbatch. The RNO receives
        whole trajectories and trains on their consecutive windows in temporal
        order (truncated backpropagation through time): each window starts from
        the hidden state the preceding window reached at this window's start
        (earlier than its end for the overlapping last window), detached so
        gradients stop at the window boundary. The state is carried from the
        reference trajectory, so Stage-B self-feeding never leaks into it.
        """
        window = self.spec.trajectory_window
        if window is None:
            yield self._window_step(model, batch, loss, hidden=None, carry_to=None, consumed_items=batch.batch_size)[0]
            return
        starts = _window_starts(batch.rollout_length, window)
        hidden: object | None = None
        for position, start in enumerate(starts):
            last = position == len(starts) - 1
            step, hidden = self._window_step(
                model,
                _slice_window(batch, start=start, length=window),
                loss,
                hidden=hidden,
                carry_to=None if last else starts[position + 1] - start,
                consumed_items=batch.batch_size if last else 0,
            )
            yield step

    def _window_step(
        self,
        model: nn.Module,
        window: TransientBatch,
        loss: nn.Module,
        *,
        hidden: object | None,
        carry_to: int | None,
        consumed_items: int,
    ) -> tuple[training_adapter.AdapterTrainingStep, object | None]:
        """Return one window's loss and the detached RNO state at the next window start."""
        kind = self.spec.model_kind
        carried: object | None = None
        warm_transitions = 0
        if self.spec.stage == "stage_a_teacher_forcing":
            target_batch = window
            rollout = teacher_forced_rollout(
                model, window, self.tensorizer, model_kind=kind, initial_hidden=hidden, capture=() if carry_to is None else (carry_to,)
            )
            if carry_to is not None:
                carried = detach_hidden(rollout.captured_hidden[carry_to])
        elif self.spec.stage == "stage_b_self_fed":
            horizon, origins = self.curriculum_state.select(
                progress=self.spec.controller.progress,
                available_length=window.rollout_length,
                batch_size=window.batch_size,
            )
            origins = origins.to(device=window.state.device)
            self._last_origin_min = int(origins.min().item())
            self._last_origin_max = int(origins.max().item())
            target_batch = _select_window(window, tensorizer=self.tensorizer, horizon=horizon, origins=origins)
            snapshots: Mapping[int, object | None] = {}
            if self.spec.trajectory_window is not None:
                # Teacher-forced warm-up over the reference window yields the state at every origin and at the next window.
                capture = {int(value) for value in origins.tolist()} | (set() if carry_to is None else {carry_to})
                with torch.no_grad():
                    warm = teacher_forced_rollout(model, window, self.tensorizer, model_kind=kind, initial_hidden=hidden, capture=capture)
                snapshots = warm.captured_hidden
                carried = None if carry_to is None else snapshots[carry_to]
                warm_transitions = window.batch_size * window.rollout_length
            rollout = (
                self_fed_rollout(model, target_batch, self.tensorizer, model_kind=kind)
                if self.spec.trajectory_window is None
                else _self_fed_from_snapshots(model, target_batch, self.tensorizer, model_kind=kind, snapshots=snapshots, origins=origins)
            )
        else:
            msg = "Transient training specification has an unsupported stage."
            raise RuntimeError(msg)
        components = _loss_components(
            loss, rollout.scaled_delta, target_batch.scaled_target, rollout.normalized_prediction, rollout.normalized_target
        )
        total = components["total"]
        if not bool(torch.isfinite(total).all().item()):
            msg = "Transient training loss is non-finite."
            raise FloatingPointError(msg)
        transitions = target_batch.batch_size * target_batch.rollout_length
        step = training_adapter.AdapterTrainingStep(
            loss=total,
            components=components,
            sample_count=target_batch.batch_size,
            processed_target_transitions=transitions,
            forward_transitions=transitions + warm_transitions,
            consumed_items=consumed_items,
        )
        return step, carried

    def evaluation_step(self, model: nn.Module, batch: TransientBatch) -> training_adapter.AdapterEvaluationViews:
        """
        Return whole-trajectory views for every stage's checkpoint selection.

        Every model starts from the reference state at the trajectory start and
        rolls out the complete trajectory from its own predictions; the RNO
        carries only its own hidden state. The teacher-forced one-step view
        predicts every transition from its reference state, the RNO with the
        state built over the reference transitions before it.
        """
        kind = self.spec.model_kind
        with torch.no_grad():
            rollout = self_fed_rollout(model, batch, self.tensorizer, model_kind=kind)
            one_step_normalized, one_step_physical = _teacher_forced_in_chunks(model, batch, self.tensorizer, model_kind=kind)
        length = batch.rollout_length
        return training_adapter.AdapterEvaluationViews(
            normalized_prediction=rollout.normalized_prediction,
            normalized_target=rollout.normalized_target,
            physical_prediction=rollout.physical_prediction,
            physical_target=rollout.physical_target,
            one_step_normalized_prediction=one_step_normalized,
            one_step_physical_prediction=one_step_physical,
            sample_count=batch.batch_size,
            processed_target_transitions=batch.batch_size * length,
            f_surf=batch.scalars[:, 2].view(batch.batch_size, 1).expand(-1, length),
        )

    def record_optimizer_work(self, work: training_adapter.OptimizerWork) -> None:
        """Forward only reported completed optimizer work to the controller."""
        if not isinstance(work, training_adapter.OptimizerWork):
            msg = "work must be an OptimizerWork."
            raise TypeError(msg)
        self.spec.controller.record_completed_work(
            successful=work.successful,
            optimizer_device_seconds=work.optimizer_device_seconds,
            microbatches=work.microbatches,
            processed_target_transitions=work.processed_target_transitions,
            forward_transitions=work.forward_transitions,
            wall_seconds=work.wall_seconds,
            epoch_index=self._epoch_index,
            microbatch_index=self._optimizer_events,
            peak_cuda_memory_bytes=work.peak_cuda_memory_bytes,
            train_seconds=work.train_seconds,
            batch_cursor=work.batch_cursor,
        )
        self._optimizer_events += 1

    def record_validation_work(self, seconds: float) -> None:
        """Record completed validation time excluded from matched-compute progress."""
        self.spec.controller.record_validation_work(seconds)

    def budget_state(self) -> dict[str, Any]:
        """Return strict stage-budget evidence for loop stop decisions."""
        return self.spec.controller.state_dict()

    def should_stop(self) -> bool:
        """Return the controller's sticky completion state."""
        return self.spec.controller.budget_complete

    def evaluates_at_budget_points(self) -> bool:
        """Return whether selection waits for equally spaced points of a measured budget."""
        return self.spec.controller.evaluation_points is not None

    def budget_evaluation_due(self) -> bool:
        """Return whether a new budget point was crossed since the last selection evaluation."""
        return self.spec.controller.budget_evaluation_due

    def record_budget_evaluation(self) -> None:
        """Mark the budget points reached so far as evaluated."""
        self.spec.controller.record_budget_evaluation()

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one eligible finite selection metric."""
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            msg = "Within-budget evaluation metric must be finite."
            raise ValueError(msg)
        self.spec.controller.record_within_budget_evaluation(float(metric), epoch_index=epoch_index)

    def validate_terminal_state(self, *, best_metric: float | None, best_epoch: int | None) -> None:
        """Require a completed budget and exact budget-boundary checkpoint selection."""
        controller = self.spec.controller
        if not controller.budget_complete:
            msg = "Transient training ended before its configured budget completed."
            raise RuntimeError(msg)
        if controller.best_within_budget_metric is None or controller.best_within_budget_epoch is None:
            msg = "Transient training has no selected metric at or before its budget boundary."
            raise RuntimeError(msg)
        if best_metric != controller.best_within_budget_metric or best_epoch != controller.best_within_budget_epoch + 1:
            msg = "Selected transient checkpoint disagrees with stage-budget evidence."
            raise RuntimeError(msg)

    def telemetry_state(self) -> dict[str, float | int | bool | None]:
        """Return compact stage, curriculum, and stage-budget telemetry."""
        controller = self.spec.controller
        return {
            "curriculum_progress": controller.progress,
            "curriculum_active_stage": self.curriculum_state.active_stage,
            "curriculum_max_horizon": self.curriculum_state.max_horizon,
            "curriculum_draw_index": self.curriculum_state.draw_index,
            "selected_rollout_horizon": self.curriculum_state.max_horizon,
            "last_origin_min": getattr(self, "_last_origin_min", None),
            "last_origin_max": getattr(self, "_last_origin_max", None),
            "self_fed_stage": int(self.spec.stage == "stage_b_self_fed"),
            "planned_stage_epochs": controller.planned_stage_epochs,
            "completed_stage_epochs": controller.completed_stage_epochs,
            "post_handoff_optimizer_device_seconds": controller.post_handoff_optimizer_device_seconds,
            "post_handoff_optimizer_steps": controller.post_handoff_optimizer_steps,
            "successful_optimizer_steps": controller.successful_optimizer_steps,
            "processed_target_transitions": controller.processed_target_transitions,
            "forward_transitions": controller.forward_transitions,
            "wall_seconds": controller.wall_seconds,
            "validation_seconds": controller.validation_seconds,
            "peak_cuda_memory_bytes": controller.peak_cuda_memory_bytes,
            "best_within_budget_metric": controller.best_within_budget_metric,
            "best_within_budget_epoch": controller.best_within_budget_epoch,
            "budget_complete": controller.budget_complete,
            "budget_crossing_epoch": controller.crossing_epoch,
            "budget_crossing_microbatch": controller.crossing_microbatch,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return exact adapter continuation state with immutable semantic digest."""
        return {
            "spec_digest": self.spec.digest,
            "controller": self.spec.controller.state_dict(),
            "curriculum": self.curriculum_state.state_dict(),
            "epoch_index": self._epoch_index,
            "optimizer_events": self._optimizer_events,
            "last_origin_min": getattr(self, "_last_origin_min", None),
            "last_origin_max": getattr(self, "_last_origin_max", None),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact matching adapter continuation state."""
        required = {"spec_digest", "controller", "curriculum", "epoch_index", "optimizer_events", "last_origin_min", "last_origin_max"}
        if not isinstance(state, dict) or set(state) != required or state["spec_digest"] != self.spec.digest:
            msg = "Saved transient adapter state conflicts with the configured semantic specification."
            raise ValueError(msg)
        if not isinstance(state["controller"], Mapping) or not isinstance(state["curriculum"], Mapping):
            message = "Saved transient adapter controller and curriculum state must be mappings."
            raise TypeError(message)
        epoch_index = state["epoch_index"]
        optimizer_events = state["optimizer_events"]
        if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
            message = "Saved epoch_index must be non-negative."
            raise ValueError(message)
        if isinstance(optimizer_events, bool) or not isinstance(optimizer_events, int) or optimizer_events < 0:
            message = "Saved optimizer_events must be non-negative."
            raise ValueError(message)
        origin_values = (state["last_origin_min"], state["last_origin_max"])
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in origin_values):
            msg = "Saved transient origin evidence must be null or non-negative integers."
            raise ValueError(msg)
        if (origin_values[0] is None) != (origin_values[1] is None) or (origin_values[0] is not None and origin_values[0] > origin_values[1]):
            msg = "Saved transient origin evidence is inconsistent."
            raise ValueError(msg)
        controller_clone = _clone_controller(self.spec.controller)
        curriculum_clone = RolloutCurriculumState.create(self.spec.curriculum, seed=self.spec.curriculum_seed)
        curriculum_clone.load_state_dict(self.curriculum_state.state_dict())
        controller_clone.load_state_dict(state["controller"])
        curriculum_clone.load_state_dict(state["curriculum"])
        self.spec.controller.load_state_dict(controller_clone.state_dict())
        self.curriculum_state.load_state_dict(curriculum_clone.state_dict())
        self._epoch_index = epoch_index
        self._optimizer_events = optimizer_events
        self._last_origin_min = state["last_origin_min"]
        self._last_origin_max = state["last_origin_max"]


def _clone_controller(controller: MatchedComputeController) -> MatchedComputeController:
    """Return one independently mutable controller with matching current state."""
    saved = controller.state_dict()
    payload = dict(saved)
    payload.pop("schema_version")
    handoff = payload["teacher_handoff"]
    if handoff is not None:
        payload["teacher_handoff"] = TeacherHandoffIdentity.from_mapping(handoff)
    return MatchedComputeController(**payload)


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensor evidence while preserving metadata values."""
    if isinstance(value, torch.Tensor):
        non_blocking = device.type == "cuda" and value.device.type == "cpu" and value.is_pinned()
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


# Bounds the memory of batched teacher-forced inference over long trajectories.
_TEACHER_FORCED_CHUNK = 32


def _teacher_forced_in_chunks(
    model: nn.Module, batch: TransientBatch, tensorizer: TransientTensorizer, *, model_kind: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized and physical one-step predictions of whole trajectories, carrying the RNO state."""
    hidden: object | None = None
    parts: list[TransientRollout] = []
    for start in range(0, batch.rollout_length, _TEACHER_FORCED_CHUNK):
        length = min(_TEACHER_FORCED_CHUNK, batch.rollout_length - start)
        capture = (length,) if model_kind == "rno" else ()
        part = teacher_forced_rollout(
            model, _slice_window(batch, start=start, length=length), tensorizer, model_kind=model_kind, initial_hidden=hidden, capture=capture
        )
        hidden = part.captured_hidden.get(length)
        parts.append(part)
    return torch.cat([part.normalized_prediction for part in parts], dim=1), torch.cat([part.physical_prediction for part in parts], dim=1)


def _window_starts(transition_count: int, window: int) -> tuple[int, ...]:
    """Return the Dataset-owned starts of consecutive windows covering one trajectory."""
    return datasets.contracts.transient.rollout_window_starts(transition_count, rollout_length=window, window_stride=window, window_offset=0)


def _slice_window(batch: TransientBatch, *, start: int, length: int) -> TransientBatch:
    """Return one teacher-forced window of every trajectory without recomputing its inputs."""
    if start < 0 or length < 1 or start + length > batch.rollout_length:
        msg = "Transient trajectory window must lie inside the batch."
        raise ValueError(msg)
    stop = start + length
    state = batch.state if start == 0 else batch.state + batch.target[:, :start].sum(dim=1)
    sequence_input = batch.sequence_input[:, start:stop]
    return TransientBatch(
        state=state,
        target=batch.target[:, start:stop],
        static=batch.static,
        boundary=batch.boundary[:, start:stop],
        scalars=batch.scalars,
        t_n=batch.t_n[:, start:stop],
        t_n_plus_1=batch.t_n_plus_1[:, start:stop],
        dt=batch.dt[:, start:stop],
        step_input=sequence_input[:, 0],
        sequence_input=sequence_input,
        scaled_target=batch.scaled_target[:, start:stop],
    )


def _index_batch(batch: TransientBatch, rows: torch.Tensor) -> TransientBatch:
    """Return the selected samples of one batch in the given order."""
    return TransientBatch(
        state=batch.state[rows],
        target=batch.target[rows],
        static=batch.static[rows],
        boundary=batch.boundary[rows],
        scalars=batch.scalars[rows],
        t_n=batch.t_n[rows],
        t_n_plus_1=batch.t_n_plus_1[rows],
        dt=batch.dt[rows],
        step_input=batch.step_input[rows],
        sequence_input=batch.sequence_input[rows],
        scaled_target=batch.scaled_target[rows],
    )


def _self_fed_from_snapshots(
    model: nn.Module,
    batch: TransientBatch,
    tensorizer: TransientTensorizer,
    *,
    model_kind: str,
    snapshots: Mapping[int, object | None],
    origins: torch.Tensor,
) -> TransientRollout:
    """
    Roll every sample out from the RNO state captured at its own origin.

    Samples whose origin is still the trajectory start keep the model's own
    initial state, which one batched call cannot mix with captured states, so
    those samples run as a separate group and are restored to batch order.
    """
    initial = torch.tensor([snapshots[int(value)] is None for value in origins.tolist()], device=origins.device)
    if bool(initial.all().item()) or not bool(initial.any().item()):
        return self_fed_rollout(model, batch, tensorizer, model_kind=model_kind, initial_hidden=select_hidden_per_sample(snapshots, origins))
    groups = (torch.nonzero(initial).flatten(), torch.nonzero(~initial).flatten())
    parts = [
        self_fed_rollout(
            model,
            _index_batch(batch, rows),
            tensorizer,
            model_kind=model_kind,
            initial_hidden=select_hidden_per_sample(snapshots, origins, rows),
        )
        for rows in groups
    ]
    order = torch.argsort(torch.cat(groups))
    return TransientRollout(
        scaled_delta=torch.cat([part.scaled_delta for part in parts])[order],
        physical_prediction=torch.cat([part.physical_prediction for part in parts])[order],
        physical_target=torch.cat([part.physical_target for part in parts])[order],
        normalized_prediction=torch.cat([part.normalized_prediction for part in parts])[order],
        normalized_target=torch.cat([part.normalized_target for part in parts])[order],
    )


def _select_window(batch: TransientBatch, *, tensorizer: TransientTensorizer, horizon: int, origins: torch.Tensor) -> TransientBatch:
    """Select one valid unpadded physical window per sample and reassemble inputs."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1 or horizon > batch.rollout_length:
        msg = "Transient window horizon must fit the tensorized rollout length."
        raise ValueError(msg)
    if (
        origins.shape != (batch.batch_size,)
        or origins.dtype != torch.long
        or bool((origins < 0).any().item())
        or bool((origins + horizon > batch.rollout_length).any().item())
    ):
        msg = "Transient window origins must be valid per-sample integer starts."
        raise ValueError(msg)
    indices = origins[:, None] + torch.arange(horizon, device=origins.device)[None, :]
    gather_target = indices[:, :, None, None, None].expand(-1, -1, 4, batch.target.shape[-2], batch.target.shape[-1])
    target = batch.target.gather(1, gather_target)
    boundary = batch.boundary.gather(1, indices[:, :, None].expand(-1, -1, batch.boundary.shape[-1]))
    t_n = batch.t_n.gather(1, indices)
    t_next = batch.t_n_plus_1.gather(1, indices)
    dt = batch.dt.gather(1, indices)
    preceding = torch.cat((torch.zeros_like(batch.target[:, :1]), batch.target), dim=1).cumsum(dim=1)
    initial = batch.state + preceding[torch.arange(batch.batch_size, device=origins.device), origins]
    inputs = torch.stack(
        [
            tensorizer.assemble_step(initial + target[:, :index].sum(dim=1), batch.static, boundary[:, index], batch.scalars, t_n[:, index])
            for index in range(horizon)
        ],
        dim=1,
    )
    return TransientBatch(
        state=initial,
        target=target,
        static=batch.static,
        boundary=boundary,
        scalars=batch.scalars,
        t_n=t_n,
        t_n_plus_1=t_next,
        dt=dt,
        step_input=inputs[:, 0],
        sequence_input=inputs,
        scaled_target=tensorizer.scaling.encode_delta(target),
    )


def _loss_components(
    loss: nn.Module, prediction: torch.Tensor, target: torch.Tensor, predicted_state: torch.Tensor, target_state: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute transient loss components through its semantic public contract."""
    if hasattr(loss, "compute_components"):
        components = loss.compute_components(prediction, y=target, predicted_state=predicted_state, target_state=target_state)
        if not isinstance(components, Mapping) or "total" not in components:
            msg = "Transient loss compute_components must return a mapping containing total."
            raise TypeError(msg)
        return dict(components)
    total = loss(prediction, target)
    return {"total": total, "data": total}


def build_transient_training_adapter(
    config: Mapping[str, Any],
    *,
    scaling: Any,
    device: torch.device,
    teacher_handoff: TeacherHandoffIdentity | None = None,
    matched_train_seconds: float | None = None,
    matched_receipt_sha256: str | None = None,
) -> TransientTrainingAdapter:
    """Build one fresh task adapter from resolved transient runtime semantics."""
    training = config.get("training")
    temporal = config.get("temporal")
    profile = config.get("input_profile")
    if not isinstance(training, Mapping) or not isinstance(temporal, Mapping) or not isinstance(profile, str):
        msg = "Resolved transient adapter config lacks persisted training or tensorizer semantics."
        raise TypeError(msg)
    tensorizer_spec = TransientTensorizerSpec.from_mapping({"input_profile": profile, "temporal_conditioning": temporal.get("temporal_conditioning")})
    stage: TrainingStage = "stage_a_teacher_forcing" if training.get("stage") == "a" else "stage_b_self_fed"
    arm_map: dict[str, ComparisonArm] = {"a0": "A0", "a1": "A1", "b": "B"}
    try:
        arm = arm_map[training["comparison_arm"]]
    except KeyError as error:
        msg = "Transient comparison arm is invalid."
        raise ValueError(msg) from error
    protocol = training.get("matched_a1_b")
    matched_time = protocol is not None and arm in {"B", "A1"}
    clock_kind: ClockKind = "train_seconds" if matched_time else "cuda_device_seconds" if device.type == "cuda" else "optimizer_steps"
    curriculum_data = training["curriculum"]
    if not isinstance(curriculum_data, Mapping):
        msg = "Transient curriculum must be a mapping."
        raise TypeError(msg)
    curriculum = RolloutCurriculum(
        lengths=tuple(curriculum_data["lengths"]),
        milestone_fractions=tuple(curriculum_data["milestone_fractions"]),
    )
    sampling = temporal.get("sampling")
    if not isinstance(sampling, Mapping):
        msg = "Resolved transient adapter config lacks its temporal sampling."
        raise TypeError(msg)
    trajectory_window = int(sampling["rollout_length"]) if sampling.get("mode") == "trajectory" else None
    semantic_config = {key: value for key, value in config.items() if key not in {"paths", "_transient_tensorizer"}}
    if protocol is not None:
        semantic_config = copy.deepcopy(semantic_config)
        semantic_config["training"].pop("epochs")
        for cap in ("a0_max_epochs", "b_max_epochs", "a1_max_epochs"):
            semantic_config["training"]["stage_schedule"].pop(cap)
    try:
        encoded_config = json.dumps(semantic_config, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        msg = "Resolved transient adapter semantics must be canonical JSON."
        raise ValueError(msg) from error
    config_digest = hashlib.sha256(encoded_config).hexdigest()
    budget_control: BudgetControl = "train_seconds" if matched_time else "stage_epochs"
    if protocol is not None and arm == "B":
        matched_train_seconds = float(protocol["train_seconds"])
    if protocol is not None and arm == "A1" and (matched_train_seconds is None or matched_receipt_sha256 is None):
        msg = "A1 requires an admitted exact B time receipt before adapter construction."
        raise ValueError(msg)
    controller = MatchedComputeController(
        arm=arm,
        stage=stage,
        clock_kind=clock_kind,
        config_digest=config_digest,
        budget_control=budget_control,
        planned_stage_epochs=int(training["epochs"]) if budget_control == "stage_epochs" else None,
        teacher_handoff=teacher_handoff,
        planned_train_seconds=matched_train_seconds if matched_time else None,
        matched_receipt_sha256=matched_receipt_sha256 if matched_time else None,
        evaluation_points=int(protocol["evaluation_points"]) if protocol is not None and matched_time else None,
    )
    spec = TransientTrainingSpec(
        stage=stage,
        model_kind=str(config["model"]["kind"]),
        controller=controller,
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        curriculum=curriculum,
        curriculum_seed=int(curriculum_data["seed"]),
        trajectory_window=trajectory_window,
    )
    return TransientTrainingAdapter(tensorizer=TransientTensorizer(tensorizer_spec, scaling.to(device)), spec=spec)
