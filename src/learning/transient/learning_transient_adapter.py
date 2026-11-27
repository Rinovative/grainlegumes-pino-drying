"""Adapt transient drying physical batches to generic training-step contracts.

Responsibilities:
  - Tensorize physical transient evidence on a caller-selected device
  - Compute every unit's loss share from teacher-forced Stage A and self-fed Stage B differentiable rollouts
  - Fix each update group's windows and Stage-B horizon and origin draws before its microbatches load
  - Train the RNO on whole trajectories window by window with a carried, detached hidden state
  - Select continuous Stage-B windows without padding or detached feedback
  - Persist compute-accounting and, in Stage B, curriculum continuation evidence; Stage A has no curriculum
  - Validate with per-case stops before non-finite feedback and count finite physical-bound violations per channel

Design principles:
  - Tensorizer and scaling artifacts remain the channel and reconstruction owners
  - A unit's loss averages its supervised transitions, in Stage B each window's and then its windows equally;
    the generic loop owns weights, grouping and steps
  - Every stage is validated by the same autonomous rollout of whole trajectories, so A0, A1, and B select checkpoints alike
  - Controller work advances only after generic-loop optimizer confirmation

This module does NOT:
  - Perform optimizer steps, checkpoint publication, or metric aggregation
  - Invent boundary, startup, static, scalar, or time conditioning evidence
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from src import datasets
from src.domain.tasks import domain_task_transient_drying as drying
from src.learning.training import learning_training_adapter as training_adapter
from src.learning.training import learning_training_checkpoint as training_checkpoint

from .learning_transient_contracts import TransientTensorizerSpec
from .learning_transient_curriculum import (
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
    autonomous_rollout,
    detach_hidden,
    encode_finite_cases,
    select_hidden_per_sample,
    self_fed_rollout,
    teacher_forced_rollout,
)
from .learning_transient_tensorizer import TransientBatch, TransientTensorizer

if TYPE_CHECKING:
    from src.datasets.runtime.dataset_runtime_units import UnitGroup


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
        # Only Stage B draws rollout horizons, so only Stage B has curriculum state.
        self.curriculum_state = None if spec.curriculum is None else RolloutCurriculumState.create(spec.curriculum, seed=_seed(spec))
        self._last_origin_min: int | None = None
        self._last_origin_max: int | None = None
        # Windows and Stage-B draws of the current update group, by group position; never persisted,
        # because checkpoints fall on group boundaries.
        self._group_windows: tuple[tuple[tuple[int, int], ...], ...] = ()
        self._group_draws: tuple[tuple[tuple[int, int], ...], ...] = ()

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

    def _windows(self, transition_count: int) -> tuple[tuple[int, int], ...]:
        """
        Return the ``(start, length)`` windows one unit is trained on.

        A stateless unit (a transition or a window item) is one window. The RNO cuts
        its trajectory into consecutive windows with a shorter rest window in Stage A,
        so every transition counts once, and into end-aligned windows in Stage B, so
        every curriculum horizon fits its window.
        """
        window = self.spec.trajectory_window
        if window is None:
            return ((0, transition_count),)
        if self.spec.stage == "stage_a_teacher_forcing":
            return datasets.contracts.transient.consecutive_windows(transition_count, window=window)
        return datasets.contracts.transient.rollout_windows(transition_count, rollout_length=window, window_stride=window, window_offset=0)

    def begin_group(self, group: UnitGroup, *, progress: float) -> None:
        """
        Fix the windows of every unit of one update group and, in Stage B, one horizon and origin per window.

        The draws follow the group's unit order and each unit's window order at the
        stage budget fraction ``progress`` committed before the group, so the memory
        microbatch split cannot change them. A horizon longer than a short window is
        capped at its length.
        """
        lengths = tuple(_transition_count(key) for key in group.keys)
        self._group_windows = tuple(self._windows(length) for length in lengths)
        if self.curriculum_state is None:
            self._group_draws = ()
            return
        flat = iter(
            self.curriculum_state.draw(
                progress=progress,
                available_lengths=[length for windows in self._group_windows for _, length in windows],
            )
        )
        self._group_draws = tuple(tuple(next(flat) for _ in windows) for windows in self._group_windows)
        origins = [origin for draws in self._group_draws for _, origin in draws]
        self._last_origin_min, self._last_origin_max = min(origins), max(origins)

    def unit_losses(self, model: nn.Module, raw_batch: Any, loss: nn.Module, *, positions: tuple[int, ...]) -> Iterator[training_adapter.UnitLosses]:
        """
        Yield the Stage-A or Stage-B loss shares of the units at ``positions`` of the current group.

        Every unit's loss averages its supervised transitions: Stage A every transition
        of its windows; Stage B each window's drawn horizon, and its windows equally, so
        a longer horizon adds no weight. The RNO trains on its
        windows in temporal order (truncated backpropagation through time): each window
        starts from the hidden state built by teacher forcing from hour 0 with the
        current weights, detached so gradients stop at the window start, and one result
        per window releases its graph before the next window. Stage-B self-feeding
        never leaks into the carried state.
        """
        batch = self.prepare_batch(raw_batch, device=self.tensorizer.scaling.device, training=True)
        unit_windows = {self._group_windows[position] for position in positions} if self._group_windows else set()
        if len(unit_windows) != 1 or batch.batch_size != len(positions):
            msg = "A transient microbatch must hold units of one trajectory length from the current update group."
            raise ValueError(msg)
        windows = unit_windows.pop()
        if windows[-1][0] + windows[-1][1] != batch.rollout_length:
            msg = "Transient microbatch length disagrees with the windows fixed for its units."
            raise ValueError(msg)
        if self.spec.stage == "stage_a_teacher_forcing":
            yield from self._teacher_forced_losses(model, batch, loss, windows)
        else:
            yield from self._self_fed_losses(model, batch, loss, windows, tuple(self._group_draws[position] for position in positions))

    def _teacher_forced_losses(
        self, model: nn.Module, batch: TransientBatch, loss: nn.Module, windows: tuple[tuple[int, int], ...]
    ) -> Iterator[training_adapter.UnitLosses]:
        """Yield one teacher-forced share per window, carrying the detached RNO state to the next window."""
        hidden: object | None = None
        units = batch.batch_size
        for index, (start, length) in enumerate(windows):
            carry_to = None if index == len(windows) - 1 else windows[index + 1][0] - start
            rollout = teacher_forced_rollout(
                model,
                _slice_window(batch, start=start, length=length),
                self.tensorizer,
                model_kind=self.spec.model_kind,
                initial_hidden=hidden,
                capture=() if carry_to is None else (carry_to,),
            )
            if carry_to is not None:
                hidden = detach_hidden(rollout.captured_hidden[carry_to])
            share = length / batch.rollout_length
            components = loss.unit_components(rollout.normalized_prediction, y=rollout.normalized_target)
            yield training_adapter.UnitLosses(
                components={name: value * share for name, value in components.items()},
                supervised_transitions=units * length,
                forward_transitions=units * length,
            )

    def _self_fed_losses(
        self,
        model: nn.Module,
        batch: TransientBatch,
        loss: nn.Module,
        windows: tuple[tuple[int, int], ...],
        draws: tuple[tuple[tuple[int, int], ...], ...],
    ) -> Iterator[training_adapter.UnitLosses]:
        """Yield one self-fed share per window and drawn horizon; the RNO starts each origin from its teacher-forced state."""
        kind = self.spec.model_kind
        recurrent = self.spec.trajectory_window is not None
        units = batch.batch_size
        device = batch.state.device
        # Each window's loss is its mean over its own horizon; the unit averages its windows equally.
        share = 1.0 / len(windows)
        hidden: object | None = None
        for index, (start, length) in enumerate(windows):
            window = _slice_window(batch, start=start, length=length)
            # Horizons, origins and rows stay Python integers: the decisions below never wait for the device.
            horizons = [unit[index][0] for unit in draws]
            origins = [unit[index][1] for unit in draws]
            snapshots: Mapping[int, object | None] = {}
            warm_transitions = 0
            if recurrent:
                # The teacher-forced pass over the reference window yields the state at every origin and at the next window.
                carry_to = None if index == len(windows) - 1 else windows[index + 1][0] - start
                capture = set(origins) | (set() if carry_to is None else {carry_to})
                with torch.no_grad():
                    warm = teacher_forced_rollout(model, window, self.tensorizer, model_kind=kind, initial_hidden=hidden, capture=capture)
                snapshots = warm.captured_hidden
                hidden = None if carry_to is None else snapshots[carry_to]
                warm_transitions = units * length
            for horizon in sorted(set(horizons)):
                rows = [row for row, value in enumerate(horizons) if value == horizon]
                row_index = torch.tensor(rows, dtype=torch.long, device=device)
                target = _select_window(
                    _index_batch(window, row_index), tensorizer=self.tensorizer, horizon=horizon, origins=[origins[row] for row in rows]
                )
                rollout = (
                    _self_fed_from_snapshots(model, target, self.tensorizer, model_kind=kind, snapshots=snapshots, origins=origins, rows=rows)
                    if recurrent
                    else self_fed_rollout(model, target, self.tensorizer, model_kind=kind)
                )
                components = loss.unit_components(rollout.normalized_prediction, y=rollout.normalized_target)
                # Units outside this horizon get a zero share, so every result keeps the microbatch's unit order.
                scattered = {name: value.new_zeros(units).index_copy(0, row_index, value * share) for name, value in components.items()}
                yield training_adapter.UnitLosses(
                    components=scattered,
                    supervised_transitions=len(rows) * horizon,
                    forward_transitions=len(rows) * horizon + warm_transitions,
                )
                warm_transitions = 0

    def evaluation_step(self, model: nn.Module, batch: TransientBatch) -> training_adapter.AdapterEvaluationViews:
        """
        Return whole-trajectory views for every stage's checkpoint selection.

        Every model starts from the reference state at the trajectory start and
        rolls out the complete trajectory from its own predictions; the RNO
        carries only its own hidden state. A case stops before its first
        non-finite state and the other cases continue. The teacher-forced one-step
        view predicts every transition from its reference state, the RNO with the
        state built over the reference transitions before it.
        """
        kind = self.spec.model_kind
        with torch.no_grad():
            rollout = autonomous_rollout(
                model,
                self.tensorizer,
                state=batch.state,
                static=batch.static,
                boundary=batch.boundary,
                scalars=batch.scalars,
                t_n=batch.t_n,
                model_kind=kind,
            )
            one_step_physical = _teacher_forced_in_chunks(model, batch, self.tensorizer, model_kind=kind)
        length = batch.rollout_length
        # Completeness is the autonomous trajectory's alone; the one-step view is a diagnostic. The standardized view of a
        # finite physical rollout can still overflow, which leaves the case incomplete as well.
        complete = encode_finite_cases(self.tensorizer.scaling, rollout.physical_prediction, reference=batch.next_state).flatten(1).isfinite().all(1)
        return training_adapter.AdapterEvaluationViews(
            physical_prediction=rollout.physical_prediction,
            physical_target=batch.next_state,
            one_step_physical_prediction=one_step_physical,
            sample_count=batch.batch_size,
            processed_target_transitions=batch.batch_size * length,
            case_complete=complete,
            bound_violations=_bound_violations(rollout.physical_prediction, batch.t_n_plus_1),
            f_surf=batch.scalars[:, 2].view(batch.batch_size, 1).expand(-1, length),
        )

    def record_optimizer_work(self, work: training_adapter.OptimizerWork) -> None:
        """Forward only reported completed optimizer work to the controller."""
        if not isinstance(work, training_adapter.OptimizerWork):
            msg = "work must be an OptimizerWork."
            raise TypeError(msg)
        self.spec.controller.record_completed_work(
            microbatches=work.microbatches,
            processed_target_transitions=work.processed_target_transitions,
            forward_transitions=work.forward_transitions,
            peak_cuda_memory_bytes=work.peak_cuda_memory_bytes,
        )

    def record_validation_work(self, seconds: float) -> None:
        """Record completed validation time, which stays outside the optimisation clock."""
        self.spec.controller.record_validation_work(seconds)

    def controller_state(self) -> dict[str, Any]:
        """Return the strict controller evidence a terminal summary mirrors."""
        return self.spec.controller.state_dict()

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one eligible finite selection metric."""
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            msg = "Within-budget evaluation metric must be finite."
            raise ValueError(msg)
        self.spec.controller.record_within_budget_evaluation(float(metric), epoch_index=epoch_index)

    def validate_terminal_state(self, *, best_metric: float | None, best_epoch: int | None) -> None:
        """Require the selected checkpoint to be the best validation within the completed stage budget."""
        controller = self.spec.controller
        if controller.best_within_budget_metric is None or controller.best_within_budget_epoch is None:
            msg = "Transient training has no selected metric at or before its budget boundary."
            raise RuntimeError(msg)
        if best_metric != controller.best_within_budget_metric or best_epoch != controller.best_within_budget_epoch + 1:
            msg = "Selected transient checkpoint disagrees with stage-budget evidence."
            raise RuntimeError(msg)

    def telemetry_state(self) -> dict[str, float | int | bool | None]:
        """Return compact stage, stage-budget and, in Stage B, curriculum telemetry."""
        controller = self.spec.controller
        curriculum = self.curriculum_state
        return {
            **(
                {}
                if curriculum is None
                else {
                    "curriculum_progress": curriculum.progress,
                    "curriculum_active_stage": curriculum.active_stage,
                    "curriculum_max_horizon": curriculum.max_horizon,
                    "curriculum_draw_index": curriculum.draw_index,
                    "selected_rollout_horizon": curriculum.max_horizon,
                }
            ),
            "last_origin_min": getattr(self, "_last_origin_min", None),
            "last_origin_max": getattr(self, "_last_origin_max", None),
            "self_fed_stage": int(self.spec.stage == "stage_b_self_fed"),
            "post_handoff_optimizer_steps": controller.post_handoff_optimizer_steps,
            "processed_target_transitions": controller.processed_target_transitions,
            "forward_transitions": controller.forward_transitions,
            "validation_seconds": controller.validation_seconds,
            "peak_cuda_memory_bytes": controller.peak_cuda_memory_bytes,
            "best_within_budget_metric": controller.best_within_budget_metric,
            "best_within_budget_epoch": controller.best_within_budget_epoch,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return exact adapter continuation state with immutable semantic digest; only Stage B saves curriculum state."""
        return {
            "spec_digest": self.spec.digest,
            "controller": self.spec.controller.state_dict(),
            **({} if self.curriculum_state is None else {"curriculum": self.curriculum_state.state_dict()}),
            "last_origin_min": getattr(self, "_last_origin_min", None),
            "last_origin_max": getattr(self, "_last_origin_max", None),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact matching adapter continuation state; a Stage-A state with curriculum state is refused."""
        required = {"spec_digest", "controller", "last_origin_min", "last_origin_max", *(() if self.curriculum_state is None else ("curriculum",))}
        if not isinstance(state, dict) or set(state) != required or state["spec_digest"] != self.spec.digest:
            msg = "Saved transient adapter state conflicts with the configured semantic specification."
            raise ValueError(msg)
        if not isinstance(state["controller"], Mapping) or not isinstance(state.get("curriculum", {}), Mapping):
            message = "Saved transient adapter controller and curriculum state must be mappings."
            raise TypeError(message)
        origin_values = (state["last_origin_min"], state["last_origin_max"])
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in origin_values):
            msg = "Saved transient origin evidence must be null or non-negative integers."
            raise ValueError(msg)
        if (origin_values[0] is None) != (origin_values[1] is None) or (origin_values[0] is not None and origin_values[0] > origin_values[1]):
            msg = "Saved transient origin evidence is inconsistent."
            raise ValueError(msg)
        controller_clone = _clone_controller(self.spec.controller)
        controller_clone.load_state_dict(state["controller"])
        if self.curriculum_state is not None and self.spec.curriculum is not None:
            curriculum_clone = RolloutCurriculumState.create(self.spec.curriculum, seed=_seed(self.spec))
            curriculum_clone.load_state_dict(self.curriculum_state.state_dict())
            curriculum_clone.load_state_dict(state["curriculum"])
            self.curriculum_state.load_state_dict(curriculum_clone.state_dict())
        self.spec.controller.load_state_dict(controller_clone.state_dict())
        self._last_origin_min = state["last_origin_min"]
        self._last_origin_max = state["last_origin_max"]


def _seed(spec: TransientTrainingSpec) -> int:
    """Return the curriculum seed a Stage-B specification carries beside its curriculum."""
    if spec.curriculum_seed is None:
        msg = "A transient curriculum needs its seed."
        raise ValueError(msg)
    return spec.curriculum_seed


def _clone_controller(controller: MatchedComputeController) -> MatchedComputeController:
    """Return one independently mutable controller with matching current state."""
    payload = dict(controller.state_dict())
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
_STATE_FIELDS = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.dynamic_state)


def _teacher_forced_in_chunks(model: nn.Module, batch: TransientBatch, tensorizer: TransientTensorizer, *, model_kind: str) -> torch.Tensor:
    """Return the physical one-step predictions of whole trajectories, carrying the RNO state."""
    hidden: object | None = None
    parts: list[TransientRollout] = []
    for start in range(0, batch.rollout_length, _TEACHER_FORCED_CHUNK):
        length = min(_TEACHER_FORCED_CHUNK, batch.rollout_length - start)
        capture = (length,) if model_kind == "rno" else ()
        part = teacher_forced_rollout(
            model,
            _slice_window(batch, start=start, length=length),
            tensorizer,
            model_kind=model_kind,
            initial_hidden=hidden,
            capture=capture,
            require_finite=False,
        )
        hidden = part.captured_hidden.get(length)
        parts.append(part)
    return torch.cat([part.physical_prediction for part in parts], dim=1)


def _bound_violations(prediction: torch.Tensor, times: torch.Tensor) -> dict[str, tuple[float | None, ...]]:
    """Return per state channel and batch row the physical time of the first finite autonomous state outside the bounds, ``None`` without one."""
    invalid = drying.physical_invalid_mask(prediction.detach().cpu().numpy())
    state_times = times.detach().cpu().numpy()
    result: dict[str, tuple[float | None, ...]] = {}
    for index, name in enumerate(_STATE_FIELDS):
        steps: np.ndarray = np.any(invalid[:, :, index], axis=(-2, -1))
        first = state_times[np.arange(len(steps)), steps.argmax(axis=1)].tolist()
        violated: np.ndarray = np.any(steps, axis=1)
        result[name] = tuple(hours if hit else None for hours, hit in zip(first, violated, strict=True))
    return result


def _transition_count(key: object) -> int:
    """Return one unit's transition count from its sampler key."""
    if isinstance(key, bool) or not isinstance(key, int) or key < 1:
        msg = f"A transient unit key must be its positive transition count, got {key!r}."
        raise ValueError(msg)
    return key


def _slice_window(batch: TransientBatch, *, start: int, length: int) -> TransientBatch:
    """Return one teacher-forced window of every trajectory without recomputing its inputs."""
    if start < 0 or length < 1 or start + length > batch.rollout_length:
        msg = "Transient trajectory window must lie inside the batch."
        raise ValueError(msg)
    stop = start + length
    state = batch.state if start == 0 else batch.next_state[:, start - 1]
    sequence_input = batch.sequence_input[:, start:stop]
    return TransientBatch(
        state=state,
        next_state=batch.next_state[:, start:stop],
        static=batch.static,
        boundary=batch.boundary[:, start:stop],
        scalars=batch.scalars,
        t_n=batch.t_n[:, start:stop],
        t_n_plus_1=batch.t_n_plus_1[:, start:stop],
        dt=batch.dt[:, start:stop],
        step_input=sequence_input[:, 0],
        sequence_input=sequence_input,
    )


def _index_batch(batch: TransientBatch, rows: torch.Tensor) -> TransientBatch:
    """Return the selected samples of one batch in the given order."""
    return TransientBatch(
        state=batch.state[rows],
        next_state=batch.next_state[rows],
        static=batch.static[rows],
        boundary=batch.boundary[rows],
        scalars=batch.scalars[rows],
        t_n=batch.t_n[rows],
        t_n_plus_1=batch.t_n_plus_1[rows],
        dt=batch.dt[rows],
        step_input=batch.step_input[rows],
        sequence_input=batch.sequence_input[rows],
    )


def _self_fed_from_snapshots(
    model: nn.Module,
    batch: TransientBatch,
    tensorizer: TransientTensorizer,
    *,
    model_kind: str,
    snapshots: Mapping[int, object | None],
    origins: Sequence[int],
    rows: Sequence[int],
) -> TransientRollout:
    """
    Roll every sample of ``batch`` out from the RNO state captured at its own origin.

    ``batch`` holds the microbatch samples ``rows``, whose snapshots and ``origins``
    cover the whole microbatch. Samples whose origin is still the trajectory start
    keep the model's own initial state, which one batched call cannot mix with
    captured states, so those samples run as a separate group and are restored to
    batch order. The split is decided on Python integers, without device reads.
    """
    initial = [snapshots[origins[row]] is None for row in rows]
    if all(initial) or not any(initial):
        return self_fed_rollout(model, batch, tensorizer, model_kind=model_kind, initial_hidden=select_hidden_per_sample(snapshots, origins, rows))
    groups = (
        [local for local, flag in enumerate(initial) if flag],
        [local for local, flag in enumerate(initial) if not flag],
    )
    device = batch.state.device
    parts = [
        self_fed_rollout(
            model,
            _index_batch(batch, torch.tensor(local, dtype=torch.long, device=device)),
            tensorizer,
            model_kind=model_kind,
            initial_hidden=select_hidden_per_sample(snapshots, origins, [rows[position] for position in local]),
        )
        for local in groups
    ]
    concatenated = [*groups[0], *groups[1]]
    order = torch.tensor(sorted(range(len(concatenated)), key=concatenated.__getitem__), dtype=torch.long, device=device)
    return TransientRollout(
        scaled_delta=torch.cat([part.scaled_delta for part in parts])[order],
        physical_prediction=torch.cat([part.physical_prediction for part in parts])[order],
        physical_target=torch.cat([part.physical_target for part in parts])[order],
        normalized_prediction=torch.cat([part.normalized_prediction for part in parts])[order],
        normalized_target=torch.cat([part.normalized_target for part in parts])[order],
    )


def _select_window(batch: TransientBatch, *, tensorizer: TransientTensorizer, horizon: int, origins: Sequence[int]) -> TransientBatch:
    """Select one valid unpadded physical window per sample and reassemble inputs; origins are validated as Python integers."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1 or horizon > batch.rollout_length:
        msg = "Transient window horizon must fit the tensorized rollout length."
        raise ValueError(msg)
    if len(origins) != batch.batch_size or any(
        isinstance(origin, bool) or not isinstance(origin, int) or origin < 0 or origin + horizon > batch.rollout_length for origin in origins
    ):
        msg = "Transient window origins must be valid per-sample integer starts."
        raise ValueError(msg)
    starts = torch.tensor(origins, dtype=torch.long, device=batch.state.device)
    indices = starts[:, None] + torch.arange(horizon, device=starts.device)[None, :]
    stored = torch.cat((batch.state.unsqueeze(1), batch.next_state), dim=1)
    gather_states = indices[:, :, None, None, None].expand(-1, -1, *batch.next_state.shape[2:])
    next_state = batch.next_state.gather(1, gather_states)
    current_states = stored.gather(1, gather_states)
    boundary = batch.boundary.gather(1, indices[:, :, None].expand(-1, -1, batch.boundary.shape[-1]))
    t_n = batch.t_n.gather(1, indices)
    t_next = batch.t_n_plus_1.gather(1, indices)
    dt = batch.dt.gather(1, indices)
    inputs = torch.stack(
        [
            tensorizer.assemble_step(current_states[:, index], batch.static, boundary[:, index], batch.scalars, t_n[:, index])
            for index in range(horizon)
        ],
        dim=1,
    )
    return TransientBatch(
        state=current_states[:, 0],
        next_state=next_state,
        static=batch.static,
        boundary=boundary,
        scalars=batch.scalars,
        t_n=t_n,
        t_n_plus_1=t_next,
        dt=dt,
        step_input=inputs[:, 0],
        sequence_input=inputs,
    )


def build_transient_training_adapter(
    config: Mapping[str, Any],
    *,
    scaling: Any,
    device: torch.device,
    curriculum_seed: int | None,
    teacher_handoff: TeacherHandoffIdentity | None = None,
    matched_receipt_sha256: str | None = None,
) -> TransientTrainingAdapter:
    """Build one fresh task adapter from resolved transient runtime semantics; only Stage B takes a curriculum and its seed."""
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
    curriculum: RolloutCurriculum | None = None
    if stage == "stage_b_self_fed":
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
    # The controller binds the stage's one training identity, the digest of resume and admission: its budget belongs to it
    # like every other scientific setting (a run is never extended), while execution settings such as the memory batch,
    # loader workers or the device stay outside it, so a resume may change them.
    config_digest = training_checkpoint.config_digest(config)
    if (arm == "A1") != (matched_receipt_sha256 is not None):
        msg = "Exactly A1 is built with an admitted exact B time receipt."
        raise ValueError(msg)
    controller = MatchedComputeController(
        arm=arm,
        stage=stage,
        config_digest=config_digest,
        teacher_handoff=teacher_handoff,
        matched_receipt_sha256=matched_receipt_sha256,
    )
    spec = TransientTrainingSpec(
        stage=stage,
        model_kind=str(config["model"]["kind"]),
        controller=controller,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
        trajectory_window=trajectory_window,
    )
    return TransientTrainingAdapter(tensorizer=TransientTensorizer(tensorizer_spec, scaling.to(device)), spec=spec)
