# ruff: noqa: EM101, TRY003, ANN202, ARG002
"""
learning_transient_adapter.py

Adapt transient drying physical batches to generic training-step contracts.

Responsibilities:
  - Tensorize physical transient evidence on a caller-selected device
  - Run teacher-forced Stage A and self-fed Stage B differentiable rollouts
  - Select continuous Stage-B windows without padding or detached feedback
  - Persist curriculum and matched-compute continuation evidence

Design principles:
  - Tensorizer and scaling artifacts remain the channel and reconstruction owners
  - Stage-A evaluation is teacher-forced while Stage-B evaluation is autonomous
  - Controller work advances only after generic-loop optimizer confirmation

This module does NOT:
  - Perform optimizer steps, checkpoint publication, or metric aggregation
  - Invent boundary, startup, static, scalar, or time conditioning evidence
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

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
from .learning_transient_rollout import self_fed_rollout, teacher_forced_rollout
from .learning_transient_tensorizer import TransientBatch, TransientTensorizer


class TransientTrainingAdapter:
    """Implement task-owned Stage-A and Stage-B training adaptation."""

    def __init__(self, *, tensorizer: TransientTensorizer, spec: TransientTrainingSpec) -> None:
        """Bind immutable tensorization and stage semantics."""
        if not isinstance(tensorizer, TransientTensorizer):
            raise TypeError("tensorizer must be a TransientTensorizer.")
        if not isinstance(spec, TransientTrainingSpec):
            raise TypeError("spec must be a TransientTrainingSpec.")
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
            raise ValueError("epoch_index must be a non-negative integer.")
        if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
            raise ValueError("total_epochs must be a positive integer.")
        self.spec.controller.begin_epoch(epoch_index=epoch_index, total_epochs=total_epochs)
        self._epoch_index = epoch_index
        self._optimizer_events = 0

    def prepare_batch(self, raw_batch: Any, *, device: torch.device, training: bool) -> TransientBatch:
        """Recursively move physical evidence and tensorize it on the scaler device."""
        if not isinstance(device, torch.device):
            raise TypeError("device must be one concrete torch.device.")
        if device != self.tensorizer.scaling.device:
            raise ValueError("Transient adapter device must match its scaling artifact device.")
        moved = _move_to_device(raw_batch, device)
        if not isinstance(moved, Mapping):
            raise TypeError("Transient raw batch must remain a mapping after device transfer.")
        batch = self.tensorizer.tensorize(moved)
        if training and self.spec.stage == "stage_b_self_fed":
            horizon, origins = self.curriculum_state.select(
                progress=self.spec.controller.progress,
                available_length=batch.rollout_length,
                batch_size=batch.batch_size,
            )
            self._last_origin_min = int(origins.min().item())
            self._last_origin_max = int(origins.max().item())
            return _select_window(
                batch,
                tensorizer=self.tensorizer,
                horizon=horizon,
                origins=origins.to(device=device),
            )
        return batch

    def training_step(self, model: nn.Module, batch: TransientBatch, loss: nn.Module) -> training_adapter.AdapterTrainingStep:
        """Return one Stage-A or Stage-B differentiable loss and logical work."""
        rollout = self._run_rollout(model, batch, evaluation=False)
        components = _loss_components(loss, rollout.scaled_delta, batch.scaled_target, rollout.normalized_prediction, rollout.normalized_target)
        total = components["total"]
        if not bool(torch.isfinite(total).all().item()):
            raise FloatingPointError("Transient training loss is non-finite.")
        transitions = batch.batch_size * batch.rollout_length
        return training_adapter.AdapterTrainingStep(
            loss=total,
            components=components,
            sample_count=batch.batch_size,
            processed_target_transitions=transitions,
            forward_transitions=transitions,
        )

    def evaluation_step(self, model: nn.Module, batch: TransientBatch) -> training_adapter.AdapterEvaluationViews:
        """Return Stage-A teacher-forced or fixed-horizon Stage-B autonomous views."""
        evaluation_batch = batch
        if self.spec.stage == "stage_b_self_fed":
            horizon = self.spec.fixed_evaluation_horizon
            if batch.rollout_length < horizon:
                raise ValueError("Stage-B evaluation batch does not contain the configured fixed autonomous horizon.")
            origins = torch.zeros(batch.batch_size, dtype=torch.long, device=batch.state.device)
            evaluation_batch = _select_window(batch, tensorizer=self.tensorizer, horizon=horizon, origins=origins)
        with torch.no_grad():
            rollout = self._run_rollout(model, evaluation_batch, evaluation=True)
        transitions = evaluation_batch.batch_size * evaluation_batch.rollout_length
        f_surf = evaluation_batch.scalars[:, 2].view(evaluation_batch.batch_size, 1).expand(-1, evaluation_batch.rollout_length)
        return training_adapter.AdapterEvaluationViews(
            normalized_prediction=rollout.normalized_prediction,
            normalized_target=rollout.normalized_target,
            physical_prediction=rollout.physical_prediction,
            physical_target=rollout.physical_target,
            sample_count=evaluation_batch.batch_size,
            processed_target_transitions=transitions,
            f_surf=f_surf,
        )

    def _run_rollout(self, model: nn.Module, batch: TransientBatch, *, evaluation: bool):
        """Select the stage-specific rollout semantics without hidden alternatives."""
        if self.spec.stage == "stage_a_teacher_forcing":
            return teacher_forced_rollout(model, batch, self.tensorizer, model_kind=self.spec.model_kind)
        if self.spec.stage == "stage_b_self_fed":
            return self_fed_rollout(model, batch, self.tensorizer, model_kind=self.spec.model_kind)
        raise RuntimeError("Transient training specification has an unsupported stage.")

    def record_optimizer_work(self, work: training_adapter.OptimizerWork) -> None:
        """Forward only reported completed optimizer work to the controller."""
        if not isinstance(work, training_adapter.OptimizerWork):
            raise TypeError("work must be an OptimizerWork.")
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
        )
        self._optimizer_events += 1

    def record_validation_work(self, seconds: float) -> None:
        """Record completed validation time excluded from matched-compute progress."""
        self.spec.controller.record_validation_work(seconds)

    def budget_state(self) -> dict[str, Any]:
        """Return strict matched-compute evidence for loop stop decisions."""
        return self.spec.controller.state_dict()

    def should_stop(self) -> bool:
        """Return the controller's sticky completion state."""
        return self.spec.controller.budget_complete

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one eligible finite selection metric."""
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise ValueError("Within-budget evaluation metric must be finite.")
        self.spec.controller.record_within_budget_evaluation(float(metric), epoch_index=epoch_index)

    def validate_terminal_state(self, *, best_metric: float | None, best_epoch: int | None) -> None:
        """Require a completed budget and exact budget-boundary checkpoint selection."""
        controller = self.spec.controller
        if controller.arm == "A0" and controller.budget_control == "matched_compute":
            return
        if not controller.budget_complete:
            raise RuntimeError("Transient training ended before its configured budget completed.")
        if controller.best_within_budget_metric is None or controller.best_within_budget_epoch is None:
            raise RuntimeError("Matched transient training has no selected metric at or before its budget boundary.")
        if best_metric != controller.best_within_budget_metric or best_epoch != controller.best_within_budget_epoch + 1:
            raise RuntimeError("Selected transient checkpoint disagrees with matched-compute budget evidence.")

    def telemetry_state(self) -> dict[str, float | int | bool | None]:
        """Return compact stage, curriculum, and matched-compute telemetry."""
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
            "planned_teacher_forcing_budget_seconds": controller.planned_teacher_forcing_budget_seconds,
            "planned_teacher_forcing_budget_steps": controller.planned_teacher_forcing_budget_steps,
            "rollout_reference_compute_seconds": controller.rollout_reference_compute_seconds,
            "rollout_reference_compute_steps": controller.rollout_reference_compute_steps,
            "post_handoff_optimizer_device_seconds": controller.post_handoff_optimizer_device_seconds,
            "post_handoff_optimizer_steps": controller.post_handoff_optimizer_steps,
            "teacher_forcing_optimizer_device_seconds": controller.teacher_forcing_optimizer_device_seconds,
            "teacher_forcing_optimizer_steps": controller.teacher_forcing_optimizer_steps,
            "successful_optimizer_steps": controller.successful_optimizer_steps,
            "processed_target_transitions": controller.processed_target_transitions,
            "forward_transitions": controller.forward_transitions,
            "wall_seconds": controller.wall_seconds,
            "validation_seconds": controller.validation_seconds,
            "peak_cuda_memory_bytes": controller.peak_cuda_memory_bytes,
            "remaining_to_planned_teacher_forcing_budget_seconds": controller.remaining_to_planned_teacher_forcing_budget_seconds,
            "remaining_teacher_forcing_compute_to_match_rollout_seconds": controller.remaining_teacher_forcing_compute_to_match_rollout_seconds,
            "remaining_to_planned_teacher_forcing_budget_steps": controller.remaining_to_planned_teacher_forcing_budget_steps,
            "remaining_teacher_forcing_compute_to_match_rollout_steps": controller.remaining_teacher_forcing_compute_to_match_rollout_steps,
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
            raise ValueError("Saved transient adapter state conflicts with the configured semantic specification.")
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
            raise ValueError("Saved transient origin evidence must be null or non-negative integers.")
        if (origin_values[0] is None) != (origin_values[1] is None) or (origin_values[0] is not None and origin_values[0] > origin_values[1]):
            raise ValueError("Saved transient origin evidence is inconsistent.")
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
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _select_window(batch: TransientBatch, *, tensorizer: TransientTensorizer, horizon: int, origins: torch.Tensor) -> TransientBatch:
    """Select one valid unpadded physical window per sample and reassemble inputs."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1 or horizon > batch.rollout_length:
        raise ValueError("Transient window horizon must fit the tensorized rollout length.")
    if (
        origins.shape != (batch.batch_size,)
        or origins.dtype != torch.long
        or bool((origins < 0).any().item())
        or bool((origins + horizon > batch.rollout_length).any().item())
    ):
        raise ValueError("Transient window origins must be valid per-sample integer starts.")
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
            raise TypeError("Transient loss compute_components must return a mapping containing total.")
        return dict(components)
    total = loss(prediction, target)
    return {"total": total, "data": total}


def build_transient_training_adapter(
    config: Mapping[str, Any],
    *,
    scaling: Any,
    device: torch.device,
    teacher_handoff: TeacherHandoffIdentity | None = None,
) -> TransientTrainingAdapter:
    """Build one fresh task adapter from resolved transient runtime semantics."""
    training = config.get("training")
    temporal = config.get("temporal")
    profile = config.get("input_profile")
    if not isinstance(training, Mapping) or not isinstance(temporal, Mapping) or not isinstance(profile, str):
        raise TypeError("Resolved transient adapter config lacks persisted training or tensorizer semantics.")
    tensorizer_spec = TransientTensorizerSpec.from_mapping({"input_profile": profile, "temporal_conditioning": temporal.get("temporal_conditioning")})
    stage: TrainingStage = "stage_a_teacher_forcing" if training.get("stage") == "a" else "stage_b_self_fed"
    arm_map: dict[str, ComparisonArm] = {"a0": "A0", "a_plus": "A+", "b": "B"}
    try:
        arm = arm_map[training["comparison_arm"]]
    except KeyError as error:
        raise ValueError("Transient comparison arm is invalid.") from error
    matched = training["matched_compute"]
    if not isinstance(matched, Mapping):
        raise TypeError("Transient matched_compute must be a mapping.")
    if device.type == "cuda":
        if matched.get("planned_steps") is not None or matched.get("rollout_reference_steps") is not None:
            raise ValueError("CUDA transient runs must use seconds budgets only.")
        clock_kind: ClockKind = "cuda_device_seconds"
    else:
        if matched.get("planned_seconds") is not None or matched.get("rollout_reference_seconds") is not None:
            raise ValueError("CPU transient runs must use optimizer-step budgets only.")
        clock_kind = "optimizer_steps"
    curriculum_data = training["curriculum"]
    if not isinstance(curriculum_data, Mapping):
        raise TypeError("Transient curriculum must be a mapping.")
    curriculum = RolloutCurriculum(
        lengths=tuple(curriculum_data["lengths"]),
        milestone_fractions=tuple(curriculum_data["milestone_fractions"]),
    )
    semantic_config = {key: value for key, value in config.items() if key not in {"paths", "_transient_tensorizer"}}
    try:
        encoded_config = json.dumps(semantic_config, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Resolved transient adapter semantics must be canonical JSON.") from error
    config_digest = hashlib.sha256(encoded_config).hexdigest()
    matched_limits = tuple(
        matched.get(key)
        for key in (
            "planned_seconds",
            "planned_steps",
            "rollout_reference_seconds",
            "rollout_reference_steps",
        )
    )
    budget_control: BudgetControl = (
        "stage_epochs" if arm == "A0" or (arm == "B" and all(value is None for value in matched_limits)) else "matched_compute"
    )
    controller = MatchedComputeController(
        arm=arm,
        stage=stage,
        clock_kind=clock_kind,
        config_digest=config_digest,
        budget_control=budget_control,
        planned_stage_epochs=int(training["epochs"]) if budget_control == "stage_epochs" else None,
        teacher_handoff=teacher_handoff,
        planned_teacher_forcing_budget_seconds=matched.get("planned_seconds"),
        planned_teacher_forcing_budget_steps=matched.get("planned_steps"),
        rollout_reference_compute_seconds=matched.get("rollout_reference_seconds"),
        rollout_reference_compute_steps=matched.get("rollout_reference_steps"),
    )
    spec = TransientTrainingSpec(
        stage=stage,
        model_kind=str(config["model"]["kind"]),
        controller=controller,
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        curriculum=curriculum,
        curriculum_seed=int(curriculum_data["seed"]),
        fixed_evaluation_horizon=int(training["fixed_evaluation_horizon"]),
    )
    return TransientTrainingAdapter(tensorizer=TransientTensorizer(tensorizer_spec, scaling.to(device)), spec=spec)
