# ruff: noqa: EM101, EM102, TRY003
"""
learning_transient_curriculum.py

Define transient rollout curricula and strict matched-compute accounting.

Responsibilities:
  - Select continuous self-fed rollout horizons and valid window origins
  - Persist private random sampling state at epoch boundaries
  - Bind matched A+ and B arms to immutable teacher-handoff evidence
  - Account completed optimizer work with CUDA-second or CPU-step clocks

Design principles:
  - Compute progress, not epoch number, controls curriculum admission
  - Shorter horizons retain positive probability after later milestones
  - Controller state changes only after a successful optimizer report

This module does NOT:
  - Load teacher files, run model forwards, or mutate optimizers
  - Treat CPU optimizer steps as device seconds
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, overload

import torch

TrainingStage = Literal["stage_a_teacher_forcing", "stage_b_self_fed"]
ComparisonArm = Literal["A0", "A+", "B"]
ClockKind = Literal["cuda_device_seconds", "optimizer_steps"]
BudgetControl = Literal["matched_compute", "stage_epochs"]
DEFAULT_ROLLOUT_LENGTHS: Final = (2, 4, 8, 16, 32)
DEFAULT_MILESTONE_FRACTIONS: Final = (0.0, 0.2, 0.4, 0.6, 0.8)
_CURRICULUM_SCHEMA_VERSION: Final = 1
_CONTROLLER_SCHEMA_VERSION: Final = 1
_HANDOFF_SCHEMA_VERSION: Final = 1
_SHA256_LENGTH: Final = 64


def _sha256(value: Any, *, label: str) -> str:
    """Require one lowercase SHA-256 digest."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest.")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """Require one positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    """Require one nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


@overload
def _finite_nonnegative(value: Any, *, label: str, allow_none: Literal[False] = False) -> float: ...


@overload
def _finite_nonnegative(value: Any, *, label: str, allow_none: Literal[True]) -> float | None: ...


@overload
def _finite_nonnegative(value: Any, *, label: str, allow_none: bool) -> float | None: ...


def _finite_nonnegative(value: Any, *, label: str, allow_none: bool = False) -> float | None:
    """Require one finite nonnegative float or an admitted null."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return float(value)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a stable digest for one strict JSON-compatible identity payload."""
    encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherHandoffIdentity:
    """Bind matched continuation arms to immutable teacher provenance."""

    source_run_name: str
    source_checkpoint_sha256: str
    source_scaling_sha256: str
    task_contract_sha256: str
    tensorizer_sha256: str
    model_kind: str
    input_profile: str

    def __post_init__(self) -> None:
        """Reject incomplete handoff identity fields."""
        for label in ("source_run_name", "model_kind", "input_profile"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string.")
        for label in ("source_checkpoint_sha256", "source_scaling_sha256", "task_contract_sha256", "tensorizer_sha256"):
            _sha256(getattr(self, label), label=label)

    @property
    def digest(self) -> str:
        """Return the immutable teacher identity digest."""
        return _canonical_digest(self.as_dict())

    def as_dict(self) -> dict[str, str | int]:
        """Return strict serializable handoff evidence."""
        return {
            "schema_version": _HANDOFF_SCHEMA_VERSION,
            "source_run_name": self.source_run_name,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_scaling_sha256": self.source_scaling_sha256,
            "task_contract_sha256": self.task_contract_sha256,
            "tensorizer_sha256": self.tensorizer_sha256,
            "model_kind": self.model_kind,
            "input_profile": self.input_profile,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TeacherHandoffIdentity:
        """Admit exact serialized handoff evidence."""
        expected = {
            "schema_version",
            "source_run_name",
            "source_checkpoint_sha256",
            "source_scaling_sha256",
            "task_contract_sha256",
            "tensorizer_sha256",
            "model_kind",
            "input_profile",
        }
        if not isinstance(value, Mapping) or set(value) != expected or value["schema_version"] != _HANDOFF_SCHEMA_VERSION:
            raise ValueError("Teacher handoff evidence does not match the current strict schema.")
        return cls(**{key: value[key] for key in expected if key != "schema_version"})


@dataclass(frozen=True, slots=True)
class RolloutCurriculum:
    """Describe an admitted continuous Stage-B horizon schedule."""

    DEFAULT_LENGTHS: ClassVar[tuple[int, ...]] = DEFAULT_ROLLOUT_LENGTHS
    DEFAULT_MILESTONE_FRACTIONS: ClassVar[tuple[float, ...]] = DEFAULT_MILESTONE_FRACTIONS
    lengths: tuple[int, ...] = DEFAULT_LENGTHS
    milestone_fractions: tuple[float, ...] = DEFAULT_MILESTONE_FRACTIONS

    def __post_init__(self) -> None:
        """Reject unbounded, unordered, or epoch-defined schedules."""
        lengths = tuple(self.lengths)
        allowed = (*DEFAULT_ROLLOUT_LENGTHS, 64)
        if lengths == (1,):
            pass
        elif not lengths or tuple(sorted(set(lengths))) != lengths or any(length not in allowed for length in lengths):
            raise ValueError("Curriculum lengths must be the Stage-A one-step horizon or increasing members of 2, 4, 8, 16, 32, optionally 64.")
        elif lengths[: len(DEFAULT_ROLLOUT_LENGTHS)] != DEFAULT_ROLLOUT_LENGTHS and lengths != DEFAULT_ROLLOUT_LENGTHS[: len(lengths)]:
            raise ValueError("Curriculum lengths must retain the canonical ordered prefix.")
        if len(self.milestone_fractions) != len(lengths) or self.milestone_fractions[0] != 0.0:
            raise ValueError("Curriculum milestones must align with lengths and begin at zero.")
        previous = -1.0
        for fraction in self.milestone_fractions:
            if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 1.0 or float(fraction) <= previous:
                raise ValueError("Curriculum milestones must be strictly increasing fractions in [0, 1).")
            previous = float(fraction)

    def active_stage(self, progress: float) -> int:
        """Return the largest admitted stage at compute progress."""
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("Curriculum progress must be finite in [0, 1].")
        stage = 0
        for index, milestone in enumerate(self.milestone_fractions):
            if progress + 1.0e-12 >= milestone:
                stage = index
        return stage

    def eligible_lengths(self, progress: float) -> tuple[int, ...]:
        """Return positive-probability horizons through the active stage."""
        return self.lengths[: self.active_stage(progress) + 1]

    def as_dict(self) -> dict[str, Any]:
        """Return strict serializable curriculum semantics."""
        return {"schema_version": _CURRICULUM_SCHEMA_VERSION, "lengths": list(self.lengths), "milestone_fractions": list(self.milestone_fractions)}

    @property
    def digest(self) -> str:
        """Return the immutable curriculum semantic digest."""
        return _canonical_digest(self.as_dict())


@dataclass(slots=True)
class RolloutCurriculumState:
    """Persist private draws and active continuous-curriculum state."""

    curriculum: RolloutCurriculum
    generator: torch.Generator
    draw_index: int = 0
    active_stage: int = 0
    max_horizon: int = 2
    progress: float = 0.0
    completed_milestones: tuple[int, ...] = ()

    @classmethod
    def create(cls, curriculum: RolloutCurriculum, *, seed: int) -> RolloutCurriculumState:
        """Create an isolated CPU generator from one explicit seed."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_nonnegative_int(seed, label="seed"))
        return cls(curriculum=curriculum, generator=generator)

    def select(self, *, progress: float, available_length: int, batch_size: int) -> tuple[int, torch.Tensor]:
        """Draw one horizon and per-sample valid origins across temporal strata."""
        _positive_int(available_length, label="available_length")
        _positive_int(batch_size, label="batch_size")
        active_stage = self.curriculum.active_stage(progress)
        active_maximum = self.curriculum.lengths[active_stage]
        if active_maximum > available_length:
            message = "The active curriculum maximum horizon does not fit the supplied rollout window."
            raise ValueError(message)
        eligible = self.curriculum.eligible_lengths(progress)
        horizon = eligible[int(torch.randint(len(eligible), (1,), generator=self.generator).item())]
        origins = self._sample_origins(available_length=available_length, horizon=horizon, batch_size=batch_size)
        self.draw_index += 1
        self.progress = float(progress)
        self.active_stage = self.curriculum.active_stage(progress)
        self.max_horizon = self.curriculum.lengths[self.active_stage]
        self.completed_milestones = tuple(
            index for index, milestone in enumerate(self.curriculum.milestone_fractions) if progress + 1.0e-12 >= milestone
        )
        return horizon, origins

    def _sample_origins(self, *, available_length: int, horizon: int, batch_size: int) -> torch.Tensor:
        """Sample valid early/middle/late origins without padding or replacement rules."""
        maximum = available_length - horizon
        if maximum < 0:
            raise ValueError("Rollout horizon exceeds available sequence length.")
        if maximum == 0:
            return torch.zeros(batch_size, dtype=torch.long)
        edges = ((0, maximum // 3), (maximum // 3 + 1, (2 * maximum) // 3), ((2 * maximum) // 3 + 1, maximum))
        draws: list[int] = []
        for index in range(batch_size):
            low, high = edges[index % len(edges)]
            if low > high:
                low, high = 0, maximum
            draws.append(int(torch.randint(low, high + 1, (1,), generator=self.generator).item()))
        return torch.tensor(draws, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        """Return exact CPU generator and curriculum continuation state."""
        return {
            "curriculum": self.curriculum.as_dict(),
            "curriculum_digest": self.curriculum.digest,
            "generator_state": self.generator.get_state().tolist(),
            "draw_index": self.draw_index,
            "active_stage": self.active_stage,
            "max_horizon": self.max_horizon,
            "progress": self.progress,
            "completed_milestones": list(self.completed_milestones),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        """Restore only matching strict curriculum state."""
        required = {
            "curriculum",
            "curriculum_digest",
            "generator_state",
            "draw_index",
            "active_stage",
            "max_horizon",
            "progress",
            "completed_milestones",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["curriculum"] != self.curriculum.as_dict()
            or value["curriculum_digest"] != self.curriculum.digest
        ):
            raise ValueError("Saved curriculum state does not match the configured curriculum.")
        state = torch.tensor(value["generator_state"], dtype=torch.uint8)
        self.generator.set_state(state)
        self.draw_index = _nonnegative_int(value["draw_index"], label="draw_index")
        self.active_stage = _nonnegative_int(value["active_stage"], label="active_stage")
        self.max_horizon = _positive_int(value["max_horizon"], label="max_horizon")
        self.progress = float(_finite_nonnegative(value["progress"], label="progress"))
        if self.progress > 1.0:
            raise ValueError("progress must not exceed one.")
        milestones = value["completed_milestones"]
        if not isinstance(milestones, list) or any(not isinstance(index, int) for index in milestones):
            raise ValueError("completed_milestones must be an integer list.")
        self.completed_milestones = tuple(milestones)


@dataclass(slots=True)
class MatchedComputeController:
    """Account completed work and derive the configured transient budget boundary."""

    arm: ComparisonArm
    stage: TrainingStage
    clock_kind: ClockKind
    config_digest: str
    budget_control: BudgetControl = "matched_compute"
    planned_stage_epochs: int | None = None
    completed_stage_epochs: int = 0
    teacher_handoff: TeacherHandoffIdentity | None = None
    planned_teacher_forcing_budget_seconds: float | None = None
    planned_teacher_forcing_budget_steps: int | None = None
    rollout_reference_compute_seconds: float | None = None
    rollout_reference_compute_steps: int | None = None
    post_handoff_optimizer_device_seconds: float | None = None
    post_handoff_optimizer_steps: int = 0
    successful_optimizer_steps: int = 0
    microbatches: int = 0
    processed_target_transitions: int = 0
    forward_transitions: int = 0
    teacher_forcing_optimizer_device_seconds: float | None = None
    teacher_forcing_optimizer_steps: int = 0
    wall_seconds: float = 0.0
    validation_seconds: float = 0.0
    peak_cuda_memory_bytes: int | None = None
    budget_complete: bool = False
    crossing_epoch: int | None = None
    crossing_microbatch: int | None = None
    best_within_budget_metric: float | None = None
    best_within_budget_epoch: int | None = None

    def __post_init__(self) -> None:
        """Validate immutable arm, clock, identity, and budget semantics."""
        if self.arm not in {"A0", "A+", "B"} or self.stage not in {"stage_a_teacher_forcing", "stage_b_self_fed"}:
            raise ValueError("Matched compute arm or stage is unsupported.")
        if self.clock_kind not in {"cuda_device_seconds", "optimizer_steps"}:
            raise ValueError("Matched compute clock kind is unsupported.")
        if self.budget_control not in {"matched_compute", "stage_epochs"}:
            raise ValueError("Transient budget control is unsupported.")
        _sha256(self.config_digest, label="config_digest")
        _nonnegative_int(self.completed_stage_epochs, label="completed_stage_epochs")
        if self.budget_control == "stage_epochs":
            if self.planned_stage_epochs is None:
                raise ValueError("Epoch-budgeted transient stages require planned_stage_epochs.")
            _positive_int(self.planned_stage_epochs, label="planned_stage_epochs")
            if self.completed_stage_epochs > self.planned_stage_epochs:
                raise ValueError("completed_stage_epochs exceeds planned_stage_epochs.")
        elif self.planned_stage_epochs is not None or self.completed_stage_epochs != 0:
            raise ValueError("Matched-compute stages cannot carry epoch-budget state.")
        if self.arm == "A0" and self.teacher_handoff is not None:
            raise ValueError("A0 must remain unmatched and cannot consume a teacher handoff.")
        if self.arm in {"A+", "B"} and not isinstance(self.teacher_handoff, TeacherHandoffIdentity):
            raise ValueError("Matched A+ and B arms require immutable teacher-handoff identity.")
        if self.stage == "stage_b_self_fed" and self.arm != "B":
            raise ValueError("Self-fed Stage B requires arm B.")
        if self.stage == "stage_a_teacher_forcing" and self.arm == "B":
            raise ValueError("Arm B requires self-fed Stage B.")
        for label in (
            "planned_teacher_forcing_budget_seconds",
            "rollout_reference_compute_seconds",
            "post_handoff_optimizer_device_seconds",
            "teacher_forcing_optimizer_device_seconds",
            "wall_seconds",
            "validation_seconds",
        ):
            _finite_nonnegative(getattr(self, label), label=label, allow_none=label not in {"wall_seconds", "validation_seconds"})
        for label in ("planned_teacher_forcing_budget_steps", "rollout_reference_compute_steps"):
            value = getattr(self, label)
            if value is not None:
                _positive_int(value, label=label)
        for label in (
            "post_handoff_optimizer_steps",
            "successful_optimizer_steps",
            "microbatches",
            "processed_target_transitions",
            "forward_transitions",
            "teacher_forcing_optimizer_steps",
        ):
            _nonnegative_int(getattr(self, label), label=label)
        matched_values = (
            self.planned_teacher_forcing_budget_seconds,
            self.planned_teacher_forcing_budget_steps,
            self.rollout_reference_compute_seconds,
            self.rollout_reference_compute_steps,
        )
        if self.budget_control == "stage_epochs" and any(value is not None for value in matched_values):
            raise ValueError("Epoch-budgeted transient stages cannot carry matched-compute limits.")
        if (
            self.budget_control == "matched_compute"
            and self.clock_kind == "cuda_device_seconds"
            and self.planned_teacher_forcing_budget_seconds is None
            and self.arm != "A0"
        ):
            raise ValueError("Matched CUDA arms require a planned teacher-forcing device-second budget.")
        if (
            self.budget_control == "matched_compute"
            and self.clock_kind == "optimizer_steps"
            and self.planned_teacher_forcing_budget_steps is None
            and self.arm != "A0"
        ):
            raise ValueError("Matched CPU arms require a planned teacher-forcing optimizer-step budget.")

    @property
    def remaining_to_planned_teacher_forcing_budget_seconds(self) -> float | None:
        """Return CUDA-only remaining teacher-forcing budget seconds."""
        if (
            self.clock_kind != "cuda_device_seconds"
            or self.planned_teacher_forcing_budget_seconds is None
            or self.post_handoff_optimizer_device_seconds is None
        ):
            return None
        return max(0.0, self.planned_teacher_forcing_budget_seconds - self.post_handoff_optimizer_device_seconds)

    @property
    def remaining_teacher_forcing_compute_to_match_rollout_seconds(self) -> float | None:
        """Return CUDA-only remaining teacher-forcing compute to rollout reference."""
        if (
            self.clock_kind != "cuda_device_seconds"
            or self.rollout_reference_compute_seconds is None
            or self.teacher_forcing_optimizer_device_seconds is None
        ):
            return None
        return max(0.0, self.rollout_reference_compute_seconds - self.teacher_forcing_optimizer_device_seconds)

    @property
    def remaining_to_planned_teacher_forcing_budget_steps(self) -> int | None:
        """Return CPU-only remaining planned successful optimizer steps."""
        if self.clock_kind != "optimizer_steps" or self.planned_teacher_forcing_budget_steps is None:
            return None
        return max(0, self.planned_teacher_forcing_budget_steps - self.successful_optimizer_steps)

    @property
    def remaining_teacher_forcing_compute_to_match_rollout_steps(self) -> int | None:
        """Return CPU-only remaining Stage-A successful steps to rollout reference."""
        if self.clock_kind != "optimizer_steps" or self.rollout_reference_compute_steps is None:
            return None
        return max(0, self.rollout_reference_compute_steps - self.teacher_forcing_optimizer_steps)

    @property
    def remaining_steps(self) -> int | None:
        """Return the compatibility alias for planned CPU remaining steps."""
        return self.remaining_to_planned_teacher_forcing_budget_steps

    @property
    def progress(self) -> float:
        """Return bounded progress under the configured budget owner."""
        if self.budget_control == "stage_epochs":
            if self.planned_stage_epochs is None:
                return 0.0
            return min(1.0, self.completed_stage_epochs / self.planned_stage_epochs)
        if self.clock_kind == "cuda_device_seconds":
            if self.planned_teacher_forcing_budget_seconds is None or self.planned_teacher_forcing_budget_seconds == 0.0:
                return 0.0
            return min(1.0, (self.post_handoff_optimizer_device_seconds or 0.0) / self.planned_teacher_forcing_budget_seconds)
        if self.planned_teacher_forcing_budget_steps is None:
            return 0.0
        return min(1.0, self.successful_optimizer_steps / self.planned_teacher_forcing_budget_steps)

    def begin_epoch(self, *, epoch_index: int, total_epochs: int) -> None:
        """Advance completed-epoch progress without completing the active epoch early."""
        if self.budget_control != "stage_epochs":
            return
        epoch = _nonnegative_int(epoch_index, label="epoch_index")
        total = _positive_int(total_epochs, label="total_epochs")
        if total != self.planned_stage_epochs:
            raise ValueError("Runtime total_epochs conflicts with the persisted stage budget.")
        if epoch < self.completed_stage_epochs or epoch >= total:
            raise ValueError("Runtime epoch progression conflicts with persisted stage-budget state.")
        self.completed_stage_epochs = epoch

    def record_completed_work(
        self,
        *,
        successful: bool,
        optimizer_device_seconds: float | None,
        microbatches: int,
        processed_target_transitions: int,
        forward_transitions: int,
        wall_seconds: float,
        epoch_index: int,
        microbatch_index: int,
        peak_cuda_memory_bytes: int | None = None,
    ) -> None:
        """
        Account one completed optimizer group; success means parameters changed.

        All measured compute and logical transition evidence is retained for an
        AMP-overflow group. Only successful optimizer-step counters exclude an
        overflow because no parameter update occurred.
        """
        if self.budget_complete:
            message = "Transient optimizer work cannot continue after the configured budget boundary."
            raise RuntimeError(message)
        _nonnegative_int(microbatches, label="microbatches")
        _nonnegative_int(processed_target_transitions, label="processed_target_transitions")
        _nonnegative_int(forward_transitions, label="forward_transitions")
        wall = _finite_nonnegative(wall_seconds, label="wall_seconds")
        if self.clock_kind == "cuda_device_seconds":
            seconds = _finite_nonnegative(optimizer_device_seconds, label="optimizer_device_seconds")
            self.post_handoff_optimizer_device_seconds = (self.post_handoff_optimizer_device_seconds or 0.0) + seconds
            if self.stage == "stage_a_teacher_forcing":
                self.teacher_forcing_optimizer_device_seconds = (self.teacher_forcing_optimizer_device_seconds or 0.0) + seconds
        elif optimizer_device_seconds is not None:
            message = "CPU matched-compute accounting must not report optimizer device seconds."
            raise ValueError(message)
        self.post_handoff_optimizer_steps += 1
        self.microbatches += microbatches
        self.processed_target_transitions += processed_target_transitions
        self.forward_transitions += forward_transitions
        self.wall_seconds += wall
        if peak_cuda_memory_bytes is not None:
            self.peak_cuda_memory_bytes = max(
                self.peak_cuda_memory_bytes or 0, _nonnegative_int(peak_cuda_memory_bytes, label="peak_cuda_memory_bytes")
            )
        if successful:
            self.successful_optimizer_steps += 1
            if self.stage == "stage_a_teacher_forcing":
                self.teacher_forcing_optimizer_steps += 1
        if self.budget_control == "matched_compute" and not self.budget_complete and self.progress >= 1.0:
            self.budget_complete = True
            self.crossing_epoch = _nonnegative_int(epoch_index, label="epoch_index")
            self.crossing_microbatch = _nonnegative_int(microbatch_index, label="microbatch_index")

    def record_validation_work(self, seconds: float) -> None:
        """Accumulate completed validation wall time outside the primary budget."""
        self.validation_seconds += _finite_nonnegative(seconds, label="validation_seconds")

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one finite selection metric within the configured budget boundary."""
        if not math.isfinite(metric):
            return
        epoch = _nonnegative_int(epoch_index, label="epoch_index")
        if self.best_within_budget_metric is None or metric < self.best_within_budget_metric:
            self.best_within_budget_metric = float(metric)
            self.best_within_budget_epoch = epoch
        if self.budget_control == "stage_epochs":
            if self.planned_stage_epochs is None or epoch + 1 > self.planned_stage_epochs:
                raise ValueError("Evaluation epoch exceeds the configured stage budget.")
            self.completed_stage_epochs = epoch + 1
            if self.completed_stage_epochs == self.planned_stage_epochs:
                self.budget_complete = True
                self.crossing_epoch = epoch
                self.crossing_microbatch = None

    def state_dict(self) -> dict[str, Any]:
        """Return strict checkpointable controller evidence."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["schema_version"] = _CONTROLLER_SCHEMA_VERSION
        values["teacher_handoff"] = None if self.teacher_handoff is None else self.teacher_handoff.as_dict()
        return values

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        """Restore only matching immutable controller semantics."""
        if not isinstance(value, Mapping):
            raise TypeError("Saved transient budget state must be a mapping.")
        expected = set(self.state_dict())
        if set(value) != expected or value.get("schema_version") != _CONTROLLER_SCHEMA_VERSION:
            raise ValueError("Saved transient budget state does not match the strict schema.")
        candidate = dict(value)
        candidate.pop("schema_version")
        handoff = candidate["teacher_handoff"]
        candidate["teacher_handoff"] = None if handoff is None else TeacherHandoffIdentity.from_mapping(handoff)
        restored = MatchedComputeController(**candidate)
        immutable = (
            "arm",
            "stage",
            "clock_kind",
            "config_digest",
            "budget_control",
            "planned_stage_epochs",
            "teacher_handoff",
            "planned_teacher_forcing_budget_seconds",
            "planned_teacher_forcing_budget_steps",
            "rollout_reference_compute_seconds",
            "rollout_reference_compute_steps",
        )
        if any(getattr(restored, name) != getattr(self, name) for name in immutable):
            raise ValueError("Saved transient budget state conflicts with configured semantic identity.")
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(restored, name))


@dataclass(frozen=True, slots=True)
class TransientTrainingSpec:
    """Bind one transient stage to a model, controller, and evaluation horizon."""

    stage: TrainingStage
    model_kind: str
    controller: MatchedComputeController
    gradient_accumulation_steps: int = 1
    curriculum: RolloutCurriculum = field(default_factory=RolloutCurriculum)
    curriculum_seed: int = 0
    fixed_evaluation_horizon: int = 32

    def __post_init__(self) -> None:
        """Reject incompatible stage, controller, and horizon selections."""
        if self.model_kind not in {"fno", "uno", "rno"}:
            raise ValueError("Transient model_kind must be fno, uno, or rno.")
        _positive_int(self.gradient_accumulation_steps, label="gradient_accumulation_steps")
        _nonnegative_int(self.curriculum_seed, label="curriculum_seed")
        _positive_int(self.fixed_evaluation_horizon, label="fixed_evaluation_horizon")
        if self.fixed_evaluation_horizon not in self.curriculum.lengths:
            raise ValueError("fixed_evaluation_horizon must be an admitted curriculum horizon.")
        if self.stage != self.controller.stage:
            raise ValueError("Training specification stage must match its compute controller.")
        if self.stage == "stage_a_teacher_forcing" and self.controller.arm not in {"A0", "A+"}:
            raise ValueError("Stage A supports only A0 or A+ arms.")
        if self.stage == "stage_b_self_fed" and self.controller.arm != "B":
            raise ValueError("Stage B requires the B comparison arm.")

    @property
    def digest(self) -> str:
        """Return immutable task-stage semantic identity excluding mutable work."""
        controller_fields = (
            "arm",
            "stage",
            "clock_kind",
            "config_digest",
            "budget_control",
            "planned_stage_epochs",
            "teacher_handoff",
            "planned_teacher_forcing_budget_seconds",
            "planned_teacher_forcing_budget_steps",
            "rollout_reference_compute_seconds",
            "rollout_reference_compute_steps",
        )
        return _canonical_digest(
            {
                "stage": self.stage,
                "model_kind": self.model_kind,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "curriculum": self.curriculum.as_dict(),
                "curriculum_seed": self.curriculum_seed,
                "fixed_evaluation_horizon": self.fixed_evaluation_horizon,
                "controller_identity": {
                    key: getattr(self.controller, key)
                    if key != "teacher_handoff"
                    else (None if self.controller.teacher_handoff is None else self.controller.teacher_handoff.as_dict())
                    for key in controller_fields
                },
            }
        )
