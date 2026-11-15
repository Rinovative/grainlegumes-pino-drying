"""Define transient rollout curricula and per-stage compute accounting.

Responsibilities:
  - Draw one self-fed rollout horizon and one valid origin per window, capped at short windows
  - Persist private random sampling state at update-group boundaries
  - Bind A0, B, and A1 comparison arms to immutable teacher-handoff and B-receipt evidence
  - Account completed optimizer work: steps, microbatches, transitions, peak memory and validation time
  - Admit a completed stage's terminal controller and curriculum values against its final history record

Design principles:
  - Curriculum progress is the stage budget fraction committed before an update group
  - Shorter horizons retain positive probability after later milestones
  - Completed groups retain transition accounting; their time is only on the stage budget's optimisation clock

This module does NOT:
  - Load teacher files, run model forwards, or mutate optimizers
  - Own the stage budget or its validation points; ``learning.training.budget`` does
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, overload

import torch

TrainingStage = Literal["stage_a_teacher_forcing", "stage_b_self_fed"]
ComparisonArm = Literal["A0", "B", "A1"]
DEFAULT_ROLLOUT_LENGTHS: Final = (2, 4, 8, 16, 32)
DEFAULT_MILESTONE_FRACTIONS: Final = (0.0, 0.2, 0.4, 0.6, 0.8)
# Immutable controller semantics shared by checkpoint admission and the stage digest.
_CONTROLLER_IDENTITY_FIELDS: Final = ("arm", "stage", "config_digest", "teacher_handoff", "matched_receipt_sha256")
_SHA256_LENGTH: Final = 64
# Every other terminal controller and curriculum value is kept in each completed-epoch history record under its adapter
# telemetry name, which is the value's own name unless renamed here.
_TERMINAL_VALUES_NOT_IN_HISTORY: Final = frozenset({"arm", "stage", "matched_receipt_sha256"})
_TERMINAL_HISTORY_RENAMES: Final = {
    "active_stage": "curriculum_active_stage",
    "max_horizon": "curriculum_max_horizon",
    "draw_index": "curriculum_draw_index",
}


def _sha256(value: Any, *, label: str) -> str:
    """Require one lowercase SHA-256 digest."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        msg = f"{label} must be a lowercase 64-character SHA-256 digest."
        raise ValueError(msg)
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """Require one positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{label} must be a positive integer."
        raise ValueError(msg)
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    """Require one nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{label} must be a non-negative integer."
        raise ValueError(msg)
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
        msg = f"{label} must be a finite non-negative number."
        raise ValueError(msg)
    return float(value)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a stable digest for one strict JSON-compatible identity payload."""
    encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_controller_state(state: Mapping[str, Any]) -> None:
    """Apply the controller's arm, stage, receipt and work-counter rules that need no handoff or digest identity."""
    if state["arm"] not in {"A0", "B", "A1"} or state["stage"] not in {"stage_a_teacher_forcing", "stage_b_self_fed"}:
        msg = "Transient comparison arm or stage is unsupported."
        raise ValueError(msg)
    if state["stage"] == "stage_b_self_fed" and state["arm"] != "B":
        msg = "Self-fed Stage B requires arm B."
        raise ValueError(msg)
    if state["stage"] == "stage_a_teacher_forcing" and state["arm"] == "B":
        msg = "Arm B requires self-fed Stage B."
        raise ValueError(msg)
    _finite_nonnegative(state["validation_seconds"], label="validation_seconds")
    for label in ("post_handoff_optimizer_steps", "processed_target_transitions", "forward_transitions"):
        _nonnegative_int(state[label], label=label)
    if state["arm"] == "A1":
        _sha256(state["matched_receipt_sha256"], label="matched_receipt_sha256")
    elif state["matched_receipt_sha256"] is not None:
        msg = "Only A1 trains for a B receipt's seconds."
        raise ValueError(msg)


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
                msg = f"{label} must be a non-empty string."
                raise ValueError(msg)
        for label in ("source_checkpoint_sha256", "source_scaling_sha256", "task_contract_sha256", "tensorizer_sha256"):
            _sha256(getattr(self, label), label=label)

    @property
    def digest(self) -> str:
        """Return the immutable teacher identity digest."""
        return _canonical_digest(self.as_dict())

    def as_dict(self) -> dict[str, str | int]:
        """Return strict serializable handoff evidence."""
        return {
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
            "source_run_name",
            "source_checkpoint_sha256",
            "source_scaling_sha256",
            "task_contract_sha256",
            "tensorizer_sha256",
            "model_kind",
            "input_profile",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            msg = "Teacher handoff evidence does not match the current strict schema."
            raise ValueError(msg)
        return cls(**{key: value[key] for key in expected})


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
            msg = "Curriculum lengths must be the one-step horizon or increasing members of 2, 4, 8, 16, 32, optionally 64."
            raise ValueError(msg)
        elif lengths[: len(DEFAULT_ROLLOUT_LENGTHS)] != DEFAULT_ROLLOUT_LENGTHS and lengths != DEFAULT_ROLLOUT_LENGTHS[: len(lengths)]:
            msg = "Curriculum lengths must retain the canonical ordered prefix."
            raise ValueError(msg)
        if len(self.milestone_fractions) != len(lengths) or self.milestone_fractions[0] != 0.0:
            msg = "Curriculum milestones must align with lengths and begin at zero."
            raise ValueError(msg)
        previous = -1.0
        for fraction in self.milestone_fractions:
            if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 1.0 or float(fraction) <= previous:
                msg = "Curriculum milestones must be strictly increasing fractions in [0, 1)."
                raise ValueError(msg)
            previous = float(fraction)

    def active_stage(self, progress: float) -> int:
        """Return the largest admitted stage at compute progress."""
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            msg = "Curriculum progress must be finite in [0, 1]."
            raise ValueError(msg)
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
        return {"lengths": list(self.lengths), "milestone_fractions": list(self.milestone_fractions)}

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

    def draw(self, *, progress: float, available_lengths: Sequence[int]) -> tuple[tuple[int, int], ...]:
        """
        Draw one horizon and one valid origin for every window, in the given order.

        Each horizon is drawn uniformly from the lengths eligible at ``progress`` and
        capped at its window's length, so a window shorter than the drawn horizon is
        rolled out over its whole length. Returns ``(horizon, origin)`` per window.
        """
        eligible = self.curriculum.eligible_lengths(progress)
        draws: list[tuple[int, int]] = []
        for length in available_lengths:
            available = _positive_int(length, label="available_length")
            horizon = min(eligible[int(torch.randint(len(eligible), (1,), generator=self.generator).item())], available)
            draws.append((horizon, self._sample_origin(available_length=available, horizon=horizon)))
            self.draw_index += 1
        self.progress = float(progress)
        self.active_stage = self.curriculum.active_stage(progress)
        self.max_horizon = self.curriculum.lengths[self.active_stage]
        self.completed_milestones = tuple(
            index for index, milestone in enumerate(self.curriculum.milestone_fractions) if progress + 1.0e-12 >= milestone
        )
        return tuple(draws)

    def _sample_origin(self, *, available_length: int, horizon: int) -> int:
        """
        Sample one valid early, middle or late origin without padding.

        The stratum rotates with the persisted draw index, so consecutive windows
        cover all three strata whatever the group size.
        """
        maximum = available_length - horizon
        if maximum == 0:
            return 0
        edges = ((0, maximum // 3), (maximum // 3 + 1, (2 * maximum) // 3), ((2 * maximum) // 3 + 1, maximum))
        low, high = edges[self.draw_index % len(edges)]
        if low > high:
            low, high = 0, maximum
        return int(torch.randint(low, high + 1, (1,), generator=self.generator).item())

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
            msg = "Saved curriculum state does not match the configured curriculum."
            raise ValueError(msg)
        state = torch.tensor(value["generator_state"], dtype=torch.uint8)
        self.generator.set_state(state)
        self.draw_index = _nonnegative_int(value["draw_index"], label="draw_index")
        self.active_stage = _nonnegative_int(value["active_stage"], label="active_stage")
        self.max_horizon = _positive_int(value["max_horizon"], label="max_horizon")
        self.progress = float(_finite_nonnegative(value["progress"], label="progress"))
        if self.progress > 1.0:
            msg = "progress must not exceed one."
            raise ValueError(msg)
        milestones = value["completed_milestones"]
        if not isinstance(milestones, list) or any(not isinstance(index, int) for index in milestones):
            msg = "completed_milestones must be an integer list."
            raise ValueError(msg)
        self.completed_milestones = tuple(milestones)


@dataclass(slots=True)
class MatchedComputeController:
    """Account a transient stage's completed optimizer work, its teacher provenance and its in-budget selection."""

    arm: ComparisonArm
    stage: TrainingStage
    config_digest: str
    teacher_handoff: TeacherHandoffIdentity | None = None
    post_handoff_optimizer_steps: int = 0
    microbatches: int = 0
    processed_target_transitions: int = 0
    forward_transitions: int = 0
    validation_seconds: float = 0.0
    peak_cuda_memory_bytes: int | None = None
    best_within_budget_metric: float | None = None
    best_within_budget_epoch: int | None = None
    matched_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate immutable arm, identity and work-counter semantics."""
        _validate_controller_state({name: getattr(self, name) for name in self.__dataclass_fields__})
        _sha256(self.config_digest, label="config_digest")
        if self.arm == "A0" and self.teacher_handoff is not None:
            msg = "A0 must remain unmatched and cannot consume a teacher handoff."
            raise ValueError(msg)
        if self.arm in {"B", "A1"} and not isinstance(self.teacher_handoff, TeacherHandoffIdentity):
            msg = "B and A1 arms require immutable teacher-handoff identity."
            raise ValueError(msg)
        _nonnegative_int(self.microbatches, label="microbatches")

    def record_completed_work(
        self,
        *,
        microbatches: int,
        processed_target_transitions: int,
        forward_transitions: int,
        peak_cuda_memory_bytes: int | None = None,
    ) -> None:
        """Account one completed optimizer group's logical transition evidence; its time is on the stage budget's clock."""
        _nonnegative_int(microbatches, label="microbatches")
        _nonnegative_int(processed_target_transitions, label="processed_target_transitions")
        _nonnegative_int(forward_transitions, label="forward_transitions")
        self.post_handoff_optimizer_steps += 1
        self.microbatches += microbatches
        self.processed_target_transitions += processed_target_transitions
        self.forward_transitions += forward_transitions
        if peak_cuda_memory_bytes is not None:
            self.peak_cuda_memory_bytes = max(
                self.peak_cuda_memory_bytes or 0, _nonnegative_int(peak_cuda_memory_bytes, label="peak_cuda_memory_bytes")
            )

    def record_validation_work(self, seconds: float) -> None:
        """Accumulate completed validation wall time outside the optimisation clock."""
        self.validation_seconds += _finite_nonnegative(seconds, label="validation_seconds")

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        """Record one finite selection metric; the loop validates only within the stage budget."""
        if not math.isfinite(metric):
            return
        epoch = _nonnegative_int(epoch_index, label="epoch_index")
        if self.best_within_budget_metric is None or metric < self.best_within_budget_metric:
            self.best_within_budget_metric = float(metric)
            self.best_within_budget_epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        """Return strict checkpointable controller evidence."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["teacher_handoff"] = None if self.teacher_handoff is None else self.teacher_handoff.as_dict()
        return values

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        """Restore only matching immutable controller semantics."""
        if not isinstance(value, Mapping):
            msg = "Saved transient budget state must be a mapping."
            raise TypeError(msg)
        if set(value) != set(self.state_dict()):
            msg = "Saved transient budget state does not match the strict schema."
            raise ValueError(msg)
        candidate = dict(value)
        handoff = candidate["teacher_handoff"]
        candidate["teacher_handoff"] = None if handoff is None else TeacherHandoffIdentity.from_mapping(handoff)
        restored = MatchedComputeController(**candidate)
        if any(getattr(restored, name) != getattr(self, name) for name in _CONTROLLER_IDENTITY_FIELDS):
            msg = "Saved transient budget state conflicts with configured semantic identity."
            raise ValueError(msg)
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(restored, name))


@dataclass(frozen=True, slots=True)
class TransientTrainingSpec:
    """
    Bind one transient stage to a model, controller and, in Stage B, its curriculum.

    Only Stage B draws self-fed rollout horizons, so only Stage B has a curriculum and
    its seed; teacher-forced Stage A has neither. ``trajectory_window`` is the window
    length the RNO cuts from whole trajectories: its hidden state carries across
    consecutive windows while gradients stop at each window boundary. Stateless models
    leave it ``None``.
    """

    stage: TrainingStage
    model_kind: str
    controller: MatchedComputeController
    curriculum: RolloutCurriculum | None
    curriculum_seed: int | None
    trajectory_window: int | None = None

    def __post_init__(self) -> None:
        """Reject incompatible stage, controller, curriculum and horizon selections."""
        if self.model_kind not in {"fno", "uno", "rno"}:
            msg = "Transient model_kind must be fno, uno, or rno."
            raise ValueError(msg)
        self_fed = self.stage == "stage_b_self_fed"
        if self_fed != (self.curriculum is not None) or self_fed != (self.curriculum_seed is not None):
            msg = "Exactly Stage B draws rollout horizons and has a curriculum and its seed; Stage A has neither."
            raise ValueError(msg)
        if self.curriculum_seed is not None:
            _nonnegative_int(self.curriculum_seed, label="curriculum_seed")
        if (self.model_kind == "rno") != (self.trajectory_window is not None):
            msg = "Exactly the recurrent RNO trains on trajectory windows."
            raise ValueError(msg)
        if self.trajectory_window is not None:
            _positive_int(self.trajectory_window, label="trajectory_window")
            if self.curriculum is not None and self.curriculum.lengths[-1] > self.trajectory_window:
                msg = "Curriculum horizons must fit the trajectory window."
                raise ValueError(msg)
        if self.stage != self.controller.stage:
            msg = "Training specification stage must match its compute controller."
            raise ValueError(msg)
        if self.stage == "stage_a_teacher_forcing" and self.controller.arm not in {"A0", "A1"}:
            msg = "Stage A supports only the A0 or A1 arms."
            raise ValueError(msg)
        if self.stage == "stage_b_self_fed" and self.controller.arm != "B":
            msg = "Stage B requires the B comparison arm."
            raise ValueError(msg)

    @property
    def digest(self) -> str:
        """Return immutable task-stage semantic identity excluding mutable work."""
        return _canonical_digest(
            {
                "stage": self.stage,
                "model_kind": self.model_kind,
                "curriculum": None if self.curriculum is None else self.curriculum.as_dict(),
                "curriculum_seed": self.curriculum_seed,
                "trajectory_window": self.trajectory_window,
                "controller_identity": {
                    key: getattr(self.controller, key)
                    if key != "teacher_handoff"
                    else (None if self.controller.teacher_handoff is None else self.controller.teacher_handoff.as_dict())
                    for key in _CONTROLLER_IDENTITY_FIELDS
                },
            }
        )


def validate_terminal_evidence(
    controller: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    *,
    final_record: Mapping[str, float],
) -> None:
    """
    Admit the terminal controller and curriculum values that a completed stage records.

    Parameters
    ----------
    controller : Mapping[str, Any]
        Terminal controller values without the handoff and digest identity, which the stage checkpoint binds.
    curriculum : Mapping[str, Any]
        Terminal ``active_stage``, ``max_horizon`` and ``draw_index`` of a Stage B; empty for Stage A, which has no curriculum.
    final_record : Mapping[str, float]
        Last admitted completed-epoch history record; empty when the history holds none.

    Raises
    ------
    ValueError
        If a value breaks the controller's state rules or disagrees with the final record.

    Notes
    -----
    The adapter telemetry records only non-null values, so a recorded value absent from the
    final record must be null. The stage budget's completion is the checkpoint's own evidence.

    """
    _validate_controller_state(controller)
    for key, value in {**controller, **curriculum}.items():
        name = _TERMINAL_HISTORY_RENAMES.get(key, key)
        if key not in _TERMINAL_VALUES_NOT_IN_HISTORY and final_record.get(f"transient/{name}") != value:
            msg = f"Terminal {key} disagrees with the final completed-epoch history record."
            raise ValueError(msg)
