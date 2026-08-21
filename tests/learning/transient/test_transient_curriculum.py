# ruff: noqa: S101, SLF001, PLR2004
"""Protect transient continuous curriculum and matched-compute contracts."""

from __future__ import annotations

import hashlib

import pytest
import torch

from src.learning.transient.learning_transient_curriculum import (
    MatchedComputeController,
    RolloutCurriculum,
    RolloutCurriculumState,
    TeacherHandoffIdentity,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _handoff() -> TeacherHandoffIdentity:
    return TeacherHandoffIdentity(
        source_run_name="synthetic-a0",
        source_checkpoint_sha256=_digest("checkpoint"),
        source_scaling_sha256=_digest("scaling"),
        task_contract_sha256=_digest("task"),
        tensorizer_sha256=_digest("tensorizer"),
        model_kind="fno",
        input_profile="canonical_physics_complete_v1",
    )


def test_curriculum_retains_shorter_horizons_and_resumes_exact_draws() -> None:
    """Use compute progress and private RNG state for reproducible valid windows."""
    curriculum = RolloutCurriculum()
    state = RolloutCurriculumState.create(curriculum, seed=17)
    first_horizon, first_origins = state.select(progress=0.85, available_length=32, batch_size=9)
    assert first_horizon in curriculum.eligible_lengths(0.85)
    assert {int(value) for value in first_origins.tolist()} <= set(range(31))
    varied_origins = state._sample_origins(available_length=32, horizon=2, batch_size=9)
    assert len(set(varied_origins.tolist())) > 1
    saved = state.state_dict()
    expected = state.select(progress=0.85, available_length=32, batch_size=6)
    restored = RolloutCurriculumState.create(curriculum, seed=999)
    restored.load_state_dict(saved)
    observed = restored.select(progress=0.85, available_length=32, batch_size=6)
    assert observed[0] == expected[0]
    assert torch.equal(observed[1], expected[1])
    assert 2 in curriculum.eligible_lengths(0.85)


def test_controller_uses_seconds_or_steps_without_cross_clock_labels() -> None:
    """Derive exact remaining quantities from the admitted measurement clock."""
    cuda = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="cuda_device_seconds",
        config_digest=_digest("cuda"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_seconds=10.0,
        rollout_reference_compute_seconds=12.0,
    )
    cuda.record_completed_work(
        successful=True,
        optimizer_device_seconds=4.0,
        microbatches=2,
        processed_target_transitions=16,
        forward_transitions=16,
        wall_seconds=5.0,
        epoch_index=0,
        microbatch_index=0,
    )
    assert cuda.remaining_to_planned_teacher_forcing_budget_seconds == pytest.approx(6.0)
    assert cuda.remaining_teacher_forcing_compute_to_match_rollout_seconds is None
    cpu = MatchedComputeController(
        arm="A+",
        stage="stage_a_teacher_forcing",
        clock_kind="optimizer_steps",
        config_digest=_digest("cpu"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_steps=3,
    )
    cpu.record_completed_work(
        successful=True,
        optimizer_device_seconds=None,
        microbatches=1,
        processed_target_transitions=2,
        forward_transitions=2,
        wall_seconds=0.1,
        epoch_index=0,
        microbatch_index=0,
    )
    assert cpu.remaining_steps == 2
    assert cpu.remaining_to_planned_teacher_forcing_budget_seconds is None
    with pytest.raises(ValueError, match="must not report optimizer device seconds"):
        cpu.record_completed_work(
            successful=True,
            optimizer_device_seconds=0.1,
            microbatches=1,
            processed_target_transitions=2,
            forward_transitions=2,
            wall_seconds=0.1,
            epoch_index=0,
            microbatch_index=1,
        )


def test_controller_state_rejects_semantic_drift() -> None:
    """Require immutable controller identity during exact continuation."""
    controller = MatchedComputeController(
        arm="A+",
        stage="stage_a_teacher_forcing",
        clock_kind="optimizer_steps",
        config_digest=_digest("state"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_steps=2,
    )
    saved = controller.state_dict()
    saved["config_digest"] = _digest("other")
    with pytest.raises(ValueError, match="conflicts"):
        controller.load_state_dict(saved)


def test_active_horizon_shortage_fails_without_sampling_shorter_windows() -> None:
    """Reject insufficient source windows after a later compute milestone."""
    state = RolloutCurriculumState.create(RolloutCurriculum(), seed=3)
    with pytest.raises(ValueError, match="active curriculum maximum"):
        state.select(progress=0.85, available_length=16, batch_size=2)
    assert state.draw_index == 0


def test_overflow_keeps_completed_compute_evidence_but_not_step_counters() -> None:
    """Preserve measured CUDA work for AMP overflow without claiming an update."""
    controller = MatchedComputeController(
        arm="A+",
        stage="stage_a_teacher_forcing",
        clock_kind="cuda_device_seconds",
        config_digest=_digest("overflow"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_seconds=10.0,
    )
    controller.record_completed_work(
        successful=False,
        optimizer_device_seconds=2.5,
        microbatches=2,
        processed_target_transitions=8,
        forward_transitions=8,
        wall_seconds=3.0,
        epoch_index=0,
        microbatch_index=0,
        peak_cuda_memory_bytes=32,
    )
    assert controller.post_handoff_optimizer_device_seconds == pytest.approx(2.5)
    assert controller.microbatches == 2
    assert controller.processed_target_transitions == 8
    assert controller.forward_transitions == 8
    assert controller.successful_optimizer_steps == 0
    assert controller.teacher_forcing_optimizer_steps == 0
    assert controller.post_handoff_optimizer_steps == 1


def test_successful_stage_a_and_cpu_remaining_steps_are_explicit() -> None:
    """Advance Stage-A successful counters and expose both CPU remaining formulas."""
    controller = MatchedComputeController(
        arm="A+",
        stage="stage_a_teacher_forcing",
        clock_kind="optimizer_steps",
        config_digest=_digest("cpu-remaining"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_steps=5,
        rollout_reference_compute_steps=7,
    )
    controller.record_completed_work(
        successful=True,
        optimizer_device_seconds=None,
        microbatches=1,
        processed_target_transitions=2,
        forward_transitions=2,
        wall_seconds=0.1,
        epoch_index=0,
        microbatch_index=0,
    )
    assert controller.post_handoff_optimizer_steps == 1
    assert controller.teacher_forcing_optimizer_steps == 1
    assert controller.remaining_to_planned_teacher_forcing_budget_steps == 4
    assert controller.remaining_teacher_forcing_compute_to_match_rollout_steps == 6
    assert controller.remaining_to_planned_teacher_forcing_budget_seconds is None
    assert controller.remaining_teacher_forcing_compute_to_match_rollout_seconds is None


def test_budget_crossing_group_is_selectable_and_no_later_work_is_admitted() -> None:
    """Treat the first crossing group as the discrete boundary and make completion sticky."""
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="optimizer_steps",
        config_digest=_digest("sticky-boundary"),
        teacher_handoff=_handoff(),
        planned_teacher_forcing_budget_steps=1,
    )
    work = {
        "successful": True,
        "optimizer_device_seconds": None,
        "microbatches": 1,
        "processed_target_transitions": 2,
        "forward_transitions": 2,
        "wall_seconds": 0.1,
        "epoch_index": 0,
        "microbatch_index": 0,
    }

    controller.record_completed_work(**work)
    controller.record_within_budget_evaluation(0.25, epoch_index=0)

    assert controller.budget_complete is True
    assert controller.successful_optimizer_steps == 1
    assert controller.best_within_budget_metric == pytest.approx(0.25)
    assert controller.best_within_budget_epoch == 0
    with pytest.raises(RuntimeError, match="cannot continue"):
        controller.record_completed_work(**work)
