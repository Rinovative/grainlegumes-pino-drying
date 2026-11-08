"""Protect transient continuous curriculum and stage-budget contracts.

Responsibilities:
  - Check horizon draws, exact resumed RNG, clock semantics, and budget completion
  - Check that origin strata reach the late part of a window for every batch size

Design principles:
  - Committed optimizer groups define budget crossing and terminal evaluation admission

This module does NOT:
  - Infer compute budgets from another clock or sample unavailable short windows
"""

from __future__ import annotations

import hashlib
from typing import Any

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
    assert curriculum.as_dict()["schema_version"] == 1
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


def _work(**overrides: Any) -> dict[str, Any]:
    """Return one completed optimizer group with test-owned defaults."""
    work: dict[str, Any] = {
        "optimizer_device_seconds": None,
        "microbatches": 1,
        "processed_target_transitions": 2,
        "forward_transitions": 2,
        "wall_seconds": 0.1,
        "epoch_index": 0,
        "microbatch_index": 0,
    }
    work.update(overrides)
    return work


@pytest.mark.parametrize("batch_size", [1, 2])
def test_origin_strata_rotate_so_small_batches_reach_every_third_of_the_window(batch_size: int) -> None:
    """
    Draw origins for batches smaller than the three temporal strata.

    Single-trajectory batches of a rare length and the maintained batch size 2
    must still start rollouts in the early, middle and late third of a window,
    in equal numbers over three consecutive draws.
    """
    state = RolloutCurriculumState.create(RolloutCurriculum(), seed=5)
    available_length = 32
    horizon = 2
    maximum = available_length - horizon
    strata = ((0, maximum // 3), (maximum // 3 + 1, (2 * maximum) // 3), ((2 * maximum) // 3 + 1, maximum))
    counts = [0, 0, 0]
    for _draw in range(3):
        # The first curriculum stage admits only the shortest horizon.
        drawn_horizon, origins = state.select(progress=0.0, available_length=available_length, batch_size=batch_size)
        assert drawn_horizon == horizon
        for origin in origins.tolist():
            counts[next(index for index, (low, high) in enumerate(strata) if low <= origin <= high)] += 1

    assert counts == [batch_size] * 3


def test_controller_accounts_device_seconds_only_on_the_cuda_clock() -> None:
    """Accumulate CUDA device time and reject device time on the CPU step clock."""
    cuda = MatchedComputeController(
        arm="A0",
        stage="stage_a_teacher_forcing",
        clock_kind="cuda_device_seconds",
        config_digest=_digest("cuda"),
        planned_stage_epochs=2,
    )
    cuda.record_completed_work(**_work(optimizer_device_seconds=4.0, microbatches=2))
    assert cuda.post_handoff_optimizer_device_seconds == pytest.approx(4.0)
    cpu = MatchedComputeController(
        arm="A0",
        stage="stage_a_teacher_forcing",
        clock_kind="optimizer_steps",
        config_digest=_digest("cpu"),
        planned_stage_epochs=2,
    )
    with pytest.raises(ValueError, match="must not report optimizer device seconds"):
        cpu.record_completed_work(**_work(optimizer_device_seconds=0.1))


def test_controller_state_rejects_semantic_drift_and_superseded_schemas() -> None:
    """Require immutable controller identity and the current schema during continuation."""
    controller = MatchedComputeController(
        arm="A1",
        stage="stage_a_teacher_forcing",
        clock_kind="train_seconds",
        config_digest=_digest("state"),
        budget_control="train_seconds",
        teacher_handoff=_handoff(),
        planned_train_seconds=5.0,
        matched_receipt_sha256=_digest("receipt"),
    )
    saved = controller.state_dict()
    drifted = dict(saved, config_digest=_digest("other"))
    with pytest.raises(ValueError, match="conflicts"):
        controller.load_state_dict(drifted)
    for version in (2, True, 1.0):  # Superseded, JSON true and 1.0 must not pass as version 1.
        with pytest.raises(ValueError, match="strict schema"):
            controller.load_state_dict(dict(saved, schema_version=version))


def test_controller_rejects_mismatched_budget_and_clock() -> None:
    """Pair the measured train_seconds budget exclusively with the train_seconds clock."""
    with pytest.raises(ValueError, match="train_seconds clock"):
        MatchedComputeController(
            arm="B",
            stage="stage_b_self_fed",
            clock_kind="cuda_device_seconds",
            config_digest=_digest("mismatch"),
            budget_control="train_seconds",
            teacher_handoff=_handoff(),
            planned_train_seconds=5.0,
        )


def test_active_horizon_shortage_fails_without_sampling_shorter_windows() -> None:
    """Reject insufficient source windows after a later compute milestone."""
    state = RolloutCurriculumState.create(RolloutCurriculum(), seed=3)
    with pytest.raises(ValueError, match="active curriculum maximum"):
        state.select(progress=0.85, available_length=16, batch_size=2)
    assert state.draw_index == 0


def test_train_seconds_crossing_group_is_selectable_and_no_later_work_is_admitted() -> None:
    """Treat the first crossing group as the discrete boundary and make completion sticky."""
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="train_seconds",
        config_digest=_digest("sticky-boundary"),
        budget_control="train_seconds",
        teacher_handoff=_handoff(),
        planned_train_seconds=1.0,
    )
    controller.record_completed_work(**_work(train_seconds=0.6, batch_cursor=1))
    assert controller.budget_complete is False
    controller.record_completed_work(**_work(train_seconds=0.6, batch_cursor=2, microbatch_index=1))
    controller.record_within_budget_evaluation(0.25, epoch_index=0)

    assert controller.budget_complete is True
    assert controller.crossing_microbatch == 1
    assert controller.train_seconds == pytest.approx(1.2)
    assert controller.best_within_budget_metric == pytest.approx(0.25)
    assert controller.best_within_budget_epoch == 0
    with pytest.raises(RuntimeError, match="cannot continue"):
        controller.record_completed_work(**_work(train_seconds=0.1, batch_cursor=3))


def test_budget_points_fall_due_once_per_crossed_fraction_and_validate_their_owner() -> None:
    """Mark k/N points of the measured budget; several crossed points share one evaluation."""
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="train_seconds",
        config_digest=_digest("points"),
        budget_control="train_seconds",
        teacher_handoff=_handoff(),
        planned_train_seconds=10.0,
        evaluation_points=4,
    )
    controller.record_completed_work(**_work(train_seconds=2.0, batch_cursor=1))
    assert controller.budget_evaluation_due is False
    controller.record_completed_work(**_work(train_seconds=3.0, batch_cursor=2, microbatch_index=1))
    assert controller.budget_evaluation_due is True
    controller.record_budget_evaluation()
    assert (controller.evaluated_points, controller.budget_evaluation_due) == (2, False)
    controller.record_completed_work(**_work(train_seconds=5.5, batch_cursor=3, microbatch_index=2))
    assert controller.budget_complete is True
    assert controller.budget_evaluation_due is True
    controller.record_budget_evaluation()
    assert controller.evaluated_points == 4
    restored = MatchedComputeController(**{name: getattr(controller, name) for name in controller.__dataclass_fields__})
    restored.load_state_dict(controller.state_dict())
    assert restored.evaluated_points == 4

    with pytest.raises(ValueError, match="only to measured train_seconds"):
        MatchedComputeController(
            arm="A0",
            stage="stage_a_teacher_forcing",
            clock_kind="optimizer_steps",
            config_digest=_digest("epochs"),
            planned_stage_epochs=2,
            evaluation_points=4,
        )
    with pytest.raises(ValueError, match="exceeds evaluation_points"):
        MatchedComputeController(**{**{name: getattr(controller, name) for name in controller.__dataclass_fields__}, "evaluated_points": 5})


def test_stage_epoch_budget_completes_only_after_the_terminal_evaluation() -> None:
    """Run every allocated epoch without a competing matched-compute stop."""
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="optimizer_steps",
        config_digest=_digest("epoch-budget"),
        budget_control="stage_epochs",
        planned_stage_epochs=2,
        teacher_handoff=_handoff(),
    )
    work = {
        "optimizer_device_seconds": None,
        "microbatches": 1,
        "processed_target_transitions": 2,
        "forward_transitions": 2,
        "wall_seconds": 0.1,
        "microbatch_index": 0,
    }

    controller.begin_epoch(epoch_index=0, total_epochs=2)
    controller.record_completed_work(**work, epoch_index=0)
    controller.record_within_budget_evaluation(0.5, epoch_index=0)
    assert controller.progress == pytest.approx(0.5)
    assert controller.budget_complete is False

    controller.begin_epoch(epoch_index=1, total_epochs=2)
    controller.record_completed_work(**work, epoch_index=1)
    assert controller.budget_complete is False
    controller.record_within_budget_evaluation(0.25, epoch_index=1)

    assert controller.progress == pytest.approx(1.0)
    assert controller.completed_stage_epochs == 2
    assert controller.budget_complete is True
    assert controller.crossing_epoch == 1
    assert controller.crossing_microbatch is None
